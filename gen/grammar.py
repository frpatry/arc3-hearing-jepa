"""Grammaire de règles : tire au hasard la SPÉCIFICATION d'un jeu (façon XLand / ProcGen), exécutée ensuite par le
moteur générique `families_rules.RuleGame`.

    spec = sample_rules(seed, attempt=0)      # dict sérialisable JSON
    signature(spec)                           # (contrôle, ensemble d'interactions, condition de victoire)
    describe(spec)                            # résumé lisible d'une ligne

Composantes d'une spec :
  - types d'objets (2 à 9) : rôle (étiquette), solidité (pass / block / push), contrôlé (1 / 2 = jumeau couplé),
    mouvement propre (patrol / chase / gravity / conveyor), porte (gated : ouverte quand la variable gate vaut 0/1),
    grille (grate : bloque l'avatar « grand »), glissant (slip), sprite (style détaillé ou motif aléatoire), placement,
    nombre par niveau (c0 + g·i, plafonné), niveau d'apparition (intro) et de retrait (until) ;
  - contrôle : scheme ∈ avatar / multi (tous les avatars bougent ensemble) / coupled (jumeau : same, miroir h/v,
    inverse, rot90) / select (clic = sélection, flèches = déplacer la pièce) / cursor (flèches = curseur, A5 = activer)
    / click (clic seul) ; slide (glisse jusqu'au blocage) ; rôle d'ACTION5 (wait, interact, grow, gflip, gate, commit,
    activate, ou rien) ; clics à distance possibles même en avatar ;
  - interactions : règles « quand A <déclencheur> B -> effets », déclencheur ∈ enter / bump / push / adjacent / step /
    click / interact ; effets ∈ remove_B, remove_A, B_to C, A_to C, teleport_A, counter_inc, spend (clé -> porte),
    gate, B_step, B_slide, gflip, toggle_all C D, grow, spawn C, lose, flip_nbrs C D, flip_mirror C D ;
  - dynamique globale : gravité (direction tirée, inversable), tapis roulants, PNJ patrouilleurs / poursuivants ;
  - victoire : reach (un X sur un Y), all_on (tous les X sur des Y), none_left (plus aucun X), counter >= k,
    pattern (motif = panneau-cible affiché), same (tous de la même couleur) ; option « et rejoindre la sortie » ;
    option commit (valider avec ACTION5, mauvaise réponse = GAME_OVER) ;
  - défaite : règles « lose », barre d'énergie (HUD du noyau), vies (look officiel), validation ratée ;
  - légende : panneau qui révèle (une partie d') une règle — affiché ou non.
Les niveaux (4 à 8) AJOUTENT des modules (types + règles) à partir d'un niveau d'apparition, comme les jeux officiels.
"""
import json
import zlib

import numpy as np

PASS, BLOCK, PUSH = 0, 1, 2
SOLID_NAMES = ("pass", "block", "push")
TRIGGERS = ("enter", "bump", "push", "adjacent", "step", "click", "interact")
EFFECTS = ("remove_B", "remove_A", "B_to", "A_to", "teleport_A", "counter_inc", "spend", "gate", "B_step", "B_slide",
           "gflip", "grot", "toggle_all", "grow", "spawn", "lose", "flip_nbrs", "flip_mirror")
SCHEMES = ("avatar", "multi", "coupled", "select", "cursor", "click")
SCHEME_P = (0.36, 0.08, 0.12, 0.12, 0.12, 0.20)
COUPLINGS = ("same", "mirror_h", "mirror_v", "invert", "rot90")
WIN_KINDS = ("reach", "all_on", "none_left", "counter", "pattern", "same")
DETAILED = ("box_dot", "eye", "ring_dot", "diamond", "frame", "small", "small_dot", "ring", "full", "cross", "dot",
            "corners")
AVATAR_FLAVORS = ("hazard", "teleport", "gate", "key", "patrol", "chase", "conveyor", "grate", "paintbox", "trail",
                  "toggleblocks", "rocks", "wake", "chest", "crack", "ice", "remote", "kick")
TOGGLE_FLAVORS = ("extra_color", "nbr_switch", "mirror")
GRAVITY_FLAVORS = ("g_hazard", "g_teleport", "g_rock2", "g_conveyor")
MAX_TYPES = 9


class _B:
    """constructeur de spec."""

    def __init__(self, r):
        self.r = r
        self.types, self.rules, self.mods = [], [], []
        self.a5 = None
        self.uses_click = False
        self.gravity = None
        self.counter_used = False
        self.grow_init = 0

    def sprite(self, role):
        r = self.r
        if role in ("tile",):
            return str(r.choice(["full", "full", "small", "box_dot"]))
        if r.random() < 0.35:
            return f"custom:{int(r.integers(1 << 30))}"
        return str(r.choice(DETAILED))

    def T(self, role, solid=PASS, ctrl=0, motion=None, gated=None, grate=False, slip=False, place="any",
          count=(1, 0.0, 1), intro=0, until=99):
        if len(self.types) >= MAX_TYPES:
            raise _Full()
        t = dict(id=len(self.types), role=role, solid=int(solid), ctrl=int(ctrl), motion=motion, gated=gated,
                 grate=bool(grate), slip=bool(slip), place=place, count=list(count) if not isinstance(count, str)
                 else count, intro=int(intro), until=int(until), sprite=self.sprite(role))
        self.types.append(t)
        return t["id"]

    def R(self, trig, a, b, effects, intro=0, until=99, tag="", hint=False):
        self.rules.append(dict(trigger=trig, a=a, b=b, effects=[list(e) for e in effects], intro=int(intro),
                               until=int(until), tag=tag, hint=bool(hint)))


class _Full(Exception):
    pass


# ============================================================================ archétypes de base (contrôle avatar)
def _players(B, scheme):
    r = B.r
    if scheme == "avatar":
        return [B.T("player", BLOCK, ctrl=1, place="start")]
    if scheme == "multi":
        return [B.T("player", BLOCK, ctrl=1, place="start", count=(2, 0.3, 3))]
    if scheme == "coupled":
        return [B.T("player", BLOCK, ctrl=1, place="start"), B.T("twin", BLOCK, ctrl=2, place="start")]
    if scheme == "select":
        return [B.T("piece", BLOCK, ctrl=1, place="start", count=(2, 0.25 * r.random(), 3))]
    return []


def _avatar_core(B, scheme, P):
    """-> (archétype, win)."""
    r = B.r
    many = scheme in ("multi", "coupled", "select")
    kinds = ["reach", "push", "collect", "paint", "destroy"]
    w = [0.26, 0.2, 0.24, 0.15, 0.15] if not many else [0.45, 0.2, 0.2, 0.1, 0.05]
    arche = str(r.choice(kinds, p=w))
    win = dict(and_reach=None, commit=False)
    if arche == "reach":
        if many:
            E = B.T("exit", PASS, place="far", count="match_winX")
            win.update(kind="all_on", X=P, Y=E)
        else:
            E = B.T("exit", PASS, place="far")
            win.update(kind="reach", X=P, Y=E)
    elif arche == "push":
        kick = r.random() < 0.25
        X = B.T("box", BLOCK if kick else PUSH, place="inner", count=(1, 0.25, 2))
        D = B.T("pad", PASS, place="any", count=f"match:{X}")
        if kick:
            for p in P:
                B.R("bump", p, X, [["B_slide"]], tag="kick")
        win.update(kind="all_on", X=[X], Y=D)
        if r.random() < 0.3:                                         # la caisse se fige sur sa plaque
            Xl = B.T("box_set", BLOCK, count=(0, 0, 0))
            B.R("enter", X, D, [["A_to", Xl]], tag="lock", hint=True)
            win["X"] = [X, Xl]
    elif arche == "collect":
        solid = BLOCK if r.random() < 0.3 else PASS
        I = B.T("item", solid, place="any", count=(2, 0.5, 5))
        trig = "enter" if solid == PASS else "bump"
        u = r.random()
        if u < 0.15 and B.a5 is None:
            trig, B.a5 = "interact", "interact"
        elif u < 0.25:
            trig = "adjacent"
        eff = int(r.integers(3))
        for p in P:
            if eff == 0:
                B.R(trig, p, I, [["remove_B"]], tag="collect")
            elif eff == 1:
                B.R(trig, p, I, [["remove_B"], ["counter_inc"]], tag="collect_count")
            else:
                pass
        if eff == 2:
            U = B.T("used", PASS, count=(0, 0, 0))
            for p in P:
                B.R(trig, p, I, [["B_to", U]], tag="activate", hint=True)
            win.update(kind="none_left", X=[I])
        elif eff == 1:
            B.counter_used = True
            win.update(kind="counter", src=I)
        else:
            win.update(kind="none_left", X=[I])
    elif arche == "paint":
        T0 = B.T("tile", PASS, place="region", count=(1, 0, 1))
        T1 = B.T("tile", PASS, place="region", count=(0, 0, 0))
        toggle = r.random() < 0.4
        for p in P:
            B.R("enter", p, T0, [["B_to", T1]], tag="paint", hint=True)
            if toggle:
                B.R("enter", p, T1, [["B_to", T0]], tag="paint_back")
        u = r.random()
        if toggle and u < 0.5:
            win.update(kind="pattern", tiles=[T0, T1], region_mix=0.5)
        elif u < 0.3:
            win.update(kind="pattern", tiles=[T0, T1], region_mix=0.0)
        else:
            win.update(kind="none_left", X=[T0], region_mix=0.0)
    else:  # destroy
        X = B.T("target", BLOCK, place="any", count=(1, 0.5, 4))
        if r.random() < 0.4:
            X2 = B.T("cracked", BLOCK, count=(0, 0, 0))
            for p in P:
                B.R("bump", p, X, [["B_to", X2]], tag="hit1", hint=True)
                B.R("bump", p, X2, [["remove_B"]], tag="hit2")
            win.update(kind="none_left", X=[X, X2])
        else:
            for p in P:
                B.R("bump", p, X, [["remove_B"]], tag="destroy")
            win.update(kind="none_left", X=[X])
    if arche in ("push", "collect", "paint", "destroy") and r.random() < 0.3 and win["kind"] != "pattern":
        E = B.T("exit", PASS, place="far")
        win["and_reach"] = E
    win["arche"] = arche
    return arche, win


def _avatar_flavor(B, name, P, win, intro):
    r = B.r
    many = len(P) > 1 or B.types[P[0]]["count"] != [1, 0.0, 1]
    boxes = [t["id"] for t in B.types if t["role"] == "box" and t["solid"] == PUSH]
    if name == "hazard":
        H = B.T("hazard", PASS, place="any", count=(1, 0.6, 4), intro=intro)
        for p in P:
            B.R("enter", p, H, [["lose"]], intro=intro, tag="hazard", hint=True)
        for b in boxes:
            if r.random() < 0.5:
                B.R("enter", b, H, [["remove_A"], ["remove_B"]], intro=intro, tag="bridge")
    elif name == "teleport":
        Tp = B.T("tele", PASS, place="any", count=(2, 0, 2), intro=intro)
        for p in P + (boxes if r.random() < 0.4 else []):
            B.R("enter", p, Tp, [["teleport_A"]], intro=intro, tag="teleport")
    elif name == "gate":
        G = B.T("gate", PASS, gated=int(r.random() < 0.8), place="path", count=(1, 0.3, 2), intro=intro)
        u = r.random()
        if u < 0.2 and B.a5 is None:
            B.a5 = "gate"
        elif u < 0.35 and not B.uses_click:
            S = B.T("switch", BLOCK, place="any", intro=intro)
            B.uses_click = True
            B.R("click", None, S, [["gate"]], intro=intro, tag="remote_gate")
        else:
            S = B.T("switch", PASS, place="any", intro=intro)
            for p in P + boxes:
                B.R("enter", p, S, [["gate"]], intro=intro, tag="switch_gate", hint=r.random() < 0.5)
    elif name == "key":
        if B.counter_used:
            raise _Skip()
        B.counter_used = True
        K = B.T("key", PASS, place="any", intro=intro)
        D = B.T("door", BLOCK, place="path", intro=intro)
        for p in P:
            B.R("enter", p, K, [["remove_B"], ["counter_inc"]], intro=intro, tag="key")
            B.R("bump", p, D, [["spend"]], intro=intro, tag="door")
    elif name in ("patrol", "chase"):
        N = B.T("npc", PASS if r.random() < 0.6 else BLOCK, motion=name, place="far" if name == "chase" else "any",
                count=(1, 0.3, 2), intro=intro)
        for p in P:
            B.R("enter", p, N, [["lose"]], intro=intro, tag=name)
            B.R("enter", N, p, [["lose"]], intro=intro, tag=name)
            if B.types[N]["solid"] == BLOCK:
                B.R("bump", p, N, [["lose"]], intro=intro, tag=name)
    elif name == "conveyor":
        B.T("conveyor", PASS, motion="conveyor", place="any", count=(2, 0.5, 5), intro=intro)
    elif name == "grate":
        B.T("grate", PASS, grate=True, place="path", count=(1, 0.3, 2), intro=intro)
        B.grow_init = 1
        if B.a5 is None and r.random() < 0.4:
            B.a5 = "grow"
        else:
            M = B.T("shroom", PASS, place="any", intro=intro)
            for p in P:
                B.R("enter", p, M, [["grow"]] + ([["remove_B"]] if r.random() < 0.5 else []), intro=intro,
                    tag="grow", hint=True)
    elif name == "paintbox":
        if not boxes:
            raise _Skip()
        Q = B.T("paint", PASS, place="any", intro=intro)
        C = B.T("box2", int(r.choice([BLOCK, PUSH])), count=(0, 0, 0), intro=intro)
        B.R("enter", boxes[0], Q, [["A_to", C]], intro=intro, tag="paintbox", hint=True)
    elif name == "trail":
        if many:
            raise _Skip()
        C = B.T("trail", BLOCK, count=(0, 0, 0), intro=intro)
        B.R("step", P[0], None, [["spawn", C]], intro=intro, tag="trail")
    elif name == "toggleblocks":
        S = B.T("lever", PASS, place="any", intro=intro)
        C = B.T("red", BLOCK, place="path", count=(1, 0.4, 3), intro=intro)
        D = B.T("blue", PASS, place="any", count=(1, 0.4, 3), intro=intro)
        for p in P:
            B.R("enter", p, S, [["toggle_all", C, D]], intro=intro, tag="toggleblocks", hint=True)
    elif name == "rocks":
        Rk = B.T("rock", PUSH, motion="gravity", place="any", count=(1, 0.4, 3), intro=intro)
        B.gravity = int(r.integers(4))
        if B.a5 is None and r.random() < 0.4:
            B.a5 = "gflip"
        _ = Rk
    elif name == "wake":
        X = B.T("sleeper", BLOCK, place="path", intro=intro)
        Y = B.T("awake", PASS, count=(0, 0, 0), intro=intro)
        for p in P:
            B.R("adjacent", p, X, [["B_to", Y]], intro=intro, tag="wake", hint=True)
    elif name == "chest":
        if B.a5 not in (None, "interact"):
            raise _Skip()
        B.a5 = "interact"
        X = B.T("chest", BLOCK, place="path", intro=intro)
        if r.random() < 0.6:
            eff = [["remove_B"]]
        else:
            eff = [["B_to", B.T("chest_open", PASS, count=(0, 0, 0), intro=intro)]]
        for p in P:
            B.R("interact", p, X, eff, intro=intro, tag="chest")
    elif name == "crack":
        X = B.T("crack", BLOCK, place="path", intro=intro)
        X2 = B.T("crack2", BLOCK, count=(0, 0, 0), intro=intro)
        for p in P:
            B.R("bump", p, X, [["B_to", X2]], intro=intro, tag="crack")
            B.R("bump", p, X2, [["remove_B"]], intro=intro, tag="crack")
    elif name == "ice":
        B.T("ice", PASS, slip=True, place="any", count=(3, 1.0, 8), intro=intro)
    elif name == "remote":
        if B.uses_click:
            raise _Skip()
        B.uses_click = True
        X = B.T("bridge_off", BLOCK, place="path", intro=intro)
        Y = B.T("bridge_on", PASS, count=(0, 0, 0), intro=intro)
        B.R("click", None, X, [["toggle_all", X, Y]], intro=intro, tag="remote")
        B.R("click", None, Y, [["toggle_all", Y, X]], intro=intro, tag="remote")
    elif name == "kick":
        X = B.T("puck", BLOCK, place="inner", intro=intro)
        for p in P:
            B.R("bump", p, X, [["B_slide"]], intro=intro, tag="kick")
    else:
        raise _Skip()


class _Skip(Exception):
    pass


# ============================================================================ archétypes à clic (click / cursor)
def _click_core(B, scheme):
    r = B.r
    if scheme == "cursor":
        arche = str(r.choice(["toggle", "gravity", "lights"], p=[0.5, 0.3, 0.2]))
    else:
        arche = str(r.choice(["toggle", "gravity", "lights"], p=[0.5, 0.32, 0.18]))
    win = dict(and_reach=None, commit=False, arche=arche)
    if arche == "toggle":
        m = int(r.choice([2, 2, 3]))
        Ts = [B.T("tile", PASS, place="region", count=(1, 0, 1) if k == 0 else (0, 0, 0)) for k in range(m)]
        nbr = m == 2 and r.random() < 0.25
        for k in range(m):
            if nbr:
                B.R("click", None, Ts[k], [["flip_nbrs", Ts[0], Ts[1]]], tag="flip_nbrs")
            else:
                B.R("click", None, Ts[k], [["B_to", Ts[(k + 1) % m]]], tag="cycle", hint=k == 0)
        u = r.random()
        if u < 0.6:
            win.update(kind="pattern", tiles=Ts, region_mix=1.0)
        elif u < 0.85:
            win.update(kind="same", tiles=Ts, region_mix=1.0)
        else:
            win.update(kind="none_left", X=[Ts[1]], tiles=Ts, region_mix=1.0)
    elif arche == "lights":
        T0 = B.T("tile", PASS, place="region", count=(1, 0, 1))
        T1 = B.T("tile", PASS, place="region", count=(0, 0, 0))
        for t in (T0, T1):
            B.R("click", None, t, [["flip_nbrs", T0, T1]], tag="flip_nbrs")
        win.update(kind="none_left", X=[T1], tiles=[T0, T1], region_mix=1.0)
    else:  # gravity : une bille tombe, on retire / bascule des blocs pour la mener à la sortie
        B.gravity = int(r.integers(4))
        Rk = B.T("ball", BLOCK, motion="gravity", place="top", count=(1, 0, 1))
        E = B.T("exit", PASS, place="wallside")
        K = B.T("block", BLOCK, place="any", count=(3, 1.0, 10))
        u = r.random()
        if u < 0.5:
            B.R("click", None, K, [["remove_B"]], tag="g_remove", hint=True)
        else:
            K2 = B.T("ghost", PASS, place="any", count=(0, 0.5, 3))
            B.R("click", None, K, [["toggle_all", K, K2]], tag="g_toggle")
            B.R("click", None, K2, [["toggle_all", K2, K]], tag="g_toggle")
        if scheme == "click" and r.random() < 0.6:          # tourner la gravité de 90° : ACTION5 ou levier cliquable
            B.a5 = "grot"
        else:
            Lv = B.T("lever", BLOCK, place="any")
            B.R("click", None, Lv, [["grot"]], tag="g_rotate", hint=True)
        win.update(kind="all_on", X=[Rk], Y=E)
    return arche, win


def _toggle_flavor(B, name, win, intro):
    r = B.r
    Ts = win.get("tiles")
    if not Ts or win["arche"] != "toggle":
        raise _Skip()
    if name == "extra_color":
        if any(e[0] == "flip_nbrs" for ru in B.rules for e in ru["effects"]):
            raise _Skip()
        Tn = B.T("tile", PASS, place="region", count=(0, 0, 0), intro=intro)
        for ru in B.rules:
            if ru["trigger"] == "click" and ru["b"] == Ts[-1]:
                ru["until"] = intro
        B.R("click", None, Ts[-1], [["B_to", Tn]], intro=intro, tag="cycle+")
        B.R("click", None, Tn, [["B_to", Ts[0]]], intro=intro, tag="cycle+")
        win["tiles"] = Ts + [Tn]
    elif name == "nbr_switch":
        if len(Ts) != 2:
            raise _Skip()
        for ru in B.rules:
            if ru["trigger"] == "click":
                ru["until"] = min(ru["until"], intro)
        for t in Ts:
            B.R("click", None, t, [["flip_nbrs", Ts[0], Ts[1]]], intro=intro, tag="flip_nbrs")
    elif name == "mirror":
        if len(Ts) != 2:
            raise _Skip()
        for ru in B.rules:
            if ru["trigger"] == "click":
                ru["until"] = min(ru["until"], intro)
        for t in Ts:
            B.R("click", None, t, [["flip_mirror", Ts[0], Ts[1]]], intro=intro, tag="mirror")
    _ = r


def _gravity_flavor(B, name, win, intro):
    r = B.r
    if win["arche"] != "gravity":
        raise _Skip()
    Rk = win["X"][0]
    if name == "g_hazard":
        H = B.T("hazard", PASS, place="any", count=(1, 0.5, 3), intro=intro)
        B.R("enter", Rk, H, [["lose"]], intro=intro, tag="g_hazard", hint=True)
    elif name == "g_teleport":
        Tp = B.T("tele", PASS, place="any", count=(2, 0, 2), intro=intro)
        B.R("enter", Rk, Tp, [["teleport_A"]], intro=intro, tag="g_teleport")
    elif name == "g_rock2":
        Rk2 = B.T("ball", BLOCK, motion="gravity", place="top", count=(1, 0, 1), intro=intro)
        win["X"] = win["X"] + [Rk2]
        B.types[win["Y"]]["count"] = "match_winX"
    elif name == "g_conveyor":
        B.T("conveyor", PASS, motion="conveyor", place="any", count=(1, 0.5, 3), intro=intro)
    _ = r


# ============================================================================ tirage
def sample_rules(seed, attempt=0):
    r = np.random.default_rng([zlib.crc32(b"rules-grammar"), int(seed), int(attempt)])
    for _ in range(50):
        try:
            return _sample(r, seed, attempt)
        except (_Full, _Skip):
            continue
    raise RuntimeError("grammaire : aucun tirage valide")


def _sample(r, seed, attempt):
    B = _B(r)
    scheme = str(r.choice(SCHEMES, p=SCHEME_P))
    couple = str(r.choice(COUPLINGS)) if scheme == "coupled" else None
    slide = bool(scheme in ("avatar", "multi", "coupled", "select") and r.random() < 0.12)
    flav_used = []
    if scheme in ("cursor", "click"):
        if scheme == "cursor":
            B.a5 = "activate"
        arche, win = _click_core(B, scheme)
        cat = TOGGLE_FLAVORS if arche == "toggle" else GRAVITY_FLAVORS if arche == "gravity" else ()
        P = []
    else:
        P = _players(B, scheme)
        if scheme == "select":
            B.uses_click = True
        arche, win = _avatar_core(B, scheme, P)
        cat = AVATAR_FLAVORS
    # modules : 0-1 au niveau 0, puis 1-3 qui ARRIVENT aux niveaux suivants
    plan = []
    if cat:
        if r.random() < 0.45:
            plan.append(0)
        nm = int(r.choice([1, 1, 2, 2, 3]))
        plan += sorted(int(x) for x in r.choice([1, 2, 3, 4], size=nm, replace=False))
    for intro in plan:
        for _ in range(6):
            name = str(r.choice(cat))
            if name in flav_used:
                continue
            nt, nr, a5, uc, cu = len(B.types), len(B.rules), B.a5, B.uses_click, B.counter_used
            try:
                if cat is AVATAR_FLAVORS:
                    _avatar_flavor(B, name, P, win, intro)
                elif cat is TOGGLE_FLAVORS:
                    _toggle_flavor(B, name, win, intro)
                else:
                    _gravity_flavor(B, name, win, intro)
            except (_Skip, _Full):
                del B.types[nt:]
                del B.rules[nr:]
                B.a5, B.uses_click, B.counter_used = a5, uc, cu
                continue
            flav_used.append(name)
            B.mods.append(dict(intro=intro, name=name))
            break
    # ACTION5
    has_dyn = any(t["motion"] in ("patrol", "chase", "gravity", "conveyor") for t in B.types)
    if B.a5 is None:
        u = r.random()
        if win["kind"] not in ("reach", "all_on") and u < 0.2:
            B.a5 = "commit"
            win["commit"] = True
        elif has_dyn and u < 0.6:
            B.a5 = "wait"
    # présentation
    legend = bool(win["kind"] == "pattern" or r.random() < 0.5)
    hints = [k for k, ru in enumerate(B.rules) if ru["hint"]]
    show_counter = bool(B.counter_used and r.random() < 0.7)
    if scheme in ("cursor", "click"):
        style = "open" if arche != "gravity" else str(r.choice(["room", "arena"]))
    elif win["kind"] in ("pattern",) or arche == "paint":
        style = str(r.choice(["arena", "open"]))
    else:
        style = str(r.choice(["room", "room", "maze", "arena"]))
    layout = dict(style=style, lo=int(r.integers(5, 8)), growth=float(r.uniform(0.5, 1.1)),
                  density=float(r.uniform(0.05, 0.22)), loops=float(r.uniform(0.05, 0.3)))
    if arche in ("toggle", "lights"):
        layout.update(lo=int(r.integers(3, 5)), growth=float(r.uniform(0.3, 0.6)))
        if scheme == "cursor":
            layout.update(lo=3, growth=float(r.uniform(0.2, 0.4)))
    if arche == "gravity":
        layout.update(style="room", lo=int(r.integers(6, 8)), growth=float(r.uniform(0.4, 0.8)),
                      density=float(r.uniform(0.12, 0.28)))
    if arche == "gravity" and scheme == "cursor":
        layout.update(lo=int(r.integers(5, 7)), growth=float(r.uniform(0.2, 0.5)))
    if arche == "paint":
        layout.update(lo=int(r.integers(4, 6)), growth=float(r.uniform(0.3, 0.6)))
    if scheme == "select":
        layout.update(lo=int(r.integers(4, 6)), growth=float(r.uniform(0.3, 0.7)))
    spec = dict(seed=int(seed), attempt=int(attempt), scheme=scheme, couple=couple, slide=slide, a5=B.a5,
                uses_click=bool(B.uses_click or scheme in ("click", "select")), types=B.types, rules=B.rules,
                win=win, gravity=B.gravity, grow_init=B.grow_init, layout=layout, legend=legend,
                hints=hints[:2] if legend else [], show_counter=show_counter, modules=B.mods, arche=arche,
                chase_every=int(r.choice([1, 2])))
    return json.loads(json.dumps(spec))               # sérialisable (et copie profonde)


# ============================================================================ signatures / diversité
def _role(spec, t):
    if t is None:
        return "-"
    return spec["types"][t]["role"]


def signature(spec):
    """(contrôle, ensemble d'interactions, victoire) : identifie la « règle du jeu » à l'apparence près."""
    ctrl = (spec["scheme"], spec["couple"], spec["slide"], spec["a5"],
            tuple(sorted({t["motion"] for t in spec["types"] if t["motion"]})))
    inter = frozenset((ru["trigger"], _role(spec, ru["a"]), _role(spec, ru["b"]),
                       tuple(e[0] for e in ru["effects"]), ru["intro"] > 0) for ru in spec["rules"])
    w = spec["win"]
    win = (w["kind"], w["and_reach"] is not None, w["commit"])
    return ctrl, inter, win


def describe(spec):
    w = spec["win"]
    rules = "; ".join(f"{ru['trigger']}({_role(spec, ru['a'])},{_role(spec, ru['b'])})->"
                      + "+".join(e[0] for e in ru["effects"]) + (f"@{ru['intro']}" if ru["intro"] else "")
                      for ru in spec["rules"])
    mods = ",".join(f"{m['name']}@{m['intro']}" for m in spec["modules"])
    return (f"{spec['scheme']}{'/' + spec['couple'] if spec['couple'] else ''}{'/slide' if spec['slide'] else ''} "
            f"a5={spec['a5']} | {spec['arche']} win={w['kind']}{'+exit' if w['and_reach'] is not None else ''}"
            f"{'+commit' if w['commit'] else ''} | mods[{mods}] | {rules}")


def diversity(n=2000, seed0=0):
    """nombre de signatures distinctes sur n specs tirées (sans construire les niveaux)."""
    sigs, ctrl, inter, win = set(), set(), set(), set()
    for s in range(seed0, seed0 + n):
        c, i, w = signature(sample_rules(s))
        sigs.add((c, i, w))
        ctrl.add(c)
        inter.add(i)
        win.add(w)
    return dict(n=n, signatures=len(sigs), controls=len(ctrl), interaction_sets=len(inter), wins=len(win))


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    for s in range(5):
        print(describe(sample_rules(s)))
    print(diversity(n))
