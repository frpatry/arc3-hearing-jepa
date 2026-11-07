"""Moteur générique `RuleGame` : exécute n'importe quelle spec tirée par `grammar.sample_rules`.

État (hashable) : (objs, V)
    objs = tuple trié de (r, c, type, aux)      aux = direction (PNJ patrouilleur, tapis roulant), 0 sinon
    V    = (compteur, gate, gravité, grand, sel_r, sel_c, phase, validé)
Un tour = action du joueur (déplacement des types contrôlés / clic / ACTION5) -> règles déclenchées au fil de l'eau
-> dynamique (tapis, PNJ, gravité) -> règles « adjacent ». Les règles d'un même (règle, A, B) ne se déclenchent
qu'une fois par tour (pas de boucle).

    make_rule_game(seed, look) -> RuleGame       ré-échantillonne la spec (attempt += 1) tant que le jeu est rejeté
    STATS                                         compteurs de rejet (spec / niveau, par raison)
"""
import time
import zlib
from collections import Counter, deque

import numpy as np

from .core import (AVATAR, BG, DETAIL, DIR_OPS, EV, GOAL, HUD_FILL, LIVES, MARK, PANEL, WALL, Canvas, Game,
                   Unsolvable, bfs_dist, connected_room, perfect_maze, sprite)
from .grammar import BLOCK, PASS, PUSH, describe, sample_rules, signature

DV = ((-1, 0), (1, 0), (0, -1), (0, 1))                 # up, down, left, right
DIDX = {"up": 0, "down": 1, "left": 2, "right": 3}
OPPO = (1, 0, 3, 2)
ROT = (3, 2, 0, 1)                                       # rotation horaire de la gravité
COUPLE = {"same": (0, 1, 2, 3), "mirror_h": (0, 1, 3, 2), "mirror_v": (1, 0, 2, 3), "invert": (1, 0, 3, 2),
          "rot90": (3, 2, 0, 1)}
CNT, GATE, GDIR, BIG, SR, SC, PH, COM = range(8)
SEMS = [AVATAR, GOAL, 4, 5, 6, 7, 8, 9, 10, 11, 12]
STATS = Counter()
_MASKS = {}


def custom_mask(seed, t):
    """motif aléatoire t×t symétrique (miroir gauche-droite), 1 = couleur, 2 = détail."""
    key = (seed, t)
    m = _MASKS.get(key)
    if m is None:
        r = np.random.default_rng([seed, t])
        if t <= 2:
            m = np.ones((t, t), np.int8)
        else:
            for _ in range(20):
                half = (r.random((t, (t + 1) // 2)) < 0.6).astype(np.int8)
                m = np.concatenate([half, half[:, :t // 2][:, ::-1]], 1)
                if m.sum() >= t * t // 3 and m[:, 0].any() and m[0].any() and m[-1].any():
                    break
            if t >= 3 and r.random() < 0.5:
                m[t // 2, t // 2] = 2
        _MASKS[key] = m
    return m


def arrow_mask(t, d):
    m = np.zeros((t, t), np.int8)
    if t <= 2:
        m[:] = 1
        return m
    mid = t // 2
    m[mid, 1:t - 1] = 1 if d in (2, 3) else 0
    m[1:t - 1, mid] = 1 if d in (0, 1) else m[1:t - 1, mid]
    tip = {0: (0, mid), 1: (t - 1, mid), 2: (mid, 0), 3: (mid, t - 1)}[d]
    m[tip] = 2
    return m


class _Sim:
    """simulation d'UN tour (état mutable local).
    Indices d'objets : j >= 0 = objet DYNAMIQUE (dans l'état) ; j < 0 = objet STATIQUE n° -j-1 (dans le niveau :
    sorties, plaques, dangers, téléporteurs, portes, tapis... types qu'aucune règle ne peut modifier)."""
    __slots__ = ("g", "lvl", "O", "alive", "occ", "V", "dead", "log", "fired", "moved", "R", "d", "prev", "S",
                 "socc")

    def __init__(self, g, lvl, st, log):
        objs, V = st
        self.g, self.lvl, self.log = g, lvl, log
        self.O = O = [list(o) for o in objs]
        self.alive = [True] * len(O)
        occ = {}
        for i, o in enumerate(O):
            k = (o[0], o[1])
            lst = occ.get(k)
            if lst is None:
                occ[k] = [i]
            else:
                lst.append(i)
        self.occ = occ
        self.S, self.socc = lvl["static"], lvl["socc"]
        self.V = list(V)
        self.dead = False
        self.fired = set()
        self.moved = set()
        self.R = lvl["R"]
        self.d = None
        self.prev = None

    # ---------------------------------------------------------------- outils
    def ob(self, j):
        return self.O[j] if j >= 0 else self.S[-j - 1]

    def live(self, j):
        return j < 0 or self.alive[j]

    def solidity(self, j):
        g, t = self.g, (self.O[j][2] if j >= 0 else self.S[-j - 1][2])
        gd = g.gated[t]
        if gd is not None:
            return PASS if self.V[GATE] == gd else BLOCK
        if g.grate[t]:
            return BLOCK if self.V[BIG] else PASS
        return g.solid[t]

    def relocate(self, i, r, c):
        o = self.O[i]
        self.occ[(o[0], o[1])].remove(i)
        o[0], o[1] = r, c
        lst = self.occ.get((r, c))
        if lst is None:
            self.occ[(r, c)] = [i]
        else:
            lst.append(i)

    def kill(self, i):
        if i >= 0 and self.alive[i]:
            self.alive[i] = False
            o = self.O[i]
            self.occ[(o[0], o[1])].remove(i)

    def spawn(self, r, c, t):
        if len(self.O) >= 80:
            return
        self.O.append([r, c, t, 0])
        self.alive.append(True)
        self.occ.setdefault((r, c), []).append(len(self.O) - 1)

    def at(self, r, c):
        lst = self.occ.get((r, c))
        out = [j for j in lst if self.alive[j]] if lst else []
        s = self.socc.get((r, c))
        if s:
            out.extend(s)
        return out

    def L(self, *x):
        if self.log is not None:
            self.log.append(x)

    # ---------------------------------------------------------------- règles
    def fire(self, trig, i, j, once=True):
        ta = self.O[i][2] if i is not None else -1
        tb = self.ob(j)[2] if j is not None else -1
        rl = self.R.get((trig, ta, tb))
        if not rl:
            return
        for ridx, effects in rl:
            if once:
                fk = (ridx, i, j)
                if fk in self.fired:
                    continue
                self.fired.add(fk)
            if self.log is not None:
                self.log.append(("rule", ridx, trig))
            for eff in effects:
                self.effect(eff, i, j)

    def effect(self, eff, i, j):
        name = eff[0]
        O, V = self.O, self.V
        self.L("eff", name)
        dyn_j = j is not None and j >= 0 and self.alive[j]
        if name == "remove_B":
            if dyn_j:
                self.kill(j)
        elif name == "remove_A":
            if i is not None:
                self.kill(i)
        elif name == "B_to":
            if dyn_j:
                O[j][2], O[j][3] = eff[1], 0
        elif name == "A_to":
            if i is not None and self.alive[i]:
                O[i][2], O[i][3] = eff[1], 0
        elif name == "teleport_A":
            if i is None or j is None or not self.alive[i]:
                return
            oj = self.ob(j)
            tb = oj[2]
            cand = [(o[0], o[1]) for k, o in enumerate(O) if self.alive[k] and k != j and o[2] == tb]
            cand += [(o[0], o[1]) for k, o in enumerate(self.S) if -k - 1 != j and o[2] == tb]
            if not cand:
                return
            cand.sort()
            here = (oj[0], oj[1])
            nxt = next((p for p in cand if p > here), cand[0])
            if any(self.solidity(k) != PASS for k in self.at(*nxt) if k != i):
                return
            self.relocate(i, *nxt)
            self.moved.add(i)
            self.L("teleport")
        elif name == "counter_inc":
            V[CNT] = min(V[CNT] + 1, 9)
        elif name == "spend":
            if V[CNT] > 0 and dyn_j:
                V[CNT] -= 1
                self.kill(j)
                self.L("open")
        elif name == "gate":
            V[GATE] ^= 1
        elif name in ("B_step", "B_slide"):
            if not dyn_j:
                return
            d = self.d if self.d is not None else (self.V[GDIR] if self.g.spec["gravity"] is not None else 1)
            for _ in range(1 if name == "B_step" else 20):
                if not self.try_move(j, d, 1, mode=2):
                    break
        elif name == "gflip":
            V[GDIR] = OPPO[V[GDIR]]
        elif name == "grot":
            V[GDIR] = ROT[V[GDIR]]
        elif name == "toggle_all":
            C, D = eff[1], eff[2]
            for k, o in enumerate(O):
                if self.alive[k]:
                    if o[2] == C:
                        o[2] = D
                    elif o[2] == D:
                        o[2] = C
        elif name == "grow":
            V[BIG] ^= 1
        elif name == "spawn":
            if self.prev is not None and i is not None:
                r, c = self.prev
                if all(self.solidity(k) == PASS for k in self.at(r, c)):
                    self.spawn(r, c, eff[1])
        elif name == "lose":
            self.dead = True
        elif name in ("flip_nbrs", "flip_mirror"):
            if j is None:
                return
            C, D = eff[1], eff[2]
            oj = self.ob(j)
            r, c = oj[0], oj[1]
            if name == "flip_nbrs":
                cells = [(r, c)] + [(r + dr, c + dc) for dr, dc in DV]
            else:
                cells = [(r, c), (r, self.lvl["W"] - 1 - c)] if self.lvl["W"] - 1 - c != c else [(r, c)]
            for p in cells:
                for k in self.at(*p):
                    if k >= 0:
                        if O[k][2] == C:
                            O[k][2] = D
                        elif O[k][2] == D:
                            O[k][2] = C

    # ---------------------------------------------------------------- déplacement
    def try_move(self, i, d, depth=0, mode=0):
        """mode 0 : pousse ; 1 : PNJ (ne pousse pas, entre sur les avatars) ; 2 : chute / glissade (ne pousse pas)."""
        if not self.alive[i]:
            return False
        o = self.O[i]
        dr, dc = DV[d]
        r, c = o[0] + dr, o[1] + dc
        lvl, g = self.lvl, self.g
        if not (0 <= r < lvl["H"] and 0 <= c < lvl["W"]) or (r, c) in lvl["wallset"]:
            return False
        for j in self.at(r, c):
            if j == i or not self.live(j):
                continue
            if mode == 1 and j >= 0 and g.ctrl[self.O[j][2]]:
                continue
            s = self.solidity(j)
            if s == PASS:
                continue
            if s == PUSH and mode == 0 and depth < 3 and j >= 0:
                if self.try_move(j, d, depth + 1, 0):
                    self.L("push")
                    self.fire("push", i, j)
                    continue
            self.L("bumpobj")
            self.fire("bump", i, j)
            return False
        if not self.alive[i]:
            return False
        prev = (o[0], o[1])
        self.relocate(i, r, c)
        self.moved.add(i)
        for j in self.at(r, c):
            if j != i and self.alive[i] and self.live(j):
                oj = self.ob(j)
                if oj[0] == r and oj[1] == c:
                    self.fire("enter", i, j)
        if self.alive[i] and self.g.has_step:
            self.prev = prev
            self.fire("step", i, None, once=False)          # à chaque pas (même en glissade)
        return True

    def on_slip(self, i):
        o = self.O[i]
        return any(self.g.slip[self.ob(k)[2]] for k in self.at(o[0], o[1]))

    def move_actor(self, i, d):
        self.d = d
        moved = False
        for _ in range(30):
            if not self.try_move(i, d):
                break
            moved = True
            if not self.alive[i] or not (self.g.slide or (self.g.has_slip and self.on_slip(i))):
                break
        if not moved:
            self.L("bump")
        return moved

    # ---------------------------------------------------------------- clic
    def click_cell(self, r, c):
        hit = False
        for j in self.at(r, c):
            if self.lvl["clickable"].get(self.ob(j)[2]):
                hit = True
                self.fire("click", None, j)
        return hit

    # ---------------------------------------------------------------- dynamique
    def dynamics(self):
        g, O = self.g, self.O
        if g.has_conv:
            carried = []
            for o in self.S:
                if g.motion[o[2]] == "conveyor":
                    for k in self.at(o[0], o[1]):
                        if k >= 0:
                            t = O[k][2]
                            if g.ctrl[t] or g.solid[t] == PUSH or g.motion[t] == "gravity":
                                carried.append((k, o[3]))
            for k, d in carried:
                self.d = d
                self.try_move(k, d, 0, 0)
        if g.has_npc:
            for k in range(len(O)):
                if not self.alive[k]:
                    continue
                m = g.motion[O[k][2]]
                if m == "patrol":
                    if not self.try_move(k, O[k][3], 0, 1) and self.alive[k]:
                        O[k][3] = OPPO[O[k][3]]
                elif m == "chase" and self.V[PH] == 0:
                    tg = [(o[0], o[1]) for q, o in enumerate(O) if self.alive[q] and g.ctrl[o[2]]]
                    if not tg:
                        continue
                    r, c = O[k][0], O[k][1]
                    tr, tc = min(tg, key=lambda p: abs(p[0] - r) + abs(p[1] - c))
                    cand = []
                    if tr != r:
                        cand.append(0 if tr < r else 1)
                    if tc != c:
                        cand.append(2 if tc < c else 3)
                    if abs(tc - c) > abs(tr - r):
                        cand = cand[::-1]
                    for d in cand:
                        if self.try_move(k, d, 0, 1):
                            break
            if g.has_chase:
                self.V[PH] = (self.V[PH] + 1) % g.spec["chase_every"]
        if g.has_grav:
            self.settle()

    def settle(self):
        g, O = self.g, self.O
        d = self.V[GDIR]
        dr, dc = DV[d]
        for _ in range(40):
            movers = [k for k in range(len(O)) if self.alive[k] and g.motion[O[k][2]] == "gravity"]
            movers.sort(key=lambda k: -(O[k][0] * dr + O[k][1] * dc))
            any_m = False
            for k in movers:
                if self.try_move(k, d, 0, 2):
                    any_m = True
            if not any_m:
                break

    def adjacency(self):
        if not self.g.has_adj:
            return
        O = self.O
        for i in list(self.moved):
            if not self.alive[i]:
                continue
            r, c = O[i][0], O[i][1]
            for dr, dc in DV:
                for j in self.at(r + dr, c + dc):
                    self.fire("adjacent", i, j)

    def state(self):
        alive = self.alive
        objs = tuple(sorted(tuple(o) for k, o in enumerate(self.O) if alive[k]))
        return objs, tuple(self.V)


class RuleGame(Game):
    family = "rules"
    n_levels = (4, 8)
    n_levels_official = (4, 8)
    bfs_cap = 30000
    table_cap = 5000
    max_tries = 12
    min_levels = 4

    def __init__(self, seed=0, look="classic", spec=None, attempt=0):
        self._t0 = time.time()
        self.spec = spec if spec is not None else sample_rules(seed, attempt)
        self.attempt = int(attempt)
        self.seed = int(seed)
        self.look = look
        self.official = look == "official"
        self.game_id = f"{self.family}-{'o' if self.official else ''}{self.seed:04d}"
        self.rng = np.random.default_rng([zlib.crc32(b"rules"), self.seed, self.attempt])
        self.orng = np.random.default_rng([zlib.crc32(b"rules"), self.seed, self.attempt, 7])
        self._rcache = {}
        self.max_cells, self.px_budget = 99, 58
        if self.official:
            self._official_view(self.orng)
        self.setup(self.rng)
        self._presentation(self.rng)
        if self.official:
            self._official_presentation(self.orng)
        lo, hi = self.n_levels_official if self.official else self.n_levels
        n = int(self.rng.integers(lo, hi + 1))
        self.levels = []
        for i in range(n):
            try:
                self.levels.append(self._make_level(i, n))
            except Unsolvable:
                if i >= self.min_levels:
                    STATS["game_truncated"] += 1
                    break
                raise
        self.baseline_actions = [L["baseline"] for L in self.levels]
        self.win_levels = len(self.levels)
        self._full_reset()

    # ------------------------------------------------------------------ mise en place
    def setup(self, rng):
        sp = self.spec
        T = sp["types"]
        self.solid = [t["solid"] for t in T]
        self.ctrl = [t["ctrl"] for t in T]
        self.motion = [t["motion"] for t in T]
        self.gated = [t["gated"] for t in T]
        self.grate = [t["grate"] for t in T]
        self.slip = [t["slip"] for t in T]
        self.scheme = sp["scheme"]
        self.slide = sp["slide"]
        self.cmap = COUPLE[sp["couple"]] if sp["couple"] else None
        trig = {ru["trigger"] for ru in sp["rules"]}
        self.has_step = "step" in trig
        self.has_adj = "adjacent" in trig
        self.has_conv = any(m == "conveyor" for m in self.motion)
        self.has_chase = any(m == "chase" for m in self.motion)
        self.has_npc = any(m in ("patrol", "chase") for m in self.motion)
        self.has_grav = any(m == "gravity" for m in self.motion)
        self.has_slip = any(self.slip)
        # chemin rapide : un seul avatar, aucune dynamique -> un pas dans une case vide = simple déplacement
        self.fast = (self.scheme == "avatar" and not (self.has_step or self.has_adj or self.has_conv or self.has_npc
                                                      or self.has_grav or self.slide))
        ops = [] if self.scheme == "click" else list(DIR_OPS)
        if sp["a5"]:
            ops.append("interact")
        self.simple_ops = tuple(ops)
        self.uses_click = bool(sp["uses_click"])
        self.col = [SEMS[k % len(SEMS)] for k in range(len(T))]
        # types STATIQUES : aucune règle ne peut les déplacer / transformer / retirer -> hors de l'état (BFS plus léger)
        mut = set()
        for ru in sp["rules"]:
            for e in ru["effects"]:
                n = e[0]
                if n in ("remove_B", "B_to", "B_step", "B_slide", "spend", "flip_nbrs", "flip_mirror") and ru["b"] is not None:
                    mut.add(ru["b"])
                if n in ("remove_A", "A_to", "teleport_A") and ru["a"] is not None:
                    mut.add(ru["a"])
                if n in ("B_to", "A_to", "spawn"):
                    mut.add(e[1])
                if n in ("toggle_all", "flip_nbrs", "flip_mirror"):
                    mut.update((e[1], e[2]))
        self.static_t = [not t["ctrl"] and t["motion"] in (None, "conveyor") and t["solid"] != PUSH and t["id"] not in mut
                         for t in T]
        self.is_tile = [t["place"] == "region" for t in T]
        self.lose_types = set()
        for ru in sp["rules"]:
            if any(e[0] == "lose" for e in ru["effects"]):
                for x in (ru["a"], ru["b"]):
                    if x is not None and not self.ctrl[x]:
                        self.lose_types.add(x)
        need = 3 if self.scheme in ("cursor", "select") else 2 if any(self.is_tile) else 1
        if self.official:
            if self.T < need:
                self.T = need
                self.s_screen = min(self.s_screen, 2)
                self.px_budget = 62 // self.s_screen - 2
            self.t_pref = self.T
            self.wall_style = "thin" if self.wall_mode == "thin" else str(self.orng.choice(["full", "small"]))
        else:
            self.t_pref = int(rng.choice([2, 3] if need > 1 else [1, 2, 3]))
            if need > 1 and self.t_pref < need:
                self.t_pref = need
            self.wall_style = str(rng.choice(["full", "full", "small"]))
        self.sty = [t["sprite"] for t in T]
        ar = sp["arche"]                                   # taille max du plateau (garde le BFS sous le plafond)
        self.max_side = (13 if ar in ("reach", "collect", "destroy") and self.scheme == "avatar" else
                         5 if ar in ("toggle", "lights", "chain") else 10)
        if self.scheme == "cursor" and ar != "gravity":
            self.max_side = 4
        self.big_used = any(self.grate) or any(e[0] == "grow" for ru in sp["rules"] for e in ru["effects"]) \
            or sp["a5"] == "grow"

    # ------------------------------------------------------------------ niveaux
    def _counts(self, act, i, win):
        cnt = {}
        for t in act:
            c = t["count"]
            if isinstance(c, list):
                cnt[t["id"]] = int(min(c[2] if len(c) > 2 else c[0], c[0] + int(c[1] * i)))
        for t in act:
            c = t["count"]
            if isinstance(c, str):
                if c.startswith("match:"):
                    cnt[t["id"]] = cnt.get(int(c.split(":")[1]), 1)
                else:   # match_winX
                    cnt[t["id"]] = max(1, sum(cnt.get(x, 0) for x in win.get("X", [])))
        return cnt

    def gen_level(self, rng, i, n):
        sp = self.spec
        W_ = sp["win"]
        Lay = sp["layout"]
        act = [t for t in sp["types"] if t["intro"] <= i < t["until"]]
        act_ids = {t["id"] for t in act}
        rules = [k for k, ru in enumerate(sp["rules"]) if ru["intro"] <= i < ru["until"]
                 and (ru["a"] is None or ru["a"] in act_ids) and (ru["b"] is None or ru["b"] in act_ids)]
        new_rules = {k for k in rules if sp["rules"][k]["intro"] == i and i > 0       # (une règle « lose » ne
                     and any(e[0] != "lose" for e in sp["rules"][k]["effects"])}           # sert jamais à gagner)
        cnt = self._counts(act, i, W_)
        # ---- taille
        t = self.t_pref
        lim = (62 // self.s_screen) if self.official else 60
        panel_est = (t + 2) if (sp["legend"] or sp["show_counter"]) else 0
        maxside = max(3, min(self.max_side, (lim - panel_est) // max(1, min(t, 2))))
        side = Lay["lo"] + Lay["growth"] * i + rng.random() - self._attempt // 4 - self._shrink
        side = int(max(3, min(maxside, side)))
        H = int(np.clip(side + rng.integers(-1, 2), 3, maxside))
        W = int(np.clip(side + rng.integers(-1, 2), 3, maxside))
        style = Lay["style"]
        if style == "maze" and min(H, W) >= 5:
            walls = perfect_maze(rng, (H - 1) // 2, (W - 1) // 2, Lay["loops"])
        elif style == "room" and min(H, W) >= 5:
            walls = connected_room(rng, H, W, Lay["density"], border=True)
            if walls is None:
                return self._fail("room")
        elif style == "arena" and min(H, W) >= 5:
            walls = np.zeros((H, W), bool)
            walls[0, :] = walls[-1, :] = walls[:, 0] = walls[:, -1] = True
        else:
            walls = np.zeros((H, W), bool)
        H, W = walls.shape
        while t > 1 and max(H, W) * t > lim - panel_est:
            t -= 1
        if max(H, W) * t > lim - panel_est:
            return self._fail("too_big")
        free = [(int(r), int(c)) for r, c in np.argwhere(~walls)]
        if len(free) < 6:
            return self._fail("few_cells")
        order = rng.permutation(len(free))
        free = [free[k] for k in order]
        used = set()
        objs = []
        gdir = sp["gravity"] if sp["gravity"] is not None else 1

        def take(cands):
            for p in cands:
                if p not in used:
                    used.add(p)
                    return p
            return None

        def put(tid, p):
            aux = int(rng.integers(4)) if self.motion[tid] in ("patrol", "conveyor") else 0
            objs.append((p[0], p[1], tid, aux))

        def wall_nb(p):
            return sum(1 for dr, dc in DV if not (0 <= p[0] + dr < H and 0 <= p[1] + dc < W) or walls[p[0] + dr, p[1] + dc])

        byplace = {}
        for tt in act:
            byplace.setdefault(tt["place"], []).append(tt["id"])
        # avatars
        starts = []
        for tid in byplace.get("start", []):
            for _ in range(cnt.get(tid, 1)):
                p = take(free)
                if p is None:
                    return self._fail("place")
                put(tid, p)
                starts.append(p)
        # région (tuiles)
        region = []
        if "region" in byplace:
            tiles = byplace["region"]
            mix = W_.get("region_mix", 1.0)
            region = sorted(free)
            if self.scheme not in ("click", "cursor"):          # peinture à l'avatar : petit rectangle qui grandit
                area = min(10, 4 + i)
                fs = set(free)
                best = []
                for _ in range(30):
                    h = int(rng.integers(2, 5))
                    w_ = max(2, -(-area // h))
                    r0, c0 = int(rng.integers(0, max(1, H - h + 1))), int(rng.integers(0, max(1, W - w_ + 1)))
                    cells = [(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w_) if (r, c) in fs]
                    if len(cells) > len(best):
                        best = cells
                    if len(cells) == h * w_:
                        break
                region = sorted(best)
                if len(region) < 3:
                    return self._fail("region")
            for p in region:
                if mix > 0 and rng.random() < mix:
                    tid = int(rng.choice(tiles))
                else:
                    tid = tiles[0]
                put(tid, p)
        anchor = starts[0] if starts else free[0]
        dist = bfs_dist(walls, anchor)
        dmax = max(1, int(dist.max()))
        far_obj = None
        for tid in byplace.get("far", []):
            for _ in range(cnt.get(tid, 1)):
                p = take([q for q in free if dist[q] >= 0.6 * dmax] + free)
                if p is None:
                    return self._fail("place")
                put(tid, p)
                far_obj = far_obj or p
        for tid in byplace.get("inner", []):
            for _ in range(cnt.get(tid, 1)):
                p = take([q for q in free if wall_nb(q) == 0] + [q for q in free if wall_nb(q) <= 1])
                if p is None:
                    return self._fail("place")
                put(tid, p)
        # chemin avatar -> objet lointain (portes, grilles, dormeurs...)
        tgt = far_obj or max(free, key=lambda q: dist[q])
        path = []
        if starts:
            d2 = bfs_dist(walls, tgt)
            if d2[anchor] > 2:
                p = anchor
                while p != tgt:
                    nb = [(p[0] + dr, p[1] + dc) for dr, dc in DV]
                    p = next(q for q in nb if 0 <= q[0] < H and 0 <= q[1] < W and d2[q] == d2[p] - 1)
                    path.append(p)
                path = path[:-1]
        for tid in byplace.get("path", []):
            for _ in range(cnt.get(tid, 1)):
                p = take([path[k] for k in rng.permutation(len(path))] + free)
                if p is None:
                    return self._fail("place")
                put(tid, p)
        # gravité : haut / repos
        dr, dc = DV[gdir]
        proj = {q: q[0] * dr + q[1] * dc for q in free}
        lo_p, hi_p = min(proj.values()), max(proj.values())
        for tid in byplace.get("top", []):
            for _ in range(cnt.get(tid, 1)):
                p = take([q for q in free if proj[q] <= lo_p + (hi_p - lo_p) / 3] + free)
                if p is None:
                    return self._fail("place")
                put(tid, p)
        for tid in byplace.get("rest", []):
            rest = [q for q in free if not (0 <= q[0] + dr < H and 0 <= q[1] + dc < W) or walls[q[0] + dr, q[1] + dc]]
            for _ in range(cnt.get(tid, 1)):
                p = take([q for q in rest if proj[q] >= lo_p + (hi_p - lo_p) / 2] + rest)
                if p is None:
                    return self._fail("place")
                put(tid, p)
        for tid in byplace.get("wallside", []):
            far_b = [o for o in objs if self.motion[o[2]] == "gravity"]
            db = bfs_dist(walls, (far_b[0][0], far_b[0][1])) if far_b else dist
            ws = [q for q in free if wall_nb(q) >= 1 and db[q] >= 0.6 * max(1, int(db.max()))]
            for _ in range(cnt.get(tid, 1)):
                p = take(ws + [q for q in free if wall_nb(q) >= 1])
                if p is None:
                    return self._fail("place")
                put(tid, p)
        for tid in byplace.get("any", []):
            for _ in range(cnt.get(tid, 0)):
                p = take(free)
                if p is None:
                    return self._fail("place")
                put(tid, p)
        if region:                                         # pas de tuile sous un objet solide (impeignable)
            solid_cells = {(o[0], o[1]) for o in objs if self.solid[o[2]] != PASS and not self.ctrl[o[2]]}
            objs = [o for o in objs if not (self.is_tile[o[2]] and (o[0], o[1]) in solid_cells)]
        # ---- règles actives indexées
        R = {}
        clickable = {}
        for k in rules:
            ru = sp["rules"][k]
            a = -1 if ru["a"] is None else ru["a"]
            b = -1 if ru["b"] is None else ru["b"]
            R.setdefault((ru["trigger"], a, b), []).append((k, ru["effects"]))
            if ru["trigger"] == "click" and b >= 0:
                clickable[b] = True
        if self.scheme == "select":
            for tid in act_ids:
                if self.ctrl[tid]:
                    clickable[tid] = True
        sel = (-1, -1)
        if self.scheme == "cursor":
            sel = (H // 2, W // 2)
        elif self.scheme == "select" and starts and rng.random() < 0.5:
            sel = starts[0]
        V0 = (0, 0, gdir, int(sp["grow_init"]), sel[0], sel[1], 0, 0)
        static = tuple(sorted(o for o in objs if self.static_t[o[2]]))
        socc = {}
        for k, o in enumerate(static):
            socc.setdefault((o[0], o[1]), []).append(-k - 1)
        lvl = dict(H=H, W=W, t=t, walls=walls, wallset={(int(r), int(c)) for r, c in np.argwhere(walls)}, R=R,
                   clickable=clickable, rules=rules, new_rules=new_rules, act=act_ids, region=region,
                   win=dict(W_), ph=0, items=[], static=static, socc={k: tuple(v) for k, v in socc.items()})
        st0 = (tuple(sorted(o for o in objs if not self.static_t[o[2]])), V0)
        if self.has_grav:                                 # la gravité s'applique dès le départ
            sim = _Sim(self, lvl, st0, None)
            sim.settle()
            if sim.dead:
                return self._fail("dead_at_start")
            st0 = sim.state()
        # ---- paramètres de victoire propres au niveau
        w = lvl["win"]
        w["tileset"] = set(w.get("tiles", []))
        w["Xs"] = set(w.get("X", []))
        w["target"] = None
        if w["kind"] == "counter":
            k = sum(1 for o in self.allo(lvl, st0[0]) if o[2] == w["src"])
            if k == 0:
                return self._fail("no_counter_src")
            w["k"] = k if rng.random() < 0.6 else max(1, k - 1)
        clicky = self.scheme in ("click", "cursor")
        ncol = len([x for x in W_.get("tiles", []) if x in act_ids])
        nk = int(2 + i // 2 + rng.integers(0, 2)) if self.scheme == "click" else int(1 + (i + 1) // 3 + rng.integers(0, 2))
        nk = min(nk, 4 if ncol <= 2 else 3) - self._shrink // 2
        if clicky and region and w["kind"] in ("same", "none_left") and sp["arche"] in ("toggle", "lights"):
            tiles = [x for x in w["tiles"] if x in act_ids]
            goal = int(rng.choice(tiles)) if w["kind"] == "same" else tiles[0]
            st0 = (tuple(sorted((o[0], o[1], goal if o[2] in tiles else o[2], o[3]) for o in st0[0])), st0[1])
            st0 = self._scramble(lvl, st0, max(2, nk), rng)       # état but brouillé par k clics
        if w["kind"] == "pattern":
            tiles = set(w["tiles"]) & act_ids
            start = {(o[0], o[1]): o[2] for o in st0[0] if o[2] in tiles}
            if clicky:
                st = self._scramble(lvl, st0, max(2, nk), rng)
            else:
                st = st0
                for _ in range(int(3 + i // 2 + rng.integers(0, 3)) * 4):
                    mv = self.moves(lvl, st)
                    s2, dead = self.apply(lvl, st, *mv[int(rng.integers(len(mv)))])
                    if not dead:
                        st = s2
            tgt_map = {(o[0], o[1]): o[2] for o in st[0] if o[2] in tiles}
            if tgt_map == start or set(tgt_map) != set(start):
                return self._fail("target_eq_start")
            w["target"] = tgt_map
        self._build_panel(lvl, i, st0)
        if max(H, W) * t + lvl["ph"] > lim:
            return self._fail("panel_too_big")
        return lvl, st0, None

    @staticmethod
    def _fail(why):
        STATS["layout_" + why] += 1
        return None

    def _scramble(self, lvl, st, k, rng):
        """k clics au hasard (sans passer par le curseur) : niveaux à clic générés à rebours depuis un état but."""
        for _ in range(k):
            cells = sorted({(o[0], o[1]) for o in self.allo(lvl, st[0]) if lvl["clickable"].get(o[2])})
            if not cells:
                break
            sim = _Sim(self, lvl, st, None)
            sim.click_cell(*cells[int(rng.integers(len(cells)))])
            sim.dynamics()
            if not sim.dead:
                st = sim.state()
        return st

    def _build_panel(self, lvl, i, st0):
        """légende : panneau-cible (motif), icônes de règles (A -> B), compteur ; au-dessus d'un séparateur."""
        sp = self.spec
        w = lvl["win"]
        t = lvl["t"]
        it = max(1, min(t, 3))
        items = []
        if w["kind"] == "pattern":
            rs = [p[0] for p in w["target"]]
            cs = [p[1] for p in w["target"]]
            r0, c0 = min(rs), min(cs)
            ps = 1 if t <= 2 or (max(cs) - c0 + 1) * 2 > lvl["W"] * t // 2 else 2
            img = np.full(((max(rs) - r0 + 1) * ps, (max(cs) - c0 + 1) * ps), BG, np.int64)
            for (r, c), ty in w["target"].items():
                img[(r - r0) * ps:(r - r0 + 1) * ps, (c - c0) * ps:(c - c0 + 1) * ps] = self.col[ty]
            items.append(img)
        if sp["legend"]:
            for k in sp["hints"]:
                if k not in lvl["rules"]:
                    continue
                ru = sp["rules"][k]
                e = ru["effects"][0]
                src = ru["b"] if ru["b"] is not None else ru["a"]
                parts = [self.icon(src, it)]
                if e[0] in ("B_to", "A_to"):
                    src = ru["a"] if e[0] == "A_to" else ru["b"]
                    parts = [self.icon(src, it), self._arrow(it), self.icon(e[1], it)]
                elif e[0] == "toggle_all":
                    parts = [self.icon(ru["b"], it), self._arrow(it), self.icon(e[1], it), self.icon(e[2], it)]
                elif e[0] == "lose":
                    parts = [self.icon(src, it), np.full((it, 1), LIVES)]
                elif e[0] == "remove_B":
                    parts = [self.icon(src, it), self._arrow(it), np.full((it, it), BG)]
                elif e[0] == "grow":
                    parts = [self.icon(src, it), self._arrow(it), np.full((it, it), AVATAR)]
                img = np.concatenate([np.pad(p, ((0, it - p.shape[0]), (0, 0)), constant_values=BG) for p in parts]
                                     + [np.full((it, 1), BG)], 1)
                items.append(img)
        if sp["show_counter"] or (w["kind"] == "counter" and sp["legend"]):
            src = {ru["b"] for ru in sp["rules"] if any(e[0] == "counter_inc" for e in ru["effects"])}
            K = w.get("k", 0) if w["kind"] == "counter" else sum(1 for o in self.allo(lvl, st0[0]) if o[2] in src)
            lvl["counter_slots"] = max(1, min(9, K))
            items.append(np.full((it, 2 * lvl["counter_slots"]), BG))   # rempli au rendu (dynamique)
        lim = (62 // self.s_screen) if self.official else 60
        counter = items[-1:] if "counter_slots" in lvl else []
        rest = items[:-1] if counter else items
        while rest and sum(x.shape[1] + 1 for x in rest + counter) > lim:   # panneau trop large : on retire des icônes
            rest = rest[:-1]
        items = rest + counter
        if counter and sum(x.shape[1] + 1 for x in items) > lim:
            items = rest
            del lvl["counter_slots"]
        if not items:
            lvl["ph"], lvl["items"] = 0, []
            return
        h = max(x.shape[0] for x in items)
        lvl["items"] = items
        lvl["ph"] = h + 2

    def icon(self, tid, it):
        if tid is None:
            return np.full((it, it), BG)
        g = np.full((it, it), BG, np.int64)
        m = self.mask(tid, it)
        g[m == 1] = self.col[tid]
        g[m == 2] = DETAIL
        return g

    @staticmethod
    def _arrow(it):
        a = np.full((it, max(1, it - 1) if it > 1 else 1), BG, np.int64)
        a[it // 2, :] = DETAIL
        return np.pad(a, ((0, 0), (1, 1)), constant_values=BG)

    def mask(self, tid, t, style=None):
        style = style or self.sty[tid]
        if style.startswith("custom:"):
            return custom_mask(int(style.split(":")[1]), t)
        return sprite(t, style)

    # ------------------------------------------------------------------ fabrication / vérification d'un niveau
    time_budget = 5.0          # s par jeu : une spec plus lente est rejetée (« slow »)

    def _make_level(self, i, n):
        self._shrink = 0
        for attempt in range(self.max_tries):
            if time.time() - self._t0 > self.time_budget:
                STATS["lvl_slow"] += 1
                raise Unsolvable(f"{self.game_id}: trop lent")
            self._attempt = attempt
            out = self.gen_level(self.rng, i, n)
            if out is None:
                STATS["lvl_layout_fail"] += 1
                continue
            lvl, st0, _ = out
            if self.is_win(lvl, st0):
                STATS["lvl_won_at_start"] += 1
                continue
            mv = self.moves(lvl, st0)
            if not any(self.apply(lvl, st0, *m)[0] != st0 for m in mv):
                STATS["lvl_degenerate"] += 1
                continue
            sol, why = self._bfs(lvl, st0)
            if sol is None:
                STATS["lvl_" + why] += 1
                if why == "bfs_cap":
                    self._shrink += 1                     # trop d'états : on rétrécit les essais suivants
                continue
            if len(sol) <= 2:
                STATS["lvl_trivial"] += 1
                continue
            need = self.min_len(i, n) - attempt // 4
            if len(sol) < need:
                STATS["lvl_too_short"] += 1
                continue
            if lvl["new_rules"] and attempt < self.max_tries - 2:
                fired = self.fired_rules(lvl, st0, sol)
                if not fired & lvl["new_rules"]:
                    STATS["lvl_module_unused"] += 1
                    continue
            base = len(sol)
            budget = int(np.ceil(base * self.energy_mult)) + 6 if self.hud else None
            STATS["lvl_ok"] += 1
            return dict(lvl=lvl, st0=st0, sol=sol, baseline=base, budget=budget)
        raise Unsolvable(f"{self.game_id}: niveau {i} impossible à générer")

    def _bfs(self, lvl, st0):
        """BFS exact (plus court chemin) plafonné : (solution, raison de l'échec)."""
        prev = {st0: None}
        q = deque([st0])
        while q:
            st = q.popleft()
            for mv in self.moves(lvl, st):
                st2, dead = self.apply(lvl, st, *mv)
                if dead or st2 in prev:
                    continue
                prev[st2] = (st, mv)
                if self.is_win(lvl, st2):
                    path = []
                    s = st2
                    while prev[s] is not None:
                        p, m = prev[s]
                        path.append(m)
                        s = p
                    return path[::-1], "ok"
                q.append(st2)
            if len(prev) > self.bfs_cap:
                return None, "bfs_cap"
        return None, "unsolvable"

    def fired_rules(self, lvl, st0, sol):
        fired, st = set(), st0
        for op, px in sol:
            log = []
            st, _ = self._apply(lvl, st, op, px, log)
            fired.update(x[1] for x in log if x[0] == "rule")
        return fired

    def min_len(self, i, n):
        if self.scheme in ("click",):
            return 3
        if self.scheme == "cursor":
            return 4 + i
        return 4 + i

    # ------------------------------------------------------------------ dynamique d'un tour
    def apply(self, lvl, st, op, px):
        return self._apply(lvl, st, op, px, None)

    def _apply(self, lvl, st, op, px, log):
        if self.fast and log is None and op in DIDX:
            objs, V = st
            me = None
            for o in objs:
                if self.ctrl[o[2]]:
                    if me is not None:
                        me = None
                        break
                    me = o
            if me is not None:
                dr, dc = DV[DIDX[op]]
                r, c = me[0] + dr, me[1] + dc
                if not (0 <= r < lvl["H"] and 0 <= c < lvl["W"]) or (r, c) in lvl["wallset"]:
                    return st, False
                if (r, c) not in lvl["socc"] and not any(o[0] == r and o[1] == c for o in objs):
                    return (tuple(sorted([o for o in objs if o is not me] + [(r, c, me[2], me[3])])), V), False
        sim = _Sim(self, lvl, st, log)
        V = sim.V
        if op in DIDX:
            d = DIDX[op]
            if self.scheme == "cursor":
                r, c = V[SR] + DV[d][0], V[SC] + DV[d][1]
                if 0 <= r < lvl["H"] and 0 <= c < lvl["W"]:
                    V[SR], V[SC] = r, c
                return sim.state(), False
            if self.scheme == "select":
                movers = [k for k in sim.at(V[SR], V[SC]) if k >= 0 and self.ctrl[sim.O[k][2]]] if V[SR] >= 0 else []
                if not movers:
                    return st, False
                k = movers[0]
                sim.move_actor(k, d)
                if sim.alive[k]:
                    V[SR], V[SC] = sim.O[k][0], sim.O[k][1]
                else:
                    V[SR], V[SC] = -1, -1
            else:
                O = sim.O
                movers = [k for k in range(len(O)) if self.ctrl[O[k][2]]]
                dirs = {k: (d if self.ctrl[O[k][2]] == 1 else self.cmap[d]) for k in movers}
                movers.sort(key=lambda k: -(O[k][0] * DV[dirs[k]][0] + O[k][1] * DV[dirs[k]][1]))
                for k in movers:
                    if sim.alive[k]:
                        sim.move_actor(k, dirs[k])
        elif op == "interact":
            role = self.spec["a5"]
            if role == "interact":
                O = sim.O
                for i in [k for k in range(len(O)) if self.ctrl[O[k][2]]]:
                    if not sim.alive[i]:
                        continue
                    r, c = O[i][0], O[i][1]
                    for dr, dc in ((0, 0),) + DV:
                        for j in sim.at(r + dr, c + dc):
                            if j != i:
                                sim.fire("interact", i, j)
            elif role == "grow":
                V[BIG] ^= 1
                sim.L("eff", "grow")
            elif role == "gflip":
                V[GDIR] = OPPO[V[GDIR]]
                sim.L("eff", "gflip")
            elif role == "grot":
                V[GDIR] = ROT[V[GDIR]]
                sim.L("eff", "grot")
            elif role == "gate":
                V[GATE] ^= 1
                sim.L("eff", "gate")
            elif role == "commit":
                if self._cond(lvl, sim.state()[0], V):
                    V[COM] = 1
                else:
                    sim.dead = True
                return sim.state(), sim.dead
            elif role == "activate":
                if not sim.click_cell(V[SR], V[SC]):
                    return st, False
        elif op == "click":
            if px is None:
                return st, False
            r, c = (px[0] - lvl["ph"]) // lvl["t"], px[1] // lvl["t"]
            if px[0] < lvl["ph"] or not (0 <= r < lvl["H"] and 0 <= c < lvl["W"]):
                return st, False
            if self.scheme == "select" and any(k >= 0 and self.ctrl[sim.O[k][2]] for k in sim.at(r, c)):
                if (V[SR], V[SC]) == (r, c):
                    return st, False
                V[SR], V[SC] = r, c                      # sélection : le temps ne passe pas
                sim.L("select")
                return sim.state(), False
            if not sim.click_cell(r, c):
                return st, False
        sim.dynamics()
        sim.adjacency()
        if self.scheme in ("avatar", "multi", "coupled") and not any(
                sim.alive[k] and self.ctrl[o[2]] for k, o in enumerate(sim.O)):
            sim.dead = True
        return sim.state(), sim.dead

    # ------------------------------------------------------------------ victoire
    @staticmethod
    def allo(lvl, objs):
        """objets dynamiques (état) + statiques (niveau)."""
        return objs + lvl["static"] if lvl["static"] else objs

    def _cond(self, lvl, objs, V):
        objs = objs + lvl["static"] if lvl["static"] else objs
        w = lvl["win"]
        kind = w["kind"]
        ok = True
        if kind in ("reach", "all_on"):
            Xs, Y = w["Xs"], w["Y"]
            ycells = {(o[0], o[1]) for o in objs if o[2] == Y}
            xs = [(o[0], o[1]) for o in objs if o[2] in Xs]
            if kind == "reach":
                ok = any(p in ycells for p in xs)
            else:
                ok = bool(xs) and all(p in ycells for p in xs)
        elif kind == "none_left":
            Xs = w["Xs"]
            ok = not any(o[2] in Xs for o in objs)
        elif kind == "counter":
            ok = V[CNT] >= w["k"]
        elif kind == "pattern":
            ts = w["tileset"]
            cur = {(o[0], o[1]): o[2] for o in objs if o[2] in ts}
            ok = w["target"] is not None and cur == w["target"]
        elif kind == "same":
            ts = w["tileset"]
            ty = {o[2] for o in objs if o[2] in ts}
            ok = len(ty) == 1 and sum(1 for o in objs if o[2] in ts) >= 2
        if ok and w["and_reach"] is not None:
            E = w["and_reach"]
            ec = {(o[0], o[1]) for o in objs if o[2] == E}
            ok = any((o[0], o[1]) in ec for o in objs if self.ctrl[o[2]])
        return ok

    def is_win(self, lvl, st):
        objs, V = st
        if lvl["win"]["commit"]:
            return V[COM] == 1
        return self._cond(lvl, objs, V)

    # ------------------------------------------------------------------ clics (BFS / oracle)
    def _cell_px(self, lvl, r, c):
        t = lvl["t"]
        return (lvl["ph"] + r * t + t // 2, c * t + t // 2)

    def click_targets(self, lvl, st):
        if not self.uses_click:
            return []
        cl = lvl["clickable"]
        cells = sorted({(o[0], o[1]) for o in self.allo(lvl, st[0]) if cl.get(o[2])})
        return [self._cell_px(lvl, r, c) for r, c in cells]

    # ------------------------------------------------------------------ rendu
    def render(self, lvl, st):
        objs, V = st
        objs = self.allo(lvl, objs)
        t, H, W = lvl["t"], lvl["H"], lvl["W"]
        cv = Canvas(H, W, t)
        walls = lvl["walls"]
        for r, c in lvl["wallset"]:
            if self.wall_style == "thin":
                cv.thin_wall(walls, r, c)
            else:
                cv.put(r, c, WALL, self.wall_style)
        g = cv.g

        def layer(o):
            ty = o[2]
            if self.is_tile[ty]:
                return -1
            if self.ctrl[ty]:
                return 4
            if self.motion[ty] in ("patrol", "chase"):
                return 3
            if self.solid[ty] != PASS or self.gated[ty] is not None:
                return 2
            return 1 if self.motion[ty] != "conveyor" else 0
        for o in sorted(objs, key=layer):
            r, c, ty, a = o
            blk = g[r * t:(r + 1) * t, c * t:(c + 1) * t]
            if self.motion[ty] == "conveyor":
                m = arrow_mask(t, a)
            elif self.gated[ty] is not None and V[GATE] == self.gated[ty]:
                m = sprite(t, "corners")                   # porte ouverte : coins
            elif self.grate[ty]:
                m = sprite(t, "cross" if t >= 3 else "full")
            elif self.is_tile[ty] and t >= 2:                # tuiles séparées d'1 px (sinon elles fusionnent)
                m = np.ones((t, t), np.int8)
                m[-1, :] = m[:, -1] = 0
            elif self.ctrl[ty] and self.big_used:
                m = sprite(t, "full") if V[BIG] else sprite(t, "dot" if t < 3 else "small_dot")
            else:
                m = self.mask(ty, t)
            col = self.col[ty]
            if self.scheme == "select" and (r, c) == (V[SR], V[SC]) and self.ctrl[ty]:
                if getattr(self, "sel_style", "white") == "white":
                    col = MARK
                else:
                    blk[t // 2, t // 2] = MARK
            blk[m == 1] = col
            if t >= 3:
                blk[m == 2] = DETAIL
            if self.scheme == "select" and (r, c) == (V[SR], V[SC]) and self.ctrl[ty] and col != MARK:
                blk[t // 2, t // 2] = MARK
        if self.scheme == "cursor":
            r, c = V[SR], V[SC]
            blk = g[r * t:(r + 1) * t, c * t:(c + 1) * t]
            if t >= 3:
                blk[0, :] = blk[-1, :] = blk[:, 0] = blk[:, -1] = MARK
            else:
                blk[0, 0] = MARK
        if lvl["ph"]:
            pw = sum(x.shape[1] + 1 for x in lvl["items"])
            width = max(g.shape[1], pw)
            pan = np.full((lvl["ph"], width), BG, np.int64)
            x = 0
            for k, img in enumerate(lvl["items"]):
                img = img.copy()
                if k == len(lvl["items"]) - 1 and "counter_slots" in lvl:
                    img[:] = BG
                    for s_ in range(lvl["counter_slots"]):
                        img[:, 2 * s_] = HUD_FILL if s_ < V[CNT] else PANEL
                pan[0:img.shape[0], x:x + img.shape[1]] = img
                x += img.shape[1] + 1
            pan[-1, :] = PANEL
            if g.shape[1] < width:
                g = np.concatenate([g, np.full((g.shape[0], width - g.shape[1]), BG)], 1)
            g = np.vstack([pan, g])
        return g

    # ------------------------------------------------------------------ événements privilégiés
    def body(self, lvl, st):
        objs, V = st
        if self.scheme == "cursor":
            return (V[SR], V[SC], lvl["H"], lvl["W"])
        if self.scheme == "select":
            return (V[SR], V[SC], lvl["H"], lvl["W"]) if V[SR] >= 0 else None
        for o in objs:
            if self.ctrl[o[2]]:
                return (o[0], o[1], lvl["H"], lvl["W"])
        return None

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        log = []
        self._apply(lvl, st, op, px, log)
        for x in log:
            k = x[0]
            if k == "push":
                ev[EV["push"]] = ev[EV["touch"]] = 1
            elif k == "bumpobj":
                ev[EV["touch"]] = 1
            elif k == "rule" and x[2] in ("enter", "bump", "adjacent", "interact", "push"):
                ev[EV["touch"]] = 1
            elif k == "teleport":
                ev[EV["teleport"]] = 1
            elif k == "open":
                ev[EV["open"]] = 1
            elif k == "select":
                ev[EV["toggle"]] = 1
            elif k == "eff":
                n = x[1]
                if n in ("remove_B", "counter_inc"):
                    ev[EV["collect" if op != "click" else "toggle"]] = 1
                elif n in ("B_to", "A_to", "toggle_all", "flip_nbrs", "flip_mirror", "spawn"):
                    ev[EV["color"]] = 1
                elif n in ("gate",):
                    ev[EV["open"]] = ev[EV["toggle"]] = 1
                elif n in ("grow", "gflip", "grot"):
                    ev[EV["toggle"]] = 1
                elif n == "lose":
                    ev[EV["danger"]] = 1
        if dead:
            ev[EV["danger"]] = 1
        elif self.lose_types:
            objs = self.allo(lvl, st2[0])
            av = [(o[0], o[1]) for o in objs if self.ctrl[o[2]]]
            hz = [(o[0], o[1]) for o in objs if o[2] in self.lose_types]
            if any(abs(a[0] - h[0]) + abs(a[1] - h[1]) <= 1 for a in av for h in hz):
                ev[EV["danger"]] = 1

    def heuristic(self, lvl, st):
        """distance au but approchée (quand la table exacte n'existe pas)."""
        objs, V = st
        objs = self.allo(lvl, objs)
        w = lvl["win"]
        kind = w["kind"]

        def md(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        av = [(o[0], o[1]) for o in objs if self.ctrl[o[2]]] or [(V[SR], V[SC])]
        h = 0.0
        if kind in ("reach", "all_on"):
            ys = [(o[0], o[1]) for o in objs if o[2] == w["Y"]]
            xs = [(o[0], o[1]) for o in objs if o[2] in w["Xs"]]
            if ys and xs:
                ds = [min(md(x, y) for y in ys) for x in xs]
                h = min(ds) if kind == "reach" else sum(ds)
        elif kind == "none_left":
            xs = [(o[0], o[1]) for o in objs if o[2] in w["Xs"]]
            h = 2 * len(xs) + (min(md(a, x) for a in av for x in xs) if xs and av[0][0] >= 0 else 0)
        elif kind == "counter":
            src = [(o[0], o[1]) for o in objs if o[2] == w["src"]]
            h = 2 * max(0, w["k"] - V[CNT]) + (min(md(a, x) for a in av for x in src) if src and av[0][0] >= 0 else 0)
        elif kind == "pattern":
            cur = {(o[0], o[1]): o[2] for o in objs if o[2] in w["tileset"]}
            h = sum(1 for p, ty in w["target"].items() if cur.get(p) != ty)
        elif kind == "same":
            ty = Counter(o[2] for o in objs if o[2] in w["tileset"])
            h = sum(ty.values()) - max(ty.values()) if ty else 0
        if w["and_reach"] is not None:
            ec = [(o[0], o[1]) for o in objs if o[2] == w["and_reach"]]
            if ec and av[0][0] >= 0:
                h += min(md(a, e) for a in av for e in ec)
        return float(h)

    def info(self):
        d = super().info()
        d.update(rules=describe(self.spec), scheme=self.scheme, attempt=self.attempt)
        return d


# ============================================================================ fabrique
def make_rule_game(seed, look="classic", max_attempts=60):
    """construit le jeu à règles de graine `seed` : on ré-échantillonne la spec (attempt) tant qu'elle est rejetée."""
    for attempt in range(max_attempts):
        STATS["spec_sampled"] += 1
        try:
            g = RuleGame(seed, look, attempt=attempt)
        except Unsolvable:
            STATS["spec_rejected"] += 1
            continue
        STATS["spec_accepted"] += 1
        return g
    raise Unsolvable(f"rules-{seed}: aucune spec acceptée en {max_attempts} essais")


def rule_signature(game):
    return signature(game.spec)
