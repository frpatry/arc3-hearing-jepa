"""Noyau du générateur de jeux « type ARC-AGI-3 ».

Un jeu = une FAMILLE de mécanique (sous-classe de Game) + des paramètres tirés au hasard (graine) :
  - niveaux procéduraux de difficulté croissante, chacun VÉRIFIÉ soluble (BFS sur l'état, ou par construction),
    avec `baseline_actions` = longueur de la meilleure solution trouvée (l'équivalent de la baseline humaine) ;
  - présentation : grille logique (couleurs « sémantiques » 0..15) agrandie d'un facteur entier dans l'écran 64×64,
    letterbox, barre d'énergie (HUD) optionnelle, palette permutée par jeu ;
  - actions : ids officiels (RESET=0, 1-5 et 7 simples, 6 = clic {x, y} écran) avec une permutation aléatoire du sens
    (quel id veut dire « haut »…), des actions inutiles ajoutées à `available_actions`, undo (ACTION7) optionnel.

API publique (miroir de l'officielle) :
    g.reset() -> obs        g.step(action_id, data=None) -> obs
    obs = {frame: np.ndarray(64, 64) int8, state: "NOT_FINISHED"|"WIN"|"GAME_OVER",
           levels_completed, win_levels, available_actions}
Sémantique RESET (comme la passerelle Kaggle) : après GAME_OVER ou en cours de niveau -> redémarre le niveau courant
(progression gardée) ; RESET alors que le niveau n'a encore aucune action -> sans effet ; après WIN -> retour au niveau 1.
Après GAME_OVER / WIN, les actions autres que RESET ne font rien (l'officiel renvoie une image vide).

Une famille implémente (fonctions PURES de (lvl, st), st hashable) :
    setup(rng)                      paramètres de la variante (tuile, style, …)
    gen_level(rng, i, n) -> (lvl, st0, sol|None)   sol = [(op, px)] si soluble par construction
    apply(lvl, st, op, px) -> (st2, dead)           op ∈ up/down/left/right/interact/click ; px = pixel LOGIQUE (r, c)
    is_win(lvl, st) -> bool
    render(lvl, st) -> np.ndarray (h, w) de couleurs sémantiques
    click_targets(lvl, st) -> [px]  (familles à clic : cibles pour le BFS)
    min_len(i, n) -> int            longueur minimale exigée d'une solution (difficulté)
"""
import zlib
from collections import deque

import numpy as np

DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
DIR_OPS = ("up", "down", "left", "right")
CANON = {"up": 1, "down": 2, "left": 3, "right": 4, "interact": 5, "undo": 7}

# couleurs sémantiques (remappées par la palette de chaque jeu)
BG, WALL, AVATAR, GOAL = 0, 1, 2, 3          # 4..12 : objets propres à chaque famille
MARK = 13                                     # surbrillance / sélection
HUD_FILL, LETTERBOX = 14, 15
DETAIL, PANEL, LIVES = 16, 17, 18             # détail interne des sprites, séparateurs de panneau, vies
OBJ2 = (19, 20, 21, 22, 23, 24)               # objets supplémentaires (familles à nombreux modules)
NSEM = 25
GREYS = [0, 1, 2, 3, 4, 5]                    # gris de la palette officielle (blanc -> noir)
BLACK, WHITE = 5, 0
# usage officiel mesuré : décor noir / gris 3-4 / bleus 9-10 ; objets rares en 8, 11, 12, 14, 15, 6
OFF_BG = [5] * 12 + [4, 4, 3, 3, 9, 10, 0, 2]          # noir ≈ 60 %
OFF_WALL = [3, 4, 2, 1, 9, 10]
OFF_OBJ = [8, 11, 12, 14, 15, 6, 9, 10, 7, 13, 0, 1]


def official_palette(r):
    pal = np.zeros(NSEM, np.int64)
    bg = int(r.choice(OFF_BG))
    wall = int(r.choice([c for c in OFF_WALL if c != bg]))
    objs = [c for c in OFF_OBJ if c not in (bg, wall)]
    objs = list(r.permutation(objs[:6])) + list(r.permutation(objs[6:]))      # couleurs rares d'abord
    frame = BLACK if (bg != BLACK and r.random() < 0.7) else int(r.choice([c for c in (4, 3, 2) if c != bg]))
    pal[BG], pal[WALL], pal[LETTERBOX] = bg, wall, bg if r.random() < 0.5 else frame   # aire de jeu ≠ pourtour
    sems = [AVATAR, GOAL] + list(range(4, 13)) + list(OBJ2)
    for k, sem in enumerate(sems):
        pal[sem] = objs[k % len(objs)]
    pal[MARK] = WHITE if bg != WHITE else 11
    pal[DETAIL] = WHITE if bg == BLACK and r.random() < 0.6 else BLACK if bg != BLACK else 11
    if pal[DETAIL] == pal[AVATAR]:
        pal[DETAIL] = WHITE if pal[AVATAR] != WHITE else BLACK
    pal[HUD_FILL] = int(r.choice([c for c in (11, 9, 14, 12, 0, 3, 8) if c != bg]))
    pal[PANEL] = wall
    pal[LIVES] = 8 if bg != 8 else 12
    return pal


INF = float("inf")

# vecteur d'événements PRIVILÉGIÉS (cibles auxiliaires d'entraînement, absents des vrais jeux) — même disposition partout
EVENT_NAMES = (
    "bump", "push", "touch", "click_obj", "click_empty",                       # 0-4   TOUCHER
    "pos_r0", "pos_c0", "pos_r1", "pos_c1", "moved",                           # 5-9   CORPS (avant/après, [0,1] ou -1)
    "collect", "open", "color", "toggle", "teleport", "danger", "energy_low",  # 10-16 SON
    "level_complete", "game_over", "reset", "undo", "noop",                    # 17-21 SON (suite)
    "progress",                                                                # 22    PROGRÈS [-1, 1]
)
EV = {n: i for i, n in enumerate(EVENT_NAMES)}
N_EVENTS = len(EVENT_NAMES)


class Unsolvable(RuntimeError):
    pass


class Game:
    family = "base"
    simple_ops = DIR_OPS          # opérations sémantiques des actions simples
    uses_click = False            # peut être redéfini par instance dans setup()
    n_levels = (4, 7)
    bfs_cap = 30000
    max_tries = 80

    # ------------------------------------------------------------------ construction
    n_levels_official = (6, 10)

    def __init__(self, seed=0, look="classic"):
        """look = "classic" (v1 : aplats agrandis) ou "official" (v2 : rendu natif façon ARC-AGI-3, barre d'1 px…)."""
        self.seed = int(seed)
        self.look = look
        self.official = look == "official"
        self.game_id = f"{self.family}-{'o' if self.official else ''}{self.seed:04d}"
        self.rng = np.random.default_rng([zlib.crc32(self.family.encode()), self.seed])
        self.orng = np.random.default_rng([zlib.crc32(self.family.encode()), self.seed, 7])   # tirages propres au look v2
        rng = self.rng
        self._rcache = {}
        self.max_cells, self.px_budget = 99, 58
        if self.official:
            self._official_view(self.orng)
        self.setup(rng)
        self._presentation(rng)
        if self.official:
            self._official_presentation(self.orng)
        lo, hi = self.n_levels_official if self.official else self.n_levels
        n = int(rng.integers(lo, hi + 1))
        self.levels = []
        for i in range(n):
            L = self._make_level(i, n)
            if L is None:                                  # (v3) niveau tardif introuvable : jeu tronqué
                break
            self.levels.append(L)
        n = len(self.levels)
        self.baseline_actions = [L["baseline"] for L in self.levels]
        self.win_levels = n
        self._full_reset()

    def setup(self, rng):
        pass

    def cap(self, n):
        return int(min(n, self.max_cells))

    def _official_view(self, r):
        """échelle native : ≈65 % ×1, 15 % ×2, 15 % ×3, 5 % ×4 ; tuiles de 3–6 px à ×1 (sprites détaillés)."""
        s = int(r.choice([1, 2, 3, 4], p=[0.65, 0.15, 0.15, 0.05]))
        T = int(r.choice({1: [3, 3, 4, 4, 5], 2: [2, 3, 3], 3: [1, 2], 4: [1, 1]}[s]))
        self.s_screen, self.T = s, T
        self.bar_edge = str(r.choice(["top", "bottom", "left", "right"]))
        self.energy_death = bool(r.random() < 0.7)
        self.lives_max = int(r.choice([0, 0, 0, 0, 3, 3, 5]))
        self.panel = bool(r.random() < 0.5)
        self.wall_mode = "thin" if r.random() < 0.6 else "block"
        self.sel_style = str(r.choice(["white", "dot"]))
        self.max_cells = max(5, (62 // s) // T)
        self.px_budget = 62 // s - 2

    def _official_presentation(self, r):
        self.palette = official_palette(r)
        self.hud = self.bar_edge                     # barre de pas d'1 px dans 100 % des jeux
        self.hud_h = 1

    def _presentation(self, rng):
        # palette : fond / murs / letterbox plutôt dans les gris (comme les jeux officiels), objets dans le reste
        perm = np.empty(16, np.int64)
        if rng.random() < 0.75:
            g = rng.permutation(GREYS)[:3]
            if rng.random() < 0.35 and BLACK not in g[:1]:      # fond noir plus fréquent (usage officiel)
                g = np.array([BLACK] + [c for c in g if c != BLACK][:2])
            if rng.random() < 0.4:
                g = np.array([g[0], g[1], BLACK if BLACK not in g[:2] else g[2]])   # pourtour noir fréquent
            elif rng.random() < 0.5:
                g = np.array([g[0], g[1], g[0]])                                     # pourtour = fond
            perm[[BG, WALL, LETTERBOX]] = g
        else:
            perm[[BG, WALL, LETTERBOX]] = rng.permutation(16)[:3]
        rest = [c for c in rng.permutation(16) if c not in perm[[BG, WALL, LETTERBOX]]]
        for k, sem in enumerate([s for s in range(16) if s not in (BG, WALL, LETTERBOX)]):
            perm[sem] = rest[k]
        self.palette = np.concatenate([perm, [perm[MARK], perm[WALL], 8], perm[4:10]])   # DETAIL, PANEL, LIVES, OBJ2
        # HUD énergie
        self.hud = rng.choice(["bottom", "top", None], p=[0.5, 0.2, 0.3])
        self.hud_h = int(rng.integers(1, 3)) if self.hud else 0
        self.energy_mult = float(rng.uniform(2.5, 5.0))
        # actions : sens des ids
        ops = list(self.simple_ops)
        self.undo = bool(rng.random() < 0.3)
        if self.undo:
            ops.append("undo")
        if rng.random() < 0.5:
            amap = {CANON[o]: o for o in ops}
        else:
            ids = [i for i in (1, 2, 3, 4, 5, 7)]
            if self.undo and rng.random() < 0.7:
                ids.remove(7)
                amap = {7: "undo"}
                ops = [o for o in ops if o != "undo"]
            else:
                amap = {}
            for o, i in zip(ops, rng.permutation(ids)):
                amap[int(i)] = o
        if self.uses_click:
            amap[6] = "click"
        unused = [i for i in range(1, 8) if i not in amap]
        n_extra = int(rng.choice([0, 1, 2], p=[0.35, 0.4, 0.25]))
        for i in rng.permutation(unused)[:n_extra]:
            amap[int(i)] = None                    # action déclarée mais inutile
        self.action_map = amap
        self.available_actions = sorted(amap)
        self.op_to_id = {o: i for i, o in amap.items() if o is not None}

    def _make_level(self, i, n):
        for attempt in range(self.max_tries):
            out = self.gen_level(self.rng, i, n)
            if out is None:
                continue
            lvl, st0, sol = out
            if sol is None or (len(sol) <= 400 and self.bfs_improve):
                s2 = self.solve(lvl, st0)
                if s2 is not None and (sol is None or len(s2) < len(sol)):
                    sol = s2
            if sol:                                                 # couper la solution au 1er état gagnant
                st = st0
                for k, mv in enumerate(sol):
                    st, _ = self.apply(lvl, st, *mv)
                    if self.is_win(lvl, st):
                        sol = sol[:k + 1]
                        break
            ml = self.min_len(i, n)
            need = max(1, ml - max(attempt // 20, (ml * attempt) // self.max_tries))   # on relâche si ça bloque
            if sol is None or len(sol) < need:
                continue
            base = len(sol)
            budget = int(np.ceil(base * self.energy_mult)) + 6 if self.hud else None
            return dict(lvl=lvl, st0=st0, sol=sol, baseline=base, budget=budget)
        raise Unsolvable(f"{self.game_id}: niveau {i} impossible à générer")

    bfs_improve = False      # True : tenter d'améliorer une solution par construction avec le BFS

    # ------------------------------------------------------------------ solveur générique
    def moves(self, lvl, st):
        mv = [(o, None) for o in self.simple_ops]
        if self.uses_click:
            mv += [("click", p) for p in self.click_targets(lvl, st)]
        return mv

    def solve(self, lvl, st0, cap=None):
        """BFS (plus court chemin en nombre d'actions) ; None si pas trouvé sous le plafond d'états."""
        cap = cap or self.bfs_cap
        if self.is_win(lvl, st0):
            return []
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
                    return path[::-1]
                q.append(st2)
            if len(prev) > cap:
                return None
        return None

    def click_targets(self, lvl, st):
        return []

    # ------------------------------------------------------------------ rendu
    def _layout(self, shape):
        h, w = shape
        if self.official:
            e = self.bar_edge
            y0, x0 = (2 if e == "top" else 0), (2 if e == "left" else 0)
            ah = 64 - (2 if e in ("top", "bottom") else 0)
            aw = 64 - (2 if e in ("left", "right") else 0)
            s = max(1, min(self.s_screen, ah // h, aw // w))
            return s, y0 + (ah - h * s) // 2, x0 + (aw - w * s) // 2
        avail_h = 64 - self.hud_h - (1 if self.hud else 0)
        s = max(1, min(avail_h // h, 64 // w))
        oy = (avail_h - h * s) // 2 + (self.hud_h + 1 if self.hud == "top" else 0)
        ox = (64 - w * s) // 2
        return s, oy, ox

    def _render(self, li, st):
        key = (li, st)
        g = self._rcache.get(key)
        if g is None:
            if len(self._rcache) > 256:
                self._rcache.clear()
            g = self._rcache[key] = np.asarray(self.render(self.levels[li]["lvl"], st))
        return g

    def frame(self):
        L = self.levels[self.level]
        g = self._render(self.level, self.st)
        s, oy, ox = self._layout(g.shape)
        lb = LETTERBOX
        if self.official and g.shape[0] * g.shape[1] * s * s < 0.4 * 4096 and self.palette[BG] != BLACK:
            lb = DETAIL if self.palette[DETAIL] == BLACK else LETTERBOX     # petite aire de jeu : pourtour noir
        scr = np.full((64, 64), lb, np.int64)
        big = np.repeat(np.repeat(g, s, 0), s, 1)
        scr[oy:oy + big.shape[0], ox:ox + big.shape[1]] = big
        if self.official:
            self._draw_bar(scr, L)
        elif self.hud:
            rows = slice(64 - self.hud_h, 64) if self.hud == "bottom" else slice(0, self.hud_h)
            frac = max(0, self.energy) / L["budget"]
            n = int(np.ceil(64 * frac))
            scr[rows, :] = LETTERBOX
            scr[rows, :n] = HUD_FILL
        return self.palette[scr].astype(np.int8)

    def _draw_bar(self, scr, L):
        """barre de pas d'1 px sur un bord (1 px par action tant que le budget ≤ longueur), vies en bout de barre."""
        e = self.bar_edge
        line = scr[0, :] if e == "top" else scr[63, :] if e == "bottom" else scr[:, 0] if e == "left" else scr[:, 63]
        line[:] = LETTERBOX
        nl = self.lives_max
        length = 64 - (3 * nl + 1 if nl else 0)
        B = L["budget"]
        left = max(0, self.energy)
        n = left if B <= length else int(np.ceil(length * left / B))
        line[:n] = HUD_FILL
        for k in range(nl):
            if k < self.lives:
                line[63 - 3 * k - 1:63 - 3 * k + 1] = LIVES

    def screen_to_px(self, data):
        if not data:
            return None
        g_shape = self._render(self.level, self.st).shape
        s, oy, ox = self._layout(g_shape)
        x, y = int(data.get("x", -1)), int(data.get("y", -1))
        r, c = (y - oy) // s, (x - ox) // s
        if y < oy or x < ox or not (0 <= r < g_shape[0] and 0 <= c < g_shape[1]):
            return None
        return (int(r), int(c))

    def px_to_screen(self, level_idx, px, st=None):
        L = self.levels[level_idx]
        shape = self._render(level_idx, L["st0"] if st is None else st).shape
        s, oy, ox = self._layout(shape)
        return {"x": int(ox + px[1] * s + s // 2), "y": int(oy + px[0] * s + s // 2)}

    # ------------------------------------------------------------------ événements privilégiés (cibles auxiliaires)
    exact_progress = True        # table exacte distance-au-but (exploration complète, plafonnée) sinon heuristique
    table_cap = 20000
    progress_scale = 1.0

    def body(self, lvl, st):
        """objet contrôlé -> (r, c, H, W) en cases logiques du jeu, ou None (jeux à clic)."""
        return None

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        pass

    def heuristic(self, lvl, st):
        return 0.0

    def table_ok(self, lvl, st0):
        """la table exacte est-elle raisonnable pour ce niveau ? (sinon heuristique)"""
        return True

    def click_hit(self, lvl, st, px):
        g = np.asarray(self.render(lvl, st))
        return int(g[px]) not in (BG, WALL)

    def _dist_table(self, L):
        lvl, st0 = L["lvl"], L["st0"]
        rev, seen, q, wins = {}, {st0}, deque([st0]), []
        while q:
            s = q.popleft()
            if self.is_win(lvl, s):
                wins.append(s)
                continue
            for mv in self.moves(lvl, s):
                s2, dead = self.apply(lvl, s, *mv)
                if dead or s2 == s:
                    continue
                rev.setdefault(s2, []).append(s)
                if s2 not in seen:
                    seen.add(s2)
                    q.append(s2)
                    if len(seen) > self.table_cap:
                        return None
        dist = dict.fromkeys(seen, INF)
        q = deque(wins)
        for w in wins:
            dist[w] = 0
        while q:
            s = q.popleft()
            for p in rev.get(s, ()):
                if dist[p] == INF:
                    dist[p] = dist[s] + 1
                    q.append(p)
        return dist

    def table(self, li):
        L = self.levels[li]
        if "dtab" not in L:
            L["dtab"] = self._dist_table(L) if (self.exact_progress and self.table_ok(L["lvl"], L["st0"])) else None
        return L["dtab"]

    def dist(self, li, st):
        """distance au but (en actions) : exacte si la table existe, sinon heuristique de la famille."""
        tab = self.table(li)
        if tab is not None and st in tab:
            return tab[st]
        return float(self.heuristic(self.levels[li]["lvl"], st))

    def _body_vec(self, li, st):
        b = self.body(self.levels[li]["lvl"], st)
        if b is None:
            return None
        r, c, H, W = b
        return (r / max(1, H - 1), c / max(1, W - 1))

    # ------------------------------------------------------------------ boucle de jeu
    def _start_level(self):
        L = self.levels[self.level]
        self.st = L["st0"]
        self.energy = L["budget"] if self.hud else None
        self.level_actions = 0
        self.history = []

    def _full_reset(self):
        self.lives = getattr(self, "lives_max", 0)
        self.level = 0
        self.levels_completed = 0
        self.state = "NOT_FINISHED"
        self.total_actions = 0
        self._start_level()

    def obs(self, events=None):
        if events is None:
            events = np.zeros(N_EVENTS, np.float32)
            events[EV["pos_r0"]:EV["pos_c1"] + 1] = -1
        return dict(frame=self.frame(), state=self.state, levels_completed=self.levels_completed,
                    win_levels=self.win_levels, available_actions=list(self.available_actions), events=events)

    def reset(self):
        return self.step(0)

    def step(self, action_id, data=None):
        """NB : après GAME_OVER l'officiel renvoie une liste d'images VIDE ; ici on renvoie la dernière image
        (figée) avec state = GAME_OVER. Les actions autres que RESET y sont sans effet (événement noop)."""
        ev = np.zeros(N_EVENTS, np.float32)
        ev[EV["pos_r0"]:EV["pos_c1"] + 1] = -1
        action_id = int(action_id)
        li, st0 = self.level, self.st
        b0 = self._body_vec(li, st0)
        if action_id == 0:
            ev[EV["reset"]] = 1
            if self.state == "WIN":
                self._full_reset()
            elif self.state == "GAME_OVER" or self.level_actions > 0:
                if self.state == "GAME_OVER":
                    self.lives = getattr(self, "lives_max", 0)
                self.state = "NOT_FINISHED"
                self._start_level()
            else:
                ev[EV["noop"]] = 1
            self._set_body(ev, b0, self._body_vec(self.level, self.st))
            return self.obs(ev)
        if self.state != "NOT_FINISHED" or action_id not in self.action_map:
            ev[EV["noop"]] = 1
            self._set_body(ev, b0, b0)
            return self.obs(ev)
        self.level_actions += 1
        self.total_actions += 1
        op = self.action_map[action_id]
        L = self.levels[li]
        dead = False
        if op == "undo":
            ev[EV["undo"]] = 1
            if self.history:
                self.st = self.history.pop()
        elif op is not None:
            px = self.screen_to_px(data) if op == "click" else None
            if op == "click":
                hit = px is not None and self.click_hit(L["lvl"], st0, px)
                ev[EV["click_obj" if hit else "click_empty"]] = 1
            if op != "click" or px is not None:
                st2, dead = self.apply(L["lvl"], st0, op, px)
                self.fam_events(L["lvl"], st0, op, px, st2, dead, ev)
                if op in DIRS and st2 == st0 and b0 is not None:
                    ev[EV["bump"]] = 1
                if st2 != st0:
                    self.history.append(st0)
                    self.st = st2
        st1 = self.st
        self._set_body(ev, b0, self._body_vec(li, st1))
        if st1 == st0:
            ev[EV["noop"]] = 1
        # progrès : variation de la distance au but (positive = on se rapproche)
        if dead:
            prog = -1.0
        else:
            d0, d1 = self.dist(li, st0), self.dist(li, st1)
            if d0 == INF and d1 == INF:
                prog = 0.0
            elif d1 == INF:
                prog = -1.0
            elif d0 == INF:
                prog = 1.0
            else:
                prog = float(np.clip((d0 - d1) / self.progress_scale, -1, 1))
        ev[EV["progress"]] = prog
        if self.hud:
            self.energy -= 1
            if self.energy <= 0.25 * L["budget"]:
                ev[EV["energy_low"]] = 1
        exhausted = self.hud and self.energy <= 0 and (not self.official or self.energy_death)
        if self.official and self.hud and self.energy <= 0 and not self.energy_death:
            self.energy = 0
        if not dead and self.is_win(L["lvl"], self.st):
            ev[EV["level_complete"]] = 1
            self.levels_completed += 1
            if self.level + 1 >= self.win_levels:
                self.state = "WIN"
            else:
                self.level += 1
                self._start_level()
        elif dead or exhausted:
            if getattr(self, "lives_max", 0) and self.lives > 1:
                self.lives -= 1                       # une vie perdue : le niveau redémarre tout seul
                ev[EV["reset"]] = 1
                self._start_level()
            else:
                ev[EV["game_over"]] = 1
                self.state = "GAME_OVER"
        return self.obs(ev)

    @staticmethod
    def _set_body(ev, b0, b1):
        if b0 is not None:
            ev[EV["pos_r0"]], ev[EV["pos_c0"]] = b0
        if b1 is not None:
            ev[EV["pos_r1"]], ev[EV["pos_c1"]] = b1
        if b0 is not None and b1 is not None and b0 != b1:
            ev[EV["moved"]] = 1

    # ------------------------------------------------------------------ oracle (politique « vers le but »)
    def _to_action(self, op, px, st):
        return self.op_to_id[op], (self.px_to_screen(self.level, px, st) if op == "click" else None)

    def oracle_action(self):
        """action optimale depuis l'état courant (table exacte) ; sinon suit la solution stockée si on est dessus,
        sinon RESET (gratuit) pour revenir au départ du niveau."""
        if self.state == "GAME_OVER":
            return 0, None
        if self.state == "WIN":
            return 0, None
        L = self.levels[self.level]
        lvl, st = L["lvl"], self.st
        tab = self.table(self.level)
        if tab is not None and tab.get(st, INF) < INF:
            d = tab[st]
            for op, px in self.moves(lvl, st):
                s2, dead = self.apply(lvl, st, op, px)
                if not dead and tab.get(s2, INF) == d - 1:
                    return self._to_action(op, px, st)
        if "sol_idx" not in L:
            idx, s = {}, L["st0"]
            for k, (op, px) in enumerate(L["sol"]):
                idx[s] = k                                   # dernière occurrence : saute les boucles
                s, _ = self.apply(lvl, s, op, px)
            L["sol_idx"] = idx
        k = L["sol_idx"].get(st)
        if k is not None:
            return self._to_action(*L["sol"][k], st)
        return 0, None

    # ------------------------------------------------------------------ outils
    def solution_actions(self, level_idx):
        """solution du niveau en (action_id, data) écran : pour l'oracle / les tests."""
        L = self.levels[level_idx]
        out, st = [], L["st0"]
        for op, px in L["sol"]:
            data = self.px_to_screen(level_idx, px, st) if op == "click" else None
            out.append((self.op_to_id[op], data))
            st, _ = self.apply(L["lvl"], st, op, px)
        return out

    def info(self):
        return dict(game_id=self.game_id, family=self.family, win_levels=self.win_levels,
                    baseline_actions=self.baseline_actions, available_actions=self.available_actions,
                    action_map={k: v for k, v in self.action_map.items()}, hud=self.hud,
                    budgets=[L["budget"] for L in self.levels])


# ====================================================================== aides de grille
def bfs_dist(walls, start):
    """distances 4-voisinage sur les cases libres (walls: bool array, True = mur)."""
    H, W = walls.shape
    dist = np.full((H, W), -1, np.int64)
    dist[start] = 0
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS.values():
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and not walls[rr, cc] and dist[rr, cc] < 0:
                dist[rr, cc] = dist[r, c] + 1
                q.append((rr, cc))
    return dist


def connected_room(rng, h, w, density, border=True):
    """pièce h×w avec obstacles aléatoires ; seules les cases de la plus grande composante restent libres."""
    walls = rng.random((h, w)) < density
    if border:
        walls[0, :] = walls[-1, :] = True
        walls[:, 0] = walls[:, -1] = True
    free = np.argwhere(~walls)
    if len(free) == 0:
        return None
    best = None
    seen = np.zeros_like(walls)
    for r, c in free:
        if seen[r, c]:
            continue
        d = bfs_dist(walls, (int(r), int(c)))
        comp = d >= 0
        seen |= comp
        if best is None or comp.sum() > best.sum():
            best = comp
    return ~best


def perfect_maze(rng, ch, cw, loops=0.0):
    """labyrinthe parfait (DFS) de ch×cw cellules -> murs (2ch+1)×(2cw+1) ; `loops` = part de murs cassés en plus."""
    H, W = 2 * ch + 1, 2 * cw + 1
    walls = np.ones((H, W), bool)
    seen = np.zeros((ch, cw), bool)
    stack = [(int(rng.integers(ch)), int(rng.integers(cw)))]
    seen[stack[0]] = True
    walls[2 * stack[0][0] + 1, 2 * stack[0][1] + 1] = False
    while stack:
        r, c = stack[-1]
        nb = [(r + dr, c + dc) for dr, dc in DIRS.values() if 0 <= r + dr < ch and 0 <= c + dc < cw and not seen[r + dr, c + dc]]
        if not nb:
            stack.pop()
            continue
        nr, nc = nb[int(rng.integers(len(nb)))]
        seen[nr, nc] = True
        walls[2 * nr + 1, 2 * nc + 1] = False
        walls[r + nr + 1, c + nc + 1] = False
        stack.append((nr, nc))
    if loops > 0:
        inner = [(r, c) for r in range(1, H - 1) for c in range(1, W - 1)
                 if walls[r, c] and ((r % 2 == 1) != (c % 2 == 1))]
        for k in rng.permutation(len(inner))[: int(loops * len(inner))]:
            walls[inner[k]] = False
    return walls


def free_cells(walls):
    return [(int(r), int(c)) for r, c in np.argwhere(~walls)]


def shortest_path(walls, a, b):
    d = bfs_dist(walls, b)
    if d[a] < 0:
        return None
    path = [a]
    H, W = walls.shape
    while path[-1] != b:
        r, c = path[-1]
        for dr, dc in DIRS.values():
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and d[rr, cc] == d[r, c] - 1:
                path.append((rr, cc))
                break
    return path


def sprite(t, style):
    """masque t×t : 0 transparent, 1 couleur principale, 2 détail (DETAIL). En t=1 tout devient plein."""
    m = np.zeros((t, t), np.int8)
    mid = t // 2
    if t == 1 or style == "full":
        m[:] = 1
    elif style == "ring":
        m[:] = 1
        if t >= 3:
            m[1:-1, 1:-1] = 0
    elif style == "dot":
        if t == 2:
            m[0, 0] = m[1, 1] = 1
        else:
            m[mid, mid] = 1
            if t >= 4 and t % 2 == 0:
                m[mid - 1:mid + 1, mid - 1:mid + 1] = 1
    elif style == "cross":
        if t == 2:
            m[0, 1] = m[1, 0] = 1
        else:
            k = 1 if t >= 5 else 0
            m[mid, k:t - k] = 1
            m[k:t - k, mid] = 1
    elif style == "corners":
        m[0, 0] = m[-1, -1] = 1
        if t >= 3:
            m[0, -1] = m[-1, 0] = 1
    elif style == "diag":
        np.fill_diagonal(m, 1)
    # ---- sprites détaillés (look officiel, t >= 3) : petits, avec point / œil / contour
    elif style == "box_dot":
        k = 1 if t >= 5 else 0
        m[k:t - k, k:t - k] = 1
        m[mid, mid] = 2
    elif style == "ring_dot":
        k = 1 if t >= 6 else 0
        m[k:t - k, k:t - k] = 1
        m[k + 1:t - k - 1, k + 1:t - k - 1] = 0
        m[mid, mid] = 2
    elif style == "eye":
        k = 1 if t >= 5 else 0
        m[k:t - k, k:t - k] = 1
        if t >= 4:
            m[k + 1, k + 1] = m[k + 1, t - k - 2] = 2
        else:
            m[0, 1] = 2
    elif style == "diamond":
        for y in range(t):
            for x in range(t):
                if abs(y - (t - 1) / 2) + abs(x - (t - 1) / 2) <= (t - 1) / 2:
                    m[y, x] = 1
        if t >= 5:
            m[mid, mid] = 2
    elif style == "small":
        k = 1 if t >= 4 else 0
        m[k:t - k, k:t - k] = 1
    elif style == "small_dot":
        k = 1 if t >= 4 else 0
        m[k:t - k, k:t - k] = 1
        m[mid, mid] = 0
    elif style == "frame":
        m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = 1
        m[mid, mid] = 2
    else:
        m[:] = 1
    return m


class Canvas:
    """grille logique de cases t×t pixels logiques."""

    def __init__(self, h, w, t, bg=BG, pad=0):
        self.t, self.pad = t, pad
        self.g = np.full((h * t + 2 * pad, w * t + 2 * pad), bg, np.int64)

    def put(self, r, c, color, style="full", color2=DETAIL):
        t, p = self.t, self.pad
        blk = self.g[p + r * t:p + (r + 1) * t, p + c * t:p + (c + 1) * t]
        m = sprite(t, style)
        blk[m == 1] = color
        blk[m == 2] = color2

    def thin_wall(self, walls, r, c, color=WALL):
        """mur fin : ligne d'1 px qui relie le centre de la case aux murs voisins (labyrinthe « dessiné au trait »)."""
        t, p = self.t, self.pad
        if t < 3:
            self.put(r, c, color, "full")
            return
        y0, x0, mid = p + r * t, p + c * t, t // 2
        H, W = walls.shape
        nb = 0
        self.g[y0 + mid, x0 + mid] = color
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and walls[rr, cc]:
                nb += 1
                if dr:
                    ys = range(y0, y0 + mid) if dr < 0 else range(y0 + mid, y0 + t)
                    for y in ys:
                        self.g[y, x0 + mid] = color
                else:
                    xs = range(x0, x0 + mid) if dc < 0 else range(x0 + mid, x0 + t)
                    for x in xs:
                        self.g[y0 + mid, x] = color
        if nb == 0:
            k = 1
            self.g[y0 + k:y0 + t - k, x0 + k:x0 + t - k] = color

    def cell(self, px):
        r, c = (px[0] - self.pad) // self.t, (px[1] - self.pad) // self.t
        return r, c

    def px(self, r, c):
        return (self.pad + r * self.t + self.t // 2, self.pad + c * self.t + self.t // 2)
