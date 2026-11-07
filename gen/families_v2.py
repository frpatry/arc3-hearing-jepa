"""Familles v2 « façon officielle » (principes absents de v1, cf. analysis/REPORT.md §3b) :
  glyph_key      (ls20, sc25, cn04) : clé-glyphe (forme, couleur, rotation) transformée par des cases ; porte = glyphe
  rule_table     (tr87, sb26, tn36) : dictionnaire glyphe -> glyphe(s) affiché ; traduire la phrase source
  coupled_bodies (m0r0, ar25, r11l, g50t) : un contrôle, plusieurs corps (miroir, inverse, retard, rotation)
  vessels        (vc33, su15) : vases, verser / transférer, vannes qui égalisent ; atteindre les graduations
Chaque niveau ajoute un élément ou une règle ; solvabilité par BFS ou par construction ; baseline = optimum trouvé."""
from collections import deque

import numpy as np

from .core import (AVATAR, BG, DETAIL, DIRS, DIR_OPS, EV, GOAL, MARK, OBJ2, PANEL, WALL, Canvas, Game, bfs_dist,
                   connected_room, free_cells, perfect_maze)
from .families_avatar import AvatarGame

_P = ["100/100/111", "111/010/010", "110/110/100", "010/111/001", "101/111/100", "001/111/010", "111/101/100"]
GLYPHS = [np.array([[ch == "1" for ch in r] for r in p.split("/")]) for p in _P]


def draw_glyph(g, y0, x0, T, k, rot, color):
    m = np.rot90(GLYPHS[k], rot)
    o = (T - 3) // 2
    sub = g[y0 + o:y0 + o + 3, x0 + o:x0 + o + 3]
    sub[m] = color


def _force_tile3(game):
    """les glyphes 3×3 exigent des tuiles >= 3 px : on borne l'échelle écran à ×2."""
    if game.official:
        game.T = max(3, game.T)
        game.s_screen = min(game.s_screen, 2)
        game.max_cells = max(5, (62 // game.s_screen) // game.T)
        game.px_budget = 62 // game.s_screen - 2


# =============================================================================== glyph_key
class GlyphKey(AvatarGame):
    family = "glyph_key"
    layouts = ("maze", "room")
    t_choices = (3,)
    bfs_cap = 60000
    CURR = [("color",), ("rot",), ("color", "rot"), ("shape",), ("shape", "color"), ("shape", "color", "rot"),
            ("shape", "color", "rot", "door2"), ("shape", "color", "rot", "door2", "reset"),
            ("shape", "color", "rot", "door2", "reset", "lava"), ("shape", "color", "rot", "door2", "reset", "lava")]

    def extra_setup(self, rng):
        _force_tile3(self)
        if self.official:
            self.t_pref = self.T
            if self.panel:
                self.max_cells = max(6, self.max_cells - 1)
        self.n_shapes = int(rng.integers(3, 5))
        self.n_cols = int(rng.integers(3, 5))
        self.shapes = [int(x) for x in rng.permutation(len(GLYPHS))[:self.n_shapes]]
        self.density = float(rng.uniform(0.05, 0.2))
        self.loops = float(rng.uniform(0.05, 0.2))

    def gen_level(self, rng, i, n):
        feats = self.CURR[min(i, len(self.CURR) - 1)]
        walls = self.make_walls(rng, i, n)
        fr = free_cells(walls)
        if len(fr) < 12:
            return None
        s, e = self.far_pair(rng, walls, 0.7)
        used = {s, e}
        g0 = (int(rng.integers(self.n_shapes)), int(rng.integers(self.n_cols)), int(rng.integers(4)))

        def target(base):
            t = list(base)
            if "shape" in feats:
                t[0] = (t[0] + int(rng.integers(1, self.n_shapes))) % self.n_shapes
            if "color" in feats:
                t[1] = (t[1] + int(rng.integers(1, self.n_cols))) % self.n_cols
            if "rot" in feats:
                t[2] = (t[2] + int(rng.integers(1, 4))) % 4
            return tuple(t)
        mods = {}
        for f, kind in (("shape", 0), ("color", 1), ("rot", 2)):
            if f in feats:
                for _ in range(1 + (i >= 4 and rng.random() < 0.5)):
                    p = self.pick(rng, fr, used)
                    if p is None:
                        return None
                    mods[p] = kind
                    used.add(p)
        doors = {e: target(g0)}
        if "door2" in feats:
            d = bfs_dist(walls, s)
            mid = [p for p in fr if p not in used and 0.3 * d[e] < d[p] < 0.7 * d[e]]
            if not mid:
                return None
            p = mid[int(rng.integers(len(mid)))]
            doors[p] = target(g0)
            used.add(p)
        if "reset" in feats:
            p = self.pick(rng, fr, used)
            if p is not None:
                mods[p] = 3
                used.add(p)
        lava = set()
        if "lava" in feats:
            for _ in range(int(len(fr) * 0.06)):
                p = self.pick(rng, fr, used)
                if p is not None:
                    lava.add(p)
                    used.add(p)
        lvl = self.mk(walls, exit=e, mods=mods, doors=doors, g0=g0, lava=frozenset(lava), feats=feats)
        return lvl, (s,) + g0, None

    def apply(self, lvl, st, op, px):
        pos, sh, co, ro = st
        p = self.nxt(lvl, pos, op)
        if p is None:
            return st, False
        if p in lvl["doors"] and lvl["doors"][p] != (sh, co, ro):
            return st, False
        if p in lvl["lava"]:
            return (p, sh, co, ro), True
        m = lvl["mods"].get(p)
        if m == 0:
            sh = (sh + 1) % self.n_shapes
        elif m == 1:
            co = (co + 1) % self.n_cols
        elif m == 2:
            ro = (ro + 1) % 4
        elif m == 3:
            sh, co, ro = lvl["g0"]
        return (p, sh, co, ro), False

    def is_win(self, lvl, st):
        return st[0] == lvl["exit"]

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        p = self.nxt(lvl, st[0], op)
        if p in lvl["doors"] or p in lvl["mods"] or dead:
            ev[EV["touch"]] = 1
        if st2[2] != st[2]:
            ev[EV["color"]] = 1
        if st2[1] != st[1] or st2[3] != st[3]:
            ev[EV["toggle"]] = 1
        if p in lvl["doors"] and st2[0] == p:
            ev[EV["open"]] = 1
        if any(abs(st2[0][0] - a) + abs(st2[0][1] - b) <= 1 for a, b in lvl["lava"]):
            ev[EV["danger"]] = 1

    def min_len(self, i, n):
        return min(14 + 7 * i, 70)

    def render(self, lvl, st):
        pos, sh, co, ro = st
        cv = self.canvas(lvl)
        T = lvl["t"]
        for p in lvl["lava"]:
            cv.put(*p, 8, "full")
        sty = {0: "cross", 1: "diamond", 2: "diag", 3: "ring"}
        for p, m in lvl["mods"].items():
            cv.put(*p, (10, 11, 12, 9)[m], sty[m])
        for p, (a, b, c) in lvl["doors"].items():
            cv.put(*p, GOAL, "ring")
            draw_glyph(cv.g, p[0] * T, p[1] * T, T, self.shapes[a], c, 4 + b)
        cv.put(*pos, AVATAR, self.av_style)
        g = cv.g
        if self.official and self.panel:                   # la clé courante dans un panneau (inventaire)
            pan = np.full((T, g.shape[1]), BG)
            draw_glyph(pan, 0, 0, T, self.shapes[sh], ro, 4 + co)
            g = np.vstack([pan, np.full((1, g.shape[1]), PANEL), g])
        else:                                              # sinon la clé est dessinée sur l'avatar
            y, x = pos[0] * T, pos[1] * T
            g[y:y + T, x:x + T] = AVATAR
            draw_glyph(g, y, x, T, self.shapes[sh], ro, 4 + co)
        return g


# =============================================================================== rule_table
class RuleTable(Game):
    family = "rule_table"
    simple_ops = ()

    def setup(self, rng):
        _force_tile3(self)
        if self.official:                                    # glyphes 3×3 natifs, affichés ×2 (comme vc33, ft09)
            self.s_screen = 2                                # (repli ×1 automatique si la mise en page est trop grande)
            self.px_budget = 60
        self.keys_mode = bool(rng.random() < 0.5)            # flèches (curseur + cycle) ou clic sur la case
        self.legend_fill = bool(rng.random() < 0.5)
        self.submit = bool(rng.random() < 0.5)               # ACTION5 = valider (mauvaise réponse -> GAME_OVER)
        ops = []
        if self.keys_mode:
            ops = list(DIR_OPS)
        if self.submit:
            ops.append("interact")
        self.simple_ops = tuple(ops)
        self.uses_click = not self.keys_mode
        self.symbols = [(int(sh), int(c)) for sh, c in zip(rng.permutation(len(GLYPHS)), rng.permutation(6))]

    def gen_level(self, rng, i, n):
        A = min(2 + (i + 1) // 2, 6)
        L = min(2 + i // 2 + int(rng.integers(0, 2)) + int(self.official), 7)
        multi = i >= 3                                         # règles 1 -> 2
        rules = {}
        for a in range(A):
            k = 2 if (multi and rng.random() < 0.35) else 1
            rules[a] = tuple(int(x) for x in rng.integers(0, A, k))
        src = tuple(int(x) for x in rng.integers(0, A, L))
        tgt = tuple(v for a in src for v in rules[a])
        if len(tgt) > 9:
            return None
        ans0 = tuple(int(x) for x in rng.integers(0, A, len(tgt))) if i >= 2 else (0,) * len(tgt)
        if ans0 == tgt:
            return None
        used = sorted(set(src))
        legend = [(a, rules[a]) for a in range(A) if a in used or rng.random() < 0.5]
        # mise en page (px logiques) : légende en haut, séparateur, source, réponse
        T = 4
        rows = len(legend)
        H = 1 + rows * T + 2 + T + 2 + T + 2
        W = max(1 + (1 + 2 + 2) * T, 1 + len(src) * T, 1 + len(tgt) * T) + 1
        if self.official and (H > self.px_budget or W > self.px_budget + 2):
            return None
        lvl = dict(A=A, rules=rules, src=src, tgt=tgt, legend=legend, T=T, H=H, W=W,
                   ans_r=1 + rows * T + 2 + T + 2)
        sol = []
        cur = 0
        if self.keys_mode:
            last = max([j for j in range(len(tgt)) if ans0[j] != tgt[j]], default=0)
            for j in range(last + 1):
                d = (tgt[j] - ans0[j]) % A
                if d:
                    sol += [("right", None)] * (j - cur)
                    cur = j
                    sol += [("up", None)] * d if d <= A - d else [("down", None)] * (A - d)
        else:
            for j in range(len(tgt)):
                sol += [("click", self.slot_px(lvl, j))] * ((tgt[j] - ans0[j]) % A)
        if self.submit:
            sol.append(("interact", None))
        return lvl, (ans0, 0, False), sol

    def slot_px(self, lvl, j):
        return (lvl["ans_r"] + 1, 1 + j * lvl["T"] + 1)

    def apply(self, lvl, st, op, px):
        ans, cur, ok = st
        A = lvl["A"]
        if op == "click":
            r, c = px
            if lvl["ans_r"] <= r < lvl["ans_r"] + 3:
                j = (c - 1) // lvl["T"]
                if 0 <= j < len(ans) and (c - 1) % lvl["T"] < 3:
                    ans = ans[:j] + ((ans[j] + 1) % A,) + ans[j + 1:]
            return (ans, cur, ok), False
        if op == "left":
            return (ans, max(0, cur - 1), ok), False
        if op == "right":
            return (ans, min(len(ans) - 1, cur + 1), ok), False
        if op in ("up", "down"):
            dv = 1 if op == "up" else -1
            return (ans[:cur] + ((ans[cur] + dv) % A,) + ans[cur + 1:], cur, ok), False
        if op == "interact":
            if ans == lvl["tgt"]:
                return (ans, cur, True), False
            return st, True                                   # mauvaise réponse validée : GAME_OVER
        return st, False

    def is_win(self, lvl, st):
        return st[0] == lvl["tgt"] and (st[2] or not self.submit)

    def heuristic(self, lvl, st):
        A = lvl["A"]
        return sum(min((t - a) % A, (a - t) % A) for a, t in zip(st[0], lvl["tgt"])) + (self.submit and not st[2])

    exact_progress = False

    def click_targets(self, lvl, st):
        return [self.slot_px(lvl, j) for j in range(len(st[0]))]

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        if st2[0] != st[0]:
            ev[EV["toggle"]] = ev[EV["color"]] = 1
        if op == "interact":
            ev[EV["touch"]] = 1

    def body(self, lvl, st):
        if not self.keys_mode:
            return None
        return (0, st[1], 1, max(2, len(st[0])))

    def min_len(self, i, n):
        return 2 + i

    def render(self, lvl, st):
        ans, cur, ok = st
        T = lvl["T"]
        g = np.full((lvl["H"], lvl["W"]), BG, np.int64)
        sym = self.symbols
        if self.legend_fill:
            g[0:1 + len(lvl["legend"]) * T, :] = WALL          # panneau de légende plein
        else:
            g[0, :] = WALL                                     # ou simple filet
        for k in range(len(lvl["legend"])):
            for m in range(2 + max(len(img) for _, img in lvl["legend"])):
                if m != 1:
                    g[1 + k * T:1 + k * T + 3, 1 + m * T:1 + m * T + 3] = BG

        def glyph(r, c, a):
            sh, co = sym[a]
            draw_glyph(g, r, c, 3, sh, 0, 4 + co)
        for k, (a, img) in enumerate(lvl["legend"]):
            r = 1 + k * T
            glyph(r, 1, a)
            g[r + 1, 1 + T:1 + T + 3] = PANEL                  # flèche « devient »
            g[r, 1 + T + 2] = g[r + 2, 1 + T + 2] = PANEL
            for m, b in enumerate(img):
                glyph(r, 1 + (2 + m) * T, b)
        sep = 1 + len(lvl["legend"]) * T
        g[sep, :] = PANEL
        for j, a in enumerate(lvl["src"]):
            glyph(sep + 2, 1 + j * T, a)
        r0 = lvl["ans_r"]
        for j, a in enumerate(ans):
            glyph(r0, 1 + j * T, a)
            g[r0 + 3, 1 + j * T:1 + j * T + 3] = MARK if (self.keys_mode and j == cur) else WALL   # soulignement
        return g


# =============================================================================== coupled_bodies
COUPLINGS = ("same", "mirror", "inverse", "flip", "rot", "delay")
_MAP = dict(same={"up": "up", "down": "down", "left": "left", "right": "right"},
            mirror={"up": "up", "down": "down", "left": "right", "right": "left"},
            flip={"up": "down", "down": "up", "left": "left", "right": "right"},
            inverse={"up": "down", "down": "up", "left": "right", "right": "left"},
            rot={"up": "right", "right": "down", "down": "left", "left": "up"})


class CoupledBodies(AvatarGame):
    family = "coupled_bodies"
    layouts = ("room",)
    bfs_cap = 40000

    def table_ok(self, lvl, st0):
        return len(st0[0]) <= 2

    def extra_setup(self, rng):
        order = ["mirror", "inverse"] + list(rng.permutation(["flip", "rot", "delay"]))
        self.order = [str(x) for x in order]
        self.goal_mode = str(rng.choice(["targets", "targets", "targets", "meet"]))
        self.density = float(rng.uniform(0.12, 0.28))
        self.body_style = self.av_style
        if self.official:
            self.wall_style = str(rng.choice(["full", "small", "thin"]))

    def couplings(self, i):
        if i == 0:
            return ["same", "same"]
        k = min(i - 1, len(self.order) - 1)
        c = ["same", self.order[k]]
        if i >= 4:
            c.append(self.order[(k + 1) % len(self.order)])
        return c

    def gen_level(self, rng, i, n):
        cps = self.couplings(i)
        fails = self.__dict__.setdefault("_fails", {})
        fails[i] = fails.get(i, -1) + 1
        if fails[i] > 40 and len(cps) > 2:              # repli : deux corps seulement
            cps = cps[:1] + cps[-1:]
        k = len(cps)
        h = w = self.cap(min(6 + i // 2 + int(rng.integers(0, 2)), 7 if k == 3 else 12))
        walls = connected_room(rng, h, w, self.density)
        fr = free_cells(walls)
        if len(fr) < 2 * k + 6:
            return None
        cells = [fr[j] for j in rng.choice(len(fr), 2 * k, replace=False)]
        if self.goal_mode == "meet":                     # corps de départ éloignés les uns des autres
            best = None
            for _ in range(20):
                cs = [fr[j] for j in rng.choice(len(fr), k, replace=False)]
                sp = min(abs(a[0] - b[0]) + abs(a[1] - b[1]) for x, a in enumerate(cs) for b in cs[x + 1:])
                if best is None or sp > best[0]:
                    best = (sp, cs)
            cells = best[1] + cells[k:]
            tg = ()
            lvl = self.mk(walls, cps=tuple(cps), targets=tg)
            return lvl, (tuple(cells[:k]), "none"), None
        # cibles = un état ATTEIGNABLE et lointain (BFS depuis le départ) : soluble et long par construction
        st0 = (tuple(cells[:k]), "none")
        lvl = self.mk(walls, cps=tuple(cps), targets=())
        depth, q = {st0: 0}, deque([st0])
        while q and len(depth) < 6000:
            s_ = q.popleft()
            for o in DIR_OPS:
                s2, _ = self.apply(lvl, s_, o, None)
                if s2 not in depth:
                    depth[s2] = depth[s_] + 1
                    q.append(s2)
        dmax = max(depth.values())
        far = [s_ for s_, dd in depth.items() if dd >= 0.75 * dmax and len(set(s_[0])) == k]
        if not far:
            return None
        tg = far[int(rng.integers(len(far)))][0]
        lvl["targets"] = tg
        return lvl, st0, None

    def apply(self, lvl, st, op, px):
        pos, last = st
        out = []
        for p, cp in zip(pos, lvl["cps"]):
            o = last if cp == "delay" else _MAP[cp][op]
            q = self.nxt(lvl, p, o) if o != "none" else None
            out.append(q or p)
        return (tuple(out), op if "delay" in lvl["cps"] else "none"), False

    def is_win(self, lvl, st):
        pos = st[0]
        if lvl["targets"]:
            return pos == lvl["targets"]
        return len(set(pos)) == 1

    def body(self, lvl, st):
        p = st[0][0]
        return (p[0], p[1], lvl["H"], lvl["W"])

    def heuristic(self, lvl, st):
        pos = st[0]
        if lvl["targets"]:
            return max(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in zip(pos, lvl["targets"]))
        return max(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a in pos for b in pos) / 2

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        moved = sum(a != b for a, b in zip(st[0], st2[0]))
        if 0 < moved < len(st[0]):
            ev[EV["touch"]] = 1                          # un corps est bloqué, l'autre avance
        if len(set(st2[0])) < len(set(st[0])):
            ev[EV["collect"]] = 1                        # deux corps se rejoignent

    def min_len(self, i, n):
        return min(4 + i, 9) if self.goal_mode == "meet" else min(8 + 4 * i, 40)

    def render(self, lvl, st):
        cv = self.canvas(lvl)
        for j, p in enumerate(lvl["targets"]):
            cv.put(*p, AVATAR if j == 0 else 4 + j, "corners" if lvl["t"] >= 3 else "dot")
        for j, p in reversed(list(enumerate(st[0]))):
            cv.put(*p, AVATAR if j == 0 else 4 + j, self.body_style)
        return cv.g


# =============================================================================== vessels
class Vessels(Game):
    family = "vessels"
    simple_ops = ()
    bfs_cap = 40000

    def setup(self, rng):
        self.uses_click = True
        self.uh = int(rng.choice([3, 4]))                 # hauteur d'une unité (px)
        self.vw = int(rng.choice([5, 6, 7]))              # largeur d'un vase
        self.order = ["unit", "pour"] if rng.random() < 0.5 else ["pour", "unit"]
        if self.official:
            self.s_screen = min(self.s_screen, 2)
            self.px_budget = 62 // self.s_screen - 2
            if self.s_screen == 2:
                self.uh, self.vw = 2, 4

    def mode(self, i):
        m = self.order[0] if i < 2 else self.order[1] if i < 4 else self.order[i % 2]
        valves = i >= 4
        return m, valves

    def gen_level(self, rng, i, n):
        mode, valves = self.mode(i)
        gap = 2 + valves * 1
        k = min(2 + (i + 1) // 2, 5)
        if self.official:
            k = max(2, min(k, (self.px_budget - 2) // (self.vw + 2 + gap)))
        cap = tuple(int(x) for x in rng.integers(3, 6 + min(i, 3), k))
        lv = tuple(int(rng.integers(0, c + 1)) for c in cap)
        if sum(lv) == 0:
            return None
        Hpx = max(cap) * self.uh + 4
        Wpx = 2 + k * (self.vw + 2 + gap)
        if self.official and (Hpx + 4 > self.px_budget or Wpx > self.px_budget + 2):
            return None
        lvl = dict(cap=cap, mode=mode, valves=valves, k=k, tgt={}, Hpx=Hpx + 3, Wpx=Wpx, gap=gap)
        # cible = un état atteignable à profondeur >= min_len (BFS) -> soluble et optimal par construction
        st0 = (lv, -1)
        depth = {st0: 0}
        q = deque([st0])
        while q and len(depth) < self.bfs_cap:
            s = q.popleft()
            for mv in self.moves(lvl, s):
                s2, _ = self.apply(lvl, s, *mv)
                if s2 not in depth:
                    depth[s2] = depth[s] + 1
                    q.append(s2)
        need = self.min_len(i, n)
        far = [s for s, d in depth.items() if d >= need and s[0] != lv]
        if not far:
            far = [s for s, d in depth.items() if s[0] != lv and d >= max(depth.values()) - 1]
        if not far:
            return None
        goal = far[int(rng.integers(len(far)))][0]
        marked = list(range(k)) if i < 2 or rng.random() < 0.5 else \
            sorted(int(x) for x in rng.choice(k, max(1, k - 1), replace=False))
        lvl["tgt"] = {j: goal[j] for j in marked}
        sol = self.solve(lvl, st0)
        if sol is None:
            return None
        return lvl, st0, sol

    def x0(self, lvl, j):
        return 2 + j * (self.vw + 2 + lvl["gap"])

    def vessel_px(self, lvl, j):
        return (lvl["Hpx"] - 4, self.x0(lvl, j) + (self.vw + 2) // 2)

    def valve_px(self, lvl, j):
        return (lvl["Hpx"] - 3, self.x0(lvl, j) + self.vw + 2)

    def click_targets(self, lvl, st):
        t = [self.vessel_px(lvl, j) for j in range(lvl["k"])]
        if lvl["valves"]:
            t += [self.valve_px(lvl, j) for j in range(lvl["k"] - 1)]
        return t

    def which(self, lvl, px):
        r, c = px
        for j in range(lvl["k"]):
            x = self.x0(lvl, j)
            if x <= c < x + self.vw + 2 and 0 <= r < lvl["Hpx"] - 2:
                return ("v", j)
            if lvl["valves"] and j < lvl["k"] - 1 and c == x + self.vw + 2 and r == lvl["Hpx"] - 3:
                return ("valve", j)
        return None

    def apply(self, lvl, st, op, px):
        lv, sel = st
        w = self.which(lvl, px) if px is not None else None
        if w is None:
            return st, False
        kind, j = w
        cap = lvl["cap"]
        if kind == "valve":
            a, b = lv[j], lv[j + 1]
            tot = a + b
            na = min(cap[j], (tot + 1) // 2)
            nb = tot - na
            if nb > cap[j + 1]:
                nb = cap[j + 1]
                na = tot - nb
            lv = lv[:j] + (na, nb) + lv[j + 2:]
            return (lv, -1), False
        if sel < 0:
            return ((lv, j) if lv[j] > 0 else st), False
        if sel == j:
            return (lv, -1), False
        room = cap[j] - lv[j]
        amt = min(lv[sel], room) if lvl["mode"] == "pour" else min(1, lv[sel], room)
        l2 = list(lv)
        l2[sel] -= amt
        l2[j] += amt
        return (tuple(l2), -1), False

    def is_win(self, lvl, st):
        return all(st[0][j] == v for j, v in lvl["tgt"].items())

    def heuristic(self, lvl, st):
        return sum(abs(st[0][j] - v) for j, v in lvl["tgt"].items())

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        if st2[1] != st[1]:
            ev[EV["toggle"]] = 1
        if st2[0] != st[0]:
            ev[EV["push"]] = 1
            ev[EV["touch"]] = 1
            if px is not None and self.which(lvl, px) and self.which(lvl, px)[0] == "valve":
                ev[EV["open"]] = 1

    def min_len(self, i, n):
        return min(2 + i, 9)

    def render(self, lvl, st):
        lv, sel = st
        g = np.full((lvl["Hpx"], lvl["Wpx"]), BG, np.int64)
        base = lvl["Hpx"] - 3
        for j, c in enumerate(lvl["cap"]):
            x = self.x0(lvl, j)
            top = base - c * self.uh - 1
            col = MARK if j == sel else WALL
            g[top:base + 1, x] = col
            g[top:base + 1, x + self.vw + 1] = col
            g[base, x:x + self.vw + 2] = col
            for u in range(lv[j]):
                y = base - 1 - u * self.uh
                g[y - self.uh + 1:y + 1, x + 1:x + self.vw + 1] = 4
            if j in lvl["tgt"]:
                y = base - lvl["tgt"][j] * self.uh
                g[y, x - 1] = GOAL                           # graduation cible
                if x + self.vw + 2 < lvl["Wpx"]:
                    g[y, x + self.vw + 2] = GOAL
            if lvl["valves"] and j < lvl["k"] - 1:
                g[base - 1, x + self.vw + 2] = 9
                g[base, x + self.vw + 2:x + self.vw + 2 + lvl["gap"] + 1] = 9
        return g


# =============================================================================== npc_rules
class NpcRules(AvatarGame):
    """PNJ à comportement propre (tu93, wa30) : patrouilleur, imitateur (copie nos pas), poursuivant (se rapproche
    un tour sur deux). Les toucher = GAME_OVER. Un type de PNJ nouveau par niveau ; BFS sur (avatar, PNJ, t)."""
    family = "npc_rules"
    layouts = ("maze", "room")
    bfs_cap = 12000
    table_cap = 8000

    def extra_setup(self, rng):
        self.kinds_order = ["patrol"] + [str(x) for x in rng.permutation(["mimic", "chaser"])]
        self.loops = float(rng.uniform(0.15, 0.3))
        self.density = float(rng.uniform(0.08, 0.2))
        self.npc_style = {"patrol": "diamond", "mimic": "eye", "chaser": "box_dot"} if self.official else \
            {"patrol": "full", "mimic": "cross", "chaser": "ring"}
        self.wait = bool(rng.random() < 0.5)
        if self.wait:
            self.simple_ops = DIR_OPS + ("interact",)

    def kinds(self, i):
        ks = self.kinds_order[:min(i + 1, 3)]
        if i >= 4:
            ks = ks + ["patrol"]                       # au plus 2 PNJ mobiles (+ patrouilleurs)
        return ks

    def gen_level(self, rng, i, n):
        mc = self.max_cells
        self.max_cells = min(mc, 13)                       # petites arènes : l'état joint reste petit
        walls = self.make_walls(rng, i, n)
        self.max_cells = mc
        fr = free_cells(walls)
        s, e = self.far_pair(rng, walls, 0.7)
        used = {s, e}
        npcs, patrols = [], []
        d0 = bfs_dist(walls, s)
        kinds = self.kinds(i)
        fails = self.__dict__.setdefault("_fails", {})
        fails[i] = fails.get(i, -1) + 1
        if fails[i] > 15:                                  # repli : moins de PNJ
            kinds = kinds[-2:] if fails[i] <= 30 else kinds[-1:]
        for kind in kinds:
            cand = [p for p in fr if p not in used and d0[p] >= 3]
            if not cand:
                return None
            p = cand[int(rng.integers(len(cand)))]
            used.add(p)
            if kind == "patrol":
                op = ("right", "down")[int(rng.integers(2))]
                line = [p]
                lv0 = self.mk(walls)
                for _ in range(int(rng.integers(2, 5))):
                    q = self.nxt(lv0, line[-1], op)
                    if q is None or q in used:
                        break
                    line.append(q)
                if len(line) < 2:
                    return None
                patrols.append(tuple(line))
                npcs.append(("patrol", len(patrols) - 1))
            else:
                npcs.append((kind, p))
        P = 1
        for pth in patrols:
            P = int(np.lcm(P, 2 * (len(pth) - 1)))
        P = int(np.lcm(P, 2))
        lvl = self.mk(walls, exit=e, patrols=tuple(patrols), kinds=tuple(k for k, _ in npcs), P=P)
        lvl["pp"] = [[self.ppos(pth, t) for t in range(P)] for pth in patrols]   # positions précalculées
        mob = tuple(p for k, p in npcs if k != "patrol")
        return lvl, (s, mob, 0), None

    @staticmethod
    def ppos(path, t):
        L = len(path)
        P = 2 * (L - 1)
        k = t % P
        return path[k if k < L else P - k]

    def apply(self, lvl, st, op, px):
        pos, mob, t = st
        p2 = self.nxt(lvl, pos, op) if op in DIRS else None
        p2 = p2 or pos
        t2 = (t + 1) % lvl["P"]
        new, j = [], 0
        for kind in lvl["kinds"]:
            if kind == "patrol":
                continue
            q = mob[j]
            j += 1
            if kind == "mimic" and op in DIRS:
                q = self.nxt(lvl, q, op) or q
            elif kind == "chaser" and t % 2 == 0:
                dr, dc = p2[0] - q[0], p2[1] - q[1]
                tries = [("down" if dr > 0 else "up") if dr else None, ("right" if dc > 0 else "left") if dc else None]
                if abs(dc) > abs(dr):
                    tries.reverse()
                for o in tries:
                    if o and self.nxt(lvl, q, o):
                        q = self.nxt(lvl, q, o)
                        break
            new.append(q)
        new = tuple(new)
        dead = p2 in new or any(a == pos and b == p2 for a, b in zip(mob, new))
        for pp in lvl["pp"]:
            a, b = pp[t], pp[t2]
            if b == p2 or (b == pos and a == p2):
                dead = True
                break
        return (p2, new, t2), dead

    def is_win(self, lvl, st):
        return st[0] == lvl["exit"]

    def npc_cells(self, lvl, st):
        out = [self.ppos(p, st[2]) for p in lvl["patrols"]]
        return out + list(st[1])

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        pos = st2[0]
        if dead or any(abs(pos[0] - a) + abs(pos[1] - b) <= 1 for a, b in self.npc_cells(lvl, st2)):
            ev[EV["danger"]] = 1
        if dead:
            ev[EV["touch"]] = 1

    def heuristic(self, lvl, st):
        if "_de" not in lvl:
            lvl["_de"] = bfs_dist(lvl["walls"], lvl["exit"])
        return float(max(0, lvl["_de"][st[0]]))

    def min_len(self, i, n):
        return min(8 + 4 * i, 40)

    def render(self, lvl, st):
        cv = self.canvas(lvl)
        cv.put(*lvl["exit"], GOAL, self.goal_style)
        cols = {"patrol": 8, "mimic": 9, "chaser": 10}
        for path in lvl["patrols"]:
            cv.put(*self.ppos(path, st[2]), cols["patrol"], self.npc_style["patrol"])
        mk = [k for k in lvl["kinds"] if k != "patrol"]
        for kind, q in zip(mk, st[1]):
            cv.put(*q, cols[kind], self.npc_style[kind])
        cv.put(*st[0], AVATAR, self.av_style)
        return cv.g


# =============================================================================== capture_merge (solitaire)
class CaptureMerge(Game):
    """solitaire (lf52) : cliquer un pion puis une case vide à 2 cases -> il saute et CAPTURE le pion du milieu.
    But : ne laisser qu'un pion (ou atteindre la case cible). Généré à rebours depuis un pion -> soluble ; baseline =
    2 clics par saut (optimal : il faut pions-1 sauts et 2 clics par saut)."""
    family = "capture_merge"
    simple_ops = ()

    def setup(self, rng):
        self.uses_click = True
        self.t = int(self.T) if self.official else int(rng.choice([2, 3]))
        self.t = max(2, self.t)
        self.peg_style = str(rng.choice(["box_dot", "diamond", "small"])) if self.official else "full"
        self.target_mode = bool(rng.random() < 0.4)        # le dernier pion doit finir sur la case marquée

    def gen_level(self, rng, i, n):
        nb = self.cap(min(4 + i // 2 + int(rng.integers(0, 2)), 9))
        holes = {(r, c) for r in range(nb) for c in range(nb) if rng.random() > 0.12 or (r, c) == (nb // 2, nb // 2)}
        last = sorted(holes)[int(rng.integers(len(holes)))]
        pegs = {last}
        moves = []
        m = 2 + i + int(rng.integers(0, 2))
        for _ in range(m * 6):
            if len(moves) >= m:
                break
            p = sorted(pegs)[int(rng.integers(len(pegs)))]
            dr, dc = list(DIRS.values())[int(rng.integers(4))]
            a, b = (p[0] + dr, p[1] + dc), (p[0] + 2 * dr, p[1] + 2 * dc)
            if a in holes and b in holes and a not in pegs and b not in pegs:
                pegs.remove(p)
                pegs |= {a, b}
                moves.append((b, p))                        # saut avant : de b vers p par-dessus a
        if len(moves) < 2:
            return None
        lvl = dict(nb=nb, holes=frozenset(holes), goal=last if self.target_mode else None, t=self.t)
        sol = []
        for src, dst in reversed(moves):
            sol += [("click", self.cell_px(lvl, src)), ("click", self.cell_px(lvl, dst))]
        return lvl, (tuple(sorted(pegs)), None), sol

    def cell_px(self, lvl, p):
        t = lvl["t"]
        return (1 + p[0] * (t + 1) + t // 2, 1 + p[1] * (t + 1) + t // 2)

    def which(self, lvl, px):
        t = lvl["t"]
        r, c = px[0] - 1, px[1] - 1
        if r < 0 or c < 0 or r % (t + 1) >= t or c % (t + 1) >= t:
            return None
        p = (r // (t + 1), c // (t + 1))
        return p if p in lvl["holes"] else None

    def apply(self, lvl, st, op, px):
        pegs, sel = st
        p = self.which(lvl, px)
        if p is None:
            return st, False
        if p in pegs:
            return (pegs, None if sel == p else p), False
        if sel is None:
            return st, False
        dr, dc = p[0] - sel[0], p[1] - sel[1]
        if sorted((abs(dr), abs(dc))) != [0, 2]:
            return st, False
        mid = (sel[0] + dr // 2, sel[1] + dc // 2)
        if mid not in pegs:
            return st, False
        new = tuple(sorted([q for q in pegs if q not in (sel, mid)] + [p]))
        return (new, None), False

    def is_win(self, lvl, st):
        return len(st[0]) == 1 and (lvl["goal"] is None or st[0][0] == lvl["goal"])

    exact_progress = False

    def heuristic(self, lvl, st):
        return 2 * (len(st[0]) - 1) - (st[1] is not None)

    def click_targets(self, lvl, st):
        return [self.cell_px(lvl, p) for p in lvl["holes"]]

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        if st2[1] != st[1]:
            ev[EV["toggle"]] = 1
        if len(st2[0]) < len(st[0]):
            ev[EV["collect"]] = ev[EV["touch"]] = ev[EV["push"]] = 1

    def min_len(self, i, n):
        return min(4 + 2 * i, 20)

    def render(self, lvl, st):
        pegs, sel = st
        t, nb = lvl["t"], lvl["nb"]
        g = np.full((2 + nb * (t + 1), 2 + nb * (t + 1)), BG, np.int64)
        cv = Canvas(1, 1, t)
        for (r, c) in lvl["holes"]:
            y, x = 1 + r * (t + 1), 1 + c * (t + 1)
            g[y + t // 2, x + t // 2] = WALL
            if lvl["goal"] == (r, c):
                g[y:y + t, x] = GOAL
                g[y:y + t, x + t - 1] = GOAL
        from .core import sprite
        for p in pegs:
            y, x = 1 + p[0] * (t + 1), 1 + p[1] * (t + 1)
            m = sprite(t, self.peg_style)
            blk = g[y:y + t, x:x + t]
            blk[m == 1] = MARK if p == sel else 4
            blk[m == 2] = DETAIL
        return g
