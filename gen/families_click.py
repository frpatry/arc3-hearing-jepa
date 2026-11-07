"""Familles « clic » (ACTION6) et mixtes clic + flèches : basculer vers un motif de référence, peindre par copie,
« lights out », remplissage (flood-it), sélectionner-puis-déplacer, tourner une pièce pour l'emboîter."""
from itertools import permutations

import numpy as np

from .core import AVATAR, BG, DIRS, DIR_OPS, EV, GOAL, MARK, WALL, Canvas, Game, bfs_dist, connected_room, free_cells
from .families_avatar import AvatarGame


class Board:
    """plateau de nr×nc cases de t px séparées de `gap` px, coin haut-gauche (r0, c0) dans la grille logique."""

    def __init__(self, nr, nc, t, gap, r0, c0):
        self.nr, self.nc, self.t, self.gap, self.r0, self.c0 = nr, nc, t, gap, r0, c0
        self.h = nr * (t + gap) - gap
        self.w = nc * (t + gap) - gap

    def cell(self, px):
        r, c = px[0] - self.r0, px[1] - self.c0
        if r < 0 or c < 0:
            return None
        a, ra = divmod(r, self.t + self.gap)
        b, rb = divmod(c, self.t + self.gap)
        if a >= self.nr or b >= self.nc or ra >= self.t or rb >= self.t:
            return None
        return a * self.nc + b

    def px(self, k):
        a, b = divmod(k, self.nc)
        return (self.r0 + a * (self.t + self.gap) + self.t // 2, self.c0 + b * (self.t + self.gap) + self.t // 2)

    def draw(self, g, vals, colmap):
        for k, v in enumerate(vals):
            a, b = divmod(k, self.nc)
            r, c = self.r0 + a * (self.t + self.gap), self.c0 + b * (self.t + self.gap)
            g[r:r + self.t, c:c + self.t] = colmap(v)


class PanelGame(Game):
    """plateau cliquable + panneau de référence (motif cible) à droite ou en dessous."""
    simple_ops = ()

    def setup(self, rng):
        self.uses_click = True
        self.t = int(rng.choice([1, 2, 3]))
        if self.official:
            self.t = max(2, self.T) if self.s_screen <= 2 else self.T
        self.gap = int(rng.choice([0, 1])) if self.t > 1 else int(rng.choice([0, 1], p=[0.3, 0.7]))
        self.ref_side = str(rng.choice(["right", "below"]))
        self.ref_t = int(rng.choice([1, self.t]))
        self.frame_ref = bool(rng.random() < 0.5)
        self.extra_setup(rng)

    def extra_setup(self, rng):
        pass

    def boards(self, nr, nc, left=1):
        t = self.t
        while t > 1 and max(nr, nc) * (t + self.gap) * 2 > self.px_budget:
            t -= 1
        B = Board(nr, nc, t, self.gap, 1, left)
        rt = min(self.ref_t, t)
        if self.ref_side == "right":
            R = Board(nr, nc, rt, 0, 1, left + B.w + 2)
            H, W = 2 + max(B.h, R.h), left + B.w + 2 + R.w + 1
        else:
            R = Board(nr, nc, rt, 0, 1 + B.h + 2, left)
            H, W = 1 + B.h + 2 + R.h + 1, left + max(B.w, R.w) + 1
        return B, R, (H, W)

    def base_grid(self, lvl):
        g = np.full(lvl["shape"], BG, np.int64)
        R = lvl["R"]
        if self.frame_ref:
            g[R.r0 - 1:R.r0 + R.h + 1, R.c0 - 1:R.c0 + R.w + 1] = WALL
        return g


# ============================================================================ 11. basculer les couleurs pour copier la référence
class ToggleMatch(PanelGame):
    family = "toggle_match"

    def extra_setup(self, rng):
        self.k = int(rng.integers(2, 5))
        self.gap = 1                                  # cases toujours séparées

    def gen_level(self, rng, i, n):
        big = int(self.official)                              # look officiel : plateaux et niveaux plus longs
        nr = min(3 + big + i // 2 + int(rng.integers(0, 2)), 7 + big)
        nc = min(3 + big + i // 2 + int(rng.integers(0, 2)), 7 + big)
        B, R, shape = self.boards(nr, nc)
        tgt = rng.integers(0, self.k, nr * nc)
        cur = tgt.copy()
        m = min(2 + 2 * i + big * (2 + i), nr * nc)
        for j in rng.choice(nr * nc, m, replace=False):
            cur[j] = (tgt[j] + int(rng.integers(1, self.k))) % self.k
        sol = []
        for j in range(nr * nc):
            sol += [("click", B.px(j))] * int((tgt[j] - cur[j]) % self.k)
        return dict(B=B, R=R, shape=shape, tgt=tuple(int(v) for v in tgt)), tuple(int(v) for v in cur), sol

    def apply(self, lvl, st, op, px):
        k = lvl["B"].cell(px)
        if k is None:
            return st, False
        s = list(st)
        s[k] = (s[k] + 1) % self.k
        return tuple(s), False

    def is_win(self, lvl, st):
        return st == lvl["tgt"]

    def click_targets(self, lvl, st):
        return [lvl["B"].px(j) for j in range(len(st))]

    exact_progress = False

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        if st2 != st:
            ev[EV["toggle"]] = ev[EV["color"]] = 1

    def heuristic(self, lvl, st):
        return sum((t - c) % self.k for t, c in zip(lvl["tgt"], st))

    def render(self, lvl, st):
        g = self.base_grid(lvl)
        lvl["B"].draw(g, st, lambda v: 4 + v)
        lvl["R"].draw(g, lvl["tgt"], lambda v: 4 + v)
        return g

    def min_len(self, i, n):
        return 2 + i


# ============================================================================ 12. peindre (choisir une couleur dans la palette, cliquer les cases)
class PaintCopy(PanelGame):
    family = "paint_copy"

    def extra_setup(self, rng):
        self.k = int(rng.integers(2, 5))
        self.erase = bool(rng.random() < 0.5)

    def gen_level(self, rng, i, n):
        big = int(self.official)
        nr = min(3 + big + i // 2 + int(rng.integers(0, 2)), 7 + big)
        nc = min(3 + big + i // 2 + int(rng.integers(0, 2)), 7 + big)
        pal = list(range(1, self.k + 1)) + ([0] if self.erase else [])     # 0 = case vide (gomme)
        st_ = self.t
        sw = Board(len(pal), 1, max(1, min(st_, 2)), 1, 1, 2)                 # colonne de pastilles
        B, R, shape = self.boards(nr, nc, left=2 + sw.w + 2)
        shape = (max(shape[0], sw.h + 2), shape[1])
        dens = 0.35 + 0.1 * min(i, 4)
        tgt = np.where(rng.random(nr * nc) < dens, rng.integers(1, self.k + 1, nr * nc), 0)
        if i >= 2 and self.erase:
            cur = np.where(rng.random(nr * nc) < 0.3, rng.integers(0, self.k + 1, nr * nc), tgt)
        else:
            cur = np.zeros(nr * nc, np.int64)
        if (cur == tgt).all():
            return None
        if not self.erase and ((cur != tgt) & (tgt == 0)).any():
            return None
        sel0 = 0
        diff = [j for j in range(nr * nc) if cur[j] != tgt[j]]
        need = sorted({int(tgt[j]) for j in diff}, key=lambda v: (pal.index(v) != sel0, v))
        sol, sel = [], sel0
        for v in need:
            if pal.index(v) != sel:
                sel = pal.index(v)
                sol.append(("click", sw.px(sel)))
            sol += [("click", B.px(j)) for j in diff if tgt[j] == v]
        lvl = dict(B=B, R=R, sw=sw, pal=pal, shape=shape, tgt=tuple(int(v) for v in tgt))
        return lvl, (tuple(int(v) for v in cur), sel0), sol

    def apply(self, lvl, st, op, px):
        board, sel = st
        j = lvl["sw"].cell(px)
        if j is not None:
            return (board, j), False
        k = lvl["B"].cell(px)
        if k is None:
            return st, False
        b = list(board)
        b[k] = lvl["pal"][sel]
        return (tuple(b), sel), False

    def is_win(self, lvl, st):
        return st[0] == lvl["tgt"]

    def click_targets(self, lvl, st):
        return [lvl["sw"].px(j) for j in range(len(lvl["pal"]))] + [lvl["B"].px(j) for j in range(len(st[0]))]

    exact_progress = False

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        if st2[1] != st[1]:
            ev[EV["toggle"]] = 1
        if st2[0] != st[0]:
            ev[EV["color"]] = 1

    def heuristic(self, lvl, st):
        board, sel = st
        need = {t for t, c in zip(lvl["tgt"], board) if t != c}
        n = sum(t != c for t, c in zip(lvl["tgt"], board))
        return n + len(need) - (lvl["pal"][sel] in need)

    def render(self, lvl, st):
        g = self.base_grid(lvl)
        cm = lambda v: 4 + v - 1 if v > 0 else 12          # noqa: E731  (12 = case vide du plateau)
        lvl["B"].draw(g, st[0], cm)
        lvl["R"].draw(g, lvl["tgt"], cm)
        sw = lvl["sw"]
        sw.draw(g, lvl["pal"], cm)
        r = sw.px(st[1])[0]
        g[r, 0] = MARK                                     # curseur de la couleur choisie
        return g

    def min_len(self, i, n):
        return 3 + i


# ============================================================================ 13. lights out (un clic bascule un motif de cases)
class LightsOut(PanelGame):
    family = "lights_out"

    def extra_setup(self, rng):
        self.pattern = str(rng.choice(["plus", "x", "rowcol", "plus"]))
        self.ref_mode = bool(rng.random() < 0.4)              # sinon : tout éteindre
        self.gap = 1

    def effect(self, j, nr, nc):
        a, b = divmod(j, nc)
        if self.pattern == "plus":
            cells = [(a, b), (a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)]
        elif self.pattern == "x":
            cells = [(a, b), (a + 1, b + 1), (a - 1, b - 1), (a - 1, b + 1), (a + 1, b - 1)]
        else:
            cells = [(a, y) for y in range(nc)] + [(x, b) for x in range(nr) if x != a]
        return tuple(x * nc + y for x, y in cells if 0 <= x < nr and 0 <= y < nc)

    def gen_level(self, rng, i, n):
        nr = nc = min(3 + i // 2, 6)
        B, R, shape = self.boards(nr, nc)
        if not self.ref_mode:
            shape = (B.h + 2, B.w + 2)
        tgt = rng.integers(0, 2, nr * nc) if self.ref_mode else np.zeros(nr * nc, np.int64)
        m = min(1 + i + int(rng.integers(0, 2)) + int(self.official) * (1 + i // 2), nr * nc // 2)
        presses = [int(j) for j in rng.choice(nr * nc, m, replace=False)]
        cur = tgt.copy()
        eff = [self.effect(j, nr, nc) for j in range(nr * nc)]
        for j in presses:
            for q in eff[j]:
                cur[q] ^= 1
        if (cur == tgt).all():
            return None
        lvl = dict(B=B, R=R, shape=shape, tgt=tuple(int(v) for v in tgt), eff=eff)
        return lvl, tuple(int(v) for v in cur), [("click", B.px(j)) for j in presses]

    def apply(self, lvl, st, op, px):
        k = lvl["B"].cell(px)
        if k is None:
            return st, False
        s = list(st)
        for q in lvl["eff"][k]:
            s[q] ^= 1
        return tuple(s), False

    def is_win(self, lvl, st):
        return st == lvl["tgt"]

    def click_targets(self, lvl, st):
        return [lvl["B"].px(j) for j in range(len(st))]

    exact_progress = False
    progress_scale = 3.0

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        if st2 != st:
            ev[EV["toggle"]] = 1

    def heuristic(self, lvl, st):
        return sum(a != b for a, b in zip(lvl["tgt"], st))

    def render(self, lvl, st):
        g = np.full(lvl["shape"], BG, np.int64) if not self.ref_mode else self.base_grid(lvl)
        lvl["B"].draw(g, st, lambda v: 4 if v else 5)
        if self.ref_mode:
            lvl["R"].draw(g, lvl["tgt"], lambda v: 4 if v else 5)
        return g

    def min_len(self, i, n):
        return 1 + min(i, 3)


# ============================================================================ 14. flood-it (remplir depuis le coin)
class FloodIt(PanelGame):
    family = "flood_it"

    def extra_setup(self, rng):
        self.k = int(rng.integers(3, 6))
        self.board_click = bool(rng.random() < 0.4)           # cliquer une case du plateau (sinon une pastille)

    @staticmethod
    def flood(board, nr, nc, color):
        b = list(board)
        old = b[0]
        if old == color:
            return tuple(b), 0
        stack, seen = [0], {0}
        while stack:
            j = stack.pop()
            b[j] = color
            a, c = divmod(j, nc)
            for x, y in ((a + 1, c), (a - 1, c), (a, c + 1), (a, c - 1)):
                if 0 <= x < nr and 0 <= y < nc:
                    q = x * nc + y
                    if q not in seen and board[q] == old:
                        seen.add(q)
                        stack.append(q)
        return tuple(b), len(seen)

    def region(self, b, nr, nc):
        return self.flood(b, nr, nc, -1)[1]

    def greedy(self, b, nr, nc):
        moves = []
        while len(set(b)) > 1 and len(moves) < 60:
            best = None
            for c in set(b) - {b[0]}:
                b1, _ = self.flood(b, nr, nc, c)
                sc = self.region(b1, nr, nc)
                if len(set(b1)) == 1:
                    sc = 10 ** 6
                else:
                    sc = sc * 100 + max(self.region(self.flood(b1, nr, nc, c2)[0], nr, nc) for c2 in set(b1) - {c})
                if best is None or sc > best[0]:
                    best = (sc, c, b1)
            moves.append(best[1])
            b = best[2]
        return moves

    def gen_level(self, rng, i, n):
        nr = nc = min(4 + i, 10) if not self.official else min(6 + i, 12)
        t = self.t
        while t > 1 and nr * (t + self.gap) > self.px_budget - 8:
            t -= 1
        B = Board(nr, nc, t, self.gap, 1, 1)
        sw = Board(1, self.k, max(t, 2), 1, 1 + B.h + 2, 1)
        shape = (1 + B.h + 2 + sw.h + 1, max(B.w, sw.w) + 2)
        b = tuple(int(v) for v in rng.integers(0, self.k, nr * nc))
        moves = self.greedy(b, nr, nc)
        if len(set(b)) == 1:
            return None
        lvl = dict(B=B, sw=sw, shape=shape, nr=nr, nc=nc, h0=len(moves))
        sol, cur = [], b
        for c in moves:
            sol.append(("click", self.target_px(lvl, cur, c)))
            cur, _ = self.flood(cur, nr, nc, c)
        return lvl, b, sol

    def target_px(self, lvl, b, c):
        if self.board_click:
            return lvl["B"].px(b.index(c))
        return lvl["sw"].px(c)

    def apply(self, lvl, st, op, px):
        if self.board_click:
            k = lvl["B"].cell(px)
            c = st[k] if k is not None else None
        else:
            c = lvl["sw"].cell(px)
        if c is None:
            return st, False
        return self.flood(st, lvl["nr"], lvl["nc"], c)[0], False

    def is_win(self, lvl, st):
        return len(set(st)) == 1

    def click_targets(self, lvl, st):
        return [self.target_px(lvl, st, c) for c in set(st)] if self.board_click else \
            [lvl["sw"].px(c) for c in range(self.k)]

    exact_progress = False

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        if st2 != st:
            ev[EV["color"]] = 1

    def heuristic(self, lvl, st):
        n = len(st)
        if len(set(st)) == 1:
            return 0.0
        return max(len(set(st)) - 1, (n - self.region(st, lvl["nr"], lvl["nc"])) / n * lvl["h0"])

    def render(self, lvl, st):
        g = np.full(lvl["shape"], BG, np.int64)
        lvl["B"].draw(g, st, lambda v: 4 + v)
        g[lvl["B"].r0, lvl["B"].c0 - 1] = MARK                        # repère du coin de départ
        if not self.board_click:
            lvl["sw"].draw(g, list(range(self.k)), lambda v: 4 + v)
        return g

    def min_len(self, i, n):
        return 3 + i // 2


# ============================================================================ 15. sélectionner une pièce (clic) puis la déplacer (flèches)
class SelectMove(AvatarGame):
    family = "select_move"
    simple_ops = DIR_OPS
    layouts = ("room",)
    t_choices = (1, 2, 3)

    def extra_setup(self, rng):
        self.uses_click = True
        self.density = float(rng.uniform(0.05, 0.2))
        self.tgt_style = str(rng.choice(["corners", "ring", "dot"]))

    def gen_level(self, rng, i, n):
        h = w = self.cap(min(5 + i + int(rng.integers(0, 2)), 11))
        walls = connected_room(rng, h, w, self.density)
        fr = free_cells(walls)
        k = 1 + (i >= 2) + (i >= 4)
        if len(fr) < 2 * k + 4:
            return None
        cells = [fr[j] for j in rng.choice(len(fr), 2 * k, replace=False)]
        lvl = self.mk(walls, targets=tuple(cells[k:]))
        st0 = (tuple(cells[:k]), -1)
        best = None
        for order in permutations(range(k)):
            sol = self.seq_solve(lvl, st0, order)
            if sol is not None and (best is None or len(sol) < len(best)):
                best = sol
        return (lvl, st0, best) if best is not None else None

    def seq_solve(self, lvl, st0, order):
        """une pièce après l'autre (BFS par pièce, les autres = obstacles) : solution valide, quasi optimale."""
        sol, st = [], st0
        for j in order:
            if st[0][j] == lvl["targets"][j]:
                continue
            sol.append(("click", self.cell_px(lvl, st[0][j])))
            st = (st[0], j)
            sub = Game.solve(_Single(self, lvl, j), lvl, st)
            if sub is None:
                return None
            for mv in sub:
                st, _ = self.apply(lvl, st, *mv)
            sol += sub
        return sol if self.is_win(lvl, st) else None

    def cell_px(self, lvl, p):
        t = lvl["t"]
        return (p[0] * t + t // 2, p[1] * t + t // 2)

    def apply(self, lvl, st, op, px):
        pos, sel = st
        if op == "click":
            p = (px[0] // lvl["t"], px[1] // lvl["t"])
            return ((pos, pos.index(p)) if p in pos else st), False
        if sel < 0:
            return st, False
        q = self.nxt(lvl, pos[sel], op)
        if q is None or q in pos:
            return st, False
        return (pos[:sel] + (q,) + pos[sel + 1:], sel), False

    def is_win(self, lvl, st):
        return st[0] == lvl["targets"]

    def click_targets(self, lvl, st):
        return [self.cell_px(lvl, p) for p in st[0]]

    def body(self, lvl, st):
        if st[1] < 0:
            return None
        p = st[0][st[1]]
        return (p[0], p[1], lvl["H"], lvl["W"])

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        if st2[1] != st[1]:
            ev[EV["toggle"]] = 1
        if op in DIRS and st[1] >= 0:
            q = self.nxt(lvl, st[0][st[1]], op)
            if q in st[0]:
                ev[EV["touch"]] = 1

    def heuristic(self, lvl, st):
        if "_td" not in lvl:
            lvl["_td"] = [bfs_dist(lvl["walls"], t) for t in lvl["targets"]]
        return sum(max(0, d[p]) for d, p in zip(lvl["_td"], st[0]))

    def table_ok(self, lvl, st0):
        return len(st0[0]) <= 2

    def render(self, lvl, st):
        cv = self.canvas(lvl)
        t = lvl["t"]
        for j, p in enumerate(lvl["targets"]):
            cv.put(*p, (4 + j) if t > 1 else (8 + j), self.tgt_style)
        for j, p in enumerate(st[0]):
            sel = j == st[1]
            white = t == 1 or (self.official and self.sel_style == "white")
            cv.put(*p, MARK if (sel and white) else 4 + j, "full")
            if sel and not white:
                cv.put(*p, MARK, "dot")
        return cv.g

    def min_len(self, i, n):
        return 3 + 2 * i


class _Single:
    """vue « une seule pièce, sélection fixée » d'un SelectMove/RotateFit pour le BFS par pièce."""

    def __init__(self, game, lvl, j):
        self.g, self.j = game, j
        self.simple_ops = game.piece_ops if hasattr(game, "piece_ops") else DIR_OPS
        self.uses_click = False
        self.bfs_cap = 20000

    def moves(self, lvl, st):
        return [(o, None) for o in self.simple_ops]

    def apply(self, lvl, st, op, px):
        return self.g.apply(lvl, st, op, px)

    def is_win(self, lvl, st):
        return self.g.piece_done(lvl, st, self.j) if hasattr(self.g, "piece_done") else \
            st[0][self.j] == lvl["targets"][self.j]


# ============================================================================ 16. tourner (ACTION5) et pousser une pièce dans sa silhouette
class RotateFit(AvatarGame):
    family = "rotate_fit"
    simple_ops = DIR_OPS + ("interact",)
    piece_ops = DIR_OPS + ("interact",)
    layouts = ("room",)

    def extra_setup(self, rng):
        self.multi = bool(rng.random() < 0.5)
        self.uses_click = self.multi
        self.density = float(rng.uniform(0.0, 0.1))
        self.tgt_style = str(rng.choice(["corners", "dot", "ring"]))

    @staticmethod
    def rots(shape):
        out, s = [], shape
        for _ in range(4):
            mr, mc = min(r for r, c in s), min(c for r, c in s)
            s = tuple(sorted((r - mr, c - mc) for r, c in s))
            out.append(s)
            s = tuple((c, -r) for r, c in s)
        return out

    @staticmethod
    def polyomino(rng, size):
        cells = {(0, 0)}
        while len(cells) < size:
            r, c = list(cells)[int(rng.integers(len(cells)))]
            dr, dc = list(DIRS.values())[int(rng.integers(4))]
            cells.add((r + dr, c + dc))
        return tuple(sorted(cells))

    def cells(self, lvl, j, pos, rot):
        return frozenset((pos[0] + dr, pos[1] + dc) for dr, dc in lvl["shapes"][j][rot])

    def fits(self, lvl, cells, other):
        for r, c in cells:
            if not (0 <= r < lvl["H"] and 0 <= c < lvl["W"]) or lvl["walls"][r, c] or (r, c) in other:
                return False
        return True

    def gen_level(self, rng, i, n):
        h = w = self.cap(min(6 + i + int(rng.integers(0, 2)), 11))
        walls = connected_room(rng, h, w, self.density)
        k = 2 if (self.multi and i >= 2) else 1
        size = min(3 + i // 2, 5)
        lvl = self.mk(walls, shapes=[], targets=[])
        fr = free_cells(walls)
        pieces = []
        occupied = set()
        for j in range(k):
            sh = self.rots(self.polyomino(rng, size))
            lvl["shapes"].append(sh)
            ok = False
            for _ in range(50):
                tp, tr = fr[int(rng.integers(len(fr)))], int(rng.integers(4))
                tc = self.cells(lvl, j, tp, tr)
                if self.fits(lvl, tc, set()):
                    ok = True
                    break
            if not ok:
                return None
            lvl["targets"].append(tc)
            ok = False
            for _ in range(50):
                pp, pr = fr[int(rng.integers(len(fr)))], int(rng.integers(4))
                pc = self.cells(lvl, j, pp, pr)
                if self.fits(lvl, pc, occupied) and pc != tc:
                    ok = True
                    break
            if not ok:
                return None
            occupied |= pc
            pieces.append((pp, pr))
        st0 = (tuple(pieces), -1 if self.multi else 0)
        best = None
        for order in permutations(range(k)):
            sol = self.seq_solve(lvl, st0, order)
            if sol is not None and (best is None or len(sol) < len(best)):
                best = sol
        return (lvl, st0, best) if best is not None else None

    def seq_solve(self, lvl, st0, order):
        sol, st = [], st0
        for j in order:
            if self.piece_done(lvl, st, j):
                continue
            if self.multi:
                sol.append(("click", self.piece_px(lvl, st, j)))
            st = (st[0], j)
            sub = Game.solve(_Single(self, lvl, j), lvl, st)
            if sub is None:
                return None
            for mv in sub:
                st, _ = self.apply(lvl, st, *mv)
            sol += sub
        return sol if self.is_win(lvl, st) else None

    def piece_done(self, lvl, st, j):
        pos, rot = st[0][j]
        return self.cells(lvl, j, pos, rot) == lvl["targets"][j]

    def piece_px(self, lvl, st, j):
        pos, rot = st[0][j]
        r, c = sorted(self.cells(lvl, j, pos, rot))[0]
        t = lvl["t"]
        return (r * t + t // 2, c * t + t // 2)

    def apply(self, lvl, st, op, px):
        pieces, sel = st
        if op == "click":
            p = (px[0] // lvl["t"], px[1] // lvl["t"])
            for j, (pos, rot) in enumerate(pieces):
                if p in self.cells(lvl, j, pos, rot):
                    return (pieces, j), False
            return st, False
        if sel < 0:
            return st, False
        pos, rot = pieces[sel]
        if op == "interact":
            pos2, rot2 = pos, (rot + 1) % 4
        else:
            dr, dc = DIRS[op]
            pos2, rot2 = (pos[0] + dr, pos[1] + dc), rot
        other = set()
        for j, (q, r) in enumerate(pieces):
            if j != sel:
                other |= self.cells(lvl, j, q, r)
        if not self.fits(lvl, self.cells(lvl, sel, pos2, rot2), other):
            return st, False
        return (pieces[:sel] + ((pos2, rot2),) + pieces[sel + 1:], sel), False

    def is_win(self, lvl, st):
        return all(self.piece_done(lvl, st, j) for j in range(len(st[0])))

    def click_targets(self, lvl, st):
        return [self.piece_px(lvl, st, j) for j in range(len(st[0]))]

    def body(self, lvl, st):
        if st[1] < 0:
            return None
        pos, rot = st[0][st[1]]
        cs = self.cells(lvl, st[1], pos, rot)
        return (sum(r for r, c in cs) / len(cs), sum(c for r, c in cs) / len(cs), lvl["H"], lvl["W"])

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        if st2[1] != st[1]:
            ev[EV["toggle"]] = 1
        if op == "interact" and st2 == st and st[1] >= 0:
            ev[EV["bump"]] = 1

    def table_ok(self, lvl, st0):
        return len(st0[0]) == 1

    def heuristic(self, lvl, st):
        h = 0.0
        for j, (pos, rot) in enumerate(st[0]):
            cs, tc = self.cells(lvl, j, pos, rot), lvl["targets"][j]
            if cs != tc:
                h += abs(sum(r for r, c in cs) / len(cs) - sum(r for r, c in tc) / len(tc)) + \
                    abs(sum(c for r, c in cs) / len(cs) - sum(c for r, c in tc) / len(tc)) + 1
        return h

    def render(self, lvl, st):
        cv = self.canvas(lvl)
        t = lvl["t"]
        for j, tc in enumerate(lvl["targets"]):
            for p in tc:
                cv.put(*p, (4 + j) if t > 1 else (GOAL if j == 0 else 8 + j), self.tgt_style)
        for j, (pos, rot) in enumerate(st[0]):
            sel = self.multi and j == st[1]
            white = t == 1 or (self.official and self.sel_style == "white")
            for p in self.cells(lvl, j, pos, rot):
                cv.put(*p, MARK if (sel and white) else 4 + j, "full")
                if sel and not white:
                    cv.put(*p, MARK, "dot")
        return cv.g

    def min_len(self, i, n):
        return 3 + i
