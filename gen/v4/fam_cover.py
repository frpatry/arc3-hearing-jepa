"""cover4 (objets, couverture dérivée) : recouvrir une SILHOUETTE-cible avec des polyominos ET leurs reflets à travers
des axes-miroirs. On choisit une pièce ou un axe, les flèches le déplacent d'une case (un axe vertical ne bouge qu'à
gauche/droite, un horizontal qu'en haut/bas) ; la couverture = pièces + reflets (gris) ; gagné quand toute la
silhouette est couverte (et aucune case interdite).
Variantes : `click` (clic = choisir, ACTION5 = tourner la pièce choisie quand la rotation existe) ou `cycle`
(ACTION5 = choisir la suivante, clic SUR la pièce choisie = la tourner ; les autres clics sont sans effet).
Règles ajoutées (ordre tiré par jeu, une par niveau) :
  mmove  : l'axe devient mobile (il faut le replacer : la symétrie de la silhouette le trahit) ;
  axis2  : un 2e axe perpendiculaire (4 copies : une pièce reflétée deux fois) ;
  rotate : pièces tournées au départ, rotation d'un quart de tour ;
  forbid : cases interdites (croix) que la couverture ne doit pas toucher ;
  half   : l'axe principal devient un miroir à un seul côté (seules les cases du côté marqué sont reflétées) ;
  more   : une pièce de plus.
Solvabilité : configuration-solution construite (copies disjointes), silhouette = sa couverture, puis dispersion ;
solution = choisir / tourner / déplacer chaque objet (plan direct, les pièces se chevauchent librement)."""
import numpy as np

from ..core import EV
from . import draw
from .base import V4Game, mods_for
from .draw import R, frame_rect, rect
from .selkit import DV
from functools import lru_cache


def _norm(cells):
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


@lru_cache(maxsize=None)
def _rot_shape(shape, k):
    s = shape
    for _ in range(k % 4):
        s = _norm([(c, -r) for r, c in s])
    return s


def _polyomino(rng, n):
    cells = {(0, 0)}
    while len(cells) < n:
        r, c = list(cells)[int(rng.integers(len(cells)))]
        dr, dc = list(DV.values())[int(rng.integers(4))]
        cells.add((r + dr, c + dc))
    return _norm(cells)


class Cover4(V4Game):
    family = "cover4"
    mechanic = "object"
    principle = ("couverture dérivée : pièces + reflets à travers des axes-miroirs mobiles doivent couvrir une "
                 "silhouette affichée", ["ar25", "m0r0", "cn04"])
    simple_ops = ("up", "down", "left", "right", "interact")
    CAT = ("mmove", "axis2", "rotate", "forbid", "half", "more")
    exact_progress = False
    n_levels_official = (6, 8)

    def setup(self, rng):
        r = self.orng
        self.uses_click = True
        self.variant = str(rng.choice(["click", "click", "cycle"]))
        self.free_miss = bool(rng.random() < 0.6)
        self.cat = [str(x) for x in rng.permutation(self.CAT)]
        self.axis0 = str(rng.choice(["v", "h"]))
        self.C = int(r.choice([3, 3, 4])) if self.s_screen == 1 else 2
        self.dotted = int(r.choice([0, 0, 0, 2]))
        self.cell_gap = False

    # ------------------------------------------------------------------ géométrie
    def _pcells(self, lvl, k, p):
        r, c, rot = p
        return [(r + a, c + b) for a, b in _rot_shape(lvl["shapes"][k], rot)]

    def _inside(self, lvl, cells):
        N, M = lvl["N"], lvl["M"]
        return all(0 <= r < N and 0 <= c < M for r, c in cells)

    def _free(self, lvl, pieces, k, q):
        cells = self._pcells(lvl, k, q)
        if not self._inside(lvl, cells):
            return False
        cs = set(cells)
        if cs & lvl["lcells"]:
            return False
        return not any(cs & set(self._pcells(lvl, j, pieces[j])) for j in range(len(pieces)) if j != k)

    def _reflect(self, lvl, cells, mirrors):
        N, M = lvl["N"], lvl["M"]
        cov = set(cells)
        for _ in range(3):
            new = set(cov)
            for j, (kind, side) in enumerate(lvl["axes"]):
                a = mirrors[j]
                for (r, c) in cov:
                    if kind == "v":
                        if side and (c - a) * side <= 0:
                            continue
                        q = (r, 2 * a - c)
                    else:
                        if side and (r - a) * side <= 0:
                            continue
                        q = (2 * a - r, c)
                    if 0 <= q[0] < N and 0 <= q[1] < M:
                        new.add(q)
            if new == cov:
                break
            cov = new
        return cov

    def _coverage(self, lvl, st):
        pieces, mirrors, _ = st
        own = set()
        for k, p in enumerate(pieces):
            own |= set(self._pcells(lvl, k, p))
        own |= lvl["lcells"]
        return own, self._reflect(lvl, own, mirrors)

    def is_win(self, lvl, st):
        _, cov = self._coverage(lvl, st)
        return lvl["target"] <= cov and not (cov & lvl["forbid"])

    def _clamp(self, lvl, k, p):
        cells = _rot_shape(lvl["shapes"][k], p[2])
        h = max(a for a, _ in cells) + 1
        w = max(b for _, b in cells) + 1
        return (min(max(p[0], 0), lvl["N"] - h), min(max(p[1], 0), lvl["M"] - w), p[2])

    # ------------------------------------------------------------------ objets sous un clic
    def _cell(self, lvl, px):
        if px is None:
            return None
        r, c = (px[0] - lvl["oy"]) // self.C, (px[1] - lvl["ox"]) // self.C
        if 0 <= r < lvl["N"] and 0 <= c < lvl["M"] and px[0] >= lvl["oy"] and px[1] >= lvl["ox"]:
            return (int(r), int(c))
        return None

    def _objs_at(self, lvl, st, cell):
        pieces, mirrors, _ = st
        out = [k for k in range(len(pieces) - 1, -1, -1) if cell in self._pcells(lvl, k, pieces[k])]
        np_ = len(pieces)
        for j, (kind, _) in enumerate(lvl["axes"]):
            if lvl["mov"][j] and ((kind == "v" and cell[1] == mirrors[j]) or (kind == "h" and cell[0] == mirrors[j])):
                out.append(np_ + j)
        return out

    def _resolve_click(self, lvl, st, cell):
        objs = self._objs_at(lvl, st, cell)
        if not objs:
            return None
        sel = st[2]
        if sel in objs:
            return objs[(objs.index(sel) + 1) % len(objs)]
        return objs[0]

    def _movables(self, lvl, st):
        return len(st[0]) + sum(lvl["mov"])

    # ------------------------------------------------------------------ règles
    def _rotate(self, lvl, st):
        pieces, mirrors, sel = st
        if not lvl["rot"] or sel < 0 or sel >= len(pieces):
            return st
        p = pieces[sel]
        q = self._clamp(lvl, sel, (p[0], p[1], (p[2] + 1) % 4))
        if set(self._pcells(lvl, sel, q)) == set(self._pcells(lvl, sel, p)) or not self._free(lvl, pieces, sel, q):
            return st
        return (pieces[:sel] + (q,) + pieces[sel + 1:], mirrors, sel)

    def _mirror_index(self, lvl, j):
        """j-ième objet mobile après les pièces -> indice d'axe."""
        idx = [a for a in range(len(lvl["axes"])) if lvl["mov"][a]]
        return idx[j]

    def apply(self, lvl, st, op, px):
        pieces, mirrors, sel = st
        if op == "click":
            cell = self._cell(lvl, px)
            if cell is None:
                return st, False
            if self.variant == "cycle":
                if 0 <= sel < len(pieces) and cell in self._pcells(lvl, sel, pieces[sel]):
                    return self._rotate(lvl, st), False
                return st, False
            k = self._resolve_click(lvl, st, cell)
            if k is None and sel >= 0 and cell not in lvl["target"]:
                return (pieces, mirrors, -1), False                    # clic dans le vide : plus rien de choisi
            if k is None or k == sel:
                return st, False
            return (pieces, mirrors, k), False
        if op == "interact":
            if self.variant == "cycle":
                n = self._movables(lvl, st)
                return ((pieces, mirrors, (sel + 1) % n), False) if n > 1 else (st, False)
            return self._rotate(lvl, st), False
        if sel < 0:
            return st, False
        dr, dc = DV[op]
        if sel < len(pieces):
            p = pieces[sel]
            q = (p[0] + dr, p[1] + dc, p[2])
            if not self._free(lvl, pieces, sel, q):
                return st, False
            return (pieces[:sel] + (q,) + pieces[sel + 1:], mirrors, sel), False
        a = self._mirror_index(lvl, sel - len(pieces))
        kind = lvl["axes"][a][0]
        d = dc if kind == "v" else dr
        lim = lvl["M"] if kind == "v" else lvl["N"]
        v = mirrors[a] + d
        if d == 0 or not (1 <= v <= lim - 2):
            return st, False
        return (pieces, mirrors[:a] + (v,) + mirrors[a + 1:], sel), False

    # ------------------------------------------------------------------ planification
    def _click_px(self, lvl, st, j):
        """pixel à cliquer pour choisir l'objet j (variante click), ou None."""
        pieces, mirrors, _ = st
        if j < len(pieces):
            cells = self._pcells(lvl, j, pieces[j])
        else:
            a = self._mirror_index(lvl, j - len(pieces))
            kind = lvl["axes"][a][0]
            cells = [(r, mirrors[a]) for r in range(lvl["N"])] if kind == "v" else \
                [(mirrors[a], c) for c in range(lvl["M"])]
        for cell in cells:
            if self._resolve_click(lvl, st, cell) == j:
                return self._cpx(lvl, cell)
        return None

    def _cpx(self, lvl, cell):
        return (lvl["oy"] + cell[0] * self.C + self.C // 2, lvl["ox"] + cell[1] * self.C + self.C // 2)

    def _plan(self, lvl, st):
        pc = lvl.setdefault("_pc", {})
        if st not in pc:
            if len(pc) > 4000:
                pc.clear()
            pc[st] = self._plan0(lvl, st)
        return pc[st]

    def _plan0(self, lvl, st):
        pieces, mirrors, sel = st
        n_p = len(pieces)
        tp, tm = lvl["tp"], lvl["tm"]
        out = []
        n = self._movables(lvl, st)
        todo = []
        for j in range(n):
            if j < n_p:
                same = set(self._pcells(lvl, j, pieces[j])) == set(self._pcells(lvl, j, tp[j]))
            else:
                a = self._mirror_index(lvl, j - n_p)
                same = mirrors[a] == tm[a]
            if not same:
                todo.append(j)
        todo.sort(key=lambda j: (j != sel, j))
        from itertools import permutations
        for order in [todo] + [list(o) for o in permutations(todo)][1:12]:
            r = self._plan_order(lvl, st, order, n, n_p)
            if r is not None:
                return r
        return None

    def _plan_order(self, lvl, st, todo, n, n_p):
        tp, tm = lvl["tp"], lvl["tm"]
        out = []
        cur = st
        for j in todo:
            if cur[2] != j:                                          # choisir
                if self.variant == "cycle":
                    for _ in range((j - cur[2]) % n):
                        out.append(("interact", None))
                        cur, _ = self.apply(lvl, cur, "interact", None)
                else:
                    px = self._click_px(lvl, cur, j)
                    if px is None:
                        return None
                    out.append(("click", px))
                    cur, _ = self.apply(lvl, cur, "click", px)
                if cur[2] != j:
                    return None
            if j < n_p:
                ops = self._piece_path(lvl, cur, j, tp[j])
                if ops is None:
                    return None
            else:
                a = self._mirror_index(lvl, j - n_p)
                d = tm[a] - cur[1][a]
                if lvl["axes"][a][0] == "v":
                    ops = ["right"] * d if d > 0 else ["left"] * -d
                else:
                    ops = ["down"] * d if d > 0 else ["up"] * -d
            for o in ops:
                if o == "rot":
                    if self.variant == "cycle":
                        px = self._cpx(lvl, self._pcells(lvl, j, cur[0][j])[0])
                        out.append(("click", px))
                    else:
                        out.append(("interact", None))
                else:
                    out.append((o, None))
                cur, _ = self.apply(lvl, cur, *out[-1])
        if not self.is_win(lvl, cur):
            return None
        return out

    def _piece_path(self, lvl, st, j, goal):
        """BFS sur (r, c, rotation) de la pièce j (les autres immobiles) -> ops ("up"…, "rot") ou None."""
        from collections import deque
        pieces = st[0]
        tc = set(self._pcells(lvl, j, goal))
        s0 = pieces[j]
        if set(self._pcells(lvl, j, s0)) == tc:
            return []
        prev, q = {s0: None}, deque([s0])
        moves = list(DV.items()) + ([("rot", None)] if lvl["rot"] else [])
        while q:
            p = q.popleft()
            for name, d in moves:
                if d is None:
                    n = self._clamp(lvl, j, (p[0], p[1], (p[2] + 1) % 4))
                    if set(self._pcells(lvl, j, n)) == set(self._pcells(lvl, j, p)):
                        continue
                else:
                    n = (p[0] + d[0], p[1] + d[1], p[2])
                if n in prev or not self._free(lvl, pieces, j, n):
                    continue
                prev[n] = (p, name)
                if set(self._pcells(lvl, j, n)) == tc:
                    path = []
                    while prev[n] is not None:
                        n0, nm = prev[n]
                        path.append(nm)
                        n = n0
                    return path[::-1]
                q.append(n)
        return None

    def oracle_action(self):
        if self.state != "NOT_FINISHED":
            return 0, None
        L = self.levels[self.level]
        p = self._plan(L["lvl"], self.st)
        if not p:
            return 0, None
        return self._to_action(*p[0], self.st)

    def heuristic(self, lvl, st):
        p = self._plan(lvl, st)
        return 99.0 if p is None else float(len(p))

    def click_targets(self, lvl, st):
        return []

    def click_hit(self, lvl, st, px):
        cell = self._cell(lvl, px)
        return cell is not None and bool(self._objs_at(lvl, st, cell))

    # ------------------------------------------------------------------ niveaux
    def gen_level(self, rng, i, n):
        mods, newest = mods_for(self.cat, i)
        C = self.C
        Nmax, Mmax = self.AH // C, self.AW // C
        sh = max(0, 3 - i)
        N, M = max(10, Nmax - 2 * sh), max(10, Mmax - 2 * sh)
        oy, ox = (self.AH - N * C) // 2, (self.AW - M * C) // 2
        axes = [(self.axis0, 1 if "half" in mods else 0)]
        if "half" in mods:
            axes[0] = (self.axis0, int(rng.choice([-1, 1])))
        if "axis2" in mods:
            axes.append(("h" if self.axis0 == "v" else "v", 0))
        mov = tuple(("mmove" in mods) for _ in axes)
        n_p = (1 if i == 0 else 2) + ("more" in mods) + (i >= 4) + (i >= 6)
        lvl = dict(N=N, M=M, oy=oy, ox=ox, axes=tuple(axes), mov=mov, rot="rotate" in mods, mods=mods,
                   forbid=frozenset(), target=frozenset(), lcells=frozenset())
        n_lock = int(i >= 1 and rng.random() < 0.5)
        for _ in range(60):
            shapes = [_polyomino(rng, int(rng.integers(3, 6 + (i >= 3)))) for _ in range(n_p + n_lock)]
            lvl["shapes"] = shapes
            mirrors = []
            for kind, _s in axes:
                lim = M if kind == "v" else N
                mirrors.append(int(rng.integers(lim // 2 - lim // 5, lim // 2 + lim // 5 + 1)))
            mirrors = tuple(mirrors)
            pieces, covs, ok = [], set(), True
            for k in range(n_p + n_lock):
                placed = False
                for _t in range(60):
                    rot = int(rng.integers(4))
                    cells0 = _rot_shape(shapes[k], rot)
                    h = max(a for a, _ in cells0) + 1
                    w = max(b for _, b in cells0) + 1
                    p = (int(rng.integers(0, N - h + 1)), int(rng.integers(0, M - w + 1)), rot)
                    cells = self._pcells(lvl, k, p)
                    bad = False
                    for j, (kind, side) in enumerate(axes):
                        coord = [c if kind == "v" else r for r, c in cells]
                        if min(coord) <= mirrors[j] <= max(coord):
                            bad = True                                  # la pièce ne chevauche pas l'axe
                        if side and j == 0 and k == 0 and any((x - mirrors[j]) * side <= 0 for x in coord):
                            bad = True                                  # 1re pièce du côté reflété
                    if bad:
                        continue
                    cov = self._reflect(lvl, cells, mirrors)
                    nexp = len(cells) * (2 ** sum(1 for j, (kind, side) in enumerate(axes)
                                                  if not side or all((x - mirrors[j]) * side > 0 for x in
                                                                     [c if kind == "v" else r for r, c in cells])))
                    if len(cov) != nexp or cov & covs:
                        continue                                        # copie coupée par le bord, ou chevauchement
                    near = {(r + a, c + b) for r, c in cov for a in (-1, 0, 1) for b in (-1, 0, 1)}
                    if near & covs:
                        continue                                        # silhouettes séparées d'au moins une case
                    pieces.append(p)
                    covs |= cov
                    placed = True
                    break
                if not placed:
                    ok = False
                    break
            if ok:
                break
        else:
            return self.why("place")
        if n_lock:
            lvl["lcells"] = frozenset(self._pcells(lvl, n_p, pieces[n_p]))
            lvl["shapes"] = shapes[:n_p]
            pieces = pieces[:n_p]
        lvl["target"] = frozenset(covs)
        lvl["tp"], lvl["tm"] = tuple(pieces), mirrors
        if "forbid" in mods:
            ring = sorted({(r + a, c + b) for r, c in covs for a, b in DV.values()} - covs)
            ring = [q for q in ring if 0 <= q[0] < N and 0 <= q[1] < M]
            pick = rng.permutation(len(ring))[:int(rng.integers(2, 5))]
            lvl["forbid"] = frozenset(ring[j] for j in pick)
        # dispersion : on garde la plus longue de plusieurs dispersions
        best = None
        for _ in range(16):
            sp = []
            for k in range(n_p):
                rot = int(rng.integers(4)) if lvl["rot"] else pieces[k][2]
                for _t in range(30):
                    q = self._clamp(lvl, k, (int(rng.integers(0, N)), int(rng.integers(0, M)), rot))
                    if abs(q[0] - pieces[k][0]) + abs(q[1] - pieces[k][1]) < (N + M) // 3 and _t < 20:
                        continue
                    cs = set(self._pcells(lvl, k, q))
                    if not (cs & lvl["lcells"]) and not any(cs & set(self._pcells(lvl, j, sp[j])) for j in range(k)):
                        break
                else:
                    break
                sp.append(q)
            if len(sp) < n_p:
                continue
            sm = list(mirrors)
            for j in range(len(axes)):
                if mov[j]:
                    lim = M if axes[j][0] == "v" else N
                    cand = [v for v in range(1, lim - 1) if abs(v - mirrors[j]) >= 3]
                    sm[j] = int(cand[int(rng.integers(len(cand)))])
            if lvl["rot"] and all(sp[k][2] == pieces[k][2] for k in range(n_p)):
                continue
            st0 = (tuple(sp), tuple(sm), 0 if self.variant == "cycle" else -1)
            if self.is_win(lvl, st0):
                continue
            sol = self._plan(lvl, st0)
            if sol is not None and (best is None or len(sol) > len(best[1])):
                best = (st0, sol)
        if best is None:
            return self.why("scatter")
        return lvl, best[0], best[1]

    def min_len(self, i, n):
        return min(14 + 7 * i, 60)

    def body(self, lvl, st):
        pieces, mirrors, sel = st
        if sel < 0:
            return None
        if sel < len(pieces):
            p = pieces[sel]
            return (p[0], p[1], lvl["N"], lvl["M"])
        a = self._mirror_index(lvl, sel - len(pieces))
        v = mirrors[a]
        return (v, 0, lvl["N"], lvl["M"]) if lvl["axes"][a][0] == "h" else (0, v, lvl["N"], lvl["M"])

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        if st2[2] != st[2]:
            ev[EV["toggle"]] = 1
        if st2[0] != st[0] and op in ("interact", "click"):
            ev[EV["color"]] = 1                                         # rotation
        if op in DV and st2 == st and st[2] >= 0:
            ev[EV["touch"]] = 1
        _, c0 = self._coverage(lvl, st)
        _, c1 = self._coverage(lvl, st2)
        if len(c1 & lvl["target"]) > len(c0 & lvl["target"]):
            ev[EV["collect"]] = 1

    # ------------------------------------------------------------------ rendu
    def render(self, lvl, st):
        pal, C = self.pal, self.C
        g = self.blank()
        oy, ox = lvl["oy"], lvl["ox"]
        tgt_c = pal.a(0)
        piece_c = pal.fg if pal.fg not in (pal.bg, pal.room) else pal.a(1)
        refl_c = next(c for c in (draw.DGREY, draw.GREY, draw.LGREY, draw.OFFW, pal.a(2))
                      if c not in (pal.bg, pal.room, tgt_c, piece_c))
        sel_c = draw.WHITE if (self.sel_style == "white" and draw.WHITE not in (pal.bg, tgt_c, piece_c)) else None
        cs = C - (1 if self.cell_gap and C >= 3 else 0)

        def cell(r, c, col):
            rect(g, oy + r * C, ox + c * C, cs, cs, col)

        own, cov = self._coverage(lvl, st)
        for (r, c) in lvl["target"]:
            cell(r, c, tgt_c)
        for (r, c) in lvl["forbid"]:
            y, x = oy + r * C, ox + c * C
            for t in range(C):
                g[y + t, x + t] = R(draw.RED if pal.bg != draw.RED else draw.MAROON)
                g[y + t, x + C - 1 - t] = R(draw.RED if pal.bg != draw.RED else draw.MAROON)
        for (r, c) in cov - own:
            cell(r, c, refl_c)
        pieces, mirrors, sel = st
        n_p = len(pieces)
        for k, p in enumerate(pieces):
            col = sel_c if (k == sel and sel_c is not None) else piece_c
            cells = self._pcells(lvl, k, p)
            for (r, c) in cells:
                cell(r, c, col)
            if k == sel and sel_c is None:
                r, c = cells[0]
                g[oy + r * C + C // 2, ox + c * C + C // 2] = R(pal.a(3))
        for (r, c) in lvl["lcells"]:
            cell(r, c, piece_c)
            if C >= 3:
                g[oy + r * C + C // 2, ox + c * C + C // 2] = R(refl_c)
        mi = 0
        for j, (kind, side) in enumerate(lvl["axes"]):
            v = mirrors[j]
            is_sel = lvl["mov"][j] and sel == n_p + sum(lvl["mov"][:j])
            col = (draw.WHITE if pal.bg != draw.WHITE else draw.BLACK) if is_sel else pal.line
            if not lvl["mov"][j]:
                col = pal.line if not is_sel else col
            dot = 0 if is_sel else self.dotted
            if kind == "v":
                x = ox + v * C + C // 2
                draw.vline(g, x, oy, oy + lvl["N"] * C - 1, col, dotted=dot)
                if lvl["mov"][j]:
                    draw.vline(g, x - 1 if C >= 3 else x + 1, oy, oy + lvl["N"] * C - 1, col, dotted=dot)
                if side:
                    for y in range(oy + 1, oy + lvl["N"] * C, 4):
                        g[y, x + side] = R(col)
            else:
                y = oy + v * C + C // 2
                draw.hline(g, y, ox, ox + lvl["M"] * C - 1, col, dotted=dot)
                if lvl["mov"][j]:
                    draw.hline(g, y - 1 if C >= 3 else y + 1, ox, ox + lvl["M"] * C - 1, col, dotted=dot)
                if side:
                    for x in range(ox + 1, ox + lvl["M"] * C, 4):
                        g[y + side, x] = R(col)
            mi += 1
        return g
