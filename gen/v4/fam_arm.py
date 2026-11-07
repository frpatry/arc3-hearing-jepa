"""arm4 (objets, bras télescopique) : un BRAS glisse sur un rail (colonne de gauche) et s'allonge vers la droite ;
quand sa tête entre dans un bloc, le bloc s'ACCROCHE au bras (il quitte le plateau et s'ajoute à la chaîne portée).
But : la chaîne portée doit reproduire, dans l'ordre, la SÉQUENCE DE RÉFÉRENCE du panneau du bas (la chaîne actuelle
est dessinée juste en dessous), puis le bras replié doit revenir au QUAI (case pleine sur le rail). Un bloc pris trop tôt ou un leurre gâche la chaîne (il faut recommencer, ou le
rendre quand la règle `drop` existe). L'ordre de prise dépend de qui cache qui sur chaque rangée.
Variantes : `fold` (le bras ne change de rangée que replié) ou `rigid` (il change de rangée déplié, mais ses segments
ne peuvent pas traverser un bloc).
Règles ajoutées (ordre tiré par jeu, une par niveau) :
  fixed : blocs fixes (gris, croix) qui arrêtent la tête ;
  long  : séquence plus longue ;
  decoy : blocs leurres d'une couleur absente de la séquence ;
  drop  : ACTION5 rend le dernier bloc pris (il retourne à sa place, si le bras ne la couvre pas) ;
  glue  : paire collée (trait) : prendre le bloc de gauche emporte aussi celui de droite.
Solvabilité : BFS exact (petit espace d'états) ; la séquence demandée = la plus lointaine atteignable."""
from collections import deque

import numpy as np

from ..core import EV
from . import draw
from .base import V4Game, mods_for
from .draw import R, blit, frame_rect, rect, sprite


class Arm4(V4Game):
    family = "arm4"
    mechanic = "object"
    principle = ("bras télescopique qui accroche des blocs ; la chaîne portée doit égaler la séquence de référence "
                 "du panneau (ordre de prise, leurres, dépose)", ["sk48", "sb26", "s5i5"])
    simple_ops = ("up", "down", "left", "right", "interact")
    CAT = ("fixed", "long", "decoy", "drop", "glue")
    exact_progress = True
    table_cap = 25000
    n_levels_official = (6, 8)

    def setup(self, rng):
        r = self.orng
        self.free_miss = bool(rng.random() < 0.5)
        self.variant = str(rng.choice(["fold", "rigid"]))
        self.cat = [str(x) for x in rng.permutation(self.CAT)]
        self.C = int(r.choice([5, 5, 6])) if self.s_screen == 1 else 3
        self.block_kind = str(r.choice(["box", "framed", "box"]))
        self.ph = 2 * self.C + 5

    # ------------------------------------------------------------------ règles
    def _loose(self, lvl, car):
        return {(bk[0], bk[1]): j for j, bk in enumerate(lvl["blocks"]) if j not in car}

    def _cols(self, lvl, car):
        return tuple(lvl["blocks"][j][2] for j in car)

    def _step(self, lvl, st, op):
        r, l, car = st
        N, M = lvl["N"], lvl["M"]
        bpos = self._loose(lvl, car)
        fixed = lvl["fixed"]
        blocks = lvl["blocks"]
        if op == "interact":                                   # rendre le dernier bloc pris à sa place
            if not lvl["drop"] or not car:
                return st, set()
            back = [car[-1]]
            if len(car) >= 2 and blocks[car[-2]][3] == car[-1]:
                back = [car[-2], car[-1]]
            if any(blocks[j][0] == r and blocks[j][1] <= l for j in back):
                return st, {"bump"}
            return (r, l, car[:-len(back)]), {"drop"}
        if op == "right":
            q = (r, l + 1)
            if q[1] > M or q in fixed:
                return st, {"bump"}
            if q in bpos:
                j = bpos[q]
                take = [j] + ([blocks[j][3]] if blocks[j][3] >= 0 and blocks[j][3] not in car else [])
                if len(car) + len(take) > lvl["cap"]:
                    return st, {"bump"}
                return (r, l + 1, car + tuple(take)), {"take", "move"}
            return (r, l + 1, car), {"move"}
        if op == "left":
            if l == 0:
                return st, {"bump"}
            return (r, l - 1, car), {"move"}
        d = -1 if op == "up" else 1
        nr = r + d
        if not (0 <= nr < N):
            return st, {"bump"}
        if self.variant == "fold":
            if l > 0:
                return st, {"bump"}
        else:
            if any((nr, c) in bpos or (nr, c) in fixed for c in range(1, l + 1)):
                return st, {"bump"}
        return (nr, l, car), {"move"}

    def apply(self, lvl, st, op, px):
        return self._step(lvl, st, op)[0], False

    def is_win(self, lvl, st):
        return st[0] == lvl["dock"] and st[1] == 0 and self._cols(lvl, st[2]) == lvl["seq"]

    def moves(self, lvl, st):
        mv = [(o, None) for o in ("up", "down", "left", "right")]
        if lvl["drop"]:
            mv.append(("interact", None))
        return mv

    # ------------------------------------------------------------------ niveaux
    def gen_level(self, rng, i, n):
        mods, newest = mods_for(self.cat, i)
        C = self.C
        Mmax = (self.AW - 2 * C) // C
        Nmax = (self.AH - self.ph - 3) // C
        N = int(min(Nmax, 4 + (i + 1) // 2 + int(rng.integers(0, 2))))
        M = int(min(Mmax, 6 + i // 2 + int(rng.integers(0, 2))))
        k = min(2 + (i >= 1) + (i >= 4) + ("long" in mods), 4)
        ndec = (1 + (i >= 5)) if "decoy" in mods else 0
        W = (M + 1) * C
        oy = 1 + (self.AH - self.ph - 2 - N * C) // 2
        ox = (self.AW - W) // 2
        lvl = dict(N=N, M=M, oy=oy, ox=ox, mods=mods, drop="drop" in mods, cap=k, seq=(), fixed=frozenset(),
                   dock=int(rng.integers(N)))
        cells = [(r, c) for r in range(N) for c in range(1, M + 1)]
        rng.shuffle(cells)
        fixed = set()
        if "fixed" in mods:
            fixed = set(cells[:2 + (i >= 4)])
            cells = cells[2 + (i >= 4):]
        lvl["fixed"] = frozenset(fixed)
        cols = [int(x) for x in rng.permutation(6)[:k]]
        bl = []
        if "glue" in mods:
            pairs = [q for q in cells if (q[0], q[1] + 1) in cells]
            if not pairs:
                return self.why("glue")
            a = pairs[0]
            b = (a[0], a[1] + 1)
            bl += [(a[0], a[1], cols[0], 1), (b[0], b[1], cols[1], -1)]
            cells = [q for q in cells if q not in (a, b)]
        # les autres blocs de préférence en files (une rangée cache l'autre)
        rows = [int(x) for x in rng.permutation(N)]
        rest = cols[len(bl):] + [6 + j for j in range(ndec)]
        for j, ccol in enumerate(rest):
            rr = rows[j % max(1, min(N, 2 + i // 3))]
            cand = [q for q in cells if q[0] == rr and q[1] > M // 2] or [q for q in cells if q[0] == rr] or cells
            q = cand[int(rng.integers(len(cand)))]
            cells.remove(q)
            bl.append((q[0], q[1], ccol, -1))
        lvl["blocks"] = tuple(bl)
        st0 = (int(rng.integers(N)), 0, ())
        depth, prev, q = {st0: 0}, {st0: None}, deque([st0])
        best = {}
        want = set(cols)
        mv = self.moves(lvl, st0)
        while q and len(depth) < 40000:
            s = q.popleft()
            car = self._cols(lvl, s[2])
            if len(car) == k and set(car) == want and s[0] == lvl["dock"] and s[1] == 0:
                if car not in best:
                    best[car] = s
                continue
            for m_ in mv:
                s2, _ = self._step(lvl, s, m_[0])
                if s2 not in depth:
                    depth[s2] = depth[s] + 1
                    prev[s2] = (s, m_)
                    q.append(s2)
        if q:
            return self.why("cap")
        if not best:
            return self.why("noseq")
        dm = max(depth[s] for s in best.values())
        far = [key for key, s in best.items() if depth[s] >= 0.85 * dm]
        key = far[int(rng.integers(len(far)))]
        lvl["seq"] = key
        s = best[key]
        sol = []
        while prev[s] is not None:
            s, m_ = prev[s]
            sol.append(m_)
        sol = sol[::-1]
        if lvl["drop"] and not any(m_[0] == "interact" for m_ in sol) and rng.random() < 0.7:
            return self.why("drop-unused")
        return lvl, st0, sol

    def min_len(self, i, n):
        return min(16 + 6 * i, 55)

    def heuristic(self, lvl, st):
        car, seq = self._cols(lvl, st[2]), lvl["seq"]
        good = 0
        while good < len(car) and good < len(seq) and car[good] == seq[good]:
            good += 1
        return 6 * (len(seq) - good) + 3 * (len(car) - good) + abs(st[0] - lvl["dock"]) + st[1]

    def body(self, lvl, st):
        return (st[0], st[1], lvl["N"], lvl["M"] + 1)

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        _, tags = self._step(lvl, st, op) if op != "click" else (st, set())
        if "take" in tags:
            ev[EV["collect"]] = ev[EV["touch"]] = 1
        if "drop" in tags:
            ev[EV["push"]] = 1
        if "bump" in tags:
            ev[EV["touch"]] = 1
        if "take" in tags and not lvl["seq"][:len(st2[2])] == self._cols(lvl, st2[2]):
            ev[EV["danger"]] = 1

    # ------------------------------------------------------------------ rendu
    def _blk(self, col, s):
        pal = self.pal
        return sprite(None, self.block_kind if s >= 3 else "box", s, s, pal.a(col), pal.fg)

    def render(self, lvl, st):
        pal, C = self.pal, self.C
        g = self.blank()
        N, M, oy, ox = lvl["N"], lvl["M"], lvl["oy"], lvl["ox"]
        frame_rect(g, oy - 1, ox + C - 1, N * C + 2, M * C + 2, pal.room)       # plateau : cadre seul (fond dominant)
        draw.vline(g, ox + C // 2, oy, oy + N * C - 1, pal.line)
        rect(g, oy + lvl["dock"] * C + 1, ox, C - 1, C - 1, pal.line)            # quai : le bras doit y revenir
        for (r, c) in lvl["fixed"]:
            y, x = oy + r * C, ox + c * C
            rect(g, y + 1, x + 1, C - 1, C - 1, pal.wall)
            for t in range(1, C):
                g[y + t, x + t] = R(pal.bg)
        r0, l, carj = st
        car = self._cols(lvl, carj)
        loose = [bk for j, bk in enumerate(lvl["blocks"]) if j not in carj]
        bset = {(b[0], b[1]) for b in loose}
        for (r, c, col, glue) in loose:
            y, x = oy + r * C, ox + c * C
            blit(g, self._blk(col, C - 1), y + 1, x + 1)
            if glue >= 0 and (r, c + 1) in bset:
                rect(g, y + C // 2, x + C - 1, 1, 3, pal.fg)
        head_c = draw.MAGENTA if draw.MAGENTA not in (pal.bg, pal.room) else draw.ORANGE
        for c in range(0, l):
            y, x = oy + r0 * C, ox + c * C
            rect(g, y + C // 2 - 1, x + (1 if c == 0 else 0), 2, C, head_c)
        y, x = oy + r0 * C, ox + l * C
        frame_rect(g, y, x, C, C, head_c)
        if car and C >= 5:
            rect(g, y + 2, x + 2, C - 4, C - 4, pal.a(car[-1]))
        # panneau : séquence de référence (haut) et chaîne portée (bas)
        py = self.AH - self.ph
        draw.hline(g, py, 0, self.AW - 1, pal.line)
        s = C - 1
        for row, seq in ((0, lvl["seq"]), (1, car)):
            yy = py + 2 + row * (s + 2)
            frame_rect(g, yy, 2, s, s, head_c if row == 0 else pal.line)
            x = 2 + s
            for j, col in enumerate(seq):
                rect(g, yy + s // 2, x, 1, 2, head_c)
                x += 2
                blit(g, self._blk(col, s), yy, x)
                x += s
            if row == 1:
                for j in range(len(car), len(lvl["seq"])):
                    rect(g, yy + s // 2, x, 1, 2, pal.line)
                    x += 2
                    frame_rect(g, yy, x, s, s, pal.line)
                    x += s
        return g
