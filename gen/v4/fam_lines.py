"""lines4 (objets, score composite) : des DESSINS AU TRAIT d'1 px (croix, L, T, cadre, X) glissent par pas de 2-3 px ;
but : chaque point-cible (petit cadre coloré) doit être traversé par un trait DE SA COULEUR (le dessin superposé
compte, pas une pièce seule). On choisit une pièce (ACTION5 = la suivante, ou clic sur son trait), les flèches la
déplacent ; la pièce choisie porte un pixel blanc en son centre.
Variantes : `cycle` (ACTION5 fait tourner la sélection, pas de clic) ou `click` (clic = choisir, clic dans le vide =
lâcher ; ACTION5 = tourner la pièce choisie d'un quart de tour, pièces tournées au départ).
Règles ajoutées (ordre tiré par jeu, une par niveau) :
  pads  : bacs de peinture : une pièce dont le centre s'arrête sur un bac prend sa couleur (pièces mal colorées au
          départ : il faut passer au bon bac AVANT la cible) ;
  pegs  : plots gris : aucun trait ne peut les chevaucher (déplacements bloqués, il faut contourner) ;
  third : une 3e pièce ;
  anti  : points interdits (quatre coins) qu'aucun trait ne doit traverser ;
  cross : un point bicolore, que DEUX traits de couleurs différentes doivent croiser ensemble.
Solvabilité : configuration-solution tirée (points posés sur ses traits), puis dispersion ; solution = BFS par pièce
sur (position, rotation, couleur) ; l'oracle replanifie depuis l'état courant."""
from collections import deque

import numpy as np

from ..core import EV
from . import draw
from .base import V4Game, mods_for
from .draw import R, rect
from .selkit import DV


def _rot(off, k):
    o = off
    for _ in range(k % 4):
        o = tuple(sorted((dx, -dy) for dy, dx in o))
    return o


def _shape(rng, kind, a):
    s = set()
    if kind == "plus":
        for t in range(-a, a + 1):
            s.add((0, t))
            s.add((t, 0))
    elif kind == "x":
        for t in range(-a, a + 1):
            s.add((t, t))
            s.add((t, -t))
    elif kind == "L":
        b = int(rng.integers(max(2, a - 1), a + 2))
        for t in range(0, a + 1):
            s.add((-t, 0))
        for t in range(0, b + 1):
            s.add((0, t))
    elif kind == "T":
        b = int(rng.integers(max(2, a - 1), a + 2))
        for t in range(-a, a + 1):
            s.add((0, t))
        for t in range(0, b + 1):
            s.add((t, 0))
    else:                                                    # cadre
        h, w = a, int(rng.integers(max(2, a - 1), a + 2))
        for t in range(-w, w + 1):
            s.add((-h, t))
            s.add((h, t))
        for t in range(-h, h + 1):
            s.add((t, -w))
            s.add((t, w))
    return tuple(sorted(s))


class Lines4(V4Game):
    family = "lines4"
    mechanic = "object"
    principle = ("dessins au trait superposés : chaque point doit être traversé par un trait de sa couleur (bacs de "
                 "peinture, plots, point bicolore)", ["re86", "cn04", "ar25"])
    simple_ops = ("up", "down", "left", "right", "interact")
    CAT = ("pads", "pegs", "third", "anti", "cross")
    exact_progress = False
    n_levels_official = (6, 8)

    def setup(self, rng):
        r = self.orng
        self.variant = str(rng.choice(["cycle", "click"]))
        self.uses_click = self.variant == "click"
        self.free_miss = bool(rng.random() < 0.5)
        self.cat = [str(x) for x in rng.permutation(self.CAT)]
        self.g = int(r.choice([2, 3, 3])) if self.s_screen == 1 else 2
        self.m = 2
        self.kinds = [str(x) for x in r.permutation(["plus", "plus", "L", "T", "rect", "x"])]
        self.peg_c = None

    # ------------------------------------------------------------------ géométrie
    def _anchor(self, ly, lx):
        return (self.m + ly * self.g, self.m + lx * self.g)

    def _pix(self, lvl, k, p):
        ay, ax = self._anchor(p[0], p[1])
        return [(ay + dy, ax + dx) for dy, dx in lvl["rs"][k][p[2]]]

    def _blocked(self, lvl, k, p):
        if not (0 <= p[0] < lvl["NY"] and 0 <= p[1] < lvl["NX"]):
            return True
        return bool(lvl["blk"][k][p[2]][p[0], p[1]])

    def _build_blk(self, lvl):
        H, W = self.AH, self.AW
        peg = np.zeros((H, W), bool)
        for (y, x) in lvl["pegs"]:
            peg[y, x] = True
        NY, NX = lvl["NY"], lvl["NX"]
        Y0 = self.m + self.g * np.arange(NY)[:, None]
        X0 = self.m + self.g * np.arange(NX)[None, :]
        blk = []
        for k in range(len(lvl["rs"])):
            per = []
            for rot in range(4):
                b = np.zeros((NY, NX), bool)
                for dy, dx in lvl["rs"][k][rot]:
                    Y, X = Y0 + dy, X0 + dx
                    ok = (Y >= 0) & (Y < H) & (X >= 0) & (X < W)
                    b |= ok & peg[np.clip(Y, 0, H - 1), np.clip(X, 0, W - 1)]
                per.append(b)
            blk.append(per)
        lvl["blk"] = blk

    def _covers(self, lvl, st):
        """pixel -> ensemble des couleurs des traits qui le traversent."""
        cov = {}
        for k, p in enumerate(st[0]):
            for q in self._pix(lvl, k, p):
                cov.setdefault(q, set()).add(p[3])
        return cov

    def _n_ok(self, lvl, st):
        cov = self._covers(lvl, st)
        good = sum(1 for (y, x, cols) in lvl["dots"] if set(cols) <= cov.get((y, x), set()))
        bad = sum(1 for q in lvl["anti"] if q in cov)
        return good, bad

    def is_win(self, lvl, st):
        good, bad = self._n_ok(lvl, st)
        return good == len(lvl["dots"]) and bad == 0

    # ------------------------------------------------------------------ règles
    def _piece_at(self, lvl, st, px):
        if px is None:
            return []
        return [k for k, p in enumerate(st[0]) if tuple(px) in set(self._pix(lvl, k, p))]

    def apply(self, lvl, st, op, px):
        pieces, sel = st
        n = len(pieces)
        if op == "click":
            ks = self._piece_at(lvl, st, px)
            if not ks:
                return ((pieces, -1), False) if sel >= 0 else (st, False)
            k = next((j for j in ks if j != sel), ks[0])
            return ((pieces, k), False) if k != sel else (st, False)
        if op == "interact":
            if self.variant == "cycle":
                return ((pieces, (sel + 1) % n), False) if n > 1 else (st, False)
            if sel < 0:
                return st, False
            p = pieces[sel]
            q = (p[0], p[1], (p[2] + 1) % 4, p[3])
            if lvl["rs"][sel][q[2]] == lvl["rs"][sel][p[2]] or self._blocked(lvl, sel, q):
                return st, False
            return (pieces[:sel] + (q,) + pieces[sel + 1:], sel), False
        if sel < 0:
            return st, False
        dr, dc = DV[op]
        p = pieces[sel]
        q = (p[0] + dr, p[1] + dc, p[2], p[3])
        if self._blocked(lvl, sel, q):
            return st, False
        if (q[0], q[1]) in lvl["pads"]:
            q = q[:3] + (lvl["pads"][(q[0], q[1])],)
        return (pieces[:sel] + (q,) + pieces[sel + 1:], sel), False

    # ------------------------------------------------------------------ planification
    def _piece_path(self, lvl, k, s0, goal):
        trs = lvl["rs"][k][goal[2]]

        def done(p):
            return p[0] == goal[0] and p[1] == goal[1] and p[3] == goal[3] and lvl["rs"][k][p[2]] == trs

        if done(s0):
            return []
        prev, q = {s0: None}, deque([s0])
        rot_ok = self.variant == "click"
        while q:
            p = q.popleft()
            nxt = []
            for name, (dr, dc) in DV.items():
                n = (p[0] + dr, p[1] + dc, p[2], p[3])
                if self._blocked(lvl, k, n):
                    continue
                if (n[0], n[1]) in lvl["pads"]:
                    n = n[:3] + (lvl["pads"][(n[0], n[1])],)
                nxt.append((name, n))
            if rot_ok:
                n = (p[0], p[1], (p[2] + 1) % 4, p[3])
                if lvl["rs"][k][n[2]] != lvl["rs"][k][p[2]] and not self._blocked(lvl, k, n):
                    nxt.append(("interact", n))
            for name, n in nxt:
                if n in prev:
                    continue
                prev[n] = (p, name)
                if done(n):
                    path = []
                    while prev[n] is not None:
                        n0, nm = prev[n]
                        path.append(nm)
                        n = n0
                    return path[::-1]
                q.append(n)
        return None

    def _select_ops(self, lvl, cur, k):
        pieces, sel = cur
        if sel == k:
            return []
        if self.variant == "cycle":
            return [("interact", None)] * ((k - sel) % len(pieces))
        others = set()
        for j, p in enumerate(pieces):
            if j != k:
                others |= set(self._pix(lvl, j, p))
        for q in self._pix(lvl, k, pieces[k]):
            if q not in others and 0 <= q[0] < self.AH and 0 <= q[1] < self.AW:
                return [("click", q)]
        return None

    def _plan(self, lvl, st):
        pc = lvl.setdefault("_pc", {})
        if st not in pc:
            if len(pc) > 4000:
                pc.clear()
            pc[st] = self._plan0(lvl, st)
        return pc[st]

    def _plan0(self, lvl, st):
        pieces, sel = st
        n = len(pieces)
        order = sorted(range(n), key=lambda k: (k != sel, k))
        out, cur = [], st
        for k in order:
            path = self._piece_path(lvl, k, cur[0][k], lvl["tp"][k])
            if path is None:
                return None
            if not path:
                continue
            ops = self._select_ops(lvl, cur, k)
            if ops is None:
                return None
            for o in ops + [(nm, None) for nm in path]:
                out.append(o)
                cur, _ = self.apply(lvl, cur, *o)
        if not self.is_win(lvl, cur):
            return None
        return out

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
        return bool(self._piece_at(lvl, st, px))

    # ------------------------------------------------------------------ niveaux
    def gen_level(self, rng, i, n):
        mods, newest = mods_for(self.cat, i)
        g, m = self.g, self.m
        NY = (self.AH - 2 * m - 1) // g + 1
        NX = (self.AW - 2 * m - 1) // g + 1
        npc = 2 + ("third" in mods) + (i >= 6)
        amax = 9 if self.s_screen == 1 else 5
        lvl = dict(NY=NY, NX=NX, mods=mods, pegs=frozenset(), pads={}, anti=(), dots=())
        H, W = self.AH, self.AW

        def inside(pix, mg=1):
            return all(mg <= y < H - mg and mg <= x < W - mg for y, x in pix)

        for _ in range(60):
            shapes = []
            for k in range(npc):
                kind = self.kinds[(k + i) % len(self.kinds)]
                shapes.append(_shape(rng, kind, int(rng.integers(amax // 2 + 1, amax + 1)) - (2 if kind == "rect" else 0)))
            lvl["rs"] = [[_rot(s, r) for r in range(4)] for s in shapes]
            tp, ok = [], True
            for k in range(npc):
                rot = int(rng.integers(4)) if self.variant == "click" else 0
                for _t in range(80):
                    p = (int(rng.integers(NY)), int(rng.integers(NX)), rot, k)
                    pix = self._pix(lvl, k, p)
                    if not inside(pix, 2):
                        continue
                    if k == 1 and "cross" in mods and not (set(pix) & set(self._pix(lvl, 0, tp[0]))):
                        continue
                    if any(abs(p[0] - q[0]) + abs(p[1] - q[1]) < 3 for q in tp):
                        continue
                    tp.append(p)
                    break
                else:
                    ok = False
                    break
            if not ok:
                continue
            lvl["tp"] = tuple(tp)
            pixs = [set(self._pix(lvl, k, tp[k])) for k in range(npc)]
            allpix = set().union(*pixs)
            dots = []
            taken = set()

            def far(q, d=2):
                return all(abs(q[0] - t[0]) > d or abs(q[1] - t[1]) > d for t in taken)

            if "cross" in mods:
                inter = sorted(pixs[0] & pixs[1])
                inter = [q for q in inter if inside([q], 2) and all(q not in pixs[j] for j in range(2, npc))]
                if not inter:
                    continue
                q = inter[int(rng.integers(len(inter)))]
                dots.append((q[0], q[1], (0, 1)))
                taken.add(q)
            nd = 2 + (i >= 3)
            for k in range(npc):
                ay, ax = self._anchor(tp[k][0], tp[k][1])
                cand = sorted(q for q in pixs[k] if all(q not in pixs[j] for j in range(npc) if j != k)
                              and inside([q], 2) and max(abs(q[0] - ay), abs(q[1] - ax)) >= 2)
                cnt = 0
                for j in rng.permutation(len(cand)):
                    q = cand[j]
                    if far(q):
                        dots.append((q[0], q[1], (k,)))
                        taken.add(q)
                        cnt += 1
                        if cnt >= nd:
                            break
                if cnt < 2:
                    ok = False
                    break
            if not ok:
                continue
            lvl["dots"] = tuple(dots)
            if "anti" in mods:
                ring = sorted({(y + a, x + b) for (y, x) in allpix for a in (-3, -2, 2, 3) for b in (-1, 0, 1)}
                              - allpix)
                ring = [q for q in ring if inside([q], 2) and far(q)]
                anti = []
                for j in rng.permutation(len(ring)):
                    if far(ring[j]):
                        anti.append(ring[j])
                        taken.add(ring[j])
                    if len(anti) >= int(rng.integers(2, 4)):
                        break
                if len(anti) < 2:
                    continue
                lvl["anti"] = tuple(anti)
            pegs = set()
            if "pegs" in mods:
                ring = sorted({(y + a, x + b) for (y, x) in allpix for a in (-3, 3) for b in (-3, 0, 3)} - allpix)
                ring = [q for q in ring if inside([q], 3) and far(q, 2)]
                for j in rng.permutation(len(ring))[:int(rng.integers(3, 6))]:
                    y, x = ring[j]
                    blk = {(y + a, x + b) for a in (-1, 0, 1) for b in (-1, 0, 1)}
                    if not (blk & allpix):
                        pegs |= blk
                if not pegs:
                    continue
            lvl["pegs"] = frozenset(pegs)
            self._build_blk(lvl)
            if any(self._blocked(lvl, k, tp[k]) for k in range(npc)):
                continue
            pads = {}
            scol = [k for k in range(npc)]
            if "pads" in mods:
                need = [0] + ([1] if npc > 1 and rng.random() < 0.5 else [])
                spare = npc + int(rng.integers(0, 2))
                for k in need:
                    scol[k] = spare if k == 0 else (spare + 1)
                    for _t in range(40):
                        c = (int(rng.integers(NY)), int(rng.integers(NX)))
                        if c not in pads and abs(c[0] - tp[k][0]) + abs(c[1] - tp[k][1]) >= 3 and \
                                not self._blocked(lvl, k, (c[0], c[1], tp[k][2], k)):
                            pads[c] = k
                            break
                for _t in range(10):                                   # bac leurre
                    c = (int(rng.integers(NY)), int(rng.integers(NX)))
                    if c not in pads and all(c != (p[0], p[1]) for p in tp):
                        pads[c] = scol[0]
                        break
            lvl["pads"] = pads
            # dispersion
            for _t in range(30):
                sp = []
                for k in range(npc):
                    for _u in range(60):
                        rot = int(rng.integers(4)) if self.variant == "click" else 0
                        p = (int(rng.integers(NY)), int(rng.integers(NX)), rot, scol[k])
                        if (p[0], p[1]) in pads or self._blocked(lvl, k, p):
                            continue
                        if abs(p[0] - tp[k][0]) + abs(p[1] - tp[k][1]) < 4:
                            continue
                        if not inside([self._anchor(p[0], p[1])], 1):
                            continue
                        sp.append(p)
                        break
                    else:
                        break
                if len(sp) < npc:
                    continue
                st0 = (tuple(sp), 0 if self.variant == "cycle" else -1)
                if self.is_win(lvl, st0):
                    continue
                lvl.pop("_pc", None)
                sol = self._plan(lvl, st0)
                if sol is None or len(sol) < self.min_len(i, n) * 0.8:
                    continue
                return lvl, st0, sol
        return self.why("place")

    def min_len(self, i, n):
        return min(14 + 6 * i, 60)

    def body(self, lvl, st):
        if st[1] < 0:
            return None
        p = st[0][st[1]]
        return (p[0], p[1], lvl["NY"], lvl["NX"])

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        if st2[1] != st[1]:
            ev[EV["toggle"]] = 1
        if op in DV and st2 == st and st[1] >= 0:
            ev[EV["touch"]] = 1
        if st2[0] != st[0] and any(a[3] != b[3] for a, b in zip(st[0], st2[0])):
            ev[EV["color"]] = 1
        g0, b0 = self._n_ok(lvl, st)
        g1, b1 = self._n_ok(lvl, st2)
        if g1 > g0:
            ev[EV["collect"]] = 1
        if b1 > b0:
            ev[EV["danger"]] = 1

    # ------------------------------------------------------------------ rendu
    def render(self, lvl, st):
        pal = self.pal
        g = self.blank()
        H, W = g.shape
        cols = [pal.a(k) for k in range(8)]
        peg_c = pal.wall
        for (y, x) in lvl["pegs"]:
            g[y, x] = R(peg_c)
        for (ly, lx), c in lvl["pads"].items():
            ay, ax = self._anchor(ly, lx)
            h = 2 if self.s_screen == 1 else 1
            rect(g, ay - h, ax - h, 2 * h + 1, 2 * h + 1, cols[c])
        pieces, sel = st
        order = [k for k in range(len(pieces)) if k != sel] + ([sel] if sel >= 0 else [])
        for k in order:
            p = pieces[k]
            for (y, x) in self._pix(lvl, k, p):
                if 0 <= y < H and 0 <= x < W:
                    g[y, x] = R(cols[p[3]])
            if k == sel:
                ay, ax = self._anchor(p[0], p[1])
                g[ay, ax] = R(draw.WHITE if pal.bg != draw.WHITE else draw.BLACK)
        for (y, x, cs) in lvl["dots"]:
            for a in (-1, 0, 1):
                for b in (-1, 0, 1):
                    if a or b:
                        c = cols[cs[0]] if (len(cs) == 1 or b < 0 or (b == 0 and a < 0)) else cols[cs[1]]
                        g[y + a, x + b] = R(c)
        anti_c = next(c for c in (draw.GREY, draw.LGREY, draw.DGREY, pal.line) if c != pal.bg)
        for (y, x) in lvl["anti"]:
            for a in (-1, 0, 1):
                for b in (-1, 0, 1):
                    if a or b:
                        g[y + a, x + b] = R(anti_c)
        return g
