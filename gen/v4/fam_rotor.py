"""rotor4 (clic, éléments partagés) : des PISTES de cases (anneaux rectangulaires ou rangées / colonnes cycliques) se
croisent et PARTAGENT des cases ; cliquer un bouton-flèche fait tourner toute une piste d'un cran (les tuiles qu'elle
porte avancent, y compris celles des cases partagées, ce qui dérange les autres pistes). But : chaque cible porte une
tuile de sa couleur. Les clics ailleurs que sur un bouton actif sont sans effet (et gratuits).
Variantes :
  géométrie : loops (anneaux + panneau de boutons à droite) | lines (rangées / colonnes cycliques, boutons aux bouts) ;
  cibles    : cells (cadres sur le plateau) | hud (vignette du plateau dans un bandeau, rien sur le plateau).
Règles ajoutées (ordre tiré par jeu, cumulées) :
  ringA / ringB : une piste de plus (qui croise les autres) ;
  reverse : certaines pistes reçoivent un 2e bouton « arrière » (de base chaque piste n'a qu'un bouton « avant » :
           revenir d'un cran coûte un tour complet) ;
  kinds  : 2e sorte de tuile (cibles colorées) ;
  gate   : une piste est VERROUILLÉE (boutons gris) tant que sa case-clé (cadre de sa couleur sur une autre piste)
           ne porte pas de tuile ;
  link   : deux pistes COUPLÉES (même couleur, pointillé) : chaque bouton de l'une fait aussi tourner l'autre.
Solvabilité : BFS ARRIÈRE exact depuis l'état résolu (mouvements inverses, plafonné) ; départ tiré dans les couches
les plus profondes -> solution optimale ; la même table sert d'oracle exact."""
from collections import deque

import numpy as np

from ..core import EV, INF
from . import draw
from .base import V4Game
from .draw import R, blit, frame_rect, rect, sprite

P = 5                                                   # pas des cases (px)


def _arrow(c, d):
    """flèche 5×5 vers d ∈ {0 haut, 1 droite, 2 bas, 3 gauche}."""
    s = np.full((5, 5), -1, np.int64)
    for y, w in enumerate((1, 3, 5, 3, 1)):
        s[y, :w] = c                                     # triangle plein vers la droite
    return np.rot90(s, {1: 0, 0: 1, 3: 2, 2: 3}[d])


class Rotor4(V4Game):
    family = "rotor4"
    mechanic = "click"
    principle = ("éléments partagés : des pistes cycliques qui se croisent tournent d'un cran par clic ; amener les "
                 "tuiles sur les cibles (sens unique, verrou à clé, pistes couplées)", ["lp85", "vc33", "s5i5"])
    simple_ops = ()
    CAT = ("ringA", "ringB", "reverse", "kinds", "gate", "link")
    free_miss = True
    scale_p = (1.0, 0.0)
    n_levels_official = (5, 8)
    back_cap = 8000

    def setup(self, rng):
        r = self.orng
        self.uses_click = True
        self.geom = str(rng.choice(["loops", "lines"]))
        self.tshow = str(rng.choice(["cells", "cells", "hud"]))
        self.variant = f"{self.geom}-{self.tshow}"
        self.cat = [str(x) for x in rng.permutation(self.CAT)]
        self.tile_kind = str(r.choice(["box", "box", "disc", "diamond"]))
        self.track_style = str(r.choice(["line", "dots"]))

    # ---------------------------------------------------------------- géométrie
    def _area(self):
        top = 20 if self.tshow == "hud" else 2
        right = 15 if self.geom == "loops" else 1
        Hc = (self.AH - top - 2) // P
        Wc = (self.AW - right - 2) // P
        return top, right, min(Hc, 11), min(Wc, 11)

    @staticmethod
    def _rect_cells(r0, c0, h, w):
        cells = [(r0, c) for c in range(c0, c0 + w)]
        cells += [(r, c0 + w - 1) for r in range(r0 + 1, r0 + h)]
        cells += [(r0 + h - 1, c) for c in range(c0 + w - 2, c0 - 1, -1)]
        cells += [(r, c0) for r in range(r0 + h - 2, r0, -1)]
        return cells

    def _make_loops(self, rng, n, Hc, Wc):
        tracks = []
        for k in range(n):
            ok = False
            for _ in range(200):
                h = int(rng.integers(3, min(7, Hc) + 1))
                w = int(rng.integers(3, min(8, Wc) + 1))
                r0 = int(rng.integers(0, Hc - h + 1))
                c0 = int(rng.integers(0, Wc - w + 1))
                cells = self._rect_cells(r0, c0, h, w)
                cs = set(cells)
                good = True
                touch = 0
                for t in tracks:
                    sh = cs & set(t)
                    if len(sh) > 4:
                        good = False
                        break
                    for a in sh:                                    # pas de bords communs
                        for b in sh:
                            if a != b and abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1:
                                good = False
                    # pas de cases voisines non partagées (lisibilité)
                    ts = set(t)
                    for (y, x) in cs - sh:
                        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                            q = (y + dy, x + dx)
                            if q in ts and q not in sh and not any(abs(q[0] - s[0]) + abs(q[1] - s[1]) <= 1 for s in sh):
                                good = False
                    touch += len(sh) > 0
                if good and (not tracks or touch >= 1) and all(cs != set(t) for t in tracks):
                    tracks.append(cells)
                    ok = True
                    break
            if not ok:
                return None
        return tracks

    def _make_lines(self, rng, n, Hc, Wc, lmin=5, p_bend=0.0):
        tracks, ends, used = [], [], set()
        for k in range(n):
            ok = False
            for _ in range(300):
                horiz = (k % 2 == 0) if rng.random() < 0.8 else bool(rng.random() < 0.5)
                bent = rng.random() < p_bend
                if bent:                                        # piste coudée (L) : rangée puis colonne
                    L1, L2 = int(rng.integers(4, 9)), int(rng.integers(3, 7))
                    sx, sy = int(rng.choice([-1, 1])), int(rng.choice([-1, 1]))
                    r = int(rng.integers(0, Hc))
                    c0 = int(rng.integers(0, Wc))
                    cells = [(r, c0 + sx * j) for j in range(L1)]
                    ce = c0 + sx * (L1 - 1)
                    cells += [(r + sy * j, ce) for j in range(1, L2 + 1)]
                    e = [(r, c0 - sx), (r + sy * (L2 + 1), ce)]
                    if horiz:                                   # ou colonne puis rangée (transposée)
                        cells = [(c[1], c[0]) for c in cells]
                        e = [(q[1], q[0]) for q in e]
                    if not all(0 <= q[0] < Hc and 0 <= q[1] < Wc for q in cells + e):
                        continue
                elif horiz:
                    L = int(rng.integers(min(lmin, Wc - 2), min(10, Wc - 2) + 1))
                    r = int(rng.integers(0, Hc))
                    c0 = int(rng.integers(1, Wc - L))
                    cells = [(r, c) for c in range(c0, c0 + L)]
                    e = [(r, c0 - 1), (r, c0 + L)]
                else:
                    L = int(rng.integers(min(lmin, Hc - 2), min(10, Hc - 2) + 1))
                    c = int(rng.integers(0, Wc))
                    r0 = int(rng.integers(1, Hc - L))
                    cells = [(r, c) for r in range(r0, r0 + L)]
                    e = [(r0 - 1, c), (r0 + L, c)]
                cs = set(cells)
                if any(q in used or q in cs for q in e) or any(q in {x for ee in ends for x in ee} for q in cells):
                    continue
                good, touch = True, 0
                for t in tracks:
                    sh = cs & set(t)
                    if len(sh) > (2 if bent else 1):
                        good = False
                    touch += len(sh)
                    ts = set(t)                                      # pas de pistes parallèles collées
                    for (y, x) in cs - sh:
                        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                            q = (y + dy, x + dx)
                            if q in ts and q not in sh and not any(abs(q[0] - s[0]) + abs(q[1] - s[1]) <= 1 for s in sh):
                                good = False
                if good and (not tracks or touch >= 1):
                    tracks.append(cells)
                    ends.append(e)
                    used |= cs | set(e)
                    ok = True
                    break
            if not ok:
                return None, None
        return tracks, ends

    # ---------------------------------------------------------------- mouvements
    def _perm(self, lvl, ti, d):
        """permutation des indices de cases pour la piste ti tournée de d (±1)."""
        idx = lvl["idx"]
        pm = list(range(len(lvl["cells"])))
        t = lvl["tracks"][ti]
        n = len(t)
        for k, c in enumerate(t):
            pm[idx[c]] = idx[t[(k + d) % n]]
        return pm

    def _build_moves(self, lvl):
        """boutons -> (piste, sens, permutation composée)."""
        moves = []
        for b in lvl["buttons"]:
            ti, d = b["track"], b["dir"]
            pm = self._perm(lvl, ti, d)
            if b.get("partner") is not None:
                p2 = self._perm(lvl, b["partner"], d)
                pm = [p2[x] for x in pm]
            moves.append(tuple(pm))
        return moves

    def _enabled(self, lvl, st, bi):
        g = lvl["gate"]
        if g is None or lvl["buttons"][bi]["track"] not in g[0]:
            return True
        return any(g[1] in grp for grp in st)

    @staticmethod
    def _mv(st, pm):
        return tuple(frozenset([pm[p] for p in grp]) for grp in st)

    def _button_at(self, lvl, px):
        if px is None:
            return None
        y, x = px
        for k, b in enumerate(lvl["buttons"]):
            by, bx = b["px"]
            if by <= y < by + 5 and bx <= x < bx + 5:
                return k
        return None

    def apply(self, lvl, st, op, px):
        if op != "click":
            return st, False
        k = self._button_at(lvl, px)
        if k is None or not self._enabled(lvl, st, k):
            return st, False
        return self._mv(st, lvl["moves"][k]), False

    def is_win(self, lvl, st):
        return st == lvl["goal"]

    def click_targets(self, lvl, st):
        return [(b["px"][0] + 2, b["px"][1] + 2) for k, b in enumerate(lvl["buttons"]) if self._enabled(lvl, st, k)]

    def click_hit(self, lvl, st, px):
        if self._button_at(lvl, px) is not None:
            return True
        y, x = px
        r, c = (y - lvl["oy"]) // P, (x - lvl["ox"]) // P
        return y >= lvl["oy"] and x >= lvl["ox"] and (r, c) in lvl["idx"]

    def table(self, li):
        return self.levels[li]["lvl"]["dtab"]

    def heuristic(self, lvl, st):
        miss = sum(len(set(a) - set(b)) for a, b in zip(st, lvl["goal"]))
        return lvl["dmax"] + 1 + miss

    def body(self, lvl, st):
        return None

    # ---------------------------------------------------------------- génération
    def gen_level(self, rng, i, n):
        k_new = min(i, len(self.cat))
        mods = tuple(self.cat[:k_new])
        newest = self.cat[k_new - 1] if 0 < i <= len(self.cat) else None
        top, right, Hc, Wc = self._area()
        ntr = (2 if self.geom == "lines" else 1) + ("ringA" in mods) + ("ringB" in mods)
        if "link" in mods or "gate" in mods:
            ntr = max(ntr, 2 + ("link" in mods and "gate" in mods))
        if self.geom == "lines":
            tracks, ends = self._make_lines(rng, ntr, Hc, Wc, 5 + (i >= 1) + (i >= 3), 0.0 if i == 0 else 0.6)
        else:
            tracks, ends = self._make_loops(rng, ntr, Hc, Wc), None
        if tracks is None:
            return self.why("tracks")
        cells = sorted({c for t in tracks for c in t})
        idx = {c: k for k, c in enumerate(cells)}
        # cadrage du plateau
        rs = [c[0] for c in cells] + ([q[0] for e in ends for q in e] if ends else [])
        cs_ = [c[1] for c in cells] + ([q[1] for e in ends for q in e] if ends else [])
        bh, bw = (max(rs) - min(rs) + 1), (max(cs_) - min(cs_) + 1)
        oy = top + (self.AH - top - 2 - bh * P) // 2 - min(rs) * P
        ox = 1 + (self.AW - right - 2 - bw * P) // 2 - min(cs_) * P
        lvl = dict(tracks=tracks, cells=cells, idx=idx, oy=oy, ox=ox, mods=mods, gate=None, links=())
        # liens
        partner = {}
        if "link" in mods:
            a, b = (int(x) for x in rng.permutation(ntr)[:2])
            partner = {a: b, b: a}
            lvl["links"] = ((a, b),)
        twoway = set()
        if "reverse" in mods:
            twoway = {int(x) for x in rng.permutation(ntr)[:max(1, ntr // 2)]}
        # boutons
        buttons = []
        for ti in range(ntr):
            dirs = [1, -1] if ti in twoway else [1]
            for d in dirs:
                b = dict(track=ti, dir=d, partner=partner.get(ti))
                if self.geom == "lines":
                    t = tracks[ti]
                    e = ends[ti][1] if d == 1 else ends[ti][0]
                    a_, b2 = (t[-2], t[-1]) if d == 1 else (t[1], t[0])
                    b["px"] = (oy + e[0] * P, ox + e[1] * P)
                    b["arrow"] = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}[(b2[0] - a_[0], b2[1] - a_[1])]
                else:
                    b["px"] = (4 + 8 * ti, self.AW - 13 + (7 if d == 1 else 0))
                    b["arrow"] = 1 if d == 1 else 3
                buttons.append(b)
        if self.geom == "loops" and 4 + 8 * ntr > self.AH:
            return self.why("panel")
        lvl["buttons"] = buttons
        # tuiles et cibles (état résolu)
        nk = 2 if "kinds" in mods else 1
        nt = int(min(7, 2 + (i >= 1) + (i >= 2) + (i >= 4) + (nk == 2) + (self.geom == "lines")))
        nt = max(nt, nk)
        sh = [c for c in cells if sum(c in t for t in tracks) > 1]
        perm = [cells[int(k)] for k in rng.permutation(len(cells))]
        tg = perm[:nt]
        kinds = [k % nk for k in range(nt)]
        goal = tuple(frozenset(idx[c] for c, kk in zip(tg, kinds) if kk == q) for q in range(nk))
        lvl["targets"] = tuple((c, kk) for c, kk in zip(tg, kinds))
        if "gate" in mods:
            g = int(rng.integers(ntr))
            locked = (g,) + ((partner[g],) if g in partner else ())
            others = [c for c in cells if not any(c in set(tracks[t]) for t in locked)]
            if not others:
                return self.why("gate")
            sw = others[int(rng.integers(len(others)))]
            if all(t in locked for t in range(ntr)):
                return self.why("gate_all")
            lvl["gate"] = (locked, idx[sw], sw)
        lvl["moves"] = self._build_moves(lvl)
        lvl["goal"] = goal
        # BFS arrière depuis l'état résolu
        inv = []
        for pm in lvl["moves"]:
            iv = [0] * len(pm)
            for a, b in enumerate(pm):
                iv[b] = a
            inv.append(tuple(iv))
        dist = {goal: 0}
        q = deque([goal])
        nb = len(inv)
        while q and len(dist) < self.back_cap:
            y = q.popleft()
            dy = dist[y]
            for k in range(nb):
                if not self._enabled(lvl, y, k):              # la clé n'est pas sur la piste verrouillée
                    continue
                x = self._mv(y, inv[k])
                if x not in dist:
                    dist[x] = dy + 1
                    q.append(x)
        dmax = max(dist.values())
        lvl["dmax"] = dmax
        full = len(dist) < self.back_cap
        want = self.min_len(i, n)
        mv = lvl["moves"]
        if full:                                               # orbite entière connue : départ le plus profond
            lo = max(1, min(dmax, want))
            cand = [s_ for s_, d in dist.items() if d >= max(lo, dmax - 2)]
            cand2 = [s_ for s_ in cand if sum(len(set(a_) & set(b_)) for a_, b_ in zip(s_, goal)) <= (nt - 1) // 2]
            cand = cand2 or cand
            if not cand or dmax < 3:
                return self.why("shallow")
            st0 = cand[int(rng.integers(len(cand)))]
            path = []
        else:                                                  # mélanges aléatoires + BFS avant jusqu'à la table
            best_all = None
            for _ in range(2):
                r_ = self._scramble_astar(rng, lvl, goal, dist, nb, mv, 30 + 8 * i)
                if r_ is not None and (best_all is None or r_[0] > best_all[0]):
                    best_all = r_
            if best_all is None:
                return self.why("no_meet")
            _, st0, path = best_all
        # solution : chemin avant puis descente dans la table
        sol, st = [], st0
        for k in path:
            b_ = lvl["buttons"][k]
            sol.append(("click", (b_["px"][0] + 2, b_["px"][1] + 2)))
            st = self._mv(st, mv[k])
        while dist[st] > 0:
            for k in range(nb):
                if not self._enabled(lvl, st, k):
                    continue
                s2 = self._mv(st, mv[k])
                if dist.get(s2, INF) == dist[st] - 1:
                    b_ = lvl["buttons"][k]
                    sol.append(("click", (b_["px"][0] + 2, b_["px"][1] + 2)))
                    st = s2
                    break
            else:
                return self.why("descent")
        if len(sol) < 3:
            return self.why("short")
        if newest == "gate":                                   # la clé doit servir
            g = lvl["gate"]
            if not any(lvl["buttons"][self._button_at(lvl, px)]["track"] in g[0] for _, px in sol):
                return self.why("gate_unused")
        if newest == "link" and not any(lvl["buttons"][self._button_at(lvl, px)]["partner"] is not None for _, px in sol):
            return self.why("link_unused")
        lvl["dtab"] = dist
        return lvl, st0, sol

    def _scramble_astar(self, rng, lvl, goal, dist, nb, mv, walk):
        """grand plateau : marche aléatoire de `walk` coups depuis l'état résolu, puis A* pondéré (heuristique : somme
        des distances case -> cible de même sorte dans le graphe des rotations) jusqu'à la table arrière exacte.
        -> (longueur trouvée, départ, chemin avant) ; solution non garantie optimale (« meilleure trouvée »)."""
        import heapq
        ncell = len(lvl["cells"])
        rev = [[] for _ in range(ncell)]
        for pm in mv:
            for a, b in enumerate(pm):
                if a != b:
                    rev[b].append(a)
        dk = []
        for grp in goal:
            d = [99] * ncell
            q = deque(grp)
            for t in grp:
                d[t] = 0
            while q:
                x = q.popleft()
                for y in rev[x]:
                    if d[y] > d[x] + 1:
                        d[y] = d[x] + 1
                        q.append(y)
            dk.append(d)

        def h(st):
            return sum(dk[k][p] for k, grp in enumerate(st) for p in grp)
        st0, last = goal, -1
        for _ in range(walk):
            ks = [k for k in range(nb) if self._enabled(lvl, st0, k)]
            ks = [k for k in ks if k != last] or ks
            k = int(rng.choice(ks))
            st0 = self._mv(st0, mv[k])
            last = k
        if st0 in dist:
            return dist[st0], st0, []
        prev, gs = {st0: None}, {st0: 0}
        heap, cnt = [(h(st0), 0, st0)], 0
        while heap and len(gs) < 15000:
            f, _, x = heapq.heappop(heap)
            if x in dist:
                path, y = [], x
                while prev[y] is not None:
                    y, k = prev[y]
                    path.append(k)
                return gs[x] + dist[x], st0, path[::-1]
            for k in range(nb):
                if not self._enabled(lvl, x, k):
                    continue
                y = self._mv(x, mv[k])
                g2 = gs[x] + 1
                if g2 < gs.get(y, 1 << 30):
                    gs[y] = g2
                    prev[y] = (x, k)
                    cnt += 1
                    heapq.heappush(heap, (g2 + h(y), cnt, y))
        return None

    def _scramble_meet(self, rng, lvl, goal, dist, nb, mv):
        """état mélangé (marche aléatoire) -> (longueur optimale trouvée, état, chemin avant jusqu'à la table)."""
        st0, last = goal, -1
        for _ in range(int(rng.integers(30, 70))):
            ks = [k for k in range(nb) if self._enabled(lvl, st0, k)]
            ks = [k for k in ks if k != last] or ks
            k = int(rng.choice(ks))
            st0 = self._mv(st0, mv[k])
            last = k
        if st0 in dist:
            return dist[st0], st0, []
        prev, q, best = {st0: None}, deque([st0]), None
        depth = {st0: 0}
        while q and len(prev) < 12000:
            x = q.popleft()
            if best is not None and depth[x] >= best[0]:
                break
            for k in range(nb):
                if not self._enabled(lvl, x, k):
                    continue
                y = self._mv(x, mv[k])
                if y in prev:
                    continue
                prev[y] = (x, k)
                depth[y] = depth[x] + 1
                if y in dist and (best is None or depth[y] + dist[y] < best[0]):
                    best = (depth[y] + dist[y], y)
                q.append(y)
        if best is None:
            return None
        y, path = best[1], []
        while prev[y] is not None:
            y, k = prev[y]
            path.append(k)
        return best[0], st0, path[::-1]

    def min_len(self, i, n):
        return min(6 + 4 * i, 30)

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        if op != "click":
            return
        k = self._button_at(lvl, px)
        if k is None:
            return
        if st2 != st:
            ev[EV["push"]] = 1
            hit0 = sum(len(set(a) & set(b)) for a, b in zip(st, lvl["goal"]))
            hit1 = sum(len(set(a) & set(b)) for a, b in zip(st2, lvl["goal"]))
            if hit1 > hit0:
                ev[EV["collect"]] = 1
            g = lvl["gate"]
            if g is not None and any(g[1] in grp for grp in st) != any(g[1] in grp for grp in st2):
                ev[EV["open"]] = 1
        else:
            ev[EV["touch"]] = 1

    # ---------------------------------------------------------------- rendu
    def _tcol(self, lvl, ti):
        for a, b in lvl["links"]:
            if ti in (a, b):
                ti = min(a, b)
        return self.pal.a(2 + ti)

    def render(self, lvl, st):
        pal = self.pal
        g = self.blank()
        oy, ox = lvl["oy"], lvl["ox"]
        kc = (pal.a(0), pal.a(1))
        cells = lvl["cells"]
        shared = {c for c in cells if sum(c in t for t in lvl["tracks"]) > 1}
        # pistes : bande pleine de 3 px passant par le centre des cases
        for ti, t in enumerate(lvl["tracks"]):
            col = self._tcol(lvl, ti)
            loop = self.geom == "loops"
            pts = t + ([t[0]] if loop else [])
            for a, b in zip(pts[:-1], pts[1:]):
                y0, x0 = oy + min(a[0], b[0]) * P + 1, ox + min(a[1], b[1]) * P + 1
                y1, x1 = oy + max(a[0], b[0]) * P + 4, ox + max(a[1], b[1]) * P + 4
                rect(g, y0, x0, y1 - y0, x1 - x0, col)
        for c in shared:                                       # croisements
            rect(g, oy + c[0] * P + 1, ox + c[1] * P + 1, 3, 3, pal.fg)
        gate = lvl["gate"]
        if gate is not None:                                   # case-clé : cadre de la couleur de la piste verrouillée
            c = gate[2]
            frame_rect(g, oy + c[0] * P, ox + c[1] * P, 5, 5, self._tcol(lvl, gate[0][0]))
        if self.tshow == "cells":
            for c, kk in lvl["targets"]:
                frame_rect(g, oy + c[0] * P, ox + c[1] * P, 5, 5, kc[kk])
        for kk, grp in enumerate(st):
            for p in grp:
                c = cells[p]
                rect(g, oy + c[0] * P + 1, ox + c[1] * P + 1, 3, 3, kc[kk])
        # boutons (triangles pleins) ; anneaux : colonne à droite, sur le fond
        for k, b in enumerate(lvl["buttons"]):
            on = self._enabled(lvl, st, k)
            col = self._tcol(lvl, b["track"]) if on else (pal.wall if pal.wall != pal.bg else pal.line)
            blit(g, _arrow(col, b["arrow"]), *b["px"])
            if self.geom == "loops" and b["dir"] == 1:
                rect(g, b["px"][0] + 1, self.AW - 7 - 1, 3, 1, self._tcol(lvl, b["track"]))
        # vignette des cibles (2 px par case)
        if self.tshow == "hud":
            rs = [c[0] for c in cells]
            cs = [c[1] for c in cells]
            r0, c0 = min(rs), min(cs)
            y0, x0 = 1, 2
            for c in cells:
                rect(g, y0 + (c[0] - r0) * 2, x0 + (c[1] - c0) * 2, 2, 2, pal.wall if pal.wall != pal.bg else pal.line)
            for c, kk in lvl["targets"]:
                rect(g, y0 + (c[0] - r0) * 2, x0 + (c[1] - c0) * 2, 2, 2, kc[kk])
        return g
