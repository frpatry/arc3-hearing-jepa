"""infer4 (clic, contrainte cachée à inférer d'exemples) : un plateau de tuiles colorées ; cliquer une tuile fait
passer sa couleur à la suivante (ordre montré par la palette du bandeau). Des tuiles-INDICES (icône sur fond sombre,
non cliquables) contraignent leurs voisines ; leur sens n'est pas écrit : le bandeau montre des EXEMPLES RÉSOLUS
(petits plateaux 3×3 autour d'un indice de chaque sorte). Victoire : toutes les contraintes satisfaites à la fois.
Variantes (règle de base) :
  match : l'indice « + » de couleur c exige que ses 4 voisines orthogonales soient de couleur c ;
  count : l'indice à k points de couleur c exige qu'exactement k voisines orthogonales soient de couleur c.
Règles ajoutées (ordre tiré par jeu, une par niveau, mods_for) :
  second  : 2e sorte d'indice (cadre = négation : match -> aucune voisine c ; count -> k voisines NON c) ;
  colors3 : une couleur de plus dans le cycle (2 au niveau 0, 3 ensuite, 4 avec la règle) ;
  diag    : indices « × » qui portent sur les 4 voisines DIAGONALES ;
  locked  : tuiles verrouillées (point central), non cliquables, qui comptent quand même ;
  line    : flèches au bord : contrainte sur toute la rangée / colonne.
Solvabilité : solution construite d'abord (indices dérivés d'elle), puis départ brouillé ; l'assignation satisfaisante
la plus proche du départ (séparation et évaluation, plafonnée) donne la solution de référence."""
import numpy as np

from ..core import EV
from . import draw
from .base import V4Game, mods_for
from .draw import R, blit, frame_rect, rect

ORTH = ((-1, 0), (0, -1), (0, 1), (1, 0))
DIAG = ((-1, -1), (-1, 1), (1, -1), (1, 1))
T = 5                                                    # tuile 5×5, pas 6


class Infer4(V4Game):
    family = "infer4"
    mechanic = "click"
    principle = ("contrainte cachée à inférer d'exemples résolus : cycler les couleurs des tuiles pour satisfaire les "
                 "indices (négation, 3 couleurs, diagonales, tuiles verrouillées, contraintes de rangée)",
                 ["ft09", "tn36", "tr87"])
    simple_ops = ()
    CAT = ("second", "colors3", "diag", "locked", "line")
    free_miss = True
    exact_progress = False
    scale_p = (1.0, 0.0)
    n_levels_official = (5, 8)

    def setup(self, rng):
        r = self.orng
        self.uses_click = True
        self.variant = str(rng.choice(["match", "count"]))
        self.cat = [str(x) for x in rng.permutation(self.CAT)]
        self.hint_bg = None
        self.gap_line = bool(r.random() < 0.3)

    # ---------------------------------------------------------------- contraintes
    def _scope(self, lvl, h):
        """cases-couleur concernées par l'indice h."""
        kind = h["nb"]
        cid = lvl["cid"]
        if kind == "row":
            return [(h["r"], c) for c in range(lvl["Wb"]) if cid.get((h["r"], c), -1) >= 0]
        if kind == "col":
            return [(r, h["c"]) for r in range(lvl["Hb"]) if cid.get((r, h["c"]), -1) >= 0]
        dd = ORTH if kind == "orth" else DIAG
        return [(h["r"] + dy, h["c"] + dx) for dy, dx in dd if cid.get((h["r"] + dy, h["c"] + dx), -1) >= 0]

    def _ok(self, lvl, h, cols):
        c = h["c_"]
        vals = cols
        if self.variant == "match":
            if h["neg"]:
                return all(v != c for v in vals)
            return all(v == c for v in vals)
        n = sum((v != c) if h["neg"] else (v == c) for v in vals)
        return n == h["k"]

    def _colour_of(self, lvl, st, cell):
        k = lvl["cid"][cell]
        return lvl["locked"].get(cell, st[k] if k >= 0 else -1)

    def satisfied(self, lvl, st):
        return [self._ok(lvl, h, [self._colour_of(lvl, st, q) for q in h["scope"]]) for h in lvl["hints"]]

    def is_win(self, lvl, st):
        return all(self.satisfied(lvl, st))

    # ---------------------------------------------------------------- géométrie / clics
    def _tile_at(self, lvl, px):
        if px is None:
            return None
        y, x = px
        r, c = (y - lvl["oy"]) // T, (x - lvl["ox"]) // T
        if y < lvl["oy"] or x < lvl["ox"]:
            return None
        if 0 <= r < lvl["Hb"] and 0 <= c < lvl["Wb"]:
            return (int(r), int(c))
        return None

    def apply(self, lvl, st, op, px):
        if op != "click":
            return st, False
        q = self._tile_at(lvl, px)
        if q is None or q not in lvl["cid"] or q in lvl["locked"]:
            return st, False
        k = lvl["cid"][q]
        v = list(st)
        v[k] = (v[k] + 1) % lvl["nc"]
        return tuple(v), False

    def click_targets(self, lvl, st):
        return [self._cpx(lvl, q) for q, k in lvl["cid"].items() if q not in lvl["locked"]]

    def _cpx(self, lvl, q):
        return (lvl["oy"] + q[0] * T + 2, lvl["ox"] + q[1] * T + 2)

    def click_hit(self, lvl, st, px):
        q = self._tile_at(lvl, px)
        return q is not None

    # ---------------------------------------------------------------- oracle
    def _plan(self, lvl, st):
        if self.is_win(lvl, st):
            return None
        tgt = lvl["target"]
        for q in lvl["order"]:
            k = lvl["cid"][q]
            if st[k] != tgt[k]:
                return ("click", self._cpx(lvl, q))
        return None

    def oracle_action(self):
        if self.state != "NOT_FINISHED":
            return 0, None
        L = self.levels[self.level]
        mv = self._plan(L["lvl"], self.st)
        if mv is None:
            return 0, None
        return self._to_action(mv[0], mv[1], self.st)

    def heuristic(self, lvl, st):
        if self.is_win(lvl, st):
            return 0
        nc = lvl["nc"]
        return sum((t - s) % nc for s, t in zip(st, lvl["target"]))

    # ---------------------------------------------------------------- génération
    def gen_level(self, rng, i, n):
        mods, newest = mods_for(self.cat, i)
        nc = (2 if i == 0 else 3) + ("colors3" in mods)
        Hb = int(min(6, 4 + (i >= 1) + (i >= 2)))
        Wb = int(min(7, 5 + (i >= 1) + (i >= 3) + int(rng.integers(0, 2)))) if i > 0 else 4
        cells = [(r, c) for r in range(Hb) for c in range(Wb)]
        # indices : jamais voisins (8-connexité)
        nh = max(1, int(round(Hb * Wb / (4.8 if i > 0 else 6))))
        hcells = []
        for k in rng.permutation(len(cells)):
            q = cells[int(k)]
            if all(max(abs(q[0] - p[0]), abs(q[1] - p[1])) > 1 for p in hcells):
                hcells.append(q)
            if len(hcells) >= nh:
                break
        cid, k = {}, 0
        for q in cells:
            if q in hcells:
                cid[q] = -1
            else:
                cid[q] = k
                k += 1
        nfree = k
        # assignation de départ de la solution
        sol = {q: int(rng.integers(nc)) for q in cells if cid[q] >= 0}
        locked = {}
        if "locked" in mods:
            fr = [q for q in cells if cid[q] >= 0]
            for j in rng.permutation(len(fr))[:2 + (i >= 4)]:
                locked[fr[int(j)]] = sol[fr[int(j)]]
        lvl = dict(Hb=Hb, Wb=Wb, cid=cid, nc=nc, locked=locked, mods=mods, newest=newest,
                   eseed=int(rng.integers(1 << 30)))
        lvl["cid"] = {q: cid[q] for q in cells}
        # sortes d'indices actives
        rels = [False] + ([True] if "second" in mods else [])
        nbs = ["orth"] + (["diag"] if "diag" in mods else [])
        fixed = set(locked)
        hints = []
        # contraintes de rangée / colonne (traitées d'abord)
        if "line" in mods:
            for _ in range(1 + (i >= 5)):
                if rng.random() < 0.5 or Hb > 4:
                    h = dict(nb="row", r=int(rng.integers(Hb)), c=-1)
                else:
                    h = dict(nb="col", r=-1, c=int(rng.integers(Wb)))
                if any(x["nb"] == h["nb"] and x["r"] == h["r"] and x["c"] == h["c"] for x in hints):
                    continue
                hints.append(h)
        for q in hcells:
            hints.append(dict(nb=str(rng.choice(nbs)), r=q[0], c=q[1]))
        # la règle nouvelle doit apparaître
        cellh = [h for h in hints if h["nb"] in ("orth", "diag")]
        if newest == "diag" and not any(h["nb"] == "diag" for h in cellh):
            cellh[int(rng.integers(len(cellh)))]["nb"] = "diag"
        for h in hints:
            h["neg"] = bool(rng.choice(rels))
            h["scope"] = tuple(self._scope(lvl, h))
        if newest == "second" and not any(h["neg"] for h in hints):
            hints[int(rng.integers(len(hints)))]["neg"] = True
        if any(len(h["scope"]) == 0 for h in hints):
            return self.why("empty_scope")
        # dériver les indices de la solution
        if self.variant == "match":
            for h in sorted(hints, key=lambda h: h["nb"] not in ("row", "col")):
                sc = h["scope"]
                fx = {sol[q] for q in sc if q in fixed}
                if not h["neg"]:
                    if len(fx) > 1:
                        return self.why("conflict")
                    c = fx.pop() if fx else int(rng.integers(nc))
                    for q in sc:
                        sol[q] = c
                else:
                    free_c = [c for c in range(nc) if c not in fx]
                    if not free_c:
                        return self.why("conflict")
                    c = int(rng.choice(free_c))
                    for q in sc:
                        if q not in fixed:
                            sol[q] = int(rng.choice([x for x in range(nc) if x != c]))
                h["c_"] = c
                fixed |= set(sc)
        else:
            for h in hints:
                h["c_"] = c = int(rng.integers(nc))
                if rng.random() < 0.6:                         # comptes extrêmes (0 ou tout) : plus contraignants
                    full = rng.random() < 0.5
                    for q in h["scope"]:
                        if q not in fixed:
                            sol[q] = c if full else int(rng.choice([x for x in range(nc) if x != c]))
                    fixed |= set(h["scope"])
            for h in hints:
                h["k"] = sum((sol[q] != h["c_"]) if h["neg"] else (sol[q] == h["c_"]) for q in h["scope"])
        for h in hints:
            h.setdefault("k", 0)
        for q, v in locked.items():
            if sol[q] != v:
                return self.why("locked")
        lvl["hints"] = tuple(hints)
        solv = tuple(sol[q] for q in cells if cid[q] >= 0)
        if not self.is_win(lvl, solv):
            return self.why("bad")
        # départ : brouillé (tout à 0 au début, aléatoire ensuite)
        cons = {q for h in hints for q in h["scope"]}
        if i == 0:
            st0 = tuple(0 for _ in range(nfree))
        else:                                                  # tuiles contraintes : une autre couleur que la solution
            st0 = tuple(((sol[q] + 1) % nc) if (q in cons and rng.random() < 0.9)
                        else int(rng.integers(nc)) for q in cells if cid[q] >= 0)
        st0 = tuple(locked[q] if q in locked else st0[cid[q]] for q in cells if cid[q] >= 0)
        if self.is_win(lvl, st0):
            return self.why("trivial")
        target = self._nearest(lvl, st0, solv)
        lvl["target"] = target
        lvl["order"] = tuple(q for q in cells if cid[q] >= 0 and q not in locked)
        self._geom(lvl, i)
        sol_moves, st = [], st0
        for _ in range(4 * nfree + 4):
            mv = self._plan(lvl, st)
            if mv is None:
                break
            sol_moves.append(mv)
            st, _ = self.apply(lvl, st, *mv)
        if not self.is_win(lvl, st):
            return self.why("plan")
        return lvl, st0, sol_moves

    def _nearest(self, lvl, st0, fallback):
        """assignation satisfaisante de coût de clics minimal depuis st0 (séparation-évaluation plafonnée)."""
        nc = lvl["nc"]
        cid = lvl["cid"]
        hints = lvl["hints"]
        var = []
        for h in hints:
            for q in h["scope"]:
                if q not in lvl["locked"] and q not in var:
                    var.append(q)
        by = {q: [j for j, h in enumerate(hints) if q in h["scope"]] for q in var}
        assign = {q: v for q, v in lvl["locked"].items()}
        best = [sum((f - s) % nc for s, f in zip(st0, fallback)), None]
        nodes = [0]

        def partial_ok(j):
            h = hints[j]
            vals = [assign[q] for q in h["scope"] if q in assign]
            un = len(h["scope"]) - len(vals)
            c = h["c_"]
            if self.variant == "match":
                return all((v != c) if h["neg"] else (v == c) for v in vals)
            m = sum((v != c) if h["neg"] else (v == c) for v in vals)
            return m <= h["k"] <= m + un

        def rec(t, cost):
            nodes[0] += 1
            if cost >= best[0] or nodes[0] > 40000:
                return
            if t == len(var):
                best[0], best[1] = cost, dict(assign)
                return
            q = var[t]
            s = st0[cid[q]]
            for d in range(nc):
                assign[q] = (s + d) % nc
                if all(partial_ok(j) for j in by[q]):
                    rec(t + 1, cost + d)
                del assign[q]

        rec(0, 0)
        if best[1] is None:
            return fallback
        out = list(st0)
        for q, v in best[1].items():
            if cid[q] >= 0:
                out[cid[q]] = v
        out = tuple(out)
        return out if self.is_win(lvl, out) else fallback

    def _geom(self, lvl, i):
        AH, AW = self.AH, self.AW
        Hb, Wb = lvl["Hb"], lvl["Wb"]
        band = 21 if any(h["nb"] == "col" for h in lvl["hints"]) else 16
        line = "line" in lvl["mods"]
        bh, bw = Hb * T + (6 if line else 0), Wb * T + (6 if line else 0)
        lvl["oy"] = band + 2 + (AH - band - 2 - bh) // 2 + (6 if line else 0)
        lvl["ox"] = (AW - bw) // 2 + (6 if line else 0)
        lvl["band"] = band

    def min_len(self, i, n):
        return min(6 + 4 * i, 34)

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        if st2 == st:
            if op == "click" and self._tile_at(lvl, px) is not None:
                ev[EV["touch"]] = 1
            return
        ev[EV["color"]] = 1
        a, b = sum(self.satisfied(lvl, st)), sum(self.satisfied(lvl, st2))
        if b > a:
            ev[EV["collect"]] = 1

    # ---------------------------------------------------------------- rendu
    def _cols(self):
        return (self.pal.a(0), self.pal.a(1), self.pal.a(2), self.pal.a(3))

    def _base(self, neg):
        """fond d'icône : sombre (indice positif) ou clair (négation), distinct du fond et des couleurs."""
        pal = self.pal
        cand = (draw.WHITE, draw.LGREY, draw.OFFW) if neg else (draw.DGREY, draw.BLACK, draw.GREY)
        for c in cand:
            if c != pal.bg and c not in self._cols():
                return c
        return pal.fg

    def _hint_sprite(self, h, t=T):
        """icône t×t d'un indice : 2-3 aplats. Voisinage orthogonal = bords colorés, diagonal = centre / coins."""
        c = self._cols()[h["c_"]]
        s = np.full((t, t), self._base(h["neg"]), np.int64)
        if self.variant == "match":
            if h["nb"] == "orth":
                s[0, :] = s[-1, :] = s[:, 0] = s[:, -1] = c
            else:
                s[1:-1, 1:-1] = c
        else:
            k = h["k"]
            if h["nb"] == "orth":
                for j, sl in enumerate(((0, slice(None)), (slice(None), 0), (slice(None), t - 1), (t - 1, slice(None)))):
                    if j < k:
                        s[sl] = c
            else:
                m = t // 2
                for j, (y, x) in enumerate(((0, 0), (0, t - m), (t - m, 0), (t - m, t - m))):
                    if j < k:
                        s[y:y + m, x:x + m] = c
        return s

    def _draw_board(self, g, lvl, colour_of, oy, ox, hints_at, t=T):
        cols = self._cols()
        for r in range(lvl["Hb"]):
            for c in range(lvl["Wb"]):
                y, x = oy + r * t, ox + c * t
                q = (r, c)
                if q in hints_at:
                    blit(g, self._hint_sprite(hints_at[q], t), y, x)
                    continue
                rect(g, y, x, t, t, cols[colour_of(q)])
                if q in lvl["locked"]:
                    rect(g, y + 1, x + 1, t - 2, t - 2, self._base(False))

    def _line_marker(self, g, h, oy, ox, t=T):
        """triangle vers la rangée / colonne (+ barre de k px pour count)."""
        col = self._cols()[h["c_"]]
        if h["nb"] == "row":
            y, x = oy + h["r"] * t + (t - 5) // 2, ox - 6
            for k, w in enumerate((1, 2, 3, 2, 1)):
                rect(g, y + k, x, 1, w, col)
            if self.variant == "count" and h["k"]:
                rect(g, y + 5 + 1, x, 1, h["k"], self._base(False))
        else:
            y, x = oy - 6, ox + h["c"] * t + (t - 5) // 2
            for k, w in enumerate((1, 2, 3, 2, 1)):
                rect(g, y, x + k, w, 1, col)
            if self.variant == "count" and h["k"]:
                rect(g, y, x + 6, h["k"], 1, self._base(False))
        if h["neg"]:
            if h["nb"] == "row":
                rect(g, y, x - 2, 5, 1, self._base(True))
            else:
                rect(g, y - 2, x, 1, 5, self._base(True))

    def _example(self, h, rng):
        """petit plateau résolu 3×3 (ou rangée 1×3) illustrant l'indice h -> (lvl_like, couleurs, indice)."""
        nc = self._nc
        if h["nb"] in ("row", "col"):
            Hb, Wb = (1, 3) if h["nb"] == "row" else (3, 1)
            cells = [(r, c) for r in range(Hb) for c in range(Wb)]
            lv = dict(Hb=Hb, Wb=Wb, locked={}, cid={q: 0 for q in cells})
            hh = dict(h, r=0, c=0, k=0)
            sc = cells
        else:
            cells = [(r, c) for r in range(3) for c in range(3)]
            lv = dict(Hb=3, Wb=3, locked={}, cid={q: (0 if q != (1, 1) else -1) for q in cells})
            hh = dict(h, r=1, c=1)
            dd = ORTH if h["nb"] == "orth" else DIAG
            sc = [(1 + a, 1 + b) for a, b in dd]
        col = {q: int(rng.integers(nc)) for q in cells}
        c = h["c_"]
        if self.variant == "match":
            for q in sc:
                col[q] = c if not h["neg"] else int(rng.choice([x for x in range(nc) if x != c]))
        else:
            k = int(rng.integers(1, len(sc) + 1))
            pick = set(int(x) for x in rng.permutation(len(sc))[:k])
            for j, q in enumerate(sc):
                hit = j in pick
                if h["neg"]:
                    col[q] = int(rng.choice([x for x in range(nc) if x != c])) if hit else c
                else:
                    col[q] = c if hit else int(rng.choice([x for x in range(nc) if x != c]))
            hh = dict(hh, k=k)
        return lv, col, hh

    def render(self, lvl, st):
        pal = self.pal
        g = self.blank()
        cols = self._cols()
        nc = lvl["nc"]
        self._nc = nc
        # en haut, sur le fond : 1-2 exemples résolus (tuiles de 4 px) + palette du cycle
        kinds, seen = [], set()
        for h in lvl["hints"]:
            key = (h["nb"] if h["nb"] in ("orth", "diag") else "line", h["neg"])
            if key not in seen:
                seen.add(key)
                kinds.append(h)
        nw = lvl.get("newest")

        def pri(h):
            return (not ((nw == "second" and h["neg"]) or (nw == "diag" and h["nb"] == "diag")
                         or (nw == "line" and h["nb"] in ("row", "col"))), h["nb"] in ("row", "col"), h["neg"])
        kinds = sorted(kinds, key=pri)[:2 if lvl["Hb"] * lvl["Wb"] <= 30 else 1]
        erng = np.random.default_rng(lvl.get("eseed", 0))
        x, te = 2, 4
        for h in kinds:
            lv, col, hh = self._example(h, erng)
            off = 6 if h["nb"] in ("row", "col") else 0
            ox = x + (off if h["nb"] == "row" else 0)
            oy = 2 + (off if h["nb"] == "col" else 0)
            if h["nb"] in ("row", "col"):
                self._draw_board(g, lv, lambda q: col[q], oy, ox, {}, te)
                self._line_marker(g, hh, oy, ox, te)
            else:
                self._draw_board(g, lv, lambda q: col[q], oy, ox, {(1, 1): hh}, te)
            x = ox + lv["Wb"] * te + 5
        for k in range(nc):                                    # palette du cycle (verticale, à droite)
            rect(g, 2 + 5 * k, self.AW - 6, 4, 4, cols[k])
        # plateau (tuiles jointives)
        hints_at = {(h["r"], h["c"]): h for h in lvl["hints"] if h["nb"] in ("orth", "diag")}
        oy, ox = lvl["oy"], lvl["ox"]
        self._draw_board(g, lvl, lambda q: self._colour_of(lvl, st, q), oy, ox, hints_at)
        for h in lvl["hints"]:
            if h["nb"] in ("row", "col"):
                self._line_marker(g, h, oy, ox)
        return g
