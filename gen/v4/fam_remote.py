"""remote4 (avatar + télécommande) : l'avatar marche sur des îlots ; les PASSERELLES entre îlots appartiennent à des
GROUPES de couleur ; un PANNEAU latéral de boutons (clic) bascule tout un groupe : ses passerelles « paires »
apparaissent et ses passerelles « impaires » disparaissent (ou l'inverse). Un bouton qui ferait disparaître la
passerelle sous l'avatar ne fait rien. Le panneau porte aussi des décors inertes (glyphes, voyants).
Variantes (tirées par jeu) :
  remote : le panneau est à droite ; ACTION5 = attendre un tour ;
  dock   : le panneau est en bas et les boutons ne marchent que si l'avatar est sur un îlot (pas sur une passerelle).
But : atteindre la sortie (verrouillée tant que la clé n'est pas prise, quand la règle key est active).
Règles ajoutées, une par niveau (ordre tiré par jeu ; niveau 0 = un groupe) :
  group2  : un 2e groupe de passerelles et son bouton ;
  key     : une clé à ramasser avant la sortie ;
  carrier : une plate-forme mobile fait la navette au-dessus du vide (un pas par action) et transporte l'avatar ;
  plate   : un bouton n'est ALLUMÉ (actif) que si l'avatar se tient sur la plaque de sa couleur ;
  group3  : un 3e groupe.
Solvabilité : BFS exact sur (case, groupes, clé, phase de la navette) ; sortie placée au plus loin."""
import numpy as np

from ..core import DIRS, DIR_OPS, EV
from . import avkit as K
from . import draw
from .base import V4Game, mods_for
from .draw import R, blit, frame_rect, rect, sprite


class Remote4(V4Game):
    family = "remote4"
    mechanic = "avatar"
    principle = ("groupes de passerelles basculés par les boutons colorés d'un panneau latéral (télécommande) ; "
                 "clé, navette, plaques, boutons allumés sous condition", ["dc22", "g50t", "m0r0", "sc25"])
    simple_ops = DIR_OPS + ("interact",)
    CAT = ("group2", "key", "carrier", "plate", "group3")
    exact_progress = True
    table_cap = 25000
    scale_p = (1.0, 0.0)                    # plateaux larges : échelle ×1
    bfs_cap = 40000

    def setup(self, rng):
        r = self.orng
        self.uses_click = True
        self.variant = str(rng.choice(["remote", "dock"]))
        cat = [str(x) for x in rng.permutation(self.CAT)]
        if cat.index("group3") < cat.index("group2"):
            a, b = cat.index("group2"), cat.index("group3")
            cat[a], cat[b] = cat[b], cat[a]
        self.cat = cat
        self.C = int(r.choice([5, 5, 6])) if self.s_screen == 1 else 3
        s1 = self.s_screen == 1
        self.pw = (11 if s1 else 7) if self.variant == "remote" else (10 if s1 else 6)
        self.panel_bg = bool(r.random() < 0.5)                 # panneau de la couleur du fond (séparé par un trait)
        self.bsz = 5 if s1 else 3
        self.player_kind = str(r.choice(["blob", "framed", "disc"]))
        self.tpl = {S: sprite(r, self.player_kind, S, S, 100, 101, 102 if S >= 5 else None)
                    for S in range(3, self.C + 5)}
        self.glyphs = [sprite(r, "glyph", self.bsz - 1, self.bsz - 1, 100) for _ in range(3)]
        self.btn_kind = str(r.choice(["hat", "box", "framed"]))

    # ---------------------------------------------------------------- géométrie / panneau
    def _panel_rect(self):
        if self.variant == "remote":
            return 0, self.AW - self.pw, self.AH, self.pw
        return self.AH - self.pw, 0, self.pw, self.AW

    def _buttons(self, lvl):
        """[(y, x, groupe ou None pour décor)]"""
        return lvl["btns"]

    def _layout_panel(self, rng, ng, ndec):
        y0, x0, h, w = self._panel_rect()
        b = self.bsz
        slots = []
        if self.variant == "remote":
            yy = y0 + 3
            while yy + b < y0 + h - 1:
                slots.append((yy, x0 + (w - b) // 2 + 1))
                yy += b + 3
        else:
            xx = x0 + 3
            while xx + b < x0 + w - 1:
                slots.append((y0 + (h - b) // 2 + 1, xx))
                xx += b + 3
        kinds = list(range(ng)) + [None] * ndec
        kinds = kinds[:len(slots)]
        order = rng.permutation(len(slots))[:len(kinds)]
        out = []
        for k, j in zip(kinds, sorted(order)):
            out.append((slots[j][0], slots[j][1], k))
        rng.shuffle(out)
        return tuple(sorted(out))

    def _btn_at(self, lvl, px):
        if px is None:
            return None
        b = self.bsz
        for (y, x, k) in lvl["btns"]:
            if y <= px[0] < y + b and x <= px[1] < x + b:
                return (y, x, k)
        return None

    # ---------------------------------------------------------------- règles
    def _solid(self, lvl, q, bits, t):
        if not (0 <= q[0] < lvl["Hc"] and 0 <= q[1] < lvl["Wc"]):
            return False
        if lvl["floor"][q]:
            return True
        br = lvl["bridges"].get(q)
        if br is not None and (bits >> br[0] & 1) == br[1]:
            return True
        return lvl["carrier"] and self._cpos(lvl, t) == q

    def _cpos(self, lvl, t):
        path = lvl["carrier"]
        return path[t % len(path)] if path else None

    def _lit(self, lvl, st, k):
        p = st[0]
        if self.variant == "dock" and p in lvl["bridges"]:
            return False
        if k in lvl["plate_of"]:
            return p == lvl["plate_of"][k]
        return True

    def _step(self, lvl, st, op, px):
        p, bits, key, t = st
        tags = set()
        if op == "click":
            b = self._btn_at(lvl, px)
            if b is None or b[2] is None:
                tags.add("inert" if b is not None else "miss")
                return st, tags
            k = b[2]
            if not self._lit(lvl, st, k):
                tags.add("dark")
                return st, tags
            nb = bits ^ (1 << k)
            if not self._solid(lvl, p, nb, t) and not (lvl["carrier"] and self._cpos(lvl, t) == p):
                tags.add("refused")                             # la passerelle sous l'avatar ne disparaît pas
                return st, tags
            tags.add("toggle")
            bits = nb
            q = p
        elif op == "interact":
            if not lvl["carrier"]:
                return st, tags
            q = p
            tags.add("wait")
        else:
            d = DIRS[op]
            q = (p[0] + d[0], p[1] + d[1])
            if not self._solid(lvl, q, bits, t):
                tags.add("bump")
                return st, tags
            if q == lvl["exit"] and lvl["key"] is not None and not key:
                tags.add("locked")
                return st, tags
        if lvl["key"] is not None and q == lvl["key"] and not key:
            key = 1
            tags.add("key")
        if lvl["carrier"]:                                      # la navette avance d'un pas par action
            c0 = self._cpos(lvl, t)
            t = (t + 1) % len(lvl["carrier"])
            c1 = self._cpos(lvl, t)
            if q == c0:                                         # sur la navette : elle l'emporte
                q = c1
                tags.add("ride")
        return (q, bits, key, t), tags

    def apply(self, lvl, st, op, px):
        return self._step(lvl, st, op, px)[0], False

    def is_win(self, lvl, st):
        return st[0] == lvl["exit"]

    def click_targets(self, lvl, st):
        b = self.bsz
        return [(y + b // 2, x + b // 2) for (y, x, k) in lvl["btns"] if k is not None and self._lit(lvl, st, k)]

    def click_hit(self, lvl, st, px):
        return self._btn_at(lvl, px) is not None

    # ---------------------------------------------------------------- niveaux
    def gen_level(self, rng, i, n):
        mods, newest = mods_for(self.cat, i, keep=3)
        ng = 1 + ("group2" in mods or "group3" in mods or i >= 4) + ("group3" in mods)
        y0, x0, ph, pw = self._panel_rect()
        if self.variant == "remote":
            area = dict(top=0, bottom=0, left=0, right=self.pw + 1)
        else:
            area = dict(top=0, bottom=self.pw + 1, left=0, right=0)
        C0 = self.C if i < 1 else max(3 if self.s_screen == 2 else 4, self.C - 1 - (i >= 4))   # plateaux plus fins
        C, Hc, Wc, oy, ox = K.board(self.AH, self.AW, C0, 999, **area)
        base = K.make_floor(rng, Hc, Wc, "rooms", n_rooms=3, need=int(0.45 * Hc * Wc), holes=0.0,
                            max_side=max(4, min(Hc, Wc) - 2), maxc=int(0.75 * Hc * Wc))
        if base is None:
            return self.why("floor")
        # couper le sol en îlots : chaque coupe laisse une passerelle (groupe, phase) ; le reste de la coupe = vide
        # coupes parallèles en serpentin : une chaîne d'îlots, passerelles alternativement en haut / en bas
        vert = Wc >= Hc if rng.random() < 0.8 else Wc < Hc
        fl = base if vert else base.T
        H2, W2 = fl.shape
        colsx = [x for x in range(W2) if fl[:, x].any()]
        if len(colsx) < 5:
            return self.why("narrow")
        ncut = max(1, min(2 + (i >= 1) + (i >= 2) + (i >= 4), (len(colsx) - 1) // 3))
        xs = [colsx[0] + int(round((k + 1) * len(colsx) / (ncut + 1))) for k in range(ncut)]
        xs = sorted(set(min(colsx[-1] - 1, max(colsx[0] + 1, x + int(rng.integers(-1, 2)))) for x in xs))
        work = fl.copy()
        bridges, lines = {}, []
        groups = []
        for j, x in enumerate(xs):
            ok = [r for r in range(H2) if work[r, x] and work[r, x - 1] and work[r, x + 1]]
            if not ok:
                return self.why("cut")
            third = max(1, len(ok) // 3)
            pick = ok[:third] if j % 2 == 0 else ok[-third:]
            r = pick[int(rng.integers(len(pick)))]
            line = [(y, x) for y in range(H2) if work[y, x] and y != r]
            for q in line:
                work[q] = False
            g = j % ng if j < ng else int(rng.integers(ng))
            groups.append(g)
            lines.append([(y, xx) if vert else (xx, y) for (y, xx) in line] + [(r, x) if vert else (x, r)])
            bridges[(r, x)] = g
        anyc = K.cells(work)[0]
        if len(K.grid_dist(work, anyc)) != int(work.sum()):
            return self.why("split")
        if not vert:
            work = work.T
            bridges = {(q[1], q[0]): g for q, g in bridges.items()}
        # phases : deux passerelles successives du même groupe ont des phases opposées (il faut rebasculer)
        last = {}
        ph = {}
        for bq in sorted(bridges, key=lambda q: q[1] if vert else q[0]):
            g = bridges[bq]
            ph[bq] = (1 - last[g]) if g in last else int(rng.integers(2))
            last[g] = ph[bq]
        bridges = {q: (bridges[q], ph[q]) for q in bridges}
        floor = work.copy()
        for q in bridges:
            floor[q] = False
        lvl = dict(floor=floor, Hc=Hc, Wc=Wc, oy=oy, ox=ox, C=C, mods=mods, bridges=bridges, key=None, exit=None,
                   carrier=(), plate_of={}, ng=ng)
        isl = K.cells(floor)
        rng.shuffle(isl)
        used = set()
        if "carrier" in mods:                                   # navette : glisse le long d'une ligne de coupe
            cands = []
            for li, ln in enumerate(lines):
                lset = set(ln)
                for q in ln:
                    for d in ((0, 1), (1, 0)):
                        seg, y, x = [], q[0], q[1]
                        while (y, x) in lset and len(seg) < 5:
                            seg.append((y, x))
                            y, x = y + d[0], x + d[1]
                        if len(seg) >= 3:
                            cands.append((li, seg))
            if not cands:
                return self.why("carrier")
            li, seg = cands[int(rng.integers(len(cands)))]
            lvl["carrier"] = tuple(seg + seg[-2:0:-1])
            bq = lines[li][-1]                                  # cette coupe n'a plus de passerelle : seule la navette
            if len(bridges) > 2:
                del bridges[bq]
        if "plate" in mods:
            k = int(rng.integers(ng))
            pl = isl[0]
            lvl["plate_of"] = {k: pl}
            used.add(pl)
        rest = [q for q in isl if q not in used]
        axis = 1 if vert else 0
        lo = min(q[axis] for q in rest)
        first = [q for q in rest if q[axis] <= lo + 1]
        b0 = min(bridges, key=lambda q: q[axis])
        p0 = max(first, key=lambda q: abs(q[0] - b0[0]) + abs(q[1] - b0[1]) + rng.random())
        used.add(p0)
        if "key" in mods:
            dk = K.grid_dist(floor, p0)
            lastx = max(q[axis] for q in bridges)
            kc = [q for q in rest[1:] if (q not in dk or dk[q] > 3) and q[axis] < lastx]
            if not kc:
                return self.why("key")
            bd = K.grid_dist(floor | np.isin(np.arange(floor.size).reshape(floor.shape),
                                             [q[0] * floor.shape[1] + q[1] for q in bridges]), list(bridges))
            kq = max(kc, key=lambda q: bd.get(q, 0) + 3 * rng.random())   # clé dans un recoin, loin des passerelles
            lvl["key"] = kq
            used.add(kq)
        lvl["btns"] = self._layout_panel(rng, ng, int(rng.integers(2, 5)))
        if sum(1 for b in lvl["btns"] if b[2] is not None) < ng:
            return self.why("panel")
        st0 = (p0, int(rng.integers(1 << ng)), 0, 0)
        depth, prev = K.explore(self, lvl, st0, 25000)
        if depth is None:
            return self.why("cap")
        best = {}
        for s, dd in depth.items():
            if (lvl["key"] is None or s[2]) and s[0] not in used and lvl["floor"][s[0]]:
                if s[0] not in best or dd < depth[best[s[0]]]:
                    best[s[0]] = s
        if not best:
            return self.why("noexit")
        items = sorted(best.items(), key=lambda x: -depth[x[1]])
        need = dict(carrier="ride", key="key").get(newest)
        for t in range(10):
            e, s = items[int(rng.integers(min(len(items), 3 + 2 * t)))]
            lvl["exit"] = e
            sol = K.path_to(prev, s)
            x = st0
            for mv in sol:
                x, _ = self.apply(lvl, x, *mv)
            if not self.is_win(lvl, x):                         # la sortie verrouillée gêne ce chemin : re-résoudre
                sol = self.solve(lvl, st0, 30000)
                if sol is None:
                    continue
            tg = K.trace_tags(self, lvl, st0, sol)
            if "toggle" not in tg or (need and need not in tg):
                continue
            if newest == "plate":
                k = next(iter(lvl["plate_of"]))
                if not any(mv[0] == "click" and self._btn_at(lvl, mv[1])[2] == k for mv in sol):
                    continue
            break
        else:
            return self.why("unused")
        return lvl, st0, sol

    def min_len(self, i, n):
        return min(16 + 5 * i, 50)

    def heuristic(self, lvl, st):
        p, e = st[0], lvl["exit"]
        return abs(p[0] - e[0]) + abs(p[1] - e[1])

    def body(self, lvl, st):
        return (st[0][0], st[0][1], lvl["Hc"], lvl["Wc"])

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        _, tags = self._step(lvl, st, op, px)
        if "bump" in tags or "locked" in tags:
            ev[EV["touch"]] = 1
        if "toggle" in tags:
            ev[EV["toggle"]] = ev[EV["color"]] = 1
        if "key" in tags:
            ev[EV["collect"]] = 1
        if "ride" in tags:
            ev[EV["push"]] = 1
        if "refused" in tags or "dark" in tags:
            ev[EV["danger"]] = 1
        if st2[0] == lvl["exit"]:
            ev[EV["open"]] = 1

    # ---------------------------------------------------------------- rendu
    def _gcol(self, k):
        return self.pal.a(k)

    def render(self, lvl, st):
        pal = self.pal
        C = lvl["C"]
        p, bits, key, t = st
        g = self.blank()
        floor = lvl["floor"]
        K.draw_walls(g, lvl, pal.wall, 2 if C >= 5 else 1)
        for q, (k, ph) in lvl["bridges"].items():
            y, x = K.cell_xy(lvl, q)
            col = self._gcol(k)
            if (bits >> k & 1) == ph:
                rect(g, y, x, C, C, col)
            else:
                rect(g, y, x, C, C, pal.bg)
                frame_rect(g, y + (C >= 5), x + (C >= 5), C - 2 * (C >= 5), C - 2 * (C >= 5), col)
        if lvl["carrier"]:
            cp = self._cpos(lvl, t)
            path = lvl["carrier"]
            (ya, xa), (yb, xb) = K.cell_center(lvl, path[0]), K.cell_center(lvl, path[len(path) // 2])
            draw.line(g, ya, xa, yb, xb, pal.line)
            y, x = K.cell_xy(lvl, cp)
            blit(g, sprite(None, "framed", C, C, pal.room, pal.fg), y, x)
        for k, pl in lvl["plate_of"].items():
            y, x = K.cell_xy(lvl, pl)
            blit(g, sprite(None, "plus_outline" if C >= 5 else "plus", C, C, self._gcol(k)), y, x)
        e = lvl["exit"]
        if e is not None:
            y, x = K.cell_xy(lvl, e)
            locked = lvl["key"] is not None and not key
            blit(g, sprite(None, "frame" if locked else "framed", C, C, pal.fg, pal.a(5)), y, x)
            if not locked:
                rect(g, y + 1, x + 1, C - 2, C - 2, pal.a(5))
        if lvl["key"] is not None and not key:
            y, x = K.cell_xy(lvl, lvl["key"])
            blit(g, sprite(None, "T" if C >= 4 else "box", C - 1, C - 1, pal.a(6)), y, x)
        y, x = K.cell_xy(lvl, p)
        blit(g, draw.recolor(self.tpl[C], {100: pal.a(7), 101: pal.fg, 102: pal.fg}), y, x)
        self._panel(g, lvl, st)
        return g

    def _panel(self, g, lvl, st):
        pal = self.pal
        y0, x0, h, w = self._panel_rect()
        if not self.panel_bg:
            rect(g, y0, x0, h, w, pal.ui)
        if self.variant == "remote":
            draw.vline(g, x0 - 1, 0, self.AH - 1, pal.line)
        else:
            draw.hline(g, y0 - 1, 0, self.AW - 1, pal.line)
        b = self.bsz
        dk = 0
        for (y, x, k) in lvl["btns"]:
            if k is None:
                gl = self.glyphs[dk % len(self.glyphs)]
                dk += 1
                blit(g, draw.recolor(gl, {100: pal.wall if pal.wall != pal.ui else pal.line}), y, x)
                continue
            col = self._gcol(k)
            lit = self._lit(lvl, st, k)
            if lit:
                blit(g, sprite(None, self.btn_kind, b, b, col, pal.fg), y, x)
                g[y + b // 2, x + b // 2] = R(draw.WHITE if (st[1] >> k & 1) else pal.fg)
            else:
                frame_rect(g, y, x, b, b, col)
        if lvl["key"] is not None:                              # case clé du panneau
            if self.variant == "remote":
                yy, xx = y0 + h - b - 2, x0 + (w - b) // 2 + 1
            else:
                yy, xx = y0 + (h - b) // 2 + 1, x0 + w - b - 2
            frame_rect(g, yy, xx, b, b, pal.line)
            if st[2]:
                rect(g, yy + 1, xx + 1, b - 2, b - 2, pal.a(6))
