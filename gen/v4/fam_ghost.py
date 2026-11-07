"""ghost4 (avatar, enregistrer puis rejouer) : ACTION5 clôt l'essai en cours : l'avatar revient au départ et son
trajet (au plus 16 pas, attentes comprises : un pas contre un mur fait passer le temps) devient un FANTÔME qui le
rejoue en même temps que les essais suivants. Les fantômes (et l'avatar) tiennent les PLAQUES, qui ouvrent les
grilles de leur couleur (fils pointillés). But : l'avatar (pas un fantôme) atteint la sortie. Deux emplacements de
fantômes (bandeau) ; ACTION5 quand ils sont pleins efface tous les fantômes.
Variantes (tirées par jeu) :
  once : un fantôme rejoue son trajet une fois puis reste immobile à sa dernière case ;
  loop : un fantôme rejoue son trajet en boucle (il revient au départ et recommence).
Règles ajoutées, une par niveau (ordre tiré par jeu ; niveau 0 = une plaque, une grille) :
  two   : deux paires plaque / grille en série (deux fantômes) ;
  timed : la grille ne reste ouverte que 4 tours après l'appui (il faut faire arriver le fantôme au bon moment) ;
  dual  : une grille à deux plaques qu'il faut tenir EN MÊME TEMPS ;
  solid : les fantômes sont solides : l'avatar ne peut ni entrer dans un fantôme ni marcher dans ses pas.
Solvabilité : plan construit (trajets d'enregistrement « aller à la plaque, attendre » énumérés, dernier essai par BFS
exact sur (case, temps, minuteries)), rejoué et vérifié ; l'oracle suit ce plan (RESET s'il s'en écarte)."""
from collections import deque
from itertools import permutations
from math import gcd

import numpy as np

from ..core import DIRS, DIR_OPS, EV
from . import avkit as K
from . import draw
from .base import V4Game, mods_for
from .draw import R, blit, frame_rect, rect, sprite

LMAX = 16          # pas enregistrés par essai
TOPEN = 4          # tours d'ouverture d'une grille minutée


def lcm(a, b):
    return a * b // gcd(a, b)


class Ghost4(V4Game):
    family = "ghost4"
    mechanic = "avatar"
    principle = ("enregistrer puis rejouer : chaque essai devient un fantôme qui tient des plaques ; grilles colorées, "
                 "minutées, à deux plaques", ["g50t", "m0r0", "sb26"])
    simple_ops = DIR_OPS + ("interact",)
    CAT = ("two", "timed", "dual", "solid")
    exact_progress = False
    scale_p = (0.95, 0.05)
    bfs_cap = 40000

    def setup(self, rng):
        r = self.orng
        self.variant = str(rng.choice(["once", "once", "loop"]))
        self.cat = [str(x) for x in rng.permutation(self.CAT)]
        self.C = int(r.choice([4, 5, 5])) if self.s_screen == 1 else 3
        self.hh = 6 if self.s_screen == 1 else 4
        self.player_kind = str(r.choice(["framed", "blob", "disc"]))
        self.tpl = {S: sprite(r, self.player_kind, S, S, 100, 101, 102 if S >= 5 else None)
                    for S in range(3, self.C + 5)}

    def table(self, li):
        return None

    # ---------------------------------------------------------------- dynamique
    def _gpos(self, gh, t):
        if self.variant == "once":
            return gh[min(t, len(gh) - 1)]
        return gh[t % len(gh)]

    def _held(self, lvl, cells_on):
        return tuple(int(pl in cells_on) for pl in lvl["plates"])

    def _open(self, lvl, held, timers, occupied):
        """grilles ouvertes (indices) selon les plaques tenues, les minuteries et l'occupation."""
        out = set()
        for k, (gc, need) in enumerate(lvl["gates"]):
            ok = all(held[j] for j in need)
            if ok and lvl["timed"][k]:
                ok = timers[k] <= TOPEN
            if ok or gc in occupied:
                out.add(k)
        return out

    def _tnext(self, t, ghosts):
        if not ghosts:                                          # sans fantôme, le temps ne compte pas
            return 0
        if self.variant == "once":
            return min(t + 1, LMAX + 1)
        L = 1
        for gh in ghosts:
            L = lcm(L, len(gh))
        return (t + 1) % L

    def _tick(self, lvl, p, t, timers, ghosts, rec):
        """après un mouvement (ou une attente) : nouveau temps, minuteries."""
        t2 = self._tnext(t, ghosts)
        on = {p} | {self._gpos(gh, t2) for gh in ghosts}
        held = self._held(lvl, on)
        tm = []
        for k, (gc, need) in enumerate(lvl["gates"]):
            h = all(held[j] for j in need)
            tm.append(min(timers[k] + 1, TOPEN + 1) if h else 0)
        rec2 = rec + (p,) if len(rec) <= LMAX else rec
        return t2, tuple(tm), rec2

    def _step(self, lvl, st, op, px):
        p, t, timers, ghosts, rec = st
        tags = set()
        if op == "interact":                                    # fin d'essai : rembobiner
            if len(rec) < 2 and len(ghosts) < 2:
                return st, tags
            tags.add("rewind")
            if len(ghosts) >= 2:
                ng = ()
                tags.add("clear")
            else:
                ng = ghosts + (rec,)
            s0 = lvl["start"]
            on = {s0} | {self._gpos(gh, 0) for gh in ng}
            held = self._held(lvl, on)
            tm = tuple(int(all(held[j] for j in need)) for (gc, need) in lvl["gates"])
            return (s0, 0, tm, ng, (s0,)), tags
        if op not in DIRS:
            return st, tags
        d = DIRS[op]
        q = (p[0] + d[0], p[1] + d[1])
        gcells = {self._gpos(gh, t) for gh in ghosts}
        on = {p} | gcells
        opened = self._open(lvl, self._held(lvl, on), timers, {p})
        blocked = not (0 <= q[0] < lvl["Hc"] and 0 <= q[1] < lvl["Wc"]) or not lvl["floor"][q]
        if not blocked:
            for k, (gc, need) in enumerate(lvl["gates"]):
                if gc == q and k not in opened:
                    blocked = True
                    tags.add("gate")
        if not blocked and lvl["solid"] and (q in gcells or q in {self._gpos(gh, self._tnext(t, ghosts)) for gh in ghosts}):
            blocked = True
            tags.add("ghost")
        if blocked:
            tags.add("bump")
            q = p                                               # le temps passe quand même (attente)
        t2, tm, rec2 = self._tick(lvl, q, t, timers, ghosts, rec)
        if q in lvl["plates"] and q != p:
            tags.add("plate")
        return (q, t2, tm, ghosts, rec2), tags

    def apply(self, lvl, st, op, px):
        return self._step(lvl, st, op, px)[0], False

    def is_win(self, lvl, st):
        return st[0] == lvl["exit"]

    def heuristic(self, lvl, st):
        L = lvl.get("sol_left", {})
        if st in L:
            return L[st]
        p, e = st[0], lvl["exit"]
        return abs(p[0] - e[0]) + abs(p[1] - e[1]) + 8

    # ---------------------------------------------------------------- planification
    def _proj(self, st):
        return st[0], st[1], st[2]

    def _run_bfs(self, lvl, st, goal, tmax=40):
        """BFS (flèches seulement) sur (case, temps, minuteries) jusqu'à goal(état) ; -> liste d'actions ou None."""
        prev = {self._proj(st): None}
        q = deque([(st, 0)])
        while q:
            s, dd = q.popleft()
            if goal(s):
                out, k = [], self._proj(s)
                while prev[k] is not None:
                    k, mv = prev[k]
                    out.append(mv)
                return out[::-1], s
            if dd >= tmax:
                continue
            for op in DIR_OPS:
                s2, _ = self._step(lvl, s, op, None)
                k2 = self._proj(s2)
                if k2 not in prev:
                    prev[k2] = (self._proj(s), (op, None))
                    q.append((s2, dd + 1))
        return None, None

    def _replay(self, lvl, st, acts):
        for mv in acts:
            st, _ = self._step(lvl, st, *mv)
        return st

    def _plan(self, lvl, st0):
        plates = lvl["plates"]
        best = None
        holds = (0, 3, 6) if self.variant == "loop" else (0,)
        orders = list(permutations(range(len(plates))))
        memo = {}
        combos = [[]]
        for j in range(len(plates)):
            tj = any(lvl["timed"][k] for k, (gc, need) in enumerate(lvl["gates"]) if j in need)
            waits = ((0, 1, 2, 3, 5, 7) if len(plates) == 1 else (0, 2, 4, 6)) if tj else (0,)
            combos = [c + [(w, h)] for c in combos for w in waits for h in holds]
        for order in orders:
            for combo in combos:
                st, acts, ok = st0, [], True
                for j, (w, h) in zip([order[k] for k in range(len(order))], [combo[o] for o in order]):
                    pl = plates[j]
                    nb = lvl["plate_in"][j]                     # case voisine de la plaque (côté accès)
                    key = (st, nb)
                    if key not in memo:
                        memo[key] = self._run_bfs(lvl, st, lambda s, nb=nb: s[0] == nb)
                    path, s1 = memo[key]
                    if path is None:
                        ok = False
                        break
                    seq = path + [(lvl["wait_op"][j][0], None)] * w + [(lvl["wait_op"][j][1], None)] + \
                        [(lvl["wait_op"][j][2], None)] * h
                    s2 = self._replay(lvl, st, seq)
                    if s2[0] != pl or s2[4][-1] != pl:
                        ok = False
                        break
                    acts += seq + [("interact", None)]
                    st = self._replay(lvl, s2, [("interact", None)])
                if not ok:
                    continue
                key = (st, "exit")
                if key not in memo:
                    memo[key] = self._run_bfs(lvl, st, lambda s: s[0] == lvl["exit"], tmax=45)
                path, sw = memo[key]
                if path is None:
                    continue
                acts = acts + path
                if best is None or len(acts) < len(best):
                    best = acts
        return best

    # ---------------------------------------------------------------- niveaux
    def gen_level(self, rng, i, n):
        mods, newest = mods_for(self.cat, i, keep=2)
        mods = K.prune(mods, ({"two", "dual"},))
        top = self.hh
        maxc = (46 if self.s_screen == 1 else 40) + 7 * min(i, 5)
        C, Hc, Wc, oy, ox = K.board(self.AH, self.AW, self.C, maxc / 0.5 + 8, top=top, grow=3)
        floor = K.make_floor(rng, Hc, Wc, "rooms", n_rooms=3, need=min(int(maxc * 0.75), int(0.45 * Hc * Wc)),
                             holes=0.14 if "solid" in mods else 0.0, max_side=max(4, min(Hc, Wc) - 2), maxc=maxc + 12)
        if floor is None:
            return self.why("floor")
        two = "two" in mods
        ww = K.wall_with_gap(rng, floor, gap=(1, 1), min_part=10 if two else 7, tries=24)
        if ww is None:
            return self.why("wall")
        floor, run1, A, B = ww
        if len(A) < len(B):
            A, B = B, A
        gates_c = [run1[0]]
        last = B
        mid = None
        if two:
            sub = np.zeros_like(floor)
            for p in B:
                sub[p] = True
            ww2 = K.wall_with_gap(rng, sub, gap=(1, 1), min_part=3, tries=24)
            if ww2 is None:
                return self.why("wall2")
            sub2, run2, A2, B2 = ww2
            for p in B:
                if not sub2[p]:
                    floor[p] = False
            near = [q for q in K.nbrs(floor, run1[0]) if q in A2 or q in B2]
            if not near:
                return self.why("adj")
            if near[0] in B2:
                A2, B2 = B2, A2
            gates_c.append(run2[0])
            mid, last = A2, B2
        # plaques : contre un mur (pour attendre en butant), loin des grilles
        dg = K.grid_dist(floor, gates_c[0])
        Al = [p for p in A if p not in gates_c]
        rng.shuffle(Al)
        nplates = 2 if (two or "dual" in mods) else 1
        plates, plate_in, wait_op = [], [], []
        far = [p for p in Al if dg.get(p, 0) >= 3]
        if not far:
            return self.why("start")
        start = max(far[:6], key=lambda p: dg.get(p, 0))
        Al = [p for p in Al if p != start]
        if "solid" in mods:                                     # plaque sur le chemin : le fantôme posé gêne
            d_sg = K.grid_dist(floor, start)
            base = d_sg.get(gates_c[0], 99)
            detour = []
            for p in Al:
                if d_sg.get(p, 99) + dg.get(p, 99) != base:
                    continue
                f2 = floor.copy()
                f2[p] = False
                dd = K.grid_dist(f2, start).get(gates_c[0], 99)
                if base < dd < 99:
                    detour.append(p)
            if not detour:
                return self.why("solid_place")
            Al = detour[:1] + [p for p in Al if p != detour[0]]
        for p in Al:
            if len(plates) >= nplates:
                break
            if dg.get(p, 0) < 3 or any(abs(p[0] - q[0]) + abs(p[1] - q[1]) < 2 for q in plates) or \
                    not (min(4, LMAX // 3) <= abs(p[0] - start[0]) + abs(p[1] - start[1]) <= LMAX - 3):
                continue
            nbs = list(K.nbrs(floor, p))
            walls = [o for o, d in DIRS.items() if not (0 <= p[0] + d[0] < Hc and 0 <= p[1] + d[1] < Wc)
                     or not floor[p[0] + d[0], p[1] + d[1]]]
            if not walls or not nbs:
                continue
            nb = nbs[int(rng.integers(len(nbs)))]
            # attendre sur la case voisine : buter contre un mur depuis nb (sinon pas d'attente possible)
            wn = [o for o, d in DIRS.items() if not (0 <= nb[0] + d[0] < Hc and 0 <= nb[1] + d[1] < Wc)
                  or not floor[nb[0] + d[0], nb[1] + d[1]]]
            if not wn:
                continue
            step_in = next(o for o, d in DIRS.items() if (nb[0] + d[0], nb[1] + d[1]) == p)
            plates.append(p)
            plate_in.append(nb)
            wait_op.append((wn[0], step_in, walls[0]))
        if len(plates) < nplates:
            return self.why("plates")
        if "dual" in mods:
            gates = [(gates_c[0], (0, 1))]
        elif two:
            gates = [(gates_c[0], (0,)), (gates_c[1], (1,))]
        else:
            gates = [(gates_c[0], (0,))]
        timed = tuple("timed" in mods and (k == len(gates) - 1) for k in range(len(gates)))
        dl = K.grid_dist(floor, gates_c[-1])
        ex = [p for p in last if p != gates_c[-1]]
        if not ex:
            return self.why("exit")
        exitc = max(ex, key=lambda p: dl.get(p, 0) + rng.random())
        lvl = dict(floor=floor, Hc=Hc, Wc=Wc, oy=oy, ox=ox, C=C, mods=mods, plates=tuple(plates),
                   plate_in=tuple(plate_in), wait_op=tuple(wait_op), gates=tuple(gates), timed=timed,
                   solid="solid" in mods, start=start, exit=exitc)
        st0 = (start, 0, tuple(0 for _ in gates), (), (start,))
        sol = self._plan(lvl, st0)
        if sol is None:
            return self.why("noplan")
        if newest == "solid":                                   # la règle doit allonger le plan
            l2 = dict(lvl, solid=False)
            s2 = self._plan(l2, st0)
            if s2 is not None and len(s2) >= len(sol) and self.__dict__.setdefault("_sf", [0])[0] < 25:
                self._sf[0] += 1
                return self.why("solid_useless")
        left, st = {}, st0
        for k, mv in enumerate(sol):
            left.setdefault(st, len(sol) - k)
            st, _ = self.apply(lvl, st, *mv)
        lvl["sol_left"] = left
        return lvl, st0, sol

    def min_len(self, i, n):
        return min(16 + 5 * i, 50)

    def body(self, lvl, st):
        return (st[0][0], st[0][1], lvl["Hc"], lvl["Wc"])

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        _, tags = self._step(lvl, st, op, px)
        if "bump" in tags:
            ev[EV["touch"]] = 1
        if "rewind" in tags:
            ev[EV["teleport"]] = ev[EV["toggle"]] = 1
        if "plate" in tags:
            ev[EV["push"]] = 1
        if "gate" in tags or "ghost" in tags:
            ev[EV["danger"]] = 1
        o0 = self._open(lvl, self._held(lvl, {st[0]} | {self._gpos(g, st[1]) for g in st[3]}), st[2], {st[0]})
        o1 = self._open(lvl, self._held(lvl, {st2[0]} | {self._gpos(g, st2[1]) for g in st2[3]}), st2[2], {st2[0]})
        if o0 != o1:
            ev[EV["open"]] = ev[EV["color"]] = 1

    # ---------------------------------------------------------------- rendu
    def render(self, lvl, st):
        pal = self.pal
        C = lvl["C"]
        p, t, timers, ghosts, rec = st
        g = self.blank()
        floor = lvl["floor"]
        K.draw_walls(g, lvl, pal.wall, 2 if C >= 5 else 1)
        gcells = {self._gpos(gh, t) for gh in ghosts}
        held = self._held(lvl, {p} | gcells)
        opened = self._open(lvl, held, timers, {p})
        cols = [pal.a(1), pal.a(2)]
        for k, (gc, need) in enumerate(lvl["gates"]):           # fils plaque -> grille
            col = cols[k % 2]
            for j in need:
                (y0, x0), (y1, x1) = K.cell_center(lvl, lvl["plates"][j]), K.cell_center(lvl, gc)
                draw.line(g, y0, x0, y1, x1, col)
        for k, (gc, need) in enumerate(lvl["gates"]):
            col = cols[k % 2]
            y, x = K.cell_xy(lvl, gc)
            if k in opened:
                frame_rect(g, y, x, C, C, col)
            else:
                rect(g, y, x, C, C, col)
            if lvl["timed"][k]:
                g[y, x] = g[y, x + C - 1] = R(pal.fg)
        for j, pl in enumerate(lvl["plates"]):
            k = next(k for k, (gc, need) in enumerate(lvl["gates"]) if j in need)
            y, x = K.cell_xy(lvl, pl)
            blit(g, sprite(None, "framed" if held[j] else "frame", C, C, cols[k % 2], cols[k % 2]), y, x)
            if C >= 5:
                frame_rect(g, y + 2, x + 2, C - 4, C - 4, pal.fg if held[j] else cols[k % 2])
        y, x = K.cell_xy(lvl, lvl["exit"])
        blit(g, sprite(None, "framed", C, C, pal.a(4), pal.fg), y, x)
        y, x = K.cell_xy(lvl, lvl["start"])
        frame_rect(g, y, x, C, C, pal.line)
        gc_ = draw.GREY if pal.bg != draw.GREY else draw.LGREY
        for gh in ghosts:                                       # fantômes : silhouettes grises
            q = self._gpos(gh, t)
            y, x = K.cell_xy(lvl, q)
            m = draw.recolor(self.tpl[C], {100: gc_, 101: gc_, 102: pal.bg})
            blit(g, m, y, x)
        y, x = K.cell_xy(lvl, p)
        blit(g, draw.recolor(self.tpl[C], {100: pal.a(0), 101: pal.fg, 102: pal.fg}), y, x)
        self._hud(g, lvl, st)
        return g

    def _hud(self, g, lvl, st):
        pal = self.pal
        h = self.hh - 1
        rect(g, 0, 0, h, self.AW, pal.ui)
        a = h - 2
        x = 1
        for k in range(2):                                      # emplacements de fantômes
            if k < len(st[3]):
                rect(g, 1, x, a, a, draw.GREY if pal.ui != draw.GREY else draw.LGREY)
            else:
                frame_rect(g, 1, x, a, a, pal.line)
            x += a + 1
        x += 2
        n = len(st[4]) - 1                                      # ruban d'enregistrement (barre pleine)
        step = 2 if self.s_screen == 1 else 1
        rect(g, 1 + a // 2 - (step > 1), x, step, step * n, pal.a(0))
        rect(g, 1 + a // 2 - (step > 1), x + step * n, step, step * (LMAX - n), pal.wall if pal.wall != pal.ui else pal.line)
