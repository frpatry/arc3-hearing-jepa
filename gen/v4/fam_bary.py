"""bary4 (clic, corps couplés) : un CORPS se tient toujours au BARYCENTRE de ses poignées (reliées par des fils
élastiques en pointillés). Clic sur une poignée = la choisir (elle blanchit) ; clic ailleurs = la poignée choisie y va
(au plus D cases par coup) ; le corps saute aussitôt au nouveau barycentre. But : amener le corps dans l'anneau-cible.
Variantes : `slide` (la poignée file en ligne droite vers le point cliqué, s'arrête devant un mur, D cases au plus)
ou `hop` (la poignée saute exactement sur la case cliquée si elle est libre et à portée, sinon rien).
Règles ajoutées (ordre tiré par jeu, une par niveau) :
  hazard : lacs : si le corps y atterrit, le coup est annulé et compte une touche (pastilles ; 3 touches = perdu) ;
  third  : une 3e poignée (le corps bouge moins à chaque coup) ;
  absorb : disques de couleur que le corps doit visiter (absorber) avant que la cible compte ; l'anneau montre les
           couleurs exigées ;
  anchor : une poignée est clouée (carré) : on ne peut plus la bouger ;
  two    : un 2e corps avec ses poignées et sa cible.
Solvabilité : A* pondéré sur (poignées, choix, couleurs absorbées), corps par corps ; l'oracle suit une route
cohérente (replanifiée hors chemin)."""
import heapq

import numpy as np

from ..core import EV
from . import draw
from .base import V4Game, mods_for
from .draw import R, blit, rect, sprite

D8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _put(g, y, x, c):
    if 0 <= y < g.shape[0] and 0 <= x < g.shape[1]:
        g[y, x] = R(c)


def _cheb(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _line(a, b):
    """cases de a (exclu) à b (inclus), Bresenham."""
    (y0, x0), (y1, x1) = a, b
    dy, dx = abs(y1 - y0), abs(x1 - x0)
    sy, sx = (1 if y1 > y0 else -1), (1 if x1 > x0 else -1)
    err = dx - dy
    y, x, out = y0, x0, []
    while (y, x) != (y1, x1):
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
        out.append((y, x))
    return out


class Bary4(V4Game):
    family = "bary4"
    mechanic = "click"
    principle = ("corps au barycentre de poignées déplacées une à une (fils élastiques) ; lacs, absorbeurs, poignée "
                 "clouée, 2e corps", ["r11l", "m0r0", "s5i5"])
    simple_ops = ()
    CAT = ("hazard", "third", "absorb", "anchor", "two")
    exact_progress = False
    n_levels_official = (6, 8)
    MAXHIT = 3

    def setup(self, rng):
        r = self.orng
        self.uses_click = True
        self.free_miss = bool(rng.random() < 0.4)
        self.variant = str(rng.choice(["slide", "hop"]))
        self.cat = [str(x) for x in rng.permutation(self.CAT)]
        self.C = int(r.choice([3, 3, 4])) if self.s_screen == 1 else 2
        self.D = 2
        self.hcount_top = bool(r.random() < 0.5)

    # ------------------------------------------------------------------ géométrie
    def _cell(self, lvl, px):
        if px is None:
            return None
        y, x = px[0] - lvl["oy"], px[1] - lvl["ox"]
        if y < 0 or x < 0:
            return None
        r, c = y // self.C, x // self.C
        if r < lvl["N"] and c < lvl["M"]:
            return (int(r), int(c))
        return None

    def _px(self, lvl, q):
        return (lvl["oy"] + q[0] * self.C + self.C // 2, lvl["ox"] + q[1] * self.C + self.C // 2)

    @staticmethod
    def _centroid(hs):
        k = len(hs)
        return ((sum(h[0] for h in hs) + k // 2) // k, (sum(h[1] for h in hs) + k // 2) // k)

    def _free(self, lvl, q):
        return 0 <= q[0] < lvl["N"] and 0 <= q[1] < lvl["M"] and q not in lvl["walls"]

    # ------------------------------------------------------------------ règles
    def _landing(self, lvl, hs_all, h, click):
        """case d'arrivée de la poignée h (case) vers la case cliquée, ou None (sans effet)."""
        occ = set(hs_all)
        if self.variant == "hop":
            if _cheb(h, click) > self.D or not self._free(lvl, click) or click in occ or click == h:
                return None
            return click
        land = h
        for q in _line(h, click):
            if _cheb(h, q) > self.D or not self._free(lvl, q) or q in occ:
                break
            land = q
        return None if land == h else land

    def _move(self, lvl, st, b, j, click):
        """-> (nouvel état, tags) pour le déplacement de la poignée j du corps b vers click."""
        hands, sel, hits, got = st
        allh = [h for hb in hands for h in hb]
        h = hands[b][j]
        land = self._landing(lvl, allh, h, click)
        if land is None:
            return st, {"refuse"}
        hb = hands[b][:j] + (land,) + hands[b][j + 1:]
        body = self._centroid(hb)
        if body in lvl["walls"]:
            return st, {"refuse"}
        if body in lvl["hazard"]:
            return (hands, sel, hits + 1, got), {"hazard"}
        g = got[b]
        tags = {"move"}
        if body in lvl["absorb"].get(b, {}):
            bit = lvl["absorb"][b][body]
            if not (g >> bit) & 1:
                g |= 1 << bit
                tags.add("absorb")
        hands = hands[:b] + (hb,) + hands[b + 1:]
        got = got[:b] + (g,) + got[b + 1:]
        return (hands, sel, hits, got), tags

    def _step(self, lvl, st, op, px):
        if op != "click":
            return st, set()
        q = self._cell(lvl, px)
        if q is None:
            return st, set()
        hands, sel, hits, got = st
        for b, hb in enumerate(hands):
            for j, h in enumerate(hb):
                if h == q:
                    if (b, j) in lvl["anchor"] or sel == (b, j):
                        return st, {"inert"}
                    return (hands, (b, j), hits, got), {"select"}
        if sel is None:
            return st, set()
        return self._move(lvl, st, sel[0], sel[1], q)

    def apply(self, lvl, st, op, px):
        st2, tags = self._step(lvl, st, op, px)
        return st2, bool(lvl["hazard"]) and st2[2] >= self.MAXHIT

    def _body_ok(self, lvl, st, b):
        body = self._centroid(st[0][b])
        return body == lvl["targets"][b] and (st[3][b] & lvl["need"][b]) == lvl["need"][b]

    def is_win(self, lvl, st):
        return all(self._body_ok(lvl, st, b) for b in range(len(st[0])))

    def click_targets(self, lvl, st):
        return []

    def click_hit(self, lvl, st, px):
        q = self._cell(lvl, px)
        return q is not None and any(h == q for hb in st[0] for h in hb)

    # ------------------------------------------------------------------ planification (A* corps par corps)
    def _astar(self, lvl, st, b, cap=None):
        cap = cap or (8000 if getattr(self, "relax", 0) else 3000)
        hands0, sel0, hits0, got0 = st
        T = lvl["targets"][b]
        need = lvl["need"][b]
        k = len(hands0[b])
        absorb = lvl["absorb"].get(b, {})
        movable = [j for j in range(k) if (b, j) not in lvl["anchor"]]
        others = {h for bb, hb in enumerate(hands0) if bb != b for h in hb}
        walls, hazard, N, M, D = lvl["walls"], lvl["hazard"], lvl["N"], lvl["M"], self.D
        hop = self.variant == "hop"

        def hfun(body, g):
            rest = [p for p, bit in absorb.items() if (need >> bit) & 1 and not (g >> bit) & 1]
            d = _cheb(body, T)
            if rest:
                d = min(_cheb(body, p) + _cheb(p, T) for p in rest) + 2 * (len(rest) - 1)
            return d * k / D

        def lands(hb, j):
            """cases d'arrivée distinctes de la poignée j -> [(case cliquée, arrivée)]."""
            h = hb[j]
            occ = others | set(hb)
            out = []
            for d in D8:
                last = h
                for dist in range(1, D + 1):
                    q = (h[0] + d[0] * dist, h[1] + d[1] * dist)
                    if not (0 <= q[0] < N and 0 <= q[1] < M):
                        break
                    if q in walls or q in occ:
                        if hop:
                            continue
                        break
                    out.append((q, q))
            return out

        if self._body_ok(lvl, st, b):
            return []
        s0 = (hands0[b], sel0[1] if (sel0 and sel0[0] == b) else -1, got0[b])
        cnt = 0
        pq = [(hfun(self._centroid(s0[0]), s0[2]), 0, cnt, s0)]
        prev = {s0: None}
        gbest = {s0: 0}
        while pq:
            f, gc, _, s = heapq.heappop(pq)
            if gbest.get(s, 1e9) < gc:
                continue
            hb, sl, g = s
            body = self._centroid(hb)
            if body == T and (g & need) == need:
                path = []
                while prev[s] is not None:
                    s, acts = prev[s]
                    path = acts + path
                return path
            if len(prev) > cap:
                return None
            sy, sx = sum(h[0] for h in hb), sum(h[1] for h in hb)
            for j in movable:
                h = hb[j]
                for click, land in lands(hb, j):
                    nb = ((sy - h[0] + land[0] + k // 2) // k, (sx - h[1] + land[1] + k // 2) // k)
                    if nb in walls or nb in hazard:
                        continue
                    ng_ = g
                    if nb in absorb:
                        ng_ |= 1 << absorb[nb]
                    nhb = hb[:j] + (land,) + hb[j + 1:]
                    n = (nhb, j, ng_)
                    cost = 1 if sl == j else 2
                    ng = gc + cost
                    if ng < gbest.get(n, 1e9):
                        gbest[n] = ng
                        prev[n] = (s, ([] if sl == j else [("sel", (b, j))]) + [("go", click)])
                        cnt += 1
                        heapq.heappush(pq, (ng + 3.0 * hfun(nb, ng_), ng, cnt, n))
        return None

    def _plan0(self, lvl, st):
        out, cur = [], st
        for b in range(len(st[0])):
            if self._body_ok(lvl, cur, b):
                continue
            p = self._astar(lvl, cur, b)
            if p is None:
                return None
            for kind, arg in p:
                if kind == "sel":
                    a = ("click", self._px(lvl, cur[0][arg[0]][arg[1]]))
                else:
                    a = ("click", self._px(lvl, arg))
                out.append(a)
                cur, dead = self.apply(lvl, cur, *a)
                if dead:
                    return None
        return out if self.is_win(lvl, cur) else None

    def _plan(self, lvl, st):
        nx = lvl.setdefault("_next", {})
        if len(nx) > 20000:
            nx.clear()
        out, cur, seen = [], st, set()
        while not self.is_win(lvl, cur):
            if cur in seen:
                return None
            seen.add(cur)
            if cur not in nx:
                p = self._plan0(lvl, cur)
                if p is None:
                    nx[cur] = None
                    return None
                c = cur
                for a in p:
                    nx.setdefault(c, a)
                    c, _ = self.apply(lvl, c, *a)
            a = nx[cur]
            if a is None:
                return None
            out.append(a)
            cur, _ = self.apply(lvl, cur, *a)
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

    # ------------------------------------------------------------------ niveaux
    def _blobs(self, rng, N, M, n, rmin, rmax, avoid):
        out = set()
        for _ in range(n):
            cy, cx = int(rng.integers(1, N - 1)), int(rng.integers(1, M - 1))
            ry, rx = float(rng.uniform(rmin, rmax)), float(rng.uniform(rmin, rmax))
            for r in range(max(0, int(cy - ry) - 1), min(N, int(cy + ry) + 2)):
                for c in range(max(0, int(cx - rx) - 1), min(M, int(cx + rx) + 2)):
                    if ((r - cy) / ry) ** 2 + ((c - cx) / rx) ** 2 <= 1.0 + 0.3 * rng.random():
                        out.add((r, c))
        return out - avoid

    def gen_level(self, rng, i, n):
        mods, newest = mods_for(self.cat, i)
        C = self.C
        top = 4 if self.s_screen == 1 else 3
        N, M = (self.AH - top) // C, self.AW // C
        oy, ox = top + (self.AH - top - N * C) // 2, (self.AW - M * C) // 2
        nb = 2 if "two" in mods else 1
        k = 3 if ("third" in mods or "anchor" in mods) else 2
        best = None
        for _ in range(5):
            walls = self._blobs(rng, N, M, int(rng.integers(2, 4)) + (i >= 3), 1.0, 2.6, set())
            hazard = self._blobs(rng, N, M, int(rng.integers(2, 4)), 1.2, 3.0, walls) if "hazard" in mods else set()
            free = [(r, c) for r in range(N) for c in range(M) if (r, c) not in walls and (r, c) not in hazard]
            hands, targets, absorb, need, anchor = [], [], {}, [], set()
            used = set()
            ok = True
            for b in range(nb):
                # départ d'un côté, cible de l'autre
                side = int(rng.integers(4))
                def zone(q, s):
                    r, c = q
                    return [r < N // 3, r >= 2 * N // 3, c < M // 3, c >= 2 * M // 3][s]
                startc = [q for q in free if zone(q, side) and q not in used]
                tgtc = [q for q in free if zone(q, side ^ 1) and q not in used]
                if not startc or not tgtc:
                    ok = False
                    self.why("p-zone")
                    break
                S = startc[int(rng.integers(len(startc)))]
                hb = []
                for j in range(k):
                    cand = [q for q in free if 2 <= _cheb(q, S) <= 3 and q not in used and q not in hb]
                    if not cand:
                        break
                    hb.append(cand[int(rng.integers(len(cand)))])
                if len(hb) < k:
                    ok = False
                    self.why("p-hands")
                    break
                body = self._centroid(hb)
                if body in walls or body in hazard:
                    ok = False
                    self.why("p-body")
                    break
                anc = int(rng.integers(k)) if "anchor" in mods else -1
                if anc >= 0:                                    # cible atteignable malgré la poignée clouée
                    a = hb[anc]
                    tgtc = [q for q in free if q not in used and 1 <= (k * q[0] - a[0]) / (k - 1) <= N - 2 and
                            1 <= (k * q[1] - a[1]) / (k - 1) <= M - 2 and _cheb(q, body) >= N // 4]
                    if not tgtc:
                        ok = False
                        self.why("p-anchor")
                        break
                dl = [_cheb(q, body) for q in tgtc]
                tgtc = [q for q, d in zip(tgtc, dl) if d >= (0.3 if getattr(self, "relax", 0) else 0.7) * max(dl)]
                T = tgtc[int(rng.integers(len(tgtc)))]
                used |= set(hb) | {T}
                hands.append(tuple(hb))
                targets.append(T)
                nd = 0
                if "absorb" in mods:
                    ab = {}
                    mid = [q for q in free if q not in used and _cheb(q, body) >= 4 and _cheb(q, T) >= 4 and
                           (anc < 0 or (1 <= (k * q[0] - hb[anc][0]) / (k - 1) <= N - 2 and
                                        1 <= (k * q[1] - hb[anc][1]) / (k - 1) <= M - 2))]
                    for bit in range(2 if i >= 4 else 1 + int(rng.integers(0, 2))):
                        if not mid:
                            break
                        p = mid.pop(int(rng.integers(len(mid))))
                        ab[p] = bit
                        used.add(p)
                        nd |= 1 << bit
                    absorb[b] = ab
                need.append(nd)
                if anc >= 0:
                    anchor.add((b, anc))
            if not ok:
                continue
            lvl = dict(N=N, M=M, oy=oy, ox=ox, mods=mods, walls=frozenset(walls), hazard=frozenset(hazard),
                       targets=tuple(targets), absorb=absorb, need=tuple(need), anchor=frozenset(anchor))
            st0 = (tuple(hands), None, 0, tuple(0 for _ in range(nb)))
            if self.is_win(lvl, st0):
                continue
            sol = self._plan0(lvl, st0)
            if sol is None:
                self.why("astar")
                continue
            goal = self.min_len(i, n) * 1.2
            if best is None or abs(len(sol) - goal) < abs(len(best[2]) - goal):
                best = (lvl, st0, sol)
            if self.min_len(i, n) <= len(sol) <= 1.6 * self.min_len(i, n):
                break
        if best is None:
            return self.why("place")
        lvl, st0, sol = best
        nx, c = lvl.setdefault("_next", {}), st0
        for a in sol:
            nx.setdefault(c, a)
            c, _ = self.apply(lvl, c, *a)
        return lvl, st0, sol

    def min_len(self, i, n):
        return min(16 + 6 * i, 55)

    def body(self, lvl, st):
        b = st[1][0] if st[1] else 0
        p = self._centroid(st[0][b])
        return (p[0], p[1], lvl["N"], lvl["M"])

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        _, tags = self._step(lvl, st, op, px)
        if "select" in tags:
            ev[EV["toggle"]] = 1
        if "hazard" in tags:
            ev[EV["danger"]] = 1
        if "absorb" in tags:
            ev[EV["collect"]] = ev[EV["color"]] = 1
        if "refuse" in tags:
            ev[EV["bump"]] = 1
        if "move" in tags:
            ev[EV["push"]] = 1
        if sum(self._body_ok(lvl, st2, b) for b in range(len(st2[0]))) > \
                sum(self._body_ok(lvl, st, b) for b in range(len(st[0]))):
            ev[EV["open"]] = 1

    # ------------------------------------------------------------------ rendu
    def render(self, lvl, st):
        pal, C = self.pal, self.C
        g = self.blank()
        oy, ox = lvl["oy"], lvl["ox"]
        for (r, c) in lvl["walls"]:
            rect(g, oy + r * C, ox + c * C, C, C, pal.wall)
        haz_c = draw.LBLUE if draw.LBLUE not in (pal.bg, pal.wall) else pal.a(5)
        for (r, c) in lvl["hazard"]:
            rect(g, oy + r * C, ox + c * C, C, C, haz_c)
        hands, sel, hits, got = st
        bcol = [pal.a(0), pal.a(1)]
        acol = [pal.a(2), pal.a(3)]
        sel_c = draw.WHITE if pal.bg != draw.WHITE else draw.BLACK
        hand_c = next(c for c in (pal.line, draw.GREY, draw.LGREY, draw.DGREY, pal.a(4))
                      if c not in (pal.wall, sel_c, pal.bg))
        rs = 5 if C >= 3 else 4
        for b, T in enumerate(lvl["targets"]):
            y, x = self._px(lvl, T)
            spr = sprite(None, "ring", rs + 2, rs + 2, bcol[b])
            blit(g, spr, y - (rs + 2) // 2, x - (rs + 2) // 2)
            for bit in range(2):
                if (lvl["need"][b] >> bit) & 1:
                    _put(g, y - (rs + 2) // 2 + (0 if bit == 0 else rs + 1), x - (rs + 2) // 2, acol[bit])
                    _put(g, y - (rs + 2) // 2 + (0 if bit == 0 else rs + 1), x - (rs + 2) // 2 + rs + 1, acol[bit])
        for b, ab in lvl["absorb"].items():
            for p, bit in ab.items():
                if not (got[b] >> bit) & 1:
                    y, x = self._px(lvl, p)
                    blit(g, sprite(None, "disc", 3, 3, acol[bit]), y - 1, x - 1)
        for b, hb in enumerate(hands):
            body = self._centroid(hb)
            by, bx = self._px(lvl, body)
            for h in hb:
                hy, hx = self._px(lvl, h)
                draw.line(g, by, bx, hy, hx, pal.line)
            for j, h in enumerate(hb):
                hy, hx = self._px(lvl, h)
                col = sel_c if sel == (b, j) else hand_c
                if (b, j) in lvl["anchor"]:
                    blit(g, sprite(None, "framed", 3, 3, pal.bg, col), hy - 1, hx - 1)
                else:
                    blit(g, sprite(None, "plus", 3, 3, col), hy - 1, hx - 1)
            spr = sprite(None, "disc", rs, rs, bcol[b])
            blit(g, spr, by - rs // 2, bx - rs // 2)
            for bit in range(2):
                if (got[b] >> bit) & 1:
                    _put(g, by, bx - 1 + 2 * bit, acol[bit])
        if lvl["hazard"]:
            y = 1 if self.hcount_top else self.AH - 2
            for j in range(self.MAXHIT):
                rect(g, y, 1 + 3 * j, 1, 2, haz_c if j >= hits else pal.fg)
        return g
