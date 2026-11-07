"""keyattr4 (avatar, serrures à attributs) : l'avatar porte une CLÉ (forme, couleur, orientation) montrée dans le
bandeau ; marcher sur une STATION fait tourner un attribut d'un cran (station d'orientation, de couleur, de forme) ;
chaque porte montre la clé qu'elle exige : avec la bonne clé on la franchit (elle reste ouverte), sinon on bute.
But : franchir la (les) porte(s) et atteindre la sortie. Salles ouvertes / cavernes, peu d'objets.
Variantes (tirées par jeu) :
  walk : chaque entrée sur une station fait tourner son attribut ;
  arm  : la station ne fait que s'ALLUMER quand on y entre ; ACTION5 sur une station allumée applique son effet
         (une action de plus, et ACTION5 ailleurs ne fait rien).
Règles ajoutées, une par niveau (ordre tiré par jeu ; niveau 0 = orientation seule) :
  colour : station de couleur (la clé a 3 couleurs) ;
  shape  : station de forme (3 formes) ;
  door2  : deux portes en série, aux exigences différentes (reconfigurer la clé entre les deux) ;
  refill : jauge d'énergie (bandeau) : chaque pas coûte 1, à zéro on perd ; des recharges la remplissent ;
  mover  : la station d'orientation fait des allers-retours (un pas tous les 2 tours).
Solvabilité : BFS exact sur (case, forme, couleur, orientation, portes ouvertes, énergie, phase)."""
from math import gcd

import numpy as np

from ..core import DIRS, DIR_OPS, EV
from . import avkit as K
from . import draw
from .base import V4Game, mods_for
from .draw import R, blit, frame_rect, rect, sprite


def _shapes(r, n=3, k=3):
    """n formes k×k, connexes, sans symétrie de rotation, distinctes à rotation près."""
    out, seen = [], set()
    while len(out) < n:
        m = r.random((k, k)) < 0.55
        m[k // 2, k // 2] = True
        if m.sum() < k + 1 or m.sum() > k * k - 2:
            continue
        lab = draw.largest_component(m)
        if lab.sum() != m.sum():
            continue
        rots = [np.rot90(m, j) for j in range(4)]
        keys = [x.tobytes() for x in rots]
        if len(set(keys)) < 4 or any(x in seen for x in keys):
            continue
        seen |= set(keys)
        out.append(m)
    return out


class KeyAttr4(V4Game):
    family = "keyattr4"
    mechanic = "avatar"
    principle = ("serrures à attributs : stations qui changent forme / couleur / orientation de la clé portée ; "
                 "portes qui montrent la clé exigée", ["ls20", "ka59", "cn04"])
    simple_ops = DIR_OPS
    CAT = ("colour", "shape", "door2", "refill", "mover")
    exact_progress = True
    table_cap = 25000
    bfs_cap = 40000

    def setup(self, rng):
        r = self.orng
        self.variant = str(rng.choice(["walk", "walk", "arm"]))
        if self.variant == "arm":
            self.simple_ops = DIR_OPS + ("interact",)
        cat = [str(x) for x in rng.permutation(self.CAT)]
        self.cat = cat
        self.C = int(r.choice([5, 6, 6, 7])) if self.s_screen == 1 else 3
        self.hh = 9 if self.s_screen == 1 else 6
        self.k = 3
        self.shapes = _shapes(r, 3, 3)
        self.floor_style = str(r.choice(["cave", "cave", "rooms"]))
        self.player_kind = str(r.choice(["framed", "blob", "box"]))
        self.tpl = {S: sprite(r, self.player_kind, S, S, 100, 101, 102 if S >= 5 else None)
                    for S in range(3, self.C + 5)}
        self.hud_bottom = bool(r.random() < 0.7)

    # ---------------------------------------------------------------- règles
    def _key_ok(self, lvl, st, k):
        return (st[1], st[2], st[3]) == lvl["req"][k]

    def _mover_pos(self, lvl, t):
        path = lvl["mover"]
        return path[(t // 2) % len(path)] if path else None         # un pas tous les 2 tours (parité)

    def _apply_station(self, lvl, kind, shp, col, rot):
        if kind == "rot":
            rot = (rot + 1) % 4
        elif kind == "col":
            col = (col + 1) % lvl["ncol"]
        else:
            shp = (shp + 1) % lvl["nshp"]
        return shp, col, rot

    def _step(self, lvl, st, op, px):
        p, shp, col, rot, dm, e, t, arm = st
        tags = set()
        if op == "interact":                                    # variante arm : appliquer la station allumée
            if arm < 0:
                return st, tags
            shp, col, rot = self._apply_station(lvl, lvl["stations"][arm][1] if arm < len(lvl["stations"]) else "rot",
                                                shp, col, rot)
            tags.add("station")
            tags.add("st_" + (lvl["stations"][arm][1] if arm < len(lvl["stations"]) else "mover"))
            return (p, shp, col, rot, dm, e, t, -1), tags
        if op not in DIRS:
            return st, tags
        d = DIRS[op]
        q = (p[0] + d[0], p[1] + d[1])
        if not (0 <= q[0] < lvl["Hc"] and 0 <= q[1] < lvl["Wc"]) or not lvl["floor"][q]:
            tags.add("bump")
            return st, tags
        for k, dc in enumerate(lvl["doors"]):
            if dc == q and not dm >> k & 1:
                if (shp, col, rot) == lvl["req"][k]:
                    dm |= 1 << k
                    tags.add("open")
                else:
                    tags.add("wrong")
                    return st, tags
        if lvl["E"]:
            e -= 1
            if q in lvl["pickups"]:
                e = lvl["E"]
                tags.add("refill")
            elif e <= 0:
                tags.add("empty")
                return (q, shp, col, rot, dm, 0, t, arm), tags
        t2 = (t + 1) % lvl["P"]
        hit = None
        for j, (sp, kind) in enumerate(lvl["stations"]):
            if sp == q:
                hit = (j, kind)
        mp0, mp1 = self._mover_pos(lvl, t), self._mover_pos(lvl, t2)
        if mp1 is not None and mp1 == q and mp0 != p:
            hit = (len(lvl["stations"]), "rot")
            tags.add("st_mover")
        arm2 = -1
        if hit is not None:
            if self.variant == "arm":
                arm2 = hit[0]
                tags.add("arm")
            else:
                shp, col, rot = self._apply_station(lvl, hit[1], shp, col, rot)
                tags.add("station")
                tags.add("st_" + hit[1])
        return (q, shp, col, rot, dm, e, t2, arm2), tags

    def apply(self, lvl, st, op, px):
        st2, tags = self._step(lvl, st, op, px)
        return st2, "empty" in tags

    def is_win(self, lvl, st):
        return st[0] == lvl["exit"]

    # ---------------------------------------------------------------- niveaux
    def gen_level(self, rng, i, n):
        mods, newest = mods_for(self.cat, i, keep=2)
        fac = dict(colour=0.8, shape=0.8, door2=1.0, refill=0.8, mover=0.8)
        maxc = 70.0 + 8 * min(i, 5)                             # plateaux plus grands aux niveaux tardifs
        for m in mods:
            maxc *= fac.get(m, 1.0)
        maxc = max(28, int(maxc))
        top = 0 if self.hud_bottom else self.hh
        bot = self.hh if self.hud_bottom else 0
        C, Hc, Wc, oy, ox = K.board(self.AH, self.AW, self.C, maxc / 0.5 + 8, top=top, bottom=bot)
        floor = K.make_floor(rng, Hc, Wc, self.floor_style, n_rooms=3 + (i >= 3),
                             need=min(int(maxc * 0.75), int(0.45 * Hc * Wc)),
                             holes=0.0, max_side=max(4, min(Hc, Wc) - 2), maxc=maxc + 10)
        if floor is None:
            return self.why("floor")
        nd = min(3, 1 + ("door2" in mods) + (i >= 4) + (i >= 6 and "refill" not in mods))
        ww = K.wall_with_gap(rng, floor, gap=(1, 1), min_part=6 if nd == 1 else 9, tries=24)
        if ww is None:
            return self.why("wall")
        floor, run1, A, B = ww
        if len(A) < len(B):
            A, B = B, A                                         # départ dans la plus grande partie
        doors = [run1[0]]
        regions = [A, B]
        for k in range(1, nd):                                  # portes en série : recouper la dernière région
            Bk = regions[-1]
            sub = np.zeros_like(floor)
            for p in Bk:
                sub[p] = True
            ww2 = K.wall_with_gap(rng, sub, gap=(1, 1), min_part=3, tries=24)
            if ww2 is None:
                return self.why("wall2")
            sub2, run2, A2, B2 = ww2
            for p in Bk:
                if not sub2[p]:
                    floor[p] = False
            near = [q for q in K.nbrs(floor, doors[-1]) if q in A2 or q in B2]
            if not near:
                return self.why("door_adj")
            if near[0] in B2:
                A2, B2 = B2, A2
            doors.append(run2[0])
            regions = regions[:-1] + [A2, B2]
        last = regions[-1]
        ncol = 3 if "colour" in mods else 1
        nshp = 3 if "shape" in mods else 1
        lvl = dict(floor=floor, Hc=Hc, Wc=Wc, oy=oy, ox=ox, C=C, mods=mods, doors=tuple(doors), ncol=ncol, nshp=nshp,
                   exit=None, pickups=frozenset(), E=0, mover=(), P=1)
        A_l = [p for p in A]
        rng.shuffle(A_l)
        dA = {}
        # stations : orientation toujours (sauf si elle bouge), puis couleur / forme
        kinds = (["rot"] if "mover" not in mods else []) + (["col"] if ncol > 1 else []) + (["shp"] if nshp > 1 else [])
        avoid = set(K.nbrs(floor, doors[0]))
        free = [p for p in A_l if p not in avoid]
        if len(free) < len(kinds) + 6:
            return self.why("room")
        stations = []
        dd0 = K.grid_dist(floor, doors[0])
        free.sort(key=lambda p: -dd0.get(p, 0) - 3 * rng.random())   # stations plutôt loin de la porte
        for kd in kinds:
            if nd >= 2 and kd != "rot" and rng.random() < 0.5 and len(regions[1]) > 4:
                mid = [p for p in regions[1] if p not in set(K.nbrs(floor, doors[1])) and p not in avoid
                       and p not in {s[0] for s in stations}]
                if mid:
                    stations.append((mid[int(rng.integers(len(mid)))], kd))
                    continue
            stations.append((free.pop(int(rng.integers(min(4, len(free))))), kd))
        if "mover" in mods:
            segs = [s for s in self._segs(floor, 3, 5) if all(q in A for q in s) and not {x[0] for x in stations} & set(s)]
            if not segs:
                return self.why("mover")
            s = segs[int(rng.integers(len(segs)))]
            lvl["mover"] = tuple(s + s[-2:0:-1])
            lvl["P"] = 2 * len(lvl["mover"])
            free = [p for p in free if p not in s]
        lvl["stations"] = tuple(stations)
        busy = {s[0] for s in stations} | set(lvl["mover"])
        free = [p for p in free if p not in busy]
        if not free:
            return self.why("start")
        dd = K.grid_dist(floor, doors[0])
        p0 = max(free[:12], key=lambda p: dd.get(p, 0))
        dl = K.grid_dist(floor, doors[-1])
        ex = [p for p in last if p != doors[-1]]
        if not ex:
            return self.why("exit")
        lvl["exit"] = max(ex, key=lambda p: dl.get(p, 0) + rng.random())
        # clé de départ et exigences : la porte 1 change les attributs dont la station est avant elle (orientation
        # toujours), la porte 2 au moins un attribut de plus
        k0 = (int(rng.integers(nshp)), int(rng.integers(ncol)), int(rng.integers(4)))
        reg_of = {}
        for r_, R_ in enumerate(regions):
            for p in R_:
                reg_of[p] = r_
        n_of = dict(shp=nshp, col=ncol, rot=4)
        idx = dict(shp=0, col=1, rot=2)
        req, prev_k = [], k0
        for k in range(nd):
            pool = {"rot"} | {kd for sp, kd in stations if reg_of.get(sp, 0) <= k}
            newk = {kd for sp, kd in stations if reg_of.get(sp, 0) == k} if k else set()
            ch = [a for a in pool if n_of[a] > 1 and (rng.random() < 0.8 or a == "rot" or a in newk)]
            rq = list(prev_k)
            for a in ch:
                j = idx[a]
                step = int(rng.choice([2, 3, 3])) if a == "rot" else int(rng.integers(1, n_of[a]))
                rq[j] = int((prev_k[j] + step) % n_of[a])
            rq = tuple(rq)
            req.append(rq)
            prev_k = rq
        lvl["req"] = tuple(req)
        st0 = (p0, k0[0], k0[1], k0[2], 0, 0, 0, -1)
        if "refill" in mods:                                    # jauge : courte, recharges sur le chemin
            depth, prev = K.explore(self, lvl, st0, 30000)
            if depth is None:
                return self.why("cap")
            wins = [s for s in depth if self.is_win(lvl, s)]
            if not wins:
                return self.why("nosol0")
            sw = min(wins, key=lambda s: depth[s])
            path = K.path_to(prev, sw)
            cells_on, s_ = [], st0
            for mv in path:
                s_, _ = self.apply(lvl, s_, *mv)
                cells_on.append(s_[0])
            L0 = len(path)
            E = int(np.clip(L0 * float(rng.uniform(0.35, 0.55)), 8, 16))
            pk, last = set(), 0                                 # une recharge tous les ~E pas du chemin
            while L0 - last > E - 1:
                cand = [k for k in range(last + E // 2, min(L0, last + E - 1))
                        if cells_on[k] not in busy and cells_on[k] not in lvl["doors"] and cells_on[k] != lvl["exit"]]
                if not cand:
                    return self.why("refill")
                k = cand[int(rng.integers(len(cand)))]
                pk.add(cells_on[k])
                last = k + 1
            lvl["pickups"] = frozenset(pk)
            lvl["E"] = E
            st0 = (p0, k0[0], k0[1], k0[2], 0, E, 0, -1)
        depth, prev = K.explore(self, lvl, st0, 30000)
        if depth is None:
            return self.why("cap")
        wins = [s for s in depth if self.is_win(lvl, s)]
        if not wins:
            return self.why("nosol")
        sw = min(wins, key=lambda s: depth[s])
        sol = K.path_to(prev, sw)
        need = dict(colour="st_col", shape="st_shp", mover="st_mover", refill="refill").get(newest)
        if need and need not in K.trace_tags(self, lvl, st0, sol):
            return self.why("unused")
        return lvl, st0, sol

    @staticmethod
    def _segs(floor, lo, hi):
        H, W = floor.shape
        out = []
        for r in range(H):
            for c in range(W):
                for d in ((0, 1), (1, 0)):
                    seg, y, x = [], r, c
                    while 0 <= y < H and 0 <= x < W and floor[y, x] and len(seg) < hi:
                        seg.append((y, x))
                        if len(seg) >= lo:
                            out.append(list(seg))
                        y, x = y + d[0], x + d[1]
        return out

    def min_len(self, i, n):
        return min(14 + 6 * i, 50)

    def heuristic(self, lvl, st):
        p = st[0]
        e = lvl["exit"]
        return abs(p[0] - e[0]) + abs(p[1] - e[1])

    def body(self, lvl, st):
        return (st[0][0], st[0][1], lvl["Hc"], lvl["Wc"])

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        _, tags = self._step(lvl, st, op, px)
        if "bump" in tags or "wrong" in tags:
            ev[EV["touch"]] = 1
        if "open" in tags:
            ev[EV["open"]] = 1
        if "station" in tags:
            ev[EV["color"]] = 1
        if "arm" in tags:
            ev[EV["toggle"]] = 1
        if "refill" in tags:
            ev[EV["collect"]] = 1
        if "empty" in tags:
            ev[EV["danger"]] = 1

    # ---------------------------------------------------------------- rendu
    def _key_sprite(self, shp, col, rot, scale, colour):
        m = np.rot90(self.shapes[shp], rot)
        if scale > 1:
            m = np.repeat(np.repeat(m, scale, 0), scale, 1)
        s = np.full(m.shape, -1, np.int64)
        s[m] = colour
        return s

    def render(self, lvl, st):
        pal = self.pal
        C = lvl["C"]
        p, shp, col, rot, dm, e, t, arm = st
        g = self.blank()
        floor = lvl["floor"]
        K.draw_walls(g, lvl, pal.wall, 2 if C >= 5 else 1)
        kc = [pal.a(0), pal.a(1), pal.a(2)]
        # sortie
        y, x = K.cell_xy(lvl, lvl["exit"])
        blit(g, sprite(None, "framed", C, C, pal.a(4), pal.fg), y, x)
        # portes : cadre + clé exigée
        for k, dc in enumerate(lvl["doors"]):
            y, x = K.cell_xy(lvl, dc)
            if dm >> k & 1:
                frame_rect(g, y, x, C, C, pal.line)
                continue
            rect(g, y, x, C, C, pal.wall)
            rq = lvl["req"][k]
            ks = self._key_sprite(rq[0], rq[1], rq[2], 1, kc[rq[1]] if lvl["ncol"] > 1 else pal.a(3))
            o = (C - 3) // 2
            rect(g, y + o, x + o, 3, 3, pal.bg)
            blit(g, ks, y + o, x + o)
        for pk in lvl["pickups"]:
            y, x = K.cell_xy(lvl, pk, 1)
            blit(g, sprite(None, "plus", C - 2, C - 2, pal.a(5)), y, x)
        stl = list(lvl["stations"])
        mp = self._mover_pos(lvl, t)
        if mp is not None:
            stl.append((mp, "rot"))
        for j, (sp, kind) in enumerate(stl):
            y, x = K.cell_xy(lvl, sp)
            lit = arm == j
            if kind == "rot":
                blit(g, sprite(None, "ring", C, C, draw.WHITE if lit else pal.line), y, x)
            elif kind == "col":
                h = C // 2
                rect(g, y, x, h, h, kc[0])
                rect(g, y, x + h, h, C - h, kc[1])
                rect(g, y + h, x, C - h, h, kc[2])
                rect(g, y + h, x + h, C - h, C - h, draw.WHITE if lit else pal.fg)
            else:
                blit(g, sprite(None, "frame", C, C, draw.WHITE if lit else pal.a(6)), y, x)
                g[y + C // 2, x + C // 2] = R(pal.a(6))
        if lvl["mover"]:
            (y0, x0), (y1, x1) = K.cell_center(lvl, lvl["mover"][0]), K.cell_center(lvl, lvl["mover"][len(lvl["mover"]) // 2])
            draw.line(g, y0, x0, y1, x1, pal.line)
            y, x = K.cell_xy(lvl, mp)
            blit(g, sprite(None, "ring", C, C, draw.WHITE if arm == len(lvl["stations"]) else pal.line), y, x)
        y, x = K.cell_xy(lvl, p)
        blit(g, draw.recolor(self.tpl[C], {100: pal.a(7), 101: pal.fg, 102: kc[col] if lvl["ncol"] > 1 else pal.fg}), y, x)
        self._hud(g, lvl, st, kc)
        return g

    def _hud(self, g, lvl, st, kc):
        pal = self.pal
        p, shp, col, rot, dm, e, t, arm = st
        h = self.hh - 1
        y0 = self.AH - h if self.hud_bottom else 0
        rect(g, y0, 0, h, self.AW, pal.ui)
        sc = 2 if self.s_screen == 1 else 1
        bw = 3 * sc + 2
        frame_rect(g, y0 + (h - bw) // 2, 1, bw, bw, pal.line)
        blit(g, self._key_sprite(shp, col, rot, sc, kc[col] if lvl["ncol"] > 1 else pal.a(3)), y0 + (h - bw) // 2 + 1, 2)
        x = bw + 3
        for k in range(len(lvl["doors"])):                       # portes restantes
            yy = y0 + (h - 3) // 2
            if dm >> k & 1:
                rect(g, yy, x, 3, 3, pal.line)
            else:
                frame_rect(g, yy, x, 3, 3, pal.line)
            x += 4
        if lvl["E"]:                                             # jauge d'énergie (2 px par point)
            x += 2
            yy = y0 + h // 2
            L = min(lvl["E"], (self.AW - x - 1) // 2)
            f = int(round(L * e / lvl["E"]))
            rect(g, yy - 1, x, 2, 2 * f, pal.a(5))
            rect(g, yy - 1, x + 2 * f, 2, 2 * (L - f), pal.wall if pal.wall != pal.ui else pal.line)
