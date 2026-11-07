"""kick4 (objets, contrôle indirect) : un « pilote » (cadre) circule dans des PIÈCES claires posées sur un pourtour ;
foncer dans un bloc le LANCE : le bloc glisse jusqu'au prochain obstacle, le pilote reste sur place (variante `kick`),
ou le bloc avance d'une case et le pilote suit (variante `push`). But : chaque cadre-cible occupé par un bloc
(de sa couleur si la cible est colorée). Un bandeau montre combien de cibles sont remplies.
Principes : contrôle indirect, cible affichée, une règle nouvelle par niveau.
Règles ajoutées (ordre tiré par jeu) :
  zone    : bandes de couleur que les blocs traversent mais pas le pilote ;
  stop    : plaques qui arrêtent un bloc qui glisse ;
  second  : 2e pilote ; clic (ou ACTION5) = choisir le pilote actif (le pilote actif est blanc) ;
  portal  : paire d'anneaux : un bloc qui entre ressort par l'autre et continue ;
  keyed   : cibles colorées, seul le bloc de la même couleur compte ;
  sticky  : plaques collantes : le PILOTE qui y entre y reste coincé un tour (ses actions suivantes sont sans effet
            une fois, case marquée).  (état : compteur de coincement)
Solvabilité : cibles = positions finales d'une partie aléatoire légale depuis le départ ; BFS exact (longueur optimale)."""
import numpy as np

from ..core import DIRS, DIR_OPS, EV
from . import draw
from .base import V4Game, mods_for
from .draw import R, blit, frame_rect, rect, sprite


class Kick4(V4Game):
    family = "kick4"
    mechanic = "object"
    principle = ("contrôle indirect : lancer des blocs qui glissent vers des cadres-cibles (zones, plaques, portails, "
                 "2e pilote)", ["ka59", "wa30", "sk48"])
    simple_ops = DIR_OPS + ("interact",)
    CAT = ("zone", "stop", "second", "portal", "keyed", "sticky")
    free_miss = True
    exact_progress = True
    table_cap = 15000

    def setup(self, rng):
        r = self.orng
        self.uses_click = True
        self.variant = str(rng.choice(["kick", "kick", "push"]))
        self.cat = [str(x) for x in rng.permutation(self.CAT)]
        self.C = int(r.choice([6, 7, 7, 8])) if self.s_screen == 1 else 3
        self.pilot_kind = str(r.choice(["frame", "framed", "ring"]))
        self.block_kind = str(r.choice(["box", "framed", "blob", "diamond", "disc"]))
        self.target_kind = str(r.choice(["frame", "frame", "plus_outline"]))
        self.strip = bool(r.random() < 0.6)
        self.top = (6 if self.s_screen == 1 else 4) if self.strip else 0
        C = self.C
        sb = C - 2 if self.block_kind in ("box", "diamond", "disc") else C
        self.tpl_block = sprite(r, self.block_kind, sb, sb, 100, 101)
        self.n_rooms = int(r.integers(1, 4))

    # ---------------------------------------------------------------- géométrie
    def _grid(self, rng, i, n):
        C = self.C
        top = self.top
        Hc, Wc = (self.AH - top) // C, self.AW // C
        d = i / max(1, n - 1)
        side = max(5, int(round(min(Hc, Wc) * (0.55 + 0.4 * d))))
        need = min(20 + 3 * i, int(0.5 * (Hc - 2) * (Wc - 2)))
        for _ in range(30):
            m = draw.rooms_mask(rng, Hc, Wc, self.n_rooms + (i >= 3), min_side=3, max_side=side)
            m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = False
            m = draw.largest_component(m)
            if m.sum() >= need:
                break
        oy = top + (self.AH - top - Hc * C) // 2
        ox = (self.AW - Wc * C) // 2
        return m, Hc, Wc, oy, ox

    def cell_of(self, lvl, px):
        if px is None:
            return None
        r, c = (px[0] - lvl["oy"]) // self.C, (px[1] - lvl["ox"]) // self.C
        if 0 <= r < lvl["Hc"] and 0 <= c < lvl["Wc"]:
            return (int(r), int(c))
        return None

    def cell_px(self, lvl, p):
        return (lvl["oy"] + p[0] * self.C + self.C // 2, lvl["ox"] + p[1] * self.C + self.C // 2)

    # ---------------------------------------------------------------- règles
    def _floor(self, lvl, q):
        return 0 <= q[0] < lvl["Hc"] and 0 <= q[1] < lvl["Wc"] and lvl["floor"][q]

    def _slide(self, lvl, q, d, occ, tags):
        """bloc en q poussé vers d -> position finale (ou None si immobile)."""
        p, moved, hops = q, False, 0
        while True:
            nq = (p[0] + d[0], p[1] + d[1])
            if not self._floor(lvl, nq) or nq in occ:
                break
            p, moved = nq, True
            if p in lvl["portal"] and hops < 2:
                dst = lvl["portal"][p]
                if dst not in occ:
                    p = dst
                    hops += 1
                    tags.add("tele")
            if p in lvl["stop"] or self.variant == "push":
                if p in lvl["stop"]:
                    tags.add("stop")
                break
        return p if moved else None

    def _step(self, lvl, st, op, px):
        pil, sel, blocks, stuck = st
        tags = set()
        if op == "click":
            q = self.cell_of(lvl, px)
            if q in pil and pil.index(q) != sel:
                tags.add("select")
                return (pil, pil.index(q), blocks, stuck), tags
            return st, tags
        if op == "interact":
            if len(pil) > 1:
                tags.add("select")
                return (pil, (sel + 1) % len(pil), blocks, stuck), tags
            return st, tags
        if stuck:
            tags.add("stuck")
            return (pil, sel, blocks, 0), tags
        d = DIRS[op]
        p = pil[sel]
        q = (p[0] + d[0], p[1] + d[1])
        bpos = {(b[0], b[1]): b[2] for b in blocks}
        if not self._floor(lvl, q) or q in lvl["zone"] or q in pil:
            tags.add("bump")
            return st, tags
        if q in bpos:
            occ = set(bpos) | set(pil)
            occ.discard(q)
            dst = self._slide(lvl, q, d, occ | {p}, tags)
            if dst is None:
                tags.add("bump")
                return st, tags
            col = bpos.pop(q)
            bpos[dst] = col
            tags.add("kick")
            nb = tuple(sorted((a, b, c) for (a, b), c in bpos.items()))
            if self.variant == "push":
                pil = pil[:sel] + (q,) + pil[sel + 1:]
            return (pil, sel, nb, 0), tags
        pil = pil[:sel] + (q,) + pil[sel + 1:]
        tags.add("move")
        return (pil, sel, blocks, 1 if q in lvl["sticky"] else 0), tags

    def apply(self, lvl, st, op, px):
        st2, _ = self._step(lvl, st, op, px)
        return st2, False

    def is_win(self, lvl, st):
        bpos = {(b[0], b[1]): b[2] for b in st[2]}
        for (r, c, col) in lvl["targets"]:
            if (r, c) not in bpos or (col >= 0 and bpos[(r, c)] != col):
                return False
        return True

    def n_filled(self, lvl, st):
        bpos = {(b[0], b[1]): b[2] for b in st[2]}
        return sum((r, c) in bpos and (col < 0 or bpos[(r, c)] == col) for r, c, col in lvl["targets"])

    def click_targets(self, lvl, st):
        return [self.cell_px(lvl, p) for k, p in enumerate(st[0]) if k != st[1]] if len(st[0]) > 1 else []

    def click_hit(self, lvl, st, px):
        q = self.cell_of(lvl, px)
        return q is not None and (q in st[0] or q in {(b[0], b[1]) for b in st[2]})

    # ---------------------------------------------------------------- niveaux
    def gen_level(self, rng, i, n):
        mods, newest = mods_for(self.cat, i)
        t2 = self.__dict__.setdefault("_t2", {})
        t2[i] = t2.get(i, 0) + 1
        if t2[i] > 40 or self.relax:                                          # repli : sans les règles qui font exploser l'état
            mods = tuple(m for m in mods if m not in ("second", "sticky") or m == newest)
        floor, Hc, Wc, oy, ox = self._grid(rng, i, n)
        fr = [tuple(map(int, p)) for p in np.argwhere(floor)]
        if len(fr) < 16:
            return self.why("r1")
        rng.shuffle(fr)
        nb = min(2 + (i >= 2) + (i >= 4) + (i >= 6), 5)
        npil = 2 if "second" in mods else 1
        lvl = dict(floor=floor, Hc=Hc, Wc=Wc, oy=oy, ox=ox, zone=frozenset(), stop=frozenset(), portal={},
                   sticky=frozenset(), targets=(), mods=mods)
        k = 0
        pil = tuple(fr[k:k + npil]); k += npil
        blocks0 = fr[k:k + nb]; k += nb
        rest = fr[k:]
        if "zone" in mods:                                      # une bande (ligne ou colonne) de la pièce
            if rng.random() < 0.5:
                r0 = int(rng.integers(1, Hc - 1))
                z = {(r0, c) for c in range(Wc) if floor[r0, c]}
            else:
                c0 = int(rng.integers(1, Wc - 1))
                z = {(r, c0) for r in range(Hc) if floor[r, c0]}
            z -= set(pil) | set(blocks0)
            lvl["zone"] = frozenset(z)
            rest = [p for p in rest if p not in z]
        if "stop" in mods:
            lvl["stop"] = frozenset(rest[:2 + i // 3]); rest = rest[2 + i // 3:]
        if "sticky" in mods:
            lvl["sticky"] = frozenset(rest[:3]); rest = rest[3:]
        if "portal" in mods and len(rest) >= 2:
            a, b = rest[0], rest[1]
            lvl["portal"] = {a: b, b: a}
            rest = rest[2:]
        keyed = "keyed" in mods
        cols = [int(x) for x in rng.integers(0, 3, nb)] if keyed else [0] * nb
        blocks = tuple(sorted((p[0], p[1], cols[j]) for j, p in enumerate(blocks0)))
        st0 = (pil, 0, blocks, 0)
        # cibles = configuration de blocs la PLUS LOINTAINE (BFS exhaustif plafonné) : soluble et solution optimale
        from collections import deque
        prev, depth, q = {st0: None}, {st0: 0}, deque([st0])
        moves = [(o, None) for o in DIR_OPS] + ([("interact", None)] if npil > 1 else [])
        while q and len(depth) < (6000 if npil > 1 else 10000):
            s_ = q.popleft()
            for mv in moves:
                s2, _ = self.apply(lvl, s_, *mv)
                if s2 not in depth:
                    depth[s2] = depth[s_] + 1
                    prev[s2] = (s_, mv)
                    q.append(s2)
        special = set(lvl["portal"]) | lvl["stop"] | lvl["sticky"]
        start = {(b[0], b[1]) for b in blocks}
        best = {}
        for s_, dd in depth.items():
            bl = s_[2]
            if any((b[0], b[1]) in special for b in bl) or sum((b[0], b[1]) not in start for b in bl) < (len(bl) + 1) // 2:
                continue
            if bl not in best or dd < depth[best[bl]]:
                best[bl] = s_
        if not best:
            return self.why("r2")
        dm = max(depth[s_] for s_ in best.values())
        nf = self.__dict__.setdefault("_tries", {})
        nf[i] = nf.get(i, 0) + 1
        ml = self.min_len(i, n) * max(0.3, 1 - nf[i] / 60)
        if dm < ml:
            return self.why("r3")
        far = [s_ for s_ in best.values() if depth[s_] >= 0.85 * dm]
        s_ = far[int(rng.integers(len(far)))]
        lvl["targets"] = tuple(sorted((b[0], b[1], b[2] if keyed else -1) for b in s_[2]))
        sol = []
        while prev[s_] is not None:
            s_, mv = prev[s_]
            sol.append(mv)
        sol = sol[::-1]
        if newest is not None:                                  # la règle nouvelle doit servir
            need = dict(zone=None, stop="stop", second="select", portal="tele", keyed=None, sticky="stuck")[newest]
            tags, x = set(), st0
            for mv in sol:
                x, tg = self._step(lvl, x, *mv)
                tags |= tg
            if need is not None and need not in tags and not self.relax and self.__dict__.setdefault("_fails", {}).setdefault(i, [0])[0] < 30:
                self._fails[i][0] += 1
                return self.why("r4")
        return lvl, st0, sol

    def min_len(self, i, n):
        return min(16 + 6 * i, 60)

    def heuristic(self, lvl, st):
        miss = len(lvl["targets"]) - self.n_filled(lvl, st)
        p = st[0][st[1]]
        far = min((abs(p[0] - b[0]) + abs(p[1] - b[1]) for b in st[2]), default=0)
        return 6 * miss + far

    def body(self, lvl, st):
        p = st[0][st[1]]
        return (p[0], p[1], lvl["Hc"], lvl["Wc"])

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        _, tags = self._step(lvl, st, op, px)
        if "kick" in tags:
            ev[EV["push"]] = ev[EV["touch"]] = 1
        if "bump" in tags or "stuck" in tags:
            ev[EV["touch"]] = 1
        if "select" in tags:
            ev[EV["toggle"]] = 1
        if "tele" in tags:
            ev[EV["teleport"]] = 1
        if "stop" in tags:
            ev[EV["color"]] = 1
        f0, f1 = self.n_filled(lvl, st), self.n_filled(lvl, st2)
        if f1 > f0:
            ev[EV["collect"]] = ev[EV["open"]] = 1

    # ---------------------------------------------------------------- rendu
    def render(self, lvl, st):
        pal, C = self.pal, self.C
        g = self.blank()
        oy, ox = lvl["oy"], lvl["ox"]
        for r, c in np.argwhere(lvl["floor"]):
            rect(g, oy + r * C, ox + c * C, C, C, pal.room)
        for (r, c) in lvl["zone"]:
            rect(g, oy + r * C, ox + c * C, C, C, pal.a(6))
        for (r, c) in lvl["stop"]:
            rect(g, oy + r * C + 1, ox + c * C + 1, C - 2, C - 2, pal.wall)
        for (r, c) in lvl["sticky"]:
            blit(g, sprite(None, "diamond", C - 2, C - 2, pal.a(7)), oy + r * C + 1, ox + c * C + 1)
        for (r, c) in lvl["portal"]:
            blit(g, sprite(None, "ring", C, C, pal.a(5)), oy + r * C, ox + c * C)
        bcol = [pal.a(0), pal.a(1), pal.a(2)]
        for (r, c, col) in lvl["targets"]:
            tc = bcol[col] if col >= 0 else pal.fg
            blit(g, sprite(None, self.target_kind, C, C, tc), oy + r * C, ox + c * C)
        o = (C - self.tpl_block.shape[0]) // 2
        for (r, c, col) in st[2]:
            blit(g, draw.recolor(self.tpl_block, {100: bcol[col], 101: pal.fg}), oy + r * C + o, ox + c * C + o)
        for k, (r, c) in enumerate(st[0]):
            col = draw.WHITE if (k == st[1] and len(st[0]) > 1 and pal.room != draw.WHITE) else pal.fg
            blit(g, sprite(None, self.pilot_kind, C, C, col, None, pal.a(3) if k == st[1] else None), oy + r * C, ox + c * C)
            if st[3] and k == st[1]:
                g[oy + r * C, ox + c * C] = R(pal.a(7))
        if self.strip:                                          # bandeau : une case par cible, pleine si remplie
            hs = self.top - 1
            rect(g, 0, 0, hs, self.AW, pal.ui)
            f = self.n_filled(lvl, st)
            b = hs - 2
            for j in range(len(lvl["targets"])):
                x = 1 + (b + 1) * j
                if j < f:
                    rect(g, 1, x, b, b, pal.a(4))
                else:
                    frame_rect(g, 1, x, b, b, pal.line)
        return g
