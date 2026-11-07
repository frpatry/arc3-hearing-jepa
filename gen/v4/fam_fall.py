"""fall4 (avatar + clic, gravité) : vue de côté ; l'avatar marche à gauche / à droite et TOMBE jusqu'à un appui ;
on remodèle le terrain en CLIQUANT : les blocs friables (fissurés) disparaissent, les autres sont inertes.
But : atteindre la gemme. Principes : remodeler le monde pour que la dynamique (chute) mène au but, la plupart des
clics sont inertes, actions sans effet visible (saut sur place, ACTION5 avant que la règle n'existe).
Variantes : `walk` (gauche / droite, haut = saut sur place sans effet) ou `slide` (gauche / droite font glisser
l'avatar jusqu'au mur, façon glace).
Règles ajoutées (ordre tiré par jeu) :
  flip   : ACTION5 inverse la gravité (l'avatar « tombe » vers le haut) ;
  spikes : piques : y tomber = mort (vie perdue / GAME_OVER) ;
  crate  : caisses qui tombent aussi et que l'avatar pousse de côté ;
  build  : emplacements pointillés : un clic y pose un bloc (une seule fois) ;
  limit  : nombre de clics limité (compteur dans le bandeau) ;
  key    : une clé à ramasser avant que la gemme ne s'ouvre.
Solvabilité : BFS exact sur (avatar, blocs retirés / posés, gravité, caisses, clé, clics restants)."""
from collections import deque

import numpy as np

from ..core import DIR_OPS, EV
from . import draw
from .base import V4Game, mods_for
from .draw import R, blit, frame_rect, rect, sprite


class Fall4(V4Game):
    family = "fall4"
    weight = 0.4                                   # construction lente (BFS à chute), niveaux courts
    mechanic = "avatar"
    principle = ("gravité : remodeler le terrain par clics (blocs friables, blocs posés, gravité inversée) pour que la "
                 "chute mène au but", ["bp35", "sp80", "lf52"])
    simple_ops = DIR_OPS + ("interact",)
    CAT = ("flip", "spikes", "crate", "build", "limit", "key")
    free_miss = False
    table_cap = 20000

    def setup(self, rng):
        r = self.orng
        self.uses_click = True
        self.variant = str(rng.choice(["walk", "walk", "slide"]))
        self.cat = [str(x) for x in rng.permutation(self.CAT)]
        self.C = int(r.choice([4, 5, 5, 6])) if self.s_screen == 1 else int(r.choice([3, 4]))
        self.tex = str(r.choice(["box", "box", "framed"]))
        self.av_tpl = sprite(r, str(r.choice(["blob", "framed", "diamond"])), self.C, self.C, 100, 101)
        self.gem_tpl = sprite(r, str(r.choice(["plus", "diamond", "ring", "blob"])), self.C, self.C, 102, 103)
        self.top = (5 if self.s_screen == 1 else 3)

    # ---------------------------------------------------------------- règles
    def _solid(self, lvl, st, q):
        H, W = lvl["Hc"], lvl["Wc"]
        if not (0 <= q[0] < H and 0 <= q[1] < W):
            return True
        if q in lvl["rock"]:
            return True
        if q in lvl["soft"] and q not in st[1]:
            return True
        if q in st[2]:                                         # blocs posés
            return True
        return False

    def _settle(self, lvl, st, av, crates, g):
        """tout tombe jusqu'au repos -> (av, caisses, mort, a chuté)."""
        fell = False
        for _ in range(4 * lvl["Hc"]):
            moved = False
            occ = set(crates) | {av}
            nc = []
            for c in sorted(crates, key=lambda p: -p[0] * g):
                q = (c[0] + g, c[1])
                if not self._solid(lvl, st, q) and q not in occ and 0 <= q[0] < lvl["Hc"]:
                    occ.discard(c)
                    occ.add(q)
                    nc.append(q)
                    moved = True
                else:
                    nc.append(c)
            crates = tuple(sorted(nc))
            q = (av[0] + g, av[1])
            if not self._solid(lvl, st, q) and q not in set(crates):
                av = q
                moved = fell = True
                if av in lvl["spikes"]:
                    return av, crates, True, fell
            if not moved:
                break
        return av, crates, av in lvl["spikes"], fell

    def _step(self, lvl, st, op, px):
        av, gone, built, g, crates, key, clicks = st
        tags = set()
        if op == "click":
            q = self.cell_of(lvl, px)
            if q is None or ("limit" in lvl["mods"] and clicks <= 0):
                return st, False, tags
            if q in lvl["soft"] and q not in gone and q != av:
                gone = tuple(sorted(gone + (q,)))
                tags.add("break")
            elif q in lvl["spots"] and q not in built and q != av and q not in crates:
                built = tuple(sorted(built + (q,)))
                tags.add("build")
            else:
                return st, False, tags
            clicks -= 1
        elif op == "interact":
            if "flip" not in lvl["mods"]:
                return st, False, tags
            g = -g
            tags.add("flip")
        elif op in ("left", "right"):
            d = -1 if op == "left" else 1
            for _ in range(lvl["Wc"] if self.variant == "slide" else 1):
                q = (av[0], av[1] + d)
                st_ = (av, gone, built, g, crates, key, clicks)
                if self._solid(lvl, st_, q):
                    break
                if q in crates:
                    q2 = (q[0], q[1] + d)
                    if self._solid(lvl, st_, q2) or q2 in crates:
                        break
                    crates = tuple(sorted([c for c in crates if c != q] + [q2]))
                    tags.add("push")
                av = q
                tags.add("walk")
                if self.variant == "slide":
                    below = (av[0] + g, av[1])
                    if not self._solid(lvl, st_, below) and below not in crates:
                        break
        else:                                                  # haut / bas : saut sur place, rien ne change
            return st, False, tags
        st2 = (av, gone, built, g, crates, key, clicks)
        av, crates, dead, fell = self._settle(lvl, st2, av, crates, g)
        if fell:
            tags.add("fall")
        if lvl["key"] is not None and av == lvl["key"] and not key:
            key = True
            tags.add("key")
        return (av, gone, built, g, crates, key, clicks), dead, tags

    def apply(self, lvl, st, op, px):
        st2, dead, _ = self._step(lvl, st, op, px)
        return st2, dead

    def is_win(self, lvl, st):
        return st[0] == lvl["gem"] and (lvl["key"] is None or st[5])

    def cell_of(self, lvl, px):
        if px is None:
            return None
        r, c = (px[0] - lvl["oy"]) // self.C, (px[1] - lvl["ox"]) // self.C
        if px[0] < lvl["oy"] or px[1] < lvl["ox"] or not (0 <= r < lvl["Hc"] and 0 <= c < lvl["Wc"]):
            return None
        return (int(r), int(c))

    def cell_px(self, lvl, p):
        return (lvl["oy"] + p[0] * self.C + self.C // 2, lvl["ox"] + p[1] * self.C + self.C // 2)

    def moves(self, lvl, st):
        mv = [("left", None), ("right", None)]
        if "flip" in lvl["mods"]:
            mv.append(("interact", None))
        mv += [("click", p) for p in self.click_targets(lvl, st)]
        return mv

    def click_targets(self, lvl, st):
        out = [self.cell_px(lvl, q) for q in lvl["soft"] if q not in st[1]]
        out += [self.cell_px(lvl, q) for q in lvl["spots"] if q not in st[2]]
        return out

    def click_hit(self, lvl, st, px):
        q = self.cell_of(lvl, px)
        return q is not None and (self._solid(lvl, st, q) or q in lvl["spots"])

    # ---------------------------------------------------------------- niveaux
    def gen_level(self, rng, i, n):
        mods, newest = mods_for(self.cat, i)
        C = self.C
        Hc, Wc = (self.AH - self.top) // C, self.AW // C
        Hc, Wc = min(Hc, 12), min(Wc, 14)
        rock, soft = set(), set()
        for c in range(Wc):
            rock.add((Hc - 1, c))
        # plates-formes horizontales à des hauteurs tirées ; certaines cases friables
        nplat = int(rng.integers(3, 5 + (i >= 3)))
        for _ in range(nplat):
            r0 = int(rng.integers(2, Hc - 2))
            c0 = int(rng.integers(0, Wc - 3))
            ln = int(rng.integers(3, max(4, Wc // 2)))
            for c in range(c0, min(Wc, c0 + ln)):
                (soft if rng.random() < 0.3 and len(soft) < 7 else rock).add((r0, c))
        for _ in range(int(rng.integers(1, 3))):                # piliers
            c0 = int(rng.integers(1, Wc - 1))
            r1 = int(rng.integers(2, Hc - 2))
            for r in range(r1, Hc - 1):
                (soft if rng.random() < 0.25 and len(soft) < 7 else rock).add((r, c0))
        if "flip" in mods:
            for c in range(Wc):
                rock.add((0, c))
        free = [(r, c) for r in range(Hc) for c in range(Wc) if (r, c) not in rock and (r, c) not in soft]
        if len(free) < 20:
            return self.why("plein")
        spikes = set()
        if "spikes" in mods:
            fl = [p for p in free if (p[0] + 1, p[1]) in rock and p[0] >= Hc // 2]
            for k in rng.permutation(len(fl))[:2 + i // 3]:
                spikes.add(fl[k])
        spots = set()
        if "build" in mods:
            for k in rng.permutation(len(free))[:3]:
                if free[k] not in spikes:
                    spots.add(free[k])
        lvl = dict(Hc=Hc, Wc=Wc, rock=frozenset(rock), soft=frozenset(soft), spikes=frozenset(spikes),
                   spots=frozenset(spots), gem=None, key=None, mods=mods,
                   oy=self.top + (self.AH - self.top - Hc * C) // 2, ox=(self.AW - Wc * C) // 2)
        cand = [p for p in free if p not in spikes and p not in spots and p[0] <= Hc // 2]
        if not cand:
            return self.why("départ")
        av0 = cand[int(rng.integers(len(cand)))]
        crates = ()
        if "crate" in mods:
            cc = [p for p in free if p not in spikes and p != av0 and p not in spots]
            crates = tuple(sorted(cc[k] for k in rng.permutation(len(cc))[:2]))
        clicks = 99
        if "limit" in mods:
            clicks = int(rng.integers(2, 5))
        st0 = (av0, (), (), 1, crates, False, clicks)
        av, cr, dead, _ = self._settle(lvl, st0, av0, crates, 1)
        if dead:
            return self.why("mort au départ")
        st0 = (av, (), (), 1, cr, False, clicks)
        # BFS : gemme = case d'avatar la plus lointaine (repos) ; clé éventuelle = case intermédiaire
        prev, depth, q = {st0: None}, {st0: 0}, deque([st0])
        while q and len(depth) < 4000:
            s_ = q.popleft()
            for mv in self.moves(lvl, s_):
                s2, dd = self.apply(lvl, s_, *mv)
                if dd or s2 in depth:
                    continue
                depth[s2] = depth[s_] + 1
                prev[s2] = (s_, mv)
                q.append(s2)
        best = {}
        for s_, d in depth.items():
            if s_[0] not in best or d < depth[best[s_[0]]]:
                best[s_[0]] = s_
        dm = max(depth[s_] for s_ in best.values())
        nf = self.__dict__.setdefault("_tries", {})
        nf[i] = nf.get(i, 0) + 1
        if dm < self.min_len(i, n) * max(0.3, 1 - nf[i] / 60):
            return self.why("court")
        far = [s_ for s_ in best.values() if depth[s_] >= 0.85 * dm]
        s_ = far[int(rng.integers(len(far)))]
        lvl["gem"] = s_[0]
        path = []
        while prev[s_] is not None:
            s_, mv = prev[s_]
            path.append(mv)
        path = path[::-1]
        if "key" in mods:                                      # clé sur une case traversée à mi-chemin
            x, seen = st0, []
            for mv in path:
                x, _ = self.apply(lvl, x, *mv)
                seen.append(x[0])
            mid = [p for p in seen[len(seen) // 3:2 * len(seen) // 3] if p != lvl["gem"]]
            if not mid:
                return self.why("clé")
            lvl["key"] = mid[len(mid) // 2]
        if newest is not None and nf[i] < 40 and not self.relax:
            need = dict(flip="flip", spikes=None, crate="push", build="build", limit=None, key=None)[newest]
            if need:
                tags, x = set(), st0
                for mv in path:
                    x, _, tg = self._step(lvl, x, *mv)
                    tags |= tg
                if need not in tags:
                    return self.why("règle inutile")
        return lvl, st0, path

    def min_len(self, i, n):
        return min(6 + 2 * i, 20)

    def heuristic(self, lvl, st):
        a, g = st[0], lvl["gem"]
        return abs(a[0] - g[0]) + abs(a[1] - g[1])

    def body(self, lvl, st):
        return (st[0][0], st[0][1], lvl["Hc"], lvl["Wc"])

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        _, _, tags = self._step(lvl, st, op, px)
        if tags & {"break", "build"}:
            ev[EV["touch"]] = ev[EV["toggle"]] = 1
        if "push" in tags:
            ev[EV["push"]] = ev[EV["touch"]] = 1
        if "fall" in tags:
            ev[EV["touch"]] = 1
        if "flip" in tags:
            ev[EV["color"]] = 1
        if "key" in tags:
            ev[EV["collect"]] = 1
        if dead or any(abs(st2[0][0] - a) + abs(st2[0][1] - b) <= 1 for a, b in lvl["spikes"]):
            ev[EV["danger"]] = 1

    # ---------------------------------------------------------------- rendu
    def render(self, lvl, st):
        pal, C = self.pal, self.C
        g = self.blank()
        oy, ox = lvl["oy"], lvl["ox"]
        av, gone, built, gr, crates, key, clicks = st

        def at(p):
            return oy + p[0] * C, ox + p[1] * C

        for p in lvl["rock"]:
            y, x = at(p)
            if self.tex == "box":
                rect(g, y, x, C, C, pal.wall)
            else:
                blit(g, sprite(None, self.tex, C, C, pal.wall, pal.room), y, x)
        for p in lvl["soft"]:
            if p in gone:
                continue
            y, x = at(p)
            rect(g, y, x, C, C, pal.a(5))
            g[y + C // 2, x + 1:x + C - 1] = R(pal.bg)                   # fissure
        for p in lvl["spots"]:
            y, x = at(p)
            if p in built:
                rect(g, y, x, C, C, pal.a(6))
            else:
                frame_rect(g, y, x, C, C, pal.line)
                g[y + 1:y + C - 1, x + 1:x + C - 1] = R(pal.bg)
        for p in lvl["spikes"]:
            y, x = at(p)
            s = sprite(None, "arrow", C, C, draw.RED if pal.bg != draw.RED else draw.ORANGE)
            blit(g, s, y, x)
        for p in crates:
            y, x = at(p)
            blit(g, sprite(None, "framed", C, C, pal.a(7), pal.fg), y, x)
        if lvl["key"] is not None and not key:
            y, x = at(lvl["key"])
            blit(g, sprite(None, "diamond", C - 2, C - 2, draw.YELLOW if pal.bg != draw.YELLOW else draw.WHITE), y + 1, x + 1)
        y, x = at(lvl["gem"])
        open_ = lvl["key"] is None or key
        blit(g, draw.recolor(self.gem_tpl, {102: pal.a(3) if open_ else pal.line, 103: pal.fg}), y, x)
        y, x = at(av)
        s = draw.recolor(self.av_tpl, {100: pal.a(0), 101: pal.fg})
        blit(g, s[::-1] if gr < 0 else s, y, x)
        # bandeau : gravité, clics restants
        rect(g, 0, 0, self.top - 1, self.AW, pal.ui)
        if "flip" in lvl["mods"]:
            blit(g, np.rot90(sprite(None, "arrow", self.top - 1, self.top - 1, pal.line), 2 if gr > 0 else 0), 0, 1)
        if "limit" in lvl["mods"]:
            for k in range(min(clicks, 8)):
                rect(g, 1, self.AW - 3 - 3 * k, max(1, self.top - 3), 2, pal.a(5))
        return g
