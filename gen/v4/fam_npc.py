"""npc4 (avatar, agents autonomes au tour par tour) : des PNJ aux règles simples et lisibles agissent après chaque
action du joueur.
Variantes (tirées par jeu) :
  dodge : ramasser les gemmes puis atteindre la sortie (elle s'allume quand toutes les gemmes sont prises) sans être
          touché ; ACTION5 = attendre un tour (les PNJ agissent) ; un pas contre un mur ne fait RIEN (les PNJ
          n'agissent qu'après un vrai coup). Touché = vie perdue / fin de partie. Types de PNJ (un nouveau par niveau) :
            patrol : va-et-vient sur sa voie pointillée, un pas par tour ;
            sentry : tourelle qui pivote d'un quart de tour tous les 2 tours ; son regard (pointillés) est mortel ;
            mimic  : double creux qui refait votre coup du tour précédent (bloqué par les murs) ;
            rail   : chien de garde sur un rail : un pas tous les 2 tours vers le point du rail le plus proche de vous ;
            gem2   : une 2e gemme à ramasser.
  haul  : déposer toutes les caisses dans la zone (ACTION5 = prendre la caisse sous soi / la poser dans la zone) ;
          les PNJ ont la MÊME routine de portage mais des destinations opposées :
            helper : porte les caisses libres jusqu'à la zone ;
            thief  : reprend les caisses déposées dans la zone et les emporte dans sa tanière (sinon il rentre) ;
                     foncer (mains vides) dans un voleur chargé lui reprend la caisse et le renvoie chez lui ;
            fence  : barrière que les PNJ ne traversent pas (le joueur oui) ;
            slow   : les PNJ n'agissent qu'un tour sur deux.
Solvabilité : BFS exact sur (joueur, phases des PNJ, gemmes / caisses) ; dodge : sortie placée là où la partie la
plus courte est la plus longue."""
from math import gcd

import numpy as np

from ..core import DIRS, DIR_OPS, EV
from . import avkit as K
from . import draw
from .base import V4Game, mods_for
from .draw import R, blit, frame_rect, rect, sprite

CW = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def lcm(a, b):
    return a * b // gcd(a, b)


class Npc4(V4Game):
    family = "npc4"
    mechanic = "avatar"
    principle = ("agents autonomes au tour par tour (patrouille, tourelle, double retardé, chien sur rail ; aides et "
                 "voleurs à la même routine de portage, destinations opposées)", ["tu93", "wa30", "g50t", "su15"])
    simple_ops = DIR_OPS + ("interact",)
    CAT_A = ("patrol", "sentry", "mimic", "rail", "gem2")
    CAT_B = ("helper", "fence", "thief", "slow")
    exact_progress = True
    table_cap = 25000
    scale_p = (0.95, 0.05)                  # grands plateaux : surtout échelle ×1
    bfs_cap = 40000

    def setup(self, rng):
        r = self.orng
        self.variant = str(rng.choice(["dodge", "dodge", "haul"]))
        if self.variant == "dodge":
            cat = [str(x) for x in rng.permutation(self.CAT_A)]
            if cat[0] == "gem2":
                cat = cat[1:] + cat[:1]
        else:
            cat = ["helper"] + [str(x) for x in rng.permutation(["fence", "thief", "slow"])]
            if cat[1] == "slow":
                cat[1], cat[2] = cat[2], cat[1]
        self.cat = cat
        self.C = int(r.choice([4, 4, 5])) if self.s_screen == 1 else 3
        self.hh = 6 if self.s_screen == 1 else 4
        self.hud_top = bool(r.random() < 0.5)
        self.floor_style = str(r.choice(["rooms", "rooms", "cave"]))
        self.player_kind = str(r.choice(["blob", "framed", "disc"]))
        self.tpl = {S: sprite(r, self.player_kind, S, S, 100, 101, 102 if S >= 5 else None)
                    for S in range(3, self.C + 5)}

    # ---------------------------------------------------------------- géométrie
    def _board(self, rng, maxc, i, n):
        top = self.hh if self.hud_top else 0
        bot = 0 if self.hud_top else self.hh
        C, Hc, Wc, oy, ox = K.board(self.AH, self.AW, self.C, maxc / 0.5 + 8, top=top, bottom=bot, grow=3)
        d = i / max(1, n - 1)
        side = max(4, int(round(min(Hc, Wc) * (0.55 + 0.35 * d))))
        floor = K.make_floor(rng, Hc, Wc, self.floor_style, n_rooms=2 + (i >= 3), need=min(int(maxc * 0.75), int(0.5 * Hc * Wc)),
                             holes=float(rng.uniform(0.03, 0.08)) + 0.015 * min(i, 6), max_side=side, maxc=maxc)
        return floor, dict(Hc=Hc, Wc=Wc, oy=oy, ox=ox, C=C)

    @staticmethod
    def _segments(floor, lo, hi):
        """segments droits de sol de longueur lo..hi."""
        H, W = floor.shape
        out = []
        for r in range(H):
            for c in range(W):
                for d in ((0, 1), (1, 0)):
                    seg = []
                    y, x = r, c
                    while 0 <= y < H and 0 <= x < W and floor[y, x] and len(seg) < hi:
                        seg.append((y, x))
                        if len(seg) >= lo:
                            out.append(list(seg))
                        y, x = y + d[0], x + d[1]
        return out

    # ================================================================= variante dodge
    def _los(self, lvl, p, d):
        out, q = [], (p[0] + d[0], p[1] + d[1])
        fl = lvl["floor"]
        while 0 <= q[0] < lvl["Hc"] and 0 <= q[1] < lvl["Wc"] and fl[q]:
            out.append(q)
            q = (q[0] + d[0], q[1] + d[1])
        return out

    def _npc_cells(self, lvl, st):
        """cases occupées par les PNJ dans l'état st (dodge)."""
        p, t, gm, rails, mim = st
        occ = [path[t % len(path)] for path in lvl["pat_t"]]
        occ += [lvl["sentries"][k][0] for k in range(len(lvl["sentries"]))]
        occ += [lvl["rails"][k][j] for k, j in enumerate(rails)]
        if mim is not None:
            occ.append(mim[0])
        return occ

    def _step_dodge(self, lvl, st, op, px):
        p, t, gm, rails, mim = st
        tags = set()
        if op == "click":
            return st, tags, False
        if op == "interact":
            q = p
            dmove = None
            tags.add("wait")
        else:
            dmove = DIRS[op]
            q = (p[0] + dmove[0], p[1] + dmove[1])
            if not (0 <= q[0] < lvl["Hc"] and 0 <= q[1] < lvl["Wc"]) or not lvl["floor"][q]:
                tags.add("bump")
                return st, tags, False
        occ0 = self._npc_cells(lvl, st)
        if q != p and q in occ0:
            tags.add("hit")
            return st, tags, True
        gm2 = gm
        for k, gp in enumerate(lvl["gems"]):
            if gp == q and not gm >> k & 1:
                gm2 |= 1 << k
                tags.add("gem")
        full = (1 << len(lvl["gems"])) - 1
        if q == lvl["exit"] and gm2 == full:
            tags.add("exit")
            return (q, t, gm2, rails, mim), tags, False
        t2 = (t + 1) % lvl["P"]
        nr = []
        for k, j in enumerate(rails):                           # chien sur rail : un pas tous les 2 tours
            if t2 % 2 == 0:
                tgt = lvl["rail_tgt"][k][q]
                j = j + (1 if tgt > j else -1 if tgt < j else 0)
            nr.append(j)
        nr = tuple(nr)
        nm = mim
        if mim is not None:                                     # double : refait le coup précédent
            mp, last = mim
            if last is not None:
                dd = CW[last]
                m2 = (mp[0] + dd[0], mp[1] + dd[1])
                if 0 <= m2[0] < lvl["Hc"] and 0 <= m2[1] < lvl["Wc"] and lvl["floor"][m2]:
                    mp = m2
            nm = (mp, CW.index(dmove) if dmove is not None else None)
        st2 = (q, t2, gm2, nr, nm)
        occ1 = self._npc_cells(lvl, st2)
        if q in occ1:
            tags.add("hit")
            return st2, tags, True
        for a, b in zip(occ0, occ1):                            # croisement
            if a == q and b == p:
                tags.add("hit")
                return st2, tags, True
        for (sp, f0) in lvl["sentries"]:
            f = (f0 + t2 // 2) % 4
            if q in lvl["los"][(sp, f)]:
                tags.add("seen")
                return st2, tags, True
        if t2 // 2 != t // 2 and lvl["sentries"]:
            tags.add("turn")
        return st2, tags, False

    def _gen_dodge(self, rng, i, n, mods, newest):
        fac = dict(mimic=0.45, rail=0.8, patrol=0.9, sentry=0.9, gem2=0.85)
        maxc = 75.0 + 6 * min(i, 6)
        for m in mods:
            maxc *= fac.get(m, 1.0)
        shr = self.__dict__.setdefault("_shr", {})             # réduit tout seul si l'exploration déborde
        maxc = max(24, int(maxc * shr.get(i, 1.0)))
        floor, geo = self._board(rng, maxc, i, n)
        if floor is None:
            return self.why("floor")
        lvl = dict(floor=floor, mods=mods, **geo)
        fr = K.cells(floor)
        rng.shuffle(fr)
        used = set()
        P = 1
        pats, sents, rails = [], [], []
        # cases « passantes » : chemin du départ probable vers le point le plus éloigné (les PNJ s'y placent)
        dm0 = K.grid_dist(floor, fr[0])
        f1 = max(dm0, key=dm0.get)
        dm1 = K.grid_dist(floor, f1)
        f2 = max(dm1, key=dm1.get)
        busy, q = {f2}, f2
        while q != f1:
            q = next(nb for nb in K.nbrs(floor, q) if dm1.get(nb, 99) == dm1[q] - 1)
            busy.add(q)
        lvl["busy"] = (f1, f2)

        def on_busy(segs):
            b = [s for s in segs if busy & set(s) and f1 not in s and f2 not in s]
            return b if b and rng.random() < 0.85 else segs
        if "patrol" in mods:
            segs = on_busy(self._segments(floor, 3, 5))
            for _ in range(1 + (i >= 4)):
                segs = [s for s in segs if not used & set(s)]
                if not segs:
                    return self.why("seg")
                s = segs[int(rng.integers(len(segs)))]
                if rng.random() < 0.5:
                    s = s[::-1]
                path = s + s[-2:0:-1]                           # aller-retour
                pats.append(path)
                used |= set(s)
                P = lcm(P, len(path))
        if "rail" in mods:
            segs = on_busy([s for s in self._segments(floor, 4, 7) if not used & set(s)])
            if not segs:
                return self.why("rail")
            s = segs[int(rng.integers(len(segs)))]
            rails.append(s)
            used |= set(s)
            P = lcm(P, 2)
        lvl["los"] = {}
        if "sentry" in mods:
            cand = [p for p in fr if p not in used and p not in (f1, f2) and len(list(K.nbrs(floor, p))) >= 2]
            best = None
            for p in cand[:30]:
                tot = sum(len(self._los(lvl, p, d)) + 2 * len(busy & set(self._los(lvl, p, d))) for d in CW)
                if best is None or tot > best[0]:
                    best = (tot, p)
            if best is None or best[0] < 5:
                return self.why("sentry")
            sp = best[1]
            sents.append((sp, int(rng.integers(4))))
            used.add(sp)
            for f in range(4):
                lvl["los"][(sp, f)] = frozenset(self._los(lvl, sp, CW[f]))
            P = lcm(P, 8)
        lvl.update(pat_t=pats, sentries=sents, rails=rails, P=P)
        lvl["rail_tgt"] = [{c: min(range(len(rl)), key=lambda j: abs(rl[j][0] - c[0]) + abs(rl[j][1] - c[1]))
                            for c in fr} for rl in rails]
        rest = [p for p in fr if p not in used]
        ng = min(4, 1 + ("gem2" in mods) + (i >= 1) + (i >= 3))
        if len(rest) < ng + 8:
            return self.why("room")
        # départ : loin des PNJ, hors de leur portée au tour 0
        danger = set()
        for path in pats:
            danger |= set(path)
        for (sp, f0) in sents:
            danger |= set(lvl["los"][(sp, f0)]) | set(lvl["los"][(sp, (f0 + 1) % 4)])
        for rl in rails:
            danger |= set(rl)
        starts = [p for p in rest if p not in danger]
        if not starts:
            return self.why("start")
        p0 = f1 if f1 in starts and rng.random() < 0.7 else starts[0]
        mim = None
        if "mimic" in mods:
            mc = [p for p in rest if p != p0 and abs(p[0] - p0[0]) + abs(p[1] - p0[1]) >= 3]
            if not mc:
                return self.why("mimic")
            mim = (mc[0], None)
            used.add(mc[0])
        rest = [p for p in rest if p != p0 and (mim is None or p != mim[0])]
        dmap = K.grid_dist(floor, p0)
        gems = []                                               # gemmes dispersées (point le plus éloigné)
        pool = [p for p in rest if p not in danger or rng.random() < 0.3]
        while len(gems) < ng and pool:
            ds = [K.grid_dist(floor, q) for q in gems]
            p = max(pool, key=lambda p: min([dmap.get(p, 0)] + [d.get(p, 0) for d in ds]) + rng.random() * 2)
            gems.append(p)
            pool.remove(p)
        lvl["gems"] = tuple(gems)
        lvl["exit"] = None
        st0 = (p0, 0, 0, tuple(0 if rng.random() < 0.5 else len(rl) - 1 for rl in rails), mim)
        depth, prev = K.explore(self, lvl, st0, 20000)
        if depth is None:
            shr[i] = shr.get(i, 1.0) * 0.85
            return self.why("cap")
        full = (1 << ng) - 1
        best = {}
        bad = set(gems) | used | {p0}
        for s, dd in depth.items():
            if s[2] == full and s[0] not in bad:
                if s[0] not in best or dd < depth[best[s[0]]]:
                    best[s[0]] = s
        if not best:
            return self.why("noexit")
        # la règle nouvelle doit compter : la sortie choisie doit être plus loin qu'en l'absence de PNJ
        plain = self._plain(lvl, p0)
        items = list(best.items())
        for t in range(10):
            e, s = K.pick_far(rng, items, lambda x: depth[x[1]], 0.8 if t < 6 else 0.5)
            if newest in (None, "gem2") or depth[s] > plain.get(e, 0) + 1:
                break
        else:
            return self.why("npc_useless")
        lvl["exit"] = e
        sol = K.path_to(prev, s)
        return lvl, st0, sol

    def _plain(self, lvl, p0):
        """longueur du plus court chemin (gemmes puis case) sans aucun PNJ."""
        from collections import deque
        gems = lvl["gems"]
        full = (1 << len(gems)) - 1
        s0 = (p0, 0)
        d = {s0: 0}
        q = deque([s0])
        while q:
            p, m = q.popleft()
            for nb in K.nbrs(lvl["floor"], p):
                m2 = m
                for k, gp in enumerate(gems):
                    if gp == nb:
                        m2 |= 1 << k
                if (nb, m2) not in d:
                    d[(nb, m2)] = d[(p, m)] + 1
                    q.append((nb, m2))
        out = {}
        for (p, m), v in d.items():
            if m == full:
                out[p] = min(out.get(p, 99), v)
        return out

    # ================================================================= variante haul
    def _near(self, lvl, p, targets):
        dm = lvl["dist"]
        best, bd = None, 999
        for t in targets:
            v = dm.get((p, t), 999)
            if v < bd:
                best, bd = t, v
        return best

    def _toward(self, lvl, p, tgt, blocked):
        if tgt is None or tgt == p:
            return p
        dm = lvl["dist"]
        cur = dm.get((p, tgt), 999)
        for d in CW:
            q = (p[0] + d[0], p[1] + d[1])
            if q in lvl["npc_ok"] and q not in blocked and dm.get((q, tgt), 999) < cur:
                return q
        return p

    def _step_haul(self, lvl, st, op, px):
        p, carry, boxes, npcs, t = st
        tags = set()
        if op == "click":
            return st, tags, False
        boxes = set(boxes)
        npcs = list(npcs)
        if op == "interact":
            if carry and p in lvl["zone"] and p not in boxes:
                boxes.add(p)
                carry = 0
                tags.add("drop")
            elif not carry and p in boxes:
                boxes.discard(p)
                carry = 1
                tags.add("grab")
            else:
                tags.add("idle")
        else:
            d = DIRS[op]
            q = (p[0] + d[0], p[1] + d[1])
            if not (0 <= q[0] < lvl["Hc"] and 0 <= q[1] < lvl["Wc"]) or not lvl["floor"][q]:
                tags.add("bump")
                return st, tags, False
            hit = [k for k, (np_, c) in enumerate(npcs) if np_ == q]
            if hit:
                k = hit[0]
                if lvl["kinds"][k] != "thief":
                    tags.add("bump")
                    return st, tags, False
                np_, c = npcs[k]                                # chasser le voleur : on lui reprend sa caisse
                if not (c and not carry):
                    tags.add("bump")
                    return st, tags, False
                carry = 1
                npcs[k] = (lvl["home"][k], 0)                   # et il rentre dans sa tanière
                tags.add("scare")
            else:
                p = q
        # tour des PNJ
        t2 = (t + 1) % 2 if "slow" in lvl["mods"] else 0
        if not ("slow" in lvl["mods"] and t == 1):
            for k, (np_, c) in enumerate(npcs):
                kind = lvl["kinds"][k]
                dest = lvl["zone"] if kind == "helper" else lvl["den"]
                blocked = {p} | {x[0] for j, x in enumerate(npcs) if j != k}
                if c:
                    if np_ in dest and np_ not in boxes:
                        boxes.add(np_)
                        npcs[k] = (np_, 0)
                        tags.add("npc_drop")
                        continue
                    free = [z for z in dest if z not in boxes]
                    npcs[k] = (self._toward(lvl, np_, self._near(lvl, np_, free), blocked), 1)
                else:
                    if kind == "helper":                        # aide : caisses libres -> zone
                        want = [b for b in boxes if b not in lvl["zone"] and b not in lvl["den"]]
                    else:                                       # voleur : caisses de la zone -> tanière
                        want = [b for b in boxes if b in lvl["zone"]]
                    want = [b for b in want if b in lvl["npc_ok"]]
                    if np_ in want:
                        boxes.discard(np_)
                        npcs[k] = (np_, 1)
                        tags.add("npc_grab")
                        continue
                    tgt = self._near(lvl, np_, want) if want else lvl["home"][k]
                    npcs[k] = (self._toward(lvl, np_, tgt, blocked), 0)
        return (p, carry, tuple(sorted(boxes)), tuple(npcs), t2), tags, False

    def _gen_haul(self, rng, i, n, mods, newest):
        fac = dict(helper=0.8, thief=0.55, fence=1.0, slow=0.75)
        maxc = 46.0 + 4 * min(i, 6)
        for m in mods:
            maxc *= fac.get(m, 1.0)
        shr = self.__dict__.setdefault("_shr", {})
        maxc = max(22, int(maxc * shr.get(i, 1.0)))
        floor, geo = self._board(rng, maxc, i, n)
        if floor is None:
            return self.why("floor")
        lvl = dict(floor=floor, mods=mods, **geo)
        fr = K.cells(floor)
        rng.shuffle(fr)
        # zone : la case la plus « coin » et ses voisines
        z0 = fr[0]
        zd = K.grid_dist(floor, z0)
        zone = sorted(zd, key=lambda p: zd[p])[:min(4, 2 + (i >= 1) + (i >= 3))]
        lvl["zone"] = frozenset(zone)
        far = sorted(fr, key=lambda p: -zd.get(p, 0))
        used = set(zone)
        den = []
        if "thief" in mods:
            den = [p for p in far[:6] if p not in used][:2]
            used |= set(den)
        lvl["den"] = frozenset(den)
        fence = set()
        if "fence" in mods:
            ww = K.wall_with_gap(rng, floor, gap=(1, 2), min_part=5)
            if ww is None:
                return self.why("fence")
            _, run, A, B = ww
            fence = set(run) | {p for p in K.cells(floor) if not ww[0][p]}  # le mur entier devient clôture
            if zone[0] in fence or set(den) & fence:
                return self.why("fence_zone")
        lvl["fence"] = frozenset(fence)
        ok = frozenset(p for p in fr if p not in fence)
        lvl["npc_ok"] = ok
        rest = [p for p in fr if p not in used and p not in fence]
        nbx = min(4, 2 + (i >= 1) + (i >= 3))
        mid = [p for p in rest if 2 <= zd.get(p, 0)]
        if len(mid) < nbx + 4:
            return self.why("room")
        mid.sort(key=lambda p: -zd.get(p, 0) - 4 * rng.random())   # caisses loin de la zone
        boxes = mid[:nbx]
        rest = [p for p in rest if p not in boxes]
        kinds, npcs, home = [], [], []
        if "helper" in mods:
            kinds.append("helper")
            hp = [p for p in rest if p in ok][0]
            npcs.append((hp, 0))
            home.append(hp)
            rest.remove(hp)
        if "thief" in mods:
            kinds.append("thief")
            npcs.append((den[0], 0))
            home.append(den[0])
        lvl["kinds"], lvl["home"] = tuple(kinds), tuple(home)
        # distances PNJ (sol sans clôture), toutes paires
        dist = {}
        for a in ok:
            for b, v in K.grid_dist(floor, a, block=frozenset(fence)).items():
                dist[(a, b)] = v
        lvl["dist"] = dist
        starts = [p for p in rest if p not in fence]
        if not starts:
            return self.why("start")
        p0 = starts[0]
        st0 = (p0, 0, tuple(sorted(boxes)), tuple(npcs), 0)
        if self.is_win(lvl, st0):
            return self.why("won")
        nf = self.__dict__.setdefault("_big", {})
        if nf.get(i, 0) < 6:                                    # préférer des niveaux à table exacte
            depth, _ = K.explore(self, lvl, st0, self.table_cap)
            if depth is None:
                nf[i] = nf.get(i, 0) + 1
                shr[i] = shr.get(i, 1.0) * 0.88
                return self.why("big")
        sol = self.solve(lvl, st0, 30000)
        if sol is None:
            return self.why("nosol")
        if newest in ("helper", "thief"):                     # le PNJ nouveau doit agir pendant la solution
            tg = K.trace_tags(self, lvl, st0, sol)
            if not ({"npc_grab", "scare"} & tg):
                return self.why("npc_idle")
        return lvl, st0, sol

    # ================================================================= commun
    def _step(self, lvl, st, op, px):
        if self.variant == "dodge":
            st2, tags, dead = self._step_dodge(lvl, st, op, px)
        else:
            st2, tags, dead = self._step_haul(lvl, st, op, px)
        tags = set(tags)
        if dead:
            tags.add("dead")
        return st2, tags

    def apply(self, lvl, st, op, px):
        if self.variant == "dodge":
            st2, _, dead = self._step_dodge(lvl, st, op, px)
        else:
            st2, _, dead = self._step_haul(lvl, st, op, px)
        return st2, dead

    def is_win(self, lvl, st):
        if self.variant == "dodge":
            return lvl["exit"] is not None and st[0] == lvl["exit"] and st[2] == (1 << len(lvl["gems"])) - 1
        return not st[1] and all(b in lvl["zone"] for b in st[2]) and not any(c for _, c in st[3]) \
            and len(st[2]) == lvl.get("nbx", len(st[2]))

    def gen_level(self, rng, i, n):
        if self.variant == "dodge":
            mods, newest = mods_for(self.cat, i, keep=2)
            return self._gen_dodge(rng, i, n, mods, newest)
        mods, newest = mods_for(self.cat, i, keep=2)
        mods = K.prune(mods, ({"thief", "slow"},))
        if i >= 1 and "helper" not in mods and "thief" not in mods:
            mods = ("helper",) + mods                           # « slow » / « fence » agissent sur un PNJ
        out = self._gen_haul(rng, i, n, mods, newest)
        if out is not None:
            out[0]["nbx"] = len(out[1][2])
        return out

    def min_len(self, i, n):
        return min(14 + 5 * i, 46)

    def heuristic(self, lvl, st):
        p = st[0]
        if self.variant == "dodge":
            todo = [g for k, g in enumerate(lvl["gems"]) if not st[2] >> k & 1] + [lvl["exit"]]
            return sum(abs(p[0] - g[0]) + abs(p[1] - g[1]) for g in todo if g is not None)
        return 4 * sum(b not in lvl["zone"] for b in st[2]) + 4 * st[1]

    def body(self, lvl, st):
        return (st[0][0], st[0][1], lvl["Hc"], lvl["Wc"])

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        _, tags = self._step(lvl, st, op, px)
        if "bump" in tags or "hit" in tags or "scare" in tags:
            ev[EV["touch"]] = 1
        if "gem" in tags or "grab" in tags or "drop" in tags:
            ev[EV["collect"]] = 1
        if "exit" in tags:
            ev[EV["open"]] = 1
        if "hit" in tags or "seen" in tags:
            ev[EV["danger"]] = 1
        if "scare" in tags:
            ev[EV["teleport"]] = ev[EV["push"]] = 1
        if "npc_grab" in tags or "npc_drop" in tags or "turn" in tags:
            ev[EV["toggle"]] = 1
        if self.variant == "dodge" and st2[2] != st[2] and st2[2] == (1 << len(lvl["gems"])) - 1:
            ev[EV["color"]] = 1                                  # la sortie s'allume

    # ---------------------------------------------------------------- rendu
    def render(self, lvl, st):
        pal = self.pal
        C = lvl["C"]
        g = self.blank()
        floor = lvl["floor"]
        K.draw_walls(g, lvl, pal.wall, 2 if C >= 5 else 1)
        if self.variant == "dodge":
            self._render_dodge(g, lvl, st, C)
        else:
            self._render_haul(g, lvl, st, C)
        y, x = K.cell_xy(lvl, st[0])
        blit(g, draw.recolor(self.tpl[C], {100: pal.a(0), 101: pal.fg, 102: pal.fg}), y, x)
        if self.variant == "haul" and st[1]:
            rect(g, y + C // 2 - 1, x + C // 2 - 1, 2 if C >= 5 else 1, 2 if C >= 5 else 1, pal.a(1))
        self._hud(g, lvl, st)
        return g

    def _mid(self, lvl, p):
        C = lvl["C"]
        return lvl["oy"] + p[0] * C + C // 2, lvl["ox"] + p[1] * C + C // 2

    def _render_dodge(self, g, lvl, st, C):
        pal = self.pal
        p, t, gm, rails, mim = st
        full = (1 << len(lvl["gems"])) - 1
        e = lvl["exit"]
        if e is not None:
            y, x = K.cell_xy(lvl, e)
            on = gm == full
            blit(g, sprite(None, "framed" if on else "frame", C, C, pal.a(4) if on else pal.room, pal.fg if on else pal.wall), y, x)
            if not on:
                frame_rect(g, y, x, C, C, pal.line)
        for k, gp in enumerate(lvl["gems"]):
            if not gm >> k & 1:
                y, x = K.cell_xy(lvl, gp, 1)
                blit(g, sprite(None, "diamond", C - 2, C - 2, pal.a(5)), y, x)
        cpat, csen, crail, cmim = pal.a(1), pal.a(2), pal.a(3), pal.a(6)
        for path in lvl["pat_t"]:                               # voie (trait fin)
            ends = (path[0], path[len(path) // 2])
            (y0, x0), (y1, x1) = self._mid(lvl, ends[0]), self._mid(lvl, ends[1])
            draw.line(g, y0, x0, y1, x1, pal.line)
        for rl in lvl["rails"]:
            (y0, x0), (y1, x1) = self._mid(lvl, rl[0]), self._mid(lvl, rl[-1])
            draw.line(g, y0, x0, y1, x1, crail)
        for (sp, f0) in lvl["sentries"]:
            f = (f0 + t // 2) % 4
            los = lvl["los"][(sp, f)]
            if los:                                             # regard : trait plein jusqu'au mur
                (y0, x0), (y1, x1) = self._mid(lvl, sp), self._mid(lvl, los[-1] if isinstance(los, list) else
                                                                 max(los, key=lambda q: abs(q[0] - sp[0]) + abs(q[1] - sp[1])))
                draw.line(g, y0, x0, y1, x1, csen)
        for path in lvl["pat_t"]:
            q = path[t % len(path)]
            y, x = K.cell_xy(lvl, q, 1 if C >= 5 else 0)
            S = C - 2 if C >= 5 else C
            blit(g, sprite(None, "box", S, S, cpat, None, pal.fg if S >= 3 else None), y, x)
        for k, j in enumerate(rails):
            y, x = K.cell_xy(lvl, lvl["rails"][k][j])
            blit(g, sprite(None, "hat", C, C, crail), y, x)
            g[y + C // 2, x + C // 2] = R(pal.fg)
        for (sp, f0) in lvl["sentries"]:
            f = (f0 + t // 2) % 4
            y, x = K.cell_xy(lvl, sp)
            blit(g, sprite(None, "diamond", C, C, csen), y, x)
            d = CW[f]
            g[y + C // 2 + d[0] * (C // 2), x + C // 2 + d[1] * (C // 2)] = R(pal.fg)
        if mim is not None:
            y, x = K.cell_xy(lvl, mim[0])
            m = draw.recolor(self.tpl[C], {100: -1, 101: cmim, 102: -1})
            if (m >= 0).sum() < 4:
                m = sprite(None, "frame", C, C, cmim)
            blit(g, m, y, x)

    def _render_haul(self, g, lvl, st, C):
        pal = self.pal
        p, carry, boxes, npcs, t = st
        for q in lvl["fence"]:
            y, x = K.cell_xy(lvl, q)
            blit(g, sprite(None, "plus", C, C, pal.wall), y, x)
        for q in lvl["zone"]:
            y, x = K.cell_xy(lvl, q)
            frame_rect(g, y, x, C, C, pal.a(4))
        for q in lvl["den"]:
            y, x = K.cell_xy(lvl, q)
            blit(g, sprite(None, "frame", C, C, pal.a(2)), y, x)
        bc = pal.a(1)
        for b in boxes:
            y, x = K.cell_xy(lvl, b, 1)
            blit(g, sprite(None, "framed", C - 2, C - 2, bc, pal.fg), y, x)
        for k, (q, c) in enumerate(npcs):
            y, x = K.cell_xy(lvl, q)
            col = pal.a(3) if lvl["kinds"][k] == "helper" else pal.a(2)
            blit(g, sprite(None, "ring" if C >= 5 else "box", C, C, col), y, x)
            if c:
                rect(g, y + C // 2 - 1, x + C // 2 - 1, 2, 2, bc)

    def _hud(self, g, lvl, st):
        pal = self.pal
        h = self.hh - 1
        y0 = 0 if self.hud_top else self.AH - h
        rect(g, y0, 0, h, self.AW, pal.ui)
        a = h - 2
        x = 2
        if self.variant == "dodge":
            for k in range(len(lvl["gems"])):
                if st[2] >> k & 1:
                    rect(g, y0 + 1, x, a, a, pal.a(5))
                else:
                    frame_rect(g, y0 + 1, x, a, a, pal.a(5))
                x += a + 2
            full = st[2] == (1 << len(lvl["gems"])) - 1
            frame_rect(g, y0 + 1, x + 2, a, a, pal.a(4) if full else pal.line)
        else:
            n_in = sum(b in lvl["zone"] for b in st[2])
            for k in range(lvl.get("nbx", len(st[2]))):
                if k < n_in:
                    rect(g, y0 + 1, x, a, a, pal.a(1))
                else:
                    frame_rect(g, y0 + 1, x, a, a, pal.a(1))
                x += a + 2
            frame_rect(g, y0 + 1, x + 2, a, a, pal.a(4))
