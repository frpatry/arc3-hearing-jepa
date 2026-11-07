"""twins4 (objets, corps couplés) : UNE commande déplace 2 (puis 3) corps, chacun sous une transformation fixe de la
direction (même sens, miroir gauche-droite, miroir haut-bas, inverse, rotation d'un quart de tour). Chaque corps est
bloqué séparément par les murs : les murs DÉSYNCHRONISENT les corps. La légende du bandeau montre, pour chaque corps,
où il va quand on appuie sur « droite » et sur « haut ». Les murs sont teintés en deux tons (une moitié par corps).
Variantes (tirées par jeu) :
  meet : les corps doivent se REJOINDRE sur la balise (tous dessus en même temps), puis sur la suivante ;
  pads : chaque corps doit atteindre le cadre de sa couleur (puis la série de cadres suivante) ; 1 à 3 étapes.
Les corps se bloquent entre eux (sauf sur la balise).
Règles ajoutées, une par niveau (ordre tiré par jeu), puis combinées :
  hazard : cases à damier : un corps qui y entre fait tout recommencer (positions de départ ; pas de fin de partie) ;
  block  : une caisse sur rail (trait) que les corps poussent le long du rail (elle bloque aussi) ;
  door   : porte colorée ouverte tant qu'un corps (ou le bloc) tient la plaque de sa couleur ;
  lazy   : un corps ne bouge qu'un tour sur deux (creux quand il se repose) ;
  freeze : ACTION5 (ou un clic sur un corps) GÈLE un corps (blanc) : il ne bouge plus jusqu'au dégel ;
  ice    : plaque de glace : un corps qui y entre glisse jusqu'à en sortir ou buter ;
  trio   : un 3e corps avec sa propre transformation (toujours la dernière règle).
Solvabilité : départ = état le plus éloigné du but dans la fermeture d'états (distance exacte, avkit.hardest)."""
import numpy as np

from ..core import DIRS, DIR_OPS, EV
from . import avkit as K
from . import draw
from .base import V4Game, mods_for
from .draw import blit, frame_rect, rect, sprite

TF = {"same": lambda d: d, "mirror": lambda d: (d[0], -d[1]), "flip": lambda d: (-d[0], d[1]),
      "inverse": lambda d: (-d[0], -d[1]), "rot": lambda d: (d[1], -d[0])}


class Twins4(V4Game):
    family = "twins4"
    mechanic = "object"
    principle = ("corps couplés : une commande déplace 2-3 corps sous des transformations fixes (miroir, inverse, "
                 "rotation) ; les murs les désynchronisent ; se rejoindre ou atteindre chacun son cadre",
                 ["m0r0", "g50t", "r11l", "s5i5"])
    simple_ops = DIR_OPS + ("interact",)
    CAT = ("hazard", "block", "door", "lazy", "freeze", "ice", "trio")
    free_miss = True
    exact_progress = True
    table_cap = 25000
    scale_p = (1.0, 0.0)                    # plateaux larges : échelle ×1
    bfs_cap = 40000

    def setup(self, rng):
        r = self.orng
        self.variant = str(rng.choice(["meet", "pads"]))
        self.uses_click = bool(rng.random() < 0.6)
        cat = [str(x) for x in rng.permutation(self.CAT)]
        cat.remove("trio")                                      # bloc et 3e corps tard (états nombreux, sols petits)
        cat.remove("block")
        cat.insert(int(rng.integers(3, 5)), "block")
        cat.append("trio")
        self.cat = cat
        t1 = str(rng.choice(["mirror", "mirror", "mirror", "inverse", "inverse", "flip", "rot", "same"]))
        t2 = str(rng.choice([t for t in ("mirror", "inverse", "flip", "rot") if t != t1]))
        self.tfs = ("same", t1, t2)
        self.C = int(r.choice([4, 4, 5])) if self.s_screen == 1 else 3
        self.hh = 7 if self.s_screen == 1 else 5
        self.style = str(r.choice(["block", "ring"]))
        self.body_kind = str(r.choice(["blob", "framed", "disc", "diamond"]))
        self.tpl_body = {S: sprite(r, self.body_kind, S, S, 100, 101, 102 if S >= 5 else None)
                         for S in range(3, self.C + 5)}
        self.floor_style = str(r.choice(["rooms", "rooms", "cave"]))

    # ---------------------------------------------------------------- règles
    def _open_doors(self, lvl, occ):
        return {dc for dc, col in lvl["doors"].items()
                if dc in occ or any(p in occ for p, c in lvl["plates"].items() if c == col)}

    def _free(self, lvl, q, opened):
        if not (0 <= q[0] < lvl["Hc"] and 0 <= q[1] < lvl["Wc"]) or not lvl["floor"][q]:
            return False
        return q not in lvl["doors"] or q in opened

    def _step(self, lvl, st, op, px):
        bodies, block, frz, par, stg = st
        mods = lvl["mods"]
        goals = lvl["goals"]
        tags = set()
        if op in ("click", "interact"):
            if "freeze" not in mods:
                return st, tags
            ids = [b[2] for b in bodies]
            if op == "interact":
                order = ids + [-1]
                nf = order[(order.index(frz) + 1) % len(order)] if frz in order else -1
            else:
                q = K.cell_of(lvl, px)
                hit = [b[2] for b in bodies if (b[0], b[1]) == q]
                if not hit:
                    return st, tags
                nf = -1 if frz == hit[0] else hit[0]
            tags.add("freeze")
            return (bodies, block, nf, par, stg), tags
        d = DIRS[op]
        pos = [(b[0], b[1]) for b in bodies]
        occ = set(pos) | ({block} if block else set())
        opened = self._open_doors(lvl, occ)
        new = list(pos)
        nblock = block
        pads = self.variant == "pads"
        for j, (r, c, k) in enumerate(bodies):
            if k == frz or ("lazy" in mods and k == lvl["lazy"] and par == 1):
                continue
            dd = TF[self.tfs[k]](d)
            q = (r + dd[0], c + dd[1])
            others = set(new[:j]) | set(pos[j + 1:])
            shared = self.variant == "meet" and stg < len(goals) and q == goals[stg][k]
            if not self._free(lvl, q, opened) or (q in others and not shared):
                tags.add("bump")
                continue
            if q == nblock:
                b2 = (q[0] + dd[0], q[1] + dd[1])
                if not self._free(lvl, b2, opened) or b2 in others or b2 in lvl["hazard"] or b2 not in lvl["rail"]:
                    tags.add("bump")
                    continue
                nblock = b2
                tags.add("push")
            while q in lvl["ice"]:                              # glace : on glisse jusqu'à sortir ou buter
                q2 = (q[0] + dd[0], q[1] + dd[1])
                if not self._free(lvl, q2, opened) or q2 in others or q2 == nblock:
                    break
                q = q2
                tags.add("slide")
            new[j] = q
            if q in lvl["doors"]:
                tags.add("door")
        npar = 1 - par if "lazy" in mods else 0
        if any(p in lvl["hazard"] for p in new):
            tags.add("hazard")
            return (lvl["reset"] if lvl["reset"] is not None else st), tags
        nb = tuple((p[0], p[1], b[2]) for p, b in zip(new, bodies))
        if stg < len(goals) and all((b[0], b[1]) == goals[stg][b[2]] for b in nb):
            stg += 1                                            # étape franchie (rendez-vous / cadres)
            tags.add("stage")
            if self.variant == "meet":
                tags.add("merge")
        return (nb, nblock, frz, npar, stg), tags

    def apply(self, lvl, st, op, px):
        return self._step(lvl, st, op, px)[0], False

    def is_win(self, lvl, st):
        return st[4] == len(lvl["goals"])

    def click_targets(self, lvl, st):
        if "freeze" not in lvl["mods"]:
            return []
        return [K.cell_center(lvl, (b[0], b[1])) for b in st[0]]

    def click_hit(self, lvl, st, px):
        q = K.cell_of(lvl, px)
        return q is not None and (q in {(b[0], b[1]) for b in st[0]} or q == st[1])

    # ---------------------------------------------------------------- niveaux
    def gen_level(self, rng, i, n):
        mods, newest = mods_for(self.cat, i)
        bad_pairs = ({"freeze", "block"}, {"freeze", "trio"}, {"block", "trio"})   # combinaisons trop coûteuses :
        keep = []                                                                  # la règle la plus ancienne part
        for m in reversed(mods):
            if all({m, k} not in bad_pairs for k in keep):
                keep.append(m)
        mods = tuple(reversed(keep))
        light = not ({"block", "trio"} & set(mods))
        self.__dict__.setdefault("_heavy", {})[i] = not light
        nstage = 1 if "trio" in mods else 1 + (i >= 1) + (i >= 3 and light)
        # taille du sol fixée par le budget d'états : cases^(corps + bloc) × gel × parité × étapes <= ~15000
        dims = 2.8 if "trio" in mods else 2
        shr = self.__dict__.setdefault("_shr", {})             # réduit tout seul si la fermeture déborde
        while True:
            mult = (3 if "freeze" in mods else 1) * (2 if "lazy" in mods else 1) * nstage * (5 if "block" in mods else 1)
            maxc = int(min(66 + 6 * min(i, 5), (15000 / mult) ** (1 / dims)) * shr.get(i, 1.0))
            if maxc >= 30 or nstage == 1:
                break
            nstage -= 1                                         # sol trop petit : moins d'étapes
        maxc = max(18, maxc)
        C, Hc, Wc, oy, ox = K.board(self.AH, self.AW, self.C, maxc / 0.45 + 8, top=self.hh, grow=3)
        d = i / max(1, n - 1)
        side = max(4, int(round(min(Hc, Wc) * (0.55 + 0.35 * d))))
        floor = K.make_floor(rng, Hc, Wc, self.floor_style, n_rooms=3 + (i >= 3), need=min(maxc - 6, 30 + 6 * i),
                             holes=float(rng.uniform(0.08, 0.16)) + 0.01 * min(i, 5), max_side=side, maxc=maxc)
        if floor is None:
            return self.why("floor")
        nb = 3 if "trio" in mods else 2
        fr = K.cells(floor)
        rng.shuffle(fr)
        lvl = dict(floor=floor, Hc=Hc, Wc=Wc, oy=oy, ox=ox, C=C, mods=mods, hazard=frozenset(), doors={}, plates={},
                   goals=(), lazy=nb - 1, reset=None, block0=None, ice=frozenset(),
                   rail=frozenset())
        used = set()
        if "door" in mods:                                      # porte = segment qui coupe le sol en deux
            cuts = K.cut_runs(floor, max_len=2, min_part=5) if rng.random() < 0.5 else []
            if cuts:
                run, A, B = cuts[int(rng.integers(len(cuts)))]
            else:
                ww = K.wall_with_gap(rng, floor, min_part=4)
                if ww is None:
                    return self.why("cut")
                floor, run, A, B = ww
                lvl["floor"] = floor
                fr = [p for p in fr if floor[p]]
            pl = [p for p in fr if p not in run and min(abs(p[0] - q[0]) + abs(p[1] - q[1]) for q in run) >= 2]
            if not pl:
                return self.why("plate")
            lvl["doors"] = {q: 0 for q in run}
            lvl["plates"] = {pl[0]: 0}
            side = A if pl[0] in A else B                       # buts du même côté que la plaque
            used |= set(run) | {pl[0]}
        rest = [p for p in fr if p not in used]
        if "hazard" in mods:                                    # cases dangereuses qui ne coupent pas le sol
            want = 2 + (i >= 4) + int(rng.integers(0, 2))
            f2, hz = floor.copy(), set()
            for p in rest:
                if len(hz) >= want:
                    break
                if len(list(K.nbrs(f2, p))) < 2:
                    continue
                f2[p] = False
                left = K.cells(f2)
                if len(K.grid_dist(f2, left[0])) == len(left):
                    hz.add(p)
                else:
                    f2[p] = True
            if not hz:
                return self.why("hazard")
            lvl["hazard"] = frozenset(hz)
            used |= hz
            rest = [p for p in fr if p not in used]
        if "ice" in mods:                                       # plaque de glace (rectangle de sol)
            best = None
            for p in rest[:30]:
                h_, w_ = int(rng.integers(2, 4)), int(rng.integers(2, 5))
                if rng.random() < 0.5:
                    h_, w_ = w_, h_
                cl = {(p[0] + a, p[1] + b) for a in range(h_) for b in range(w_)}
                cl = {q for q in cl if 0 <= q[0] < Hc and 0 <= q[1] < Wc and floor[q] and q not in used}
                if best is None or len(cl) > len(best):
                    best = cl
            if not best or len(best) < 3:
                return self.why("ice")
            lvl["ice"] = frozenset(best)
            used |= best
            rest = [p for p in fr if p not in used]
        if "block" in mods:                                     # caisse sur rail (ligne ou colonne de 4-6 cases)
            rails = []
            for p in rest:
                for d in ((0, 1), (1, 0)):
                    seg, y, x = [], p[0], p[1]
                    while 0 <= y < Hc and 0 <= x < Wc and floor[y, x] and (y, x) not in used and len(seg) < 6:
                        seg.append((y, x))
                        y, x = y + d[0], x + d[1]
                    if len(seg) >= 4:
                        rails.append(seg)
            if not rails:
                return self.why("block")
            seg = rails[int(rng.integers(len(rails)))]
            lvl["rail"] = frozenset(seg)
            lvl["block0"] = seg[int(rng.integers(1, len(seg) - 1))]
            used.add(lvl["block0"])
            rest = [p for p in fr if p not in used]
        goals = []
        far_from = []
        if "door" not in mods:
            side = None
        for k in range(nstage):                                 # étapes : balises / cadres éloignés les uns des autres
            dists = [K.grid_dist(floor, q) for q in far_from]
            pool = [p for p in rest if (self.variant == "pads" or len(list(K.nbrs(floor, p))) >= 3)
                    and (side is None or p in side)] or [p for p in rest if side is None or p in side]
            if len(pool) < nb + 3:
                return self.why("goals")
            key = (lambda p: min([d.get(p, 0) for d in dists]) + 3 * rng.random()) if dists else (lambda p: rng.random())
            pool.sort(key=lambda p: -key(p))
            if self.variant == "meet":
                goals.append({kk: pool[0] for kk in range(nb)})
                far_from.append(pool[0])
                used.add(pool[0])
            else:
                g_ = {kk: pool[kk] for kk in range(nb)}
                goals.append(g_)
                far_from += list(g_.values())
                used |= set(g_.values())
            rest = [p for p in fr if p not in used]
        lvl["goals"] = tuple(goals)
        if len(rest) < 8:
            return self.why("room")
        allpairs = nb == 2 and lvl["block0"] is None and len(rest) ** 2 * nstage <= 12000
        for nseed in ((0, 24, 6) if allpairs else (24, 6) if nb == 2 else (8, 2)):
            seeds = []
            if nseed == 0:                                      # toutes les paires : le départ le plus dur global
                seeds = [(((a[0], a[1], 0), (b[0], b[1], 1)), None, -1, 0, 0) for a in rest for b in rest if a != b]
            for _ in range(nseed):
                ix = rng.permutation(len(rest))[:nb]
                seeds.append((tuple((rest[j][0], rest[j][1], k) for k, j in enumerate(ix)), lvl["block0"], -1, 0, 0))
            dist, succ = K.closure_dist(self, lvl, seeds, 18000)
            if dist is not None:
                break
        else:                                                   # trop d'états : plus petit au prochain essai
            shr[i] = shr.get(i, 1.0) * 0.85
            return self.why("cap")
        bad = set(lvl["hazard"]) | set(lvl["plates"]) | set(lvl["doors"]) | {q for g_ in goals for q in g_.values()}
        cand = [s for s, v in dist.items() if v > 0 and s[4] == 0 and s[2] == -1 and s[3] == 0
                and not any((b[0], b[1]) in bad for b in s[0]) and s[1] not in lvl["plates"]]
        if not cand:
            return self.why("nocand")
        need = {"block": "push", "door": "door", "freeze": "freeze", "hazard": None, "lazy": None, "trio": None,
                "ice": "slide"}
        tag = need.get(newest)
        cand.sort(key=lambda s: -dist[s])
        top = cand[:max(1, len(cand) // 5)]
        for t in range(12):
            st0 = top[int(rng.integers(len(top)))] if t < 8 else cand[int(rng.integers(len(cand)))]
            sol = K.descend(dist, succ, st0)
            if sol is None:
                continue
            if tag is None or tag in K.trace_tags(self, lvl, st0, sol):
                break
        else:
            return self.why("unused_rule")
        lvl["reset"] = st0
        return lvl, st0, sol

    def min_len(self, i, n):
        if self.__dict__.get("_heavy", {}).get(i):            # bloc / 3e corps : petits sols
            return 14
        return min(12 + 7 * i, 36)

    def heuristic(self, lvl, st):
        b, stg, G = st[0], st[4], lvl["goals"]
        if stg >= len(G):
            return 0
        return sum(abs(x[0] - G[stg][x[2]][0]) + abs(x[1] - G[stg][x[2]][1]) for x in b) + 10 * (len(G) - stg - 1)

    def body(self, lvl, st):
        b = st[0][0]
        return (b[0], b[1], lvl["Hc"], lvl["Wc"])

    def n_ok(self, lvl, st):
        G, stg = lvl["goals"], st[4]
        if stg >= len(G):
            return 100 * stg
        return 100 * stg + sum((b[0], b[1]) == G[stg][b[2]] for b in st[0])

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        _, tags = self._step(lvl, st, op, px)
        if "push" in tags:
            ev[EV["push"]] = ev[EV["touch"]] = 1
        if "bump" in tags:
            ev[EV["touch"]] = 1
        if "freeze" in tags:
            ev[EV["toggle"]] = 1
        if "hazard" in tags:
            ev[EV["danger"]] = ev[EV["teleport"]] = 1
        if "merge" in tags or self.n_ok(lvl, st2) > self.n_ok(lvl, st):
            ev[EV["collect"]] = 1
        o0 = self._open_doors(lvl, {(b[0], b[1]) for b in st[0]} | ({st[1]} if st[1] else set()))
        o1 = self._open_doors(lvl, {(b[0], b[1]) for b in st2[0]} | ({st2[1]} if st2[1] else set()))
        if o0 != o1:
            ev[EV["open"]] = 1

    # ---------------------------------------------------------------- rendu
    def _cols(self):
        pal = self.pal
        return [pal.a(0), pal.a(1), pal.a(2)]

    def render(self, lvl, st):
        pal, C = self.pal, lvl["C"]
        g = self.blank()
        Hc, Wc, oy, ox = lvl["Hc"], lvl["Wc"], lvl["oy"], lvl["ox"]
        floor = lvl["floor"]
        cols = self._cols()
        wl, wr = cols[0], cols[1]
        fl = pal.bg
        mid = ox + Wc * C // 2
        K.draw_walls(g, lvl, lambda y, x: wl if x < mid else wr, (2 if self.style == "ring" else 3) if C >= 5 else 1)
        icec = draw.LBLUE if draw.LBLUE not in (fl, wl, wr) else draw.WHITE if draw.WHITE != fl else draw.LGREY
        for p in lvl["ice"]:
            y, x = K.cell_xy(lvl, p)
            rect(g, y, x, C, C, icec)
        hzc = draw.RED if draw.RED not in (fl, wl, wr) else pal.a(5)
        for p in lvl["hazard"]:
            y, x = K.cell_xy(lvl, p)
            rect(g, y + 1, x + 1, C - 2, C - 2, hzc)
        dcol = pal.a(3)
        occ = {(b[0], b[1]) for b in st[0]} | ({st[1]} if st[1] else set())
        opened = self._open_doors(lvl, occ)
        for p in lvl["plates"]:
            y, x = K.cell_xy(lvl, p)
            blit(g, sprite(None, "plus_outline" if C >= 5 else "plus", C, C, dcol), y, x)
        for p in lvl["doors"]:
            y, x = K.cell_xy(lvl, p)
            if p in opened:
                frame_rect(g, y, x, C, C, dcol)
            else:
                blit(g, sprite(None, "framed", C, C, dcol, pal.fg), y, x)
        stg = st[4]
        for j in range(len(lvl["goals"]) - 1, stg - 1, -1):    # étapes : la courante en entier, les suivantes en petit
            gl = lvl["goals"][j]
            cur = j == stg
            if self.variant == "meet":
                y, x = K.cell_xy(lvl, gl[0])
                if cur:
                    ring = sprite(None, "ring" if C >= 5 else "frame", C, C, wl)
                    ring[:, C // 2:][ring[:, C // 2:] >= 0] = wr
                    blit(g, ring, y, x)
                else:
                    frame_rect(g, y + 1, x + 1, C - 2, C - 2, pal.line)
            else:
                for k, p in gl.items():
                    y, x = K.cell_xy(lvl, p)
                    if cur:
                        frame_rect(g, y, x, C, C, cols[k])
                        if C >= 6:
                            frame_rect(g, y + 2, x + 2, C - 4, C - 4, cols[k])
                    else:
                        rect(g, y + C // 2 - 1, x + C // 2 - 1, 2, 2, cols[k])
        if lvl["rail"]:
            rl = sorted(lvl["rail"])
            (y0, x0), (y1, x1) = K.cell_center(lvl, rl[0]), K.cell_center(lvl, rl[-1])
            draw.line(g, y0, x0, y1, x1, pal.line)
        if st[1] is not None:
            y, x = K.cell_xy(lvl, st[1])
            blit(g, sprite(None, "framed", C, C, pal.wall if pal.wall != fl else draw.GREY, pal.fg), y, x)
        for (r, c, k) in st[0]:
            y, x = K.cell_xy(lvl, (r, c))
            col = cols[k]
            if k == st[2]:
                col = draw.WHITE if fl != draw.WHITE else draw.LGREY
            spr = draw.recolor(self.tpl_body[C], {100: col, 101: pal.fg, 102: pal.fg})
            if "lazy" in lvl["mods"] and k == lvl["lazy"] and st[3] == 1:
                spr = sprite(None, "frame", C, C, col)               # au repos ce tour-ci : creux
            blit(g, spr, y, x)
        self._hud(g, lvl, st)
        return g

    def _hud(self, g, lvl, st):
        pal = self.pal
        h = self.hh - 1
        rect(g, 0, 0, h, self.AW, pal.ui)
        cols = self._cols()
        n = 3 if "trio" in lvl["mods"] else 2
        a = 5 if self.s_screen == 1 else 3
        x = 1
        alive = {b[2] for b in st[0]}
        stg, G = st[4], len(lvl["goals"])
        for k in range(n):
            col = cols[k] if k in alive else pal.wall
            sq = a - 1
            rect(g, 1, x, sq, sq, col)
            if k == st[2]:
                frame_rect(g, 0, x - 1 if x > 0 else 0, sq + 2, sq + 2, draw.WHITE)
            x += sq + 1
            for dvec in ((0, 1), (-1, 0)):
                K.put_mask(g, K.arrow_glyph(TF[self.tfs[k]](dvec), a), 1 if a == 5 else 1, x, pal.fg if pal.ui != pal.fg else draw.WHITE)
                x += a + 1
            x += 2 if a == 5 else 1
        # cible : nombre de corps restants (meet) ou cadres atteints (pads)
        xr = self.AW - 2
        for j in range(G):                                       # étapes franchies / restantes
            xx = xr - (j + 1) * a
            if j < stg:
                rect(g, 1, xx, a - 1, a - 1, pal.line)
            else:
                frame_rect(g, 1, xx, a - 1, a - 1, pal.line)
