"""flow4 (objets, éditer PUIS lancer) : un réservoir tombe d'une SOURCE selon une gravité dessinée (flèche du bandeau) ;
on édite d'abord la scène (clic = choisir un déflecteur, flèches = le déplacer d'une case), puis ACTION5 LANCE le flot :
il tombe, glisse sur les déflecteurs, se divise aux bords et doit remplir TOUS les godets sans toucher le sol ni un
piège. Un lancer raté est annulé (l'image finale ne change pas, sauf la barre), les réglages restent.
Variantes : `split` (un déflecteur plat déverse des deux côtés) ou `ramp` (chaque déflecteur est incliné : le flot
ne sort que du côté bas, marqué d'un pixel clair).
Règles ajoutées (ordre tiré par jeu, une par niveau) :
  corner  : une pièce d'angle au comportement inverse (à sens unique en `split`, diviseur en `ramp`) ;
  msource : la source elle-même se sélectionne et glisse le long du bord ;
  second  : une 2e source d'une autre couleur ;
  keyed   : godets colorés : chacun n'accepte que la couleur de son bord ;
  limit   : nombre de lancers ratés limité (pastilles du bandeau) ; au-delà = perdu ;
  hazard  : pièges (pics) que le flot ne doit pas toucher ;
  rotate  : la scène entière est tournée (gravité vers une autre direction, flèches et clics suivent l'écran).
Solvabilité : configuration-solution construite d'abord (godets placés là où le flot tombe), puis pièces dispersées ;
solution = choisir chaque pièce et la ramener (BFS par pièce), puis lancer. L'oracle replanifie depuis l'état courant."""
import numpy as np

from ..core import EV
from . import draw
from .base import V4Game, mods_for
from .draw import R, blit, frame_rect, rect, sprite
from .selkit import DV, RotView, plan_moves


class Flow4(V4Game):
    family = "flow4"
    mechanic = "object"
    principle = ("éditer puis lancer : déplacer des déflecteurs pour guider un flot qui tombe vers des godets ; "
                 "lancer raté annulé, gravité tournée", ["sp80", "sb26", "tn36"])
    simple_ops = ("up", "down", "left", "right", "interact")
    CAT = ("corner", "msource", "second", "keyed", "limit", "hazard", "rotate")
    exact_progress = False
    n_levels_official = (6, 8)

    def setup(self, rng):
        r = self.orng
        self.uses_click = True
        self.free_miss = bool(rng.random() < 0.5)
        self.variant = str(rng.choice(["split", "ramp"]))
        cat = [str(x) for x in rng.permutation(self.CAT)]
        if cat.index("keyed") < cat.index("second"):              # keyed suppose deux sources : second d'abord
            a, b = cat.index("keyed"), cat.index("second")
            cat[a], cat[b] = cat[b], cat[a]
        self.cat = cat
        self.k0 = int(rng.choice([0, 0, 0, 2, 1, 3]))
        self.C = int(r.choice([3, 4, 4])) if self.s_screen == 1 else 2
        self.hud_top = bool(r.random() < 0.7)
        self.hh = 7 if self.s_screen == 1 else 4
        self.thin = bool(r.random() < 0.5)                         # déflecteurs plus minces que la case
        self.src_kind = str(r.choice(["box", "framed", "hat"]))

    # ------------------------------------------------------------------ géométrie
    def _dims(self, k, i):
        rh, rw = self.AH - self.hh - 1, self.AW
        ih, iw = (rh, rw) if k % 2 == 0 else (rw, rh)
        C = self.C
        Rmax, Cmax = ih // C - 1, iw // C - 1
        sh = max(0, 3 - i)
        Rn = max(9, min(Rmax, Rmax - sh))
        Cn = max(10, min(Cmax, Cmax - 2 * sh))
        return Rn, Cn, rh, rw

    def _cells(self, lvl, k, p):
        m = lvl["movers"][k]
        if m["kind"] == "src":
            return [p]
        return [(p[0], p[1] + j) for j in range(m["L"])]

    def _ok(self, lvl, k, p, pos):
        m, Rn, Cn = lvl["movers"][k], lvl["R"], lvl["Cc"]
        if m["kind"] == "src":
            if p[0] != 0 or not (1 <= p[1] <= Cn - 2):
                return False
            others = {pos[j][1] for j in range(len(pos)) if j != k and lvl["movers"][j]["kind"] == "src"}
            others |= {c for c, _ in lvl["fixed_src"]}
            return p[1] not in others
        if not (2 <= p[0] <= Rn - 3 and 0 <= p[1] and p[1] + m["L"] - 1 <= Cn - 1):
            return False
        cells = set(self._cells(lvl, k, p))
        if cells & lvl["hazard"] or cells & lvl["fixed_cells"]:
            return False
        for j in range(len(pos)):
            if j != k and lvl["movers"][j]["kind"] != "src" and cells & set(self._cells(lvl, j, pos[j])):
                return False
        return True

    # ------------------------------------------------------------------ simulation du flot
    def _spread_dirs(self, lvl, m):
        flip = m["kind"] == "corner"
        if (self.variant == "split") != flip:
            return (-1, 1)
        return (m["d"],)

    def _run(self, lvl, pos, cups=None):
        """-> (échec, {col: couleurs arrivées au sol}, pièces touchées, cases mouillées, {godet: couleurs})."""
        Rn, Cn = lvl["R"], lvl["Cc"]
        solid = {}
        for k, m in enumerate(lvl["movers"]):
            if m["kind"] != "src":
                for q in self._cells(lvl, k, pos[k]):
                    solid[q] = k
        for q in lvl["fixed_cells"]:
            solid[q] = -2
        cups = lvl["cups"] if cups is None else cups
        interior = {}
        for j, (x, col) in enumerate(cups):
            interior[(Rn - 1, x)] = j
            solid[(Rn - 1, x - 1)] = solid[(Rn - 1, x + 1)] = -1
        srcs = list(lvl["fixed_src"]) + [(pos[k][1], m["col"]) for k, m in enumerate(lvl["movers"]) if m["kind"] == "src"]
        fail, land, used, wet, fill, falls = False, {}, set(), set(), {}, set()
        haz = lvl["hazard"]
        for sc, col in srcs:
            stack, seen = [("f", 0, sc, 0)], set()
            while stack:
                it = stack.pop()
                if it in seen:
                    continue
                seen.add(it)
                mode, r, c, d = it
                if mode == "f":
                    while True:
                        nr = r + 1
                        if nr >= Rn:
                            land.setdefault(c, set()).add(col)
                            if cups:
                                fail = True
                            break
                        q = (nr, c)
                        if q in interior:
                            fill.setdefault(interior[q], set()).add(col)
                            break
                        if q in haz:
                            fail = True
                            break
                        if q in solid:
                            k = solid[q]
                            if k >= 0:
                                used.add(k)
                                dirs = self._spread_dirs(lvl, lvl["movers"][k])
                            else:
                                dirs = (-1, 1)
                            for dd in dirs:
                                stack.append(("s", r, c, dd))
                            break
                        r = nr
                        wet.add((r, c))
                        falls.add((r, c))
                else:
                    x = c
                    while True:
                        x += d
                        if x < 0 or x >= Cn:
                            fail = True
                            break
                        q = (r, x)
                        if q in solid:
                            break                                   # flaque : le flot s'arrête
                        if q in haz:
                            fail = True
                            break
                        wet.add(q)
                        if (r + 1, x) not in solid:
                            stack.append(("f", r, x, 0))
                            break
        return fail, land, used, wet, fill, falls

    def _success(self, lvl, pos):
        fail, _, _, _, fill, _ = self._run(lvl, pos)
        if fail or len(fill) < len(lvl["cups"]):
            return False
        if lvl["keyed"]:
            return all(fill[j] == {col} for j, (_, col) in enumerate(lvl["cups"]))
        return True

    # ------------------------------------------------------------------ règles
    def apply(self, lvl, st, op, px):
        pos, sel, fails, done = st
        if done:
            return st, False
        if op == "click":
            k = self._mover_at(lvl, pos, px)
            if k is not None and k != sel:
                return (pos, k, fails, done), False
            return st, False
        if op == "interact":
            if self._success(lvl, pos):
                return (pos, sel, fails, 1), False
            if lvl["limit"]:
                return (pos, sel, fails + 1, 0), fails + 1 > lvl["limit"]
            return st, False
        if sel < 0:
            return st, False
        dr, dc = DV[lvl["view"].s2c[op]]
        p = pos[sel]
        q = (p[0] + dr, p[1] + dc)
        if not self._ok(lvl, sel, q, pos):
            return st, False
        return (pos[:sel] + (q,) + pos[sel + 1:], sel, fails, done), False

    def is_win(self, lvl, st):
        return bool(st[3])

    def _mover_at(self, lvl, pos, px):
        cp = lvl["view"].to_canon(px)
        if cp is None:
            return None
        cell = (cp[0] // self.C, cp[1] // self.C)
        for k in range(len(pos)):
            if cell in self._cells(lvl, k, pos[k]):
                return k
        return None

    def _px_of(self, lvl, k, p):
        m = lvl["movers"][k]
        L = 1 if m["kind"] == "src" else m["L"]
        cy = p[0] * self.C + self.C - 1
        cx = p[1] * self.C + (L * self.C) // 2
        return lvl["view"].to_screen((cy, cx))

    def click_targets(self, lvl, st):
        return [self._px_of(lvl, k, p) for k, p in enumerate(st[0]) if k != st[1]]

    def click_hit(self, lvl, st, px):
        return self._mover_at(lvl, st[0], px) is not None

    # ------------------------------------------------------------------ planification (solution et oracle)
    def _plan(self, lvl, st):
        pos, sel, fails, done = st
        if done:
            return []
        mv = plan_moves(list(pos), list(lvl["target"]), lambda k, p, cur: self._ok(lvl, k, p, tuple(cur)), first=sel)
        if mv is None:
            return None
        out, cur, s = [], list(pos), sel
        c2s_op = {n: m for m, n in lvl["view"].s2c.items()}
        for k, path in mv:
            if path and s != k:
                out.append(("click", self._px_of(lvl, k, cur[k])))
                s = k
            for d in path:
                out.append((c2s_op[d], None))
            cur[k] = lvl["target"][k]
        out.append(("interact", None))
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
    def gen_level(self, rng, i, n):
        mods, newest = mods_for(self.cat, i)
        k = self.k0 if "rotate" not in mods else (self.k0 + int(rng.integers(1, 4))) % 4
        Rn, Cn, rh, rw = self._dims(k, i)
        C = self.C
        Hi, Wi = (Rn + 1) * C, Cn * C
        sh = (Hi, Wi) if k % 2 == 0 else (Wi, Hi)
        top = self.hh + 1 if self.hud_top else 0
        oy, ox = top + (rh - sh[0]) // 2, (rw - sh[1]) // 2
        view = RotView(Hi, Wi, k, oy, ox)
        two = "second" in mods or "keyed" in mods
        n_plat = 1 if i == 0 else 2 if i <= 1 else 3 if i <= 3 else 4
        n_plat = max(1, min(n_plat, (Cn - 1) // 4))
        lvl = dict(R=Rn, Cc=Cn, k=k, view=view, mods=mods, keyed="keyed" in mods, limit=2 if "limit" in mods else 0,
                   hazard=frozenset(), cups=(), fixed_src=(), movers=[], Hi=Hi, Wi=Wi, fixed_cells=frozenset())
        cols = [0, 1] if two else [0]
        for _ in range(150):
            scs = sorted(int(x) for x in rng.choice(np.arange(2, Cn - 2), len(cols), replace=False))
            if two and scs[1] - scs[0] < 4:
                continue
            movers = []
            for j in range(n_plat):
                L = int(rng.choice([3, 4, 5] if self.variant == "split" else [2, 3, 4]))
                movers.append(dict(kind="plat", L=L, d=int(rng.choice([-1, 1]))))
            if "corner" in mods:
                movers[int(rng.integers(n_plat))]["kind"] = "corner"
            lvl["movers"] = movers
            lvl["fixed_cells"] = frozenset()
            lvl["hazard"] = frozenset()
            lvl["fixed_src"] = tuple((scs[j], cols[j]) for j in range(len(cols)))
            # placement de la solution : chaque déflecteur sous un filet qui tombe
            pos = []
            ok = True
            for j in range(n_plat):
                lvl["movers"] = movers[:j]
                falls = self._run(lvl, tuple(pos), cups=())[5]
                if j == 0 or (j == 1 and two and rng.random() < 0.4):    # sous une source
                    x = scs[j]
                    r0 = int(rng.integers(2, Rn - 2))
                else:
                    cand = sorted(q for q in falls if 2 <= q[0] <= Rn - 3)
                    if not cand:
                        ok = False
                        break
                    r0, x = cand[int(rng.integers(len(cand)))]
                L = movers[j]["L"]
                p = (r0, x - int(rng.integers(0, L)))
                lvl["movers"] = movers[:j + 1]
                if not self._ok(lvl, j, p, tuple(pos) + (p,)):
                    ok = False
                    break
                pos.append(p)
            if not ok:
                continue
            lvl["movers"] = movers
            pos = tuple(pos)
            lvl["fixed_cells"] = frozenset()
            nf = int(rng.integers(0, 3)) if i >= 1 else int(rng.integers(0, 2))
            fc = set()
            for _f in range(nf):                                    # rebords fixes (décor solide, non cliquable)
                L = int(rng.integers(2, 5))
                r0, c0 = int(rng.integers(2, Rn - 2)), int(rng.integers(0, Cn - L + 1))
                cells = {(r0, c0 + j) for j in range(L)}
                if any(self._cells(lvl, kk, pos[kk])[0][0] == r0 and set(self._cells(lvl, kk, pos[kk])) & cells
                       for kk in range(n_plat)) or any(abs(q[1] - sc) <= 0 and q[0] <= 2 for q in cells for sc in scs):
                    continue
                if not any(set(self._cells(lvl, kk, pos[kk])) & cells for kk in range(n_plat)):
                    fc |= cells
            lvl["fixed_cells"] = frozenset(fc)
            fail, land, used, wet, _, _ = self._run(lvl, pos, cups=())
            if fail or len(used) < n_plat or not (1 <= len(land) <= 4):
                self.why("p-run")
                continue
            xs = sorted(land)
            if xs[0] < 1 or xs[-1] > Cn - 2 or any(b - a < 3 for a, b in zip(xs, xs[1:])):
                self.why("p-cups")
                continue
            if lvl["keyed"] and any(len(land[x]) > 1 for x in xs):
                self.why("p-mix")
                continue
            if lvl["keyed"] and len({next(iter(land[x])) for x in xs}) < 2:
                self.why("p-one")
                continue
            lvl["cups"] = tuple((x, min(land[x])) for x in xs)
            if "hazard" in mods:
                occ = set(wet)
                for kk in range(n_plat):
                    occ |= set(self._cells(lvl, kk, pos[kk]))
                cand = sorted({(r + dr, c + dc) for (r, c) in wet for dr, dc in DV.values()} - occ)
                cand = [q for q in cand if 2 <= q[0] <= Rn - 2 and 0 <= q[1] < Cn and abs(q[1] - scs[0]) > 0]
                if len(cand) < 2:
                    continue
                pick = rng.permutation(len(cand))[:int(rng.integers(2, 4))]
                lvl["hazard"] = frozenset(cand[j] for j in pick)
            if not self._success(lvl, pos):
                continue
            break
        else:
            return self.why("place")
        # source mobile : elle devient une pièce (sa position-solution est celle tirée)
        target = list(pos)
        if "msource" in mods:
            sc, col = lvl["fixed_src"][0]
            lvl["fixed_src"] = lvl["fixed_src"][1:]
            lvl["movers"] = lvl["movers"] + [dict(kind="src", L=1, d=0, col=col)]
            target.append((0, sc))
        lvl["target"] = tuple(target)
        # dispersion : positions de départ éloignées ; on garde la plus longue de plusieurs dispersions
        nm = len(target)
        best = None
        for _ in range(14):
            start = []
            good = True
            for kk in range(nm):
                m = lvl["movers"][kk]
                if m["kind"] == "src":
                    cand = [(0, c) for c in range(1, Cn - 1) if abs(c - target[kk][1]) >= 3]
                else:
                    cand = [(r, c) for r in range(2, Rn - 2) for c in range(0, Cn - m["L"] + 1)
                            if abs(r - target[kk][0]) + abs(c - target[kk][1]) >= 4]
                cand = [p for p in cand if self._ok(lvl, kk, p, tuple(start) + (p,) + tuple(target[kk + 1:]))]
                if not cand:
                    good = False
                    break
                dist = [abs(p[0] - target[kk][0]) + abs(p[1] - target[kk][1]) for p in cand]
                far = [p for p, d in zip(cand, dist) if d >= 0.6 * max(dist)]
                start.append(far[int(rng.integers(len(far)))])
            if not good:
                continue
            start = tuple(start)
            if self._success(lvl, start):
                continue
            st0 = (start, -1, 0, 0)
            sol = self._plan(lvl, st0)
            if sol is not None and (best is None or len(sol) > len(best[1])):
                best = (st0, sol)
        if best is None:
            return self.why("scatter")
        return lvl, best[0], best[1]

    def min_len(self, i, n):
        return min(14 + 7 * i, 60)

    def body(self, lvl, st):
        if st[1] < 0:
            return None
        p = st[0][st[1]]
        return (p[0], p[1], lvl["R"], lvl["Cc"])

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        if op == "click" and st2[1] != st[1]:
            ev[EV["toggle"]] = 1
        if op == "interact":
            if st2[3]:
                ev[EV["collect"]] = ev[EV["open"]] = 1
            else:
                ev[EV["danger"]] = 1
        if op in DV and st2 == st and st[1] >= 0:
            ev[EV["touch"]] = 1

    # ------------------------------------------------------------------ rendu
    def render(self, lvl, st):
        pal, C = self.pal, self.C
        g = self.blank()
        Rn, Cn = lvl["R"], lvl["Cc"]
        img = np.full((lvl["Hi"], lvl["Wi"]), R(pal.bg), np.int64)
        liq = [pal.a(1), pal.a(2)]
        plat_c, corner_c, haz_c = pal.a(0), pal.a(4), pal.a(5)
        sel_c = draw.WHITE if pal.bg != draw.WHITE and self.sel_style == "white" else None
        # sol (danger) et parois latérales
        rect(img, Rn * C, 0, C, Cn * C, pal.wall)
        for x, col in lvl["cups"]:
            cc = liq[col] if lvl["keyed"] else pal.fg
            y0, x0 = (Rn - 1) * C, (x - 1) * C
            rect(img, y0, x0, C, max(1, C // 2 + (C == 2)), cc)
            rect(img, y0, x0 + 3 * C - max(1, C // 2 + (C == 2)), C, max(1, C // 2 + (C == 2)), cc)
            rect(img, y0 + C - 1, x0, 1, 3 * C, cc)
        for (r, c) in lvl["fixed_cells"]:
            rect(img, r * C + (C // 3 if self.thin else 0), c * C, C - (C // 3 if self.thin else 0), C, pal.wall)
        for (r, c) in lvl["hazard"]:
            blit(img, sprite(None, "diamond", C, C, haz_c), r * C, c * C)
        pos = st[0]
        for sc, col in lvl["fixed_src"]:
            blit(img, sprite(None, self.src_kind, C, C, liq[col], pal.fg), 0, sc * C)
        for k, m in enumerate(lvl["movers"]):
            r, c = pos[k]
            selected = k == st[1]
            if m["kind"] == "src":
                col = liq[m["col"]]
                blit(img, sprite(None, "framed" if self.src_kind != "framed" else "box", C, C, col,
                                 sel_c if (selected and sel_c is not None) else pal.fg), 0, c * C)
                if selected and sel_c is None:
                    img[C // 2, c * C + C // 2] = R(pal.fg)
                continue
            base = corner_c if m["kind"] == "corner" else plat_c
            col = sel_c if (selected and sel_c is not None) else base
            t = max(1, C - 1 - (C // 2 if self.thin else 0))
            y0, x0, w = r * C + C - t, c * C, m["L"] * C
            rect(img, y0, x0, t, w, col)
            dirs = self._spread_dirs(lvl, m)
            if len(dirs) == 1:                                   # sens unique : bord haut côté fermé, pixel côté bas
                closed = x0 if dirs[0] > 0 else x0 + w - 1
                rect(img, r * C, closed, C - t, 1, col)
                open_x = x0 + w - 1 if dirs[0] > 0 else x0
                img[y0 + t - 1, open_x] = R(pal.fg)
            elif m["kind"] == "corner":                         # diviseur : pointe au milieu
                rect(img, r * C, x0 + w // 2, C - t, 1, col)
            if selected and sel_c is None:
                img[y0, x0 + w // 2] = R(pal.fg)
        lvl["view"].paste(g, img)
        # bandeau : godets, flèche de gravité, lancers restants
        hy = 0 if self.hud_top else self.AH - self.hh
        rect(g, hy, 0, self.hh, self.AW, pal.ui)
        s = self.hh - 2
        arrow = np.rot90(sprite(None, "arrow", s if s % 2 else s + 1, s if s % 2 else s + 1, pal.fg),
                         lvl["view"].gravity_rot())
        blit(g, arrow[:s, :s], hy + 1, 1)
        x = 3 + s
        for xx, col in lvl["cups"]:
            cc = liq[col] if lvl["keyed"] else pal.line
            frame_rect(g, hy + 1, x, s, s, cc)
            rect(g, hy + 1, x + 1, 1, max(1, s - 2), pal.ui)
            x += s + 1
        if lvl["limit"]:
            x = self.AW - 2
            for j in range(lvl["limit"] + 1):
                c = pal.a(3) if j >= st[2] else pal.ui
                rect(g, hy + 1, x - 1, s, 2 if s > 2 else 1, c)
                if j < st[2]:
                    frame_rect(g, hy + 1, x - 1, s, 2 if s > 2 else 1, pal.line)
                x -= 4
        return g
