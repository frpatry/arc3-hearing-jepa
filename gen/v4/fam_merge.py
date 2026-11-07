"""merge4 (clic, impulsion + fusions) : cliquer dans le champ émet une IMPULSION qui attire d'UN pas vers le point
cliqué chaque objet à moins de ρ cases (variante `pull`) ou chaque objet sur la même ligne / colonne, à toute
distance (variante `beam`). Deux objets ÉGAUX qui se
rencontrent fusionnent en l'objet du rang suivant (plus gros, autre couleur) ; inégaux, ils se bloquent.
But : la RECETTE du bandeau (rang demandé) doit se trouver dans la (les) zone(s) pointillée(s).
Un clic qui n'atteint aucun objet ne fait rien ; les clics dans le bandeau sont ignorés (et gratuits).
Règles ajoutées (ordre tiré par jeu, une par niveau) :
  walls  : murs (blocs) qui arrêtent les objets ;
  chaser : un rôdeur : après chaque impulsion efficace il s'approche de l'objet le plus proche (jusqu'à 2 cases) ;
           s'il est lui-même aspiré contre un objet, celui-ci perd un rang ;
  two    : deux recettes, deux zones ;
  rocks  : pierres qui réagissent À L'INVERSE de l'impulsion (repoussées par l'impulsion) et bloquent ;
  big    : impulsion élargie (rayon + 1 en `pull`, les diagonales aussi en `beam`).
Solvabilité : objets = découpage binaire de la recette ; solution = planificateur glouton (potentiel : fusions
restantes + distance des paires / distance aux zones) simulé exactement ; niveau rejeté s'il échoue."""
import numpy as np

from ..core import EV
from . import draw
from .base import V4Game, mods_for
from .draw import R, blit, frame_rect, rect, sprite

NORMAL, ROCK, CHASER = 0, 1, 2
D8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _sgn(x):
    return (x > 0) - (x < 0)


class Merge4(V4Game):
    family = "merge4"
    mechanic = "click"
    principle = ("impulsion qui attire tout dans un rayon ; fusions d'objets égaux vers une recette affichée "
                 "(rôdeur, pierres, murs)", ["su15", "lf52", "vc33"])
    simple_ops = ()
    CAT = ("walls", "chaser", "two", "rocks", "big")
    exact_progress = False
    free_miss = True
    n_levels_official = (6, 8)
    TMAX = 5

    def setup(self, rng):
        r = self.orng
        self.uses_click = True
        self.variant = str(rng.choice(["pull", "beam"]))
        self.cat = [str(x) for x in rng.permutation(self.CAT)]
        self.C = int(r.choice([5, 5, 6])) if self.s_screen == 1 else 3
        self.hh = 9 if self.s_screen == 1 else 5
        self.rho0 = 2
        self.obj_kind = str(r.choice(["disc", "box", "diamond", "framed"]))

    # ------------------------------------------------------------------ simulation
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

    def _pulse(self, lvl, objs, P):
        """objs : tuple trié de (r, c, rang, sorte) -> nouvel état (trié), événements."""
        N, M, rho = lvl["N"], lvl["M"], lvl["rho"]
        occ = {(o[0], o[1]): [o[2], o[3]] for o in objs}
        walls = lvl["walls"]
        tags = set()
        reach = 1
        order = sorted(occ, key=lambda q: (max(abs(q[0] - P[0]), abs(q[1] - P[1])), q))
        done = set()
        for q in order:
            d = max(abs(q[0] - P[0]), abs(q[1] - P[1]))
            if d == 0 or q not in occ or q in done or not self._hit(lvl, q, P, d):
                continue
            t, kind = occ[q]
            s = -1 if kind == ROCK else 1
            cur = q
            for _k in range(reach):
                if s > 0 and cur == P:
                    break
                nq = (cur[0] + s * _sgn(P[0] - cur[0]), cur[1] + s * _sgn(P[1] - cur[1]))
                if nq == cur or not (0 <= nq[0] < N and 0 <= nq[1] < M) or nq in walls:
                    tags.add("bump")
                    break
                if nq in occ:
                    t2, k2 = occ[nq]
                    if kind == NORMAL and k2 == NORMAL and t2 == t and t < self.TMAX:
                        occ[nq] = [t + 1, NORMAL]
                        del occ[cur]
                        cur = nq
                        tags.add("merge")
                    elif kind == CHASER and k2 == NORMAL:
                        occ[nq][0] -= 1
                        if occ[nq][0] <= 0:
                            del occ[nq]
                        tags.add("hit")
                    else:
                        tags.add("bump")
                    break
                occ[nq] = occ.pop(cur)
                cur = nq
                tags.add("move")
            done.add(cur)
        if lvl["chaser"] and tags:
            ch = [q for q, v in occ.items() if v[1] == CHASER]
            for q in ch:                                         # le rôdeur s'approche (sans toucher tout seul)
                tg = [p for p, v in occ.items() if v[1] == NORMAL]
                if not tg:
                    continue
                p = min(tg, key=lambda p: (max(abs(p[0] - q[0]), abs(p[1] - q[1])), p))
                if max(abs(p[0] - q[0]), abs(p[1] - q[1])) >= 3:
                    nq = (q[0] + _sgn(p[0] - q[0]), q[1] + _sgn(p[1] - q[1]))
                    if nq not in occ and nq not in walls:
                        occ[nq] = occ.pop(q)
        out = tuple(sorted((q[0], q[1], v[0], v[1]) for q, v in occ.items()))
        return out, tags

    def _hit(self, lvl, q, P, d):
        if self.variant == "pull":
            return d <= lvl["rho"]
        dy, dx = q[0] - P[0], q[1] - P[1]
        return dy == 0 or dx == 0 or (lvl["rho"] > self.rho0 and abs(dy) == abs(dx))

    def apply(self, lvl, st, op, px):
        if op != "click":
            return st, False
        P = self._cell(lvl, px)
        if P is None:
            return st, False
        return self._pulse(lvl, st, P)[0], False

    def _zone_ok(self, lvl, st):
        at = {(o[0], o[1]): o[2] for o in st if o[3] == NORMAL}
        return [at.get((z[0], z[1])) == z[2] for z in lvl["zones"]]

    def is_win(self, lvl, st):
        return all(self._zone_ok(lvl, st))

    def click_targets(self, lvl, st):
        return []

    def click_hit(self, lvl, st, px):
        P = self._cell(lvl, px)
        return P is not None and any((o[0], o[1]) == P for o in st)

    # ------------------------------------------------------------------ planificateur glouton
    def _units(self, st):
        return sum(2 ** (o[2] - 1) for o in st if o[3] == NORMAL)

    def _phi(self, lvl, st):
        if self._units(st) < lvl["units"]:
            return 10 ** 6
        norm = [o for o in st if o[3] == NORMAL]
        tiers = {}
        for o in norm:
            tiers.setdefault(o[2], []).append((o[0], o[1]))
        extra = len(norm) - len(lvl["zones"])
        if extra > 0:
            t = min(t for t, v in tiers.items() if len(v) >= 2) if any(len(v) >= 2 for v in tiers.values()) else None
            if t is None:
                return 10 ** 5
            v = tiers[t]
            dmin = min(max(abs(a[0] - b[0]), abs(a[1] - b[1])) for i_, a in enumerate(v) for b in v[i_ + 1:])
            return 100 * extra + dmin
        tot = 0
        for z in lvl["zones"]:
            v = tiers.get(z[2])
            if not v:
                return 10 ** 5
            tot += min(max(abs(a[0] - z[0]), abs(a[1] - z[1])) for a in v)
        return tot

    def _focus(self, lvl, st):
        norm = [o for o in st if o[3] == NORMAL]
        tiers = {}
        for o in norm:
            tiers.setdefault(o[2], []).append((o[0], o[1]))
        if len(norm) > len(lvl["zones"]):
            dup = [t for t, v in tiers.items() if len(v) >= 2]
            if dup:
                v = tiers[min(dup)]
                pr = min(((a, b) for i_, a in enumerate(v) for b in v[i_ + 1:]),
                         key=lambda ab: (max(abs(ab[0][0] - ab[1][0]), abs(ab[0][1] - ab[1][1])), ab))
                return list(pr)
            return [(o[0], o[1]) for o in norm]
        out = []
        for z in lvl["zones"]:
            out += [(z[0], z[1])] + tiers.get(z[2], [])
        return out

    def _candidates(self, lvl, st):
        N, M, rho = lvl["N"], lvl["M"], lvl["rho"]
        cand = []
        seen = set()
        if self.variant == "beam":
            rho = max(N, M)
        for f in self._focus(lvl, st):
            for k in range(0, rho + 1):
                for d in (D8 if k else [(0, 0)]):
                    q = (f[0] + d[0] * k, f[1] + d[1] * k)
                    if 0 <= q[0] < N and 0 <= q[1] < M and q not in seen:
                        seen.add(q)
                        cand.append(q)
        return cand

    def _plan(self, lvl, st):
        """route cohérente : chaque état déjà planifié garde son action suivante (le glouton n'est pas
        « cohérent par suffixe », on suit donc toujours le même chemin)."""
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

    def _plan0(self, lvl, st, limit=150):
        out, cur, seen = [], st, {st}
        low, since = self._phi(lvl, st), 0
        for _ in range(limit):
            if self.is_win(lvl, cur):
                return out
            best = None
            for q in self._candidates(lvl, cur):
                s2, _ = self._pulse(lvl, cur, q)
                if s2 in seen:
                    continue
                v = self._phi(lvl, s2)
                if v >= 10 ** 5:
                    continue
                key = (v, q)
                if best is None or key < best[0]:
                    best = (key, q, s2)
            if best is None:
                return None
            (v, _), q, cur = best
            seen.add(cur)
            out.append(("click", self._px(lvl, q)))
            if v < low:
                low, since = v, 0
            else:
                since += 1
                if since > 24:                                  # stagnation : abandon
                    return None
        return out if self.is_win(lvl, cur) else None

    def _px(self, lvl, q):
        return (lvl["oy"] + q[0] * self.C + self.C // 2, lvl["ox"] + q[1] * self.C + self.C // 2)

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
        C = self.C
        top = self.hh + 1
        Nmax, Mmax = (self.AH - top) // C, self.AW // C
        sh = 1 if i == 0 else 0
        N, M = Nmax - sh, Mmax - sh
        oy, ox = top + (self.AH - top - N * C) // 2, (self.AW - M * C) // 2
        rho = self.rho0 + ("big" in mods)
        lvl = dict(N=N, M=M, oy=oy, ox=ox, rho=rho, mods=mods, walls=frozenset(), chaser="chaser" in mods)
        if "two" in mods:
            T = [[2, 4], [3, 4], [3, 4]][int(rng.integers(3))]
        else:
            T = [3 if i <= 1 else 4 if i < 5 else int(rng.integers(4, 6))]
        best = None
        for _ in range(10):
            free = [(r, c) for r in range(N) for c in range(M)]
            rng.shuffle(free)
            walls = set()
            if "walls" in mods:
                for _w in range(int(rng.integers(2, 4))):
                    r0, c0 = free[int(rng.integers(len(free)))]
                    L = int(rng.integers(2, 4))
                    hor = rng.random() < 0.5
                    for j in range(L):
                        q = (r0, c0 + j) if hor else (r0 + j, c0)
                        if 0 <= q[0] < N and 0 <= q[1] < M:
                            walls.add(q)
            free = [q for q in free if q not in walls]
            zones, objs = [], []

            def pick(dz):
                """case libre éloignée des objets (≥ 2) et des zones (≥ dz) si possible."""
                for j in range(len(free) - 1, -1, -1):
                    q = free[j]
                    if all(max(abs(q[0] - o[0]), abs(q[1] - o[1])) >= 2 for o in objs) and \
                            all(max(abs(q[0] - z[0]), abs(q[1] - z[1])) >= dz for z in zones):
                        return free.pop(j)
                return free.pop()

            for t in T:
                z = pick(4)
                zones.append((z[0], z[1], t))
                parts = [t]
                want = min(2 ** (t - 1), 4 + i + int(rng.integers(0, 2)), 8 if len(T) == 1 else 5)
                while len(parts) < want:
                    big = [j for j, x in enumerate(parts) if x > 1]
                    if not big:
                        break
                    j = big[int(rng.integers(len(big)))]
                    x = parts.pop(j)
                    parts += [x - 1, x - 1]
                for x in parts:
                    q = pick(3)
                    objs.append((q[0], q[1], x, NORMAL))
            if "rocks" in mods:
                for _r in range(int(rng.integers(2, 4))):
                    q = free.pop()
                    objs.append((q[0], q[1], 1, ROCK))
            if lvl["chaser"]:
                zs = [(z[0], z[1]) for z in zones]
                far = [q for q in free if all(max(abs(q[0] - o[0]), abs(q[1] - o[1])) >= 3 for o in objs)]
                if not far:
                    continue
                q = far[0]
                objs.append((q[0], q[1], 1, CHASER))
            lvl["walls"] = frozenset(walls)
            lvl["zones"] = tuple(zones)
            lvl["units"] = sum(2 ** (t - 1) for t in T)
            st0 = tuple(sorted(objs))
            if self.is_win(lvl, st0):
                continue
            lvl.pop("_next", None)
            sol = self._plan0(lvl, st0)
            if sol is None:
                self.why("plan-none")
                continue
            goal = self.min_len(i, n) * 1.2
            if best is None or abs(len(sol) - goal) < abs(len(best[2]) - goal):
                best = (dict(lvl), st0, sol)
            if self.min_len(i, n) <= len(best[2]) <= 1.6 * self.min_len(i, n):
                break
        if best is None:
            return self.why("plan")
        lvl, st0, sol = best
        lvl["_next"] = {}
        nx, c = lvl["_next"], st0
        for a in sol:
            nx.setdefault(c, a)
            c, _ = self.apply(lvl, c, *a)
        return lvl, st0, sol

    def min_len(self, i, n):
        return min(16 + 7 * i, 60)

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        P = self._cell(lvl, px) if op == "click" else None
        if P is None:
            return
        _, tags = self._pulse(lvl, st, P)
        if "merge" in tags:
            ev[EV["collect"]] = ev[EV["color"]] = 1
        if "bump" in tags:
            ev[EV["touch"]] = 1
        if "move" in tags:
            ev[EV["push"]] = 1
        if "hit" in tags:
            ev[EV["danger"]] = 1
        if sum(self._zone_ok(lvl, st2)) > sum(self._zone_ok(lvl, st)):
            ev[EV["open"]] = 1

    # ------------------------------------------------------------------ rendu
    def _tier_sprite(self, t, size=None):
        pal = self.pal
        s = size or min(self.C - 1, 1 + t + (self.C >= 7))
        if self.C <= 4 and size is None:
            s = min(self.C, 1 + (t + 1) // 2 + (t >= 4))
        return sprite(None, self.obj_kind if s >= 3 else "box", s, s, pal.a(t), pal.fg if self.obj_kind == "framed" else None)

    def render(self, lvl, st):
        pal, C = self.pal, self.C
        g = self.blank()
        oy, ox = lvl["oy"], lvl["ox"]
        for (r, c) in lvl["walls"]:
            rect(g, oy + r * C, ox + c * C, C, C, pal.wall)
        ok = self._zone_ok(lvl, st)
        for j, (r, c, t) in enumerate(lvl["zones"]):
            y, x = oy + r * C, ox + c * C
            col = pal.a(t) if ok[j] else pal.line
            frame_rect(g, y, x, C, C, col)
        for (r, c, t, kind) in st:
            y, x = oy + r * C, ox + c * C
            if kind == NORMAL:
                s = self._tier_sprite(t)
                o = (C - s.shape[0]) // 2
                blit(g, s, y + o, x + o)
            elif kind == ROCK:
                blit(g, sprite(None, "checker", C - 2, C - 2, pal.wall, pal.line), y + 1, x + 1)
            else:
                cc = draw.RED if pal.bg != draw.RED else draw.MAROON
                for a in range(C - 2):
                    g[y + 1 + a, x + 1 + a] = R(cc)
                    g[y + 1 + a, x + C - 2 - a] = R(cc)
                g[y + C // 2, x + C // 2] = R(pal.fg)
        # bandeau : échelle des rangs (gauche) et recette (droite)
        hh = self.hh
        rect(g, 0, 0, hh, self.AW, pal.ui)
        draw.hline(g, hh, 0, self.AW - 1, pal.line)
        x = 1
        smax = hh - 2
        for t in range(1, self.TMAX + 1):
            s = max(1, min(smax, t + 1)) if self.s_screen == 1 else max(1, min(smax, (t + 2) // 2))
            spr = sprite(None, self.obj_kind if s >= 3 else "box", s, s, pal.a(t))
            blit(g, spr, 1 + (smax - s) // 2, x)
            x += s + 1
        x = self.AW - 1
        for j, (r, c, t) in reversed(list(enumerate(lvl["zones"]))):
            w = smax + 2
            x -= w
            frame_rect(g, 0, x, hh, w, pal.a(t) if ok[j] else pal.line)
            s = max(1, min(smax - 1, t + 1)) if self.s_screen == 1 else max(1, min(smax - 1, (t + 2) // 2))
            spr = sprite(None, self.obj_kind if s >= 3 else "box", s, s, pal.a(t))
            blit(g, spr, (hh - s) // 2, x + (w - s) // 2)
            x -= 2
        return g
