"""Monde « aventure » à MODULES (v2) : un avatar navigue vers la sortie ; chaque niveau AJOUTE un module (clés-portes,
boutons-portails, recoloration, téléporteurs, lave, ennemis, glace, caisses, objets à ramasser), comme les jeux officiels
où chaque niveau introduit un élément ou une règle. Les familles d'avatar v1 (maze, keys_doors, switches, color_pads,
teleport, hazards, ice, collect) deviennent, en look officiel, ce monde avec leur module comme BASE.

État : (pos, clés, portails, couleur, caisses, objets, t) — BFS exact (plafonné), solution optimale = baseline.
Chaque module nouvellement introduit doit être UTILISÉ par la solution optimale (vérifié en rejouant la solution)."""
import numpy as np

from .core import (AVATAR, BG, DETAIL, DIRS, DIR_OPS, EV, GOAL, MARK, OBJ2, PANEL, WALL, Canvas, bfs_dist,
                   free_cells, perfect_maze)
from .families_avatar import AvatarGame

KEYC = (4, 5)
GATE, BUTTON = 6, 7
PADC = (8, 9)
LAVA, ENEMY, ICE = 10, 11, 12
TELE, BOX, ITEM, PLATE, PGATE = OBJ2[0], OBJ2[1], OBJ2[2], OBJ2[3], OBJ2[4]

MODULES = ("keys", "gates", "color", "teleport", "lava", "enemy", "ice", "boxes", "collect")
USE_TAG = dict(keys="key", gates="toggle", color="color", teleport="tele", boxes="push", collect="item",
               lava="near_lava", enemy="near_enemy", ice="slide")
BASE_OF = dict(maze=None, keys_doors="keys", switches="gates", color_pads="color", teleport="teleport",
               hazards="lava", ice="ice", collect="collect", quest=None)


def _neighbors(walls, p):
    H, W = walls.shape
    for dr, dc in DIRS.values():
        q = (p[0] + dr, p[1] + dc)
        if 0 <= q[0] < H and 0 <= q[1] < W and not walls[q]:
            yield q


class World(AvatarGame):
    family = "quest"
    layouts = ("maze",)
    bfs_cap = 60000
    max_tries = 300

    def __init__(self, seed=0, look="official", family=None):
        if family is not None:
            self.family = family
        super().__init__(seed, look)

    def extra_setup(self, rng):
        r = self.orng if self.official else rng
        base = BASE_OF.get(self.family)
        cat = [m for m in MODULES if m != base]
        if base in ("lava",):
            cat = [m for m in cat if m not in ("enemy", "boxes")]
            cat.insert(0, "enemy")                     # dangers : la lave d'abord, puis les ennemis
        else:
            cat = list(r.permutation(cat))
        self.base = base
        self.catalogue = cat
        self.wait = bool(r.random() < 0.4)             # ACTION5 = attendre (utile face aux ennemis)
        if self.wait:
            self.simple_ops = DIR_OPS + ("interact",)
        self.loops = float(r.uniform(0.0, 0.06))
        self.key_style = str(r.choice(["dot", "diamond", "small_dot"])) if self.official else "dot"
        self.item_style = str(r.choice(["dot", "cross", "diamond"])) if self.official else "cross"
        self.enemy_style = str(r.choice(["eye", "box_dot", "diamond"])) if self.official else "full"
        self.use_panel = self.official and self.panel
        if self.use_panel:
            self.max_cells = max(7, self.max_cells - 1)

    # ------------------------------------------------------------------ curriculum
    def active_modules(self, i):
        act = [] if self.base is None else [self.base]
        if self.base == "lava" and i >= 1:
            act.append("enemy")
        extras = [m for m in self.catalogue if m not in act]
        k = i - (1 if self.base == "lava" and i >= 1 else 0)
        new = extras[:max(0, k)]
        keep = new[-3:]                                # au plus 3 modules en plus de la base (le plus ancien sort)
        if "boxes" in keep and "enemy" in keep:        # explosion d'états : caisses et ennemis jamais ensemble
            old = "boxes" if keep.index("boxes") < keep.index("enemy") else "enemy"
            keep = [m for m in keep if m != old]
        newest = new[-1] if new else (self.base if i == 0 else None)
        return act + [m for m in keep if m not in act], newest

    # ------------------------------------------------------------------ génération
    def gen_level(self, rng, i, n):
        mods, newest = self.active_modules(i)
        fails = self.__dict__.setdefault("_fails", {})
        nf = fails[i] = fails.get(i, -1) + 1
        if nf > 120:                                   # repli : on garde la base et le module nouveau seulement
            mods = [m for m in mods if m in (self.base, newest)]
        if nf > 220:
            mods, newest = [m for m in mods if m == newest], newest
        d = i / max(1, n - 1)
        mc = (self.cap(21) - 1) // 2
        lo = max(3, mc // 2) if self.official else 3           # look officiel : l'aire de jeu remplit l'écran
        ch = min(mc, lo + int(round(d * (mc - lo))) + int(rng.integers(0, 2)))
        cw = min(mc, lo + int(round(d * (mc - lo))) + int(rng.integers(0, 2)))
        loops = self.loops + (0.08 if "ice" in mods else 0)
        if "teleport" in mods:
            loops = min(loops, 0.02)
        walls = perfect_maze(rng, max(2, ch), max(2, cw), loops)
        fr = free_cells(walls)
        s0 = fr[int(rng.integers(len(fr)))]
        dd = bfs_dist(walls, s0)
        far = [p for p in fr if dd[p] >= 0.7 * dd.max()]
        e = far[int(rng.integers(len(far)))]
        L = dict(walls=walls, exit=e, keys={}, doors={}, buttons={}, gates={}, pads={}, cbar={}, lava=frozenset(),
                 enemies=[], P=1, ice=frozenset(), tele={}, pairs=[], boxes0=(), items=(), plates=frozenset(), pgates=frozenset(), mods=tuple(mods))
        used = {s0, e}
        order = sorted(mods, key=lambda m: {"teleport": 0, "collect": 2, "ice": 3, "enemy": 4, "lava": 5}.get(m, 1))
        why = self.__dict__.setdefault("why", {})
        for m in order:
            ok = getattr(self, "_place_" + m)(rng, L, s0, used, d)
            if not ok:
                why["place_" + m] = why.get("place_" + m, 0) + 1
                return None
        lvl = self.mk(walls, **{k: v for k, v in L.items() if k != "walls"})
        st0 = (s0, 0, 0, -1, tuple(sorted(L["boxes0"])), 0, 0)
        sol = self.solve(lvl, st0)
        if sol is None:
            why["bfs_" + str(newest)] = why.get("bfs_" + str(newest), 0) + 1
            return None
        if newest is not None:                         # le module nouveau doit servir
            tags, st = set(), st0
            for op, px in sol:
                st, _, tg = self._step(lvl, st, op)
                tags |= tg
            if USE_TAG[newest] not in tags:
                why["unused_" + str(newest)] = why.get("unused_" + str(newest), 0) + 1
                return None
        return lvl, st0, sol

    def _path(self, L, s0):
        """chemin le plus court départ -> sortie dans le graphe (murs + téléporteurs), autres modules ignorés."""
        walls, tele, e = L["walls"], L["tele"], L["exit"]
        prev, q = {s0: None}, [s0]
        for p in q:
            if p == e:
                break
            for nq in _neighbors(walls, p):
                nq = tele.get(nq, nq)
                if nq not in prev:
                    prev[nq] = p
                    q.append(nq)
        if e not in prev:
            return None
        path, p = [], e
        while p is not None:
            path.append(p)
            p = prev[p]
        return path[::-1]

    def _corridor(self, L, s0, used):
        """union des plus courts chemins départ -> chaque élément utile et -> sortie (à garder libres)."""
        walls, tele = L["walls"], L["tele"]
        prev, q = {s0: None}, [s0]
        for p in q:
            for nq in _neighbors(walls, p):
                nq2 = tele.get(nq, nq)
                for x in {nq, nq2}:
                    if x not in prev:
                        prev[x] = p
                        q.append(x)
        cor = set()
        for t in list(used) + [L["exit"]]:
            p = t
            while p is not None and p in prev and p not in cor:
                cor.add(p)
                p = prev[p]
        return cor

    def _dead_end(self, rng, cells, s0, L, used):
        ends = [p for p in cells if len(list(_neighbors(L["walls"], p))) == 1]
        return self._far_cell(rng, ends, s0, L, used) or self._far_cell(rng, cells, s0, L, used)

    def _reach(self, L, s0, blocked):
        w = L["walls"].copy()
        for p in blocked:
            w[p] = True
        seen, q = {s0}, [s0]
        for p in q:
            for nq in _neighbors(w, p):
                nq = L["tele"].get(nq, nq)
                if nq not in seen:
                    seen.add(nq)
                    q.append(nq)
        return seen

    def _blockers(self, L):
        return set(L["doors"]) | set(L["gates"]) | set(L["cbar"]) | set(L["pgates"])

    def _on_path(self, rng, L, s0, used, lo=0.25, hi=0.85, cut=False):
        """case du chemin départ->sortie ; cut=True : seulement les points de passage obligés (articulations)."""
        path = self._path(L, s0)
        if path is None or len(path) < 6:
            return None
        cand = [p for p in path[max(1, int(lo * len(path))):int(hi * len(path))] if p not in used and p not in L["tele"]]
        if cut:
            bl = self._blockers(L)
            cand = [p for p in cand if L["exit"] not in self._reach(L, s0, bl | {p})]
        return cand[int(rng.integers(len(cand)))] if cand else None

    def _far_cell(self, rng, cells, s0, L, used):
        cells = [p for p in cells if p not in used and p not in L["tele"]]
        if not cells:
            return None
        d = bfs_dist(L["walls"], s0)
        cells.sort(key=lambda p: -d[p])
        top = cells[:max(1, len(cells) // 2)]
        return top[int(rng.integers(len(top)))]

    def _place_keys(self, rng, L, s0, used, d):
        for k in range(1 + (d > 0.6 and rng.random() < 0.5)):
            p = self._on_path(rng, L, s0, used, cut=True)
            if p is None:
                return False
            L["doors"][p] = k
            used.add(p)
            if L["exit"] in self._reach(L, s0, self._blockers(L)):
                return False                          # la porte doit couper le chemin
            reach = self._reach(L, s0, self._blockers(L))
            kp = self._far_cell(rng, list(reach), s0, L, used)
            if kp is None:
                return False
            L["keys"][kp] = k
            used.add(kp)
        return True

    def _place_gates(self, rng, L, s0, used, d):
        p = self._on_path(rng, L, s0, used, cut=True)
        if p is None:
            return False
        L["gates"][p] = 0
        used.add(p)
        reach = self._reach(L, s0, self._blockers(L))
        if L["exit"] in reach:
            return False
        for _ in range(1 + (d > 0.5)):
            b = self._dead_end(rng, list(reach), s0, L, used)
            if b is None:
                return False
            L["buttons"][b] = 0
            used.add(b)
        return True

    def _place_color(self, rng, L, s0, used, d):
        c = int(rng.integers(2))
        p = self._on_path(rng, L, s0, used, cut=True)
        if p is None:
            return False
        L["cbar"][p] = c
        used.add(p)
        reach = self._reach(L, s0, self._blockers(L))
        pad = self._far_cell(rng, list(reach), s0, L, used)
        if pad is None:
            return False
        L["pads"][pad] = c
        used.add(pad)
        if d > 0.3:                                    # leurre : une pastille de l'autre couleur
            ends = [p for p in reach if len(list(_neighbors(L["walls"], p))) == 1]       # cul-de-sac : évitable
            dec = self._far_cell(rng, ends, s0, L, used)
            if dec is not None:
                L["pads"][dec] = 1 - c
                used.add(dec)
        return True

    def _place_teleport(self, rng, L, s0, used, d):
        path = self._path(L, s0)
        if path is None or len(path) < 8:
            return False
        cut = self._on_path(rng, L, s0, used, 0.3, 0.7, cut=True)
        if cut is None:
            return False
        L["walls"][cut] = True
        a_side = self._reach(L, s0, set())
        if L["exit"] in a_side:
            return False
        b_side = [p for p in free_cells(L["walls"]) if p not in a_side]
        a = self._far_cell(rng, list(a_side), s0, L, used)
        if a is None:
            return False
        used.add(a)
        db = bfs_dist(L["walls"], L["exit"])
        cand = [p for p in b_side if p not in used and db[p] >= 0]
        if not cand:
            return False
        b = cand[int(rng.integers(len(cand)))]
        used.add(b)
        L["tele"] = {a: b, b: a}
        L["pairs"] = [(a, b)]
        return True

    def _place_lava(self, rng, L, s0, used, d):
        """lave le long du chemin, sans jamais couper l'accès à un élément utile ni à la sortie."""
        path = set(self._path(L, s0) or [])
        fr = [p for p in free_cells(L["walls"]) if p not in used and p not in path and p not in L["tele"]]
        near = [p for p in fr if any(q in path for q in _neighbors(L["walls"], p))]
        cor = self._corridor(L, s0, used | {c for e in L["enemies"] for c in e})
        near = [p for p in near if p not in cor]
        lava = set()
        for j in rng.permutation(len(near)):
            if len(lava) >= 2 + int(4 * d):
                break
            lava.add(near[j])
        if not lava:
            return False
        L["lava"] = frozenset(lava)
        used |= lava
        return True

    def _place_enemy(self, rng, L, s0, used, d):
        """ennemi qui patrouille EN TRAVERS du chemin (sinon, de face dans un couloir, on ne passe jamais)."""
        path = self._path(L, s0) or []
        onp = self._corridor(L, s0, used)
        P = L["P"]
        H, W = L["walls"].shape
        js = list(rng.permutation(np.arange(2, max(3, len(path) - 1))))
        for j in js:
            if len(L["enemies"]) >= 1 + (d > 0.6):
                break
            p0 = path[j]
            if p0 in used or p0 in L["tele"]:
                continue
            for pr, pc in list(DIRS.values()):
                line = [p0]
                for sgn in (1, -1):
                    q = p0
                    for _ in range(2):
                        q = (q[0] + sgn * pr, q[1] + sgn * pc)
                        if not (0 <= q[0] < H and 0 <= q[1] < W) or L["walls"][q] or q in used or q in onp \
                                or q in L["tele"]:
                            break
                        line.append(q) if sgn > 0 else line.insert(0, q)
                if len(line) >= 2:
                    L["enemies"].append(tuple(line))
                    P = int(np.lcm(P, 2 * (len(line) - 1)))
                    used.add(p0)
                    break
        L["P"] = P
        return bool(L["enemies"])

    def _place_ice(self, rng, L, s0, used, d):
        """glace sur des tronçons DROITS (on glisse jusqu'au bout) : jamais sur un carrefour (sinon on ne s'y arrête pas)."""
        walls = L["walls"]
        fr = [p for p in free_cells(walls) if p not in used and p not in L["tele"]]
        straight = set()
        for p in fr:
            nb = list(_neighbors(walls, p))
            if len(nb) == 2 and (nb[0][0] == nb[1][0] or nb[0][1] == nb[1][1]):
                straight.add(p)
        path = self._path(L, s0) or []
        runs, cur = [], []
        for p in path:
            if p in straight:
                cur.append(p)
            else:
                if len(cur) >= 2:
                    runs.append(cur)
                cur = []
        if len(cur) >= 2:
            runs.append(cur)
        if not runs:
            return False
        ice = set()
        for j in rng.permutation(len(runs))[:1 + int(2 * d)]:
            ice |= set(runs[j])
        L["ice"] = frozenset(ice)
        return True

    def _place_boxes(self, rng, L, s0, used, d):
        """caisse à pousser sur une plaque de pression qui ouvre un portail (tant que la caisse y reste)."""
        g = self._on_path(rng, L, s0, used, cut=True)
        if g is None:
            return False
        L["pgates"] = frozenset([g])
        used.add(g)
        R = self._reach(L, s0, self._blockers(L))
        if L["exit"] in R:
            return False
        trip = []
        for a in R:
            if a in used or a in L["tele"] or a == s0:
                continue
            for dr, dc in DIRS.values():
                back = (a[0] - dr, a[1] - dc)
                run = []
                q = a
                for _ in range(1 + int(d > 0.4)):
                    q = (q[0] + dr, q[1] + dc)
                    if q not in R or q in used or q in L["tele"]:
                        break
                    run.append(q)
                if back in R and back not in used and run:
                    trip.append((a, run[-1]))
        if not trip:
            return False
        box, plate = trip[int(rng.integers(len(trip)))]
        L["boxes0"] = (box,)
        L["plates"] = frozenset([plate])
        dr, dc = (plate[0] - box[0]) // max(1, abs(plate[0] - box[0]) + abs(plate[1] - box[1])), \
            (plate[1] - box[1]) // max(1, abs(plate[0] - box[0]) + abs(plate[1] - box[1]))
        cells = {box, plate, (box[0] - dr, box[1] - dc), (box[0] + dr, box[1] + dc)}
        used |= cells
        return True

    def _place_collect(self, rng, L, s0, used, d):
        fr = list(self._reach(L, s0, set()))
        items = []
        for _ in range(2 + int(2 * d)):
            p = self._far_cell(rng, fr, s0, L, used)
            if p is None:
                break
            items.append(p)
            used.add(p)
        L["items"] = tuple(items)
        return len(items) >= 2

    # ------------------------------------------------------------------ dynamique
    @staticmethod
    def epos(path, t):
        L = len(path)
        P = 2 * (L - 1)
        k = t % P
        return path[k if k < L else P - k]

    def _special(self, lvl, q):
        if q in lvl["plates"]:
            return False
        return q in lvl["keys"] or q in lvl["buttons"] or q in lvl["pads"] or q in lvl["tele"] or \
            q in lvl["items"] or q == lvl["exit"] or q in lvl["doors"] or q in lvl["gates"] or q in lvl["cbar"] \
            or q in lvl["lava"] or q in lvl["ice"]

    def _blocked(self, lvl, q, keys, gates, col, boxes=()):
        if not (0 <= q[0] < lvl["H"] and 0 <= q[1] < lvl["W"]) or lvl["walls"][q]:
            return True
        if q in lvl["pgates"] and not any(b in lvl["plates"] for b in boxes):
            return True
        if q in lvl["doors"] and not keys >> lvl["doors"][q] & 1:
            return True
        if q in lvl["gates"] and not gates & 1:
            return True
        if q in lvl["cbar"] and col != lvl["cbar"][q]:
            return True
        return False

    def _step(self, lvl, st, op):
        pos, keys, gates, col, boxes, items, t = st
        tags = set()
        if op in DIRS:
            dr, dc = DIRS[op]
            p = pos
            while True:
                q = (p[0] + dr, p[1] + dc)
                if self._blocked(lvl, q, keys, gates, col, boxes):
                    tags.add("bump_" + ("door" if (q in lvl["doors"] or q in lvl["gates"] or q in lvl["cbar"]
                                                   or q in lvl["pgates"]) else "wall"))
                    break
                if q in boxes:
                    q2 = (q[0] + dr, q[1] + dc)
                    if self._blocked(lvl, q2, keys, gates, col, boxes) or q2 in boxes or self._special(lvl, q2) \
                            or q2 in lvl["pgates"]:
                        tags.add("bump_box")
                        break
                    was = any(b in lvl["plates"] for b in boxes)
                    boxes = tuple(sorted([b for b in boxes if b != q] + [q2]))
                    tags.add("push")
                    if not was and any(b in lvl["plates"] for b in boxes):
                        tags.add("open")
                p = q
                if p in lvl["keys"] and not keys >> lvl["keys"][p] & 1:
                    keys |= 1 << lvl["keys"][p]
                    tags |= {"key", "open"}
                if p in lvl["buttons"]:
                    gates ^= 1
                    tags.add("toggle")
                    if gates & 1:
                        tags.add("open")
                if p in lvl["pads"] and lvl["pads"][p] != col:
                    col = lvl["pads"][p]
                    tags.add("color")
                if p in lvl["items"]:
                    j = lvl["items"].index(p)
                    if not items >> j & 1:
                        items |= 1 << j
                        tags.add("item")
                if p in lvl["tele"]:
                    p = lvl["tele"][p]
                    tags.add("tele")
                    break
                if p in lvl["ice"]:
                    tags.add("slide")
                    continue
                break
            pos = p
        t2 = (t + 1) % lvl["P"]
        dead = pos in lvl["lava"]
        for path in lvl["enemies"]:
            a, b = self.epos(path, t), self.epos(path, t2)
            if b == pos or (b == st[0] and a == pos):
                dead = True
        if any(abs(pos[0] - a) + abs(pos[1] - b) <= 1 for a, b in lvl["lava"]):
            tags.add("near_lava")
        if any(abs(pos[0] - a) + abs(pos[1] - b) <= 1 for a, b in (self.epos(pp, t2) for pp in lvl["enemies"])):
            tags.add("near_enemy")
        return (pos, keys, gates, col, boxes, items, t2), dead, tags

    def apply(self, lvl, st, op, px):
        st2, dead, _ = self._step(lvl, st, op)
        return st2, dead

    def is_win(self, lvl, st):
        return st[0] == lvl["exit"] and st[5] == (1 << len(lvl["items"])) - 1

    def heuristic(self, lvl, st):
        if "_de" not in lvl:
            lvl["_de"] = bfs_dist(lvl["walls"], lvl["exit"])
        v = lvl["_de"][st[0]]
        return float(v if v >= 0 else 60) + 3 * (len(lvl["items"]) - bin(st[5]).count("1"))

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        _, _, tags = self._step(lvl, st, op)
        if "push" in tags:
            ev[EV["push"]] = 1
        if tags & {"key", "item", "toggle", "color", "tele", "bump_door", "bump_box", "push"} or dead:
            ev[EV["touch"]] = 1
        if tags & {"key", "item"}:
            ev[EV["collect"]] = 1
        if "open" in tags:
            ev[EV["open"]] = 1
        if "color" in tags:
            ev[EV["color"]] = 1
        if "toggle" in tags:
            ev[EV["toggle"]] = 1
        if "tele" in tags:
            ev[EV["teleport"]] = 1
        if tags & {"near_lava", "near_enemy"} or dead:
            ev[EV["danger"]] = 1

    def min_len(self, i, n):
        mc = (self.cap(21) - 1) // 2
        return int(min(12 + 7 * i, 90, 0.35 * (2 * mc + 1) ** 2))

    # ------------------------------------------------------------------ rendu
    def render(self, lvl, st):
        pos, keys, gates, col, boxes, items, t = st
        cv = self.canvas(lvl)
        done = items == (1 << len(lvl["items"])) - 1
        for p in lvl["ice"]:
            cv.put(*p, ICE, "corners" if lvl["t"] >= 3 else "dot")
        for p in lvl["lava"]:
            cv.put(*p, LAVA, "full")
        cv.put(*lvl["exit"], GOAL, self.goal_style if done else "corners")
        for p, k in lvl["doors"].items():
            if not keys >> k & 1:
                cv.put(*p, KEYC[k], "full")
        for p, k in lvl["keys"].items():
            if not keys >> k & 1:
                cv.put(*p, KEYC[k], self.key_style)
        for p in lvl["gates"]:
            cv.put(*p, GATE, "corners" if gates & 1 else "full")
        for p in lvl["buttons"]:
            cv.put(*p, BUTTON, "ring")
        for p, c in lvl["pads"].items():
            cv.put(*p, PADC[c], "cross")
        for p, c in lvl["cbar"].items():
            cv.put(*p, PADC[c], "full")
        for a, b in lvl["pairs"]:
            cv.put(*a, TELE, "ring_dot" if lvl["t"] >= 3 else "ring")
            cv.put(*b, TELE, "ring_dot" if lvl["t"] >= 3 else "ring")
        for j, p in enumerate(lvl["items"]):
            if not items >> j & 1:
                cv.put(*p, ITEM, self.item_style)
        on = any(b in lvl["plates"] for b in boxes)
        for p in lvl["plates"]:
            cv.put(*p, PLATE, "frame" if lvl["t"] >= 3 else "ring")
        for p in lvl["pgates"]:
            cv.put(*p, PGATE, "corners" if on else "full")
        for b in boxes:
            cv.put(*b, BOX, "small" if lvl["t"] >= 4 else "full")
        for path in lvl["enemies"]:
            cv.put(*self.epos(path, t), ENEMY, self.enemy_style)
        if col >= 0:
            cv.put(*pos, PADC[col], "full" if lvl["t"] < 3 else "small", DETAIL)
            cv.put(*pos, DETAIL, "dot")
        else:
            cv.put(*pos, AVATAR, self.av_style)
        g = cv.g
        if self.use_panel:                             # inventaire / couleur courante sous un séparateur
            T = lvl["t"]
            pan = Canvas(1, lvl["W"], T).g
            icons = [(KEYC[k], self.key_style) for k in range(2) if keys >> k & 1]
            if col >= 0:
                icons.append((PADC[col], "full"))
            icons += [(ITEM, self.item_style)] * (len(lvl["items"]) - bin(items).count("1"))
            pc = Canvas(1, lvl["W"], T)
            for j, (c, sty) in enumerate(icons[:lvl["W"]]):
                pc.put(0, j, c, sty)
            pan = pc.g
            sep = np.full((1, g.shape[1]), PANEL)
            g = np.vstack([pan, sep, g])
        return g
