"""Familles « avatar / grille » (flèches) : labyrinthe, clés-portes, sokoban, collecte, interrupteurs, pastilles de
couleur, glace, téléporteurs, dangers, inclinaison (gravité)."""
from math import lcm

import numpy as np

from .core import (AVATAR, BG, DIRS, DIR_OPS, EV, GOAL, WALL, Canvas, Game, bfs_dist, connected_room, free_cells,
                   perfect_maze, shortest_path)

OPP = {"up": "down", "down": "up", "left": "right", "right": "left"}


class AvatarGame(Game):
    layouts = ("maze", "room")
    t_choices = (1, 2, 3)

    def setup(self, rng):
        self.t_pref = int(rng.choice(self.t_choices))
        self.av_style = str(rng.choice(["full", "dot", "ring", "cross"]))
        self.goal_style = str(rng.choice(["full", "ring", "corners", "cross"]))
        self.wall_style = str(rng.choice(["full", "full", "ring"]))
        self.layout_style = str(rng.choice(self.layouts))
        self.density = float(rng.uniform(0.08, 0.28))
        self.loops = float(rng.uniform(0.0, 0.15))
        self.size_bias = int(rng.integers(0, 3))
        if self.official:                                   # sprites natifs détaillés, murs fins
            r = self.orng
            self.t_pref = self.T
            self.av_style = str(r.choice(["box_dot", "eye", "ring_dot", "diamond"]))
            self.goal_style = str(r.choice(["ring", "frame", "corners", "ring_dot"]))
            self.wall_style = "thin" if self.wall_mode == "thin" else str(r.choice(["full", "small"]))
        self.extra_setup(rng)

    def extra_setup(self, rng):
        pass

    def body(self, lvl, st):
        return (st[0][0], st[0][1], lvl["H"], lvl["W"])

    def step_target(self, lvl, st, op):
        return self.nxt(lvl, st[0], op) if op in DIRS else None

    # ---- disposition
    def make_walls(self, rng, i, n, style=None):
        style = style or self.layout_style
        d = i / max(1, n - 1)
        if style == "maze":
            mc = (self.cap(15 if not self.official else 19) - 1) // 2
            lo, grow = (2, 4) if not self.official else (max(2, mc // 2), mc - max(2, mc // 2))   # officiel : remplit l'écran
            ch = lo + int(round(d * grow)) + int(rng.integers(0, 2)) + self.size_bias // 2
            cw = lo + int(round(d * grow)) + int(rng.integers(0, 2)) + self.size_bias // 2
            return perfect_maze(rng, min(ch, mc), min(cw, mc), self.loops)
        lo = 6 if not self.official else max(6, self.cap(16) * 2 // 3)
        h = lo + int(round(d * 7)) + int(rng.integers(0, 2)) + self.size_bias
        w = lo + int(round(d * 7)) + int(rng.integers(0, 2)) + self.size_bias
        return connected_room(rng, self.cap(min(h, 16)), self.cap(min(w, 16)), self.density)

    def mk(self, walls, **kw):
        H, W = walls.shape
        t = self.t_pref
        lim = 60 if not self.official else 62 // self.s_screen
        while t > 1 and max(H, W) * t > lim:
            t -= 1
        return dict(walls=walls, H=H, W=W, t=t, **kw)

    @staticmethod
    def nxt(lvl, pos, op):
        dr, dc = DIRS[op]
        r, c = pos[0] + dr, pos[1] + dc
        if 0 <= r < lvl["H"] and 0 <= c < lvl["W"] and not lvl["walls"][r, c]:
            return (r, c)
        return None

    def canvas(self, lvl):
        cv = Canvas(lvl["H"], lvl["W"], lvl["t"])
        for r, c in np.argwhere(lvl["walls"]):
            if self.wall_style == "thin":
                cv.thin_wall(lvl["walls"], r, c)
            else:
                cv.put(r, c, WALL, self.wall_style)
        return cv

    def far_pair(self, rng, walls, q=0.6):
        fr = free_cells(walls)
        s = fr[int(rng.integers(len(fr)))]
        d = bfs_dist(walls, s)
        far = [p for p in fr if d[p] >= max(1, q * d.max())]
        return s, far[int(rng.integers(len(far)))]

    def pick(self, rng, cells, exclude):
        c = [p for p in cells if p not in exclude]
        return c[int(rng.integers(len(c)))] if c else None


# ============================================================================ 1. labyrinthe
class Maze(AvatarGame):
    family = "maze"

    def gen_level(self, rng, i, n):
        walls = self.make_walls(rng, i, n)
        s, e = self.far_pair(rng, walls)
        return self.mk(walls, exit=e), (s,), None

    def apply(self, lvl, st, op, px):
        p = self.nxt(lvl, st[0], op)
        return ((p,) if p else st), False

    def is_win(self, lvl, st):
        return st[0] == lvl["exit"]

    def render(self, lvl, st):
        cv = self.canvas(lvl)
        cv.put(*lvl["exit"], GOAL, self.goal_style)
        cv.put(*st[0], AVATAR, self.av_style)
        return cv.g

    def min_len(self, i, n):
        return 3 + 2 * i


# ============================================================================ 2. clés et portes
class KeysDoors(AvatarGame):
    family = "keys_doors"
    layouts = ("maze",)

    def extra_setup(self, rng):
        self.key_style = str(rng.choice(["dot", "cross", "corners", "diag"]))
        self.loops = float(rng.uniform(0, 0.06))

    def gen_level(self, rng, i, n):
        walls = self.make_walls(rng, i, n)
        s, e = self.far_pair(rng, walls, 0.7)
        path = shortest_path(walls, s, e)
        k = 1 + min(2, i // 2)
        if path is None or len(path) < 3 * k + 3:
            return None
        idx = sorted(rng.choice(np.arange(2, len(path) - 1), k, replace=False))
        doors = {path[j]: m for m, j in enumerate(idx)}
        w_all = walls.copy()
        for p in doors:
            w_all[p] = True
        if bfs_dist(w_all, s)[e] >= 0:          # les portes doivent couper le chemin
            return None
        keys = {}
        for m in range(k):
            w2 = walls.copy()
            for p, mm in doors.items():
                if mm >= m:
                    w2[p] = True
            dd = bfs_dist(w2, s)
            reach = [p for p in free_cells(w2) if dd[p] >= 0]
            kp = self.pick(rng, reach, set(doors) | set(keys) | {s, e})
            if kp is None:
                return None
            keys[kp] = m
        return self.mk(walls, exit=e, doors=doors, keys=keys), (s, 0), None

    def apply(self, lvl, st, op, px):
        pos, mask = st
        p = self.nxt(lvl, pos, op)
        if p is None:
            return st, False
        if p in lvl["doors"] and not mask >> lvl["doors"][p] & 1:
            return st, False
        if p in lvl["keys"]:
            mask |= 1 << lvl["keys"][p]
        return (p, mask), False

    def is_win(self, lvl, st):
        return st[0] == lvl["exit"]

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        p = self.step_target(lvl, st, op)
        if p in lvl["keys"] and not st[1] >> lvl["keys"][p] & 1:
            ev[EV["collect"]] = ev[EV["open"]] = ev[EV["touch"]] = 1
        if p in lvl["doors"]:
            ev[EV["touch"]] = 1

    def render(self, lvl, st):
        cv = self.canvas(lvl)
        pos, mask = st
        cv.put(*lvl["exit"], GOAL, self.goal_style)
        for p, m in lvl["doors"].items():
            if not mask >> m & 1:
                cv.put(*p, 4 + m, "full")
        for p, m in lvl["keys"].items():
            if not mask >> m & 1:
                cv.put(*p, 4 + m, self.key_style)
        cv.put(*pos, AVATAR, self.av_style)
        return cv.g

    def min_len(self, i, n):
        return 4 + 2 * i


# ============================================================================ 3. sokoban léger
class Sokoban(AvatarGame):
    family = "sokoban"
    layouts = ("room",)
    bfs_cap = 20000
    bfs_improve = True

    def extra_setup(self, rng):
        self.density = float(rng.uniform(0.05, 0.18))
        self.box_style = str(rng.choice(["full", "ring", "cross"]))

    def gen_level(self, rng, i, n):
        h = w = self.cap(min(6 + i // 2 + int(rng.integers(0, 2)), 10))
        walls = connected_room(rng, h, w, self.density)
        fr = free_cells(walls)
        nb = 1 + min(2, i // 2)
        if len(fr) < nb + 6:
            return None
        tg = [fr[j] for j in rng.choice(len(fr), nb, replace=False)]
        boxes = set(tg)
        pl = self.pick(rng, fr, boxes)
        moves = []
        lvl = self.mk(walls, targets=frozenset(tg))
        for _ in range(12 + 10 * i):                      # tirages inverses (« tirer » les caisses)
            op = DIR_OPS[int(rng.integers(4))]
            p2 = self.nxt(lvl, pl, op)
            if p2 is None or p2 in boxes:
                continue
            dr, dc = DIRS[op]
            back = (pl[0] - dr, pl[1] - dc)
            if back in boxes and rng.random() < 0.8:
                boxes.remove(back)
                boxes.add(pl)
            pl = p2
            moves.append(op)
        if boxes == set(tg):
            return None
        sol = [(OPP[o], None) for o in reversed(moves)]
        return lvl, (pl, tuple(sorted(boxes))), sol

    def apply(self, lvl, st, op, px):
        pl, boxes = st
        p = self.nxt(lvl, pl, op)
        if p is None:
            return st, False
        if p in boxes:
            q = self.nxt(lvl, p, op)
            if q is None or q in boxes:
                return st, False
            boxes = tuple(sorted([b for b in boxes if b != p] + [q]))
        return (p, boxes), False

    def is_win(self, lvl, st):
        return set(st[1]) == lvl["targets"]

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        p = self.step_target(lvl, st, op)
        if st2[1] != st[1]:
            ev[EV["push"]] = ev[EV["touch"]] = 1
        elif p in st[1]:
            ev[EV["touch"]] = 1

    def heuristic(self, lvl, st):
        if "_td" not in lvl:
            lvl["_td"] = [bfs_dist(lvl["walls"], t) for t in lvl["targets"]]
        h = 0
        for b in st[1]:
            ds = [d[b] for d in lvl["_td"] if d[b] >= 0]
            h += min(ds) if ds else 50
        return h

    def table_ok(self, lvl, st0):
        return len(st0[1]) <= 2

    def render(self, lvl, st):
        cv = self.canvas(lvl)
        for p in lvl["targets"]:
            cv.put(*p, GOAL, "full")
        for b in st[1]:
            cv.put(*b, 5 if b in lvl["targets"] else 4, self.box_style)
        cv.put(*st[0], AVATAR, self.av_style)
        return cv.g

    def min_len(self, i, n):
        return 3 + 2 * i


# ============================================================================ 4. tout ramasser (+ sortie optionnelle)
class Collect(AvatarGame):
    family = "collect"

    def extra_setup(self, rng):
        self.exit_mode = bool(rng.random() < 0.5)
        self.item_style = str(rng.choice(["full", "dot", "cross", "diag"]))

    def gen_level(self, rng, i, n):
        walls = self.make_walls(rng, i, n)
        fr = free_cells(walls)
        k = min(2 + i, 6)
        if len(fr) < k + 3:
            return None
        cells = [fr[j] for j in rng.choice(len(fr), k + 2, replace=False)]
        s, e, items = cells[0], cells[1], tuple(cells[2:])
        return self.mk(walls, items=items, exit=e if self.exit_mode else None), (s, 0), None

    def apply(self, lvl, st, op, px):
        pos, mask = st
        p = self.nxt(lvl, pos, op)
        if p is None:
            return st, False
        if p in lvl["items"]:
            mask |= 1 << lvl["items"].index(p)
        return (p, mask), False

    def is_win(self, lvl, st):
        done = st[1] == (1 << len(lvl["items"])) - 1
        return done and (lvl["exit"] is None or st[0] == lvl["exit"])

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        if st2[1] != st[1]:
            ev[EV["collect"]] = ev[EV["touch"]] = 1

    def render(self, lvl, st):
        cv = self.canvas(lvl)
        pos, mask = st
        if lvl["exit"] is not None and mask == (1 << len(lvl["items"])) - 1:
            cv.put(*lvl["exit"], GOAL, self.goal_style)        # la sortie n'apparaît qu'une fois tout ramassé
        for j, p in enumerate(lvl["items"]):
            if not mask >> j & 1:
                cv.put(*p, 4, self.item_style)
        cv.put(*pos, AVATAR, self.av_style)
        return cv.g

    def min_len(self, i, n):
        return 4 + 2 * i


# ============================================================================ 5. boutons -> portails
class Switches(AvatarGame):
    family = "switches"
    layouts = ("maze",)

    def extra_setup(self, rng):
        self.btn_style = str(rng.choice(["full", "dot", "ring", "corners"]))
        self.loops = float(rng.uniform(0, 0.05))

    def gen_level(self, rng, i, n):
        walls = self.make_walls(rng, i, n)
        s, e = self.far_pair(rng, walls, 0.7)
        path = shortest_path(walls, s, e)
        k = 1 + min(2, i // 2)
        if path is None or len(path) < 3 * k + 3:
            return None
        idx = sorted(rng.choice(np.arange(2, len(path) - 1), k, replace=False))
        gates = {path[j]: m for m, j in enumerate(idx)}
        inv = set()
        fr = free_cells(walls)
        if i >= 2:                                 # portails inversés (ouverts au départ, le bouton les ferme)
            for _ in range(int(rng.integers(1, 3))):
                p = self.pick(rng, fr, set(gates) | {s, e})
                gates[p] = int(rng.integers(k))
                inv.add(p)
        buttons = {}
        for m in range(k):
            w2 = walls.copy()
            for p, mm in gates.items():
                if mm >= m and p not in inv:
                    w2[p] = True
            dd = bfs_dist(w2, s)
            reach = [p for p in fr if dd[p] >= 0]
            bp = self.pick(rng, reach, set(gates) | set(buttons) | {s, e})
            if bp is None:
                return None
            buttons[bp] = m
        return self.mk(walls, exit=e, gates=gates, inv=frozenset(inv), buttons=buttons), (s, 0), None

    def closed(self, lvl, p, bits):
        m = lvl["gates"][p]
        return bool(bits >> m & 1) == (p in lvl["inv"])

    def apply(self, lvl, st, op, px):
        pos, bits = st
        p = self.nxt(lvl, pos, op)
        if p is None or (p in lvl["gates"] and self.closed(lvl, p, bits)):
            return st, False
        if p in lvl["buttons"]:
            bits ^= 1 << lvl["buttons"][p]
        return (p, bits), False

    def is_win(self, lvl, st):
        return st[0] == lvl["exit"]

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        p = self.step_target(lvl, st, op)
        if p in lvl["gates"]:
            ev[EV["touch"]] = 1
        if st2[1] != st[1]:
            ev[EV["toggle"]] = ev[EV["touch"]] = 1
            if any(self.closed(lvl, g, st[1]) and not self.closed(lvl, g, st2[1]) for g in lvl["gates"]):
                ev[EV["open"]] = 1

    def render(self, lvl, st):
        cv = self.canvas(lvl)
        pos, bits = st
        cv.put(*lvl["exit"], GOAL, self.goal_style)
        for p, m in lvl["gates"].items():
            cv.put(*p, 4 + m, "full" if self.closed(lvl, p, bits) else "corners")
            if not self.closed(lvl, p, bits) and lvl["t"] == 1:
                cv.put(*p, BG)
        for p, m in lvl["buttons"].items():
            cv.put(*p, 7 + m, self.btn_style)
        cv.put(*pos, AVATAR, self.av_style)
        return cv.g

    def min_len(self, i, n):
        return 4 + 2 * i


# ============================================================================ 6. pastilles de couleur (l'avatar change de couleur)
class ColorPads(AvatarGame):
    family = "color_pads"
    layouts = ("room",)
    t_choices = (3,)

    def gen_level(self, rng, i, n):
        walls = self.make_walls(rng, i, n)
        walls = walls if max(walls.shape) <= 20 else walls[:20, :20]
        fr = free_cells(walls)
        k = 2 + (i >= 3)
        ce = int(rng.integers(k))
        s, e = self.far_pair(rng, walls, 0.5)
        used = {s, e}
        pads = {}
        for c in range(k):
            for _ in range(1 + (i >= 2 and rng.random() < 0.5)):
                p = self.pick(rng, fr, used)
                if p is None:
                    return None
                pads[p] = c
                used.add(p)
        bar = {}
        if i >= 1:
            for _ in range(int(len(fr) * (0.06 + 0.04 * min(i, 4)))):
                p = self.pick(rng, fr, used)
                if p is None:
                    break
                bar[p] = int(rng.integers(k))
                used.add(p)
        return self.mk(walls, exit=e, ce=ce, pads=pads, bar=bar), (s, -1), None

    def apply(self, lvl, st, op, px):
        pos, col = st
        p = self.nxt(lvl, pos, op)
        if p is None:
            return st, False
        if p in lvl["bar"] and lvl["bar"][p] != col:
            return st, False
        if p == lvl["exit"] and col != lvl["ce"]:
            return st, False
        if p in lvl["pads"]:
            col = lvl["pads"][p]
        return (p, col), False

    def is_win(self, lvl, st):
        return st[0] == lvl["exit"]

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        p = self.step_target(lvl, st, op)
        if p in lvl["pads"] or p in lvl["bar"] or p == lvl["exit"]:
            ev[EV["touch"]] = 1
        if st2[1] != st[1]:
            ev[EV["color"]] = 1

    def render(self, lvl, st):
        cv = self.canvas(lvl)
        pos, col = st
        cv.put(*lvl["exit"], GOAL, "full")
        cv.put(*lvl["exit"], 4 + lvl["ce"], "dot")
        for p, c in lvl["pads"].items():
            cv.put(*p, 4 + c, "cross")
        for p, c in lvl["bar"].items():
            cv.put(*p, 4 + c, "full")
        if col < 0:
            cv.put(*pos, AVATAR, "full")
        else:
            cv.put(*pos, 4 + col, "full")
            cv.put(*pos, AVATAR, "dot")
        return cv.g

    def min_len(self, i, n):
        return 4 + 2 * i


# ============================================================================ 7. glace (on glisse jusqu'au blocage)
class Ice(AvatarGame):
    family = "ice"
    layouts = ("room",)

    def extra_setup(self, rng):
        self.exit_stops = bool(rng.random() < 0.5)       # la sortie arrête-t-elle la glissade ?
        self.rough = bool(rng.random() < 0.5)            # cases rugueuses qui arrêtent
        self.density = float(rng.uniform(0.08, 0.2))

    def gen_level(self, rng, i, n):
        h = w = self.cap(min(6 + i + int(rng.integers(0, 3)), 14))
        walls = connected_room(rng, h, w, self.density)
        fr = free_cells(walls)
        if len(fr) < 8:
            return None
        s, e = [fr[j] for j in rng.choice(len(fr), 2, replace=False)]
        rough = set()
        if self.rough:
            for j in rng.choice(len(fr), max(1, len(fr) // 12), replace=False):
                if fr[j] not in (s, e):
                    rough.add(fr[j])
        return self.mk(walls, exit=e, rough=frozenset(rough)), (s,), None

    def apply(self, lvl, st, op, px):
        pos = st[0]
        while True:
            p = self.nxt(lvl, pos, op)
            if p is None:
                break
            pos = p
            if p in lvl["rough"] or (self.exit_stops and p == lvl["exit"]):
                break
        return (pos,), False

    def is_win(self, lvl, st):
        return st[0] == lvl["exit"]

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        if st2[0] in lvl["rough"] and st2 != st:
            ev[EV["touch"]] = 1

    def render(self, lvl, st):
        cv = self.canvas(lvl)
        for p in lvl["rough"]:
            cv.put(*p, 4, "corners")
        cv.put(*lvl["exit"], GOAL, self.goal_style)
        cv.put(*st[0], AVATAR, self.av_style)
        return cv.g

    def min_len(self, i, n):
        return 2 + min(i, 5)


# ============================================================================ 8. téléporteurs
class Teleport(AvatarGame):
    family = "teleport"
    layouts = ("room",)

    def extra_setup(self, rng):
        self.tp_style = str(rng.choice(["ring", "cross", "full", "corners"]))
        self.density = float(rng.uniform(0.05, 0.25))

    def gen_level(self, rng, i, n):
        k = 1 + min(2, i // 2)
        ws = int(rng.integers(3, 6))
        h = min(6 + i + int(rng.integers(0, 2)), 12)
        W = (ws + 1) * (k + 1) + 1
        walls = np.ones((h, W), bool)
        regions = []
        for j in range(k + 1):
            room = connected_room(rng, h, ws + 2, self.density)
            a = j * (ws + 1)
            walls[:, a:a + ws + 2] &= room
            regions.append([(r, c + a) for r, c in free_cells(room)])
        tele, used = {}, set()
        s = self.pick(rng, regions[0], used)
        used.add(s)
        e = self.pick(rng, regions[-1], used)
        used.add(e)
        pairs = []
        for j in range(k):
            a = self.pick(rng, regions[j], used)
            used.add(a)
            b = self.pick(rng, regions[j + 1], used)
            used.add(b)
            pairs.append((a, b))
        for _ in range(int(i >= 2) + int(i >= 4)):           # leurres
            j1, j2 = int(rng.integers(k + 1)), int(rng.integers(k + 1))
            a = self.pick(rng, regions[j1], used)
            used.add(a)
            b = self.pick(rng, regions[j2], used)
            used.add(b)
            pairs.append((a, b))
        if None in used:
            return None
        for a, b in pairs:
            tele[a], tele[b] = b, a
        return self.mk(walls, exit=e, tele=tele, pairs=pairs), (s,), None

    def apply(self, lvl, st, op, px):
        p = self.nxt(lvl, st[0], op)
        if p is None:
            return st, False
        return (lvl["tele"].get(p, p),), False

    def is_win(self, lvl, st):
        return st[0] == lvl["exit"]

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        p = self.step_target(lvl, st, op)
        if p in lvl["tele"]:
            ev[EV["teleport"]] = ev[EV["touch"]] = 1

    def render(self, lvl, st):
        cv = self.canvas(lvl)
        for j, (a, b) in enumerate(lvl["pairs"]):
            cv.put(*a, 4 + j, self.tp_style)
            cv.put(*b, 4 + j, self.tp_style)
        cv.put(*lvl["exit"], GOAL, self.goal_style)
        cv.put(*st[0], AVATAR, self.av_style)
        return cv.g

    def min_len(self, i, n):
        return 3 + 2 * i


# ============================================================================ 9. dangers : lave + ennemis qui patrouillent (GAME_OVER)
class Hazards(AvatarGame):
    family = "hazards"
    layouts = ("room",)

    def extra_setup(self, rng):
        self.enemy_style = str(rng.choice(["full", "cross", "diag"]))
        self.density = float(rng.uniform(0.0, 0.15))

    @staticmethod
    def epos(path, t):
        L = len(path)
        if L == 1:
            return path[0]
        P = 2 * (L - 1)
        k = t % P
        return path[k if k < L else P - k]

    def gen_level(self, rng, i, n):
        walls = self.make_walls(rng, i, n)
        fr = free_cells(walls)
        s, e = self.far_pair(rng, walls, 0.6)
        lava = set()
        for j in rng.choice(len(fr), int(len(fr) * (0.04 + 0.03 * min(i, 4))), replace=False):
            if fr[j] not in (s, e):
                lava.add(fr[j])
        enemies, P = [], 1
        for _ in range(min(i, 4)):
            p0 = self.pick(rng, fr, lava | {s, e})
            op = ("right", "down")[int(rng.integers(2))]
            path = [p0]
            lvl0 = self.mk(walls)
            for _ in range(int(rng.integers(1, 5))):
                q = self.nxt(lvl0, path[-1], op)
                if q is None or q in lava:
                    break
                path.append(q)
            if len(path) < 2 or s in path:
                continue
            enemies.append(tuple(path))
            P = lcm(P, 2 * (len(path) - 1))
        lvl = self.mk(walls, exit=e, lava=frozenset(lava), enemies=enemies, P=P)
        return lvl, (s, 0), None

    def apply(self, lvl, st, op, px):
        pos, t = st
        p = self.nxt(lvl, pos, op) or pos
        t2 = (t + 1) % lvl["P"]
        if p in lvl["lava"]:
            return (p, t2), True
        for path in lvl["enemies"]:
            a, b = self.epos(path, t), self.epos(path, t2)
            if b == p or (b == pos and a == p):
                return (p, t2), True
        return (p, t2), False

    def is_win(self, lvl, st):
        return st[0] == lvl["exit"]

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        (r, c), t2 = st2
        near = dead or any(abs(r - a) + abs(c - b) <= 1 for a, b in lvl["lava"])
        if not near:
            near = any(abs(r - a) + abs(c - b) <= 1 for a, b in (self.epos(pth, t2) for pth in lvl["enemies"]))
        ev[EV["danger"]] = float(near)
        if dead:
            ev[EV["touch"]] = 1

    def render(self, lvl, st):
        cv = self.canvas(lvl)
        pos, t = st
        for p in lvl["lava"]:
            cv.put(*p, 8, "full")
        cv.put(*lvl["exit"], GOAL, self.goal_style)
        for path in lvl["enemies"]:
            cv.put(*self.epos(path, t), 9, self.enemy_style)
        cv.put(*pos, AVATAR, self.av_style)
        return cv.g

    def min_len(self, i, n):
        return 4 + 2 * i


# ============================================================================ 10. inclinaison : tous les blocs glissent ensemble
class Tilt(AvatarGame):
    family = "tilt"
    layouts = ("room",)

    def extra_setup(self, rng):
        self.blk_style = str(rng.choice(["full", "ring", "cross"]))
        self.density = float(rng.uniform(0.1, 0.25))

    def gen_level(self, rng, i, n):
        h = w = self.cap(min(5 + i + int(rng.integers(0, 2)), 11))
        walls = connected_room(rng, h, w, self.density)
        fr = free_cells(walls)
        no = min(i // 2 + int(rng.integers(0, 2)), 3)
        if len(fr) < no + 4:
            return None
        cells = [fr[j] for j in rng.choice(len(fr), no + 2, replace=False)]
        return self.mk(walls, target=cells[0]), (cells[1], tuple(sorted(cells[2:]))), None

    def apply(self, lvl, st, op, px):
        sp, others = st
        blocks = [sp] + list(others)
        dr, dc = DIRS[op]
        order = sorted(range(len(blocks)), key=lambda j: -(blocks[j][0] * dr + blocks[j][1] * dc))
        occ = set(blocks)
        for j in order:
            p = blocks[j]
            occ.discard(p)
            while True:
                q = self.nxt(lvl, p, op)
                if q is None or q in occ:
                    break
                p = q
            occ.add(p)
            blocks[j] = p
        return (blocks[0], tuple(sorted(blocks[1:]))), False

    def is_win(self, lvl, st):
        return st[0] == lvl["target"]

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        if st2[1] != st[1]:
            ev[EV["push"]] = 1
        if st2 != st:
            ev[EV["touch"]] = 1          # les blocs viennent heurter un obstacle

    def render(self, lvl, st):
        cv = self.canvas(lvl)
        cv.put(*lvl["target"], GOAL, self.goal_style)
        for b in st[1]:
            cv.put(*b, 5, self.blk_style)
        cv.put(*st[0], 4, "full")
        return cv.g

    def min_len(self, i, n):
        return 2 + min(i // 2, 3)
