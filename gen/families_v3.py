"""Familles v3 : jeux plus proches des 25 jeux officiels ARC-AGI-3 (cf. analysis/REPORT.md).

Écart visé (mesuré par gen/measure_v3.py) :
  - niveaux LONGS : solutions oracle de l'ordre de 40-80 actions (officiel : baseline médiane 56, IQR [40, 99]) ;
  - rendu natif (cellule 1 px dans ~75 % des jeux, sprites de 3-5 px), fond uniforme ~70-75 %, murs fins, palette officielle,
    barre de pas d'1 px (100 %), panneau-cible en haut à gauche, vies en option ;
  - UNE RÈGLE NOUVELLE PAR NIVEAU (curriculum tiré par jeu, au plus 3 règles ajoutées actives, la plus ancienne sort) ;
  - mélange de mécaniques proche des 25 officiels : ~36 % avatar, ~32 % clic seul, ~32 % interaction d'objets.

Familles :
  world3   (avatar)  monde à modules v2 (families_world) agrandi : base tirée parmi maze / keys_doors / switches /
                     color_pads / teleport / hazards / ice / collect / quest, un module nouveau par niveau, BFS exact.
  glyph3   (avatar)  glyph_key v2 agrandi (clé-glyphe transformée par des cases), BFS exact.
  tiles3   (clic)    NOUVEAU. Tuiles à p états (p = 2, 3, 5) ; un clic ajoute un POCHOIR (soi, croix, diagonales,
                     miroir, cases inversées, cases verrouillées, trous) ; reproduire la cible en haut à gauche.
                     Solveur EXACT par élimination de Gauss mod p : distance optimale depuis n'importe quel état
                     (progrès exact, oracle exact) ; niveau rejeté si le système n'est pas de rang plein.
  jump3    (clic)    solitaire (capture_merge v2) long : 18-40 sauts ; règles ajoutées : case-cible, sauts en diagonale,
                     2 couleurs (on ne capture que l'autre couleur), trous. Généré à rebours (soluble par construction).
  fleet3   (objets)  NOUVEAU. Sélectionner une pièce (clic, ou ACTION5 qui fait tourner la sélection) puis la déplacer
                     aux flèches ; toutes les pièces sur leur cible de même couleur. Règles ajoutées : barrières de
                     couleur, poussée, paire liée, glissade, lave, portails. Cibles construites par une suite légale de
                     coups (soluble) ; planificateur (BFS par pièce, tous les ordres) = baseline + oracle depuis tout état.
  coupled3 (objets)  coupled_bodies v2 agrandi : un contrôle, 2-3 corps couplés (miroir, inverse, rotation, retard) ;
                     cibles = état atteignable lointain (BFS exhaustif).
Tous : vérification de solvabilité (BFS exact, solveur exact ou construction) ET solution rejouée par apply() jusqu'à la
victoire ; les 23 événements privilégiés (toucher, corps, son, progrès) sont remplis comme dans le reste du générateur."""
import zlib
from collections import deque
from itertools import permutations

import numpy as np

from .core import (BG, HUD_FILL, LETTERBOX, DETAIL, DIRS, DIR_OPS, EV, GOAL, MARK, OBJ2, PANEL, WALL, Game, Unsolvable, bfs_dist,
                   connected_room, free_cells, perfect_maze, sprite)
from .families_avatar import AvatarGame
from .families_v2 import CaptureMerge, CoupledBodies, GlyphKey
from .families_world import World


# ============================================================================ mixin : vue « officielle v3 »
class V3:
    """rendu natif plus fréquent, murs fins, 6-9 niveaux ; chaque solution est REJOUÉE (apply) jusqu'à la victoire."""
    n_levels_official = (6, 9)
    p_screen = (0.75, 0.17, 0.08)                     # échelle ×1, ×2, ×3 (officiel : cellule 1 px dans 68 % des images)
    T_choices = {1: [3, 3, 4, 4, 5], 2: [2, 3, 3], 3: [2, 2]}

    def _official_view(self, r):
        s = int(r.choice([1, 2, 3], p=list(self.p_screen)))
        T = int(r.choice(self.T_choices[s]))
        self.s_screen, self.T = s, T
        self.bar_edge = str(r.choice(["top", "bottom", "left", "right"], p=[0.4, 0.3, 0.15, 0.15]))
        self.energy_death = bool(r.random() < 0.6)
        self.lives_max = int(r.choice([0, 0, 0, 3, 3, 5]))
        self.panel = bool(r.random() < 0.5)
        self.wall_mode = "thin" if r.random() < 0.55 else "block"
        self.sel_style = str(r.choice(["white", "dot"]))
        self.max_cells = max(5, (62 // s) // T)
        self.px_budget = 62 // s - 2

    min_levels = 4

    def _official_presentation(self, r):
        super()._official_presentation(r)
        pal = self.palette
        if pal[BG] == 5 and r.random() < 0.25:        # fond noir ≈ 52 % des officiels (60 % dans la palette v2)
            alt = [c for c in (2, 3, 4) if c not in (pal[WALL], pal[HUD_FILL])]
            nb = int(r.choice(alt))
            if pal[LETTERBOX] == 5:
                pal[LETTERBOX] = nb
            pal[BG] = nb
        if r.random() < 0.75:                         # pourtour = fond (le fond uniforme domine l'écran, ~74 % officiel)
            self.palette[LETTERBOX] = self.palette[BG]

    def _make_level(self, i, n):
        try:
            L = super()._make_level(i, n)
        except Unsolvable:
            if i >= self.min_levels:                  # niveaux 5+ introuvables : jeu tronqué (au moins 4 niveaux)
                return None
            raise
        lvl, st = L["lvl"], L["st0"]
        for mv in L["sol"]:                           # vérification oracle : la solution gagne, sans mourir
            st, dead = self.apply(lvl, st, *mv)
            assert not dead, (self.game_id, i, "solution mortelle")
        assert self.is_win(lvl, st), (self.game_id, i, "solution non gagnante")
        return L

    def info(self):
        d = super().info()
        d["rules_per_level"] = [list(L["lvl"].get("mods", ())) if isinstance(L["lvl"], dict) else []
                                for L in self.levels]
        return d


def _mods(cat, i, keep=3):
    """curriculum : le niveau i ajoute cat[i-1] ; au plus `keep` règles ajoutées actives (la plus ancienne sort)."""
    new = list(cat[:max(0, i)])
    return tuple(new[-keep:]), (new[-1] if new else None)


# ============================================================================ world3 (avatar, modules)
class World3(V3, World):
    family = "world3"
    p_screen = (0.85, 0.15, 0.0)                      # labyrinthes : assez de cases pour des niveaux longs
    T_choices = {1: [3, 4, 4, 4, 5], 2: [2, 2, 3]}
    BASES = ("maze", "keys_doors", "switches", "color_pads", "teleport", "hazards", "ice", "collect", "quest")

    def __init__(self, seed=0, look="official"):
        r = np.random.default_rng([zlib.crc32(b"world3-base"), int(seed)])
        self.base_name = str(r.choice(self.BASES))
        super().__init__(seed, "official")

    def extra_setup(self, rng):
        fam, self.family = self.family, self.base_name
        try:
            super().extra_setup(rng)
        finally:
            self.family = fam
        self.loops = float(self.orng.uniform(0.0, 0.05))

    def gen_level(self, rng, i, n):
        """= World.gen_level avec des labyrinthes plus grands dès le niveau 0 (niveaux longs)."""
        mods, newest = self.active_modules(i)
        fails = self.__dict__.setdefault("_fails", {})
        nf = fails[i] = fails.get(i, -1) + 1
        if nf > 120:
            mods = [m for m in mods if m in (self.base, newest)]
        if nf > 220:
            mods = [m for m in mods if m == newest]
        d = i / max(1, n - 1)
        mc = (self.cap(21) - 1) // 2
        lo = max(3, (2 * mc) // 3)
        ch = min(mc, lo + int(round(d * (mc - lo))) + int(rng.integers(0, 2)))
        cw = min(mc, lo + int(round(d * (mc - lo))) + int(rng.integers(0, 2)))
        loops = self.loops + (0.08 if "ice" in mods else 0)
        if "teleport" in mods:
            loops = min(loops, 0.02)
        walls = perfect_maze(rng, max(2, ch), max(2, cw), loops)
        fr = free_cells(walls)
        s0 = fr[int(rng.integers(len(fr)))]
        dd = bfs_dist(walls, s0)
        far = [p for p in fr if dd[p] >= 0.8 * dd.max()]
        e = far[int(rng.integers(len(far)))]
        L = dict(walls=walls, exit=e, keys={}, doors={}, buttons={}, gates={}, pads={}, cbar={}, lava=frozenset(),
                 enemies=[], P=1, ice=frozenset(), tele={}, pairs=[], boxes0=(), items=(), plates=frozenset(),
                 pgates=frozenset(), mods=tuple(mods))
        used = {s0, e}
        order = sorted(mods, key=lambda m: {"teleport": 0, "collect": 2, "ice": 3, "enemy": 4, "lava": 5}.get(m, 1))
        for m in order:
            if not getattr(self, "_place_" + m)(rng, L, s0, used, d):
                return None
        lvl = self.mk(walls, **{k: v for k, v in L.items() if k != "walls"})
        st0 = (s0, 0, 0, -1, tuple(sorted(L["boxes0"])), 0, 0)
        sol = self.solve(lvl, st0)
        if sol is None:
            return None
        if newest is not None:
            from .families_world import USE_TAG
            tags, st = set(), st0
            for op, px in sol:
                st, _, tg = self._step(lvl, st, op)
                tags |= tg
            if USE_TAG[newest] not in tags:
                return None
        return lvl, st0, sol

    def min_len(self, i, n):
        mc = (self.cap(21) - 1) // 2
        return int(min(34 + 7 * i, 95, 0.45 * (2 * mc + 1) ** 2))


# ============================================================================ glyph3 (avatar, clé-glyphe)
class GlyphKey3(V3, GlyphKey):
    family = "glyph3"
    layouts = ("maze",)
    p_screen = (1.0, 0.0, 0.0)                        # glyphes 3×3 natifs (tuiles >= 3 px)
    T_choices = {1: [3, 4, 4, 5]}

    def extra_setup(self, rng):
        super().extra_setup(rng)
        self.loops = float(rng.uniform(0.0, 0.08))
        self.wall_style = str(self.orng.choice(["thin", "full", "full"]))

    def make_walls(self, rng, i, n, style=None):
        mc = (self.cap(21) - 1) // 2
        d = i / max(1, n - 1)
        lo = max(3, (2 * mc) // 3)
        ch = min(mc, lo + int(round(d * (mc - lo))) + int(rng.integers(0, 2)))
        cw = min(mc, lo + int(round(d * (mc - lo))) + int(rng.integers(0, 2)))
        return perfect_maze(rng, ch, cw, self.loops)

    def min_len(self, i, n):
        mc = (self.cap(21) - 1) // 2
        return int(min(32 + 7 * i, 90, 0.45 * (2 * mc + 1) ** 2))


# ============================================================================ coupled3 (un contrôle, plusieurs corps)
class Coupled3(V3, CoupledBodies):
    family = "coupled3"
    p_screen = (0.85, 0.15, 0.0)
    T_choices = {1: [3, 3, 4, 4, 5], 2: [2, 2, 3]}

    def extra_setup(self, rng):
        super().extra_setup(rng)
        self.goal_mode = "targets"
        self.maze_loops = float(rng.uniform(0.1, 0.3))
        self.wall_style = str(rng.choice(["thin", "full", "small"]))

    def gen_level(self, rng, i, n):
        cps = self.couplings(i)
        fails = self.__dict__.setdefault("_fails", {})
        fails[i] = fails.get(i, -1) + 1
        if fails[i] > 8 and len(cps) > 2:
            cps = cps[:1] + cps[-1:]
        k = len(cps)
        mc = (self.cap(15) - 1) // 2                        # labyrinthe à boucles : longs détours, état joint petit
        ch = min(mc, 4 + i // 2 + int(rng.integers(0, 2))) if k == 2 else 3
        cw = min(mc, 4 + i // 2 + int(rng.integers(0, 2))) if k == 2 else int(rng.integers(3, 5))
        walls = perfect_maze(rng, ch, cw, self.maze_loops)
        fr = free_cells(walls)
        if len(fr) < 2 * k + 6:
            return None
        cells = [fr[j] for j in rng.choice(len(fr), k, replace=False)]
        st0 = (tuple(cells), "none")
        lvl = self.mk(walls, cps=tuple(cps), targets=(), mods=tuple(cps[1:]))
        prev, depth, q = {st0: None}, {st0: 0}, deque([st0])
        while q and len(depth) < 60000:                      # BFS exhaustif de l'état joint (soluble + optimal)
            s_ = q.popleft()
            for o in DIR_OPS:
                s2, _ = self.apply(lvl, s_, o, None)
                if s2 not in depth:
                    depth[s2] = depth[s_] + 1
                    prev[s2] = (s_, o)
                    q.append(s2)
        if q:
            return None
        best = {}                                            # positions -> état le moins profond (retard : `last` varie)
        for s_, dd in depth.items():
            if s_[0] not in best or dd < depth[best[s_[0]]]:
                best[s_[0]] = s_
        dmax = max(depth[s_] for s_ in best.values())
        ml = self.min_len(i, n)
        if dmax < 0.75 * ml:
            return None
        far = [s_ for s_ in best.values() if depth[s_] >= max(0.8 * dmax, min(ml, dmax)) and len(set(s_[0])) == k]
        if not far:
            return None
        s_ = far[int(rng.integers(len(far)))]
        lvl["targets"] = s_[0]
        sol = []
        while prev[s_] is not None:
            s_, o = prev[s_]
            sol.append((o, None))
        return lvl, st0, sol[::-1]

    bfs_cap = 60000
    table_cap = 30000

    def table_ok(self, lvl, st0):
        return True

    def min_len(self, i, n):
        return min(30 + 4 * i, 56)


# ============================================================================ jump3 (solitaire long)
class Jump3(V3, CaptureMerge):
    family = "jump3"
    CAT = ("goal", "diag", "colors", "holes")

    def setup(self, rng):
        super().setup(rng)
        self.px_budget = 60
        self.t = max(3, min(self.T, 4))                     # pions de 3-4 px (objet médian officiel ≈ 9 px)
        self.target_mode = False
        self.cat = [str(x) for x in rng.permutation(self.CAT)]
        self.peg_style = str(rng.choice(["box_dot", "full", "diamond", "eye"]))
        self.col_style = str(rng.choice(["box_dot", "full", "ring", "eye"]))
        self.plate = bool(rng.random() < 0.6)               # plateau plein (cases = trous du fond), sinon points

    def gen_level(self, rng, i, n):
        mods, newest = _mods(self.cat, i)
        diag = "diag" in mods
        dirs = list(DIRS.values()) + ([(1, 1), (1, -1), (-1, 1), (-1, -1)] if diag else [])
        want = int(min(18 + 3 * i, 40) * rng.uniform(1.0, 1.25))
        nb = 7 if want <= 20 else 8 if want <= 28 else 9
        nb = min(nb, (self.px_budget - 2) // (self.t + 1))
        hole_p = 0.2 if "holes" in mods else 0.0
        holes = {(r, c) for r in range(nb) for c in range(nb) if rng.random() >= hole_p}
        if len(holes) < want + 6:
            return None
        last = sorted(holes)[int(rng.integers(len(holes)))]
        two = "colors" in mods
        pegs = {last: int(rng.integers(2)) if two else 0}
        moves = []
        for _ in range(want * 30):
            if len(moves) >= want:
                break
            p = sorted(pegs)[int(rng.integers(len(pegs)))]
            dr, dc = dirs[int(rng.integers(len(dirs)))]
            a, b = (p[0] + dr, p[1] + dc), (p[0] + 2 * dr, p[1] + 2 * dc)
            if a in holes and b in holes and a not in pegs and b not in pegs:
                c = pegs.pop(p)
                pegs[b] = c
                pegs[a] = 1 - c if two else 0
                moves.append((b, p))
        if len(moves) < max(8, want - 6):
            return None
        lvl = dict(nb=nb, holes=frozenset(holes), goal=last if "goal" in mods else None, t=self.t, diag=diag,
                   two=two, mods=mods)
        sol = []
        for src, dst in reversed(moves):
            sol += [("click", self.cell_px(lvl, src)), ("click", self.cell_px(lvl, dst))]
        st0 = (tuple(sorted((r, c, k) for (r, c), k in pegs.items())), None)
        return lvl, st0, sol

    def apply(self, lvl, st, op, px):
        pegs, sel = st
        p = self.which(lvl, px)
        if p is None:
            return st, False
        at = {(r, c): k for r, c, k in pegs}
        if p in at:
            return (pegs, None if sel == p else p), False
        if sel is None:
            return st, False
        dr, dc = p[0] - sel[0], p[1] - sel[1]
        ok = sorted((abs(dr), abs(dc))) == [0, 2] or (lvl["diag"] and abs(dr) == 2 and abs(dc) == 2)
        if not ok:
            return st, False
        mid = (sel[0] + dr // 2, sel[1] + dc // 2)
        if mid not in at or (lvl["two"] and at[mid] == at[sel]):
            return st, False
        k = at[sel]
        new = tuple(sorted([q for q in pegs if q[:2] not in (sel, mid)] + [(p[0], p[1], k)]))
        return (new, None), False

    def is_win(self, lvl, st):
        return len(st[0]) == 1 and (lvl["goal"] is None or st[0][0][:2] == lvl["goal"])

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        if st2[1] != st[1]:
            ev[EV["toggle"]] = 1
            ev[EV["touch"]] = 1
        if len(st2[0]) < len(st[0]):
            ev[EV["collect"]] = ev[EV["touch"]] = ev[EV["push"]] = 1

    def min_len(self, i, n):
        return min(30 + 6 * i, 76)

    def render(self, lvl, st):
        pegs, sel = st
        t, nb = lvl["t"], lvl["nb"]
        g = np.full((2 + nb * (t + 1), 2 + nb * (t + 1)), BG, np.int64)
        if self.plate:
            g[:, :] = WALL
        for (r, c) in lvl["holes"]:
            y, x = 1 + r * (t + 1), 1 + c * (t + 1)
            if self.plate:
                g[y:y + t, x:x + t] = BG
            else:
                g[y + t // 2, x + t // 2] = WALL
            if lvl["goal"] == (r, c):
                g[y:y + t, x] = GOAL
                g[y:y + t, x + t - 1] = GOAL
        for r, c, k in pegs:
            y, x = 1 + r * (t + 1), 1 + c * (t + 1)
            m = sprite(t, self.peg_style if k == 0 else self.col_style)
            blk = g[y:y + t, x:x + t]
            blk[m == 1] = MARK if (r, c) == sel else (4 + k)
            blk[m == 2] = DETAIL
        return g


# ============================================================================ tiles3 (clic, pochoirs mod p, solveur exact)
TILEC = (None, 4, 5, 6, 7)                            # valeur v > 0 -> couleur sémantique ; v = 0 : tuile « éteinte »


def _left_inverse_mod(A, p):
    """A (n×m) sur Z_p, p premier -> M (m×n) avec M A = I_m (mod p), ou None si rang(A) < m."""
    n, m = A.shape
    Aug = np.concatenate([A % p, np.eye(n, dtype=np.int64)], 1)
    row = 0
    for col in range(m):
        piv = next((r for r in range(row, n) if Aug[r, col] % p), None)
        if piv is None:
            return None
        Aug[[row, piv]] = Aug[[piv, row]]
        inv = pow(int(Aug[row, col]), p - 2, p)
        Aug[row] = (Aug[row] * inv) % p
        f = Aug[:, col].copy()
        f[row] = 0
        Aug = (Aug - np.outer(f, Aug[row])) % p
        row += 1
    return Aug[:m, m:]


class Tiles3(V3, Game):
    family = "tiles3"
    simple_ops = ()
    exact_progress = False
    CAT = ("plus", "diag", "locked", "mirror", "reverse", "holes")

    def setup(self, rng):
        self.uses_click = True
        self.px_budget = 60                                     # plateau trop grand pour ×2 : la mise en page repasse à ×1
        self.p = int(rng.choice([2, 3, 5], p=[0.4, 0.45, 0.15]))
        self.base = str(rng.choice(["self", "plus"]))
        cat = [c for c in self.CAT if not (c == "plus" and self.base == "plus") and not (c == "reverse" and self.p == 2)]
        self.cat = [str(x) for x in rng.permutation(cat)]
        self.t = int(min(4, max(2, self.T)))
        self.rs = int(rng.choice([1, 2]))                       # px par tuile dans le panneau-cible
        self.legend = bool(rng.random() < 0.6)                  # icône du pochoir courant à côté de la cible
        self.t = max(3, self.t)
        self.plate = bool(rng.random() < 0.5)               # tuiles posées sur un plateau (un seul blob multicolore)
        self.off_style = str(rng.choice(["ring", "full", "corners"]))

    # ---- géométrie (pixels logiques)
    def _geom(self, nr, nc, rs):
        t = self.t
        top = 1 + nr * rs + 3
        H = top + nr * (t + 1) + 1
        W = 2 + max(nc * (t + 1) - 1, nc * rs + 2 + (7 if self.legend else 0))
        return top, H, W

    def tile_px(self, lvl, k):
        r, c = lvl["cells"][k]
        t = lvl["t"]
        return (lvl["top"] + r * (t + 1) + t // 2, 1 + c * (t + 1) + t // 2)

    def which(self, lvl, px):
        if px is None:
            return None
        t = lvl["t"]
        y, x = px[0] - lvl["top"], px[1] - 1
        if y < 0 or x < 0 or y % (t + 1) >= t or x % (t + 1) >= t:
            return None
        return lvl["index"].get((y // (t + 1), x // (t + 1)))

    def stencil(self, mods):
        st = {(0, 0)}
        if self.base == "plus" or "plus" in mods:
            st |= {(1, 0), (-1, 0), (0, 1), (0, -1)}
        if "diag" in mods:
            st |= {(1, 1), (1, -1), (-1, 1), (-1, -1)}
        return st

    def gen_level(self, rng, i, n):
        mods, newest = _mods(self.cat, i)
        p, t = self.p, self.t
        target_len = int(min(36 + 6 * i, 88) * rng.uniform(1.0, 1.2))
        fill = 0.75
        lock_f = 0.15 if "locked" in mods else 0.0
        hole_f = 0.12 if "holes" in mods else 0.0
        ntile = target_len / ((p - 1) * fill * (1 - lock_f) * (1 - hole_f)) if p == 2 else \
            target_len / (((p - 1) / 2 + 0.5) * fill * (1 - lock_f) * (1 - hole_f))
        M = None
        for _try in range(40):                                # tailles / motifs : jusqu'à un système de rang plein
            side = int(np.clip(np.ceil(np.sqrt(ntile)) + rng.integers(-1, 2), 4, 12))
            nr, nc = side, int(np.clip(np.ceil(ntile / side) + rng.integers(-1, 2), 4, 12))
            rs = self.rs
            while True:
                top, H, W = self._geom(nr, nc, rs)
                if H <= self.px_budget + 1 and W <= self.px_budget + 1:
                    break
                if rs > 1:
                    rs = 1
                elif nr >= nc and nr > 4:
                    nr -= 1
                elif nc > 4:
                    nc -= 1
                else:
                    break
            hole = rng.random((nr, nc)) < hole_f
            hole |= hole[:, ::-1]                            # trous symétriques (gauche-droite)
            cells = [(r, c) for r in range(nr) for c in range(nc) if not hole[r, c]]
            index = {cc: k for k, cc in enumerate(cells)}
            nt = len(cells)
            locked = {k for k in range(nt) if rng.random() < lock_f}
            clickable = [k for k in range(nt) if k not in locked]
            rev = {k for k in range(nt) if "reverse" in mods and rng.random() < 0.2}
            sten = self.stencil(mods)
            A = np.zeros((nt, len(clickable)), np.int64)
            for j, k in enumerate(clickable):
                r, c = cells[k]
                hit = {(r + dr, c + dc) for dr, dc in sten}
                if "mirror" in mods:                             # miroir : c'est la case SYMÉTRIQUE qui bascule
                    hit = (hit - {(r, c)}) | {(r, nc - 1 - c)}
                for cc in hit:
                    if cc in index:
                        A[index[cc], j] = (p - 1) if index[cc] in rev else 1
            M = _left_inverse_mod(A, p)
            if M is not None:
                break
        if M is None:
            return None
        target = np.where(rng.random(nt) < 0.45, 0, rng.integers(1, p, nt))
        x = np.zeros(len(clickable), np.int64)
        order = rng.permutation(len(clickable))
        tot = 0
        for j in order:
            if tot >= target_len:
                break
            v = int(rng.integers(1, p))
            v = min(v, target_len - tot) if target_len - tot < p - 1 else v
            x[j] = v
            tot += v
        start = (target - A @ x) % p
        if (start == target).all():
            return None
        lvl = dict(p=p, t=t, rs=rs, top=top, H=H, W=W, nr=nr, nc=nc, cells=cells, index=index, locked=locked,
                   clickable=clickable, col_of={k: j for j, k in enumerate(clickable)}, rev=rev, A=A, M=M,
                   target=tuple(int(v) for v in target), sten=sten, mirror="mirror" in mods, mods=mods)
        sol = []
        for j in range(len(clickable)):
            sol += [("click", self.tile_px(lvl, clickable[j]))] * int(x[j])
        return lvl, tuple(int(v) for v in start), sol

    def solve_vec(self, lvl, st):
        b = (np.array(lvl["target"]) - np.array(st)) % lvl["p"]
        return (lvl["M"] @ b) % lvl["p"]

    def dist(self, li, st):
        return float(self.solve_vec(self.levels[li]["lvl"], st).sum())

    def table(self, li):
        return None

    def apply(self, lvl, st, op, px):
        k = self.which(lvl, px)
        if k is None or k in lvl["locked"]:
            return st, False
        col = lvl["A"][:, lvl["col_of"][k]]
        return tuple(int(v) for v in (np.array(st) + col) % lvl["p"]), False

    def is_win(self, lvl, st):
        return st == lvl["target"]

    def click_targets(self, lvl, st):
        return [self.tile_px(lvl, k) for k in lvl["clickable"]]

    def click_hit(self, lvl, st, px):
        return self.which(lvl, px) is not None

    def oracle_action(self):
        if self.state != "NOT_FINISHED":
            return 0, None
        lvl = self.levels[self.level]["lvl"]
        y = self.solve_vec(lvl, self.st)
        j = int(np.flatnonzero(y)[0]) if y.any() else 0
        return self._to_action("click", self.tile_px(lvl, lvl["clickable"][j]), self.st)

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        k = self.which(lvl, px)
        if k is not None:
            ev[EV["touch"]] = 1
        if st2 != st:
            ev[EV["toggle"]] = ev[EV["color"]] = 1
            tg = lvl["target"]
            if sum(a == b for a, b in zip(st2, tg)) > sum(a == b for a, b in zip(st, tg)):
                ev[EV["collect"]] = 1

    def min_len(self, i, n):
        return min(30 + 5 * i, 72)

    def render(self, lvl, st):
        t, rs, top = lvl["t"], lvl["rs"], lvl["top"]
        g = np.full((lvl["H"], lvl["W"]), BG, np.int64)
        nr, nc = lvl["nr"], lvl["nc"]
        # panneau-cible (haut gauche), encadré
        g[0, 0:nc * rs + 2] = PANEL
        g[1 + nr * rs, 0:nc * rs + 2] = PANEL
        g[0:2 + nr * rs, 0] = PANEL
        g[0:2 + nr * rs, nc * rs + 1] = PANEL
        for k, (r, c) in enumerate(lvl["cells"]):
            v = lvl["target"][k]
            if v:
                g[1 + r * rs:1 + (r + 1) * rs, 1 + c * rs:1 + (c + 1) * rs] = TILEC[v]
        if self.legend:                                      # icône du pochoir (3×3 ou 5×5 si miroir)
            x0, y0 = nc * rs + 4, 1
            for dr, dc in lvl["sten"]:
                g[y0 + 1 + dr, x0 + 1 + dc] = MARK
            if lvl["mirror"]:
                g[y0:y0 + 3, x0 + 4] = PANEL
        if self.plate:
            g[top - 1:top + nr * (t + 1), 0:nc * (t + 1) + 1] = WALL
        for k, (r, c) in enumerate(lvl["cells"]):
            y, x = top + r * (t + 1), 1 + c * (t + 1)
            v = st[k]
            blk = g[y:y + t, x:x + t]
            if v:
                blk[:] = TILEC[v]
            elif self.plate:
                blk[:] = BG
            else:
                m = sprite(t, self.off_style)
                blk[m >= 1] = PANEL
            if k in lvl["locked"]:
                blk[t // 2, t // 2] = DETAIL
            if k in lvl["rev"]:
                blk[0, 0] = DETAIL
        return g


# ============================================================================ fleet3 (sélectionner puis déplacer)
PIECE = (4, 5, 6, 7)
LAVA3, PORTAL3, ICE3 = 8, OBJ2[0], OBJ2[1]


class Fleet3(V3, AvatarGame):
    family = "fleet3"
    layouts = ("room",)
    exact_progress = False
    CAT = ("gates", "push", "pair", "slide", "lava", "portal")

    def extra_setup(self, rng):
        self.uses_click = True
        self.cycle = bool(rng.random() < 0.3)                # ACTION5 fait tourner la sélection
        if self.cycle:
            self.simple_ops = DIR_OPS + ("interact",)
        self.cat = [str(x) for x in rng.permutation(self.CAT)]
        self.density = float(rng.uniform(0.1, 0.22))
        self.piece_style = str(rng.choice(["box_dot", "small", "diamond", "eye"]))
        self.tgt_style = str(rng.choice(["corners", "frame", "ring"]))
        self.wall_style = str(rng.choice(["full", "full", "small", "thin"]))   # pièces : murs pleins surtout

    # ---- règles
    def _free(self, lvl, q, j, pos):
        if not (0 <= q[0] < lvl["H"] and 0 <= q[1] < lvl["W"]) or lvl["walls"][q]:
            return False
        g = lvl["gates"].get(q)
        if g is not None and g != j:
            return False
        return q not in pos

    def _move1(self, lvl, pos, j, d, allow_push):
        """déplace la pièce j d'un pas (glissade, poussée, portail) -> (pos', dead, tags) ou None si bloquée."""
        pos = list(pos)
        tags = set()
        q = (pos[j][0] + d[0], pos[j][1] + d[1])
        others = [p for k, p in enumerate(pos) if k != j]
        if q in others and allow_push and "push" in lvl["mods"]:
            k = pos.index(q)
            q2 = (q[0] + d[0], q[1] + d[1])
            if not self._free(lvl, q2, k, pos):
                return None
            pos[k] = q2
            tags.add("push")
            if q2 in lvl["lava"]:
                return tuple(pos), True, tags
        elif not self._free(lvl, q, j, others):
            return None
        pos[j] = q
        while q in lvl["ice"] and q not in lvl["lava"] and q not in lvl["portal"]:     # glace : on glisse
            q2 = (q[0] + d[0], q[1] + d[1])
            if not self._free(lvl, q2, j, [p for k, p in enumerate(pos) if k != j]):
                break
            q = pos[j] = q2
            tags.add("slide")
        if q in lvl["portal"]:
            dst = lvl["portal"][q]
            if dst not in pos:
                pos[j] = dst
                tags.add("tele")
        if lvl["gates"].get(pos[j]) == j:
            tags.add("gate")
        return tuple(pos), pos[j] in lvl["lava"], tags

    def _step(self, lvl, st, op, px):
        pos, sel = st
        tags = set()
        if op == "click":
            if px is None:
                return st, False, tags
            T = lvl["t"]
            cell = (px[0] // T, px[1] // T)
            if cell in pos:
                j = pos.index(cell)
                if j != sel:
                    tags.add("select")
                return (pos, j), False, tags
            return st, False, tags
        if op == "interact":
            tags.add("select")
            return (pos, (sel + 1) % len(pos)), False, tags
        if op not in DIRS or sel < 0:
            return st, False, tags
        d = DIRS[op]
        r = self._move1(lvl, pos, sel, d, True)
        if r is None:
            tags.add("bump")
            return st, False, tags
        pos2, dead, tg = r
        tags |= tg
        if not dead and "pair" in lvl["mods"] and sel in (0, 1):
            o = 1 - sel
            r2 = self._move1(lvl, pos2, o, d, False)
            if r2 is not None:
                pos2, dead, tg2 = r2
                tags |= tg2 | {"pair"}
        return (pos2, sel), dead, tags

    def apply(self, lvl, st, op, px):
        st2, dead, _ = self._step(lvl, st, op, px)
        return st2, dead

    def is_win(self, lvl, st):
        return st[0] == lvl["targets"]

    # ---- sélection / planification
    def _select_moves(self, lvl, st, j):
        """coups pour sélectionner la pièce j depuis st."""
        if st[1] == j:
            return []
        if self.cycle and not self.uses_click:
            return [("interact", None)] * ((j - st[1]) % len(st[0]))
        return [("click", self.piece_px(lvl, st[0][j]))]

    def piece_px(self, lvl, p):
        T = lvl["t"]
        return (p[0] * T + T // 2, p[1] * T + T // 2)

    def _bfs_piece(self, lvl, st, j, goal, cap=4000):
        """BFS (flèches seulement, pièce j sélectionnée) jusqu'à un état où goal(st) est vrai."""
        if goal(st):
            return []
        prev, q = {st: None}, deque([st])
        while q:
            s = q.popleft()
            for o in DIR_OPS:
                s2, dead = self.apply(lvl, s, o, None)
                if dead or s2 in prev:
                    continue
                prev[s2] = (s, o)
                if goal(s2):
                    path = []
                    while prev[s2] is not None:
                        s2, o2 = prev[s2]
                        path.append((o2, None))
                    return path[::-1]
                q.append(s2)
            if len(prev) > cap:
                return None
        return None

    def plan(self, lvl, st, max_orders=24):
        """meilleur plan séquentiel (une pièce après l'autre, les pièces placées restent en place), tous les ordres."""
        tg = lvl["targets"]
        k = len(tg)
        best = None
        for n_o, order in enumerate(permutations(range(k))):
            if n_o >= max_orders:
                break
            s, seq, done, ok = st, [], [], True
            for j in order:
                if s[0][j] == tg[j] and all(s[0][x] == tg[x] for x in done):
                    done.append(j)
                    continue
                for mv in self._select_moves(lvl, s, j):
                    s, _ = self.apply(lvl, s, *mv)
                    seq.append(mv)
                dn = tuple(done) + (j,)
                path = self._bfs_piece(lvl, s, j, lambda x, dn=dn: all(x[0][y] == tg[y] for y in dn))
                if path is None:
                    ok = False
                    break
                for mv in path:
                    s, _ = self.apply(lvl, s, *mv)
                seq += path
                done.append(j)
                if best is not None and len(seq) >= len(best):
                    ok = False
                    break
            if ok and self.is_win(lvl, s) and (best is None or len(seq) < len(best)):
                best = seq
        return best

    def gen_level(self, rng, i, n):
        mods, newest = _mods(self.cat, i)
        k = 3 + int(i >= 3 and rng.random() < 0.6)
        mc = self.cap(15)
        h = self.cap(min(mc, 11 + i // 2 + int(rng.integers(0, 3))))
        w = self.cap(min(mc, 11 + i // 2 + int(rng.integers(0, 3))))
        walls = connected_room(rng, h, w, self.density)
        if walls is None:
            return None
        fr = free_cells(walls)
        if len(fr) < 4 * k + 12:
            return None
        cells = [fr[j] for j in rng.choice(len(fr), k, replace=False)]
        used = set(cells)
        lvl = self.mk(walls, gates={}, lava=frozenset(), portal={}, ice=frozenset(), mods=mods, targets=())
        rest = [p for p in fr if p not in used]
        rng.shuffle(rest)
        if "gates" in mods:                                  # barrières de couleur : seule la pièce de même couleur passe
            for j in range(min(k, 2 + (i > 3))):
                if rest:
                    lvl["gates"][rest.pop()] = j
        if "lava" in mods:
            lvl["lava"] = frozenset(rest.pop() for _ in range(min(len(rest), 3 + i // 2)))
        if "slide" in mods:                                  # plaques de glace (amas autour de quelques germes)
            ice = set()
            for sd in rest[:3]:
                ice |= {q for q in fr if abs(q[0] - sd[0]) + abs(q[1] - sd[1]) <= 2 and q not in used
                        and q not in lvl["gates"] and q not in lvl["lava"]}
            lvl["ice"] = frozenset(ice)
            rest = [q for q in rest if q not in ice]
        if "portal" in mods and len(rest) >= 2:
            a, b = rest.pop(), rest.pop()
            lvl["portal"] = {a: b, b: a}
        special = set(lvl["gates"]) | set(lvl["lava"]) | set(lvl["portal"]) | set(lvl["ice"])
        # construction : chaque pièce (ordre tiré) est emmenée loin par une suite légale de coups
        st = (tuple(cells), -1)
        cons = []
        for j in rng.permutation(k):
            j = int(j)
            for mv in self._select_moves(lvl, st, j):
                st, _ = self.apply(lvl, st, *mv)
                cons.append(mv)
            prev, q = {st: None}, deque([st])
            depth = {st: 0}
            while q and len(prev) < 3000:
                s = q.popleft()
                for o in DIR_OPS:
                    s2, dead = self.apply(lvl, s, o, None)
                    if dead or s2 in prev:
                        continue
                    prev[s2] = (s, o)
                    depth[s2] = depth[s] + 1
                    q.append(s2)
            ok = [s for s in depth if not (set(s[0]) & special) and s[0][j] != st[0][j]]
            if not ok:
                return None
            dm = max(depth[s] for s in ok)
            far = [s for s in ok if depth[s] >= 0.7 * dm]
            s2 = far[int(rng.integers(len(far)))]
            path = []
            while prev[s2] is not None:
                s2, o = prev[s2]
                path.append((o, None))
            for mv in path[::-1]:
                st, _ = self.apply(lvl, st, *mv)
            cons += path[::-1]
        tg = st[0]
        if any(a == b for a, b in zip(tg, cells)) or set(tg) & special:
            return None
        lvl["targets"] = tg
        st0 = (tuple(cells), -1)
        fails = self.__dict__.setdefault("_fails", {})
        fails[i] = fails.get(i, 0) + 1
        if len(cons) < 0.8 * self.min_len(i, n) * max(0.3, 1 - fails[i] / 100):   # le plan ne fera que raccourcir
            return None
        pl = self.plan(lvl, st0, max_orders=12)
        sol = pl if (pl is not None and len(pl) <= len(cons)) else cons
        if newest is not None and fails[i] < 40:             # la règle nouvelle doit servir (relâché après 40 essais)
            tags, s = set(), st0
            for mv in sol:
                s, _, tgs = self._step(lvl, s, *mv)
                tags |= tgs
            need = dict(gates="gate", push="push", pair="pair", slide="slide", lava=None, portal="tele")[newest]
            if need is not None and need not in tags:
                return None
        return lvl, st0, sol

    def heuristic(self, lvl, st):
        if "_dm" not in lvl:                                 # distances (pièce seule) vers sa cible, par pièce
            lvl["_dm"] = []
            for j, t in enumerate(lvl["targets"]):
                w = lvl["walls"].copy()
                for q, gj in lvl["gates"].items():
                    if gj != j:
                        w[q] = True
                for q in lvl["lava"]:
                    w[q] = True
                lvl["_dm"].append(bfs_dist(w, t))
        h = 0.0
        for j, p in enumerate(st[0]):
            if p != lvl["targets"][j]:
                v = lvl["_dm"][j][p]
                h += (v if v >= 0 else 30) + (st[1] != j)
        return h

    def oracle_action(self):
        if self.state != "NOT_FINISHED":
            return 0, None
        L = self.levels[self.level]
        lvl, st = L["lvl"], self.st
        if "sol_idx" not in L:
            idx, s = {}, L["st0"]
            for k, (op, px) in enumerate(L["sol"]):
                idx[s] = k
                s, _ = self.apply(lvl, s, op, px)
            L["sol_idx"] = idx
        k = L["sol_idx"].get(st)
        if k is not None:
            return self._to_action(*L["sol"][k], st)
        pc = getattr(self, "_plan_cache", None)
        if pc is None or pc[0] != (self.level, st) or not pc[1]:
            pl = self.plan(lvl, st, max_orders=6)
            if not pl:
                return 0, None
            pc = self._plan_cache = [(self.level, st), list(pl)]
        mv = pc[1].pop(0)
        s2, _ = self.apply(lvl, st, *mv)
        pc[0] = (self.level, s2)
        return self._to_action(*mv, st)

    def body(self, lvl, st):
        if st[1] < 0:
            return None
        p = st[0][st[1]]
        return (p[0], p[1], lvl["H"], lvl["W"])

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        _, _, tags = self._step(lvl, st, op, px)
        if "push" in tags:
            ev[EV["push"]] = 1
        if tags & {"push", "pair", "bump", "gate"} or dead:
            ev[EV["touch"]] = 1
        if "select" in tags:
            ev[EV["toggle"]] = 1
        if "gate" in tags:
            ev[EV["color"]] = 1
        if "tele" in tags:
            ev[EV["teleport"]] = 1
        tg = lvl["targets"]
        if sum(a == b for a, b in zip(st2[0], tg)) > sum(a == b for a, b in zip(st[0], tg)):
            ev[EV["open"]] = 1
        if dead or any(abs(p[0] - a) + abs(p[1] - b) <= 1 for p in st2[0] for a, b in lvl["lava"]):
            ev[EV["danger"]] = 1

    def min_len(self, i, n):
        return min(38 + 4 * i, 66)

    def render(self, lvl, st):
        pos, sel = st
        cv = self.canvas(lvl)
        T = lvl["t"]
        for q in lvl["ice"]:
            cv.put(*q, ICE3, "corners" if T >= 3 else "dot")
        for q in lvl["lava"]:
            cv.put(*q, LAVA3, "full")
        for q in lvl["portal"]:
            cv.put(*q, PORTAL3, "ring_dot" if T >= 3 else "ring")
        for q, j in lvl["gates"].items():
            cv.put(*q, PIECE[j], "cross")
        for j, q in enumerate(lvl["targets"]):
            cv.put(*q, PIECE[j], self.tgt_style if T >= 3 else "dot")
        for j, q in enumerate(pos):
            if j == sel and self.sel_style == "white":
                cv.put(*q, MARK, self.piece_style)
            else:
                cv.put(*q, PIECE[j], self.piece_style, MARK if j == sel else DETAIL)
                if j == sel:
                    cv.g[q[0] * T + T // 2, q[1] * T + T // 2] = MARK
        if "pair" in lvl["mods"] and T >= 3:                 # la paire liée porte un point de détail commun
            for q in pos[:2]:
                cv.g[q[0] * T, q[1] * T] = DETAIL
        return cv.g


FAMILY_CLASSES_V3 = {c.family: c for c in (World3, GlyphKey3, Tiles3, Jump3, Fleet3, Coupled3)}
MECHANIC_V3 = dict(world3="avatar", glyph3="avatar", tiles3="click", jump3="click", fleet3="object", coupled3="object")
# proportions visées (25 officiels : ~9 avatar, ~8 clic seul, ~8 interaction d'objets / contrôle indirect)
WEIGHTS_V3 = dict(world3=0.26, glyph3=0.10, tiles3=0.20, jump3=0.12, fleet3=0.20, coupled3=0.12)
