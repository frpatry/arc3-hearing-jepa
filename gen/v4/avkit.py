"""Outils communs aux familles « avatars, corps couplés, agents autonomes » (twins4, npc4, keyattr4, remote4, ghost4).
Ce module n'est PAS une famille (pas de préfixe fam_) : géométrie de cases, recherches exactes, petits dessins.

Recherche « départ le plus dur » (hardest_start) : on part de quelques états-graines, on calcule la fermeture avant
(tous les états atteignables, plafonnée), puis la distance EXACTE au but de chacun par BFS arrière ; on choisit comme
départ un état parmi les plus éloignés. La solution (plus courte) se lit en descendant la distance. Une transition
« remise à zéro vers le départ » ne raccourcit jamais rien si le départ est l'état le plus lointain : on peut donc la
traiter comme un blocage pendant le calcul (lvl["reset"] = None), puis la brancher sur le départ choisi."""
from collections import deque

import numpy as np

from ..core import DIRS
from . import draw
from .draw import R, rect


# ----------------------------------------------------------------------------- géométrie de cases
def layout(AH, AW, C, top=0, bottom=0, left=0, right=0):
    Hc, Wc = (AH - top - bottom) // C, (AW - left - right) // C
    oy = top + (AH - top - bottom - Hc * C) // 2
    ox = left + (AW - left - right - Wc * C) // 2
    return Hc, Wc, oy, ox


def board(AH, AW, C, area, top=0, bottom=0, left=0, right=0, grow=2):
    """plateau de ~area cases (centré) ; petites cartes -> cases plus grandes (jusqu'à C+grow) -> (C, Hc, Wc, oy, ox)."""
    Hm, Wm = (AH - top - bottom) // C, (AW - left - right) // C
    if Hm * Wm > area:
        a = Wm / Hm
        Hc = max(4, min(Hm, int(round((area / a) ** 0.5))))
        Wc = max(4, min(Wm, int(round(area / Hc))))
        C2 = min(C + grow, (AH - top - bottom) // Hc, (AW - left - right) // Wc)
        C = max(C, C2)
    else:
        Hc, Wc = Hm, Wm
    oy = top + (AH - top - bottom - Hc * C) // 2
    ox = left + (AW - left - right - Wc * C) // 2
    return C, Hc, Wc, oy, ox


def make_floor(rng, Hc, Wc, style="rooms", n_rooms=2, need=20, holes=0.0, max_side=None, tries=30, maxc=None):
    """sol connexe (bool Hc×Wc, bord exclu) de need..maxc cases ; style rooms / cave / open ; `holes` = piliers."""
    best, bscore = None, None
    maxc = maxc or Hc * Wc
    for _ in range(tries):
        if style == "cave":
            m = draw.cave_mask(rng, Hc, Wc, fill=float(rng.uniform(0.55, 0.66)), iters=int(rng.integers(2, 4)))
        elif style == "open":
            m = np.ones((Hc, Wc), bool)
        else:
            m = draw.rooms_mask(rng, Hc, Wc, n_rooms, min_side=3, max_side=max_side)
        m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = False
        if holes > 0:
            h = rng.random((Hc, Wc)) < holes
            m &= ~h
        m = draw.largest_component(m)
        n = int(m.sum())
        if need <= n <= maxc:
            return m
        score = (need - n) if n < need else (n - maxc) * 0.5
        if best is None or score < bscore:
            best, bscore = m, score
    return best if best is not None and best.sum() >= max(8, need // 2) else None


def cells(m):
    return [(int(r), int(c)) for r, c in np.argwhere(m)]


def nbrs(m, p):
    H, W = m.shape
    for d in DIRS.values():
        q = (p[0] + d[0], p[1] + d[1])
        if 0 <= q[0] < H and 0 <= q[1] < W and m[q]:
            yield q


def grid_dist(m, src, block=frozenset()):
    """distances BFS 4-voisins sur le sol m depuis src (case ou liste de cases) -> dict."""
    srcs = [src] if isinstance(src, tuple) else list(src)
    d = {s: 0 for s in srcs}
    q = deque(srcs)
    while q:
        p = q.popleft()
        for n in nbrs(m, p):
            if n not in d and n not in block:
                d[n] = d[p] + 1
                q.append(n)
    return d


def chokepoints(m):
    """cases de sol à exactement 2 voisins opposés (couloir) : bons emplacements de portes."""
    out = []
    for p in cells(m):
        nb = list(nbrs(m, p))
        if len(nb) == 2 and (nb[0][0] == nb[1][0] or nb[0][1] == nb[1][1]):
            out.append(p)
    return out


# ----------------------------------------------------------------------------- recherches exactes
def explore(game, lvl, st0, cap):
    """BFS avant -> (depth, prev) ou (None, None) si plus de `cap` états."""
    depth, prev, q = {st0: 0}, {st0: None}, deque([st0])
    while q:
        s = q.popleft()
        for mv in game.moves(lvl, s):
            s2, dead = game.apply(lvl, s, *mv)
            if dead or s2 in depth:
                continue
            depth[s2] = depth[s] + 1
            prev[s2] = (s, mv)
            q.append(s2)
            if len(depth) > cap:
                return None, None
    return depth, prev


def path_to(prev, s):
    sol = []
    while prev[s] is not None:
        s, mv = prev[s]
        sol.append(mv)
    return sol[::-1]


def closure_dist(game, lvl, seeds, cap):
    """fermeture avant depuis les graines + distance exacte au but (BFS arrière) -> (dist, succ) ou (None, None)."""
    seen, q, succ = set(seeds), deque(seeds), {}
    while q:
        s = q.popleft()
        out = []
        for mv in game.moves(lvl, s):
            s2, dead = game.apply(lvl, s, *mv)
            if dead or s2 == s:
                continue
            out.append((mv, s2))
            if s2 not in seen:
                seen.add(s2)
                q.append(s2)
        succ[s] = out
        if len(seen) > cap:
            return None, None
    rev = {}
    for s, out in succ.items():
        for _, s2 in out:
            rev.setdefault(s2, []).append(s)
    dist = {}
    q = deque()
    for s in seen:
        if game.is_win(lvl, s):
            dist[s] = 0
            q.append(s)
    while q:
        s = q.popleft()
        for p in rev.get(s, ()):
            if p not in dist and not game.is_win(lvl, p):
                dist[p] = dist[s] + 1
                q.append(p)
    return dist, succ


def descend(dist, succ, st):
    """solution optimale depuis st en suivant la distance décroissante."""
    sol = []
    while dist[st] > 0:
        for mv, s2 in succ[st]:
            if dist.get(s2, -1) == dist[st] - 1:
                sol.append(mv)
                st = s2
                break
        else:
            return None
    return sol


def pick_far(rng, items, key, frac=0.85):
    """un élément parmi ceux dont key >= frac × max."""
    if not items:
        return None
    m = max(key(x) for x in items)
    far = [x for x in items if key(x) >= frac * m]
    return far[int(rng.integers(len(far)))]


def trace_tags(game, lvl, st0, sol):
    tags, st = set(), st0
    for mv in sol:
        st2, tg = game._step(lvl, st, *mv)
        tags |= tg
        st = st2
    return tags


# ----------------------------------------------------------------------------- petits dessins
DIGITS = {  # chiffres 3×5 (lignes)
    0: ("111", "101", "101", "101", "111"), 1: ("010", "110", "010", "010", "111"), 2: ("111", "001", "111", "100", "111"),
    3: ("111", "001", "111", "001", "111"), 4: ("101", "101", "111", "001", "001"), 5: ("111", "100", "111", "001", "111"),
    6: ("111", "100", "111", "101", "111"), 7: ("111", "001", "010", "010", "010"), 8: ("111", "101", "111", "101", "111"),
    9: ("111", "101", "111", "001", "111")}


def arrow_glyph(d, n=5):
    """flèche n×n (bool) pointant dans la direction d = (dr, dc)."""
    m = np.zeros((n, n), bool)
    h = n // 2
    m[h, :] = True                                   # flèche vers la droite, puis rotation
    for k in range(1, h + 1):
        m[h - k, n - 1 - k] = m[h + k, n - 1 - k] = True
    k = {(0, 1): 0, (-1, 0): 1, (0, -1): 2, (1, 0): 3}[tuple(d)]
    return np.rot90(m, k)


def put_mask(g, m, y, x, c):
    H, W = g.shape
    for yy, xx in np.argwhere(m):
        if 0 <= y + yy < H and 0 <= x + xx < W:
            g[y + yy, x + xx] = R(c)


def digit(g, v, y, x, c):
    rows = DIGITS[int(v) % 10]
    put_mask(g, np.array([[ch == "1" for ch in r] for r in rows]), y, x, c)


def cell_rect(g, lvl, p, c, inset=0):
    C = lvl["C"]
    rect(g, lvl["oy"] + p[0] * C + inset, lvl["ox"] + p[1] * C + inset, C - 2 * inset, C - 2 * inset, c)


def cell_xy(lvl, p, inset=0):
    return lvl["oy"] + p[0] * lvl["C"] + inset, lvl["ox"] + p[1] * lvl["C"] + inset


def cell_of(lvl, px):
    if px is None:
        return None
    C = lvl["C"]
    r, c = (px[0] - lvl["oy"]) // C, (px[1] - lvl["ox"]) // C
    if px[0] >= lvl["oy"] and px[1] >= lvl["ox"] and 0 <= r < lvl["Hc"] and 0 <= c < lvl["Wc"]:
        return (int(r), int(c))
    return None


def cell_center(lvl, p):
    C = lvl["C"]
    return (lvl["oy"] + p[0] * C + C // 2, lvl["ox"] + p[1] * C + C // 2)


def wall_ring(m):
    """cases hors sol voisines (8-voisinage) du sol : contour des pièces."""
    p = np.pad(m, 1)
    H, W = m.shape
    near = np.zeros_like(m)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            near |= p[1 + dy:H + 1 + dy, 1 + dx:W + 1 + dx]
    return near & ~m


def cut_runs(m, max_len=4, min_part=6):
    """segments droits de sol (ligne ou colonne, longueur <= max_len) dont le retrait coupe le sol en deux parties
    d'au moins min_part cases -> [(cases, partie A, partie B)] (portes, barrières)."""
    H, W = m.shape
    out = []
    runs = []
    for r in range(H):
        c = 0
        while c < W:
            if m[r, c]:
                c0 = c
                while c < W and m[r, c]:
                    c += 1
                runs.append([(r, x) for x in range(c0, c)])
            c += 1
    for c in range(W):
        r = 0
        while r < H:
            if m[r, c]:
                r0 = r
                while r < H and m[r, c]:
                    r += 1
                runs.append([(y, c) for y in range(r0, r)])
            r += 1
    for run in runs:
        if not 1 <= len(run) <= max_len:
            continue
        m2 = m.copy()
        for p in run:
            m2[p] = False
        rest = cells(m2)
        if not rest:
            continue
        a = grid_dist(m2, rest[0])
        if len(a) == len(rest) or len(a) < min_part:
            continue
        b = grid_dist(m2, next(p for p in rest if p not in a))
        if len(a) + len(b) == len(rest) and len(b) >= min_part:
            out.append((run, set(a), set(b)))
    return out


def wall_with_gap(rng, m, gap=(1, 2), min_part=6, tries=12):
    """trace un mur droit (ligne ou colonne) à travers le sol, percé d'une ouverture de 1-2 cases (la porte)
    -> (sol modifié, cases de l'ouverture, partie A, partie B) ou None."""
    H, W = m.shape
    for _ in range(tries):
        vert = bool(rng.random() < 0.5)
        k = int(rng.integers(2, (W if vert else H) - 2))
        line = [(r, k) for r in range(H) if m[r, k]] if vert else [(k, c) for c in range(W) if m[k, c]]
        if len(line) < 2:
            continue
        g = int(rng.integers(gap[0], gap[1] + 1))
        j = int(rng.integers(0, len(line)))
        run = [line[j]]
        for q in line[j + 1:]:                                  # ouverture contiguë
            if len(run) >= g:
                break
            if abs(q[0] - run[-1][0]) + abs(q[1] - run[-1][1]) == 1:
                run.append(q)
        m2 = m.copy()
        for p in line:
            if p not in run:
                m2[p] = False
        m3 = m2.copy()
        for p in run:
            m3[p] = False
        rest = cells(m3)
        if not rest:
            continue
        a = grid_dist(m3, rest[0])
        if len(a) == len(rest) or len(a) < min_part:
            continue
        b = grid_dist(m3, next(p for p in rest if p not in a))
        if len(a) + len(b) != len(rest) or len(b) < min_part:
            continue
        if len(grid_dist(m2, rest[0])) != int(m2.sum()):
            continue
        return m2, run, set(a), set(b)
    return None


def prune(mods, bad_pairs):
    """retire les règles anciennes incompatibles avec une plus récente (combinaisons trop coûteuses)."""
    keep = []
    for m in reversed(mods):
        if all({m, k} not in bad_pairs for k in keep):
            keep.append(m)
    return tuple(reversed(keep))


def draw_walls(g, lvl, col, th=2, floor=None):
    """murs FINS : bande de `th` px le long de chaque bord du sol (le sol reste couleur de fond) ;
    col = couleur ou fonction (y, x) -> couleur (murs deux tons)."""
    fl = lvl["floor"] if floor is None else floor
    H, W = fl.shape
    C, oy, ox = lvl["C"], lvl["oy"], lvl["ox"]
    th = max(1, min(th, C // 2))
    cf = col if callable(col) else (lambda y, x: col)

    def solid(r, c):
        return 0 <= r < H and 0 <= c < W and fl[r, c]

    def band(y, x, h, w):
        for yy in range(max(0, y), min(g.shape[0], y + h)):
            for xx in range(max(0, x), min(g.shape[1], x + w)):
                g[yy, xx] = R(cf(yy, xx))
    for r, c in np.argwhere(fl):
        y0, x0 = oy + r * C, ox + c * C
        up, dn, lf, rt = not solid(r - 1, c), not solid(r + 1, c), not solid(r, c - 1), not solid(r, c + 1)
        if up:
            band(y0 - th, x0, th, C)
        if dn:
            band(y0 + C, x0, th, C)
        if lf:
            band(y0, x0 - th, C, th)
        if rt:
            band(y0, x0 + C, C, th)
        for dy, dx, a, b in ((-1, -1, up, lf), (-1, 1, up, rt), (1, -1, dn, lf), (1, 1, dn, rt)):
            if (a or b) and not solid(r + dy, c + dx):
                yy = y0 - th if dy < 0 else y0 + C
                xx = x0 - th if dx < 0 else x0 + C
                band(yy, xx, th, th)
