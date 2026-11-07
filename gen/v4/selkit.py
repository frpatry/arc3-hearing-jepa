"""selkit : outils partagés des familles v4 « sélectionner puis déplacer » (flow4, cover4, lines4).
  bfs_path   : plus court chemin d'UNE pièce (les autres immobiles) sous un prédicat de validité ;
  plan_moves : ordre de déplacement des pièces vers leurs positions-cibles (essaie plusieurs ordres) ;
  RotView    : scène canonique (gravité vers le bas) tournée de k quarts de tour à l'écran : pixels, clics, flèches.
Rien ici n'est propre à un jeu officiel."""
from collections import deque
from itertools import permutations

import numpy as np

DV = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def bfs_path(start, goal, ok, dirs=DV, cap=6000):
    """-> liste de noms de directions menant start -> goal (cases toutes valides selon ok), ou None."""
    if start == goal:
        return []
    prev = {start: None}
    q = deque([start])
    while q:
        p = q.popleft()
        for name, (dr, dc) in dirs.items():
            n = (p[0] + dr, p[1] + dc)
            if n in prev or not ok(n):
                continue
            prev[n] = (p, name)
            if n == goal:
                path = []
                while prev[n] is not None:
                    n0, nm = prev[n]
                    path.append(nm)
                    n = n0
                return path[::-1]
            q.append(n)
            if len(prev) > cap:
                return None
    return None


def plan_moves(pos, targets, ok_fn, first=None, max_orders=30, waypoints=None):
    """pos, targets : listes de positions (r, c) ; ok_fn(k, p, cur) -> bool (pièce k en p, autres en cur).
    waypoints : {k: [positions à visiter avant la cible]} (ex. passer sur un bac de peinture).
    -> [(k, [directions])] dans l'ordre de déplacement, ou None si aucun ordre essayé ne marche."""
    todo = [k for k in range(len(pos)) if pos[k] != targets[k] or (waypoints and waypoints.get(k))]
    if not todo:
        return []
    base = sorted(todo, key=lambda k: (k != first, k))
    orders = [tuple(base)]
    if len(todo) > 1:
        for o in permutations(base):
            if o != orders[0]:
                orders.append(o)
            if len(orders) >= max_orders:
                break
    for order in orders:
        cur, out, good = list(pos), [], True
        for k in order:
            path, p = [], cur[k]
            for goal in list((waypoints or {}).get(k, [])) + [targets[k]]:
                seg = bfs_path(p, goal, lambda q, k=k: ok_fn(k, q, cur))
                if seg is None:
                    good = False
                    break
                path += seg
                p = goal
            if not good:
                break
            out.append((k, path))
            cur[k] = targets[k]
        if good:
            return out
    return None


def _rot_vec(v, k):
    dr, dc = v
    for _ in range(k % 4):
        dr, dc = -dc, dr                 # np.rot90 (sens trigo) : (r, c) -> (W-1-c, r)
    return (dr, dc)


class RotView:
    """image canonique (Hi, Wi) tournée de k quarts de tour (np.rot90) et posée en (oy, ox) dans l'aire de rendu."""

    def __init__(self, Hi, Wi, k, oy, ox):
        self.Hi, self.Wi, self.k, self.oy, self.ox = Hi, Wi, k % 4, oy, ox
        idx = np.rot90(np.arange(Hi * Wi).reshape(Hi, Wi), self.k)
        self.idx = idx
        self.sh = idx.shape
        inv = np.zeros((Hi * Wi, 2), np.int64)
        yy, xx = np.mgrid[:idx.shape[0], :idx.shape[1]]
        inv[idx.ravel(), 0] = yy.ravel()
        inv[idx.ravel(), 1] = xx.ravel()
        self.inv = inv
        self.c2s = {n: _rot_vec(v, self.k) for n, v in DV.items()}           # direction canonique -> écran
        vs = {v: n for n, v in DV.items()}
        self.s2c = {vs[self.c2s[n]]: n for n in DV}                          # flèche écran -> direction canonique

    def paste(self, g, img):
        r = np.rot90(img, self.k)
        g[self.oy:self.oy + r.shape[0], self.ox:self.ox + r.shape[1]] = r

    def to_canon(self, px):
        """pixel de rendu -> pixel canonique (py, px) ou None."""
        if px is None:
            return None
        y, x = px[0] - self.oy, px[1] - self.ox
        if not (0 <= y < self.sh[0] and 0 <= x < self.sh[1]):
            return None
        q = int(self.idx[y, x])
        return (q // self.Wi, q % self.Wi)

    def to_screen(self, cp):
        y, x = self.inv[cp[0] * self.Wi + cp[1]]
        return (int(y) + self.oy, int(x) + self.ox)

    def gravity_rot(self):
        """quarts de tour à appliquer à un sprite « flèche vers le haut » pour montrer la gravité (canonique = bas)."""
        v = self.c2s["down"]
        return {(-1, 0): 0, (0, -1): 1, (1, 0): 2, (0, 1): 3}[v]
