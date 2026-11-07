"""
AGENT ARC-AGI-3 v1 — « le bébé curieux » (principes du projet JEPA, pas ses poids) :
  1 PERCEVOIR DES OBJETS : composantes connexes de même couleur (la perception géométrique d'abord, leçon Push-T) ;
    les clics ne visent que des objets (centre d'une composante), jamais un pixel au hasard.
  2 ÉTATS = images SANS le compteur : les lignes/colonnes du HUD (barre d'énergie, compteur d'actions) changent à presque
    chaque pas -> repérées en ligne et masquées, sinon chaque image serait « nouvelle » et l'exploration tournerait en rond.
  3 EXPLORER AVEC CURIOSITÉ, SANS GASPILLER : graphe état -> action -> état ; d'abord les (état, action) jamais essayés,
    dans l'ordre de ce que l'agent a APPRIS sur les actions (taux d'effet par type d'action / couleur d'objet cliqué :
    « ACTION3 fait bouger quelque chose », « cliquer le rouge ne fait rien ») ; quand tout est essayé ici, il retourne
    (chemin le plus court dans le graphe) vers l'état connu le plus proche qui a encore de l'inconnu.
  4 TRANSFERT ENTRE NIVEAUX : au niveau suivant, le graphe repart de zéro mais ce qu'il sait des actions est gardé.
  RESET est gratuit (ne compte pas) : utilisé après GAME_OVER ; les transitions vers GAME_OVER sont marquées dangereuses.
Indépendant du framework : Explorer.act(grid, state_name, levels_completed, available_actions) -> (action_id, data).
"""
import hashlib
from collections import deque, defaultdict
import numpy as np

BIG = 4          # un changement de plus de BIG pixels (hors HUD) = « l'action a fait quelque chose »


def components(g, bg_frac=0.25, max_n=48):
    """objets = composantes 4-connexes de même couleur (hors fond géant) -> [(couleur, taille, (y, x) à cliquer)]."""
    H, W = g.shape; seen = np.zeros((H, W), bool); out = []
    for y0 in range(H):
        for x0 in range(W):
            if seen[y0, x0]: continue
            col = g[y0, x0]; q = [(y0, x0)]; seen[y0, x0] = True; pix = []
            while q:
                y, x = q.pop(); pix.append((y, x))
                for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                    if 0 <= yy < H and 0 <= xx < W and not seen[yy, xx] and g[yy, xx] == col:
                        seen[yy, xx] = True; q.append((yy, xx))
            if len(pix) > bg_frac * H * W: continue                       # fond
            P = np.array(pix); cy, cx = P.mean(0); k = int(((P - (cy, cx)) ** 2).sum(1).argmin())
            out.append((int(col), len(pix), (int(P[k, 0]), int(P[k, 1]))))
    out.sort(key=lambda o: o[1])                                            # petits objets d'abord (boutons, pièces)
    uniq, keys = [], set()
    for o in out:                                                           # un clic par objet distinct
        kk = (o[0], o[2][0] // 2, o[2][1] // 2)
        if kk not in keys: keys.add(kk); uniq.append(o)
    return uniq[:max_n]


class Explorer:
    def __init__(self, seed=0, coarse=0):
        self.rng = np.random.default_rng(seed); self.coarse = coarse      # coarse=c : états = image RÉDUITE (cases c×c), façon Go-Explore
        self.eff = defaultdict(lambda: [1.0, 1.0])     # type d'action -> [nb d'effets, nb d'essais] (optimiste au départ)
        self.new_level(None)
        self.n_steps = 0

    # ---------- mémoire d'un niveau
    def new_level(self, grid):
        self.G = {}                                     # état -> {action: état suivant | None (pas essayé) | "GO"}
        self.cands = {}                                 # état -> [(action_id, data, type)]
        self.row_chg = None; self.col_chg = None; self.n_obs = 0
        self.mask_rows = np.zeros(0, bool); self.mask_cols = np.zeros(0, bool)
        self.prev = None; self.plan = deque()
        self.last = None                                # (état, action) joué au pas précédent

    def update_hud(self, a, b):
        """lignes / colonnes qui changent (un peu) à presque chaque pas = compteur, barre d'énergie -> masquées."""
        if a.shape != b.shape: return
        d = a != b
        if self.row_chg is None or self.row_chg.shape[0] != d.shape[0]:
            self.row_chg = np.zeros(d.shape[0]); self.col_chg = np.zeros(d.shape[1]); self.n_obs = 0
        self.n_obs += 1
        self.row_chg += (d.sum(1) > 0) & (d.sum(1) <= 3); self.col_chg += (d.sum(0) > 0) & (d.sum(0) <= 3)
        if self.n_obs >= 6:
            self.mask_rows = self.row_chg / self.n_obs > 0.7; self.mask_cols = self.col_chg / self.n_obs > 0.7

    def key(self, g):
        h = g.copy()
        if self.mask_rows.shape[0] == g.shape[0]: h[self.mask_rows, :] = 0
        if self.mask_cols.shape[0] == g.shape[1]: h[:, self.mask_cols] = 0
        if self.coarse > 1:                                                   # écrans semblables = même « case » : l'exploration couvre au lieu de se noyer
            c = self.coarse; h = np.stack([h[::c, ::c], h[c // 2::c, c // 2::c]])
        return hashlib.md5(np.ascontiguousarray(h).tobytes()).hexdigest()

    def candidates(self, g, avail):
        c = [(a, None, f"A{a}") for a in avail if a not in (0, 6)]
        if 6 in avail:
            for col, size, (y, x) in components(g):
                c.append((6, (x, y), f"C{col}"))
        return c

    def prior(self, typ):
        e, n = self.eff[typ]; return e / n

    # ---------- un pas
    def act(self, grid, state, levels, avail):
        g = np.asarray(grid, dtype=np.int16)
        if state in ("NOT_PLAYED", "GAME_OVER"):
            if state == "GAME_OVER" and self.last is not None:
                s, a = self.last; self.G[s][a] = "GO"
            self.last = None; self.plan.clear(); self.prev = None
            return 0, None                                                    # RESET (gratuit)
        if getattr(self, "levels", None) is not None and levels > self.levels: self.new_level(g)
        self.levels = levels
        if self.prev is not None: self.update_hud(self.prev, g)
        k = self.key(g)
        if self.last is not None:                                             # ce qu'a fait l'action précédente
            s, a = self.last; typ = a[2]
            moved = self.prev is not None and self.prev.shape == g.shape and int(self.masked_diff(self.prev, g)) > BIG
            self.eff[typ][0] += moved; self.eff[typ][1] += 1
            if s in self.G: self.G[s][a] = k
        if k not in self.G:
            cs = self.candidates(g, avail); self.cands[k] = cs; self.G[k] = {c: None for c in cs}
        self.prev = g; self.n_steps += 1
        a = self.choose(k)
        self.last = (k, a)
        return a[0], ({"x": a[1][0], "y": a[1][1]} if a[1] is not None else None)

    def masked_diff(self, a, b):
        d = a != b
        if self.mask_rows.shape[0] == d.shape[0]: d[self.mask_rows, :] = False
        if self.mask_cols.shape[0] == d.shape[1]: d[:, self.mask_cols] = False
        return d.sum()

    def choose(self, k):
        if self.plan:                                                         # on suit un chemin : vérifier qu'on est où prévu
            exp, a = self.plan[0]
            if exp == k: self.plan.popleft(); return a
            self.plan.clear()
        untried = [a for a, nxt in self.G[k].items() if nxt is None]
        if untried:                                                           # CURIOSITÉ : l'inconnu, le plus prometteur d'abord
            sc = np.array([self.prior(a[2]) for a in untried]) + 1e-3 * self.rng.random(len(untried))
            return untried[int(sc.argmax())]
        path = self.path_to_unknown(k)                                        # retourner vers de l'inconnu
        if path:
            self.plan = deque(path[1:]); return path[0][1]
        safe = [a for a, nxt in self.G[k].items() if nxt != "GO" and nxt != k] or list(self.G[k])
        return safe[int(self.rng.integers(len(safe)))]                       # tout est connu : on bouge quand même

    def path_to_unknown(self, k):
        """BFS dans le graphe connu -> [(état attendu, action), ...] jusqu'à un état avec des actions jamais essayées."""
        prev = {k: None}; q = deque([k])
        while q:
            s = q.popleft()
            if s != k and any(n is None for n in self.G.get(s, {}).values()):
                path = []
                while prev[s] is not None:
                    p, a = prev[s]; path.append((p, a)); s = p
                return path[::-1]
            for a, n in self.G.get(s, {}).items():
                if n is not None and n != "GO" and n not in prev and n in self.G:
                    prev[n] = (s, a); q.append(n)
        return None
