"""CARTE MENTALE de l'agent (idée « hippocampe » / carte cognitive de Tolman, validée par l'user le 25 oct. : « c'est une mémoire,
une DB ») : graphe des états RÉELS vécus dans le niveau en cours (HUD masqué : lignes / colonnes qui changent un peu à presque chaque
pas = compteur, barre d'énergie), des gestes essayés dans chacun et de leur résultat (état suivant, rien, GAME OVER).
Elle ne joue pas à la place du modèle du monde : elle ORIENTE la note du planificateur (bonus / malus sur le 1er geste) —
  - geste connu qui ne fait rien (même état) ou qui mène à GAME OVER : malus
  - geste déjà essayé : léger malus (préférer l'inconnu), PLUS un malus qui croît avec le nombre de passages dans l'état où il mène
    (exploration par comptage : ft09 26 oct., 118 clics sur 2 cases = une tuile retournée puis remise, A -> B -> A -> B ...)
  - tout est déjà essayé ici : bonus sur le 1er pas du plus court chemin CONNU vers un état qui a encore des gestes jamais essayés
    (« la frontière de l'inconnu est par là ») — direction à long terme que l'imagination à H pas ne voit pas."""
import hashlib
from collections import deque
import numpy as np


class CogMap:
    def __init__(s, w_tried=0.5, w_noop=1.5, w_go=3.0, w_path=2.5, w_rev=0.5):
        s.w = dict(tried=w_tried, noop=w_noop, go=w_go, path=w_path, rev=w_rev); s.reset()

    def reset(s):                                     # nouveau niveau : nouvelle carte
        s.G = {}; s.F = {}; s.cnt = {}; s.start = None; s.row = s.col = None; s.n = 0         # F : une image de chaque état (pour rejouer un chemin)
        s.mr = np.zeros(0, bool); s.mc = np.zeros(0, bool)

    def observe(s, a, b):                             # deux images successives -> lignes / colonnes « HUD »
        if a is None or a.shape != b.shape: return
        d = a != b
        if s.row is None or s.row.shape[0] != d.shape[0]: s.row = np.zeros(d.shape[0]); s.col = np.zeros(d.shape[1]); s.n = 0
        s.n += 1; s.row += (d.sum(1) > 0) & (d.sum(1) <= 3); s.col += (d.sum(0) > 0) & (d.sum(0) <= 3)
        if s.n >= 6: s.mr = s.row / s.n > 0.7; s.mc = s.col / s.n > 0.7

    def key(s, g):
        h = np.array(g, dtype=np.int8, copy=True)
        if s.mr.shape[0] == h.shape[0]: h[s.mr, :] = 0
        if s.mc.shape[0] == h.shape[1]: h[:, s.mc] = 0
        return hashlib.md5(h.tobytes()).hexdigest()

    def visit(s, k, gestures, frame=None):            # état atteint : ses gestes possibles (jamais essayés au départ)
        if s.start is None: s.start = k
        s.cnt[k] = s.cnt.get(k, 0) + 1
        if frame is not None and k not in s.F: s.F[k] = frame
        node = s.G.setdefault(k, {})
        for gk in gestures: node.setdefault(gk, None)

    def record(s, k, gk, nxt):                        # résultat d'un geste : clé de l'état suivant, ou "GO"
        if k in s.G: s.G[k][gk] = nxt

    def path(s, k):                                   # plus court chemin connu -> 1er geste vers un état qui a de l'inconnu
        prev = {k: None}; q = deque([k])
        while q:
            u = q.popleft()
            if u != k and any(v is None for v in s.G.get(u, {}).values()):
                while prev[u][0] != k: u = prev[u][0]
                return prev[u][1], None
            for gk, v in s.G.get(u, {}).items():
                if v is not None and v != "GO" and v != u and v not in prev and v in s.G:
                    prev[v] = (u, gk); q.append(v)
        return None, None

    def bonus(s, k):                                  # gk -> ajustement de la note du planificateur (1er geste)
        node = s.G.get(k, {}); b = {}
        for gk, v in node.items():
            if v == "GO": b[gk] = -s.w["go"]
            elif v == k: b[gk] = -s.w["noop"]
            elif v is not None: b[gk] = -s.w["tried"] - s.w["rev"] * np.log1p(s.cnt.get(v, 0))
        if node and all(v is not None for v in node.values()):
            g0, _ = s.path(k)
            if g0 is not None: b[g0] = b.get(g0, 0.0) + s.w["path"]
        return b

    def path_between(s, k0, k1):                      # plus court chemin CONNU k0 -> k1 : [(état, geste), ...] (None si inconnu)
        if k0 is None or k1 is None: return None
        prev = {k0: None}; q = deque([k0])
        while q:
            u = q.popleft()
            if u == k1:
                out = []
                while prev[u] is not None: out.append(prev[u]); u = prev[u][0]
                return out[::-1]
            for gk, v in s.G.get(u, {}).items():
                if v is not None and v != "GO" and v not in prev and v in s.G: prev[v] = (u, gk); q.append(v)
        return None

    def untried(s, k):
        return {gk for gk, v in s.G.get(k, {}).items() if v is None}
