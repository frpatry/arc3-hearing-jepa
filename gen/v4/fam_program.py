"""program4 (clic, programme puis exécution) : un petit robot sur un plateau, une rangée de cases-instructions en bas,
une colonne-palette de jetons à droite. Clic sur un jeton de la palette = le prendre (cadre blanc autour), clic sur une
case = l'y poser (la prise est consommée). ACTION5 (ou le bouton « lecture » du bandeau) EXÉCUTE le programme : le robot
déroule les instructions ; si le résultat est le but affiché, le niveau est gagné ; sinon tout est REMIS comme avant
l'exécution (image finale identique, seule la barre de pas bouge).
Variantes (règle de base) :
  reach : le robot doit FINIR sur le cadre-but, après être passé sur chaque repère (petits carrés de la couleur du
          but, de plus en plus nombreux avec les niveaux) ;
  order : le robot doit entrer sur les pastilles colorées exactement dans l'ordre affiché dans le bandeau.
Règles ajoutées (ordre tiré par jeu, une par niveau, mods_for) :
  walls  : blocs-murs ; s'y cogner = échec ;
  jump   : jeton SAUT (2 cases dans le sens du regard) et fossés (rangée / colonne de trous) qu'on ne peut que sauter ;
  loop   : 2e rangée = sous-programme ; jeton APPEL qui l'exécute ; la rangée principale est trop courte sans lui ;
  turn   : jetons relatifs (avancer, tourner à gauche / à droite) au lieu des 4 flèches absolues ;
  fixed  : cases verrouillées (soulignées), pré-remplies, non modifiables ;
  decoy  : cases pré-remplies de jetons faux, à remplacer (gomme dans la palette).
Solvabilité : programme construit (chemins BFS les plus courts entre points de passage, ou motif répété pour loop),
but = résultat de son exécution ; vérifié par apply()."""
from collections import deque

import numpy as np

from ..core import EV
from . import draw
from .base import V4Game, mods_for
from .draw import R, blit, frame_rect, rect, sprite

DV = ((-1, 0), (1, 0), (0, -1), (0, 1))          # U D L R (aussi : regard)
CW = {0: 3, 3: 1, 1: 2, 2: 0}
CCW = {v: k for k, v in CW.items()}
U, D, L, RR, F, TL, TR, J, C, E = range(10)      # jetons
ABS = (U, D, L, RR)
REL = (F, TL, TR)


def _mask(rows, c):
    s = np.full((len(rows), len(rows[0])), -1, np.int64)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch == "X":
                s[y, x] = c
    return s


ARROW = ["..X..", ".XXX.", "XXXXX", ".XXX.", ".XXX."]        # sprites PLEINS (un seul blob, sans trou)
FWD = ["..X..", ".XXX.", "XXXXX", "..X..", "..X.."]
TURN = ["XX...", "XX...", "XX...", "XXXXX", "XXXXX"]
JUMP = ["..X..", ".XXX.", "..X..", ".XXX.", "XX.XX"]
CALL = [".XXX.", "XXXXX", "XXXXX", "XXXXX", ".XXX."]
ERASE = [".....", ".....", "XXXXX", "XXXXX", "....."]


class Program4(V4Game):
    family = "program4"
    mechanic = "click"
    principle = ("éditer puis exécuter : poser des jetons-instructions dans des cases, ACTION5 exécute ; échec = tout "
                 "revient comme avant (sous-programme, sauts, virages, cases verrouillées)", ["sb26", "tn36", "sp80"])
    simple_ops = ("interact",)
    CAT = ("walls", "jump", "loop", "turn", "fixed", "decoy")
    free_miss = True
    exact_progress = False
    scale_p = (1.0, 0.0)
    n_levels_official = (5, 8)

    def setup(self, rng):
        r = self.orng
        self.uses_click = True
        self.variant = str(rng.choice(["reach", "order"]))
        self.cat = [str(x) for x in rng.permutation(self.CAT)]
        self.C = 5
        self.bot_kind = str(r.choice(["framed", "disc", "box"]))
        self.goal_kind = str(r.choice(["frame", "plus_outline", "ring"]))
        self.grid_dots = bool(r.random() < 0.4)

    # ---------------------------------------------------------------- géométrie de l'écran
    def _geom(self, lvl_hb, lvl_wb, nrows):
        AH, AW, C = self.AH, self.AW, self.C
        pal_x = AW - 8
        rows_y = [AH - 8 - 7 * (nrows - 1 - k) for k in range(nrows)]
        top = 9
        avail_h = rows_y[0] - 2 - top
        oy = top + max(0, (avail_h - lvl_hb * C) // 2)
        ox = 3 + max(0, (pal_x - 4 - 3 - lvl_wb * C) // 2)
        return dict(pal_x=pal_x, rows_y=rows_y, oy=oy, ox=ox)

    def slot_px(self, lvl, j):
        """coin haut-gauche de la case j (rangées de 8)."""
        nm = lvl["n_main"]
        if j < nm:
            row, k = divmod(j, 8)
            x0 = 3
        else:
            row, k = (nm + 7) // 8, j - nm
            x0 = 9
        return lvl["rows_y"][row], x0 + 6 * k

    def tok_px(self, lvl, k):
        return 9 + 7 * k, lvl["pal_x"]

    def play_px(self, lvl):
        return 1, self.AW - 8

    def _hit(self, lvl, px):
        if px is None:
            return None
        y, x = px
        py, pxx = self.play_px(lvl)
        if py <= y < py + 5 and pxx <= x < pxx + 5:
            return ("play", 0)
        for k in range(len(lvl["tokens"])):
            ty, tx = self.tok_px(lvl, k)
            if ty <= y < ty + 5 and tx <= x < tx + 5:
                return ("tok", lvl["tokens"][k])
        for j in range(lvl["n_slots"]):
            sy, sx = self.slot_px(lvl, j)
            if sy <= y < sy + 5 and sx <= x < sx + 5:
                return ("slot", j)
        return None

    # ---------------------------------------------------------------- exécution
    def _exec1(self, lvl, pos, f, t):
        """une instruction -> (pos, regard, case entrée ou None) ; None si le robot s'écrase."""
        Hb, Wb, block = lvl["Hb"], lvl["Wb"], lvl["block"]
        if t in ABS:
            d = DV[t]
            q = (pos[0] + d[0], pos[1] + d[1])
            f = t
        elif t == F:
            d = DV[f]
            q = (pos[0] + d[0], pos[1] + d[1])
        elif t == TL:
            return pos, CCW[f], None
        elif t == TR:
            return pos, CW[f], None
        elif t == J:
            d = DV[f]
            q = (pos[0] + 2 * d[0], pos[1] + 2 * d[1])
        else:
            return pos, f, None
        if not (0 <= q[0] < Hb and 0 <= q[1] < Wb) or q in block:
            return None
        return q, f, q

    def flat(self, lvl, slots):
        nm = lvl["n_main"]
        out = []
        for t in slots[:nm]:
            if t == C and lvl["loop"]:
                out += [u for u in slots[nm:] if u not in (-1, C, E)]
            elif t not in (-1, C, E):
                out.append(t)
        return out

    def run(self, lvl, slots):
        pos, f = lvl["start"], lvl["face"]
        seq, seen = [], {pos}
        for t in self.flat(lvl, slots):
            r = self._exec1(lvl, pos, f, t)
            if r is None:
                return False
            pos, f, q = r
            if q is not None:
                seen.add(q)
                if q in lvl["pads"]:
                    seq.append(lvl["pads"][q])
        if self.variant == "reach":
            return pos == lvl["goal"] and lvl["checks"] <= seen
        return tuple(seq) == lvl["seq"]

    # ---------------------------------------------------------------- règles du jeu
    def _step(self, lvl, st, op, px):
        slots, sel, won = st
        if won:
            return st, "none"
        if op == "interact":
            h = ("play", 0)
        elif op == "click":
            h = self._hit(lvl, px)
        else:
            return st, "none"
        if h is None:
            return st, "none"
        kind, v = h
        if kind == "play":
            if self.run(lvl, slots):
                return (slots, -1, 1), "win"
            return st, "fail"
        if kind == "tok":
            if sel == v:
                return st, "none"
            return (slots, v, 0), "select"
        j = v
        if sel < 0 or j in lvl["locked"]:
            return st, "none"
        new = -1 if sel == E else sel
        if sel == C and j >= lvl["n_main"]:
            return st, "none"
        if slots[j] == new:
            return st, "none"
        s2 = slots[:j] + (new,) + slots[j + 1:]
        return (s2, -1, 0), "place"

    def apply(self, lvl, st, op, px):
        return self._step(lvl, st, op, px)[0], False

    def is_win(self, lvl, st):
        return st[2] == 1

    def click_targets(self, lvl, st):
        out = [(self.tok_px(lvl, k)[0] + 2, self.tok_px(lvl, k)[1] + 2) for k in range(len(lvl["tokens"]))]
        out += [(self.slot_px(lvl, j)[0] + 2, self.slot_px(lvl, j)[1] + 2) for j in range(lvl["n_slots"])]
        return out

    def click_hit(self, lvl, st, px):
        if self._hit(lvl, px) is not None:
            return True
        y, x = px
        r, c = (y - lvl["oy"]) // self.C, (x - lvl["ox"]) // self.C
        return y >= lvl["oy"] and x >= lvl["ox"] and 0 <= r < lvl["Hb"] and 0 <= c < lvl["Wb"]

    # ---------------------------------------------------------------- oracle
    def _plan(self, lvl, st):
        slots, sel, won = st
        if won:
            return None
        if self.run(lvl, slots):
            return ("interact", None)
        want = lvl["want"]
        diffs = [j for j in range(lvl["n_slots"]) if slots[j] != want[j] and j not in lvl["locked"]]
        if not diffs:
            return ("interact", None)

        def tok_for(j):
            return E if want[j] == -1 else want[j]
        for j in diffs:
            if sel >= 0 and tok_for(j) == sel:
                y, x = self.slot_px(lvl, j)
                return ("click", (y + 2, x + 2))
        k = lvl["tokens"].index(tok_for(diffs[0]))
        y, x = self.tok_px(lvl, k)
        return ("click", (y + 2, x + 2))

    def oracle_action(self):
        if self.state != "NOT_FINISHED":
            return 0, None
        L = self.levels[self.level]
        mv = self._plan(L["lvl"], self.st)
        if mv is None:
            return 0, None
        return self._to_action(mv[0], mv[1], self.st)

    def heuristic(self, lvl, st):
        if st[2]:
            return 0
        want = lvl["want"]
        diffs = [j for j in range(lvl["n_slots"]) if st[0][j] != want[j] and j not in lvl["locked"]]
        if not diffs and self.run(lvl, st[0]):
            return 1
        useful = st[1] >= 0 and any((E if want[j] == -1 else want[j]) == st[1] for j in diffs)
        return 2 * len(diffs) + 1 - (1 if useful else 0)

    # ---------------------------------------------------------------- génération
    def _moves(self, toks):
        return [t for t in toks if t in ABS + REL + (J,)]

    def _bfs(self, lvl, pos, f, toks, avoid=frozenset()):
        """BFS sur (case, regard) -> dist, prev."""
        s0 = (pos, f)
        prev, dist = {s0: None}, {s0: 0}
        q = deque([s0])
        mv = list(self._moves(toks))
        while q:
            s = q.popleft()
            order = [mv[k] for k in self._perm(len(mv))]
            for t in order:
                r = self._exec1(lvl, s[0], s[1], t)
                if r is None:
                    continue
                s2 = (r[0], r[1])
                if r[2] is not None and r[2] in avoid:
                    continue
                if s2 not in dist:
                    dist[s2] = dist[s] + 1
                    prev[s2] = (s, t)
                    q.append(s2)
        return dist, prev

    def _perm(self, n):
        return self._grng.permutation(n)

    @staticmethod
    def _path(prev, s):
        out = []
        while prev[s] is not None:
            s, t = prev[s]
            out.append(t)
        return out[::-1]

    def gen_level(self, rng, i, n):
        self._grng = rng
        mods, newest = mods_for(self.cat, i)
        loop = "loop" in mods
        rel = "turn" in mods
        jump = "jump" in mods
        d = i / max(1, n - 1)
        Hb = int(min(5, 4 + (i >= 1)))
        Wb = int(min(9, 6 + i + int(rng.integers(0, 2))))
        if loop:
            Hb, Wb = 5, int(rng.integers(7, 10))
        toks = list(REL if rel else ABS) + ([J] if jump else []) + ([C] if loop else [])
        lvl = dict(Hb=Hb, Wb=Wb, block=frozenset(), pits=frozenset(), walls=frozenset(), pads={}, seq=(), goal=None,
                   loop=loop, mods=mods, newest=newest, checks=frozenset())
        if loop:
            out = self._gen_loop(rng, i, lvl, toks, mods, rel, jump)
        else:
            out = self._gen_flat(rng, i, lvl, toks, mods, rel, jump)
        if out is None:
            return None
        main, sub = out
        # ----- cases, jetons, verrous, leurres
        slack = int(rng.integers(0, 2)) if (i >= 3 and not loop and len(main) < 24) else 0
        n_main = len(main) + slack
        want = tuple(main) + (-1,) * slack + tuple(sub)
        lvl["n_main"], lvl["n_slots"] = n_main, len(want)
        tokens = sorted(set(t for t in want if t >= 0) | set(toks))
        if "decoy" in mods or slack:
            tokens.append(E)
        lvl["tokens"] = tuple(tokens)
        if len(tokens) > 7:
            return self.why("palette")
        locked = set()
        init = [-1] * len(want)
        idx = [j for j in range(len(want)) if want[j] >= 0]
        if "fixed" in mods:
            for j in rng.permutation(idx)[:min(3, max(1, len(idx) // 4))]:
                locked.add(int(j))
                init[int(j)] = want[int(j)]
        if "decoy" in mods:
            free = [j for j in range(len(want)) if j not in locked]
            for j in rng.permutation(free)[:3]:
                j = int(j)
                cands = [t for t in tokens if t not in (E, want[j]) and not (t == C and j >= n_main)]
                init[j] = int(rng.choice(cands))
        lvl["locked"] = frozenset(locked)
        lvl["want"] = want
        nrows = (n_main + 7) // 8 + (1 if loop else 0)
        lvl.update(self._geom(Hb, Wb, nrows))
        if n_main > 24 or (loop and n_main > 16):
            return self.why("long")
        st0 = (tuple(init), -1, 0)
        if self.run(lvl, st0[0]):
            return self.why("trivial")
        if not self.run(lvl, want):
            return self.why("bad")
        sol = []
        st = st0
        for _ in range(4 * len(want) + 4):
            mv = self._plan(lvl, st)
            if mv is None:
                break
            sol.append(mv)
            st, _ = self.apply(lvl, st, *mv)
            if st[2]:
                break
        if not st[2]:
            return self.why("plan")
        if newest == "loop" and lvl["need_loop"] is False:
            return self.why("loop_unused")
        return lvl, st0, sol

    def _place_walls(self, rng, lvl, keep, dens):
        cells = [(r, c) for r in range(lvl["Hb"]) for c in range(lvl["Wb"]) if (r, c) not in keep]
        k = int(round(dens * lvl["Hb"] * lvl["Wb"]))
        w = set()
        for t in rng.permutation(len(cells))[:k]:
            w.add(cells[int(t)])
        return w

    def _gen_flat(self, rng, i, lvl, toks, mods, rel, jump):
        Hb, Wb = lvl["Hb"], lvl["Wb"]
        walls, pits = set(), set()
        if jump:                                                # fossé : une rangée ou colonne entière de trous
            if rng.random() < 0.5 and Wb >= 5:
                c0 = int(rng.integers(2, Wb - 2))
                pits = {(r, c0) for r in range(Hb)}
            else:
                r0 = int(rng.integers(1, Hb - 1))
                pits = {(r0, c) for c in range(Wb)}
        if "walls" in mods:
            walls = self._place_walls(rng, lvl, pits, float(rng.uniform(0.14, 0.24)))
        lvl["walls"], lvl["pits"] = frozenset(walls), frozenset(pits)
        lvl["block"] = frozenset(walls | pits)
        free = [(r, c) for r in range(Hb) for c in range(Wb) if (r, c) not in lvl["block"]]
        face = int(rng.integers(4))
        if self.variant == "reach" and i > 0:                  # départ excentré : chemins plus longs
            cs = [free[int(k)] for k in rng.permutation(len(free))[:6]]
            ecc = [max(self._bfs(lvl, c, face, toks)[0].values()) for c in cs]
            start = cs[int(np.argmax(ecc))]
        else:
            start = free[int(rng.integers(len(free)))]
        lvl["start"], lvl["face"] = start, face
        lo, hi = (5, 8) if i == 0 else (8 + i, 11 + 2 * i)
        lo, hi = min(lo, 13), min(hi, 16)
        if self.variant == "reach":
            ncp = (0, 2, 3, 3, 4, 4, 4, 5)[min(i, 7)]
            if ncp == 0:
                dist, prev = self._bfs(lvl, start, face, toks)
                best = {}
                for s_, dd in dist.items():
                    if s_[0] != start and (s_[0] not in best or dd < dist[best[s_[0]]]):
                        best[s_[0]] = s_
                if jump:                                        # le but de l'autre côté du fossé
                    side = (lambda p: p[1] < next(iter(pits))[1]) if len({p[1] for p in pits}) == 1 else \
                        (lambda p: p[0] < next(iter(pits))[0])
                    best = {p: s_ for p, s_ in best.items() if side(p) != side(start)}
                cand = [p for p, s_ in best.items() if lo <= dist[s_] <= hi]
                if not cand:
                    cand = [p for p, s_ in best.items() if dist[s_] <= hi]
                    if not cand:
                        return self.why("flat_goal")
                    m = max(dist[best[p]] for p in cand)
                    cand = [p for p in cand if dist[best[p]] == m]
                g = cand[int(rng.integers(len(cand)))]
                lvl["goal"] = g
                main = self._path(prev, best[g])
                lvl["need_loop"] = None
                return main, []
            # points de passage (repères à visiter, dans n'importe quel ordre) puis le but
            prog, pos, f, visited, wps = [], start, face, {start}, []
            for k in range(ncp + 1):
                dist, prev = self._bfs(lvl, pos, f, toks)
                best = {}
                for s_, dd in dist.items():
                    if s_[0] not in visited and (s_[0] not in best or dd < dist[best[s_[0]]]):
                        best[s_[0]] = s_
                cand = [p for p, s_ in best.items() if 4 <= dist[s_] <= 8]
                if not cand:
                    return self.why("reach_wp")
                p = cand[int(rng.integers(len(cand)))]
                seg = self._path(prev, best[p])
                q, ff = pos, f
                for t in seg:
                    q, ff, e = self._exec1(lvl, q, ff, t)
                    if e is not None:
                        visited.add(e)
                wps.append(p)
                prog += seg
                pos, f = best[p]
            if jump and J not in prog:
                return self.why("jump_unused")
            if len(prog) > 24:
                return self.why("reach_long")
            lvl["goal"], lvl["checks"] = wps[-1], frozenset(wps[:-1])
            lvl["need_loop"] = None
            return prog, []
        # order : points de passage successifs (pastilles), chacun par le plus court chemin
        npads = int(min(6, 2 + 2 * (i >= 1) + i // 2))
        ncol = 2 if i < 2 else 3
        pads, seq, prog = {}, [], []
        pos, f, visited = start, face, {start}
        for k in range(npads):
            avoid = frozenset(pads)
            dist, prev = self._bfs(lvl, pos, f, toks, avoid)
            best = {}
            for s, dd in dist.items():
                if s[0] not in visited and s[0] not in pads and (s[0] not in best or dd < dist[best[s[0]]]):
                    best[s[0]] = s
            lo_k, hi_k = (2, 5) if i == 0 else (3, 7)
            cand = [p for p, s in best.items() if lo_k <= dist[best[p]] <= hi_k]
            if not cand:
                return self.why("order_pad")
            p = cand[int(rng.integers(len(cand)))]
            seg = self._path(prev, best[p])
            # cases traversées par le segment
            q, ff = pos, f
            for t in seg:
                q, ff, e = self._exec1(lvl, q, ff, t)
                if e is not None:
                    visited.add(e)
            col = int(rng.integers(ncol)) if k > 0 else int(rng.integers(ncol))
            pads[p] = col
            seq.append(col)
            prog += seg
            pos, f = best[p]
        if jump and J not in prog:
            return self.why("jump_unused")
        if len(prog) > 24:
            return self.why("order_long")
        # pastilles leurres (hors chemin) aux niveaux avancés
        if i >= 3:
            off = [c for c in free if c not in visited and c not in pads]
            for t in rng.permutation(len(off))[:2]:
                pads[off[int(t)]] = int(rng.integers(ncol))
        lvl["pads"], lvl["seq"] = pads, tuple(seq)
        lvl["need_loop"] = None
        return prog, []

    def _gen_loop(self, rng, i, lvl, toks, mods, rel, jump):
        Hb, Wb = lvl["Hb"], lvl["Wb"]
        mv = [t for t in self._moves(toks)]
        big = dict(Hb=200, Wb=200, block=frozenset())
        best, ncand = None, 0
        for _ in range(600):
            ls = int(rng.integers(3, 6))
            if rel:
                sub = [int(rng.choice(mv)) for _ in range(ls)]
            else:                                           # escalier ou zigzag : avance toujours selon a
                a = int(rng.integers(4))
                b = int(rng.choice([x for x in range(4) if x != a and DV[x] != tuple(-v for v in DV[a])]))
                ob = next(x for x in range(4) if DV[x] == tuple(-v for v in DV[b]))
                dirs = [a, b] + ([ob] if rng.random() < 0.6 else [])
                sub = [int(rng.choice(dirs)) for _ in range(ls)]
                if a not in sub:
                    sub[int(rng.integers(ls))] = a
                if J in mv and rng.random() < 0.5:
                    sub[int(rng.integers(ls))] = J
            if "jump" == lvl.get("newest") and J not in sub:
                sub[int(rng.integers(ls))] = J
            if rel and not any(t in (F, J) for t in sub):
                continue
            r1, r2 = int(rng.integers(2, 4)), int(rng.integers(0, 3))
            pre = [int(rng.choice(mv)) for _ in range(int(rng.integers(1, 4)))]
            mid = [int(rng.choice(mv)) for _ in range(int(rng.integers(1, 4)))] if r2 else []
            post = [int(rng.choice(mv)) for _ in range(int(rng.integers(0, 4)))]
            main = pre + [C] * r1 + mid + [C] * r2 + post
            if len(main) > 14:
                continue
            flat = pre + sub * r1 + mid + sub * r2 + post
            face = int(rng.integers(4))
            pos, f, vis, jumped, ok = (100, 100), face, [(100, 100)], set(), True
            for t in flat:
                r = self._exec1(big, pos, f, t)
                if t == J:
                    jumped.add(((pos[0] + r[0][0]) // 2, (pos[1] + r[0][1]) // 2))
                pos, f, e = r
                if e is not None:
                    vis.append(e)
            ys, xs = [p[0] for p in vis + list(jumped)], [p[1] for p in vis + list(jumped)]
            if max(ys) - min(ys) >= Hb or max(xs) - min(xs) >= Wb:
                continue
            if len(set(vis)) < 6 or len(set(vis)) < 0.6 * len(vis):      # repasser est permis (avec mesure)
                continue
            dy = int(rng.integers(-min(ys), Hb - max(ys)))
            dx = int(rng.integers(-min(xs), Wb - max(xs)))
            vis = [(p[0] + dy, p[1] + dx) for p in vis]
            jumped = {(p[0] + dy, p[1] + dx) for p in jumped}
            start, pos = vis[0], (pos[0] + dy, pos[1] + dx)
            lvl["start"], lvl["face"] = start, face
            lvl["block"] = frozenset()
            if jumped & set(vis):
                continue
            pits = set(jumped)
            walls = self._place_walls(rng, lvl, set(vis) | pits, float(rng.uniform(0.1, 0.2))) if "walls" in mods else set()
            lvl["walls"], lvl["pits"], lvl["block"] = frozenset(walls), frozenset(pits), frozenset(walls | pits)
            once = [k_ for k_ in range(1, len(vis)) if vis.count(vis[k_]) == 1 and vis[k_] != pos]
            if self.variant == "reach":
                lvl["goal"] = pos
                k = int(min(len(once), 2 + (i >= 4)))
                lvl["checks"] = frozenset(vis[p] for p in rng.choice(once, k, replace=False)) if k else frozenset()
                dmin = self._reach_dist(lvl, start, face, toks)
            else:
                k = int(min(4, 2 + i // 3, len(once)))
                if k < 2:
                    continue
                picks = sorted(int(x) for x in rng.choice(once, k, replace=False))
                ncol = 3
                pads = {vis[p]: int(rng.integers(ncol)) for p in picks}
                lvl["pads"], lvl["seq"] = pads, tuple(pads[vis[p]] for p in picks)
                dmin = self._order_dist(lvl, start, face, toks)
            lvl["need_loop"] = dmin > len(main)
            if not lvl["need_loop"]:
                continue
            score = len(main) + len(sub) + 0.01 * rng.random()   # garder le plus long de plusieurs candidats
            if best is None or score > best[0]:
                keys = ("start", "face", "block", "walls", "pits", "goal", "pads", "seq", "need_loop", "checks")
                best = (score, main, sub, {k: lvl[k] for k in keys})
                ncand += 1
                if ncand >= 6:
                    break
        if best is None:
            return self.why("loop_gen")
        lvl.update(best[3])
        return best[1], best[2]

    def _reach_dist(self, lvl, start, face, toks):
        """plus court programme « à plat » : passer sur tous les repères puis finir sur le but (BFS case, regard, masque)."""
        ck = sorted(lvl["checks"])
        bit = {c: 1 << j for j, c in enumerate(ck)}
        full = (1 << len(ck)) - 1
        s0 = (start, face, bit.get(start, 0))
        dist = {s0: 0}
        q = deque([s0])
        mv = self._moves(toks)
        while q:
            s_ = q.popleft()
            if s_[2] == full and s_[0] == lvl["goal"]:
                return dist[s_]
            for t in mv:
                r = self._exec1(lvl, s_[0], s_[1], t)
                if r is None:
                    continue
                m = s_[2] | (bit.get(r[2], 0) if r[2] is not None else 0)
                s2 = (r[0], r[1], m)
                if s2 not in dist:
                    dist[s2] = dist[s_] + 1
                    q.append(s2)
        return 99

    def _order_dist(self, lvl, start, face, toks):
        """plus court programme « à plat » qui réalise la séquence de pastilles (BFS sur case, regard, k)."""
        s0 = (start, face, 0)
        dist = {s0: 0}
        q = deque([s0])
        seq = lvl["seq"]
        mv = self._moves(toks)
        while q:
            s = q.popleft()
            if s[2] == len(seq):
                return dist[s]
            for t in mv:
                r = self._exec1(lvl, s[0], s[1], t)
                if r is None:
                    continue
                k = s[2]
                if r[2] is not None and r[2] in lvl["pads"]:
                    if k < len(seq) and lvl["pads"][r[2]] == seq[k]:
                        k += 1
                    else:
                        continue
                s2 = (r[0], r[1], k)
                if s2 not in dist:
                    dist[s2] = dist[s] + 1
                    q.append(s2)
        return 99

    def min_len(self, i, n):
        return min(10 + 4 * i, 26)

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        _, tag = self._step(lvl, st, op, px)
        if tag == "select":
            ev[EV["toggle"]] = 1
        elif tag == "place":
            ev[EV["color"]] = 1
        elif tag == "fail":
            ev[EV["danger"]] = ev[EV["touch"]] = 1
        elif tag == "win":
            ev[EV["open"]] = 1

    # ---------------------------------------------------------------- rendu
    def _tok_sprite(self, t):
        pal = self.pal
        if t in ABS:
            s = _mask(ARROW, pal.a(0))
            return np.rot90(s, {U: 0, L: 1, D: 2, RR: 3}[t])
        if t == F:
            return _mask(FWD, pal.a(1))
        if t == TL:
            return _mask(TURN, pal.a(2))
        if t == TR:
            return _mask(TURN, pal.a(2))[:, ::-1]
        if t == J:
            return _mask(JUMP, pal.a(3))
        if t == C:
            return _mask(CALL, pal.a(4))
        return _mask(ERASE, pal.line)

    def render(self, lvl, st):
        pal, Cc = self.pal, self.C
        g = self.blank()
        slots, sel, won = st
        hi = draw.WHITE if pal.bg != draw.WHITE else draw.BLACK
        # en haut : séquence-cible (order) et bouton « lecture », posés sur le fond
        if self.variant == "order":
            for k, col in enumerate(lvl["seq"]):
                rect(g, 1, 2 + 6 * k, 4, 4, self._padc(col))
        py, px_ = self.play_px(lvl)
        blit(g, np.rot90(_mask(ARROW, pal.line), 3), py, px_)
        # plateau : sol = fond, bordure d'1 px, murs et fossés pleins
        oy, ox, Hb, Wb = lvl["oy"], lvl["ox"], lvl["Hb"], lvl["Wb"]
        frame_rect(g, oy - 1, ox - 1, Hb * Cc + 2, Wb * Cc + 2, pal.wall if pal.wall != pal.bg else pal.line)
        for (r, c) in lvl["walls"]:
            rect(g, oy + r * Cc, ox + c * Cc, Cc, Cc, pal.wall)
        for (r, c) in lvl["pits"]:
            rect(g, oy + r * Cc, ox + c * Cc, Cc, Cc, pal.room if pal.room != pal.wall else pal.ui)
        for (r, c), col in lvl["pads"].items():
            rect(g, oy + r * Cc + 1, ox + c * Cc + 1, 3, 3, self._padc(col))
        for (r, c) in lvl["checks"]:                           # repères à visiter (variante reach)
            rect(g, oy + r * Cc + 1, ox + c * Cc + 1, 3, 3, pal.a(5))
        if lvl["goal"] is not None:
            r, c = lvl["goal"]
            blit(g, sprite(None, "frame", 5, 5, pal.a(5)), oy + r * Cc, ox + c * Cc)
        r, c = lvl["start"]
        blit(g, np.rot90(_mask(ARROW, pal.fg), {0: 0, 2: 1, 1: 2, 3: 3}[lvl["face"]]), oy + r * Cc, ox + c * Cc)
        # palette
        for k, t in enumerate(lvl["tokens"]):
            y, x = self.tok_px(lvl, k)
            blit(g, self._tok_sprite(t), y, x)
            if t == sel:                                      # sélection : petit trait à gauche du jeton
                rect(g, y, x - 2, 5, 1, hi)
        # programme : case vide = petit carré plein
        if lvl["loop"]:
            y = lvl["rows_y"][-1]
            blit(g, _mask(CALL, pal.a(4)), y, 2)
        sc = pal.wall if pal.wall != pal.bg else pal.line
        for j in range(lvl["n_slots"]):
            y, x = self.slot_px(lvl, j)
            t = slots[j]
            if t < 0:
                rect(g, y + 1, x + 1, 3, 3, sc)
            else:
                blit(g, self._tok_sprite(t), y, x)
            if j in lvl["locked"]:
                draw.hline(g, y + 6, x, x + 4, pal.fg)
        return g

    def _padc(self, col):
        return (self.pal.a(5), self.pal.a(6), self.pal.a(3 if self.pal.a(3) != self.pal.a(5) else 2))[col]
