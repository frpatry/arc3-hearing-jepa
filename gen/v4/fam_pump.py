"""pump4 (clic, quantités conservées) : des COLONNES de liquide, des POMPES entre colonnes (cliquer = transférer)
et des FLOTTEURS colorés posés à la surface ; but : amener chaque flotteur à l'encoche de sa couleur sur le bord de
sa colonne. Principes : contrôle indirect (on ne touche jamais le flotteur), quantités conservées, but affiché
(encoches), actionneurs rares (cliquer une colonne ou le décor ne fait rien et ne coûte rien), transformation globale
(gravité tournée) au fil des niveaux.
Variantes : `pump` (un clic = une unité de A vers B) ou `valve` (un clic = les deux colonnes s'égalisent, le reste
impair reste du côté source).
Règles ajoutées (ordre tiré par jeu) :
  twin    : une pompe double (déplace 2 unités d'un coup) ;
  lid     : une colonne à couvercle (capacité réduite, le couvercle est dessiné) ;
  marker2 : un 2e flotteur d'une autre couleur ;
  pipe    : une pompe à tuyau entre deux colonnes NON voisines (tuyau en pointillés) ;
  rotate  : la scène est tournée (gravité vers le haut, la gauche ou la droite) ;
  reverse : les pompes deviennent réversibles : clic sur la flèche du haut = A->B, du bas = B->A.
Solvabilité : BFS exact sur les niveaux (tuple des hauteurs) ; cible = état lointain atteignable."""
from collections import deque

import numpy as np

from ..core import EV
from . import draw
from .base import V4Game, mods_for
from .draw import R, blit, frame_rect, rect, sprite


class Pump4(V4Game):
    family = "pump4"
    mechanic = "click"
    principle = ("quantités conservées : pompes / vannes entre colonnes, flotteurs à aligner sur des encoches, gravité "
                 "tournée", ["vc33", "su15", "sp80"])
    simple_ops = ()
    CAT = ("twin", "lid", "marker2", "pipe", "rotate", "reverse")
    free_miss = True
    scale_p = (1.0, 0.0)
    table_cap = 20000
    bfs_cap = 60000

    def setup(self, rng):
        r = self.orng
        self.uses_click = True
        self.variant = str(rng.choice(["pump", "pump", "pump", "valve"]))
        cat = [c for c in self.CAT if not (self.variant == "valve" and c == "twin")]
        self.cat = [str(x) for x in rng.permutation(cat)]
        self.U = int(r.choice([3, 4, 4, 5]))                  # px par unité de liquide
        self.colw = int(r.choice([5, 6, 7, 8]))
        self.pump_kind = str(r.choice(["arrow", "diamond", "framed", "plus"]))
        self.mk_kind = str(r.choice(["hat", "diamond", "box", "blob"]))
        self.tpl_mk = sprite(r, self.mk_kind, 3, max(3, self.colw - 2), 100, 101)
        self.frame_cols = bool(r.random() < 0.6)

    # ---------------------------------------------------------------- géométrie (repère « gravité vers le bas »)
    def _geom(self, K, cap):
        """aire non tournée (h, w) ; x0 de chaque colonne ; y du fond ; y des pompes."""
        gap = 6
        W = K * self.colw + (K - 1) * gap + 4
        Hcol = cap * self.U + 2
        H = Hcol + 12
        return H, W, gap, Hcol

    def _rot_dims(self, lvl):
        return (lvl["H"], lvl["W"]) if lvl["rot"] % 2 == 0 else (lvl["W"], lvl["H"])

    def _to_scene(self, lvl, y, x):
        """repère local (gravité vers le bas) -> pixel de la scène tournée (dans l'aire AH×AW)."""
        H, W = lvl["H"], lvl["W"]
        k = lvl["rot"]
        if k == 0:
            yy, xx = y, x
        elif k == 1:                                          # 90° : gravité vers la droite
            yy, xx = x, H - 1 - y
        elif k == 2:
            yy, xx = H - 1 - y, W - 1 - x
        else:
            yy, xx = W - 1 - x, y
        return yy + lvl["oy"], xx + lvl["ox"]

    def _from_scene(self, lvl, yy, xx):
        H, W = lvl["H"], lvl["W"]
        y, x = yy - lvl["oy"], xx - lvl["ox"]
        k = lvl["rot"]
        if k == 0:
            return y, x
        if k == 1:
            return H - 1 - x, y
        if k == 2:
            return H - 1 - y, W - 1 - x
        return x, W - 1 - y

    def col_x(self, lvl, j):
        return 2 + j * (self.colw + lvl["gap"])

    def pump_boxes(self, lvl):
        """[(y, x, h, w, pump index, sens)] en repère local. Rangée 1 : pompes entre voisines (sous l'intervalle) ;
        rangée 2 (si réversibles) : leurs boutons inverses ; dernière rangée : boutons des tuyaux, de gauche à droite."""
        out = []
        yb = lvl["Hcol"] + 3
        rows = 2 if lvl["reverse"] else 1
        xp = 2
        for j, (a, b, amt, pipe) in enumerate(lvl["pumps"]):
            if pipe:
                yp = yb + 5 * rows
                out.append((yp, xp, 4, 4, j, 1))
                if lvl["reverse"]:
                    out.append((yp, xp + 5, 4, 4, j, -1))
                    xp += 5
                xp += 6
            else:
                x = self.col_x(lvl, min(a, b)) + self.colw + (lvl["gap"] - 4) // 2
                out.append((yb, x, 4, 4, j, 1))
                if lvl["reverse"]:
                    out.append((yb + 5, x, 4, 4, j, -1))
        return out

    def which(self, lvl, px):
        if px is None:
            return None
        y, x = self._from_scene(lvl, px[0], px[1])
        for (py, pxx, h, w, j, sgn) in self.pump_boxes(lvl):
            if py <= y < py + h and pxx <= x < pxx + w:
                return j, sgn
        return None

    # ---------------------------------------------------------------- règles
    def apply(self, lvl, st, op, px):
        hit = self.which(lvl, px) if op == "click" else None
        if hit is None:
            return st, False
        j, sgn = hit
        a, b, amt, _ = lvl["pumps"][j]
        if sgn < 0:
            a, b = b, a
        h = list(st)
        caps = lvl["caps"]
        if self.variant == "valve":
            tot = h[a] + h[b]
            na = (tot + 1) // 2
            nb = tot - na
            if nb > caps[b]:
                nb, na = caps[b], tot - caps[b]
            if na > caps[a]:
                return st, False
            h[a], h[b] = na, nb
        else:
            m = min(amt, h[a], caps[b] - h[b])
            if m <= 0:
                return st, False
            h[a] -= m
            h[b] += m
        return tuple(h), False

    def is_win(self, lvl, st):
        return all(st[c] == t for c, t, _ in lvl["marks"])

    def click_targets(self, lvl, st):
        out = []
        for (py, pxx, h, w, j, sgn) in self.pump_boxes(lvl):
            out.append(self._to_scene(lvl, py + h // 2, pxx + w // 2))
        return out

    def click_hit(self, lvl, st, px):
        return self.which(lvl, px) is not None

    def heuristic(self, lvl, st):
        return float(sum(abs(st[c] - t) for c, t, _ in lvl["marks"]))

    # ---------------------------------------------------------------- niveaux
    def gen_level(self, rng, i, n):
        mods, newest = mods_for(self.cat, i)
        K = int(min(3 + (i + 1) // 2 + int(rng.integers(0, 2)), 6))
        rot = int(rng.choice([1, 2, 3])) if "rotate" in mods else 0
        room = (self.AH - 7 if rot % 2 == 0 else self.AW) - 16 - 5 * (1 + ("reverse" in mods))
        cap = int(min(rng.integers(7, 12), room // self.U))
        while K > 3 and K * (self.colw + 6) - 2 > (self.AW if rot % 2 == 0 else self.AH - 7):
            K -= 1
        H, W, gap, Hcol = self._geom(K, cap)
        H += 5 * (1 + ("reverse" in mods))
        AH, AW = self.AH - 7, self.AW                        # bandeau du haut (encoches de rappel) : 7 px
        hh, ww = (H, W) if rot % 2 == 0 else (W, H)
        if hh > AH or ww > AW:
            return self.why("trop grand")
        caps = [cap] * K
        if "lid" in mods:
            caps[int(rng.integers(K))] = cap - int(rng.integers(2, 4))
        pumps = []
        for a in range(K - 1):                                # pompe entre voisines, sens tiré
            if rng.random() < 0.85:
                ab = (a, a + 1) if rng.random() < 0.5 else (a + 1, a)
                pumps.append((ab[0], ab[1], 1, False))
        if K >= 3:                                            # tuyau de retour (cycle) : l'état n'est pas monotone
            fwd = sum(1 for a, b, _, _ in pumps if b > a) >= len(pumps) / 2
            pumps.append((K - 1, 0, 1, True) if fwd else (0, K - 1, 1, True))
        if "twin" in mods and pumps:
            k = int(rng.integers(len(pumps)))
            a, b, _, p = pumps[k]
            pumps[k] = (a, b, 2, p)
        if "pipe" in mods and K >= 3:
            a = int(rng.integers(0, K - 2))
            b = int(rng.integers(a + 2, K))
            pumps.append((b, a, 1, True) if rng.random() < 0.5 else (a, b, 1, True))
        if len(pumps) < 2:
            return self.why("pompes")
        h0 = tuple(int(rng.integers(1, c)) for c in caps)
        nm = int(min(K - 1, 2 + i // 2 + ("marker2" in mods)))
        mcols = [int(x) for x in rng.choice(K, nm, replace=False)]
        lvl = dict(K=K, cap=cap, caps=tuple(caps), H=H, W=W, gap=gap, Hcol=Hcol, rot=rot, pumps=tuple(pumps),
                   reverse="reverse" in mods, marks=(), mods=mods,
                   oy=7 + (AH - hh) // 2, ox=(AW - ww) // 2)
        boxes = self.pump_boxes(lvl)
        occ = set()
        for (py, pxx, h, w, j, sg) in boxes:                  # boutons dans l'aire et sans chevauchement
            cells = {(py + a, pxx + b) for a in range(h) for b in range(w)}
            if pxx < 0 or pxx + w > W or py + h > H or cells & occ:
                return self.why("boutons")
            occ |= cells
        # BFS exhaustif (états = hauteurs) : cible = hauteurs lointaines des colonnes à flotteur
        moves = self.click_targets(lvl, h0)
        prev, depth, q = {h0: None}, {h0: 0}, deque([h0])
        while q and len(depth) < 30000:
            s = q.popleft()
            for p in moves:
                s2, _ = self.apply(lvl, s, "click", p)
                if s2 not in depth:
                    depth[s2] = depth[s] + 1
                    prev[s2] = (s, p)
                    q.append(s2)
        best = {}
        for s, d in depth.items():
            key = tuple(s[c] for c in mcols)
            if any(s[c] == h0[c] for c in mcols):
                continue
            if key not in best or d < depth[best[key]]:
                best[key] = s
        if not best:
            return self.why("immobile")
        dm = max(depth[s] for s in best.values())
        nf = self.__dict__.setdefault("_tries", {})
        nf[i] = nf.get(i, 0) + 1
        if dm < self.min_len(i, n) * max(0.3, 1 - nf[i] / 60):
            return self.why("court")
        far = [s for s in best.values() if depth[s] >= 0.85 * dm]
        s = far[int(rng.integers(len(far)))]
        lvl["marks"] = tuple((c, int(s[c]), k) for k, c in enumerate(mcols))
        sol = []
        while prev[s] is not None:
            s, p = prev[s]
            sol.append(("click", p))
        sol = sol[::-1]
        if newest in ("twin", "pipe", "reverse"):
            used = set()
            x = h0
            for op, p in sol:
                used.add(self.which(lvl, p))
                x, _ = self.apply(lvl, x, op, p)
            ok = dict(twin=any(lvl["pumps"][j][2] == 2 for j, _ in used if True),
                      pipe=any(lvl["pumps"][j][3] for j, _ in used),
                      reverse=any(sg < 0 for _, sg in used))[newest]
            if not ok and nf[i] < 40 and not self.relax:
                return self.why("règle inutile")
        return lvl, h0, sol

    def min_len(self, i, n):
        return min(14 + 6 * i, 55) if self.variant == "pump" else min(6 + 3 * i, 24)

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        hit = self.which(lvl, px) if op == "click" else None
        if hit is not None:
            ev[EV["touch"]] = 1
        if st2 != st:
            ev[EV["push"]] = 1
            before = sum(st[c] == t for c, t, _ in lvl["marks"])
            after = sum(st2[c] == t for c, t, _ in lvl["marks"])
            if after > before:
                ev[EV["collect"]] = 1
            if after < before:
                ev[EV["danger"]] = 1

    # ---------------------------------------------------------------- rendu
    def render(self, lvl, st):
        pal = self.pal
        g = self.blank()
        loc = np.full((lvl["H"], lvl["W"]), -1, np.int64)          # dessin en repère local puis rotation
        U, cw = self.U, self.colw
        fluid = pal.a(0)
        mk_cols = [pal.a(1), pal.a(2), pal.a(5), pal.a(6), pal.a(7)]
        mk_at = {c: k for c, t, k in lvl["marks"]}
        for j in range(lvl["K"]):
            x = self.col_x(lvl, j)
            y1 = lvl["Hcol"]                                   # fond de la colonne (y exclu)
            cap = lvl["caps"][j]
            top = y1 - cap * U - 1
            if self.frame_cols:
                loc[top:y1 + 1, x - 1] = pal.wall
                loc[top:y1 + 1, x + cw] = pal.wall
                loc[y1, x - 1:x + cw + 1] = pal.wall
            else:
                loc[top:y1, x:x + cw] = pal.room
            if cap < lvl["cap"]:
                loc[top, x - 1:x + cw + 1] = pal.line                # couvercle
            hgt = st[j]
            loc[y1 - hgt * U:y1, x:x + cw] = fluid
            if j in mk_at:
                k = mk_at[j]
                s = draw.recolor(self.tpl_mk, {100: mk_cols[k], 101: pal.fg})
                yy = y1 - hgt * U - s.shape[0]
                sub = loc[max(0, yy):yy + s.shape[0], x + (cw - s.shape[1]) // 2:x + (cw - s.shape[1]) // 2 + s.shape[1]]
                ss = s[s.shape[0] - sub.shape[0]:, :sub.shape[1]]
                sub[ss >= 0] = ss[ss >= 0]
                t = [tt for c, tt, kk in lvl["marks"] if c == j][0]          # encoche : petit trait à gauche
                ty = y1 - t * U - 1
                loc[ty:ty + 2, max(0, x - 4):x - 1] = mk_cols[k]
        for (py, pxx, h, w, j, sgn) in self.pump_boxes(lvl):
            a, b, amt, pipe = lvl["pumps"][j]
            if sgn < 0:
                a, b = b, a
            col = pal.a(3) if amt == 1 else pal.a(4)
            if self.pump_kind == "arrow":                       # flèche dans le sens du transfert
                s = sprite(None, "arrow", h, w, col)
                s = np.rot90(s, 3 if b > a else 1)
            else:
                s = sprite(None, self.pump_kind, h, w, col, pal.fg)
            sub = loc[py:py + h, pxx:pxx + w]
            sub[s[:sub.shape[0], :sub.shape[1]] >= 0] = s[:sub.shape[0], :sub.shape[1]][s[:sub.shape[0], :sub.shape[1]] >= 0]
            if pipe:                                            # tuyau en pointillés sous les colonnes
                xa, xb = self.col_x(lvl, a) + cw // 2, self.col_x(lvl, b) + cw // 2
                yb = lvl["Hcol"] + 1 + (j % 2)
                loc[yb, min(xa, xb):max(xa, xb) + 1] = pal.line
                loc[yb, xa] = loc[yb, xb] = col
        # rotation et collage
        k = lvl["rot"]
        rl = np.rot90(loc, -k) if k else loc
        oy, ox = lvl["oy"], lvl["ox"]
        sub = g[oy:oy + rl.shape[0], ox:ox + rl.shape[1]]
        m = rl[:sub.shape[0], :sub.shape[1]] >= 0
        sub[m] = R(0) + rl[:sub.shape[0], :sub.shape[1]][m]
        # bandeau du haut : rappel des flotteurs à placer (plein = en place)
        rect(g, 0, 0, 6, self.AW, pal.ui)
        for c, t, kk in lvl["marks"]:
            x = 2 + 6 * kk
            if st[c] == t:
                rect(g, 1, x, 4, 4, mk_cols[kk])
            else:
                frame_rect(g, 1, x, 4, 4, mk_cols[kk])
        return g
