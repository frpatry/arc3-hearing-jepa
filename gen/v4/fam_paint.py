"""paint4 (objet, peinture en couches) : une TOILE de G×G cases et une VIGNETTE-cible (bandeau en haut à gauche) ;
choisir une couleur dans la palette (clic), une forme dans la boîte à outils (clic), puis l'appliquer sur la toile ;
les couches suivantes recouvrent les précédentes, donc l'ORDRE compte. But : toile = vignette.
Variantes (contrôle) :
  stamp : clic sur la toile = tamponner la forme choisie à cet endroit (coin haut-gauche) ; l'encre et
          l'outil sont alors consommés (à reprendre pour le tampon suivant) ;
  brush : un cadre-pinceau (forme choisie) se déplace sur la toile avec les flèches ; ACTION5 peint dessous.
Règles ajoutées (ordre tiré par jeu, une par niveau, mods_for) :
  colors  : une couleur de plus dans la palette ;
  shapes  : barres 1×3 / 3×1 dans la boîte à outils ;
  shapes2 : croix et anneau 3×3 ;
  under   : bouton « sous-couche » : en mode actif, on ne peint que les cases encore vierges ;
  rotate  : la vignette est montrée TOURNÉE (la toile ne l'est pas) ;
  mirror  : peinture en miroir : chaque application peint aussi son reflet gauche-droite.
Solvabilité : la cible est le résultat d'une suite d'applications tirée au hasard (élaguée des couches invisibles) ;
cette suite (plus sélections et déplacements) est la solution ; peindre est irréversible (RESET pour recommencer)."""
import numpy as np

from ..core import DIR_OPS, DIRS, EV
from . import draw
from .base import V4Game, mods_for
from .draw import R, frame_rect, rect

SHAPES = {
    "sq": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "bh": ((0, 0), (0, 1), (0, 2)),
    "bv": ((0, 0), (1, 0), (2, 0)),
    "plus": ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    "ring": ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)),
}


def _hw(shape):
    m = SHAPES[shape]
    return max(a for a, _ in m) + 1, max(b for _, b in m) + 1


class Paint4(V4Game):
    family = "paint4"
    mechanic = "object"
    principle = ("peinture en couches dépendant de l'ordre : reproduire une vignette-cible avec des formes colorées "
                 "(sous-couche, miroir, vignette tournée)", ["cd82", "sb26", "lp85"])
    CAT = ("colors", "shapes", "under", "rotate", "mirror", "shapes2")
    free_miss = True
    exact_progress = False
    scale_p = (1.0, 0.0)
    n_levels_official = (5, 8)

    def setup(self, rng):
        r = self.orng
        self.variant = str(rng.choice(["stamp", "brush"]))
        self.uses_click = True
        self.simple_ops = (DIR_OPS + ("interact",)) if self.variant == "brush" else ()
        self.cat = [str(x) for x in rng.permutation(self.CAT)]
        self.P = 5

    # ---------------------------------------------------------------- géométrie
    def _geom(self, lvl):
        AH, AW, G, P = self.AH, self.AW, lvl["G"], self.P
        band = 17
        lvl["band"] = band
        lvl["oy"] = band + 2 + (AH - band - 2 - G * P) // 2
        lvl["ox"] = (AW - G * P) // 2
        x = 18
        lvl["pal_px"] = tuple((2, x + 7 * k) for k in range(lvl["nc"]))
        lvl["tool_px"] = tuple((9, x + 7 * k) for k in range(len(lvl["shapes"])))
        xe = x + 7 * len(lvl["shapes"]) + 1
        lvl["under_px"] = (9, xe) if lvl["under"] else None
        if xe + 6 > AW - 1:
            return False
        return True

    def _hit(self, lvl, px):
        if px is None:
            return None
        y, x = px
        for k, (py, pxx) in enumerate(lvl["pal_px"]):
            if py <= y < py + 5 and pxx <= x < pxx + 5:
                return ("col", k + 1)
        for k, (py, pxx) in enumerate(lvl["tool_px"]):
            if py <= y < py + 6 and pxx <= x < pxx + 6:
                return ("tool", k)
        if lvl["under_px"] is not None:
            py, pxx = lvl["under_px"]
            if py <= y < py + 5 and pxx <= x < pxx + 5:
                return ("under", 0)
        G, P = lvl["G"], self.P
        r, rr = divmod(y - lvl["oy"], P)
        c, cc = divmod(x - lvl["ox"], P)
        if y >= lvl["oy"] and x >= lvl["ox"] and 0 <= r < G and 0 <= c < G:
            return ("cell", (int(r), int(c)))
        return None

    # ---------------------------------------------------------------- peinture
    def _paint(self, lvl, canvas, col, shape, anchor, under):
        G = lvl["G"]
        cv = list(canvas)
        cells = [(anchor[0] + a, anchor[1] + b) for a, b in SHAPES[shape]]
        if lvl["mirror"]:
            cells += [(r, G - 1 - c) for r, c in cells]
        for r, c in cells:
            if 0 <= r < G and 0 <= c < G and (not under or canvas[r * G + c] == 0):
                cv[r * G + c] = col
        return tuple(cv)

    def _fits(self, lvl, shape, anchor):
        h, w = _hw(shape)
        return 0 <= anchor[0] <= lvl["G"] - h and 0 <= anchor[1] <= lvl["G"] - w

    # state : (toile, couleur choisie (0 = aucune), outil, ancre (brush) ou None, sous-couche)
    def _step(self, lvl, st, op, px):
        cv, col, tool, anc, und = st
        if op == "click":
            h = self._hit(lvl, px)
            if h is None:
                return st, "none"
            kind, v = h
            if kind == "col":
                return ((cv, v, tool, anc, und), "select") if v != col else (st, "none")
            if kind == "tool":
                if v == tool:
                    return st, "none"
                sh = lvl["shapes"][v]
                if anc is not None:
                    hh, ww = _hw(sh)
                    anc = (min(anc[0], lvl["G"] - hh), min(anc[1], lvl["G"] - ww))
                return (cv, col, v, anc, und), "tool"
            if kind == "under":
                return (cv, col, tool, anc, 1 - und), "toggle"
            if self.variant != "stamp" or col == 0 or tool < 0:
                return st, "none"
            sh = lvl["shapes"][tool]
            if not self._fits(lvl, sh, v):
                return st, "none"
            cv2 = self._paint(lvl, cv, col, sh, v, und)          # tamponner consomme l'encre et l'outil choisis
            return ((cv2, 0, -1, anc, und), "paint") if cv2 != cv else (st, "none")
        if self.variant != "brush":
            return st, "none"
        if op == "interact":
            if col == 0:
                return st, "none"
            cv2 = self._paint(lvl, cv, col, lvl["shapes"][tool], anc, und)
            return ((cv2, col, tool, anc, und), "paint") if cv2 != cv else (st, "none")
        d = DIRS[op]
        a2 = (anc[0] + d[0], anc[1] + d[1])
        if not self._fits(lvl, lvl["shapes"][tool], a2):
            return st, "bump"
        return (cv, col, tool, a2, und), "move"

    def apply(self, lvl, st, op, px):
        return self._step(lvl, st, op, px)[0], False

    def is_win(self, lvl, st):
        return st[0] == lvl["target"]

    def click_targets(self, lvl, st):
        out = [(y + 2, x + 2) for y, x in lvl["pal_px"]] + [(y + 2, x + 2) for y, x in lvl["tool_px"]]
        return out

    def click_hit(self, lvl, st, px):
        return self._hit(lvl, px) is not None

    def heuristic(self, lvl, st):
        if st[0] == lvl["target"]:
            return 0
        r = lvl.get("rem", {}).get(st)
        if r is not None:
            return r
        return 3 * sum(a != b for a, b in zip(st[0], lvl["target"])) + 2

    def body(self, lvl, st):
        if self.variant != "brush":
            return None
        return (st[3][0], st[3][1], lvl["G"], lvl["G"])

    # ---------------------------------------------------------------- génération
    def gen_level(self, rng, i, n):
        mods, newest = mods_for(self.cat, i)
        G = int(min(7, 4 + (i >= 1) + (i >= 2) + (i >= 4)))
        nc = 2 + ("colors" in mods) + (i >= 5)
        shapes = ["sq"] + (["bh", "bv"] if "shapes" in mods else []) + (["plus", "ring"] if "shapes2" in mods else [])
        if i == 0:
            shapes = ["sq", "bh"] if rng.random() < 0.5 else ["sq"]
        lvl = dict(G=G, nc=nc, shapes=tuple(shapes), under="under" in mods, mirror="mirror" in mods, mods=mods,
                   rot=int(rng.integers(1, 4)) if "rotate" in mods else 0)
        if not self._geom(lvl):
            return self.why("geom")
        K = int(min(11, 3 + i + (i >= 1) + int(rng.integers(0, 2)))) if self.variant == "brush" else \
            int(min(16, 6 + 2 * i + 2 * (i >= 1) + int(rng.integers(0, 3))))
        ops = []
        for _ in range(K):
            k = int(rng.integers(len(shapes)))
            if newest in ("shapes", "shapes2") and not ops:
                k = len(shapes) - 1 - int(rng.integers(2))
            sh = shapes[k]
            h, w = _hw(sh)
            anc = (int(rng.integers(0, G - h + 1)), int(rng.integers(0, G - w + 1)))
            und = int(lvl["under"] and rng.random() < 0.4)
            ops.append((int(rng.integers(1, nc + 1)), k, anc, und))
        blank = tuple([0] * (G * G))

        def run(seq):
            cv = blank
            for col, k, anc, und in seq:
                cv = self._paint(lvl, cv, col, shapes[k], anc, und)
            return cv
        F = run(ops)
        changed = True
        while changed:                                         # élaguer les couches invisibles
            changed = False
            for j in range(len(ops)):
                if run(ops[:j] + ops[j + 1:]) == F:
                    ops = ops[:j] + ops[j + 1:]
                    changed = True
                    break
        if len(ops) < max(2, K - 3 - K // 4):
            return self.why("pruned")
        if newest == "under":
            if not any(o[3] for o in ops) or run([(a, b, c, 0) for a, b, c, d in ops]) == F:
                return self.why("under_unused")
        if newest == "colors" and len({o[0] for o in ops}) < nc:
            return self.why("colors_unused")
        if newest in ("shapes", "shapes2") and not any(shapes[o[1]] in (("bh", "bv") if newest == "shapes"
                                                                          else ("plus", "ring")) for o in ops):
            return self.why("shape_unused")
        if sum(v > 0 for v in F) < G:
            return self.why("sparse")
        lvl["target"] = F
        anc0 = (0, 0) if self.variant == "brush" else None
        st0 = (blank, 0, 0 if self.variant == "brush" else -1, anc0, 0)
        # solution : sélections + (déplacements) + application
        sol, st = [], st0
        for col, k, anc, und in ops:
            if st[1] != col:
                y, x = lvl["pal_px"][col - 1]
                sol.append(("click", (y + 2, x + 2)))
            if st[2] != k:
                y, x = lvl["tool_px"][k]
                sol.append(("click", (y + 2, x + 2)))
            if st[4] != und:
                y, x = lvl["under_px"]
                sol.append(("click", (y + 2, x + 2)))
            # rejouer les sélections
            st = self._replay(lvl, st0, sol)
            if self.variant == "stamp":
                sol.append(("click", (lvl["oy"] + anc[0] * self.P + 2, lvl["ox"] + anc[1] * self.P + 2)))
            else:
                a = st[3]
                while a[0] != anc[0]:
                    sol.append(("down" if anc[0] > a[0] else "up", None))
                    a = (a[0] + (1 if anc[0] > a[0] else -1), a[1])
                while a[1] != anc[1]:
                    sol.append(("right" if anc[1] > a[1] else "left", None))
                    a = (a[0], a[1] + (1 if anc[1] > a[1] else -1))
                sol.append(("interact", None))
            st = self._replay(lvl, st0, sol)
        if st[0] != F:
            return self.why("replay")
        rem, x = {}, st0                                       # progrès : reste de la solution (sur le chemin)
        for k, mv in enumerate(sol):
            rem[x] = len(sol) - k
            x, _ = self.apply(lvl, x, *mv)
        lvl["rem"] = rem
        return lvl, st0, sol

    def _replay(self, lvl, st, sol):
        for mv in sol:
            st, _ = self.apply(lvl, st, *mv)
        return st

    def min_len(self, i, n):
        return min(8 + 5 * i, 40)

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        _, tag = self._step(lvl, st, op, px)
        if tag == "paint":
            ev[EV["color"]] = 1
            a = sum(x == y for x, y in zip(st[0], lvl["target"]))
            b = sum(x == y for x, y in zip(st2[0], lvl["target"]))
            if b > a:
                ev[EV["collect"]] = 1
        elif tag in ("select", "tool", "toggle"):
            ev[EV["toggle"]] = 1
        elif tag == "bump":
            ev[EV["touch"]] = 1

    # ---------------------------------------------------------------- rendu
    def _blankc(self):
        pal = self.pal
        for c in (draw.WHITE, draw.OFFW, draw.LGREY, draw.BLACK):
            if c != pal.bg and c != pal.ui:
                return c
        return draw.GREY

    def _colours(self, lvl):
        return [self.pal.bg] + [self.pal.a(k) for k in range(lvl["nc"])]       # case vierge = fond

    def render(self, lvl, st):
        pal = self.pal
        g = self.blank()
        cv, col, tool, anc, und = st
        G, P = lvl["G"], self.P
        cols = self._colours(lvl)
        hi = draw.WHITE if pal.bg != draw.WHITE else draw.BLACK
        # bandeau : vignette, palette, outils
        band = lvl["band"]
        tgt = np.array(lvl["target"]).reshape(G, G)
        tgt = np.rot90(tgt, lvl["rot"])
        s = 2
        frame_rect(g, 1, 1, G * s + 2, G * s + 2, hi)
        for r in range(G):
            for c in range(G):
                rect(g, 2 + r * s, 2 + c * s, s, s, cols[int(tgt[r, c])])
        for k, (y, x) in enumerate(lvl["pal_px"]):
            rect(g, y, x, 5, 5, cols[k + 1])
            if col == k + 1:                                  # sélection : petit trait sous l'échantillon
                rect(g, y + 6, x, 1, 5, hi)
        for k, (y, x) in enumerate(lvl["tool_px"]):
            sh = lvl["shapes"][k]
            h, w = _hw(sh)
            sc = 2
            oy, ox = y + (6 - h * sc) // 2, x + (6 - w * sc) // 2
            for a, b in SHAPES[sh]:
                rect(g, oy + a * sc, ox + b * sc, sc, sc, pal.line)
            if tool == k:
                rect(g, y + 7, x, 1, 6, hi)
        if lvl["under_px"] is not None:
            y, x = lvl["under_px"]
            frame_rect(g, y, x, 5, 5, pal.line)
            rect(g, y + 3, x + 1, 1, 3, pal.line)
            if und:
                rect(g, y + 1, x + 1, 2, 3, pal.a(5))
                frame_rect(g, y - 1, x - 1, 7, 7, hi)
        if lvl["rot"]:                                         # flèche de rotation sous la vignette
            draw.hline(g, G * s + 4, 2, G * s + 1, pal.line, dotted=2)
        # toile
        oy, ox = lvl["oy"], lvl["ox"]
        frame_rect(g, oy - 1, ox - 1, G * P + 2, G * P + 2, pal.wall if pal.wall != pal.bg else pal.line)
        for r in range(G):
            for c in range(G):
                rect(g, oy + r * P, ox + c * P, P, P, cols[cv[r * G + c]])
        if lvl["mirror"]:                                      # axe du miroir : repères au-dessus et au-dessous
            ax = ox + (G * P) // 2
            draw.vline(g, ax, oy - 4, oy - 2, pal.line)
            draw.vline(g, ax, oy + G * P + 1, oy + G * P + 3, pal.line)
        if self.variant == "brush":
            sh = lvl["shapes"][tool]
            for a, b in SHAPES[sh]:
                r, c = anc[0] + a, anc[1] + b
                y, x = oy + r * P, ox + c * P
                frame_rect(g, y, x, P, P, hi if cols[cv[r * G + c]] != hi else pal.line)
        return g
