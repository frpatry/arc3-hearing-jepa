"""rewrite4 (clic / curseur, réécriture de symboles) : un TABLEAU DE RÈGLES (runes gauche -> runes droite) en haut,
une rangée SOURCE, une rangée RÉPONSE à éditer en bas ; la réponse doit être la traduction de la source par les
règles (vérifiée automatiquement après chaque changement). Une légende en bas montre l'ordre de cycle des runes.
Variantes (contrôle) :
  cycle  : clic sur une case éditable = rune suivante dans l'ordre de la légende ;
  cursor : gauche / droite déplacent le crochet, haut = rune suivante sous le crochet, bas = effacer la case (un clic
           sur une case éditable y place le crochet).
Règles ajoutées (ordre tiré par jeu, une par niveau, mods_for) :
  chain   : règles en chaîne (A -> B, B -> C : on réécrit jusqu'au point fixe) ;
  pair    : règle 2 -> 1 (AB -> C, lue en priorité, de gauche à droite) ;
  expand  : règle 1 -> 2 (A -> BC) ;
  rotated : runes de la source (et des règles) affichées tournées : seule l'identité compte ;
  decoy   : règles leurres (jamais utilisées) ;
  invert  : INVERSION : la réponse est affichée, fixe ; on édite le côté droit des règles.
Solvabilité : par construction (source, règles, traduction) ; coût exact = distance cyclique par case (+ trajet du
curseur) -> oracle et progrès exacts."""
import numpy as np

from ..core import EV
from . import draw
from .base import V4Game, mods_for
from .draw import R, blit, frame_rect, rect


def _glyphs(rng, n):
    """n runes 5×5 distinctes même à rotation près, sans symétrie de rotation."""
    out, keys = [], set()
    while len(out) < n:
        m = np.zeros((5, 5), bool)                        # traits : 2-4 segments droits d'épaisseur 1
        for _ in range(int(rng.integers(2, 5))):
            L = int(rng.integers(3, 6))
            if rng.random() < 0.5:
                y, x = int(rng.integers(5)), int(rng.integers(0, 6 - L))
                m[y, x:x + L] = True
            else:
                y, x = int(rng.integers(0, 6 - L)), int(rng.integers(5))
                m[y:y + L, x] = True
        ys, xs = np.nonzero(m)
        if m.sum() < 8 or m.sum() > 15 or np.ptp(ys) < 3 or np.ptp(xs) < 3:
            continue
        lab = draw.largest_component(m)
        if lab.sum() != m.sum():
            continue
        rots = [np.rot90(m, k) for k in range(4)]
        if any((rots[0] == r).all() for r in rots[1:]):
            continue
        ks = {r.tobytes() for r in rots}
        if ks & keys:
            continue
        keys |= ks
        out.append(m)
    return out


class Rewrite4(V4Game):
    family = "rewrite4"
    mechanic = "click"
    principle = ("réécriture de symboles : traduire une rangée de runes par un tableau de règles affiché "
                 "(chaînes, 2->1, 1->2, runes tournées, inversion : éditer les règles)", ["tr87", "sb26", "ft09"])
    simple_ops = ()
    CAT = ("chain", "pair", "expand", "rotated", "decoy", "invert")
    free_miss = True
    exact_progress = False
    scale_p = (1.0, 0.0)
    n_levels_official = (5, 8)

    def setup(self, rng):
        r = self.orng
        self.variant = str(rng.choice(["cycle", "cursor"]))
        self.uses_click = True
        if self.variant == "cursor":
            self.simple_ops = ("up", "down", "left", "right")
        self.cat = [str(x) for x in rng.permutation(self.CAT)]
        if "invert" in self.cat[:2]:                         # l'inversion jamais trop tôt
            k = self.cat.index("invert")
            self.cat[k], self.cat[3] = self.cat[3], self.cat[k]
        self.K = int(rng.choice([7, 8]))                     # 0 = vide, 1..K-1 = runes
        self.gl = _glyphs(r, self.K - 1)
        self.ncol = int(r.choice([2, 3]))
        self.src_panel = bool(r.random() < 0.5)

    # ---------------------------------------------------------------- traduction
    def rules_of(self, lvl, st):
        """liste de (lhs, rhs) avec les côtés droits éventuellement édités."""
        if not lvl["invert"]:
            return lvl["rules"]
        out, k = [], 0
        for lhs, rhs in lvl["rules"]:
            n = len(rhs)
            out.append((lhs, tuple(st[0][k:k + n])))
            k += n
        return out

    @staticmethod
    def translate(rules, src, chain):
        pairs = {lhs: rhs for lhs, rhs in rules if len(lhs) == 2}
        singles = {lhs[0]: rhs for lhs, rhs in rules if len(lhs) == 1}
        out, p = [], 0
        while p < len(src):
            if p + 1 < len(src) and (src[p], src[p + 1]) in pairs:
                out += list(pairs[(src[p], src[p + 1])])
                p += 2
            else:
                out += list(singles.get(src[p], (src[p],)))
                p += 1
        if chain:
            for _ in range(4):
                nxt = []
                for g in out:
                    r = singles.get(g)
                    nxt += list(r) if (r is not None and len(r) == 1) else [g]
                if nxt == out:
                    break
                out = nxt
        return tuple(out)

    def is_win(self, lvl, st):
        if lvl["invert"]:
            rules = self.rules_of(lvl, st)
            return all(self.translate(rules, sr, lvl["chain"]) == o for sr, o in zip(lvl["srcs"], lvl["outs"]))
        return tuple(st[0]) == lvl["out"]

    # ---------------------------------------------------------------- géométrie
    def _edit_px(self, lvl, k):
        """coin haut-gauche de la case éditable k."""
        if lvl["invert"]:
            return lvl["rhs_px"][k]
        return lvl["ans_pos"][k]

    def _edit_at(self, lvl, px):
        if px is None:
            return None
        y, x = px
        for k in range(lvl["n_edit"]):
            ey, ex = self._edit_px(lvl, k)
            if ey <= y < ey + 5 and ex <= x < ex + 5:
                return k
        return None

    # ---------------------------------------------------------------- règles du jeu
    def _src_at(self, lvl, px):
        y, x = px
        for j, (sy, sx) in enumerate(lvl["src_pos"]):
            if sy <= y < sy + 5 and sx <= x < sx + 5:
                return j
        return None

    def apply(self, lvl, st, op, px):
        vals, cur, hl = st
        K = self.K
        if op == "click":
            k = self._edit_at(lvl, px)
            if k is None:
                j = self._src_at(lvl, px) if px is not None else None
                if j is not None and j != hl:                  # inspecter : souligne la rune et sa règle
                    return (vals, cur, j), False
                return st, False
            if self.variant == "cursor":
                return ((vals, k, hl) if k != cur else st), False
            v = list(vals)
            v[k] = (v[k] + 1) % K
            return (tuple(v), cur, hl), False
        if self.variant != "cursor":
            return st, False
        if op == "left":
            return ((vals, cur - 1, hl) if cur > 0 else st), False
        if op == "right":
            return ((vals, cur + 1, hl) if cur < lvl["n_edit"] - 1 else st), False
        v = list(vals)
        if op == "up":                                         # haut : rune suivante ; bas : effacer la case
            v[cur] = (v[cur] + 1) % K
        elif v[cur] != 0:
            v[cur] = 0
        else:
            return st, False
        return (tuple(v), cur, hl), False

    def _want(self, lvl, st):
        if not lvl["invert"]:
            return lvl["out"]
        w = list(st[0])
        for k in lvl["used_idx"]:
            w[k] = lvl["rhs_true"][k]
        return tuple(w)

    def _cost(self, lvl, st):
        vals, cur, _ = st
        if self.is_win(lvl, st):
            return 0, None
        want = self._want(lvl, st)
        K = self.K
        wrong = [k for k in range(len(vals)) if vals[k] != want[k]]
        if not wrong:
            return 1, None
        if self.variant == "cycle":
            c = sum((want[k] - vals[k]) % K for k in wrong)
            return c, (wrong[0] if wrong else None)
        lo, hi = min(wrong), max(wrong)
        travel = (hi - lo) + min(abs(cur - lo), abs(cur - hi))
        c = travel + sum(min((want[k] - vals[k]) % K, 1 + want[k]) for k in wrong)
        return c, (lo, hi)

    def _plan(self, lvl, st):
        vals, cur, _ = st
        c, info = self._cost(lvl, st)
        if c == 0 or info is None:
            return None
        want = self._want(lvl, st)
        K = self.K
        if self.variant == "cycle":
            y, x = self._edit_px(lvl, info)
            return ("click", (y + 2, x + 2))
        if vals[cur] != want[cur]:
            return ("up", None) if (vals[cur] == 0 or (want[cur] - vals[cur]) % K <= 1 + want[cur]) else ("down", None)
        lo, hi = info
        # aller vers l'extrémité la plus proche (trajet optimal sur une ligne)
        if cur < lo:
            return ("right", None)
        if cur > hi:
            return ("left", None)
        return ("left", None) if abs(cur - lo) <= abs(cur - hi) else ("right", None)

    def oracle_action(self):
        if self.state != "NOT_FINISHED":
            return 0, None
        L = self.levels[self.level]
        mv = self._plan(L["lvl"], self.st)
        if mv is None:
            return 0, None
        return self._to_action(mv[0], mv[1], self.st)

    def heuristic(self, lvl, st):
        return self._cost(lvl, st)[0]

    def click_targets(self, lvl, st):
        return [(self._edit_px(lvl, k)[0] + 2, self._edit_px(lvl, k)[1] + 2) for k in range(lvl["n_edit"])]

    def click_hit(self, lvl, st, px):
        if self._edit_at(lvl, px) is not None:
            return True
        return int(self.render(lvl, st)[px]) != draw.RAW + self.pal.bg

    def body(self, lvl, st):
        if self.variant != "cursor":
            return None
        return (0, st[1], 1, lvl["n_edit"])

    # ---------------------------------------------------------------- génération
    def gen_level(self, rng, i, n):
        mods, newest = mods_for(self.cat, i)
        K = self.K
        chain, pair, expand = "chain" in mods, "pair" in mods, "expand" in mods
        invert = "invert" in mods
        letters = list(range(1, K))
        rng.shuffle(letters)
        m = 3 if i == 0 else (5 if invert else 4)
        lhs1 = letters[:m]                                    # runes de la source
        others = [g for g in letters if g not in lhs1]
        rules = []
        for k, a in enumerate(lhs1):
            if chain and k == 0:
                b = lhs1[1]                                   # A -> B, B -> …
            else:
                b = int(rng.choice(others if not chain else others + lhs1[2:]))
                if b == a:
                    b = int(rng.choice(others))
            rules.append(((a,), (b,)))
        if expand:
            k = int(rng.integers(1, m))
            a = rules[k][0]
            rules[k] = (a, (int(rng.choice(others)), int(rng.choice(letters))))
        if pair:
            a, b = int(rng.choice(lhs1)), int(rng.choice(lhs1))
            rules.append(((a, b), (int(rng.choice(others)),)))
        if "decoy" in mods:
            for g in others[:1 + (i >= 6)]:
                rules.append(((g,), (int(rng.choice(letters)),)))
        # sources : 1 rangée, puis 2 (niveaux >= 2) ; la traduction se fait rangée par rangée
        nrows = 1 + (i >= 1 and len(rules) <= 6)
        srcs = []
        for r_ in range(nrows):
            L = int(min(8, 4 + (i >= 1) + (i >= 3) + int(rng.integers(0, 3))))
            src = [int(rng.choice(lhs1)) for _ in range(L)]
            if pair and (r_ == 0 or rng.random() < 0.5):      # au moins une paire, et ses runes seules ailleurs
                pa = next(lhs for lhs, _ in rules if len(lhs) == 2)
                p = int(rng.integers(0, L - 1))
                src[p], src[p + 1] = pa
            if chain and r_ == 0 and lhs1[0] not in src:
                src[int(rng.integers(L))] = lhs1[0]
            srcs.append(tuple(src))
        srcs = tuple(srcs)
        outs = tuple(self.translate(rules, sr, chain) for sr in srcs)
        if any(len(o) > 8 or len(o) < 3 for o in outs):
            return self.why("len")
        if any(0 in o for o in outs):
            return self.why("blank")
        src = tuple(g_ for sr in srcs for g_ in sr)
        out = tuple(g_ for o in outs for g_ in o)
        # la règle nouvelle doit changer la traduction
        if newest in ("chain", "pair", "expand"):
            if newest == "pair":
                base = [(l_, r_) for l_, r_ in rules if len(l_) == 1]
            elif newest == "expand":
                base = [(l_, r_[:1]) for l_, r_ in rules]
            else:
                base = list(rules)
            plain = tuple(self.translate(base, sr, chain and newest != "chain") for sr in srcs)
            if plain == outs:
                return self.why("rule_unused")
        if len(rules) > 8:
            return self.why("rules")
        rng_rot = rng.integers(0, 4, len(src)) if "rotated" in mods else np.zeros(len(src), int)
        lvl = dict(rules=tuple(rules), src=src, out=out, srcs=srcs, outs=outs, chain=chain, invert=invert, mods=mods,
                   rot=tuple(int(x) for x in rng_rot),
                   lrot=tuple(int(x) for x in (rng.integers(0, 4, 2 * len(rules)) if "rotated" in mods
                                                else np.zeros(2 * len(rules), int))))
        if not self._geom(lvl):
            return self.why("geom")
        if invert:
            rhs_true = [g for _, rhs in rules for g in rhs]
            used = set()                                      # règles utilisées par la source
            k0 = 0
            for (lhs, rhs) in rules:
                test = list(rules)
                alt = tuple(((rhs[0] % (K - 1)) + 1,) + rhs[1:])
                test[rules.index((lhs, rhs))] = (lhs, alt)
                if any(self.translate(test, sr, chain) != o for sr, o in zip(srcs, outs)):
                    used |= set(range(k0, k0 + len(rhs)))
                k0 += len(rhs)
            lvl["used_idx"] = tuple(sorted(used))
            lvl["rhs_true"] = tuple(rhs_true)
            vals = list(rhs_true)
            for k in used:
                vals[k] = 0
            lvl["n_edit"] = len(vals)
            vals = tuple(vals)
        else:
            lvl["n_edit"] = len(out)
            vals = tuple(0 if (i < 2 or rng.random() < 0.5) else int(rng.integers(1, K)) for _ in out)
        st0 = (vals, 0, -1)
        if self.is_win(lvl, st0):
            return self.why("trivial")
        sol, st = [], st0
        for _ in range(400):
            mv = self._plan(lvl, st)
            if mv is None:
                break
            sol.append(mv)
            st, _ = self.apply(lvl, st, *mv)
        if not self.is_win(lvl, st):
            return self.why("plan")
        return lvl, st0, sol

    def _geom(self, lvl):
        AH, AW = self.AH, self.AW
        rules = lvl["rules"]
        nr = (len(rules) + 1) // 2
        lvl["panel_h"] = 3 + 8 * nr
        pos, rhs_px = [], []
        for k, (lhs, rhs) in enumerate(rules):
            y = 3 + 8 * (k // 2)
            x = 2 if k % 2 == 0 else AW // 2 + 1
            pos.append((y, x))
            xr = x + 6 * 2 + 5
            for j in range(len(rhs)):
                rhs_px.append((y, xr + 6 * j))
        lvl["rule_px"] = tuple(pos)
        lvl["rhs_px"] = tuple(rhs_px)
        top = lvl["panel_h"] + 3
        nr_ = len(lvl["srcs"])
        step = 16 if nr_ > 1 else 20
        sp, ap = [], []
        for r, (sr, o) in enumerate(zip(lvl["srcs"], lvl["outs"])):
            sy = top + 3 + step * r if nr_ > 1 else top + 4
            ay = sy + 7 if nr_ > 1 else min(AH - 8, top + 20)
            sx = (AW - 7 * len(sr)) // 2 + 1
            ax = (AW - 7 * len(o)) // 2 + 1
            sp += [(sy, sx + 7 * j) for j in range(len(sr))]
            ap += [(ay, ax + 7 * j) for j in range(len(o))]
        lvl["src_pos"], lvl["ans_pos"] = tuple(sp), tuple(ap)
        lvl["src_y"] = sp[0][0]
        if ap[-1][0] + 5 > AH:
            return False
        return True

    def min_len(self, i, n):
        return min(8 + 5 * i, 40)

    def fam_events(self, lvl, st, op, px, st2, dead, ev):
        if st2 == st:
            if op == "click" and self._edit_at(lvl, px) is None:
                return
            ev[EV["touch"]] = 1
            return
        if st2[2] != st[2]:
            ev[EV["toggle"]] = 1
            return
        if st2[0] != st[0]:
            ev[EV["color"]] = 1
            want = self._want(lvl, st)
            if sum(a == b for a, b in zip(st2[0], want)) > sum(a == b for a, b in zip(st[0], want)):
                ev[EV["collect"]] = 1
        else:
            ev[EV["toggle"]] = 1

    # ---------------------------------------------------------------- rendu
    def _gcol(self, g):
        return self.pal.a(g % self.ncol)

    def _glyph(self, g, rot=0):
        s = np.full((5, 5), -1, np.int64)
        if g == 0:                                             # case vide : petit carré plein
            s[1:4, 1:4] = self.pal.wall if self.pal.wall != self.pal.bg else self.pal.line
            return s
        m = np.rot90(self.gl[g - 1], rot)
        s[m] = self._gcol(g)
        return s

    def render(self, lvl, st):
        pal = self.pal
        g = self.blank()
        vals, cur, hl = st
        rules = self.rules_of(lvl, st)
        hc = draw.WHITE if pal.bg != draw.WHITE else draw.BLACK
        # tableau des règles, posé sur le fond (gauche  ·  droite)
        for k, (lhs, rhs) in enumerate(rules):
            y, x = lvl["rule_px"][k]
            for j, a in enumerate(lhs):
                blit(g, self._glyph(a, lvl["lrot"][2 * k + j] if j < 2 else 0), y, x + 6 * j)
        k0 = 0
        for k, (lhs, rhs) in enumerate(rules):
            for j, b in enumerate(rhs):
                y, x = lvl["rhs_px"][k0 + j]
                if lvl["invert"]:
                    b = vals[k0 + j]
                blit(g, self._glyph(b), y, x)
            k0 += len(rhs)
        sep = lvl["panel_h"] + 1                                # un seul trait de séparation
        draw.hline(g, sep, 2, self.AW - 3, pal.wall if pal.wall != pal.bg else pal.line)
        # sources et réponses
        for j, a in enumerate(lvl["src"]):
            y, x = lvl["src_pos"][j]
            blit(g, self._glyph(a, lvl["rot"][j]), y, x)
        if hl >= 0:                                            # rune inspectée : soulignée, sa règle encadrée
            y, x = lvl["src_pos"][hl]
            draw.hline(g, y + 6, x, x + 4, hc)
            a = lvl["src"][hl]
            for k, (lhs, rhs) in enumerate(rules):
                if lhs[0] == a and len(lhs) == 1:
                    ry, rx = lvl["rule_px"][k]
                    frame_rect(g, ry - 1, rx - 1, 7, 7, hc)
        lc = pal.wall if pal.wall != pal.bg else pal.line
        for j in range(len(lvl["out"])):
            y, x = lvl["ans_pos"][j]
            b = lvl["out"][j] if lvl["invert"] else vals[j]
            if b:
                blit(g, self._glyph(b), y, x)
        if not lvl["invert"]:                                   # une ligne de base sous chaque rangée-réponse
            rows = {}
            for (y, x) in lvl["ans_pos"]:
                rows.setdefault(y, []).append(x)
            for y, xs in rows.items():
                draw.hline(g, y + 6, min(xs), max(xs) + 4, lc)
        # curseur
        if self.variant == "cursor":
            cy, cx = self._edit_px(lvl, cur)
            frame_rect(g, cy - 1, cx - 1, 7, 7, hc)
        return g
