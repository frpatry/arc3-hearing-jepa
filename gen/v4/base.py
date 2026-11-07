"""V4Game : socle commun des familles v4.

Différences avec v3 (mixin V3) :
  - rendu NATIF PLEIN ÉCRAN : render() dessine toute l'aire de jeu (62×64 ou 64×62 à côté de la barre d'1 px, ou la
    moitié à l'échelle ×2) en couleurs officielles (draw.RAW + c) : scènes clairsemées, pièces, sprites dessinés ;
  - palette par RÔLES (draw.Pal) tirée selon l'usage officiel (fond noir ~ 46 %, sinon gris, bleus, blanc, orange) ;
  - sens des actions CANONIQUE dans 80 % des jeux (ACTION1-4 = haut, bas, gauche, droite comme les officiels) ;
  - coût des actions : chaque action coûte un pas de la barre, SAUF (free_miss) les clics « dans le vide » de certains
    jeux (officiel : un clic raté ne coûte rien dans plusieurs jeux) ; budget plus serré (1,6-3× la solution) ;
  - vérification oracle : chaque solution est rejouée par apply() jusqu'à la victoire, sans mort ;
  - niveaux tardifs introuvables : jeu tronqué (au moins `min_levels`).
Les 23 événements (toucher, corps, son, progrès) sont remplis par le noyau + fam_events() de chaque famille."""
import numpy as np

from ..core import BG, DETAIL, HUD_FILL, LETTERBOX, LIVES, NSEM, CANON, Game, Unsolvable
from . import draw
from .draw import RAW, Pal


class V4Game(Game):
    family = "v4base"
    mechanic = "object"                     # avatar | click | object (mélange v4)
    principle = ("", [])                    # (principe, jeux officiels qui le partagent)
    weight = 1.0                            # poids dans le mélange v4 (0 = famille désactivée)
    n_levels_official = (5, 9)
    min_levels = 4
    scale_p = (0.8, 0.2)                    # échelle d'écran ×1 / ×2
    free_miss = False                       # un clic sans effet ne coûte pas de pas
    bfs_cap = 40000
    table_cap = 20000
    canon_p = 0.8
    budget_mult = (1.6, 4.0)
    relax = 0                               # 1 pendant la 2e passe de génération d'un niveau difficile
    p_big_budget = 0.4                      # budget > longueur de la barre : beaucoup de pas ne changent AUCUN pixel

    def __init__(self, seed=0, look="official"):
        super().__init__(seed, "official")

    # ------------------------------------------------------------------ vue / présentation
    def _official_view(self, r):
        s = int(r.choice([1, 2], p=list(self.scale_p)))
        self.s_screen, self.T = s, 1
        self.bar_edge = str(r.choice(["top", "bottom", "left", "right"], p=[0.2, 0.6, 0.1, 0.1]))   # officiel : bas surtout
        self.bar_rev = bool(r.random() < 0.3)            # la barre se vide par l'autre bout
        self.energy_death = bool(r.random() < 0.8)
        self.lives_max = int(r.choice([0] * 9 + [3]))       # vies rares (1 jeu officiel sur 25 en montre)
        self.panel = bool(r.random() < 0.5)
        self.wall_mode = "block"
        self.sel_style = str(r.choice(["white", "dot"]))
        ah = 62 if self.bar_edge in ("top", "bottom") else 64
        aw = 64 if self.bar_edge in ("top", "bottom") else 62
        self.AH, self.AW = ah // s, aw // s          # aire logique dessinée par render()
        self.max_cells = 99
        self.px_budget = min(self.AH, self.AW)

    def prefer_bg(self, r):
        """couleur de fond imposée par la famille (None = tirage selon l'usage officiel)."""
        return None

    def _official_presentation(self, r):
        super()._official_presentation(r)
        self.pal = Pal(r, self.prefer_bg(r))
        pal = np.zeros(RAW + 16, np.int64)
        pal[:NSEM] = self.palette
        pal[RAW:] = np.arange(16)
        pal[BG] = self.pal.bg
        pal[LETTERBOX] = self.pal.bg if r.random() < 0.75 else self.pal.edge
        bar_c = [c for c in (draw.LBLUE, draw.YELLOW, draw.GREEN, draw.WHITE, draw.ORANGE, draw.BLUE, draw.MAGENTA,
                             draw.LGREY, draw.RED) if c not in (pal[LETTERBOX], self.pal.bg)]
        pal[HUD_FILL] = int(r.choice(bar_c))
        pal[LIVES] = draw.RED if pal[LETTERBOX] != draw.RED else draw.ORANGE
        self.bar_rest = None if r.random() < 0.6 else int(r.choice([c for c in (draw.OFFW, draw.LGREY, draw.GREY,
                                                                                   draw.DGREY) if c != pal[HUD_FILL]]))
        pal[DETAIL] = self.pal.fg
        self.palette = pal
        self.energy_mult = float(r.uniform(*self.budget_mult))
        self.big_budget = bool(r.random() < self.p_big_budget)
        if r.random() < self.canon_p:                # sens canonique des actions (comme les officiels)
            amap = {CANON[o]: o for o in self.simple_ops}
            if self.undo:
                amap[7] = "undo"
            if self.uses_click:
                amap[6] = "click"
            extra = [i for i in range(1, 8) if i not in amap]
            for i in r.permutation(extra)[:int(r.choice([0, 0, 1]))]:
                amap[int(i)] = None
            self.action_map = amap
            self.available_actions = sorted(amap)
            self.op_to_id = {o: i for i, o in amap.items() if o is not None}

    # ------------------------------------------------------------------ rendu plein écran
    def blank(self, c=None):
        return np.full((self.AH, self.AW), RAW + (self.pal.bg if c is None else c), np.int64)

    def frame(self):
        L = self.levels[self.level]
        g = self._render(self.level, self.st)
        s, oy, ox = self._layout(g.shape)
        scr = np.full((64, 64), LETTERBOX, np.int64)
        big = np.repeat(np.repeat(g, s, 0), s, 1) if s > 1 else g
        scr[oy:oy + big.shape[0], ox:ox + big.shape[1]] = big
        self._draw_bar(scr, L)
        return self.palette[scr].astype(np.int8)

    def _draw_bar(self, scr, L):
        """barre de pas d'1 px PROPORTIONNELLE (pleine sur toute la longueur au début du niveau, comme les officiels),
        partie vide = pourtour ou un gris ; vies (rares) en bout de barre."""
        e = self.bar_edge
        line = scr[0, :] if e == "top" else scr[63, :] if e == "bottom" else scr[:, 0] if e == "left" else scr[:, 63]
        line[:] = LETTERBOX if self.bar_rest is None else RAW + self.bar_rest
        nl = self.lives_max
        length = 64 - (3 * nl + 1 if nl else 0)
        B = L["budget"]
        left = max(0, self.energy)
        n = int(np.ceil(length * left / B)) if B else 0
        if self.bar_rev:
            line[length - n:length] = HUD_FILL
        else:
            line[:n] = HUD_FILL
        for k in range(nl):
            if k < self.lives:
                line[63 - 3 * k - 1:63 - 3 * k + 1] = LIVES

    # ------------------------------------------------------------------ coût des actions
    def step(self, action_id, data=None):
        e0, la0, ta0, li0, st0, state0 = self.energy, self.level_actions, self.total_actions, self.level, self.st, self.state
        obs = super().step(action_id, data)
        if (self.free_miss and int(action_id) == 6 and state0 == "NOT_FINISHED" and self.state == "NOT_FINISHED"
                and self.level == li0 and self.st == st0 and e0 is not None and self.energy == e0 - 1):
            self.energy, self.level_actions, self.total_actions = e0, la0, ta0       # clic raté : gratuit
            obs["frame"] = self.frame()
        return obs

    # ------------------------------------------------------------------ niveaux vérifiés
    def _make_level(self, i, n):
        try:
            L = super()._make_level(i, n)
        except Unsolvable:
            if i >= self.min_levels:
                return None
            self.relax = 1                            # 2e passe relâchée (les familles peuvent lire self.relax)
            try:
                L = super()._make_level(i, n)
            finally:
                self.relax = 0
        if L["budget"] is not None:
            L["budget"] = int(max(L["budget"], 24))   # barre proportionnelle : <= 3 px par pas
            if self.big_budget:
                L["budget"] = int(max(L["budget"], self.orng.integers(80, 220)))
        lvl, st = L["lvl"], L["st0"]
        for mv in L["sol"]:
            st, dead = self.apply(lvl, st, *mv)
            assert not dead, (self.game_id, i, "solution mortelle")
        assert self.is_win(lvl, st), (self.game_id, i, "solution non gagnante")
        return L

    def why(self, reason):
        """journal des rejets de niveau (diagnostic) -> None."""
        self.__dict__.setdefault("rejects", {})
        self.rejects[reason] = self.rejects.get(reason, 0) + 1
        return None

    def info(self):
        d = super().info()
        d["rules_per_level"] = [list(L["lvl"].get("mods", ())) if isinstance(L["lvl"], dict) else [] for L in self.levels]
        d["variant"] = getattr(self, "variant", None)
        return d


def mods_for(cat, i, keep=3, start=1):
    """curriculum : les niveaux >= start ajoutent cat[0], cat[1], … (au plus `keep` règles ajoutées actives) ;
    -> (règles actives, règle nouvelle de ce niveau ou None)."""
    k = max(0, i - start + 1)
    new = list(cat[:k])
    return tuple(new[-keep:]), (new[-1] if new and k <= len(cat) else None)
