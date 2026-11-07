"""Registre des familles.
    make_game(family, seed, look="classic"|"official") -> Game      family="rules" : jeu à règles (grammaire)
    sample_game(seed, rules_share=0.25) -> Game   mélange v2 (+ part de jeux à règles ; rules_share=0 = v2 d'origine)
    FAMILIES, OFFICIAL_LIKE, PRINCIPLES (famille -> principe + jeux officiels qui le partagent : évaluation « principe tenu à l'écart »)
En look officiel, les familles d'avatar v1 (maze, keys_doors, switches, color_pads, teleport, hazards, ice, collect)
deviennent le monde à modules (families_world.World) avec leur mécanique comme base + un module nouveau par niveau."""
import zlib

import numpy as np

from .families_avatar import Collect, ColorPads, Hazards, Ice, KeysDoors, Maze, Sokoban, Switches, Teleport, Tilt
from .families_rules import make_rule_game
from .families_click import FloodIt, LightsOut, PaintCopy, RotateFit, SelectMove, ToggleMatch
from .families_v2 import CaptureMerge, CoupledBodies, GlyphKey, NpcRules, RuleTable, Vessels
from .families_world import BASE_OF, World
from .families_v3 import FAMILY_CLASSES_V3, MECHANIC_V3, WEIGHTS_V3
from .families_v4 import FAMILY_CLASSES_V4, PRINCIPLES_V4, WEIGHTS_V4

_CLASSES = [Maze, KeysDoors, Sokoban, Collect, Switches, ColorPads, Ice, Teleport, Hazards, Tilt,
            ToggleMatch, PaintCopy, LightsOut, FloodIt, SelectMove, RotateFit]
_V2 = [GlyphKey, RuleTable, CoupledBodies, Vessels, NpcRules, CaptureMerge]
FAMILY_CLASSES = {c.family: c for c in _CLASSES + _V2}
FAMILY_CLASSES["quest"] = World
FAMILIES_V1 = [c.family for c in _CLASSES]
FAMILIES_V2 = [c.family for c in _V2] + ["quest"]
FAMILIES = FAMILIES_V1 + FAMILIES_V2
FAMILIES_V3 = list(FAMILY_CLASSES_V3)          # v3 : toujours en look officiel (gen/families_v3.py)
FAMILIES_V4 = list(FAMILY_CLASSES_V4)          # v4 : rendu natif plein écran, mécaniques officielles (gen/v4/)

# principe à découvrir -> jeux officiels qui le partagent (cf. analysis/REPORT.md)
PRINCIPLES = {
    "maze": ("navigation vers une sortie", ["ls20", "tu93"]),
    "keys_doors": ("clé de couleur -> porte", ["ls20", "sc25"]),
    "sokoban": ("pousser des blocs sur des cibles", ["wa30", "ka59", "sk48"]),
    "collect": ("tout ramasser", ["wa30", "su15"]),
    "switches": ("boutons -> portails à distance", ["dc22", "g50t"]),
    "color_pads": ("recoloration de l'avatar, barrières de couleur", ["ls20", "re86"]),
    "ice": ("glissade jusqu'au blocage", ["bp35", "sp80"]),
    "teleport": ("téléporteurs", ["sc25", "lf52"]),
    "hazards": ("dangers / PNJ à éviter", ["tu93", "su15"]),
    "tilt": ("gravité : tout glisse ensemble", ["bp35", "sp80"]),
    "toggle_match": ("cycler des couleurs pour copier une référence", ["ft09", "cd82"]),
    "paint_copy": ("palette + peindre pour copier", ["cd82", "ft09"]),
    "lights_out": ("clic à effet de voisinage", ["ft09", "lp85"]),
    "flood_it": ("remplissage de région", ["cd82", "sp80"]),
    "select_move": ("sélectionner (clic) puis déplacer", ["lf52", "s5i5", "sk48"]),
    "rotate_fit": ("tourner / placer une pièce dans une silhouette", ["ar25", "cn04"]),
    "quest": ("navigation + un module nouveau par niveau", ["ls20", "tu93", "wa30"]),
    "glyph_key": ("transformer un glyphe-clé pour qu'il corresponde à la porte", ["ls20", "sc25", "cn04"]),
    "rule_table": ("appliquer un dictionnaire de réécriture affiché", ["tr87", "sb26", "tn36"]),
    "coupled_bodies": ("un contrôle, plusieurs corps couplés", ["m0r0", "ar25", "r11l", "g50t"]),
    "vessels": ("quantités conservées : verser, vannes", ["vc33", "su15"]),
    "npc_rules": ("prédire des PNJ à comportement propre (patrouille, imitation, poursuite)", ["tu93", "wa30", "g50t"]),
    "capture_merge": ("capturer en sautant (solitaire)", ["lf52", "su15"]),
    "rules": ("règles tirées par une grammaire (contrôle, interactions, but) : découvrir la règle, pas la mémoriser",
              ["ls20", "tu93", "wa30", "ft09", "bp35", "m0r0", "g50t", "sc25", "lp85"]),
    # v3 (niveaux longs, une règle nouvelle par niveau, statistiques visuelles officielles)
    "world3": ("navigation + un module nouveau par niveau (niveaux longs)", ["ls20", "tu93", "wa30", "dc22"]),
    "glyph3": ("transformer un glyphe-clé pour qu'il corresponde à la porte (niveaux longs)", ["ls20", "sc25", "cn04"]),
    "tiles3": ("clic à pochoir mod p (soi, croix, miroir, inversées, verrous) pour reproduire la cible", ["ft09", "lp85", "cd82"]),
    "jump3": ("capturer en sautant ; règles ajoutées (diagonale, 2 couleurs, case-cible)", ["lf52", "su15"]),
    "fleet3": ("sélectionner puis déplacer plusieurs pièces (barrières, poussée, paire liée, glissade, portails)",
               ["sk48", "s5i5", "ka59", "wa30"]),
    "coupled3": ("un contrôle, plusieurs corps couplés (niveaux longs)", ["m0r0", "ar25", "r11l", "g50t"]),
}


PRINCIPLES.update(PRINCIPLES_V4)


def make_game(family: str, seed: int, look: str = "classic"):
    if family in FAMILY_CLASSES_V4:             # v4 : look officiel imposé
        return FAMILY_CLASSES_V4[family](seed, "official")
    if family in FAMILY_CLASSES_V3:             # v3 : look officiel imposé
        return FAMILY_CLASSES_V3[family](seed, "official")
    if family == "rules":                       # jeu à règles tiré par la grammaire (gen/grammar.py)
        return make_rule_game(seed, look)
    if look == "official" and family in BASE_OF:
        return World(seed, look, family=family)
    return FAMILY_CLASSES[family](seed, look)


def game_specs(n: int, seed0: int = 0):
    """v1 : n couples (famille, graine) distincts, familles en tourniquet, look classique."""
    return [(FAMILIES_V1[k % len(FAMILIES_V1)], seed0 + k // len(FAMILIES_V1)) for k in range(n)]


def all_games(n: int, seed0: int = 0):
    return [make_game(f, s) for f, s in game_specs(n, seed0)]


# ---------------------------------------------------------------- mélange v2
OFFICIAL_SHARE = 0.7                  # part de jeux « façon officielle » (look officiel)
RULES_SHARE = 0.25                    # part de jeux à RÈGLES (grammaire) dans le mélange ; 0 = mélange v2 d'origine


def sample_spec(seed: int, official_share: float = OFFICIAL_SHARE, exclude=(), rules_share: float = RULES_SHARE):
    """(famille, graine, look) tiré de façon déterministe ; les familles v2 et le monde à modules sont privilégiés
    côté officiel ; `exclude` retire des familles (évaluation à principe tenu à l'écart) ; une part `rules_share`
    des graines donne un jeu à règles (tirage séparé : les autres graines gardent exactement le tirage v2)."""
    if rules_share > 0 and "rules" not in exclude:
        rr = np.random.default_rng([zlib.crc32(b"mixture-rules"), seed])
        if rr.random() < rules_share:
            return "rules", seed, "official" if rr.random() < official_share else "classic"
    r = np.random.default_rng([zlib.crc32(b"mixture"), seed])
    if r.random() < official_share:
        pool = [f for f in FAMILIES_V2 + FAMILIES_V1 if f not in exclude]
        w = np.array([3.0 if f in FAMILIES_V2 else 3.0 if f in BASE_OF else 1.0 for f in pool])
        fam = str(r.choice(pool, p=w / w.sum()))
        return fam, seed, "official"
    pool = [f for f in FAMILIES_V1 if f not in exclude]
    return str(r.choice(pool)), seed, "classic"


def sample_game(seed: int, official_share: float = OFFICIAL_SHARE, exclude=(), rules_share: float = RULES_SHARE):
    fam, s, look = sample_spec(seed, official_share, exclude, rules_share)
    return make_game(fam, s, look)


def mixture_specs(n: int, seed0: int = 0, official_share: float = OFFICIAL_SHARE, exclude=(),
                  rules_share: float = RULES_SHARE):
    return [sample_spec(seed0 + k, official_share, exclude, rules_share) for k in range(n)]


def rules_specs(n: int, seed0: int = 0, official_share: float = OFFICIAL_SHARE):
    """n jeux à règles seulement (graines seed0..seed0+n-1), look officiel pour une part `official_share`."""
    out = []
    for s in range(seed0, seed0 + n):
        r = np.random.default_rng([zlib.crc32(b"rules-look"), s])
        out.append(("rules", s, "official" if r.random() < official_share else "classic"))
    return out


# ---------------------------------------------------------------- mélange v3
def v3_spec(seed: int, exclude=()):
    """(famille v3, graine, "official") tiré de façon déterministe selon WEIGHTS_V3 (≈ 36 % avatar, 32 % clic,
    32 % interaction d'objets, comme les 25 jeux officiels)."""
    pool = [f for f in FAMILIES_V3 if f not in exclude]
    w = np.array([WEIGHTS_V3[f] for f in pool])
    r = np.random.default_rng([zlib.crc32(b"mixture-v3"), seed])
    return str(r.choice(pool, p=w / w.sum())), seed, "official"


def v3_specs(n: int, seed0: int = 0, exclude=()):
    return [v3_spec(seed0 + k, exclude) for k in range(n)]


# ---------------------------------------------------------------- mélange v4
def v4_spec(seed: int, exclude=()):
    """(famille v4, graine, "official") tiré de façon déterministe selon WEIGHTS_V4."""
    pool = [f for f in FAMILIES_V4 if f not in exclude]
    w = np.array([WEIGHTS_V4[f] for f in pool])
    r = np.random.default_rng([zlib.crc32(b"mixture-v4"), seed])
    return str(r.choice(pool, p=w / w.sum())), seed, "official"


def v4_specs(n: int, seed0: int = 0, exclude=()):
    return [v4_spec(seed0 + k, exclude) for k in range(n)]
