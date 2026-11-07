"""Familles v4 (gen/v4/fam_*.py) : jeux plus proches des officiels par la MÉCANIQUE et le COMPORTEMENT.
Principes : analysis/MECHANICS.md ; mesure de ressemblance : gen/discriminator.py (détecteur de faux).
Découverte automatique : chaque module gen/v4/fam_*.py expose des sous-classes de V4Game avec les attributs de classe
    family    = "xxx4"           nom de la famille
    mechanic  = "avatar" | "click" | "object"
    principle = ("principe", ["jeux officiels qui le partagent"])
    weight    = 1.0              poids dans le mélange v4
    FAMILY_CLASSES_V4 : famille -> classe (toujours en look officiel, rendu natif plein écran)
    MECHANIC_V4, PRINCIPLES_V4, WEIGHTS_V4"""
import importlib
import pkgutil
from pathlib import Path

from .v4.base import V4Game

FAMILY_CLASSES_V4 = {}
for _m in sorted(pkgutil.iter_modules([str(Path(__file__).resolve().parent / "v4")]), key=lambda m: m.name):
    if not _m.name.startswith("fam_"):
        continue
    _mod = importlib.import_module(f".v4.{_m.name}", __package__)
    for _v in vars(_mod).values():
        if isinstance(_v, type) and issubclass(_v, V4Game) and _v.__module__ == _mod.__name__ and \
                getattr(_v, "family", "v4base") != "v4base" and getattr(_v, "weight", 1.0) > 0:
            FAMILY_CLASSES_V4[_v.family] = _v
FAMILY_CLASSES_V4 = dict(sorted(FAMILY_CLASSES_V4.items()))
MECHANIC_V4 = {f: c.mechanic for f, c in FAMILY_CLASSES_V4.items()}
PRINCIPLES_V4 = {f: tuple(c.principle) for f, c in FAMILY_CLASSES_V4.items()}
WEIGHTS_V4 = {f: float(c.weight) for f, c in FAMILY_CLASSES_V4.items()}
