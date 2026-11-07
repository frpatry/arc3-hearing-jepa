"""Tests des familles v3 :   python -m gen.test_v3 [n_seeds] [seed0]
  - construction, solutions rejouées par step() niveau par niveau, l'oracle finit le jeu (gen.test_gen.check_game) ;
  - les 23 événements privilégiés (toucher / corps / son / progrès) sont émis, même disposition que le reste ;
  - curriculum : chaque niveau après le 1er ajoute ou change une règle (sauf world3/glyph3 : modules / CURR v2) ;
  - niveaux longs : baseline médiane par jeu >= 30 ;
  - demo_play : seulement oracle (0) et Explorer (1), aucune action aléatoire, épisodes cohérents."""
import sys
import time

import numpy as np

from .core import EV, EVENT_NAMES, N_EVENTS
from .families_v3 import FAMILY_CLASSES_V3, MECHANIC_V3
from .registry import make_game, v3_specs
from .rollouts import demo_game
from .test_gen import check_game


def check_rules_change(g):
    rl = [L["lvl"].get("mods", ()) if isinstance(L["lvl"], dict) else () for L in g.levels]
    if g.family in ("world3", "glyph3"):
        return
    if g.family == "coupled3":
        rl = [L["lvl"]["cps"] for L in g.levels]
    for i in range(1, min(len(rl), 5)):
        assert tuple(rl[i]) != tuple(rl[i - 1]), (g.game_id, i, rl[:i + 1])


def check_demo(g, seed):
    rng = np.random.default_rng(seed)
    rows, _ = demo_game(g, seed, 200, rng)
    assert rows, g.game_id
    beh = {r["behav"] for r in rows}
    assert beh <= {0, 1}, beh                                  # 0 oracle, 1 explorer : pas d'aléatoire
    ev = np.array([r["events"] for r in rows])
    assert ev.shape[1] == N_EVENTS == 23 and len(EVENT_NAMES) == 23
    prog = ev[:, EV["progress"]]
    assert np.all((prog >= -1) & (prog <= 1))
    snd = [EV[k] for k in ("bump", "push", "touch", "click_obj", "collect", "open", "color", "toggle", "teleport",
                           "danger", "level_complete", "game_over")]      # toucher + son (comme gen.stats)
    return len(rows), float((ev[:, snd] > 0).any(1).mean()), float((prog != 0).mean())


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    s0 = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
    t00 = time.time()
    for fam in FAMILY_CLASSES_V3:
        t0 = time.time()
        meds = []
        for s in range(s0, s0 + n):
            g = make_game(fam, s)
            check_game(g)
            check_rules_change(g)
            meds.append(float(np.median(g.baseline_actions)))
            nr, evr, pr = check_demo(make_game(fam, s), s)
        assert np.median(meds) >= 30, (fam, meds)
        print(f"{fam:9s} ({MECHANIC_V3[fam]:6s}) ok {time.time() - t0:6.1f}s  baseline méd. par jeu {meds}  "
              f"demo : {nr} pas, événement {100 * evr:.0f} %, progrès≠0 {100 * pr:.0f} %", flush=True)
    specs = v3_specs(400, 8000)
    mix = {m: sum(MECHANIC_V3[f] == m for f, _, _ in specs) / len(specs) for m in ("avatar", "click", "object")}
    print("mélange v3 (graines 8000-8399) :", {k: round(v, 2) for k, v in mix.items()})
    print(f"total {time.time() - t00:.0f} s")


if __name__ == "__main__":
    main()
