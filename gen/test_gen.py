"""Tests rapides : chaque famille × quelques graines -> construction, solutions rejouées via l'API publique
(step) niveau par niveau, cohérence des observations.   python -m gen.test_gen [n_seeds]"""
import sys
import time

import numpy as np

from .core import EV, N_EVENTS
from .registry import FAMILIES, make_game


def check_game(g):
    obs = g.reset()
    assert obs["frame"].shape == (64, 64) and obs["frame"].dtype == np.int8
    assert obs["frame"].min() >= 0 and obs["frame"].max() <= 15
    for li in range(g.win_levels):
        acts = g.solution_actions(li)
        assert len(acts) == g.baseline_actions[li]
        progs = []
        for a, d in acts:
            obs = g.step(a, d)
            assert obs["events"].shape == (N_EVENTS,) and obs["events"].dtype == np.float32
            progs.append(obs["events"][EV["progress"]])
        assert obs["events"][EV["level_complete"]] == 1
        if g.table(li) is not None:                     # progrès exact : la solution optimale avance toujours
            assert np.mean(progs) > 0, (g.game_id, li, progs)
        exp = "WIN" if li == g.win_levels - 1 else "NOT_FINISHED"
        assert obs["levels_completed"] == li + 1 and obs["state"] == exp, (g.game_id, li, obs["state"], obs["levels_completed"])
    # RESET après WIN -> niveau 1
    obs = g.step(0)
    assert obs["levels_completed"] == 0 and obs["state"] == "NOT_FINISHED"
    # l'oracle (table exacte ou solution stockée) finit le jeu depuis le début
    n = 0
    while obs["state"] != "WIN" and n < 3 * sum(g.baseline_actions) + 50:
        obs = g.step(*g.oracle_action())
        n += 1
    assert obs["state"] == "WIN", (g.game_id, n)
    return True


def check_semantics():
    """énergie -> GAME_OVER ; RESET garde la progression ; double RESET sans effet ; undo ; HUD qui bouge."""
    g = next(make_game("maze", s) for s in range(50) if make_game("maze", s).hud and make_game("maze", s).undo)
    g.reset()
    for a, d in g.solution_actions(0):
        g.step(a, d)
    assert g.levels_completed == 1
    f0 = g.obs()["frame"]
    o = g.step(0)                                   # RESET sans action dans le niveau : sans effet
    assert (o["frame"] == f0).all() and o["levels_completed"] == 1
    a_id = g.op_to_id["up"]
    o1 = g.step(a_id)
    assert (o1["frame"] != f0).sum() >= 1           # au moins le HUD a bougé
    undo = g.op_to_id["undo"]
    while o1["state"] == "NOT_FINISHED":
        o1 = g.step(undo)
    assert o1["state"] == "GAME_OVER" and o1["levels_completed"] == 1
    assert g.step(a_id)["state"] == "GAME_OVER"     # rien ne bouge avant RESET
    o2 = g.step(0)
    assert o2["state"] == "NOT_FINISHED" and o2["levels_completed"] == 1 and (o2["frame"] == f0).all()
    print("sémantique RESET / énergie / undo : ok")


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    tot = time.time()
    check_semantics()
    sig = set()
    looks = sys.argv[2].split(",") if len(sys.argv) > 2 else ["classic", "official"]
    for fam, look in [(f, lk) for lk in looks for f in FAMILIES]:
        t0 = time.time()
        bl = []
        for s in range(n_seeds):
            g = make_game(fam, s, look)
            check_game(g)
            bl.append(g.baseline_actions)
            sig.add(g.reset()["frame"].tobytes())
        print(f"{fam:14s} {look:8s} ok {time.time() - t0:5.2f}s  baselines {bl[0]}  actions {g.available_actions}", flush=True)
    print(f"{len(sig)} premières images distinctes sur {n_seeds * len(FAMILIES) * len(looks)} jeux ; total {time.time() - tot:.1f}s")


if __name__ == "__main__":
    main()
