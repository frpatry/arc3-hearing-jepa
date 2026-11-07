"""Tests de la grammaire de règles : tire N specs (graines 0..N-1, attempt 0), construit les jeux, rejoue la solution
BFS de CHAQUE niveau via step() (doit gagner), vérifie l'oracle, RESET / énergie, et rapporte : taux d'acceptation,
raisons de rejet, actions de baseline moyennes par niveau, distribution des règles, diversité, temps.
    python -m gen.test_grammar [--n 300] [--workers 6]"""
import argparse
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np

from . import families_rules as fr
from .core import EV, Unsolvable
from .grammar import signature
from .test_gen import check_game


def _one(seed):
    before = Counter(fr.STATS)
    look = "official" if seed % 10 < 7 else "classic"
    t0 = time.time()
    try:
        g = fr.RuleGame(seed, look, attempt=0)
    except Unsolvable:
        sp = fr.sample_rules(seed, 0)
        return dict(seed=seed, ok=False, t=time.time() - t0, stats=Counter(fr.STATS) - before,
                    scheme=sp["scheme"], arche=sp["arche"])
    tb = time.time() - t0
    check_game(g)                                    # solutions rejouées par step(), RESET après WIN, oracle
    sp = g.spec
    return dict(seed=seed, ok=True, t=time.time() - t0, t_build=tb, stats=Counter(fr.STATS) - before,
                base=g.baseline_actions, scheme=sp["scheme"], win=sp["win"]["kind"], arche=sp["arche"],
                a5=sp["a5"], sig=repr(signature(sp)), mods=[m["name"] for m in sp["modules"]],
                trig=[ru["trigger"] for ru in sp["rules"]], eff=[e[0] for ru in sp["rules"] for e in ru["effects"]],
                motions=[t["motion"] for t in sp["types"] if t["motion"]], ntypes=len(sp["types"]),
                and_reach=sp["win"]["and_reach"] is not None, commit=sp["win"]["commit"], look=look,
                legend=bool(g.levels[0]["lvl"]["ph"]))


def check_semantics():
    """énergie -> GAME_OVER ; RESET garde la progression ; RESET sans action sans effet ; GAME_OVER figé."""
    for s in range(200):
        try:
            g = fr.RuleGame(s, "classic", attempt=0)
        except Unsolvable:
            continue
        if g.hud and g.spec["scheme"] in ("avatar", "multi") and not g.spec["a5"]:
            break
    g.reset()
    for a, d in g.solution_actions(0):
        g.step(a, d)
    assert g.levels_completed == 1
    f0 = g.obs()["frame"]
    o = g.step(0)
    assert (o["frame"] == f0).all() and o["levels_completed"] == 1 and o["events"][EV["noop"]] == 1
    # une action répétée (qui ne gagne pas) jusqu'à épuisement de l'énergie
    acts = [i for i, op in g.action_map.items() if op in ("up", "down", "left", "right")]
    for a in acts:
        g.step(0)
        o = None
        for _ in range(10000):
            o = g.step(a)
            if o["state"] != "NOT_FINISHED" or o["levels_completed"] != 1:
                break
        if o["state"] == "GAME_OVER" and o["levels_completed"] == 1:
            break
    assert o["state"] == "GAME_OVER" and o["levels_completed"] == 1, (g.game_id, o["state"])
    assert o["events"][EV["game_over"]] == 1
    assert g.step(acts[0])["state"] == "GAME_OVER"            # rien ne bouge avant RESET
    o2 = g.step(0)
    assert o2["state"] == "NOT_FINISHED" and o2["levels_completed"] == 1 and (o2["frame"] == f0).all()
    print(f"sémantique RESET / énergie ({g.game_id}) : ok")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    t0 = time.time()
    check_semantics()
    with Pool(a.workers) as pool:
        res = pool.map(_one, range(a.n), chunksize=1)
    wall = time.time() - t0
    ok = [r for r in res if r["ok"]]
    st = Counter()
    for r in res:
        st.update(r["stats"])
    print(f"\n{a.n} specs tirées -> {len(ok)} acceptées ({100 * len(ok) / a.n:.0f} %), toutes rejouées OK ; "
          f"{wall:.0f} s mur ({a.workers} proc.), {np.mean([r['t'] for r in res]):.2f} s CPU / spec "
          f"(construction {np.mean([r['t_build'] for r in ok]):.2f} s / jeu accepté)")
    nl = sum(len(r["base"]) for r in ok)
    lv_ok = st["lvl_ok"]
    lv_tries = sum(v for k, v in st.items() if k.startswith("lvl_"))
    print(f"niveaux : {nl} ({nl / len(ok):.1f} par jeu, {st['game_truncated']} jeux tronqués à < 8) ; "
          f"essais de niveau acceptés {lv_ok}/{lv_tries} ({100 * lv_ok / max(1, lv_tries):.0f} %)")
    print("rejets de niveau :", {k[4:]: v for k, v in st.most_common() if k.startswith("lvl_") and k != "lvl_ok"})
    print("échecs de mise en place :", {k[7:]: v for k, v in st.most_common() if k.startswith("layout_")})
    B = [r["base"] for r in ok]
    mx = max(len(b) for b in B)
    per = [np.mean([b[i] for b in B if len(b) > i]) for i in range(mx)]
    print("baseline moyenne par niveau :", " ".join(f"{x:.1f}" for x in per),
          f"| médiane globale {np.median([x for b in B for x in b]):.0f}, max {max(x for b in B for x in b)}")
    for name in ("scheme", "arche"):
        tot = Counter(r[name] for r in res)
        acc = Counter(r[name] for r in ok)
        print(f"acceptés / tirés par {name} :", {k: f"{acc[k]}/{v}" for k, v in tot.most_common()})
    for name in ("win", "a5", "look"):
        print(f"{name:7s}", dict(Counter(str(r[name]) for r in ok).most_common()))
    print("modules   ", dict(Counter(m for r in ok for m in r["mods"]).most_common()))
    print("déclench. ", dict(Counter(x for r in ok for x in r["trig"]).most_common()))
    print("effets    ", dict(Counter(x for r in ok for x in r["eff"]).most_common()))
    print("dynamique ", dict(Counter(x for r in ok for x in r["motions"]).most_common()))
    print(f"+sortie {sum(r['and_reach'] for r in ok)}, commit {sum(r['commit'] for r in ok)}, "
          f"légende/panneau au niveau 1 {sum(r['legend'] for r in ok)}, types/jeu {np.mean([r['ntypes'] for r in ok]):.1f}")
    print(f"signatures distinctes (contrôle, interactions, victoire) : {len({r['sig'] for r in ok})} / {len(ok)} jeux")


if __name__ == "__main__":
    main()
