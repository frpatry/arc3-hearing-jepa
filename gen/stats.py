"""Statistiques des shards : fréquence de chaque événement, distribution du PROGRÈS, politiques.
    python -m gen.stats [dossier]"""
import sys
from pathlib import Path

import numpy as np

from .core import EV, EVENT_NAMES


def main():
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "data"
    files = sorted(d.glob("*.npz"))
    E, B, A, S, G, EP, DU, MO = [], [], [], [], [], [], [], []
    names = ("explorer", "aléa", "oracle")
    for f in files:
        z = np.load(f)
        E.append(z["events"]); B.append(z["behav"]); A.append(z["act"]); S.append(z["state"]); G.append(z["game"])
        if "ep" in z:                                    # mode demo_play : épisodes, doublons, behav 0 oracle / 1 explorer
            names = tuple(str(x) for x in z["behav_names"])
            EP.append(z["ep"].astype(np.int64) + 10 ** 7 * len(EP)); DU.append(z["dup"]); MO.append(z["mode"])
    E, B, A, S, G = map(np.concatenate, (E, B, A, S, G))
    size = sum(f.stat().st_size for f in files) / 1e6
    print(f"{len(files)} shards, {len(A)} transitions, {len(np.unique(G))} jeux, {size:.1f} Mo")
    print(f"politique ({'/'.join(names)}) :", np.round(np.bincount(B, minlength=len(names)) / len(B), 3))
    if EP:
        EP, DU, MO = map(np.concatenate, (EP, DU, MO))
        _, ln = np.unique(EP, return_counts=True)
        print(f"épisodes : {len(ln)}, longueur moyenne {ln.mean():.1f} (médiane {np.median(ln):.0f}, "
              f"{100 * (ln >= 16).mean():.0f} % >= 16) ; doublons (préfixe recopié) {100 * DU.mean():.1f} % ; "
              f"pas en démo complète {100 * (MO == 0).mean():.0f} %")
        ev = [EV[k] for k in ("bump", "push", "touch", "click_obj", "collect", "open", "color", "toggle", "teleport",
                              "danger", "level_complete", "game_over")]
        print(f"pas avec un événement {100 * (E[:, ev] > 0).any(1).mean():.1f} % | progrès != 0 "
              f"{100 * (E[:, EV['progress']] != 0).mean():.1f} % | level_complete {100 * E[:, EV['level_complete']].mean():.2f} %")
    print("RESET :", round(float((A == 0).mean()), 3), "| GAME_OVER :", round(float((S == 2).mean()), 4))
    for i, n in enumerate(EVENT_NAMES):
        if n.startswith("pos_") or n == "progress":
            continue
        print(f"  {n:15s} {100 * (E[:, i] > 0).mean():6.2f} %")
    pv = E[:, EV["pos_r1"]] >= 0
    print(f"  avatar présent  {100 * pv.mean():6.2f} %")
    p = E[:, EV["progress"]]
    print("progrès : <0 %.1f %% | =0 %.1f %% | >0 %.1f %% | moyenne %.3f" % (
        100 * (p < 0).mean(), 100 * (p == 0).mean(), 100 * (p > 0).mean(), p.mean()))
    print("progrès histogramme :", np.histogram(p, bins=[-1, -0.99, -0.5, -1e-6, 1e-6, 0.5, 0.99, 1.0])[0])
    for b, name in enumerate(names):
        m = B == b
        if m.any():
            print(f"  progrès moyen {name:8s} {p[m].mean():+.3f}  (level_complete {100 * E[m, EV['level_complete']].mean():.2f} %)")


if __name__ == "__main__":
    main()
