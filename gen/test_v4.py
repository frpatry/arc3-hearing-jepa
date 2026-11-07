"""Tests des familles v4 :   python -m gen.test_v4 [--n 3] [--seed0 30000] [--families kick4,pump4]
  - construction (temps), solutions rejouées par step() niveau par niveau, l'oracle finit le jeu (test_gen.check_game) ;
  - les 23 événements privilégiés (toucher / corps / son, dont son spatial via pos_*, progrès) sont émis ;
  - curriculum : chaque niveau 1..4 ajoute ou change une règle (lvl["mods"]) ;
  - demo_play : seulement oracle (0) et Explorer (1), aucune action aléatoire ;
  - comportement : part de pas SANS effet visible (hors barre) et de pas changeant > 4 px, par politique."""
import argparse
import time

import numpy as np

from .core import EV, EVENT_NAMES, N_EVENTS
from .fastcomp import components
from .families_v4 import FAMILY_CLASSES_V4, MECHANIC_V4
from .registry import make_game, v4_specs
from .rollouts import demo_game
from .test_gen import check_game


def inner_diff(o, n):
    d = o != n
    d[0, :] = d[-1, :] = False                      # barre de pas d'1 px sur un bord
    d[:, 0] = d[:, -1] = False
    return int(d.sum())


def check_rules_change(g):
    rl = [tuple(L["lvl"].get("mods", ())) if isinstance(L["lvl"], dict) else () for L in g.levels]
    for i in range(1, min(len(rl), 5)):
        assert rl[i] != rl[i - 1], (g.game_id, i, rl[:i + 1])


def check_demo(g, seed, steps=240):
    rng = np.random.default_rng(seed)
    rows, _ = demo_game(g, seed, steps, rng)
    assert rows, g.game_id
    beh = np.array([r["behav"] for r in rows])
    assert set(beh.tolist()) <= {0, 1}, set(beh.tolist())
    ev = np.array([r["events"] for r in rows])
    assert ev.shape[1] == N_EVENTS == 23 and len(EVENT_NAMES) == 23
    prog = ev[:, EV["progress"]]
    assert np.all((prog >= -1) & (prog <= 1))
    act = np.array([r["act"] for r in rows])
    d = np.array([inner_diff(r["obs"], r["next_obs"]) for r in rows])
    keep = act != 0
    snd = [EV[k] for k in ("bump", "push", "touch", "click_obj", "collect", "open", "color", "toggle", "teleport",
                           "danger", "level_complete", "game_over")]
    out = dict(n=len(rows), ev=float((ev[:, snd] > 0).any(1).mean()), prog=float((prog != 0).mean()),
               lvl=int(max(r["level"] for r in rows)))
    for name, b in (("oracle", 0), ("explorer", 1)):
        m = keep & (beh == b)
        out[f"noop_{name}"] = float((d[m] == 0).mean()) if m.any() else float("nan")
    out["gt4"] = float((d[keep] > 4).mean())
    out["small"] = float(((d[keep] > 0) & (d[keep] <= 4)).mean())
    # aspect (officiel, médianes des 25 jeux : fond 0.74, ~28 blobs hors fond, blob médian 9 px, 7 couleurs)
    bgs, nbl, med = [], [], []
    for r in rows[::4]:
        o = r["obs"]
        _, cnt = np.unique(o, return_counts=True)
        bgs.append(cnt.max() / 4096)
        cs = components(o.astype(np.int64), bg_frac=0.25, max_n=999)
        nbl.append(len(cs))
        med.append(float(np.median([c[1] for c in cs])) if cs else 0.0)
    out["bg"], out["blobs"], out["blob_med"] = float(np.median(bgs)), float(np.median(nbl)), float(np.median(med))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--seed0", type=int, default=30000)
    p.add_argument("--families", default="")
    p.add_argument("--no_demo", action="store_true")
    a = p.parse_args()
    fams = [f for f in a.families.split(",") if f] or list(FAMILY_CLASSES_V4)
    t00 = time.time()
    for fam in fams:
        t0 = time.time()
        meds, tb, outs = [], [], []
        for s in range(a.seed0, a.seed0 + a.n):
            t1 = time.time()
            g = make_game(fam, s)
            tb.append(time.time() - t1)
            check_game(g)
            check_rules_change(g)
            meds.append(float(np.median(g.baseline_actions)))
            if not a.no_demo:
                outs.append(check_demo(make_game(fam, s), s))
        msg = (f"{fam:10s} ({MECHANIC_V4[fam]:6s}) ok {time.time() - t0:6.1f}s  construction max {max(tb):.1f}s  "
               f"baseline méd. {meds}  niveaux {g.win_levels}")
        if outs:
            mean = {k: float(np.nanmean([o[k] for o in outs])) for k in outs[0]}
            msg += (f"\n           demo : {mean['n']:.0f} pas, événement {100 * mean['ev']:.0f} %, progrès≠0 "
                    f"{100 * mean['prog']:.0f} %, sans effet oracle {100 * mean['noop_oracle']:.0f} % / explorer "
                    f"{100 * mean['noop_explorer']:.0f} %, >4 px {100 * mean['gt4']:.0f} %, 1-4 px {100 * mean['small']:.0f} %"
                    f"\n           aspect : fond {mean['bg']:.2f} (off. 0.74), blobs {mean['blobs']:.0f} (off. ~28), "
                    f"blob médian {mean['blob_med']:.1f} px (off. 9)")
        print(msg, flush=True)
    if not a.families:
        specs = v4_specs(400, 10000)
        mix = {m: sum(MECHANIC_V4[f] == m for f, _, _ in specs) / len(specs) for m in ("avatar", "click", "object")}
        print("mélange v4 (graines 10000-10399) :", {k: round(v, 2) for k, v in mix.items()})
    print(f"total {time.time() - t00:.0f} s")


if __name__ == "__main__":
    main()
