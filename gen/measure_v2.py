"""Mesure de ressemblance « façon officielle » du mélange v2 (même métriques que analysis/compare_stats.py).
    python -m gen.measure_v2 [--n 80] [--steps 120] [--frames 30] [--data gen/data_v2]
Pour chaque jeu généré : taille de cellule (médiane), taille d'objet médiane, part du fond, fond noir (oui/non),
actions par niveau (médiane des baselines). On compte la part des jeux qui tombent dans l'IQR des 25 jeux officiels
(lu dans analysis/stats_per_game.csv et les metadata.json officiels). Avec --data : images prises dans les shards."""
import argparse
import glob
import importlib.util
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from .registry import make_game, mixture_specs
from .rollouts import RandomAgent

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("compare_stats", ROOT / "analysis" / "compare_stats.py")
cs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cs)

KEYS = ["cell", "obj_size_med", "bg_frac", "bg_black", "actions_per_level"]


def official_ranges():
    rows = [l.strip().split(",") for l in open(ROOT / "analysis" / "stats_per_game.csv")]
    hdr = rows[0]
    off = [r for r in rows[1:] if r[0] == "official"]
    out = {}
    for k in ("cell", "obj_size_med", "bg_frac", "bg_black"):
        x = np.array([float(r[hdr.index(k)]) for r in off])
        out[k] = x
    per = []
    for f in glob.glob(str(ROOT / "data" / "environment_files" / "*" / "*" / "metadata.json")):
        per.append(float(np.median(json.load(open(f))["baseline_actions"])))
    out["actions_per_level"] = np.array(per)
    return out


def game_frames(g, steps, n_frames, rng):
    """politique mixte (aléatoire + oracle) pour voir plusieurs niveaux ; renvoie un échantillon d'images."""
    rnd = RandomAgent(int(rng.integers(1 << 30)))
    obs = g.reset()
    frames = [obs["frame"]]
    for _ in range(steps):
        if obs["state"] == "WIN":
            obs = g.step(0)
        a, d = g.oracle_action() if rng.random() < 0.5 else rnd.act(obs["frame"], obs["state"], 0, obs["available_actions"])
        obs = g.step(a, d)
        frames.append(obs["frame"])
    idx = rng.choice(len(frames), min(n_frames, len(frames)), replace=False)
    return [frames[i] for i in idx]


def stats_of(frames):
    fs = [cs.frame_stats(f) for f in frames]
    sizes = np.concatenate([f["obj_sizes"] for f in fs])
    return dict(cell=float(np.median([f["cell"] for f in fs])),
                obj_size_med=float(np.median(sizes)) if sizes.size else 0.0,
                bg_frac=float(np.mean([f["bg_frac"] for f in fs])),
                bg_black=float(np.mean([f["bg_color"] == 5 for f in fs])),
                tiny_frac=float(np.mean([f["tiny_frac"] for f in fs])),
                n_colors=float(np.mean([f["n_colors"] for f in fs])),
                colors_per_blob=float(np.mean([f["colors_per_blob"] for f in fs])))


def report(rows, off, label):
    print(f"\n== {label} : {len(rows)} jeux ==")
    res = {}
    for k in KEYS:
        o = off[k]
        lo, hi = np.percentile(o, 25), np.percentile(o, 75)
        v = np.array([r[k] for r in rows])
        if k == "bg_black":
            share_o = float((o > 0.5).mean())
            share_g = float((v > 0.5).mean())
            res[k] = dict(official_black_share=share_o, generated_black_share=share_g)
            print(f"  {k:18s} jeux à fond noir : officiels {100 * share_o:.0f} % | générés {100 * share_g:.0f} %")
            continue
        inside = float(((v >= lo) & (v <= hi)).mean())
        res[k] = dict(iqr=[float(lo), float(hi)], official_median=float(np.median(o)), generated_median=float(np.median(v)),
                      inside=inside)
        print(f"  {k:18s} IQR officiel [{lo:.3g}, {hi:.3g}] méd {np.median(o):.3g} | généré méd {np.median(v):.3g} "
              f"| dans l'IQR : {100 * inside:.0f} %")
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--frames", type=int, default=30)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--data", default="")
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "measure_v2.json"))
    a = p.parse_args()
    off = official_ranges()
    rng = np.random.default_rng(0)
    rows = []
    if a.data:
        acc = defaultdict(list)
        ids = None
        for f in sorted(glob.glob(os.path.join(a.data, "shard_*.npz"))):
            z = np.load(f)
            ids = z["game_ids"]
            for gi in np.unique(z["game"]):
                sel = np.flatnonzero(z["game"] == gi)
                acc[int(gi)].append(z["obs"][rng.choice(sel, min(a.frames, len(sel)), replace=False)])
        specs = json.load(open(os.path.join(a.data, "games.json")))
        for gi, fr in acc.items():
            frames = list(np.concatenate(fr))[:a.frames]
            sp = specs[gi]
            r = stats_of(frames)
            r.update(game=sp["game_id"], family=sp["family"], look=sp["look"],
                     actions_per_level=float(np.median(sp["baseline_actions"])))
            rows.append(r)
    else:
        for fam, seed, look in mixture_specs(a.n, a.seed0):
            g = make_game(fam, seed, look)
            r = stats_of(game_frames(g, a.steps, a.frames, rng))
            r.update(game=g.game_id, family=fam, look=look, actions_per_level=float(np.median(g.baseline_actions)))
            rows.append(r)
    out = dict(all=report(rows, off, "mélange complet"),
               official_look=report([r for r in rows if r["look"] == "official"], off, "look officiel seulement"),
               classic_look=report([r for r in rows if r["look"] == "classic"], off, "look classique seulement"))
    fam = defaultdict(list)
    for r in rows:
        fam[(r["family"], r["look"])].append(r)
    print("\npar famille (look) : cellule, objet, fond, noir, actions/niveau")
    for (f, lk), rs in sorted(fam.items()):
        print(f"  {f:15s} {lk:8s} n={len(rs):2d}  " + "  ".join(f"{np.mean([r[k] for r in rs]):6.2f}" for k in KEYS))
    out["rows"] = rows
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
