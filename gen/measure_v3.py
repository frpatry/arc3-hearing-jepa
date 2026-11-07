"""Ressemblance aux 25 jeux officiels : v2 vs v3 vs officiel (métriques de analysis/REPORT.md et compare_stats.py).
    python -m gen.measure_v3 --data v2=gen/data_v2 --data v3=gen/data_v3 [--gen 60] [--frames 30] [--out gen/measure_v3.json]
    --gen N : en plus (ou à la place), N jeux v3 joués en demo_play (graines 20000+, hors de tout jeu de données).
Pour chaque métrique et chaque source :
  - « dans l'IQR » = part des jeux dont la valeur tombe dans l'IQR des 25 officiels (la mesure de measure_v2) ;
  - « méd/off » = médiane des jeux / médiane officielle ;
  - pour le fond noir : part des jeux à fond noir, et rapport min(g/o, o/g).
Valeurs officielles : analysis/stats_per_game.csv (images des enregistrements publics) et baselines des metadata.json."""
import argparse
import glob
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .measure_v2 import ROOT, cs, stats_of

KEYS = ["cell", "obj_size_med", "bg_frac", "bg_black", "tiny_frac", "n_colors", "colors_per_blob", "d_px_med",
        "frac_gt4", "actions_per_level"]
LABEL = dict(cell="taille de cellule (px)", obj_size_med="taille d'objet médiane (px)", bg_frac="part du fond",
             bg_black="jeux à fond noir", tiny_frac="objets ≤ 4 px", n_colors="couleurs par image",
             colors_per_blob="couleurs par blob", d_px_med="px changés par action (méd.)",
             frac_gt4="actions changeant > 4 px", actions_per_level="actions par niveau (baseline méd.)")
REPORT_KEYS = ["cell", "obj_size_med", "bg_frac", "bg_black", "actions_per_level"]


def official():
    rows = [l.strip().split(",") for l in open(ROOT / "analysis" / "stats_per_game.csv")]
    hdr = rows[0]
    off = [r for r in rows[1:] if r[0] == "official"]
    out = {k: np.array([float(r[hdr.index(k)]) for r in off]) for k in KEYS if k in hdr}
    per = []
    for f in sorted(glob.glob(str(ROOT / "data" / "environment_files" / "*" / "*" / "metadata.json"))):
        per.append(float(np.median(json.load(open(f))["baseline_actions"])))
    out["actions_per_level"] = np.array(per)
    return out


def dyn_stats(obs, nxt, act):
    keep = act != 0
    d = (obs[keep] != nxt[keep]).sum((1, 2))
    return dict(d_px_med=float(np.median(d)) if d.size else 0.0, frac_gt4=float((d > 4).mean()) if d.size else 0.0)


def rows_from_data(path, n_frames, rng):
    acc = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(os.path.join(path, "*.npz"))):
        z = np.load(f)
        gm = z["game"]
        for gi in np.unique(gm):
            sel = np.flatnonzero(gm == gi)
            for k in ("obs", "next_obs", "act"):
                acc[int(gi)][k].append(z[k][sel])
    specs = json.load(open(os.path.join(path, "games.json")))
    rows = []
    for gi, d in acc.items():
        obs, nxt, act = (np.concatenate(d[k]) for k in ("obs", "next_obs", "act"))
        idx = rng.choice(len(obs), min(n_frames, len(obs)), replace=False)
        r = stats_of([obs[i] for i in idx])
        r.update(dyn_stats(obs, nxt, act))
        sp = specs[gi]
        r.update(game=sp["game_id"], family=sp["family"], look=sp.get("look", "official"),
                 actions_per_level=float(np.median(sp["baseline_actions"])))
        rows.append(r)
    return rows


def rows_from_gen(n, seed0, n_frames, rng, steps=300):
    from .registry import make_game, v3_specs
    from .rollouts import demo_game
    rows = []
    for fam, seed, look in v3_specs(n, seed0):
        try:
            g = make_game(fam, seed, look)
        except Exception as e:
            print(f"  {fam}-o{seed:04d} ignoré : {e!r}", flush=True)
            continue
        tr, _ = demo_game(g, seed, steps, rng)
        obs = np.array([t["obs"] for t in tr])
        nxt = np.array([t["next_obs"] for t in tr])
        act = np.array([t["act"] for t in tr])
        idx = rng.choice(len(obs), min(n_frames, len(obs)), replace=False)
        r = stats_of([obs[i] for i in idx])
        r.update(dyn_stats(obs, nxt, act))
        r.update(game=g.game_id, family=fam, look=look, actions_per_level=float(np.median(g.baseline_actions)))
        rows.append(r)
        print(f"  {g.game_id:16s} base {g.baseline_actions}", flush=True)
    return rows


def summarize(rows, off):
    res = {}
    for k in KEYS:
        o = off[k]
        v = np.array([r[k] for r in rows])
        if k == "bg_black":
            so, sg = float((o > 0.5).mean()), float((v > 0.5).mean())
            res[k] = dict(official=so, generated=sg, inside=float(min(sg / so, so / sg)) if sg > 0 else 0.0,
                          ratio=float(sg / so))
            continue
        lo, hi = np.percentile(o, 25), np.percentile(o, 75)
        res[k] = dict(iqr=[float(lo), float(hi)], official=float(np.median(o)), generated=float(np.median(v)),
                      inside=float(((v >= lo) & (v <= hi)).mean()),
                      ratio=float(np.median(v) / np.median(o)) if np.median(o) else float("nan"))
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", action="append", default=[], help="label=dossier de shards (avec games.json)")
    p.add_argument("--gen", type=int, default=0)
    p.add_argument("--seed0", type=int, default=20000)
    p.add_argument("--frames", type=int, default=30)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "measure_v3.json"))
    a = p.parse_args()
    off = official()
    rng = np.random.default_rng(0)
    srcs = {}
    for spec in a.data:
        lab, path = spec.split("=", 1)
        t0 = time.time()
        srcs[lab] = rows_from_data(path, a.frames, rng)
        print(f"{lab}: {len(srcs[lab])} jeux ({time.time() - t0:.0f} s)", flush=True)
    if a.gen:
        srcs["v3_gen"] = rows_from_gen(a.gen, a.seed0, a.frames, rng)
    summ = {lab: summarize(rows, off) for lab, rows in srcs.items()}
    labs = list(srcs)
    print("\n| métrique | officiel (méd., IQR) | " + " | ".join(f"{l} méd. | {l} dans l'IQR | {l} méd/off" for l in labs) + " |")
    print("|---|---|" + "---|---|---|" * len(labs))
    for k in KEYS:
        o = summ[labs[0]][k] if labs else None
        if k == "bg_black":
            ostr = f"{100 * o['official']:.0f} % des jeux"
            cells = [f"{100 * summ[l][k]['generated']:.0f} % | {100 * summ[l][k]['inside']:.0f} % (rapport) | "
                     f"{100 * summ[l][k]['ratio']:.0f} %" for l in labs]
        else:
            ostr = f"{o['official']:.3g} [{o['iqr'][0]:.3g}, {o['iqr'][1]:.3g}]"
            cells = [f"{summ[l][k]['generated']:.3g} | {100 * summ[l][k]['inside']:.0f} % | {100 * summ[l][k]['ratio']:.0f} %"
                     for l in labs]
        star = " (REPORT)" if k in REPORT_KEYS else ""
        print(f"| {LABEL[k]}{star} | {ostr} | " + " | ".join(cells) + " |")
    print("\npar famille (méd.) : cellule, objet, fond, noir, actions/niveau")
    for lab, rows in srcs.items():
        fam = defaultdict(list)
        for r in rows:
            fam[r["family"]].append(r)
        for f, rs in sorted(fam.items()):
            print(f"  {lab:7s} {f:15s} n={len(rs):3d}  " + "  ".join(
                f"{np.median([r[k] for r in rs]):6.2f}" for k in REPORT_KEYS))
    Path(a.out).write_text(json.dumps(dict(summary=summ, rows={l: [{k: v for k, v in r.items()} for r in rs]
                                                               for l, rs in srcs.items()}), indent=1))
    print("écrit", a.out)


if __name__ == "__main__":
    main()
