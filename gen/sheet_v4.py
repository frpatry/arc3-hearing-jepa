"""Planche contact v4 : images de jeux v4 (niveau 0-2, un tiers de la solution joué) à côté d'images officielles.
    python -m gen.sheet_v4 [--n 32] [--seed0 10000] [--families kick4,...] [--out gen/contact_sheet_v4.png]
Images officielles : wm/data/public/*.npz (comparaison d'aspect seulement, rien n'en est copié)."""
import argparse
import glob
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .registry import make_game, v4_specs
from .sheet import to_rgb

ROOT = Path(__file__).resolve().parent.parent


def tile_of(fam, seed, frac=1 / 3):
    g = make_game(fam, seed)
    g.reset()
    lv = int(np.random.default_rng(seed).integers(0, min(3, g.win_levels)))
    for li in range(lv):
        for act, d in g.solution_actions(li):
            g.step(act, d)
    sol = g.solution_actions(lv)
    for act, d in sol[:int(len(sol) * frac)]:
        g.step(act, d)
    return f"{g.game_id} L{lv} b{g.baseline_actions[lv]}", g.frame()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--seed0", type=int, default=10000)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--families", default="")
    p.add_argument("--n_off", type=int, default=16)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "contact_sheet_v4.png"))
    a = p.parse_args()
    fams = [f for f in a.families.split(",") if f]
    specs = [(fams[k % len(fams)], a.seed0 + k, "official") for k in range(a.n)] if fams else v4_specs(a.n, a.seed0)
    tiles = []
    for fam, seed, _ in specs:
        try:
            tiles.append(tile_of(fam, seed))
        except Exception as e:                                   # jeu impossible : noté sur la planche
            print(f"{fam} {seed}: {e!r}")
    rng = np.random.default_rng(0)
    off = []
    for f in sorted(glob.glob(str(ROOT / "wm" / "data" / "public" / "*.npz"))):
        z = np.load(f)
        obs = z["obs"]
        off.append((f"OFFICIEL {Path(f).stem}", obs[int(rng.integers(len(obs) // 4, 3 * len(obs) // 4))]))
    off = [off[k] for k in rng.permutation(len(off))[:a.n_off]]
    S, pad, lab = 3 * 64, 6, 14
    rows_g = (len(tiles) + a.cols - 1) // a.cols
    rows_off = (len(off) + a.cols - 1) // a.cols
    H = (rows_g + rows_off) * (S + pad + lab) + pad + 24
    sheet = Image.new("RGB", (a.cols * (S + pad) + pad, H), (40, 40, 40))
    dr = ImageDraw.Draw(sheet)
    y0 = pad
    for k, (name, fr) in enumerate(tiles):
        x, y = pad + (k % a.cols) * (S + pad), y0 + (k // a.cols) * (S + pad + lab)
        dr.text((x, y), name, fill=(230, 230, 230))
        sheet.paste(to_rgb(fr), (x, y + lab))
    y0 += rows_g * (S + pad + lab) + 8
    dr.text((pad, y0), "jeux officiels (enregistrements publics), pour comparaison :", fill=(255, 200, 80))
    y0 += 16
    for k, (name, fr) in enumerate(off):
        x, y = pad + (k % a.cols) * (S + pad), y0 + (k // a.cols) * (S + pad + lab)
        dr.text((x, y), name, fill=(255, 200, 80))
        sheet.paste(to_rgb(fr), (x, y + lab))
    sheet.save(a.out)
    print("écrit", a.out)


if __name__ == "__main__":
    main()
