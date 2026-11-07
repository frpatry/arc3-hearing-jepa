"""Planche contact v3 : images de jeux v3 (niveau 0 et un niveau plus loin) à côté d'images officielles.
    python -m gen.sheet_v3 [--n 24] [--seed0 8000] [--out gen/contact_sheet_v3.png]
Les images officielles viennent de wm/data/public/*.npz (une image prise au milieu de l'enregistrement de chaque jeu) :
seulement pour comparer l'aspect, rien n'en est copié dans le générateur."""
import argparse
import glob
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .registry import make_game, v3_specs
from .sheet import to_rgb

ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--seed0", type=int, default=8000)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "contact_sheet_v3.png"))
    a = p.parse_args()
    tiles = []
    for fam, seed, look in v3_specs(a.n, a.seed0):
        g = make_game(fam, seed, look)
        g.reset()
        lv = int(np.random.default_rng(seed).integers(0, min(3, g.win_levels)))
        for li in range(lv):
            for act, d in g.solution_actions(li):
                g.step(act, d)
        sol = g.solution_actions(lv)
        for act, d in sol[:len(sol) // 3]:                    # un peu de jeu dans le niveau
            g.step(act, d)
        tiles.append((f"{g.game_id} L{lv} b{g.baseline_actions[lv]}", g.frame()))
    rng = np.random.default_rng(0)
    off = []
    for f in sorted(glob.glob(str(ROOT / "wm" / "data" / "public" / "*.npz"))):
        z = np.load(f)
        obs = z["obs"]
        off.append((f"OFFICIEL {Path(f).stem}", obs[int(rng.integers(len(obs) // 4, 3 * len(obs) // 4))]))
    off = off[:16]
    S, pad, lab = 3 * 64, 6, 14
    rows_v3 = (len(tiles) + a.cols - 1) // a.cols
    rows_off = (len(off) + a.cols - 1) // a.cols
    H = (rows_v3 + rows_off) * (S + pad + lab) + pad + 24
    sheet = Image.new("RGB", (a.cols * (S + pad) + pad, H), (40, 40, 40))
    dr = ImageDraw.Draw(sheet)
    y0 = pad
    for k, (name, fr) in enumerate(tiles):
        x, y = pad + (k % a.cols) * (S + pad), y0 + (k // a.cols) * (S + pad + lab)
        dr.text((x, y), name, fill=(230, 230, 230))
        sheet.paste(to_rgb(fr), (x, y + lab))
    y0 += rows_v3 * (S + pad + lab) + 8
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
