"""Planche contact : 1re image de N jeux générés (palette officielle ARC-AGI-3).
    python -m gen.sheet [--n 32] [--out gen/contact_sheet.png] [--level 0]"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .registry import game_specs, make_game, mixture_specs

COLORS = ['#FFFFFF', '#CCCCCC', '#999999', '#666666', '#333333', '#000000', '#E53AA3', '#FF7BCC', '#F93C31',
          '#1E93FF', '#88D8F1', '#FFDC00', '#FF851B', '#921231', '#4FCC30', '#A356D6']
PALETTE = np.array([[int(c[k:k + 2], 16) for k in (1, 3, 5)] for c in COLORS], np.uint8)


def to_rgb(frame, scale=3):
    rgb = PALETTE[np.clip(frame.astype(np.int64), 0, 15)]
    return Image.fromarray(rgb, "RGB").resize((64 * scale, 64 * scale), Image.NEAREST)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--out", default="")
    p.add_argument("--mix", action="store_true", help="mélange v2 (sample_spec) au lieu des familles v1")
    a = p.parse_args()
    a.out = a.out or str(Path(__file__).resolve().parent / ("contact_sheet_v2.png" if a.mix else "contact_sheet.png"))
    specs = mixture_specs(a.n) if a.mix else [(f, s, "classic") for f, s in game_specs(a.n)]
    S, pad, lab = 3 * 64, 6, 14
    rows = (a.n + a.cols - 1) // a.cols
    sheet = Image.new("RGB", (a.cols * (S + pad) + pad, rows * (S + pad + lab) + pad), (40, 40, 40))
    dr = ImageDraw.Draw(sheet)
    for k, (fam, seed, look) in enumerate(specs):
        g = make_game(fam, seed, look)
        g.reset()
        for li in range(min(a.level, g.win_levels - 1)):
            for act, d in g.solution_actions(li):
                g.step(act, d)
        x, y = pad + (k % a.cols) * (S + pad), pad + (k // a.cols) * (S + pad + lab)
        dr.text((x, y), f"{g.game_id}  {g.available_actions}", fill=(230, 230, 230))
        sheet.paste(to_rgb(g.frame()), (x, y + lab))
    sheet.save(a.out)
    print("écrit", a.out)


if __name__ == "__main__":
    main()
