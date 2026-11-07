"""Planche contact des jeux à RÈGLES (grammaire) : 1re image de N jeux + leur règle résumée.
    python -m gen.sheet_rules [--n 40] [--cols 8] [--out gen/contact_sheet_rules.png]"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from .families_rules import make_rule_game
from .sheet import to_rgb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "contact_sheet_rules.png"))
    a = p.parse_args()
    S, pad, lab = 3 * 64, 6, 36
    rows = (a.n + a.cols - 1) // a.cols
    sheet = Image.new("RGB", (a.cols * (S + pad) + pad, rows * (S + pad + lab) + pad), (40, 40, 40))
    dr = ImageDraw.Draw(sheet)
    for k in range(a.n):
        seed = a.seed0 + k
        g = make_rule_game(seed, "official" if seed % 10 < 7 else "classic")
        g.reset()
        sp = g.spec
        w = sp["win"]
        mods = ",".join(m["name"] for m in sp["modules"])
        x, y = pad + (k % a.cols) * (S + pad), pad + (k // a.cols) * (S + pad + lab)
        dr.text((x, y), f"{g.game_id} {sp['scheme']}{'/' + sp['couple'] if sp['couple'] else ''} a5={sp['a5']}",
                fill=(230, 230, 230))
        dr.text((x, y + 11), f"{sp['arche']} win={w['kind']}{'+exit' if w['and_reach'] is not None else ''}"
                f"{'+commit' if w['commit'] else ''} base={g.baseline_actions[0]}", fill=(200, 200, 120))
        dr.text((x, y + 22), f"+[{mods}]"[:34], fill=(150, 200, 230))
        sheet.paste(to_rgb(g.frame()), (x, y + lab))
    sheet.save(a.out)
    print("écrit", a.out)


if __name__ == "__main__":
    main()
