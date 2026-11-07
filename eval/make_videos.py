"""VIDÉOS des parties enregistrées (run_agent.py --record) : « une image vaut mille mots ».
Chaque image : le jeu (palette officielle, ×4), le clic (croix rouge) ou la flèche jouée, et un bandeau avec le pas, l'action,
les niveaux et ce que le planificateur « pensait » (critique = fin de niveau proche, nouveauté, effet) + une courbe de la critique.
    python make_videos.py <dossier .npz> <dossier sortie> [--games lf52,sc25] [--every 2] [--fps 12]"""
import argparse, glob, os
import numpy as np
from PIL import Image, ImageDraw
from gen.sheet import PALETTE

ARROW = {1: "haut (1)", 2: "bas (2)", 3: "gauche (3)", 4: "droite (4)", 5: "espace (5)", 6: "clic (6)", 7: "annuler (7)"}   # police par défaut : pas d'accents


def frame_img(f, a, x, y, step, lv, crit, nov, eff, hist, S=4):
    W = 64 * S; im = Image.fromarray(np.kron(PALETTE[np.clip(f.astype(np.int64), 0, 15)], np.ones((S, S, 1), np.uint8)))
    out = Image.new("RGB", (W, W + 70), (20, 20, 20)); out.paste(im, (0, 0)); d = ImageDraw.Draw(out)
    if a == 6 and x >= 0:
        cx, cy = x * S + S // 2, y * S + S // 2
        d.line([cx - 9, cy, cx + 9, cy], fill=(255, 40, 40), width=2); d.line([cx, cy - 9, cx, cy + 9], fill=(255, 40, 40), width=2)
    d.text((4, W + 3), f"pas {step}  niveaux {lv}  action {ARROW.get(int(a), a)}", fill=(255, 255, 255))
    d.text((4, W + 17), f"critique {crit:.2f}  nouveaute {nov:.2f}  effet {eff:.2f}", fill=(200, 200, 120))
    if len(hist) > 1:                                         # courbe de la critique (jusqu'ici)
        h = np.nan_to_num(np.array(hist[-W:])); y0 = W + 66
        pts = [(i, y0 - 30 * float(v)) for i, v in enumerate(h)]; d.line(pts, fill=(120, 200, 255), width=1)
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument("src"); p.add_argument("dst"); p.add_argument("--games", default="")
    p.add_argument("--every", type=int, default=2); p.add_argument("--fps", type=int, default=12); a = p.parse_args()
    os.makedirs(a.dst, exist_ok=True)
    for f in sorted(glob.glob(f"{a.src}/*.npz")):
        g = os.path.basename(f)[:4]
        if a.games and g not in a.games.split(","): continue
        z = np.load(f); n = len(z["a"]); hist = []; ims = []
        for t in range(n):
            hist.append(z["crit"][t])
            if t % a.every and z["lv"][t] == z["lv"][max(t - 1, 0)]: continue
            ims.append(frame_img(z["f"][t], z["a"][t], z["x"][t], z["y"][t], t + 1, z["lv"][t], z["crit"][t], z["nov"][t], z["eff"][t], hist))
        ims[0].save(f"{a.dst}/{g}.gif", save_all=True, append_images=ims[1:], duration=int(1000 / a.fps), loop=0, optimize=True)
        print(f"{g} : {n} actions, {len(ims)} images -> {a.dst}/{g}.gif", flush=True)


if __name__ == "__main__":
    main()
