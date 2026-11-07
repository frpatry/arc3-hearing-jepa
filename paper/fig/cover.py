import sys, numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pathlib import Path; H = str(Path(__file__).resolve().parents[2]); sys.path.insert(0, H)
from random_baseline import PALETTE
z = np.load(H + "/paper/fig/cover_frames.npz"); game = sys.argv[1]; F = z[game + "_F"]
W, Hh = 1600, 900; BG = (14, 17, 26); im = Image.new("RGB", (W, Hh), BG); d = ImageDraw.Draw(im)
def font(s, b=False):
    for p in ["/System/Library/Fonts/Supplemental/Arial Bold.ttf" if b else "/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf"]:
        try: return ImageFont.truetype(p, s)
        except Exception: pass
    return ImageFont.load_default()
def rgb(f, s): return Image.fromarray(np.kron(PALETTE[np.clip(f.astype(int), 0, 15)], np.ones((s, s, 1), np.uint8)))
def toc(a, b): return (a != b).reshape(8, 8, 8, 8).any((1, 3))
n = len(F) - 1; k = max(1, n // 2)
# panneau principal : l'écran au pas k, zones qui ont « fait toc » en halo jaune
S = 8; big = rgb(F[k], S).convert("RGBA"); ov = Image.new("RGBA", big.size, (0, 0, 0, 0)); od = ImageDraw.Draw(ov)
for (i, j) in zip(*np.nonzero(toc(F[k - 1], F[k]))):
    x0, y0 = j * 8 * S, i * 8 * S; cx, cy = x0 + 4 * S, y0 + 4 * S
    for r, al in ((54, 70), (40, 110), (26, 160)): od.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 214, 64, al), width=4)
    od.rectangle([x0, y0, x0 + 8 * S - 1, y0 + 8 * S - 1], outline=(255, 214, 64, 255), width=3)
big = Image.alpha_composite(big, ov); X0, Y0 = 80, 250; im.paste(big.convert("RGB"), (X0, Y0))
# frise sonore : une carte 8x8 de tocs par geste, puis DING
d.text((X0, 60), "LISTENING TO THE GAME", font=font(64, True), fill=(240, 240, 245))
d.text((X0, 145), "a hearing JEPA world model for ARC-AGI-3", font=font(34), fill=(170, 178, 196))
x, y, c = 720, 300, 28; d.text((x, y - 50), "what it hears, gesture after gesture", font=font(26), fill=(170, 178, 196))
cells = list(range(1, n + 1))[-8:]; row = 0
for t_i, t in enumerate(cells):
    m = toc(F[t - 1], F[t]); gx = x + (t_i % 4) * (8 * c // 2 + 44); gy = y + (t_i // 4) * (8 * c // 2 + 40)
    cc = c // 2
    for i in range(8):
        for j in range(8):
            col = (255, 214, 64) if m[i, j] else (36, 42, 58)
            d.rectangle([gx + j * cc, gy + i * cc, gx + j * cc + cc - 2, gy + i * cc + cc - 2], fill=col)
    lab, colr = (f"t{t}  DING", (120, 230, 160)) if t == n else (f"t{t}", (120, 128, 146))
    if t == n: d.rectangle([gx - 5, gy - 5, gx + 8 * cc + 3, gy + 8 * cc + 3], outline=colr, width=3)
    d.text((gx, gy + 8 * cc + 6), lab, font=font(20, t == n), fill=colr)
# DING : écran final + cloche
fy = y + 2 * (8 * c // 2 + 44) + 30
d.text((x, fy), "the goal:", font=font(26), fill=(170, 178, 196))
d.text((x + 150, fy - 18), "DING", font=font(64, True), fill=(120, 230, 160))
d.text((x + 400, fy + 4), "level won", font=font(26), fill=(120, 230, 160))
for r in (16, 28, 40): d.arc([x + 340 - r, fy + 20 - r, x + 340 + r, fy + 20 + r], -45, 45, fill=(120, 230, 160), width=4)
d.text((x, fy + 90), "toc = where the screen changed     ding = level won     buzz = game over", font=font(22), fill=(130, 138, 156))
d.text((x, fy + 125), "heard as input  ·  predicted as target  ·  planned toward in imagination", font=font(22), fill=(130, 138, 156))
im.save(H + f"/paper/fig/cover_{game}.png"); print("ok", n, k)
