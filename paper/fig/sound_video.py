"""VIDÉO SONORE pour le Writeup (7 nov.) : la solution oracle de ls20 (niveaux 1-2) rejouée, avec les sons fabriqués que le modèle
entend — toc (zones 8×8 qui changent ; panoramique = position x, hauteur = position y, volume = nb de zones), silence si rien ne
bouge, ding (cloche) au niveau gagné, buzz à la partie perdue.  python paper/fig/sound_video.py ls20 2 paper/fig/ls20_sound.mp4
    python paper/fig/sound_video.py vc33 paper/fig/vc33_H12_c3_focus_s2.gif paper/fig/vc33_model_sound.mp4 8,32   (vraie partie du modèle)"""
import json, subprocess, sys, wave
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
H = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(H), str(H / "data" / "oracle_public")]
import oracle_lib as L
from arcengine import GameAction
from random_baseline import PALETTE

game, src, out = sys.argv[1], sys.argv[2], sys.argv[3]
FPS, SR = 30, 44100
PAL = PALETTE.astype(np.int32)
EV = []                                               # (image avant, geste, image après, ding, buzz, niveau)
if src.endswith(".gif"):                              # VRAIE PARTIE filmée par record_game.py (image AVANT chaque geste, croix rouge = clic, puis image de victoire)
    STEP, MODEL, Q = 0.55, True, 7; lv_at = [int(v) for v in sys.argv[4].split(",")]
    from PIL import Image as _I
    g = _I.open(src); F, CL = [], []
    for i in range(g.n_frames):
        g.seek(i); rgb = np.asarray(g.convert("RGB"), np.int32)[:256, :256]
        F.append(np.abs(rgb[::4, ::4, None, :] - PAL[None, None]).sum(-1).argmin(-1).astype(np.int8))   # coin (0,0) de chaque bloc 4×4 : jamais sous la croix
        red = (rgb[..., 0] > 230) & (rgb[..., 1] < 70) & (rgb[..., 2] < 70)
        CL.append((int(np.nonzero(red)[1].mean()) // 4, int(np.nonzero(red)[0].mean()) // 4) if red.sum() > 8 else None)
    i = 0
    for li, n in enumerate(lv_at):
        for k in range(n):
            c = CL[i]; EV.append((F[i], (6, c[0], c[1]) if c else (0, -1, -1), F[i + 1], k == n - 1, False, li)); i += 1
        i += 1                                        # image de victoire = image avant le 1er geste du niveau suivant
else:                                                 # solution ORACLE rejouée : src = nombre de niveaux
    STEP, MODEL, Q = 0.40, False, 2
    sol = json.load(open(H / "data" / "oracle_public" / f"{game}_solution.json"))["solution"]
    env = L.make_env(game); fr = np.asarray(env.step(GameAction.RESET).frame[-1], np.int8); g = L.clone(env._game)
    for li in range(int(src)):
        for (x, cx, cy) in sol[li]:
            s0 = g._score; nx = L.frame_of(L.gstep(g, x, cx, cy))
            EV.append((fr, (x, cx, cy), nx, g._score > s0 or g._state.name == "WIN", g._state.name == "GAME_OVER", li)); fr = nx

def toc(a, b): return (a != b).reshape(8, 8, 8, 8).any((1, 3))
def font(s, b=False):
    try: return ImageFont.truetype(f"/System/Library/Fonts/Supplemental/Arial{' Bold' if b else ''}.ttf", s)
    except Exception: return ImageFont.load_default()
ARROW = {1: "↑", 2: "↓", 3: "←", 4: "→", 5: "space", 6: "click", 7: "undo"}

# ---------- son ----------
T = 2.0 + len(EV) * STEP + 2.5; audio = np.zeros((int(T * SR), 2), np.float32)
def add(t0, mono, pan=0.0):
    i = int(t0 * SR); n = min(len(mono), len(audio) - i); l, r = np.sqrt((1 - pan) / 2), np.sqrt((1 + pan) / 2)
    audio[i:i + n, 0] += mono[:n] * l; audio[i:i + n, 1] += mono[:n] * r
def tone(f, d, decay, parts=((1, 1.0),)):
    t = np.arange(int(d * SR)) / SR; return sum(a * np.sin(2 * np.pi * f * k * t) for k, a in parts) * np.exp(-t / decay)
for k, (a, _, b, ding, buzz, li) in enumerate(EV):
    t0 = 2.0 + k * STEP + Q / FPS; m = toc(a, b)
    if ding:
        for j, f in enumerate((1318.5, 1760.0, 2637.0)): add(t0 + 0.12 * j, 0.35 * tone(f, 1.6, 0.45, ((1, 1), (2.76, 0.3), (5.4, 0.12))))
    elif buzz:
        t = np.arange(int(0.5 * SR)) / SR; add(t0, 0.25 * np.sign(np.sin(2 * np.pi * 110 * t)) * np.exp(-t / 0.3))
    elif m.any():
        ys, xs = np.nonzero(m); pan = (xs.mean() - 3.5) / 3.5 * 0.9; f = 300 * 2 ** ((7 - ys.mean()) / 7 * 1.5)
        vol = 0.18 + 0.12 * np.log1p(m.sum()) / np.log1p(64)
        t = np.arange(int(0.12 * SR)) / SR; click = np.random.default_rng(k).normal(0, 1, len(t)) * np.exp(-t / 0.004) * 0.3
        add(t0, vol * (tone(f, 0.12, 0.03) + click), pan)
audio /= max(1.0, np.abs(audio).max() / 0.9)
wav = out[:-4] + ".wav"
with wave.open(wav, "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((audio * 32767).astype(np.int16).tobytes())

# ---------- image ----------
W, Hh, S = 1280, 720, 9; BG = (14, 17, 26); GX, GY = 60, 72
def scene(f, m=None, ding=0.0, act=None, lvl=0, k=None):
    im = Image.new("RGB", (W, Hh), BG); d = ImageDraw.Draw(im)
    gm = Image.fromarray(np.kron(PALETTE[np.clip(f.astype(int), 0, 15)], np.ones((S, S, 1), np.uint8))).convert("RGBA")
    if m is not None:
        ov = Image.new("RGBA", gm.size, (0, 0, 0, 0)); od = ImageDraw.Draw(ov)
        for i, j in zip(*np.nonzero(m)): od.rectangle([j * 8 * S, i * 8 * S, (j + 1) * 8 * S - 1, (i + 1) * 8 * S - 1], outline=(255, 214, 64, 255), width=3)
        gm = Image.alpha_composite(gm, ov)
    if act is not None and act[0] == 6 and act[1] >= 0 and m is None:
        cd = ImageDraw.Draw(gm); cx, cy = act[1] * S + S // 2, act[2] * S + S // 2
        for w_, c_ in ((7, (0, 0, 0, 255)), (3, (255, 60, 60, 255))): cd.line([cx - 16, cy, cx + 16, cy], fill=c_, width=w_); cd.line([cx, cy - 16, cx, cy + 16], fill=c_, width=w_)
    im.paste(gm.convert("RGB"), (GX, GY))
    d.text((GX, 22), f"{game} · level {lvl + 1}" + ("  ·  played by the model" if MODEL else ""), font=font(30, True), fill=(235, 235, 240))
    px = GX + 64 * S + 60; d.text((px, 80), "what the model hears", font=font(30, True), fill=(235, 235, 240))
    c = 34
    for i in range(8):
        for j in range(8):
            on = m is not None and m[i, j]
            d.rectangle([px + j * c, 140 + i * c, px + j * c + c - 4, 140 + i * c + c - 4], fill=(255, 214, 64) if on else (36, 42, 58))
    lab = "DING  —  level won" if ding > 0 else ("toc" if m is not None and m.any() else ("silence" if m is not None else ""))
    col = (120, 230, 160) if ding > 0 else (255, 214, 64) if lab == "toc" else (130, 138, 156)
    d.text((px, 140 + 8 * c + 20), lab, font=font(44 if ding > 0 else 36, True), fill=col)
    if act is not None: d.text((px, 140 + 8 * c + 80), f"gesture {k}: {ARROW.get(act[0], act[0])}", font=font(26), fill=(170, 178, 196))
    for n, s in enumerate(["toc = where the screen changed", "   left/right = stereo, higher on screen = higher pitch", "ding = level won", ("gestures: the submitted model (recorded 27 Oct)" if MODEL else "gestures: oracle replay · sounds: computed from frames")]):
        d.text((px, Hh - 150 + 28 * n), s, font=font(20), fill=(130, 138, 156))
    if ding > 0:
        a = int(255 * min(1, ding)); ov = Image.new("RGBA", (W, Hh), (0, 0, 0, 0)); ImageDraw.Draw(ov).rectangle([GX - 6, GY - 6, GX + 64 * S + 5, GY + 64 * S + 5], outline=(120, 230, 160, a), width=8)
        im = Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")
    return im
def card(lines):
    im = Image.new("RGB", (W, Hh), BG); d = ImageDraw.Draw(im)
    for n, (txt, sz, col) in enumerate(lines): d.text((90, 250 + n * 70), txt, font=font(sz, n == 0), fill=col)
    return im
ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{Hh}", "-r", str(FPS), "-i", "-", "-i", wav,
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-c:a", "aac", "-b:a", "160k", "-shortest", out], stdin=subprocess.PIPE)
def emit(im, sec):
    b = np.asarray(im, np.uint8).tobytes()
    for _ in range(int(round(sec * FPS))): ff.stdin.write(b)
emit(card([("Listening to the Game", 64, (240, 240, 245)), ("a hearing JEPA world model for ARC-AGI-3", 34, (170, 178, 196)), ("the submitted model plays vc33 — turn the sound on" if MODEL else "turn the sound on", 30, (255, 214, 64))]), 2.0)
for k, (a, act, b, ding, buzz, li) in enumerate(EV):
    m = toc(a, b); n = int(round(STEP * FPS))
    for q in range(n):
        frame = a if q < Q else b
        kk = k + 1 - sum(1 for e in EV[:k] if e[5] < li)
        emit(scene(frame, m if q >= Q else None, ding=(1 - q / n) if ding and q >= Q else 0, act=act, lvl=li, k=kk), 1 / FPS)
emit(card([("A ding is universal.", 60, (120, 230, 160)), ("The path to it is silent.", 44, (235, 235, 240)), ("arc3-submit-critim · submission 57710722", 26, (130, 138, 156))]), 2.5)
ff.stdin.close(); ff.wait(); Path(wav).unlink(); print("écrit", out, len(EV), "gestes", sum(e[3] for e in EV), "dings")
