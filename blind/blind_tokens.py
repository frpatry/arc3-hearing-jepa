"""MODÈLE AVEUGLE — étape 1 : l'EXTRACTEUR (1er nov., idée user : « JEPA sans l'œil, juste les actions, les sons et des objets
hypothétiques qui représentent ce qui bouge/change, pas en pixels » ; l'œil reviendra ensuite, pour qu'il VOIE ce qu'il comprend).
Par pas (image t -> t+1), sans apprentissage, une règle fixe :
  OBJETS HYPOTHÉTIQUES = taches de cases qui viennent de changer (8-connexes, « destin commun ») ; par tache : centre, taille,
    couleur avant / après, et DÉPLACEMENT si une même couleur a quitté une partie de la tache pour en occuper une autre
    (genre : bouge / apparaît / disparaît / recolore).
  SONS : ding (niveau franchi), buzz (partie perdue), TOC (geste sans aucun effet), + les sons des jeux générés quand ils existent
    (bump, push, touch, collect, open, color, toggle, teleport, danger…), « inconnu » sur les jeux officiels.
  ACTION : le geste et, pour un clic, où.
Ce fichier : l'extracteur + une planche de contrôle (diag/blind_<jeu>.png) pour vérifier que les jetons sont bien « ce qui bouge ».
    .venv/bin/python blind_tokens.py --games ls20,vc33,ar25 --gen gen/data_v4/v4tr_g0000_000.npz --steps 8"""
import argparse, glob, os
from collections import deque
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
GEN_SOUNDS = ["bump", "push", "touch", "click_obj", "collect", "open", "color", "toggle", "teleport", "danger", "energy_low"]


try:
    from scipy import ndimage as _ndi
except ImportError:
    _ndi = None


def components(mask):
    """étiquettes 8-connexes d'un masque booléen (64,64) -> liste de tableaux (n,2) de cases (y,x)."""
    if _ndi is not None:                                                         # rapide (Kaggle)
        lab, k = _ndi.label(mask, structure=np.ones((3, 3), int))
        if k == 0: return []
        ys, xs = np.nonzero(lab); o = np.argsort(lab[ys, xs], kind="stable"); ys, xs = ys[o], xs[o]
        cut = np.searchsorted(lab[ys, xs], np.arange(2, k + 1))
        return [np.stack([y, x], 1) for y, x in zip(np.split(ys, cut), np.split(xs, cut))]
    lab = np.zeros(mask.shape, np.int32); out = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if lab[y0, x0]: continue
        q = deque([(y0, x0)]); lab[y0, x0] = len(out) + 1; cells = []
        while q:
            y, x = q.popleft(); cells.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    v, u = y + dy, x + dx
                    if 0 <= v < 64 and 0 <= u < 64 and mask[v, u] and not lab[v, u]: lab[v, u] = len(out) + 1; q.append((v, u))
        out.append(np.array(cells))
    return out


def objects(a, b):
    """images a -> b (64,64) : une entrée par tache de cases changées."""
    R = []; ch = a != b
    for cells in components(ch):
        ys, xs = cells[:, 0], cells[:, 1]; ca, cb = a[ys, xs], b[ys, xs]
        y0, y1, x0, x1 = max(ys.min() - 1, 0), min(ys.max() + 2, 64), max(xs.min() - 1, 0), min(xs.max() + 2, 64)   # cadre + 1 case
        m = np.zeros((y1 - y0, x1 - x0), bool); m[ys - y0, xs - x0] = True; g = m.copy()
        g[1:] |= m[:-1]; g[:-1] |= m[1:]; g2 = g.copy(); g2[:, 1:] |= g[:, :-1]; g2[:, :-1] |= g[:, 1:]
        ring = g2 & ~ch[y0:y1, x0:x1]                                             # cases voisines restées pareilles = le FOND local
        ar = a[y0:y1, x0:x1][ring]
        bg = int(np.bincount(ar, minlength=16).argmax()) if len(ar) else -1
        na, nb = np.bincount(ca, minlength=16), np.bincount(cb, minlength=16)
        mm0 = np.minimum(na, nb); mm = mm0.copy(); mm[np.bincount(ar, minlength=16) > 0] = 0      # hors TOUTES les couleurs du voisinage (sol, mur…)
        if not mm.any():
            mm = mm0.copy(); mm[bg] = 0 if bg >= 0 else mm[bg]                     # repli : hors fond local seulement
        k = int(np.argmax(mm))                                                   # couleur (hors fond local) qui était là ET y est encore = ce qui s'est déplacé
        fa, fb = na.copy(), nb.copy(); fa[bg] = fb[bg] = 0 if bg >= 0 else 0
        o = dict(y=float(ys.mean()), x=float(xs.mean()), h=int(np.ptp(ys) + 1), w=int(np.ptp(xs) + 1), n=len(cells), fond=bg,
                 avant=int(np.argmax(fa if fa.any() else na)), apres=int(np.argmax(fb if fb.any() else nb)), dy=0.0, dx=0.0, _na=na, _nb=nb)
        if mm[k] > 0:
            pa, pb = cells[ca == k], cells[cb == k]
            gone = [c for c in range(16) if c not in (bg, k) and na[c] > 0 and nb[c] == 0 and np.bincount(ar, minlength=16)[c] == 0]
            o.update(genre="bouge", couleur=k, dy=float(pb[:, 0].mean() - pa[:, 0].mean()), dx=float(pb[:, 1].mean() - pa[:, 1].mean()), _a=pa, _b=pb,
                     avale=bool(gone))                                           # une couleur présente avant a disparu SOUS l'objet qui arrive
        elif not fa.any(): o.update(genre="apparaît", couleur=o["apres"], na=int(nb[o["apres"]]), _a=cells[:0], _b=cells[cb != bg])      # il n'y avait que du fond
        elif not fb.any(): o.update(genre="disparaît", couleur=o["avant"], na=int(na[o["avant"]]), _a=cells[ca != bg], _b=cells[:0])     # il ne reste que du fond
        else: o.update(genre="recolore", _a=cells, _b=cells)
        R.append(o)
    # un objet qui saute plus loin que sa taille = 2 taches : « disparaît » ici + « apparaît » là, même couleur, même nombre de cases -> « bouge »
    for v in [o for o in R if o["genre"] == "disparaît"]:
        c = [o for o in R if o["genre"] == "apparaît" and o["couleur"] == v["couleur"] and o["na"] == v["na"]]
        if c:
            ap = min(c, key=lambda o: (o["y"] - v["y"]) ** 2 + (o["x"] - v["x"]) ** 2); R.remove(ap)
            v.update(genre="bouge", dy=ap["y"] - v["y"], dx=ap["x"] - v["x"], y=ap["y"], x=ap["x"], apres=ap["apres"], saut=True, _b=ap["_b"])
    return sorted(R, key=lambda o: -o["n"])


def _mask(c):
    m = np.zeros((64, 64), bool)
    if len(c): m[c[:, 0], c[:, 1]] = True
    return m


def _grow(m, r=1):
    g = m.copy()
    for _ in range(r):
        h = g.copy(); h[1:] |= g[:-1]; h[:-1] |= g[1:]; h[:, 1:] |= g[:, :-1]; h[:, :-1] |= g[:, 1:]; g = h
    return g


def made_sounds(a, b, O):
    """SONS FABRIQUÉS depuis les mouvements (règles fixes, existent dans TOUS les jeux) :
    toc (rien de notable n'a changé), contact (un objet qui bouge se met à toucher quelque chose de plus qu'avant),
    ramassé (quelque chose disparaît là où un objet arrive), couleur (un objet change de couleur sur place),
    téléport (un objet saute loin d'un coup)."""
    sc = int(np.bincount(a.ravel(), minlength=16).argmax()); mv = [o for o in O if o["genre"] == "bouge"]
    s = dict(toc=not any(o["n"] > 4 for o in O), contact=False, ramassé=False, couleur=any(o["genre"] == "recolore" for o in O),
             téléport=False)
    # TÉLÉPORT : une même quantité d'une couleur (≥ 4 cases, pas le fond de la scène) disparaît d'une tache et apparaît dans une autre, loin
    if len(O) <= 8:
        for i, p in enumerate(O):
            for q in O[i + 1:]:
                if (p["y"] - q["y"]) ** 2 + (p["x"] - q["x"]) ** 2 < 36: continue
                for u, v in ((p, q), (q, p)):
                    d1, d2 = u["_na"] - u["_nb"], v["_nb"] - v["_na"]; d1[sc] = d2[sc] = 0
                    back = (u["_nb"][u["fond"]] - u["_na"][u["fond"]]) if u["fond"] >= 0 else 0      # l'endroit quitté redevient du FOND
                    s["téléport"] |= bool(((d1 >= 4) & (d1 == d2) & (back >= d1) & (u["_nb"] == 0) & (v["_na"] == 0)).any())   # partie EN ENTIER d'ici, absente de là-bas
    for o in mv:
        f = o["fond"]; Ma, Mb = _mask(o["_a"]), _mask(o["_b"])
        ta = (_grow(Ma) & ~Ma & (a != sc) & (a != f) & (a != o["couleur"])).sum(); tb = (_grow(Mb) & ~Mb & (b != sc) & (b != f) & (b != o["couleur"])).sum()
        s["contact"] |= bool(tb > ta)
        s["ramassé"] |= o.get("avale", False) or any(v["genre"] == "disparaît" and (_grow(Mb, 2) & _mask(v["_a"])).any() for v in O)
    return s


def sounds(z, t):
    """sons du pas t (transition obs[t] -> next_obs[t])."""
    o, n = z["obs"][t], z["next_obs"][t]; L, S = z["level"], z["state"]; last = t == len(L) - 1 or z["ep"][t + 1] != z["ep"][t]
    s = dict(ding=bool((not last and L[t + 1] > L[t]) or (last and S[t] == 1)), buzz=bool(not last and S[t + 1] == 2 or (last and S[t] == 2)),
             )
    e = z["events"][t]; names = list(map(str, z["event_names"]))
    s.update(made_sounds(o.astype(np.int64), n.astype(np.int64), objects(o.astype(np.int64), n.astype(np.int64))))
    s["jeu"] = None if np.isnan(e).any() else [nm for nm in GEN_SOUNDS if e[names.index(nm)] > 0]   # None = inconnu (jeux officiels)
    return s


def sheet(z, ts, out, title):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from gen.sheet import PALETTE
    AN = {0: "reset", 1: "↑", 2: "↓", 3: "←", 4: "→", 5: "espace", 6: "clic", 7: "annule"}
    fig, ax = plt.subplots(2, len(ts), figsize=(3.1 * len(ts), 7.4), gridspec_kw=dict(height_ratios=[3, 1.6]))
    for j, t in enumerate(ts):
        a, b = z["obs"][t].astype(np.int64), z["next_obs"][t].astype(np.int64); O = objects(a, b); s = sounds(z, t)
        A = ax[0, j]; A.imshow(PALETTE[np.clip(b, 0, 15)]); A.imshow(np.ma.masked_where(a == b, np.ones_like(a)), cmap="gray", alpha=0.35, vmin=0, vmax=1)
        for i, o in enumerate(O[:8]):
            A.add_patch(plt.Rectangle((o["x"] - o["w"] / 2, o["y"] - o["h"] / 2), o["w"], o["h"], fill=False, ec="red", lw=1.2))
            if o["genre"] == "bouge" and abs(o["dy"]) + abs(o["dx"]) > 0.5:
                A.annotate("", (o["x"] + o["dx"] / 2, o["y"] + o["dy"] / 2), (o["x"] - o["dx"] / 2, o["y"] - o["dy"] / 2), arrowprops=dict(arrowstyle="->", color="yellow", lw=1.8))
            A.text(o["x"] - o["w"] / 2, o["y"] - o["h"] / 2 - 0.5, str(i + 1), color="red", fontsize=7)
        act = AN.get(int(z["act"][t]), str(z["act"][t])) + (f" ({z['click'][t][0]},{z['click'][t][1]})" if int(z["act"][t]) == 6 else "")
        A.set_title(f"pas {t} | geste {act}", fontsize=8); A.axis("off")
        if act.startswith("clic"): A.plot(z["click"][t][0], z["click"][t][1], "x", color="cyan", ms=8, mew=2)
        snd = [k.upper() for k in ("ding", "buzz", "toc") if s[k]] + [k for k in ("contact", "ramassé", "couleur", "téléport") if s[k]]
        txt = "SONS : " + (", ".join(snd) or "—") + "\n(générateur : " + ("aucun" if s["jeu"] is None else ", ".join(s["jeu"]) or "—") + ")\n" + "\n".join(
            f"{i + 1}. {o['genre']} n={o['n']} {o['h']}×{o['w']} coul {o['avant']}→{o['apres']}" + (f" d=({o['dy']:+.1f},{o['dx']:+.1f})" if o["genre"] == "bouge" else "")
            for i, o in enumerate(O[:8])) + (f"\n… +{len(O) - 8}" if len(O) > 8 else "")
        ax[1, j].text(0, 1, txt, va="top", fontsize=6.5, family="monospace"); ax[1, j].axis("off")
    fig.suptitle(title + "  (gris = cases changées ; rouge = objets hypothétiques ; jaune = déplacement)", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig); print(out)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--games", default="ls20,vc33,ar25"); p.add_argument("--gen", default="gen/data_v4/v4tr_g0000_000.npz")
    p.add_argument("--steps", type=int, default=8); p.add_argument("--start", type=int, default=3); a = p.parse_args()
    os.makedirs(os.path.join(HERE, "diag"), exist_ok=True); stats = []
    for g in a.games.split(","):
        z = np.load(glob.glob(os.path.join(HERE, f"data/oracle_public/{g}*_oracle.npz"))[0])
        sheet(z, list(range(a.start, a.start + a.steps)), os.path.join(HERE, f"diag/blind_{g}.png"), f"{g} (officiel)")
        n = [len(objects(z["obs"][t].astype(np.int64), z["next_obs"][t].astype(np.int64))) for t in range(min(200, len(z["obs"])))]
        stats.append((g, np.median(n), np.percentile(n, 90), max(n)))
    z = np.load(os.path.join(HERE, a.gen)); e0 = int(z["ep"][0]); ii = np.flatnonzero(z["ep"] == e0)
    sheet(z, list(ii[a.start:a.start + a.steps]), os.path.join(HERE, "diag/blind_gen.png"), f"jeu généré {z['game_ids'][z['game'][ii[0]]]}")
    for g, md, p90, mx in stats: print(f"{g} : objets par pas médiane {md:.0f}, 90 % {p90:.0f}, max {mx}")


if __name__ == "__main__":
    main()
