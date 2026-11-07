"""MODÈLE AVEUGLE — étape 2 : nos parties -> suites de pas SANS PIXELS (1er nov.). Par pas :
  ACTION (geste, clic) ; SONS FABRIQUÉS (toc, contact, ramassé, couleur, téléport) + DING (niveau franchi) + BUZZ (perdu) ;
  OBJETS HYPOTHÉTIQUES (≤ K taches qui ont changé : position, taille, genre, couleurs avant/après, déplacement, saut, avalé, habitué) ;
  CIBLES : étiquettes du générateur (11 sons, -1 = inconnu sur les jeux officiels) ; pas restants avant le prochain ding (sondes).
HABITUATION (comme un bébé) : une tache qui recolore au MÊME endroit (≤ 3 cases, même taille ±2) qu'au pas précédent est « habituée »
  et ne fait plus sonner « couleur » (le compteur d'énergie qui bouge à chaque geste = tic-tac, pas une information).
    python blind_data.py --src gen/data_v3,gen/data_v4,... --official data/oracle_public --out /kaggle/tmp/blind --procs 4"""
import argparse, glob, os, time
from multiprocessing import Pool
import numpy as np
import blind_tokens as B
K = 12                                                     # objets gardés par pas (les plus gros)
EYE = os.environ.get("BLIND_EYE", "0") == "1"             # œil minimal (liste des objets à l'écran) dans les données
NF = 14                                                    # y x h w logn genre×4 dy dx saut avalé habitué
GENRES = ["bouge", "apparaît", "disparaît", "recolore"]
SND = ["toc", "contact", "ramassé", "couleur", "téléport", "ding", "buzz"]
TEST = {"g50t", "ka59", "lf52", "r11l", "sk48"}


KS, NFS = 24, 5                                            # ŒIL MINIMAL : ≤ 24 objets à l'écran, (y x h w logn) + couleur


def scene(g):
    """ŒIL MINIMAL (1er nov., go user) : les objets PRÉSENTS à l'écran (composantes 8-connexes d'une même couleur, sans le fond géant),
    les plus gros d'abord -> (KS,NFS), couleurs (KS,), nombre. Pas de pixels : de quoi situer l'état."""
    out = []
    for c in np.unique(g):
        for cells in B.components(g == c):
            if len(cells) > 0.25 * g.size: continue
            ys, xs = cells[:, 0], cells[:, 1]; out.append((len(cells), int(c), ys.mean(), xs.mean(), np.ptp(ys) + 1, np.ptp(xs) + 1))
    out.sort(key=lambda o: -o[0]); f = np.zeros((KS, NFS), np.float16); col = np.full(KS, 16, np.int8)
    for i, (n, c, y, x, h, w) in enumerate(out[:KS]): f[i] = [y / 64, x / 64, h / 64, w / 64, np.log1p(n) / 8]; col[i] = c
    return f, col, min(len(out), KS)


def step_feats(a, b, prev_rec, ding, buzz):
    """UN pas (image a -> b) -> objets (K,NF), couleurs (K,2), nb d'objets, sons (7,), taches « recolore » (pour l'habituation au pas suivant).
    Utilisé à l'identique pour fabriquer les données ET en jeu (blind_play.py)."""
    O = B.objects(a, b); rec = [o for o in O if o["genre"] == "recolore"]
    for o in rec:
        o["hab"] = any(abs(o["y"] - p["y"]) <= 3 and abs(o["x"] - p["x"]) <= 3 and abs(o["n"] - p["n"]) <= 2 for p in prev_rec)
    s = B.made_sounds(a, b, O); s["couleur"] = any(not o["hab"] for o in rec); s["ding"], s["buzz"] = ding, buzz
    obj = np.zeros((K, NF), np.float16); col = np.full((K, 2), 16, np.int8)
    for i, o in enumerate(O[:K]):
        g = GENRES.index(o["genre"])
        obj[i] = [o["y"] / 64, o["x"] / 64, o["h"] / 64, o["w"] / 64, np.log1p(o["n"]) / 8, g == 0, g == 1, g == 2, g == 3,
                  o["dy"] / 16, o["dx"] / 16, float(o.get("saut", False)), float(o.get("avale", False)), float(o.get("hab", False))]
        col[i] = [o["avant"], o["apres"]]
    return obj, col, min(len(O), K), np.array([s[k] for k in SND], np.int8), rec


def encode(path, src, cap=0):
    z = dict(np.load(path)); N = len(z["obs"]) if not cap else min(cap, len(z["obs"]))   # dict : tout en mémoire UNE fois (np.load redécompresse à chaque accès)
    z["files"] = list(z)
    if "ep" not in z: z["ep"] = np.r_[0, np.cumsum(z["done"][:-1])].astype(np.int32)        # data_rules : épisodes déduits de « done »
    OB, NX = z["obs"], z["next_obs"]; names = list(map(str, z["event_names"])); E = z["events"]; L, S, EP = z["level"], z["state"], z["ep"]
    OBJ = np.zeros((N, K, NF), np.float16); COL = np.full((N, K, 2), 16, np.int8); NO = np.zeros(N, np.int8)
    SCN = np.zeros((N, KS, NFS), np.float16); SCOL = np.full((N, KS), 16, np.int8); NS = np.zeros(N, np.int8)
    SN = np.zeros((N, len(SND)), np.int8); LAB = np.full((N, len(B.GEN_SOUNDS)), -1, np.int8); STW = np.full(N, -1, np.int16)
    prev_rec, prev_ep = [], None
    for t in range(N):
        if EP[t] != prev_ep: prev_rec, prev_ep = [], EP[t]
        last = t == len(L) - 1 or EP[t + 1] != EP[t]
        ding = bool((not last and L[t + 1] > L[t]) or (last and S[t] == 1)); buzz = bool((not last and S[t + 1] == 2) or (last and S[t] == 2))
        OBJ[t], COL[t], NO[t], SN[t], prev_rec = step_feats(OB[t].astype(np.int64), NX[t].astype(np.int64), prev_rec, ding, buzz)
        if EYE: SCN[t], SCOL[t], NS[t] = scene(NX[t].astype(np.int64))               # l'écran APRÈS le pas
        if not np.isnan(E[t]).any(): LAB[t] = [E[t, names.index(nm)] > 0 for nm in B.GEN_SOUNDS]
    for e in np.unique(EP[:N]):                            # pas restants avant le prochain ding (dans l'épisode)
        ii = np.flatnonzero(EP[:N] == e); nxt = -1
        for j in ii[::-1]:
            if SN[j, 5]: nxt = j
            STW[j] = nxt - j if nxt >= 0 else -1
    gid = z["game_ids"][z["game"][:N]] if "game_ids" in z["files"] else np.array([os.path.basename(path)[:4]] * N)
    bn = list(map(str, z["behav_names"])) if "behav_names" in z["files"] else ["oracle"]
    return dict(OBJ=OBJ, COL=COL, NO=NO, **(dict(SCN=SCN, SCOL=SCOL, NS=NS) if EYE else {}), SND=SN, LAB=LAB, STW=STW, ACT=z["act"][:N].astype(np.int8), CLK=z["click"][:N].astype(np.int8),
                EP=EP[:N].astype(np.int32), LV=L[:N].astype(np.int8), GAME=np.array([str(g) for g in gid]),
                ORACLE=(z["behav"][:N] == bn.index("oracle")) if "behav_names" in z["files"] else np.full(N, "behav" not in z["files"]), SRC=np.full(N, src, np.int8))


def job(args):
    path, src, cap, out = args; t0 = time.time()
    fn = os.path.join(out, f"{src}_{os.path.basename(os.path.dirname(path))}_{os.path.basename(path)}")
    try:
        if not os.path.exists(fn): np.savez_compressed(fn, **encode(path, src, cap))
    except Exception as e:                                                        # un fichier qui plante n'arrête plus tout
        return f"ÉCHEC {fn} : {type(e).__name__} {e}", time.time() - t0
    return fn, time.time() - t0


def main():
    p = argparse.ArgumentParser(); p.add_argument("--src", default="gen/data_v3,gen/data_v4,gen/data_demo,gen/data_rules")
    p.add_argument("--val", default="gen/data_v3_val,gen/data_v4_val,gen/data_demo_val"); p.add_argument("--official", default="data/oracle_public")
    p.add_argument("--out", default="data/blind"); p.add_argument("--procs", type=int, default=4); p.add_argument("--cap", type=int, default=0)
    p.add_argument("--max_files", type=int, default=0); a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
    J = []
    for src, dirs in ((0, a.src), (1, a.val)):                                # 0 = générés (entraînement), 1 = générés inédits (mesure)
        for d in dirs.split(","):
            fs = sorted(glob.glob(os.path.join(d, "*.npz")));  fs = fs[:a.max_files] if a.max_files else fs
            J += [(f, src, a.cap, a.out) for f in fs]
    for f in sorted(glob.glob(os.path.join(a.official, "*_oracle.npz"))):     # 2 = officiels vus, 3 = officiels EXCLUS (le vrai test)
        J.append((f, 3 if os.path.basename(f)[:4] in TEST else 2, 0, a.out))
    t0 = time.time()
    with Pool(a.procs) as P:
        for i, (fn, dt) in enumerate(P.imap_unordered(job, J)):
            print(f"[{(time.time() - t0) / 60:.1f} min] {i + 1}/{len(J)} {os.path.basename(fn)} {dt:.0f}s", flush=True)


if __name__ == "__main__":
    main()
