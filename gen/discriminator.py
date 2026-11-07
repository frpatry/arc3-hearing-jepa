"""Détecteur de « faux » : un petit CNN (CPU, quelques minutes) qui classe OFFICIEL vs GÉNÉRÉ.
Mesure de ressemblance qu'on ne peut pas tricher à la main : 50 % = indiscernable, 100 % = famille reconnaissable.

    python -m gen.discriminator --gen v2=gen/data_v2 --gen v3=gen/data_v3 --gen v4=gen/data_v4 \
        [--modes frames,trans] [--perm 1] [--folds 5] [--out gen/disc_results.json]

Entrées :
  frames : une image (one-hot 16 couleurs) ;
  trans  : (image, image suivante, différence) = 16 + 16 + 1 canaux -> dynamique / comportement.
Données : transitions publiques officielles (wm/data/public/*.npz, 25 jeux) vs un jeu de données généré (shards + games.json).
  - classes équilibrées (train ET test), chaque jeu officiel pèse autant ;
  - validation croisée PAR JEU (k plis) : les jeux de test (officiels et générés) ne sont jamais vus à l'entraînement ;
  - côté généré : pas de RESET (act 0, absents des enregistrements officiels) ni de lignes recopiées (dup) ;
  - --perm 1 : permutation aléatoire des 16 couleurs par exemple (même permutation pour l'image et la suivante),
    pour que le détecteur ne puisse pas se contenter de la palette ; --perm 0 : couleurs brutes.
Sorties : exactitude moyenne sur les plis (tenus à l'écart), AUC, et par famille générée la part détectée
(prédite « générée ») = ce qui rend la famille facile à repérer ; par jeu officiel la part prise pour un jeu officiel.
Contrôles : --only_explorer (côté généré, seulement les pas joués par l'Explorer, comme les enregistrements officiels),
--max_level k (seulement les niveaux < k, les enregistrements officiels sont surtout au niveau 0)."""
import argparse
import glob
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


# ----------------------------------------------------------------------------- données
def load_official(per_game, rng):
    """-> dict jeu -> (obs, next_obs) sous-échantillonnés (per_game transitions par jeu)."""
    out = {}
    for f in sorted(glob.glob(str(ROOT / "wm" / "data" / "public" / "*.npz"))):
        z = np.load(f)
        n = len(z["obs"])
        idx = np.sort(rng.choice(n, min(per_game, n), replace=False))
        out[Path(f).stem] = (z["obs"][idx], z["next_obs"][idx])
    return out


def load_generated(path, per_game, rng, only_explorer=False, max_level=None):
    """-> dict game_id -> (obs, next_obs), et game_id -> famille."""
    games = json.load(open(os.path.join(path, "games.json")))
    fam_of = {g["game_id"]: g["family"] for g in games if g}
    acc = defaultdict(lambda: ([], []))
    for f in sorted(glob.glob(os.path.join(path, "*.npz"))):
        z = np.load(f)
        ids = z["game_ids"]
        keep = z["act"] != 0
        if "dup" in z.files:
            keep &= ~z["dup"]
        if only_explorer:
            names = [str(x) for x in z["behav_names"]] if "behav_names" in z.files else ["explorer", "random", "oracle"]
            keep &= z["behav"] == names.index("explorer")
        if max_level is not None:
            keep &= z["level"] < max_level
        gm = z["game"]
        sel_all = np.flatnonzero(keep)
        if not len(sel_all):
            continue
        obs, nxt = z["obs"], z["next_obs"]
        for gi in np.unique(gm[sel_all]):
            sel = sel_all[gm[sel_all] == gi]
            if len(sel) > 4 * per_game:                      # sous-échantillonnage tôt (mémoire)
                sel = np.sort(rng.choice(sel, 4 * per_game, replace=False))
            a, b = acc[str(ids[gi])]
            a.append(obs[sel])
            b.append(nxt[sel])
    out = {}
    for gid, (a, b) in acc.items():
        o, n = np.concatenate(a), np.concatenate(b)
        idx = np.sort(rng.choice(len(o), min(per_game, len(o)), replace=False))
        out[gid] = (o[idx], n[idx])
    return out, fam_of


def folds_of(keys, k, rng):
    keys = sorted(keys)
    perm = rng.permutation(len(keys))
    return [[keys[j] for j in perm[f::k]] for f in range(k)]


def take(d, keys, n, rng):
    """n transitions réparties également sur les jeux `keys` -> (obs, nxt, game_key list)."""
    per = max(1, int(np.ceil(n / max(1, len(keys)))))
    O, N, K = [], [], []
    for key in keys:
        o, x = d[key]
        idx = rng.choice(len(o), min(per, len(o)), replace=False)
        O.append(o[idx])
        N.append(x[idx])
        K += [key] * len(idx)
    O, N, K = np.concatenate(O), np.concatenate(N), np.array(K)
    if len(O) > n:
        idx = rng.choice(len(O), n, replace=False)
        O, N, K = O[idx], N[idx], K[idx]
    return O, N, K


# ----------------------------------------------------------------------------- modèle
class Det(nn.Module):
    """CNN minuscule : patchify 2×2 -> 3 blocs conv -> pooling global moyen + max -> logit."""

    def __init__(self, cin, w=24):
        super().__init__()
        self.p = nn.Conv2d(cin, w, 2, stride=2)                     # 64 -> 32
        self.c1 = nn.Conv2d(w, 2 * w, 3, padding=1)                 # 32
        self.c2 = nn.Conv2d(2 * w, 3 * w, 3, padding=1)             # 16
        self.c3 = nn.Conv2d(3 * w, 4 * w, 3, padding=1)             # 8
        self.fc = nn.Linear(8 * w, 1)

    def forward(self, x):
        x = F.relu(self.p(x))
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = F.relu(self.c3(x))
        x = torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1)
        return self.fc(x).squeeze(1)


MASK_BORDER = 0                                                   # --mask_border k : bords (barre de pas) masqués


def encode(obs, nxt, mode, perm, rng):
    """int8 (B,64,64) -> float (B,C,64,64) ; permutation de couleurs par exemple si perm."""
    if MASK_BORDER:
        k = MASK_BORDER
        obs, nxt = obs.copy(), nxt.copy()
        for a in (obs, nxt):
            a[:, :k, :] = a[:, -k:, :] = 0
            a[:, :, :k] = a[:, :, -k:] = 0
    o = torch.from_numpy(obs.astype(np.int64)) & 15
    if perm:
        P = torch.from_numpy(np.argsort(rng.random((len(obs), 16)), 1))
        o = torch.gather(P, 1, o.view(len(o), -1)).view_as(o)
    x = F.one_hot(o, 16).permute(0, 3, 1, 2).float()
    if mode == "frames":
        return x
    n = torch.from_numpy(nxt.astype(np.int64)) & 15
    diff = (torch.from_numpy(obs) != torch.from_numpy(nxt)).float().unsqueeze(1)
    if perm:
        n = torch.gather(P, 1, n.view(len(n), -1)).view_as(n)
    y = F.one_hot(n, 16).permute(0, 3, 1, 2).float()
    return torch.cat([x, y, diff], 1)


def train_eval(tr, te, mode, perm, a, seed):
    """tr / te = (obs, nxt, label) -> (proba « généré » sur te, temps)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cin = 16 if mode == "frames" else 33
    net = Det(cin, a.width)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    O, N, Y = tr
    n = len(Y)
    steps = a.epochs * (n // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=max(1, steps))
    t0 = time.time()
    net.train()
    for ep in range(a.epochs):
        order = rng.permutation(n)
        for k in range(n // a.bs):
            b = order[k * a.bs:(k + 1) * a.bs]
            x = encode(O[b], N[b], mode, perm, rng)
            y = torch.from_numpy(Y[b]).float()
            loss = F.binary_cross_entropy_with_logits(net(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    net.eval()
    O2, N2, _ = te
    ps = []
    with torch.no_grad():
        for k in range(0, len(O2), 512):
            ps.append(torch.sigmoid(net(encode(O2[k:k + 512], N2[k:k + 512], mode, perm, rng))).numpy())
    return np.concatenate(ps), time.time() - t0


def auc(p, y):
    order = np.argsort(p)
    r = np.empty(len(p))
    r[order] = np.arange(1, len(p) + 1)
    n1, n0 = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0))


def best_acc(p, y):
    """exactitude équilibrée au MEILLEUR seuil (borne haute, indépendante du calibrage du seuil 0,5)."""
    order = np.argsort(p)
    ys = y[order]
    n1, n0 = ys.sum(), len(ys) - ys.sum()
    # seuil juste après k exemples : vrais négatifs = zéros parmi les k premiers, vrais positifs = uns après
    tn = np.concatenate([[0], np.cumsum(1 - ys)])
    tp = n1 - np.concatenate([[0], np.cumsum(ys)])
    return float(np.max(0.5 * (tn / max(1, n0) + tp / max(1, n1))))


def run_one(label, gen, fam_of, off, mode, perm, a):
    rng = np.random.default_rng(a.seed)
    fo = folds_of(list(off), a.folds, rng)
    fg = folds_of(list(gen), a.folds, rng)
    accs, aucs, bests, fam_det, off_pass, gen_pass = [], [], [], defaultdict(list), defaultdict(list), []
    tt = 0.0
    for f in range(a.folds):
        tr_o = [k for j in range(a.folds) if j != f for k in fo[j]]
        tr_g = [k for j in range(a.folds) if j != f for k in fg[j]]
        Oo, No, _ = take(off, tr_o, a.n_train, rng)
        Og, Ng, _ = take(gen, tr_g, len(Oo), rng)
        m = min(len(Oo), len(Og))
        tr = (np.concatenate([Oo[:m], Og[:m]]), np.concatenate([No[:m], Ng[:m]]),
              np.concatenate([np.zeros(m), np.ones(m)]).astype(np.float32))
        To, TNo, Ko = take(off, fo[f], a.n_test, rng)
        Tg, TNg, Kg = take(gen, fg[f], len(To), rng)
        m2 = min(len(To), len(Tg))
        te = (np.concatenate([To[:m2], Tg[:m2]]), np.concatenate([TNo[:m2], TNg[:m2]]), None)
        y = np.concatenate([np.zeros(m2), np.ones(m2)])
        p, dt = train_eval(tr, te, mode, perm, a, a.seed + f)
        tt += dt
        acc = float(((p > 0.5) == y).mean())
        accs.append(acc)
        aucs.append(auc(p, y))
        bests.append(best_acc(p, y))
        for key, pp in zip(Ko[:m2], p[:m2]):
            off_pass[key].append(pp < 0.5)
        for key, pp in zip(Kg[:m2], p[m2:]):
            fam_det[fam_of.get(key, "?")].append(pp > 0.5)
        print(f"  [{label} {mode} perm={perm}] pli {f}: exactitude {100 * acc:.1f} %  (meilleur seuil {100 * bests[-1]:.1f} %)  AUC {aucs[-1]:.3f}  "
              f"({len(tr[2])} ex. train, {2 * m2} test, {dt:.0f} s)", flush=True)
    res = dict(acc=float(np.mean(accs)), acc_std=float(np.std(accs)), auc=float(np.mean(aucs)), folds=accs,
               acc_best=float(np.mean(bests)), acc_best_std=float(np.std(bests)),
               family_detected={k: float(np.mean(v)) for k, v in sorted(fam_det.items())},
               family_n={k: len(v) for k, v in fam_det.items()},
               official_passed={k: float(np.mean(v)) for k, v in sorted(off_pass.items())}, sec=tt)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen", action="append", default=[], help="label=dossier de shards (avec games.json)")
    p.add_argument("--modes", default="frames,trans")
    p.add_argument("--perm", default="1", help="1, 0 ou 0,1")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--per_game_off", type=int, default=1200)
    p.add_argument("--per_game_gen", type=int, default=60)
    p.add_argument("--n_train", type=int, default=12000, help="exemples par classe à l'entraînement")
    p.add_argument("--n_test", type=int, default=3000, help="exemples par classe en test (par pli)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--width", type=int, default=24)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--only_explorer", action="store_true")
    p.add_argument("--max_level", type=int, default=None)
    p.add_argument("--max_games", type=int, default=0, help="garder au plus N jeux générés (sondes à petit échantillon)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="")
    p.add_argument("--mask_border", type=int, default=0, help="ablation : masquer k px sur chaque bord (barre de pas)")
    a = p.parse_args()
    global MASK_BORDER
    MASK_BORDER = a.mask_border
    torch.set_num_threads(a.threads)
    rng = np.random.default_rng(a.seed)
    off = load_official(a.per_game_off, rng)
    results = {}
    if a.out and Path(a.out).exists():
        results = json.loads(Path(a.out).read_text())
    for spec in a.gen:
        label, path = spec.split("=", 1)
        t0 = time.time()
        gen, fam_of = load_generated(path, a.per_game_gen, rng, a.only_explorer, a.max_level)
        if a.max_games and len(gen) > a.max_games:              # sonde équitable : même nombre de jeux partout
            keep = sorted(gen)[::max(1, len(gen) // a.max_games)][:a.max_games]
            gen = {k: gen[k] for k in keep}
        print(f"{label}: {len(gen)} jeux, {sum(len(v[0]) for v in gen.values())} transitions ({time.time() - t0:.0f} s)",
              flush=True)
        for mode in a.modes.split(","):
            for perm in [int(x) for x in a.perm.split(",")]:
                key = f"{label}|{mode}|perm{perm}" + ("|explorer" if a.only_explorer else "") + \
                      (f"|mask{a.mask_border}" if a.mask_border else "") + \
                      (f"|lvl<{a.max_level}" if a.max_level is not None else "")
                r = run_one(label, gen, fam_of, off, mode, perm, a)
                r.update(data=path, n_games=len(gen))
                results[key] = r
                fams = sorted(r["family_detected"].items(), key=lambda kv: -kv[1])
                print(f"==> {key}: exactitude {100 * r['acc']:.1f} ± {100 * r['acc_std']:.1f} %  meilleur seuil "
                      f"{100 * r['acc_best']:.1f} ± {100 * r['acc_best_std']:.1f} %  AUC {r['auc']:.3f}  "
                      f"({r['sec']:.0f} s)\n    familles détectées : " +
                      ", ".join(f"{k} {100 * v:.0f} %" for k, v in fams) +
                      "\n    officiels pris pour officiels : " +
                      ", ".join(f"{k} {100 * v:.0f} %" for k, v in sorted(r["official_passed"].items(), key=lambda kv: kv[1])),
                      flush=True)
                if a.out:
                    Path(a.out).write_text(json.dumps(results, indent=1))
    print("\n| données | entrée | couleurs | exactitude seuil 0,5 (plis par jeu) | exactitude meilleur seuil | AUC |")
    print("|---|---|---|---|---|---|")
    for k, r in results.items():
        lab, mode, perm = k.split("|")[:3]
        extra = " ".join(k.split("|")[3:])
        print(f"| {lab} {extra} | {mode} | {'permutées' if perm == 'perm1' else 'brutes'} | "
              f"{100 * r['acc']:.1f} ± {100 * r['acc_std']:.1f} % | "
              f"{100 * r.get('acc_best', float('nan')):.1f} % | {r['auc']:.3f} |")


if __name__ == "__main__":
    main()
