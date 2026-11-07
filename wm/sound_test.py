"""CANAL SON EN ENTRÉE (idée user, 26 oct.) — test rapide, même protocole que dream_test.py (3 familles v4 × 10 jeux, mesure sur jeux
inédits des mêmes familles et sur 3 familles jamais vues). Le modèle ENTEND à chaque pas ce que le geste précédent a produit :
zones qui ont bougé, ding de fin de niveau, buzz de game over ; et reçoit l'INTENTION (succès dans les 8 prochains pas ? appris après
coup, mis à « oui » en jeu). Rêve = action inconnue « ? » + intention « gagner », comparé à « ? » + intention « rien ».
Mesures : geste du joueur qui a réussi retrouvé (dist / contraste / direction), et « l'ouïe aide-t-elle à prédire ? ».
    python wm/sound_test.py --ckpt A_actor.pt --data datav4 --val datav4_val --steps 3000 [--heard 0 = témoin sans ouïe]"""
import argparse, glob, os, sys, time
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import NA
from model_seq import SeqWorldModel
from dream_test import TokEmb

Q = NA                                                # action inconnue « ? »


def episodes(dirs, fams, n_games, T, seed=0, H=8):
    W = []; keep = {}
    for f in sorted(sum([glob.glob(f"{d}/*.npz") for d in dirs], [])):
        z = np.load(f); gid = [str(x) for x in z["game_ids"]]; fam = np.array([gid[i].split("-")[0] for i in z["game"]])
        sel = np.isin(fam, fams)
        if not sel.any(): continue
        o, n, a, c, ev, st, ep, gm = z["obs"], z["next_obs"], z["act"].astype(np.int64), z["click"].astype(np.int64), z["events"].astype(np.float32), z["state"], z["ep"], z["game"]
        en = [str(x) for x in z["event_names"]]; N = len(a)
        succ = ((st == 1) | (ev[:, en.index("level_complete")] > 0.5)).astype(np.int64); go = (st == 2).astype(np.int64)   # ding / buzz de CETTE transition
        rtg = np.zeros(N, np.int64)                                                  # intention : un succès dans les H prochains pas (même épisode)
        for i in range(N):
            j = i
            while j < min(N, i + H) and ep[j] == ep[i]:
                if succ[j]: rtg[i] = 1; break
                j += 1
        wins = {e for e in np.unique(ep[sel]) if succ[ep == e].any()}
        cont = np.concatenate([[False], (o[1:] == n[:-1]).all((1, 2)) & (ep[1:] == ep[:-1])]); run = np.zeros(N, np.int64)
        for i in range(1, N): run[i] = run[i - 1] + 1 if cont[i] else 0
        avail = {g: sorted(set(a[gm == g].tolist()) - {0}) for g in np.unique(gm[sel])}
        for i in np.nonzero(sel & (run >= T - 1))[0]:
            g = gm[i]; k = (fam[i], gid[g])
            if k not in keep:
                if sum(1 for kk in keep if kk[0] == fam[i]) >= n_games: continue
                keep[k] = True
            s = slice(i - T + 1, i + 1); pr = slice(i - T, i)                         # son entendu au pas t = transition t-1 -> t
            lcp = np.concatenate([[0], succ[pr][1:]]) if i - T >= 0 else np.zeros(T, np.int64); gop = np.concatenate([[0], go[pr][1:]]) if i - T >= 0 else np.zeros(T, np.int64)
            lab = "win" if rtg[i] else ("lose" if ep[i] not in wins else "none")
            W.append(dict(g=np.concatenate([o[s], n[i][None]]), a=a[s], c=c[s], e=ev[s], lc=lcp, go=gop, rtg=rtg[s], lab=lab, fam=fam[i], avail=avail[g]))
    return W


def snd(m, g, lc, go, rtg, heard):
    if heard: return m.sound(g, lc, go, rtg)
    return m.pred.s_rtg[rtg.long()][:, :, None].expand(-1, -1, 64, -1)                # témoin : intention seule, sourd


def batch(W, idx, dev, tok, rng):
    t = lambda k, dt=None: torch.tensor(np.stack([W[i][k] for i in idx]), device=dev)
    g, a, c, e, lc, go, rtg = t("g").long(), t("a"), t("c"), t("e"), t("lc"), t("go"), t("rtg")
    T = a.size(1)
    for b in range(len(idx)):                         # action CACHÉE à partir d'un pas tiré au hasard (le rêve s'apprend sur TOUTES les fenêtres)
        if rng.rand() < tok: t0 = rng.randint(0, T); a[b, t0:] = Q; c[b, t0:] = -1
    return g, a, c, e, lc, go, rtg


@torch.no_grad()
def evaluate(m, W, dev, heard, n=200, seed=1):
    m.eval(); rng = np.random.RandomState(seed); idx = [i for i, w in enumerate(W) if w["lab"] == "win" and w["a"][-1] > 0]
    if not idx: return None
    idx = rng.choice(idx, min(n, len(idx)), replace=False); M = ("dist", "contraste", "direction")
    R = {k: 0 for k in M} | {k + "_clk": 0 for k in M} | dict(chance=0.0, nclk=0, e_true=0.0, e_sourd=0.0, e_copy=0.0)
    for i in idx:
        w = W[i]; T = len(w["a"]); ta, tc = int(w["a"][-1]), tuple(int(v) for v in w["c"][-1])
        cand = [(x, (-1, -1)) for x in w["avail"] if x != 6]
        if 6 in w["avail"]:
            last = w["g"][T - 1]; bgc = np.bincount(last.ravel().astype(np.int64), minlength=16).argmax(); ys, xs = np.nonzero(last != bgc)
            cl = [tc] if ta == 6 else []
            for j in rng.permutation(len(ys))[:40]:
                p = (int(xs[j]), int(ys[j]))
                if all((p[0] // 8, p[1] // 8) != (q[0] // 8, q[1] // 8) for q in cl): cl.append(p)
                if len(cl) >= 8: break
            cand += [(6, p) for p in cl]
        cand = [cand[k] for k in rng.permutation(len(cand))]
        tr = [k for k, (x, p) in enumerate(cand) if x == ta and (ta != 6 or p == tc)]
        if not tr: continue
        true = tr[0]; acts = cand + [(Q, (-1, -1)), (Q, (-1, -1))]; K = len(acts)
        g = torch.tensor(w["g"], device=dev).long()[None].expand(K, -1, -1, -1)
        a = torch.tensor(w["a"], device=dev)[None].repeat(K, 1); c = torch.tensor(w["c"], device=dev)[None].repeat(K, 1, 1)
        a[:, -1] = torch.tensor([x for x, _ in acts], device=dev); c[:, -1] = torch.tensor([p for _, p in acts], device=dev)
        lc = torch.tensor(w["lc"], device=dev)[None].repeat(K, 1); go = torch.tensor(w["go"], device=dev)[None].repeat(K, 1)
        rtg = torch.tensor(w["rtg"], device=dev)[None].repeat(K, 1); rtg[K - 1, -1] = 0           # dernier rêve : intention « rien »
        _, zp, zt, L0 = m.run(g, a, c, snd(m, g, lc, go, rtg, heard)); y = zp[:, -1].float(); tgt = zt[0, -1].float(); now = L0[0, -1].float()
        yc, yw, yl = y[:K - 2], y[K - 2], y[K - 1]
        d_w = ((yc - yw) ** 2).mean((1, 2)); d_l = ((yc - yl) ** 2).mean((1, 2))
        S = dict(dist=-d_w, contraste=d_l - d_w, direction=torch.nn.functional.cosine_similarity((yc - now).flatten(1), (yw - now).flatten()[None], dim=1))
        for k, v in S.items():
            ok = int(v.argmax()) == true; R[k] += ok
            if ta == 6: R[k + "_clk"] += ok
        R["chance"] += 1 / (K - 2); R["nclk"] += ta == 6
        R["e_true"] += float(((y[true] - tgt) ** 2).mean()); R["e_copy"] += float(((now - tgt) ** 2).mean())
        _, zs, _, _ = m.run(g[:1], a[true:true + 1], c[true:true + 1], snd(m, g[:1], lc[:1], go[:1], rtg[:1], 0))   # même geste, intention seule (SOURD)
        R["e_sourd"] += float(((zs[0, -1].float() - tgt) ** 2).mean())                # même geste, SANS rien entendre
    N = len(idx); m.train()
    return {k: (v / N if k != "nclk" and not k.endswith("_clk") else v) for k, v in R.items()} | dict(n=N)


def show(tag, R):
    if R is None: print(f"  {tag:28s} | (aucune fenêtre)", flush=True); return
    nc = max(R["nclk"], 1); f = lambda k: f"{R[k]:.0%}" + (f" (clics {R[k + '_clk'] / nc:.0%})" if R["nclk"] else "")
    print(f"  {tag:28s} | geste gagnant retrouvé — dist {f('dist')} | CONTRASTE {f('contraste')} | DIRECTION {f('direction')} | hasard {R['chance']:.0%} ; "
          f"{R['nclk']} clics / {R['n']} | erreur de prédiction (vrai geste) : en ENTENDANT {R['e_true']:.4f} / sourd {R['e_sourd']:.4f} / copie {R['e_copy']:.4f}", flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--ckpt", required=True); p.add_argument("--data", required=True); p.add_argument("--val", default="")
    p.add_argument("--train_fams", default="keyattr4,lines4,merge4"); p.add_argument("--test_fams", default="ghost4,cover4,rotor4")
    p.add_argument("--n_games", type=int, default=10); p.add_argument("--steps", type=int, default=3000); p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--new_lr", type=float, default=1e-2); p.add_argument("--bs", type=int, default=32); p.add_argument("--tok", type=float, default=0.5)
    p.add_argument("--every", type=int, default=1000); p.add_argument("--n_eval", type=int, default=200); p.add_argument("--heard", type=int, default=1)
    p.add_argument("--save", default=""); a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0); rng = np.random.RandomState(0); t0 = time.time()
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False); T = ck["T"]
    m = SeqWorldModel(ck["d"], T, ck["L"], ck.get("tgt_local", False), ck.get("head_src", "zp"), ck.get("head", "px"), crit=ck.get("crit", False), actor=ck.get("actor", False)).to(dev)
    m.load_state_dict(ck["m"], strict=False); m.pred.aemb = TokEmb(m.pred.aemb).to(dev)
    TF, XF = a.train_fams.split(","), a.test_fams.split(",")
    tr = episodes([a.data], TF, a.n_games, T)
    S = {"VUES, jeux d'entraînement": tr, "VUES, jeux INÉDITS": episodes([a.val], TF, 999, T, 1),
         "familles INÉDITES": episodes([a.val], XF, 999, T, 2) + episodes([a.data], XF, 10, T, 3)}
    print(f"{'OUÏE + intention' if a.heard else 'TÉMOIN SOURD (intention seule)'} | entraînement {len(tr)} fenêtres ; intention « succès proche » {np.mean([w['rtg'].mean() for w in tr]):.1%} des pas", flush=True)
    for k, W in S.items(): print(f"  « {k} » : {len(W)} fenêtres ({sum(w['lab'] == 'win' for w in W)} avec succès imminent)", flush=True)
    print(f"AVANT réglage — {time.time() - t0:.0f}s", flush=True)
    for k, W in S.items(): show(k, evaluate(m, W, dev, a.heard, a.n_eval))
    new = [m.pred.aemb.tok, m.pred.s_chg, m.pred.s_lc, m.pred.s_go, m.pred.s_rtg]; ids = {id(q) for q in new}
    ps = [q for n_, q in m.named_parameters() if not n_.startswith("tgt") and id(q) not in ids]
    opt = torch.optim.AdamW([dict(params=ps, lr=a.lr), dict(params=new, lr=a.new_lr, weight_decay=0.0)]); m.train()
    for it in range(1, a.steps + 1):
        g, ac, c, e, lc, go, rtg = batch(tr, rng.randint(0, len(tr), a.bs), dev, a.tok, rng)
        l, st = m.loss(g, ac, c, e, snd=snd(m, g, lc, go, rtg, a.heard)); opt.zero_grad(); l.backward(); opt.step(); m.ema()
        if it % a.every == 0 or it == a.steps:
            r = m.pred.s_rtg.detach(); print(f"pas {it} | perte {float(l):.3f} jepa {st['jepa']:.4f} | écart intention {float((r[1] - r[0]).norm()):.2f} | "
                                             f"son : bouge {float(m.pred.s_chg.norm()):.2f} ding {float(m.pred.s_lc.norm()):.2f} buzz {float(m.pred.s_go.norm()):.2f} — {time.time() - t0:.0f}s", flush=True)
            for k, W in S.items(): show(k, evaluate(m, W, dev, a.heard, a.n_eval))
    if a.save: torch.save(dict(ck, m=m.state_dict(), dream=True, sound=True), a.save); print("sauvé :", a.save, flush=True)
    print("FINI", flush=True)


if __name__ == "__main__":
    main()
