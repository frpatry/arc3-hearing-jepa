"""Entraîne le MODÈLE DU MONDE VIDÉO (model_seq.py) : séquences de T+1 images consécutives d'une même partie + T actions.
    python train_seq.py --data <dirs> --heldout ls20,vc33,sp80,lf52,tr87 --val2 rval --steps 60000
Mesure clé de la MÉMOIRE : l'erreur au DÉBUT de la séquence (pas 1–4 : il ne connaît pas encore le jeu) vs à la FIN
(pas 13–16 : il a vu 12 pas de CE jeu). Si la fin est nettement meilleure, il apprend la règle en contexte."""
import argparse, glob, os, time
import numpy as np, torch, torch.nn.functional as F
from model import EV_PROG, NA
from model_seq import SeqWorldModel
from model import NE


def load(dirs, names, want, T, per_family=0):
    out = []; fam = {}                                                       # jeux GÉNÉRÉS gardés par famille (28 oct., user : moins de jeux, toutes les familles)
    for d in dirs:
        kind = "offi" if "offi" in d.lower() else ("pub" if "public" in d.lower() else "gen")
        for f in sorted(glob.glob(os.path.join(d, "**", "*.npz"), recursive=True)):
            if (os.path.basename(f)[:4] in names) != want: continue
            z = np.load(f, allow_pickle=True); o, n = z["obs"], z["next_obs"]; N = len(o)
            if "game" in z.files and "game_ids" in z.files:
                ids = np.array([str(x) for x in z["game_ids"]]); gname = ids[z["game"].astype(np.int64)]
            else: gname = np.full(N, os.path.basename(f)[:4])
            cont = np.concatenate([[False], (o[1:] == n[:-1]).all((1, 2))]); run = np.zeros(N, np.int64)
            for i in range(1, N): run[i] = run[i - 1] + 1 if cont[i] else 0
            ends = np.nonzero(run >= T - 1)[0]                              # fenêtres de T transitions consécutives finissant en i
            if per_family and kind == "gen":                                # plafond : per_family jeux par famille (« arm4-o10000 » -> « arm4 »)
                ok = {}
                for gn in np.unique(gname):
                    fs = fam.setdefault(gn.split("-")[0], set())
                    if gn in fs or len(fs) < per_family: fs.add(gn); ok[gn] = True
                ends = ends[np.array([gname[i] in ok for i in ends], bool)] if len(ends) else ends
            if not len(ends): continue
            ev = z["events"].astype(np.float32) if "events" in z.files else np.full((N, NE), np.nan, np.float32)
            # « il se passe quelque chose » à ce pas : contact / son / progrès / niveau (jeux générés) ; sinon > 4 px changent
            if "events" in z.files: inter = (np.nan_to_num(ev[:, :22]) > 0.5).any(1) & ~(np.nan_to_num(ev[:, 21]) > 0.5) | (np.abs(np.nan_to_num(ev[:, 22])) > 0)
            else: inter = (o != n).reshape(N, -1).sum(1) > 4
            st = z["state"].astype(np.int64) if "state" in z.files else np.zeros(N, np.int64)   # 1 = cette action TERMINE le niveau, 2 = game over
            if "behav" in z.files:                                          # 0 = l'ORACLE joue (geste d'expert), 1 = exploration — conventions des fichiers :
                names = [str(x) for x in z["behav_names"]] if "behav_names" in z.files else ["explorer", "random", "oracle"]   # (anciens : 0 explo, 1 hasard, 2 oracle)
                bh = np.where(z["behav"].astype(np.int64) == (names.index("oracle") if "oracle" in names else -9), 0, 1)
            else: bh = np.ones(N, np.int64)
            succ = (st == 1) | ((np.nan_to_num(ev[:, 17]) > 0.5) if ev.shape[1] > 17 else False)   # succès = fin de niveau (state 1 OU son level_complete)
            rtg = np.zeros(N, np.int64); dv = np.zeros(N, np.float32); nxt = N                                         # intention : un succès dans les 8 prochains pas (même séquence continue)
            for i in range(N - 1, -1, -1):
                if i + 1 < N and not cont[i + 1]: nxt = N                            # i+1 commence une autre séquence : ses succès ne comptent pas pour i
                if succ[i]: nxt = i
                rtg[i] = int(nxt - i < 8); dv[i] = 0.97 ** (nxt - i) if nxt < N else 0.0     # DISTANCE AU BUT : γ^(pas restants) ; 0 = pas de fin de niveau devant
            # NIVEAU DE CHALEUR (son continu, 28 oct.) : progrès CUMULÉ depuis le début du niveau (jeux générés : distance exacte) ; remis à 0 à chaque
            # nouvelle séquence / nouveau niveau ; tanh(·/8) -> -1..1 ; NaN sans étiquette de progrès (démos officielles)
            pr = ev[:, 22] if ev.shape[1] > 22 else np.full(N, np.nan, np.float32); lvl = z["level"].astype(np.int64) if "level" in z.files else np.zeros(N, np.int64)
            hl = np.full(N, np.nan, np.float32); acc = 0.0
            for i in range(N):
                if i == 0 or not cont[i] or lvl[i] != lvl[i - 1]: acc = 0.0
                if np.isnan(pr[i]): continue
                acc += float(pr[i]); hl[i] = np.tanh(acc / 8.0)
            out.append((np.concatenate([o, n]), z["act"].astype(np.int64), z["click"].astype(np.int64), ev, ends, inter.astype(np.float32), st, bh, rtg, dv, hl, kind, gname))   # images : obs (N) puis next (N)
    return out or None


class Data:
    def __init__(s, raw, T, ev_w=0.0, shares=None):
        s.T = T; s.parts = []; s.ev_w = ev_w; s.groups = None
        for o, a, c, e, ends, inter, *rest in raw:
            N = len(a); st = rest[0] if rest else np.zeros(N, np.int64); bh = rest[1] if len(rest) > 1 else np.ones(N, np.int64); cs = np.concatenate([[0.0], np.cumsum(inter)])
            rg = rest[2] if len(rest) > 2 else np.zeros(N, np.int64); dv = rest[3] if len(rest) > 3 else np.zeros(N, np.float32)
            hl = rest[4] if len(rest) > 4 else np.full(N, np.nan, np.float32); kind = rest[5] if len(rest) > 5 else "gen"; gname = rest[6] if len(rest) > 6 else np.full(N, "?")
            ww = 1.0 + s.ev_w * (cs[ends + 1] - cs[np.maximum(ends - T + 1, 0)])          # poids d'une fenêtre = 1 + λ × nb de pas « intéressants »
            s.parts.append(dict(o=torch.from_numpy(o.astype(np.uint8)), a=torch.from_numpy(a), c=torch.from_numpy(c), e=torch.from_numpy(e), s=torch.from_numpy(st), b=torch.from_numpy(bh), r=torch.from_numpy(rg), v=torch.from_numpy(dv), N=N,
                                ends=torch.from_numpy(ends.astype(np.int64)), ww=ww / ww.sum(), W=ww.sum(), hl=torch.from_numpy(hl), kind=kind, gend=gname[ends]))
        s.w = np.array([p["W"] for p in s.parts], np.float64); s.n = int(sum(len(p["ends"]) for p in s.parts)); s.w /= max(s.w.sum(), 1e-9)
        if shares:                                                           # RÉÉQUILIBRAGE (28 oct., user) : chaque JEU pèse autant dans sa sorte ; sortes en parts fixes
            G = []
            for pi, p in enumerate(s.parts):
                for gn in np.unique(p["gend"]):
                    pos = np.nonzero(p["gend"] == gn)[0]; pr = p["ww"][pos]; G.append((pi, pos, pr / pr.sum(), p["kind"], gn))
            key = lambda g: (g[3], str(g[4]) if g[3] == "gen" else str(g[4])[:4])   # vrai JEU : démos officielles = 1 fichier par démo -> regrouper par code du jeu
            cnt, nw = {}, {}
            for g in G: nw[key(g)] = nw.get(key(g), 0) + len(g[1])
            for k in nw: cnt[k[0]] = cnt.get(k[0], 0) + 1
            tot = sum(shares.get(k, 0) for k in cnt) or 1.0
            s.groups = G; s.gw = np.array([shares.get(g[3], 0) / tot / cnt[g[3]] * len(g[1]) / nw[key(g)] for g in G]); s.gw /= s.gw.sum()
            print("groupes (jeux) :", {k: (v, round(shares.get(k, 0) / tot, 2)) for k, v in cnt.items()}, "| fichiers", len(G), flush=True)
    def sample(s, B, dev, rng):
        G, A, C, E, S, Bh, Rg, V, HL = [], [], [], [], [], [], [], [], []
        picks = []
        if s.groups is not None:
            for gi in rng.choice(len(s.groups), size=B, p=s.gw):
                pi, pos, pr, _, _ = s.groups[gi]; picks.append((pi, int(pos[rng.choice(len(pos), p=pr)])))
        else:
            for pi in rng.choice(len(s.parts), size=B, p=s.w): picks.append((pi, rng.choice(len(s.parts[pi]["ends"]), p=s.parts[pi]["ww"])))
        for pi, j in picks:
            p = s.parts[pi]; i = int(p["ends"][j]); idx = torch.arange(i - s.T + 1, i + 1); HL.append(p["hl"][idx])
            G.append(torch.cat([p["o"][idx], p["o"][p["N"] + i][None]])); A.append(p["a"][idx]); C.append(p["c"][idx]); E.append(p["e"][idx]); S.append(p["s"][idx]); Bh.append(p["b"][idx]); Rg.append(p["r"][idx]); V.append(p["v"][idx])
        s.last_HL = torch.stack(HL).to(dev)
        s.last_S = torch.stack(S).to(dev); s.last_B = torch.stack(Bh).to(dev); s.last_R = torch.stack(Rg).to(dev); s.last_V = torch.stack(V).to(dev)          # états (fin de niveau / game over) et qui joue (oracle ?) dans ce lot
        return torch.stack(G).to(dev).long(), torch.stack(A).to(dev), torch.stack(C).to(dev), torch.stack(E).to(dev)


def hear(m, D, e):                                    # modèle qui ENTEND (--hear 1) : ding / buzz du lot qu'on vient de tirer
    if getattr(m, "hear", False): m.cur = m.heard(D.last_S, e)


@torch.no_grad()
def eval_rollout(m, D, dev, n=256, seed=1, H=(1, 2, 4, 8, 12)):
    """IMAGINATION : vraies images jusqu'à t0 = T-1-max(H), puis le modèle enchaîne ses prédictions ; JEPA/copie à chaque horizon
    (copie = « rien ne change depuis la dernière vraie image »), sur toutes les cases et sur les cases qui ont changé."""
    T = D.T; H = tuple(h for h in H if h <= T - 2)                                   # au moins 1 vraie image de contexte
    m.eval(); rng = np.random.default_rng(seed); t0 = T - 1 - max(H); R = {h: [0.0, 0.0, 0.0, 0.0] for h in H}
    for _ in range(max(1, n // 32)):
        g, a, c, e = D.sample(32, dev, rng); B = len(g); hear(m, D, e)
        z = m.eye(g[:, :T].reshape(B * T, 64, 64)).view(B, T, 64, -1); L = m.lnt(g.reshape(B * (T + 1), 64, 64)).view(B, T + 1, 64, -1)
        zp = m.rollout(z, L, a, c, t0, max(H))
        for h in H:
            p_, t_, c_ = zp[:, t0 + h - 1], L[:, t0 + h], L[:, t0]
            ej, ec = F.smooth_l1_loss(p_, t_, reduction="none").mean(-1), F.smooth_l1_loss(c_, t_, reduction="none").mean(-1)
            mv = (g[:, t0 + h] != g[:, t0]).view(B, 8, 8, 8, 8).any(2).any(3).reshape(B, 64)
            R[h][0] += float(ej.sum()); R[h][1] += float(ec.sum()); R[h][2] += float(ej[mv].sum()); R[h][3] += float(ec[mv].sum())
    m.train(); return {h: (r[0] / max(r[1], 1e-9), r[2] / max(r[3], 1e-9)) for h, r in R.items()}


@torch.no_grad()
def eval_actor(m, D, dev, n=1024, seed=3):
    """ACTEUR : sur les pas où l'ORACLE joue (jeux inédits) — l'action de l'expert est-elle le 1er choix ? (et pour un clic : la bonne case 8×8)"""
    m.eval(); rng = np.random.default_rng(seed); ok = tot = okc = totc = 0; cnt = np.zeros(8)
    for _ in range(max(1, n // 32)):
        g, a, c, e = D.sample(32, dev, rng); Bh = D.last_B; hear(m, D, e)
        z, zp, _, _ = m.run(g, a, c); hp = torch.cat([torch.zeros_like(m.pred.hid[:, :1]), m.pred.hid[:, :-1]], 1)
        la, lt, lf = m.actor(z, hp); mk = (Bh == 0) & (a > 0)
        if not mk.any(): continue
        ok += int((la.argmax(-1) == a)[mk].sum()); tot += int(mk.sum()); cnt += np.bincount(a[mk].cpu().numpy(), minlength=8)[:8]
        mc = mk & (a == 6) & (c[..., 0] >= 0)
        if mc.any():
            cell = (c[..., 1].clamp(0, 63) // 8) * 8 + c[..., 0].clamp(0, 63) // 8; okc += int((lt.argmax(-1) == cell)[mc].sum()); totc += int(mc.sum())
    m.train(); return ok / max(tot, 1), tot, okc / max(totc, 1), totc, cnt.max() / max(cnt.sum(), 1)


def auc(p, y):                                         # aire sous la courbe ROC (rangs) ; 0.5 = hasard
    p, y = np.asarray(p, np.float64), np.asarray(y).astype(bool)
    if y.all() or not y.any(): return float("nan")
    r = np.empty(len(p)); r[np.argsort(p)] = np.arange(1, len(p) + 1)
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


@torch.no_grad()
def eval_crit(m, D, dev, n=1024, seed=2):
    """CRITIQUE : AUC de « cette action termine le niveau » et « niveau terminé dans les 8 prochains pas » (+ nb de positifs)."""
    m.eval(); rng = np.random.default_rng(seed); P1, P2, Y1, Y2 = [], [], [], []
    for _ in range(max(1, n // 32)):
        g, a, c, e = D.sample(32, dev, rng); S = D.last_S; hear(m, D, e)
        z, zp, _, _ = m.run(g, a, c); lo = m.critic()
        y1, y2 = m.crit_targets(S, e); P1.append(lo[..., 0].float().flatten().cpu()); P2.append(lo[..., 1].float().flatten().cpu())
        Y1.append(y1.flatten().cpu()); Y2.append(y2.flatten().cpu())
    m.train(); y1, y2 = torch.cat(Y1).numpy(), torch.cat(Y2).numpy()
    return auc(torch.cat(P1).numpy(), y1), auc(torch.cat(P2).numpy(), y2), int(y1.sum()), int(y2.sum())


@torch.no_grad()
def evaluate(m, D, dev, n=256, seed=0):
    m.eval(); rng = np.random.default_rng(seed); T = D.T; J = torch.zeros(T); Jc = torch.zeros(T); TP = torch.zeros(T); FP = torch.zeros(T); FN = torch.zeros(T)
    acc = base = cnt = 0; Jm = Jcm = nm = 0.0; cok = cn = 0.0   # erreur sur les CASES QUI CHANGENT ; couleur juste sur les pixels qui changent
    for _ in range(max(1, n // 32)):
        g, a, c, e = D.sample(32, dev, rng); B = len(g); hear(m, D, e)
        z, zp, zt, zc = m.run(g, a, c)
        ej, ec = F.smooth_l1_loss(zp, zt, reduction="none").mean(-1), F.smooth_l1_loss(zc, zt, reduction="none").mean(-1)          # (B,T,64)
        J += ej.mean((0, 2)).cpu(); Jc += ec.mean((0, 2)).cpu()
        mv = (g[:, 1:] != g[:, :T]).view(B, T, 8, 8, 8, 8).any(3).any(4).view(B, T, 64); Jm += float(ej[mv].sum()); Jcm += float(ec[mv].sum()); nm += float(mv.sum())
        g0, g1 = g[:, :T].reshape(B * T, 64, 64), g[:, 1:].reshape(B * T, 64, 64)
        ch, _, _ = m.heads(z.reshape(B * T, 64, -1), m.hin(zp).reshape(B * T, 64, -1), g0); pc = (ch > 0).view(B, T, 64, 64); real = (g1 != g0).view(B, T, 64, 64)
        TP += (pc & real).sum((0, 2, 3)).cpu(); FP += (pc & ~real).sum((0, 2, 3)).cpu(); FN += (~pc & real).sum((0, 2, 3)).cpu()
        if getattr(m, "head", "px") == "col":
            cc = (m.last_col.argmax(1).view(B, T, 64, 64) == g[:, 1:]); cok += float((cc & real).sum()); cn += float(real.sum())
        pe, re_ = pc.sum((2, 3)) > 4, real.sum((2, 3)) > 4; acc += int((pe == re_).sum()); base += int(re_.sum()); cnt += pe.numel()
    m.train(); k = max(1, n // 32); f1 = lambda s_: float(2 * TP[s_].sum() / (2 * TP[s_].sum() + FP[s_].sum() + FN[s_].sum()).clamp_min(1))
    q = T // 4
    return dict(jepa=float(J.mean() / k), copy=float(Jc.mean() / k), f1=f1(slice(0, T)), f1_debut=f1(slice(0, q)), f1_fin=f1(slice(T - q, T)),
                f1_16=f1(slice(12, 16)) if T > 16 else None,                       # pas 13–16 : la « fin » du modèle T=16, pour comparer
                j_debut=float(J[:q].mean() / k), j_fin=float(J[T - q:].mean() / k), c_debut=float(Jc[:q].mean() / k), c_fin=float(Jc[T - q:].mean() / k),
                eff=acc / max(cnt, 1), base=max(base, cnt - base) / max(cnt, 1), jm=Jm / max(nm, 1), jcm=Jcm / max(nm, 1), col=cok / cn if cn else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True); p.add_argument("--heldout", default="ls20,vc33,sp80,lf52,tr87"); p.add_argument("--val2", default="rval")
    p.add_argument("--val3", default="", help="3e jeu de validation (ex. dval : vidéos parent-montre-bébé-joue inédites)")
    p.add_argument("--T", type=int, default=16); p.add_argument("--steps", type=int, default=60000); p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--d", type=int, default=256); p.add_argument("--L", type=int, default=6)
    p.add_argument("--eval_every", type=int, default=2000); p.add_argument("--eval_n", type=int, default=256); p.add_argument("--max_minutes", type=float, default=0)
    p.add_argument("--w_ev", type=float, default=1.0); p.add_argument("--w_snd", type=float, default=0.5, help="poids du « son spatial » (0 = ablation sans son)")
    p.add_argument("--w_roll", type=float, default=0.0, help="poids de la perte d'IMAGINATION multi-pas (0 = aucune)")
    p.add_argument("--roll_k", type=int, default=4, help="nb de pas imaginés à l'entraînement")
    p.add_argument("--w_crit", type=float, default=0.0, help="poids de la CRITIQUE (fin de niveau imminente) ; 0 = sans")
    p.add_argument("--w_crit_im", type=float, default=0.0, help="poids de la CRITIQUE SUR ÉTATS IMAGINÉS (ce que lit le planificateur) ; 0 = sans")
    p.add_argument("--w_prog_im", type=float, default=0.0, help="poids de la CHALEUR (progrès) SUR ÉTATS IMAGINÉS (27 oct.) ; 0 = sans")
    p.add_argument("--hear_heat", type=float, default=0.0, help="SON DE CHALEUR (28 oct.) : part des séquences où le modèle ENTEND la vraie chaleur de chaque geste (le reste : silence) ; 0 = sans")
    p.add_argument("--per_family", type=int, default=0, help="jeux GÉNÉRÉS gardés par famille (0 = tous) ; 28 oct. : moins de jeux, toutes les familles")
    p.add_argument("--shares", default="", help="ex. gen:0.8,offi:0.15,pub:0.05 : part de chaque sorte de données, chaque JEU pesant autant dans sa sorte")
    p.add_argument("--w_hl", type=float, default=0.0, help="poids de la cible NIVEAU DE CHALEUR (progrès cumulé depuis le début du niveau)")
    p.add_argument("--voice", type=int, default=0, help="1 = VOIX INTÉRIEURE : le modèle entend TOUJOURS son propre niveau de chaleur prédit au pas précédent")
    p.add_argument("--warm", type=float, default=0.03, help="part du calendrier en montée du taux d'apprentissage (OneCycle pct_start) ; 28 oct. : 0.05 pour éviter le choc")
    p.add_argument("--w_act", type=float, default=0.0, help="poids de l'ACTEUR (imiter l'oracle, Mode 1) ; 0 = sans tête acteur")
    p.add_argument("--head", default="px", choices=["px", "obj", "col"], help="obj = carte de changement NETTE (pixel + objets) ; col = couleur de chaque pixel suivant (U-Net)")
    p.add_argument("--train_only", default="", help="ex. heads : n'entraîner que les têtes (tronc du modèle gelé)")
    p.add_argument("--resume", default="", help="REPRENDRE un entraînement interrompu (modèle + optimiseur + calendrier + pas) si ce fichier existe")
    p.add_argument("--init", default="", help="repartir d'un modèle (.pt) au lieu de zéro")
    p.add_argument("--w_jepa", type=float, default=1.0, help="poids de la perte JEPA (elle est ~50× plus petite que celle des têtes)")
    p.add_argument("--head_src", default="zp", choices=["zp", "hid"], help="hid = les têtes lisent l'état interne du prédicteur, pas la sortie JEPA")
    p.add_argument("--tgt_local", type=int, default=0, help="1 = œil-cible LOCAL (chaque case 8×8 seule) : une case immobile garde son latent")
    p.add_argument("--ev_w", type=float, default=0.0, help="tirage des fenêtres d'entraînement ∝ 1 + ev_w × (pas où il se passe quelque chose)"); p.add_argument("--out", default="wm_seq.pt")
    p.add_argument("--dream", type=float, default=0.0, help="RÊVE : proba de cacher l'action (jeton « ? ») à partir d'un pas tiré, + intention (succès dans 8 pas ?) donnée à ces pas")
    p.add_argument("--hear", type=int, default=0, help="1 = le modèle ENTEND (entrée) : zones qui ont bougé, ding de fin de niveau, buzz de game over (idée user, 26 oct.)")
    a = p.parse_args(); dev = "cuda" if torch.cuda.is_available() else ("mps" if os.environ.get("WM_MPS") and torch.backends.mps.is_available() else "cpu"); torch.backends.cuda.matmul.allow_tf32 = True
    held, v2, v3 = (set(x.split(",")) - {""} for x in (a.heldout, a.val2, a.val3)); dirs = a.data.split(",")
    shares = {k: float(v) for k, v in (x.split(":") for x in a.shares.split(",") if x)} or None
    TR = Data(load(dirs, held | v2 | v3, False, a.T, a.per_family), a.T, a.ev_w, shares); VA = load(dirs, held, True, a.T); VB = load(dirs, v2, True, a.T) if v2 else None
    VC = load(dirs, v3, True, a.T) if v3 else None
    VA = Data(VA, a.T) if VA else None; VB = Data(VB, a.T) if VB else None; VC = Data(VC, a.T) if VC else None
    print(f"séquences de {a.T} pas : entraînement {TR.n} | officiels mis de côté {VA.n if VA else 0} | règles inédites {VB.n if VB else 0} | démos inédites {VC.n if VC else 0} | {dev}", flush=True)
    m = SeqWorldModel(a.d, a.T, a.L, bool(a.tgt_local), a.head_src, a.head, crit=a.w_crit > 0 or bool(a.init and torch.load(a.init, map_location="cpu", weights_only=False).get("crit")),
                      actor=a.w_act > 0 or bool(a.init and torch.load(a.init, map_location="cpu", weights_only=False).get("actor")),
                      dream=a.dream > 0 or bool(a.init and torch.load(a.init, map_location="cpu", weights_only=False).get("dream"))).to(dev)
    m.hear = bool(a.hear); m.voice = bool(a.voice)                                                            # le modèle ENTEND (canal son en entrée)
    if a.init:
        sd = torch.load(a.init, map_location=dev, weights_only=False)["m"]
        te = sd.get("pred.temb")
        if te is not None and te.size(1) != a.T:                                    # T PLUS LONG (27 oct. : 24 images = 12 réelles + 12 imaginées) :
            T0 = te.size(1); ext = te[:, -1:].expand(-1, max(a.T - T0, 0), -1, -1)   # positions connues gardées, les nouvelles = la dernière
            sd["pred.temb"] = torch.cat([te, ext], 1)[:, :a.T]; print(f"temps : {T0} -> {a.T} positions", flush=True)
        sd = {k_: v_ for k_, v_ in sd.items() if not k_.endswith(".mask")}           # masque bloc-causal : recalculé pour le nouveau T
        miss = m.load_state_dict(sd, strict=False)
        print(f"repart de {a.init} (manquants : {miss.missing_keys})", flush=True)
    if a.train_only == "heads":                                                      # tronc gelé : seules les têtes apprennent
        for n_, q in m.named_parameters():
            if not n_.startswith(("up.", "chg.", "obj.", "col.", "ev.", "snd.", "crit.", "act_")): q.requires_grad_(False)
    if a.train_only == "noeye":                                                     # tout apprend SAUF l'œil (perception gardée intacte)
        for n_, q in m.named_parameters():
            if n_.startswith(("eye.", "tgt.")): q.requires_grad_(False)
        print(f"œil gelé : {sum(q.numel() for q in m.parameters() if q.requires_grad) / 1e6:.2f} M paramètres entraînés", flush=True)
    if a.train_only in ("actor", "crit", "crit+proj"):                               # SEULE la tête acteur (ou critique) apprend : tout le reste gelé
        keep = {"actor": ("act_",), "crit": ("crit.",), "crit+proj": ("crit.", "crit_d.", "proj.")}[a.train_only]   # crit+proj (27 oct.) : la nuit avait w_roll 0 -> proj jamais appris
        for n_, q in m.named_parameters():                                            # (leçon 25 oct. : « heads » abîmait les têtes pixels)
            if not n_.startswith(keep): q.requires_grad_(False)
        print(f"tronc gelé : {sum(q.numel() for q in m.parameters() if q.requires_grad) / 1e6:.2f} M paramètres entraînés", flush=True)
    opt = torch.optim.AdamW([q for q in m.parameters() if q.requires_grad], a.lr, weight_decay=0.02)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=a.warm)
    print(f"paramètres {sum(q.numel() for q in m.parameters() if q.requires_grad) / 1e6:.1f} M", flush=True)
    bf16 = dev == "cuda" and torch.cuda.get_device_capability()[0] >= 8                # T4 / P100 : pas de bf16 natif -> fp32
    print(f"calcul : {'bf16' if bf16 else 'fp32'}", flush=True)
    it0 = 0
    if a.resume and os.path.exists(a.resume):                                       # reprise (ex. session Colab coupée)
        r_ = torch.load(a.resume, map_location=dev, weights_only=False); m.load_state_dict(r_["m"], strict=False)
        if "opt" in r_: opt.load_state_dict(r_["opt"]); sch.load_state_dict(r_["sch"])
        it0 = int(r_.get("it", 0)); print(f"REPRISE au pas {it0} depuis {a.resume}", flush=True)
        if "sch" in r_ and r_["sch"].get("total_steps") != a.steps:                 # calendrier ALLONGÉ (ex. 320k -> 600k) : même cycle, nouvelle longueur
            sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=a.warm, last_epoch=it0 - 1)
            print(f"calendrier allongé à {a.steps} pas (lr actuel {sch.get_last_lr()[0]:.2e})", flush=True)
    rng = np.random.default_rng(it0); t0 = time.time(); hist = []
    for it in range(it0 + 1, a.steps + 1):
        g, ac, c, e = TR.sample(a.bs, dev, rng); hear(m, TR, e)
        if rng.random() < 0.5: perm = torch.randperm(16, device=dev); g = perm[g]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            snd = None; ac_in, c_in = ac, c
            if a.dream > 0 or a.hear:                                                    # SON entendu + RÊVE (action « ? » + intention)
                B_, T_ = ac.shape; rin = torch.full_like(ac, -1)
                if a.dream > 0:
                    ac_in, c_in = ac.clone(), c.clone()
                    for b_ in np.nonzero(rng.random(B_) < a.dream)[0]:
                        tm = int(rng.integers(0, T_)); ac_in[b_, tm:] = NA; c_in[b_, tm:] = -1; rin[b_, tm:] = TR.last_R[b_, tm:]
                lc_, go_ = m.heard(TR.last_S, e) if a.hear else (torch.zeros(B_, T_, device=dev),) * 2
                hh = None
                if a.voice:                                                              # VOIX INTÉRIEURE : sa propre prédiction de niveau de chaleur, entendue au pas suivant
                    m.cur = m.heard(TR.last_S, e) if a.hear else None
                    hp0 = m.heat_level(g, ac_in, c_in); hh = torch.cat([torch.zeros_like(hp0[:, :1]), hp0[:, :-1]], 1)
                elif a.hear_heat > 0 and e is not None:                                    # SON DE CHALEUR : progrès de la transition t-1 -> t, entendu au pas t ;
                    pr = torch.nan_to_num(e[..., EV_PROG].float(), 0.0)                      # silence pour une part des séquences (en jeu : sa propre chaleur prédite)
                    hh = torch.cat([torch.zeros_like(pr[:, :1]), pr[:, :-1]], 1) * (torch.rand(B_, 1, device=dev) < a.hear_heat).float()
                snd = m.sound(g, lc_, go_, rin, hh) if a.hear else m.pred.s_rtg[rin.clamp(min=0)][:, :, None] * (rin >= 0).float()[..., None, None]
                snd_rtg = m.pred.s_rtg[rin.clamp(min=0)][:, :, None] * (rin >= 0).float()[..., None, None]   # l'intention seule (gardée dans les pas imaginés)
            l, st = m.loss(g, ac_in, c_in, e, a.w_ev, a.w_jepa, a.w_roll, a.roll_k, S=TR.last_S, w_crit=a.w_crit, w_snd=a.w_snd, Bh=TR.last_B, w_act=a.w_act, snd=snd, a_true=ac, c_true=c, w_crit_im=a.w_crit_im, w_prog_im=a.w_prog_im, HL=TR.last_HL, w_hl=a.w_hl, V=TR.last_V, snd_rtg=snd_rtg if a.dream > 0 else None)
        opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step(); m.ema(); hist.append(st)
        late = a.max_minutes > 0 and time.time() - t0 > 60 * a.max_minutes
        if it in (1, 50) or it % a.eval_every == 0 or it == a.steps or late:
            h = {k: np.mean([x[k] for x in hist if k in x]) for k in hist[-1]}; hist = []
            fmt = lambda e_: (f"jepa {e_['jepa']:.3f} (copie {e_['copy']:.3f}) | cases qui changent jepa/copie {e_['jm'] / max(e_['jcm'], 1e-6):.2f} | f1 {e_['f1']:.2f} [DÉBUT {e_['f1_debut']:.2f} -> FIN {e_['f1_fin']:.2f}] | "
                              f"jepa/copie DÉBUT {e_['j_debut'] / max(e_['c_debut'], 1e-6):.2f} -> FIN {e_['j_fin'] / max(e_['c_fin'], 1e-6):.2f} | effet {e_['eff']:.0%} (constante {e_['base']:.0%})"
                              + (f" | f1 pas 13-16 {e_['f1_16']:.2f}" if e_.get('f1_16') is not None else "")
                              + (f" | COULEUR juste (pixels qui changent) {e_['col']:.0%}" if e_.get('col') is not None else ""))
            if a.w_roll > 0 or a.init:
                for nm_, V_ in (("OFFICIELS", VA), ("RÈGLES", VB), ("VAL3", VC)):
                    if V_: print(f"   IMAGINATION {nm_:9s} jepa/copie [toutes cases | cases qui changent] : " + " ; ".join(f"{h} pas {r[0]:.2f} | {r[1]:.2f}" for h, r in eval_rollout(m, V_, dev, a.eval_n).items()), flush=True)
            if a.w_act > 0:
                for nm_, V_ in (("RÈGLES", VB), ("VAL3", VC)):
                    if V_: r_ = eval_actor(m, V_, dev); print(f"   ACTEUR {nm_:9s} geste de l'oracle en 1er choix {r_[0]:.0%} ({r_[1]} pas ; action la plus fréquente {r_[4]:.0%}) | clic : bonne case {r_[2]:.0%} ({r_[3]} clics)", flush=True)
            if a.w_crit > 0:
                for nm_, V_ in (("OFFICIELS", VA), ("RÈGLES", VB), ("VAL3", VC)):
                    if V_: r_ = eval_crit(m, V_, dev); print(f"   CRITIQUE {nm_:9s} AUC « cette action finit le niveau » {r_[0]:.2f} ({r_[2]} cas) | « fini dans ≤ 8 pas » {r_[1]:.2f} ({r_[3]} cas)", flush=True)
            e1 = evaluate(m, VA, dev, a.eval_n) if VA else None; e2 = evaluate(m, VB, dev, a.eval_n) if VB else None; e3 = evaluate(m, VC, dev, a.eval_n) if VC else None
            print(f"pas {it} | entr. jepa {h['jepa']:.3f} chg {h['chg']:.3f}" + (f" prog {h['ev_prog']:.3f}" if 'ev_prog' in h else "") + (f" imagination {h['roll']:.3f} proj {h['proj']:.3f}" if 'roll' in h else "") + (f" critique {h['crit']:.3f}" if 'crit' in h else "") + (f" critique-imaginé {h['crit_im']:.3f}" if 'crit_im' in h else "") + (f" chaleur-imaginée {h['prog_im']:.3f}" if 'prog_im' in h else "") + (f" niveau-chaleur {h['hl']:.3f}" if 'hl' in h else "") + (f" distance {h['dist']:.3f}" if 'dist' in h else "") + (f" acteur {h['act']:.3f}" if 'act' in h else "")
                  + (f"\n   OFFICIELS INÉDITS : {fmt(e1)}" if e1 else "") + (f"\n   RÈGLES INÉDITES  : {fmt(e2)}" if e2 else "") + (f"\n   DÉMOS INÉDITES   : {fmt(e3)}" if e3 else "") + f"\n   ({time.time() - t0:.0f}s)", flush=True)
            torch.save(dict(m=m.state_dict(), d=a.d, T=a.T, L=a.L, seq=True, tgt_local=bool(a.tgt_local), head_src=a.head_src, head=a.head, crit=hasattr(m, "crit"), actor=hasattr(m, "act_a"), hear=bool(a.hear), heat_snd=a.hear_heat > 0, voice=bool(a.voice), dream=hasattr(m.pred.aemb, "tok"), intent=a.dream > 0 or bool(getattr(a, "intent", False)), args=vars(a), it=it, opt=opt.state_dict(), sch=sch.state_dict()), a.out)
            if late: print(f"ARRÊT : limite de temps au pas {it}", flush=True); break


if __name__ == "__main__":
    main()
