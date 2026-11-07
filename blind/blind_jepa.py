"""MODÈLE AVEUGLE — étape 3 : JEPA SANS ŒIL (1er nov., idée user). Entrées par pas (blind_data.py) : action + clic, sons fabriqués
(toc, contact, ramassé, couleur, téléport, ding, buzz), ≤ 12 objets hypothétiques. AUCUN pixel.
  ENCODEUR (transformeur, causal par pas, fenêtre T) : jetons [pas] + [objets] -> état latent z_t (après le pas t).
  PRÉDICTEUR : z_≤t + geste du pas t+1 -> ẑ_{t+1} ; cible = encodeur EMA (copie lente) sur le pas t+1 complet (JEPA).
  MASQUE : 25 % des pas perdent leurs sons et leurs objets (le geste reste) -> l'encodeur doit les deviner (cible EMA).
  CIBLES À DEVINER depuis ẑ_{t+1} (avant de l'entendre) : les sons fabriqués + les étiquettes du générateur (quand connues).
SONDES (modèle gelé ; ajustées sur générés inédits + officiels vus, mesurées sur les 5 OFFICIELS EXCLUS) :
  1. « j'entends d'avance » : AUC des sons prédits avant le pas ;
  2. « moi » : position de ce qui va bouger sous mon geste fléché, depuis ẑ (avant de le voir) vs recopier la dernière position ;
  3. « combien de gestes avant le ding » (parties oracle) depuis z_t : Spearman vs « temps passé dans le niveau ».
    python blind_jepa.py --data data/blind --max_minutes 60 --out blind.pt"""
import argparse, copy, glob, json, math, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from blind_data import K, NF, SND, KS, NFS
NL = 11


def load(folder):
    fs = sorted(glob.glob(os.path.join(folder, "*.npz"))); D = {}
    for i, f in enumerate(fs):
        z = np.load(f)
        for k in z.files: D.setdefault(k, []).append(z[k])
        D.setdefault("FILE", []).append(np.full(len(z["ACT"]), i, np.int32))
    D = {k: np.concatenate(v) for k, v in D.items()}
    key = D["FILE"].astype(np.int64) * 10_000_000 + D["EP"]; new = np.r_[True, key[1:] != key[:-1]]
    D["EPS"] = np.cumsum(new) - 1                                                  # identifiant global d'épisode
    st = np.flatnonzero(new); ln = np.diff(np.r_[st, len(key)]); D["REM"] = np.zeros(len(key), np.int64)
    for s, l in zip(st, ln): D["REM"][s:s + l] = np.arange(l, 0, -1)               # pas restants dans l'épisode (ce pas compris)
    D["T0"] = np.zeros(len(key), np.int64)                                         # pas depuis le début du NIVEAU (sonde 3, repère)
    lv_new = new | np.r_[True, D["LV"][1:] != D["LV"][:-1]]; c = 0
    for i in range(len(key)): c = 0 if lv_new[i] else c + 1; D["T0"][i] = c
    return D


def batch(D, starts, T, dev):
    idx = starts[:, None] + np.arange(T)[None]; V = np.arange(T)[None] < D["REM"][starts][:, None]
    idx = np.where(V, idx, starts[:, None]); t = lambda x, dt=torch.float32: torch.as_tensor(x, device=dev, dtype=dt)
    clk = D["CLK"][idx].astype(np.float32); has = (clk[..., 0] >= 0).astype(np.float32)
    return dict(OBJ=t(D["OBJ"][idx].astype(np.float32)), COL=t(D["COL"][idx], torch.long), NO=t(D["NO"][idx], torch.long),
                ACT=t(D["ACT"][idx].clip(0, 8), torch.long), CLK=t(np.concatenate([clk.clip(0, 63) / 64, has[..., None]], -1)),
                SND=t(D["SND"][idx]), LAB=t(D["LAB"][idx]), V=t(V, torch.bool),
                **(dict(SCN=t(D["SCN"][idx].astype(np.float32)), SCOL=t(D["SCOL"][idx], torch.long), NS=t(D["NS"][idx], torch.long)) if "SCN" in D else {}))


class Blind(nn.Module):
    def __init__(s, d=256, L=6, Lp=3, T=16, late=True):
        super().__init__(); s.T, s.d, s.late = T, d, late
        lay = lambda: nn.TransformerEncoderLayer(d, 8, 4 * d, 0.0, "gelu", batch_first=True, norm_first=True)
        s.cemb = nn.Embedding(17, 16); s.oin = nn.Sequential(nn.Linear(NF + 32, d), nn.GELU(), nn.Linear(d, d))
        s.aemb = nn.Embedding(9, d); s.clk = nn.Linear(3, d); s.snd = nn.Linear(len(SND), d); s.mtok = nn.Parameter(torch.zeros(d))
        s.typ = nn.Parameter(torch.randn(2, d) * 0.02); s.tem = nn.Parameter(torch.randn(T, d) * 0.02)
        s.enc = nn.TransformerEncoder(lay(), L); s.ln = nn.LayerNorm(d, elementwise_affine=False)
        s.pin = nn.Linear(d, d); s.pa = nn.Embedding(9, d); s.pclk = nn.Linear(3, d); s.ptem = nn.Parameter(torch.randn(T, d) * 0.02)
        s.pred = nn.TransformerEncoder(lay(), Lp)
        if late:                                      # geste À LA FIN du prédicteur (+ deviner le geste)
            s.pout = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
            s.inv = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 9 + 65))   # DEVINER LE GESTE (avant, après) : action + zone du clic
        else: s.pout = nn.Linear(d, d)                # ancien prédicteur (geste en ENTRÉE ; ckpt arc3-blind-long)
        s.h_snd = nn.Linear(d, len(SND)); s.h_lab = nn.Linear(d, NL)
        n = T * (1 + K); stp = torch.arange(T).repeat_interleave(1 + K)
        s.register_buffer("bmask", stp[None, :] > stp[:, None]); s.register_buffer("cmask", torch.triu(torch.ones(T - 1, T - 1, dtype=torch.bool), 1))

    def encode(s, b, masked=None):                    # -> z (B,T,d) normalisé
        B, T = b["ACT"].shape; m = torch.zeros(B, T, dtype=torch.bool, device=b["ACT"].device) if masked is None else masked
        keep = (~m).float()[..., None]
        step = s.aemb(b["ACT"]) + s.clk(b["CLK"]) + s.snd(b["SND"]) * keep + s.mtok * (1 - keep) + s.typ[0] + s.tem[:T]
        obj = s.oin(torch.cat([b["OBJ"], s.cemb(b["COL"].clamp(0, 16)).flatten(-2)], -1)) + s.typ[1] + s.tem[:T, None]
        pad = (torch.arange(K, device=obj.device)[None, None] >= b["NO"][..., None]) | m[..., None]
        x = torch.cat([step[:, :, None], obj], 2).reshape(B, T * (1 + K), -1)
        kp = torch.cat([torch.zeros(B, T, 1, dtype=torch.bool, device=x.device), pad], 2).reshape(B, -1)
        h = s.enc(x, mask=s.bmask[:T * (1 + K), :T * (1 + K)], src_key_padding_mask=kp)
        return s.ln(h.view(B, T, 1 + K, -1)[:, :, 0])

    def ctx(s, z):                                    # le passé lu par le prédicteur (sans le geste suivant) -> (B,T-1,d)
        T = z.size(1); return s.pred(s.pin(z[:, :-1]) + s.ptem[:T - 1], mask=s.cmask[:T - 1, :T - 1])

    def act_out(s, c, act, clk):                      # le geste n'entre qu'À LA FIN -> essayer d'autres gestes ne coûte presque rien
        return s.ln(s.pout(c + s.pa(act) + s.pclk(clk)))

    def predict(s, z, b):                             # ẑ_{t+1} depuis z_≤t et le geste du pas t+1 -> (B,T-1,d)
        if not s.late:
            T = z.size(1); x = s.pin(z[:, :-1]) + s.pa(b["ACT"][:, 1:]) + s.pclk(b["CLK"][:, 1:]) + s.ptem[:T - 1]
            return s.ln(s.pout(s.pred(x, mask=s.cmask[:T - 1, :T - 1])))
        return s.act_out(s.ctx(z), b["ACT"][:, 1:], b["CLK"][:, 1:])


class BlindLong(Blind):
    """MÉMOIRE COMPLÈTE (1er nov., go user) : chaque pas (geste, sons, objets) est d'abord RÉSUMÉ en un jeton (petit transformeur
    dans le pas, sans temps), puis un transformeur causal parcourt jusqu'à T = 256 pas = presque toute la partie."""
    def __init__(s, d=256, L=6, Lp=3, T=256, Ls=2, late=True, eye=False):
        super().__init__(d, L, Lp, T, late); s.eye = eye
        if eye:                                        # ŒIL MINIMAL : les objets à l'écran après le pas (≤ KS jetons de plus par pas)
            s.scemb = nn.Embedding(17, 16); s.scin = nn.Sequential(nn.Linear(NFS + 16, d), nn.GELU(), nn.Linear(d, d)); s.typ_s = nn.Parameter(torch.randn(d) * 0.02)
        lay = lambda: nn.TransformerEncoderLayer(d, 8, 4 * d, 0.0, "gelu", batch_first=True, norm_first=True)
        s.senc = nn.TransformerEncoder(lay(), Ls); s.sln = nn.LayerNorm(d)
        s.register_buffer("tmask", torch.triu(torch.ones(T, T, dtype=torch.bool), 1)); del s.bmask

    def steps(s, b, masked=None, with_action=True):  # 1 jeton par pas ; sans geste = « ce qui s'est passé » seulement (pour deviner le geste)
        B, T = b["ACT"].shape; m = torch.zeros(B, T, dtype=torch.bool, device=b["ACT"].device) if masked is None else masked
        keep = (~m).float()[..., None]
        step = (s.aemb(b["ACT"]) + s.clk(b["CLK"]) if with_action else 0) + s.snd(b["SND"]) * keep + s.mtok * (1 - keep) + s.typ[0]
        obj = s.oin(torch.cat([b["OBJ"], s.cemb(b["COL"].clamp(0, 16)).flatten(-2)], -1)) + s.typ[1]
        pad = (torch.arange(K, device=obj.device)[None, None] >= b["NO"][..., None]) | m[..., None]
        toks, pads = [step[:, :, None], obj], [torch.zeros(B, T, 1, dtype=torch.bool, device=obj.device), pad]
        if s.eye:
            toks.append(s.scin(torch.cat([b["SCN"], s.scemb(b["SCOL"].clamp(0, 16))], -1)) + s.typ_s)
            pads.append((torch.arange(KS, device=obj.device)[None, None] >= b["NS"][..., None]) | m[..., None])
        x = torch.cat(toks, 2); n = x.size(2); x = x.reshape(B * T, n, -1); kp = torch.cat(pads, 2).reshape(B * T, n)
        return s.sln(s.senc(x, src_key_padding_mask=kp)[:, 0]).view(B, T, -1)

    def encode(s, b, masked=None):
        T = b["ACT"].shape[1]; v = s.steps(b, masked)
        return s.ln(s.enc(v + s.tem[:T], mask=s.tmask[:T, :T]))


def naive(D, rows):
    """REPÈRES sans apprentissage, en ligne dans chaque épisode : (1) fréquence de chaque son pour CE geste (clic par zone 8×8) depuis
    le début de la partie, lissée par la moyenne de la partie ; (2) le son du pas d'avant."""
    P1 = np.zeros((len(rows), len(SND)), np.float32); P2 = np.zeros_like(P1); pos = {r: i for i, r in enumerate(rows)}
    for e in np.unique(D["EPS"][rows]):
        ii = np.flatnonzero(D["EPS"] == e); cnt, tot, gs, g = {}, {}, np.zeros(len(SND)), 0
        for k, t in enumerate(ii):
            key = (int(D["ACT"][t]), tuple((D["CLK"][t] // 8).tolist()) if D["ACT"][t] == 6 else None)
            n = cnt.get(key, np.zeros(len(SND))); mm = tot.get(key, 0)
            if t in pos:
                P1[pos[t]] = (n + 0.5 * (gs + 0.5) / (g + 1)) / (mm + 0.5); P2[pos[t]] = D["SND"][ii[k - 1]] if k else 0
            y = D["SND"][t]; cnt[key] = n + y; tot[key] = mm + 1; gs = gs + y; g += 1
    return P1, P2


def v_shape(b): return b["ACT"].shape[0], b["ACT"].shape[1] - 1


def cov_pen(x):
    x = x - x.mean(0); c = (x.T @ x) / max(1, len(x) - 1); d = c.size(0)
    return (c.pow(2).sum() - c.diagonal().pow(2).sum()) / d


def auc(y, p):
    y = np.asarray(y, bool); p = np.asarray(p, float)
    if y.all() or not y.any(): return float("nan")
    r = np.argsort(np.argsort(p)) + 1.0; return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])


def ridge(Xtr, ytr, Xte, lam=10.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; A = (Xtr - mu) / sd; ym = ytr.mean(0)
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (ytr - ym)); return ((Xte - mu) / sd) @ W + ym


@torch.no_grad()
def features(m, D, rows_mask, T, dev):
    """chaque pas des épisodes choisis -> z (après le pas) et ẑ (prédit AVANT le pas) ; fenêtres de T qui se chevauchent d'un pas."""
    m.eval(); eps = np.unique(D["EPS"][rows_mask]); S = []
    for e in eps:
        ii = np.flatnonzero(D["EPS"] == e); S += list(range(ii[0], ii[-1] + 1, T - 1))
    S = np.array(S); Z = np.zeros((len(D["ACT"]), m.d), np.float32); P = np.full((len(D["ACT"]), m.d), np.nan, np.float32)
    for i in range(0, len(S), 256):
        st = S[i:i + 256]; b = batch(D, st, T, dev); z = m.encode(b); zp = m.predict(z, b)
        idx = st[:, None] + np.arange(T)[None]; V = b["V"].cpu().numpy()
        Z[idx[V]] = z.float().cpu().numpy()[V]; P[idx[:, 1:][V[:, 1:]]] = zp.float().cpu().numpy()[V[:, 1:]]
    m.train(); return Z, P


@torch.no_grad()
def probes(m, D, T, dev, quick=False):
    src = D["SRC"]; sel = src >= 1
    if quick: sel &= np.isin(D["EPS"], np.unique(D["EPS"][sel])[::7]) | (src == 3)
    Z, P = features(m, D, sel, T, dev); R = {}
    hs = lambda X: torch.sigmoid(m.h_snd(torch.as_tensor(X, device=dev))).cpu().numpy()
    hl = lambda X: torch.sigmoid(m.h_lab(torch.as_tensor(X, device=dev))).cpu().numpy()
    ok = ~np.isnan(P[:, 0])
    for nm, g in (("officiels EXCLUS", sel & ok & (src == 3)), ("générés inédits", sel & ok & (src == 1))):
        ps = hs(P[g]); R[f"sons prédits AUC ({nm})"] = {k: round(auc(D["SND"][g, j], ps[:, j]), 3) for j, k in enumerate(SND)}
        if nm == "officiels EXCLUS":
            q1, q2 = naive(D, np.flatnonzero(g))
            R["repère compter par geste AUC (officiels EXCLUS)"] = {k: round(auc(D["SND"][g, j], q1[:, j]), 3) for j, k in enumerate(SND)}
            R["repère son d'avant AUC (officiels EXCLUS)"] = {k: round(auc(D["SND"][g, j], q2[:, j] + 1e-3 * q1[:, j]), 3) for j, k in enumerate(SND)}
        if nm == "générés inédits":
            pl = hl(P[g]); lab = D["LAB"][g]; from blind_tokens import GEN_SOUNDS
            R["étiquettes générateur AUC (générés inédits)"] = {k: round(auc(lab[:, j] > 0, pl[:, j]), 3) for j, k in enumerate(GEN_SOUNDS) if (lab[:, j] >= 0).any()}
    # 2. « MOI » : où sera ce qui bouge sous mon geste fléché (depuis ẑ, avant de voir le pas) vs recopier la dernière position connue
    mv = (D["OBJ"][:, :, 5] > 0.5) & (np.abs(D["OBJ"][:, :, 9]) + np.abs(D["OBJ"][:, :, 10]) > 0.5 / 16)
    has = mv.any(1); first = np.argmax(mv, 1); pos = D["OBJ"][np.arange(len(first)), first, :2].astype(np.float32) * 64
    last = np.full((len(has), 2), 32.0, np.float32); cur = None; pe = -1
    for i in range(len(has)):
        if D["EPS"][i] != pe: cur, pe = None, D["EPS"][i]
        if cur is not None: last[i] = cur
        if has[i]: cur = pos[i]
    g = sel & ok & has & np.isin(D["ACT"], [1, 2, 3, 4]); tr, te = g & (src < 3), g & (src == 3)
    if te.sum() > 10 and tr.sum() > 50:
        pr = ridge(P[tr], pos[tr], P[te]); pr2 = ridge(np.c_[P[tr], last[tr]], pos[tr], np.c_[P[te], last[te]])
        R["moi : erreur position (cases, officiels EXCLUS)"] = dict(latent=round(float(np.abs(pr - pos[te]).sum(1).mean()), 2),
            latent_plus_derniere=round(float(np.abs(pr2 - pos[te]).sum(1).mean()), 2), recopier_derniere=round(float(np.abs(last[te] - pos[te]).sum(1).mean()), 2), n=int(te.sum()))
    # 3. GESTES AVANT LE DING (parties oracle) depuis z_t
    g = sel & D["ORACLE"] & (D["STW"] >= 0); tr, te = g & (src == 2), g & (src == 3)
    if te.sum() > 10:
        y = np.log1p(D["STW"].astype(np.float32)); pr = ridge(Z[tr], y[tr], Z[te])
        R["gestes avant le ding : Spearman (officiels EXCLUS)"] = dict(latent=round(spearman(pr, y[te]), 3),
            temps_dans_le_niveau=round(spearman(-D["T0"][te], y[te]), 3), n=int(te.sum()))
    Zs = Z[sel][::5]; Zs = Zs - Zs.mean(0); ev = np.linalg.svd(Zs, compute_uv=False) ** 2; ev = ev / ev.sum()
    R["rang effectif z"] = round(float(np.exp(-(ev * np.log(ev + 1e-12)).sum())), 1)
    return R


def loss_fn(m, tgt, b, a, dev):
    """toutes les pertes d'un lot (entraînement ET apprentissage pendant la partie) -> (perte, journal)."""
    V = b["V"]; g = lambda k, d=0.0: getattr(a, k, d)
    msk = (torch.rand(V.shape, device=dev) < g("mask_p", 0.25)) & V
    with torch.autocast("cuda", torch.bfloat16, enabled=dev == "cuda"):
        with torch.no_grad(): zt = tgt.encode(b)
        z = m.encode(b, msk)
        if m.late: c = m.ctx(z); zp = m.act_out(c, b["ACT"][:, 1:], b["CLK"][:, 1:])
        else: zp = m.predict(z, b)
        if m.late and g("arch", "flat") == "long" and (g("w_inv") > 0 or g("w_ctr") > 0):
            vo = m.steps(b, with_action=False)[:, 1:]                          # l'APRÈS sans le geste
            il = m.inv(torch.cat([z[:, :-1], vo], -1))
            # MAUVAIS GESTES : M gestes d'autres pas de la même fenêtre, à la place du vrai
            j = torch.randint(0, b["ACT"].shape[1] - 1, (*v_shape(b), g("n_alt", 6)), device=dev) + 1
            aa = torch.gather(b["ACT"][:, None, :].expand(-1, j.shape[1], -1), 2, j)
            ca = torch.gather(b["CLK"][:, None, :, :].expand(-1, j.shape[1], -1, -1), 2, j[..., None].expand(-1, -1, -1, 3))
            za = m.act_out(c[:, :, None].expand(-1, -1, g("n_alt", 6), -1), aa, ca)
    v1 = V[:, 1:]; zt, z, zp = zt.float(), z.float(), zp.float()
    l_pred = F.smooth_l1_loss(zp[v1], zt[:, 1:][v1]) * 10
    l_mask = F.smooth_l1_loss(z[msk], zt[msk]) * 10 if msk.any() else z.sum() * 0
    l_snd = F.binary_cross_entropy_with_logits(m.h_snd(zp[v1]), b["SND"][:, 1:][v1])
    lab = b["LAB"][:, 1:][v1]; lk = lab >= 0
    l_lab = F.binary_cross_entropy_with_logits(m.h_lab(zp[v1])[lk], lab[lk]) if lk.any() else z.sum() * 0
    l_var = F.relu(0.5 - z[V].std(0)).mean() + F.relu(0.5 - zp[v1].std(0)).mean()
    l_cov = cov_pen(z[V]) + cov_pen(zp[v1])                                   # VICReg : dimensions décorrélées (sinon tout tient sur 1 axe)
    l_inv = l_ctr = z.sum() * 0; acc_inv = acc_ctr = float("nan")
    if m.late and g("arch", "flat") == "long" and (g("w_inv") > 0 or g("w_ctr") > 0):
        il = il.float(); act1 = b["ACT"][:, 1:]; ck = b["CLK"][:, 1:]
        zone = torch.where(act1 == 6, (ck[..., 1] * 64).long().clamp(0, 63) // 8 * 8 + (ck[..., 0] * 64).long().clamp(0, 63) // 8, torch.full_like(act1, 64))
        l_inv = F.cross_entropy(il[..., :9][v1], act1[v1]) + (F.cross_entropy(il[..., 9:][v1 & (act1 == 6)], zone[v1 & (act1 == 6)]) if (v1 & (act1 == 6)).any() else 0)
        acc_inv = float((il[..., :9].argmax(-1) == act1)[v1].float().mean())
        same = (aa == act1[..., None]) & ((ca[..., :2] - ck[..., None, :2]).abs().sum(-1) * 64 < 8)      # même geste (clic à < 8 cases) = pas un mauvais geste
        tgt1 = zt[:, 1:].detach(); d0 = ((zp - tgt1) ** 2).sum(-1); da = ((za.float() - tgt1[:, :, None]) ** 2).sum(-1)
        lg = -torch.cat([d0[..., None], da], -1) / (g("tau", 0.05) * zp.size(-1)); lg[..., 1:] = lg[..., 1:].masked_fill(same, -1e4)
        ok = v1 & (b["SND"][:, 1:, 0] < 0.5) & (~same).any(-1)                  # pas de toc (le mauvais geste aurait pu faire pareil)
        if ok.any():
            l_ctr = F.cross_entropy(lg[ok], torch.zeros(int(ok.sum()), dtype=torch.long, device=dev)); acc_ctr = float((lg[ok].argmax(-1) == 0).float().mean())
    loss = l_pred + l_mask + g("w_heads", 0.5) * (l_snd + l_lab) + l_var + g("w_cov", 1.0) * l_cov + g("w_inv") * l_inv + g("w_ctr") * l_ctr
    return loss, dict(l_pred=l_pred, l_mask=l_mask, l_snd=l_snd, l_lab=l_lab, l_var=l_var, l_cov=l_cov, l_inv=l_inv, l_ctr=l_ctr, acc_inv=acc_inv, acc_ctr=acc_ctr)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--data", default="data/blind"); p.add_argument("--T", type=int, default=16)
    p.add_argument("--d", type=int, default=256); p.add_argument("--L", type=int, default=6); p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--steps", type=int, default=100000); p.add_argument("--max_minutes", type=float, default=60)
    p.add_argument("--mask_p", type=float, default=0.25); p.add_argument("--w_heads", type=float, default=0.5); p.add_argument("--eval_every", type=int, default=3000); p.add_argument("--w_cov", type=float, default=1.0); p.add_argument("--arch", default="flat", choices=["flat", "long"])
    p.add_argument("--w_inv", type=float, default=0.0); p.add_argument("--w_ctr", type=float, default=0.0); p.add_argument("--init", default=""); p.add_argument("--eye", type=int, default=0); p.add_argument("--n_alt", type=int, default=6); p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--out", default="blind.pt"); a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0); rng = np.random.default_rng(0)
    D = load(a.data); first = np.r_[True, D["EPS"][1:] != D["EPS"][:-1]]; trn = np.isin(D["SRC"], [0, 2])
    tr_rows = np.flatnonzero(trn & (first | (D["REM"] >= min(a.T, 64))))
    ep_starts = np.flatnonzero(trn & first)                                       # moitié des fenêtres : depuis le DÉBUT de la partie (mémoire complète)
    print(f"pas : {len(D['ACT'])} | entraînement (générés + officiels vus) départs possibles {len(tr_rows)} | officiels exclus {int((D['SRC'] == 3).sum())} pas | {dev}", flush=True)
    print("sons (part des pas, générés / officiels) :", {k: (round(float(D['SND'][D['SRC'] == 0, j].mean()), 3), round(float(D['SND'][D['SRC'] >= 2, j].mean()), 3)) for j, k in enumerate(SND)}, flush=True)
    late = True
    if a.init:                                                                    # --init : repartir d'un modèle aveugle (même architecture)
        ck0 = torch.load(a.init, map_location=dev, weights_only=False); A0 = ck0["args"]
        a.d, a.L, a.T, a.arch = A0["d"], A0["L"], A0["T"], A0["arch"]; late = A0.get("late", "w_inv" in A0)
    m = (BlindLong(a.d, a.L, 3, a.T, late=late, eye=bool(a.eye)) if a.arch == "long" else Blind(a.d, a.L, 3, a.T, late=late)).to(dev); a.late = late
    if a.init:
        miss = m.load_state_dict(ck0["m"], strict=False).missing_keys; print(f"repart de {a.init} (T={a.T}, late={late}, œil={bool(a.eye)}) ; nouveaux : {miss}", flush=True)
    tgt = copy.deepcopy(m).requires_grad_(False)
    print(f"paramètres {sum(x.numel() for x in m.parameters()) / 1e6:.1f} M", flush=True)
    opt = torch.optim.AdamW(m.parameters(), a.lr, weight_decay=0.05); t0 = time.time(); hist = []
    for it in range(1, a.steps + 1):
        frac = min(1.0, (time.time() - t0) / 60 / a.max_minutes); lr = a.lr * min(1, it / 1000) * 0.5 * (1 + math.cos(math.pi * frac))
        for gp in opt.param_groups: gp["lr"] = lr
        st = np.where(rng.random(a.bs) < 0.5, rng.choice(ep_starts, a.bs), rng.choice(tr_rows, a.bs))
        b = batch(D, st, a.T, dev)
        loss, Lg = loss_fn(m, tgt, b, a, dev); l_pred, l_mask, l_snd, l_lab, l_var, l_cov, l_inv, l_ctr, acc_inv, acc_ctr = Lg.values()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        with torch.no_grad():
            mom = 0.996 + 0.003 * frac
            for q, w in zip(tgt.parameters(), m.parameters()): q.mul_(mom).add_(w.detach(), alpha=1 - mom)
        if it % 250 == 0:
            print(f"pas {it} | prédiction {l_pred.item():.3f} masque {l_mask.item():.3f} sons {l_snd.item():.3f} étiquettes {l_lab.item():.3f} "
                  f"var {l_var.item():.3f} cov {l_cov.item():.3f} | deviner le geste {float(l_inv):.3f} (juste {acc_inv:.0%}) mauvais geste {float(l_ctr):.3f} (bon choisi {acc_ctr:.0%}) | {(time.time() - t0) / 60:.0f} min", flush=True)
        stop = (time.time() - t0) / 60 >= a.max_minutes
        if it % a.eval_every == 0 or stop:
            R = probes(m, D, a.T, dev, quick=not stop); R["pas"] = it; hist.append(R)
            print(("FINAL " if stop else "") + json.dumps(R, ensure_ascii=False), flush=True)
            json.dump(hist, open(a.out.replace(".pt", ".json"), "w"), ensure_ascii=False, indent=1)
            torch.save(dict(m=m.state_dict(), args=vars(a)), a.out)
        if stop: break


if __name__ == "__main__":
    main()
