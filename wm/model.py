"""
MODÈLE DU MONDE JEPA pour les jeux ARC (grilles 64×64, 16 couleurs) — le cœur du « bébé » transposé :
  œil      : grille (one-hot 16) -> latents par case (d, 16×16) ; œil-CIBLE = moyenne mobile (EMA) de l'œil (JEPA).
  prédicteur : latents(t) + ACTION (id 0..7 diffusé partout + carte du clic) -> latents(t+1) PRÉDITS (dans l'espace latent,
             pas en pixels) ; convolutions dilatées + un chemin global (effets lointains : gravité, compteur, portes).
  sondes   : à partir de [latents(t), latents(t+1) prédits] -> carte « ce pixel va changer » (64×64) et couleur suivante ;
             elles servent à l'AGENT (curiosité : quelles actions font quelque chose, où) et à mesurer le modèle.
Perte = JEPA (latents prédits vs œil-cible normalisé) + anti-effondrement (variance des latents) + sondes (change, couleur).
"""
import torch, torch.nn as nn, torch.nn.functional as F

NA = 8          # actions 0..7 (0 = RESET, jamais une transition apprise)


class Res(nn.Module):
    def __init__(s, d, dil=1):
        super().__init__(); s.c1 = nn.Conv2d(d, d, 3, padding=dil, dilation=dil); s.c2 = nn.Conv2d(d, d, 3, padding=1); s.n = nn.GroupNorm(8, d)
    def forward(s, x): return x + s.c2(F.gelu(s.n(s.c1(F.gelu(x)))))


class Eye(nn.Module):                                   # 64×64 one-hot -> (d, 16, 16)
    def __init__(s, d=128):
        super().__init__()
        s.stem = nn.Sequential(nn.Conv2d(16, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, 96, 4, stride=2, padding=1), nn.GELU(),
                               nn.Conv2d(96, d, 4, stride=2, padding=1))
        s.pos = nn.Parameter(torch.randn(1, d, 16, 16) * 0.02)
        s.blocks = nn.Sequential(Res(d), Res(d, 2), Res(d, 4))
    def forward(s, g):                                  # g (B,64,64) long 0..15
        x = F.one_hot(g.clamp(0, 15).long(), 16).permute(0, 3, 1, 2).float()
        return s.blocks(s.stem(x) + s.pos)


class Predictor(nn.Module):
    def __init__(s, d=128, h=192):
        super().__init__()
        s.inp = nn.Conv2d(d + 1 + 16, h, 1); s.aemb = nn.Embedding(NA, h)
        s.blocks = nn.ModuleList([Res(h, dl) for dl in (1, 2, 4, 8, 1, 2)])
        s.glob = nn.ModuleList([nn.Linear(h, 2 * h) for _ in range(6)])     # chemin GLOBAL (FiLM) : effets à distance
        s.out = nn.Conv2d(h, d, 1); nn.init.zeros_(s.out.weight); nn.init.zeros_(s.out.bias)   # départ = « rien ne change »
    def forward(s, z, a, click):                        # z (B,d,16,16), a (B,), click (B,2) x,y (-1 = aucun)
        B = z.size(0); cm = torch.zeros(B, 1, 64, 64, device=z.device)
        has = click[:, 0] >= 0; idx = torch.nonzero(has).flatten()
        if len(idx): cm[idx, 0, click[idx, 1].long().clamp(0, 63), click[idx, 0].long().clamp(0, 63)] = 1.0
        cfine = F.pixel_unshuffle(cm, 4)                # (B,16,16,16) : quelle case + où DANS la case (précision 1 px)
        x = s.inp(torch.cat([z, cfine.sum(1, keepdim=True), cfine], 1)) + s.aemb(a.long()).view(B, -1, 1, 1)
        for blk, gl in zip(s.blocks, s.glob):
            sc, sh = gl(x.mean((2, 3))).chunk(2, -1); x = blk(x * (1 + sc.view(B, -1, 1, 1)) + sh.view(B, -1, 1, 1))
        return s.out(x)                                 # la DIFFÉRENCE causée par l'action (ajoutée aux latents cibles de t)


class Probe(nn.Module):                                 # [z_t, ẑ_t+1] -> logits changement (64×64) + couleur suivante (16×64×64)
    def __init__(s, d=128):
        super().__init__()
        s.up = nn.Sequential(nn.Conv2d(2 * d, 128, 1), nn.GELU(), nn.ConvTranspose2d(128, 96, 4, 2, 1), nn.GELU(),
                             nn.ConvTranspose2d(96, 64, 4, 2, 1), nn.GELU())
        s.chg = nn.Conv2d(64 + 16, 1, 3, padding=1); s.col = nn.Conv2d(64 + 16, 16, 3, padding=1)
    def forward(s, z, zp, g):
        u = torch.cat([s.up(torch.cat([z, zp], 1)), F.one_hot(g.clamp(0, 15).long(), 16).permute(0, 3, 1, 2).float()], 1)
        return s.chg(u)[:, 0], s.col(u)


EV_BIN = list(range(0, 5)) + [9] + list(range(10, 22))   # toucher, « j'ai bougé », sons (événements oui/non)
EV_POS = [5, 6, 7, 8]                                    # corps : position de MON objet avant/après (-1 = aucun)
EV_PROG = 22                                             # progrès vers le but (-1..1)
NE = 23


class EventHead(nn.Module):                             # SON / TOUCHER / CORPS / PROGRÈS lus dans [z_t, ẑ_t+1] (enseignants privilégiés)
    def __init__(s, d=128):
        super().__init__(); s.red = nn.Conv2d(2 * d, 8, 1)
        s.mlp = nn.Sequential(nn.Linear(8 * 256, 256), nn.GELU(), nn.Linear(256, NE))
    def forward(s, z, zp): return s.mlp(s.red(torch.cat([z, zp], 1)).flatten(1))


class WorldModel(nn.Module):
    def __init__(s, d=128):
        super().__init__(); s.d = d; s.eye = Eye(d); s.pred = Predictor(d); s.probe = Probe(d); s.ev = EventHead(d)
        s.tgt = Eye(d); s.tgt.load_state_dict(s.eye.state_dict())
        for p in s.tgt.parameters(): p.requires_grad_(False)

    @torch.no_grad()
    def ema(s, m=0.996):
        for p, q in zip(s.eye.parameters(), s.tgt.parameters()): q.mul_(m).add_(p.detach(), alpha=1 - m)

    def lnt(s, g):                                       # latents de l'œil-CIBLE, normalisés par case
        return F.layer_norm(s.tgt(g).permute(0, 2, 3, 1), (s.d,)).permute(0, 3, 1, 2)

    def predict(s, g, a, click):                        # pour l'agent : (z, ẑ_t+1, logits changement, logits couleur)
        z = s.eye(g); zp = s.lnt(g).detach() + s.pred(z, a, click); ch, col = s.probe(z, zp, g); return z, zp, ch, col

    def loss(s, g, a, click, g2, w_probe=1.0, ev=None, w_ev=0.0):
        z = s.eye(g)
        with torch.no_grad(): zc, zt = s.lnt(g), s.lnt(g2)
        zp = zc + s.pred(z, a, click)                    # ẑ_t+1 = latents cibles de t + effet prédit de l'action
        jepa = F.smooth_l1_loss(zp, zt)
        zf = z.permute(0, 2, 3, 1).reshape(-1, s.d); var = F.relu(1 - zf.std(0)).mean()        # anti-effondrement (VICReg-var)
        ch, col = s.probe(z, zp, g); chg = (g2 != g).float()
        pw = ((1 - chg.mean()) / chg.mean().clamp_min(1e-3)).clamp(1, 50)                     # peu de pixels changent : pondérer
        l_ch = F.binary_cross_entropy_with_logits(ch, chg, pos_weight=pw)
        l_col = F.cross_entropy(col, g2.clamp(0, 15).long(), reduction="none")
        l_col = (l_col * (1 + 9 * chg)).mean()                                                  # surtout là où ça change
        with torch.no_grad():
            pc = ch > 0; tp = (pc & (chg > 0)).sum().float()
            f1 = 2 * tp / (pc.sum() + (chg > 0).sum()).clamp_min(1)
        tot = jepa + 0.5 * var + w_probe * (l_ch + l_col); st = dict(jepa=float(jepa), var=float(var), chg=float(l_ch), col=float(l_col), f1=float(f1))
        if w_ev > 0 and ev is not None:                 # le son et le toucher n'existent qu'à l'ENTRAÎNEMENT (jeux générés)
            has = ~torch.isnan(ev[:, 0])
            if has.any():
                e = ev[has]; o = s.ev(z[has], zp[has])
                l_bin = F.binary_cross_entropy_with_logits(o[:, EV_BIN], e[:, EV_BIN])
                pm = e[:, EV_POS] >= 0; l_pos = F.smooth_l1_loss(o[:, EV_POS][pm], e[:, EV_POS][pm]) if pm.any() else o.sum() * 0
                l_prog = F.smooth_l1_loss(torch.tanh(o[:, EV_PROG]), e[:, EV_PROG])
                tot = tot + w_ev * (l_bin + l_pos + l_prog); st.update(ev_bin=float(l_bin), ev_pos=float(l_pos), ev_prog=float(l_prog))
        return tot, st
