"""
MODÈLE DU MONDE VIDÉO (avec MÉMOIRE) pour ARC — chaque partie est une VIDÉO découpée en séquences de T images + actions,
comme le bébé (V-JEPA, prédicteur causal) : pour prédire l'image t+1, il voit TOUTES les images ≤ t de la séquence et
TOUTES les actions ≤ t. Il peut donc accumuler des indices DANS la partie (« ACTION3 me déplace », « la clé ouvre la porte »)
= apprendre la règle de CE jeu en contexte, au lieu de prédire la moyenne de tous les jeux.
  œil       : chaque image 64×64 -> 64 jetons (cases de 8×8 px) de dimension d ; œil-CIBLE = EMA (JEPA)
  prédicteur : transformeur BLOC-CAUSAL sur T×64 jetons (+ l'action et le clic de chaque pas injectés sur les jetons de son
              image) -> latents de l'image suivante pour chaque t (cible = œil-cible de l'image t+1, normalisé)
  têtes     : carte « ce pixel va changer » (t -> t+1), événements (son/toucher/progrès, jeux générés), son spatial 8×8
  tgt_local : l'œil-CIBLE lit chaque case 8×8 SEULE (sans ses couches d'attention) -> une case immobile garde exactement son
              latent ; leçon du bébé (D3) : l'entraînement JEPA globalise les cases, il faut qu'elles décrivent LEUR endroit,
              le global (scène, mémoire) restant dans l'œil en ligne et le prédicteur.
  head_src  : « zp » = les têtes lisent la sortie JEPA (historique) ; « hid » = elles lisent l'état INTERNE du prédicteur
              (avant sa projection de sortie) -> leurs pertes (40× la perte JEPA) ne déforment plus la prédiction latente.
  head obj  : « CRAYON » NET (diagnostic visuel 2026-10-24 : la carte de changement était une TACHE FLOUE au bon endroit, même
              sur les jeux connus -> F1 plafonné ~0.5) : la sortie travaille à la résolution du PIXEL avec l'image actuelle
              (couleurs exactes) + regroupement par OBJET (composantes connexes de même couleur : tous les pixels d'un objet
              partagent l'information « cet objet change ») ; les contours viennent de l'image, le modèle décide QUOI change.
  head col  : CRAYON « COULEUR » (mesure 2026-10-24 : même sur ses jeux VUS et déterministes le modèle plafonnait à F1 ~0.5) :
              il prédit la COULEUR de chaque pixel de l'image suivante (choix net parmi 16, base = « le pixel garde sa
              couleur ») avec un décodeur U-Net à pleine résolution qui voit l'image actuelle à 64/32/16/8 px et reçoit les
              jetons du modèle au niveau 8×8 ; « ce pixel change » = probabilité de ne PAS garder sa couleur.
  rollout   : IMAGINER PLUSIEURS PAS (System-2, étape 1) : jusqu'à t0 il voit les vraies images, ensuite il ne reçoit que
              SES PROPRES PRÉDICTIONS (latents imaginés, ramenés dans l'espace de l'œil par « proj ») et continue ; k passes
              du prédicteur bloc-causal = k pas d'imagination exacts ; le gradient traverse toute la chaîne (corrige la dérive).
"""
import math
import torch, torch.nn as nn, torch.nn.functional as F
from model import NA, NE, EV_BIN, EV_POS, EV_PROG

P = 8; G = 64 // P; NT = G * G                       # 64 jetons par image


class Eye(nn.Module):
    def __init__(s, d=256):
        super().__init__()
        s.c = nn.Sequential(nn.Conv2d(16, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, 128, 4, stride=4), nn.GELU(),
                            nn.Conv2d(128, d, 2, stride=2))                                   # 64 -> 16 -> 8
        s.pos = nn.Parameter(torch.randn(1, NT, d) * 0.02); s.tr = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 8, 4 * d, 0.0, "gelu", batch_first=True, norm_first=True), 2)
    def forward(s, g, local=False):                   # g (N,64,64) -> (N,64,d) ; local = chaque case ne voit que ses 8×8 px
        x = s.c(F.one_hot(g.clamp(0, 15).long(), 16).permute(0, 3, 1, 2).float()).flatten(2).transpose(1, 2)
        return x + s.pos if local else s.tr(x + s.pos)


def comp_labels(g, iters=64):
    """objets = composantes 4-connexes de même couleur ; g (N,64,64) -> étiquettes (N,64,64) dans [0, 4096) (= plus petit
    indice de pixel de l'objet). Propagation du minimum entre voisins de même couleur + SAUT DE POINTEURS (chaque pixel
    adopte l'étiquette de son étiquette) -> converge en quelques dizaines de tours même pour un fond qui serpente."""
    N = g.size(0); big = 1 << 20; gp = F.pad(g, (1, 1, 1, 1), value=-1)
    lab = torch.arange(4096, device=g.device).view(1, 64, 64).expand(N, -1, -1).contiguous()
    for _ in range(iters):
        lp = F.pad(lab, (1, 1, 1, 1), value=big); new = lab
        for sy, sx in ((0, 1), (2, 1), (1, 0), (1, 2)):                         # haut, bas, gauche, droite
            new = torch.minimum(new, torch.where(gp[:, sy:sy + 64, sx:sx + 64] == g, lp[:, sy:sy + 64, sx:sx + 64], big))
        new = torch.gather(new.view(N, -1), 1, new.view(N, -1)).view(N, 64, 64)      # saut de pointeurs
        if torch.equal(new, lab): break
        lab = new
    return lab


class ObjHead(nn.Module):
    """carte « ce pixel va changer », nette : convolutions à la résolution du pixel sur [traits du modèle, couleurs exactes],
    puis moyenne des traits PAR OBJET renvoyée à chacun de ses pixels, puis décision."""
    def __init__(s, cin=32 + 16, c=64):
        super().__init__()
        s.a = nn.Sequential(nn.Conv2d(cin, c, 3, padding=1), nn.GELU(), nn.Conv2d(c, c, 3, padding=2, dilation=2), nn.GELU(),
                            nn.Conv2d(c, c, 3, padding=4, dilation=4), nn.GELU())
        s.b = nn.Sequential(nn.Conv2d(2 * c, c, 3, padding=1), nn.GELU(), nn.Conv2d(c, 1, 1))
    def forward(s, u, g):                               # u (N,32,64,64), g (N,64,64)
        x = s.a(torch.cat([u, F.one_hot(g.clamp(0, 15).long(), 16).permute(0, 3, 1, 2).to(u.dtype)], 1))
        N, C = x.shape[:2]; lab = comp_labels(g) + torch.arange(N, device=g.device).view(N, 1, 1) * 4096
        xf = x.permute(0, 2, 3, 1).reshape(-1, C).float(); idx = lab.reshape(-1)
        sm = torch.zeros(N * 4096, C, device=x.device).index_add_(0, idx, xf); cn = torch.zeros(N * 4096, device=x.device).index_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
        po = (sm / cn.clamp_min(1)[:, None])[idx].view(N, 64, 64, C).permute(0, 3, 1, 2).to(x.dtype)   # trait moyen de MON objet
        return s.b(torch.cat([x, po], 1))[:, 0]


class ColorHead(nn.Module):
    """image actuelle (one-hot, pleine résolution) -> U-Net ; les jetons du modèle (2d) entrent au niveau 8×8 ; sortie : logits de
    couleur (N,16,64,64) + biais appris « garder sa couleur »."""
    def __init__(s, d2, c=48):
        super().__init__()
        cb = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GELU(), nn.Conv2d(o, o, 3, padding=1), nn.GELU())
        s.e64, s.e32, s.e16, s.e8 = cb(16, c), cb(c, 2 * c), cb(2 * c, 4 * c), cb(4 * c, 4 * c)
        s.tok = nn.Linear(d2, 4 * c); s.m8 = cb(8 * c, 4 * c)
        s.d16, s.d32, s.d64 = cb(8 * c, 2 * c), cb(4 * c, c), cb(2 * c, c)
        s.out = nn.Conv2d(c, 16, 1); s.keep = nn.Parameter(torch.tensor(4.0))
    def forward(s, h, g):                               # h (N,64,2d) jetons, g (N,64,64)
        oh = F.one_hot(g.clamp(0, 15).long(), 16).permute(0, 3, 1, 2).to(h.dtype); N = g.size(0)
        x64 = s.e64(oh); x32 = s.e32(F.max_pool2d(x64, 2)); x16 = s.e16(F.max_pool2d(x32, 2)); x8 = s.e8(F.max_pool2d(x16, 2))
        t = s.tok(h).transpose(1, 2).reshape(N, -1, G, G)
        y = s.m8(torch.cat([x8, t], 1))
        y = s.d16(torch.cat([F.interpolate(y, scale_factor=2), x16], 1))
        y = s.d32(torch.cat([F.interpolate(y, scale_factor=2), x32], 1))
        y = s.d64(torch.cat([F.interpolate(y, scale_factor=2), x64], 1))
        return s.out(y) + s.keep * oh                   # par défaut : chaque pixel garde sa couleur


class SeqPredictor(nn.Module):
    def __init__(s, d=256, T=16, L=6):
        super().__init__(); s.T = T
        s.aemb = nn.Embedding(NA, d); s.cemb = nn.Embedding(NT + 1, d); s.cfine = nn.Embedding(P * P + 1, d)
        s.temb = nn.Parameter(torch.randn(1, T, 1, d) * 0.02); s.semb = nn.Parameter(torch.randn(1, 1, NT, d) * 0.02)
        s.layers = nn.ModuleList([nn.TransformerEncoderLayer(d, 8, 4 * d, 0.0, "gelu", batch_first=True, norm_first=True) for _ in range(L)])
        s.out = nn.Linear(d, d); nn.init.zeros_(s.out.weight); nn.init.zeros_(s.out.bias)
        # CANAL SON EN ENTRÉE (26 oct., idée user « le modèle doit ENTENDRE ») : ce que le geste précédent a produit (zones qui ont bougé,
        # ding de fin de niveau, buzz de game over) + l'INTENTION (un succès dans les 8 prochains pas ?). Zéro au départ.
        s.s_chg = nn.Parameter(torch.zeros(d)); s.s_lc = nn.Parameter(torch.zeros(d)); s.s_go = nn.Parameter(torch.zeros(d)); s.s_rtg = nn.Parameter(torch.zeros(2, d))
        blk = torch.arange(T).repeat_interleave(NT); s.register_buffer("mask", blk[None, :] > blk[:, None])   # bloc-causal (True = interdit)
    def inject(s, a, click):                          # actions DURES -> ce qu'on ajoute aux jetons de chaque image (B,T,64,d)
        hit = click[..., 0] >= 0; cx, cy = click[..., 0].clamp(0, 63), click[..., 1].clamp(0, 63)
        cell = torch.where(hit, (cy // P) * G + cx // P, torch.full_like(cx, NT)); fine = torch.where(hit, (cy % P) * P + cx % P, torch.full_like(cx, P * P))
        onc = F.one_hot(cell.clamp(max=NT - 1), NT).float() * hit[..., None]                        # le jeton cliqué reçoit le clic
        return s.aemb(a.long())[:, :, None] + (s.cemb(cell) + s.cfine(fine))[:, :, None] * (0.5 + onc[..., None])

    def inject_soft(s, a_soft, pw, pc):               # actions ASSOUPLIES (planification par gradient) : a_soft (B,T,NA) mélange d'actions,
        """pw (B,T,4096) poids sur les pixels cliqués (somme 1), pc (B,T) part « clic » ; identique à inject() si tout est one-hot."""
        if not hasattr(s, "_cell"):
            pix = torch.arange(64 * 64, device=pw.device); y, x = pix // 64, pix % 64
            s._cell, s._fine = (y // P) * G + x // P, (y % P) * P + x % P
        Epix = s.cemb.weight[s._cell] + s.cfine.weight[s._fine]                     # (4096,d) : plongement de chaque pixel cliquable
        Enone = s.cemb.weight[NT] + s.cfine.weight[P * P]                           # « pas de clic »
        ce = pc[..., None] * (pw @ Epix) + (1 - pc[..., None]) * Enone             # (B,T,d)
        onc = pc[..., None] * pw.view(*pw.shape[:2], G, P, G, P).sum((3, 5)).reshape(*pw.shape[:2], NT)
        return (a_soft @ s.aemb.weight[:a_soft.size(-1)])[:, :, None] + ce[:, :, None] * (0.5 + onc[..., None])

    def forward(s, z, a, click, inj=None, snd=None):  # z (B,T,64,d) ; a (B,T) ; click (B,T,2) ; snd (B,T,64,d) son entendu -> (B,T,64,d) delta pour t+1
        B, T = z.shape[:2]
        x = z + s.temb[:, :T] + s.semb + (s.inject(a, click) if inj is None else inj) + (0 if snd is None else snd)
        x = x.reshape(B, T * NT, -1); m = s.mask[:T * NT, :T * NT]
        for l in s.layers: x = l(x, src_mask=m)
        s.hid = x.reshape(B, T, NT, -1)                                              # état interne (lu par les têtes si head_src = hid)
        return s.out(x).reshape(B, T, NT, -1)

    def ctx_cache(s, z, inj):                         # PLANIFICATION : clés/valeurs des C images RÉELLES du contexte (calculées 1 fois)
        """z, inj (1,C,64,d) -> liste par couche de (K, V) (1,C*64,d) ; bloc-causal : le contexte ne dépend pas du plan."""
        C = z.size(1); d = z.size(-1)
        if C == 0: s.ctx_last = None; return [(z.new_zeros(1, 0, d), z.new_zeros(1, 0, d)) for _ in s.layers]
        x = (z + s.temb[:, :C] + s.semb + inj).reshape(1, C * NT, -1); m = s.mask[:C * NT, :C * NT]; kv = []
        for l in s.layers:
            W, b = l.self_attn.in_proj_weight, l.self_attn.in_proj_bias; h = l.norm1(x)
            kv.append((F.linear(h, W[d:2 * d], b[d:2 * d]), F.linear(h, W[2 * d:], b[2 * d:]))); x = l(x, src_mask=m)
        s.ctx_last = x.reshape(1, C, NT, -1)[:, -1]                                    # état interne au dernier pas réel (lu par l'acteur)
        return kv

    def forward_cached(s, kv, z, inj, t0):            # positions t0..t0+J-1 seulement (S séquences), le contexte 0..t0-1 vient du cache
        S, J = z.shape[:2]; d = z.size(-1)
        x = (z + s.temb[:, t0:t0 + J] + s.semb + inj).reshape(S, J * NT, -1)
        allow = torch.cat([torch.ones(J * NT, kv[0][0].size(1), dtype=torch.bool, device=x.device),
                           ~s.mask[t0 * NT:(t0 + J) * NT, t0 * NT:(t0 + J) * NT]], 1)
        for l, (k0, v0) in zip(s.layers, kv):
            a = l.self_attn; nh = a.num_heads; sp = lambda t: t.reshape(S, -1, nh, d // nh).transpose(1, 2)
            q, k, v = F.linear(l.norm1(x), a.in_proj_weight, a.in_proj_bias).chunk(3, -1)
            K, V = torch.cat([k0.to(k.dtype).expand(S, -1, -1), k], 1), torch.cat([v0.to(v.dtype).expand(S, -1, -1), v], 1)
            o = F.scaled_dot_product_attention(sp(q), sp(K), sp(V), attn_mask=allow).transpose(1, 2).reshape(S, J * NT, d)
            x = x + a.out_proj(o); x = x + l.linear2(l.activation(l.linear1(l.norm2(x))))
        s.hid = x.reshape(S, J, NT, -1)
        return s.out(x).reshape(S, J, NT, -1)


class TokEmb(nn.Module):                              # plongement des actions + jetons appris à part (index NA = « ? » action inconnue)
    def __init__(s, old):
        super().__init__(); s.base = nn.Parameter(old.weight.detach().clone())
        s.tok = nn.Parameter(old.weight[1:].detach().mean(0, keepdim=True).repeat(2, 1).clone()); s.embedding_dim = old.embedding_dim
    @property
    def weight(s): return torch.cat([s.base, s.tok])
    def forward(s, a): return F.embedding(a, s.weight)


class SeqWorldModel(nn.Module):
    seq = True

    def __init__(s, d=256, T=16, L=6, tgt_local=False, head_src="zp", head="px", crit=False, actor=False, dream=False):
        super().__init__(); s.d, s.T, s.tgt_local, s.head_src, s.head = d, T, tgt_local, head_src, head
        if crit: s.crit = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 2))   # CRITIQUE : [cette action finit le niveau, fini dans ≤ 8 pas]
        if crit: s.crit_d = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 1))  # DISTANCE AU BUT (27 oct.) : γ^(pas restants avant la fin du niveau), γ = 0.97
        if actor:                                                                       # ACTEUR (Mode 1 de LeCun) : le geste de l'EXPERT (oracle)
            s.act_tok = nn.Sequential(nn.Linear(2 * d, 256), nn.GELU())                 # lit l'image ACTUELLE (œil) + la mémoire du pas d'avant (prédicteur),
            s.act_a = nn.Linear(256, NA); s.act_c = nn.Linear(256, 1 + P * P)           # jamais l'action à prédire : action, case cliquée (64), pixel dans la case (64)
        s.eye, s.tgt = Eye(d), Eye(d); s.tgt.load_state_dict(s.eye.state_dict())
        for p in s.tgt.parameters(): p.requires_grad_(False)
        s.pred = SeqPredictor(d, T, L)
        if dream: s.pred.aemb = TokEmb(s.pred.aemb)                                   # RÊVE : action « ? » + intention (succès dans les 8 pas ?)
        s.dream = dream
        s.up = nn.Sequential(nn.Linear(2 * d, P * P * 32), nn.GELU())                # jeton -> 8×8 px × 32 canaux
        s.chg = nn.Conv2d(32 + 16, 1, 3, padding=1)
        if head == "obj": s.obj = ObjHead()                                           # crayon net (pixel + objets)
        if head == "col": s.col = ColorHead(2 * d)                                     # crayon couleur (U-Net pleine résolution)
        s.ev = nn.Sequential(nn.Linear(2 * d, 256), nn.GELU(), nn.Linear(256, NE))
        s.snd = nn.Linear(2 * d, 1)                                                   # son spatial : par jeton (= zone 8×8)
        s.proj = nn.Linear(d, d)                                                      # latent imaginé (espace cible) -> entrée du prédicteur (espace de l'œil)

    @torch.no_grad()
    def ema(s, m=0.996):
        for p, q in zip(s.eye.parameters(), s.tgt.parameters()): q.mul_(m).add_(p.detach(), alpha=1 - m)

    def lnt(s, g): return F.layer_norm(s.tgt(g, s.tgt_local), (s.d,))   # (N,64,d)

    def heads(s, z, zp, g):                           # z, zp (N,64,d), g (N,64,64) -> change logits (N,64,64), events (N,NE), son (N,8,8)
        h = torch.cat([z, zp], -1); N = h.size(0)
        u = s.up(h).view(N, G, G, P, P, 32).permute(0, 5, 1, 3, 2, 4).reshape(N, 32, 64, 64)
        if s.head == "col":                                                          # p(change) = 1 - p(garder sa couleur)
            lc = s.col(h, g); s.last_col = lc
            lp = F.log_softmax(lc.float(), 1); keep = lp.gather(1, g.clamp(0, 15).long()[:, None])[:, 0]
            ch = torch.log1p(-keep.exp().clamp(max=1 - 1e-6)) - keep                  # logit de « change »
        elif s.head == "obj": ch = s.obj(u, g)
        else: ch = s.chg(torch.cat([u, F.one_hot(g.clamp(0, 15).long(), 16).permute(0, 3, 1, 2).float()], 1))[:, 0]
        return ch, s.ev(h.mean(1)), s.snd(h)[..., 0].view(N, G, G)

    def sound(s, g, lc, go, rtg):                     # SON ENTENDU au pas t (= produit par la transition t-1 -> t) + intention -> (B,T,64,d)
        """g (B,T(+1),64,64) images ; lc, go (B,T) ding / buzz de la transition t-1 -> t (déjà décalés) ; rtg (B,T) 0/1 succès dans les 8 prochains pas"""
        B, T = lc.shape; f = g[:, :T]; prev = torch.cat([f[:, :1], f[:, :-1]], 1)
        chg = (f != prev).view(B, T, G, P, G, P).any(5).any(3).float().view(B, T, NT)    # zones 8×8 qui viennent de bouger (« le bruit vient de là »)
        p = s.pred; out = chg[..., None] * p.s_chg + lc.float()[..., None, None] * p.s_lc + go.float()[..., None, None] * p.s_go
        if rtg is None: return out
        r = rtg.long(); return out + (r >= 0).float()[..., None, None] * p.s_rtg[r.clamp(min=0)][:, :, None]

    def heard(s, S, E=None):                          # ding / buzz ENTENDUS au pas t = produits par la transition t-1 -> t (S, E : états / événements des pas)
        lc, go = s.crit_targets(S, E)[0], (S == 2).float()
        sh = lambda x: torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], 1)
        return sh(lc), sh(go)

    def run(s, g, a, click, snd=None):                # g (B,T+1,64,64) -> z (B,T,64,d), zp (B,T,64,d), zt (B,T,64,d)
        B, T1 = g.shape[:2]; T = T1 - 1
        if snd is None and getattr(s, "hear", False):   # modèle qui ENTEND : son de la dernière transition (s.cur = ding, buzz posés par l'appelant)
            lc, go = s.cur if getattr(s, "cur", None) is not None and s.cur[0].shape == (B, T) else (torch.zeros(B, T, device=g.device),) * 2
            snd = s.sound(g, lc, go, None)
        z = s.eye(g[:, :T].reshape(B * T, 64, 64)).view(B, T, NT, -1)
        with torch.no_grad(): L = s.lnt(g.reshape(B * T1, 64, 64)).view(B, T1, NT, -1)
        zp = L[:, :T] + s.pred(z, a, click, snd=snd); return z, zp, L[:, 1:], L[:, :T]

    def rollout(s, z, L, a, click, t0, k, snd=None):
        """z (B,T,64,d) œil sur les vraies images, L (B,T+1,64,d) cibles ; vraies images jusqu'à t0, ensuite imagination.
        Renvoie zp (B,T,64,d) : zp[:, t] = latent imaginé de l'image t+1 ; exact jusqu'à t0 + k.
        snd : son entendu, gardé seulement sur les pas RÉELS (≤ t0) — en imagination on n'entend rien (comme le planificateur)."""
        T = z.size(1)
        if snd is not None: snd = snd * (torch.arange(T, device=z.device) <= t0).float()[None, :, None, None]
        zp = L[:, :T] + s.pred(z, a, click, snd=snd)                                    # passe 0 : vraies images partout
        for _ in range(k):
            im = zp[:, t0:T - 1]                                                        # images t0+1.. imaginées (prédites au pas d'avant)
            base = torch.cat([L[:, :t0 + 1], im], 1); zin = torch.cat([z[:, :t0 + 1], s.proj(im)], 1)
            zp = base + s.pred(zin, a, click, snd=snd)
        return zp

    def actor(s, z, hprev):                           # z (B,T,64,d) œil des images t ; hprev (B,T,64,d) état interne au pas t-1 -> logits
        h = s.act_tok(torch.cat([z, hprev], -1).float())                                # (B,T,64,256)
        o = s.act_c(h); return s.act_a(h.mean(2)), o[..., 0], o[..., 1:]                # action (B,T,NA), case (B,T,64), pixel dans la case (B,T,64,64)

    def critic(s):                                    # sur l'état interne du prédicteur (mémoire comprise), réel OU imaginé -> logits (B,T,2)
        return s.crit(s.pred.hid.mean(2))

    @staticmethod
    def crit_targets(S, E=None, H=8):                 # fin de niveau à l'action t : state = 1 (jeux officiels) OU son « level_complete »
        y1 = (S == 1)                                 # (événement 17, jeux générés : leur state ne marque presque jamais les fins de niveau)
        if E is not None: y1 = y1 | (torch.nan_to_num(E[..., 17].float()) > 0.5)
        y1 = y1.float(); T = S.size(1)
        y2 = torch.stack([y1[:, t:t + H].amax(1) for t in range(T)], 1)
        return y1, y2

    @staticmethod
    def actor_loss(la, lt, lf, a, click, Bh):          # entropie croisée sur les pas OÙ L'ORACLE JOUE (et pas RESET) ; clic : case puis pixel
        m_ = (Bh == 0) & (a > 0)
        if not m_.any(): return None
        l = F.cross_entropy(la[m_].float(), a[m_].long())
        mc = m_ & (a == 6) & (click[..., 0] >= 0)
        if mc.any():
            cx, cy = click[..., 0].clamp(0, 63), click[..., 1].clamp(0, 63); cell = (cy // P) * G + cx // P; fine = (cy % P) * P + cx % P
            l = l + F.cross_entropy(lt[mc].float(), cell[mc].long()) + F.cross_entropy(lf[mc].float().gather(1, cell[mc].long()[:, None, None].expand(-1, 1, P * P))[:, 0], fine[mc].long())
        return l

    def hin(s, zp):                                   # ce que lisent les têtes : la sortie JEPA ou l'état interne du prédicteur
        return s.pred.hid if s.head_src == "hid" else zp

    def loss(s, g, a, click, ev=None, w_ev=1.0, w_jepa=1.0, w_roll=0.0, roll_k=4, S=None, w_crit=0.0, w_snd=0.5, Bh=None, w_act=0.0, snd=None, a_true=None, c_true=None, w_crit_im=0.0, V=None):      # g (B,T+1,64,64) ; a (B,T) ; click (B,T,2) ; ev (B,T,NE) ; snd = son ENTENDU (entrée)
        B, T1 = g.shape[:2]; T = T1 - 1
        z, zp, zt, _ = s.run(g, a, click, snd); jepa_t = F.smooth_l1_loss(zp, zt, reduction="none").mean((0, 2, 3))   # par position t
        zf = z.reshape(-1, s.d); var = F.relu(1 - zf.std(0)).mean()
        g0, g1 = g[:, :T].reshape(B * T, 64, 64), g[:, 1:].reshape(B * T, 64, 64)
        ch, eo, so = s.heads(z.reshape(B * T, NT, -1), s.hin(zp).reshape(B * T, NT, -1), g0); chg = (g1 != g0).float()
        pw = ((1 - chg.mean()) / chg.mean().clamp_min(1e-3)).clamp(1, 50); l_ch = F.binary_cross_entropy_with_logits(ch, chg, pos_weight=pw)
        if s.head == "col":                                                            # + la COULEUR exacte de l'image suivante (pixels qui changent ×10)
            l_ch = l_ch + (F.cross_entropy(s.last_col.float(), g1.clamp(0, 15).long(), reduction="none") * (1 + 9 * chg)).mean()
        snd_t = chg.view(-1, G, P, G, P).mean((2, 4)); l_snd = F.binary_cross_entropy_with_logits(so, snd_t)
        tot = w_jepa * jepa_t.mean() + 0.5 * var + l_ch + w_snd * l_snd                 # w_snd 0 = sans « son spatial » (ablation)
        st = dict(jepa=float(jepa_t.mean().detach()), var=float(var.detach()), chg=float(l_ch.detach()))
        if w_crit > 0 and S is not None and hasattr(s, "crit"):                          # CRITIQUE (avant l'imagination, qui réécrit pred.hid)
            lo = s.critic().float(); y1, y2 = s.crit_targets(S, ev); l_cr = 0
            for k_, y_ in ((0, y1), (1, y2)):
                pw_ = ((1 - y_.mean()) / y_.mean().clamp_min(1e-3)).clamp(1, 100)
                l_cr = l_cr + F.binary_cross_entropy_with_logits(lo[..., k_], y_, pos_weight=pw_)
            if V is not None and hasattr(s, "crit_d"):                                   # distance au but (vue sur toute la partie, au-delà de la fenêtre)
                l_d = F.binary_cross_entropy_with_logits(s.crit_d(s.pred.hid.mean(2)).float()[..., 0], V.float()); l_cr = l_cr + l_d; st.update(dist=float(l_d.detach()))
            tot = tot + w_crit * l_cr; st.update(crit=float(l_cr.detach()))
        if w_act > 0 and Bh is not None and hasattr(s, "act_a"):                        # ACTEUR : imiter l'oracle (behav 0), sur un monde déjà compris
            hp = torch.cat([torch.zeros_like(s.pred.hid[:, :1]), s.pred.hid[:, :-1]], 1).detach()
            la, lt, lf = s.actor(z.detach(), hp); l_act = s.actor_loss(la, lt, lf, a if a_true is None else a_true, click if c_true is None else c_true, Bh)   # rêve : l'acteur apprend le VRAI geste
            if l_act is not None: tot = tot + w_act * l_act; st.update(act=float(l_act.detach()))
        if w_roll > 0:                                                                  # IMAGINATION multi-pas : vraies images jusqu'à t0 puis les siennes
            t0 = int(torch.randint(T // 4, T - 1, ()).item()); k = min(roll_k, T - 1 - t0)
            with torch.no_grad(): L = s.lnt(g.reshape(B * T1, 64, 64)).view(B, T1, NT, -1)
            zr = s.rollout(z, L, a, click, t0, k)
            l_roll = F.smooth_l1_loss(zr[:, t0:t0 + k + 1], L[:, t0 + 1:t0 + k + 2])
            l_proj = F.mse_loss(s.proj(L[:, :T].detach()), z.detach())                 # proj : latent cible -> ce que l'œil aurait vu
            tot = tot + w_roll * w_jepa * l_roll + l_proj; st.update(roll=float(l_roll.detach()), proj=float(l_proj.detach()))
        if w_crit_im > 0 and S is not None and hasattr(s, "crit"):                      # CRITIQUE SUR L'IMAGINÉ (27 oct., test de rang : sur les vraies images
            if snd is None and getattr(s, "hear", False) and getattr(s, "cur", None) is not None and s.cur[0].shape == (B, T):   # elle reconnaît le chemin
                snd = s.sound(g, s.cur[0], s.cur[1], None)                               # oracle à 1.00, sur les états IMAGINÉS elle tombe à 0.3 = hasard)
            t0 = int(torch.randint(T // 4, T - 1, ()).item()); k = min(roll_k, T - 1 - t0)
            with torch.no_grad():                                                        # le tronc n'apprend pas à plaire au juge : seule la tête critique bouge
                L = s.lnt(g.reshape(B * T1, 64, 64)).view(B, T1, NT, -1); s.rollout(s.eye(g[:, :T].reshape(B * T, 64, 64)).view(B, T, NT, -1), L, a, click, t0, k, snd)
                hid = s.pred.hid[:, t0:t0 + k + 1].detach()                              # pas t0 = image réelle + action ; après = images imaginées
            lo = s.crit(hid.mean(2)).float(); y1, y2 = s.crit_targets(S, ev); l_ci = 0
            for k_, y_ in ((0, y1[:, t0:t0 + k + 1]), (1, y2[:, t0:t0 + k + 1])):
                pw_ = ((1 - y_.mean()) / y_.mean().clamp_min(1e-3)).clamp(1, 100)
                l_ci = l_ci + F.binary_cross_entropy_with_logits(lo[..., k_], y_, pos_weight=pw_)
            if V is not None and hasattr(s, "crit_d"):
                l_ci = l_ci + F.binary_cross_entropy_with_logits(s.crit_d(hid.mean(2)).float()[..., 0], V[:, t0:t0 + k + 1].float())
            tot = tot + w_crit_im * l_ci; st.update(crit_im=float(l_ci.detach()))
        if ev is not None and w_ev > 0:
            e = ev.reshape(B * T, NE); has = ~torch.isnan(e[:, 0])
            if has.any():
                o, e = eo[has], e[has]; l_bin = F.binary_cross_entropy_with_logits(o[:, EV_BIN], e[:, EV_BIN])
                pm = e[:, EV_POS] >= 0; l_pos = F.smooth_l1_loss(o[:, EV_POS][pm], e[:, EV_POS][pm]) if pm.any() else o.sum() * 0
                l_prog = F.smooth_l1_loss(torch.tanh(o[:, EV_PROG]), e[:, EV_PROG]); tot = tot + w_ev * (l_bin + l_pos + l_prog)
                st.update(ev_bin=float(l_bin.detach()), ev_prog=float(l_prog.detach()))
        return tot, st
