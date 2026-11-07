"""
SYSTEM-2 — PLANIFICATEUR dans l'imagination du modèle du monde vidéo (model_seq.SeqWorldModel).

Idée (user, 25 oct. ; c'est aussi le « Mode 2 » de LeCun, A Path Towards Autonomous Machine Intelligence) : au lieu d'énumérer
tous les coups, on tient une SÉQUENCE d'actions imaginée et on la fait descendre le GRADIENT d'un coût prédit par le modèle.
  - actions ASSOUPLIES pendant la recherche : à chaque pas un MÉLANGE d'actions (softmax) et une position de clic CONTINUE
    (x, y) -> le modèle les reçoit via SeqPredictor.inject_soft (identique à l'entrée normale quand tout est « dur »)
  - plusieurs séquences de départ (variées : contre les minima locaux), affinées EN LOT par gradient, puis DURCIES et
    re-vérifiées dans l'imagination ; on joue la 1re action de la meilleure
  - NOTE d'un pas imaginé : + critique (fin de niveau proche, si la tête existe) + événement « level_complete » prédit
    + NOUVEAUTÉ (l'état imaginé est loin de ce que j'ai vécu dans ce niveau) + EFFET (l'action change quelque chose)
    + SOUVENIR (ressemble à un état qui a précédé une réussite) − DANGER (game over prédit)
Point de comparaison : CEM / tirs aléatoires (recherche par échantillons, actions dures).
Le déroulé imaginé : positions 0..C-1 = vraies transitions de la partie (mémoire), position C = l'image ACTUELLE + 1re action
planifiée, puis C+1..C+H-1 = images IMAGINÉES ; H passes du prédicteur bloc-causal = H pas d'imagination exacts.
"""
import time
import numpy as np
import torch, torch.nn.functional as F
from model import NA

EV_LC, EV_GO = 17, 18                                # événements prédits : « level_complete », « game_over »


def try_key(a, xy):                                  # « ce geste-là » : action, et pour un clic l'objet visé (case 2×2, comme les objets de v1)
    return (int(a), int(xy[1]) // 2, int(xy[0]) // 2) if int(a) == 6 and xy is not None else (int(a),)


def actor_prior(im, cands):                          # gk -> probabilité que l'expert fasse CE geste (clic : action 6 × case 8×8 visée)
    if getattr(im, "act", None) is None: return {}
    pa, pt = im.act; out = {}
    for a, xy in cands:
        p = float(pa[a]) if a < len(pa) else 0.0
        if a == 6 and xy is not None: p *= float(pt[(int(xy[1]) // 8) * 8 + int(xy[0]) // 8])
        out[try_key(a, xy)] = p
    return out


def actor_points(im, k=3):                          # CLICS « POINTS LIBRES » (27 oct., étude des 25 jeux : su15/r11l/tn36 cliquent dans le vide) :
    if getattr(im, "act", None) is None or getattr(im, "actf", None) is None: return []   # le pixel exact que l'acteur viserait dans ses k cases préférées
    pt = im.act[1]; out = []
    for c in torch.topk(pt, k).indices.tolist():
        f = int(im.actf[c].argmax()); out.append((int(c % 8) * 8 + f % 8, int(c // 8) * 8 + f // 8))
    return out


def pool(lat):                                       # (…,64,d) -> (…,d) normalisé : résumé d'un état pour nouveauté / souvenir
    return F.normalize(lat.float().mean(-2), dim=-1)


class Imagination:
    """déroulé imaginé (différentiable) de S séquences de H actions à partir du contexte réel ; renvoie la note de chaque séquence."""
    def __init__(s, m, H=4, gamma=0.9, w=None):
        s.m, s.H, s.gamma = m, H, gamma
        s.w = dict(crit=1.0, lc=1.0, nov=1.0, eff=0.2, mem=0.5, go=1.0, dream=3.0, dist=3.0) if w is None else w   # dream : actif seulement si le modèle sait rêver
        pix = torch.arange(64 * 64); s.py, s.px = (pix // 64).float(), (pix % 64).float()

    def context(s, frames, acts, clicks):            # frames (C+1,64,64) vraies images ; acts (C,), clicks (C,2)
        m = s.m; dev = next(m.parameters()).device
        fr = torch.as_tensor(np.asarray(frames), device=dev).long()
        with torch.no_grad():
            s.zc, s.Lc = m.eye(fr), m.lnt(fr)                                     # (C+1,64,d)
            C = fr.size(0) - 1
            s.injc = m.pred.inject(torch.as_tensor(np.asarray(acts), device=dev).long().view(1, C),
                                   torch.as_tensor(np.asarray(clicks), device=dev).long().view(1, C, 2)) if C > 0 else None
            s.sndC = None
            if getattr(m, "hear", False):                                            # le modèle ENTEND la partie : zones qui ont bougé + ding de fin de niveau
                dg = getattr(s, "ding", None); dg = dg if dg is not None and len(dg) == C + 1 else [0] * (C + 1)
                lc = torch.tensor(dg, device=dev, dtype=torch.float32)[None]
                snd = m.sound(fr[None], lc, torch.zeros_like(lc), None)[0]           # (C+1,64,d)
                if C > 0: s.injc = s.injc + snd[None, :C]
                s.sndC = snd[C]                                                      # ce qu'il entend MAINTENANT (ajouté au 1er pas imaginé)
            s.kv = m.pred.ctx_cache(s.zc[None, :C], s.injc if C > 0 else s.zc[None, :0])  # contexte réel : calculé UNE fois par coup
            s.act = None; s.dw = None
            if hasattr(m.pred.aemb, "tok") and C >= 0:                               # RÊVE : la suite qu'imagine un GAGNANT / un PERDANT
                z0, L0 = s.zc[C:C + 1][None], s.Lc[C:C + 1][None]; nc = torch.full((1, 1, 2), -1, device=dev, dtype=torch.long)
                add = 0 if s.sndC is None else s.sndC[None, None]
                if getattr(m, "intent", False):                                      # action « ? » + intention « succès bientôt » (1) / « non » (0)
                    q = m.pred.inject(torch.full((1, 1), NA, device=dev, dtype=torch.long), nc) + add
                    y = [L0 + m.pred.forward_cached(s.kv, z0, q + m.pred.s_rtg[r][None, None, None], C) for r in (1, 0)]
                else:                                                                # ancien rêve (dream_test.py) : jetons ?gagne / ?perd
                    y = [L0 + m.pred.forward_cached(s.kv, z0, m.pred.inject(torch.full((1, 1), tk, device=dev, dtype=torch.long), nc) + add, C) for tk in (NA, NA + 1)]
                s.dw, s.dl = y[0][0, 0].float(), y[1][0, 0].float(); s.dgap = ((s.dw - s.dl) ** 2).mean() + 1e-8
            if hasattr(m, "act_a"):                                                  # ACTEUR (Mode 1) : « ce que ferait l'expert » ici
                hp = m.pred.ctx_last if m.pred.ctx_last is not None else torch.zeros_like(s.zc[C:C + 1])
                la, lt, lf = m.actor(s.zc[C][None, None], hp[None])
                s.act = (torch.softmax(la[0, 0].float(), -1), torch.softmax(lt[0, 0].float(), -1))
                s.actf = torch.softmax(lf[0, 0].float(), -1)                                  # (64 cases, 64 pixels) : le pixel visé dans chaque case
        s.C = C; s.dev = dev; s.px, s.py = s.px.to(dev), s.py.to(dev)

    def pix_weights(s, xy, sig=1.5):                 # (S,H,2) positions continues -> poids sur les 4096 pixels
        d2 = (s.px - xy[..., :1]) ** 2 + (s.py - xy[..., 1:]) ** 2
        return torch.softmax(-d2 / (2 * sig ** 2), -1)

    def score(s, a_soft, xy, bank=None, succ=None):  # a_soft (S,H,NA), xy (S,H,2) -> note (S,) [différentiable], détails
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=s.dev.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8):
            return s._score(a_soft, xy, bank, succ)

    def _score(s, a_soft, xy, bank, succ):
        m, C, H = s.m, s.C, s.H; S = a_soft.size(0)
        pc = a_soft[..., 6]; inj_p = m.pred.inject_soft(a_soft, s.pix_weights(xy), pc)               # (S,H,64,d)
        if s.sndC is not None: inj_p = torch.cat([inj_p[:, :1] + s.sndC, inj_p[:, 1:]], 1)        # 1er pas imaginé : il entend ce qui vient de se passer
        z0 = s.zc[C:C + 1][None].expand(S, -1, -1, -1); L0 = s.Lc[C:C + 1][None].expand(S, -1, -1, -1)
        base, zin = L0, z0                              # position C = image ACTUELLE ; 0..C-1 (réelles) sont dans le cache s.kv
        for j in range(H):                                                                           # H passes = H pas imaginés
            zp = base + m.pred.forward_cached(s.kv, zin, inj_p[:, :j + 1], C)                       # positions C..C+j
            if j == H - 1: break
            im = zp[:, :j + 1]                                                                       # images imaginées C+1..C+j+1
            base = torch.cat([L0, im], 1); zin = torch.cat([z0, m.proj(im)], 1)
        imag = zp[:, :H]                                                                             # (S,H,64,d) latents imaginés
        hid = m.pred.hid[:, :H]                                                                      # état interne aux pas planifiés
        ev = torch.sigmoid(m.ev(torch.cat([zin[:, :H], hid], -1).mean(2)).float())                  # événements prédits (S,H,NE)
        prev = torch.cat([s.Lc[C:C + 1].expand(S, 1, -1, -1), imag[:, :-1]], 1)
        eff = (imag - prev).float().norm(dim=-1).mean(-1); eff = eff / (eff.detach().mean() + 1e-6)  # ampleur du changement
        v = pool(imag); terms = dict(lc=ev[..., EV_LC], go=ev[..., EV_GO], eff=eff.clamp(max=3))
        if hasattr(m, "crit"): terms["crit"] = torch.sigmoid(m.crit(hid.mean(2)).float()[..., 1])
        if getattr(m, "has_dist", False) and s.w.get("dist", 0): terms["dist"] = torch.sigmoid(m.crit_d(hid.mean(2)).float()[..., 0])   # pente vers le but, même loin
        if bank is not None and len(bank): terms["nov"] = 1 - torch.logsumexp(10 * v @ bank.T, -1) / 10 # loin du déjà-vécu
        if succ is not None and len(succ): terms["mem"] = torch.logsumexp(10 * v @ succ.T, -1) / 10  # près d'une réussite passée
        st = getattr(s, "succ_tok", None)                                            # SOUVENIR PAR JETONS (27 oct.) : les latents des zones modifiées pendant un niveau
        if st is not None and len(st) and s.w.get("memt", 0):                        # gagnant, cherchés N'IMPORTE OÙ dans l'image imaginée (indépendant de la disposition)
            sim = F.normalize(imag.float(), dim=-1) @ st.T                           # (S,H,64,K)
            terms["memt"] = sim.max(2).values.mean(-1)                               # chaque jeton gagnant retrouvé quelque part ?
        if s.dw is not None and s.w.get("dream", 0):                                                # 1er pas : plus près du rêve GAGNANT que du PERDANT
            y1 = imag[:, 0].float(); c = (((y1 - s.dl) ** 2).mean((1, 2)) - ((y1 - s.dw) ** 2).mean((1, 2))) / s.dgap
            terms["dream"] = torch.cat([c.clamp(-3, 3)[:, None], torch.zeros(S, H - 1, device=s.dev)], 1)
        g = s.gamma ** torch.arange(H, device=s.dev, dtype=torch.float32)
        tot = sum((-1 if k == "go" else 1) * s.w.get(k, 0) * t for k, t in terms.items())          # (S,H)
        return (tot * g).sum(1), terms


class GradPlanner:
    """descente de gradient sur des séquences d'actions assouplies (plusieurs départs), puis durcissement + vérification."""
    def __init__(s, m, H=4, seeds=8, iters=12, lr=0.4, lr_xy=3.0, temp=1.0, gamma=0.9, w=None, w_tried=2.0, w_actor=3.0, shoot=0):
        s.im = Imagination(m, H, gamma, w); s.H, s.S, s.iters, s.lr, s.lr_xy, s.temp = H, seeds, iters, lr, lr_xy, temp; s.w_actor = w_actor
        s.shoot = shoot                                                              # TIR AU HASARD (27 oct.) : + N suites tirées parmi les gestes candidats, notées en 1 passe
        s.w_tried = w_tried                                                          # pénalité : 1er geste déjà essayé dans cet état réel
        s.prev = None                                                                # plan précédent (départ « chaud », comme un MPC)

    def plan(s, frames, acts, clicks, avail, objs, bank=None, succ=None, rng=np.random, tried=(), bonus=None, fast=False):
        im = s.im; im.context(frames, acts, clicks); H, S, dev = s.H, s.S, im.dev
        mask = torch.full((NA,), -1e4, device=dev); mask[[a for a in avail if 0 < a < NA]] = 0
        obj_xy = [(x, y) for (_c, _n, (y, x)) in objs] or [(32, 32)]
        if 6 in avail: obj_xy += [p for p in actor_points(im) if p not in obj_xy]           # + les pixels que viserait l'acteur
        cands = [(a, None) for a in avail if 0 < a < NA and a != 6] + ([(6, xy) for xy in obj_xy] if 6 in avail else [])
        new = [c for c in cands if try_key(*c) not in tried] or cands               # gestes PAS ENCORE ESSAYÉS dans cet état réel
        best = [c for c in cands if (bonus or {}).get(try_key(*c), 0) > 0]           # carte mentale : 1er pas vers la frontière de l'inconnu
        pri = actor_prior(im, cands)                                                # acteur : les gestes favoris de l'expert -> départs + note
        if pri:
            bonus = dict(bonus or {}); top = sorted(cands, key=lambda c: -pri[try_key(*c)])[:3]
            for c in cands: bonus[try_key(*c)] = bonus.get(try_key(*c), 0.0) + s.w_actor * pri[try_key(*c)]
            best = best[:2] + [c for c in top if c not in best[:2]]
        if fast and pri:                                                            # acteur HÉSITANT (meilleur geste < 30 % de la masse) -> Système 2
            q = np.array([pri[try_key(*c)] for c in cands]); fast = q.max() >= getattr(s, "conf", 0.3) * max(q.sum(), 1e-9)
        if fast:                                                                    # SYSTÈME 1 (idée user) : l'acteur + la carte décident en UN passage,
            sc = np.array([(bonus or {}).get(try_key(*c), 0.0) + 0.5 * (try_key(*c) not in tried) for c in cands]) + 1e-3 * rng.rand(len(cands))
            if pri:                                                                 # ACTEUR CONFIANT (≥ 50 % de la masse) : il prime sur la carte (test 27 oct. :
                q = np.array([pri[try_key(*c)] for c in cands]); i = int(q.argmax())   # 55 % de gestes oracle justes, mais la carte l'écrasait),
                if q[i] >= getattr(s, "conf_hi", 0.5) * max(q.sum(), 1e-9) and (bonus or {}).get(try_key(*cands[i]), 0.0) > -1.4:   # sauf geste connu inutile / mortel ici
                    sc[i] = sc.max() + 1.0
            a0, xy0 = cands[int(sc.argmax())]; s.prev = None                         # sans imagination ni gradient (~10× plus rapide)
            return (a0, tuple(int(v) for v in xy0) if a0 == 6 and xy0 is not None else None), dict(score=float(sc.max()), fast=1.0)
        L0 = torch.randn(S, H, NA, device=dev) * 0.5; XY0 = torch.zeros(S, H, 2, device=dev)
        for i in range(S):                                                          # départs variés : 1er geste non essayé, puis au hasard
            for h in range(H):
                a, xy = (best[i] if (h == 0 and i < len(best)) else new[rng.randint(len(new))]) if h == 0 else (int(rng.choice([a for a in avail if 0 < a < NA])), None)
                L0[i, h, a] += 3
                XY0[i, h] = torch.tensor(xy or obj_xy[rng.randint(len(obj_xy))], dtype=torch.float32)
        if s.prev is not None:                                                      # départ chaud : le plan d'avant, décalé d'un pas
            L0[0, :-1], XY0[0, :-1] = s.prev[0][1:], s.prev[1][1:]
        logits = L0.clone().requires_grad_(True); u = torch.logit((XY0 / 63).clamp(0.02, 0.98)).requires_grad_(True)
        opt = torch.optim.Adam([{"params": [logits], "lr": s.lr}, {"params": [u], "lr": s.lr_xy / 63 * 4}])
        for _ in range(s.iters):                                                    # le gradient améliore les séquences
            a_soft = torch.softmax((logits + mask) / s.temp, -1); xy = 63 * torch.sigmoid(u)
            sc, _ = im.score(a_soft, xy, bank, succ); loss = -sc.sum()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():                                                       # DURCIR puis re-vérifier
            a_hard = (logits + mask).argmax(-1); xy = 63 * torch.sigmoid(u)
            O = torch.tensor(obj_xy, device=dev, dtype=torch.float32)               # le clic s'ACCROCHE au pixel d'objet le plus proche
            xy = O[((xy[..., None, :] - O) ** 2).sum(-1).argmin(-1)]                 # (un clic dans le vide ne fait rien)
            if s.shoot:                                                             # (test de rang : la bonne suite sort 1re parmi 200 tirées au hasard,
                ci = torch.as_tensor(rng.randint(len(cands), size=(s.shoot, H)), device=dev)   # mais la descente de gradient à 8 départs ne la trouve pas)
                ca = torch.tensor([c[0] for c in cands], device=dev); cxy = torch.tensor([c[1] or (32, 32) for c in cands], device=dev, dtype=torch.float32)
                a_hard = torch.cat([a_hard, ca[ci]]); xy = torch.cat([xy, cxy[ci]])
            S = a_hard.size(0)
            sc, terms = im.score(F.one_hot(a_hard, NA).float(), xy, bank, succ)
            if bonus is not None:                                                   # CARTE MENTALE : oriente la note (1er geste)
                sc = sc + torch.tensor([bonus.get(try_key(a_hard[i, 0], xy[i, 0]), 0.0) for i in range(S)], device=dev).float()
            else:
                sc = sc - s.w_tried * torch.tensor([try_key(a_hard[i, 0], xy[i, 0]) in tried for i in range(S)], device=dev).float()
            b = int(sc.argmax())
            s.prev = (logits[b].detach(), (63 * torch.sigmoid(u[b])).detach()) if b < s.S else (F.one_hot(a_hard[b], NA).float() * 3, xy[b].detach())
        a0 = int(a_hard[b, 0]); x, y = int(xy[b, 0, 0]), int(xy[b, 0, 1])
        return (a0, (x, y) if a0 == 6 else None), dict(score=float(sc[b]), seq=a_hard[b].tolist(), **{k: float(t[b].mean()) for k, t in terms.items()})


class CEMPlanner:
    """point de comparaison : recherche par ÉCHANTILLONS (actions dures, clics sur objets) ; quelques tours de CEM."""
    def __init__(s, m, H=4, n=64, rounds=2, elite=8, gamma=0.9, w=None, w_tried=2.0, w_actor=3.0):
        s.im = Imagination(m, H, gamma, w); s.H, s.n, s.rounds, s.elite, s.w_tried, s.w_actor = H, n, rounds, elite, w_tried, w_actor

    def plan(s, frames, acts, clicks, avail, objs, bank=None, succ=None, rng=np.random, tried=(), bonus=None):
        im = s.im; im.context(frames, acts, clicks); H, dev = s.H, im.dev
        cands = [(a, None) for a in avail if 0 < a < NA and a != 6]
        if 6 in avail: cands += [(6, (x, y)) for (_c, _n, (y, x)) in objs[:24]] or [(6, (32, 32))]
        K = len(cands); p = np.full((H, K), 1.0 / K); best = None
        pri = actor_prior(im, cands)
        if pri:                                                                     # acteur : échantillonner d'abord ce que ferait l'expert
            bonus = dict(bonus or {}); q = np.array([pri[try_key(*c)] for c in cands]) + 1e-3; p[0] = 0.5 * p[0] + 0.5 * q / q.sum()
            for c in cands: bonus[try_key(*c)] = bonus.get(try_key(*c), 0.0) + s.w_actor * pri[try_key(*c)]
        with torch.no_grad():
            for _ in range(s.rounds):
                idx = np.stack([[rng.choice(K, p=p[h]) for h in range(H)] for _ in range(s.n)])        # (n,H)
                a = torch.tensor([[cands[k][0] for k in row] for row in idx], device=dev)
                xy = torch.tensor([[cands[k][1] or (32, 32) for k in row] for row in idx], device=dev, dtype=torch.float32)
                sc, terms = im.score(F.one_hot(a, NA).float(), xy, bank, succ)
                if bonus is not None: sc = sc + torch.tensor([bonus.get(try_key(*cands[row[0]]), 0.0) for row in idx], device=dev).float()
                else: sc = sc - s.w_tried * torch.tensor([try_key(*cands[row[0]]) in tried for row in idx], device=dev).float()
                el = idx[sc.topk(min(s.elite, s.n)).indices.cpu().numpy()]
                p = 0.3 * p + 0.7 * np.stack([np.bincount(el[:, h], minlength=K) / len(el) for h in range(H)])
                b = int(sc.argmax())
                if best is None or float(sc[b]) > best[0]: best = (float(sc[b]), idx[b], {k: float(t[b].mean()) for k, t in terms.items()})
        a0, c0 = cands[best[1][0]]
        return (a0, c0), dict(score=best[0], seq=[cands[k][0] for k in best[1]], **best[2])
