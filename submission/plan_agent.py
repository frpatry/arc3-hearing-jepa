"""
AGENT SYSTEM-2 : à chaque pas, il PLANIFIE dans l'imagination du modèle du monde vidéo (wm/planner.py) et joue la 1re action
du meilleur plan. Pas de graphe d'exploration v1 : l'exploration vient de la NOUVEAUTÉ imaginée (loin de ce que j'ai vécu
dans ce niveau), le but de la CRITIQUE / de l'événement « fin de niveau » prédit, et du SOUVENIR des réussites (états qui ont
précédé une fin de niveau dans CETTE partie : sous-buts découverts, pas codés). Le modèle continue d'apprendre en jouant.
Interface : act(grid, state_name, levels_completed, available_actions) -> (action_id, data) (comme Explorer).
"""
import os, sys, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.join(HERE, "wm")]
from explorer import components                      # objets = composantes connexes (pour les départs / candidats de clic)


def click_candidates(g, cap=96):
    """CANDIDATS DE CLIC (corrigé le 26 oct., bilan des jeux ratés : l'ancienne liste = les 48 PLUS PETITS objets seulement ->
    plateau / bouton / blocs jamais proposés sur ft09, tn36, bp35). Tous les objets, regroupés par TYPE (couleur, taille ~ log2) ;
    tour à tour un exemplaire de chaque type, les types RARES d'abord (un bouton unique avant 190 points de roche identiques),
    exemplaires d'un même type pris en ordre étalé (pas seulement les premiers du balayage) ; au plus `cap` clics."""
    objs = components(g, max_n=10 ** 6)
    groups = {}
    for o in objs: groups.setdefault((o[0], int(np.log2(max(o[1], 1)))), []).append(o)
    order = sorted(groups, key=lambda t: (len(groups[t]), -max(o[1] for o in groups[t])))   # rares d'abord, puis les plus gros
    spread = {t: [groups[t][i] for i in np.argsort([(i * 0.618034) % 1 for i in range(len(groups[t]))], kind="stable")] for t in order}
    out, k = [], 0
    while len(out) < cap and any(len(spread[t]) > k for t in order):
        out += [spread[t][k] for t in order if len(spread[t]) > k]; k += 1
    return out[:cap]
from model_seq import SeqWorldModel
from planner import GradPlanner, CEMPlanner, pool, try_key
from cogmap import CogMap
from model import NA


class PlanAgent:
    def __init__(self, ckpt, seed=0, kind="grad", H=4, device=None, ttt_every=32, ttt_steps=64, ttt_lr=1e-4, mem=2048, win_steps=192, sys1=False, slow_after=8, focus=False, w_map=1.0, **kw):
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu"); self.rng = np.random.RandomState(seed)
        ck = torch.load(ckpt, map_location=self.dev, weights_only=False)
        self.m = SeqWorldModel(ck["d"], ck["T"], ck["L"], ck.get("tgt_local", False), ck.get("head_src", "zp"), ck.get("head", "px"),
                               crit=ck.get("crit", False), actor=ck.get("actor", False), dream=ck.get("dream", False)).to(self.dev)
        self.m.load_state_dict(ck["m"], strict=False); self.m.eval()
        self.m.has_dist = any(k.startswith("crit_d.") for k in ck["m"])                 # tête DISTANCE AU BUT présente dans ce modèle ?
        self.m.hear, self.m.intent, self.m.cur = ck.get("hear", False), ck.get("intent", False), None   # modèle qui ENTEND / qui RÊVE avec intention
        for p in self.m.parameters(): p.requires_grad_(False)                       # le planificateur ne dérive que les ACTIONS
        self.H = H; self.C = self.m.T - H                                           # contexte réel = T - H pas
        self.pl = (GradPlanner if kind == "grad" else CEMPlanner)(self.m, H=H, **kw)
        self.ttt = (ttt_every, ttt_steps); self.opt = None; self.ttt_lr = ttt_lr
        # INTÉRIORISER (idée user) : 64 pas d'apprentissage toutes les 32 actions (= ×16). Mesuré le 25 oct. sur les 5 officiels inédits
        # (eval_ttt --mults) : ×16 meilleur ou égal partout (tr87 F1 à 50 actions 0.36 ×1 -> 0.59 ×4 -> 0.81 ×16 ; lf52 à 500 : 0.51 -> 0.81) ;
        # ×64 guère mieux, 4× plus cher, abîme vc33 au début
        self.ntr = 0; self.mem = mem
        self.vid, self.segs = [], []                                                # morceau de partie en cours + morceaux PASSÉS (gardés après un GAME OVER)
        self.bank, self.succ = [], []                                               # états vécus (niveau), réussites
        self.map = CogMap(); self.kprev = None; self.gprev = None                   # CARTE MENTALE : états réels (HUD masqué) × gestes -> résultat
        self.sys1, self.slow_after, self.since_new, self.n_fast, self.lvl_steps = sys1, slow_after, 0, 0, 0   # SYSTÈME 1 par défaut, SYSTÈME 2 si bloqué (rien de nouveau depuis slow_after coups)
        self.kstat = {}; self.kind_prev = None; self.w_dead = 1.5                  # SORTES DE GESTES (flèche, clic sur telle couleur) : effet ou rien, TOUS états confondus
        self.demos = []                                                             # RÉUSSITES de CETTE partie, rejouées comme des démos d'expert
        self.win_steps = win_steps                                                  # INTÉRIORISER LA VICTOIRE : pas d'apprentissage juste après un niveau réussi
        self.prev = None; self.last = None; self.levels = 0; self.n = 0; self.t_plan = 0.0; self.info = None
        # FOCUS (27 oct., films vc33 : au niveau 2, 100 clics sur ~33 cases, l'écran ne bouge JAMAIS, dont le fond blanc / gris) :
        # 1) ne plus recliquer une case 8×8 restée sans effet dans CET écran ; après 3 coups sans effet, ne viser que les PETITS objets ;
        # 2) RECETTE : le genre d'objet (couleur, taille) des clics qui ont eu un effet dans un niveau GAGNÉ est favorisé au niveau suivant
        self.w_map = w_map                                                          # CARTE MENTALE RENFORCÉE (27 oct.) : poids des bonus de la carte (frontière / déjà essayé)
        self.focus = focus; self.dead = set(); self.noeff = 0; self.desc_prev = None; self.eff_desc = []; self.win_desc = []

    def _lat(self, g):
        with torch.no_grad():
            return pool(self.m.lnt(torch.as_tensor(g, device=self.dev).long()[None]))[0]

    def act(self, grid, state, levels, avail, anim=False):                        # anim : l'écran a bougé PENDANT l'animation du dernier geste (image finale identique ou non)
        g = np.asarray(grid, dtype=np.int8)                                         # 8 bits (couleurs 0..15) : mémoire ÷8 (jeux cachés tous en parallèle)
        if state in ("NOT_PLAYED", "GAME_OVER") or g.ndim != 2 or g.shape != (64, 64):
            if state == "GAME_OVER" and self.kprev is not None: self.map.record(self.kprev, self.gprev, "GO")
            self.prev = None; self.last = None; self.kprev = None; return 0, None    # RESET (gratuit)
        ding = int(levels > self.levels)                                            # le « ding » de fin de niveau (entendu par le modèle)
        if self.focus and self.prev is not None and self.last is not None:          # FOCUS : ce clic a-t-il eu un effet ? (le clic gagnant compte)
            eff = bool(ding) or bool(anim) or self.kprev is None or self.map.key(g) != self.kprev   # clé de la carte = écran sans le HUD
            if eff:
                self.noeff = 0
                if self.desc_prev is not None: self.eff_desc.append(self.desc_prev)
            else:
                self.noeff += 1
                if self.last[0] == 6 and self.last[1] is not None and self.kprev is not None: self.dead.add((self.kprev, int(self.last[1][1]) // 8, int(self.last[1][0]) // 8))
            if ding: self.win_desc = (self.eff_desc[-8:] or self.win_desc); self.eff_desc = []; self.dead = set(); self.noeff = 0
        if levels > self.levels:                                                    # NIVEAU RÉUSSI : souvenir de ce qui a précédé
            if self.prev is not None:
                self.succ.append(self._lat(self.prev))
                if getattr(self, "lvl_first", None) is not None:                     # jetons des cases 8×8 modifiées pendant ce niveau gagnant
                    ch = (self.prev != self.lvl_first).reshape(8, 8, 8, 8).any((1, 3)).reshape(-1)
                    if ch.any():
                        with torch.no_grad(): tk = torch.nn.functional.normalize(self.m.lnt(torch.as_tensor(self.prev, device=self.dev).long()[None])[0].float(), dim=-1)
                        self.succ_tok = torch.cat([getattr(self, "succ_tok", tk[:0]), tk[torch.as_tensor(ch, device=self.dev)]])[-256:]
            self.remember_win(g)                                                    # + le plus court chemin gagnant devient une démo
            self.levels = levels; self.bank = []; self.map.reset(); self.kprev = None; self.lvl_steps = 0
            if hasattr(self.pl, "prev"): self.pl.prev = None
        if self.prev is not None and self.last is not None:                         # la transition qui vient d'avoir lieu
            a, c = self.last; tr = (self.prev, a, c if c is not None else (-1, -1), g, ding)
            if self.vid and not np.array_equal(self.vid[-1][3], tr[0]):            # discontinuité (GAME OVER, reprise) : le morceau est GARDÉ
                if len(self.vid) >= 2: self.segs.append(self.vid)
                self.vid = []
            self.vid.append(tr); self.ntr += 1
            while sum(len(v) for v in self.segs) + len(self.vid) > self.mem:         # mémoire bornée : on oublie les plus vieux morceaux
                if self.segs: self.segs.pop(0)
                else: self.vid = self.vid[-self.mem:]
            if self.ttt[0] and self.ntr % self.ttt[0] == 0: self.finetune()
        ctx = self.vid[-self.C:] if self.vid and np.array_equal(self.vid[-1][3], g) else []
        frames = [t[0] for t in ctx] + [g]; acts = [t[1] for t in ctx]; clicks = [t[2] for t in ctx]
        self.pl.im.ding = [0] + [t[4] if len(t) > 4 else 0 for t in ctx]            # ding entendu à chaque image du contexte
        if getattr(self, "lvl_first", None) is None or self.lvl_steps == 0: self.lvl_first = g.copy()   # 1er écran du niveau (souvenir par jetons)
        self.pl.im.succ_tok = getattr(self, "succ_tok", None)
        objs = click_candidates(g.astype(np.int16)) if 6 in avail else []
        self.map.observe(self.prev, g); k = self.map.key(g); self.since_new = 0 if k not in self.map.G else self.since_new + 1   # carte mentale : où suis-je, qu'ai-je déjà essayé ici
        if 6 in avail:                                                              # 5 CLICS « POINTS LIBRES » au hasard par écran (idée user : en gaspiller quelques-uns peut payer)
            rs = np.random.RandomState(int(k[:8], 16)); objs = objs + [(-1, 0, (int(y_), int(x_))) for y_, x_ in rs.randint(0, 64, (5, 2))]
        if self.focus and 6 in avail and objs:                                      # FOCUS 1 : pas de clic déjà vain ici ; après 3 coups vains, petits objets seulement
            keep = [o for o in objs if (k, o[2][0] // 8, o[2][1] // 8) not in self.dead]
            if self.noeff >= 3: keep = [o for o in keep if o[0] >= 0 and 0 < o[1] <= 64] or keep
            objs = keep or objs
        self.map.visit(k, [try_key(a_, None) for a_ in avail if 0 < a_ < NA and a_ != 6] + [try_key(6, (x_, y_)) for (_c, _n, (y_, x_)) in objs], g)
        if self.kprev is not None: self.map.record(self.kprev, self.gprev, "ANIM" if (anim and k == self.kprev) else k)   # effet seulement animé : pas « rien »
        if self.kprev is not None and self.kind_prev is not None:                  # ce geste a-t-il changé l'écran (HUD masqué) ?
            st = self.kstat.setdefault(self.kind_prev, [0, 0]); st[0] += 1; st[1] += int(k != self.kprev or bool(anim))
        tried = {gk for gk, v in self.map.G[k].items() if v is not None}; bonus = self.map.bonus(k)
        if self.w_map != 1.0: bonus = {gk: v * self.w_map for gk, v in bonus.items()}  # la carte (aller vers l'inconnu) pèse plus que la note plate
        for gk, kd in [(try_key(a_, None), (a_,)) for a_ in avail if 0 < a_ < NA and a_ != 6] + [(try_key(6, (x_, y_)), (6, int(g[y_, x_]), y_ // 16, x_ // 16)) for (_c, _n, (y_, x_)) in objs]:
            n_, e_ = self.kstat.get(kd, (0, 0))                                      # sorte de geste qui ne fait (presque) jamais rien, où qu'on soit : malus
            if n_ >= 3: bonus[gk] = bonus.get(gk, 0.0) - self.w_dead * (1 - e_ / n_)   # (ft09 26 oct. : 42 % des clics sur des couleurs inertes)
        if self.focus and self.win_desc and 6 in avail:                            # FOCUS 2 : RECETTE du niveau gagné (même taille d'objet ; même couleur)
            ws = {(c_, int(np.log2(max(n_, 1)))) for c_, n_ in self.win_desc if n_}; sz = {s_ for _, s_ in ws}; co = {c_ for c_, _ in self.win_desc}
            for (c_, n_, (y_, x_)) in objs:
                if c_ < 0: continue
                s_ = int(np.log2(max(n_, 1))); gk = try_key(6, (x_, y_))
                r_ = 2.0 if (c_, s_) in ws else 1.2 if s_ in sz else 0.6 if c_ in co else 0.0
                bonus[gk] = bonus.get(gk, 0.0) + r_ * (self.w_map if gk not in tried else 1.0)   # bouton de la recette JAMAIS essayé dans cet état : exploration systématique
        bank = torch.stack(self.bank[-512:]) if self.bank else None; succ = torch.stack(self.succ) if self.succ else None
        t = time.time()
        with torch.enable_grad():
            fast = self.sys1 and self.since_new < self.slow_after and self.lvl_steps >= 12 and hasattr(self.pl, "S")   # PLANIFICATEUR aux 12 premiers coups de chaque niveau
            (a, c), self.info = self.pl.plan(frames, acts, clicks, list(avail), objs, bank, succ, self.rng, tried, bonus, **({"fast": True} if fast else {}))
        self.kprev, self.gprev = k, try_key(a, c); self.n_fast += int((self.info or {}).get("fast", 0)); self.lvl_steps += 1
        yy, xx = (int(np.clip(c[1], 0, 63)), int(np.clip(c[0], 0, 63))) if a == 6 and c is not None else (0, 0)
        self.kind_prev = (6, int(g[yy, xx]), yy // 16, xx // 16) if a == 6 and c is not None else (int(a),)
        if self.focus:                                                              # genre de l'objet cliqué (couleur, taille) pour la recette
            od = {(x_, y_): (c_, n_) for (c_, n_, (y_, x_)) in objs if c_ >= 0}
            self.desc_prev = (od.get((int(c[0]), int(c[1])), (int(g[yy, xx]), 0))) if a == 6 and c is not None else None   # sorte = couleur × RÉGION (ft09 : exemples inertes ≠ plateau jouable)
        self.t_plan += time.time() - t; self.n += 1
        self.bank.append(self._lat(g)); self.prev = g; self.last = (a, c)
        return a, ({"x": int(c[0]), "y": int(c[1])} if a == 6 and c is not None else None)

    def remember_win(self, g_after):                 # APPRENDRE CE QU'EST GAGNER DANS CE JEU : plus court chemin connu (carte) jusqu'au geste gagnant
        M = self.map; path = M.path_between(M.start, self.kprev)
        if path is not None and self.kprev in M.F:
            steps = path + [(self.kprev, self.gprev)]; ep = []
            for j, (k, gk) in enumerate(steps):
                nxt = M.F.get(steps[j + 1][0]) if j + 1 < len(steps) else g_after
                if k not in M.F or nxt is None: ep = []; break
                a = int(gk[0]); c = (2 * gk[2], 2 * gk[1]) if a == 6 and len(gk) == 3 else (-1, -1)
                ep.append((M.F[k], a, c, nxt))
        else: ep = []
        if not ep and len(self.vid) >= 2: ep = list(self.vid[-min(16, len(self.vid)):])   # repli : les derniers pas réels avant la réussite
        if len(ep) >= 2:
            self.demos = (self.demos + [ep])[-3:]
            if self.win_steps: self.finetune(n_exp=self.win_steps // 2, n_demo=self.win_steps)   # « on pense, on pense, puis on joue » (idée user) : au moment où
                                                                                    # l'agent tient enfin une démo du VRAI jeu, il l'apprend avant le niveau suivant

    def _batch(self, W):
        g = torch.tensor(np.stack([np.stack([e[0] for e in w] + [w[-1][3]]) for w in W]), device=self.dev).long()
        a = torch.tensor([[e[1] for e in w] for w in W], device=self.dev).long(); c = torch.tensor([[list(e[2]) for e in w] for w in W], device=self.dev).long()
        return g, a, c

    def finetune(self, n_exp=None, n_demo=16):         # apprendre EN JOUANT sur TOUTE l'expérience de CETTE partie (comme le bébé) + les réussites
        S = [v for v in self.segs + [self.vid] if len(v) >= 2]
        if not S and not self.demos: return
        ps = [p for n, p in self.m.named_parameters() if not n.startswith("tgt")]
        for p in ps: p.requires_grad_(True)
        if self.opt is None: self.opt = torch.optim.AdamW(ps, self.ttt_lr)
        self.m.train()
        bf = torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev == "cuda" and torch.cuda.get_device_capability()[0] >= 8)   # mémoire ÷2 (27 oct. : 6 Go/partie -> OOM à 25 parties)
        if S:
            k = min(self.m.T, max(len(v) for v in S)); S = [v for v in S if len(v) >= k]
            wts = np.array([len(v) - k + 1 for v in S], dtype=float); wts /= wts.sum()
            for _ in range(self.ttt[1] if n_exp is None else n_exp):
                W = []
                for j in np.random.choice(len(S), size=8, p=wts):
                    i = np.random.randint(0, len(S[j]) - k + 1); W.append(S[j][i:i + k])
                g, a, c = self._batch(W)
                with bf: l, _ = self.m.loss(g, a, c)
                self.opt.zero_grad(); l.backward(); self.opt.step(); self.m.ema()
        for _ in range(n_demo if self.demos else 0):                                # RÉUSSITES : l'acteur imite le chemin gagnant, le critique apprend la victoire
            ep = self.demos[np.random.randint(len(self.demos))]; k2 = min(self.m.T, len(ep))
            st = np.random.randint(0, len(ep) - k2 + 1, size=8); g, a, c = self._batch([ep[i:i + k2] for i in st])
            S1 = torch.zeros_like(a); S1[:, -1] = torch.tensor([int(i + k2 == len(ep)) for i in st], device=self.dev)   # 1 = cette action finit le niveau
            with bf: l, _ = self.m.loss(g, a, c, None, S=S1, w_crit=1.0 if hasattr(self.m, "crit") else 0.0, Bh=torch.zeros_like(a), w_act=1.0 if hasattr(self.m, "act_a") else 0.0)
            self.opt.zero_grad(); l.backward(); self.opt.step(); self.m.ema()
        for p in ps: p.requires_grad_(False)                                        # le planificateur ne dérive que les actions
        self.m.eval()
        if self.dev == "cuda": torch.cuda.empty_cache()                            # RENDRE la mémoire de l'apprentissage (sinon les autres jeux la croient prise)

    def ms_per_action(self):
        return 1000 * self.t_plan / max(self.n, 1)
