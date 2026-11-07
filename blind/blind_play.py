"""JOUER LES JEUX AVEC LE MODÈLE AVEUGLE (1er nov., go user : « sans image c'est rapide, faisons jouer les jeux »).
Base = explorateur v1 (explorer.py : graphe, curiosité, retour vers l'inconnu ; il propose les gestes candidats de chaque état).
Le modèle aveugle ENTEND la partie en direct : chaque pas réel -> mêmes jetons qu'à l'entraînement (blind_data.step_feats :
objets qui ont changé, sons fabriqués + habituation, ding = niveau franchi, buzz = perdu), mémoire = toute la partie (≤ 255 pas).
À chaque pas, il imagine le SON de chaque geste candidat (avant de le jouer) ; s'il croit qu'un geste fait DING (seuil absolu,
ou nettement au-dessus de ce qu'il entend d'habitude dans ce niveau), il le joue ; sinon v1 continue.
APPRENTISSAGE PENDANT LA PARTIE (« +appr ») : tous les N pas, quelques pas de gradient (mêmes pertes JEPA + sons) sur les pas de
CETTE partie seulement -> au niveau 2, il a entendu le ding du niveau 1 : reconnaît-il ce qui fait gagner ?
Mesures : niveaux par jeu (5 OFFICIELS EXCLUS), et à chaque vrai ding : la note « ding » qu'il donnait au geste gagnant
(rang parmi les candidats) — au niveau 1 (rien entendu) vs aux niveaux suivants (ding déjà entendu).
    python blind_play.py --ckpt blind_long.pt --env data/environment_files --games g50t,ka59,lf52,r11l,sk48 --l_budget 300 --seeds 2"""
import argparse, copy, json, logging, os, sys, time
from argparse import Namespace
from collections import deque
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; sys.path[:0] = [str(HERE)]
os.environ["OPERATION_MODE"] = "offline"
from explorer import Explorer  # noqa: E402
from blind_data import step_feats, scene, K, NF, SND  # noqa: E402
from blind_jepa import Blind, BlindLong, batch, loss_fn  # noqa: E402
from rel_agent import RelAgent, RelSolo  # noqa: E402
DING = SND.index("ding")


def load_model(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False); a = Namespace(**ck["args"])
    late = ck["args"].get("late", "w_inv" in ck["args"])                            # ancien prédicteur (geste en entrée) si absent
    m = (BlindLong(a.d, a.L, 3, a.T, late=late, eye=bool(ck["args"].get("eye", 0))) if a.arch == "long" else Blind(a.d, a.L, 3, a.T, late=late)).to(dev)
    m.load_state_dict(ck["m"]); m.eval(); return m, a


class BlindAgent(Explorer):
    def __init__(s, m0, args, dev, seed=0, fire=True, ttt=False, tau=0.5, rel=4.0, ttt_every=8, ttt_steps=4, lr=1e-4, listen=False, hist=0):
        super().__init__(seed); s.dev, s.args, s.fire, s.ttt, s.tau, s.rel, s.listen, s.hist = dev, args, fire, ttt, tau, rel, listen, hist
        s.m = copy.deepcopy(m0) if ttt else m0; s.T = s.m.T
        if ttt:
            s.tgt = copy.deepcopy(s.m).requires_grad_(False); s.opt = torch.optim.AdamW(s.m.parameters(), lr, weight_decay=0.0)
            s.ttt_every, s.ttt_steps = ttt_every, ttt_steps; s.n_upd = 0
        s.H = dict(OBJ=[], COL=[], NO=[], ACT=[], CLK=[], SND=[], **(dict(SCN=[], SCOL=[], NS=[]) if getattr(s.m, "eye", False) else {})); s.rec = []; s.pv = None
        s.lvl_scores = []; s.fired = set(); s.n_fire = 0; s.pd_last = None; s.pd_log = []

    # ---------- entendre le pas réel qui vient d'avoir lieu
    def hear(s, g, state, levels):
        pg, pa, pd, plv = s.pv
        ding, buzz = levels > plv, state == "GAME_OVER"
        o, c, n, snd, s.rec = step_feats(pg.astype(np.int64), g.astype(np.int64), s.rec, ding, buzz)
        for k, v in zip(("OBJ", "COL", "NO", "ACT", "CLK", "SND"), (o, c, n, pa, (pd["x"], pd["y"]) if pd else (-1, -1), snd)): s.H[k].append(v)
        if "SCN" in s.H:
            f, sc, ns = scene(g.astype(np.int64)); s.H["SCN"].append(f); s.H["SCOL"].append(sc); s.H["NS"].append(ns)
        if ding: s.lvl_scores = []
        if s.ttt and len(s.H["ACT"]) >= 16 and len(s.H["ACT"]) % s.ttt_every == 0: s.learn()
        return ding

    def D(s):
        n = len(s.H["ACT"]); D = {k: np.array(v) for k, v in s.H.items()}
        D["ACT"] = D["ACT"].astype(np.int64); D["CLK"] = D["CLK"].astype(np.int64)
        D["LAB"] = np.full((n, 11), -1, np.int8); D["REM"] = n - np.arange(n); return D

    def observe(s, grid, state, levels, a_id, data):  # TUTORIEL : on regarde jouer (solution oracle), on entend, on ne choisit pas
        g = np.asarray(grid, dtype=np.int16)
        if s.pv is not None: s.hear(g, state, levels)
        s.pv = (g, int(a_id), data, levels)

    def learn(s, steps=None):                         # APPRENTISSAGE PENDANT LA PARTIE (pas de cette partie seulement)
        D = s.D(); n = len(D["ACT"]); T = min(s.T, n); s.m.train()
        for _ in range(steps or s.ttt_steps):
            st = np.where(np.random.rand(4) < 0.5, 0, np.random.randint(0, max(1, n - 8), 4))
            loss, _ = loss_fn(s.m, s.tgt, batch(D, st, T, s.dev), s.args, s.dev)
            s.opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(s.m.parameters(), 1.0); s.opt.step()
            with torch.no_grad():
                for q, w in zip(s.tgt.parameters(), s.m.parameters()): q.mul_(0.99).add_(w.detach(), alpha=0.01)
            s.n_upd += 1
        s.m.eval()

    # ---------- imaginer le son de chaque geste candidat
    @torch.no_grad()
    def imagine(s, cs):
        D = s.D(); n = len(D["ACT"]); L = min(n, s.T - 1, s.hist or 10 ** 9); i0 = n - L; M = len(cs)        # hist : mémoire limitée aux derniers pas
        b = batch(D, np.array([i0]), L, s.dev)                                     # historique (1,L,...)
        pad = lambda x, v: torch.cat([x, torch.full_like(x[:, :1], v)], 1)
        PK = ("OBJ", "COL", "NO", "SND", "LAB", "SCN", "SCOL", "NS")
        b = {k: pad(v, 16 if k in ("COL", "SCOL") else 0) if k in PK else v for k, v in b.items()}
        A = torch.tensor([c[0] for c in cs], device=s.dev)
        XY = torch.tensor([[c[1][0] / 64, c[1][1] / 64, 1.0] if c[1] is not None else [0.0, 0.0, 0.0] for c in cs], device=s.dev)
        b["ACT"] = torch.cat([b["ACT"].expand(M, -1), A[:, None]], 1); b["CLK"] = torch.cat([b["CLK"].expand(M, -1, -1), XY[:, None]], 1)
        b["V"] = torch.ones(M, L + 1, dtype=torch.bool, device=s.dev)
        for k in PK:
            if k in b: b[k] = b[k].expand(M, *b[k].shape[1:])
        z = s.m.encode(b); zp = s.m.predict(z, b)[:, -1]
        return torch.sigmoid(s.m.h_snd(zp.float())).cpu().numpy()                 # (M,7)

    def act(s, grid, state, levels, avail):
        g = np.asarray(grid, dtype=np.int16)
        if s.pv is not None and state != "NOT_PLAYED": s.hear(g, state, levels)
        a_id, data = super().act(grid, state, levels, avail)
        if state in ("NOT_PLAYED", "GAME_OVER") or a_id == 0:
            s.pv = None; return a_id, data
        k = s.last[0]; cs = [c for c in s.G[k] if s.G[k][c] != "GO"]
        if (s.fire or s.listen) and len(s.H["ACT"]) >= 1 and cs:                   # listen : il écoute et note, sans jamais choisir (trajet = v1)
            P = s.imagine(cs); pd = P[:, DING]; order = np.argsort(-pd); best = int(order[0]); sc = float(pd[best])
            med = float(np.median(s.lvl_scores)) if len(s.lvl_scores) >= 5 else None; s.lvl_scores.append(sc)
            fire = s.fire and (sc >= s.tau or (s.rel > 0 and med is not None and sc >= 0.05 and sc >= s.rel * med))
            if fire and (k, cs[best]) not in s.fired:
                s.fired.add((k, cs[best])); s.n_fire += 1; a = cs[best]; s.last = (k, a); s.plan.clear()
                a_id, data = a[0], ({"x": a[1][0], "y": a[1][1]} if a[1] is not None else None)
            j = cs.index(s.last[1]) if s.last[1] in cs else None
            s.pd_last = (float(pd[j]), int(np.flatnonzero(order == j)[0]) + 1, len(cs)) if j is not None else None
        s.pv = (g, int(a_id), data, levels); return a_id, data


class Boussole(BlindAgent):
    """BOUSSOLE (1er nov., go user ; suite du diagnostic : le modèle sent « la fin est proche » dans un ÉTAT, pas quel geste) :
    valeur d'un état = max des notes DING imaginées sur ses gestes candidats (calculée à la 1re visite). v1 explore comme
    d'habitude, mais pour choisir OÙ aller ensuite il préfère les états à forte valeur (valeur relative au niveau − λ × distance),
    et quitte l'état courant pour un état bien plus « chaud » (× ratio) même s'il reste de l'inconnu ici."""
    def __init__(s, *a, lam=0.02, ratio=2.0, **k):
        super().__init__(*a, fire=False, **k); s.val = {}; s.lam, s.ratio = lam, ratio; s.n_steer = 0

    def new_level(s, grid):
        super().new_level(grid); s.val = {}

    def value(s, k):
        if k not in s.val and len(s.H["ACT"]) >= 1:
            cs = [c for c in s.G[k] if s.G[k][c] != "GO"]
            if cs: s.val[k] = float(s.imagine(cs)[:, DING].max())
        return s.val.get(k)

    def frontier(s, k):                               # BFS : chaque état connu qui a encore de l'inconnu -> (distance, chemin)
        prev = {k: None}; q = deque([k]); dist = {k: 0}; out = []
        while q:
            u = q.popleft()
            if u != k and any(n is None for n in s.G.get(u, {}).values()): out.append(u)
            for a, n in s.G.get(u, {}).items():
                if n is not None and n != "GO" and n not in prev and n in s.G:
                    prev[n] = (u, a); dist[n] = dist[u] + 1; q.append(n)
        def path(u):
            P = []
            while prev[u] is not None:
                p, a = prev[u]; P.append((p, a)); u = p
            return P[::-1]
        return out, dist, path

    def choose(s, k):
        if s.plan: return super().choose(k)
        vk = s.value(k); out, dist, path = s.frontier(k)
        V = [v for v in s.val.values() if v is not None]; top = max(V) if V else 0.0
        sc = {u: s.val[u] / (top + 1e-9) - s.lam * dist[u] for u in out if s.val.get(u) is not None}
        if sc:
            u = max(sc, key=sc.get); untried_here = any(n is None for n in s.G[k].values())
            if not untried_here or (vk is not None and s.val[u] > s.ratio * vk and s.val[u] >= 0.02):
                P = path(u)
                if P: s.n_steer += untried_here; s.plan = deque(P[1:]); return P[0][1]
        return super().choose(k)


class Vote(BlindAgent):
    """VOTE (1er nov., idée user « utiliser tous nos modèles à bon escient, ensemble ») : v1 garde le volant ; seul changement, l'ORDRE
    dans lequel il essaie les gestes JAMAIS essayés de l'état courant. Chacun vote là où il a fait ses preuves :
      effet   = 1 − ½ (toc prédit par le modèle aveugle + toc COMPTÉ pour ce geste dans la partie)   [règles naïves + modèle] ;
      nouveauté = sons prédits (contact, ramassé, couleur, téléport, ding) pondérés par leur RARETÉ dans la partie (curiosité) ;
      danger  = buzz prédit ;  départage = l'ordre de v1 (taux d'effet par type de geste).
    Pas de sous-but : on ne fait qu'économiser des actions."""
    def __init__(s, *a, w_new=0.5, **k):
        super().__init__(*a, fire=False, **k); s.w_new = w_new; s.n_vote = 0

    def counts(s, cs):                                # règle naïve : fréquence de chaque son pour CE geste (clic par zone 8×8), lissée
        A, C, S = s.H["ACT"], s.H["CLK"], s.H["SND"]; key = lambda a, c: (int(a), (c[0] // 8, c[1] // 8) if int(a) == 6 else None)
        tot = np.zeros(len(SND)); per = {}
        for a, c, x in zip(A, C, S):
            k = key(a, c); n, m = per.get(k, (np.zeros(len(SND)), 0)); per[k] = (n + x, m + 1); tot = tot + x
        g = (tot + 0.5) / (len(A) + 1)
        return np.array([(per.get(key(c[0], c[1] or (-1, -1)), (np.zeros(len(SND)), 0))[0] + 0.5 * g) / (per.get(key(c[0], c[1] or (-1, -1)), (None, 0))[1] + 0.5) for c in cs]), tot

    def choose(s, k):
        if s.plan or len(s.H["ACT"]) < 1: return super().choose(k)
        untried = [a for a, nxt in s.G[k].items() if nxt is None]
        if len(untried) < 2: return super().choose(k)
        P = s.imagine(untried); Q, tot = s.counts(untried); T, BZ = SND.index("toc"), SND.index("buzz")
        rare = 1.0 / (1.0 + tot); idx = [SND.index(x) for x in ("contact", "ramassé", "couleur", "téléport", "ding")]
        sc = (1 - 0.5 * (P[:, T] + Q[:, T])) + s.w_new * (P[:, idx] * rare[idx]).sum(1) - P[:, BZ] \
             + 0.1 * np.array([s.prior(a[2]) for a in untried]) + 1e-3 * s.rng.random(len(untried))
        s.n_vote += 1; return untried[int(sc.argmax())]




def play(arc, gid, l_budget, card, ag, max_levels=10, tutorial=None, burst=0):
    """tutorial = [[[a, x, y], ...] par niveau] : ces niveaux sont JOUÉS PAR LA SOLUTION ; l'agent aveugle les ENTEND (+ rafale
    d'apprentissage à la fin s'il apprend), puis joue SEUL la suite. v1 ne regarde pas (il démarre au niveau suivant)."""
    from arcengine import GameAction
    env = arc.make(gid, scorecard_id=card); r = env.observation_space; n = n_lv = 0; lv_at = []; t0 = time.time(); wins = []
    for lvl in (tutorial or []):
        for a, x, y in lvl:
            if not r.frame: r = env.step(GameAction.RESET)
            data = {"x": x, "y": y} if x >= 0 else None
            if isinstance(ag, BlindAgent): ag.observe(r.frame[-1], r.state.name, r.levels_completed, a, data)
            r = env.step(GameAction.from_id(a), data=data or {})
    lv_tut = r.levels_completed if tutorial else 0
    if tutorial and isinstance(ag, BlindAgent):
        ag.observe(r.frame[-1], r.state.name, r.levels_completed, 0, None); ag.pv = None   # entendre le DING du dernier geste du tutoriel
        if ag.ttt and burst: ag.learn(burst)
    while n_lv < l_budget and r is not None and r.state.name != "WIN" and len(lv_at) < max_levels:
        if not r.frame: r = env.step(GameAction.RESET); continue
        a, data = ag.act(r.frame[-1], r.state.name, r.levels_completed, list(r.available_actions))
        pdl = getattr(ag, "pd_last", None); lv0 = r.levels_completed; r = env.step(GameAction.from_id(a), data=data or {})
        if a == 0: continue
        n += 1; n_lv += 1
        if r is not None and r.levels_completed > lv0:
            lv_at.append(n_lv); n_lv = 0
            if pdl is not None: wins.append(dict(niveau=lv0 + 1, ding_imagine=round(pdl[0], 3), rang=pdl[1], candidats=pdl[2]))
    out = dict(levels=len(lv_at), level_at=lv_at, actions=n, sec=round(time.time() - t0, 1), wins=wins, tutorial_levels=lv_tut)
    if isinstance(ag, RelAgent): out.update(fired=ag.n_fire, updates=0, steer=0, votes=0)
    if isinstance(ag, BlindAgent): out.update(fired=ag.n_fire, updates=getattr(ag, "n_upd", 0), steer=getattr(ag, "n_steer", 0), votes=getattr(ag, "n_vote", 0))
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument("--ckpt", default=""); p.add_argument("--relnet", default=""); p.add_argument("--env", default=str(HERE / "data" / "environment_files"))
    p.add_argument("--games", default="g50t,ka59,lf52,r11l,sk48"); p.add_argument("--l_budget", type=int, default=300); p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--agents", default="v1,v1+aveugle,v1+aveugle+appr"); p.add_argument("--tau", type=float, default=0.5); p.add_argument("--rel", type=float, default=4.0)
    p.add_argument("--ttt_every", type=int, default=8); p.add_argument("--ttt_steps", type=int, default=4); p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--tutorial", type=int, default=0, help="niveaux joués par la solution oracle (entendus) avant de jouer seul")
    p.add_argument("--burst", type=int, default=32, help="pas d'apprentissage juste après le tutoriel (agent +appr)"); p.add_argument("--oracle", default=str(HERE / "data" / "oracle_public"))
    p.add_argument("--hist", type=int, default=0, help="mémoire du modèle limitée aux N derniers pas (0 = toute la partie)"); p.add_argument("--w_new", type=float, default=0.5); p.add_argument("--lam", type=float, default=0.02); p.add_argument("--ratio", type=float, default=2.0)
    p.add_argument("--out", default="blind_play.json"); a = p.parse_args(); logging.disable(logging.INFO); os.environ["ENVIRONMENTS_DIR"] = a.env
    from arc_agi import Arcade, OperationMode
    dev = "cuda" if torch.cuda.is_available() else "cpu"; m0, ma = load_model(a.ckpt, dev) if a.ckpt else (None, Namespace(arch="—", T=0))
    if a.relnet:
        from rel_model import RelNet
        ck = torch.load(a.relnet, map_location=dev, weights_only=False); rm = RelNet(**ck.get("cfg", {})).to(dev); rm.load_state_dict(ck["m"]); rm.eval()
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=a.env); card = arc.open_scorecard(tags=["blindplay"])
    ids = sorted(e.game_id for e in arc.get_environments()); res = []
    print(f"modèle {a.ckpt} ({ma.arch}, T={ma.T}) | {dev} | seuil ding {a.tau} ou ×{a.rel} la médiane du niveau | appr. tous les {a.ttt_every} pas × {a.ttt_steps} | ≤ {a.l_budget} actions par niveau", flush=True)
    for game in a.games.split(","):
        gid = [g for g in ids if g.startswith(game)][0]
        for name in a.agents.split(","):
            for sd in range(a.seeds):
                torch.manual_seed(sd); np.random.seed(sd)
                kw = dict(ttt=name.endswith("+appr"), ttt_every=a.ttt_every, ttt_steps=a.ttt_steps, lr=a.lr)
                ag = (Explorer(sd) if name == "v1" else RelSolo(rm if a.relnet else None, dev, sd, hasard=True) if name == "hasard" else RelSolo(rm, dev, sd) if name == "rel-seul" else RelAgent(rm, dev, sd, listen="écoute" in name, tau=a.tau) if name.startswith("v1+rel") else Boussole(m0, ma, dev, sd, lam=a.lam, ratio=a.ratio, **kw) if "boussole" in name
                      else BlindAgent(m0, ma, dev, sd, fire=False, listen=True, hist=a.hist, **kw) if "écoute" in name
                      else Vote(m0, ma, dev, sd, hist=a.hist, w_new=a.w_new, **kw) if "vote" in name
                      else BlindAgent(m0, ma, dev, sd, fire=True, tau=a.tau, rel=a.rel, hist=a.hist, **kw))
                tut = json.load(open(f"{a.oracle}/{game}_solution.json"))["solution"][:a.tutorial] if a.tutorial else None
                r = play(arc, gid, a.l_budget, card, ag, tutorial=tut, burst=a.burst); r.update(game=game, agent=name, seed=sd); res.append(r)
                ex = f" | déclenchés {r['fired']} | détours boussole {r['steer']} | appr. {r['updates']}" if "fired" in r else ""
                print(f"{game} {name:16s} graine {sd} | tutoriel {r['tutorial_levels']} niv. | niveaux SEUL {r['levels']} à {r['level_at']} | {r['actions']} actions | {r['sec']}s{ex} | ding imaginé au geste gagnant {r['wins']}", flush=True)
                Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    def game_score(r):                                # règle officielle approchée (colab_off.score) : niveau i pèse i+1, (humain/agent)² plafonné à 1
        sj = json.load(open(f"{a.oracle}/{r['game']}_solution.json")); hum, W = sj["baseline"], sj["win_levels"]
        return sum((i + 1) * min(1, (hum[i] / max(1, n)) ** 2) for i, n in enumerate(r["level_at"]) if i < len(hum)) / sum(i + 1 for i in range(W))
    print("\nSCORE approché (0-100, moyenne des graines puis des jeux ; 5 EXCLUS à part)", flush=True)
    TEST = {"g50t", "ka59", "lf52", "r11l", "sk48"}
    for name in a.agents.split(","):
        R = [r for r in res if r["agent"] == name]; G = sorted({r["game"] for r in R})
        per = {g: np.mean([game_score(r) for r in R if r["game"] == g]) for g in G}
        ex = [per[g] for g in G if g in TEST]; vu = [per[g] for g in G if g not in TEST]
        print(f"  {name:16s} : tous {100 * np.mean(list(per.values())):.2f} | exclus {100 * np.mean(ex) if ex else float('nan'):.2f} | autres {100 * np.mean(vu) if vu else float('nan'):.2f} | actions moyennes par niveau gagné {np.mean([n for r in R for n in r['level_at']]) if any(r['level_at'] for r in R) else float('nan'):.0f}", flush=True)
    print("\nBILAN (niveaux par jeu, somme des graines)", flush=True)
    for name in a.agents.split(","):
        R = [r for r in res if r["agent"] == name]
        print(f"  {name:16s} : " + " | ".join(f"{g} {sum(r['levels'] for r in R if r['game'] == g)}" for g in a.games.split(",")) + f" | total {sum(r['levels'] for r in R)}", flush=True)


if __name__ == "__main__":
    main()
