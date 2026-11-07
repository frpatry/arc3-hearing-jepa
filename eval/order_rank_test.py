"""ORDRE DES GESTES (27 oct., question user : l'imagination longue est-elle ce qui manque ?) : à un état réel du chemin de l'oracle,
on prend ses N prochains gestes (la BONNE suite), jusqu'à P permutations de ces MÊMES gestes (mauvais ordre)
et P autres combinaisons des boutons de ce niveau (quels boutons / combien de fois).
Chaque suite est notée deux fois :
  - IMAGINÉE : le modèle imagine les N pas (comme le planificateur) -> note totale, critique « ≤ 8 pas », distance au but ;
  - RÉELLE   : on la joue vraiment dans le jeu, puis la critique lit les VRAIES images (imagination parfaite).
Seules les permutations dont le résultat RÉEL diffère de la bonne suite comptent (ordre sans effet = ex aequo, retiré).
Lecture : bonne suite en tête en RÉEL mais noyée en IMAGINÉ quand N grandit -> l'imagination longue est le manque ;
noyée même en RÉEL -> c'est la critique ; en tête dans les deux -> c'est la recherche du planificateur.
    python order_rank_test.py --ckpt wm_seq_critim.pt --games vc33 --spots 2:0,3:0,3:8 --Ns 2,4,8,12 --P 20"""
import argparse, json, logging, os, sys
import numpy as np, torch
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.join(HERE, "wm")]
W = dict(crit=3.0, lc=1.0, nov=1.0, eff=0.2, mem=0.5, go=1.0, dream=3.0, dist=3.0)                  # = agent soumis (sans nouveauté ni souvenir)


def rank(sc):                                                     # rang de l'entrée 0 (1 = la meilleure)
    sc = np.asarray(sc, np.float64); return int((sc[1:] > sc[0]).sum()) + 1


def main():
    p = argparse.ArgumentParser(); p.add_argument("--ckpt", required=True); p.add_argument("--games", default="vc33")
    p.add_argument("--spots", default="2:0,3:0,3:8", help="niveau:geste de départ dans la solution de l'oracle")
    p.add_argument("--Ns", default="2,4,8,12"); p.add_argument("--P", type=int, default=20)
    p.add_argument("--sols", default=os.path.join(HERE, "data", "oracle_public")); p.add_argument("--out", default="ordre.json")
    p.add_argument("--env", default=os.path.join(HERE, "data", "environment_files")); a = p.parse_args()
    os.environ["OPERATION_MODE"] = "offline"; os.environ["ENVIRONMENTS_DIR"] = a.env; logging.disable(logging.INFO)
    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction
    from plan_agent import PlanAgent
    from planner import Imagination
    from model import NA
    from rank_test import real_crit
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=a.env); card = arc.open_scorecard(tags=["ordre"])
    m = PlanAgent(a.ckpt, 0, "grad", H=4, w=W).m; T = m.T; rows = []

    for game in a.games.split(","):
        gid = [e.game_id for e in arc.get_environments() if e.game_id.startswith(game)][0]
        sol = [[tuple(int(v) for v in g) for g in lv] for lv in json.load(open(os.path.join(a.sols, f"{game}_solution.json")))["solution"]]

        def play(seq):                                            # rejoue depuis RESET ; renvoie images, actions, clics, dings
            env = arc.make(gid, scorecard_id=card); r = env.step(GameAction.RESET)
            fr = [np.asarray(r.frame[-1], np.int8)]; ac, cl, dg = [], [], [0]
            for act, x, y in seq:
                lv0 = r.levels_completed; r = env.step(GameAction.from_id(act), data={"x": x, "y": y} if act == 6 else {})
                fr.append(np.asarray(r.frame[-1], np.int8)); ac.append(act); cl.append((x, y) if act == 6 else (-1, -1)); dg.append(int(r.levels_completed > lv0))
            return fr, ac, cl, dg

        for spot in a.spots.split(","):
            lv, k0 = (int(v) for v in spot.split(":"))
            pre = [g for l in sol[:lv - 1] for g in l] + list(sol[lv - 1][:k0]); rest = list(sol[lv - 1][k0:])
            fr0, ac0, cl0, dg0 = play(pre)
            for N in [int(n) for n in a.Ns.split(",")]:
                if N > len(rest): continue
                good = rest[:N]; rng = np.random.RandomState(N); seen = {tuple(good)}; perms = []
                for _ in range(50 * a.P):                         # permutations DISTINCTES des mêmes gestes
                    q = tuple(good[i] for i in rng.permutation(N))
                    if q not in seen: seen.add(q); perms.append(list(q))
                    if len(perms) >= a.P: break
                pool_g = sorted(set(sol[lv - 1]))                 # + AUTRES COMBINAISONS des boutons du niveau (vc33 : l'ordre des pompes ne change rien,
                for _ in range(50 * a.P):                         #   le vrai choix = quels boutons et combien de fois)
                    if len(perms) >= 2 * a.P: break
                    q = tuple(pool_g[i] for i in rng.randint(len(pool_g), size=N))
                    if q not in seen: seen.add(q); perms.append(list(q))
                seqs = [good] + perms; real = [play(pre + list(q)) for q in seqs]
                end0 = real[0][0][-1]; keep = [0] + [i for i in range(1, len(seqs)) if not np.array_equal(real[i][0][-1], end0)]
                if len(keep) < 2:
                    print(f"{game} niv.{lv} geste {k0} N={N} : {len(perms)} autres suites, AUCUNE ne change le résultat réel -> ordre sans effet", flush=True); continue
                seqs = [seqs[i] for i in keep]; real = [real[i] for i in keep]
                # --- IMAGINÉ : même calcul que le planificateur (contexte réel = T - N images)
                im = Imagination(m, H=N, w=dict(W, prog=1.0)); C = T - N
                im.ding = dg0[-C - 1:]; im.context(fr0[-C - 1:], ac0[-C:] if C > 0 else [], cl0[-C:] if C > 0 else np.zeros((0, 2)))
                A = torch.tensor([[g[0] for g in q] for q in seqs], device=im.dev)
                XY = torch.tensor([[(g[1], g[2]) if g[0] == 6 else (32, 32) for g in q] for q in seqs], device=im.dev, dtype=torch.float32)
                with torch.no_grad():
                    tot, terms = im.score(F.one_hot(A, NA).float(), XY, None, None)
                tot = tot.float().cpu().numpy()
                icrit = terms["crit"][:, -1].float().cpu().numpy() if "crit" in terms else None
                idist = terms["dist"][:, -1].float().cpu().numpy() if "dist" in terms else None
                iprog = terms["prog"][:, -1].float().cpu().numpy()                 # chaleur imaginée cumulée en fin de suite
                # --- RÉEL : la critique lit les vraies images à la fin de chaque suite jouée
                rc = np.array([real_crit(m, *r_, None) for r_ in real], dtype=object)
                rcrit = np.array([float(c[0]) for c in rc]); rdist = np.array([float(c[1]) if c[1] is not None else np.nan for c in rc])
                rprog = np.array([real_heat(m, *r_, N) for r_ in real])                # chaleur lue sur les VRAIES images, cumulée sur les N gestes
                row = dict(game=game, level=lv, start=k0, N=N, n=len(seqs) - 1, rank_tot=rank(tot),
                           rank_crit_im=rank(icrit) if icrit is not None else None, rank_dist_im=rank(idist) if idist is not None else None,
                           rank_prog_im=rank(iprog), rank_prog_real=rank(rprog), prog_im_good=float(iprog[0]), prog_im_perm=float(iprog[1:].mean()),
                           prog_real_good=float(rprog[0]), prog_real_perm=float(rprog[1:].mean()),
                           rank_crit_real=rank(rcrit), rank_dist_real=rank(rdist) if not np.isnan(rdist).any() else None,
                           dist_im_good=float(idist[0]) if idist is not None else None, dist_im_perm=float(idist[1:].mean()) if idist is not None else None,
                           dist_real_good=float(rdist[0]), dist_real_perm=float(np.nanmean(rdist[1:])), level_done=bool(real[0][3][-1]))
                rows.append(row)
                print(f"{game} niv.{lv} geste {k0:2d} N={N:2d} ({row['n']:2d} autres suites{' ; la bonne suite FINIT le niveau' if row['level_done'] else ''}) | "
                      f"rang IMAGINÉ : note {row['rank_tot']:2d}, critique {row['rank_crit_im']}, distance {row['rank_dist_im']} "
                      f"(bonne {row['dist_im_good']:.2f} vs autres {row['dist_im_perm']:.2f}) | "
                      f"rang RÉEL : critique {row['rank_crit_real']}, distance {row['rank_dist_real']} (bonne {row['dist_real_good']:.2f} vs autres {row['dist_real_perm']:.2f}) | "
                      f"CHALEUR rang imaginé {row['rank_prog_im']} ({row['prog_im_good']:+.2f} vs {row['prog_im_perm']:+.2f}), réel {row['rank_prog_real']} "
                      f"({row['prog_real_good']:+.2f} vs {row['prog_real_perm']:+.2f})", flush=True)
    json.dump(rows, open(a.out, "w"), indent=1)
    print("\nRÉSUMÉ (rang de la bonne suite, 1 = en tête ; hasard ≈ (n+2)/2)")
    for N in sorted({r["N"] for r in rows}):
        R = [r for r in rows if r["N"] == N]
        print(f"N={N:2d} : imaginé distance {np.mean([r['rank_dist_im'] for r in R]):.1f} | réel distance {np.mean([r['rank_dist_real'] for r in R]):.1f} | "
              f"CHALEUR imaginée {np.mean([r['rank_prog_im'] for r in R]):.1f} | réelle {np.mean([r['rank_prog_real'] for r in R]):.1f} | "
              f"hasard {np.mean([(r['n'] + 2) / 2 for r in R]):.1f}  ({len(R)} cas)", flush=True)


@torch.no_grad()
def real_heat(m, fr, acts, clicks, ding, N):
    """progrès prédit par le modèle sur les VRAIES images (fenêtre T qui finit au dernier geste), cumulé sur les N derniers gestes."""
    from model import EV_PROG
    dev = next(m.parameters()).device; T = min(m.T, len(acts))
    g = torch.as_tensor(np.asarray(fr[-T - 1:]), device=dev).long()[None]
    a_ = torch.as_tensor(acts[-T:], device=dev).long()[None]; c_ = torch.as_tensor(np.asarray(clicks[-T:]), device=dev).long()[None]
    if getattr(m, "hear", False):
        lc = torch.tensor(ding[-T - 1:-1], device=dev, dtype=torch.float32)[None]; m.cur = (lc, torch.zeros_like(lc))
    m.run(g, a_, c_)
    z = m.eye(g[0, :T]); hid = m.pred.hid[0]                                      # (T,64,d) : même lecture que le planificateur (entrée du pas + état interne)
    o = m.ev(torch.cat([z, hid], -1).mean(1)).float()[:, EV_PROG]
    return float(torch.tanh(o[-N:]).sum())


if __name__ == "__main__":
    main()
