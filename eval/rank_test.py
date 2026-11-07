"""TEST DU POINT DE RUPTURE (27 oct.) : le planificateur RECONNAÎT-IL la bonne solution quand on la lui montre ?
Le long du chemin de l'oracle (niveaux choisis), à chaque état : on note avec l'imagination du modèle (même note que le planificateur,
sans nouveauté ni souvenir) la suite oracle des H prochains gestes ET N suites au hasard (flèches + clics sur objets).
Rang de la suite oracle (1 = la meilleure). Si elle est noyée -> la NOTE (modèle / critique) est en cause ; si elle est en tête
mais que l'agent ne la joue pas -> c'est la RECHERCHE.
    python rank_test.py --ckpt nuit_final.pt --sols sols.json --game lp85 --levels 1,2 --n 200"""
import argparse, json, logging, os, sys
import numpy as np, torch
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.join(HERE, "wm")]


def main():
    p = argparse.ArgumentParser(); p.add_argument("--ckpt", required=True); p.add_argument("--sols", required=True)
    p.add_argument("--game", default="lp85"); p.add_argument("--levels", default="1,2"); p.add_argument("--n", type=int, default=200)
    p.add_argument("--H", type=int, default=4); p.add_argument("--dist", type=float, default=3.0); p.add_argument("--env", default=os.path.join(HERE, "data", "environment_files")); a = p.parse_args()
    os.environ["OPERATION_MODE"] = "offline"; os.environ["ENVIRONMENTS_DIR"] = a.env; logging.disable(logging.INFO)
    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction
    from plan_agent import PlanAgent, click_candidates
    from model import NA
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=a.env); card = arc.open_scorecard(tags=["rang"])
    sol = json.load(open(a.sols))[a.game][0]; want = [int(x) for x in a.levels.split(",")]
    ag = PlanAgent(a.ckpt, 0, "grad", H=a.H, seeds=8, iters=12, w=dict(crit=1.0, lc=1.0, nov=1.0, eff=0.2, mem=0.5, go=1.0, dream=3.0, dist=a.dist)); im = ag.pl.im; C = ag.C; H = a.H; dev = im.dev if hasattr(im, "dev") else "cpu"
    rng = np.random.RandomState(0)
    env = arc.make(a.game, scorecard_id=card); r = env.step(GameAction.RESET)
    fr = [np.asarray(r.frame[-1], np.int8)]; acts, clicks = [], []; ding = [0]
    for li, lv in enumerate(sol, 1):
        if li > max(want): break
        flat = [tuple(s) for s in lv]
        if li in want: print(f"\n=== {a.game} niveau {li} : {len(flat)} gestes oracle ===", flush=True)
        for k, (act, x, y) in enumerate(flat):
            if li in want:
                g = fr[-1]; avail = [int(v) for v in r.available_actions]
                objs = click_candidates(g.astype(np.int16)) if 6 in avail else []
                pts = [(xx, yy) for (_c, _n, (yy, xx)) in objs] or [(32, 32)]
                ges = [(b, (32, 32)) for b in avail if 0 < b < NA and b != 6] + ([(6, q) for q in pts] if 6 in avail else [])
                im.ding = ding[-C - 1:]; im.context(fr[-C - 1:], acts[-C:], clicks[-C:])
                orc = flat[k:k + H]; orc = orc + [orc[-1]] * (H - len(orc))                         # suite oracle (complétée si fin de niveau)
                seqs = [[(int(b), (int(xx), int(yy)) if int(b) == 6 else (32, 32)) for b, xx, yy in orc]]
                seqs += [[ges[rng.randint(len(ges))] for _ in range(H)] for _ in range(a.n)]
                A = torch.tensor([[s[0] for s in q] for q in seqs], device=im.dev); XY = torch.tensor([[s[1] for s in q] for q in seqs], device=im.dev, dtype=torch.float32)
                with torch.no_grad(): sc, terms = im.score(F.one_hot(A, NA).float(), XY, None, None)
                sc = sc.float().cpu().numpy(); rank = int((sc > sc[0]).sum()) + 1
                tcrit = terms.get("crit"); tlc = terms.get("lc"); td = terms.get("dist")
                cs = f" | critique oracle {float(tcrit[0].mean()):.2f} vs hasard {float(tcrit[1:].mean()):.2f}" if tcrit is not None else ""
                ds = f" | DISTANCE oracle {float(td[0].mean()):.2f} vs hasard {float(td[1:].mean()):.2f}" if td is not None else ""
                ls = f" | « fin de niveau » prédite oracle {float(tlc[0].max()):.2f} vs hasard {float(tlc[1:].max(1).values.mean()):.2f}" if tlc is not None else ""
            lv0 = r.levels_completed; r = env.step(GameAction.from_id(int(act)), data={"x": x, "y": y} if int(act) == 6 else {})
            fr.append(np.asarray(r.frame[-1], np.int8)); acts.append(int(act)); clicks.append((x, y) if int(act) == 6 else (-1, -1))
            ding.append(int(r.levels_completed > lv0))
            if li in want:                                                                   # CRITIQUE SUR LES VRAIES IMAGES (pas d'imagination) :
                cr = real_crit(im.m, fr, acts, clicks, ding, dev)                            # « fini dans ≤ 8 pas » pour le geste oracle qu'on vient de jouer
                print(f"geste {k + 1:2d}/{len(flat)} ({'clic' if act == 6 else 'A' + str(act)}) : rang de la suite oracle {rank:3d}/{len(seqs)}"
                      f" (top {100 * rank / len(seqs):4.0f} %){cs} | critique VRAIES images {cr[0]:.2f}{'' if cr[1] is None else f' (distance {cr[1]:.2f})'}{ds}{ls}", flush=True)


@torch.no_grad()
def real_crit(m, fr, acts, clicks, ding, dev):
    """critique du modèle sur la fenêtre RÉELLE qui finit au geste qu'on vient de jouer (comme à l'entraînement)."""
    dev = next(m.parameters()).device
    T = min(m.T if hasattr(m, "T") else 16, len(acts)); g = torch.as_tensor(np.asarray(fr[-T - 1:]), device=dev).long()[None]
    a_ = torch.as_tensor(acts[-T:], device=dev).long()[None]; c_ = torch.as_tensor(np.asarray(clicks[-T:]), device=dev).long()[None]
    if getattr(m, "hear", False):
        lc = torch.tensor(ding[-T - 1:-1], device=dev, dtype=torch.float32)[None]; m.cur = (lc, torch.zeros_like(lc))
    m.run(g, a_, c_); c1 = float(torch.sigmoid(m.critic()[0, -1, 1].float()))
    return (c1, float(torch.sigmoid(m.crit_d(m.pred.hid.mean(2))[0, -1, 0].float()))) if getattr(m, "has_dist", False) else (c1, None)


if __name__ == "__main__":
    main()
