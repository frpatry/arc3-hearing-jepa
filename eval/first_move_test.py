"""TEST DU 1er GESTE (27 oct., idée user) : sur la 1re image d'un niveau, le modèle sait-il par où commencer ?
On rejoue la solution oracle jusqu'au début de chaque niveau choisi, puis on demande au modèle de classer tous les 1ers gestes possibles :
  acteur (ce que ferait l'expert), rêve (geste qui mène le plus près de l'image imaginée d'un gagnant), distance au but, critique, note totale.
On lit le rang du VRAI 1er geste oracle (clic : même case 8×8). Rang 1 = le modèle l'aurait joué.
    python first_move_test.py --ckpt crit_im.pt --sols sols.json --levels 1,2,3"""
import argparse, json, logging, os, sys
import numpy as np, torch
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.join(HERE, "wm")]


def main():
    p = argparse.ArgumentParser(); p.add_argument("--ckpt", required=True); p.add_argument("--sols", required=True)
    p.add_argument("--levels", default="1,2,3"); p.add_argument("--games", default="")
    p.add_argument("--env", default=os.path.join(HERE, "data", "environment_files")); a = p.parse_args()
    os.environ["OPERATION_MODE"] = "offline"; os.environ["ENVIRONMENTS_DIR"] = a.env; logging.disable(logging.INFO)
    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction
    from plan_agent import PlanAgent, click_candidates
    from planner import Imagination, actor_prior, actor_points, try_key
    from model import NA
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=a.env); card = arc.open_scorecard(tags=["geste1"])
    S = json.load(open(a.sols)); want = [int(x) for x in a.levels.split(",")]
    ag = PlanAgent(a.ckpt, 0, "grad", H=4, seeds=8, iters=12); m = ag.m; C = ag.C
    im = Imagination(m, 1, 0.9, dict(crit=1.0, lc=1.0, dist=3.0, dream=3.0))           # 1 seul pas imaginé : « que se passe-t-il si je fais ce geste ? »
    cell = lambda b, xy: (int(b),) if int(b) != 6 else (6, int(xy[1]) // 8, int(xy[0]) // 8)
    KEYS = ("acteur", "rêve", "distance", "critique", "total"); R = {k: [] for k in KEYS}; NC = []
    games = [g for g in S if not a.games or g in a.games.split(",")]
    for g in games:
        sol = S[g][0]; env = arc.make(g, scorecard_id=card); r = env.step(GameAction.RESET)
        fr = [np.asarray(r.frame[-1], np.int8)]; acts, clicks, ding = [], [], [0]; line = []
        for li, lv in enumerate(sol, 1):
            if li > max(want) or r is None: break
            if li in want and lv:
                gr = fr[-1]; avail = [int(v) for v in r.available_actions]
                im.ding = ding[-C - 1:]; im.context(fr[-C - 1:], acts[-C:], clicks[-C:])
                objs = click_candidates(gr.astype(np.int16)) if 6 in avail else []
                pts = [(xx, yy) for (_c, _n, (yy, xx)) in objs] + (actor_points(im) if 6 in avail else [])
                cands = [(b, (32, 32)) for b in avail if 0 < b < NA and b != 6] + ([(6, q) for q in pts] if 6 in avail else [])
                act, x, y = lv[0]; ko = cell(act, (x, y))
                if not any(cell(*c) == ko for c in cands): cands.append((int(act), (int(x), int(y))))   # le geste oracle doit être classable
                pri = actor_prior(im, [(b, q if b == 6 else None) for b, q in cands]) or {}
                A = torch.tensor([[b] for b, _ in cands], device=im.dev); XY = torch.tensor([[q] for _, q in cands], device=im.dev, dtype=torch.float32)
                with torch.no_grad(): tot, terms = im.score(F.one_hot(A, NA).float(), XY, None, None)
                sc = dict(acteur=np.array([pri.get(try_key(b, q if b == 6 else None), 0.0) for b, q in cands]), total=tot.float().cpu().numpy())
                for k, t in (("rêve", "dream"), ("distance", "dist"), ("critique", "crit")):
                    if t in terms: sc[k] = terms[t][:, 0].float().cpu().numpy()
                cs = [cell(*c) for c in cands]; NC.append(len(set(cs)))
                txt = []
                for k in KEYS:
                    if k not in sc: continue
                    best = {}                                                            # plusieurs candidats par case : on garde le meilleur
                    for c_, v in zip(cs, sc[k]): best[c_] = max(best.get(c_, -1e9), float(v))
                    order = sorted(best, key=lambda c_: -best[c_]); rk = order.index(ko) + 1; R[k].append(rk); txt.append(f"{k} {rk}")
                line.append(f"niv.{li} [{len(set(cs))} gestes] " + " ".join(txt))
            for act, x, y in lv:                                                         # on rejoue l'oracle jusqu'au niveau suivant
                lv0 = r.levels_completed; r = env.step(GameAction.from_id(int(act)), data={"x": x, "y": y} if int(act) == 6 else {})
                if r is None or not r.frame: break
                fr.append(np.asarray(r.frame[-1], np.int8)); acts.append(int(act)); clicks.append((x, y) if int(act) == 6 else (-1, -1))
                ding.append(int(r.levels_completed > lv0))
        print(f"{g} : " + " | ".join(line), flush=True)
    n = len(R["total"]); print(f"\n=== {n} débuts de niveau, ~{int(np.mean(NC)) if NC else 0} gestes possibles en moyenne ===")
    for k in KEYS:
        if R[k]:
            v = np.array(R[k]); print(f"{k:9s} : oracle 1er choix {np.mean(v == 1):.0%} | dans le top 3 {np.mean(v <= 3):.0%} | rang médian {int(np.median(v))}", flush=True)
    print(f"hasard   : 1er choix ~{np.mean([1 / c for c in NC]):.0%} | top 3 ~{np.mean([min(3 / c, 1) for c in NC]):.0%}")


if __name__ == "__main__":
    main()
