"""REGARDER CE QUI SE PASSE (27 oct., user) : joue un jeu avec un réglage donné, enregistre chaque image, l'action, le clic et
ce que pensait le planificateur ; écrit un GIF par partie + une planche (début niv. 1, victoire niv. 1, puis le niveau 2 tous les k pas).
    python record_game.py --ckpt crit_im.pt --game vc33 --seeds 0,2 --H 12 --crit 3 --out dossier"""
import argparse, json, logging, os, sys
import numpy as np
from PIL import Image, ImageDraw
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.join(HERE, "wm")]
import types                                                                        # palette officielle sans importer gen/ (gen.registry absent sur Colab)
_C = ['#FFFFFF', '#CCCCCC', '#999999', '#666666', '#333333', '#000000', '#E53AA3', '#FF7BCC', '#F93C31',           '#1E93FF', '#88D8F1', '#FFDC00', '#FF851B', '#921231', '#4FCC30', '#A356D6']
_sh = types.ModuleType("gen.sheet"); _sh.PALETTE = np.array([[int(c[k:k + 2], 16) for k in (1, 3, 5)] for c in _C], np.uint8)
import gen; sys.modules["gen.sheet"] = _sh
from make_videos import frame_img

W0 = dict(crit=1.0, lc=1.0, nov=1.0, eff=0.2, mem=0.5, go=1.0, dream=3.0, dist=3.0)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--ckpt", required=True); p.add_argument("--game", default="vc33")
    p.add_argument("--seeds", default="0,2"); p.add_argument("--H", type=int, default=12); p.add_argument("--crit", type=float, default=3.0)
    p.add_argument("--l1", type=int, default=100); p.add_argument("--l2", type=int, default=100); p.add_argument("--every", type=int, default=6); p.add_argument("--focus", type=int, default=0)
    p.add_argument("--prog", type=float, default=0.0); p.add_argument("--heat_nav", type=float, default=0.0); p.add_argument("--shoot", type=int, default=0)   # CHALEUR / ZONES (27 oct.)
    p.add_argument("--frozen", type=int, default=0); p.add_argument("--heat_rel", type=float, default=0.0); p.add_argument("--heat_look", type=int, default=0)   # 1 = MODÈLE FIGÉ, comme la soumission (30 oct.)
    p.add_argument("--env", default=os.path.join(HERE, "data", "environment_files")); p.add_argument("--out", default="rec"); a = p.parse_args()
    os.environ["OPERATION_MODE"] = "offline"; os.environ["ENVIRONMENTS_DIR"] = a.env; logging.disable(logging.INFO)
    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction
    from plan_agent import PlanAgent
    os.makedirs(a.out, exist_ok=True)
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=a.env); card = arc.open_scorecard(tags=["film"])
    gid = [e.game_id for e in arc.get_environments() if e.game_id.startswith(a.game)][0]
    for seed in [int(s) for s in a.seeds.split(",")]:
        ag = PlanAgent(a.ckpt, seed, "grad", H=a.H, seeds=8, iters=12, w=dict(W0, crit=a.crit, prog=a.prog), heat_nav=a.heat_nav, shoot=a.shoot, **({"focus": True} if a.focus else {}), **({"ttt_every": 0, "win_steps": 0} if a.frozen else {}), heat_rel=a.heat_rel, heat_look=a.heat_look)
        jr = []                                                                     # JOURNAL : geste, chaleur prédite, température de la carte, retour en arrière
        env = arc.make(gid, scorecard_id=card); obs = env.step(GameAction.RESET)
        rec = []; n = 0; t_lv = 0; lv_at = []; hist = []
        while obs is not None and obs.state.name != "WIN" and len(lv_at) < 2:
            if n - t_lv >= (a.l1 if not lv_at else a.l2): break
            st = obs.state.name; grid = obs.frame[-1] if obs.frame else []
            if st in ("NOT_PLAYED", "GAME_OVER") or not obs.frame:
                ag.act(grid, "GAME_OVER" if st != "NOT_PLAYED" else st, obs.levels_completed, list(obs.available_actions))
                rec.append(dict(f=None, ev="GAME OVER -> RESET")); obs = env.step(GameAction.RESET); continue
            anim = len(obs.frame) > 1 and any(not np.array_equal(f, obs.frame[-1]) for f in obs.frame[:-1])
            g = np.asarray(obs.frame[-1], np.int8)
            act, data = ag.act(grid, st, obs.levels_completed, list(obs.available_actions), anim=anim)
            inf = ag.info or {}; lv0 = obs.levels_completed
            obs = env.step(GameAction.from_id(int(act)), data=data or {})
            nav = getattr(ag, "nav_go", None); tp = getattr(ag, "temp", (0.0, 0.0))
            if int(act) == 0:
                if st == "NOT_FINISHED" and nav == "RESET": jr.append(f"{n:3d} RETOUR EN ARRIÈRE : RESET vers la piste la plus chaude (ici T={tp[0]:+.2f})")
                continue
            n += 1; hist.append(float(tp[0]) if a.heat_nav else float(inf.get("crit", 0.0)))
            jr.append(f"{n:3d} niv{len(lv_at) + 1} {'clic (%d,%d) couleur %d' % (data['x'], data['y'], int(g[data['y'], data['x']])) if data else 'A%d' % int(act):24s} "
                      f"chaleur prédite {inf.get('prog0', 0.0):+.2f} | T ici {tp[0]:+.2f} (habituée {tp[1]:+.2f}) | critique {inf.get('crit', 0.0):.2f}"
                      f"{' | RETOUR -> ' + str(nav) if nav not in (None, 'RESET') else ''} | états connus {len(ag.map.G)}")
            rec.append(dict(f=g, a=int(act), x=(data or {}).get("x", -1), y=(data or {}).get("y", -1), lv=len(lv_at), step=n - t_lv,
                            crit=float(inf.get("crit", 0.0)), dist=float(inf.get("dist", 0.0)), nov=float(inf.get("nov", 0.0)), eff=float(inf.get("eff", 0.0)), hist=list(hist)))
            if obs is not None and obs.levels_completed > lv0:
                lv_at.append(n - t_lv); t_lv = n; hist = []
                rec.append(dict(f=np.asarray(obs.frame[-1], np.int8), a=0, x=-1, y=-1, lv=len(lv_at), step=0, crit=0, dist=0, nov=0, eff=0, hist=[], ev="NIVEAU REUSSI"))
        fr = [frame_img(r["f"], r["a"], r["x"], r["y"], r["step"], r["lv"], r["crit"], r["nov"], r["eff"], r["hist"]) for r in rec if r.get("f") is not None]
        tag = f"{a.game}_H{a.H}_c{a.crit:g}{'_focus' if a.focus else ''}{'_chaleur' if a.prog else ''}{'_zones' if a.heat_nav else ''}_s{seed}"
        open(os.path.join(a.out, tag + "_journal.txt"), "w").write("\n".join(jr) + "\n")
        if fr: fr[0].save(os.path.join(a.out, tag + ".gif"), save_all=True, append_images=fr[1:], duration=250, loop=0)
        # PLANCHE : 1re image niv. 1, image de la victoire, puis le niveau 2 tous les « every » pas (avec ce qu'a joué l'agent)
        R = [r for r in rec if r.get("f") is not None]
        l2 = [r for r in R if r["lv"] == 1 and r["step"] > 0]
        pick = [R[0]] + [r for r in R if r.get("ev") == "NIVEAU REUSSI"][:1] + l2[::a.every][:22]
        tiles = [frame_img(r["f"], r["a"], r["x"], r["y"], r["step"], r["lv"], r["crit"], r["nov"], r["eff"], [], S=3) for r in pick]
        if tiles:
            w, h = tiles[0].size; cols = 6; rows = (len(tiles) + cols - 1) // cols
            sheet = Image.new("RGB", (cols * w, rows * h), (0, 0, 0))
            for i, t in enumerate(tiles): sheet.paste(t, ((i % cols) * w, (i // cols) * h))
            sheet.save(os.path.join(a.out, tag + "_planche.png"))
        acts2 = [(r["a"], r["x"], r["y"]) for r in l2]
        R2 = [r for r in R if r["lv"] == 1]                                         # niveau 2 : quels clics ont VRAIMENT changé l'écran (hors 2 lignes du haut = HUD)
        eff2 = []
        for i in range(len(R2) - 1):
            if R2[i]["step"] == 0: continue
            d = int((R2[i]["f"][2:] != R2[i + 1]["f"][2:]).sum())
            if d > 0: eff2.append((R2[i]["step"], R2[i]["x"], R2[i]["y"], int(R2[i]["f"][R2[i]["y"], R2[i]["x"]]) if R2[i]["y"] >= 0 else -1, d))
        print("RECETTE", tag, "win_desc (couleur, taille) =", getattr(ag, "win_desc", None), flush=True)
        print("EFFETS_NIV2", tag, len(eff2), "clics efficaces sur", len(l2), ":", eff2[:30], flush=True)
        print(json.dumps(dict(run=tag, levels_at=lv_at, actions=n, niveau2_gestes=len(acts2),
                              niveau2_types={str(k): sum(1 for t in acts2 if t[0] == k) for k in sorted({t[0] for t in acts2})},
                              niveau2_clics_distincts=len({(t[1] // 8, t[2] // 8) for t in acts2 if t[0] == 6}))), flush=True)


if __name__ == "__main__":
    main()
