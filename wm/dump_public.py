"""Récolte d'expériences sur les jeux PUBLICS (hors ligne) pour le modèle du monde JEPA :
une trajectoire par partie = suite d'images 64×64 + actions (id, clic x,y) + niveau + état.
Politique de comportement = mélange Explorer (curiosité, objets) et hasard (sur les objets pour les clics),
pour voir à la fois des actions « utiles » et variées. RESET après GAME_OVER (gratuit, non enregistré comme action).
    .venv/bin/python wm/dump_public.py --steps 2000 --eps 0.3        -> wm/data/public/<jeu>.npz
NB : copier chaque image (certains jeux, ex. lf52/bp35, réutilisent le MÊME tableau d'une image à l'autre -> enregistrements faux avant 2026-10-24).
Format npz : frames (T+1,64,64) int8 | act (T,) int8 | click (T,2) int8 (-1 sans clic) | level (T,) int8 | state (T,) int8
(0 en cours, 1 niveau réussi, 2 GAME_OVER, 3 WIN : ce qu'a produit l'action t) ; un segment s'arrête à chaque RESET :
seg (T,) int16 = numéro de segment (frames[t] -> frames[t+1] n'est une vraie transition que si seg[t] == seg[t+1] ou t = fin)."""
import argparse, logging, os, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
os.environ["OPERATION_MODE"] = "offline"
os.environ.setdefault("ENVIRONMENTS_DIR", str(ROOT / "data" / "environment_files"))
from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction  # noqa: E402
from explorer import Explorer, components  # noqa: E402


def dump(arc, gid, steps, eps, seed, out):
    rng = np.random.default_rng(seed); env = arc.make(gid); r = env.observation_space; ag = Explorer(seed)
    frames, act, click, level, state, seg = [], [], [], [], [], []
    s = 0; cur = np.array(r.frame[-1], np.int8, copy=True); n = 0; t0 = time.time()
    trans = []                                                        # (frame_t, a, x, y, frame_t1, lvl, st, seg)
    while n < steps:
        if r is None: break
        if r.state.name == "WIN": break
        if not r.frame or r.state.name == "GAME_OVER":
            ag.act(cur, "GAME_OVER", r.levels_completed, list(r.available_actions))
            r = env.step(GameAction.RESET); s += 1
            if r is None or not r.frame: break
            cur = np.array(r.frame[-1], np.int8, copy=True); continue
        avail = [a for a in r.available_actions if a != 0]
        a, data = ag.act(cur, r.state.name, r.levels_completed, list(r.available_actions))
        if a == 0 or rng.random() < eps:                              # exploration au hasard (clics sur des objets)
            a = int(rng.choice(avail)); data = None
            if a == 6:
                objs = components(cur.astype(np.int16)) or [(0, 1, (int(rng.integers(64)), int(rng.integers(64))))]
                y, x = objs[int(rng.integers(len(objs)))][2]; data = {"x": x, "y": y}
        lv0 = r.levels_completed
        r = env.step(GameAction.from_id(a), data=data or {}); n += 1
        if r is None or not r.frame: nxt = cur; st = 2
        else:
            nxt = np.array(r.frame[-1], np.int8, copy=True)
            st = 3 if r.state.name == "WIN" else 2 if r.state.name == "GAME_OVER" else 1 if r.levels_completed > lv0 else 0
        trans.append((cur, a, (data or {}).get("x", -1), (data or {}).get("y", -1), nxt, lv0, st, s))
        cur = nxt
    if not trans: return 0
    np.savez_compressed(out, obs=np.stack([t[0] for t in trans]), next_obs=np.stack([t[4] for t in trans]),
                        act=np.array([t[1] for t in trans], np.int8), click=np.array([(t[2], t[3]) for t in trans], np.int8),
                        level=np.array([t[5] for t in trans], np.int8), state=np.array([t[6] for t in trans], np.int8),
                        seg=np.array([t[7] for t in trans], np.int16))
    return len(trans)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--steps", type=int, default=2000); p.add_argument("--eps", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--games", default="")
    a = p.parse_args(); logging.disable(logging.INFO)
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=os.environ["ENVIRONMENTS_DIR"])
    ids = sorted(e.game_id for e in arc.get_environments())
    if a.games: ids = [g for g in ids if any(g.startswith(k) for k in a.games.split(","))]
    od = HERE / "data" / "public"; od.mkdir(parents=True, exist_ok=True); tot = 0
    for gid in ids:
        t0 = time.time(); n = dump(arc, gid, a.steps, a.eps, a.seed, od / f"{gid[:4]}.npz"); tot += n
        print(f"{gid[:4]} {n} transitions ({time.time() - t0:.0f}s)", flush=True)
    print(f"total {tot} transitions -> {od}")


if __name__ == "__main__":
    main()
