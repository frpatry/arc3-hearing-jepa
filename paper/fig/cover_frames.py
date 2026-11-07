# rejoue la solution oracle du niveau 1 de quelques jeux publics, garde chaque image + le geste
import sys, json, numpy as np
from pathlib import Path; H = str(Path(__file__).resolve().parents[2]); sys.path[:0] = [H, H + "/data/oracle_public"]
import oracle_lib as L
from arcengine import GameAction
out = {}
for game in sys.argv[1].split(","):
    sol = json.load(open(f"{H}/data/oracle_public/{game}_solution.json"))["solution"]
    env = L.make_env(game); fr = np.asarray(env.step(GameAction.RESET).frame[-1], np.int8); g = L.clone(env._game)
    F, A = [fr], []
    for (x, cx, cy) in sol[0]:
        fr = L.frame_of(L.gstep(g, x, cx, cy)); F.append(fr); A.append((x, cx, cy))
    out[game] = (np.stack(F), np.array(A)); print(game, len(A), "gestes, score", g._score)
np.savez(sys.argv[2], **{k + "_F": v[0] for k, v in out.items()}, **{k + "_A": v[1] for k, v in out.items()})
