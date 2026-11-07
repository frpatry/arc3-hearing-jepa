"""Random baseline ARC-AGI-3, 100 % hors-ligne (aucune clé API, aucun réseau).

Usage (depuis le dossier arc3/) :
    .venv/bin/python random_baseline.py                # 25 jeux, 300 actions/jeu
    .venv/bin/python random_baseline.py --budget 100 --games ls20,vc33

Pour chaque jeu public de data/environment_files :
  - reset, forme de la grille, actions disponibles, nb de niveaux (win_levels),
    baseline humaine (metadata.json), résolution logique de la caméra
    (privilégié : lu dans l'objet jeu, info de diagnostic seulement),
    nb de cibles cliquables ACTION6 (privilégié, _get_valid_actions) ;
  - agent aléatoire uniforme sur les actions disponibles (ACTION6 : x,y aléatoires
    dans 0..63), RESET sur GAME_OVER ;
  - mesure : niveaux réussis, nb de GAME_OVER, frames distinctes vues,
    % d'actions qui changent l'image (tout changement / changement > 4 px, pour
    ignorer le compteur d'actions du HUD qui bouge à chaque action), par action ;
  - sauvegarde la 1re image en arc3/frames/<jeu>.png ; résumé JSON dans
    arc3/random_baseline_results.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ["OPERATION_MODE"] = "offline"  # l'env var a priorité sur tout le reste
os.environ.setdefault("ENVIRONMENTS_DIR", str(HERE / "data" / "environment_files"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from arc_agi import Arcade, OperationMode  # noqa: E402
from arc_agi.rendering import COLOR_MAP  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402

PALETTE = np.array(
    [[int(COLOR_MAP[i][k : k + 2], 16) for k in (1, 3, 5)] for i in range(16)],
    dtype=np.uint8,
)


# Beaucoup de jeux ont une barre de compteur d'actions/énergie dans le HUD qui
# change de 1-2 pixels à CHAQUE action : on compte aussi les changements > HUD_PIX.
HUD_PIX = 4


def save_png(grid: np.ndarray, path: Path, scale: int = 4) -> None:
    rgb = PALETTE[np.clip(grid.astype(np.int64), 0, 15)]
    img = Image.fromarray(rgb, "RGB").resize(
        (grid.shape[1] * scale, grid.shape[0] * scale), Image.NEAREST
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def fhash(grid: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(grid).tobytes()).hexdigest()


def play(arc: Arcade, game_id: str, budget: int, rng: random.Random, card: str) -> dict:
    t0 = time.time()
    env = arc.make(game_id, scorecard_id=card)  # make() fait déjà le RESET initial
    obs = env.observation_space
    first = obs.frame[-1]
    save_png(first, HERE / "frames" / f"{game_id.split('-')[0]}.png")

    game = getattr(env, "_game", None)  # accès privilégié, diagnostic uniquement
    cam = None
    n_click_targets = None
    if game is not None:
        try:
            scale, xo, yo = game.camera._calculate_scale_and_offset()
            cam = f"{game.camera.width}x{game.camera.height} (x{scale})"
        except Exception:
            pass
        try:
            n_click_targets = sum(
                1 for a in game._get_valid_actions() if a.id == GameAction.ACTION6
            )
        except Exception:
            pass

    info = env.environment_info
    avail = list(obs.available_actions)
    actions = [GameAction.from_id(a) for a in avail if a != 0]

    seen = {fhash(first)}
    tried = defaultdict(int)
    changed = defaultdict(int)
    changed_big = defaultdict(int)  # > HUD_PIX pixels : hors simple compteur d'actions
    n_frames_per_step = []
    game_over = 0
    max_levels = obs.levels_completed
    first_level_at = None
    n_actions = 0
    prev = first
    state = obs.state

    while n_actions < budget:
        if state == GameState.WIN:
            break
        if state == GameState.GAME_OVER:
            game_over += 1
            r = env.step(GameAction.RESET)  # RESET ne compte pas comme action
            if r is None or not r.frame:
                break
            prev, state = r.frame[-1], r.state
            seen.add(fhash(prev))
            continue
        a = rng.choice(actions)
        data = {}
        if a.is_complex():
            data = {"x": rng.randint(0, 63), "y": rng.randint(0, 63)}
        r = env.step(a, data=data)
        n_actions += 1
        if r is None:
            break
        state = r.state
        name = a.name
        tried[name] += 1
        if r.frame:
            cur = r.frame[-1]
            n_frames_per_step.append(len(r.frame))
            npix = int((cur != prev).sum())
            if npix > 0:
                changed[name] += 1
            if npix > HUD_PIX:
                changed_big[name] += 1
            seen.add(fhash(cur))
            prev = cur
        if r.levels_completed > max_levels:
            max_levels = r.levels_completed
            if first_level_at is None:
                first_level_at = n_actions

    return {
        "game": game_id,
        "grid": "x".join(map(str, first.shape)),
        "camera": cam,
        "available_actions": avail,
        "win_levels": obs.win_levels,
        "human_baseline": info.baseline_actions,
        "tags": info.tags,
        "click_targets_at_start": n_click_targets,
        "actions_taken": n_actions,
        "levels_completed": max_levels,
        "first_level_at_action": first_level_at,
        "game_overs": game_over,
        "distinct_frames": len(seen),
        "mean_frames_per_step": round(float(np.mean(n_frames_per_step)), 2)
        if n_frames_per_step
        else None,
        "change_rate": {
            k: round(changed[k] / tried[k], 2) for k in sorted(tried)
        },
        "big_change_rate": {
            k: round(changed_big[k] / tried[k], 2) for k in sorted(tried)
        },
        "final_state": state.name,
        "seconds": round(time.time() - t0, 2),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=int, default=300, help="actions max par jeu")
    p.add_argument("--games", default="", help="ex: ls20,vc33 (défaut : tous)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="random_baseline_results.json")
    args = p.parse_args()

    logging.disable(logging.INFO)  # arc_agi est très bavard
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=os.environ["ENVIRONMENTS_DIR"],
    )
    ids = sorted(e.game_id for e in arc.get_environments())
    if args.games:
        keep = args.games.split(",")
        ids = [g for g in ids if any(g.startswith(k) for k in keep)]
    card = arc.open_scorecard(tags=["random_baseline"])
    rng = random.Random(args.seed)

    results = []
    for gid in ids:
        try:
            res = play(arc, gid, args.budget, rng, card)
        except Exception as e:  # un jeu cassé ne doit pas tout arrêter
            res = {"game": gid, "error": repr(e)}
        results.append(res)
        if "error" in res:
            print(f"{gid:16s} ERREUR {res['error']}")
            continue
        cr = " ".join(
            f"{k[6:]}:{v:.0%}/{res['big_change_rate'][k]:.0%}"
            for k, v in res["change_rate"].items()
        )
        print(
            f"{gid:16s} cam={res['camera'] or '?':14s} act={res['available_actions']} "
            f"lv={res['levels_completed']}/{res['win_levels']} GO={res['game_overs']:3d} "
            f"distinct={res['distinct_frames']:4d} clicks={res['click_targets_at_start']} "
            f"fps={res['mean_frames_per_step']} | {cr} ({res['seconds']}s)",
            flush=True,
        )

    sc = arc.close_scorecard(card)
    if sc is not None:
        print(f"\nScore local (scorecard hors-ligne) : {sc.score:.3f}")
        for env_list in sc.environments:
            for run in env_list.runs:
                if run.score > 0:
                    print(f"  {env_list.id}: score {run.score:.2f}, niveaux {run.levels_completed}")
    out = HERE / args.out
    out.write_text(json.dumps(results, indent=1, default=str))
    print(f"Résultats -> {out}")


if __name__ == "__main__":
    sys.exit(main())
