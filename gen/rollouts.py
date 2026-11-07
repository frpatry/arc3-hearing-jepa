"""Rollouts sur les jeux générés.

(a) évaluation : Explorer (arc3/explorer.py) vs agent aléatoire, niveaux réussis + score type ARC (baseline/actions)²
    python -m gen.rollouts eval  [--n 32] [--budget 300] [--seed0 0]
(b) transitions pour l'entraînement du world model (politique mixte explorer + aléatoire + oracle), shards .npz
    python -m gen.rollouts dump  [--n 160] [--steps 1500] [--shard 20000] [--g0 0 --g1 0] [--out gen/data]
    arrays : obs (N,64,64) int8, act (N,) int8 (0 = RESET), click (N,2) int8 (x, y ; -1 sinon), next_obs (N,64,64) int8,
             events (N,E) float32 (cf. core.EVENT_NAMES, aussi dans event_names), game (N,) int16 -> game_ids[game],
             level (N,) int8 (niveaux réussis avant l'action), state (N,) int8 après l'action
             (0 NOT_FINISHED, 1 WIN, 2 GAME_OVER), done (N,) bool (niveau gagné ou GAME_OVER),
             behav (N,) int8 (0 explorer, 1 aléatoire, 2 oracle)
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                     # arc3/ -> explorer.py
import explorer as _explorer_mod  # noqa: E402
from explorer import Explorer  # noqa: E402

from . import fastcomp  # noqa: E402

_explorer_mod.components = fastcomp.components          # segmentation en objets ~5x plus rapide, même sortie

from .core import EV, EVENT_NAMES, N_EVENTS  # noqa: E402
from .registry import PRINCIPLES, game_specs, make_game, mixture_specs, rules_specs, v3_specs, v4_specs  # noqa: E402
from .families_v3 import MECHANIC_V3  # noqa: E402
from .families_v4 import MECHANIC_V4  # noqa: E402

STATE_CODE = {"NOT_FINISHED": 0, "WIN": 1, "GAME_OVER": 2}


class RandomAgent:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, grid, state, levels, avail):
        if state == "GAME_OVER":
            return 0, None
        a = int(self.rng.choice([x for x in avail if x != 0]))
        if a == 6:
            return 6, {"x": int(self.rng.integers(64)), "y": int(self.rng.integers(64))}
        return a, None


def level_score(base, n):
    return min(1.0, (base / n) ** 2) if n > 0 else 0.0


def play(game, agent, budget):
    """joue jusqu'à WIN ou `budget` actions (RESET gratuit) -> niveaux, actions/niveau, score du jeu (pondéré par indice)."""
    obs = game.reset()
    n, n_lvl, per_level, resets = 0, 0, [], 0
    while n < budget and obs["state"] != "WIN":
        a, d = agent.act(obs["frame"], obs["state"], obs["levels_completed"], obs["available_actions"])
        lv0 = obs["levels_completed"]
        obs = game.step(a, d)
        if a == 0:
            resets += 1
            if resets > 10 * budget:
                break
            continue
        n += 1
        n_lvl += 1
        if obs["levels_completed"] > lv0:
            per_level.append(n_lvl)
            n_lvl = 0
    w = np.arange(1, game.win_levels + 1)
    sc = [level_score(game.baseline_actions[k], per_level[k]) if k < len(per_level) else 0.0 for k in range(game.win_levels)]
    return dict(levels=obs["levels_completed"], win_levels=game.win_levels, actions=n, per_level=per_level,
                score=float((w * np.array(sc)).sum() / w.sum()))


def _specs(a):
    if getattr(a, "v4", False):                      # mélange v4 (gen/v4/, mécaniques officielles)
        fams = [f for f in a.families.split(",") if f] if getattr(a, "families", "") else None
        if fams:
            return [(fams[k % len(fams)], a.seed0 + k, "official") for k in range(a.n)]
        return v4_specs(a.n, a.seed0)
    if getattr(a, "v3", False):                      # mélange v3 (gen/families_v3.py, niveaux longs)
        return v3_specs(a.n, a.seed0)
    if a.demo_mix:                                   # moitié jeux à règles, moitié familles v2 (look officiel surtout)
        R = rules_specs((a.n + 1) // 2, a.seed0)
        V2 = mixture_specs(a.n // 2, a.seed0, official_share=0.85, rules_share=0)
        return [x for pair in zip(R, V2 + [None]) for x in pair if x is not None][:a.n]
    if a.rules:
        return rules_specs(a.n, a.seed0)
    if a.mix:
        return mixture_specs(a.n, a.seed0, rules_share=a.rules_share)
    return [(f, s, "classic") for f, s in game_specs(a.n, a.seed0)]


def cmd_eval(a):
    rows = []
    t0 = time.time()
    specs = _specs(a)
    for fam, seed, look in specs:
        g = make_game(fam, seed, look)
        rx = play(g, Explorer(seed), a.budget)
        rr = play(make_game(fam, seed, look), RandomAgent(seed), a.budget)
        rows.append(dict(game=g.game_id, baseline=g.baseline_actions, explorer=rx, random=rr))
        print(f"{g.game_id:18s} niv {g.win_levels}  base niv1 {g.baseline_actions[0]:3d} | explorer {rx['levels']} "
              f"{rx['per_level']} score {100 * rx['score']:.1f} | aléa {rr['levels']} {rr['per_level']}", flush=True)
    ex = [r["explorer"] for r in rows]
    ra = [r["random"] for r in rows]
    print(f"\n{len(rows)} jeux, budget {a.budget} actions, {time.time() - t0:.0f}s")
    for name, rs in (("explorer", ex), ("aléatoire", ra)):
        print(f"{name:10s} niveaux {sum(r['levels'] for r in rs)}/{sum(r['win_levels'] for r in rs)}  "
              f"jeux avec >=1 niveau {sum(r['levels'] > 0 for r in rs)}/{len(rs)}  jeux finis {sum(r['levels'] == r['win_levels'] for r in rs)}  "
              f"score moyen {100 * np.mean([r['score'] for r in rs]):.2f}")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1))


FIELDS = dict(obs=((64, 64), np.int8), next_obs=((64, 64), np.int8), act=((), np.int8), click=((2,), np.int8),
              events=((N_EVENTS,), np.float32), game=((), np.int16), level=((), np.int8), state=((), np.int8),
              done=((), bool), behav=((), np.int8))


def cmd_dump(a):
    """politique mixte : explorer (avec ε aléatoire) + segments « oracle » (chemin optimal vers le but)."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    specs = _specs(a)
    game_ids = np.array([f"{f}-{'o' if lk == 'official' else ''}{s:04d}" for f, s, lk in specs])
    meta = []
    g1 = a.g1 if a.g1 > 0 else len(specs)
    rng = np.random.default_rng([a.seed0, a.g0])
    buf = {k: np.zeros((a.shard,) + sh, dt) for k, (sh, dt) in FIELDS.items()}
    shard, nb, total, t0 = 0, 0, 0, time.time()

    def flush():
        nonlocal shard, nb
        if nb == 0:
            return
        p = out / f"shard_g{a.g0:03d}_{shard:03d}.npz"
        np.savez_compressed(p, game_ids=game_ids, event_names=np.array(EVENT_NAMES), **{k: v[:nb] for k, v in buf.items()})
        print(f"  {p.name}: {nb} transitions, {p.stat().st_size / 1e6:.1f} Mo, {time.time() - t0:.0f}s", flush=True)
        shard += 1
        nb = 0

    for gi in range(a.g0, g1):
        fam, seed, look = specs[gi]
        g = make_game(fam, seed, look)
        assert g.game_id == game_ids[gi]
        meta.append(dict(index=gi, game_id=g.game_id, family=fam, look=look, seed=seed, win_levels=g.win_levels,
                         baseline_actions=g.baseline_actions, available_actions=g.available_actions,
                         principle=PRINCIPLES.get(fam, ("", []))[0], official_games=PRINCIPLES.get(fam, ("", []))[1],
                         **({"rules": g.info()["rules"], "attempt": g.attempt, "scheme": g.spec["scheme"],
                             "win": g.spec["win"]["kind"], "arche": g.spec["arche"],
                             "modules": [m["name"] for m in g.spec["modules"]]} if fam == "rules" else {})))
        ex, rnd = Explorer(seed), RandomAgent(seed + 1000)
        eps = float(rng.uniform(0.15, 0.6))                  # part d'actions aléatoires de ce jeu
        obs = g.reset()
        oracle_left = 0
        for _ in range(a.steps):
            if obs["state"] == "WIN":
                obs = g.step(0)
            act, d = ex.act(obs["frame"], obs["state"], obs["levels_completed"], obs["available_actions"])
            behav = 0
            if oracle_left == 0 and rng.random() < a.p_oracle:
                oracle_left = int(rng.integers(8, 41))
            if act != 0 and oracle_left > 0:
                act, d = g.oracle_action()                    # privilégié : vers le but (RESET si hors du chemin connu)
                ex.last = None
                behav = 2
                oracle_left -= 1
            elif act != 0 and rng.random() < eps:
                act, d = rnd.act(obs["frame"], obs["state"], obs["levels_completed"], obs["available_actions"])
                ex.last = None                                # l'explorer n'attribue pas cet effet à son action
                behav = 1
            nxt = g.step(act, d)
            if nxt["events"][EV["level_complete"]]:
                oracle_left = 0
            row = dict(obs=obs["frame"], next_obs=nxt["frame"], act=act, click=(d["x"], d["y"]) if d else (-1, -1),
                       events=nxt["events"], game=gi, level=obs["levels_completed"], state=STATE_CODE[nxt["state"]],
                       done=nxt["levels_completed"] > obs["levels_completed"] or nxt["state"] == "GAME_OVER", behav=behav)
            for k, v in row.items():
                buf[k][nb] = v
            nb += 1
            total += 1
            obs = nxt
            if nb >= a.shard:
                flush()
        if (gi - a.g0) % 10 == 9:
            print(f"jeu {gi + 1}/{g1} : {total} transitions, {time.time() - t0:.0f}s", flush=True)
    flush()
    (out / ("games.json" if a.g0 == 0 and g1 == len(specs) else f"games_g{a.g0:03d}.json")).write_text(
        json.dumps([None] * a.g0 + meta, indent=0))
    size = sum(p.stat().st_size for p in out.glob(f"shard_g{a.g0:03d}_*.npz"))
    print(f"{total} transitions, jeux {a.g0}..{g1 - 1}, {size / 1e6:.1f} Mo dans {out}, {time.time() - t0:.0f}s")


# ============================================================================ mode « demo_play » (vidéos parent -> bébé)
DEMO_FIELDS = dict(FIELDS, ep=((), np.int32), dup=((), bool), mode=((), np.int8))
DEMO_BEHAV = ("oracle", "explorer")
DEMO_MODES = ("full_demo", "demo_play")
BRANCH_P = np.array([0.5, 0.3, 0.2])


def snapshot(g):
    """état dynamique complet d'un Game (pour rejouer plusieurs variantes depuis le même point de démonstration)."""
    return (g.level, g.st, g.energy, g.level_actions, tuple(g.history), g.lives, g.state, g.levels_completed,
            g.total_actions)


def restore(g, s):
    (g.level, g.st, g.energy, g.level_actions, h, g.lives, g.state, g.levels_completed, g.total_actions) = s
    g.history = list(h)


def demo_game(g, seed, steps, rng, p_full=0.4, max_branch=3, ep0=0):
    """Un jeu joué en mode demo_play -> liste de lignes (dict). behav : 0 oracle, 1 explorer.
    - démonstration complète (p_full) : l'oracle résout les niveaux l'un après l'autre (un épisode) ;
    - sinon, pour chaque niveau : l'oracle joue un préfixe de la solution (0..len), puis 1 à `max_branch` VARIANTES
      (50/30/20 %) où l'Explorer joue m ~ U(16, 40) pas depuis ce même point (instantané du jeu et de l'Explorer) ;
      enfin l'oracle reprend (plan exact, ou suit la solution, ou RESET + démo) jusqu'au niveau suivant.
    Épisode (`ep`) = images consécutives sans coupure : il ne change qu'au retour à un instantané (variante n° 2, 3) ;
    une variante recopie alors les ≤ 12 dernières lignes du préfixe (dup=True) pour garder le contexte de la démo."""
    rows, n = [], 0
    ep = [ep0]
    mode = 0 if rng.random() < p_full else 1

    def rec(obs, act, d, nxt, behav, dup=False):
        rows.append(dict(obs=obs["frame"], next_obs=nxt["frame"], act=act, click=(d["x"], d["y"]) if d else (-1, -1),
                         events=nxt["events"], level=obs["levels_completed"], state=STATE_CODE[nxt["state"]],
                         done=nxt["levels_completed"] > obs["levels_completed"] or nxt["state"] == "GAME_OVER",
                         behav=behav, ep=ep[0], dup=dup, mode=mode))

    def oracle_step(obs):
        act, d = g.oracle_action()
        nxt = g.step(act, d)
        rec(obs, act, d, nxt, 0)
        return nxt

    obs = g.reset()
    if mode == 0:
        while n < steps and obs["state"] != "WIN":
            obs = oracle_step(obs)
            n += 1
        return rows, ep[0] + 1
    ex = Explorer(seed)
    while n < steps and obs["state"] != "WIN":
        if g.state != "NOT_FINISHED" or g.level_actions > 0:          # revenir au début du niveau
            obs = oracle_step(obs) if g.state == "GAME_OVER" else _reset_rec(g, obs, rec)
            n += 1
            continue
        li, lv0 = g.level, g.levels_completed
        sol = g.solution_actions(li)
        k = int(rng.integers(0, len(sol) + 1))
        prefix = []
        for act, d in sol[:k]:
            if n >= steps:
                break
            nxt = g.step(act, d)
            rec(obs, act, d, nxt, 0)
            prefix.append(rows[-1])
            obs, n = nxt, n + 1
        if g.levels_completed > lv0 or obs["state"] == "WIN" or n >= steps:
            continue
        P, exP, obsP = snapshot(g), copy.deepcopy(ex), obs
        exP.last = None                                                # l'Explorer n'attribue pas les coups de l'oracle
        nb = int(rng.choice(np.arange(1, max_branch + 1), p=BRANCH_P[:max_branch] / BRANCH_P[:max_branch].sum()))
        for b in range(nb):
            if b > 0:                                                  # variante contrefactuelle du même point
                restore(g, P)
                obs = obsP
                ep[0] += 1
                for r in prefix[-12:]:
                    rows.append(dict(r, ep=ep[0], dup=True))
            ex = copy.deepcopy(exP)
            for _ in range(int(rng.integers(16, 41))):
                if n >= steps or obs["state"] == "WIN" or g.levels_completed > lv0:
                    break
                act, d = ex.act(obs["frame"], obs["state"], obs["levels_completed"], obs["available_actions"])
                nxt = g.step(act, d)
                rec(obs, act, d, nxt, 1)
                obs, n = nxt, n + 1
        ex.last = None
        for _ in range(3 * len(sol) + 40):                             # l'oracle reprend jusqu'au niveau suivant
            if n >= steps or obs["state"] == "WIN" or g.levels_completed > lv0:
                break
            obs = oracle_step(obs)
            n += 1
    return rows, ep[0] + 1


def _reset_rec(g, obs, rec):
    nxt = g.step(0)
    rec(obs, 0, None, nxt, 0)
    return nxt


def cmd_demo(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    specs = _specs(a)
    game_ids = np.array([f"{f}-{'o' if lk == 'official' else ''}{s:04d}" for f, s, lk in specs])
    g1 = a.g1 if a.g1 > 0 else len(specs)
    rng = np.random.default_rng([a.seed0, a.g0, 77])
    buf, meta, shard, total, t0 = [], [], 0, 0, time.time()

    def flush():
        nonlocal shard, buf
        if not buf:
            return
        p = out / f"{a.prefix}_g{a.g0:04d}_{shard:03d}.npz"
        arrs = {k: np.array([r[k] for r in buf], dtype=dt).reshape((len(buf),) + sh)
                for k, (sh, dt) in DEMO_FIELDS.items()}
        np.savez_compressed(p, game_ids=game_ids, event_names=np.array(EVENT_NAMES), behav_names=np.array(DEMO_BEHAV),
                            mode_names=np.array(DEMO_MODES), **arrs)
        print(f"  {p.name}: {len(buf)} transitions, {p.stat().st_size / 1e6:.1f} Mo, {time.time() - t0:.0f}s", flush=True)
        shard += 1
        buf = []

    for gi in range(a.g0, g1):
        fam, seed, look = specs[gi]
        try:
            g = make_game(fam, seed, look)
        except Exception as e:                                         # jeu impossible à construire : ignoré
            print(f"ERREUR construction {game_ids[gi]} ({fam}) : {e!r} -> jeu ignoré", flush=True)
            continue
        assert g.game_id == game_ids[gi]
        try:
            steps = a.steps if a.adapt <= 0 else int(a.adapt * (sum(g.baseline_actions) + 60 * g.win_levels))
            rows, _ = demo_game(g, seed, steps, rng, p_full=a.p_full, ep0=gi * 1000)
        except Exception as e:                                         # jeu fautif : ignoré (noté dans le journal)
            import traceback
            print(f"ERREUR {g.game_id} ({fam}) : {e!r} -> jeu ignoré\n{traceback.format_exc()}", flush=True)
            continue
        for r in rows:
            r["game"] = gi
        buf += rows
        total += len(rows)
        meta.append(dict(index=gi, game_id=g.game_id, family=fam, look=look, seed=seed, win_levels=g.win_levels,
                         baseline_actions=g.baseline_actions, available_actions=g.available_actions,
                         principle=PRINCIPLES.get(fam, ("", []))[0], official_games=PRINCIPLES.get(fam, ("", []))[1],
                         mode=DEMO_MODES[rows[0]["mode"]] if rows else None,
                         **({"mechanic": MECHANIC_V3[fam], "rules_per_level": g.info().get("rules_per_level"),
                             "base": getattr(g, "base_name", None)} if fam in MECHANIC_V3 else {}),
                         **({"mechanic": MECHANIC_V4[fam], "rules_per_level": g.info().get("rules_per_level"),
                             "variant": getattr(g, "variant", None)} if fam in MECHANIC_V4 else {}),
                         **({"rules": g.info()["rules"], "attempt": g.attempt} if fam == "rules" else {})))
        if len(buf) >= a.shard:                                        # coupe aux frontières de jeu (épisodes entiers)
            flush()
        if (gi - a.g0) % 10 == 9:
            print(f"jeu {gi + 1}/{g1} : {total} transitions, {time.time() - t0:.0f}s", flush=True)
    flush()
    (out / ("games.json" if a.g0 == 0 and g1 == len(specs) else f"games_g{a.g0:04d}.json")).write_text(
        json.dumps([None] * a.g0 + meta, indent=0))
    print(f"{total} transitions, jeux {a.g0}..{g1 - 1}, dans {out}, {time.time() - t0:.0f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["eval", "dump"])
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--budget", type=int, default=300)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--shard", type=int, default=20000)
    p.add_argument("--g0", type=int, default=0, help="dump : premier jeu (index dans game_specs(n))")
    p.add_argument("--g1", type=int, default=0, help="dump : jeu de fin exclu (0 = tous)")
    p.add_argument("--p_oracle", type=float, default=0.025, help="dump : proba de lancer un segment oracle")
    p.add_argument("--mix", action="store_true", help="dump : mélange v2 (sample_spec) au lieu des 16 familles v1")
    p.add_argument("--rules_share", type=float, default=0.25, help="--mix : part de jeux à règles (0 = mélange v2)")
    p.add_argument("--rules", action="store_true", help="seulement des jeux à règles (grammaire, gen/grammar.py)")
    p.add_argument("--v3", action="store_true", help="mélange v3 (familles v3 : niveaux longs, règle nouvelle par niveau)")
    p.add_argument("--v4", action="store_true", help="mélange v4 (gen/v4 : mécaniques et rendu officiels)")
    p.add_argument("--families", default="", help="--v4 : liste de familles en tourniquet (au lieu du mélange)")
    p.add_argument("--demo_mix", action="store_true", help="moitié jeux à règles, moitié familles v2 (entrelacés)")
    p.add_argument("--behav", default="mixed", choices=["mixed", "demo_play"],
                   help="dump : mixed = explorer/aléa/oracle ; demo_play = démonstrations de l'oracle + jeu de l'Explorer")
    p.add_argument("--p_full", type=float, default=0.4, help="demo_play : part de démonstrations complètes")
    p.add_argument("--prefix", default="shard", help="demo_play : préfixe des fichiers")
    p.add_argument("--adapt", type=float, default=0.0,
                   help="demo_play : budget ADAPTATIF par jeu = adapt × (somme des solutions + 60 par niveau pour les variantes "
                        "de l'Explorer) au lieu de --steps fixe -> les jeux longs / à beaucoup de niveaux sont joués jusqu'au bout "
                        "(v4 à --steps fixe : ~40 %% des derniers niveaux, les plus durs, jamais montrés résolus)")
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--out", default="")
    a = p.parse_args()
    if a.cmd == "eval":
        cmd_eval(a)
    else:
        a.out = a.out or str(HERE / "data")
        if a.behav == "demo_play":
            cmd_demo(a)
        else:
            cmd_dump(a)


if __name__ == "__main__":
    main()
