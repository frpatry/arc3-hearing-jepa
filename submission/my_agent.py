"""Agent ARC-AGI-3 « System-2 » (projet JEPA) : un modèle du monde JEPA VIDÉO (mémoire des derniers pas de la partie) imagine
plusieurs pas ; un planificateur fait descendre le GRADIENT d'une note imaginée (critique « fin de niveau proche », nouveauté,
effet, souvenir des réussites, danger) sur des séquences d'actions assouplies (Mode 2 de LeCun), puis joue la 1re action.
Mémoire des essais (pas deux fois le même geste dans le même état), clics accrochés aux objets, apprentissage pendant la partie
(renforcé au début). Pas de repli sur le v1. Arrêt des parties vers 8 h (limite 9 h)."""
from __future__ import annotations
from typing import Any
import base64, glob, os, sys, threading, time, zlib
from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent

FILES = {...}  # embedded copies of the files in this folder (zlib+base64), see submission.ipynb
D = "/tmp/arc3code"
for p, b in FILES.items():
    os.makedirs(os.path.dirname(f"{D}/{p}"), exist_ok=True)
    open(f"{D}/{p}", "wb").write(zlib.decompress(base64.b64decode(b)))
sys.path[:0] = [D, f"{D}/wm"]
T0 = time.time()
class FairSem:                                # verrou ÉQUITABLE (file d'attente) : threading.Semaphore laissait 7 jeux sur 25 monopoliser
    def __init__(s, n):                      # la carte (un fil qui relâche reprend aussitôt) -> 18 jeux affamés (30 actions en 30 min, 27 oct.)
        s.n, s.c, s.q = n, threading.Condition(), []

    def __enter__(s):
        me = object()
        with s.c:
            s.q.append(me)
            while s.q[0] is not me or s.n == 0: s.c.wait()
            s.q.pop(0); s.n -= 1; s.c.notify_all()

    def __exit__(s, *a):
        with s.c: s.n += 1; s.c.notify_all()


GPU = FairSem(3)              # au plus 3 jeux CALCULENT en même temps sur la carte, chacun son tour (file d'attente)
NEED_GB = 2.0                             # mémoire GPU libre exigée avant de charger le modèle d'un nouveau jeu
STOP_SEC = 8.0 * 3600                     # marge sous la limite de 9 h de la compétition
LIGHT_SEC = 5.0 * 3600                    # au-delà : réflexion allégée (4 itérations au lieu de 12)
LVL_BUDGET = 100                          # 27 oct. (user) : au-delà de 100 actions sans nouveau niveau, le jeu s'arrête (score ≈ 0 de toute façon, temps rendu aux autres jeux)
W = dict(crit=3.0, lc=1.0, nov=1.0, eff=0.2, mem=0.5, go=1.0, dream=3.0, dist=3.0)   # critique ×3 + distance au but (étude vc33 27 oct.)
CKPT = (sorted(glob.glob("/kaggle/input/**/wm_seq*.pt", recursive=True)) or [None])[0]
print("modèle du monde :", CKPT, flush=True)


class MyAgent(Agent):
    MAX_ACTIONS = 3000                    # choix user 26 oct. : les actions en trop sur un niveau raté ne coûtent rien au score

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = abs(hash(self.game_id)) % 100000
        from plan_agent import PlanAgent
        assert CKPT, "modèle du monde introuvable dans /kaggle/input"
        import torch
        while torch.cuda.is_available() and time.time() - T0 < STOP_SEC:          # tous les jeux cachés démarrent ensemble : attendre la place
            free = torch.cuda.mem_get_info()[0] + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()   # la réserve inutilisée de PyTorch compte comme libre
            if free > NEED_GB * 2 ** 30: break
            time.sleep(20)
        with GPU:
            self.ex = PlanAgent(CKPT, seed, "grad", H=12, seeds=8, iters=12, w=W, focus=True)   # H=12 + critique ×3 + FOCUS (meilleur réglage du 27 oct.)
        self.n_act, self.lv, self.lv_at = 0, 0, 0

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        if latest_frame.levels_completed > self.lv: self.lv, self.lv_at = latest_frame.levels_completed, self.n_act
        return latest_frame.state is GameState.WIN or time.time() - T0 > STOP_SEC or self.n_act - self.lv_at >= LVL_BUDGET

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        if time.time() - T0 > LIGHT_SEC: self.ex.pl.iters = 4
        st = latest_frame.state.name
        grid = latest_frame.frame[-1] if latest_frame.frame else []
        if st in ("NOT_PLAYED", "GAME_OVER") or not latest_frame.frame:
            with GPU: self.ex.act(grid, "GAME_OVER" if st != "NOT_PLAYED" else st, latest_frame.levels_completed, list(latest_frame.available_actions))
            return GameAction.RESET
        import numpy as np
        fr = latest_frame.frame; anim = len(fr) > 1 and any(not np.array_equal(f, fr[-1]) for f in fr[:-1])   # effet visible seulement PENDANT l'animation
        with GPU: a, data = self.ex.act(grid, st, latest_frame.levels_completed, list(latest_frame.available_actions), anim=anim)
        self.n_act += int(a) != 0
        action = GameAction.from_id(int(a))
        if action.is_complex():
            action.set_data(data or {"x": 32, "y": 32})
        return action
