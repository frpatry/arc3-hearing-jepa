# Listening to the Game: a Hearing JEPA World Model for ARC-AGI-3

Code for our **ARC Prize 2026** entry (ARC-AGI-3 track, Kaggle submission **57710722**, public LB 0.16) and its Paper Track writeup.

A block-causal JEPA video world model (23.6 M parameters) **hears** a synthetic sound channel computed from the frames alone — a spatial *toc* map of where the screen changed, a *ding* when a level is won, a *buzz* on game over — as input, predicts those sounds as targets, and plans 12 imagined steps ahead toward the ding.

- Paper: [`paper/writeup.md`](paper/writeup.md) · [Kaggle writeup](https://www.kaggle.com/competitions/arc-prize-2026-paper-track/writeups/listening-to-the-game-a-hearing-jepa-world-model)
- Submission notebook: [kaggle.com/code/francoispatry/arc3-submit-critim](https://www.kaggle.com/code/francoispatry/arc3-submit-critim) (model weights: Kaggle dataset `francoispatry/arc3-jepa-critim`, file `wm_seq_critim.pt`)
- Video, the submitted model playing vc33 with its sounds: [`paper/fig/vc33_model_sound.mp4`](paper/fig/vc33_model_sound.mp4)

## Layout

| Folder | Content |
|---|---|
| `submission/` | **Exact code of submission 57710722**, extracted from the notebook: `my_agent.py` (Kaggle agent), `plan_agent.py` (planner-driven agent, focus rule, in-game learning), `cogmap.py` (cognitive map), `explorer.py`, `wm/model_seq.py` (world model), `wm/planner.py` (gradient planner in imagination), plus `submission.ipynb` |
| `wm/` | World model and training as of the submission period: `model_seq.py`, `train_seq.py` (training, hearing `--hear`, intent/dream, imagination repair `--train_only crit+proj --w_roll`), `planner.py`, `sound_test.py` (hearing vs deaf test), `dump_public.py` (recording public games). These files evolved after the submission (heat, voice); use `submission/` for the exact submitted versions |
| `gen/` | Procedural ARC-AGI-3-like game generator (19 families in `gen/v4/`), written from the mechanics principles in `analysis/MECHANICS.md`; no official code or sprites reused |
| `analysis/` | `MECHANICS.md`: 21 mechanics principles distilled from the 25 public games |
| `blind/` | The "blind" model explored later: no pixels, only actions, objects and fabricated sounds (`blind_tokens.py` defines the sounds) |
| `eval/` | Approximate local scoring (`colab_off.py`), rank / first-move / order tests, game recording and videos, random baseline |
| `paper/` | Writeup, cover figure, and `fig/sound_video.py` which renders the sound videos |

## Notes

- Training was run on Colab (one RTX PRO 6000, ≈16.5 h, 500 k steps); scripts carry Colab/Drive paths and French comments.
- Training data (generated games and recordings of the public games) and the oracle solutions of the public games are not included; the generator in `gen/` regenerates the former.
- The public ARC-AGI-3 environments come from the competition data (`arc_agi_3_wheels`, `environment_files`).

License: Apache 2.0.
