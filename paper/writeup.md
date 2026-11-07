Submission **57710722** · public LB 0.16 · notebook: [arc3-submit-critim](https://www.kaggle.com/code/francoispatry/arc3-submit-critim) · code: [github.com/frpatry/arc3-hearing-jepa](https://github.com/frpatry/arc3-hearing-jepa) · video: the submitted model playing vc33 with its sounds (media gallery)

## Abstract

ARC-AGI-3 gives an agent no reward and no rules: only 64×64 frames and a level counter. Infants face the same problem, and they do not learn from vision alone: sound tells them what their actions did. We give a JEPA video world model a synthetic auditory channel computed from the frames alone — a spatial "toc" map of where the screen changed, a "ding" when a level is won, a "buzz" on game over — and train it, from scratch, both to hear these sounds as input and to predict them as targets. At test time, a gradient planner imagines 12 steps ahead and optimizes for predicted ding and distance-to-ding. The full system scores 0.16 on the hidden set. Hearing helps transfer to unseen game families (54% vs 48% winning-gesture recovery); judges that are right on real frames collapse in imagination.

## 1. Idea: sound as the missing modality

Pixels are dense and mostly irrelevant; the level counter arrives only at the end. Nothing tells a learner which gesture *mattered*. Animals get it from sound: a click or a chime is low-dimensional, action-locked, and indifferent to appearance.

ARC-AGI-3 games are silent, so we fabricate the sound from frame differences — no game-specific code:

- **toc map**: for each of the 64 8×8 zones, did any pixel change after the action?
- **ding**: the level counter increased;
- **buzz**: game over.

These exist for *every* game, hidden ones included: the auditory channel is universal where pixels are not. As **input**, sound tells the predictor what just happened. As **target**, it forces the latents to encode consequences, and the ding becomes a goal the planner can imagine reaching.

## 2. Method

**World model.** A block-causal JEPA (`SeqWorldModel`, 23.6 M trainable parameters). An "eye" (one-hot colours → strided convolutions → 2-layer transformer) turns each frame into 64 patch tokens (8×8 px). Targets come from an EMA copy of the eye (0.996), one per patch. A predictor (8 layers, d = 384, 8 heads) runs over 16 frames × 64 tokens with a block-causal mask; each frame's tokens receive the action embedding, the click and **the heard sound of the previous transition** (learned vectors for changed zones, ding, buzz). It predicts the next latent as a residual (smooth-L1 to EMA targets plus a variance hinge).

**Heads** read [eye latent, predictor state], so auxiliary losses do not distort the JEPA output:

- `snd`: next toc map (per-zone change fraction);
- `ev`: 23 generator events (push, collect, open, teleport, danger…) (generated games only);
- `crit`: "this action ends the level" / "level ends within 8 steps" — a predicted ding;
- `crit_d`: distance to ding, target 0.97^(steps to level end);
- `act`: an actor imitating oracle gestures;
- **intent**: in half of the training windows the future actions are replaced by "?" plus an intent token *(win within 8 steps: yes/no)*.

**Data.** We wrote a procedural generator of 19 game families (pushing, keys, gravity, painting, rotors…) from 21 mechanics principles distilled by watching the public games — no official code or sprites reused. Training used about 7,000 generated games (oracle-solved at ~99% of levels) plus explorer recordings of the 25 public games and oracle replays of 19 of them; **5 public games were never trained on** (g50t, ka59, lf52, r11l, sk48). 3.3 M windows, 500 k steps, ≈16.5 h on one RTX PRO 6000.

**Repairing imagination.** Multi-step planning feeds imagined latents back through a projection `proj`, which the long run never trained: imagination was noise from step 2 (JEPA/copy error ratio 147). A 4,000-step fine-tune of `proj` and the judges only (trunk frozen), with rollout losses through 4 imagined steps and judges trained on *imagined* states, brought the ratio to 0.98 / 1.05 / 1.28 at 2 / 4 / 8 steps.

**Agent.** At each step, 8 soft action sequences of length 12 are optimized by gradient descent through the frozen model (soft actions, continuous clicks), then hardened and rescored. The score of an imagined step is: predicted ding-within-8 ×3, distance-to-ding ×3, novelty ×1, effect ×0.2, similarity to this game's past pre-win states ×0.5, minus predicted buzz; plus a "dream" term rewarding closeness to the future imagined under intent = win. The first gesture also receives priors from a **cognitive map** (graph of real states: penalize tried gestures and deaths, reward the path to the frontier), the actor, and a **focus** rule (never re-click a silent cell; after a win, prefer clicking objects of the same kind as those that produced tocs). The agent keeps learning in-game (a won level becomes a demonstration) and gives up a game after 100 actions without a new level.

## 3. Results

| Agent (hidden set, 0–100 scale) | Score |
|---|---|
| Random (local estimate, public games) | ≈0.04 |
| JEPA world model, no hearing, H = 4 (57672881) | 0.14 |
| **Hearing JEPA + repaired imagination + distance (57710722)** | **0.16** |
| Same + inner "voice" (hears its own predicted progress, 57786538) | 0.13 |
| Hand-written curious explorer, no learning (57590051) | 0.25 |

On public games the hearing model solves vc33 level 1 in 4/4 seeds (the video shows one run: levels 1 and 2 in 40 gestures) and sp80 / r11l level 1 in a 25-game simulation (r11l is a held-out game). On the first frame of 57 level starts, the distance-to-ding head ranks the right gesture top-3 in 47% of cases (random 29%).

**Isolating hearing.** In a controlled fine-tune (3 generated families × 10 games, same model, hearing vs deaf), winning-gesture recovery was 55% vs 52% on training games, 48% vs 48% on new games of seen families, and **54% vs 48% on unseen families** (clicks: 43% vs 32%; chance 13–15%), with no change in next-frame error. Hearing does not make prediction better; it makes the model better at **finding the gesture that matters in a world it has never seen**. We did not run a matched deaf version of the submitted model, which is the main limitation of this paper.

## 4. Why it falls short — what we learned

1. **Judges die in imagination.** The critic recognizes the end of a level on real frames (AUC ≈ 1.0) but, on unseen games, its imagined ding stays at 0.005–0.04 even 8 steps before a real win. When the right sequence ends a level, real frames rank it 1st–2nd among the candidates; imagined frames rank it 22nd–35th. Long imagination loses exactly the rare event the plan is aimed at.
2. **No sound in the middle of a level.** Mid-level, even real frames rank the right sequence at chance. A learned progress sound ("heat") reached AUC 0.88 on new games of *seen* families but 0.56 and 0.44 on official games. Generators copy mechanics, not goals.
3. **Winning screens are nearly unique.** Pre-win screens are near-unique by construction; 0% of the pre-win screens our explorer reached appeared in our recorded winning paths. A model that memorizes states cannot recognize a win it has never heard; variety must come from the path, not the final state.
4. **Learning in-game hurt.** Freezing the model solved vc33 level 2 in 4/4 seeds versus 0–2/4 with in-game updates, and ran 7× faster. Online updates destabilized more than they adapted.

## 5. Why sound, and what next

Sound works where it works because it is the right *abstraction level*: it throws away appearance and keeps consequence, so an event heard in a pushing game ("something moved there") means the same in a painting game. That is what the unseen-family result suggests. What our model did not get is a **sound of progress** that transfers: a ding is universal, but the path to it is silent.

Next steps: (i) plan in sound space, predicting events rather than full latents over long horizons; (ii) learn progress from many diverse winning paths rather than final screens; (iii) let the agent hear its own violated expectations (surprise as sound), a curiosity signal defined for any game. Infants learn mostly through vision, but developmental studies show that information redundant across sight and sound is what first captures their attention and drives perceptual learning (Bahrick & Lickliter, 2000); in machine learning, sound is a free supervisory signal for vision (Owens et al., 2016). We think ARC-AGI-3 agents can use the same bridge.

We also explored the opposite extreme: a model that starts **blind** (no pixels; only actions, objects and fabricated sounds such as toc, contact, pick-up, recolour and ding). It matched but did not beat the scripted explorer on held-out games.

**References.** Bahrick, L. E. & Lickliter, R. (2000). Intersensory redundancy guides attentional selectivity and perceptual learning in infancy. *Developmental Psychology* 36(2). · Owens, A. et al. (2016). Ambient sound provides supervision for visual learning. *ECCV*.
