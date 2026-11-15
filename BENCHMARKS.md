# BENCHMARKS — what was measured, how, and what it does and does not show

Every number in this file was produced by code in this repository. Raw data is in
`results/` (JSONL per episode, JSON per benchmark); each file starts with the
hardware, library versions, git commit and full command-line arguments.

## 0. Read this first

* **CPU only.** All measurements ran on a 4-core cloud VM (Intel Xeon 2.1 GHz,
  15 GB RAM, **no GPU**), PyTorch 2.15 on CPU, LeRobot 0.7.0, MuJoCo 3.8 with
  software rendering. **No GPU result exists in this repository.** Absolute
  latencies here are 10–100× slower than a GPU would be and must not be read as
  GPU performance.
* **Simulation only.** LIBERO-Spatial (10 tasks). No real robot.
* **Real model.** Closed-loop and latency numbers come from the pretrained
  `HuggingFaceVLA/smolvla_libero` checkpoint executed through LeRobot's modules.
  The one synthetic benchmark (§8) is labelled as such.
* **Two kinds of time.** Closed-loop experiments use the simulator-locked
  executor: the model really runs for every inference, and the *latency charged
  to the control loop* comes from a stated latency model (`zero`, or `calibrated`
  = a linear fit to CPU measurements, optionally with an injected disturbance).
  Each table says which. Wall-clock model time is reported separately and was
  measured while two or three benchmark processes shared the four cores, so it is
  only meaningful as a *ratio* between controllers within one table.
* **Small samples.** 20–23 paired episodes per configuration. Success-rate
  confidence intervals are ±20 percentage points wide. Compute ratios are
  resolved; most success differences are **not**.

## 1. Summary of findings

MAIN_SUMMARY_PLACEHOLDER

## 2. Model validation (`krt-bench validate-model`)

File: `results/validate_model.json`. All checks pass.

| Check | Result |
|---|---|
| Pretrained weights loaded | 42 tensors sampled from the 788 in the checkpoint file compared with the loaded model: no mismatch |
| Adapter vs. LeRobot reference inference (`predict_action_chunk`, same noise) | max abs difference **0.0** |
| Determinism given the same noise | max abs difference 0.0 |
| Output validity | shape (50, 7), finite; 5.7% of values outside [-1, 1] (clipped by the action validator) |
| Reference rollout equivalence (LeRobot `make_env` + `select_action` vs. `LiberoSim` + `SmolVLABackend`, 6 steps) | observation difference 0.0, action difference 0.0 |

Step-count sweep on one real observation with the same initial noise (difference
of the 50×7 chunk from the 10-step result):

| denoising steps | 1 | 2 | 3 | 5 | 20 |
|---|---|---|---|---|---|
| RMS difference | 0.196 | 0.155 | 0.130 | 0.033 | 0.007 |
| max abs difference | 1.83 | 1.83 | 1.63 | 0.52 | 0.04 |

The large max-abs values at ≤3 steps are gripper commands flipping sign at a
different index, not small numeric noise.

The opt-in model tests (`KRT_MODEL_TESTS=1 pytest tests/test_model.py`, 4 tests)
repeat the adapter-equivalence and determinism checks and pass (136 s on this CPU).

## 3. Do the physical signals predict anything? (`krt-bench signals`)

Before using any signal in a controller, it was compared with *measured* error
proxies on states visited by the real policy. 318 samples from 9 episodes (5
successful), initial states 40 and 41, which are not used in any closed-loop
evaluation below. Files: `results/raw/signals_*.jsonl`, `results/signals.md`.

Error proxies (translational action error over the next 10 actions, in action
units, where a typical commanded step is ~0.3):

| Proxy | Meaning | Mean |
|---|---|---|
| `sens3` | 3-step vs. 10-step solve, same noise | 0.057 |
| `sens5` | 5-step vs. 10-step solve, same noise | 0.026 |
| sample disagreement | two 10-step solves with different noise | 0.295 |
| `drift_5` / `10` / `20` / `40` | plan made 5 / 10 / 20 / 40 ticks ago vs. a fresh plan | 0.334 / 0.399 / 0.479 / 0.581 |

**Finding 1 — the step count is a low-sensitivity knob for this checkpoint.**
Going from 10 to 3 denoising steps changes the action five times less than
drawing a different noise sample does. The gripper command differs somewhere in
the next 10 actions in 44% of the samples between two noise draws, but in 11%
between 3 and 10 steps.

**Finding 2 — staleness dominates.** A 5-tick-old plan already differs from a
fresh one by as much as two noise samples differ from each other, and the gap
grows with age.

Spearman rank correlation between candidate run-time signals and the proxies:

| signal | sens3 | sens5 | drift_5 | drift_10 | drift_20 |
|---|---|---|---|---|---|
| event proximity (plan predicts a gripper flip within 25 ticks) | +0.44 | +0.44 | +0.40 | +0.39 | +0.36 |
| ticks since last gripper event (negated) | +0.31 | +0.33 | +0.35 | +0.38 | +0.39 |
| planned speed | -0.06 | -0.10 | -0.09 | -0.08 | -0.09 |
| proprioceptive step | +0.02 | +0.03 | +0.00 | -0.08 | -0.11 |
| image change, scene camera | -0.06 | -0.08 | -0.16 | -0.25 | -0.29 |
| image change, wrist camera | -0.11 | -0.13 | -0.24 | -0.17 | -0.13 |
| sample disagreement (2 noise draws) | +0.73 | +0.72 | +0.62 | +0.58 | +0.54 |

By phase (mean error; n in the second column):

| phase | n | sens3 | sample disagreement | drift_10 | drift_20 |
|---|---|---|---|---|---|
| gripper event within 12 ticks | 138 | 0.069 | 0.348 | 0.501 | 0.573 |
| event 13–25 ticks away | 47 | 0.050 | 0.263 | 0.315 | 0.376 |
| no event within 25 ticks | 133 | 0.047 | 0.251 | 0.312 | 0.392 |
| within 8 ticks after an event | 103 | 0.065 | 0.337 | 0.482 | 0.575 |

**Finding 3 — only one cheap signal carries information.** Proximity to a
gripper event predicted by the current plan correlates moderately (+0.3 to +0.44)
with every error proxy; plans go stale about 1.6× faster near events. Raw image
change, proprioceptive speed and planned speed do not predict error (correlations
near zero or negative), so the physical-aware controller does not use them for
budgeting. Sample disagreement is the best predictor (+0.73) but costs a second
full denoising solve, so it is not used at run time.

**Negative result.** Mean sample disagreement in the first 60 ticks does not
separate episodes that later succeed from those that fail (0.219 vs. 0.229).
This repository has no validated run-time predictor of task failure.

**Scene surprise.** The scene-change detector (per-cell image change minus a
dilated running map of the robot's own motion) was tuned on five recorded frame
sequences (initial state 43) and then left fixed. Its end-to-end behaviour is
reported with the closed-loop experiments: false alarms in §5.1, detection of an
injected object displacement in §5.3.

## 5. Closed-loop experiments (`krt-bench closed-loop`)

Protocol common to all experiments: LIBERO-Spatial, 280-step limit, 20 Hz
control, the benchmark's own stored initial states, success as reported by the
simulator. Every controller sees the same task, initial state and model noise
seed, so comparisons are paired. Comparisons use the exact McNemar test for
success and a paired bootstrap for compute ratios. "Inference" = one model call
producing a 50-action chunk; "horizon" H = actions executed before the next
refresh is due.

Controllers:

| Name | Denoising steps | Refresh horizon H | Notes |
|---|---|---|---|
| `reference_replan` | 10 | 1 | The checkpoint's own setting (`n_action_steps=1`): one inference per tick |
| `fixed_high` | 10 | 10 | Fixed baseline |
| `fixed_reduced_steps` | 3 | 10 | Fixed, fewer steps |
| `fixed_matched` | 10 | 17 | Fixed, chosen *after* seeing `physical_aware` to match its inference count |
| `fixed_long_horizon` | 10 | 50 | Fixed, execute the whole chunk |
| `runtime_only` | 3 / 5 / 10 by deadline slack | 10 | Uses latency and buffer state only |
| `physical_aware` | 5 calm, 10 near gripper events | 25 calm, 10 near events | + scene-surprise refresh, + deadline fit |
| `physical_precise` | as above | 25 calm, **5** near events | Spends more near events |

### 5.1 E1 — nominal conditions, synchronous protocol (latency model: `zero`)

The control loop is not charged for inference time (the standard simulated
evaluation protocol). This isolates the question "how much model work can be
removed without hurting the task". Tasks 0–9 × initial states 0 and 1 (plus three
episodes on initial state 2 from an interrupted earlier batch). Files:
`results/raw/e1_zero_shard*.jsonl`, `results/e1_zero.md`.

| controller | episodes | success (95% CI) | inferences / episode | denoising steps / episode | mean action age (ticks) |
|---|---|---|---|---|---|
| `fixed_high` | 23 | 47.8% (29–67) | 19.5 | 195 | 4.5 |
| `fixed_reduced_steps` | 23 | 52.2% (33–71) | 18.8 | 56 | 4.5 |
| `fixed_matched` | 20 | 65.0% (43–82) | 10.9 | 109 | 7.8 |
| `physical_aware` | 23 | 52.2% (33–71) | 13.3 | 117 | 8.4 |
| `fixed_long_horizon` | 23 | 17.4% (7–37) | 5.4 | 54 | 22.9 |
| `physical_precise` | 10 | 40.0% (17–69) | 27.1 | 260 | 6.4 |

Paired against `fixed_high`:

| controller | pairs | success difference | wins / losses | McNemar p | inference ratio (95% CI) | denoising-step ratio | model wall-time ratio |
|---|---|---|---|---|---|---|---|
| `fixed_reduced_steps` | 23 | +4.3 pp | 1 / 0 | 1.00 | 0.96 (0.90–1.00) | 0.29 (0.27–0.30) | 0.69 (0.63–0.75) |
| `physical_aware` | 23 | +4.3 pp | 5 / 4 | 1.00 | 0.68 (0.52–0.88) | 0.60 (0.45–0.78) | 0.65 (0.50–0.84) |
| `fixed_matched` | 20 | +15.0 pp | 6 / 3 | 0.51 | 0.57 (0.44–0.76) | 0.57 (0.44–0.76) | 0.59 (0.45–0.81) |
| `fixed_long_horizon` | 23 | −30.4 pp | 2 / 9 | 0.07 | 0.28 (0.23–0.35) | 0.28 (0.23–0.35) | 0.29 (0.24–0.37) |
| `physical_precise` | 10 | −10.0 pp | 1 / 2 | 1.00 | 1.44 (0.99–1.99) | 1.39 (0.94–1.93) | 1.54 (1.02–2.14) |

`physical_aware` paired against `fixed_matched` (20 pairs): success −10 pp
(2 wins / 4 losses, p = 0.69), inference ratio 1.17 (0.92–1.45), model-time
ratio 1.04 (0.79–1.31).

What this shows:

* **Fewer denoising steps are nearly free here.** 3 instead of 10 steps removed
  71% of denoising work and 31% of model wall time; the outcome was identical in
  22 of 23 pairs (and better in the remaining one). This matches §3: the step
  count barely moves the actions. The saving in wall time is bounded by the
  vision + prefix cost that every inference pays (§4).
* **The physical-aware controller removed 32% of inferences and 35% of model
  time with no measurable change in success** (5 wins, 4 losses).
* **It did not beat a constant.** A fixed horizon of 17 removes as much work
  (43% of inferences) with a success rate that is nominally higher and
  statistically indistinguishable. On these tasks the evidence does not show
  that *when* compute is saved (calm phases vs. uniformly) matters.
* **Reuse has a limit.** Executing whole 50-action chunks open-loop cuts
  inferences by 72% and loses most of the task success (2 wins, 9 losses,
  p = 0.07; against `fixed_matched` 1 win / 10 losses, p = 0.01).
* **Spending more near gripper events did not help.** `physical_precise`
  (H = 5 near events) used 44% more inferences with no gain in 10 pairs.
* **Success is not monotonic in the refresh horizon** in this sample
  (H = 5-near-events 40%, H = 10 48%, H = 17 65%, H = 50 17%); with these sample
  sizes the first three cannot be ranked. More frequent refresh is not
  automatically better for a policy that samples a new noise vector at every
  refresh (§3: two samples disagree more than a 5-tick-old plan does).

Scene-surprise false alarms without any injected disturbance: 1 of 23
`physical_aware` episodes had a plan invalidation (one invalidation at tick 52,
in an episode that failed); 1 of 10 `physical_precise` episodes had four. Whether
those were detector errors or real unexpected scene changes (for example a
dropped object) was not determined.

Phase split of `physical_aware`: 57% of ticks classified calm, 43% critical.
Because the tasks are short pick-and-place motions, nearly half of every episode
lies within the event windows, which caps what phase-based saving can achieve.

Reproducibility: 12 episodes that ran twice in independent processes produced
identical step counts, inference counts and outcomes.
