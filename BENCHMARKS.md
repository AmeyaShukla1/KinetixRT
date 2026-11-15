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
