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

The question: can a runtime allocate inference compute adaptively — fewer
denoising steps, less frequent refresh, reuse — from physical and runtime
signals, and save compute without hurting what the robot does?

**Short answer: partly shown, mostly unresolved, and not yet attributable to
"physical awareness".**

What the measurements support:

1. **The runtime is correct and does what it claims.** The adapter and the
   runtime's chunk execution are bit-identical to LeRobot's reference inference
   (§2). The control loop never waited for the model: its longest step was
   9.5 ms while inferences took 1.3–4.1 s (§6). Stale results were rejected and
   fallbacks engaged in every overload episode without an invalid command (§5.2).
2. **Large compute reductions produced no measurable loss of task success.**
   * 3 instead of 10 denoising steps: −71% denoising work, −31% model time, same
     outcome in 22 of 23 paired episodes (§5.1).
   * Phase-dependent refresh (every 2 ticks near predicted gripper events, every
     10 otherwise) matched the checkpoint's reference setting — 13 vs. 12
     successes in 20 paired episodes — with 26% of its inferences (95% CI
     19–36%) (§5.5).
3. **Under overload, latency-aware step reduction halves the time the robot has
   no valid action** (88 → 41 control ticks per episode in simulation, §5.2;
   43% → 6% of ticks in the threaded run, §6).

What they do not support:

4. **"No measurable loss" is a weak statement at this sample size.** With 20–23
   paired episodes, a success rate is known to about ±20 percentage points. The
   reference setting succeeded in 10 of 10 episodes on one set of initial states
   and 2 of 10 on the next (§5.5). No success difference between the main
   controllers is statistically significant.
5. **It is not shown that physical signals are what makes saving safe.** In E1 a
   constant refresh horizon saved as much as the physical-aware controller with
   equal success (§5.1). In E6 the phase-aware rule did nominally better than
   its phase-swapped control (5 wins / 1 loss) and than the nearest uniform rate
   (5 / 2), which is the direction the hypothesis predicts, but neither is
   significant (§5.5). Under a binding latency constraint the physical signals
   added nothing over latency-only control (§5.2).
6. **Negative results.** Scene-triggered refresh detected every injected object
   displacement and changed no outcome, because the policy does not recover from
   displaced objects (§5.3). Cross-observation vision reuse never hit (§7).
   Refreshing more often near gripper events did not help in E1 (§5.1). No
   run-time signal predicted task failure (§3). The Python threaded executor
   slows inference by 35–60% (§6).
7. **Nothing here is a GPU or real-robot result.** On this CPU the policy cannot
   run in real time at its reference setting by a factor of about 30 (§4); the
   closed-loop success numbers come from simulation with modelled latency.

A caveat on absolute numbers: success rates in this repository are 45–65% on
LIBERO-Spatial, including for the reference setting (60%, CI 39–78). The SmolVLA
paper reports roughly 90% for this benchmark. The observation and action
pipeline was verified against LeRobot's own (§2), and the cause of the gap was
not determined (candidates: software rendering, simulator or library versions,
evaluation details). All conclusions here are relative comparisons inside one
setup.

## 2. Model validation (`krt-bench validate-model`)

File: `results/validate_model.json`. All checks pass.

| Check | Result |
|---|---|
| Pretrained weights loaded | 42 tensors sampled from the 788 in the checkpoint file compared with the loaded model: no mismatch |
| Adapter vs. LeRobot reference inference (`predict_action_chunk`, same noise) | max abs difference **0.0** |
| Determinism given the same noise | max abs difference 0.0 |
| Output validity | shape (50, 7), finite; 5.4% of values outside [-1, 1] (clipped by the action validator) |
| Reference rollout equivalence (LeRobot `make_env` + `select_action` vs. `LiberoSim` + `SmolVLABackend`, 8 steps) | observation difference 0.0, action difference 0.0 |
| Chunked execution equivalence (LeRobot's own action queue with `n_action_steps=10` vs. the KinetixRT runtime with `FixedController(10, 10)` at zero latency, same noise, 25 ticks, 3 refreshes) | action difference **0.0**, state difference 0.0, same number of inferences |

Step-count sweep on one real observation with the same initial noise (difference
of the 50×7 chunk from the 10-step result):

| denoising steps | 1 | 2 | 3 | 5 | 20 |
|---|---|---|---|---|---|
| RMS difference | 0.192 | 0.148 | 0.120 | 0.022 | 0.007 |
| max abs difference | 1.82 | 1.84 | 1.65 | 0.28 | 0.03 |

The large max-abs values at ≤3 steps are gripper commands flipping sign at a
different index, not small numeric noise.

The last row of the first table matters for §5: executing ten actions of a chunk
through this runtime is bit-identical to LeRobot's own chunked execution, so the
success differences between refresh horizons reported below are a property of
the policy, not of the runtime's chunk handling.

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

## 4. Inference latency on CPU (`krt-bench latency`)

Idle machine, 4 PyTorch threads, real LIBERO observations from the recorded
fixture, 20 timed repetitions per budget after 2 warm-ups. File:
`results/latency_cpu.json`. The model has 604.9 M parameters (bfloat16 VLM,
float32 action expert); the process used 3.9 GB of RAM.

| denoising steps | p50 (ms) | p95 (ms) | vision, 2 cameras (ms) | prefix / KV build (ms) | denoising (ms) |
|---|---|---|---|---|---|
| 1 | 868 | 1046 | 556 | 203 | 102 |
| 3 | 980 | 1076 | 516 | 184 | 259 |
| 5 | 1100 | 1248 | 487 | 181 | 429 |
| 10 (default) | 1508 | 1828 | 497 | 179 | 813 |
| 20 | 2367 | 2583 | 495 | 179 | 1678 |

Linear fit: latency ≈ 0.74 s + 0.080 s × steps. Pre- and post-processing take
under 2 ms. With only 20 repetitions no p99 is reported.

* About 0.68 s of every inference (45% at 10 steps) is the vision tower and the
  prefix pass, which do not depend on the step count. Going from 10 to 3 steps
  removes 70% of the denoising work but 35% of the latency.
* When both cameras' vision features are reused (the vision-cache hit path),
  10 steps take 1055 ms (−30%) and 3 steps 442 ms (−55%). §7 shows this path is
  almost never taken on LIBERO.
* At 20 Hz one inference spans 20 (3 steps) to 31 (10 steps) control ticks on
  this CPU. A 50-action chunk can only be refreshed without gaps if latency stays
  under about 25 ticks, so the 10-step budget is not sustainable here and the
  3-step budget barely is. **On this hardware the policy cannot be run in real
  time at its reference setting (one inference per tick) by a factor of ~30.**

**Calibration used in the closed-loop latency experiments.** E2 and E4 (§5.2,
§5.4) charge latency = 0.61 s + 0.071 s × steps. Those constants came from an
earlier run of this benchmark on the same VM; the re-measurement reported above
(after a VM restart) is 17–19% slower. The closed-loop latency experiments
therefore model a CPU slightly faster than the one in this table; the regime
(inference lasting 17–27 control ticks) is the same.

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
| `fixed_h2`, `fixed_h3`, `fixed_h5` | 10 | 2, 3, 5 | Uniform rates between the reference and `fixed_high` (§5.5) |
| `physical_h1`, `physical_h2` | 5 calm, 10 near events | 5 calm / 1 near events; 10 calm / 2 near events | Phase-dependent refresh near the reference setting (§5.5) |
| `physical_h1_inverted`, `physical_h2_inverted` | — | phases swapped | Controls: same rule, dense refresh in the *calm* phase |

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

### 5.2 E2 — slow inference with a contention window (latency model: `calibrated` + disturbance)

Here the control loop *is* charged for inference. The latency model is the linear
CPU fit from §4 (0.82 s for 3 steps, 1.32 s for 10 steps, i.e. 17 and 27 control
ticks), multiplied by 2.5 for every inference started between t = 1 s and
t = 7 s (2.06 s and 3.30 s). This is a deliberately hostile regime: a 50-action
chunk lasts 2.5 s, so during the window not even the cheapest budget can keep
the buffer covered. The robot holds still (zero motion, gripper held) whenever
it has no valid action.

Because the default presets assume fast inference, the controllers were given
horizons suited to ~1 s latency (`scripts/e2_overrides.json`): fixed controllers
and `runtime_only` refresh every 25 actions; `physical_aware` uses 5 steps /
H = 40 when calm and 3 steps / H = 10 near gripper events. Tasks 0–9 × initial
states 0, 1. Files: `results/raw/e2_edge_shard*.jsonl`, `results/e2_edge.md`.

| controller | episodes | success (95% CI) | ticks without a valid action / episode | start-up wait (ticks) | share of ticks driven by a plan | mean action age (ticks) | steps used |
|---|---|---|---|---|---|---|---|
| `fixed_high` | 20 | 30.0% (15–52) | 88.0 | 27 | 55% | 51 | 10 |
| `fixed_reduced_steps` | 20 | 35.0% (18–57) | 40.8 | 17 | 76% | 41 | 3 |
| `runtime_only` | 20 | 30.0% (15–52) | 40.9 | 27 | 72% | 43 | 3 (89%), 10 (11%) |
| `physical_aware` | 20 | 20.0% (8–42) | 45.9 | 20 | 73% | 41 | 3 (79%), 5 (21%) |

Paired against `fixed_high`: `fixed_reduced_steps` 3 wins / 2 losses,
`runtime_only` 3 / 3, `physical_aware` 2 / 4 (all p ≥ 0.69).

What this shows:

* **The runtime behaves as designed under overload.** Every `fixed_high` episode
  had a stale result rejected and spent 88 ticks (4.4 s) without a valid action.
  Controllers that lower the step count when the deadline cannot be met halved
  that (41–46 ticks). No episode produced a non-finite or out-of-range command;
  the fallback and re-anchoring paths were exercised in every episode.
* **That did not turn into a measurable success difference.** All four
  controllers succeed in 20–35% of episodes (versus ~50% in E1), and the
  differences are within noise. When inference takes 17–66 control ticks, every
  controller executes actions that are on average 40–50 ticks old, and that is
  what hurts.
* **Physical signals added nothing here.** `runtime_only` and a fixed 3-step
  budget do as well as `physical_aware`, whose nominal success is the lowest.
  Under a binding latency constraint the deadline logic picks the cheapest
  budget almost always, and the phase rules have no room to act.

### 5.3 E3 — dynamic scene: the object to be grasped is displaced (latency model: `zero`)

At tick 20, while the arm is approaching, the object of interest is moved 5 cm
sideways. The displacement is applied to the simulator state; the runtime sees
it only through the camera images. Initial state 0. The run was **stopped after
tasks 0–4** because the outcome made further episodes uninformative (see below).
Files: `results/raw/e3_dynamic_shard*.jsonl`, `results/e3_dynamic.md`.

| controller | episodes | successes | same episodes without displacement (E1) | episodes in which the plan was invalidated | ticks from displacement to invalidation |
|---|---|---|---|---|---|
| `fixed_high` | 5 (+1 on task 5) | 0 (+1) | 3 of 5 | n/a | n/a |
| `physical_aware` | 5 | 0 | 4 of 5 | 5 of 5 | 2 in every episode |
| `physical_no_scene` (ablation) | 4 | 0 | n/a | 0 of 4 | n/a |

What this shows:

* **The scene-surprise detector works end to end on this perturbation.** It
  fired in 5 of 5 episodes, on the first frame in which the displacement was
  visible (2 ticks after it was applied), exactly once per episode, with a
  surprise value of 0.11–0.15 against a threshold of 0.04. The runtime then
  invalidated the plan, held for one tick and requested a fresh chunk.
* **It made no difference to the task.** Every paired episode failed with every
  controller, including the baseline that already replans every 10 ticks.
  The same episodes mostly succeed without the displacement. The limiting factor
  is the policy: this checkpoint was trained on LIBERO-Spatial's fixed layouts
  and does not recover from a 5 cm object displacement, so replanning sooner
  cannot help. **The value of scene-triggered refresh for task success is
  therefore untested**, not demonstrated; testing it needs a policy that is
  robust to scene changes.

### 5.4 E4 — delayed inference (latency model: `calibrated` + delay)

+3 s added to every inference started between t = 1 s and t = 3 s; same
overrides as E2; tasks 0–2, initial state 0. **Three pairs only — an
illustration of behaviour, not a statistical result.** Files:
`results/raw/e4_delay_shard0.jsonl`, `results/e4_delay.md`.

| controller | episodes | successes | ticks without a valid action / episode | stale results rejected / episode | max latency (s) |
|---|---|---|---|---|---|
| `fixed_high` | 3 | 1 | 69.3 | 1.0 | 4.32 |
| `physical_aware` | 3 | 1 | 46.0 | 0.0 | 3.96 |

The delayed result arrived after its whole chunk had expired and was rejected in
every `fixed_high` episode; the robot held still until the next result. Outcomes
were identical in all three pairs.

### 5.5 E6 — refresh rate against the checkpoint's reference setting (latency model: `zero`)

The checkpoint ships with `n_action_steps = 1`: one inference per control tick.
The fixed baseline of E1 (refresh every 10 actions) is therefore already a
5-fold reduction of the reference compute. This experiment places the reference,
uniform refresh rates between the two, and phase-dependent refresh rates on the
same episodes. Tasks 0–9; initial state 0 for all controllers and initial state 1
for five of them; single-threaded so that all runs share one numerical
configuration. Files: `results/raw/e0_reference_*.jsonl`, `e6_frontier_*.jsonl`,
`e8_*.jsonl`, `results/e6_frontier.md`.

| controller | refresh rule | initial state 0 | initial state 1 | total (95% CI) | inferences / episode | inferences vs. reference (95% CI) |
|---|---|---|---|---|---|---|
| `reference_replan` | every tick | 10 / 10 | 2 / 10 | 12 / 20 = 60% (39–78) | 171 | 1.00 |
| `fixed_h2` | every 2 | 7 / 10 | – | – | 79 | 0.80 (0.53–1.13) |
| `fixed_h3` | every 3 | 6 / 10 | – | – | 64 | 0.65 (0.47–0.86) |
| `fixed_h5` | every 5 | 5 / 10 | 5 / 10 | 10 / 20 = 50% (30–70) | 39 | 0.23 (0.17–0.31) |
| `fixed_high` | every 10 | 7 / 10 | 2 / 10 | 9 / 20 = 45% (26–66) | 20 | 0.12 (0.09–0.15) |
| `physical_h1` | every tick near gripper events, every 5 when calm | 7 / 10 | – | – | 88 | 0.89 (0.52–1.39) |
| `physical_h1_inverted` | the same with the phases swapped | 8 / 10 | – | – | 92 | 0.93 (0.72–1.27) |
| `physical_h2` | every 2 near events, every 10 (5 steps) when calm | 9 / 10 | 4 / 10 | 13 / 20 = 65% (43–82) | 45 | 0.26 (0.19–0.36) |
| `physical_h2_inverted` | the same with the phases swapped | 6 / 10 | 3 / 10 | 9 / 20 = 45% (26–66) | 59 | 0.34 (0.26–0.45) |

(Inference counts include failed episodes, which run to the 280-step limit; the
ratios are paired per episode. The reference needs 98 inferences for a
successful episode and 280 for a failed one.)

Paired results for `physical_h2` (20 pairs each):

| compared with | wins / losses | McNemar p | inference ratio (95% CI) |
|---|---|---|---|
| `reference_replan` | 3 / 2 | 1.00 | 0.26 (0.19–0.36) |
| `physical_h2_inverted` (phase-swapped control) | 5 / 1 | 0.22 | 0.77 (0.54–1.08) |
| `fixed_h5` (closest uniform compute) | 5 / 2 | 0.45 | 1.16 (0.86–1.50) |
| `fixed_high` | 5 / 1 | 0.22 | 2.25 (1.68–2.84) |

What this shows:

* **The most favourable result for the hypothesis in this repository:**
  refreshing every 2 ticks near predicted gripper events and every 10 ticks
  otherwise matched the reference setting's success (13 vs. 12 of 20; 3 wins,
  2 losses) with **26% of its inferences** (95% CI 19–36%).
* **It is not established that the phase signal is the reason.** The same rule
  with the phases swapped did worse (9 of 20; 5 wins / 1 loss for the correct
  assignment) while using *more* inferences, and the closest uniform refresh
  rate did worse too (10 of 20; 5 / 2). Both differences point in the direction
  the hypothesis predicts; neither is statistically significant (p = 0.22 and
  0.45).
* **The reference setting is not reliably better than cheaper settings.** After
  the first ten episodes it led every other controller (10 of 10, never beaten
  in any pair). On the next ten (initial state 1) it succeeded twice. This is
  the clearest illustration in the data of how misleading ten episodes are.
* **Success is not a smooth function of the refresh rate.** On initial state 0:
  every tick 10, every 2 ticks 7, every 3 ticks 6, every 5 ticks 5, every 10
  ticks 7 of 10.
* **Initial conditions dominate.** Initial state 1 is much harder than initial
  state 0 for almost every controller (reference 2 vs. 10, `fixed_high` 2 vs. 7,
  `physical_h2` 4 vs. 9). Variation between initial states is larger than any
  difference between controllers measured here.

SEC56_PLACEHOLDER

## 6. Runtime overhead and real-time behaviour (`krt-bench realtime`)

**Fast-path cost in closed loop.** Per-tick cost of `Runtime.step` (analysis of
two 360×360 camera frames, controller decision, buffer lookup, validation,
telemetry), excluding model time, over all E1 episodes: median 1.5 ms per tick;
the worst per-episode p95 was 18.7 ms while three benchmark processes competed
for the four cores.

**Threaded executor with the real model.** A 20 Hz control loop in the main
thread and the model in a worker thread, for 40 s, on replayed real observations
(no simulator, so no task outcome). Files: `results/realtime_cpu_idle.json`
(idle machine), `results/realtime_cpu.json` (three CPU-burning processes started
at t = 10 s and stopped at t = 20 s).

| condition | controller | fast-path step p50 / p99 / max (ms) | control-period error p50 / p99 / max (ms) | inference latency p50 / max (s) | steps used | ticks driven by a plan |
|---|---|---|---|---|---|---|
| idle | `fixed_high` | 1.7 / 2.8 / 9.5 | 0.12 / 3.1 / 4.0 | 2.45 / 2.62 | 10 | 400 of 800 |
| idle | `runtime_only` | 1.6 / 3.9 / 6.1 | 0.12 / 2.7 / 4.7 | 1.33 / 2.44 | 3 (27 of 28) | 700 of 800 |
| CPU load | `fixed_high` | 1.7 / 4.8 / 8.0 | 0.14 / 5.7 / 104 | 2.64 / 4.14 | 10 | 318 of 799 |
| CPU load | `runtime_only` | 1.7 / 3.8 / 26.1 | 0.15 / 6.0 / 102 | 1.41 / 2.71 | 3 (23 of 24) | 579 of 799 |

* **The fast path does not block on inference.** While single inferences took
  1.3–4.1 s, the longest control step took 9.5 ms on the idle machine and 26 ms
  under CPU load.
* **Timing is not hard real-time.** Under CPU load the control loop itself was
  late by about 100 ms once in each run (operating-system scheduling on an
  oversubscribed 4-core VM). Nothing in this repository prevents that.
* **The Python threaded executor slows inference.** With the control loop
  running in the same process, a 10-step inference took 2.45 s instead of the
  1.51 s measured stand-alone (3 steps: 1.33 s vs. 0.98 s), i.e. +35% to +60%.
  The control thread and PyTorch's worker threads share the interpreter lock and
  the cores. This is the first measured argument in this project for moving the
  fast path out of the model's process (or into native code); it was not acted on.
* With a 2.45 s latency a fixed 10-step budget leaves the robot without a valid
  action for 43% of the time; the latency-aware controller switches to 3 steps
  after the first inference and brings that to 6% (21% under the load window).
  Both numbers describe buffer coverage only, not task success.

## 7. Cross-observation vision-feature reuse

`VisionFeatureCache` reuses a camera's vision-tower output when the downsampled
frame changed by less than a threshold since the cached one. Measured in closed
loop with the cache enabled (`--vision-cache`; 6 episodes, tasks 0–2,
`fixed_high` and `physical_aware`; `results/raw/e5_vision_cache.jsonl`):

| | value |
|---|---|
| cache lookups | 114 |
| hits | **0** |
| rejected because the frame changed | 108 (the other 6 were the first lookup of a camera in an episode) |

Both cameras change between any two refreshes while the arm moves (one is
mounted on the wrist), so nothing is ever reused. The hit path is real and saves
0.45 s per inference when it is taken (§4), but **on this benchmark it
contributes nothing**. It could matter for a stationary robot waiting for an
event, or for a static third-person camera if hits were allowed per camera at a
looser threshold; neither was evaluated.

## 8. Scheduler policies under multi-robot overload — SYNTHETIC

**No model is executed in this benchmark.** `krt-bench scheduler` simulates N
robots sharing one inference executor with an assumed service time
(0.06 s + 0.007 s × steps), each requesting a refresh every 0.25 s for 1 s
chunks, to exercise the queueing policies.
It says nothing about real model performance. File: `results/scheduler_sim.json`.

| streams | policy | result age p50 / p95 (s) | time without a valid action | mean steps |
|---|---|---|---|---|
| 1 | any | 0.13 / 0.16 | 0.1% | 10 |
| 2 | fifo | 0.31 / 0.39 | 0.1% | 10 |
| 2 | deadline | 0.20 / 0.35 | 0.1% | 10 |
| 4 | fifo | 0.62 / 0.69 | 45.4% | 10 |
| 4 | latest-only | 0.34 / 0.39 | 34.3% | 10 |
| 4 | deadline | 0.23 / 0.34 | 3.2% | 10 |
| 4 | deadline + step degradation | 0.23 / 0.36 | 0.3% | 8.8 |

In the simulation, serving the stream closest to running out of actions and
dropping superseded observations keeps four streams alive on an executor that
FIFO saturates. This has not been run with several real policies.
