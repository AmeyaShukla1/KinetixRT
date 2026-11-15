# KinetixRT architecture

KinetixRT is an execution layer that sits between a robot's control loop and a
chunked robot policy (a VLA with a flow-matching action expert). It decides, on
every control tick, **whether to run the policy, with how much computation, and
which already-computed actions are still valid to execute**.

```
          robot / simulator observation (images, proprio)
                          |
                          v
   +--------------------- FAST PATH: Runtime.step() --------------------+
   |  PhysicalEnvironmentAnalyzer   measured scene / robot / plan signals |
   |            |                                                         |
   |            v                                                         |
   |  ComputeController  ->  Decision {request?, steps, horizon, hold,    |
   |            |                      invalidate, reasons[]}             |
   |            v                                                         |
   |  InferenceScheduler.submit()  (bounded, never blocks)                |
   |                                                                      |
   |  ActionBuffer.get(tick) -> ActionValidator -> action | fallback      |
   +----------------------------------------------------------------------+
                          ^                        |
            chunk + timing metadata                v
   +---------- SLOW PATH: inference executor ----------+     command to robot
   |  take request (FIFO | latest | deadline/EDF)      |
   |  PolicyBackend.infer(obs, ComputeBudget)          |
   |     SmolVLA: vision -> VLM prefix (KV cache)      |
   |              -> N Euler steps of action expert    |
   |  ActionBuffer.install(chunk)  (stale -> rejected) |
   +---------------------------------------------------+
```

## Module map

| Path | Responsibility |
|---|---|
| `kinetixrt/types.py` | `Observation`, `ComputeBudget`, `Decision`, `ActionChunk`, `ControlOutput` — the only types that cross module boundaries (numpy + dataclasses). |
| `kinetixrt/core/runtime.py` | `Runtime`: fast path, slow-path executors (threaded and virtual-time), telemetry, summary. |
| `kinetixrt/core/action_buffer.py` | Thread-safe, bounded, tick-aligned chunk buffer; stale-result rejection; chunk-to-chunk disagreement. |
| `kinetixrt/core/scheduler.py` | Bounded request queue with FIFO / latest-wins / deadline (EDF) policies; `LatencyPredictor`. |
| `kinetixrt/core/latency.py` | Latency models for virtual time: zero, measured, calibrated, with injected disturbances. |
| `kinetixrt/core/telemetry.py` | Bounded in-memory rings + optional JSONL; percentile helpers that refuse to report p99 below n=100. |
| `kinetixrt/intelligence/analyzer.py` | Physical signals: image change, scene surprise, proprio divergence, plan-predicted actuator events, planned speed. |
| `kinetixrt/intelligence/controller.py` | `FixedController`, `RuntimeOnlyController`, `PhysicalAwareController` (deterministic, explainable). |
| `kinetixrt/intelligence/temporal_cache.py` | Per-camera vision feature cache with explicit invalidation rules. |
| `kinetixrt/control/safety.py` | Action validation: finite, shape, bounds, rate limit, expiry. Independent of the scheduler. |
| `kinetixrt/backends/base.py` | `PolicyBackend` — the portable execution interface. |
| `kinetixrt/backends/smolvla.py` | Real SmolVLA adapter built from LeRobot's own modules. |
| `kinetixrt/backends/synthetic.py` | Analytic test backend (not a model) for CPU-only tests and the toy demo. |
| `kinetixrt/simulation/` | `LiberoSim` (real MuJoCo/LIBERO), `ToyPickPlace` (test fixture), `run_episode`. |
| `kinetixrt/sdk.py`, `presets.py` | `Runtime.from_pretrained(...)` and named controller configurations. |
| `kinetixrt/bench/` | `krt-bench`: validate-model, record-obs, latency, signals, closed-loop, realtime, scheduler, report. |
| `kinetixrt/dashboard/` | Stdlib HTTP dashboard + live demo drivers. |

No `native/` module exists; see "Why there is no C++ yet".

## The model, as actually verified

Checkpoint: `HuggingFaceVLA/smolvla_libero` (Apache-2.0), LeRobot 0.7.0. Facts
below were read from the code and confirmed by `krt-bench validate-model`.

* 605 M parameters: SigLIP vision tower 86 M, connector 12 M, SmolVLM2 text
  model 362 M (all 32 layers used in this checkpoint), action expert 97 M. The VLM
  is stored and executed in bfloat16, projections in float32.
* Input: two 256×256-trained cameras (resized with padding to 512×512 → 64 visual
  tokens each), ≤48 language tokens, one state token. Prefix length 149 tokens for
  the benchmark task.
* Inference = vision tower per camera → one VLM prefix pass that produces a
  per-layer KV cache → `num_steps` forward passes of the action expert that
  cross-attend to that cache (uniform forward-Euler integration from noise at t=1
  to actions at t=0) → un-normalisation. Output: a chunk of 50 seven-dimensional
  delta-EEF actions.
* **Within-inference KV caching is already built in** (`use_cache=True`). KinetixRT
  does not claim it.
* **The step count is a free parameter of the integrator.** Any positive integer is a
  well-defined (coarser or finer) discretisation; it is not "truncation" of a fixed
  schedule. The checkpoint default is 10. Accuracy vs. the 10-step solve is
  measured, not assumed (BENCHMARKS.md §2, §4).
* Image and language tokens do not attend to the state token, but the state token
  is part of the prefix and the suffix attends to all prefix K/V. A cached prefix
  is therefore valid only for the identical (images, instruction, state) triple.
* The checkpoint ships with `n_action_steps=1` (re-plan every control tick), which
  costs one full inference per 50 ms tick.

The adapter (`SmolVLABackend`) calls LeRobot's `embed_prefix`,
`vlm_with_expert.forward`, `denoise_step` and `euler_integrate` directly, wrapping
only `embed_image` to time and optionally cache the vision tower. Result: actions
are bit-identical to `SmolVLAPolicy.predict_action_chunk` for the same noise.

## Time, ticks and chunk alignment

* The control loop runs at period `control_dt` (LIBERO: 50 ms). Tick `k` is the
  k-th call to `Runtime.step`.
* An `ActionChunk` records `obs_tick`, `t_obs`, `t_request`, `t_start`, `t_done`,
  the budget used, the request→delivery latency and the measured model wall time.
* Action `i` of a chunk is meant for tick `anchor_tick + i`. Normally
  `anchor_tick = obs_tick`, so a chunk delivered `d` ticks late has its first `d`
  actions skipped: they describe motion the previous plan already performed.
* If the robot was standing still in fallback during part of that delay (start-up,
  starvation, abstention), no planned motion was consumed, so the anchor is moved
  forward by the number of stationary ticks. Without this, a late first chunk
  would skip the beginning of the motion.
* A chunk is rejected as stale if it is computed from an observation older than
  the active chunk's, older than an explicit invalidation point, or if all its
  actions are already in the past.

## Compute controllers

All controllers implement `decide(ControlContext) -> Decision` and are pure
functions of measured state plus small internal counters. Every `Decision`
carries `reasons`, which end up in telemetry and on the dashboard.

Refresh timing rule shared by all controllers (H = target execution horizon,
`lead` = predicted latency of the chosen budget in ticks):

    request when   plan_steps_consumed + lead >= H   or   remaining <= lead + 1

so that the new chunk lands when H actions of the current one have executed.

| Controller | Steps | Horizon | Inputs |
|---|---|---|---|
| `FixedController(n, H)` | constant | constant | none |
| `RuntimeOnlyController` | largest option whose predicted latency fits `min(H·dt, remaining plan)` | constant | latency predictor, buffer level |
| `PhysicalAwareController` | per phase, then the same deadline fit | per phase | everything above + physical signals |

`PhysicalAwareController` (the decision model `B_t = π(O_t, Q_t, U_t, P_t, D_t, R_t)`):

1. **Phase classification** (P_t, U_t, Q_t)
   * `critical` if the *active plan itself* flips a discrete actuator (gripper)
     within `event_window_pre` ticks, or such a flip was executed within
     `event_window_post` ticks; optionally also on slow planned motion, high
     chunk-to-chunk disagreement, or large proprio divergence since the plan.
   * `calm` if no critical cue fires and at least one enabled signal family gives
     positive evidence; otherwise `default`.
2. **Budget per phase**: `(steps, horizon)`; shipped LIBERO preset: calm (5, 25),
   default/critical (10, 10); `physical_precise` uses critical (10, 5).
   The setting that did best against the checkpoint's one-inference-per-tick
   reference (BENCHMARKS.md §5.5) is the preset `physical_h2`: calm (5, 10),
   near events (10, 2). Whether the phase split is what makes it work is not
   established.
3. **Scene surprise** (O_t): if the scene camera shows a localized change that
   recent activity in the same image region does not explain, the active plan is
   invalidated (`invalidate_before_tick`), a high-priority refresh is requested
   and, by default, the robot holds (abstains) until a post-change plan arrives.
4. **Deadline fit** (D_t, R_t): if the predicted latency of the chosen step count
   exceeds the slack until the plan runs out, pick the largest step option that
   fits, else the cheapest.

The thresholds are not universal constants. They were calibrated for
SmolVLA/LIBERO on held-out initial states with `krt-bench signals` and
`scripts/collect_surprise_frames.py`; the evidence is in BENCHMARKS.md §3.

### Signals and what they cost

| Signal | Source | Cost per tick |
|---|---|---|
| `img_step`, `img_since_plan` | mean abs change of a 36×36 grayscale downsample | numpy, sub-millisecond per camera |
| `scene_surprise` | per-cell change minus 3× dilated running activity map; suppressed for 3 ticks after the robot starts moving from rest | same |
| `state_step`, `state_since_plan` | proprio vector | negligible |
| `ticks_to_event`, `planned_speed` | the not-yet-executed part of the active plan | negligible |
| `ticks_since_event` | executed commands | negligible |
| chunk disagreement | overlap of the new chunk with the plan it replaces | free at install time |
| sample disagreement (two noise draws) | **not used at run time**: costs a second denoising solve; used only in the validation study |

No simulator ground truth is used by any signal.

## Scheduler

`InferenceScheduler` holds at most `capacity` requests.

* `fifo`: arrival order; rejects the newest when full (baseline).
* `latest`: per stream, a new observation supersedes queued older ones.
* `deadline`: latest-wins + earliest-deadline-first across streams, priority for
  invalidation-triggered refreshes, shedding of observations older than
  `max_obs_age_s`, and optional budget degradation to fit the slack.

With one robot the queue depth is 0–1 and the interesting behaviour is in the
controller. The scheduler matters when several robots/streams share one
accelerator; that case is covered by unit tests and a synthetic simulation only
(BENCHMARKS.md §8) — it has not been run with multiple real policies.

## Two executors, one runtime

* `mode="threaded"`: a worker thread pulls requests and runs the backend; the
  control loop calls `step()` from its own thread on the wall clock. `step()`
  takes two short mutexes and never waits for inference.
* `mode="virtual"`: simulator-locked. The model really runs (inline), but the
  runtime clock is advanced by a **latency model**, and the result is delivered at
  the first tick after `t_request + latency`. This reproduces staleness, deadline
  misses and fallbacks deterministically on hardware where the simulator cannot
  run in real time.

Latency models: `zero` (the standard synchronous benchmark protocol), `measured`
(actual wall time × scale), `calibrated` (`base + per_step·n`, constants fitted to
measurements), optionally wrapped with injected contention or delay windows.
Every result file records which model was used.

Both executors share the scheduler, buffer, controller, validator and telemetry
code, so virtual-time closed-loop results exercise the same logic as the threaded
runtime. The threaded path is additionally tested for non-blocking behaviour.

## Concurrency and ownership

| Object | Writer | Reader | Protection |
|---|---|---|---|
| `ActionBuffer` chunks | slow path only (`install`) | fast path only (`get`) | one mutex; chunks immutable after install |
| scheduler queue | fast path (`submit`) | slow path (`take`) | condition variable; bounded |
| in-flight marker, executed-tick ring, installed-tick list | both | both | runtime mutex |
| analyzer, validator, controller state | fast path only | fast path only | single-threaded by construction (plan-installed notifications are queued and drained on the fast path) |
| telemetry rings | both | dashboard thread | mutex; bounded deques |

Nothing is unbounded: queue (`capacity`), buffer (one active + one previous
chunk), telemetry (ring size), analyzer history (256 ticks).

Timing claims are best-effort. Python threads, a general-purpose OS and a GIL
are in the loop; nothing here is hard real-time.

## Safety layer

`ActionValidator` runs on every action leaving the buffer: shape and finite
checks, bounds clipping, optional per-tick rate limit on selected dimensions,
and expiry of plans older than `max_chunk_age_ticks`. A rejected or missing
action is replaced by the configured fallback (LIBERO: zero end-effector delta,
gripper command held). Fallback causes are counted separately (start-up,
starved, hold, expired, invalid). This is a simulation-grade guard, not a
certified safety function, and it must not be the only protection on a real
robot. Real actuation is not implemented in this repository.

## Cross-observation reuse: what was built and why

KV-cache delta patching across observations is not exact for this model (the
state token changes every tick and every prefix K/V depends on all image tokens).
What *is* exact is the vision tower output of a camera whose frame did not change.
`VisionFeatureCache` reuses it when the downsampled frame changed by less than
`tau_mean` on average and `tau_peak` in any cell, the instruction is the same and
the entry is younger than `max_age_ticks`. Measured hit rates and savings are in
BENCHMARKS.md §7; on LIBERO both cameras change almost every refresh, so this
optimisation is of limited use there.

The reuse that pays off for this model is **plan reuse**: SmolVLA predicts 50
actions per inference; how many to execute before refreshing is exactly the
`horizon` the controller decides.

## Why there is no C++ yet

The roadmap allows native code "when justified by evidence". Measured on the
development CPU, the Python fast path costs well under 1 ms per tick with
64×64 images and about 1.5 ms (median) with two 360×360 cameras, with a worst
per-episode p95 below 20 ms while other benchmark processes competed for the
same four cores (dominated by the numpy downsample; BENCHMARKS.md §6), against
inference times of hundreds of milliseconds to seconds. A native scheduler would not change any result in this
repository. The first native candidates, if profiling on a GPU shows the fast
path matters at 100 Hz+, are the analyzer's downsample and the buffer.

## Portability

Everything above `PolicyBackend` is numpy-only and device-agnostic.
A new accelerator or runtime is a new backend implementing
`infer(obs, budget) -> InferenceResult` and `info`. `SmolVLABackend` resolves
`device="auto"` to CUDA, MPS or CPU. **Only CPU has been executed.** The CUDA
environment (`docker/Dockerfile.cuda`, `scripts/setup_cuda.sh`,
`scripts/gpu_benchmark.sh`) is prepared but unverified; ROCm, Jetson and
TensorRT backends do not exist.
