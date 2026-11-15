# PROJECT STATE

Last updated: 2026-11-15. This file is the hand-over document: what exists, what
was actually run, what was decided and why, what is broken or missing.

## 1. Status in one paragraph

KinetixRT is a working research prototype, **not production software**. The
runtime, controllers, scheduler, safety layer, SmolVLA adapter, LIBERO harness,
benchmark tool, SDK and dashboard are implemented and were executed end to end
with the real pretrained `HuggingFaceVLA/smolvla_libero` policy **on a CPU, in
simulation**. Nothing has run on a GPU or a real robot. The central hypothesis
was tested with small samples; the result is mixed and is stated in
[BENCHMARKS.md §1](BENCHMARKS.md): substantial compute reductions showed no
measurable loss of task success, but it is not established that physical signals
are the reason, and success rates are known only to about ±20 percentage points.

## 2. Components

| Component | Location | State |
|---|---|---|
| Dual-rate runtime (threaded + simulator-locked executors, tick-aligned chunks, re-anchoring, stale rejection, fallbacks) | `kinetixrt/core/runtime.py`, `action_buffer.py` | Implemented, unit-tested, run with the real model |
| Deadline-aware scheduler (FIFO / latest / deadline, bounded queue, supersede, optional degrade) + latency predictor | `kinetixrt/core/scheduler.py` | Implemented, unit-tested. Multi-stream behaviour only simulated (synthetic) |
| Latency models (zero / measured / calibrated / disturbed) | `kinetixrt/core/latency.py` | Implemented, used in all closed-loop runs |
| Physical environment analyzer (image change, scene surprise, proprio, plan events) | `kinetixrt/intelligence/analyzer.py` | Implemented; signals validated against error proxies |
| Compute controllers: fixed, runtime-only, physical-aware (+ ablation flags) | `kinetixrt/intelligence/controller.py`, `kinetixrt/presets.py` | Implemented, rule-based, every decision carries its reasons. Not learned |
| Vision feature cache (cross-observation reuse) | `kinetixrt/intelligence/temporal_cache.py` | Implemented and exact when a frame is unchanged; of little use on LIBERO (BENCHMARKS.md §7) |
| Action validation (finite, shape, bounds, rate limit, expiry) | `kinetixrt/control/safety.py` | Implemented, unit-tested. Not a certified safety system |
| SmolVLA backend (real weights via LeRobot modules, stage timings, variable step count) | `kinetixrt/backends/smolvla.py` | Implemented; bit-identical to LeRobot's reference inference on CPU |
| Synthetic backend + toy environment | `kinetixrt/backends/synthetic.py`, `simulation/toy.py` | For tests and the laptop demo only. **Not a trained model** |
| LIBERO harness with object-displacement perturbation | `kinetixrt/simulation/libero.py`, `rollout.py` | Implemented, run headless with software rendering |
| `krt-bench` (validate-model, latency, signals, closed-loop, realtime, scheduler, report, record-obs) | `kinetixrt/bench/` | Every command has been run on CPU |
| Python SDK | `kinetixrt/sdk.py` | Implemented; exercised by tests (synthetic) and by the benchmarks/dashboard (real model) |
| Live dashboard | `kinetixrt/dashboard/` | Implemented; verified in a browser in toy mode and with the real model in LIBERO |
| CUDA environment (requirements, Dockerfile, setup and benchmark scripts) | `requirements-cuda.txt`, `docker/`, `scripts/` | **Written, never executed** |

## 3. Commands that have been run successfully

On the CPU development VM (Ubuntu 24.04, Python 3.12):

| Command | Verified |
|---|---|
| `make setup`, `make test`, `make demo` | Yes, including from a fresh clone with only numpy + pytest installed (41 tests pass, 4 model tests skipped; the clean-clone check was done at 40 tests, before one regression test was added) |
| `make lint` | Yes |
| `KRT_MODEL_TESTS=1 .venv/bin/python -m pytest tests/test_model.py` | Yes (4 tests, real model) |
| `make validate-model` | Yes |
| `make dashboard-libero` | Yes |
| `make benchmark-scheduler`, `make benchmark-latency DEVICE=cpu`, `make benchmark-realtime DEVICE=cpu` | Yes |
| `krt-bench closed-loop`, `signals`, `report`, `record-obs` | Yes (these produced `results/`) |
| `make setup-model` | The same install steps were run by hand in this VM; the target itself was not re-run from a clean machine |
| `make benchmark`, `make benchmark-closed-loop`, `make suite` | Their constituent commands were run individually; the composite targets were not run end to end (hours of CPU time) |
| `make gpu-check`, `make gpu-benchmark`, `scripts/setup_cuda.sh`, `docker/Dockerfile.cuda` | **Never run** (no GPU available) |

## 4. Model selection

`HuggingFaceVLA/smolvla_libero` — SmolVLA (SigLIP vision 86 M + connector 12 M +
SmolVLM2 text model 362 M + action expert 97 M parameters), Apache-2.0, trained
on LIBERO, 2 cameras + 8-dim state, 50-action chunks of 7-dim end-effector
deltas, 10 flow-matching Euler steps by default. Chosen because: the spec prefers
SmolVLA; it is small enough for CPU development; it has an open simulation
benchmark it was trained for (LIBERO); LeRobot provides a reference
implementation to validate against; and its denoising step count is a free
inference-time parameter. `lerobot/smolvla_libero` is an alternative with
different camera key names and was not used.

## 5. Key technical decisions

1. **Python + PyTorch through LeRobot's own modules, no native core.** Inference
   costs 10²–10³× more than the runtime's fast path on the measured hardware; a
   native scheduler would not change any result yet (ARCHITECTURE.md).
2. **Tick-aligned chunks with re-anchoring** rather than a plain queue, so a late
   result skips exactly the actions whose time has passed and stale results are
   rejected explicitly.
3. **Two executors.** A threaded wall-clock executor for real deployment and a
   simulator-locked one for reproducible closed-loop evaluation in which the real
   model runs but the latency charged to the control loop is a stated model.
4. **Signals are validated before use.** Signals that did not predict any
   measured error (image change, speed) were removed from budgeting; the
   controller is driven by plan-predicted gripper events, scene surprise and
   latency/deadline state.
5. **No scheduler-side silent degradation of baselines.** Budget degradation by
   the scheduler is opt-in so that "fixed" baselines stay fixed.
6. **Exact reuse only.** KV patching across observations is not exact for this
   model; only per-camera vision features of unchanged frames are reused.
7. **Rule-based controller first**, as required; no learned component.

## 6. Test status

* `make test`: 41 passed, 4 skipped (the skipped ones need the model).
* Model tests: 4 passed.
* `ruff check`: clean.
* Reproducibility: 12 closed-loop episodes that were run twice in independent
  processes (an interrupted batch that was restarted) reproduced step counts,
  inference counts and outcomes exactly.

## 7. Measurements

See [BENCHMARKS.md](BENCHMARKS.md). Raw data: `results/raw/*.jsonl`.

## 8. Known issues and limitations

Evidence and methodology

1. **Small samples.** 20–23 paired episodes per configuration; success-rate
   intervals are about ±20 pp. No success difference between the main
   controllers is statistically resolved. The reference setting scored 10 of 10
   on one set of initial states and 2 of 10 on the next.
1a. **Absolute success is low and unexplained.** 45–65% on LIBERO-Spatial,
   including the checkpoint's reference setting, versus roughly 90% reported in
   the SmolVLA paper. The observation/action pipeline matches LeRobot's exactly;
   the cause (software rendering, versions, evaluation details) was not found.
   Check this first on a GPU machine with LeRobot's own evaluation script.
1b. **Outcomes are sensitive to numerics.** Changing only the PyTorch thread
   count changed the step count of one decision mid-episode and flipped 2 of 3
   `physical_aware` episodes; 10 of 10 `fixed_high` episodes kept their outcome.
2. **Closed-loop latency is modelled, not wall-clock.** The simulator does not
   run in real time (≈0.2 s per simulated 50 ms step on this CPU), so closed-loop
   runs use the simulator-locked executor with a stated latency model. The
   threaded executor was measured only on replayed observations
   (`krt-bench realtime`), not in closed loop.
3. **One policy, one benchmark.** SmolVLA on LIBERO-Spatial: short, fixed-layout
   pick-and-place tasks where ~45% of each episode lies near a gripper event.
4. **The dynamic-scene experiment is uninformative for task success** because
   the policy fails on displaced objects under every controller.
5. Wall-clock model times in closed-loop tables were measured while several
   benchmark processes shared four cores; only ratios are meaningful.
6. The E1 result file contains episodes from two batches (an interrupted run and
   its restart, hence a few extra episodes on initial state 2 and 10
   `physical_precise` episodes); reports pair only episodes present for both
   controllers.
7. `fixed_matched` was defined after seeing the adaptive controller's inference
   count (it is a post-hoc equal-compute baseline, and is labelled as such).

Runtime

8. **No real-time guarantees.** The threaded executor uses Python threads; the
   control thread shares the GIL and the CPU cores with PyTorch. Measured jitter
   is in BENCHMARKS.md §6. Nothing here is hard real-time or safety certified.
9. A running inference cannot be cancelled. `stop()` waits up to 30 s, then
   abandons the worker thread; its result is discarded when it finishes.
10. The latency predictor is a per-step-count running average with linear
    extrapolation; it has no model of contention onset, so the first inference
    in a slow period is always under-predicted.
11. Scene surprise assumes a static scene camera. It is suppressed for three
    ticks when the robot starts moving and is masked where the arm itself is
    moving, so a change next to the gripper can be missed. The threshold was
    tuned on five recorded sequences.
12. Phase rules depend on gripper commands in the predicted plan; a task without
    gripper events has no "critical" phase. Thresholds are specific to this
    policy and embodiment.
13. The action validator clips, rate-limits and expires actions. It does no
    collision, workspace or force checking.
14. The dashboard is a single-process demo server (stdlib HTTP, no
    authentication, binds to localhost). It is not a monitoring product.

Environment

15. Only software rendering (`MUJOCO_GL=osmesa`) was exercised; EGL is untested.
16. `hf-libero` asks an interactive question on first import;
    `ensure_libero_config()` writes its config file to avoid that.
17. PyTorch was installed from PyPI (a CUDA build running on CPU); CPU-only
    wheels were not tested.

## 9. Not implemented / unverified (do not assume these work)

* GPU execution of any kind; TensorRT, ONNX, `torch.compile` (a `--compile` flag
  exists in `krt-bench latency`, untested), quantization.
* Real-robot I/O. There is no actuator driver; real actuation is not implemented.
* ROS 2 integration, C++/Rust core, Jetson/ROCm backends.
* Real-Time Chunking (inpainting across chunk boundaries) is not wired into the
  adapter; chunk hand-over is a hard switch (optional linear blend).
* Multi-robot scheduling with real models (only the synthetic simulation).
* Any learned controller or learned uncertainty estimate.
* Benchmarks other than LIBERO-Spatial; policies other than SmolVLA.

## 10. Remaining work, in order of value

1. Run `scripts/setup_cuda.sh` and `make gpu-benchmark` on the RTX 4090. First
   question: does LeRobot's own evaluation reach the published success rate
   there? Second: does any saving survive when inference is 10–100× faster?
2. Increase sample size to ≥100 paired episodes per configuration (feasible on a
   GPU) for `reference_replan`, `physical_h2`, `physical_h2_inverted`,
   `fixed_h5` and `fixed_high`. This is the experiment that decides whether the
   phase signal matters; the CPU study could only show the direction.
3. Add LIBERO-10 / LIBERO-Object and a task family with genuinely moving objects;
   the fixed-layout LIBERO-Spatial tasks make the policy fail on displaced
   objects regardless of the runtime.
4. Wire RTC into the adapter and re-run the latency experiments.
5. Replace hand-set phase rules with a calibrated estimator trained on the
   `krt-bench signals` data, and compare against the rule set.
6. Threaded executor against a real-time simulator or hardware-in-the-loop rig.

## 11. Repository and environment notes

* Development branch: `claude/modest-tesla-ytnh7c`.
* The cloud environment needs outbound access to `huggingface.co` and its CDN
  hosts to download the checkpoint and LIBERO assets, and
  `apt-get install -y libosmesa6 libosmesa6-dev libegl1 libgles2` for headless
  MuJoCo rendering.
* One benchmark process needs ~4.5 GB RAM (model + simulator); run at most three
  in parallel on a 16 GB machine. Do not run the real model on an 8 GB laptop;
  the lightweight path (`make setup && make test && make demo`) is intended for it.
