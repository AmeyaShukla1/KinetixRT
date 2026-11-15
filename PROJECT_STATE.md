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
[BENCHMARKS.md §1](BENCHMARKS.md).

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
| `make setup`, `make test`, `make demo` | Yes, including from a fresh clone with only numpy + pytest installed (40 tests pass, 4 model tests skipped) |
| `make lint` | Yes |
| `KRT_MODEL_TESTS=1 .venv/bin/python -m pytest tests/test_model.py` | Yes (4 tests, real model) |
| `make validate-model` | Yes |
| `make dashboard-libero` | Yes |
| `make benchmark-scheduler`, `make benchmark-latency`, `make benchmark-realtime` | Yes |
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

* `make test`: 40 passed, 4 skipped (the skipped ones need the model).
* Model tests: 4 passed.
* `ruff check`: clean.
* Reproducibility: 12 closed-loop episodes that were run twice in independent
  processes (an interrupted batch that was restarted) reproduced step counts,
  inference counts and outcomes exactly.

## 7. Measurements

See [BENCHMARKS.md](BENCHMARKS.md). Raw data: `results/raw/*.jsonl`.

## 8. Known issues and limitations

KNOWN_ISSUES_PLACEHOLDER

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

1. Run `scripts/setup_cuda.sh` and `make gpu-benchmark` on the RTX 4090: does any
   saving survive when inference is 10–100× faster and the latency constraint
   disappears?
2. Increase sample size to ≥100 paired episodes per configuration (minutes on a
   GPU) to resolve the success differences that this CPU study could not.
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
