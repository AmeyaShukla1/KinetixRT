# KinetixRT

An adaptive inference runtime for chunked robot policies (VLAs with a
flow-matching action expert). KinetixRT sits between the control loop and the
policy and decides on every control tick **whether to run the model, with how
many denoising steps, and which already-computed actions are still valid** —
using measured scene, robot, plan and latency signals.

It runs a real pretrained policy end to end:
[`HuggingFaceVLA/smolvla_libero`](https://huggingface.co/HuggingFaceVLA/smolvla_libero)
(SmolVLA, 605 M parameters, Apache-2.0) through Hugging Face LeRobot's own model
code, in closed loop in the LIBERO MuJoCo benchmark.

> **What has and has not been run.** Everything reported in this repository was
> executed on a 4-core CPU cloud VM without a GPU. The NVIDIA CUDA environment
> (RTX 4090 target) is prepared but **has never been run**; no GPU performance is
> claimed anywhere. No real robot was involved. See [PROJECT_STATE.md](PROJECT_STATE.md).

## Documents

| File | Content |
|---|---|
| [BENCHMARKS.md](BENCHMARKS.md) | All measurements: model validation, latency, signal validation, closed-loop comparisons, negative results |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Runtime design, timing semantics, controllers, scheduler, concurrency |
| [RESEARCH.md](RESEARCH.md) | Existing systems, competitor matrix, the gap targeted |
| [STARTUP_THESIS.md](STARTUP_THESIS.md) | Commercial analysis with assumptions separated from measurements |
| [PROJECT_STATE.md](PROJECT_STATE.md) | What works, what does not, known issues, remaining work |

## Quick start

### 1. Lightweight CPU path (laptop-friendly; no PyTorch, no model weights)

Runs the full runtime, scheduler, controllers, tests and dashboard against a toy
environment and a synthetic backend. The synthetic backend is an analytic
controller, **not a trained model**; it exists to exercise the runtime.

```bash
make setup
```

```bash
make test
```

```bash
make demo
```

`make demo` serves the live dashboard at http://127.0.0.1:8050.

### 2. Real model (Linux; ~8 GB disk, ~5 GB RAM per process; CPU works, slowly)

```bash
make setup-model
```

```bash
make validate-model
```

```bash
make benchmark
```

```bash
make dashboard-libero
```

Headless MuJoCo needs a GL library: `apt-get install -y libosmesa6 libegl1` (the
Makefile defaults to software rendering, `MUJOCO_GL=osmesa`).
Model tests: `KRT_MODEL_TESTS=1 .venv/bin/python -m pytest tests/test_model.py`.

### 3. NVIDIA GPU (prepared, not yet executed)

```bash
bash scripts/setup_cuda.sh
```

```bash
make gpu-check
```

```bash
make gpu-benchmark
```

or `docker build -f docker/Dockerfile.cuda -t kinetixrt:cuda .`. GPU results are
written to separate `*_cuda` files so they can never be confused with the CPU
measurements.

## SDK

```python
from kinetixrt import Runtime

runtime = Runtime.from_pretrained(
    "HuggingFaceVLA/smolvla_libero",
    adaptive_compute=True,   # or False, a preset name, a spec dict, or a controller
    backend="auto",          # cuda | mps | cpu
)
runtime.reset(seed=0, task="put the bowl on the plate")
runtime.start()              # starts the inference worker thread

while robot.ok():
    out = runtime.step(images={"image": rgb, "image2": wrist_rgb}, robot_state=state)
    robot.send(out.action)   # never blocks on inference
    # out.source: chunk | fallback_startup | fallback_starved | fallback_hold | ...
    # out.decision.reasons: why the runtime did (not) request inference this tick

runtime.stop()
```

`Runtime.from_pretrained("synthetic/toy-pickplace-controller", ...)` gives the
same object backed by the synthetic test backend.

## krt-bench

```bash
.venv/bin/python -m kinetixrt.bench.cli --help
```

| Command | What it measures | Needs |
|---|---|---|
| `validate-model` | Real weights loaded; adapter bit-identical to LeRobot's reference inference; reference rollout equivalence | model (+ LIBERO, or `--source fixture`) |
| `latency` | Per-budget latency percentiles, stage breakdown, memory, optional real CPU contention | model |
| `signals` | Physical signals vs. measured error proxies | model + LIBERO |
| `closed-loop` | Paired closed-loop comparison of compute controllers, scenarios nominal / dynamic / contention / delay | model + LIBERO |
| `realtime` | Threaded dual-rate run on replayed real observations: fast-path blocking, jitter, deadline misses | model (or `--synthetic`) |
| `scheduler` | **Synthetic** multi-robot overload simulation of the scheduling policies | nothing |
| `report` | Aggregates JSONL results into tables with confidence intervals | nothing |
| `record-obs` | Records the real-observation fixture used by model-only benchmarks | model + LIBERO |

The full suite behind BENCHMARKS.md is `scripts/run_suite.sh` (hours on a CPU).

## Repository layout

```
kinetixrt/
  types.py                 shared dataclasses
  core/                    runtime, action buffer, scheduler, latency models, telemetry, clock
  intelligence/            environment analyzer, compute controllers, vision feature cache
  control/                 action validation
  backends/                PolicyBackend interface, SmolVLA (real), synthetic (tests)
  simulation/              LIBERO harness, toy environment, episode runner
  bench/                   krt-bench commands, statistics, real-observation fixture
  dashboard/               live dashboard server, page and demo drivers
  sdk.py, presets.py       Runtime.from_pretrained, named controller presets
tests/                     CPU-only suite + opt-in real-model tests
scripts/                   suite runner, GPU setup and benchmark, calibration data collection
docker/                    CUDA image definition (unverified)
results/                   measured results (raw JSONL + summaries)
```

## Licence and third-party components

KinetixRT code: Apache-2.0 (as declared in `pyproject.toml`). It depends on
LeRobot (Apache-2.0), PyTorch (BSD-3), the SmolVLA checkpoint (Apache-2.0),
LIBERO (MIT) and MuJoCo (Apache-2.0). Check the licences of any other checkpoint
or dataset before using it with this runtime.
