# KinetixRT - every target below has been run on the CPU development VM unless
# it is marked [GPU: not yet run on GPU hardware].
PY ?= .venv/bin/python
BENCH = $(PY) -m kinetixrt.bench.cli
DEVICE ?= auto
RESULTS ?= results
# Headless MuJoCo rendering: software (osmesa) without a GPU, EGL with one.
export MUJOCO_GL ?= osmesa
export PYOPENGL_PLATFORM ?= $(MUJOCO_GL)

.PHONY: setup setup-model test lint demo dashboard dashboard-libero validate-model benchmark \
        benchmark-latency benchmark-realtime benchmark-closed-loop benchmark-signals benchmark-scheduler \
        report gpu-check gpu-benchmark

## Lightweight CPU setup (numpy + pytest only): enough for `make test` and `make demo`.
setup:
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) -e ".[dev]"

## Full setup with PyTorch, LeRobot (SmolVLA) and LIBERO. Linux; ~8 GB disk, ~5 GB RAM per process.
setup-model:
	uv venv --python 3.12 .venv --allow-existing
	uv pip install --python $(PY) -r requirements-cuda.txt
	uv pip install --python $(PY) -e ".[dev]" --no-deps

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check kinetixrt tests

## CPU-only closed-loop demo with the live dashboard (synthetic backend, NOT a trained model).
demo dashboard:
	$(PY) -m kinetixrt.dashboard.server --demo toy --port 8050

## Live dashboard on the real SmolVLA policy in LIBERO (needs `make setup-model`).
dashboard-libero:
	$(PY) -m kinetixrt.dashboard.server --demo libero --device $(DEVICE) --port 8050

## Load the real pretrained checkpoint and check it against LeRobot's reference inference path.
validate-model:
	$(BENCH) validate-model --device $(DEVICE) --out $(RESULTS)/validate_model.json

benchmark-latency:
	$(BENCH) latency --device $(DEVICE) --cache --out $(RESULTS)/latency_$(DEVICE).json

benchmark-realtime:
	$(BENCH) realtime --device $(DEVICE) --duration 40 --background-load 3 --out $(RESULTS)/realtime_$(DEVICE).json

benchmark-signals:
	$(BENCH) signals --device $(DEVICE) --tasks 0 1 2 --init-states 40 41 --out $(RESULTS)/raw/signals_$(DEVICE).jsonl

## Small paired closed-loop comparison (10 tasks x 1 initial state x 4 controllers).
benchmark-closed-loop:
	$(BENCH) closed-loop --device $(DEVICE) --init-states 0 --out $(RESULTS)/raw/closed_loop_quick_$(DEVICE).jsonl
	$(BENCH) report $(RESULTS)/raw/closed_loop_quick_$(DEVICE).jsonl

## SYNTHETIC scheduler-policy simulation (no model).
benchmark-scheduler:
	$(BENCH) scheduler --out $(RESULTS)/scheduler_sim.json

benchmark: benchmark-latency benchmark-realtime benchmark-scheduler benchmark-closed-loop

## Full experiment suite used for BENCHMARKS.md (hours on CPU; see scripts/run_suite.sh).
suite:
	bash scripts/run_suite.sh

## [GPU: not yet run on GPU hardware] sanity check on an NVIDIA GPU.
gpu-check:
	$(PY) -c "import torch; assert torch.cuda.is_available(), 'no CUDA device'; print(torch.cuda.get_device_name(0))"
	MUJOCO_GL=egl PYOPENGL_PLATFORM=egl $(BENCH) validate-model --device cuda --out $(RESULTS)/validate_model_cuda.json

## [GPU: not yet run on GPU hardware] latency, real-time and closed-loop benchmarks on CUDA.
gpu-benchmark:
	bash scripts/gpu_benchmark.sh
