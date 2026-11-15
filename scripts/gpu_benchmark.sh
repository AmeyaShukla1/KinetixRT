#!/usr/bin/env bash
# GPU benchmark suite (target: RTX 4090, 24 GB, Ubuntu). NOT yet executed on GPU
# hardware by the authors; results land in results/*_cuda.json and
# results/raw/*_cuda.jsonl so they never mix with the CPU measurements.
set -euo pipefail
PY=${PY:-.venv/bin/python}
BENCH="$PY -m kinetixrt.bench.cli"
export MUJOCO_GL=${MUJOCO_GL:-egl} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
mkdir -p results/raw

# 1. reference equivalence on the GPU
$BENCH validate-model --device cuda --out results/validate_model_cuda.json
# 2. latency per budget, stage breakdown, memory (200 repeats -> p99 is reportable)
$BENCH latency --device cuda --repeats 200 --warmup 10 --cache --out results/latency_cuda.json
# 3. threaded dual-rate run at 20 Hz and 100 Hz control with real contention
$BENCH realtime --device cuda --control-hz 20 --duration 60 --out results/realtime_cuda_20hz.json
$BENCH realtime --device cuda --control-hz 100 --duration 60 --out results/realtime_cuda_100hz.json
# 4. closed loop: synchronous protocol and wall-clock-measured latency on this GPU
# The CPU study found the reference setting (one inference per tick) far ahead of every chunked
# setting on task success, so the GPU run must include it and the settings between it and fixed_high.
CTRL="reference_replan fixed_h2 fixed_h5 fixed_high fixed_matched fixed_reduced_steps runtime_only physical_h2 physical_h2_inverted physical_aware"
$BENCH closed-loop --device cuda --latency zero --controllers $CTRL --init-states 0 1 2 3 4 --out results/raw/e1_zero_cuda.jsonl
$BENCH closed-loop --device cuda --latency measured --controllers $CTRL --init-states 0 1 2 3 4 --out results/raw/e2_measured_cuda.jsonl
$BENCH closed-loop --device cuda --latency measured --scenario dynamic --controllers fixed_high fixed_long_horizon physical_aware --init-states 0 1 2 --out results/raw/e3_dynamic_cuda.jsonl
$BENCH report results/raw/e1_zero_cuda.jsonl
$BENCH report results/raw/e2_measured_cuda.jsonl
$BENCH report results/raw/e3_dynamic_cuda.jsonl
