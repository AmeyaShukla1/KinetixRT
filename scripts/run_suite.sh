#!/usr/bin/env bash
# Full experiment suite behind BENCHMARKS.md (real SmolVLA + LIBERO).
# Two shards per experiment because one benchmark process needs ~4.5 GB RAM;
# set SHARDS=1 on small machines. Runs are resumable: finished episodes are skipped.
set -euo pipefail
PY=${PY:-.venv/bin/python}
BENCH="$PY -m kinetixrt.bench.cli"
SHARDS=${SHARDS:-2}
THREADS=${THREADS:-2}
DEVICE=${DEVICE:-auto}
TAG=${TAG:-}
export MUJOCO_GL=${MUJOCO_GL:-osmesa} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}
mkdir -p results/raw results/logs

sharded() {  # sharded <name> <closed-loop args...>
  local name=$1; shift
  local pids=()
  for s in $(seq 0 $((SHARDS - 1))); do
    $BENCH closed-loop --device "$DEVICE" --threads "$THREADS" --shard "$s" --num-shards "$SHARDS" \
      --out "results/raw/${name}${TAG}_shard${s}.jsonl" "$@" > "results/logs/${name}${TAG}_shard${s}.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
}

# S. signal validation on held-out initial states (40, 41)
$BENCH signals --device "$DEVICE" --threads "$THREADS" --tasks 0 1 2 3 4 --init-states 40 --out results/raw/signals_a${TAG}.jsonl &
$BENCH signals --device "$DEVICE" --threads "$THREADS" --tasks 0 1 2 --init-states 41 --out results/raw/signals_b${TAG}.jsonl &
wait

# E1. synchronous protocol (zero latency): compute vs task success
sharded e1_zero --latency zero --init-states 0 1 2 \
  --controllers fixed_high fixed_reduced_steps fixed_long_horizon physical_aware physical_precise

# E2. edge regime: CPU-calibrated latency with an injected 2.5x contention window
# (horizons and the physical preset are adapted to ~1 s latency: scripts/e2_overrides.json)
sharded e2_edge --latency calibrated --scenario contention --contention-multiplier 2.5 --disturb-t0 1.0 --disturb-t1 7.0 \
  --init-states 0 1 --controllers fixed_high fixed_reduced_steps runtime_only physical_aware \
  --controller-overrides "$(cat scripts/e2_overrides.json)"

# E3. dynamic scene: the object of interest is displaced by 5 cm at tick 20
sharded e3_dynamic --latency zero --scenario dynamic --perturb-tick 20 --perturb-dy 0.05 \
  --init-states 0 --controllers fixed_high fixed_long_horizon physical_aware

# E4. delayed inference: +3 s on every inference requested in a 2 s window
sharded e4_delay --latency calibrated --scenario delay --delay-s 3.0 --disturb-t0 1.0 --disturb-t1 3.0 \
  --tasks 0 1 2 3 4 --init-states 0 --controllers fixed_high physical_aware

# Model-only measurements (run alone: they need an otherwise idle machine)
$BENCH latency --device "$DEVICE" --cache --out results/latency${TAG}.json
$BENCH realtime --device "$DEVICE" --duration 40 --background-load 3 --out results/realtime${TAG}.json
$BENCH scheduler --out results/scheduler_sim.json

for e in e1_zero e2_edge e3_dynamic e4_delay; do
  $BENCH report results/raw/${e}${TAG}_shard*.jsonl --out results/${e}${TAG}.md
done
$BENCH report results/raw/signals_*${TAG}.jsonl --kind signals --out results/signals${TAG}.md
