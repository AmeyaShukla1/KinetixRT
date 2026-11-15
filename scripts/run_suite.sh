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
# (the published E1 file additionally holds a few init-state-2 and physical_precise episodes from
#  an earlier, interrupted batch; `krt-bench report` pairs only episodes present for both controllers)
sharded e1_zero --latency zero --init-states 0 1 \
  --controllers fixed_high fixed_reduced_steps fixed_long_horizon fixed_matched physical_aware

# E2. edge regime: CPU-calibrated latency with an injected 2.5x contention window
# (horizons and the physical preset are adapted to ~1 s latency: scripts/e2_overrides.json)
sharded e2_edge --latency calibrated --scenario contention --contention-multiplier 2.5 --disturb-t0 1.0 --disturb-t1 7.0 \
  --init-states 0 1 --controllers fixed_high fixed_reduced_steps runtime_only physical_aware \
  --controller-overrides "$(cat scripts/e2_overrides.json)"

# E3. dynamic scene: the object of interest is displaced by 5 cm at tick 20
sharded e3_dynamic --latency zero --scenario dynamic --perturb-tick 20 --perturb-dy 0.05 \
  --init-states 0 --controllers fixed_high physical_aware physical_no_scene

# E4. delayed inference: +3 s on every inference requested in a 2 s window
sharded e4_delay --latency calibrated --scenario delay --delay-s 3.0 --disturb-t0 1.0 --disturb-t1 3.0 \
  --tasks 0 1 2 --init-states 0 --controllers fixed_high physical_aware \
  --controller-overrides "$(cat scripts/e2_overrides.json)"

# E0/E6/E8. refresh-rate frontier against the checkpoint's reference setting (one inference per tick).
# Single-threaded so that all controllers share one numerical configuration.
sharded e0_reference --latency zero --threads 1 --init-states 0 --controllers reference_replan
sharded e6_frontier --latency zero --threads 1 --init-states 0 \
  --controllers fixed_high fixed_h5 fixed_h3 fixed_h2 physical_h1 physical_h2 physical_h1_inverted
sharded e8_frontier_init1 --latency zero --threads 1 --init-states 1 \
  --controllers physical_h2 fixed_high fixed_h5 physical_h2_inverted reference_replan
sharded e8_inverted_init0 --latency zero --threads 1 --init-states 0 --controllers physical_h2_inverted

# E5. vision-feature cache in closed loop; E7. one noise sample per episode (experimental)
$BENCH closed-loop --device "$DEVICE" --threads 1 --latency zero --vision-cache --tasks 0 1 2 --init-states 0 \
  --controllers fixed_high physical_aware --out results/raw/e5_vision_cache${TAG}.jsonl
sharded e7_noise --latency zero --threads 1 --noise-mode episode --init-states 0 --controllers fixed_high fixed_matched

# Model-only measurements (run alone: they need an otherwise idle machine)
$BENCH latency --device "$DEVICE" --cache --out results/latency${TAG}.json
$BENCH realtime --device "$DEVICE" --duration 40 --out results/realtime${TAG}.json
$BENCH realtime --device "$DEVICE" --duration 40 --background-load 3 --out results/realtime_loaded${TAG}.json
$BENCH scheduler --out results/scheduler_sim.json

for e in e1_zero e2_edge e3_dynamic e4_delay e7_noise; do
  $BENCH report results/raw/${e}${TAG}_shard*.jsonl --out results/${e}${TAG}.md
done
$BENCH report results/raw/e6_frontier${TAG}_shard*.jsonl results/raw/e0_reference${TAG}_shard*.jsonl results/raw/e8_*${TAG}_shard*.jsonl \
  --baseline reference_replan --out results/e6_frontier${TAG}.md
$BENCH report results/raw/signals_*${TAG}.jsonl --kind signals --out results/signals${TAG}.md
for e in e1_zero e2_edge e3_dynamic e4_delay; do
  $PY scripts/extra_analysis.py results/raw/${e}${TAG}_shard*.jsonl > results/${e}${TAG}_extra.md
done
