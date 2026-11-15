#!/usr/bin/env python
"""Secondary statistics from closed-loop JSONL results (used in BENCHMARKS.md).

    python scripts/extra_analysis.py results/raw/e1_zero_shard*.jsonl

Prints, per controller: plan invalidations and holds (scene-surprise activity),
phase tick fractions, fallback-source tick fractions, fast-path cost, and
per-task success counts.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np


def load(paths):
    latest = {}
    for p in paths:
        with open(p) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("kind") == "episode":  # same rule as `krt-bench report`: last run of a key wins
                    latest[(r["controller"], r["task"], r["init"])] = r
    return list(latest.values())


def main(paths):
    rows = load(paths)
    by = defaultdict(list)
    for r in rows:
        by[r["controller"]].append(r)
    tasks = sorted({r["task"] for r in rows})

    print("| controller | episodes | episodes with a plan invalidation | invalidations/ep | hold ticks/ep "
          "| stale results rejected/ep | startup wait ticks | starved+expired ticks/ep | chunk-sourced ticks | "
          "fast path p50 / max-of-p95 (ms) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for c in sorted(by):
        ep = by[c]
        n = len(ep)
        inv = [r.get("plan_invalidations", 0) for r in ep]
        total_ticks = sum(r["steps"] for r in ep)
        src = defaultdict(int)
        for r in ep:
            for k, v in r.get("sources", {}).items():
                src[k] += v
        starved = src.get("fallback_starved", 0) + src.get("fallback_expired", 0)
        fp50 = np.median([r["fast_path_ms"]["p50"] for r in ep])
        fp95 = max((r["fast_path_ms"]["p95"] or 0.0) for r in ep)
        print(f"| `{c}` | {n} | {sum(1 for v in inv if v > 0)} | {np.mean(inv):.2f} | "
              f"{np.mean([r.get('hold_ticks', 0) for r in ep]):.1f} | "
              f"{np.mean([r.get('stale_rejected', 0) for r in ep]):.2f} | "
              f"{np.mean([r.get('startup_wait_ticks', 0) for r in ep]):.1f} | {starved / n:.1f} | "
              f"{100 * src.get('chunk', 0) / max(total_ticks, 1):.1f}% | {fp50:.1f} / {fp95:.1f} |")

    print("\nPhase tick fractions (controllers that classify phases):\n")
    print("| controller | calm | default | critical |")
    print("|---|---|---|---|")
    for c in sorted(by):
        ph = defaultdict(int)
        for r in by[c]:
            for k, v in (r.get("phase_ticks") or {}).items():
                ph[k] += v
        tot = sum(ph.values())
        if tot == 0 or len(ph) < 2:
            continue
        print(f"| `{c}` | {100 * ph.get('calm', 0) / tot:.0f}% | {100 * ph.get('default', 0) / tot:.0f}% | "
              f"{100 * ph.get('critical', 0) / tot:.0f}% |")

    print("\nSuccesses per task (successes / episodes):\n")
    print("| controller | " + " | ".join(f"t{t}" for t in tasks) + " |")
    print("|---|" + "---|" * len(tasks))
    for c in sorted(by):
        cells = []
        for t in tasks:
            e = [r for r in by[c] if r["task"] == t]
            cells.append(f"{sum(r['success'] for r in e)}/{len(e)}" if e else "-")
        print(f"| `{c}` | " + " | ".join(cells) + " |")

    pert = [r for r in rows if r.get("perturbed_tick") is not None]
    if pert:
        print("\nPerturbed episodes: scene-surprise response\n")
        print("| controller | perturbed episodes | max surprise median | episodes with invalidation | "
              "median ticks from perturbation to first invalidation |")
        print("|---|---|---|---|---|")
        for c in sorted(by):
            e = [r for r in by[c] if r.get("perturbed_tick") is not None]
            if not e:
                continue
            lat = [min(t for t in r["invalidation_ticks"] if t >= r["perturbed_tick"]) - r["perturbed_tick"]
                   for r in e if any(t >= r["perturbed_tick"] for t in r.get("invalidation_ticks", []))]
            print(f"| `{c}` | {len(e)} | {np.median([r.get('surprise_max', 0) for r in e]):.3f} | "
                  f"{len(lat)} | {np.median(lat) if lat else float('nan'):.0f} |")


if __name__ == "__main__":
    main(sys.argv[1:])
