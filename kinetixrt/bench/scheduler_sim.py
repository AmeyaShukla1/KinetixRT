"""Scheduler policy comparison under overload.

SYNTHETIC WORKLOAD: a discrete-event simulation of several robots sharing one
accelerator, with a latency model ``base + per_step * steps``. It exercises the
real :class:`InferenceScheduler` code but no model. Its numbers characterise
the scheduling policies only and are never reported as model results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from kinetixrt.bench.common import RESULTS, write_json
from kinetixrt.core.scheduler import InferenceRequest, InferenceScheduler, LatencyPredictor
from kinetixrt.types import ComputeBudget, Observation


def simulate(policy: str, streams: int, base_s: float, per_step_s: float, duration_s: float, period_s: float, chunk_s: float, seed: int, degrade: bool) -> dict:
    rng = np.random.default_rng(seed)
    pred = LatencyPredictor()
    sched = InferenceScheduler(policy, capacity=max(2, streams), predictor=pred, step_options=(3, 5, 10), max_obs_age_s=chunk_s, allow_degrade=degrade)
    for n in (3, 5, 10):
        pred.observe(n, base_s + per_step_s * n)
    next_req = [rng.uniform(0, period_s) for _ in range(streams)]
    plan_until = [0.0] * streams
    starved = [0.0] * streams
    t, busy_until, rid = 0.0, 0.0, 0
    current = None
    ages, steps_used, served, waits = [], [], 0, []
    dt = 0.005
    while t < duration_s:
        for sidx in range(streams):
            if t >= next_req[sidx]:
                obs = Observation(rid, int(t / 0.05), t, {}, np.zeros(1), stream_id=f"robot{sidx}")
                sched.submit(InferenceRequest(rid, obs, ComputeBudget(10), t, deadline=max(t, plan_until[sidx])))
                rid += 1
                next_req[sidx] += period_s
            if t > plan_until[sidx] and t > period_s:
                starved[sidx] += dt
        if current is not None and t >= busy_until:
            req = current
            sidx = int(req.obs.stream_id[5:])
            plan_until[sidx] = req.obs.t_capture + chunk_s
            ages.append(t - req.obs.t_capture)
            steps_used.append(req.budget.num_steps)
            served += 1
            current = None
        if current is None:
            req, _ = sched.take(t)
            if req is not None:
                current = req
                waits.append(t - req.t_submit)
                jitter = rng.lognormal(0.0, 0.1)
                busy_until = t + (base_s + per_step_s * req.budget.num_steps) * jitter
        t += dt
    return {
        "policy": policy + ("+degrade" if degrade else ""),
        "streams": streams,
        "served": served,
        "result_age_p50_s": float(np.percentile(ages, 50)) if ages else None,
        "result_age_p95_s": float(np.percentile(ages, 95)) if ages else None,
        "queue_wait_p95_s": float(np.percentile(waits, 95)) if waits else None,
        "mean_steps": float(np.mean(steps_used)) if steps_used else None,
        "starved_fraction": float(np.sum(starved) / (streams * duration_s)),
        "max_queue_depth": sched.stats["max_depth"],
        "superseded": sched.stats["superseded"],
        "rejected_full": sched.stats["rejected_full"],
        "dropped_obsolete": sched.stats["dropped_obsolete"],
    }


def run(args: argparse.Namespace) -> None:
    rows = []
    for streams in args.streams:
        for policy, degrade in [("fifo", False), ("latest", False), ("deadline", False), ("deadline", True)]:
            rows.append(simulate(policy, streams, args.base_s, args.per_step_s, args.duration, args.period, args.chunk_s, args.seed, degrade))
    write_json(Path(args.out), {"synthetic": True, "args": vars(args), "rows": rows})
    print("SYNTHETIC scheduler simulation (no model executed)")
    print("| streams | policy | served | result age p50 / p95 (s) | mean steps | starved time | max queue | superseded | rejected |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['streams']} | {r['policy']} | {r['served']} | {r['result_age_p50_s']:.2f} / {r['result_age_p95_s']:.2f} | {r['mean_steps']:.1f} | "
            f"{100 * r['starved_fraction']:.1f}% | {r['max_queue_depth']} | {r['superseded']} | {r['rejected_full']} |"
        )


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("scheduler", help="SYNTHETIC multi-robot overload simulation of the scheduling policies")
    p.add_argument("--streams", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--base-s", type=float, default=0.06)
    p.add_argument("--per-step-s", type=float, default=0.007)
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--period", type=float, default=0.25)
    p.add_argument("--chunk-s", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(RESULTS / "scheduler_sim.json"))
    p.set_defaults(func=run)
