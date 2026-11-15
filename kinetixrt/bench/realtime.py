"""Threaded (wall-clock) dual-rate benchmark on replayed real observations.

The real policy runs in the inference worker thread while a control loop calls
``Runtime.step`` at a fixed rate. Observations are replayed from a recorded
LIBERO fixture, so this measures *systems* behaviour (fast-path blocking,
period jitter, latency, deadline misses, adaptation to real CPU contention);
it does not measure task success.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from kinetixrt.bench.common import RESULTS, controller_from_name, hardware_info, libero_analyzer, write_json
from kinetixrt.bench.model import _burn, load_fixture
from kinetixrt.core.runtime import Runtime, RuntimeConfig, hold_fallback
from kinetixrt.core.telemetry import latency_summary
from kinetixrt.intelligence.analyzer import AnalyzerConfig, PhysicalEnvironmentAnalyzer


def run_one(backend, name: str, args: argparse.Namespace, obs_list: list) -> dict:
    real = backend.info.is_real_model
    cfg = RuntimeConfig(
        control_dt=1.0 / args.control_hz,
        mode="threaded",
        fallback=hold_fallback((6,) if real else (2,)),
        disagreement_dims=(0, 1, 2) if real else (0, 1),
        scheduler_policy=args.scheduler,
    )
    analyzer = libero_analyzer() if real else PhysicalEnvironmentAnalyzer(AnalyzerConfig(event_dims=(2,), motion_dims=(0, 1)))
    rt = Runtime(backend, controller_from_name(name), cfg, analyzer)
    rt.reset(seed=0, task=obs_list[0].task)
    stop = mp.Event()
    procs: list[mp.Process] = []
    rt.start()
    period = 1.0 / args.control_hz
    step_ms, period_err_ms, marks = [], [], []
    t_start = time.perf_counter()
    next_t = t_start
    tick = 0
    try:
        while True:
            now = time.perf_counter()
            el = now - t_start
            if el >= args.duration:
                break
            if args.background_load and not procs and el >= args.load_start:
                procs = [mp.Process(target=_burn, args=(stop,), daemon=True) for _ in range(args.background_load)]
                for p in procs:
                    p.start()
                marks.append(("load_on", el))
            if procs and not stop.is_set() and el >= args.load_end:
                stop.set()
                marks.append(("load_off", el))
            o = obs_list[(tick // args.frame_hold) % len(obs_list)]
            t0 = time.perf_counter()
            rt.step(images=o.images, robot_state=o.state)
            step_ms.append((time.perf_counter() - t0) * 1e3)
            period_err_ms.append((t0 - next_t) * 1e3)
            tick += 1
            next_t += period
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.perf_counter()
    finally:
        stop.set()
        rt.stop()
        for p in procs:
            p.join(2)
    s = rt.summary()
    infs = list(rt.telemetry.inferences)
    t0_clock = infs[0]["t_submit"] - 0.0 if infs else 0.0
    windows = {}
    if args.background_load:
        for label, lo, hi in [("before_load", 0, args.load_start), ("during_load", args.load_start, args.load_end), ("after_load", args.load_end, args.duration)]:
            sel = [i for i in infs if lo <= i["t_submit"] - (rt.telemetry.ticks[0]["t"] if rt.telemetry.ticks else t0_clock) < hi]
            windows[label] = {
                "inferences": len(sel),
                "mean_steps": float(np.mean([i["steps"] for i in sel])) if sel else None,
                "latency_s": latency_summary([i["latency_s"] for i in sel]),
            }
    return {
        "controller": name,
        "control_hz": args.control_hz,
        "ticks": tick,
        "fast_path_step_ms": latency_summary(step_ms),
        "control_period_error_ms": latency_summary(np.abs(period_err_ms)),
        "inference_latency_s": latency_summary(s["latency_s"]),
        "queue_wait_s": latency_summary(s["queue_wait_s"]),
        "inferences": s["inferences"],
        "mean_steps": s["mean_steps"],
        "steps_histogram": {str(k): int(v) for k, v in zip(*np.unique([i["steps"] for i in infs], return_counts=True))} if infs else {},
        "deadline_miss_ticks": s["deadline_miss_ticks"],
        "startup_wait_ticks": s["startup_wait_ticks"],
        "stale_rejected": s["stale_rejected"],
        "sources": s["sources"],
        "mean_action_age_ticks": s["mean_action_age_ticks"],
        "windows": windows,
        "marks": marks,
    }


def run(args: argparse.Namespace) -> None:
    if args.synthetic:
        from kinetixrt.backends.synthetic import SyntheticBackend
        from kinetixrt.simulation.toy import ToyPickPlace
        from kinetixrt.types import Observation

        backend = SyntheticBackend(base_latency_s=0.05, per_step_latency_s=0.01, sleep=True)
        env = ToyPickPlace()
        o = env.reset(0)
        obs_list = [Observation(0, 0, 0.0, o["images"], o["state"], "toy")]
    else:
        from kinetixrt.backends.smolvla import SmolVLABackend

        backend = SmolVLABackend(args.model, device=args.device, num_threads=args.threads)
        obs_list = load_fixture()
    report = {"hardware": hardware_info(), "backend": vars(backend.info), "args": vars(args), "runs": []}
    for name in args.controllers:
        r = run_one(backend, name, args, obs_list)
        report["runs"].append(r)
        fp, lat = r["fast_path_step_ms"], r["inference_latency_s"]
        print(
            f"[realtime] {name:20s} ticks={r['ticks']} fast-path p50={fp['p50']:.3f} ms p99={fp['p99'] if fp['p99'] is not None else float('nan'):.3f} ms max={fp['max']:.2f} ms | "
            f"inference p50={lat['p50']:.3f} s n={lat['n']} mean_steps={r['mean_steps']:.1f} | deadline-miss ticks={r['deadline_miss_ticks']} steps={r['steps_histogram']}",
            flush=True,
        )
    write_json(Path(args.out), report)
    print(f"[realtime] -> {args.out}")


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("realtime", help="threaded dual-rate run with the real model on replayed observations (systems metrics)")
    p.add_argument("--model", default="HuggingFaceVLA/smolvla_libero")
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--synthetic", action="store_true", help="use the synthetic backend (CPU smoke test, not a model result)")
    p.add_argument("--controllers", nargs="+", default=["fixed_high", "runtime_only"])
    p.add_argument("--control-hz", type=float, default=20.0)
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--frame-hold", type=int, default=10)
    p.add_argument("--scheduler", default="deadline")
    p.add_argument("--background-load", type=int, default=0)
    p.add_argument("--load-start", type=float, default=10.0)
    p.add_argument("--load-end", type=float, default=20.0)
    p.add_argument("--out", default=str(RESULTS / "realtime.json"))
    p.set_defaults(func=run)
