"""Closed-loop benchmark: real policy through the runtime in LIBERO.

Every episode is one JSON line; runs are resumable and shardable so several
processes can share the work. Episodes are paired across controllers: the same
(task, init_state) pairs and the same noise seed are used for each controller.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from kinetixrt.bench.common import RESULTS, controller_from_name, hardware_info, make_latency_model, make_libero_runtime
from kinetixrt.core.latency import Disturbance
from kinetixrt.core.telemetry import latency_summary


def _done_keys(path: Path) -> set[tuple]:
    keys = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("kind") == "episode":
                keys.add((r["controller"], r["task"], r["init"]))
    return keys


def run(args: argparse.Namespace) -> None:
    from kinetixrt.backends.smolvla import SmolVLABackend
    from kinetixrt.intelligence.temporal_cache import VisionFeatureCache
    from kinetixrt.simulation.libero import LiberoSim, Perturbation
    from kinetixrt.simulation.rollout import run_episode

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_keys(out_path)
    cache = VisionFeatureCache(tau_mean=args.cache_tau_mean, tau_peak=args.cache_tau_peak) if args.vision_cache else None
    backend = SmolVLABackend(args.model, device=args.device, num_threads=args.threads, vision_cache=cache)
    disturbances = []
    if args.scenario == "contention":
        disturbances = [Disturbance(t0=args.disturb_t0, t1=args.disturb_t1, multiplier=args.contention_multiplier)]
    elif args.scenario == "delay":
        disturbances = [Disturbance(t0=args.disturb_t0, t1=args.disturb_t1, extra_s=args.delay_s)]
    overrides = json.loads(args.controller_overrides) if args.controller_overrides else {}
    jobs = [(c, t, i) for t in args.tasks for i in args.init_states for c in args.controllers]
    jobs = [j for n, j in enumerate(jobs) if n % args.num_shards == args.shard and j not in done]
    meta = {
        "kind": "meta",
        "hardware": hardware_info(),
        "args": vars(args),
        "backend": vars(backend.info),
        "latency_model": args.latency,
        "scenario": args.scenario,
    }
    sims: dict[int, LiberoSim] = {}
    with open(out_path, "a") as fh:
        fh.write(json.dumps(meta, default=str) + "\n")
        for name, task_id, init in jobs:
            if task_id not in sims:
                for s in sims.values():
                    s.close()
                sims = {task_id: LiberoSim(args.suite, task_id)}
            sim = sims[task_id]
            controller = controller_from_name(name, overrides.get(name))
            if args.vision_cache and hasattr(controller, "cfg"):
                controller.cfg.reuse_vision = True
            if args.vision_cache and hasattr(controller, "reuse_vision"):
                controller.reuse_vision = True
            latency = make_latency_model(args.latency, args.scale, disturbances=disturbances)
            rt = make_libero_runtime(backend, controller, latency, control_dt=sim.control_dt)
            perturb = None
            if args.scenario == "dynamic":
                p = Perturbation(tick=args.perturb_tick, dx=args.perturb_dx, dy=args.perturb_dy)

                def perturb(s, tick, p=p):  # noqa: ANN001
                    if tick == 0:
                        s.remember_object_height()
                    return tick == p.tick and s.perturb(p)

            seed = 100000 * task_id + init
            if cache is not None:
                cache.stats = dict.fromkeys(cache.stats, 0)
            r = run_episode(sim, rt, sim.task, seed=seed, reset_kwargs={"init_state": init, "seed": init}, perturb=perturb)
            s = r.summary
            rec = {
                "kind": "episode",
                "controller": name,
                "task": task_id,
                "init": init,
                "suite": args.suite,
                "scenario": args.scenario,
                "latency_model": args.latency,
                "scale": args.scale,
                "success": r.success,
                "steps": r.steps,
                "inferences": s["inferences"],
                "accepted_chunks": s["accepted_chunks"],
                "denoise_steps_total": s["denoise_steps_total"],
                "mean_steps": s["mean_steps"],
                "model_wall_s_total": s["model_wall_s_total"],
                "latency_s": latency_summary(s["latency_s"]),
                "latency_all_s": [round(x, 4) for x in s["latency_s"]],
                "exec_wall_s": [round(x, 4) for x in [i["model_wall_s"] for i in rt.telemetry.inferences]],
                "steps_per_inference": [i["steps"] for i in rt.telemetry.inferences],
                "deadline_miss_ticks": s["deadline_miss_ticks"],
                "startup_wait_ticks": s["startup_wait_ticks"],
                "hold_ticks": s["hold_ticks"],
                "stale_rejected": s["stale_rejected"],
                "plan_invalidations": s["plan_invalidations"],
                "mean_action_age_ticks": s["mean_action_age_ticks"],
                "max_action_age_ticks": s["max_action_age_ticks"],
                "phase_ticks": s["phase_ticks"],
                "sources": s["sources"],
                "validator": s["validator"],
                "cache_hits": s["cache_hits"],
                "cache_misses": s["cache_misses"],
                "cache_stats": dict(cache.stats) if cache is not None else None,
                "smoothness": r.smoothness,
                "max_action_jump": r.max_action_jump,
                "fast_path_ms": latency_summary(s["fast_path_ms"]),
                "perturbed_tick": r.extra.get("perturbed_tick"),
                "surprise_max": max((t["surprise"] for t in rt.telemetry.ticks), default=0.0),
                "invalidation_ticks": [e["tick"] for e in rt.telemetry.events if e["event"] == "plan_invalidated"],
                "wall_s": r.wall_s,
                "sim_wall_s": r.sim_wall_s,
                "t": time.time(),
            }
            fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()
            print(
                f"[closed-loop] {name:22s} task={task_id} init={init} success={int(r.success)} steps={r.steps} "
                f"inf={s['inferences']} dsteps={s['denoise_steps_total']} miss={s['deadline_miss_ticks']} wall={r.wall_s:.0f}s",
                flush=True,
            )
    for s in sims.values():
        s.close()


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("closed-loop", help="closed-loop LIBERO evaluation of compute controllers (real model)")
    p.add_argument("--model", default="HuggingFaceVLA/smolvla_libero")
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--tasks", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--init-states", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--controllers", nargs="+", default=["fixed_high", "fixed_reduced_steps", "runtime_only", "physical_aware"])
    p.add_argument("--controller-overrides", default=None, help='JSON: {"preset": {field: value}}')
    p.add_argument("--latency", choices=["zero", "calibrated", "measured"], default="zero")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--scenario", choices=["nominal", "dynamic", "contention", "delay"], default="nominal")
    p.add_argument("--perturb-tick", type=int, default=30)
    p.add_argument("--perturb-dx", type=float, default=0.0)
    p.add_argument("--perturb-dy", type=float, default=0.08)
    p.add_argument("--disturb-t0", type=float, default=2.0)
    p.add_argument("--disturb-t1", type=float, default=8.0)
    p.add_argument("--contention-multiplier", type=float, default=3.0)
    p.add_argument("--delay-s", type=float, default=2.0)
    p.add_argument("--vision-cache", action="store_true")
    p.add_argument("--cache-tau-mean", type=float, default=0.004)
    p.add_argument("--cache-tau-peak", type=float, default=0.03)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--out", default=str(RESULTS / "closed_loop.jsonl"))
    p.set_defaults(func=run)
