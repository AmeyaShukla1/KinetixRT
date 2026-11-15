"""Signal validation study (real model, real simulator).

For every ``stride`` control ticks of a closed-loop episode it solves the same
observation with several budgets and noise samples (sharing one prefix pass)
and records, next to the runtime's physical signals:

* ``sens_n``     - action error of an n-step solve vs the 10-step solve
                   (same noise): where does cutting denoising steps hurt?
* ``sample_dis`` - disagreement of two independent noise samples (10 steps).
* ``drift_k``    - difference between the plan made k ticks ago and the fresh
                   plan for the same future ticks: what would executing the
                   old plan open-loop for k more ticks have cost?

These are error *proxies* measured against the model's own best prediction;
task success is evaluated separately by ``krt-bench closed-loop``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from kinetixrt.bench.common import RESULTS, hardware_info, libero_analyzer
from kinetixrt.types import Observation

K = 10
MOTION = [0, 1, 2]
ROT = [3, 4, 5]
GRIP = 6


def _first_flip(plan: np.ndarray, ref: float) -> int | None:
    flips = np.nonzero(np.sign(plan[:, GRIP]) != np.sign(ref))[0]
    return int(flips[0]) if flips.size else None


def run(args: argparse.Namespace) -> None:
    from kinetixrt.backends.smolvla import SmolVLABackend
    from kinetixrt.simulation.libero import LiberoSim

    backend = SmolVLABackend(args.model, device=args.device, num_threads=args.threads)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"kind": "meta", "hardware": hardware_info(), "args": vars(args), "backend": vars(backend.info)}
    with open(out_path, "a") as fh:
        fh.write(json.dumps(meta, default=str) + "\n")
        for task_id in args.tasks:
            sim = LiberoSim(args.suite, task_id)
            for init in args.init_states:
                t0 = time.time()
                obs = sim.reset(init_state=init, seed=init)
                backend.reset(seed=1000 * task_id + init)
                analyzer = libero_analyzer()
                plans: list[tuple[int, np.ndarray]] = []
                plan, anchor = None, 0
                rows = []
                success = False
                last_grip = -1.0
                for tick in range(sim.max_steps):
                    o = Observation(tick, tick, tick * sim.control_dt, obs["images"], obs["state"], sim.task)
                    remaining = plan[tick - anchor :] if plan is not None else None
                    sig = analyzer.update(o, remaining)
                    if tick % args.stride == 0:
                        noises = [backend.sample_noise(), backend.sample_noise()]
                        res = backend.infer_sweep(o, [10, 5, 3], noises[:1])
                        res_b = backend.infer_sweep(o, [10], noises[1:])
                        a10, a5, a3, a10b = res[(0, 10)], res[(0, 5)], res[(0, 3)], res_b[(0, 10)]
                        row = {
                            "kind": "sample",
                            "task": task_id,
                            "init": init,
                            "tick": tick,
                            "sens3": float(np.linalg.norm((a10 - a3)[:K, MOTION], axis=1).mean()),
                            "sens5": float(np.linalg.norm((a10 - a5)[:K, MOTION], axis=1).mean()),
                            "sens3_rot": float(np.linalg.norm((a10 - a3)[:K, ROT], axis=1).mean()),
                            "sens3_grip": float(np.mean(np.sign(a10[:K, GRIP]) != np.sign(a3[:K, GRIP]))),
                            "sample_dis": float(np.linalg.norm((a10 - a10b)[:K, MOTION], axis=1).mean()),
                            "sample_grip": float(np.mean(np.sign(a10[:K, GRIP]) != np.sign(a10b[:K, GRIP]))),
                            "ticks_to_event_new": _first_flip(a10, last_grip),
                            "speed_new": float(np.linalg.norm(a10[:8, MOTION], axis=1).mean()),
                            "signals": sig.as_dict(),
                        }
                        for age_tick, old in plans[-8:]:
                            k = tick - age_tick
                            if k + K <= len(old):
                                row[f"drift_{k}"] = float(np.linalg.norm((old[k : k + K] - a10[:K])[:, MOTION], axis=1).mean())
                                row[f"drift_grip_{k}"] = float(np.mean(np.sign(old[k : k + K, GRIP]) != np.sign(a10[:K, GRIP])))
                        rows.append(row)
                        plans.append((tick, a10))
                        plan, anchor = a10, tick
                        analyzer.note_plan_installed(tick)
                    action = plan[tick - anchor]
                    analyzer.note_executed(action, tick, True)
                    last_grip = float(action[GRIP])
                    obs, success, done = sim.step(action)
                    if done:
                        break
                for r in rows:
                    r["episode_success"] = success
                    r["episode_len"] = tick + 1
                    fh.write(json.dumps(r) + "\n")
                fh.flush()
                print(f"[signals] task={task_id} init={init} success={success} ticks={tick + 1} samples={len(rows)} wall={time.time() - t0:.0f}s", flush=True)
            sim.close()


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("signals", help="validate physical signals against model error proxies (real model + LIBERO)")
    p.add_argument("--model", default="HuggingFaceVLA/smolvla_libero")
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--tasks", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--init-states", type=int, nargs="+", default=[40, 41])
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--out", default=str(RESULTS / "signals.jsonl"))
    p.set_defaults(func=run)
