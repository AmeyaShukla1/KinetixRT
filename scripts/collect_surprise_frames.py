"""Collect downsampled real LIBERO frame sequences (with and without an object
displacement) to calibrate the scene-surprise detector offline.

Usage: python scripts/collect_surprise_frames.py results/raw/surprise_frames.npz
"""

import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from kinetixrt.backends.smolvla import SmolVLABackend  # noqa: E402
from kinetixrt.bench.common import controller_from_name, make_latency_model, make_libero_runtime  # noqa: E402
from kinetixrt.intelligence.analyzer import downsample_gray  # noqa: E402
from kinetixrt.simulation.libero import LiberoSim, Perturbation  # noqa: E402
from kinetixrt.simulation.rollout import run_episode  # noqa: E402

out = sys.argv[1]
tasks = [int(t) for t in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2, 3, 4, 5]
backend = SmolVLABackend(num_threads=2)
data = {}
for task in tasks:
    sim = LiberoSim("libero_spatial", task)
    for label, ptick in [("nominal", None), ("perturbed", 15 + 5 * (task % 4))]:
        rt = make_libero_runtime(backend, controller_from_name("fixed_high"), make_latency_model("zero"))
        frames, moving = [], []

        def on_tick(tick, obs, o, frames=frames, moving=moving):
            frames.append(downsample_gray(obs["images"]["image"], 72).astype(np.float16))
            moving.append(o.source.value == "chunk")

        def perturb(s, tick, ptick=ptick):
            if tick == 0:
                s.remember_object_height()
            return ptick is not None and tick == ptick and s.perturb(Perturbation(tick=ptick))

        r = run_episode(sim, rt, sim.task, seed=task, reset_kwargs={"init_state": 43, "seed": 43}, perturb=perturb, on_tick=on_tick)
        key = f"t{task}_{label}"
        data[key] = np.stack(frames)
        data[key + "_ptick"] = np.array(-1 if r.extra["perturbed_tick"] is None else r.extra["perturbed_tick"])
        print(key, "success", r.success, "steps", r.steps, "perturbed", r.extra["perturbed_tick"], flush=True)
        np.savez_compressed(out, **data)
    sim.close()
