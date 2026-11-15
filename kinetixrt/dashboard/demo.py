"""Live demo drivers feeding the dashboard with real runtime telemetry.

``toy``    - CPU-only: synthetic backend (NOT a trained model) + ToyPickPlace,
             threaded runtime paced in real time. For laptops and smoke tests.
``libero`` - real SmolVLA + LIBERO in simulator-locked virtual time.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from kinetixrt.core.runtime import Runtime, RuntimeConfig, hold_fallback
from kinetixrt.intelligence.analyzer import AnalyzerConfig, PhysicalEnvironmentAnalyzer
from kinetixrt.intelligence.controller import make_controller
from kinetixrt.presets import PRESETS

TOY_PRESETS: dict[str, dict] = {
    "fixed_high": {"type": "fixed", "num_steps": 10, "horizon": 5},
    "fixed_reduced_steps": {"type": "fixed", "num_steps": 3, "horizon": 5},
    "fixed_long_horizon": {"type": "fixed", "num_steps": 10, "horizon": 25},
    "runtime_only": {"type": "runtime_only", "step_options": (3, 5, 10), "default_steps": 10, "horizon": 5},
    "physical_aware": {
        "type": "physical_aware",
        "step_options": (3, 5, 10),
        "steps_calm": 5,
        "horizon_calm": 20,
        "horizon_default": 8,
        "horizon_critical": 4,
        "event_window_pre": 6,
        "event_window_post": 3,
        "surprise_threshold": 0.02,
    },
}


class LiveDemo(threading.Thread):
    def __init__(self, kind: str = "toy", controller: str = "physical_aware", dynamic: bool = True, device: str = "auto", suite: str = "libero_spatial", tasks: tuple[int, ...] = (0, 1, 2)) -> None:
        super().__init__(name="krt-demo", daemon=True)
        self.kind, self.dynamic, self.device, self.suite, self.tasks = kind, dynamic, device, suite, tasks
        self.presets = TOY_PRESETS if kind == "toy" else PRESETS
        self.controller_name = controller
        self._pending: str | None = None
        self._stop = threading.Event()
        self.frame: np.ndarray | None = None
        self.aggregates: dict[str, dict[str, float]] = {}
        self.episode = 0
        self.status: dict[str, Any] = {"state": "starting"}
        self.runtime: Runtime | None = None

    def request_controller(self, name: str) -> None:
        if name not in self.presets:
            raise KeyError(name)
        self._pending = name
        if self.runtime is not None:
            self.runtime.set_controller(make_controller(self.presets[name]))
            self.controller_name = name
            self._pending = None

    def stop(self) -> None:
        self._stop.set()

    def _aggregate(self, name: str, success: bool, summary: dict) -> None:
        a = self.aggregates.setdefault(name, {"episodes": 0, "successes": 0, "inferences": 0, "denoise_steps": 0, "miss_ticks": 0, "ticks": 0, "hold_ticks": 0})
        a["episodes"] += 1
        a["successes"] += int(success)
        a["inferences"] += summary["inferences"]
        a["denoise_steps"] += summary["denoise_steps_total"]
        a["miss_ticks"] += summary["deadline_miss_ticks"]
        a["hold_ticks"] += summary["hold_ticks"]
        a["ticks"] += summary["ticks"]

    def run(self) -> None:
        try:
            (self._run_toy if self.kind == "toy" else self._run_libero)()
        except Exception as exc:  # noqa: BLE001
            self.status = {"state": "error", "error": repr(exc)}
            raise

    # ----------------------------------------------------------------- toy
    def _run_toy(self) -> None:
        from kinetixrt.backends.synthetic import SyntheticBackend
        from kinetixrt.simulation.toy import ToyConfig, ToyPickPlace

        backend = SyntheticBackend(base_latency_s=0.06, per_step_latency_s=0.012, sleep=True)
        dt = 0.05
        rt = Runtime(
            backend,
            make_controller(self.presets[self.controller_name]),
            RuntimeConfig(control_dt=dt, mode="threaded", fallback=hold_fallback((2,)), disagreement_dims=(0, 1)),
            PhysicalEnvironmentAnalyzer(AnalyzerConfig(event_dims=(2,), motion_dims=(0, 1))),
        )
        self.runtime = rt
        rt.start()
        rng = np.random.default_rng(0)
        while not self._stop.is_set():
            seed = int(rng.integers(0, 10_000))
            move_at = int(rng.integers(8, 18)) if self.dynamic and rng.random() < 0.6 else None
            env = ToyPickPlace(ToyConfig(move_object_at=move_at), seed)
            obs = env.reset(seed)
            rt.telemetry.reset()
            rt.reset(seed=seed, task="toy pick and place")
            name = self.controller_name
            self.status = {"state": "running", "episode": self.episode, "scenario": "object moves" if move_at else "static"}
            success, done = False, False
            next_t = time.perf_counter()
            while not done and not self._stop.is_set():
                out = rt.step(images=obs["images"], robot_state=obs["state"])
                obs, success, done = env.step(out.action)
                self.frame = np.kron(env.render(), np.ones((4, 4, 1), dtype=np.uint8))
                next_t += dt
                time.sleep(max(0.0, next_t - time.perf_counter()))
            self._aggregate(name, success, rt.summary())
            self.episode += 1
            time.sleep(0.4)
        rt.stop()

    # -------------------------------------------------------------- libero
    def _run_libero(self) -> None:
        from kinetixrt.backends.smolvla import SmolVLABackend
        from kinetixrt.bench.common import make_latency_model, make_libero_runtime
        from kinetixrt.simulation.libero import LiberoSim, Perturbation
        from kinetixrt.simulation.rollout import run_episode

        self.status = {"state": "loading model"}
        backend = SmolVLABackend(device=self.device)
        rt = make_libero_runtime(backend, make_controller(self.presets[self.controller_name]), make_latency_model("calibrated"))
        self.runtime = rt
        sims: dict[int, LiberoSim] = {}
        init = 0
        while not self._stop.is_set():
            task_id = self.tasks[self.episode % len(self.tasks)]
            if task_id not in sims:
                self.status = {"state": "building simulator", "task": task_id}
                sims[task_id] = LiberoSim(self.suite, task_id)
            sim = sims[task_id]
            name = self.controller_name
            self.status = {"state": "running", "episode": self.episode, "task": sim.task, "scenario": "object moves at tick 30" if self.dynamic else "static"}
            p = Perturbation(tick=30)

            def perturb(s, tick, p=p):  # noqa: ANN001
                if tick == 0:
                    s.remember_object_height()
                return self.dynamic and tick == p.tick and s.perturb(p)

            def on_tick(tick, obs, out):  # noqa: ANN001
                self.frame = obs["images"]["image"]

            r = run_episode(sim, rt, sim.task, seed=init, reset_kwargs={"init_state": init, "seed": init}, perturb=perturb, on_tick=on_tick)
            self._aggregate(name, r.success, r.summary)
            self.episode += 1
            init += 1
