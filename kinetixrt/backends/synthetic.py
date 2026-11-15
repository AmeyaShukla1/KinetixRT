"""Synthetic backend for runtime tests and CPU-only demos.

THIS IS NOT A TRAINED MODEL. It is a small analytic visuomotor controller for
:class:`kinetixrt.simulation.toy.ToyPickPlace` whose cost and accuracy depend
on the compute budget in the same *shape* as a flow-matching policy (fixed
prefix cost + per-step cost; integration error shrinking with step count).
It exists so the scheduler, buffers, controllers and dashboard can be tested
on a laptop without model weights. Results from it are labelled synthetic and
are never reported as model results.
"""

from __future__ import annotations

import time

import numpy as np

from kinetixrt.backends.base import BackendInfo, PolicyBackend
from kinetixrt.types import ComputeBudget, InferenceResult, Observation, StageTimings


def _centroid(img: np.ndarray, channel: int) -> np.ndarray | None:
    """Centroid (x, y in [0, 1]) of pixels dominated by one color channel."""
    a = img.astype(np.float32)
    others = [c for c in range(3) if c != channel]
    mask = (a[..., channel] > 150) & (a[..., others[0]] < 100) & (a[..., others[1]] < 100)
    if mask.sum() < 2:
        return None
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    return np.array([xs.mean() / (w - 1), ys.mean() / (h - 1)], dtype=np.float32)


class SyntheticBackend(PolicyBackend):
    def __init__(
        self,
        chunk_size: int = 30,
        base_latency_s: float = 0.0,
        per_step_latency_s: float = 0.0,
        error_scale: float = 0.6,
        sleep: bool = False,
        fail_every: int = 0,
        seed: int = 0,
    ) -> None:
        self.base_latency_s, self.per_step_latency_s = base_latency_s, per_step_latency_s
        self.error_scale, self.sleep, self.fail_every = error_scale, sleep, fail_every
        self.extra_delay_s = 0.0
        self.calls = 0
        self.info = BackendInfo(
            name="synthetic-analytic",
            model_id="synthetic/toy-pickplace-controller",
            device="cpu",
            chunk_size=chunk_size,
            action_dim=3,
            default_steps=10,
            supported_steps=(1, 3, 5, 10),
            is_real_model=False,
        )
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def modeled_latency(self, num_steps: int) -> float:
        return self.base_latency_s + self.per_step_latency_s * num_steps + self.extra_delay_s

    def infer(self, obs: Observation, budget: ComputeBudget) -> InferenceResult:
        self.calls += 1
        if self.fail_every and self.calls % self.fail_every == 0:
            raise RuntimeError("synthetic backend failure (injected)")
        t0 = time.perf_counter()
        img = obs.images["image"]
        obj, goal = _centroid(img, 0), _centroid(img, 1)
        pos = np.asarray(obs.state[:2], dtype=np.float32)
        holding = obs.state[2] > 0.5 and obj is not None and np.linalg.norm(obj - pos) < 0.06
        T = self.info.chunk_size
        actions = np.zeros((T, 3), dtype=np.float32)
        p = pos.copy()
        grip = 1.0 if obs.state[2] > 0.5 else -1.0
        max_step = 0.03
        sigma = self.error_scale * max_step / budget.num_steps
        for i in range(T):
            if holding and goal is not None:
                target, at_grip = goal, -1.0
            elif obj is not None:
                target, at_grip = obj, 1.0
                if not holding and float(np.linalg.norm(obj - p)) >= 0.015:
                    grip = -1.0  # approach with an open gripper
            else:
                target, at_grip = p, grip
            delta = target - p
            dist = float(np.linalg.norm(delta))
            if dist < 0.015:
                grip = at_grip
                if at_grip > 0:
                    holding = True
                step = np.zeros(2, dtype=np.float32)
            else:
                step = delta / dist * min(max_step, dist)
            noisy = step + self._rng.normal(0.0, sigma, size=2).astype(np.float32)
            actions[i, :2] = noisy / max_step
            actions[i, 2] = grip
            p = p + step
        latency = self.modeled_latency(budget.num_steps)
        if self.sleep and latency > 0:
            time.sleep(latency)
        wall = time.perf_counter() - t0
        timings = StageTimings(
            prefix_s=self.base_latency_s,
            denoise_s=self.per_step_latency_s * budget.num_steps,
            total_s=wall if self.sleep else latency,
        )
        return InferenceResult(np.clip(actions, -1.0, 1.0), budget, timings, extras={"synthetic": True})
