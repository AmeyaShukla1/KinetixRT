"""Closed-loop episode runner (simulator-locked virtual time).

Workflow per tick: sim observation -> Runtime.step (analyzer -> controller ->
scheduler -> real policy -> action buffer -> validator) -> sim.step(action).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kinetixrt.core.runtime import Runtime
from kinetixrt.types import ActionSource, ControlOutput


@dataclass
class EpisodeResult:
    success: bool
    steps: int
    summary: dict
    smoothness: float | None
    max_action_jump: float | None
    wall_s: float
    sim_wall_s: float
    extra: dict[str, Any] = field(default_factory=dict)


def run_episode(
    sim: Any,
    runtime: Runtime,
    task: str,
    seed: int = 0,
    reset_kwargs: dict | None = None,
    max_steps: int | None = None,
    motion_dims: tuple[int, ...] = (0, 1, 2),
    perturb: Callable[[Any, int], bool] | None = None,
    on_tick: Callable[[int, dict, ControlOutput], None] | None = None,
) -> EpisodeResult:
    if runtime.cfg.mode != "virtual":
        raise ValueError("run_episode drives the runtime in virtual-time mode")
    t0 = time.perf_counter()
    obs = sim.reset(**(reset_kwargs or {}))
    runtime.telemetry.reset()
    runtime.reset(seed=seed, task=task)
    limit = max_steps or getattr(sim, "max_steps", 1000)
    sim_wall = 0.0
    executed: list[np.ndarray] = []
    success = False
    perturbed_tick: int | None = None
    steps = 0
    for tick in range(limit):
        out = runtime.step(images=obs["images"], robot_state=obs["state"])
        if on_tick is not None:
            on_tick(tick, obs, out)
        if out.source is ActionSource.CHUNK:
            executed.append(out.action)
        s0 = time.perf_counter()
        obs, success, done = sim.step(out.action)
        if perturb is not None and perturbed_tick is None and perturb(sim, tick):
            perturbed_tick = tick
        sim_wall += time.perf_counter() - s0
        steps = tick + 1
        if done:
            break
    smooth = jump = None
    if len(executed) > 2:
        a = np.stack(executed)[:, list(motion_dims)]
        d = np.linalg.norm(np.diff(a, axis=0), axis=1)
        smooth, jump = float(d.mean()), float(d.max())
    return EpisodeResult(
        success=bool(success),
        steps=steps,
        summary=runtime.summary(),
        smoothness=smooth,
        max_action_jump=jump,
        wall_s=time.perf_counter() - t0,
        sim_wall_s=sim_wall,
        extra={"perturbed_tick": perturbed_tick},
    )
