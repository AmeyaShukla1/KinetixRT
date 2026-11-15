"""Latency models for simulator-locked (virtual-time) execution.

In virtual-time mode the model really runs, but the simulator waits for it.
A latency model decides how much *runtime-clock* time that inference is
charged, so that staleness, deadline misses and fallbacks are reproduced
deterministically. Which model was used is recorded with every result, and
BENCHMARKS.md states for each table whether latency was measured or modeled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from kinetixrt.types import InferenceResult


class LatencyModel(Protocol):
    name: str

    def latency(self, result: InferenceResult, measured_wall_s: float, now: float) -> float: ...


@dataclass
class ZeroLatency:
    """Standard synchronous benchmark protocol: the world waits for the model."""

    name: str = "zero"

    def latency(self, result: InferenceResult, measured_wall_s: float, now: float) -> float:
        return 0.0


@dataclass
class MeasuredLatency:
    """Charge the wall-clock time actually measured on this machine (x scale)."""

    scale: float = 1.0
    name: str = "measured"

    def latency(self, result: InferenceResult, measured_wall_s: float, now: float) -> float:
        return measured_wall_s * self.scale


@dataclass
class CalibratedLatency:
    """``base + per_step * num_steps`` with constants fitted to measurements.

    Deterministic stand-in for :class:`MeasuredLatency` so that closed-loop
    results do not depend on how many benchmark processes share the CPU.
    ``vision_s`` is the per-camera vision cost removed on a feature-cache hit.
    """

    base_s: float
    per_step_s: float
    vision_s: float = 0.0
    scale: float = 1.0
    name: str = "calibrated"

    def latency(self, result: InferenceResult, measured_wall_s: float, now: float) -> float:
        t = self.base_s + self.per_step_s * result.budget.num_steps
        t -= self.vision_s * result.timings.vision_cache_hits
        return max(0.0, t) * self.scale


@dataclass
class Disturbance:
    """Injected slowdown window: ``latency * multiplier + extra_s`` in [t0, t1)."""

    t0: float
    t1: float
    multiplier: float = 1.0
    extra_s: float = 0.0


@dataclass
class DisturbedLatency:
    """Wrap a base model with injected contention / delay windows (scenarios D, E)."""

    base: LatencyModel
    disturbances: list[Disturbance] = field(default_factory=list)
    name: str = "disturbed"

    def latency(self, result: InferenceResult, measured_wall_s: float, now: float) -> float:
        t = self.base.latency(result, measured_wall_s, now)
        for d in self.disturbances:
            if d.t0 <= now < d.t1:
                t = t * d.multiplier + d.extra_s
        return t
