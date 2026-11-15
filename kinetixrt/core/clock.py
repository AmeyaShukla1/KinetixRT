"""Clocks. The runtime never calls ``time`` directly so that the same logic
runs in wall-clock (threaded) mode and in simulator-locked virtual time."""

from __future__ import annotations

import time
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float: ...


class WallClock:
    """Monotonic wall clock."""

    def now(self) -> float:
        return time.perf_counter()


class VirtualClock:
    """Clock advanced explicitly by the driver (one control period per step)."""

    def __init__(self, start: float = 0.0) -> None:
        self._t = float(start)

    def now(self) -> float:
        return self._t

    def set(self, t: float) -> None:
        if t < self._t - 1e-12:
            raise ValueError("virtual clock cannot go backwards")
        self._t = float(t)

    def advance(self, dt: float) -> None:
        self.set(self._t + dt)
