"""Action validation layer, deliberately independent of the inference scheduler.

This is a simulation-grade guard (finite checks, bounds, rate limits, expiry).
It is NOT a certified safety controller and makes no hard real-time claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ValidatorConfig:
    low: np.ndarray | float = -1.0
    high: np.ndarray | float = 1.0
    max_delta: np.ndarray | float | None = None
    """Per-tick limit on the change of each action dimension (None = disabled)."""
    max_chunk_age_ticks: int | None = None
    """Actions from a plan older than this many ticks are refused."""
    rate_limit_dims: tuple[int, ...] | None = None


@dataclass
class ValidationResult:
    action: np.ndarray | None
    ok: bool
    reason: str = "ok"
    clipped: bool = False
    rate_limited: bool = False


class ActionValidator:
    def __init__(self, cfg: ValidatorConfig | None = None) -> None:
        self.cfg = cfg or ValidatorConfig()
        self._last: np.ndarray | None = None
        self.counts = {"non_finite": 0, "expired": 0, "clipped": 0, "rate_limited": 0, "shape": 0}

    def reset(self) -> None:
        self._last = None

    def validate(self, action: np.ndarray, chunk_age_ticks: int, action_dim: int | None = None) -> ValidationResult:
        cfg = self.cfg
        a = np.asarray(action, dtype=np.float32)
        if a.ndim != 1 or (action_dim is not None and a.shape[0] != action_dim):
            self.counts["shape"] += 1
            return ValidationResult(None, False, "bad_shape")
        if not np.all(np.isfinite(a)):
            self.counts["non_finite"] += 1
            return ValidationResult(None, False, "non_finite")
        if cfg.max_chunk_age_ticks is not None and chunk_age_ticks > cfg.max_chunk_age_ticks:
            self.counts["expired"] += 1
            return ValidationResult(None, False, "expired")
        clipped_a = np.clip(a, cfg.low, cfg.high)
        clipped = bool(np.any(clipped_a != a))
        rate_limited = False
        if cfg.max_delta is not None and self._last is not None:
            lim = np.broadcast_to(np.asarray(cfg.max_delta, dtype=np.float32), a.shape)
            delta = clipped_a - self._last
            limited = np.clip(delta, -lim, lim)
            if cfg.rate_limit_dims is not None:
                mask = np.zeros(a.shape, dtype=bool)
                mask[list(cfg.rate_limit_dims)] = True
                limited = np.where(mask, limited, delta)
            if np.any(limited != delta):
                rate_limited = True
                clipped_a = self._last + limited
        if clipped:
            self.counts["clipped"] += 1
        if rate_limited:
            self.counts["rate_limited"] += 1
        self._last = clipped_a.copy()
        return ValidationResult(clipped_a, True, "ok", clipped, rate_limited)

    def note_fallback(self, action: np.ndarray) -> None:
        self._last = np.asarray(action, dtype=np.float32).copy()
