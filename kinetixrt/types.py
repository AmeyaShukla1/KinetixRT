"""Shared typed data structures for the KinetixRT runtime.

Everything that crosses a module boundary (observations, budgets, decisions,
action chunks) is defined here so that backends, controllers and drivers only
depend on this file and on numpy.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ComputeBudget:
    """How much model computation one inference call is allowed to spend."""

    num_steps: int
    """Number of flow-matching (Euler) integration steps of the action expert."""
    reuse_vision: bool = False
    """Allow the backend to reuse cached per-camera vision features when its
    invalidation rules say the cached entry is still valid."""
    label: str = "default"

    def __post_init__(self) -> None:
        if self.num_steps < 1:
            raise ValueError("num_steps must be >= 1")


@dataclass
class Observation:
    """One robot observation as captured by the fast loop."""

    seq: int
    tick: int
    t_capture: float
    images: dict[str, np.ndarray]
    """Camera name -> HxWx3 uint8 image."""
    state: np.ndarray
    """Proprioceptive state vector."""
    task: str = ""
    stream_id: str = "robot0"


@dataclass
class StageTimings:
    """Wall-clock seconds per inference stage (measured, not modeled)."""

    preprocess_s: float = 0.0
    vision_s: float = 0.0
    prefix_s: float = 0.0
    denoise_s: float = 0.0
    postprocess_s: float = 0.0
    total_s: float = 0.0
    vision_cache_hits: int = 0
    vision_cache_misses: int = 0


@dataclass
class InferenceResult:
    actions: np.ndarray
    """[T, action_dim] actions in environment units; row i is meant for the
    control tick ``obs.tick + i``."""
    budget: ComputeBudget
    timings: StageTimings
    extras: dict[str, Any] = field(default_factory=dict)


class DecisionKind(str, enum.Enum):
    CONTINUE = "continue_chunk"
    RUN_DEFAULT = "run_default"
    RUN_REDUCED = "run_reduced"
    RUN_HIGH = "run_high"
    REFRESH_EARLY = "refresh_early"
    DEFER = "defer"
    HOLD = "hold_fallback"


@dataclass
class Decision:
    """Output of a compute controller for one control tick."""

    kind: DecisionKind
    request: bool
    budget: ComputeBudget | None = None
    horizon: int = 0
    """Target number of actions to execute from a chunk before its refresh lands."""
    phase: str = "default"
    hold: bool = False
    """Abstain: command the safe fallback instead of the buffered plan."""
    invalidate_before_tick: int | None = None
    """Reject any result computed from an observation older than this tick."""
    priority: int = 0
    reasons: list[str] = field(default_factory=list)


@dataclass
class ActionChunk:
    """A policy output with the timing metadata needed to judge freshness."""

    seq: int
    obs_seq: int
    obs_tick: int
    t_obs: float
    t_request: float
    t_start: float
    t_done: float
    actions: np.ndarray
    budget: ComputeBudget
    latency_s: float
    """Request-to-delivery latency in runtime-clock seconds."""
    model_wall_s: float = 0.0
    """Measured wall-clock seconds the backend spent on this chunk."""
    stream_id: str = "robot0"
    anchor_tick: int | None = None
    """Tick that action 0 is aligned to. Normally the observation tick; moved
    later by the number of ticks the robot stood still (fallback) while this
    chunk was being computed, because no planned motion was consumed then."""

    def __post_init__(self) -> None:
        if self.anchor_tick is None:
            self.anchor_tick = self.obs_tick

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    def index_for_tick(self, tick: int) -> int:
        return tick - int(self.anchor_tick)

    def remaining_at(self, tick: int) -> int:
        """Number of not-yet-executed actions if the next action is taken at ``tick``."""
        return max(0, len(self) - max(0, self.index_for_tick(tick)))


class ActionSource(str, enum.Enum):
    CHUNK = "chunk"
    FALLBACK_STARTUP = "fallback_startup"
    FALLBACK_STARVED = "fallback_starved"
    FALLBACK_EXPIRED = "fallback_expired"
    FALLBACK_INVALID = "fallback_invalid"
    FALLBACK_HOLD = "fallback_hold"


@dataclass
class ControlOutput:
    """What the fast path returns for one control tick."""

    action: np.ndarray
    source: ActionSource
    tick: int
    chunk_seq: int | None = None
    chunk_age_ticks: int | None = None
    clipped: bool = False
    rate_limited: bool = False
    decision: Decision | None = None
