"""Thread-safe action-chunk buffer shared by the slow and the fast path.

Ownership and concurrency semantics
-----------------------------------
* The slow path (inference worker) is the only writer of chunks: it calls
  :meth:`install`. Installed chunks are never mutated afterwards.
* The fast path (control loop) is the only consumer: it calls :meth:`get`.
* Both sides take one short, non-blocking-in-practice mutex; no call ever
  waits for model inference. The buffer is bounded: it holds exactly one
  active chunk (plus the previous one for blending / disagreement metrics).
* Tick alignment: action ``i`` of a chunk is valid for control tick
  ``anchor_tick + i``. The anchor is the observation tick, so a chunk that
  arrives ``d`` ticks late has its first ``d`` actions skipped - except for
  ticks during which the robot was holding still in fallback (start-up,
  starvation, abstention): no planned motion was consumed then, so the
  runtime moves the anchor forward by that many ticks.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from kinetixrt.types import ActionChunk


@dataclass
class InstallReport:
    accepted: bool
    reason: str
    skipped_actions: int = 0
    disagreement: float | None = None
    """Mean L2 distance between the new chunk and the plan it replaces over the
    ticks both cover (a measured temporal-instability signal)."""
    gap_ticks: int = 0


@dataclass
class BufferStatus:
    has_chunk: bool
    chunk_seq: int | None
    obs_tick: int | None
    executed: int
    """Plan steps already consumed (ticks since the chunk's anchor tick)."""
    remaining: int
    """Valid actions left, counting the one for the current tick."""
    chunk_len: int


class ActionBuffer:
    def __init__(
        self,
        blend_ticks: int = 0,
        disagreement_window: int = 10,
        disagreement_dims: tuple[int, ...] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._active: ActionChunk | None = None
        self._previous: ActionChunk | None = None
        self._installed_tick: int = 0
        self._min_obs_tick: int = -(10**12)
        self.blend_ticks = int(blend_ticks)
        self.disagreement_window = int(disagreement_window)
        self.disagreement_dims = disagreement_dims
        self.accepted = 0
        self.rejected_stale = 0

    # ------------------------------------------------------------------ writer
    def install(self, chunk: ActionChunk, now_tick: int) -> InstallReport:
        """Install a freshly computed chunk, or reject it if it is stale.

        A chunk is stale when (a) it was computed from an observation that is
        not newer than the active chunk's, (b) its observation predates an
        explicit invalidation point, or (c) every action it contains is already
        in the past.
        """
        with self._lock:
            if chunk.obs_tick < self._min_obs_tick:
                self.rejected_stale += 1
                return InstallReport(False, "invalidated_observation")
            if self._active is not None and chunk.obs_tick <= self._active.obs_tick:
                self.rejected_stale += 1
                return InstallReport(False, "superseded_by_newer_chunk")
            skipped = max(0, chunk.index_for_tick(now_tick))
            if skipped >= len(chunk):
                self.rejected_stale += 1
                return InstallReport(False, "all_actions_expired", skipped_actions=skipped)
            disagreement = self._disagreement(self._active, chunk, now_tick)
            self._previous = self._active
            self._active = chunk
            self._installed_tick = now_tick
            self.accepted += 1
            return InstallReport(True, "accepted", skipped_actions=skipped, disagreement=disagreement)

    def invalidate_before(self, obs_tick: int) -> None:
        """Reject future chunks computed from observations older than ``obs_tick``."""
        with self._lock:
            self._min_obs_tick = max(self._min_obs_tick, int(obs_tick))

    def clear(self) -> None:
        with self._lock:
            self._active = None
            self._previous = None
            self._min_obs_tick = -(10**12)

    # ---------------------------------------------------------------- consumer
    def get(self, tick: int) -> tuple[np.ndarray | None, ActionChunk | None]:
        """Action for ``tick`` (a copy) or ``(None, chunk)`` if the plan ran out."""
        with self._lock:
            chunk = self._active
            if chunk is None:
                return None, None
            idx = chunk.index_for_tick(tick)
            if idx < 0 or idx >= len(chunk):
                return None, chunk
            action = chunk.actions[idx].astype(np.float32, copy=True)
            if self.blend_ticks > 0 and self._previous is not None:
                k = tick - self._installed_tick
                pidx = self._previous.index_for_tick(tick)
                if 0 <= k < self.blend_ticks and 0 <= pidx < len(self._previous):
                    w = (k + 1) / (self.blend_ticks + 1)
                    action = (1 - w) * self._previous.actions[pidx] + w * action
            return action, chunk

    def status(self, tick: int) -> BufferStatus:
        with self._lock:
            c = self._active
            if c is None:
                return BufferStatus(False, None, None, 0, 0, 0)
            return BufferStatus(True, c.seq, c.obs_tick, max(0, c.index_for_tick(tick)), c.remaining_at(tick), len(c))

    def plan_from(self, tick: int) -> np.ndarray | None:
        """Not-yet-executed part of the active plan starting at ``tick`` (copy)."""
        with self._lock:
            c = self._active
            if c is None:
                return None
            idx = max(0, c.index_for_tick(tick))
            return c.actions[idx:].copy()

    # ----------------------------------------------------------------- helpers
    def _disagreement(self, old: ActionChunk | None, new: ActionChunk, now_tick: int) -> float | None:
        if old is None:
            return None
        a0 = max(0, old.index_for_tick(now_tick))
        b0 = max(0, new.index_for_tick(now_tick))
        n = min(len(old) - a0, len(new) - b0, self.disagreement_window)
        if n <= 0:
            return None
        a, b = old.actions[a0 : a0 + n], new.actions[b0 : b0 + n]
        if self.disagreement_dims is not None:
            dims = list(self.disagreement_dims)
            a, b = a[:, dims], b[:, dims]
        return float(np.linalg.norm(a - b, axis=1).mean())
