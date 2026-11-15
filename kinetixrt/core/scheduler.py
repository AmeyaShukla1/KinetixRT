"""Bounded, deadline-aware inference request scheduler (slow-path admission).

Timing claims are best-effort: this orders and sheds work, it does not provide
hard real-time guarantees.
"""

from __future__ import annotations

import enum
import math
import threading
from dataclasses import dataclass, field, replace

from kinetixrt.types import ComputeBudget, Observation


class LatencyPredictor:
    """Online per-budget latency estimate (EWMA mean + EW mean-abs-deviation).

    For step counts never observed it fits ``latency = a + b * num_steps`` to
    the observed means, which matches the structure of flow-matching policies
    (one fixed prefix pass plus ``num_steps`` expert passes).
    """

    def __init__(self, alpha: float = 0.3, prior: dict[int, float] | None = None) -> None:
        self.alpha = alpha
        self._mean: dict[int, float] = dict(prior or {})
        self._dev: dict[int, float] = {k: 0.0 for k in self._mean}
        self._n: dict[int, int] = {k: 0 for k in self._mean}
        self._lock = threading.Lock()

    def observe(self, num_steps: int, latency_s: float) -> None:
        with self._lock:
            if self._n.get(num_steps, 0) == 0:
                self._mean[num_steps] = latency_s
                self._dev[num_steps] = 0.0
            else:
                err = latency_s - self._mean[num_steps]
                self._mean[num_steps] += self.alpha * err
                self._dev[num_steps] += self.alpha * (abs(err) - self._dev[num_steps])
            self._n[num_steps] = self._n.get(num_steps, 0) + 1

    def predict(self, num_steps: int, pessimism: float = 0.0) -> float:
        """Predicted latency in seconds; ``pessimism`` adds that many deviations."""
        with self._lock:
            if num_steps in self._mean:
                return self._mean[num_steps] + pessimism * self._dev.get(num_steps, 0.0)
            if not self._mean:
                return 0.0
            pts = sorted(self._mean.items())
            if len(pts) == 1:
                return pts[0][1]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            sxx = sum((x - mx) ** 2 for x in xs)
            b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx > 0 else 0.0
            dev = max(self._dev.values()) if self._dev else 0.0
            return max(0.0, my + b * (num_steps - mx)) + pessimism * dev

    def snapshot(self) -> dict[int, float]:
        with self._lock:
            return dict(self._mean)


class SchedulingPolicy(str, enum.Enum):
    FIFO = "fifo"
    """Serve in arrival order; when full, reject the newest request."""
    LATEST = "latest"
    """Per stream keep only the newest observation (older ones are superseded)."""
    DEADLINE = "deadline"
    """LATEST + earliest-deadline-first across streams, obsolete-observation
    shedding and budget degradation to fit the remaining slack."""


@dataclass
class InferenceRequest:
    req_id: int
    obs: Observation
    budget: ComputeBudget
    t_submit: float
    deadline: float
    """Clock time at which the consumer runs out of valid actions."""
    priority: int = 0
    horizon: int = 0
    reasons: list[str] = field(default_factory=list)
    degraded_from: int | None = None


@dataclass
class Dropped:
    request: InferenceRequest
    reason: str


class InferenceScheduler:
    def __init__(
        self,
        policy: SchedulingPolicy | str = SchedulingPolicy.DEADLINE,
        capacity: int = 4,
        predictor: LatencyPredictor | None = None,
        step_options: tuple[int, ...] = (),
        max_obs_age_s: float | None = None,
        allow_degrade: bool = True,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.policy = SchedulingPolicy(policy)
        self.capacity = capacity
        self.predictor = predictor or LatencyPredictor()
        self.step_options = tuple(sorted(step_options))
        self.max_obs_age_s = max_obs_age_s
        self.allow_degrade = allow_degrade
        self._q: list[InferenceRequest] = []
        self._cv = threading.Condition()
        self._closed = False
        self.stats = {
            "submitted": 0,
            "served": 0,
            "rejected_full": 0,
            "superseded": 0,
            "dropped_obsolete": 0,
            "degraded": 0,
            "max_depth": 0,
        }

    # ---------------------------------------------------------------- producer
    def submit(self, req: InferenceRequest) -> list[Dropped]:
        dropped: list[Dropped] = []
        with self._cv:
            if self._closed:
                return [Dropped(req, "scheduler_closed")]
            self.stats["submitted"] += 1
            if self.policy in (SchedulingPolicy.LATEST, SchedulingPolicy.DEADLINE):
                keep = []
                for r in self._q:
                    if r.obs.stream_id == req.obs.stream_id:
                        dropped.append(Dropped(r, "superseded"))
                        self.stats["superseded"] += 1
                    else:
                        keep.append(r)
                self._q = keep
            if len(self._q) >= self.capacity:
                if self.policy is SchedulingPolicy.FIFO:
                    self.stats["rejected_full"] += 1
                    dropped.append(Dropped(req, "queue_full"))
                    return dropped
                # Shed the request with the most slack (latest deadline, lowest priority).
                victim = max(self._q, key=lambda r: (-r.priority, r.deadline))
                self._q.remove(victim)
                self.stats["rejected_full"] += 1
                dropped.append(Dropped(victim, "queue_full"))
            self._q.append(req)
            self.stats["max_depth"] = max(self.stats["max_depth"], len(self._q))
            self._cv.notify()
        return dropped

    # ---------------------------------------------------------------- consumer
    def take(self, now: float) -> tuple[InferenceRequest | None, list[Dropped]]:
        """Non-blocking: next request to run according to the policy."""
        with self._cv:
            return self._take_locked(now)

    def take_blocking(self, clock_now, timeout: float) -> tuple[InferenceRequest | None, list[Dropped]]:
        with self._cv:
            if not self._q and not self._closed:
                self._cv.wait(timeout)
            return self._take_locked(clock_now())

    def _take_locked(self, now: float) -> tuple[InferenceRequest | None, list[Dropped]]:
        dropped: list[Dropped] = []
        if self.policy is SchedulingPolicy.DEADLINE and self.max_obs_age_s is not None:
            keep = []
            for r in self._q:
                if now - r.obs.t_capture > self.max_obs_age_s:
                    dropped.append(Dropped(r, "obsolete_observation"))
                    self.stats["dropped_obsolete"] += 1
                else:
                    keep.append(r)
            self._q = keep
        if not self._q:
            return None, dropped
        if self.policy is SchedulingPolicy.DEADLINE:
            req = min(self._q, key=lambda r: (-r.priority, r.deadline, r.t_submit))
            self._q.remove(req)
            req = self._fit_budget(req, now)
        else:
            req = self._q.pop(0)
        self.stats["served"] += 1
        return req, dropped

    def _fit_budget(self, req: InferenceRequest, now: float) -> InferenceRequest:
        """Degrade the step count if the requested budget cannot meet the deadline."""
        if not self.allow_degrade or not self.step_options:
            return req
        slack = req.deadline - now
        if self.predictor.predict(req.budget.num_steps, pessimism=1.0) <= slack:
            return req
        lower = [s for s in self.step_options if s < req.budget.num_steps]
        if not lower:
            return req
        fitting = [s for s in lower if self.predictor.predict(s, pessimism=1.0) <= slack]
        chosen = max(fitting) if fitting else min(lower)
        self.stats["degraded"] += 1
        reasons = req.reasons + [
            f"scheduler: steps {req.budget.num_steps}->{chosen} to fit slack {max(slack, 0.0) * 1e3:.0f} ms"
        ]
        return replace(
            req,
            budget=replace(req.budget, num_steps=chosen, label="degraded"),
            degraded_from=req.budget.num_steps,
            reasons=reasons,
        )

    def depth(self) -> int:
        with self._cv:
            return len(self._q)

    def clear(self) -> list[Dropped]:
        with self._cv:
            dropped = [Dropped(r, "cleared") for r in self._q]
            self._q = []
            return dropped

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    def reopen(self) -> None:
        with self._cv:
            self._closed = False


def latency_ticks(latency_s: float, dt: float) -> int:
    """Whole control ticks a result is delayed by a latency of ``latency_s``."""
    return int(math.ceil(max(0.0, latency_s) / dt - 1e-9))
