"""Adaptive compute controllers.

All controllers are deterministic and explainable: each decision carries the
list of rules that fired. Three families are provided so that the value of
physical-world awareness can be separated from plain latency awareness:

* :class:`FixedController`        - fixed step count and execution horizon.
* :class:`RuntimeOnlyController`  - adapts the step count to measured latency
  and deadline slack only (no knowledge of the scene or task phase).
* :class:`PhysicalAwareController` - additionally uses measured scene, robot
  and plan signals to decide *when* to refresh and *how much* to compute.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from kinetixrt.core.action_buffer import BufferStatus
from kinetixrt.core.scheduler import LatencyPredictor
from kinetixrt.intelligence.analyzer import PhysicalSignals
from kinetixrt.types import ComputeBudget, Decision, DecisionKind


@dataclass
class ControlContext:
    tick: int
    now: float
    dt: float
    status: BufferStatus
    in_flight: bool
    in_flight_obs_tick: int | None
    queue_depth: int
    signals: PhysicalSignals
    predictor: LatencyPredictor
    chunk_size: int
    disagreement: float | None = None
    """Disagreement measured when the active chunk replaced the previous one."""
    load_factor: float = 1.0


class ComputeController(ABC):
    name: str = "controller"

    def reset(self) -> None:  # noqa: B027
        pass

    @abstractmethod
    def decide(self, ctx: ControlContext) -> Decision: ...

    def describe(self) -> dict:
        return {"name": self.name}

    # shared helper ---------------------------------------------------------
    @staticmethod
    def lead_ticks(ctx: ControlContext, num_steps: int, pessimism: float = 1.0) -> int:
        """Ticks before the target refresh point at which to start inference."""
        lat = ctx.predictor.predict(num_steps, pessimism=pessimism)
        return int(math.ceil(max(0.0, lat) / ctx.dt - 1e-9))


def _kind_for(num_steps: int, default_steps: int) -> DecisionKind:
    if num_steps < default_steps:
        return DecisionKind.RUN_REDUCED
    if num_steps > default_steps:
        return DecisionKind.RUN_HIGH
    return DecisionKind.RUN_DEFAULT


class FixedController(ComputeController):
    """Baseline: constant denoising steps, refresh every ``horizon`` actions."""

    def __init__(self, num_steps: int, horizon: int, reuse_vision: bool = False, name: str | None = None) -> None:
        self.num_steps, self.horizon, self.reuse_vision = num_steps, horizon, reuse_vision
        self.name = name or f"fixed(n={num_steps},H={horizon})"

    def describe(self) -> dict:
        return {"name": self.name, "num_steps": self.num_steps, "horizon": self.horizon}

    def decide(self, ctx: ControlContext) -> Decision:
        budget = ComputeBudget(self.num_steps, self.reuse_vision, "fixed")
        if ctx.in_flight:
            return Decision(DecisionKind.CONTINUE, False, horizon=self.horizon, reasons=["inference in flight"])
        if not ctx.status.has_chunk:
            return Decision(DecisionKind.RUN_DEFAULT, True, budget, self.horizon, reasons=["no plan available"])
        lead = self.lead_ticks(ctx, self.num_steps)
        if ctx.status.executed + lead >= self.horizon or ctx.status.remaining <= lead + 1:
            return Decision(
                DecisionKind.RUN_DEFAULT,
                True,
                budget,
                self.horizon,
                reasons=[f"fixed cadence: plan age {ctx.status.executed} + lead {lead} >= H {self.horizon}"],
            )
        return Decision(DecisionKind.CONTINUE, False, horizon=self.horizon, reasons=["within fixed horizon"])


class RuntimeOnlyController(ComputeController):
    """Latency/deadline-aware budget selection without physical-world signals.

    Keeps a fixed target horizon and picks the largest step count whose
    predicted latency fits both the refresh period and the remaining plan.
    """

    def __init__(
        self,
        step_options: tuple[int, ...] = (3, 5, 10),
        default_steps: int = 10,
        horizon: int = 10,
        reuse_vision: bool = False,
        pessimism: float = 1.0,
    ) -> None:
        self.step_options = tuple(sorted(step_options))
        self.default_steps, self.horizon, self.reuse_vision = default_steps, horizon, reuse_vision
        self.pessimism = pessimism
        self.name = f"runtime_only(H={horizon})"

    def describe(self) -> dict:
        return {"name": self.name, "step_options": self.step_options, "horizon": self.horizon}

    def _fit(self, ctx: ControlContext, want: int, slack_s: float) -> tuple[int, str | None]:
        if ctx.predictor.predict(want, self.pessimism) <= slack_s:
            return want, None
        fitting = [s for s in self.step_options if s <= want and ctx.predictor.predict(s, self.pessimism) <= slack_s]
        chosen = max(fitting) if fitting else min(self.step_options)
        if chosen == want:
            return want, None
        return chosen, (
            f"latency: steps {want}->{chosen} (predicted {ctx.predictor.predict(want, self.pessimism) * 1e3:.0f} ms"
            f" > slack {slack_s * 1e3:.0f} ms)"
        )

    def decide(self, ctx: ControlContext) -> Decision:
        if ctx.in_flight:
            return Decision(DecisionKind.CONTINUE, False, horizon=self.horizon, reasons=["inference in flight"])
        period_s = self.horizon * ctx.dt
        if not ctx.status.has_chunk:
            n, why = self._fit(ctx, self.default_steps, period_s)
            return Decision(
                _kind_for(n, self.default_steps),
                True,
                ComputeBudget(n, self.reuse_vision, "runtime"),
                self.horizon,
                reasons=["no plan available"] + ([why] if why else []),
            )
        slack_s = min(period_s, max(0, ctx.status.remaining - 1) * ctx.dt)
        n, why = self._fit(ctx, self.default_steps, slack_s)
        lead = self.lead_ticks(ctx, n, self.pessimism)
        if ctx.status.executed + lead >= self.horizon or ctx.status.remaining <= lead + 1:
            return Decision(
                _kind_for(n, self.default_steps),
                True,
                ComputeBudget(n, self.reuse_vision, "runtime"),
                self.horizon,
                reasons=[f"cadence: plan age {ctx.status.executed} + lead {lead} >= H {self.horizon}"]
                + ([why] if why else []),
            )
        return Decision(DecisionKind.CONTINUE, False, horizon=self.horizon, reasons=["within horizon"])


@dataclass
class PhysicalPolicyConfig:
    """Thresholds of the physical-aware rule set (see docs/ARCHITECTURE.md).

    Defaults for LIBERO/SmolVLA were calibrated on held-out initial states
    with ``krt-bench signals``; they are not universal constants.
    """

    step_options: tuple[int, ...] = (3, 5, 10)
    steps_calm: int = 5
    steps_default: int = 10
    steps_critical: int = 10
    horizon_calm: int = 25
    horizon_default: int = 10
    horizon_critical: int = 5
    event_window_pre: int = 12
    """Plan-predicted actuator event within this many ticks => critical phase."""
    event_window_post: int = 8
    """This many ticks after an executed actuator event stay critical."""
    slow_speed: float | None = None
    """Planned translational speed below this => precision (critical) motion."""
    fast_speed: float | None = None
    """Planned speed above this (and nothing else firing) => calm transit."""
    disagreement_high: float | None = None
    """Chunk-to-chunk disagreement above this => uncertain => critical."""
    disagreement_low: float | None = None
    """Calm phase additionally requires disagreement below this, if set."""
    surprise_threshold: float | None = None
    """scene_surprise above this => abrupt scene change => refresh now."""
    surprise_action: str = "hold"
    """'hold' (abstain until a fresh plan arrives) or 'continue'."""
    surprise_cooldown: int = 5
    state_divergence: float | None = None
    """Proprio divergence since the plan's observation above this => refresh."""
    reuse_vision: bool = False
    fit_deadline: bool = True
    pessimism: float = 1.0
    use_phase: bool = True
    use_scene: bool = True
    use_uncertainty: bool = True


class PhysicalAwareController(ComputeController):
    """Physical-world-aware compute allocation.

    Per tick it classifies the situation into ``critical`` / ``default`` /
    ``calm`` from measured signals, maps the phase to a (steps, horizon)
    budget, and applies the same deadline-fitting rule as the runtime-only
    controller on top.
    """

    def __init__(self, cfg: PhysicalPolicyConfig | None = None) -> None:
        self.cfg = cfg or PhysicalPolicyConfig()
        self.name = "physical_aware"
        self.reset()

    def reset(self) -> None:
        self._last_surprise_tick: int | None = None
        self._hold_until_obs_tick: int | None = None
        self._cooldown_until: int = -1

    def describe(self) -> dict:
        return {"name": self.name, **{k: getattr(self.cfg, k) for k in self.cfg.__dataclass_fields__}}

    # -------------------------------------------------------------- phase logic
    def classify(self, ctx: ControlContext) -> tuple[str, list[str]]:
        c, s = self.cfg, ctx.signals
        critical: list[str] = []
        if c.use_phase:
            if s.ticks_to_event is not None and s.ticks_to_event <= c.event_window_pre:
                critical.append(f"actuator event planned in {s.ticks_to_event} ticks")
            if s.ticks_since_event is not None and s.ticks_since_event <= c.event_window_post:
                critical.append(f"actuator event {s.ticks_since_event} ticks ago")
            if c.slow_speed is not None and s.planned_speed is not None and s.planned_speed < c.slow_speed:
                critical.append(f"slow planned motion {s.planned_speed:.3f} < {c.slow_speed}")
        if c.use_uncertainty and c.disagreement_high is not None and ctx.disagreement is not None:
            if ctx.disagreement > c.disagreement_high:
                critical.append(f"chunk disagreement {ctx.disagreement:.3f} > {c.disagreement_high}")
        if c.use_scene and c.state_divergence is not None and s.state_since_plan > c.state_divergence:
            critical.append(f"state diverged {s.state_since_plan:.3f} since plan")
        if critical:
            return "critical", critical
        # "calm" needs positive evidence from at least one enabled signal family.
        blocked = False
        why: list[str] = []
        if c.use_phase:
            if c.fast_speed is not None:
                if s.planned_speed is None or s.planned_speed < c.fast_speed:
                    blocked = True
                else:
                    why.append(f"transit speed {s.planned_speed:.3f}")
            else:
                why.append("no actuator event near")
        if c.use_uncertainty and c.disagreement_low is not None:
            if ctx.disagreement is None or ctx.disagreement > c.disagreement_low:
                blocked = True
            else:
                why.append(f"disagreement {ctx.disagreement:.3f} low")
        if why and not blocked:
            return "calm", why
        return "default", ["no critical signal, not clearly calm"]

    def _fit(self, ctx: ControlContext, want: int, slack_s: float) -> tuple[int, str | None]:
        c = self.cfg
        if not c.fit_deadline or ctx.predictor.predict(want, c.pessimism) <= slack_s:
            return want, None
        fitting = [s for s in c.step_options if s <= want and ctx.predictor.predict(s, c.pessimism) <= slack_s]
        chosen = max(fitting) if fitting else min(c.step_options)
        if chosen >= want:
            return want, None
        return chosen, f"deadline: steps {want}->{chosen} to fit slack {slack_s * 1e3:.0f} ms"

    def decide(self, ctx: ControlContext) -> Decision:
        c, s = self.cfg, ctx.signals
        phase, reasons = self.classify(ctx)
        steps = {"critical": c.steps_critical, "default": c.steps_default, "calm": c.steps_calm}[phase]
        horizon = {"critical": c.horizon_critical, "default": c.horizon_default, "calm": c.horizon_calm}[phase]
        reasons = [f"phase={phase}: " + "; ".join(reasons)]

        # --- abrupt scene change: invalidate the plan and refresh immediately ---
        surprised = (
            c.use_scene
            and c.surprise_threshold is not None
            and s.scene_surprise > c.surprise_threshold
            and ctx.tick > self._cooldown_until
        )
        if surprised:
            self._cooldown_until = ctx.tick + c.surprise_cooldown
            self._hold_until_obs_tick = ctx.tick
            n, why = self._fit(ctx, c.steps_critical, max(0, ctx.status.remaining - 1) * ctx.dt)
            return Decision(
                DecisionKind.REFRESH_EARLY,
                True,
                ComputeBudget(n, False, "surprise"),
                c.horizon_critical,
                phase="surprise",
                hold=c.surprise_action == "hold",
                invalidate_before_tick=ctx.tick,
                priority=1,
                reasons=[f"scene surprise {s.scene_surprise:.3f} > {c.surprise_threshold}: plan invalidated"]
                + ([why] if why else []),
            )
        hold = False
        if self._hold_until_obs_tick is not None:
            if ctx.status.has_chunk and ctx.status.obs_tick is not None and ctx.status.obs_tick >= self._hold_until_obs_tick:
                self._hold_until_obs_tick = None
            else:
                hold = c.surprise_action == "hold"
                reasons.append("holding until post-surprise plan arrives")

        if ctx.in_flight:
            return Decision(DecisionKind.CONTINUE, False, horizon=horizon, phase=phase, hold=hold, reasons=reasons + ["inference in flight"])
        if not ctx.status.has_chunk:
            n, why = self._fit(ctx, c.steps_default, horizon * ctx.dt)
            return Decision(
                _kind_for(n, c.steps_default),
                True,
                ComputeBudget(n, False, phase),
                horizon,
                phase=phase,
                hold=hold,
                reasons=["no plan available"] + ([why] if why else []),
            )
        slack_s = max(0, ctx.status.remaining - 1) * ctx.dt
        n, why = self._fit(ctx, steps, slack_s)
        lead = self.lead_ticks(ctx, n, c.pessimism)
        if ctx.status.executed + lead >= horizon or ctx.status.remaining <= lead + 1:
            kind = DecisionKind.REFRESH_EARLY if phase == "critical" and horizon < c.horizon_default else _kind_for(n, c.steps_default)
            return Decision(
                kind,
                True,
                ComputeBudget(n, c.reuse_vision and phase != "critical", phase),
                horizon,
                phase=phase,
                hold=hold,
                reasons=reasons
                + [f"plan age {ctx.status.executed} + lead {lead} >= H {horizon} -> steps={n}"]
                + ([why] if why else []),
            )
        return Decision(DecisionKind.CONTINUE, False, horizon=horizon, phase=phase, hold=hold, reasons=reasons + [f"plan age {ctx.status.executed} < H {horizon}"])


def make_controller(spec: dict) -> ComputeController:
    """Build a controller from a plain dict (used by configs and the CLI)."""
    spec = dict(spec)
    kind = spec.pop("type")
    if kind == "fixed":
        return FixedController(**spec)
    if kind == "runtime_only":
        if "step_options" in spec:
            spec["step_options"] = tuple(spec["step_options"])
        return RuntimeOnlyController(**spec)
    if kind == "physical_aware":
        if "step_options" in spec:
            spec["step_options"] = tuple(spec["step_options"])
        return PhysicalAwareController(PhysicalPolicyConfig(**spec))
    raise ValueError(f"unknown controller type: {kind}")
