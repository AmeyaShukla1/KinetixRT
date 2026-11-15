"""KinetixRT dual-rate runtime.

Fast path  : :meth:`Runtime.step` - called once per control tick. It never
             waits for the model. It analyses the observation, asks the
             compute controller for a decision, optionally submits an
             inference request and returns a validated action from the buffer
             (or a defined fallback).
Slow path  : policy inference, driven either by a worker thread
             (``mode="threaded"``, wall clock) or by a simulator-locked
             virtual-time executor (``mode="virtual"``) that really runs the
             model but charges its latency on the runtime clock.

Both modes share every component (scheduler, buffer, controller, validator,
telemetry), so closed-loop results in virtual time exercise the same logic
that runs with real threads.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from kinetixrt.backends.base import PolicyBackend
from kinetixrt.control.safety import ActionValidator, ValidatorConfig
from kinetixrt.core.action_buffer import ActionBuffer
from kinetixrt.core.clock import VirtualClock, WallClock
from kinetixrt.core.latency import LatencyModel, ZeroLatency
from kinetixrt.core.scheduler import InferenceRequest, InferenceScheduler, LatencyPredictor, SchedulingPolicy
from kinetixrt.core.telemetry import Telemetry
from kinetixrt.intelligence.analyzer import AnalyzerConfig, PhysicalEnvironmentAnalyzer
from kinetixrt.intelligence.controller import ComputeController, ControlContext
from kinetixrt.types import (
    ActionChunk,
    ActionSource,
    ControlOutput,
    Decision,
    InferenceResult,
    Observation,
)


def hold_fallback(hold_dims: tuple[int, ...] = ()) -> Callable[[np.ndarray | None, int], np.ndarray]:
    """Fallback for delta-action robots: zero motion, keep ``hold_dims`` (e.g.
    the gripper command) at their last commanded value."""

    def fb(last: np.ndarray | None, action_dim: int) -> np.ndarray:
        a = np.zeros(action_dim, dtype=np.float32)
        for d in hold_dims:
            a[d] = last[d] if last is not None else -1.0
        return a

    return fb


@dataclass
class RuntimeConfig:
    control_dt: float = 0.05
    mode: str = "virtual"
    scheduler_policy: str = "deadline"
    queue_capacity: int = 2
    max_chunk_age_ticks: int | None = None
    blend_ticks: int = 0
    disagreement_dims: tuple[int, ...] | None = None
    max_obs_age_s: float | None = None
    scheduler_degrade: bool = False
    """Let the scheduler lower the step count of a request that cannot meet
    its deadline. Off by default: single-robot controllers already fit the
    budget themselves, and fixed baselines must stay fixed."""
    action_low: float = -1.0
    action_high: float = 1.0
    max_delta: float | None = None
    rate_limit_dims: tuple[int, ...] | None = None
    fallback: Callable[[np.ndarray | None, int], np.ndarray] = field(default_factory=hold_fallback)
    latency_model: LatencyModel = field(default_factory=ZeroLatency)
    record_ticks: bool = True


class Runtime:
    def __init__(
        self,
        backend: PolicyBackend,
        controller: ComputeController,
        config: RuntimeConfig | None = None,
        analyzer: PhysicalEnvironmentAnalyzer | None = None,
        telemetry: Telemetry | None = None,
    ) -> None:
        self.backend = backend
        self.controller = controller
        self.cfg = config or RuntimeConfig()
        if self.cfg.mode not in ("virtual", "threaded"):
            raise ValueError("mode must be 'virtual' or 'threaded'")
        self.analyzer = analyzer or PhysicalEnvironmentAnalyzer(AnalyzerConfig())
        self.telemetry = telemetry or Telemetry()
        self.clock = VirtualClock() if self.cfg.mode == "virtual" else WallClock()
        self.predictor = LatencyPredictor()
        self.scheduler = InferenceScheduler(
            SchedulingPolicy(self.cfg.scheduler_policy),
            capacity=self.cfg.queue_capacity,
            predictor=self.predictor,
            step_options=backend.info.supported_steps,
            max_obs_age_s=self.cfg.max_obs_age_s,
            allow_degrade=self.cfg.scheduler_degrade,
        )
        self.buffer = ActionBuffer(self.cfg.blend_ticks, disagreement_dims=self.cfg.disagreement_dims)
        max_age = self.cfg.max_chunk_age_ticks or 2 * backend.info.chunk_size
        self.validator = ActionValidator(
            ValidatorConfig(
                low=self.cfg.action_low,
                high=self.cfg.action_high,
                max_delta=self.cfg.max_delta,
                max_chunk_age_ticks=max_age,
                rate_limit_dims=self.cfg.rate_limit_dims,
            )
        )
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._running = False
        self._epoch = 0  # bumped by stop(): results of an inference that outlives it are discarded
        self._task = ""
        self.telemetry.meta.update(
            backend=vars(backend.info).copy(),
            controller=controller.describe(),
            mode=self.cfg.mode,
            control_dt=self.cfg.control_dt,
            scheduler=self.cfg.scheduler_policy,
            latency_model=getattr(self.cfg.latency_model, "name", "n/a") if self.cfg.mode == "virtual" else "wall-clock",
        )
        self._reset_state()

    # -------------------------------------------------------------- lifecycle
    def _reset_state(self) -> None:
        self._tick = 0
        self._obs_seq = 0
        self._req_seq = 0
        self._chunk_seq = 0
        self._in_flight: InferenceRequest | None = None
        self._virtual_job: tuple[InferenceRequest, InferenceResult, float, float] | None = None
        self._last_action: np.ndarray | None = None
        self._disagreement: float | None = None
        self._installed_obs_ticks: list[int] = []
        self._moved_ticks: deque[int] = deque(maxlen=1024)
        self._inline_model_s = 0.0

    def set_controller(self, controller: ComputeController) -> None:
        """Hot-swap the compute policy (used by the dashboard)."""
        with self._lock:
            controller.reset()
            self.controller = controller
            self.telemetry.meta["controller"] = controller.describe()
        self.telemetry.event("controller_changed", name=controller.name, tick=self._tick)

    def start(self) -> None:
        if self.cfg.mode == "threaded" and not self._running:
            self.scheduler.reopen()
            self._running = True
            self._worker = threading.Thread(target=self._worker_loop, args=(self._epoch,), name="krt-inference", daemon=True)
            self._worker.start()

    def stop(self, timeout: float = 30.0) -> None:
        if self.cfg.mode == "threaded" and self._running:
            self._running = False
            with self._lock:
                self._epoch += 1
                self._in_flight = None
            self.scheduler.close()
            if self._worker is not None:
                self._worker.join(timeout)
                if self._worker.is_alive():
                    # An inference is still running. Its thread exits by itself and its
                    # result is dropped (epoch check); it must never reach a later episode.
                    self.telemetry.event("worker_stop_timeout", tick=self._tick)
            self._worker = None

    def reset(self, seed: int | None = None, task: str = "") -> None:
        """Start a new episode. Safe to call between episodes in either mode."""
        was_running = self._running
        self.stop()
        self.scheduler.clear()
        self.buffer.clear()
        self.analyzer.reset()
        self.validator.reset()
        self.controller.reset()
        self.backend.reset(seed)
        if isinstance(self.clock, VirtualClock):
            self.clock = VirtualClock()
        self._task = task
        self._reset_state()
        if was_running:
            self.start()

    @property
    def tick(self) -> int:
        return self._tick

    # ---------------------------------------------------------------- fast path
    def step(
        self,
        observation: Observation | None = None,
        *,
        images: dict[str, np.ndarray] | None = None,
        robot_state: np.ndarray | None = None,
        task: str | None = None,
    ) -> ControlOutput:
        """One control tick. Never blocks on model inference in threaded mode."""
        t_step0 = time.perf_counter()
        tick = self._tick
        now = self.clock.now()
        if observation is None:
            if images is None or robot_state is None:
                raise ValueError("pass an Observation or images= and robot_state=")
            observation = Observation(
                seq=self._obs_seq,
                tick=tick,
                t_capture=now,
                images=images,
                state=np.asarray(robot_state),
                task=self._task if task is None else task,
            )
        self._obs_seq += 1
        if self.cfg.mode == "virtual":
            self._virtual_deliver(now, tick)

        with self._lock:
            installed, self._installed_obs_ticks = self._installed_obs_ticks, []
        for obs_tick in installed:
            self.analyzer.note_plan_installed(obs_tick)
        self._inline_model_s = 0.0
        plan = self.buffer.plan_from(tick)
        signals = self.analyzer.update(observation, plan)
        status = self.buffer.status(tick)
        with self._lock:
            in_flight = self._in_flight
            controller = self.controller
        ctx = ControlContext(
            tick=tick,
            now=now,
            dt=self.cfg.control_dt,
            status=status,
            in_flight=in_flight is not None or self.scheduler.depth() > 0,
            in_flight_obs_tick=in_flight.obs.tick if in_flight is not None else None,
            queue_depth=self.scheduler.depth(),
            signals=signals,
            predictor=self.predictor,
            chunk_size=self.backend.info.chunk_size,
            disagreement=self._disagreement,
        )
        decision = controller.decide(ctx)
        if decision.invalidate_before_tick is not None:
            self.buffer.invalidate_before(decision.invalidate_before_tick)
            self.telemetry.event("plan_invalidated", tick=tick, reasons=decision.reasons)
        if decision.request and decision.budget is not None:
            self._submit(observation, decision, status, now)
        if self.cfg.mode == "virtual":
            self._virtual_pump(now, tick)

        out = self._select_action(tick, decision)
        if out.source is ActionSource.CHUNK:
            with self._lock:
                self._moved_ticks.append(tick)
        self.analyzer.note_executed(out.action, tick, out.source is ActionSource.CHUNK)
        self._last_action = out.action
        if self.cfg.record_ticks:
            st = self.buffer.status(tick)
            self.telemetry.tick(
                {
                    "tick": tick,
                    "t": now,
                    "source": out.source.value,
                    "chunk_seq": out.chunk_seq,
                    "chunk_age": out.chunk_age_ticks,
                    "remaining": st.remaining,
                    "phase": decision.phase,
                    "decision": decision.kind.value,
                    "request": decision.request,
                    "steps": decision.budget.num_steps if decision.request and decision.budget else None,
                    "horizon": decision.horizon,
                    "reasons": decision.reasons,
                    "surprise": signals.scene_surprise,
                    "img_step": signals.img_step,
                    "state_step": signals.state_step,
                    "ticks_to_event": signals.ticks_to_event,
                    "planned_speed": signals.planned_speed,
                    "disagreement": self._disagreement,
                    "queue": self.scheduler.depth(),
                    # fast-path overhead only: inline model time of virtual mode excluded
                    "step_wall_ms": (time.perf_counter() - t_step0 - self._inline_model_s) * 1e3,
                }
            )
        self._tick = tick + 1
        if isinstance(self.clock, VirtualClock):
            self.clock.set(self._tick * self.cfg.control_dt)
        out.decision = decision
        return out

    def _submit(self, obs: Observation, decision: Decision, status, now: float) -> None:
        deadline = now + max(0, status.remaining - 1) * self.cfg.control_dt if status.has_chunk else now
        req = InferenceRequest(
            req_id=self._req_seq,
            obs=obs,
            budget=decision.budget,
            t_submit=now,
            deadline=deadline,
            priority=decision.priority,
            horizon=decision.horizon,
            reasons=list(decision.reasons),
        )
        self._req_seq += 1
        for d in self.scheduler.submit(req):
            self.telemetry.event("request_dropped", reason=d.reason, req_id=d.request.req_id, tick=self._tick)

    def _select_action(self, tick: int, decision: Decision) -> ControlOutput:
        adim = self.backend.info.action_dim
        if decision.hold:
            return self._fallback(tick, ActionSource.FALLBACK_HOLD, None)
        action, chunk = self.buffer.get(tick)
        if chunk is None:
            return self._fallback(tick, ActionSource.FALLBACK_STARTUP, None)
        if action is None:
            self.telemetry.count("deadline_miss_ticks")
            return self._fallback(tick, ActionSource.FALLBACK_STARVED, chunk)
        age = tick - chunk.obs_tick
        res = self.validator.validate(action, age, adim)
        if not res.ok or res.action is None:
            src = ActionSource.FALLBACK_EXPIRED if res.reason == "expired" else ActionSource.FALLBACK_INVALID
            self.telemetry.event("action_rejected", reason=res.reason, tick=tick, chunk_seq=chunk.seq)
            return self._fallback(tick, src, chunk)
        return ControlOutput(res.action, ActionSource.CHUNK, tick, chunk.seq, age, res.clipped, res.rate_limited)

    def _fallback(self, tick: int, source: ActionSource, chunk: ActionChunk | None) -> ControlOutput:
        a = self.cfg.fallback(self._last_action, self.backend.info.action_dim)
        self.validator.note_fallback(a)
        self.telemetry.count(source.value)
        age = tick - chunk.obs_tick if chunk is not None else None
        return ControlOutput(a, source, tick, chunk.seq if chunk else None, age)

    # ---------------------------------------------------------------- slow path
    def _install(self, req: InferenceRequest, result: InferenceResult, t_start: float, t_done: float, wall_s: float, now_tick: int) -> None:
        latency = t_done - req.t_submit
        self.predictor.observe(result.budget.num_steps, t_done - t_start)
        with self._lock:
            moved = sum(1 for t in self._moved_ticks if req.obs.tick <= t < now_tick)
        chunk = ActionChunk(
            seq=self._chunk_seq,
            obs_seq=req.obs.seq,
            obs_tick=req.obs.tick,
            t_obs=req.obs.t_capture,
            t_request=req.t_submit,
            t_start=t_start,
            t_done=t_done,
            actions=np.asarray(result.actions, dtype=np.float32),
            budget=result.budget,
            latency_s=latency,
            model_wall_s=wall_s,
            stream_id=req.obs.stream_id,
            anchor_tick=now_tick - moved,
        )
        self._chunk_seq += 1
        if not np.all(np.isfinite(chunk.actions)):
            self.telemetry.event("invalid_result", tick=now_tick, chunk_seq=chunk.seq)
        report = self.buffer.install(chunk, now_tick)
        if report.accepted:
            with self._lock:
                self._installed_obs_ticks.append(chunk.obs_tick)
            if report.disagreement is not None:
                self._disagreement = report.disagreement
        else:
            self.telemetry.event("stale_result_rejected", reason=report.reason, tick=now_tick, obs_tick=chunk.obs_tick)
        t = result.timings
        self.telemetry.inference(
            {
                "chunk_seq": chunk.seq,
                "obs_tick": chunk.obs_tick,
                "deliver_tick": now_tick,
                "t_submit": req.t_submit,
                "t_start": t_start,
                "t_done": t_done,
                "queue_wait_s": t_start - req.t_submit,
                "latency_s": latency,
                "exec_s": t_done - t_start,
                "model_wall_s": wall_s,
                "steps": result.budget.num_steps,
                "label": result.budget.label,
                "horizon": req.horizon,
                "degraded_from": req.degraded_from,
                "accepted": report.accepted,
                "reject_reason": None if report.accepted else report.reason,
                "skipped_actions": report.skipped_actions,
                "reanchored_ticks": int(chunk.anchor_tick) - chunk.obs_tick,
                "disagreement": report.disagreement,
                "vision_s": t.vision_s,
                "prefix_s": t.prefix_s,
                "denoise_s": t.denoise_s,
                "pre_s": t.preprocess_s,
                "post_s": t.postprocess_s,
                "cache_hits": t.vision_cache_hits,
                "cache_misses": t.vision_cache_misses,
                "reasons": req.reasons,
            }
        )

    # virtual-time executor ----------------------------------------------------
    def _virtual_deliver(self, now: float, tick: int) -> None:
        job = self._virtual_job
        if job is not None and job[3] <= now + 1e-9:
            req, result, t_start, t_done = job
            self._virtual_job = None
            with self._lock:
                self._in_flight = None
            self._install(req, result, t_start, t_done, result.timings.total_s, tick)

    def _virtual_pump(self, now: float, tick: int) -> None:
        while self._virtual_job is None:
            req, dropped = self.scheduler.take(now)
            for d in dropped:
                self.telemetry.event("request_dropped", reason=d.reason, req_id=d.request.req_id, tick=tick)
            if req is None:
                return
            w0 = time.perf_counter()
            try:
                result = self.backend.infer(req.obs, req.budget)
            except Exception as exc:  # noqa: BLE001 - a backend error must not kill the control loop
                self._inline_model_s += time.perf_counter() - w0
                self.telemetry.event("backend_error", tick=tick, error=repr(exc))
                continue
            wall = time.perf_counter() - w0
            self._inline_model_s += wall
            result.timings.total_s = result.timings.total_s or wall
            latency = float(self.cfg.latency_model.latency(result, wall, now))
            with self._lock:
                self._in_flight = req
            self._virtual_job = (req, result, now, now + latency)
            if latency <= 1e-12:
                self._virtual_deliver(now, tick)

    # threaded executor ----------------------------------------------------------
    def _worker_loop(self, epoch: int) -> None:
        while self._running and self._epoch == epoch:
            req, dropped = self.scheduler.take_blocking(self.clock.now, timeout=0.05)
            for d in dropped:
                self.telemetry.event("request_dropped", reason=d.reason, req_id=d.request.req_id, tick=self._tick)
            if req is None:
                continue
            with self._lock:
                if self._epoch != epoch:
                    return
                self._in_flight = req
            t_start = self.clock.now()
            try:
                result = self.backend.infer(req.obs, req.budget)
            except Exception as exc:  # noqa: BLE001
                self.telemetry.event("backend_error", tick=self._tick, error=repr(exc))
                with self._lock:
                    if self._epoch == epoch:
                        self._in_flight = None
                continue
            t_done = self.clock.now()
            if self._epoch != epoch:
                return  # stopped or reset while the model was running: discard
            self._install(req, result, t_start, t_done, t_done - t_start, self._tick)
            with self._lock:
                if self._epoch == epoch:
                    self._in_flight = None

    # ------------------------------------------------------------------ summary
    def summary(self) -> dict:
        tel = self.telemetry
        infs = list(tel.inferences)
        ticks = list(tel.ticks)
        acc = [i for i in infs if i["accepted"]]
        src = {}
        for t in ticks:
            src[t["source"]] = src.get(t["source"], 0) + 1
        ages = [t["chunk_age"] for t in ticks if t["source"] == "chunk" and t["chunk_age"] is not None]
        return {
            "ticks": len(ticks),
            "inferences": len(infs),
            "accepted_chunks": len(acc),
            "denoise_steps_total": int(sum(i["steps"] for i in infs)),
            "mean_steps": float(np.mean([i["steps"] for i in infs])) if infs else None,
            "model_wall_s_total": float(sum(i["model_wall_s"] for i in infs)),
            "latency_s": [i["latency_s"] for i in infs],
            "exec_s": [i["exec_s"] for i in infs],
            "queue_wait_s": [i["queue_wait_s"] for i in infs],
            "cache_hits": int(sum(i["cache_hits"] for i in infs)),
            "cache_misses": int(sum(i["cache_misses"] for i in infs)),
            "sources": src,
            "deadline_miss_ticks": int(tel.counters.get("deadline_miss_ticks", 0)),
            "startup_wait_ticks": int(tel.counters.get(ActionSource.FALLBACK_STARTUP.value, 0)),
            "hold_ticks": int(tel.counters.get(ActionSource.FALLBACK_HOLD.value, 0)),
            "stale_rejected": int(tel.counters.get("stale_result_rejected", 0)),
            "requests_dropped": int(tel.counters.get("request_dropped", 0)),
            "backend_errors": int(tel.counters.get("backend_error", 0)),
            "plan_invalidations": int(tel.counters.get("plan_invalidated", 0)),
            "mean_action_age_ticks": float(np.mean(ages)) if ages else None,
            "max_action_age_ticks": int(max(ages)) if ages else None,
            "phase_ticks": _count(ticks, "phase"),
            "validator": dict(self.validator.counts),
            "scheduler": dict(self.scheduler.stats),
            "fast_path_ms": [t["step_wall_ms"] for t in ticks],
        }


def _count(rows: list[dict], key: str) -> dict:
    out: dict = {}
    for r in rows:
        out[r[key]] = out.get(r[key], 0) + 1
    return out
