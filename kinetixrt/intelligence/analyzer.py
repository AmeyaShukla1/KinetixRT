"""Physical environment analyzer: cheap, measured signals about the scene,
the robot and the active plan. Pure numpy; runs on the fast path every tick.

None of these signals uses simulator ground truth. They are *proxies*; their
relation to prediction error is validated by ``krt-bench signals`` rather than
assumed (see BENCHMARKS.md).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from kinetixrt.types import Observation


@dataclass
class AnalyzerConfig:
    grid: int = 36
    """Images are area-averaged to grid x grid grayscale before differencing."""
    cells: int = 6
    """The grid is split into cells x cells blocks for localized change."""
    scene_camera: str | None = None
    """Camera used for scene-surprise detection (ideally a static, external
    camera). Defaults to the first camera of the observation."""
    activity_alpha: float = 0.5
    activity_gain: float = 3.0
    event_dims: tuple[int, ...] = ()
    """Action dimensions that behave like discrete actuators (e.g. a gripper
    command whose sign flips between open and close)."""
    motion_dims: tuple[int, ...] = (0, 1, 2)
    """Action dimensions whose norm measures commanded translational speed."""
    speed_window: int = 8
    history: int = 256
    onset_guard: int = 3
    """Scene surprise is suppressed for this many ticks after the robot starts
    moving from rest: the onset of its own motion is "novel" image change too."""


@dataclass
class PhysicalSignals:
    tick: int = 0
    img_step: dict[str, float] = field(default_factory=dict)
    """Mean abs grayscale change vs the previous tick, per camera (0..1)."""
    img_since_plan: dict[str, float] = field(default_factory=dict)
    """Mean abs change vs the frame the active plan was computed from."""
    scene_surprise: float = 0.0
    """Largest localized change on the scene camera that is *not* explained by
    recent activity in the same image region (0..1). Spikes on abrupt scene
    discontinuities, stays low for smooth robot motion."""
    state_step: float = 0.0
    state_since_plan: float = 0.0
    ticks_to_event: int | None = None
    """Ticks until the active plan flips a discrete actuator (e.g. gripper)."""
    ticks_since_event: int | None = None
    planned_speed: float | None = None
    """Mean commanded translational magnitude over the next few plan steps."""

    def as_dict(self) -> dict:
        return {
            "img_step": self.img_step,
            "img_since_plan": self.img_since_plan,
            "scene_surprise": self.scene_surprise,
            "state_step": self.state_step,
            "state_since_plan": self.state_since_plan,
            "ticks_to_event": self.ticks_to_event,
            "ticks_since_event": self.ticks_since_event,
            "planned_speed": self.planned_speed,
        }


def downsample_gray(img: np.ndarray, grid: int) -> np.ndarray:
    """Area-average an HxWx3 (or HxW) uint8 image to grid x grid floats in [0, 1]."""
    a = np.asarray(img)
    if a.ndim == 3:
        a = a.mean(axis=2, dtype=np.float32)
    else:
        a = a.astype(np.float32)
    h, w = a.shape
    fh, fw = max(1, h // grid), max(1, w // grid)
    gh, gw = min(grid, h // fh), min(grid, w // fw)
    a = a[: gh * fh, : gw * fw].reshape(gh, fh, gw, fw).mean(axis=(1, 3))
    return a / 255.0


def _dilate3(x: np.ndarray) -> np.ndarray:
    p = np.pad(x, 1, mode="edge")
    out = x.copy()
    for dy in range(3):
        for dx in range(3):
            np.maximum(out, p[dy : dy + x.shape[0], dx : dx + x.shape[1]], out=out)
    return out


class PhysicalEnvironmentAnalyzer:
    def __init__(self, cfg: AnalyzerConfig | None = None) -> None:
        self.cfg = cfg or AnalyzerConfig()
        self.reset()

    def reset(self) -> None:
        self._prev_gray: dict[str, np.ndarray] = {}
        self._prev_state: np.ndarray | None = None
        self._activity: np.ndarray | None = None
        self._hist: OrderedDict[int, tuple[dict[str, np.ndarray], np.ndarray]] = OrderedDict()
        self._plan_ref: tuple[dict[str, np.ndarray], np.ndarray] | None = None
        self._last_exec: np.ndarray | None = None
        self._last_event_tick: int | None = None
        self._motion_streak = 0
        self.last = PhysicalSignals()

    # ------------------------------------------------------------ notifications
    def note_plan_installed(self, obs_tick: int) -> None:
        """A new plan computed from the observation at ``obs_tick`` became active."""
        ref = self._hist.get(obs_tick)
        if ref is not None:
            self._plan_ref = ref

    def note_executed(self, action: np.ndarray, tick: int, from_plan: bool) -> None:
        """Record the command actually sent, to detect executed actuator events."""
        self._motion_streak = self._motion_streak + 1 if from_plan else 0
        if not from_plan:
            return
        if self._last_exec is not None:
            for d in self.cfg.event_dims:
                if np.sign(action[d]) != np.sign(self._last_exec[d]) and np.sign(action[d]) != 0:
                    self._last_event_tick = tick
        self._last_exec = np.asarray(action, dtype=np.float32).copy()

    # ------------------------------------------------------------------ update
    def update(self, obs: Observation, plan: np.ndarray | None) -> PhysicalSignals:
        cfg = self.cfg
        s = PhysicalSignals(tick=obs.tick)
        grays = {name: downsample_gray(img, cfg.grid) for name, img in obs.images.items()}
        scene_cam = cfg.scene_camera or (next(iter(grays)) if grays else None)
        for name, g in grays.items():
            prev = self._prev_gray.get(name)
            if prev is not None and prev.shape == g.shape:
                diff = np.abs(g - prev)
                s.img_step[name] = float(diff.mean())
                if name == scene_cam:
                    s.scene_surprise = self._surprise(diff)
            else:
                s.img_step[name] = 0.0
            if self._plan_ref is not None and name in self._plan_ref[0]:
                s.img_since_plan[name] = float(np.abs(g - self._plan_ref[0][name]).mean())
        state = np.asarray(obs.state, dtype=np.float32).reshape(-1)
        if self._prev_state is not None and self._prev_state.shape == state.shape:
            s.state_step = float(np.linalg.norm(state - self._prev_state))
        if self._plan_ref is not None and self._plan_ref[1].shape == state.shape:
            s.state_since_plan = float(np.linalg.norm(state - self._plan_ref[1]))
        self._plan_signals(s, plan, obs.tick)
        self._prev_gray = grays
        self._prev_state = state
        self._hist[obs.tick] = (grays, state)
        while len(self._hist) > cfg.history:
            self._hist.popitem(last=False)
        self.last = s
        return s

    def _surprise(self, diff: np.ndarray) -> float:
        cfg = self.cfg
        c = cfg.cells
        gh, gw = diff.shape
        bh, bw = gh // c, gw // c
        cell = diff[: bh * c, : bw * c].reshape(c, bh, c, bw).mean(axis=(1, 3))
        if self._activity is None or self._activity.shape != cell.shape:
            self._activity = cell.copy()
            return 0.0
        novelty = cell - cfg.activity_gain * _dilate3(self._activity)
        self._activity = (1 - cfg.activity_alpha) * self._activity + cfg.activity_alpha * cell
        if 0 < self._motion_streak <= cfg.onset_guard:
            return 0.0  # the robot just started moving: its own motion onset, not the scene
        return float(max(0.0, novelty.max()))

    def _plan_signals(self, s: PhysicalSignals, plan: np.ndarray | None, tick: int) -> None:
        cfg = self.cfg
        if self._last_event_tick is not None:
            s.ticks_since_event = tick - self._last_event_tick
        if plan is None or len(plan) == 0:
            return
        if cfg.motion_dims:
            w = plan[: cfg.speed_window, list(cfg.motion_dims)]
            s.planned_speed = float(np.linalg.norm(w, axis=1).mean())
        for d in cfg.event_dims:
            ref = self._last_exec[d] if self._last_exec is not None else plan[0, d]
            flips = np.nonzero(np.sign(plan[:, d]) != np.sign(ref))[0]
            if flips.size:
                k = int(flips[0])
                s.ticks_to_event = k if s.ticks_to_event is None else min(s.ticks_to_event, k)
