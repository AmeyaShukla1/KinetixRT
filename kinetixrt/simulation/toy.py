"""Lightweight CPU-only closed-loop environment for runtime tests and demos.

A 2-D pick-and-place: a point end-effector with a gripper must grasp a red
object and release it on a green goal. Observations are a rendered RGB image
plus proprioception, so the same Observation -> Runtime -> action path is used
as with a real simulator. This is a test fixture, not a robotics benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ToyConfig:
    size: int = 64
    max_step: float = 0.03
    grasp_radius: float = 0.04
    goal_radius: float = 0.05
    max_steps: int = 200
    move_object_at: int | None = None
    """Tick at which the object is abruptly displaced (dynamic-scene scenario)."""
    move_object_by: tuple[float, float] = (0.25, -0.2)


class ToyPickPlace:
    action_dim = 3

    def __init__(self, cfg: ToyConfig | None = None, seed: int = 0) -> None:
        self.cfg = cfg or ToyConfig()
        self.rng = np.random.default_rng(seed)
        self.reset(seed)

    def reset(self, seed: int | None = None) -> dict:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        r = self.rng
        self.pos = r.uniform(0.1, 0.3, size=2).astype(np.float32)
        self.obj = r.uniform(0.35, 0.65, size=2).astype(np.float32)
        self.goal = r.uniform(0.7, 0.9, size=2).astype(np.float32)
        self.grip = 0.0
        self.holding = False
        self.t = 0
        self.success = False
        return self._obs()

    def _blob(self, img: np.ndarray, p: np.ndarray, color: tuple[int, int, int], rad: int) -> None:
        s = self.cfg.size
        cx, cy = int(round(p[0] * (s - 1))), int(round(p[1] * (s - 1)))
        y0, y1, x0, x1 = max(0, cy - rad), min(s, cy + rad + 1), max(0, cx - rad), min(s, cx + rad + 1)
        img[y0:y1, x0:x1] = color

    def render(self) -> np.ndarray:
        s = self.cfg.size
        img = np.full((s, s, 3), 30, dtype=np.uint8)
        self._blob(img, self.goal, (20, 220, 20), 4)
        self._blob(img, self.obj, (220, 20, 20), 3)
        self._blob(img, self.pos, (60, 60, 230), 1)
        return img

    def _obs(self) -> dict:
        return {
            "images": {"image": self.render()},
            "state": np.array([self.pos[0], self.pos[1], self.grip], dtype=np.float32),
        }

    def step(self, action: np.ndarray) -> tuple[dict, bool, bool]:
        c = self.cfg
        a = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
        self.pos = np.clip(self.pos + a[:2] * c.max_step, 0.0, 1.0)
        want_closed = a[2] > 0
        if want_closed and not self.holding and self.grip < 0.5:
            if np.linalg.norm(self.pos - self.obj) < c.grasp_radius:
                self.holding = True
        if not want_closed and self.holding:
            self.holding = False
        self.grip = 1.0 if want_closed else 0.0
        if self.holding:
            self.obj = self.pos.copy()
        self.t += 1
        if c.move_object_at is not None and self.t == c.move_object_at and not self.holding:
            self.obj = np.clip(self.obj + np.asarray(c.move_object_by, dtype=np.float32), 0.05, 0.95)
        self.success = bool(not self.holding and np.linalg.norm(self.obj - self.goal) < c.goal_radius)
        done = self.success or self.t >= c.max_steps
        return self._obs(), self.success, done
