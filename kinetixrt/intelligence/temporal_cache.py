"""Cross-observation reuse of per-camera vision features.

Why this form of reuse (and not KV-cache patching): in SmolVLA the proprio
state is part of the VLM prefix and the prefix K/V of every layer depend on
all image tokens, so patching the KV cache for a changed observation is not
exact. The vision tower output for one camera, however, depends on that
camera's pixels only. Reusing it is exact when the frame is unchanged and a
bounded approximation when the frame changed by less than the configured
tolerance. Entries are invalidated explicitly by:

* image change above ``tau_mean`` (mean abs change of the downsampled frame),
* localized change above ``tau_peak`` (max abs change of any grid cell),
* age above ``max_age_ticks``,
* task / instruction change, episode reset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from kinetixrt.intelligence.analyzer import downsample_gray


@dataclass
class CacheEntry:
    gray: np.ndarray
    features: Any
    tick: int
    task: str


class VisionFeatureCache:
    def __init__(self, tau_mean: float = 0.004, tau_peak: float = 0.03, max_age_ticks: int = 60, grid: int = 36) -> None:
        self.tau_mean, self.tau_peak, self.max_age_ticks, self.grid = tau_mean, tau_peak, max_age_ticks, grid
        self._entries: dict[str, CacheEntry] = {}
        self.stats = {"hits": 0, "misses": 0, "invalid_change": 0, "invalid_age": 0, "invalid_task": 0}

    def clear(self) -> None:
        self._entries.clear()

    def lookup(self, camera: str, image: np.ndarray, tick: int, task: str) -> tuple[Any | None, np.ndarray]:
        """Return (features or None, downsampled frame for a later ``store``)."""
        gray = downsample_gray(image, self.grid)
        e = self._entries.get(camera)
        if e is None or e.gray.shape != gray.shape:
            self.stats["misses"] += 1
            return None, gray
        if e.task != task:
            self.stats["invalid_task"] += 1
            self.stats["misses"] += 1
            return None, gray
        if tick - e.tick > self.max_age_ticks:
            self.stats["invalid_age"] += 1
            self.stats["misses"] += 1
            return None, gray
        diff = np.abs(gray - e.gray)
        if float(diff.mean()) > self.tau_mean or float(diff.max()) > self.tau_peak:
            self.stats["invalid_change"] += 1
            self.stats["misses"] += 1
            return None, gray
        self.stats["hits"] += 1
        return e.features, gray

    def store(self, camera: str, gray: np.ndarray, features: Any, tick: int, task: str) -> None:
        self._entries[camera] = CacheEntry(gray, features, tick, task)
