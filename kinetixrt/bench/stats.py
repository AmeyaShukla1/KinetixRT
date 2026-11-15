"""Small, dependency-free statistics helpers for benchmark reports."""

from __future__ import annotations

import math

import numpy as np


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pair counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_ratio_ci(x: np.ndarray, base: np.ndarray, n_boot: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    """Ratio of sums (x / base) over paired episodes with a bootstrap 95% CI."""
    x, base = np.asarray(x, dtype=float), np.asarray(base, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    ratios = x[idx].sum(axis=1) / np.maximum(base[idx].sum(axis=1), 1e-12)
    return float(x.sum() / max(base.sum(), 1e-12)), float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3:
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0 or sy == 0:
        return float("nan")
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy))


def _rank(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(len(a))
    for v in np.unique(a):  # average ties
        m = a == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return ranks
