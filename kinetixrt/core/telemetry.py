"""Structured runtime telemetry: bounded in-memory rings plus optional JSONL.

Every number shown by the dashboard or written to benchmark results comes from
records created here by the runtime; nothing is synthesized for display.
"""

from __future__ import annotations

import json
import threading
from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np


def _jsonable(v: Any) -> Any:
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if hasattr(v, "value") and not isinstance(v, (int, float, str, bool)):
        return v.value
    return v


class Telemetry:
    def __init__(self, maxlen: int = 20000, jsonl_path: str | Path | None = None) -> None:
        self._lock = threading.Lock()
        self.ticks: deque[dict] = deque(maxlen=maxlen)
        self.inferences: deque[dict] = deque(maxlen=maxlen)
        self.events: deque[dict] = deque(maxlen=maxlen)
        self.counters: Counter[str] = Counter()
        self.meta: dict[str, Any] = {}
        self._fh = open(jsonl_path, "a") if jsonl_path else None  # noqa: SIM115

    def _write(self, kind: str, rec: dict) -> None:
        if self._fh is not None:
            self._fh.write(json.dumps({"kind": kind, **_jsonable(rec)}) + "\n")

    def tick(self, rec: dict) -> None:
        with self._lock:
            self.ticks.append(rec)
            self._write("tick", rec)

    def inference(self, rec: dict) -> None:
        with self._lock:
            self.inferences.append(rec)
            self._write("inference", rec)

    def event(self, kind: str, **fields: Any) -> None:
        with self._lock:
            rec = {"event": kind, **fields}
            self.events.append(rec)
            self.counters[kind] += 1
            self._write("event", rec)

    def count(self, key: str, n: int = 1) -> None:
        with self._lock:
            self.counters[key] += n

    def reset(self) -> None:
        with self._lock:
            self.ticks.clear()
            self.inferences.clear()
            self.events.clear()
            self.counters.clear()

    def snapshot(self, n_ticks: int = 400, n_inferences: int = 60, n_events: int = 40) -> dict:
        with self._lock:
            return _jsonable(
                {
                    "meta": self.meta,
                    "counters": dict(self.counters),
                    "ticks": list(self.ticks)[-n_ticks:],
                    "inferences": list(self.inferences)[-n_inferences:],
                    "events": list(self.events)[-n_events:],
                }
            )

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None


def percentile(values: list[float] | np.ndarray, q: float) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return None
    return float(np.percentile(arr, q))


def latency_summary(values: list[float] | np.ndarray) -> dict[str, float | int | None]:
    """p50/p95/p99 with the sample size; p99 is only reported for n >= 100."""
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    out: dict[str, float | int | None] = {"n": n}
    if n == 0:
        return {**out, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    out.update(
        mean=float(arr.mean()),
        p50=float(np.percentile(arr, 50)),
        p95=float(np.percentile(arr, 95)) if n >= 20 else None,
        p99=float(np.percentile(arr, 99)) if n >= 100 else None,
        max=float(arr.max()),
    )
    return out
