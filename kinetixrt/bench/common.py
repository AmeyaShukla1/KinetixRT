"""Shared builders for benchmark commands."""

from __future__ import annotations

import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from kinetixrt.core.latency import CalibratedLatency, Disturbance, DisturbedLatency, LatencyModel, MeasuredLatency, ZeroLatency
from kinetixrt.core.runtime import Runtime, RuntimeConfig, hold_fallback
from kinetixrt.core.telemetry import Telemetry
from kinetixrt.intelligence.analyzer import AnalyzerConfig, PhysicalEnvironmentAnalyzer
from kinetixrt.intelligence.controller import ComputeController, make_controller
from kinetixrt.presets import PRESETS

RESULTS = Path("results")

# Calibrated on this repository's CPU measurement run (see BENCHMARKS.md,
# "CPU latency calibration"): latency ~= base + per_step * num_steps.
CPU_CALIBRATION = {"base_s": 0.61, "per_step_s": 0.071, "vision_s": 0.19}


def libero_analyzer() -> PhysicalEnvironmentAnalyzer:
    return PhysicalEnvironmentAnalyzer(AnalyzerConfig(scene_camera="image", event_dims=(6,), motion_dims=(0, 1, 2)))


def make_latency_model(kind: str, scale: float = 1.0, calibration: dict | None = None, disturbances: list[Disturbance] | None = None) -> LatencyModel:
    if kind == "zero":
        base: LatencyModel = ZeroLatency()
    elif kind == "measured":
        base = MeasuredLatency(scale=scale)
    elif kind == "calibrated":
        base = CalibratedLatency(**(calibration or CPU_CALIBRATION), scale=scale)
    else:
        raise ValueError(f"unknown latency model: {kind}")
    return DisturbedLatency(base, disturbances, name=f"{base.name}+disturbed") if disturbances else base


def controller_from_name(name: str, overrides: dict | None = None) -> ComputeController:
    spec = dict(PRESETS[name])
    spec.update(overrides or {})
    return make_controller(spec)


def make_libero_runtime(backend: Any, controller: ComputeController, latency: LatencyModel, control_dt: float = 0.05, telemetry: Telemetry | None = None, **cfg_kwargs: Any) -> Runtime:
    cfg = RuntimeConfig(
        control_dt=control_dt,
        mode="virtual",
        fallback=hold_fallback((6,)),
        disagreement_dims=(0, 1, 2),
        latency_model=latency,
        **cfg_kwargs,
    )
    return Runtime(backend, controller, cfg, libero_analyzer(), telemetry)


def hardware_info() -> dict:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except ImportError:
        info["torch"] = None
    try:
        import lerobot

        info["lerobot"] = lerobot.__version__
    except ImportError:
        pass
    try:
        info["git_commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        pass
    return info


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_default) + "\n")


def _default(o: Any) -> Any:
    import numpy as np

    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "value"):
        return o.value
    return str(o)
