"""Developer SDK: the short path from a model id to a running runtime.

    from kinetixrt import Runtime

    runtime = Runtime.from_pretrained("HuggingFaceVLA/smolvla_libero",
                                      adaptive_compute=True, backend="auto")
    runtime.start()
    out = runtime.step(images={"image": rgb, "image2": wrist}, robot_state=state)
    env.step(out.action)

``Runtime`` is the core runtime class plus constructors. The class has no
Python-only constructs in its public surface (plain arrays and dataclasses),
which keeps a future C/C++ binding straightforward.
"""

from __future__ import annotations

from typing import Any

from kinetixrt.backends.base import PolicyBackend
from kinetixrt.core.runtime import Runtime as _CoreRuntime
from kinetixrt.core.runtime import RuntimeConfig, hold_fallback
from kinetixrt.intelligence.analyzer import AnalyzerConfig, PhysicalEnvironmentAnalyzer
from kinetixrt.intelligence.controller import (
    ComputeController,
    FixedController,
    PhysicalAwareController,
    make_controller,
)
from kinetixrt.presets import PRESETS, libero_physical_config

SYNTHETIC_ID = "synthetic/toy-pickplace-controller"


class Runtime(_CoreRuntime):
    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        adaptive_compute: bool | str | dict | ComputeController = True,
        backend: str = "auto",
        mode: str = "threaded",
        control_hz: float = 20.0,
        vision_cache: bool = False,
        **runtime_kwargs: Any,
    ) -> Runtime:
        """Load a supported policy and wrap it in a KinetixRT runtime.

        ``adaptive_compute``: True (physical-aware controller), False (fixed
        default budget), a preset name from ``kinetixrt.presets.PRESETS``, a
        controller spec dict, or a controller instance.
        ``backend``: "auto" | "cuda" | "cpu" | "mps" (device for real models).
        """
        be = cls._make_backend(model_id, backend, vision_cache)
        if isinstance(adaptive_compute, ComputeController):
            controller = adaptive_compute
        elif isinstance(adaptive_compute, dict):
            controller = make_controller(adaptive_compute)
        elif isinstance(adaptive_compute, str):
            controller = make_controller(PRESETS[adaptive_compute])
        elif adaptive_compute:
            controller = PhysicalAwareController(libero_physical_config())
        else:
            controller = FixedController(be.info.default_steps, horizon=10)
        real = be.info.is_real_model
        cfg = RuntimeConfig(
            control_dt=1.0 / control_hz,
            mode=mode,
            fallback=hold_fallback((6,) if real else (2,)),
            disagreement_dims=(0, 1, 2) if real else (0, 1),
            **runtime_kwargs,
        )
        analyzer = PhysicalEnvironmentAnalyzer(
            AnalyzerConfig(event_dims=(6,) if real else (2,), motion_dims=(0, 1, 2) if real else (0, 1))
        )
        return cls(be, controller, cfg, analyzer)

    @staticmethod
    def _make_backend(model_id: str, backend: str, vision_cache: bool) -> PolicyBackend:
        if model_id == SYNTHETIC_ID:
            from kinetixrt.backends.synthetic import SyntheticBackend

            return SyntheticBackend()
        from kinetixrt.backends.smolvla import SmolVLABackend
        from kinetixrt.intelligence.temporal_cache import VisionFeatureCache

        return SmolVLABackend(model_id, device=backend, vision_cache=VisionFeatureCache() if vision_cache else None)
