"""Portable execution interface.

The adaptive scheduler, controllers and drivers only talk to this interface,
so a new accelerator/back-end (TensorRT, ROCm, Jetson, a native runtime) is a
new ``PolicyBackend`` subclass and nothing else changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from kinetixrt.types import ComputeBudget, InferenceResult, Observation


@dataclass
class BackendInfo:
    name: str
    model_id: str
    device: str
    chunk_size: int
    action_dim: int
    default_steps: int
    supported_steps: tuple[int, ...]
    """Step counts the adapter has validated for this model."""
    is_real_model: bool
    """False for synthetic test backends; their numbers must never be reported
    as model results."""
    extra: dict[str, Any] = field(default_factory=dict)


class PolicyBackend(ABC):
    info: BackendInfo

    @abstractmethod
    def infer(self, obs: Observation, budget: ComputeBudget) -> InferenceResult:
        """Run one action-chunk inference. Must be safe to call from a single
        worker thread while the control loop runs in another."""

    def reset(self, seed: int | None = None) -> None:  # noqa: B027
        """Start a new episode (reseed sampling noise, clear caches)."""

    def close(self) -> None:  # noqa: B027
        pass
