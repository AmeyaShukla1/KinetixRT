"""Named controller configurations used by the SDK, the CLI and the benchmarks."""

from __future__ import annotations

from kinetixrt.intelligence.controller import PhysicalPolicyConfig


def libero_physical_config(**overrides) -> PhysicalPolicyConfig:
    """Physical-aware rule set for SmolVLA on LIBERO (20 Hz delta-EEF control).

    Thresholds were calibrated with ``krt-bench signals`` on initial states
    that are not used for evaluation; see BENCHMARKS.md for the evidence.
    """
    base = dict(
        step_options=(3, 5, 10),
        steps_calm=5,
        steps_default=10,
        steps_critical=10,
        horizon_calm=25,
        horizon_default=10,
        horizon_critical=10,
        event_window_pre=12,
        event_window_post=8,
        surprise_threshold=0.04,
        surprise_action="hold",
    )
    base.update(overrides)
    return PhysicalPolicyConfig(**base)


def _phys(**overrides) -> dict:
    cfg = libero_physical_config(**overrides)
    return {"type": "physical_aware", **{k: getattr(cfg, k) for k in cfg.__dataclass_fields__}}


PRESETS: dict[str, dict] = {
    "fixed_high": {"type": "fixed", "num_steps": 10, "horizon": 10},
    "fixed_reduced_steps": {"type": "fixed", "num_steps": 3, "horizon": 10},
    "fixed_long_horizon": {"type": "fixed", "num_steps": 10, "horizon": 50},
    # Fixed baseline with roughly the same inference count as physical_aware.
    "fixed_matched": {"type": "fixed", "num_steps": 10, "horizon": 17},
    # The checkpoint's own setting (n_action_steps=1): one inference per control tick.
    "reference_replan": {"type": "fixed", "num_steps": 10, "horizon": 1},
    # Refresh-rate frontier between the reference and fixed_high.
    "fixed_h2": {"type": "fixed", "num_steps": 10, "horizon": 2},
    "fixed_h3": {"type": "fixed", "num_steps": 10, "horizon": 3},
    "fixed_h5": {"type": "fixed", "num_steps": 10, "horizon": 5},
    "runtime_only": {"type": "runtime_only", "step_options": (3, 5, 10), "default_steps": 10, "horizon": 10},
    # Saves compute in calm phases, keeps the fixed_high budget near actuator events.
    "physical_aware": _phys(),
    # Same, but refreshes twice as often near actuator events (spends more where it may matter).
    "physical_precise": _phys(horizon_critical=5),
    # Phase-dependent refresh near the reference setting: replan every tick (or every
    # 2nd tick) near gripper events, less often in calm phases.
    "physical_h1": _phys(horizon_critical=1, horizon_default=1, horizon_calm=5, steps_calm=5),
    "physical_h2": _phys(horizon_critical=2, horizon_default=2, horizon_calm=10, steps_calm=5),
    # Control for physical_h1: the same rule with the phases swapped (dense refresh in
    # calm phases, sparse near events). If this does as well, the phase signal is not what helps.
    "physical_h1_inverted": _phys(horizon_critical=5, horizon_default=5, horizon_calm=1, steps_calm=10),
    # Ablations of the physical-aware rule set.
    "physical_no_phase": _phys(use_phase=False),
    "physical_no_scene": _phys(use_scene=False),
    "physical_steps_only": _phys(horizon_calm=10),
    "physical_horizon_only": _phys(steps_calm=10),
}
