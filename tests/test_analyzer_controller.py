import numpy as np

from kinetixrt.core.action_buffer import BufferStatus
from kinetixrt.core.scheduler import LatencyPredictor
from kinetixrt.intelligence.analyzer import AnalyzerConfig, PhysicalEnvironmentAnalyzer, PhysicalSignals, downsample_gray
from kinetixrt.intelligence.controller import (
    ControlContext,
    FixedController,
    PhysicalAwareController,
    PhysicalPolicyConfig,
    RuntimeOnlyController,
    make_controller,
)
from kinetixrt.types import DecisionKind, Observation


def obs(tick, img, state=None):
    return Observation(tick, tick, tick * 0.05, {"image": img}, np.zeros(3) if state is None else state)


def scene(arm_x, obj=(40, 40)):
    img = np.full((72, 72, 3), 40, dtype=np.uint8)
    img[obj[1] : obj[1] + 8, obj[0] : obj[0] + 8] = 220
    img[10:14, arm_x : arm_x + 6] = 150  # a slowly moving "arm"
    return img


def test_downsample_gray_shape_and_range():
    g = downsample_gray(np.full((360, 360, 3), 255, dtype=np.uint8), 36)
    assert g.shape == (36, 36) and np.allclose(g, 1.0)
    assert downsample_gray(np.zeros((50, 70, 3), dtype=np.uint8), 36).shape[0] <= 36


def test_scene_surprise_ignores_smooth_motion_but_fires_on_abrupt_change():
    an = PhysicalEnvironmentAnalyzer(AnalyzerConfig())
    smooth = [an.update(obs(t, scene(10 + t)), None).scene_surprise for t in range(20)]
    jump = an.update(obs(20, scene(30, obj=(8, 50))), None).scene_surprise  # object teleports
    assert max(smooth[2:]) < 0.02
    assert jump > 0.05 and jump > 5 * max(smooth[2:])


def test_plan_signals_event_and_speed():
    an = PhysicalEnvironmentAnalyzer(AnalyzerConfig(event_dims=(2,), motion_dims=(0, 1)))
    plan = np.zeros((20, 3), dtype=np.float32)
    plan[:, 0] = 0.3
    plan[:, 2] = -1.0
    plan[7:, 2] = 1.0  # gripper closes at plan index 7
    s = an.update(obs(0, scene(10)), plan)
    assert s.ticks_to_event == 7 and abs(s.planned_speed - 0.3) < 1e-6 and s.ticks_since_event is None
    an.note_executed(np.array([0.3, 0, -1.0]), 0, True)
    an.note_executed(np.array([0.3, 0, 1.0]), 1, True)  # executed toggle
    assert an.update(obs(4, scene(14)), plan[4:]).ticks_since_event == 3


def test_state_and_image_divergence_since_plan():
    an = PhysicalEnvironmentAnalyzer(AnalyzerConfig())
    an.update(obs(0, scene(10), np.zeros(3)), None)
    an.note_plan_installed(0)
    s = an.update(obs(1, scene(30), np.array([0.3, 0.4, 0.0])), None)
    assert abs(s.state_since_plan - 0.5) < 1e-6 and s.img_since_plan["image"] > 0


def ctx(executed=0, remaining=50, has=True, in_flight=False, signals=None, predictor=None, disagreement=None, tick=100):
    st = BufferStatus(has, 0 if has else None, tick - executed if has else None, executed, remaining if has else 0, 50 if has else 0)
    return ControlContext(tick, tick * 0.05, 0.05, st, in_flight, None, 0, signals or PhysicalSignals(), predictor or LatencyPredictor(), 50, disagreement)


def test_fixed_controller_cadence_and_lead_time():
    c = FixedController(10, 10)
    assert c.decide(ctx(has=False)).request
    assert not c.decide(ctx(executed=9)).request
    assert c.decide(ctx(executed=10)).request
    assert not c.decide(ctx(executed=10, in_flight=True)).request
    p = LatencyPredictor()
    p.observe(10, 0.2)  # 4 ticks of latency -> start 4 ticks early
    assert not c.decide(ctx(executed=5, predictor=p)).request
    assert c.decide(ctx(executed=6, predictor=p)).request


def test_runtime_only_controller_fits_steps_to_latency():
    p = LatencyPredictor()
    for n, lat in [(10, 1.3), (5, 0.95), (3, 0.8)]:
        p.observe(n, lat)
    d = RuntimeOnlyController((3, 5, 10), 10, horizon=20).decide(ctx(executed=20, predictor=p))
    assert d.request and d.budget.num_steps == 5 and d.kind is DecisionKind.RUN_REDUCED and "latency" in d.reasons[-1]
    fast = LatencyPredictor()
    fast.observe(10, 0.05)
    d = RuntimeOnlyController((3, 5, 10), 10, horizon=20).decide(ctx(executed=20, predictor=fast))
    assert d.budget.num_steps == 10 and d.kind is DecisionKind.RUN_DEFAULT


def test_physical_controller_phases_change_budget_and_horizon():
    cfg = PhysicalPolicyConfig(steps_calm=5, steps_critical=10, horizon_calm=25, horizon_critical=5, event_window_pre=12, surprise_threshold=0.05)
    c = PhysicalAwareController(cfg)
    calm = c.decide(ctx(executed=12, signals=PhysicalSignals(ticks_to_event=None, planned_speed=0.5)))
    assert calm.phase == "calm" and not calm.request and calm.horizon == 25
    due = c.decide(ctx(executed=25, signals=PhysicalSignals(planned_speed=0.5)))
    assert due.request and due.budget.num_steps == 5 and due.kind is DecisionKind.RUN_REDUCED
    crit = c.decide(ctx(executed=6, signals=PhysicalSignals(ticks_to_event=8)))
    assert crit.phase == "critical" and crit.request and crit.budget.num_steps == 10
    assert crit.kind is DecisionKind.REFRESH_EARLY and "actuator event planned in 8 ticks" in crit.reasons[0]
    post = c.decide(ctx(executed=5, signals=PhysicalSignals(ticks_since_event=2)))
    assert post.phase == "critical" and post.request


def test_physical_controller_surprise_invalidates_and_holds_until_fresh_plan():
    c = PhysicalAwareController(PhysicalPolicyConfig(surprise_threshold=0.05, surprise_action="hold"))
    d = c.decide(ctx(executed=3, in_flight=True, signals=PhysicalSignals(scene_surprise=0.2), tick=100))
    assert d.kind is DecisionKind.REFRESH_EARLY and d.request and d.hold and d.invalidate_before_tick == 100 and d.priority == 1
    # still holding while the active plan predates the surprise
    assert c.decide(ctx(executed=4, in_flight=True, tick=101)).hold
    # released once a plan computed at/after the surprise tick is active
    assert not c.decide(ctx(executed=1, tick=103)).hold
    # cooldown: a second spike right after does not re-trigger
    c2 = PhysicalAwareController(PhysicalPolicyConfig(surprise_threshold=0.05, surprise_cooldown=5))
    c2.decide(ctx(signals=PhysicalSignals(scene_surprise=0.2), tick=10))
    assert c2.decide(ctx(executed=0, signals=PhysicalSignals(scene_surprise=0.2), tick=12)).kind is not DecisionKind.REFRESH_EARLY


def test_physical_controller_ablation_flags_and_uncertainty():
    sig = PhysicalSignals(ticks_to_event=3, scene_surprise=0.5)
    off = PhysicalAwareController(PhysicalPolicyConfig(use_phase=False, use_scene=False, surprise_threshold=0.05))
    assert off.decide(ctx(executed=1, signals=sig)).phase == "default"
    unc = PhysicalAwareController(PhysicalPolicyConfig(disagreement_high=0.2))
    assert unc.decide(ctx(executed=1, disagreement=0.5)).phase == "critical"
    assert unc.decide(ctx(executed=1, disagreement=0.1)).phase == "calm"


def test_make_controller_from_presets():
    from kinetixrt.presets import PRESETS

    for name, spec in PRESETS.items():
        c = make_controller(spec)
        assert c.describe()["name"], name
