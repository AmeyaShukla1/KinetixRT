import time

import numpy as np

from kinetixrt.backends.synthetic import SyntheticBackend
from kinetixrt.core.latency import CalibratedLatency, Disturbance, DisturbedLatency
from kinetixrt.intelligence.controller import FixedController, PhysicalAwareController, PhysicalPolicyConfig, RuntimeOnlyController
from kinetixrt.simulation.rollout import run_episode
from kinetixrt.simulation.toy import ToyConfig, ToyPickPlace
from kinetixrt.types import ActionSource


def test_zero_latency_matches_synchronous_protocol(make_runtime, blank_obs):
    rt = make_runtime(FixedController(10, 5))
    outs = [rt.step(**_kw(blank_obs())) for _ in range(12)]
    assert all(o.source is ActionSource.CHUNK for o in outs)
    assert [o.chunk_age_ticks for o in outs] == [0, 1, 2, 3, 4] * 2 + [0, 1]
    infs = list(rt.telemetry.inferences)
    assert [i["obs_tick"] for i in infs] == [0, 5, 10] and all(i["latency_s"] == 0 for i in infs)


def _kw(o):
    return {"images": o["images"], "robot_state": o["state"]}


def test_chunk_timing_metadata_under_latency(make_runtime, blank_obs):
    rt = make_runtime(FixedController(10, 10), latency=CalibratedLatency(0.1, 0.01))  # 0.2 s = 4 ticks
    outs = [rt.step(**_kw(blank_obs())) for _ in range(30)]
    assert [o.source for o in outs[:4]] == [ActionSource.FALLBACK_STARTUP] * 4
    assert outs[4].source is ActionSource.CHUNK and outs[4].chunk_age_ticks == 4
    infs = list(rt.telemetry.inferences)
    first, second = infs[0], infs[1]
    assert first["obs_tick"] == 0 and first["deliver_tick"] == 4 and abs(first["latency_s"] - 0.2) < 1e-9
    assert first["reanchored_ticks"] == 4 and first["skipped_actions"] == 0  # robot stood still: nothing to skip
    assert second["deliver_tick"] - second["obs_tick"] == 4 and second["skipped_actions"] == 4
    assert second["reanchored_ticks"] == 0
    assert rt.summary()["deadline_miss_ticks"] == 0


def test_delayed_inference_causes_starvation_fallback_then_recovers(make_runtime, blank_obs):
    lat = DisturbedLatency(CalibratedLatency(0.05, 0.0), [Disturbance(t0=1.0, t1=1.5, extra_s=3.0)])
    rt = make_runtime(FixedController(10, 10), latency=lat, backend=SyntheticBackend(chunk_size=20))
    outs = [rt.step(**_kw(blank_obs())) for _ in range(140)]
    starved = [o.tick for o in outs if o.source is ActionSource.FALLBACK_STARVED]
    assert starved, "a 3 s delay with a 1 s plan must starve the controller"
    assert all(np.allclose(outs[t].action[:2], 0.0) for t in starved)  # safe fallback: zero motion
    assert outs[-1].source is ActionSource.CHUNK  # recovered
    s = rt.summary()
    assert s["deadline_miss_ticks"] == len(starved)


def test_backend_errors_do_not_kill_the_control_loop(make_runtime, blank_obs):
    rt = make_runtime(FixedController(10, 5), backend=SyntheticBackend(fail_every=2))
    outs = [rt.step(**_kw(blank_obs())) for _ in range(40)]
    assert rt.summary()["backend_errors"] > 0
    assert sum(o.source is ActionSource.CHUNK for o in outs) > 30


def test_invalid_model_output_is_replaced_by_fallback(make_runtime, blank_obs):
    class NaNBackend(SyntheticBackend):
        def infer(self, obs, budget):
            r = super().infer(obs, budget)
            r.actions[3:] = np.nan
            return r

    rt = make_runtime(FixedController(10, 8), backend=NaNBackend())
    outs = [rt.step(**_kw(blank_obs())) for _ in range(8)]
    assert [o.source for o in outs[:3]] == [ActionSource.CHUNK] * 3
    assert all(o.source is ActionSource.FALLBACK_INVALID for o in outs[3:])
    assert all(np.all(np.isfinite(o.action)) for o in outs)
    assert rt.validator.counts["non_finite"] == 5


def test_surprise_rejects_in_flight_stale_result(make_runtime):
    env = ToyPickPlace(ToyConfig(move_object_at=12), seed=3)
    ctrl = PhysicalAwareController(PhysicalPolicyConfig(surprise_threshold=0.02, horizon_calm=25, horizon_default=25, horizon_critical=25, event_window_pre=0, event_window_post=0))
    rt = make_runtime(ctrl, latency=CalibratedLatency(0.25, 0.0))
    r = run_episode(env, rt, "toy", seed=3, reset_kwargs={"seed": 3}, motion_dims=(0, 1))
    s = r.summary
    assert s["plan_invalidations"] >= 1 and s["hold_ticks"] >= 1
    inval = [e["tick"] for e in rt.telemetry.events if e["event"] == "plan_invalidated"]
    assert 12 <= inval[0] <= 14  # detected within two ticks of the scene change
    accepted_after = [i for i in rt.telemetry.inferences if i["accepted"] and i["deliver_tick"] > inval[0]]
    assert accepted_after and all(i["obs_tick"] >= inval[0] for i in accepted_after[:1])


def test_adaptive_controllers_change_actual_compute(make_runtime):
    def total(ctrl, lat=None):
        rt = make_runtime(ctrl, latency=lat)
        steps = inf = succ = 0
        for seed in range(8):
            r = run_episode(ToyPickPlace(ToyConfig(), seed), rt, "toy", seed=seed, reset_kwargs={"seed": seed}, motion_dims=(0, 1))
            steps += r.summary["denoise_steps_total"]
            inf += r.summary["inferences"]
            succ += r.success
        return steps, inf, succ

    fixed = total(FixedController(10, 10))
    phys = total(PhysicalAwareController(PhysicalPolicyConfig(steps_calm=3, horizon_calm=20, event_window_pre=2, event_window_post=1)))
    assert phys[0] < fixed[0], "physical-aware controller must spend fewer denoising steps on the toy task"
    slow = CalibratedLatency(0.1, 0.05)  # 10 steps = 0.6 s > 0.5 s refresh period
    rt_only = make_runtime(RuntimeOnlyController((3, 5, 10), 10, horizon=10), latency=slow)
    r = run_episode(ToyPickPlace(ToyConfig(), 0), rt_only, "toy", seed=0, reset_kwargs={"seed": 0}, motion_dims=(0, 1))
    used = {i["steps"] for i in rt_only.telemetry.inferences}
    assert used - {10}, f"runtime-only controller should reduce steps under latency pressure, used {used}"


def test_threaded_fast_path_does_not_block_on_inference(make_runtime, blank_obs):
    backend = SyntheticBackend(base_latency_s=0.25, sleep=True)  # every inference blocks the worker 250 ms
    rt = make_runtime(FixedController(10, 10), mode="threaded", backend=backend, control_dt=0.01)
    rt.start()
    durations, sources = [], []
    t_end = time.perf_counter() + 1.2
    while time.perf_counter() < t_end:
        t0 = time.perf_counter()
        out = rt.step(**_kw(blank_obs()))
        durations.append(time.perf_counter() - t0)
        sources.append(out.source)
        time.sleep(0.01)
    rt.stop()
    assert max(durations) < 0.05, f"fast path blocked for {max(durations):.3f}s"
    assert ActionSource.CHUNK in sources and sources[0] is ActionSource.FALLBACK_STARTUP
    infs = list(rt.telemetry.inferences)
    assert len(infs) >= 2 and all(i["exec_s"] >= 0.24 for i in infs)
    assert all(i["t_done"] >= i["t_start"] >= i["t_submit"] for i in infs)


def test_threaded_stop_start_and_reset(make_runtime, blank_obs):
    rt = make_runtime(FixedController(10, 5), mode="threaded", backend=SyntheticBackend(base_latency_s=0.01, sleep=True), control_dt=0.01)
    for _ in range(2):
        rt.start()
        for _ in range(20):
            rt.step(**_kw(blank_obs()))
            time.sleep(0.005)
        assert len(rt.telemetry.inferences) >= 1
        rt.reset(seed=1)
        assert rt.tick == 0 and not rt.buffer.status(0).has_chunk
        rt.stop()
    assert rt._worker is None


def test_hot_swap_controller(make_runtime, blank_obs):
    rt = make_runtime(FixedController(10, 10))
    for _ in range(10):
        rt.step(**_kw(blank_obs()))
    rt.set_controller(FixedController(3, 2))
    for _ in range(10):
        rt.step(**_kw(blank_obs()))
    steps = [i["steps"] for i in rt.telemetry.inferences]
    assert steps[0] == 10 and steps[-1] == 3 and rt.telemetry.counters["controller_changed"] == 1


def test_sdk_from_pretrained_synthetic():
    from kinetixrt import Runtime

    rt = Runtime.from_pretrained("synthetic/toy-pickplace-controller", adaptive_compute=True, mode="virtual")
    env = ToyPickPlace(ToyConfig(), 1)
    obs = env.reset(1)
    out = rt.step(images=obs["images"], robot_state=obs["state"])
    assert out.action.shape == (3,) and out.decision is not None and out.decision.reasons
