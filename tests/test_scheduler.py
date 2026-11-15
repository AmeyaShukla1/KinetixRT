import numpy as np
import pytest

from kinetixrt.core.scheduler import InferenceRequest, InferenceScheduler, LatencyPredictor, latency_ticks
from kinetixrt.types import ComputeBudget, Observation


def req(i, t=0.0, deadline=1.0, steps=10, stream="robot0", priority=0, t_capture=None):
    obs = Observation(i, i, t if t_capture is None else t_capture, {}, np.zeros(1), stream_id=stream)
    return InferenceRequest(i, obs, ComputeBudget(steps), t, deadline, priority)


def test_latency_predictor_learns_and_extrapolates_linearly():
    p = LatencyPredictor(alpha=0.5)
    assert p.predict(10) == 0.0
    for _ in range(5):
        p.observe(10, 1.3)
        p.observe(3, 0.8)
    assert p.predict(10) == pytest.approx(1.3)
    assert p.predict(5) == pytest.approx(0.8 + 2 * (0.5 / 7), rel=1e-3)
    p.observe(10, 2.3)
    assert p.predict(10, pessimism=1.0) > p.predict(10) > 1.3


def test_latency_ticks_rounds_up():
    assert latency_ticks(0.0, 0.05) == 0
    assert latency_ticks(0.13, 0.05) == 3
    assert latency_ticks(0.15, 0.05) == 3


def test_fifo_is_bounded_and_rejects_newest_when_full():
    s = InferenceScheduler("fifo", capacity=2)
    assert not s.submit(req(0)) and not s.submit(req(1))
    dropped = s.submit(req(2))
    assert [d.reason for d in dropped] == ["queue_full"] and dropped[0].request.req_id == 2
    assert s.take(0.0)[0].req_id == 0 and s.depth() == 1


def test_latest_supersedes_older_observation_of_same_stream():
    s = InferenceScheduler("latest", capacity=4)
    s.submit(req(0))
    dropped = s.submit(req(1))
    assert [d.reason for d in dropped] == ["superseded"]
    s.submit(req(2, stream="robot1"))
    assert s.depth() == 2 and s.take(0.0)[0].req_id == 1


def test_deadline_policy_orders_by_priority_then_deadline():
    s = InferenceScheduler("deadline", capacity=4)
    s.submit(req(0, deadline=5.0, stream="a"))
    s.submit(req(1, deadline=1.0, stream="b"))
    s.submit(req(2, deadline=9.0, stream="c", priority=1))
    assert [s.take(0.0)[0].req_id for _ in range(3)] == [2, 1, 0]


def test_deadline_policy_sheds_obsolete_observations():
    s = InferenceScheduler("deadline", capacity=4, max_obs_age_s=0.5)
    s.submit(req(0, t=0.0, stream="a"))
    s.submit(req(1, t=0.9, stream="b"))
    got, dropped = s.take(1.0)
    assert got.req_id == 1 and [d.reason for d in dropped] == ["obsolete_observation"]


def test_deadline_policy_degrades_budget_to_fit_slack():
    p = LatencyPredictor()
    for n, lat in [(10, 1.3), (5, 0.95), (3, 0.8)]:
        p.observe(n, lat)
    s = InferenceScheduler("deadline", capacity=2, predictor=p, step_options=(3, 5, 10), allow_degrade=True)
    s.submit(req(0, deadline=1.0, steps=10))
    got, _ = s.take(0.0)
    assert got.budget.num_steps == 5 and got.degraded_from == 10 and "scheduler" in got.reasons[-1]
    s.submit(req(1, deadline=0.1, steps=10))
    assert s.take(0.0)[0].budget.num_steps == 3  # nothing fits: use the cheapest
    s.submit(req(2, deadline=2.0, steps=10))
    assert s.take(0.0)[0].budget.num_steps == 10


def test_queue_never_exceeds_capacity_under_overload():
    s = InferenceScheduler("deadline", capacity=3)
    for i in range(100):
        s.submit(req(i, deadline=float(i % 7), stream=f"r{i % 10}"))
        assert s.depth() <= 3
    assert s.stats["max_depth"] <= 3 and s.stats["rejected_full"] > 0


def test_close_rejects_submissions_and_reopen_restores():
    s = InferenceScheduler("latest")
    s.close()
    assert s.submit(req(0))[0].reason == "scheduler_closed"
    s.reopen()
    assert not s.submit(req(1))
