import numpy as np

from kinetixrt.control.safety import ActionValidator, ValidatorConfig
from kinetixrt.intelligence.temporal_cache import VisionFeatureCache


def test_validator_rejects_non_finite_and_bad_shape():
    v = ActionValidator()
    assert v.validate(np.array([0.0, np.nan]), 0).reason == "non_finite"
    assert v.validate(np.array([0.0, np.inf]), 0).reason == "non_finite"
    assert v.validate(np.zeros((2, 2)), 0).reason == "bad_shape"
    assert v.validate(np.zeros(3), 0, action_dim=2).reason == "bad_shape"
    assert v.counts["non_finite"] == 2 and v.counts["shape"] == 2


def test_validator_clips_to_bounds_and_expires_old_plans():
    v = ActionValidator(ValidatorConfig(low=-1.0, high=1.0, max_chunk_age_ticks=10))
    r = v.validate(np.array([2.0, -3.0, 0.5]), 0)
    assert r.ok and r.clipped and np.allclose(r.action, [1.0, -1.0, 0.5])
    assert v.validate(np.zeros(3), 11).reason == "expired"


def test_validator_rate_limit_only_on_selected_dims():
    v = ActionValidator(ValidatorConfig(max_delta=0.1, rate_limit_dims=(0,)))
    v.validate(np.array([0.0, 0.0]), 0)
    r = v.validate(np.array([1.0, 1.0]), 0)
    assert r.rate_limited and np.allclose(r.action, [0.1, 1.0])


def test_vision_cache_hit_requires_small_change_same_task_and_fresh_entry():
    cache = VisionFeatureCache(tau_mean=0.01, tau_peak=0.05, max_age_ticks=20)
    img = np.full((72, 72, 3), 100, dtype=np.uint8)
    feats, gray = cache.lookup("cam", img, tick=0, task="a")
    assert feats is None
    cache.store("cam", gray, "F", tick=0, task="a")
    assert cache.lookup("cam", img, 5, "a")[0] == "F"
    assert cache.lookup("cam", img, 5, "b")[0] is None  # instruction changed
    assert cache.lookup("cam", img, 30, "a")[0] is None  # too old
    moved = img.copy()
    moved[10:20, 10:20] = 255  # localized change (an object moved)
    assert cache.lookup("cam", moved, 5, "a")[0] is None
    brighter = (img.astype(np.int16) + 5).astype(np.uint8)  # global change above tau_mean
    assert cache.lookup("cam", brighter, 5, "a")[0] is None
    s = cache.stats
    assert s["hits"] == 1 and s["invalid_task"] == 1 and s["invalid_age"] == 1 and s["invalid_change"] == 2
    cache.clear()
    assert cache.lookup("cam", img, 5, "a")[0] is None
