"""Real-model tests. Run with KRT_MODEL_TESTS=1 (needs `make setup-model`, ~5 GB RAM)."""

import numpy as np
import pytest

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def backend():
    from kinetixrt.backends.smolvla import SmolVLABackend
    from kinetixrt.intelligence.temporal_cache import VisionFeatureCache

    return SmolVLABackend(vision_cache=VisionFeatureCache())


@pytest.fixture(scope="module")
def observations():
    from kinetixrt.bench.model import load_fixture

    return load_fixture()


def _noise(backend, seed):
    import torch

    cfg = backend.policy.config
    return torch.randn(1, cfg.chunk_size, cfg.max_action_dim, generator=torch.Generator().manual_seed(seed))


def test_adapter_is_bit_identical_to_reference(backend, observations):
    import torch

    from kinetixrt.types import ComputeBudget

    obs = observations[1]
    noise = _noise(backend, 3)
    with torch.inference_mode():
        ref = backend.post(backend.policy.predict_action_chunk(dict(backend.make_batch(obs)), noise=noise.clone()).clone())
    ours = backend.infer(obs, ComputeBudget(backend.policy.config.num_steps), noise=noise.clone()).actions
    assert np.array_equal(ours, ref[0].float().cpu().numpy())
    assert ours.shape == (50, 7) and np.all(np.isfinite(ours))


def test_budget_changes_actual_compute(backend, observations):
    from kinetixrt.types import ComputeBudget

    obs = observations[0]
    backend.infer(obs, ComputeBudget(3))  # warm-up
    t3 = min(backend.infer(obs, ComputeBudget(3)).timings.denoise_s for _ in range(2))
    t10 = min(backend.infer(obs, ComputeBudget(10)).timings.denoise_s for _ in range(2))
    assert t10 > 2.0 * t3, f"10 steps ({t10:.3f}s) should cost well over twice 3 steps ({t3:.3f}s)"


def test_vision_cache_is_exact_on_identical_frames_and_invalidates_on_change(backend, observations):
    from kinetixrt.types import ComputeBudget

    backend.reset(seed=0)
    a, b = observations[0], observations[3]
    noise = _noise(backend, 5)
    cold = backend.infer(a, ComputeBudget(5, reuse_vision=True), noise=noise.clone())
    warm = backend.infer(a, ComputeBudget(5, reuse_vision=True), noise=noise.clone())
    assert cold.timings.vision_cache_hits == 0 and warm.timings.vision_cache_hits == 2
    assert np.array_equal(cold.actions, warm.actions)
    assert warm.timings.vision_s < 0.2 * cold.timings.vision_s
    changed = backend.infer(b, ComputeBudget(5, reuse_vision=True), noise=noise.clone())
    assert changed.timings.vision_cache_hits == 0, "a different frame must not reuse cached features"


def test_real_chunks_flow_through_the_runtime(backend, observations):
    from kinetixrt.bench.common import controller_from_name, make_latency_model, make_libero_runtime
    from kinetixrt.types import ActionSource

    rt = make_libero_runtime(backend, controller_from_name("fixed_high"), make_latency_model("calibrated"))
    rt.reset(seed=0, task=observations[0].task)
    sources = []
    for tick in range(45):
        o = observations[min(tick // 10, len(observations) - 1)]
        out = rt.step(images=o.images, robot_state=o.state)
        sources.append(out.source)
        assert out.action.shape == (7,) and np.all(np.abs(out.action) <= 1.0)
    assert sources[0] is ActionSource.FALLBACK_STARTUP and sources[-1] is ActionSource.CHUNK
    inf = rt.telemetry.inferences[0]
    assert inf["steps"] == 10 and inf["deliver_tick"] > inf["obs_tick"] and inf["model_wall_s"] > 0
