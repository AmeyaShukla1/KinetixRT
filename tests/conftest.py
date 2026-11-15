import os

import numpy as np
import pytest

from kinetixrt.backends.synthetic import SyntheticBackend
from kinetixrt.core.runtime import Runtime, RuntimeConfig, hold_fallback
from kinetixrt.intelligence.analyzer import AnalyzerConfig, PhysicalEnvironmentAnalyzer


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "model" in item.keywords and os.environ.get("KRT_MODEL_TESTS") != "1":
            item.add_marker(pytest.mark.skip(reason="set KRT_MODEL_TESTS=1 to run real-model tests"))
        if "sim" in item.keywords and os.environ.get("KRT_SIM_TESTS") != "1":
            item.add_marker(pytest.mark.skip(reason="set KRT_SIM_TESTS=1 to run LIBERO tests"))


@pytest.fixture
def make_runtime():
    created = []

    def _make(controller, mode="virtual", latency=None, backend=None, **cfg_kwargs):
        be = backend or SyntheticBackend()
        cfg = RuntimeConfig(mode=mode, fallback=hold_fallback((2,)), disagreement_dims=(0, 1), **cfg_kwargs)
        if latency is not None:
            cfg.latency_model = latency
        rt = Runtime(be, controller, cfg, PhysicalEnvironmentAnalyzer(AnalyzerConfig(event_dims=(2,), motion_dims=(0, 1))))
        created.append(rt)
        return rt

    yield _make
    for rt in created:
        rt.stop()


@pytest.fixture
def blank_obs():
    def _obs(seed=0):
        rng = np.random.default_rng(seed)
        img = np.full((64, 64, 3), 30, dtype=np.uint8)
        img[20:25, 30:35] = (220, 20, 20)
        img[50:57, 50:57] = (20, 220, 20)
        return {"images": {"image": img}, "state": np.array([0.2, 0.2, 0.0], dtype=np.float32) + rng.normal(0, 1e-4, 3).astype(np.float32)}

    return _obs
