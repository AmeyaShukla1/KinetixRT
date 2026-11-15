"""LIBERO closed-loop harness (real MuJoCo simulation via LeRobot's integration).

Uses ``lerobot.sim.integrations.libero`` for environment construction and
observation conventions, i.e. the same code path as ``lerobot-eval``.
Simulator ground truth is used only to (a) report task success and (b) inject
an explicitly documented disturbance in the dynamic-scene scenario. It is
never given to the policy or to the runtime.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GRIPPER_DIM = 6
MOTION_DIMS = (0, 1, 2)


def ensure_libero_config() -> None:
    """LIBERO prompts on stdin at first import; write its config file instead."""
    cfg_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", Path.home() / ".libero"))
    cfg_file = cfg_dir / "config.yaml"
    if cfg_file.exists():
        return
    import importlib.util

    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError("LIBERO is not installed: pip install 'lerobot[libero]'")
    root = Path(list(spec.submodule_search_locations)[0]) / "libero"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_file.write_text(
        f"benchmark_root: {root}\nbddl_files: {root / 'bddl_files'}\ninit_states: {root / 'init_files'}\n"
        f"datasets: {root.parent / 'datasets'}\nassets: {root / 'assets'}\n"
    )


def configure_headless_rendering() -> None:
    """Pick a MuJoCo GL backend: EGL with a GPU, OSMesa (software) otherwise."""
    if "MUJOCO_GL" in os.environ:
        return
    try:
        import torch

        has_gpu = torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        has_gpu = False
    backend = "egl" if has_gpu else "osmesa"
    os.environ["MUJOCO_GL"] = backend
    os.environ.setdefault("PYOPENGL_PLATFORM", backend)


@dataclass
class Perturbation:
    """Abruptly displace the task's object of interest at a given tick."""

    tick: int
    dx: float = 0.0
    dy: float = 0.08
    only_if_not_lifted: bool = True


class LiberoSim:
    def __init__(self, suite: str = "libero_spatial", task_id: int = 0, resolution: int = 360) -> None:
        ensure_libero_config()
        configure_headless_rendering()
        from lerobot.sim.integrations import libero as L

        self._L = L
        fns = L.env_fns(
            suite,
            n_envs=1,
            gym_kwargs={
                "obs_type": "pixels_agent_pos",
                "task_ids": [task_id],
                "observation_height": resolution,
                "observation_width": resolution,
            },
        )
        self.env = fns[suite][task_id][0]()
        self.suite, self.task_id = suite, task_id
        self.task = self.env.task_description
        self.max_steps = int(self.env._max_episode_steps)
        self.action_dim = 7
        self.control_dt = 1.0 / float(self.env.control_freq)
        self._tick = 0
        self.perturbed_at: int | None = None

    def _convert(self, raw: dict) -> dict:
        obs = self._L.canonical_observation(raw)
        images = {k.split("observation.images.", 1)[1]: v for k, v in obs.items() if k.startswith("observation.images.")}
        return {"images": images, "state": np.asarray(obs["observation.state"], dtype=np.float32), "canonical": obs}

    def reset(self, init_state: int = 0, seed: int = 0) -> dict:
        self.env.init_state_id = int(init_state)
        raw, _ = self.env.reset(seed=seed)
        self._tick = 0
        self.perturbed_at = None
        return self._convert(raw)

    def step(self, action: np.ndarray) -> tuple[dict, bool, bool]:
        raw, _reward, terminated, _trunc, info = self.env.step(np.asarray(action, dtype=np.float32))
        self._tick += 1
        success = bool(info.get("is_success", False))
        done = bool(terminated) or self._tick >= self.max_steps
        return self._convert(raw), success, done

    # ----------------------------------------------------- scenario injection
    def object_of_interest(self) -> str | None:
        inner = self.env._env.env
        names = list(getattr(inner, "obj_of_interest", []) or [])
        for n in names:
            if n in getattr(inner, "objects_dict", {}):
                return n
        return None

    def perturb(self, p: Perturbation) -> bool:
        """Teleport the first movable object of interest by (dx, dy) metres."""
        inner = self.env._env.env
        name = self.object_of_interest()
        if name is None:
            return False
        obj = inner.objects_dict[name]
        joint = obj.joints[-1]
        qpos = np.array(inner.sim.data.get_joint_qpos(joint))
        if p.only_if_not_lifted:
            table_z = getattr(self, "_obj_z0", None)
            if table_z is not None and qpos[2] > table_z + 0.02:
                return False
        qpos[0] += p.dx
        qpos[1] += p.dy
        inner.sim.data.set_joint_qpos(joint, qpos)
        qvel = np.array(inner.sim.data.get_joint_qvel(joint))
        inner.sim.data.set_joint_qvel(joint, np.zeros_like(qvel))
        inner.sim.forward()
        self.perturbed_at = self._tick
        return True

    def remember_object_height(self) -> None:
        inner = self.env._env.env
        name = self.object_of_interest()
        if name is not None:
            joint = inner.objects_dict[name].joints[-1]
            self._obj_z0 = float(inner.sim.data.get_joint_qpos(joint)[2])

    def close(self) -> None:
        self.env.close()
