"""Real SmolVLA backend (Hugging Face LeRobot implementation, PyTorch).

The adapter runs the *reference* modules of ``lerobot`` stage by stage
(vision tower -> VLM prefix pass with KV cache -> flow-matching Euler loop over
the action expert) so that each stage can be timed and budgeted. It does not
re-implement any model math: the prefix embedding, the denoise step and the
Euler integrator are the library's own functions. ``krt-bench validate-model``
checks bit-level agreement with ``SmolVLAPolicy.predict_action_chunk``.

Verified architecture facts used here (lerobot 0.7.0, checkpoint
``HuggingFaceVLA/smolvla_libero``):

* per-inference KV caching of the prefix is already built in (``use_cache``);
* the number of Euler steps is a free parameter of the uniform integrator, so
  any positive step count is a valid (coarser or finer) discretization;
* image and language tokens do not attend to the state token, but the state
  token is part of the prefix, so the prefix must be recomputed per observation.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from kinetixrt.backends.base import BackendInfo, PolicyBackend
from kinetixrt.intelligence.temporal_cache import VisionFeatureCache
from kinetixrt.types import ComputeBudget, InferenceResult, Observation, StageTimings

DEFAULT_MODEL_ID = "HuggingFaceVLA/smolvla_libero"


def resolve_device(device: str = "auto") -> str:
    import torch

    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SmolVLABackend(PolicyBackend):
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        revision: str | None = None,
        num_threads: int | None = None,
        vision_cache: VisionFeatureCache | None = None,
        supported_steps: tuple[int, ...] = (3, 5, 10),
        seed: int = 0,
        noise_mode: str = "fresh",
    ) -> None:
        if noise_mode not in ("fresh", "episode"):
            raise ValueError("noise_mode must be 'fresh' (new noise per inference, the reference behaviour) or 'episode'")
        self.noise_mode = noise_mode
        self._episode_noise: Any = None
        import torch
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.common.flow_matching import euler_integrate
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        self._torch = torch
        self._euler_integrate = euler_integrate
        self._make_att_2d_masks = make_att_2d_masks
        self._k_tokens, self._k_mask = OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
        if num_threads:
            torch.set_num_threads(num_threads)
        self.device = resolve_device(device)
        policy = SmolVLAPolicy.from_pretrained(model_id, revision=revision)
        policy.config.device = self.device
        policy.to(self.device).eval()
        self.policy = policy
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=model_id,
            pretrained_revision=revision,
            preprocessor_overrides={"device_processor": {"device": self.device}},
        )
        cfg = policy.config
        self.vision_cache = vision_cache
        self._cam_keys = list(cfg.image_features)
        self._cam_names = [k.split("observation.images.", 1)[-1] for k in self._cam_keys]
        action_feature = cfg.action_feature
        dtypes = sorted({str(p.dtype).replace("torch.", "") for p in policy.parameters()})
        self.info = BackendInfo(
            name="smolvla-pytorch",
            model_id=model_id,
            device=self.device,
            chunk_size=cfg.chunk_size,
            action_dim=int(action_feature.shape[0]),
            default_steps=cfg.num_steps,
            supported_steps=tuple(supported_steps),
            is_real_model=True,
            extra={
                "params_million": round(sum(p.numel() for p in policy.parameters()) / 1e6, 1),
                "dtypes": dtypes,
                "torch": torch.__version__,
                "threads": torch.get_num_threads(),
                "cameras": self._cam_names,
                "use_cache": bool(cfg.use_cache),
                "revision": revision,
            },
        )
        self._call: dict[str, Any] = {}
        self._install_vision_hook()
        self.reset(seed)

    # ------------------------------------------------------------------ helpers
    def _sync(self) -> None:
        if self.device.startswith("cuda"):
            self._torch.cuda.synchronize()

    def _install_vision_hook(self) -> None:
        """Wrap the vision tower so it can be timed and (optionally) cached."""
        vw = self.policy.model.vlm_with_expert
        original = vw.embed_image
        backend = self

        def embed_image(image):  # noqa: ANN001
            call = backend._call
            idx = call.get("cam_idx", 0)
            call["cam_idx"] = idx + 1
            name = backend._cam_names[idx] if idx < len(backend._cam_names) else f"cam{idx}"
            t0 = time.perf_counter()
            cache = backend.vision_cache if call.get("reuse_vision") else None
            obs: Observation | None = call.get("obs")
            feats, gray = None, None
            if cache is not None and obs is not None and name in obs.images:
                feats, gray = cache.lookup(name, obs.images[name], obs.tick, obs.task)
            if feats is None:
                feats = original(image)
                call["misses"] = call.get("misses", 0) + 1
                store = backend.vision_cache
                if store is not None and obs is not None and name in obs.images:
                    if gray is None:
                        from kinetixrt.intelligence.analyzer import downsample_gray

                        gray = downsample_gray(obs.images[name], store.grid)
                    store.store(name, gray, feats, obs.tick, obs.task)
            else:
                call["hits"] = call.get("hits", 0) + 1
            backend._sync()
            call["vision_s"] = call.get("vision_s", 0.0) + time.perf_counter() - t0
            return feats

        vw.embed_image = embed_image
        self._original_embed_image = original

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._gen = self._torch.Generator().manual_seed(int(seed))
        self._episode_noise = None
        if self.vision_cache is not None:
            self.vision_cache.clear()

    def sample_noise(self) -> Any:
        """Initial flow-matching noise for one inference.

        ``noise_mode="episode"`` (experimental) reuses one noise tensor for every
        inference of an episode instead of drawing a new one each time."""
        cfg = self.policy.config
        if self.noise_mode == "episode" and self._episode_noise is not None:
            return self._episode_noise.clone()
        noise = self._torch.randn((1, cfg.chunk_size, cfg.max_action_dim), generator=self._gen, dtype=self._torch.float32)
        if self.noise_mode == "episode":
            self._episode_noise = noise.clone()
        return noise

    def make_batch(self, obs: Observation) -> dict[str, Any]:
        """Observation -> policy batch through LeRobot's own preprocessor."""
        torch = self._torch
        raw: dict[str, Any] = {}
        for key, name in zip(self._cam_keys, self._cam_names):
            if name in obs.images:
                img = torch.from_numpy(np.ascontiguousarray(obs.images[name]))
                raw[key] = img.permute(2, 0, 1).contiguous().float().div(255)[None]
        raw["observation.state"] = torch.from_numpy(np.asarray(obs.state, dtype=np.float32))[None]
        raw["task"] = [obs.task]
        return self.pre(raw)

    def _encode(self, batch: dict[str, Any], obs: Observation | None, reuse_vision: bool) -> dict[str, Any]:
        torch, policy, model = self._torch, self.policy, self.policy.model
        batch = policy._prepare_batch(batch)
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        self._call = {"obs": obs, "reuse_vision": reuse_vision}
        t0 = time.perf_counter()
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, batch[self._k_tokens], batch[self._k_mask], state=state
        )
        att_2d = self._make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = model.vlm_with_expert.forward(
            attention_mask=att_2d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=policy.config.use_cache,
        )
        self._sync()
        encode_s = time.perf_counter() - t0
        call = self._call
        return {
            "pad_masks": prefix_pad_masks,
            "kv": past_key_values,
            "vision_s": call.get("vision_s", 0.0),
            "prefix_s": encode_s - call.get("vision_s", 0.0),
            "hits": call.get("hits", 0),
            "misses": call.get("misses", 0),
        }

    def _denoise(self, enc: dict[str, Any], noise: Any, num_steps: int) -> Any:
        model = self.policy.model
        x = self._euler_integrate(
            lambda x_t, t: model.denoise_step(
                x_t=x_t, prefix_pad_masks=enc["pad_masks"], past_key_values=enc["kv"], timestep=t
            ),
            noise.to(self.device),
            num_steps,
            rtc_processor=None,
            rtc_enabled=False,
        )
        actions = x[:, :, : self.info.action_dim]
        if self.policy.config.adapt_to_pi_aloha:
            actions = self.policy._pi_aloha_encode_actions(actions)
        return actions

    # ---------------------------------------------------------------- inference
    def infer(self, obs: Observation, budget: ComputeBudget, noise: Any | None = None) -> InferenceResult:
        torch = self._torch
        t0 = time.perf_counter()
        batch = self.make_batch(obs)
        t1 = time.perf_counter()
        if noise is None:
            noise = self.sample_noise()
        with torch.inference_mode():
            enc = self._encode(batch, obs, budget.reuse_vision)
            t2 = time.perf_counter()
            actions = self._denoise(enc, noise, budget.num_steps)
            self._sync()
            t3 = time.perf_counter()
            env_actions = self.post(actions.clone())
        out = env_actions[0].to("cpu", dtype=torch.float32).numpy()
        t4 = time.perf_counter()
        timings = StageTimings(
            preprocess_s=t1 - t0,
            vision_s=enc["vision_s"],
            prefix_s=enc["prefix_s"],
            denoise_s=t3 - t2,
            postprocess_s=t4 - t3,
            total_s=t4 - t0,
            vision_cache_hits=enc["hits"],
            vision_cache_misses=enc["misses"],
        )
        return InferenceResult(out, budget, timings)

    def infer_sweep(self, obs: Observation, step_counts: list[int], noises: list[Any]) -> dict[tuple[int, int], np.ndarray]:
        """Analysis helper: one prefix pass, then several (noise, steps) solves.

        Returns ``{(noise_index, num_steps): [T, action_dim] env-space actions}``.
        """
        torch = self._torch
        batch = self.make_batch(obs)
        out: dict[tuple[int, int], np.ndarray] = {}
        with torch.inference_mode():
            enc = self._encode(batch, obs, False)
            for ni, noise in enumerate(noises):
                for n in step_counts:
                    a = self.post(self._denoise(enc, noise.clone(), n).clone())
                    out[(ni, n)] = a[0].to("cpu", dtype=torch.float32).numpy()
        return out
