"""Model-level benchmark commands: validate-model, record-obs, latency."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from kinetixrt.bench.common import RESULTS, hardware_info, rss_gb, write_json
from kinetixrt.core.telemetry import latency_summary
from kinetixrt.types import ComputeBudget, Observation

FIXTURE = Path(__file__).parent / "fixtures" / "libero_obs.npz"


def load_fixture(path: Path = FIXTURE) -> list[Observation]:
    """Real LIBERO observations recorded with ``krt-bench record-obs``."""
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `krt-bench record-obs` (needs LIBERO)")
    z = np.load(path, allow_pickle=False)
    task = str(z["task"])
    out = []
    for i in range(len(z["ticks"])):
        out.append(
            Observation(i, int(z["ticks"][i]), float(z["ticks"][i]) * 0.05, {"image": z["image"][i], "image2": z["image2"][i]}, z["state"][i], task)
        )
    return out


# ------------------------------------------------------------------ record-obs
def record_obs(args: argparse.Namespace) -> None:
    from kinetixrt.backends.smolvla import SmolVLABackend
    from kinetixrt.simulation.libero import LiberoSim

    backend = SmolVLABackend(args.model, device=args.device)
    sim = LiberoSim(args.suite, args.task)
    obs = sim.reset(init_state=args.init_state, seed=args.init_state)
    frames: dict[str, list] = {"image": [], "image2": [], "state": [], "ticks": []}
    plan, anchor = None, 0
    for tick in range(args.every * args.count):
        if tick % args.every == 0:
            frames["image"].append(obs["images"]["image"])
            frames["image2"].append(obs["images"]["image2"])
            frames["state"].append(obs["state"])
            frames["ticks"].append(tick)
        if plan is None or tick - anchor >= 10:
            o = Observation(tick, tick, tick * sim.control_dt, obs["images"], obs["state"], sim.task)
            plan, anchor = backend.infer(o, ComputeBudget(10)).actions, tick
        obs, _success, done = sim.step(plan[tick - anchor])
        if done:
            break
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, task=np.array(sim.task), **{k: np.stack(v) for k, v in frames.items()})
    print(f"recorded {len(frames['ticks'])} real LIBERO observations -> {out} ({out.stat().st_size / 1e6:.2f} MB)")


# -------------------------------------------------------------- validate-model
def validate_model(args: argparse.Namespace) -> None:
    import torch

    from kinetixrt.backends.smolvla import SmolVLABackend

    report: dict = {"hardware": hardware_info(), "model_id": args.model, "checks": {}}
    checks = report["checks"]
    t0 = time.time()
    backend = SmolVLABackend(args.model, device=args.device)
    report["backend"] = vars(backend.info)
    report["load_s"] = round(time.time() - t0, 1)
    policy = backend.policy

    # 1. the weights in memory are the published checkpoint, not an initialisation
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    ckpt = hf_hub_download(args.model, "model.safetensors")
    report["checkpoint_file"] = str(Path(ckpt).resolve().name)
    report["checkpoint_commit"] = Path(ckpt).parent.name
    sd = policy.state_dict()
    with safe_open(ckpt, framework="pt") as f:
        keys = list(f.keys())
        probe = keys[:: max(1, len(keys) // 40)]
        mism = [k for k in probe if k not in sd or not torch.equal(f.get_tensor(k).to(sd[k].dtype), sd[k].cpu())]
    checks["pretrained_weights_loaded"] = {"pass": not mism, "tensors_compared": len(probe), "tensors_in_checkpoint": len(keys), "mismatched": mism[:5]}

    # 2. adapter == reference inference path (same observation, same noise)
    if args.source == "libero":
        from kinetixrt.simulation.libero import LiberoSim

        sim = LiberoSim(args.suite, args.task)
        raw = sim.reset(init_state=0, seed=0)
        obs_list = [Observation(0, 0, 0.0, raw["images"], raw["state"], sim.task)]
    else:
        sim = None
        obs_list = load_fixture()[:3]
    diffs, dets = [], []
    for i, obs in enumerate(obs_list):
        noise = torch.randn(1, policy.config.chunk_size, policy.config.max_action_dim, generator=torch.Generator().manual_seed(i))
        with torch.inference_mode():
            ref = backend.post(policy.predict_action_chunk(dict(backend.make_batch(obs)), noise=noise.clone()).clone())[0].float().cpu().numpy()
        ours = backend.infer(obs, ComputeBudget(policy.config.num_steps), noise=noise.clone()).actions
        again = backend.infer(obs, ComputeBudget(policy.config.num_steps), noise=noise.clone()).actions
        diffs.append(float(np.abs(ours - ref).max()))
        dets.append(float(np.abs(ours - again).max()))
    checks["adapter_matches_reference"] = {"pass": max(diffs) <= 1e-6, "max_abs_diff": max(diffs), "observations": len(obs_list)}
    checks["deterministic_given_noise"] = {"pass": max(dets) == 0.0, "max_abs_diff": max(dets)}

    # 3. outputs are valid actions
    a = backend.infer(obs_list[0], ComputeBudget(policy.config.num_steps)).actions
    checks["action_validity"] = {
        "pass": bool(np.all(np.isfinite(a)) and a.shape == (policy.config.chunk_size, backend.info.action_dim)),
        "shape": list(a.shape),
        "min": float(a.min()),
        "max": float(a.max()),
        "fraction_outside_unit_box": float(np.mean(np.abs(a) > 1.0)),
    }

    # 4. step-count sweep: how far is an n-step solve from the reference 10-step solve
    noise = torch.randn(1, policy.config.chunk_size, policy.config.max_action_dim, generator=torch.Generator().manual_seed(7))
    sweep = backend.infer_sweep(obs_list[0], [1, 2, 3, 5, 10, 20], [noise])
    ref10 = sweep[(0, 10)]
    checks["step_sweep_vs_10"] = {
        str(n): {"max_abs": float(np.abs(sweep[(0, n)] - ref10).max()), "rms": float(np.sqrt(np.mean((sweep[(0, n)] - ref10) ** 2)))}
        for n in [1, 2, 3, 5, 20]
    }

    # 5. reference rollout equivalence: lerobot's env + select_action vs our adapter, step by step
    if sim is not None and args.rollout_steps > 0:
        from lerobot.envs.configs import LiberoEnvConfig
        from lerobot.envs.factory import make_env, make_env_pre_post_processors
        from lerobot.envs.utils import prepare_canonical_observation

        cfg = LiberoEnvConfig(task=args.suite, task_ids=[args.task])
        env_pre, env_post = make_env_pre_post_processors(env_cfg=cfg, policy_cfg=policy.config)
        venv = make_env(cfg, n_envs=1)[args.suite][args.task]
        vobs, _ = venv.reset(seed=[0])
        raw = sim.reset(init_state=0, seed=0)
        policy.reset()
        obs_diff, act_diff = 0.0, 0.0
        for step in range(args.rollout_steps):
            o = prepare_canonical_observation(vobs)
            obs_diff = max(obs_diff, float(np.abs(vobs["observation.images.image"][0].astype(np.int16) - raw["images"]["image"].astype(np.int16)).max()))
            obs_diff = max(obs_diff, float(np.abs(vobs["observation.state"][0] - raw["state"]).max()))
            o["task"] = list(venv.call("task_description"))
            noise = torch.randn(1, policy.config.chunk_size, policy.config.max_action_dim, generator=torch.Generator().manual_seed(100 + step))
            with torch.inference_mode():
                policy.reset()
                ref_a = policy.select_action(backend.pre(env_pre(o)), noise=noise.clone())
            ref_a = env_post({"action": backend.post(ref_a)})["action"].float().cpu().numpy()
            ours = backend.infer(Observation(step, step, step * 0.05, raw["images"], raw["state"], sim.task), ComputeBudget(policy.config.num_steps), noise=noise.clone()).actions[0]
            act_diff = max(act_diff, float(np.abs(ours - ref_a[0]).max()))
            vobs, _, _, _, _ = venv.step(ref_a)
            raw, _, _ = sim.step(ref_a[0])
        venv.close()
        checks["reference_rollout_equivalence"] = {
            "pass": obs_diff == 0.0 and act_diff <= 1e-6,
            "steps": args.rollout_steps,
            "max_observation_diff": obs_diff,
            "max_action_diff": act_diff,
            "note": "lerobot make_env + prepare_canonical_observation + select_action vs LiberoSim + SmolVLABackend",
        }

        # 6. chunked execution equivalence: lerobot's own action queue with n_action_steps=H
        #    vs. the KinetixRT runtime with FixedController(steps, H) at zero latency.
        from kinetixrt.bench.common import make_latency_model, make_libero_runtime
        from kinetixrt.intelligence.controller import FixedController

        H, n_ticks = 10, args.chunked_steps
        if n_ticks > 0:
            saved = policy.config.n_action_steps
            policy.config.n_action_steps = H
            venv = make_env(cfg, n_envs=1)[args.suite][args.task]
            try:
                vobs, _ = venv.reset(seed=[0])
                raw = sim.reset(init_state=0, seed=0)
                policy.reset()
                rt = make_libero_runtime(backend, FixedController(policy.config.num_steps, H), make_latency_model("zero"))
                rt.reset(seed=1234, task=sim.task)
                gen = torch.Generator().manual_seed(1234)
                act_diff, obs_diff, refills = 0.0, 0.0, 0
                for step in range(n_ticks):
                    o = prepare_canonical_observation(vobs)
                    o["task"] = list(venv.call("task_description"))
                    obs_diff = max(obs_diff, float(np.abs(vobs["observation.state"][0] - raw["state"]).max()))
                    noise = None
                    if step % H == 0:  # lerobot refills its queue here: give it the noise our backend will draw
                        noise = torch.randn((1, policy.config.chunk_size, policy.config.max_action_dim), generator=gen, dtype=torch.float32)
                        refills += 1
                    with torch.inference_mode():
                        ref_a = policy.select_action(backend.pre(env_pre(o)), noise=noise)
                    ref_a = env_post({"action": backend.post(ref_a)})["action"].float().cpu().numpy()
                    out = rt.step(images=raw["images"], robot_state=raw["state"])
                    act_diff = max(act_diff, float(np.abs(out.action - np.clip(ref_a[0], -1.0, 1.0)).max()))
                    vobs, _, _, _, _ = venv.step(out.action[None])
                    raw, _, _ = sim.step(out.action)
                checks["chunked_execution_equivalence"] = {
                    "pass": act_diff <= 1e-6 and obs_diff == 0.0 and len(rt.telemetry.inferences) == refills,
                    "ticks": n_ticks,
                    "horizon": H,
                    "reference_queue_refills": refills,
                    "runtime_inferences": len(rt.telemetry.inferences),
                    "max_action_diff": act_diff,
                    "max_state_diff": obs_diff,
                    "note": "lerobot select_action action queue (n_action_steps=10) vs KinetixRT Runtime + FixedController(10, 10), zero latency, same noise; reference actions clipped to [-1, 1] as the runtime's validator does",
                }
            finally:
                policy.config.n_action_steps = saved
                policy.reset()
                venv.close()
    report["all_passed"] = all(c.get("pass", True) for c in checks.values() if isinstance(c, dict))
    report["rss_gb"] = rss_gb()
    write_json(Path(args.out), report)
    for k, v in checks.items():
        print(f"[validate-model] {k}: {v}")
    print(f"[validate-model] all_passed={report['all_passed']} -> {args.out}")
    if not report["all_passed"]:
        raise SystemExit(1)


# --------------------------------------------------------------------- latency
def _burn(stop: mp.Event) -> None:  # pragma: no cover - load generator
    x = 0.0
    while not stop.is_set():
        for i in range(200000):
            x += i * 1e-9


def latency(args: argparse.Namespace) -> None:
    import torch

    from kinetixrt.backends.smolvla import SmolVLABackend
    from kinetixrt.intelligence.temporal_cache import VisionFeatureCache

    backend = SmolVLABackend(args.model, device=args.device, num_threads=args.threads, vision_cache=VisionFeatureCache())
    if args.compile:
        vw = backend.policy.model.vlm_with_expert
        vw.get_vlm_model().vision_model = torch.compile(vw.get_vlm_model().vision_model)
    obs_list = load_fixture()
    stop = mp.Event()
    procs = [mp.Process(target=_burn, args=(stop,), daemon=True) for _ in range(args.background_load)]
    for p in procs:
        p.start()
    report: dict = {
        "hardware": hardware_info(),
        "backend": vars(backend.info),
        "repeats": args.repeats,
        "warmup": args.warmup,
        "background_load_procs": args.background_load,
        "compile": args.compile,
        "budgets": {},
    }
    if backend.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    try:
        for reuse in [False, True] if args.cache else [False]:
            for n in args.steps:
                budget = ComputeBudget(n, reuse_vision=reuse)
                rows = []
                for i in range(args.warmup + args.repeats):
                    obs = obs_list[0] if reuse else obs_list[i % len(obs_list)]
                    r = backend.infer(obs, budget)
                    if i >= args.warmup:
                        rows.append(r.timings)
                tot = [t.total_s for t in rows]
                key = f"steps={n}" + (",vision_cache_hit" if reuse else "")
                report["budgets"][key] = {
                    "num_steps": n,
                    "vision_cache": reuse,
                    "total_s": latency_summary(tot),
                    "jitter_p95_minus_p50_s": (float(np.percentile(tot, 95) - np.percentile(tot, 50)) if len(tot) >= 20 else None),
                    "stage_median_s": {
                        "preprocess": float(np.median([t.preprocess_s for t in rows])),
                        "vision": float(np.median([t.vision_s for t in rows])),
                        "prefix": float(np.median([t.prefix_s for t in rows])),
                        "denoise": float(np.median([t.denoise_s for t in rows])),
                        "postprocess": float(np.median([t.postprocess_s for t in rows])),
                    },
                    "cache_hits": int(sum(t.vision_cache_hits for t in rows)),
                }
                s = report["budgets"][key]
                print(f"[latency] {key:28s} p50={s['total_s']['p50'] * 1e3:8.1f} ms p95={(s['total_s']['p95'] or float('nan')) * 1e3:8.1f} ms stages={ {k: round(v * 1e3, 1) for k, v in s['stage_median_s'].items()} }", flush=True)
    finally:
        stop.set()
        for p in procs:
            p.join(2)
    plain = [(v["num_steps"], v["total_s"]["p50"]) for v in report["budgets"].values() if not v["vision_cache"]]
    if len(plain) >= 2:
        b, a = np.polyfit([p[0] for p in plain], [p[1] for p in plain], 1)
        report["linear_fit"] = {"base_s": float(a), "per_step_s": float(b), "note": "p50 latency ~= base + per_step * num_steps"}
    report["rss_gb"] = rss_gb()
    if backend.device.startswith("cuda"):
        report["cuda_peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    write_json(Path(args.out), report)
    print(f"[latency] fit={report.get('linear_fit')} rss={report['rss_gb']:.2f} GB -> {args.out}")


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("validate-model", help="load real pretrained weights and validate against the reference inference path")
    p.add_argument("--model", default="HuggingFaceVLA/smolvla_libero")
    p.add_argument("--device", default="auto")
    p.add_argument("--source", choices=["libero", "fixture"], default="libero")
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--rollout-steps", type=int, default=8)
    p.add_argument("--chunked-steps", type=int, default=25, help="ticks of the chunked-execution equivalence check (0 = skip)")
    p.add_argument("--out", default=str(RESULTS / "validate_model.json"))
    p.set_defaults(func=validate_model)

    p = sub.add_parser("record-obs", help="record real LIBERO observations as a fixture for model-only benchmarks")
    p.add_argument("--model", default="HuggingFaceVLA/smolvla_libero")
    p.add_argument("--device", default="auto")
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--init-state", type=int, default=45)
    p.add_argument("--every", type=int, default=10)
    p.add_argument("--count", type=int, default=6)
    p.add_argument("--out", default=str(FIXTURE))
    p.set_defaults(func=record_obs)

    p = sub.add_parser("latency", help="per-budget inference latency and stage breakdown (real model, recorded observations)")
    p.add_argument("--model", default="HuggingFaceVLA/smolvla_libero")
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--steps", type=int, nargs="+", default=[1, 3, 5, 10, 20])
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--cache", action="store_true", help="also measure the vision-feature-cache hit path")
    p.add_argument("--compile", action="store_true", help="experimental: torch.compile the vision tower")
    p.add_argument("--background-load", type=int, default=0, help="number of CPU-burning processes (real contention)")
    p.add_argument("--out", default=str(RESULTS / "latency.json"))
    p.set_defaults(func=latency)
