# RESEARCH — what already exists, and the gap KinetixRT targets

Scope: inference/runtime systems and methods for chunked, flow-matching or diffusion
robot policies (VLAs), as of November 2026. Sources are listed at the end. Numbers
quoted from other projects are their authors' own claims; none were reproduced here.

## 1. What is already established (not novel, and not claimed by KinetixRT)

| Technique | State of the art | Where it lives |
|---|---|---|
| Within-inference KV caching of the VLM prefix | Standard. The prefix is encoded once and reused across all denoising steps. | π0 / openpi, SmolVLA (`use_cache`), GR00T |
| Asynchronous inference with an action queue | Standard. Client keeps a queue and requests a new chunk when it falls below a threshold `g·n`. | LeRobot async inference (policy server / robot client), SmolVLA paper |
| Handling inference delay across chunk boundaries | Real-Time Chunking (RTC): inpaint the next chunk against the actions that will execute during the delay; a training-time variant removes the inference overhead. | Physical Intelligence (RTC papers); integrated in LeRobot (`RTCProcessor`) |
| Dual-rate execution (slow policy, fast control) | Standard in robotics stacks. | ROS 2 control, Isaac ROS, most VLA deployments |
| Graph compilation, quantization, memory pre-allocation | Standard. | TensorRT, `torch.compile`, ONNX Runtime, IREE, GGML |
| Model serving with dynamic batching, multi-model scheduling | Standard for request/response serving. | NVIDIA Triton, Ray Serve |
| Native, Python-free VLA runtimes | Recent and active. | vla.cpp (GGUF, llama.cpp/ggml based; SmolVLA, π0, GR00T …), Embodied.cpp (portable C++ runtime, multi-rate execution) |
| Cross-frame token/KV reuse | Published. Reuses KV entries of visually static tokens with task-relevance filtering. | VLA-Cache; follow-ups such as TTF-VLA |
| Early exit / dynamic depth | Published. | DeeR-VLA, A1, RD-VLA, AVA-VLA, "Decoupled Early Exits" |
| Adaptive denoising budgets / adaptive execution horizon | **Active research area in 2026.** | VLA-Rolling (learned confidence → denoising budget), FASTER (horizon-aware schedule), GeCO (convergence-based exit), GeoAAC (velocity-field change → how many actions to execute), "Decoupled Early Exits" (step count as a free inference-time knob) |

Consequences for this project:

* Adaptive inference budgeting **is not new as an idea**. Several 2026 papers adapt
  the denoising step count or the execution horizon, usually with a learned
  estimator or a model-internal signal, evaluated as a *model* technique.
* A faster kernel path is not a gap either: vla.cpp and Embodied.cpp are credible
  native runtimes and TensorRT covers NVIDIA hardware. KinetixRT deliberately does
  **not** compete on kernels (see §3).

## 2. Competitor matrix

| System | Executes real VLAs | Async chunk execution | Deadline-aware scheduling | Adapts compute per observation | Uses physical-world signals | Explains each decision | Closed-loop task-quality evaluation of the adaptation | Hardware focus |
|---|---|---|---|---|---|---|---|---|
| LeRobot (sync / async / RTC) | yes (reference impl.) | yes (queue threshold) | no | no (fixed steps, fixed threshold) | no | no | n/a | PyTorch: CUDA, MPS, CPU |
| openpi + RTC | yes | yes (delay-aware inpainting) | no | no | no | no | yes (for RTC itself) | CUDA / JAX |
| NVIDIA Isaac ROS / GR00T + TensorRT | yes | via ROS graph | ROS executors, not model-aware | no | no | no | n/a | NVIDIA GPU / Jetson |
| NVIDIA Triton | model-agnostic | no (request/response) | queue policies, dynamic batching | no | no | no | n/a | NVIDIA GPU |
| vla.cpp | yes (GGUF) | client-side | no | no | no | no | parity vs reference | CPU, CUDA, Metal, Jetson |
| Embodied.cpp | yes | multi-rate execution | not documented | no | no | no | not documented | heterogeneous edge |
| VLA-Cache | research code | n/a | n/a | token reuse | image-token similarity only | no | yes | — |
| VLA-Rolling / GeCO / GeoAAC / FASTER | research code | partly | no | yes (learned or model-internal) | model-internal signals | no | yes | — |
| **KinetixRT (this repo)** | yes (SmolVLA via LeRobot modules) | yes (threaded + virtual-time) | yes (bounded queue, EDF, supersede, degrade) | yes (steps **and** refresh timing) | yes (scene surprise, plan-predicted actuator events, proprio, chunk disagreement, latency) | yes (rule trace per decision) | yes (paired, with CIs; see BENCHMARKS.md) | PyTorch backend; CUDA path prepared, **not yet run** |

"n/a" = the property is not what the system is for. "not documented" = I could not
confirm it from the sources below.

## 3. The precise gap

Existing work splits into two camps that do not meet:

1. **Runtimes** (LeRobot async, Triton, TensorRT, vla.cpp, Embodied.cpp) make a
   *fixed* computation cheaper or better scheduled. Their knobs (steps per chunk,
   actions per chunk, queue threshold) are constants chosen offline.
2. **Adaptive-compute methods** (VLA-Rolling, GeCO, GeoAAC, early-exit VLAs) change
   the computation per observation, but as model-specific research code, usually
   requiring training or model internals, without deadline awareness, fallbacks,
   observability, or a way to compare policies on equal footing.

KinetixRT targets the seam: a **model-agnostic runtime layer that treats "how much to
compute and when to refresh" as a per-tick, explainable scheduling decision driven
by measured physical and runtime signals**, with

* the decision separated from the backend (any backend exposing step count and
  chunk timing can be driven; a learned or model-internal estimator can be plugged
  in as one more signal),
* deadline awareness, stale-result rejection and defined fallbacks around it,
* a measurement harness that reports compute *and* closed-loop task quality for
  fixed, runtime-only and physical-aware policies under identical seeds.

This is an integration-and-evidence contribution, not a new algorithmic primitive.
Whether it is *valuable* depends on measured results; see BENCHMARKS.md, including
the negative findings.

## 4. What this implies for engineering priorities

* Do not write kernels, a tensor compiler or a file format. Use PyTorch now, keep
  `PolicyBackend` small so TensorRT / vla.cpp / Embodied.cpp can sit behind it.
* Do not invent an uncertainty score. Use signals that can be measured and
  validated against error proxies (`krt-bench signals`).
* Treat RTC as complementary: RTC fixes *how* consecutive chunks are stitched under
  delay; KinetixRT decides *when* to compute a chunk and with what budget. RTC is
  not yet wired into the KinetixRT adapter (listed as remaining work).
* Cross-observation KV patching (VLA-Cache style) is model-specific and not exact;
  for SmolVLA this repo implements the exact-when-unchanged per-camera vision
  feature reuse instead and reports its measured hit rate honestly.

## 5. Sources

* SmolVLA: Shukor et al., 2025 — https://huggingface.co/papers/2506.01844
* LeRobot asynchronous inference — https://huggingface.co/docs/lerobot/async and https://huggingface.co/blog/async-robot-inference
* Real-Time Chunking: Black, Galliker, Levine, "Real-Time Execution of Action Chunking Flow Policies" — https://arxiv.org/abs/2506.07339
* Training-time RTC: Black, Ren, Equi, Levine — https://arxiv.org/abs/2512.05964
* vla.cpp — https://arxiv.org/abs/2606.08094, https://github.com/AIWintermuteAI/vla.cpp
* Embodied.cpp — https://arxiv.org/abs/2607.02501
* VLA-Cache — https://arxiv.org/abs/2502.02175
* Decoupled Early Exits for Task-Dependent Compute Allocation in Flow-Matching VLAs — https://arxiv.org/abs/2609.29382
* A1 (early termination + flow warm start) — https://arxiv.org/abs/2604.05672
* RD-VLA — https://arxiv.org/abs/2602.07845
* GeCO — https://arxiv.org/abs/2603.17834
* FASTER — https://hyper.ai/en/papers/2603.19199
* GeoAAC — https://www.alphaxiv.org/abs/2609.20776
* VLA-Rolling — https://huggingface.co/Chenbao77/VLA-Rolling
* NVIDIA Isaac GR00T — https://www.nvidia.com/en-us/industries/robotics/isaac-groot/ ; Jetson Thor TensorRT tutorial — https://www.jetson-ai-lab.com/tutorials/groot_n17_on_thor/

Caveat: the 2026 items were found through web search summaries and abstracts; I
did not read every paper in full, and details in the matrix marked "not documented"
should be verified before being relied on. IREE and Triton entries are from general
knowledge of those systems rather than a fresh review.
