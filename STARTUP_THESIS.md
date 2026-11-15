# STARTUP THESIS — KinetixRT

Status of this document: an analysis, not a pitch. Section 5 contains only what
was measured in this repository (CPU, simulation). Everything about demand,
pricing and ROI is an assumption and is labelled as such. No customer, revenue or
partnership exists.

## 1. Initial target customer

**Robotics teams that deploy chunked VLA / diffusion policies on a fixed on-robot
compute budget and do not own the model's inference stack** — in practice:
integrators and robot OEM software teams shipping manipulation cells or mobile
manipulators on Jetson-class or single-GPU edge hardware, fine-tuning open
policies (SmolVLA, π0-family, GR00T) rather than training their own foundation
model.

Why this segment first:

* They feel the constraint directly: compute per robot is a line item on the bill
  of materials, and the control rate / chunk settings are tuned by hand per task.
* They do not have a runtime team. Foundation-model companies (Physical
  Intelligence, NVIDIA, Figure) do, and will build this in-house.
* Their models are standard open architectures, so one adapter covers many
  customers.

Segments deliberately **not** first: foundation-model labs (build it themselves),
warehouse fleet operators (buy whole robots, not runtimes), cloud inference
providers (latency budget dominated by the network, different product).

## 2. Specific customer problem

A chunked policy has three run-time knobs: denoising steps, how many predicted
actions to execute before re-planning, and when to start the next inference.
Today they are constants. That creates three concrete costs:

1. **Over-provisioning.** Settings are chosen for the hardest phase of the hardest
   task and paid for during every easy phase.
2. **Fragility under load.** When inference gets slower (thermal throttling, a
   second model on the same GPU, a perception stack spike), a fixed budget
   overruns its action buffer; the robot stalls or executes stale actions.
3. **No visibility.** Teams cannot see why the robot hesitated: there is no record
   of plan age, deadline slack or which inference result was used.

## 3. Existing alternatives

| Alternative | What it solves | What it leaves open |
|---|---|---|
| LeRobot async inference / RTC | Non-blocking execution, smooth chunk hand-over under delay | Constants chosen by hand; no budget adaptation; no deadline logic |
| TensorRT / `torch.compile` / quantization | Makes the fixed computation faster | Same computation every tick; NVIDIA-specific |
| vla.cpp, Embodied.cpp | Native, low-memory execution on edge hardware | Fixed budgets; no physical-world scheduling |
| Triton | Multi-model serving, batching | Request/response semantics; not chunk- or deadline-aware |
| Research code for adaptive denoising / early exit | Per-observation compute | Model-specific, often needs training; not a deployable runtime |
| In-house scripts | Whatever the team needed last month | Unmaintained, untested, unobservable |

A faster kernel path (rows 2–3) is complementary: KinetixRT decides how much work
to submit; those systems execute it.

## 4. Technical differentiation

1. Per-tick, explainable decision about **steps and refresh timing** from measured
   physical signals plus runtime statistics, independent of the model backend.
2. Reliability machinery around that decision: bounded queue, stale-result
   rejection, re-anchoring, deadline fitting, defined fallbacks, action validation.
3. A measurement harness that reports compute and closed-loop task success for
   fixed, runtime-only and physical-aware policies on identical seeds, with
   confidence intervals — so a customer can check the claim on their own task.
4. Observability: every inference carries its reason, latency, age and outcome.

None of the individual techniques is new (RESEARCH.md). The differentiation is
the integration and the evidence discipline.

## 5. Actual measured advantage

<!-- MEASURED:BEGIN -->
Everything below was measured with the real SmolVLA LIBERO checkpoint **on a
4-core CPU in simulation**; details, raw data and caveats are in
[BENCHMARKS.md](BENCHMARKS.md). There is no GPU, edge-device or real-robot
measurement.

| Claim | Measurement | Strength of evidence |
|---|---|---|
| Fewer denoising steps cost nothing measurable | 3 vs. 10 steps: −71% denoising work, −31% model time, same task outcome in 22 of 23 paired episodes | Good for compute; success known to ±20 pp |
| Refresh frequency can be cut far below the reference setting | Phase-dependent refresh reached 13 of 20 successes vs. 12 of 20 for one-inference-per-tick, with 26% of the inferences (CI 19–36%) | Compute ratio solid; success equivalence only within ±20 pp |
| Physical signals are what makes the saving safe | Phase-aware rule vs. phase-swapped control: 5 wins / 1 loss; vs. nearest uniform rate: 5 / 2. In another experiment a constant horizon did as well as the physical-aware controller | **Not established** (p ≥ 0.2) |
| Graceful behaviour under overload | Latency-aware step reduction halved the time without a valid action (88 → 41 ticks per episode); stale results rejected, no invalid command | Solid for buffer coverage; **no** measurable task-success gain |
| Non-blocking control path | Longest control step 9.5 ms while inferences took 1.3–4.1 s | Solid on this CPU; not hard real-time (one ~100 ms overrun under load) |
| Scene-change-triggered refresh | Detected 5 of 5 injected object displacements within the first visible frame | Detection solid; **no task benefit** (the policy fails on displaced objects regardless) |
| Cross-observation vision reuse | 0 hits in 114 lookups on LIBERO | Negative |
| Multi-robot scheduling | Deadline policy keeps 4 simulated streams alive where FIFO starves 45% | **Synthetic simulation only** |

What this means commercially, stated plainly:

* The measured advantage that is solid today is **"the same checkpoint at a
  quarter to a third of the inference work, with no loss detectable at this
  sample size"**, plus defined behaviour under overload. Most of that advantage
  comes from two constants (fewer steps, a longer refresh horizon) that a
  customer could also set by hand once they have a harness that tells them it is
  safe. The harness is, at present, the more defensible part.
* The differentiating claim — that *physical* signals let the runtime save more
  than a well-chosen constant — is **not demonstrated**. The data lean that way
  in one experiment and not in another.
* The quantity `r` in §6 is therefore between 0.3 and 0.75 for this checkpoint on
  this benchmark, with the upper part of that range resting on a success
  equivalence known only to ±20 percentage points, and with no evidence yet that
  it holds on a GPU, where inference may not be the binding constraint.
<!-- MEASURED:END -->

## 6. Commercial ROI under transparent assumptions

All figures in this section are **illustrative assumptions**, not measurements and
not customer data. The purpose is to show which measured quantity the business
case depends on.

Let `r` be the fraction of expensive inference work removed at equal task success
(the quantity measured in §5), and assume inference is the binding load on the
robot's accelerator.

* *Hardware tier.* If a cell needs accelerator tier A at fixed budgets and
  `(1 − r)` of that work fits tier B, the saving is `price(A) − price(B)` per
  robot. Assumed example: a 400 USD difference between two edge modules × 500
  robots = 200k USD one-off. This only materialises if the reduction is large
  enough to cross a tier boundary, which depends on the customer's other
  workloads.
* *Shared accelerator.* If one GPU serves `k` arms at fixed budgets, it serves up
  to `k / (1 − r)` at reduced work, if the latency deadline still holds. With
  `r = 0.3` that is 1.43×, i.e. 10 arms instead of 7 per GPU.
* *Throughput.* Fewer stalls under contention raise completed tasks per hour.
  The value is the customer's margin per task × recovered tasks. Not estimated:
  this repository has no real-robot throughput data.
* *Energy.* On battery robots, accelerator energy scales roughly with executed
  work; saving `r` of inference work saves at most `r` of the accelerator's share
  of the power budget. Not measured here.

What would falsify the case: `r` shrinking toward zero on faster hardware (where
inference is no longer binding), on harder tasks (where every phase is
"critical"), or when success drops outside the customer's tolerance.

## 7. Possible pricing models

* **Per-robot runtime licence** (annual), tiered by control stack — simplest to
  explain, aligns with fleet growth. Assumption: tens to low hundreds of USD per
  robot-year is the plausible range for infrastructure middleware; unvalidated.
* **Open-core**: runtime, SmolVLA adapter and benchmark tool open; commercial
  adapters (TensorRT, Jetson, ROS 2 packaging), fleet telemetry and support paid.
  Likely necessary for adoption, because robotics teams will not put a closed
  component in their control path without inspecting it.
* **Design-partner services** first: paid integration + benchmark report on the
  customer's task, converting to a licence.

## 8. Integration difficulty

* Low: Python policy stacks built on LeRobot — one adapter, already written for
  SmolVLA.
* Medium: other flow/diffusion policies. The adapter must expose the step count
  and chunk timing (≈200 lines for SmolVLA). Models with discrete-token action
  heads have no step knob; only refresh timing applies.
* High: production stacks in C++/ROS 2 with real-time executors. Requires a native
  core, a ROS 2 node and safety review. None of this exists yet.
* The physical-signal thresholds are task- and embodiment-specific. A calibration
  procedure exists (`krt-bench signals`), but every new task family currently
  needs it. This is the main integration cost and risk.

## 9. Competitive risks

* **Absorption.** LeRobot, openpi or Isaac can add adaptive step counts and
  adaptive queue thresholds in a release. The idea is public and simple.
* **Model progress.** One-step / distilled / consistency policies remove the
  denoising knob. Training-time RTC removes part of the delay problem. Faster
  edge hardware removes the pressure.
* **Model-side adaptivity.** Learned estimators inside the policy (VLA-Rolling,
  GeCO, early exit) can outperform external heuristics because they see model
  internals.
* **Native runtimes.** vla.cpp or Embodied.cpp can add scheduling and own the
  deployment surface.

## 10. Early design-partner validation plan

1. Reproduce the benchmark on a GPU (RTX 4090) and a Jetson: does any saving
   survive when inference is 10–20× faster? (`make gpu-benchmark`; not yet run.)
2. Extend to LIBERO-10 / long-horizon and one contact-rich benchmark, at ≥100
   episodes per configuration, to bound the success difference within ±2 pp.
3. Three design partners with a real manipulation cell on edge hardware. For each:
   baseline their current settings, run the calibration, report compute, stall
   rate and task success for two weeks. Success criterion agreed in advance (for
   example: ≥25% fewer inferences, no measurable success loss, zero additional
   safety stops).
4. Kill criteria: if on partner hardware the saving is <10% or the calibration
   takes more than a day per task family, the product as specified is not viable.

## 11. Long-term defensibility

Weak on the algorithm, potentially real on:

* **Evidence and data**: a growing corpus of per-task signal/error measurements
  across models and embodiments, which is what a learned scheduler would be
  trained on.
* **Breadth of adapters and hardware backends** behind one scheduling interface.
* **Trust in the control path**: test suites, deterministic replay, safety
  reviews, and a record of running on real robots. This takes time to earn and is
  hard to copy quickly.
* **Fleet-level observability** built on the per-inference telemetry.

## 12. Reasons the startup might fail

1. The saving is real only where compute is binding; on current GPUs it may not
   be, and edge hardware keeps getting faster.
2. The measured advantage is small or statistically unresolved (see §5) and
   customers do not pay for single-digit improvements in the control path.
3. Framework maintainers ship the same feature for free.
4. Hand-set, task-specific thresholds do not generalise; a learned scheduler
   needs data nobody has yet.
5. Policies move to one-step generation or on-robot world models with different
   bottlenecks.
6. Selling middleware into safety-relevant control loops is slow; sales cycles
   outlast the runway.
7. The team would need real robots, GPU hardware and robotics customers; this
   repository has none of the three.
