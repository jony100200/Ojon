# Ojon Learn — Hardware-First Model Improvement Roadmap

**Research cut-off:** 2026-10-27  
**Product:** [Ojon](https://github.com/jony100200/Ojon)  
**Status:** proposed roadmap, NOT a claim these features have been implemented  
**First user and acceptance machine:** Kalponic Studio developer workstation (Windows; RTX 4060 Ti 16 GB, Intel i5-6600K, 24 GB DDR4)  
**Companion documents:** [Dataset registry](DATASET_REGISTRY.md) and [first-user experiments / acceptance](FIRST_USER_ACCEPTANCE.md)

## 0. Product goal and non-goals

One desktop application to **make a usable AI model more capable, then smaller**, while collecting only validated learning signals from model use in opted-in applications.

    Import model -> establish baseline -> collect or import examples
       -> verify -> deduplicate -> curate -> SFT/LoRA or distillation
       -> re-evaluate -> quantize via Ojon -> re-evaluate -> deploy
       -> collect verified new experience -> repeat

- **Primary metric:** success on a representative held-out user task suite, under an explicit memory/latency budget.
- **Secondary metrics:** model bytes, total working RAM/VRAM, tool-call validity, long-horizon task completion, speed, stability, and regressions.
- **Not the goal:** highest benchmark headline, synthetic datasets without verification, a gigantic foundation-model training stack, or pretending quantization adds intelligence.
- **Important boundary:** the newest published teacher or dataset is a candidate, not truth; keep versioned provenance and measure every gain locally.
- **Preserve current Ojon behaviour.** This roadmap must extend rather than replace native Rust quantization, the existing settings, evaluation and Kathamo GUI.

## 1. First-customer hardware budget (hard constraints)

First-user inventory to confirm at runtime: RTX 4060 Ti with **16 GB VRAM**, **24 GB** system RAM, an older **i5-6600K**, Windows, and available disk capacity **measured**, not assumed.

| Family | Inference on this machine | Fine-tuning verdict |
|---|---|---|
| 0.6B–2B | Easy with supported runtime | Local SFT/BF16 LoRA; excellent test candidates |
| 4B | Comfortable with GGUF/compatible runner | **Default local improvement target**; BF16 LoRA candidate |
| 7B–9B | Quantized inference practical | **Careful architecture-specific QLoRA**, or rented GPU for BF16 LoRA |
| 14B | Quantized inference often viable | Experimental partial/offloaded LoRA; not a first milestone |
| 27B dense | Low-bit CPU+GPU offload, reduced context | Remote training preferable |
| 35B–40B MoE | Low-bit with substantial CPU+GPU offload; avoid assuming 3B active means 3B stored | Remote 80GB-class GPU for BF16 LoRA trials; architecture-specific preflight mandatory |
| 35B–40B dense | Quantized inference with memory pressure/offload | Remote 48–80GB QLoRA or larger BF16 setup; not 16GB local training |

**Published resource examples, not peak guarantees:**

- The official Ornith-1.5-35B-A3B Q4_K_M GGUF file is **21.7 GB** before KV cache/runtime buffers. Its full model needs more memory than 16GB local VRAM. Source: https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF/tree/main
- Unsloth Qwen3.5 fine-tuning lists **4B BF16 LoRA ~10GB**, **9B BF16 LoRA ~22GB**, **35B-A3B BF16 LoRA ~74GB**; its documentation recommends BF16 approaches for Qwen3.5 MoE rather than forcing 4-bit training. Source: https://unsloth.ai/docs/models/qwen3.5/fine-tune
- Unsloth generic minimums list **40B dense QLoRA ~30GB**, BF16 LoRA ~96GB; actual long contexts, vision modules, optimizers and OS overhead can exceed them. Source: https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements
- Quantized inference is not LoRA training. 35B MoE **3B active** means fewer active weights for arithmetic per token, **not** 3B resident weight memory.
- The Intel i5-6600K and 24GB RAM are real bottlenecks for data prep, offloading, and executing many testing containers. Use bounded worker counts and streamed datasets; do not promise interactive 35B offload speeds.

**Operator defaults for this machine:** inspect free VRAM/RAM/disk, reserve at least 2–3GB VRAM headroom for Windows/runtime where practical, train one experiment at a time, microbatch 1, gradient accumulation, 1,024–2,048 token pilot context, gradient checkpointing, no surprise downloads, no CPU-resident 35B checkpoint load, hard preflight and job budget. Benchmark 4K+ context only after a pilot fits. Use an isolated Python/CUDA environment and keep Ojon's Rust binary lean.

**Hardware decision:** user can personally train and compare 2B/4B first; collect and curate data for 35B **now**, train 35B/40B later on an explicitly chosen rented GPU or a future hardware upgrade. Do not gate collector development on obtaining an 80GB GPU.

## 2. What advanced community projects actually do

### 2.1 Ornith-1.5: jointly improve tasks, scaffolds and solutions

[Ornith self-improvement description](https://ornith.ai/ornith_1_5.html) reports three coupled policies:

1. **Task proposer:** generate new, clear, executable tasks at the student model's frontier.
2. **Harness/scaffold proposer:** construct useful tool interfaces, procedures and checkers, then check that they match the task.
3. **Solver:** generate trajectories, execute tasks, learn from actual outcomes.

Its reward incorporates **validity**, **difficulty** (authors describe a target roughly 20% solver success) and **novelty**, plus harness fidelity and resistance to reward hacking. This is a promising *research mechanism*, not an available one-click training recipe. Ornith's scores are **developer-reported**, typically using much larger infrastructure/harness contexts than our single-PC tests; do not promise to replicate them by downloading one dataset.

Our attainable adaptation: start with fixed executable tasks and existing test harnesses; measure student success rates; prioritize hard-but-solvable tasks; synthesize variations only when independently verified; introduce scaffold generation **after** the verifier is trustworthy.

### 2.2 SI2CA: real before/after 35B student training evidence

[SI2CA](https://github.com/Self-Improving-Coding-Agents/SI2CA) releases [32,340 trajectories](https://huggingface.co/datasets/Self-Improving-Coding-Agents/SI2CA-Training-Trajectories) over 10,780 SWE tasks, produced with a Qwen3.5-122B teacher and three action-selection strategies: standard, full self-judgement, selective self-judgement. Authors report Qwen3.5-35B-A3B-Base at **24.0 SWE-bench Verified**, **57.4 after standard trajectory SFT**, and **59.6 after self-judgement trajectory SFT**, under the documented evaluation. Those are **paper results**, not Ojon tests or a general 35B uplift guarantee. Controlled curation choice alone contributed +2.2 Verified and +4.2 Pro (Python) points against standard trajectory SFT.

The lesson: trajectory **selection/verification and harness behaviour** can matter as much as dataset scale. Import both resolved/unresolved traces, but do not train the final assistant to imitate failed answers. Use failed traces for contrastive analysis, recovery and preference comparisons only where validity is established.

### 2.3 Orchard SWE: productive segments and iterative RL

[Orchard](https://arxiv.org/abs/2605.15040) reports 107K teacher-generated SWE trajectories and uses credit-assignment SFT plus adaptive rollouts, taking a Qwen3-30B-A3B-Thinking student to reported 64.3% SWE-bench Verified after SFT and 67.5% after SFT+RL. Model, harness, sample selection, environment and evaluation all matter. Potential later research route for Ojon: extract useful *verified substeps* from failed overall trajectories instead of discarding every partial success.

### 2.4 On-policy and stepwise distillation

- [TRL DistillationTrainer](https://huggingface.co/docs/trl/distillation_trainer) learns from **student-generated trajectories** with teacher feedback; it supports agent tool loops and masks tool-result tokens from training loss.
- [SOD](https://arxiv.org/abs/2605.07725) warns that erroneous tool steps propagate and advocates step-level weighting.
- [Small-scale OPD limits](https://arxiv.org/abs/2609.37326) shows that smaller students can struggle to **stop reasoning** even when they learn to solve; measure termination, latency and malformed tool calls, not accuracy alone.
- [Reward-aligned OPD](https://arxiv.org/abs/2608.27960) notes teacher gradients may conflict with correctness signals; prefer verified outcomes.
- [DistillKit](https://github.com/arcee-ai/DistillKit) provides packed teacher probability datasets and offline/online approaches; cross-tokenizer alignment and memory preflight are mandatory.

**Practical ordering:** verified SFT -> error-focused SFT -> chosen/rejected preference experiments -> on-policy/stepwise distillation -> GRPO with executable reward -> self-generated tasks. Never implement custom RL math unless existing stable trainer cannot handle the validated use case.

## 3. Model selection by jobs-to-be-done

The best local model is whichever **passes our workloads** while fitting the hardware, not the largest parameter count.

**Pilot track A: affordable small model**
- Start with Qwen3.5-4B **BF16 LoRA** (first preference) plus an unchanged baseline.
- Compare 2B for routing/simple tools, 4B for broad assistant utility, and a 7B/8B model with supported QLoRA for efficient code SFT.
- Optionally compare an existing strong 9B model such as [Ornith-1.5-9B](https://huggingface.co/ornith-ai/Ornith-1.5-9B) **without training it** first. Strong weights may already beat a weaker 4B after lots of expensive experimentation.
- Keep prompt templates, thinking-mode and generation settings matched during before/after testing.

**Scale-up track B: 35B–40B**
- Evaluate, do **not automatically train locally**, [Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B), a matching [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B), and any locally supported 35B/40B target.
- MoE and dense models must have **different training plans**; the MoE router, shared experts, active experts, special hybrid-attention layers, vision encoder and output heads cannot all be treated as identical linear layers.
- For MoE LoRA, start with a backend-validated selective adapter and preserve routing quality; change routers only as a separately gated experiment.
- Reserve cloud resources only after Ojon curates enough **verified, task-matched** training examples and a held-out suite, with full checkpoint provenance and reproducible config.
- Validate BF16/LoRA merge quality before Ojon quantization; compare 8/6/5/4/3-bit candidates and measure true task regressions.
- 40B dense QLoRA published minimum ~30GB is **not** a promise 32GB devices suffice; target 48–80GB GPU for a comfortable controlled pilot, and substantially more for BF16-LoRA.

**Long-term:** support image/video/audio/utility training through **family-specific adapters**, not a universal LLM text-data trainer. First shippable loop is LLM/code/tool use.

## 4. Ojon architecture — extend, do not clone

Current repo inspection (2026-10-27):

- [src/eval/distill.rs](../../src/eval/distill.rs): JSON/JSONL parsing, knowledge-decay and liveness audit, distill-plan and retrieval. It is **data curation, not teacher-student training**.
- [src/workflows/train.rs](../../src/workflows/train.rs): guarded local LLaMA-Factory **LoRA/QLoRA SFT launcher**, only; **no distillation engine**.
- [src/eval/task_eval.rs](../../src/eval/task_eval.rs): task evaluation; extend adapters rather than inventing offline scores.
- [src/core/settings.rs](../../src/core/settings.rs): portable per-user settings and validation.
- [src/frontend/](../../src/frontend/): native Kathamo UI; preserve existing navigation/optimisation features.
- Current README calls full fine-tuning/distillation a planned/experimental capability, correctly.

### Proposed modules (implementation future work; avoid unverified status claims)

- **learn/collector:** local, explicitly enabled API relay (OpenAI-compatible request/stream forwarding) and Rust/Python/C# SDK event ingestion. No stealth interception.
- **learn/store:** versioned append-only task/event records, atomic journaling, bounded disk usage, rotation, replay, subject-level deletion, redaction, project/model identities and resume markers.
- **learn/verification:** independent task success checks: tests, compilation, JSON schema validity, exact answer, validated human correction; include confidence, evidence hash and tool environment. Unknown remains unknown.
- **learn/curation:** streaming HF Parquet/JSONL loaders, splits, minhash/dedup, source provenance, per-project competency profiles, filter malformed/chopped/off-topic/contaminated entries, inspect accepted/rejected samples.
- **learn/train:** plug-in bridge to LLaMA-Factory/Unsloth/TR L (TRL), LoRA/SFT first, then DPO and on-policy distillation; no separate Rust gradient engine.
- **learn/taskgen:** use student failures and repository context to propose tasks; sandbox verify them; learn task difficulty from observed rollouts.
- **learn/experiment:** immutable model/dataset/training/eval/runtime hashes; before-after comparison; reproducible receipts; promotion/rollback.
- **learn/gui:** Kathamo wizard Collect -> Data -> Train -> Results, with real progress and resource gating.

Use existing confirmation, non-overwrite, settings, device preflight, logging and output conventions. An opt-in local relay should bind loopback only, show target host and model, and not silently proxy remote chats. Collect streaming tool/event structure without flattening it into one user/assistant pair. Protect tool output boundaries and do not train on untrusted tool output as if it were an assistant response.

### Minimum canonical event schema

Every collected event must preserve:
- schema_version, collection_id, project_id, session_id, trace_id, event_id/sequence, timestamp
- original model ID, revision/hash, quant variant, adapter, chat template, generation parameters, runtime/harness version
- role, content or private blob reference, tool call ID/args, corresponding tool result, truncated flag
- action and test outcome: passed/failed/unknown, validator identity, evidence hash, human-corrected flag, source
- privacy/consent flag and retention policy, dedup fingerprint and original dataset provenance
- session-level terminal status and **separate** outcome confidence

Never infer success from eloquent answers, exit code alone, a user continuing the conversation, or self-reported completion.

## 5. Data flywheel (quality over quantity)

**Source priority:**
1. **First-party verified app tasks:** coding patches that pass tests, valid tool calls that meet goal, successful document conversion, reproducible results. Highest task alignment.
2. **Verified SWE trajectories:** SI2CA, NVIDIA Open-SWE-Traces, SWE-smith, Nemotron SWE, source-specific resolved flags. Quality and contamination audits required.
3. **Credible observed user agent traces:** SWE-chat and Share-Codex where metadata verifies the agent/model; a Claude Code harness can be driven by a different teacher model.
4. **Diverse reasoning/instruction data:** OpenThoughts3, carefully targeted Nemotron Chat, math/code tasks with independently executable answers.
5. **Unverified named GPT/Claude/Fable dumps:** quarantine until actual teacher provenance, utility and quality are demonstrated. Never equate a name in a dataset title with a provenance guarantee.

**Do not download everything.** Use streaming samples and category quotas; preserve official source revision and hash, normalized JSONL plus full tool traces in sidecar files; show local disk impact before writing.

**First pilot dataset design (a hypothesis, not an optimal formula):**
- 2,000–5,000 HIGH-confidence examples selected from verified coding/task logs and SI2CA success cases.
- Add 1,000–3,000 instruction/tool examples, 500–1,000 diverse reasoning samples.
- Hold out distinct **repositories, users/sessions, tasks** where feasible; never split steps from the same trajectory across train and test.
- Track provenance by task identity across different dataset mirrors: a mere random row split is not sufficient.
- Balance short/direct answers and longer reasoning modes to prevent verbosity or stopping regressions.
- Compare **small verified** dataset with **larger lower-filter** dataset to establish whether our quality scoring helps.

**Per-task improvement cycle:**
1. Run current student N times in the same deterministic validated harness.
2. Estimate empirical success rate, reject invalid graders.
3. Queue tasks near, but not over, the student's capability frontier; 20% is Ornith's example target, **not a universal constant**.
4. For failed but solvable cases, ask a stronger available teacher for an executable attempt, or explore candidate repairs; run hidden tests.
5. Retain **successful** trajectories; store chosen/rejected contrast pairs only if preferences are meaningful and verified.
6. Fine-tune on a stable approved dataset; evaluate on sequestered tasks; quantify regressions.
7. Deploy only after promotion gate; retain rollback and previous evaluation receipts.
8. Repeat on new mistakes. Never silently modify production weights during inference.

## 6. Training strategy & honest resource plan

**Local 4B pilot (default):**
- Original HF checkpoint, pinned revision; experimental environment containing tested Unsloth/Transformers/PEFT or LLaMA-Factory.
- BF16 LoRA if a real preflight confirms within 16GB VRAM. Begin 1,024–2,048 token sequence length, microbatch 1, gradient accumulation, checkpointing; try LoRA rank 8/16, small learning-rate sweep instead of a fixed magical recipe.
- Keep separate untouched baseline + validation + held-out test; archive adapter and training metadata.
- Record *real* max observed VRAM and OS memory, iteration/s, tokens/s, convergence, seed, OOM and restart behaviour.
- If an architecture cannot fit BF16 LoRA, do not silently force unsupported QLoRA; use a compatible smaller model or a trainer/device path with measured fit.

**9B pilot:**
- Either benchmark quantized 9B without training, or use a 7B/8B architecture proven to support QLoRA on 16GB.
- For Qwen3.5-9B, BF16 LoRA published ~22GB **exceeds 16GB**; avoid proclaiming local BF16 feasible.
- Short LoRA runs or remote BF16 experiments only after local 4B pilot succeeds.

**35B MoE pilot:**
- First curate/inspect/evaluate locally using compressed model and a tiny context; choose whether inference itself is acceptable on 24GB host RAM.
- No assumption 35B full weights can be loaded for training here. Use GPU instance with measured **>=80GB VRAM class** where the exact trainer/model combination supports ~74GB BF16 LoRA baseline; allow additional headroom as needed for context/vision and memory fragmentation.
- Train text-only adapter first, frozen nontext modules, controlled module targeting; validate router functionality and full prompt template.
- Export adapter + merge in an environment that really has memory for it; Ojon local or hosted pipeline quantizes compatible merged checkpoint; evaluate on first-user apps.

**40B dense pilot:** minimum generic 4-bit QLoRA 30GB published, so specify an instance with **>=48GB**, preferably more headroom for longer context; BF16 LoRA minima ~96GB, choose **>=128GB** budget if targeting it. Not equivalent to sparse 35B MoE. Cloud resource use is explicitly user-approved with expenditure boundaries and cancel/resume support.

**Ablation matrix:** baseline; filtered vs unfiltered SFT; successes only vs repaired; short vs mixed reasoning; SFT vs preference; self-judgement vs baseline; teacher vs no teacher; quant Q8/Q6/Q5/Q4/Q3; with/without imatrix. One experimental change at a time for interpretable receipts.

## 7. Evaluation and anti-regression requirements

**Never score a dataset on itself.** For coding, use repository-disjoint tasks with tests and clear contamination filters. Also include actual user tasks from local apps and private held-out checks.

Small-first local suite:
- 30–50 **new** executable coding tasks (Rust, C#, Python), 20+ tool use/multi-turn tasks, 20+ reasoning/exact-answer tasks, plus instruction-format checks.
- Include short-answer stop tests, malformed/unsafe shell command attempts, structured JSON schemas, rollback safety and corrupted-input tests.
- Prefer at least 3 independently sampled rollouts for small suites when feasible, then report sample counts and confidence intervals.
- Track task pass%, pass@1 and repeatability; tool correctness and call sequencing; end-state success; tokens/step, TTFT, generation throughput, memory and disk bytes.
- A model only improves if matched before/after trials show real benefits on target tasks and no unacceptable regressions. Establish first-user quality thresholds in [acceptance plan](FIRST_USER_ACCEPTANCE.md).
- Separate instruction quality, numerical correctness, code compilation, tool success and general conversational quality. Do not compress to one inflated score.
- Evaluate **the fine-tuned BF16/adapter** and **every proposed compressed deployment** independently.
- Check that runtime actually loads each output in its target backend. A smaller file is not a usable model until verified.

Specific reward-hacking controls:
- isolate verifier from model's editable files and network;
- never show gold patches or secret hidden tests to the solving policy;
- consider version-control history and external retrieval contamination;
- run benign canary failures to ensure a grader rejects bad solutions;
- model-authored test results cannot substitute for trusted runner receipts.

## 8. Milestone roadmap and go/no-go gates

### M0 — Honest hardware / baseline audit
Deliverable: exact hardware/resource snapshot, 4B and strong 9B quant baseline in Ojon, tool compatibility report, dataset format inventory. **Gate:** repeatable tests and true numbers, not placeholder scores.

### M1 — Learning collector & data store
Implement opt-in OpenAI-compatible local relay + SDK/JSONL ingestion, multi-turn/streaming tool traces, atomic storage, redaction, privacy and deletion, model revision provenance. **Gate:** enabled collects real requests, disabled stores nothing, no secrets captured by default.

### M2 — Curated HF and first-party dataset builder
Implement pinned streaming imports, schema conversion, source manifests, validator results, dedup, leak/contamination checks, project-aware sample rankings and reproducible frozen splits. **Gate:** 1,000+ valid examples; failures do not become SFT gold; can reproduce hashes and export.

### M3 — Real 4B fine-tuning vertical slice
Use Ojon's guarded LLaMA-Factory adapter; add compatible Unsloth/TRL as needed. Execute LoRA locally and produce merged/adapter artifact, validated SFT dataset, full resource metrics, before/after test. **Gate:** completed real training and repeated held-out eval; if no gain, document a negative result and iterate data rather than calling milestone a success.

### M4 — Quantization-aware promotion
Reuse current Ojon candidate sweep, imatrix, runtime tests; compare BF16/adapted vs quantized and untouched baseline on identical tests. **Gate:** promote only a loadable candidate meeting first-user quality and memory thresholds; keep rollback.

### M5 — Self-improvement pilot
Add verifier-backed task generation, sampling at measured frontier, trajectory self-judgement (without leaking gold to solver), verified repair and failure mining. Try DPO/TRL distillation after SFT established. **Gate:** same-budget controlled ablation beats standard curated SFT or honestly reports no measurable benefit.

### M6 — 35B–40B scale-up
Recreate a SI2CA-style 35B MoE experiment on available remote GPU(s); optionally 40B dense as distinct route. Keep first-user local Ojon collection, orchestration, offloaded inference evaluation and compressed deployment. **Gate:** job reproducibility, demonstrable target-task gain over exact unchanged checkpoint, quantized local loadability, documented compute cost, revertible deployment.

### M7 — Beta user experience
Polish Kathamo Collect / Datasets / Train / Results; first-user presets and one-click **plan** (not unapproved automatic execution); accessible logs; beginner glossary; stability tests and recovery. **Gate:** first user completes a full 4B data -> train -> validate -> quantize -> deploy loop without opening/editing source files.

## 9. Suggested Ojon interface

- **Learn > Overview:** connected apps, data storage, verified task count, latest accepted model, exact experiment result.
- **Collect:** explicit per-app connection and live capture status, redaction preview, stop/delete/retention controls.
- **Datasets:** HF search and user paths, verified provenance, samples, accepted/rejected reasons, train/test contamination status.
- **Train:** student/teacher, actual RAM/VRAM preflight, compatible methods, config preview, budget/cancel/checkpoint, no false 35B training readiness.
- **Compare:** model version A/B, task suites, accuracy, tool success, response length, speed, RAM/VRAM, output size.
- **Promote:** export GGUF/native supported target, rollback, optional app integration; visible reason for pass/fail.

User intent is **improve model**, not "run GRPO"; hide algorithm complexity until Advanced Mode. Retain all meaningful quality and safety gates.

## 10. Anti-patterns to explicitly reject

1. Treating 35B-A3B as a 3B GPU-memory model.
2. Fine-tuning arbitrary 35B/40B weights on 16GB without proof of actual fit.
3. Assuming dataset title "GPT", "Claude", "Fable", "Codex" proves teacher origin.
4. Training from every app response, or treating user silence as a pass.
5. Mistaking ChatGPT/Codex/Claude-Code **harness** for a model generator.
6. Training on benchmarks then claiming results on those benchmarks.
7. Trying reinforcement learning before having independent verifiers.
8. Overwriting baseline model or changing running production weights.
9. Merging LoRA and quantizing without a full before/after score.
10. Universal preprocessing that discards tool order, model modality or verified outcome.

## 11. Primary research & tools

- Ornith-1.5 self-improvement: https://ornith.ai/ornith_1_5.html
- Ornith model files: https://huggingface.co/ornith-ai/models
- SI2CA source: https://github.com/Self-Improving-Coding-Agents/SI2CA
- SI2CA trajectory dataset: https://huggingface.co/datasets/Self-Improving-Coding-Agents/SI2CA-Training-Trajectories
- Orchard: https://arxiv.org/abs/2605.15040
- OPEN-SWE-TRACES paper: https://aclanthology.org/2026.findings-emnlp.131/
- Open-SWE-Traces HF, versioned (2026-10-26 update): https://huggingface.co/datasets/nvidia/Open-SWE-Traces
- SWE-chat provenance: https://huggingface.co/datasets/SALT-NLP/SWE-chat
- TRL DistillationTrainer: https://huggingface.co/docs/trl/distillation_trainer
- SOD step-wise distillation: https://arxiv.org/abs/2605.07725
- Reward-aligned on-policy distillation: https://arxiv.org/abs/2608.27960
- Small-scale stopping limitation: https://arxiv.org/abs/2609.37326
- Unsloth Qwen3.5 training: https://unsloth.ai/docs/models/qwen3.5/fine-tune
- Unsloth hardware minimums: https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements
- LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory
- DistillKit: https://github.com/arcee-ai/DistillKit
- SWE-bench-Live verification: https://swe-bench-live.github.io/

**Bottom line:** use the existing Ojon foundation to ship a measurable **4B first-user success**, collect verified experience for every project, and only then scale the same curation/evaluation discipline to 35B–40B on realistic training hardware. Quantization makes the improved checkpoint usable locally; it does not replace post-training.
