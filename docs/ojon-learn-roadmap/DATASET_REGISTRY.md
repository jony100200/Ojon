# Ojon Learn — High-Signal Dataset Registry

**Research cut-off:** 2026-10-27. **Purpose:** sources of verifiable training examples, not a recommendation to download or mix all examples blindly. Companion to [roadmap](README.md) and [acceptance](FIRST_USER_ACCEPTANCE.md).

## Selection principle

The best dataset **for Ojon** is one that improves a local student's results on *held-out actual tasks*. Dataset marketing size, model names and teacher claims are weak indicators. Store model identity, dataset revision/hash, source task, harness, outcome, test evidence, schema map, provenance confidence and exact filtering rules.

Levels:
- **A — verifiable and reproducible:** source contains executable task, trajectories, explicit resolution/test signal and enough provenance to check.
- **B — reputable training source:** useful documented instruction/reasoning/agent content, but must verify at least a sample and isolate benchmark overlaps.
- **C — experimental/unverified:** community GPT, Claude, Fable or "thinking" dumps without independently established provenance; quarantine, sample-check and benchmark before promotion.

A label is a **dataset quality review status**, not a guarantee that every row is correct.

## Must-evaluate first: coding agents and 35B distillation

| Priority | Dataset / URL | Why it matters | Import rule |
|---|---|---|---|
| A | [SI2CA-Training-Trajectories](https://huggingface.co/datasets/Self-Improving-Coding-Agents/SI2CA-Training-Trajectories) | **32,340** Qwen3.5-122B-generated trajectories for 10,780 executable SWE tasks with test reward, full self-judgement and a discovered-strategy variant. Associated paper actually fine-tuned 35B-A3B. | Select verified resolution, stratify source repository, compare standard vs judgement at same token budget; do not expose gold patches to student. |
| A/B | [Open-SWE-Traces](https://huggingface.co/datasets/nvidia/Open-SWE-Traces) | 2026-10-26 v2 includes Qwen3.8-Flash-Next rollouts in several harnesses; earlier release also contains MiniMax, DeepSeek, Qwen families. Tool history + outcome. | Freeze revision; keep teacher identity distinct from *harness* name; prefer validated resolved entries; audit patch and tests. |
| B | [Nemotron-SFT-SWE-v3](https://huggingface.co/datasets/nvidia/Nemotron-SFT-SWE-v3) | **237,970** SWE trajectories; practical large-scale repo-navigation/coding instruction tuning. | Stream selected rows, dedup against Open-SWE, check tasks and tool schema. |
| A/B | [SWE-smith trajectories](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories) | Executable synthetic bug-fix tasks, relevant to local verifiable task generation. | Use task IDs; keep entire task family and repo in same split; verify code patches. |
| B | [SWE-Hero OpenHands](https://huggingface.co/datasets/nvidia/SWE-Hero-openhands-trajectories) | Execution-derived SWE trajectories and stronger teacher demonstrations. | Import only if source, outcome and tool transcript survive format conversion. |
| B | [AgentTrove](https://huggingface.co/datasets/open-thoughts/AgentTrove) | ~1.7M traces merged from 219 datasets; broad tool, terminal and agent behaviour. | Never treat collection size as unique tasks; group by original dataset and deduplicate. |

**Crucial 35B replication candidate:** SI2CA official https://github.com/Self-Improving-Coding-Agents/SI2CA with fixed mini-SWE-agent / bash harness. Its documentation reports controlled Qwen3.5-35B-A3B improvements on SWE-bench after Qwen3.5-122B teacher trajectories. Reproduce *process* and validate locally; don't claim that the test scores are transferable to Ojon.

## Credible real GPT / Claude sessions (for breadth, not blindly accepted gold)

| Review | Dataset / URL | What it contains | Caveat |
|---|---|---|---|
| B | [SWE-chat](https://huggingface.co/datasets/SALT-NLP/SWE-chat) | ~17,968 developer coding sessions, 229,909 user prompts from Claude Code, Codex and other agents; original real-use sessions. | Gated; model and user attribution vary by session, outcomes are not uniformly proven; redact PII and retest representative patches. |
| B | [Share-Codex](https://huggingface.co/datasets/nmuendler/share-codex) | Publicly shared Codex and Claude Code sessions with tool calls. | Audit format and version; keep actual model metadata; not every finished session was successful. |
| B | [Trace Commons](https://huggingface.co/datasets/trace-commons/agent-traces) | Donated real Claude Code/Codex/Cursor/OpenCode-style traces. | Samples may be incomplete or duplicated; tag generator when supported. |
| B | [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) | Documented GPT-3.5/GPT-4 conversations. | General chat; many low-signal/unsafe/noisy dialogues; teacher output alone is not correctness. |
| B | [OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) | Broad GPT-4-heavy synthetic instruction mixture. | Source-by-source filtering; overlapping/legacy instruction sets. |
| B | [GPT4-LLM-Cleaned](https://huggingface.co/datasets/teknium/GPT4-LLM-Cleaned) | Smaller GPT-4 instruction set. | Narrower domain and older behaviour; benchmark on current needs. |
| B | [Nectar](https://huggingface.co/datasets/berkeley-nest/Nectar) | Ranked multi-response preference data scored by GPT-4. | Model-of-preference signal, not executable verifier; use in separate DPO experiment. |

**Do not conflate** an OpenAI-format JSON transcript, Claude-Code *harness*, or filename containing "Codex" with generation by a GPT or Claude model. Need per-row model ID and evidence where possible.

## Broader complementary sources, if user tasks reveal gaps

- [OpenThoughts3](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) — multi-domain reasoning/coding; teacher-generated; many samples per source question, dedup by question and test for exact correctness.
- [Nemotron Agentic v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2) — tool-use/planning examples; validate tool schema/role ordering.
- [Nemotron Chat v4](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Instruction-Following-Chat-v4) — instruction/conversation; check source/model provenance and domain balance.
- [OpenCodeReasoning-2](https://huggingface.co/datasets/nvidia/OpenCodeReasoning-2) — code and reasoning with verifiable problems; prioritize executable tests.
- [UltraData Agent 2609](https://huggingface.co/datasets/XiSSiX/UltraData-SFT-Agent-2609) — 2B-scale agent-oriented data; inspect data format and reward labels.
- [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) — math problems, verify final answers rather than copying long reasoning blindly.
- [DistillKit teacher packs](https://github.com/arcee-ai/DistillKit) — Qwen3-235B and DeepSeek V3/R1 saved probability distributions. This is a separate **logit distillation** path requiring token alignment, more memory, training backend support.

## Community GPT / Claude / Fable dumps — quarantined sources

Candidate IDs that appeared in research, **not approved as truthful teacher traces**:

- [Complete FABLE.5 traces](https://huggingface.co/datasets/Crownelius/Complete-FABLE.5-traces-2M)
- [Fable GPT mixed traces](https://huggingface.co/datasets/RESMP-DEV/Fable-GPT-5.5-Distillation-Traces)
- [GPT-5.6 labelled coding traces](https://huggingface.co/datasets/Crownelius/GPT-5.6-Sol-Luna-Terra-Traces)
- [Claude-labelled combined data](https://huggingface.co/datasets/agentlans/claude)
- [Claude-labelled reasoning](https://huggingface.co/datasets/agentlans/claude-reasoning)

The registry must display **claimed teacher**, **verified generator**, and **confidence** separately. Check random samples, exact duplicate fraction, leaked benchmarks and tool-call authenticity before admitting even a small portion.

## Ojon dataset pipeline specification

1. **Discover:** inspect HF dataset card, config/subsets, schema, sizes, revision/hash, tool fields, resolved/test signals, required access. No silent download.
2. **Preview:** show 10–50 diverse rows with source, labels, content length, tokens, tool call/result structure, PII indicators and dedup clusters.
3. **Stream:** bounded memory conversion from HF Parquet/JSONL to versioned Ojon event and training schema; record rejected rows + reasons.
4. **Verify:** run actual offline validators if possible; treat published success flag as **source-reported** unless re-run.
5. **Dedup:** exact text, normalized tasks/patches and near-duplicate clusters; never let different versions of the same task straddle held-out and train.
6. **Quality rank:** reward verification, relevance to target apps, concise successful trajectories, model/source confidence and diversity; penalize corrupted/repeated/mismatched data.
7. **Freeze:** build immutable sample IDs, source revision+SHA, deterministic seed, sorted source/task split manifest, training JSONL and dataset_info.json or trainer-specific map.
8. **Evaluate:** before/after on completely held-out new tasks, including first-party app-derived problems with checked ground truth.
9. **Report:** show raw/accepted/rejected record counts and storage bytes, task/repo coverage, overlap, validation pass rate and reason histogram.

**Recommended first 4B pilot:** 2–5K verified SWE/SI2CA examples + 1–3K carefully selected tool/instruction examples + 500–1K verified reasoning problems. Train and compare against both untouched baseline and a *similarly sized unfiltered sample* to prove whether curation helps. Scale data only when measured results justify it.

**Recommended 35B pilot:** using rented compute, evaluate controlled SFT of ~10K verified distinct SWE tasks and compare standard vs self-judgement variants (SI2CA). Report separate improvements for 35B **MoE** and 40B **dense**; do not mix their compute assumptions.

## Required manifest fields

    dataset_id, revision, sha256_or_hf_commit, configuration, split
    actual_generator, claimed_generator, harness, transcript_schema
    origin_task_id, source_repo, provenance_confidence
    source_reported_outcome, independently_verified_outcome
    validator_version, validation_receipt_hash, privacy_approval
    import_timestamp, filters, token_count, dedup_cluster_id
    train_validation_test_assignment, rejection_reasons

Source documentation:
- https://huggingface.co/datasets/Self-Improving-Coding-Agents/SI2CA-Training-Trajectories
- https://huggingface.co/datasets/nvidia/Open-SWE-Traces
- https://huggingface.co/datasets/nvidia/Nemotron-SFT-SWE-v3
- https://huggingface.co/datasets/SALT-NLP/SWE-chat
- https://huggingface.co/datasets/open-thoughts/AgentTrove

**Product stance:** Ojon should choose a *measured winner* per student model, not sell one universal pre-selected "best data" recipe.
