//! Flint MoE 256-expert pipeline, RoCo search, and promotion commands.

use crate::cli::common::{invalid, next_argument};
use ojon_backend::*;
use std::ffi::OsString;
use std::io;
use std::path::PathBuf;

pub fn handle(cmd: &str, mut args: impl Iterator<Item = OsString>) -> io::Result<bool> {
    match cmd {
        "flint-inspect" => {
            let path = PathBuf::from(next_argument(&mut args, "Flint GGUF file or directory")?);
            let mut json = false;
            while let Some(opt) = args.next() {
                if opt.to_str() == Some("--json") { json = true; }
            }
            let topology = ojon_backend::inspect_flint_moe(&path)?;
            if json {
                println!("{}", topology.as_json());
            } else {
                println!("Flint 256-Expert MoE Architecture:");
                println!("  model: {}", topology.model_path.display());
                println!("  file_size: {:.2} GiB", topology.file_bytes as f64 / 1073741824.0);
                println!("  total_tensors: {}", topology.total_tensors);
                println!("  blocks: {}", topology.block_count);
                println!("  experts: {} total, {} active per token, {} shared", topology.expert_count, topology.expert_used_count, topology.expert_shared_count);
                println!("  context_length: {} tokens", topology.context_length);
                println!("  router_tensors (frozen under RoCo): {}", topology.router_tensor_count);
                println!("  packed_sparse_experts: {}", topology.packed_expert_tensor_count);
                println!("  shared_experts: {}", topology.shared_expert_tensor_count);
                println!("  attention_tensors: {}", topology.attention_tensor_count);
            }
            Ok(true)
        }
        "flint-pipeline-plan" => {
            let path = PathBuf::from(next_argument(&mut args, "Flint GGUF file or directory")?);
            let mut json = false;
            while let Some(opt) = args.next() {
                if opt.to_str() == Some("--json") { json = true; }
            }
            let plan = ojon_backend::plan_flint_pipeline(&path)?;
            if json {
                println!("{}", plan.as_json());
            } else {
                println!("Flint Multi-Stage Optimization Pipeline Plan:");
                println!("  flint_directory: {}", plan.flint_dir.display());
                println!("  bf16_model: {}", plan.bf16_model.display());
                if let Some(im) = &plan.imatrix_path { println!("  imatrix: {}", im.display()); }
                if let Some(log) = &plan.baseline_logits { println!("  baseline_logits: {}", log.display()); }
                println!("  pending_candidates: {}", plan.pending_candidates.len());
                for (p, cand) in &plan.pending_candidates {
                    println!("    - {:?}: {}", p, cand.display());
                }
                println!("  target_hardware: {} ({} MiB free VRAM)", plan.hardware_fit.target_device, plan.hardware_fit.vram_free_mib);
                println!("  recommended_tier: {:?}", plan.hardware_fit.recommended_tier);
                println!("  reasoning: {}", plan.hardware_fit.reasoning);
                println!("  actions:");
                for (i, act) in plan.recommended_actions.iter().enumerate() {
                    println!("    {}. {}", i + 1, act);
                }
            }
            Ok(true)
        }
        "flint-evaluate" => {
            let candidate = PathBuf::from(next_argument(&mut args, "Flint pending candidate (.gguf.pending)")?);
            let mut ref_logits: Option<PathBuf> = None;
            let mut tool: Option<PathBuf> = None;
            let mut gpu_layers: Option<i32> = None;
            let mut json = false;

            while let Some(opt) = args.next() {
                match opt.to_str() {
                    Some("--reference-logits") => {
                        ref_logits = Some(PathBuf::from(next_argument(&mut args, "reference logits path")?));
                    }
                    Some("--tool") => {
                        tool = Some(PathBuf::from(next_argument(&mut args, "llama-perplexity path")?));
                    }
                    Some("--gpu-layers") => {
                        let val = next_argument(&mut args, "gpu layers")?;
                        gpu_layers = Some(val.to_str().unwrap_or("0").parse().unwrap_or(0));
                    }
                    Some("--json") => json = true,
                    _ => return Err(invalid(format!("unknown option: {}", opt.to_string_lossy()))),
                }
            }

            let reference = match ref_logits {
                Some(p) => p,
                None => {
                    let parent = candidate.parent().ok_or_else(|| invalid("candidate has no parent"))?;
                    let v2 = parent.join("Flint-BF16-eval-v2.logits");
                    if v2.is_file() {
                        v2
                    } else {
                        parent.join("Flint-BF16-eval.logits")
                    }
                }
            };

            let perplexity = match tool {
                Some(p) => p,
                None => discovered_path(BackendId::LlamaPerplexity)
                    .ok_or_else(|| invalid("llama-perplexity executable not found; install it or pass --tool"))?,
            };

            let receipt = ojon_backend::evaluate_flint_candidate_logits(
                &candidate,
                &reference,
                &perplexity,
                gpu_layers,
                &mut |line| {
                    if !json { eprintln!("{}", line); }
                },
            )?;

            if json {
                println!("{}", receipt.as_json());
            } else {
                println!("Flint Parity Evaluation Receipt:");
                println!("  candidate: {}", receipt.candidate_model_path.display());
                println!("  reference_logits: {}", receipt.reference_logits_path.display());
                println!("  evaluated_tokens: {}", receipt.evaluated_tokens);
                println!("  top1_token_agreement: {:.2}%", receipt.top1_token_agreement_pct);
                println!("  kl_divergence: {:.4}", receipt.kl_divergence);
                println!("  passes_negligible_loss_gate: {}", receipt.passes_negligible_loss_gate);
                println!("  certified_for_production: {}", receipt.certified_for_production);
            }
            Ok(true)
        }
        "flint-compare-logits" => {
            let reference = PathBuf::from(next_argument(&mut args, "reference _logits_ dump")?);
            let candidate = PathBuf::from(next_argument(&mut args, "candidate _logits_ dump")?);
            let mut json = false;
            while let Some(opt) = args.next() {
                match opt.to_str() {
                    Some("--json") => json = true,
                    _ => return Err(invalid(format!("unknown option: {}", opt.to_string_lossy()))),
                }
            }
            let cmp = ojon_backend::compare_flint_logits_files(&reference, &candidate)?;
            if json {
                println!("{}", cmp.as_json());
            } else {
                println!("Flint Direct Logits Comparison:");
                println!("  scored_tokens: {}", cmp.scored_tokens);
                println!("  vocab_size: {}", cmp.vocab_size);
                println!("  top1_token_agreement: {:.2}%", cmp.top1_token_agreement_pct);
                println!("  top5_overlap: {:.2}%", cmp.top5_overlap_pct);
                println!("  mean_kl_divergence: {:.6}", cmp.mean_kl_divergence);
                println!("  max_kl_divergence: {:.6}", cmp.max_kl_divergence);
            }
            Ok(true)
        }
        "flint-functional" => {
            let model = PathBuf::from(next_argument(&mut args, "Flint model (.gguf)")?);
            let mut tool: Option<PathBuf> = None;
            let mut gpu_layers = 0i32;
            let mut json = false;

            while let Some(opt) = args.next() {
                match opt.to_str() {
                    Some("--tool") => {
                        tool = Some(PathBuf::from(next_argument(&mut args, "llama-cli path")?));
                    }
                    Some("--gpu-layers") => {
                        let val = next_argument(&mut args, "gpu layers")?;
                        gpu_layers = val.to_str().unwrap_or("0").parse().unwrap_or(0);
                    }
                    Some("--json") => json = true,
                    _ => return Err(invalid(format!("unknown option: {}", opt.to_string_lossy()))),
                }
            }

            let cli = match tool {
                Some(p) => p,
                None => discovered_path(BackendId::LlamaCli)
                    .ok_or_else(|| invalid("llama-cli executable not found; pass --tool"))?,
            };

            let report = ojon_backend::run_flint_functional_suite(
                &model,
                &cli,
                gpu_layers,
                &mut |line| {
                    if !json { eprintln!("{}", line); }
                },
            )?;

            if json {
                println!("{}", report);
            } else {
                println!("Flint Functional Suite Result:");
                println!("  model: {}", report["model"]);
                println!("  all_passed: {}", report["all_passed"]);
                println!("  cases_passed: {}/{}", report["passed_cases"], report["total_cases"]);
            }
            Ok(true)
        }
        "flint-uncensor-plan" => {
            let model = PathBuf::from(next_argument(&mut args, "source Flint model path")?);
            let output = PathBuf::from(next_argument(&mut args, "output uncensored model path")?);
            let mut strength = 0.10f64;
            let mut blocks: Option<Vec<usize>> = None;
            let mut json = false;

            while let Some(opt) = args.next() {
                match opt.to_str() {
                    Some("--strength") => {
                        let val = next_argument(&mut args, "strength")?;
                        strength = val.to_str().unwrap_or("0.10").parse().unwrap_or(0.10);
                    }
                    Some("--blocks") => {
                        let val = next_argument(&mut args, "comma-separated blocks")?;
                        let val_str = val.to_str().unwrap_or("");
                        let parsed: Vec<usize> = val_str
                            .split(',')
                            .filter_map(|s| s.trim().parse().ok())
                            .collect();
                        blocks = Some(parsed);
                    }
                    Some("--json") => json = true,
                    _ => return Err(invalid(format!("unknown option: {}", opt.to_string_lossy()))),
                }
            }

            let plan = ojon_backend::plan_flint_uncensor(
                &model,
                &output,
                strength,
                blocks.as_deref(),
            )?;

            if json {
                println!("{}", plan.as_json());
            } else {
                println!("Flint MoE Uncensor Plan:");
                println!("  source_model: {}", plan.source_model.display());
                println!("  output_model: {}", plan.output_model.display());
                println!("  target_blocks: {:?}", plan.target_blocks);
                println!("  strength: {:.2}", plan.strength);
                println!("  paired_prompts: {}", plan.paired_prompts_dir.display());
                println!("  target_tensors: {} tensors", plan.target_tensors.len());
                for t in &plan.target_tensors {
                    println!("    - {}", t);
                }
                println!("  router_protection: ProQuant RoCo: all blk.N.ffn_gate_inp router matrices are 100% frozen");
            }
            Ok(true)
        }
        "flint-promote" => {
            let pending = PathBuf::from(next_argument(&mut args, "Flint candidate (.gguf.pending)")?);
            let mut receipt_path: Option<PathBuf> = None;
            let mut confirm = false;

            while let Some(opt) = args.next() {
                match opt.to_str() {
                    Some("--receipt") => {
                        receipt_path = Some(PathBuf::from(next_argument(&mut args, "receipt path")?));
                    }
                    Some("--confirm") => confirm = true,
                    _ => return Err(invalid(format!("unknown option: {}", opt.to_string_lossy()))),
                }
            }

            if !confirm {
                return Err(invalid("flint-promote requires --confirm"));
            }

            let r_path = receipt_path.ok_or_else(|| invalid("flint-promote requires --receipt <receipt.json>"))?;
            let raw = std::fs::read(&r_path)?;
            let json_val: serde_json::Value = serde_json::from_slice(&raw).map_err(|e| invalid(e.to_string()))?;

            let receipt = ojon_backend::FlintParityReceipt {
                reference_logits_path: PathBuf::from(json_val["reference_logits_path"].as_str().unwrap_or("")),
                candidate_model_path: PathBuf::from(json_val["candidate_model_path"].as_str().unwrap_or("")),
                evaluated_tokens: json_val["evaluated_tokens"].as_u64().unwrap_or(0) as usize,
                vocab_size: json_val["vocab_size"].as_u64().unwrap_or(0) as usize,
                top1_token_agreement_pct: json_val["top1_token_agreement_pct"].as_f64().unwrap_or(0.0),
                top5_token_agreement_pct: json_val["top5_token_agreement_pct"].as_f64(),
                average_cosine_similarity: json_val["average_cosine_similarity"].as_f64(),
                kl_divergence: json_val["kl_divergence"].as_f64().unwrap_or(0.0),
                median_kl_divergence: json_val["median_kl_divergence"].as_f64(),
                max_kl_divergence: json_val["max_kl_divergence"].as_f64(),
                rms_delta_p_pct: json_val["rms_delta_p_pct"].as_f64(),
                ln_ppl_correlation_pct: json_val["ln_ppl_correlation_pct"].as_f64(),
                mean_ppl_q: json_val["mean_ppl_q"].as_f64(),
                mean_ppl_base: json_val["mean_ppl_base"].as_f64(),
                router_storage_verified: json_val["router_storage_verified"].as_bool().unwrap_or(true),
                moe_top_k_routing_agreement_pct: json_val["moe_top_k_routing_agreement_pct"].as_f64(),
                passes_negligible_loss_gate: json_val["passes_negligible_loss_gate"].as_bool().unwrap_or(false),
                certified_for_production: json_val["certified_for_production"].as_bool().unwrap_or(false),
            };

            let promoted = ojon_backend::promote_flint_candidate(&pending, &receipt)?;
            println!("promoted_model: {}", promoted.display());
            println!("status: PROMOTED_TO_OFFICIAL_PRODUCTION_GGUF");
            Ok(true)
        }
        "flint-rco-assemble" => {
            let output = PathBuf::from(next_argument(&mut args, "output hybrid GGUF path")?);
            let mut template: Option<PathBuf> = None;
            let mut spine_source: Option<PathBuf> = None;
            let mut expert_source: Option<PathBuf> = None;
            let mut router_source: Option<PathBuf> = None;
            let mut confirm = false;

            while let Some(opt) = args.next() {
                match opt.to_str() {
                    Some("--template") => {
                        template = Some(PathBuf::from(next_argument(&mut args, "template GGUF path")?));
                    }
                    Some("--spine") => {
                        spine_source = Some(PathBuf::from(next_argument(&mut args, "spine source GGUF (e.g. Q6_K)")?));
                    }
                    Some("--experts") => {
                        expert_source = Some(PathBuf::from(next_argument(&mut args, "experts source GGUF (e.g. Q4_K_M)")?));
                    }
                    Some("--routers") => {
                        router_source = Some(PathBuf::from(next_argument(&mut args, "routers source GGUF (e.g. F32)")?));
                    }
                    Some("--confirm") => confirm = true,
                    _ => return Err(invalid(format!("unknown option: {}", opt.to_string_lossy()))),
                }
            }

            if !confirm {
                return Err(invalid("flint-rco-assemble writes model checkpoint; pass --confirm"));
            }

            let t_path = template.ok_or_else(|| invalid("requires --template"))?;
            let spine = spine_source.ok_or_else(|| invalid("requires --spine"))?;
            let experts = expert_source.unwrap_or_else(|| t_path.clone());
            let routers = router_source.unwrap_or_else(|| t_path.clone());

            let t_tensors = ojon_backend::inspect_gguf_tensors(&t_path)?;
            let mut recipe = std::collections::HashMap::new();

            for t in t_tensors {
                let is_router = t.name.contains("ffn_gate_inp") || t.name.contains("exp_probs_b");
                let is_expert = t.name.contains("_exps");
                let is_spine = t.name.contains("attn_")
                    || t.name.contains("shexp")
                    || t.name.contains("token_embd")
                    || t.name.contains("output.weight")
                    || t.name.contains("blk.0.ffn_")
                    || t.name.contains("blk.1.ffn_");

                if is_router {
                    recipe.insert(t.name, routers.clone());
                } else if is_spine {
                    recipe.insert(t.name, spine.clone());
                } else if is_expert {
                    recipe.insert(t.name, experts.clone());
                } else {
                    recipe.insert(t.name, t_path.clone());
                }
            }

            let final_bytes = ojon_backend::assemble_flint_hybrid_gguf(
                &t_path,
                &recipe,
                &output,
                &mut |line| eprintln!("{}", line),
            )?;

            println!("assembled_model: {}", output.display());
            println!("final_bytes: {}", final_bytes);
            println!("status: RCO_HYBRID_ASSEMBLED_SUCCESSFULLY");
            Ok(true)
        }
        _ => Ok(false),
    }
}
