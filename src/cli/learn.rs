//! Ojon Learn continuous data collection, knowledge acquisition, and model update commands.

use crate::cli::common::{invalid, next_argument};
use std::ffi::OsString;
use std::io;
use std::path::{Path, PathBuf};

pub fn handle(cmd: &str, mut args: impl Iterator<Item = OsString>) -> io::Result<bool> {
    match cmd {
        "learn-collector-status" => {
            let mut data_dir = PathBuf::from("collector");
            let mut port = 8089u16;
            let mut json = false;
            while let Some(opt) = args.next() {
                match opt.to_str() {
                    Some("--data-dir") => {
                        data_dir = PathBuf::from(next_argument(&mut args, "collector data directory")?);
                    }
                    Some("--port") => {
                        port = next_argument(&mut args, "collector listen port")?.to_string_lossy().parse().unwrap_or(8089);
                    }
                    Some("--json") => json = true,
                    _ => return Err(invalid(format!("unknown option: {}", opt.to_string_lossy()))),
                }
            }
            let collector = ojon_backend::LearnCollector::new(data_dir)?;
            let stats = collector.get_stats(port);
            if json {
                println!("{}", serde_json::to_string_pretty(&stats).unwrap());
            } else {
                println!("Ojon Learn Collector Status:");
                println!("  Status: active={}", stats.is_active);
                println!("  Port: {}", stats.listen_port);
                println!("  Total Episodes: {}", stats.total_episodes);
                println!("  Verified Successes: {}", stats.verified_success_count);
                println!("  Failed Retries: {}", stats.failed_retry_count);
                println!("  Unverified: {}", stats.unverified_count);
                println!("  Storage: {}", stats.storage_dir);
            }
            Ok(true)
        }
        "learn-collector-start" => {
            let mut data_dir = PathBuf::from("collector");
            let mut port = 8089u16;
            let mut non_blocking = false;
            let mut upstream: Option<String> = None;
            let mut auth_token: Option<String> = None;
            while let Some(opt) = args.next() {
                match opt.to_str() {
                    Some("--data-dir") => {
                        data_dir = PathBuf::from(next_argument(&mut args, "collector data directory")?);
                    }
                    Some("--port") => {
                        port = next_argument(&mut args, "collector listen port")?.to_string_lossy().parse().unwrap_or(8089);
                    }
                    Some("--upstream") => {
                        upstream = Some(next_argument(&mut args, "upstream OpenAI endpoint URL")?.to_string_lossy().to_string());
                    }
                    Some("--auth-token") => {
                        auth_token = Some(next_argument(&mut args, "collector management auth token")?.to_string_lossy().to_string());
                    }
                    Some("--non-blocking") => non_blocking = true,
                    _ => return Err(invalid(format!("unknown option: {}", opt.to_string_lossy()))),
                }
            }
            let collector = ojon_backend::LearnCollector::new(data_dir.clone())?;
            if let Some(up) = upstream {
                collector.set_upstream_endpoint(Some(up));
            }
            if let Some(tok) = auth_token {
                collector.set_auth_token(Some(tok));
            }
            let collector = std::sync::Arc::new(collector);
            let handle = ojon_backend::LearnCollector::start_server(collector, port)?;
            println!("Ojon Learn Collector running on http://127.0.0.1:{}", port);
            println!("  Recording episodes to {}", data_dir.display());
            if !non_blocking {
                let _ = handle.join();
            }
            Ok(true)
        }
        "learn-harvest" => {
            let source_dir = PathBuf::from(next_argument(&mut args, "source directory containing agent runs")?);
            let mut data_dir = PathBuf::from("collector");
            let mut json = false;
            while let Some(opt) = args.next() {
                match opt.to_str() {
                    Some("--data-dir") => {
                        data_dir = PathBuf::from(next_argument(&mut args, "collector data directory")?);
                    }
                    Some("--json") => json = true,
                    _ => return Err(invalid(format!("unknown option: {}", opt.to_string_lossy()))),
                }
            }
            let collector = ojon_backend::LearnCollector::new(data_dir)?;
            let receipt = ojon_backend::harvest_agent_sessions(&source_dir, &collector)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&receipt).unwrap());
            } else {
                println!("Ojon Learn Agent Session Harvest Complete:");
                println!("  Source: {}", receipt.source_dir);
                println!("  Files Scanned: {}", receipt.files_scanned);
                println!("  Episodes Harvested: {}", receipt.episodes_harvested);
                println!("  Verified Successes: {}", receipt.verified_success_count);
                println!("  Failed Retries: {}", receipt.failed_retry_count);
                println!("  Unverified: {}", receipt.unverified_count);
                println!("  Duplicates Skipped: {}", receipt.duplicates_skipped);
                println!("  Secrets Redacted: {}", receipt.redacted_count);
            }
            Ok(true)
        }
        "learn-utility-models" => {
            let mut json = false;
            while let Some(opt) = args.next() {
                match opt.to_str() {
                    Some("--json") => json = true,
                    _ => return Err(invalid(format!("unknown option: {}", opt.to_string_lossy()))),
                }
            }
            let models = ojon_backend::default_utility_models();
            if json {
                println!("{}", serde_json::to_string_pretty(&models).unwrap());
            } else {
                println!("Ojon Learn Utility Model Registry ({} models):", models.len());
                for m in models {
                    println!(
                        "\n[{}] {} ({}) - {}",
                        m.id, m.name, m.parameter_scale, m.repo_or_engine
                    );
                    println!("    Job: {}", m.role);
                    println!("    Recommendation: {} (CPU-friendly: {})", m.recommendation, m.cpu_friendly);
                }
            }
            Ok(true)
        }
        "learn-dataset-list" => {
            let mut category_filter: Option<String> = None;
            let mut json = false;
            while let Some(opt) = args.next() {
                match opt.to_str() {
                    Some("--category") => {
                        category_filter = Some(next_argument(&mut args, "category filter")?.to_string_lossy().to_string());
                    }
                    Some("--json") => json = true,
                    _ => return Err(invalid(format!("unknown option: {}", opt.to_string_lossy()))),
                }
            }
            let all = ojon_backend::list_curated_datasets(None);
            let filtered: Vec<_> = all.into_iter().filter(|d| {
                if let Some(cat) = &category_filter {
                    d.category.as_str().eq_ignore_ascii_case(cat)
                } else {
                    true
                }
            }).collect();
            if json {
                println!("{}", serde_json::to_string_pretty(&filtered).unwrap());
            } else {
                println!("Curated Distillation Datasets ({} available):", filtered.len());
                for d in filtered {
                    println!("\n[{}] {} ({}) - {}", d.id, d.name, d.category.as_str(), d.hf_repo);
                    println!("    Est Rows: {}, Tier: {}", d.estimated_rows, d.confidence_tier);
                    println!("    {}", d.description);
                }
            }
            Ok(true)
        }
        "learn-dataset-import" => {
            let input_path = PathBuf::from(next_argument(&mut args, "input JSONL file")?);
            let output_path = PathBuf::from(next_argument(&mut args, "output sanitized JSONL file")?);
            let mut json = false;
            while let Some(opt) = args.next() {
                match opt.to_str() {
                    Some("--json") => json = true,
                    _ => return Err(invalid(format!("unknown option: {}", opt.to_string_lossy()))),
                }
            }
            let receipt = ojon_backend::import_and_sanitize_jsonl(&input_path, &output_path, None)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&receipt).unwrap());
            } else {
                println!("Dataset Import Sanitization Complete:");
                println!("  Total Rows: {}", receipt.total_input_rows);
                println!("  Accepted Rows: {}", receipt.accepted_rows);
                println!("  Deduplicated: {}", receipt.rejected_duplicates);
                println!("  Secrets Redacted: {}", receipt.redacted_secrets_count);
                println!("  Output: {}", receipt.output_path);
            }
            Ok(true)
        }
        "learn-curate" => {
            let episodes_path = PathBuf::from(next_argument(&mut args, "episodes JSONL file")?);
            let output_dir = PathBuf::from(next_argument(&mut args, "curation output directory")?);
            let mut anchor_ratio = 0.08f64;
            let mut recency_k = 3usize;
            let mut json = false;
            while let Some(opt) = args.next() {
                match opt.to_str() {
                    Some("--anchor-ratio") => {
                        anchor_ratio = next_argument(&mut args, "anchor buffer ratio")?.to_string_lossy().parse().unwrap_or(0.08);
                    }
                    Some("--recency-k") => {
                        recency_k = next_argument(&mut args, "recency sliding window K")?.to_string_lossy().parse().unwrap_or(3);
                    }
                    Some("--json") => json = true,
                    _ => return Err(invalid(format!("unknown option: {}", opt.to_string_lossy()))),
                }
            }
            let episodes = ojon_backend::load_episodes_from_jsonl(&episodes_path)?;
            let mut config = ojon_backend::DualBufferConfig::default();
            config.anchor_ratio = anchor_ratio;
            config.recency_window_k = recency_k;
            let batch = ojon_backend::partition_dual_buffer(&episodes, &config);
            let receipt = ojon_backend::export_to_llamafactory(&batch, &output_dir)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&receipt).unwrap());
            } else {
                println!("Ojon Learn Curation Complete:");
                println!("  Anchor Samples: {}", receipt.anchor_count);
                println!("  Recent Samples: {}", receipt.recency_count);
                println!("  DPO Pairs: {}", receipt.dpo_pair_count);
                println!("  Dataset Info: {}", receipt.dataset_info_file.display());
            }
            Ok(true)
        }
        "learn-train-plan" => {
            let model_dir = PathBuf::from(next_argument(&mut args, "base model directory")?);
            let dataset_dir = PathBuf::from(next_argument(&mut args, "dataset directory")?);
            let output_dir = PathBuf::from(next_argument(&mut args, "training output directory")?);
            let mut quant_bit = 4u32;
            let mut json = false;
            while let Some(opt) = args.next() {
                match opt.to_str() {
                    Some("--quant") => {
                        quant_bit = next_argument(&mut args, "quantization bits (4 or 8)")?.to_string_lossy().parse().unwrap_or(4);
                    }
                    Some("--json") => json = true,
                    _ => return Err(invalid(format!("unknown option: {}", opt.to_string_lossy()))),
                }
            }
            let plan = ojon_backend::LearnTrainPlan::new(model_dir, dataset_dir, output_dir, Some(Some(quant_bit)));
            if json {
                println!("{}", serde_json::to_string_pretty(&plan).unwrap());
            } else {
                println!("Ojon Learn Post-Training (SFT) Plan:");
                println!("  Base Model: {}", plan.base_model_path.display());
                println!("  Stage: {:?}", plan.stage);
                println!("  Hardware Tier: {:?}", plan.hardware.tier);
                println!("  Device Target: {}", plan.hardware.device_name);
                println!("  LoRA Rank: r={}, alpha={}", plan.lora_rank, plan.lora_alpha);
                println!("  Target Modules: {:?}", plan.target_modules);
                println!("  Quantization Bits: {:?}", plan.quantization_bit);
                println!("  Gradient Checkpointing: {}", plan.gradient_checkpointing);
            }
            Ok(true)
        }
        "learn-train-run" => {
            let model_dir = PathBuf::from(next_argument(&mut args, "base model directory")?);
            let dataset_dir = PathBuf::from(next_argument(&mut args, "dataset directory")?);
            let output_dir = PathBuf::from(next_argument(&mut args, "training output directory")?);
            let mut quant_bit = 4u32;
            let mut dry_run = false;
            let mut confirm = false;
            let mut trainer_bin: Option<String> = None;
            while let Some(opt) = args.next() {
                match opt.to_str() {
                    Some("--quant") => {
                        quant_bit = next_argument(&mut args, "quantization bits (4 or 8)")?.to_string_lossy().parse().unwrap_or(4);
                    }
                    Some("--dry-run") => dry_run = true,
                    Some("--confirm") => confirm = true,
                    Some("--trainer") => {
                        trainer_bin = Some(next_argument(&mut args, "trainer binary")?.to_string_lossy().to_string());
                    }
                    _ => return Err(invalid(format!("unknown option: {}", opt.to_string_lossy()))),
                }
            }
            if !confirm && !dry_run {
                return Err(invalid("Training modifies hardware state and weights: specify --confirm or --dry-run"));
            }
            let plan = ojon_backend::LearnTrainPlan::new(model_dir, dataset_dir, output_dir, Some(Some(quant_bit)));
            let receipt = ojon_backend::execute_training_plan(&plan, dry_run, trainer_bin.as_deref().map(Path::new))?;
            println!("{}", serde_json::to_string_pretty(&receipt).unwrap());
            Ok(true)
        }
        "learn-promote" => {
            let eval_matrix_file = PathBuf::from(next_argument(&mut args, "evaluation matrix JSON file")?);
            let target_candidate = next_argument(&mut args, "target candidate ID to promote")?.to_string_lossy().to_string();
            let pointer_file = PathBuf::from(next_argument(&mut args, "promoted pointer destination JSON file")?);
            let mut json = false;
            while let Some(opt) = args.next() {
                match opt.to_str() {
                    Some("--json") => json = true,
                    _ => return Err(invalid(format!("unknown option: {}", opt.to_string_lossy()))),
                }
            }
            let matrix_str = std::fs::read_to_string(&eval_matrix_file)?;
            let candidates: Vec<ojon_backend::CandidateModelEvaluation> = serde_json::from_str(&matrix_str)
                .map_err(|e| invalid(format!("invalid candidate evaluation matrix JSON: {e}")))?;
            let policy = ojon_backend::PromotionPolicy::default();
            let decision = ojon_backend::evaluate_and_promote(&candidates, &target_candidate, &policy, &pointer_file)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&decision).unwrap());
            } else {
                println!("Ojon Learn Promotion Verdict: {}", decision.decision.to_uppercase());
                println!("  Candidate: {}", decision.candidate_id);
                if let Some(rb) = &decision.rollback_pointer {
                    println!("  Rollback pointer preserved: {}", rb);
                }
                println!("  Gate audit checks:");
                for g in &decision.gates {
                    let status = if g.passed { "PASS" } else { "FAIL" };
                    println!("    [{}] {}: target {}, observed {} ({})", status, g.gate_name, g.target, g.observed, g.reason);
                }
            }
            Ok(true)
        }
        "learn-knowledge-add" => {
            let store_path = PathBuf::from(next_argument(&mut args, "knowledge store JSONL path")?);
            let source_url = next_argument(&mut args, "source URL")?.to_string_lossy().to_string();
            let title = next_argument(&mut args, "document title")?.to_string_lossy().to_string();
            let domain = next_argument(&mut args, "knowledge domain")?.to_string_lossy().to_string();
            let content_arg = next_argument(&mut args, "content file path, raw text, or --fetch")?.to_string_lossy().to_string();

            let mut kind = ojon_backend::KnowledgeSourceKind::SoftwareDocumentation;
            let mut version: Option<String> = None;
            let mut published_epoch_s: u64 = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let mut tier = "Authoritative".to_string();
            let mut license = "Public Reference".to_string();
            let mut fetch_online = content_arg == "--fetch";
            let mut json = false;

            while let Some(opt) = args.next() {
                match opt.to_str() {
                    Some("--kind") => {
                        let k = next_argument(&mut args, "source kind")?.to_string_lossy().to_string();
                        kind = match k.to_ascii_lowercase().as_str() {
                            "software_documentation" | "docs" => ojon_backend::KnowledgeSourceKind::SoftwareDocumentation,
                            "github_release_changelog" | "changelog" => ojon_backend::KnowledgeSourceKind::GitHubReleaseChangelog,
                            "api_deprecation_notice" | "deprecation" => ojon_backend::KnowledgeSourceKind::ApiDeprecationNotice,
                            "ai_research_paper" | "paper" => ojon_backend::KnowledgeSourceKind::AiResearchPaper,
                            "authoritative_fact" | "fact" => ojon_backend::KnowledgeSourceKind::AuthoritativeFact,
                            other => return Err(invalid(format!("unknown source kind: {other}"))),
                        };
                    }
                    Some("--version") => {
                        version = Some(next_argument(&mut args, "software version")?.to_string_lossy().to_string());
                    }
                    Some("--published") => {
                        let p_str = next_argument(&mut args, "published epoch timestamp")?.to_string_lossy().to_string();
                        published_epoch_s = p_str.parse::<u64>().map_err(|e| invalid(format!("invalid published epoch: {e}")))?;
                    }
                    Some("--tier") => {
                        tier = next_argument(&mut args, "reliability tier")?.to_string_lossy().to_string();
                    }
                    Some("--license") => {
                        license = next_argument(&mut args, "license")?.to_string_lossy().to_string();
                    }
                    Some("--fetch") => fetch_online = true,
                    Some("--json") => json = true,
                    _ => return Err(invalid(format!("unknown option: {}", opt.to_string_lossy()))),
                }
            }

            let content = if fetch_online {
                ojon_backend::fetch_online_knowledge_url(&source_url, 15, 2 * 1024 * 1024)?
            } else if Path::new(&content_arg).is_file() {
                std::fs::read_to_string(&content_arg)?
            } else {
                content_arg
            };

            let item_id = format!("doc-{}", ojon_backend::compute_content_hash(&format!("{}:{}:{}", domain, title, published_epoch_s)));
            let item = ojon_backend::create_knowledge_item(
                &item_id,
                kind,
                &source_url,
                &title,
                &domain,
                &content,
                published_epoch_s,
                version.as_deref(),
                &tier,
                &license,
                vec![format!("Source: {source_url} (Published: {published_epoch_s})")],
            );

            let mut store = ojon_backend::load_knowledge_store(&store_path).unwrap_or_default();
            let superseded_ids = ojon_backend::apply_supersession(&mut store, &item);
            store.push(item.clone());
            ojon_backend::save_knowledge_store(&store, &store_path)?;

            if json {
                #[derive(serde::Serialize)]
                struct AddReceipt {
                    item_id: String,
                    title: String,
                    domain: String,
                    content_hash: String,
                    superseded_ids: Vec<String>,
                    total_store_items: usize,
                }
                let receipt = AddReceipt {
                    item_id: item.item_id,
                    title: item.title,
                    domain: item.domain,
                    content_hash: item.content_hash,
                    superseded_ids,
                    total_store_items: store.len(),
                };
                println!("{}", serde_json::to_string_pretty(&receipt).unwrap());
            } else {
                println!("Ingested Knowledge Item: {}", item.item_id);
                println!("  Title: {}", item.title);
                println!("  Domain: {}", item.domain);
                println!("  Content Hash: {}", item.content_hash);
                if !superseded_ids.is_empty() {
                    println!("  Superseded {} older fact(s): {:?}", superseded_ids.len(), superseded_ids);
                }
                println!("  Store total items: {}", store.len());
            }
            Ok(true)
        }
        "learn-temporal-query" => {
            let store_path = PathBuf::from(next_argument(&mut args, "knowledge store JSONL path")?);
            let query_text = next_argument(&mut args, "search query text")?.to_string_lossy().to_string();

            let mut as_of_epoch_s: Option<u64> = None;
            let mut domain_filter: Option<String> = None;
            let mut limit = 5usize;
            let mut augment = false;
            let mut json = false;

            while let Some(opt) = args.next() {
                match opt.to_str() {
                    Some("--as-of") => {
                        let val = next_argument(&mut args, "as-of timestamp or YYYY-MM-DD")?.to_string_lossy().to_string();
                        if let Ok(ts) = val.parse::<u64>() {
                            as_of_epoch_s = Some(ts);
                        } else if let Some(parsed) = ojon_backend::parse_iso_date_to_epoch(&val) {
                            as_of_epoch_s = Some(parsed);
                        } else {
                            return Err(invalid(format!("invalid as-of date format '{val}': use epoch seconds or YYYY-MM-DD")));
                        }
                    }
                    Some("--domain") => {
                        domain_filter = Some(next_argument(&mut args, "domain filter")?.to_string_lossy().to_string());
                    }
                    Some("--limit") => {
                        limit = next_argument(&mut args, "result limit")?.to_string_lossy().parse().unwrap_or(5);
                    }
                    Some("--augment") => augment = true,
                    Some("--json") => json = true,
                    _ => return Err(invalid(format!("unknown option: {}", opt.to_string_lossy()))),
                }
            }

            let engine = ojon_backend::TemporalKnowledgeEngine::load_from_store(&store_path)?;
            if augment {
                let resp = engine.build_augmented_prompt(&query_text, as_of_epoch_s, limit);
                if json {
                    println!("{}", serde_json::to_string_pretty(&resp).unwrap());
                } else {
                    println!("--- Augmented Prompt ---");
                    println!("{}", resp.augmented_prompt);
                    println!("\n--- Verified Citations ({}) ---", resp.retrieved_evidence.len());
                    for (i, ev) in resp.retrieved_evidence.iter().enumerate() {
                        let status = if ev.is_currently_valid { "CURRENT" } else { "SUPERSEDED/HISTORICAL" };
                        println!("[{}] {} ({}) - {}", i + 1, ev.title, status, ev.source_url);
                        println!("    Valid from: {}", ojon_backend::format_epoch_iso(ev.valid_from_epoch_s));
                        if let Some(to) = ev.valid_to_epoch_s {
                            println!("    Valid to: {}", ojon_backend::format_epoch_iso(to));
                        }
                    }
                }
            } else {
                let q = ojon_backend::TemporalRetrievalQuery {
                    query_text,
                    as_of_epoch_s,
                    domain_filter,
                    limit,
                };
                let results = engine.search(&q);
                if json {
                    println!("{}", serde_json::to_string_pretty(&results).unwrap());
                } else {
                    println!("Temporal Search Results (found {}):", results.len());
                    for (i, res) in results.iter().enumerate() {
                        let status = if res.is_currently_valid { "CURRENT" } else { "HISTORICAL" };
                        println!("\n[{}] {} [Score: {:.2}] [{}]", i + 1, res.title, res.relevance_score, status);
                        println!("    URL: {}", res.source_url);
                        println!("    Valid: {} -> {}", ojon_backend::format_epoch_iso(res.valid_from_epoch_s), res.valid_to_epoch_s.map(ojon_backend::format_epoch_iso).unwrap_or_else(|| "Present".to_string()));
                        println!("    Snippet: {}", res.snippet);
                    }
                }
            }
            Ok(true)
        }
        "learn-package-create" => {
            let kind_arg = next_argument(&mut args, "package kind (knowledge|skill|release)")?.to_string_lossy().to_string();
            let id = next_argument(&mut args, "package ID")?.to_string_lossy().to_string();
            let name = next_argument(&mut args, "package name")?.to_string_lossy().to_string();
            let version = next_argument(&mut args, "package version")?.to_string_lossy().to_string();
            let input_path = PathBuf::from(next_argument(&mut args, "input file path")?);
            let dest_dir = PathBuf::from(next_argument(&mut args, "package destination directory")?);

            let mut base_model = "universal".to_string();
            let mut description = format!("Ojon update package {id} v{version}");
            let mut author = "Ojon Community".to_string();
            let mut license = "Apache-2.0".to_string();
            let mut changelog = vec![format!("Initial build of {id} v{version}")];
            let mut json = false;

            while let Some(opt) = args.next() {
                match opt.to_str() {
                    Some("--base-model") => {
                        base_model = next_argument(&mut args, "base model compatibility")?.to_string_lossy().to_string();
                    }
                    Some("--desc") => {
                        description = next_argument(&mut args, "package description")?.to_string_lossy().to_string();
                    }
                    Some("--author") => {
                        author = next_argument(&mut args, "author name")?.to_string_lossy().to_string();
                    }
                    Some("--license") => {
                        license = next_argument(&mut args, "license")?.to_string_lossy().to_string();
                    }
                    Some("--changelog") => {
                        changelog = next_argument(&mut args, "changelog entry")?.to_string_lossy().split(';').map(|s| s.trim().to_string()).collect();
                    }
                    Some("--json") => json = true,
                    _ => return Err(invalid(format!("unknown option: {}", opt.to_string_lossy()))),
                }
            }

            let manifest = match kind_arg.to_ascii_lowercase().as_str() {
                "knowledge" | "knowledge_pack" => {
                    ojon_backend::create_knowledge_pack(
                        &id, &name, &version, &input_path, &dest_dir,
                        &description, &author, &license, changelog,
                    )?
                }
                "skill" | "skill_pack" => {
                    ojon_backend::create_skill_pack(
                        &id, &name, &version, &input_path, &base_model, &dest_dir,
                        std::collections::HashMap::new(), &description, &author, &license, changelog,
                    )?
                }
                "release" | "model_release" => {
                    ojon_backend::create_model_release(
                        &id, &name, &version, &input_path, &dest_dir,
                        std::collections::HashMap::new(), &description, &author, &license, changelog,
                    )?
                }
                other => return Err(invalid(format!("unknown package kind '{other}': use knowledge, skill, or release"))),
            };

            if json {
                println!("{}", serde_json::to_string_pretty(&manifest).unwrap());
            } else {
                println!("Created Ojon Package Manifest:");
                println!("  Package ID: {}", manifest.package_id);
                println!("  Kind: {:?}", manifest.kind);
                println!("  Version: {}", manifest.version);
                println!("  Content Hash: {}", manifest.content_hash);
                println!("  Size: {} bytes", manifest.size_bytes);
                println!("  Destination: {}", dest_dir.display());
            }
            Ok(true)
        }
        "learn-package-install" => {
            let pkg_dir = PathBuf::from(next_argument(&mut args, "package directory")?);
            let install_root = PathBuf::from(next_argument(&mut args, "installation root directory")?);
            let mut base_model: Option<String> = None;
            let mut json = false;
            while let Some(opt) = args.next() {
                match opt.to_str() {
                    Some("--base-model") => {
                        base_model = Some(next_argument(&mut args, "expected base model")?.to_string_lossy().to_string());
                    }
                    Some("--json") => json = true,
                    _ => return Err(invalid(format!("unknown option: {}", opt.to_string_lossy()))),
                }
            }
            let receipt = ojon_backend::install_package_for_model(&pkg_dir, &install_root, base_model.as_deref())?;
            if json {
                println!("{}", serde_json::to_string_pretty(&receipt).unwrap());
            } else {
                println!("Ojon Package Installed: {}", receipt.package_id);
                println!("  Version: {}", receipt.version);
                println!("  Kind: {:?}", receipt.kind);
                println!("  Active Path: {}", receipt.active_path.display());
                if receipt.rollback_available {
                    println!("  Rollback snapshot saved to: {:?}", receipt.rollback_path);
                }
            }
            Ok(true)
        }
        "learn-package-list" => {
            let install_root = PathBuf::from(next_argument(&mut args, "installation root directory")?);
            let mut json = false;
            while let Some(opt) = args.next() {
                match opt.to_str() {
                    Some("--json") => json = true,
                    _ => return Err(invalid(format!("unknown option: {}", opt.to_string_lossy()))),
                }
            }
            let manifests = ojon_backend::list_active_packages(&install_root)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&manifests).unwrap());
            } else {
                println!("Active Installed Ojon Packages ({}):", manifests.len());
                for m in manifests {
                    println!("  [{}] {} v{} ({:?}) - {}", m.package_id, m.name, m.version, m.kind, m.content_hash);
                }
            }
            Ok(true)
        }
        "learn-package-rollback" => {
            let kind_arg = next_argument(&mut args, "package kind (knowledge|skill|release)")?.to_string_lossy().to_string();
            let install_root = PathBuf::from(next_argument(&mut args, "installation root directory")?);
            let mut json = false;
            while let Some(opt) = args.next() {
                match opt.to_str() {
                    Some("--json") => json = true,
                    _ => return Err(invalid(format!("unknown option: {}", opt.to_string_lossy()))),
                }
            }
            let kind = match kind_arg.to_ascii_lowercase().as_str() {
                "knowledge" | "knowledge_pack" => ojon_backend::PackageKind::KnowledgePack,
                "skill" | "skill_pack" => ojon_backend::PackageKind::SkillPack,
                "release" | "model_release" => ojon_backend::PackageKind::ModelRelease,
                other => return Err(invalid(format!("unknown package kind '{other}': use knowledge, skill, or release"))),
            };
            let receipt = ojon_backend::rollback_package(kind, &install_root)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&receipt).unwrap());
            } else {
                println!("Ojon Package Rollback Complete: {}", receipt.restored_package_id);
                println!("  Restored Version: {}", receipt.restored_version);
                println!("  Kind: {:?}", receipt.kind);
                println!("  Status: {}", receipt.status);
            }
            Ok(true)
        }
        "learn-daemon-run" => {
            let mut config = ojon_backend::DaemonConfig::default();
            let mut timeout_s: Option<u64> = None;

            while let Some(opt) = args.next() {
                match opt.to_str() {
                    Some("--port") => {
                        config.listen_port = next_argument(&mut args, "listen port")?.to_string_lossy().parse().unwrap_or(8089);
                    }
                    Some("--data-dir") => {
                        config.storage_dir = PathBuf::from(next_argument(&mut args, "storage data directory")?);
                    }
                    Some("--max-disk-mb") => {
                        config.max_disk_mb = next_argument(&mut args, "max disk mb")?.to_string_lossy().parse().unwrap_or(1024);
                    }
                    Some("--timeout-s") => {
                        timeout_s = Some(next_argument(&mut args, "daemon run timeout seconds")?.to_string_lossy().parse().unwrap_or(5));
                    }
                    _ => return Err(invalid(format!("unknown option: {}", opt.to_string_lossy()))),
                }
            }

            println!("Starting Ojon Learn Headless Daemon on port {}...", config.listen_port);
            println!("  Storage directory: {}", config.storage_dir.display());
            println!("  Max storage limit: {} MB", config.max_disk_mb);

            let daemon = ojon_backend::LearnDaemon::new(config)?;
            let handle = daemon.start()?;

            if let Some(secs) = timeout_s {
                println!("Running for {secs} seconds before bounded auto-shutdown...");
                std::thread::sleep(std::time::Duration::from_secs(secs));
                daemon.stop();
                let _ = handle.join();
                println!("Ojon Learn Daemon stopped cleanly.");
            } else {
                println!("Daemon active. Press Ctrl+C to terminate.");
                let _ = handle.join();
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}
