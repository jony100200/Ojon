//! Inspection, verification, settings, and hardware probe commands.

use crate::cli::common::{invalid, next_argument, show_gguf};
use ojon_backend::*;
use std::ffi::OsString;
use std::io;
use std::path::PathBuf;
use std::str::FromStr;

pub fn handle(cmd: &str, mut args: impl Iterator<Item = OsString>) -> io::Result<bool> {
    match cmd {
        "settings" => {
            let op = next_argument(&mut args, "settings subcommand (path/show/set/clear)")?;
            match op.to_str() {
                Some("path") => {
                    if args.next().is_some() {
                        return Err(invalid("settings path takes no arguments"));
                    }
                    println!("{}", ojon_backend::settings_path()?.display());
                }
                Some("show") => {
                    let extra = args.next();
                    if extra.as_deref() != Some(std::ffi::OsStr::new("--json")) && extra.is_some() {
                        return Err(invalid("settings show accepts only --json"));
                    }
                    if args.next().is_some() {
                        return Err(invalid("too many settings arguments"));
                    }
                    let path = ojon_backend::settings_path()?;
                    let settings = ojon_backend::load_settings()?;
                    println!(
                        "{}",
                        serde_json::json!({
                            "path":path.to_string_lossy(),
                            "stored":path.is_file(),
                            "settings":settings.as_json()
                        })
                    );
                }
                Some("set") => {
                    let key = next_argument(&mut args, "setting key")?;
                    let value = next_argument(&mut args, "setting value")?;
                    if args.next().is_some() {
                        return Err(invalid("too many settings arguments"));
                    }
                    let mut settings = ojon_backend::load_settings()?;
                    settings.set(
                        key.to_str()
                            .ok_or_else(|| invalid("setting name must be UTF-8"))?,
                        value
                            .to_str()
                            .ok_or_else(|| invalid("setting value must be UTF-8"))?,
                    )?;
                    let path = ojon_backend::save_settings(&settings)?;
                    println!("saved_local_preferences: {}", path.display());
                }
                Some("clear") => {
                    let key = next_argument(&mut args, "setting key")?;
                    if args.next().is_some() {
                        return Err(invalid("too many settings arguments"));
                    }
                    let mut settings = ojon_backend::load_settings()?;
                    settings.clear(
                        key.to_str()
                            .ok_or_else(|| invalid("setting name must be UTF-8"))?,
                    )?;
                    let path = ojon_backend::save_settings(&settings)?;
                    println!("updated_local_preferences: {}", path.display());
                }
                _ => return Err(invalid("settings command must be path, show, set or clear")),
            }
            Ok(true)
        }
        "probe" => {
            let path = PathBuf::from(next_argument(&mut args, "local model file or directory")?);
            let json = match args.next() {
                None => false,
                Some(value) if value == "--json" => true,
                _ => return Err(invalid("probe accepts only an optional --json")),
            };
            if args.next().is_some() {
                return Err(invalid("too many probe arguments"));
            }
            let report = probe_model(&path)?;
            if json {
                println!("{}", report.as_json());
            } else {
                println!("source: {}", report.path.display());
                println!("format: {}", report.format);
                println!(
                    "category: {} (confidence: {})",
                    report.category.as_str(),
                    report.confidence
                );
                println!("identification: {}", report.identification);
                println!("bytes: {}", report.bytes);
                println!("conversion_route: {}", report.conversion_route);
                println!("conversion_status: {}", report.conversion_status);
                for note in report.notes {
                    println!("note: {note}");
                }
            }
            Ok(true)
        }
        "dataset-audit" => {
            let path = PathBuf::from(next_argument(&mut args, "dataset file (.jsonl or .json)")?);
            let mut json = false;
            for opt in args.by_ref() {
                match opt.to_str() {
                    Some("--json") => json = true,
                    _ => {
                        return Err(invalid(format!(
                            "unknown dataset-audit option: {}",
                            opt.to_string_lossy()
                        )))
                    }
                }
            }
            let report = audit_dataset_file(&path)?;
            if json {
                println!("{}", report.json());
            } else {
                println!("dataset: {}", report.source_path.display());
                println!("total_records: {}", report.total_records);
                println!("active_records: {}", report.active_records);
                println!("superseded_records: {}", report.superseded_records);
                println!("references_live: {}", report.total_live_refs);
                println!("references_dead: {}", report.total_dead_refs);
                println!("evergreen_records: {}", report.evergreen_count);
                println!("operational_records: {}", report.operational_count);
                println!("volatile_records: {}", report.volatile_count);
                println!("ephemeral_records: {}", report.ephemeral_count);
                println!("avg_decay_factor: {:.3}", report.avg_decay_factor);
                println!("avg_composite_score: {:.3}", report.avg_composite_score);
            }
            Ok(true)
        }
        "distill-plan" => {
            let source = PathBuf::from(next_argument(&mut args, "source dataset file (.jsonl)")?);
            let out_dir = PathBuf::from(next_argument(&mut args, "new output directory")?);
            let mut min_score = 0.50f64;
            let mut json = false;
            while let Some(opt) = args.next() {
                match opt.to_str() {
                    Some("--min-score") => {
                        let os = next_argument(&mut args, "minimum score threshold")?;
                        let s = os
                            .to_str()
                            .ok_or_else(|| invalid("min-score must be text"))?;
                        min_score = s
                            .parse::<f64>()
                            .map_err(|_| invalid("invalid min-score float"))?;
                    }
                    Some("--json") => json = true,
                    _ => {
                        return Err(invalid(format!(
                            "unknown distill-plan option: {}",
                            opt.to_string_lossy()
                        )))
                    }
                }
            }
            let plan = prepare_distill_plan(&source, &out_dir, min_score)?;
            if json {
                println!("{}", plan.json());
            } else {
                println!("source: {}", plan.source_dataset.display());
                println!("output_directory: {}", plan.output_dir.display());
                println!("total_input_rows: {}", plan.total_input_rows);
                println!("curated_rows: {}", plan.curated_rows);
                println!("curated_split: {}", plan.curated_split_path.display());
                println!("min_score_threshold: {:.2}", plan.min_score_threshold);
                println!(
                    "live_reference_ratio: {:.1}%",
                    plan.live_reference_ratio * 100.0
                );
                println!("status: CURATED_SPLIT_AND_PLAN_WRITTEN");
            }
            Ok(true)
        }
        "retrieve" => {
            let dataset = PathBuf::from(next_argument(&mut args, "dataset file (.jsonl)")?);
            let query = next_argument(&mut args, "query search text")?
                .to_str()
                .ok_or_else(|| invalid("query must be valid Unicode"))?
                .to_owned();
            let mut limit = 5usize;
            let mut min_score = 0.35f64;
            let mut json = false;
            let mut markdown = false;
            while let Some(opt) = args.next() {
                match opt.to_str() {
                    Some("--limit") => {
                        let os = next_argument(&mut args, "match limit")?;
                        let s = os.to_str().ok_or_else(|| invalid("limit must be text"))?;
                        limit = s
                            .parse::<usize>()
                            .map_err(|_| invalid("invalid limit integer"))?;
                    }
                    Some("--min-score") => {
                        let os = next_argument(&mut args, "minimum score threshold")?;
                        let s = os
                            .to_str()
                            .ok_or_else(|| invalid("min-score must be text"))?;
                        min_score = s
                            .parse::<f64>()
                            .map_err(|_| invalid("invalid min-score float"))?;
                    }
                    Some("--json") => json = true,
                    Some("--markdown") => markdown = true,
                    _ => {
                        return Err(invalid(format!(
                            "unknown retrieve option: {}",
                            opt.to_string_lossy()
                        )))
                    }
                }
            }
            let matches = retrieve_recipes(&dataset, &query, limit, min_score)?;
            if json {
                let arr: Vec<_> = matches.iter().map(|m| m.json()).collect();
                println!(
                    "{}",
                    serde_json::json!({"query": query, "dataset": dataset.to_string_lossy(), "count": arr.len(), "matches": arr})
                );
            } else if markdown {
                println!(
                    "### Distilled Recipes: \"{}\" ({} matches)\n",
                    query,
                    matches.len()
                );
                for (idx, m) in matches.iter().enumerate() {
                    println!(
                        "{}. **{}** `[{}]` (Score: {:.2}, Live: {:.0}%, Decay: {:.0}%)",
                        idx + 1,
                        m.title,
                        m.tier,
                        m.match_score,
                        m.liveness_factor * 100.0,
                        m.decay_factor * 100.0
                    );
                    println!(
                        "   - **Intent**: {}",
                        m.instruction.lines().next().unwrap_or("")
                    );
                    let preview = m
                        .summary_or_output
                        .lines()
                        .take(3)
                        .collect::<Vec<_>>()
                        .join("\n     ");
                    println!("   - **Recipe**:\n     {}\n", preview);
                }
            } else {
                println!("Query: \"{}\" | Matches: {}", query, matches.len());
                for (idx, m) in matches.iter().enumerate() {
                    println!(
                        "[{}] {} (tier: {}, score: {:.2})",
                        idx + 1,
                        m.title,
                        m.tier,
                        m.match_score
                    );
                    println!(
                        "    Instruction: {}",
                        m.instruction.lines().next().unwrap_or("")
                    );
                    let snippet = if m.summary_or_output.len() > 160 {
                        format!("{}...", &m.summary_or_output[..160])
                    } else {
                        m.summary_or_output.clone()
                    };
                    println!("    Preview: {}\n", snippet.replace('\n', " "));
                }
            }
            Ok(true)
        }
        "inspect" | "verify" => {
            let path = PathBuf::from(next_argument(&mut args, "GGUF file")?);
            if args.next().is_some() {
                return Err(invalid("too many arguments"));
            }
            let summary = inspect_gguf(&path)?;
            show_gguf(&path, &summary);
            println!("verification: GGUF_METADATA_AND_TENSOR_CATALOG_ONLY");
            println!("runtime_load_test: NOT_RUN");
            println!("task_quality_test: NOT_RUN");
            Ok(true)
        }
        "inspect-hf" => {
            let path = PathBuf::from(next_argument(&mut args, "HF model folder")?);
            if args.next().is_some() {
                return Err(invalid("too many arguments"));
            }
            let source = inspect_hf_folder(&path)?;
            println!("source: {}", source.directory.display());
            println!("weight_files: {}", source.weight_files);
            println!("weight_bytes: {}", source.source_weight_bytes);
            println!("conversion_script: must support this model family");
            println!("inspection: SOURCE_FILES_ONLY; no tensor payloads loaded");
            Ok(true)
        }
        "quantize-hf" => {
            let source = PathBuf::from(next_argument(&mut args, "HF model directory")?);
            let output = PathBuf::from(next_argument(&mut args, "output GGUF path")?);
            let mut converter = discovered_path(BackendId::HfConverter)
                .unwrap_or_else(|| PathBuf::from("convert_hf_to_gguf.py"));
            let mut python = hf_python_path();
            let mut quantizer = discovered_path(BackendId::LlamaQuantize)
                .unwrap_or_else(|| PathBuf::from("llama-quantize"));
            let mut precision = Precision::default();
            let mut threads = std::thread::available_parallelism()
                .map(|v| v.get())
                .unwrap_or(4);
            let mut mmproj_out: Option<PathBuf> = None;
            let mut mmproj_precision = ojon_backend::mmproj::MmprojPrecision::default();
            while let Some(option) = args.next() {
                match option.to_str() {
                    Some("--converter") => {
                        converter = PathBuf::from(next_argument(&mut args, "HF converter script")?)
                    }
                    Some("--mmproj-out") => {
                        mmproj_out = Some(PathBuf::from(next_argument(
                            &mut args,
                            "vision projector output GGUF",
                        )?))
                    }
                    Some("--mmproj-precision") => {
                        let v = next_argument(&mut args, "mmproj precision")?;
                        mmproj_precision = v
                            .to_str()
                            .ok_or_else(|| invalid("mmproj precision must be text"))?
                            .parse()
                            .map_err(invalid)?;
                    }
                    Some("--python") => {
                        python = PathBuf::from(next_argument(&mut args, "Python executable")?)
                    }
                    Some("--quantizer") => {
                        quantizer = PathBuf::from(next_argument(&mut args, "quantizer executable")?)
                    }
                    Some("--precision") => {
                        let v = next_argument(&mut args, "GGUF precision")?;
                        let v = v
                            .to_str()
                            .ok_or_else(|| invalid("precision must be text"))?;
                        precision = Precision::from_str(v).map_err(invalid)?;
                    }
                    Some("--threads") => {
                        let v = next_argument(&mut args, "thread count")?;
                        threads = v
                            .to_str()
                            .ok_or_else(|| invalid("invalid thread count"))?
                            .parse()
                            .map_err(|_| invalid("invalid thread count"))?;
                    }
                    _ => {
                        return Err(invalid(format!(
                            "unknown quantize-hf option: {}",
                            option.to_string_lossy()
                        )))
                    }
                }
            }
            let plan = prepare_hf(
                &source, &output, &converter, &python, &quantizer, precision, threads,
            )?;
            let plan = match &mmproj_out {
                Some(out) => plan.with_mmproj(out, mmproj_precision)?,
                None => plan,
            };
            println!("model_folder: {}", plan.source.directory.display());
            println!("source_weight_bytes: {}", plan.source.source_weight_bytes);
            println!("converter: {}", plan.converter_script.display());
            println!("quantizer: {}", plan.quantizer.display());
            println!("python: {}", plan.python.display());
            println!("precision: {}", plan.precision);
            if let Some(mm) = &plan.mmproj {
                println!("mmproj_out: {}", mm.output.display());
                println!("mmproj_precision: {}", mm.precision);
            }
            println!("scratch_space: LARGE / NOT ESTIMATED; intermediate F16/BF16 GGUF required");
            println!("runtime_load: NOT_TESTED");
            let done = quantize_hf(&plan)?;
            println!("status: HF_CONVERTER_AND_QUANTIZER_EXITED_SUCCESSFULLY");
            println!("output: {}", done.output.display());
            println!("output_bytes: {}", done.info.file_bytes);
            println!("verification: GGUF_METADATA_AND_TENSOR_CATALOG_ONLY");
            if let Some(mm) = &done.mmproj {
                println!("mmproj: {}", mm.as_json());
            }
            if let Some(receipt) = done.receipt {
                println!("receipt: {}", receipt.display());
            }
            Ok(true)
        }
        "backends" => {
            let mut json = false;
            let mut probe = false;
            for option in args {
                match option.to_str() {
                    Some("--json") if !json => json = true,
                    Some("--probe") if !probe => probe = true,
                    _ => return Err(invalid("backends accepts only --json and --probe")),
                }
            }
            let detected = discover_backends();
            if probe {
                let result = ojon_backend::probe_backend_capabilities(&detected);
                if json {
                    println!("{result}");
                } else {
                    for item in result["probes"].as_array().into_iter().flatten() {
                        println!(
                            "{}: {}\n  devices: {}\n  formats: {}",
                            item["name"].as_str().unwrap_or("backend"),
                            item["status"].as_str().unwrap_or("unknown"),
                            item["device_ids"],
                            item["precision_formats"]
                        );
                    }
                }
                return Ok(true);
            }
            if json {
                let arr: Vec<_> = detected.iter().map(|item| serde_json::json!({
                    "name": item.spec.name,
                    "role": item.spec.role,
                    "found": item.executable.is_some(),
                    "path": item.executable.as_ref().map(|p| p.to_string_lossy().into_owned()),
                    "source": item.found_via,
                    "interface_wired": item.spec.integration == Integration::Wired,
                    "compatibility_verified": false,
                    "official_url": item.spec.website,
                    "override_environment": item.spec.override_env,
                })).collect();
                println!(
                    "{}",
                    serde_json::json!({"schema":"ojon.backends.v1","backends":arr,"optional_optimizers":discover_advanced()["optimizers"],"diffusion_backends":discover_diffusion()["backends"]})
                );
                return Ok(true);
            }
            for item in detected {
                let status = match (&item.executable, item.spec.integration) {
                    (Some(_), Integration::Wired) => "DETECTED / ADAPTER CONNECTED",
                    (Some(_), Integration::DiscoveryOnly) => "DETECTED / ADAPTER NOT YET CONNECTED",
                    (None, Integration::Wired) => "NOT FOUND / MANUAL PATH OR DOWNLOAD",
                    (None, Integration::DiscoveryOnly) => "NOT FOUND / FUTURE ADAPTER",
                };
                println!("{}: {}", item.spec.name, status);
                println!("  purpose: {}", item.spec.role);
                if let Some(path) = &item.executable {
                    println!("  executable: {}", path.display());
                }
                if let Some(origin) = item.found_via {
                    println!("  discovery: {origin}");
                }
                println!("  publisher: {}", item.spec.website);
                println!("  override: {}", item.spec.override_env);
            }
            for opt in discover_advanced()["optimizers"]
                .as_array()
                .into_iter()
                .flatten()
            {
                println!(
                    "optional optimizer {}: checkout={}, hardware=UNVERIFIED",
                    opt["backend"].as_str().unwrap_or("?"),
                    if opt["source_code_present"].as_bool() == Some(true) {
                        "DETECTED"
                    } else {
                        "NOT FOUND"
                    }
                );
                println!(
                    "  publisher: {}",
                    opt["official_repository"].as_str().unwrap_or("?")
                );
                println!(
                    "  configure: {}",
                    opt["environment_variable"].as_str().unwrap_or("?")
                );
            }
            for backend in discover_diffusion()["backends"]
                .as_array()
                .into_iter()
                .flatten()
            {
                println!(
                    "diffusion backend {}: {}, image/video loader and quality NOT VERIFIED",
                    backend["backend"].as_str().unwrap_or("?"),
                    if backend["source_present"].as_bool() == Some(true) {
                        "CHECKOUT DETECTED"
                    } else {
                        "NOT FOUND"
                    }
                );
                println!(
                    "  configure: {}",
                    backend["environment_variable"].as_str().unwrap_or("?")
                );
                println!(
                    "  publisher: {}",
                    backend["upstream"].as_str().unwrap_or("?")
                );
            }
            println!("Native Ojon 2/3/4-bit model pack/unpack: built in, no RebelUI, Python or GPU required; direct GPU loading not yet implemented.");
            println!("Supported GGUF: public llama-quantize via K/IQ precision methods; optional GSQ/RCO local Python stages require explicit user configuration.");
            println!("Detected means filename found; binary version, model compatibility and runtime must still be verified.");
            println!("Add custom tool directories via OJON_BACKENDS_DIR or explicit --tool path.");
            Ok(true)
        }
        "smoke" => {
            let model = PathBuf::from(next_argument(&mut args, "GGUF model file")?);
            let mut tool = ojon_backend::discovered_path(BackendId::LlamaCli)
                .unwrap_or_else(|| PathBuf::from("llama-cli"));
            let mut tokens = 16u32;
            let mut gpu_layers = 0u32;
            let mut prompt = String::from("Reply with one short sentence: what is two plus two?");
            while let Some(option) = args.next() {
                match option.to_str() {
                    Some("--tool") => {
                        tool = PathBuf::from(next_argument(&mut args, "llama-cli executable")?)
                    }
                    Some("--tokens") => {
                        let os = next_argument(&mut args, "token count")?;
                        let str_value = os
                            .to_str()
                            .ok_or_else(|| invalid("token count must be text"))?;
                        tokens = str_value
                            .parse::<u32>()
                            .map_err(|_| invalid("invalid token count"))?;
                    }
                    Some("--gpu-layers") => {
                        let os = next_argument(&mut args, "GPU layer count")?;
                        let str_value = os
                            .to_str()
                            .ok_or_else(|| invalid("GPU layer count must be text"))?;
                        gpu_layers = str_value
                            .parse::<u32>()
                            .map_err(|_| invalid("invalid GPU layer count"))?;
                    }
                    Some("--prompt") => {
                        let os = next_argument(&mut args, "prompt text")?;
                        prompt = os
                            .to_str()
                            .ok_or_else(|| invalid("prompt must be valid Unicode"))?
                            .to_owned();
                    }
                    _ => {
                        return Err(invalid(format!(
                            "unknown smoke option: {}",
                            option.to_string_lossy()
                        )))
                    }
                }
            }
            smoke_gguf(&model, &tool, &prompt, tokens, gpu_layers)?;
            println!("runtime_smoke: LLAMA_CLI_EXITED_SUCCESSFULLY");
            println!("quality_retention: NOT_EVALUATED");
            println!("direct_baseline: NOT_RUN");
            Ok(true)
        }
        _ => Ok(false),
    }
}
