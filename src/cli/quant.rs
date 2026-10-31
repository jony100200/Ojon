//! Precision optimization, adaptive bit allocation, sweeps, parity, and benchmark commands.

use crate::cli::common::{invalid, next_argument, parse_plan, show_plan};
use ojon_backend::*;
use std::ffi::OsString;
use std::io;
use std::path::{Path, PathBuf};

pub fn handle(cmd: &str, mut args: impl Iterator<Item = OsString>) -> io::Result<bool> {
    match cmd {
        "ara-fit" => {
            let source = PathBuf::from(next_argument(
                &mut args,
                "module input/output calibration JSON",
            )?);
            let output = PathBuf::from(next_argument(
                &mut args,
                "output low-rank edit recipe JSON",
            )?);
            let (mut confirmed, mut json) = (false, false);
            for arg in args.by_ref() {
                match arg.to_str() {
                    Some("--confirm") => confirmed = true,
                    Some("--json") => json = true,
                    _ => {
                        return Err(invalid(format!(
                            "unknown ARA option: {}",
                            arg.to_string_lossy()
                        )))
                    }
                }
            }
            let result = ojon_backend::fit_ara_file(&source)?;
            if !confirmed {
                if json {
                    println!("{}", result.as_json());
                } else {
                    println!("plan_only: ARA-like rank {}, {} steps; add --confirm to save low-rank recipe",
                    result.rank,result.steps);
                }
            } else {
                let path = ojon_backend::save_ara_recipe(&result, &output)?;
                if json {
                    println!(
                        "{}",
                        serde_json::json!({"output":path,"status":"ARA_RECIPE_WRITTEN","refusal_rate":"NOT_EVALUATED","initial_loss":result.initial_loss,"final_loss":result.final_loss})
                    );
                } else {
                    println!("ARA-like adapter recipe: {}", path.display());
                    println!(
                        "module_loss: {} -> {}",
                        result.initial_loss, result.final_loss
                    );
                    println!("refusal_rate: NOT_EVALUATED");
                }
            }
            Ok(true)
        }
        "method-select" => {
            let path = PathBuf::from(next_argument(&mut args, "measured quality evidence JSON")?);
            for option in args {
                match option.to_str() {
                    // Output is always the JSON receipt; the flag stays accepted for CLI parity.
                    Some("--json") => {}
                    _ => {
                        return Err(invalid(format!(
                            "unknown selection option: {}",
                            option.to_string_lossy()
                        )))
                    }
                }
            }
            let result = ojon_backend::choose_edit_method_file(&path)?;
            println!("{}", result.as_json());
            Ok(true)
        }
        "adapter-smoke-plan" | "adapter-smoke-run" => {
            let execute = cmd == "adapter-smoke-run";
            let bundle = PathBuf::from(next_argument(
                &mut args,
                "Ojon compressed adapter bundle folder",
            )?);
            let report = PathBuf::from(next_argument(&mut args, "new JSON receipt file")?);
            let mut tool =
                ojon_backend::discovery::resolve_backend_command(BackendId::LlamaCli, "llama-cli")?;
            let mut gpu_layers = 0i32; // GPU-free by default.
            let mut device = None::<String>;
            let mut image = None::<PathBuf>;
            let mut prompt = None::<String>;
            let mut tokens = 64u32;
            let (mut confirmed, mut json) = (false, false);
            while let Some(option) = args.next() {
                match option.to_str() {
                    Some("--tool") => {
                        tool = PathBuf::from(next_argument(&mut args, "llama-cli executable")?)
                    }
                    Some("--gpu-layers") => {
                        gpu_layers = next_argument(&mut args, "GPU layers")?
                            .to_string_lossy()
                            .parse()
                            .map_err(|_| invalid("gpu layers must be an integer"))?;
                    }
                    Some("--device") => {
                        device = Some(
                            next_argument(&mut args, "device name")?
                                .to_string_lossy()
                                .to_string(),
                        )
                    }
                    Some("--image") => {
                        image = Some(PathBuf::from(next_argument(&mut args, "image")?))
                    }
                    Some("--prompt") => {
                        prompt = Some(
                            next_argument(&mut args, "prompt")?
                                .to_string_lossy()
                                .to_string(),
                        )
                    }
                    Some("--tokens") => {
                        tokens = next_argument(&mut args, "token limit")?
                            .to_string_lossy()
                            .parse()
                            .map_err(|_| invalid("tokens must be a positive integer"))?;
                    }
                    Some("--confirm") => confirmed = true,
                    Some("--json") => json = true,
                    _ => {
                        return Err(invalid(format!(
                            "unknown adapter smoke option: {}",
                            option.to_string_lossy()
                        )))
                    }
                }
            }
            if confirmed && !execute {
                return Err(invalid("--confirm may only be used with adapter-smoke-run"));
            }
            let plan = ojon_backend::prepare_adapter_smoke(
                ojon_backend::AdapterSmokeRequest {
                    bundle: &bundle,
                    report: &report,
                    tool: &tool,
                    gpu_layers,
                    device: device.as_deref(),
                    image: image.as_deref(),
                    prompt: prompt.as_deref(),
                    tokens,
                },
            )?;
            if execute {
                if !confirmed {
                    return Err(invalid("adapter-smoke-run requires --confirm"));
                }
                let result = ojon_backend::run_adapter_smoke(&plan, |line| eprintln!("{line}"))?;
                if json {
                    println!("{}", result.as_json());
                } else {
                    println!("smoke_receipt: {}", result.report.display());
                    println!("llama-cli process exited successfully");
                    println!("model_compatibility: RUNTIME_PROCESS_SUCCESS");
                    println!("refusal_rate: NOT_EVALUATED");
                    println!("vision_quality: NOT_EVALUATED");
                }
            } else {
                println!("{}", plan.as_json());
            }
            Ok(true)
        }
        "compressed-ara-plan" | "compressed-ara-run" => {
            let execute = cmd == "compressed-ara-run";
            let base = PathBuf::from(next_argument(&mut args, "already-quantized GGUF base")?);
            let calibration =
                PathBuf::from(next_argument(&mut args, "recorded module I/O calibration")?);
            let output = PathBuf::from(next_argument(&mut args, "new adapter folder")?);
            let mut mmproj = None::<PathBuf>;
            let (mut confirmed, mut json) = (false, false);
            while let Some(option) = args.next() {
                match option.to_str() {
                    Some("--mmproj") => {
                        mmproj = Some(PathBuf::from(next_argument(&mut args, "mmproj GGUF")?))
                    }
                    Some("--confirm") => confirmed = true,
                    Some("--json") => json = true,
                    _ => {
                        return Err(invalid(format!(
                            "unknown compressed ARA option: {}",
                            option.to_string_lossy()
                        )))
                    }
                }
            }
            let fitted = ojon_backend::fit_ara_file(&calibration)?;
            let plan = ojon_backend::prepare_compressed_edit_with_recipe(
                &base,
                &fitted.recipe,
                &output,
                mmproj.as_deref(),
            )?;
            if execute {
                if !confirmed {
                    return Err(invalid("compressed-ara-run requires --confirm"));
                }
                let result = ojon_backend::run_compressed_edit(&plan, |line| eprintln!("{line}"))?;
                if json {
                    println!(
                        "{}",
                        serde_json::json!({
                            "adapter":result.as_json(),
                            "fit":{"initial_module_loss":fitted.initial_loss,"final_module_loss":fitted.final_loss,
                            "rank":fitted.rank,"steps":fitted.steps,
                            "refusal_rate":"NOT_EVALUATED","optimizer":"RUST_ADAM_PROXY"}
                        })
                    );
                } else {
                    println!("adapter: {}", result.output.join("adapter.gguf").display());
                    println!(
                        "module_loss: {} -> {}",
                        fitted.initial_loss, fitted.final_loss
                    );
                    println!("base GGUF unchanged; model load, refusal and vision tests NOT RUN");
                }
            } else if json {
                println!(
                    "{}",
                    serde_json::json!({"plan":plan.as_json(),
                    "fit_module_loss":{"before":fitted.initial_loss,"after":fitted.final_loss},
                    "refusal_quality":"NOT_EVALUATED"})
                );
            } else {
                println!("{}", plan.as_json());
            }
            Ok(true)
        }
        "bench-plan" | "bench-run" => {
            let execute = cmd == "bench-run";
            let model = PathBuf::from(next_argument(
                &mut args,
                "GGUF model for performance measurement",
            )?);
            let receipt = if execute {
                Some(PathBuf::from(next_argument(
                    &mut args,
                    "new JSON receipt path",
                )?))
            } else {
                None
            };
            let mut tool = ojon_backend::discovery::resolve_backend_command(
                BackendId::LlamaBench,
                "llama-bench",
            )?;
            let (mut prompt_tokens, mut gen_tokens, mut reps, mut gpu_layers) =
                (512u32, 128u32, 3u32, 0i32);
            let mut confirmed = false;
            let mut device: Option<String> = None;
            while let Some(arg) = args.next() {
                match arg.to_str() {
                    Some("--tool") => {
                        tool = PathBuf::from(next_argument(&mut args, "llama-bench executable")?)
                    }
                    Some("--prompt-tokens") => {
                        prompt_tokens = next_argument(&mut args, "prompt token count")?
                            .to_str()
                            .ok_or_else(|| invalid("prompt count must be text"))?
                            .parse()
                            .map_err(|_| invalid("prompt tokens require integer"))?
                    }
                    Some("--gen-tokens") => {
                        gen_tokens = next_argument(&mut args, "generation token count")?
                            .to_str()
                            .ok_or_else(|| invalid("generation count must be text"))?
                            .parse()
                            .map_err(|_| invalid("generation tokens require integer"))?
                    }
                    Some("--repetitions") => {
                        reps = next_argument(&mut args, "repetitions")?
                            .to_str()
                            .ok_or_else(|| invalid("repetitions must be text"))?
                            .parse()
                            .map_err(|_| invalid("repetitions require integer"))?
                    }
                    Some("--gpu-layers") => {
                        gpu_layers = next_argument(&mut args, "GPU layer count")?
                            .to_str()
                            .ok_or_else(|| invalid("GPU layers must be text"))?
                            .parse()
                            .map_err(|_| invalid("GPU layers require integer"))?
                    }
                    Some("--device") => {
                        device = Some(
                            next_argument(&mut args, "explicit llama-bench device")?
                                .to_str()
                                .ok_or_else(|| invalid("device must be text"))?
                                .to_owned(),
                        )
                    }
                    Some("--confirm") if execute => confirmed = true,
                    Some("--json") => {}
                    _ => {
                        return Err(invalid(format!(
                            "unknown benchmark option: {}",
                            arg.to_string_lossy()
                        )))
                    }
                }
            }
            let plan = prepare_bench(
                &model,
                &tool,
                prompt_tokens,
                gen_tokens,
                reps,
                gpu_layers,
                device.as_deref(),
            )?;
            if !execute {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&plan.as_json()).map_err(io::Error::other)?
                );
            } else {
                if !confirmed {
                    return Err(invalid("real llama-bench execution loads weights; add --confirm after reviewing model, installed tool and available RAM/VRAM"));
                }
                let path = receipt.expect("bench-run supplies a receipt");
                let record = run_bench(&plan, &path)?;
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "saved_receipt": path.to_string_lossy(),
                        "benchmark":record
                    }))
                    .map_err(io::Error::other)?
                );
            }
            Ok(true)
        }
        "bench-compare" => {
            let reference = PathBuf::from(next_argument(&mut args, "baseline receipt JSON")?);
            let candidate = PathBuf::from(next_argument(&mut args, "candidate receipt JSON")?);
            for arg in args {
                if arg != "--json" {
                    return Err(invalid(format!(
                        "unknown benchmark comparison option: {}",
                        arg.to_string_lossy()
                    )));
                }
            }
            let value = compare_bench_files(&reference, &candidate)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&value).map_err(io::Error::other)?
            );
            Ok(true)
        }
        "gsq-plan" | "gsq-run" => {
            let execute = cmd == "gsq-run";
            let checkout =
                PathBuf::from(next_argument(&mut args, "public GSQ repository checkout")?);
            let config = PathBuf::from(next_argument(&mut args, "GSQ local YAML configuration")?);
            let mut python = PathBuf::from("python");
            let mut max_layers = Some(1u32);
            let mut confirm = false;
            while let Some(arg) = args.next() {
                match arg.to_str() {
                    Some("--python") => {
                        python = PathBuf::from(next_argument(&mut args, "Python executable")?)
                    }
                    Some("--layers") => {
                        max_layers = Some(
                            next_argument(&mut args, "GSQ max layers")?
                                .to_str()
                                .ok_or_else(|| invalid("layers require integer"))?
                                .parse()
                                .map_err(|_| invalid("invalid GSQ layer count"))?,
                        )
                    }
                    Some("--all-layers") => max_layers = None,
                    Some("--confirm") if execute => confirm = true,
                    _ => {
                        return Err(invalid(format!(
                            "unknown GSQ option: {}",
                            arg.to_string_lossy()
                        )))
                    }
                }
            }
            let plan = plan_gsq(&checkout, &config, &python, max_layers)?;
            if !execute {
                println!("{}", plan.json());
            } else {
                if !confirm {
                    return Err(invalid("GSQ execution requires --confirm after reviewing source, dataset, VRAM and output settings"));
                }
                println!("{}", plan.json());
                run_advanced(&plan, |line| eprintln!("{line}"))?;
                println!("gsq_stage: external_process_exited; model_quality: NOT_TESTED");
            }
            Ok(true)
        }
        "gsq-assemble-plan" | "gsq-assemble" => {
            let execute = cmd == "gsq-assemble";
            let checkout = PathBuf::from(next_argument(&mut args, "public GSQ checkout")?);
            let config = PathBuf::from(next_argument(&mut args, "GSQ YAML config")?);
            let output = PathBuf::from(next_argument(&mut args, "new output directory")?);
            let mut python = PathBuf::from("python");
            let mut confirm = false;
            while let Some(arg) = args.next() {
                match arg.to_str() {
                    Some("--python") => python = PathBuf::from(next_argument(&mut args, "Python")?),
                    Some("--confirm") if execute => confirm = true,
                    _ => {
                        return Err(invalid(format!(
                            "unknown GSQ assembly option: {}",
                            arg.to_string_lossy()
                        )))
                    }
                }
            }
            let plan = plan_gsq_assemble(&checkout, &config, &output, &python)?;
            if !execute {
                println!("{}", plan.json());
            } else {
                if !confirm {
                    return Err(invalid(
                        "assembly writes a model checkpoint; pass --confirm",
                    ));
                }
                println!("{}", plan.json());
                run_advanced(&plan, |line| eprintln!("{line}"))?;
                println!("gsq_output: {}", output.display());
                println!("loadability_and_quality: NOT_VERIFIED");
            }
            Ok(true)
        }
        "rco-db-plan" | "rco-db-run" => {
            let execute = cmd == "rco-db-run";
            let checkout = PathBuf::from(next_argument(&mut args, "public RCO checkout")?);
            let model = PathBuf::from(next_argument(
                &mut args,
                "original local dense Qwen3 model",
            )?);
            let output =
                PathBuf::from(next_argument(&mut args, "new RCO database root directory")?);
            let mut python = PathBuf::from("python");
            let mut tokens = 16384u32;
            let mut confirm = false;
            while let Some(arg) = args.next() {
                match arg.to_str() {
                    Some("--python") => {
                        python = PathBuf::from(next_argument(&mut args, "Python executable")?)
                    }
                    Some("--tokens") => {
                        tokens = next_argument(&mut args, "calibration token count")?
                            .to_str()
                            .ok_or_else(|| invalid("tokens must be integer"))?
                            .parse()
                            .map_err(|_| invalid("invalid calibration token count"))?
                    }
                    Some("--confirm") if execute => confirm = true,
                    _ => {
                        return Err(invalid(format!(
                            "unknown RCO database option: {}",
                            arg.to_string_lossy()
                        )))
                    }
                }
            }
            let plan = plan_rco_database(&checkout, &model, &output, &python, tokens)?;
            if !execute {
                println!("{}", plan.json());
            } else {
                if !confirm {
                    return Err(invalid("RCO database construction uses CUDA and substantial resources; pass --confirm after verifying the plan"));
                }
                println!("{}", plan.json());
                run_advanced(&plan, |line| eprintln!("{line}"))?;
                println!("rco_database: external_backend_exited; database_contents: REVIEW_BEFORE_SEARCH");
            }
            Ok(true)
        }
        "rco-plan" | "rco-run" => {
            let execute = cmd == "rco-run";
            let checkout = PathBuf::from(next_argument(&mut args, "public RCO checkout")?);
            let model = PathBuf::from(next_argument(&mut args, "original local HF model folder")?);
            let database =
                PathBuf::from(next_argument(&mut args, "prebuilt per-bit layer database")?);
            let bits = next_argument(&mut args, "target average bits")?
                .to_str()
                .ok_or_else(|| invalid("target bits must be a number"))?
                .parse::<f64>()
                .map_err(|_| invalid("target bits must be numeric"))?;
            let mut python = PathBuf::from("python");
            let mut steps = 30u32;
            let mut confirm = false;
            while let Some(arg) = args.next() {
                match arg.to_str() {
                    Some("--python") => {
                        python = PathBuf::from(next_argument(&mut args, "Python executable")?)
                    }
                    Some("--steps") => {
                        steps = next_argument(&mut args, "optimization steps")?
                            .to_str()
                            .ok_or_else(|| invalid("steps require an integer"))?
                            .parse()
                            .map_err(|_| invalid("RCO steps must be numeric"))?
                    }
                    Some("--confirm") if execute => confirm = true,
                    _ => {
                        return Err(invalid(format!(
                            "unknown RCO option: {}",
                            arg.to_string_lossy()
                        )))
                    }
                }
            }
            let plan = plan_rco_search(&checkout, &model, &database, bits, &python, steps)?;
            if !execute {
                println!("{}", plan.json());
            } else {
                if !confirm {
                    return Err(invalid(
                        "RCO search uses local compute/disk; pass --confirm",
                    ));
                }
                println!("{}", plan.json());
                run_advanced(&plan, |line| eprintln!("{line}"))?;
                println!("rco_search: completed_external_process; output_assignment: inspect_layer_database");
                println!("model_checkpoint_and_quality: NOT_VERIFIED");
            }
            Ok(true)
        }
        "rco-assemble-plan" | "rco-assemble" => {
            let execute = cmd == "rco-assemble";
            let checkout = PathBuf::from(next_argument(&mut args, "public RCO checkout")?);
            let model = PathBuf::from(next_argument(&mut args, "original local HF model folder")?);
            let database = PathBuf::from(next_argument(&mut args, "prebuilt layer database")?);
            let assignment =
                PathBuf::from(next_argument(&mut args, "RCO search assignment JSON/TXT")?);
            let output = PathBuf::from(next_argument(&mut args, "new output directory")?);
            let mut python = PathBuf::from("python");
            let mut confirm = false;
            while let Some(arg) = args.next() {
                match arg.to_str() {
                    Some("--python") => {
                        python = PathBuf::from(next_argument(&mut args, "Python executable")?)
                    }
                    Some("--confirm") if execute => confirm = true,
                    _ => {
                        return Err(invalid(format!(
                            "unknown RCO assembly option: {}",
                            arg.to_string_lossy()
                        )))
                    }
                }
            }
            let plan =
                plan_rco_assemble(&checkout, &model, &database, &assignment, &output, &python)?;
            if !execute {
                println!("{}", plan.json());
            } else {
                if !confirm {
                    return Err(invalid(
                        "assembly writes compressed-tensors checkpoint; pass --confirm",
                    ));
                }
                println!("{}", plan.json());
                run_advanced(&plan, |line| eprintln!("{line}"))?;
                println!("rco_output: {}", output.display());
                println!("format: HF_COMPRESSED_TENSORS; GGUF_NOT_IMPLIED; quality: NOT_TESTED");
            }
            Ok(true)
        }
        "imatrix-plan" | "imatrix-run" => {
            let execute = cmd == "imatrix-run";
            let model = PathBuf::from(next_argument(&mut args, "original F16/BF16 GGUF model")?);
            let corpus = PathBuf::from(next_argument(&mut args, "local UTF-8 calibration text")?);
            let output = PathBuf::from(next_argument(&mut args, "new imatrix.gguf output")?);
            let mut tool = ojon_backend::discovery::resolve_backend_command(
                BackendId::LlamaImatrix,
                "llama-imatrix",
            )?;
            let mut gpu_layers = 0u32;
            let mut context = 2048u32;
            let mut confirmed = false;
            while let Some(option) = args.next() {
                match option.to_str() {
                    Some("--tool") => {
                        tool = PathBuf::from(next_argument(
                            &mut args,
                            "public llama-imatrix executable",
                        )?)
                    }
                    Some("--gpu-layers") => {
                        gpu_layers = next_argument(&mut args, "GPU layer offload count")?
                            .to_str()
                            .ok_or_else(|| invalid("GPU layer count must be numeric"))?
                            .parse()
                            .map_err(|_| invalid("GPU layers must be integer"))?;
                    }
                    Some("--context") => {
                        context = next_argument(&mut args, "calibration context")?
                            .to_str()
                            .ok_or_else(|| invalid("context must be numeric"))?
                            .parse()
                            .map_err(|_| invalid("context must be integer"))?;
                    }
                    Some("--confirm") if execute => confirmed = true,
                    _ => {
                        return Err(invalid(format!(
                            "unknown imatrix option: {}",
                            option.to_string_lossy()
                        )))
                    }
                }
            }
            let plan = prepare_imatrix(&model, &corpus, &output, &tool, gpu_layers, context)?;
            if !execute {
                println!("{}", plan.json());
            } else {
                if !confirmed {
                    return Err(invalid("importance-matrix calibration loads the original model and can consume significant RAM/VRAM; pass --confirm"));
                }
                println!("{}", plan.json());
                let saved = run_imatrix(&plan, |line| eprintln!("{line}"))?;
                println!("imatrix: {}", saved.display());
                println!("next_step: use --imatrix with ojon sweep-plan / sweep-run");
                println!("quality: NOT_EVALUATED; held_out_task_tests_required");
            }
            Ok(true)
        }
        "eval-sweep" => {
            let manifest =
                PathBuf::from(next_argument(&mut args, "completed Ojon sweep manifest")?);
            let corpus =
                PathBuf::from(next_argument(&mut args, "held-out evaluation text corpus")?);
            let mut settings = PplSettings {
                tool: discovered_path(BackendId::LlamaPerplexity)
                    .unwrap_or_else(|| PathBuf::from("llama-perplexity")),
                corpus,
                gpu_layers: 0,
                context: 2048,
                max_relative_degradation_pct: 1.0,
            };
            let mut confirmed = false;
            while let Some(arg) = args.next() {
                match arg.to_str() {
                    Some("--tool") => {
                        settings.tool =
                            PathBuf::from(next_argument(&mut args, "perplexity executable")?)
                    }
                    Some("--gpu-layers") => {
                        settings.gpu_layers = next_argument(&mut args, "GPU offload layer count")?
                            .to_str()
                            .ok_or_else(|| invalid("GPU layers must be UTF-8 digits"))?
                            .parse()
                            .map_err(|_| invalid("GPU layers must be an integer"))?;
                    }
                    Some("--context") => {
                        settings.context = next_argument(&mut args, "context size")?
                            .to_str()
                            .ok_or_else(|| invalid("context must be UTF-8 digits"))?
                            .parse()
                            .map_err(|_| invalid("context size must be an integer"))?;
                    }
                    Some("--max-degradation") => {
                        settings.max_relative_degradation_pct =
                            next_argument(&mut args, "max relative PPL increase")?
                                .to_str()
                                .ok_or_else(|| invalid("degradation must be text"))?
                                .parse()
                                .map_err(|_| invalid("degradation must be numeric"))?;
                    }
                    Some("--confirm") => confirmed = true,
                    _ => {
                        return Err(invalid(format!(
                            "unknown eval-sweep option: {}",
                            arg.to_string_lossy()
                        )))
                    }
                }
            }
            if !confirmed {
                return Err(invalid("eval-sweep loads all sweep variants sequentially and may use substantial RAM/VRAM; pass --confirm"));
            }
            settings.validate()?;
            let result = evaluate_sweep(&manifest, &settings, |line| eprintln!("{line}"))?;
            println!("perplexity_report: {}", result.display());
            println!("overall_task_quality_status: NOT_EVALUATED_BY_THIS_COMMAND");
            println!("recommendation: PPL_ONLY_PROVISIONAL");
            Ok(true)
        }
        "sweep-plan" | "sweep-run" => {
            let execute = cmd == "sweep-run";
            let source = PathBuf::from(next_argument(&mut args, "original F16/BF16 GGUF model")?);
            let destination = PathBuf::from(next_argument(&mut args, "existing output directory")?);
            let mut formats = String::from("Q5_K_M,Q4_K_M,IQ4_XS,IQ3_M");
            let mut imatrix: Option<PathBuf> = None;
            let mut tool = ojon_backend::discovery::resolve_backend_command(
                BackendId::LlamaQuantize,
                "llama-quantize",
            )?;
            let mut threads = std::thread::available_parallelism()
                .map(|v| v.get())
                .unwrap_or(4);
            let mut confirm = false;
            while let Some(option) = args.next() {
                match option.to_str() {
                    Some("--precisions") => {
                        let v = next_argument(&mut args, "comma-separated precision list")?;
                        formats = v
                            .to_str()
                            .ok_or_else(|| invalid("precisions require UTF-8"))?
                            .to_owned();
                    }
                    Some("--imatrix") => {
                        imatrix = Some(PathBuf::from(next_argument(
                            &mut args,
                            "importance matrix file",
                        )?));
                    }
                    Some("--tool") => {
                        tool = PathBuf::from(next_argument(&mut args, "converter executable")?)
                    }
                    Some("--threads") => {
                        let value = next_argument(&mut args, "thread count")?;
                        threads = value
                            .to_str()
                            .ok_or_else(|| invalid("invalid thread count"))?
                            .parse::<usize>()
                            .map_err(|_| invalid("thread count must be a number"))?;
                    }
                    Some("--confirm") if execute => confirm = true,
                    _ => {
                        return Err(invalid(format!(
                            "unknown sweep option: {}",
                            option.to_string_lossy()
                        )))
                    }
                }
            }
            let types = parse_precisions(&formats)?;
            let plan = prepare_sweep(
                &source,
                &destination,
                &tool,
                &types,
                imatrix.as_deref(),
                threads,
            )?;
            if !execute {
                println!("{}", plan.json());
            } else {
                if !confirm {
                    return Err(invalid("sweep-run executes multiple model conversions and uses significant RAM/disk; pass --confirm to proceed"));
                }
                eprintln!("Ojon: executing {} sequential model conversions. Original source is read-only.", types.len());
                let path = execute_sweep(&plan, |line| eprintln!("{line}"))?;
                println!("sweep_manifest: {}", path.display());
                println!("quantization_task_quality: NOT_EVALUATED");
                println!("recommended_variant: NONE_UNTIL_REAL_BENCHMARKS");
            }
            Ok(true)
        }
        "sensitivity-extract" => {
            let source = PathBuf::from(next_argument(&mut args, "original GGUF")?);
            let imatrix = PathBuf::from(next_argument(&mut args, "GGUF calibration imatrix")?);
            let output = PathBuf::from(next_argument(&mut args, "new sensitivity JSON")?);
            let mut confirm = false;
            for arg in args.by_ref() {
                match arg.to_str() {
                    Some("--confirm") => confirm = true,
                    _ => return Err(invalid("sensitivity-extract supports only --confirm")),
                }
            }
            if !confirm {
                return Err(invalid(
                    "sensitivity-extract creates new output; pass --confirm",
                ));
            }
            let result = extract_sensitivity_from_imatrix(&source, &imatrix, &output)?;
            println!(
                "sensitivity_entries: {}",
                result["tensors"].as_array().map(|v| v.len()).unwrap_or(0)
            );
            println!("output: {}", output.display());
            println!("status: ACTIVATION_ENERGY_PROXY_NOT_TASK_SENSITIVITY");
            Ok(true)
        }
        "quant-compare-plan" | "quant-compare-run" => {
            let execute = cmd == "quant-compare-run";
            let baseline =
                PathBuf::from(next_argument(&mut args, "existing Q4_K_M reference GGUF")?);
            let suite = PathBuf::from(next_argument(&mut args, "held-out task suite JSON")?);
            let report = PathBuf::from(next_argument(&mut args, "new comparison report JSON")?);
            let mut candidates = Vec::<(String, PathBuf)>::new();
            let mut options = ojon_backend::QuantParityOptions {
                quantizer: ojon_backend::discovery::resolve_backend_command(
                    BackendId::LlamaQuantize,
                    "llama-quantize",
                )?,
                eval_tool: ojon_backend::discovery::resolve_backend_command(
                    BackendId::LlamaCli,
                    "llama-cli",
                )?,
                server_tool: ojon_backend::discovered_path(BackendId::LlamaServer),
                ppl_tool: ojon_backend::discovery::resolve_backend_command(
                    BackendId::LlamaPerplexity,
                    "llama-perplexity",
                )?,
                ..ojon_backend::QuantParityOptions::default()
            };
            let (mut confirmed, mut json) = (false, false);
            while let Some(option) = args.next() {
                match option.to_str() {
                    Some("--candidate") => {
                        let raw = next_argument(&mut args, "name=path to candidate GGUF")?;
                        let raw = raw
                            .to_str()
                            .ok_or_else(|| invalid("candidate must be UTF-8"))?;
                        let (name, path) = raw
                            .split_once('=')
                            .ok_or_else(|| invalid("candidate must be name=path.gguf"))?;
                        if name.is_empty() || path.is_empty() {
                            return Err(invalid("invalid candidate path"));
                        }
                        candidates.push((name.to_owned(), PathBuf::from(path)));
                    }
                    Some("--eval-tool") => {
                        options.eval_tool =
                            PathBuf::from(next_argument(&mut args, "llama-cli path")?)
                    }
                    Some("--server-tool") => {
                        options.server_tool = Some(PathBuf::from(next_argument(
                            &mut args,
                            "llama-server executable",
                        )?))
                    }
                    Some("--gpu-layers") => {
                        options.gpu_layers = next_argument(&mut args, "GPU layers")?
                            .to_string_lossy()
                            .parse()
                            .map_err(|_| invalid("gpu-layers must be integer"))?;
                    }
                    Some("--device") => {
                        options.device = Some(
                            next_argument(&mut args, "GPU device")?
                                .to_string_lossy()
                                .to_string(),
                        )
                    }
                    Some("--tokens") => {
                        options.tokens = next_argument(&mut args, "generation tokens")?
                            .to_string_lossy()
                            .parse()
                            .map_err(|_| invalid("tokens must be integer"))?;
                    }
                    Some("--max-loss-points") => {
                        options.max_loss_points =
                            next_argument(&mut args, "max loss percentage points")?
                                .to_string_lossy()
                                .parse()
                                .map_err(|_| invalid("loss must be numeric"))?;
                    }
                    Some("--min-saving-percent") => {
                        options.min_saving_percent =
                            next_argument(&mut args, "minimum smaller-size percent")?
                                .to_string_lossy()
                                .parse()
                                .map_err(|_| invalid("savings must be numeric"))?;
                    }
                    Some("--vision-suite") => {
                        options.vision_suite =
                            Some(PathBuf::from(next_argument(&mut args, "vision suite")?))
                    }
                    Some("--mmproj") => {
                        options.mmproj = Some(PathBuf::from(next_argument(&mut args, "mmproj")?))
                    }
                    Some("--ppl-corpus") => {
                        options.ppl_corpus = Some(PathBuf::from(next_argument(
                            &mut args,
                            "held-out text corpus",
                        )?))
                    }
                    Some("--ppl-tool") => {
                        options.ppl_tool =
                            PathBuf::from(next_argument(&mut args, "llama-perplexity tool")?)
                    }
                    Some("--ppl-context") => {
                        options.ppl_context = next_argument(&mut args, "perplexity context")?
                            .to_string_lossy()
                            .parse()
                            .map_err(|_| invalid("PPL context must be integer"))?;
                    }
                    Some("--max-ppl-regression-pct") => {
                        options.max_ppl_regression_pct =
                            next_argument(&mut args, "PPL regression limit")?
                                .to_string_lossy()
                                .parse()
                                .map_err(|_| invalid("PPL regression must be numeric"))?;
                    }
                    Some("--json") => json = true,
                    Some("--confirm") => confirmed = true,
                    _ => {
                        return Err(invalid(format!(
                            "unknown comparison option: {}",
                            option.to_string_lossy()
                        )))
                    }
                }
            }
            if confirmed && !execute {
                return Err(invalid("--confirm only works with quant-compare-run"));
            }
            let plan = ojon_backend::prepare_existing_quant_comparison(
                &baseline,
                &candidates,
                &suite,
                &report,
                options,
            )?;
            if execute {
                if !confirmed {
                    return Err(invalid("quant-compare-run requires --confirm"));
                }
                let result =
                    ojon_backend::run_existing_quant_comparison(&plan, |msg| eprintln!("{msg}"))?;
                if json {
                    println!("{result}");
                } else {
                    println!("comparison_report: {}", report.display());
                    println!(
                        "smallest_provisional_candidate: {}",
                        result["provisional_smallest_candidate"]
                    );
                    println!("global_quality_loss_proven_absent: NO");
                }
            } else {
                println!("{}", plan.as_json());
            }
            Ok(true)
        }
        "quant-parity-plan" | "quant-parity-run" => {
            let execute = cmd == "quant-parity-run";
            let source = PathBuf::from(next_argument(&mut args, "original high-precision GGUF")?);
            let baseline = PathBuf::from(next_argument(
                &mut args,
                "original teacher or quantized Q4 GGUF reference",
            )?);
            let suite = PathBuf::from(next_argument(
                &mut args,
                "held-out coding/reasoning/tool suite JSON",
            )?);
            let output_dir = PathBuf::from(next_argument(&mut args, "existing output directory")?);
            let mut options = ojon_backend::QuantParityOptions {
                quantizer: ojon_backend::discovery::resolve_backend_command(
                    BackendId::LlamaQuantize,
                    "llama-quantize",
                )?,
                eval_tool: ojon_backend::discovery::resolve_backend_command(
                    BackendId::LlamaCli,
                    "llama-cli",
                )?,
                server_tool: ojon_backend::discovered_path(BackendId::LlamaServer),
                ppl_tool: ojon_backend::discovery::resolve_backend_command(
                    BackendId::LlamaPerplexity,
                    "llama-perplexity",
                )?,
                ..ojon_backend::QuantParityOptions::default()
            };
            let (mut confirmed, mut json) = (false, false);
            while let Some(opt) = args.next() {
                match opt.to_str() {
                    Some("--precisions") => {
                        let raw = next_argument(&mut args, "comma-separated precisions")?;
                        let text = raw
                            .to_str()
                            .ok_or_else(|| invalid("precision types must be UTF-8"))?;
                        options.precisions = text
                            .split(',')
                            .map(|s| s.parse::<Precision>().map_err(invalid))
                            .collect::<std::io::Result<Vec<_>>>()?;
                    }
                    Some("--imatrix") => {
                        options.imatrix =
                            Some(PathBuf::from(next_argument(&mut args, "imatrix path")?))
                    }
                    Some("--quantizer") => {
                        options.quantizer =
                            PathBuf::from(next_argument(&mut args, "llama-quantize path")?)
                    }
                    Some("--eval-tool") => {
                        options.eval_tool =
                            PathBuf::from(next_argument(&mut args, "llama-cli path")?)
                    }
                    Some("--server-tool") => {
                        options.server_tool = Some(PathBuf::from(next_argument(
                            &mut args,
                            "llama-server executable",
                        )?))
                    }
                    Some("--threads") => {
                        options.threads = next_argument(&mut args, "threads")?
                            .to_string_lossy()
                            .parse()
                            .map_err(|_| invalid("threads must be an integer"))?;
                    }
                    Some("--max-buffer-mib") => {
                        options.max_buffer_mib = Some(
                            next_argument(&mut args, "quantizer slab buffer MiB")?
                                .to_string_lossy()
                                .parse()
                                .map_err(|_| invalid("buffer size must be an integer"))?,
                        );
                    }
                    Some("--no-buffer-limit") => options.max_buffer_mib = None,
                    Some("--gpu-layers") => {
                        options.gpu_layers = next_argument(&mut args, "GPU layers")?
                            .to_string_lossy()
                            .parse()
                            .map_err(|_| invalid("GPU layers must be an integer"))?;
                    }
                    Some("--device") => {
                        options.device = Some(
                            next_argument(&mut args, "device")?
                                .to_string_lossy()
                                .to_string(),
                        )
                    }
                    Some("--tokens") => {
                        options.tokens = next_argument(&mut args, "tokens")?
                            .to_string_lossy()
                            .parse()
                            .map_err(|_| invalid("tokens must be an integer"))?;
                    }
                    Some("--max-loss-points") => {
                        options.max_loss_points =
                            next_argument(&mut args, "maximum capability loss points")?
                                .to_string_lossy()
                                .parse()
                                .map_err(|_| invalid("loss must be numeric"))?;
                    }
                    Some("--min-saving-percent") => {
                        options.min_saving_percent =
                            next_argument(&mut args, "minimum saving percent")?
                                .to_string_lossy()
                                .parse()
                                .map_err(|_| invalid("saving must be numeric"))?;
                    }
                    Some("--ppl-corpus") => {
                        options.ppl_corpus = Some(PathBuf::from(next_argument(
                            &mut args,
                            "held-out perplexity corpus",
                        )?));
                    }
                    Some("--ppl-tool") => {
                        options.ppl_tool =
                            PathBuf::from(next_argument(&mut args, "llama-perplexity tool")?);
                    }
                    Some("--ppl-context") => {
                        options.ppl_context =
                            next_argument(&mut args, "perplexity context tokens")?
                                .to_string_lossy()
                                .parse()
                                .map_err(|_| invalid("PPL context must be integer"))?;
                    }
                    Some("--max-ppl-regression-pct") => {
                        options.max_ppl_regression_pct =
                            next_argument(&mut args, "max PPL regression percent")?
                                .to_string_lossy()
                                .parse()
                                .map_err(|_| invalid("PPL regression must be numeric"))?;
                    }
                    Some("--vision-suite") => {
                        options.vision_suite = Some(PathBuf::from(next_argument(
                            &mut args,
                            "vision suite JSON",
                        )?));
                    }
                    Some("--mmproj") => {
                        options.mmproj = Some(PathBuf::from(next_argument(
                            &mut args,
                            "matching vision projector GGUF",
                        )?));
                    }
                    Some("--mmproj-candidate") => {
                        options.mmproj_candidate = Some(PathBuf::from(next_argument(
                            &mut args,
                            "candidate-side vision projector GGUF",
                        )?));
                    }
                    Some("--vision-tool") => {
                        options.vision_tool = Some(PathBuf::from(next_argument(
                            &mut args,
                            "llama-mtmd-cli path",
                        )?));
                    }
                    Some("--json") => json = true,
                    Some("--confirm") => confirmed = true,
                    _ => {
                        return Err(invalid(format!(
                            "unknown quant-parity option: {}",
                            opt.to_string_lossy()
                        )))
                    }
                }
            }
            if confirmed && !execute {
                return Err(invalid("--confirm is only valid for quant-parity-run"));
            }
            let plan = ojon_backend::prepare_quant_parity(
                &source,
                &baseline,
                &suite,
                &output_dir,
                options,
            )?;
            if execute {
                if !confirmed {
                    return Err(invalid("quant-parity-run requires --confirm"));
                }
                let report = ojon_backend::run_quant_parity(&plan, |msg| eprintln!("{msg}"))?;
                if json {
                    println!("{report}");
                } else {
                    println!("reference: {}", baseline.display());
                    println!("candidate_results: {}", report["results"]);
                    println!(
                        "provisional_smallest_task_parity_candidate: {}",
                        report["provisional_smallest_measured_candidate"]
                    );
                    println!("proof_of_lossless_quality: NO (limited tests only)");
                }
            } else {
                println!("{}", plan.as_json());
            }
            Ok(true)
        }
        "adaptive-plan" | "adaptive-run" => {
            let execute = cmd == "adaptive-run";
            let source = PathBuf::from(next_argument(&mut args, "original full precision GGUF")?);
            let scores = PathBuf::from(next_argument(&mut args, "measured sensitivity JSON")?);
            let output = PathBuf::from(next_argument(&mut args, "existing output directory")?);
            let suite = PathBuf::from(next_argument(&mut args, "held-out suite JSON")?);
            let mut targets = vec![3.0, 3.5, 4.0, 4.5];
            let mut quantizer = ojon_backend::discovery::resolve_backend_command(
                BackendId::LlamaQuantize,
                "llama-quantize",
            )?;
            let mut eval_tool =
                ojon_backend::discovery::resolve_backend_command(BackendId::LlamaCli, "llama-cli")?;
            let mut imatrix: Option<PathBuf> = None;
            let mut threads = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4);
            let mut gpu_layers = 0i32;
            let mut device: Option<String> = None;
            let mut tokens = 128u32;
            let mut max_loss = 1.0f64;
            let mut server_tool = ojon_backend::discovered_path(BackendId::LlamaServer);
            let mut confirm = false;
            let mut json = false;
            while let Some(option) = args.next() {
                match option.to_str() {
                    Some("--targets") => {
                        let value = next_argument(&mut args, "comma-separated BPW targets")?;
                        let value = value
                            .to_str()
                            .ok_or_else(|| invalid("invalid BPW targets"))?;
                        targets = value
                            .split(',')
                            .map(|x| x.parse::<f64>().map_err(|_| invalid("invalid BPW target")))
                            .collect::<io::Result<Vec<_>>>()?;
                    }
                    Some("--quantizer") => {
                        quantizer = PathBuf::from(next_argument(&mut args, "llama-quantize")?)
                    }
                    Some("--eval-tool") => {
                        eval_tool = PathBuf::from(next_argument(&mut args, "llama-cli")?)
                    }
                    Some("--server-tool") => {
                        server_tool = Some(PathBuf::from(next_argument(&mut args, "llama-server")?))
                    }
                    Some("--imatrix") => {
                        imatrix = Some(PathBuf::from(next_argument(&mut args, "imatrix")?))
                    }
                    Some("--threads") => {
                        threads = next_argument(&mut args, "threads")?
                            .to_str()
                            .ok_or_else(|| invalid("invalid threads"))?
                            .parse()
                            .map_err(|_| invalid("invalid threads"))?
                    }
                    Some("--gpu-layers") => {
                        gpu_layers = next_argument(&mut args, "gpu layers")?
                            .to_str()
                            .ok_or_else(|| invalid("invalid layers"))?
                            .parse()
                            .map_err(|_| invalid("invalid layers"))?
                    }
                    Some("--device") => {
                        device = Some(
                            next_argument(&mut args, "device")?
                                .to_str()
                                .ok_or_else(|| invalid("invalid device"))?
                                .to_owned(),
                        )
                    }
                    Some("--tokens") => {
                        tokens = next_argument(&mut args, "tokens")?
                            .to_str()
                            .ok_or_else(|| invalid("invalid token limit"))?
                            .parse()
                            .map_err(|_| invalid("invalid token limit"))?
                    }
                    Some("--max-loss") => {
                        max_loss = next_argument(&mut args, "absolute accuracy points")?
                            .to_str()
                            .ok_or_else(|| invalid("invalid quality limit"))?
                            .parse()
                            .map_err(|_| invalid("invalid quality limit"))?
                    }
                    Some("--confirm") if execute => confirm = true,
                    Some("--json") => json = true,
                    _ => {
                        return Err(invalid(format!(
                            "unknown adaptive option: {}",
                            option.to_string_lossy()
                        )))
                    }
                }
            }
            let mut plan = prepare_adaptive(
                AdaptiveRequest {
                    source: &source,
                    scores: &scores,
                    output_dir: &output,
                    suite: &suite,
                    quantizer: &quantizer,
                    eval_tool: &eval_tool,
                    imatrix: imatrix.as_deref(),
                    targets: &targets,
                    threads,
                    gpu_layers,
                    device: device.as_deref(),
                    tokens,
                    max_absolute_loss_pct: max_loss,
                },
            )?;
            plan.server_tool = server_tool;
            if execute {
                if !confirm {
                    return Err(invalid(
                        "adaptive search launches quantizers and model inference; pass --confirm",
                    ));
                }
                let results = run_adaptive(&plan, |line| eprintln!("{line}"))?;
                if json {
                    println!("{results}");
                } else {
                    println!(
                        "provisional_smallest_task_passing_output: {}",
                        results["provisional_smallest_task_passing_output"]
                    );
                }
            } else {
                println!("{}", plan.as_json());
            }
            Ok(true)
        }
        "task-suite-check" => {
            let path = PathBuf::from(next_argument(&mut args, "held-out task suite JSON")?);
            let mut json = false;
            for option in args.by_ref() {
                match option.to_str() {
                    Some("--json") if !json => json = true,
                    _ => {
                        return Err(invalid(
                            "task-suite-check accepts only a suite file and optional --json",
                        ))
                    }
                }
            }
            let inspection = inspect_task_suite(&path)?;
            if json {
                println!("{inspection}");
            } else {
                println!("suite: {}", inspection["id"].as_str().unwrap_or("unknown"));
                println!(
                    "status: {}",
                    inspection["status"].as_str().unwrap_or("unknown")
                );
                println!("tasks: {}", inspection["task_count"]);
                println!("coding: {}", inspection["categories"]["coding"]);
                println!("reasoning: {}", inspection["categories"]["reasoning"]);
                println!("tool_call: {}", inspection["categories"]["tool_call"]);
                println!("quality_preservation_verified: false");
                println!("warning: {}", inspection["warning"].as_str().unwrap_or(""));
            }
            Ok(true)
        }
        "task-eval-plan" | "task-eval-run" => {
            let execute = cmd == "task-eval-run";
            let suite = PathBuf::from(next_argument(&mut args, "held-out task suite JSON")?);
            let output = PathBuf::from(next_argument(&mut args, "new quality evidence JSON")?);
            let mut baseline: Option<PathBuf> = None;
            let mut candidates: Vec<(String, PathBuf)> = Vec::new();
            let mut tool =
                ojon_backend::discovery::resolve_backend_command(BackendId::LlamaCli, "llama-cli")?;
            let mut gpu_layers = 0i32;
            let mut device: Option<String> = None;
            let mut tokens = 128u32;
            let mut server_tool = ojon_backend::discovered_path(BackendId::LlamaServer);
            let mut confirm = false;
            let mut json = false;
            while let Some(option) = args.next() {
                match option.to_str() {
                    Some("--baseline") => {
                        baseline = Some(PathBuf::from(next_argument(&mut args, "baseline GGUF")?))
                    }
                    Some("--candidate") => {
                        let pair = next_argument(&mut args, "label=model.gguf")?;
                        let pair = pair
                            .to_str()
                            .ok_or_else(|| invalid("candidate must be UTF-8"))?;
                        let (name, path) = pair
                            .split_once('=')
                            .ok_or_else(|| invalid("candidate must be label=path"))?;
                        candidates.push((name.to_owned(), PathBuf::from(path)));
                    }
                    Some("--tool") => {
                        tool = PathBuf::from(next_argument(&mut args, "llama-cli executable")?)
                    }
                    Some("--server-tool") => {
                        server_tool = Some(PathBuf::from(next_argument(
                            &mut args,
                            "llama-server executable",
                        )?))
                    }
                    Some("--gpu-layers") => {
                        gpu_layers = next_argument(&mut args, "GPU layer count")?
                            .to_str()
                            .ok_or_else(|| invalid("invalid layers"))?
                            .parse()
                            .map_err(|_| invalid("invalid layers"))?
                    }
                    Some("--device") => {
                        device = Some(
                            next_argument(&mut args, "Vulkan2 or SYCL0")?
                                .to_str()
                                .ok_or_else(|| invalid("invalid device"))?
                                .to_owned(),
                        )
                    }
                    Some("--tokens") => {
                        tokens = next_argument(&mut args, "token limit")?
                            .to_str()
                            .ok_or_else(|| invalid("invalid tokens"))?
                            .parse()
                            .map_err(|_| invalid("invalid tokens"))?
                    }
                    Some("--confirm") if execute => confirm = true,
                    Some("--json") => json = true,
                    _ => {
                        return Err(invalid(format!(
                            "unknown task evaluation option: {}",
                            option.to_string_lossy()
                        )))
                    }
                }
            }
            let baseline = baseline.ok_or_else(|| invalid("--baseline is required"))?;
            let plan = ojon_backend::task_eval::prepare_eval_with_server(
                ojon_backend::task_eval::EvalRequest {
                    suite: &suite,
                    baseline: &baseline,
                    candidates: &candidates,
                    tool: &tool,
                    server_tool: server_tool.as_deref(),
                    gpu_layers,
                    device: device.as_deref(),
                    tokens,
                    report: &output,
                },
            )?;
            if execute {
                if !confirm {
                    return Err(invalid("task evaluation loads real models; pass --confirm"));
                }
                let result = run_eval(&plan, |line| eprintln!("{line}"))?;
                if json {
                    println!("{result}");
                } else {
                    println!("task_evaluation_receipt: {}", plan.report.display());
                }
            } else {
                println!("{}", plan.as_json());
            }
            Ok(true)
        }
        "recommend" => {
            let file = PathBuf::from(next_argument(&mut args, "quality evidence JSON file")?);
            let json = match args.next() {
                None => false,
                Some(flag) if flag == "--json" => true,
                _ => return Err(invalid("recommend supports only optional --json")),
            };
            if args.next().is_some() {
                return Err(invalid("too many recommend options"));
            }
            let report = parse_quality_report(&file)?;
            let result = recommend(&report);
            if json {
                println!("{result}");
            } else {
                println!("evaluation_suite: {}", report.evaluation_suite);
                if let Some(best) = result["recommendation"].as_str() {
                    println!("recommended_smallest_quality_preserving_variant: {best}");
                } else {
                    println!("recommendation: NONE; no measured candidate meets all requirements");
                }
                println!("evidence: USER_SUPPLIED_NOT_INDEPENDENTLY_VERIFIED");
                for candidate in result["candidates"].as_array().into_iter().flatten() {
                    println!(
                        "{}: {} bytes, eligible={}",
                        candidate["label"].as_str().unwrap_or("?"),
                        candidate["bytes"],
                        candidate["eligible"]
                    );
                }
            }
            Ok(true)
        }
        "resource-plan" => {
            let source = PathBuf::from(next_argument(&mut args, "source model path")?);
            let output_parent = PathBuf::from(next_argument(
                &mut args,
                "existing destination parent directory",
            )?);
            let mut mode = ResourceMode::Gguf;
            let mut device = String::from("cpu");
            let mut json = false;
            while let Some(arg) = args.next() {
                match arg.to_str() {
                    Some("--mode") => {
                        let value = next_argument(&mut args, "preflight mode")?;
                        mode = ResourceMode::parse(
                            value.to_str().ok_or_else(|| invalid("mode must be text"))?,
                        )?;
                    }
                    Some("--device") => {
                        device = next_argument(&mut args, "cpu|cuda|xpu|auto")?
                            .to_str()
                            .ok_or_else(|| invalid("device must be text"))?
                            .to_owned()
                    }
                    Some("--json") => json = true,
                    _ => {
                        return Err(invalid(format!(
                            "unknown resource-plan option: {}",
                            arg.to_string_lossy()
                        )))
                    }
                }
            }
            let result = resource_plan(&source, &output_parent, mode, &device)?;
            if json {
                println!("{result}");
            } else {
                println!("method: {}", result["mode"]);
                println!("source_bytes: {}", result["source_weight_bytes"]);
                println!("host_free_ram_bytes: {}", result["free_host_ram_bytes"]);
                println!("scratch_free_bytes: {}", result["free_output_volume_bytes"]);
                println!(
                    "host_safety_budget_bytes: {}",
                    result["host_ram_safety_budget_bytes"]
                );
                println!(
                    "disk_safety_budget_bytes: {}",
                    result["output_volume_safety_budget_bytes"]
                );
                println!("status: {}", result["status"]);
                println!("note: budget is a conservative policy floor, not converter peak-memory prediction");
                println!("note: Intel GPU free VRAM is measured only when installed xpu-smi returns a valid sample; otherwise UNKNOWN");
            }
            Ok(true)
        }
        "exl3-plan" | "exl3-run" => {
            let execute = cmd == "exl3-run";
            let source = PathBuf::from(next_argument(
                &mut args,
                "local unquantized HF model directory",
            )?);
            let output = PathBuf::from(next_argument(&mut args, "new EXL3 output directory")?);
            let work_dir =
                PathBuf::from(next_argument(&mut args, "dedicated EXL3 work directory")?);
            let mut checkout: Option<PathBuf> = None;
            let mut python = PathBuf::from("python");
            let mut bits = 3.5f64;
            let mut high_quality = false;
            let mut resume = false;
            let mut confirm = false;
            let mut allow_resource_risk = false;
            let mut json = false;
            while let Some(arg) = args.next() {
                match arg.to_str() {
                    Some("--checkout") => {
                        checkout = Some(PathBuf::from(next_argument(&mut args, "EXL3 checkout")?))
                    }
                    Some("--python") => {
                        python = PathBuf::from(next_argument(&mut args, "Python interpreter")?)
                    }
                    Some("--bits") => {
                        bits = next_argument(&mut args, "target bitrate")?
                            .to_str()
                            .ok_or_else(|| invalid("bitrate must be text"))?
                            .parse()
                            .map_err(|_| invalid("invalid EXL3 bitrate"))?
                    }
                    Some("--hq") => high_quality = true,
                    Some("--resume") => resume = true,
                    Some("--confirm") if execute => confirm = true,
                    Some("--allow-resource-risk") if execute => allow_resource_risk = true,
                    Some("--json") => json = true,
                    _ => {
                        return Err(invalid(format!(
                            "unknown EXL3 option: {}",
                            arg.to_string_lossy()
                        )))
                    }
                }
            }
            let checkout = checkout.ok_or_else(|| invalid("EXL3 --checkout is required"))?;
            let plan = prepare_exl3(
                Exl3Request {
                    source: &source,
                    output: &output,
                    work_dir: &work_dir,
                    checkout: &checkout,
                    python: &python,
                    bits,
                    high_quality,
                    resume,
                },
            )?;
            if !execute {
                if json {
                    println!("{}", plan.as_json());
                } else {
                    println!("backend: EXL3 / ExLlamaV3 (CUDA, not GGUF)");
                    println!("plan: {}", plan.as_json());
                }
            } else {
                if !confirm {
                    return Err(invalid("EXL3 conversion writes model data; pass --confirm"));
                }
                println!("{}", plan.as_json());
                let result = run_exl3(&plan, allow_resource_risk)?;
                println!("{}", result);
            }
            Ok(true)
        }
        "research-plan" | "research-run" => {
            let execute = cmd == "research-run";
            let kind_str = next_argument(&mut args, "hqq or openvino backend")?;
            let kind = ResearchKind::parse(
                kind_str
                    .to_str()
                    .ok_or_else(|| invalid("backend must be text"))?,
            )?;
            let source = PathBuf::from(next_argument(
                &mut args,
                "local original HF checkpoint directory",
            )?);
            let output = PathBuf::from(next_argument(&mut args, "new absolute output directory")?);
            let mut bits = 4u32;
            let mut group = if kind == ResearchKind::Hqq {
                64u32
            } else {
                128u32
            };
            let mut ratio = 1.0f64;
            let mut device = String::from("cpu");
            let mut tool = PathBuf::from(if kind == ResearchKind::Hqq {
                "python"
            } else {
                "optimum-cli"
            });
            let mut confirm = false;
            let mut allow_resource_risk = false;
            let mut json = false;
            while let Some(arg) = args.next() {
                match arg.to_str() {
                    Some("--bits") => {
                        bits = next_argument(&mut args, "bit width")?
                            .to_str()
                            .ok_or_else(|| invalid("bits must be text"))?
                            .parse()
                            .map_err(|_| invalid("invalid bits"))?
                    }
                    Some("--group") => {
                        group = next_argument(&mut args, "group size")?
                            .to_str()
                            .ok_or_else(|| invalid("group must be text"))?
                            .parse()
                            .map_err(|_| invalid("invalid group size"))?
                    }
                    Some("--ratio") => {
                        ratio = next_argument(&mut args, "OpenVINO 4-bit layer ratio")?
                            .to_str()
                            .ok_or_else(|| invalid("ratio must be text"))?
                            .parse()
                            .map_err(|_| invalid("invalid ratio"))?
                    }
                    Some("--device") => {
                        device = next_argument(&mut args, "device")?
                            .to_str()
                            .ok_or_else(|| invalid("device must be text"))?
                            .to_owned()
                    }
                    Some("--tool") => {
                        tool = PathBuf::from(next_argument(&mut args, "backend executable")?)
                    }
                    Some("--json") => json = true,
                    Some("--confirm") if execute => confirm = true,
                    Some("--allow-resource-risk") if execute => allow_resource_risk = true,
                    _ => {
                        return Err(invalid(format!(
                            "unknown research option: {}",
                            arg.to_string_lossy()
                        )))
                    }
                }
            }
            let plan =
                prepare_research(
                    ResearchRequest {
                        kind,
                        source: &source,
                        output: &output,
                        executable: &tool,
                        bits,
                        group,
                        ratio,
                        device: &device,
                    },
                )?;
            if !execute {
                if json {
                    println!("{}", plan.as_json());
                } else {
                    println!("backend: {}", plan.as_json()["backend"]);
                    println!("output_format: {}", plan.as_json()["output_format"]);
                    println!("source: {}", plan.source.directory.display());
                    println!("output: {}", plan.output.display());
                    println!("status: PLAN_ONLY; no model weights loaded");
                    println!("run: repeat the same options using research-run and --confirm");
                }
            } else {
                if !confirm {
                    return Err(invalid("research conversion writes output; pass --confirm"));
                }
                println!("{}", plan.as_json());
                ojon_backend::research_routes::run_research_with_policy(
                    &plan,
                    allow_resource_risk,
                )?;
                println!("research_status: EXPORTED_FORMAT_ONLY");
                println!("runtime_load_test: NOT_RUN");
                println!("held_out_task_quality: NOT_TESTED");
            }
            Ok(true)
        }
        "autoround-plan" | "autoround-run" => {
            let execute = cmd == "autoround-run";
            let model_dir = PathBuf::from(next_argument(&mut args, "local HF model directory")?);
            let output_dir = PathBuf::from(next_argument(&mut args, "new output directory")?);
            let mut bits = 4u32;
            let mut format: Option<String> = None;
            let mut iters = 0u32;
            let mut dataset: Option<PathBuf> = None;
            let mut device = String::from("xpu");
            let mut python = PathBuf::from("python");
            let mut nsamples = 128u32;
            let mut seqlen = 2048u32;
            let mut json = false;
            let mut confirm = false;
            let mut allow_resource_risk = false;
            while let Some(arg) = args.next() {
                match arg.to_str() {
                    Some("--bits") => {
                        bits = next_argument(&mut args, "bits (2, 3, 4, 5, 6, 8)")?
                            .to_str()
                            .ok_or_else(|| invalid("bits require integer"))?
                            .parse()
                            .map_err(|_| invalid("invalid bit width"))?;
                    }
                    Some("--format") => {
                        format = Some(
                            next_argument(&mut args, "GGUF export format")?
                                .to_str()
                                .ok_or_else(|| invalid("format requires UTF-8"))?
                                .to_owned(),
                        );
                    }
                    Some("--iters") => {
                        iters = next_argument(&mut args, "AutoRound iterations")?
                            .to_str()
                            .ok_or_else(|| invalid("iterations require integer"))?
                            .parse()
                            .map_err(|_| invalid("invalid iterations"))?;
                    }
                    Some("--dataset") => {
                        dataset = Some(PathBuf::from(next_argument(
                            &mut args,
                            "local calibration JSON",
                        )?));
                    }
                    Some("--device") => {
                        device = next_argument(&mut args, "device (xpu, cuda, cpu)")?
                            .to_string_lossy()
                            .to_string();
                    }
                    Some("--python") => {
                        python = PathBuf::from(next_argument(&mut args, "Python executable")?);
                    }
                    Some("--nsamples") => {
                        nsamples = next_argument(&mut args, "calibration samples")?
                            .to_str()
                            .ok_or_else(|| invalid("nsamples require integer"))?
                            .parse()
                            .map_err(|_| invalid("invalid nsamples"))?;
                    }
                    Some("--seqlen") => {
                        seqlen = next_argument(&mut args, "sequence length")?
                            .to_str()
                            .ok_or_else(|| invalid("seqlen requires integer"))?
                            .parse()
                            .map_err(|_| invalid("invalid seqlen"))?;
                    }
                    Some("--json") => json = true,
                    Some("--confirm") if execute => confirm = true,
                    Some("--allow-resource-risk") if execute => allow_resource_risk = true,
                    _ => {
                        return Err(invalid(format!(
                            "unknown autoround option: {}",
                            arg.to_string_lossy()
                        )))
                    }
                }
            }
            let plan = ojon_backend::autoround::prepare_autoround_recipe_plan(
                ojon_backend::autoround::AutoRoundRecipeRequest {
                    model_dir: &model_dir,
                    output_dir: &output_dir,
                    bits,
                    device_str: &device,
                    python: &python,
                    nsamples,
                    seqlen,
                    requested_format: format.as_deref(),
                    iters,
                    dataset: dataset.as_deref(),
                },
            )?;
            if !execute {
                if json {
                    println!("{}", plan.json());
                } else {
                    println!("backend: intel_autoround");
                    println!("model_directory: {}", plan.model_dir.display());
                    println!("output_directory: {}", plan.output_dir.display());
                    println!("bits: {}", plan.bits);
                    println!("device: {}", plan.device.as_str());
                    println!("format: {}", plan.format);
                    println!(
                        "method: {}",
                        if plan.iters == 0 {
                            "optimized_rtn"
                        } else {
                            "calibrated_auto_round"
                        }
                    );
                    println!("iterations: {}", plan.iters);
                    println!(
                        "dataset: {}",
                        plan.dataset
                            .as_ref()
                            .map(|p| p.display().to_string())
                            .unwrap_or_else(|| "none (offline RTN)".into())
                    );
                    println!("python: {}", plan.python.display());
                    println!("nsamples: {}", plan.nsamples);
                    println!("seqlen: {}", plan.seqlen);
                    println!("status: PLAN_ONLY");
                }
            } else {
                if !confirm {
                    return Err(invalid(
                        "autoround execution writes a quantized model; pass --confirm",
                    ));
                }
                println!("{}", plan.json());
                ojon_backend::autoround::run_autoround_with_policy(&plan, allow_resource_risk)?;
                println!("autoround_status: COMPLETED");
            }
            Ok(true)
        }
        "optimize-workflow-plan" | "optimize-workflow-run" => {
            let execute = cmd == "optimize-workflow-run";
            let source = PathBuf::from(next_argument(&mut args, "model file or directory")?);
            let mut input = ojon_backend::OptimizeWorkflowRequest::new(source);
            // Saved defaults apply only when a CLI flag doesn't explicitly
            // override the field. Loading does not mutate settings.
            let prefs = ojon_backend::load_settings()?;
            input.goal = ojon_backend::OptimizeGoal::parse(&prefs.optimization_goal)?;
            input.min_saving_percent = prefs.min_saving_percent;
            input.max_loss_points = prefs.max_loss_points;
            input.max_ppl_regression_pct = prefs.max_ppl_regression_pct;
            input.precision_candidates = prefs
                .precision_candidates
                .as_deref()
                .map(ojon_backend::parse_precisions)
                .transpose()?;
            input.bench_prompt_tokens = prefs.bench_prompt_tokens;
            input.bench_generation_tokens = prefs.bench_generation_tokens;
            input.bench_repetitions = prefs.bench_repetitions;
            input.balanced_min_speed_ratio = prefs.balanced_min_speed_ratio;
            input.device = prefs.preferred_device.clone();
            if matches!(input.device.as_str(), "cuda" | "xpu" | "vulkan") {
                input.runtime_device = prefs.runtime_device.clone();
                input.gpu_layers = prefs.gpu_layers as i32;
            }
            input.server_tool = ojon_backend::discovered_path(BackendId::LlamaServer);
            let mut json = false;
            let mut confirmed = false;
            let mut auto_hardware = false;
            while let Some(option) = args.next() {
                match option.to_str() {
                    Some("--baseline") => {
                        input.current_baseline = Some(PathBuf::from(next_argument(
                            &mut args,
                            "optional existing GGUF quality reference",
                        )?))
                    }
                    Some("--candidate") => {
                        let arg = next_argument(&mut args, "candidate label=path.gguf")?;
                        let txt = arg
                            .to_str()
                            .ok_or_else(|| invalid("candidate must be UTF-8"))?;
                        let (name, path) = txt
                            .split_once('=')
                            .ok_or_else(|| invalid("--candidate requires label=path.gguf"))?;
                        if name.is_empty() || path.is_empty() {
                            return Err(invalid("empty candidate label/path"));
                        }
                        input
                            .candidates
                            .push((name.to_owned(), PathBuf::from(path)));
                    }
                    Some("--server-tool") => {
                        input.server_tool = Some(PathBuf::from(next_argument(
                            &mut args,
                            "local llama-server executable",
                        )?))
                    }
                    Some("--suite") => {
                        input.suite = Some(PathBuf::from(next_argument(
                            &mut args,
                            "held-out task suite JSON",
                        )?))
                    }
                    Some("--output-dir") => {
                        input.output_dir = Some(PathBuf::from(next_argument(
                            &mut args,
                            "existing output directory",
                        )?))
                    }
                    Some("--tensor-sensitivity") => {
                        input.tensor_sensitivity = Some(PathBuf::from(next_argument(
                            &mut args,
                            "tensor sensitivity JSON",
                        )?))
                    }
                    Some("--imatrix") => {
                        input.imatrix = Some(PathBuf::from(next_argument(
                            &mut args,
                            "optional calibration imatrix",
                        )?))
                    }
                    Some("--ppl-corpus") => {
                        input.ppl_corpus = Some(PathBuf::from(next_argument(
                            &mut args,
                            "held-out perplexity corpus",
                        )?))
                    }
                    Some("--max-ppl-regression-pct") => {
                        input.max_ppl_regression_pct =
                            next_argument(&mut args, "maximum perplexity regression")?
                                .to_string_lossy()
                                .parse()
                                .map_err(|_| invalid("PPL regression must be numeric"))?;
                    }
                    Some("--precisions") => {
                        let spec = next_argument(&mut args, "comma separated guided GGUF formats")?;
                        input.precision_candidates = Some(ojon_backend::parse_precisions(
                            spec.to_str()
                                .ok_or_else(|| invalid("precision list must be UTF-8"))?,
                        )?);
                    }
                    Some("--bench-prompt-tokens") => {
                        input.bench_prompt_tokens =
                            next_argument(&mut args, "benchmark prefill tokens")?
                                .to_string_lossy()
                                .parse()
                                .map_err(|_| invalid("invalid benchmark prompt tokens"))?;
                    }
                    Some("--bench-gen-tokens") => {
                        input.bench_generation_tokens =
                            next_argument(&mut args, "benchmark decode tokens")?
                                .to_string_lossy()
                                .parse()
                                .map_err(|_| invalid("invalid benchmark decode tokens"))?;
                    }
                    Some("--bench-repetitions") => {
                        input.bench_repetitions =
                            next_argument(&mut args, "benchmark repetitions")?
                                .to_string_lossy()
                                .parse()
                                .map_err(|_| invalid("invalid benchmark repetitions"))?;
                    }
                    Some("--balanced-speed-ratio") => {
                        input.balanced_min_speed_ratio =
                            next_argument(&mut args, "Balanced speed ratio")?
                                .to_string_lossy()
                                .parse()
                                .map_err(|_| invalid("invalid Balanced speed ratio"))?;
                    }
                    Some("--goal") => {
                        let name = next_argument(&mut args, "goal")?;
                        input.goal = ojon_backend::OptimizeGoal::parse(
                            name.to_str().ok_or_else(|| invalid("goal must be UTF-8"))?,
                        )?;
                    }
                    Some("--device") => {
                        input.device = next_argument(&mut args, "device")?
                            .to_str()
                            .ok_or_else(|| invalid("device must be UTF-8"))?
                            .to_owned();
                        if matches!(input.device.as_str(), "auto" | "cpu") {
                            input.runtime_device = None;
                            input.gpu_layers = 0;
                        } else if input.device != prefs.preferred_device {
                            // Do not apply the saved device ID from a different vendor.
                            input.runtime_device = None;
                        }
                    }
                    Some("--auto-hardware") => auto_hardware = true,
                    Some("--runtime-device") => {
                        input.runtime_device = Some(
                            next_argument(
                                &mut args,
                                "llama.cpp device name, e.g. SYCL0, CUDA0 or Vulkan2",
                            )?
                            .to_string_lossy()
                            .into_owned(),
                        )
                    }
                    Some("--gpu-layers") => {
                        input.gpu_layers = next_argument(&mut args, "GPU offload layers")?
                            .to_string_lossy()
                            .parse()
                            .map_err(|_| invalid("GPU layers must be integer"))?
                    }
                    Some("--min-saving-percent") => {
                        input.min_saving_percent =
                            next_argument(&mut args, "minimum size saving percent")?
                                .to_string_lossy()
                                .parse()
                                .map_err(|_| invalid("saving must be numeric"))?;
                    }
                    Some("--max-loss-points") => {
                        input.max_loss_points =
                            next_argument(&mut args, "maximum quality score loss")?
                                .to_string_lossy()
                                .parse()
                                .map_err(|_| invalid("loss must be numeric"))?;
                    }
                    Some("--json") => json = true,
                    Some("--confirm") => confirmed = true,
                    _ => {
                        return Err(invalid(format!(
                            "unknown workflow option: {}",
                            option.to_string_lossy()
                        )))
                    }
                }
            }
            if confirmed && !execute {
                return Err(invalid(
                    "--confirm is only valid with optimize-workflow-run",
                ));
            }
            if execute && !confirmed {
                return Err(invalid("optimize-workflow-run requires --confirm"));
            }
            if auto_hardware
                || (matches!(input.device.as_str(), "vulkan" | "cuda" | "xpu")
                    && input.runtime_device.is_none())
            {
                input = input.with_auto_hardware();
            }
            if input.suite.is_none() {
                input.suite = ojon_backend::discovery::bundled_data_file("sanity-tasks.json");
            }
            if input.output_dir.is_none() {
                input.output_dir = input.model.parent().map(Path::to_path_buf);
            }
            if input.imatrix.is_none() {
                input.auto_imatrix = true;
            }
            let backends = discover_backends();
            let plan = ojon_backend::plan_optimize_workflow(&input, &backends)?;
            if execute {
                if plan["ready_for_user_approved_execution"] != true {
                    return Err(invalid(format!(
                        "optimizer prerequisites not met: {}",
                        plan["checks"]
                    )));
                }
                let result =
                    ojon_backend::run_optimize_workflow(&input, &backends, true, |line| {
                        eprintln!("{line}")
                    })?;
                if json {
                    println!("{result}");
                } else {
                    println!("operation: {}", result["route"]);
                    println!("status: {}", result["status"]);
                    println!("provisional_candidate: {}", result["provisional_candidate"]);
                    println!("global_quality_equivalence: NOT_PROVEN");
                    println!("details: {}", result["evidence"]);
                }
            } else if json {
                println!("{plan}");
            } else {
                println!("model: {}", plan["model"]["source"]);
                println!("goal: {} / device: {}", plan["goal"], plan["target_device"]);
                println!("route: {}", plan["route"]);
                println!(
                    "ready_for_approved_run: {}",
                    plan["ready_for_user_approved_execution"]
                );
                if let Some(step) = plan["next_user_step"].as_str() {
                    println!("next: {step}");
                }
                if let Some(cmd) = plan["recommended_command_argv"].as_array() {
                    println!(
                        "execution argv (display only; never started): {}",
                        serde_json::Value::Array(cmd.clone())
                    );
                }
                for note in plan["checks"].as_array().into_iter().flatten() {
                    if let Some(text) = note.as_str() {
                        println!("check: {text}");
                    }
                }
                println!("quality: NOT_TESTED; no GPU allocated");
            }
            Ok(true)
        }
        "optimize-plan" => {
            let path = PathBuf::from(next_argument(&mut args, "local model file or directory")?);
            let mut device = String::from("auto");
            let mut json = false;
            while let Some(option) = args.next() {
                match option.to_str() {
                    Some("--device") => {
                        device = next_argument(&mut args, "device")?
                            .to_str()
                            .ok_or_else(|| invalid("device must be UTF-8"))?
                            .to_owned();
                    }
                    Some("--json") => json = true,
                    _ => {
                        return Err(invalid(format!(
                            "unknown optimize-plan option: {}",
                            option.to_string_lossy()
                        )))
                    }
                }
            }
            let probe = probe_model(&path)?;
            let plan = ojon_backend::plan_options(&probe, &device, &discover_backends())?;
            if json {
                println!("{}", plan);
            } else {
                println!("source: {}", probe.path.display());
                println!("requested_device: {}", device);
                println!("mode: READ_ONLY / MULTIPLE ALTERNATIVES / NO WINNER");
                println!("actual_free_ram_vram_scratch: NOT_PROBED");
                for item in plan["options"].as_array().into_iter().flatten() {
                    println!(
                        "{} [{}] {} / {}",
                        item["id"].as_str().unwrap_or("?"),
                        item["state"].as_str().unwrap_or("?"),
                        item["backend"].as_str().unwrap_or("?"),
                        item["output_format"].as_str().unwrap_or("?")
                    );
                    println!("  check: {}", item["reason"].as_str().unwrap_or("unknown"));
                }
                println!(
                    "warning: {}",
                    plan["warning"].as_str().unwrap_or("not tested")
                );
            }
            Ok(true)
        }
        "train-plan" | "train-run" => {
            let run = cmd == "train-run";
            let config = PathBuf::from(next_argument(&mut args, "training YAML config")?);
            let mut trainer = discovered_path(BackendId::LlamaFactory)
                .unwrap_or_else(|| PathBuf::from("llamafactory-cli"));
            let mut json = false;
            let mut confirm = false;
            while let Some(option) = args.next() {
                match option.to_str() {
                    Some("--tool") => {
                        trainer = PathBuf::from(next_argument(&mut args, "trainer executable")?)
                    }
                    Some("--json") if !run => json = true,
                    Some("--confirm") if run => confirm = true,
                    _ => {
                        return Err(invalid(format!(
                            "unknown train option: {}",
                            option.to_string_lossy()
                        )))
                    }
                }
            }
            let plan = prepare_train(&config, &trainer)?;
            if !run {
                if json {
                    println!("{}", plan.json());
                } else {
                    println!("trainer: {}", plan.trainer.display());
                    println!("config: {}", plan.config.display());
                    println!("model: {}", plan.model.display());
                    println!(
                        "dataset: {} (from {})",
                        plan.dataset,
                        plan.data_dir.display()
                    );
                    println!("output: {}", plan.output_dir.display());
                    println!(
                        "training: {}",
                        if plan.quantization_bit.is_some() {
                            "QLoRA"
                        } else {
                            "LoRA"
                        }
                    );
                    println!("model_compatibility: NOT_VERIFIED; peak_VRAM: UNKNOWN");
                    println!("status: PLAN_ONLY; no external process launched");
                }
            } else {
                if !confirm {
                    return Err(invalid("training may use substantial RAM/VRAM and write output; explicitly pass --confirm to start"));
                }
                eprintln!("Ojon: user-confirmed external training job; examine YAML/data permissions before executing.");
                run_train(&plan)?;
                println!(
                    "trainer_exit: SUCCESS; output_dir: {}",
                    plan.output_dir.display()
                );
                println!("quality_benchmark: NOT_RUN; distillation: NOT_IMPLEMENTED");
            }
            Ok(true)
        }
        "plan" => {
            let plan = parse_plan(args)?;
            show_plan(&plan);
            println!("status: PLANNED_ONLY; no weights modified");
            Ok(true)
        }
        "quantize" => {
            let plan = parse_plan(args)?;
            show_plan(&plan);
            let result = quantize_gguf(&plan)?;
            println!("status: CONVERTER_EXITED_SUCCESSFULLY");
            println!("output: {}", result.output.display());
            println!("output_bytes: {}", result.output_info.file_bytes);
            println!("verification: GGUF_METADATA_AND_TENSOR_CATALOG_ONLY");
            println!("runtime_load_test: NOT_RUN");
            println!("task_quality_test: NOT_RUN");
            if let Some(path) = result.receipt {
                println!("receipt: {}", path.display());
            } else {
                eprintln!(
                    "warning: output was created, but the optional receipt could not be written"
                );
            }
            Ok(true)
        }
        "corpus-build" => {
            // Calibration corpus from the user's own text: files or one-level directories.
            let output = PathBuf::from(next_argument(&mut args, "new corpus .txt output")?);
            let mut inputs: Vec<PathBuf> = Vec::new();
            let mut max_file_mib = 64u64;
            let mut json = false;
            while let Some(option) = args.next() {
                match option.to_str() {
                    Some("--max-file-mib") => {
                        max_file_mib = next_argument(&mut args, "max file size MiB")?
                            .to_str()
                            .and_then(|v| v.parse().ok())
                            .ok_or_else(|| invalid("max-file-mib must be a number"))?
                    }
                    Some("--json") => json = true,
                    Some(other) if other.starts_with("--") => {
                        return Err(invalid(format!("unknown corpus-build option: {other}")))
                    }
                    _ => inputs.push(PathBuf::from(option)),
                }
            }
            if inputs.is_empty() {
                return Err(invalid("corpus-build needs at least one input file or directory"));
            }
            let report = ojon_backend::corpus::build_corpus(&inputs, &output, max_file_mib * 1024 * 1024)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report.as_json())?);
            } else {
                println!("corpus: {}", output.display());
                println!(
                    "{} inputs, {} skipped, {} paragraphs kept ({} duplicates dropped), {} bytes (~{} tokens){}",
                    report.inputs.len(),
                    report.skipped.len(),
                    report.paragraphs_kept,
                    report.duplicates_dropped,
                    report.bytes_out,
                    report.approx_tokens,
                    if report.bytes_out >= ojon_backend::corpus::RECOMMENDED_BYTES { "" } else { " - under the 1 MiB recommended for calibration" }
                );
            }
            Ok(true)
        }
        "recipe-apply" => {
            // Reuse a measured recipe on another original of the same family, optionally verifying KLD.
            let source = PathBuf::from(next_argument(&mut args, "original F16/BF16 GGUF model")?);
            let output_dir = PathBuf::from(next_argument(&mut args, "output directory")?);
            let receipt = PathBuf::from(next_argument(&mut args, "recipe-receipt.json or recipe JSON")?);
            let mut imatrix: Option<PathBuf> = None;
            let mut quantize_tool = ojon_backend::discovery::resolve_backend_command(
                BackendId::LlamaQuantize,
                "llama-quantize",
            )?;
            let mut perplexity_tool = ojon_backend::discovery::resolve_backend_command(
                BackendId::LlamaPerplexity,
                "llama-perplexity",
            )?;
            let mut corpus: Option<PathBuf> = None;
            let mut verify_budget: Option<f64> = None;
            let (mut context, mut gpu_layers, mut chunks) = (512u32, 99u32, 40u32);
            let mut device: Option<String> = None;
            let mut fit = false;
            let mut threads = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(4);
            let mut confirm = false;
            let text = |v: std::ffi::OsString, what: &str| -> io::Result<String> {
                v.to_str().map(str::to_owned).ok_or_else(|| invalid(format!("{what} requires UTF-8")))
            };
            while let Some(option) = args.next() {
                match option.to_str() {
                    Some("--imatrix") => imatrix = Some(PathBuf::from(next_argument(&mut args, "importance matrix file")?)),
                    Some("--quantize-tool") => quantize_tool = PathBuf::from(next_argument(&mut args, "llama-quantize executable")?),
                    Some("--perplexity-tool") => perplexity_tool = PathBuf::from(next_argument(&mut args, "llama-perplexity executable")?),
                    Some("--corpus") => corpus = Some(PathBuf::from(next_argument(&mut args, "held-out text corpus")?)),
                    Some("--verify-budget") => {
                        verify_budget = Some(text(next_argument(&mut args, "absolute mean KLD budget")?, "verify-budget")?
                            .parse::<f64>()
                            .map_err(|_| invalid("verify-budget must be a number"))?)
                    }
                    Some("--ctx") => context = text(next_argument(&mut args, "context")?, "ctx")?.parse().map_err(|_| invalid("ctx must be a number"))?,
                    Some("--ngl") => {
                        let v = text(next_argument(&mut args, "gpu layers or 'auto'")?, "ngl")?;
                        if v == "auto" { fit = true; } else { gpu_layers = v.parse().map_err(|_| invalid("ngl must be a number or 'auto'"))?; }
                    }
                    Some("--device") => device = Some(text(next_argument(&mut args, "backend device list, e.g. Vulkan0")?, "device")?),
                    Some("--chunks") => chunks = text(next_argument(&mut args, "chunk limit")?, "chunks")?.parse().map_err(|_| invalid("chunks must be a number"))?,
                    Some("--threads") => threads = text(next_argument(&mut args, "thread count")?, "threads")?.parse().map_err(|_| invalid("threads must be a number"))?,
                    Some("--confirm") => confirm = true,
                    _ => return Err(invalid(format!("unknown recipe-apply option: {}", option.to_string_lossy()))),
                }
            }
            if !confirm {
                return Err(invalid("recipe-apply writes a quantized model; add --confirm to execute"));
            }
            let settings = match (&corpus, verify_budget) {
                (Some(c), Some(_)) => Some(ojon_backend::kld::KldSettings { tool: perplexity_tool, corpus: c.clone(), gpu_layers, fit, context, chunks, device: device.clone() }),
                (None, Some(_)) => return Err(invalid("--verify-budget needs --corpus")),
                _ => None,
            };
            let verify = settings.as_ref().map(|s| (s, verify_budget.unwrap_or(0.0)));
            let receipt_out = ojon_backend::recipe_run::apply_recipe(
                &source, &output_dir, &receipt, &quantize_tool, imatrix.as_deref(), threads, verify, |line| eprintln!("{line}"),
            )?;
            println!("receipt: {}", receipt_out.display());
            Ok(true)
        }
        "embed-parity-plan" | "embed-parity-run" => {
            // Embedding / reranker parity: both models embed the same held-out
            // lines through llama-embedding; cosine, neighbour overlap and
            // ranking agreement are measured and gated. Plan writes nothing.
            let execute = cmd == "embed-parity-run";
            let reference = PathBuf::from(next_argument(&mut args, "reference GGUF (original or accepted quant)")?);
            let candidate = PathBuf::from(next_argument(&mut args, "candidate GGUF")?);
            let output_dir = PathBuf::from(next_argument(&mut args, "empty output directory")?);
            let mut settings = ojon_backend::embed_parity::EmbedSettings {
                tool: ojon_backend::discovery::resolve_backend_command(BackendId::LlamaEmbedding, "llama-embedding")?,
                ..Default::default()
            };
            let mut gate = ojon_backend::embed_parity::EmbedGate::default();
            let (mut confirm, mut json) = (false, false);
            let text = |v: std::ffi::OsString, what: &str| -> io::Result<String> {
                v.to_str().map(str::to_owned).ok_or_else(|| invalid(format!("{what} requires UTF-8")))
            };
            let number = |v: std::ffi::OsString, what: &str| -> io::Result<f64> {
                text(v, what)?.parse::<f64>().map_err(|_| invalid(format!("{what} must be a number")))
            };
            while let Some(option) = args.next() {
                match option.to_str() {
                    Some("--texts") => settings.texts = Some(PathBuf::from(next_argument(&mut args, "held-out texts, one per line")?)),
                    Some("--queries") => settings.queries = Some(PathBuf::from(next_argument(&mut args, "queries, one per line")?)),
                    Some("--documents") => settings.documents = Some(PathBuf::from(next_argument(&mut args, "documents, one per line")?)),
                    Some("--embedding-tool") => settings.tool = PathBuf::from(next_argument(&mut args, "llama-embedding executable")?),
                    Some("--pooling") => {
                        let v = text(next_argument(&mut args, "pooling: mean, last, cls or rank")?, "pooling")?;
                        settings.pooling = ojon_backend::embed_parity::Pooling::parse(&v)
                            .ok_or_else(|| invalid(format!("unknown pooling '{v}' (mean, last, cls, rank)")))?;
                    }
                    Some("--no-normalise") | Some("--no-normalize") => settings.normalise = false,
                    Some("--ctx") => settings.context = text(next_argument(&mut args, "context")?, "ctx")?.parse().map_err(|_| invalid("ctx must be a number"))?,
                    Some("--batch") => settings.batch = text(next_argument(&mut args, "batch size")?, "batch")?.parse().map_err(|_| invalid("batch must be a number"))?,
                    Some("--ngl") => {
                        let v = text(next_argument(&mut args, "gpu layers or 'auto'")?, "ngl")?;
                        if v == "auto" { settings.fit = true; } else { settings.gpu_layers = v.parse().map_err(|_| invalid("ngl must be a number or 'auto'"))?; }
                    }
                    Some("--top-k") => settings.top_k = text(next_argument(&mut args, "neighbour depth k")?, "top-k")?.parse().map_err(|_| invalid("top-k must be a number"))?,
                    Some("--min-cosine") => gate.min_cosine = number(next_argument(&mut args, "minimum per-text cosine")?, "min-cosine")?,
                    Some("--min-recall") => gate.min_recall = Some(number(next_argument(&mut args, "minimum Recall@k")?, "min-recall")?),
                    Some("--min-ndcg") => gate.min_ndcg = Some(number(next_argument(&mut args, "minimum NDCG@10")?, "min-ndcg")?),
                    Some("--json") => json = true,
                    Some("--confirm") if execute => confirm = true,
                    _ => return Err(invalid(format!("unknown embed-parity option: {}", option.to_string_lossy()))),
                }
            }
            let plan = ojon_backend::embed_parity::prepare_embed_parity(ojon_backend::embed_parity::EmbedParityJob {
                reference,
                candidate,
                output_dir,
                settings,
                gate,
            })?;
            if json {
                println!("{}", serde_json::to_string_pretty(&plan.plan_json())?);
            } else {
                println!("reference: {}", plan.job.reference.display());
                println!("candidate: {}", plan.job.candidate.display());
                println!(
                    "pooling {} | texts {} | queries {} | documents {} | k {} | gate min cosine {}",
                    plan.job.settings.pooling.as_str(),
                    plan.inputs.texts.len(),
                    plan.inputs.queries.len(),
                    plan.inputs.documents.len(),
                    plan.job.settings.top_k,
                    plan.job.gate.min_cosine
                );
                for run in &plan.runs {
                    println!("run {} {} ({} lines): {}", run.role, run.set, run.lines, run.command);
                }
            }
            if !execute {
                println!("writes: nothing");
                println!("next_step: ojon embed-parity-run <same arguments> --confirm");
                return Ok(true);
            }
            if !confirm {
                return Err(invalid("embed-parity-run loads both models and writes prompt files, captures and a receipt; add --confirm to execute"));
            }
            let receipt = ojon_backend::embed_parity::run_embed_parity(&plan, |line| eprintln!("{line}"))?;
            if json {
                println!("{}", std::fs::read_to_string(&receipt)?);
            } else {
                let doc: serde_json::Value = serde_json::from_slice(&std::fs::read(&receipt)?)?;
                println!("receipt: {}", receipt.display());
                println!("metrics: {}", doc["metrics"]);
                println!("gate: {}", doc["gate"]);
                println!("claim: {}", doc["claim"]);
            }
            Ok(true)
        }
        "recipe-plan" | "recipe-run" => {
            // KLD-gated mixed-precision recipe search: measure how low each tensor
            // class can go before the output distribution moves past a budget.
            let execute = cmd == "recipe-run";
            let source = PathBuf::from(next_argument(&mut args, "original F16/BF16 GGUF model")?);
            let output_dir = PathBuf::from(next_argument(&mut args, "empty output directory")?);
            let mut corpus: Option<PathBuf> = None;
            let mut eval_corpus: Option<PathBuf> = None;
            let mut base = String::from("q5_k");
            let mut floor = String::from("iq2_s");
            let mut budget = 0.01f64;
            let mut imatrix: Option<PathBuf> = None;
            let mut imatrix_tool: Option<PathBuf> = None;
            let mut budget_p99: Option<f64> = None;
            let mut quantize_tool = ojon_backend::discovery::resolve_backend_command(
                BackendId::LlamaQuantize,
                "llama-quantize",
            )?;
            let mut perplexity_tool = ojon_backend::discovery::resolve_backend_command(
                BackendId::LlamaPerplexity,
                "llama-perplexity",
            )?;
            let (mut context, mut gpu_layers, mut chunks) = (512u32, 99u32, 40u32);
            let mut device: Option<String> = None;
            let mut fit = false;
            let mut threads = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(4);
            let (mut keep_probes, mut confirm, mut json, mut resume) = (false, false, false, false);
            let mut refine_boundaries = true;
            let text = |v: std::ffi::OsString, what: &str| -> io::Result<String> {
                v.to_str().map(str::to_owned).ok_or_else(|| invalid(format!("{what} requires UTF-8")))
            };
            while let Some(option) = args.next() {
                match option.to_str() {
                    Some("--corpus") => corpus = Some(PathBuf::from(next_argument(&mut args, "calibration text corpus")?)),
                    Some("--eval-corpus") => eval_corpus = Some(PathBuf::from(next_argument(&mut args, "held-out evaluation text")?)),
                    Some("--base") => base = text(next_argument(&mut args, "base level")?, "base")?,
                    Some("--floor") => floor = text(next_argument(&mut args, "floor level")?, "floor")?,
                    Some("--budget") => {
                        budget = text(next_argument(&mut args, "KLD budget")?, "budget")?
                            .parse::<f64>()
                            .map_err(|_| invalid("budget must be a number (mean KLD delta over base)"))?
                    }
                    Some("--imatrix") => {
                        // `--imatrix auto` computes one from the corpus first; anything else is an existing file
                        let value = next_argument(&mut args, "importance matrix file, or 'auto'")?;
                        if value.to_str() == Some("auto") {
                            imatrix_tool = Some(ojon_backend::discovery::resolve_backend_command(
                                BackendId::LlamaImatrix,
                                "llama-imatrix",
                            )?);
                        } else {
                            imatrix = Some(PathBuf::from(value));
                        }
                    }
                    Some("--imatrix-tool") => imatrix_tool = Some(PathBuf::from(next_argument(&mut args, "llama-imatrix executable")?)),
                    Some("--budget-p99") => {
                        budget_p99 = Some(text(next_argument(&mut args, "p99 KLD budget")?, "budget-p99")?
                            .parse::<f64>()
                            .map_err(|_| invalid("budget-p99 must be a number (99th-percentile KLD delta over base)"))?)
                    }
                    Some("--quantize-tool") => quantize_tool = PathBuf::from(next_argument(&mut args, "llama-quantize executable")?),
                    Some("--perplexity-tool") => perplexity_tool = PathBuf::from(next_argument(&mut args, "llama-perplexity executable")?),
                    Some("--ctx") => context = text(next_argument(&mut args, "context")?, "ctx")?.parse().map_err(|_| invalid("ctx must be a number"))?,
                    Some("--ngl") => {
                        let v = text(next_argument(&mut args, "gpu layers or 'auto'")?, "ngl")?;
                        if v == "auto" { fit = true; } else { gpu_layers = v.parse().map_err(|_| invalid("ngl must be a number or 'auto'"))?; }
                    }
                    Some("--device") => device = Some(text(next_argument(&mut args, "backend device list, e.g. Vulkan0")?, "device")?),
                    Some("--chunks") => chunks = text(next_argument(&mut args, "chunk limit")?, "chunks")?.parse().map_err(|_| invalid("chunks must be a number"))?,
                    Some("--threads") => threads = text(next_argument(&mut args, "thread count")?, "threads")?.parse().map_err(|_| invalid("threads must be a number"))?,
                    Some("--keep-probes") => keep_probes = true,
                    Some("--resume") if execute => resume = true,
                    Some("--no-boundary-refine") => refine_boundaries = false,
                    Some("--json") => json = true,
                    Some("--confirm") if execute => confirm = true,
                    _ => return Err(invalid(format!("unknown recipe option: {}", option.to_string_lossy()))),
                }
            }
            let corpus = corpus.ok_or_else(|| invalid("--corpus <held-out text file> is required"))?;
            let base = ojon_backend::recipe::ladder_index(&base)
                .map(|i| ojon_backend::recipe::LADDER[i].0)
                .ok_or_else(|| invalid(format!("unknown base level '{base}'")))?;
            let floor = ojon_backend::recipe::ladder_index(&floor)
                .map(|i| ojon_backend::recipe::LADDER[i].0)
                .ok_or_else(|| invalid(format!("unknown floor level '{floor}'")))?;
            let plan = ojon_backend::recipe_run::prepare_recipe(ojon_backend::recipe_run::RecipeJob {
                source,
                output_dir,
                base,
                floor,
                budget_kld: budget,
                quantize_tool,
                imatrix,
                imatrix_tool,
                threads,
                kld: ojon_backend::kld::KldSettings { tool: perplexity_tool, corpus, gpu_layers, fit, context, chunks, device: device.clone() },
                keep_probes,
                budget_p99,
                resume,
                eval_corpus,
                refine_boundaries,
            })?;
            let plan_path = plan.write_plan()?;
            if json {
                println!("{}", serde_json::to_string_pretty(&plan.plan_json())?);
            } else {
                println!("plan: {}", plan_path.display());
                println!(
                    "classes: {} | probes: {} | quantizations: {} | base {} floor {} budget +{} mean KLD",
                    plan.present.len(),
                    plan.probes.len(),
                    plan.probes.len() + 2,
                    plan.job.base,
                    plan.job.floor,
                    plan.job.budget_kld
                );
            }
            if !execute {
                println!("next_step: ojon recipe-run <same arguments> --confirm");
                return Ok(true);
            }
            if !confirm {
                return Err(invalid("recipe-run quantizes the model once per probe; add --confirm to execute"));
            }
            let receipt = ojon_backend::recipe_run::run_recipe(&plan, |line| eprintln!("{line}"))?;
            println!("receipt: {}", receipt.display());
            Ok(true)
        }
        _ => Ok(false),
    }
}
