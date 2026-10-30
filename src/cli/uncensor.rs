//! Model abliteration, refusal evaluation, and auto-uncensoring commands.

use crate::cli::common::{invalid, next_argument};
use ojon_backend::*;
use std::ffi::OsString;
use std::io;
use std::path::{Path, PathBuf};
use std::str::FromStr;

pub fn handle(cmd: &str, mut args: impl Iterator<Item = OsString>) -> io::Result<bool> {
    match cmd {
        "compressed-edit-plan" | "compressed-edit-run" => {
            let execute = cmd == "compressed-edit-run";
            let base = PathBuf::from(next_argument(&mut args, "already quantized GGUF base")?);
            let recipe = PathBuf::from(next_argument(&mut args, "LoRA edit recipe JSON")?);
            let output = PathBuf::from(next_argument(&mut args, "new adapter bundle directory")?);
            let mut mmproj = None::<PathBuf>;
            let mut confirmed = false;
            let mut json = false;
            while let Some(option) = args.next() {
                match option.to_str() {
                    Some("--mmproj") => {
                        mmproj = Some(PathBuf::from(next_argument(
                            &mut args,
                            "multimodal mmproj GGUF",
                        )?))
                    }
                    Some("--confirm") => confirmed = true,
                    Some("--json") => json = true,
                    _ => {
                        return Err(invalid(format!(
                            "unknown compressed-edit option: {}",
                            option.to_string_lossy()
                        )))
                    }
                }
            }
            if confirmed && !execute {
                return Err(invalid("--confirm is only valid on compressed-edit-run"));
            }
            let plan =
                ojon_backend::prepare_compressed_edit(&base, &recipe, &output, mmproj.as_deref())?;
            if execute {
                if !confirmed {
                    return Err(invalid("compressed-edit-run requires --confirm"));
                }
                let result = ojon_backend::run_compressed_edit(&plan, |line| eprintln!("{line}"))?;
                if json {
                    println!("{}", result.as_json());
                } else {
                    println!(
                        "adapter_gguf: {}",
                        result.output.join("adapter.gguf").display()
                    );
                    println!(
                        "bundle_manifest: {}",
                        result.output.join("ojon-compressed-bundle.json").display()
                    );
                    println!("adapter_bytes: {}", result.adapter_bytes);
                    println!("original_model: NOT_COPIED_OR_MODIFIED");
                    println!("inference_compatibility: NOT_TESTED");
                    println!("refusal_rate: NOT_EVALUATED");
                    println!("vision_qa: NOT_EVALUATED");
                    println!("runtime: llama-cli -m <base.gguf> --lora <adapter.gguf> [--mmproj <vision.gguf>]");
                }
            } else {
                println!("{}", plan.as_json());
            }
            Ok(true)
        }
        "model-build-plan" | "model-build-run" => {
            let execute = cmd == "model-build-run";
            let source = PathBuf::from(next_argument(
                &mut args,
                "original Hugging Face model folder",
            )?);
            let recipe = PathBuf::from(next_argument(&mut args, "direction recipe JSON file")?);
            let output = PathBuf::from(next_argument(&mut args, "new output folder")?);
            let mut converter = None::<PathBuf>;
            let mut python = PathBuf::from("python");
            let mut quantizer = ojon_backend::discovered_path(BackendId::LlamaQuantize)
                .unwrap_or_else(|| PathBuf::from("llama-quantize"));
            let mut precision = Precision::default();
            let mut threads = std::thread::available_parallelism()
                .map(|x| x.get())
                .unwrap_or(4);
            let (mut confirmed, mut json) = (false, false);
            while let Some(option) = args.next() {
                match option.to_str() {
                    Some("--converter") => {
                        converter = Some(PathBuf::from(next_argument(&mut args, "converter path")?))
                    }
                    Some("--python") => {
                        python = PathBuf::from(next_argument(&mut args, "python executable")?)
                    }
                    Some("--quantizer") => {
                        quantizer =
                            PathBuf::from(next_argument(&mut args, "llama-quantize executable")?)
                    }
                    Some("--precision") => {
                        let raw = next_argument(&mut args, "precision")?;
                        precision = raw
                            .to_str()
                            .ok_or_else(|| invalid("precision must be text"))?
                            .parse::<Precision>()
                            .map_err(invalid)?;
                    }
                    Some("--threads") => {
                        threads = next_argument(&mut args, "threads")?
                            .to_string_lossy()
                            .parse()
                            .map_err(|_| invalid("threads must be a positive integer"))?;
                    }
                    Some("--json") => json = true,
                    Some("--confirm") => confirmed = true,
                    _ => {
                        return Err(invalid(format!(
                            "unknown build option: {}",
                            option.to_string_lossy()
                        )))
                    }
                }
            }
            let converter = converter.ok_or_else(|| {
                invalid(
                    "model build requires --converter <installed llama.cpp convert_hf_to_gguf.py>",
                )
            })?;
            let plan = ojon_backend::prepare_model_build(ojon_backend::ModelBuildRequest {
                source: &source,
                recipe: &recipe,
                output: &output,
                converter: &converter,
                python: &python,
                quantizer: &quantizer,
                precision,
                threads,
            })?;
            if execute {
                if !confirmed {
                    return Err(invalid("model-build-run requires --confirm"));
                }
                let result = ojon_backend::run_model_build(&plan, |msg| eprintln!("{msg}"))?;
                if json {
                    println!("{}", result.as_json());
                } else {
                    println!("output_folder: {}", result.directory.display());
                    println!(
                        "language_gguf: {}",
                        result.directory.join("model.gguf").display()
                    );
                    if result.has_mmproj {
                        println!(
                            "mmproj_gguf: {}",
                            result.directory.join("mmproj.gguf").display()
                        );
                    }
                    println!("model_load: NOT_TESTED");
                    println!("image_quality: NOT_TESTED");
                    println!("refusal_quality: NOT_TESTED");
                }
            } else {
                println!("{}", plan.as_json());
            }
            Ok(true)
        }
        "activation-compile" => {
            let source = PathBuf::from(next_argument(&mut args, "activation trace JSONL file")?);
            let dest = PathBuf::from(next_argument(&mut args, "output directions recipe JSON")?);
            let mut strength = 0.7_f64;
            let mut preserve = true;
            let mut confirmed = false;
            let mut json = false;
            while let Some(arg) = args.next() {
                match arg.to_str() {
                    Some("--strength") => {
                        strength = next_argument(&mut args, "strength")?
                            .to_string_lossy()
                            .parse()
                            .map_err(|_| invalid("strength must be decimal number"))?;
                    }
                    Some("--no-preserve") => preserve = false,
                    Some("--confirm") => confirmed = true,
                    Some("--json") => json = true,
                    _ => {
                        return Err(invalid(format!(
                            "unknown activation option: {}",
                            arg.to_string_lossy()
                        )))
                    }
                }
            }
            let result = ojon_backend::compile_activation_recipe(&source, strength, preserve)?;
            if !confirmed {
                if json {
                    println!("{}", result.as_json());
                } else {
                    println!(
                        "plan: {} records across {} matrices; add --confirm to write {}",
                        result.records,
                        result.tensors,
                        dest.display()
                    );
                }
            } else {
                let path = ojon_backend::save_activation_recipe(&result, &dest)?;
                if json {
                    println!(
                        "{}",
                        serde_json::json!({"recipe":path,"records":result.records,"tensors":result.tensors,"status":"RECIPE_WRITTEN"})
                    );
                } else {
                    println!("recipe_written: {}", path.display());
                }
            }
            Ok(true)
        }
        "uncensor-trace-plan" | "uncensor-trace-run" => {
            let execute = cmd == "uncensor-trace-run";
            let source = PathBuf::from(next_argument(
                &mut args,
                "original HF folder or float GGUF",
            )?);
            let trace = PathBuf::from(next_argument(&mut args, "paired activation JSONL")?);
            let output = PathBuf::from(next_argument(&mut args, "new model output")?);
            let (mut strength, mut preserve, mut confirmed, mut json) =
                (0.7f64, true, false, false);
            while let Some(option) = args.next() {
                match option.to_str() {
                    Some("--strength") => {
                        strength = next_argument(&mut args, "strength")?
                            .to_string_lossy()
                            .parse()
                            .map_err(|_| invalid("strength must be numeric"))?
                    }
                    Some("--no-preserve") => preserve = false,
                    Some("--confirm") => confirmed = true,
                    Some("--json") => json = true,
                    _ => {
                        return Err(invalid(format!(
                            "unknown uncensor-trace option: {}",
                            option.to_string_lossy()
                        )))
                    }
                }
            }
            if confirmed && !execute {
                return Err(invalid("--confirm only works with uncensor-trace-run"));
            }
            if execute && !confirmed {
                return Err(invalid("uncensor-trace-run requires --confirm"));
            }
            let result = if source.extension().and_then(|x| x.to_str()) == Some("gguf") {
                let activations =
                    ojon_backend::compile_activation_recipe(&trace, strength, preserve)?;
                let plan = ojon_backend::prepare_gguf_abliteration_value(
                    &source,
                    &activations.recipe,
                    &output,
                )?;
                if execute {
                    let edited = ojon_backend::run_gguf_abliteration(&plan, |message| {
                        eprintln!("{message}")
                    })?;
                    serde_json::json!({"schema":"ojon.uncensor_trace_result.v1",
                        "model":edited,"activation_records":activations.records,
                        "activation_tensors":activations.tensors,"trace":trace,
                        "activation_capture":"EXTERNAL_INSTRUMENTED_RUNTIME",
                        "strength_optimization":"NOT_RUN","refusal_rate":"NOT_EVALUATED",
                        "capability_retention":"NOT_EVALUATED","status":"EDITED_GGUF_NOT_VALIDATED"})
                } else {
                    serde_json::json!({"schema":"ojon.uncensor_trace_plan.v1",
                    "model_edit":plan.as_json(),"activation_records":activations.records,
                    "activation_tensors":activations.tensors,"trace":trace,
                    "activation_capture":"EXTERNAL_INSTRUMENTED_RUNTIME",
                    "strength_optimization":"NOT_RUN","refusal_rate":"NOT_EVALUATED",
                    "capability_retention":"NOT_EVALUATED","status":"PLAN_ONLY"})
                }
            } else {
                let plan = ojon_backend::prepare_uncensor_from_trace(
                    &source, &trace, &output, strength, preserve,
                )?;
                if execute {
                    ojon_backend::run_uncensor_from_trace(&plan, |message| eprintln!("{message}"))?
                } else {
                    plan.as_json()
                }
            };
            if json {
                println!("{result}");
            } else {
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
            Ok(true)
        }
        "abliterate-gguf-plan" | "abliterate-gguf-run" => {
            let execute = cmd == "abliterate-gguf-run";
            let source = PathBuf::from(next_argument(&mut args, "original float GGUF")?);
            let recipe = PathBuf::from(next_argument(&mut args, "direction recipe JSON")?);
            let output = PathBuf::from(next_argument(&mut args, "new edited GGUF")?);
            let (mut confirmed, mut json) = (false, false);
            for option in args.by_ref() {
                match option.to_str() {
                    Some("--confirm") => confirmed = true,
                    Some("--json") => json = true,
                    _ => {
                        return Err(invalid(format!(
                            "unknown abliterate-gguf option: {}",
                            option.to_string_lossy()
                        )))
                    }
                }
            }
            if confirmed && !execute {
                return Err(invalid("--confirm only works with abliterate-gguf-run"));
            }
            if execute && !confirmed {
                return Err(invalid("abliterate-gguf-run requires --confirm"));
            }
            let plan = ojon_backend::prepare_gguf_abliteration(&source, &recipe, &output)?;
            let result = if execute {
                ojon_backend::run_gguf_abliteration(&plan, |message| eprintln!("{message}"))?
            } else {
                plan.as_json()
            };
            if json {
                println!("{result}");
            } else {
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
            Ok(true)
        }
        "uncensor-cvector-plan" | "uncensor-cvector-run" => {
            let execute = cmd == "uncensor-cvector-run";
            let source = PathBuf::from(next_argument(&mut args, "original float GGUF")?);
            let vector = PathBuf::from(next_argument(&mut args, "llama.cpp control vector GGUF")?);
            let output = PathBuf::from(next_argument(&mut args, "new edited GGUF")?);
            let (mut block, mut strength, mut confirmed, mut json) = (None, 0.7f64, false, false);
            while let Some(option) = args.next() {
                match option.to_str() {
                    Some("--block") => {
                        block = Some(
                            next_argument(&mut args, "zero-based block index")?
                                .to_string_lossy()
                                .parse::<usize>()
                                .map_err(|_| invalid("block must be a nonnegative integer"))?,
                        )
                    }
                    Some("--strength") => {
                        strength = next_argument(&mut args, "strength")?
                            .to_string_lossy()
                            .parse()
                            .map_err(|_| invalid("strength must be numeric"))?
                    }
                    Some("--confirm") => confirmed = true,
                    Some("--json") => json = true,
                    _ => {
                        return Err(invalid(format!(
                            "unknown uncensor-cvector option: {}",
                            option.to_string_lossy()
                        )))
                    }
                }
            }
            if confirmed && !execute {
                return Err(invalid("--confirm only works with uncensor-cvector-run"));
            }
            if execute && !confirmed {
                return Err(invalid("uncensor-cvector-run requires --confirm"));
            }
            let block = block.ok_or_else(|| invalid("--block N is required"))?;
            let recipe =
                ojon_backend::cvector_import::recipe_from_control_vector(&vector, block, strength)?;
            let plan = ojon_backend::prepare_gguf_abliteration_value(&source, &recipe, &output)?;
            let model = if execute {
                ojon_backend::run_gguf_abliteration(&plan, |line| eprintln!("{line}"))?
            } else {
                plan.as_json()
            };
            let result = serde_json::json!({"schema":"ojon.uncensor_cvector_result.v1",
                "model":model,"control_vector":vector,"selected_block":block,
                "direction_origin":"LLAMA_CPP_RESIDUAL_CONTROL_VECTOR",
                "projection_target":"ATTENTION_OUTPUT_WEIGHT",
                "model_load":"NOT_TESTED","refusal_rate":"NOT_EVALUATED",
                "capability_retention":"NOT_EVALUATED",
                "status":if execute{"EDITED_GGUF_NOT_VALIDATED"}else{"PLAN_ONLY"}});
            if json {
                println!("{result}");
            } else {
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
            Ok(true)
        }
        "abliterate-model-plan" | "abliterate-model-run" => {
            let execute = cmd == "abliterate-model-run";
            let source = PathBuf::from(next_argument(
                &mut args,
                "original Hugging Face model folder",
            )?);
            let recipe = PathBuf::from(next_argument(&mut args, "direction recipe JSON")?);
            let output = PathBuf::from(next_argument(&mut args, "new model output folder")?);
            let (mut confirmed, mut json) = (false, false);
            for option in args.by_ref() {
                match option.to_str() {
                    Some("--confirm") => confirmed = true,
                    Some("--json") => json = true,
                    _ => {
                        return Err(invalid(format!(
                            "unknown option: {}",
                            option.to_string_lossy()
                        )))
                    }
                }
            }
            let plan = prepare_model_abliteration(&source, &recipe, &output)?;
            if execute {
                if !confirmed {
                    return Err(invalid(
                        "abliterate-model-run writes a new checkpoint and requires --confirm",
                    ));
                }
                let result = run_model_abliteration(&plan, |msg| eprintln!("{msg}"))?;
                if json {
                    println!("{}", result.as_json());
                } else {
                    println!("model_folder: {}", result.output.display());
                    println!("model_files: {}", result.files);
                    println!("edited_weight_tensors: {}", result.edited_tensors);
                    println!("edited_weight_shards: {}", result.edited_shards);
                    println!("original_model: PRESERVED");
                    println!("load_test: NOT_RUN");
                    println!("refusal_rate: NOT_EVALUATED");
                    println!("vision_qa: NOT_EVALUATED");
                    println!("quantization: NOT_RUN");
                }
            } else {
                println!("{}", plan.as_json());
            }
            Ok(true)
        }
        "abliterate-plan" | "abliterate-run" => {
            let execute = cmd == "abliterate-run";
            let source = PathBuf::from(next_argument(&mut args, "original safetensors shard")?);
            let recipe = PathBuf::from(next_argument(&mut args, "direction recipe JSON")?);
            let output = PathBuf::from(next_argument(&mut args, "new output safetensors shard")?);
            let (mut confirmed, mut json) = (false, false);
            for option in args.by_ref() {
                match option.to_str() {
                    Some("--confirm") => confirmed = true,
                    Some("--json") => json = true,
                    _ => {
                        return Err(invalid(format!(
                            "unknown option: {}",
                            option.to_string_lossy()
                        )))
                    }
                }
            }
            let plan = ojon_backend::prepare_abliteration(&source, &recipe, &output)?;
            if execute {
                if !confirmed {
                    return Err(invalid(
                        "abliterate-run modifies model output and requires --confirm",
                    ));
                }
                let result = ojon_backend::run_abliteration(&plan, |line| eprintln!("{line}"))?;
                if json {
                    println!("{}", result.as_json());
                } else {
                    println!("output: {}", result.destination.display());
                    println!("edited_tensors: {}", result.edited_tensors);
                    println!("output_bytes: {}", result.bytes);
                    println!("model_load: NOT_TESTED");
                    println!("refusal_rate: NOT_EVALUATED");
                    println!("vision_qa: NOT_EVALUATED");
                    println!("artifact: single safetensors shard, not a complete HF checkpoint");
                }
            } else {
                println!("{}", plan.as_json());
            }
            Ok(true)
        }
        "simple-plan" | "simple-run" => {
            let execute = cmd == "simple-run";
            let source = PathBuf::from(next_argument(
                &mut args,
                "original GGUF or HF model folder",
            )?);
            let output_dir = PathBuf::from(next_argument(&mut args, "existing output directory")?);
            let mut request = ojon_backend::simple_workflow::SimpleRequest::new(source, output_dir);
            let (mut confirmed, mut json) = (false, false);
            while let Some(option) = args.next() {
                match option.to_str() {
                    Some("--precision") => {
                        request.precision = Precision::from_str(
                            next_argument(&mut args, "Q4_K_M or Q5_K_M")?
                                .to_str()
                                .ok_or_else(|| invalid("precision must be UTF-8"))?,
                        )
                        .map_err(invalid)?
                    }
                    Some("--quantizer") => {
                        request.quantizer =
                            PathBuf::from(next_argument(&mut args, "llama-quantize executable")?)
                    }
                    Some("--runtime") => {
                        request.runtime =
                            PathBuf::from(next_argument(&mut args, "llama-cli executable")?)
                    }
                    Some("--gpu-layers") => {
                        request.gpu_layers = next_argument(&mut args, "GPU offload layers")?
                            .to_string_lossy()
                            .parse()
                            .map_err(|_| invalid("GPU layers must be integer"))?
                    }
                    Some("--refusal-edit") => {
                        request.refusal_examples = Some(PathBuf::from(next_argument(
                            &mut args,
                            "folder with paired positive.txt and negative.txt",
                        )?))
                    }
                    Some("--cvector-generator") => {
                        request.cvector_generator = Some(PathBuf::from(next_argument(
                            &mut args,
                            "llama-cvector-generator executable",
                        )?))
                    }
                    Some("--strength") => {
                        request.edit_strength = next_argument(&mut args, "edit strength")?
                            .to_string_lossy()
                            .parse()
                            .map_err(|_| invalid("strength must be numeric"))?
                    }
                    Some("--block") => {
                        request.edit_block = Some(
                            next_argument(&mut args, "zero-based block index")?
                                .to_string_lossy()
                                .parse::<usize>()
                                .map_err(|_| invalid("block must be an integer"))?,
                        )
                    }
                    Some("--refusal-suite") => {
                        request.refusal_suite = Some(PathBuf::from(next_argument(
                            &mut args,
                            "ojon.refusal_suite.v1 JSON",
                        )?))
                    }
                    Some("--task-suite") => {
                        request.task_suite = Some(PathBuf::from(next_argument(
                            &mut args,
                            "ojon.task_suite.v1 JSON",
                        )?))
                    }
                    Some("--json") => json = true,
                    Some("--confirm") => confirmed = true,
                    _ => {
                        return Err(invalid(format!(
                            "unknown simple option: {}",
                            option.to_string_lossy()
                        )))
                    }
                }
            }
            if confirmed && !execute {
                return Err(invalid("--confirm only works with simple-run"));
            }
            if execute && !confirmed {
                return Err(invalid(
                    "simple-run writes model files and requires --confirm",
                ));
            }
            let plan = ojon_backend::simple_workflow::prepare_simple_run(request)?;
            if execute {
                let result =
                    ojon_backend::simple_workflow::run_simple(&plan, |line| eprintln!("{line}"))?;
                if json {
                    println!("{result}");
                } else {
                    println!("{}", serde_json::to_string_pretty(&result)?);
                }
            } else if json {
                println!("{}", plan.as_json());
            } else {
                println!("output: {}", plan.as_json()["output"]);
                println!("status: {}", plan.as_json()["status"]);
            }
            Ok(true)
        }
        "refusal-eval-plan" | "refusal-eval-run" => {
            let execute = cmd == "refusal-eval-run";
            let model = PathBuf::from(next_argument(&mut args, "model GGUF")?);
            let suite = PathBuf::from(next_argument(&mut args, "ojon.refusal_suite.v1 JSON")?);
            let report = PathBuf::from(next_argument(&mut args, "new refusal report JSON")?);
            let (mut tool, mut server, mut device) = (None, None, None);
            let (mut gpu_layers, mut tokens) = (0i32, 96u32);
            let (mut confirmed, mut json) = (false, false);
            while let Some(option) = args.next() {
                match option.to_str() {
                    Some("--tool") => {
                        tool = Some(PathBuf::from(next_argument(
                            &mut args,
                            "llama-cli executable",
                        )?))
                    }
                    Some("--server") => {
                        server = Some(PathBuf::from(next_argument(
                            &mut args,
                            "local llama-server executable",
                        )?))
                    }
                    Some("--gpu-layers") => {
                        gpu_layers = next_argument(&mut args, "GPU offload layers")?
                            .to_string_lossy()
                            .parse()
                            .map_err(|_| invalid("GPU layers must be integer"))?
                    }
                    Some("--device") => {
                        device = Some(
                            next_argument(&mut args, "llama.cpp device ID")?
                                .to_string_lossy()
                                .into_owned(),
                        )
                    }
                    Some("--tokens") => {
                        tokens = next_argument(&mut args, "generation tokens")?
                            .to_string_lossy()
                            .parse()
                            .map_err(|_| invalid("tokens must be integer"))?
                    }
                    Some("--json") => json = true,
                    Some("--confirm") => confirmed = true,
                    _ => {
                        return Err(invalid(format!(
                            "unknown refusal-eval option: {}",
                            option.to_string_lossy()
                        )))
                    }
                }
            }
            if confirmed && !execute {
                return Err(invalid("--confirm only works with refusal-eval-run"));
            }
            if execute && !confirmed {
                return Err(invalid(
                    "refusal-eval-run runs real inference and requires --confirm",
                ));
            }
            let tool = tool
                .or_else(|| discovered_path(BackendId::LlamaCli))
                .ok_or_else(|| invalid("no llama-cli discovered; pass --tool"))?;
            let plan = ojon_backend::refusal::prepare_refusal_eval_with_server(
                ojon_backend::refusal::RefusalEvalRequest {
                    model: &model,
                    suite: &suite,
                    tool: &tool,
                    server_tool: server.as_deref(),
                    gpu_layers,
                    device: device.as_deref(),
                    tokens,
                    report: &report,
                },
            )?;
            if execute {
                let result =
                    ojon_backend::refusal::run_refusal_eval(&plan, |line| eprintln!("{line}"))?;
                if json {
                    println!("{result}");
                } else {
                    println!("model: {}", result["model"]);
                    println!("refusal_rate: {}", result["refusal_rate"]);
                    println!(
                        "overrefusal_rate: {}",
                        result["overrefusal_rate"]
                            .as_f64()
                            .map(|v| v.to_string())
                            .unwrap_or_else(|| "NOT_CONFIGURED".to_owned())
                    );
                    println!(
                        "probes_measured: {}",
                        result["results"].as_array().map(|r| r.len()).unwrap_or(0)
                    );
                }
            } else if json {
                println!("{}", plan.as_json());
            } else {
                println!("probes: {}", plan.as_json()["probes"]);
                println!("status: PLAN_ONLY");
            }
            Ok(true)
        }
        "uncensor-auto-plan" | "uncensor-auto-run" => {
            let execute = cmd == "uncensor-auto-run";
            let source = PathBuf::from(next_argument(&mut args, "original F16/BF16/F32 GGUF")?);
            let output_dir = PathBuf::from(next_argument(&mut args, "existing output directory")?);
            let mut request =
                ojon_backend::uncensor_auto::UncensorAutoRequest::new(source, output_dir);
            let (mut confirmed, mut json, mut explicit_gpu) = (false, false, false);
            while let Some(option) = args.next() {
                match option.to_str() {
                    Some("--refusal-suite") => {
                        request.refusal_probes =
                            PathBuf::from(next_argument(&mut args, "ojon.refusal_suite.v1 JSON")?)
                    }
                    Some("--task-suite") => {
                        request.capability_suite =
                            PathBuf::from(next_argument(&mut args, "ojon.task_suite.v1 JSON")?)
                    }
                    Some("--paired") => {
                        request.paired_prompts = PathBuf::from(next_argument(
                            &mut args,
                            "folder with paired positive.txt and negative.txt",
                        )?)
                    }
                    Some("--cvector-generator") => {
                        request.cvector_generator = PathBuf::from(next_argument(
                            &mut args,
                            "llama-cvector-generator executable",
                        )?)
                    }
                    Some("--runtime") => {
                        request.runtime =
                            PathBuf::from(next_argument(&mut args, "llama-cli executable")?)
                    }
                    Some("--server") => {
                        request.server_runtime = Some(PathBuf::from(next_argument(
                            &mut args,
                            "local llama-server executable",
                        )?))
                    }
                    Some("--gpu-layers") => {
                        explicit_gpu = true;
                        request.gpu_layers = next_argument(&mut args, "GPU offload layers")?
                            .to_string_lossy()
                            .parse()
                            .map_err(|_| invalid("GPU layers must be integer"))?;
                    }
                    Some("--device") => {
                        explicit_gpu = true;
                        request.device = Some(
                            next_argument(&mut args, "llama.cpp device ID")?
                                .to_string_lossy()
                                .into_owned(),
                        );
                    }
                    Some("--auto-hardware") => {
                        let (_, rdev, ngl) =
                            ojon_backend::resolve_auto_runtime_device("auto", None, 0);
                        request.device = rdev;
                        request.gpu_layers = ngl as i32;
                        explicit_gpu = true;
                    }
                    Some("--refusal-tokens") => {
                        request.refusal_tokens =
                            next_argument(&mut args, "refusal generation tokens")?
                                .to_string_lossy()
                                .parse()
                                .map_err(|_| invalid("tokens must be integer"))?
                    }
                    Some("--task-tokens") => {
                        request.task_tokens = next_argument(&mut args, "task generation tokens")?
                            .to_string_lossy()
                            .parse()
                            .map_err(|_| invalid("tokens must be integer"))?
                    }
                    Some("--strengths") => {
                        let spec = next_argument(&mut args, "comma separated strengths")?
                            .to_string_lossy()
                            .into_owned();
                        let mut parsed = Vec::new();
                        for part in spec.split(',').filter(|p| !p.trim().is_empty()) {
                            parsed.push(
                                part.trim()
                                    .parse::<f64>()
                                    .map_err(|_| invalid("each strength must be numeric"))?,
                            );
                        }
                        request.strengths = parsed;
                    }
                    Some("--blocks") => {
                        let spec = next_argument(&mut args, "comma separated block indices")?
                            .to_string_lossy()
                            .into_owned();
                        let mut parsed = Vec::new();
                        for part in spec.split(',').filter(|p| !p.trim().is_empty()) {
                            parsed.push(part.trim().parse::<usize>().map_err(|_| {
                                invalid("each block must be a nonnegative integer")
                            })?);
                        }
                        request.blocks = Some(parsed);
                    }
                    Some("--max-capability-loss") => {
                        request.max_capability_loss =
                            next_argument(&mut args, "maximum capability fraction loss")?
                                .to_string_lossy()
                                .parse()
                                .map_err(|_| invalid("loss must be numeric"))?
                    }
                    Some("--max-refusal-rate") => {
                        request.max_refusal_rate = Some(
                            next_argument(&mut args, "maximum measured refusal rate")?
                                .to_string_lossy()
                                .parse()
                                .map_err(|_| invalid("rate must be numeric"))?,
                        )
                    }
                    Some("--quantize") => {
                        request.quantize = Some(
                            Precision::from_str(
                                next_argument(&mut args, "Q4_K_M, Q5_K_M or Q6_K")?
                                    .to_str()
                                    .ok_or_else(|| invalid("precision must be UTF-8"))?,
                            )
                            .map_err(invalid)?,
                        )
                    }
                    Some("--quantizer") => {
                        request.quantizer =
                            PathBuf::from(next_argument(&mut args, "llama-quantize executable")?)
                    }
                    Some("--allow-resource-risk") => request.allow_resource_risk = true,
                    Some("--json") => json = true,
                    Some("--confirm") => confirmed = true,
                    _ => {
                        return Err(invalid(format!(
                            "unknown uncensor-auto option: {}",
                            option.to_string_lossy()
                        )))
                    }
                }
            }
            if confirmed && !execute {
                return Err(invalid("--confirm only works with uncensor-auto-run"));
            }
            if execute && !confirmed {
                return Err(invalid(
                    "uncensor-auto-run edits model files and requires --confirm",
                ));
            }
            if !explicit_gpu {
                let (_, rdev, ngl) = ojon_backend::resolve_auto_runtime_device("auto", None, 0);
                if rdev.is_some() {
                    request.device = rdev;
                    request.gpu_layers = ngl as i32;
                }
            }
            if request.refusal_probes.as_os_str().is_empty() {
                if let Some(bundled) =
                    ojon_backend::discovery::bundled_data_file("refusal-probes.json")
                {
                    request.refusal_probes = bundled;
                }
            }
            if request.capability_suite.as_os_str().is_empty() {
                if let Some(bundled) =
                    ojon_backend::discovery::bundled_data_file("sanity-tasks.json")
                {
                    request.capability_suite = bundled;
                }
            }
            if request.paired_prompts.as_os_str().is_empty() {
                if let Some(bundled) =
                    ojon_backend::discovery::bundled_data_dir("paired").or_else(|| {
                        ojon_backend::discovery::bundled_data_file("positive.txt")
                            .and_then(|p| p.parent().map(Path::to_path_buf))
                    })
                {
                    request.paired_prompts = bundled;
                }
            }
            if request.cvector_generator.as_os_str().is_empty()
                || request.cvector_generator == Path::new("llama-cvector-generator")
            {
                request.cvector_generator = discovered_path(BackendId::LlamaCvectorGenerator)
                    .ok_or_else(|| {
                        invalid("no llama-cvector-generator discovered; pass --cvector-generator")
                    })?;
            }
            if request.runtime.as_os_str().is_empty() || request.runtime == Path::new("llama-cli") {
                request.runtime = discovered_path(BackendId::LlamaCli)
                    .ok_or_else(|| invalid("no llama-cli discovered; pass --runtime"))?;
            }
            if request.server_runtime.is_none() {
                request.server_runtime = discovered_path(BackendId::LlamaServer);
            }
            if request.quantize.is_some()
                && (request.quantizer.as_os_str().is_empty()
                    || request.quantizer == Path::new("llama-quantize"))
            {
                request.quantizer = discovered_path(BackendId::LlamaQuantize)
                    .unwrap_or_else(|| PathBuf::from("llama-quantize"));
            }

            let plan = ojon_backend::uncensor_auto::prepare_uncensor_auto(request)?;
            if execute {
                let result = ojon_backend::uncensor_auto::run_uncensor_auto(&plan, |line| {
                    eprintln!("{line}")
                })?;
                if json {
                    println!("{result}");
                } else {
                    println!("status: {}", result["status"]);
                    println!("source: {}", result["source"]);
                    println!("output_edited: {}", result["output_edited"]);
                    println!(
                        "output_quantized: {}",
                        result["output_quantized"]
                            .as_str()
                            .unwrap_or("NOT_REQUESTED")
                    );
                    println!(
                        "baseline_refusal_rate: {}",
                        result["baseline"]["refusal_rate"]
                    );
                    for tested in result["candidates"].as_array().into_iter().flatten() {
                        println!(
                            "candidate: {} refusal_rate={} quality_pass={}",
                            tested["name"], tested["refusal_rate"], tested["quality_pass"]
                        );
                    }
                    if let Some(w) = result["method_choice"]["winner"].as_object() {
                        println!("winner: {} ({})", w["method"], w["artifact"]);
                    } else {
                        println!("winner: NONE");
                    }
                }
            } else if json {
                println!("{}", plan.as_json());
            } else {
                println!("output: {}", plan.as_json()["output"]);
                println!("candidates: {}", plan.as_json()["candidate_count"]);
                println!("status: {}", plan.as_json()["status"]);
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}
