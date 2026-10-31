//! Diffusion model quantization, packing, unpacking, and smoke test commands.

use crate::cli::common::{invalid, next_argument};
use ojon_backend::models::comfy_checkpoint::{prepare_minimax_checkpoint, run_minimax_checkpoint};
use ojon_backend::models::orbitquant::{prepare_minimax_w4a4, run_minimax_w4a4, MiniMaxComponent};
use ojon_backend::diffusion_gguf::*;
use ojon_backend::*;
use std::ffi::OsString;
use std::io;
use std::path::PathBuf;

pub fn handle(cmd: &str, mut args: impl Iterator<Item = OsString>) -> io::Result<bool> {
    match cmd {
        "minimax-video-test-plan" | "minimax-video-test-run" => {
            let execute = cmd == "minimax-video-test-run";
            let model = PathBuf::from(next_argument(&mut args, "packed MiniMax-H3 .safetensors")?);
            let workflow = PathBuf::from(next_argument(&mut args, "ComfyUI API-format video workflow JSON")?);
            let video_out = PathBuf::from(next_argument(&mut args, "new saved video filename")?);
            let receipt = PathBuf::from(next_argument(&mut args, "new validation receipt JSON")?);
            let mut python = PathBuf::from("python");
            let mut runtime = "safetensors".to_owned();
            let mut server = None::<String>;
            let mut model_name = None::<String>;
            let mut node_id = None::<String>;
            let mut ffprobe = None::<PathBuf>;
            let mut timeout = None::<u32>;
            let mut confirmed = false;
            while let Some(arg) = args.next() {
                match arg.to_str() {
                    Some("--python") => python = PathBuf::from(next_argument(&mut args, "python executable")?),
                    Some("--runtime") => {
                        runtime = next_argument(&mut args, "safetensors|orbitquant")?
                            .to_str().ok_or_else(|| invalid("runtime must be UTF-8"))?.to_owned();
                        if !matches!(runtime.as_str(), "safetensors" | "orbitquant") {
                            return Err(invalid("runtime must be safetensors or orbitquant"));
                        }
                    },
                    Some("--server") => server = Some(next_argument(&mut args, "loopback ComfyUI URL")?
                        .to_str().ok_or_else(|| invalid("server must be UTF-8"))?.to_owned()),
                    Some("--model-name") => model_name = Some(next_argument(&mut args, "ComfyUI listed model filename")?
                        .to_str().ok_or_else(|| invalid("model name must be UTF-8"))?.to_owned()),
                    Some("--node-id") => node_id = Some(next_argument(&mut args, "UNETLoader node ID")?
                        .to_str().ok_or_else(|| invalid("node id must be UTF-8"))?.to_owned()),
                    Some("--ffprobe") => ffprobe = Some(PathBuf::from(next_argument(&mut args, "ffprobe executable")?)),
                    Some("--timeout") => timeout = Some(next_argument(&mut args, "timeout seconds")?
                        .to_str().ok_or_else(|| invalid("timeout must be UTF-8"))?
                        .parse().map_err(|_| invalid("timeout must be integer seconds"))?),
                    Some("--confirm") if execute => confirmed = true,
                    _ => return Err(invalid(format!("unknown MiniMax video test argument: {}",
                        arg.to_string_lossy()))),
                }
            }
            if execute && !confirmed {
                return Err(invalid("real MiniMax video generation requires --confirm"));
            }
            // Embedded adapter keeps the compiled CLI independent of the working directory.
            // It submits only to loopback ComfyUI and requires a saved video artifact.
            let mut command = std::process::Command::new(python);
            command.arg("-c").arg(include_str!("../../scripts/comfy_safetensors_video_smoke.py"))
                .arg("--model").arg(model)
                .arg("--workflow").arg(workflow)
                .arg("--video-out").arg(video_out)
                .arg("--receipt").arg(receipt)
                .arg("--runtime").arg(runtime);
            if let Some(value) = server { command.arg("--server").arg(value); }
            if let Some(value) = model_name { command.arg("--model-name").arg(value); }
            if let Some(value) = node_id { command.arg("--node-id").arg(value); }
            if let Some(value) = ffprobe { command.arg("--ffprobe").arg(value); }
            if let Some(value) = timeout { command.arg("--timeout").arg(value.to_string()); }
            if execute { command.arg("--confirm"); }
            ojon_backend::jobs::run_with_log(&mut command, |line| println!("{line}"))?;
            Ok(true)
        }
        "minimax-checkpoint-plan" | "minimax-checkpoint-run" => {
            let execute = cmd == "minimax-checkpoint-run";
            let source = PathBuf::from(next_argument(&mut args, "single-file MiniMax-H3 BF16 .safetensors checkpoint")?);
            let output = PathBuf::from(next_argument(&mut args, "new ComfyUI NVFP4 .safetensors checkpoint")?);
            let mut converter = PathBuf::from("ctq");
            let mut device = "auto".to_owned();
            let mut confirmed = false;
            while let Some(opt) = args.next() {
                match opt.to_str() {
                    Some("--converter") => converter = PathBuf::from(next_argument(&mut args, "installed ctq executable")?),
                    Some("--device") => {
                        device = next_argument(&mut args, "cpu|cuda|xpu|auto")?
                            .to_str().ok_or_else(|| invalid("device must be UTF-8"))?.to_owned()
                    }
                    Some("--confirm") if execute => confirmed = true,
                    _ => return Err(invalid(format!(
                        "unknown MiniMax-H3 checkpoint option: {}",
                        opt.to_string_lossy()
                    ))),
                }
            }
            let plan = prepare_minimax_checkpoint(&source, &output, &converter, &device)?;
            if !execute {
                println!("{}", plan.json());
            } else {
                if !confirmed {
                    return Err(invalid("real MiniMax-H3 NVFP4 conversion requires --confirm"));
                }
                let result = run_minimax_checkpoint(&plan, |line| eprintln!("{line}"))?;
                println!("{result}");
            }
            Ok(true)
        }
        "minimax-w4a4-plan" | "minimax-w4a4-run" => {
            let execute = cmd == "minimax-w4a4-run";
            let source = PathBuf::from(next_argument(&mut args, "local MiniMax-H3 Diffusers folder")?);
            let output = PathBuf::from(next_argument(&mut args, "new packed W4A4 artifact directory")?);
            let mut python = PathBuf::from("python");
            let mut component = MiniMaxComponent::Fl2va;
            let mut device = "auto".to_owned();
            let mut confirm = false;
            while let Some(option) = args.next() {
                match option.to_str() {
                    Some("--python") => python = PathBuf::from(next_argument(&mut args, "Python interpreter")?),
                    Some("--component") => {
                        component = MiniMaxComponent::parse(
                            next_argument(&mut args, "fl2va|ref2va|text_encoder")?
                                .to_str().ok_or_else(|| invalid("component must be UTF-8"))?,
                        )?;
                    }
                    Some("--device") => {
                        device = next_argument(&mut args, "auto|cpu|cuda|xpu")?
                            .to_str().ok_or_else(|| invalid("device must be UTF-8"))?.to_owned();
                    }
                    Some("--confirm") if execute => confirm = true,
                    _ => return Err(invalid(format!(
                        "unknown MiniMax-H3 W4A4 option: {}",
                        option.to_string_lossy()
                    ))),
                }
            }
            let plan = prepare_minimax_w4a4(&source, &output, &python, component, &device)?;
            if !execute {
                println!("{}", plan.json());
            } else {
                if !confirm {
                    return Err(invalid("MiniMax-H3 W4A4 conversion requires --confirm"));
                }
                let result = run_minimax_w4a4(&plan, |line| eprintln!("{line}"))?;
                println!("{result}");
            }
            Ok(true)
        }
        "pack-plan" | "pack-run" => {
            let execute = cmd == "pack-run";
            let source = PathBuf::from(next_argument(
                &mut args,
                "original full-precision safetensors",
            )?);
            let output = PathBuf::from(next_argument(&mut args, "new Ojon .ojonq model file")?);
            let mut bits = 4u8;
            let mut group = 64usize;
            let mut confirmed = false;
            while let Some(option) = args.next() {
                match option.to_str() {
                    Some("--bits") => {
                        bits = next_argument(&mut args, "2,3,4 bit precision")?
                            .to_str()
                            .ok_or_else(|| invalid("bits must be text"))?
                            .parse()
                            .map_err(|_| invalid("invalid bits"))?;
                    }
                    Some("--group") => {
                        group = next_argument(&mut args, "32,64,128,256 group size")?
                            .to_str()
                            .ok_or_else(|| invalid("group must be text"))?
                            .parse()
                            .map_err(|_| invalid("invalid group size"))?;
                    }
                    Some("--confirm") if execute => confirmed = true,
                    _ => {
                        return Err(invalid(format!(
                            "unknown native pack option: {}",
                            option.to_string_lossy()
                        )))
                    }
                }
            }
            let plan = plan_pack(&source, &output, bits, group)?;
            if !execute {
                println!("{}", plan.json());
            } else {
                if !confirmed {
                    return Err(invalid(
                        "native pack-run writes a model file; pass --confirm",
                    ));
                }
                let report = run_pack(&plan, |line| eprintln!("{line}"))?;
                println!(
                    "{}",
                    serde_json::json!({
                        "output":report.output.to_string_lossy(),
                        "original_bytes":report.original_bytes,
                        "packed_bytes":report.output_bytes,
                        "quantized_tensors":report.packed_tensors,
                        "quantized_elements":report.elements,
                        "quantization_mse":report.mse,
                        "gpu_loader":"NOT_IMPLEMENTED",
                        "task_quality":"NOT_EVALUATED"
                    })
                );
            }
            Ok(true)
        }
        "unpack" => {
            let src = PathBuf::from(next_argument(&mut args, "Ojon .ojonq packed model")?);
            let out = PathBuf::from(next_argument(&mut args, "new ordinary .safetensors path")?);
            let confirm = next_argument(&mut args, "--confirm")?;
            if confirm != "--confirm" || args.next().is_some() {
                return Err(invalid(
                    "unpack writes a restored checkpoint: --confirm is required",
                ));
            }
            let saved = unpack(&src, &out)?;
            println!("restored_safetensors: {}", saved.display());
            println!(
                "warning: model weights are lossily reconstructed; no full task-quality claim"
            );
            Ok(true)
        }
        "diffusion-gguf-plan" | "diffusion-gguf-run" => {
            let execute = cmd == "diffusion-gguf-run";
            let source = PathBuf::from(next_argument(&mut args, "local denoiser safetensors")?);
            let output = PathBuf::from(next_argument(&mut args, "new output GGUF")?);
            let mut tool = ojon_backend::discovered_path(BackendId::StableDiffusionCpp)
                .unwrap_or_else(|| PathBuf::from("sd-cli"));
            let mut kind = MediaKind::Image;
            let mut layout = InputLayout::Diffusion;
            let mut precision = "q4_0".to_string();
            let mut confirm = false;
            while let Some(opt) = args.next() {
                match opt.to_str() {
                    Some("--tool") => tool = PathBuf::from(next_argument(&mut args, "sd-cli")?),
                    Some("--kind") => {
                        kind = MediaKind::parse(
                            next_argument(&mut args, "image|video")?
                                .to_str()
                                .ok_or_else(|| invalid("invalid model kind"))?,
                        )?
                    }
                    Some("--layout") => {
                        layout = InputLayout::parse(
                            next_argument(&mut args, "diffusion|checkpoint")?
                                .to_str()
                                .ok_or_else(|| invalid("invalid layout"))?,
                        )?
                    }
                    Some("--format") => {
                        precision = next_argument(&mut args, "GGUF quantization")?
                            .to_str()
                            .ok_or_else(|| invalid("invalid GGUF format"))?
                            .to_owned()
                    }
                    Some("--confirm") if execute => confirm = true,
                    _ => {
                        return Err(invalid(format!(
                            "unknown diffusion GGUF option: {}",
                            opt.to_string_lossy()
                        )))
                    }
                }
            }
            let plan = prepare_gguf_convert(&source, &output, &tool, kind, layout, &precision)?;
            if execute {
                if !confirm {
                    return Err(invalid("real sd-cli conversion requires --confirm"));
                }
                let result = run_gguf_convert(&plan, |line| eprintln!("{line}"))?;
                println!("{result}");
            } else {
                println!("{}", plan.json());
            }
            Ok(true)
        }
        "diffusion-sweep-plan" | "diffusion-sweep-run" => {
            // Image counterpart of recipe-run: convert each format, generate beside the
            // reference on one fixed-seed workflow, keep the smallest that passes the gate.
            let execute = cmd == "diffusion-sweep-run";
            let source = PathBuf::from(next_argument(&mut args, "local denoiser safetensors")?);
            let unet_dir = PathBuf::from(next_argument(&mut args, "ComfyUI models/unet directory (outputs land here)")?);
            let mut tool = ojon_backend::discovered_path(BackendId::StableDiffusionCpp)
                .unwrap_or_else(|| PathBuf::from("sd-cli"));
            let mut kind = MediaKind::Image;
            let mut layout = InputLayout::Diffusion;
            let mut formats = String::from("q8_0,q5_0,q4_K,q4_0,q3_K");
            let mut formats_given = false;
            let mut converter_name = String::from("sd-cli");
            let mut convert_script: Option<PathBuf> = None;
            let mut workflow: Option<PathBuf> = None;
            let mut reference: Option<String> = None;
            let mut server = String::from("http://127.0.0.1:8188");
            let mut python = PathBuf::from("python");
            let mut ffmpeg: Option<PathBuf> = None;
            let mut min_psnr = 30.0f64;
            let mut max_temporal_delta: Option<f64> = None;
            let mut node_id: Option<String> = None;
            let mut timeout_s = 1800u32;
            let (mut confirmed, mut json) = (false, false);
            let utf8 = |v: std::ffi::OsString, what: &str| -> io::Result<String> {
                v.to_str().map(str::to_owned).ok_or_else(|| invalid(format!("{what} must be UTF-8")))
            };
            while let Some(opt) = args.next() {
                match opt.to_str() {
                    Some("--tool") => tool = PathBuf::from(next_argument(&mut args, "sd-cli")?),
                    Some("--kind") => kind = MediaKind::parse(&utf8(next_argument(&mut args, "image|video")?, "kind")?)?,
                    Some("--layout") => layout = InputLayout::parse(&utf8(next_argument(&mut args, "diffusion|checkpoint")?, "layout")?)?,
                    Some("--formats") => { formats = utf8(next_argument(&mut args, "comma-separated formats")?, "formats")?; formats_given = true; }
                    Some("--converter") => converter_name = utf8(next_argument(&mut args, "sd-cli|comfy-gguf")?, "converter")?,
                    Some("--convert-script") => convert_script = Some(PathBuf::from(next_argument(&mut args, "ComfyUI-GGUF tools/convert.py")?)),
                    Some("--workflow") => workflow = Some(PathBuf::from(next_argument(&mut args, "API-format workflow JSON")?)),
                    Some("--reference-name") => reference = Some(utf8(next_argument(&mut args, "reference weights name")?, "reference name")?),
                    Some("--server") => server = utf8(next_argument(&mut args, "ComfyUI server URL")?, "server")?,
                    Some("--python") => python = PathBuf::from(next_argument(&mut args, "Python executable")?),
                    Some("--ffmpeg") => ffmpeg = Some(PathBuf::from(next_argument(&mut args, "ffmpeg executable")?)),
                    Some("--min-psnr") => min_psnr = utf8(next_argument(&mut args, "minimum PSNR dB")?, "min-psnr")?.parse().map_err(|_| invalid("min-psnr must be a number"))?,
                    Some("--max-temporal-delta") => max_temporal_delta = Some(utf8(next_argument(&mut args, "max temporal delta")?, "max-temporal-delta")?.parse().map_err(|_| invalid("max-temporal-delta must be a number"))?),
                    Some("--node-id") => node_id = Some(utf8(next_argument(&mut args, "loader node id")?, "node id")?),
                    Some("--timeout") => timeout_s = utf8(next_argument(&mut args, "timeout seconds")?, "timeout")?.parse().map_err(|_| invalid("timeout must be integer seconds"))?,
                    Some("--json") => json = true,
                    Some("--confirm") if execute => confirmed = true,
                    _ => return Err(invalid(format!("unknown diffusion-sweep option: {}", opt.to_string_lossy()))),
                }
            }
            let workflow = workflow.ok_or_else(|| invalid("--workflow is required"))?;
            let reference = reference.ok_or_else(|| invalid("--reference-name is required"))?;
            // The ComfyUI-GGUF converter (molbal fork) is what loads FIBO 2 and
            // friends; it is a Python script, so it becomes the "tool" and the
            // format vocabulary switches to its quant types.
            let converter = ojon_backend::diffusion_gguf::GgufConverter::parse(&converter_name, &python)?;
            if let ojon_backend::diffusion_gguf::GgufConverter::ComfyGguf { .. } = converter {
                tool = convert_script.clone().ok_or_else(|| invalid("--converter comfy-gguf needs --convert-script <ComfyUI-GGUF tools/convert.py>"))?;
                if !formats_given {
                    formats = String::from("Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q4_0");
                }
            }
            let plan = ojon_backend::diffusion_sweep::prepare_diffusion_sweep(ojon_backend::diffusion_sweep::DiffusionSweepRequest {
                source,
                comfy_unet_dir: unet_dir,
                tool,
                kind,
                layout,
                formats: ojon_backend::diffusion_sweep::parse_formats_for(&formats, &converter)?,
                converter: converter.clone(),
                workflow,
                reference_name: reference,
                server,
                python,
                ffmpeg,
                min_psnr,
                max_temporal_delta,
                node_id,
                timeout_s,
            })?;
            if json {
                println!("{}", serde_json::to_string_pretty(&plan.plan_json())?);
            } else {
                println!(
                    "sweep: {} formats ({}), {} generations, gate min PSNR {} dB{}",
                    plan.request.formats.len(),
                    plan.request.formats.join(","),
                    plan.request.formats.len() * 2,
                    plan.request.min_psnr,
                    plan.request.max_temporal_delta.map(|t| format!(" + temporal delta <= {t}")).unwrap_or_default()
                );
            }
            if !execute {
                println!("next_step: ojon diffusion-sweep-run <same arguments> --confirm");
                return Ok(true);
            }
            if !confirmed {
                return Err(invalid("diffusion-sweep-run converts and generates for every format; add --confirm"));
            }
            let receipt = ojon_backend::diffusion_sweep::run_diffusion_sweep(&plan, |line| eprintln!("{line}"))?;
            println!("receipt: {}", receipt.display());
            Ok(true)
        }
        "diffusion-parity" => {
            // Measured image parity: reference vs candidate on one fixed-seed ComfyUI workflow.
            let mut workflow: Option<PathBuf> = None;
            let mut reference: Option<String> = None;
            let mut candidate: Option<String> = None;
            let mut node_id: Option<String> = None;
            let mut server: Option<String> = None;
            let mut receipt: Option<PathBuf> = None;
            let mut python = PathBuf::from("python");
            let mut min_psnr: Option<String> = None;
            let mut pixel_tolerance: Option<String> = None;
            let mut kind: Option<String> = None;
            let mut ffmpeg: Option<PathBuf> = None;
            let mut max_temporal_delta: Option<String> = None;
            let mut timeout: Option<u32> = None;
            let mut confirmed = false;
            let utf8 = |v: std::ffi::OsString, what: &str| -> io::Result<String> {
                v.to_str().map(str::to_owned).ok_or_else(|| invalid(format!("{what} must be UTF-8")))
            };
            while let Some(opt) = args.next() {
                match opt.to_str() {
                    Some("--workflow") => workflow = Some(PathBuf::from(next_argument(&mut args, "API-format workflow JSON")?)),
                    Some("--reference-name") => reference = Some(utf8(next_argument(&mut args, "reference weights name")?, "reference name")?),
                    Some("--candidate-name") => candidate = Some(utf8(next_argument(&mut args, "candidate GGUF name")?, "candidate name")?),
                    Some("--node-id") => node_id = Some(utf8(next_argument(&mut args, "loader node id")?, "node id")?),
                    Some("--server") => server = Some(utf8(next_argument(&mut args, "ComfyUI server URL")?, "server")?),
                    Some("--receipt") => receipt = Some(PathBuf::from(next_argument(&mut args, "receipt JSON path")?)),
                    Some("--python") => python = PathBuf::from(next_argument(&mut args, "Python executable")?),
                    Some("--min-psnr") => {
                        let v = utf8(next_argument(&mut args, "minimum PSNR dB")?, "min-psnr")?;
                        v.parse::<f64>().map_err(|_| invalid("min-psnr must be a number"))?;
                        min_psnr = Some(v);
                    }
                    Some("--kind") => {
                        let v = utf8(next_argument(&mut args, "image|video")?, "kind")?;
                        if v != "image" && v != "video" {
                            return Err(invalid("kind must be image or video"));
                        }
                        kind = Some(v);
                    }
                    Some("--ffmpeg") => ffmpeg = Some(PathBuf::from(next_argument(&mut args, "ffmpeg executable")?)),
                    Some("--max-temporal-delta") => {
                        let v = utf8(next_argument(&mut args, "max temporal delta")?, "max-temporal-delta")?;
                        v.parse::<f64>().map_err(|_| invalid("max-temporal-delta must be a number"))?;
                        max_temporal_delta = Some(v);
                    }
                    Some("--pixel-tolerance") => {
                        let v = utf8(next_argument(&mut args, "pixel tolerance")?, "pixel-tolerance")?;
                        v.parse::<u32>().map_err(|_| invalid("pixel-tolerance must be an integer"))?;
                        pixel_tolerance = Some(v);
                    }
                    Some("--timeout") => {
                        timeout = Some(
                            utf8(next_argument(&mut args, "timeout seconds")?, "timeout")?
                                .parse()
                                .map_err(|_| invalid("timeout must be integer seconds"))?,
                        )
                    }
                    Some("--confirm") => confirmed = true,
                    _ => return Err(invalid(format!("unknown diffusion-parity option: {}", opt.to_string_lossy()))),
                }
            }
            let workflow = workflow.ok_or_else(|| invalid("--workflow is required"))?;
            let reference = reference.ok_or_else(|| invalid("--reference-name is required"))?;
            let candidate = candidate.ok_or_else(|| invalid("--candidate-name is required"))?;
            if !workflow.is_file() {
                return Err(invalid("workflow JSON not found"));
            }
            if !confirmed {
                return Err(invalid("diffusion-parity generates twice on the ComfyUI server; add --confirm"));
            }
            // Same embedding approach as the smoke gate: stdlib-only adapter, no shell,
            // loopback-only server enforced inside the script.
            let mut command = std::process::Command::new(python);
            command
                .arg("-c")
                .arg(include_str!("../../scripts/comfy_gguf_parity.py"))
                .arg("--workflow")
                .arg(workflow)
                .arg("--reference-name")
                .arg(reference)
                .arg("--candidate-name")
                .arg(candidate);
            if let Some(v) = node_id {
                command.arg("--node-id").arg(v);
            }
            if let Some(v) = server {
                command.arg("--server").arg(v);
            }
            if let Some(v) = receipt {
                command.arg("--receipt").arg(v);
            }
            if let Some(v) = min_psnr {
                command.arg("--min-psnr").arg(v);
            }
            if let Some(v) = pixel_tolerance {
                command.arg("--pixel-tolerance").arg(v);
            }
            if let Some(v) = kind {
                command.arg("--kind").arg(v);
            }
            if let Some(v) = ffmpeg {
                command.arg("--ffmpeg").arg(v);
            }
            if let Some(v) = max_temporal_delta {
                command.arg("--max-temporal-delta").arg(v);
            }
            if let Some(v) = timeout {
                command.arg("--timeout").arg(v.to_string());
            }
            ojon_backend::jobs::run_with_log(&mut command, |line| println!("{line}"))?;
            Ok(true)
        }
        "diffusion-gguf-smoke-plan" | "diffusion-gguf-smoke-run" => {
            let execute = cmd == "diffusion-gguf-smoke-run";
            let model = PathBuf::from(next_argument(&mut args, "converted GGUF denoiser")?);
            let output = PathBuf::from(next_argument(&mut args, "new PNG or WebM output")?);
            let mut tool = ojon_backend::discovered_path(BackendId::StableDiffusionCpp)
                .unwrap_or_else(|| PathBuf::from("sd-cli"));
            let mut kind = MediaKind::Image;
            let mut layout = InputLayout::Diffusion;
            let mut backend: Option<String> = None;
            let mut components: Vec<(String, PathBuf)> = Vec::new();
            let mut prompt = "a simple wooden cube on a plain background".to_owned();
            let mut seed = 42i64;
            let mut steps = 4u32;
            let mut width = 512u32;
            let mut height = 512u32;
            let mut frames = 9u32;
            let mut fps = 8u32;
            let mut offload = false;
            let mut confirm = false;
            while let Some(opt) = args.next() {
                match opt.to_str() {
                    Some("--tool") => tool = PathBuf::from(next_argument(&mut args, "sd-cli")?),
                    Some("--kind") => {
                        kind = MediaKind::parse(
                            next_argument(&mut args, "image|video")?
                                .to_str()
                                .ok_or_else(|| invalid("invalid kind"))?,
                        )?
                    }
                    Some("--layout") => {
                        layout = InputLayout::parse(
                            next_argument(&mut args, "diffusion|checkpoint")?
                                .to_str()
                                .ok_or_else(|| invalid("invalid layout"))?,
                        )?
                    }
                    Some("--backend") => {
                        backend = Some(
                            next_argument(&mut args, "sd-cli device assignment")?
                                .to_str()
                                .ok_or_else(|| invalid("invalid backend"))?
                                .to_owned(),
                        )
                    }
                    Some("--component") => {
                        let value = next_argument(&mut args, "name=local-file")?;
                        let value = value
                            .to_str()
                            .ok_or_else(|| invalid("component must be UTF-8"))?;
                        let (name, path) = value
                            .split_once('=')
                            .ok_or_else(|| invalid("component must be name=path"))?;
                        components.push((name.to_owned(), PathBuf::from(path)));
                    }
                    Some("--prompt") => {
                        prompt = next_argument(&mut args, "prompt")?
                            .to_str()
                            .ok_or_else(|| invalid("prompt must be UTF-8"))?
                            .into()
                    }
                    Some("--seed") => {
                        seed = next_argument(&mut args, "seed")?
                            .to_str()
                            .ok_or_else(|| invalid("seed must be numeric"))?
                            .parse()
                            .map_err(|_| invalid("invalid seed"))?
                    }
                    Some("--steps") => {
                        steps = next_argument(&mut args, "steps")?
                            .to_str()
                            .ok_or_else(|| invalid("steps must be numeric"))?
                            .parse()
                            .map_err(|_| invalid("invalid steps"))?
                    }
                    Some("--width") => {
                        width = next_argument(&mut args, "width")?
                            .to_str()
                            .ok_or_else(|| invalid("width must be numeric"))?
                            .parse()
                            .map_err(|_| invalid("invalid width"))?
                    }
                    Some("--height") => {
                        height = next_argument(&mut args, "height")?
                            .to_str()
                            .ok_or_else(|| invalid("height must be numeric"))?
                            .parse()
                            .map_err(|_| invalid("invalid height"))?
                    }
                    Some("--frames") => {
                        frames = next_argument(&mut args, "frames")?
                            .to_str()
                            .ok_or_else(|| invalid("frames must be numeric"))?
                            .parse()
                            .map_err(|_| invalid("invalid frames"))?
                    }
                    Some("--fps") => {
                        fps = next_argument(&mut args, "fps")?
                            .to_str()
                            .ok_or_else(|| invalid("fps must be numeric"))?
                            .parse()
                            .map_err(|_| invalid("invalid fps"))?
                    }
                    Some("--offload-to-cpu") => offload = true,
                    Some("--confirm") if execute => confirm = true,
                    _ => {
                        return Err(invalid(format!(
                            "unknown diffusion smoke option: {}",
                            opt.to_string_lossy()
                        )))
                    }
                }
            }
            let plan = prepare_gguf_smoke(
                &model,
                &output,
                &tool,
                kind,
                layout,
                &components,
                &prompt,
                seed,
                steps,
                width,
                height,
                frames,
                fps,
                offload,
                backend.as_deref(),
            )?;
            if execute {
                if !confirm {
                    return Err(invalid(
                        "generating media loads real weights and requires --confirm",
                    ));
                }
                let result = run_gguf_smoke(&plan, |line| eprintln!("{line}"))?;
                println!("{result}");
            } else {
                println!("{}", plan.json());
            }
            Ok(true)
        }
        "diffusion-gguf-comfy-plan" | "diffusion-gguf-comfy-run" => {
            let execute = cmd == "diffusion-gguf-comfy-run";
            let model = PathBuf::from(next_argument(&mut args, "local GGUF denoiser")?);
            let workflow = PathBuf::from(next_argument(
                &mut args,
                "ComfyUI API-format workflow JSON",
            )?);
            let receipt = PathBuf::from(next_argument(
                &mut args,
                "new ComfyUI runtime receipt JSON",
            )?);
            let mut python = PathBuf::from("python");
            let mut kind = MediaKind::Image;
            let mut node_id = None::<String>;
            let mut model_name = None::<String>;
            let mut server = None::<String>;
            let mut timeout = None::<u32>;
            let mut confirmed = false;
            while let Some(opt) = args.next() {
                match opt.to_str() {
                    Some("--python") => {
                        python = PathBuf::from(next_argument(&mut args, "Python executable")?)
                    }
                    Some("--kind") => {
                        kind = MediaKind::parse(
                            next_argument(&mut args, "image|video")?
                                .to_str()
                                .ok_or_else(|| invalid("invalid media kind"))?,
                        )?
                    }
                    Some("--node-id") => {
                        node_id = Some(
                            next_argument(&mut args, "ComfyUI loader node ID")?
                                .to_str()
                                .ok_or_else(|| invalid("invalid node ID"))?
                                .to_owned(),
                        )
                    }
                    Some("--model-name") => {
                        model_name = Some(
                            next_argument(&mut args, "ComfyUI listed model name")?
                                .to_str()
                                .ok_or_else(|| invalid("invalid model name"))?
                                .to_owned(),
                        )
                    }
                    Some("--server") => {
                        server = Some(
                            next_argument(&mut args, "loopback ComfyUI API URL")?
                                .to_str()
                                .ok_or_else(|| invalid("invalid server"))?
                                .to_owned(),
                        )
                    }
                    Some("--timeout") => {
                        timeout = Some(
                            next_argument(&mut args, "timeout seconds")?
                                .to_str()
                                .ok_or_else(|| invalid("invalid timeout"))?
                                .parse()
                                .map_err(|_| invalid("timeout must be integer seconds"))?,
                        )
                    }
                    Some("--confirm") if execute => confirmed = true,
                    _ => {
                        return Err(invalid(format!(
                            "unknown ComfyUI GGUF option: {}",
                            opt.to_string_lossy()
                        )))
                    }
                }
            }
            if execute && !confirmed {
                return Err(invalid("ComfyUI generation requires --confirm"));
            }
            // Embedding the stdlib-only adapter makes the compiled Ojon CLI
            // independent of the caller's working directory and script files.
            // No shell is used. The Python adapter itself requires loopback-only
            // ComfyUI and refuses to publish a receipt for failed generations.
            let mut command = std::process::Command::new(python);
            command
                .arg("-c")
                .arg(include_str!("../../scripts/comfy_gguf_smoke.py"))
                .arg("--model")
                .arg(model)
                .arg("--workflow")
                .arg(workflow)
                .arg("--receipt")
                .arg(receipt)
                .arg("--kind")
                .arg(kind.name());
            if let Some(value) = node_id {
                command.arg("--node-id").arg(value);
            }
            if let Some(value) = model_name {
                command.arg("--model-name").arg(value);
            }
            if let Some(value) = server {
                command.arg("--server").arg(value);
            }
            if let Some(value) = timeout {
                command.arg("--timeout").arg(value.to_string());
            }
            if execute {
                command.arg("--confirm");
            }
            ojon_backend::jobs::run_with_log(&mut command, |line| println!("{line}"))?;
            Ok(true)
        }
        "diffusion-plan" | "diffusion-inspect" | "diffusion-run" => {
            let stage = cmd.to_string();
            let source = PathBuf::from(next_argument(
                &mut args,
                "local source model or Diffusers pipeline",
            )?);
            let output = PathBuf::from(next_argument(
                &mut args,
                "new quantized safetensors output",
            )?);
            let mut engine = DiffusionEngine::Svdq;
            let mut kind = DiffusionKind::Image;
            let mut precision = DiffusionFormat::Int4;
            let mut checkout: Option<PathBuf> = None;
            let mut python = PathBuf::from("python");
            let mut samples = 32u32;
            let mut confirm = false;
            while let Some(option) = args.next() {
                match option.to_str() {
                    Some("--checkout") => {
                        checkout = Some(PathBuf::from(next_argument(
                            &mut args,
                            "upstream Python repository checkout",
                        )?))
                    }
                    Some("--python") => {
                        python = PathBuf::from(next_argument(&mut args, "Python interpreter")?)
                    }
                    Some("--backend") => {
                        let name = next_argument(&mut args, "backend svdq")?;
                        engine = DiffusionEngine::from_name(
                            name.to_str()
                                .ok_or_else(|| invalid("invalid backend name"))?,
                        )?;
                    }
                    Some("--kind") => {
                        let name = next_argument(&mut args, "image|video")?;
                        kind = DiffusionKind::from_name(
                            name.to_str().ok_or_else(|| invalid("invalid model kind"))?,
                        )?;
                    }
                    Some("--format") => {
                        let name = next_argument(&mut args, "int4")?;
                        precision = DiffusionFormat::from_name(
                            name.to_str().ok_or_else(|| invalid("invalid format"))?,
                        )?;
                    }
                    Some("--samples") => {
                        samples = next_argument(&mut args, "calibration sample count")?
                            .to_str()
                            .ok_or_else(|| invalid("samples must be a number"))?
                            .parse()
                            .map_err(|_| invalid("samples must be an integer"))?;
                    }
                    Some("--confirm") if stage != "diffusion-plan" => confirm = true,
                    _ => {
                        return Err(invalid(format!(
                            "unknown diffusion option: {}",
                            option.to_string_lossy()
                        )))
                    }
                }
            }
            let checkout = checkout
                .or_else(|| std::env::var_os("OJON_SVDQ_DIR").map(PathBuf::from))
                .ok_or_else(|| invalid("set --checkout to diffuse-compressor or OJON_SVDQ_DIR"))?;
            let plan = plan_diffusion(
                DiffusionPlanRequest {
                    source: &source,
                    output: &output,
                    checkout: &checkout,
                    python: &python,
                    engine,
                    kind,
                    format: precision,
                    samples,
                },
            )?;
            if stage == "diffusion-plan" {
                println!("{}", plan.json());
            } else {
                if !confirm {
                    return Err(invalid("diffusion inspection/conversion may load large models and run third-party Python; explicitly pass --confirm"));
                }
                println!("{}", plan.json());
                if stage == "diffusion-inspect" {
                    inspect_diffusion_recipe(&plan, |line| eprintln!("{line}"))?;
                    println!(
                        "diffusion_recipe: INSPECTED_BY_UPSTREAM; model_conversion: NOT_STARTED"
                    );
                } else {
                    let outcome = run_diffusion(&plan, |line| eprintln!("{line}"))?;
                    println!("output: {}", outcome.output.display());
                    println!("output_bytes: {}", outcome.output_bytes);
                    println!("conversion_seconds: {:.2}", outcome.elapsed_seconds);
                    println!("runtime_load: NOT_TESTED");
                    println!("image_video_quality: NOT_EVALUATED");
                    if let Some(config) = outcome.sidecar {
                        println!("config: {}", config.display());
                    }
                    if let Some(receipt) = outcome.receipt {
                        println!("receipt: {}", receipt.display());
                    }
                }
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}
