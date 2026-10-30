//! Read-only, multi-route quantization comparison planning.
//! Model classification is advisory; finding an executable does not prove
//! model family compatibility, GPU kernel availability, or inference quality.
use crate::discovery::{BackendId, FoundBackend};
use crate::gguf::inspect_gguf;
use crate::probe::{ModelCategory, ModelProbe};
use serde_json::{json, Value};
use std::io::{self, ErrorKind};

/// One advisory quantization route row before JSON serialisation.
#[derive(Clone, Copy, Debug)]
struct Candidate<'a> {
    pub id: &'a str,
    pub family: &'a str,
    pub method: &'a str,
    pub format: &'a str,
    pub loader: &'a str,
    pub state: &'a str,
    pub reason: &'a str,
    pub next_command: &'a str,
}
fn candidate(row: Candidate<'_>) -> Value {
    let Candidate {
        id,
        family,
        method,
        format,
        loader,
        state,
        reason,
        next_command,
    } = row;
    json!({
        "id": id, "backend": family, "method": method,
        "output_format": format, "candidate_loader": loader,
        "state": state, "reason": reason,
        "existing_manual_command": next_command,
        "backend_version_verified": false,
        "model_compatibility_verified": false,
        "runtime_load_verified": false,
        "task_quality_verified": false,
        "estimated_peak_ram_bytes": null,
        "estimated_peak_vram_bytes": null,
        "measured_output_bytes": null,
        "measured_tokens_per_second": null,
        "measured_task_quality": null,
    })
}

fn found(tools: &[FoundBackend], id: BackendId) -> bool {
    tools.iter().any(|b| b.spec.id == id && b.installed())
}

fn plan_for(
    probe: &ModelProbe,
    device: &str,
    tools: &[FoundBackend],
    gguf: Option<(Option<u32>, u16)>,
) -> io::Result<Value> {
    if !matches!(device, "auto" | "cpu" | "cuda" | "xpu") {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "device must be auto, cpu, cuda, or xpu",
        ));
    }
    let mut options: Vec<Value> = Vec::new();
    let quantizer = found(tools, BackendId::LlamaQuantize);
    let converter = found(tools, BackendId::HfConverter);
    let autoround = found(tools, BackendId::IntelAutoRound);
    let text_like = matches!(
        probe.category,
        ModelCategory::Language | ModelCategory::VisionLanguage
    );
    let multi = probe.category == ModelCategory::VisionLanguage;

    if probe.format == "gguf" && text_like {
        let original = matches!(gguf, Some((Some(0 | 1 | 32), 1)));
        let state = if !original {
            "blocked_source"
        } else if !quantizer {
            "missing_tool"
        } else {
            "requires_compatibility_test"
        };
        let reason = if !original {
            "A comparison sweep requires original F32/F16/BF16 single-part GGUF; avoid requantizing an already compressed file"
        } else if !quantizer {
            "llama-quantize not detected; install or explicitly configure the official executable"
        } else {
            "Quantizer detected; model-family support, target runtime, RAM/VRAM, and output quality are not verified"
        };
        for (id, method, format, command) in [
            (
                "gguf_q5",
                "K-quant quality-oriented",
                "Q5_K_M",
                "sweep-plan",
            ),
            ("gguf_q4", "K-quant balanced", "Q4_K_M", "sweep-plan"),
            (
                "gguf_iq4",
                "importance-calibrated low-bit",
                "IQ4_XS",
                "imatrix-plan / sweep-plan",
            ),
        ] {
            let mut item = candidate(Candidate {
                id,
                family: "llama.cpp",
                method,
                format,
                loader: "llama.cpp (runtime unverified)",
                state,
                reason,
                next_command: command,
            });
            if multi {
                item["multimodal_requirement"] =
                    json!("Vision encoder, processor and projector require separate validation");
            }
            if id == "gguf_iq4" {
                item["calibration"] = json!("Representative importance matrix recommended; use a separate held-out quality corpus");
            }
            options.push(item);
        }
    } else if probe.format == "hf_checkpoint_directory" && text_like {
        for (id, format) in [("hf_gguf_q5", "Q5_K_M"), ("hf_gguf_q4", "Q4_K_M")] {
            let state = if converter && quantizer {
                "requires_compatibility_test"
            } else {
                "missing_tool"
            };
            let reason = if converter && quantizer {
                "HF converter and quantizer detected; architecture, dependencies, source shards, scratch, target loader and quality remain unverified"
            } else {
                "Requires an installed convert_hf_to_gguf.py and llama-quantize; detected filenames are not capability proof"
            };
            options.push(candidate(Candidate {
                id,
                family: "llama.cpp",
                method: "HF -> high-precision GGUF -> K-quant",
                format,
                loader: "llama.cpp (runtime unverified)",
                state,
                reason,
                next_command: "quantize-hf",
            }));
        }
        let ar_reason = if autoround {
            "AutoRound CLI detected, but matching Python package, device kernels, GGUF export and loader must be verified"
        } else {
            "AutoRound CLI not detected; Python module, device kernels, output and inference loader must be checked separately"
        };
        options.push(candidate(Candidate {
            id: "hf_autoround_4",
            family: "intel_auto_round",
            method: "optimized RTN or calibrated AutoRound (select --iters and local --dataset)",
            format: "GGUF Q4_K_M or Q5_K_M (experimental upstream export)",
            loader: "compatible GGUF runtime, not yet verified",
            state: if autoround {
                "requires_compatibility_test"
            } else {
                "missing_tool"
            },
            reason: ar_reason,
            next_command: "autoround-plan --bits 4|5 --format gguf:q4_k_m|gguf:q5_k_m",
        }));
        if !multi {
            options.push(candidate(
                Candidate {
                    id: "hf_exl3",
                    family: "exllamav3",
                    method: "QTIP-inspired trellis-coded fractional-bitrate quantization",
                    format: "EXL3 (NOT GGUF)",
                    loader: "ExLlamaV3 CUDA runtime; NVIDIA only, not Intel Arc",
                    state: "external_python_unverified",
                    reason: "Requires user-installed ExLlamaV3 checkout, original safetensors, local CUDA and memory preflight; output and loader are not verified",
                    next_command: "exl3-plan <source> <new-output> <new-work> --checkout <exllamav3> --bits 3.5",
                },
            ));
            options.push(candidate(
                Candidate {
                    id: "hf_hqq_q4",
                    family: "dropbox_hqq",
                    method: "calibration-free half-quadratic weight quantization",
                    format: "HQQ native qmodel.pt (not GGUF)",
                    loader: "HQQ AutoHQQHFModel loader only (unverified)",
                    state: "external_python_unverified",
                    reason: "Needs installed torch, hqq and transformers; full HF checkpoint loaded; no GGUF converter or Intel XPU kernel compatibility implied",
                    next_command: "research-plan hqq <source> <new-output> --bits 4 --group 64",
                },
            ));
            for (id, ratio) in [("hf_openvino_int4", "1.0"), ("hf_openvino_mixed", "0.8")] {
                options.push(candidate(
                    Candidate {
                        id,
                        family: "openvino_nncf",
                        method: "OpenVINO/NNCF weight-only INT4 with optional INT8 backup layers",
                        format: "OpenVINO IR .xml/.bin (not GGUF)",
                        loader: "OpenVINO / Optimum Intel (loader unverified)",
                        state: "external_python_unverified",
                        reason: "Requires installed optimum-cli and model-specific OpenVINO support; quantization can use significant CPU RAM",
                        next_command: if ratio == "1.0" { "research-plan openvino <source> <new-output> --ratio 1.0" }
                        else { "research-plan openvino <source> <new-output> --ratio 0.8" },
                    },
                ));
            }
        }
        if multi {
            for item in &mut options {
                item["multimodal_requirement"] = json!("Preserve and verify vision encoder, processor, projector, and actual image-input inference");
            }
        }
    } else if probe.category == ModelCategory::Diffusion && probe.format != "ojonq_packed_weights" {
        let is_pipeline = probe.format == "diffusers_pipeline_directory";
        let is_weights = probe.format == "safetensors_single_file";
        options.push(candidate(
            Candidate {
                id: "diffusion_svdq",
                family: "diffuse-compressor",
                method: "SVDQuant transformer INT4",
                format: "SVDQ INT4",
                loader: "specific supported Diffusers/CUDA runtime",
                state: if is_pipeline { "experimental" } else { "blocked_source_layout" },
                reason: if is_pipeline { "Separate CUDA checkout and model-specific quantization and low-bit loader tests required" }
                else { "Requires an eligible complete Diffusers pipeline, not a bare weight file" },
                next_command: "diffusion-plan",
            },
        ));
        options.push(candidate(
            Candidate {
                id: "diffusion_native_packed",
                family: "ojon_native",
                method: "groupwise scalar 4-bit weight packing",
                format: ".ojonq (storage only)",
                loader: "NONE: low-bit inference kernel not implemented",
                state: if is_weights { "storage_only" } else { "blocked_source_layout" },
                reason: if is_weights { "Native packing is implemented but unpacking restores standard-sized safetensors; this is not a GPU memory reduction" }
                else { "Native packing accepts a standalone original safetensors weight file, not a Diffusers folder" },
                next_command: "pack-plan",
            },
        ));
        options.push(candidate(
            Candidate {
                id: "diffusion_gguf_sdcli",
                family: "stable-diffusion.cpp",
                method: "model-specific image/video denoiser conversion via sd-cli",
                format: "GGUF",
                loader: "stable-diffusion.cpp image/video runtime",
                state: if is_weights { "requires_compatibility_test" } else { "blocked_source_layout" },
                reason: if is_weights {
                "Standalone safetensors weights can be planned and converted. Exact sd-cli architecture support, text encoder/VAE, runtime load and visual parity still require verification"
                } else {
                "Generic Diffusers folders are not accepted by this standalone GGUF route; select a valid standalone denoiser safetensors file"
                },
                next_command: "diffusion-gguf-plan /path/denoiser.safetensors /path/denoiser-Q4.gguf --kind image|video --format q4_0",
            },
        ));
    } else if probe.format == "ojonq_packed_weights" {
        options.push(candidate(
            Candidate {
                id: "ojon_unpack",
                family: "ojon_native",
                method: "unpack storage container",
                format: "safetensors (lossily reconstructed)",
                loader: "ordinary inference loader after unpack",
                state: "storage_only",
                reason: "Unpack does not preserve the original model precision or achieve low-bit VRAM usage",
                next_command: "unpack",
            },
        ));
    }
    // Other modalities are deliberately not advertised as compatible.
    let warning = if options.is_empty() {
        "No implemented candidate family applies to this source. Unsupported architectures remain unsupported."
    } else {
        "All routes are proposals pending exact model/backend/runtime checks. This command neither quantizes nor benchmarks; no winner is selected."
    };
    Ok(json!({
        "schema": "ojon.optimize_options.v1",
        "mode": "read_only_multi_route_comparison",
        "model": probe.as_json(),
        "requested_device": device,
        "device_detected": false,
        "free_ram_bytes": null,
        "free_vram_bytes": null,
        "free_scratch_disk_bytes": null,
        "weights_loaded": false,
        "model_downloaded": false,
        "files_written": false,
        "alternatives_retained": true,
        "options": options,
        "comparison_requirements": [
            "actual_output_bundle_bytes",
            "verified_loader_and_device",
            "measured_peak_ram_and_vram",
            "measured_latency_or_throughput",
            "held_out_task_quality",
            "same_model_revision_and_protocol"
        ],
        "recommendation": null,
        "warning": warning,
    }))
}

/// Metadata-only comparison of currently known routes. No converter is called.
pub fn plan_options(probe: &ModelProbe, device: &str, tools: &[FoundBackend]) -> io::Result<Value> {
    let gguf = if probe.format == "gguf" {
        let summary = inspect_gguf(&probe.path)?;
        Some((summary.file_type, summary.split_count.unwrap_or(1)))
    } else {
        None
    };
    plan_for(probe, device, tools, gguf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture(format: &'static str, category: ModelCategory) -> ModelProbe {
        ModelProbe {
            path: PathBuf::from("not-accessed-in-unit-test"),
            format,
            category,
            identification: "fixture".to_owned(),
            confidence: "fixture",
            bytes: 4096,
            tensor_count: None,
            conversion_route: "none",
            conversion_status: "candidate_only",
            notes: vec![],
        }
    }

    #[test]
    fn hf_keeps_three_routes_without_inventing_a_winner() {
        let value = plan_for(
            &fixture("hf_checkpoint_directory", ModelCategory::Language),
            "xpu",
            &[],
            None,
        )
        .unwrap();
        let rows = value["options"].as_array().unwrap();
        assert_eq!(rows.len(), 7);
        assert_eq!(rows[0]["id"], "hf_gguf_q5");
        assert_eq!(rows[1]["id"], "hf_gguf_q4");
        assert_eq!(rows[2]["id"], "hf_autoround_4");
        assert_eq!(rows[3]["id"], "hf_exl3");
        assert_eq!(rows[4]["id"], "hf_hqq_q4");
        assert_eq!(rows[5]["id"], "hf_openvino_int4");
        assert_eq!(rows[6]["id"], "hf_openvino_mixed");
        assert_eq!(rows[3]["runtime_load_verified"], false);
        assert!(value["recommendation"].is_null());
        assert_eq!(value["weights_loaded"], false);
        assert_eq!(value["files_written"], false);
    }

    #[test]
    fn gguf_rejects_requantized_source_and_preserves_three_choices() {
        let p = fixture("gguf", ModelCategory::Language);
        let blocked = plan_for(&p, "auto", &[], Some((Some(2), 1))).unwrap();
        let rows = blocked["options"].as_array().unwrap();
        assert_eq!(rows.len(), 3);
        assert!(rows.iter().all(|r| r["state"] == "blocked_source"));
        let original = plan_for(&p, "cpu", &[], Some((Some(1), 1))).unwrap();
        assert!(original["options"][2]["calibration"].as_str().is_some());
        assert!(original["options"][0]["runtime_load_verified"] == false);
    }

    #[test]
    fn diffusion_storage_does_not_claim_gpu_inference() {
        let planned = plan_for(
            &fixture("safetensors_single_file", ModelCategory::Diffusion),
            "cuda",
            &[],
            None,
        )
        .unwrap();
        assert_eq!(planned["options"].as_array().unwrap().len(), 3);
        assert_eq!(planned["options"][1]["state"], "storage_only");
        assert_eq!(planned["options"][1]["runtime_load_verified"], false);
        assert_eq!(
            planned["options"][2]["state"],
            "requires_compatibility_test"
        );
        assert_eq!(planned["options"][2]["id"], "diffusion_gguf_sdcli");
        assert_eq!(planned["options"][2]["runtime_load_verified"], false);
    }

    #[test]
    fn packed_diffusion_weights_only_offer_unpack_not_fake_loaders() {
        let v = plan_for(
            &fixture("ojonq_packed_weights", ModelCategory::Diffusion),
            "cpu",
            &[],
            None,
        )
        .unwrap();
        assert_eq!(v["options"].as_array().unwrap().len(), 1);
        assert_eq!(v["options"][0]["id"], "ojon_unpack");
        assert_eq!(v["options"][0]["state"], "storage_only");
    }

    #[test]
    fn unsupported_models_and_invalid_devices_do_not_fabricate_routes() {
        let p = fixture("hf_checkpoint_directory", ModelCategory::Speech);
        assert!(plan_for(&p, "invalid", &[], None).is_err());
        let v = plan_for(&p, "cpu", &[], None).unwrap();
        assert!(v["options"].as_array().unwrap().is_empty());
        assert!(v["recommendation"].is_null());
    }
}
