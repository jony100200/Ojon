//! Read-only, multi-route quantization comparison planning.
//! Model classification is advisory; finding an executable does not prove
//! model family compatibility, GPU kernel availability, or inference quality.
use crate::discovery::{BackendId, FoundBackend};
use crate::gguf::inspect_gguf;
use crate::probe::{ModelCategory, ModelProbe};
use serde_json::{json, Value};
use std::io::{self, ErrorKind};

fn candidate(
    id: &str, family: &str, method: &str, format: &str,
    loader: &str, state: &str, reason: &str, next_command: &str,
) -> Value {
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
    probe: &ModelProbe, device: &str, tools: &[FoundBackend],
    gguf: Option<(Option<u32>, u16)>,
) -> io::Result<Value> {
    if !matches!(device, "auto" | "cpu" | "cuda" | "xpu") {
        return Err(io::Error::new(ErrorKind::InvalidInput, "device must be auto, cpu, cuda, or xpu"));
    }
    let mut options: Vec<Value> = Vec::new();
    let quantizer = found(tools, BackendId::LlamaQuantize);
    let converter = found(tools, BackendId::HfConverter);
    let autoround = found(tools, BackendId::IntelAutoRound);
    let text_like = matches!(probe.category, ModelCategory::Language | ModelCategory::VisionLanguage);
    let multi = probe.category == ModelCategory::VisionLanguage;

    if probe.format == "gguf" && text_like {
        let original = matches!(gguf, Some((Some(0 | 1 | 32), 1)));
        let state = if !original { "blocked_source" }
                    else if !quantizer { "missing_tool" }
                    else { "requires_compatibility_test" };
        let reason = if !original {
            "A comparison sweep requires original F32/F16/BF16 single-part GGUF; avoid requantizing an already compressed file"
        } else if !quantizer {
            "llama-quantize not detected; install or explicitly configure the official executable"
        } else {
            "Quantizer detected; model-family support, target runtime, RAM/VRAM, and output quality are not verified"
        };
        for (id, method, format, command) in [
            ("gguf_q5", "K-quant quality-oriented", "Q5_K_M", "sweep-plan"),
            ("gguf_q4", "K-quant balanced", "Q4_K_M", "sweep-plan"),
            ("gguf_iq4", "importance-calibrated low-bit", "IQ4_XS", "imatrix-plan / sweep-plan"),
        ] {
            let mut item = candidate(id, "llama.cpp", method, format, "llama.cpp (runtime unverified)", state, reason, command);
            if multi {
                item["multimodal_requirement"] = json!("Vision encoder, processor and projector require separate validation");
            }
            if id == "gguf_iq4" {
                item["calibration"] = json!("Representative importance matrix recommended; use a separate held-out quality corpus");
            }
            options.push(item);
        }
    } else if probe.format == "hf_checkpoint_directory" && text_like {
        for (id, format) in [("hf_gguf_q5", "Q5_K_M"), ("hf_gguf_q4", "Q4_K_M")] {
            let state = if converter && quantizer { "requires_compatibility_test" } else { "missing_tool" };
            let reason = if converter && quantizer {
                "HF converter and quantizer detected; architecture, dependencies, source shards, scratch, target loader and quality remain unverified"
            } else {
                "Requires an installed convert_hf_to_gguf.py and llama-quantize; detected filenames are not capability proof"
            };
            options.push(candidate(id, "llama.cpp", "HF -> high-precision GGUF -> K-quant", format,
                "llama.cpp (runtime unverified)", state, reason, "quantize-hf"));
        }
        let ar_reason = if autoround {
            "AutoRound CLI detected, but matching Python package, device kernels, GGUF export and loader must be verified"
        } else {
            "AutoRound CLI not detected; Python module, device kernels, output and inference loader must be checked separately"
        };
        options.push(candidate("hf_autoround_4", "intel_auto_round",
            "optimized RTN or calibrated AutoRound (select --iters and local --dataset)",
            "GGUF Q4_K_M or Q5_K_M (experimental upstream export)",
            "compatible GGUF runtime, not yet verified",
            if autoround { "requires_compatibility_test" } else { "missing_tool" },
            ar_reason, "autoround-plan --bits 4|5 --format gguf:q4_k_m|gguf:q5_k_m"));
        if multi {
            for item in &mut options {
                item["multimodal_requirement"] = json!("Preserve and verify vision encoder, processor, projector, and actual image-input inference");
            }
        }
    } else if probe.category == ModelCategory::Diffusion && probe.format != "ojonq_packed_weights" {
        let is_pipeline = probe.format == "diffusers_pipeline_directory";
        let is_weights = probe.format == "safetensors";
        options.push(candidate("diffusion_svdq", "diffuse-compressor",
            "SVDQuant transformer INT4", "SVDQ INT4",
            "specific supported Diffusers/CUDA runtime", if is_pipeline { "experimental" } else { "blocked_source_layout" },
            if is_pipeline { "Separate CUDA checkout and model-specific quantization and low-bit loader tests required" }
            else { "Requires an eligible complete Diffusers pipeline, not a bare weight file" },
            "diffusion-plan"));
        options.push(candidate("diffusion_native_packed", "ojon_native",
            "groupwise scalar 4-bit weight packing", ".ojonq (storage only)",
            "NONE: low-bit inference kernel not implemented", if is_weights { "storage_only" } else { "blocked_source_layout" },
            if is_weights { "Native packing is implemented but unpacking restores standard-sized safetensors; this is not a GPU memory reduction" }
            else { "Native packing accepts a standalone original safetensors weight file, not a Diffusers folder" },
            "pack-plan"));
        options.push(candidate("diffusion_gguf_research", "ComfyUI-GGUF",
            "model-specific diffusion GGUF export", "GGUF (unverified)",
            "ComfyUI-GGUF with model-specific loader", "not_wired",
            "No generic diffusion GGUF conversion adapter in Ojon; architecture and runtime support must be proven",
            "none"));
    } else if probe.format == "ojonq_packed_weights" {
        options.push(candidate("ojon_unpack", "ojon_native", "unpack storage container",
            "safetensors (lossily reconstructed)", "ordinary inference loader after unpack",
            "storage_only", "Unpack does not preserve the original model precision or achieve low-bit VRAM usage", "unpack"));
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
    } else { None };
    plan_for(probe, device, tools, gguf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture(format: &'static str, category: ModelCategory) -> ModelProbe {
        ModelProbe {
            path: PathBuf::from("not-accessed-in-unit-test"),
            format, category, identification: "fixture".to_owned(),
            confidence: "fixture", bytes: 4096, tensor_count: None,
            conversion_route: "none", conversion_status: "candidate_only",
            notes: vec![],
        }
    }

    #[test]
    fn hf_keeps_three_routes_without_inventing_a_winner() {
        let value = plan_for(&fixture("hf_checkpoint_directory", ModelCategory::Language),
            "xpu", &[], None).unwrap();
        let rows = value["options"].as_array().unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0]["id"], "hf_gguf_q5");
        assert_eq!(rows[1]["id"], "hf_gguf_q4");
        assert_eq!(rows[2]["id"], "hf_autoround_4");
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
        assert_eq!(original["options"][2]["calibration"].as_str().is_some(), true);
        assert!(original["options"][0]["runtime_load_verified"] == false);
    }

    #[test]
    fn diffusion_storage_does_not_claim_gpu_inference() {
        let planned = plan_for(&fixture("safetensors", ModelCategory::Diffusion),
            "cuda", &[], None).unwrap();
        assert_eq!(planned["options"].as_array().unwrap().len(), 3);
        assert_eq!(planned["options"][1]["state"], "storage_only");
        assert_eq!(planned["options"][1]["runtime_load_verified"], false);
        assert_eq!(planned["options"][2]["state"], "not_wired");
    }

    #[test]
    fn packed_diffusion_weights_only_offer_unpack_not_fake_loaders() {
        let v = plan_for(&fixture("ojonq_packed_weights", ModelCategory::Diffusion),
            "cpu", &[], None).unwrap();
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
