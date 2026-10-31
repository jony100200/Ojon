//! Diffusion precision sweep: the image-model counterpart of `recipe-run`.
//!
//! For each requested GGUF format (largest first) the denoiser is converted
//! with sd-cli, then the embedded ComfyUI parity script generates with the
//! reference and with that candidate on the same fixed-seed workflow and
//! measures PSNR (and the temporal delta for video). The receipt lists every
//! format with its size and worst-frame PSNR and names the smallest format
//! that passed, or none.
//!
//! The output directory must be the directory ComfyUI loads UNet GGUFs from
//! (its `models/unet`), because the loader node refers to files by name. That
//! is stated in the plan and refused if the directory is missing.
use serde_json::{json, Value};
use std::fs;
use std::io;
use std::path::PathBuf;

use super::diffusion_gguf::{prepare_gguf_convert_with, run_gguf_convert, GgufConvertPlan, GgufConverter, InputLayout, MediaKind};

fn bad(s: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, s.into())
}

/// Formats the sd-cli converter accepts, with nominal bits per weight used only
/// to order the sweep (largest first) and to break ties.
pub const SWEEP_FORMATS: [(&str, f64); 5] = [("q8_0", 8.5), ("q5_0", 5.5), ("q4_K", 4.5), ("q4_0", 4.5), ("q3_K", 3.4)];

pub fn nominal_bits(format: &str) -> Option<f64> {
    SWEEP_FORMATS.iter().find(|(f, _)| *f == format).map(|(_, b)| *b)
}

/// Parse and order a comma-separated format list, largest first, no duplicates.
pub fn parse_formats(spec: &str) -> io::Result<Vec<String>> {
    parse_formats_for(spec, &GgufConverter::SdCli)
}

/// Same, against the format vocabulary of the chosen converter.
pub fn parse_formats_for(spec: &str, converter: &GgufConverter) -> io::Result<Vec<String>> {
    let mut out: Vec<String> = Vec::new();
    for raw in spec.split(',') {
        let f = raw.trim();
        if f.is_empty() {
            continue;
        }
        if converter.nominal_bits(f).is_none() {
            return Err(bad(format!(
                "unknown sweep format '{f}' for {}; use any of {}",
                converter.name(),
                converter.formats().iter().map(|(x, _)| *x).collect::<Vec<_>>().join(",")
            )));
        }
        if !out.iter().any(|x| x == f) {
            out.push(f.to_string());
        }
    }
    if out.is_empty() {
        return Err(bad("sweep needs at least one format"));
    }
    out.sort_by(|a, b| converter.nominal_bits(b).partial_cmp(&converter.nominal_bits(a)).unwrap());
    Ok(out)
}

#[derive(Debug, Clone)]
pub struct DiffusionSweepRequest {
    pub source: PathBuf,
    pub comfy_unet_dir: PathBuf,
    pub tool: PathBuf,
    pub kind: MediaKind,
    pub layout: InputLayout,
    pub formats: Vec<String>,
    pub workflow: PathBuf,
    pub reference_name: String,
    pub server: String,
    pub python: PathBuf,
    pub ffmpeg: Option<PathBuf>,
    pub min_psnr: f64,
    pub max_temporal_delta: Option<f64>,
    pub node_id: Option<String>,
    pub timeout_s: u32,
    /// sd-cli or the ComfyUI-GGUF converter; decides the format vocabulary too.
    pub converter: GgufConverter,
}

#[derive(Debug)]
pub struct DiffusionSweepPlan {
    pub request: DiffusionSweepRequest,
    pub conversions: Vec<GgufConvertPlan>,
    pub receipt_path: PathBuf,
}

pub fn prepare_diffusion_sweep(request: DiffusionSweepRequest) -> io::Result<DiffusionSweepPlan> {
    if request.formats.is_empty() {
        return Err(bad("no formats"));
    }
    if !request.comfy_unet_dir.is_dir() {
        return Err(bad("the output directory must exist and be ComfyUI's UNet model directory (models/unet)"));
    }
    if !request.workflow.is_file() {
        return Err(bad("workflow JSON not found"));
    }
    if request.reference_name.trim().is_empty() {
        return Err(bad("reference name (as the loader lists it) is required"));
    }
    if !(request.min_psnr.is_finite() && (0.0..=100.0).contains(&request.min_psnr)) {
        return Err(bad("min PSNR must be 0..=100 dB"));
    }
    if let Some(t) = request.max_temporal_delta {
        if !(t.is_finite() && (0.0..=255.0).contains(&t)) {
            return Err(bad("max temporal delta must be 0..=255"));
        }
    }
    let stem = request
        .source
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| bad("source has no file name"))?
        .to_string();
    let receipt_path = request.comfy_unet_dir.join(format!("{stem}.sweep-receipt.json"));
    if receipt_path.exists() {
        return Err(bad(format!("sweep receipt already exists: {}", receipt_path.display())));
    }
    let mut conversions = Vec::new();
    for f in &request.formats {
        let out = request.comfy_unet_dir.join(format!("{stem}-{f}.gguf"));
        conversions.push(prepare_gguf_convert_with(&request.source, &out, &request.tool, request.kind, request.layout, f, request.converter.clone())?);
    }
    Ok(DiffusionSweepPlan { request, conversions, receipt_path })
}

impl DiffusionSweepPlan {
    pub fn plan_json(&self) -> Value {
        json!({
            "schema": "ojon.diffusion_sweep_plan.v1",
            "source": self.request.source.to_string_lossy(),
            "comfy_unet_dir": self.request.comfy_unet_dir.to_string_lossy(),
            "kind": self.request.kind.name(),
            "converter": self.request.converter.name(),
            "formats_largest_first": self.request.formats,
            "candidates": self.conversions.iter().map(|c| json!({"format": c.format, "output": c.output.to_string_lossy()})).collect::<Vec<_>>(),
            "workflow": self.request.workflow.to_string_lossy(),
            "reference_name": self.request.reference_name,
            "min_psnr_db": self.request.min_psnr,
            "max_temporal_delta": self.request.max_temporal_delta,
            "generations_required": self.request.formats.len() * 2,
            "execution": "plan_only",
            "note": "each format is converted then generated once beside the reference on the same seed; the smallest format that passes the gate is named, or none"
        })
    }
}

/// The parity script prints one JSON object at the end; tolerate log lines before it.
pub fn parse_parity_stdout(text: &str) -> Option<Value> {
    let start = text.rfind("\n{").map(|i| i + 1).or_else(|| text.find('{'))?;
    serde_json::from_str::<Value>(text[start..].trim()).ok()
}

#[derive(Debug, Clone, PartialEq)]
pub struct SweepRow {
    pub format: String,
    pub bytes: u64,
    pub min_psnr_db: Option<f64>,
    pub temporal_delta: Option<f64>,
    pub passed: bool,
    pub status: String,
}

/// Smallest passing candidate by file size; ties by fewer nominal bits.
pub fn select_smallest_passing(rows: &[SweepRow]) -> Option<&SweepRow> {
    rows.iter()
        .filter(|r| r.passed)
        .min_by(|a, b| a.bytes.cmp(&b.bytes).then_with(|| {
            nominal_bits(&a.format).partial_cmp(&nominal_bits(&b.format)).unwrap_or(std::cmp::Ordering::Equal)
        }))
}

fn run_parity<F>(plan: &DiffusionSweepPlan, candidate_name: &str, log: &mut F) -> io::Result<Value>
where
    F: FnMut(&str),
{
    let r = &plan.request;
    let mut cmd = crate::jobs::tool_command(&r.python);
    cmd.arg("-c")
        .arg(include_str!("../../scripts/comfy_gguf_parity.py"))
        .arg("--workflow")
        .arg(&r.workflow)
        .arg("--reference-name")
        .arg(&r.reference_name)
        .arg("--candidate-name")
        .arg(candidate_name)
        .arg("--server")
        .arg(&r.server)
        .arg("--min-psnr")
        .arg(r.min_psnr.to_string())
        .arg("--kind")
        .arg(r.kind.name())
        .arg("--timeout")
        .arg(r.timeout_s.to_string());
    if let Some(n) = &r.node_id {
        cmd.arg("--node-id").arg(n);
    }
    if let Some(f) = &r.ffmpeg {
        cmd.arg("--ffmpeg").arg(f);
    }
    if let Some(t) = r.max_temporal_delta {
        cmd.arg("--max-temporal-delta").arg(t.to_string());
    }
    let mut stdout = String::new();
    // The parity script exits 2 on a failed gate; that is a result, not an error.
    let result = crate::jobs::run_with_log(&mut cmd, |line| {
        if let Some(rest) = line.strip_prefix("[stdout] ") {
            stdout.push_str(rest);
            stdout.push('\n');
        }
        log(line);
    });
    match parse_parity_stdout(&stdout) {
        Some(v) => Ok(v),
        None => Err(result.err().unwrap_or_else(|| io::Error::other("parity script produced no JSON receipt"))),
    }
}

pub fn run_diffusion_sweep<F>(plan: &DiffusionSweepPlan, mut log: F) -> io::Result<PathBuf>
where
    F: FnMut(&str),
{
    let mut rows: Vec<SweepRow> = Vec::new();
    let mut receipts = Vec::new();
    for conv in &plan.conversions {
        log(&format!("[ojon] sweep: converting {} -> {}", conv.format, conv.output.display()));
        let converted = match run_gguf_convert(conv, |l| log(l)) {
            Ok(v) => v,
            Err(e) => {
                log(&format!("[ojon]   conversion failed: {e}"));
                rows.push(SweepRow { format: conv.format.clone(), bytes: 0, min_psnr_db: None, temporal_delta: None, passed: false, status: format!("conversion_failed: {e}") });
                continue;
            }
        };
        let bytes = converted.get("output_bytes").and_then(Value::as_u64).unwrap_or(0);
        let name = conv.output.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        log(&format!("[ojon] sweep: parity for {name}"));
        match run_parity(plan, &name, &mut log) {
            Ok(v) => {
                let passed = v.get("passed").and_then(Value::as_bool).unwrap_or(false);
                let min_psnr = v.pointer("/summary/min_psnr_db").and_then(Value::as_f64);
                let td = v.pointer("/summary/temporal_delta").and_then(Value::as_f64);
                log(&format!("[ojon]   {}: {} bytes, min PSNR {:?} dB{} -> {}", conv.format, bytes, min_psnr,
                    td.map(|t| format!(", temporal delta {t:.3}")).unwrap_or_default(), if passed { "PASS" } else { "fail" }));
                rows.push(SweepRow { format: conv.format.clone(), bytes, min_psnr_db: min_psnr, temporal_delta: td, passed, status: "measured".into() });
                receipts.push(v);
            }
            Err(e) => {
                log(&format!("[ojon]   parity failed: {e}"));
                rows.push(SweepRow { format: conv.format.clone(), bytes, min_psnr_db: None, temporal_delta: None, passed: false, status: format!("parity_failed: {e}") });
            }
        }
    }
    let winner = select_smallest_passing(&rows).cloned();
    let receipt = json!({
        "schema": "ojon.diffusion_sweep.v1",
        "plan": plan.plan_json(),
        "rows": rows.iter().map(|r| json!({"format": r.format, "bytes": r.bytes, "min_psnr_db": r.min_psnr_db,
                                            "temporal_delta": r.temporal_delta, "passed": r.passed, "status": r.status})).collect::<Vec<_>>(),
        "smallest_passing": winner.as_ref().map(|w| json!({"format": w.format, "bytes": w.bytes, "min_psnr_db": w.min_psnr_db,
                                                             "file": plan.request.comfy_unet_dir.join(format!("{}-{}.gguf",
                                                                 plan.request.source.file_stem().and_then(|s| s.to_str()).unwrap_or("model"), w.format)).to_string_lossy()})),
        "parity_receipts": receipts,
        "claim": if winner.is_some() {
            "smallest format within the PSNR (and temporal) gate on this workflow and seed; not a statement about other prompts"
        } else {
            "no format passed the gate on this workflow and seed; keep the reference weights"
        }
    });
    fs::write(&plan.receipt_path, serde_json::to_vec_pretty(&receipt)?)?;
    Ok(plan.receipt_path.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_are_validated_deduplicated_and_ordered_largest_first() {
        assert_eq!(parse_formats("q4_0,q8_0,q4_0,q3_K").unwrap(), vec!["q8_0", "q4_0", "q3_K"]);
        assert!(parse_formats("q2_K").is_err());
        assert!(parse_formats(" , ").is_err());
    }

    #[test]
    fn smallest_passing_prefers_size_then_fewer_bits() {
        let rows = vec![
            SweepRow { format: "q8_0".into(), bytes: 800, min_psnr_db: Some(40.0), temporal_delta: None, passed: true, status: "measured".into() },
            SweepRow { format: "q5_0".into(), bytes: 500, min_psnr_db: Some(33.0), temporal_delta: None, passed: true, status: "measured".into() },
            SweepRow { format: "q4_K".into(), bytes: 450, min_psnr_db: Some(28.0), temporal_delta: None, passed: false, status: "measured".into() },
            SweepRow { format: "q3_K".into(), bytes: 0, min_psnr_db: None, temporal_delta: None, passed: false, status: "conversion_failed".into() },
        ];
        assert_eq!(select_smallest_passing(&rows).unwrap().format, "q5_0");
        assert!(select_smallest_passing(&rows[2..]).is_none());
    }

    #[test]
    fn parity_stdout_parser_takes_the_last_json_object() {
        let text = "[ojon] something\n{\"ok\": true, \"passed\": false}\n";
        assert_eq!(parse_parity_stdout(text).unwrap()["passed"], false);
        let only = "{\"ok\": true, \"passed\": true, \"summary\": {\"min_psnr_db\": 31.5}}";
        assert_eq!(parse_parity_stdout(only).unwrap().pointer("/summary/min_psnr_db").unwrap(), 31.5);
        assert!(parse_parity_stdout("no json here").is_none());
    }
}
