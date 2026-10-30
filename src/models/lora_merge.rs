//! LoRA-then-quant, step one: merge an adapter into its high-precision original
//! with llama.cpp's `llama-export-lora`, so the recipe search and `auto-quant`
//! then measure a quant of the *improved* original.
//!
//! Upstream facts (`tools/export-lora/export-lora.cpp`, fork main 2026-10-30):
//! `llama-export-lora -m base.gguf --lora adapter.gguf -o merged.gguf`
//! (`--lora-scaled adapter.gguf SCALE` weights the delta). Tensors that receive
//! a delta are written F16 ("output model is F16"); a quantized adapter is
//! refused by the tool ("quantized LoRA adapters is not supported, please retry
//! with f16 or f32"). A PEFT adapter directory (`adapter_config.json` +
//! `adapter_model.safetensors`) is first converted with
//! `convert_lora_to_gguf.py --base <hf-dir> --outtype f16 --outfile <gguf> <dir>`.
//!
//! What this step does not measure: whether the LoRA improved anything. A
//! merged model differs from its base on purpose, so KLD against the base is
//! not a gate here. The receipt names the real gate (`quant-compare-run <base>
//! <suite> <report> --candidate merged=<merged>` on the held-out task suite):
//! quantize the merged original only if it holds or beats the base, or the
//! quant inherits an unmeasured change.
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

use crate::gguf::inspect_gguf;
use crate::jobs::{run_with_log, tool_command};
use crate::recipe_run::plain;

fn invalid(s: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, s.into())
}

pub const LORA_MERGE_SCHEMA: &str = "ojon.lora_merge.v1";
pub const LORA_MERGE_CLAIM: &str = "merge only; nothing here measures whether the adapter improved the model. Gate merged against base on the held-out task suite before quantizing";

/// `general.file_type` values that count as a high-precision original
/// (`llama_ftype`): ALL_F32 = 0, MOSTLY_F16 = 1, MOSTLY_BF16 = 32.
pub fn is_high_precision_file_type(file_type: u32) -> bool {
    matches!(file_type, 0 | 1 | 32)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdapterKind {
    Gguf,
    PeftDir,
}

impl AdapterKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Gguf => "gguf",
            Self::PeftDir => "peft_dir",
        }
    }
}

/// A GGUF adapter starts with the ASCII magic `GGUF`; a PEFT adapter is a
/// directory holding `adapter_config.json`. Anything else is refused here
/// rather than handed to a tool that would fail later with a worse message.
pub fn detect_adapter_kind(path: &Path) -> io::Result<AdapterKind> {
    if path.is_dir() {
        if path.join("adapter_config.json").is_file() {
            return Ok(AdapterKind::PeftDir);
        }
        return Err(invalid(format!(
            "{} is a directory without adapter_config.json; not a PEFT adapter",
            path.display()
        )));
    }
    let mut f = fs::File::open(path).map_err(|e| invalid(format!("adapter {}: {e}", path.display())))?;
    let mut magic = [0u8; 4];
    let n = f.read(&mut magic)?;
    if n == 4 && &magic == b"GGUF" {
        return Ok(AdapterKind::Gguf);
    }
    Err(invalid(format!(
        "{} is neither a GGUF adapter nor a PEFT adapter directory",
        path.display()
    )))
}

#[derive(Clone, Debug)]
pub struct LoraMergeJob {
    pub base: PathBuf,
    pub adapter: PathBuf,
    pub output: PathBuf,
    pub export_tool: PathBuf,
    /// Multiplies the adapter delta (`--lora-scaled`); None = 1.0 (`--lora`).
    pub scale: Option<f64>,
    /// Only for a PEFT directory: the converter script, the Python that runs
    /// it, and the Hugging Face base folder it reads the architecture from.
    pub convert_script: Option<PathBuf>,
    pub python: PathBuf,
    pub hf_base: Option<PathBuf>,
    /// Held-out task suite the receipt's gate command names.
    pub task_suite: Option<PathBuf>,
}

#[derive(Clone, Debug)]
pub struct LoraMergePlan {
    pub job: LoraMergeJob,
    pub adapter_kind: AdapterKind,
    /// The GGUF adapter the export tool reads: the input itself, or the file
    /// the converter writes beside the output.
    pub adapter_gguf: PathBuf,
    pub base_bytes: u64,
    pub base_file_type: u32,
    pub base_architecture: Option<String>,
}

fn quote(parts: Vec<String>) -> String {
    parts
        .into_iter()
        .map(|p| if p.contains(' ') { format!("\"{p}\"") } else { p })
        .collect::<Vec<_>>()
        .join(" ")
}

impl LoraMergePlan {
    fn output_stem(&self) -> String {
        self.job.output.file_stem().unwrap_or_default().to_string_lossy().into_owned()
    }

    pub fn receipt_path(&self) -> PathBuf {
        self.job.output.with_file_name(format!("{}.lora-merge-receipt.json", self.output_stem()))
    }

    pub fn gate_report_path(&self) -> PathBuf {
        self.job.output.with_file_name(format!("{}.vs-base-report.json", self.output_stem()))
    }

    /// `python convert_lora_to_gguf.py --base <hf> --outtype f16 --outfile <gguf> <dir>`;
    /// None for a GGUF adapter.
    pub fn convert_args(&self) -> Option<Vec<String>> {
        if self.adapter_kind != AdapterKind::PeftDir {
            return None;
        }
        let script = self.job.convert_script.as_ref()?;
        let hf_base = self.job.hf_base.as_ref()?;
        Some(vec![
            plain(script),
            "--base".into(),
            plain(hf_base),
            "--outtype".into(),
            "f16".into(),
            "--outfile".into(),
            plain(&self.adapter_gguf),
            plain(&self.job.adapter),
        ])
    }

    pub fn convert_command_line(&self) -> Option<String> {
        self.convert_args().map(|a| {
            let mut p = vec![plain(&self.job.python)];
            p.extend(a);
            quote(p)
        })
    }

    pub fn export_args(&self) -> Vec<String> {
        let mut a = vec!["-m".to_string(), plain(&self.job.base)];
        match self.job.scale {
            Some(s) => {
                a.push("--lora-scaled".into());
                a.push(plain(&self.adapter_gguf));
                a.push(format!("{s}"));
            }
            None => {
                a.push("--lora".into());
                a.push(plain(&self.adapter_gguf));
            }
        }
        a.push("-o".into());
        a.push(plain(&self.job.output));
        a
    }

    pub fn export_command_line(&self) -> String {
        let mut p = vec![plain(&self.job.export_tool)];
        p.extend(self.export_args());
        quote(p)
    }

    /// The gate: merged against base on the held-out task suite.
    pub fn gate_command(&self) -> String {
        let suite = self
            .job
            .task_suite
            .as_ref()
            .map(|p| plain(p))
            .unwrap_or_else(|| "<held-out-task-suite.json>".to_string());
        quote(vec![
            "ojon".into(),
            "quant-compare-run".into(),
            plain(&self.job.base),
            suite,
            plain(&self.gate_report_path()),
            "--candidate".into(),
            format!("merged={}", plain(&self.job.output)),
        ])
    }

    pub fn quant_command(&self) -> String {
        quote(vec![
            "ojon".into(),
            "auto-quant".into(),
            plain(&self.job.output),
            "<empty-output-dir>".into(),
            "--preset".into(),
            "balanced".into(),
        ])
    }

    pub fn plan_json(&self) -> Value {
        json!({
            "schema": "ojon.lora_merge_plan.v1",
            "base": {
                "path": plain(&self.job.base),
                "bytes": self.base_bytes,
                "file_type": self.base_file_type,
                "architecture": self.base_architecture,
            },
            "adapter": {
                "path": plain(&self.job.adapter),
                "kind": self.adapter_kind.as_str(),
                "gguf": plain(&self.adapter_gguf),
            },
            "scale": self.job.scale,
            "convert_command": self.convert_command_line(),
            "export_command": self.export_command_line(),
            "output": plain(&self.job.output),
            "receipt": plain(&self.receipt_path()),
            "writes": "nothing until lora-merge-run --confirm",
            "after_merge": {
                "gate": self.gate_command(),
                "quantize": self.quant_command(),
                "rule": "quantize the merged file only if the gate report shows it holds or beats the base on the suite",
            },
        })
    }
}

pub fn prepare_lora_merge(job: LoraMergeJob) -> io::Result<LoraMergePlan> {
    if job.export_tool.as_os_str().is_empty() {
        return Err(invalid("llama-export-lora executable missing"));
    }
    let summary = inspect_gguf(&job.base).map_err(|e| invalid(format!("base {}: {e}", job.base.display())))?;
    if summary.split_count.unwrap_or(1) > 1 {
        return Err(invalid("base is a split GGUF; merge needs a single file"));
    }
    let file_type = summary
        .file_type
        .ok_or_else(|| invalid("base GGUF has no general.file_type; cannot confirm it is a high-precision original"))?;
    if !is_high_precision_file_type(file_type) {
        return Err(invalid(format!(
            "base general.file_type {file_type} is a quantized file; llama-export-lora merges into the F16/BF16/F32 original (the quant is made from the merged file, not before it)"
        )));
    }
    let adapter_kind = detect_adapter_kind(&job.adapter)?;
    if job.output.extension().and_then(|e| e.to_str()).map(|e| e.eq_ignore_ascii_case("gguf")) != Some(true) {
        return Err(invalid("output must end in .gguf"));
    }
    if job.output == job.base || job.output == job.adapter {
        return Err(invalid("output must be a new file, not the base or the adapter"));
    }
    let parent = job.output.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or_else(|| Path::new("."));
    if !parent.is_dir() {
        return Err(invalid(format!("output directory {} does not exist", parent.display())));
    }
    let output_stem = job.output.file_stem().unwrap_or_default().to_string_lossy().into_owned();
    let adapter_gguf = match adapter_kind {
        AdapterKind::Gguf => {
            let a = inspect_gguf(&job.adapter).map_err(|e| invalid(format!("adapter {}: {e}", job.adapter.display())))?;
            if let (Some(b), Some(x)) = (summary.architecture.as_deref(), a.architecture.as_deref()) {
                if b != x {
                    return Err(invalid(format!("adapter architecture {x} does not match base architecture {b}")));
                }
            }
            job.adapter.clone()
        }
        AdapterKind::PeftDir => {
            if job.convert_script.is_none() {
                return Err(invalid("a PEFT adapter needs convert_lora_to_gguf.py (--convert-script, or found beside convert_hf_to_gguf.py)"));
            }
            if job.hf_base.is_none() {
                return Err(invalid("a PEFT adapter needs --hf-base <Hugging Face base folder> for the converter"));
            }
            job.output.with_file_name(format!("{output_stem}.adapter-f16.gguf"))
        }
    };
    let plan = LoraMergePlan {
        base_bytes: summary.file_bytes,
        base_file_type: file_type,
        base_architecture: summary.architecture.clone(),
        adapter_kind,
        adapter_gguf,
        job,
    };
    let mut planned = vec![plan.job.output.clone(), plan.receipt_path()];
    if plan.adapter_kind == AdapterKind::PeftDir {
        planned.push(plan.adapter_gguf.clone());
    }
    if let Some(existing) = planned.iter().find(|p| p.exists()) {
        return Err(invalid(format!("{} already exists; Ojon never overwrites", existing.display())));
    }
    Ok(plan)
}

fn write_new(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut f = fs::OpenOptions::new().write(true).create_new(true).open(path)?;
    f.write_all(bytes)
}

fn file_bytes(path: &Path) -> u64 {
    fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

/// Convert (PEFT only), then export. Returns the receipt path. A failed step
/// returns the error; nothing partial is recorded as success.
pub fn run_lora_merge<F>(plan: &LoraMergePlan, mut log: F) -> io::Result<PathBuf>
where
    F: FnMut(&str),
{
    let receipt_path = plan.receipt_path();
    if receipt_path.exists() || plan.job.output.exists() {
        return Err(invalid("merge output or receipt already exists"));
    }
    let started = Instant::now();
    let mut steps = Vec::new();
    if let Some(args) = plan.convert_args() {
        let line = plan.convert_command_line().unwrap_or_default();
        log(&format!("convert: {line}"));
        let t = Instant::now();
        let mut cmd = tool_command(&plan.job.python);
        cmd.args(&args);
        run_with_log(&mut cmd, &mut log)?;
        if !plan.adapter_gguf.is_file() {
            return Err(invalid(format!("converter exited successfully but {} was not written", plan.adapter_gguf.display())));
        }
        steps.push(json!({
            "step": "convert_adapter", "command": line, "seconds": t.elapsed().as_secs_f64(),
            "output": plain(&plan.adapter_gguf), "bytes": file_bytes(&plan.adapter_gguf),
        }));
    }
    let line = plan.export_command_line();
    log(&format!("export: {line}"));
    let t = Instant::now();
    let mut cmd = tool_command(&plan.job.export_tool);
    cmd.args(plan.export_args());
    run_with_log(&mut cmd, &mut log)?;
    if !plan.job.output.is_file() {
        return Err(invalid(format!("llama-export-lora exited successfully but {} was not written", plan.job.output.display())));
    }
    let merged = inspect_gguf(&plan.job.output).map_err(|e| invalid(format!("merged file is not a readable GGUF: {e}")))?;
    steps.push(json!({
        "step": "export_lora", "command": line, "seconds": t.elapsed().as_secs_f64(),
        "output": plain(&plan.job.output), "bytes": merged.file_bytes,
    }));
    let receipt = json!({
        "schema": LORA_MERGE_SCHEMA,
        "created_unix": SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
        "base": {"path": plain(&plan.job.base), "bytes": plan.base_bytes, "file_type": plan.base_file_type, "architecture": plan.base_architecture},
        "adapter": {"path": plain(&plan.job.adapter), "kind": plan.adapter_kind.as_str(), "gguf": plain(&plan.adapter_gguf), "gguf_bytes": file_bytes(&plan.adapter_gguf)},
        "scale": plan.job.scale,
        "merged": {"path": plain(&plan.job.output), "bytes": merged.file_bytes, "file_type": merged.file_type, "tensor_count": merged.tensor_count},
        "steps": steps,
        "seconds": started.elapsed().as_secs_f64(),
        "claim": LORA_MERGE_CLAIM,
        "next_steps": {
            "gate": plan.gate_command(),
            "quantize": plan.quant_command(),
            "rule": "quantize the merged file only if the gate report shows it holds or beats the base on the suite",
        },
    });
    write_new(&receipt_path, serde_json::to_string_pretty(&receipt)?.as_bytes())?;
    Ok(receipt_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_gguf(path: &Path, file_type: u32, arch: &str) {
        use std::io::Seek;
        fn val_string(f: &mut fs::File, s: &str) {
            f.write_all(&(s.len() as u64).to_le_bytes()).unwrap();
            f.write_all(s.as_bytes()).unwrap();
        }
        let mut f = fs::File::create(path).unwrap();
        f.write_all(b"GGUF").unwrap();
        f.write_all(&3u32.to_le_bytes()).unwrap();
        f.write_all(&1u64.to_le_bytes()).unwrap();
        f.write_all(&2u64.to_le_bytes()).unwrap();
        val_string(&mut f, "general.architecture");
        f.write_all(&8u32.to_le_bytes()).unwrap();
        val_string(&mut f, arch);
        val_string(&mut f, "general.file_type");
        f.write_all(&4u32.to_le_bytes()).unwrap();
        f.write_all(&file_type.to_le_bytes()).unwrap();
        val_string(&mut f, "blk.0.attn_output.weight");
        f.write_all(&2u32.to_le_bytes()).unwrap();
        f.write_all(&2u64.to_le_bytes()).unwrap();
        f.write_all(&2u64.to_le_bytes()).unwrap();
        f.write_all(&2u32.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        let pos = f.stream_position().unwrap();
        f.write_all(&vec![0u8; ((32 - pos % 32) % 32) as usize]).unwrap();
        f.write_all(&[0u8; 32]).unwrap();
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let tick = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let d = std::env::temp_dir().join(format!("ojon-lora-{tag}-{}-{tick}", std::process::id()));
        fs::create_dir_all(&d).unwrap();
        d
    }

    fn job(dir: &Path, adapter: &Path) -> LoraMergeJob {
        LoraMergeJob {
            base: dir.join("base-f16.gguf"),
            adapter: adapter.to_path_buf(),
            output: dir.join("merged-f16.gguf"),
            export_tool: PathBuf::from("llama-export-lora"),
            scale: None,
            convert_script: None,
            python: PathBuf::from("python"),
            hf_base: None,
            task_suite: None,
        }
    }

    #[test]
    fn adapter_kind_is_detected_from_magic_or_peft_config() {
        let dir = temp_dir("kind");
        let g = dir.join("a.gguf");
        tiny_gguf(&g, 1, "qwen3");
        assert_eq!(detect_adapter_kind(&g).unwrap(), AdapterKind::Gguf);
        let peft = dir.join("peft");
        fs::create_dir(&peft).unwrap();
        assert!(detect_adapter_kind(&peft).is_err());
        fs::write(peft.join("adapter_config.json"), "{}").unwrap();
        assert_eq!(detect_adapter_kind(&peft).unwrap(), AdapterKind::PeftDir);
        let txt = dir.join("x.safetensors");
        fs::write(&txt, b"not gguf").unwrap();
        assert!(detect_adapter_kind(&txt).is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn gguf_adapter_plan_renders_export_and_gate_commands() {
        let dir = temp_dir("plan");
        tiny_gguf(&dir.join("base-f16.gguf"), 1, "qwen3");
        let adapter = dir.join("adapter.gguf");
        tiny_gguf(&adapter, 1, "qwen3");
        let mut j = job(&dir, &adapter);
        j.task_suite = Some(dir.join("suite.json"));
        let plan = prepare_lora_merge(j).unwrap();
        assert_eq!(plan.adapter_kind, AdapterKind::Gguf);
        assert_eq!(plan.adapter_gguf, adapter);
        assert!(plan.convert_args().is_none());
        let line = plan.export_command_line();
        assert!(line.contains(" --lora ") && line.contains(" -o ") && !line.contains("--lora-scaled"), "{line}");
        assert!(plan.gate_command().contains("quant-compare-run") && plan.gate_command().contains("--candidate merged="));
        assert!(plan.gate_command().contains("suite.json"));
        assert!(plan.receipt_path().ends_with("merged-f16.lora-merge-receipt.json"));
        assert_eq!(plan.plan_json()["adapter"]["kind"], "gguf");
        let mut scaled = job(&dir, &adapter);
        scaled.scale = Some(0.5);
        let p2 = prepare_lora_merge(scaled).unwrap();
        assert!(p2.export_command_line().contains("--lora-scaled"));
        assert!(p2.export_args().contains(&"0.5".to_string()));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn quantized_base_mismatched_arch_and_existing_output_are_refused() {
        let dir = temp_dir("refuse");
        let adapter = dir.join("adapter.gguf");
        tiny_gguf(&adapter, 1, "qwen3");
        tiny_gguf(&dir.join("base-f16.gguf"), 15, "qwen3");
        let e = prepare_lora_merge(job(&dir, &adapter)).unwrap_err().to_string();
        assert!(e.contains("quantized"), "{e}");
        fs::remove_file(dir.join("base-f16.gguf")).unwrap();
        tiny_gguf(&dir.join("base-f16.gguf"), 32, "llama");
        let e = prepare_lora_merge(job(&dir, &adapter)).unwrap_err().to_string();
        assert!(e.contains("architecture"), "{e}");
        fs::remove_file(dir.join("base-f16.gguf")).unwrap();
        tiny_gguf(&dir.join("base-f16.gguf"), 0, "qwen3");
        fs::write(dir.join("merged-f16.gguf"), b"x").unwrap();
        let e = prepare_lora_merge(job(&dir, &adapter)).unwrap_err().to_string();
        assert!(e.contains("never overwrites"), "{e}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn peft_adapter_needs_converter_and_hf_base_then_converts_to_f16_beside_output() {
        let dir = temp_dir("peft");
        tiny_gguf(&dir.join("base-f16.gguf"), 1, "qwen3");
        let peft = dir.join("adapter_out");
        fs::create_dir(&peft).unwrap();
        fs::write(peft.join("adapter_config.json"), "{}").unwrap();
        let e = prepare_lora_merge(job(&dir, &peft)).unwrap_err().to_string();
        assert!(e.contains("convert_lora_to_gguf.py"), "{e}");
        let mut j = job(&dir, &peft);
        j.convert_script = Some(dir.join("convert_lora_to_gguf.py"));
        let e = prepare_lora_merge(j.clone()).unwrap_err().to_string();
        assert!(e.contains("--hf-base"), "{e}");
        j.hf_base = Some(dir.join("hf-base"));
        let plan = prepare_lora_merge(j).unwrap();
        assert_eq!(plan.adapter_kind, AdapterKind::PeftDir);
        assert!(plan.adapter_gguf.ends_with("merged-f16.adapter-f16.gguf"));
        let args = plan.convert_args().unwrap();
        assert_eq!(args[args.len() - 1], plain(&peft));
        assert!(args.windows(2).any(|w| w[0] == "--outtype" && w[1] == "f16"));
        assert!(plan.convert_command_line().unwrap().starts_with("python "));
        let _ = fs::remove_dir_all(&dir);
    }
}
