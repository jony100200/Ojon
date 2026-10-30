//! Multimodal projector (mmproj) GGUF conversion shared by `quantize-hf` and
//! `model-build-run`. The projector is converted by the official llama.cpp
//! `convert_hf_to_gguf.py --mmproj` run; Ojon only builds the arguments,
//! checks the output and records a receipt. No vision inference is run.

use crate::gguf::{inspect_gguf, GgufSummary};
use crate::jobs::{run_with_log, tool_command};
use serde_json::{json, Value};
use std::ffi::OsString;
use std::fmt;
use std::fs::{self, File};
use std::io::{self, ErrorKind, Read};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::{SystemTime, UNIX_EPOCH};

fn invalid(s: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, s.into())
}

/// Projector precisions the upstream converter accepts for `--mmproj`.
/// The projector is small next to the language tower, so lower types than
/// Q8_0 are not offered: the saving is negligible and the vision quality
/// cost is not.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MmprojPrecision {
    #[default]
    F16,
    Bf16,
    Q8_0,
}
impl MmprojPrecision {
    /// Exact `--outtype` token for `convert_hf_to_gguf.py`.
    pub fn outtype(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::Q8_0 => "q8_0",
        }
    }
}
impl fmt::Display for MmprojPrecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.outtype())
    }
}
impl FromStr for MmprojPrecision {
    type Err = String;
    fn from_str(input: &str) -> Result<Self, Self::Err> {
        match input.trim().to_ascii_lowercase().as_str() {
            "f16" => Ok(Self::F16),
            "bf16" => Ok(Self::Bf16),
            "q8_0" => Ok(Self::Q8_0),
            other => Err(format!(
                "unsupported mmproj precision '{other}' (use f16, bf16 or q8_0)"
            )),
        }
    }
}

/// `config.json` says the checkpoint carries a vision tower. Same rule as the
/// model-build and model-edit planners: a `vision_config` block, or a
/// `model_type` naming a VLM (`*_vl`, `*vlm*`, `*vision*`).
pub fn config_has_vision_tower(config: &Value) -> bool {
    let model_type = config
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase();
    config.get("vision_config").is_some()
        || model_type.contains("vlm")
        || model_type.contains("_vl")
        || model_type.contains("vision")
}

/// Reads `<folder>/config.json` (bounded) and applies [`config_has_vision_tower`].
pub fn folder_has_vision_tower(folder: &Path) -> io::Result<bool> {
    let cfg = folder.join("config.json");
    let size = fs::metadata(&cfg)?.len();
    if size == 0 || size > 8 * 1024 * 1024 {
        return Err(invalid("model config.json missing or too large"));
    }
    let mut raw = Vec::with_capacity(size as usize);
    File::open(cfg)?
        .take(8 * 1024 * 1024 + 1)
        .read_to_end(&mut raw)?;
    let metadata: Value = serde_json::from_slice(&raw)
        .map_err(|e| invalid(format!("invalid model config.json: {e}")))?;
    Ok(config_has_vision_tower(&metadata))
}

/// The exact converter argument list, in order, for one projector conversion.
/// Pure: nothing is spawned or checked here.
pub fn mmproj_converter_args(
    converter: &Path,
    source_dir: &Path,
    outfile: &Path,
    precision: MmprojPrecision,
) -> Vec<OsString> {
    vec![
        converter.as_os_str().to_os_string(),
        source_dir.as_os_str().to_os_string(),
        OsString::from("--mmproj"),
        OsString::from("--outfile"),
        outfile.as_os_str().to_os_string(),
        OsString::from("--outtype"),
        OsString::from(precision.outtype()),
    ]
}

/// One converted projector, with the fields its receipt records.
#[derive(Clone, Debug)]
pub struct MmprojResult {
    pub path: PathBuf,
    pub precision: MmprojPrecision,
    pub bytes: u64,
    pub tensor_count: u64,
}
impl MmprojResult {
    /// The `mmproj` receipt object. `vision_inference_test` is always
    /// `NOT_RUN`: conversion never loads the projector into a runtime.
    pub fn as_json(&self) -> Value {
        json!({
            "path": self.path.to_string_lossy(),
            "precision": self.precision.to_string(),
            "bytes": self.bytes,
            "tensors": self.tensor_count,
            "vision_inference_test": "NOT_RUN",
        })
    }
    /// A standalone receipt document for the sidecar next to the projector.
    pub fn receipt_json(&self, source_dir: &Path, converter: &Path) -> Value {
        json!({
            "schema": "ojon.mmproj_receipt.v1",
            "source": source_dir.to_string_lossy(),
            "converter_script": converter.to_string_lossy(),
            "mmproj": self.as_json(),
            "verification": "GGUF_METADATA_AND_TENSOR_CATALOG_ONLY",
        })
    }
}

/// Runs the converter so that the projector lands at `expected`. Certain
/// llama.cpp revisions prepend a second `mmproj-` to a user-supplied output
/// name; both exact variants are detected and the file is moved to `expected`.
/// Does not stage: the caller owns the directory `expected` lives in.
pub fn convert_mmproj_into(
    python: &Path,
    converter: &Path,
    source_dir: &Path,
    expected: &Path,
    precision: MmprojPrecision,
    log: &mut dyn FnMut(&str),
) -> io::Result<GgufSummary> {
    if expected.symlink_metadata().is_ok() {
        return Err(io::Error::new(
            ErrorKind::AlreadyExists,
            "mmproj output exists; never overwrite it",
        ));
    }
    let mut cmd = tool_command(python);
    cmd.args(mmproj_converter_args(
        converter, source_dir, expected, precision,
    ));
    run_with_log(&mut cmd, |line| log(line))?;
    let parent = expected
        .parent()
        .ok_or_else(|| invalid("mmproj output has no parent directory"))?;
    let name = expected
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| invalid("mmproj output name must be UTF-8"))?;
    let alternative = parent.join(format!("mmproj-{name}"));
    let generated = if expected.is_file() {
        expected.to_path_buf()
    } else if alternative.is_file() {
        alternative
    } else {
        return Err(invalid(
            "mmproj converter did not generate expected vision GGUF",
        ));
    };
    let info = inspect_gguf(&generated)?;
    if info.tensor_count == 0 {
        return Err(invalid("vision mmproj GGUF has zero tensors"));
    }
    if generated != expected {
        fs::rename(&generated, expected)?;
    }
    Ok(info)
}

struct Stage(PathBuf);
impl Drop for Stage {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn create_stage(parent: &Path) -> io::Result<Stage> {
    for retry in 0..8 {
        let nano = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| invalid("invalid clock"))?
            .as_nanos();
        let path = parent.join(format!(
            ".ojon-mmproj-{}-{nano}-{retry}",
            std::process::id()
        ));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(Stage(path)),
            Err(e) if e.kind() == ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    Err(invalid("unable to create mmproj staging directory"))
}

/// Validates a user-chosen projector destination: `.gguf`, parent exists,
/// nothing there yet, outside the source folder. Returns the absolute path.
pub fn resolve_mmproj_output(output: &Path, source_dir: &Path) -> io::Result<PathBuf> {
    if !output
        .extension()
        .and_then(|s| s.to_str())
        .is_some_and(|s| s.eq_ignore_ascii_case("gguf"))
    {
        return Err(invalid("mmproj output must end in .gguf"));
    }
    let name = output
        .file_name()
        .ok_or_else(|| invalid("missing mmproj output file name"))?;
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
        .canonicalize()?;
    let output = parent.join(name);
    if output.symlink_metadata().is_ok() {
        return Err(io::Error::new(
            ErrorKind::AlreadyExists,
            "mmproj output exists; never overwrite it",
        ));
    }
    if output.starts_with(source_dir) {
        return Err(invalid(
            "mmproj output must be outside the source model directory",
        ));
    }
    Ok(output)
}

/// Staged projector conversion to a resolved destination (see
/// [`resolve_mmproj_output`]): converts into a temporary sibling directory,
/// checks the GGUF, then moves it into place and writes
/// `<output>.ojon-receipt.json`. The destination is never overwritten.
pub fn convert_mmproj(
    python: &Path,
    converter: &Path,
    source_dir: &Path,
    output: &Path,
    precision: MmprojPrecision,
    log: &mut dyn FnMut(&str),
) -> io::Result<MmprojResult> {
    if output.symlink_metadata().is_ok() {
        return Err(io::Error::new(
            ErrorKind::AlreadyExists,
            "mmproj output exists; never overwrite it",
        ));
    }
    let parent = output
        .parent()
        .ok_or_else(|| invalid("mmproj output has no parent directory"))?;
    let stage = create_stage(parent)?;
    let staged = stage.0.join("mmproj.gguf");
    log(&format!(
        "[ojon] converting VLM projector ({precision}, separate from the language GGUF)"
    ));
    let info = convert_mmproj_into(python, converter, source_dir, &staged, precision, log)?;
    if output.symlink_metadata().is_ok() {
        return Err(io::Error::new(
            ErrorKind::AlreadyExists,
            "mmproj output appeared during conversion",
        ));
    }
    fs::rename(&staged, output)?;
    let result = MmprojResult {
        path: output.to_path_buf(),
        precision,
        bytes: info.file_bytes,
        tensor_count: info.tensor_count,
    };
    let mut receipt_name = output.as_os_str().to_os_string();
    receipt_name.push(".ojon-receipt.json");
    let receipt = result.receipt_json(source_dir, converter);
    if let Ok(mut file) = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(PathBuf::from(receipt_name))
    {
        use std::io::Write;
        let _ = serde_json::to_string_pretty(&receipt)
            .map_err(io::Error::other)
            .and_then(|text| file.write_all(text.as_bytes()));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precision_parses_the_three_converter_outtypes_only() {
        assert_eq!("f16".parse::<MmprojPrecision>(), Ok(MmprojPrecision::F16));
        assert_eq!("BF16".parse::<MmprojPrecision>(), Ok(MmprojPrecision::Bf16));
        assert_eq!(" q8_0 ".parse::<MmprojPrecision>(), Ok(MmprojPrecision::Q8_0));
        assert!("q4_0".parse::<MmprojPrecision>().is_err());
        assert!("Q4_K_M".parse::<MmprojPrecision>().is_err());
        assert!("".parse::<MmprojPrecision>().is_err());
        assert_eq!(MmprojPrecision::default(), MmprojPrecision::F16);
        assert_eq!(MmprojPrecision::Q8_0.to_string(), "q8_0");
    }

    #[test]
    fn converter_args_are_the_upstream_mmproj_invocation() {
        let args = mmproj_converter_args(
            Path::new("convert_hf_to_gguf.py"),
            Path::new("src-model"),
            Path::new("out/mmproj.gguf"),
            MmprojPrecision::Bf16,
        );
        let text: Vec<String> = args
            .iter()
            .map(|a| a.to_string_lossy().to_string())
            .collect();
        assert_eq!(
            text,
            vec![
                "convert_hf_to_gguf.py",
                "src-model",
                "--mmproj",
                "--outfile",
                "out/mmproj.gguf",
                "--outtype",
                "bf16",
            ]
        );
        let default_args = mmproj_converter_args(
            Path::new("c.py"),
            Path::new("m"),
            Path::new("o.gguf"),
            MmprojPrecision::default(),
        );
        assert_eq!(default_args.last().unwrap(), "f16");
    }

    #[test]
    fn receipt_mmproj_object_shape_and_not_run_claim() {
        let r = MmprojResult {
            path: PathBuf::from("out/mmproj-f16.gguf"),
            precision: MmprojPrecision::F16,
            bytes: 4096,
            tensor_count: 12,
        };
        let obj = r.as_json();
        assert_eq!(obj["path"], "out/mmproj-f16.gguf");
        assert_eq!(obj["precision"], "f16");
        assert_eq!(obj["bytes"], 4096);
        assert_eq!(obj["vision_inference_test"], "NOT_RUN");
        let receipt = r.receipt_json(Path::new("src"), Path::new("convert_hf_to_gguf.py"));
        assert_eq!(receipt["schema"], "ojon.mmproj_receipt.v1");
        assert_eq!(receipt["mmproj"]["bytes"], 4096);
        assert_eq!(receipt["mmproj"]["vision_inference_test"], "NOT_RUN");
        assert_eq!(receipt["verification"], "GGUF_METADATA_AND_TENSOR_CATALOG_ONLY");
    }

    #[test]
    fn vision_tower_detection_matches_model_build_rule() {
        assert!(config_has_vision_tower(&json!({"model_type":"llava","vision_config":{}})));
        assert!(config_has_vision_tower(&json!({"model_type":"qwen2_vl"})));
        assert!(config_has_vision_tower(&json!({"model_type":"SmolVLM"})));
        assert!(!config_has_vision_tower(&json!({"model_type":"llama"})));
        assert!(!config_has_vision_tower(&json!({})));
    }

    #[test]
    fn mmproj_output_must_be_gguf_and_new() {
        let dir = std::env::temp_dir().join(format!(
            "ojon-mmproj-test-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        let source = dir.join("src");
        fs::create_dir_all(&source).unwrap();
        let source = source.canonicalize().unwrap();
        assert!(resolve_mmproj_output(&dir.join("mmproj.bin"), &source).is_err());
        assert!(resolve_mmproj_output(&source.join("mmproj.gguf"), &source).is_err());
        let existing = dir.join("taken.gguf");
        fs::write(&existing, b"x").unwrap();
        assert_eq!(
            resolve_mmproj_output(&existing, &source).unwrap_err().kind(),
            ErrorKind::AlreadyExists
        );
        let fresh = resolve_mmproj_output(&dir.join("mmproj.gguf"), &source).unwrap();
        assert!(fresh.is_absolute());
        assert!(!fresh.exists());
        let _ = fs::remove_dir_all(&dir);
    }
}
