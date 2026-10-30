//! Per-user, versioned, portable Ojon preferences.
//! Never write settings into the checkout, application release or model folder.
//! Model measurement receipts and experiment recipes are separate.
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::env;
use std::fs::{self, OpenOptions};
use std::io::{self, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const MAX_SETTINGS_BYTES: usize = 64 * 1024;
const BACKENDS: [(&str, &str); 13] = [
    ("llama_quantize", "OJON_LLAMA_QUANTIZE"),
    ("llama_cli", "OJON_LLAMA_CLI"),
    ("llama_server", "OJON_LLAMA_SERVER"),
    ("llama_bench", "OJON_LLAMA_BENCH"),
    ("llama_perplexity", "OJON_LLAMA_PERPLEXITY"),
    ("llama_imatrix", "OJON_LLAMA_IMATRIX"),
    ("hf_converter", "OJON_CONVERT_HF"),
    ("llama_factory", "OJON_LLAMAFAC_CLI"),
    ("sd_cli", "OJON_SD_CLI"),
    ("whisper_cli", "OJON_WHISPER_CLI"),
    ("llama_cvector", "OJON_LLAMA_CVECTOR"),
    ("autoround", "OJON_AUTOROUND"),
    ("llama_sycl", "OJON_LLAMA_SYCL"),
];

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, message.into())
}
fn valid_text(value: &str) -> bool {
    !value.trim().is_empty() && value.len() <= 4096 && !value.chars().any(char::is_control)
}
fn optional_path(value: &Value, key: &str) -> io::Result<Option<String>> {
    match value.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) if valid_text(s) => Ok(Some(s.clone())),
        _ => Err(invalid(format!("invalid settings field {key}"))),
    }
}
fn string_field<'a>(value: &'a Value, key: &str, default: &'a str) -> io::Result<&'a str> {
    match value.get(key) {
        None => Ok(default),
        Some(Value::String(s)) => Ok(s),
        _ => Err(invalid(format!("invalid settings field {key}"))),
    }
}
fn numeric_field(value: &Value, key: &str, default: f64, max: f64) -> io::Result<f64> {
    let n = match value.get(key) {
        None => default,
        Some(n) => n
            .as_f64()
            .ok_or_else(|| invalid(format!("invalid number {key}")))?,
    };
    if !n.is_finite() || !(0.0..=max).contains(&n) {
        return Err(invalid(format!("{key} is outside 0..={max}")));
    }
    Ok(n)
}

/// The JSON file holds user choices, never secrets, benchmark results or
/// unverified capabilities. Absent fields take safe defaults.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalSettings {
    pub backend_paths: BTreeMap<String, String>,
    pub last_model: Option<String>,
    pub model_directory: Option<String>,
    pub output_directory: Option<String>,
    pub optimization_goal: String,
    pub min_saving_percent: f64,
    pub max_loss_points: f64,
    pub max_ppl_regression_pct: f64,
    /// Optional manual candidate list, checked against current parity adapter.
    pub precision_candidates: Option<String>,
    pub bench_prompt_tokens: u32,
    pub bench_generation_tokens: u32,
    pub bench_repetitions: u32,
    /// Fraction of fastest measured decode throughput accepted by Balanced.
    pub balanced_min_speed_ratio: f64,
    pub preferred_device: String,
    pub runtime_device: Option<String>,
    pub gpu_layers: u32,
}
impl Default for LocalSettings {
    fn default() -> Self {
        Self {
            backend_paths: BTreeMap::new(),
            last_model: None,
            model_directory: None,
            output_directory: None,
            optimization_goal: "balanced".into(),
            min_saving_percent: 5.0,
            max_loss_points: 0.0,
            max_ppl_regression_pct: 0.0,
            precision_candidates: None,
            bench_prompt_tokens: 512,
            bench_generation_tokens: 128,
            bench_repetitions: 3,
            balanced_min_speed_ratio: 0.85,
            preferred_device: "auto".into(),
            runtime_device: None,
            gpu_layers: 0,
        }
    }
}
impl LocalSettings {
    pub fn backend(&self, environment_key: &str) -> Option<&str> {
        self.backend_paths.get(environment_key).map(String::as_str)
    }
    pub fn as_json(&self) -> Value {
        json!({"schema_version":1,"backend_paths":self.backend_paths,
            "last_model":self.last_model,"model_directory":self.model_directory,
            "output_directory":self.output_directory,
            "optimization":{"goal":self.optimization_goal,
                "min_saving_percent":self.min_saving_percent,
                "max_loss_points":self.max_loss_points,
                "max_ppl_regression_pct":self.max_ppl_regression_pct,
                "precision_candidates":self.precision_candidates,
                "bench_prompt_tokens":self.bench_prompt_tokens,
                "bench_generation_tokens":self.bench_generation_tokens,
                "bench_repetitions":self.bench_repetitions,
                "balanced_min_speed_ratio":self.balanced_min_speed_ratio},
            "hardware":{"preferred_device":self.preferred_device,
                "runtime_device":self.runtime_device,"gpu_layers":self.gpu_layers}})
    }
    pub fn from_json(value: &Value) -> io::Result<Self> {
        let obj = value
            .as_object()
            .ok_or_else(|| invalid("settings must be a JSON object"))?;
        for key in obj.keys() {
            if ![
                "schema_version",
                "backend_paths",
                "last_model",
                "model_directory",
                "output_directory",
                "optimization",
                "hardware",
            ]
            .contains(&key.as_str())
            {
                return Err(invalid(format!(
                    "unknown settings field {key}; preserve file for review"
                )));
            }
        }
        if value["schema_version"] != 1 {
            return Err(invalid(
                "unsupported settings schema_version; file not overwritten",
            ));
        }
        let mut result = Self::default();
        if let Some(backends) = value.get("backend_paths") {
            let map = backends
                .as_object()
                .ok_or_else(|| invalid("backend_paths must be an object"))?;
            for (name, path) in map {
                if !BACKENDS.iter().any(|(_, env)| env == name) {
                    return Err(invalid(format!("unknown backend preference: {name}")));
                }
                let path = path
                    .as_str()
                    .filter(|v| valid_text(v))
                    .ok_or_else(|| invalid(format!("invalid backend path: {name}")))?;
                result.backend_paths.insert(name.clone(), path.to_owned());
            }
        }
        result.last_model = optional_path(value, "last_model")?;
        result.model_directory = optional_path(value, "model_directory")?;
        result.output_directory = optional_path(value, "output_directory")?;
        let opt = value.get("optimization").unwrap_or(&Value::Null);
        if !opt.is_null() && !opt.is_object() {
            return Err(invalid("optimization must be an object"));
        }
        if let Some(obj) = opt.as_object() {
            for key in obj.keys() {
                if ![
                    "goal",
                    "min_saving_percent",
                    "max_loss_points",
                    "max_ppl_regression_pct",
                    "precision_candidates",
                    "bench_prompt_tokens",
                    "bench_generation_tokens",
                    "bench_repetitions",
                    "balanced_min_speed_ratio",
                ]
                .contains(&key.as_str())
                {
                    return Err(invalid(format!("unknown optimization setting: {key}")));
                }
            }
        }
        result.optimization_goal = string_field(opt, "goal", "balanced")?.into();
        result.min_saving_percent = numeric_field(opt, "min_saving_percent", 5.0, 95.0)?;
        result.max_loss_points = numeric_field(opt, "max_loss_points", 0.0, 100.0)?;
        result.max_ppl_regression_pct = numeric_field(opt, "max_ppl_regression_pct", 0.0, 100.0)?;
        result.precision_candidates = optional_path(opt, "precision_candidates")?;
        let quantity = |key: &str, default: u32, min: u32, max: u32| -> io::Result<u32> {
            let v = match opt.get(key) {
                None => default,
                Some(value) => value
                    .as_u64()
                    .and_then(|n| u32::try_from(n).ok())
                    .ok_or_else(|| invalid(format!("{key} must be an integer")))?,
            };
            if !(min..=max).contains(&v) {
                return Err(invalid(format!("{key} must be {min}..={max}")));
            }
            Ok(v)
        };
        result.bench_prompt_tokens = quantity("bench_prompt_tokens", 512, 16, 8192)?;
        result.bench_generation_tokens = quantity("bench_generation_tokens", 128, 16, 2048)?;
        result.bench_repetitions = quantity("bench_repetitions", 3, 1, 10)?;
        result.balanced_min_speed_ratio =
            numeric_field(opt, "balanced_min_speed_ratio", 0.85, 1.0)?;
        let hw = value.get("hardware").unwrap_or(&Value::Null);
        if !hw.is_null() && !hw.is_object() {
            return Err(invalid("hardware must be an object"));
        }
        if let Some(obj) = hw.as_object() {
            for key in obj.keys() {
                if !["preferred_device", "runtime_device", "gpu_layers"].contains(&key.as_str()) {
                    return Err(invalid(format!("unknown hardware setting: {key}")));
                }
            }
        }
        result.preferred_device = string_field(hw, "preferred_device", "auto")?.into();
        result.runtime_device = optional_path(hw, "runtime_device")?;
        result.gpu_layers = match hw.get("gpu_layers") {
            None => 0,
            Some(v) => v
                .as_u64()
                .filter(|v| *v <= 512)
                .ok_or_else(|| invalid("gpu_layers must be 0..512"))? as u32,
        };
        result.validate()?;
        Ok(result)
    }
    pub fn validate(&self) -> io::Result<()> {
        if !matches!(
            self.optimization_goal.as_str(),
            "smallest" | "balanced" | "fastest" | "maximum_quality"
        ) {
            return Err(invalid(
                "goal must be smallest/balanced/maximum_quality/fastest",
            ));
        }
        if !matches!(
            self.preferred_device.as_str(),
            "auto" | "cpu" | "cuda" | "xpu" | "vulkan"
        ) {
            return Err(invalid("device must be auto/cpu/cuda/xpu/vulkan"));
        }
        if !self.min_saving_percent.is_finite()
            || !(0.0..=95.0).contains(&self.min_saving_percent)
            || !self.max_loss_points.is_finite()
            || !(0.0..=100.0).contains(&self.max_loss_points)
            || !self.max_ppl_regression_pct.is_finite()
            || !(0.0..=100.0).contains(&self.max_ppl_regression_pct)
            || self.gpu_layers > 512
            || !(16..=8192).contains(&self.bench_prompt_tokens)
            || !(16..=2048).contains(&self.bench_generation_tokens)
            || !(1..=10).contains(&self.bench_repetitions)
            || !self.balanced_min_speed_ratio.is_finite()
            || !(0.50..=1.0).contains(&self.balanced_min_speed_ratio)
        {
            return Err(invalid("invalid quality or hardware preference"));
        }
        if let Some(recipe) = self.precision_candidates.as_deref() {
            let parsed = crate::sweep::parse_precisions(recipe)?;
            if parsed.len() > 6
                || parsed
                    .iter()
                    .any(|p| !crate::quant_parity::allowed_parity_precision(*p))
            {
                return Err(invalid(
                    "guided precision list must contain 1..=6 lower-bit Q4 parity candidates",
                ));
            }
        }
        for value in self.backend_paths.values() {
            if !valid_text(value) || !Path::new(value).is_absolute() {
                return Err(invalid(
                    "backend preference requires an absolute executable path",
                ));
            }
        }
        for value in [
            &self.last_model,
            &self.model_directory,
            &self.output_directory,
            &self.runtime_device,
        ] {
            if value.as_deref().is_some_and(|s| !valid_text(s)) {
                return Err(invalid("invalid local path or runtime device"));
            }
        }
        Ok(())
    }
    /// Set a single supported preference. A backend path can be cleared instead
    /// of replacing it with an empty string.
    pub fn set(&mut self, key: &str, value: &str) -> io::Result<()> {
        if let Some(alias) = key.strip_prefix("backend.") {
            let Some((_, env)) = BACKENDS.iter().find(|(name, _)| *name == alias) else {
                return Err(invalid("unknown backend preference"));
            };
            if !valid_text(value) {
                return Err(invalid("backend path must be nonempty"));
            }
            self.backend_paths.insert((*env).into(), value.into());
            return self.validate();
        }
        match key {
            "last_model" => self.last_model = Some(value.into()),
            "model_directory" => self.model_directory = Some(value.into()),
            "output_directory" => self.output_directory = Some(value.into()),
            "optimization.goal" => self.optimization_goal = value.into(),
            "optimization.min_saving_percent" => {
                self.min_saving_percent = value
                    .parse()
                    .map_err(|_| invalid("expected numeric saving percent"))?
            }
            "optimization.max_loss_points" => {
                self.max_loss_points = value
                    .parse()
                    .map_err(|_| invalid("expected numeric quality loss"))?
            }
            "optimization.max_ppl_regression_pct" => {
                self.max_ppl_regression_pct = value
                    .parse()
                    .map_err(|_| invalid("expected numeric PPL regression"))?
            }
            "optimization.precision_candidates" => self.precision_candidates = Some(value.into()),
            "optimization.bench_prompt_tokens" => {
                self.bench_prompt_tokens = value
                    .parse()
                    .map_err(|_| invalid("expected integer prompt tokens"))?
            }
            "optimization.bench_generation_tokens" => {
                self.bench_generation_tokens = value
                    .parse()
                    .map_err(|_| invalid("expected integer decode tokens"))?
            }
            "optimization.bench_repetitions" => {
                self.bench_repetitions = value
                    .parse()
                    .map_err(|_| invalid("expected integer benchmark repetitions"))?
            }
            "optimization.balanced_min_speed_ratio" => {
                self.balanced_min_speed_ratio = value
                    .parse()
                    .map_err(|_| invalid("expected numeric speed ratio"))?
            }
            "hardware.preferred_device" => self.preferred_device = value.into(),
            "hardware.runtime_device" => self.runtime_device = Some(value.into()),
            "hardware.gpu_layers" => {
                self.gpu_layers = value
                    .parse()
                    .map_err(|_| invalid("expected integer GPU layers"))?
            }
            _ => return Err(invalid("unknown setting")),
        }
        self.validate()
    }
    pub fn clear(&mut self, key: &str) -> io::Result<()> {
        if let Some(alias) = key.strip_prefix("backend.") {
            let Some((_, env)) = BACKENDS.iter().find(|(name, _)| *name == alias) else {
                return Err(invalid("unknown backend setting"));
            };
            self.backend_paths.remove(*env);
            return Ok(());
        }
        match key {
            "last_model" => self.last_model = None,
            "model_directory" => self.model_directory = None,
            "output_directory" => self.output_directory = None,
            "hardware.runtime_device" => self.runtime_device = None,
            "hardware.gpu_layers" => self.gpu_layers = 0,
            "hardware.preferred_device" => self.preferred_device = "auto".into(),
            "optimization.goal" => self.optimization_goal = "balanced".into(),
            "optimization.min_saving_percent" => self.min_saving_percent = 5.0,
            "optimization.max_loss_points" => self.max_loss_points = 0.0,
            "optimization.max_ppl_regression_pct" => self.max_ppl_regression_pct = 0.0,
            "optimization.precision_candidates" => self.precision_candidates = None,
            "optimization.bench_prompt_tokens" => self.bench_prompt_tokens = 512,
            "optimization.bench_generation_tokens" => self.bench_generation_tokens = 128,
            "optimization.bench_repetitions" => self.bench_repetitions = 3,
            "optimization.balanced_min_speed_ratio" => self.balanced_min_speed_ratio = 0.85,
            _ => return Err(invalid("unknown setting")),
        }
        Ok(())
    }
}

/// Uses standard per-user application config directories, never the repository.
/// Override exists for portable deployments and isolated automated tests.
pub fn settings_path() -> io::Result<PathBuf> {
    if let Some(path) = env::var_os("OJON_SETTINGS_PATH") {
        let path = PathBuf::from(path);
        if !path.is_absolute() {
            return Err(invalid("OJON_SETTINGS_PATH must be absolute"));
        }
        return Ok(path);
    }
    #[cfg(windows)]
    let path = {
        let base = env::var_os("APPDATA")
            .or_else(|| {
                env::var_os("USERPROFILE").map(|p| {
                    PathBuf::from(p)
                        .join("AppData")
                        .join("Roaming")
                        .into_os_string()
                })
            })
            .ok_or_else(|| invalid("APPDATA/USERPROFILE not set"))?;
        PathBuf::from(base).join("Ojon").join("settings.json")
    };
    #[cfg(target_os = "macos")]
    let path = {
        let home = env::var_os("HOME").ok_or_else(|| invalid("HOME not set"))?;
        PathBuf::from(home).join("Library/Application Support/Ojon/settings.json")
    };
    #[cfg(not(any(windows, target_os = "macos")))]
    let path = {
        let base = env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .filter(|p| p.is_absolute())
            .or_else(|| env::var_os("HOME").map(|p| PathBuf::from(p).join(".config")))
            .ok_or_else(|| invalid("HOME/XDG_CONFIG_HOME not set"))?;
        base.join("ojon").join("settings.json")
    };
    Ok(path)
}
pub fn load_settings() -> io::Result<LocalSettings> {
    load_from_path(&settings_path()?)
}
pub fn save_settings(settings: &LocalSettings) -> io::Result<PathBuf> {
    let path = settings_path()?;
    save_to_path(&path, settings)?;
    Ok(path)
}
pub fn load_from_path(path: &Path) -> io::Result<LocalSettings> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(LocalSettings::default()),
        Err(e) => return Err(e),
    };
    if bytes.len() > MAX_SETTINGS_BYTES {
        return Err(invalid("settings file exceeds 64 KiB"));
    }
    let value: Value = serde_json::from_slice(&bytes)
        .map_err(|e| invalid(format!("invalid settings JSON: {e}")))?;
    LocalSettings::from_json(&value)
}
#[cfg(windows)]
fn replace_file(tmp: &Path, destination: &Path) -> io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    #[link(name = "kernel32")]
    extern "system" {
        fn MoveFileExW(old: *const u16, new: *const u16, flags: u32) -> i32;
    }
    let a = tmp
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect::<Vec<_>>();
    let b = destination
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect::<Vec<_>>();
    // MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH
    if unsafe { MoveFileExW(a.as_ptr(), b.as_ptr(), 0x1 | 0x8) } == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}
#[cfg(not(windows))]
fn replace_file(tmp: &Path, destination: &Path) -> io::Result<()> {
    fs::rename(tmp, destination)
}
pub fn save_to_path(path: &Path, settings: &LocalSettings) -> io::Result<()> {
    settings.validate()?;
    let payload = serde_json::to_vec_pretty(&settings.as_json()).map_err(io::Error::other)?;
    if payload.len() > MAX_SETTINGS_BYTES {
        return Err(invalid("settings file exceeds 64 KiB"));
    }
    let parent = path
        .parent()
        .ok_or_else(|| invalid("settings path has no parent"))?;
    fs::create_dir_all(parent)?;
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| invalid("system clock not initialized"))?
        .as_nanos();
    let base = path
        .file_name()
        .ok_or_else(|| invalid("settings filename missing"))?
        .to_string_lossy();
    let temp = parent.join(format!(".{base}.{}.{stamp}.tmp", std::process::id()));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(&temp)?;
    let result = (|| {
        file.write_all(&payload)?;
        file.sync_all()?;
        drop(file);
        replace_file(&temp, path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    static TEST_SEQ: AtomicU64 = AtomicU64::new(0);
    fn fixture() -> PathBuf {
        let seq = TEST_SEQ.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir()
            .join(format!(
                "ojon-settings-test-{}-{}-{}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
                seq
            ))
            .join("settings.json")
    }
    #[test]
    fn first_launch_and_atomic_roundtrip() {
        let path = fixture();
        assert_eq!(load_from_path(&path).unwrap(), LocalSettings::default());
        let mut settings = LocalSettings::default();
        settings
            .set(
                "backend.llama_cli",
                &std::env::temp_dir()
                    .join("ojon-cli-test.exe")
                    .to_string_lossy(),
            )
            .unwrap();
        settings.set("last_model", "models/model.gguf").unwrap();
        settings.set("optimization.goal", "smallest").unwrap();
        save_to_path(&path, &settings).unwrap();
        assert_eq!(load_from_path(&path).unwrap(), settings);
        settings.set("hardware.gpu_layers", "12").unwrap();
        save_to_path(&path, &settings).unwrap();
        assert_eq!(load_from_path(&path).unwrap(), settings);
        let _ = fs::remove_dir_all(path.parent().unwrap());
    }
    #[test]
    fn user_defined_search_and_bench_parameters_round_trip() {
        let mut settings = LocalSettings::default();
        settings
            .set("optimization.precision_candidates", "IQ3_S,Q3_K_M")
            .unwrap();
        settings
            .set("optimization.bench_prompt_tokens", "1024")
            .unwrap();
        settings
            .set("optimization.bench_generation_tokens", "256")
            .unwrap();
        settings.set("optimization.bench_repetitions", "4").unwrap();
        settings
            .set("optimization.balanced_min_speed_ratio", "0.92")
            .unwrap();
        let decoded = LocalSettings::from_json(&settings.as_json()).unwrap();
        assert_eq!(decoded, settings);
        assert!(settings
            .set("optimization.precision_candidates", "Q5_K_M")
            .is_err());
        settings.clear("optimization.precision_candidates").unwrap();
        assert_eq!(settings.precision_candidates, None);
    }
    #[test]
    fn rejects_corrupt_and_future_schema_without_overwrite() {
        let path = fixture();
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, br#"{"schema_version":999}"#).unwrap();
        assert!(load_from_path(&path).is_err());
        let before = fs::read(&path).unwrap();
        assert_eq!(before, fs::read(&path).unwrap());
        fs::write(&path, b"{broken").unwrap();
        assert!(load_from_path(&path).is_err());
        let _ = fs::remove_dir_all(path.parent().unwrap());
    }
    #[test]
    fn rejects_unknown_preferences_and_invalid_quality_policy() {
        let mut settings = LocalSettings::default();
        assert!(settings.set("backend.not_really", "/tmp/a").is_err());
        assert!(settings.set("optimization.goal", "imaginary").is_err());
        assert!(LocalSettings::from_json(&json!({"schema_version":1,
            "backend_paths":{"CUSTOM":"./fake"}}))
        .is_err());
    }
}
