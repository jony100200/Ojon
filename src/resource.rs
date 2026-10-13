//! Conservative hardware preflight for quantization jobs.
//! These are current host observations and safety budgets, NOT peak-use estimates.
//! No model weights are loaded; Intel VRAM is measured via optional xpu-smi
//! only when the returned telemetry is valid, otherwise UNKNOWN.
use crate::probe::probe_model;
use serde_json::{json, Value};
#[cfg(target_os = "linux")]
use std::fs;
use std::io::{self, ErrorKind};
use std::path::Path;
use std::process::Command;

pub const GIB: u64 = 1024 * 1024 * 1024;
const RAM_RESERVE_BYTES: u64 = 4 * GIB;
const SCRATCH_RESERVE_BYTES: u64 = 2 * GIB;

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, message.into())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceMode {
    Hqq, OpenVino, AutoRound, Gguf, Pack, Exl3, Heretic,
}
impl ResourceMode {
    pub fn parse(s: &str) -> io::Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "hqq" => Ok(Self::Hqq),
            "openvino" | "nncf" => Ok(Self::OpenVino),
            "autoround" => Ok(Self::AutoRound),
            "gguf" => Ok(Self::Gguf),
            "pack" => Ok(Self::Pack),
            "exl3" => Ok(Self::Exl3),
            "heretic" => Ok(Self::Heretic),
            _ => Err(invalid("mode must be hqq, openvino, autoround, gguf, pack, exl3 or heretic")),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Hqq => "hqq",
            Self::OpenVino => "openvino",
            Self::AutoRound => "autoround",
            Self::Gguf => "gguf",
            Self::Pack => "pack",
            Self::Exl3 => "exl3",
            Self::Heretic => "heretic",
        }
    }
    // A minimum conservative OPERATOR POLICY budget, not estimated peak memory.
    fn memory_budget_multiplier(self) -> Option<u64> {
        match self {
            Self::Hqq | Self::OpenVino | Self::AutoRound | Self::Exl3 | Self::Heretic => Some(1),
            Self::Gguf => Some(0), // reserve host RAM only; tensor stream peak is unknown.
            Self::Pack => None, // packed pathways may stream tensors.
        }
    }
    fn disk_budget_multiplier(self) -> u64 {
        match self {
            Self::AutoRound | Self::Gguf | Self::Exl3 | Self::Heretic => 2,
            Self::Hqq | Self::OpenVino | Self::Pack => 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ResourceAssessment {
    pub source_bytes: u64,
    pub mode: ResourceMode,
    pub device: String,
    pub available_ram_bytes: Option<u64>,
    pub free_disk_bytes: Option<u64>,
    pub ram_safety_budget_bytes: Option<u64>,
    pub scratch_safety_budget_bytes: u64,
    pub gpu_devices: Vec<Value>,
    pub ram_safe: Option<bool>,
    pub disk_safe: Option<bool>,
    pub device_measured: bool,
    pub output_parent: String,
}
impl ResourceAssessment {
    pub fn status(&self) -> &'static str {
        if self.ram_safe == Some(false) || self.disk_safe == Some(false) {
            "budget_below_policy"
        } else if self.ram_safe.is_none() || self.disk_safe.is_none() || !self.device_measured {
            "unverified"
        } else {
            "passes_minimum_policy_only"
        }
    }
    pub fn as_json(&self) -> Value {
        json!({
            "schema": "ojon.resource_preflight.v1",
            "mode": self.mode.name(),
            "requested_device": self.device,
            "source_weight_bytes": self.source_bytes,
            "output_parent": self.output_parent,
            "free_host_ram_bytes": self.available_ram_bytes,
            "free_output_volume_bytes": self.free_disk_bytes,
            "host_ram_safety_budget_bytes": self.ram_safety_budget_bytes,
            "output_volume_safety_budget_bytes": self.scratch_safety_budget_bytes,
            "reserve_host_ram_bytes": RAM_RESERVE_BYTES,
            "reserve_output_volume_bytes": SCRATCH_RESERVE_BYTES,
            "host_ram_policy_passed": self.ram_safe,
            "output_volume_policy_passed": self.disk_safe,
            "requested_gpu_measured": self.device_measured,
            "gpu_devices": self.gpu_devices,
            "status": self.status(),
            "weights_loaded": false,
            "predicted_peak_ram_bytes": null,
            "predicted_peak_gpu_bytes": null,
            "model_compatibility_verified": false,
            "intel_telemetry_reference": "https://intel.github.io/xpumanager/2.0/xpu-smi/overview.html",
            "notes": [
                "Safety budgets are explicit conservative policy floors (multiples of source weight bytes), NOT predictions of converter peak usage.",
                "Source weight bytes may exclude auxiliary files. Some methods need more than budget. Filesystem capacity can change between check and write.",
                "Intel GPU free VRAM is measured only from a valid xpu-smi single-shot result; missing, N/A or incoherent telemetry stays UNKNOWN.",
                "Passing the policy does not mean this model, converter, quantization kernel or loader will work."
            ]
        })
    }
    /// Block jobs when the conservative RAM/disk floor is known to be violated
    /// or no required telemetry is available. Operators may opt in to risk.
    pub fn enforce(&self, allow_risk: bool) -> io::Result<()> {
        if !allow_risk && self.status() != "passes_minimum_policy_only" {
            return Err(invalid(format!(
                "resource preflight is {}; inspect 'ojon resource-plan' and pass --allow-resource-risk explicitly to run despite uncertainty",
                self.status()
            )));
        }
        Ok(())
    }
}

#[cfg(windows)]
fn available_ram() -> Option<u64> {
    #[repr(C)]
    struct Status {
        size: u32, load: u32, total_phys: u64, avail_phys: u64,
        total_page: u64, avail_page: u64, total_virtual: u64, avail_virtual: u64,
        avail_extended: u64,
    }
    #[link(name="kernel32")]
    extern "system" {
        fn GlobalMemoryStatusEx(status: *mut Status) -> i32;
    }
    let mut status = Status {
        size: std::mem::size_of::<Status>() as u32,
        load: 0, total_phys: 0, avail_phys: 0,
        total_page: 0, avail_page: 0, total_virtual: 0, avail_virtual: 0,
        avail_extended: 0,
    };
    // SAFETY: valid writable C-compatible struct, correct documented length.
    let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
    if ok != 0 { Some(status.avail_phys) } else { None }
}

#[cfg(target_os="linux")]
fn available_ram() -> Option<u64> {
    let data = fs::read_to_string("/proc/meminfo").ok()?;
    data.lines().find_map(|line| {
        let rest = line.strip_prefix("MemAvailable:")?;
        let value: u64 = rest.split_whitespace().next()?.parse().ok()?;
        value.checked_mul(1024)
    })
}
#[cfg(not(any(windows, target_os="linux")))]
fn available_ram() -> Option<u64> { None }

#[cfg(windows)]
fn free_disk(directory: &Path) -> Option<u64> {
    use std::os::windows::ffi::OsStrExt;
    #[link(name="kernel32")]
    extern "system" {
        fn GetDiskFreeSpaceExW(
            directory_name: *const u16,
            free_bytes_available: *mut u64,
            total_bytes: *mut u64,
            total_free_bytes: *mut u64
        ) -> i32;
    }
    let wide: Vec<u16> = directory.as_os_str().encode_wide().chain(Some(0)).collect();
    let mut available = 0u64;
    // SAFETY: NUL-terminated wide path, valid writable u64 pointer;
    // optional unused output pointers are null.
    let ok = unsafe {
        GetDiskFreeSpaceExW(
            wide.as_ptr(), &mut available, std::ptr::null_mut(), std::ptr::null_mut()
        )
    };
    if ok != 0 { Some(available) } else { None }
}

#[cfg(not(windows))]
fn free_disk(directory: &Path) -> Option<u64> {
    // POSIX 'df' requires no Python dependency; safely passes path as an
    // argument (never through a shell). Failure => UNKNOWN, never 0.
    let result = Command::new("df").arg("-Pk").arg(directory).output().ok()?;
    if !result.status.success() { return None; }
    let data = String::from_utf8(result.stdout).ok()?;
    let last = data.lines().last()?;
    let values: Vec<&str> = last.split_whitespace().collect();
    let available_kib: u64 = values.get(3)?.parse().ok()?;
    available_kib.checked_mul(1024)
}

fn nvidia_devices() -> Vec<Value> {
    let mut result = Vec::new();
    let output = match Command::new("nvidia-smi")
        .args(["--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader,nounits"])
        .output() {
            Ok(v) if v.status.success() => v,
            _ => return result,
        };
    let text = match String::from_utf8(output.stdout) {
        Ok(s) => s,
        Err(_) => return result,
    };
    for line in text.lines().take(16) {
        let fields: Vec<&str> = line.splitn(4, ',').map(str::trim).collect();
        if fields.len() != 4 { continue; }
        if let (Ok(index), Ok(total_mib), Ok(free_mib)) =
            (fields[0].parse::<u32>(), fields[2].parse::<u64>(), fields[3].parse::<u64>())
        {
            if let (Some(total), Some(free)) =
                (total_mib.checked_mul(1024 * 1024), free_mib.checked_mul(1024 * 1024))
            {
                result.push(json!({
                    "vendor": "nvidia", "device_index": index, "device_name": fields[1],
                    "total_vram_bytes": total, "free_vram_bytes": free,
                    "telemetry_source": "nvidia-smi",
                }));
            }
        }
    }
    result
}

/// Optional Intel Level Zero telemetry. Missing xpu-smi, N/A and malformed data
/// are UNKNOWN; we never infer free VRAM from the card's rated capacity.
/// API: Intel XPU Manager 2.0 --query-gpu and --format=csv,noheader,nounits.
fn xpu_devices() -> Vec<Value> {
    let output = match Command::new("xpu-smi")
        .args([
            "--query-gpu=index,name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output() {
            Ok(v) if v.status.success() => v,
            _ => return vec![],
        };
    let response = match String::from_utf8(output.stdout) {
        Ok(s) => s,
        Err(_) => return vec![],
    };
    parse_xpu_csv(&response)
}

/// Used as an isolated fixture test, not a claim that the owner's driver works.
fn parse_xpu_csv(text: &str) -> Vec<Value> {
    let mut out = Vec::new();
    for line in text.lines().take(32) {
        let cells: Vec<&str> = line.trim().splitn(4, ',').map(str::trim).collect();
        if cells.len() != 4 { continue; }
        let index: u32 = match cells[0].parse() { Ok(v) => v, Err(_) => continue };
        let to_bytes = |s: &str| -> Option<u64> {
            let mb: f64 = s.parse().ok()?;
            if !mb.is_finite() || mb < 0.0 { return None; }
            let bytes = mb * (1024.0 * 1024.0);
            if bytes > u64::MAX as f64 { return None; }
            Some(bytes.floor() as u64)
        };
        let (Some(total), Some(free)) = (to_bytes(cells[2]), to_bytes(cells[3]))
            else { continue };
        // Incoherent samples (N/A, free > total) are rejected, not trusted.
        if total == 0 || free > total { continue; }
        out.push(json!({
            "vendor": "intel", "device_index": index,
            "device_name": cells[1],
            "total_vram_bytes": total,
            "free_vram_bytes": free,
            "telemetry_source": "xpu-smi --query-gpu",
        }));
    }
    out
}

fn budget(source_bytes: u64, mode: ResourceMode) -> io::Result<(Option<u64>, u64)> {
    if source_bytes == 0 { return Err(invalid("no source weight bytes found")); }
    let ram = mode.memory_budget_multiplier().map(|m|
        source_bytes.checked_mul(m).and_then(|v| v.checked_add(RAM_RESERVE_BYTES))
            .ok_or_else(|| invalid("RAM budget overflow"))
    ).transpose()?;
    let disk = source_bytes.checked_mul(mode.disk_budget_multiplier())
        .and_then(|v| v.checked_add(SCRATCH_RESERVE_BYTES))
        .ok_or_else(|| invalid("scratch budget overflow"))?;
    Ok((ram, disk))
}

/// Called from the GUI, CLI and guarded converters. For non-CPU devices,
/// false/unknown VRAM telemetry never becomes an invented positive.
pub fn assess_resources(
    source_bytes: u64, output_parent: &Path, mode: ResourceMode, device: &str
) -> io::Result<ResourceAssessment> {
    if !matches!(device, "cpu" | "cuda" | "xpu" | "auto") {
        return Err(invalid("device must be cpu, cuda, xpu, auto"));
    }
    let parent = output_parent.canonicalize()?;
    if !parent.is_dir() { return Err(invalid("output parent is not a directory")); }
    let (ram, disk) = budget(source_bytes, mode)?;
    let free_ram = available_ram();
    let free_volume = free_disk(&parent);
    let mut gpu_devices = nvidia_devices();
    gpu_devices.extend(xpu_devices());
    let device_measured = match device {
        "cpu" => true,
        "cuda" => gpu_devices.iter().any(|d| d["vendor"] == "nvidia" && d["free_vram_bytes"].as_u64().is_some()),
        "xpu" => gpu_devices.iter().any(|d| d["vendor"] == "intel" && d["free_vram_bytes"].as_u64().is_some()),
        // Auto-routing requires a selected physical device; enumeration alone is insufficient.
        "auto" => false,
        _ => false,
    };
    Ok(ResourceAssessment {
        source_bytes, mode, device: device.to_owned(),
        available_ram_bytes: free_ram,
        free_disk_bytes: free_volume,
        ram_safety_budget_bytes: ram,
        scratch_safety_budget_bytes: disk,
        gpu_devices,
        ram_safe: match (ram, free_ram) {
            (Some(required), Some(free)) => Some(free >= required),
            (None, _) => None,
            _ => None,
        },
        disk_safe: free_volume.map(|available| available >= disk),
        device_measured,
        output_parent: parent.display().to_string(),
    })
}

/// Read-only command entry point: model probe + host snapshot.
/// Model variants and target tool support are still unverified.
pub fn resource_plan(
    input: &Path, output_parent: &Path, mode: ResourceMode, device: &str
) -> io::Result<Value> {
    let model = probe_model(input)?;
    let mut result = assess_resources(model.bytes, output_parent, mode, device)?.as_json();
    result["source"] = json!(model.path.to_string_lossy().to_string());
    result["source_format"] = json!(model.format);
    result["source_category"] = json!(model.category.as_str());
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn conservative_budget_is_not_fake_peak() {
        let base = 7 * GIB;
        let (ram, disk) = budget(base, ResourceMode::Hqq).unwrap();
        assert_eq!(ram, Some(11 * GIB));
        assert_eq!(disk, 9 * GIB);
        let (ram, disk) = budget(base, ResourceMode::AutoRound).unwrap();
        assert_eq!(ram, Some(11 * GIB));
        assert_eq!(disk, 16 * GIB);
        let (ram, disk) = budget(base, ResourceMode::Gguf).unwrap();
        assert_eq!(ram, Some(4 * GIB)); // reserve only; NOT predicted peak
        assert_eq!(disk, 16 * GIB);
        assert!(budget(u64::MAX, ResourceMode::AutoRound).is_err());
    }
    #[test]
    fn intel_xpu_csv_requires_measured_free_memory() {
        let valid = "0, Intel Arc Pro B60, 24576, 20118.5\n1, Intel Arc Pro B70, N/A, 1000\n2, Intel GPU, 100, 101\n";
        let rows = parse_xpu_csv(valid);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["vendor"], "intel");
        assert_eq!(rows[0]["device_index"], 0);
        assert_eq!(rows[0]["free_vram_bytes"], (20118.5 * 1024.0 * 1024.0) as u64);
        assert!(parse_xpu_csv("0, Intel Arc, N/A, N/A\n").is_empty());
        assert!(parse_xpu_csv("0, Intel Arc, 24000, NaN\n").is_empty());
    }
    #[test]
    fn insufficient_and_unknown_resources_do_not_pass() {
        let base = ResourceAssessment {
            source_bytes: 1, mode: ResourceMode::Hqq, device: "xpu".into(),
            available_ram_bytes: Some(100), free_disk_bytes: Some(100),
            ram_safety_budget_bytes: Some(1), scratch_safety_budget_bytes: 1,
            gpu_devices: vec![], ram_safe: Some(true), disk_safe: Some(true),
            device_measured: false, output_parent: ".".into(),
        };
        assert_eq!(base.status(), "unverified");
        assert!(base.enforce(false).is_err());
        assert!(base.enforce(true).is_ok());
        let failed = ResourceAssessment {disk_safe: Some(false), device_measured: true, ..base};
        assert_eq!(failed.status(), "budget_below_policy");
        assert!(failed.enforce(false).is_err());
        assert_eq!(failed.as_json()["predicted_peak_ram_bytes"], Value::Null);
    }
}
