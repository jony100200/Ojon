//! Capture factual, bounded, line-oriented subprocess output for the native GUI.
//! Both stdout and stderr are read concurrently to avoid pipe deadlocks.
//! No fake completion percentages, inferred GPU usage or synthetic progress.

use std::io::Read;
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

static JOB_CANCEL_REQUESTED: AtomicBool = AtomicBool::new(false);
static JOB_SEQ: AtomicU64 = AtomicU64::new(1);

/// Request cooperative/immediate cancellation of any active Ojon subprocess job.
pub fn request_job_cancel() {
    JOB_CANCEL_REQUESTED.store(true, Ordering::SeqCst);
}

/// Clear the cancellation flag before starting a new job.
pub fn clear_job_cancel() {
    JOB_CANCEL_REQUESTED.store(false, Ordering::SeqCst);
}

/// Returns true when the user or caller has requested cancellation.
pub fn is_job_cancelled() -> bool {
    JOB_CANCEL_REQUESTED.load(Ordering::SeqCst)
}

/// Peak memory is the model subprocess working-set high-water mark where
/// the host OS exposes it, not whole-system RAM, allocator commit or GPU VRAM.
#[cfg(windows)]
fn peak_process_ram(child: &Child) -> Option<u64> {
    use std::ffi::c_void;
    use std::os::windows::io::AsRawHandle;
    #[repr(C)]
    struct Counters {
        cb: u32,
        faults: u32,
        peak_ws: usize,
        working_ws: usize,
        peak_paged: usize,
        paged: usize,
        peak_nonpaged: usize,
        nonpaged: usize,
        pagefile: usize,
        peak_pagefile: usize,
    }
    #[link(name = "psapi")]
    extern "system" {
        fn GetProcessMemoryInfo(process: *mut c_void, counters: *mut Counters, cb: u32) -> i32;
    }
    let mut value = Counters {
        cb: std::mem::size_of::<Counters>() as u32,
        faults: 0,
        peak_ws: 0,
        working_ws: 0,
        peak_paged: 0,
        paged: 0,
        peak_nonpaged: 0,
        nonpaged: 0,
        pagefile: 0,
        peak_pagefile: 0,
    };
    let size = value.cb;
    let success =
        unsafe { GetProcessMemoryInfo(child.as_raw_handle() as *mut c_void, &mut value, size) };
    if success == 0 {
        None
    } else {
        Some(value.peak_ws as u64)
    }
}
#[cfg(target_os = "linux")]
fn peak_process_ram(child: &Child) -> Option<u64> {
    let status = std::fs::read_to_string(format!("/proc/{}/status", child.id())).ok()?;
    let kb = status
        .lines()
        .find(|v| v.starts_with("VmHWM:"))?
        .split_whitespace()
        .nth(1)?
        .parse::<u64>()
        .ok()?;
    kb.checked_mul(1024)
}
#[cfg(not(any(windows, target_os = "linux")))]
fn peak_process_ram(_child: &Child) -> Option<u64> {
    None
}

pub struct CapturedProcess {
    pub output: Output,
    pub peak_ram_bytes: Option<u64>,
}

/// Bounded output for untrusted local model executables. Both pipes are
/// drained concurrently even after reaching the configured memory cap.
/// On timeout the child is killed and reaped; no success receipt is produced.
pub fn capture_with_timeout(
    cmd: &mut Command,
    limit: usize,
    timeout: Duration,
) -> io::Result<Output> {
    capture_with_timeout_measured(cmd, limit, timeout).map(|r| r.output)
}
/// Also records actual child-process peak working set, if available.
pub fn capture_with_timeout_measured(
    cmd: &mut Command,
    limit: usize,
    timeout: Duration,
) -> io::Result<CapturedProcess> {
    if limit == 0 || limit > 16 * 1024 * 1024 || timeout.is_zero() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid capture limit or timeout",
        ));
    }
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    fn drain<R: Read>(mut pipe: R, limit: usize) -> io::Result<(Vec<u8>, bool)> {
        let mut data = Vec::new();
        let mut extra = false;
        let mut block = [0u8; 8192];
        loop {
            let n = pipe.read(&mut block)?;
            if n == 0 {
                break;
            }
            let room = limit.saturating_sub(data.len());
            data.extend_from_slice(&block[..n.min(room)]);
            if n > room {
                extra = true;
            }
        }
        Ok((data, extra))
    }
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| io::Error::other("missing stdout pipe"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| io::Error::other("missing stderr pipe"))?;
    let out = thread::spawn(move || drain(stdout, limit));
    let err = thread::spawn(move || drain(stderr, limit));
    let started = Instant::now();
    let mut timed_out = false;
    let mut cancelled = false;
    let mut peak_ram_bytes = peak_process_ram(&child);
    let status = loop {
        if let Some(value) = peak_process_ram(&child) {
            peak_ram_bytes = Some(peak_ram_bytes.unwrap_or(0).max(value));
        }
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if is_job_cancelled() {
            cancelled = true;
            let _ = child.kill();
            break child.wait()?;
        }
        if started.elapsed() > timeout {
            timed_out = true;
            let _ = child.kill();
            break child.wait()?;
        }
        thread::sleep(Duration::from_millis(100));
    };
    let stdout = out
        .join()
        .map_err(|_| io::Error::other("stdout worker panicked"))??;
    let stderr = err
        .join()
        .map_err(|_| io::Error::other("stderr worker panicked"))??;
    if cancelled {
        return Err(io::Error::new(
            io::ErrorKind::Interrupted,
            "Ojon job cancelled by user",
        ));
    }
    if timed_out {
        return Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "model backend exceeded execution timeout",
        ));
    }
    if stdout.1 || stderr.1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "backend output exceeded capture size limit",
        ));
    }
    Ok(CapturedProcess {
        output: Output {
            status,
            stdout: stdout.0,
            stderr: stderr.0,
        },
        peak_ram_bytes,
    })
}

const MAX_LINE_BYTES: usize = 2048;
const MAX_TOTAL_LOG_BYTES: usize = 8 * 1024 * 1024;

/// Streams stdout/stderr text from an existing, explicitly selected backend.
/// The callback must be cheap; GUI callers should forward events to a channel.
/// Lines longer than MAX_LINE_BYTES are split into fragments to bound memory.
/// Total display output is limited; conversion continues even after that limit.
/// The default bound is 4 hours; users may explicitly configure a longer
/// legitimate conversion via OJON_JOB_TIMEOUT_SECONDS (5..=86400).
/// This is a local process limit, not a model performance estimate.
pub fn run_with_log<F>(cmd: &mut Command, emit: F) -> io::Result<()>
where
    F: FnMut(&str),
{
    let timeout = match std::env::var("OJON_JOB_TIMEOUT_SECONDS") {
        Ok(raw) => Duration::from_secs(
            raw.parse::<u64>()
                .ok()
                .filter(|seconds| (5..=86400).contains(seconds))
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "OJON_JOB_TIMEOUT_SECONDS must be 5..=86400",
                    )
                })?,
        ),
        Err(std::env::VarError::NotPresent) => Duration::from_secs(4 * 3600),
        Err(e) => return Err(io::Error::new(io::ErrorKind::InvalidInput, e)),
    };
    run_with_log_timeout(cmd, timeout, emit)
}
/// Bound even silent subprocesses. Timeout kills and reaps the selected
/// direct child, closes the receiver and never claims successful output.
pub fn run_with_log_timeout<F>(cmd: &mut Command, timeout: Duration, mut emit: F) -> io::Result<()>
where
    F: FnMut(&str),
{
    if timeout.is_zero() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "external backend timeout must be positive",
        ));
    }
    let started = Instant::now();
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| io::Error::other("missing stdout pipe"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| io::Error::other("missing stderr pipe"))?;
    let (tx, rx) = mpsc::sync_channel::<String>(64);

    let readers: Vec<_> = vec![
        thread::spawn({
            let tx = tx.clone();
            move || read_pipe(stdout, tx, "stdout")
        }),
        thread::spawn(move || read_pipe(stderr, tx, "stderr")),
    ];

    // Drain both pipes until they close. Even after reaching the display cap,
    // keep consuming to avoid stalling the converter.
    let mut forwarded = 0usize;
    loop {
        if is_job_cancelled() {
            let _ = child.kill();
            let _ = child.wait();
            drop(rx);
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "Ojon job cancelled by user",
            ));
        }
        let remaining = timeout.saturating_sub(started.elapsed());
        if remaining.is_zero() {
            let _ = child.kill();
            let _ = child.wait();
            drop(rx);
            // Detached readers exit once the killed process closes its pipes;
            // joining a pipe inherited by a grandchild would reintroduce a
            // potentially unbounded wait on this error path.
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "external backend exceeded Ojon subprocess timeout",
            ));
        }
        match rx.recv_timeout(remaining.min(Duration::from_millis(250))) {
            Ok(line) => {
                if forwarded < MAX_TOTAL_LOG_BYTES {
                    forwarded = forwarded.saturating_add(line.len());
                    emit(&line);
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    let mut reader_errors = Vec::new();
    for reader in readers {
        match reader.join() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => reader_errors.push(e),
            Err(_) => reader_errors.push(io::Error::other("backend output reader panicked")),
        }
    }
    let status = child.wait()?;
    if !status.success() {
        return Err(io::Error::other(format!(
            "external backend exited with {status}"
        )));
    }
    if let Some(error) = reader_errors.into_iter().next() {
        return Err(error);
    }
    Ok(())
}
fn read_pipe<R: io::Read>(input: R, tx: mpsc::SyncSender<String>, stream: &str) -> io::Result<()> {
    let mut reader = BufReader::new(input);
    let mut partial = Vec::new();
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            if !partial.is_empty() {
                let line = String::from_utf8_lossy(&partial);
                let _ = tx.send(format!("[{stream}] {line}"));
            }
            return Ok(());
        }
        let newline = available.iter().position(|b| *b == b'\n');
        let take = newline.map(|n| n + 1).unwrap_or(available.len());
        let mut consumed = 0;
        while consumed < take {
            let room = MAX_LINE_BYTES.saturating_sub(partial.len());
            let chunk = room.min(take - consumed);
            partial.extend_from_slice(&available[consumed..consumed + chunk]);
            consumed += chunk;
            if partial.len() == MAX_LINE_BYTES {
                let line = String::from_utf8_lossy(&partial);
                if tx.send(format!("[{stream}] {line}")).is_err() {
                    return Ok(());
                }
                partial.clear();
            }
        }
        reader.consume(take);
        if newline.is_some() {
            let line = String::from_utf8_lossy(&partial);
            if !line.trim().is_empty()
                && tx.send(format!("[{stream}] {}", line.trim_end())).is_err()
            {
                return Ok(());
            }
            partial.clear();
        }
    }
}

/// Persistent job history and checkpoint recovery record stored across app restarts.
#[derive(Debug, Clone, PartialEq)]
pub struct JobHistoryEntry {
    pub id: String,
    pub workflow: String,
    pub source_model: String,
    pub output_dir: String,
    pub objective_or_mode: String,
    pub status: String,
    pub current_stage: String,
    pub started_epoch_secs: u64,
    pub finished_epoch_secs: Option<u64>,
    pub result_artifact: Option<String>,
    pub summary_note: String,
}

impl JobHistoryEntry {
    pub fn as_json(&self) -> serde_json::Value {
        serde_json::json!({
            "id": self.id,
            "workflow": self.workflow,
            "source_model": self.source_model,
            "output_dir": self.output_dir,
            "objective_or_mode": self.objective_or_mode,
            "status": self.status,
            "current_stage": self.current_stage,
            "started_epoch_secs": self.started_epoch_secs,
            "finished_epoch_secs": self.finished_epoch_secs,
            "result_artifact": self.result_artifact,
            "summary_note": self.summary_note,
        })
    }

    pub fn from_json(val: &serde_json::Value) -> Option<Self> {
        Some(Self {
            id: val.get("id")?.as_str()?.to_owned(),
            workflow: val.get("workflow")?.as_str()?.to_owned(),
            source_model: val.get("source_model")?.as_str()?.to_owned(),
            output_dir: val.get("output_dir")?.as_str()?.to_owned(),
            objective_or_mode: val.get("objective_or_mode")?.as_str()?.to_owned(),
            status: val.get("status")?.as_str()?.to_owned(),
            current_stage: val
                .get("current_stage")
                .and_then(|v| v.as_str())
                .unwrap_or("completed")
                .to_owned(),
            started_epoch_secs: val
                .get("started_epoch_secs")
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            finished_epoch_secs: val.get("finished_epoch_secs").and_then(|v| v.as_u64()),
            result_artifact: val
                .get("result_artifact")
                .and_then(|v| v.as_str())
                .map(str::to_owned),
            summary_note: val
                .get("summary_note")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_owned(),
        })
    }
}

pub fn job_history_path() -> io::Result<PathBuf> {
    if let Ok(custom) = std::env::var("OJON_JOB_HISTORY_FILE") {
        let trimmed = custom.trim();
        if !trimmed.is_empty() {
            return Ok(PathBuf::from(trimmed));
        }
    }
    let settings = crate::core::settings::settings_path()?;
    let parent = settings
        .parent()
        .ok_or_else(|| io::Error::other("settings path has no parent directory"))?;
    Ok(parent.join("jobs-history.json"))
}

pub fn load_job_history_from(path: &Path) -> io::Result<Vec<JobHistoryEntry>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let raw = std::fs::read_to_string(path)?;
    let parsed: serde_json::Value = serde_json::from_str(&raw).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid job history JSON: {e}"),
        )
    })?;
    let mut entries = Vec::new();
    for item in parsed["jobs"].as_array().into_iter().flatten() {
        if let Some(mut entry) = JobHistoryEntry::from_json(item) {
            if entry.status == "RUNNING" {
                entry.status = "CHECKPOINT_RECOVERABLE".to_owned();
                if entry.summary_note.is_empty() {
                    entry.summary_note = "Interrupted session detected; ready to resume from existing output directory checkpoints.".to_owned();
                }
            }
            entries.push(entry);
        }
    }
    Ok(entries)
}

pub fn load_job_history() -> io::Result<Vec<JobHistoryEntry>> {
    let path = job_history_path()?;
    load_job_history_from(&path)
}

pub fn save_job_history_to(path: &Path, entries: &[JobHistoryEntry]) -> io::Result<PathBuf> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let retained: Vec<serde_json::Value> = entries
        .iter()
        .rev()
        .take(50)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .map(JobHistoryEntry::as_json)
        .collect();
    let payload = serde_json::to_string_pretty(&serde_json::json!({
        "schema": "ojon.jobs_history.v1",
        "jobs": retained,
    }))
    .map_err(io::Error::other)?;
    let tmp = path.with_extension(format!("json.tmp.{}", std::process::id()));
    std::fs::write(&tmp, payload)?;
    if path.exists() {
        let _ = std::fs::remove_file(path);
    }
    std::fs::rename(&tmp, path)?;
    Ok(path.to_path_buf())
}

pub fn save_job_history(entries: &[JobHistoryEntry]) -> io::Result<PathBuf> {
    let path = job_history_path()?;
    save_job_history_to(&path, entries)
}

fn now_epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub fn record_job_start(
    workflow: &str,
    source_model: &Path,
    output_dir: &Path,
    objective_or_mode: &str,
    initial_stage: &str,
) -> io::Result<String> {
    let path = job_history_path()?;
    let mut entries = load_job_history_from(&path).unwrap_or_default();
    let seq = JOB_SEQ.fetch_add(1, Ordering::Relaxed);
    let id = format!("job-{}-{seq}", now_epoch_secs());
    entries.push(JobHistoryEntry {
        id: id.clone(),
        workflow: workflow.to_owned(),
        source_model: source_model.to_string_lossy().into_owned(),
        output_dir: output_dir.to_string_lossy().into_owned(),
        objective_or_mode: objective_or_mode.to_owned(),
        status: "RUNNING".to_owned(),
        current_stage: initial_stage.to_owned(),
        started_epoch_secs: now_epoch_secs(),
        finished_epoch_secs: None,
        result_artifact: None,
        summary_note: "In progress".to_owned(),
    });
    save_job_history_to(&path, &entries)?;
    Ok(id)
}

pub fn update_job_stage(job_id: &str, stage: &str) -> io::Result<()> {
    let path = job_history_path()?;
    let mut entries = load_job_history_from(&path).unwrap_or_default();
    if let Some(entry) = entries.iter_mut().rev().find(|e| e.id == job_id) {
        entry.status = "RUNNING".to_owned();
        entry.current_stage = stage.to_owned();
        save_job_history_to(&path, &entries)?;
    }
    Ok(())
}

pub fn finish_job_history(
    job_id: &str,
    status: &str,
    result_artifact: Option<&Path>,
    summary_note: &str,
) -> io::Result<()> {
    let path = job_history_path()?;
    let mut entries = load_job_history_from(&path).unwrap_or_default();
    if let Some(entry) = entries.iter_mut().rev().find(|e| e.id == job_id) {
        entry.status = status.to_owned();
        entry.finished_epoch_secs = Some(now_epoch_secs());
        entry.result_artifact = result_artifact.map(|p| p.to_string_lossy().into_owned());
        entry.summary_note = summary_note.to_owned();
        save_job_history_to(&path, &entries)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stalled_external_process_is_terminated() {
        let mut command = if cfg!(windows) {
            let mut c = Command::new("powershell");
            c.args(["-NoProfile", "-Command", "Start-Sleep -Seconds 5"]);
            c
        } else {
            let mut c = Command::new("sh");
            c.args(["-c", "sleep 5"]);
            c
        };
        let started = Instant::now();
        let error =
            run_with_log_timeout(&mut command, Duration::from_millis(200), |_| {}).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::TimedOut);
        assert!(started.elapsed() < Duration::from_secs(4));
    }
    #[test]
    fn reports_failing_command_without_faking_success() {
        let mut command = if cfg!(windows) {
            let mut c = Command::new("cmd");
            c.args(["/C", "exit 7"]);
            c
        } else {
            let mut c = Command::new("sh");
            c.args(["-c", "exit 7"]);
            c
        };
        assert!(run_with_log(&mut command, |_| {}).is_err());
    }
    #[test]
    fn persistent_job_history_recovers_running_entries_as_checkpoints() {
        let tmp = std::env::temp_dir().join(format!("ojon-jobs-test-{}.json", std::process::id()));
        let entries = vec![JobHistoryEntry {
            id: "job-1".into(),
            workflow: "optimize".into(),
            source_model: "D:/models/base.gguf".into(),
            output_dir: "D:/models/out".into(),
            objective_or_mode: "smallest".into(),
            status: "RUNNING".into(),
            current_stage: "Quantizing candidate Q4_K_M".into(),
            started_epoch_secs: 1000,
            finished_epoch_secs: None,
            result_artifact: None,
            summary_note: String::new(),
        }];
        save_job_history_to(&tmp, &entries).unwrap();
        let loaded = load_job_history_from(&tmp).unwrap();
        let _ = std::fs::remove_file(&tmp);
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].status, "CHECKPOINT_RECOVERABLE");
        assert!(loaded[0]
            .summary_note
            .contains("Interrupted session detected"));
    }
}
