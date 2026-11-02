//! Optional bridge to Bunon, which owns training data and training jobs.
//!
//! On 2026-11-02 Ojon's dataset collection, curation, distillation and LoRA/QLoRA
//! training engine moved to Bunon (`bunon-learn`). Ojon keeps quantisation,
//! conversion, calibration, model editing, evaluation, adapter merge and release
//! tooling, and stays fully usable when Bunon is absent. The legacy commands
//! below are forwarded verbatim to `bunon-learn`; nothing is linked or duplicated.
//!
//! Discovery order: `OJON_BUNON_BIN`, then `bunon-learn` on `PATH`.

use std::ffi::OsString;
use std::io;
use std::path::PathBuf;
use std::process::{Command, ExitStatus};

pub const BUNON_BIN_ENV: &str = "OJON_BUNON_BIN";
pub const BUNON_REPOSITORY: &str = "https://github.com/jony100200/Bunon";

/// Commands that now belong to Bunon. Keep in sync with `bunon-learn` COMMANDS.
pub const FORWARDED_COMMANDS: &[&str] = &[
    "learn-collector-status",
    "learn-collector-start",
    "learn-harvest",
    "learn-utility-models",
    "learn-dataset-list",
    "learn-dataset-import",
    "learn-curate",
    "learn-daemon-run",
    "learn-train-plan",
    "learn-train-run",
    "learn-promote",
    "learn-knowledge-add",
    "learn-temporal-query",
    "learn-package-create",
    "learn-package-install",
    "learn-package-list",
    "learn-package-rollback",
    "dataset-audit",
    "distill-plan",
    "retrieve",
    "train-plan",
    "train-run",
];

pub fn is_forwarded(cmd: &str) -> bool {
    FORWARDED_COMMANDS.contains(&cmd)
}

/// Locates `bunon-learn`, or `None` when Bunon is not installed.
pub fn bunon_learn_binary() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os(BUNON_BIN_ENV).filter(|v| !v.is_empty()) {
        let path = PathBuf::from(path);
        return path.is_file().then_some(path);
    }
    let paths = std::env::var_os("PATH")?;
    let names: &[&str] = if cfg!(windows) {
        &["bunon-learn.exe", "bunon-learn"]
    } else {
        &["bunon-learn"]
    };
    std::env::split_paths(&paths)
        .flat_map(|dir| names.iter().map(move |n| dir.join(n)))
        .find(|candidate| candidate.is_file())
}

/// Clear, actionable error used when a forwarded command runs without Bunon.
pub fn missing_bunon_error(cmd: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::NotFound,
        format!(
            "`{cmd}` moved to Bunon (training data and training jobs are Bunon's). \
             Install Bunon ({BUNON_REPOSITORY}, `cargo build --release --bin bunon-learn`) \
             and put `bunon-learn` on PATH or set {BUNON_BIN_ENV}. \
             Quantisation, conversion, model editing, evaluation and adapter merge stay in Ojon."
        ),
    )
}

/// Runs a forwarded command with inherited stdio and returns the child status.
pub fn forward(cmd: &str, args: impl IntoIterator<Item = OsString>) -> io::Result<ExitStatus> {
    let bin = bunon_learn_binary().ok_or_else(|| missing_bunon_error(cmd))?;
    Command::new(bin).arg(cmd).args(args).status()
}

/// Runs a forwarded command and captures stdout (for the GUI and agents).
pub fn forward_capture(cmd: &str, args: &[OsString]) -> io::Result<String> {
    let bin = bunon_learn_binary().ok_or_else(|| missing_bunon_error(cmd))?;
    let out = Command::new(bin).arg(cmd).args(args).output()?;
    if !out.status.success() {
        return Err(io::Error::other(format!(
            "bunon-learn {cmd} failed ({}): {}",
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forwarded_commands_cover_legacy_training_surface() {
        for cmd in ["learn-daemon-run", "learn-curate", "train-run", "distill-plan"] {
            assert!(is_forwarded(cmd), "{cmd} must forward to Bunon");
        }
        for cmd in ["quantize", "lora-merge-run", "task-eval-run", "resource-plan"] {
            assert!(!is_forwarded(cmd), "{cmd} must stay in Ojon");
        }
    }

    #[test]
    fn missing_bunon_error_names_the_route() {
        let e = missing_bunon_error("learn-curate").to_string();
        assert!(e.contains("Bunon") && e.contains(BUNON_BIN_ENV) && e.contains("bunon-learn"));
    }
}
