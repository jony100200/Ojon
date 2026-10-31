//! Ojon Learn V1 job controller: Preflight -> Train -> Verify -> Promote / Rollback.
//!
//! Safety contract: preflight never starts work; `start()` is explicit; the
//! trainer writes only to an isolated adapter directory; the source model is
//! never modified (fingerprint re-checked before promotion); a mock trainer is
//! always tagged `trainer=mock` with `real_gpu_training=NOT_RUN`.

use super::job_preflight::{
    run_preflight, scan_dataset, scan_model, CheckStatus, PreflightEnv, PreflightInput,
    PreflightReport, SCHEMA,
};
use super::job_verify::{compare, Evaluator, VerifyInputs, VerifyReport};
use super::knowledge::sha256_hex;
use super::train::LearnTrainPlan;
use crate::workflows::train::prepare_train;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub const MOCK_TRAINER_ENV: &str = "OJON_LEARN_MOCK_TRAINER";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MockBehavior {
    Succeed,
    Fail,
    /// Runs until `stop()`; used to test cancellation.
    Hang,
}

#[derive(Clone, Debug)]
pub enum TrainerKind {
    /// Installed `llamafactory-cli` (or shim) invoked as `<trainer> train <config>`.
    Llamafactory(PathBuf),
    /// A script standing in for the trainer (env `OJON_LEARN_MOCK_TRAINER`).
    MockScript(PathBuf),
    /// In-process mock, no subprocess.
    Mock(MockBehavior),
}

impl TrainerKind {
    pub fn is_mock(&self) -> bool {
        !matches!(self, Self::Llamafactory(_))
    }
    pub fn label(&self) -> String {
        match self {
            Self::Llamafactory(_) => "llamafactory".into(),
            _ => "mock".into(),
        }
    }
    /// `OJON_LEARN_MOCK_TRAINER` (when set) replaces the real trainer.
    pub fn from_env(real: PathBuf) -> Self {
        match std::env::var_os(MOCK_TRAINER_ENV).filter(|v| !v.is_empty()) {
            Some(p) => Self::MockScript(PathBuf::from(p)),
            None => Self::Llamafactory(real),
        }
    }
    fn exe(&self) -> PathBuf {
        match self {
            Self::Llamafactory(p) | Self::MockScript(p) => p.clone(),
            Self::Mock(_) => PathBuf::from("ojon-mock-trainer"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct LearnSpec {
    pub job_id: String,
    pub model_dir: PathBuf,
    pub dataset_file: PathBuf,
    pub heldout_suite: Option<PathBuf>,
    pub work_root: PathBuf,
    pub skill: String,
    pub device_request: String,
    pub trainer: TrainerKind,
    pub min_accepted: u64,
    pub env: PreflightEnv,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainSummary {
    pub adapter_dir: String,
    pub adapter_bytes: u64,
    pub elapsed_ms: u64,
    pub trainer: String,
    pub real_gpu_training: String,
    pub peak_vram: String,
    pub peak_ram: String,
    pub config_path: String,
    pub log_path: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "state", content = "detail", rename_all = "snake_case")]
pub enum LearnState {
    Idle,
    Preflight(PreflightReport),
    Ready,
    Training,
    Trained(TrainSummary),
    Verifying,
    Verified(VerifyReport),
    Promoted,
    RolledBack,
    Failed(String),
    Cancelled,
}

enum Outcome {
    Done,
    Failed(String),
    Cancelled,
}

struct Running {
    cancel: Arc<AtomicBool>,
    handle: JoinHandle<Outcome>,
    started: Instant,
}

pub struct LearnJob {
    spec: LearnSpec,
    state: LearnState,
    preflight: Option<PreflightReport>,
    train: Option<TrainSummary>,
    verify: Option<VerifyReport>,
    running: Option<Running>,
    receipt_count: u32,
}

fn err(msg: impl Into<String>) -> io::Error {
    io::Error::other(msg.into())
}

fn epoch_s() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// Windows `canonicalize` yields `\\?\C:\..`; trainers and YAML prefer plain paths.
fn plain(p: PathBuf) -> PathBuf {
    let s = p.to_string_lossy();
    match s.strip_prefix(r"\\?\") {
        Some(rest) if !rest.starts_with("UNC") => PathBuf::from(rest),
        _ => p,
    }
}

fn dir_bytes(dir: &Path) -> u64 {
    fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .filter_map(|e| e.metadata().ok())
                .filter(|m| m.is_file())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0)
}

fn write_atomic(path: &Path, value: &Value) -> io::Result<()> {
    let tmp = path.with_extension("tmp");
    fs::write(
        &tmp,
        serde_json::to_vec_pretty(value).map_err(io::Error::other)?,
    )?;
    fs::rename(&tmp, path)
}

fn run_mock(b: MockBehavior, out: &Path, cancel: &AtomicBool) -> Outcome {
    match b {
        MockBehavior::Fail => Outcome::Failed("mock trainer failed (exit status: 1)".into()),
        MockBehavior::Hang => {
            while !cancel.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(20));
            }
            Outcome::Cancelled
        }
        MockBehavior::Succeed => {
            let files = [
                (
                    "adapter_config.json",
                    &b"{\"peft_type\":\"LORA\",\"mock\":true}"[..],
                ),
                (
                    "adapter_model.safetensors",
                    b"OJON-MOCK-ADAPTER-NOT-REAL-WEIGHTS",
                ),
                (
                    "OJON_MOCK_TRAINER.txt",
                    b"trainer=mock; real GPU training NOT_RUN",
                ),
            ];
            let wrote = fs::create_dir_all(out).and_then(|_| {
                files
                    .iter()
                    .try_for_each(|(n, c)| fs::write(out.join(n), c))
            });
            match wrote {
                Ok(()) => Outcome::Done,
                Err(e) => Outcome::Failed(format!("mock trainer write failed: {e}")),
            }
        }
    }
}

/// Supervised subprocess: poll for exit, kill on cancel, surface non-zero status.
fn run_subprocess(exe: &Path, config: &Path, log: &Path, cancel: &AtomicBool) -> Outcome {
    let open = |p: &Path| File::create(p);
    let (Ok(so), Ok(se)) = (open(log), open(log)) else {
        return Outcome::Failed("cannot create trainer log file".into());
    };
    let spawned = Command::new(exe)
        .arg("train")
        .arg(config)
        .env("HF_HUB_OFFLINE", "1")
        .env("TRANSFORMERS_OFFLINE", "1")
        .env("HF_DATASETS_OFFLINE", "1")
        .stdin(Stdio::null())
        .stdout(so)
        .stderr(se)
        .spawn();
    let mut child = match spawned {
        Ok(c) => c,
        Err(e) => return Outcome::Failed(format!("cannot launch trainer {}: {e}", exe.display())),
    };
    loop {
        if cancel.load(Ordering::SeqCst) {
            let _ = child.kill();
            let _ = child.wait();
            return Outcome::Cancelled;
        }
        match child.try_wait() {
            Ok(Some(s)) if s.success() => return Outcome::Done,
            Ok(Some(s)) => return Outcome::Failed(format!("trainer failed: {s}")),
            Ok(None) => std::thread::sleep(Duration::from_millis(25)),
            Err(e) => return Outcome::Failed(format!("trainer wait failed: {e}")),
        }
    }
}

impl LearnJob {
    pub fn new(spec: LearnSpec) -> io::Result<Self> {
        let ok = !spec.job_id.is_empty()
            && spec.job_id.len() <= 64
            && spec
                .job_id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
        if !ok {
            return Err(err("job_id must be 1..=64 chars of [A-Za-z0-9_-]"));
        }
        Ok(Self {
            spec,
            state: LearnState::Idle,
            preflight: None,
            train: None,
            verify: None,
            running: None,
            receipt_count: 0,
        })
    }

    pub fn state(&self) -> &LearnState {
        &self.state
    }
    pub fn preflight_report(&self) -> Option<&PreflightReport> {
        self.preflight.as_ref()
    }
    pub fn job_dir(&self) -> PathBuf {
        self.spec.work_root.join("jobs").join(&self.spec.job_id)
    }
    pub fn adapter_dir(&self) -> PathBuf {
        self.spec.work_root.join("adapters").join(&self.spec.job_id)
    }
    fn releases_dir(&self) -> PathBuf {
        self.spec.work_root.join("releases")
    }

    fn real_gpu_label(&self, status: &str) -> &'static str {
        match (self.spec.trainer.is_mock(), status) {
            (true, _) => "NOT_RUN",
            (false, "ok") => "PASS",
            (false, "failed") => "FAIL",
            _ => "NOT_RUN",
        }
    }

    /// Stable `ojon.learn.v1` receipt; persisted under the job dir once it exists.
    fn receipt(&mut self, action: &str, status: &str, body: Value) -> Value {
        let mock = self.spec.trainer.is_mock();
        let mut v = json!({
            "schema": SCHEMA, "job_id": self.spec.job_id, "action": action,
            "status": status, "skill": self.spec.skill,
            "trainer": self.spec.trainer.label(), "mock": mock,
            "real_gpu_training": self.real_gpu_label(status),
            "timestamp_epoch_s": epoch_s(),
        });
        if let (Some(dst), Some(src)) = (v.as_object_mut(), body.as_object()) {
            dst.extend(src.iter().map(|(k, x)| (k.clone(), x.clone())));
        }
        let dir = self.job_dir().join("receipts");
        if self.job_dir().is_dir() && fs::create_dir_all(&dir).is_ok() {
            self.receipt_count += 1;
            let name = format!("{:02}_{action}.json", self.receipt_count);
            let _ = write_atomic(&dir.join(name), &v);
        }
        v
    }

    pub fn preflight(&mut self) -> io::Result<Value> {
        if matches!(self.state, LearnState::Training | LearnState::Verifying) {
            return Err(err("cannot run preflight while a job is active"));
        }
        let input = PreflightInput {
            model_dir: self.spec.model_dir.clone(),
            dataset_file: self.spec.dataset_file.clone(),
            heldout_suite: self.spec.heldout_suite.clone(),
            work_root: self.spec.work_root.clone(),
            device_request: self.spec.device_request.clone(),
            trainer_is_mock: self.spec.trainer.is_mock(),
            trainer_label: self.spec.trainer.label(),
            min_accepted: self.spec.min_accepted,
        };
        let report = run_preflight(&input, &self.spec.env);
        let passed = report.passed;
        let body = json!({ "preflight": report, "blockers": blockers(&report) });
        self.preflight = Some(report.clone());
        self.state = if passed {
            LearnState::Ready
        } else {
            LearnState::Preflight(report)
        };
        Ok(self.receipt("preflight", if passed { "ok" } else { "blocked" }, body))
    }

    /// Explicit start. Never called implicitly by preflight.
    pub fn start(&mut self) -> io::Result<Value> {
        if !matches!(self.state, LearnState::Ready) {
            return Err(err("start() requires a passing preflight (state Ready)"));
        }
        let pf = self
            .preflight
            .clone()
            .ok_or_else(|| err("no preflight report"))?;
        match self.prepare_run(&pf) {
            Ok((config, out)) => {
                let cancel = Arc::new(AtomicBool::new(false));
                let (kind, flag) = (self.spec.trainer.clone(), cancel.clone());
                let log = self.job_dir().join("trainer.log");
                let handle = std::thread::spawn(move || match &kind {
                    TrainerKind::Mock(b) => run_mock(*b, &out, &flag),
                    k => run_subprocess(&k.exe(), &config, &log, &flag),
                });
                self.running = Some(Running {
                    cancel,
                    handle,
                    started: Instant::now(),
                });
                self.state = LearnState::Training;
                Ok(self.receipt(
                    "start",
                    "started",
                    json!({ "adapter_dir": self.adapter_dir() }),
                ))
            }
            Err(e) => {
                self.state = LearnState::Failed(e.to_string());
                self.receipt("start", "failed", json!({ "error": e.to_string() }));
                Err(e)
            }
        }
    }

    /// Re-verifies inputs are unchanged, stages data, writes config, validates via `prepare_train`.
    fn prepare_run(&mut self, pf: &PreflightReport) -> io::Result<(PathBuf, PathBuf)> {
        let (_, model) = scan_model(&self.spec.model_dir);
        let scan = scan_dataset(&self.spec.dataset_file).map_err(err)?;
        let unchanged = model.map(|m| m.fingerprint) == pf.model_fingerprint
            && Some(&scan.sha256) == pf.dataset_sha256.as_ref();
        if !unchanged || scan.is_parquet {
            return Err(err("model/dataset changed since preflight (or parquet unsupported in V1); rerun preflight"));
        }
        fs::create_dir_all(self.spec.work_root.join("adapters"))?;
        let job_dir = self.job_dir();
        let data = job_dir.join("data");
        fs::create_dir_all(&data)?;
        let jsonl: String = scan
            .rows
            .iter()
            .map(|(i, o)| format!("{}\n", json!({"instruction": i, "input": "", "output": o})))
            .collect();
        fs::write(data.join("train.jsonl"), jsonl)?;
        fs::write(
            data.join("dataset_info.json"),
            json!({"ojon_sft": {"file_name": "train.jsonl"}}).to_string(),
        )?;
        write_atomic(&job_dir.join("preflight.json"), &json!(pf))?;
        let root = plain(self.spec.work_root.canonicalize()?);
        let out = root.join("adapters").join(&self.spec.job_id);
        let plan = LearnTrainPlan::new(
            plain(self.spec.model_dir.canonicalize()?),
            plain(data.canonicalize()?),
            out.clone(),
            None,
        );
        let config = job_dir.join("train_config.yaml");
        plan.save_recipe(&config)?;
        let tplan = prepare_train(&config, &self.spec.trainer.exe())?;
        write_atomic(&job_dir.join("train_plan.json"), &tplan.json())?;
        Ok((config, out))
    }

    /// Non-blocking: folds a finished trainer into the state machine.
    pub fn poll(&mut self) -> &LearnState {
        if self
            .running
            .as_ref()
            .is_some_and(|r| r.handle.is_finished())
        {
            let _ = self.finish();
        }
        &self.state
    }

    /// Blocks until the trainer ends. Err on trainer failure or cancellation.
    pub fn wait(&mut self) -> io::Result<Value> {
        self.finish()
    }

    fn finish(&mut self) -> io::Result<Value> {
        let r = self
            .running
            .take()
            .ok_or_else(|| err("no training in progress"))?;
        let elapsed = r.started.elapsed().as_millis() as u64;
        let outcome = r
            .handle
            .join()
            .unwrap_or_else(|_| Outcome::Failed("trainer thread panicked".into()));
        let outcome = match outcome {
            Outcome::Done => self.check_artifacts(),
            other => other,
        };
        match outcome {
            Outcome::Done => {
                let out = self.adapter_dir();
                let s = TrainSummary {
                    adapter_dir: out.to_string_lossy().into(),
                    adapter_bytes: dir_bytes(&out),
                    elapsed_ms: elapsed,
                    trainer: self.spec.trainer.label(),
                    real_gpu_training: self.real_gpu_label("ok").into(),
                    peak_vram: "NOT_RUN".into(),
                    peak_ram: "NOT_RUN".into(),
                    config_path: self
                        .job_dir()
                        .join("train_config.yaml")
                        .to_string_lossy()
                        .into(),
                    log_path: self.job_dir().join("trainer.log").to_string_lossy().into(),
                };
                self.train = Some(s.clone());
                self.state = LearnState::Trained(s.clone());
                Ok(self.receipt("train", "ok", json!({ "train": s })))
            }
            Outcome::Cancelled => {
                self.state = LearnState::Cancelled;
                self.receipt(
                    "train",
                    "cancelled",
                    json!({ "partial_adapter_dir": self.adapter_dir() }),
                );
                Err(err("training cancelled"))
            }
            Outcome::Failed(m) => {
                self.state = LearnState::Failed(m.clone());
                self.receipt(
                    "train",
                    "failed",
                    json!({ "error": m, "log": self.job_dir().join("trainer.log") }),
                );
                Err(err(m))
            }
        }
    }

    fn check_artifacts(&self) -> Outcome {
        let out = self.adapter_dir();
        let missing: Vec<&str> = ["adapter_config.json", "adapter_model.safetensors"]
            .into_iter()
            .filter(|f| fs::metadata(out.join(f)).map_or(true, |m| m.len() == 0))
            .collect();
        if missing.is_empty() {
            Outcome::Done
        } else {
            Outcome::Failed(format!(
                "trainer exited 0 but adapter output is missing: {}",
                missing.join(", ")
            ))
        }
    }

    pub fn stop(&mut self) -> io::Result<Value> {
        let Some(r) = self.running.as_ref() else {
            return Err(err("stop() requires a running training job"));
        };
        r.cancel.store(true, Ordering::SeqCst);
        match self.finish() {
            Err(e) if matches!(self.state, LearnState::Cancelled) => {
                let _ = e;
                Ok(self.receipt(
                    "stop",
                    "cancelled",
                    json!({ "partial_adapter_dir": self.adapter_dir() }),
                ))
            }
            Ok(v) => Ok(v), // trainer finished before the kill landed
            Err(e) => Err(e),
        }
    }

    pub fn verify(
        &mut self,
        inputs: &VerifyInputs,
        evaluator: &dyn Evaluator,
    ) -> io::Result<Value> {
        let LearnState::Trained(_) = self.state else {
            return Err(err(
                "verify() requires a finished training run (state Trained)",
            ));
        };
        let suite = self
            .spec
            .heldout_suite
            .clone()
            .ok_or_else(|| err("no held-out suite"))?;
        let suite_sha = sha256_hex(&fs::read(&suite)?);
        let pf = self
            .preflight
            .clone()
            .ok_or_else(|| err("no preflight report"))?;
        if Some(&suite_sha) != pf.heldout_suite_sha256.as_ref() {
            return Err(err(
                "held-out suite changed since preflight: it must stay frozen",
            ));
        }
        let mut rep = VerifyReport {
            schema: "ojon.learn.verify.v1".into(),
            status: CheckStatus::NotRun,
            reason: String::new(),
            suite_sha256: suite_sha,
            evaluator: evaluator.id().into(),
            evaluator_is_mock: evaluator.is_mock(),
            trainer_is_mock: self.spec.trainer.is_mock(),
            real_gpu_training: self.real_gpu_label("ok").into(),
            metrics: Vec::new(),
            evidence_paths: vec![suite.to_string_lossy().into()],
        };
        let Some(cand) = inputs.candidate_model.as_ref() else {
            rep.reason = "NOT_RUN: adapter not merged/converted into an evaluable model; no scores fabricated".into();
            return Ok(self.finish_verify(rep));
        };
        self.state = LearnState::Verifying;
        let report_path = self.job_dir().join("verify_eval.json");
        if report_path.exists() {
            let _ = fs::remove_file(&report_path);
        }
        match evaluator.evaluate(&suite, &inputs.base_model, cand, &report_path) {
            Ok((base, candidate)) => {
                let (metrics, status, why) = compare(&base, &candidate);
                rep.metrics = metrics;
                rep.status = status;
                rep.reason = why;
                rep.evidence_paths
                    .push(report_path.to_string_lossy().into());
                Ok(self.finish_verify(rep))
            }
            Err(e) => {
                if let Some(t) = &self.train {
                    self.state = LearnState::Trained(t.clone());
                }
                self.receipt("verify", "failed", json!({ "error": e.to_string() }));
                Err(e)
            }
        }
    }

    fn finish_verify(&mut self, rep: VerifyReport) -> Value {
        let status = match rep.status {
            CheckStatus::Pass => "ok",
            CheckStatus::Fail => "failed",
            CheckStatus::NotRun => "not_run",
        };
        self.verify = Some(rep.clone());
        self.state = LearnState::Verified(rep.clone());
        self.receipt("verify", status, json!({ "verify": rep }))
    }

    pub fn promote(&mut self) -> io::Result<Value> {
        let LearnState::Verified(rep) = self.state.clone() else {
            return Err(err("promote() requires a Verified candidate"));
        };
        if rep.status != CheckStatus::Pass {
            return Err(err(format!(
                "candidate did not pass verification ({:?}); not promoting",
                rep.status
            )));
        }
        let (_, model) = scan_model(&self.spec.model_dir);
        let pf = self
            .preflight
            .clone()
            .ok_or_else(|| err("no preflight report"))?;
        if model.map(|m| m.fingerprint) != pf.model_fingerprint {
            return Err(err(
                "original model changed during the job: refusing to promote",
            ));
        }
        let train = self
            .train
            .clone()
            .ok_or_else(|| err("no training summary"))?;
        let rel = self.releases_dir();
        fs::create_dir_all(&rel)?;
        let (cur, prev) = (rel.join("CURRENT.json"), rel.join("PREVIOUS.json"));
        if cur.is_file() {
            fs::copy(&cur, &prev)?;
        } else if prev.is_file() {
            fs::remove_file(&prev)?;
        }
        let pointer = json!({
            "schema": "ojon.learn.release.v1", "job_id": self.spec.job_id, "skill": self.spec.skill,
            "adapter_dir": train.adapter_dir, "adapter_bytes": train.adapter_bytes,
            "base_model_dir": self.spec.model_dir, "base_model_preserved": true,
            "base_model_fingerprint": pf.model_fingerprint, "dataset_sha256": pf.dataset_sha256,
            "trainer": self.spec.trainer.label(), "mock": self.spec.trainer.is_mock(),
            "real_gpu_training": train.real_gpu_training, "elapsed_ms": train.elapsed_ms,
            "peak_vram": train.peak_vram, "peak_ram": train.peak_ram,
            "scores": rep.metrics, "evaluator": rep.evaluator, "evaluator_is_mock": rep.evaluator_is_mock,
            "promoted_at_epoch_s": epoch_s(),
        });
        write_atomic(&cur, &pointer)?;
        self.state = LearnState::Promoted;
        Ok(self.receipt("promote", "ok", json!({ "release": pointer })))
    }

    pub fn rollback(&mut self) -> io::Result<Value> {
        if !matches!(self.state, LearnState::Promoted) {
            return Err(err("rollback() requires a Promoted job"));
        }
        let restored = rollback_release(&self.spec.work_root)?;
        self.state = LearnState::RolledBack;
        Ok(self.receipt("rollback", "ok", restored))
    }
}

fn blockers(r: &PreflightReport) -> Vec<String> {
    r.checks
        .iter()
        .filter(|c| c.status != CheckStatus::Pass)
        .map(|c| format!("{}: {}", c.name, c.reason))
        .collect()
}

/// The active release pointer, if any.
pub fn current_release(work_root: &Path) -> Option<Value> {
    let bytes = fs::read(work_root.join("releases").join("CURRENT.json")).ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// Restores the previous release, or clears the pointer (back to the untouched
/// base model). Adapter directories and original checkpoints are never deleted.
pub fn rollback_release(work_root: &Path) -> io::Result<Value> {
    let rel = work_root.join("releases");
    let (cur, prev) = (rel.join("CURRENT.json"), rel.join("PREVIOUS.json"));
    let current =
        current_release(work_root).ok_or_else(|| err("no active release to roll back"))?;
    let archive = rel.join("rolled_back");
    fs::create_dir_all(&archive)?;
    let name = format!(
        "{}_{}.json",
        epoch_s(),
        current["job_id"].as_str().unwrap_or("unknown")
    );
    fs::rename(&cur, archive.join(name))?;
    if prev.is_file() {
        fs::rename(&prev, &cur)?;
    }
    let restored = current_release(work_root);
    Ok(json!({
        "rolled_back_job_id": current["job_id"],
        "restored_job_id": restored.as_ref().map(|r| r["job_id"].clone()),
        "active": if restored.is_some() { "previous_release" } else { "original_base_model" },
    }))
}
