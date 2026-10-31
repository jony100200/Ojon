//! CPU-only tests for the Ojon Learn V1 controller. No GPU is ever used.

use super::job::*;
use super::job_preflight::{CheckStatus, DeviceInfo, PreflightEnv};
use super::job_verify::{Evaluator, Scores, VerifyInputs};
use serde_json::{json, Value};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

const GIB: u64 = 1 << 30;

fn scratch(tag: &str) -> PathBuf {
    static N: AtomicU32 = AtomicU32::new(0);
    let p = std::env::temp_dir().join(format!(
        "ojon_learn_{tag}_{}_{}",
        std::process::id(),
        N.fetch_add(1, Ordering::SeqCst)
    ));
    let _ = fs::remove_dir_all(&p);
    fs::create_dir_all(&p).unwrap();
    p.canonicalize().unwrap()
}

fn make_model(root: &Path) -> PathBuf {
    let m = root.join("base_hf");
    fs::create_dir_all(&m).unwrap();
    fs::write(m.join("config.json"), r#"{"model_type":"llama"}"#).unwrap();
    fs::write(m.join("tokenizer.json"), "{}").unwrap();
    fs::write(m.join("model.safetensors"), vec![7u8; 4096]).unwrap();
    m
}

fn make_dataset(root: &Path, extra: &[&str]) -> PathBuf {
    let p = root.join("skill.jsonl");
    let mut lines: Vec<String> = (0..10)
        .map(|i| json!({"instruction": format!("train question number {i}"), "output": format!("answer {i}")}).to_string())
        .collect();
    lines.push(
        json!({"instruction": "unverified trace", "output": "x", "verified": false}).to_string(),
    );
    lines.push("not json".into());
    lines.extend(extra.iter().map(|s| s.to_string()));
    fs::write(&p, lines.join("\n")).unwrap();
    p
}

fn make_suite(root: &Path) -> PathBuf {
    let cats = ["coding", "reasoning", "tool_call"];
    let tasks: Vec<Value> = (0..6)
        .map(|i| {
            json!({"id": format!("t{i}"), "category": cats[i % 3],
            "prompt": format!("heldout probe {i}"), "expected": "ok", "match": "exact"})
        })
        .collect();
    let p = root.join("suite.json");
    fs::write(
        &p,
        json!({"schema": "ojon.task_suite.v1", "id": "frozen", "tasks": tasks}).to_string(),
    )
    .unwrap();
    p
}

fn spec(root: &Path, id: &str, trainer: TrainerKind) -> LearnSpec {
    LearnSpec {
        job_id: id.into(),
        model_dir: make_model(root),
        dataset_file: make_dataset(root, &[]),
        heldout_suite: Some(make_suite(root)),
        work_root: root.join("work"),
        skill: "unit-test-skill".into(),
        device_request: "cpu".into(),
        trainer,
        min_accepted: 8,
        env: PreflightEnv {
            devices: Some(Vec::new()),
            free_disk_bytes: Some(100 * GIB),
            xpu_gate_proven: false,
        },
    }
}

fn failing_names(job: &LearnJob) -> Vec<String> {
    job.preflight_report()
        .unwrap()
        .checks
        .iter()
        .filter(|c| c.status != CheckStatus::Pass)
        .map(|c| c.name.clone())
        .collect()
}

struct Fixed {
    base: Scores,
    cand: Scores,
}
impl Evaluator for Fixed {
    fn id(&self) -> &str {
        "fixed-test-evaluator"
    }
    fn is_mock(&self) -> bool {
        true
    }
    fn evaluate(&self, _: &Path, _: &Path, _: &Path, _: &Path) -> io::Result<(Scores, Scores)> {
        Ok((self.base.clone(), self.cand.clone()))
    }
}
fn scores(c: f64) -> Scores {
    [
        ("coding_accuracy".to_string(), c),
        ("reasoning_accuracy".to_string(), 50.0),
    ]
    .into()
}
fn inputs(root: &Path) -> VerifyInputs {
    VerifyInputs {
        base_model: root.join("base.gguf"),
        candidate_model: Some(root.join("cand.gguf")),
    }
}

fn trained(root: &Path, id: &str) -> LearnJob {
    let mut job = LearnJob::new(spec(root, id, TrainerKind::Mock(MockBehavior::Succeed))).unwrap();
    job.preflight().unwrap();
    job.start().unwrap();
    job.wait().unwrap();
    job
}

#[test]
fn invalid_model_dir_blocks_cleanly() {
    let root = scratch("badmodel");
    let mut s = spec(&root, "j1", TrainerKind::Mock(MockBehavior::Succeed));
    s.model_dir = root.join("does_not_exist");
    let mut job = LearnJob::new(s).unwrap();
    let r = job.preflight().unwrap();
    assert_eq!(r["status"], "blocked");
    assert!(failing_names(&job).contains(&"model_trainable".to_string()));
    assert!(
        job.start().is_err(),
        "start must refuse without a passing preflight"
    );
}

#[test]
fn gguf_is_refused_with_actionable_message() {
    let root = scratch("gguf");
    let mut s = spec(&root, "j2", TrainerKind::Mock(MockBehavior::Succeed));
    let g = root.join("m.gguf");
    fs::write(&g, b"GGUF").unwrap();
    s.model_dir = g;
    let mut job = LearnJob::new(s.clone()).unwrap();
    job.preflight().unwrap();
    let reason = &job
        .preflight_report()
        .unwrap()
        .checks
        .iter()
        .find(|c| c.name == "model_trainable")
        .unwrap()
        .reason;
    assert!(
        reason.contains("trainable base") && reason.contains("GGUF"),
        "{reason}"
    );
    let dir = root.join("only_gguf");
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("m.gguf"), b"GGUF").unwrap();
    s.model_dir = dir;
    let mut job = LearnJob::new(s).unwrap();
    job.preflight().unwrap();
    assert!(failing_names(&job).contains(&"model_trainable".to_string()));
}

#[test]
fn insufficient_disk_blocks() {
    let root = scratch("disk");
    let mut s = spec(&root, "j3", TrainerKind::Mock(MockBehavior::Succeed));
    s.env.free_disk_bytes = Some(1024);
    let mut job = LearnJob::new(s).unwrap();
    assert_eq!(job.preflight().unwrap()["status"], "blocked");
    assert_eq!(failing_names(&job), vec!["disk_space".to_string()]);
}

#[test]
fn intel_xpu_trainer_path_is_unsupported() {
    let root = scratch("xpu");
    let mut s = spec(&root, "j4", TrainerKind::Mock(MockBehavior::Succeed));
    s.device_request = "auto".into();
    s.env.devices = Some(vec![DeviceInfo {
        vendor: "intel".into(),
        name: "Arc Pro B60".into(),
        vram_bytes: 24 * GIB,
        integrated: false,
    }]);
    let mut job = LearnJob::new(s).unwrap();
    job.preflight().unwrap();
    let pf = job.preflight_report().unwrap();
    let dev = pf.checks.iter().find(|c| c.name == "device").unwrap();
    assert_eq!(dev.status, CheckStatus::Fail);
    assert!(dev.reason.contains("UNPROVEN"), "{}", dev.reason);
    assert_eq!(pf.device.as_ref().unwrap().trainer_path, "xpu_unproven");
}

#[test]
fn devices_are_never_pooled_and_integrated_is_skipped() {
    let root = scratch("pool");
    let mut s = spec(&root, "j5", TrainerKind::Mock(MockBehavior::Succeed));
    s.device_request = "auto".into();
    s.env.devices = Some(vec![
        DeviceInfo {
            vendor: "intel".into(),
            name: "Arc Pro B60".into(),
            vram_bytes: 24 * GIB,
            integrated: false,
        },
        DeviceInfo {
            vendor: "nvidia".into(),
            name: "RTX 4060 Ti".into(),
            vram_bytes: 16 * GIB,
            integrated: false,
        },
        DeviceInfo {
            vendor: "intel".into(),
            name: "UHD Graphics".into(),
            vram_bytes: 0,
            integrated: true,
        },
    ]);
    let mut job = LearnJob::new(s).unwrap();
    job.preflight().unwrap();
    let d = job.preflight_report().unwrap().device.clone().unwrap();
    assert_eq!((d.family.as_str(), d.vram_bytes), ("cuda", 16 * GIB));
}

#[test]
fn heldout_overlap_and_missing_suite_block() {
    let root = scratch("overlap");
    let mut s = spec(&root, "j6", TrainerKind::Mock(MockBehavior::Succeed));
    s.dataset_file = make_dataset(
        &root,
        &[r#"{"instruction":"heldout probe 3","output":"leak"}"#],
    );
    let mut job = LearnJob::new(s.clone()).unwrap();
    job.preflight().unwrap();
    assert!(failing_names(&job).contains(&"heldout_overlap".to_string()));
    s.heldout_suite = None;
    let mut job = LearnJob::new(s).unwrap();
    job.preflight().unwrap();
    assert!(failing_names(&job).contains(&"heldout_suite".to_string()));
}

#[test]
fn positive_dry_run_mock_is_not_a_gpu_pass() {
    let root = scratch("positive");
    let mut job =
        LearnJob::new(spec(&root, "ok1", TrainerKind::Mock(MockBehavior::Succeed))).unwrap();
    let pre = job.preflight().unwrap();
    assert_eq!(pre["status"], "ok");
    let pf = job.preflight_report().unwrap();
    assert_eq!((pf.accepted_rows, pf.rejected_rows), (Some(10), Some(2)));
    assert!(
        pf.model_fingerprint.is_some()
            && pf.dataset_sha256.is_some()
            && pf.heldout_suite_sha256.is_some()
    );
    assert_eq!(job.start().unwrap()["status"], "started");
    let done = job.wait().unwrap();
    assert_eq!(done["trainer"], "mock");
    assert_eq!(done["mock"], true);
    assert_eq!(done["real_gpu_training"], "NOT_RUN");
    assert_eq!(done["train"]["peak_vram"], "NOT_RUN");
    assert_eq!(done["schema"], "ojon.learn.v1");
    assert!(matches!(job.state(), LearnState::Trained(_)));
}

#[test]
fn mock_run_writes_isolated_checkpoint_and_leaves_original_untouched() {
    let root = scratch("ckpt");
    let model_before = fs::read(root.join("base_hf").join("model.safetensors")).ok();
    let job = trained(&root, "ck1");
    let out = job.adapter_dir();
    for f in ["adapter_config.json", "adapter_model.safetensors"] {
        assert!(fs::metadata(out.join(f)).unwrap().len() > 0, "{f}");
    }
    assert!(!out.starts_with(root.join("base_hf")));
    assert!(
        model_before.is_none()
            || model_before == fs::read(root.join("base_hf").join("model.safetensors")).ok()
    );
    assert_eq!(
        fs::read(root.join("base_hf").join("model.safetensors")).unwrap(),
        vec![7u8; 4096]
    );
    assert!(job.job_dir().join("train_config.yaml").is_file());
    assert!(job.job_dir().join("receipts").is_dir());
}

#[test]
fn failed_trainer_mock_surfaces_error() {
    let root = scratch("fail");
    let mut job = LearnJob::new(spec(&root, "f1", TrainerKind::Mock(MockBehavior::Fail))).unwrap();
    job.preflight().unwrap();
    job.start().unwrap();
    assert!(job.wait().is_err());
    assert!(matches!(job.state(), LearnState::Failed(m) if m.contains("failed")));
    assert!(job
        .verify(
            &inputs(&root),
            &Fixed {
                base: scores(1.0),
                cand: scores(2.0)
            }
        )
        .is_err());
}

#[test]
fn nonzero_exit_from_supervised_subprocess_is_a_failure() {
    let root = scratch("script");
    let script = if cfg!(windows) {
        root.join("t.cmd")
    } else {
        root.join("t.sh")
    };
    fs::write(
        &script,
        if cfg!(windows) {
            "@echo off\r\nexit /b 3\r\n"
        } else {
            "#!/bin/sh\nexit 3\n"
        },
    )
    .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&script, fs::Permissions::from_mode(0o755)).unwrap();
    }
    let mut job = LearnJob::new(spec(&root, "s1", TrainerKind::MockScript(script))).unwrap();
    job.preflight().unwrap();
    job.start().unwrap();
    let e = job.wait().unwrap_err().to_string();
    assert!(e.contains("trainer failed"), "{e}");
    assert!(matches!(job.state(), LearnState::Failed(_)));
}

#[test]
fn cancelled_trainer_mock_stops_cleanly() {
    let root = scratch("cancel");
    let mut job = LearnJob::new(spec(&root, "c1", TrainerKind::Mock(MockBehavior::Hang))).unwrap();
    assert!(job.stop().is_err(), "nothing to stop yet");
    job.preflight().unwrap();
    job.start().unwrap();
    assert!(matches!(job.poll(), LearnState::Training));
    let r = job.stop().unwrap();
    assert_eq!(r["status"], "cancelled");
    assert!(matches!(job.state(), LearnState::Cancelled));
    assert!(job.promote().is_err());
}

#[test]
fn verify_without_candidate_is_not_run_and_cannot_promote() {
    let root = scratch("notrun");
    let mut job = trained(&root, "v1");
    let ev = Fixed {
        base: scores(1.0),
        cand: scores(9.0),
    };
    let r = job
        .verify(
            &VerifyInputs {
                base_model: root.join("b.gguf"),
                candidate_model: None,
            },
            &ev,
        )
        .unwrap();
    assert_eq!(r["status"], "not_run");
    assert_eq!(r["verify"]["status"], "NOT_RUN");
    assert!(job.promote().is_err());
}

#[test]
fn regression_fails_verification_and_blocks_promotion() {
    let root = scratch("regress");
    let mut job = trained(&root, "v2");
    let r = job
        .verify(
            &inputs(&root),
            &Fixed {
                base: scores(60.0),
                cand: scores(40.0),
            },
        )
        .unwrap();
    assert_eq!(r["verify"]["status"], "FAIL");
    assert!(job.promote().is_err());
}

#[test]
fn promote_records_scores_and_mock_tags() {
    let root = scratch("promote");
    let mut job = trained(&root, "p1");
    let v = job
        .verify(
            &inputs(&root),
            &Fixed {
                base: scores(40.0),
                cand: scores(55.0),
            },
        )
        .unwrap();
    assert_eq!(v["verify"]["status"], "PASS");
    assert_eq!(v["verify"]["evaluator_is_mock"], true);
    job.promote().unwrap();
    let cur = current_release(&root.join("work")).unwrap();
    assert_eq!(cur["job_id"], "p1");
    assert_eq!(cur["mock"], true);
    assert_eq!(cur["real_gpu_training"], "NOT_RUN");
    assert_eq!(cur["base_model_preserved"], true);
    assert!(cur["adapter_bytes"].as_u64().unwrap() > 0);
    assert_eq!(cur["scores"][0]["base"], 40.0);
}

#[test]
fn rollback_restores_prior_release() {
    let root = scratch("rollback");
    let work = root.join("work");
    let mut first = trained(&root, "r1");
    first
        .verify(
            &inputs(&root),
            &Fixed {
                base: scores(40.0),
                cand: scores(50.0),
            },
        )
        .unwrap();
    first.promote().unwrap();
    assert_eq!(current_release(&work).unwrap()["job_id"], "r1");

    let mut second =
        LearnJob::new(spec(&root, "r2", TrainerKind::Mock(MockBehavior::Succeed))).unwrap();
    second.preflight().unwrap();
    second.start().unwrap();
    second.wait().unwrap();
    second
        .verify(
            &inputs(&root),
            &Fixed {
                base: scores(40.0),
                cand: scores(60.0),
            },
        )
        .unwrap();
    second.promote().unwrap();
    assert_eq!(current_release(&work).unwrap()["job_id"], "r2");

    let rb = second.rollback().unwrap();
    assert_eq!(rb["restored_job_id"], "r1");
    assert_eq!(current_release(&work).unwrap()["job_id"], "r1");
    assert!(matches!(second.state(), LearnState::RolledBack));

    // Rolling back the last release returns to the untouched original model.
    let rb = rollback_release(&work).unwrap();
    assert_eq!(rb["active"], "original_base_model");
    assert!(current_release(&work).is_none());
    assert!(
        root.join("work").join("adapters").join("r1").is_dir(),
        "adapters are never deleted"
    );
}

#[test]
fn bad_job_id_is_rejected() {
    let root = scratch("id");
    assert!(LearnJob::new(spec(
        &root,
        "../escape",
        TrainerKind::Mock(MockBehavior::Succeed)
    ))
    .is_err());
}
