//! Universal Headless Background Service & Resource Guard.
//!
//! Features:
//! 1. Lightweight, cross-platform background operation (Windows, Linux, macOS).
//! 2. Zero GUI dependency; operates independently of desktop applications.
//! 3. Bounded Resource Enforcer:
//!    - Monitors disk usage and enforces maximum storage limits.
//!    - Auto-prunes oldest unverified records while preserving static anchors and verified successes.
//! 4. Continuous Health Telemetry:
//!    - Writes atomic heartbeat to `daemon_status.json`.

use crate::learn::collector::LearnCollector;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Runtime configuration for the Ojon Learn Headless Daemon.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DaemonConfig {
    pub listen_port: u16,
    pub storage_dir: PathBuf,
    pub max_disk_mb: u64,
    pub max_memory_mb: u64,
    pub redaction_enabled: bool,
    pub poll_interval_s: u64,
    pub auto_supersede: bool,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            listen_port: 8089,
            storage_dir: PathBuf::from("collector"),
            max_disk_mb: 1024,
            max_memory_mb: 128,
            redaction_enabled: true,
            poll_interval_s: 30,
            auto_supersede: true,
        }
    }
}

/// Live telemetry reported by the running headless service.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DaemonStatus {
    pub schema: String,
    pub running: bool,
    pub uptime_seconds: u64,
    pub port: u16,
    pub storage_dir: String,
    pub current_disk_bytes: u64,
    pub max_disk_bytes: u64,
    pub episodes_recorded: u64,
    pub last_heartbeat_epoch_s: u64,
}

/// Headless Daemon Controller.
pub struct LearnDaemon {
    config: DaemonConfig,
    is_running: Arc<AtomicBool>,
    collector: Arc<LearnCollector>,
}

impl LearnDaemon {
    pub fn new(config: DaemonConfig) -> io::Result<Self> {
        fs::create_dir_all(&config.storage_dir)?;
        let collector = Arc::new(LearnCollector::new(config.storage_dir.clone())?);
        Ok(Self {
            config,
            is_running: Arc::new(AtomicBool::new(false)),
            collector,
        })
    }

    /// Calculate total directory size in bytes.
    pub fn calculate_disk_usage(dir: &Path) -> u64 {
        let mut total = 0u64;
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                if let Ok(meta) = entry.metadata() {
                    if meta.is_file() {
                        total += meta.len();
                    } else if meta.is_dir() {
                        total += Self::calculate_disk_usage(&entry.path());
                    }
                }
            }
        }
        total
    }

    /// Prune oldest Unverified (and then oldest FailedRetry) episodes from collected_episodes.jsonl
    /// while preserving VerifiedSuccess and static anchor episodes when storage exceeds budget.
    pub fn prune_unverified_episodes_to_budget(
        storage_dir: &Path,
        max_bytes: u64,
    ) -> io::Result<usize> {
        let episodes_path = storage_dir.join("collected_episodes.jsonl");
        if !episodes_path.is_file() {
            return Ok(0);
        }
        let mut episodes = crate::learn::curate::load_episodes_from_jsonl(&episodes_path)?;
        if episodes.is_empty() {
            return Ok(0);
        }
        let initial_len = episodes.len();
        let mut current_bytes = Self::calculate_disk_usage(storage_dir);

        while current_bytes > max_bytes {
            // First try to drop the oldest Unverified episode
            let unverified_idx = episodes.iter().position(|e| {
                e.outcome == crate::learn::collector::EpisodeOutcome::Unverified
                    && e.app_id != "anchor"
                    && e.session_id != "static-anchor-session"
            });
            let remove_idx = unverified_idx.or_else(|| {
                episodes.iter().position(|e| {
                    e.outcome == crate::learn::collector::EpisodeOutcome::FailedRetry
                        && e.app_id != "anchor"
                        && e.session_id != "static-anchor-session"
                })
            });
            let Some(idx) = remove_idx else {
                break; // Only VerifiedSuccess or anchor episodes remain; never delete verified gold data
            };
            let removed = episodes.remove(idx);
            let approx_line_bytes = serde_json::to_string(&removed)
                .map(|s| s.len() as u64 + 1)
                .unwrap_or(256);
            current_bytes = current_bytes.saturating_sub(approx_line_bytes);
        }

        let pruned_count = initial_len.saturating_sub(episodes.len());
        if pruned_count > 0 {
            let tmp_path = episodes_path.with_extension("jsonl.prune.tmp");
            {
                let mut f = File::create(&tmp_path)?;
                for ep in &episodes {
                    if let Ok(line) = serde_json::to_string(ep) {
                        writeln!(f, "{line}")?;
                    }
                }
                f.flush()?;
            }
            fs::rename(&tmp_path, &episodes_path)?;
        }
        Ok(pruned_count)
    }

    /// Write heartbeat status file. `running` is false for the shutdown record.
    pub fn write_heartbeat(
        status_path: &Path,
        port: u16,
        storage_dir: &str,
        uptime_s: u64,
        disk_bytes: u64,
        max_bytes: u64,
        episodes: u64,
        running: bool,
    ) -> io::Result<()> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let status = DaemonStatus {
            schema: "ojon.daemon_status.v1".to_string(),
            running,
            uptime_seconds: uptime_s,
            port,
            storage_dir: storage_dir.to_string(),
            current_disk_bytes: disk_bytes,
            max_disk_bytes: max_bytes,
            episodes_recorded: episodes,
            last_heartbeat_epoch_s: now,
        };

        let temp_path = status_path.with_extension("tmp");
        {
            let mut file = File::create(&temp_path)?;
            serde_json::to_writer_pretty(&mut file, &status)?;
            file.flush()?;
        }
        fs::rename(temp_path, status_path)?;
        Ok(())
    }

    /// Start the headless background service and worker loop.
    pub fn start(&self) -> io::Result<thread::JoinHandle<()>> {
        self.is_running.store(true, Ordering::SeqCst);
        let port = self.config.listen_port;
        let collector_handle = LearnCollector::start_server(self.collector.clone(), port)?;

        let is_running = self.is_running.clone();
        let config = self.config.clone();
        let collector = self.collector.clone();

        let daemon_handle = thread::spawn(move || {
            let start_time = Instant::now();
            let status_file = config.storage_dir.join("daemon_status.json");
            let max_bytes = config.max_disk_mb * 1024 * 1024;

            while is_running.load(Ordering::SeqCst) {
                let uptime_s = start_time.elapsed().as_secs();
                let disk_bytes = Self::calculate_disk_usage(&config.storage_dir);
                let stats = collector.get_stats(port);

                let _ = Self::write_heartbeat(
                    &status_file,
                    port,
                    &config.storage_dir.to_string_lossy(),
                    uptime_s,
                    disk_bytes,
                    max_bytes,
                    stats.total_episodes,
                    true,
                );

                // Enforce bounded disk storage: if size exceeds limit, purge transient tmp files
                // and prune oldest Unverified episodes while preserving VerifiedSuccess episodes.
                if disk_bytes > max_bytes {
                    let tmp_dir = config.storage_dir.join("tmp");
                    if tmp_dir.exists() {
                        let _ = fs::remove_dir_all(&tmp_dir);
                    }
                    let after_tmp = Self::calculate_disk_usage(&config.storage_dir);
                    if after_tmp > max_bytes {
                        let _ = Self::prune_unverified_episodes_to_budget(
                            &config.storage_dir,
                            max_bytes,
                        );
                    }
                }

                // Sleep in small increments to allow rapid shutdown
                for _ in 0..(config.poll_interval_s.max(1) * 10) {
                    if !is_running.load(Ordering::SeqCst) {
                        break;
                    }
                    thread::sleep(Duration::from_millis(100));
                }
            }

            let uptime_s = start_time.elapsed().as_secs();
            let disk_bytes = Self::calculate_disk_usage(&config.storage_dir);
            let episodes = collector.get_stats(port).total_episodes;
            let _ = Self::write_heartbeat(
                &status_file,
                port,
                &config.storage_dir.to_string_lossy(),
                uptime_s,
                disk_bytes,
                max_bytes,
                episodes,
                false,
            );
            collector.stop();
            let _ = collector_handle.join();
        });

        Ok(daemon_handle)
    }

    /// Signal daemon to stop.
    pub fn stop(&self) {
        self.is_running.store(false, Ordering::SeqCst);
        self.collector.stop();
    }
}

/// Prune unverified episodes when disk usage exceeds `target_max_bytes`.
pub fn prune_unverified_episodes_to_budget(
    storage_dir: &Path,
    target_max_bytes: u64,
) -> io::Result<usize> {
    LearnDaemon::prune_unverified_episodes_to_budget(storage_dir, target_max_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_daemon_heartbeat_and_disk_usage() {
        let temp_dir = std::env::temp_dir().join(format!(
            "ojon_daemon_test_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&temp_dir).unwrap();

        let dummy_file = temp_dir.join("test.bin");
        fs::write(&dummy_file, vec![0u8; 4096]).unwrap();

        let bytes = LearnDaemon::calculate_disk_usage(&temp_dir);
        assert!(bytes >= 4096);

        let status_file = temp_dir.join("daemon_status.json");
        LearnDaemon::write_heartbeat(
            &status_file,
            8089,
            &temp_dir.to_string_lossy(),
            10,
            bytes,
            1024 * 1024,
            5,
            true,
        )
        .unwrap();

        assert!(status_file.is_file());
        let content = fs::read_to_string(&status_file).unwrap();
        let status: DaemonStatus = serde_json::from_str(&content).unwrap();
        assert!(status.running);
        assert_eq!(status.port, 8089);
        assert_eq!(status.episodes_recorded, 5);

        LearnDaemon::write_heartbeat(
            &status_file,
            8089,
            &temp_dir.to_string_lossy(),
            11,
            bytes,
            1024 * 1024,
            5,
            false,
        )
        .unwrap();
        let stopped: DaemonStatus =
            serde_json::from_str(&fs::read_to_string(&status_file).unwrap()).unwrap();
        assert!(!stopped.running);

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_prune_unverified_episodes_preserves_verified_anchors() {
        let temp_dir = std::env::temp_dir().join(format!(
            "ojon_daemon_prune_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&temp_dir).unwrap();
        let coll = LearnCollector::new(temp_dir.clone()).unwrap();

        let mut verified = crate::learn::collector::CollectedEpisode::new(
            "ep-verified-1",
            "s-1",
            "pothik",
            "qwen3",
            "Write tests",
            "assert_eq!(2+2, 4);",
        );
        verified.outcome = crate::learn::collector::EpisodeOutcome::VerifiedSuccess;
        verified.unit_tests_passed = Some(true);
        coll.record_episode(verified).unwrap();

        for i in 0..5 {
            let unv = crate::learn::collector::CollectedEpisode::new(
                format!("ep-unv-{i}"),
                format!("s-unv-{i}"),
                "pothik",
                "qwen3",
                "Long unverified chat ".repeat(20),
                "Unverified reply ".repeat(20),
            );
            coll.record_episode(unv).unwrap();
        }

        let before = LearnDaemon::calculate_disk_usage(&temp_dir);
        // Set target budget to half of current disk usage
        let pruned = prune_unverified_episodes_to_budget(&temp_dir, before / 2).unwrap();
        assert!(pruned >= 1);

        let mut coll2 = LearnCollector::new(temp_dir.clone()).unwrap();
        coll2.refresh_counts_from_disk().unwrap();
        let st = coll2.get_stats(8089);
        assert_eq!(st.verified_success_count, 1);
        assert!(st.unverified_count < 5);

        let _ = fs::remove_dir_all(&temp_dir);
    }
}
