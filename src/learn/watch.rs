//! Opt-in incremental session watcher.
//!
//! Stats configured directories and opens a session file only when its length
//! or modification time changed. Cursors are stored in `watch_state.json` after
//! a successful read. Collection never starts training.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use super::collector::{parse_session_file, LearnCollector};

const MAX_DEPTH: usize = 6;
const MAX_FILES: usize = 4000;
const STATE_SCHEMA: &str = "ojon.watch_state.v1";

/// One configured session directory.
#[derive(Clone, Debug)]
pub struct WatchSource {
    pub id: String,
    pub agent: String,
    pub path: PathBuf,
}

/// Result of one watcher poll. Safe to publish in the local daemon status.
#[derive(Clone, Debug, Default)]
pub struct WatchReport {
    pub sources_configured: usize,
    pub files_checked: usize,
    pub files_read: usize,
    pub new_episodes: usize,
    pub episodes_by_agent: BTreeMap<String, u64>,
    pub last_collection_epoch_s: Option<u64>,
    pub last_error: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct FileCursor {
    len: u64,
    mtime_ms: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct WatchState {
    schema: String,
    files: BTreeMap<String, FileCursor>,
    last_error: Option<String>,
}

impl Default for WatchState {
    fn default() -> Self {
        Self {
            schema: STATE_SCHEMA.to_string(),
            files: BTreeMap::new(),
            last_error: None,
        }
    }
}

/// Incremental harvester for the configured sources of one collector.
pub struct SessionWatcher {
    storage_dir: PathBuf,
    sources_path: Option<PathBuf>,
    sources: Vec<WatchSource>,
    state: WatchState,
}

impl SessionWatcher {
    /// `explicit` overrides `{storage_dir}/sources.json`. A missing default file
    /// leaves the daemon in HTTP-only mode.
    pub fn open(storage_dir: &Path, explicit: Option<&Path>) -> Self {
        let sources_path = match explicit {
            Some(path) => Some(path.to_path_buf()),
            None => {
                let default_path = storage_dir.join("sources.json");
                default_path.is_file().then_some(default_path)
            }
        };
        let state_path = storage_dir.join("watch_state.json");
        let mut watcher = Self {
            storage_dir: storage_dir.to_path_buf(),
            sources_path,
            sources: Vec::new(),
            state: load_state(&state_path),
        };
        let _ = watcher.reload_sources();
        watcher
    }

    /// Stat every configured source and ingest files whose cursor changed.
    pub fn poll(&mut self, collector: &LearnCollector) -> WatchReport {
        let mut errors = Vec::new();
        if let Err(err) = self.reload_sources() {
            errors.push(err);
        }
        let mut files_checked = 0usize;
        let mut files_read = 0usize;
        let mut new_episodes = 0usize;
        let mut live_keys = HashSet::new();
        let mut walked_roots = Vec::new();
        let storage_root = canonical_path(&self.storage_dir);

        for source in self.sources.clone() {
            if !source.path.exists() {
                errors.push(format!(
                    "{}: source path does not exist: {}",
                    source.id,
                    source.path.display()
                ));
                continue;
            }
            let root = canonical_path(&source.path);
            if root == storage_root || root.starts_with(&storage_root) {
                errors.push(format!(
                    "{}: refused to watch the collector storage directory",
                    source.id
                ));
                continue;
            }
            let mut files = Vec::new();
            if let Err(err) = collect_session_files(&root, &storage_root, &mut files) {
                errors.push(format!("{}: {err}", source.id));
                continue;
            }
            let capped = files.len() >= MAX_FILES;
            if capped {
                errors.push(format!(
                    "{}: file cap of {MAX_FILES} reached; remaining files stay unread",
                    source.id
                ));
            } else {
                walked_roots.push(root);
            }
            for file in files {
                files_checked += 1;
                let key = cursor_key(&file);
                let stamp = match file_stamp(&file) {
                    Ok(stamp) => stamp,
                    Err(err) => {
                        errors.push(format!("{}: {err}", file.display()));
                        continue;
                    }
                };
                live_keys.insert(key.clone());
                if self
                    .state
                    .files
                    .get(&key)
                    .is_some_and(|prev| prev.len == stamp.len && prev.mtime_ms == stamp.mtime_ms)
                {
                    continue;
                }
                match parse_session_file(&file, &source.agent) {
                    Ok(episodes) => {
                        files_read += 1;
                        let mut write_failed = false;
                        for episode in episodes {
                            match collector.record_episode(episode) {
                                Ok(true) => new_episodes += 1,
                                Ok(false) => {}
                                Err(err) => {
                                    write_failed = true;
                                    errors.push(format!("{}: {err}", file.display()));
                                }
                            }
                        }
                        // A failed append must be retried. Advancing the cursor
                        // here would skip the file until its bytes changed again.
                        if !write_failed {
                            self.state.files.insert(key, stamp);
                        }
                    }
                    Err(err) => errors.push(format!("{}: {err}", file.display())),
                }
            }
        }

        if !walked_roots.is_empty() {
            self.state.files.retain(|key, _| {
                let path = Path::new(key);
                let under_walked = walked_roots.iter().any(|root| path.starts_with(root));
                !under_walked || live_keys.contains(key)
            });
        }
        self.state.last_error = join_errors(&errors);
        if let Err(err) = self.save_state() {
            self.state.last_error = Some(match &self.state.last_error {
                Some(existing) => format!("{existing}; watch state: {err}"),
                None => format!("watch state: {err}"),
            });
        }

        WatchReport {
            sources_configured: self.sources.len(),
            files_checked,
            files_read,
            new_episodes,
            episodes_by_agent: collector.agent_counts(),
            last_collection_epoch_s: collector.last_collection_epoch_s(),
            last_error: self.state.last_error.clone(),
        }
    }

    fn reload_sources(&mut self) -> Result<(), String> {
        let Some(path) = &self.sources_path else {
            self.sources.clear();
            return Ok(());
        };
        match load_sources(path) {
            Ok(sources) => {
                self.sources = sources;
                Ok(())
            }
            Err(err) => Err(format!("sources file: {err}")),
        }
    }

    fn save_state(&self) -> io::Result<()> {
        let path = self.storage_dir.join("watch_state.json");
        let tmp = path.with_extension("tmp");
        {
            let mut file = File::create(&tmp)?;
            serde_json::to_writer_pretty(&mut file, &self.state)?;
            file.flush()?;
        }
        fs::rename(&tmp, &path)?;
        Ok(())
    }
}

fn load_state(path: &Path) -> WatchState {
    let Ok(text) = fs::read_to_string(path) else {
        return WatchState::default();
    };
    serde_json::from_str(&text).unwrap_or_default()
}

fn load_sources(path: &Path) -> io::Result<Vec<WatchSource>> {
    let text = fs::read_to_string(path)?;
    let value: Value = serde_json::from_str(&text)
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
    let entries = if let Some(list) = value.get("sources").and_then(Value::as_array) {
        list
    } else if let Some(list) = value.as_array() {
        list
    } else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "sources must be a list",
        ));
    };
    let mut sources = Vec::new();
    for entry in entries {
        let raw_path = entry
            .get("path")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim();
        if raw_path.is_empty() {
            continue;
        }
        let mut agent = entry
            .get("agent")
            .and_then(Value::as_str)
            .unwrap_or("auto")
            .trim()
            .to_ascii_lowercase();
        if agent.is_empty() {
            agent = "auto".to_string();
        }
        let id = entry.get("id").and_then(Value::as_str).unwrap_or("").trim();
        let id = if id.is_empty() {
            format!("{agent}-{}", sources.len() + 1)
        } else {
            id.to_string()
        };
        sources.push(WatchSource {
            id,
            agent,
            path: expand_path(raw_path),
        });
    }
    Ok(sources)
}

fn expand_path(raw: &str) -> PathBuf {
    if raw == "~" {
        return home_dir().unwrap_or_else(|| PathBuf::from(raw));
    }
    if let Some(rest) = raw.strip_prefix("~/").or_else(|| raw.strip_prefix("~\\")) {
        if let Some(home) = home_dir() {
            return home.join(rest);
        }
    }
    PathBuf::from(raw)
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .filter(|path| !path.as_os_str().is_empty())
}

fn canonical_path(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

fn cursor_key(path: &Path) -> String {
    canonical_path(path).to_string_lossy().into_owned()
}

fn file_stamp(path: &Path) -> io::Result<FileCursor> {
    let meta = fs::metadata(path)?;
    let mtime_ms = meta
        .modified()
        .ok()
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map(|elapsed| u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0);
    Ok(FileCursor {
        len: meta.len(),
        mtime_ms,
    })
}

fn is_session_file(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    if matches!(
        name,
        "daemon_status.json" | "sources.json" | "watch_state.json" | "collected_episodes.jsonl"
    ) {
        return false;
    }
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("json") || ext.eq_ignore_ascii_case("jsonl"))
}

fn collect_session_files(
    root: &Path,
    storage_root: &Path,
    out: &mut Vec<PathBuf>,
) -> io::Result<()> {
    fn walk(
        dir: &Path,
        storage_root: &Path,
        depth: usize,
        out: &mut Vec<PathBuf>,
    ) -> io::Result<()> {
        if out.len() >= MAX_FILES || depth > MAX_DEPTH {
            return Ok(());
        }
        if dir.is_file() {
            if is_session_file(dir) && !dir.starts_with(storage_root) {
                out.push(dir.to_path_buf());
            }
            return Ok(());
        }
        for entry in fs::read_dir(dir)? {
            if out.len() >= MAX_FILES {
                break;
            }
            let path = entry?.path();
            if path.starts_with(storage_root) {
                continue;
            }
            if path.is_dir() {
                let name = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("");
                if matches!(name, ".git" | "target" | "node_modules" | ".venv") {
                    continue;
                }
                walk(&path, storage_root, depth + 1, out)?;
            } else if is_session_file(&path) {
                if path.file_name().and_then(|name| name.to_str()) == Some("transcript.jsonl") {
                    let full = path.with_file_name("transcript_full.jsonl");
                    if full.is_file() {
                        continue;
                    }
                }
                out.push(path);
            }
        }
        Ok(())
    }
    walk(root, storage_root, 0, out)
}

fn join_errors(errors: &[String]) -> Option<String> {
    if errors.is_empty() {
        return None;
    }
    let mut text = errors.join("; ");
    if text.chars().count() > 500 {
        text = text.chars().take(500).collect();
        text.push_str("...");
    }
    Some(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learn::collector::EpisodeOutcome;
    use std::time::SystemTime;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "ojon-{name}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_sources(dir: &Path, body: &str) -> PathBuf {
        fs::create_dir_all(dir).unwrap();
        let path = dir.join("sources.json");
        fs::write(&path, body).unwrap();
        path
    }

    fn stored(dir: &Path) -> Vec<crate::learn::collector::CollectedEpisode> {
        let text = fs::read_to_string(dir.join("collected_episodes.jsonl")).unwrap_or_default();
        text.lines()
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect()
    }

    #[test]
    fn a_failed_append_is_retried_without_a_cursor() {
        let dir = temp_dir("watch-retry");
        let storage = dir.join("collector");
        let runs = dir.join("runs");
        fs::create_dir_all(&runs).unwrap();
        fs::write(
            runs.join("trace.jsonl"),
            "{\"episode_id\":\"e1\",\"session_id\":\"s\",\"agent\":\"pothik\",\"prompt\":\"one\",\"response\":\"done\"}\n",
        )
        .unwrap();
        let sources = write_sources(
            &storage,
            &format!(
                r#"{{"sources":[{{"id":"runs","agent":"auto","path":"{}"}}]}}"#,
                runs.display().to_string().replace('\\', "\\\\")
            ),
        );
        let collector = LearnCollector::new(storage.clone()).unwrap();
        let log = storage.join("collected_episodes.jsonl");
        fs::create_dir(&log).unwrap();

        let mut watcher = SessionWatcher::open(&storage, Some(&sources));
        let failed = watcher.poll(&collector);
        assert_eq!(failed.new_episodes, 0);
        assert!(failed.last_error.is_some());
        let state_text = fs::read_to_string(storage.join("watch_state.json")).unwrap();
        let state: serde_json::Value = serde_json::from_str(&state_text).unwrap();
        assert!(state["files"].as_object().unwrap().is_empty());

        fs::remove_dir(&log).unwrap();
        let recovered = watcher.poll(&collector);
        assert_eq!(recovered.new_episodes, 1, "{:?}", recovered.last_error);
        assert!(recovered.last_error.is_none());
        assert_eq!(stored(&storage).len(), 1);

        let again = watcher.poll(&collector);
        assert_eq!(again.files_read, 0);
        assert_eq!(again.new_episodes, 0);
        assert_eq!(stored(&storage).len(), 1);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn no_sources_file_does_not_scan() {
        let dir = temp_dir("watch-empty");
        let collector = LearnCollector::new(dir.join("collector")).unwrap();
        let mut watcher = SessionWatcher::open(collector.storage_dir(), None);
        let report = watcher.poll(&collector);
        assert_eq!(report.sources_configured, 0);
        assert_eq!(report.files_checked, 0);
        assert_eq!(report.files_read, 0);
        assert!(report.last_error.is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn multi_agent_ingest_is_incremental_and_keeps_failures() {
        let dir = temp_dir("watch-agents");
        let storage = dir.join("collector");
        let sessions = dir.join("sessions");
        fs::create_dir_all(sessions.join("pothik")).unwrap();
        fs::create_dir_all(sessions.join("dolil")).unwrap();
        fs::create_dir_all(sessions.join("agy")).unwrap();
        fs::write(
            sessions.join("pothik").join("run.json"),
            r#"{"run_id":"p1","agent":"pothik","model":"local","prompt":"fix ghp_1234567890abcdefghijklmnopqrstuvwx","response":"fixed","unit_tests_passed":true,"status":"passed"}"#,
        )
        .unwrap();
        fs::write(
            sessions.join("pothik").join("diff-only.json"),
            r#"{"run_id":"p2","agent":"pothik","prompt":"edit","response":"edited","status":"passed","git_diff":"@@ -1 +1 @@\n-a\n+b"}"#,
        )
        .unwrap();
        fs::write(
            sessions.join("dolil").join("run.jsonl"),
            "{\"event\":\"started\",\"goal\":\"compile the crate\"}\n{\"event\":\"model_error\",\"error\":\"connection refused by the local model\"}\n",
        )
        .unwrap();
        fs::write(
            sessions.join("agy").join("transcript.jsonl"),
            "{\"type\":\"USER_INPUT\",\"content\":\"add a function\",\"step_index\":1}\n{\"type\":\"PLANNER_RESPONSE\",\"content\":\"edited the file\",\"step_index\":2,\"tool_calls\":[{\"name\":\"edit_file\",\"arguments\":{\"Path\":\"src/a.rs\",\"Target\":\"old\",\"Replacement\":\"new\"}}]}\n",
        )
        .unwrap();
        let sources = write_sources(
            &storage,
            &format!(
                r#"{{"sources":[
                    {{"id":"pothik","agent":"pothik","path":"{}"}},
                    {{"id":"dolil","agent":"dolil","path":"{}"}},
                    {{"id":"agy","agent":"agy","path":"{}"}}
                ]}}"#,
                sessions
                    .join("pothik")
                    .display()
                    .to_string()
                    .replace('\\', "\\\\"),
                sessions
                    .join("dolil")
                    .display()
                    .to_string()
                    .replace('\\', "\\\\"),
                sessions
                    .join("agy")
                    .display()
                    .to_string()
                    .replace('\\', "\\\\"),
            ),
        );
        let collector = LearnCollector::new(storage.clone()).unwrap();
        let mut watcher = SessionWatcher::open(&storage, Some(&sources));
        let first = watcher.poll(&collector);
        assert_eq!(first.new_episodes, 4, "{:?}", first.last_error);
        assert_eq!(first.files_read, 4);
        assert_eq!(first.episodes_by_agent.get("pothik"), Some(&2));
        assert_eq!(first.episodes_by_agent.get("dolil"), Some(&1));
        assert_eq!(first.episodes_by_agent.get("agy"), Some(&1));
        assert!(first.last_collection_epoch_s.is_some());

        let episodes = stored(&storage);
        let diff_only = episodes
            .iter()
            .find(|ep| ep.episode_id.contains("p2"))
            .unwrap();
        assert_eq!(diff_only.outcome, EpisodeOutcome::Unverified);
        assert!(diff_only
            .git_diff
            .as_ref()
            .is_some_and(|diff| !diff.is_empty()));
        let verified = episodes
            .iter()
            .find(|ep| ep.episode_id.contains("p1"))
            .unwrap();
        assert_eq!(verified.outcome, EpisodeOutcome::VerifiedSuccess);
        assert!(verified.user_prompt.contains("[REDACTED_GITHUB_TOKEN]"));
        assert!(!verified.user_prompt.contains("ghp_"));
        let failed = episodes.iter().find(|ep| ep.app_id == "dolil").unwrap();
        assert_eq!(failed.outcome, EpisodeOutcome::FailedRetry);
        let rendered = serde_json::to_string(failed).unwrap();
        assert!(rendered.contains("connection refused by the local model"));
        let agy = episodes.iter().find(|ep| ep.app_id == "agy").unwrap();
        assert_eq!(agy.outcome, EpisodeOutcome::Unverified);
        assert_eq!(agy.tool_calls[0].name, "edit_file");
        assert!(agy.git_diff.is_some());

        let second = watcher.poll(&collector);
        assert_eq!(second.files_read, 0);
        assert_eq!(second.new_episodes, 0);
        assert_eq!(stored(&storage).len(), 4);

        let mut restarted = SessionWatcher::open(&storage, Some(&sources));
        let third = restarted.poll(&collector);
        assert_eq!(third.files_read, 0);
        assert_eq!(third.new_episodes, 0);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn incomplete_lines_rotation_and_restart_without_cursor_do_not_duplicate() {
        let dir = temp_dir("watch-cursor");
        let storage = dir.join("collector");
        let runs = dir.join("runs");
        fs::create_dir_all(&runs).unwrap();
        let log = runs.join("trace.jsonl");
        fs::write(
            &log,
            "{\"episode_id\":\"e1\",\"session_id\":\"s\",\"agent\":\"pothik\",\"prompt\":\"one\",\"response\":\"done\"}\n{\"episode_id\":\"e2\",\"prompt\":\"two\",\"response\":\"partial\n",
        )
        .unwrap();
        let sources = write_sources(
            &storage,
            &format!(
                r#"{{"sources":[{{"id":"runs","agent":"auto","path":"{}"}}]}}"#,
                runs.display().to_string().replace('\\', "\\\\")
            ),
        );
        let collector = LearnCollector::new(storage.clone()).unwrap();
        let mut watcher = SessionWatcher::open(&storage, Some(&sources));
        let first = watcher.poll(&collector);
        assert_eq!(first.new_episodes, 1, "{:?}", first.last_error);
        assert_eq!(stored(&storage)[0].episode_id, "e1");

        let mut file = fs::OpenOptions::new().append(true).open(&log).unwrap();
        writeln!(
            file,
            "\"}}\n{{\"episode_id\":\"e2\",\"session_id\":\"s\",\"agent\":\"pothik\",\"prompt\":\"two\",\"response\":\"finished\"}}"
        )
        .unwrap();
        file.flush().unwrap();
        drop(file);
        let second = watcher.poll(&collector);
        assert_eq!(second.new_episodes, 1, "{:?}", second.last_error);
        assert_eq!(stored(&storage).len(), 2);

        let third = watcher.poll(&collector);
        assert_eq!(third.files_read, 0);
        assert_eq!(third.new_episodes, 0);

        fs::write(
            &log,
            "{\"episode_id\":\"e3\",\"session_id\":\"s\",\"agent\":\"pothik\",\"prompt\":\"three\",\"response\":\"rotated\"}\n",
        )
        .unwrap();
        let rotated = watcher.poll(&collector);
        assert_eq!(rotated.new_episodes, 1);
        let ids: Vec<_> = stored(&storage)
            .into_iter()
            .map(|episode| episode.episode_id)
            .collect();
        assert_eq!(ids, vec!["e1", "e2", "e3"]);

        let mut restarted = SessionWatcher::open(&storage, Some(&sources));
        assert_eq!(restarted.poll(&collector).files_read, 0);

        fs::remove_file(storage.join("watch_state.json")).unwrap();
        let mut cold = SessionWatcher::open(&storage, Some(&sources));
        let reread = cold.poll(&collector);
        assert!(reread.files_read >= 1);
        assert_eq!(reread.new_episodes, 0);
        assert_eq!(stored(&storage).len(), 3);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_source_does_not_block_the_others() {
        let dir = temp_dir("watch-missing");
        let storage = dir.join("collector");
        let runs = dir.join("runs");
        fs::create_dir_all(&runs).unwrap();
        fs::write(
            runs.join("ok.json"),
            r#"{"run_id":"ok","agent":"pothik","prompt":"keep going","response":"done","unit_tests_passed":false,"status":"failed"}"#,
        )
        .unwrap();
        let missing = dir.join("does-not-exist");
        let sources = write_sources(
            &storage,
            &format!(
                r#"{{"sources":[{{"id":"gone","agent":"claude","path":"{}"}},{{"id":"runs","agent":"pothik","path":"{}"}}]}}"#,
                missing.display().to_string().replace('\\', "\\\\"),
                runs.display().to_string().replace('\\', "\\\\")
            ),
        );
        let collector = LearnCollector::new(storage.clone()).unwrap();
        let mut watcher = SessionWatcher::open(&storage, Some(&sources));
        let report = watcher.poll(&collector);
        assert_eq!(report.new_episodes, 1);
        assert!(report
            .last_error
            .as_ref()
            .is_some_and(|err| err.contains("gone")));
        assert_eq!(stored(&storage)[0].outcome, EpisodeOutcome::FailedRetry);
        assert_eq!(report.episodes_by_agent.get("pothik"), Some(&1));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn home_relative_sources_expand_without_a_machine_path() {
        let home = super::home_dir().expect("home");
        let expanded = super::expand_path("~/agent-sessions");
        assert_eq!(expanded, home.join("agent-sessions"));
        assert_ne!(expanded, PathBuf::from("~/agent-sessions"));
    }
}
