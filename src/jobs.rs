//! Capture factual, bounded, line-oriented subprocess output for the native GUI.
//! Both stdout and stderr are read concurrently to avoid pipe deadlocks.
//! No fake completion percentages, inferred GPU usage or synthetic progress.

use std::io::{self, BufRead, BufReader};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;

const MAX_LINE_BYTES: usize = 2048;
const MAX_TOTAL_LOG_BYTES: usize = 8 * 1024 * 1024;

/// Streams stdout/stderr text from an existing, explicitly selected backend.
/// The callback must be cheap; GUI callers should forward events to a channel.
/// Lines longer than MAX_LINE_BYTES are split into fragments to bound memory.
/// Total display output is limited; conversion continues even after that limit.
pub fn run_with_log<F>(cmd: &mut Command, mut emit: F) -> io::Result<()>
where
    F: FnMut(&str),
{
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let stdout = child.stdout.take().ok_or_else(|| io::Error::other("missing stdout pipe"))?;
    let stderr = child.stderr.take().ok_or_else(|| io::Error::other("missing stderr pipe"))?;
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
    for line in rx {
        if forwarded < MAX_TOTAL_LOG_BYTES {
            forwarded = forwarded.saturating_add(line.len());
            emit(&line);
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
        return Err(io::Error::other(format!("external backend exited with {status}")));
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
                if tx.send(format!("[{stream}] {line}")).is_err() { return Ok(()); }
                partial.clear();
            }
        }
        reader.consume(take);
        if newline.is_some() {
            let line = String::from_utf8_lossy(&partial);
            if !line.trim().is_empty() && tx.send(format!("[{stream}] {}", line.trim_end())).is_err() {
                return Ok(());
            }
            partial.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reports_failing_command_without_faking_success() {
        let mut command = if cfg!(windows) {
            let mut c = Command::new("cmd"); c.args(["/C","exit 7"]); c
        } else {
            let mut c = Command::new("sh"); c.args(["-c","exit 7"]); c
        };
        assert!(run_with_log(&mut command, |_| {}).is_err());
    }
}
