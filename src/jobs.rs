//! Capture factual, bounded, line-oriented subprocess output for the native GUI.
//! Both stdout and stderr are read concurrently to avoid pipe deadlocks.
//! No fake completion percentages, inferred GPU usage or synthetic progress.

use std::io::{self, BufRead, BufReader};
use std::process::{Child,Command, Output, Stdio};
use std::io::Read;
use std::time::{Duration,Instant};
use std::sync::mpsc;
use std::thread;

/// Peak memory is the model subprocess working-set high-water mark where
/// the host OS exposes it, not whole-system RAM, allocator commit or GPU VRAM.
#[cfg(windows)]
fn peak_process_ram(child:&Child)->Option<u64>{
    use std::os::windows::io::AsRawHandle;
    use std::ffi::c_void;
    #[repr(C)]
    struct Counters{
        cb:u32,faults:u32,peak_ws:usize,working_ws:usize,
        peak_paged:usize,paged:usize,peak_nonpaged:usize,nonpaged:usize,
        pagefile:usize,peak_pagefile:usize,
    }
    #[link(name="psapi")]
    extern "system" {
        fn GetProcessMemoryInfo(process:*mut c_void, counters:*mut Counters, cb:u32)->i32;
    }
    let mut value=Counters{cb:std::mem::size_of::<Counters>() as u32,
        faults:0,peak_ws:0,working_ws:0,peak_paged:0,paged:0,
        peak_nonpaged:0,nonpaged:0,pagefile:0,peak_pagefile:0};
    let size=value.cb;
    let success=unsafe{GetProcessMemoryInfo(child.as_raw_handle() as *mut c_void,
        &mut value,size)};
    if success==0{None}else{Some(value.peak_ws as u64)}
}
#[cfg(target_os="linux")]
fn peak_process_ram(child:&Child)->Option<u64>{
    let status=std::fs::read_to_string(format!("/proc/{}/status",child.id())).ok()?;
    let kb=status.lines().find(|v|v.starts_with("VmHWM:"))?
        .split_whitespace().nth(1)?.parse::<u64>().ok()?;
    kb.checked_mul(1024)
}
#[cfg(not(any(windows,target_os="linux")))]
fn peak_process_ram(_child:&Child)->Option<u64>{None}

pub struct CapturedProcess {
    pub output:Output,
    pub peak_ram_bytes:Option<u64>,
}

/// Bounded output for untrusted local model executables. Both pipes are
/// drained concurrently even after reaching the configured memory cap.
/// On timeout the child is killed and reaped; no success receipt is produced.
pub fn capture_with_timeout(
    cmd:&mut Command,limit:usize,timeout:Duration
)->io::Result<Output>{
    capture_with_timeout_measured(cmd,limit,timeout).map(|r|r.output)
}
/// Also records actual child-process peak working set, if available.
pub fn capture_with_timeout_measured(
    cmd:&mut Command,limit:usize,timeout:Duration
)->io::Result<CapturedProcess>{
    if limit==0||limit>16*1024*1024||timeout.is_zero(){
        return Err(io::Error::new(io::ErrorKind::InvalidInput,"invalid capture limit or timeout"));
    }
    let mut child=cmd.stdin(Stdio::null()).stdout(Stdio::piped())
        .stderr(Stdio::piped()).spawn()?;
    fn drain<R:Read>(mut pipe:R,limit:usize)->io::Result<(Vec<u8>,bool)>{
        let mut data=Vec::new();
        let mut extra=false;
        let mut block=[0u8;8192];
        loop{
            let n=pipe.read(&mut block)?;
            if n==0{break;}
            let room=limit.saturating_sub(data.len());
            data.extend_from_slice(&block[..n.min(room)]);
            if n>room{extra=true;}
        }
        Ok((data,extra))
    }
    let stdout=child.stdout.take().ok_or_else(||io::Error::other("missing stdout pipe"))?;
    let stderr=child.stderr.take().ok_or_else(||io::Error::other("missing stderr pipe"))?;
    let out=thread::spawn(move||drain(stdout,limit));
    let err=thread::spawn(move||drain(stderr,limit));
    let started=Instant::now();
    let mut timed_out=false;
    let mut peak_ram_bytes=peak_process_ram(&child);
    let status=loop{
        if let Some(value)=peak_process_ram(&child){
            peak_ram_bytes=Some(peak_ram_bytes.unwrap_or(0).max(value));
        }
        if let Some(status)=child.try_wait()?{break status;}
        if started.elapsed()>timeout{
            timed_out=true;
            let _=child.kill();
            break child.wait()?;
        }
        thread::sleep(Duration::from_millis(100));
    };
    let stdout=out.join().map_err(|_|io::Error::other("stdout worker panicked"))??;
    let stderr=err.join().map_err(|_|io::Error::other("stderr worker panicked"))??;
    if timed_out{return Err(io::Error::new(io::ErrorKind::TimedOut,"model backend exceeded execution timeout"));}
    if stdout.1||stderr.1{return Err(io::Error::new(io::ErrorKind::InvalidData,"backend output exceeded capture size limit"));}
    Ok(CapturedProcess{
        output:Output{status,stdout:stdout.0,stderr:stderr.0},
        peak_ram_bytes
    })
}

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
