use std::path::{Path, PathBuf};
use std::process::Command;

fn find_windows_rc() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("RC") {
        let p = PathBuf::from(path);
        if p.is_file() {
            return Some(p);
        }
    }
    if let Some(paths) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&paths) {
            let candidate = dir.join("rc.exe");
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    for env_key in ["ProgramFiles(x86)", "ProgramFiles"] {
        let Ok(pf) = std::env::var(env_key) else { continue };
        let kits = Path::new(&pf).join("Windows Kits").join("10").join("bin");
        let Ok(entries) = std::fs::read_dir(&kits) else { continue };
        let mut versions: Vec<PathBuf> = entries
            .filter_map(Result::ok)
            .map(|e| e.path().join("x64").join("rc.exe"))
            .filter(|p| p.is_file())
            .collect();
        versions.sort();
        if let Some(latest) = versions.pop() {
            return Some(latest);
        }
    }
    None
}

fn main() {
    println!("cargo:rerun-if-changed=assets/ojon.ico");
    println!("cargo:rerun-if-changed=assets/ojon.rc");

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows")
        && std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc")
    {
        let rc_path = Path::new("assets/ojon.rc");
        let ico_path = Path::new("assets/ojon.ico");
        if rc_path.is_file() && ico_path.is_file() {
            if let Some(rc) = find_windows_rc() {
                let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
                let res = out_dir.join("ojon.res");
                let status = Command::new(&rc)
                    .args(["/nologo", "/fo"])
                    .arg(&res)
                    .arg(rc_path)
                    .status();
                if let Ok(st) = status {
                    if st.success() && res.is_file() {
                        println!("cargo:rustc-link-arg-bin=ojon-gui={}", res.display());
                    }
                }
            }
        }
    }
}
