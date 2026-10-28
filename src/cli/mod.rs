//! Ojon CLI dispatch router and domain submodules.

pub mod common;
pub mod diffusion;
pub mod flint;
pub mod inspect;
pub mod learn;
pub mod quant;
pub mod uncensor;

use std::env;
use std::ffi::OsString;
use std::io;

pub fn run() -> io::Result<()> {
    let mut args = env::args_os().skip(1);
    let cmd_os = args.next().unwrap_or_else(|| OsString::from("--help"));
    let cmd = match cmd_os.to_str() {
        Some(c) => c,
        None => return Err(common::invalid("command must be valid Unicode")),
    };

    if inspect::handle(cmd, &mut args)? {
        return Ok(());
    }
    if quant::handle(cmd, &mut args)? {
        return Ok(());
    }
    if uncensor::handle(cmd, &mut args)? {
        return Ok(());
    }
    if flint::handle(cmd, &mut args)? {
        return Ok(());
    }
    if diffusion::handle(cmd, &mut args)? {
        return Ok(());
    }
    if learn::handle(cmd, &mut args)? {
        return Ok(());
    }

    match cmd {
        "--help" | "-h" | "help" => common::usage(),
        _ => {
            common::usage();
            return Err(common::invalid("unknown command"));
        }
    }
    Ok(())
}
