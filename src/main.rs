//! Ojon (ওজন) — native quantization, uncensoring, and continuous model improvement CLI.
mod cli;

fn main() {
    if let Err(error) = cli::run() {
        eprintln!("Ojon error: {error}");
        eprintln!("Run 'ojon --help' for usage.");
        std::process::exit(1);
    }
}
