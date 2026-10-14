//! Cargo binary entry point. The Kathamo UI itself lives in src/frontend/.
//! src/bin means "binary executable target" in Rust, not a backend directory.
#[path = "../frontend/mod.rs"]
mod frontend;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    frontend::run()
}
