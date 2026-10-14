//! Kathamo desktop frontend for Ojon.
//! Business logic lives in ojon_backend; this module owns the user interface.
//! Cargo exposes the GUI through a minimal src/bin/ launcher.

mod workstation;

/// Entry point used by the ojon-gui Cargo executable.
pub(crate) fn run() -> Result<(), Box<dyn std::error::Error>> {
    workstation::run()
}
