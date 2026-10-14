//! Ojon's Kathamo-powered desktop frontend.
//! UI logic lives here; the bin target only launches the application.

mod workstation;

pub(crate) use workstation::run;
