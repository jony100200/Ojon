//! W4A8 / W6A8 (comfy-kitchen grouped INT8 layout) converter, native Rust.
//! Output loads in ComfyUI (comfy-kitchen), Rupokar (KS sd.cpp) and Probaho.

pub mod codebook;
pub mod convert;
pub mod hadamard;
pub mod levels;
pub mod matrix;
pub mod numeric;
pub mod pack;
pub mod quantize;
pub mod st_writer;

pub use convert::{h3_default_layers, run, Request};
pub use quantize::Bits;
