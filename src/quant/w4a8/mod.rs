//! W4A8 / W6A8 (comfy-kitchen grouped INT8 layout) converter, native Rust.
//! Output follows Comfy-Kitchen's safetensors layout. Actual inference support
//! also depends on the model architecture and the installed runtime backend.

pub mod codebook;
pub mod convert;
pub mod hadamard;
pub mod levels;
pub mod matrix;
pub mod numeric;
pub mod pack;
pub mod quantize;
pub mod st_writer;

pub use convert::{describe_plan, h3_default_layers, run, wan_default_layers, Request};
pub use quantize::Bits;
