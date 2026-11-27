//! EchoMimicV3-Flash Rust-only checkpoint inspection and weight quantisation.
//! This creates data artifacts, not a tested inference/runtime format.
use crate::models::weight_patch::{read_safetensors_header, TensorLoc};
use crate::quant::packed::{decode_float, encode_float};
use crate::quant::w4a8::st_writer::{header, Entry};
use pytorch_reader::PytorchReader;
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, BufWriter, ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::Path;

const MAX_TENSOR_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const COMPONENTS: [(&str, &str, bool); 6] = [
    (
        "t5",
        "Wan2.1-Fun-V1.1-1.3B-InP/models_t5_umt5-xxl-enc-bf16.pth",
        true,
    ),
    (
        "clip",
        "Wan2.1-Fun-V1.1-1.3B-InP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        true,
    ),
    (
        "wan",
        "Wan2.1-Fun-V1.1-1.3B-InP/diffusion_pytorch_model.safetensors",
        true,
    ),
    (
        "flash",
        "echomimicv3-flash-pro/diffusion_pytorch_model.safetensors",
        true,
    ),
    ("vae", "Wan2.1-Fun-V1.1-1.3B-InP/Wan2.1_VAE.pth", false),
    ("audio", "chinese-wav2vec2-base/pytorch_model.bin", false),
];
fn invalid(s: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, s.into())
}
fn bad(s: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, s.into())
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bits {
    Eight,
    Four,
}
impl Bits {
    pub fn parse(s: &str) -> io::Result<Self> {
        match s {
            "8" => Ok(Self::Eight),
            "4" => Ok(Self::Four),
            _ => Err(invalid("--bits must be 8 or 4")),
        }
    }
    fn count(self) -> usize {
        if self == Self::Eight {
            8
        } else {
            4
        }
    }
    fn label(self) -> &'static str {
        if self == Self::Eight {
            "int8_symmetric_row"
        } else {
            "int4_symmetric_row"
        }
    }
}
#[derive(Clone, Debug)]
struct Tensor {
    name: String,
    dtype: &'static str,
    shape: Vec<usize>,
    bytes: u64,
}
impl Tensor {
    fn candidate(&self) -> bool {
        let name = self.name.as_str();
        let transformer = name.contains("blocks.")
            || name.contains("encoder.block.")
            || name.contains("decoder.block.");
        let linear_weight = name.ends_with(".weight") || name.ends_with(".in_proj_weight");
        let sensitive = [
            "embed",
            "norm",
            "position",
            "relative_attention_bias",
            "audio",
            "head.",
            "time_embedding",
        ]
        .iter()
        .any(|part| name.to_ascii_lowercase().contains(part));
        transformer
            && linear_weight
            && !sensitive
            && matches!(self.dtype, "F16" | "BF16" | "F32")
            && matches!(self.shape.as_slice(), [r, c] if *r > 0 && *c >= 16 && *c % 2 == 0)
    }
    fn output_bytes(&self, bits: Bits) -> io::Result<u64> {
        let n = self.shape[0]
            .checked_mul(self.shape[1])
            .ok_or_else(|| bad("tensor shape overflow"))?;
        Ok((n as u64) * bits.count() as u64 / 8)
    }
}
fn dtype(s: &str) -> io::Result<&'static str> {
    Ok(match s {
        "BF16" => "BF16",
        "F16" => "F16",
        "F32" => "F32",
        "F64" => "F64",
        "I8" => "I8",
        "U8" => "U8",
        "I16" => "I16",
        "U16" => "U16",
        "I32" => "I32",
        "U32" => "U32",
        "I64" => "I64",
        "U64" => "U64",
        "Bool" | "BOOL" => "BOOL",
        _ => return Err(bad(format!("unsupported tensor dtype {s}"))),
    })
}
enum Checkpoint {
    Torch(PytorchReader),
    Safe {
        file: File,
        locations: BTreeMap<String, TensorLoc>,
    },
}
impl Checkpoint {
    fn open(path: &Path) -> io::Result<Self> {
        if path
            .extension()
            .is_some_and(|s| s.eq_ignore_ascii_case("safetensors"))
        {
            Ok(Self::Safe {
                file: File::open(path)?,
                locations: read_safetensors_header(path)?,
            })
        } else {
            PytorchReader::new(path)
                .map(Self::Torch)
                .map_err(|e| bad(format!("{}: {e}", path.display())))
        }
    }
    fn tensors(&self) -> io::Result<Vec<Tensor>> {
        let mut out = Vec::new();
        match self {
            Self::Torch(r) => {
                for t in r.tensors().values() {
                    out.push(Tensor {
                        name: t.name.clone(),
                        dtype: dtype(&format!("{:?}", t.dtype()))?,
                        shape: t.shape().to_vec(),
                        bytes: t.byte_len() as u64,
                    });
                }
            }
            Self::Safe { locations, .. } => {
                for (name, t) in locations {
                    out.push(Tensor {
                        name: name.clone(),
                        dtype: dtype(&t.dtype)?,
                        shape: t.shape.clone(),
                        bytes: t.bytes,
                    });
                }
            }
        }
        out.sort_by(|a, b| a.name.cmp(&b.name));
        if out.is_empty() {
            return Err(bad("no tensors in checkpoint"));
        }
        Ok(out)
    }
    fn read(&mut self, t: &Tensor) -> io::Result<Vec<u8>> {
        if t.bytes > MAX_TENSOR_BYTES {
            return Err(invalid(format!(
                "{} exceeds 2 GiB safe-memory limit",
                t.name
            )));
        }
        match self {
            Self::Torch(r) => r
                .get(&t.name)
                .ok_or_else(|| bad("missing PyTorch tensor"))?
                .read()
                .map_err(|e| bad(format!("{}: {e}", t.name))),
            Self::Safe { file, locations } => {
                let loc = locations
                    .get(&t.name)
                    .ok_or_else(|| bad("missing safetensors tensor"))?;
                file.seek(SeekFrom::Start(loc.offset))?;
                let mut bytes = vec![0; t.bytes as usize];
                file.read_exact(&mut bytes)?;
                Ok(bytes)
            }
        }
    }
    fn copy_original(&mut self, t: &Tensor, out: &mut impl Write) -> io::Result<()> {
        match self {
            Self::Safe { file, locations } => {
                let loc = locations
                    .get(&t.name)
                    .ok_or_else(|| bad("missing tensor"))?;
                file.seek(SeekFrom::Start(loc.offset))?;
                let copied = io::copy(&mut file.take(t.bytes), out)?;
                if copied != t.bytes {
                    return Err(bad("source file unexpectedly truncated"));
                }
                Ok(())
            }
            Self::Torch(_) => {
                let bytes = self.read(t)?;
                out.write_all(&bytes)
            }
        }
    }
}
pub fn inspect(root: &Path, bits: Bits) -> io::Result<Value> {
    let mut components = Vec::new();
    for (name, relative, convert) in COMPONENTS {
        let path = root.join(relative);
        let size = fs::metadata(&path)?.len();
        if convert {
            let tensors = Checkpoint::open(&path)?.tensors()?;
            let candidates: Vec<_> = tensors.iter().filter(|t| t.candidate()).collect();
            if candidates.is_empty() {
                return Err(invalid(format!(
                    "{name}: no candidate linear layers; do not emit empty quant"
                )));
            }
            let saved = candidates
                .iter()
                .try_fold(0u64, |acc, t| -> io::Result<u64> {
                    Ok(acc
                        + t.bytes
                            .saturating_sub(t.output_bytes(bits)? + (t.shape[0] as u64 * 4)))
                })?;
            components.push(json!({"component":name,"path":path,"bytes":size,
                "tensors":tensors.len(),"linear_layers":candidates.len(),"estimated_saving_bytes":saved}));
        } else {
            let tensors = Checkpoint::open(&path)?.tensors()?;
            components.push(json!({"component":name,"path":path,"bytes":size,"tensors":tensors.len(),"preserved":true}));
        }
    }
    Ok(
        json!({"schema":"ojon.echo.plan.v1","precision":bits.label(),"components":components,
        "source_root":root,"runtime_verified":false}),
    )
}
fn quantize_rows(
    t: &Tensor,
    bytes: &[u8],
    bits: Bits,
    writer: &mut impl Write,
) -> io::Result<Vec<f32>> {
    let (rows, cols) = (t.shape[0], t.shape[1]);
    let width = match t.dtype {
        "BF16" | "F16" => 2,
        "F32" => 4,
        _ => return Err(bad("unhandled source precision")),
    };
    let size = rows
        .checked_mul(cols)
        .and_then(|n| n.checked_mul(width))
        .ok_or_else(|| bad("overflow"))?;
    if bytes.len() != size {
        return Err(bad("unexpected input bytes"));
    }
    let mut scales = Vec::with_capacity(rows);
    for row in bytes.chunks_exact(cols * width) {
        let mut values = Vec::with_capacity(cols);
        let mut max_abs = 0.0f32;
        for x in row.chunks_exact(width) {
            let val = decode_float(x, t.dtype)?;
            max_abs = max_abs.max(val.abs());
            values.push(val);
        }
        let limit = if bits == Bits::Eight { 127.0 } else { 7.0 };
        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / limit };
        scales.push(scale);
        if bits == Bits::Eight {
            for v in values {
                // Use the normalized ratio: dividing by a rounded FP32 scale
                // can shift exact half-step values across a rounding boundary.
                let q = (if max_abs == 0.0 {
                    0.0
                } else {
                    v / max_abs * limit
                })
                .round()
                .clamp(-127.0, 127.0) as i8;
                writer.write_all(&[q as u8])?;
            }
        } else {
            for pair in values.chunks_exact(2) {
                let normalized = |v: f32| {
                    if max_abs == 0.0 {
                        0.0
                    } else {
                        v / max_abs * limit
                    }
                };
                let lo = normalized(pair[0]).round().clamp(-7.0, 7.0) as i8 as u8 & 15;
                let hi = normalized(pair[1]).round().clamp(-7.0, 7.0) as i8 as u8 & 15;
                writer.write_all(&[lo | (hi << 4)])?;
            }
        }
    }
    Ok(scales)
}

/// Decode a single packed Echo linear-weight row back into the original float
/// precision. A future runtime loader can request rows independently; this does
/// not imply that the current Echo Python runtime supports packed GPU weights.
pub fn decode_packed_row(
    packed: &[u8],
    scale: f32,
    cols: usize,
    bits: Bits,
    target_dtype: &str,
) -> io::Result<Vec<u8>> {
    if cols == 0 || (bits == Bits::Four && cols % 2 != 0) {
        return Err(invalid("packed Echo row needs a nonzero, even column count for INT4"));
    }
    if !scale.is_finite() || scale <= 0.0 {
        return Err(bad("invalid packed Echo row scale"));
    }
    let width = match target_dtype {
        "F16" | "BF16" => 2,
        "F32" => 4,
        _ => return Err(invalid("decoder target must be F16, BF16 or F32")),
    };
    let expected = cols.checked_mul(bits.count()).and_then(|n| n.checked_div(8))
        .ok_or_else(|| bad("packed row size overflow"))?;
    if packed.len() != expected {
        return Err(bad("packed Echo row byte length mismatch"));
    }
    let mut out = Vec::with_capacity(cols.checked_mul(width).ok_or_else(|| bad("decoded row size overflow"))?);
    for c in 0..cols {
        let q = if bits == Bits::Eight {
            packed[c] as i8
        } else {
            let nibble = if c % 2 == 0 { packed[c / 2] & 15 } else { packed[c / 2] >> 4 };
            ((nibble as i8) << 4) >> 4
        };
        out.extend_from_slice(&encode_float(q as f32 * scale, target_dtype)?);
    }
    Ok(out)
}

/// Read quantized Echo weights a row at a time from an existing Ojon-produced
/// safetensors file. This is a Rust decoding primitive, not a GPU quantized
/// linear layer; the caller supplies the source dtype from the bundle manifest.
pub struct PackedEchoReader {
    file: File,
    tensors: BTreeMap<String, TensorLoc>,
    bits: Bits,
}

impl PackedEchoReader {
    pub fn open(path: &Path, bits: Bits) -> io::Result<Self> {
        let tensors = read_safetensors_header(path)?;
        if !tensors.keys().any(|name| name.ends_with(".ojon_scale")) {
            return Err(bad("not an Ojon Echo packed checkpoint"));
        }
        Ok(Self { file: File::open(path)?, tensors, bits })
    }

    /// Decode only one row; never materialize an entire matrix or a model copy.
    pub fn decode_weight_row(
        &mut self, weight_name: &str, row: usize, logical_cols: usize, original_dtype: &str,
    ) -> io::Result<Vec<u8>> {
        let loc = self.tensors.get(weight_name)
            .ok_or_else(|| invalid(format!("missing weight {weight_name}")))?;
        let scale_name = format!("{weight_name}.ojon_scale");
        let scales = self.tensors.get(&scale_name)
            .ok_or_else(|| invalid(format!("missing scales {scale_name}")))?;
        let expected_dtype = if self.bits == Bits::Eight { "I8" } else { "U8" };
        let row_bytes = logical_cols.checked_mul(self.bits.count())
            .and_then(|n| n.checked_div(8)).ok_or_else(|| bad("invalid packed weight shape"))?;
        let expected_data_bytes = (loc.shape.first().copied().unwrap_or(0) as u64)
            .checked_mul(row_bytes as u64).ok_or_else(|| bad("packed weight size overflow"))?;
        let expected_scale_bytes = (loc.shape.first().copied().unwrap_or(0) as u64)
            .checked_mul(4).ok_or_else(|| bad("packed scale size overflow"))?;
        if loc.dtype != expected_dtype || loc.shape.len() != 2
            || loc.shape[1] != row_bytes || row >= loc.shape[0]
            || scales.dtype != "F32" || scales.shape != [loc.shape[0]]
            || loc.bytes != expected_data_bytes || scales.bytes != expected_scale_bytes
            || row_bytes > 16 * 1024 * 1024
            || (self.bits == Bits::Four && logical_cols % 2 != 0)
        {
            return Err(bad("packed Echo weight/scale shape or precision mismatch"));
        }
        let mut packed = vec![0; row_bytes];
        let weight_offset = (row as u64).checked_mul(row_bytes as u64)
            .and_then(|delta| loc.offset.checked_add(delta))
            .ok_or_else(|| bad("packed Echo weight offset overflow"))?;
        self.file.seek(SeekFrom::Start(weight_offset))?;
        self.file.read_exact(&mut packed)?;
        let scale_offset = (row as u64).checked_mul(4)
            .and_then(|delta| scales.offset.checked_add(delta))
            .ok_or_else(|| bad("packed Echo scale offset overflow"))?;
        self.file.seek(SeekFrom::Start(scale_offset))?;
        let mut raw_scale = [0u8; 4];
        self.file.read_exact(&mut raw_scale)?;
        decode_packed_row(&packed, f32::from_le_bytes(raw_scale), logical_cols, self.bits, original_dtype)
    }
}
fn convert(src: &Path, dst: &Path, bits: Bits) -> io::Result<Value> {
    let mut original = Checkpoint::open(src)?;
    let tensors = original.tensors()?;
    let mut entries = Vec::new();
    let mut details = Vec::new();
    for t in &tensors {
        if t.candidate() {
            if tensors
                .iter()
                .any(|x| x.name == format!("{}.ojon_scale", t.name))
            {
                return Err(invalid("source contains reserved .ojon_scale tensor"));
            }
            let (r, c) = (t.shape[0], t.shape[1]);
            entries.push(Entry {
                name: t.name.clone(),
                dtype: if bits == Bits::Eight { "I8" } else { "U8" },
                shape: vec![r, c * bits.count() / 8],
                bytes: t.output_bytes(bits)?,
            });
            entries.push(Entry {
                name: format!("{}.ojon_scale", t.name),
                dtype: "F32",
                shape: vec![r],
                bytes: r as u64 * 4,
            });
            details.push(
                json!({"name":t.name,"source_dtype":t.dtype,"source_shape":t.shape,
                "source_bytes":t.bytes,"output_dtype":if bits==Bits::Eight {"I8"} else {"U8"},
                "output_shape":[r,c*bits.count()/8],"output_bytes":t.output_bytes(bits)?,
                "quantization":bits.label(),"logical_shape":t.shape,
                "scale_tensor":format!("{}.ojon_scale",t.name),"scale_dtype":"F32",
                "scale_shape":[r],"scale_bytes":r as u64*4}),
            );
        } else {
            entries.push(Entry {
                name: t.name.clone(),
                dtype: t.dtype,
                shape: t.shape.clone(),
                bytes: t.bytes,
            });
            details.push(
                json!({"name":t.name,"source_dtype":t.dtype,"source_shape":t.shape,
                "source_bytes":t.bytes,"output_dtype":t.dtype,"output_shape":t.shape,
                "output_bytes":t.bytes,"quantization":null,"preserved":true}),
            );
        }
    }
    let part = dst.with_extension("safetensors.part");
    let mut metadata = Map::new();
    metadata.insert(
        "ojon.echo.format".into(),
        Value::String(bits.label().into()),
    );
    metadata.insert(
        "ojon.echo.inference_verified".into(),
        Value::String("false".into()),
    );
    let mut writer = BufWriter::with_capacity(4 * 1024 * 1024, File::create(&part)?);
    writer.write_all(&header(&entries, &metadata))?;
    for t in &tensors {
        if t.candidate() {
            let data = original.read(t)?;
            let scales = quantize_rows(t, &data, bits, &mut writer)?;
            for s in scales {
                writer.write_all(&s.to_le_bytes())?;
            }
        } else {
            original.copy_original(t, &mut writer)?;
        }
    }
    writer.flush()?;
    drop(writer);
    let actual = read_safetensors_header(&part)?;
    if actual.len() != entries.len() {
        return Err(bad("output tensor count mismatch"));
    }
    for e in &entries {
        let t = actual
            .get(&e.name)
            .ok_or_else(|| bad("missing output tensor"))?;
        if t.dtype != e.dtype || t.shape != e.shape || t.bytes != e.bytes {
            return Err(bad(format!("output tensor {} mismatched", e.name)));
        }
    }
    fs::rename(&part, dst)?;
    Ok(
        json!({"source":src,"output":dst,"source_bytes":fs::metadata(src)?.len(),
        "output_bytes":fs::metadata(dst)?.len(),"tensors":details,
        "structurally_verified":true,"runtime_verified":false}),
    )
}
pub fn run(root: &Path, destination: &Path, bits: Bits) -> io::Result<Value> {
    let plan = inspect(root, bits)?;
    if destination.exists() {
        return Err(invalid("output exists; never overwrite a model bundle"));
    }
    if destination.starts_with(root) {
        return Err(invalid("output must not be nested under source model"));
    }
    fs::create_dir(destination)?;
    let mut components = Vec::new();
    for (name, rel, convert_weights) in COMPONENTS {
        let src = root.join(rel);
        if convert_weights {
            let dst = destination.join(format!("{name}.safetensors"));
            components.push(convert(&src, &dst, bits)?);
        } else {
            let tensors = Checkpoint::open(&src)?.tensors()?;
            let tensor_info:Vec<_>=tensors.iter().map(|t|json!({
                "name":t.name,"dtype":t.dtype,"shape":t.shape,"bytes":t.bytes,"preserved":true
            })).collect();
            components.push(
                json!({"component":name,"source":src,"source_bytes":fs::metadata(&src)?.len(),
                "tensors":tensor_info,"preserved":true,"external_original":true}),
            );
        }
    }
    let manifest = json!({"schema":"ojon.echo.bundle.v1","model":"EchoMimicV3-Flash",
        "plan":plan,"components":components,"precision":bits.label(),
        "status":"STRUCTURALLY_VERIFIED_NOT_RUNTIME_TESTED","runtime_verified":false,"quality_verified":false});
    let serialized = serde_json::to_vec_pretty(&manifest)?;
    let temp = destination.join("echo-manifest.json.part");
    let mut writer = File::create(&temp)?;
    writer.write_all(&serialized)?;
    writer.sync_all()?;
    fs::rename(temp, destination.join("echo-manifest.json"))?;
    Ok(manifest)
}
#[cfg(test)]
mod tests {
    use super::*;
    fn sample(name: &str, shape: Vec<usize>) -> Tensor {
        Tensor {
            name: name.into(),
            dtype: "BF16",
            shape,
            bytes: 2048,
        }
    }
    #[test]
    fn preserve_norm_embedding_audio() {
        assert!(sample("blocks.12.self_attn.q.weight", vec![32, 32]).candidate());
        assert!(sample("transformer.resblocks.1.mlp.fc.weight", vec![32, 32]).candidate());
        assert!(sample(
            "encoder.block.0.layer.0.SelfAttention.q.weight",
            vec![32, 32]
        )
        .candidate());
        assert!(sample(
            "encoder.block.0.layer.1.DenseReluDense.wi_0.weight",
            vec![32, 32]
        )
        .candidate());
        assert!(sample("transformer.resblocks.1.attn.in_proj_weight", vec![32, 32]).candidate());
        assert!(!sample("token_embedding.weight", vec![32, 32]).candidate());
        assert!(!sample("blocks.0.norm.weight", vec![32]).candidate());
        assert!(!sample("audio_injection.linear.weight", vec![32, 32]).candidate());
    }
    #[test]
    fn expected_int8_int4_lengths() {
        let t = sample("blocks.0.attn.q.weight", vec![1, 4]);
        let original = [0x80, 0x3f, 0x00, 0xbf, 0x00, 0x00, 0x00, 0x3f];
        for (bits, expected) in [(Bits::Eight, 4), (Bits::Four, 2)] {
            let mut out = Vec::new();
            let scales = quantize_rows(&t, &original, bits, &mut out).unwrap();
            assert_eq!(out.len(), expected);
            assert_eq!(scales.len(), 1);
            let expected_scale = if bits == Bits::Eight {
                1.0 / 127.0
            } else {
                1.0 / 7.0
            };
            assert!((scales[0] - expected_scale).abs() < 0.000_001);
            if bits == Bits::Eight {
                assert_eq!(out, [127, 192, 0, 64]);
            } else {
                assert_eq!(out, [0xc7, 0x40]);
            }
            let decoded = decode_packed_row(&out, scales[0], 4, bits, "BF16").unwrap();
            let source_values = original.chunks_exact(2)
                .map(|b| decode_float(b, "BF16").unwrap())
                .collect::<Vec<_>>();
            for (original, bytes) in source_values.iter().zip(decoded.chunks_exact(2)) {
                let value = decode_float(bytes, "BF16").unwrap();
                let tolerance = if bits == Bits::Eight { 0.01 } else { 0.09 };
                assert!((original - value).abs() < tolerance);
            }
            assert!(decode_packed_row(&out, f32::NAN, 4, bits, "BF16").is_err());
            assert!(decode_packed_row(&out[..out.len() - 1], scales[0], 4, bits, "BF16").is_err());
        }
    }
    #[test]
    fn converts_and_verifies_real_safetensors_layout() {
        use std::time::{SystemTime, UNIX_EPOCH};
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("ojon_echo_{}_{}", std::process::id(), unique));
        fs::create_dir(&dir).unwrap();
        let src = dir.join("in.safetensors");
        let dst = dir.join("out.safetensors");
        let entries = vec![
            Entry {
                name: "encoder.block.0.layer.0.SelfAttention.q.weight".into(),
                dtype: "BF16",
                shape: vec![2, 16],
                bytes: 64,
            },
            Entry {
                name: "shared.weight".into(),
                dtype: "BF16",
                shape: vec![2, 16],
                bytes: 64,
            },
        ];
        let original = vec![0x80u8, 0x3f].repeat(32);
        let mut file = File::create(&src).unwrap();
        file.write_all(&header(&entries, &Map::new())).unwrap();
        file.write_all(&original).unwrap();
        file.write_all(&original).unwrap();
        drop(file);
        let manifest = convert(&src, &dst, Bits::Four).unwrap();
        assert_eq!(manifest["structurally_verified"], true);
        assert_eq!(manifest["tensors"].as_array().unwrap().len(), 2);
        let output = read_safetensors_header(&dst).unwrap();
        assert_eq!(output.len(), 3);
        assert_eq!(output["shared.weight"].dtype, "BF16");
        assert_eq!(
            output["encoder.block.0.layer.0.SelfAttention.q.weight"].dtype,
            "U8"
        );
        assert_eq!(
            output["encoder.block.0.layer.0.SelfAttention.q.weight"].bytes,
            16
        );
        // Decode one row directly from the compact safetensors output without
        // unpacking the full checkpoint or loading the model.
        let prefix = "encoder.block.0.layer.0.SelfAttention.q.weight";
        let packed_loc = &output[prefix];
        let scale_loc = &output[&format!("{prefix}.ojon_scale")];
        let mut compact = File::open(&dst).unwrap();
        compact.seek(SeekFrom::Start(packed_loc.offset)).unwrap();
        let mut packed_row = [0u8; 8];
        compact.read_exact(&mut packed_row).unwrap();
        compact.seek(SeekFrom::Start(scale_loc.offset)).unwrap();
        let mut scale_bytes = [0u8; 4];
        compact.read_exact(&mut scale_bytes).unwrap();
        let decoded = decode_packed_row(&packed_row, f32::from_le_bytes(scale_bytes), 16, Bits::Four, "BF16").unwrap();
        assert_eq!(decoded, original[..32]);
        let mut reader = PackedEchoReader::open(&dst, Bits::Four).unwrap();
        assert_eq!(reader.decode_weight_row(prefix, 0, 16, "BF16").unwrap(), original[..32]);
        assert_eq!(reader.decode_weight_row(prefix, 1, 16, "BF16").unwrap(), original[32..64]);
        assert!(reader.decode_weight_row(prefix, 2, 16, "BF16").is_err());
        assert!(reader.decode_weight_row(prefix, 0, 32, "BF16").is_err());
        let mut wrong_bits = PackedEchoReader::open(&dst, Bits::Eight).unwrap();
        assert!(wrong_bits.decode_weight_row(prefix, 0, 16, "BF16").is_err());
        drop(wrong_bits);
        drop(reader);
        drop(compact);
        fs::remove_file(src).unwrap();
        fs::remove_file(dst).unwrap();
        fs::remove_dir(dir).unwrap();
    }
    #[test]
    fn optional_real_torch_read_and_quantize_smoke() {
        let Some(root) = std::env::var_os("OJON_ECHO_TEST_ROOT") else {
            return;
        };
        let root = std::path::PathBuf::from(root);
        for (component, relative, _) in [COMPONENTS[0], COMPONENTS[1]] {
            let mut source = Checkpoint::open(&root.join(relative)).unwrap();
            let tensors = source.tensors().unwrap();
            let small = tensors
                .iter()
                .filter(|t| t.candidate() && t.bytes <= 64 * 1024 * 1024)
                .min_by_key(|t| t.bytes)
                .unwrap_or_else(|| panic!("{component}: no bounded linear sample"));
            let raw = source.read(small).unwrap();
            assert_eq!(raw.len() as u64, small.bytes);
            let mut packed = Vec::new();
            let scales = quantize_rows(small, &raw, Bits::Four, &mut packed).unwrap();
            assert_eq!(packed.len() as u64, small.output_bytes(Bits::Four).unwrap());
            assert_eq!(scales.len(), small.shape[0]);
            assert!(scales.iter().all(|s| s.is_finite() && *s > 0.0));
            eprintln!(
                "Echo {component} PyTorch read/INT4 smoke: {} bytes, {} rows",
                raw.len(),
                scales.len()
            );
        }
    }
}
