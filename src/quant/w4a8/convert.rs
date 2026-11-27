//! Model-level converter: stream a floating safetensors checkpoint into comfy-kitchen W4A8/W6A8
//! layout (loads in ComfyUI, Rupokar and Probaho). Selected 2-D linear weights become
//! weight (I8 packed) + weight_s_rel (F8_E4M3) + weight_s_channel (F32) [+ weight_codebook] +
//! comfy_quant (JSON); every other tensor and the file metadata are copied unchanged.

use super::matrix::{check_shape, quantize_matrix};
use super::quantize::{Bits, GROUP};
use super::st_writer::{header, Entry};
use crate::models::weight_patch::{read_safetensors_header, TensorLoc};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

// Whole-matrix quantization still needs the source matrix and its packed result
// in memory. Refuse oversized matrices instead of exhausting small machines.
const MAX_QUANT_TENSOR_BYTES: u64 = 512 * 1024 * 1024;

struct PartialOutput(PathBuf);
impl Drop for PartialOutput {
    fn drop(&mut self) {
        // This path is created exclusively with create_new below; never remove a
        // pre-existing user's partial file.
        if !self.0.as_os_str().is_empty() {
            let _ = fs::remove_file(&self.0);
        }
    }
}

pub struct Request {
    pub source: PathBuf,
    pub output: PathBuf,
    pub bits: Bits,
    /// Dot-separated globs; `*` matches one name segment. A layer is quantized when
    /// `<pattern>.weight` matches the tensor name.
    pub layers: Vec<String>,
    pub threads: usize,
}

/// Default for MiniMax H3 / H3-Image, matching Comfy-Org's published W4A8 file.
pub fn h3_default_layers() -> Vec<String> {
    ["blocks.*.attn.qkv_proj", "blocks.*.attn.out_proj", "blocks.*.mlp.fc1", "blocks.*.mlp.fc2"].map(String::from).to_vec()
}

/// Wan-family baseline transformer linears shared by the Wan 1.3B and Echo Flash
/// checkpoints. Echo adds separate audio cross-attention projections; these are
/// deliberately excluded pending calibrated audio/video quality measurements.
/// Rank, dtype and ConvRot alignment checks still apply to every candidate.
pub fn wan_default_layers() -> Vec<String> {
    let modules = [
        "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
        "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
        "cross_attn.k_img", "cross_attn.v_img", "ffn.*",
    ];
    ["blocks.*", "transformer_blocks.*"]
        .into_iter()
        .flat_map(|prefix| modules.iter().map(move |module| format!("{prefix}.{module}")))
        .collect()
}

fn glob(pattern: &str, name: &str) -> bool {
    let (p, n): (Vec<&str>, Vec<&str>) = (pattern.split('.').collect(), name.split('.').collect());
    p.len() == n.len() && p.iter().zip(&n).all(|(a, b)| *a == "*" || a == b)
}

fn layer_family(name: &str) -> String {
    name.split('.')
        .map(|segment| if !segment.is_empty() && segment.bytes().all(|c| c.is_ascii_digit()) { "*" } else { segment })
        .collect::<Vec<_>>()
        .join(".")
}

fn format_name(bits: Bits) -> &'static str {
    match bits {
        Bits::Four => "asym_w4a8_int8",
        Bits::Six => "w6a8_int8",
    }
}

fn selected(req: &Request, name: &str, t: &TensorLoc) -> bool {
    t.shape.len() == 2
        && matches!(t.dtype.as_str(), "F16" | "BF16" | "F32")
        && name.strip_suffix(".weight").is_some_and(|p| req.layers.iter().any(|g| glob(g, p)))
        && check_shape(t.shape[0], t.shape[1], req.bits).is_ok()
}

fn read_metadata(path: &Path) -> io::Result<Map<String, Value>> {
    let mut f = File::open(path)?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len)?;
    let mut h = vec![0u8; u64::from_le_bytes(len) as usize];
    f.read_exact(&mut h)?;
    let v: Value = serde_json::from_slice(&h).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
    Ok(v.get("__metadata__").and_then(Value::as_object).cloned().unwrap_or_default())
}

fn quant_entries(prefix: &str, n: usize, k: usize, bits: Bits) -> Vec<Entry> {
    let conf = json!({"format": format_name(bits)}).to_string();
    let mut v = vec![
        Entry { name: format!("{prefix}.comfy_quant"), dtype: "U8", shape: vec![conf.len()], bytes: conf.len() as u64 },
        Entry { name: format!("{prefix}.weight"), dtype: "I8", shape: vec![n, k * bits.count() / 8], bytes: (n * k * bits.count() / 8) as u64 },
        Entry { name: format!("{prefix}.weight_s_channel"), dtype: "F32", shape: vec![n], bytes: (n * 4) as u64 },
        Entry { name: format!("{prefix}.weight_s_rel"), dtype: "F8_E4M3", shape: vec![n, k / GROUP], bytes: (n * k / GROUP) as u64 },
    ];
    if bits == Bits::Four {
        v.push(Entry { name: format!("{prefix}.weight_codebook"), dtype: "F32", shape: vec![16], bytes: 64 });
    }
    v
}

const DTYPES: [&str; 12] = ["BF16", "F16", "F32", "F64", "I8", "U8", "I16", "I32", "I64", "BOOL", "F8_E4M3", "F8_E5M2"];

fn static_dtype(d: &str) -> io::Result<&'static str> {
    DTYPES.iter().copied().find(|&s| s == d).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("unsupported dtype {d}")))
}

struct Prepared {
    tensors: BTreeMap<String, TensorLoc>,
    metadata: Map<String, Value>,
    entries: Vec<Entry>,
    actions: Vec<(String, bool)>,
}

/// Shared, read-only tensor catalogue for both planning and conversion.
fn prepare(req: &Request) -> io::Result<Prepared> {
    let tensors = read_safetensors_header(&req.source)?;
    let metadata = read_metadata(&req.source)?;
    let mut entries = Vec::new();
    let mut actions = Vec::new(); // (source name, quantize?)
    for (name, t) in &tensors {
        if selected(req, name, t) {
            entries.extend(quant_entries(name.trim_end_matches(".weight"), t.shape[0], t.shape[1], req.bits));
            actions.push((name.clone(), true));
        } else {
            entries.push(Entry { name: name.clone(), dtype: static_dtype(&t.dtype)?, shape: t.shape.clone(), bytes: t.bytes });
            actions.push((name.clone(), false));
        }
    }
    if !actions.iter().any(|p| p.1) {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "no tensor matched the layer patterns"));
    }
    let mut output_names = BTreeSet::new();
    for entry in &entries {
        if !output_names.insert(&entry.name) {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, format!(
                "quantized output would duplicate tensor name {}; choose original floating weights",
                entry.name
            )));
        }
    }
    Ok(Prepared { tensors, metadata, entries, actions })
}

/// Inspect the real source tensor catalogue without allocating weights or writing output.
/// Output bytes are exact for this format; peak RAM is a conservative estimate.
pub fn describe_plan(req: &Request) -> io::Result<Value> {
    let p = prepare(req)?;
    let selected: Vec<_> = p.actions.iter().filter(|(_, q)| *q).collect();
    let mut selected_layer_families = BTreeMap::<String, usize>::new();
    for (name, _) in &selected {
        *selected_layer_families.entry(layer_family(name)).or_default() += 1;
    }
    let largest = selected.iter().map(|(name, _)| p.tensors[name].bytes).max().unwrap_or(0);
    let input_quant_bytes: u64 = selected.iter().map(|(name, _)| p.tensors[name].bytes).sum();
    let output_bytes = header(&p.entries, &p.metadata).len() as u64 + p.entries.iter().map(|e| e.bytes).sum::<u64>();
    let part = req.output.with_extension("safetensors.part");
    let output_exists = req.output.exists();
    let partial_exists = part.exists();
    let needs_streaming = largest > MAX_QUANT_TENSOR_BYTES;
    let parent = req.output.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let parent_exists = parent.is_dir();
    Ok(json!({
        "schema": "ojon.w4a8_plan.v1", "source": req.source, "output": req.output,
        "format": format_name(req.bits), "bits": req.bits.count(), "threads": req.threads,
        "layers": req.layers, "tensors_quantized": selected.len(),
        "selected_layer_families": selected_layer_families,
        "tensors_copied": p.actions.len() - selected.len(),
        "source_bytes": fs::metadata(&req.source)?.len(), "quantized_input_bytes": input_quant_bytes,
        "estimated_output_bytes": output_bytes,
        "largest_selected_tensor_bytes": largest,
        "estimated_peak_ram_bytes": largest.saturating_mul(4).saturating_add(128 * 1024 * 1024),
        "max_whole_tensor_bytes": MAX_QUANT_TENSOR_BYTES,
        "requires_row_streaming": needs_streaming,
        "output_exists": output_exists, "partial_exists": partial_exists,
        "output_parent_exists": parent_exists,
        "run_ready": !output_exists && !partial_exists && parent_exists && !needs_streaming,
        "disk_space_verified": false, "runtime_verified": false,
        "status": "PLAN_ONLY",
        "run": "w4a8-run <source.safetensors> <output.safetensors> [--bits 4|6] [--profile h3|wan] [--layers glob,glob] --confirm"
    }))
}

/// Plan, then stream. Returns a JSON receipt. Never overwrites `output`.
pub fn run(req: &Request, mut progress: impl FnMut(&str)) -> io::Result<Value> {
    if req.output.exists() {
        return Err(io::Error::new(io::ErrorKind::AlreadyExists, format!("{} exists", req.output.display())));
    }
    let Prepared { tensors, metadata, entries, actions: plan } = prepare(req)?;
    let quantized = plan.iter().filter(|p| p.1).count();
    for (name, quant) in &plan {
        if *quant && tensors[name].bytes > MAX_QUANT_TENSOR_BYTES {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, format!(
                "{name} is {} MiB; whole-matrix W4A8 has a 512 MiB per-tensor safety limit. This model requires a row-streaming quantizer",
                tensors[name].bytes.div_ceil(1024 * 1024)
            )));
        }
    }
    let part = req.output.with_extension("safetensors.part");
    let started = Instant::now();
    let mut src = File::open(&req.source)?;
    let temp = File::options().write(true).create_new(true).open(&part)?;
    let mut guard = PartialOutput(part.clone());
    let mut out = BufWriter::with_capacity(16 << 20, temp);
    let head = header(&entries, &metadata);
    let expected_bytes = head.len() as u64 + entries.iter().map(|e| e.bytes).sum::<u64>();
    out.write_all(&head)?;
    let mut fitted = Vec::new();
    for (i, (name, quant)) in plan.iter().enumerate() {
        let t = &tensors[name];
        src.seek(SeekFrom::Start(t.offset))?;
        if !quant {
            let copied = io::copy(&mut (&mut src).take(t.bytes), &mut out)?;
            if copied != t.bytes {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, format!("short copy of {name}")));
            }
            continue;
        }
        let mut buf = vec![0u8; t.bytes as usize];
        src.read_exact(&mut buf)?;
        let (n, k) = (t.shape[0], t.shape[1]);
        let p = quantize_matrix(&buf, &t.dtype, n, k, req.bits, req.threads).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("{name}: {e}")))?;
        out.write_all(json!({"format": format_name(req.bits)}).to_string().as_bytes())?;
        out.write_all(&p.qdata)?;
        out.write_all(&p.s_channel.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
        out.write_all(&p.s_rel)?;
        if let Some(cb) = &p.codebook {
            out.write_all(&cb.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
        }
        if p.codebook_fitted {
            fitted.push(name.clone());
        }
        progress(&format!("[{}/{}] {name} {n}x{k} -> {}", i + 1, plan.len(), format_name(req.bits)));
    }
    out.flush()?;
    drop(out);
    if fs::metadata(&part)?.len() != expected_bytes {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "W4A8 output length differs from the planned header"));
    }
    let checked = read_safetensors_header(&part)?;
    if checked.len() != entries.len() || entries.iter().any(|e| {
        checked.get(&e.name).is_none_or(|t| t.dtype != e.dtype || t.shape != e.shape || t.bytes != e.bytes)
    }) {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "W4A8 output tensor catalog differs from the planned header"));
    }
    fs::rename(&part, &req.output)?;
    guard.0.clear();
    Ok(json!({
        "schema": "ojon.w4a8_convert.v1",
        "source": req.source, "output": req.output, "format": format_name(req.bits),
        "tensors_quantized": quantized, "tensors_copied": plan.len() - quantized,
        "source_bytes": fs::metadata(&req.source)?.len(), "output_bytes": fs::metadata(&req.output)?.len(),
        "fitted_codebooks": fitted, "seconds": started.elapsed().as_secs_f64().round(),
        "layout": "comfy-kitchen grouped INT8 (ConvRot 256, group 16, fp8 e4m3 s_rel, f32 s_channel)",
        "runtime_verified": false
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn optional_comfy_interop(source: &Path, output: &Path, tensor_name: &str) {
        let Some(python) = std::env::var_os("OJON_W4A8_INTEROP_PYTHON") else { return };
        let script = r#"
import importlib.util, json, sys, torch, comfy_kitchen as ck
from safetensors.torch import load_file

source, quant = load_file(sys.argv[1]), load_file(sys.argv[2])
name = sys.argv[3]
prefix = name.removesuffix('.weight')
weights = source[name].to(torch.float32)
config = json.loads(bytes(quant[prefix + '.comfy_quant'].tolist()))
assert config['format'] == 'asym_w4a8_int8'
q = quant[name]
s_rel = quant[prefix + '.weight_s_rel']
s_channel = quant[prefix + '.weight_s_channel']
codebook = quant[prefix + '.weight_codebook']
assert q.dtype == torch.int8 and s_rel.dtype == torch.float8_e4m3fn
with ck.use_backend('eager'):
    back = ck.tensor.dequantize_w4a8_int8_weight(
        q, s_rel, s_channel, codebook=codebook, output_dtype=torch.float32)
    x = torch.linspace(-0.5, 0.5, weights.shape[1]).reshape(1, -1)
    y = ck.tensor.w4a8_int8_linear(
        x, q, s_rel, s_channel, codebook=codebook, out_dtype=torch.float32)
assert back.shape == weights.shape and torch.isfinite(back).all()
relative_error = (back - weights).norm().item() / weights.norm().item()
assert relative_error < 0.25, f'weight relative error: {relative_error}'
assert y.shape == (1, weights.shape[0]) and torch.isfinite(y).all()
assert torch.allclose(y, x @ back.T, atol=0.05, rtol=0.05)

# Exercise the actual optional nn.Linear replacement path on a 3-D sequence
# input, with nonzero bias. This is a CPU module-forward smoke test, not Echo.
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('ojon_w4a8_bridge', sys.argv[4])
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)
root, node = torch.nn.Module(), None
node = root
for segment in prefix.split('.')[:-1]:
    child = torch.nn.Module()
    node.add_module(segment, child)
    node = child
leaf = prefix.split('.')[-1]
original = torch.nn.Linear(weights.shape[1], weights.shape[0], bias=True)
with torch.no_grad():
    original.weight.copy_(weights)
    original.bias.copy_(torch.linspace(-0.1, 0.1, weights.shape[0]))
node.add_module(leaf, original)
replaced = bridge.replace_w4a8_linears(root, sys.argv[2], layer_names=[prefix], backend='eager')
assert replaced == [prefix]
packed_module = root.get_submodule(prefix)
assert isinstance(packed_module, bridge.PackedW4A8Linear)
x3 = torch.linspace(-0.5, 0.5, 6 * weights.shape[1]).reshape(2, 3, weights.shape[1])
actual = packed_module(x3)
expected = torch.nn.functional.linear(x3, back, original.bias)
assert actual.shape == expected.shape
assert torch.allclose(actual, expected, atol=0.05, rtol=0.05)
bf16_input = x3.to(torch.bfloat16)
bf16_actual = packed_module(bf16_input)
bf16_expected = torch.nn.functional.linear(bf16_input.float(), back, original.bias)
assert bf16_actual.dtype == torch.bfloat16 and bf16_actual.shape == expected.shape
assert torch.allclose(bf16_actual.float(), bf16_expected, atol=0.15, rtol=0.10)
print(f'Ojon W4A8 nn.Linear replacement PASS; tensor={name}; sequence shape={tuple(actual.shape)}')

# The low-memory loader must materialise only packed weights plus remaining
# base tensors. Test on an actual meta model and a tiny base fallback archive.
import tempfile
from pathlib import Path
from accelerate import init_empty_weights
from safetensors.torch import save_file
with tempfile.TemporaryDirectory() as tempdir:
    extra_weight = torch.full((weights.shape[0], weights.shape[1]), 0.125)
    extra_bias = torch.linspace(-0.2, 0.2, weights.shape[0])
    fallback = str(Path(tempdir) / 'base.safetensors')
    save_file({'extra.weight': extra_weight, 'extra.bias': extra_bias}, fallback)
    with init_empty_weights():
        streaming = torch.nn.Module()
        node = streaming
        for segment in prefix.split('.')[:-1]:
            child = torch.nn.Module()
            node.add_module(segment, child)
            node = child
        node.add_module(prefix.split('.')[-1], torch.nn.Linear(weights.shape[1], weights.shape[0], bias=False))
        streaming.add_module('extra', torch.nn.Linear(weights.shape[1], weights.shape[0], bias=True))
    assert all(p.is_meta for p in streaming.parameters())
    try:
        bridge.load_w4a8_meta_model(streaming, sys.argv[2], dtype=torch.float32, backend='eager')
        raise AssertionError('missing fallback should fail before mutation')
    except ValueError as exc:
        assert 'extra.' in str(exc)
    assert all(p.is_meta for p in streaming.parameters())
    selected_names = bridge.load_w4a8_meta_model(
        streaming, sys.argv[2], fallback_path=fallback, dtype=torch.float32, backend='eager')
    assert selected_names == [prefix]
    assert all(not p.is_meta for p in streaming.parameters())
    assert isinstance(streaming.get_submodule(prefix), bridge.PackedW4A8Linear)
    assert torch.equal(streaming.extra.weight, extra_weight)
    assert torch.equal(streaming.extra.bias, extra_bias)
    packed_forward = streaming.get_submodule(prefix)(x3)
    assert torch.allclose(packed_forward, torch.nn.functional.linear(x3, back), atol=0.05, rtol=0.05)
    assert torch.allclose(streaming.extra(x3), torch.nn.functional.linear(x3, extra_weight, extra_bias))
print(f'Ojon W4A8 meta/streaming loader PASS; tensor={name}; dense packed weight never allocated')
print(f'Comfy-Kitchen eager W4A8 interop PASS; tensor={name}; relative weight error={relative_error:.4f}')
"#;
        let result = std::process::Command::new(python)
            .arg("-c").arg(script).arg(source).arg(output).arg(tensor_name)
            .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/quant/w4a8/torch_bridge.py"))
            .output().expect("launch W4A8 interop interpreter");
        assert!(result.status.success(), "interop failure: {} {}",
            String::from_utf8_lossy(&result.stdout), String::from_utf8_lossy(&result.stderr));
        println!("{}", String::from_utf8_lossy(&result.stdout));
    }

    #[test]
    fn layer_globs_match_one_segment_per_star() {
        assert!(glob("blocks.*.mlp.fc1", "blocks.12.mlp.fc1"));
        assert!(!glob("blocks.*.mlp.fc1", "token_refiner.blocks.0.mlp.fc1"));
        assert!(!glob("blocks.*.mlp.fc1", "blocks.1.mlp.fc2"));
        assert_eq!(layer_family("blocks.12.self_attn.q.weight"), "blocks.*.self_attn.q.weight");
    }

    #[test]
    fn selected_requires_a_float_matrix() {
        let req = Request { source: PathBuf::new(), output: PathBuf::new(), bits: Bits::Four,
            layers: h3_default_layers(), threads: 1 };
        let mut t = TensorLoc { file: PathBuf::new(), offset: 0, dtype: "F32".into(),
            shape: vec![2, 256], bytes: 2048 };
        assert!(selected(&req, "blocks.0.mlp.fc1.weight", &t));
        t.dtype = "I8".into();
        assert!(!selected(&req, "blocks.0.mlp.fc1.weight", &t));
    }

    #[test]
    fn wan_profile_matches_transformer_matrices_but_not_norms_or_other_models() {
        let req = Request { source: PathBuf::new(), output: PathBuf::new(), bits: Bits::Four,
            layers: wan_default_layers(), threads: 1 };
        let mut t = TensorLoc { file: PathBuf::new(), offset: 0, dtype: "BF16".into(),
            shape: vec![64, 256], bytes: 32768 };
        assert!(selected(&req, "blocks.0.self_attn.q.weight", &t));
        assert!(selected(&req, "transformer_blocks.12.self_attn.q.weight", &t));
        assert!(selected(&req, "blocks.0.cross_attn.k_img.weight", &t));
        assert!(!selected(&req, "blocks.0.cross_attn.k_audio.weight", &t));
        assert!(!selected(&req, "blocks.0.cross_attn.q_audio.weight", &t));
        assert!(!selected(&req, "blocks.0.cross_attn.v_audio.weight", &t));
        assert!(!selected(&req, "blocks.0.norm1.weight", &t));
        assert!(!selected(&req, "text_embedding.weight", &t));
        t.shape = vec![64, 255];
        assert!(!selected(&req, "blocks.0.self_attn.q.weight", &t));
    }

    #[test]
    fn converts_and_verifies_header_while_preserving_other_payloads() {
        let stamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let base = std::env::temp_dir().join(format!("ojon-w4a8-test-{}-{stamp}", std::process::id()));
        fs::create_dir(&base).unwrap();
        let source = base.join("original.safetensors");
        let output = base.join("converted.safetensors");
        let mut weights = Vec::new();
        for i in 0..512 {
            weights.extend_from_slice(&((i as f32 - 256.0) / 512.0).to_le_bytes());
        }
        let unchanged = vec![0xA5u8; 2 << 20];
        let entries = vec![
            Entry { name: "blocks.0.mlp.fc1.weight".into(), dtype: "F32", shape: vec![2, 256], bytes: weights.len() as u64 },
            Entry { name: "untouched.blob".into(), dtype: "U8", shape: vec![unchanged.len()], bytes: unchanged.len() as u64 },
        ];
        {
            let mut f = File::create(&source).unwrap();
            f.write_all(&header(&entries, &Map::new())).unwrap();
            f.write_all(&weights).unwrap();
            f.write_all(&unchanged).unwrap();
        }
        let req = Request { source: source.clone(), output: output.clone(), bits: Bits::Four,
            layers: h3_default_layers(), threads: 1 };
        let planned = describe_plan(&req).unwrap();
        assert_eq!(planned["schema"], "ojon.w4a8_plan.v1");
        assert_eq!(planned["run_ready"], true);
        assert_eq!(planned["tensors_quantized"], 1);
        assert_eq!(planned["tensors_copied"], 1);
        assert_eq!(planned["requires_row_streaming"], false);
        let receipt = run(&req, |_| {}).unwrap();
        assert_eq!(receipt["tensors_quantized"], 1);
        assert_eq!(receipt["tensors_copied"], 1);
        assert_eq!(planned["estimated_output_bytes"], receipt["output_bytes"]);
        let catalog = read_safetensors_header(&output).unwrap();
        assert_eq!(catalog["blocks.0.mlp.fc1.weight"].dtype, "I8");
        assert!(catalog.contains_key("blocks.0.mlp.fc1.weight_s_rel"));
        let loc = &catalog["untouched.blob"];
        let mut f = File::open(&output).unwrap();
        f.seek(SeekFrom::Start(loc.offset)).unwrap();
        let mut copied = vec![0u8; unchanged.len()];
        f.read_exact(&mut copied).unwrap();
        assert_eq!(copied, unchanged);
        assert!(!output.with_extension("safetensors.part").exists());

        optional_comfy_interop(&source, &output, "blocks.0.mlp.fc1.weight");

        // An existing partial output is not a scratch file we may truncate.
        let output2 = base.join("second.safetensors");
        let part2 = output2.with_extension("safetensors.part");
        fs::write(&part2, b"keep my checkpoint").unwrap();
        let req2 = Request { output: output2, ..req };
        let blocked = describe_plan(&req2).unwrap();
        assert_eq!(blocked["partial_exists"], true);
        assert_eq!(blocked["run_ready"], false);
        assert_eq!(run(&req2, |_| {}).unwrap_err().kind(), io::ErrorKind::AlreadyExists);
        assert_eq!(fs::read(&part2).unwrap(), b"keep my checkpoint");

        // Never silently overwrite a pre-existing tensor in the output header.
        let colliding = base.join("colliding.safetensors");
        let collision_entries = vec![
            Entry { name: "blocks.0.mlp.fc1.comfy_quant".into(), dtype: "U8", shape: vec![1], bytes: 1 },
            Entry { name: "blocks.0.mlp.fc1.weight".into(), dtype: "F32", shape: vec![2, 256], bytes: weights.len() as u64 },
        ];
        {
            let mut f = File::create(&colliding).unwrap();
            f.write_all(&header(&collision_entries, &Map::new())).unwrap();
            f.write_all(&[0]).unwrap();
            f.write_all(&weights).unwrap();
        }
        let conflict = Request { source: colliding, output: base.join("conflict.safetensors"),
            bits: Bits::Four, layers: h3_default_layers(), threads: 1 };
        assert!(describe_plan(&conflict).unwrap_err().to_string().contains("duplicate tensor name"));
        assert!(!conflict.output.exists());
        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn optional_real_checkpoint_one_layer_comfy_interop() {
        let Some(real_path) = std::env::var_os("OJON_W4A8_REAL_SAFETENSORS") else { return };
        assert!(std::env::var_os("OJON_W4A8_INTEROP_PYTHON").is_some(),
            "set OJON_W4A8_INTEROP_PYTHON for the real checkpoint interop test");
        let real_path = PathBuf::from(real_path);
        let name = std::env::var("OJON_W4A8_REAL_TENSOR")
            .unwrap_or_else(|_| "blocks.0.self_attn.q.weight".to_owned());
        let tensors = read_safetensors_header(&real_path).unwrap();
        let loc = tensors.get(&name).expect("selected real tensor exists");
        let layer = name.strip_suffix(".weight").expect("linear weight name").to_owned();
        let stamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let base = std::env::temp_dir().join(format!("ojon-real-w4a8-{}-{stamp}", std::process::id()));
        fs::create_dir(&base).unwrap();
        struct Cleanup(PathBuf);
        impl Drop for Cleanup { fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); } }
        let _cleanup = Cleanup(base.clone());
        let source = base.join("one_original.safetensors");
        let output = base.join("one_quantized.safetensors");
        let req = Request { source: source.clone(), output: output.clone(), bits: Bits::Four,
            layers: vec![layer], threads: 1 };
        assert!(selected(&req, &name, loc), "real tensor must be ConvRot-compatible");
        assert!(loc.bytes <= MAX_QUANT_TENSOR_BYTES, "sample exceeds the safe tensor limit");
        {
            let entry = Entry { name: name.clone(), dtype: static_dtype(&loc.dtype).unwrap(),
                shape: loc.shape.clone(), bytes: loc.bytes };
            let mut src = File::open(&real_path).unwrap();
            src.seek(SeekFrom::Start(loc.offset)).unwrap();
            let mut tiny = File::create(&source).unwrap();
            tiny.write_all(&header(&[entry], &Map::new())).unwrap();
            assert_eq!(io::copy(&mut src.take(loc.bytes), &mut tiny).unwrap(), loc.bytes);
        }
        let plan = describe_plan(&req).unwrap();
        assert_eq!(plan["tensors_quantized"], 1);
        let receipt = run(&req, |_| {}).unwrap();
        assert_eq!(receipt["tensors_quantized"], 1);
        optional_comfy_interop(&source, &output, &name);
    }
}
