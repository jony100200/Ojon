//! Whole-tensor W4A8/W6A8 quantization: one codebook decision per tensor, then rows in parallel.

use super::codebook;
use super::levels::uniform6;
use super::numeric::decode;
use super::quantize::{quantize_row, rotate_row, Bits, RotDtype, CONVROT, GROUP};

const CODEBOOK_SAMPLE_ELEMS: usize = 1 << 22;

pub struct Packed {
    pub qdata: Vec<u8>,
    pub s_rel: Vec<u8>,
    pub s_channel: Vec<f32>,
    /// 4-bit only: the 16-entry table stored as `weight_codebook`.
    pub codebook: Option<Vec<f32>>,
    pub codebook_fitted: bool,
}

pub fn check_shape(n: usize, k: usize, bits: Bits) -> Result<(), String> {
    if n == 0 || k % GROUP != 0 || k % CONVROT != 0 {
        return Err(format!("K={k} must be divisible by {GROUP} and the ConvRot group {CONVROT}"));
    }
    if bits == Bits::Six && k % 32 != 0 {
        return Err(format!("6-bit storage needs K divisible by 32, got {k}"));
    }
    Ok(())
}

fn rot_dtype(dtype: &str) -> RotDtype {
    match dtype {
        "BF16" => RotDtype::Bf16,
        "F16" => RotDtype::F16,
        _ => RotDtype::F32,
    }
}

fn load_row(src: &[u8], width: usize, dtype: &str, r: usize, k: usize, out: &mut [f32]) {
    let base = r * k * width;
    for (c, v) in out.iter_mut().enumerate() {
        *v = decode(dtype, &src[base + c * width..base + (c + 1) * width]);
    }
}

/// Evenly strided rows, like `torch.linspace(0, n-1, max_rows).long()`.
fn sample_rows(n: usize, k: usize) -> Vec<usize> {
    let max_rows = (CODEBOOK_SAMPLE_ELEMS / k.max(1)).max(1);
    if n <= max_rows {
        return (0..n).collect();
    }
    let step = (n - 1) as f32 / (max_rows - 1).max(1) as f32;
    (0..max_rows).map(|i| ((i as f32 * step) as usize).min(n - 1)).collect()
}

fn choose_table(src: &[u8], width: usize, dtype: &str, n: usize, k: usize) -> codebook::Choice {
    let mut row = vec![0f32; k];
    let mut scratch = vec![0f32; CONVROT];
    let mut normalized = Vec::new();
    for r in sample_rows(n, k) {
        load_row(src, width, dtype, r, k, &mut row);
        rotate_row(&mut row, rot_dtype(dtype), &mut scratch);
        for g in row.chunks(GROUP) {
            let amax = g.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-8);
            normalized.extend(g.iter().map(|v| v / amax));
        }
    }
    codebook::choose(&normalized)
}

/// Quantize an [n, k] floating tensor given as raw little-endian bytes of `dtype`.
pub fn quantize_matrix(src: &[u8], dtype: &str, n: usize, k: usize, bits: Bits, threads: usize) -> Result<Packed, String> {
    check_shape(n, k, bits)?;
    let width = super::numeric::source_width(dtype).ok_or_else(|| format!("unsupported source dtype {dtype}"))?;
    if src.len() != n * k * width {
        return Err("tensor byte size does not match its shape".into());
    }
    let (table, codebook, fitted) = match bits {
        Bits::Six => (uniform6(), None, false),
        Bits::Four => {
            let c = choose_table(src, width, dtype, n, k);
            (c.table.clone(), Some(c.table), c.fitted)
        }
    };
    let row_bytes = k * bits.count() / 8;
    let groups = k / GROUP;
    let mut qdata = vec![0u8; n * row_bytes];
    let mut s_rel = vec![0u8; n * groups];
    let mut s_channel = vec![0f32; n];
    let per = n.div_ceil(threads.max(1));
    std::thread::scope(|scope| {
        let parts = qdata.chunks_mut(per * row_bytes).zip(s_rel.chunks_mut(per * groups)).zip(s_channel.chunks_mut(per));
        for (t, ((q, s), c)) in parts.enumerate() {
            let table = &table;
            scope.spawn(move || {
                let mut row = vec![0f32; k];
                let mut scratch = vec![0f32; CONVROT];
                for (i, ch) in c.iter_mut().enumerate() {
                    load_row(src, width, dtype, t * per + i, k, &mut row);
                    rotate_row(&mut row, rot_dtype(dtype), &mut scratch);
                    *ch = quantize_row(&row, table, bits, &mut q[i * row_bytes..(i + 1) * row_bytes], &mut s[i * groups..(i + 1) * groups]);
                }
            });
        }
    });
    Ok(Packed { qdata, s_rel, s_channel, codebook, codebook_fitted: fitted })
}
