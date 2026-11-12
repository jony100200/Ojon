//! Row-wise port of comfy-kitchen `_quantize_rotated_w4a8_int8_weight` (symmetric):
//! ConvRot rotate -> per-group ALS scale (2 passes) -> per-row s_channel -> fp8 s_rel ->
//! assignment on the decoded INT8 grid (6-bit: grid-aware fp8 scale search) -> pack.

use super::hadamard::rotate_group;
use super::levels::{grid_levels, nearest};
use super::numeric::{e4m3_to_f32, f32_to_e4m3, round_bf16, round_f16};
use super::pack::pack_row;

pub const GROUP: usize = 16;
pub const CONVROT: usize = 256;
const ALS_ITERS: usize = 2;
const E4M3_MAX_FINITE: u8 = 0x7E;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Bits {
    Four,
    Six,
}

impl Bits {
    pub fn count(self) -> usize {
        match self {
            Bits::Four => 4,
            Bits::Six => 6,
        }
    }
}

/// Rounding applied after rotation: torch rotates in the weight's own dtype.
#[derive(Clone, Copy)]
pub enum RotDtype {
    Bf16,
    F16,
    F32,
}

pub fn rotate_row(row: &mut [f32], dtype: RotDtype, scratch: &mut [f32]) {
    for g in row.chunks_mut(CONVROT) {
        rotate_group(g, scratch);
        for v in g.iter_mut() {
            *v = match dtype {
                RotDtype::Bf16 => round_bf16(*v),
                RotDtype::F16 => round_f16(*v),
                RotDtype::F32 => *v,
            };
        }
    }
}

/// Quantize one already-rotated row. Writes the packed row and per-group fp8 scales; returns s_channel.
pub fn quantize_row(w: &[f32], table: &[f32], bits: Bits, packed: &mut [u8], s_rel: &mut [u8]) -> f32 {
    let groups = w.len() / GROUP;
    let mut q = vec![0usize; w.len()];
    let mut scale = vec![0f32; groups];
    for g in 0..groups {
        let gw = &w[g * GROUP..(g + 1) * GROUP];
        let gq = &mut q[g * GROUP..(g + 1) * GROUP];
        scale[g] = als_group(gw, table, bits, gq);
    }
    let shifted_max = (0..w.len()).map(|i| (table[q[i]] * scale[i / GROUP]).abs()).fold(0f32, f32::max);
    let s_channel = (shifted_max / 127.0).max(1e-8);
    let mut codes = vec![0u8; w.len()];
    let mut levels = Vec::with_capacity(table.len());
    for g in 0..groups {
        let gw = &w[g * GROUP..(g + 1) * GROUP];
        let raw = f32_to_e4m3(scale[g] / s_channel);
        let chosen = match bits {
            Bits::Four => raw,
            Bits::Six => search_scale(gw, s_channel, table, raw, &mut levels),
        };
        s_rel[g] = chosen;
        grid_levels(table, e4m3_to_f32(chosen), &mut levels);
        let offset = if bits == Bits::Six { 1 } else { 0 }; // 6-bit stores level index + 1 (q + 32)
        for (i, &x) in gw.iter().enumerate() {
            codes[g * GROUP + i] = (nearest(&levels, x / s_channel) + offset) as u8;
        }
    }
    pack_row(&codes, bits, packed);
    s_channel
}

/// Initial assignment plus the ALS group-scale refinement; returns the group scale.
fn als_group(gw: &[f32], table: &[f32], bits: Bits, gq: &mut [usize]) -> f32 {
    let amax = gw.iter().fold(0f32, |m, v| m.max(v.abs()));
    let assign = |s: f32, gq: &mut [usize]| {
        for (i, &x) in gw.iter().enumerate() {
            gq[i] = match bits {
                Bits::Six => ((x / s).round_ties_even().clamp(-31.0, 31.0) + 31.0) as usize,
                Bits::Four => nearest(table, x / s),
            };
        }
    };
    let mut s = match bits {
        Bits::Six => (amax / 31.0).max(1e-8),
        Bits::Four => amax.max(1e-8),
    };
    assign(s, gq);
    for _ in 0..ALS_ITERS {
        let (mut num, mut den) = (0f32, 0f32);
        for (i, &x) in gw.iter().enumerate() {
            let l = table[gq[i]];
            num += x * l;
            den += l * l;
        }
        s = (num / den.max(1e-8)).max(1e-8);
        assign(s, gq);
    }
    s
}

/// 6-bit only: try the ALS scale and its two fp8 neighbours, keep the best through-grid fit.
fn search_scale(gw: &[f32], s_channel: f32, table: &[f32], raw: u8, levels: &mut Vec<f32>) -> u8 {
    let candidates = [raw, raw.saturating_sub(1).max(1), (raw + 1).min(E4M3_MAX_FINITE)];
    let (mut best, mut best_err) = (raw, f32::INFINITY);
    for &c in &candidates {
        grid_levels(table, e4m3_to_f32(c), levels);
        let err: f32 = gw
            .iter()
            .map(|&x| {
                let t = x / s_channel;
                let d = levels[nearest(levels, t)] - t;
                d * d
            })
            .sum();
        if err < best_err {
            best = c;
            best_err = err;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::super::levels::{uniform6, FIXED_LUT};
    use super::super::pack::unpack_row;
    use super::*;

    fn decode(codes: &[u8], s_rel: &[u8], s_channel: f32, table: &[f32], bits: Bits) -> Vec<f32> {
        let mut levels = Vec::new();
        codes
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                grid_levels(table, e4m3_to_f32(s_rel[i / GROUP]), &mut levels);
                let idx = c as usize - if bits == Bits::Six { 1 } else { 0 };
                levels[idx] * s_channel
            })
            .collect()
    }

    #[test]
    fn six_bit_reconstructs_closer_than_four_bit() {
        let w: Vec<f32> = (0..512).map(|i| ((i * 7919 % 1000) as f32 / 500.0 - 1.0) * 0.02).collect();
        let mut errs = Vec::new();
        for (bits, table) in [(Bits::Four, FIXED_LUT.to_vec()), (Bits::Six, uniform6())] {
            let mut packed = vec![0u8; w.len() * bits.count() / 8];
            let mut s_rel = vec![0u8; w.len() / GROUP];
            let s_ch = quantize_row(&w, &table, bits, &mut packed, &mut s_rel);
            let codes = unpack_row(&packed, w.len(), bits);
            let back = decode(&codes, &s_rel, s_ch, &table, bits);
            let e: f32 = w.iter().zip(&back).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt()
                / w.iter().map(|a| a * a).sum::<f32>().sqrt();
            errs.push(e);
        }
        assert!(errs[0] < 0.15 && errs[1] < errs[0] / 2.0, "errors {errs:?}");
    }
}
