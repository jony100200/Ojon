//! Scalar reference k-quant block quantizers (Q4_K, Q5_K, Q6_K) in the ggml block layouts, so any
//! ggml loader reads the rows unchanged. Rows must be a multiple of 256 values. No dependencies.

pub const QK_K: usize = 256;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum KQuant {
    Q4K,
    Q5K,
    Q6K,
}

impl KQuant {
    /// Bytes of one 256-value block.
    pub fn block_bytes(self) -> usize {
        match self {
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
        }
    }
    /// The ggml_type number stored in a GGUF tensor record.
    pub fn ggml_type(self) -> u32 {
        match self {
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
        }
    }
    /// Bytes for a row of `n` values (`n` must be a multiple of 256).
    pub fn row_bytes(self, n: usize) -> usize {
        n / QK_K * self.block_bytes()
    }
}

/// ggml's round-to-nearest-even trick; valid for |f| < 4194303.
fn nearest_int(f: f32) -> i32 {
    let bits = (f + 12582912.0).to_bits() as i32;
    (bits & 0x007f_ffff) - 0x0040_0000
}

pub fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let man = x & 0x007f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x0200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x0080_0000;
        let shift = (14 - e) as u32;
        let half = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let mid = 1 << (shift - 1);
        let round = (rem > mid || (rem == mid && half & 1 == 1)) as u32;
        return sign | (half + round) as u16;
    }
    let half = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    let round = (rem > 0x1000 || (rem == 0x1000 && half & 1 == 1)) as u32;
    sign | (half + round) as u16
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x03ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            let mut e = 127 - 15 + 1;
            let mut m = man;
            while m & 0x0400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((m & 0x03ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

/// Weighted least-squares affine quantizer for one 32-value group (ggml `make_qkx2_quants`).
/// Returns `(scale, -min)`; the quantized levels land in `l`.
#[allow(clippy::too_many_arguments)]
fn make_qkx2(x: &[f32], weights: &[f32], nmax: i32, l: &mut [u8], laux: &mut [u8], rmin: f32, rdelta: f32, nstep: i32) -> (f32, f32) {
    let n = x.len();
    let (mut min, mut max) = (x[0], x[0]);
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * x[0];
    for i in 1..n {
        min = min.min(x[i]);
        max = max.max(x[i]);
        sum_w += weights[i];
        sum_x += weights[i] * x[i];
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max == min {
        l.fill(0);
        return (0.0, -min);
    }
    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_mad = 0.0f32;
    for i in 0..n {
        let li = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
        l[i] = li as u8;
        let diff = scale * li as f32 + min - x[i];
        best_mad += weights[i] * diff * diff;
    }
    for step in 0..=nstep {
        iscale = (rmin + rdelta * step as f32 + nmax as f32) / (max - min);
        let (mut sum_l, mut sum_l2, mut sum_xl) = (0.0f32, 0.0f32, 0.0f32);
        for i in 0..n {
            let li = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
            laux[i] = li as u8;
            let lf = li as f32;
            sum_l += weights[i] * lf;
            sum_l2 += weights[i] * lf * lf;
            sum_xl += weights[i] * lf * x[i];
        }
        let d = sum_w * sum_l2 - sum_l * sum_l;
        if d > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / d;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / d;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut mad = 0.0f32;
            for i in 0..n {
                let diff = this_scale * laux[i] as f32 + this_min - x[i];
                mad += weights[i] * diff * diff;
            }
            if mad < best_mad {
                l.copy_from_slice(laux);
                best_mad = mad;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    (scale, -min)
}

/// Symmetric quantizer for 16-value Q6_K groups (ggml `make_qx_quants`, rmse_type 1).
fn make_qx(x: &[f32], nmax: i32, l: &mut [i8]) -> f32 {
    let (mut amax, mut max) = (0.0f32, 0.0f32);
    for &v in x {
        if v.abs() > amax {
            amax = v.abs();
            max = v;
        }
    }
    if amax < 1e-15 {
        l.fill(0);
        return 0.0;
    }
    let level = |iscale: f32, v: f32| nearest_int(iscale * v).clamp(-nmax, nmax - 1);
    let mut iscale = -nmax as f32 / max;
    let (mut sumlx, mut suml2) = (0.0f32, 0.0f32);
    for i in 0..x.len() {
        let li = level(iscale, x[i]);
        l[i] = li as i8;
        let w = x[i] * x[i];
        sumlx += w * x[i] * li as f32;
        suml2 += w * (li * li) as f32;
    }
    let mut scale = if suml2 != 0.0 { sumlx / suml2 } else { 0.0 };
    let mut best = scale * sumlx;
    for is in -9..=9 {
        if is == 0 {
            continue;
        }
        iscale = -(nmax as f32 + 0.1 * is as f32) / max;
        let (mut slx, mut sl2) = (0.0f32, 0.0f32);
        for &v in x {
            let li = level(iscale, v);
            let w = v * v;
            slx += w * v * li as f32;
            sl2 += w * (li * li) as f32;
        }
        if sl2 > 0.0 && slx * slx > best * sl2 {
            for i in 0..x.len() {
                l[i] = level(iscale, x[i]) as i8;
            }
            scale = slx / sl2;
            best = scale * slx;
        }
    }
    scale
}

fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

fn quantize_block_k45(x: &[f32], q5: bool, out: &mut [u8]) {
    let (nmax, rmin, nstep) = if q5 { (31, -0.5, 15) } else { (15, -1.0, 20) };
    let (mut l, mut laux) = ([0u8; QK_K], [0u8; 32]);
    let (mut weights, mut scales, mut mins) = ([0f32; 32], [0f32; 8], [0f32; 8]);
    for j in 0..8 {
        let xs = &x[32 * j..32 * j + 32];
        let av_x = (xs.iter().map(|v| v * v).sum::<f32>() / 32.0).sqrt();
        for (w, v) in weights.iter_mut().zip(xs) {
            *w = av_x + v.abs();
        }
        let (s, m) = make_qkx2(xs, &weights, nmax, &mut l[32 * j..32 * j + 32], &mut laux, rmin, 0.1, nstep);
        scales[j] = s;
        mins[j] = m;
    }
    let max_scale = scales.iter().cloned().fold(0.0f32, f32::max);
    let max_min = mins.iter().cloned().fold(0.0f32, f32::max);
    let inv_scale = if max_scale > 0.0 { 63.0 / max_scale } else { 0.0 };
    let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
    let mut sc = [0u8; 12];
    for j in 0..8 {
        let ls = nearest_int(inv_scale * scales[j]).min(63) as u8;
        let lm = nearest_int(inv_min * mins[j]).min(63) as u8;
        if j < 4 {
            sc[j] = ls;
            sc[j + 4] = lm;
        } else {
            sc[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            sc[j - 4] |= (ls >> 4) << 6;
            sc[j] |= (lm >> 4) << 6;
        }
    }
    let d_h = f32_to_f16(max_scale / 63.0);
    let dmin_h = f32_to_f16(max_min / 63.0);
    out[0..2].copy_from_slice(&d_h.to_le_bytes());
    out[2..4].copy_from_slice(&dmin_h.to_le_bytes());
    out[4..16].copy_from_slice(&sc);
    let (d_all, dmin_all) = (f16_to_f32(d_h), f16_to_f32(dmin_h));
    let top = if q5 { 31 } else { 15 };
    for j in 0..8 {
        let (s, m) = scale_min_k4(j, &sc);
        let d = d_all * s as f32;
        if d == 0.0 {
            continue;
        }
        let dm = dmin_all * m as f32;
        for ii in 0..32 {
            l[32 * j + ii] = nearest_int((x[32 * j + ii] + dm) / d).clamp(0, top) as u8;
        }
    }
    if q5 {
        let (head, rest) = out[16..].split_at_mut(32);
        head.fill(0);
        let (mut m1, mut m2) = (1u8, 2u8);
        for (chunk, n) in (0..QK_K).step_by(64).enumerate() {
            for j in 0..32 {
                let (mut l1, mut l2) = (l[n + j], l[n + j + 32]);
                if l1 > 15 {
                    l1 -= 16;
                    head[j] |= m1;
                }
                if l2 > 15 {
                    l2 -= 16;
                    head[j] |= m2;
                }
                rest[chunk * 32 + j] = l1 | (l2 << 4);
            }
            m1 <<= 2;
            m2 <<= 2;
        }
    } else {
        for (chunk, n) in (0..QK_K).step_by(64).enumerate() {
            for j in 0..32 {
                out[16 + chunk * 32 + j] = l[n + j] | (l[n + j + 32] << 4);
            }
        }
    }
}

fn quantize_block_q6(x: &[f32], out: &mut [u8]) {
    let (mut l, mut scales) = ([0i8; QK_K], [0f32; 16]);
    let (mut max_scale, mut max_abs) = (0.0f32, 0.0f32);
    for ib in 0..16 {
        let s = make_qx(&x[16 * ib..16 * ib + 16], 32, &mut l[16 * ib..16 * ib + 16]);
        scales[ib] = s;
        if s.abs() > max_abs {
            max_abs = s.abs();
            max_scale = s;
        }
    }
    out.fill(0);
    if max_abs < 1e-15 {
        return;
    }
    let iscale = -128.0 / max_scale;
    let d_h = f32_to_f16(1.0 / iscale);
    out[208..210].copy_from_slice(&d_h.to_le_bytes());
    let d_all = f16_to_f32(d_h);
    let mut sc = [0i8; 16];
    for ib in 0..16 {
        sc[ib] = nearest_int(iscale * scales[ib]).min(127) as i8;
        out[192 + ib] = sc[ib] as u8;
    }
    let mut lq = [0u8; QK_K];
    for j in 0..16 {
        let d = d_all * sc[j] as f32;
        for ii in 0..16 {
            let li = if d == 0.0 { 0 } else { nearest_int(x[16 * j + ii] / d).clamp(-32, 31) };
            lq[16 * j + ii] = (li + 32) as u8;
        }
    }
    let (ql, qh) = out[..192].split_at_mut(128);
    for (chunk, j) in (0..QK_K).step_by(128).enumerate() {
        for k in 0..32 {
            let q = |o: usize| lq[j + k + o];
            ql[chunk * 64 + k] = (q(0) & 0xF) | ((q(64) & 0xF) << 4);
            ql[chunk * 64 + k + 32] = (q(32) & 0xF) | ((q(96) & 0xF) << 4);
            qh[chunk * 32 + k] = (q(0) >> 4) | ((q(32) >> 4) << 2) | ((q(64) >> 4) << 4) | ((q(96) >> 4) << 6);
        }
    }
}

/// Quantize one row (`x.len()` a multiple of 256) into `out` (`kind.row_bytes(x.len())` bytes).
pub fn quantize_row(kind: KQuant, x: &[f32], out: &mut [u8]) {
    assert!(x.len() % QK_K == 0 && out.len() == kind.row_bytes(x.len()), "row shape");
    for (xb, ob) in x.chunks_exact(QK_K).zip(out.chunks_exact_mut(kind.block_bytes())) {
        match kind {
            KQuant::Q4K => quantize_block_k45(xb, false, ob),
            KQuant::Q5K => quantize_block_k45(xb, true, ob),
            KQuant::Q6K => quantize_block_q6(xb, ob),
        }
    }
}

/// Inverse of [`quantize_row`], used by tests and parity checks.
pub fn dequantize_row(kind: KQuant, data: &[u8], out: &mut [f32]) {
    for (ob, yb) in data.chunks_exact(kind.block_bytes()).zip(out.chunks_exact_mut(QK_K)) {
        match kind {
            KQuant::Q4K | KQuant::Q5K => dequant_k45(ob, kind == KQuant::Q5K, yb),
            KQuant::Q6K => dequant_q6(ob, yb),
        }
    }
}

fn dequant_k45(b: &[u8], q5: bool, y: &mut [f32]) {
    let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([b[2], b[3]]));
    let sc = &b[4..16];
    let (qh, qs) = if q5 { (&b[16..48], &b[48..176]) } else { (&b[0..0], &b[16..144]) };
    let (mut u1, mut u2) = (1u8, 2u8);
    for (chunk, n) in (0..QK_K).step_by(64).enumerate() {
        let (s1, m1) = scale_min_k4(2 * chunk, sc);
        let (s2, m2) = scale_min_k4(2 * chunk + 1, sc);
        let (d1, mm1, d2, mm2) = (d * s1 as f32, dmin * m1 as f32, d * s2 as f32, dmin * m2 as f32);
        for l in 0..32 {
            let q = qs[chunk * 32 + l];
            let (hi1, hi2) = if q5 { (if qh[l] & u1 != 0 { 16 } else { 0 }, if qh[l] & u2 != 0 { 16 } else { 0 }) } else { (0, 0) };
            y[n + l] = d1 * ((q & 0xF) as i32 + hi1) as f32 - mm1;
            y[n + l + 32] = d2 * ((q >> 4) as i32 + hi2) as f32 - mm2;
        }
        u1 <<= 2;
        u2 <<= 2;
    }
}

fn dequant_q6(b: &[u8], y: &mut [f32]) {
    let d = f16_to_f32(u16::from_le_bytes([b[208], b[209]]));
    for (chunk, n) in (0..QK_K).step_by(128).enumerate() {
        let (ql, qh, sc) = (&b[chunk * 64..], &b[128 + chunk * 32..], &b[192 + chunk * 8..]);
        for l in 0..32 {
            let is = l / 16;
            let v = |lo: u8, shift: u32| ((lo | (((qh[l] >> shift) & 3) << 4)) as i32) - 32;
            let (q1, q2) = (v(ql[l] & 0xF, 0), v(ql[l + 32] & 0xF, 2));
            let (q3, q4) = (v(ql[l] >> 4, 4), v(ql[l + 32] >> 4, 6));
            y[n + l] = d * (sc[is] as i8) as f32 * q1 as f32;
            y[n + l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
            y[n + l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
            y[n + l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(n: usize) -> Vec<f32> {
        let mut s = 0x2545F491u32;
        (0..n)
            .map(|i| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                let u = (s as f32 / u32::MAX as f32 - 0.5) * 2.0;
                u * (0.02 + 0.05 * ((i / 32) % 5) as f32) + 0.01
            })
            .collect()
    }

    fn rel_rmse(kind: KQuant, x: &[f32]) -> f32 {
        let mut q = vec![0u8; kind.row_bytes(x.len())];
        quantize_row(kind, x, &mut q);
        let mut y = vec![0f32; x.len()];
        dequantize_row(kind, &q, &mut y);
        let err: f32 = x.iter().zip(&y).map(|(a, b)| (a - b) * (a - b)).sum();
        let sig: f32 = x.iter().map(|a| a * a).sum();
        (err / sig).sqrt()
    }

    #[test]
    fn roundtrip_error_orders_by_bit_width() {
        let x = sample(4096);
        let (e4, e5, e6) = (rel_rmse(KQuant::Q4K, &x), rel_rmse(KQuant::Q5K, &x), rel_rmse(KQuant::Q6K, &x));
        assert!(e4 < 0.12 && e5 < 0.06 && e6 < 0.03, "{e4} {e5} {e6}");
        assert!(e6 < e5 && e5 < e4);
    }

    #[test]
    fn constant_and_zero_rows_are_exact_or_zero() {
        for kind in [KQuant::Q4K, KQuant::Q5K, KQuant::Q6K] {
            assert!(rel_rmse(kind, &vec![0.0; 256]).is_nan() || rel_rmse(kind, &vec![0.0; 256]) == 0.0);
            assert!(rel_rmse(kind, &vec![0.5; 256]) < 0.02);
        }
    }

    #[test]
    fn half_conversion_roundtrips_representable_values() {
        for v in [0.0f32, 1.0, -2.5, 0.000061035156, 65504.0, 0.333251953125] {
            assert_eq!(f16_to_f32(f32_to_f16(v)), v);
        }
        assert_eq!(f32_to_f16(1.0e-9), 0);
        assert_eq!(f32_to_f16(1.0e9), 0x7c00);
    }
}
