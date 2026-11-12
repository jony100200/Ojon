//! Scalar formats used by the W4A8/W6A8 layout: bf16/f16 decode, bf16 rounding and fp8 e4m3fn.

/// Decode one little-endian source element to f32.
pub fn decode(dtype: &str, b: &[u8]) -> f32 {
    match dtype {
        "BF16" => f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16),
        "F16" => f16_to_f32(u16::from_le_bytes([b[0], b[1]])),
        _ => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
    }
}

/// Bytes per element for the floating source dtypes the converter reads.
pub fn source_width(dtype: &str) -> Option<usize> {
    match dtype {
        "BF16" | "F16" => Some(2),
        "F32" => Some(4),
        _ => None,
    }
}

/// Round an f32 to the nearest bf16 (ties to even), returned as f32.
pub fn round_bf16(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    let bits = x.to_bits();
    let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
    f32::from_bits(rounded & 0xFFFF_0000)
}

/// Round an f32 to the nearest f16 (ties to even), returned as f32.
pub fn round_f16(x: f32) -> f32 {
    f16_to_f32(f32_to_f16(x))
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1F) as i32;
    let man = (h & 0x3FF) as u32;
    let mag = match exp {
        0 => (man as f32) * 2f32.powi(-24),
        31 => if man == 0 { f32::INFINITY } else { f32::NAN },
        _ => f32::from_bits((((exp - 15 + 127) as u32) << 23) | (man << 13)),
    };
    f32::from_bits(mag.to_bits() | sign)
}

fn f32_to_f16(x: f32) -> u16 {
    let sign = ((x.to_bits() >> 16) & 0x8000) as u16;
    let a = x.abs();
    if a.is_nan() {
        return sign | 0x7E00;
    }
    if a >= 65520.0 {
        return sign | 0x7C00;
    }
    if a < 2f32.powi(-14) {
        let m = (a / 2f32.powi(-24)).round_ties_even() as u16;
        return sign | m;
    }
    let e = a.log2().floor() as i32;
    let step = 2f32.powi(e - 10);
    let q = (a / step).round_ties_even() * step;
    let e2 = q.log2().floor() as i32;
    let man = ((q / 2f32.powi(e2 - 10)) as u32 - 1024) as u16;
    sign | (((e2 + 15) as u16) << 10) | man
}

/// e4m3fn: bias 7, max finite 448, no infinity, 0x7F/0xFF are NaN. Ties to even, like torch.
pub fn f32_to_e4m3(x: f32) -> u8 {
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    let a = x.abs();
    if a.is_nan() || a > 464.0 {
        return sign | 0x7F;
    }
    if a < 2f32.powi(-6) {
        let m = (a / 2f32.powi(-9)).round_ties_even() as u8; // 8 reaches the smallest normal
        return sign | m;
    }
    let e = a.log2().floor() as i32;
    let step = 2f32.powi(e - 3);
    let q = (a / step).round_ties_even() * step;
    if q > 448.0 {
        return sign | 0x7E;
    }
    let e2 = q.log2().floor() as i32;
    let man = (q / 2f32.powi(e2 - 3)) as u32 - 8;
    sign | (((e2 + 7) as u8) << 3) | man as u8
}

pub fn e4m3_to_f32(v: u8) -> f32 {
    let sign = if v & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = ((v >> 3) & 0xF) as i32;
    let man = (v & 0x7) as f32;
    if v & 0x7F == 0x7F {
        return f32::NAN;
    }
    sign * if exp == 0 { man * 2f32.powi(-9) } else { (1.0 + man / 8.0) * 2f32.powi(exp - 7) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_round_trips_every_finite_code() {
        for v in 0u8..=0x7E {
            assert_eq!(f32_to_e4m3(e4m3_to_f32(v)), v, "code {v:#x}");
        }
        assert_eq!(e4m3_to_f32(0x7E), 448.0);
        assert_eq!(f32_to_e4m3(1.0), 0x38);
    }

    #[test]
    fn e4m3_ties_go_to_even() {
        // 1.0625 sits halfway between 1.0 (even mantissa 0) and 1.125 (odd mantissa 1)
        assert_eq!(e4m3_to_f32(f32_to_e4m3(1.0625)), 1.0);
        assert_eq!(e4m3_to_f32(f32_to_e4m3(1.1875)), 1.25);
    }

    #[test]
    fn bf16_and_f16_rounding() {
        assert_eq!(round_bf16(1.0 + 1.0 / 512.0), 1.0); // tie -> even
        assert_eq!(round_f16(1.0 + 1.0 / 4096.0), 1.0);
        assert_eq!(decode("BF16", &[0x80, 0x3F]), 1.0);
        assert_eq!(decode("F16", &[0x00, 0x3C]), 1.0);
    }
}
