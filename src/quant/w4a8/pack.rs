//! comfy-kitchen storage contract: a K/2-byte nibble plane (even column = low nibble), and at
//! 6 bits a following K/4-byte plane holding each code's top 2 bits (col c: byte c/4, bit 2*(c%4)).

use super::quantize::Bits;

pub fn pack_row(codes: &[u8], bits: Bits, out: &mut [u8]) {
    let k = codes.len();
    for (j, pair) in codes.chunks_exact(2).enumerate() {
        out[j] = (pair[0] & 0xF) | ((pair[1] & 0xF) << 4);
    }
    if bits == Bits::Six {
        for (b, quad) in codes.chunks_exact(4).enumerate() {
            out[k / 2 + b] = quad.iter().enumerate().fold(0u8, |acc, (j, &c)| acc | (((c >> 4) & 0x3) << (2 * j)));
        }
    }
}

pub fn unpack_row(packed: &[u8], k: usize, bits: Bits) -> Vec<u8> {
    let mut codes = vec![0u8; k];
    for c in 0..k {
        let byte = packed[c / 2];
        codes[c] = if c % 2 == 0 { byte & 0xF } else { byte >> 4 };
        if bits == Bits::Six {
            codes[c] |= ((packed[k / 2 + c / 4] >> (2 * (c % 4))) & 0x3) << 4;
        }
    }
    codes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_round_trips_4_and_6_bit() {
        let c4: Vec<u8> = (0..64).map(|i| (i * 5 % 16) as u8).collect();
        let mut p4 = vec![0u8; 32];
        pack_row(&c4, Bits::Four, &mut p4);
        assert_eq!(unpack_row(&p4, 64, Bits::Four), c4);
        assert_eq!(p4[0], 0x50); // codes 0 (low) and 5 (high)

        let c6: Vec<u8> = (0..64).map(|i| (i * 11 % 63 + 1) as u8).collect();
        let mut p6 = vec![0u8; 48];
        pack_row(&c6, Bits::Six, &mut p6);
        assert_eq!(unpack_row(&p6, 64, Bits::Six), c6);
    }
}
