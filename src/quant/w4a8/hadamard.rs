//! ConvRot rotation: each 256-wide input group is multiplied by H^T, where H is the normalized
//! regular Hadamard matrix built as kron(H4, H4, ...). H is symmetric, so H^T = H, and the
//! Kronecker structure lets us apply it as log4(size) passes of 4x4 butterflies.

/// The 4x4 regular Hadamard base used by comfy-kitchen.
const H4: [[f32; 4]; 4] = [[1., 1., 1., -1.], [1., 1., -1., 1.], [1., -1., 1., 1.], [-1., 1., 1., 1.]];

/// Size must be a power of 4 (comfy-kitchen's default ConvRot group is 256).
pub fn check_size(size: usize) -> Result<(), String> {
    let mut s = size;
    while s > 1 && s % 4 == 0 {
        s /= 4;
    }
    if size < 4 || s != 1 {
        return Err(format!("regular Hadamard size must be a power of 4, got {size}"));
    }
    Ok(())
}

/// Rotate one group in place: out = H @ g (H symmetric), scaled by 1/sqrt(size).
pub fn rotate_group(g: &mut [f32], scratch: &mut [f32]) {
    let n = g.len();
    let mut stride = 1;
    while stride < n {
        // One Kronecker factor: mix the 4 elements that differ only in this base-4 digit.
        for block in (0..n).step_by(stride * 4) {
            for off in 0..stride {
                let i = block + off;
                let x = [g[i], g[i + stride], g[i + 2 * stride], g[i + 3 * stride]];
                for (r, row) in H4.iter().enumerate() {
                    scratch[i + r * stride] = row[0] * x[0] + row[1] * x[1] + row[2] * x[2] + row[3] * x[3];
                }
            }
        }
        g.copy_from_slice(&scratch[..n]);
        stride *= 4;
    }
    let scale = 1.0 / (n as f32).sqrt();
    for v in g.iter_mut() {
        *v *= scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense(size: usize) -> Vec<Vec<f32>> {
        // kron(H4, H4, ...) entry = product of H4 entries over the base-4 digits.
        (0..size)
            .map(|a| {
                (0..size)
                    .map(|b| {
                        let (mut x, mut y, mut p) = (a, b, 1.0);
                        for _ in 0..size.trailing_zeros() / 2 {
                            p *= H4[x % 4][y % 4];
                            x /= 4;
                            y /= 4;
                        }
                        p / (size as f32).sqrt()
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn butterflies_match_dense_kron_and_are_orthonormal() {
        let size = 64;
        let h = dense(size);
        let input: Vec<f32> = (0..size).map(|i| ((i * 37 % 11) as f32) - 5.0).collect();
        let mut g = input.clone();
        let mut s = vec![0.0; size];
        rotate_group(&mut g, &mut s);
        for (j, row) in h.iter().enumerate() {
            let want: f32 = row.iter().zip(&input).map(|(a, b)| a * b).sum();
            assert!((g[j] - want).abs() < 1e-4, "row {j}: {} vs {want}", g[j]);
        }
        let n0: f32 = input.iter().map(|v| v * v).sum();
        let n1: f32 = g.iter().map(|v| v * v).sum();
        assert!((n0 - n1).abs() / n0 < 1e-5);
        assert!(check_size(256).is_ok() && check_size(128).is_err());
    }
}
