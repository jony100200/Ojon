//! Level tables and nearest-level assignment with comfy-kitchen's exact tie rule
//! (searchsorted-left insertion point, then the upper neighbour only when strictly closer).

/// The frozen Lloyd-Max codebook shipped as comfy-kitchen's `_FIXED_LUT`.
pub const FIXED_LUT: [f32; 16] = [
    -0.980602, -0.794529, -0.638165, -0.500986, -0.377321, -0.263187, -0.155210, -0.050720,
    0.052541, 0.156985, 0.265284, 0.379533, 0.502636, 0.638953, 0.794876, 0.980671,
];

/// Uniform 6-bit levels -31..=31.
pub const W6_LEVELS: i32 = 31;

pub fn uniform6() -> Vec<f32> {
    (-W6_LEVELS..=W6_LEVELS).map(|v| v as f32).collect()
}

/// Index of the nearest entry of a sorted (non-decreasing) table.
#[inline]
pub fn nearest(sorted: &[f32], t: f32) -> usize {
    let last = sorted.len() - 1;
    let pos = sorted.partition_point(|&v| v < t); // torch.searchsorted(side="left")
    let lo = pos.saturating_sub(1).min(last);
    let hi = pos.min(last);
    if (t - sorted[hi]).abs() < (t - sorted[lo]).abs() { hi } else { lo }
}

/// The int8 values each level decodes to for one group: round(level * s_rel) clamped to +-127.
pub fn grid_levels(table: &[f32], s_rel: f32, out: &mut Vec<f32>) {
    out.clear();
    out.extend(table.iter().map(|&l| (l * s_rel).round_ties_even().clamp(-127.0, 127.0)));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_matches_searchsorted_tie_rule() {
        let t = [-1.0, 0.0, 1.0];
        assert_eq!(nearest(&t, 0.5), 1); // tie -> lower
        assert_eq!(nearest(&t, 0.51), 2);
        assert_eq!(nearest(&t, -9.0), 0);
        assert_eq!(nearest(&t, 9.0), 2);
        let dup = [0.0, 0.0, 1.0, 1.0];
        assert_eq!(nearest(&dup, 1.0), 2); // insertion point is the first 1.0
        assert_eq!(nearest(&dup, 0.0), 0);
    }
}
