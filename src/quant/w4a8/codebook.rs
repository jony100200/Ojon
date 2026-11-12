//! 4-bit codebook choice: the frozen LUT unless the group-normalized rotated weight is
//! heavy-tailed (excess kurtosis above -0.1), then a data-free Lloyd-Max fit.
//! comfy-kitchen samples randomly with torch's RNG; we sample deterministically (even stride),
//! so a fitted table can differ slightly from theirs. The LUT path (all real layers so far) is exact.

use super::levels::FIXED_LUT;

const GATE_KURTOSIS: f64 = -0.1;
const FIT_LEVELS: usize = 16;
const FIT_ITERS: usize = 25;
const FIT_SAMPLE: usize = 300_000;

pub struct Choice {
    pub table: Vec<f32>,
    pub fitted: bool,
    pub excess_kurtosis: f64,
}

pub fn choose(normalized: &[f32]) -> Choice {
    let k = excess_kurtosis(normalized);
    if k <= GATE_KURTOSIS {
        return Choice { table: FIXED_LUT.to_vec(), fitted: false, excess_kurtosis: k };
    }
    Choice { table: fit(&strided(normalized, FIT_SAMPLE)), fitted: true, excess_kurtosis: k }
}

fn strided(x: &[f32], max: usize) -> Vec<f32> {
    if x.len() <= max {
        return x.to_vec();
    }
    let step = x.len() as f64 / max as f64;
    (0..max).map(|i| x[(i as f64 * step) as usize]).collect()
}

/// Excess kurtosis with torch's unbiased std, as in `_codebook_for`.
fn excess_kurtosis(x: &[f32]) -> f64 {
    let n = x.len() as f64;
    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = x.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    let sd = var.sqrt() + 1e-9;
    x.iter().map(|&v| ((v as f64 - mean) / sd).powi(4)).sum::<f64>() / n - 3.0
}

/// Lloyd-Max: start at the 16 quantiles (linear interpolation), then 25 assign/mean passes.
fn fit(samples: &[f32]) -> Vec<f32> {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let last = (sorted.len() - 1) as f64;
    let mut table: Vec<f32> = (0..FIT_LEVELS)
        .map(|i| {
            let p = i as f64 / (FIT_LEVELS - 1) as f64 * last;
            let (lo, hi) = (p.floor() as usize, p.ceil() as usize);
            (sorted[lo] as f64 + (sorted[hi] as f64 - sorted[lo] as f64) * (p - lo as f64)) as f32
        })
        .collect();
    for _ in 0..FIT_ITERS {
        let mut sum = [0f64; FIT_LEVELS];
        let mut cnt = [0usize; FIT_LEVELS];
        for &s in samples {
            let mut best = 0;
            for j in 1..FIT_LEVELS {
                if (s - table[j]).abs() < (s - table[best]).abs() {
                    best = j; // argmin keeps the first index on ties
                }
            }
            sum[best] += s as f64;
            cnt[best] += 1;
        }
        for j in 0..FIT_LEVELS {
            if cnt[j] > 0 {
                table[j] = (sum[j] / cnt[j] as f64) as f32;
            }
        }
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn light_tailed_data_keeps_the_frozen_lut() {
        // A uniform distribution has excess kurtosis -1.2.
        let x: Vec<f32> = (0..10_000).map(|i| (i as f32 / 5_000.0) - 1.0).collect();
        let c = choose(&x);
        assert!(!c.fitted && c.table == FIXED_LUT.to_vec());
    }

    #[test]
    fn heavy_tailed_data_gets_a_sorted_fitted_table() {
        let x: Vec<f32> = (0..20_000).map(|i| { let t = (i as f32 / 20_000.0) - 0.5; t * t * t * 8.0 }).collect();
        let c = choose(&x);
        assert!(c.fitted && c.table.windows(2).all(|w| w[0] <= w[1]));
    }
}
