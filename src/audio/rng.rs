// [WFGY] Zone: SAFE | λ: 0.15 | Fallbacks: 0 | Action: Deterministic seeded RNG (xoshiro256** + Box-Muller) for reproducible latent noise

//! Small deterministic RNG so a given `seed` always yields the same initial
//! latent noise, independent of the platform. Not intended to be
//! cryptographically secure.

use candle_core::{DType, Device, Result, Tensor};

/// xoshiro256** generator seeded via SplitMix64.
#[derive(Debug, Clone)]
pub struct SeededRng {
    s: [u64; 4],
    spare: Option<f32>,
}

impl SeededRng {
    pub fn new(seed: u64) -> Self {
        // SplitMix64 expansion of the 64-bit seed.
        let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut next = || {
            x = x.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        };
        let s = [next(), next(), next(), next()];
        Self { s, spare: None }
    }

    #[inline]
    fn rotl(x: u64, k: u64) -> u64 {
        (x << k) | (x >> (64 - k))
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = Self::rotl(self.s[1].wrapping_mul(5), 7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = Self::rotl(self.s[3], 45);
        result
    }

    /// Uniform sample in `[0, 1)`.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / (1u64 << 24) as f32
    }

    /// Standard normal sample via the polar Box-Muller transform.
    pub fn next_normal(&mut self) -> f32 {
        if let Some(z) = self.spare.take() {
            return z;
        }
        let mut u1 = self.next_f32();
        if u1 < 1e-12 {
            u1 = 1e-12;
        }
        let u2 = self.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        self.spare = Some(r * theta.sin());
        r * theta.cos()
    }
}

/// Sample `[1, frames, dim]` Gaussian noise `N(mean, std)` deterministically from `seed`.
pub fn seeded_randn(
    frames: usize,
    dim: usize,
    mean: f32,
    std: f32,
    seed: u64,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let mut rng = SeededRng::new(seed);
    let mut data = Vec::with_capacity(frames * dim);
    for _ in 0..(frames * dim) {
        data.push(mean + std * rng.next_normal());
    }
    Tensor::from_vec(data, (1, frames, dim), device)?.to_dtype(dtype)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_rng_is_reproducible() {
        let mut a = SeededRng::new(42);
        let mut b = SeededRng::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn seeded_rng_differs_between_seeds() {
        let mut a = SeededRng::new(1);
        let mut b = SeededRng::new(2);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn normal_samples_are_finite() {
        let mut r = SeededRng::new(7);
        for _ in 0..4096 {
            assert!(r.next_normal().is_finite());
        }
    }
}
