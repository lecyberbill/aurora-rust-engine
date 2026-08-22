// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Pure Rust RoPE (Rotary Position Embeddings 2D/3D), Timestep Modulation, and AdaLN-Zero

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

/// Compute Rotary Position Embeddings (RoPE) frequencies for 2D/3D grids (Image x/y + Time/Text).
pub fn create_rope_frequencies(
    dim: usize,
    max_positions: usize,
    theta: f64,
    device: &Device,
) -> Result<Tensor> {
    let half_dim = dim / 2;
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / (theta.powf((i * 2) as f64 / dim as f64) as f32))
        .collect();
    let inv_freq_t = Tensor::from_vec(inv_freq, (half_dim,), device)?;

    let pos: Vec<f32> = (0..max_positions).map(|i| i as f32).collect();
    let pos_t = Tensor::from_vec(pos, (max_positions, 1), device)?;

    // Outer product: [max_positions, half_dim]
    let freqs = pos_t.matmul(&inv_freq_t.unsqueeze(0)?)?;
    Ok(freqs)
}

/// Apply 2D/3D Rotary Position Embeddings (RoPE) in-place to Q or K tensors.
pub fn apply_rope(x: &Tensor, freqs_cos: &Tensor, freqs_sin: &Tensor) -> Result<Tensor> {
    let (b, seq_len, _heads, head_dim) = x.dims4()?;
    let half_dim = head_dim / 2;

    let x1 = x.narrow(3, 0, half_dim)?;
    let x2 = x.narrow(3, half_dim, half_dim)?;

    // Reshape frequencies for broadcasting across heads: [B, Seq_Len, 1, Half_Dim]
    let cos = freqs_cos.reshape((b, seq_len, 1, half_dim))?;
    let sin = freqs_sin.reshape((b, seq_len, 1, half_dim))?;

    // Standard RoPE rotation: [x1 * cos - x2 * sin, x2 * cos + x1 * sin]
    let rx1 = ((&x1 * &cos)? - (&x2 * &sin)?)?;
    let rx2 = ((&x2 * &cos)? + (&x1 * &sin)?)?;

    Tensor::cat(&[&rx1, &rx2], 3)
}

/// Adaptive Layer Normalization with Zero-Initialization (AdaLN-Zero) modulation.
#[derive(Debug, Clone)]
pub struct AdaLNZeroModulation {
    linear: Linear,
}

impl AdaLNZeroModulation {
    pub fn new(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        let linear = linear(in_dim, out_dim, vb.pp("linear"))?;
        Ok(Self { linear })
    }

    /// Predict scale, shift, and gate multipliers from conditioning vector (temb + text_emb).
    /// Output chunks: (shift, scale, gate)
    pub fn modulate(&self, conditioning: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let proj = self.linear.forward(conditioning)?;
        let chunks = proj.chunk(3, proj.dims().len() - 1)?;
        Ok((chunks[0].clone(), chunks[1].clone(), chunks[2].clone()))
    }

    /// Predict 6 parameters for DoubleStreamBlock: (shift_qkv, scale_qkv, gate_qkv, shift_mlp, scale_mlp, gate_mlp)
    pub fn modulate_double(&self, conditioning: &Tensor) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let proj = self.linear.forward(conditioning)?;
        let chunks = proj.chunk(6, proj.dims().len() - 1)?;
        Ok((
            chunks[0].clone(), chunks[1].clone(), chunks[2].clone(),
            chunks[3].clone(), chunks[4].clone(), chunks[5].clone(),
        ))
    }
}

/// Timestep & Guidance Embedding module for Rectified Flow / Diffusion Transformers.
#[derive(Debug, Clone)]
pub struct TimestepEmbedder {
    linear_1: Linear,
    linear_2: Linear,
    freq_dim: usize,
}

impl TimestepEmbedder {
    pub fn new(hidden_dim: usize, freq_dim: usize, vb: VarBuilder) -> Result<Self> {
        let linear_1 = linear(freq_dim, hidden_dim, vb.pp("linear_1"))?;
        let linear_2 = linear(hidden_dim, hidden_dim, vb.pp("linear_2"))?;
        Ok(Self {
            linear_1,
            linear_2,
            freq_dim,
        })
    }

    pub fn forward(&self, timesteps: &Tensor) -> Result<Tensor> {
        let device = timesteps.device();
        let dtype = DType::F32;
        let half_dim = self.freq_dim / 2;
        
        let freq_factor = -(std::f64::consts::LN_2 * 2.0 / half_dim as f64);
        let freqs: Vec<f32> = (0..half_dim)
            .map(|i| (freq_factor * i as f64).exp() as f32)
            .collect();
        let freqs_t = Tensor::from_vec(freqs, (1, half_dim), device)?.to_dtype(dtype)?;

        let timesteps_f32 = timesteps.to_dtype(dtype)?.unsqueeze(1)?;
        let args = timesteps_f32.matmul(&freqs_t)?;

        let sin = args.sin()?;
        let cos = args.cos()?;
        let emb = Tensor::cat(&[&sin, &cos], 1)?;

        let h = self.linear_1.forward(&emb)?;
        let h = candle_nn::ops::silu(&h)?;
        self.linear_2.forward(&h)
    }
}
