// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Pure Rust RoPE (Rotary Position Embeddings 2D/3D), Timestep Modulation, and AdaLN-Zero

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

/// Generate 3D RoPE coordinates for Flux.1 (txt_ids + img_ids)
/// axes_dim: [16, 56, 56] (Total pe_dim = 128 = head_dim)
pub fn create_flux_rope_embeddings(
    txt_len: usize,
    h_patches: usize,
    w_patches: usize,
    theta: f64,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let img_len = h_patches * w_patches;
    let axes_dim = [16, 56, 56]; // time/text, height, width

    // 1. Text IDs: [txt_len, 3] -> (0, 0, 0)
    let mut txt_ids_vec = vec![0f32; txt_len * 3];
    // 2. Image IDs: [img_len, 3] -> (0, row, col)
    let mut img_ids_vec = Vec::with_capacity(img_len * 3);
    for row in 0..h_patches {
        for col in 0..w_patches {
            img_ids_vec.push(0f32);
            img_ids_vec.push(row as f32);
            img_ids_vec.push(col as f32);
        }
    }

    let mut combined_ids_vec = txt_ids_vec;
    combined_ids_vec.extend(img_ids_vec);
    let total_seq = txt_len + img_len;

    let mut cos_parts = Vec::new();
    let mut sin_parts = Vec::new();

    // Compute RoPE for each axis
    for (axis_idx, &dim) in axes_dim.iter().enumerate() {
        let half_dim = dim / 2;
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / (theta.powf((i * 2) as f64 / dim as f64) as f32))
            .collect();
        let inv_freq_t = Tensor::from_vec(inv_freq, (half_dim,), device)?;

        let axis_coords: Vec<f32> = (0..total_seq)
            .map(|i| combined_ids_vec[i * 3 + axis_idx])
            .collect();
        let axis_coords_t = Tensor::from_vec(axis_coords, (total_seq, 1), device)?;

        // [total_seq, half_dim]
        let freqs = axis_coords_t.matmul(&inv_freq_t.unsqueeze(0)?)?;
        let cos_axis = freqs.cos()?;
        let sin_axis = freqs.sin()?;

        cos_parts.push(cos_axis);
        sin_parts.push(sin_axis);
    }

    let cos_slices: Vec<&Tensor> = cos_parts.iter().collect();
    let sin_slices: Vec<&Tensor> = sin_parts.iter().collect();

    let freqs_cos = Tensor::cat(&cos_slices, 1)?; // [total_seq, 64]
    let freqs_sin = Tensor::cat(&sin_slices, 1)?; // [total_seq, 64]

    Ok((freqs_cos, freqs_sin))
}

/// Apply 2D/3D Rotary Position Embeddings (RoPE) in-place to Q or K tensors matching Black Forest Labs math.py.
/// x: [B, Seq_Len, Heads, Head_Dim]
/// freqs_cos, freqs_sin: [Seq_Len, Head_Dim / 2]
pub fn apply_rope(x: &Tensor, freqs_cos: &Tensor, freqs_sin: &Tensor) -> Result<Tensor> {
    let (b, seq_len, heads, head_dim) = x.dims4()?;
    let half_dim = head_dim / 2;
    let orig_dtype = x.dtype();

    // 1. Reshape x to [B, Seq_Len, Heads, Half_Dim, 2]
    let x_pairs = x.reshape((b, seq_len, heads, half_dim, 2))?.to_dtype(DType::F32)?;
    let x0 = x_pairs.narrow(4, 0, 1)?.squeeze(4)?; // [B, Seq_Len, Heads, Half_Dim]
    let x1 = x_pairs.narrow(4, 1, 1)?.squeeze(4)?; // [B, Seq_Len, Heads, Half_Dim]

    // 2. Broadcast cos and sin: [1, Seq_Len, 1, Half_Dim]
    let cos = freqs_cos.reshape((1, seq_len, 1, half_dim))?.to_dtype(DType::F32)?;
    let sin = freqs_sin.reshape((1, seq_len, 1, half_dim))?.to_dtype(DType::F32)?;

    // 3. Matrix-vector product with [[cos, -sin], [sin, cos]]:
    // out0 = cos * x0 - sin * x1
    // out1 = sin * x0 + cos * x1
    let out0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?.unsqueeze(4)?;
    let out1 = (x0.broadcast_mul(&sin)? + x1.broadcast_mul(&cos)?)?.unsqueeze(4)?;

    // 4. Cat and reshape back to [B, Seq_Len, Heads, Head_Dim]
    let out = Tensor::cat(&[&out0, &out1], 4)?;
    out.reshape((b, seq_len, heads, head_dim))?.to_dtype(orig_dtype)
}

/// Adaptive Layer Normalization with Zero-Initialization (AdaLN-Zero) modulation.
#[derive(Debug, Clone)]
pub struct AdaLNZeroModulation {
    linear: Linear,
}

impl AdaLNZeroModulation {
    pub fn new(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        let linear = linear(in_dim, out_dim, vb.pp("lin"))?;
        Ok(Self { linear })
    }

    /// Predict scale, shift, and gate multipliers from conditioning vector (temb + text_emb).
    /// Output chunks: (shift, scale, gate)
    pub fn modulate(&self, conditioning: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let act = candle_nn::ops::silu(conditioning)?;
        let proj = self.linear.forward(&act)?;
        let chunks = proj.chunk(3, proj.dims().len() - 1)?;
        Ok((chunks[0].clone(), chunks[1].clone(), chunks[2].clone()))
    }

    /// Predict 6 parameters for DoubleStreamBlock: (shift_qkv, scale_qkv, gate_qkv, shift_mlp, scale_mlp, gate_mlp)
    pub fn modulate_double(&self, conditioning: &Tensor) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let act = candle_nn::ops::silu(conditioning)?;
        let proj = self.linear.forward(&act)?;
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
    in_layer: Linear,
    out_layer: Linear,
    freq_dim: usize,
}

impl TimestepEmbedder {
    pub fn new(hidden_dim: usize, freq_dim: usize, vb: VarBuilder) -> Result<Self> {
        let in_layer = linear(freq_dim, hidden_dim, vb.pp("in_layer"))?;
        let out_layer = linear(hidden_dim, hidden_dim, vb.pp("out_layer"))?;
        Ok(Self {
            in_layer,
            out_layer,
            freq_dim,
        })
    }

    pub fn forward(&self, timesteps: &Tensor) -> Result<Tensor> {
        let device = timesteps.device();
        let dtype = DType::F32;
        let half_dim = self.freq_dim / 2;
        
        // Exact Black Forest Labs timestep_embedding implementation:
        // t = time_factor (1000.0) * t
        // freqs = exp(-ln(10000.0) * (0..half_dim) / half_dim)
        // emb = cat([cos(args), sin(args)], dim=-1)
        let max_period: f64 = 10000.0;
        let time_factor: f64 = 1000.0;

        let freqs: Vec<f32> = (0..half_dim)
            .map(|i| (-(max_period.ln()) * (i as f64) / (half_dim as f64)).exp() as f32)
            .collect();
        let freqs_t = Tensor::from_vec(freqs, (1, half_dim), device)?.to_dtype(dtype)?;

        let timesteps_scaled = (timesteps.to_dtype(dtype)? * time_factor)?;
        let timesteps_f32 = timesteps_scaled.unsqueeze(1)?;
        let args = timesteps_f32.matmul(&freqs_t)?;

        let cos = args.cos()?;
        let sin = args.sin()?;
        let emb = Tensor::cat(&[&cos, &sin], 1)?.to_dtype(self.in_layer.weight().dtype())?;

        let h = self.in_layer.forward(&emb)?;
        let h = candle_nn::ops::silu(&h)?;
        self.out_layer.forward(&h)
    }
}
