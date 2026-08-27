// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Pure Rust RoPE (Rotary Position Embeddings 2D/3D), Timestep Modulation, and AdaLN-Zero

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

/// Generate 3D/4D RoPE coordinates for Flux.1 [16, 56, 56] and Flux.2 [32, 32, 32, 32] (txt_ids + img_ids)
pub fn create_flux_rope_embeddings(
    txt_len: usize,
    h_patches: usize,
    w_patches: usize,
    axes_dim: &[usize],
    theta: f64,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let img_len = h_patches * w_patches;
    let num_axes = axes_dim.len();

    // 1. Text IDs: [txt_len, num_axes]
    // Flux.2 reference _prepare_text_ids: (T=0, H=0, W=0, L=token_index)
    // Flux.1 text ids: (0, 0, 0) collision-free; text length axis is the last one.
    let mut txt_ids_vec = Vec::with_capacity(txt_len * num_axes);
    for i in 0..txt_len {
        for axis in 0..num_axes {
            if num_axes == 4 && axis == num_axes - 1 {
                txt_ids_vec.push(i as f32);
            } else {
                txt_ids_vec.push(0f32);
            }
        }
    }
    
    // 2. Image IDs: [img_len, num_axes]
    let mut img_ids_vec = Vec::with_capacity(img_len * num_axes);
    if num_axes == 4 {
        // Flux.2 4D axes: [time (T), height (Y), width (X), canvas/ref (Ref)]
        for row in 0..h_patches {
            for col in 0..w_patches {
                img_ids_vec.push(0f32);       // Axis 1: Time (T)
                img_ids_vec.push(row as f32); // Axis 2: Height (Y)
                img_ids_vec.push(col as f32); // Axis 3: Width (X)
                img_ids_vec.push(0f32);       // Axis 4: Canvas / Ref ID
            }
        }
    } else {
        // Flux.1 3D axes: [time/text, row, col]
        for row in 0..h_patches {
            for col in 0..w_patches {
                img_ids_vec.push(0f32);
                img_ids_vec.push(row as f32);
                img_ids_vec.push(col as f32);
            }
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
            .map(|i| combined_ids_vec[i * num_axes + axis_idx])
            .collect();
        let axis_coords_t = Tensor::from_vec(axis_coords, (total_seq, 1), device)?;

        // [total_seq, half_dim]
        let freqs = axis_coords_t.matmul(&inv_freq_t.unsqueeze(0)?)?;
        let cos_half = freqs.cos()?;
        let sin_half = freqs.sin()?;

        // repeat_interleave(2, dim=-1) matching Diffusers get_1d_rotary_pos_embed:
        // [cos0, cos1, ...] -> [[cos0, cos0], [cos1, cos1], ...] -> [total_seq, dim]
        let cos_axis = Tensor::cat(&[&cos_half.unsqueeze(2)?, &cos_half.unsqueeze(2)?], 2)?
            .reshape((total_seq, dim))?;
        let sin_axis = Tensor::cat(&[&sin_half.unsqueeze(2)?, &sin_half.unsqueeze(2)?], 2)?
            .reshape((total_seq, dim))?;

        cos_parts.push(cos_axis);
        sin_parts.push(sin_axis);
    }

    let cos_slices: Vec<&Tensor> = cos_parts.iter().collect();
    let sin_slices: Vec<&Tensor> = sin_parts.iter().collect();

    let freqs_cos = Tensor::cat(&cos_slices, 1)?; // [total_seq, sum(axes_dim)]
    let freqs_sin = Tensor::cat(&sin_slices, 1)?; // [total_seq, sum(axes_dim)]

    Ok((freqs_cos, freqs_sin))
}

/// Apply 2D/3D Rotary Position Embeddings (RoPE) in-place to Q or K tensors matching Black Forest Labs math.py.
/// x: [B, Seq_Len, Heads, Head_Dim]
/// freqs_cos, freqs_sin: [Seq_Len, Head_Dim / 2]
pub fn apply_rope(x: &Tensor, freqs_cos: &Tensor, freqs_sin: &Tensor) -> Result<Tensor> {
    let (b, seq_len, heads, head_dim) = x.dims4()?;
    let half_dim = head_dim / 2;
    let orig_dtype = x.dtype();

    // Exact Diffusers apply_rotary_emb (use_real_unbind_dim = -1):
    // 1. Reshape x to [B, Seq_Len, Heads, Half_Dim, 2]
    let x_f32 = x.to_dtype(DType::F32)?;
    let x_pairs = x_f32.reshape((b, seq_len, heads, half_dim, 2))?;
    let x_real = x_pairs.narrow(4, 0, 1)?.squeeze(4)?; // [B, Seq_Len, Heads, Half_Dim]
    let x_imag = x_pairs.narrow(4, 1, 1)?.squeeze(4)?; // [B, Seq_Len, Heads, Half_Dim]

    // 2. x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    let neg_x_imag = (x_imag * -1.0)?.unsqueeze(4)?;
    let pos_x_real = x_real.unsqueeze(4)?;
    let x_rotated = Tensor::cat(&[&neg_x_imag, &pos_x_real], 4)?.reshape((b, seq_len, heads, head_dim))?;

    // 3. Duplicate cos and sin along last dimension: [freqs, freqs] -> [Seq_Len, Head_Dim]
    let cos_full = if freqs_cos.dim(freqs_cos.dims().len() - 1)? == half_dim {
        Tensor::cat(&[freqs_cos, freqs_cos], freqs_cos.dims().len() - 1)?
    } else {
        freqs_cos.clone()
    };
    let sin_full = if freqs_sin.dim(freqs_sin.dims().len() - 1)? == half_dim {
        Tensor::cat(&[freqs_sin, freqs_sin], freqs_sin.dims().len() - 1)?
    } else {
        freqs_sin.clone()
    };

    let cos = cos_full.reshape((1, seq_len, 1, head_dim))?.to_dtype(DType::F32)?;
    let sin = sin_full.reshape((1, seq_len, 1, head_dim))?.to_dtype(DType::F32)?;

    // 4. out = (x * cos + x_rotated * sin)
    let out = (x_f32.broadcast_mul(&cos)? + x_rotated.broadcast_mul(&sin)?)?;
    out.to_dtype(orig_dtype)
}

/// Adaptive Layer Normalization with Zero-Initialization (AdaLN-Zero) modulation.
#[derive(Debug, Clone)]
pub struct AdaLNZeroModulation {
    linear: Linear,
}

impl AdaLNZeroModulation {
    pub fn new(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        let linear = candle_nn::linear(in_dim, out_dim, vb.pp("lin"))
            .or_else(|_| candle_nn::linear_no_bias(in_dim, out_dim, vb.pp("lin")))?;
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
        let linear_layer = |in_d: usize, out_d: usize, path: VarBuilder| -> Result<Linear> {
            candle_nn::linear(in_d, out_d, path.clone()).or_else(|_| candle_nn::linear_no_bias(in_d, out_d, path))
        };
        let in_layer = linear_layer(freq_dim, hidden_dim, vb.pp("in_layer"))?;
        let out_layer = linear_layer(hidden_dim, hidden_dim, vb.pp("out_layer"))?;
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
