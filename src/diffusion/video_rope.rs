// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Official CogVideoX 3D Spatio-Temporal RoPE Implementation
// Invariant: Exact crop region grid calculation, repeat_interleave(2) 1D freqs, pair-rotation apply_rope.

use candle_core::{DType, Device, Result, Tensor};

/// Compute resize and crop region for grid, matching diffusers `get_resize_crop_region_for_grid`
pub fn get_resize_crop_region_for_grid(
    src: (usize, usize),
    tgt_width: usize,
    tgt_height: usize,
) -> ((usize, usize), (usize, usize)) {
    let (h, w) = src;
    let r = h as f64 / w as f64;
    let (resize_height, resize_width) = if r > (tgt_height as f64 / tgt_width as f64) {
        let rh = tgt_height;
        let rw = ((tgt_height as f64 / h as f64) * w as f64).round() as usize;
        (rh, rw)
    } else {
        let rw = tgt_width;
        let rh = ((tgt_width as f64 / w as f64) * h as f64).round() as usize;
        (rh, rw)
    };

    let crop_top = (((tgt_height as f64 - resize_height as f64) / 2.0).round().max(0.0)) as usize;
    let crop_left = (((tgt_width as f64 - resize_width as f64) / 2.0).round().max(0.0)) as usize;

    ((crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width))
}

/// Compute 1D rotary position embeddings with repeat_interleave(2)
fn get_1d_rotary_pos_embed(
    dim: usize,
    pos: &[f32],
    theta: f64,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let half_dim = dim / 2;
    let s = pos.len();
    let mut freqs = Vec::with_capacity(half_dim);
    for i in 0..half_dim {
        let exponent = (2 * i) as f64 / dim as f64;
        freqs.push((1.0 / theta.powf(exponent)) as f32);
    }

    let mut cos_data = Vec::with_capacity(s * dim);
    let mut sin_data = Vec::with_capacity(s * dim);

    for &p in pos {
        for &f in &freqs {
            let angle = p * f;
            let c = angle.cos();
            let s_val = angle.sin();
            // repeat_interleave(2)
            cos_data.push(c);
            cos_data.push(c);
            sin_data.push(s_val);
            sin_data.push(s_val);
        }
    }

    let cos_t = Tensor::from_vec(cos_data, (s, dim), device)?;
    let sin_t = Tensor::from_vec(sin_data, (s, dim), device)?;
    Ok((cos_t, sin_t))
}

/// Compute 3D spatio-temporal rotary embeddings for CogVideoX (Time x Height x Width)
pub fn get_3d_rotary_pos_embed(
    embed_dim: usize,
    crops_coords: ((usize, usize), (usize, usize)),
    grid_size: (usize, usize),
    temporal_size: usize,
    theta: f64,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let (grid_size_h, grid_size_w) = grid_size;
    let ((start_h, start_w), (stop_h, stop_w)) = crops_coords;

    // grid_h
    let grid_h: Vec<f32> = if grid_size_h > 1 {
        let start = start_h as f32;
        let stop = (stop_h as f32) * ((grid_size_h - 1) as f32) / (grid_size_h as f32);
        (0..grid_size_h)
            .map(|i| start + (stop - start) * (i as f32) / ((grid_size_h - 1) as f32))
            .collect()
    } else {
        vec![start_h as f32]
    };

    // grid_w
    let grid_w: Vec<f32> = if grid_size_w > 1 {
        let start = start_w as f32;
        let stop = (stop_w as f32) * ((grid_size_w - 1) as f32) / (grid_size_w as f32);
        (0..grid_size_w)
            .map(|i| start + (stop - start) * (i as f32) / ((grid_size_w - 1) as f32))
            .collect()
    } else {
        vec![start_w as f32]
    };

    // grid_t
    let grid_t: Vec<f32> = if temporal_size > 1 {
        let stop = (temporal_size - 1) as f32;
        (0..temporal_size)
            .map(|i| stop * (i as f32) / ((temporal_size - 1) as f32))
            .collect()
    } else {
        vec![0.0f32]
    };

    let dim_t = embed_dim / 4;          // 16 for head_dim=64
    let dim_h = (embed_dim / 8) * 3;     // 24 for head_dim=64
    let dim_w = (embed_dim / 8) * 3;     // 24 for head_dim=64

    let (t_cos, t_sin) = get_1d_rotary_pos_embed(dim_t, &grid_t, theta, device)?;
    let (h_cos, h_sin) = get_1d_rotary_pos_embed(dim_h, &grid_h, theta, device)?;
    let (w_cos, w_sin) = get_1d_rotary_pos_embed(dim_w, &grid_w, theta, device)?;

    let total_tokens = temporal_size * grid_size_h * grid_size_w;

    let t_cos_exp = t_cos.unsqueeze(1)?.unsqueeze(2)?.broadcast_as((temporal_size, grid_size_h, grid_size_w, dim_t))?;
    let h_cos_exp = h_cos.unsqueeze(0)?.unsqueeze(2)?.broadcast_as((temporal_size, grid_size_h, grid_size_w, dim_h))?;
    let w_cos_exp = w_cos.unsqueeze(0)?.unsqueeze(0)?.broadcast_as((temporal_size, grid_size_h, grid_size_w, dim_w))?;

    let t_sin_exp = t_sin.unsqueeze(1)?.unsqueeze(2)?.broadcast_as((temporal_size, grid_size_h, grid_size_w, dim_t))?;
    let h_sin_exp = h_sin.unsqueeze(0)?.unsqueeze(2)?.broadcast_as((temporal_size, grid_size_h, grid_size_w, dim_h))?;
    let w_sin_exp = w_sin.unsqueeze(0)?.unsqueeze(0)?.broadcast_as((temporal_size, grid_size_h, grid_size_w, dim_w))?;

    let cos = Tensor::cat(&[&t_cos_exp, &h_cos_exp, &w_cos_exp], 3)?
        .reshape((total_tokens, embed_dim))?;
    let sin = Tensor::cat(&[&t_sin_exp, &h_sin_exp, &w_sin_exp], 3)?
        .reshape((total_tokens, embed_dim))?;

    Ok((cos, sin))
}

/// Apply CogVideoX rotary embedding to query or key tensor `[B, Heads, Seq_Video, Head_Dim]`
/// using cos and sin tensors of shape `[Seq_Video, Head_Dim]`.
/// Implements exact pair-wise Givens rotation: `[-x_imag, x_real]` with `cos` and `sin`.
pub fn apply_cogvideox_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b, h, s, d) = x.dims4()?;
    let half_d = d / 2;
    // Reshape [B, H, S, D] -> [B, H, S, D/2, 2]
    let x_pairs = x.reshape((b, h, s, half_d, 2))?;
    let x_real = x_pairs.narrow(4, 0, 1)?; // [B, H, S, D/2, 1]
    let x_imag = x_pairs.narrow(4, 1, 1)?; // [B, H, S, D/2, 1]
    let neg_imag = x_imag.neg()?;
    // x_rotated: stack([-x_imag, x_real], dim=-1).flatten(3)
    let x_rot = Tensor::cat(&[&neg_imag, &x_real], 4)?.reshape((b, h, s, d))?;

    let cos_exp = cos.unsqueeze(0)?.unsqueeze(0)?.broadcast_as((b, h, s, d))?;
    let sin_exp = sin.unsqueeze(0)?.unsqueeze(0)?.broadcast_as((b, h, s, d))?;

    let orig_dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let x_rot_f32 = x_rot.to_dtype(DType::F32)?;
    let cos_f32 = cos_exp.to_dtype(DType::F32)?;
    let sin_f32 = sin_exp.to_dtype(DType::F32)?;

    let out = (x_f32.broadcast_mul(&cos_f32)? + x_rot_f32.broadcast_mul(&sin_f32)?)?
        .to_dtype(orig_dtype)?;
    Ok(out)
}

/// Backwards compatible VideoRope3D wrapper
#[derive(Debug, Clone)]
pub struct VideoRope3D {
    pub dim_t: usize,
    pub dim_h: usize,
    pub dim_w: usize,
    pub theta: f64,
}

impl VideoRope3D {
    pub fn new(dim_t: usize, dim_h: usize, dim_w: usize, theta: f64) -> Self {
        Self { dim_t, dim_h, dim_w, theta }
    }

    /// Precompute combined (Cos, Sin) tensors for CogVideoX grid
    pub fn precompute_cogvideox(
        height: usize,
        width: usize,
        temporal_size: usize,
        head_dim: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let gh = height / 16;
        let gw = width / 16;
        let crops = get_resize_crop_region_for_grid((gh, gw), 360, 240);
        get_3d_rotary_pos_embed(head_dim, crops, (gh, gw), temporal_size, 10000.0, device)
    }

    pub fn precompute_3d_freqs(
        &self,
        num_frames: usize,
        grid_h: usize,
        grid_w: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let crops = get_resize_crop_region_for_grid((grid_h, grid_w), 360, 240);
        get_3d_rotary_pos_embed(self.dim_t + self.dim_h + self.dim_w, crops, (grid_h, grid_w), num_frames, self.theta, device)
    }

    pub fn apply_rope_3d(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        apply_cogvideox_rope(x, cos, sin)
    }
}

/// Compute 1D sinusoidal position embeddings: [M, dim] (sin followed by cos)
pub fn get_1d_sincos_pos_embed(dim: usize, pos: &[f64], device: &Device) -> Result<Tensor> {
    let half = dim / 2;
    let s = pos.len();
    let mut omega = Vec::with_capacity(half);
    for i in 0..half {
        omega.push(1.0 / 10000.0f64.powf(i as f64 / half as f64));
    }
    let mut data = Vec::with_capacity(s * dim);
    for &p in pos {
        // sin first, then cos
        for &om in &omega {
            data.push((p * om).sin() as f32);
        }
        for &om in &omega {
            data.push((p * om).cos() as f32);
        }
    }
    Tensor::from_vec(data, (s, dim), device)
}

/// Compute official CogVideoX 3D sinusoidal position embedding:
/// Returns Tensor `[1, text_len + T * H_p * W_p, embed_dim]` with first `text_len` tokens as zeros.
pub fn get_cogvideox_3d_sincos_pos_embed(
    embed_dim: usize, // 1920
    h_p: usize,       // 30
    w_p: usize,       // 45
    t: usize,         // 13
    text_len: usize,  // 226
    device: &Device,
) -> Result<Tensor> {
    let s_scale = 1.875f64;
    let t_scale = 1.0f64;
    let dim_t = embed_dim / 4; // 480
    let dim_s = (3 * embed_dim / 4) / 2; // 720

    let mut pos_t = Vec::with_capacity(t);
    for i in 0..t {
        pos_t.push(i as f64 / t_scale);
    }
    let emb_t = get_1d_sincos_pos_embed(dim_t, &pos_t, device)?; // [T, 480]

    let hw = h_p * w_p;
    let mut pos_w = Vec::with_capacity(hw);
    let mut pos_h = Vec::with_capacity(hw);
    for h in 0..h_p {
        for w in 0..w_p {
            pos_w.push(w as f64 / s_scale);
            pos_h.push(h as f64 / s_scale);
        }
    }
    let emb_w = get_1d_sincos_pos_embed(dim_s, &pos_w, device)?; // [H_p * W_p, 720]
    let emb_h = get_1d_sincos_pos_embed(dim_s, &pos_h, device)?; // [H_p * W_p, 720]
    let emb_s = Tensor::cat(&[&emb_w, &emb_h], 1)?;              // [H_p * W_p, 1440]

    // Broadcast across T and S
    let emb_t_exp = emb_t.unsqueeze(1)?.broadcast_as((t, hw, dim_t))?.reshape((t * hw, dim_t))?;
    let emb_s_exp = emb_s.unsqueeze(0)?.broadcast_as((t, hw, dim_s * 2))?.reshape((t * hw, dim_s * 2))?;
    let full_vid = Tensor::cat(&[&emb_t_exp, &emb_s_exp], 1)?; // [T * H_p * W_p, 1920]

    // Prepend text zeros: [1, text_len + T * H_p * W_p, embed_dim]
    let zeros = Tensor::zeros((1, text_len, embed_dim), DType::F32, device)?;
    let full_vid_b = full_vid.unsqueeze(0)?;
    Tensor::cat(&[&zeros, &full_vid_b], 1)
}

/// Spatio-Temporal Video Generation Parameters
#[derive(Debug, Clone)]
pub struct VideoGenerationParams {
    pub prompt: String,
    pub num_frames: usize,
    pub height: usize,
    pub width: usize,
    pub fps: usize,
    pub num_steps: usize,
    pub guidance_scale: f64,
    pub seed: Option<u64>,
}

impl Default for VideoGenerationParams {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            num_frames: 49,
            height: 480,
            width: 720,
            fps: 8,
            num_steps: 30,
            guidance_scale: 6.0,
            seed: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crop_region_and_shapes() -> Result<()> {
        let dev = Device::Cpu;
        let gh = 30;
        let gw = 45;
        let crops = get_resize_crop_region_for_grid((gh, gw), 360, 240);
        assert_eq!(crops, ((0, 0), (240, 360)));

        let (cos, sin) = get_3d_rotary_pos_embed(64, crops, (gh, gw), 13, 10000.0, &dev)?;
        assert_eq!(cos.dims2()?, (13 * 30 * 45, 64));
        assert_eq!(sin.dims2()?, (13 * 30 * 45, 64));

        let q = Tensor::randn(0f32, 1f32, (1, 30, 13 * 30 * 45, 64), &dev)?;
        let q_rot = apply_cogvideox_rope(&q, &cos, &sin)?;
        assert_eq!(q_rot.dims4()?, (1, 30, 13 * 30 * 45, 64));
        Ok(())
    }

    #[test]
    fn test_cogvideox_3d_sincos_pos_embed_shapes() -> Result<()> {
        let dev = Device::Cpu;
        let pos = get_cogvideox_3d_sincos_pos_embed(1920, 30, 45, 13, 226, &dev)?;
        assert_eq!(pos.dims(), &[1, 17776, 1920]);

        // Check text prefix is all zeros
        let text_part = pos.narrow(1, 0, 226)?;
        let max_text = text_part.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        assert_eq!(max_text, 0.0);

        // Check video part has non-zero signal
        let vid_part = pos.narrow(1, 226, 17550)?;
        let max_vid = vid_part.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        assert!(max_vid > 0.99);
        Ok(())
    }
}
