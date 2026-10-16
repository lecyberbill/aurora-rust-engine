// [WFGY] Zone: SAFE | Fallbacks: 0 | Action: Portable, backend-agnostic tensor ops shared by all quantized models.
//!
//! These primitives implement the numerical kernels that Candle's native ops do not
//! expose consistently across backends. They compute in F32 internally and cast back,
//! so they are correct on **CPU, CUDA, ROCm and Metal** alike — no per-backend
//! special-casing (the only backend-gated helper is `ensure_contiguous`, which is a
//! no-op everywhere except ROCm where the fork requires contiguous matmul operands).

use candle_core::quantized::QTensor;
use candle_core::{DType, Device, Result, Tensor};

/// RMSNorm without a learnable offset (used by Qwen/Llama-style models).
pub fn rms_norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    let last_dim = x.dims().len() - 1;
    let x_f32 = x.to_dtype(DType::F32)?;
    let sq = x_f32.sqr()?;
    let mean = sq.mean_keepdim(last_dim)?;
    let rms = (mean + eps)?.sqrt()?;
    let norm = x_f32.broadcast_div(&rms)?;
    norm.to_dtype(x.dtype())
}

/// RMSNorm layer with a dequantized weight, scaling as `norm(x) * weight`.
#[derive(Debug, Clone)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn from_qtensor(weight: QTensor, eps: f64) -> Result<Self> {
        let weight = weight.dequantize(&weight.device())?;
        Ok(Self { weight, eps })
    }

    pub fn from_tensor(weight: Tensor, eps: f64) -> Result<Self> {
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let norm = rms_norm(x, self.eps)?;
        let scaled = norm
            .to_dtype(DType::F32)?
            .broadcast_mul(&self.weight.to_dtype(DType::F32)?)?;
        scaled.to_dtype(x.dtype())
    }
}

/// Repeats key/value heads for grouped-query attention.
/// Input shape `(batch, num_kv_heads, seq_len, head_dim)`.
pub fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(xs)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = xs.dims4()?;
        Tensor::cat(&vec![&xs; n_rep], 2)?.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}

/// RoPE (HF `rotate_half` convention) in F32.
/// `i`: `[B, H, S, D]`; `cos`/`sin`: `[S, D/2]`.
pub fn rope(i: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let dim = i.dim(3)?;
    let half = dim / 2;
    let x_f32 = i.to_dtype(DType::F32)?;
    let cos_full = Tensor::cat(&[cos, cos], 1)?.unsqueeze(0)?.unsqueeze(1)?;
    let sin_full = Tensor::cat(&[sin, sin], 1)?.unsqueeze(0)?.unsqueeze(1)?;
    let x1 = x_f32.narrow(3, 0, half)?;
    let x2 = x_f32.narrow(3, half, half)?;
    let x_rot = Tensor::cat(&[&(x2.neg()?), &x1], 3)?;
    let out = (x_f32.broadcast_mul(&cos_full)? + x_rot.broadcast_mul(&sin_full)?)?;
    out.to_dtype(i.dtype())
}

/// Numerically-stable softmax over the last dimension, computed in F32.
pub fn softmax_last_dim(x: &Tensor) -> Result<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let max = x_f32.max_keepdim(candle_core::D::Minus1)?;
    let shifted = x_f32.broadcast_sub(&max)?;
    let exp = shifted.exp()?;
    let sum = exp.sum_keepdim(candle_core::D::Minus1)?;
    let out = exp.broadcast_div(&sum)?;
    out.to_dtype(x.dtype())
}

/// Additive causal (optionally sliding-window) attention mask.
pub fn build_additive_causal_mask(
    tgt: usize,
    offset: usize,
    sw: Option<usize>,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let minf = f32::NEG_INFINITY;
    let mask: Vec<_> = (0..tgt)
        .flat_map(|i| {
            (0..(tgt + offset)).map(move |j| {
                let past_ok = j <= i + offset;
                let sw_ok = match sw {
                    Some(w) => j + w >= i + offset,
                    None => true,
                };
                if past_ok && sw_ok {
                    0.
                } else {
                    minf
                }
            })
        })
        .collect();
    Tensor::from_slice(&mask, (1, 1, tgt, tgt + offset), device)?.to_dtype(dtype)
}

/// Returns a contiguous view where a backend requires it.
///
/// On ROCm the Candle fork only supports contiguous operands for `matmul`; on every
/// other backend this is a cheap no-op (Candle `Tensor` clones are Arc-based).
pub fn ensure_contiguous(t: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "rocm")]
    {
        if t.device().is_rocm() {
            return t.contiguous();
        }
    }
    Ok(t.clone())
}
