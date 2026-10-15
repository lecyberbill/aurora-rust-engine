// [WFGY] Zone: SAFE | λ: 0.20 | Action: Dedicated Qwen Transformer Block with QK-Norm & Hybrid SSM support
//
// Clean modular design inspired by HuggingFace Transformers & Candle.

use candle_core::{Device, Tensor};
use crate::error::{LuminaError, Result};
use crate::models::common::LinearLayer;
use crate::models::text::causal::{CausalLMConfig, LayerKVCache};

/// Qwen Attention Layer
///
/// Implements:
/// - Distinct Q, K, V projections or fused projections
/// - Per-head RMSNorm on Queries and Keys (QK-Norm)
/// - Rotary Positional Embeddings (RoPE)
/// - Grouped-Query Attention (GQA) & Sliding Window Attention
/// - Output projection
pub struct QwenAttention {
    pub q_proj: LinearLayer,
    pub k_proj: LinearLayer,
    pub v_proj: LinearLayer,
    pub o_proj: LinearLayer,
    pub q_norm: Option<Tensor>,
    pub k_norm: Option<Tensor>,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
}

impl QwenAttention {
    pub fn new(
        q_proj: LinearLayer,
        k_proj: LinearLayer,
        v_proj: LinearLayer,
        o_proj: LinearLayer,
        q_norm: Option<Tensor>,
        k_norm: Option<Tensor>,
        config: &CausalLMConfig,
    ) -> Self {
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            rms_norm_eps: config.rms_norm_eps,
        }
    }

    pub fn forward(
        &self,
        normed_x: &Tensor,
        kv_cache: &mut LayerKVCache,
        pos: usize,
        seq_len: usize,
        sliding_window: Option<usize>,
        rope_fn: impl Fn(&Tensor, &Tensor, usize, usize) -> Result<(Tensor, Tensor)>,
        mask_fn: impl Fn(usize, &Device) -> Result<Tensor>,
    ) -> Result<Tensor> {
        let b_sz = normed_x.dim(0).map_err(LuminaError::Candle)?;

        // 1. Project Q, K, V
        let q = self.q_proj.forward(normed_x)?;
        let k = self.k_proj.forward(normed_x)?;
        let v = self.v_proj.forward(normed_x)?;

        // 2. Reshape to [B, H, S, D]
        let mut q = q.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let mut k = k.reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;

        // 3. Per-head RMSNorm on Q and K before RoPE (Qwen2 / Qwen2.5 / Qwen3)
        if let Some(qn) = &self.q_norm {
            q = apply_rms_norm(&q, qn, self.rms_norm_eps)?;
        }
        if let Some(kn) = &self.k_norm {
            k = apply_rms_norm(&k, kn, self.rms_norm_eps)?;
        }

        // 4. RoPE
        let (q, k) = rope_fn(&q, &k, pos, seq_len)?;

        // 5. KV Cache append with sliding window
        let (k, v) = kv_cache
            .append_with_window(&k, &v, sliding_window)
            .map_err(LuminaError::Candle)?;

        // 6. GQA repetition
        let n_rep = self.num_heads / self.num_kv_heads;
        let (k, v) = if n_rep > 1 {
            let (b, n_kv, s, d) = k.shape().dims4().map_err(LuminaError::Candle)?;
            let k = k.unsqueeze(2)?.expand((b, n_kv, n_rep, s, d))?.reshape((b, n_kv * n_rep, s, d))?;
            let v = v.unsqueeze(2)?.expand((b, n_kv, n_rep, s, d))?.reshape((b, n_kv * n_rep, s, d))?;
            (k, v)
        } else {
            (k, v)
        };

        // 7. Scaled Dot-Product Attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let k_t = k.transpose(2, 3).map_err(LuminaError::Candle)?;
        let mut att = (q.matmul(&k_t)? * scale).map_err(LuminaError::Candle)?;

        if seq_len > 1 {
            let mask = mask_fn(seq_len, att.device())?;
            att = att.broadcast_add(&mask).map_err(LuminaError::Candle)?;
        }

        let att = crate::device::softmax_last_dim(&att).map_err(LuminaError::Candle)?;
        let out = att.matmul(&v).map_err(LuminaError::Candle)?;
        let attn_dim = self.num_heads * self.head_dim;
        let out = out.transpose(1, 2)?.reshape((b_sz, seq_len, attn_dim))?;

        // 8. Output projection
        self.o_proj.forward(&out)
    }
}

/// Helper function to apply RMSNorm along the last dimension
fn apply_rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let dtype = x.dtype();
    let x_f32 = x.to_dtype(candle_core::DType::F32).map_err(LuminaError::Candle)?;
    let norm_factor = (&x_f32 * &x_f32)
        .map_err(LuminaError::Candle)?
        .mean_keepdim(candle_core::D::Minus1)
        .map_err(LuminaError::Candle)?;
    let rms = (norm_factor + eps)
        .map_err(LuminaError::Candle)?
        .sqrt()
        .map_err(LuminaError::Candle)?;
    let x_normed = x_f32.broadcast_div(&rms).map_err(LuminaError::Candle)?;
    let w_f32 = weight.to_dtype(candle_core::DType::F32).map_err(LuminaError::Candle)?;
    let out = x_normed.broadcast_mul(&w_f32).map_err(LuminaError::Candle)?;
    out.to_dtype(dtype).map_err(LuminaError::Candle)
}
