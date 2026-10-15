// [WFGY] Zone: SAFE | λ: 0.20 | Action: Dedicated Phi (Phi-3 / Phi-4) Transformer Block with native QMatMul support

use candle_core::{Device, Tensor};
use crate::error::{LuminaError, Result};
use crate::models::common::LinearLayer;
use crate::models::text::causal::{CausalLMConfig, LayerKVCache};

/// Phi (Phi-3 / Phi-3.5 / Phi-4) Attention Layer
///
/// Implements:
/// - Grouped-Query Attention (GQA) with fused `attn_qkv` projection
/// - Rotary Positional Embeddings (RoPE)
/// - Scaled Dot-Product Attention with KV-cache
/// - Output projection `attn_output`
pub struct PhiAttention {
    pub qkv_proj: LinearLayer,
    pub o_proj: LinearLayer,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl PhiAttention {
    pub fn new(qkv_proj: LinearLayer, o_proj: LinearLayer, config: &CausalLMConfig) -> Self {
        Self {
            qkv_proj,
            o_proj,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
        }
    }

    pub fn forward(
        &self,
        normed_x: &Tensor,
        _layer_idx: usize,
        kv_cache: &mut LayerKVCache,
        pos: usize,
        seq_len: usize,
        rope_fn: impl Fn(&Tensor, &Tensor, usize, usize) -> Result<(Tensor, Tensor)>,
        mask_fn: impl Fn(usize, &Device) -> Result<Tensor>,
    ) -> Result<Tensor> {
        let b_sz = normed_x.dim(0).map_err(LuminaError::Candle)?;

        // 1. Fused QKV Projection
        let qkv = self.qkv_proj.forward(normed_x)?;
        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;

        let q = qkv.narrow(candle_core::D::Minus1, 0, q_dim)?;
        let k = qkv.narrow(candle_core::D::Minus1, q_dim, kv_dim)?;
        let v = qkv.narrow(candle_core::D::Minus1, q_dim + kv_dim, kv_dim)?;

        // 2. Reshape to [B, H, S, D]
        let q = q.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;

        // 3. Rotary Position Embedding
        let (q, k) = rope_fn(&q, &k, pos, seq_len)?;

        // 4. KV Cache append
        let (k, v) = kv_cache.append(&k, &v).map_err(LuminaError::Candle)?;

        // 5. GQA repeat KV if needed
        let n_rep = self.num_heads / self.num_kv_heads;
        let (k, v) = if n_rep > 1 {
            let (b, n_kv, s, d) = k.shape().dims4().map_err(LuminaError::Candle)?;
            let k = k.unsqueeze(2)?.expand((b, n_kv, n_rep, s, d))?.reshape((b, n_kv * n_rep, s, d))?;
            let v = v.unsqueeze(2)?.expand((b, n_kv, n_rep, s, d))?.reshape((b, n_kv * n_rep, s, d))?;
            (k, v)
        } else {
            (k, v)
        };

        // 6. Attention dot product
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

        // 7. Output projection
        self.o_proj.forward(&out)
    }
}

/// Phi (Phi-3 / Phi-4) SwiGLU MLP
///
/// Implements:
/// - Fused or split up/gate projection: `ffn_up` [2 * intermediate_size, hidden_size]
/// - SwiGLU non-linearity: `silu(gate) * up`
/// - Down projection: `ffn_down` [hidden_size, intermediate_size]
pub struct PhiMlp {
    pub ffn_up: LinearLayer,
    pub ffn_down: LinearLayer,
    pub intermediate_size: usize,
}

impl PhiMlp {
    pub fn new(ffn_up: LinearLayer, ffn_down: LinearLayer, intermediate_size: usize) -> Self {
        Self {
            ffn_up,
            ffn_down,
            intermediate_size,
        }
    }

    pub fn forward(&self, normed_x: &Tensor) -> Result<Tensor> {
        let up_states = self.ffn_up.forward(normed_x)?;
        let gate = up_states.narrow(candle_core::D::Minus1, 0, self.intermediate_size)?;
        let up = up_states.narrow(candle_core::D::Minus1, self.intermediate_size, self.intermediate_size)?;
        let silu_gate = candle_nn::ops::silu(&gate).map_err(LuminaError::Candle)?;
        let mlp_act = (silu_gate * up).map_err(LuminaError::Candle)?;
        self.ffn_down.forward(&mlp_act)
    }
}
