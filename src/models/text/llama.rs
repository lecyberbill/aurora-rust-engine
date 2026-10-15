// [WFGY] Zone: SAFE | λ: 0.20 | Action: Dedicated LLaMA / Mistral Transformer Blocks
//
// Clean modular design inspired by HuggingFace Transformers & Candle.

use candle_core::{Device, Tensor};
use crate::error::{LuminaError, Result};
use crate::models::common::LinearLayer;
use crate::models::text::causal::{CausalLMConfig, LayerKVCache};

/// LLaMA / Mistral Attention Layer
///
/// Implements standard GQA with distinct Q, K, V projections and optional sliding window.
pub struct LlamaAttention {
    pub q_proj: LinearLayer,
    pub k_proj: LinearLayer,
    pub v_proj: LinearLayer,
    pub o_proj: LinearLayer,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl LlamaAttention {
    pub fn new(
        q_proj: LinearLayer,
        k_proj: LinearLayer,
        v_proj: LinearLayer,
        o_proj: LinearLayer,
        config: &CausalLMConfig,
    ) -> Self {
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
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
        let q = q.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;

        // 3. RoPE
        let (q, k) = rope_fn(&q, &k, pos, seq_len)?;

        // 4. KV Cache append with sliding window
        let (k, v) = kv_cache
            .append_with_window(&k, &v, sliding_window)
            .map_err(LuminaError::Candle)?;

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

        // 6. Scaled Dot-Product Attention
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

/// Standard LLaMA / Mistral 3-tensor SwiGLU MLP
///
/// Implements: `down_proj(silu(gate_proj(x)) * up_proj(x))`
pub struct LlamaMlp {
    pub gate_proj: LinearLayer,
    pub up_proj: LinearLayer,
    pub down_proj: LinearLayer,
}

impl LlamaMlp {
    pub fn new(gate_proj: LinearLayer, up_proj: LinearLayer, down_proj: LinearLayer) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    pub fn forward(&self, normed_x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(normed_x)?;
        let up = self.up_proj.forward(normed_x)?;
        let silu_gate = candle_nn::ops::silu(&gate).map_err(LuminaError::Candle)?;
        let mlp_act = (silu_gate * up).map_err(LuminaError::Candle)?;
        self.down_proj.forward(&mlp_act)
    }
}
