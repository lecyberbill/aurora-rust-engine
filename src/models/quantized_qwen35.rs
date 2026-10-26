// [WFGY] Zone: TRANSIT | λ: 0.20 | Fallbacks: 0 | Action: Native Qwen 3.5 / Qwen 3.8 Hybrid Architecture (DeltaNet SSM + Full Softmax Attention)
//!
//! Native pure-Rust implementation of Qwen 3.5 / Qwen 3.8 hybrid architecture for Candle.
//! Works across CPU, CUDA, and AMD ROCm using universal ops (`crate::models::ops`).
//!
//! Architecture specifics:
//! - 65 layers total (48 Linear Attention / Gated DeltaNet layers + 17 Full Softmax Attention layers).
//! - Gated DeltaNet recurrence: causal 1D conv, L2 norm on Q & K, gated state update (decay + delta error).
//! - Full Attention: joint Q+Gate projection, QK-RMSNorm, partial RoPE (first 64 dims of 256), gated attention.
//! - SwiGLU MLP: ffn_gate, ffn_up, ffn_down with SiLU activation.

use candle_core::quantized::gguf_file;
use candle_core::quantized::{QMatMul, QTensor};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, Module};
use std::io::{Read, Seek};
use std::sync::Arc;

use crate::models::ops::{repeat_kv, softmax_last_dim, undo_tiled_v_heads_first_dim, undo_tiled_v_heads_last_dim, RmsNorm};

/// Re-quantize a dense (untiled) tensor back to its original GGML dtype so the
/// GDN projections stay compact instead of becoming dense F32/F16 weights.
fn restored_qmatmul(t: &Tensor, dtype: candle_core::quantized::GgmlDType) -> Result<QMatMul> {
    QMatMul::from_qtensor(QTensor::quantize(t, dtype)?)
}

pub const MAX_SEQ_LEN: usize = 32768;

pub struct Gguf<R: Read + Seek> {
    ct: gguf_file::Content,
    reader: R,
    device: Device,
}

impl<R: Read + Seek> Gguf<R> {
    pub fn new(ct: gguf_file::Content, reader: R, device: Device) -> Self {
        Self { ct, reader, device }
    }

    pub fn qmatmul(&mut self, name: &str) -> Result<QMatMul> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        QMatMul::from_qtensor(ws)
    }

    pub fn rms_norm(&mut self, name: &str, eps: f64) -> Result<RmsNorm> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        RmsNorm::from_qtensor(ws, eps)
    }

    pub fn metadata(&self) -> &std::collections::HashMap<String, gguf_file::Value> {
        &self.ct.metadata
    }

    pub fn tensor(&mut self, name: &str) -> Result<QTensor> {
        self.ct.tensor(&mut self.reader, name, &self.device)
    }

    pub fn dequantize_tensor(&mut self, name: &str) -> Result<Tensor> {
        let qt = self.ct.tensor(&mut self.reader, name, &self.device)?;
        qt.dequantize(&self.device)
    }
}

fn sigmoid(x: &Tensor) -> Result<Tensor> {
    candle_nn::ops::sigmoid(x)
}

fn silu(x: &Tensor) -> Result<Tensor> {
    candle_nn::ops::silu(x)
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    let orig_dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    // Numerically stable: max(x, 0) + log(1 + exp(-|x|))
    let zero = Tensor::zeros_like(&x_f32)?;
    let max_x = x_f32.maximum(&zero)?;
    let abs_x = x_f32.abs()?;
    let sp = ((abs_x.neg()?.exp()? + 1.0)?.log()? + max_x)?;
    sp.to_dtype(orig_dtype)
}

fn l2norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    let orig_dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let sum_sq = x_f32.sqr()?.sum_keepdim(candle_core::D::Minus1)?;
    let rsqrt = (sum_sq + eps)?.sqrt()?.recip()?;
    let normed = x_f32.broadcast_mul(&rsqrt)?;
    normed.to_dtype(orig_dtype)
}

#[derive(Debug, Clone)]
struct MlpWeights {
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
}

impl MlpWeights {
    fn new<R: Read + Seek>(gg: &mut Gguf<R>, prefix: &str) -> Result<Self> {
        let gate_proj = gg.qmatmul(&format!("{prefix}.ffn_gate.weight"))?;
        let up_proj = gg.qmatmul(&format!("{prefix}.ffn_up.weight"))?;
        let down_proj = gg.qmatmul(&format!("{prefix}.ffn_down.weight"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let gated = (silu(&gate)? * up)?;
        self.down_proj.forward(&gated)
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
    rotary_dim: usize,
}

impl RotaryEmbedding {
    fn new(
        _head_dim: usize,
        rotary_dim: usize,
        freq_base: f64,
        device: &Device,
    ) -> Result<Self> {
        let half_rot = rotary_dim / 2;
        let mut inv_freq_vec = Vec::with_capacity(half_rot);
        for i in 0..half_rot {
            inv_freq_vec.push(1.0f32 / (freq_base as f32).powf((2 * i) as f32 / rotary_dim as f32));
        }

        let inv_freq = Tensor::from_vec(inv_freq_vec, (1, half_rot), device)?;
        let t = Tensor::arange(0u32, MAX_SEQ_LEN as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((MAX_SEQ_LEN, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;
        Ok(Self { cos, sin, rotary_dim })
    }

    fn apply(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        // x: [B, H, S, head_dim]
        let (_b, _h, s, head_dim) = x.dims4()?;
        if self.rotary_dim == 0 {
            return Ok(x.clone());
        }

        let cos = self.cos.narrow(0, offset, s)?;
        let sin = self.sin.narrow(0, offset, s)?;

        if self.rotary_dim == head_dim {
            // Full RoPE
            crate::models::ops::rope(x, &cos, &sin)
        } else {
            // Partial RoPE (e.g. first 64 dims rotated, remaining 192 passed unrotated)
            let x_rot = x.narrow(3, 0, self.rotary_dim)?;
            let x_pass = x.narrow(3, self.rotary_dim, head_dim - self.rotary_dim)?;
            let x_rotated = crate::models::ops::rope(&x_rot.contiguous()?, &cos, &sin)?;
            Tensor::cat(&[&x_rotated, &x_pass], 3)
        }
    }
}

/// Full Softmax Attention Layer
#[derive(Debug, Clone)]
struct FullAttentionWeights {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: Option<(Tensor, Tensor)>,
    // HIP-graph static decode buffers.
    static_k: Option<Tensor>,   // [1, num_heads, max_len, head_dim]
    static_v: Option<Tensor>,   // [1, num_heads, max_len, head_dim]
    static_idx: Option<Tensor>, // [1, num_heads, 1, head_dim]
    static_mask: Option<Tensor>,// [1, 1, 1, max_len]
    static_rope: Option<(Tensor, Tensor)>, // cos/sin [1, rotary_dim/2]
}

impl FullAttentionWeights {
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary_emb: Arc<RotaryEmbedding>,
        prefix: &str,
    ) -> Result<Self> {
        let hidden_size = num_heads * head_dim;
        let q_proj = gg.qmatmul(&format!("{prefix}.attn_q.weight"))?;
        let k_proj = gg.qmatmul(&format!("{prefix}.attn_k.weight"))?;
        let v_proj = gg.qmatmul(&format!("{prefix}.attn_v.weight"))?;
        let o_proj = gg.qmatmul(&format!("{prefix}.attn_output.weight"))?;

        let q_norm = gg.rms_norm(&format!("{prefix}.attn_q_norm.weight"), rms_norm_eps)?;
        let k_norm = gg.rms_norm(&format!("{prefix}.attn_k_norm.weight"), rms_norm_eps)?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_size,
            rotary_emb,
            kv_cache: None,
            static_k: None,
            static_v: None,
            static_idx: None,
            static_mask: None,
            static_rope: None,
        })
    }

    fn forward(&mut self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;

        // Q projection outputs [B, L, num_heads * head_dim * 2] (query + gate)
        // In HuggingFace: q_proj(x).view(b, l, -1, head_dim * 2).chunk(2, dim=-1)
        // chunk 0: query_states of shape [B, L, num_heads, head_dim]
        // chunk 1: gate of shape [B, L, num_heads, head_dim]
        let qg = self.q_proj.forward(x)?;
        let qg = qg.reshape((b, l, self.num_heads, self.head_dim * 2))?;
        let qg_chunks = qg.chunk(2, 3)?;
        let q = &qg_chunks[0];
        let gate = &qg_chunks[1];

        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let k = k.reshape((b, l, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((b, l, self.num_kv_heads, self.head_dim))?;

        // RMSNorm on Q and K across head_dim
        let q = self.q_norm.forward(&q.contiguous()?)?;
        let k = self.k_norm.forward(&k.contiguous()?)?;

        // Transpose to [B, num_heads, L, head_dim]
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        // Apply RoPE: static buffers for the graph path, shared table otherwise.
        let n_rep = self.num_heads / self.num_kv_heads;
        let (q, k) = if let Some((cos_buf, sin_buf)) = &self.static_rope {
            let rd = self.rotary_emb.rotary_dim;
            if rd == self.head_dim {
                (
                    crate::models::ops::rope(&q.contiguous()?, cos_buf, sin_buf)?,
                    crate::models::ops::rope(&k.contiguous()?, cos_buf, sin_buf)?,
                )
            } else {
                let q_rot = q.narrow(3, 0, rd)?;
                let q_pass = q.narrow(3, rd, self.head_dim - rd)?;
                let k_rot = k.narrow(3, 0, rd)?;
                let k_pass = k.narrow(3, rd, self.head_dim - rd)?;
                let qe = crate::models::ops::rope(&q_rot.contiguous()?, cos_buf, sin_buf)?;
                let ke = crate::models::ops::rope(&k_rot.contiguous()?, cos_buf, sin_buf)?;
                (
                    Tensor::cat(&[&qe, &q_pass], 3)?,
                    Tensor::cat(&[&ke, &k_pass], 3)?,
                )
            }
        } else {
            (
                self.rotary_emb.apply(&q, offset)?,
                self.rotary_emb.apply(&k, offset)?,
            )
        };

        // Update the KV cache: static graph cache (full max_len) or growing cache.
        let (k_cache, v_rep, static_on, flash_ctx) = if let (Some(kt), Some(vv)) = (&self.static_k, &self.static_v) {
            let (b_, _, s_, _) = k.dims4()?;
            let (kr, vr) = if n_rep > 1 {
                let rk = k
                    .unsqueeze(2)?
                    .expand((b_, self.num_kv_heads, n_rep, s_, self.head_dim))?
                    .reshape((b_, self.num_heads, s_, self.head_dim))?;
                let rv = v
                    .unsqueeze(2)?
                    .expand((b_, self.num_kv_heads, n_rep, s_, self.head_dim))?
                    .reshape((b_, self.num_heads, s_, self.head_dim))?;
                (rk, rv)
            } else {
                (k.clone(), v.clone())
            };
            let idx = self.static_idx.as_ref().unwrap();
            kt.scatter_set(idx, &kr, 2)?;
            vv.scatter_set(idx, &vr, 2)?;
            (kt.clone(), vv.clone(), true, None)
        } else {
            let (k_all, v_all) = match &self.kv_cache {
                Some((prev_k, prev_v)) => (
                    Tensor::cat(&[prev_k, &k], 2)?,
                    Tensor::cat(&[prev_v, &v], 2)?,
                ),
                None => (k, v),
            };
            self.kv_cache = Some((k_all.clone(), v_all.clone()));
            let scale = 1.0 / (self.head_dim as f64).sqrt();
            let flash = if l == 1 {
                crate::device::flash_decode(&q, &k_all, &v_all, scale)?
            } else {
                None
            };
            if flash.is_some() {
                (k_all, v_all, false, flash)
            } else {
                (repeat_kv(k_all, n_rep)?, repeat_kv(v_all, n_rep)?, false, None)
            }
        };

        let ctx = if let Some(f) = flash_ctx {
            f
        } else {
            let scale = 1.0 / (self.head_dim as f64).sqrt();
            let q_cont = q.contiguous()?;
            let k_t_cont = k_cache.transpose(2, 3)?.contiguous()?;
            let scores = (q_cont.matmul(&k_t_cont)? * scale)?;

            // Mask: static row for the graph path, materialised causal mask for prefill.
            let (_b, _h, q_len, kv_len) = scores.dims4()?;
            let scores = if static_on {
                if let Some(m) = &self.static_mask {
                    scores.broadcast_add(m)?
                } else {
                    scores
                }
            } else if q_len > 1 {
                let mask: Vec<f32> = (0..q_len)
                    .flat_map(|i| {
                        (0..kv_len).map(move |j| {
                            if j > i + offset {
                                f32::NEG_INFINITY
                            } else {
                                0f32
                            }
                        })
                    })
                    .collect();
                let mask = Tensor::from_slice(&mask, (1, 1, q_len, kv_len), x.device())?;
                scores.broadcast_add(&mask.to_dtype(scores.dtype())?)?
            } else {
                scores
            };

            let probs = softmax_last_dim(&scores)?;
            let probs_cont = probs.contiguous()?;
            let v_cont = v_rep.contiguous()?;
            probs_cont.matmul(&v_cont)?
        };

        // Reshape ctx to [B, L, num_heads, head_dim]
        let ctx = ctx.transpose(1, 2)?.contiguous()?;

        // Gate modulation: ctx * sigmoid(gate)
        let gate_sig = sigmoid(&gate)?;
        let gated_ctx = (ctx * gate_sig)?;

        let flat_ctx = gated_ctx.reshape((b, l, self.hidden_size))?;
        self.o_proj.forward(&flat_ctx)
    }

    fn reset_cache(&mut self) {
        self.kv_cache = None;
    }
}

/// Gated DeltaNet (SSM / Linear Attention) Layer
#[derive(Debug, Clone)]
struct GatedDeltaNetWeights {
    attn_qkv: QMatMul,
    attn_gate: QMatMul,
    ssm_alpha: QMatMul,
    ssm_beta: QMatMul,
    ssm_out: QMatMul,

    ssm_conv1d_weight: Tensor, // [10240, 4] F32
    ssm_a: Tensor,             // [48] F32
    ssm_dt_bias: Tensor,       // [48] F32
    ssm_norm: RmsNorm,         // head_v_dim = 128

    key_dim: usize,     // 16 * 128 = 2048
    value_dim: usize,   // 48 * 128 = 6144
    _conv_dim: usize,   // 2048 * 2 + 6144 = 10240
    num_k_heads: usize, // 16
    num_v_heads: usize, // 48
    head_k_dim: usize,  // 128
    head_v_dim: usize,  // 128

    // Recurrent state cache
    conv_state: Option<Tensor>, // [10240, 3] F32
    recurrent_state: Option<Tensor>, // [48, 128, 128] F32
    // HIP-graph static state (updated in place via copy2).
    static_conv: Option<Tensor>,
    static_state: Option<Tensor>,
    /// Ones weight for the fused L2-norm (RMSNorm with a ones weight).
    l2_ones: Tensor,
}

impl GatedDeltaNetWeights {
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        rms_norm_eps: f64,
        prefix: &str,
    ) -> Result<Self> {
        // Qwen3.5 GGUF stores GDN v-heads tiled as [num_v_per_k, num_k_heads, ...].
        // Untile back to [num_v_heads, ...] before use (candle-vllm reference).
        const NUM_K_HEADS: usize = 16;
        const NUM_V_HEADS: usize = 48;
        const HEAD_V_DIM: usize = 128;
        const KEY_DIM: usize = 2048;
        const VALUE_DIM: usize = 6144;

        let attn_qkv = {
            let qt = gg.tensor(&format!("{prefix}.attn_qkv.weight"))?;
            let dtype = qt.dtype();
            let dev = qt.device().clone();
            let w = qt.dequantize(&dev)?;
            let q = w.narrow(0, 0, KEY_DIM)?;
            let k = w.narrow(0, KEY_DIM, KEY_DIM)?;
            let v_raw = w.narrow(0, KEY_DIM * 2, VALUE_DIM)?;
            let v = undo_tiled_v_heads_first_dim(&v_raw, NUM_K_HEADS, NUM_V_HEADS, HEAD_V_DIM)?;
            let restored = Tensor::cat(&[&q, &k, &v], 0)?.contiguous()?;
            restored_qmatmul(&restored, dtype)?
        };
        let attn_gate = {
            let qt = gg.tensor(&format!("{prefix}.attn_gate.weight"))?;
            let dtype = qt.dtype();
            let dev = qt.device().clone();
            let w = qt.dequantize(&dev)?;
            let w = undo_tiled_v_heads_first_dim(&w, NUM_K_HEADS, NUM_V_HEADS, HEAD_V_DIM)?;
            restored_qmatmul(&w, dtype)?
        };
        let ssm_alpha = {
            let qt = gg.tensor(&format!("{prefix}.ssm_alpha.weight"))?;
            let dtype = qt.dtype();
            let dev = qt.device().clone();
            let w = qt.dequantize(&dev)?;
            let w = undo_tiled_v_heads_first_dim(&w, NUM_K_HEADS, NUM_V_HEADS, 1)?;
            restored_qmatmul(&w, dtype)?
        };
        let ssm_beta = {
            let qt = gg.tensor(&format!("{prefix}.ssm_beta.weight"))?;
            let dtype = qt.dtype();
            let dev = qt.device().clone();
            let w = qt.dequantize(&dev)?;
            let w = undo_tiled_v_heads_first_dim(&w, NUM_K_HEADS, NUM_V_HEADS, 1)?;
            restored_qmatmul(&w, dtype)?
        };
        let ssm_out = {
            let qt = gg.tensor(&format!("{prefix}.ssm_out.weight"))?;
            let dtype = qt.dtype();
            let dev = qt.device().clone();
            let w = qt.dequantize(&dev)?;
            let w = undo_tiled_v_heads_last_dim(&w, NUM_K_HEADS, NUM_V_HEADS, HEAD_V_DIM)?;
            restored_qmatmul(&w, dtype)?
        };

        let ssm_conv1d_weight = {
            let w = gg.dequantize_tensor(&format!("{prefix}.ssm_conv1d.weight"))?.to_dtype(DType::F32)?;
            // conv weight is [conv_dim, kernel] with the v block tiled along dim 0
            let q_w = w.narrow(0, 0, KEY_DIM)?;
            let k_w = w.narrow(0, KEY_DIM, KEY_DIM)?;
            let v_w = w.narrow(0, KEY_DIM * 2, VALUE_DIM)?;
            let v_w = undo_tiled_v_heads_first_dim(&v_w, NUM_K_HEADS, NUM_V_HEADS, HEAD_V_DIM)?;
            Tensor::cat(&[&q_w, &k_w, &v_w], 0)?
        };
        let ssm_a = {
            let a = gg.dequantize_tensor(&format!("{prefix}.ssm_a"))?.to_dtype(DType::F32)?;
            undo_tiled_v_heads_first_dim(&a, NUM_K_HEADS, NUM_V_HEADS, 1)?
        };
        let ssm_dt_bias = {
            let d = gg.dequantize_tensor(&format!("{prefix}.ssm_dt.bias"))?.to_dtype(DType::F32)?;
            undo_tiled_v_heads_first_dim(&d, NUM_K_HEADS, NUM_V_HEADS, 1)?
        };
        let ssm_norm = gg.rms_norm(&format!("{prefix}.ssm_norm.weight"), rms_norm_eps)?;
        let l2_ones = Tensor::ones((HEAD_V_DIM,), DType::F32, &gg.device)?;

        Ok(Self {
            attn_qkv,
            attn_gate,
            ssm_alpha,
            ssm_beta,
            ssm_out,
            ssm_conv1d_weight,
            ssm_a,
            ssm_dt_bias,
            ssm_norm,
            key_dim: 2048,
            value_dim: 6144,
            _conv_dim: 10240,
            num_k_heads: 16,
            num_v_heads: 48,
            head_k_dim: 128,
            head_v_dim: 128,
            conv_state: None,
            recurrent_state: None,
            static_conv: None,
            static_state: None,
            l2_ones,
        })
    }

    fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let (b, seq_len, _hidden) = x.dims3()?;
        let orig_dtype = x.dtype();

        // 1. Projections
        let qkv_mixed = self.attn_qkv.forward(x)?; // [B, L, 10240]
        let z = self.attn_gate.forward(x)?;       // [B, L, 6144]
        let alpha = self.ssm_alpha.forward(x)?;   // [B, L, 48]
        let beta = self.ssm_beta.forward(x)?;     // [B, L, 48]

        let beta_sig = sigmoid(&beta)?.to_dtype(DType::F32)?; // [B, L, 48]
        let alpha_f32 = alpha.to_dtype(DType::F32)?;

        // GGUF stores ssm_a = -exp(A_log) already; g = ssm_a * softplus(alpha + dt_bias)
        let alpha_biased = alpha_f32.broadcast_add(&self.ssm_dt_bias.reshape((1, 1, self.num_v_heads))?)?;
        let sp = softplus(&alpha_biased)?;
        let ssm_a = self.ssm_a.reshape((1, 1, self.num_v_heads))?;
        let decay = (sp.broadcast_mul(&ssm_a)?).exp()?; // [B, L, 48]

        // 2. 1D Causal Convolution over channels
        // qkv_mixed: [B, L, 10240] -> work in F32
        let qkv_f32 = qkv_mixed.to_dtype(DType::F32)?;
        let convolved = self.run_causal_conv(&qkv_f32)?; // [B, L, 10240]
        let convolved_act = silu(&convolved)?;

        // 3. Split Q, K, V
        let q_raw = convolved_act.narrow(2, 0, self.key_dim)?;
        let k_raw = convolved_act.narrow(2, self.key_dim, self.key_dim)?;
        let v_raw = convolved_act.narrow(2, self.key_dim * 2, self.value_dim)?;

        let q_heads = q_raw.reshape((b, seq_len, self.num_k_heads, self.head_k_dim))?;
        let k_heads = k_raw.reshape((b, seq_len, self.num_k_heads, self.head_k_dim))?;
        let v_heads = v_raw.reshape((b, seq_len, self.num_v_heads, self.head_v_dim))?;

        // 4. L2 Norm on Q and K (fused RMSNorm with a ones weight)
        let q_norm = self.l2_norm(&q_heads, 1e-6)?;
        let k_norm = self.l2_norm(&k_heads, 1e-6)?;

        // Q scaling: q / sqrt(128)
        let q_scaled = (q_norm * (1.0f64 / (self.head_k_dim as f64).sqrt()))?;

        // Repeat Q and K 3x (from 16 to 48 heads)
        // Repeat interleave along head dimension (dim 2)
        let repeat_factor = self.num_v_heads / self.num_k_heads; // 3
        let q_expanded = q_scaled.unsqueeze(3)?
            .broadcast_as((b, seq_len, self.num_k_heads, repeat_factor, self.head_k_dim))?
            .contiguous()?
            .reshape((b, seq_len, self.num_v_heads, self.head_k_dim))?;
        let k_expanded = k_norm.unsqueeze(3)?
            .broadcast_as((b, seq_len, self.num_k_heads, repeat_factor, self.head_k_dim))?
            .contiguous()?
            .reshape((b, seq_len, self.num_v_heads, self.head_k_dim))?;

        // 5. Recurrent State Update (per token)
        let out_v = self.run_delta_recurrence(
            &q_expanded,
            &k_expanded,
            &v_heads,
            &decay,
            &beta_sig,
        )?; // [B, L, 48, 128]

        // 6. Gated RMSNorm: RMSNorm(out_v) * SiLU(z)
        // Reshape out_v to [B, L, 48, 128] -> apply ssm_norm across last dim 128
        let out_v_norm = self.ssm_norm.forward(&out_v.contiguous()?)?;
        let out_v_flat = out_v_norm.reshape((b, seq_len, self.value_dim))?;
        let z_act = silu(&z)?;
        let gated_out = (out_v_flat * z_act)?;

        // 7. Output projection back to hidden_size 5120
        self.ssm_out.forward(&gated_out.to_dtype(orig_dtype)?)
    }

    fn run_causal_conv(&mut self, qkv: &Tensor) -> Result<Tensor> {
        let (b, seq_len, channels) = qkv.dims3()?;
        // Kernel weights: [10240, 4]
        // Conv is depthwise causal:
        // For each time step t, out[t] = sum_{k=0..3} x[t - 3 + k] * weight[channel, k]
        let prev = match &self.static_conv {
            Some(s) => Some(s.clone()),
            None => self.conv_state.clone(),
        };
        let (full_x, new_conv_state) = match prev {
            Some(prev) => {
                // prev: [channels, 3] -> transpose to [1, 3, channels]
                let prev_t = prev.transpose(0, 1)?.unsqueeze(0)?;
                let full = Tensor::cat(&[&prev_t, qkv], 1)?;
                let next_state = full.narrow(1, full.dim(1)? - 3, 3)?.squeeze(0)?.transpose(0, 1)?;
                (full, next_state)
            }
            None => {
                let zeros = Tensor::zeros((b, 3, channels), DType::F32, qkv.device())?;
                let full = Tensor::cat(&[&zeros, qkv], 1)?;
                let next_state = full.narrow(1, full.dim(1)? - 3, 3)?.squeeze(0)?.transpose(0, 1)?;
                (full, next_state)
            }
        };
        #[cfg(feature = "hip-graph")]
        if let Some(s) = &self.static_conv {
            s.copy2(&new_conv_state)?;
        } else {
            self.conv_state = Some(new_conv_state);
        }
        #[cfg(not(feature = "hip-graph"))]
        {
            self.conv_state = Some(new_conv_state);
        }

        // Compute convolution output for steps 3..(3 + seq_len)
        // With padded buffer full_x = [pad0, pad1, pad2, x0, x1, ...]:
        // At step t (corresponding to token x[t] at index t + 3):
        // Receptive field window is full_x[t .. t + 4] = [full_x[t], full_x[t+1], full_x[t+2], full_x[t+3]]
        // Under standard PyTorch F.conv1d(padding=3)[:, :, :seq_len]:
        // out[t] = w0 * full_x[t] + w1 * full_x[t+1] + w2 * full_x[t+2] + w3 * full_x[t+3]
        // where full_x[t+3] is current token x[t], full_x[t+2] is x[t-1], etc.
        let w0 = self.ssm_conv1d_weight.narrow(1, 0, 1)?.reshape((1, 1, channels))?;
        let w1 = self.ssm_conv1d_weight.narrow(1, 1, 1)?.reshape((1, 1, channels))?;
        let w2 = self.ssm_conv1d_weight.narrow(1, 2, 1)?.reshape((1, 1, channels))?;
        let w3 = self.ssm_conv1d_weight.narrow(1, 3, 1)?.reshape((1, 1, channels))?;

        let x0 = full_x.narrow(1, 0, seq_len)?; // t
        let x1 = full_x.narrow(1, 1, seq_len)?; // t + 1
        let x2 = full_x.narrow(1, 2, seq_len)?; // t + 2
        let x3 = full_x.narrow(1, 3, seq_len)?; // t + 3 (current token)

        let conv_out = (
            (x0.broadcast_mul(&w0)? + x1.broadcast_mul(&w1)?)? +
            (x2.broadcast_mul(&w2)? + x3.broadcast_mul(&w3)?)?
        )?;
        Ok(conv_out)
    }

    fn run_delta_recurrence(
        &mut self,
        q: &Tensor,     // [B, L, 48, 128]
        k: &Tensor,     // [B, L, 48, 128]
        v: &Tensor,     // [B, L, 48, 128]
        decay: &Tensor, // [B, L, 48]
        beta: &Tensor,  // [B, L, 48]
    ) -> Result<Tensor> {
        let (_b, seq_len, _num_heads, _dim) = q.dims4()?;
        let device = q.device();

        let mut current_state = match &self.static_state {
            Some(s) => s.clone(),
            None => match &self.recurrent_state {
                Some(s) => s.clone(),
                None => Tensor::zeros((self.num_v_heads, self.head_k_dim, self.head_v_dim), DType::F32, device)?,
            },
        };

        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let q_t = q.narrow(1, t, 1)?.squeeze(1)?.squeeze(0)?; // [48, 128]
            let k_t = k.narrow(1, t, 1)?.squeeze(1)?.squeeze(0)?; // [48, 128]
            let v_t = v.narrow(1, t, 1)?.squeeze(1)?.squeeze(0)?; // [48, 128]
            let g_t = decay.narrow(1, t, 1)?.squeeze(1)?.squeeze(0)?.reshape((self.num_v_heads, 1, 1))?; // [48, 1, 1]
            let b_t = beta.narrow(1, t, 1)?.squeeze(1)?.squeeze(0)?.reshape((self.num_v_heads, 1))?;     // [48, 1]

            // kv_mem = S_{t-1}^T * k_t  (standard delta rule uses the PRE-decay state)
            let k_t_row = k_t.unsqueeze(1)?.contiguous()?; // [48, 1, 128]
            let cs_cont = current_state.contiguous()?;
            let kv_mem = k_t_row.matmul(&cs_cont)?.squeeze(1)?; // [48, 128]

            // delta = (v_t - kv_mem) * b_t
            let delta = ((v_t - kv_mem)?.broadcast_mul(&b_t))?; // [48, 128]

            // S_t = g_t * S_{t-1} + k_t * delta^T
            let k_t_col = k_t.unsqueeze(2)?.contiguous()?; // [48, 128, 1]
            let delta_row = delta.unsqueeze(1)?.contiguous()?; // [48, 1, 128]
            let outer = k_t_col.matmul(&delta_row)?; // [48, 128, 128]
            let new_state = (cs_cont.broadcast_mul(&g_t)? + outer)?;
            #[cfg(feature = "hip-graph")]
            if self.static_state.is_some() {
                current_state.copy2(&new_state)?;
            } else {
                current_state = new_state;
            }
            #[cfg(not(feature = "hip-graph"))]
            {
                current_state = new_state;
            }

            // Readout: out_t = S^T * q_t = q_t.matmul(current_state)
            let q_t_row = q_t.unsqueeze(1)?.contiguous()?; // [48, 1, 128]
            let cs_final = current_state.contiguous()?;
            let out_t = q_t_row.matmul(&cs_final)?.squeeze(1)?; // [48, 128]
            outputs.push(out_t);
        }

        if self.static_state.is_none() {
            self.recurrent_state = Some(current_state);
        }

        // Stack outputs along sequence dimension -> [B, L, 48, 128]
        let out_stacked = Tensor::stack(&outputs, 0)?; // [L, 48, 128]
        out_stacked.unsqueeze(0) // [1, L, 48, 128]
    }

    /// L2 normalization on the last dim, computed by the fused RMSNorm kernel:
    /// `rms_norm(x, 1, eps/d) / sqrt(d) == x / sqrt(sum(x^2) + eps)`.
    fn l2_norm(&self, x: &Tensor, eps: f64) -> Result<Tensor> {
        let d = x.dim(candle_core::D::Minus1)?;
        let n = candle_nn::ops::rms_norm(x, &self.l2_ones, (eps / d as f64) as f32)?;
        n * (1.0 / (d as f64).sqrt())
    }

    fn reset_cache(&mut self) {
        self.conv_state = None;
        self.recurrent_state = None;
    }
}

#[derive(Debug, Clone)]
enum LayerKind {
    LinearAttn(GatedDeltaNetWeights),
    FullAttn(FullAttentionWeights),
}

#[derive(Debug, Clone)]
struct LayerWeights {
    attn: LayerKind,
    mlp: MlpWeights,
    attn_norm: RmsNorm,
    post_attention_norm: RmsNorm,
}

impl LayerWeights {
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        rms_norm_eps: f64,
        rotary: Arc<RotaryEmbedding>,
        layer_idx: usize,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");
        let attn_norm = gg.rms_norm(&format!("{prefix}.attn_norm.weight"), rms_norm_eps)?;
        let post_attention_norm = gg.rms_norm(&format!("{prefix}.post_attention_norm.weight"), rms_norm_eps)?;

        let is_full_attn = (layer_idx + 1) % 4 == 0;
        let attn = if is_full_attn {
            let full = FullAttentionWeights::new(
                gg,
                24,  // num_heads
                4,   // num_kv_heads
                256, // head_dim
                rms_norm_eps,
                rotary,
                &prefix,
            )?;
            LayerKind::FullAttn(full)
        } else {
            let delta = GatedDeltaNetWeights::new(gg, rms_norm_eps, &prefix)?;
            LayerKind::LinearAttn(delta)
        };

        let mlp = MlpWeights::new(gg, &prefix)?;

        Ok(Self {
            attn,
            mlp,
            attn_norm,
            post_attention_norm,
        })
    }

    fn forward(&mut self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let norm_x = self.attn_norm.forward(x)?;
        let attn_out = match &mut self.attn {
            LayerKind::LinearAttn(delta) => delta.forward(&norm_x)?,
            LayerKind::FullAttn(full) => full.forward(&norm_x, offset)?,
        };
        let residual = (x + attn_out)?;

        let post_norm = self.post_attention_norm.forward(&residual)?;
        let mlp_out = self.mlp.forward(&post_norm)?;
        residual + mlp_out
    }

    fn reset_cache(&mut self) {
        match &mut self.attn {
            LayerKind::LinearAttn(delta) => delta.reset_cache(),
            LayerKind::FullAttn(full) => full.reset_cache(),
        }
    }

    fn set_static_full(&mut self, k: Tensor, v: Tensor, idx: Tensor, mask: Tensor, rope: (Tensor, Tensor)) {
        if let LayerKind::FullAttn(full) = &mut self.attn {
            full.static_k = Some(k);
            full.static_v = Some(v);
            full.static_idx = Some(idx);
            full.static_mask = Some(mask);
            full.static_rope = Some(rope);
        }
    }

    fn set_static_delta(&mut self, conv: Tensor, state: Tensor) {
        if let LayerKind::LinearAttn(d) = &mut self.attn {
            d.static_conv = Some(conv);
            d.static_state = Some(state);
        }
    }

    fn clear_static(&mut self) {
        match &mut self.attn {
            LayerKind::LinearAttn(d) => {
                d.static_conv = None;
                d.static_state = None;
            }
            LayerKind::FullAttn(f) => {
                f.static_k = None;
                f.static_v = None;
                f.static_idx = None;
                f.static_mask = None;
                f.static_rope = None;
            }
        }
    }
}

#[derive(Debug)]
pub struct ModelWeights {
    embed_tokens: Embedding,
    layers: Vec<LayerWeights>,
    output_norm: RmsNorm,
    lm_head: QMatMul,
    _device: Device,
    _dtype: DType,
    vocab_size: usize,
    eos_token_id: Option<u32>,
    rotary: Arc<RotaryEmbedding>,
    // HIP-graph decode state.
    static_h0: Option<Tensor>,
    static_idx: Option<Tensor>,
    static_mask_row: Option<Tensor>,
    mask_src: Option<Tensor>,
    static_rope: Option<(Tensor, Tensor)>,
    graph_exec: Option<crate::models::text::causal::HipGraphExec>,
    graph_logits: Option<Tensor>,
    graph_max_len: usize,
}

impl ModelWeights {
    pub fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let mut gg = Gguf::new(ct, reader, device.clone());
        let eos_token_id = gg
            .metadata()
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.to_u32().ok());

        let num_layers = 64; // 64 main layers; block 64 is the MTP draft head (not part of the forward)
        let hidden_size = 5120;
        let head_dim = 256;
        let rotary_dim = 64; // partial RoPE: 64 of 256
        let rms_norm_eps = 1e-6;
        let rope_freq_base = 10000000.0;

        let dtype = DType::F32;

        let embed_tensor = gg.tensor("token_embd.weight")?;
        let vocab_size = embed_tensor.shape().dims()[0];
        let embed_tokens = Embedding::new(embed_tensor.dequantize(device)?, hidden_size);

        let rotary = Arc::new(RotaryEmbedding::new(
            head_dim,
            rotary_dim,
            rope_freq_base,
            device,
        )?);

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            if i % 10 == 0 || i == num_layers - 1 {
                eprintln!("[Qwen 3.5] Loading layer {i}/{num_layers}...");
            }
            layers.push(LayerWeights::new(&mut gg, rms_norm_eps, rotary.clone(), i)?);
        }

        let output_norm = gg.rms_norm("output_norm.weight", rms_norm_eps)?;
        let lm_head_tensor = gg.tensor("output.weight")?;
        let lm_head = QMatMul::from_qtensor(lm_head_tensor)?;

        Ok(Self {
            embed_tokens,
            layers,
            output_norm,
            lm_head,
            _device: device.clone(),
            _dtype: dtype,
            vocab_size,
            eos_token_id,
            rotary,
            static_h0: None,
            static_idx: None,
            static_mask_row: None,
            mask_src: None,
            static_rope: None,
            graph_exec: None,
            graph_logits: None,
            graph_max_len: 0,
        })
    }

    /// Number of rows in the token embedding table (may include unused PAD rows).
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// EOS token id declared by the GGUF tokenizer metadata, when present.
    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    /// Whether the static HIP-graph decode path can serve this model.
    pub fn supports_static_graph(&self) -> bool {
        !self.layers.is_empty()
    }

    #[cfg(feature = "hip-graph")]
    pub fn setup_static_graph(&mut self, prefill_len: usize, max_len: usize) -> Result<()> {
        use crate::models::text::causal::HipGraphExec;
        use candle_core::rocm_backend::rocm_rs::hip::bindings;

        if !self.supports_static_graph() {
            return Err(candle_core::Error::Msg("HIP graph: unsupported model".into()));
        }
        if prefill_len + 16 >= max_len {
            return Err(candle_core::Error::Msg("HIP graph: context does not fit".into()));
        }
        let dev = self._device.clone();
        if !dev.is_rocm() {
            return Err(candle_core::Error::Msg("HIP graph: not a ROCm device".into()));
        }
        let dtype = DType::F32;
        let (num_heads, num_kv_heads, head_dim, rotary_dim) = self
            .layers
            .iter()
            .find_map(|l| match &l.attn {
                LayerKind::FullAttn(f) => Some((f.num_heads, f.num_kv_heads, f.head_dim, f.rotary_emb.rotary_dim)),
                _ => None,
            })
            .ok_or_else(|| candle_core::Error::Msg("HIP graph: no attention layer".into()))?;
        let hidden = self.embed_tokens.embeddings().dim(1)?;
        let half_rot = rotary_dim / 2;

        // Static causal mask + decode row.
        let mut mv = vec![0f32; max_len * max_len];
        for i in 0..max_len {
            for j in (i + 1)..max_len {
                mv[i * max_len + j] = f32::NEG_INFINITY;
            }
        }
        let mask_src = Tensor::from_vec(mv, (max_len, max_len), &dev)?;
        let mask_row = Tensor::zeros((1, 1, 1, max_len), dtype, &dev)?;
        let row0 = mask_src.narrow(0, prefill_len, 1)?.reshape((1, 1, 1, max_len))?;
        mask_row.copy2(&row0)?;

        // Shared scatter index + RoPE buffers, seeded at the prefill position.
        let static_idx = Tensor::zeros((1, num_heads, 1, head_dim), DType::U32, &dev)?;
        static_idx.copy_from_host_u32(&vec![prefill_len as u32; num_heads * head_dim])?;
        let cos_buf = Tensor::zeros((1, half_rot), dtype, &dev)?;
        let sin_buf = Tensor::zeros((1, half_rot), dtype, &dev)?;
        cos_buf.copy2(&self.rotary.cos.narrow(0, prefill_len, 1)?)?;
        sin_buf.copy2(&self.rotary.sin.narrow(0, prefill_len, 1)?)?;

        let n_rep = num_heads / num_kv_heads;
        for layer in self.layers.iter_mut() {
            match &mut layer.attn {
                LayerKind::FullAttn(f) => {
                    let (pk, pv) = f
                        .kv_cache
                        .clone()
                        .ok_or_else(|| candle_core::Error::Msg("HIP graph: empty prefill KV".into()))?;
                    let (b, nkv, s, d) = pk.shape().dims4()?;
                    let (rk, rv) = if n_rep > 1 {
                        let rk = pk
                            .unsqueeze(2)?
                            .expand((b, nkv, n_rep, s, d))?
                            .reshape((b, num_heads, s, d))?;
                        let rv = pv
                            .unsqueeze(2)?
                            .expand((b, nkv, n_rep, s, d))?
                            .reshape((b, num_heads, s, d))?;
                        (rk, rv)
                    } else {
                        (pk.clone(), pv.clone())
                    };
                    let sk = Tensor::zeros((1, num_heads, max_len, head_dim), dtype, &dev)?;
                    let sv = Tensor::zeros((1, num_heads, max_len, head_dim), dtype, &dev)?;
                    let mut idxv = Vec::with_capacity(num_heads * s * head_dim);
                    for _ in 0..num_heads {
                        for si in 0..s {
                            for _ in 0..head_dim {
                                idxv.push(si as u32);
                            }
                        }
                    }
                    let idx = Tensor::from_vec(idxv, (1, num_heads, s, head_dim), &dev)?;
                    sk.scatter_set(&idx, &rk, 2)?;
                    sv.scatter_set(&idx, &rv, 2)?;
                    f.static_k = Some(sk);
                    f.static_v = Some(sv);
                    f.static_idx = Some(static_idx.clone());
                    f.static_mask = Some(mask_row.clone());
                    f.static_rope = Some((cos_buf.clone(), sin_buf.clone()));
                }
                LayerKind::LinearAttn(g) => {
                    let conv = g
                        .conv_state
                        .as_ref()
                        .ok_or_else(|| candle_core::Error::Msg("HIP graph: empty conv state".into()))?
                        .copy()?
                        .contiguous()?;
                    let state = g
                        .recurrent_state
                        .as_ref()
                        .ok_or_else(|| candle_core::Error::Msg("HIP graph: empty recurrent state".into()))?
                        .copy()?
                        .contiguous()?;
                    g.static_conv = Some(conv);
                    g.static_state = Some(state);
                }
            }
        }

        self.static_h0 = Some(Tensor::zeros((1, 1, hidden), dtype, &dev)?);
        self.static_idx = Some(static_idx);
        self.static_mask_row = Some(mask_row);
        self.mask_src = Some(mask_src);
        self.static_rope = Some((cos_buf, sin_buf));
        self.graph_max_len = max_len;
        self.graph_exec = None;
        self.graph_logits = None;

        // Warm up (this advances the recurrent/conv states), then capture.
        let dummy = Tensor::zeros((1, 1), DType::U32, &dev)?;
        for _ in 0..3 {
            let _ = self.forward(&dummy, prefill_len)?;
        }
        dev.synchronize()?;

        let stream = match &dev {
            Device::Rocm(d) => d.stream().as_raw(),
            _ => unreachable!(),
        };
        let r = unsafe {
            bindings::hipStreamBeginCapture(
                stream,
                bindings::hipStreamCaptureMode_hipStreamCaptureModeThreadLocal,
            )
        };
        if r != bindings::hipError_t_hipSuccess {
            return Err(candle_core::Error::Msg(format!("hipStreamBeginCapture: {r}")));
        }
        let logits_res = self.forward(&dummy, prefill_len);
        let mut graph = std::ptr::null_mut();
        let r = unsafe { bindings::hipStreamEndCapture(stream, &mut graph) };
        if logits_res.is_err() {
            if !graph.is_null() {
                unsafe { bindings::hipGraphDestroy(graph); }
            }
            return logits_res.map(|_| ());
        }
        let logits = logits_res?;
        if r != bindings::hipError_t_hipSuccess {
            return Err(candle_core::Error::Msg(format!("hipStreamEndCapture: {r}")));
        }
        dev.synchronize()?;
        let mut exec = std::ptr::null_mut();
        let r = unsafe {
            bindings::hipGraphInstantiate(&mut exec, graph, std::ptr::null_mut(), std::ptr::null_mut(), 0)
        };
        if r != bindings::hipError_t_hipSuccess {
            return Err(candle_core::Error::Msg(format!("hipGraphInstantiate: {r}")));
        }
        unsafe { bindings::hipGraphDestroy(graph); }

        // Warm-up/capture advanced the recurrent and conv states; restore the
        // prefill state so the first replay starts from the right point.
        for layer in self.layers.iter() {
            if let LayerKind::LinearAttn(g) = &layer.attn {
                if let (Some(sc), Some(c0)) = (&g.static_conv, &g.conv_state) {
                    sc.copy2(c0)?;
                }
                if let (Some(ss), Some(s0)) = (&g.static_state, &g.recurrent_state) {
                    ss.copy2(s0)?;
                }
            }
        }

        self.graph_logits = Some(logits);
        self.graph_exec = Some(HipGraphExec {
            exec: exec as usize,
            stream: stream as usize,
        });
        Ok(())
    }

    #[cfg(not(feature = "hip-graph"))]
    pub fn setup_static_graph(&mut self, _prefill_len: usize, _max_len: usize) -> Result<()> {
        Err(candle_core::Error::Msg("HIP graph requires the hip-graph feature".into()))
    }

    #[cfg(feature = "hip-graph")]
    fn graph_step(&mut self, input: &Tensor, pos: usize) -> Result<Tensor> {
        let flat = input.flatten_all()?;
        let tok = match flat.dtype() {
            DType::U32 => flat.to_vec1::<u32>()?,
            DType::I64 => flat.to_vec1::<i64>()?.into_iter().map(|v| v as u32).collect(),
            other => return Err(candle_core::Error::Msg(format!("graph step: token dtype {other:?}"))),
        };
        let tok = *tok.first().ok_or_else(|| candle_core::Error::Msg("graph step: empty".into()))?;
        let hidden = self.embed_tokens.embeddings().dim(1)?;
        let h0 = self
            .embed_tokens
            .embeddings()
            .narrow(0, tok as usize, 1)?
            .reshape((1, 1, hidden))?;
        self.static_h0
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("graph h0 missing".into()))?
            .copy2(&h0)?;

        let (num_heads, head_dim) = self
            .layers
            .iter()
            .find_map(|l| match &l.attn {
                LayerKind::FullAttn(f) => Some((f.num_heads, f.head_dim)),
                _ => None,
            })
            .unwrap_or((24, 256));
        let max_len = self.graph_max_len;
        if let Some(idx) = &self.static_idx {
            idx.copy_from_host_u32(&vec![pos as u32; num_heads * head_dim])?;
        }
        if let (Some(buf), Some(src)) = (&self.static_mask_row, &self.mask_src) {
            let row = src.narrow(0, pos, 1)?.reshape((1, 1, 1, max_len))?;
            buf.copy2(&row)?;
        }
        if let Some((cb, sb)) = &self.static_rope {
            cb.copy2(&self.rotary.cos.narrow(0, pos, 1)?)?;
            sb.copy2(&self.rotary.sin.narrow(0, pos, 1)?)?;
        }

        // Debug: run the static path directly (no replay).
        if std::env::var_os("AURORA_GRAPH_STATIC_ONLY").is_some() {
            let saved = self.graph_exec.take();
            let out = self.forward(input, pos);
            self.graph_exec = saved;
            return out;
        }

        self.graph_exec
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("graph exec missing".into()))?
            .launch()
            .map_err(|e| candle_core::Error::Msg(format!("{e}")))?;
        self.graph_logits
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("graph logits missing".into()))
            .map(|t| t.clone())
    }

    #[cfg(not(feature = "hip-graph"))]
    fn graph_step(&mut self, _input: &Tensor, _pos: usize) -> Result<Tensor> {
        Err(candle_core::Error::Msg("HIP graph requires the hip-graph feature".into()))
    }

    /// Drop the static graph state and return to the normal path.
    pub fn clear_static_graph(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_static();
        }
        self.static_h0 = None;
        self.static_idx = None;
        self.static_mask_row = None;
        self.mask_src = None;
        self.static_rope = None;
        self.graph_exec = None;
        self.graph_logits = None;
        self.graph_max_len = 0;
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (_b, seq_len) = input.dims2()?;
        if seq_len == 1 && self.graph_exec.is_some() {
            return self.graph_step(input, offset);
        }
        let mut h = if let Some(h0) = &self.static_h0 {
            h0.clone()
        } else {
            self.embed_tokens.forward(input)?
        };

        for layer in &mut self.layers {
            h = layer.forward(&h, offset)?;
        }

        let h_norm = self.output_norm.forward(&h)?;
        let last_hidden = h_norm.narrow(1, seq_len - 1, 1)?;
        self.lm_head.forward(&last_hidden)?.squeeze(1)
    }

    pub fn forward_all(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let mut h = self.embed_tokens.forward(input)?;

        for layer in &mut self.layers {
            h = layer.forward(&h, offset)?;
        }

        let h_norm = self.output_norm.forward(&h)?;
        self.lm_head.forward(&h_norm)
    }

    pub fn reset_cache(&mut self) {
        for layer in &mut self.layers {
            layer.reset_cache();
        }
    }
}
