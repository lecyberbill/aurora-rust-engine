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
    let orig_dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    // sigmoid(x) = (1 + exp(-x))^-1
    let neg_x = x_f32.neg()?;
    let exp_neg = neg_x.exp()?;
    let denom = (exp_neg + 1.0)?;
    let sig = denom.recip()?;
    sig.to_dtype(orig_dtype)
}

fn silu(x: &Tensor) -> Result<Tensor> {
    let orig_dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let sig = sigmoid(&x_f32)?;
    let res = (x_f32 * sig)?;
    res.to_dtype(orig_dtype)
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

        // Apply RoPE
        let q = self.rotary_emb.apply(&q, offset)?;
        let k = self.rotary_emb.apply(&k, offset)?;

        // Update KV cache
        let (k_all, v_all) = match &self.kv_cache {
            Some((prev_k, prev_v)) => {
                let cat_k = Tensor::cat(&[prev_k, &k], 2)?;
                let cat_v = Tensor::cat(&[prev_v, &v], 2)?;
                (cat_k, cat_v)
            }
            None => (k, v),
        };
        self.kv_cache = Some((k_all.clone(), v_all.clone()));

        // GQA repeat
        let n_rep = self.num_heads / self.num_kv_heads;
        let k_rep = repeat_kv(k_all, n_rep)?;
        let v_rep = repeat_kv(v_all, n_rep)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let q_cont = q.contiguous()?;
        let k_t_cont = k_rep.transpose(2, 3)?.contiguous()?;
        let scores = (q_cont.matmul(&k_t_cont)? * scale)?;

        // Causal mask if seq_len > 1
        let (_b, _h, q_len, kv_len) = scores.dims4()?;
        let scores = if q_len > 1 {
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
        let ctx = probs_cont.matmul(&v_cont)?; // [B, num_heads, L, head_dim]

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

        // 4. L2 Norm on Q and K
        let q_norm = l2norm(&q_heads, 1e-6)?;
        let k_norm = l2norm(&k_heads, 1e-6)?;

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
        let (full_x, new_conv_state) = match &self.conv_state {
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
        self.conv_state = Some(new_conv_state);

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

        let mut current_state = match &self.recurrent_state {
            Some(s) => s.clone(),
            None => Tensor::zeros((self.num_v_heads, self.head_k_dim, self.head_v_dim), DType::F32, device)?,
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
            current_state = (cs_cont.broadcast_mul(&g_t)? + outer)?;

            // Readout: out_t = S^T * q_t = q_t.matmul(current_state)
            let q_t_row = q_t.unsqueeze(1)?.contiguous()?; // [48, 1, 128]
            let cs_final = current_state.contiguous()?;
            let out_t = q_t_row.matmul(&cs_final)?.squeeze(1)?; // [48, 128]
            outputs.push(out_t);
        }

        self.recurrent_state = Some(current_state);

        // Stack outputs along sequence dimension -> [B, L, 48, 128]
        let out_stacked = Tensor::stack(&outputs, 0)?; // [L, 48, 128]
        out_stacked.unsqueeze(0) // [1, L, 48, 128]
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
}

#[derive(Debug, Clone)]
pub struct ModelWeights {
    embed_tokens: Embedding,
    layers: Vec<LayerWeights>,
    output_norm: RmsNorm,
    lm_head: QMatMul,
    _device: Device,
    _dtype: DType,
    vocab_size: usize,
    eos_token_id: Option<u32>,
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

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (_b, seq_len) = input.dims2()?;
        let mut h = self.embed_tokens.forward(input)?;

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
