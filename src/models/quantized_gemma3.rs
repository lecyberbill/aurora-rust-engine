// [WFGY] Zone: TRANSIT | λ: 0.15 | Fallbacks: 0 | Action: ROCm-compatible quantized Gemma 3 and Gemma 4 implementation
//! Custom implementation of quantized Gemma 3 and Gemma 4 with universal RMSNorm (works on CPU, CUDA, and AMD ROCm).
//! Solves Candle kernel gaps for ROCm: `rms_norm`, `rotary_emb`, `softmax_last_dim`.
//! Supports Gemma 4 hybrid per-layer head dimension (256 SWA / 512 Global), alternating head_count_kv,
//! and final_logit_softcapping.

use candle_core::quantized::gguf_file;
use candle_core::quantized::QTensor;
use candle_core::D;
use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module};

pub const MAX_SEQ_LEN: usize = 8192;
pub const DEFAULT_SLIDING_WINDOW_TYPE: usize = 6;
pub const DEFAULT_ROPE_FREQUENCY: f32 = 1_000_000.;
pub const DEFAULT_ROPE_FREQUENCY_SLIDING: f32 = 10_000.;
pub const DEFAULT_ROPE_FREQUENCY_SCALE_FACTOR: f32 = 1.;

pub fn rms_norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    let last_dim = x.dims().len() - 1;
    let x_f32 = x.to_dtype(DType::F32)?;
    let sq = x_f32.sqr()?;
    let mean = sq.mean_keepdim(last_dim)?;
    let rms = (mean + eps)?.sqrt()?;
    let norm = x_f32.broadcast_div(&rms)?;
    norm.to_dtype(x.dtype())
}

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

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Universal RMSNorm compatible with CPU, CUDA, and ROCm
        let norm = rms_norm(x, self.eps)?;
        let scaled = norm.to_dtype(DType::F32)?.broadcast_mul(&self.weight.to_dtype(DType::F32)?)?;
        scaled.to_dtype(x.dtype())
    }
}

#[derive(Debug, Clone)]
struct QMatMul {
    inner: candle_core::quantized::QMatMul,
}

impl QMatMul {
    fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        let inner = candle_core::quantized::QMatMul::from_qtensor(qtensor)?;
        Ok(Self { inner })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.inner.forward(xs)
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    feed_forward_gate: QMatMul,
    feed_forward_up: QMatMul,
    feed_forward_down: QMatMul,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.feed_forward_gate.forward(xs)?;
        let up = self.feed_forward_up.forward(xs)?;
        let silu = candle_nn::ops::silu(&gate)
            .or_else(|_| {
                let sig = candle_nn::ops::sigmoid(&gate)?;
                gate.mul(&sig)
            })?;
        let gated = (silu * up)?;
        self.feed_forward_down.forward(&gated)
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(head_dim: usize, rope_frequency: f32, partial_rotary_factor: f32, device: &Device) -> Result<Self> {
        let half_dim = head_dim / 2;
        let rope_angles = (partial_rotary_factor * (head_dim as f32) / 2.0).round() as usize;
        let rope_angles = rope_angles.min(half_dim).max(1);

        let mut inv_freq_vec = Vec::with_capacity(half_dim);
        for i in 0..rope_angles {
            inv_freq_vec.push(1f32 / rope_frequency.powf((2 * i) as f32 / head_dim as f32));
        }
        if half_dim > rope_angles {
            inv_freq_vec.extend(std::iter::repeat_n(0f32, half_dim - rope_angles));
        }

        let theta = Tensor::new(inv_freq_vec.as_slice(), device)?;
        let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((MAX_SEQ_LEN, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        let cos = idx_theta.cos()?;
        let sin = idx_theta.sin()?;
        Ok(Self { sin, cos })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        index_pos: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, head_dim) = q.dims4()?;
        let half_dim = head_dim / 2;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;

        // Duplicate cos and sin across the second half of head_dim: [seq_len, head_dim]
        // Then reshape to [1, 1, seq_len, head_dim] for broadcast over (batch, heads, seq, head_dim)
        let cos = Tensor::cat(&[&cos, &cos], 1)?.unsqueeze(0)?.unsqueeze(0)?;
        let sin = Tensor::cat(&[&sin, &sin], 1)?.unsqueeze(0)?.unsqueeze(0)?;

        let rotate_fn = |x: &Tensor| -> Result<Tensor> {
            let x_f32 = x.to_dtype(DType::F32)?;
            let x1 = x_f32.narrow(3, 0, half_dim)?;
            let x2 = x_f32.narrow(3, half_dim, half_dim)?;
            let x_rot = Tensor::cat(&[&(x2.neg()?), &x1], 3)?;
            let out = (x_f32.broadcast_mul(&cos)? + x_rot.broadcast_mul(&sin)?)?;
            out.to_dtype(x.dtype())
        };

        let q_embed = rotate_fn(&q.contiguous()?)?;
        let k_embed = rotate_fn(&k.contiguous()?)?;
        Ok((q_embed, k_embed))
    }
}

fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(xs)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = xs.dims4()?;
        xs.unsqueeze(2)?
            .expand((b_sz, n_kv_head, n_rep, seq_len, head_dim))?
            .reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}

#[derive(Debug, Clone)]
struct LayerWeights {
    attention_wq: QMatMul,
    attention_wk: QMatMul,
    attention_wv: Option<QMatMul>,
    attention_wo: QMatMul,

    attention_q_norm: RmsNorm,
    attention_k_norm: RmsNorm,

    attention_norm: RmsNorm,
    post_attention_norm: RmsNorm,
    ffn_norm: RmsNorm,
    post_ffn_norm: RmsNorm,

    mlp: Mlp,

    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    q_dim: usize,

    sliding_window_size: Option<usize>,

    rotary_embedding: RotaryEmbedding,
    neg_inf: Tensor,

    layer_output_scale: Option<Tensor>,
    rms_norm_eps: f64,
    attention_scale: f64,

    kv_cache: Option<(Tensor, Tensor)>,
}

impl LayerWeights {
    fn mask(
        &self,
        b_sz: usize,
        seq_len: usize,
        index_pos: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let mask: Vec<_> = if let Some(sliding_window_size) = self.sliding_window_size {
            (0..seq_len)
                .flat_map(|i| {
                    (0..seq_len).map(move |j| {
                        if i < j || j + sliding_window_size < i {
                            0u32
                        } else {
                            1u32
                        }
                    })
                })
                .collect()
        } else {
            (0..seq_len)
                .flat_map(|i| (0..seq_len).map(move |j| if i < j { 0u32 } else { 1u32 }))
                .collect()
        };
        let mask = Tensor::from_slice(&mask, (seq_len, seq_len), device)?;
        let mask = if index_pos > 0 {
            let mask0 = Tensor::zeros((seq_len, index_pos), DType::F32, device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b_sz, 1, seq_len, seq_len + index_pos))?
            .to_dtype(dtype)
    }

    fn forward_attn(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        let q = self.attention_wq.forward(x)?;
        let k = self.attention_wk.forward(x)?;
        let v = match &self.attention_wv {
            Some(wv) => wv.forward(x)?,
            None => k.clone(),
        };

        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;

        let q = self.attention_q_norm.forward(&q.contiguous()?)?;
        let k = self.attention_k_norm.forward(&k.contiguous()?)?;
        let v = if self.attention_scale == 1.0f64 {
            // Gemma 4: always apply scale-free RMSNorm to Vcur
            rms_norm(&v.contiguous()?, self.rms_norm_eps)?
        } else if self.attention_wv.is_none() {
            rms_norm(&v.contiguous()?, self.rms_norm_eps)?
        } else {
            v
        };

        let (q, k) = self
            .rotary_embedding
            .apply_rotary_emb_qkv(&q, &k, index_pos)?;

        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((k_cache, v_cache)) => {
                if index_pos == 0 {
                    (k, v)
                } else {
                    let k = Tensor::cat(&[k_cache, &k], 2)?;
                    let v = Tensor::cat(&[v_cache, &v], 2)?;
                    (k, v)
                }
            }
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        // Repeat KV for GQA
        let k = repeat_kv(k, self.n_head / self.n_kv_head)?;
        let v = repeat_kv(v, self.n_head / self.n_kv_head)?;

        // Scaled Dot-Product Attention
        let scale = self.attention_scale;
        let mut attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

        if let Some(mask) = mask {
            let mask = mask.broadcast_as(attn_weights.shape())?;
            let neg_inf = self.neg_inf.broadcast_as(attn_weights.dims())?;
            attn_weights = mask.eq(0u32)?.where_cond(&neg_inf, &attn_weights)?;
        }

        let attn_weights = crate::device::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, self.q_dim))?;

        self.attention_wo.forward(&attn_output)
    }
}

#[derive(Debug, Clone)]
pub struct ModelWeights {
    tok_embeddings: Embedding,
    embedding_length: usize,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,
    final_logit_softcapping: Option<f64>,
}

impl ModelWeights {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let prefix = ["gemma4", "gemma3", "gemma2", "gemma", "gemma-embedding"]
            .iter()
            .find(|p| {
                ct.metadata
                    .contains_key(&format!("{}.attention.head_count", p))
                    || ct.metadata.contains_key(&format!("{}.block_count", p))
            })
            .copied()
            .unwrap_or("gemma3");

        let md_get = |s: &str| {
            let key = format!("{prefix}.{s}");
            match ct.metadata.get(&key) {
                None => candle_core::bail!("cannot find {key} in metadata"),
                Some(v) => Ok(v),
            }
        };

        let block_count = md_get("block_count")?.to_u32()? as usize;
        let embedding_length = md_get("embedding_length")?.to_u32()? as usize;
        let head_count = md_get("attention.head_count")?.to_u32()? as usize;
        let rms_norm_eps = md_get("attention.layer_norm_rms_epsilon")?.to_f32()? as f64;

        // Final logit softcapping (e.g. 30.0 in Gemma 4)
        let final_logit_softcapping = md_get("final_logit_softcapping")
            .ok()
            .and_then(|v| v.to_f32().ok())
            .map(|f| f as f64);

        // Pattern of head_count_kv across layers
        let head_count_kv_arr: Option<Vec<usize>> = ct.metadata
            .get(&format!("{prefix}.attention.head_count_kv"))
            .and_then(|v| match v {
                gguf_file::Value::Array(arr) => {
                    let res: std::result::Result<Vec<usize>, _> = arr.iter().map(|x| match x {
                        gguf_file::Value::I32(i) => Ok(*i as usize),
                        gguf_file::Value::U32(u) => Ok(*u as usize),
                        _ => Err(()),
                    }).collect();
                    res.ok()
                }
                _ => None,
            });

        // Default head_count_kv if not array
        let default_head_count_kv = if head_count_kv_arr.is_none() {
            md_get("attention.head_count_kv").and_then(|v| Ok(v.to_u32()? as usize)).unwrap_or(head_count)
        } else {
            head_count
        };

        // Sliding window pattern
        let sliding_window_pattern: Option<Vec<bool>> = ct.metadata
            .get(&format!("{prefix}.attention.sliding_window_pattern"))
            .and_then(|v| match v {
                gguf_file::Value::Array(arr) => {
                    let res: std::result::Result<Vec<bool>, _> = arr.iter().map(|x| match x {
                        gguf_file::Value::Bool(b) => Ok(*b),
                        _ => Err(()),
                    }).collect();
                    res.ok()
                }
                _ => None,
            });

        let default_sliding_window_size = md_get("attention.sliding_window")
            .and_then(|v| Ok(v.to_u32()? as usize))
            .unwrap_or(1024);

        let sliding_window_type = md_get("attention.sliding_window_type")
            .and_then(|m| Ok(m.to_u32()? as usize))
            .unwrap_or(DEFAULT_SLIDING_WINDOW_TYPE);

        // Base RoPE frequencies
        let rope_freq_base = md_get("rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(DEFAULT_ROPE_FREQUENCY);

        let rope_freq_base_sliding = md_get("rope.freq_base_swa")
            .or_else(|_| md_get("rope.local_freq_base"))
            .and_then(|m| m.to_f32())
            .unwrap_or(DEFAULT_ROPE_FREQUENCY_SLIDING);

        let key_length_global = md_get("attention.key_length")
            .and_then(|m| Ok(m.to_u32()? as usize))
            .unwrap_or(256);

        let key_length_swa = md_get("attention.key_length_swa")
            .and_then(|m| Ok(m.to_u32()? as usize))
            .unwrap_or(key_length_global);

        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        let tok_embeddings = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings.dequantize(device)?;
        let norm = RmsNorm::from_qtensor(
            ct.tensor(reader, "output_norm.weight", device)?,
            rms_norm_eps,
        )?;
        let output = match ct.tensor(reader, "output.weight", device) {
            Ok(tensor) => tensor,
            Err(_) => ct.tensor(reader, "token_embd.weight", device)?,
        };

        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");

            let attention_wq = ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?;
            let attention_wk = ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?;
            let attention_wv = ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device).ok();
            let attention_wo =
                ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?;

            let attention_q_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.attn_q_norm.weight"), device)?,
                rms_norm_eps,
            )?;

            let attention_k_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.attn_k_norm.weight"), device)?,
                rms_norm_eps,
            )?;

            let attention_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?,
                rms_norm_eps,
            )?;

            let post_attention_norm = RmsNorm::from_qtensor(
                ct.tensor(
                    reader,
                    &format!("{prefix}.post_attention_norm.weight"),
                    device,
                )?,
                rms_norm_eps,
            )?;

            let ffn_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?,
                rms_norm_eps,
            )?;

            let post_ffn_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.post_ffw_norm.weight"), device)?,
                rms_norm_eps,
            )?;

            let feed_forward_gate =
                ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)?;
            let feed_forward_up = ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?;
            let feed_forward_down =
                ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?;

            let layer_output_scale = ct
                .tensor(reader, &format!("{prefix}.layer_output_scale.weight"), device)
                .ok()
                .and_then(|t| t.dequantize(device).ok());

            let mlp = Mlp {
                feed_forward_gate: QMatMul::from_qtensor(feed_forward_gate)?,
                feed_forward_up: QMatMul::from_qtensor(feed_forward_up)?,
                feed_forward_down: QMatMul::from_qtensor(feed_forward_down)?,
            };

            // Determine if current layer is sliding window
            let is_sliding = if let Some(pattern) = &sliding_window_pattern {
                pattern.get(layer_idx).copied().unwrap_or(true)
            } else {
                (layer_idx + 1) % sliding_window_type > 0
            };

            let sliding_window_size = is_sliding.then_some(default_sliding_window_size);
            let layer_rope_frequency = if is_sliding {
                rope_freq_base_sliding
            } else {
                rope_freq_base
            };

            let layer_kv_heads = if let Some(arr) = &head_count_kv_arr {
                arr.get(layer_idx).copied().unwrap_or(default_head_count_kv)
            } else {
                default_head_count_kv
            };

            // Head dim: 256 for sliding (SWA), 512 for global
            let layer_head_dim = if is_sliding {
                key_length_swa
            } else {
                key_length_global
            };

            // In Gemma 4 global layers, partial rotary factor is 0.25 (512 * 0.25 = 128 angles)
            let partial_rotary_factor = if is_sliding {
                1.0f32
            } else {
                0.25f32
            };

            let rotary_embedding = RotaryEmbedding::new(layer_head_dim, layer_rope_frequency, partial_rotary_factor, device)?;
            let q_dim = head_count * layer_head_dim;

            let attention_wv = match attention_wv {
                Some(wv) => Some(QMatMul::from_qtensor(wv)?),
                None => None,
            };

            let is_gemma4 = prefix == "gemma4";
            let attention_scale = if is_gemma4 {
                1.0f64
            } else {
                1.0 / (layer_head_dim as f64).sqrt()
            };

            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_q_norm,
                attention_k_norm,
                attention_norm,
                post_attention_norm,
                ffn_norm,
                post_ffn_norm,
                mlp,
                n_head: head_count,
                n_kv_head: layer_kv_heads,
                head_dim: layer_head_dim,
                q_dim,
                sliding_window_size,
                rotary_embedding,
                neg_inf: neg_inf.clone(),
                layer_output_scale,
                rms_norm_eps,
                attention_scale,
                kv_cache: None,
            })
        }

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            embedding_length,
            layers,
            norm,
            output: QMatMul::from_qtensor(output)?,
            final_logit_softcapping,
        })
    }

    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (b_sz, seq_len) = x.dims2()?;

        let mut layer_in = self.tok_embeddings.forward(x)?;
        layer_in = (layer_in * (self.embedding_length as f64).sqrt())?;

        for layer in self.layers.iter_mut() {
            let attention_mask = if seq_len == 1 {
                None
            } else {
                Some(layer.mask(b_sz, seq_len, index_pos, x.dtype(), x.device())?)
            };

            // Attention block
            let residual = &layer_in;
            let x = layer.attention_norm.forward(&layer_in)?;
            let x = layer.forward_attn(&x, attention_mask.as_ref(), index_pos)?;
            let x = layer.post_attention_norm.forward(&x)?;
            let x = (x + residual)?;

            // Feed-forward block
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp.forward(&x)?;
            let x = layer.post_ffn_norm.forward(&x)?;
            let mut x = (x + residual)?;

            if let Some(scale) = &layer.layer_output_scale {
                let scale_broad = scale.broadcast_as(x.shape())?;
                x = (x * scale_broad)?;
            }

            layer_in = x;
        }

        let x = layer_in.i((.., seq_len - 1, ..))?;
        let x = self.norm.forward(&x)?;
        let logits = self.output.forward(&x)?;

        if let Some(sc) = self.final_logit_softcapping {
            let scaled = (logits / sc)?;
            let tanh = scaled.tanh()?;
            tanh * sc
        } else {
            Ok(logits)
        }
    }
}
