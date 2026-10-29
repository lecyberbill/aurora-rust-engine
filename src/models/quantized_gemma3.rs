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

use crate::models::ops::{repeat_kv, rms_norm, rope, RmsNorm};

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

fn gelu_pytorch_tanh(x: &Tensor) -> Result<Tensor> {
    let orig_dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let c = (2.0f64 / std::f64::consts::PI).sqrt();
    let x_cubed = (x_f32.sqr()? * &x_f32)?;
    let inner = (&x_f32 + (x_cubed * 0.044715f64)?)?;
    let tanh = (inner * c)?.tanh()?;
    let result = ((x_f32 * 0.5)? * (tanh + 1.0)?)?;
    result.to_dtype(orig_dtype)
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
        let gelu = gelu_pytorch_tanh(&gate)?;
        let gated = (gelu * up)?;
        self.feed_forward_down.forward(&gated)
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(
        head_dim: usize,
        rope_frequency: f32,
        partial_rotary_factor: f32,
        device: &Device,
    ) -> Result<Self> {
        let half_dim = head_dim / 2;
        let rope_angles = (partial_rotary_factor * head_dim as f32 / 2.0).round() as usize;
        let rope_angles = rope_angles.min(half_dim).max(1);

        let mut inv_freq_vec = Vec::with_capacity(half_dim);
        for i in 0..rope_angles {
            inv_freq_vec.push(1f32 / rope_frequency.powf((2 * i) as f32 / head_dim as f32));
        }
        // Pad with zeros for non-rotated dimensions -> cos=1, sin=0 -> identity
        inv_freq_vec.extend(std::iter::repeat_n(0f32, half_dim - rope_angles));

        let inv_freq = Tensor::from_vec(inv_freq_vec, (1, half_dim), device)?;
        let t = Tensor::arange(0u32, MAX_SEQ_LEN as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((MAX_SEQ_LEN, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;
        Ok(Self { cos, sin })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        index_pos: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _head_dim) = q.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        let q_embed = rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
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

    layer_output_scale: Option<Tensor>,
    rms_norm_eps: f64,
    attention_scale: f64,
    is_gemma4: bool,

    kv_cache: Option<(Tensor, Tensor)>,

    // HIP-graph static decode buffers.
    static_k: Option<Tensor>,   // [1, n_head, head_dim, max_len] (transposed K)
    static_v: Option<Tensor>,   // [1, n_head, max_len, head_dim]
    static_rope: Option<(Tensor, Tensor)>, // cos/sin [1, head_dim/2]
    static_idx_k: Option<Tensor>,
    static_idx_v: Option<Tensor>,
    static_mask: Option<Tensor>,
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
                            f32::NEG_INFINITY
                        } else {
                            0f32
                        }
                    })
                })
                .collect()
        } else {
            (0..seq_len)
                .flat_map(|i| (0..seq_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0f32 }))
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
        let v = if self.is_gemma4 || self.attention_wv.is_none() {
            // Gemma 4 & Gemma 3 without v_proj: always apply scale-free RMSNorm to Vcur
            rms_norm(&v.contiguous()?, self.rms_norm_eps)?
        } else {
            v
        };

        // RoPE: static buffers for the graph path, table + fused kernel otherwise.
        let (q, k) = if let Some((cos_buf, sin_buf)) = &self.static_rope {
            (
                rope(&q.contiguous()?, cos_buf, sin_buf)?,
                rope(&k.contiguous()?, cos_buf, sin_buf)?,
            )
        } else {
            self.rotary_embedding
                .apply_rotary_emb_qkv(&q, &k, index_pos)?
        };

        // K/V: static graph cache (full max_len) or the growing cache.
        let (k_cache, v, static_on, flash_ctx) = if let (Some(kt), Some(vv)) = (&self.static_k, &self.static_v) {
            let idx = self.static_idx_v.as_ref().unwrap();
            kt.scatter_set(idx, &k, 2)?;
            vv.scatter_set(idx, &v, 2)?;
            let flash = if seq_len == 1 {
                match &self.static_mask {
                    Some(m) => {
                        let mask = m.flatten_all()?;
                        crate::device::flash_decode_masked(&q, kt, vv, self.attention_scale, &mask)?
                    }
                    None => None,
                }
            } else {
                None
            };
            if flash.is_some() {
                (kt.clone(), vv.clone(), true, flash)
            } else {
                let n_rep = self.n_head / self.n_kv_head;
                let kr = repeat_kv(kt.clone(), n_rep)?;
                let vr = repeat_kv(vv.clone(), n_rep)?;
                (kr, vr, true, None)
            }
        } else {
            let (k, v) = match &self.kv_cache {
                None => (k, v),
                Some((k_cache, v_cache)) => {
                    if index_pos == 0 {
                        (k, v)
                    } else {
                        (
                            Tensor::cat(&[k_cache, &k], 2)?,
                            Tensor::cat(&[v_cache, &v], 2)?,
                        )
                    }
                }
            };
            self.kv_cache = Some((k.clone(), v.clone()));
            let flash = if seq_len == 1 {
                crate::device::flash_decode(&q, &k, &v, self.attention_scale)?
            } else {
                None
            };
            if flash.is_some() {
                (k, v, false, flash)
            } else {
                let k = repeat_kv(k, self.n_head / self.n_kv_head)?;
                let v = repeat_kv(v, self.n_head / self.n_kv_head)?;
                (k, v, false, None)
            }
        };

        // Scaled Dot-Product Attention (or the split-K flash kernel for decode).
        let attn_output = if let Some(ctx) = flash_ctx {
            ctx
        } else {
            let k_t = k_cache.transpose(2, 3)?.contiguous()?;
            let scale = self.attention_scale;
            let mut attn_weights = (q.matmul(&k_t)? * scale)?;

            if static_on {
                if let Some(m) = &self.static_mask {
                    attn_weights = attn_weights.broadcast_add(m)?;
                }
            } else if let Some(mask) = mask {
                attn_weights = attn_weights.broadcast_add(mask)?;
            }

            let attn_weights = crate::device::softmax_last_dim(&attn_weights)?;
            attn_weights.matmul(&v)?
        };

        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, self.q_dim))?;

        self.attention_wo.forward(&attn_output)
    }
}

#[derive(Debug)]
pub struct ModelWeights {
    tok_embeddings: Embedding,
    embedding_length: usize,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,
    final_logit_softcapping: Option<f64>,
    emb_weight: Tensor,
    // HIP-graph decode state.
    static_h0: Option<Tensor>,
    static_idx: Option<Tensor>,
    mask_src_global: Option<Tensor>,
    mask_src_slide: Option<Tensor>,
    mask_row_global: Option<Tensor>,
    mask_row_slide: Option<Tensor>,
    graph_exec: Option<crate::models::text::causal::HipGraphExec>,
    graph_logits: Option<Tensor>,
    graph_max_len: usize,
}

impl ModelWeights {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let model_prefix = ["gemma4", "gemma3", "gemma2", "gemma", "gemma-embedding"]
            .iter()
            .find(|p| {
                ct.metadata
                    .contains_key(&format!("{}.attention.head_count", p))
                    || ct.metadata.contains_key(&format!("{}.block_count", p))
            })
            .copied()
            .unwrap_or("gemma3");

        let md_get = |s: &str| {
            let key = format!("{model_prefix}.{s}");
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
            .get(&format!("{model_prefix}.attention.head_count_kv"))
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
            .get(&format!("{model_prefix}.attention.sliding_window_pattern"))
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

            let is_gemma4 = model_prefix == "gemma4";
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
                layer_output_scale,
                rms_norm_eps,
                attention_scale,
                is_gemma4,
                kv_cache: None,
                static_k: None,
                static_v: None,
                static_rope: None,
                static_idx_k: None,
                static_idx_v: None,
                static_mask: None,
            })
        }

        let emb_weight = tok_embeddings.clone();
        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            embedding_length,
            layers,
            norm,
            output: QMatMul::from_qtensor(output)?,
            final_logit_softcapping,
            emb_weight,
            static_h0: None,
            static_idx: None,
            mask_src_global: None,
            mask_src_slide: None,
            mask_row_global: None,
            mask_row_slide: None,
            graph_exec: None,
            graph_logits: None,
            graph_max_len: 0,
        })
    }

    /// Whether the static HIP-graph decode path can serve this model.
    pub fn supports_static_graph(&self) -> bool {
        !self.layers.is_empty() && self.layers.iter().all(|l| !l.is_gemma4)
    }

    /// Allocate static decode buffers, seed them from the prefill K/V, warm up
    /// and capture one decode step into a HIP graph.
    #[cfg(feature = "hip-graph")]
    pub fn setup_static_graph(&mut self, prefill_len: usize, max_len: usize) -> Result<()> {
        use crate::models::text::causal::HipGraphExec;
        use candle_core::rocm_backend::rocm_rs::hip::bindings;

        if !self.supports_static_graph() {
            return Err(candle_core::Error::Msg("HIP graph: unsupported gemma variant".into()));
        }
        if prefill_len + 16 >= max_len {
            return Err(candle_core::Error::Msg("HIP graph: context does not fit".into()));
        }
        for l in &self.layers {
            if let Some(w) = l.sliding_window_size {
                if w < max_len {
                    return Err(candle_core::Error::Msg("HIP graph: sliding window < max_len".into()));
                }
            }
        }
        let dev = self.emb_weight.device().clone();
        if !dev.is_rocm() {
            return Err(candle_core::Error::Msg("HIP graph: not a ROCm device".into()));
        }
        let dtype = self.emb_weight.dtype();

        let build = |window: Option<usize>| -> Vec<f32> {
            let mut m = vec![0f32; max_len * max_len];
            for i in 0..max_len {
                for j in 0..max_len {
                    let blocked = j > i || window.map_or(false, |w| j + w < i);
                    if blocked {
                        m[i * max_len + j] = -1e4;
                    }
                }
            }
            m
        };
        let sw = self.layers.iter().find_map(|l| l.sliding_window_size).unwrap_or(1024);
        let mask_src_global = Tensor::from_vec(build(None), (max_len, max_len), &dev)?;
        let mask_src_slide = Tensor::from_vec(build(Some(sw)), (max_len, max_len), &dev)?;
        let mask_row_global = Tensor::zeros((1, 1, 1, max_len), DType::F32, &dev)?;
        let mask_row_slide = Tensor::zeros((1, 1, 1, max_len), DType::F32, &dev)?;

        let head_dim = self.layers[0].head_dim;
        let n_kv = self.layers[0].n_kv_head;
        let shared_idx = Tensor::zeros((1, n_kv, 1, head_dim), DType::U32, &dev)?;
        shared_idx.copy_from_host_u32(&vec![prefill_len as u32; n_kv * head_dim])?;

        for layer in self.layers.iter_mut() {
            let (pk, pv) = layer
                .kv_cache
                .clone()
                .ok_or_else(|| candle_core::Error::Msg("HIP graph: empty prefill KV".into()))?;
            let (b, nkv, s, d) = pk.shape().dims4().map_err(|e| candle_core::Error::Msg(format!("{e}")))?;
            let _ = b;

            // Unique GQA KV (the flash kernel broadcasts over the query heads).
            let sk = Tensor::zeros((1, nkv, max_len, layer.head_dim), dtype, &dev)?;
            let sv = Tensor::zeros((1, nkv, max_len, layer.head_dim), dtype, &dev)?;
            let mut idx_kv = Vec::with_capacity(nkv * s * d);
            for _ in 0..nkv {
                for si in 0..s {
                    for _ in 0..d {
                        idx_kv.push(si as u32);
                    }
                }
            }
            let idx_v = Tensor::from_vec(idx_kv, (1, nkv, s, d), &dev)?;
            sk.scatter_set(&idx_v, &pk, 2)?;
            sv.scatter_set(&idx_v, &pv, 2)?;

            let cos_buf = Tensor::zeros((1, layer.head_dim / 2), dtype, &dev)?;
            let sin_buf = Tensor::zeros((1, layer.head_dim / 2), dtype, &dev)?;
            cos_buf.copy2(&layer.rotary_embedding.cos.narrow(0, prefill_len, 1)?)?;
            sin_buf.copy2(&layer.rotary_embedding.sin.narrow(0, prefill_len, 1)?)?;

            let mask_row = if layer.sliding_window_size.is_some() {
                mask_row_slide.clone()
            } else {
                mask_row_global.clone()
            };

            layer.static_k = Some(sk);
            layer.static_v = Some(sv);
            layer.static_rope = Some((cos_buf, sin_buf));
            layer.static_idx_k = None;
            layer.static_idx_v = Some(shared_idx.clone());
            layer.static_mask = Some(mask_row);
        }

        let h0 = Tensor::zeros((1, 1, self.embedding_length), dtype, &dev)?;
        self.static_idx = Some(shared_idx);
        let row_g = mask_src_global.narrow(0, prefill_len, 1)?.reshape((1, 1, 1, max_len))?;
        mask_row_global.copy2(&row_g)?;
        let row_s = mask_src_slide.narrow(0, prefill_len, 1)?.reshape((1, 1, 1, max_len))?;
        mask_row_slide.copy2(&row_s)?;

        self.static_h0 = Some(h0);
        self.mask_src_global = Some(mask_src_global);
        self.mask_src_slide = Some(mask_src_slide);
        self.mask_row_global = Some(mask_row_global);
        self.mask_row_slide = Some(mask_row_slide);
        self.graph_max_len = max_len;
        self.graph_exec = None;
        self.graph_logits = None;

        let dummy = Tensor::zeros((1, 1), DType::U32, &dev)?;
        for _ in 0..3 {
            let _ = self.forward(&dummy, prefill_len)?;
        }
        dev.synchronize().map_err(|e| candle_core::Error::Msg(format!("{e}")))?;

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
        dev.synchronize().map_err(|e| candle_core::Error::Msg(format!("{e}")))?;
        let mut exec = std::ptr::null_mut();
        let r = unsafe {
            bindings::hipGraphInstantiate(&mut exec, graph, std::ptr::null_mut(), std::ptr::null_mut(), 0)
        };
        if r != bindings::hipError_t_hipSuccess {
            return Err(candle_core::Error::Msg(format!("hipGraphInstantiate: {r}")));
        }
        unsafe { bindings::hipGraphDestroy(graph); }

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
    fn graph_step(&mut self, x: &Tensor, pos: usize) -> Result<Tensor> {
        let flat = x.flatten_all()?;
        let tok = match flat.dtype() {
            DType::U32 => flat.to_vec1::<u32>()?,
            DType::I64 => flat.to_vec1::<i64>()?.into_iter().map(|v| v as u32).collect(),
            other => return Err(candle_core::Error::Msg(format!("graph step: token dtype {other:?}"))),
        };
        let tok = *tok.first().ok_or_else(|| candle_core::Error::Msg("graph step: empty token".into()))?;
        let h0 = self
            .emb_weight
            .narrow(0, tok as usize, 1)?
            .reshape((1, 1, self.embedding_length))?;
        self.static_h0
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("graph h0 missing".into()))?
            .copy2(&h0)?;

        let max_len = self.graph_max_len;
        let n_kv = self.layers[0].n_kv_head;
        let head_dim = self.layers[0].head_dim;
        // One host->device sync per token: the scatter index and the two mask
        // rows are shared by every layer.
        if let Some(idx) = &self.static_idx {
            idx.copy_from_host_u32(&vec![pos as u32; n_kv * head_dim])?;
        }
        if let (Some(buf), Some(src)) = (&self.mask_row_global, &self.mask_src_global) {
            let row = src.narrow(0, pos, 1)?.reshape((1, 1, 1, max_len))?;
            buf.copy2(&row)?;
        }
        if let (Some(buf), Some(src)) = (&self.mask_row_slide, &self.mask_src_slide) {
            let row = src.narrow(0, pos, 1)?.reshape((1, 1, 1, max_len))?;
            buf.copy2(&row)?;
        }
        for layer in self.layers.iter() {
            if let Some((cb, sb)) = &layer.static_rope {
                cb.copy2(&layer.rotary_embedding.cos.narrow(0, pos, 1)?)?;
                sb.copy2(&layer.rotary_embedding.sin.narrow(0, pos, 1)?)?;
            }
        }

        // Debug: run the static path directly (no replay) to separate a static
        // math bug from a capture/replay bug.
        if std::env::var_os("AURORA_GRAPH_STATIC_ONLY").is_some() {
            let saved = self.graph_exec.take();
            let out = self.forward(x, pos);
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
    fn graph_step(&mut self, _x: &Tensor, _pos: usize) -> Result<Tensor> {
        Err(candle_core::Error::Msg("HIP graph requires the hip-graph feature".into()))
    }

    /// Drop the static graph state and return to the normal path.
    pub fn clear_static_graph(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.static_k = None;
            layer.static_v = None;
            layer.static_rope = None;
            layer.static_idx_k = None;
            layer.static_idx_v = None;
            layer.static_mask = None;
        }
        self.static_h0 = None;
        self.static_idx = None;
        self.mask_src_global = None;
        self.mask_src_slide = None;
        self.mask_row_global = None;
        self.mask_row_slide = None;
        self.graph_exec = None;
        self.graph_logits = None;
        self.graph_max_len = 0;
    }

    /// Number of tokens currently represented in the KV cache (first layer).
    pub fn prefix_cache_len(&self) -> Option<usize> {
        for layer in &self.layers {
            if let Some((k, _)) = &layer.kv_cache {
                return k.dim(2).ok();
            }
        }
        None
    }

    /// Keep only the first `len` KV positions (prefix-cache rollback).
    pub fn truncate_prefix_cache(&mut self, len: usize) -> bool {
        for layer in &mut self.layers {
            if let Some((k, v)) = &layer.kv_cache {
                let k = match k.narrow(2, 0, len) {
                    Ok(t) => t,
                    Err(_) => return false,
                };
                let v = match v.narrow(2, 0, len) {
                    Ok(t) => t,
                    Err(_) => return false,
                };
                layer.kv_cache = Some((k, v));
            }
        }
        true
    }

    pub fn reset_prefix_cache(&mut self) {
        for layer in &mut self.layers {
            layer.kv_cache = None;
        }
    }

    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (b_sz, seq_len) = x.dims2()?;

        if seq_len == 1 && self.graph_exec.is_some() {
            return self.graph_step(x, index_pos);
        }

        // A static h0 (graph capture) holds the raw embedding row; the sqrt
        // scale is applied inside the captured graph.
        let mut layer_in = if let Some(h0) = &self.static_h0 {
            (h0 * (self.embedding_length as f64).sqrt())?
        } else {
            let e = self.tok_embeddings.forward(x)?;
            (e * (self.embedding_length as f64).sqrt())?
        };

        for layer in self.layers.iter_mut() {
            let attention_mask = if seq_len == 1 {
                None
            } else {
                Some(layer.mask(b_sz, seq_len, index_pos, layer_in.dtype(), x.device())?)
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

    /// Embed token IDs and apply Gemma sqrt(embedding_length) scaling
    pub fn embed_tokens(&self, x: &Tensor) -> Result<Tensor> {
        let embeds = self.tok_embeddings.forward(x)?;
        embeds * (self.embedding_length as f64).sqrt()
    }

    /// Forward pass starting from pre-computed multimodal embeddings [B, SeqLen, HiddenDim]
    pub fn forward_embeds(&mut self, mut layer_in: Tensor, index_pos: usize) -> Result<Tensor> {
        let (b_sz, seq_len, _) = layer_in.dims3()?;

        for layer in self.layers.iter_mut() {
            let attention_mask = if seq_len == 1 {
                None
            } else {
                Some(layer.mask(b_sz, seq_len, index_pos, layer_in.dtype(), layer_in.device())?)
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
