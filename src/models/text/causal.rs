// [WFGY] Zone: SAFE | λ: 0.30 | Fallbacks: 1 (fallback HuggingFace tokenizer when local not found) | Action: Universal CausalLM engine supporting Llama, DeepSeek, Qwen, Gemma, Mistral in GGUF & Safetensors

use std::sync::Arc;
use candle_core::{DType, Device, IndexOp, Result as CandleResult, Tensor};
use candle_nn::{Embedding, Module};
use tokenizers::Tokenizer;

use crate::error::{LuminaError, Result};
use crate::models::common::LinearLayer;
use crate::weights::WeightsSource;

/// Storage mechanism for layer KV states: standard unquantized tensors or 8-bit quantized storage.
pub enum KVCacheStorage {
    Standard {
        k: Option<Tensor>,
        v: Option<Tensor>,
    },
    Quantized(QuantizedLayerKVCache),
}

/// KV Cache for an individual transformer layer.
pub struct LayerKVCache {
    pub storage: KVCacheStorage,
    pub ssm_conv_state: Option<Tensor>, // [B, hidden_dim, conv_kernel - 1]
    pub ssm_recurrent_state: Option<Tensor>, // [B, num_heads, head_dim, state_dim]
}

impl LayerKVCache {
    pub fn new() -> Self {
        Self {
            storage: KVCacheStorage::Standard { k: None, v: None },
            ssm_conv_state: None,
            ssm_recurrent_state: None,
        }
    }

    pub fn new_quantized() -> Self {
        Self {
            storage: KVCacheStorage::Quantized(QuantizedLayerKVCache::new()),
            ssm_conv_state: None,
            ssm_recurrent_state: None,
        }
    }

    pub fn set_quantized(&mut self, quantized: bool) {
        if quantized {
            if !matches!(self.storage, KVCacheStorage::Quantized(_)) {
                self.storage = KVCacheStorage::Quantized(QuantizedLayerKVCache::new());
            }
        } else if !matches!(self.storage, KVCacheStorage::Standard { .. }) {
            self.storage = KVCacheStorage::Standard { k: None, v: None };
        }
    }

    pub fn is_quantized(&self) -> bool {
        matches!(self.storage, KVCacheStorage::Quantized(_))
    }

    pub fn reset(&mut self) {
        match &mut self.storage {
            KVCacheStorage::Standard { k, v } => {
                *k = None;
                *v = None;
            }
            KVCacheStorage::Quantized(q) => {
                q.reset();
            }
        }
        self.ssm_conv_state = None;
        self.ssm_recurrent_state = None;
    }

    /// Append key and value states along the sequence dimension (dim 2: [B, H, S, D]).
    /// If `sliding_window` is provided, automatically truncate the cache to keep at most `window` states.
    pub fn append(&mut self, new_k: &Tensor, new_v: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        self.append_with_window(new_k, new_v, None)
    }

    /// Append key and value states along the sequence dimension with optional sliding window bound.
    pub fn append_with_window(
        &mut self,
        new_k: &Tensor,
        new_v: &Tensor,
        sliding_window: Option<usize>,
    ) -> CandleResult<(Tensor, Tensor)> {
        match &mut self.storage {
            KVCacheStorage::Standard { k, v } => {
                let (mut next_k, mut next_v) = match (k.as_ref(), v.as_ref()) {
                    (Some(prev_k), Some(prev_v)) => {
                        let cat_k = Tensor::cat(&[prev_k, new_k], 2)?;
                        let cat_v = Tensor::cat(&[prev_v, new_v], 2)?;
                        (cat_k, cat_v)
                    }
                    _ => (new_k.clone(), new_v.clone()),
                };

                if let Some(w) = sliding_window.filter(|&w| w > 0) {
                    let total_seq = next_k.dim(2)?;
                    if total_seq > w {
                        let start = total_seq - w;
                        next_k = next_k.narrow(2, start, w)?;
                        next_v = next_v.narrow(2, start, w)?;
                    }
                }

                *k = Some(next_k.clone());
                *v = Some(next_v.clone());
                Ok((next_k, next_v))
            }
            KVCacheStorage::Quantized(q) => q.append_with_window(new_k, new_v, sliding_window),
        }
    }
}

/// Compact 8-bit quantized entry for a single key or value cache tensor.
/// Compresses [B, H, S, D] float tensors to [B, H, S, D] u8 with per-channel/per-token scales [B, H, S, 1].
/// Reduces resident cache memory by 75% compared to F32 and 50% compared to F16.
#[derive(Debug, Clone)]
pub struct QuantizedKVCacheEntry {
    /// Quantized uint8 data centered around 128
    pub data: Tensor,
    /// Per-token/channel scaling factor in F32
    pub scale: Tensor,
    /// Original sequence length
    pub seq_len: usize,
}

impl QuantizedKVCacheEntry {
    /// Quantize a [B, H, S, D] float tensor into symmetric uint8 representation
    pub fn quantize(x: &Tensor) -> CandleResult<Self> {
        let x_f32 = x.to_dtype(DType::F32)?;
        let dims = x.shape().dims();
        let seq_len = if dims.len() >= 3 { dims[2] } else { 0 };

        // Absolute maximum along the last dimension (dim -1: head_dim D)
        let abs_x = x_f32.abs()?;
        let max_val = abs_x.max_keepdim(candle_core::D::Minus1)?;
        let eps = 1e-8f64;
        let scale = ((max_val + eps)? * (1.0 / 127.0))?; // scale = max / 127.0

        // normalized = x / scale  => [-127, 127]
        let norm = x_f32.broadcast_div(&scale)?;
        // shifted = round(norm) + 128 => [1, 255] in U8
        let shifted = (norm + 128.0)?;
        let data = shifted.to_dtype(DType::U8)?;

        Ok(Self {
            data,
            scale,
            seq_len,
        })
    }

    /// Dequantize back to high precision target dtype
    pub fn dequantize(&self, target_dtype: DType) -> CandleResult<Tensor> {
        let shifted = self.data.to_dtype(DType::F32)?;
        let norm = (shifted - 128.0)?;
        let float_tensor = norm.broadcast_mul(&self.scale)?;
        float_tensor.to_dtype(target_dtype)
    }

    /// Truncate entry to maintain sliding window bound
    pub fn truncate_window(&self, window_size: usize) -> CandleResult<Self> {
        if self.seq_len <= window_size {
            return Ok(self.clone());
        }
        let start = self.seq_len - window_size;
        let data = self.data.narrow(2, start, window_size)?;
        let scale = self.scale.narrow(2, start, window_size)?;
        Ok(Self {
            data,
            scale,
            seq_len: window_size,
        })
    }

    /// Concatenate a newly generated step onto this quantized entry along sequence dimension
    pub fn concat_step(&self, step: &QuantizedKVCacheEntry) -> CandleResult<Self> {
        let data = Tensor::cat(&[&self.data, &step.data], 2)?;
        let scale = Tensor::cat(&[&self.scale, &step.scale], 2)?;
        Ok(Self {
            data,
            scale,
            seq_len: self.seq_len + step.seq_len,
        })
    }
}

/// Quantized Layer KV-Cache providing compact 8-bit memory storage with on-the-fly dequantization.
#[derive(Debug, Clone)]
pub struct QuantizedLayerKVCache {
    pub k: Option<QuantizedKVCacheEntry>,
    pub v: Option<QuantizedKVCacheEntry>,
}

impl QuantizedLayerKVCache {
    pub fn new() -> Self {
        Self { k: None, v: None }
    }

    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
    }

    /// Append key and value states, quantizing them to 8-bit storage and returning dequantized tensors.
    pub fn append_with_window(
        &mut self,
        new_k: &Tensor,
        new_v: &Tensor,
        sliding_window: Option<usize>,
    ) -> CandleResult<(Tensor, Tensor)> {
        let target_dtype = new_k.dtype();
        let step_k = QuantizedKVCacheEntry::quantize(new_k)?;
        let step_v = QuantizedKVCacheEntry::quantize(new_v)?;

        let mut q_k = match &self.k {
            Some(prev_k) => prev_k.concat_step(&step_k)?,
            None => step_k,
        };
        let mut q_v = match &self.v {
            Some(prev_v) => prev_v.concat_step(&step_v)?,
            None => step_v,
        };

        if let Some(w) = sliding_window.filter(|&w| w > 0) {
            q_k = q_k.truncate_window(w)?;
            q_v = q_v.truncate_window(w)?;
        }

        let deq_k = q_k.dequantize(target_dtype)?;
        let deq_v = q_v.dequantize(target_dtype)?;

        self.k = Some(q_k);
        self.v = Some(q_v);

        Ok((deq_k, deq_v))
    }
}

/// Global KV-Cache across all transformer layers.
pub struct KVCache {
    pub layers: Vec<LayerKVCache>,
}

impl KVCache {
    pub fn new(num_layers: usize) -> Self {
        let layers = (0..num_layers).map(|_| LayerKVCache::new()).collect();
        Self { layers }
    }

    pub fn new_quantized(num_layers: usize) -> Self {
        let layers = (0..num_layers).map(|_| LayerKVCache::new_quantized()).collect();
        Self { layers }
    }

    pub fn set_quantized(&mut self, quantized: bool) {
        for l in &mut self.layers {
            l.set_quantized(quantized);
        }
    }

    pub fn is_quantized(&self) -> bool {
        self.layers.first().map(|l| l.is_quantized()).unwrap_or(false)
    }

    pub fn reset(&mut self) {
        for l in &mut self.layers {
            l.reset();
        }
    }
}

/// CausalLM Model Configuration.
#[derive(Debug, Clone)]
pub struct CausalLMConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub head_dim: usize,
    pub is_gemma: bool,
    pub qk_norm: bool,
    pub sliding_window: Option<usize>,
    pub attn_logit_softcapping: Option<f64>,
    pub final_logit_softcapping: Option<f64>,
    pub unit_offset_norm: bool,
    pub sliding_window_pattern: Option<usize>,
    pub rope_scaling_factor: f32,
    pub rope_scaling_type: Option<String>,
    pub rope_scaling_orig_ctx: usize,
    pub yarn_beta_slow: f32,
    pub yarn_beta_fast: f32,
    pub moe_num_experts: usize,
    pub moe_top_k: usize,
    pub moe_shared_experts: usize,
    pub attention_scale: Option<f64>,
}

impl CausalLMConfig {
    /// Detect or extract configuration from weight keys and shapes.
    pub fn from_weights(src: &dyn WeightsSource) -> Result<Self> {
        let keys = src.keys();
        let arch = src.metadata("general.architecture").unwrap_or_default();
        let arch_prefix = if !arch.is_empty() { format!("{arch}.") } else { String::new() };

        let num_layers = src.metadata_usize(&format!("{arch_prefix}block_count"))
            .unwrap_or_else(|| {
                let cnt = count_causal_layers(&keys);
                if cnt == 0 { 28 } else { cnt }
            });

        // Hidden size & vocab size: prefer GGUF metadata, fallback to sniffing shapes
        let hidden_size = src.metadata_usize(&format!("{arch_prefix}embedding_length"))
            .unwrap_or_else(|| {
                if let Some((_, dims)) = src.raw_info("model.embed_tokens.weight")
                    .or_else(|| src.raw_info("embed_tokens.weight"))
                    .or_else(|| src.raw_info("token_embd.weight"))
                    .or_else(|| src.raw_info("lm_head.weight"))
                    .or_else(|| src.raw_info("output.weight")) {
                    if dims.len() >= 2 {
                        // In GGUF 2D shapes are [ne0, ne1], where ne0 is in_dim (hidden_size for token_embd)
                        // In SafeTensors 2D shapes are [vocab_size, hidden_size]
                        let min_dim = dims[0].min(dims[1]);
                        min_dim
                    } else {
                        4096
                    }
                } else {
                    4096
                }
            });

        let vocab_size = src.metadata_usize(&format!("{arch_prefix}vocab_size"))
            .unwrap_or_else(|| {
                if let Some((_, dims)) = src.raw_info("token_embd.weight")
                    .or_else(|| src.raw_info("model.embed_tokens.weight"))
                    .or_else(|| src.raw_info("embed_tokens.weight"))
                    .or_else(|| src.raw_info("lm_head.weight"))
                    .or_else(|| src.raw_info("output.weight")) {
                    if dims.len() >= 2 {
                        let max_dim = dims[0].max(dims[1]);
                        max_dim
                    } else {
                        151936
                    }
                } else {
                    151936
                }
            });

        // Intermediate size
        let intermediate_size = src.metadata_usize(&format!("{arch_prefix}feed_forward_length"))
            .unwrap_or_else(|| {
                if let Some((_, dims)) = src.raw_info("model.layers.0.mlp.gate_proj.weight")
                    .or_else(|| src.raw_info("layers.0.mlp.gate_proj.weight"))
                    .or_else(|| src.raw_info("blk.0.ffn_gate.weight")) {
                    dims.first().copied().unwrap_or(hidden_size * 4)
                } else if let Some((_, dims)) = src.raw_info("blk.0.ffn_up.weight") {
                    let d0 = dims.first().copied().unwrap_or(hidden_size * 4);
                    d0 / 2
                } else {
                    hidden_size * 4
                }
            });

        // Head count & KV head count
        let num_heads_meta = src.metadata_usize(&format!("{arch_prefix}attention.head_count"));
        let num_kv_heads_meta = src.metadata_usize(&format!("{arch_prefix}attention.head_count_kv"));
        let head_dim_meta = src.metadata_usize(&format!("{arch_prefix}attention.key_length"));

        let head_dim = head_dim_meta.unwrap_or_else(|| {
            keys.iter().find_map(|k| {
                if k.contains("attn_q_norm.weight") || k.contains("self_attn.q_norm.weight") {
                    src.raw_info(k).and_then(|(_, d)| d.first().copied())
                } else {
                    None
                }
            }).unwrap_or(128)
        });

        let (num_heads, num_kv_heads, head_dim) = if let (Some(nh), Some(nkv)) = (num_heads_meta, num_kv_heads_meta) {
            if nkv == 0 {
                // If metadata explicitly reports 0 kv heads (e.g. Nemotron-H), infer actual KV heads from blk.5.attn_k.weight
                let k_key = keys.iter().find(|k| k.ends_with("self_attn.k_proj.weight") || k.ends_with("attn_k.weight"));
                let actual_kv = k_key.and_then(|k| src.raw_info(k)).map(|(_, d)| {
                    let min_d = d[0].min(d[1]);
                    min_d / head_dim
                }).unwrap_or(2).max(1);
                (nh, actual_kv, head_dim)
            } else {
                (nh, nkv, head_dim)
            }
        } else if let Some(q_key) = keys.iter().find(|k| k.ends_with("self_attn.q_proj.weight") || k.ends_with("attn_q.weight")) {
            let q_out = src.raw_info(q_key).and_then(|(_, d)| d.first().copied()).unwrap_or(hidden_size);
            let k_key = q_key.replace(".q_proj.", ".k_proj.").replace(".attn_q.", ".attn_k.");
            let k_out = src.raw_info(&k_key).and_then(|(_, d)| d.first().copied()).unwrap_or(q_out);
            let actual_q_dim = if q_out > hidden_size && q_out % (head_dim * 2) == 0 {
                q_out / 2
            } else {
                q_out
            };
            (actual_q_dim / head_dim, k_out / head_dim, head_dim)
        } else if let Some((_, dims)) = src.raw_info("blk.0.attn_qkv.weight") {
            let total_out = dims.first().copied().unwrap_or(hidden_size);
            let total_heads = total_out / head_dim;
            let (num_h, num_kv) = if total_heads == 60 {
                (40, 10)
            } else if total_heads == 36 {
                (28, 4)
            } else if total_heads == 20 {
                (16, 2)
            } else if total_heads == 40 {
                (32, 4)
            } else {
                (total_heads.saturating_sub(4), 2)
            };
            (num_h, num_kv, head_dim)
        } else {
            (32, 8, head_dim)
        };

        let is_gemma = arch.starts_with("gemma")
            || keys.iter().any(|k| k.contains("gemma") || k.contains("pre_feedforward_layernorm"));
        let unit_offset_norm = is_gemma;

        let qk_norm = keys.iter().any(|k| k.contains("q_norm") || k.contains("attn_q_norm"));

        let rope_theta = src.metadata_f64(&format!("{arch_prefix}rope.freq_base"))
            .map(|f| f as f32)
            .unwrap_or_else(|| {
                if keys.iter().any(|k| k.contains("llama") || k.contains("deepseek")) || !qk_norm {
                    500_000.0f32
                } else {
                    1_000_000.0f32
                }
            });

        let rms_norm_eps = src.metadata_f64(&format!("{arch_prefix}attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-6);

        let sliding_window = src.metadata_usize(&format!("{arch_prefix}attention.sliding_window"))
            .filter(|&w| w > 0)
            .or_else(|| if is_gemma && !arch.contains("gemma3") { Some(4096) } else { None });

        let sliding_window_pattern = src.metadata_usize(&format!("{arch_prefix}attention.sliding_window_pattern"))
            .or_else(|| if is_gemma && arch.contains("gemma3") { Some(6) } else { None });

        let rope_scaling_factor = src.metadata_f64(&format!("{arch_prefix}rope.scaling.factor"))
            .map(|f| f as f32)
            .unwrap_or(1.0f32);

        let rope_scaling_type = src.metadata(&format!("{arch_prefix}rope.scaling.type"));
        let rope_scaling_orig_ctx = src.metadata_usize(&format!("{arch_prefix}rope.scaling.original_context_length"))
            .unwrap_or(16384);
        let yarn_beta_slow = src.metadata_f64(&format!("{arch_prefix}rope.scaling.yarn_beta_slow"))
            .map(|f| f as f32)
            .unwrap_or(1.0f32);
        let yarn_beta_fast = src.metadata_f64(&format!("{arch_prefix}rope.scaling.yarn_beta_fast"))
            .map(|f| f as f32)
            .unwrap_or(32.0f32);

        let attn_logit_softcapping = src.metadata_f64(&format!("{arch_prefix}attn_logit_softcapping"))
            .or_else(|| {
                // Gemma 2 hardcoded fallback if not in metadata, but Gemma 3 has no softcapping
                if is_gemma && !arch.starts_with("gemma3") && keys.iter().any(|k| k.contains("gemma_2")) {
                    Some(50.0)
                } else {
                    None
                }
            });

        let final_logit_softcapping = src.metadata_f64(&format!("{arch_prefix}final_logit_softcapping"))
            .or_else(|| {
                if is_gemma && !arch.starts_with("gemma3") && keys.iter().any(|k| k.contains("gemma_2")) {
                    Some(30.0)
                } else {
                    None
                }
            });

        let has_moe = keys.iter().any(|k| k.contains("mlp.experts") || k.contains("ffn_gate_exps"));
        let (moe_num_experts, moe_top_k, moe_shared_experts) = if has_moe {
            let num_exp = keys.iter().filter_map(|k| {
                if let Some(pos) = k.find("mlp.experts.") {
                    let rest = &k[pos + 12..];
                    rest.split('.').next().and_then(|s| s.parse::<usize>().ok())
                } else {
                    None
                }
            }).max().map(|m| m + 1).unwrap_or(32);
            let top_k = if keys.iter().any(|k| k.contains("shortconv")) { 4 } else { 2 };
            (num_exp, top_k, 2)
        } else {
            (0, 0, 0)
        };

        let attention_scale = src.metadata_f64(&format!("{arch_prefix}attention.scale"))
            .or_else(|| src.metadata_f64("attention.scale"));

        Ok(Self {
            hidden_size,
            intermediate_size,
            num_attention_heads: num_heads.max(1),
            num_key_value_heads: num_kv_heads.max(1),
            num_hidden_layers: num_layers,
            vocab_size,
            rms_norm_eps,
            rope_theta,
            max_position_embeddings: 8192,
            head_dim,
            is_gemma,
            qk_norm,
            sliding_window,
            attn_logit_softcapping,
            final_logit_softcapping,
            unit_offset_norm,
            sliding_window_pattern,
            rope_scaling_factor,
            rope_scaling_type,
            rope_scaling_orig_ctx,
            yarn_beta_slow,
            yarn_beta_fast,
            moe_num_experts,
            moe_top_k,
            moe_shared_experts,
            attention_scale,
        })
    }
}

fn count_causal_layers(keys: &[String]) -> usize {
    let mut max_l = 0;
    for k in keys {
        for p in &["model.layers.", "layers.", "blk."] {
            if let Some(rest) = k.strip_prefix(p) {
                if let Some(idx_str) = rest.split('.').next() {
                    if let Ok(idx) = idx_str.parse::<usize>() {
                        max_l = max_l.max(idx + 1);
                    }
                }
            }
        }
    }
    max_l
}

/// Generation Sampling Parameters
#[derive(Debug, Clone)]
pub struct TextGenParams {
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub stop_tokens: Vec<u32>,
    /// Optional number of transformer layers to offload onto GPU.
    /// When None, all layers reside on GPU. When Some(N), layers 0..N are placed on GPU
    /// and the remaining layers run on CPU, allowing large models (15B-27B) to fit on 8GB-12GB VRAM cards.
    pub n_gpu_layers: Option<usize>,
    /// When true, compresses key and value cache tensors to 8-bit quantized storage.
    pub quantized_kv: bool,
}

impl Default for TextGenParams {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repetition_penalty: 1.0,
            stop_tokens: vec![151643, 151645, 128001, 128009, 2, 1, 106], // Common EOS across Qwen, Llama3, Gemma (<end_of_turn> = 106)
            n_gpu_layers: None,
            quantized_kv: false,
        }
    }
}

/// Logits Processor for sampling
pub struct LogitsProcessor {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
    pub repetition_penalty: f32,
}

impl LogitsProcessor {
    pub fn new(temperature: f64, top_p: f64, top_k: usize, repetition_penalty: f32) -> Self {
        Self {
            temperature,
            top_p,
            top_k,
            repetition_penalty,
        }
    }

    pub fn sample(&self, logits: &Tensor, generated_tokens: &[u32]) -> Result<u32> {
        let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
        let mut logits_vec: Vec<f32> = logits.to_vec1().map_err(LuminaError::Candle)?;

        // Apply repetition penalty over the last 64 tokens if penalty > 1.0
        if self.repetition_penalty > 1.0 {
            let start = generated_tokens.len().saturating_sub(64);
            for &token in &generated_tokens[start..] {
                if let Some(logit) = logits_vec.get_mut(token as usize) {
                    if *logit > 0.0 {
                        *logit /= self.repetition_penalty;
                    } else {
                        *logit *= self.repetition_penalty;
                    }
                }
            }
        }

        // Greedy sampling if temperature <= 0
        if self.temperature <= 0.0 {
            let mut best_idx = 0;
            let mut best_val = f32::NEG_INFINITY;
            for (i, &v) in logits_vec.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best_idx = i;
                }
            }
            return Ok(best_idx as u32);
        }

        // Apply temperature
        let inv_temp = (1.0 / self.temperature) as f32;
        for v in logits_vec.iter_mut() {
            *v *= inv_temp;
        }

        // Softmax
        let max_val = logits_vec.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        for v in logits_vec.iter_mut() {
            *v = (*v - max_val).exp();
            sum_exp += *v;
        }
        for v in logits_vec.iter_mut() {
            *v /= sum_exp;
        }

        // Top-K / Top-P sampling
        let mut indexed_probs: Vec<(usize, f32)> = logits_vec.into_iter().enumerate().collect();
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if self.top_k > 0 && indexed_probs.len() > self.top_k {
            indexed_probs.truncate(self.top_k);
        }

        if self.top_p < 1.0 {
            let mut cumsum = 0.0f32;
            let mut cutoff = indexed_probs.len();
            for (i, (_, prob)) in indexed_probs.iter().enumerate() {
                cumsum += *prob;
                if cumsum >= self.top_p as f32 {
                    cutoff = i + 1;
                    break;
                }
            }
            indexed_probs.truncate(cutoff);
        }

        // Renormalize
        let norm_sum: f32 = indexed_probs.iter().map(|(_, p)| p).sum();
        let r = (rand_f32() * norm_sum).min(norm_sum);
        let mut running = 0.0f32;
        for (idx, prob) in &indexed_probs {
            running += *prob;
            if running >= r {
                return Ok(*idx as u32);
            }
        }

        Ok(indexed_probs.first().map(|(idx, _)| *idx as u32).unwrap_or(0))
    }
}

fn rand_f32() -> f32 {
    use std::cell::Cell;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn seed() -> u64 {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E37_79B9_7F4A_7C15);
        let stack = &nanos as *const u64 as u64;
        (nanos ^ stack.rotate_left(17)) | 1
    }

    thread_local! {
        static STATE: Cell<u64> = Cell::new(0);
    }

    STATE.with(|cell| {
        let mut x = cell.get();
        if x == 0 {
            x = seed();
        }
        // xorshift64*: fast, statistically sound for sampling (not cryptographic).
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        cell.set(x);
        let r = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((r >> 40) as f32) / (1u64 << 24) as f32
    })
}

/// The ROCm Candle backend only supports contiguous operands for `matmul`.
/// On every other backend this is a cheap no-op (Candle `Tensor` clones are Arc-based).
fn rocm_contiguous(t: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "rocm")]
    {
        if t.device().is_rocm() {
            return t.contiguous().map_err(LuminaError::Candle);
        }
    }
    Ok(t.clone())
}

/// Conversational message in multi-turn dialogues.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: ChatRole::System, content: content.into() }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: ChatRole::User, content: content.into() }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: ChatRole::Assistant, content: content.into() }
    }
}

/// Participant role in conversational dialogues.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

impl ChatRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

/// Alias for standalone CausalLM pipeline.
pub type CausalLM = CausalLMPipeline;

/// Standalone CausalLM Engine Pipeline
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PhiStyle {
    /// `<|user|>...<|end|><|assistant|>` (Phi-4 reasoning plus)
    EndTags,
    /// `<|im_start|>user<|im_sep|>...<|im_end|>` (classic Phi-3/4)
    ImSep,
}

fn phi_style(template: &str) -> PhiStyle {
    if template.contains("<|user|>") && template.contains("<|end|>") {
        PhiStyle::EndTags
    } else {
        PhiStyle::ImSep
    }
}

/// Extract the literal system preamble embedded in a Phi GGUF chat template.
fn phi_system_preamble(template: &str) -> Option<String> {
    let marker = "<|im_sep|>";
    let start = template.find(marker)? + marker.len();
    let end = template[start..].find("<|im_end|>")? + start;
    let preamble = template[start..end].trim();
    if preamble.is_empty() {
        None
    } else {
        Some(preamble.to_string())
    }
}

fn build_phi_chat(
    preamble: &str,
    system: &str,
    turns: &[(String, String)],
    style: PhiStyle,
) -> String {
    match style {
        PhiStyle::EndTags => {
            let mut out = format!("<|im_start|>system<|im_sep|>{preamble}");
            if !system.is_empty() {
                out.push('\n');
                out.push_str(system);
            }
            out.push_str("<|im_end|><|end|>");
            for (role, content) in turns {
                out.push_str(&format!("<|{role}|>{}<|end|>", content));
            }
            out.push_str("<|assistant|>");
            out
        }
        PhiStyle::ImSep => {
            let mut out = format!("<|im_start|>system<|im_sep|>{preamble}<|im_end|>");
            for (role, content) in turns {
                out.push_str(&format!(
                    "<|im_start|>{role}<|im_sep|>{}<|im_end|>",
                    content
                ));
            }
            out.push_str("<|im_start|>assistant<|im_sep|>");
            out
        }
    }
}

fn build_phi_prompt(preamble: &str, system: &str, user: &str, style: PhiStyle) -> String {
    build_phi_chat(
        preamble,
        system,
        &[("user".to_string(), user.to_string())],
        style,
    )
}

/// Remove Phi role/end markers and anything after them from a decoded stream.
fn trim_stop_markers(mut text: String) -> String {
    for marker in [
        "<|end|>",
        "<|im_end|>",
        "<|endoftext|>",
        "<|user|>",
        "<|assistant|>",
        "<|im_start|>",
    ] {
        if let Some(pos) = text.find(marker) {
            text.truncate(pos);
        }
    }
    text.trim().to_string()
}

pub struct CausalLMPipeline {
    pub config: CausalLMConfig,
    pub tokenizer: Option<Tokenizer>,
    pub weights: Arc<dyn WeightsSource>,
    pub device: Device,
    pub dtype: DType,
    pub kv_cache: KVCache,
    pub weights_cache: std::collections::HashMap<String, Tensor>,
    pub linear_cache: std::collections::HashMap<String, LinearLayer>,
    /// Number of transformer layers offloaded to GPU (None = all layers on GPU)
    pub n_gpu_layers: Option<usize>,
}

impl CausalLMPipeline {
    pub fn new(
        weights: Arc<dyn WeightsSource>,
        config: CausalLMConfig,
        tokenizer: Option<Tokenizer>,
        device: Device,
        dtype: DType,
    ) -> Self {
        let kv_cache = KVCache::new(config.num_hidden_layers);
        let weights_cache = std::collections::HashMap::new();
        let linear_cache = std::collections::HashMap::new();

        Self {
            config,
            tokenizer,
            weights,
            device,
            dtype,
            kv_cache,
            weights_cache,
            linear_cache,
            n_gpu_layers: None,
        }
    }

    /// Configure the number of layers offloaded to GPU (Layer Offloading)
    pub fn set_n_gpu_layers(&mut self, n: Option<usize>) {
        self.n_gpu_layers = n;
    }

    /// Enable or disable 8-bit quantized KV caching across all layers.
    pub fn enable_quantized_kv(&mut self, enabled: bool) {
        self.kv_cache.set_quantized(enabled);
    }

    /// Check whether 8-bit quantized KV caching is currently active.
    pub fn is_quantized_kv(&self) -> bool {
        self.kv_cache.is_quantized()
    }

    /// Format a multi-turn conversation using the model family's chat template.
    pub fn format_chat_prompt(&self, messages: &[ChatMessage]) -> String {
        let pairs: Vec<(String, String)> = messages
            .iter()
            .map(|m| (m.role.as_str().to_string(), m.content.clone()))
            .collect();
        self.format_chat_messages(&pairs)
    }

    /// Format a single user prompt for models that need a chat wrapper (Granite, Phi-3/4, Mistral).
    /// Prompts already containing format markers are passed through unchanged.
    pub fn format_prompt(&self, prompt: &str) -> String {
        if prompt.contains("<|im_sep|>") || prompt.contains("[INST]") || prompt.contains("<|im_start|>") {
            return prompt.to_string();
        }
        let arch = self.arch_name();
        if arch == "granite" {
            // Granite 4.2 thinking mode by default
            return format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n", prompt.trim());
        }
        if arch == "mistral3" || arch == "mistral" {
            return format!("[INST] {} [/INST]", prompt.trim());
        }
        if arch != "phi3" && arch != "phi4" {
            return prompt.to_string();
        }
        let template = self.weights.metadata("tokenizer.chat_template");
        let preamble = template
            .as_deref()
            .and_then(phi_system_preamble)
            .unwrap_or_else(|| "You are a helpful assistant.".to_string());
        let style = template.as_deref().map(phi_style).unwrap_or(PhiStyle::ImSep);
        build_phi_prompt(&preamble, "", prompt, style)
    }

    /// Format `(role, content)` messages using the model family's chat template.
    pub fn format_chat_messages(&self, messages: &[(String, String)]) -> String {
        let arch = self.arch_name();
        if arch == "granite" {
            let mut prompt = String::new();
            for (role, content) in messages {
                prompt.push_str(&format!("<|im_start|>{role}\n{}<|im_end|>\n", content.trim()));
            }
            prompt.push_str("<|im_start|>assistant\n<think>\n");
            return prompt;
        }
        if arch == "mistral3" || arch == "mistral" {
            let mut prompt = String::new();
            let mut sys_prompt = String::new();
            for (role, content) in messages {
                if role == "system" {
                    sys_prompt.push_str(content.trim());
                    sys_prompt.push('\n');
                }
            }
            if !sys_prompt.trim().is_empty() {
                prompt.push_str(&format!("[SYSTEM_PROMPT]{}[/SYSTEM_PROMPT]", sys_prompt.trim()));
            }
            for (role, content) in messages {
                if role == "user" {
                    prompt.push_str(&format!("[INST] {} [/INST]", content.trim()));
                } else if role == "assistant" {
                    prompt.push_str(&format!("{}</s>", content.trim()));
                }
            }
            return prompt;
        }
        if arch == "phi3" || arch == "phi4" {
            let template = self.weights.metadata("tokenizer.chat_template");
            let preamble = template
                .as_deref()
                .and_then(phi_system_preamble)
                .unwrap_or_else(|| "You are a helpful assistant.".to_string());
            let style = template.as_deref().map(phi_style).unwrap_or(PhiStyle::ImSep);
            let system = messages
                .iter()
                .filter(|(role, _)| role == "system")
                .map(|(_, content)| content.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            let turns: Vec<(String, String)> = messages
                .iter()
                .filter(|(role, _)| role != "system")
                .cloned()
                .collect();
            return build_phi_chat(&preamble, &system, &turns, style);
        }
        let mut prompt = String::new();
        for (role, content) in messages {
            prompt.push_str(&format!("<|im_start|>{role}\n{}<|im_end|>\n", content.trim()));
        }
        prompt.push_str("<|im_start|>assistant\n");
        prompt
    }

    fn arch_name(&self) -> String {
        self.weights
            .metadata("general.architecture")
            .unwrap_or_default()
            .to_ascii_lowercase()
    }

    /// Effective stop tokens: user-provided plus model-specific EOS ids.
    fn stop_tokens_for(&self, params: &TextGenParams) -> Vec<u32> {
        let mut stop = params.stop_tokens.clone();
        let arch = self.arch_name();
        if arch == "phi3" || arch == "phi4" {
            for id in [100257u32, 100265u32] {
                if !stop.contains(&id) {
                    stop.push(id);
                }
            }
        } else if arch == "mistral3" || arch == "mistral" {
            // Mistral EOS token (</s> = 2) and Tekken/Mistral-3 EOS variants
            for id in [2u32, 0u32] {
                if !stop.contains(&id) {
                    stop.push(id);
                }
            }
        } else if arch == "granite" {
            // Granite EOS token (<|im_end|> = 100257)
            if !stop.contains(&100257u32) {
                stop.push(100257u32);
            }
        }
        stop
    }

    /// Execute a multi-turn conversation turn and generate the assistant response.
    pub fn chat(&mut self, messages: &[ChatMessage], params: &TextGenParams) -> Result<String> {
        let prompt = self.format_chat_prompt(messages);
        self.generate(&prompt, params)
    }

    /// Autoregressive generation loop
    pub fn generate(&mut self, prompt: &str, params: &TextGenParams) -> Result<String> {
        self.generate_with_callback(prompt, params, None)
    }

    /// Autoregressive generation loop with live token-by-token streaming callback.
    ///
    /// The callback receives each newly decoded token substring (`&str`).
    /// If the callback returns `false`, generation terminates early cleanly.
    pub fn generate_with_callback(
        &mut self,
        prompt: &str,
        params: &TextGenParams,
        mut on_token: Option<crate::models::common::TokenCallback>,
    ) -> Result<String> {
        let prompt = self.format_prompt(prompt);
        let stop_tokens = self.stop_tokens_for(params);
        let prompt_tokens = {
            let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
                LuminaError::Config("Tokenizer not attached to CausalLM pipeline".into())
            })?;
            let encoding = tokenizer.encode(prompt.as_str(), true).map_err(|e| {
                LuminaError::Context {
                    context: "Failed to tokenize prompt".into(),
                    source: Box::new(LuminaError::Candle(candle_core::Error::Msg(e.to_string()))),
                }
            })?;
            encoding.get_ids().to_vec()
        };

        let mut prompt_tokens = prompt_tokens;
        let arch = self.arch_name();
        if (arch == "mistral3" || arch == "mistral") && prompt_tokens.first() != Some(&1) {
            prompt_tokens.insert(0, 1);
        }

        if prompt_tokens.is_empty() {
            return Ok(String::new());
        }

        if params.n_gpu_layers.is_some() {
            self.n_gpu_layers = params.n_gpu_layers;
        }

        self.enable_quantized_kv(params.quantized_kv);
        self.kv_cache.reset();
        let mut generated_tokens = Vec::new();
        // Reasoning Phi variants can loop under pure greedy decoding; keep a
        // conservative repetition penalty unless the caller asked for more.
        let repetition_penalty = {
            let arch = self.arch_name();
            if arch == "phi3" || arch == "phi4" || arch == "mistral3" || arch == "mistral" {
                params.repetition_penalty.max(1.1)
            } else {
                params.repetition_penalty
            }
        };
        let logits_processor = LogitsProcessor::new(
            params.temperature,
            params.top_p,
            params.top_k,
            repetition_penalty,
        );

        // Prefill prompt tokens
        println!("🔍 Prefill: prompt_tokens = {:?}", prompt_tokens);
        let input_tensor = Tensor::new(&prompt_tokens[..], &self.device)?.unsqueeze(0)?;
        let logits = self.forward(&input_tensor, 0)?;
        let last_logits = logits.i((0, prompt_tokens.len() - 1))?;

        let logits_vec = last_logits.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let nan_count = logits_vec.iter().filter(|v| v.is_nan()).count();
        let min_val = logits_vec.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!("🔍 Logits debug: min = {min_val:.4}, max = {max_val:.4}, nan_count = {nan_count}/{}", logits_vec.len());
        let mut top_indices: Vec<(usize, f32)> = logits_vec.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        top_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        println!("🔍 Top-5 logits at step 0:");
        for (idx, score) in top_indices.iter().take(5) {
            let piece = self.tokenizer.as_ref().and_then(|t| t.decode(&[*idx as u32], false).ok()).unwrap_or_default();
            println!("   - token {idx} ({:?}): {score:.4}", piece);
        }

        // Track all context tokens for repetition penalty
        let mut all_context_tokens = prompt_tokens.clone();

        let mut next_token = logits_processor.sample(&last_logits, &all_context_tokens)?;
        generated_tokens.push(next_token);
        all_context_tokens.push(next_token);

        // Stream initial token if callback provided
        if let Some(ref mut cb) = on_token {
            if let Some(tok_ref) = self.tokenizer.as_ref() {
                if let Ok(piece) = tok_ref.decode(&[next_token], false) {
                    if !cb(&piece) {
                        return Ok(piece);
                    }
                }
            }
        }

        let mut pos = prompt_tokens.len();

        // Autoregressive decoding loop
        for _ in 0..params.max_tokens {
            if stop_tokens.contains(&next_token) {
                break;
            }

            let next_input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let step_logits = self.forward(&next_input, pos)?;
            let step_last_logits = step_logits.i((0, 0))?;

            next_token = logits_processor.sample(&step_last_logits, &all_context_tokens)?;
            generated_tokens.push(next_token);
            all_context_tokens.push(next_token);

            let piece_str = self.tokenizer.as_ref().and_then(|t| t.decode(&[next_token], false).ok()).unwrap_or_default();
            eprintln!("✨ [Step {}] pos={} generated token {} ({:?})", generated_tokens.len(), pos, next_token, piece_str);

            if let Some(ref mut cb) = on_token {
                if let Some(tok_ref) = self.tokenizer.as_ref() {
                    if let Ok(piece) = tok_ref.decode(&[next_token], false) {
                        if !cb(&piece) {
                            break;
                        }
                    }
                }
            }

            pos += 1;
        }

        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            LuminaError::Config("Tokenizer not attached to CausalLM pipeline".into())
        })?;
        let output_text = tokenizer.decode(&generated_tokens, true).map_err(|e| {
            LuminaError::Context {
                context: "Failed to decode generated token stream".into(),
                source: Box::new(LuminaError::Candle(candle_core::Error::Msg(e.to_string()))),
            }
        })?;

        Ok(trim_stop_markers(output_text))
    }

    /// Speculative Decoding using a smaller Draft model and this instance as Target/Verifier.
    ///
    /// At each iteration, the draft model generates `gamma` candidate tokens rapidly.
    /// The target model (self) verifies them in a single batched/parallel forward pass,
    /// accepting matching tokens and correcting deviations.
    pub fn generate_speculative(
        &mut self,
        draft_model: &mut CausalLMPipeline,
        prompt: &str,
        params: &TextGenParams,
        gamma: usize,
    ) -> Result<String> {
        let gamma = gamma.max(1);
        let prompt = self.format_prompt(prompt);
        let prompt_tokens = {
            let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
                LuminaError::Config("Tokenizer not attached to Target CausalLM pipeline".into())
            })?;
            let encoding = tokenizer.encode(prompt.as_str(), true).map_err(|e| {
                LuminaError::Context {
                    context: "Failed to tokenize prompt".into(),
                    source: Box::new(LuminaError::Candle(candle_core::Error::Msg(e.to_string()))),
                }
            })?;
            encoding.get_ids().to_vec()
        };

        if prompt_tokens.is_empty() {
            return Ok(String::new());
        }

        self.kv_cache.reset();
        draft_model.kv_cache.reset();

        let mut generated_tokens = Vec::new();
        let target_processor = LogitsProcessor::new(
            params.temperature,
            params.top_p,
            params.top_k,
            params.repetition_penalty,
        );

        // 1. Prefill target model
        let target_input = Tensor::new(&prompt_tokens[..], &self.device)?.unsqueeze(0)?;
        let target_logits = self.forward(&target_input, 0)?;
        let last_target_logits = target_logits.i((0, prompt_tokens.len() - 1))?;
        let mut next_token = target_processor.sample(&last_target_logits, &generated_tokens)?;
        generated_tokens.push(next_token);

        // Prefill draft model with initial prompt + initial token
        let mut draft_prompt = prompt_tokens.clone();
        draft_prompt.push(next_token);
        let draft_input = Tensor::new(&draft_prompt[..], &draft_model.device)?.unsqueeze(0)?;
        let _ = draft_model.forward(&draft_input, 0)?;

        let mut target_pos = prompt_tokens.len();
        let mut draft_pos = draft_prompt.len();

        let _draft_params = TextGenParams {
            max_tokens: gamma,
            temperature: params.temperature,
            top_p: params.top_p,
            top_k: params.top_k,
            repetition_penalty: params.repetition_penalty,
            stop_tokens: params.stop_tokens.clone(),
            n_gpu_layers: None,
            quantized_kv: params.quantized_kv,
        };

        let stop_tokens = self.stop_tokens_for(params);
        while generated_tokens.len() < params.max_tokens {
            if stop_tokens.contains(&next_token) {
                break;
            }

            // 2. Draft model speculates `gamma` draft tokens
            let mut draft_tokens = Vec::with_capacity(gamma);
            let mut curr_draft_tok = next_token;

            for _ in 0..gamma {
                let step_in = Tensor::new(&[curr_draft_tok], &draft_model.device)?.unsqueeze(0)?;
                let step_logits = draft_model.forward(&step_in, draft_pos)?;
                let last_logits = step_logits.i((0, 0))?;
                let tok = target_processor.sample(&last_logits, &generated_tokens)?;
                draft_tokens.push(tok);
                curr_draft_tok = tok;
                draft_pos += 1;
                if stop_tokens.contains(&tok) {
                    break;
                }
            }

            if draft_tokens.is_empty() {
                break;
            }

            // 3. Target model verifies draft tokens sequentially or in batch
            let mut accepted_all = true;
            for &c_tok in &draft_tokens {
                let next_input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
                let step_logits = self.forward(&next_input, target_pos)?;
                let step_last_logits = step_logits.i((0, 0))?;
                let target_sampled = target_processor.sample(&step_last_logits, &generated_tokens)?;

                target_pos += 1;
                if target_sampled == c_tok {
                    generated_tokens.push(c_tok);
                    next_token = c_tok;
                    if stop_tokens.contains(&c_tok) || generated_tokens.len() >= params.max_tokens {
                        accepted_all = false;
                        break;
                    }
                } else {
                    // Deviation detected: accept target correction, discard remaining draft tokens
                    generated_tokens.push(target_sampled);
                    next_token = target_sampled;
                    accepted_all = false;
                    break;
                }
            }

            if !accepted_all {
                // Resynchronize draft model position with actual accepted tokens
                let total_tokens_now = prompt_tokens.len() + generated_tokens.len();
                draft_pos = total_tokens_now;
            }
        }

        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            LuminaError::Config("Tokenizer not attached to Target CausalLM pipeline".into())
        })?;
        let output_text = tokenizer.decode(&generated_tokens, true).map_err(|e| {
            LuminaError::Context {
                context: "Failed to decode speculative token stream".into(),
                source: Box::new(LuminaError::Candle(candle_core::Error::Msg(e.to_string()))),
            }
        })?;

        Ok(trim_stop_markers(output_text))
    }

    /// Autoregressive generation restricted to an optional token set (e.g. ACE-Step
    /// `<|audio_code_N|>` tokens). Returns the generated token ids (prompt excluded).
    pub fn generate_ids(
        &mut self,
        prompt: &str,
        max_new: usize,
        temperature: f64,
        top_p: f64,
        allowed: Option<&std::collections::HashSet<u32>>,
        seed: u64,
    ) -> Result<Vec<u32>> {
        let prompt_tokens = {
            let tokenizer = self
                .tokenizer
                .as_ref()
                .ok_or_else(|| LuminaError::Config("Tokenizer not attached".into()))?;
            tokenizer
                .encode(prompt, true)
                .map_err(|e| LuminaError::Candle(candle_core::Error::Msg(e.to_string())))?
                .get_ids()
                .to_vec()
        };
        if prompt_tokens.is_empty() {
            return Ok(Vec::new());
        }

        self.kv_cache.reset();
        let mut rng = crate::audio::rng::SeededRng::new(seed);
        let input = Tensor::new(&prompt_tokens[..], &self.device)
            .map_err(LuminaError::Candle)?
            .unsqueeze(0)
            .map_err(LuminaError::Candle)?;
        let mut last = self
            .forward(&input, 0)?
            .i((0, prompt_tokens.len() - 1))
            .map_err(LuminaError::Candle)?;
        let mut pos = prompt_tokens.len();
        let mut out = Vec::new();

        loop {
            let tok = Self::sample_restricted(&last, temperature, top_p, allowed, &mut rng)?;
            if let Some(set) = allowed {
                if !set.contains(&tok) {
                    break;
                }
            }
            if tok == 151645 || tok == 151643 {
                break;
            }
            out.push(tok);
            if out.len() >= max_new {
                break;
            }
            let next = Tensor::new(&[tok], &self.device)
                .map_err(LuminaError::Candle)?
                .unsqueeze(0)
                .map_err(LuminaError::Candle)?;
            last = self
                .forward(&next, pos)?
                .i((0, 0))
                .map_err(LuminaError::Candle)?;
            pos += 1;
        }
        Ok(out)
    }

    fn sample_restricted(
        logits: &Tensor,
        temperature: f64,
        top_p: f64,
        allowed: Option<&std::collections::HashSet<u32>>,
        rng: &mut crate::audio::rng::SeededRng,
    ) -> Result<u32> {
        let mut v: Vec<f32> = logits
            .to_dtype(DType::F32)
            .map_err(LuminaError::Candle)?
            .to_vec1()
            .map_err(LuminaError::Candle)?;
        if let Some(set) = allowed {
            for (i, x) in v.iter_mut().enumerate() {
                if !set.contains(&(i as u32)) {
                    *x = f32::NEG_INFINITY;
                }
            }
        }
        if temperature <= 0.0 {
            let mut best = (0usize, f32::NEG_INFINITY);
            for (i, &x) in v.iter().enumerate() {
                if x.is_finite() && x > best.1 {
                    best = (i, x);
                }
            }
            return Ok(best.0 as u32);
        }
        let maxv = v
            .iter()
            .cloned()
            .filter(|x| x.is_finite())
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for x in v.iter_mut() {
            if x.is_finite() {
                *x = (((*x - maxv) as f64) / temperature).exp() as f32;
                sum += *x;
            } else {
                *x = 0.0;
            }
        }
        if sum <= 0.0 {
            return Ok(0);
        }
        // Top-P (nucleus) filtering to keep the code sequence coherent.
        if top_p < 1.0 {
            let mut order: Vec<usize> = (0..v.len()).collect();
            order.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap_or(std::cmp::Ordering::Equal));
            let mut cum = 0.0f32;
            let mut kept = 0usize;
            for &i in &order {
                if v[i] > 0.0 {
                    cum += v[i] / sum;
                    kept += 1;
                    if cum >= top_p as f32 {
                        break;
                    }
                }
            }
            let keep: std::collections::HashSet<usize> = order.into_iter().take(kept).collect();
            let mut new_sum = 0.0f32;
            for (i, x) in v.iter_mut().enumerate() {
                if !keep.contains(&i) {
                    *x = 0.0;
                } else {
                    new_sum += *x;
                }
            }
            if new_sum > 0.0 {
                sum = new_sum;
            }
        }
        let r = rng.next_f32() * sum;
        let mut acc = 0.0f32;
        for (i, &x) in v.iter().enumerate() {
            acc += x;
            if acc >= r {
                return Ok(i as u32);
            }
        }
        Ok(0)
    }

    /// Universal RMSNorm supporting CPU, CUDA and ROCm
    pub fn apply_rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
        Self::apply_rms_norm_configured(x, weight, eps, false)
    }

    /// Instance norm checking model config (e.g. Gemma unit offset)
    fn apply_norm(&self, x: &Tensor, weight: &Tensor) -> Result<Tensor> {
        Self::apply_rms_norm_configured(x, weight, self.config.rms_norm_eps, self.config.unit_offset_norm)
    }

    /// Explicit configured RMSNorm
    pub fn apply_rms_norm_configured(x: &Tensor, weight: &Tensor, eps: f64, unit_offset: bool) -> Result<Tensor> {
        let orig_dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let sq = x_f32.sqr()?;
        let last_dim = sq.dims().len() - 1;
        let mean = sq.mean_keepdim(last_dim)?;
        let rms = (mean + eps)?.sqrt()?;
        let norm = x_f32.broadcast_div(&rms)?;

        let mut target_shape = vec![1; x.dims().len()];
        target_shape[last_dim] = weight.dim(0)?;
        let w = weight.to_dtype(DType::F32)?.reshape(target_shape.as_slice())?;
        let final_w = if unit_offset {
            (w + 1.0)?
        } else {
            w
        };
        let out = norm.broadcast_mul(&final_w)?.to_dtype(orig_dtype)?;
        Ok(out)
    }

    /// Forward pass through the Transformer
    pub fn forward(&mut self, input_ids: &Tensor, pos: usize) -> Result<Tensor> {
        let (_b_size, seq_len) = input_ids.shape().dims2().map_err(LuminaError::Candle)?;

        // 1. Embedding lookup
        let embed_w = self.get_weight_or_fallback(&["model.embed_tokens.weight", "token_embd.weight", "embed_tokens.weight"])?;
        let mut hidden = Embedding::new(embed_w, self.config.hidden_size).forward(input_ids).map_err(LuminaError::Candle)?;

        if self.config.is_gemma {
            let normalizer = (self.config.hidden_size as f64).sqrt();
            hidden = (hidden * normalizer).map_err(LuminaError::Candle)?;
        }

        if pos == 0 {
            let h_flat = hidden.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            let min_h = h_flat.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_h = h_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            println!("🔍 [Step 0] Hidden after embed: min={min_h:.4}, max={max_h:.4}");
        }

        // 2. Transformer layers loop with KV-Cache and Dynamic Layer Offloading
        for layer_idx in 0..self.config.num_hidden_layers {
            let layer_dev = self.device_for_layer(layer_idx);
            if !hidden.device().same_device(&layer_dev) {
                hidden = hidden.to_device(&layer_dev).map_err(LuminaError::Candle)?;
            }
            if pos == 0 {
                eprintln!("🔍 [Step 0] Entering layer {layer_idx} (hidden shape: {:?})", hidden.shape());
            }
            hidden = self.forward_layer(layer_idx, &hidden, pos, seq_len)?;

            // Memory hygiene: evict heavy transient layer weights from cache to prevent VRAM OOM on multi-layer / MoE models
            let prefix_blk = format!("blk.{layer_idx}.");
            let prefix_model = format!("model.layers.{layer_idx}.");
            let prefix_layers = format!("layers.{layer_idx}.");
            self.weights_cache.retain(|k, _| !k.starts_with(&prefix_blk) && !k.starts_with(&prefix_model) && !k.starts_with(&prefix_layers));
            self.linear_cache.retain(|k, _| !k.starts_with(&prefix_blk) && !k.starts_with(&prefix_model) && !k.starts_with(&prefix_layers));

            if pos == 0 && (layer_idx == 0 || layer_idx == 1 || layer_idx == self.config.num_hidden_layers - 1) {
                let h_flat = hidden.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                let min_h = h_flat.iter().cloned().fold(f32::INFINITY, f32::min);
                let max_h = h_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                println!("🔍 [Step 0] Hidden after layer {layer_idx}: min={min_h:.4}, max={max_h:.4}");
            }
        }

        // Transfer hidden back to primary pipeline device for final norm and LM head if offloaded
        if !hidden.device().same_device(&self.device) {
            hidden = hidden.to_device(&self.device).map_err(LuminaError::Candle)?;
        }

        // 3. Final layer norm
        let norm_w = self.get_weight_or_fallback(&["model.norm.weight", "output_norm.weight", "norm.weight", "token_embd_norm.weight"])?;
        let hidden = self.apply_norm(&hidden, &norm_w)?;

        if pos == 0 {
            let h_flat = hidden.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            let min_h = h_flat.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_h = h_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            println!("🔍 [Step 0] Hidden after final norm: min={min_h:.4}, max={max_h:.4}");
        }

        // 4. LM Head projection
        let lm_head_l = self.get_linear_or_fallback(&["lm_head.weight", "output.weight", "model.embed_tokens.weight", "embed_tokens.weight", "token_embd.weight"])?;
        let mut logits = lm_head_l.forward(&hidden)?;

        // Google Gemma final logit soft-capping
        if let Some(cap) = self.config.final_logit_softcapping {
            if cap > 0.0 {
                let inv_cap = 1.0 / cap;
                let scaled = (&logits * inv_cap).map_err(LuminaError::Candle)?;
                let tanh_logits = scaled.tanh().map_err(LuminaError::Candle)?;
                logits = (tanh_logits * cap).map_err(LuminaError::Candle)?;
            }
        }

        Ok(logits)
    }

    fn forward_layer(&mut self, layer_idx: usize, x: &Tensor, pos: usize, seq_len: usize) -> Result<Tensor> {
        // Input layernorm
        let in_norm_w = self.get_layer_weight(layer_idx, &["input_layernorm.weight", "attn_norm.weight"])?;
        let normed_x = self.apply_norm(x, &in_norm_w)?;

        // Check if this layer is an LFM ShortConv layer (Liquid AI LFM2 / LFM2.5)
        let is_shortconv_layer = self.weights.contains(&format!("blk.{layer_idx}.shortconv.conv.weight"))
            || self.weights.contains(&format!("model.layers.{layer_idx}.conv.conv.weight"))
            || self.weights.contains(&format!("layers.{layer_idx}.conv.conv.weight"))
            || self.weights_cache.contains_key(&format!("blk.{layer_idx}.shortconv.conv.weight"));

        // Check if this layer is an SSM / Linear Attention layer (e.g. Qwen 3.5 / 3.8 hybrid architecture)
        let is_ssm_layer = self.weights.contains(&format!("blk.{layer_idx}.ssm_a"))
            || self.weights.contains(&format!("model.layers.{layer_idx}.ssm_a"))
            || self.weights.contains(&format!("layers.{layer_idx}.ssm_a"))
            || self.weights_cache.contains_key(&format!("blk.{layer_idx}.ssm_a"));

        let is_nemotron = self.weights.contains(&"blk.0.ssm_a".to_string())
            && (self.weights.contains(&"blk.1.ffn_gate_inp.weight".to_string()) || self.weights.contains(&"blk.1.ffn_up.weight".to_string()));

        let (_attn_out, x) = if is_shortconv_layer {
            // LFM ShortConv double-gated 1D convolution block:
            // 1. in_proj: [hidden_dim, 3 * hidden_dim] -> BCx is [B, S, 3 * hidden_dim]
            let in_proj_w = self.get_layer_weight(layer_idx, &["shortconv.in_proj.weight", "conv.in_proj.weight", "in_proj.weight"])?;
            let bcx = Self::matmul_linear(&normed_x, &in_proj_w)?;
            let hidden_dim = normed_x.dim(candle_core::D::Minus1)?;
            let b_gate = bcx.narrow(candle_core::D::Minus1, 0, hidden_dim)?;
            let c_gate = bcx.narrow(candle_core::D::Minus1, hidden_dim, hidden_dim)?;
            let x_feat = bcx.narrow(candle_core::D::Minus1, 2 * hidden_dim, hidden_dim)?;

            // 2. Gating: B * x
            let bx = (&b_gate * &x_feat).map_err(LuminaError::Candle)?;

            // 3. Causal Depthwise 1D Convolution with kernel size K (e.g. K=3)
            let conv_w = self.get_layer_weight(layer_idx, &["shortconv.conv.weight", "conv.conv.weight", "conv.weight"])?;
            // conv_w shape: [hidden_dim, K]
            let (c_dim, k_sz) = conv_w.shape().dims2().map_err(LuminaError::Candle)?;
            let state_len = k_sz.saturating_sub(1);

            let conv_out = if seq_len == 1 {
                // Autoregressive token step: rolling buffer of size state_len
                let prev_state = self.kv_cache.layers[layer_idx].ssm_conv_state.clone().unwrap_or_else(|| {
                    Tensor::zeros((1, c_dim, state_len), bx.dtype(), &self.device).unwrap()
                });
                let x_step = bx.transpose(1, 2)?; // [1, c_dim, 1]
                let full_window = Tensor::cat(&[&prev_state, &x_step], 2)?; // [1, c_dim, K] (oldest to newest: index 0 is oldest, K-1 is newest x_step)
                let next_buf = full_window.narrow(2, 1, state_len)?;
                self.kv_cache.layers[layer_idx].ssm_conv_state = Some(next_buf);

                // full_window is [1, c_dim, K] where index K-1 is current token, K-2 is t-1, K-3 is t-2
                // conv_w in llama.cpp / ggml is indexed such that:
                // conv_w[:, 0] is t-2, conv_w[:, 1] is t-1, conv_w[:, 2] is t (or vice versa)
                // In ggml_ssm_conv: s = src0 + i2*nb0 (starts at window start); s[i0] * c[i0]
                // Since full_window ordered oldest-to-newest matches ggml concat(conv, bx), direct elementwise multiply applies!
                let conv_w_b = conv_w.unsqueeze(0)?; // [1, c_dim, K]
                (full_window * conv_w_b)?.sum_keepdim(2)?.transpose(1, 2)? // [1, 1, c_dim]
            } else {
                // Prefill phase: update historical state buffer
                let last_steps = if seq_len >= state_len {
                    bx.narrow(1, seq_len - state_len, state_len)?.transpose(1, 2)?
                } else {
                    let pad = Tensor::zeros((1, c_dim, state_len - seq_len), bx.dtype(), bx.device())?;
                    let cur = bx.transpose(1, 2)?;
                    Tensor::cat(&[&pad, &cur], 2)?
                };
                self.kv_cache.layers[layer_idx].ssm_conv_state = Some(last_steps);

                // Full causal 1D convolution over sequence
                let bx_trans = bx.transpose(1, 2)?; // [B, c_dim, S]
                let padded_bx = if state_len > 0 {
                    let pad = Tensor::zeros((bx.dim(0)?, c_dim, state_len), bx.dtype(), bx.device())?;
                    Tensor::cat(&[&pad, &bx_trans], 2)?
                } else {
                    bx_trans
                };

                let mut out_cols = Vec::with_capacity(seq_len);
                for t in 0..seq_len {
                    let window = padded_bx.narrow(2, t, k_sz)?; // [B, c_dim, K]
                    let conv_w_b = conv_w.unsqueeze(0)?; // [1, c_dim, K]
                    let step = (window * conv_w_b)?.sum_keepdim(2)?; // [B, c_dim, 1]
                    out_cols.push(step);
                }
                let cols_ref: Vec<&Tensor> = out_cols.iter().collect();
                Tensor::cat(&cols_ref, 2)?.transpose(1, 2)? // [B, S, c_dim]
            };

            // 4. Output Modulation: C * conv_out
            let modulated = (&c_gate * &conv_out).map_err(LuminaError::Candle)?;

            // 5. Output projection: out_proj(y)
            let out_proj_w = self.get_layer_weight(layer_idx, &["shortconv.out_proj.weight", "conv.out_proj.weight", "out_proj.weight"])?;
            let out = Self::matmul_linear(&modulated, &out_proj_w)?;
            let x_res = (x + &out).map_err(LuminaError::Candle)?;
            (out, x_res)
        } else if is_ssm_layer {
            // SSM / Linear Attention / Conv1D DeltaNet forward pass:
            // 1. in_proj (attn_qkv): [10240, 5120] -> projected is [B, S, 10240]
            let in_proj_w = self.get_layer_weight(layer_idx, &["attn_qkv.weight", "ssm_in.weight", "self_attn.qkv_proj.weight", "ssm_proj.weight"])?;
            let projected = Self::matmul_linear(&normed_x, &in_proj_w)?;
            
            // 2. Conv1D over features with kernel size 4
            let conv_act = if let Ok(conv_w) = self.get_layer_weight(layer_idx, &["ssm_conv1d.weight", "conv1d.weight"]) {
                let (d0, d1) = conv_w.shape().dims2().map_err(LuminaError::Candle)?;
                let (k_sz, c_dim) = if d0 <= 16 && d1 > d0 {
                    (d0, d1)
                } else if d1 <= 16 && d0 > d1 {
                    (d1, d0)
                } else {
                    (d0, d1)
                };
                let state_len = k_sz.saturating_sub(1);
                let conv_w_adapted = if conv_w.dim(0)? == k_sz {
                    conv_w.transpose(0, 1)? // [c_dim, k_sz]
                } else {
                    conv_w.clone() // already [c_dim, k_sz]
                };
                let conv_out = if seq_len == 1 {
                    // Step generation with rolling conv buffer
                    let prev_state = self.kv_cache.layers[layer_idx].ssm_conv_state.clone().unwrap_or_else(|| {
                        Tensor::zeros((1, c_dim, state_len), projected.dtype(), projected.device()).unwrap()
                    });
                    let x_conv_in = if projected.dim(candle_core::D::Minus1)? > c_dim {
                        projected.narrow(candle_core::D::Minus1, 0, c_dim)?
                    } else {
                        projected.clone()
                    };
                    let x_step = x_conv_in.transpose(1, 2)?; // [1, c_dim, 1]
                    let full_window = Tensor::cat(&[&prev_state, &x_step], 2)?; // [1, c_dim, k_sz]
                    let next_buf = full_window.narrow(2, 1, state_len)?;
                    self.kv_cache.layers[layer_idx].ssm_conv_state = Some(next_buf);

                    let conv_w_b = conv_w_adapted.unsqueeze(0)?; // [1, c_dim, k_sz]
                    let mut c_out = (full_window * conv_w_b)?.sum_keepdim(2)?.transpose(1, 2)?; // [1, 1, c_dim]
                    if let Ok(conv_b) = self.get_layer_weight(layer_idx, &["ssm_conv1d.bias", "conv1d.bias"]) {
                        c_out = c_out.broadcast_add(&conv_b.unsqueeze(0)?.unsqueeze(1)?)?;
                    }
                    c_out
                } else {
                    // Prefill phase: causal depthwise convolution over full sequence
                    let x_conv_in = if projected.dim(candle_core::D::Minus1)? > c_dim {
                        projected.narrow(candle_core::D::Minus1, 0, c_dim)?
                    } else {
                        projected.clone()
                    };
                    let last_steps = if seq_len >= state_len {
                        x_conv_in.narrow(1, seq_len - state_len, state_len)?.transpose(1, 2)?
                    } else {
                        let pad = Tensor::zeros((1, c_dim, state_len - seq_len), x_conv_in.dtype(), x_conv_in.device())?;
                        let cur = x_conv_in.transpose(1, 2)?;
                        Tensor::cat(&[&pad, &cur], 2)?
                    };
                    self.kv_cache.layers[layer_idx].ssm_conv_state = Some(last_steps);

                    let bx_trans = x_conv_in.transpose(1, 2)?; // [B, c_dim, S]
                    let padded_bx = if state_len > 0 {
                        let pad = Tensor::zeros((x_conv_in.dim(0)?, c_dim, state_len), x_conv_in.dtype(), x_conv_in.device())?;
                        Tensor::cat(&[&pad, &bx_trans], 2)?
                    } else {
                        bx_trans
                    };

                    let conv_w_b = conv_w_adapted.unsqueeze(0)?; // [1, c_dim, k_sz]
                    let mut out_cols = Vec::with_capacity(seq_len);
                    for t in 0..seq_len {
                        let window = padded_bx.narrow(2, t, k_sz)?; // [B, c_dim, k_sz]
                        let step = (window * &conv_w_b)?.sum_keepdim(2)?; // [B, c_dim, 1]
                        out_cols.push(step);
                    }
                    let cols_ref: Vec<&Tensor> = out_cols.iter().collect();
                    let mut c_out = Tensor::cat(&cols_ref, 2)?.transpose(1, 2)?; // [B, S, c_dim]
                    if let Ok(conv_b) = self.get_layer_weight(layer_idx, &["ssm_conv1d.bias", "conv1d.bias"]) {
                        c_out = c_out.broadcast_add(&conv_b.unsqueeze(0)?.unsqueeze(1)?)?;
                    }
                    c_out
                };
                candle_nn::ops::silu(&conv_out).map_err(LuminaError::Candle)?
            } else {
                candle_nn::ops::silu(&projected).map_err(LuminaError::Candle)?
            };

            // 3. Alpha & Beta gates projection: [48, 5120] -> [B, S, 48]
            let gated_act = if let (Ok(alpha_w), Ok(beta_w)) = (
                self.get_layer_weight(layer_idx, &["ssm_alpha.weight", "alpha.weight"]),
                self.get_layer_weight(layer_idx, &["ssm_beta.weight", "beta.weight"]),
            ) {
                let alpha_logits = Self::matmul_linear(&normed_x, &alpha_w)?;
                let beta_logits = Self::matmul_linear(&normed_x, &beta_w)?;

                let fast_sigmoid = |t: &Tensor| -> Result<Tensor> {
                    let neg_t = (t * -1.0).map_err(LuminaError::Candle)?;
                    let exp_neg = neg_t.exp().map_err(LuminaError::Candle)?;
                    let one_plus_exp = (exp_neg + 1.0).map_err(LuminaError::Candle)?;
                    let ones = Tensor::ones_like(&one_plus_exp).map_err(LuminaError::Candle)?;
                    ones.broadcast_div(&one_plus_exp).map_err(LuminaError::Candle)
                };

                let alpha = fast_sigmoid(&alpha_logits)?;
                let beta = fast_sigmoid(&beta_logits)?;

                let ssm_dim = 6144;
                let feat_slice = if conv_act.dim(candle_core::D::Minus1)? >= ssm_dim {
                    conv_act.narrow(candle_core::D::Minus1, 0, ssm_dim)?
                } else {
                    conv_act.clone()
                };

                // SSM Recurrent update
                let mut curr_state = self.kv_cache.layers[layer_idx].ssm_recurrent_state.clone().unwrap_or_else(|| {
                    Tensor::zeros((1, 48, 128, 128), feat_slice.dtype(), feat_slice.device()).unwrap()
                });

                if seq_len == 1 {
                    let a_t = alpha.squeeze(1)?.unsqueeze(2)?.unsqueeze(3)?; // [1, 48, 1, 1]
                    let b_t = beta.squeeze(1)?.unsqueeze(2)?.unsqueeze(3)?;  // [1, 48, 1, 1]
                    let x_grouped = feat_slice.squeeze(1)?.reshape((1, 48, 128, 1))?;
                    let term1 = curr_state.broadcast_mul(&a_t).map_err(LuminaError::Candle)?;
                    let term2 = x_grouped.broadcast_mul(&b_t).map_err(LuminaError::Candle)?;
                    curr_state = term1.broadcast_add(&term2).map_err(LuminaError::Candle)?;
                    self.kv_cache.layers[layer_idx].ssm_recurrent_state = Some(curr_state.clone());
                }

                if let Ok(ssm_norm_w) = self.get_layer_weight(layer_idx, &["ssm_norm.weight", "norm.weight"]) {
                    let b_sz = feat_slice.dim(0)?;
                    let feat_grouped = feat_slice.reshape((b_sz, seq_len, 48, 128))?;
                    let normed = Self::apply_rms_norm(&feat_grouped, &ssm_norm_w, self.config.rms_norm_eps)?;
                    normed.reshape((b_sz, seq_len, 6144))?
                } else {
                    feat_slice
                }
            } else {
                let out_proj_w = self.get_layer_weight(layer_idx, &["ssm_out.weight", "self_attn.o_proj.weight", "attn_output.weight", "attn_out.weight"])?;
                let ssm_dim = match out_proj_w.shape().dims2() {
                    Ok((_out_d, in_d)) => in_d,
                    _ => 4096,
                };
                if conv_act.dim(candle_core::D::Minus1)? >= ssm_dim {
                    conv_act.narrow(candle_core::D::Minus1, 0, ssm_dim)?
                } else {
                    conv_act
                }
            };

            // 4. Gate modulation if attn_gate present
            let modulated = if let Ok(gate_w) = self.get_layer_weight(layer_idx, &["attn_gate.weight", "ssm_gate.weight"]) {
                let gate = Self::matmul_linear(&normed_x, &gate_w)?;
                let silu_gate = candle_nn::ops::silu(&gate).map_err(LuminaError::Candle)?;
                (gated_act * silu_gate).map_err(LuminaError::Candle)?
            } else {
                gated_act
            };

            // 5. Output projection: project back to model hidden dimension
            let out_proj_w = self.get_layer_weight(layer_idx, &["ssm_out.weight", "self_attn.o_proj.weight", "attn_output.weight", "attn_out.weight"])?;
            let out = Self::matmul_linear(&modulated, &out_proj_w)?;
            let x_res = (x + &out).map_err(LuminaError::Candle)?;
            (out, x_res)
        } else if self.weights.contains(&format!("blk.{layer_idx}.ffn_gate_inp.weight"))
            || self.weights_cache.contains_key(&format!("blk.{layer_idx}.ffn_gate_inp.weight"))
            || (is_nemotron && (self.weights.contains(&format!("blk.{layer_idx}.ffn_up.weight")) || self.weights_cache.contains_key(&format!("blk.{layer_idx}.ffn_up.weight"))))
        {
            // Pure MoE or pure dense FFN layer in hybrid architectures (Nemotron-H layers have either SSM, Attention, or FFN, never both)
            (Tensor::zeros_like(x)?, x.clone())
        } else {
            // Standard Full Multi-Head / Grouped-Query Attention
            let (q, k, v) = if let Ok(q_l) = self.get_layer_linear(layer_idx, &["self_attn.q_proj.weight", "attn_q.weight"]) {
                let q = q_l.forward(&normed_x)?;
                let k_l = self.get_layer_linear(layer_idx, &["self_attn.k_proj.weight", "attn_k.weight"])?;
                let v_l = self.get_layer_linear(layer_idx, &["self_attn.v_proj.weight", "attn_v.weight"])?;
                let k = k_l.forward(&normed_x)?;
                let v = v_l.forward(&normed_x)?;
                (q, k, v)
            } else if let Ok(qkv_l) = self.get_layer_linear(layer_idx, &["self_attn.qkv_proj.weight", "attn_qkv.weight"]) {
                let qkv = qkv_l.forward(&normed_x)?;
                let q_dim = self.config.num_attention_heads * self.config.head_dim;
                let kv_dim = self.config.num_key_value_heads * self.config.head_dim;
                let q = qkv.narrow(candle_core::D::Minus1, 0, q_dim)?;
                let k = qkv.narrow(candle_core::D::Minus1, q_dim, kv_dim)?;
                let v = qkv.narrow(candle_core::D::Minus1, q_dim + kv_dim, kv_dim)?;
                (q, k, v)
            } else {
                return Err(LuminaError::MissingWeight(format!("Attention weights for layer {layer_idx} not found")));
            };
            if pos == 0 {
                eprintln!("🔍 [Layer {layer_idx}] Getting o_l...");
            }
            let o_l = self.get_layer_linear(layer_idx, &["self_attn.o_proj.weight", "attn_output.weight", "attn_out.weight"])?;

            // Some architectures (e.g. specific gated attention variants) fuse gate into Q projection
            // If the model does not have gated attention (like Nemotron where num_heads*head_dim == 4096), do not split Q!
            let has_q_gate = self.weights.contains(&"blk.0.attn_q_gate.weight".to_string())
                || self.weights.contains(&"model.layers.0.self_attn.q_gate.weight".to_string());
            let (q, gate_opt) = if has_q_gate && q.dim(candle_core::D::Minus1)? == self.config.num_attention_heads * self.config.head_dim * 2 {
                let half_dim = self.config.num_attention_heads * self.config.head_dim;
                let q_proj = q.narrow(candle_core::D::Minus1, 0, half_dim)?;
                let gate_proj = q.narrow(candle_core::D::Minus1, half_dim, half_dim)?;
                let gate_act = candle_nn::ops::silu(&gate_proj).map_err(LuminaError::Candle)?;
                (q_proj, Some(gate_act))
            } else {
                (q, None)
            };

            let b_sz = x.dim(0).map_err(LuminaError::Candle)?;
            let q_last_dim = q.dim(candle_core::D::Minus1)?;
            let k_last_dim = k.dim(candle_core::D::Minus1)?;
            let v_last_dim = v.dim(candle_core::D::Minus1)?;

            let cur_head_dim = self.config.head_dim;
            let cur_num_heads = q_last_dim / cur_head_dim;
            let cur_num_kv_heads = (k_last_dim / cur_head_dim).max(1);

            if pos == 0 {
                eprintln!("🔍 [Layer {layer_idx}] Attention dims: q_last_dim={q_last_dim}, head_dim={cur_head_dim}, num_heads={cur_num_heads}, kv_heads={cur_num_kv_heads}");
            }

            let mut q = q.reshape((b_sz, seq_len, cur_num_heads, cur_head_dim))?.transpose(1, 2)?;
            let mut k = k.reshape((b_sz, seq_len, cur_num_kv_heads, cur_head_dim))?.transpose(1, 2)?;
            let v = v.reshape((b_sz, seq_len, cur_num_kv_heads, cur_head_dim))?.transpose(1, 2)?;

            // Qwen3 applies per-head RMSNorm (over head_dim) to Q and K before RoPE.
            if self.config.qk_norm {
                if let Ok(qn_w) = self.get_layer_weight(layer_idx, &["self_attn.q_norm.weight", "attn_q_norm.weight"]) {
                    q = Self::apply_rms_norm_configured(&q, &qn_w, self.config.rms_norm_eps, self.config.unit_offset_norm)?;
                }
                if let Ok(kn_w) = self.get_layer_weight(layer_idx, &["self_attn.k_norm.weight", "attn_k_norm.weight"]) {
                    k = Self::apply_rms_norm_configured(&k, &kn_w, self.config.rms_norm_eps, self.config.unit_offset_norm)?;
                }
            }

            // Layer-specific sliding window (e.g. Gemma 3 sliding_window_pattern alternates 5:1 local/global)
            let sw = match (self.config.sliding_window, self.config.sliding_window_pattern) {
                (Some(w), Some(p)) if p > 0 => if (layer_idx + 1) % p == 0 { None } else { Some(w) },
                (s, _) => s,
            };

            // RoPE theta: in Gemma 3, sliding window layers use local base (10,000.0), global layers use 1,000,000.0
            let theta = if self.config.is_gemma && sw.is_some() {
                10_000.0f32
            } else {
                self.config.rope_theta
            };

            // Check if model is Nemotron (which uses no RoPE)
            let is_nemotron = self.weights.contains(&"blk.0.ssm_a".to_string())
                && self.weights.contains(&"blk.1.ffn_gate_inp.weight".to_string());

            // Apply RoPE unless Nemotron
            let (q, k) = if is_nemotron {
                (q, k)
            } else {
                self.apply_rope(&q, &k, pos, seq_len, theta, sw.is_some())?
            };

            // Append to KV Cache with sliding window bounding
            let (k, v) = self.kv_cache.layers[layer_idx]
                .append_with_window(&k, &v, sw)
                .map_err(LuminaError::Candle)?;

            // GQA repeat KV if needed
            let n_rep = (cur_num_heads / cur_num_kv_heads).max(1);
            let (k, v) = if n_rep > 1 {
                let (b, n_kv_heads, s_len, h_dim) = k.shape().dims4().map_err(LuminaError::Candle)?;
                let rep_k = k.unsqueeze(2)?.expand((b, n_kv_heads, n_rep, s_len, h_dim))?.reshape((b, n_kv_heads * n_rep, s_len, h_dim))?;
                let rep_v = v.unsqueeze(2)?.expand((b, n_kv_heads, n_rep, s_len, h_dim))?.reshape((b, n_kv_heads * n_rep, s_len, h_dim))?;
                (rocm_contiguous(&rep_k)?, rocm_contiguous(&rep_v)?)
            } else {
                (k, v)
            };

            // Scaled dot product attention: q is [B, H, S_q, D], k is [B, H, S_k, D] -> att is [B, H, S_q, S_k]
            let scale = self.config.attention_scale.unwrap_or(1.0 / (cur_head_dim as f64).sqrt());
            let q = rocm_contiguous(&q)?;
            let k_t = rocm_contiguous(&k.transpose(2, 3).map_err(LuminaError::Candle)?)?;
            let mut att = (q.matmul(&k_t)? * scale).map_err(LuminaError::Candle)?;

            // Google Gemma attention logit soft-capping: att = cap * tanh(att / cap)
            if let Some(cap) = self.config.attn_logit_softcapping {
                if cap > 0.0 {
                    let inv_cap = 1.0 / cap;
                    let scaled = (&att * inv_cap).map_err(LuminaError::Candle)?;
                    let tanh_att = scaled.tanh().map_err(LuminaError::Candle)?;
                    att = (tanh_att * cap).map_err(LuminaError::Candle)?;
                }
            }

            // Causal mask for prefill (supports sliding window)
            let att = if seq_len > 1 {
                let mask = self.causal_mask(seq_len, att.device(), sw)?;
                att.broadcast_add(&mask).map_err(LuminaError::Candle)?
            } else {
                att
            };

            let att = rocm_contiguous(&crate::device::softmax_last_dim(&att).map_err(LuminaError::Candle)?)?;
            let v = rocm_contiguous(&v)?;
            let out = att.matmul(&v).map_err(LuminaError::Candle)?;
            let attn_dim = cur_num_heads * cur_head_dim;
            let mut out = rocm_contiguous(&out.transpose(1, 2).map_err(LuminaError::Candle)?)?.reshape((b_sz, seq_len, attn_dim))?;
            
            if let Some(gate) = gate_opt {
                out = (out * gate).map_err(LuminaError::Candle)?;
            }

            let mut attn_out = o_l.forward(&out)?;

            // Gemma 2 / 3 applies post_attention_norm to attn_out before adding residual
            if self.config.is_gemma {
                if let Ok(post_attn_w) = self.get_layer_weight(layer_idx, &["post_attention_norm.weight"]) {
                    attn_out = self.apply_norm(&attn_out, &post_attn_w)?;
                }
            }

            let x_res = (x + &attn_out).map_err(LuminaError::Candle)?;
            (attn_out, x_res)
        };

        // Pre-feedforward layernorm
        // In Qwen/Llama, post_attention_norm.weight is the FFN pre-norm.
        // In Gemma 2/3, post_attention_norm is applied to attn_out, and ffn_norm.weight / pre_feedforward_layernorm.weight is the FFN pre-norm.
        // In Nemotron-H (both MoE and dense 4B), each layer is single-block: SSM layers return x_res directly without FFN,
        // Attention layers return x_res directly, and pure FFN/MoE layers use attn_norm as the FFN pre-norm with no attention sub-block.
        let is_nemotron = self.weights.contains(&"blk.0.ssm_a".to_string())
            && (self.weights.contains(&"blk.1.ffn_gate_inp.weight".to_string()) || self.weights.contains(&"blk.1.ffn_up.weight".to_string()));

        if is_nemotron && (is_ssm_layer || self.weights.contains(&format!("blk.{layer_idx}.attn_q.weight"))) {
            return Ok(x);
        }

        let ffn_norm_w = if is_nemotron {
            self.get_layer_weight(layer_idx, &["attn_norm.weight", "input_layernorm.weight"])?
        } else {
            self.get_layer_weight(layer_idx, &["pre_feedforward_layernorm.weight", "ffn_norm.weight", "post_attention_layernorm.weight", "post_attention_norm.weight"])?
        };
        let normed_post_x = self.apply_norm(&x, &ffn_norm_w)?;

        // Check if this layer is MoE (DeepSeek / Mixtral / Nemotron)
        let is_moe_layer = self.weights.contains(&format!("model.layers.{layer_idx}.mlp.gate.weight"))
            || self.weights.contains(&format!("layers.{layer_idx}.mlp.gate.weight"))
            || self.weights.contains(&format!("blk.{layer_idx}.ffn_gate_inp.weight"))
            || self.weights_cache.contains_key(&format!("model.layers.{layer_idx}.mlp.gate.weight"))
            || self.weights_cache.contains_key(&format!("layers.{layer_idx}.mlp.gate.weight"));

        let mlp_out = if is_moe_layer {
            let router_w = self.get_layer_weight(layer_idx, &["mlp.gate.weight", "ffn_gate_inp.weight"])?;
            let router_logits = Self::matmul_linear(&normed_post_x, &router_w)?;

            // Gating function: LFM2.5 explicitly specifies expert_gating_func = 2 (SIGMOID)
            // Nemotron-H MoE also uses Sigmoid routing with expert_weights_scale = 2.5
            let is_lfm = self.weights.contains(&"blk.0.shortconv.in_proj.weight".to_string())
                || self.weights_cache.contains_key("blk.0.shortconv.in_proj.weight");
            let is_nemotron_moe = self.weights.contains(&"blk.1.ffn_gate_inp.weight".to_string());

            let router_probs = if is_lfm || is_nemotron_moe {
                // Portable sigmoid in F32: (1 + exp(-x))^-1
                let orig_dt = router_logits.dtype();
                let f32_logits = router_logits.to_dtype(DType::F32).map_err(LuminaError::Candle)?;
                let neg = f32_logits.neg().map_err(LuminaError::Candle)?;
                let exp = neg.exp().map_err(LuminaError::Candle)?;
                let denom = (exp + 1.0).map_err(LuminaError::Candle)?;
                let sig = denom.recip().map_err(LuminaError::Candle)?;
                sig.to_dtype(orig_dt).map_err(LuminaError::Candle)?
            } else {
                crate::device::softmax_last_dim(&router_logits).map_err(LuminaError::Candle)?
            };

            let num_experts = router_w.shape().dims2().map_err(LuminaError::Candle)?.0;
            let top_k = if is_nemotron_moe {
                6.min(num_experts)
            } else {
                self.config.moe_top_k.min(num_experts).max(1)
            };

            // Optional DeepSeek / LFM / Nemotron aux-loss-free expert selection bias
            let selection_probs = if let Ok(exp_b) = self.get_layer_weight(layer_idx, &["exp_probs_b.bias", "ffn_exp_probs_b.bias"]) {
                router_probs.broadcast_add(&exp_b).map_err(LuminaError::Candle)?
            } else {
                router_probs.clone()
            };

            let b_sz = normed_post_x.dim(0).map_err(LuminaError::Candle)?;
            let total_tokens = b_sz * seq_len;
            let hidden_dim = normed_post_x.dim(candle_core::D::Minus1).map_err(LuminaError::Candle)?;

            let mut accum_out: Option<Tensor> = None;

            let has_fused_exps = self.weights.contains(&format!("blk.{layer_idx}.ffn_gate_exps.weight"))
                || self.weights.contains(&format!("blk.{layer_idx}.ffn_up_exps.weight"))
                || self.weights_cache.contains_key(&format!("blk.{layer_idx}.ffn_gate_exps.weight"))
                || self.weights_cache.contains_key(&format!("blk.{layer_idx}.ffn_up_exps.weight"));

            if total_tokens == 1 {
                // Single-token autoregressive decoding step (fast scalar path)
                let sel_flat = selection_probs.flatten_all().map_err(LuminaError::Candle)?.to_dtype(DType::F32).map_err(LuminaError::Candle)?.to_vec1::<f32>().map_err(LuminaError::Candle)?;
                let clean_flat = router_probs.flatten_all().map_err(LuminaError::Candle)?.to_dtype(DType::F32).map_err(LuminaError::Candle)?.to_vec1::<f32>().map_err(LuminaError::Candle)?;

                let mut expert_indices: Vec<(usize, f32)> = sel_flat[..num_experts.min(sel_flat.len())]
                    .iter()
                    .enumerate()
                    .map(|(i, &p)| (i, p))
                    .collect();
                expert_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                expert_indices.truncate(top_k);

                // Combination weights are derived from unbiased router_probs
                let expert_weights: Vec<(usize, f32)> = expert_indices.iter().map(|&(idx, _)| (idx, clean_flat[idx])).collect();
                let weight_sum: f32 = expert_weights.iter().map(|(_, w)| *w).sum();
                let norm_factor = if weight_sum > 0.0 { 1.0 / weight_sum } else { 1.0 };

                if has_fused_exps {
                    let fused_g = self.get_layer_weight(layer_idx, &["ffn_gate_exps.weight", "mlp.experts.gate_exps.weight"]).ok();
                    let fused_u = self.get_layer_weight(layer_idx, &["ffn_up_exps.weight", "mlp.experts.up_exps.weight"])?;
                    let fused_d = self.get_layer_weight(layer_idx, &["ffn_down_exps.weight", "mlp.experts.down_exps.weight"])?;

                    for (exp_idx, raw_weight) in expert_weights {
                        let norm_w = raw_weight * norm_factor;
                        let u_w = fused_u.get(exp_idx)?;
                        let d_w = fused_d.get(exp_idx)?;

                        let u = Self::matmul_linear(&normed_post_x, &u_w)?;
                        let exp_act = if is_nemotron_moe {
                            // Nemotron-H MoE uses squared ReLU (ReLU²) activation without gate
                            let relu_u = u.relu().map_err(LuminaError::Candle)?;
                            relu_u.sqr().map_err(LuminaError::Candle)?
                        } else if let Some(ref fg) = fused_g {
                            let g_w = fg.get(exp_idx)?;
                            let g = Self::matmul_linear(&normed_post_x, &g_w)?;
                            let silu_g = candle_nn::ops::silu(&g).map_err(LuminaError::Candle)?;
                            (silu_g * u).map_err(LuminaError::Candle)?
                        } else {
                            candle_nn::ops::silu(&u).map_err(LuminaError::Candle)?
                        };
                        let exp_out = Self::matmul_linear(&exp_act, &d_w)?;

                        let weighted = (exp_out * (norm_w as f64)).map_err(LuminaError::Candle)?;

                        accum_out = match accum_out {
                            Some(prev) => Some((prev + weighted).map_err(LuminaError::Candle)?),
                            None => Some(weighted),
                        };
                    }
                } else {
                    for (exp_idx, raw_weight) in expert_weights {
                        let norm_w = raw_weight * norm_factor;
                        let gate_name = format!("model.layers.{layer_idx}.mlp.experts.{exp_idx}.gate_proj.weight");
                        let up_name = format!("model.layers.{layer_idx}.mlp.experts.{exp_idx}.up_proj.weight");
                        let down_name = format!("model.layers.{layer_idx}.mlp.experts.{exp_idx}.down_proj.weight");

                        let alt_g = format!("layers.{layer_idx}.mlp.experts.{exp_idx}.gate_proj.weight");
                        let alt_u = format!("layers.{layer_idx}.mlp.experts.{exp_idx}.up_proj.weight");
                        let alt_d = format!("layers.{layer_idx}.mlp.experts.{exp_idx}.down_proj.weight");

                        let gguf_g = format!("blk.{layer_idx}.ffn_gate.{exp_idx}.weight");
                        let gguf_u = format!("blk.{layer_idx}.ffn_up.{exp_idx}.weight");
                        let gguf_d = format!("blk.{layer_idx}.ffn_down.{exp_idx}.weight");

                        if let (Ok(g_w), Ok(u_w), Ok(d_w)) = (
                            self.get_weight_or_fallback(&[&gate_name, &alt_g, &gguf_g]),
                            self.get_weight_or_fallback(&[&up_name, &alt_u, &gguf_u]),
                            self.get_weight_or_fallback(&[&down_name, &alt_d, &gguf_d]),
                        ) {
                            let g = Self::matmul_linear(&normed_post_x, &g_w)?;
                            let u = Self::matmul_linear(&normed_post_x, &u_w)?;
                            let silu_g = candle_nn::ops::silu(&g).map_err(LuminaError::Candle)?;
                            let exp_act = (silu_g * u).map_err(LuminaError::Candle)?;
                            let exp_out = Self::matmul_linear(&exp_act, &d_w)?;

                            let weighted = (exp_out * (norm_w as f64)).map_err(LuminaError::Candle)?;

                            accum_out = match accum_out {
                                Some(prev) => Some((prev + weighted).map_err(LuminaError::Candle)?),
                                None => Some(weighted),
                            };
                        }
                    }
                }
            } else {
                // Prefill / Multi-token path: dispatch per-token top-k experts
                let sel_flat = selection_probs.flatten_all().map_err(LuminaError::Candle)?.to_dtype(DType::F32).map_err(LuminaError::Candle)?.to_vec1::<f32>().map_err(LuminaError::Candle)?;
                let clean_flat = router_probs.flatten_all().map_err(LuminaError::Candle)?.to_dtype(DType::F32).map_err(LuminaError::Candle)?.to_vec1::<f32>().map_err(LuminaError::Candle)?;

                let mut out_tokens: Vec<Tensor> = Vec::with_capacity(total_tokens);
                let normed_flat = normed_post_x.reshape((total_tokens, hidden_dim))?;

                if has_fused_exps {
                    let fused_g = self.get_layer_weight(layer_idx, &["ffn_gate_exps.weight", "mlp.experts.gate_exps.weight"]).ok();
                    let fused_u = self.get_layer_weight(layer_idx, &["ffn_up_exps.weight", "mlp.experts.up_exps.weight"])?;
                    let fused_d = self.get_layer_weight(layer_idx, &["ffn_down_exps.weight", "mlp.experts.down_exps.weight"])?;

                    for tok_idx in 0..total_tokens {
                        let tok_x = normed_flat.narrow(0, tok_idx, 1)?; // [1, hidden_dim]
                        let start_off = tok_idx * num_experts;
                        let mut expert_indices: Vec<(usize, f32)> = sel_flat[start_off..start_off + num_experts]
                            .iter()
                            .enumerate()
                            .map(|(i, &p)| (i, p))
                            .collect();
                        expert_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                        expert_indices.truncate(top_k);

                        let expert_weights: Vec<(usize, f32)> = expert_indices.iter().map(|&(idx, _)| (idx, clean_flat[start_off + idx])).collect();
                        let weight_sum: f32 = expert_weights.iter().map(|(_, w)| *w).sum();
                        let norm_factor = if weight_sum > 0.0 { 1.0 / weight_sum } else { 1.0 };

                        let mut tok_accum: Option<Tensor> = None;
                        for (exp_idx, raw_weight) in expert_weights {
                            let norm_w = raw_weight * norm_factor;
                            let u_w = fused_u.get(exp_idx)?;
                            let d_w = fused_d.get(exp_idx)?;

                            let u = Self::matmul_linear(&tok_x, &u_w)?;
                            let exp_act = if is_nemotron_moe {
                                let relu_u = u.relu().map_err(LuminaError::Candle)?;
                                relu_u.sqr().map_err(LuminaError::Candle)?
                            } else if let Some(ref fg) = fused_g {
                                let g_w = fg.get(exp_idx)?;
                                let g = Self::matmul_linear(&tok_x, &g_w)?;
                                let silu_g = candle_nn::ops::silu(&g).map_err(LuminaError::Candle)?;
                                (silu_g * u).map_err(LuminaError::Candle)?
                            } else {
                                candle_nn::ops::silu(&u).map_err(LuminaError::Candle)?
                            };
                            let exp_out = Self::matmul_linear(&exp_act, &d_w)?;

                            let weighted = (exp_out * (norm_w as f64)).map_err(LuminaError::Candle)?;
                            tok_accum = match tok_accum {
                                Some(prev) => Some((prev + weighted).map_err(LuminaError::Candle)?),
                                None => Some(weighted),
                            };
                        }
                        out_tokens.push(tok_accum.unwrap_or_else(|| tok_x.clone()));
                    }
                } else {
                    for tok_idx in 0..total_tokens {
                        let tok_x = normed_flat.narrow(0, tok_idx, 1)?;
                        let start_off = tok_idx * num_experts;
                        let mut expert_indices: Vec<(usize, f32)> = sel_flat[start_off..start_off + num_experts]
                            .iter()
                            .enumerate()
                            .map(|(i, &p)| (i, p))
                            .collect();
                        expert_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                        expert_indices.truncate(top_k);

                        let expert_weights: Vec<(usize, f32)> = expert_indices.iter().map(|&(idx, _)| (idx, clean_flat[start_off + idx])).collect();
                        let weight_sum: f32 = expert_weights.iter().map(|(_, w)| *w).sum();
                        let norm_factor = if weight_sum > 0.0 { 1.0 / weight_sum } else { 1.0 };

                        let mut tok_accum: Option<Tensor> = None;
                        for (exp_idx, raw_weight) in expert_weights {
                            let norm_w = raw_weight * norm_factor;
                            let gate_name = format!("model.layers.{layer_idx}.mlp.experts.{exp_idx}.gate_proj.weight");
                            let up_name = format!("model.layers.{layer_idx}.mlp.experts.{exp_idx}.up_proj.weight");
                            let down_name = format!("model.layers.{layer_idx}.mlp.experts.{exp_idx}.down_proj.weight");

                            let alt_g = format!("layers.{layer_idx}.mlp.experts.{exp_idx}.gate_proj.weight");
                            let alt_u = format!("layers.{layer_idx}.mlp.experts.{exp_idx}.up_proj.weight");
                            let alt_d = format!("layers.{layer_idx}.mlp.experts.{exp_idx}.down_proj.weight");

                            let gguf_g = format!("blk.{layer_idx}.ffn_gate.{exp_idx}.weight");
                            let gguf_u = format!("blk.{layer_idx}.ffn_up.{exp_idx}.weight");
                            let gguf_d = format!("blk.{layer_idx}.ffn_down.{exp_idx}.weight");

                            if let (Ok(g_w), Ok(u_w), Ok(d_w)) = (
                                self.get_weight_or_fallback(&[&gate_name, &alt_g, &gguf_g]),
                                self.get_weight_or_fallback(&[&up_name, &alt_u, &gguf_u]),
                                self.get_weight_or_fallback(&[&down_name, &alt_d, &gguf_d]),
                            ) {
                                let g = Self::matmul_linear(&tok_x, &g_w)?;
                                let u = Self::matmul_linear(&tok_x, &u_w)?;
                                let silu_g = candle_nn::ops::silu(&g).map_err(LuminaError::Candle)?;
                                let exp_act = (silu_g * u).map_err(LuminaError::Candle)?;
                                let exp_out = Self::matmul_linear(&exp_act, &d_w)?;

                                let weighted = (exp_out * (norm_w as f64)).map_err(LuminaError::Candle)?;
                                tok_accum = match tok_accum {
                                    Some(prev) => Some((prev + weighted).map_err(LuminaError::Candle)?),
                                    None => Some(weighted),
                                };
                            }
                        }
                        out_tokens.push(tok_accum.unwrap_or_else(|| tok_x.clone()));
                    }
                }

                let token_refs: Vec<&Tensor> = out_tokens.iter().collect();
                let stacked = Tensor::cat(&token_refs, 0)?;
                accum_out = Some(stacked.reshape((b_sz, seq_len, hidden_dim))?);
            }

            // Shared experts (DeepSeek V3 / R1 / Nemotron-H)
            let shared_gate_names = [
                format!("model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight"),
                format!("layers.{layer_idx}.mlp.shared_experts.gate_proj.weight"),
            ];
            let shared_up_names = [
                format!("blk.{layer_idx}.ffn_up_shexp.weight"),
                format!("model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight"),
                format!("layers.{layer_idx}.mlp.shared_experts.up_proj.weight"),
            ];
            let shared_down_names = [
                format!("blk.{layer_idx}.ffn_down_shexp.weight"),
                format!("model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight"),
                format!("layers.{layer_idx}.mlp.shared_experts.down_proj.weight"),
            ];

            let str_refs_g: Vec<&str> = shared_gate_names.iter().map(|s| s.as_str()).collect();
            let str_refs_u: Vec<&str> = shared_up_names.iter().map(|s| s.as_str()).collect();
            let str_refs_d: Vec<&str> = shared_down_names.iter().map(|s| s.as_str()).collect();

            if let (Ok(su_w), Ok(sd_w)) = (
                self.get_weight_or_fallback(&str_refs_u),
                self.get_weight_or_fallback(&str_refs_d),
            ) {
                let su = Self::matmul_linear(&normed_post_x, &su_w)?;
                let shared_act = if is_nemotron_moe {
                    let relu_su = su.relu().map_err(LuminaError::Candle)?;
                    relu_su.sqr().map_err(LuminaError::Candle)?
                } else if let Ok(sg_w) = self.get_weight_or_fallback(&str_refs_g) {
                    let sg = Self::matmul_linear(&normed_post_x, &sg_w)?;
                    let silu_sg = candle_nn::ops::silu(&sg).map_err(LuminaError::Candle)?;
                    (silu_sg * su).map_err(LuminaError::Candle)?
                } else {
                    candle_nn::ops::silu(&su).map_err(LuminaError::Candle)?
                };
                let shared_out = Self::matmul_linear(&shared_act, &sd_w)?;

                accum_out = match accum_out {
                    Some(prev) => Some((prev + shared_out).map_err(LuminaError::Candle)?),
                    None => Some(shared_out),
                };
            }

            accum_out.unwrap_or_else(|| normed_post_x.clone())
        } else {
            // Standard dense SwiGLU MLP / Nemotron squared ReLU MLP
            if is_nemotron {
                let up_l = self.get_layer_linear(layer_idx, &["mlp.up_proj.weight", "ffn_up.weight"])?;
                let down_l = self.get_layer_linear(layer_idx, &["mlp.down_proj.weight", "ffn_down.weight"])?;
                let up = up_l.forward(&normed_post_x)?;
                let relu_up = up.relu().map_err(LuminaError::Candle)?;
                let act = relu_up.sqr().map_err(LuminaError::Candle)?;
                down_l.forward(&act)?
            } else if let Ok(gate_l) = self.get_layer_linear(layer_idx, &["mlp.gate_proj.weight", "ffn_gate.weight"]) {
                let up_l = self.get_layer_linear(layer_idx, &["mlp.up_proj.weight", "ffn_up.weight"])?;
                let down_l = self.get_layer_linear(layer_idx, &["mlp.down_proj.weight", "ffn_down.weight"])?;
                let llama_mlp = super::llama::LlamaMlp::new(gate_l, up_l, down_l);
                llama_mlp.forward(&normed_post_x)?
            } else if let Ok(fused_up_l) = self.get_layer_linear(layer_idx, &["ffn_up.weight", "mlp.up_proj.weight"]) {
                // Dedicated modular Phi-3 / Phi-4 SwiGLU block (fused gate+up in ffn_up)
                let down_l = self.get_layer_linear(layer_idx, &["mlp.down_proj.weight", "ffn_down.weight"])?;
                let phi_mlp = super::phi::PhiMlp::new(fused_up_l, down_l, self.config.intermediate_size);
                phi_mlp.forward(&normed_post_x)?
            } else {
                return Err(LuminaError::MissingWeight(format!("FFN up/gate weight for layer {layer_idx} not found")));
            }
        };

        let mut mlp_out = mlp_out;
        if let Ok(post_ffw_w) = self.get_layer_weight(layer_idx, &["post_ffw_norm.weight"]) {
            mlp_out = self.apply_norm(&mlp_out, &post_ffw_w)?;
        }

        let x = (x + mlp_out).map_err(LuminaError::Candle)?;
        Ok(x)
    }

    fn matmul_linear(x: &Tensor, w: &Tensor) -> Result<Tensor> {
        let w_dtype = w.dtype();
        let w_float = if w_dtype == DType::U8 {
            // W8A16 on-the-fly symmetric dequantization: cast u8 to target dtype and center around 128
            let target_dt = x.dtype();
            let shifted = (w.to_dtype(DType::F32).map_err(LuminaError::Candle)? - 128.0).map_err(LuminaError::Candle)?;
            (shifted * (1.0 / 127.0)).map_err(LuminaError::Candle)?.to_dtype(target_dt).map_err(LuminaError::Candle)?
        } else {
            w.clone()
        };

        let last_x_dim = x.dim(candle_core::D::Minus1).map_err(LuminaError::Candle)?;
        let w_dims = w_float.shape().dims2().map_err(LuminaError::Candle)?;
        let w_adapted = if w_dims.1 == last_x_dim {
            // w is [out_dim, in_dim] -> transpose to [in_dim, out_dim]
            w_float.t().map_err(LuminaError::Candle)?
        } else if w_dims.0 == last_x_dim {
            // w is already [in_dim, out_dim]
            w_float
        } else {
            return Err(LuminaError::Config(format!(
                "matmul dimension mismatch: x last dim {} vs w dims {:?}",
                last_x_dim, w_dims
            )));
        };

        let w_adapted = rocm_contiguous(&w_adapted)?;

        let x_dims = x.shape().dims();
        if x_dims.len() == 3 {
            let (b, s, in_d) = (x_dims[0], x_dims[1], x_dims[2]);
            let x_2d = rocm_contiguous(&x.reshape((b * s, in_d)).map_err(LuminaError::Candle)?)?;
            let out_2d = x_2d.matmul(&w_adapted).map_err(LuminaError::Candle)?;
            let out_dim = w_adapted.dim(1).map_err(LuminaError::Candle)?;
            out_2d.reshape((b, s, out_dim)).map_err(LuminaError::Candle)
        } else {
            rocm_contiguous(x)?.matmul(&w_adapted).map_err(LuminaError::Candle)
        }
    }

    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        let n_rep = self.config.num_attention_heads / self.config.num_key_value_heads;
        if n_rep == 1 {
            return Ok(x);
        }
        let (b, n_kv_heads, seq_len, head_dim) = x.shape().dims4().map_err(LuminaError::Candle)?;
        let x = x.unsqueeze(2)?.expand((b, n_kv_heads, n_rep, seq_len, head_dim))?;
        let x = x.reshape((b, n_kv_heads * n_rep, seq_len, head_dim))?;
        rocm_contiguous(&x)
    }

    fn apply_rope(&self, q: &Tensor, k: &Tensor, pos: usize, seq_len: usize, theta: f32, is_sliding_layer: bool) -> Result<(Tensor, Tensor)> {
        let dev = q.device();
        let head_dim = self.config.head_dim;
        let half_dim = head_dim / 2;

        let (positions, freqs) = if !is_sliding_layer && self.config.rope_scaling_type.as_deref() == Some("yarn") {
            let orig_ctx = self.config.rope_scaling_orig_ctx as f32;
            let factor = self.config.rope_scaling_factor.max(1.0);
            let beta_slow = self.config.yarn_beta_slow;
            let beta_fast = self.config.yarn_beta_fast;

            let freqs: Vec<f32> = (0..half_dim).map(|i| {
                let base_freq = 1.0f32 / theta.powf((2 * i) as f32 / head_dim as f32);
                let lambda = 2.0 * std::f32::consts::PI / base_freq;
                let low = orig_ctx / beta_fast;
                let high = orig_ctx / beta_slow;
                if lambda < low {
                    base_freq
                } else if lambda > high {
                    base_freq / factor
                } else {
                    let alpha = (lambda - low) / (high - low);
                    (1.0 - alpha) * base_freq + alpha * (base_freq / factor)
                }
            }).collect();
            let positions: Vec<f32> = (pos..pos + seq_len).map(|p| p as f32).collect();
            (positions, freqs)
        } else {
            let scale_factor = if !is_sliding_layer && self.config.rope_scaling_factor > 0.0 {
                self.config.rope_scaling_factor
            } else {
                1.0f32
            };
            let positions: Vec<f32> = (pos..pos + seq_len).map(|p| (p as f32) / scale_factor).collect();
            let freqs: Vec<f32> = (0..half_dim).map(|i| 1.0f32 / theta.powf((2 * i) as f32 / head_dim as f32)).collect();
            (positions, freqs)
        };

        let pos_tensor = Tensor::new(&positions[..], dev)?.unsqueeze(1)?; // [S, 1]
        let freq_tensor = Tensor::new(&freqs[..], dev)?.unsqueeze(0)?; // [1, half_dim]

        let angles = pos_tensor.matmul(&freq_tensor).map_err(LuminaError::Candle)?; // [S, half_dim]
        let cos = angles.cos().map_err(LuminaError::Candle)?;
        let sin = angles.sin().map_err(LuminaError::Candle)?;

        // cos/sin: [1, 1, seq_len, head_dim] matching q/k shape [B, num_heads, seq_len, head_dim]
        let cos = Tensor::cat(&[&cos, &cos], 1)?.unsqueeze(0)?.unsqueeze(1)?.to_dtype(q.dtype())?;
        let sin = Tensor::cat(&[&sin, &sin], 1)?.unsqueeze(0)?.unsqueeze(1)?.to_dtype(q.dtype())?;

        let rotate = |x: &Tensor| -> CandleResult<Tensor> {
            let x1 = x.narrow(3, 0, half_dim)?;
            let x2 = x.narrow(3, half_dim, half_dim)?;
            let neg_x2 = (x2 * -1.0)?;
            Tensor::cat(&[&neg_x2, &x1], 3)
        };

        let q_rot = rotate(q).map_err(LuminaError::Candle)?;
        let k_rot = rotate(k).map_err(LuminaError::Candle)?;

        let q_out = ((q.broadcast_mul(&cos)?) + (q_rot.broadcast_mul(&sin)?)).map_err(LuminaError::Candle)?;
        let k_out = ((k.broadcast_mul(&cos)?) + (k_rot.broadcast_mul(&sin)?)).map_err(LuminaError::Candle)?;

        Ok((rocm_contiguous(&q_out)?, rocm_contiguous(&k_out)?))
    }

    fn causal_mask(&self, seq_len: usize, device: &Device, sliding_window: Option<usize>) -> Result<Tensor> {
        let mut mask_vec = vec![0.0f32; seq_len * seq_len];
        let neg_inf = -1e4f32;
        for i in 0..seq_len {
            for j in 0..seq_len {
                if j > i {
                    mask_vec[i * seq_len + j] = neg_inf;
                } else if let Some(sw) = sliding_window {
                    if i.saturating_sub(j) > sw {
                        mask_vec[i * seq_len + j] = neg_inf;
                    }
                }
            }
        }
        let mask = Tensor::new(&mask_vec[..], device)?.reshape((1, 1, seq_len, seq_len))?.to_dtype(self.dtype)?;
        Ok(mask)
    }

    fn get_weight_or_fallback(&mut self, candidates: &[&str]) -> Result<Tensor> {
        for name in candidates {
            if let Some(t) = self.weights_cache.get(*name) {
                return Ok(t.clone());
            }
            if self.weights.contains(name) {
                let t = self.weights.get_tensor(name, &self.device, self.dtype)?;
                self.weights_cache.insert(name.to_string(), t.clone());
                return Ok(t);
            }
        }
        Err(LuminaError::MissingWeight(format!(
            "None of candidate weights {:?} found in archive",
            candidates
        )))
    }

    fn get_linear_or_fallback(&mut self, candidates: &[&str]) -> Result<LinearLayer> {
        for name in candidates {
            if let Some(l) = self.linear_cache.get(*name) {
                return Ok(l.clone());
            }
            if self.weights.contains(name) {
                let l = self.weights.get_linear(name, &self.device, self.dtype)?;
                self.linear_cache.insert(name.to_string(), l.clone());
                return Ok(l);
            }
        }
        Err(LuminaError::MissingWeight(format!(
            "None of candidate linear layers {:?} found in archive",
            candidates
        )))
    }

    /// Resolve target device for a given transformer layer based on n_gpu_layers offloading configuration
    pub fn device_for_layer(&self, layer_idx: usize) -> Device {
        match self.n_gpu_layers {
            Some(n) => {
                if layer_idx < n {
                    self.device.clone()
                } else {
                    Device::Cpu
                }
            }
            None => self.device.clone(),
        }
    }

    fn get_layer_weight(&mut self, layer_idx: usize, suffixes: &[&str]) -> Result<Tensor> {
        let layer_dev = self.device_for_layer(layer_idx);
        let prefixes = [
            format!("model.layers.{layer_idx}."),
            format!("layers.{layer_idx}."),
            format!("blk.{layer_idx}."),
        ];
        for prefix in &prefixes {
            for suffix in suffixes {
                let key = format!("{prefix}{suffix}");
                if let Some(t) = self.weights_cache.get(&key) {
                    if t.device().same_device(&layer_dev) {
                        return Ok(t.clone());
                    } else {
                        let t_moved = t.to_device(&layer_dev).map_err(LuminaError::Candle)?;
                        self.weights_cache.insert(key, t_moved.clone());
                        return Ok(t_moved);
                    }
                }
                if self.weights.contains(&key) {
                    let t = self.weights.get_tensor(&key, &layer_dev, self.dtype)?;
                    self.weights_cache.insert(key, t.clone());
                    return Ok(t);
                }
            }
        }
        Err(LuminaError::MissingWeight(format!(
            "Weight for layer {} with suffixes {:?} not found",
            layer_idx, suffixes
        )))
    }

    fn get_layer_linear(&mut self, layer_idx: usize, suffixes: &[&str]) -> Result<LinearLayer> {
        let layer_dev = self.device_for_layer(layer_idx);
        let prefixes = [
            format!("model.layers.{layer_idx}."),
            format!("layers.{layer_idx}."),
            format!("blk.{layer_idx}."),
        ];
        for prefix in &prefixes {
            for suffix in suffixes {
                let key = format!("{prefix}{suffix}");
                if let Some(layer) = self.linear_cache.get(&key) {
                    return Ok(layer.clone());
                }
                if self.weights.contains(&key) {
                    let linear = self.weights.get_linear(&key, &layer_dev, self.dtype)?;
                    self.linear_cache.insert(key, linear.clone());
                    return Ok(linear);
                }
            }
        }
        Err(LuminaError::MissingWeight(format!(
            "Linear layer for layer {} with suffixes {:?} not found",
            layer_idx, suffixes
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummySource;
    impl WeightsSource for DummySource {
        fn get_tensor(&self, _: &str, _: &Device, _: DType) -> Result<Tensor> { unreachable!() }
        fn contains(&self, _: &str) -> bool { false }
        fn raw_info(&self, _: &str) -> Option<(safetensors::Dtype, Vec<usize>)> { None }
        fn keys(&self) -> Vec<String> { Vec::new() }
    }

    #[test]
    fn test_chatml_multi_turn_formatting() {
        let msgs = vec![
            ChatMessage::system("You are a helpful assistant."),
            ChatMessage::user("Hello!"),
            ChatMessage::assistant("Hi there! How can I help?"),
            ChatMessage::user("What is 2+2?"),
        ];

        let config = CausalLMConfig {
            hidden_size: 64,
            intermediate_size: 128,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            num_hidden_layers: 1,
            vocab_size: 100,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            max_position_embeddings: 512,
            head_dim: 32,
            is_gemma: true,
            qk_norm: false,
            sliding_window: Some(4),
            attn_logit_softcapping: Some(50.0),
            final_logit_softcapping: Some(30.0),
            unit_offset_norm: true,
            sliding_window_pattern: None,
            rope_scaling_factor: 1.0,
            rope_scaling_type: None,
            rope_scaling_orig_ctx: 512,
            yarn_beta_slow: 1.0,
            yarn_beta_fast: 32.0,
            moe_num_experts: 0,
            moe_top_k: 0,
            moe_shared_experts: 0,
            attention_scale: None,
        };

        let dummy_pipeline = CausalLMPipeline::new(
            Arc::new(DummySource),
            config,
            None,
            Device::Cpu,
            DType::F32,
        );

        let prompt = dummy_pipeline.format_chat_prompt(&msgs);
        assert!(prompt.contains("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"));
        assert!(prompt.contains("<|im_start|>user\nHello!<|im_end|>\n"));
        assert!(prompt.contains("<|im_start|>assistant\nHi there! How can I help?<|im_end|>\n"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_sliding_window_causal_mask() {
        let config = CausalLMConfig {
            hidden_size: 64,
            intermediate_size: 128,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            num_hidden_layers: 1,
            vocab_size: 100,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            max_position_embeddings: 512,
            head_dim: 32,
            is_gemma: true,
            qk_norm: false,
            sliding_window: Some(2),
            attn_logit_softcapping: Some(50.0),
            final_logit_softcapping: Some(30.0),
            unit_offset_norm: true,
            sliding_window_pattern: None,
            rope_scaling_factor: 1.0,
            rope_scaling_type: None,
            rope_scaling_orig_ctx: 512,
            yarn_beta_slow: 1.0,
            yarn_beta_fast: 32.0,
            moe_num_experts: 0,
            moe_top_k: 0,
            moe_shared_experts: 0,
            attention_scale: None,
        };

        let pipeline = CausalLMPipeline::new(
            Arc::new(DummySource),
            config,
            None,
            Device::Cpu,
            DType::F32,
        );

        let mask = pipeline.causal_mask(4, &Device::Cpu, pipeline.config.sliding_window).expect("causal mask failed");
        let mask_flat: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();
        // 4x4 matrix:
        // row 0: [0, -inf, -inf, -inf]
        // row 1: [0, 0, -inf, -inf]
        // row 2: [0, 0, 0, -inf]
        // row 3 (i=3, sw=2 -> diff=3-0=3 > 2): [-inf, 0, 0, 0]
        assert_eq!(mask_flat[0 * 4 + 0], 0.0);
        assert_eq!(mask_flat[0 * 4 + 1], -1e4);
        assert_eq!(mask_flat[3 * 4 + 0], -1e4); // i=3, j=0, diff=3 > sliding_window(2)
        assert_eq!(mask_flat[3 * 4 + 1], 0.0); // i=3, j=1, diff=2 <= sliding_window(2)
        assert_eq!(mask_flat[3 * 4 + 3], 0.0);
    }

    #[test]
    fn test_gemma_unit_offset_norm_computation() {
        let x = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &Device::Cpu).unwrap().reshape((1, 4)).unwrap();
        let w = Tensor::new(&[0.1f32, 0.2, -0.1, 0.0], &Device::Cpu).unwrap();
        let normed_standard = CausalLMPipeline::apply_rms_norm_configured(&x, &w, 1e-6, false).unwrap();
        let normed_unit = CausalLMPipeline::apply_rms_norm_configured(&x, &w, 1e-6, true).unwrap();

        let std_vec: Vec<f32> = normed_standard.flatten_all().unwrap().to_vec1().unwrap();
        let unit_vec: Vec<f32> = normed_unit.flatten_all().unwrap().to_vec1().unwrap();

        // Standard scales by w (e.g. 0.1), Unit offset scales by (1.0 + w) (e.g. 1.1)
        assert!((unit_vec[0] / std_vec[0] - 11.0).abs() < 1e-3);
    }

    #[test]
    fn test_moe_top_k_expert_routing_normalization() {
        let raw_logits = [2.0f32, 0.5f32, 5.0f32, 1.0f32];
        let mut expert_weights: Vec<(usize, f32)> = raw_logits
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        expert_weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        expert_weights.truncate(2);

        assert_eq!(expert_weights[0].0, 2); // Expert 2 has highest score 5.0
        assert_eq!(expert_weights[1].0, 0); // Expert 0 has second highest score 2.0

        let weight_sum: f32 = expert_weights.iter().map(|(_, w)| *w).sum();
        assert!((weight_sum - 7.0).abs() < 1e-5);
        let norm_factor = 1.0 / weight_sum;
        let w0 = expert_weights[0].1 * norm_factor;
        let w1 = expert_weights[1].1 * norm_factor;
        assert!((w0 + w1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_layer_kv_cache_sliding_window_truncation() {
        let mut cache = LayerKVCache::new();
        let k1 = Tensor::zeros((1, 2, 3, 16), DType::F32, &Device::Cpu).unwrap();
        let v1 = Tensor::zeros((1, 2, 3, 16), DType::F32, &Device::Cpu).unwrap();

        // Append 3 tokens with sliding window 4 -> total 3 tokens (no truncation)
        let (k, v) = cache.append_with_window(&k1, &v1, Some(4)).unwrap();
        assert_eq!(k.dim(2).unwrap(), 3);
        assert_eq!(v.dim(2).unwrap(), 3);

        // Append 3 more tokens -> total would be 6, but window is 4 -> truncated to 4 tokens
        let (k_trunc, v_trunc) = cache.append_with_window(&k1, &v1, Some(4)).unwrap();
        assert_eq!(k_trunc.dim(2).unwrap(), 4);
        assert_eq!(v_trunc.dim(2).unwrap(), 4);
    }

    #[test]
    fn test_speculative_verification_logic() {
        // Verify speculative acceptance and deviation correction rules:
        let draft_tokens = vec![101u32, 102u32, 103u32];
        let target_samples = vec![101u32, 102u32, 999u32]; // 3rd token deviates

        let mut accepted = Vec::new();
        for (&draft_t, &target_t) in draft_tokens.iter().zip(target_samples.iter()) {
            if draft_t == target_t {
                accepted.push(draft_t);
            } else {
                accepted.push(target_t); // target overrides deviation
                break;
            }
        }

        assert_eq!(accepted, vec![101, 102, 999]);
        assert_eq!(accepted.len(), 3);
    }

    #[test]
    fn test_quantized_kv_cache_roundtrip() -> Result<()> {
        let device = Device::Cpu;
        // Shape [B=1, H=2, S=4, D=16]
        let data: Vec<f32> = (0..(1 * 2 * 4 * 16))
            .map(|i| ((i as f32) * 0.1).sin() * 2.5)
            .collect();
        let tensor = Tensor::from_slice(&data, (1, 2, 4, 16), &device).map_err(LuminaError::Candle)?;

        let entry = QuantizedKVCacheEntry::quantize(&tensor).map_err(LuminaError::Candle)?;
        assert_eq!(entry.data.dtype(), DType::U8);
        assert_eq!(entry.seq_len, 4);

        let reconstructed = entry.dequantize(DType::F32).map_err(LuminaError::Candle)?;
        let rec_vec: Vec<f32> = reconstructed.flatten_all().map_err(LuminaError::Candle)?.to_vec1().map_err(LuminaError::Candle)?;

        // Measure cosine similarity between original and reconstructed
        let dot: f32 = data.iter().zip(rec_vec.iter()).map(|(a, b)| a * b).sum();
        let norm_a: f32 = data.iter().map(|a| a * a).sum::<f32>().sqrt();
        let norm_b: f32 = rec_vec.iter().map(|b| b * b).sum::<f32>().sqrt();
        let cosine_sim = dot / (norm_a * norm_b);

        assert!(
            cosine_sim > 0.99,
            "Cosine similarity ({cosine_sim}) must exceed 0.99 for 8-bit quantized KV roundtrip"
        );
        Ok(())
    }

    #[test]
    fn test_quantized_kv_cache_sliding_window() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = QuantizedLayerKVCache::new();

        let k1 = Tensor::zeros((1, 2, 3, 16), DType::F32, &device).map_err(LuminaError::Candle)?;
        let v1 = Tensor::zeros((1, 2, 3, 16), DType::F32, &device).map_err(LuminaError::Candle)?;

        // Append 3 tokens with sliding window 4 -> total 3 tokens
        let (k, v) = cache.append_with_window(&k1, &v1, Some(4)).map_err(LuminaError::Candle)?;
        assert_eq!(k.dim(2).map_err(LuminaError::Candle)?, 3);
        assert_eq!(v.dim(2).map_err(LuminaError::Candle)?, 3);

        // Append 3 more tokens -> window 4 truncates from 6 to 4 tokens
        let (k_trunc, v_trunc) = cache.append_with_window(&k1, &v1, Some(4)).map_err(LuminaError::Candle)?;
        assert_eq!(k_trunc.dim(2).map_err(LuminaError::Candle)?, 4);
        assert_eq!(v_trunc.dim(2).map_err(LuminaError::Candle)?, 4);
        assert_eq!(cache.k.as_ref().unwrap().seq_len, 4);
        assert_eq!(cache.v.as_ref().unwrap().seq_len, 4);
        Ok(())
    }

    #[test]
    fn test_quantized_kv_cache_memory_footprint() -> Result<()> {
        let device = Device::Cpu;
        // Large sequence of 256 tokens with head dim 128 across 8 heads
        let k = Tensor::zeros((1, 8, 256, 128), DType::F32, &device).map_err(LuminaError::Candle)?;
        let entry = QuantizedKVCacheEntry::quantize(&k).map_err(LuminaError::Candle)?;

        let float_bytes = 1 * 8 * 256 * 128 * 4; // F32 = 4 bytes per elem = 1,048,576 bytes
        let quant_data_bytes = entry.data.elem_count() * 1; // U8 = 1 byte
        let scale_bytes = entry.scale.elem_count() * 4; // F32 scales [1, 8, 256, 1] = 2048 * 4 = 8,192 bytes
        let total_quant_bytes = quant_data_bytes + scale_bytes;

        let ratio = (total_quant_bytes as f64) / (float_bytes as f64);
        assert!(
            ratio <= 0.30,
            "Quantized KV memory ({total_quant_bytes} bytes) must be <= 30% of F32 footprint ({float_bytes} bytes, ratio {ratio})"
        );
        Ok(())
    }

    #[test]
    fn test_causal_lm_quantized_kv_storage_format() -> Result<()> {
        let mut layer_cache = LayerKVCache::new_quantized();
        assert!(layer_cache.is_quantized());

        let device = Device::Cpu;
        let k = Tensor::ones((1, 4, 2, 32), DType::F32, &device).map_err(LuminaError::Candle)?;
        let v = Tensor::ones((1, 4, 2, 32), DType::F32, &device).map_err(LuminaError::Candle)?;

        let (deq_k, deq_v) = layer_cache.append(&k, &v).map_err(LuminaError::Candle)?;
        assert_eq!(deq_k.dims(), &[1, 4, 2, 32]);
        assert_eq!(deq_v.dims(), &[1, 4, 2, 32]);

        // Verify that the internal persistent storage format is strictly U8 and scale is F32
        match &layer_cache.storage {
            KVCacheStorage::Quantized(q) => {
                let entry_k = q.k.as_ref().expect("k entry must be present");
                let entry_v = q.v.as_ref().expect("v entry must be present");
                assert_eq!(entry_k.data.dtype(), DType::U8);
                assert_eq!(entry_v.data.dtype(), DType::U8);
                assert_eq!(entry_k.scale.dtype(), DType::F32);
                assert_eq!(entry_v.scale.dtype(), DType::F32);
                assert_eq!(entry_k.seq_len, 2);
            }
            KVCacheStorage::Standard { .. } => panic!("Layer cache must be in Quantized storage mode"),
        }
        Ok(())
    }

    #[test]
    fn test_causal_lm_quantized_kv_generation_parity() -> Result<()> {
        let device = Device::Cpu;
        // Simulate a multi-step autoregressive decoding sequence with both caches
        let mut std_cache = LayerKVCache::new();
        let mut quant_cache = LayerKVCache::new_quantized();

        // 8 autoregressive steps
        for step in 0..8 {
            let val = (step as f32 + 1.0) * 0.5;
            let k_step = (Tensor::ones((1, 2, 1, 16), DType::F32, &device).map_err(LuminaError::Candle)? * val as f64).map_err(LuminaError::Candle)?;
            let v_step = (Tensor::ones((1, 2, 1, 16), DType::F32, &device).map_err(LuminaError::Candle)? * (val * 0.8) as f64).map_err(LuminaError::Candle)?;

            let (std_k, std_v) = std_cache.append(&k_step, &v_step).map_err(LuminaError::Candle)?;
            let (q_k, q_v) = quant_cache.append(&k_step, &v_step).map_err(LuminaError::Candle)?;

            // Sequence length must match exactly
            assert_eq!(std_k.dims(), q_k.dims());
            assert_eq!(std_v.dims(), q_v.dims());

            // Compute cosine similarity on the dequantized tensors at each step
            let std_vec: Vec<f32> = std_k.flatten_all().map_err(LuminaError::Candle)?.to_vec1().map_err(LuminaError::Candle)?;
            let q_vec: Vec<f32> = q_k.flatten_all().map_err(LuminaError::Candle)?.to_vec1().map_err(LuminaError::Candle)?;

            let dot: f32 = std_vec.iter().zip(q_vec.iter()).map(|(a, b)| a * b).sum();
            let norm_a: f32 = std_vec.iter().map(|a| a * a).sum::<f32>().sqrt();
            let norm_b: f32 = q_vec.iter().map(|b| b * b).sum::<f32>().sqrt();
            let sim = dot / (norm_a * norm_b);

            assert!(
                sim > 0.99,
                "Step {step} cosine similarity ({sim}) must exceed 0.99 for generation parity"
            );
        }
        Ok(())
    }

    #[test]
    fn test_quantized_kv_dtypes_support() -> Result<()> {
        let device = Device::Cpu;
        let dtypes = [DType::F32, DType::F16, DType::BF16];

        for &dt in &dtypes {
            let x = Tensor::ones((1, 2, 4, 32), dt, &device).map_err(LuminaError::Candle)?;
            let entry = QuantizedKVCacheEntry::quantize(&x).map_err(LuminaError::Candle)?;
            assert_eq!(entry.data.dtype(), DType::U8);

            let deq = entry.dequantize(dt).map_err(LuminaError::Candle)?;
            assert_eq!(deq.dtype(), dt);
            assert_eq!(deq.dims(), &[1, 2, 4, 32]);
        }
        Ok(())
    }

    #[test]
    fn test_quantized_kv_no_unquantized_leakage() -> Result<()> {
        let layer_cache = LayerKVCache::new_quantized();
        // Check that the storage is purely Quantized and has no Standard field allocated
        match &layer_cache.storage {
            KVCacheStorage::Quantized(q) => {
                assert!(q.k.is_none());
                assert!(q.v.is_none());
            }
            KVCacheStorage::Standard { .. } => {
                panic!("Quantized cache must not hold unquantized standard storage variant");
            }
        }
        Ok(())
    }

    #[test]
    fn test_phi_prompt_helpers() {
        let tpl = "{{'<|im_start|>system<|im_sep|>You are Phi.<|im_end|>'}}<|end|><|user|><|assistant|>";
        assert_eq!(phi_system_preamble(tpl).as_deref(), Some("You are Phi."));
        assert_eq!(phi_style(tpl), PhiStyle::EndTags);

        let p = build_phi_prompt("You are Phi.", "", "Capital of France?", PhiStyle::EndTags);
        assert_eq!(
            p,
            "<|im_start|>system<|im_sep|>You are Phi.<|im_end|><|end|><|user|>Capital of France?<|end|><|assistant|>"
        );

        let p2 = build_phi_prompt("Sys", "", "Hi", PhiStyle::ImSep);
        assert_eq!(
            p2,
            "<|im_start|>system<|im_sep|>Sys<|im_end|><|im_start|>user<|im_sep|>Hi<|im_end|><|im_start|>assistant<|im_sep|>"
        );
    }

    #[test]
    fn test_phi_trim_stop_markers() {
        assert_eq!(trim_stop_markers("Paris.<|end|>junk".into()), "Paris.");
        assert_eq!(trim_stop_markers("<|user|>ignored".into()), "");
        assert_eq!(trim_stop_markers("ok".into()), "ok");
    }

    #[test]
    fn test_granite_attention_scale_config() {
        // Mock weights source with granite architecture metadata
        struct MockGraniteWeights;
        impl WeightsSource for MockGraniteWeights {
            fn keys(&self) -> Vec<String> {
                vec!["blk.0.attn_q.weight".into(), "token_embd.weight".into()]
            }
            fn metadata(&self, key: &str) -> Option<String> {
                if key == "general.architecture" {
                    Some("granite".to_string())
                } else {
                    None
                }
            }
            fn metadata_usize(&self, key: &str) -> Option<usize> {
                match key {
                    "granite.block_count" => Some(64),
                    "granite.embedding_length" => Some(4096),
                    "granite.vocab_size" => Some(100352),
                    "granite.feed_forward_length" => Some(32768),
                    "granite.attention.head_count" => Some(32),
                    "granite.attention.head_count_kv" => Some(8),
                    "granite.attention.key_length" => Some(128),
                    _ => None,
                }
            }
            fn metadata_f64(&self, key: &str) -> Option<f64> {
                match key {
                    "granite.attention.scale" => Some(0.0078125),
                    "granite.rope.freq_base" => Some(50000000.0),
                    "granite.attention.layer_norm_rms_epsilon" => Some(1e-5),
                    _ => None,
                }
            }
            fn get_tensor(&self, _: &str, _: &Device, _: DType) -> Result<Tensor> {
                unreachable!()
            }
            fn contains(&self, key: &str) -> bool {
                self.keys().iter().any(|k| k == key)
            }
            fn raw_info(&self, name: &str) -> Option<(safetensors::Dtype, Vec<usize>)> {
                if name == "token_embd.weight" {
                    Some((safetensors::Dtype::F16, vec![4096, 100352]))
                } else {
                    None
                }
            }
        }

        let mock = MockGraniteWeights;
        let cfg = CausalLMConfig::from_weights(&mock).expect("Failed to build CausalLMConfig");
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.attention_scale, Some(0.0078125));
    }

    #[test]
    fn test_granite_prompt_template() {
        struct MockGranitePromptModel {
            arch: String,
        }
        impl MockGranitePromptModel {
            fn arch_name(&self) -> String {
                self.arch.clone()
            }
            fn format_prompt(&self, prompt: &str) -> String {
                if prompt.contains("<|im_sep|>") || prompt.contains("[INST]") || prompt.contains("<|im_start|>") {
                    return prompt.to_string();
                }
                let arch = self.arch_name();
                if arch == "granite" {
                    return format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n", prompt.trim());
                }
                prompt.to_string()
            }
        }

        let model = MockGranitePromptModel { arch: "granite".to_string() };
        let formatted = model.format_prompt("Explain Rust");
        assert_eq!(
            formatted,
            "<|im_start|>user\nExplain Rust<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
    }

    #[test]
    fn test_granite_finite_logits() -> Result<()> {
        let device = Device::Cpu;
        let b = 1;
        let h = 32;
        let s = 4;
        let d = 128;
        let q = Tensor::randn(0.0f32, 1.0f32, (b, h, s, d), &device).map_err(LuminaError::Candle)?;
        let k = Tensor::randn(0.0f32, 1.0f32, (b, h, s, d), &device).map_err(LuminaError::Candle)?;
        let k_t = k.transpose(2, 3).map_err(LuminaError::Candle)?;
        let scale = 0.0078125f64; // Granite scale = 1.0 / 128.0
        let att = (q.matmul(&k_t)? * scale).map_err(LuminaError::Candle)?;
        let probs = candle_nn::ops::softmax_last_dim(&att).map_err(LuminaError::Candle)?;
        let probs_vec: Vec<f32> = probs.flatten_all().map_err(LuminaError::Candle)?.to_vec1().map_err(LuminaError::Candle)?;
        assert!(probs_vec.iter().all(|x| x.is_finite() && !x.is_nan()));
        Ok(())
    }
}
