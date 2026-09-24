// [WFGY] Zone: SAFE | λ: 0.30 | Fallbacks: 1 (fallback HuggingFace tokenizer when local not found) | Action: Universal CausalLM engine supporting Llama, DeepSeek, Qwen, Gemma, Mistral in GGUF & Safetensors

use std::sync::Arc;
use candle_core::{DType, Device, IndexOp, Result as CandleResult, Tensor};
use candle_nn::{Embedding, Module, RmsNorm};
use tokenizers::Tokenizer;

use crate::error::{LuminaError, Result};
use crate::weights::WeightsSource;

/// KV Cache for an individual transformer layer.
pub struct LayerKVCache {
    pub k: Option<Tensor>,
    pub v: Option<Tensor>,
}

impl LayerKVCache {
    pub fn new() -> Self {
        Self { k: None, v: None }
    }

    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
    }

    /// Append key and value states along the sequence dimension (dim 2: [B, H, S, D]).
    pub fn append(&mut self, new_k: &Tensor, new_v: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        let (k, v) = match (&self.k, &self.v) {
            (Some(prev_k), Some(prev_v)) => {
                let k = Tensor::cat(&[prev_k, new_k], 2)?;
                let v = Tensor::cat(&[prev_v, new_v], 2)?;
                (k, v)
            }
            _ => (new_k.clone(), new_v.clone()),
        };
        self.k = Some(k.clone());
        self.v = Some(v.clone());
        Ok((k, v))
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
}

impl CausalLMConfig {
    /// Detect or extract configuration from weight keys and shapes.
    pub fn from_weights(src: &dyn WeightsSource) -> Result<Self> {
        let keys = src.keys();
        let num_layers = count_causal_layers(&keys);
        let num_layers = if num_layers == 0 { 28 } else { num_layers };

        // Sniff vocab_size and hidden_size from embed_tokens or lm_head
        let (vocab_size, hidden_size) = if let Some((_, dims)) = src.raw_info("model.embed_tokens.weight") {
            (dims.first().copied().unwrap_or(151936), dims.get(1).copied().unwrap_or(4096))
        } else if let Some((_, dims)) = src.raw_info("embed_tokens.weight") {
            (dims.first().copied().unwrap_or(151936), dims.get(1).copied().unwrap_or(4096))
        } else if let Some((_, dims)) = src.raw_info("token_embd.weight") {
            (dims.first().copied().unwrap_or(151936), dims.get(1).copied().unwrap_or(4096))
        } else if let Some((_, dims)) = src.raw_info("lm_head.weight") {
            (dims.first().copied().unwrap_or(151936), dims.get(1).copied().unwrap_or(4096))
        } else if let Some((_, dims)) = src.raw_info("output.weight") {
            (dims.first().copied().unwrap_or(151936), dims.get(1).copied().unwrap_or(4096))
        } else {
            (151936, 4096)
        };

        // Sniff intermediate_size from mlp gate or up projection
        let intermediate_size = if let Some((_, dims)) = src.raw_info("model.layers.0.mlp.gate_proj.weight") {
            dims.first().copied().unwrap_or(hidden_size * 4)
        } else if let Some((_, dims)) = src.raw_info("layers.0.mlp.gate_proj.weight") {
            dims.first().copied().unwrap_or(hidden_size * 4)
        } else if let Some((_, dims)) = src.raw_info("blk.0.ffn_gate.weight") {
            dims.first().copied().unwrap_or(hidden_size * 4)
        } else {
            hidden_size * 4
        };

        // Sniff num_heads & num_kv_heads
        let (num_heads, num_kv_heads, head_dim) = if let Some((_, dims)) = src.raw_info("model.layers.0.self_attn.q_proj.weight") {
            let q_out = dims.first().copied().unwrap_or(hidden_size);
            let k_out = src.raw_info("model.layers.0.self_attn.k_proj.weight").map(|(_, d)| d[0]).unwrap_or(q_out);
            let h_dim = 128;
            (q_out / h_dim, k_out / h_dim, h_dim)
        } else if let Some((_, dims)) = src.raw_info("layers.0.self_attn.q_proj.weight") {
            let q_out = dims.first().copied().unwrap_or(hidden_size);
            let k_out = src.raw_info("layers.0.self_attn.k_proj.weight").map(|(_, d)| d[0]).unwrap_or(q_out);
            let h_dim = 128;
            (q_out / h_dim, k_out / h_dim, h_dim)
        } else if let Some((_, dims)) = src.raw_info("blk.0.attn_q.weight") {
            let q_out = dims.first().copied().unwrap_or(hidden_size);
            let k_out = src.raw_info("blk.0.attn_k.weight").map(|(_, d)| d[0]).unwrap_or(q_out);
            let h_dim = 128;
            (q_out / h_dim, k_out / h_dim, h_dim)
        } else {
            (32, 8, 128)
        };

        let is_gemma = keys.iter().any(|k| k.contains("gemma") || k.contains("pre_feedforward_layernorm"));
        let qk_norm = keys.iter().any(|k| k.contains("q_norm") || k.contains("attn_q_norm"));
        let rope_theta = if keys.iter().any(|k| k.contains("llama") || k.contains("deepseek")) || !qk_norm {
            500_000.0f32
        } else {
            1_000_000.0f32
        };

        Ok(Self {
            hidden_size,
            intermediate_size,
            num_attention_heads: num_heads.max(1),
            num_key_value_heads: num_kv_heads.max(1),
            num_hidden_layers: num_layers,
            vocab_size,
            rms_norm_eps: 1e-6,
            rope_theta,
            max_position_embeddings: 8192,
            head_dim,
            is_gemma,
            qk_norm,
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
}

impl Default for TextGenParams {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repetition_penalty: 1.1,
            stop_tokens: vec![151643, 151645, 128001, 128009, 2, 1], // Common EOS across Qwen, Llama3, Gemma
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

        // Apply repetition penalty
        if self.repetition_penalty != 1.0 {
            for &token in generated_tokens {
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
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().subsec_nanos();
    (nanos as f32) / (1_000_000_000.0f32)
}

/// Standalone CausalLM Engine Pipeline
pub struct CausalLMPipeline {
    pub config: CausalLMConfig,
    pub tokenizer: Option<Tokenizer>,
    pub weights: Arc<dyn WeightsSource>,
    pub device: Device,
    pub dtype: DType,
    pub kv_cache: KVCache,
    pub weights_cache: std::collections::HashMap<String, Tensor>,
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
        let mut weights_cache = std::collections::HashMap::new();

        // Preload all weights onto GPU to avoid dequantizing from disk on every single token step!
        for key in weights.keys() {
            if let Ok(t) = weights.get_tensor(&key, &device, dtype) {
                weights_cache.insert(key, t);
            }
        }

        Self {
            config,
            tokenizer,
            weights,
            device,
            dtype,
            kv_cache,
            weights_cache,
        }
    }

    /// Autoregressive generation loop
    pub fn generate(&mut self, prompt: &str, params: &TextGenParams) -> Result<String> {
        let prompt_tokens = {
            let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
                LuminaError::Config("Tokenizer not attached to CausalLM pipeline".into())
            })?;
            let encoding = tokenizer.encode(prompt, true).map_err(|e| {
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
        let mut generated_tokens = Vec::new();
        let logits_processor = LogitsProcessor::new(
            params.temperature,
            params.top_p,
            params.top_k,
            params.repetition_penalty,
        );

        // Prefill prompt tokens
        let input_tensor = Tensor::new(&prompt_tokens[..], &self.device)?.unsqueeze(0)?;
        let logits = self.forward(&input_tensor, 0)?;
        let last_logits = logits.i((0, prompt_tokens.len() - 1))?;

        let mut next_token = logits_processor.sample(&last_logits, &generated_tokens)?;
        generated_tokens.push(next_token);

        let mut pos = prompt_tokens.len();

        // Autoregressive decoding loop
        for _ in 0..params.max_tokens {
            if params.stop_tokens.contains(&next_token) {
                break;
            }

            let next_input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let step_logits = self.forward(&next_input, pos)?;
            let step_last_logits = step_logits.i((0, 0))?;

            next_token = logits_processor.sample(&step_last_logits, &generated_tokens)?;
            generated_tokens.push(next_token);
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

        Ok(output_text)
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

    /// Forward pass through the Transformer
    fn forward(&mut self, input_ids: &Tensor, pos: usize) -> Result<Tensor> {
        let (_b_size, seq_len) = input_ids.shape().dims2().map_err(LuminaError::Candle)?;

        // 1. Embedding lookup
        let embed_w = self.get_weight_or_fallback(&["model.embed_tokens.weight", "token_embd.weight", "embed_tokens.weight"])?;
        let mut hidden = Embedding::new(embed_w, self.config.hidden_size).forward(input_ids).map_err(LuminaError::Candle)?;

        if self.config.is_gemma {
            let normalizer = (self.config.hidden_size as f64).sqrt();
            hidden = (hidden * normalizer).map_err(LuminaError::Candle)?;
        }

        // 2. Transformer layers loop with KV-Cache
        for layer_idx in 0..self.config.num_hidden_layers {
            hidden = self.forward_layer(layer_idx, &hidden, pos, seq_len)?;
        }

        // 3. Final layer norm
        let norm_w = self.get_weight_or_fallback(&["model.norm.weight", "output_norm.weight", "norm.weight"])?;
        let norm = RmsNorm::new(norm_w, self.config.rms_norm_eps);
        let hidden = norm.forward(&hidden).map_err(LuminaError::Candle)?;

        // 4. LM Head projection
        let lm_head_w = self.get_weight_or_fallback(&["lm_head.weight", "output.weight", "model.embed_tokens.weight", "embed_tokens.weight", "token_embd.weight"])?;
        let logits = Self::matmul_linear(&hidden, &lm_head_w)?;

        Ok(logits)
    }

    fn forward_layer(&mut self, layer_idx: usize, x: &Tensor, pos: usize, seq_len: usize) -> Result<Tensor> {
        // Input layernorm
        let in_norm_w = self.get_layer_weight(layer_idx, &["input_layernorm.weight", "attn_norm.weight"])?;
        let in_norm = RmsNorm::new(in_norm_w, self.config.rms_norm_eps);
        let normed_x = in_norm.forward(x).map_err(LuminaError::Candle)?;

        // Attention projections
        let q_w = self.get_layer_weight(layer_idx, &["self_attn.q_proj.weight", "attn_q.weight"])?;
        let k_w = self.get_layer_weight(layer_idx, &["self_attn.k_proj.weight", "attn_k.weight"])?;
        let v_w = self.get_layer_weight(layer_idx, &["self_attn.v_proj.weight", "attn_v.weight"])?;
        let o_w = self.get_layer_weight(layer_idx, &["self_attn.o_proj.weight", "attn_output.weight"])?;

        let q = Self::matmul_linear(&normed_x, &q_w)?;
        let k = Self::matmul_linear(&normed_x, &k_w)?;
        let v = Self::matmul_linear(&normed_x, &v_w)?;

        let b_sz = x.dim(0).map_err(LuminaError::Candle)?;
        let mut q = q.reshape((b_sz, seq_len, self.config.num_attention_heads, self.config.head_dim))?.transpose(1, 2)?;
        let mut k = k.reshape((b_sz, seq_len, self.config.num_key_value_heads, self.config.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b_sz, seq_len, self.config.num_key_value_heads, self.config.head_dim))?.transpose(1, 2)?;

        // Qwen3 applies per-head RMSNorm (over head_dim) to Q and K before RoPE.
        if self.config.qk_norm {
            if let Ok(qn_w) = self.get_layer_weight(layer_idx, &["self_attn.q_norm.weight", "attn_q_norm.weight"]) {
                let qn = RmsNorm::new(qn_w, self.config.rms_norm_eps);
                q = qn.forward(&q).map_err(LuminaError::Candle)?;
            }
            if let Ok(kn_w) = self.get_layer_weight(layer_idx, &["self_attn.k_norm.weight", "attn_k_norm.weight"]) {
                let kn = RmsNorm::new(kn_w, self.config.rms_norm_eps);
                k = kn.forward(&k).map_err(LuminaError::Candle)?;
            }
        }

        // Apply RoPE
        let (q, k) = self.apply_rope(&q, &k, pos, seq_len)?;

        // Append to KV Cache
        let (k, v) = self.kv_cache.layers[layer_idx].append(&k, &v).map_err(LuminaError::Candle)?;

        // GQA repeat KV if needed
        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;

        // Scaled dot product attention: q is [B, H, S_q, D], k is [B, H, S_k, D] -> att is [B, H, S_q, S_k]
        let scale = 1.0 / (self.config.head_dim as f64).sqrt();
        let k_t = k.transpose(2, 3).map_err(LuminaError::Candle)?;
        let att = (q.matmul(&k_t)? * scale).map_err(LuminaError::Candle)?;

        // Causal mask for prefill
        let att = if seq_len > 1 {
            let mask = self.causal_mask(seq_len)?;
            att.broadcast_add(&mask).map_err(LuminaError::Candle)?
        } else {
            att
        };

        let att = candle_nn::ops::softmax_last_dim(&att).map_err(LuminaError::Candle)?;
        let out = att.matmul(&v).map_err(LuminaError::Candle)?;
        let out = out.transpose(1, 2)?.reshape((b_sz, seq_len, self.config.hidden_size))?;
        let attn_out = Self::matmul_linear(&out, &o_w)?;

        let x = (x + attn_out).map_err(LuminaError::Candle)?;

        // Post-attention layernorm
        let post_norm_w = self.get_layer_weight(layer_idx, &["post_attention_layernorm.weight", "ffn_norm.weight", "post_attention_norm.weight"])?;
        let post_norm = RmsNorm::new(post_norm_w, self.config.rms_norm_eps);
        let normed_post_x = post_norm.forward(&x).map_err(LuminaError::Candle)?;

        // SwiGLU MLP
        let gate_w = self.get_layer_weight(layer_idx, &["mlp.gate_proj.weight", "ffn_gate.weight"])?;
        let up_w = self.get_layer_weight(layer_idx, &["mlp.up_proj.weight", "ffn_up.weight"])?;
        let down_w = self.get_layer_weight(layer_idx, &["mlp.down_proj.weight", "ffn_down.weight"])?;

        let gate = Self::matmul_linear(&normed_post_x, &gate_w)?;
        let up = Self::matmul_linear(&normed_post_x, &up_w)?;
        let silu_gate = candle_nn::ops::silu(&gate).map_err(LuminaError::Candle)?;
        let mlp_act = (silu_gate * up).map_err(LuminaError::Candle)?;
        let mlp_out = Self::matmul_linear(&mlp_act, &down_w)?;

        let x = (x + mlp_out).map_err(LuminaError::Candle)?;
        Ok(x)
    }

    fn matmul_linear(x: &Tensor, w: &Tensor) -> Result<Tensor> {
        let last_x_dim = x.dim(candle_core::D::Minus1).map_err(LuminaError::Candle)?;
        let w_dims = w.shape().dims2().map_err(LuminaError::Candle)?;
        let w_adapted = if w_dims.1 == last_x_dim {
            // w is [out_dim, in_dim] -> transpose to [in_dim, out_dim]
            w.t().map_err(LuminaError::Candle)?
        } else if w_dims.0 == last_x_dim {
            // w is already [in_dim, out_dim]
            w.clone()
        } else {
            return Err(LuminaError::Config(format!(
                "matmul dimension mismatch: x last dim {} vs w dims {:?}",
                last_x_dim, w_dims
            )));
        };

        let x_dims = x.shape().dims();
        if x_dims.len() == 3 {
            let (b, s, in_d) = (x_dims[0], x_dims[1], x_dims[2]);
            let x_2d = x.reshape((b * s, in_d)).map_err(LuminaError::Candle)?;
            let out_2d = x_2d.matmul(&w_adapted).map_err(LuminaError::Candle)?;
            let out_dim = w_adapted.dim(1).map_err(LuminaError::Candle)?;
            out_2d.reshape((b, s, out_dim)).map_err(LuminaError::Candle)
        } else {
            x.matmul(&w_adapted).map_err(LuminaError::Candle)
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
        Ok(x)
    }

    fn apply_rope(&self, q: &Tensor, k: &Tensor, pos: usize, seq_len: usize) -> Result<(Tensor, Tensor)> {
        let head_dim = self.config.head_dim;
        let positions: Vec<f32> = (pos..pos + seq_len).map(|p| p as f32).collect();
        let pos_tensor = Tensor::new(&positions[..], &self.device)?.unsqueeze(1)?; // [S, 1]

        let half_dim = head_dim / 2;
        let theta = self.config.rope_theta;
        let freqs: Vec<f32> = (0..half_dim).map(|i| 1.0f32 / theta.powf((2 * i) as f32 / head_dim as f32)).collect();
        let freq_tensor = Tensor::new(&freqs[..], &self.device)?.unsqueeze(0)?; // [1, half_dim]

        let angles = pos_tensor.matmul(&freq_tensor).map_err(LuminaError::Candle)?; // [S, half_dim]
        let cos = angles.cos().map_err(LuminaError::Candle)?;
        let sin = angles.sin().map_err(LuminaError::Candle)?;

        // cos/sin: [1, 1, seq_len, head_dim] matching q/k shape [B, num_heads, seq_len, head_dim]
        let cos = Tensor::cat(&[&cos, &cos], 1)?.unsqueeze(0)?.unsqueeze(0)?.to_dtype(q.dtype())?;
        let sin = Tensor::cat(&[&sin, &sin], 1)?.unsqueeze(0)?.unsqueeze(0)?.to_dtype(q.dtype())?;

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

        Ok((q_out, k_out))
    }

    fn causal_mask(&self, seq_len: usize) -> Result<Tensor> {
        let mut mask_vec = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                if j > i {
                    mask_vec[i * seq_len + j] = f32::NEG_INFINITY;
                }
            }
        }
        let mask = Tensor::new(&mask_vec[..], &self.device)?.reshape((1, 1, seq_len, seq_len))?.to_dtype(self.dtype)?;
        Ok(mask)
    }

    fn get_weight_or_fallback(&self, candidates: &[&str]) -> Result<Tensor> {
        for name in candidates {
            if let Some(t) = self.weights_cache.get(*name) {
                return Ok(t.clone());
            }
            if self.weights.contains(name) {
                return self.weights.get_tensor(name, &self.device, self.dtype);
            }
        }
        Err(LuminaError::MissingWeight(format!(
            "None of candidate weights {:?} found in archive",
            candidates
        )))
    }

    fn get_layer_weight(&self, layer_idx: usize, suffixes: &[&str]) -> Result<Tensor> {
        let prefixes = [
            format!("model.layers.{layer_idx}."),
            format!("layers.{layer_idx}."),
            format!("blk.{layer_idx}."),
        ];
        for prefix in &prefixes {
            for suffix in suffixes {
                let key = format!("{prefix}{suffix}");
                if let Some(t) = self.weights_cache.get(&key) {
                    return Ok(t.clone());
                }
                if self.weights.contains(&key) {
                    return self.weights.get_tensor(&key, &self.device, self.dtype);
                }
            }
        }
        Err(LuminaError::MissingWeight(format!(
            "Weight for layer {} with suffixes {:?} not found",
            layer_idx, suffixes
        )))
    }
}
