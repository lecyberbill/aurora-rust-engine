// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Pure Rust ChatTTS Conversational Engine (GPT + DVAE + Vocos)

use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Embedding, LayerNorm, LayerNormConfig, Linear, VarBuilder};
use candle_transformers::generation::LogitsProcessor;
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::audio::{Vocos, VocosConfig, WavAudio};

/// Hyperparameters for ChatTTS models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatTtsConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub vocab_size: usize,
    pub num_codebooks: usize,
    pub codebook_size: usize,
    pub spk_emb_dim: usize,
    pub dvae_dim: usize,
    pub dvae_layers: usize,
    pub vocos: VocosConfig,
}

impl Default for ChatTtsConfig {
    fn default() -> Self {
        Self {
            hidden_size: 768,
            num_layers: 16,
            num_heads: 12,
            vocab_size: 32000,
            num_codebooks: 1,
            codebook_size: 2048,
            spk_emb_dim: 768,
            dvae_dim: 512,
            dvae_layers: 6,
            vocos: VocosConfig::default(),
        }
    }
}

/// Rotary Position Embedding for ChatTTS GPT layers.
pub struct ChatTtsRoPE {
    sin: Tensor,
    cos: Tensor,
}

impl ChatTtsRoPE {
    pub fn new(head_dim: usize, max_len: usize, device: &Device, dtype: DType) -> Result<Self> {
        let half = head_dim / 2;
        let mut inv_freq = Vec::with_capacity(half);
        for i in 0..half {
            let theta = 1.0 / 10000.0f32.powf((2 * i) as f32 / head_dim as f32);
            inv_freq.push(theta);
        }
        let inv_freq_t = Tensor::from_vec(inv_freq, (1, half), device)?;
        let t = Tensor::arange(0u32, max_len as u32, device)?
            .to_dtype(DType::F32)?
            .unsqueeze(1)?;
        let freqs = t.matmul(&inv_freq_t)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;

        Ok(Self { sin, cos })
    }

    pub fn apply(&self, x: &Tensor, offset: usize) -> candle_core::Result<Tensor> {
        let (_b, _h, seq_len, head_dim) = x.dims4()?;
        let half = head_dim / 2;
        let sin = self.sin.narrow(0, offset, seq_len)?.unsqueeze(0)?.unsqueeze(0)?;
        let cos = self.cos.narrow(0, offset, seq_len)?.unsqueeze(0)?.unsqueeze(0)?;

        let x1 = x.narrow(3, 0, half)?;
        let x2 = x.narrow(3, half, half)?;

        let rotated_x1 = (x1.broadcast_mul(&cos)?).sub(&x2.broadcast_mul(&sin)?)?;
        let rotated_x2 = (x1.broadcast_mul(&sin)?).add(&x2.broadcast_mul(&cos)?)?;

        Tensor::cat(&[&rotated_x1, &rotated_x2], 3)
    }
}

/// Attention layer with KV-caching.
pub struct ChatTtsAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl ChatTtsAttention {
    pub fn load(hidden_size: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = hidden_size / num_heads;
        let q_proj = candle_nn::linear(hidden_size, hidden_size, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear(hidden_size, hidden_size, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear(hidden_size, hidden_size, vb.pp("v_proj"))?;
        let out_proj = candle_nn::linear(hidden_size, hidden_size, vb.pp("out_proj"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
            kv_cache: None,
        })
    }

    pub fn clear_cache(&mut self) {
        self.kv_cache = None;
    }

    pub fn forward(&mut self, x: &Tensor, rope: &ChatTtsRoPE, offset: usize) -> candle_core::Result<Tensor> {
        let (b, seq_len, _d) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape((b, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((b, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;

        let q = rope.apply(&q, offset)?;
        let k = rope.apply(&k, offset)?;

        let (k, v) = match &self.kv_cache {
            Some((prev_k, prev_v)) => {
                let new_k = Tensor::cat(&[prev_k, &k], 2)?;
                let new_v = Tensor::cat(&[prev_v, &v], 2)?;
                (new_k, new_v)
            }
            None => (k, v),
        };

        self.kv_cache = Some((k.clone(), v.clone()));

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let total_seq = k.dim(2)?;

        let attn_weights = if seq_len > 1 {
            let mask = causal_mask(seq_len, total_seq, x.device())?;
            attn_weights.broadcast_add(&mask)?
        } else {
            attn_weights
        };

        let attn_probs = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let output = attn_probs.matmul(&v)?; // [B, num_heads, seq_len, head_dim]
        let output = output.transpose(1, 2)?.reshape((b, seq_len, self.num_heads * self.head_dim))?;
        self.out_proj.forward(&output)
    }
}

fn causal_mask(query_len: usize, key_len: usize, device: &Device) -> candle_core::Result<Tensor> {
    let mut mask_vec = vec![0.0f32; query_len * key_len];
    for q in 0..query_len {
        for k in 0..key_len {
            if k > (key_len - query_len) + q {
                mask_vec[q * key_len + k] = -1e9;
            }
        }
    }
    Tensor::from_vec(mask_vec, (1, 1, query_len, key_len), device)
}

/// ChatTTS Transformer Block (Self-Attention + MLP).
pub struct ChatTtsBlock {
    attn: ChatTtsAttention,
    attn_norm: LayerNorm,
    mlp_in: Linear,
    mlp_out: Linear,
    mlp_norm: LayerNorm,
}

impl ChatTtsBlock {
    pub fn load(hidden_size: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let ln_cfg = LayerNormConfig { eps: 1e-5, ..Default::default() };
        let attn_norm = candle_nn::layer_norm(hidden_size, ln_cfg, vb.pp("attn_norm"))?;
        let attn = ChatTtsAttention::load(hidden_size, num_heads, vb.pp("attn"))?;
        let mlp_norm = candle_nn::layer_norm(hidden_size, ln_cfg, vb.pp("mlp_norm"))?;
        let intermediate = hidden_size * 4;
        let mlp_in = candle_nn::linear(hidden_size, intermediate, vb.pp("mlp_in"))?;
        let mlp_out = candle_nn::linear(intermediate, hidden_size, vb.pp("mlp_out"))?;

        Ok(Self {
            attn,
            attn_norm,
            mlp_in,
            mlp_out,
            mlp_norm,
        })
    }

    pub fn forward(&mut self, x: &Tensor, rope: &ChatTtsRoPE, offset: usize) -> candle_core::Result<Tensor> {
        let norm_x = self.attn_norm.forward(x)?;
        let attn_out = self.attn.forward(&norm_x, rope, offset)?;
        let x = x.add(&attn_out)?;

        let norm_mlp = self.mlp_norm.forward(&x)?;
        let h = self.mlp_in.forward(&norm_mlp)?;
        let h = h.gelu()?;
        let mlp_out = self.mlp_out.forward(&h)?;
        x.add(&mlp_out)
    }

    pub fn clear_cache(&mut self) {
        self.attn.clear_cache();
    }
}

/// ChatTTS Autoregressive GPT Model for predicting prosody and acoustic tokens.
pub struct ChatTtsGpt {
    text_embed: Embedding,
    code_embed: Embedding,
    spk_proj: Linear,
    blocks: Vec<ChatTtsBlock>,
    final_norm: LayerNorm,
    #[allow(dead_code)]
    lm_head: Linear,
    code_head: Linear,
    rope: ChatTtsRoPE,
}

impl ChatTtsGpt {
    pub fn load(cfg: &ChatTtsConfig, vb: VarBuilder, device: &Device, dtype: DType) -> Result<Self> {
        let text_embed = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("text_embed"))?;
        let code_embed = candle_nn::embedding(cfg.codebook_size, cfg.hidden_size, vb.pp("code_embed"))?;
        let spk_proj = candle_nn::linear(cfg.spk_emb_dim, cfg.hidden_size, vb.pp("spk_proj"))?;

        let mut blocks = Vec::with_capacity(cfg.num_layers);
        let blocks_vb = vb.pp("layers");
        for i in 0..cfg.num_layers {
            blocks.push(ChatTtsBlock::load(cfg.hidden_size, cfg.num_heads, blocks_vb.pp(i))?);
        }

        let ln_cfg = LayerNormConfig { eps: 1e-5, ..Default::default() };
        let final_norm = candle_nn::layer_norm(cfg.hidden_size, ln_cfg, vb.pp("final_norm"))?;
        let lm_head = candle_nn::linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
        let code_head = candle_nn::linear(cfg.hidden_size, cfg.codebook_size, vb.pp("code_head"))?;

        let head_dim = cfg.hidden_size / cfg.num_heads;
        let rope = ChatTtsRoPE::new(head_dim, 4096, device, dtype)?;

        Ok(Self {
            text_embed,
            code_embed,
            spk_proj,
            blocks,
            final_norm,
            lm_head,
            code_head,
            rope,
        })
    }

    pub fn clear_cache(&mut self) {
        for block in &mut self.blocks {
            block.clear_cache();
        }
    }

    /// Autoregressive token sampling loop generating discrete acoustic codes.
    pub fn generate_codes(
        &mut self,
        text_tokens: &[u32],
        spk_emb: &Tensor,
        max_steps: usize,
        temperature: f64,
        seed: u64,
    ) -> Result<Vec<u32>> {
        self.clear_cache();
        let device = spk_emb.device();

        // 1. Embed text tokens and speaker embedding
        let text_tensor = Tensor::new(text_tokens, device)?.unsqueeze(0)?;
        let text_emb = self.text_embed.forward(&text_tensor)?; // [1, T, D]
        let spk_cond = self.spk_proj.forward(spk_emb)?.unsqueeze(1)?; // [1, 1, D]

        let mut input_emb = Tensor::cat(&[&spk_cond, &text_emb], 1)?;
        let mut offset = 0;
        let mut generated_codes = Vec::new();
        let mut lp = LogitsProcessor::new(seed, Some(temperature), Some(0.9));

        for step in 0..max_steps {
            let mut h = input_emb.clone();
            for block in &mut self.blocks {
                h = block.forward(&h, &self.rope, offset)?;
            }
            offset += input_emb.dim(1)?;

            let h = self.final_norm.forward(&h)?;
            let last_token_h = h.narrow(1, h.dim(1)? - 1, 1)?; // [1, 1, D]

            let code_logits = self.code_head.forward(&last_token_h)?.squeeze(0)?.squeeze(0)?;
            let sampled_token = lp.sample(&code_logits)?;

            // 0 is the EOS codebook token for ChatTTS
            if sampled_token == 0 && step > 10 {
                break;
            }

            generated_codes.push(sampled_token);

            let next_code_tensor = Tensor::new(&[sampled_token], device)?.unsqueeze(0)?;
            input_emb = self.code_embed.forward(&next_code_tensor)?;
        }

        self.clear_cache();
        Ok(generated_codes)
    }
}

/// ChatTTS Discrete Variational Autoencoder (DVAE) Decoder.
pub struct ChatTtsDvae {
    code_embed: Embedding,
    in_proj: Linear,
    layers: Vec<Linear>,
    out_proj: Linear,
}

impl ChatTtsDvae {
    pub fn load(cfg: &ChatTtsConfig, vb: VarBuilder) -> Result<Self> {
        let code_embed = candle_nn::embedding(cfg.codebook_size, cfg.dvae_dim, vb.pp("code_embed"))?;
        let in_proj = candle_nn::linear(cfg.dvae_dim, cfg.dvae_dim, vb.pp("in_proj"))?;

        let mut layers = Vec::with_capacity(cfg.dvae_layers);
        let layers_vb = vb.pp("layers");
        for i in 0..cfg.dvae_layers {
            layers.push(candle_nn::linear(cfg.dvae_dim, cfg.dvae_dim, layers_vb.pp(i))?);
        }

        let out_proj = candle_nn::linear(cfg.dvae_dim, cfg.vocos.input_dim, vb.pp("out_proj"))?;

        Ok(Self {
            code_embed,
            in_proj,
            layers,
            out_proj,
        })
    }

    /// Decode discrete acoustic tokens into continuous Vocos acoustic latents `[1, C, T]`.
    pub fn decode(&self, codes: &[u32], device: &Device) -> Result<Tensor> {
        let code_tensor = Tensor::new(codes, device)?.unsqueeze(0)?;
        let mut h = self.code_embed.forward(&code_tensor)?;
        h = self.in_proj.forward(&h)?;

        for layer in &self.layers {
            let res = h.clone();
            h = layer.forward(&h)?;
            h = h.gelu()?;
            h = res.add(&h)?;
        }

        let out = self.out_proj.forward(&h)?; // [1, T, input_dim]
        Ok(out.transpose(1, 2)?) // [1, input_dim, T]
    }
}

/// Unified ChatTTS Model containing GPT, DVAE, and Vocos.
pub struct ChatTtsModel {
    pub config: ChatTtsConfig,
    pub gpt: ChatTtsGpt,
    pub dvae: ChatTtsDvae,
    pub vocos: Vocos,
    pub device: Device,
}

impl ChatTtsModel {
    pub fn load_from_safetensors(
        gpt_path: &Path,
        dvae_path: &Path,
        vocos_path: &Path,
        config: ChatTtsConfig,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let gpt_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[gpt_path], dtype, &device)
                .with_context(|| format!("Failed to load ChatTTS GPT weights from {:?}", gpt_path))?
        };
        let dvae_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[dvae_path], dtype, &device)
                .with_context(|| format!("Failed to load ChatTTS DVAE weights from {:?}", dvae_path))?
        };
        let vocos_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[vocos_path], dtype, &device)
                .with_context(|| format!("Failed to load Vocos weights from {:?}", vocos_path))?
        };

        let gpt = ChatTtsGpt::load(&config, gpt_vb, &device, dtype)?;
        let dvae = ChatTtsDvae::load(&config, dvae_vb)?;
        let vocos = Vocos::load(config.vocos.clone(), vocos_vb)?;

        Ok(Self {
            config,
            gpt,
            dvae,
            vocos,
            device,
        })
    }

    /// Generate full audio waveform from text tokens and speaker embedding.
    pub fn synthesize(
        &mut self,
        text_tokens: &[u32],
        spk_emb: &Tensor,
        max_steps: usize,
        temperature: f64,
        seed: u64,
    ) -> Result<WavAudio> {
        let codes = self.gpt.generate_codes(text_tokens, spk_emb, max_steps, temperature, seed)
            .context("Failed during ChatTTS GPT code generation")?;

        anyhow::ensure!(!codes.is_empty(), "Generated 0 acoustic codes from prompt");

        let latents = self.dvae.decode(&codes, &self.device)
            .context("Failed during ChatTTS DVAE latent decoding")?;

        let waveform = self.vocos.decode(&latents)
            .context("Failed during Vocos waveform synthesis")?;

        Ok(WavAudio::new(waveform, self.config.vocos.sample_rate as u32, 1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chattts_rope_and_causal_mask() {
        let device = Device::Cpu;
        let rope = ChatTtsRoPE::new(64, 512, &device, DType::F32).unwrap();
        let q = Tensor::zeros((1, 8, 16, 64), DType::F32, &device).unwrap();
        let rotated = rope.apply(&q, 0).unwrap();
        assert_eq!(rotated.shape().dims(), &[1, 8, 16, 64]);

        let mask = causal_mask(10, 10, &device).unwrap();
        assert_eq!(mask.shape().dims(), &[1, 1, 10, 10]);
    }
}
