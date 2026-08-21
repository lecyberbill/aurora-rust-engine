// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Native SDXL OpenCLIP-G Text Encoder with Penultimate Layer Extraction & Standard GELU

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{embedding, layer_norm, linear, Embedding, LayerNorm, Linear, VarBuilder};
use tokenizers::Tokenizer;
use std::path::Path;

use crate::text::clip::{build_causal_mask, ClipAttention};

/// OpenCLIP MLP: Linear -> standard GELU -> Linear
#[derive(Debug)]
pub struct OpenClipMlp {
    fc1: Linear,
    fc2: Linear,
}

impl OpenClipMlp {
    pub fn new(vb: VarBuilder, in_dim: usize, hidden_dim: usize) -> Result<Self> {
        let fc1 = linear(in_dim, hidden_dim, vb.pp("mlp.fc1"))?;
        let fc2 = linear(hidden_dim, in_dim, vb.pp("mlp.fc2"))?;
        Ok(Self { fc1, fc2 })
    }

    pub fn apply_lora_deltas(&mut self, prefix: &str, deltas: &std::collections::HashMap<String, Tensor>) -> Result<()> {
        crate::diffusion::attention::apply_linear_delta(&mut self.fc1, &format!("{}.mlp.fc1", prefix), deltas)?;
        crate::diffusion::attention::apply_linear_delta(&mut self.fc2, &format!("{}.mlp.fc2", prefix), deltas)?;
        Ok(())
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = self.fc1.forward(xs)?;
        let gelu = h.gelu_erf()?;
        self.fc2.forward(&gelu)
    }
}

/// OpenCLIP Encoder Layer with standard GELU
#[derive(Debug)]
pub struct OpenClipEncoderLayer {
    self_attn: ClipAttention,
    layer_norm1: LayerNorm,
    mlp: OpenClipMlp,
    layer_norm2: LayerNorm,
}

impl OpenClipEncoderLayer {
    pub fn new(vb: VarBuilder, embed_dim: usize, num_heads: usize, mlp_dim: usize) -> Result<Self> {
        let layer_norm1 = layer_norm(embed_dim, 1e-5, vb.pp("layer_norm1"))?;
        let self_attn = ClipAttention::new(vb.pp("self_attn"), embed_dim, num_heads)?;
        let layer_norm2 = layer_norm(embed_dim, 1e-5, vb.pp("layer_norm2"))?;
        let mlp = OpenClipMlp::new(vb.clone(), embed_dim, mlp_dim)?;

        Ok(Self {
            self_attn,
            layer_norm1,
            mlp,
            layer_norm2,
        })
    }

    pub fn apply_lora_deltas(&mut self, prefix: &str, deltas: &std::collections::HashMap<String, Tensor>) -> Result<()> {
        self.self_attn.apply_lora_deltas(&format!("{}.self_attn", prefix), deltas)?;
        self.mlp.apply_lora_deltas(prefix, deltas)?;
        Ok(())
    }

    pub fn forward(&self, xs: &Tensor, causal_mask: Option<&Tensor>) -> Result<Tensor> {
        let residual = xs;
        let normed = self.layer_norm1.forward(xs)?;
        let attn_out = self.self_attn.forward(&normed, causal_mask)?;
        let xs = (residual + attn_out)?;

        let residual = &xs;
        let normed = self.layer_norm2.forward(&xs)?;
        let mlp_out = self.mlp.forward(&normed)?;
        residual + mlp_out
    }
}

/// SDXL OpenCLIP-G Text Encoder (ViT-bigG/14: dim=1280, layers=32, heads=20, intermediate=5120)
pub struct OpenClipTextEncoder {
    token_embedding: Embedding,
    position_embedding: Embedding,
    layers: Vec<OpenClipEncoderLayer>,
    final_layer_norm: LayerNorm,
    text_projection: Option<Tensor>,
    causal_mask: Tensor,
    tokenizer: Option<Tokenizer>,
    device: Device,
}

impl OpenClipTextEncoder {
    pub fn new_sdxl(vb: VarBuilder) -> Result<Self> {
        let dev = vb.device().clone();
        let dtype = vb.dtype();
        let embed_dim = 1280;
        let num_layers = 32;
        let num_heads = 20;
        let intermediate_size = 5120;

        let emb_vb = if vb.contains_tensor("text_model.embeddings.token_embedding.weight") {
            vb.pp("text_model.embeddings")
        } else {
            vb.pp("embeddings")
        };
        let token_embedding = embedding(49408, embed_dim, emb_vb.pp("token_embedding"))?;
        let position_embedding = embedding(77, embed_dim, emb_vb.pp("position_embedding"))?;

        let enc_vb = if vb.contains_tensor("text_model.encoder.layers.0.self_attn.q_proj.weight") {
            vb.pp("text_model.encoder.layers")
        } else {
            vb.pp("encoder.layers")
        };

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let layer = OpenClipEncoderLayer::new(enc_vb.pp(i), embed_dim, num_heads, intermediate_size)?;
            layers.push(layer);
        }

        let ln_vb = if vb.contains_tensor("text_model.final_layer_norm.weight") {
            vb.pp("text_model.final_layer_norm")
        } else {
            vb.pp("final_layer_norm")
        };
        let final_layer_norm = layer_norm(embed_dim, 1e-5, ln_vb)?;

        let text_projection = if vb.contains_tensor("text_projection") {
            Some(vb.get((1280, 1280), "text_projection")?)
        } else {
            None
        };

        let causal_mask = build_causal_mask(77, &dev, dtype)?;

        Ok(Self {
            token_embedding,
            position_embedding,
            layers,
            final_layer_norm,
            text_projection,
            causal_mask,
            tokenizer: None,
            device: dev,
        })
    }

    pub fn apply_lora_deltas(&mut self, deltas: &std::collections::HashMap<String, Tensor>) -> Result<()> {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let p1 = format!("te2.text_model.encoder.layers.{}", i);
            let p2 = format!("text_model.encoder.layers.{}", i);
            layer.apply_lora_deltas(&p1, deltas)?;
            layer.apply_lora_deltas(&p2, deltas)?;
        }
        if let Some(proj) = &mut self.text_projection {
            if let Some(delta) = deltas.get("te2.text_model.text_projection")
                .or_else(|| deltas.get("text_model.text_projection"))
                .or_else(|| deltas.get("text_projection"))
            {
                let delta = delta.to_device(proj.device())?.to_dtype(proj.dtype())?;
                *proj = (proj.as_ref() + &delta)?;
            }
        }
        Ok(())
    }

    pub fn load_tokenizer<P: AsRef<Path>>(&mut self, path: P) -> crate::error::Result<()> {
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|e| crate::error::LuminaError::Tokenizer(e.to_string()))?;
        self.tokenizer = Some(tokenizer);
        Ok(())
    }

    /// Encode prompt into Penultimate Layer hidden states [1, 77, 1280] (Layer 31 / index 30) AND Pooled Vector [1, 1280]
    pub fn encode_prompt_with_pooled(&self, prompt: &str) -> crate::error::Result<(Tensor, Tensor)> {
        let tokenizer = self.tokenizer.as_ref()
            .ok_or_else(|| crate::error::LuminaError::Tokenizer("OpenCLIP Tokenizer not initialized".to_string()))?;

        let encoding = tokenizer.encode(prompt, true)
            .map_err(|e| crate::error::LuminaError::Tokenizer(e.to_string()))?;

        let mut tokens = encoding.get_ids().to_vec();
        let max_len = 77;
        let eos_pos = tokens.len().saturating_sub(1);
        if tokens.len() > max_len {
            tokens.truncate(max_len);
            if let Some(last) = tokens.last_mut() { *last = 49407; }
        } else {
            tokens.resize(max_len, 0); // OpenCLIP zero-padding
        }

        let input_ids = Tensor::from_slice(&tokens, (1, max_len), &self.device)?;
        let positions = Tensor::arange(0u32, max_len as u32, &self.device)?.unsqueeze(0)?;

        let tok_emb = self.token_embedding.forward(&input_ids)?;
        let pos_emb = self.position_embedding.forward(&positions)?;
        let mut hidden_states = (tok_emb + pos_emb)?;

        let mut penultimate_hidden_states: Option<Tensor> = None;
        let penultimate_layer_idx = self.layers.len() - 2; // index 30 (31st layer)

        for (idx, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward(&hidden_states, Some(&self.causal_mask))?;
            if idx == penultimate_layer_idx {
                penultimate_hidden_states = Some(hidden_states.clone());
            }
        }

        let penultimate_states = penultimate_hidden_states
            .ok_or_else(|| candle_core::Error::Msg("Failed to capture penultimate layer".into()))?;

        // Pooled embedding is extracted from the final layer output (layer 32) after final_layer_norm at eos_pos
        let final_normed = self.final_layer_norm.forward(&hidden_states)?;
        let eos_idx = eos_pos.min(max_len - 1);
        let eos_token_embed = final_normed.narrow(1, eos_idx, 1)?.squeeze(1)?; // [1, 1280]

        let pooled = if let Some(ref proj) = self.text_projection {
            eos_token_embed.matmul(proj)?
        } else {
            eos_token_embed
        };

        Ok((penultimate_states, pooled))
    }

    pub fn encode_prompt(&self, prompt: &str) -> crate::error::Result<Tensor> {
        let (hidden_states, _) = self.encode_prompt_with_pooled(prompt)?;
        Ok(hidden_states)
    }
}
