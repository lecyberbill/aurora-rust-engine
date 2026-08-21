// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Native SDXL CLIP-L and OpenCLIP-G Text Models with Penultimate Layer Extraction

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{embedding, layer_norm, linear, Embedding, LayerNorm, Linear, VarBuilder};
use tokenizers::Tokenizer;
use std::path::Path;

/// CLIP Multi-Head Self Attention with Causal Mask
#[derive(Debug)]
pub struct ClipAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl ClipAttention {
    pub fn new(vb: VarBuilder, embed_dim: usize, num_heads: usize) -> Result<Self> {
        let q_proj = linear(embed_dim, embed_dim, vb.pp("q_proj"))?;
        let k_proj = linear(embed_dim, embed_dim, vb.pp("k_proj"))?;
        let v_proj = linear(embed_dim, embed_dim, vb.pp("v_proj"))?;
        let out_proj = linear(embed_dim, embed_dim, vb.pp("out_proj"))?;
        let head_dim = embed_dim / num_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
            scale,
        })
    }

    pub fn forward(&self, xs: &Tensor, causal_mask: Option<&Tensor>) -> Result<Tensor> {
        let (b_sz, seq_len, _) = xs.dims3()?;

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let q = q.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;

        let mut attn_weights = (q.matmul(&k.transpose(2, 3)?)? * self.scale)?;
        if let Some(mask) = causal_mask {
            attn_weights = attn_weights.broadcast_add(mask)?;
        }
        let attn_probs = candle_nn::ops::softmax(&attn_weights, candle_core::D::Minus1)?;
        let attn_output = attn_probs.matmul(&v)?;

        let attn_output = attn_output.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?;
        self.out_proj.forward(&attn_output)
    }
}

/// CLIP MLP: Linear -> QuickGELU -> Linear
#[derive(Debug)]
pub struct ClipMlp {
    fc1: Linear,
    fc2: Linear,
}

impl ClipMlp {
    pub fn new(vb: VarBuilder, in_dim: usize, hidden_dim: usize) -> Result<Self> {
        let fc1 = linear(in_dim, hidden_dim, vb.pp("fc1"))?;
        let fc2 = linear(hidden_dim, in_dim, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = self.fc1.forward(xs)?;
        // QuickGELU: x * sigmoid(1.702 * x)
        let qgelu = (&h * candle_nn::ops::sigmoid(&(&h * 1.702)?)?)?;
        self.fc2.forward(&qgelu)
    }
}

/// Single CLIP Encoder Layer
#[derive(Debug)]
pub struct ClipEncoderLayer {
    self_attn: ClipAttention,
    layer_norm1: LayerNorm,
    mlp: ClipMlp,
    layer_norm2: LayerNorm,
}

impl ClipEncoderLayer {
    pub fn new(vb: VarBuilder, embed_dim: usize, num_heads: usize, mlp_dim: usize) -> Result<Self> {
        let layer_norm1 = layer_norm(embed_dim, 1e-5, vb.pp("layer_norm1"))?;
        let self_attn = ClipAttention::new(vb.pp("self_attn"), embed_dim, num_heads)?;
        let layer_norm2 = layer_norm(embed_dim, 1e-5, vb.pp("layer_norm2"))?;
        let mlp = ClipMlp::new(vb.pp("mlp"), embed_dim, mlp_dim)?;

        Ok(Self {
            self_attn,
            layer_norm1,
            mlp,
            layer_norm2,
        })
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

/// Causal mask generator for 77 tokens
pub(crate) fn build_causal_mask(seq_len: usize, dev: &Device, dtype: DType) -> Result<Tensor> {
    let mut mask = vec![0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask[i * seq_len + j] = -1e4;
        }
    }
    Tensor::from_vec(mask, (1, 1, seq_len, seq_len), dev)?.to_dtype(dtype)
}

/// SDXL CLIP-L Text Encoder (ViT-L/14: dim=768, layers=12, heads=12, intermediate=3072)
pub struct ClipTextEncoder {
    token_embedding: Embedding,
    position_embedding: Embedding,
    layers: Vec<ClipEncoderLayer>,
    causal_mask: Tensor,
    tokenizer: Option<Tokenizer>,
    device: Device,
}

impl ClipTextEncoder {
    pub fn new_sd15(vb: VarBuilder) -> Result<Self> {
        let dev = vb.device().clone();
        let dtype = vb.dtype();
        let embed_dim = 768;
        let num_layers = 12;
        let num_heads = 12;
        let intermediate_size = 3072;

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
            let layer = ClipEncoderLayer::new(enc_vb.pp(i), embed_dim, num_heads, intermediate_size)?;
            layers.push(layer);
        }

        let causal_mask = build_causal_mask(77, &dev, dtype)?;

        Ok(Self {
            token_embedding,
            position_embedding,
            layers,
            causal_mask,
            tokenizer: None,
            device: dev,
        })
    }

    pub fn load_tokenizer<P: AsRef<Path>>(&mut self, path: P) -> crate::error::Result<()> {
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|e| crate::error::LuminaError::Tokenizer(e.to_string()))?;
        self.tokenizer = Some(tokenizer);
        Ok(())
    }

    /// Encode prompt into Penultimate Layer hidden states [1, 77, 768] (Layer 11 / index 10)
    pub fn encode_prompt(&self, prompt: &str) -> crate::error::Result<Tensor> {
        let tokenizer = self.tokenizer.as_ref()
            .ok_or_else(|| crate::error::LuminaError::Tokenizer("CLIP Tokenizer not initialized".to_string()))?;

        let encoding = tokenizer.encode(prompt, true)
            .map_err(|e| crate::error::LuminaError::Tokenizer(e.to_string()))?;

        let mut tokens = encoding.get_ids().to_vec();
        let max_len = 77;
        if tokens.len() > max_len {
            tokens.truncate(max_len);
            if let Some(last) = tokens.last_mut() { *last = 49407; }
        } else {
            tokens.resize(max_len, 49407);
        }

        let input_ids = Tensor::from_slice(&tokens, (1, max_len), &self.device)?;
        let positions = Tensor::arange(0u32, max_len as u32, &self.device)?.unsqueeze(0)?;

        let tok_emb = self.token_embedding.forward(&input_ids)?;
        let pos_emb = self.position_embedding.forward(&positions)?;
        let mut hidden_states = (tok_emb + pos_emb)?;

        // Execute through Penultimate Layer (SDXL uses layer index 10 out of 12)
        let penultimate_layer_idx = self.layers.len() - 2; // index 10 (11th layer)
        for (idx, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward(&hidden_states, Some(&self.causal_mask))?;
            if idx == penultimate_layer_idx {
                break;
            }
        }

        Ok(hidden_states)
    }
}
