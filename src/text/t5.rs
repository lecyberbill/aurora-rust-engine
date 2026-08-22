// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Pure Rust T5-XXL Text Encoder for Flux.1 and SD 3.5 with CPU offloading

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::t5::{ActivationWithOptionalGating, Config as T5Config, T5EncoderModel};
use std::path::Path;
use tokenizers::Tokenizer;

/// High-level T5-XXL Text Encoder with automatic tokenization and CPU offloading support.
pub struct T5TextEncoder {
    model: T5EncoderModel,
    tokenizer: Tokenizer,
    device: Device,
    dtype: DType,
}

impl T5TextEncoder {
    pub fn new(vb: VarBuilder, tokenizer_path: Option<&Path>) -> Result<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();

        // T5-XXL standard architecture config (24 layers, 4096 hidden dim, 64 heads)
        let config = T5Config {
            d_model: 4096,
            d_kv: 64,
            d_ff: 10240,
            num_layers: 24,
            num_decoder_layers: None,
            num_heads: 64,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            dropout_rate: 0.0,
            layer_norm_epsilon: 1e-6,
            initializer_factor: 1.0,
            feed_forward_proj: ActivationWithOptionalGating {
                gated: true,
                activation: candle_nn::Activation::Gelu,
            },
            tie_word_embeddings: false,
            is_decoder: false,
            is_encoder_decoder: false,
            use_cache: false,
            pad_token_id: 0,
            eos_token_id: 1,
            vocab_size: 32128,
            decoder_start_token_id: None,
        };

        let model = T5EncoderModel::load(vb, &config)?;

        let tokenizer = if let Some(p) = tokenizer_path {
            Tokenizer::from_file(p).map_err(|e| candle_core::Error::Msg(e.to_string()))?
        } else {
            let api = hf_hub::api::sync::Api::new()
                .map_err(|e| candle_core::Error::Msg(format!("HF API error: {}", e)))?;
            let tokenizer_file = api.model("google-t5/t5-v1_1-xxl".to_string()).get("tokenizer.json")
                .map_err(|e| candle_core::Error::Msg(format!("Failed to fetch tokenizer: {}", e)))?;
            Tokenizer::from_file(tokenizer_file).map_err(|e| candle_core::Error::Msg(e.to_string()))?
        };

        Ok(Self {
            model,
            tokenizer,
            device,
            dtype,
        })
    }

    /// Encode prompt into [1, max_sequence_length, 4096] text embeddings
    pub fn encode(&mut self, prompt: &str, max_length: usize) -> Result<Tensor> {
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        let mut token_ids: Vec<u32> = encoding.get_ids().to_vec();
        if token_ids.len() > max_length {
            token_ids.truncate(max_length);
        } else {
            token_ids.resize(max_length, 0); // Pad with 0
        }

        let input_ids = Tensor::from_vec(
            token_ids.iter().map(|&x| x as i64).collect(),
            (1, max_length),
            &self.device,
        )?;

        let out = self.model.forward(&input_ids)?;
        out.to_dtype(self.dtype)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}
