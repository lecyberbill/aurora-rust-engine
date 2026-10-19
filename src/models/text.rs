// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Text model wrapper (encoders and CausalLM text generation)

pub mod causal;
pub mod llama;
pub mod moe;
pub mod phi;
pub mod qwen;

use candle_core::{DType, Device, Tensor};
use crate::error::{LuminaError, Result};
use crate::text::{Mistral3TextEncoder, Qwen3TextEncoder, T5TextEncoder};
pub use causal::{CausalLM, CausalLMConfig, CausalLMPipeline, ChatMessage, ChatRole, TextGenParams};
use super::common::{AnyModel, EncodeModel, GenerationModel, ModelKind};

/// A text backend boxed into one type.
pub enum TextBackend {
    Qwen(Qwen3TextEncoder),
    Mistral(Mistral3TextEncoder),
    T5(T5TextEncoder),
    CausalLM(CausalLMPipeline),
    QuantizedGemma3 {
        model: crate::models::quantized_gemma3::ModelWeights,
        tokenizer: Option<tokenizers::Tokenizer>,
        device: Device,
    },
    QuantizedQwen3 {
        model: crate::models::quantized_qwen3::ModelWeights,
        tokenizer: Option<tokenizers::Tokenizer>,
        device: Device,
    },
    QuantizedQwen35 {
        model: crate::models::quantized_qwen35::ModelWeights,
        tokenizer: Option<tokenizers::Tokenizer>,
        device: Device,
    },
}

impl TextBackend {
    fn device(&self) -> &Device {
        match self {
            TextBackend::Qwen(_) | TextBackend::Mistral(_) | TextBackend::T5(_) => &Device::Cpu,
            TextBackend::CausalLM(c) => &c.device,
            TextBackend::QuantizedGemma3 { device, .. } => device,
            TextBackend::QuantizedQwen3 { device, .. } => device,
            TextBackend::QuantizedQwen35 { device, .. } => device,
        }
    }

    fn dtype(&self) -> DType {
        match self {
            TextBackend::Qwen(_) | TextBackend::Mistral(_) | TextBackend::T5(_) => DType::F16,
            TextBackend::CausalLM(c) => c.dtype,
            TextBackend::QuantizedGemma3 { .. } => DType::F32,
            TextBackend::QuantizedQwen3 { .. } => DType::F32,
            TextBackend::QuantizedQwen35 { .. } => DType::F32,
        }
    }
}

pub struct TextModel {
    id: String,
    family: String,
    inner: TextBackend,
}

impl TextModel {
    pub fn qwen(id: String, encoder: Qwen3TextEncoder) -> Self {
        Self { id, family: "qwen3".into(), inner: TextBackend::Qwen(encoder) }
    }
    pub fn mistral(id: String, encoder: Mistral3TextEncoder) -> Self {
        Self { id, family: "mistral3".into(), inner: TextBackend::Mistral(encoder) }
    }
    pub fn t5(id: String, encoder: T5TextEncoder) -> Self {
        Self { id, family: "t5".into(), inner: TextBackend::T5(encoder) }
    }
    pub fn causal_lm(id: String, family: String, pipeline: CausalLMPipeline) -> Self {
        Self { id, family, inner: TextBackend::CausalLM(pipeline) }
    }
    pub fn quantized_gemma3(id: String, model: crate::models::quantized_gemma3::ModelWeights, tokenizer: Option<tokenizers::Tokenizer>, device: Device) -> Self {
        Self { id, family: "gemma".into(), inner: TextBackend::QuantizedGemma3 { model, tokenizer, device } }
    }
    pub fn quantized_qwen3(id: String, model: crate::models::quantized_qwen3::ModelWeights, tokenizer: Option<tokenizers::Tokenizer>, device: Device) -> Self {
        Self { id, family: "qwen3".into(), inner: TextBackend::QuantizedQwen3 { model, tokenizer, device } }
    }
    pub fn quantized_qwen35(id: String, model: crate::models::quantized_qwen35::ModelWeights, tokenizer: Option<tokenizers::Tokenizer>, device: Device) -> Self {
        Self { id, family: "qwen3".into(), inner: TextBackend::QuantizedQwen35 { model, tokenizer, device } }
    }

    pub fn as_causal_lm_mut(&mut self) -> Option<&mut CausalLMPipeline> {
        match &mut self.inner {
            TextBackend::CausalLM(ref mut p) => Some(p),
            _ => None,
        }
    }
}

impl AnyModel for TextModel {
    fn kind(&self) -> ModelKind {
        match &self.inner {
            TextBackend::CausalLM(_) => ModelKind::Text,
            TextBackend::QuantizedGemma3 { .. } => ModelKind::Text,
            TextBackend::QuantizedQwen3 { .. } => ModelKind::Text,
            TextBackend::QuantizedQwen35 { .. } => ModelKind::Text,
            _ => ModelKind::TextEncoder,
        }
    }
    fn family(&self) -> &str { &self.family }
    fn id(&self) -> &str { &self.id }
    fn device(&self) -> &Device { self.inner.device() }
    fn dtype(&self) -> DType { self.inner.dtype() }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

impl EncodeModel for TextModel {
    fn encode(&mut self, prompt: &str, max_len: usize) -> Result<Tensor> {
        match &mut self.inner {
            TextBackend::Qwen(e) => e.encode(prompt, max_len).map_err(LuminaError::Candle),
            TextBackend::Mistral(e) => e.encode(prompt, max_len).map_err(LuminaError::Candle),
            TextBackend::T5(e) => e.encode(prompt, max_len).map_err(LuminaError::Candle),
            _ => Err(LuminaError::UnsupportedOp("CausalLM is for generation; use TextEncoder for conditioning".into())),
        }
    }
}

impl GenerationModel for TextModel {
    fn generate(&mut self, prompt: &str, max_tokens: usize, temperature: f64) -> Result<String> {
        let (thinking, resp, _) = self.generate_with_reasoning(prompt, max_tokens, temperature, 0.5)?;
        if resp.is_empty() {
            Ok(thinking)
        } else if thinking.is_empty() {
            Ok(resp)
        } else {
            Ok(format!("<think>\n{}\n</think>\n\n{}", thinking, resp))
        }
    }

    fn generate_with_reasoning(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
        _reasoning_effort: f64,
    ) -> Result<(String, String, f64)> {
        match &mut self.inner {
            TextBackend::CausalLM(pipeline) => {
                let mut params = TextGenParams::default();
                params.max_tokens = max_tokens;
                params.temperature = temperature;
                if let Ok(n_str) = std::env::var("N_GPU_LAYERS") {
                    if let Ok(n) = n_str.parse::<usize>() {
                        params.n_gpu_layers = Some(n);
                    }
                }

                // If prompt doesn't already contain chat markers, wrap it in family chat template
                let formatted_prompt = if !prompt.contains("<start_of_turn>")
                    && !prompt.contains("<|im_start|>")
                    && !prompt.contains("<|start_header_id|>")
                {
                    let family_lower = self.family.to_lowercase();
                    if family_lower.contains("gemma") {
                        format!("<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n", prompt.trim())
                    } else if family_lower.contains("llama") || family_lower.contains("deepseek") {
                        format!("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", prompt.trim())
                    } else {
                        format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", prompt.trim())
                    }
                } else {
                    prompt.to_string()
                };

                let text = pipeline.generate(&formatted_prompt, &params)?;
                Ok((String::new(), text, 0.0))
            }
            TextBackend::QuantizedGemma3 { model, tokenizer, device } => {
                let tok = tokenizer.as_ref().ok_or_else(|| {
                    LuminaError::Config("Tokenizer not attached to QuantizedGemma3 model".into())
                })?;

                let formatted_prompt = if !prompt.contains("<start_of_turn>") {
                    format!("<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n", prompt.trim())
                } else {
                    prompt.to_string()
                };

                let tokens = tok.encode(formatted_prompt.as_str(), true).map_err(|e| {
                    LuminaError::Context {
                        context: "Failed to tokenize prompt for Gemma 3".into(),
                        source: Box::new(LuminaError::Candle(candle_core::Error::Msg(e.to_string()))),
                    }
                })?.get_ids().to_vec();

                let input = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;
                let logits = model.forward(&input, 0)?;
                let last = logits.squeeze(0)?;
                let l_vec = last.to_dtype(DType::F32)?.to_vec1::<f32>()?;

                let mut top: Vec<_> = l_vec.iter().enumerate().collect();
                top.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
                let mut curr_tok = top[0].0 as u32;
                let mut generated = vec![curr_tok];

                for i in 0..max_tokens {
                    if curr_tok == 1 || curr_tok == 106 { break; }
                    let inp = Tensor::new(&[curr_tok], device)?.unsqueeze(0)?;
                    let step_logits = model.forward(&inp, tokens.len() + i)?;
                    let step_last = step_logits.squeeze(0)?;
                    let step_vec = step_last.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                    let mut t: Vec<_> = step_vec.iter().enumerate().collect();
                    t.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
                    curr_tok = t[0].0 as u32;
                    generated.push(curr_tok);
                }

                let text = tok.decode(&generated, true).map_err(|e| {
                    LuminaError::Context {
                        context: "Failed to decode generated token stream".into(),
                        source: Box::new(LuminaError::Candle(candle_core::Error::Msg(e.to_string()))),
                    }
                })?;
                Ok((String::new(), text.trim().to_string(), 0.0))
            }
            TextBackend::QuantizedQwen3 { model, tokenizer, device } => {
                let tok = tokenizer.as_ref().ok_or_else(|| {
                    LuminaError::Config("Tokenizer not attached to QuantizedQwen3 model".into())
                })?;

                let formatted_prompt = if prompt.contains("<|im_start|>") {
                    prompt.to_string()
                } else {
                    format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", prompt.trim())
                };

                let tokens = tok.encode(formatted_prompt.as_str(), true).map_err(|e| {
                    LuminaError::Context {
                        context: "Failed to tokenize prompt for Qwen3".into(),
                        source: Box::new(LuminaError::Candle(candle_core::Error::Msg(e.to_string()))),
                    }
                })?.get_ids().to_vec();

                model.clear_kv_cache();
                let input = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;
                let logits = model.forward(&input, 0)?;
                let l_vec = logits.squeeze(0)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;

                let mut top: Vec<_> = l_vec.iter().enumerate().collect();
                top.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
                let mut curr_tok = top[0].0 as u32;
                let mut generated = vec![curr_tok];

                for i in 0..max_tokens {
                    // Qwen3 stop tokens: <|endoftext|>=151643, <|im_end|>=151645
                    if curr_tok == 151643 || curr_tok == 151645 {
                        break;
                    }
                    let inp = Tensor::new(&[curr_tok], device)?.unsqueeze(0)?;
                    let step_logits = model.forward(&inp, tokens.len() + i)?;
                    let step_vec = step_logits.squeeze(0)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                    let mut t: Vec<_> = step_vec.iter().enumerate().collect();
                    t.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
                    curr_tok = t[0].0 as u32;
                    generated.push(curr_tok);
                }

                let text = tok.decode(&generated, true).map_err(|e| {
                    LuminaError::Context {
                        context: "Failed to decode generated token stream".into(),
                        source: Box::new(LuminaError::Candle(candle_core::Error::Msg(e.to_string()))),
                    }
                })?;
                Ok((String::new(), text.trim().to_string(), 0.0))
            }
            TextBackend::QuantizedQwen35 { model, tokenizer, device } => {
                let tok = tokenizer.as_ref().ok_or_else(|| {
                    LuminaError::Config("Tokenizer not attached to QuantizedQwen35 model".into())
                })?;

                let formatted_prompt = if prompt.contains("<|im_start|>") {
                    prompt.to_string()
                } else {
                    format!("<|im_start|>user\n{} /no_think<|im_end|>\n<|im_start|>assistant\n", prompt.trim())
                };

                let tokens = tok.encode(formatted_prompt.as_str(), true).map_err(|e| {
                    LuminaError::Context {
                        context: "Failed to tokenize prompt for Qwen3.8".into(),
                        source: Box::new(LuminaError::Candle(candle_core::Error::Msg(e.to_string()))),
                    }
                })?.get_ids().to_vec();

                model.reset_cache();
                let input = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;
                let logits = model.forward(&input, 0)?;
                let l_vec = logits.squeeze(0)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;

                let mut top: Vec<_> = l_vec.iter().enumerate().collect();
                top.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
                let mut curr_tok = top[0].0 as u32;
                let mut generated = vec![curr_tok];

                let stop_ids: Vec<u32> = [
                    model.eos_token_id(),
                    tok.token_to_id("<|im_end|>"),
                    tok.token_to_id("<|endoftext|>"),
                ]
                .into_iter()
                .flatten()
                .collect();

                for i in 0..max_tokens {
                    if stop_ids.contains(&curr_tok) {
                        break;
                    }
                    let inp = Tensor::new(&[curr_tok], device)?.unsqueeze(0)?;
                    let step_logits = model.forward(&inp, tokens.len() + i)?;
                    let step_vec = step_logits.squeeze(0)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                    let mut t: Vec<_> = step_vec.iter().enumerate().collect();
                    t.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
                    curr_tok = t[0].0 as u32;
                    generated.push(curr_tok);
                }

                let text = tok.decode(&generated, true).map_err(|e| {
                    LuminaError::Context {
                        context: "Failed to decode generated token stream".into(),
                        source: Box::new(LuminaError::Candle(candle_core::Error::Msg(e.to_string()))),
                    }
                })?;
                Ok((String::new(), text.trim().to_string(), 0.0))
            }
            _ => Err(LuminaError::UnsupportedOp("This model is a text encoder only, not a CausalLM".into())),
        }
    }

    fn generate_chat(
        &mut self,
        messages: &[(String, String)],
        max_tokens: usize,
        temperature: f64,
        reasoning_effort: f64,
    ) -> Result<(String, String, f64)> {
        let mut prompt = String::new();
        let family_lower = self.family.to_lowercase();
        if family_lower.contains("gemma") {
            // Gemma chat template: <start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n
            for (role, content) in messages {
                let r = if role == "assistant" { "model" } else { role.as_str() };
                prompt.push_str(&format!("<start_of_turn>{r}\n{content}<end_of_turn>\n"));
            }
            prompt.push_str("<start_of_turn>model\n");
        } else if family_lower.contains("llama") || family_lower.contains("deepseek") {
            // Llama 3 / DeepSeek format
            prompt.push_str("<|begin_of_text|>");
            for (role, content) in messages {
                prompt.push_str(&format!("<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"));
            }
            prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
        } else {
            // Default ChatML template (Qwen, Phi, Mistral, LFM)
            for (role, content) in messages {
                prompt.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
            }
            prompt.push_str("<|im_start|>assistant\n");
        }
        self.generate_with_reasoning(&prompt, max_tokens, temperature, reasoning_effort)
    }

    fn generate_stream(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
        on_token: Option<crate::models::common::TokenCallback>,
    ) -> Result<String> {
        match &mut self.inner {
            TextBackend::CausalLM(pipeline) => {
                let mut params = TextGenParams::default();
                params.max_tokens = max_tokens;
                params.temperature = temperature;
                pipeline.generate_with_callback(prompt, &params, on_token)
            }
            TextBackend::QuantizedGemma3 { model, tokenizer, device } => {
                let tok = tokenizer.as_ref().ok_or_else(|| {
                    LuminaError::Config("Tokenizer not attached to QuantizedGemma3 model".into())
                })?;

                let formatted_prompt = if !prompt.contains("<start_of_turn>") {
                    format!("<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n", prompt.trim())
                } else {
                    prompt.to_string()
                };

                let tokens = tok.encode(formatted_prompt.as_str(), true).map_err(|e| {
                    LuminaError::Context {
                        context: "Failed to tokenize prompt for Gemma 3".into(),
                        source: Box::new(LuminaError::Candle(candle_core::Error::Msg(e.to_string()))),
                    }
                })?.get_ids().to_vec();

                let mut mut_cb = on_token;
                let input = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;
                let logits = model.forward(&input, 0)?;
                let last = logits.squeeze(0)?;
                let l_vec = last.to_dtype(DType::F32)?.to_vec1::<f32>()?;

                let mut top: Vec<_> = l_vec.iter().enumerate().collect();
                top.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
                let mut curr_tok = top[0].0 as u32;
                let mut generated = vec![curr_tok];

                if let Some(ref mut cb) = mut_cb {
                    if let Ok(piece) = tok.decode(&[curr_tok], false) {
                        cb(&piece);
                    }
                }

                for i in 0..max_tokens {
                    if curr_tok == 1 || curr_tok == 106 { break; }
                    let inp = Tensor::new(&[curr_tok], device)?.unsqueeze(0)?;
                    let step_logits = model.forward(&inp, tokens.len() + i)?;
                    let step_last = step_logits.squeeze(0)?;
                    let step_vec = step_last.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                    let mut t: Vec<_> = step_vec.iter().enumerate().collect();
                    t.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
                    curr_tok = t[0].0 as u32;
                    generated.push(curr_tok);

                    if let Some(ref mut cb) = mut_cb {
                        if let Ok(piece) = tok.decode(&[curr_tok], false) {
                            if !cb(&piece) { break; }
                        }
                    }
                }

                let text = tok.decode(&generated, true).map_err(|e| {
                    LuminaError::Context {
                        context: "Failed to decode generated token stream".into(),
                        source: Box::new(LuminaError::Candle(candle_core::Error::Msg(e.to_string()))),
                    }
                })?;
                Ok(text.trim().to_string())
            }
            _ => {
                let text = self.generate(prompt, max_tokens, temperature)?;
                if let Some(cb) = on_token {
                    cb(&text);
                }
                Ok(text)
            }
        }
    }
}
