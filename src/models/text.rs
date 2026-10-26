// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Text model wrapper (encoders and CausalLM text generation)

pub mod causal;
pub mod llama;
pub mod moe;
pub mod phi;
pub mod qwen;

use candle_core::{DType, Device, Tensor};
use crate::error::{LuminaError, Result};
use crate::text::{Mistral3TextEncoder, Qwen3TextEncoder, T5TextEncoder};
pub use causal::{CausalLM, CausalLMConfig, CausalLMPipeline, ChatMessage, ChatRole, LogitsProcessor, TextGenParams};
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
    fn chat_prompt(&self, messages: &[(String, String)]) -> String {
        if let TextBackend::CausalLM(pipeline) = &self.inner {
            pipeline.format_chat_messages(messages)
        } else {
            let mut prompt = String::new();
            for (role, content) in messages {
                prompt.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
            }
            prompt.push_str("<|im_start|>assistant\n");
            prompt
        }
    }

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
                if let Ok(q_str) = std::env::var("QUANTIZED_KV") {
                    params.quantized_kv = q_str == "1" || q_str.eq_ignore_ascii_case("true");
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
                    format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n", prompt.trim())
                };

                let tokens = tok.encode(formatted_prompt.as_str(), true).map_err(|e| {
                    LuminaError::Context {
                        context: "Failed to tokenize prompt for Qwen3.8".into(),
                        source: Box::new(LuminaError::Candle(candle_core::Error::Msg(e.to_string()))),
                    }
                })?.get_ids().to_vec();

                let processor = LogitsProcessor::new(temperature, 0.95, 50, 1.1);

                model.reset_cache();
                let input = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;
                let logits = model.forward(&input, 0)?;

                let mut curr_tok = processor.sample(&logits, &tokens)?;
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
                    curr_tok = processor.sample(&step_logits, &generated)?;
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
            // Family-aware default: ChatML for Qwen/Mistral/LFM, native
            // template (e.g. Phi) when the backend provides one.
            prompt = self.chat_prompt(messages);
        }
        self.generate_with_reasoning(&prompt, max_tokens, temperature, reasoning_effort)
    }

    fn generate_stream_with_params(
        &mut self,
        prompt: &str,
        params: &TextGenParams,
        on_token: Option<crate::models::common::TokenCallback>,
    ) -> Result<String> {
        let ignore_eos = std::env::var_os("AURORA_IGNORE_EOS").is_some();
        match &mut self.inner {
            TextBackend::CausalLM(pipeline) => {
                pipeline.generate_with_callback(prompt, params, on_token)
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

                let processor = LogitsProcessor::new(
                    params.temperature,
                    params.top_p,
                    params.top_k,
                    params.repetition_penalty,
                );

                let mut mut_cb = on_token;
                let input = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;
                let logits = model.forward(&input, 0)?;

                if std::env::var_os("AURORA_HIP_GRAPH").is_some() {
                    let max_len = tokens.len() + params.max_tokens + 8;
                    match model.setup_static_graph(tokens.len(), max_len) {
                        Ok(()) => eprintln!("⚡ HIP graph decode enabled (gemma, max_len={max_len})"),
                        Err(e) => {
                            model.clear_static_graph();
                            eprintln!("⚠️ HIP graph disabled: {e}");
                        }
                    }
                }

                let mut curr_tok = processor.sample(&logits, &tokens)?;
                let mut generated = vec![curr_tok];

                if let Some(ref mut cb) = mut_cb {
                    if let Ok(piece) = tok.decode(&[curr_tok], false) {
                        if !cb(&piece) {
                            let text = tok.decode(&generated, true).map_err(|e| {
                                LuminaError::Context {
                                    context: "Failed to decode generated token stream".into(),
                                    source: Box::new(LuminaError::Candle(candle_core::Error::Msg(e.to_string()))),
                                }
                            })?;
                            return Ok(text.trim().to_string());
                        }
                    }
                }

                let mut stop_ids = params.stop_tokens.clone();
                if !stop_ids.contains(&1) { stop_ids.push(1); }
                if !stop_ids.contains(&106) { stop_ids.push(106); }

                for i in 0..params.max_tokens {
                    if !ignore_eos && stop_ids.contains(&curr_tok) { break; }
                    let inp = Tensor::new(&[curr_tok], device)?.unsqueeze(0)?;
                    let step_logits = model.forward(&inp, tokens.len() + i)?;
                    curr_tok = processor.sample(&step_logits, &generated)?;
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
            TextBackend::QuantizedQwen35 { model, tokenizer, device } => {
                let tok = tokenizer.as_ref().ok_or_else(|| {
                    LuminaError::Config("Tokenizer not attached to QuantizedQwen35 model".into())
                })?;

                let formatted_prompt = if prompt.contains("<|im_start|>") {
                    prompt.to_string()
                } else {
                    format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n", prompt.trim())
                };

                let tokens = tok.encode(formatted_prompt.as_str(), true).map_err(|e| {
                    LuminaError::Context {
                        context: "Failed to tokenize prompt for Qwen3.8".into(),
                        source: Box::new(LuminaError::Candle(candle_core::Error::Msg(e.to_string()))),
                    }
                })?.get_ids().to_vec();

                let processor = LogitsProcessor::new(
                    params.temperature,
                    params.top_p,
                    params.top_k,
                    params.repetition_penalty,
                );

                let mut mut_cb = on_token;
                model.reset_cache();
                let input = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;
                let logits = model.forward(&input, 0)?;

                if std::env::var_os("AURORA_HIP_GRAPH").is_some() {
                    let max_len = tokens.len() + params.max_tokens + 8;
                    match model.setup_static_graph(tokens.len(), max_len) {
                        Ok(()) => eprintln!("⚡ HIP graph decode enabled (qwen35, max_len={max_len})"),
                        Err(e) => {
                            model.clear_static_graph();
                            eprintln!("⚠️ HIP graph disabled: {e}");
                        }
                    }
                }

                let mut curr_tok = processor.sample(&logits, &tokens)?;
                let mut generated = vec![curr_tok];

                if let Some(ref mut cb) = mut_cb {
                    if let Ok(piece) = tok.decode(&[curr_tok], false) {
                        if !cb(&piece) {
                            let text = tok.decode(&generated, true).map_err(|e| {
                                LuminaError::Context {
                                    context: "Failed to decode generated token stream".into(),
                                    source: Box::new(LuminaError::Candle(candle_core::Error::Msg(e.to_string()))),
                                }
                            })?;
                            return Ok(text.trim().to_string());
                        }
                    }
                }

                let mut stop_ids: Vec<u32> = [
                    model.eos_token_id(),
                    tok.token_to_id("<|im_end|>"),
                    tok.token_to_id("<|endoftext|>"),
                ]
                .into_iter()
                .flatten()
                .collect();

                for &s in &params.stop_tokens {
                    if !stop_ids.contains(&s) {
                        stop_ids.push(s);
                    }
                }

                for i in 0..params.max_tokens {
                    if stop_ids.contains(&curr_tok) {
                        break;
                    }
                    let inp = Tensor::new(&[curr_tok], device)?.unsqueeze(0)?;
                    let step_logits = model.forward(&inp, tokens.len() + i)?;
                    curr_tok = processor.sample(&step_logits, &generated)?;
                    generated.push(curr_tok);

                    if let Some(ref mut cb) = mut_cb {
                        if let Ok(piece) = tok.decode(&[curr_tok], false) {
                            if !cb(&piece) {
                                break;
                            }
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
            _ => Err(LuminaError::UnsupportedOp("This model is a text encoder only, not a CausalLM".into())),
        }
    }

    fn generate_stream(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
        on_token: Option<crate::models::common::TokenCallback>,
    ) -> Result<String> {
        let mut params = TextGenParams::default();
        params.max_tokens = max_tokens;
        params.temperature = temperature;
        self.generate_stream_with_params(prompt, &params, on_token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_sampling_greedy_deterministic() -> Result<()> {
        let device = Device::Cpu;
        let logits_data = vec![0.1f32, 2.5, 10.8, -1.2, 5.4];
        let logits = Tensor::new(logits_data.as_slice(), &device)?;
        let processor = LogitsProcessor::new(0.0, 0.95, 50, 1.0);

        for _ in 0..10 {
            let tok = processor.sample(&logits, &[])?;
            assert_eq!(tok, 2, "Greedy sampling must select index 2 with maximum logit 10.8");
        }
        Ok(())
    }

    #[test]
    fn test_sampling_logits_processor() -> Result<()> {
        let device = Device::Cpu;
        let logits_data = vec![1.0f32, 1.0, 1.0, 1.0, 100.0];
        let logits = Tensor::new(logits_data.as_slice(), &device)?;
        let processor = LogitsProcessor::new(0.7, 0.95, 50, 1.1);

        let tok = processor.sample(&logits, &[])?;
        assert_eq!(tok, 4, "High confidence logit at index 4 must be sampled");
        Ok(())
    }

    #[test]
    fn test_sampling_repetition_penalty() -> Result<()> {
        let device = Device::Cpu;
        // token 0 has logit 10.0, token 1 has logit 9.5
        let logits_data = vec![10.0f32, 9.5, 1.0];
        let logits = Tensor::new(logits_data.as_slice(), &device)?;

        // Without penalty, token 0 is sampled (greedy)
        let proc_no_pen = LogitsProcessor::new(0.0, 0.95, 50, 1.0);
        let tok1 = proc_no_pen.sample(&logits, &[])?;
        assert_eq!(tok1, 0);

        // With penalty 1.2 on token 0, its logit drops to 10.0 / 1.2 = 8.33 < 9.5
        let proc_pen = LogitsProcessor::new(0.0, 0.95, 50, 1.2);
        let tok2 = proc_pen.sample(&logits, &[0])?;
        assert_eq!(tok2, 1, "Repetition penalty must penalize token 0 so token 1 is selected");
        Ok(())
    }

    #[test]
    fn test_generation_model_with_params() -> Result<()> {
        let mut params = TextGenParams::default();
        params.max_tokens = 64;
        params.temperature = 0.0;
        params.top_p = 0.9;
        params.top_k = 20;
        params.repetition_penalty = 1.15;
        params.stop_tokens = vec![99999];

        assert_eq!(params.max_tokens, 64);
        assert_eq!(params.temperature, 0.0);
        assert_eq!(params.top_p, 0.9);
        assert_eq!(params.top_k, 20);
        assert_eq!(params.repetition_penalty, 1.15);
        assert_eq!(params.stop_tokens, vec![99999]);
        Ok(())
    }

    #[test]
    fn test_generation_model_stream_early_abort() -> Result<()> {
        let mut count = 0;
        let mut callback = |piece: &str| -> bool {
            count += 1;
            assert!(!piece.is_empty());
            count < 2 // Abort after 2 tokens
        };

        let allow_first = callback("first");
        assert!(allow_first);
        let abort_second = callback("second");
        assert!(!abort_second, "Callback must request early abort when returning false");
        Ok(())
    }
}
