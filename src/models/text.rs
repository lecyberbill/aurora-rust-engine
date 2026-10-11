// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Text model wrapper (encoders and CausalLM text generation)

pub mod causal;

use candle_core::{DType, Device, Tensor};
use crate::error::{LuminaError, Result};
use crate::text::{Mistral3TextEncoder, Qwen3TextEncoder, T5TextEncoder};
pub use causal::{CausalLMConfig, CausalLMPipeline, TextGenParams};
use super::common::{AnyModel, EncodeModel, GenerationModel, ModelKind};

/// A text backend boxed into one type.
pub enum TextBackend {
    Qwen(Qwen3TextEncoder),
    Mistral(Mistral3TextEncoder),
    T5(T5TextEncoder),
    CausalLM(CausalLMPipeline),
    NativeAccelerated {
        model_tag: String,
        endpoint: String,
        device: Device,
        dtype: DType,
    },
}

impl TextBackend {
    fn device(&self) -> &Device {
        match self {
            TextBackend::Qwen(_) | TextBackend::Mistral(_) | TextBackend::T5(_) => &Device::Cpu,
            TextBackend::CausalLM(c) => &c.device,
            TextBackend::NativeAccelerated { device, .. } => device,
        }
    }

    fn dtype(&self) -> DType {
        match self {
            TextBackend::Qwen(_) | TextBackend::Mistral(_) | TextBackend::T5(_) => DType::F16,
            TextBackend::CausalLM(c) => c.dtype,
            TextBackend::NativeAccelerated { dtype, .. } => *dtype,
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
    pub fn native_accelerated(id: String, family: String, model_tag: String, endpoint: String, device: Device, dtype: DType) -> Self {
        Self {
            id,
            family,
            inner: TextBackend::NativeAccelerated {
                model_tag,
                endpoint,
                device,
                dtype,
            },
        }
    }
}

impl AnyModel for TextModel {
    fn kind(&self) -> ModelKind {
        match &self.inner {
            TextBackend::CausalLM(_) | TextBackend::NativeAccelerated { .. } => ModelKind::Text,
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
        match &mut self.inner {
            TextBackend::CausalLM(pipeline) => {
                let mut params = TextGenParams::default();
                params.max_tokens = max_tokens;
                params.temperature = temperature;
                pipeline.generate(prompt, &params)
            }
            TextBackend::NativeAccelerated { model_tag, endpoint, .. } => {
                let payload = serde_json::json!({
                    "model": model_tag,
                    "prompt": prompt,
                    "stream": false,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature,
                        "num_gpu": 99,
                        "num_thread": 8,
                        "low_vram": false,
                    }
                });

                let url = format!("{}/api/generate", endpoint.trim_end_matches('/'));
                let response = ureq::post(&url)
                    .timeout(std::time::Duration::from_secs(120))
                    .send_json(payload)
                    .map_err(|e| LuminaError::Context {
                        context: format!("Native ROCm Engine generation failed at {url}: {e}"),
                        source: Box::new(LuminaError::Config(e.to_string())),
                    })?;

                let res_json: serde_json::Value = response.into_json().map_err(|e| LuminaError::Context {
                    context: "Failed to parse Native GPU generation response".into(),
                    source: Box::new(LuminaError::Config(e.to_string())),
                })?;

                if let (Some(eval_count), Some(eval_duration)) = (
                    res_json.get("eval_count").and_then(|v| v.as_u64()),
                    res_json.get("eval_duration").and_then(|v| v.as_u64()),
                ) {
                    if eval_duration > 0 {
                        let tok_per_sec = (eval_count as f64) / (eval_duration as f64 / 1e9);
                        println!(
                            "⚡ GPU Eval Engine: {} tokens in {:.2}s ({:.2} tok/s)",
                            eval_count,
                            eval_duration as f64 / 1e9,
                            tok_per_sec
                        );
                    }
                }

                let resp = res_json.get("response")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let thinking = res_json.get("thinking")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");

                let text = if resp.is_empty() {
                    thinking.to_string()
                } else if thinking.is_empty() {
                    resp.to_string()
                } else {
                    format!("<think>\n{}\n</think>\n\n{}", thinking, resp)
                };

                Ok(text)
            }
            _ => Err(LuminaError::UnsupportedOp("This model is a text encoder only, not a CausalLM".into())),
        }
    }
}
