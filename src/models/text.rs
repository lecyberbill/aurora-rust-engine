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
}

impl TextBackend {
    fn device(&self) -> &Device {
        match self {
            TextBackend::Qwen(_) | TextBackend::Mistral(_) | TextBackend::T5(_) => &Device::Cpu,
            TextBackend::CausalLM(c) => &c.device,
        }
    }

    fn dtype(&self) -> DType {
        match self {
            TextBackend::Qwen(_) | TextBackend::Mistral(_) | TextBackend::T5(_) => DType::F16,
            TextBackend::CausalLM(c) => c.dtype,
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
}

impl AnyModel for TextModel {
    fn kind(&self) -> ModelKind {
        match &self.inner {
            TextBackend::CausalLM(_) => ModelKind::Text,
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
            TextBackend::CausalLM(_) => Err(LuminaError::UnsupportedOp("CausalLM is for generation; use TextEncoder for conditioning".into())),
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
            _ => Err(LuminaError::UnsupportedOp("This model is a text encoder only, not a CausalLM".into())),
        }
    }
}
