use candle_core::quantized::QMatMul;
use candle_core::{DType, Device, Module, Tensor};
use image::RgbImage;
use std::any::Any;
use std::path::Path;
use std::sync::Arc;
use crate::error::{LuminaError, Result};
use crate::traits::{DiffusionParams, Img2ImgParams, InpaintParams};

/// Unified Linear Layer supporting either full-precision/safetensors [`Tensor`]
/// or hardware-accelerated quantized [`QMatMul`] (W4A16 / GGUF).
///
/// Designed like HuggingFace `transformers` linear layers: callers simply call `.forward(x)`
/// or `.forward(&x)` and the underlying kernel (Float GEMM or Quantized GEMM) is executed
/// seamlessly on CPU, CUDA or AMD ROCm.
#[derive(Clone)]
pub enum LinearLayer {
    Tensor(Tensor),
    Quantized(Arc<QMatMul>),
}

impl std::fmt::Debug for LinearLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tensor(t) => write!(f, "LinearLayer::Tensor({:?}, {:?})", t.shape(), t.dtype()),
            Self::Quantized(_) => write!(f, "LinearLayer::Quantized(QMatMul)"),
        }
    }
}

impl LinearLayer {
    pub fn from_tensor(t: Tensor) -> Self {
        Self::Tensor(t)
    }

    pub fn from_qmatmul(q: QMatMul) -> Self {
        Self::Quantized(Arc::new(q))
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Quantized(q) => {
                match q.forward(x) {
                    Ok(out) => {
                        if out.dtype() != x.dtype() {
                            out.to_dtype(x.dtype()).map_err(LuminaError::Candle)
                        } else {
                            Ok(out)
                        }
                    }
                    Err(e) => {
                        // In some GGUF architectures (such as Nemotron-H blk.5.attn_q.weight [2688, 4096]),
                        // the tensor is stored with reversed dimensions [in_dim, out_dim] instead of [out_dim, in_dim].
                        // Dequantize the QMatMul tensor and use the robust transposed dense matmul path.
                        let dev = x.device();
                        let w = q.dequantize_f16().or_else(|_| match q.as_ref() {
                            candle_core::quantized::QMatMul::QTensor(t) => t.dequantize(dev),
                            candle_core::quantized::QMatMul::Tensor(t) => Ok(t.clone()),
                            candle_core::quantized::QMatMul::TensorF16(t) => Ok(t.clone()),
                        }).map_err(|_| LuminaError::Candle(e))?;
                        Self::from_tensor(w).forward(x)
                    }
                }
            }
            Self::Tensor(w) => {
                let w_dtype = w.dtype();
                let w_float = if w_dtype == DType::U8 {
                    let target_dt = x.dtype();
                    let shifted = (w.to_dtype(DType::F32).map_err(LuminaError::Candle)? - 128.0).map_err(LuminaError::Candle)?;
                    (shifted * (1.0 / 127.0)).map_err(LuminaError::Candle)?.to_dtype(target_dt).map_err(LuminaError::Candle)?
                } else {
                    w.clone()
                };

                let last_x_dim = x.dim(candle_core::D::Minus1).map_err(LuminaError::Candle)?;
                let w_dims = w_float.shape().dims2().map_err(LuminaError::Candle)?;
                let w_adapted = if w_dims.1 == last_x_dim {
                    w_float.t().map_err(LuminaError::Candle)?
                } else if w_dims.0 == last_x_dim {
                    w_float
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
        }
    }

    /// Whether the weights are held in a truly quantized (GGUF/W4A16) form.
    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::Quantized(q) if matches!(q.as_ref(), QMatMul::QTensor(_)))
    }

    /// GGML dtype of the packed weights, when they are a `QTensor`.
    pub fn quantized_dtype(&self) -> Option<candle_core::quantized::GgmlDType> {
        match self {
            Self::Quantized(q) => match q.as_ref() {
                QMatMul::QTensor(t) => Some(t.dtype()),
                _ => None,
            },
            Self::Tensor(_) => None,
        }
    }

    /// Fused indexed mixture-of-experts matmul for a `(num_experts, n, k)` expert
    /// stack. `x` is `(batch, 1 or topk, k)` and `ids` is `(batch, topk)` u32;
    /// the result is `(batch, topk, n)`. Only available for quantized weights on
    /// CUDA/ROCm — callers should fall back to a dense loop otherwise.
    pub fn indexed_moe_forward(&self, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::Quantized(q) => q.indexed_moe_forward(x, ids).map_err(LuminaError::Candle),
            Self::Tensor(_) => Err(LuminaError::Config(
                "indexed_moe_forward requires quantized expert weights".to_string(),
            )),
        }
    }
}

/// Progress callback (step_index, total_steps, latent). Used object-safe (boxed, not generic).
pub type ProgressFn<'a> = &'a mut (dyn FnMut(usize, usize, &Tensor) + 'a);

/// Token generation callback (token_str) returning whether generation should continue.
pub type TokenCallback<'a> = &'a mut (dyn FnMut(&str) -> bool + 'a);


/// The broad category a loaded model belongs to. Mirrors the way HuggingFace groups
/// `AutoModelForCausalLM`, `DiffusionPipeline`, `AutoModelForImageClassification`, etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    /// Autoregressive text (Qwen / Mistral as a CausalLM) — Milestone 16.
    Text,
    /// Text-encoder side of a diffused model (Qwen / Mistral / T5 / CLIP used for conditioning).
    TextEncoder,
    /// Text-to-image / img2img / inpaint diffusion model (Flux, SDXL, SD1.5).
    Diffusion,
    /// Could not be determined from the checkpoint/`config.json`.
    Unknown,
}

/// Everything a model agrees to expose (metadata only). Every concrete model implements this.
/// Object-safe: this is the type boxed as `Arc<Mutex<dyn AnyModel>>`.
pub trait AnyModel: Send + Sync {
    fn kind(&self) -> ModelKind;
    /// Stable family slug, e.g. "qwen3", "mistral3", "flux2-dev", "sdxl".
    fn family(&self) -> &str;
    fn id(&self) -> &str;
    fn device(&self) -> &Device;
    fn dtype(&self) -> DType;

    /// Upcast to `&dyn Any` so a caller can downcast to a concrete sub-trait
    /// (`ImageGenerationModel`, `EncodeModel`, ...). Implement each concrete model with `self`.
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Downcast a boxed trait object held behind a lock to a concrete sub-trait.
///
/// The engine returns `Arc<Mutex<dyn AnyModel>>`; callers that want to drive a specific capability
/// (image vs text) downcast once. Returns `None` when the model is a different kind.
pub fn downcast_model<M: AnyModel + 'static>(m: &mut dyn AnyModel) -> Option<&mut M> {
    m.as_any_mut().downcast_mut::<M>()
}

/// Text-to-text generator (the CausalLM use case). Implemented by a decoder-only pipeline.
pub trait GenerationModel: AnyModel {
    fn generate(&mut self, prompt: &str, max_tokens: usize, temperature: f64) -> Result<String>;

    /// Primary universal generation method with full sampling parameters.
    fn generate_with_params(
        &mut self,
        prompt: &str,
        params: &crate::models::text::TextGenParams,
    ) -> Result<String> {
        self.generate_stream_with_params(prompt, params, None)
    }

    /// Universal token generation stream with early-abort callback and full sampling parameters.
    fn generate_stream_with_params(
        &mut self,
        prompt: &str,
        params: &crate::models::text::TextGenParams,
        on_token: Option<TokenCallback>,
    ) -> Result<String> {
        let full = self.generate(prompt, params.max_tokens, params.temperature)?;
        if let Some(cb) = on_token {
            cb(&full);
        }
        Ok(full)
    }

    /// Generation with explicit reasoning effort (0.0 = low/none, 0.5 = medium, 1.0 = high).
    /// Returns (thinking_text, response_text, tokens_per_sec).
    fn generate_with_reasoning(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
        _reasoning_effort: f64,
    ) -> Result<(String, String, f64)> {
        let mut params = crate::models::text::TextGenParams::default();
        params.max_tokens = max_tokens;
        params.temperature = temperature;
        let full = self.generate_with_params(prompt, &params)?;
        Ok((String::new(), full, 0.0))
    }

    /// Build a prompt from role/content messages (default ChatML).
    /// Models with a different chat format override this.
    fn chat_prompt(&self, messages: &[(String, String)]) -> String {
        let mut prompt = String::new();
        for (role, content) in messages {
            prompt.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
        }
        prompt.push_str("<|im_start|>assistant\n");
        prompt
    }

    /// Chat generation accepting a sequence of role/content pairs.
    /// Returns (thinking_text, response_text, tokens_per_sec).
    fn generate_chat(
        &mut self,
        messages: &[(String, String)],
        max_tokens: usize,
        temperature: f64,
        reasoning_effort: f64,
    ) -> Result<(String, String, f64)> {
        let prompt = self.chat_prompt(messages);
        self.generate_with_reasoning(&prompt, max_tokens, temperature, reasoning_effort)
    }

    /// Streaming token generation with early-abort callback.
    fn generate_stream(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
        on_token: Option<TokenCallback>,
    ) -> Result<String> {
        let mut params = crate::models::text::TextGenParams::default();
        params.max_tokens = max_tokens;
        params.temperature = temperature;
        self.generate_stream_with_params(prompt, &params, on_token)
    }

    /// Streaming chat generation with early-abort callback.
    fn generate_chat_stream(
        &mut self,
        messages: &[(String, String)],
        max_tokens: usize,
        temperature: f64,
        reasoning_effort: f64,
        on_token: Option<TokenCallback>,
    ) -> Result<(String, String, f64)> {
        let (thinking, resp, tok_s) = self.generate_chat(messages, max_tokens, temperature, reasoning_effort)?;
        if let Some(cb) = on_token {
            if !thinking.is_empty() {
                cb(&format!("<think>{thinking}</think>"));
            }
            cb(&resp);
        }
        Ok((thinking, resp, tok_s))
    }
}

/// Text-encoder (conditioning) use case: prompt -> embedding tensor.
pub trait EncodeModel: AnyModel {
    fn encode(&mut self, prompt: &str, max_len: usize) -> Result<Tensor>;
}

/// Image diffusion (T2I / img2img / inpaint). Object-safe: callbacks are boxed, not generic.
pub trait ImageGenerationModel: AnyModel {
    fn generate_t2i(&mut self, params: DiffusionParams, on_step: Option<ProgressFn>) -> Result<RgbImage>;
    fn generate_img2img(&mut self, params: Img2ImgParams, on_step: Option<ProgressFn>) -> Result<RgbImage>;
    fn generate_inpaint(&mut self, params: InpaintParams, on_step: Option<ProgressFn>) -> Result<RgbImage>;
    fn load_lora(&mut self, path: &Path, multiplier: f64) -> Result<()>;
    fn unload_lora(&mut self, id: &str) -> Result<()>;
    fn lora_ids(&self) -> Vec<String>;
}
