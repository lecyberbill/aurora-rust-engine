// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Pure Rust Vision-Language Pipeline (Qwen-VL / PaliGemma / Gemma-3 Vision / SmolVLM)

use std::path::Path;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use image::DynamicImage;
use tokenizers::Tokenizer;

use crate::models::vlm::{VisionTransformerConfig, VlmDecoderConfig, VlmModel};

/// Generation parameters for Vision-Language Models
#[derive(Debug, Clone)]
pub struct VlmParams {
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub stop_tokens: Vec<u32>,
}

impl Default for VlmParams {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repetition_penalty: 1.1,
            stop_tokens: vec![151643, 151645, 128001, 128009, 2, 1],
        }
    }
}

/// Pure Rust Vision-Language Inference Pipeline
pub struct VlmPipeline {
    pub model: VlmModel,
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub image_size: usize,
}

impl VlmPipeline {
    pub fn new(
        model: VlmModel,
        tokenizer: Tokenizer,
        device: Device,
        image_size: usize,
    ) -> Self {
        Self {
            model,
            tokenizer,
            device,
            image_size,
        }
    }

    /// Load VLM pipeline from a directory containing weights and tokenizer
    pub fn from_pretrained<P: AsRef<Path>>(model_dir: P, device: &Device) -> Result<Self> {
        let dir = model_dir.as_ref();
        let tokenizer_path = dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            candle_core::bail!("Tokenizer not found at: {}", tokenizer_path.display());
        }
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to load tokenizer: {}", e)))?;

        // Find safetensors files
        let mut safetensors_files = Vec::new();
        if dir.join("model.safetensors").exists() {
            safetensors_files.push(dir.join("model.safetensors"));
        } else {
            for entry in std::fs::read_dir(dir).map_err(|e| candle_core::Error::Msg(e.to_string()))? {
                let entry = entry.map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                let path = entry.path();
                if path.extension().map_or(false, |ext| ext == "safetensors") {
                    safetensors_files.push(path);
                }
            }
        }
        safetensors_files.sort();

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&safetensors_files, DType::F32, device)?
        };

        // Detect or use standard default configuration for Qwen2-VL / Qwen3-VL
        let vision_cfg = VisionTransformerConfig::default();
        let decoder_cfg = VlmDecoderConfig::default();
        let image_token_id = tokenizer.token_to_id("<|image_pad|>")
            .or_else(|| tokenizer.token_to_id("<image>"))
            .unwrap_or(151655);

        let model = VlmModel::new(&vision_cfg, &decoder_cfg, image_token_id, vb)?;

        Ok(Self::new(model, tokenizer, device.clone(), vision_cfg.image_size))
    }

    /// Preprocess an image into normalized [1, 3, H, W] tensor
    pub fn preprocess_image(&self, img: &DynamicImage) -> Result<Tensor> {
        let resized = img.resize_exact(
            self.image_size as u32,
            self.image_size as u32,
            image::imageops::FilterType::CatmullRom,
        );
        let rgb = resized.to_rgb8();
        let raw = rgb.into_raw();

        let num_pixels = self.image_size * self.image_size;
        let mut float_data = vec![0f32; 3 * num_pixels];

        // SigLIP / ImageNet normalization (mean = 0.5, std = 0.5)
        for i in 0..num_pixels {
            let r = raw[i * 3] as f32 / 255.0;
            let g = raw[i * 3 + 1] as f32 / 255.0;
            let b = raw[i * 3 + 2] as f32 / 255.0;

            float_data[i] = (r - 0.5) / 0.5;
            float_data[num_pixels + i] = (g - 0.5) / 0.5;
            float_data[2 * num_pixels + i] = (b - 0.5) / 0.5;
        }

        Tensor::from_vec(float_data, (1, 3, self.image_size, self.image_size), &self.device)
    }

    /// Generate visual description or answer a question about an image
    pub fn generate(
        &mut self,
        image: Option<&DynamicImage>,
        prompt: &str,
        params: &VlmParams,
    ) -> Result<String> {
        self.model.reset_kv_cache();

        let pixel_tensor = if let Some(img) = image {
            Some(self.preprocess_image(img)?)
        } else {
            None
        };

        // Format prompt with image token if image is present
        let formatted_prompt = if image.is_some() {
            if prompt.contains("<image>") || prompt.contains("<|image_pad|>") {
                prompt.to_string()
            } else {
                format!("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n")
            }
        } else {
            format!("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
        };

        let encoding = self.tokenizer.encode(formatted_prompt, true)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let token_ids = encoding.get_ids();
        let input_ids = Tensor::new(token_ids, &self.device)?.unsqueeze(0)?;

        let mut current_pos = 0;
        let mut generated_tokens: Vec<u32> = Vec::new();

        // 1. Prefill step
        let logits = self.model.forward(&input_ids, pixel_tensor.as_ref(), current_pos)?;
        current_pos += token_ids.len();

        let last_logits = logits.narrow(1, logits.dim(1)? - 1, 1)?.squeeze(1)?;
        let mut next_token = self.sample_token(&last_logits, &generated_tokens, params)?;
        generated_tokens.push(next_token);

        // 2. Autoregressive decoding loop
        for _ in 1..params.max_tokens {
            if params.stop_tokens.contains(&next_token) {
                break;
            }

            let next_input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let step_logits = self.model.forward(&next_input, None, current_pos)?;
            current_pos += 1;

            let step_last = step_logits.squeeze(0)?.squeeze(0)?;
            next_token = self.sample_token(&step_last, &generated_tokens, params)?;
            generated_tokens.push(next_token);
        }

        let output_text = self.tokenizer.decode(&generated_tokens, true)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        Ok(output_text)
    }

    fn sample_token(&self, logits: &Tensor, generated: &[u32], params: &VlmParams) -> Result<u32> {
        let mut l_vec = logits.to_dtype(DType::F32)?.to_vec1::<f32>()?;

        // Repetition penalty
        if params.repetition_penalty != 1.0 {
            for &tok in generated {
                if let Some(val) = l_vec.get_mut(tok as usize) {
                    if *val > 0.0 {
                        *val /= params.repetition_penalty;
                    } else {
                        *val *= params.repetition_penalty;
                    }
                }
            }
        }

        // Greedy sampling if temperature <= 0
        if params.temperature <= 0.0 {
            let mut best_idx = 0;
            let mut best_val = f32::NEG_INFINITY;
            for (i, &v) in l_vec.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best_idx = i;
                }
            }
            return Ok(best_idx as u32);
        }

        // Apply temperature
        let inv_temp = (1.0 / params.temperature) as f32;
        for v in l_vec.iter_mut() {
            *v *= inv_temp;
        }

        // Softmax
        let max_logit = l_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        for v in l_vec.iter_mut() {
            *v = (*v - max_logit).exp();
            sum_exp += *v;
        }
        for v in l_vec.iter_mut() {
            *v /= sum_exp;
        }

        // Top-K / Top-P filter
        let mut indexed: Vec<(usize, f32)> = l_vec.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if params.top_k > 0 && indexed.len() > params.top_k {
            indexed.truncate(params.top_k);
        }

        let mut cum_prob = 0.0f32;
        let mut p_cutoff = indexed.len();
        for (idx, (_, p)) in indexed.iter().enumerate() {
            cum_prob += p;
            if cum_prob >= params.top_p as f32 {
                p_cutoff = idx + 1;
                break;
            }
        }
        indexed.truncate(p_cutoff);

        // Renormalize
        let total_p: f32 = indexed.iter().map(|(_, p)| p).sum();
        let mut rng = crate::audio::rng::SeededRng::new(
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64,
        );
        let r = rng.next_f32() * total_p;

        let mut acc = 0.0f32;
        for (tok_id, p) in indexed {
            acc += p;
            if acc >= r {
                return Ok(tok_id as u32);
            }
        }

        Ok(0)
    }
}
