// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: ACE-Step 5Hz audio codec (FSQ tokenizer + attention pooler + detokenizer)

//! ACE-Step 1.5 "5Hz" audio semantic codec.
//!
//! * `AceStepAudioTokenizer` : acoustic features `[B, T, 64]` → pooled 2048-d tokens
//!   + Finite-Scalar-Quantization indices (`ResidualFSQ`, levels `[8,8,8,5,5,5]`).
//! * `AudioTokenDetokenizer` : quantized tokens `[B, T, 2048]` → 25Hz latents `[B, T*5, 64]`.
//!
//! Mirrors the reference `vector_quantize_pytorch.ResidualFSQ` (single quantizer) and the
//! `AceStepAudioTokenizer` / `AudioTokenDetokenizer` modules.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Linear, VarBuilder};
use std::path::Path;

use crate::models::acestep::{AceStepLyricLayer, AudioRmsNorm, AudioRotaryEmbedding};

pub const POOL_WINDOW_SIZE: usize = 5;
const HIDDEN: usize = 2048;
const ACOUSTIC: usize = 64;
const LEVELS: [usize; 6] = [8, 8, 8, 5, 5, 5];

/// Apply a Linear on the last dimension of an N-D tensor.
fn linear_last(lin: &Linear, x: &Tensor) -> candle_core::Result<Tensor> {
    let dims = x.dims().to_vec();
    let last = *dims.last().unwrap();
    let rows: usize = dims[..dims.len() - 1].iter().product();
    let flat = x.reshape((rows, last))?;
    let out = lin.forward(&flat)?;
    let out_dim = out.dim(1)?;
    let mut out_dims = dims.clone();
    *out_dims.last_mut().unwrap() = out_dim;
    out.reshape(out_dims)
}

/// Single-quantizer `ResidualFSQ` (preserve_symmetry + hard clamp) on a 6-d codebook.
pub struct ResidualFsq {
    project_in: Linear,  // 2048 -> 6
    project_out: Linear, // 6 -> 2048
    basis: Tensor,       // [1, 1, 6]
    half_lm1: Tensor,    // [1, 1, 6] = (levels - 1) / 2
    two_over_lm1: Tensor, // [1, 1, 6] = 2 / (levels - 1)
    clamp: Tensor,       // [1, 1, 6] = 1 + 1/(levels - 1)
    codebook: Tensor,    // [64000, 6] f32
}

impl ResidualFsq {
    pub fn load(vb: VarBuilder, dtype: DType, dev: &Device) -> Result<Self> {
        let project_in = candle_nn::linear(HIDDEN, LEVELS.len(), vb.pp("project_in"))?;
        let project_out = candle_nn::linear(LEVELS.len(), HIDDEN, vb.pp("project_out"))?;

        let mut basis = Vec::with_capacity(LEVELS.len());
        let mut acc = 1usize;
        for &l in LEVELS.iter() {
            basis.push(acc as f32);
            acc *= l;
        }
        let codebook_size = acc; // 64000

        let half_lm1: Vec<f32> = LEVELS.iter().map(|&l| (l - 1) as f32 / 2.0).collect();
        let two_over_lm1: Vec<f32> = LEVELS.iter().map(|&l| 2.0 / (l - 1) as f32).collect();
        let clamp: Vec<f32> = LEVELS.iter().map(|&l| 1.0 + 1.0 / (l - 1) as f32).collect();

        // Implicit codebook: level index -> zhat * 2/(L-1) - 1.
        let mut cb = Vec::with_capacity(codebook_size * LEVELS.len());
        for idx in 0..codebook_size {
            for d in 0..LEVELS.len() {
                let level_idx = (idx / (basis[d] as usize)) % LEVELS[d];
                let v = level_idx as f32 * two_over_lm1[d] - 1.0;
                cb.push(v);
            }
        }
        let codebook = Tensor::from_vec(cb, (codebook_size, LEVELS.len()), dev)?;

        Ok(Self {
            project_in,
            project_out,
            basis: Tensor::from_vec(basis, (1, 1, LEVELS.len()), dev)?.to_dtype(dtype)?,
            half_lm1: Tensor::from_vec(half_lm1, (1, 1, LEVELS.len()), dev)?.to_dtype(dtype)?,
            two_over_lm1: Tensor::from_vec(two_over_lm1, (1, 1, LEVELS.len()), dev)?.to_dtype(dtype)?,
            clamp: Tensor::from_vec(clamp, (1, 1, LEVELS.len()), dev)?.to_dtype(dtype)?,
            codebook,
        })
    }

    /// Encode `[B, T, 2048]` into `(quantized [B, T, 2048], indices [B, T])`.
    pub fn encode(&self, x: &Tensor) -> candle_core::Result<(Tensor, Tensor)> {
        let z = self.project_in.forward(x)?; // [B, T, 6]
        // soft clamp: (z / clamp).tanh() * clamp
        let z = z
            .broadcast_div(&self.clamp)?
            .tanh()?
            .broadcast_mul(&self.clamp)?;
        // symmetry-preserving bound with hard clamp (clamp instead of tanh)
        let t = z.clamp(-1.0f32, 1.0f32)?;
        // bracket = floor((levels-1)*(t+1)/2 + 0.5)
        let bracket = t
            .broadcast_mul(&self.half_lm1)?
            .broadcast_add(&self.half_lm1)?
            .affine(1.0, 0.5)?
            .floor()?;
        // codes = 2/(levels-1) * bracket - 1
        let codes = bracket
            .broadcast_mul(&self.two_over_lm1)?
            .affine(1.0, -1.0)?;
        // indices = round(sum_d (codes+1)/two_over_lm1 * basis)
        let zhat = codes.broadcast_add(&Tensor::ones_like(&codes)?)?.broadcast_mul(&self.half_lm1)?;
        let indices = zhat
            .broadcast_mul(&self.basis)?
            .sum(2)?
            .round()?
            .to_dtype(DType::U32)?;
        let quantized = self.project_out.forward(&codes)?; // [B, T, 2048]
        Ok((quantized, indices))
    }

    /// Map codebook indices `[B, T]` (u32) back to quantized tokens `[B, T, 2048]`.
    pub fn get_output_from_indices(&self, indices: &Tensor) -> candle_core::Result<Tensor> {
        let dims = indices.dims().to_vec();
        let flat = indices.flatten_all()?.to_dtype(DType::U32)?;
        let codes = self.codebook.index_select(&flat, 0)?; // [N, 6]
        let mut new_dims = dims.clone();
        new_dims.push(LEVELS.len());
        let codes = codes.reshape(new_dims)?;
        self.project_out.forward(&codes)
    }
}

/// Attention pooler: `[B, T, P, 2048]` → `[B, T, 2048]` (special-token readout).
pub struct AttentionPooler {
    embed_tokens: Linear,
    norm: AudioRmsNorm,
    special_token: Tensor,
    layers: Vec<AceStepLyricLayer>,
    rope: AudioRotaryEmbedding,
}

impl AttentionPooler {
    fn load(vb: VarBuilder) -> Result<Self> {
        let embed_tokens = candle_nn::linear(HIDDEN, HIDDEN, vb.pp("embed_tokens"))?;
        let norm = AudioRmsNorm::new(HIDDEN, 1e-6, vb.pp("norm"))?;
        let special_token = vb.get((1, 1, HIDDEN), "special_token")?;
        let mut layers = Vec::with_capacity(2);
        for i in 0..2 {
            layers.push(AceStepLyricLayer::load(vb.pp(format!("layers.{}", i)))?);
        }
        Ok(Self {
            embed_tokens,
            norm,
            special_token,
            layers,
            rope: AudioRotaryEmbedding::new(128, 1_000_000.0),
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (b, t, _p, _d) = x.dims4()?;
        let x = linear_last(&self.embed_tokens, x)?; // [B,T,P,D]
        let special = self.special_token.expand((b, t, 1, HIDDEN))?;
        let x = Tensor::cat(&[&special, &x], 2)?; // [B,T,P+1,D]
        let seq = x.dim(2)?;
        let mut h = x.reshape((b * t, seq, HIDDEN))?;
        for layer in &self.layers {
            h = layer.forward(&h, &self.rope, None)?;
        }
        let h = self.norm.forward(&h)?;
        let cls = h.narrow(1, 0, 1)?.squeeze(1)?; // [B*T, D]
        cls.reshape((b, t, HIDDEN))
    }
}

/// `AceStepAudioTokenizer`: acoustic features → quantized 5Hz tokens + FSQ indices.
pub struct AceStepAudioTokenizer {
    audio_acoustic_proj: Linear,
    attention_pooler: AttentionPooler,
    pub quantizer: ResidualFsq,
}

impl AceStepAudioTokenizer {
    pub fn load(vb: VarBuilder, dtype: DType, dev: &Device) -> Result<Self> {
        let audio_acoustic_proj = candle_nn::linear(ACOUSTIC, HIDDEN, vb.pp("audio_acoustic_proj"))?;
        let attention_pooler = AttentionPooler::load(vb.pp("attention_pooler"))?;
        let quantizer = ResidualFsq::load(vb.pp("quantizer"), dtype, dev)?;
        Ok(Self {
            audio_acoustic_proj,
            attention_pooler,
            quantizer,
        })
    }

    /// `x` : `[B, Tp, P, 64]` → `(quantized [B, Tp, 2048], indices [B, Tp])`.
    pub fn forward(&self, x: &Tensor) -> candle_core::Result<(Tensor, Tensor)> {
        let h = linear_last(&self.audio_acoustic_proj, x)?; // [B,Tp,P,2048]
        let h = self.attention_pooler.forward(&h)?; // [B,Tp,2048]
        self.quantizer.encode(&h)
    }
}

/// `AudioTokenDetokenizer`: quantized tokens → 25Hz latents.
pub struct AudioTokenDetokenizer {
    embed_tokens: Linear,
    special_tokens: Tensor, // [1, P, 2048]
    layers: Vec<AceStepLyricLayer>,
    norm: AudioRmsNorm,
    proj_out: Linear,
    rope: AudioRotaryEmbedding,
}

impl AudioTokenDetokenizer {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let embed_tokens = candle_nn::linear(HIDDEN, HIDDEN, vb.pp("embed_tokens"))?;
        let special_tokens = vb.get((1, POOL_WINDOW_SIZE, HIDDEN), "special_tokens")?;
        let mut layers = Vec::with_capacity(2);
        for i in 0..2 {
            layers.push(AceStepLyricLayer::load(vb.pp(format!("layers.{}", i)))?);
        }
        let norm = AudioRmsNorm::new(HIDDEN, 1e-6, vb.pp("norm"))?;
        let proj_out = candle_nn::linear(HIDDEN, ACOUSTIC, vb.pp("proj_out"))?;
        Ok(Self {
            embed_tokens,
            special_tokens,
            layers,
            norm,
            proj_out,
            rope: AudioRotaryEmbedding::new(128, 1_000_000.0),
        })
    }

    /// `tokens` : `[B, T, 2048]` → `[B, T*P, 64]`.
    pub fn forward(&self, tokens: &Tensor) -> candle_core::Result<Tensor> {
        let (b, t, _d) = tokens.dims3()?;
        let x = self.embed_tokens.forward(tokens)?; // [B,T,2048]
        let x = x.unsqueeze(2)?.repeat((1, 1, POOL_WINDOW_SIZE, 1))?; // [B,T,P,2048]
        let x = x.broadcast_add(&self.special_tokens)?;
        let mut h = x.reshape((b * t, POOL_WINDOW_SIZE, HIDDEN))?;
        for layer in &self.layers {
            h = layer.forward(&h, &self.rope, None)?;
        }
        let h = self.norm.forward(&h)?;
        let h = linear_last(&self.proj_out, &h)?; // [B*T, P, 64]
        h.reshape((b, t * POOL_WINDOW_SIZE, ACOUSTIC))
    }
}

/// Full 5Hz codec facade.
pub struct AceStepAudioCodec {
    pub tokenizer: AceStepAudioTokenizer,
    pub detokenizer: AudioTokenDetokenizer,
}

impl AceStepAudioCodec {
    pub fn from_safetensors<P: AsRef<Path>>(path: P, device: &Device, dtype: DType) -> Result<Self> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[path.as_ref()], dtype, device)
                .with_context(|| format!("failed to load audio codec at {:?}", path.as_ref()))?
        };
        let tokenizer = AceStepAudioTokenizer::load(vb.pp("tokenizer"), dtype, device)?;
        let detokenizer = AudioTokenDetokenizer::load(vb.pp("detokenizer"))?;
        Ok(Self {
            tokenizer,
            detokenizer,
        })
    }

    /// Load from the split diffusers layout: `audio_tokenizer/…` (pooler + FSQ) and
    /// `audio_token_detokenizer/…` (each file's keys are already at the module root).
    pub fn from_diffusers<P: AsRef<Path>, Q: AsRef<Path>>(
        tokenizer_path: P,
        detokenizer_path: Q,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let vb_tok = unsafe {
            VarBuilder::from_mmaped_safetensors(&[tokenizer_path.as_ref()], dtype, device)
                .with_context(|| format!("failed to load audio tokenizer at {:?}", tokenizer_path.as_ref()))?
        };
        let vb_detok = unsafe {
            VarBuilder::from_mmaped_safetensors(&[detokenizer_path.as_ref()], dtype, device)
                .with_context(|| format!("failed to load audio detokenizer at {:?}", detokenizer_path.as_ref()))?
        };
        let tokenizer = AceStepAudioTokenizer::load(vb_tok, dtype, device)?;
        let detokenizer = AudioTokenDetokenizer::load(vb_detok)?;
        Ok(Self {
            tokenizer,
            detokenizer,
        })
    }

    /// Acoustic features `[B, T, 64]` (T divisible by 5) → `(quantized [B, T/5, 2048], indices [B, T/5])`.
    pub fn tokenize(&self, features: &Tensor) -> Result<(Tensor, Tensor)> {
        let (b, t, d) = features.dims3()?;
        anyhow::ensure!(d == ACOUSTIC, "expected {ACOUSTIC} features, got {d}");
        anyhow::ensure!(t % POOL_WINDOW_SIZE == 0, "T must be divisible by {POOL_WINDOW_SIZE}");
        let tp = t / POOL_WINDOW_SIZE;
        let x = features.reshape((b, tp, POOL_WINDOW_SIZE, d))?;
        Ok(self.tokenizer.forward(&x)?)
    }

    /// Quantized tokens `[B, Tp, 2048]` → latents `[B, Tp*5, 64]`.
    pub fn detokenize(&self, tokens: &Tensor) -> Result<Tensor> {
        Ok(self.detokenizer.forward(tokens)?)
    }

    /// Codebook indices `[B, Tp]` (u32) → latents `[B, Tp*5, 64]`.
    pub fn detokenize_from_indices(&self, indices: &Tensor) -> Result<Tensor> {
        let tokens = self.tokenizer.quantizer.get_output_from_indices(indices)?;
        self.detokenize(&tokens)
    }
}
