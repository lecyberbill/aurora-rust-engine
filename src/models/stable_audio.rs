// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Stable Audio Open conditioning (number projection)

//! Stable Audio Open conditioning: the Fourier "number" conditioner that maps
//! `seconds_start` / `seconds_end` to the shared conditioning space (768-d), mirroring
//! `diffusers.pipelines.stable_audio.modeling_stable_audio`.

use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{linear, Linear, VarBuilder};
use std::f32::consts::PI;
use std::path::Path;

/// `StableAudioPositionalEmbedding`: `[B] -> [B, 1 + 2*(dim/2)]` (`cat([times, sin, cos])`).
pub struct StableAudioPositionalEmbedding {
    weights: Tensor, // [dim/2]
}

impl StableAudioPositionalEmbedding {
    pub fn load(vb: VarBuilder, dim: usize) -> Result<Self> {
        Ok(Self {
            weights: vb.get((dim / 2,), "weights")?,
        })
    }

    pub fn forward(&self, times: &Tensor) -> candle_core::Result<Tensor> {
        let t = times.unsqueeze(1)?; // [B,1]
        let w = self.weights.unsqueeze(0)?.to_dtype(times.dtype())?; // [1, dim/2]
        let freqs = t.broadcast_mul(&w)?.affine((2.0 * PI) as f64, 0.0)?; // [B, dim/2]
        let fouriered = Tensor::cat(&[&freqs.sin()?, &freqs.cos()?], 1)?; // [B, dim]
        Tensor::cat(&[&t, &fouriered], 1) // [B, 1 + dim]
    }
}

/// `StableAudioNumberConditioner`: clamp → normalize → Fourier embed → linear → `[B, 1, dim]`.
pub struct StableAudioNumberConditioner {
    time_positional_embedding: StableAudioPositionalEmbedding,
    proj: Linear,
    min_value: f32,
    max_value: f32,
    cond_dim: usize,
}

impl StableAudioNumberConditioner {
    pub fn load(
        vb: VarBuilder,
        cond_dim: usize,
        min_value: f32,
        max_value: f32,
        internal_dim: usize,
        dtype: DType,
        dev: &Device,
    ) -> Result<Self> {
        let vb_pe = vb.pp("time_positional_embedding");
        let time_positional_embedding = StableAudioPositionalEmbedding::load(vb_pe.pp("0"), internal_dim)?;
        let proj = linear(internal_dim + 1, cond_dim, vb_pe.pp("1"))?;
        let _ = (dtype, dev);
        Ok(Self {
            time_positional_embedding,
            proj,
            min_value,
            max_value,
            cond_dim,
        })
    }

    pub fn forward(&self, floats: &Tensor) -> candle_core::Result<Tensor> {
        let clamped = floats.clamp(self.min_value as f64, self.max_value as f64)?;
        let normalized = ((&clamped - self.min_value as f64)? / (self.max_value - self.min_value) as f64)?;
        let embedding = self.time_positional_embedding.forward(&normalized)?; // [B, 1+dim]
        let out = self.proj.forward(&embedding)?; // [B, cond_dim]
        out.reshape(((), 1, self.cond_dim))
    }
}

/// Outputs of [`StableAudioProjectionModel`].
pub struct StableAudioProjectionOutput {
    pub text_hidden_states: Option<Tensor>,
    pub seconds_start_hidden_states: Option<Tensor>,
    pub seconds_end_hidden_states: Option<Tensor>,
}

/// `StableAudioProjectionModel`: identity text projection + start/end number conditioners.
pub struct StableAudioProjectionModel {
    start_number_conditioner: StableAudioNumberConditioner,
    end_number_conditioner: StableAudioNumberConditioner,
    min_value: f32,
    max_value: f32,
}

impl StableAudioProjectionModel {
    /// Load from the diffusers `projection_model/diffusion_pytorch_model.safetensors`.
    pub fn from_safetensors<P: AsRef<Path>>(path: P, device: &Device, dtype: DType) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path.as_ref()], dtype, device)? };
        // `projection_model/config.json`: conditioning_dim 768, min 0, max 512.
        let cond_dim = 768;
        let internal_dim = 256;
        let start_number_conditioner =
            StableAudioNumberConditioner::load(vb.pp("start_number_conditioner"), cond_dim, 0.0, 512.0, internal_dim, dtype, device)?;
        let end_number_conditioner =
            StableAudioNumberConditioner::load(vb.pp("end_number_conditioner"), cond_dim, 0.0, 512.0, internal_dim, dtype, device)?;
        Ok(Self {
            start_number_conditioner,
            end_number_conditioner,
            min_value: 0.0,
            max_value: 512.0,
        })
    }

    pub fn forward(
        &self,
        text_hidden_states: Option<&Tensor>,
        start_seconds: Option<&Tensor>,
        end_seconds: Option<&Tensor>,
    ) -> Result<StableAudioProjectionOutput> {
        let seconds_start_hidden_states = match start_seconds {
            Some(s) => Some(self.start_number_conditioner.forward(s)?),
            None => None,
        };
        let seconds_end_hidden_states = match end_seconds {
            Some(s) => Some(self.end_number_conditioner.forward(s)?),
            None => None,
        };
        Ok(StableAudioProjectionOutput {
            text_hidden_states: text_hidden_states.cloned(),
            seconds_start_hidden_states,
            seconds_end_hidden_states,
        })
    }
}
