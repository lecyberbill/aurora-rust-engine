// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Pure Rust Vocos Neural Vocoder & ISTFT Acoustic Synthesis

use anyhow::{Context, Result};
use candle_core::{DType, Module, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, LayerNorm, LayerNormConfig, Linear, VarBuilder};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Configuration for the Vocos neural vocoder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocosConfig {
    pub input_dim: usize,
    pub dim: usize,
    pub intermediate_dim: usize,
    pub num_layers: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub sample_rate: usize,
}

impl Default for VocosConfig {
    fn default() -> Self {
        Self {
            input_dim: 512,
            dim: 512,
            intermediate_dim: 1536,
            num_layers: 8,
            n_fft: 1024,
            hop_length: 256,
            sample_rate: 24000,
        }
    }
}

/// 1D ConvNeXt block tailored for audio signals.
pub struct ConvNeXtBlock {
    dwconv: Conv1d,
    norm: LayerNorm,
    pwconv1: Linear,
    pwconv2: Linear,
    gamma: Option<Tensor>,
}

impl ConvNeXtBlock {
    pub fn load(dim: usize, intermediate_dim: usize, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv1dConfig {
            padding: 3,
            groups: dim,
            ..Default::default()
        };
        let dwconv = candle_nn::conv1d(dim, dim, 7, conv_cfg, vb.pp("dwconv"))?;
        let ln_cfg = LayerNormConfig {
            eps: 1e-6,
            ..Default::default()
        };
        let norm = candle_nn::layer_norm(dim, ln_cfg, vb.pp("norm"))?;
        let pwconv1 = candle_nn::linear(dim, intermediate_dim, vb.pp("pwconv1"))?;
        let pwconv2 = candle_nn::linear(intermediate_dim, dim, vb.pp("pwconv2"))?;
        let gamma = vb.get(dim, "gamma").ok();

        Ok(Self {
            dwconv,
            norm,
            pwconv1,
            pwconv2,
            gamma,
        })
    }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // x: [B, C, T]
        let residual = x.clone();
        let h = self.dwconv.forward(x)?; // [B, C, T]

        // Permute to [B, T, C] for LayerNorm & Linear layers
        let h = h.transpose(1, 2)?;
        let h = self.norm.forward(&h)?;
        let h = self.pwconv1.forward(&h)?;
        let h = h.gelu()?;
        let mut h = self.pwconv2.forward(&h)?;

        if let Some(ref gamma) = self.gamma {
            h = h.broadcast_mul(gamma)?;
        }

        let h = h.transpose(1, 2)?; // [B, C, T]
        residual.add(&h)
    }
}

/// ConvNeXt backbone for Vocos.
pub struct VocosBackbone {
    in_proj: Conv1d,
    norm: LayerNorm,
    blocks: Vec<ConvNeXtBlock>,
    final_norm: LayerNorm,
}

impl VocosBackbone {
    pub fn load(cfg: &VocosConfig, vb: VarBuilder) -> Result<Self> {
        let in_conv_cfg = Conv1dConfig {
            padding: 3,
            ..Default::default()
        };
        let in_proj = candle_nn::conv1d(cfg.input_dim, cfg.dim, 7, in_conv_cfg, vb.pp("embed"))?;
        let ln_cfg = LayerNormConfig {
            eps: 1e-6,
            ..Default::default()
        };
        let norm = candle_nn::layer_norm(cfg.dim, ln_cfg, vb.pp("norm"))?;

        let mut blocks = Vec::with_capacity(cfg.num_layers);
        let blocks_vb = vb.pp("convnext");
        for i in 0..cfg.num_layers {
            blocks.push(ConvNeXtBlock::load(cfg.dim, cfg.intermediate_dim, blocks_vb.pp(i))?);
        }

        let final_norm = candle_nn::layer_norm(cfg.dim, ln_cfg, vb.pp("final_layer_norm"))?;

        Ok(Self {
            in_proj,
            norm,
            blocks,
            final_norm,
        })
    }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // x: [B, C, T]
        let h = self.in_proj.forward(x)?;
        let h = h.transpose(1, 2)?;
        let h = self.norm.forward(&h)?;
        let mut h = h.transpose(1, 2)?;
        for block in &self.blocks {
            h = block.forward(&h)?;
        }
        let h = h.transpose(1, 2)?;
        let h = self.final_norm.forward(&h)?;
        h.transpose(1, 2)
    }
}

/// ISTFT prediction head mapping features to spectral coefficients and waveform.
pub struct ISTFTHead {
    out_proj: Linear,
    n_fft: usize,
    hop_length: usize,
}

impl ISTFTHead {
    pub fn load(dim: usize, n_fft: usize, hop_length: usize, vb: VarBuilder) -> Result<Self> {
        let num_bins = n_fft / 2 + 1;
        let out_proj = candle_nn::linear(dim, num_bins * 2, vb.pp("out"))?;
        Ok(Self {
            out_proj,
            n_fft,
            hop_length,
        })
    }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // x: [B, C, T] -> transpose to [B, T, C]
        let h = x.transpose(1, 2)?;
        let logits = self.out_proj.forward(&h)?; // [B, T, 2 * num_bins]
        logits.transpose(1, 2) // [B, 2 * num_bins, T]
    }

    /// Reconstruct continuous audio waveform from predicted spectral coefficients.
    pub fn decode_waveform(&self, spectral_tensor: &Tensor) -> Result<Vec<f32>> {
        let (b, total_bins, t) = spectral_tensor.dims3()?;
        let num_bins = self.n_fft / 2 + 1;
        anyhow::ensure!(b >= 1, "Batch size must be >= 1");
        anyhow::ensure!(total_bins >= num_bins * 2, "Invalid spectral tensor channels: expected at least {}", num_bins * 2);

        let data = spectral_tensor.squeeze(0)?.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        
        let mut mag = Vec::with_capacity(num_bins * t);
        let mut phase = Vec::with_capacity(num_bins * t);

        for bin in 0..num_bins {
            for time_idx in 0..t {
                let mag_log = data[bin][time_idx];
                let p = data[num_bins + bin][time_idx];
                mag.push(mag_log.clamp(-15.0, 15.0).exp());
                phase.push(p);
            }
        }

        let waveform = istft_synthesis(&mag, &phase, num_bins, t, self.n_fft, self.hop_length);
        Ok(waveform)
    }
}

/// Full Vocos neural vocoder model.
pub struct Vocos {
    pub cfg: VocosConfig,
    pub backbone: VocosBackbone,
    pub head: ISTFTHead,
}

impl Vocos {
    pub fn load(cfg: VocosConfig, vb: VarBuilder) -> Result<Self> {
        let backbone = VocosBackbone::load(&cfg, vb.pp("backbone"))?;
        let head = ISTFTHead::load(cfg.dim, cfg.n_fft, cfg.hop_length, vb.pp("head"))?;
        Ok(Self { cfg, backbone, head })
    }

    pub fn decode(&self, features: &Tensor) -> Result<Vec<f32>> {
        let hidden = self.backbone.forward(features)
            .context("Failed in Vocos backbone forward pass")?;
        let spectral = self.head.forward(&hidden)
            .context("Failed in Vocos ISTFT head forward pass")?;
        self.head.decode_waveform(&spectral)
            .context("Failed during ISTFT waveform reconstruction")
    }
}

/// Pure Rust Inverse Short-Time Fourier Transform (ISTFT) with Hann window and Overlap-Add.
pub fn istft_synthesis(
    mag: &[f32],
    phase: &[f32],
    num_bins: usize,
    num_frames: usize,
    n_fft: usize,
    hop_length: usize,
) -> Vec<f32> {
    let win_len = n_fft;
    let out_len = (num_frames - 1) * hop_length + win_len;
    let mut out = vec![0.0f32; out_len];
    let mut win_sum = vec![0.0f32; out_len];

    // Precompute symmetric Hann window
    let mut window = Vec::with_capacity(win_len);
    for n in 0..win_len {
        let w = 0.5 * (1.0 - (2.0 * PI * (n as f32) / (win_len as f32)).cos());
        window.push(w);
    }

    // Precompute cos/sin tables for IDFT
    let mut real_spec = vec![0.0f32; n_fft];
    let mut imag_spec = vec![0.0f32; n_fft];
    let mut frame_out = vec![0.0f32; win_len];

    for k in 0..num_frames {
        // Build full Hermitian symmetric spectrum from (mag, phase)
        for bin in 0..num_bins {
            let m = mag[bin * num_frames + k];
            let p = phase[bin * num_frames + k];
            let re = m * p.cos();
            let im = m * p.sin();
            real_spec[bin] = re;
            imag_spec[bin] = im;
            if bin > 0 && bin < num_bins - 1 {
                real_spec[n_fft - bin] = re;
                imag_spec[n_fft - bin] = -im;
            }
        }

        // Inverse Discrete Fourier Transform (IDFT)
        let inv_n = 1.0 / (n_fft as f32);
        for n in 0..win_len {
            let mut sum_val = 0.0f32;
            for m in 0..n_fft {
                let angle = 2.0 * PI * (m as f32) * (n as f32) * inv_n;
                sum_val += real_spec[m] * angle.cos() - imag_spec[m] * angle.sin();
            }
            frame_out[n] = sum_val * inv_n;
        }

        // Overlap-Add with Hann window
        let start = k * hop_length;
        for n in 0..win_len {
            if start + n < out_len {
                let w = window[n];
                out[start + n] += frame_out[n] * w;
                win_sum[start + n] += w * w;
            }
        }
    }

    // Window normalization
    for i in 0..out_len {
        if win_sum[i] > 1e-6 {
            out[i] /= win_sum[i];
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_istft_reconstruction_shape() {
        let num_bins = 513; // n_fft = 1024 -> 513 bins
        let num_frames = 10;
        let n_fft = 1024;
        let hop_length = 256;

        let mag = vec![1.0f32; num_bins * num_frames];
        let phase = vec![0.0f32; num_bins * num_frames];

        let audio = istft_synthesis(&mag, &phase, num_bins, num_frames, n_fft, hop_length);
        let expected_len = (num_frames - 1) * hop_length + n_fft;
        assert_eq!(audio.len(), expected_len);
    }
}
