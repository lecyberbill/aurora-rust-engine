// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Pure Rust AutoencoderOobleck Audio VAE Decoder (48kHz DAC / Snake1d)

use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, VarBuilder};
use std::path::Path;

use crate::audio::WavAudio;

/// 1-Dimensional Snake activation module: y = x + (1 / (exp(beta) + 1e-9)) * sin(exp(alpha) * x)^2
#[derive(Debug, Clone)]
pub struct Snake1d {
    alpha: Tensor,
    beta: Tensor,
    logscale: bool,
}

impl Snake1d {
    pub fn load(vb: VarBuilder, hidden_dim: usize, logscale: bool) -> Result<Self> {
        let alpha = vb.get((1, hidden_dim, 1), "alpha")?;
        let beta = vb.get((1, hidden_dim, 1), "beta")?;
        Ok(Self {
            alpha,
            beta,
            logscale,
        })
    }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let alpha = if self.logscale {
            self.alpha.exp()?
        } else {
            self.alpha.clone()
        };
        let beta = if self.logscale {
            self.beta.exp()?
        } else {
            self.beta.clone()
        };

        let sin_term = (x.broadcast_mul(&alpha)?).sin()?.sqr()?;
        let denom = (beta + 1e-9f64)?;
        let scaled = sin_term.broadcast_div(&denom)?;
        x + scaled
    }
}

/// Compute weight-normalized weight tensor for Conv1d: W = g * (v / ||v||_2)
fn compute_weight_norm_conv1d(weight_g: &Tensor, weight_v: &Tensor) -> candle_core::Result<Tensor> {
    let (out_c, _, _) = weight_v.dims3()?;
    let norm = weight_v.sqr()?.sum_keepdim(2)?.sum_keepdim(1)?.sqrt()?;
    let normalized = weight_v.broadcast_div(&(norm + 1e-9)?)?;
    let g = weight_g.reshape((out_c, 1, 1))?;
    normalized.broadcast_mul(&g)
}

/// Compute weight-normalized weight tensor for ConvTranspose1d: W = g * (v / ||v||_2)
fn compute_weight_norm_conv_transpose1d(weight_g: &Tensor, weight_v: &Tensor) -> candle_core::Result<Tensor> {
    let (in_c, _, _) = weight_v.dims3()?;
    let norm = weight_v.sqr()?.sum_keepdim(2)?.sum_keepdim(1)?.sqrt()?;
    let normalized = weight_v.broadcast_div(&(norm + 1e-9)?)?;
    let g = weight_g.reshape((in_c, 1, 1))?;
    normalized.broadcast_mul(&g)
}

/// Weight-normalized 1D Convolution
#[derive(Debug, Clone)]
pub struct WeightNormConv1d {
    conv: Conv1d,
}

impl WeightNormConv1d {
    pub fn load(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        config: Conv1dConfig,
        has_bias: bool,
    ) -> Result<Self> {
        let weight_g = vb.get((out_channels, 1, 1), "weight_g")?;
        let weight_v = vb.get((out_channels, in_channels, kernel_size), "weight_v")?;
        let weight = compute_weight_norm_conv1d(&weight_g, &weight_v)
            .context("Failed to compute weight norm for Conv1d")?;

        let bias = if has_bias {
            Some(vb.get(out_channels, "bias")?)
        } else {
            None
        };

        Ok(Self {
            conv: Conv1d::new(weight, bias, config),
        })
    }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.conv.forward(x)
    }
}

/// Weight-normalized 1D Transposed Convolution
#[derive(Debug, Clone)]
pub struct WeightNormConvTranspose1d {
    conv_t: ConvTranspose1d,
}

impl WeightNormConvTranspose1d {
    pub fn load(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        config: ConvTranspose1dConfig,
        has_bias: bool,
    ) -> Result<Self> {
        let weight_g = vb.get((in_channels, 1, 1), "weight_g")?;
        let weight_v = vb.get((in_channels, out_channels, kernel_size), "weight_v")?;
        let weight = compute_weight_norm_conv_transpose1d(&weight_g, &weight_v)
            .context("Failed to compute weight norm for ConvTranspose1d")?;

        let bias = if has_bias {
            Some(vb.get(out_channels, "bias")?)
        } else {
            None
        };

        Ok(Self {
            conv_t: ConvTranspose1d::new(weight, bias, config),
        })
    }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.conv_t.forward(x)
    }
}

/// Oobleck Residual Unit with Snake1d and dilated Conv1d
#[derive(Debug, Clone)]
pub struct OobleckResidualUnit {
    snake1: Snake1d,
    conv1: WeightNormConv1d,
    snake2: Snake1d,
    conv2: WeightNormConv1d,
}

impl OobleckResidualUnit {
    pub fn load(vb: VarBuilder, dimension: usize, dilation: usize) -> Result<Self> {
        let pad = ((7 - 1) * dilation) / 2;
        let snake1 = Snake1d::load(vb.pp("snake1"), dimension, true)?;
        let conv1 = WeightNormConv1d::load(
            vb.pp("conv1"),
            dimension,
            dimension,
            7,
            Conv1dConfig {
                padding: pad,
                dilation,
                groups: 1,
                stride: 1,
                ..Default::default()
            },
            true,
        )?;

        let snake2 = Snake1d::load(vb.pp("snake2"), dimension, true)?;
        let conv2 = WeightNormConv1d::load(
            vb.pp("conv2"),
            dimension,
            dimension,
            1,
            Conv1dConfig {
                padding: 0,
                dilation: 1,
                groups: 1,
                stride: 1,
                ..Default::default()
            },
            true,
        )?;

        Ok(Self {
            snake1,
            conv1,
            snake2,
            conv2,
        })
    }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let res = self.conv1.forward(&self.snake1.forward(x)?)?;
        let res = self.conv2.forward(&self.snake2.forward(&res)?)?;

        let pad = (x.dim(2)? as i64 - res.dim(2)? as i64) / 2;
        let x_trimmed = if pad > 0 {
            let p = pad as usize;
            x.narrow(2, p, res.dim(2)?)?
        } else {
            x.clone()
        };

        &x_trimmed + &res
    }
}

/// Oobleck Decoder Block with Transposed Conv Upsampling + 3 MRF Residual Units
#[derive(Debug, Clone)]
pub struct OobleckDecoderBlock {
    snake1: Snake1d,
    conv_t1: WeightNormConvTranspose1d,
    res_unit1: OobleckResidualUnit,
    res_unit2: OobleckResidualUnit,
    res_unit3: OobleckResidualUnit,
}

impl OobleckDecoderBlock {
    pub fn load(vb: VarBuilder, input_dim: usize, output_dim: usize, stride: usize) -> Result<Self> {
        let pad = (stride as f64 / 2.0).ceil() as usize;
        let snake1 = Snake1d::load(vb.pp("snake1"), input_dim, true)?;
        let conv_t1 = WeightNormConvTranspose1d::load(
            vb.pp("conv_t1"),
            input_dim,
            output_dim,
            2 * stride,
            ConvTranspose1dConfig {
                padding: pad,
                output_padding: 0,
                stride,
                dilation: 1,
                groups: 1,
            },
            true,
        )?;

        let res_unit1 = OobleckResidualUnit::load(vb.pp("res_unit1"), output_dim, 1)?;
        let res_unit2 = OobleckResidualUnit::load(vb.pp("res_unit2"), output_dim, 3)?;
        let res_unit3 = OobleckResidualUnit::load(vb.pp("res_unit3"), output_dim, 9)?;

        Ok(Self {
            snake1,
            conv_t1,
            res_unit1,
            res_unit2,
            res_unit3,
        })
    }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.snake1.forward(x)?;
        let x = self.conv_t1.forward(&x)?;
        let x = self.res_unit1.forward(&x)?;
        let x = self.res_unit2.forward(&x)?;
        self.res_unit3.forward(&x)
    }
}

/// AutoencoderOobleck Decoder Configuration
#[derive(Debug, Clone)]
pub struct OobleckConfig {
    pub audio_channels: usize,
    pub channel_multiples: Vec<usize>,
    pub decoder_channels: usize,
    pub decoder_input_channels: usize,
    pub downsampling_ratios: Vec<usize>,
    pub sampling_rate: u32,
}

impl Default for OobleckConfig {
    fn default() -> Self {
        Self {
            audio_channels: 2,
            channel_multiples: vec![1, 2, 4, 8, 16],
            decoder_channels: 128,
            decoder_input_channels: 64,
            downsampling_ratios: vec![2, 4, 4, 6, 10],
            sampling_rate: 48000,
        }
    }
}

/// Pure Rust AutoencoderOobleck Audio VAE Decoder (48kHz Master Stereo Synthesizer)
pub struct AutoencoderOobleck {
    conv1: WeightNormConv1d,
    blocks: Vec<OobleckDecoderBlock>,
    snake1: Snake1d,
    conv2: WeightNormConv1d,
    pub config: OobleckConfig,
}

impl AutoencoderOobleck {
    pub fn load(vb: VarBuilder, config: OobleckConfig) -> Result<Self> {
        let vb_dec = vb.pp("decoder");
        let strides: Vec<usize> = config.downsampling_ratios.iter().cloned().rev().collect();
        let mut multiples = vec![1];
        multiples.extend_from_slice(&config.channel_multiples);

        let c_in = config.decoder_input_channels;
        let c_out_conv1 = config.decoder_channels * multiples[multiples.len() - 1];

        let conv1 = WeightNormConv1d::load(
            vb_dec.pp("conv1"),
            c_in,
            c_out_conv1,
            7,
            Conv1dConfig {
                padding: 3,
                dilation: 1,
                groups: 1,
                stride: 1,
                ..Default::default()
            },
            true,
        )?;

        let mut blocks = Vec::with_capacity(strides.len());
        for (i, &stride) in strides.iter().enumerate() {
            let in_dim = config.decoder_channels * multiples[strides.len() - i];
            let out_dim = config.decoder_channels * multiples[strides.len() - i - 1];
            let blk = OobleckDecoderBlock::load(
                vb_dec.pp(format!("block.{}", i)),
                in_dim,
                out_dim,
                stride,
            )?;
            blocks.push(blk);
        }

        let snake1 = Snake1d::load(vb_dec.pp("snake1"), config.decoder_channels, true)?;
        let conv2 = WeightNormConv1d::load(
            vb_dec.pp("conv2"),
            config.decoder_channels,
            config.audio_channels,
            7,
            Conv1dConfig {
                padding: 3,
                dilation: 1,
                groups: 1,
                stride: 1,
                ..Default::default()
            },
            false,
        )?;

        Ok(Self {
            conv1,
            blocks,
            snake1,
            conv2,
            config,
        })
    }

    /// Load directly from a safetensors file.
    pub fn from_safetensors<P: AsRef<Path>>(
        weights_path: P,
        config: OobleckConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path.as_ref()], dtype, device)
                .with_context(|| format!("Failed to mmap Oobleck weights at {:?}", weights_path.as_ref()))?
        };
        Self::load(vb, config)
    }

    /// Decode audio latent tokens `[batch, channels, time_frames]` into a WavAudio buffer.
    pub fn decode(&self, latents: &Tensor) -> candle_core::Result<WavAudio> {
        let mut x = self.conv1.forward(latents)?;

        for blk in &self.blocks {
            x = blk.forward(&x)?;
        }

        x = self.snake1.forward(&x)?;
        let waveform_tensor = self.conv2.forward(&x)?;

        // waveform_tensor is [batch, audio_channels, total_samples]
        let (batch, ch, total_samples) = waveform_tensor.dims3()?;
        let flat: Vec<f32> = waveform_tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;

        // If stereo (ch = 2), interleave [L0, R0, L1, R1...]
        let samples = if ch == 2 && batch == 1 {
            let mut interleaved = Vec::with_capacity(ch * total_samples);
            let l_chan = &flat[0..total_samples];
            let r_chan = &flat[total_samples..2 * total_samples];
            for i in 0..total_samples {
                interleaved.push(l_chan[i].clamp(-1.0, 1.0));
                interleaved.push(r_chan[i].clamp(-1.0, 1.0));
            }
            interleaved
        } else {
            flat.into_iter().map(|s| s.clamp(-1.0, 1.0)).collect()
        };

        Ok(WavAudio::new(
            samples,
            self.config.sampling_rate,
            ch as u16,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_snake1d_forward() -> candle_core::Result<()> {
        let dev = Device::Cpu;
        let alpha = Tensor::zeros((1, 4, 1), DType::F32, &dev)?;
        let beta = Tensor::zeros((1, 4, 1), DType::F32, &dev)?;
        let snake = Snake1d {
            alpha,
            beta,
            logscale: true,
        };

        let x = Tensor::randn(0.0f32, 1.0f32, (1, 4, 16), &dev)?;
        let y = snake.forward(&x)?;
        assert_eq!(y.dims(), &[1, 4, 16]);
        Ok(())
    }
}
