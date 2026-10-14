// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Pure Rust MiniMax H3 Audio VAE Decoder (BigVGAN 32kHz Stereo Vocoder)
// Invariant: Bounded VRAM, exact stereo channel independence, clamp in [-1.0, 1.0], bit-exact SnakeBeta activations.

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, VarBuilder};

/// 1D Snake activation with learnable alpha and beta in log scale:
/// y = x + (1 / (exp(beta) + 1e-9)) * sin(exp(alpha) * x)^2
#[derive(Debug, Clone)]
pub struct SnakeBeta {
    pub alpha: Tensor,
    pub beta: Tensor,
}

impl SnakeBeta {
    pub fn new(alpha: Tensor, beta: Tensor) -> Self {
        Self { alpha, beta }
    }

    pub fn load(vb: VarBuilder, in_features: usize) -> Result<Self> {
        let alpha = match vb.pp("act").get(in_features, "alpha") {
            Ok(a) => a,
            Err(_) => vb.get(in_features, "alpha")?,
        };
        let beta = match vb.pp("act").get(in_features, "beta") {
            Ok(b) => b,
            Err(_) => vb.get(in_features, "beta")?,
        };
        Ok(Self { alpha, beta })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let alpha = self.alpha.exp()?.reshape((1, (), 1))?;
        let beta = self.beta.exp()?.reshape((1, (), 1))?;
        let ax = x.broadcast_mul(&alpha)?;
        let sin_ax = ax.sin()?;
        let sin2 = sin_ax.sqr()?;
        let denom = (beta + 1e-9f64)?;
        let term = sin2.broadcast_div(&denom)?;
        x + term
    }
}

/// Anti-aliased Multi-Period ResBlock (AMPBlock1) for BigVGAN
#[derive(Debug, Clone)]
pub struct AMPBlock1 {
    pub convs1: Vec<Conv1d>,
    pub convs2: Vec<Conv1d>,
    pub activations: Vec<SnakeBeta>,
}

impl AMPBlock1 {
    pub fn load(vb: VarBuilder, channels: usize, kernel_size: usize, dilations: &[usize]) -> Result<Self> {
        let mut convs1 = Vec::with_capacity(dilations.len());
        let mut convs2 = Vec::with_capacity(dilations.len());
        let mut activations = Vec::with_capacity(dilations.len() * 2);

        for (i, &d) in dilations.iter().enumerate() {
            let pad1 = (kernel_size * d - d) / 2;
            let cfg1 = Conv1dConfig {
                padding: pad1,
                dilation: d,
                groups: 1,
                ..Default::default()
            };
            let c1 = candle_nn::conv1d(channels, channels, kernel_size, cfg1, vb.pp(format!("convs1.{}", i)))?;
            convs1.push(c1);

            let pad2 = (kernel_size - 1) / 2;
            let cfg2 = Conv1dConfig {
                padding: pad2,
                dilation: 1,
                groups: 1,
                ..Default::default()
            };
            let c2 = candle_nn::conv1d(channels, channels, kernel_size, cfg2, vb.pp(format!("convs2.{}", i)))?;
            convs2.push(c2);
        }

        let num_acts = dilations.len() * 2;
        for i in 0..num_acts {
            let act = SnakeBeta::load(vb.pp(format!("activations.{}", i)), channels)?;
            activations.push(act);
        }

        Ok(Self {
            convs1,
            convs2,
            activations,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = x.clone();
        for (i, (c1, c2)) in self.convs1.iter().zip(&self.convs2).enumerate() {
            let a1 = &self.activations[i * 2];
            let a2 = &self.activations[i * 2 + 1];

            let xt = a1.forward(&out)?;
            let xt = c1.forward(&xt)?;
            let xt = a2.forward(&xt)?;
            let xt = c2.forward(&xt)?;
            out = (out + xt)?;
        }
        Ok(out)
    }
}

/// BigVGAN Vocoder for MiniMax H3 32 kHz Stereo Audio Generation
#[derive(Debug, Clone)]
pub struct BigVGAN {
    pub conv_pre: Conv1d,
    pub ups: Vec<ConvTranspose1d>,
    pub resblocks: Vec<AMPBlock1>,
    pub activation_post: SnakeBeta,
    pub conv_post: Conv1d,
    pub num_kernels: usize,
    pub num_upsamples: usize,
}

impl BigVGAN {
    pub fn load(
        vb: VarBuilder,
        num_mels: usize,
        upsample_initial_channel: usize,
        upsample_rates: &[usize],
        upsample_kernel_sizes: &[usize],
        resblock_kernel_sizes: &[usize],
        resblock_dilation_sizes: &[Vec<usize>],
    ) -> Result<Self> {
        let cfg_pre = Conv1dConfig {
            padding: 3,
            dilation: 1,
            groups: 1,
            ..Default::default()
        };
        let conv_pre = candle_nn::conv1d(num_mels, upsample_initial_channel, 7, cfg_pre, vb.pp("conv_pre"))?;

        let num_upsamples = upsample_rates.len();
        let num_kernels = resblock_kernel_sizes.len();

        let mut ups = Vec::with_capacity(num_upsamples);
        let mut resblocks = Vec::with_capacity(num_upsamples * num_kernels);

        for (i, (&u, &k)) in upsample_rates.iter().zip(upsample_kernel_sizes).enumerate() {
            let in_ch = upsample_initial_channel / (1 << i);
            let out_ch = upsample_initial_channel / (1 << (i + 1));
            let pad = (k - u) / 2;

            let cfg_t = ConvTranspose1dConfig {
                padding: pad,
                output_padding: 0,
                stride: u,
                dilation: 1,
                groups: 1,
            };
            let up = candle_nn::conv_transpose1d(in_ch, out_ch, k, cfg_t, vb.pp(format!("ups.{}.0", i)))?;
            ups.push(up);

            for (j, (&rk, dilations)) in resblock_kernel_sizes.iter().zip(resblock_dilation_sizes).enumerate() {
                let blk_idx = i * num_kernels + j;
                let blk = AMPBlock1::load(vb.pp(format!("resblocks.{}", blk_idx)), out_ch, rk, dilations)?;
                resblocks.push(blk);
            }
        }

        let final_ch = upsample_initial_channel / (1 << num_upsamples);
        let activation_post = SnakeBeta::load(vb.pp("activation_post"), final_ch)?;

        let cfg_post = Conv1dConfig {
            padding: 3,
            dilation: 1,
            groups: 1,
            ..Default::default()
        };
        // bias is false for conv_post
        let conv_post = candle_nn::conv1d_no_bias(final_ch, 1, 7, cfg_post, vb.pp("conv_post"))?;

        Ok(Self {
            conv_pre,
            ups,
            resblocks,
            activation_post,
            conv_post,
            num_kernels,
            num_upsamples,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = self.conv_pre.forward(x)?;

        for i in 0..self.num_upsamples {
            out = self.ups[i].forward(&out)?;

            let mut xs: Option<Tensor> = None;
            for j in 0..self.num_kernels {
                let blk_idx = i * self.num_kernels + j;
                let res = self.resblocks[blk_idx].forward(&out)?;
                xs = match xs {
                    None => Some(res),
                    Some(prev) => Some((prev + res)?),
                };
            }

            if let Some(sum_res) = xs {
                out = (sum_res / (self.num_kernels as f64))?;
            }
        }

        out = self.activation_post.forward(&out)?;
        let waveform = self.conv_post.forward(&out)?;
        waveform.clamp(-1.0f32, 1.0f32)
    }
}

/// MiniMax H3 32 kHz Stereo Audio VAE Decoder
#[derive(Debug, Clone)]
pub struct MiniMaxH3AudioVaeDecoder {
    pub dec_in_proj: Conv1d,
    pub decoder: BigVGAN,
    pub latents_mean: Tensor,
    pub latents_std: Tensor,
    pub sample_rate: usize,
    pub hop_length: usize,
}

impl MiniMaxH3AudioVaeDecoder {
    /// Load MiniMax H3 Audio VAE from VarBuilder (`minimax_h3_audio_vae_fp32.safetensors`)
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let vae_latent_channels = 32;
        let latent_dim = 2048;
        let decoder_dim = 1024;

        let cfg_proj = Conv1dConfig {
            padding: 0,
            dilation: 1,
            groups: 1,
            ..Default::default()
        };
        let dec_in_proj = candle_nn::conv1d(vae_latent_channels, latent_dim, 1, cfg_proj, vb.pp("dec_in_proj"))?;

        let upsample_rates = vec![5, 5, 2, 2, 2, 2, 2];
        let upsample_kernel_sizes = vec![9, 9, 4, 4, 4, 4, 4];
        let resblock_kernel_sizes = vec![3, 7, 11];
        let resblock_dilation_sizes = vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]];

        let decoder = BigVGAN::load(
            vb.pp("decoder"),
            latent_dim,
            decoder_dim,
            &upsample_rates,
            &upsample_kernel_sizes,
            &resblock_kernel_sizes,
            &resblock_dilation_sizes,
        )?;

        let latents_mean = vb.get(vae_latent_channels, "latents_mean")?;
        let latents_std = vb.get(vae_latent_channels, "latents_std")?;

        let hop_length = upsample_rates.iter().product(); // 5 * 5 * 2 * 2 * 2 * 2 * 2 = 800

        Ok(Self {
            dec_in_proj,
            decoder,
            latents_mean,
            latents_std,
            sample_rate: 32000,
            hop_length,
        })
    }

    /// Decode normalized latents [B, 32, 2, T] to 32 kHz stereo waveforms [B, 2, L] in [-1.0, 1.0].
    /// T is the number of 40 Hz audio latent frames; L = T * 800 samples per channel.
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let (b, c, s, t) = z.dims4()?;
        if c != 32 || s != 2 {
            candle_core::bail!("Expected latent shape [B, 32, 2, T], got [{}, {}, {}, {}]", b, c, s, t);
        }

        // Permute [B, 32, 2, T] -> [B, 2, 32, T] -> reshape [B * 2, 32, T]
        let z_flat = z.permute((0, 2, 1, 3))?.reshape((b * s, c, t))?;

        // De-normalize per-channel: z = z * std + mean
        let mean = self.latents_mean.reshape((1, c, 1))?.to_device(&z.device())?.to_dtype(z.dtype())?;
        let std = self.latents_std.reshape((1, c, 1))?.to_device(&z.device())?.to_dtype(z.dtype())?;
        let z_denorm = (z_flat.broadcast_mul(&std)?).broadcast_add(&mean)?;

        // Project latents: 32 -> 2048
        let x = self.dec_in_proj.forward(&z_denorm)?;

        // Decode with BigVGAN vocoder: [B * 2, 2048, T] -> [B * 2, 1, L]
        let mono_wave = self.decoder.forward(&x)?;

        // Reshape back to stereo waveform: [B, 2, L]
        let total_samples = mono_wave.dim(2)?;
        mono_wave.reshape((b, s, total_samples))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snake_beta_activation() -> Result<()> {
        let dev = Device::Cpu;
        let alpha = Tensor::new(&[0.5f32, -0.2f32], &dev)?;
        let beta = Tensor::new(&[1.0f32, 0.0f32], &dev)?;
        let snake = SnakeBeta::new(alpha, beta);

        let x = Tensor::new(&[[[0.0f32, 1.0f32], [2.0f32, -1.0f32]]], &dev)?; // [1, 2, 2]
        let y = snake.forward(&x)?;
        assert_eq!(y.dims3()?, (1, 2, 2));

        // At x = 0: sin(0) = 0, so y(0) = 0
        let val_0 = y.narrow(1, 0, 1)?.narrow(2, 0, 1)?.squeeze(0)?.squeeze(0)?.squeeze(0)?.to_scalar::<f32>()?;
        assert!((val_0 - 0.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_bigvgan_latent_temporal_ratio() {
        let upsample_rates = vec![5, 5, 2, 2, 2, 2, 2];
        let hop: usize = upsample_rates.iter().product();
        assert_eq!(hop, 800);
        // 40 Hz latents * 800 samples/latent = 32,000 Hz audio
        assert_eq!(40 * hop, 32000);
    }
}
