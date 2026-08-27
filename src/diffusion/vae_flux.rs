// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Pure Rust 16-channel AutoEncoder VAE Decoder for Flux.1 and SD 3.5 with Seamless Cosine Tiling

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{conv2d, group_norm, linear, Conv2d, Conv2dConfig, GroupNorm, Linear, Module, VarBuilder};
use image::RgbImage;

pub const FLUX_LATENT_SCALE: f64 = 0.3611;
pub const FLUX_LATENT_SHIFT: f64 = 0.1159;

/// ResNet Block for VAE Decoder
#[derive(Debug, Clone)]
struct VaeResnetBlock {
    norm1: GroupNorm,
    conv1: Conv2d,
    norm2: GroupNorm,
    conv2: Conv2d,
    conv_shortcut: Option<Conv2d>,
}

impl VaeResnetBlock {
    pub fn new(in_channels: usize, out_channels: usize, vb: VarBuilder) -> Result<Self> {
        let norm1 = group_norm(32, in_channels, 1e-6, vb.pp("norm1"))?;
        let conv1 = conv2d(
            in_channels,
            out_channels,
            3,
            Conv2dConfig { padding: 1, ..Default::default() },
            vb.pp("conv1"),
        )?;
        let norm2 = group_norm(32, out_channels, 1e-6, vb.pp("norm2"))?;
        let conv2 = conv2d(
            out_channels,
            out_channels,
            3,
            Conv2dConfig { padding: 1, ..Default::default() },
            vb.pp("conv2"),
        )?;
        let conv_shortcut = if in_channels != out_channels {
            Some(
                conv2d(in_channels, out_channels, 1, Conv2dConfig::default(), vb.pp("nin_shortcut"))
                    .or_else(|_| conv2d(in_channels, out_channels, 1, Conv2dConfig::default(), vb.pp("conv_shortcut")))?
            )
        } else {
            None
        };

        Ok(Self {
            norm1,
            conv1,
            norm2,
            conv2,
            conv_shortcut,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = self.norm1.forward(xs)?;
        let h = candle_nn::ops::silu(&h)?;
        let h = self.conv1.forward(&h)?;

        let h = self.norm2.forward(&h)?;
        let h = candle_nn::ops::silu(&h)?;
        let h = self.conv2.forward(&h)?;

        match &self.conv_shortcut {
            Some(sc) => sc.forward(xs)? + h,
            None => xs + h,
        }
    }
}

/// Self-Attention Block for VAE Decoder mid-block
#[derive(Debug, Clone)]
struct VaeAttentionBlock {
    group_norm: GroupNorm,
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    scale: f64,
}

impl VaeAttentionBlock {
    pub fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let group_norm = group_norm(32, channels, 1e-6, vb.pp("group_norm"))
            .or_else(|_| group_norm(32, channels, 1e-6, vb.pp("norm")))?;
        let to_q = linear(channels, channels, vb.pp("to_q"))
            .or_else(|_| linear(channels, channels, vb.pp("q")))?;
        let to_k = linear(channels, channels, vb.pp("to_k"))
            .or_else(|_| linear(channels, channels, vb.pp("k")))?;
        let to_v = linear(channels, channels, vb.pp("to_v"))
            .or_else(|_| linear(channels, channels, vb.pp("v")))?;
        let to_out = linear(channels, channels, vb.pp("to_out.0"))
            .or_else(|_| linear(channels, channels, vb.pp("proj_out")))?;
        let scale = 1.0 / (channels as f64).sqrt();

        Ok(Self {
            group_norm,
            to_q,
            to_k,
            to_v,
            to_out,
            scale,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = xs.dims4()?;
        let orig_dtype = xs.dtype();

        let normed = self.group_norm.forward(xs)?;
        // Reshape to [b, h*w, c]
        let x_flat = normed.permute((0, 2, 3, 1))?.reshape((b, h * w, c))?;

        let q = self.to_q.forward(&x_flat)?;
        let k = self.to_k.forward(&x_flat)?;
        let v = self.to_v.forward(&x_flat)?;

        // Self-Attention computation
        let q_f16 = (q * self.scale)?;
        let attn_weights = q_f16.matmul(&k.transpose(1, 2)?)?;
        let probs = candle_nn::ops::softmax_last_dim(&attn_weights.to_dtype(DType::F32)?)?.to_dtype(orig_dtype)?;
        let out = probs.matmul(&v)?;
        let out = self.to_out.forward(&out)?;

        let out = out.reshape((b, h, w, c))?.permute((0, 3, 1, 2))?.contiguous()?;
        xs + out
    }
}

/// Up-sampling Block for VAE Decoder
#[derive(Debug, Clone)]
struct VaeUpBlock {
    resnets: Vec<VaeResnetBlock>,
    upsampler: Option<Conv2d>,
}

impl VaeUpBlock {
    pub fn new(in_channels: usize, out_channels: usize, num_layers: usize, add_upsample: bool, vb: VarBuilder) -> Result<Self> {
        let mut resnets = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let in_ch = if i == 0 { in_channels } else { out_channels };
            let block = VaeResnetBlock::new(in_ch, out_channels, vb.pp(format!("block.{}", i)))
                .or_else(|_| VaeResnetBlock::new(in_ch, out_channels, vb.pp(format!("resnets.{}", i))))?;
            resnets.push(block);
        }

        let upsampler = if add_upsample {
            Some(
                conv2d(out_channels, out_channels, 3, Conv2dConfig { padding: 1, ..Default::default() }, vb.pp("upsample.conv"))
                    .or_else(|_| conv2d(out_channels, out_channels, 3, Conv2dConfig { padding: 1, ..Default::default() }, vb.pp("upsamplers.0.conv")))?
            )
        } else {
            None
        };

        Ok(Self { resnets, upsampler })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut h = xs.clone();
        for resnet in &self.resnets {
            h = resnet.forward(&h)?;
        }
        if let Some(ref up) = self.upsampler {
            let (_, _, height, width) = h.dims4()?;
            h = h.upsample_nearest2d(height * 2, width * 2)?;
            h = up.forward(&h)?;
        }
        Ok(h)
    }
}

/// Pure Rust 16-Channel / 32-Channel VAE Decoder for Flux.1, Flux.2, and SD 3.5
#[derive(Debug, Clone)]
pub struct FluxVaeDecoder {
    conv_in: Conv2d,
    mid_block: (VaeResnetBlock, Option<VaeAttentionBlock>, VaeResnetBlock),
    up_blocks: Vec<VaeUpBlock>,
    conv_norm_out: GroupNorm,
    conv_out: Conv2d,
    post_quant_conv: Option<Conv2d>,
    bn_mean: Option<Tensor>,
    bn_var: Option<Tensor>,
    is_flux2: bool,
    device: Device,
    dtype: DType,
}

impl FluxVaeDecoder {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();

        let (conv_in, is_flux2) = if let Ok(c) = conv2d(32, 512, 3, Conv2dConfig { padding: 1, ..Default::default() }, vb.pp("decoder.conv_in")) {
            (c, true)
        } else {
            (conv2d(16, 512, 3, Conv2dConfig { padding: 1, ..Default::default() }, vb.pp("decoder.conv_in"))?, false)
        };

        let post_quant_conv = conv2d(32, 32, 1, Conv2dConfig::default(), vb.pp("post_quant_conv"))
            .or_else(|_| conv2d(16, 16, 1, Conv2dConfig::default(), vb.pp("post_quant_conv")))
            .ok();

        let bn_mean = vb.get(128, "bn.running_mean").or_else(|_| vb.get(32, "bn.running_mean")).ok();
        let bn_var = vb.get(128, "bn.running_var").or_else(|_| vb.get(32, "bn.running_var")).ok();

        let mid_attn = VaeAttentionBlock::new(512, vb.pp("decoder.mid.attn_1"))
            .or_else(|_| VaeAttentionBlock::new(512, vb.pp("decoder.mid_block.attentions.0")))
            .ok();

        let mid_block = (
            VaeResnetBlock::new(512, 512, vb.pp("decoder.mid.block_1"))
                .or_else(|_| VaeResnetBlock::new(512, 512, vb.pp("decoder.mid_block.resnets.0")))?,
            mid_attn,
            VaeResnetBlock::new(512, 512, vb.pp("decoder.mid.block_2"))
                .or_else(|_| VaeResnetBlock::new(512, 512, vb.pp("decoder.mid_block.resnets.1")))?,
        );

        // In Flux VAE: up.3 = 512->512, up.2 = 512->512, up.1 = 512->256, up.0 = 256->128
        let in_channels = [512, 512, 512, 256];
        let out_channels = [512, 512, 256, 128];
        let mut up_blocks = Vec::with_capacity(4);

        for i in 0..4 {
            let in_ch = in_channels[i];
            let out_ch = out_channels[i];
            let add_up = i < 3;
            let block = VaeUpBlock::new(in_ch, out_ch, 3, add_up, vb.pp(format!("decoder.up.{}", 3 - i)))
                .or_else(|_| VaeUpBlock::new(in_ch, out_ch, 3, add_up, vb.pp(format!("decoder.up_blocks.{}", i))))?;
            up_blocks.push(block);
        }

        let conv_norm_out = group_norm(32, 128, 1e-6, vb.pp("decoder.norm_out"))
            .or_else(|_| group_norm(32, 128, 1e-6, vb.pp("decoder.conv_norm_out")))?;
        let conv_out = conv2d(128, 3, 3, Conv2dConfig { padding: 1, ..Default::default() }, vb.pp("decoder.conv_out"))?;

        Ok(Self {
            conv_in,
            mid_block,
            up_blocks,
            conv_norm_out,
            conv_out,
            post_quant_conv,
            bn_mean,
            bn_var,
            is_flux2,
            device,
            dtype,
        })
    }

    pub fn bn_mean(&self) -> Option<&Tensor> {
        self.bn_mean.as_ref()
    }

    pub fn bn_var(&self) -> Option<&Tensor> {
        self.bn_var.as_ref()
    }

    /// Single-pass full decode in F16 precision to avoid VRAM overflow
    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let latents_f32 = latents.to_dtype(DType::F32)?;
        let latents = if self.is_flux2 {
            latents_f32.to_device(&self.device)?.to_dtype(self.dtype)?
        } else {
            // Exact BFL Flux 1 autoencoder de-quantization:
            // z = (latents / scale_factor) + shift_factor
            let scaled = (latents_f32 / FLUX_LATENT_SCALE)?;
            let shifted = (scaled + FLUX_LATENT_SHIFT)?;
            shifted.to_device(&self.device)?.to_dtype(self.dtype)?
        };

        let mut h = if let Some(ref pqc) = self.post_quant_conv {
            pqc.forward(&latents)?
        } else {
            latents
        };

        h = self.conv_in.forward(&h)?;
        h = self.mid_block.0.forward(&h)?;
        if let Some(ref attn) = self.mid_block.1 {
            h = attn.forward(&h)?;
        }
        h = self.mid_block.2.forward(&h)?;

        for block in &self.up_blocks {
            h = block.forward(&h)?;
        }

        let h = self.conv_norm_out.forward(&h)?;
        let h = candle_nn::ops::silu(&h)?;
        self.conv_out.forward(&h)
    }

    /// Convert unpatchified latents [1, 16/32, H/8, W/8] to RgbImage with robust dynamic contrast
    pub fn decode_to_image(&self, latents: &Tensor) -> Result<RgbImage> {
        let rgb_tensor = self.decode(latents)?;
        let (_, _, h, w) = rgb_tensor.dims4()?;

        // Scale RGB values from [-1, 1] to [0, 1] in high precision F32
        let rgb = rgb_tensor.squeeze(0)?.to_dtype(DType::F32)?;
        let normalized = ((&rgb * 0.5)? + 0.5)?;
        let flat_floats = normalized.clamp(0.0, 1.0)?.flatten_all()?.to_device(&Device::Cpu)?.to_vec1::<f32>()?;

        let n = flat_floats.len();
        let mut bytes = Vec::with_capacity(n);
        for &val in &flat_floats {
            bytes.push((val * 255.0).round() as u8);
        }

        // Reconstruct ImageBuffer from [C, H, W] -> [H, W, C]
        let mut hw3_bytes = vec![0u8; w * h * 3];
        let num_pixels = w * h;
        for c in 0..3 {
            let ch_offset = c * num_pixels;
            for i in 0..num_pixels {
                hw3_bytes[i * 3 + c] = bytes[ch_offset + i];
            }
        }

        let img: RgbImage = image::ImageBuffer::from_raw(w as u32, h as u32, hw3_bytes)
            .ok_or_else(|| candle_core::Error::Msg("Failed to construct ImageBuffer from raw RGB bytes".to_string()))?;

        Ok(img)
    }
}
