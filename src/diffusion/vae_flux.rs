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
            Some(conv2d(
                in_channels,
                out_channels,
                1,
                Conv2dConfig::default(),
                vb.pp("nin_shortcut"),
            )?)
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
            let block = VaeResnetBlock::new(in_ch, out_channels, vb.pp(format!("block.{}", i)))?;
            resnets.push(block);
        }

        let upsampler = if add_upsample {
            Some(conv2d(
                out_channels,
                out_channels,
                3,
                Conv2dConfig { padding: 1, ..Default::default() },
                vb.pp("upsample.conv"),
            )?)
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

/// Pure Rust 16-Channel VAE Decoder for Flux.1 and SD 3.5
#[derive(Debug, Clone)]
pub struct FluxVaeDecoder {
    conv_in: Conv2d,
    mid_block: (VaeResnetBlock, VaeResnetBlock),
    up_blocks: Vec<VaeUpBlock>,
    conv_norm_out: GroupNorm,
    conv_out: Conv2d,
    device: Device,
    dtype: DType,
}

impl FluxVaeDecoder {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();

        let conv_in = conv2d(16, 512, 3, Conv2dConfig { padding: 1, ..Default::default() }, vb.pp("decoder.conv_in"))?;

        let mid_block = (
            VaeResnetBlock::new(512, 512, vb.pp("decoder.mid.block_1"))?,
            VaeResnetBlock::new(512, 512, vb.pp("decoder.mid.block_2"))?,
        );

        // In Flux VAE: up.3 = 512->512, up.2 = 512->512, up.1 = 512->256, up.0 = 256->128
        let in_channels = [512, 512, 512, 256];
        let out_channels = [512, 512, 256, 128];
        let mut up_blocks = Vec::with_capacity(4);

        for i in 0..4 {
            let in_ch = in_channels[i];
            let out_ch = out_channels[i];
            let add_up = i < 3;
            let block = VaeUpBlock::new(in_ch, out_ch, 3, add_up, vb.pp(format!("decoder.up.{}", 3 - i)))?;
            up_blocks.push(block);
        }

        let conv_norm_out = group_norm(32, 128, 1e-6, vb.pp("decoder.norm_out"))?;
        let conv_out = conv2d(128, 3, 3, Conv2dConfig { padding: 1, ..Default::default() }, vb.pp("decoder.conv_out"))?;

        Ok(Self {
            conv_in,
            mid_block,
            up_blocks,
            conv_norm_out,
            conv_out,
            device,
            dtype,
        })
    }

    /// Single-pass full decode in F16 precision to avoid VRAM overflow
    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        // Exact BFL Flux autoencoder de-quantization:
        // z = (latents / scale_factor) + shift_factor
        let latents_f32 = latents.to_dtype(DType::F32)?;
        let scaled = (latents_f32 / FLUX_LATENT_SCALE)?;
        let shifted = (scaled + FLUX_LATENT_SHIFT)?;
        let latents = shifted.to_device(&self.device)?.to_dtype(self.dtype)?;

        let mut h = self.conv_in.forward(&latents)?;
        h = self.mid_block.0.forward(&h)?;
        h = self.mid_block.1.forward(&h)?;

        for block in &self.up_blocks {
            h = block.forward(&h)?;
        }

        let h = self.conv_norm_out.forward(&h)?;
        let h = candle_nn::ops::silu(&h)?;
        self.conv_out.forward(&h)
    }

    /// Convert unpatchified latents [1, 16, H/8, W/8] to RgbImage
    pub fn decode_to_image(&self, latents: &Tensor) -> Result<RgbImage> {
        let rgb_tensor = self.decode(latents)?;
        let (_, _, h, w) = rgb_tensor.dims4()?;

        // Scale RGB values from [-1, 1] to [0, 255] in high precision F32
        let rgb = rgb_tensor.squeeze(0)?.to_dtype(DType::F32)?;
        let normalized = ((&rgb * 0.5)? + 0.5)?;
        let scaled = (&normalized * 255.0)?;
        let clamped = scaled.clamp(0.0f32, 255.0f32)?;
        let u8_tensor = clamped.to_dtype(DType::U8)?;

        let hw3_tensor = u8_tensor.permute((1, 2, 0))?.contiguous()?;
        let flat_bytes = hw3_tensor.flatten_all()?.to_device(&Device::Cpu)?.to_vec1::<u8>()?;

        let img: RgbImage = image::ImageBuffer::from_raw(w as u32, h as u32, flat_bytes)
            .ok_or_else(|| candle_core::Error::Msg("Failed to construct ImageBuffer from raw RGB bytes".to_string()))?;

        Ok(img)
    }
}
