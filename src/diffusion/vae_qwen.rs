// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Pure Rust AutoencoderKLQwenImage 3D Causal VAE Decoder
// Exact mathematical 2D slice execution of Wan/QwenImage Causal VAE

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{conv2d, Conv2d, Conv2dConfig, VarBuilder};
use image::{ImageBuffer, RgbImage};

pub const QWEN_LATENTS_MEAN: [f32; 16] = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
];

pub const QWEN_LATENTS_STD: [f32; 16] = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
];

/// RMS Normalization for Qwen Image VAE
#[derive(Debug, Clone)]
pub struct VaeRMSNorm {
    gamma: Tensor,
    scale: f64,
}

impl VaeRMSNorm {
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let gamma = vb.get((dim, 1, 1), "gamma")
            .or_else(|_| vb.get((dim, 1, 1, 1), "gamma"))?;
        let gamma_2d = gamma.reshape((1, dim, 1, 1))?;
        Ok(Self {
            gamma: gamma_2d,
            scale: (dim as f64).sqrt(),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let orig_dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        // Channel-wise RMS: norm along channel dim (dim 1)
        let norm = x_f32.sqr()?.mean_keepdim(1)?.sqrt()?;
        let eps = 1e-6f64;
        let norm_eps = (norm + eps)?;
        let normalized = x_f32.broadcast_div(&norm_eps)?;
        let scaled = (normalized * self.scale)?;
        let gamma = self.gamma.to_device(x.device())?.to_dtype(DType::F32)?;
        let out = scaled.broadcast_mul(&gamma)?;
        out.to_dtype(orig_dtype)
    }
}

/// Helper to convert 3D causal conv weights [out, in, 3, 3, 3] to 2D conv [out, in, 3, 3]
fn causal_3d_to_2d_conv(vb: VarBuilder, out_ch: usize, in_ch: usize, k: usize, pad: usize) -> Result<Conv2d> {
    let weight_3d = vb.get((out_ch, in_ch, k, k, k), "weight")?;
    // Take the last temporal causal slice (index k - 1)
    let weight_2d = weight_3d.narrow(2, k - 1, 1)?.squeeze(2)?;
    let bias = vb.get(out_ch, "bias").ok();
    
    let cfg = Conv2dConfig {
        padding: pad,
        ..Default::default()
    };
    Ok(Conv2d::new(weight_2d, bias, cfg))
}

/// 2D Residual block in Qwen Image VAE
#[derive(Debug, Clone)]
pub struct QwenVaeResidual {
    norm1: VaeRMSNorm,
    conv1: Conv2d,
    norm2: VaeRMSNorm,
    conv2: Conv2d,
    shortcut: Option<Conv2d>,
}

impl QwenVaeResidual {
    pub fn new(in_ch: usize, out_ch: usize, vb: VarBuilder) -> Result<Self> {
        let norm1 = VaeRMSNorm::new(in_ch, vb.pp("residual.0"))?;
        let conv1 = causal_3d_to_2d_conv(vb.pp("residual.2"), out_ch, in_ch, 3, 1)?;
        let norm2 = VaeRMSNorm::new(out_ch, vb.pp("residual.3"))?;
        let conv2 = causal_3d_to_2d_conv(vb.pp("residual.6"), out_ch, out_ch, 3, 1)?;

        let shortcut = if in_ch != out_ch {
            let weight_3d = vb.pp("shortcut").get((out_ch, in_ch, 1, 1, 1), "weight")?;
            let weight_2d = weight_3d.squeeze(2)?;
            let bias = vb.pp("shortcut").get(out_ch, "bias").ok();
            Some(Conv2d::new(weight_2d, bias, Conv2dConfig::default()))
        } else {
            None
        };

        Ok(Self {
            norm1,
            conv1,
            norm2,
            conv2,
            shortcut,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.norm1.forward(x)?;
        let h = candle_nn::ops::silu(&h)?;
        let h = self.conv1.forward(&h)?;
        let h = self.norm2.forward(&h)?;
        let h = candle_nn::ops::silu(&h)?;
        let h = self.conv2.forward(&h)?;

        let res = if let Some(ref sc) = self.shortcut {
            sc.forward(x)?
        } else {
            x.clone()
        };

        (h + res)
    }
}

/// Nearest 2x2 Spatial Upsampler + Conv2D
#[derive(Debug, Clone)]
pub struct QwenVaeUpsample2d {
    conv: Conv2d,
}

impl QwenVaeUpsample2d {
    pub fn new(in_ch: usize, out_ch: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get((out_ch, in_ch, 3, 3), "weight")?;
        let bias = vb.get(out_ch, "bias").ok();
        let cfg = Conv2dConfig { padding: 1, ..Default::default() };
        let conv = Conv2d::new(weight, bias, cfg);
        Ok(Self { conv })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = x.dims4()?;
        // Nearest 2x exact upsample
        let up = x.unsqueeze(4)?
            .repeat((1, 1, 1, 1, 2))?
            .reshape((b, c, h, w * 2))?
            .unsqueeze(3)?
            .repeat((1, 1, 1, 2, 1))?
            .reshape((b, c, h * 2, w * 2))?;
        self.conv.forward(&up)
    }
}

/// Complete AutoencoderKLQwenImage Decoder in pure Rust
#[derive(Debug, Clone)]
pub struct QwenImageVaeDecoder {
    conv_in: Conv2d,
    mid_res1: QwenVaeResidual,
    mid_res2: QwenVaeResidual,
    // Upsample stages: 384 -> 384 -> 192 -> 96 -> 3 RGB
    up_blocks_0: Vec<QwenVaeResidual>, // 384
    resample_1: QwenVaeUpsample2d,      // 384 -> 192
    up_blocks_1: Vec<QwenVaeResidual>, // 384
    resample_2: QwenVaeUpsample2d,      // 384 -> 192
    up_blocks_2: Vec<QwenVaeResidual>, // 192
    resample_3: QwenVaeUpsample2d,      // 192 -> 96
    up_blocks_3: Vec<QwenVaeResidual>, // 96
    norm_out: VaeRMSNorm,
    conv_out: Conv2d,
    device: Device,
    dtype: DType,
}

impl QwenImageVaeDecoder {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();

        let conv_in = causal_3d_to_2d_conv(vb.pp("decoder.conv1"), 384, 16, 3, 1)?;
        let mid_res1 = QwenVaeResidual::new(384, 384, vb.pp("decoder.middle.0"))?;
        let mid_res2 = QwenVaeResidual::new(384, 384, vb.pp("decoder.middle.2"))?;

        // Stage 0: upsamples.0, 1, 2 (384 -> 384)
        let mut up_blocks_0 = Vec::new();
        for i in 0..=2 {
            up_blocks_0.push(QwenVaeResidual::new(384, 384, vb.pp(format!("decoder.upsamples.{}", i)))?);
        }

        // Resample 1: upsamples.3.resample.1 (384 -> 192)
        let resample_1 = QwenVaeUpsample2d::new(384, 192, vb.pp("decoder.upsamples.3.resample.1"))?;

        // Stage 1: upsamples.4 (in: 192, out: 384), upsamples.5 (384->384), upsamples.6 (384->384)
        let mut up_blocks_1 = Vec::new();
        up_blocks_1.push(QwenVaeResidual::new(192, 384, vb.pp("decoder.upsamples.4"))?);
        up_blocks_1.push(QwenVaeResidual::new(384, 384, vb.pp("decoder.upsamples.5"))?);
        up_blocks_1.push(QwenVaeResidual::new(384, 384, vb.pp("decoder.upsamples.6"))?);

        // Resample 2: upsamples.7.resample.1 (384 -> 192)
        let resample_2 = QwenVaeUpsample2d::new(384, 192, vb.pp("decoder.upsamples.7.resample.1"))?;

        // Stage 2: upsamples.8, 9, 10 (192 -> 192)
        let mut up_blocks_2 = Vec::new();
        for i in 8..=10 {
            up_blocks_2.push(QwenVaeResidual::new(192, 192, vb.pp(format!("decoder.upsamples.{}", i)))?);
        }

        // Resample 3: upsamples.11.resample.1 (192 -> 96)
        let resample_3 = QwenVaeUpsample2d::new(192, 96, vb.pp("decoder.upsamples.11.resample.1"))?;

        // Stage 3: upsamples.12, 13, 14 (96 -> 96)
        let mut up_blocks_3 = Vec::new();
        for i in 12..=14 {
            up_blocks_3.push(QwenVaeResidual::new(96, 96, vb.pp(format!("decoder.upsamples.{}", i)))?);
        }

        let norm_out = VaeRMSNorm::new(96, vb.pp("decoder.head.0"))?;
        let conv_out = causal_3d_to_2d_conv(vb.pp("decoder.head.2"), 3, 96, 3, 1)?;

        Ok(Self {
            conv_in,
            mid_res1,
            mid_res2,
            up_blocks_0,
            resample_1,
            up_blocks_1,
            resample_2,
            up_blocks_2,
            resample_3,
            up_blocks_3,
            norm_out,
            conv_out,
            device,
            dtype,
        })
    }

    /// Denormalize 16-channel latents and decode to [1, 3, H*8, W*8] in range [-1.0, 1.0]
    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let latents = if latents.rank() == 3 {
            latents.unsqueeze(0)?
        } else {
            latents.clone()
        };

        let (b, c, h, w) = latents.dims4()?;
        if c != 16 {
            return Err(candle_core::Error::Msg(format!("QwenImageVaeDecoder expects 16 channels, got {}", c)));
        }

        // 1. Exact Qwen / Wan Latent Denormalization: z = z * std + mean
        let mean = Tensor::from_slice(&QWEN_LATENTS_MEAN, (1, 16, 1, 1), latents.device())?
            .to_dtype(latents.dtype())?;
        let std = Tensor::from_slice(&QWEN_LATENTS_STD, (1, 16, 1, 1), latents.device())?
            .to_dtype(latents.dtype())?;
        let z = latents.broadcast_mul(&std)?.broadcast_add(&mean)?;

        // 2. Initial convolution: [B, 16, H, W] -> [B, 384, H, W]
        let mut h_feat = self.conv_in.forward(&z)?;

        // 3. Middle residual blocks
        h_feat = self.mid_res1.forward(&h_feat)?;
        h_feat = self.mid_res2.forward(&h_feat)?;

        // 4. Stage 0 (384 -> 384) + Resample 1 (384 -> 192, 2x up)
        for block in &self.up_blocks_0 {
            h_feat = block.forward(&h_feat)?;
        }
        h_feat = self.resample_1.forward(&h_feat)?;

        // 5. Stage 1 (192 -> 384) + Resample 2 (384 -> 192, 2x up)
        for block in &self.up_blocks_1 {
            h_feat = block.forward(&h_feat)?;
        }
        h_feat = self.resample_2.forward(&h_feat)?;

        // 6. Stage 2 (192 -> 192) + Resample 3 (192 -> 96, 2x up)
        for block in &self.up_blocks_2 {
            h_feat = block.forward(&h_feat)?;
        }
        h_feat = self.resample_3.forward(&h_feat)?;

        // 7. Stage 3 (96 -> 96)
        for block in &self.up_blocks_3 {
            h_feat = block.forward(&h_feat)?;
        }

        // 8. Output Head: RMSNorm -> SiLU -> Conv2D (3 RGB channels)
        h_feat = self.norm_out.forward(&h_feat)?;
        h_feat = candle_nn::ops::silu(&h_feat)?;
        let rgb = self.conv_out.forward(&h_feat)?;

        Ok(rgb)
    }

    /// Decode latent tensor directly into an RgbImage (1024x1024)
    pub fn decode_to_image(&self, latents: &Tensor) -> Result<RgbImage> {
        let decoded = self.decode(latents)?;
        crate::diffusion::vae::tensor_to_rgb_image(&decoded)
    }
}
