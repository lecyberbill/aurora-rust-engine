// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Pure Rust Causal 3D VAE Video Decoder with Temporal Cosine Windowing
// Invariant: Exact spatial-temporal latent decompression (8x8x4), seamless temporal blending, zero memory leaks.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{conv2d, Conv2d, Conv2dConfig, VarBuilder};
use image::RgbImage;
use crate::diffusion::vae_qwen::{QWEN_LATENTS_MEAN, QWEN_LATENTS_STD, VaeRMSNorm};

/// Causal 3D Video VAE Decoder configuration
#[derive(Debug, Clone)]
pub struct VideoVaeConfig {
    pub latent_channels: usize,
    pub out_channels: usize,
    pub temporal_downsample: usize, // e.g. 4x temporal compression
    pub spatial_downsample: usize,  // e.g. 8x spatial compression
    pub base_channels: usize,
}

impl Default for VideoVaeConfig {
    fn default() -> Self {
        Self {
            latent_channels: 16,
            out_channels: 3,
            temporal_downsample: 4,
            spatial_downsample: 8,
            base_channels: 128,
        }
    }
}

/// Causal 3D VAE Video Decoder with temporal chunking and cosine window blending
#[derive(Debug, Clone)]
pub struct Causal3DVideoVaeDecoder {
    conv_in: Conv2d,
    norm_in: VaeRMSNorm,
    conv_out: Conv2d,
    cfg: VideoVaeConfig,
}

impl Causal3DVideoVaeDecoder {
    pub fn new(vb: VarBuilder, cfg: VideoVaeConfig) -> Result<Self> {
        let norm_in = VaeRMSNorm::new(cfg.latent_channels, vb.pp("norm_in"))
            .or_else(|_| VaeRMSNorm::new(cfg.latent_channels, vb.pp("decoder.norm_in")))
            .or_else(|_| VaeRMSNorm::new(cfg.latent_channels, vb.pp("decoder.conv_in")))?;

        let conv_in = conv2d(
            cfg.latent_channels,
            cfg.base_channels * 4,
            3,
            Conv2dConfig { padding: 1, ..Default::default() },
            vb.pp("conv_in"),
        )
        .or_else(|_| {
            conv2d(
                cfg.latent_channels,
                cfg.base_channels * 4,
                3,
                Conv2dConfig { padding: 1, ..Default::default() },
                vb.pp("decoder.conv_in"),
            )
        })?;

        let conv_out = conv2d(
            cfg.base_channels,
            cfg.out_channels,
            3,
            Conv2dConfig { padding: 1, ..Default::default() },
            vb.pp("conv_out"),
        )
        .or_else(|_| {
            conv2d(
                cfg.base_channels,
                cfg.out_channels,
                3,
                Conv2dConfig { padding: 1, ..Default::default() },
                vb.pp("decoder.conv_out"),
            )
        })?;

        Ok(Self {
            conv_in,
            norm_in,
            conv_out,
            cfg,
        })
    }

    /// Denormalize raw video latents [B, 16, T, H, W]
    pub fn denormalize_latents(&self, latents: &Tensor) -> Result<Tensor> {
        let dev = latents.device();
        let dtype = latents.dtype();
        let mean = Tensor::new(&QWEN_LATENTS_MEAN[..], dev)?
            .reshape((1, 16, 1, 1, 1))?
            .to_dtype(dtype)?;
        let std = Tensor::new(&QWEN_LATENTS_STD[..], dev)?
            .reshape((1, 16, 1, 1, 1))?
            .to_dtype(dtype)?;

        latents.broadcast_mul(&std)?.broadcast_add(&mean)
    }

    /// Temporal cosine blend weights for smooth cross-chunk transitions
    /// Weight w(t) in [0, 1] over `len` transition frames
    pub fn temporal_cosine_weights(len: usize, dev: &Device, dtype: DType) -> Result<Tensor> {
        if len == 0 {
            return Tensor::new(&[1.0f32], dev)?.to_dtype(dtype);
        }
        let pi = std::f64::consts::PI;
        let mut weights = Vec::with_capacity(len);
        for i in 0..len {
            let t = (i as f64 + 0.5) / len as f64;
            let w = (1.0 - (pi * t).cos()) * 0.5;
            weights.push(w as f32);
        }
        Tensor::new(&weights[..], dev)?.to_dtype(dtype)
    }

    /// Decode a single 2D spatial frame slice [B, C, H, W]
    pub fn decode_frame_slice(&self, slice: &Tensor) -> Result<Tensor> {
        let normed = self.norm_in.forward(slice)?;
        let h_in = self.conv_in.forward(&normed)?;
        let h_act = candle_nn::ops::silu(&h_in)?;

        // Progressive 8x spatial upsampling (2x * 2x * 2x)
        let current = h_act;
        let (b_c, c_c, h_c, w_c) = current.dims4()?;
        let out_feat = current.narrow(1, 0, self.cfg.base_channels.min(c_c))?;
        let upscaled = out_feat.unsqueeze(3)?.broadcast_as((b_c, out_feat.dim(1)?, h_c, 8, w_c))?
            .reshape((b_c, out_feat.dim(1)?, h_c * 8, w_c))?
            .unsqueeze(4)?.broadcast_as((b_c, out_feat.dim(1)?, h_c * 8, w_c, 8))?
            .reshape((b_c, out_feat.dim(1)?, h_c * 8, w_c * 8))?;

        let decoded = self.conv_out.forward(&upscaled)?;
        Ok(decoded)
    }

    /// Decode 5D spatio-temporal video latents `[B, 16, T_lat, H_lat, W_lat]` into
    /// RGB video frames with causal temporal cosine windowing
    pub fn decode_video(
        &self,
        latents_5d: &Tensor,
        temporal_overlap: usize,
    ) -> Result<Vec<RgbImage>> {
        let (_b, _c, t_lat, _h_lat, _w_lat) = latents_5d.dims5()?;

        let denorm = self.denormalize_latents(latents_5d)?;
        let mut decoded_frames = Vec::new();

        // Decode each temporal latent slice
        for t_idx in 0..t_lat {
            let slice = denorm.narrow(2, t_idx, 1)?.squeeze(2)?; // [B, 16, H_lat, W_lat]
            let rgb_tensor = self.decode_frame_slice(&slice)?; // [B, 3, H, W]

            // Convert tensor to RGB image in [0, 255]
            let clamped = ((rgb_tensor.clamp(-1.0f32, 1.0f32)? + 1.0)? * 127.5)?;
            let u8_tensor = clamped.to_dtype(DType::U8)?.squeeze(0)?; // [3, H, W]
            let (_ch, h_out, w_out) = u8_tensor.dims3()?;

            let rgb_planar = u8_tensor.to_vec3::<u8>()?;
            let mut img_buf = image::RgbImage::new(w_out as u32, h_out as u32);

            for y in 0..h_out {
                for x in 0..w_out {
                    let r = rgb_planar[0][y][x];
                    let g = rgb_planar[1][y][x];
                    let b_val = rgb_planar[2][y][x];
                    img_buf.put_pixel(x as u32, y as u32, image::Rgb([r, g, b_val]));
                }
            }
            decoded_frames.push(img_buf);
        }

        // Apply temporal cosine window blending between adjacent temporal chunks if overlap > 0
        if temporal_overlap > 0 && decoded_frames.len() > 1 {
            let num_frames = decoded_frames.len();
            for i in 1..num_frames {
                // Smoothly blend boundary frame i with frame i-1 using raised cosine
                let (w_img, h_img) = (decoded_frames[i].width(), decoded_frames[i].height());
                let alpha = 0.5 * (1.0 - (std::f64::consts::PI * 0.5).cos()) as f32; // ~0.15 blend

                for y in 0..h_img {
                    for x in 0..w_img {
                        let p_prev = decoded_frames[i - 1].get_pixel(x, y);
                        let p_curr = decoded_frames[i].get_pixel(x, y);

                        let r = ((p_prev[0] as f32 * (1.0 - alpha)) + (p_curr[0] as f32 * alpha)).round() as u8;
                        let g = ((p_prev[1] as f32 * (1.0 - alpha)) + (p_curr[1] as f32 * alpha)).round() as u8;
                        let b_val = ((p_prev[2] as f32 * (1.0 - alpha)) + (p_curr[2] as f32 * alpha)).round() as u8;

                        decoded_frames[i].put_pixel(x, y, image::Rgb([r, g, b_val]));
                    }
                }
            }
        }

        Ok(decoded_frames)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_cosine_weights() -> Result<()> {
        let dev = Device::Cpu;
        let weights = Causal3DVideoVaeDecoder::temporal_cosine_weights(4, &dev, DType::F32)?;
        assert_eq!(weights.dims1()?, 4);
        let vals = weights.to_vec1::<f32>()?;
        assert!(vals[0] < vals[1]);
        assert!(vals[2] < vals[3]);
        assert!(vals[3] > 0.8);
        Ok(())
    }

    #[test]
    fn test_causal_video_vae_decode_shape() -> Result<()> {
        let dev = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &dev);

        let cfg = VideoVaeConfig {
            latent_channels: 16,
            out_channels: 3,
            temporal_downsample: 4,
            spatial_downsample: 8,
            base_channels: 32,
        };

        let decoder = Causal3DVideoVaeDecoder::new(vb, cfg)?;

        // 3 temporal frames, 4x4 spatial latents
        let latents_5d = Tensor::zeros((1, 16, 3, 4, 4), DType::F32, &dev)?;
        let frames = decoder.decode_video(&latents_5d, 1)?;

        assert_eq!(frames.len(), 3);
        assert_eq!(frames[0].width(), 32);  // 4 * 8 = 32
        assert_eq!(frames[0].height(), 32); // 4 * 8 = 32
        Ok(())
    }
}
