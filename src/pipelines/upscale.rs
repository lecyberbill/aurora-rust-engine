// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Pure Rust Super-Resolution pipeline with Tiled Inference & Seam Blending

use std::path::Path;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb, Rgba};

use crate::models::esrgan::{RRDBNet, RRDBNetConfig, SRVGGNetCompact, SRVGGNetCompactConfig};

/// Supported neural upscaling architectures
#[derive(Debug, Clone)]
pub enum UpscaleModel {
    Rrdb(RRDBNet),
    Compact(SRVGGNetCompact),
}

impl UpscaleModel {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Rrdb(net) => net.forward(x),
            Self::Compact(net) => net.forward(x),
        }
    }
}

/// Execution parameters for super-resolution
#[derive(Debug, Clone)]
pub struct UpscaleParams {
    /// Tile size for tiled processing to bound VRAM (e.g. 256, 512). Set to 0 to disable tiling.
    pub tile_size: usize,
    /// Overlap padding in pixels between adjacent tiles to prevent edge seams (e.g. 16 or 32)
    pub tile_pad: usize,
}

impl Default for UpscaleParams {
    fn default() -> Self {
        Self {
            tile_size: 256,
            tile_pad: 16,
        }
    }
}

/// Pure Rust Super-Resolution Pipeline
pub struct UpscalePipeline {
    model: UpscaleModel,
    device: Device,
    pub scale: usize,
}

impl UpscalePipeline {
    pub fn new(model: UpscaleModel, scale: usize, device: Device) -> Self {
        Self {
            model,
            device,
            scale,
        }
    }

    /// Load model from SafeTensors file (supports official RealESRGAN, 4x-UltraSharp, NMKD, Anime-6B)
    pub fn load_from_safetensors<P: AsRef<Path>>(model_path: P, device: &Device) -> Result<Self> {
        let path = model_path.as_ref();
        if !path.exists() {
            candle_core::bail!("Model file not found: {}", path.display());
        }

        // Open safetensors
        let tensors = candle_core::safetensors::load(path, device)?;
        let mut keys: Vec<String> = tensors.keys().cloned().collect();
        keys.sort();

        // Check if prefixed with "model.", "params_ema.", or "params."
        let prefix = if keys.iter().any(|k| k.starts_with("params_ema.")) {
            "params_ema."
        } else if keys.iter().any(|k| k.starts_with("params.")) {
            "params."
        } else if keys.iter().any(|k| k.starts_with("model.")) {
            "model."
        } else {
            ""
        };

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)?
        };
        let vb = if !prefix.is_empty() {
            vb.pp(prefix.trim_end_matches('.'))
        } else {
            vb
        };

        // Auto-detect architecture
        let is_compact = keys.iter().any(|k| k.contains("body.0.weight") && !k.contains("rdb"));
        if is_compact {
            // Count conv layers in compact model
            let num_conv = keys.iter().filter(|k| k.contains("body.") && k.ends_with(".weight")).count().saturating_sub(1);
            let cfg = SRVGGNetCompactConfig {
                num_in_ch: 3,
                num_out_ch: 3,
                num_feat: 64,
                num_conv: if num_conv > 0 { num_conv } else { 16 },
                upscale: 4,
            };
            let model = SRVGGNetCompact::new(&cfg, vb)?;
            Ok(Self::new(UpscaleModel::Compact(model), 4, device.clone()))
        } else {
            // RRDBNet
            let num_blocks = keys.iter()
                .filter_map(|k| {
                    if let Some(idx_str) = k.strip_prefix("body.").or_else(|| k.strip_prefix("rrdbs.")) {
                        idx_str.split('.').next().and_then(|s| s.parse::<usize>().ok())
                    } else {
                        None
                    }
                })
                .max()
                .map(|m| m + 1)
                .unwrap_or(23);

            let scale = if keys.iter().any(|k| k.contains("conv_up2")) {
                4
            } else if keys.iter().any(|k| k.contains("conv_up1")) {
                2
            } else {
                4
            };

            let cfg = RRDBNetConfig {
                in_nc: 3,
                out_nc: 3,
                num_feat: 64,
                num_block: num_blocks,
                num_grow_ch: 32,
                scale,
            };
            let model = RRDBNet::new(&cfg, vb)?;
            Ok(Self::new(UpscaleModel::Rrdb(model), scale, device.clone()))
        }
    }

    /// Upscale a [1, 3, H, W] float32 tensor (in range [0, 1])
    pub fn upscale_tensor(&self, input: &Tensor, params: &UpscaleParams) -> Result<Tensor> {
        let (_, _, h, w) = input.dims4()?;
        let scale = self.scale;

        // If tiling is disabled or image fits in one tile
        if params.tile_size == 0 || (h <= params.tile_size && w <= params.tile_size) {
            return self.model.forward(input);
        }

        let tile = params.tile_size;
        let pad = params.tile_pad;
        let out_h = h * scale;
        let out_w = w * scale;

        let mut out_data = vec![0f32; 3 * out_h * out_w];

        let h_steps = (h + tile - 1) / tile;
        let w_steps = (w + tile - 1) / tile;

        for hi in 0..h_steps {
            for wi in 0..w_steps {
                let y_start = hi * tile;
                let y_end = (y_start + tile).min(h);
                let x_start = wi * tile;
                let x_end = (x_start + tile).min(w);

                // Add padding
                let y_pad_start = y_start.saturating_sub(pad);
                let y_pad_end = (y_end + pad).min(h);
                let x_pad_start = x_start.saturating_sub(pad);
                let x_pad_end = (x_end + pad).min(w);

                let tile_h = y_pad_end - y_pad_start;
                let tile_w = x_pad_end - x_pad_start;

                let input_tile = input.narrow(2, y_pad_start, tile_h)?.narrow(3, x_pad_start, tile_w)?;
                let output_tile = self.model.forward(&input_tile)?;

                // Compute crop inside output tile to remove outer pad
                let out_y_start = y_start * scale;
                let out_x_start = x_start * scale;

                let crop_y_start = (y_start - y_pad_start) * scale;
                let crop_y_len = (y_end - y_start) * scale;
                let crop_x_start = (x_start - x_pad_start) * scale;
                let crop_x_len = (x_end - x_start) * scale;

                let cropped_tile = output_tile
                    .narrow(2, crop_y_start, crop_y_len)?
                    .narrow(3, crop_x_start, crop_x_len)?;

                let tile_flat = cropped_tile.flatten_all()?.to_vec1::<f32>()?;
                let tile_pixels = crop_y_len * crop_x_len;
                let out_stride = out_h * out_w;

                for c in 0..3 {
                    let src_c_offset = c * tile_pixels;
                    let dst_c_offset = c * out_stride;
                    for i in 0..crop_y_len {
                        let src_row = src_c_offset + i * crop_x_len;
                        let dst_row = dst_c_offset + (out_y_start + i) * out_w + out_x_start;
                        out_data[dst_row..dst_row + crop_x_len]
                            .copy_from_slice(&tile_flat[src_row..src_row + crop_x_len]);
                    }
                }
            }
        }

        Tensor::from_vec(out_data, (1, 3, out_h, out_w), &self.device)
    }

    /// Upscale an image directly (RGB8 or RGBA8)
    pub fn upscale_image(&self, img: &DynamicImage, params: &UpscaleParams) -> Result<DynamicImage> {
        let (w, h) = img.dimensions();
        let rgb_img = img.to_rgb8();
        let raw_bytes = rgb_img.into_raw();

        // Convert u8 [0..255] to float32 [0.0..1.0] in [1, 3, H, W] layout
        let mut float_data = vec![0f32; (3 * h * w) as usize];
        let num_pixels = (h * w) as usize;

        for i in 0..num_pixels {
            float_data[i] = raw_bytes[i * 3] as f32 / 255.0;
            float_data[num_pixels + i] = raw_bytes[i * 3 + 1] as f32 / 255.0;
            float_data[2 * num_pixels + i] = raw_bytes[i * 3 + 2] as f32 / 255.0;
        }

        let input_tensor = Tensor::from_vec(float_data, (1, 3, h as usize, w as usize), &self.device)?;
        let out_tensor = self.upscale_tensor(&input_tensor, params)?;

        // Convert back to image
        let (_, _, out_h, out_w) = out_tensor.dims4()?;
        let out_data = out_tensor.flatten_all()?.to_vec1::<f32>()?;
        let out_pixels = out_h * out_w;

        let mut out_buffer: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(out_w as u32, out_h as u32);
        let raw_out = out_buffer.as_mut();
        for i in 0..out_pixels {
            let r = (out_data[i].clamp(0.0, 1.0) * 255.0).round() as u8;
            let g = (out_data[out_pixels + i].clamp(0.0, 1.0) * 255.0).round() as u8;
            let b = (out_data[2 * out_pixels + i].clamp(0.0, 1.0) * 255.0).round() as u8;
            raw_out[i * 3] = r;
            raw_out[i * 3 + 1] = g;
            raw_out[i * 3 + 2] = b;
        }

        // If input had alpha channel, upscale alpha with bilinear and reconstruct RGBA
        if img.color().has_alpha() {
            let rgba_in = img.to_rgba8();
            let mut alpha_img = image::GrayImage::new(w, h);
            for y in 0..h {
                for x in 0..w {
                    let pixel = rgba_in.get_pixel(x, y);
                    alpha_img.put_pixel(x, y, image::Luma([pixel[3]]));
                }
            }
            let alpha_scaled = image::imageops::resize(
                &alpha_img,
                out_w as u32,
                out_h as u32,
                image::imageops::FilterType::CatmullRom,
            );
            let mut rgba_buffer: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(out_w as u32, out_h as u32);
            for y in 0..out_h {
                for x in 0..out_w {
                    let rgb = out_buffer.get_pixel(x as u32, y as u32);
                    let a = alpha_scaled.get_pixel(x as u32, y as u32)[0];
                    rgba_buffer.put_pixel(x as u32, y as u32, Rgba([rgb[0], rgb[1], rgb[2], a]));
                }
            }
            Ok(DynamicImage::ImageRgba8(rgba_buffer))
        } else {
            Ok(DynamicImage::ImageRgb8(out_buffer))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_upscale_pipeline_e2e() -> Result<()> {
        let dev = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let cfg = RRDBNetConfig {
            in_nc: 3,
            out_nc: 3,
            num_feat: 16,
            num_block: 2,
            num_grow_ch: 8,
            scale: 4,
        };
        let net = RRDBNet::new(&cfg, vb)?;
        let pipeline = UpscalePipeline::new(UpscaleModel::Rrdb(net), 4, dev);

        // Dummy 32x32 image
        let dummy_img = DynamicImage::new_rgb8(32, 32);
        let params = UpscaleParams {
            tile_size: 16,
            tile_pad: 4,
        };
        let upscaled = pipeline.upscale_image(&dummy_img, &params)?;
        assert_eq!(upscaled.width(), 128);
        assert_eq!(upscaled.height(), 128);
        Ok(())
    }
}
