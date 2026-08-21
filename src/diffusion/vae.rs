// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: High-Performance VAE Decoder with configurable Tiled and Direct decoding

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::stable_diffusion::vae::{AutoEncoderKL, AutoEncoderKLConfig};
use image::{ImageBuffer, RgbImage};

pub const SD_LATENT_SCALE: f64 = 0.18215;
pub const SDXL_LATENT_SCALE: f64 = 0.13025;

/// High-level VAE Wrapper supporting direct and configurable low-memory Tiled decoding
pub struct VaeDecoder {
    vae: AutoEncoderKL,
    scaling_factor: f64,
    device: Device,
    dtype: DType,
}

impl VaeDecoder {
    pub fn new(vb: VarBuilder, is_sdxl: bool) -> Result<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();

        let config = AutoEncoderKLConfig {
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            latent_channels: 4,
            norm_num_groups: 32,
            ..Default::default()
        };

        let vae = AutoEncoderKL::new(vb, 3, 3, config)?;
        let scaling_factor = if is_sdxl {
            SDXL_LATENT_SCALE
        } else {
            SD_LATENT_SCALE
        };

        Ok(Self {
            vae,
            scaling_factor,
            device,
            dtype,
        })
    }

    /// Decode latent tensor directly in a single pass (fastest, requires ~6GB VRAM for 1024x1024)
    pub fn decode_direct(&self, latents: &Tensor) -> Result<RgbImage> {
        let latents = if latents.rank() == 4 {
            latents.clone()
        } else {
            latents.unsqueeze(0)?
        };

        let latents_matched = latents.to_dtype(self.dtype)?;
        let scaled_latents = (&latents_matched / self.scaling_factor)?;
        let decoded = self.vae.decode(&scaled_latents)?;
        tensor_to_rgb_image(&decoded)
    }

    /// Encode image tensor [1, 3, H, W] in [-1.0, 1.0] to scaled latent space [1, 4, H/8, W/8]
    pub fn encode_direct(&self, img_tensor: &Tensor) -> Result<Tensor> {
        let img_matched = img_tensor.to_device(&self.device)?.to_dtype(self.dtype)?;
        let dist = self.vae.encode(&img_matched)?;
        let latents = (dist.sample()? * self.scaling_factor)?;
        Ok(latents)
    }

    /// Encode an RgbImage into latent space
    pub fn encode_image(&self, img: &RgbImage) -> Result<Tensor> {
        let tensor = rgb_image_to_tensor(img, &self.device, self.dtype)?;
        self.encode_direct(&tensor)
    }

    /// Tiled Latent Decoding: decodes latents in custom tiles with overlap blending to prevent VRAM spikes
    pub fn decode_tiled(&self, latents: &Tensor, tile_size: usize, overlap: usize) -> Result<RgbImage> {
        let latents = if latents.rank() == 4 {
            latents.squeeze(0)?
        } else {
            latents.clone()
        };

        let (_c, h, w) = latents.dims3()?;
        let out_h = h * 8;
        let out_w = w * 8;

        // If latents are smaller than tile_size, decode directly
        if h <= tile_size && w <= tile_size {
            return self.decode_direct(&latents.unsqueeze(0)?);
        }

        let stride = tile_size.saturating_sub(overlap).max(1);

        let mut output_floats = vec![0.0f32; 3 * out_h * out_w];
        let mut weight_mask = vec![0.0f32; out_h * out_w];

        // Generate non-redundant tile coverage slices
        let mut y_slices = Vec::new();
        let mut y = 0;
        loop {
            let start = y.min(h.saturating_sub(tile_size));
            let len = tile_size.min(h - start);
            if !y_slices.contains(&(start, len)) {
                y_slices.push((start, len));
            }
            if start + len >= h { break; }
            y += stride;
        }

        let mut x_slices = Vec::new();
        let mut x = 0;
        loop {
            let start = x.min(w.saturating_sub(tile_size));
            let len = tile_size.min(w - start);
            if !x_slices.contains(&(start, len)) {
                x_slices.push((start, len));
            }
            if start + len >= w { break; }
            x += stride;
        }

        let total_tiles = y_slices.len() * x_slices.len();
        let mut tile_idx = 0;

        for &(y_start, y_len) in &y_slices {
            for &(x_start, x_len) in &x_slices {
                tile_idx += 1;

                let latent_tile = latents
                    .narrow(1, y_start, y_len)?
                    .narrow(2, x_start, x_len)?
                    .unsqueeze(0)?
                    .to_dtype(self.dtype)?;

                let scaled_tile = (&latent_tile / self.scaling_factor)?;
                let decoded_tile = self.vae.decode(&scaled_tile)?.squeeze(0)?;

                // Transfer decoded tile [3, tile_h*8, tile_w*8] to CPU
                let tile_cpu = decoded_tile.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
                let tile_floats = tile_cpu.flatten_all()?.to_vec1::<f32>()?;
                println!("    [VAE Tile {}/{}] Decoded & blended", tile_idx, total_tiles);
                let _ = std::io::Write::flush(&mut std::io::stdout());

                let tile_pix_h = y_len * 8;
                let tile_pix_w = x_len * 8;
                let tile_plane = tile_pix_h * tile_pix_w;

                let py_start = y_start * 8;
                let px_start = x_start * 8;
                let out_plane = out_h * out_w;
                let overlap_pix = (overlap * 8).max(1) as f32;

                for ty in 0..tile_pix_h {
                    let gy = py_start + ty;
                    if gy >= out_h { continue; }

                    // Smooth cosine feathering on Y boundary
                    let w_y = if overlap > 0 && ty < overlap * 8 && y_start > 0 {
                        let t = ty as f32 / overlap_pix;
                        0.5 * (1.0 - (std::f32::consts::PI * t).cos())
                    } else if overlap > 0 && ty > tile_pix_h.saturating_sub(overlap * 8) && (y_start + y_len) < h {
                        let t = (tile_pix_h - ty) as f32 / overlap_pix;
                        0.5 * (1.0 - (std::f32::consts::PI * t).cos())
                    } else {
                        1.0f32
                    };

                    for tx in 0..tile_pix_w {
                        let gx = px_start + tx;
                        if gx >= out_w { continue; }

                        // Smooth cosine feathering on X boundary
                        let w_x = if overlap > 0 && tx < overlap * 8 && x_start > 0 {
                            let t = tx as f32 / overlap_pix;
                            0.5 * (1.0 - (std::f32::consts::PI * t).cos())
                        } else if overlap > 0 && tx > tile_pix_w.saturating_sub(overlap * 8) && (x_start + x_len) < w {
                            let t = (tile_pix_w - tx) as f32 / overlap_pix;
                            0.5 * (1.0 - (std::f32::consts::PI * t).cos())
                        } else {
                            1.0f32
                        };

                        let weight = w_y * w_x;
                        let g_idx = gy * out_w + gx;
                        let t_idx = ty * tile_pix_w + tx;

                        output_floats[g_idx] += tile_floats[t_idx] * weight;
                        output_floats[out_plane + g_idx] += tile_floats[tile_plane + t_idx] * weight;
                        output_floats[2 * out_plane + g_idx] += tile_floats[2 * tile_plane + t_idx] * weight;
                        weight_mask[g_idx] += weight;
                    }
                }
            }
        }

        // Normalize by accumulated blend weights and construct RGB Image
        let out_plane = out_h * out_w;
        let mut rgb_buffer = Vec::with_capacity(out_h * out_w * 3);

        for y in 0..out_h {
            for x in 0..out_w {
                let idx = y * out_w + x;
                let w = weight_mask[idx].max(1e-5);
                let r = ((output_floats[idx] / w * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
                let g = ((output_floats[out_plane + idx] / w * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
                let b = ((output_floats[2 * out_plane + idx] / w * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
                rgb_buffer.push(r);
                rgb_buffer.push(g);
                rgb_buffer.push(b);
            }
        }

        let img: RgbImage = ImageBuffer::from_raw(out_w as u32, out_h as u32, rgb_buffer)
            .ok_or_else(|| candle_core::Error::Msg("Failed to construct ImageBuffer from raw RGB bytes".to_string()))?;

        Ok(img)
    }

    pub fn decode_to_image(&self, latents: &Tensor) -> Result<RgbImage> {
        self.decode_tiled(latents, 64, 8)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

pub fn tensor_to_rgb_image(tensor: &Tensor) -> Result<RgbImage> {
    let tensor = if tensor.rank() == 4 {
        tensor.squeeze(0)?
    } else {
        tensor.clone()
    };

    let (c, h, w) = tensor.dims3()?;
    if c != 3 {
        return Err(candle_core::Error::Msg(format!(
            "Expected 3 channels for RGB conversion, got {}",
            c
        )));
    }

    let tensor_cpu = tensor.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let raw_floats = tensor_cpu.flatten_all()?.to_vec1::<f32>()?;
    let mut rgb_buffer = Vec::with_capacity(h * w * 3);

    let plane_size = h * w;
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let r = ((raw_floats[idx] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
            let g = ((raw_floats[plane_size + idx] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
            let b = ((raw_floats[2 * plane_size + idx] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
            rgb_buffer.push(r);
            rgb_buffer.push(g);
            rgb_buffer.push(b);
        }
    }

    let img: RgbImage = ImageBuffer::from_raw(w as u32, h as u32, rgb_buffer)
        .ok_or_else(|| candle_core::Error::Msg("Failed to construct ImageBuffer from raw RGB bytes".to_string()))?;

    Ok(img)
}

pub fn rgb_image_to_tensor(img: &RgbImage, device: &Device, dtype: DType) -> Result<Tensor> {
    let (w, h) = img.dimensions();
    let mut raw_floats = vec![0.0f32; 3 * (h as usize) * (w as usize)];
    let plane_size = (h as usize) * (w as usize);

    for y in 0..h {
        for x in 0..w {
            let pixel = img.get_pixel(x, y);
            let idx = (y as usize) * (w as usize) + (x as usize);
            raw_floats[idx] = (pixel[0] as f32 / 127.5) - 1.0;
            raw_floats[plane_size + idx] = (pixel[1] as f32 / 127.5) - 1.0;
            raw_floats[2 * plane_size + idx] = (pixel[2] as f32 / 127.5) - 1.0;
        }
    }

    Tensor::from_vec(raw_floats, (1, 3, h as usize, w as usize), &Device::Cpu)?
        .to_device(device)?
        .to_dtype(dtype)
}

pub struct FastLatentPreviewer;

impl FastLatentPreviewer {
    pub fn preview_latent(latent: &Tensor) -> Result<RgbImage> {
        let latent = if latent.rank() == 4 {
            latent.squeeze(0)?
        } else {
            latent.clone()
        };

        let (_c, h, w) = latent.dims3()?;
        let weights_data = [
            0.298f32, 0.207f32, 0.208f32, -0.040f32,
            0.224f32, 0.232f32, -0.279f32, 0.045f32,
            0.179f32, -0.341f32, 0.138f32, 0.098f32,
        ];

        let weight_tensor = Tensor::from_slice(&weights_data, (3, 4), latent.device())?
            .to_dtype(latent.dtype())?;

        let latent_flat = latent.reshape((4, h * w))?;
        let rgb_flat = weight_tensor.matmul(&latent_flat)?;
        let rgb_chw = rgb_flat.reshape((3, h, w))?;

        tensor_to_rgb_image(&rgb_chw)
    }
}
