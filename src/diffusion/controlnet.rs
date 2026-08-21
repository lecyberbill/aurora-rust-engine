// [WFGY] Zone: SAFE | λ: 0.30 | Fallbacks: 0 | Action: Pure Rust Multi-ControlNet Architecture for SDXL

use candle_core::{DType, Result, Tensor};
use candle_nn::{conv2d, Conv2d, Conv2dConfig, Module, VarBuilder};
use crate::diffusion::attention::SpatialTransformer;
use crate::diffusion::unet_2d::{get_timestep_embedding, Downsample2D, ResnetBlock2D, TimestepEmbedding};
use std::path::Path;

/// ControlNet 6-layer convolutional stem downsampling conditioning image [B, 3, H, W] to latent space [B, 320, H/8, W/8]
#[derive(Debug)]
pub struct ControlNetConditioningEmbedding {
    conv_in: Conv2d,
    blocks: Vec<Conv2d>,
    conv_out: Conv2d,
}

impl ControlNetConditioningEmbedding {
    pub fn new(vb: VarBuilder, conditioning_channels: usize, block_out_channels: &[usize]) -> Result<Self> {
        let conv_in = conv2d(
            conditioning_channels,
            block_out_channels[0],
            3,
            Conv2dConfig { padding: 1, stride: 1, ..Default::default() },
            vb.pp("conv_in"),
        )?;

        let mut blocks = Vec::new();
        let mut in_ch = block_out_channels[0];

        // 5 intermediate convolution blocks with stride 1 or 2
        for (i, &out_ch) in block_out_channels[1..].iter().enumerate() {
            let stride = if i % 2 == 1 { 2 } else { 1 };
            let block = conv2d(
                in_ch,
                out_ch,
                3,
                Conv2dConfig { padding: 1, stride, ..Default::default() },
                vb.pp(&format!("blocks.{}", i)),
            )?;
            blocks.push(block);
            in_ch = out_ch;
        }

        let conv_out = conv2d(
            in_ch,
            320,
            3,
            Conv2dConfig { padding: 1, stride: 1, ..Default::default() },
            vb.pp("conv_out"),
        )?;

        Ok(Self {
            conv_in,
            blocks,
            conv_out,
        })
    }

    pub fn forward(&self, conditioning: &Tensor) -> Result<Tensor> {
        let mut h = self.conv_in.forward(conditioning)?;
        h = candle_nn::ops::silu(&h)?;

        for block in &self.blocks {
            h = block.forward(&h)?;
            h = candle_nn::ops::silu(&h)?;
        }

        self.conv_out.forward(&h)
    }
}

/// ControlNet Model for SDXL with Zero-Convolution Skip Injections
#[derive(Debug)]
pub struct ControlNetModel {
    conv_in: Conv2d,
    controlnet_cond_embedding: ControlNetConditioningEmbedding,
    time_embedding: TimestepEmbedding,
    add_embedding: Option<TimestepEmbedding>,
    down_resnets: Vec<Vec<ResnetBlock2D>>,
    down_attns: Vec<Vec<Option<SpatialTransformer>>>,
    down_samplers: Vec<Option<Downsample2D>>,
    mid_resnet1: ResnetBlock2D,
    mid_attn: SpatialTransformer,
    mid_resnet2: ResnetBlock2D,
    controlnet_down_blocks: Vec<Conv2d>,
    controlnet_mid_block: Conv2d,
    time_proj_dim: usize,
    pub is_sdxl: bool,
}

impl ControlNetModel {
    pub fn from_safetensors<P: AsRef<Path>>(
        path: P,
        device: &candle_core::Device,
        dtype: DType,
    ) -> Result<Self> {
        let tensors = candle_core::safetensors::load(path, device)?;
        let vb = VarBuilder::from_tensors(tensors, dtype, device);
        Self::new(vb)
    }

    pub fn new(vb: VarBuilder) -> Result<Self> {
        let is_sdxl = vb.contains_tensor("add_embedding.linear_1.weight");
        let in_channels = 4;
        let block_out_channels = if is_sdxl { vec![320, 640, 1280] } else { vec![320, 640, 1280, 1280] };
        let layers_per_block = 2;
        let time_proj_dim = 320;
        let time_embed_dim = 1280;

        let conv_in = conv2d(
            in_channels,
            block_out_channels[0],
            3,
            Conv2dConfig { padding: 1, ..Default::default() },
            vb.pp("conv_in"),
        )?;

        let cond_channels = [16, 16, 32, 32, 96, 256];
        let controlnet_cond_embedding = ControlNetConditioningEmbedding::new(
            vb.pp("controlnet_cond_embedding"),
            3,
            &cond_channels,
        )?;

        let time_embedding = TimestepEmbedding::new(vb.pp("time_embedding"), time_proj_dim, time_embed_dim)?;
        let add_embedding = if is_sdxl {
            Some(TimestepEmbedding::new(vb.pp("add_embedding"), 2816, time_embed_dim)?)
        } else {
            None
        };

        // Down Blocks
        let mut down_resnets = Vec::new();
        let mut down_attns = Vec::new();
        let mut down_samplers = Vec::new();
        let mut current_ch = block_out_channels[0];

        let num_down_blocks = block_out_channels.len();
        let num_heads_map = if is_sdxl { vec![Some(5), Some(10), Some(20)] } else { vec![Some(8), Some(8), Some(8), None] };
        let tf_depths = if is_sdxl { vec![1, 2, 10] } else { vec![1, 1, 1] };

        for (i, &out_ch) in block_out_channels.iter().enumerate() {
            let mut block_resnets = Vec::new();
            let mut block_attns = Vec::new();
            let block_vb = vb.pp(&format!("down_blocks.{}", i));

            for j in 0..layers_per_block {
                let resnet_vb = block_vb.pp(&format!("resnets.{}", j));
                let resnet = ResnetBlock2D::new(resnet_vb, current_ch, out_ch, Some(time_embed_dim))?;
                block_resnets.push(resnet);
                current_ch = out_ch;

                if let Some(heads) = num_heads_map[i] {
                    let attn_vb = block_vb.pp(&format!("attentions.{}", j));
                    let d_head = current_ch / heads;
                    let depth = if is_sdxl { tf_depths[i] } else { 1 };
                    let attn = SpatialTransformer::new(attn_vb, current_ch, heads, d_head, depth, Some(2048), is_sdxl)?;
                    block_attns.push(Some(attn));
                } else {
                    block_attns.push(None);
                }
            }

            down_resnets.push(block_resnets);
            down_attns.push(block_attns);

            if i < num_down_blocks - 1 {
                let downsampler = Downsample2D::new(block_vb.pp("downsamplers.0"), current_ch)?;
                down_samplers.push(Some(downsampler));
            } else {
                down_samplers.push(None);
            }
        }

        // Mid Block
        let mid_vb = vb.pp("mid_block");
        let mid_resnet1 = ResnetBlock2D::new(mid_vb.pp("resnets.0"), current_ch, current_ch, Some(time_embed_dim))?;
        let mid_heads = if is_sdxl { 20 } else { 8 };
        let mid_d_head = current_ch / mid_heads;
        let mid_depth = if is_sdxl { 10 } else { 1 };
        let mid_attn = SpatialTransformer::new(mid_vb.pp("attentions.0"), current_ch, mid_heads, mid_d_head, mid_depth, Some(2048), is_sdxl)?;
        let mid_resnet2 = ResnetBlock2D::new(mid_vb.pp("resnets.1"), current_ch, current_ch, Some(time_embed_dim))?;

        // Zero Convolutions for Down Blocks
        let mut controlnet_down_blocks = Vec::new();
        // Number of down outputs = conv_in (1) + (layers_per_block + downsampler) per block
        // For SDXL: block 0 (2 resnets + 1 down = 3), block 1 (2 resnets + 1 down = 3), block 2 (2 resnets = 2) -> Total 1 + 3 + 3 + 2 = 9
        let num_down_zero_convs = if is_sdxl { 9 } else { 12 };
        let mut zero_ch_list = Vec::new();
        zero_ch_list.push(320); // for conv_in

        for (i, &out_ch) in block_out_channels.iter().enumerate() {
            for _ in 0..layers_per_block {
                zero_ch_list.push(out_ch);
            }
            if i < num_down_blocks - 1 {
                zero_ch_list.push(out_ch);
            }
        }

        for (k, &ch) in zero_ch_list.iter().take(num_down_zero_convs).enumerate() {
            let zero_vb = vb.pp(&format!("controlnet_down_blocks.{}", k));
            let z_conv = conv2d(ch, ch, 1, Conv2dConfig::default(), zero_vb)?;
            controlnet_down_blocks.push(z_conv);
        }

        // Zero Convolution for Mid Block
        let controlnet_mid_block = conv2d(
            current_ch,
            current_ch,
            1,
            Conv2dConfig::default(),
            vb.pp("controlnet_mid_block"),
        )?;

        Ok(Self {
            conv_in,
            controlnet_cond_embedding,
            time_embedding,
            add_embedding,
            down_resnets,
            down_attns,
            down_samplers,
            mid_resnet1,
            mid_attn,
            mid_resnet2,
            controlnet_down_blocks,
            controlnet_mid_block,
            time_proj_dim,
            is_sdxl,
        })
    }

    /// Pre-compute SDXL Add Embedding (time_ids + pooled text projection)
    pub fn compute_add_embedding(
        &self,
        b_size: usize,
        h: usize,
        w: usize,
        pooled_embeds: Option<&Tensor>,
        dev: &candle_core::Device,
        dtype: DType,
    ) -> Result<Option<Tensor>> {
        if let Some(ref add_emb) = self.add_embedding {
            let orig_h = (h * 8) as f32;
            let orig_w = (w * 8) as f32;
            let ids = [orig_h, orig_w, 0.0f32, 0.0f32, orig_h, orig_w];
            let mut time_embs = Vec::with_capacity(6);
            for &id in &ids {
                let t = Tensor::new(&[id; 1], dev)?.broadcast_as(b_size)?;
                let emb = get_timestep_embedding(&t, 256)?.to_dtype(dtype)?;
                time_embs.push(emb);
            }
            let time_embs_ref: Vec<&Tensor> = time_embs.iter().collect();
            let time_ids_emb = Tensor::cat(&time_embs_ref, 1)?;

            let default_pooled = Tensor::zeros((b_size, 1280), dtype, dev)?;
            let pooled = pooled_embeds.unwrap_or(&default_pooled);
            let add_vec = Tensor::cat(&[pooled, &time_ids_emb], 1)?;
            let add_proj = add_emb.forward(&add_vec)?;
            Ok(Some(add_proj))
        } else {
            Ok(None)
        }
    }

    /// Forward pass through ControlNet
    /// Returns: (down_block_res_samples, mid_block_res_sample)
    pub fn forward(
        &self,
        sample: &Tensor,
        timestep: f64,
        encoder_hidden_states: &Tensor,
        precomputed_add_proj: Option<&Tensor>,
        controlnet_cond: &Tensor,
        conditioning_scale: f64,
    ) -> Result<(Vec<Tensor>, Tensor)> {
        let dev = sample.device();
        let dtype = sample.dtype();
        let b_size = sample.dim(0)?;

        // 1. Timestep embedding
        let t_tensor = Tensor::new(&[timestep as f32; 1], dev)?.broadcast_as(b_size)?;
        let t_emb = get_timestep_embedding(&t_tensor, self.time_proj_dim)?.to_dtype(dtype)?;
        let mut temb = self.time_embedding.forward(&t_emb)?;

        if let Some(add_proj) = precomputed_add_proj {
            temb = (&temb + add_proj)?;
        }

        // 2. Conditioning embedding
        let cond_feature = self.controlnet_cond_embedding.forward(controlnet_cond)?;

        // 3. Conv In + conditioning addition
        let mut h = (self.conv_in.forward(sample)? + cond_feature)?;
        let mut raw_down_res = Vec::new();
        raw_down_res.push(h.clone());

        // 4. Down Blocks
        for (i, resnets) in self.down_resnets.iter().enumerate() {
            for (j, resnet) in resnets.iter().enumerate() {
                h = resnet.forward(&h, Some(&temb))?;
                if let Some(ref attn) = self.down_attns[i][j] {
                    h = attn.forward(&h, Some(encoder_hidden_states))?;
                }
                raw_down_res.push(h.clone());
            }

            if let Some(ref downsampler) = self.down_samplers[i] {
                h = downsampler.forward(&h)?;
                raw_down_res.push(h.clone());
            }
        }

        // 5. Mid Block
        h = self.mid_resnet1.forward(&h, Some(&temb))?;
        h = self.mid_attn.forward(&h, Some(encoder_hidden_states))?;
        h = self.mid_resnet2.forward(&h, Some(&temb))?;

        // 6. Zero Convolutions application with conditioning scale
        let mut down_block_res_samples = Vec::with_capacity(raw_down_res.len());
        for (raw_res, zero_conv) in raw_down_res.iter().zip(self.controlnet_down_blocks.iter()) {
            let zero_out = zero_conv.forward(raw_res)?;
            let scaled_res = if (conditioning_scale - 1.0).abs() < 1e-4 {
                zero_out
            } else {
                (&zero_out * conditioning_scale)?
            };
            down_block_res_samples.push(scaled_res);
        }

        let mid_zero_out = self.controlnet_mid_block.forward(&h)?;
        let mid_block_res_sample = if (conditioning_scale - 1.0).abs() < 1e-4 {
            mid_zero_out
        } else {
            (&mid_zero_out * conditioning_scale)?
        };

        Ok((down_block_res_samples, mid_block_res_sample))
    }
}

/// MultiControlNet manages multiple simultaneous ControlNet conditioners (e.g. Canny + Depth + OpenPose)
#[derive(Debug)]
pub struct MultiControlNet {
    pub controlnets: Vec<(ControlNetModel, f64)>,
}

impl MultiControlNet {
    pub fn new() -> Self {
        Self { controlnets: Vec::new() }
    }

    pub fn add(&mut self, model: ControlNetModel, scale: f64) {
        self.controlnets.push((model, scale));
    }

    pub fn is_empty(&self) -> bool {
        self.controlnets.is_empty()
    }

    pub fn len(&self) -> usize {
        self.controlnets.len()
    }

    pub fn forward(
        &self,
        sample: &Tensor,
        timestep: f64,
        encoder_hidden_states: &Tensor,
        precomputed_add_proj: Option<&Tensor>,
        cond_images: &[Tensor],
    ) -> Result<(Vec<Tensor>, Tensor)> {
        if self.controlnets.is_empty() {
            return Err(candle_core::Error::Msg("MultiControlNet is empty".into()));
        }

        let mut total_down_res = Vec::new();
        let mut total_mid_res: Option<Tensor> = None;

        for (idx, (cnet, scale)) in self.controlnets.iter().enumerate() {
            let cond_img = cond_images.get(idx).ok_or_else(|| {
                candle_core::Error::Msg(format!("Missing conditioning image for ControlNet {}", idx))
            })?;

            let (down_res, mid_res) = cnet.forward(
                sample,
                timestep,
                encoder_hidden_states,
                precomputed_add_proj,
                cond_img,
                *scale,
            )?;

            if total_down_res.is_empty() {
                total_down_res = down_res;
                total_mid_res = Some(mid_res);
            } else {
                for (t_down, d) in total_down_res.iter_mut().zip(down_res.iter()) {
                    *t_down = (&*t_down + d)?;
                }
                if let Some(ref mut t_mid) = total_mid_res {
                    *t_mid = (&*t_mid + &mid_res)?;
                }
            }
        }

        let mid = total_mid_res.ok_or_else(|| candle_core::Error::Msg("No mid block residual".into()))?;
        Ok((total_down_res, mid))
    }
}

/// Pure Rust Canny edge detector for spatial ControlNet conditioning
pub fn compute_canny_edge_map(
    img: &image::RgbImage,
    low_threshold: f32,
    high_threshold: f32,
) -> image::RgbImage {
    let (w, h) = img.dimensions();
    let gray = image::imageops::grayscale(img);
    let blurred = image::imageops::blur(&gray, 1.0);

    // Sobel gradients
    let mut grad_x = vec![0.0f32; (w * h) as usize];
    let mut grad_y = vec![0.0f32; (w * h) as usize];
    let mut magnitude = vec![0.0f32; (w * h) as usize];

    for y in 1..(h - 1) {
        for x in 1..(w - 1) {
            let idx = (y * w + x) as usize;
            let p00 = blurred.get_pixel(x - 1, y - 1)[0] as f32;
            let p02 = blurred.get_pixel(x + 1, y - 1)[0] as f32;
            let p10 = blurred.get_pixel(x - 1, y)[0] as f32;
            let p12 = blurred.get_pixel(x + 1, y)[0] as f32;
            let p20 = blurred.get_pixel(x - 1, y + 1)[0] as f32;
            let p22 = blurred.get_pixel(x + 1, y + 1)[0] as f32;

            let gx = (p02 + 2.0 * p12 + p22) - (p00 + 2.0 * p10 + p20);
            let gy = (p20 + 2.0 * blurred.get_pixel(x, y + 1)[0] as f32 + p22) - (p00 + 2.0 * blurred.get_pixel(x, y - 1)[0] as f32 + p02);

            grad_x[idx] = gx;
            grad_y[idx] = gy;
            magnitude[idx] = (gx * gx + gy * gy).sqrt();
        }
    }

    let mut out_img = image::RgbImage::new(w, h);
    for y in 1..(h - 1) {
        for x in 1..(w - 1) {
            let idx = (y * w + x) as usize;
            let mag = magnitude[idx];
            let val = if mag >= high_threshold {
                255u8
            } else if mag >= low_threshold {
                128u8
            } else {
                0u8
            };
            out_img.put_pixel(x, y, image::Rgb([val, val, val]));
        }
    }

    out_img
}
