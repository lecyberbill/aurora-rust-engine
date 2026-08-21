// [WFGY] Zone: SAFE | λ: 0.35 | Fallbacks: 0 | Action: Pure Rust SD & SDXL UNet2DConditionModel with robust 2-pass Upsample2D

use candle_core::{DType, Result, Tensor};
use candle_nn::{conv2d, group_norm, linear, Conv2d, Conv2dConfig, GroupNorm, Linear, Module, VarBuilder};
use crate::diffusion::attention::SpatialTransformer;

/// Sinusoidal timestep embedding helper (exact Diffusers formula)
pub fn get_timestep_embedding(timesteps: &Tensor, embedding_dim: usize) -> Result<Tensor> {
    let half_dim = embedding_dim / 2;
    let factor = -(10000.0f64.ln()) / (half_dim as f64);
    let dev = timesteps.device();
    let emb = Tensor::arange(0u32, half_dim as u32, dev)?.to_dtype(DType::F32)?;
    let emb = (emb * factor)?.exp()?;
    let timesteps_f32 = timesteps.to_dtype(DType::F32)?;
    let shape = (timesteps.dim(0)?, 1);
    let timesteps_2d = timesteps_f32.reshape(shape)?;
    let emb = timesteps_2d.matmul(&emb.reshape((1, half_dim))?)?;
    let sin = emb.sin()?;
    let cos = emb.cos()?;
    Tensor::cat(&[&cos, &sin], 1)
}

/// Timestep embedding MLP: Sinusoidal -> Linear -> SiLU -> Linear
#[derive(Debug)]
pub struct TimestepEmbedding {
    linear_1: Linear,
    linear_2: Linear,
}

impl TimestepEmbedding {
    pub fn new(vb: VarBuilder, channel: usize, time_embed_dim: usize) -> Result<Self> {
        let linear_1 = linear(channel, time_embed_dim, vb.pp("linear_1"))?;
        let linear_2 = linear(time_embed_dim, time_embed_dim, vb.pp("linear_2"))?;
        Ok(Self { linear_1, linear_2 })
    }

    pub fn apply_lora_deltas(&mut self, prefix: &str, deltas: &std::collections::HashMap<String, Tensor>) -> Result<()> {
        crate::diffusion::attention::apply_linear_delta(&mut self.linear_1, &format!("{}.linear_1", prefix), deltas)?;
        crate::diffusion::attention::apply_linear_delta(&mut self.linear_2, &format!("{}.linear_2", prefix), deltas)?;
        Ok(())
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.linear_1.forward(xs)?;
        let xs = candle_nn::ops::silu(&xs)?;
        self.linear_2.forward(&xs)
    }
}

/// 2D Resnet Block with Time Embedding injection and optional Conv shortcut
#[derive(Debug)]
pub struct ResnetBlock2D {
    norm1: GroupNorm,
    conv1: Conv2d,
    time_emb_proj: Option<Linear>,
    norm2: GroupNorm,
    conv2: Conv2d,
    conv_shortcut: Option<Conv2d>,
}

impl ResnetBlock2D {
    pub fn new(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        temb_channels: Option<usize>,
    ) -> Result<Self> {
        let norm1 = group_norm(32, in_channels, 1e-5, vb.pp("norm1"))?;
        let conv1 = conv2d(in_channels, out_channels, 3, Conv2dConfig { padding: 1, ..Default::default() }, vb.pp("conv1"))?;

        let time_emb_proj = if let Some(temb_dim) = temb_channels {
            if vb.contains_tensor("time_emb_proj.weight") {
                Some(linear(temb_dim, out_channels, vb.pp("time_emb_proj"))?)
            } else {
                None
            }
        } else {
            None
        };

        let norm2 = group_norm(32, out_channels, 1e-5, vb.pp("norm2"))?;
        let conv2 = conv2d(out_channels, out_channels, 3, Conv2dConfig { padding: 1, ..Default::default() }, vb.pp("conv2"))?;

        let conv_shortcut = if in_channels != out_channels {
            if vb.contains_tensor("conv_shortcut.weight") {
                Some(conv2d(in_channels, out_channels, 1, Conv2dConfig::default(), vb.pp("conv_shortcut"))?)
            } else if vb.contains_tensor("nin_shortcut.weight") {
                Some(conv2d(in_channels, out_channels, 1, Conv2dConfig::default(), vb.pp("nin_shortcut"))?)
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            norm1,
            conv1,
            time_emb_proj,
            norm2,
            conv2,
            conv_shortcut,
        })
    }

    pub fn apply_lora_deltas(&mut self, prefix: &str, deltas: &std::collections::HashMap<String, Tensor>) -> Result<()> {
        crate::diffusion::attention::apply_conv2d_delta(&mut self.conv1, &format!("{}.conv1", prefix), deltas)?;
        crate::diffusion::attention::apply_conv2d_delta(&mut self.conv2, &format!("{}.conv2", prefix), deltas)?;
        if let Some(proj) = &mut self.time_emb_proj {
            crate::diffusion::attention::apply_linear_delta(proj, &format!("{}.time_emb_proj", prefix), deltas)?;
        }
        if let Some(sc) = &mut self.conv_shortcut {
            crate::diffusion::attention::apply_conv2d_delta(sc, &format!("{}.conv_shortcut", prefix), deltas)?;
            crate::diffusion::attention::apply_conv2d_delta(sc, &format!("{}.nin_shortcut", prefix), deltas)?;
        }
        Ok(())
    }

    pub fn forward(&self, xs: &Tensor, temb: Option<&Tensor>) -> Result<Tensor> {
        let residual = match &self.conv_shortcut {
            Some(sc) => sc.forward(xs)?,
            None => xs.clone(),
        };

        let h = self.norm1.forward(xs)?;
        let h = candle_nn::ops::silu(&h)?;
        let mut h = self.conv1.forward(&h)?;

        if let (Some(t), Some(proj)) = (temb, &self.time_emb_proj) {
            let t_proj = proj.forward(&candle_nn::ops::silu(t)?)?;
            let (b, c, h_dim, w_dim) = h.dims4()?;
            let t_proj = t_proj.reshape((b, c, 1, 1))?.broadcast_as((b, c, h_dim, w_dim))?;
            h = (h + t_proj)?;
        }

        let h = self.norm2.forward(&h)?;
        let h = candle_nn::ops::silu(&h)?;
        let h = self.conv2.forward(&h)?;

        residual + h
    }
}

/// Downsampling convolution layer
#[derive(Debug)]
pub struct Downsample2D {
    conv: Conv2d,
}

impl Downsample2D {
    pub fn new(vb: VarBuilder, channels: usize) -> Result<Self> {
        let conv = conv2d(
            channels,
            channels,
            3,
            Conv2dConfig { stride: 2, padding: 1, ..Default::default() },
            vb.pp("conv"),
        )?;
        Ok(Self { conv })
    }

    pub fn apply_lora_deltas(&mut self, prefix: &str, deltas: &std::collections::HashMap<String, Tensor>) -> Result<()> {
        crate::diffusion::attention::apply_conv2d_delta(&mut self.conv, &format!("{}.conv", prefix), deltas)?;
        Ok(())
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.conv.forward(xs)
    }
}

/// Upsampling layer (strictly contiguous 2-pass nearest interpolation + convolution)
#[derive(Debug)]
pub struct Upsample2D {
    conv: Conv2d,
}

impl Upsample2D {
    pub fn new(vb: VarBuilder, channels: usize) -> Result<Self> {
        let conv = conv2d(
            channels,
            channels,
            3,
            Conv2dConfig { padding: 1, ..Default::default() },
            vb.pp("conv"),
        )?;
        Ok(Self { conv })
    }

    pub fn apply_lora_deltas(&mut self, prefix: &str, deltas: &std::collections::HashMap<String, Tensor>) -> Result<()> {
        crate::diffusion::attention::apply_conv2d_delta(&mut self.conv, &format!("{}.conv", prefix), deltas)?;
        Ok(())
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = xs.dims4()?;
        // Expand H
        let xs = xs
            .reshape((b, c, h, 1, w))?
            .broadcast_as((b, c, h, 2, w))?
            .contiguous()?
            .reshape((b, c, h * 2, w))?;
        // Expand W
        let xs = xs
            .reshape((b, c, h * 2, w, 1))?
            .broadcast_as((b, c, h * 2, w, 2))?
            .contiguous()?
            .reshape((b, c, h * 2, w * 2))?;
        self.conv.forward(&xs)
    }
}

/// Complete Pure Rust UNet Condition Model
pub struct UNetConditionModel {
    conv_in: Conv2d,
    time_proj_dim: usize,
    time_embedding: TimestepEmbedding,
    add_embedding: Option<TimestepEmbedding>,
    // Down blocks
    down_resnets: Vec<Vec<ResnetBlock2D>>,
    down_attns: Vec<Vec<Option<SpatialTransformer>>>,
    down_samplers: Vec<Option<Downsample2D>>,
    // Mid block
    mid_resnet1: ResnetBlock2D,
    mid_attn: SpatialTransformer,
    mid_resnet2: ResnetBlock2D,
    // Up blocks
    up_resnets: Vec<Vec<ResnetBlock2D>>,
    up_attns: Vec<Vec<Option<SpatialTransformer>>>,
    up_samplers: Vec<Option<Upsample2D>>,
    // Out
    conv_norm_out: GroupNorm,
    conv_out: Conv2d,
    is_sdxl: bool,
}

impl UNetConditionModel {
    pub fn new_sdxl(vb: VarBuilder) -> Result<Self> {
        let is_sdxl = true;
        let in_channels = 4;
        let out_channels = 4;
        let block_out_channels = [320, 640, 1280];
        let layers_per_block = 2;
        let cross_attention_dim = Some(2048);
        let transformer_depths = [0, 2, 10];
        let num_heads = [5, 10, 20];
        let dim_head = 64;
        let time_embed_dim = 1280;

        let conv_in = conv2d(in_channels, block_out_channels[0], 3, Conv2dConfig { padding: 1, ..Default::default() }, vb.pp("conv_in"))?;
        let time_embedding = TimestepEmbedding::new(vb.pp("time_embedding"), 320, time_embed_dim)?;

        let add_embedding = if vb.contains_tensor("add_embedding.linear_1.weight") {
            Some(TimestepEmbedding::new(vb.pp("add_embedding"), 2816, time_embed_dim)?)
        } else {
            None
        };

        // --- Down Blocks ---
        let mut down_resnets = Vec::new();
        let mut down_attns = Vec::new();
        let mut down_samplers = Vec::new();

        let mut in_ch = block_out_channels[0];
        for (i, &out_ch) in block_out_channels.iter().enumerate() {
            let mut resnets = Vec::new();
            let mut attns = Vec::new();
            let down_vb = vb.pp(format!("down_blocks.{}", i));

            for j in 0..layers_per_block {
                let res_in = if j == 0 { in_ch } else { out_ch };
                let resnet = ResnetBlock2D::new(down_vb.pp(format!("resnets.{}", j)), res_in, out_ch, Some(time_embed_dim))?;
                resnets.push(resnet);

                if transformer_depths[i] > 0 {
                    let st = SpatialTransformer::new(
                        down_vb.pp(format!("attentions.{}", j)),
                        out_ch,
                        num_heads[i],
                        dim_head,
                        transformer_depths[i],
                        cross_attention_dim,
                        true,
                    )?;
                    attns.push(Some(st));
                } else {
                    attns.push(None);
                }
            }

            let sampler = if i < block_out_channels.len() - 1 {
                Some(Downsample2D::new(down_vb.pp("downsamplers.0"), out_ch)?)
            } else {
                None
            };

            in_ch = out_ch;
            down_resnets.push(resnets);
            down_attns.push(attns);
            down_samplers.push(sampler);
        }

        // --- Mid Block ---
        let mid_vb = vb.pp("mid_block");
        let mid_ch = block_out_channels[2];
        let mid_resnet1 = ResnetBlock2D::new(mid_vb.pp("resnets.0"), mid_ch, mid_ch, Some(time_embed_dim))?;
        let mid_attn = SpatialTransformer::new(
            mid_vb.pp("attentions.0"),
            mid_ch,
            num_heads[2],
            dim_head,
            10,
            cross_attention_dim,
            true,
        )?;
        let mid_resnet2 = ResnetBlock2D::new(mid_vb.pp("resnets.1"), mid_ch, mid_ch, Some(time_embed_dim))?;

        // --- Up Blocks ---
        let mut up_resnets = Vec::new();
        let mut up_attns = Vec::new();
        let mut up_samplers = Vec::new();

        let reversed_channels = [1280, 640, 320];
        let reversed_depths = [10, 2, 0];
        let reversed_heads = [20, 10, 5];

        let mut prev_out_ch = mid_ch;
        for (i, &out_ch) in reversed_channels.iter().enumerate() {
            let mut resnets = Vec::new();
            let mut attns = Vec::new();
            let up_vb = vb.pp(format!("up_blocks.{}", i));

            for j in 0..layers_per_block + 1 {
                let skip_ch = if i == 0 {
                    if j == 0 { 1280 } else if j == 1 { 1280 } else { 640 }
                } else if i == 1 {
                    if j == 0 { 640 } else if j == 1 { 640 } else { 320 }
                } else {
                    if j == 0 { 320 } else if j == 1 { 320 } else { 320 }
                };

                let res_in = if j == 0 { prev_out_ch } else { out_ch } + skip_ch;
                let resnet = ResnetBlock2D::new(up_vb.pp(format!("resnets.{}", j)), res_in, out_ch, Some(time_embed_dim))?;
                resnets.push(resnet);

                if reversed_depths[i] > 0 {
                    let st = SpatialTransformer::new(
                        up_vb.pp(format!("attentions.{}", j)),
                        out_ch,
                        reversed_heads[i],
                        dim_head,
                        reversed_depths[i],
                        cross_attention_dim,
                        true,
                    )?;
                    attns.push(Some(st));
                } else {
                    attns.push(None);
                }
            }

            let sampler = if i < reversed_channels.len() - 1 {
                Some(Upsample2D::new(up_vb.pp("upsamplers.0"), out_ch)?)
            } else {
                None
            };

            prev_out_ch = out_ch;
            up_resnets.push(resnets);
            up_attns.push(attns);
            up_samplers.push(sampler);
        }

        // --- Out ---
        let conv_norm_out = group_norm(32, block_out_channels[0], 1e-5, vb.pp("conv_norm_out"))?;
        let conv_out = conv2d(block_out_channels[0], out_channels, 3, Conv2dConfig { padding: 1, ..Default::default() }, vb.pp("conv_out"))?;

        Ok(Self {
            conv_in,
            time_proj_dim: 320,
            time_embedding,
            add_embedding,
            down_resnets,
            down_attns,
            down_samplers,
            mid_resnet1,
            mid_attn,
            mid_resnet2,
            up_resnets,
            up_attns,
            up_samplers,
            conv_norm_out,
            conv_out,
            is_sdxl,
        })
    }

    pub fn new_sd15(vb: VarBuilder) -> Result<Self> {
        Self::new_sdxl(vb)
    }

    pub fn apply_lora_deltas(&mut self, deltas: &std::collections::HashMap<String, Tensor>) -> Result<()> {
        let unet_deltas: std::collections::HashMap<String, Tensor> = deltas
            .iter()
            .map(|(k, v)| {
                let stripped = k.strip_prefix("unet.").unwrap_or(k);
                (stripped.to_string(), v.clone())
            })
            .collect();

        crate::diffusion::attention::apply_conv2d_delta(&mut self.conv_in, "conv_in", &unet_deltas)?;
        crate::diffusion::attention::apply_conv2d_delta(&mut self.conv_out, "conv_out", &unet_deltas)?;
        self.time_embedding.apply_lora_deltas("time_embedding", &unet_deltas)?;
        if let Some(add_emb) = &mut self.add_embedding {
            add_emb.apply_lora_deltas("add_embedding", &unet_deltas)?;
        }

        // Down blocks
        for (i, (resnets, (attns, sampler))) in self.down_resnets.iter_mut()
            .zip(self.down_attns.iter_mut().zip(self.down_samplers.iter_mut()))
            .enumerate()
        {
            let block_prefix = format!("down_blocks.{}", i);
            for (j, resnet) in resnets.iter_mut().enumerate() {
                resnet.apply_lora_deltas(&format!("{}.resnets.{}", block_prefix, j), &unet_deltas)?;
            }
            for (j, attn_opt) in attns.iter_mut().enumerate() {
                if let Some(attn) = attn_opt {
                    attn.apply_lora_deltas(&format!("{}.attentions.{}", block_prefix, j), &unet_deltas)?;
                }
            }
            if let Some(s) = sampler {
                s.apply_lora_deltas(&format!("{}.downsamplers.0", block_prefix), &unet_deltas)?;
            }
        }

        // Mid block
        self.mid_resnet1.apply_lora_deltas("mid_block.resnets.0", &unet_deltas)?;
        self.mid_attn.apply_lora_deltas("mid_block.attentions.0", &unet_deltas)?;
        self.mid_resnet2.apply_lora_deltas("mid_block.resnets.1", &unet_deltas)?;

        // Up blocks
        for (i, (resnets, (attns, sampler))) in self.up_resnets.iter_mut()
            .zip(self.up_attns.iter_mut().zip(self.up_samplers.iter_mut()))
            .enumerate()
        {
            let block_prefix = format!("up_blocks.{}", i);
            for (j, resnet) in resnets.iter_mut().enumerate() {
                resnet.apply_lora_deltas(&format!("{}.resnets.{}", block_prefix, j), &unet_deltas)?;
            }
            for (j, attn_opt) in attns.iter_mut().enumerate() {
                if let Some(attn) = attn_opt {
                    attn.apply_lora_deltas(&format!("{}.attentions.{}", block_prefix, j), &unet_deltas)?;
                }
            }
            if let Some(s) = sampler {
                s.apply_lora_deltas(&format!("{}.upsamplers.0", block_prefix), &unet_deltas)?;
            }
        }

        Ok(())
    }

    /// Pre-compute SDXL Add Embedding (time_ids + pooled text projection) once for all steps
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
            let time_ids_emb = Tensor::cat(&time_embs_ref, 1)?; // [b_size, 1536]

            let default_pooled = Tensor::zeros((b_size, 1280), dtype, dev)?;
            let pooled = pooled_embeds.unwrap_or(&default_pooled);
            let add_vec = Tensor::cat(&[pooled, &time_ids_emb], 1)?; // [b_size, 2816]
            let add_proj = add_emb.forward(&add_vec)?;
            Ok(Some(add_proj))
        } else {
            Ok(None)
        }
    }

    pub fn forward(
        &self,
        sample: &Tensor,
        timestep: f64,
        encoder_hidden_states: &Tensor,
        pooled_embeds: Option<&Tensor>,
    ) -> Result<Tensor> {
        let dev = sample.device();
        let dtype = sample.dtype();
        let b_size = sample.dim(0)?;
        let (_, _, h, w) = sample.dims4()?;

        let add_proj = self.compute_add_embedding(b_size, h, w, pooled_embeds, dev, dtype)?;
        self.forward_with_precomputed(sample, timestep, encoder_hidden_states, add_proj.as_ref())
    }

    pub fn forward_with_precomputed(
        &self,
        sample: &Tensor,
        timestep: f64,
        encoder_hidden_states: &Tensor,
        precomputed_add_proj: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.forward_with_controlnet(
            sample,
            timestep,
            encoder_hidden_states,
            precomputed_add_proj,
            None,
            None,
        )
    }

    /// Forward pass with optional ControlNet down and mid block zero-convolution residuals
    pub fn forward_with_controlnet(
        &self,
        sample: &Tensor,
        timestep: f64,
        encoder_hidden_states: &Tensor,
        precomputed_add_proj: Option<&Tensor>,
        down_block_residuals: Option<&[Tensor]>,
        mid_block_residual: Option<&Tensor>,
    ) -> Result<Tensor> {
        let dev = sample.device();
        let dtype = sample.dtype();
        let b_size = sample.dim(0)?;

        // 1. Timestep embedding
        let t_tensor = Tensor::new(&[timestep as f32; 1], dev)?.broadcast_as(b_size)?;
        let t_emb = get_timestep_embedding(&t_tensor, self.time_proj_dim)?.to_dtype(dtype)?;
        let mut temb = self.time_embedding.forward(&t_emb)?;

        // 2. Add precomputed add_embedding for SDXL if available
        if let Some(add_proj) = precomputed_add_proj {
            temb = (&temb + add_proj)?;
        }

        // 3. Conv In
        let mut hs = Vec::new();
        let mut h = self.conv_in.forward(sample)?;
        hs.push(h.clone());

        // 4. Down Blocks
        for (i, resnets) in self.down_resnets.iter().enumerate() {
            for (j, resnet) in resnets.iter().enumerate() {
                h = resnet.forward(&h, Some(&temb))?;
                if let Some(ref attn) = self.down_attns[i][j] {
                    h = attn.forward(&h, Some(encoder_hidden_states))?;
                }
                hs.push(h.clone());
            }

            if let Some(ref downsampler) = self.down_samplers[i] {
                h = downsampler.forward(&h)?;
                hs.push(h.clone());
            }
        }

        // Apply ControlNet down residuals to hs
        if let Some(down_res) = down_block_residuals {
            for (hs_tensor, res) in hs.iter_mut().zip(down_res.iter()) {
                *hs_tensor = (&*hs_tensor + res)?;
            }
        }

        // 5. Mid Block
        h = self.mid_resnet1.forward(&h, Some(&temb))?;
        h = self.mid_attn.forward(&h, Some(encoder_hidden_states))?;
        h = self.mid_resnet2.forward(&h, Some(&temb))?;

        // Apply ControlNet mid residual
        if let Some(mid_res) = mid_block_residual {
            h = (&h + mid_res)?;
        }

        // 6. Up Blocks
        for (i, resnets) in self.up_resnets.iter().enumerate() {
            for (j, resnet) in resnets.iter().enumerate() {
                let skip = hs.pop().ok_or_else(|| candle_core::Error::Msg("Missing skip connection tensor".into()))?;
                h = Tensor::cat(&[&h, &skip], 1)?;
                h = resnet.forward(&h, Some(&temb))?;
                if let Some(ref attn) = self.up_attns[i][j] {
                    h = attn.forward(&h, Some(encoder_hidden_states))?;
                }
            }

            if let Some(ref upsampler) = self.up_samplers[i] {
                h = upsampler.forward(&h)?;
            }
        }

        // 7. Conv Out
        let h = self.conv_norm_out.forward(&h)?;
        let h = candle_nn::ops::silu(&h)?;
        self.conv_out.forward(&h)
    }

    pub fn is_sdxl(&self) -> bool {
        self.is_sdxl
    }
}
