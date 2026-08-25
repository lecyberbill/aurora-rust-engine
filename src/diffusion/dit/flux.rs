// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Pure Rust Flux.1 / MMDiT Transformer Architecture with FlashAttention-2

use candle_core::{Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};
use crate::diffusion::dit::blocks::{DoubleStreamBlock, SingleStreamBlock};
use crate::diffusion::dit::embeddings::{AdaLNZeroModulation, TimestepEmbedder};

/// Configuration for Flux.1 / SD 3.5 MMDiT models
#[derive(Debug, Clone)]
pub struct FluxConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_double_blocks: usize,
    pub num_single_blocks: usize,
    pub mlp_ratio: usize,
    pub theta: f64,
    pub guidance_embed: bool,
}

impl FluxConfig {
    /// Flux.1-schnell configuration (19 double blocks, 38 single blocks, 4 steps fast inference)
    pub fn schnell() -> Self {
        Self {
            in_channels: 64, // 16 latent channels * 2x2 patchify
            out_channels: 64,
            hidden_size: 3072,
            num_heads: 24,
            num_double_blocks: 19,
            num_single_blocks: 38,
            mlp_ratio: 4,
            theta: 10_000.0,
            guidance_embed: false,
        }
    }

    /// Flux.1-dev configuration (with guidance vector embedding)
    pub fn dev() -> Self {
        Self {
            in_channels: 64,
            out_channels: 64,
            hidden_size: 3072,
            num_heads: 24,
            num_double_blocks: 19,
            num_single_blocks: 38,
            mlp_ratio: 4,
            theta: 10_000.0,
            guidance_embed: true,
        }
    }

    /// Stable Diffusion 3.5 Large (24 DoubleStreamBlocks, 1536 hidden dim)
    pub fn sd35_large() -> Self {
        Self {
            in_channels: 16,
            out_channels: 16,
            hidden_size: 1536,
            num_heads: 24,
            num_double_blocks: 24,
            num_single_blocks: 0,
            mlp_ratio: 4,
            theta: 10_000.0,
            guidance_embed: false,
        }
    }
}

/// Pure Rust Flux.1 / SD 3.5 Multimodal Diffusion Transformer
#[derive(Debug, Clone)]
pub struct FluxTransformer {
    pub img_in: Linear,
    pub txt_in: Linear,
    pub time_embedder: TimestepEmbedder,
    pub vector_in: Option<(Linear, Linear)>,
    pub guidance_embedder: Option<TimestepEmbedder>,
    pub double_blocks: Vec<DoubleStreamBlock>,
    pub single_blocks: Vec<SingleStreamBlock>,
    pub final_mod: Linear,
    pub final_linear: Linear,
    pub config: FluxConfig,
}

impl FluxTransformer {
    pub fn new(config: FluxConfig, vb: VarBuilder) -> Result<Self> {
        let img_in = linear(config.in_channels, config.hidden_size, vb.pp("img_in"))?;
        let txt_in = linear(4096, config.hidden_size, vb.pp("txt_in"))?; // 4096 dim from T5-XXL

        let time_embedder = TimestepEmbedder::new(config.hidden_size, 256, vb.pp("time_in"))?;
        let vector_in = match (linear(768, config.hidden_size, vb.pp("vector_in.in_layer")), linear(config.hidden_size, config.hidden_size, vb.pp("vector_in.out_layer"))) {
            (Ok(in_l), Ok(out_l)) => Some((in_l, out_l)),
            _ => None,
        };
        let guidance_embedder = if config.guidance_embed {
            Some(TimestepEmbedder::new(config.hidden_size, 256, vb.pp("guidance_in"))?)
        } else {
            None
        };

        let mut double_blocks = Vec::with_capacity(config.num_double_blocks);
        for i in 0..config.num_double_blocks {
            let block = DoubleStreamBlock::new(
                config.hidden_size,
                config.num_heads,
                config.mlp_ratio,
                vb.pp(format!("double_blocks.{}", i)),
            )?;
            double_blocks.push(block);
        }

        let mut single_blocks = Vec::with_capacity(config.num_single_blocks);
        for i in 0..config.num_single_blocks {
            let block = SingleStreamBlock::new(
                config.hidden_size,
                config.num_heads,
                config.mlp_ratio,
                vb.pp(format!("single_blocks.{}", i)),
            )?;
            single_blocks.push(block);
        }

        let final_mod = linear(config.hidden_size, config.hidden_size * 2, vb.pp("final_layer.adaLN_modulation.1"))?;
        let final_linear = linear(config.hidden_size, config.out_channels, vb.pp("final_layer.linear"))?;

        Ok(Self {
            img_in,
            txt_in,
            time_embedder,
            vector_in,
            guidance_embedder,
            double_blocks,
            single_blocks,
            final_mod,
            final_linear,
            config,
        })
    }

    /// Construct FluxTransformer in Streaming Mode (< 100MB VRAM header footprint)
    pub fn new_streaming(config: FluxConfig, vb: VarBuilder) -> Result<Self> {
        let img_in = linear(config.in_channels, config.hidden_size, vb.pp("img_in"))?;
        let txt_in = linear(4096, config.hidden_size, vb.pp("txt_in"))?;

        let time_embedder = TimestepEmbedder::new(config.hidden_size, 256, vb.pp("time_in"))?;
        let vector_in = match (linear(768, config.hidden_size, vb.pp("vector_in.in_layer")), linear(config.hidden_size, config.hidden_size, vb.pp("vector_in.out_layer"))) {
            (Ok(in_l), Ok(out_l)) => Some((in_l, out_l)),
            _ => None,
        };
        let guidance_embedder = if config.guidance_embed {
            Some(TimestepEmbedder::new(config.hidden_size, 256, vb.pp("guidance_in"))?)
        } else {
            None
        };

        let final_mod = linear(config.hidden_size, config.hidden_size * 2, vb.pp("final_layer.adaLN_modulation.1"))?;
        let final_linear = linear(config.hidden_size, config.out_channels, vb.pp("final_layer.linear"))?;

        Ok(Self {
            img_in,
            txt_in,
            time_embedder,
            vector_in,
            guidance_embedder,
            double_blocks: Vec::new(),
            single_blocks: Vec::new(),
            final_mod,
            final_linear,
            config,
        })
    }

    /// Forward pass with optional Sequential Block Streamer for Ultra-Low VRAM (< 6GB) execution
    pub fn forward_with_streamer(
        &self,
        img: &Tensor,
        txt: &Tensor,
        timesteps: &Tensor,
        y: Option<&Tensor>,
        guidance: Option<&Tensor>,
        streamer: Option<&crate::diffusion::dit::streamer::SequentialBlockStreamer>,
    ) -> Result<Tensor> {
        // 1. Timestep (+ Vector In + Guidance) Embedding
        let mut temb = self.time_embedder.forward(timesteps)?;

        if let (Some((in_l, out_l)), Some(y_vec)) = (&self.vector_in, y) {
            let h = in_l.forward(y_vec)?.silu()?;
            let v_emb = out_l.forward(&h)?;
            temb = (&temb + &v_emb)?;
        }

        if let (Some(g_emb), Some(g_val)) = (&self.guidance_embedder, guidance) {
            let g = g_emb.forward(g_val)?;
            temb = (&temb + &g)?;
        }

        // 2. Project Input Sequences
        let mut img_h = self.img_in.forward(img)?;
        let mut txt_h = self.txt_in.forward(txt)?;

        // Compute 3D Rotary Position Embeddings (RoPE)
        let txt_len = txt_h.dim(1)?;
        let img_seq = img_h.dim(1)?;
        let patch_side = (img_seq as f64).sqrt() as usize;
        let (freqs_cos, freqs_sin) = crate::diffusion::dit::embeddings::create_flux_rope_embeddings(
            txt_len,
            patch_side,
            patch_side,
            self.config.theta,
            img.device(),
        )?;

        let txt_cos = freqs_cos.narrow(0, 0, txt_len)?;
        let txt_sin = freqs_sin.narrow(0, 0, txt_len)?;
        let img_cos = freqs_cos.narrow(0, txt_len, img_seq)?;
        let img_sin = freqs_sin.narrow(0, txt_len, img_seq)?;

        // 3. Double Stream (Joint Attention) Blocks
        if let Some(s) = streamer {
            for i in 0..self.config.num_double_blocks {
                let (next_img, next_txt) = s.execute_double_block(
                    i,
                    &img_h,
                    &txt_h,
                    &temb,
                    Some(&img_cos),
                    Some(&img_sin),
                    Some(&txt_cos),
                    Some(&txt_sin),
                )?;
                img_h = next_img;
                txt_h = next_txt;
            }
        } else {
            for block in &self.double_blocks {
                let (next_img, next_txt) = block.forward(
                    &img_h,
                    &txt_h,
                    &temb,
                    Some(&img_cos),
                    Some(&img_sin),
                    Some(&txt_cos),
                    Some(&txt_sin),
                )?;
                img_h = next_img;
                txt_h = next_txt;
            }
        }

        // 4. Single Stream Blocks (Tokens concatenation for Flux.1)
        if self.config.num_single_blocks > 0 {
            let mut unified = Tensor::cat(&[&txt_h, &img_h], 1)?;
            if let Some(s) = streamer {
                for i in 0..self.config.num_single_blocks {
                    unified = s.execute_single_block(i, &unified, &temb, Some(&freqs_cos), Some(&freqs_sin))?;
                }
            } else {
                for block in &self.single_blocks {
                    unified = block.forward(&unified, &temb, Some(&freqs_cos), Some(&freqs_sin))?;
                }
            }
            let txt_len = txt_h.dim(1)?;
            img_h = unified.narrow(1, txt_len, img_h.dim(1)?)?;
        }

        // 5. Final AdaLN-Zero Modulation and Linear Output Projection
        let temb_silu = candle_nn::ops::silu(&temb)?;
        let mod_out = self.final_mod.forward(&temb_silu)?;
        let chunks = mod_out.chunk(2, mod_out.dims().len() - 1)?;
        let shift = &chunks[0];
        let scale = &chunks[1];

        // LayerNorm(elementwise_affine=False) on img_h
        let orig_dtype = img_h.dtype();
        let img_f32 = img_h.to_dtype(candle_core::DType::F32)?;
        let mean = img_f32.mean_keepdim(img_f32.dims().len() - 1)?;
        let diff = img_f32.broadcast_sub(&mean)?;
        let var = diff.sqr()?.mean_keepdim(diff.dims().len() - 1)?;
        let std = (var + 1e-6)?.sqrt()?;
        let img_normed = diff.broadcast_div(&std)?.to_dtype(orig_dtype)?;

        let scale = (scale.unsqueeze(1)? + 1.0)?;
        let shift = shift.unsqueeze(1)?;
        let img_modulated = img_normed.broadcast_mul(&scale)?.broadcast_add(&shift)?;

        self.final_linear.forward(&img_modulated)
    }

    /// Forward pass through the Multimodal Diffusion Transformer (in-memory blocks)
    pub fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        timesteps: &Tensor,
        y: Option<&Tensor>,
        guidance: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.forward_with_streamer(img, txt, timesteps, y, guidance, None)
    }

    pub fn config(&self) -> &FluxConfig {
        &self.config
    }
}
