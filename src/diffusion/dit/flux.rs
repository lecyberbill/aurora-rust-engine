// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Pure Rust Flux.1 / MMDiT Transformer Architecture with FlashAttention-2

use candle_core::{Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};
use crate::diffusion::dit::blocks::{DoubleStreamBlock, SingleStreamBlock};
use crate::diffusion::dit::embeddings::TimestepEmbedder;

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
    img_in: Linear,
    txt_in: Linear,
    time_embedder: TimestepEmbedder,
    guidance_embedder: Option<TimestepEmbedder>,
    double_blocks: Vec<DoubleStreamBlock>,
    single_blocks: Vec<SingleStreamBlock>,
    final_linear: Linear,
    config: FluxConfig,
}

impl FluxTransformer {
    pub fn new(config: FluxConfig, vb: VarBuilder) -> Result<Self> {
        let img_in = linear(config.in_channels, config.hidden_size, vb.pp("img_in"))?;
        let txt_in = linear(4096, config.hidden_size, vb.pp("txt_in"))?; // 4096 dim from T5-XXL

        let time_embedder = TimestepEmbedder::new(config.hidden_size, 256, vb.pp("time_in"))?;
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

        let final_linear = linear(config.hidden_size, config.out_channels, vb.pp("final_layer.linear"))?;

        Ok(Self {
            img_in,
            txt_in,
            time_embedder,
            guidance_embedder,
            double_blocks,
            single_blocks,
            final_linear,
            config,
        })
    }

    /// Forward pass through the Multimodal Diffusion Transformer
    pub fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        timesteps: &Tensor,
        guidance: Option<&Tensor>,
    ) -> Result<Tensor> {
        // 1. Timestep (+ Guidance) Embedding
        let mut temb = self.time_embedder.forward(timesteps)?;
        if let (Some(g_emb), Some(g_val)) = (&self.guidance_embedder, guidance) {
            let g = g_emb.forward(g_val)?;
            temb = (&temb + &g)?;
        }

        // 2. Project Input Sequences
        let mut img_h = self.img_in.forward(img)?;
        let mut txt_h = self.txt_in.forward(txt)?;

        // 3. Double Stream (Joint Attention) Blocks
        for block in &self.double_blocks {
            let (next_img, next_txt) = block.forward(&img_h, &txt_h, &temb, None, None)?;
            img_h = next_img;
            txt_h = next_txt;
        }

        // 4. Single Stream Blocks (Tokens concatenation for Flux.1)
        if !self.single_blocks.is_empty() {
            let mut unified = Tensor::cat(&[&txt_h, &img_h], 1)?;
            for block in &self.single_blocks {
                unified = block.forward(&unified, &temb)?;
            }
            let txt_len = txt_h.dim(1)?;
            img_h = unified.narrow(1, txt_len, img_h.dim(1)?)?;
        }

        // 5. Final Output Projection
        self.final_linear.forward(&img_h)
    }

    pub fn config(&self) -> &FluxConfig {
        &self.config
    }
}
