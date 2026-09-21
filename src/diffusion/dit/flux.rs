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
    /// RoPE axis widths. Sum must equal `hidden_size / num_heads` (128 typically).
    /// Flux.2-Klein-4B/Dev use 4 axes `[32,32,32,32]`; Flux.2-Klein-9B uses 3 axes `[16,56,56]`.
    pub axes_dim: Vec<usize>,
    /// Multiplier applied to the incoming timestep before the sinusoidal time embedding. The pipeline
    /// feeds the FlowMatch sigma in `[0, 1]`; BFL/Flux and diffusers SD3/SD3.5 both expect the raw
    /// `0..1000` timestep in `time_text_embed`, so this is `1000.0` for every family.
    pub timestep_scale: f64,
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
            axes_dim: vec![16, 56, 56],
            timestep_scale: 1000.0,
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
            axes_dim: vec![16, 56, 56],
            timestep_scale: 1000.0,
        }
    }

    /// Flux.2 Klein 4B configuration (5 double blocks, 20 single blocks, shared modulation)
    pub fn klein_4b() -> Self {
        Self {
            in_channels: 128, // 32 latent channels * 2x2 patchify
            out_channels: 128,
            hidden_size: 3072,
            num_heads: 24,
            num_double_blocks: 5,
            num_single_blocks: 20,
            mlp_ratio: 6, // 3072 * 6 = 18432
            theta: 2000.0,
            guidance_embed: false,
            axes_dim: vec![32, 32, 32, 32], // Flux.2-Klein-4B (4 axes, validated)
            timestep_scale: 1000.0,
        }
    }

    /// Flux.2 Klein 9B configuration (8 double blocks, 24 single blocks, shared modulation, 4096 hidden)
    pub fn klein_9b() -> Self {
        Self {
            in_channels: 128, // 32 latent channels * 2x2 patchify
            out_channels: 128,
            hidden_size: 4096,
            num_heads: 32,
            num_double_blocks: 8,
            num_single_blocks: 24,
            mlp_ratio: 6,
            theta: 2000.0,
            guidance_embed: false,
            axes_dim: vec![32, 32, 32, 32], // Flux.2-Klein-9B (4 axes; 3D [16,56,56] produced a flat grey render)
            timestep_scale: 1000.0,
        }
    }

    /// Flux.2 Dev configuration (8 double blocks, 48 single blocks, 6144 hidden, guidance embed)
    pub fn flux2_dev() -> Self {
        Self {
            in_channels: 128,
            out_channels: 128,
            hidden_size: 6144,
            num_heads: 48,
            num_double_blocks: 8,
            num_single_blocks: 48,
            mlp_ratio: 6,
            theta: 2000.0,
            guidance_embed: true,
            axes_dim: vec![32, 32, 32, 32], // Flux.2-Dev (4 axes, validated by recognisable fox render)
            timestep_scale: 1000.0,
        }
    }

    /// Stable Diffusion 3.5 **Large** (38 DoubleStreamBlocks, 2432 hidden, head_dim 64, 16-ch).
    /// Verified against `stableDiffusion35Fp8_v35LargeTurbo.safetensors`.
    pub fn sd35_large() -> Self {
        Self {
            in_channels: 64, // 16 latent channels * 2x2 patchify
            out_channels: 64,
            hidden_size: 2432,
            num_heads: 38,
            num_double_blocks: 38,
            num_single_blocks: 0,
            mlp_ratio: 4,
            theta: 10_000.0,
            guidance_embed: false,
            axes_dim: vec![16, 56, 56], // SD3.5 (3 axes)
            timestep_scale: 1000.0,     // diffusers SD3 passes the raw 0..1000 timestep to `time_text_embed`
        }
    }

    /// Stable Diffusion 3.5 **Medium** (24 DoubleStreamBlocks, 1536 hidden, head_dim 64, 16-ch).
    pub fn sd35_medium() -> Self {
        Self {
            in_channels: 64, // 16 latent channels * 2x2 patchify
            out_channels: 64,
            hidden_size: 1536,
            num_heads: 24,
            num_double_blocks: 24,
            num_single_blocks: 0,
            mlp_ratio: 4,
            theta: 10_000.0,
            guidance_embed: false,
            axes_dim: vec![16, 56, 56], // SD3.5 (3 axes)
            timestep_scale: 1000.0,     // diffusers SD3 passes the raw 0..1000 timestep to `time_text_embed`
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
    /// SD3 / SD3.5 fixed 2D-sincos positional embedding `[1, max_h*max_w, hidden]`, centre-cropped to
    /// the actual patch grid at forward time (added to the image tokens after `img_in`). When set,
    /// the model does **not** use RoPE (SD3 attention is position-agnostic). `None` for Flux.
    pub pos_embed: Option<Tensor>,
    /// When true, the final AdaLN chunk order is `[shift, scale]` (BFL native); when false it is
    /// `[scale, shift]` (Diffusers convention). Diffusers `norm_out` chunk(2) yields [scale, shift];
    /// BFL-exported checkpoints store [shift, scale]. Mismatching this collapses the output variance
    /// (flat blue frame).
    pub swap_scale_shift: bool,
}

impl FluxTransformer {
    pub fn new(config: FluxConfig, vb: VarBuilder) -> Result<Self> {
        let img_in = linear(config.in_channels, config.hidden_size, vb.pp("img_in"))?;
        let txt_in = linear(4096, config.hidden_size, vb.pp("txt_in"))?; // 4096 dim from T5-XXL

        let time_embedder = TimestepEmbedder::new_with_factor(config.hidden_size, 256, config.timestep_scale, vb.pp("time_in"))?;
        // Pooled-projection input width varies by family: Flux = 768 (CLIP-L pooled), SD3.5 = 2048
        // (CLIP-L+G pooled). Probe the checkpoint's actual width.
        let vector_in = {
            let mut found = None;
            for in_dim in [768usize, 2048, 1280] {
                if let (Ok(in_l), Ok(out_l)) = (
                    linear(in_dim, config.hidden_size, vb.pp("vector_in.in_layer")),
                    linear(config.hidden_size, config.hidden_size, vb.pp("vector_in.out_layer")),
                ) { found = Some((in_l, out_l)); break; }
            }
            found
        };
        let guidance_embedder = if config.guidance_embed {
            Some(TimestepEmbedder::new_with_factor(config.hidden_size, 256, 1000.0, vb.pp("guidance_in"))?)
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
            pos_embed: None,
            swap_scale_shift: true, // BFL native order
        })
    }

    /// Attach the SD3/SD3.5 fixed sincos positional embedding `[1, max_h*max_w, hidden]`. Setting it
    /// also switches the model into position-agnostic (no-RoPE) mode. Loaded from the checkpoint by
    /// the pipeline (its shape varies with `pos_embed_max_size`, so it is set explicitly rather than
    /// through the fixed-shape `VarBuilder`).
    pub fn set_pos_embed(&mut self, pos_embed: Tensor) {
        self.pos_embed = Some(pos_embed);
    }

    /// Construct FluxTransformer in Streaming Mode (< 100MB VRAM header footprint)
    pub fn new_streaming(config: FluxConfig, vb: VarBuilder) -> Result<Self> {
        let linear_layer = |in_d: usize, out_d: usize, path: VarBuilder| -> Result<Linear> {
            linear(in_d, out_d, path.clone()).or_else(|_| candle_nn::linear_no_bias(in_d, out_d, path))
        };

        let img_in = linear_layer(config.in_channels, config.hidden_size, vb.pp("img_in"))?;
        let txt_in = linear_layer(config.hidden_size, config.hidden_size, vb.pp("txt_in"))
            .or_else(|_| linear_layer(4096, config.hidden_size, vb.pp("txt_in")))
            .or_else(|_| linear_layer(7680, config.hidden_size, vb.pp("txt_in")))
            .or_else(|_| linear_layer(12288, config.hidden_size, vb.pp("txt_in")))
            .or_else(|_| linear_layer(15360, config.hidden_size, vb.pp("txt_in")))?;

        let time_embedder = TimestepEmbedder::new_with_factor(config.hidden_size, 256, config.timestep_scale, vb.pp("time_in"))?;
        let vector_in = {
            let mut found = None;
            for in_dim in [768usize, 2048, 1280] {
                if let (Ok(in_l), Ok(out_l)) = (
                    linear_layer(in_dim, config.hidden_size, vb.pp("vector_in.in_layer")),
                    linear_layer(config.hidden_size, config.hidden_size, vb.pp("vector_in.out_layer")),
                ) { found = Some((in_l, out_l)); break; }
            }
            found
        };
        let guidance_embedder = if config.guidance_embed {
            Some(TimestepEmbedder::new_with_factor(config.hidden_size, 256, 1000.0, vb.pp("guidance_in"))?)
        } else {
            None
        };

        let final_mod = linear_layer(config.hidden_size, config.hidden_size * 2, vb.pp("final_layer.adaLN_modulation.1"))?;
        let final_linear = linear_layer(config.hidden_size, config.out_channels, vb.pp("final_layer.linear"))?;

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
            pos_embed: None,
            swap_scale_shift: true, // BFL native order
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
        self.forward_with_streamer_and_refs(img, txt, timesteps, y, guidance, None, streamer)
    }

    /// Forward pass through the Multimodal Diffusion Transformer with support for sequential streaming
    /// and optional multi-image reference latents (Flux.2 4D RoPE conditioning).
    pub fn forward_with_streamer_and_refs(
        &self,
        img: &Tensor,
        txt: &Tensor,
        timesteps: &Tensor,
        y: Option<&Tensor>,
        guidance: Option<&Tensor>,
        ref_latents: Option<&[(Tensor, usize, usize)]>, // (tokens [1, rh*rw, 128], rh, rw)
        streamer: Option<&crate::diffusion::dit::streamer::SequentialBlockStreamer>,
    ) -> Result<Tensor> {
        if let Ok(dir) = std::env::var("FLUX_MODEL_DUMP") {
            let f = |t: &Tensor| -> String { let v=t.to_dtype(candle_core::DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap(); format!("{:?}", v) };
            let _ = std::fs::write(format!("{dir}/m_img.txt"), f(img));
            let _ = std::fs::write(format!("{dir}/m_txt.txt"), f(txt));
            let _ = std::fs::write(format!("{dir}/m_sigma.txt"), format!("{:?}", timesteps.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1::<f32>()?));
            if let Some(y) = y { let _ = std::fs::write(format!("{dir}/m_y.txt"), f(y)); }
        }
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

        // Compute 3D/4D Rotary Position Embeddings (RoPE) from the model's declared axes
        let txt_len = txt_h.dim(1)?;
        let img_seq = img_h.dim(1)?;
        let patch_side = (img_seq as f64).sqrt() as usize;

        // SD3 / SD3.5: add the fixed sincos positional embedding (centre-cropped to the patch grid)
        // to the image tokens. When present, the model uses NO RoPE (position is entirely in the
        // pos_embed; SD3 attention is position-agnostic).
        let ropeless = self.pos_embed.is_some();
        if let Some(pe) = &self.pos_embed {
            let total = pe.dim(1)?;
            let max_side = (total as f64).sqrt() as usize;
            if max_side >= patch_side {
                let pe4 = pe.reshape((1, max_side, max_side, self.config.hidden_size))?;
                let top = (max_side - patch_side) / 2;
                let left = (max_side - patch_side) / 2;
                let cropped = pe4
                    .narrow(1, top, patch_side)?
                    .narrow(2, left, patch_side)?
                    .reshape((1, patch_side * patch_side, self.config.hidden_size))?
                    .to_dtype(img_h.dtype())?;
                img_h = (img_h + cropped)?;
            }
        }

        // Project reference latents and collect patch coordinates for 4D RoPE
        let mut ref_patches = Vec::new();
        if let Some(refs) = ref_latents {
            for (ref_tok, rh, rw) in refs {
                let ref_h = self.img_in.forward(ref_tok)?;
                img_h = Tensor::cat(&[&img_h, &ref_h], 1)?;
                ref_patches.push((*rh, *rw));
            }
        }

        let axes_dim = self.config.axes_dim.clone();
        let (freqs_cos, freqs_sin) = crate::diffusion::dit::embeddings::create_flux_rope_embeddings_with_refs(
            txt_len,
            patch_side,
            patch_side,
            &ref_patches,
            &axes_dim,
            self.config.theta,
            img.device(),
        )?;

        let total_img_seq = img_h.dim(1)?;
        let txt_cos = freqs_cos.narrow(0, 0, txt_len)?;
        let txt_sin = freqs_sin.narrow(0, 0, txt_len)?;
        let img_cos = freqs_cos.narrow(0, txt_len, total_img_seq)?;
        let img_sin = freqs_sin.narrow(0, txt_len, total_img_seq)?;

        // 3. Double Stream (Joint Attention) Blocks
        if std::env::var("FLUX_TRACE").is_ok() {
            let rms = |t: &Tensor, tag: &str| -> f32 {
                let f = t.to_dtype(candle_core::DType::F32).unwrap().flatten_all().unwrap();
                if let Ok(v) = f.to_vec1::<f32>() {
                    let m = v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / v.len() as f64;
                    eprintln!("    [TRACE] {tag} rms={:.5}", m.sqrt());
                    m.sqrt() as f32
                } else { 0.0 }
            };
            rms(&img_h, "img_h-in"); rms(&txt_h, "txt_h-in"); rms(&temb, "temb");
        }
        // SD3/SD3.5 pass no rotary (position lives in pos_embed); Flux passes the RoPE tables.
        let img_cos_o = if ropeless { None } else { Some(&img_cos) };
        let img_sin_o = if ropeless { None } else { Some(&img_sin) };
        let txt_cos_o = if ropeless { None } else { Some(&txt_cos) };
        let txt_sin_o = if ropeless { None } else { Some(&txt_sin) };

        if let Some(s) = streamer {
            for i in 0..self.config.num_double_blocks {
                let (next_img, next_txt) = s.execute_double_block(
                    i,
                    &img_h,
                    &txt_h,
                    &temb,
                    img_cos_o,
                    img_sin_o,
                    txt_cos_o,
                    txt_sin_o,
                )?;
                img_h = next_img;
                txt_h = next_txt;
                if std::env::var("FLUX_TRACE").is_ok() {
                    let f = img_h.to_dtype(candle_core::DType::F32).unwrap().flatten_all().unwrap();
                    if let Ok(v) = f.to_vec1::<f32>() {
                        let m = v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / v.len() as f64;
                        eprintln!("    [TRACE] after double.{i} img_rms={:.5}", m.sqrt());
                    }
                }
            }
        } else {
            for block in &self.double_blocks {
                let (next_img, next_txt) = block.forward(
                    &img_h,
                    &txt_h,
                    &temb,
                    img_cos_o,
                    img_sin_o,
                    txt_cos_o,
                    txt_sin_o,
                )?;
                img_h = next_img;
                txt_h = next_txt;
            }
        }

        // 4. Single Stream Blocks (Tokens concatenation for Flux.1)
        if self.config.num_single_blocks > 0 {
            let mut unified = Tensor::cat(&[&txt_h, &img_h], 1)?;
            if std::env::var("FLUX_TRACE").is_ok() {
                let rms = |t: &Tensor| -> f32 {
                    let f = t.to_dtype(candle_core::DType::F32).unwrap().flatten_all().unwrap();
                    if let Ok(v) = f.to_vec1::<f32>() {
                        let m = v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / v.len() as f64;
                        m.sqrt() as f32
                    } else { 0.0 }
                };
                eprintln!("    [TRACE] pre-single: unified_rms={:.4} img_rms={:.4} txt_rms={:.4} txt_len={} img_len={}",
                    rms(&unified), rms(&img_h), rms(&txt_h), txt_h.dim(1)?, img_h.dim(1)?);
            }
            let fcos = if ropeless { None } else { Some(&freqs_cos) };
            let fsin = if ropeless { None } else { Some(&freqs_sin) };
            if let Some(s) = streamer {
                for i in 0..self.config.num_single_blocks {
                    unified = s.execute_single_block(i, &unified, &temb, fcos, fsin)?;
                }
            } else {
                for block in &self.single_blocks {
                    unified = block.forward(&unified, &temb, fcos, fsin)?;
                }
            }
            let txt_len = txt_h.dim(1)?;
            img_h = unified.narrow(1, txt_len, img_h.dim(1)?)?;
            if std::env::var("FLUX_TRACE").is_ok() {
                let f = img_h.to_dtype(candle_core::DType::F32).unwrap().flatten_all().unwrap();
                if let Ok(v) = f.to_vec1::<f32>() {
                    let m = v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / v.len() as f64;
                    eprintln!("    [TRACE] after single blocks img_rms={:.5}", m.sqrt());
                }
            }
        }

        // 5. Final AdaLN-Zero Modulation and Linear Output Projection
        let temb_silu = candle_nn::ops::silu(&temb)?;
        let mod_out = self.final_mod.forward(&temb_silu)?;
        let chunks = mod_out.chunk(2, mod_out.dims().len() - 1)?;
        // BFL native stores [shift, scale]; Diffusers convention is [scale, shift]. Using the wrong
        // order collapses the output variance (flat blue frame).
        let (shift, scale) = if self.swap_scale_shift {
            (&chunks[0], &chunks[1])
        } else {
            (&chunks[1], &chunks[0])
        };

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

        let mut out = self.final_linear.forward(&img_modulated)?;
        // If reference latents were attached, only extract the velocity tokens corresponding to the main canvas
        if ref_latents.is_some() {
            out = out.narrow(1, 0, img_seq)?;
        }
        // SD3/SD3.5 `proj_out` yields the fused 2x2 patch channels as `(ph, pw, c)` (diffusers
        // `einsum "nhwpqc->nchpwq"`), whereas the packed input consumed by `img_in` (from the patch
        // Conv2d) is `(c, ph, pw)`. Reorder the velocity so the output packing matches the input,
        // keeping the latent state (and the Euler step) self-consistent.
        let out = if self.pos_embed.is_some() {
            let (b, seq, oc) = out.dims3()?;
            out.reshape((b, seq, 2, 2, oc / 4))?
                .permute((0, 1, 4, 2, 3))?
                .contiguous()?
                .reshape((b, seq, oc))?
        } else {
            out
        };
        if let Ok(dir) = std::env::var("FLUX_MODEL_DUMP") {
            let f = |t: &Tensor| -> String { let v=t.to_dtype(candle_core::DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap(); format!("{:?}", v) };
            let _ = std::fs::write(format!("{dir}/m_out.txt"), f(&out));
        }
        Ok(out)
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

    /// The input feature width expected by `txt_in` (w - 1 from the weight shape).
    /// Flux.2-Klein-9B: 12288, Flux.2-Dev: 15360, Flux.1: 4096 (T5) / 7680 (Qwen3-4B).
    pub fn txt_in_expected_in(&self) -> usize {
        let w = self.txt_in.weight();
        if w.dims().len() == 2 { w.dims()[1] } else { 0 }
    }
}
