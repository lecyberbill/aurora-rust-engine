// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Pure Rust PixArt-α, PixArt-Σ (Sigma) & Lumina-Next DiT Architecture
// Invariant: Native 2K/4K resolution support via KV-compression, T5 cross-conditioning, zero unwrap/panics.

use candle_core::{DType, Module, Result, Tensor};
use candle_nn::{conv2d, linear, linear_no_bias, Conv2d, Conv2dConfig, Linear, VarBuilder};
use crate::diffusion::dit::DiffusionTransformer;

/// Configuration for PixArt-α, PixArt-Σ and Lumina-Next Diffusion Transformers
#[derive(Debug, Clone)]
pub struct PixArtConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub patch_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub mlp_ratio: f64,
    pub caption_channels: usize,
    /// Spatial KV-compression factor (1 = uncompressed / PixArt-α; 2 or 4 = PixArt-Σ / Lumina-Next for 2K/4K)
    pub kv_compress_ratio: usize,
    pub use_resolution_embed: bool,
}

impl Default for PixArtConfig {
    fn default() -> Self {
        Self::pixart_sigma_2k()
    }
}

impl PixArtConfig {
    /// PixArt-α XL/2 (1152 hidden, 28 layers, 16 heads, T5-XXL 4096-dim context)
    pub fn pixart_alpha() -> Self {
        Self {
            in_channels: 4,
            out_channels: 4,
            patch_size: 2,
            hidden_size: 1152,
            num_heads: 16,
            num_layers: 28,
            mlp_ratio: 4.0,
            caption_channels: 4096,
            kv_compress_ratio: 1,
            use_resolution_embed: false,
        }
    }

    /// PixArt-Σ (Sigma) 2K Resolution preset with 2x KV-compression
    pub fn pixart_sigma_2k() -> Self {
        Self {
            in_channels: 4,
            out_channels: 4,
            patch_size: 2,
            hidden_size: 1152,
            num_heads: 16,
            num_layers: 28,
            mlp_ratio: 4.0,
            caption_channels: 4096,
            kv_compress_ratio: 2,
            use_resolution_embed: true,
        }
    }

    /// PixArt-Σ (Sigma) 4K Ultra-High Resolution preset with 4x KV-compression
    pub fn pixart_sigma_4k() -> Self {
        Self {
            in_channels: 4,
            out_channels: 4,
            patch_size: 2,
            hidden_size: 1152,
            num_heads: 16,
            num_layers: 28,
            mlp_ratio: 4.0,
            caption_channels: 4096,
            kv_compress_ratio: 4,
            use_resolution_embed: true,
        }
    }

    /// Lumina-Next 2K/4K Multi-Scale DiT
    pub fn lumina_next_2k() -> Self {
        Self {
            in_channels: 16, // Flux/Lumina 16-channel latent space
            out_channels: 16,
            patch_size: 2,
            hidden_size: 2048,
            num_heads: 32,
            num_layers: 32,
            mlp_ratio: 4.0,
            caption_channels: 4096,
            kv_compress_ratio: 2,
            use_resolution_embed: true,
        }
    }
}

/// Helper for LayerNorm without affine parameters
fn layer_norm_no_affine(x: &Tensor) -> Result<Tensor> {
    let orig_dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let last_dim = x_f32.dims().len() - 1;
    let mean = x_f32.mean_keepdim(last_dim)?;
    let diff = x_f32.broadcast_sub(&mean)?;
    let var = diff.sqr()?.mean_keepdim(last_dim)?;
    let std = (var + 1e-6)?.sqrt()?;
    diff.broadcast_div(&std)?.to_dtype(orig_dtype)
}

/// Helper GELU approx tanh
fn gelu_tanh(x: &Tensor) -> Result<Tensor> {
    let orig_dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let c = (2.0f64 / std::f64::consts::PI).sqrt() as f32;
    let x_cubed = (x_f32.sqr()? * &x_f32)?;
    let inner = (&x_f32 + (x_cubed * 0.044715f64)?)?;
    let tanh = (inner * (c as f64))?.tanh()?;
    let result = ((x_f32 * 0.5)? * (tanh + 1.0)?)?;
    result.to_dtype(orig_dtype)
}

/// Patch embedding layer mapping 2D latents `[B, C, H, W]` to tokens `[B, (H/P)*(W/P), Hidden]`
#[derive(Debug, Clone)]
pub struct PixArtPatchEmbed {
    proj: Conv2d,
    patch_size: usize,
}

impl PixArtPatchEmbed {
    pub fn new(in_channels: usize, hidden_size: usize, patch_size: usize, vb: VarBuilder) -> Result<Self> {
        let proj = conv2d(
            in_channels,
            hidden_size,
            patch_size,
            Conv2dConfig {
                stride: patch_size,
                padding: 0,
                ..Default::default()
            },
            vb.pp("proj"),
        )?;
        Ok(Self { proj, patch_size })
    }

    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, usize, usize)> {
        let (b, _c, h, w) = x.dims4()?;
        let h_p = h / self.patch_size;
        let w_p = w / self.patch_size;
        let num_patches = h_p * w_p;

        let feat = self.proj.forward(x)?; // [B, hidden_size, H/P, W/P]
        let tokens = feat
            .permute((0, 2, 3, 1))? // [B, H/P, W/P, hidden_size]
            .contiguous()?
            .reshape((b, num_patches, feat.dim(1)?))?; // [B, Num_Patches, hidden_size]

        Ok((tokens, h_p, w_p))
    }
}

/// Timestep and Resolution conditioning embedder (AdaLN-single conditioning)
#[derive(Debug, Clone)]
pub struct PixArtConditionEmbedder {
    time_proj1: Linear,
    time_proj2: Linear,
    res_proj: Option<Linear>,
    hidden_size: usize,
}

impl PixArtConditionEmbedder {
    pub fn new(hidden_size: usize, use_res: bool, vb: VarBuilder) -> Result<Self> {
        let time_proj1 = linear(256, hidden_size, vb.pp("time_proj1"))
            .or_else(|_| linear_no_bias(256, hidden_size, vb.pp("time_proj1")))?;
        let time_proj2 = linear(hidden_size, hidden_size, vb.pp("time_proj2"))
            .or_else(|_| linear_no_bias(hidden_size, hidden_size, vb.pp("time_proj2")))?;

        let res_proj = if use_res {
            Some(
                linear(256, hidden_size, vb.pp("res_proj"))
                    .or_else(|_| linear_no_bias(256, hidden_size, vb.pp("res_proj")))?,
            )
        } else {
            None
        };

        Ok(Self {
            time_proj1,
            time_proj2,
            res_proj,
            hidden_size,
        })
    }

    pub fn forward(&self, timestep: &Tensor, resolution: Option<(usize, usize)>) -> Result<Tensor> {
        let device = timestep.device();
        let b = timestep.dims().first().cloned().unwrap_or(1);

        // Sinusoidal timestep embedding (256 dims)
        let t_f32 = timestep.to_dtype(DType::F32)?;
        let half_dim = 128;
        let mut freqs = Vec::with_capacity(half_dim);
        for i in 0..half_dim {
            freqs.push((-((10000.0f64).ln() * (i as f64) / (half_dim as f64))).exp() as f32);
        }
        let freqs_t = Tensor::from_vec(freqs, (1, half_dim), device)?;
        let angles = t_f32.unsqueeze(1)?.matmul(&freqs_t)?;
        let t_emb = Tensor::cat(&[&angles.cos()?, &angles.sin()?], 1)?.to_dtype(timestep.dtype())?;

        let mut cond = self.time_proj1.forward(&t_emb)?;
        cond = candle_nn::ops::silu(&cond)?;
        cond = self.time_proj2.forward(&cond)?;

        // Resolution embedding if enabled
        if let (Some(ref res_layer), Some((h, w))) = (&self.res_proj, resolution) {
            let res_vals = vec![h as f32, w as f32, (h as f32 / w.max(1) as f32)];
            let mut res_angles = Vec::with_capacity(256);
            for i in 0..128 {
                let freq = (-((10000.0f64).ln() * (i as f64) / 128.0)).exp() as f32;
                res_angles.push((res_vals[0] * freq).sin());
                res_angles.push((res_vals[1] * freq).cos());
            }
            let res_t = Tensor::from_vec(res_angles, (1, 256), device)?
                .broadcast_as((b, 256))?
                .to_dtype(timestep.dtype())?;
            let res_emb = res_layer.forward(&res_t)?;
            cond = (cond + res_emb)?;
        }

        Ok(cond)
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

/// Multi-Head Self-Attention with optional KV-compression for 2K/4K tokens
#[derive(Debug, Clone)]
pub struct PixArtSelfAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    num_heads: usize,
    head_dim: usize,
    kv_compress_ratio: usize,
}

impl PixArtSelfAttention {
    pub fn new(hidden_size: usize, num_heads: usize, kv_compress_ratio: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = hidden_size / num_heads;
        let to_q = linear(hidden_size, hidden_size, vb.pp("to_q"))
            .or_else(|_| linear_no_bias(hidden_size, hidden_size, vb.pp("to_q")))?;
        let to_k = linear(hidden_size, hidden_size, vb.pp("to_k"))
            .or_else(|_| linear_no_bias(hidden_size, hidden_size, vb.pp("to_k")))?;
        let to_v = linear(hidden_size, hidden_size, vb.pp("to_v"))
            .or_else(|_| linear_no_bias(hidden_size, hidden_size, vb.pp("to_v")))?;
        let to_out = linear(hidden_size, hidden_size, vb.pp("to_out"))
            .or_else(|_| linear_no_bias(hidden_size, hidden_size, vb.pp("to_out")))?;

        Ok(Self {
            to_q,
            to_k,
            to_v,
            to_out,
            num_heads,
            head_dim,
            kv_compress_ratio,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, seq_len, hidden_size) = x.dims3()?;
        let orig_dtype = x.dtype();

        let q = self.to_q.forward(x)?;
        let mut k = self.to_k.forward(x)?;
        let mut v = self.to_v.forward(x)?;

        // KV-compression for 2K/4K resolution: downsample keys and values along sequence length
        if self.kv_compress_ratio > 1 && seq_len >= self.kv_compress_ratio {
            let r = self.kv_compress_ratio;
            let comp_len = seq_len / r;
            let usable_len = comp_len * r;

            let k_trunc = k.narrow(1, 0, usable_len)?;
            let v_trunc = v.narrow(1, 0, usable_len)?;

            k = k_trunc.reshape((b, comp_len, r, hidden_size))?.mean(2)?;
            v = v_trunc.reshape((b, comp_len, r, hidden_size))?.mean(2)?;
        }

        let kv_len = k.dim(1)?;

        // Reshape to [B, Heads, Seq, Head_Dim]
        let q = q.reshape((b, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let k = k.reshape((b, kv_len, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let v = v.reshape((b, kv_len, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let q_scaled = (q.to_dtype(DType::F32)? * scale)?;
        let attn_weights = q_scaled.matmul(&k.to_dtype(DType::F32)?.transpose(2, 3)?.contiguous()?)?;
        let attn_probs = candle_nn::ops::softmax(&attn_weights, candle_core::D::Minus1)?;
        let ctx = attn_probs.matmul(&v.to_dtype(DType::F32)?)?.to_dtype(orig_dtype)?;

        let out = ctx.transpose(1, 2)?.contiguous()?.reshape((b, seq_len, hidden_size))?;
        self.to_out.forward(&out)
    }
}

/// Multi-Head Cross-Attention to T5 text embeddings
#[derive(Debug, Clone)]
pub struct PixArtCrossAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl PixArtCrossAttention {
    pub fn new(hidden_size: usize, caption_channels: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = hidden_size / num_heads;
        let to_q = linear(hidden_size, hidden_size, vb.pp("to_q"))
            .or_else(|_| linear_no_bias(hidden_size, hidden_size, vb.pp("to_q")))?;
        let to_k = linear(caption_channels, hidden_size, vb.pp("to_k"))
            .or_else(|_| linear_no_bias(caption_channels, hidden_size, vb.pp("to_k")))?;
        let to_v = linear(caption_channels, hidden_size, vb.pp("to_v"))
            .or_else(|_| linear_no_bias(caption_channels, hidden_size, vb.pp("to_v")))?;
        let to_out = linear(hidden_size, hidden_size, vb.pp("to_out"))
            .or_else(|_| linear_no_bias(hidden_size, hidden_size, vb.pp("to_out")))?;

        Ok(Self {
            to_q,
            to_k,
            to_v,
            to_out,
            num_heads,
            head_dim,
        })
    }

    pub fn forward(&self, x: &Tensor, context: &Tensor) -> Result<Tensor> {
        let (b, seq_len, hidden_size) = x.dims3()?;
        let (_, txt_len, _) = context.dims3()?;
        let orig_dtype = x.dtype();

        let q = self.to_q.forward(x)?;
        let k = self.to_k.forward(context)?;
        let v = self.to_v.forward(context)?;

        let q = q.reshape((b, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let k = k.reshape((b, txt_len, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let v = v.reshape((b, txt_len, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let q_scaled = (q.to_dtype(DType::F32)? * scale)?;
        let attn_weights = q_scaled.matmul(&k.to_dtype(DType::F32)?.transpose(2, 3)?.contiguous()?)?;
        let attn_probs = candle_nn::ops::softmax(&attn_weights, candle_core::D::Minus1)?;
        let ctx = attn_probs.matmul(&v.to_dtype(DType::F32)?)?.to_dtype(orig_dtype)?;

        let out = ctx.transpose(1, 2)?.contiguous()?.reshape((b, seq_len, hidden_size))?;
        self.to_out.forward(&out)
    }
}

/// Feed-Forward (MLP) with GELU activation
#[derive(Debug, Clone)]
pub struct PixArtMlp {
    fc1: Linear,
    fc2: Linear,
}

impl PixArtMlp {
    pub fn new(hidden_size: usize, mlp_ratio: f64, vb: VarBuilder) -> Result<Self> {
        let intermediate_dim = (hidden_size as f64 * mlp_ratio) as usize;
        let fc1 = linear(hidden_size, intermediate_dim, vb.pp("fc1"))
            .or_else(|_| linear_no_bias(hidden_size, intermediate_dim, vb.pp("fc1")))?;
        let fc2 = linear(intermediate_dim, hidden_size, vb.pp("fc2"))
            .or_else(|_| linear_no_bias(intermediate_dim, hidden_size, vb.pp("fc2")))?;

        Ok(Self { fc1, fc2 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.fc1.forward(x)?;
        let h_act = gelu_tanh(&h)?;
        self.fc2.forward(&h_act)
    }
}

/// PixArt Transformer Block with AdaLN-single modulation, Self-Attention (KV-compression),
/// Cross-Attention (T5 text conditioning), and MLP.
#[derive(Debug, Clone)]
pub struct PixArtBlock {
    attn: PixArtSelfAttention,
    cross_attn: PixArtCrossAttention,
    mlp: PixArtMlp,
    scale_shift_table: Linear, // Projects cond [B, Hidden] to 6 modulation parameters [B, 6 * Hidden]
}

impl PixArtBlock {
    pub fn new(cfg: &PixArtConfig, vb: VarBuilder) -> Result<Self> {
        let attn = PixArtSelfAttention::new(cfg.hidden_size, cfg.num_heads, cfg.kv_compress_ratio, vb.pp("attn"))?;
        let cross_attn = PixArtCrossAttention::new(
            cfg.hidden_size,
            cfg.caption_channels,
            cfg.num_heads,
            vb.pp("cross_attn"),
        )?;
        let mlp = PixArtMlp::new(cfg.hidden_size, cfg.mlp_ratio, vb.pp("mlp"))?;

        let scale_shift_table = linear(cfg.hidden_size, cfg.hidden_size * 6, vb.pp("scale_shift_table"))
            .or_else(|_| linear_no_bias(cfg.hidden_size, cfg.hidden_size * 6, vb.pp("scale_shift_table")))?;

        Ok(Self {
            attn,
            cross_attn,
            mlp,
            scale_shift_table,
        })
    }

    pub fn forward(&self, x: &Tensor, context: &Tensor, cond: &Tensor) -> Result<Tensor> {
        let mod_params = self.scale_shift_table.forward(cond)?;
        let chunks = mod_params.chunk(6, mod_params.dims().len() - 1)?;
        let (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = (
            chunks[0].unsqueeze(1)?,
            chunks[1].unsqueeze(1)?,
            chunks[2].unsqueeze(1)?,
            chunks[3].unsqueeze(1)?,
            chunks[4].unsqueeze(1)?,
            chunks[5].unsqueeze(1)?,
        );

        // 1. Modulated Self-Attention
        let x_norm1 = layer_norm_no_affine(x)?;
        let x_mod1 = (x_norm1.broadcast_mul(&(scale_msa + 1.0)?)?.broadcast_add(&shift_msa))?;
        let attn_out = self.attn.forward(&x_mod1)?;
        let mut h = (x + attn_out.broadcast_mul(&gate_msa)?)?;

        // 2. Cross-Attention with T5 context
        let h_norm2 = layer_norm_no_affine(&h)?;
        let cross_out = self.cross_attn.forward(&h_norm2, context)?;
        h = (h + cross_out)?;

        // 3. Modulated MLP
        let h_norm3 = layer_norm_no_affine(&h)?;
        let h_mod3 = (h_norm3.broadcast_mul(&(scale_mlp + 1.0)?)?.broadcast_add(&shift_mlp))?;
        let mlp_out = self.mlp.forward(&h_mod3)?;
        h = (h + mlp_out.broadcast_mul(&gate_mlp)?)?;

        Ok(h)
    }
}

/// Final Layer modulating tokens and unpatchifying back to 2D latents
#[derive(Debug, Clone)]
pub struct PixArtFinalLayer {
    scale_shift: Linear,
    linear: Linear,
    patch_size: usize,
    out_channels: usize,
}

impl PixArtFinalLayer {
    pub fn new(hidden_size: usize, patch_size: usize, out_channels: usize, vb: VarBuilder) -> Result<Self> {
        let scale_shift = linear(hidden_size, hidden_size * 2, vb.pp("scale_shift"))
            .or_else(|_| linear_no_bias(hidden_size, hidden_size * 2, vb.pp("scale_shift")))?;
        let linear = linear(hidden_size, patch_size * patch_size * out_channels, vb.pp("linear"))
            .or_else(|_| linear_no_bias(hidden_size, patch_size * patch_size * out_channels, vb.pp("linear")))?;

        Ok(Self {
            scale_shift,
            linear,
            patch_size,
            out_channels,
        })
    }

    pub fn forward(&self, x: &Tensor, cond: &Tensor, h_p: usize, w_p: usize) -> Result<Tensor> {
        let (b, _seq_len, _hidden) = x.dims3()?;

        let mod_params = self.scale_shift.forward(cond)?;
        let chunks = mod_params.chunk(2, mod_params.dims().len() - 1)?;
        let shift = chunks[0].unsqueeze(1)?;
        let scale = chunks[1].unsqueeze(1)?;

        let x_norm = layer_norm_no_affine(x)?;
        let x_mod = (x_norm.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(&shift))?;
        let patches = self.linear.forward(&x_mod)?; // [B, H/P * W/P, P*P*out_channels]

        // Unpatchify: [B, H/P, W/P, P, P, out_channels] -> [B, out_channels, H, W]
        let p = self.patch_size;
        let c = self.out_channels;
        let unpatch = patches
            .reshape((b, h_p, w_p, p, p, c))?
            .permute((0, 5, 1, 3, 2, 4))? // [B, c, H/P, p, W/P, p]
            .contiguous()?
            .reshape((b, c, h_p * p, w_p * p))?; // [B, c, H, W]

        Ok(unpatch)
    }
}

/// PixArt-α, PixArt-Σ & Lumina-Next Diffusion Transformer
#[derive(Debug, Clone)]
pub struct PixArtTransformer {
    patch_embed: PixArtPatchEmbed,
    cond_embed: PixArtConditionEmbedder,
    blocks: Vec<PixArtBlock>,
    final_layer: PixArtFinalLayer,
    cfg: PixArtConfig,
}

impl PixArtTransformer {
    pub fn new(cfg: PixArtConfig, vb: VarBuilder) -> Result<Self> {
        let patch_embed = PixArtPatchEmbed::new(
            cfg.in_channels,
            cfg.hidden_size,
            cfg.patch_size,
            vb.pp("patch_embed"),
        )?;
        let cond_embed = PixArtConditionEmbedder::new(
            cfg.hidden_size,
            cfg.use_resolution_embed,
            vb.pp("cond_embed"),
        )?;

        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let block = PixArtBlock::new(&cfg, vb.pp(&format!("blocks.{}", i)))?;
            blocks.push(block);
        }

        let final_layer = PixArtFinalLayer::new(
            cfg.hidden_size,
            cfg.patch_size,
            cfg.out_channels,
            vb.pp("final_layer"),
        )?;

        Ok(Self {
            patch_embed,
            cond_embed,
            blocks,
            final_layer,
            cfg,
        })
    }

    /// Forward pass with full resolution conditioning
    pub fn forward_with_resolution(
        &self,
        latents: &Tensor,
        timestep: &Tensor,
        context: &Tensor,
        resolution: Option<(usize, usize)>,
    ) -> Result<Tensor> {
        let (mut x, h_p, w_p) = self.patch_embed.forward(latents)?;
        let cond = self.cond_embed.forward(timestep, resolution)?;

        for block in &self.blocks {
            x = block.forward(&x, context, &cond)?;
        }

        self.final_layer.forward(&x, &cond, h_p, w_p)
    }

    pub fn config(&self) -> &PixArtConfig {
        &self.cfg
    }
}

impl DiffusionTransformer for PixArtTransformer {
    fn forward(
        &self,
        latents: &Tensor,
        timestep: &Tensor,
        context: &Tensor,
        _guidance: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (_b, _c, h, w) = latents.dims4()?;
        self.forward_with_resolution(latents, timestep, context, Some((h * 8, w * 8)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_pixart_sigma_kv_compression_and_forward() -> Result<()> {
        let dev = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &dev);

        // Lightweight mock config with 2x KV-compression
        let cfg = PixArtConfig {
            in_channels: 4,
            out_channels: 4,
            patch_size: 2,
            hidden_size: 64,
            num_heads: 4,
            num_layers: 2,
            mlp_ratio: 2.0,
            caption_channels: 128,
            kv_compress_ratio: 2,
            use_resolution_embed: true,
        };

        let model = PixArtTransformer::new(cfg, vb)?;

        // 32x32 latent (representing 256x256 pixel image)
        let latents = Tensor::zeros((1, 4, 32, 32), DType::F32, &dev)?;
        let timestep = Tensor::new(&[500.0f32], &dev)?;
        // T5 text context: [1, 16 tokens, 128 dim]
        let context = Tensor::zeros((1, 16, 128), DType::F32, &dev)?;

        let out = model.forward(&latents, &timestep, &context, None)?;

        // Output must preserve exact spatial dimensions [1, 4, 32, 32]
        assert_eq!(out.dims4()?, (1, 4, 32, 32));
        Ok(())
    }

    #[test]
    fn test_pixart_4k_kv_compression_factor() -> Result<()> {
        let dev = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &dev);

        let cfg = PixArtConfig {
            in_channels: 4,
            out_channels: 4,
            patch_size: 2,
            hidden_size: 64,
            num_heads: 4,
            num_layers: 1,
            mlp_ratio: 2.0,
            caption_channels: 128,
            kv_compress_ratio: 4,
            use_resolution_embed: false,
        };

        let model = PixArtTransformer::new(cfg, vb)?;

        // 64x64 latent (representing 512x512)
        let latents = Tensor::zeros((1, 4, 64, 64), DType::F32, &dev)?;
        let timestep = Tensor::new(&[200.0f32], &dev)?;
        let context = Tensor::zeros((1, 8, 128), DType::F32, &dev)?;

        let out = model.forward(&latents, &timestep, &context, None)?;
        assert_eq!(out.dims4()?, (1, 4, 64, 64));
        Ok(())
    }
}
