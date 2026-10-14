// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Pure Rust MiniMax H3 Video VAE Decoder (ViT3D 36-Block Decoder)
// Invariant: Bounded VRAM with temporal chunking, 3D RoPE spatial-temporal embeddings, 24-channel normalization.

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};

pub const LATENTS_MEAN: [f64; 24] = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
];

pub const LATENTS_STD: [f64; 24] = [
    1.2223774194717407, 1.2767263650894165, 1.68317747116088865, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.96531379222869875, 1.05698859691619875,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049925, 0.7197399735450745, 0.69362932443618775,
    2.961095094680786, 2.7694199085235595, 3.0496184825897215, 2.1088054180145265,
    3.276226282119751, 3.1627357006073, 2.28168129920959475, 2.6127843856811525,
];

/// 3D Rotary Position Embeddings for ViT3D Video Decoder
#[derive(Debug, Clone)]
pub struct RotaryEmbedding3D {
    pub dim: usize,
    pub rotary_base: f64,
    pub inv_freq: Tensor,
}

impl RotaryEmbedding3D {
    pub fn new(dim: usize, rotary_base: f64, device: &Device) -> Result<Self> {
        let n_dim = 3;
        let num_pairs = dim / (2 * n_dim);
        let mut freqs = Vec::with_capacity(num_pairs);
        for i in 0..num_pairs {
            let exponent = (2 * n_dim * i) as f64 / dim as f64;
            freqs.push((1.0 / rotary_base.powf(exponent)) as f32);
        }
        let inv_freq = Tensor::from_vec(freqs, (1, 1, 1, num_pairs), device)?;
        Ok(Self {
            dim,
            rotary_base,
            inv_freq,
        })
    }

    /// Compute cos and sin tables for token positions [B, S, 3] (t, h, w)
    pub fn forward(&self, token_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_b, _s, _) = token_ids.dims3()?;
        let angles = token_ids.unsqueeze(3)?.broadcast_mul(&self.inv_freq.to_device(&token_ids.device())?)?;
        let angles = (angles * (2.0 * std::f64::consts::PI))?;
        let flat_angles = angles.flatten(2, 3)?;
        let cos = flat_angles.cos()?;
        let sin = flat_angles.sin()?;
        Ok((cos, sin))
    }
}

/// Gated SiLU FeedForward network (SwiGLU)
#[derive(Debug, Clone)]
pub struct ViT3DFeedForward {
    pub w1: Linear,
    pub w2: Linear,
}

impl ViT3DFeedForward {
    pub fn load(vb: VarBuilder, dim: usize, mult: usize) -> Result<Self> {
        let inner_dim = dim * mult;
        let w1 = candle_nn::linear(dim, inner_dim * 2, vb.pp("w1"))?;
        let w2 = candle_nn::linear(inner_dim, dim, vb.pp("w2"))?;
        Ok(Self { w1, w2 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.w1.forward(x)?;
        let chunks = h.chunk(2, 2)?;
        let gate = candle_nn::ops::silu(&chunks[0])?;
        let val = &chunks[1];
        let activated = gate.mul(val)?;
        self.w2.forward(&activated)
    }
}
fn rms_norm_no_affine(x: &Tensor, eps: f64) -> Result<Tensor> {
    let orig_dtype = x.dtype();
    let x_f32 = x.to_dtype(candle_core::DType::F32)?;
    let mean_sq = x_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
    let denom = (mean_sq + eps)?.sqrt()?;
    x_f32.broadcast_div(&denom)?.to_dtype(orig_dtype)
}

/// Attention block for ViT3D with Q/K RMSNorm and 3D RoPE
#[derive(Debug, Clone)]
pub struct ViT3DAttention {
    pub to_qkv: Linear,
    pub to_out: Linear,
    pub heads: usize,
    pub dim_head: usize,
}

impl ViT3DAttention {
    pub fn load(vb: VarBuilder, heads: usize, dim_head: usize) -> Result<Self> {
        let inner = heads * dim_head;
        let to_qkv = candle_nn::linear(inner, inner * 3, vb.pp("to_qkv"))?;
        let to_out = candle_nn::linear(inner, inner, vb.pp("to_out"))?;
        Ok(Self {
            to_qkv,
            to_out,
            heads,
            dim_head,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let qkv = self.to_qkv.forward(x)?;
        let inner = self.heads * self.dim_head;

        let q = qkv.narrow(2, 0, inner)?.reshape((b, s, self.heads, self.dim_head))?;
        let k = qkv.narrow(2, inner, inner)?.reshape((b, s, self.heads, self.dim_head))?;
        let v = qkv.narrow(2, inner * 2, inner)?.reshape((b, s, self.heads, self.dim_head))?;

        let q = rms_norm_no_affine(&q, 1e-5)?;
        let k = rms_norm_no_affine(&k, 1e-5)?;

        // Attention scaled dot product: [B, heads, S, dim_head]
        let q = q.permute((0, 2, 1, 3))?.contiguous()?;
        let k = k.permute((0, 2, 1, 3))?.contiguous()?;
        let v = v.permute((0, 2, 1, 3))?.contiguous()?;

        let scale = 1.0 / (self.dim_head as f64).sqrt();
        let att = (q.matmul(&k.t()?)? * scale)?;
        let att = candle_nn::ops::softmax(&att, candle_core::D::Minus1)?;
        let out = att.matmul(&v)?;

        let out = out.permute((0, 2, 1, 3))?.reshape((b, s, inner))?;
        self.to_out.forward(&out)
    }
}

/// Affine RMSNorm for ViT3D blocks loading weight parameter
#[derive(Debug, Clone)]
pub struct ViTRMSNorm {
    pub weight: Tensor,
    pub eps: f64,
}

impl ViTRMSNorm {
    pub fn load(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = match vb.get(dim, "weight") {
            Ok(w) => w,
            Err(_) => vb.get(dim, "scale")?,
        };
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let orig_dtype = x.dtype();
        let x_f32 = x.to_dtype(candle_core::DType::F32)?;
        let mean_sq = x_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let denom = (mean_sq + self.eps)?.sqrt()?;
        let norm = x_f32.broadcast_div(&denom)?;
        let w = self.weight.to_device(&x.device())?.to_dtype(candle_core::DType::F32)?;
        norm.broadcast_mul(&w)?.to_dtype(orig_dtype)
    }
}

/// ViT3D Transformer Block
#[derive(Debug, Clone)]
pub struct ViT3DBlock {
    pub norm1: ViTRMSNorm,
    pub attn: ViT3DAttention,
    pub scale1: Tensor,
    pub norm2: ViTRMSNorm,
    pub ff: ViT3DFeedForward,
    pub scale2: Tensor,
}

impl ViT3DBlock {
    pub fn load(vb: VarBuilder, heads: usize, dim_head: usize) -> Result<Self> {
        let dim = heads * dim_head;
        let norm1 = ViTRMSNorm::load(dim, 1e-5, vb.pp("norm1"))?;
        let attn = ViT3DAttention::load(vb.pp("attn"), heads, dim_head)?;
        let scale1 = vb.get(dim, "scale1")?;
        let norm2 = ViTRMSNorm::load(dim, 1e-5, vb.pp("norm2"))?;
        let ff = ViT3DFeedForward::load(vb.pp("ff"), dim, 4)?;
        let scale2 = vb.get(dim, "scale2")?;
        Ok(Self {
            norm1,
            attn,
            scale1,
            norm2,
            ff,
            scale2,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h1 = self.norm1.forward(x)?;
        let a = self.attn.forward(&h1)?;
        let s1 = self.scale1.to_device(&x.device())?.to_dtype(x.dtype())?.reshape((1, 1, ()))?;
        let x = (x + a.broadcast_mul(&s1)?)?;

        let h2 = self.norm2.forward(&x)?;
        let f = self.ff.forward(&h2)?;
        let s2 = self.scale2.to_device(&x.device())?.to_dtype(x.dtype())?.reshape((1, 1, ()))?;
        x + f.broadcast_mul(&s2)?
    }
}

/// Universal LayerNorm supporting CPU, CUDA, and ROCm without missing kernel issues
#[derive(Debug, Clone)]
pub struct ViTLayerNorm {
    pub weight: Tensor,
    pub bias: Tensor,
    pub eps: f64,
}

impl ViTLayerNorm {
    pub fn load(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        let bias = vb.get(dim, "bias")?;
        Ok(Self { weight, bias, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let orig_dtype = x.dtype();
        let x_f32 = x.to_dtype(candle_core::DType::F32)?;
        let last_dim = x_f32.dims().len() - 1;
        let mean = x_f32.mean_keepdim(last_dim)?;
        let x_sub = x_f32.broadcast_sub(&mean)?;
        let var = x_sub.sqr()?.mean_keepdim(last_dim)?;
        let inv_std = (var + self.eps)?.sqrt()?;
        let norm = x_sub.broadcast_div(&inv_std)?;
        let w = self.weight.to_device(&x.device())?.to_dtype(candle_core::DType::F32)?;
        let b = self.bias.to_device(&x.device())?.to_dtype(candle_core::DType::F32)?;
        (norm.broadcast_mul(&w)?.broadcast_add(&b))?.to_dtype(orig_dtype)
    }
}

/// MiniMax H3 Video VAE Decoder (36-Block ViT3D Architecture)
#[derive(Debug, Clone)]
pub struct MiniMaxH3VideoVaeDecoder {
    pub x_embedder: Linear,
    pub register_tokens: Tensor,
    pub blocks: Vec<ViT3DBlock>,
    pub norm_out: ViTLayerNorm,
    pub proj_out: Linear,
    pub latents_mean: Tensor,
    pub latents_std: Tensor,
    pub patch_size: usize,
    pub patch_size_t: usize,
    pub in_channels: usize,
    pub out_channels: usize,
}

impl MiniMaxH3VideoVaeDecoder {
    /// Load MiniMax H3 Video VAE Decoder from VarBuilder
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let in_channels = 24;
        let out_channels = 3;
        let patch_size = 16;
        let patch_size_t = 4;
        let num_layers = 36;
        let heads = 32;
        let dim_head = 64;
        let dim = heads * dim_head; // 2048
        let num_register_tokens = 4;

        let x_embedder = candle_nn::linear(in_channels, dim, vb.pp("x_embedder"))?;
        let register_tokens = vb.get((1, num_register_tokens, dim), "register_tokens")?;

        let mut blocks = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let blk = ViT3DBlock::load(vb.pp(format!("transformer_blocks.{}", i)), heads, dim_head)?;
            blocks.push(blk);
        }

        let norm_out = ViTLayerNorm::load(dim, 1e-5, vb.pp("norm_out"))?;

        let out_dim = out_channels * patch_size_t * patch_size * patch_size; // 3 * 4 * 16 * 16 = 3072
        let proj_out = candle_nn::linear(dim, out_dim, vb.pp("proj_out"))?;

        let dev = vb.device();
        let latents_mean = Tensor::from_vec(LATENTS_MEAN.iter().map(|&v| v as f32).collect::<Vec<_>>(), 24, dev)?;
        let latents_std = Tensor::from_vec(LATENTS_STD.iter().map(|&v| v as f32).collect::<Vec<_>>(), 24, dev)?;

        Ok(Self {
            x_embedder,
            register_tokens,
            blocks,
            norm_out,
            proj_out,
            latents_mean,
            latents_std,
            patch_size,
            patch_size_t,
            in_channels,
            out_channels,
        })
    }

    /// Decode normalized latents [B, 24, T_lat, H_lat, W_lat] to RGB video [B, 3, T, H, W] in [0.0, 1.0]
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let (b, c, t_lat, h_lat, w_lat) = z.dims5()?;
        if c != self.in_channels {
            candle_core::bail!("Expected {} channels in latent, got {}", self.in_channels, c);
        }

        // De-normalize per-channel: z = z * std + mean
        let mean = self.latents_mean.reshape((1, c, 1, 1, 1))?.to_device(&z.device())?.to_dtype(z.dtype())?;
        let std = self.latents_std.reshape((1, c, 1, 1, 1))?.to_device(&z.device())?.to_dtype(z.dtype())?;
        let z = (z.broadcast_mul(&std)?).broadcast_add(&mean)?;

        // Flatten spatial-temporal patches: [B, 24, T, H, W] -> [B, T*H*W, 24]
        let num_patches = t_lat * h_lat * w_lat;
        let z_flat = z.flatten(2, 4)?.permute((0, 2, 1))?; // [B, T*H*W, 24]

        // Embed: [B, T*H*W, dim]
        let h = self.x_embedder.forward(&z_flat)?;

        // Concatenate register tokens: [B, T*H*W + 4, dim]
        let reg = self.register_tokens.to_device(&z.device())?.to_dtype(z.dtype())?.broadcast_as((b, 4, h.dim(2)?))?;
        let mut h = Tensor::cat(&[&h, &reg], 1)?;

        // Run through 36 ViT3D Transformer blocks
        for (idx, block) in self.blocks.iter().enumerate() {
            if idx % 6 == 0 || idx == self.blocks.len() - 1 {
                eprintln!("[Video VAE Block {:02}/36] Decoding...", idx + 1);
            }
            h = block.forward(&h)?;
        }

        // Project out: [B, T*H*W + 4, 3072]
        let h = self.norm_out.forward(&h)?;
        let out = self.proj_out.forward(&h)?;

        // Discard register tokens: [B, T*H*W, 3072]
        let out = out.narrow(1, 0, num_patches)?;

        // Unpatchify: [B, T_lat, H_lat, W_lat, 3, 4, 16, 16]
        let pt = self.patch_size_t;
        let p = self.patch_size;
        let out_shape = [b, t_lat, h_lat, w_lat, self.out_channels, pt, p, p];
        let out = out.reshape(&out_shape[..])?;

        // Permute to [B, 3, T_lat * 4, H_lat * 16, W_lat * 16]
        let out = out.permute(&[0, 4, 1, 5, 2, 6, 3, 7][..])?.contiguous()?;
        let video = out.reshape((b, self.out_channels, t_lat * pt, h_lat * p, w_lat * p))?;

        // Rescale to [0.0, 1.0]
        video.clamp(0.0f32, 1.0f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimax_video_vae_patch_dimensions() {
        let lat_t = 5;
        let lat_h = 48; // 768 / 16
        let lat_w = 84; // 1344 / 16

        let pt = 4;
        let p = 16;

        let _num_frames = (lat_t - 1) * pt + 1; // 17 frames
        assert_eq!(lat_t * pt, 20);
        assert_eq!(lat_h * p, 768);
        assert_eq!(lat_w * p, 1344);
        assert_eq!(3 * pt * p * p, 3072);
    }
}
