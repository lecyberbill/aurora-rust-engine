// [WFGY] Zone: SAFE | λ: 0.30 | Fallbacks: 0 | Action: Pure Rust Multi-Head CrossAttention with exact 4D matrix transpose & GeGLU

use candle_core::{Result, Tensor};
use candle_nn::{conv2d, group_norm, linear, linear_no_bias, Conv2d, Conv2dConfig, GroupNorm, Linear, Module, VarBuilder};

fn linear_flexible(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    if vb.contains_tensor("bias") {
        linear(in_dim, out_dim, vb)
    } else {
        linear_no_bias(in_dim, out_dim, vb)
    }
}

/// Pure Rust Multi-Head Attention supporting Self-Attention & Cross-Attention
#[derive(Debug)]
pub struct CrossAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Vec<Linear>,
    heads: usize,
    head_dim: usize,
    scale: f64,
}

impl CrossAttention {
    pub fn new(vb: VarBuilder, query_dim: usize, context_dim: Option<usize>, heads: usize, dim_head: usize) -> Result<Self> {
        let inner_dim = dim_head * heads;
        let context_dim = context_dim.unwrap_or(query_dim);
        let scale = 1.0 / (dim_head as f64).sqrt();

        let to_q = linear_flexible(query_dim, inner_dim, vb.pp("to_q"))?;
        let to_k = linear_flexible(context_dim, inner_dim, vb.pp("to_k"))?;
        let to_v = linear_flexible(context_dim, inner_dim, vb.pp("to_v"))?;
        let to_out_0 = linear_flexible(inner_dim, query_dim, vb.pp("to_out.0"))?;

        Ok(Self {
            to_q,
            to_k,
            to_v,
            to_out: vec![to_out_0],
            heads,
            head_dim: dim_head,
            scale,
        })
    }

    pub fn forward(&self, xs: &Tensor, context: Option<&Tensor>) -> Result<Tensor> {
        let (b_size, seq_len, _) = xs.dims3()?;
        let context = context.unwrap_or(xs);
        let (_, context_len, _) = context.dims3()?;

        let q = self.to_q.forward(xs)?;
        let k = self.to_k.forward(context)?;
        let v = self.to_v.forward(context)?;

        let q = q.reshape((b_size, seq_len, self.heads, self.head_dim))?;
        let k = k.reshape((b_size, context_len, self.heads, self.head_dim))?;
        let v = v.reshape((b_size, context_len, self.heads, self.head_dim))?;

        #[cfg(feature = "flash-attn")]
        {
            if q.device().is_cuda() && (q.dtype() == candle_core::DType::F16 || q.dtype() == candle_core::DType::BF16) {
                let q_c = q.contiguous()?;
                let k_c = k.contiguous()?;
                let v_c = v.contiguous()?;
                let out = candle_flash_attn::flash_attn(&q_c, &k_c, &v_c, self.scale as f32, false)?;
                let out = out.reshape((b_size, seq_len, self.heads * self.head_dim))?;
                return self.to_out[0].forward(&out);
            }
        }

        // Standard Attention fallback (cuBLAS GEMM)
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        // Exact 4D matrix transpose on spatial dims [B, H, head_dim, context_len]
        let k_t = k.transpose(2, 3)?.contiguous()?;

        // Scaled dot-product attention: pre-scale Q to avoid scaling the huge [B, H, seq_len, context_len] matrix
        let q_scaled = (q * self.scale)?;
        let attn_scores = q_scaled.matmul(&k_t)?;
        let attn_probs = candle_nn::ops::softmax_last_dim(&attn_scores)?;

        // Attention output: [b_size, heads, seq_len, head_dim]
        let out = attn_probs.matmul(&v)?;

        // Reshape back to [b_size, seq_len, inner_dim]
        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b_size, seq_len, self.heads * self.head_dim))?;

        self.to_out[0].forward(&out)
    }
}

/// GeGLU / Standard Feed-Forward module
#[derive(Debug)]
pub struct FeedForward {
    net_0_proj: Linear,
    net_2: Linear,
    is_geglu: bool,
}

impl FeedForward {
    pub fn new(vb: VarBuilder, dim: usize, mult: usize) -> Result<Self> {
        let inner_dim = dim * mult;
        let is_geglu = vb.contains_tensor("net.0.proj.weight");
        let net_0_proj = if is_geglu {
            linear_flexible(dim, inner_dim * 2, vb.pp("net.0.proj"))?
        } else {
            linear_flexible(dim, inner_dim, vb.pp("net.0"))?
        };
        let net_2 = linear_flexible(inner_dim, dim, vb.pp("net.2"))?;

        Ok(Self {
            net_0_proj,
            net_2,
            is_geglu,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let hidden = self.net_0_proj.forward(xs)?;
        let act = if self.is_geglu {
            let parts = hidden.chunk(2, candle_core::D::Minus1)?;
            let gate = parts[1].gelu_erf()?;
            parts[0].mul(&gate)?
        } else {
            hidden.gelu_erf()?
        };
        self.net_2.forward(&act)
    }
}

/// Basic Transformer Block: Self-Attention + Cross-Attention + Feed-Forward
#[derive(Debug)]
pub struct BasicTransformerBlock {
    attn1: CrossAttention,
    attn2: CrossAttention,
    ff: FeedForward,
    norm1: candle_nn::LayerNorm,
    norm2: candle_nn::LayerNorm,
    norm3: candle_nn::LayerNorm,
}

impl BasicTransformerBlock {
    pub fn new(vb: VarBuilder, dim: usize, num_heads: usize, dim_head: usize, context_dim: Option<usize>) -> Result<Self> {
        let norm_cfg = candle_nn::LayerNormConfig { eps: 1e-5, remove_mean: true, affine: true };
        let norm1 = candle_nn::layer_norm(dim, norm_cfg, vb.pp("norm1"))?;
        let attn1 = CrossAttention::new(vb.pp("attn1"), dim, None, num_heads, dim_head)?;
        let norm2 = candle_nn::layer_norm(dim, norm_cfg, vb.pp("norm2"))?;
        let attn2 = CrossAttention::new(vb.pp("attn2"), dim, context_dim, num_heads, dim_head)?;
        let norm3 = candle_nn::layer_norm(dim, norm_cfg, vb.pp("norm3"))?;
        let ff = FeedForward::new(vb.pp("ff"), dim, 4)?;

        Ok(Self {
            attn1,
            attn2,
            ff,
            norm1,
            norm2,
            norm3,
        })
    }

    pub fn forward(&self, xs: &Tensor, context: Option<&Tensor>) -> Result<Tensor> {
        let norm_xs = self.norm1.forward(xs)?;
        let attn1_out = self.attn1.forward(&norm_xs, None)?;
        let xs = (xs + attn1_out)?;

        let norm_xs = self.norm2.forward(&xs)?;
        let attn2_out = self.attn2.forward(&norm_xs, context)?;
        let xs = (xs + attn2_out)?;

        let norm_xs = self.norm3.forward(&xs)?;
        let ff_out = self.ff.forward(&norm_xs)?;
        xs + ff_out
    }
}

/// Spatial Transformer: 2D feature map -> Transformer Blocks -> 2D feature map
#[derive(Debug)]
pub enum ProjIn {
    Conv(Conv2d),
    Lin(Linear),
}

#[derive(Debug)]
pub enum ProjOut {
    Conv(Conv2d),
    Lin(Linear),
}

#[derive(Debug)]
pub struct SpatialTransformer {
    norm: GroupNorm,
    proj_in: ProjIn,
    transformer_blocks: Vec<BasicTransformerBlock>,
    proj_out: ProjOut,
}

impl SpatialTransformer {
    pub fn new(
        vb: VarBuilder,
        in_channels: usize,
        num_heads: usize,
        dim_head: usize,
        depth: usize,
        context_dim: Option<usize>,
        use_linear_projection: bool,
    ) -> Result<Self> {
        let inner_dim = num_heads * dim_head;
        let norm = group_norm(32, in_channels, 1e-6, vb.pp("norm"))?;

        let proj_in = if use_linear_projection {
            ProjIn::Lin(linear_flexible(in_channels, inner_dim, vb.pp("proj_in"))?)
        } else {
            ProjIn::Conv(conv2d(in_channels, inner_dim, 1, Conv2dConfig::default(), vb.pp("proj_in"))?)
        };

        let mut transformer_blocks = Vec::with_capacity(depth);
        let blocks_vb = vb.pp("transformer_blocks");
        for i in 0..depth {
            let block = BasicTransformerBlock::new(
                blocks_vb.pp(i),
                inner_dim,
                num_heads,
                dim_head,
                context_dim,
            )?;
            transformer_blocks.push(block);
        }

        let proj_out = if use_linear_projection {
            ProjOut::Lin(linear_flexible(inner_dim, in_channels, vb.pp("proj_out"))?)
        } else {
            ProjOut::Conv(conv2d(inner_dim, in_channels, 1, Conv2dConfig::default(), vb.pp("proj_out"))?)
        };

        Ok(Self {
            norm,
            proj_in,
            transformer_blocks,
            proj_out,
        })
    }

    pub fn forward(&self, xs: &Tensor, context: Option<&Tensor>) -> Result<Tensor> {
        let (b_size, in_channels, h, w) = xs.dims4()?;
        let residual = xs;

        let norm_xs = self.norm.forward(xs)?;
        let proj_in = match &self.proj_in {
            ProjIn::Conv(c) => {
                let p = c.forward(&norm_xs)?;
                let (_, c_dim, _, _) = p.dims4()?;
                p.reshape((b_size, c_dim, h * w))?.transpose(1, 2)?.contiguous()?
            }
            ProjIn::Lin(l) => {
                let flat = norm_xs.reshape((b_size, in_channels, h * w))?.transpose(1, 2)?.contiguous()?;
                l.forward(&flat)?
            }
        };

        let mut cur = proj_in;
        for block in &self.transformer_blocks {
            cur = block.forward(&cur, context)?;
        }

        let proj_out = match &self.proj_out {
            ProjOut::Conv(c) => {
                let (_, _, c_dim) = cur.dims3()?;
                let p = cur.transpose(1, 2)?.contiguous()?.reshape((b_size, c_dim, h, w))?;
                c.forward(&p)?
            }
            ProjOut::Lin(l) => {
                let p = l.forward(&cur)?;
                p.transpose(1, 2)?.contiguous()?.reshape((b_size, in_channels, h, w))?
            }
        };

        residual + proj_out
    }
}
