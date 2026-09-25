// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Stable Audio Open DiT (continuous transformer)

//! Stable Audio Open DiT (`StableAudioDiTModel`): 24 layers, 1536-dim, GQA cross-attention,
//! partial RoPE on self-attention, global conditioning prepended as an extra token.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{
    conv1d_no_bias, layer_norm, linear, linear_no_bias, Conv1d, Conv1dConfig, LayerNorm, LayerNormConfig,
    Linear, Module, VarBuilder,
};
use std::f32::consts::PI;
use std::path::Path;

const DIM: usize = 1536;
const HEADS: usize = 24;
const HEAD_DIM: usize = 64;
const CROSS_DIM: usize = 768;
const TIME_PROJ_DIM: usize = 256;
const GLOBAL_DIM: usize = 1536;
const IO: usize = 64;
const NORM_EPS: f64 = 1e-5;
const ROT_DIM: usize = 32; // attention_head_dim / 2

fn layernorm(dim: usize, vb: VarBuilder) -> Result<LayerNorm> {
    Ok(layer_norm(
        dim,
        LayerNormConfig {
            eps: NORM_EPS,
            remove_mean: true,
            affine: true,
        },
        vb,
    )?)
}

/// `StableAudioGaussianFourierProjection` (log=false, flip_sin_to_cos=true).
struct GaussianFourierProjection {
    weight: Tensor, // [128]
}

impl GaussianFourierProjection {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get((TIME_PROJ_DIM / 2,), "weight")?,
        })
    }
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = x.unsqueeze(1)?; // [B,1]
        let w = self.weight.unsqueeze(0)?.to_dtype(x.dtype())?;
        let proj = x.broadcast_mul(&w)?.affine((2.0 * PI) as f64, 0.0)?; // [B,128]
        Tensor::cat(&[&proj.cos()?, &proj.sin()?], 1) // flip_sin_to_cos
    }
}

struct SwiGlu {
    proj: Linear,
    out: Linear,
}

impl SwiGlu {
    fn load(dim: usize, inner: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            proj: linear(dim, inner * 2, vb.pp("net.0.proj"))?,
            out: linear(inner, dim, vb.pp("net.2"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let p = self.proj.forward(x)?;
        let halves = p.chunk(2, candle_core::D::Minus1)?;
        let gated = halves[0].mul(&halves[1].silu()?)?;
        self.out.forward(&gated)
    }
}

struct DitAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    is_cross: bool,
    kv_heads: usize,
}

impl DitAttention {
    fn load(vb: VarBuilder, is_cross: bool, kv_heads: usize) -> Result<Self> {
        let q_in = DIM;
        let kv_in = if is_cross { CROSS_DIM } else { DIM };
        Ok(Self {
            to_q: linear_no_bias(q_in, HEADS * HEAD_DIM, vb.pp("to_q"))?,
            to_k: linear_no_bias(kv_in, kv_heads * HEAD_DIM, vb.pp("to_k"))?,
            to_v: linear_no_bias(kv_in, kv_heads * HEAD_DIM, vb.pp("to_v"))?,
            to_out: linear_no_bias(HEADS * HEAD_DIM, DIM, vb.pp("to_out.0"))?,
            is_cross,
            kv_heads,
        })
    }

    fn rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
        let (b, h, l, hd) = x.dims4()?;
        let rot = cos.dim(1)?;
        let half = rot / 2;
        let x_rot = x.narrow(3, 0, rot)?;
        let x_real = x_rot.narrow(3, 0, half)?;
        let x_imag = x_rot.narrow(3, half, half)?.affine(-1.0, 0.0)?;
        let x_rotated = Tensor::cat(&[&x_imag, &x_real], 3)?;
        let cos_b = cos.unsqueeze(0)?.unsqueeze(0)?; // [1,1,L,rot]
        let sin_b = sin.unsqueeze(0)?.unsqueeze(0)?;
        let out = x_rot
            .broadcast_mul(&cos_b)?
            .broadcast_add(&x_rotated.broadcast_mul(&sin_b)?)?;
        let x_un = x.narrow(3, rot, hd - rot)?;
        let _ = (b, h);
        Tensor::cat(&[&out, &x_un], 3)
    }

    fn forward(
        &self,
        hidden: &Tensor,
        enc: Option<&Tensor>,
        rotary: Option<&(Tensor, Tensor)>,
    ) -> candle_core::Result<Tensor> {
        let (b, l, _) = hidden.dims3()?;
        let q = self
            .to_q
            .forward(hidden)?
            .reshape((b, l, HEADS, HEAD_DIM))?
            .transpose(1, 2)?;
        let src = enc.unwrap_or(hidden);
        let (bs, ls, _) = src.dims3()?;
        let k = self
            .to_k
            .forward(src)?
            .reshape((bs, ls, self.kv_heads, HEAD_DIM))?
            .transpose(1, 2)?;
        let v = self
            .to_v
            .forward(src)?
            .reshape((bs, ls, self.kv_heads, HEAD_DIM))?
            .transpose(1, 2)?;

        // GQA: repeat kv heads to match query heads.
        let (k, v) = if self.kv_heads != HEADS {
            let rep = HEADS / self.kv_heads;
            let k = k
                .unsqueeze(2)?
                .broadcast_as((bs, self.kv_heads, rep, ls, HEAD_DIM))?
                .reshape((bs, self.kv_heads * rep, ls, HEAD_DIM))?;
            let v = v
                .unsqueeze(2)?
                .broadcast_as((bs, self.kv_heads, rep, ls, HEAD_DIM))?
                .reshape((bs, self.kv_heads * rep, ls, HEAD_DIM))?;
            (k, v)
        } else {
            (k, v)
        };

        let (q, k) = if !self.is_cross {
            if let Some((cos, sin)) = rotary {
                (Self::rope(&q, cos, sin)?, Self::rope(&k, cos, sin)?)
            } else {
                (q, k)
            }
        } else {
            (q, k)
        };

        let scale = 1.0 / (HEAD_DIM as f64).sqrt();
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let scores = q.matmul(&k.transpose(2, 3)?.contiguous()?)?.affine(scale, 0.0)?;
        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = attn.matmul(&v.contiguous()?)?; // [B,H,L,HEAD_DIM]
        let ctx = ctx.transpose(1, 2)?.reshape((b, l, HEADS * HEAD_DIM))?;
        self.to_out.forward(&ctx)
    }
}

struct DitBlock {
    norm1: LayerNorm,
    attn1: DitAttention,
    norm2: LayerNorm,
    attn2: DitAttention,
    norm3: LayerNorm,
    ff: SwiGlu,
}

impl DitBlock {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm1: layernorm(DIM, vb.pp("norm1"))?,
            attn1: DitAttention::load(vb.pp("attn1"), false, HEADS)?,
            norm2: layernorm(DIM, vb.pp("norm2"))?,
            attn2: DitAttention::load(vb.pp("attn2"), true, 12)?,
            norm3: layernorm(DIM, vb.pp("norm3"))?,
            ff: SwiGlu::load(DIM, DIM * 4, vb.pp("ff"))?,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        enc: &Tensor,
        rotary: &(Tensor, Tensor),
    ) -> candle_core::Result<Tensor> {
        let h = (hidden + &self.attn1.forward(&self.norm1.forward(hidden)?, None, Some(rotary))?)?;
        let h = (&h + &self.attn2.forward(&self.norm2.forward(&h)?, Some(enc), None)?)?;
        let h = (&h + &self.ff.forward(&self.norm3.forward(&h)?)?)?;
        Ok(h)
    }
}

/// 1D RoPE (`use_real=True, repeat_interleave_real=False`), `dim = 32`, `theta = 10000`.
pub fn rotary_embed(seq_len: usize, device: &Device, dtype: DType) -> candle_core::Result<(Tensor, Tensor)> {
    let half = ROT_DIM / 2;
    let inv: Vec<f32> = (0..half)
        .map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / ROT_DIM as f32))
        .collect();
    let mut cos = vec![0f32; seq_len * ROT_DIM];
    let mut sin = vec![0f32; seq_len * ROT_DIM];
    for p in 0..seq_len {
        for i in 0..half {
            let f = p as f32 * inv[i];
            cos[p * ROT_DIM + i] = f.cos();
            cos[p * ROT_DIM + half + i] = f.cos();
            sin[p * ROT_DIM + i] = f.sin();
            sin[p * ROT_DIM + half + i] = f.sin();
        }
    }
    Ok((
        Tensor::from_vec(cos, (seq_len, ROT_DIM), device)?.to_dtype(dtype)?,
        Tensor::from_vec(sin, (seq_len, ROT_DIM), device)?.to_dtype(dtype)?,
    ))
}

pub struct StableAudioDit {
    time_proj: GaussianFourierProjection,
    timestep_proj: (Linear, Linear),
    global_proj: (Linear, Linear),
    cross_proj: (Linear, Linear),
    preprocess_conv: Conv1d,
    proj_in: Linear,
    blocks: Vec<DitBlock>,
    proj_out: Linear,
    postprocess_conv: Conv1d,
}

impl StableAudioDit {
    pub fn from_safetensors<P: AsRef<Path>>(path: P, device: &Device, dtype: DType) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path.as_ref()], dtype, device)? };
        let conv_cfg = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let time_proj = GaussianFourierProjection::load(vb.pp("time_proj"))?;
        let timestep_proj = (
            linear(TIME_PROJ_DIM, DIM, vb.pp("timestep_proj.0"))?,
            linear(DIM, DIM, vb.pp("timestep_proj.2"))?,
        );
        let global_proj = (
            linear_no_bias(GLOBAL_DIM, DIM, vb.pp("global_proj.0"))?,
            linear_no_bias(DIM, DIM, vb.pp("global_proj.2"))?,
        );
        let cross_proj = (
            linear_no_bias(CROSS_DIM, CROSS_DIM, vb.pp("cross_attention_proj.0"))?,
            linear_no_bias(CROSS_DIM, CROSS_DIM, vb.pp("cross_attention_proj.2"))?,
        );
        let preprocess_conv = conv1d_no_bias(IO, IO, 1, conv_cfg, vb.pp("preprocess_conv"))?;
        let proj_in = linear_no_bias(IO, DIM, vb.pp("proj_in"))?;
        let mut blocks = Vec::with_capacity(24);
        for i in 0..24 {
            blocks.push(DitBlock::load(vb.pp(format!("transformer_blocks.{i}")))?);
        }
        let proj_out = linear_no_bias(DIM, IO, vb.pp("proj_out"))?;
        let postprocess_conv = conv1d_no_bias(IO, IO, 1, conv_cfg, vb.pp("postprocess_conv"))?;
        Ok(Self {
            time_proj,
            timestep_proj,
            global_proj,
            cross_proj,
            preprocess_conv,
            proj_in,
            blocks,
            proj_out,
            postprocess_conv,
        })
    }

    /// `hidden_states` `[B, 64, L]`, `timestep` `[B]`, `encoder_hidden_states` `[B, S, 768]`,
    /// `global_hidden_states` `[B, Sg, 1536]` → velocity `[B, 64, L]`.
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        timestep: &Tensor,
        encoder_hidden_states: &Tensor,
        global_hidden_states: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let enc = self.cross_proj.1.forward(&self.cross_proj.0.forward(encoder_hidden_states)?.silu()?)?;
        let glob = self.global_proj.1.forward(&self.global_proj.0.forward(global_hidden_states)?.silu()?)?;
        let tfeat = self.time_proj.forward(timestep)?;
        let tmid = self.timestep_proj.0.forward(&tfeat)?.silu()?;
        let tproj = self.timestep_proj.1.forward(&tmid)?; // [B,1536]
        let glob = glob.broadcast_add(&tproj.unsqueeze(1)?)?; // [B,Sg,1536]

        let pre = self.preprocess_conv.forward(hidden_states)?;
        let hidden = (hidden_states + pre)?.transpose(1, 2)?; // [B,L,64]
        let hidden = self.proj_in.forward(&hidden)?; // [B,L,1536]

        let seq = hidden.dim(1)?;
        let hidden = Tensor::cat(&[&glob, &hidden], 1)?; // [B, Sg+L, 1536]
        let rot_len = seq + glob.dim(1)?;
        let rotary = rotary_embed(rot_len, hidden.device(), hidden.dtype())?;

        let mut h = hidden;
        for blk in &self.blocks {
            h = blk.forward(&h, &enc, &rotary)?;
        }
        let h = self.proj_out.forward(&h)?; // [B, Sg+L, 64]
        let h = h.transpose(1, 2)?; // [B,64,Sg+L]
        // drop the prepended global token(s)
        let h = h.narrow(2, glob.dim(1)?, seq)?;
        let post = self.postprocess_conv.forward(&h)?;
        h + post
    }

    /// Verbose: `(enc, glob, hidden_after_proj_in, after_block0, final_out)`.
    #[allow(clippy::type_complexity)]
    pub fn forward_verbose(
        &self,
        hidden_states: &Tensor,
        timestep: &Tensor,
        encoder_hidden_states: &Tensor,
        global_hidden_states: &Tensor,
    ) -> candle_core::Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let enc = self.cross_proj.1.forward(&self.cross_proj.0.forward(encoder_hidden_states)?.silu()?)?;
        let glob = self.global_proj.1.forward(&self.global_proj.0.forward(global_hidden_states)?.silu()?)?;
        let tfeat = self.time_proj.forward(timestep)?;
        let tmid = self.timestep_proj.0.forward(&tfeat)?.silu()?;
        let tproj = self.timestep_proj.1.forward(&tmid)?;
        let glob = glob.broadcast_add(&tproj.unsqueeze(1)?)?;

        let pre = self.preprocess_conv.forward(hidden_states)?;
        let hidden = (hidden_states + pre)?.transpose(1, 2)?;
        let xin = self.proj_in.forward(&hidden)?;
        let seq = xin.dim(1)?;
        let seq_glob = glob.dim(1)?;
        let mut h = Tensor::cat(&[&glob, &xin], 1)?;
        let rotary = rotary_embed(seq + seq_glob, h.device(), h.dtype())?;
        let mut after0 = None;
        for (i, blk) in self.blocks.iter().enumerate() {
            h = blk.forward(&h, &enc, &rotary)?;
            if i == 0 {
                after0 = Some(h.clone());
            }
        }
        let hh = self.proj_out.forward(&h)?;
        let hh = hh.transpose(1, 2)?.narrow(2, seq_glob, seq)?;
        let post = self.postprocess_conv.forward(&hh)?;
        Ok((enc, glob, xin, after0.unwrap(), (hh + post)?))
    }
}

