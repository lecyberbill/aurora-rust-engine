// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Moonshine STT (encoder-decoder) pure Rust

//! Moonshine tiny/base speech-to-text: conv audio frontend + 6-layer encoder (partial interleaved
//! RoPE) + 6-layer decoder (causal self + cross attention, gated MLP), tied embeddings.

use anyhow::Result;
use candle_core::{D, DType, Device, Module, Tensor};
use candle_nn::{
    conv1d, conv1d_no_bias, linear, linear_no_bias, Conv1d, Conv1dConfig, Linear, VarBuilder,
};
use std::path::Path;

const HIDDEN: usize = 288;
const HEADS: usize = 8;
const HEAD_DIM: usize = 36;
const INTER: usize = 1152; // encoder ffn + decoder value/gate
const VOCAB: usize = 32768;
const LAYERS: usize = 6;
const ROT_DIM: usize = 32; // int(36 * 0.9)
const EPS: f64 = 1e-5;
const GROUP_EPS: f64 = 1e-5;

fn ln_nb(x: &Tensor, w: &Tensor, eps: f64) -> candle_core::Result<Tensor> {
    let xf = x.to_dtype(DType::F32)?;
    let mean = xf.mean_keepdim(D::Minus1)?;
    let xc = xf.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = xc.broadcast_div(&(var + eps)?.sqrt()?)?;
    let out = normed.broadcast_mul(&w.to_dtype(DType::F32)?)?;
    out.to_dtype(x.dtype())
}

fn groupnorm(x: &Tensor, gamma: &Tensor, beta: &Tensor, eps: f64) -> candle_core::Result<Tensor> {
    let (_, c, _) = x.dims3()?;
    let xf = x.to_dtype(DType::F32)?;
    let mean = xf.mean_keepdim(D::Minus1)?.mean_keepdim(D::Minus2)?;
    let xc = xf.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?.mean_keepdim(D::Minus2)?;
    let normed = xc.broadcast_div(&(var + eps)?.sqrt()?)?;
    let g = gamma.to_dtype(DType::F32)?.reshape((1, c, 1))?;
    let b = beta.to_dtype(DType::F32)?.reshape((1, c, 1))?;
    let out = normed.broadcast_mul(&g)?.broadcast_add(&b)?;
    out.to_dtype(x.dtype())
}

/// Interleaved RoPE cos/sin `[L, ROT_DIM]` (GPT-NeoX style, `rope_theta=10000`).
fn rope_cos_sin(seq_len: usize, device: &Device, dtype: DType) -> candle_core::Result<(Tensor, Tensor)> {
    let half = ROT_DIM / 2;
    let inv: Vec<f32> = (0..half).map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / ROT_DIM as f32)).collect();
    let mut cos = vec![0f32; seq_len * ROT_DIM];
    let mut sin = vec![0f32; seq_len * ROT_DIM];
    for p in 0..seq_len {
        for i in 0..half {
            let f = p as f32 * inv[i];
            // repeat_interleave(2) of the first half
            cos[p * ROT_DIM + 2 * i] = f.cos();
            cos[p * ROT_DIM + 2 * i + 1] = f.cos();
            sin[p * ROT_DIM + 2 * i] = f.sin();
            sin[p * ROT_DIM + 2 * i + 1] = f.sin();
        }
    }
    Ok((
        Tensor::from_vec(cos, (seq_len, ROT_DIM), device)?.to_dtype(dtype)?,
        Tensor::from_vec(sin, (seq_len, ROT_DIM), device)?.to_dtype(dtype)?,
    ))
}

/// Apply interleaved RoPE to x `[B,H,L,HEAD_DIM]` on the first `ROT_DIM` channels.
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
    let (b, h, l, hd) = x.dims4()?;
    let x_rot = x.narrow(3, 0, ROT_DIM)?; // [B,H,L,ROT]
    let x_pass = x.narrow(3, ROT_DIM, hd - ROT_DIM)?;
    let half = ROT_DIM / 2;
    let even = x_rot.narrow(3, 0, ROT_DIM)?.reshape((b, h, l, half, 2))?;
    let e = even.narrow(4, 0, 1)?.squeeze(4)?; // [B,H,L,half]
    let o = even.narrow(4, 1, 1)?.squeeze(4)?;
    let cb = cos.reshape((1, 1, l, ROT_DIM))?.narrow(3, 0, ROT_DIM)?;
    let sb = sin.reshape((1, 1, l, ROT_DIM))?;
    // cos/sin are already [.,.,L,ROT] with interleave; take even positions
    let ce = cb.reshape((1, 1, l, half, 2))?.narrow(4, 0, 1)?.squeeze(4)?;
    let se = sb.reshape((1, 1, l, half, 2))?.narrow(4, 0, 1)?.squeeze(4)?;
    let out_e = (e.broadcast_mul(&ce)? - o.broadcast_mul(&se)?)?;
    let out_o = (e.broadcast_mul(&se)? + o.broadcast_mul(&ce)?)?;
    let stacked = Tensor::stack(&[&out_e, &out_o], 4)?.reshape((b, h, l, ROT_DIM))?;
    Tensor::cat(&[&stacked, &x_pass], 3)
}

struct Attn {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    is_causal: bool,
}

impl Attn {
    fn load(vb: VarBuilder, is_causal: bool) -> Result<Self> {
        Ok(Self {
            q: linear_no_bias(HIDDEN, HEADS * HEAD_DIM, vb.pp("q_proj"))?,
            k: linear_no_bias(HIDDEN, HEADS * HEAD_DIM, vb.pp("k_proj"))?,
            v: linear_no_bias(HIDDEN, HEADS * HEAD_DIM, vb.pp("v_proj"))?,
            o: linear_no_bias(HEADS * HEAD_DIM, HIDDEN, vb.pp("o_proj"))?,
            is_causal,
        })
    }

    fn forward(&self, x: &Tensor, kv: Option<&Tensor>, rope: Option<(&Tensor, &Tensor)>) -> candle_core::Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        let q = self.q.forward(x)?.reshape((b, l, HEADS, HEAD_DIM))?.transpose(1, 2)?.contiguous()?;
        let src = kv.unwrap_or(x);
        let (bs, ls, _) = src.dims3()?;
        let k = self.k.forward(src)?.reshape((bs, ls, HEADS, HEAD_DIM))?.transpose(1, 2)?.contiguous()?;
        let v = self.v.forward(src)?.reshape((bs, ls, HEADS, HEAD_DIM))?.transpose(1, 2)?.contiguous()?;
        let (q, k) = match rope {
            Some((cos, sin)) => (apply_rope(&q, cos, sin)?, apply_rope(&k, cos, sin)?),
            None => (q, k),
        };
        let scale = 1.0 / (HEAD_DIM as f64).sqrt();
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let mut scores = q.matmul(&k.transpose(2, 3)?.contiguous()?)?.affine(scale, 0.0)?;
        if self.is_causal {
            scores = scores.broadcast_add(&causal_mask(l, ls, scores.device(), scores.dtype())?)?;
        }
        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = attn.matmul(&v.contiguous()?)?;
        let ctx = ctx.transpose(1, 2)?.reshape((b, l, HEADS * HEAD_DIM))?;
        self.o.forward(&ctx)
    }
}

fn causal_mask(l: usize, ls: usize, device: &Device, dtype: DType) -> candle_core::Result<Tensor> {
    let mut m = vec![0f32; l * ls];
    for i in 0..l {
        for j in 0..ls {
            if j > i {
                m[i * ls + j] = f32::MIN;
            }
        }
    }
    Tensor::from_vec(m, (l, ls), device)?.to_dtype(dtype)?.unsqueeze(0)?.unsqueeze(0)
}

struct EncLayer {
    in_ln: Tensor,
    attn: Attn,
    post_ln: Tensor,
    fc1: Linear,
    fc2: Linear,
}

struct DecLayer {
    in_ln: Tensor,
    self_attn: Attn,
    post_ln: Tensor,
    cross_attn: Attn,
    final_ln: Tensor,
    fc1: Linear,
    fc2: Linear,
}

pub struct MoonshineModel {
    conv1: Conv1d,
    conv2: Conv1d,
    conv3: Conv1d,
    groupnorm_w: Tensor,
    groupnorm_b: Tensor,
    enc_layers: Vec<EncLayer>,
    enc_norm: Tensor,
    embed: Tensor, // [VOCAB, HIDDEN]
    dec_layers: Vec<DecLayer>,
    dec_norm: Tensor,
    device: Device,
    dtype: DType,
}

impl MoonshineModel {
    pub fn from_safetensors<P: AsRef<Path>>(path: P, device: &Device, dtype: DType) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path.as_ref()], dtype, device)? };
        let enc = vb.pp("model.encoder");
        let cfg = Conv1dConfig { padding: 0, stride: 64, dilation: 1, groups: 1, cudnn_fwd_algo: None };
        let conv1 = conv1d_no_bias(1, HIDDEN, 127, cfg, enc.pp("conv1"))?;
        let cfg2 = Conv1dConfig { padding: 0, stride: 3, dilation: 1, groups: 1, cudnn_fwd_algo: None };
        let conv2 = conv1d(HIDDEN, 2 * HIDDEN, 7, cfg2, enc.pp("conv2"))?;
        let cfg3 = Conv1dConfig { padding: 0, stride: 2, dilation: 1, groups: 1, cudnn_fwd_algo: None };
        let conv3 = conv1d(2 * HIDDEN, HIDDEN, 3, cfg3, enc.pp("conv3"))?;
        let groupnorm_w = enc.pp("groupnorm").get((HIDDEN,), "weight")?;
        let groupnorm_b = enc.pp("groupnorm").get((HIDDEN,), "bias")?;
        let mut enc_layers = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let vbl = enc.pp(format!("layers.{i}"));
            enc_layers.push(EncLayer {
                in_ln: vbl.pp("input_layernorm").get((HIDDEN,), "weight")?,
                attn: Attn::load(vbl.pp("self_attn"), false)?,
                post_ln: vbl.pp("post_attention_layernorm").get((HIDDEN,), "weight")?,
                fc1: linear(HIDDEN, INTER, vbl.pp("mlp.fc1"))?,
                fc2: linear(INTER, HIDDEN, vbl.pp("mlp.fc2"))?,
            });
        }
        let enc_norm = enc.pp("layer_norm").get((HIDDEN,), "weight")?;
        let embed = vb.pp("model.decoder").get((VOCAB, HIDDEN), "embed_tokens.weight")?;
        let mut dec_layers = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let vbl = vb.pp(format!("model.decoder.layers.{i}"));
            dec_layers.push(DecLayer {
                in_ln: vbl.pp("input_layernorm").get((HIDDEN,), "weight")?,
                self_attn: Attn::load(vbl.pp("self_attn"), true)?,
                post_ln: vbl.pp("post_attention_layernorm").get((HIDDEN,), "weight")?,
                cross_attn: Attn::load(vbl.pp("encoder_attn"), false)?,
                final_ln: vbl.pp("final_layernorm").get((HIDDEN,), "weight")?,
                fc1: linear(HIDDEN, 2 * INTER, vbl.pp("mlp.fc1"))?,
                fc2: linear(INTER, HIDDEN, vbl.pp("mlp.fc2"))?,
            });
        }
        let dec_norm = vb.pp("model.decoder").get((HIDDEN,), "norm.weight")?;
        Ok(Self {
            conv1,
            conv2,
            conv3,
            groupnorm_w,
            groupnorm_b,
            enc_layers,
            enc_norm,
            embed,
            dec_layers,
            dec_norm,
            device: device.clone(),
            dtype,
        })
    }

    /// `audio` `[1, L]` (16 kHz mono) → encoder hidden `[1, T, 288]`.
    pub fn encode(&self, audio: &Tensor) -> candle_core::Result<Tensor> {
        let x = audio.unsqueeze(1)?.to_dtype(self.dtype)?;
        let h = self.conv1.forward(&x)?.tanh()?;
        let h = groupnorm(&h, &self.groupnorm_w, &self.groupnorm_b, GROUP_EPS)?;
        let h = self.conv2.forward(&h)?.gelu_erf()?;
        let h = self.conv3.forward(&h)?.gelu_erf()?;
        let mut hidden = h.transpose(1, 2)?.contiguous()?; // [B,T,288]
        let seq = hidden.dim(1)?;
        let (cos, sin) = rope_cos_sin(seq, hidden.device(), hidden.dtype())?;
        for l in &self.enc_layers {
            let resid = hidden.clone();
            let n = ln_nb(&hidden, &l.in_ln, EPS)?;
            let a = l.attn.forward(&n, None, Some((&cos, &sin)))?;
            hidden = (&resid + &a)?;
            let resid = hidden.clone();
            let n = ln_nb(&hidden, &l.post_ln, EPS)?;
            let m = l.fc2.forward(&l.fc1.forward(&n)?.gelu_erf()?)?;
            hidden = (&resid + &m)?;
        }
        ln_nb(&hidden, &self.enc_norm, EPS)
    }

    /// Greedy decode. `enc` `[1, T, 288]`, `start` token, `eos` token, `max_new` cap.
    pub fn greedy_decode(&self, enc: &Tensor, start: u32, eos: u32, max_new: usize) -> candle_core::Result<Vec<u32>> {
        let mut ids: Vec<u32> = vec![start];
        for _ in 0..max_new {
            let (b, l) = (1usize, ids.len());
            let id_t = Tensor::from_vec(ids.clone(), (b, l), &self.device)?;
            let mut hidden = self.embed.index_select(&id_t.flatten_all()?, 0)?.reshape((b, l, HIDDEN))?;
            hidden = hidden.to_dtype(self.dtype)?;
            let (cos, sin) = rope_cos_sin(l, hidden.device(), hidden.dtype())?;
            for dl in &self.dec_layers {
                let resid = hidden.clone();
                let n = ln_nb(&hidden, &dl.in_ln, EPS)?;
                let a = dl.self_attn.forward(&n, None, Some((&cos, &sin)))?;
                hidden = (&resid + &a)?;
                let resid = hidden.clone();
                let n = ln_nb(&hidden, &dl.post_ln, EPS)?;
                let c = dl.cross_attn.forward(&n, Some(enc), None)?;
                hidden = (&resid + &c)?;
                let resid = hidden.clone();
                let n = ln_nb(&hidden, &dl.final_ln, EPS)?;
                let p = dl.fc1.forward(&n)?.chunk(2, D::Minus1)?;
                let gated = p[1].silu()?.mul(&p[0])?;
                let m = dl.fc2.forward(&gated)?;
                hidden = (&resid + &m)?;
            }
            let hidden = ln_nb(&hidden, &self.dec_norm, EPS)?;
            let last = hidden.narrow(1, l - 1, 1)?.squeeze(1)?; // [1,288]
            let logits = last.matmul(&self.embed.t()?)?; // [1,VOCAB]
            let next = logits.argmax(D::Minus1)?.squeeze(0)?.to_scalar::<u32>()?;
            ids.push(next);
            if next == eos {
                break;
            }
        }
        Ok(ids)
    }
}
