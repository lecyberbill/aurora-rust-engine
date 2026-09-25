// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: T5 encoder (t5-base) for Stable Audio Open

//! Minimal T5 encoder (English `t5-base`) used by Stable Audio Open: relative position bias,
//! RMS-style `T5LayerNorm`, non-gated ReLU FFN.

use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{linear_no_bias, Linear, VarBuilder};
use std::path::Path;

const D_MODEL: usize = 768;
const D_FF: usize = 3072;
const D_KV: usize = 64;
const NUM_HEADS: usize = 12;
const NUM_LAYERS: usize = 12;
const NUM_BUCKETS: usize = 32;
const MAX_DISTANCE: usize = 128;
const EPS: f64 = 1e-6;

/// `T5LayerNorm` (weight-only RMS norm).
struct T5LayerNorm {
    weight: Tensor,
}

impl T5LayerNorm {
    fn load(vb: VarBuilder, dim: usize) -> Result<Self> {
        Ok(Self {
            weight: vb.get((dim,), "weight")?,
        })
    }
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let dt = x.dtype();
        let xf = x.to_dtype(DType::F32)?;
        let var = xf.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let normed = xf.broadcast_div(&(var + EPS)?.sqrt()?)?;
        normed.to_dtype(dt)?.broadcast_mul(&self.weight)
    }
}

struct T5Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    relative_attention_bias: Option<Tensor>, // [NUM_BUCKETS, NUM_HEADS]
}

impl T5Attention {
    fn load(vb: VarBuilder, has_bias: bool) -> Result<Self> {
        Ok(Self {
            q: linear_no_bias(D_MODEL, NUM_HEADS * D_KV, vb.pp("q"))?,
            k: linear_no_bias(D_MODEL, NUM_HEADS * D_KV, vb.pp("k"))?,
            v: linear_no_bias(D_MODEL, NUM_HEADS * D_KV, vb.pp("v"))?,
            o: linear_no_bias(NUM_HEADS * D_KV, D_MODEL, vb.pp("o"))?,
            relative_attention_bias: if has_bias {
                Some(vb.get((NUM_BUCKETS, NUM_HEADS), "relative_attention_bias.weight")?)
            } else {
                None
            },
        })
    }

    /// Relative position buckets (bidirectional, `num_buckets/2=16`, `max_exact=8`).
    fn compute_buckets(seq_len: usize, device: &Device) -> candle_core::Result<Tensor> {
        let mut buckets = vec![0u32; seq_len * seq_len];
        let nb: usize = NUM_BUCKETS / 2; // 16
        let max_exact: usize = nb / 2; // 8
        for i in 0..seq_len {
            for j in 0..seq_len {
                let rp = j as i64 - i as i64;
                let mut b: u32 = if rp > 0 { nb as u32 } else { 0 };
                let n = rp.unsigned_abs() as f32;
                let v: u32 = if (n as usize) < max_exact {
                    n as u32
                } else {
                    let val = max_exact as f32
                        + ((n / max_exact as f32).ln() / ((MAX_DISTANCE as f32) / (max_exact as f32)).ln())
                            * ((nb - max_exact) as f32);
                    (val as u32).min((nb - 1) as u32)
                };
                b += v;
                buckets[i * seq_len + j] = b;
            }
        }
        Tensor::from_vec(buckets, (seq_len, seq_len), device)
    }

    /// `position_bias` `[1, NUM_HEADS, L, L]`.
    fn compute_bias(&self, seq_len: usize, device: &Device) -> candle_core::Result<Tensor> {
        let bias = self.relative_attention_bias.as_ref().unwrap();
        let buckets = Self::compute_buckets(seq_len, device)?; // [L,L] u32
        let flat = buckets.flatten_all()?;
        let values = bias.index_select(&flat, 0)?; // [L*L, NUM_HEADS]
        values
            .reshape((seq_len, seq_len, NUM_HEADS))?
            .permute((2, 0, 1))?
            .unsqueeze(0)
    }

    fn forward(
        &self,
        hidden: &Tensor,
        mask: Option<&Tensor>,
        position_bias: Option<&Tensor>,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let (o, bias, _q, _k, _v, _aw) = self.forward_full(hidden, mask, position_bias)?;
        Ok((o, bias))
    }

    /// Scores only: `q @ k^T + position_bias (+ mask)`.
    fn forward_scores(
        &self,
        hidden: &Tensor,
        mask: Option<&Tensor>,
        position_bias: Option<&Tensor>,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let (b, l, _) = hidden.dims3()?;
        let q = self.q.forward(hidden)?.reshape((b, l, NUM_HEADS, D_KV))?.transpose(1, 2)?;
        let k = self.k.forward(hidden)?.reshape((b, l, NUM_HEADS, D_KV))?.transpose(1, 2)?;
        let scores = q.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        let bias = match position_bias {
            Some(pb) => pb.clone(),
            None => self.compute_bias(l, hidden.device())?,
        };
        let mut scores = scores.broadcast_add(&bias)?;
        if let Some(m) = mask {
            scores = scores.broadcast_add(&m.reshape((b, 1, 1, l))?)?;
        }
        Ok((scores, bias))
    }

    /// Full forward returning `(out, bias, q_lin, k_lin, v_lin, attn_weights)`.
    #[allow(clippy::type_complexity)]
    fn forward_full(
        &self,
        hidden: &Tensor,
        mask: Option<&Tensor>,
        position_bias: Option<&Tensor>,
    ) -> candle_core::Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let (b, l, _) = hidden.dims3()?;
        let q_lin = self.q.forward(hidden)?;
        let k_lin = self.k.forward(hidden)?;
        let v_lin = self.v.forward(hidden)?;
        let q = q_lin.reshape((b, l, NUM_HEADS, D_KV))?.transpose(1, 2)?.contiguous()?;
        let k = k_lin.reshape((b, l, NUM_HEADS, D_KV))?.transpose(1, 2)?.contiguous()?;
        let v = v_lin.reshape((b, l, NUM_HEADS, D_KV))?.transpose(1, 2)?.contiguous()?;

        let scores = q.matmul(&k.transpose(2, 3)?.contiguous()?)?; // [B,H,L,L]
        let bias = match position_bias {
            Some(pb) => pb.clone(),
            None => self.compute_bias(l, hidden.device())?,
        };
        let mut scores = scores.broadcast_add(&bias)?;
        if let Some(m) = mask {
            let m = m.reshape((b, 1, 1, l))?;
            scores = scores.broadcast_add(&m)?;
        }
        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = attn.matmul(&v)?; // [B,H,L,D_KV]
        let ctx = ctx.transpose(1, 2)?.reshape((b, l, NUM_HEADS * D_KV))?;
        let o = self.o.forward(&ctx)?;
        Ok((o, bias, q_lin, k_lin, v_lin, attn))
    }
}

struct T5Block {
    attn_norm: T5LayerNorm,
    attn: T5Attention,
    ffn_norm: T5LayerNorm,
    wi: Linear,
    wo: Linear,
}

impl T5Block {
    fn load(vb: VarBuilder, has_bias: bool) -> Result<Self> {
        let vb0 = vb.pp("layer.0");
        let attn_norm = T5LayerNorm::load(vb0.pp("layer_norm"), D_MODEL)?;
        let attn = T5Attention::load(vb0.pp("SelfAttention"), has_bias)?;
        let vb1 = vb.pp("layer.1");
        let ffn_norm = T5LayerNorm::load(vb1.pp("layer_norm"), D_MODEL)?;
        let dense = vb1.pp("DenseReluDense");
        let wi = linear_no_bias(D_MODEL, D_FF, dense.pp("wi"))?;
        let wo = linear_no_bias(D_FF, D_MODEL, dense.pp("wo"))?;
        Ok(Self {
            attn_norm,
            attn,
            ffn_norm,
            wi,
            wo,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        mask: Option<&Tensor>,
        position_bias: Option<&Tensor>,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let (out, _ln0, _a, _r, _ln1, _f) = self.forward_debug(hidden, mask, position_bias)?;
        let (h, pb) = out;
        Ok((h, pb))
    }

    /// Debug: `(hidden_out, position_bias)` plus `(ln0, attn_out, resid1, ln1, ff)`.
    #[allow(clippy::type_complexity)]
    fn forward_debug(
        &self,
        hidden: &Tensor,
        mask: Option<&Tensor>,
        position_bias: Option<&Tensor>,
    ) -> candle_core::Result<((Tensor, Tensor), Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let ln0 = self.attn_norm.forward(hidden)?;
        let (attn_out, pb) = self.attn.forward(&ln0, mask, position_bias)?;
        let resid1 = (hidden + &attn_out)?;
        let ln1 = self.ffn_norm.forward(&resid1)?;
        let ff = self.wi.forward(&ln1)?.relu()?;
        let ff = self.wo.forward(&ff)?;
        let out = (&resid1 + &ff)?;
        Ok(((out, pb), ln0, attn_out, resid1, ln1, ff))
    }
}

/// T5 encoder (t5-base).
pub struct T5Encoder {
    shared: Tensor, // [vocab, 768]
    blocks: Vec<T5Block>,
    final_norm: T5LayerNorm,
    device: Device,
    dtype: DType,
}

impl T5Encoder {
    pub fn from_safetensors<P: AsRef<Path>>(path: P, device: &Device, dtype: DType) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path.as_ref()], dtype, device)? };
        let shared = vb.get((32128, D_MODEL), "shared.weight")?;
        let mut blocks = Vec::with_capacity(NUM_LAYERS);
        for i in 0..NUM_LAYERS {
            blocks.push(T5Block::load(vb.pp(format!("encoder.block.{i}")), i == 0)?);
        }
        let final_norm = T5LayerNorm::load(vb.pp("encoder.final_layer_norm"), D_MODEL)?;
        Ok(Self {
            shared,
            blocks,
            final_norm,
            device: device.clone(),
            dtype,
        })
    }

    /// `input_ids` `[1, L]` (u32), `attention_mask` `[1, L]` (1/0) → hidden `[1, L, 768]`.
    pub fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> candle_core::Result<Tensor> {
        let (_emb, _blocks, final_h) = self.forward_blocks(input_ids, attention_mask)?;
        Ok(final_h)
    }

    /// Debug: returns `(embeddings, per-block outputs, final)`.
    pub fn forward_blocks(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> candle_core::Result<(Tensor, Vec<Tensor>, Tensor)> {
        let (b, l) = input_ids.dims2()?;
        let ids = input_ids.flatten_all()?;
        let mut hidden = self.shared.index_select(&ids, 0)?.reshape((b, l, D_MODEL))?;
        hidden = hidden.to_dtype(self.dtype)?;
        let emb = hidden.clone();

        // additive mask: 0 for real tokens, `f32::MIN` for padding -> [B, L]
        let mask_f = attention_mask.to_dtype(DType::F32)?;
        let additive = mask_f.affine(-1.0, 0.0)?.affine(f32::MIN as f64, 0.0)?; // (1-mask)*MIN
        let additive = additive.to_dtype(self.dtype)?;

        let mut position_bias: Option<Tensor> = None;
        let mut block_outs = Vec::with_capacity(self.blocks.len());
        for blk in &self.blocks {
            let (h, pb) = blk.forward(&hidden, Some(&additive), position_bias.as_ref())?;
            hidden = h;
            position_bias = Some(pb);
            block_outs.push(hidden.clone());
        }
        let final_h = self.final_norm.forward(&hidden)?;
        Ok((emb, block_outs, final_h))
    }

    /// Debug: `(emb, ln0, attn_out, resid1, ln1, ff, out)` for block 0.
    #[allow(clippy::type_complexity)]
    pub fn debug_block0(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> candle_core::Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let (b, l) = input_ids.dims2()?;
        let ids = input_ids.flatten_all()?;
        let emb = self.shared.index_select(&ids, 0)?.reshape((b, l, D_MODEL))?.to_dtype(self.dtype)?;
        let mask_f = attention_mask.to_dtype(DType::F32)?;
        let additive = mask_f.affine(-1.0, 0.0)?.affine(f32::MIN as f64, 0.0)?.to_dtype(self.dtype)?;
        let ((out, _pb), ln0, a, r, ln1, ff) = self.blocks[0].forward_debug(&emb, Some(&additive), None)?;
        Ok((emb, ln0, a, r, ln1, ff, out))
    }

    /// Debug: block-0 relative position bias `[1, heads, L, L]`.
    pub fn debug_bias(&self, seq_len: usize) -> candle_core::Result<Tensor> {
        self.blocks[0].attn.compute_bias(seq_len, &self.device)
    }

    /// Debug: `(q, k, v, attn_weights, attn_out)` for block-0 self-attention.
    #[allow(clippy::type_complexity)]
    pub fn debug_attn0(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> candle_core::Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let (b, l) = input_ids.dims2()?;
        let ids = input_ids.flatten_all()?;
        let emb = self.shared.index_select(&ids, 0)?.reshape((b, l, D_MODEL))?.to_dtype(self.dtype)?;
        let ln0 = self.blocks[0].attn_norm.forward(&emb)?;
        let mask_f = attention_mask.to_dtype(DType::F32)?;
        let additive = mask_f.affine(-1.0, 0.0)?.affine(f32::MIN as f64, 0.0)?.to_dtype(self.dtype)?;
        let (o, _bias, q, k, v, aw) = self.blocks[0].attn.forward_full(&ln0, Some(&additive), None)?;
        Ok((q, k, v, aw, o))
    }

    /// Debug: block-0 pre-softmax scores `[B, heads, L, L]`.
    pub fn debug_scores0(&self, input_ids: &Tensor, attention_mask: &Tensor) -> candle_core::Result<Tensor> {
        let (b, l) = input_ids.dims2()?;
        let ids = input_ids.flatten_all()?;
        let emb = self.shared.index_select(&ids, 0)?.reshape((b, l, D_MODEL))?.to_dtype(self.dtype)?;
        let ln0 = self.blocks[0].attn_norm.forward(&emb)?;
        let mask_f = attention_mask.to_dtype(DType::F32)?;
        let additive = mask_f.affine(-1.0, 0.0)?.affine(f32::MIN as f64, 0.0)?.to_dtype(self.dtype)?;
        let (scores, _b) = self.blocks[0].attn.forward_scores(&ln0, Some(&additive), None)?;
        Ok(scores)
    }
}
