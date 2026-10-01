// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: MusicGen autoregressive decoder (pure Rust)

//! MusicGen decoder: 24-layer causal transformer with cross-attention to a projected text
//! encoder, summed per-codebook token embeddings, learned position table, 4 LM heads.

use anyhow::Result;
use candle_core::{D, DType, Device, Module, Tensor};
use candle_nn::{layer_norm, linear_no_bias, LayerNorm, LayerNormConfig, Linear, VarBuilder};
use std::path::Path;

const HIDDEN: usize = 1024;
const HEADS: usize = 16;
const HEAD_DIM: usize = 64;
const FFN: usize = 4096;
const LAYERS: usize = 24;
const VOCAB: usize = 2048;
const N_CODEBOOKS: usize = 4;
const MAX_POS: usize = 2048;

fn causal_mask(l: usize, device: &Device, dtype: DType) -> candle_core::Result<Tensor> {
    let mut m = vec![0f32; l * l];
    for i in 0..l {
        for j in 0..l {
            if j > i {
                m[i * l + j] = f32::MIN;
            }
        }
    }
    Tensor::from_vec(m, (l, l), device)?.to_dtype(dtype)?.unsqueeze(0)?.unsqueeze(0)
}

struct Attn {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    causal: bool,
}

impl Attn {
    fn load(vb: VarBuilder, causal: bool) -> Result<Self> {
        Ok(Self {
            q: linear_no_bias(HIDDEN, HIDDEN, vb.pp("q_proj"))?,
            k: linear_no_bias(HIDDEN, HIDDEN, vb.pp("k_proj"))?,
            v: linear_no_bias(HIDDEN, HIDDEN, vb.pp("v_proj"))?,
            o: linear_no_bias(HIDDEN, HIDDEN, vb.pp("out_proj"))?,
            causal,
        })
    }

    fn forward(&self, x: &Tensor, kv: Option<&Tensor>) -> candle_core::Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        let q = self.q.forward(x)?.reshape((b, l, HEADS, HEAD_DIM))?.transpose(1, 2)?.contiguous()?;
        let src = kv.unwrap_or(x);
        let (bs, ls, _) = src.dims3()?;
        let k = self.k.forward(src)?.reshape((bs, ls, HEADS, HEAD_DIM))?.transpose(1, 2)?.contiguous()?;
        let v = self.v.forward(src)?.reshape((bs, ls, HEADS, HEAD_DIM))?.transpose(1, 2)?.contiguous()?;
        let scale = 1.0 / (HEAD_DIM as f64).sqrt();
        let mut scores = q.matmul(&k.transpose(2, 3)?.contiguous()?)?.affine(scale, 0.0)?;
        if self.causal {
            scores = scores.broadcast_add(&causal_mask(l, scores.device(), scores.dtype())?)?;
        }
        let attn = crate::device::softmax_last_dim(&scores)?;
        let ctx = attn.matmul(&v.contiguous()?)?;
        let ctx = ctx.transpose(1, 2)?.reshape((b, l, HIDDEN))?;
        self.o.forward(&ctx)
    }
}

struct Layer {
    norm1: LayerNorm,
    self_attn: Attn,
    norm2: LayerNorm,
    cross_attn: Attn,
    norm3: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

impl Layer {
    fn load(vb: VarBuilder) -> Result<Self> {
        let ln = |p| {
            layer_norm(
                HIDDEN,
                LayerNormConfig {
                    eps: 1e-5,
                    remove_mean: true,
                    affine: true,
                },
                vb.pp(p),
            )
        };
        Ok(Self {
            norm1: ln("self_attn_layer_norm").unwrap(),
            self_attn: Attn::load(vb.pp("self_attn"), true)?,
            norm2: ln("encoder_attn_layer_norm").unwrap(),
            cross_attn: Attn::load(vb.pp("encoder_attn"), false)?,
            norm3: ln("final_layer_norm").unwrap(),
            fc1: linear_no_bias(HIDDEN, FFN, vb.pp("fc1"))?,
            fc2: linear_no_bias(FFN, HIDDEN, vb.pp("fc2"))?,
        })
    }

    fn forward(&self, x: &Tensor, enc: &Tensor) -> candle_core::Result<Tensor> {
        let h = (x + &self.self_attn.forward(&self.norm1.forward(x)?, None)?)?;
        let h = (&h + &self.cross_attn.forward(&self.norm2.forward(&h)?, Some(enc))?)?;
        let ff = self.fc2.forward(&self.fc1.forward(&self.norm3.forward(&h)?)?.gelu_erf()?)?;
        &h + &ff
    }
}

pub struct MusicgenDecoder {
    embed_tokens: Vec<Tensor>, // 4 × [2049,1024]
    embed_positions: Tensor,   // [2048,1024]
    layers: Vec<Layer>,
    norm: LayerNorm,
    lm_heads: Vec<Linear>,
}

impl MusicgenDecoder {
    pub fn from_musicgen<P: AsRef<Path>>(path: P, device: &Device, dtype: DType) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path.as_ref()], dtype, device)? };
        let vb = vb.pp("decoder");
        let dec = vb.pp("model.decoder");
        let mut embed_tokens = Vec::new();
        for cb in 0..N_CODEBOOKS {
            embed_tokens.push(dec.pp(format!("embed_tokens.{cb}")).get((VOCAB + 1, HIDDEN), "weight")?);
        }
        let embed_positions = dec.pp("embed_positions").get((MAX_POS, HIDDEN), "weights")?;
        let mut layers = Vec::new();
        for i in 0..LAYERS {
            layers.push(Layer::load(dec.pp(format!("layers.{i}")))?);
        }
        let norm = layer_norm(
            HIDDEN,
            LayerNormConfig {
                eps: 1e-5,
                remove_mean: true,
                affine: true,
            },
            dec.pp("layer_norm"),
        )?;
        let mut lm_heads = Vec::new();
        for cb in 0..N_CODEBOOKS {
            lm_heads.push(linear_no_bias(HIDDEN, VOCAB, vb.pp(format!("lm_heads.{cb}")))?);
        }
        Ok(Self {
            embed_tokens,
            embed_positions,
            layers,
            norm,
            lm_heads,
        })
    }

    /// `input_ids` `[1, 4, L]`, `enc` `[1, S, 1024]` → logits `[1, 4, L, 2048]`.
    pub fn forward(&self, input_ids: &Tensor, enc: &Tensor) -> candle_core::Result<Tensor> {
        let b = input_ids.dim(0)?;
        let l = input_ids.dim(2)?;
        let mut hidden: Option<Tensor> = None;
        for cb in 0..N_CODEBOOKS {
            let idx = input_ids.narrow(1, cb, 1)?.squeeze(1)?; // [B,L]
            let e = self.embed_tokens[cb]
                .index_select(&idx.flatten_all()?.contiguous()?, 0)?
                .reshape((b, l, HIDDEN))?;
            hidden = Some(match hidden {
                None => e,
                Some(h) => (h + e)?,
            });
        }
        let pos = self.embed_positions.narrow(0, 0, l)?.unsqueeze(0)?; // [1,L,1024]
        let mut h = hidden.unwrap().broadcast_add(&pos)?;
        for layer in &self.layers {
            h = layer.forward(&h, enc)?;
        }
        let h = self.norm.forward(&h)?;
        let mut heads = Vec::new();
        for head in &self.lm_heads {
            heads.push(head.forward(&h)?.unsqueeze(1)?); // [1,1,L,2048]
        }
        let logits = Tensor::cat(&heads.iter().collect::<Vec<_>>(), 1)?; // [1,4,L,2048]
        let _ = D::Minus1;
        Ok(logits)
    }
}
