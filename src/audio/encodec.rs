// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: EnCodec 32kHz decoder (MusicGen audio decode)

//! EnCodec 32 kHz **decoder** (SEANet): weight-norm conv/convT + reflect padding + LSTM
//! bottleneck + residual blocks, plus the residual vector-quantizer decode (codes → latents).

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::path::Path;

fn weight_norm(g: &Tensor, v: &Tensor) -> candle_core::Result<Tensor> {
    let norm = v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
    v.broadcast_div(&norm)?.broadcast_mul(g)
}

fn pad_reflect(x: &Tensor, left: usize, right: usize) -> candle_core::Result<Tensor> {
    if left == 0 && right == 0 {
        return Ok(x.clone());
    }
    let l = x.dim(2)?;
    let mut parts: Vec<Tensor> = Vec::new();
    if left > 0 {
        parts.push(x.narrow(2, 1, left)?.contiguous()?.flip(&[2])?);
    }
    parts.push(x.clone());
    if right > 0 {
        parts.push(x.narrow(2, l - 1 - right, right)?.contiguous()?.flip(&[2])?);
    }
    Tensor::cat(&parts.iter().collect::<Vec<_>>(), 2)
}

struct WnConv1d {
    weight: Tensor,
    bias: Option<Tensor>,
    kernel: usize,
    stride: usize,
    dilation: usize,
}

impl WnConv1d {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let eff_k = (self.kernel - 1) * self.dilation + 1;
        let padding_total = eff_k - self.stride;
        let l = x.dim(2)?;
        let n_frames = (((l as f64 - eff_k as f64 + padding_total as f64) / self.stride as f64) + 1.0).ceil() - 1.0;
        let ideal = (n_frames.max(0.0) as usize) * self.stride + eff_k - padding_total;
        let extra = ideal.saturating_sub(l);
        let right = padding_total / 2;
        let left = padding_total - right;
        let x = pad_reflect(x, left, right + extra)?;
        let out = x.conv1d(&self.weight, 0, self.stride, self.dilation, 1)?;
        match &self.bias {
            Some(b) => out.broadcast_add(&b.reshape((1, (), 1))?),
            None => Ok(out),
        }
    }
}

struct WnConvT {
    weight: Tensor,
    bias: Option<Tensor>,
    kernel: usize,
    stride: usize,
}

impl WnConvT {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let padding_total = self.kernel - self.stride;
        let padding_right = padding_total / 2;
        let padding_left = padding_total - padding_right;
        let out = x.conv_transpose1d(&self.weight, 0, 0, self.stride, 1, 1)?;
        let end = out.dim(2)? - padding_right;
        let out = out.narrow(2, padding_left, end - padding_left)?;
        match &self.bias {
            Some(b) => out.broadcast_add(&b.reshape((1, (), 1))?),
            None => Ok(out),
        }
    }
}

fn wn_conv(vb: &VarBuilder, in_c: usize, out_c: usize, k: usize, stride: usize, dilation: usize) -> Result<WnConv1d> {
    let g = vb.get((out_c, 1, 1), "weight_g")?;
    let v = vb.get((out_c, in_c, k), "weight_v")?;
    let weight = weight_norm(&g, &v)?.to_dtype(v.dtype())?.contiguous()?;
    let bias = vb.get((out_c,), "bias").ok();
    Ok(WnConv1d {
        weight,
        bias,
        kernel: k,
        stride,
        dilation,
    })
}

fn wn_convt(vb: &VarBuilder, in_c: usize, out_c: usize, k: usize, stride: usize) -> Result<WnConvT> {
    let g = vb.get((in_c, 1, 1), "weight_g")?;
    let v = vb.get((in_c, out_c, k), "weight_v")?;
    let weight = weight_norm(&g, &v)?.to_dtype(v.dtype())?.contiguous()?;
    let bias = vb.get((out_c,), "bias").ok();
    Ok(WnConvT {
        weight,
        bias,
        kernel: k,
        stride,
    })
}

struct Lstm {
    w_ih: Vec<Tensor>,
    w_hh: Vec<Tensor>,
    b_ih: Vec<Tensor>,
    b_hh: Vec<Tensor>,
    hidden: usize,
}

impl Lstm {
    fn load(vb: VarBuilder, hidden: usize, layers: usize) -> Result<Self> {
        let mut w_ih = Vec::new();
        let mut w_hh = Vec::new();
        let mut b_ih = Vec::new();
        let mut b_hh = Vec::new();
        for i in 0..layers {
            w_ih.push(vb.get((4 * hidden, hidden), &format!("weight_ih_l{i}"))?);
            w_hh.push(vb.get((4 * hidden, hidden), &format!("weight_hh_l{i}"))?);
            b_ih.push(vb.get((4 * hidden,), &format!("bias_ih_l{i}"))?);
            b_hh.push(vb.get((4 * hidden,), &format!("bias_hh_l{i}"))?);
        }
        Ok(Self {
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            hidden,
        })
    }

    /// `x` `[B, C, T]` → `[B, C, T]`.
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (b, c, t) = x.dims3()?; let _ = c;
        let mut input = x.permute((2, 0, 1))?.contiguous()?; // [T,B,C]
        for l in 0..self.w_ih.len() {
            let wih = self.w_ih[l].t()?;
            let whh = self.w_hh[l].t()?;
            let gates_b = (&self.b_ih[l] + &self.b_hh[l])?;
            let mut h = Tensor::zeros((b, self.hidden), x.dtype(), x.device())?;
            let mut cst = Tensor::zeros((b, self.hidden), x.dtype(), x.device())?;
            let mut outs: Vec<Tensor> = Vec::with_capacity(t);
            for step in 0..t {
                let xt = input.narrow(0, step, 1)?.squeeze(0)?;
                let g = xt
                    .matmul(&wih)?
                    .broadcast_add(&h.matmul(&whh)?)?
                    .broadcast_add(&gates_b)?;
                let i = candle_nn::ops::sigmoid(&g.narrow(1, 0, self.hidden)?)?;
                let f = candle_nn::ops::sigmoid(&g.narrow(1, self.hidden, self.hidden)?)?;
                let gg = g.narrow(1, 2 * self.hidden, self.hidden)?.tanh()?;
                let o = candle_nn::ops::sigmoid(&g.narrow(1, 3 * self.hidden, self.hidden)?)?;
                cst = ((f * &cst)? + (i * gg)?)?;
                h = (o * cst.tanh()?)?;
                outs.push(h.clone());
            }
            input = Tensor::stack(&outs, 0)?;
        }
        input.permute((1, 2, 0))?.contiguous()
    }
}

struct Resnet {
    conv1: WnConv1d,
    conv2: WnConv1d,
}

impl Resnet {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let h = self.conv1.forward(&x.elu(1.0)?)?;
        let h = self.conv2.forward(&h.elu(1.0)?)?;
        x + h
    }
}

enum DecLayer {
    Conv(WnConv1d),
    Lstm(Lstm),
    ConvT(WnConvT),
    Resnet(Resnet),
    Elu,
}

/// EnCodec 32 kHz decoder + RVQ codebooks.
pub struct EncodecDecoder {
    layers: Vec<DecLayer>,
    codebooks: Vec<Tensor>,
    pub sample_rate: u32,
}

impl EncodecDecoder {
    pub fn from_musicgen<P: AsRef<Path>>(path: P, device: &Device, dtype: DType) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path.as_ref()], dtype, device)? };
        let dec = vb.pp("audio_encoder.decoder");
        let mut layers = Vec::new();
        layers.push(DecLayer::Conv(wn_conv(&dec.pp("layers.0.conv"), 128, 1024, 7, 1, 1)?));
        layers.push(DecLayer::Lstm(Lstm::load(dec.pp("layers.1.lstm"), 1024, 2)?));
        let ratios = [8usize, 5, 4, 4];
        let nf = 64usize;
        let mut scaling = 1usize << ratios.len();
        let mut idx = 2usize;
        for &ratio in ratios.iter() {
            let cur = scaling * nf;
            let out_c = cur / 2;
            layers.push(DecLayer::Elu);
            layers.push(DecLayer::ConvT(wn_convt(
                &dec.pp(format!("layers.{}.conv", idx + 1)),
                cur,
                out_c,
                ratio * 2,
                ratio,
            )?));
            layers.push(DecLayer::Resnet(Resnet {
                conv1: wn_conv(&dec.pp(format!("layers.{}.block.1.conv", idx + 2)), out_c, out_c / 2, 3, 1, 1)?,
                conv2: wn_conv(&dec.pp(format!("layers.{}.block.3.conv", idx + 2)), out_c / 2, out_c, 1, 1, 1)?,
            }));
            idx += 3;
            scaling /= 2;
        }
        layers.push(DecLayer::Elu);
        layers.push(DecLayer::Conv(wn_conv(&dec.pp("layers.15.conv"), nf, 1, 7, 1, 1)?));

        let q = vb.pp("audio_encoder.quantizer");
        let mut codebooks = Vec::new();
        for i in 0..4 {
            codebooks.push(q.pp(format!("layers.{i}.codebook")).get((2048, 128), "embed")?.contiguous()?);
        }
        Ok(Self {
            layers,
            codebooks,
            sample_rate: 32000,
        })
    }

    /// `codes` `[num_q]` each `[B, T]` (u32) → mono audio `[B, 1, samples]`.
    pub fn decode(&self, codes: &[Tensor]) -> candle_core::Result<Tensor> {
        let mut quant: Option<Tensor> = None;
        for (i, idx) in codes.iter().enumerate() {
            let (b, t) = idx.dims2()?;
            let emb = self.codebooks[i].index_select(&idx.flatten_all()?.contiguous()?, 0)?.reshape((b, t, 128))?;
            let e = emb.transpose(1, 2)?.contiguous()?;
            quant = Some(match quant {
                None => e,
                Some(q) => (q + e)?,
            });
        }
        let mut h = quant.unwrap();
        for layer in &self.layers {
            h = match layer {
                DecLayer::Conv(c) => c.forward(&h)?,
                DecLayer::Lstm(l) => (l.forward(&h)? + &h)?,
                DecLayer::ConvT(c) => c.forward(&h)?,
                DecLayer::Resnet(r) => r.forward(&h)?,
                DecLayer::Elu => h.elu(1.0)?,
            };
        }
        Ok(h)
    }
}
