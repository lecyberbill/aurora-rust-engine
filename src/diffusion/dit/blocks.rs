// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Pure Rust DoubleStreamBlock and SingleStreamBlock for MMDiT (SD 3.5 & Flux.1)

use candle_core::{Result, Tensor};
use candle_nn::{layer_norm, linear, LayerNorm, Linear, Module, VarBuilder};
use crate::diffusion::dit::embeddings::AdaLNZeroModulation;

#[cfg(feature = "flash-attn")]
use candle_flash_attn::flash_attn;

/// Joint Multimodal Attention Block (DoubleStreamBlock) for Image + Text streams (SD 3.5 / Flux.1).
#[derive(Debug, Clone)]
pub struct DoubleStreamBlock {
    // Image stream transformations
    img_norm1: LayerNorm,
    img_qkv: Linear,
    img_proj: Linear,
    img_norm2: LayerNorm,
    img_mlp: (Linear, Linear),
    img_mod: AdaLNZeroModulation,

    // Text stream transformations
    txt_norm1: LayerNorm,
    txt_qkv: Linear,
    txt_proj: Linear,
    txt_norm2: LayerNorm,
    txt_mlp: (Linear, Linear),
    txt_mod: AdaLNZeroModulation,

    heads: usize,
    head_dim: usize,
    scale: f64,
}

impl DoubleStreamBlock {
    pub fn new(
        dim: usize,
        heads: usize,
        mlp_ratio: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = dim / heads;
        let scale = 1.0 / (head_dim as f64).sqrt();
        let mlp_dim = dim * mlp_ratio;

        // Image stream layers
        let img_norm1 = layer_norm(dim, 1e-6, vb.pp("img_norm1"))?;
        let img_qkv = linear(dim, dim * 3, vb.pp("img_qkv"))?;
        let img_proj = linear(dim, dim, vb.pp("img_proj"))?;
        let img_norm2 = layer_norm(dim, 1e-6, vb.pp("img_norm2"))?;
        let img_mlp = (
            linear(dim, mlp_dim, vb.pp("img_mlp.0"))?,
            linear(mlp_dim, dim, vb.pp("img_mlp.1"))?,
        );
        let img_mod = AdaLNZeroModulation::new(dim, dim * 6, vb.pp("img_mod"))?;

        // Text stream layers
        let txt_norm1 = layer_norm(dim, 1e-6, vb.pp("txt_norm1"))?;
        let txt_qkv = linear(dim, dim * 3, vb.pp("txt_qkv"))?;
        let txt_proj = linear(dim, dim, vb.pp("txt_proj"))?;
        let txt_norm2 = layer_norm(dim, 1e-6, vb.pp("txt_norm2"))?;
        let txt_mlp = (
            linear(dim, mlp_dim, vb.pp("txt_mlp.0"))?,
            linear(mlp_dim, dim, vb.pp("txt_mlp.1"))?,
        );
        let txt_mod = AdaLNZeroModulation::new(dim, dim * 6, vb.pp("txt_mod"))?;

        Ok(Self {
            img_norm1,
            img_qkv,
            img_proj,
            img_norm2,
            img_mlp,
            img_mod,
            txt_norm1,
            txt_qkv,
            txt_proj,
            txt_norm2,
            txt_mlp,
            txt_mod,
            heads,
            head_dim,
            scale,
        })
    }

    /// Forward pass executing Joint Multimodal Self-Attention across unified Image and Text tokens.
    pub fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        temb: &Tensor,
        freqs_cos: Option<&Tensor>,
        freqs_sin: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let (b, img_len, d) = img.dims3()?;
        let (_, txt_len, _) = txt.dims3()?;

        // 1. Modulate Image tokens with AdaLN-Zero
        let (img_shift1, img_scale1, img_gate1, img_shift2, img_scale2, img_gate2) =
            self.img_mod.modulate_double(temb)?;
        let img_normed = self.img_norm1.forward(img)?;
        let img_scale1 = (img_scale1.unsqueeze(1)? + 1.0)?;
        let img_shift1 = img_shift1.unsqueeze(1)?;
        let img_normed = img_normed.broadcast_mul(&img_scale1)?.broadcast_add(&img_shift1)?;

        // 2. Modulate Text tokens with AdaLN-Zero
        let (txt_shift1, txt_scale1, txt_gate1, txt_shift2, txt_scale2, txt_gate2) =
            self.txt_mod.modulate_double(temb)?;
        let txt_normed = self.txt_norm1.forward(txt)?;
        let txt_scale1 = (txt_scale1.unsqueeze(1)? + 1.0)?;
        let txt_shift1 = txt_shift1.unsqueeze(1)?;
        let txt_normed = txt_normed.broadcast_mul(&txt_scale1)?.broadcast_add(&txt_shift1)?;

        // 3. Project Q, K, V
        let img_qkv = self.img_qkv.forward(&img_normed)?;
        let txt_qkv = self.txt_qkv.forward(&txt_normed)?;

        let img_qkv = img_qkv.reshape((b, img_len, 3, self.heads, self.head_dim))?;
        let txt_qkv = txt_qkv.reshape((b, txt_len, 3, self.heads, self.head_dim))?;

        let q_img = img_qkv.narrow(2, 0, 1)?.squeeze(2)?;
        let k_img = img_qkv.narrow(2, 1, 1)?.squeeze(2)?;
        let v_img = img_qkv.narrow(2, 2, 1)?.squeeze(2)?;

        let q_txt = txt_qkv.narrow(2, 0, 1)?.squeeze(2)?;
        let k_txt = txt_qkv.narrow(2, 1, 1)?.squeeze(2)?;
        let v_txt = txt_qkv.narrow(2, 2, 1)?.squeeze(2)?;

        // Apply RoPE on image and text tokens if provided
        let (q_img, k_img) = if let (Some(cos), Some(sin)) = (freqs_cos, freqs_sin) {
            (
                crate::diffusion::dit::embeddings::apply_rope(&q_img, cos, sin)?,
                crate::diffusion::dit::embeddings::apply_rope(&k_img, cos, sin)?,
            )
        } else {
            (q_img, k_img)
        };

        // 4. Concatenate tokens for Joint Attention: [B, txt_len + img_len, Heads, Head_Dim]
        let q = Tensor::cat(&[&q_txt, &q_img], 1)?;
        let k = Tensor::cat(&[&k_txt, &k_img], 1)?;
        let v = Tensor::cat(&[&v_txt, &v_img], 1)?;

        // 5. Joint Attention Computation (FlashAttention-2 when available)
        let attn_out = {
            #[cfg(feature = "flash-attn")]
            {
                if q.device().is_cuda() && q.dtype() == candle_core::DType::F16 {
                    flash_attn(&q, &k, &v, self.scale as f32, false)?
                } else {
                    self.standard_sdpa(&q, &k, &v)?
                }
            }
            #[cfg(not(feature = "flash-attn"))]
            {
                self.standard_sdpa(&q, &k, &v)?
            }
        };

        let attn_out = attn_out.reshape((b, txt_len + img_len, d))?;

        // 6. Split back into Text and Image streams
        let txt_attn = attn_out.narrow(1, 0, txt_len)?;
        let img_attn = attn_out.narrow(1, txt_len, img_len)?;

        // 7. Apply Attention Output Projection & Gated Residual
        let img_attn_proj = self.img_proj.forward(&img_attn)?;
        let img_gate1 = img_gate1.unsqueeze(1)?;
        let img = (img + img_attn_proj.broadcast_mul(&img_gate1)?)?;

        let txt_attn_proj = self.txt_proj.forward(&txt_attn)?;
        let txt_gate1 = txt_gate1.unsqueeze(1)?;
        let txt = (txt + txt_attn_proj.broadcast_mul(&txt_gate1)?)?;

        // 8. MLP Forward Passes with AdaLN-Zero Gating
        let img_normed2 = self.img_norm2.forward(&img)?;
        let img_scale2 = (img_scale2.unsqueeze(1)? + 1.0)?;
        let img_shift2 = img_shift2.unsqueeze(1)?;
        let img_normed2 = img_normed2.broadcast_mul(&img_scale2)?.broadcast_add(&img_shift2)?;
        let img_mlp_h = self.img_mlp.0.forward(&img_normed2)?.gelu_erf()?;
        let img_mlp_out = self.img_mlp.1.forward(&img_mlp_h)?;
        let img_gate2 = img_gate2.unsqueeze(1)?;
        let img = (&img + img_mlp_out.broadcast_mul(&img_gate2)?)?;

        let txt_normed2 = self.txt_norm2.forward(&txt)?;
        let txt_scale2 = (txt_scale2.unsqueeze(1)? + 1.0)?;
        let txt_shift2 = txt_shift2.unsqueeze(1)?;
        let txt_normed2 = txt_normed2.broadcast_mul(&txt_scale2)?.broadcast_add(&txt_shift2)?;
        let txt_mlp_h = self.txt_mlp.0.forward(&txt_normed2)?.gelu_erf()?;
        let txt_mlp_out = self.txt_mlp.1.forward(&txt_mlp_h)?;
        let txt_gate2 = txt_gate2.unsqueeze(1)?;
        let txt = (&txt + txt_mlp_out.broadcast_mul(&txt_gate2)?)?;

        Ok((img, txt))
    }

    fn standard_sdpa(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let (b, seq, h, d) = q.dims4()?;
        let q = (q.transpose(1, 2)? * self.scale)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        let scores = q.matmul(&k.transpose(2, 3)?)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?;
        ctx.transpose(1, 2)?.reshape((b, seq, h, d))
    }
}

/// Unified Single-Stream Block (SingleStreamBlock) for concatenated sequences in Flux.1.
#[derive(Debug, Clone)]
pub struct SingleStreamBlock {
    norm: LayerNorm,
    qkv: Linear,
    mlp: (Linear, Linear),
    proj: Linear,
    modulation: AdaLNZeroModulation,
    heads: usize,
    head_dim: usize,
    scale: f64,
}

impl SingleStreamBlock {
    pub fn new(dim: usize, heads: usize, mlp_ratio: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = dim / heads;
        let scale = 1.0 / (head_dim as f64).sqrt();
        let mlp_dim = dim * mlp_ratio;

        let norm = layer_norm(dim, 1e-6, vb.pp("norm"))?;
        let qkv = linear(dim, dim * 3, vb.pp("qkv"))?;
        let mlp = (
            linear(dim, mlp_dim, vb.pp("mlp.0"))?,
            linear(mlp_dim, dim, vb.pp("mlp.1"))?,
        );
        let proj = linear(dim + mlp_dim, dim, vb.pp("proj"))?;
        let modulation = AdaLNZeroModulation::new(dim, dim * 3, vb.pp("modulation"))?;

        Ok(Self {
            norm,
            qkv,
            mlp,
            proj,
            modulation,
            heads,
            head_dim,
            scale,
        })
    }

    pub fn forward(&self, x: &Tensor, temb: &Tensor) -> Result<Tensor> {
        let (b, seq, d) = x.dims3()?;
        let (shift, scale, gate) = self.modulation.modulate(temb)?;

        let normed = self.norm.forward(x)?;
        let scale = (scale.unsqueeze(1)? + 1.0)?;
        let shift = shift.unsqueeze(1)?;
        let normed = normed.broadcast_mul(&scale)?.broadcast_add(&shift)?;

        // 1. Parallel QKV + MLP projections
        let qkv = self.qkv.forward(&normed)?;
        let mlp_h = self.mlp.0.forward(&normed)?.gelu_erf()?;

        let qkv = qkv.reshape((b, seq, 3, self.heads, self.head_dim))?;
        let q = qkv.narrow(2, 0, 1)?.squeeze(2)?;
        let k = qkv.narrow(2, 1, 1)?.squeeze(2)?;
        let v = qkv.narrow(2, 2, 1)?.squeeze(2)?;

        // 2. Self-Attention
        let attn_out = {
            #[cfg(feature = "flash-attn")]
            {
                if q.device().is_cuda() && q.dtype() == candle_core::DType::F16 {
                    flash_attn(&q, &k, &v, self.scale as f32, false)?
                } else {
                    self.standard_sdpa(&q, &k, &v)?
                }
            }
            #[cfg(not(feature = "flash-attn"))]
            {
                self.standard_sdpa(&q, &k, &v)?
            }
        };

        let attn_out = attn_out.reshape((b, seq, d))?;

        // 3. Unified Linear Output Projection
        let combined = Tensor::cat(&[&attn_out, &mlp_h], 2)?;
        let out = self.proj.forward(&combined)?;

        // 4. Gated Residual
        let gate = gate.unsqueeze(1)?;
        x + out.broadcast_mul(&gate)?
    }

    fn standard_sdpa(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let (b, seq, h, d) = q.dims4()?;
        let q = (q.transpose(1, 2)? * self.scale)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        let scores = q.matmul(&k.transpose(2, 3)?)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?;
        ctx.transpose(1, 2)?.reshape((b, seq, h, d))
    }
}
