// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Pure Rust DoubleStreamBlock and SingleStreamBlock for MMDiT (SD 3.5 & Flux.1)

use candle_core::{Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};
use crate::diffusion::dit::embeddings::AdaLNZeroModulation;

#[cfg(feature = "flash-attn")]
use candle_flash_attn::flash_attn;

/// RMS Normalization layer for QK Normalization in Flux.1
#[derive(Debug, Clone)]
pub struct RMSNorm {
    scale: Tensor,
}

impl RMSNorm {
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let scale = vb.get(dim, "scale")?;
        Ok(Self { scale })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let orig_dtype = x.dtype();
        let x_f32 = x.to_dtype(candle_core::DType::F32)?;
        let sq = x_f32.sqr()?;
        let mean = sq.mean_keepdim(sq.dims().len() - 1)?;
        let rms = (mean + 1e-6)?.sqrt()?;
        let norm = x_f32.broadcast_div(&rms)?;
        let scale_f32 = self.scale.to_dtype(candle_core::DType::F32)?;
        norm.broadcast_mul(&scale_f32)?.to_dtype(orig_dtype)
    }
}

/// Exact GELU(approximate="tanh") for Flux.1 MLP transformations
fn gelu_tanh(x: &Tensor) -> Result<Tensor> {
    let orig_dtype = x.dtype();
    let x_f32 = x.to_dtype(candle_core::DType::F32)?;
    let c = (2.0f64 / std::f64::consts::PI).sqrt() as f32;
    let x_cubed = (x_f32.sqr()? * &x_f32)?;
    let inner = (&x_f32 + (x_cubed * 0.044715f64)?)?;
    let tanh = (inner * (c as f64))?.tanh()?;
    let result = ((x_f32 * 0.5)? * (tanh + 1.0)?)?;
    result.to_dtype(orig_dtype)
}

/// Joint Multimodal Attention Block (DoubleStreamBlock) for Image + Text streams (SD 3.5 / Flux.1).
#[derive(Debug, Clone)]
pub struct DoubleStreamBlock {
    // Image stream transformations
    img_qkv: Linear,
    img_q_norm: Option<RMSNorm>,
    img_k_norm: Option<RMSNorm>,
    img_proj: Linear,
    img_mlp: (Linear, Linear),
    img_mod: AdaLNZeroModulation,

    // Text stream transformations
    txt_qkv: Linear,
    txt_q_norm: Option<RMSNorm>,
    txt_k_norm: Option<RMSNorm>,
    txt_proj: Linear,
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

        let linear_layer = |in_d: usize, out_d: usize, path: VarBuilder| -> Result<Linear> {
            linear(in_d, out_d, path.clone()).or_else(|_| candle_nn::linear_no_bias(in_d, out_d, path))
        };

        // Image stream layers
        let img_qkv = linear_layer(dim, dim * 3, vb.pp("img_attn.qkv"))?;
        let img_q_norm = RMSNorm::new(head_dim, vb.pp("img_attn.norm.query_norm")).ok();
        let img_k_norm = RMSNorm::new(head_dim, vb.pp("img_attn.norm.key_norm")).ok();
        let img_proj = linear_layer(dim, dim, vb.pp("img_attn.proj"))?;
        let swiglu_in_dim = dim * 6;
        let swiglu_mid_dim = dim * 3;
        let img_mlp = (
            linear_layer(dim, mlp_dim, vb.pp("img_mlp.0"))
                .or_else(|_| linear_layer(dim, swiglu_in_dim, vb.pp("img_mlp.0")))?,
            linear_layer(mlp_dim, dim, vb.pp("img_mlp.2"))
                .or_else(|_| linear_layer(swiglu_mid_dim, dim, vb.pp("img_mlp.2")))?,
        );
        let img_mod = AdaLNZeroModulation::new(dim, dim * 6, vb.pp("img_mod"))?;

        // Text stream layers
        let txt_qkv = linear_layer(dim, dim * 3, vb.pp("txt_attn.qkv"))?;
        let txt_q_norm = RMSNorm::new(head_dim, vb.pp("txt_attn.norm.query_norm")).ok();
        let txt_k_norm = RMSNorm::new(head_dim, vb.pp("txt_attn.norm.key_norm")).ok();
        let txt_proj = linear_layer(dim, dim, vb.pp("txt_attn.proj"))?;
        let txt_mlp = (
            linear_layer(dim, mlp_dim, vb.pp("txt_mlp.0"))
                .or_else(|_| linear_layer(dim, swiglu_in_dim, vb.pp("txt_mlp.0")))?,
            linear_layer(mlp_dim, dim, vb.pp("txt_mlp.2"))
                .or_else(|_| linear_layer(swiglu_mid_dim, dim, vb.pp("txt_mlp.2")))?,
        );
        let txt_mod = AdaLNZeroModulation::new(dim, dim * 6, vb.pp("txt_mod"))?;

        Ok(Self {
            img_qkv,
            img_q_norm,
            img_k_norm,
            img_proj,
            img_mlp,
            img_mod,
            txt_qkv,
            txt_q_norm,
            txt_k_norm,
            txt_proj,
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
        img_freqs_cos: Option<&Tensor>,
        img_freqs_sin: Option<&Tensor>,
        txt_freqs_cos: Option<&Tensor>,
        txt_freqs_sin: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let (b, img_len, d) = img.dims3()?;
        let (_, txt_len, _) = txt.dims3()?;
        let orig_dtype = img.dtype();
        if std::env::var("FLUX_TRACE").is_ok() {
            let rms = |t: &Tensor| -> f32 {
                let f = t.to_dtype(candle_core::DType::F32).unwrap().flatten_all().unwrap();
                if let Ok(v) = f.to_vec1::<f32>() {
                    let m = v.iter().map(|x| (*x as f64)*(*x as f64)).sum::<f64>() / v.len() as f64;
                    m.sqrt() as f32
                } else { 0.0 }
            };
            let mx = |t: &Tensor| -> f32 {
                let f = t.to_dtype(candle_core::DType::F32).unwrap().flatten_all().unwrap();
                if let Ok(v) = f.to_vec1::<f32>() { v.iter().fold(0f32, |a,&b| a.max(if b<0.0 {-b} else {b})) } else { 0.0 }
            };
            eprintln!("      [DBLOCK-IN] img_rms={:.3} txt_rms={:.3} txt_max={:.3}", rms(img), rms(txt), mx(txt));
        }

        // LayerNorm(elementwise_affine=False) helper
        let norm_layer = |x: &Tensor| -> Result<Tensor> {
            let orig_dtype = x.dtype();
            let x_f32 = x.to_dtype(candle_core::DType::F32)?;
            let mean = x_f32.mean_keepdim(x_f32.dims().len() - 1)?;
            let diff = x_f32.broadcast_sub(&mean)?;
            let var = diff.sqr()?.mean_keepdim(diff.dims().len() - 1)?;
            let std = (var + 1e-6)?.sqrt()?;
            diff.broadcast_div(&std)?.to_dtype(orig_dtype)
        };

        // 1. Modulate Image tokens with AdaLN-Zero: (1 + scale) * LayerNorm(img) + shift
        let (img_shift1, img_scale1, img_gate1, img_shift2, img_scale2, img_gate2) =
            self.img_mod.modulate_double(temb)?;
        let img_norm1 = norm_layer(img)?;
        let img_scale1 = (img_scale1.unsqueeze(1)? + 1.0)?;
        let img_shift1 = img_shift1.unsqueeze(1)?;
        let img_normed = img_norm1.broadcast_mul(&img_scale1)?.broadcast_add(&img_shift1)?;

        // 2. Modulate Text tokens with AdaLN-Zero: (1 + scale) * LayerNorm(txt) + shift
        let (txt_shift1, txt_scale1, txt_gate1, txt_shift2, txt_scale2, txt_gate2) =
            self.txt_mod.modulate_double(temb)?;
        let txt_norm1 = norm_layer(txt)?;
        let txt_scale1 = (txt_scale1.unsqueeze(1)? + 1.0)?;
        let txt_shift1 = txt_shift1.unsqueeze(1)?;
        let txt_normed = txt_norm1.broadcast_mul(&txt_scale1)?.broadcast_add(&txt_shift1)?;

        // 3. Project Q, K, V
        let img_qkv = self.img_qkv.forward(&img_normed)?;
        let txt_qkv = self.txt_qkv.forward(&txt_normed)?;

        let img_qkv = img_qkv.reshape((b, img_len, 3, self.heads, self.head_dim))?;
        let txt_qkv = txt_qkv.reshape((b, txt_len, 3, self.heads, self.head_dim))?;

        let mut q_img = img_qkv.narrow(2, 0, 1)?.squeeze(2)?;
        let mut k_img = img_qkv.narrow(2, 1, 1)?.squeeze(2)?;
        let v_img = img_qkv.narrow(2, 2, 1)?.squeeze(2)?;

        let mut q_txt = txt_qkv.narrow(2, 0, 1)?.squeeze(2)?;
        let mut k_txt = txt_qkv.narrow(2, 1, 1)?.squeeze(2)?;
        let v_txt = txt_qkv.narrow(2, 2, 1)?.squeeze(2)?;

        // Apply QK RMSNorm if present
        if let Some(ref q_norm) = self.img_q_norm {
            q_img = q_norm.forward(&q_img)?;
        }
        if let Some(ref k_norm) = self.img_k_norm {
            k_img = k_norm.forward(&k_img)?;
        }
        if let Some(ref q_norm) = self.txt_q_norm {
            q_txt = q_norm.forward(&q_txt)?;
        }
        if let Some(ref k_norm) = self.txt_k_norm {
            k_txt = k_norm.forward(&k_txt)?;
        }

        // Apply RoPE on image and text tokens if provided
        let (q_txt, k_txt) = if let (Some(cos), Some(sin)) = (txt_freqs_cos, txt_freqs_sin) {
            (
                crate::diffusion::dit::embeddings::apply_rope(&q_txt, cos, sin)?,
                crate::diffusion::dit::embeddings::apply_rope(&k_txt, cos, sin)?,
            )
        } else {
            (q_txt, k_txt)
        };

        let (q_img, k_img) = if let (Some(cos), Some(sin)) = (img_freqs_cos, img_freqs_sin) {
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

        // 5. Joint Attention Computation (FlashAttention-2 fast path or F32 SDPA fallback)
        let attn_out = self.sdpa(&q, &k, &v)?;

        let attn_out = attn_out.reshape((b, txt_len + img_len, d))?;

        // 6. Split back into Text and Image streams: tokens 0..txt_len is Text, txt_len..total is Image
        let txt_attn = attn_out.narrow(1, 0, txt_len)?;
        let img_attn = attn_out.narrow(1, txt_len, img_len)?;

        // 7. Apply Attention Output Projection & Gated Residual (in F32 to preserve numerical dynamic range)
        let txt_attn_proj = self.txt_proj.forward(&txt_attn)?;
        let txt_gate1 = txt_gate1.unsqueeze(1)?;
        let txt_after_attn = (txt.to_dtype(candle_core::DType::F32)? + txt_attn_proj.to_dtype(candle_core::DType::F32)?.broadcast_mul(&txt_gate1.to_dtype(candle_core::DType::F32)?)?)?.clamp(-50000.0f32, 50000.0f32)?.to_dtype(orig_dtype)?;
        let txt = txt_after_attn.clone();

        let img_attn_proj = self.img_proj.forward(&img_attn)?;
        let img_gate1 = img_gate1.unsqueeze(1)?;
        let img = (img.to_dtype(candle_core::DType::F32)? + img_attn_proj.to_dtype(candle_core::DType::F32)?.broadcast_mul(&img_gate1.to_dtype(candle_core::DType::F32)?)?)?.to_dtype(orig_dtype)?;

        // 8. MLP Forward Passes with AdaLN-Zero Gating: (1 + scale2) * LayerNorm(x) + shift2
        let img_norm2 = norm_layer(&img)?;
        let img_scale2 = (img_scale2.unsqueeze(1)? + 1.0)?;
        let img_shift2 = img_shift2.unsqueeze(1)?;
        let img_normed2 = img_norm2.broadcast_mul(&img_scale2)?.broadcast_add(&img_shift2)?;
        let img_h1 = self.img_mlp.0.forward(&img_normed2)?;
        let img_mlp_h = if img_h1.dim(2)? > self.heads * self.head_dim * 4 {
            // SwiGLU activation for Klein (dim * 6 input, dim * 3 output)
            let mid_dim = img_h1.dim(2)? / 2;
            let gate = candle_nn::ops::silu(&img_h1.narrow(2, 0, mid_dim)?)?;
            let val = img_h1.narrow(2, mid_dim, mid_dim)?;
            (gate * val)?
        } else {
            gelu_tanh(&img_h1)?
        };
        let img_mlp_out = self.img_mlp.1.forward(&img_mlp_h)?;
        let img_gate2 = img_gate2.unsqueeze(1)?;
        let img = (img.to_dtype(candle_core::DType::F32)? + img_mlp_out.to_dtype(candle_core::DType::F32)?.broadcast_mul(&img_gate2.to_dtype(candle_core::DType::F32)?)?)?.to_dtype(orig_dtype)?;

        let txt_norm2 = norm_layer(&txt)?;
        let txt_scale2 = (txt_scale2.unsqueeze(1)? + 1.0)?;
        let txt_shift2 = txt_shift2.unsqueeze(1)?;
        let txt_normed2 = txt_norm2.broadcast_mul(&txt_scale2)?.broadcast_add(&txt_shift2)?;
        let txt_h1 = self.txt_mlp.0.forward(&txt_normed2)?;
        let txt_mlp_h = if txt_h1.dim(2)? > self.heads * self.head_dim * 4 {
            // SwiGLU activation for Klein (dim * 6 input, dim * 3 output)
            let mid_dim = txt_h1.dim(2)? / 2;
            let gate = candle_nn::ops::silu(&txt_h1.narrow(2, 0, mid_dim)?)?;
            let val = txt_h1.narrow(2, mid_dim, mid_dim)?;
            (gate * val)?
        } else {
            gelu_tanh(&txt_h1)?
        };
        let txt_mlp_out = self.txt_mlp.1.forward(&txt_mlp_h)?;
        let txt_gate2 = txt_gate2.unsqueeze(1)?;
        let txt = (txt.to_dtype(candle_core::DType::F32)? + txt_mlp_out.to_dtype(candle_core::DType::F32)?.broadcast_mul(&txt_gate2.to_dtype(candle_core::DType::F32)?)?)?.clamp(-50000.0f32, 50000.0f32)?.to_dtype(orig_dtype)?;

        if std::env::var("FLUX_TRACE").is_ok() {
            let rms = |t: &Tensor| -> f32 {
                let f = t.to_dtype(candle_core::DType::F32).unwrap().flatten_all().unwrap();
                if let Ok(v) = f.to_vec1::<f32>() {
                    let m = v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / v.len() as f64;
                    m.sqrt() as f32
                } else { 0.0 }
            };
            // Note txt_attn_proj / txt_mlp_out / txt are only in scope at this point; capture under names.
            eprintln!("      [DBLOCK] txt_mod_rms={:.3} txt_attn_rms={:.3} txt_attn_proj_rms={:.3} txt_mlp_out_rms={:.3} txt_after_attn={:.3} txt_out_rms={:.3}",
                rms(&txt_normed), rms(&txt_attn), rms(&txt_attn_proj), rms(&txt_mlp_out), rms(&txt_after_attn), rms(&txt));
        }

        Ok((img, txt))
    }

    fn standard_sdpa(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let (b, seq, h, d) = q.dims4()?;
        let orig_dtype = q.dtype();

        // 1. High precision F32 Attention computation to eliminate F16 exponent overflow (> 65504)
        let q_f32 = (q.transpose(1, 2)?.contiguous()?.to_dtype(candle_core::DType::F32)? * self.scale)?;
        let k_f32 = k.transpose(1, 2)?.contiguous()?.to_dtype(candle_core::DType::F32)?;
        let v_f32 = v.transpose(1, 2)?.contiguous()?.to_dtype(candle_core::DType::F32)?;

        let k_t = k_f32.transpose(2, 3)?.contiguous()?;
        let scores = q_f32.matmul(&k_t)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v_f32)?;

        ctx.transpose(1, 2)?.contiguous()?.to_dtype(orig_dtype)?.reshape((b, seq, h, d))
    }

    /// Scaled dot-product attention with an optional FlashAttention-2 fast path (F16/BF16 on CUDA).
    ///
    /// **Manette**: `FLUX_FLASH_ATTN` (default `0`).
    /// - `0` → always `standard_sdpa` (F32 fallback, model-safe, slower).
    /// - `1` → use `candle_flash_attn` when available, else fall back to `standard_sdpa`.
    ///
    /// Flash runs in the input dtype and is only taken on CUDA. Any error safely falls back.
    fn sdpa(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "flash-attn")]
        {
            let use_flash = std::env::var("FLUX_FLASH_ATTN").ok().map(|s| s == "1").unwrap_or(false);
            if use_flash && q.device().is_cuda() && (q.dtype() == candle_core::DType::F16 || q.dtype() == candle_core::DType::BF16) {
                let out = flash_attn(q, k, v, self.scale as f32, false);
                if let Ok(out) = out {
                    return Ok(out);
                }
            }
        }
        self.standard_sdpa(q, k, v)
    }
}

/// Unified Single-Stream Block (SingleStreamBlock) for concatenated sequences in Flux.1.
#[derive(Debug, Clone)]
pub struct SingleStreamBlock {
    pub linear1: Linear,
    pub linear2: Linear,
    pub modulation: AdaLNZeroModulation,
    pub q_norm: Option<RMSNorm>,
    pub k_norm: Option<RMSNorm>,
    pub heads: usize,
    pub head_dim: usize,
    pub scale: f64,
}

impl SingleStreamBlock {
    pub fn new(dim: usize, heads: usize, mlp_ratio: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = dim / heads;
        let scale = 1.0 / (head_dim as f64).sqrt();
        let mlp_dim = dim * mlp_ratio;
        let linear_layer = |in_d: usize, out_d: usize, path: VarBuilder| -> Result<Linear> {
            linear(in_d, out_d, path.clone()).or_else(|_| candle_nn::linear_no_bias(in_d, out_d, path))
        };

        // In Flux.1, linear1 projects from dim (3072) to 3*dim (Q,K,V: 9216) + mlp_dim (12288) = 21504
        // In Klein 4B, linear1 projects from 3072 to 27648 (3*dim + 6*dim = 9*dim = 27648)
        // In Klein 9B, linear1 projects from 4096 to 36864 (9 * 4096 = 36864)
        let swiglu_in_proj = dim * 3 + dim * 6; // dim * 9
        let swiglu_mid_in = dim + dim * 3;     // dim * 4
        let linear1 = linear_layer(dim, dim * 3 + mlp_dim, vb.pp("linear1"))
            .or_else(|_| linear_layer(dim, swiglu_in_proj, vb.pp("linear1")))?;
        let q_norm = RMSNorm::new(head_dim, vb.pp("norm.query_norm")).ok();
        let k_norm = RMSNorm::new(head_dim, vb.pp("norm.key_norm")).ok();

        // linear2 projects from dim (3072) + mlp_dim (12288) = 15360 back to dim (3072)
        // In Klein 4B, linear2 projects from 12288 back to 3072 (4 * 3072 = 12288)
        // In Klein 9B, linear2 projects from 16384 back to 4096 (4 * 4096 = 16384)
        let linear2 = linear_layer(dim + mlp_dim, dim, vb.pp("linear2"))
            .or_else(|_| linear_layer(swiglu_mid_in, dim, vb.pp("linear2")))?;
        let modulation = AdaLNZeroModulation::new(dim, dim * 3, vb.pp("modulation"))?;

        Ok(Self {
            linear1,
            linear2,
            modulation,
            q_norm,
            k_norm,
            heads,
            head_dim,
            scale,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        temb: &Tensor,
        freqs_cos: Option<&Tensor>,
        freqs_sin: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, seq, d) = x.dims3()?;
        let (shift, scale, gate) = self.modulation.modulate(temb)?;

        // LayerNorm(elementwise_affine=False) helper
        let orig_dtype = x.dtype();
        let x_f32 = x.to_dtype(candle_core::DType::F32)?;
        let mean = x_f32.mean_keepdim(x_f32.dims().len() - 1)?;
        let diff = x_f32.broadcast_sub(&mean)?;
        let var = diff.sqr()?.mean_keepdim(diff.dims().len() - 1)?;
        let std = (var + 1e-6)?.sqrt()?;
        let x_normed = diff.broadcast_div(&std)?.to_dtype(orig_dtype)?;

        let scale = (scale.unsqueeze(1)? + 1.0)?;
        let shift = shift.unsqueeze(1)?;
        let normed = x_normed.broadcast_mul(&scale)?.broadcast_add(&shift)?;

        // 1. Single-pass linear1 projection for Q, K, V and MLP
        let h1 = self.linear1.forward(&normed)?;
        let qkv = h1.narrow(2, 0, d * 3)?;
        let mlp_raw = h1.narrow(2, d * 3, h1.dim(2)? - d * 3)?;
        let mlp_h = if mlp_raw.dim(2)? > d * 4 {
            // SwiGLU for Klein SingleStreamBlock: dim * 6 -> dim * 3
            let mid_dim = mlp_raw.dim(2)? / 2;
            let gate = candle_nn::ops::silu(&mlp_raw.narrow(2, 0, mid_dim)?)?;
            let val = mlp_raw.narrow(2, mid_dim, mid_dim)?;
            (gate * val)?
        } else {
            gelu_tanh(&mlp_raw)?
        };

        let qkv = qkv.reshape((b, seq, 3, self.heads, self.head_dim))?;
        let mut q = qkv.narrow(2, 0, 1)?.squeeze(2)?;
        let mut k = qkv.narrow(2, 1, 1)?.squeeze(2)?;
        let v = qkv.narrow(2, 2, 1)?.squeeze(2)?;

        if let Some(ref qn) = self.q_norm {
            q = qn.forward(&q)?;
        }
        if let Some(ref kn) = self.k_norm {
            k = kn.forward(&k)?;
        }

        // Apply RoPE in SingleStreamBlock
        let (q, k) = if let (Some(cos), Some(sin)) = (freqs_cos, freqs_sin) {
            (
                crate::diffusion::dit::embeddings::apply_rope(&q, cos, sin)?,
                crate::diffusion::dit::embeddings::apply_rope(&k, cos, sin)?,
            )
        } else {
            (q, k)
        };

        // 2. Self-Attention (FlashAttention-2 fast path or F32 SDPA fallback)
        let attn_out = self.sdpa(&q, &k, &v)?;

        let attn_out = attn_out.reshape((b, seq, d))?;

        // 3. Unified Linear2 Output Projection
        let combined = Tensor::cat(&[&attn_out, &mlp_h], 2)?;
        let out = self.linear2.forward(&combined)?;

        // 4. Gated Residual in F32 with FP16 protection
        let gate = gate.unsqueeze(1)?;
        let res = (x.to_dtype(candle_core::DType::F32)? + out.to_dtype(candle_core::DType::F32)?.broadcast_mul(&gate.to_dtype(candle_core::DType::F32)?)?)?;
        res.clamp(-50000.0f32, 50000.0f32)?.to_dtype(orig_dtype)
    }

    fn standard_sdpa(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let (b, seq, h, d) = q.dims4()?;
        let orig_dtype = q.dtype();

        // 1. High precision F32 Attention computation to eliminate F16 exponent overflow (> 65504)
        let q_f32 = (q.transpose(1, 2)?.contiguous()?.to_dtype(candle_core::DType::F32)? * self.scale)?;
        let k_f32 = k.transpose(1, 2)?.contiguous()?.to_dtype(candle_core::DType::F32)?;
        let v_f32 = v.transpose(1, 2)?.contiguous()?.to_dtype(candle_core::DType::F32)?;

        let k_t = k_f32.transpose(2, 3)?.contiguous()?;
        let scores = q_f32.matmul(&k_t)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v_f32)?;

        ctx.transpose(1, 2)?.contiguous()?.to_dtype(orig_dtype)?.reshape((b, seq, h, d))
    }

    /// Scaled dot-product attention with an optional FlashAttention-2 fast path (F16/BF16 on CUDA).
    ///
    /// **Manette**: `FLUX_FLASH_ATTN` (default `0`).
    /// - `0` → always `standard_sdpa` (F32 fallback, model-safe, slower).
    /// - `1` → use `candle_flash_attn` when available, else fall back to `standard_sdpa`.
    ///
    /// Flash runs in the input dtype and is only taken on CUDA. Any error safely falls back.
    fn sdpa(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "flash-attn")]
        {
            let use_flash = std::env::var("FLUX_FLASH_ATTN").ok().map(|s| s == "1").unwrap_or(false);
            if use_flash && q.device().is_cuda() && (q.dtype() == candle_core::DType::F16 || q.dtype() == candle_core::DType::BF16) {
                let out = flash_attn(q, k, v, self.scale as f32, false);
                if let Ok(out) = out {
                    return Ok(out);
                }
            }
        }
        self.standard_sdpa(q, k, v)
    }
}
