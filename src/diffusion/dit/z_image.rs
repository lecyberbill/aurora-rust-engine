// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Pure Rust Z-Image Turbo DiT Architecture (30 Layers + Refiners)

use candle_core::{DType, Module, Result, Tensor};
use candle_nn::{linear, Linear, VarBuilder};
use crate::diffusion::dit::blocks::RMSNorm;

/// Z-Image Turbo Transformer Configuration
#[derive(Debug, Clone)]
pub struct ZImageConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub intermediate_dim: usize,
    pub cap_dim: usize,
    pub time_embed_dim: usize,
    pub theta: f64,
}

impl Default for ZImageConfig {
    fn default() -> Self {
        Self {
            in_channels: 64, // 16 VAE channels * 2x2 patchify
            out_channels: 64,
            hidden_size: 3840,
            num_heads: 30,
            num_kv_heads: 30,
            head_dim: 128, // 30 * 128 = 3840
            num_layers: 30,
            intermediate_dim: 10240,
            cap_dim: 2560, // Qwen3-4B hidden size
            time_embed_dim: 256,
            theta: 10000.0,
        }
    }
}

impl ZImageConfig {
    pub fn from_tensors(tensors: &std::collections::HashMap<String, Tensor>) -> Self {
        let mut cfg = Self::default();

        // 1. Detect hidden size & in_channels from first.weight or x_embedder.weight
        if let Some(t) = tensors.get("first.weight").or_else(|| tensors.get("x_embedder.weight")) {
            let dims = t.dims();
            if dims.len() >= 2 {
                cfg.hidden_size = dims[0];
                cfg.in_channels = dims[1];
                cfg.out_channels = dims[1];
                cfg.num_heads = cfg.hidden_size / cfg.head_dim;
                cfg.num_kv_heads = cfg.num_heads;
            }
        }

        // 1b. Detect num_kv_heads for Grouped-Query Attention (GQA) from wk.weight
        if let Some(t) = tensors.get("blocks.0.attn.wk.weight")
            .or_else(|| tensors.get("layers.0.attention.wk.weight")) {
            let dims = t.dims();
            if dims.len() >= 2 {
                cfg.num_kv_heads = dims[0] / cfg.head_dim;
            }
        }

        // 2. Detect intermediate_dim from mlp.gate.weight or feed_forward.w1.weight
        if let Some(t) = tensors.get("blocks.0.mlp.gate.weight")
            .or_else(|| tensors.get("layers.0.feed_forward.w1.weight"))
            .or_else(|| tensors.get("blocks.0.mlp.up.weight")) {
            let dims = t.dims();
            if dims.len() >= 2 {
                cfg.intermediate_dim = dims[0];
            }
        }

        // 3. Detect num_layers
        let mut max_layer = 0;
        for k in tensors.keys() {
            if let Some(rest) = k.strip_prefix("blocks.") {
                if let Some(idx_str) = rest.split('.').next() {
                    if let Ok(idx) = idx_str.parse::<usize>() {
                        if idx + 1 > max_layer {
                            max_layer = idx + 1;
                        }
                    }
                }
            } else if let Some(rest) = k.strip_prefix("layers.") {
                if let Some(idx_str) = rest.split('.').next() {
                    if let Ok(idx) = idx_str.parse::<usize>() {
                        if idx + 1 > max_layer {
                            max_layer = idx + 1;
                        }
                    }
                }
            }
        }
        if max_layer > 0 {
            cfg.num_layers = max_layer;
        }

        // 4. Detect cap_dim (Qwen3 text encoder hidden dim) from txtmlp / cap_embedder
        if let Some(t) = tensors.get("txtmlp.1.weight").or_else(|| tensors.get("cap_embedder.1.weight")) {
            let dims = t.dims();
            if dims.len() >= 2 {
                cfg.cap_dim = dims[1];
            }
        }

        cfg
    }
}

fn linear_or_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    linear(in_dim, out_dim, vb.clone())
        .or_else(|_| candle_nn::linear_no_bias(in_dim, out_dim, vb))
}

/// SwiGLU Feed-Forward Network for Z-Image / Krea2
#[derive(Debug, Clone)]
pub struct ZImageFeedForward {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl ZImageFeedForward {
    pub fn new(hidden_size: usize, intermediate_dim: usize, vb: VarBuilder) -> Result<Self> {
        let w1 = linear_or_no_bias(hidden_size, intermediate_dim, vb.pp("w1"))
            .or_else(|_| linear_or_no_bias(hidden_size, intermediate_dim, vb.pp("gate")))?;
        let w2 = linear_or_no_bias(intermediate_dim, hidden_size, vb.pp("w2"))
            .or_else(|_| linear_or_no_bias(intermediate_dim, hidden_size, vb.pp("down")))?;
        let w3 = linear_or_no_bias(hidden_size, intermediate_dim, vb.pp("w3"))
            .or_else(|_| linear_or_no_bias(hidden_size, intermediate_dim, vb.pp("up")))?;
        Ok(Self { w1, w2, w3 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x1 = self.w1.forward(x)?;
        let x3 = self.w3.forward(x)?;
        let gate = candle_nn::ops::silu(&x1)?;
        let activated = (gate * x3)?;
        self.w2.forward(&activated)
    }
}

#[derive(Debug, Clone)]
pub enum ZImageQkv {
    Merged(Linear),
    Separate { wq: Linear, wk: Linear, wv: Linear },
}

/// Self-Attention with QK-Norm, RoPE & GQA for Z-Image / Krea2
#[derive(Debug, Clone)]
pub struct ZImageAttention {
    qkv: ZImageQkv,
    out: Linear,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl ZImageAttention {
    pub fn new(hidden_size: usize, num_heads: usize, num_kv_heads: usize, head_dim: usize, vb: VarBuilder) -> Result<Self> {
        let qkv = if let Ok(merged) = linear_or_no_bias(hidden_size, hidden_size * 3, vb.pp("qkv")) {
            ZImageQkv::Merged(merged)
        } else {
            let wq = linear_or_no_bias(hidden_size, num_heads * head_dim, vb.pp("wq"))?;
            let wk = linear_or_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("wk"))?;
            let wv = linear_or_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("wv"))?;
            ZImageQkv::Separate { wq, wk, wv }
        };

        let out = linear_or_no_bias(hidden_size, hidden_size, vb.pp("out"))
            .or_else(|_| linear_or_no_bias(hidden_size, hidden_size, vb.pp("wo")))?;
        let q_norm = RMSNorm::new(head_dim, vb.pp("q_norm"))
            .or_else(|_| RMSNorm::new(head_dim, vb.pp("qknorm.qnorm")))?;
        let k_norm = RMSNorm::new(head_dim, vb.pp("k_norm"))
            .or_else(|_| RMSNorm::new(head_dim, vb.pp("qknorm.knorm")))?;
        let scale = 1.0 / (head_dim as f64).sqrt();

        Ok(Self {
            qkv,
            out,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        rotary_cos_sin: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        let (b, seq_len, _hidden) = x.dims3()?;
        let orig_dtype = x.dtype();

        let (q_raw, k_raw, v_raw) = match &self.qkv {
            ZImageQkv::Merged(linear) => {
                let qkv = linear.forward(x)?;
                let chunks = qkv.chunk(3, 2)?;
                (
                    chunks[0].reshape((b, seq_len, self.num_heads, self.head_dim))?,
                    chunks[1].reshape((b, seq_len, self.num_heads, self.head_dim))?,
                    chunks[2].reshape((b, seq_len, self.num_heads, self.head_dim))?,
                )
            }
            ZImageQkv::Separate { wq, wk, wv } => {
                let q_proj = wq.forward(x)?;
                let k_proj = wk.forward(x)?;
                let v_proj = wv.forward(x)?;
                (
                    q_proj.reshape((b, seq_len, self.num_heads, self.head_dim))?,
                    k_proj.reshape((b, seq_len, self.num_kv_heads, self.head_dim))?,
                    v_proj.reshape((b, seq_len, self.num_kv_heads, self.head_dim))?,
                )
            }
        };

        let mut q = self.q_norm.forward(&q_raw)?;
        let mut k = self.k_norm.forward(&k_raw)?;
        let mut v = v_raw;

        // Apply RoPE if provided
        if let Some((cos, sin)) = rotary_cos_sin {
            let half = self.head_dim / 2;
            let cos_f32 = cos.to_dtype(DType::F32)?;
            let sin_f32 = sin.to_dtype(DType::F32)?;
            let apply_rope = |t: &Tensor, heads: usize| -> Result<Tensor> {
                let t_f32 = t.to_dtype(DType::F32)?;
                let t_pairs = t_f32.reshape((b, seq_len, heads, half, 2))?;
                let t0 = t_pairs.narrow(4, 0, 1)?.squeeze(4)?;
                let t1 = t_pairs.narrow(4, 1, 1)?.squeeze(4)?;
                let neg_t1 = (t1 * -1.0)?.unsqueeze(4)?;
                let pos_t0 = t0.unsqueeze(4)?;
                let rotated = Tensor::cat(&[&neg_t1, &pos_t0], 4)?.reshape((b, seq_len, heads, self.head_dim))?;
                let out = (t_f32.broadcast_mul(&cos_f32)? + rotated.broadcast_mul(&sin_f32)?)?;
                out.to_dtype(orig_dtype)
            };
            q = apply_rope(&q, self.num_heads)?;
            k = apply_rope(&k, self.num_kv_heads)?;
        }

        // Expand KV heads for GQA if num_kv_heads < num_heads
        if self.num_kv_heads < self.num_heads {
            let n_rep = self.num_heads / self.num_kv_heads;
            k = k.unsqueeze(3)?.repeat((1, 1, 1, n_rep, 1))?.reshape((b, seq_len, self.num_heads, self.head_dim))?;
            v = v.unsqueeze(3)?.repeat((1, 1, 1, n_rep, 1))?.reshape((b, seq_len, self.num_heads, self.head_dim))?;
        }

        // Fast-path: FlashAttention-2 if feature enabled and activated (F16/BF16 on CUDA)
        #[cfg(feature = "flash-attn")]
        {
            let use_flash = std::env::var("AURORA_FLASH_ATTN")
                .or_else(|_| std::env::var("FLUX_FLASH_ATTN"))
                .or_else(|_| std::env::var("ZIMAGE_FLASH_ATTN"))
                .ok()
                .map(|s| s == "1")
                .unwrap_or(false);
            if use_flash && q.device().is_cuda() && (q.dtype() == DType::F16 || q.dtype() == DType::BF16) {
                let q_c = q.contiguous()?;
                let k_c = k.contiguous()?;
                let v_c = v.contiguous()?;
                if let Ok(attn_out) = candle_flash_attn::flash_attn(&q_c, &k_c, &v_c, self.scale as f32, false) {
                    let out = attn_out
                        .reshape((b, seq_len, self.num_heads * self.head_dim))?
                        .to_dtype(orig_dtype)?;
                    return self.out.forward(&out);
                }
            }
        }

        // Standard scaled dot product attention fallback (F32)
        let q_t = (q.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)? * self.scale)?;
        let k_t = k.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;
        let v_t = v.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;

        let attn_scores = q_t.matmul(&k_t.transpose(2, 3)?)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_scores)?;
        let attn_out = attn_weights.matmul(&v_t)?;

        let out = attn_out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, seq_len, self.num_heads * self.head_dim))?
            .to_dtype(orig_dtype)?;

        self.out.forward(&out)
    }
}

/// Z-Image DiT Transformer Layer (AdaLN-4 Modulation + Attention + SwiGLU FFN)
#[derive(Debug, Clone)]
pub struct ZImageBlock {
    pub ada_ln: Linear,
    attention_norm1: RMSNorm,
    attention: ZImageAttention,
    attention_norm2: RMSNorm,
    ffn_norm1: RMSNorm,
    feed_forward: ZImageFeedForward,
    ffn_norm2: RMSNorm,
    _hidden_size: usize,
}

impl ZImageBlock {
    pub fn new(cfg: &ZImageConfig, vb: VarBuilder) -> Result<Self> {
        let ada_ln = if let Ok(aln) = linear(cfg.time_embed_dim, cfg.hidden_size * 4, vb.pp("adaLN_modulation.0")) {
            aln
        } else if let Ok(aln) = linear_or_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("attn.gate")) {
            aln
        } else {
            linear(cfg.time_embed_dim, cfg.hidden_size * 4, vb.pp("attn.gate"))?
        };
        let attention_norm1 = RMSNorm::new(cfg.hidden_size, vb.pp("attention_norm1"))
            .or_else(|_| RMSNorm::new(cfg.hidden_size, vb.pp("prenorm")))?;
        let attention = ZImageAttention::new(cfg.hidden_size, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, vb.pp("attention"))
            .or_else(|_| ZImageAttention::new(cfg.hidden_size, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, vb.pp("attn")))?;
        let attention_norm2 = RMSNorm::new(cfg.hidden_size, vb.pp("attention_norm2"))
            .or_else(|_| RMSNorm::new(cfg.hidden_size, vb.pp("postnorm")))?;
        let ffn_norm1 = RMSNorm::new(cfg.hidden_size, vb.pp("ffn_norm1"))
            .or_else(|_| RMSNorm::new(cfg.hidden_size, vb.pp("prenorm")))?;
        let feed_forward = ZImageFeedForward::new(cfg.hidden_size, cfg.intermediate_dim, vb.pp("feed_forward"))
            .or_else(|_| ZImageFeedForward::new(cfg.hidden_size, cfg.intermediate_dim, vb.pp("mlp")))?;
        let ffn_norm2 = RMSNorm::new(cfg.hidden_size, vb.pp("ffn_norm2"))
            .or_else(|_| RMSNorm::new(cfg.hidden_size, vb.pp("postnorm")))?;

        Ok(Self {
            ada_ln,
            attention_norm1,
            attention,
            attention_norm2,
            ffn_norm1,
            feed_forward,
            ffn_norm2,
            _hidden_size: cfg.hidden_size,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        temb: &Tensor,
        rotary_cos_sin: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        let orig_dtype = x.dtype();
        let mod_params = self.ada_ln.forward(temb)?; // (z_image_modulation = True: no SiLU before adaLN)
        let (scale_msa, gate_msa, scale_mlp, gate_mlp) = if mod_params.dim(1)? == self._hidden_size * 4 {
            let chunks = mod_params.chunk(4, 1)?;
            (
                Some(chunks[0].unsqueeze(1)?),
                chunks[1].unsqueeze(1)?.tanh()?,
                Some(chunks[2].unsqueeze(1)?),
                chunks[3].unsqueeze(1)?.tanh()?,
            )
        } else {
            // mod_params is gate_msa [B, hidden_size]
            (
                None,
                mod_params.unsqueeze(1)?.tanh()?,
                None,
                Tensor::ones((temb.dim(0)?, 1, self._hidden_size), mod_params.dtype(), mod_params.device())?,
            )
        };

        // Attention path
        let ones = Tensor::ones((1, 1, 1), x.dtype(), x.device())?;
        let norm_x = self.attention_norm1.forward(x)?;
        let modulated_x = if let Some(ref scale) = scale_msa {
            let scale_p1 = scale.broadcast_add(&ones)?;
            norm_x.broadcast_mul(&scale_p1)?
        } else {
            norm_x
        };
        let attn_out = self.attention.forward(&modulated_x, rotary_cos_sin)?;
        let norm_attn = self.attention_norm2.forward(&attn_out)?;
        let x = x.broadcast_add(&norm_attn.broadcast_mul(&gate_msa)?)?;

        // FFN path
        let norm_x2 = self.ffn_norm1.forward(&x)?;
        let modulated_x2 = if let Some(ref scale) = scale_mlp {
            let scale_p1 = scale.broadcast_add(&ones)?;
            norm_x2.broadcast_mul(&scale_p1)?
        } else {
            norm_x2
        };
        let ffn_out = self.feed_forward.forward(&modulated_x2)?;
        let norm_ffn = self.ffn_norm2.forward(&ffn_out)?;
        let x = x.broadcast_add(&norm_ffn.broadcast_mul(&gate_mlp)?)?;

        x.to_dtype(orig_dtype)
    }
}

/// Z-Image Refiner Layer (Context Refiner & Noise Refiner)
#[derive(Debug, Clone)]
pub struct ZImageRefinerBlock {
    ada_ln: Option<Linear>,
    attention_norm1: RMSNorm,
    attention: ZImageAttention,
    attention_norm2: RMSNorm,
    ffn_norm1: RMSNorm,
    feed_forward: ZImageFeedForward,
    ffn_norm2: RMSNorm,
    _hidden_size: usize,
}

impl ZImageRefinerBlock {
    pub fn new(cfg: &ZImageConfig, vb: VarBuilder, has_adaln: bool) -> Result<Self> {
        let ada_ln = if has_adaln {
            if let Ok(aln) = linear(cfg.time_embed_dim, cfg.hidden_size * 4, vb.pp("adaLN_modulation.0")) {
                Some(aln)
            } else if let Ok(aln) = linear_or_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("attn.gate")) {
                Some(aln)
            } else {
                Some(linear(cfg.time_embed_dim, cfg.hidden_size * 4, vb.pp("attn.gate"))?)
            }
        } else {
            None
        };
        let attention_norm1 = RMSNorm::new(cfg.hidden_size, vb.pp("attention_norm1"))
            .or_else(|_| RMSNorm::new(cfg.hidden_size, vb.pp("prenorm")))?;
        let attention = ZImageAttention::new(cfg.hidden_size, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, vb.pp("attention"))
            .or_else(|_| ZImageAttention::new(cfg.hidden_size, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, vb.pp("attn")))?;
        let attention_norm2 = RMSNorm::new(cfg.hidden_size, vb.pp("attention_norm2"))
            .or_else(|_| RMSNorm::new(cfg.hidden_size, vb.pp("postnorm")))?;
        let ffn_norm1 = RMSNorm::new(cfg.hidden_size, vb.pp("ffn_norm1"))
            .or_else(|_| RMSNorm::new(cfg.hidden_size, vb.pp("prenorm")))?;
        let feed_forward = ZImageFeedForward::new(cfg.hidden_size, cfg.intermediate_dim, vb.pp("feed_forward"))
            .or_else(|_| ZImageFeedForward::new(cfg.hidden_size, cfg.intermediate_dim, vb.pp("mlp")))?;
        let ffn_norm2 = RMSNorm::new(cfg.hidden_size, vb.pp("ffn_norm2"))
            .or_else(|_| RMSNorm::new(cfg.hidden_size, vb.pp("postnorm")))?;

        Ok(Self {
            ada_ln,
            attention_norm1,
            attention,
            attention_norm2,
            ffn_norm1,
            feed_forward,
            ffn_norm2,
            _hidden_size: cfg.hidden_size,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        temb: Option<&Tensor>,
        rotary_cos_sin: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        let orig_dtype = x.dtype();

        if let (Some(ref aln), Some(t)) = (&self.ada_ln, temb) {
            let mod_params = aln.forward(t)?; // (z_image_modulation = True: no SiLU before adaLN)
            let (scale_msa, gate_msa, scale_mlp, gate_mlp) = if mod_params.dim(1)? == self._hidden_size * 4 {
                let chunks = mod_params.chunk(4, 1)?;
                (
                    Some(chunks[0].unsqueeze(1)?),
                    chunks[1].unsqueeze(1)?.tanh()?,
                    Some(chunks[2].unsqueeze(1)?),
                    chunks[3].unsqueeze(1)?.tanh()?,
                )
            } else {
                (
                    None,
                    mod_params.unsqueeze(1)?.tanh()?,
                    None,
                    Tensor::ones((t.dim(0)?, 1, self._hidden_size), mod_params.dtype(), mod_params.device())?,
                )
            };

            let ones = Tensor::ones((1, 1, 1), x.dtype(), x.device())?;
            let norm_x = self.attention_norm1.forward(x)?;
            let modulated_x = if let Some(ref scale) = scale_msa {
                let scale_p1 = scale.broadcast_add(&ones)?;
                norm_x.broadcast_mul(&scale_p1)?
            } else {
                norm_x
            };
            let attn_out = self.attention.forward(&modulated_x, rotary_cos_sin)?;
            let norm_attn = self.attention_norm2.forward(&attn_out)?;
            let x = x.broadcast_add(&norm_attn.broadcast_mul(&gate_msa)?)?;

            let norm_x2 = self.ffn_norm1.forward(&x)?;
            let modulated_x2 = if let Some(ref scale) = scale_mlp {
                let scale_p1 = scale.broadcast_add(&ones)?;
                norm_x2.broadcast_mul(&scale_p1)?
            } else {
                norm_x2
            };
            let ffn_out = self.feed_forward.forward(&modulated_x2)?;
            let norm_ffn = self.ffn_norm2.forward(&ffn_out)?;
            let x = x.broadcast_add(&norm_ffn.broadcast_mul(&gate_mlp)?)?;
            x.to_dtype(orig_dtype)
        } else {
            let norm_x = self.attention_norm1.forward(x)?;
            let attn_out = self.attention.forward(&norm_x, rotary_cos_sin)?;
            let norm_attn = self.attention_norm2.forward(&attn_out)?;
            let x = (x + norm_attn)?;

            let norm_x2 = self.ffn_norm1.forward(&x)?;
            let ffn_out = self.feed_forward.forward(&norm_x2)?;
            let norm_ffn = self.ffn_norm2.forward(&ffn_out)?;
            let x = (x + norm_ffn)?;
            x.to_dtype(orig_dtype)
        }
    }
}

/// Timestep Embedder for Z-Image
#[derive(Debug, Clone)]
pub struct ZImageTimestepEmbedder {
    pub mlp_0: Linear,
    pub mlp_2: Linear,
    pub frequency_embedding_size: usize,
}

impl ZImageTimestepEmbedder {
    pub fn new(time_embed_dim: usize, hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let (mlp_0, mlp_2) = if let Ok(l0) = linear(time_embed_dim, 1024, vb.pp("mlp.0")) {
            let l2 = linear(1024, time_embed_dim, vb.pp("mlp.2"))?;
            (l0, l2)
        } else if let Ok(l0) = linear(time_embed_dim, hidden_size, vb.pp("0")) {
            let l2 = linear(hidden_size, hidden_size, vb.pp("2"))
                .or_else(|_| linear(hidden_size, time_embed_dim, vb.pp("2")))?;
            (l0, l2)
        } else {
            let l0 = linear(time_embed_dim, 1024, vb.pp("0"))
                .or_else(|_| linear(time_embed_dim, hidden_size, vb.pp("mlp.0")))?;
            let l2 = linear(1024, time_embed_dim, vb.pp("2"))
                .or_else(|_| linear(hidden_size, hidden_size, vb.pp("mlp.2")))?;
            (l0, l2)
        };
        Ok(Self {
            mlp_0,
            mlp_2,
            frequency_embedding_size: time_embed_dim,
        })
    }

    pub fn forward(&self, t: &Tensor) -> Result<Tensor> {
        let orig_dtype = t.dtype();
        let half = self.frequency_embedding_size / 2;
        let max_period = 10000.0f32;
        let freqs: Vec<f32> = (0..half)
            .map(|i| (-max_period.ln() * (i as f32) / (half as f32)).exp())
            .collect();
        let freqs_t = Tensor::from_vec(freqs, (1, half), t.device())?;
        let t_2d = t.to_dtype(DType::F32)?.reshape((t.dims()[0], 1))?;
        let args = t_2d.matmul(&freqs_t)?;
        let sin = args.sin()?;
        let cos = args.cos()?;
        let emb = Tensor::cat(&[&cos, &sin], 1)?.to_dtype(orig_dtype)?;

        let x = candle_nn::ops::silu(&self.mlp_0.forward(&emb)?)?;
        self.mlp_2.forward(&x)
    }
}

/// LayerNorm without learned affine weights (elementwise_affine=False, eps=1e-6)
fn layer_norm_no_affine(x: &Tensor, eps: f64) -> Result<Tensor> {
    let orig_dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let mean = x_f32.mean_keepdim(candle_core::D::Minus1)?;
    let x_sub_mean = x_f32.broadcast_sub(&mean)?;
    let var = x_sub_mean.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
    let std = (var + eps)?.sqrt()?;
    let norm = x_sub_mean.broadcast_div(&std)?;
    norm.to_dtype(orig_dtype)
}

/// Caption Embedder (Qwen3 -> DiT Hidden Dim Projection)
#[derive(Debug, Clone)]
pub struct ZImageCaptionEmbedder {
    norm: RMSNorm,
    proj: Linear,
}

impl ZImageCaptionEmbedder {
    pub fn new(cfg: &ZImageConfig, vb: VarBuilder) -> Result<Self> {
        let norm = RMSNorm::new(cfg.cap_dim, vb.pp("cap_embedder.0"))
            .or_else(|_| RMSNorm::new(cfg.cap_dim, vb.pp("txtmlp.0")))?;
        let proj = linear(cfg.cap_dim, cfg.hidden_size, vb.pp("cap_embedder.1"))
            .or_else(|_| linear(cfg.cap_dim, cfg.hidden_size, vb.pp("txtmlp.1")))?;
        Ok(Self { norm, proj })
    }

    pub fn forward(&self, context: &Tensor) -> Result<Tensor> {
        let norm_ctx = self.norm.forward(context)?;
        self.proj.forward(&norm_ctx)
    }
}

/// Final Layer of NextDiT / Z-Image:
/// LayerNorm(elementwise_affine=False, eps=1e-6) -> AdaLN modulation (1 + scale) -> Linear(hidden_size, 64)
#[derive(Debug, Clone)]
pub struct ZImageFinalLayer {
    pub ada_ln: Linear,
    pub linear: Linear,
}

impl ZImageFinalLayer {
    pub fn new(cfg: &ZImageConfig, vb: VarBuilder) -> Result<Self> {
        let ada_ln = linear(cfg.time_embed_dim, cfg.hidden_size, vb.pp("adaLN_modulation.1"))
            .or_else(|_| linear_or_no_bias(cfg.hidden_size, 2, vb.pp("modulation.lin")))
            .or_else(|_| linear_or_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("modulation.lin")))?;
        let linear = linear(cfg.hidden_size, cfg.out_channels, vb.pp("linear"))
            .or_else(|_| linear_or_no_bias(cfg.hidden_size, cfg.out_channels, vb.clone()))?;
        Ok(Self { ada_ln, linear })
    }

    pub fn forward(&self, x: &Tensor, temb: &Tensor) -> Result<Tensor> {
        let orig_dtype = x.dtype();
        let norm_x = layer_norm_no_affine(x, 1e-6)?;
        let act_t = candle_nn::ops::silu(temb)?;
        let mod_out = self.ada_ln.forward(&act_t)?;
        let modulated = if mod_out.dim(1)? == 2 {
            let chunks = mod_out.chunk(2, 1)?;
            let scale = chunks[0].unsqueeze(1)?;
            let shift = chunks[1].unsqueeze(1)?;
            let ones = Tensor::ones((1, 1, 1), scale.dtype(), scale.device())?;
            let scale_p1 = scale.broadcast_add(&ones)?;
            norm_x.broadcast_mul(&scale_p1)?.broadcast_add(&shift)?
        } else {
            let scale = mod_out.unsqueeze(1)?;
            let ones = Tensor::ones((1, 1, 1), scale.dtype(), scale.device())?;
            let scale_p1 = scale.broadcast_add(&ones)?;
            norm_x.broadcast_mul(&scale_p1)?
        };
        let out = self.linear.forward(&modulated)?;
        out.to_dtype(orig_dtype)
    }
}

/// Complete Z-Image Turbo Diffusion Transformer
#[derive(Debug, Clone)]
pub struct ZImageTransformer {
    pub config: ZImageConfig,
    pub x_embedder: Linear,
    pub x_pad_token: Tensor,
    pub cap_pad_token: Tensor,
    pub t_embedder: ZImageTimestepEmbedder,
    pub cap_embedder: ZImageCaptionEmbedder,
    pub context_refiner: Vec<ZImageRefinerBlock>,
    pub layers: Vec<ZImageBlock>,
    pub noise_refiner: Vec<ZImageRefinerBlock>,
    pub final_layer: ZImageFinalLayer,
}

impl ZImageTransformer {
    pub fn new(cfg: ZImageConfig, vb: VarBuilder) -> Result<Self> {
        let x_embedder = linear(cfg.in_channels, cfg.hidden_size, vb.pp("x_embedder"))
            .or_else(|_| linear(cfg.in_channels, cfg.hidden_size, vb.pp("first")))?;
        let x_pad_token = vb.get((1, cfg.hidden_size), "x_pad_token")
            .unwrap_or_else(|_| Tensor::zeros((1, cfg.hidden_size), vb.dtype(), vb.device()).unwrap());
        let cap_pad_token = vb.get((1, cfg.hidden_size), "cap_pad_token")
            .unwrap_or_else(|_| Tensor::zeros((1, cfg.hidden_size), vb.dtype(), vb.device()).unwrap());
        let t_embedder = ZImageTimestepEmbedder::new(cfg.time_embed_dim, cfg.hidden_size, vb.pp("t_embedder"))
            .or_else(|_| ZImageTimestepEmbedder::new(cfg.time_embed_dim, cfg.hidden_size, vb.pp("tmlp")))?;
        let cap_embedder = ZImageCaptionEmbedder::new(&cfg, vb.clone())?;

        let mut context_refiner = Vec::with_capacity(2);
        for i in 0..2 {
            if let Ok(refiner) = ZImageRefinerBlock::new(&cfg, vb.pp(format!("context_refiner.{}", i)), false) {
                context_refiner.push(refiner);
            } else if let Ok(refiner) = ZImageRefinerBlock::new(&cfg, vb.pp(format!("txtfusion.refiner_blocks.{}", i)), false) {
                context_refiner.push(refiner);
            }
        }

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let layer = ZImageBlock::new(&cfg, vb.pp(format!("layers.{}", i)))
                .or_else(|_| ZImageBlock::new(&cfg, vb.pp(format!("blocks.{}", i))))?;
            layers.push(layer);
        }

        let mut noise_refiner = Vec::with_capacity(2);
        for i in 0..2 {
            if let Ok(refiner) = ZImageRefinerBlock::new(&cfg, vb.pp(format!("noise_refiner.{}", i)), true) {
                noise_refiner.push(refiner);
            }
        }

        let final_layer = ZImageFinalLayer::new(&cfg, vb.pp("final_layer"))
            .or_else(|_| ZImageFinalLayer::new(&cfg, vb.pp("last")))?;

        Ok(Self {
            config: cfg,
            x_embedder,
            x_pad_token,
            cap_pad_token,
            t_embedder,
            cap_embedder,
            context_refiner,
            layers,
            noise_refiner,
            final_layer,
        })
    }

    /// Forward pass: input latents [B, C, H, W], timestep [B] (normalized sigma), text context [B, L, D]
    pub fn forward(
        &self,
        latents: &Tensor,
        timestep: &Tensor,
        context: &Tensor,
    ) -> Result<Tensor> {
        let (b, c, h, w) = latents.dims4()?;
        let p_h = h / 2;
        let p_w = w / 2;
        let n_img = p_h * p_w;
        let pad_multiple = 32usize;

        // 1. Exact Lumina2 / ComfyUI Patchify:
        // x.view(B, C, H // 2, 2, W // 2, 2).permute(0, 2, 4, 3, 5, 1).flatten(3).flatten(1, 2)
        let x_patch = latents
            .reshape((b, c, p_h, 2, p_w, 2))?
            .permute((0, 2, 4, 3, 5, 1))?
            .contiguous()?
            .reshape((b, n_img, c * 4))?;

        let mut x_proj = self.x_embedder.forward(&x_patch)?;
        let img_pad_extra = (pad_multiple - (n_img % pad_multiple)) % pad_multiple;
        if img_pad_extra > 0 {
            let x_pad = self.x_pad_token
                .to_dtype(x_proj.dtype())?
                .to_device(x_proj.device())?
                .unsqueeze(0)?
                .repeat((b, img_pad_extra, 1))?;
            x_proj = Tensor::cat(&[&x_proj, &x_pad], 1)?;
        }
        let total_img_tokens = x_proj.dim(1)?;

        // 2. Timestep embedding (Lumina: t = (1.0 - sigma) * 1000.0)
        let temb = self.t_embedder.forward(timestep)?;

        // 3. Caption context embedding & padding to multiple of 32
        let raw_text_feat = self.cap_embedder.forward(context)?;
        let cap_feats_len = raw_text_feat.dim(1)?;
        let cap_pad_extra = (pad_multiple - (cap_feats_len % pad_multiple)) % pad_multiple;
        let mut text_feat = if cap_pad_extra > 0 {
            let cap_pad = self.cap_pad_token
                .to_dtype(raw_text_feat.dtype())?
                .to_device(raw_text_feat.device())?
                .unsqueeze(0)?
                .repeat((b, cap_pad_extra, 1))?;
            Tensor::cat(&[&raw_text_feat, &cap_pad], 1)?
        } else {
            raw_text_feat
        };
        let total_text_tokens = text_feat.dim(1)?;

        let theta = 256.0f64;
        let axes_dim = [32, 48, 48];

        // 3a. Context RoPE for context_refiner: pos_ids = [1.0 + i, 0, 0]
        let mut txt_t = Vec::with_capacity(total_text_tokens);
        let mut txt_zero = Vec::with_capacity(total_text_tokens);
        for i in 0..total_text_tokens {
            txt_t.push((i as f32) + 1.0);
            txt_zero.push(0f32);
        }

        let compute_rope = |t_coords: &[f32], r_coords: &[f32], c_coords: &[f32], seq_len: usize| -> Result<(Tensor, Tensor)> {
            let compute_axis = |coords: &[f32], dim: usize| -> Result<Tensor> {
                let half = dim / 2;
                let inv_freq: Vec<f32> = (0..half)
                    .map(|i| 1.0 / (theta.powf((i * 2) as f64 / dim as f64) as f32))
                    .collect();
                let inv_freq_t = Tensor::from_vec(inv_freq, (half,), latents.device())?;
                let coords_t = Tensor::from_vec(coords.to_vec(), (seq_len, 1), latents.device())?;
                coords_t.matmul(&inv_freq_t.unsqueeze(0)?)
            };
            let f0 = compute_axis(t_coords, axes_dim[0])?;
            let f1 = compute_axis(r_coords, axes_dim[1])?;
            let f2 = compute_axis(c_coords, axes_dim[2])?;
            let full = Tensor::cat(&[&f0, &f1, &f2], 1)?; // [seq_len, 64]
            let cos_half = full.cos()?;
            let sin_half = full.sin()?;
            // Interleaved complex representation for pairs (x0, x1) -> cos_half on both x0 and x1
            let cos = Tensor::cat(&[&cos_half.unsqueeze(2)?, &cos_half.unsqueeze(2)?], 2)?
                .reshape((1, seq_len, 1, 128))?
                .to_dtype(latents.dtype())?;
            let sin = Tensor::cat(&[&sin_half.unsqueeze(2)?, &sin_half.unsqueeze(2)?], 2)?
                .reshape((1, seq_len, 1, 128))?
                .to_dtype(latents.dtype())?;
            Ok((cos, sin))
        };

        let (txt_cos, txt_sin) = compute_rope(&txt_t, &txt_zero, &txt_zero, total_text_tokens)?;
        let txt_rope = (&txt_cos, &txt_sin);
        for refiner in &self.context_refiner {
            text_feat = refiner.forward(&text_feat, None, Some(txt_rope))?;
        }

        // 4. Noise Refiner on image tokens BEFORE backbone
        // Image pos_ids: t = total_text_tokens + 1, r = 0..p_h, c = 0..p_w, padded tokens = 0
        let start_t = (total_text_tokens as f32) + 1.0;
        let mut img_t = Vec::with_capacity(total_img_tokens);
        let mut img_r = Vec::with_capacity(total_img_tokens);
        let mut img_c = Vec::with_capacity(total_img_tokens);
        for r in 0..p_h {
            for c in 0..p_w {
                img_t.push(start_t);
                img_r.push(r as f32);
                img_c.push(c as f32);
            }
        }
        for _ in 0..img_pad_extra {
            img_t.push(0f32);
            img_r.push(0f32);
            img_c.push(0f32);
        }

        let (img_cos, img_sin) = compute_rope(&img_t, &img_r, &img_c, total_img_tokens)?;
        let img_rope = (&img_cos, &img_sin);
        let mut x_img = x_proj;
        for refiner in &self.noise_refiner {
            x_img = refiner.forward(&x_img, Some(&temb), Some(img_rope))?;
        }

        // 5. Concatenate refined image + refined text tokens: [B, total_img + total_text, Hidden]
        // In official diffusers & Lumina2: unified = torch.cat([x[i][:x_len], cap_feats[i][:cap_len]])
        // Image tokens are FIRST, caption tokens are SECOND
        let mut x_seq = Tensor::cat(&[&x_img, &text_feat], 1)?;
        let total_tokens = total_img_tokens + total_text_tokens;

        let mut all_t = img_t;
        all_t.extend(txt_t);
        let mut all_r = img_r;
        all_r.extend(txt_zero);
        let mut all_c = img_c;
        for _ in 0..total_text_tokens { all_c.push(0f32); }

        let (all_cos, all_sin) = compute_rope(&all_t, &all_r, &all_c, total_tokens)?;
        let all_rope = (&all_cos, &all_sin);

        // 6. DiT Backbone Layers (0..29)
        for layer in &self.layers {
            x_seq = layer.forward(&x_seq, &temb, Some(all_rope))?;
        }

        // 7. Extract unpadded image token slice (first n_img tokens)
        let x_img_out = x_seq.narrow(1, 0, n_img)?;

        // 8. Output projection: final_layer (LayerNorm unweighted + AdaLN(1+scale) + Linear)
        let out_patch = self.final_layer.forward(&x_img_out, &temb)?;

        // 9. Exact Lumina2 / ComfyUI Unpatchify:
        // out_patch.view(H // 2, W // 2, 2, 2, C).permute(4, 0, 2, 1, 3).flatten(3, 4).flatten(1, 2)
        // In Candle tensor layout: (B, p_h, p_w, 2, 2, C) -> permute(0, 5, 1, 3, 2, 4) -> reshape(B, C, H, W)
        let out_latents = out_patch
            .reshape((b, p_h, p_w, 2, 2, c))?
            .permute((0, 5, 1, 3, 2, 4))?
            .contiguous()?
            .reshape((b, c, h, w))?;

        // Return model velocity prediction for Flow Matching Euler ODE
        Ok(out_latents)
    }
}
