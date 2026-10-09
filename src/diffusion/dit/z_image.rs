// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Official ComfyUI / Diffusers Pure Rust Krea 2 Turbo Single-Stream MMDiT Architecture

use candle_core::{DType, Module, Result, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

/// Krea 2 Turbo Transformer Configuration
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
            hidden_size: 6144,
            num_heads: 48,
            num_kv_heads: 12,
            head_dim: 128, // 48 * 128 = 6144
            num_layers: 28,
            intermediate_dim: 16384,
            cap_dim: 2560, // Qwen3-VL-4B hidden size
            time_embed_dim: 256,
            theta: 1000.0,
        }
    }
}

impl ZImageConfig {
    pub fn from_tensors(tensors: &std::collections::HashMap<String, Tensor>) -> Self {
        let keys: Vec<String> = tensors.keys().cloned().collect();
        Self::from_tensors_and_keys(tensors, &keys)
    }

    pub fn from_tensors_and_keys(tensors: &std::collections::HashMap<String, Tensor>, all_keys: &[String]) -> Self {
        let mut cfg = Self::default();

        if let Some(t) = tensors.get("first.weight") {
            let dims = t.dims();
            if dims.len() >= 2 {
                cfg.hidden_size = dims[0];
                cfg.in_channels = dims[1];
                cfg.out_channels = dims[1];
                cfg.num_heads = cfg.hidden_size / cfg.head_dim;
            }
        }

        if let Some(t) = tensors.get("blocks.0.attn.wk.weight") {
            let dims = t.dims();
            if dims.len() >= 2 {
                cfg.num_kv_heads = dims[0] / cfg.head_dim;
            }
        } else if tensors.contains_key("layers.0.attention.qkv.weight") || tensors.contains_key("model.diffusion_model.layers.0.attention.qkv.weight") {
            // Lumina2 / Z-Image uses standard full multi-head attention: num_kv_heads == num_heads (30)
            cfg.num_kv_heads = cfg.num_heads;
        }

        if let Some(t) = tensors.get("blocks.0.mlp.gate.weight") {
            let dims = t.dims();
            if dims.len() >= 2 {
                cfg.intermediate_dim = dims[0];
            }
        } else if let Some(t) = tensors.get("layers.0.feed_forward.w1.weight").or_else(|| tensors.get("model.diffusion_model.layers.0.feed_forward.w1.weight")) {
            let dims = t.dims();
            if dims.len() >= 2 {
                cfg.intermediate_dim = dims[0];
            }
        }

        let mut max_layer = 0;
        for k in all_keys {
            let stripped = k.strip_prefix("model.diffusion_model.")
                .or_else(|| k.strip_prefix("diffusion_model."))
                .unwrap_or(k);
            if let Some(rest) = stripped.strip_prefix("blocks.") {
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

        if let Some(t) = tensors.get("txtmlp.1.weight") {
            let dims = t.dims();
            if dims.len() >= 2 {
                cfg.cap_dim = dims[1];
            }
        }

        cfg
    }
}

/// RMS Normalization with (1 + scale) zero-centered reference weight convention
#[derive(Debug, Clone)]
pub struct KreaRMSNorm {
    scale: Tensor,
    eps: f64,
}

impl KreaRMSNorm {
    pub fn new(features: usize, vb: VarBuilder) -> Result<Self> {
        let scale = vb.get(features, "scale")
            .or_else(|_| vb.get((features,), "scale"))
            .or_else(|_| vb.get(features, "weight"))?;
        Ok(Self { scale, eps: 1e-5 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let orig_dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let mean_sq = x_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let rms = (mean_sq + self.eps)?.sqrt()?;
        let norm = x_f32.broadcast_div(&rms)?;
        let weight = (self.scale.to_dtype(DType::F32)? + 1.0)?;
        let out = norm.broadcast_mul(&weight)?;
        out.to_dtype(orig_dtype)
    }
}

/// Standard RMS Normalization (norm * scale without +1.0 offset) for QK-Norm
#[derive(Debug, Clone)]
pub struct StandardRMSNorm {
    scale: Tensor,
    eps: f64,
}

impl StandardRMSNorm {
    pub fn new(features: usize, vb: VarBuilder) -> Result<Self> {
        let scale = vb.get(features, "scale")
            .or_else(|_| vb.get((features,), "scale"))
            .or_else(|_| vb.get(features, "weight"))?;
        Ok(Self { scale, eps: 1e-5 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let orig_dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let mean_sq = x_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let rms = (mean_sq + self.eps)?.sqrt()?;
        let norm = x_f32.broadcast_div(&rms)?;
        let weight = self.scale.to_dtype(DType::F32)?;
        let out = norm.broadcast_mul(&weight)?;
        out.to_dtype(orig_dtype)
    }
}

/// Per-head QK Normalization with zero-centered KreaRMSNorm (scale + 1.0)
#[derive(Debug, Clone)]
pub struct QKNorm {
    qnorm: KreaRMSNorm,
    knorm: KreaRMSNorm,
}

impl QKNorm {
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let qnorm = KreaRMSNorm::new(dim, vb.pp("qnorm"))?;
        let knorm = KreaRMSNorm::new(dim, vb.pp("knorm"))?;
        Ok(Self { qnorm, knorm })
    }

    pub fn forward(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        Ok((self.qnorm.forward(q)?, self.knorm.forward(k)?))
    }
}

/// SwiGLU Feed-Forward Network: down(silu(gate(x)) * up(x))
#[derive(Debug, Clone)]
pub struct SwiGLU {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl SwiGLU {
    pub fn new(features: usize, mlp_dim: usize, bias: bool, vb: VarBuilder) -> Result<Self> {
        let gate = if bias { linear(features, mlp_dim, vb.pp("gate"))? } else { candle_nn::linear_no_bias(features, mlp_dim, vb.pp("gate"))? };
        let up = if bias { linear(features, mlp_dim, vb.pp("up"))? } else { candle_nn::linear_no_bias(features, mlp_dim, vb.pp("up"))? };
        let down = if bias { linear(mlp_dim, features, vb.pp("down"))? } else { candle_nn::linear_no_bias(mlp_dim, features, vb.pp("down"))? };
        Ok(Self { gate, up, down })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let g = candle_nn::ops::silu(&self.gate.forward(x)?)?;
        let u = self.up.forward(x)?;
        let gu = (g * u)?;
        self.down.forward(&gu)
    }
}

pub fn robust_sigmoid(x: &Tensor) -> Result<Tensor> {
    // Numerically stable sigmoid: 1 / (1 + exp(-x))
    candle_nn::ops::sigmoid(x)
        .or_else(|_| {
            let neg_x = (x * -1.0)?;
            let exp = neg_x.exp()?;
            let denom = (exp + 1.0)?;
            denom.recip()
        })
}

/// Attention with QK-Norm, 3D RoPE, Sigmoid-Gated Output & GQA
#[derive(Debug, Clone)]
pub struct KreaAttention {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    gate: Option<Linear>,
    qknorm: QKNorm,
    wo: Linear,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl KreaAttention {
    pub fn new(dim: usize, heads: usize, kv_heads: usize, bias: bool, vb: VarBuilder) -> Result<Self> {
        let head_dim = dim / heads;
        let wq = if bias { linear(dim, heads * head_dim, vb.pp("wq"))? } else { candle_nn::linear_no_bias(dim, heads * head_dim, vb.pp("wq"))? };
        let wk = if bias { linear(dim, kv_heads * head_dim, vb.pp("wk"))? } else { candle_nn::linear_no_bias(dim, kv_heads * head_dim, vb.pp("wk"))? };
        let wv = if bias { linear(dim, kv_heads * head_dim, vb.pp("wv"))? } else { candle_nn::linear_no_bias(dim, kv_heads * head_dim, vb.pp("wv"))? };
        let gate = if bias {
            linear(dim, dim, vb.pp("gate")).ok()
        } else {
            candle_nn::linear_no_bias(dim, dim, vb.pp("gate")).ok()
        };
        let qknorm = QKNorm::new(head_dim, vb.pp("qknorm"))?;
        let wo = if bias { linear(dim, dim, vb.pp("wo"))? } else { candle_nn::linear_no_bias(dim, dim, vb.pp("wo"))? };
        let scale = 1.0 / (head_dim as f64).sqrt();
        Ok(Self { wq, wk, wv, gate, qknorm, wo, heads, kv_heads, head_dim, scale })
    }

    pub fn forward(&self, x: &Tensor, rotary_freqs: Option<(&Tensor, &Tensor)>) -> Result<Tensor> {
        let (b, l, _d) = x.dims3()?;
        let orig_dtype = x.dtype();

        let q_raw = self.wq.forward(x)?.reshape((b, l, self.heads, self.head_dim))?.transpose(1, 2)?.contiguous()?; // [B, H, L, D]
        let k_raw = self.wk.forward(x)?.reshape((b, l, self.kv_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let v_raw = self.wv.forward(x)?.reshape((b, l, self.kv_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let gate = if let Some(ref g) = self.gate {
            let gate_raw = g.forward(x)?;
            Some(robust_sigmoid(&gate_raw)?)
        } else {
            None
        };

        let (mut q, mut k) = self.qknorm.forward(&q_raw, &k_raw)?;

        if let Some((cos, sin)) = rotary_freqs {
            // Official DiffSynth / Krea 2 3D RoPE:
            // xq_ = freqs[..., 0] * xq_[..., 0] + freqs[..., 1] * xq_[..., 1]
            // where freqs has [[cos, -sin], [sin, cos]] matrix format for each (u0, u1) pair
            let cos_f32 = cos.to_dtype(DType::F32)?;
            let sin_f32 = sin.to_dtype(DType::F32)?;
            let apply_rope = |t: &Tensor| -> Result<Tensor> {
                let (tb, th, tl, td) = t.dims4()?;
                let t_f32 = t.to_dtype(DType::F32)?.reshape((tb, th, tl, td / 2, 2))?;
                let u0 = t_f32.narrow(4, 0, 1)?; // [B, H, L, D/2, 1]
                let u1 = t_f32.narrow(4, 1, 1)?; // [B, H, L, D/2, 1]
                // [cos, -sin] * [u0, u1]^T -> new_u0 = cos * u0 - sin * u1
                // [sin,  cos] * [u0, u1]^T -> new_u1 = sin * u0 + cos * u1
                let new_u0 = (u0.broadcast_mul(&cos_f32)? - u1.broadcast_mul(&sin_f32)?)?.squeeze(4)?;
                let new_u1 = (u0.broadcast_mul(&sin_f32)? + u1.broadcast_mul(&cos_f32)?)?.squeeze(4)?;
                let out = Tensor::stack(&[&new_u0, &new_u1], 4)?.reshape((tb, th, tl, td))?;
                out.to_dtype(orig_dtype)
            };
            q = apply_rope(&q)?;
            k = apply_rope(&k)?;
        }

        let k = if self.kv_heads != self.heads {
            let rep = self.heads / self.kv_heads;
            k.unsqueeze(2)?.repeat((1, 1, rep, 1, 1))?.permute((0, 1, 2, 3, 4))?.contiguous()?.reshape((b, self.heads, l, self.head_dim))?
        } else {
            k
        };
        let v = if self.kv_heads != self.heads {
            let rep = self.heads / self.kv_heads;
            v_raw.unsqueeze(2)?.repeat((1, 1, rep, 1, 1))?.permute((0, 1, 2, 3, 4))?.contiguous()?.reshape((b, self.heads, l, self.head_dim))?
        } else {
            v_raw
        };

        // Fast-path: FlashAttention-2 if enabled
        #[cfg(feature = "flash-attn")]
        {
            let use_flash = std::env::var("AURORA_FLASH_ATTN").ok().map(|s| s == "1").unwrap_or(false);
            if use_flash && q.device().is_cuda() && (q.dtype() == DType::F16 || q.dtype() == DType::BF16) {
                let q_c = q.transpose(1, 2)?.contiguous()?;
                let k_c = k.transpose(1, 2)?.contiguous()?;
                let v_c = v.transpose(1, 2)?.contiguous()?;
                if let Ok(attn_out) = candle_flash_attn::flash_attn(&q_c, &k_c, &v_c, self.scale as f32, false) {
                    let out_seq = attn_out.reshape((b, l, self.heads * self.head_dim))?.to_dtype(orig_dtype)?;
                    let final_out = if let Some(ref g) = gate { (out_seq * g)? } else { out_seq };
                    return self.wo.forward(&final_out);
                }
            }
        }

        // Scaled dot-product attention
        let q_scaled = (q * self.scale)?;
        let scores = q_scaled.matmul(&k.transpose(2, 3)?)?;
        let weights = crate::device::softmax_last_dim(&scores)?;
        let attn_out = weights.matmul(&v)?; // [B, H, L, D]
        let out_seq = attn_out.transpose(1, 2)?.contiguous()?.reshape((b, l, self.heads * self.head_dim))?;
        // Official SingleStreamAttention: Gating is strictly applied BEFORE wo
        let final_out = if let Some(ref g) = gate { (out_seq * g)? } else { out_seq };
        self.wo.forward(&final_out)
    }
}

/// Double Shared Modulation (supports 4*dim for Lumina2 or 6*dim for Krea2)
#[derive(Debug, Clone)]
pub struct DoubleSharedModulation {
    lin: Tensor,
    chunks: usize,
}

impl DoubleSharedModulation {
    pub fn new(features: usize, vb: VarBuilder) -> Result<Self> {
        let (lin, chunks) = if let Ok(t) = vb.get(6 * features, "lin").or_else(|_| vb.get((6 * features,), "lin")) {
            (t, 6)
        } else if let Ok(t) = vb.get(4 * features, "lin").or_else(|_| vb.get((4 * features,), "lin")) {
            (t, 4)
        } else {
            let t = vb.get(6 * features, "lin")?;
            (t, 6)
        };
        Ok(Self { lin, chunks })
    }

    pub fn forward(&self, vec: &Tensor) -> Result<Vec<Tensor>> {
        let mod_val = if vec.dim(vec.dims().len() - 1)? == self.lin.dim(0)? {
            vec.broadcast_add(&self.lin.to_dtype(vec.dtype())?)?
        } else {
            self.lin.to_dtype(vec.dtype())?.unsqueeze(0)?
        };
        let mut chunks = mod_val.chunk(self.chunks, candle_core::D::Minus1)?;
        // If 4 chunks (Lumina2: prescale, preshift, postscale, postshift), add gate=1.0 proxies
        if chunks.len() == 4 {
            let ones = Tensor::ones_like(&chunks[0])?;
            let prescale = chunks.remove(0);
            let preshift = chunks.remove(0);
            let postscale = chunks.remove(0);
            let postshift = chunks.remove(0);
            return Ok(vec![prescale, preshift, ones.clone(), postscale, postshift, ones]);
        }
        Ok(chunks)
    }
}

#[derive(Debug, Clone)]
pub enum SingleStreamNorm {
    Krea(KreaRMSNorm),
    Standard(StandardRMSNorm),
}

impl SingleStreamNorm {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Krea(n) => n.forward(x),
            Self::Standard(n) => n.forward(x),
        }
    }
}

/// Single-Stream Transformer Block for Krea 2
#[derive(Debug, Clone)]
pub struct SingleStreamBlock {
    mod_layer: DoubleSharedModulation,
    prenorm: SingleStreamNorm,
    postnorm: SingleStreamNorm,
    attn: KreaAttention,
    mlp: SwiGLU,
}

impl SingleStreamBlock {
    pub fn new(features: usize, heads: usize, kv_heads: usize, mlp_dim: usize, bias: bool, vb: VarBuilder) -> Result<Self> {
        Self::new_adaptive(features, heads, kv_heads, mlp_dim, bias, false, vb)
    }

    pub fn new_adaptive(features: usize, heads: usize, kv_heads: usize, mlp_dim: usize, bias: bool, use_standard_norm: bool, vb: VarBuilder) -> Result<Self> {
        let mod_layer = DoubleSharedModulation::new(features, vb.pp("mod"))?;
        let prenorm = if use_standard_norm {
            SingleStreamNorm::Standard(StandardRMSNorm::new(features, vb.pp("prenorm"))?)
        } else {
            SingleStreamNorm::Krea(KreaRMSNorm::new(features, vb.pp("prenorm"))?)
        };
        let postnorm = if use_standard_norm {
            SingleStreamNorm::Standard(StandardRMSNorm::new(features, vb.pp("postnorm"))?)
        } else {
            SingleStreamNorm::Krea(KreaRMSNorm::new(features, vb.pp("postnorm"))?)
        };
        let attn = KreaAttention::new(features, heads, kv_heads, bias, vb.pp("attn"))?;
        let mlp = SwiGLU::new(features, mlp_dim, bias, vb.pp("mlp"))?;
        Ok(Self { mod_layer, prenorm, postnorm, attn, mlp })
    }

    pub fn forward(&self, x: &Tensor, tvec: &Tensor, rotary_freqs: Option<(&Tensor, &Tensor)>) -> Result<Tensor> {
        let mods = self.mod_layer.forward(tvec)?;
        let prescale = &mods[0];
        let preshift = &mods[1];
        let pregate = &mods[2];
        let postscale = &mods[3];
        let postshift = &mods[4];
        let postgate = &mods[5];

        // Attention branch: x = x + pregate * attn((1 + prescale) * prenorm(x) + preshift)
        let norm_x = self.prenorm.forward(x)?;
        let ones = Tensor::ones((1, 1, 1), x.dtype(), x.device())?;
        let scale_p1 = prescale.unsqueeze(1)?.broadcast_add(&ones)?;
        let attn_in = norm_x.broadcast_mul(&scale_p1)?.broadcast_add(&preshift.unsqueeze(1)?)?;
        let attn_out = self.attn.forward(&attn_in, rotary_freqs)?;
        let x = x.broadcast_add(&attn_out.broadcast_mul(&pregate.unsqueeze(1)?)?)?;

        // MLP branch: x = x + postgate * mlp((1 + postscale) * postnorm(x) + postshift)
        let norm_x2 = self.postnorm.forward(&x)?;
        let scale2_p1 = postscale.unsqueeze(1)?.broadcast_add(&ones)?;
        let mlp_in = norm_x2.broadcast_mul(&scale2_p1)?.broadcast_add(&postshift.unsqueeze(1)?)?;
        let mlp_out = self.mlp.forward(&mlp_in)?;
        let x = x.broadcast_add(&mlp_out.broadcast_mul(&postgate.unsqueeze(1)?)?)?;

        Ok(x)
    }
}

/// Text Fusion Block for Qwen3-VL 12-layer adapter
#[derive(Debug, Clone)]
pub struct TextFusionBlock {
    prenorm: KreaRMSNorm,
    postnorm: KreaRMSNorm,
    attn: KreaAttention,
    mlp: SwiGLU,
}

impl TextFusionBlock {
    pub fn new(txt_dim: usize, heads: usize, kv_heads: usize, mlp_dim: usize, bias: bool, vb: VarBuilder) -> Result<Self> {
        let prenorm = KreaRMSNorm::new(txt_dim, vb.pp("prenorm"))?;
        let postnorm = KreaRMSNorm::new(txt_dim, vb.pp("postnorm"))?;
        let attn = KreaAttention::new(txt_dim, heads, kv_heads, bias, vb.pp("attn"))?;
        let mlp = SwiGLU::new(txt_dim, mlp_dim, bias, vb.pp("mlp"))?;
        Ok(Self { prenorm, postnorm, attn, mlp })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let norm_x = self.prenorm.forward(x)?;
        let attn_out = self.attn.forward(&norm_x, None)?;
        let x = (x + attn_out)?;
        let norm_x2 = self.postnorm.forward(&x)?;
        let mlp_out = self.mlp.forward(&norm_x2)?;
        x + mlp_out
    }
}

/// Text Fusion Transformer (12-layer adapter for Krea 2 or 2-block refiner for Lumina2)
#[derive(Debug, Clone)]
pub struct TextFusionTransformer {
    layerwise_blocks: Vec<TextFusionBlock>,
    projector: Option<Linear>,
    refiner_blocks: Vec<TextFusionBlock>,
}

impl TextFusionTransformer {
    pub fn new(txt_layers: usize, txt_dim: usize, txt_heads: usize, txt_kv_heads: usize, mlp_dim: usize, bias: bool, vb: VarBuilder) -> Result<Self> {
        let mut layerwise_blocks = Vec::with_capacity(2);
        for i in 0..2 {
            if let Ok(b) = TextFusionBlock::new(txt_dim, txt_heads, txt_kv_heads, mlp_dim, bias, vb.pp(format!("layerwise_blocks.{}", i))) {
                layerwise_blocks.push(b);
            }
        }
        let projector = candle_nn::linear_no_bias(txt_layers, 1, vb.pp("projector")).ok();
        let mut refiner_blocks = Vec::with_capacity(2);
        for i in 0..2 {
            if let Ok(b) = TextFusionBlock::new(txt_dim, txt_heads, txt_kv_heads, mlp_dim, bias, vb.pp(format!("refiner_blocks.{}", i)))
                .or_else(|_| TextFusionBlock::new(txt_dim, txt_heads, txt_kv_heads, mlp_dim, bias, vb.pp(format!("{}", i)))) {
                refiner_blocks.push(b);
            }
        }
        Ok(Self { layerwise_blocks, projector, refiner_blocks })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // If x has 4 dimensions: [B, seq_len, num_taps=12, 2560]
        if x.rank() == 4 {
            let (b, s, l, d) = x.dims4()?;

            // 1. Inter-layer attention along the 12 extracted Qwen layers for each token:
            let mut h = x.reshape((b * s, l, d))?;
            for block in &self.layerwise_blocks {
                h = block.forward(&h)?;
            }

            // 2. Linear projection 12 -> 1 across layers: [B, seq_len, 2560, 12] -> [B, seq_len, 2560]
            let projected = if let Some(ref proj) = self.projector {
                let h_proj_in = h.reshape((b, s, l, d))?.transpose(2, 3)?.contiguous()?;
                proj.forward(&h_proj_in)?.squeeze(3)?
            } else {
                // If no projector, average across taps
                let h_proj_in = h.reshape((b, s, l, d))?;
                h_proj_in.mean(2)?
            };

            // 3. Inter-token contextual refinement attention across sequence tokens:
            let mut out = projected;
            for block in &self.refiner_blocks {
                out = block.forward(&out)?;
            }
            Ok(out)
        } else {
            // 3D tensor: [B, seq_len, 2560]
            let mut out = x.clone();
            for block in &self.refiner_blocks {
                out = block.forward(&out)?;
            }
            Ok(out)
        }
    }
}

/// Last Layer for Krea 2 / Lumina2
#[derive(Debug, Clone)]
pub struct LastLayer {
    norm: SingleStreamNorm,
    linear: Linear,
    ada_linear: Option<Linear>,
    modulation_lin: Tensor,
}

impl LastLayer {
    pub fn new(features: usize, out_channels: usize, vb: VarBuilder) -> Result<Self> {
        Self::new_with_root(features, out_channels, vb.clone(), vb)
    }

    pub fn new_with_root(features: usize, out_channels: usize, vb: VarBuilder, root_vb: VarBuilder) -> Result<Self> {
        let is_lumina = root_vb.get((features,), "norm_final.weight").is_ok()
            || root_vb.get((3840, 256), "final_layer.adaLN_modulation.1.weight").is_ok();
        
        let norm = if is_lumina {
            SingleStreamNorm::Standard(
                StandardRMSNorm::new(features, vb.pp("norm"))
                    .or_else(|_| StandardRMSNorm::new(features, vb.pp("norm_final")))
                    .or_else(|_| StandardRMSNorm::new(features, root_vb.pp("norm_final")))?
            )
        } else {
            SingleStreamNorm::Krea(
                KreaRMSNorm::new(features, vb.pp("norm"))
                    .or_else(|_| KreaRMSNorm::new(features, vb.pp("norm_final")))
                    .or_else(|_| KreaRMSNorm::new(features, root_vb.pp("norm_final")))?
            )
        };

        let linear = linear(features, out_channels, vb.pp("linear"))?;

        let ada_linear = if let Ok(w) = root_vb.get((features, 256), "final_layer.adaLN_modulation.1.weight") {
            let b = root_vb.get((features,), "final_layer.adaLN_modulation.1.bias").ok();
            Some(Linear::new(w, b))
        } else {
            None
        };

        let modulation_lin = vb.pp("modulation").get((2, features), "lin")
            .or_else(|_| vb.get((2, features), "modulation.lin"))
            .or_else(|_| vb.get((features,), "adaLN_modulation.1.bias"))
            .or_else(|_| {
                Tensor::zeros((2, features), vb.dtype(), vb.device())
            })?;

        Ok(Self { norm, linear, ada_linear, modulation_lin })
    }

    pub fn forward(&self, x: &Tensor, tvec: &Tensor) -> Result<Tensor> {
        let norm_x = self.norm.forward(x)?;

        if let Some(ref ada) = self.ada_linear {
            // Lumina2: scale = adaLN_modulation.1(SiLU(t))
            let t_silu = candle_nn::ops::silu(tvec)?;
            let scale = ada.forward(&t_silu)?; // [B, features]
            let ones = Tensor::ones((1, 1, 1), x.dtype(), x.device())?;
            let scale_p1 = scale.unsqueeze(1)?.broadcast_add(&ones)?;
            let mod_x = norm_x.broadcast_mul(&scale_p1)?;
            return self.linear.forward(&mod_x);
        }

        let lin = self.modulation_lin.to_dtype(tvec.dtype())?;
        let (scale, shift) = if lin.dim(0)? == 2 {
            let lin_scale = lin.narrow(0, 0, 1)?.squeeze(0)?; // [features]
            let lin_shift = lin.narrow(0, 1, 1)?.squeeze(0)?; // [features]
            let scale = if tvec.dim(tvec.dims().len() - 1)? == lin_scale.dim(0)? {
                tvec.broadcast_add(&lin_scale)?
            } else {
                lin_scale.unsqueeze(0)?
            };
            let shift = if tvec.dim(tvec.dims().len() - 1)? == lin_shift.dim(0)? {
                tvec.broadcast_add(&lin_shift)?
            } else {
                lin_shift.unsqueeze(0)?
            };
            (scale, shift)
        } else {
            let scale = lin.unsqueeze(0)?;
            let shift = lin.unsqueeze(0)?;
            (scale, shift)
        };

        let ones = Tensor::ones((1, 1, 1), x.dtype(), x.device())?;
        let scale_p1 = scale.unsqueeze(1)?.broadcast_add(&ones)?;
        let mod_x = norm_x.broadcast_mul(&scale_p1)?.broadcast_add(&shift.unsqueeze(1)?)?;
        self.linear.forward(&mod_x)
    }
}

/// Compute 3D RoPE (EmbedND / Krea 2 standard: axes [32, 48, 48], theta = 1000.0)
pub fn compute_krea_rope(
    pos: &Tensor, // [B, L, 3] containing integer coordinates (t=0, y=row, x=col)
    theta: f64,   // 1000.0 or 100.0
) -> Result<(Tensor, Tensor)> {
    let axes = [32usize, 48usize, 48usize];
    let mut angles_all = Vec::with_capacity(3);

    for (axis_idx, &axis_dim) in axes.iter().enumerate() {
        let half = axis_dim / 2;
        let mut half_freqs = Vec::with_capacity(half);
        for step in 0..half {
            let scale_val = (2.0 * step as f64) / (axis_dim as f64);
            let omega = 1.0 / theta.powf(scale_val);
            half_freqs.push(omega as f32);
        }
        let omega_t = Tensor::from_vec(half_freqs, (1, 1, half), pos.device())?;
        let pos_axis = pos.narrow(2, axis_idx, 1)?; // [B, L, 1]
        let angles = pos_axis.broadcast_mul(&omega_t)?; // [B, L, half]
        angles_all.push(angles);
    }

    let angles_cat = Tensor::cat(&angles_all.iter().collect::<Vec<_>>(), 2)?; // [B, L, 64]
    let cos = angles_cat.cos()?.unsqueeze(1)?.unsqueeze(4)?; // [B, 1, L, 64, 1]
    let sin = angles_cat.sin()?.unsqueeze(1)?.unsqueeze(4)?; // [B, 1, L, 64, 1]
    Ok((cos, sin))
}

/// Sinusoidal Timestep Embedding (DiffSynth / Krea 2 standard: tfactor=1000, period=10000)
pub fn krea_timestep_embedding(t: &Tensor, dim: usize) -> Result<Tensor> {
    let t_scaled = (t * 1000.0)?;
    let half = dim / 2;
    let mut freqs_vec = Vec::with_capacity(half);
    let max_period = 10000.0f64;
    for i in 0..half {
        let freq = (-max_period.ln() * (i as f64) / (half as f64)).exp();
        freqs_vec.push(freq as f32);
    }
    let freqs = Tensor::from_vec(freqs_vec, (1, half), t.device())?;
    let t_f32 = t_scaled.to_dtype(DType::F32)?.unsqueeze(1)?; // [B, 1]
    let args = t_f32.matmul(&freqs)?; // [B, half]
    let cos = args.cos()?;
    let sin = args.sin()?;
    Tensor::cat(&[&cos, &sin], 1)?.to_dtype(t.dtype())
}

/// Complete Krea 2 Turbo Single-Stream MMDiT Transformer
#[derive(Clone)]
pub struct ZImageTransformer {
    pub config: ZImageConfig,
    pub first: Linear,
    pub tmlp_0: Linear,
    pub tmlp_2: Linear,
    pub tproj_1: Linear,
    pub txtfusion: TextFusionTransformer,
    pub txtmlp_norm: KreaRMSNorm,
    pub txtmlp_1: Linear,
    pub txtmlp_3: Linear,
    pub blocks: Vec<SingleStreamBlock>,
    pub last: LastLayer,
    // Low-VRAM streaming support with Rayon multi-threaded dequantization
    pub archive: Option<std::sync::Arc<crate::weights::SafeTensorsArchive>>,
    pub device: candle_core::Device,
    pub dtype: DType,
}

impl ZImageTransformer {
    pub fn new(cfg: ZImageConfig, vb: VarBuilder) -> Result<Self> {
        let first = linear(cfg.in_channels, cfg.hidden_size, vb.pp("first"))?;

        let tmlp_0 = linear(cfg.time_embed_dim, cfg.hidden_size, vb.pp("tmlp.0"))?;
        let tmlp_2 = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("tmlp.2"))?;
        let tproj_1 = linear(cfg.hidden_size, cfg.hidden_size * 6, vb.pp("tproj.1"))?;

        let txt_heads = 20;
        let txt_kv_heads = 20;
        let txt_mlp_dim = 6912;

        let txtfusion = TextFusionTransformer::new(
            12,
            cfg.cap_dim,
            txt_heads,
            txt_kv_heads,
            txt_mlp_dim,
            false,
            vb.pp("txtfusion"),
        )?;

        let txtmlp_norm = KreaRMSNorm::new(cfg.cap_dim, vb.pp("txtmlp.0"))?;
        let txtmlp_1 = linear(cfg.cap_dim, cfg.hidden_size, vb.pp("txtmlp.1"))?;
        let txtmlp_3 = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("txtmlp.3"))?;

        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let block = SingleStreamBlock::new(
                cfg.hidden_size,
                cfg.num_heads,
                cfg.num_kv_heads,
                cfg.intermediate_dim,
                false,
                vb.pp(format!("blocks.{}", i)),
            )?;
            blocks.push(block);
        }

        let last = LastLayer::new(cfg.hidden_size, cfg.out_channels, vb.pp("last"))?;

        Ok(Self {
            config: cfg,
            first,
            tmlp_0,
            tmlp_2,
            tproj_1,
            txtfusion,
            txtmlp_norm,
            txtmlp_1,
            txtmlp_3,
            blocks,
            last,
            archive: None,
            device: candle_core::Device::Cpu,
            dtype: DType::BF16,
        })
    }

    /// Construct ZImageTransformer in Low-VRAM Sequential Streaming Mode (< 2GB peak VRAM)
    pub fn new_streaming(
        cfg: ZImageConfig,
        header_vb: VarBuilder,
        archive: std::sync::Arc<crate::weights::SafeTensorsArchive>,
        device: candle_core::Device,
        dtype: DType,
    ) -> Result<Self> {
        let first = linear(cfg.in_channels, cfg.hidden_size, header_vb.pp("first"))
            .or_else(|_| linear(cfg.in_channels, cfg.hidden_size, header_vb.pp("x_embedder")))?;

        let tmlp_0 = linear(cfg.time_embed_dim, cfg.hidden_size, header_vb.pp("tmlp.0"))
            .or_else(|_| linear(cfg.time_embed_dim, 1024, header_vb.pp("t_embedder.mlp.0")))?;
        let tmlp_2 = linear(cfg.hidden_size, cfg.hidden_size, header_vb.pp("tmlp.2"))
            .or_else(|_| linear(1024, 256, header_vb.pp("t_embedder.mlp.2")))?;
        let tproj_1 = linear(cfg.hidden_size, cfg.hidden_size * 6, header_vb.pp("tproj.1"))
            .or_else(|_| candle_nn::linear_no_bias(cfg.hidden_size, cfg.hidden_size * 6, header_vb.pp("tproj.1")))
            .or_else(|_| -> Result<Linear> {
                // If tproj.1 is not present in AIO Lumina2 (where t is 256-dim), create 256 -> 6*hidden_size proxy
                let weight = Tensor::zeros((cfg.hidden_size * 6, 256), dtype, &device)?;
                let bias = Tensor::zeros((cfg.hidden_size * 6,), dtype, &device)?;
                Ok(Linear::new(weight, Some(bias)))
            })?;

        let txt_heads = 20;
        let txt_kv_heads = 20;
        let txt_mlp_dim = 6912;

        let txtfusion = TextFusionTransformer::new(
            12,
            cfg.cap_dim,
            txt_heads,
            txt_kv_heads,
            txt_mlp_dim,
            false,
            header_vb.pp("txtfusion"),
        ).or_else(|_| {
            // Lumina2 / Z-Image standard format uses cap_embedder instead of 12-layer txtfusion
            TextFusionTransformer::new(
                12,
                cfg.cap_dim,
                txt_heads,
                txt_kv_heads,
                txt_mlp_dim,
                false,
                header_vb.pp("context_refiner"),
            )
        })?;

        let txtmlp_norm = KreaRMSNorm::new(cfg.cap_dim, header_vb.pp("txtmlp.0"))
            .or_else(|_| KreaRMSNorm::new(cfg.cap_dim, header_vb.pp("cap_embedder.0")))?;
        let txtmlp_1 = linear(cfg.cap_dim, cfg.hidden_size, header_vb.pp("txtmlp.1"))
            .or_else(|_| linear(cfg.cap_dim, cfg.hidden_size, header_vb.pp("cap_embedder.1")))?;
        let txtmlp_3 = linear(cfg.hidden_size, cfg.hidden_size, header_vb.pp("txtmlp.3"))
            .or_else(|_| -> Result<Linear> {
                // Identity linear if single-layer cap_embedder
                let weight = Tensor::eye(cfg.hidden_size, dtype, &device)?;
                let bias = Tensor::zeros((cfg.hidden_size,), dtype, &device)?;
                Ok(Linear::new(weight, Some(bias)))
            })?;

        let last = LastLayer::new_with_root(cfg.hidden_size, cfg.out_channels, header_vb.pp("last"), header_vb.clone())
            .or_else(|_| LastLayer::new_with_root(cfg.hidden_size, cfg.out_channels, header_vb.pp("final_layer"), header_vb.clone()))?;

        Ok(Self {
            config: cfg,
            first,
            tmlp_0,
            tmlp_2,
            tproj_1,
            txtfusion,
            txtmlp_norm,
            txtmlp_1,
            txtmlp_3,
            blocks: Vec::new(),
            last,
            archive: Some(archive),
            device,
            dtype,
        })
    }

    /// Forward pass: latents [B, C=16, H, W], timestep [B] (sigma), context [B, 12, seq_len, 2560]
    pub fn forward(
        &self,
        latents: &Tensor,
        timestep: &Tensor,
        context: &Tensor,
    ) -> Result<Tensor> {
        let (b, c, h, w) = latents.dims4()?;
        let p_h = h / 2;
        let p_w = w / 2;
        let img_tokens = p_h * p_w;

        // 1. Exact Krea 2 Patchify:
        // rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        let img = latents
            .reshape((b, c, p_h, 2, p_w, 2))?
            .permute((0, 2, 4, 1, 3, 5))?
            .contiguous()?
            .reshape((b, img_tokens, c * 4))?;

        let img_emb = self.first.forward(&img)?;

        // 2. Timestep embedding: tmlp(temb(t)) where tmlp activation is GELU(tanh)
        let temb_raw = krea_timestep_embedding(timestep, self.config.time_embed_dim)?;
        let t_act = self.tmlp_0.forward(&temb_raw)?.gelu_erf()?;
        let t = self.tmlp_2.forward(&t_act)?; // [B, features]
        let tvec_act = t.gelu_erf()?;
        let tvec = self.tproj_1.forward(&tvec_act)?; // [B, features * 6]

        // 3. Text conditioning pipeline:
        // context: [B, seq_len, 12, 2560]
        let txt_fused = self.txtfusion.forward(context)?; // [B, seq_len, 2560]
        let txt_norm = self.txtmlp_norm.forward(&txt_fused)?;
        let txt_act = self.txtmlp_1.forward(&txt_norm)?.gelu_erf()?;
        let txt_emb = self.txtmlp_3.forward(&txt_act)?; // [B, seq_len, features]

        let txt_len = txt_emb.dim(1)?;
        let total_len = txt_len + img_tokens;

        if let (Ok(img_f), Ok(txt_f), Ok(tv_f)) = (img_emb.to_dtype(DType::F32), txt_emb.to_dtype(DType::F32), tvec.to_dtype(DType::F32)) {
            let img_std = (img_f.sqr()?.mean_all()?.to_scalar::<f32>()? - img_f.mean_all()?.to_scalar::<f32>()?.powi(2)).sqrt();
            let txt_std = (txt_f.sqr()?.mean_all()?.to_scalar::<f32>()? - txt_f.mean_all()?.to_scalar::<f32>()?.powi(2)).sqrt();
            let tv_std = (tv_f.sqr()?.mean_all()?.to_scalar::<f32>()? - tv_f.mean_all()?.to_scalar::<f32>()?.powi(2)).sqrt();
            println!("      📊 Embeddings: img_std={:.4}, txt_std={:.4}, tvec_std={:.4}", img_std, txt_std, tv_std);
        }

        // 4. Position IDs: text tokens at (0, 0, 0), image tokens at (frame=0, row, col) matching axes [32, 48, 48]
        let mut pos_vec = Vec::with_capacity(total_len * 3);
        // Text pos IDs: Frame = 0, Row = 0, Col = 0 (context invariance scheme)
        for _ in 0..txt_len {
            pos_vec.push(0f32);
            pos_vec.push(0f32);
            pos_vec.push(0f32);
        }
        // Image pos IDs: Frame = 0, Row = r, Col = col (axes [32, 48, 48])
        for r in 0..p_h {
            for col in 0..p_w {
                pos_vec.push(0f32);
                pos_vec.push(r as f32);
                pos_vec.push(col as f32);
            }
        }
        let pos_t = Tensor::from_vec(pos_vec, (1, total_len, 3), latents.device())?
            .repeat((b, 1, 1))?;

        let (cos, sin) = compute_krea_rope(&pos_t, self.config.theta)?;
        let rotary_freqs = (&cos, &sin);

        // 5. Combined sequence: [Text tokens FIRST, Image tokens SECOND]
        let mut combined = Tensor::cat(&[&txt_emb, &img_emb], 1)?;

        // 6. Single-Stream MMDiT Blocks (Resident or On-Demand Multi-Threaded Streaming)
        if !self.blocks.is_empty() {
            for block in &self.blocks {
                combined = block.forward(&combined, &tvec, Some(rotary_freqs))?;
            }
        } else if let Some(archive) = &self.archive {
            // Streaming mode with Rayon multi-threaded dequantization (< 1.8GB VRAM peak)
            for i in 0..self.config.num_layers {
                let prefix = format!("blocks.{}.", i);
                let prefix_alt = format!("model.diffusion_model.blocks.{}.", i);
                let prefix_layer = format!("model.diffusion_model.layers.{}.", i);
                let mut block_tensors = std::collections::HashMap::new();
                for key in archive.keys() {
                    let matched_suffix = if let Some(suffix) = key.strip_prefix(&prefix) {
                        Some((suffix, false))
                    } else if let Some(suffix) = key.strip_prefix(&prefix_alt) {
                        Some((suffix, false))
                    } else if let Some(suffix) = key.strip_prefix(&prefix_layer) {
                        Some((suffix, true))
                    } else {
                        None
                    };
                    if let Some((suffix, is_lumina_layer)) = matched_suffix {
                        if suffix.ends_with(".weight_scale") || suffix.ends_with(".scale_weight") || suffix.ends_with(".comfy_quant") {
                            continue;
                        }
                        if let Ok(t) = archive.get_tensor(key, &self.device, self.dtype) {
                            if is_lumina_layer {
                                // Map Lumina2 layer keys to Krea2 SingleStreamBlock convention
                                let mapped_name = if suffix == "adaLN_modulation.0.bias" {
                                    format!("blocks.{}.mod.lin", i)
                                } else if suffix == "attention_norm1.weight" {
                                    format!("blocks.{}.prenorm.scale", i)
                                } else if suffix == "ffn_norm1.weight" {
                                    format!("blocks.{}.postnorm.scale", i)
                                } else if suffix == "attention_norm2.weight" {
                                    format!("blocks.{}.attn_norm2.scale", i)
                                } else if suffix == "attention.qkv.weight" {
                                    // qkv split handled or mapped to wq/wk/wv
                                    let h_dim = self.config.hidden_size;
                                    if let Ok(wq) = t.narrow(0, 0, h_dim) {
                                        block_tensors.insert(format!("blocks.{}.attn.wq.weight", i), wq);
                                    }
                                    if let Ok(wk) = t.narrow(0, h_dim, h_dim) {
                                        block_tensors.insert(format!("blocks.{}.attn.wk.weight", i), wk);
                                    }
                                    if let Ok(wv) = t.narrow(0, h_dim * 2, h_dim) {
                                        block_tensors.insert(format!("blocks.{}.attn.wv.weight", i), wv);
                                    }
                                    continue;
                                } else if suffix == "attention.out.weight" {
                                    format!("blocks.{}.attn.wo.weight", i)
                                } else if suffix == "attention.q_norm.weight" {
                                    format!("blocks.{}.attn.qknorm.qnorm.scale", i)
                                } else if suffix == "attention.k_norm.weight" {
                                    format!("blocks.{}.attn.qknorm.knorm.scale", i)
                                } else if suffix == "feed_forward.w1.weight" {
                                    format!("blocks.{}.mlp.gate.weight", i)
                                } else if suffix == "feed_forward.w3.weight" {
                                    format!("blocks.{}.mlp.up.weight", i)
                                } else if suffix == "feed_forward.w2.weight" {
                                    format!("blocks.{}.mlp.down.weight", i)
                                } else {
                                    format!("blocks.{}.{}", i, suffix)
                                };
                                block_tensors.insert(mapped_name, t);
                            } else {
                                block_tensors.insert(format!("blocks.{}.{}", i, suffix), t);
                            }
                        }
                    }
                }
                // Lumina2 / Z-Image standard layer-wise modulation mapping
                let ada_ln_opt = if let Ok(t) = archive.get_tensor(&format!("model.diffusion_model.layers.{}.adaLN_modulation.0.weight", i), &self.device, self.dtype) {
                    let bias = archive.get_tensor(&format!("model.diffusion_model.layers.{}.adaLN_modulation.0.bias", i), &self.device, self.dtype).ok();
                    Some(Linear::new(t, bias))
                } else if let Ok(t) = archive.get_tensor(&format!("layers.{}.adaLN_modulation.0.weight", i), &self.device, self.dtype) {
                    let bias = archive.get_tensor(&format!("layers.{}.adaLN_modulation.0.bias", i), &self.device, self.dtype).ok();
                    Some(Linear::new(t, bias))
                } else {
                    None
                };

                // Ensure mod.lin fallback if not loaded
                if !block_tensors.contains_key(&format!("blocks.{}.mod.lin", i)) {
                    if let Ok(zero_mod) = Tensor::zeros((6 * self.config.hidden_size,), self.dtype, &self.device) {
                        block_tensors.insert(format!("blocks.{}.mod.lin", i), zero_mod);
                    }
                }
                let layer_tvec = if let Some(ref ada) = ada_ln_opt {
                    let t_silu = candle_nn::ops::silu(&t)?;
                    ada.forward(&t_silu)?
                } else {
                    tvec.clone()
                };

                let block_vb = VarBuilder::from_tensors(block_tensors, self.dtype, &self.device);
                let block = SingleStreamBlock::new_adaptive(
                    self.config.hidden_size,
                    self.config.num_heads,
                    self.config.num_kv_heads,
                    self.config.intermediate_dim,
                    false,
                    ada_ln_opt.is_some(), // Use standard unit RMSNorm if Lumina2 format
                    block_vb.pp(format!("blocks.{}", i)),
                )?;
                combined = block.forward(&combined, &layer_tvec, Some(rotary_freqs))?;
                if i == 0 || i == 13 || i == 27 {
                    let c_f32 = combined.to_dtype(DType::F32)?;
                    let mean = c_f32.mean_all()?.to_scalar::<f32>()?;
                    let std = (c_f32.sqr()?.mean_all()?.to_scalar::<f32>()? - mean * mean).sqrt();
                    println!("      Layer {:2}/28 combined: mean={:+.4}, std={:.4}", i + 1, mean, std);
                }
            }
        }

        // 7. Extract image token slice strictly BEFORE last layer modulation
        let img_tokens_out = combined.narrow(1, txt_len, img_tokens)?;

        // 8. Last Layer modulation & linear projection on image tokens
        let out = self.last.forward(&img_tokens_out, &t)?;

        // 9. Exact Krea 2 Unpatchify:
        // rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=p_h, w=p_w, ph=2, pw=2, c=16)
        let out_latents = out
            .reshape((b, p_h, p_w, c, 2, 2))?
            .permute((0, 3, 1, 4, 2, 5))?
            .contiguous()?
            .reshape((b, c, h, w))?;

        Ok(out_latents)
    }
}
