// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Pure Rust ACE-Step 1.5 Turbo 1D Transformer Model for Audio/Music Diffusion

use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Linear, VarBuilder};
use std::path::Path;

/// Timestep sinusoidal embedding generator for 1D diffusion (cos, sin convention)
pub fn get_timestep_embedding(timesteps: &Tensor, embedding_dim: usize) -> candle_core::Result<Tensor> {
    let half_dim = embedding_dim / 2;
    let factor = (-(10000.0f64.ln()) / (half_dim as f64)).exp();
    let dev = timesteps.device();
    let mut freqs_vec = Vec::with_capacity(half_dim);
    let mut cur = 1.0f64;
    for _ in 0..half_dim {
        freqs_vec.push(cur as f32);
        cur *= factor;
    }
    let freqs = Tensor::new(freqs_vec.as_slice(), dev)?.to_dtype(timesteps.dtype())?;
    let t = if timesteps.dims().len() == 1 { timesteps.unsqueeze(1)? } else { timesteps.clone() };
    let args = t.broadcast_mul(&freqs.unsqueeze(0)?)?;
    let cos = args.cos()?;
    let sin = args.sin()?;
    Tensor::cat(&[&cos, &sin], 1)
}

/// ACE-Step 2-layer MLP Timestep Embedding + 6-way AdaLN Projection
#[derive(Debug, Clone)]
pub struct AceStepTimestepEmbedding {
    linear1: Linear,
    linear2: Linear,
    time_proj: Linear,
    hidden_size: usize,
}

impl AceStepTimestepEmbedding {
    pub fn load(vb: VarBuilder, hidden_size: usize) -> Result<Self> {
        let linear1 = candle_nn::linear(256, hidden_size, vb.pp("linear_1"))?;
        let linear2 = candle_nn::linear(hidden_size, hidden_size, vb.pp("linear_2"))?;
        let time_proj = candle_nn::linear(hidden_size, 6 * hidden_size, vb.pp("time_proj"))?;
        Ok(Self {
            linear1,
            linear2,
            time_proj,
            hidden_size,
        })
    }

    pub fn forward(&self, timesteps: &Tensor) -> candle_core::Result<(Tensor, Tensor)> {
        // Reference: temb = linear_2(silu(linear_1(x))); proj = time_proj(silu(temb))
        let scaled_t = (timesteps * 1000.0)?;
        let t_emb = get_timestep_embedding(&scaled_t, 256)?;
        let t_hid = candle_nn::ops::silu(&self.linear1.forward(&t_emb)?)?;
        let temb = self.linear2.forward(&t_hid)?;
        let b = timesteps.dim(0)?;
        let proj = self.time_proj.forward(&candle_nn::ops::silu(&temb)?)?.reshape((b, 6, self.hidden_size))?;
        Ok((temb, proj))
    }
}

/// Rotary Position Embedding (1D RoPE) for audio sequence tokens
#[derive(Debug, Clone)]
pub struct AudioRotaryEmbedding {
    dim: usize,
    theta: f64,
}

impl AudioRotaryEmbedding {
    pub fn new(dim: usize, theta: f64) -> Self {
        Self { dim, theta }
    }

    pub fn apply_rope(&self, q: &Tensor, k: &Tensor, seq_len: usize) -> candle_core::Result<(Tensor, Tensor)> {
        let dev = q.device();
        let half_dim = self.dim / 2;
        let mut inv_freq = Vec::with_capacity(half_dim);
        for i in 0..half_dim {
            let freq = 1.0 / self.theta.powf((2 * i) as f64 / self.dim as f64);
            inv_freq.push(freq as f32);
        }
        let inv_freq_t = Tensor::new(inv_freq.as_slice(), dev)?;
        let t: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
        let t_t = Tensor::new(t.as_slice(), dev)?;
        let freqs = t_t.unsqueeze(1)?.broadcast_mul(&inv_freq_t.unsqueeze(0)?)?; // [seq_len, half_dim]
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;

        let q_rot = self.rotate_half(q, &cos, &sin)?;
        let k_rot = self.rotate_half(k, &cos, &sin)?;
        Ok((q_rot, k_rot))
    }

    fn rotate_half(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
        // x: [batch, heads, seq_len, head_dim]
        let (_b, _h, _s, d) = x.dims4()?;
        let half = d / 2;
        let x1 = x.narrow(3, 0, half)?;
        let x2 = x.narrow(3, half, half)?;

        let cos = cos.to_dtype(x.dtype())?.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, seq_len, half_dim]
        let sin = sin.to_dtype(x.dtype())?.unsqueeze(0)?.unsqueeze(0)?;

        let out1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let out2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;
        Tensor::cat(&[&out1, &out2], 3)
    }
}

/// Numerically stable RMSNorm for Audio Transformer (computed in FP32 to prevent FP16 sqr overflow)
#[derive(Debug, Clone)]
pub struct AudioRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl AudioRmsNorm {
    pub fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let orig_dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let sq = x_f32.sqr()?;
        let last_dim = sq.dims().len() - 1;
        let mean = sq.mean_keepdim(last_dim)?;
        let rms = (mean + self.eps)?.sqrt()?;
        let norm = x_f32.broadcast_div(&rms)?;
        let w_f32 = self.weight.to_dtype(DType::F32)?;
        norm.broadcast_mul(&w_f32)?.to_dtype(orig_dtype)
    }
}

/// SwiGLU Feed-Forward Network for Audio Transformer
#[derive(Debug, Clone)]
pub struct AceStepMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl AceStepMlp {
    pub fn load(vb: VarBuilder, hidden_size: usize, intermediate_size: usize) -> Result<Self> {
        let gate_proj = candle_nn::linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = candle_nn::linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = candle_nn::linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let orig_dtype = x.dtype();
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        let gate_f32 = gate.to_dtype(DType::F32)?;
        let up_f32 = up.to_dtype(DType::F32)?;
        let intermediate = (gate_f32 * up_f32)?.to_dtype(orig_dtype)?;
        self.down_proj.forward(&intermediate)
    }
}

/// Grouped-Query Attention with QK RMSNorm for 1D Audio Transformer
#[derive(Debug, Clone)]
pub struct AceStepAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    norm_q: AudioRmsNorm,
    norm_k: AudioRmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl AceStepAttention {
    pub fn load(
        vb: VarBuilder,
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let to_q = candle_nn::linear_no_bias(hidden_size, num_heads * head_dim, vb.pp("to_q"))?;
        let to_k = candle_nn::linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("to_k"))?;
        let to_v = candle_nn::linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("to_v"))?;
        let to_out = candle_nn::linear_no_bias(num_heads * head_dim, hidden_size, vb.pp("to_out.0"))?;
        let norm_q = AudioRmsNorm::new(head_dim, 1e-6, vb.pp("norm_q"))?;
        let norm_k = AudioRmsNorm::new(head_dim, 1e-6, vb.pp("norm_k"))?;

        Ok(Self {
            to_q,
            to_k,
            to_v,
            to_out,
            norm_q,
            norm_k,
            num_heads,
            num_kv_heads,
            head_dim,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        context: Option<&Tensor>,
        rope: Option<&AudioRotaryEmbedding>,
        mask: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let ctx = context.unwrap_or(x);
        let (_, s_ctx, _) = ctx.dims3()?;

        let q = self.to_q.forward(x)?;
        let k = self.to_k.forward(ctx)?;
        let v = self.to_v.forward(ctx)?;

        let mut q = q.reshape((b, s, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let mut k = k.reshape((b, s_ctx, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let mut v = v.reshape((b, s_ctx, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;

        q = self.norm_q.forward(&q)?;
        k = self.norm_k.forward(&k)?;

        if let Some(rope_mod) = rope {
            let (q_rot, k_rot) = rope_mod.apply_rope(&q, &k, s)?;
            q = q_rot;
            k = k_rot;
        }

        // Expand KV heads for GQA if num_kv_heads < num_heads
        if self.num_kv_heads < self.num_heads {
            let repeat = self.num_heads / self.num_kv_heads;
            k = k.unsqueeze(2)?.repeat((1, 1, repeat, 1, 1))?.flatten(1, 2)?;
            v = v.unsqueeze(2)?.repeat((1, 1, repeat, 1, 1))?.flatten(1, 2)?;
        }

        // Memory-efficient scaled-dot-product attention: chunk the query dimension so
        // the transient score tensor is `[b, h, chunk, s]` instead of `[b, h, s, s]`.
        // Essential for long audio (s ~ 2250 at 3 min) which would otherwise OOM.
        let orig_dtype = q.dtype();
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let q_f32 = q.to_dtype(DType::F32)?;
        let k_t = k.to_dtype(DType::F32)?.transpose(2, 3)?.contiguous()?;
        let v_f32 = v.to_dtype(DType::F32)?;

        let q_chunk = 256usize;
        let mut outs: Vec<Tensor> = Vec::new();
        let mut start = 0usize;
        while start < s {
            let len = q_chunk.min(s - start);
            let qc = q_f32.narrow(2, start, len)?;
            let mut sc = (qc.matmul(&k_t)? * scale)?;
            if let Some(m) = mask {
                let mc = m.narrow(0, start, len)?.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, len, s]
                sc = sc.broadcast_add(&mc)?;
            }
            let p = candle_nn::ops::softmax(&sc, D::Minus1)?;
            outs.push(p.matmul(&v_f32)?);
            start += len;
        }
        let att = if outs.len() == 1 {
            outs.pop().unwrap()
        } else {
            Tensor::cat(&outs, 2)?
        };
        let out = att.to_dtype(orig_dtype)?;
        let out = out.transpose(1, 2)?.reshape((b, s, self.num_heads * self.head_dim))?;
        self.to_out.forward(&out)
    }
}

/// Build an additive sliding-window attention mask `[s, s]` (0 for |i-j| <= window, -inf otherwise).
fn build_sliding_mask(s: usize, window: usize, dev: &Device) -> candle_core::Result<Tensor> {
    let idx = Tensor::arange(0u32, s as u32, dev)?.to_dtype(DType::F32)?;
    let diff = idx.unsqueeze(1)?.broadcast_sub(&idx.unsqueeze(0)?)?;
    let valid = diff.abs()?.le(window as f64)?;
    let neg = Tensor::full(f32::NEG_INFINITY, (s, s), dev)?;
    let zero = Tensor::zeros((s, s), DType::F32, dev)?;
    valid.where_cond(&zero, &neg)
}

/// Adaptive Projected Guidance (ACE-Step base/sft CFG), matching `apg_forward`.
/// `pred_*` are `[B, C, T]`; the projection/normalisation runs along the channel dim (1).
pub fn apg_forward(
    pred_cond: &Tensor,
    pred_uncond: &Tensor,
    guidance_scale: f32,
    momentum: &mut Option<Tensor>,
) -> candle_core::Result<Tensor> {
    let orig = pred_cond.dtype();
    let cond = pred_cond.to_dtype(DType::F32)?;
    let uncond = pred_uncond.to_dtype(DType::F32)?;

    let diff = (&cond - &uncond)?;
    // Momentum buffer: running = diff + (-0.75) * running_prev (0 on the first step).
    let diff = match momentum {
        Some(prev) => (&diff + prev.affine(-0.75, 0.0)?)?,
        None => diff.clone(),
    };
    *momentum = Some(diff.clone());

    // Norm clamp: factor = min(1, 2.5 / ||diff||_2) over dim 1.
    let norm = diff.sqr()?.sum_keepdim(1)?.sqrt()?;
    let ratio = norm.recip()?.affine(2.5, 0.0)?;
    let factor = ratio.clamp(0.0, 1.0)?;
    let diff = diff.broadcast_mul(&factor)?;

    // Decompose diff into parallel/orthogonal components w.r.t. the conditional prediction.
    let cond_norm = cond.sqr()?.sum_keepdim(1)?.sqrt()?;
    let v1 = cond.broadcast_div(&(cond_norm + 1e-12)?)?;
    let parallel = diff.broadcast_mul(&v1)?.sum_keepdim(1)?.broadcast_mul(&v1)?;
    let orthogonal = (&diff - &parallel)?;

    let guided = (&cond + orthogonal.affine((guidance_scale - 1.0) as f64, 0.0)?)?;
    guided.to_dtype(orig)
}

/// Full Flow-Matching configuration (cover / repaint extras). All tensors are `[1, ...]`.
pub struct FlowMatchConfig<'a> {
    pub condition: &'a Tensor,
    pub null_condition: Option<&'a Tensor>,
    pub condition_non_cover: Option<&'a Tensor>,
    pub null_condition_non_cover: Option<&'a Tensor>,
    /// `[1, T, 128] = [src_latents, chunk_mask]`.
    pub context_latents: &'a Tensor,
    pub context_latents_non_cover: Option<&'a Tensor>,
    /// `[1, T, 64]` initial noise.
    pub noise: &'a Tensor,
    pub t_schedule: &'a [f64],
    pub guidance_scale: f32,
    /// Fraction of steps conditioned on the cover source (1.0 = always).
    pub cover_strength: f32,
    /// `> 0` initializes `x_t` from `clean_src` at the nearest timestep.
    pub cover_noise_strength: f32,
    /// `[1, T, 64]` clean source latents (repaint / cover init).
    pub clean_src: Option<&'a Tensor>,
    /// `[1, T]` repaint mask (True = generate, False = preserve).
    pub repaint_mask: Option<&'a Tensor>,
    pub repaint_injection_ratio: f32,
    pub repaint_crossfade_frames: usize,
}

/// Single 1D Transformer Layer with Self-Attention, Cross-Attention and AdaLN-Zero
pub struct AceStepTransformerBlock {
    self_attn: AceStepAttention,
    self_attn_norm: AudioRmsNorm,
    cross_attn: AceStepAttention,
    cross_attn_norm: AudioRmsNorm,
    mlp: AceStepMlp,
    mlp_norm: AudioRmsNorm,
    scale_shift_table: Tensor,
}

impl AceStepTransformerBlock {
    pub fn load(
        vb: VarBuilder,
        hidden_size: usize,
        intermediate_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let self_attn = AceStepAttention::load(
            vb.pp("self_attn"),
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
        )?;
        let self_attn_norm = AudioRmsNorm::new(hidden_size, 1e-6, vb.pp("self_attn_norm"))?;

        let cross_attn = AceStepAttention::load(
            vb.pp("cross_attn"),
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
        )?;
        let cross_attn_norm = AudioRmsNorm::new(hidden_size, 1e-6, vb.pp("cross_attn_norm"))?;

        let mlp = AceStepMlp::load(vb.pp("mlp"), hidden_size, intermediate_size)?;
        let mlp_norm = AudioRmsNorm::new(hidden_size, 1e-6, vb.pp("mlp_norm"))?;
        let scale_shift_table = vb.get((1, 6, hidden_size), "scale_shift_table")?;

        Ok(Self {
            self_attn,
            self_attn_norm,
            cross_attn,
            cross_attn_norm,
            mlp,
            mlp_norm,
            scale_shift_table,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        context: &Tensor,
        ada_modulation: &Tensor,
        rope: &AudioRotaryEmbedding,
        self_attn_mask: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        // ada_modulation is [batch, 6, hidden_size]; scale_shift_table is [1, 6, hidden_size].
        let mod_table = ada_modulation.broadcast_add(&self.scale_shift_table)?;
        let shift_msa = mod_table.narrow(1, 0, 1)?;
        let scale_msa = mod_table.narrow(1, 1, 1)?;
        let gate_msa = mod_table.narrow(1, 2, 1)?;
        let shift_mlp = mod_table.narrow(1, 3, 1)?;
        let scale_mlp = mod_table.narrow(1, 4, 1)?;
        let gate_mlp = mod_table.narrow(1, 5, 1)?;

        // 1. Modulated Self-Attention
        let norm1 = self.self_attn_norm.forward(x)?;
        let norm1 = norm1.broadcast_mul(&(scale_msa + 1.0)?)?.broadcast_add(&shift_msa)?;
        let attn_out = self.self_attn.forward(&norm1, None, Some(rope), self_attn_mask)?;
        let gated_attn = attn_out.broadcast_mul(&gate_msa)?;
        let mut h = (x + &gated_attn)?;

        // 2. Cross-Attention with text/lyrics context
        let norm_cross = self.cross_attn_norm.forward(&h)?;
        let cross_out = self.cross_attn.forward(&norm_cross, Some(context), None, None)?;
        h = (&h + &cross_out)?;

        // 3. Modulated MLP
        let norm2 = self.mlp_norm.forward(&h)?;
        let norm2 = norm2.broadcast_mul(&(scale_mlp + 1.0)?)?.broadcast_add(&shift_mlp)?;
        let mlp_out = self.mlp.forward(&norm2)?;
        let gated_mlp = mlp_out.broadcast_mul(&gate_mlp)?;
        Ok((&h + &gated_mlp)?)
    }
}

/// Architectural configuration of the ACE-Step 1D DiT (Turbo vs Base/SFT differ in width/depth).
#[derive(Debug, Clone)]
pub struct AceStepTransformerConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub in_channels: usize,
    pub audio_channels: usize,
    pub patch_size: usize,
    pub sliding_window: usize,
    /// Per-layer flag: `true` = sliding attention, `false` = full attention.
    pub layer_types: Vec<bool>,
    pub encoder_hidden_size: usize,
}

impl Default for AceStepTransformerConfig {
    fn default() -> Self {
        // ACE-Step 1.5 Turbo (2B DiT).
        Self {
            hidden_size: 2560,
            intermediate_size: 9728,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            in_channels: 192,
            audio_channels: 64,
            patch_size: 2,
            sliding_window: 128,
            layer_types: (0..32).map(|i| i % 2 == 0).collect(),
            encoder_hidden_size: 2048,
        }
    }
}

impl AceStepTransformerConfig {
    /// Parse from a diffusers `transformer/config.json` (falls back to defaults on missing fields).
    pub fn from_json_str(json: &str) -> Option<Self> {
        let v: serde_json::Value = serde_json::from_str(json).ok()?;
        let get = |k: &str, d: usize| v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize).unwrap_or(d);
        let hidden = get("hidden_size", 2560);
        let num_layers = get("num_hidden_layers", 32);
        let layer_types = v
            .get("layer_types")
            .and_then(|x| x.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|t| t.as_str().map(|s| s == "sliding_attention").unwrap_or(false))
                    .collect::<Vec<bool>>()
            })
            .filter(|v: &Vec<bool>| v.len() == num_layers)
            .unwrap_or_else(|| (0..num_layers).map(|i| i % 2 == 0).collect());
        Some(Self {
            hidden_size: hidden,
            intermediate_size: get("intermediate_size", 9728),
            num_layers,
            num_heads: get("num_attention_heads", 32),
            num_kv_heads: get("num_key_value_heads", 8),
            head_dim: get("head_dim", hidden / 32),
            in_channels: get("in_channels", 192),
            audio_channels: get("audio_acoustic_hidden_dim", 64),
            patch_size: get("patch_size", 2),
            sliding_window: get("sliding_window", 128),
            layer_types,
            encoder_hidden_size: get("encoder_hidden_size", 2048),
        })
    }

    pub fn from_json_file<P: AsRef<Path>>(path: P) -> Option<Self> {
        std::fs::read_to_string(path).ok().and_then(|s| Self::from_json_str(&s))
    }
}

/// ACE-Step 1.5 Turbo 1D Transformer Model
pub struct AceStepTransformer1D {
    proj_in_conv: Conv1d,
    time_embed: AceStepTimestepEmbedding,
    time_embed_r: AceStepTimestepEmbedding,
    condition_embedder: Linear,
    blocks: Vec<AceStepTransformerBlock>,
    norm_out: AudioRmsNorm,
    scale_shift_table: Tensor,
    proj_out_conv: ConvTranspose1d,
    rope: AudioRotaryEmbedding,
    sliding_layers: Vec<bool>,
    sliding_window: usize,
    pub in_channels: usize,
    pub hidden_size: usize,
    pub patch_size: usize,
}

impl AceStepTransformer1D {
    /// Load with the default (Turbo 2B) architecture.
    pub fn load(vb: VarBuilder) -> Result<Self> {
        Self::load_with_config(vb, &AceStepTransformerConfig::default())
    }

    /// Load with an explicit architecture (Turbo vs Base/SFT).
    pub fn load_with_config(vb: VarBuilder, config: &AceStepTransformerConfig) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;
        let in_channels = config.in_channels;
        let num_heads = config.num_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let num_layers = config.num_layers;
        let patch_size = config.patch_size;

        let proj_in_conv = candle_nn::conv1d(
            in_channels,
            hidden_size,
            patch_size,
            Conv1dConfig {
                stride: patch_size,
                padding: 0,
                dilation: 1,
                groups: 1,
                ..Default::default()
            },
            vb.pp("proj_in_conv"),
        )?;

        let time_embed = AceStepTimestepEmbedding::load(vb.pp("time_embed"), hidden_size)?;
        let time_embed_r = AceStepTimestepEmbedding::load(vb.pp("time_embed_r"), hidden_size)?;

        let condition_embedder =
            candle_nn::linear(config.encoder_hidden_size, hidden_size, vb.pp("condition_embedder"))?;

        let mut blocks = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let blk = AceStepTransformerBlock::load(
                vb.pp(format!("layers.{}", i)),
                hidden_size,
                intermediate_size,
                num_heads,
                num_kv_heads,
                head_dim,
            )?;
            blocks.push(blk);
        }

        let norm_out = AudioRmsNorm::new(hidden_size, 1e-6, vb.pp("norm_out"))?;
        let scale_shift_table = vb.get((1, 2, hidden_size), "scale_shift_table")?;

        let proj_out_conv = candle_nn::conv_transpose1d(
            hidden_size,
            config.audio_channels,
            patch_size,
            ConvTranspose1dConfig {
                stride: patch_size,
                padding: 0,
                output_padding: 0,
                dilation: 1,
                groups: 1,
            },
            vb.pp("proj_out_conv"),
        )?;

        let rope = AudioRotaryEmbedding::new(head_dim, 1000000.0);
        let sliding_window = config.sliding_window;
        let sliding_layers: Vec<bool> = if config.layer_types.len() == num_layers {
            config.layer_types.clone()
        } else {
            (0..num_layers).map(|i| i % 2 == 0).collect()
        };

        Ok(Self {
            proj_in_conv,
            time_embed,
            time_embed_r,
            condition_embedder,
            blocks,
            norm_out,
            scale_shift_table,
            proj_out_conv,
            rope,
            sliding_layers,
            sliding_window,
            in_channels,
            hidden_size,
            patch_size,
        })
    }

    /// Load from 2-shard safetensors files
    pub fn from_safetensors_shards<P: AsRef<Path>>(
        shard_paths: &[P],
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let refs: Vec<&Path> = shard_paths.iter().map(|p| p.as_ref()).collect();
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&refs, dtype, device)
                .with_context(|| "Failed to mmap AceStep transformer safetensors shards")?
        };
        Self::load(vb)
    }

    /// Load from shards with an explicit architecture config.
    pub fn from_safetensors_shards_with_config<P: AsRef<Path>>(
        shard_paths: &[P],
        device: &Device,
        dtype: DType,
        config: &AceStepTransformerConfig,
    ) -> Result<Self> {
        let refs: Vec<&Path> = shard_paths.iter().map(|p| p.as_ref()).collect();
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&refs, dtype, device)
                .with_context(|| "Failed to mmap AceStep transformer safetensors shards")?
        };
        Self::load_with_config(vb, config)
    }

    /// Probe the dual timestep embeddings (for bit-exact validation vs PyTorch).
    pub fn timestep_probe(
        &self,
        timestep: &Tensor,
        timestep_r: &Tensor,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let (temb_t, proj_t) = self.time_embed.forward(timestep)?;
        let (temb_r, proj_r) = self.time_embed_r.forward(&(timestep - timestep_r)?)?;
        Ok(((temb_t + temb_r)?, (proj_t + proj_r)?))
    }

    /// Forward pass through the 1D Diffusion Transformer
    ///
    /// # Arguments
    /// * `latents` - [batch, 192, time_steps]
    /// * `timestep` - [batch] (float in [0, 1])
    /// * `timestep_r` - [batch] (float in [0, 1], equals timestep for standard inference)
    /// * `condition` - [batch, context_len, 2048]
    pub fn forward(
        &self,
        latents: &Tensor,
        timestep: &Tensor,
        timestep_r: &Tensor,
        condition: &Tensor,
    ) -> candle_core::Result<Tensor> {
        // 0. Pad sequence to a multiple of patch_size (reference pads then crops back)
        let seq_len = latents.dim(2)?;
        let padded = if seq_len % self.patch_size != 0 {
            let pad = self.patch_size - (seq_len % self.patch_size);
            let zeros = Tensor::zeros((latents.dim(0)?, latents.dim(1)?, pad), latents.dtype(), latents.device())?;
            Tensor::cat(&[latents, &zeros], 2)?
        } else {
            latents.clone()
        };

        // 1. Patchify input latents: [batch, 192, T] -> [batch, 2560, T / 2] -> [batch, T / 2, 2560]
        let mut x = self.proj_in_conv.forward(&padded)?.transpose(1, 2)?;

        // 2. Dual Timestep adaLN embedding (t and t - r)
        let (temb_t, proj_t) = self.time_embed.forward(timestep)?;
        let delta_t = (timestep - timestep_r)?;
        let (temb_r, proj_r) = self.time_embed_r.forward(&delta_t)?;
        let ada_mod = (proj_t + proj_r)?;
        // Final AdaLN uses the sum of the plain (linear_2) timestep embeddings
        let temb = (&temb_t + &temb_r)?;

        // 3. Condition projection
        let ctx = self.condition_embedder.forward(condition)?;

        // 4. Pass through transformer blocks (sliding-window mask on "sliding_attention" layers)
        let patched_len = x.dim(1)?;
        let sliding_mask = if self.sliding_layers.iter().any(|&b| b) {
            Some(build_sliding_mask(patched_len, self.sliding_window, x.device())?)
        } else {
            None
        };
        for (i, blk) in self.blocks.iter().enumerate() {
            let m = if self.sliding_layers[i] { sliding_mask.as_ref() } else { None };
            x = blk.forward(&x, &ctx, &ada_mod, &self.rope, m)?;
        }

        // 5. Final norm & output projection: shift,scale = (scale_shift_table + temb.unsqueeze(1)).chunk(2)
        let norm_x = self.norm_out.forward(&x)?;
        let scale_shift = self.scale_shift_table.broadcast_add(&temb.unsqueeze(1)?)?;
        let shift = scale_shift.narrow(1, 0, 1)?;
        let scale = scale_shift.narrow(1, 1, 1)?;
        let out_modulated = norm_x.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(&shift)?;

        // De-patchify: [batch, T / 2, 2560] -> [batch, 2560, T / 2] -> [batch, 64, T]
        let out_t = out_modulated.transpose(1, 2)?;
        let out = self.proj_out_conv.forward(&out_t)?;

        // Crop back to the original (unpadded) sequence length
        out.narrow(2, 0, seq_len)
    }

    /// Flow-Matching Euler sampler (ACE-Step `infer_method="ode"`).
    ///
    /// * `condition`       - `[1, L, 2048]` cross-attention conditioning
    /// * `context_latents` - `[1, T, 128]` = `[src_latents, chunk_mask]`
    /// * `noise`           - `[1, T, 64]` initial Gaussian latent `x_t`
    /// * `t_schedule`      - descending timesteps; the final step directly computes
    ///   `x0 = x_t - v * t` (matching the reference), other steps use `dt = t - t_next`.
    ///
    /// Returns the final latents `[1, 64, T]`.
    pub fn flow_match_euler(
        &self,
        condition: &Tensor,
        context_latents: &Tensor,
        noise: &Tensor,
        t_schedule: &[f64],
    ) -> candle_core::Result<Tensor> {
        let mut xt = noise.transpose(1, 2)?.contiguous()?; // [1, 64, T]
        let ctx_t = context_latents.transpose(1, 2)?.contiguous()?; // [1, 128, T]
        let n = t_schedule.len();
        for step in 0..n {
            let t_curr = t_schedule[step];
            let timestep = Tensor::new(&[t_curr as f32], xt.device())?.to_dtype(xt.dtype())?;
            let in_latents = Tensor::cat(&[&ctx_t, &xt], 1)?; // [1, 192, T]
            let v = self.forward(&in_latents, &timestep, &timestep, condition)?;
            let dt = if step == n - 1 { t_curr } else { t_curr - t_schedule[step + 1] };
            xt = (xt - v.affine(dt, 0.0)?)?;
        }
        Ok(xt)
    }

    /// Classifier-free guided Flow-Matching Euler sampler (ACE-Step base/sft models).
    ///
    /// Runs the conditional and null (unconditional) velocity predictions as a batch of
    /// two and combines them with Adaptive Projected Guidance (`apg_forward`), matching
    /// the reference `AceStepConditionGenerationModel.generate_audio` for non-turbo models.
    pub fn flow_match_euler_cfg(
        &self,
        condition: &Tensor,
        null_condition: &Tensor,
        context_latents: &Tensor,
        noise: &Tensor,
        t_schedule: &[f64],
        guidance_scale: f32,
    ) -> candle_core::Result<Tensor> {
        let mut xt = noise.transpose(1, 2)?.contiguous()?; // [1, 64, T]
        let cond2 = Tensor::cat(&[condition, null_condition], 0)?; // [2, L, 2048]
        let ctx_t = context_latents.transpose(1, 2)?.contiguous()?; // [1, 128, T]
        let ctx2 = Tensor::cat(&[&ctx_t, &ctx_t], 0)?; // [2, 128, T]
        let n = t_schedule.len();
        let mut momentum: Option<Tensor> = None;
        for step in 0..n {
            let t_curr = t_schedule[step];
            let timestep =
                Tensor::new(&[t_curr as f32, t_curr as f32], xt.device())?.to_dtype(xt.dtype())?;
            let x2 = Tensor::cat(&[&xt, &xt], 0)?; // [2, 64, T]
            let in_latents = Tensor::cat(&[&ctx2, &x2], 1)?; // [2, 192, T]
            let v = self.forward(&in_latents, &timestep, &timestep, &cond2)?; // [2, 64, T]
            // APG operates on `[B, T, C]` tensors with `dims=[1]` (projection over time),
            // matching the reference `generate_audio`. Our DiT output is channel-major.
            let v_t = v.transpose(1, 2)?.contiguous()?; // [2, T, 64]
            let pred_cond = v_t.narrow(0, 0, 1)?; // [1, T, 64]
            let pred_uncond = v_t.narrow(0, 1, 1)?; // [1, T, 64]
            let vt = apg_forward(&pred_cond, &pred_uncond, guidance_scale, &mut momentum)?; // [1, T, 64]
            let vt = vt.transpose(1, 2)?.contiguous()?; // [1, 64, T]
            let dt = if step == n - 1 { t_curr } else { t_curr - t_schedule[step + 1] };
            xt = (xt - vt.affine(dt, 0.0)?)?;
        }
        Ok(xt)
    }

    /// One denoising step: returns the (CFG-guided) velocity `[1, 64, T]`.
    #[allow(clippy::too_many_arguments)]
    fn sample_velocity(
        &self,
        condition: &Tensor,
        null_condition: Option<&Tensor>,
        ctx_t: &Tensor, // [1, 128, T]
        xt: &Tensor,    // [1, 64, T]
        t_curr: f64,
        guidance_scale: f32,
        momentum: &mut Option<Tensor>,
    ) -> candle_core::Result<Tensor> {
        let dev = xt.device();
        let dt = xt.dtype();
        if let Some(null) = null_condition {
            let cond2 = Tensor::cat(&[condition, null], 0)?;
            let ctx2 = Tensor::cat(&[ctx_t, ctx_t], 0)?;
            let x2 = Tensor::cat(&[xt, xt], 0)?;
            let in_latents = Tensor::cat(&[&ctx2, &x2], 1)?;
            let ts = Tensor::new(&[t_curr as f32, t_curr as f32], dev)?.to_dtype(dt)?;
            let v = self.forward(&in_latents, &ts, &ts, &cond2)?; // [2, 64, T]
            let v_t = v.transpose(1, 2)?.contiguous()?; // [2, T, 64]
            let guided = apg_forward(&v_t.narrow(0, 0, 1)?, &v_t.narrow(0, 1, 1)?, guidance_scale, momentum)?;
            guided.transpose(1, 2)?.contiguous()
        } else {
            let in_latents = Tensor::cat(&[ctx_t, xt], 1)?;
            let ts = Tensor::new(&[t_curr as f32], dev)?.to_dtype(dt)?;
            self.forward(&in_latents, &ts, &ts, condition)
        }
    }

    /// Generic Flow-Matching Euler sampler (Turbo & Base/XL) with cover / repaint extras.
    pub fn flow_match(&self, cfg: &FlowMatchConfig) -> candle_core::Result<Tensor> {
        let mut xt = cfg.noise.transpose(1, 2)?.contiguous()?; // [1,64,T]
        let noise_t = cfg.noise.transpose(1, 2)?.contiguous()?;
        let n = cfg.t_schedule.len();

        let src_t = match cfg.clean_src {
            Some(s) => Some(s.transpose(1, 2)?.contiguous()?),
            None => None,
        };

        // Cover-noise initialization: renoise src at the nearest timestep, truncate schedule.
        let mut start_step = 0usize;
        if cfg.cover_noise_strength > 0.0 {
            if let Some(src_t) = &src_t {
                let eff = 1.0 - cfg.cover_noise_strength as f64;
                let (idx, t) = cfg
                    .t_schedule
                    .iter()
                    .enumerate()
                    .min_by(|a, b| (a.1 - eff).abs().partial_cmp(&(b.1 - eff).abs()).unwrap())
                    .map(|(i, t)| (i, *t))
                    .unwrap();
                xt = ((&noise_t * t)? + (src_t * (1.0 - t))?)?;
                start_step = idx;
            }
        }

        let cover_steps = (n as f64 * cfg.cover_strength as f64) as usize;
        let mut momentum: Option<Tensor> = None;
        let mut switched = false;
        let mut cur_cond = cfg.condition;
        let mut cur_null = cfg.null_condition;
        let mut cur_ctx = cfg.context_latents.transpose(1, 2)?.contiguous()?;

        for step in start_step..n {
            if !switched
                && step >= cover_steps
                && cfg.condition_non_cover.is_some()
                && cfg.context_latents_non_cover.is_some()
            {
                switched = true;
                cur_cond = cfg.condition_non_cover.unwrap();
                cur_null = cfg.null_condition_non_cover;
                cur_ctx = cfg.context_latents_non_cover.unwrap().transpose(1, 2)?.contiguous()?;
                momentum = None;
            }
            let t_curr = cfg.t_schedule[step];
            let dt_step = if step == n - 1 { t_curr } else { t_curr - cfg.t_schedule[step + 1] };
            let vt = self.sample_velocity(cur_cond, cur_null, &cur_ctx, &xt, t_curr, cfg.guidance_scale, &mut momentum)?;
            xt = (xt - vt.affine(dt_step, 0.0)?)?;

            // Repaint step injection on the first `injection_cutoff` steps.
            if let (Some(mask), Some(src_t)) = (cfg.repaint_mask, &src_t) {
                let cutoff = (cfg.repaint_injection_ratio * n as f32).round() as usize;
                if step < cutoff {
                    let t_after = if step == n - 1 { 0.0 } else { cfg.t_schedule[step + 1] };
                    let zt = ((&noise_t * t_after)? + (src_t * (1.0 - t_after))?)?;
                    xt = mask_blend(mask, &xt, &zt)?;
                }
            }
        }

        // Final repaint boundary blend (soft crossfade).
        if let (Some(mask), Some(src_t)) = (cfg.repaint_mask, &src_t) {
            if cfg.repaint_crossfade_frames > 0 {
                let soft = soft_repaint_mask(mask, cfg.repaint_crossfade_frames)?
                    .unsqueeze(1)?
                    .to_dtype(xt.dtype())?
                    .broadcast_as(xt.shape())?;
                let inv = soft.affine(-1.0, 1.0)?;
                xt = ((xt.broadcast_mul(&soft)? + src_t.broadcast_mul(&inv)?))?;
            }
        }
        Ok(xt)
    }
}

/// `xt*mask + zt*(1-mask)` with `mask` broadcast to `xt`.
fn mask_blend(mask: &Tensor, xt: &Tensor, zt: &Tensor) -> candle_core::Result<Tensor> {
    let m = mask.to_dtype(xt.dtype())?.unsqueeze(1)?.broadcast_as(xt.shape())?;
    let inv = m.affine(-1.0, 1.0)?;
    xt.broadcast_mul(&m)? + zt.broadcast_mul(&inv)?
}

/// Build a soft repaint mask with linear crossfade ramps at the repaint boundaries.
fn soft_repaint_mask(mask: &Tensor, crossfade: usize) -> candle_core::Result<Tensor> {
    let dims = mask.dims().to_vec();
    let t = *dims.last().unwrap();
    let m: Vec<f32> = mask.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let mut soft = m.clone();
    let idxs: Vec<usize> = m.iter().enumerate().filter(|(_, &v)| v > 0.5).map(|(i, _)| i).collect();
    if !idxs.is_empty() && idxs.len() < t {
        let left = idxs[0];
        let right = idxs[idxs.len() - 1] + 1;
        let fs = left.saturating_sub(crossfade);
        for k in 0..(left - fs) {
            soft[fs + k] = (k as f32 + 1.0) / ((left - fs) as f32 + 1.0);
        }
        let fe = (right + crossfade).min(t);
        for k in 0..(fe - right) {
            soft[right + k] = 1.0 - (k as f32 + 1.0) / ((fe - right) as f32 + 1.0);
        }
    }
    Tensor::from_vec(soft, dims, mask.device())
}

/// Single Lyric Encoder Transformer Layer (2048 hidden, 6144 intermediate, 16 heads)
pub struct AceStepLyricLayer {
    self_attn: AceStepAttention,
    input_layernorm: AudioRmsNorm,
    mlp: AceStepMlp,
    post_attention_layernorm: AudioRmsNorm,
}

impl AceStepLyricLayer {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let hidden_size = 2048;
        let intermediate_size = 6144;
        let num_heads = 16;
        let num_kv_heads = 8;
        let head_dim = 128;

        let self_attn = AceStepAttention::load(
            vb.pp("self_attn"),
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
        )?;
        let input_layernorm = AudioRmsNorm::new(hidden_size, 1e-6, vb.pp("input_layernorm"))?;
        let mlp = AceStepMlp::load(vb.pp("mlp"), hidden_size, intermediate_size)?;
        let post_attention_layernorm = AudioRmsNorm::new(hidden_size, 1e-6, vb.pp("post_attention_layernorm"))?;

        Ok(Self {
            self_attn,
            input_layernorm,
            mlp,
            post_attention_layernorm,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        rope: &AudioRotaryEmbedding,
        self_attn_mask: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        let norm_x = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward(&norm_x, None, Some(rope), self_attn_mask)?;
        let h = (x + attn_out)?;

        let norm_h = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&norm_h)?;
        &h + mlp_out
    }
}

/// AceStepConditionEncoder: builds the [B, L, 2048] cross-attention conditioning
/// from text_hidden (Qwen3 caption), lyric_hidden (Qwen3 lyric embeddings), and timbre
/// (reference-audio latents, or the learned `silence_latent` for text2music).
pub struct AceStepConditionEncoder {
    pub text_projector: Linear,
    pub lyric_embed_tokens: Linear,
    pub lyric_layers: Vec<AceStepLyricLayer>,
    pub lyric_norm: AudioRmsNorm,
    pub timbre_embed_tokens: Linear,
    pub timbre_special_token: Tensor,
    pub timbre_layers: Vec<AceStepLyricLayer>,
    pub timbre_norm: AudioRmsNorm,
    pub silence_latent: Tensor,
    pub null_condition_emb: Tensor,
    pub rope: AudioRotaryEmbedding,
}

impl AceStepConditionEncoder {
    pub const TIMBRE_FIX_FRAME: usize = 750;
    pub const SLIDING_WINDOW: usize = 128;

    pub fn load(vb: VarBuilder) -> Result<Self> {
        let text_projector = candle_nn::linear_no_bias(1024, 2048, vb.pp("text_projector"))?;

        // Lyric encoder (8 bidirectional layers, Qwen3-style attention + RoPE).
        let vb_lyric = vb.pp("lyric_encoder");
        let lyric_embed_tokens = candle_nn::linear(1024, 2048, vb_lyric.pp("embed_tokens"))?;
        let mut lyric_layers = Vec::with_capacity(8);
        for i in 0..8 {
            lyric_layers.push(AceStepLyricLayer::load(vb_lyric.pp(format!("layers.{}", i)))?);
        }
        let lyric_norm = AudioRmsNorm::new(2048, 1e-6, vb_lyric.pp("norm"))?;

        // Timbre encoder (4 bidirectional layers, 64 -> 2048 projection).
        let vb_timbre = vb.pp("timbre_encoder");
        let timbre_embed_tokens = candle_nn::linear(64, 2048, vb_timbre.pp("embed_tokens"))?;
        let timbre_special_token = vb_timbre.get((1, 1, 2048), "special_token")?;
        let mut timbre_layers = Vec::with_capacity(4);
        for i in 0..4 {
            timbre_layers.push(AceStepLyricLayer::load(vb_timbre.pp(format!("layers.{}", i)))?);
        }
        let timbre_norm = AudioRmsNorm::new(2048, 1e-6, vb_timbre.pp("norm"))?;

        let silence_latent = vb.get((1, 15000, 64), "silence_latent")?;
        let null_condition_emb = vb.get((1, 1, 2048), "null_condition_emb")?;

        let rope = AudioRotaryEmbedding::new(128, 1000000.0);

        Ok(Self {
            text_projector,
            lyric_embed_tokens,
            lyric_layers,
            lyric_norm,
            timbre_embed_tokens,
            timbre_special_token,
            timbre_layers,
            timbre_norm,
            silence_latent,
            null_condition_emb,
            rope,
        })
    }

    pub fn from_safetensors<P: AsRef<Path>>(
        weights_path: P,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path.as_ref()], dtype, device)
                .with_context(|| format!("Failed to load condition encoder at {:?}", weights_path.as_ref()))?
        };
        Self::load(vb)
    }

    /// Project Qwen3 description embeddings [batch, seq_len, 1024] -> [batch, seq_len, 2048]
    pub fn forward_text(&self, text_embeds: &Tensor) -> candle_core::Result<Tensor> {
        self.text_projector.forward(text_embeds)
    }

    /// Encode lyrics through 8 Lyric Transformer layers into [batch, seq_len, 2048]
    pub fn forward_lyrics(&self, lyric_embeds: &Tensor) -> candle_core::Result<Tensor> {
        let mut h = self.lyric_embed_tokens.forward(lyric_embeds)?;
        let s = h.dim(1)?;
        let sliding = if s > Self::SLIDING_WINDOW + 1 {
            Some(build_sliding_mask(s, Self::SLIDING_WINDOW, h.device())?)
        } else {
            None
        };
        for (i, layer) in self.lyric_layers.iter().enumerate() {
            let m = if i % 2 == 0 { sliding.as_ref() } else { None };
            h = layer.forward(&h, &self.rope, m)?;
        }
        self.lyric_norm.forward(&h)
    }

    /// Encode reference-audio latents [1, T, 64] into a timbre embedding [1, 1, 2048]
    /// (first-token output of the timbre encoder, matching the reference).
    pub fn encode_timbre(&self, ref_latents: &Tensor) -> candle_core::Result<Tensor> {
        let mut h = self.timbre_embed_tokens.forward(ref_latents)?;
        let s = h.dim(1)?;
        let sliding = if s > Self::SLIDING_WINDOW + 1 {
            Some(build_sliding_mask(s, Self::SLIDING_WINDOW, h.device())?)
        } else {
            None
        };
        for (i, layer) in self.timbre_layers.iter().enumerate() {
            let m = if i % 2 == 0 { sliding.as_ref() } else { None };
            h = layer.forward(&h, &self.rope, m)?;
        }
        let h = self.timbre_norm.forward(&h)?;
        h.narrow(1, 0, 1)
    }

    /// Full text2music conditioning: `[lyric_encoded, timbre, text_projected]`
    /// (valid tokens first; with all-ones masks this equals the reference packing order).
    pub fn forward_condition(
        &self,
        text_embeds: &Tensor,
        lyric_embeds: &Tensor,
    ) -> candle_core::Result<Tensor> {
        self.forward_condition_ex(text_embeds, lyric_embeds, None)
    }

    /// Like [`Self::forward_condition`] but with explicit reference-audio latents for the
    /// timbre encoder (`None` uses the learned `silence_latent`).
    pub fn forward_condition_ex(
        &self,
        text_embeds: &Tensor,
        lyric_embeds: &Tensor,
        refer_latents: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        let text_p = self.text_projector.forward(text_embeds)?;
        let lyric_e = self.forward_lyrics(lyric_embeds)?;
        let ref_lat = match refer_latents {
            Some(r) => r.clone(),
            None => self.silence_latent.narrow(1, 0, Self::TIMBRE_FIX_FRAME)?,
        };
        let timbre = self.encode_timbre(&ref_lat)?;
        Tensor::cat(&[&lyric_e, &timbre, &text_p], 1)
    }

    /// Default forward: text projection
    pub fn forward(&self, text_embeds: &Tensor) -> candle_core::Result<Tensor> {
        self.forward_text(text_embeds)
    }
}
