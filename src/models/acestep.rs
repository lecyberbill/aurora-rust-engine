// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Pure Rust ACE-Step 1.5 Turbo 1D Transformer Model for Audio/Music Diffusion

use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Linear, RmsNorm, VarBuilder};
use std::path::Path;

/// Timestep sinusoidal embedding generator for 1D diffusion
pub fn get_timestep_embedding(timesteps: &Tensor, embedding_dim: usize) -> candle_core::Result<Tensor> {
    let half_dim = embedding_dim / 2;
    let factor = (-(10000.0f64.ln()) / (half_dim as f64 - 1.0)).exp();
    let dev = timesteps.device();
    let mut freqs_vec = Vec::with_capacity(half_dim);
    let mut cur = 1.0f64;
    for _ in 0..half_dim {
        freqs_vec.push(cur as f32);
        cur *= factor;
    }
    let freqs = Tensor::new(freqs_vec.as_slice(), dev)?;
    let args = timesteps.unsqueeze(1)?.broadcast_mul(&freqs.unsqueeze(0)?)?;
    let sin = args.sin()?;
    let cos = args.cos()?;
    Tensor::cat(&[&sin, &cos], 1)
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

        let cos = cos.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, seq_len, half_dim]
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

        let out1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let out2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;
        Tensor::cat(&[&out1, &out2], 3)
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
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        let intermediate = (gate * up)?;
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
    norm_q: RmsNorm,
    norm_k: RmsNorm,
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
        let norm_q = candle_nn::rms_norm(head_dim, 1e-6, vb.pp("norm_q"))?;
        let norm_k = candle_nn::rms_norm(head_dim, 1e-6, vb.pp("norm_k"))?;

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

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let att = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let att = candle_nn::ops::softmax(&att, D::Minus1)?;
        let out = att.matmul(&v)?;
        let out = out.transpose(1, 2)?.reshape((b, s, self.num_heads * self.head_dim))?;
        self.to_out.forward(&out)
    }
}

/// Single 1D Transformer Layer with Self-Attention, Cross-Attention and AdaLN-Zero
pub struct AceStepTransformerBlock {
    self_attn: AceStepAttention,
    self_attn_norm: RmsNorm,
    cross_attn: AceStepAttention,
    cross_attn_norm: RmsNorm,
    mlp: AceStepMlp,
    mlp_norm: RmsNorm,
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
        let self_attn_norm = candle_nn::rms_norm(hidden_size, 1e-6, vb.pp("self_attn_norm"))?;

        let cross_attn = AceStepAttention::load(
            vb.pp("cross_attn"),
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
        )?;
        let cross_attn_norm = candle_nn::rms_norm(hidden_size, 1e-6, vb.pp("cross_attn_norm"))?;

        let mlp = AceStepMlp::load(vb.pp("mlp"), hidden_size, intermediate_size)?;
        let mlp_norm = candle_nn::rms_norm(hidden_size, 1e-6, vb.pp("mlp_norm"))?;
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
    ) -> candle_core::Result<Tensor> {
        // ada_modulation is [batch, 6, hidden_size]
        let mod_table = (ada_modulation + &self.scale_shift_table)?;
        let shift_msa = mod_table.narrow(1, 0, 1)?;
        let scale_msa = mod_table.narrow(1, 1, 1)?;
        let gate_msa = mod_table.narrow(1, 2, 1)?;
        let shift_mlp = mod_table.narrow(1, 3, 1)?;
        let scale_mlp = mod_table.narrow(1, 4, 1)?;
        let gate_mlp = mod_table.narrow(1, 5, 1)?;

        // 1. Modulated Self-Attention
        let norm1 = self.self_attn_norm.forward(x)?;
        let norm1 = norm1.broadcast_mul(&(scale_msa + 1.0)?)?.broadcast_add(&shift_msa)?;
        let attn_out = self.self_attn.forward(&norm1, None, Some(rope))?;
        let mut h = (x + attn_out.broadcast_mul(&gate_msa)?)?;

        // 2. Cross-Attention with text/lyrics context
        let norm_cross = self.cross_attn_norm.forward(&h)?;
        let cross_out = self.cross_attn.forward(&norm_cross, Some(context), None)?;
        h = (&h + cross_out)?;

        // 3. Modulated MLP
        let norm2 = self.mlp_norm.forward(&h)?;
        let norm2 = norm2.broadcast_mul(&(scale_mlp + 1.0)?)?.broadcast_add(&shift_mlp)?;
        let mlp_out = self.mlp.forward(&norm2)?;
        &h + mlp_out.broadcast_mul(&gate_mlp)?
    }
}

/// ACE-Step 1.5 Turbo 1D Transformer Model
pub struct AceStepTransformer1D {
    proj_in_conv: Conv1d,
    time_linear1: Linear,
    time_linear2: Linear,
    time_proj: Linear,
    condition_embedder: Linear,
    blocks: Vec<AceStepTransformerBlock>,
    norm_out: RmsNorm,
    scale_shift_table: Tensor,
    proj_out_conv: ConvTranspose1d,
    rope: AudioRotaryEmbedding,
    pub in_channels: usize,
    pub hidden_size: usize,
    pub patch_size: usize,
}

impl AceStepTransformer1D {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let hidden_size = 2560;
        let intermediate_size = 9728;
        let in_channels = 192;
        let num_heads = 32;
        let num_kv_heads = 8;
        let head_dim = 128;
        let num_layers = 32;
        let patch_size = 2;

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

        let time_linear1 = candle_nn::linear(256, hidden_size, vb.pp("time_embed.linear_1"))?;
        let time_linear2 = candle_nn::linear(hidden_size, hidden_size, vb.pp("time_embed.linear_2"))?;
        let time_proj = candle_nn::linear(hidden_size, 6 * hidden_size, vb.pp("time_embed.time_proj"))?;

        let condition_embedder = candle_nn::linear(2048, hidden_size, vb.pp("condition_embedder"))?;

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

        let norm_out = candle_nn::rms_norm(hidden_size, 1e-6, vb.pp("norm_out"))?;
        let scale_shift_table = vb.get((1, 2, hidden_size), "scale_shift_table")?;

        let proj_out_conv = candle_nn::conv_transpose1d(
            hidden_size,
            64,
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

        Ok(Self {
            proj_in_conv,
            time_linear1,
            time_linear2,
            time_proj,
            condition_embedder,
            blocks,
            norm_out,
            scale_shift_table,
            proj_out_conv,
            rope,
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

    /// Forward pass through the 1D Diffusion Transformer
    ///
    /// # Arguments
    /// * `latents` - [batch, 192, time_steps]
    /// * `timestep` - [batch] (float or integer timestep in [0, 1000])
    /// * `condition` - [batch, context_len, 2048]
    pub fn forward(
        &self,
        latents: &Tensor,
        timestep: &Tensor,
        condition: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let (b, _, _) = latents.dims3()?;

        // 1. Patchify input latents: [batch, 192, T] -> [batch, 2560, T / 2] -> [batch, T / 2, 2560]
        let mut x = self.proj_in_conv.forward(latents)?.transpose(1, 2)?;

        // 2. Timestep adaLN embedding
        let t_emb = get_timestep_embedding(timestep, 256)?;
        let t_hid = candle_nn::ops::silu(&self.time_linear1.forward(&t_emb)?)?;
        let t_hid = self.time_linear2.forward(&t_hid)?;
        let ada_mod = self.time_proj.forward(&t_hid)?
            .reshape((b, 6, self.hidden_size))?;

        // 3. Condition projection
        let ctx = self.condition_embedder.forward(condition)?;

        // 4. Pass through 32 Transformer Blocks
        for blk in &self.blocks {
            x = blk.forward(&x, &ctx, &ada_mod, &self.rope)?;
        }

        // 5. Final norm & output projection
        let norm_x = self.norm_out.forward(&x)?;
        let scale_shift = self.scale_shift_table.squeeze(0)?;
        let shift = scale_shift.narrow(0, 0, 1)?.unsqueeze(0)?;
        let scale = scale_shift.narrow(0, 1, 1)?.unsqueeze(0)?;
        let out_modulated = norm_x.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(&shift)?;

        // De-patchify: [batch, T / 2, 2560] -> [batch, 2560, T / 2] -> [batch, 64, T]
        let out_t = out_modulated.transpose(1, 2)?;
        self.proj_out_conv.forward(&out_t)
    }
}
