// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Pure Rust CogVideoX-2B/5B Spatio-Temporal 3D DiT Transformer Architecture
// Invariant: Non-intrusive AdaLN-single 6-factor modulation, 3D spatio-temporal attention, exact unpatchify inversion.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{conv2d, linear, Conv2d, Conv2dConfig, Linear, VarBuilder};

/// CogVideoX Transformer 3D Configuration
#[derive(Debug, Clone)]
pub struct CogVideoXConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub patch_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub hidden_size: usize,
    pub text_embed_dim: usize,
    pub time_embed_dim: usize,
    pub sample_frames: usize,
    pub temporal_compression_ratio: usize,
    pub use_rotary_positional_embeddings: bool,
    pub use_positional_embeddings: bool,
    pub max_text_seq_length: usize,
}

impl Default for CogVideoXConfig {
    fn default() -> Self {
        // CogVideoX-2B parameters: additive 3D sincos pos embed, NO RoPE
        let num_heads = 30;
        let head_dim = 64;
        Self {
            in_channels: 16,
            out_channels: 16,
            patch_size: 2,
            num_layers: 30,
            num_attention_heads: num_heads,
            attention_head_dim: head_dim,
            hidden_size: num_heads * head_dim, // 1920
            text_embed_dim: 4096,               // T5-XXL
            time_embed_dim: 512,
            sample_frames: 49,
            temporal_compression_ratio: 4,
            use_rotary_positional_embeddings: false,
            use_positional_embeddings: true,
            max_text_seq_length: 226,
        }
    }
}

/// Official CogVideoX GELU with tanh approximation (gelu-approximate)
pub fn gelu_approximate(x: &Tensor) -> Result<Tensor> {
    let sqrt_2_over_pi = 0.7978845608028654f64;
    let x_f32 = x.to_dtype(DType::F32)?;
    let x_cube = (x_f32.broadcast_mul(&x_f32)? * &x_f32)?;
    let inner = ((&x_f32 + &(&x_cube * 0.044715)?)? * sqrt_2_over_pi)?;
    let tanh = inner.tanh()?;
    let one_plus_tanh = (&tanh + 1.0)?;
    let half_x = (&x_f32 * 0.5)?;
    let out = (&half_x * &one_plus_tanh)?;
    out.to_dtype(x.dtype())
}

/// CogVideoX Patch & Text Embedding
#[derive(Debug, Clone)]
pub struct CogVideoXPatchEmbed {
    proj: Conv2d,
    text_proj: Linear,
    patch_size: usize,
    pub pos_embedding: Option<Tensor>,
}

impl CogVideoXPatchEmbed {
    pub fn new(cfg: &CogVideoXConfig, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            stride: cfg.patch_size,
            ..Default::default()
        };
        let proj = conv2d(cfg.in_channels, cfg.hidden_size, cfg.patch_size, conv_cfg, vb.pp("proj"))?;
        let text_proj = linear(cfg.text_embed_dim, cfg.hidden_size, vb.pp("text_proj"))?;
        Ok(Self {
            proj,
            text_proj,
            patch_size: cfg.patch_size,
            pos_embedding: None,
        })
    }

    pub fn set_pos_embedding(&mut self, pos_emb: Tensor) {
        self.pos_embedding = Some(pos_emb);
    }

    /// Forward pass for video latents [B, C, T, H, W] -> [B, T * (H/2) * (W/2), Hidden]
    pub fn forward_video(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, t, h, w) = x.dims5()?;
        // Reshape [B, C, T, H, W] -> [B * T, C, H, W] for 2D patch convolution
        let x_2d = x.transpose(1, 2)?.reshape((b * t, c, h, w))?;
        let proj = self.proj.forward(&x_2d)?; // [B * T, Hidden, H / 2, W / 2]
        let (_, hidden, p_h, p_w) = proj.dims4()?;
        let tokens = proj
            .flatten_from(2)? // [B * T, Hidden, p_h * p_w]
            .transpose(1, 2)? // [B * T, p_h * p_w, Hidden]
            .reshape((b, t * p_h * p_w, hidden))?; // [B, Seq_Video, Hidden]
        Ok(tokens)
    }

    /// Forward pass for text context [B, Seq_Text, text_embed_dim] -> [B, Seq_Text, Hidden]
    pub fn forward_text(&self, text_context: &Tensor) -> Result<Tensor> {
        self.text_proj.forward(text_context)
    }
}

/// CogVideoX Timestep Sinusoidal & MLP Embedder
#[derive(Debug, Clone)]
pub struct CogVideoXTimeEmbed {
    linear_1: Linear,
    linear_2: Linear,
    freq_dim: usize,
}

impl CogVideoXTimeEmbed {
    pub fn new(freq_dim: usize, time_embed_dim: usize, vb: VarBuilder) -> Result<Self> {
        let linear_1 = linear(freq_dim, time_embed_dim, vb.pp("linear_1"))?;
        let linear_2 = linear(time_embed_dim, time_embed_dim, vb.pp("linear_2"))?;
        Ok(Self {
            linear_1,
            linear_2,
            freq_dim,
        })
    }

    /// Compute sinusoidal timestep embeddings and forward MLP
    pub fn forward(&self, timestep: &Tensor) -> Result<Tensor> {
        let dev = timestep.device();
        let dtype = timestep.dtype();
        let half_dim = self.freq_dim / 2;
        let mut freqs = Vec::with_capacity(half_dim);
        for i in 0..half_dim {
            let exponent = (2 * i) as f64 / self.freq_dim as f64;
            freqs.push((1.0 / 10000.0f64.powf(exponent)) as f32);
        }
        let freqs_t = Tensor::new(&freqs[..], dev)?;
        let t_f32 = timestep.to_dtype(DType::F32)?;
        let angles = t_f32.unsqueeze(1)?.broadcast_mul(&freqs_t.unsqueeze(0)?)?; // [B, half_dim]
        let cos = angles.cos()?;
        let sin = angles.sin()?;
        let emb = Tensor::cat(&[&cos, &sin], 1)?.to_dtype(dtype)?; // [B, freq_dim]

        let h1 = self.linear_1.forward(&emb)?;
        let act = candle_nn::ops::silu(&h1)?;
        self.linear_2.forward(&act)
    }
}

/// Universal LayerNorm using pure tensor math to ensure 100% hardware compatibility across ROCm, CUDA, and CPU
#[derive(Debug, Clone)]
pub struct CogVideoXLayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl CogVideoXLayerNorm {
    pub fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        let bias = vb.get(dim, "bias")?;
        Ok(Self { weight, bias, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let orig_dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let mean = x_f32.mean_keepdim(candle_core::D::Minus1)?;
        let diff = x_f32.broadcast_sub(&mean)?;
        let var = diff.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let std = (var + self.eps)?.sqrt()?;
        let norm = diff.broadcast_div(&std)?.to_dtype(orig_dtype)?;
        let scaled = norm.broadcast_mul(&self.weight)?;
        scaled.broadcast_add(&self.bias)
    }
}

/// CogVideoX Single 3D Transformer Block with AdaLN-Single Modulation
#[derive(Debug, Clone)]
pub struct CogVideoXBlock3D {
    norm1_linear: Linear,
    norm1_norm: CogVideoXLayerNorm,
    norm2_linear: Linear,
    norm2_norm: CogVideoXLayerNorm,
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    norm_q: CogVideoXLayerNorm,
    norm_k: CogVideoXLayerNorm,
    ff_proj: Linear,
    ff_out: Linear,
    num_heads: usize,
    head_dim: usize,
    hidden_size: usize,
}

impl CogVideoXBlock3D {
    pub fn new(cfg: &CogVideoXConfig, vb: VarBuilder) -> Result<Self> {
        let norm1_linear = linear(cfg.time_embed_dim, cfg.hidden_size * 6, vb.pp("norm1.linear"))?;
        let norm1_norm = CogVideoXLayerNorm::new(cfg.hidden_size, 1e-5, vb.pp("norm1.norm"))?;

        let to_q = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("attn1.to_q"))?;
        let to_k = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("attn1.to_k"))?;
        let to_v = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("attn1.to_v"))?;
        let to_out = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("attn1.to_out.0"))?;

        let norm_q = CogVideoXLayerNorm::new(cfg.attention_head_dim, 1e-5, vb.pp("attn1.norm_q"))?;
        let norm_k = CogVideoXLayerNorm::new(cfg.attention_head_dim, 1e-5, vb.pp("attn1.norm_k"))?;

        let norm2_linear = linear(cfg.time_embed_dim, cfg.hidden_size * 6, vb.pp("norm2.linear"))?;
        let norm2_norm = CogVideoXLayerNorm::new(cfg.hidden_size, 1e-5, vb.pp("norm2.norm"))?;

        let intermediate_dim = cfg.hidden_size * 4; // 7680
        let ff_proj = linear(cfg.hidden_size, intermediate_dim, vb.pp("ff.net.0.proj"))?;
        let ff_out = linear(intermediate_dim, cfg.hidden_size, vb.pp("ff.net.2"))?;

        Ok(Self {
            norm1_linear,
            norm1_norm,
            norm2_linear,
            norm2_norm,
            to_q,
            to_k,
            to_v,
            to_out,
            norm_q,
            norm_k,
            ff_proj,
            ff_out,
            num_heads: cfg.num_attention_heads,
            head_dim: cfg.attention_head_dim,
            hidden_size: cfg.hidden_size,
        })
    }

    /// Forward pass of 3D transformer block with dual-stream text and video modulation
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        temb: &Tensor,
        image_rotary_emb: Option<(&Tensor, &Tensor)>,
    ) -> Result<(Tensor, Tensor)> {
        let (b, _video_seq, _) = hidden_states.dims3()?;
        let text_seq_len = encoder_hidden_states.dim(1)?;

        // 1. AdaLN modulation 1 for Attention
        let temb_act = candle_nn::ops::silu(temb)?;
        let mod1 = self.norm1_linear.forward(&temb_act)?;
        let chunks1 = mod1.chunk(6, 1)?;
        let shift1 = chunks1[0].unsqueeze(1)?;
        let scale1 = chunks1[1].unsqueeze(1)?;
        let gate_msa = chunks1[2].unsqueeze(1)?;
        let enc_shift1 = chunks1[3].unsqueeze(1)?;
        let enc_scale1 = chunks1[4].unsqueeze(1)?;
        let enc_gate_msa = chunks1[5].unsqueeze(1)?;

        let norm_hidden = self.norm1_norm.forward(hidden_states)?;
        let mod_hidden = norm_hidden.broadcast_mul(&(scale1 + 1.0)?)?.broadcast_add(&shift1)?;

        let norm_enc = self.norm1_norm.forward(encoder_hidden_states)?;
        let mod_enc = norm_enc.broadcast_mul(&(enc_scale1 + 1.0)?)?.broadcast_add(&enc_shift1)?;

        // 2. Concatenate [Text, Video] for Joint Spatio-temporal Attention
        let attn_in = Tensor::cat(&[&mod_enc, &mod_hidden], 1)?;
        let total_seq = attn_in.dim(1)?;

        let q = self.to_q.forward(&attn_in)?;
        let k = self.to_k.forward(&attn_in)?;
        let v = self.to_v.forward(&attn_in)?;

        // Reshape to [B, Heads, Seq, Head_Dim]
        let q = q.reshape((b, total_seq, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let k = k.reshape((b, total_seq, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let v = v.reshape((b, total_seq, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;

        // QK-Norm per head
        let q = self.norm_q.forward(&q)?;
        let k = self.norm_k.forward(&k)?;

        // Apply 3D Rotary Position Embedding to video queries and keys only
        let (q, k) = if let Some((cos, sin)) = image_rotary_emb {
            let q_text = q.narrow(2, 0, text_seq_len)?;
            let q_vid = q.narrow(2, text_seq_len, total_seq - text_seq_len)?;
            let q_vid_rot = crate::diffusion::video_rope::apply_cogvideox_rope(&q_vid, cos, sin)?;
            let q_full = Tensor::cat(&[&q_text, &q_vid_rot], 2)?;

            let k_text = k.narrow(2, 0, text_seq_len)?;
            let k_vid = k.narrow(2, text_seq_len, total_seq - text_seq_len)?;
            let k_vid_rot = crate::diffusion::video_rope::apply_cogvideox_rope(&k_vid, cos, sin)?;
            let k_full = Tensor::cat(&[&k_text, &k_vid_rot], 2)?;

            (q_full, k_full)
        } else {
            (q, k)
        };

        // Attention computation: Q @ K.T / sqrt(head_dim) with sequence chunking
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let q_dtype = q.dtype();
        let q_scaled = (q * scale)?;
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let v_cont = v.contiguous()?;

        let chunk_size = 2048;
        let attn_out = if total_seq <= chunk_size {
            let attn_weights = q_scaled.matmul(&k_t)?;
            let attn_probs = candle_nn::ops::softmax(&attn_weights.to_dtype(DType::F32)?, candle_core::D::Minus1)?
                .to_dtype(q_dtype)?;
            attn_probs.matmul(&v_cont)?
        } else {
            let mut out_chunks = Vec::new();
            for start in (0..total_seq).step_by(chunk_size) {
                let len = (start + chunk_size).min(total_seq) - start;
                let q_chunk = q_scaled.narrow(2, start, len)?.contiguous()?;
                let attn_weights = q_chunk.matmul(&k_t)?;
                let attn_probs = candle_nn::ops::softmax(&attn_weights.to_dtype(DType::F32)?, candle_core::D::Minus1)?
                    .to_dtype(q_dtype)?;
                let out_chunk = attn_probs.matmul(&v_cont)?;
                out_chunks.push(out_chunk);
            }
            let refs: Vec<&Tensor> = out_chunks.iter().collect();
            Tensor::cat(&refs, 2)?
        };

        // Reshape [B, Heads, Seq, Head_Dim] -> [B, Seq, Hidden]
        let attn_out = attn_out.transpose(1, 2)?.contiguous()?.reshape((b, total_seq, self.hidden_size))?;
        let attn_proj = self.to_out.forward(&attn_out)?;

        // Split back text and video streams
        let attn_enc = attn_proj.narrow(1, 0, text_seq_len)?;
        let attn_vid = attn_proj.narrow(1, text_seq_len, total_seq - text_seq_len)?;

        // Residual with gating
        let hidden_states = (hidden_states + gate_msa.broadcast_mul(&attn_vid)?)?;
        let encoder_hidden_states = (encoder_hidden_states + enc_gate_msa.broadcast_mul(&attn_enc)?)?;

        // 3. AdaLN modulation 2 for FeedForward
        let mod2 = self.norm2_linear.forward(&temb_act)?;
        let chunks2 = mod2.chunk(6, 1)?;
        let shift2 = chunks2[0].unsqueeze(1)?;
        let scale2 = chunks2[1].unsqueeze(1)?;
        let gate_ff = chunks2[2].unsqueeze(1)?;
        let enc_shift2 = chunks2[3].unsqueeze(1)?;
        let enc_scale2 = chunks2[4].unsqueeze(1)?;
        let enc_gate_ff = chunks2[5].unsqueeze(1)?;

        let norm_hidden2 = self.norm2_norm.forward(&hidden_states)?;
        let mod_hidden2 = norm_hidden2.broadcast_mul(&(scale2 + 1.0)?)?.broadcast_add(&shift2)?;

        let norm_enc2 = self.norm2_norm.forward(&encoder_hidden_states)?;
        let mod_enc2 = norm_enc2.broadcast_mul(&(enc_scale2 + 1.0)?)?.broadcast_add(&enc_shift2)?;

        // 4. FeedForward MLP (concatenated [Text, Video])
        let ff_in = Tensor::cat(&[&mod_enc2, &mod_hidden2], 1)?;
        let ff_h = self.ff_proj.forward(&ff_in)?;
        let ff_act = gelu_approximate(&ff_h)?;
        let ff_out = self.ff_out.forward(&ff_act)?;

        // Split back text and video streams
        let ff_enc = ff_out.narrow(1, 0, text_seq_len)?;
        let ff_vid = ff_out.narrow(1, text_seq_len, total_seq - text_seq_len)?;

        // Residual with gating
        let hidden_states = (hidden_states + gate_ff.broadcast_mul(&ff_vid)?)?;
        let encoder_hidden_states = (encoder_hidden_states + enc_gate_ff.broadcast_mul(&ff_enc)?)?;

        Ok((hidden_states, encoder_hidden_states))
    }
}

/// Full CogVideoX 3D Spatio-Temporal Diffusion Transformer
#[derive(Debug, Clone)]
pub struct CogVideoXTransformer3D {
    pub patch_embed: CogVideoXPatchEmbed,
    pub time_embed: CogVideoXTimeEmbed,
    pub blocks: Vec<CogVideoXBlock3D>,
    pub norm_final: CogVideoXLayerNorm,
    pub norm_out_linear: Linear,
    pub norm_out_norm: CogVideoXLayerNorm,
    pub proj_out: Linear,
    pub cfg: CogVideoXConfig,
}

impl CogVideoXTransformer3D {
    pub fn new(cfg: CogVideoXConfig, vb: VarBuilder) -> Result<Self> {
        let patch_embed = CogVideoXPatchEmbed::new(&cfg, vb.pp("patch_embed"))?;
        let time_embed = CogVideoXTimeEmbed::new(cfg.hidden_size, cfg.time_embed_dim, vb.pp("time_embedding"))?;

        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(CogVideoXBlock3D::new(&cfg, vb.pp(format!("transformer_blocks.{i}")))?);
        }

        let norm_final = CogVideoXLayerNorm::new(cfg.hidden_size, 1e-5, vb.pp("norm_final"))?;
        let norm_out_linear = linear(cfg.time_embed_dim, cfg.hidden_size * 2, vb.pp("norm_out.linear"))?;
        let norm_out_norm = CogVideoXLayerNorm::new(cfg.hidden_size, 1e-5, vb.pp("norm_out.norm"))?;
        let proj_out = linear(cfg.hidden_size, cfg.patch_size * cfg.patch_size * cfg.out_channels, vb.pp("proj_out"))?;

        Ok(Self {
            patch_embed,
            time_embed,
            blocks,
            norm_final,
            norm_out_linear,
            norm_out_norm,
            proj_out,
            cfg,
        })
    }

    /// Full forward pass for video generation:
    /// - `latents`: [B, 16, T, H, W]
    /// - `timestep`: [B] or scalar tensor
    /// - `text_context`: [B, Seq_Text, 4096] (from T5-XXL)
    /// - `image_rotary_emb`: optional precomputed (cos, sin) tensors [T * h_p * w_p, Head_Dim]
    /// Returns predicted velocity `[B, 16, T, H, W]`
    pub fn forward(
        &self,
        latents: &Tensor,
        timestep: &Tensor,
        text_context: &Tensor,
        image_rotary_emb: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        let (b, c, t, h, w) = latents.dims5()?;
        let p = self.cfg.patch_size;
        let h_p = h / p;
        let w_p = w / p;

        // 1. Compute timestep embedding: [B, 512]
        let temb = self.time_embed.forward(timestep)?;

        // 2. Prepare 3D Spatio-Temporal Rotary Position Embeddings (only if use_rotary_positional_embeddings)
        let auto_rope;
        let rotary_refs = if self.cfg.use_rotary_positional_embeddings {
            if let Some((cos, sin)) = image_rotary_emb {
                Some((cos, sin))
            } else {
                let (cos, sin) = crate::diffusion::video_rope::VideoRope3D::precompute_cogvideox(
                    h * 8,
                    w * 8,
                    t,
                    self.cfg.attention_head_dim,
                    latents.device(),
                )?;
                auto_rope = (cos, sin);
                Some((&auto_rope.0, &auto_rope.1))
            }
        } else {
            None
        };

        // 3. Patchify video and project text tokens
        let video_tokens = self.patch_embed.forward_video(latents)?; // [B, T * h_p * w_p, Hidden]
        let text_tokens = self.patch_embed.forward_text(text_context)?; // [B, Seq_Text, Hidden]

        // 4. Inject 3D Spatio-Temporal Positional Embedding (additive for CogVideoX-2B)
        let (mut hidden, mut enc) = if self.cfg.use_positional_embeddings {
            let embeds = Tensor::cat(&[&text_tokens, &video_tokens], 1)?;
            let total_seq = embeds.dim(1)?;
            let text_seq_len = text_tokens.dim(1)?;
            let pos_emb = if let Some(ref p) = self.patch_embed.pos_embedding {
                if p.dim(1).unwrap_or(0) == total_seq {
                    p.clone()
                } else {
                    crate::diffusion::video_rope::get_cogvideox_3d_sincos_pos_embed(
                        self.cfg.hidden_size,
                        h_p,
                        w_p,
                        t,
                        text_seq_len,
                        latents.device(),
                    )?
                }
            } else {
                crate::diffusion::video_rope::get_cogvideox_3d_sincos_pos_embed(
                    self.cfg.hidden_size,
                    h_p,
                    w_p,
                    t,
                    text_seq_len,
                    latents.device(),
                )?
            };
            let pos_emb = pos_emb.to_dtype(embeds.dtype())?.to_device(embeds.device())?;
            let embeds = (&embeds + &pos_emb)?;
            let enc = embeds.narrow(1, 0, text_seq_len)?;
            let hidden = embeds.narrow(1, text_seq_len, total_seq - text_seq_len)?;
            (hidden, enc)
        } else {
            (video_tokens, text_tokens)
        };

        // 5. Process Spatio-Temporal Transformer Blocks
        for block in &self.blocks {
            let (next_h, next_e) = block.forward(&hidden, &enc, &temb, rotary_refs)?;
            hidden = next_h;
            enc = next_e;
        }

        // 5. Final AdaLN modulation & Output projection on video stream
        let hidden_norm = self.norm_final.forward(&hidden)?;
        let temb_act = candle_nn::ops::silu(&temb)?;
        let mod_out = self.norm_out_linear.forward(&temb_act)?;
        let mod_chunks = mod_out.chunk(2, 1)?;
        let shift_out = mod_chunks[0].unsqueeze(1)?;
        let scale_out = mod_chunks[1].unsqueeze(1)?;

        let x_out_norm = self.norm_out_norm.forward(&hidden_norm)?;
        let x_final = x_out_norm.broadcast_mul(&(scale_out + 1.0)?)?.broadcast_add(&shift_out)?;

        // 6. Project to unpatchified channels: [B, Seq_Video, p * p * C] = [B, Seq_Video, 64]
        let projected = self.proj_out.forward(&x_final)?;

        // 7. Unpatchify matching diffusers geometry:
        // [B * T, h_p, w_p, C, p, p] -> permute(0, 3, 1, 4, 2, 5) -> [B * T, C, H, W] -> [B, C, T, H, W]
        let h_actual = h_p * p;
        let w_actual = w_p * p;
        let unpatch = projected
            .reshape((b * t, h_p, w_p, c, p, p))?
            .permute((0, 3, 1, 4, 2, 5))?
            .contiguous()?
            .reshape((b * t, c, h_actual, w_actual))?
            .reshape((b, t, c, h_actual, w_actual))?
            .transpose(1, 2)?;

        Ok(unpatch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cogvideox_patch_and_time_shapes() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = CogVideoXConfig {
            num_layers: 1,
            hidden_size: 64,
            num_attention_heads: 2,
            attention_head_dim: 32,
            text_embed_dim: 128,
            time_embed_dim: 32,
            ..Default::default()
        };

        let vb = VarBuilder::zeros(DType::F32, &dev);
        let model = CogVideoXTransformer3D::new(cfg, vb)?;

        // 1 sample, 16 channels, 4 frames, 16x16 spatial
        let latents = Tensor::zeros((1, 16, 4, 16, 16), DType::F32, &dev)?;
        let timestep = Tensor::new(&[500.0f32], &dev)?;
        let text = Tensor::zeros((1, 8, 128), DType::F32, &dev)?;

        let pred_v = model.forward(&latents, &timestep, &text, None)?;
        assert_eq!(pred_v.dims5()?, (1, 16, 4, 16, 16));
        Ok(())
    }
}
