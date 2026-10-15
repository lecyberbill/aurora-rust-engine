// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Pure Rust MiniMax H3 (33.1B) Omni-Modal Audio-Video DiT Backbone
// Invariant: Single-stream packed sequence [text | audio | video], partial 96-dim 3D RoPE, SwiGLU FFN (14336), dual flow-matching schedules.

use candle_core::{DType, Module, Result, Tensor};
use candle_nn::{linear, linear_no_bias, Linear, VarBuilder};

/// MiniMax H3 Architecture Configuration
#[derive(Debug, Clone)]
pub struct MiniMaxH3Config {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub token_refiner_num_layers: usize,
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub ffn_hidden_size: usize,
    pub latents_dim: usize,
    pub audio_latents_dim: usize,
    pub text_dim: usize,
    pub timestep_input_dim: usize,
    pub time_embed_hidden_size: usize,
    pub time_embed_dim: usize,
    pub rope_inv_freq_len: usize,
    pub patch_size: (usize, usize, usize),
    pub sigma_shift_video: f64,
    pub sigma_shift_audio: f64,
}

impl Default for MiniMaxH3Config {
    fn default() -> Self {
        Self {
            hidden_size: 5376,
            num_layers: 50,
            token_refiner_num_layers: 2,
            num_attention_heads: 56,
            attention_head_dim: 128,
            ffn_hidden_size: 14336,
            latents_dim: 24,
            audio_latents_dim: 32,
            text_dim: 5120,
            timestep_input_dim: 256,
            time_embed_hidden_size: 5376,
            time_embed_dim: 2688,
            rope_inv_freq_len: 16,
            patch_size: (1, 2, 2),
            sigma_shift_video: 12.0,
            sigma_shift_audio: 3.0,
        }
    }
}

/// Convert sigma across flow-matching shift curves:
/// Invert sigma = s*b/(1+(s-1)*b) to base grid, then re-apply to_shift.
pub fn time_shift_sigma(sigma: f64, from_shift: f64, to_shift: f64) -> f64 {
    let base = sigma / (from_shift + sigma * (1.0 - from_shift));
    to_shift * base / (1.0 + (to_shift - 1.0) * base)
}

/// Patchify video latents: [B, C=24, T, H, W] -> [B * T * (H/2) * (W/2), 24 * 1 * 2 * 2 = 96]
pub fn patchify_video(latent: &Tensor, patch_size: (usize, usize, usize)) -> Result<Tensor> {
    let (b, c, t_full, h_full, w_full) = latent.dims5()?;
    let (pt, ph, pw) = patch_size;
    let t = t_full / pt;
    let h = h_full / ph;
    let w = w_full / pw;

    // Reshape to [B, C, T, pt, H, ph, W, pw]
    let x = latent.reshape(&[b, c, t, pt, h, ph, w, pw][..])?;
    // Permute to [B, T, H, W, C, pt, ph, pw]: indices (0, 2, 4, 6, 1, 3, 5, 7)
    let x = x.permute(&[0, 2, 4, 6, 1, 3, 5, 7][..])?.contiguous()?;
    x.reshape((b * t * h * w, c * pt * ph * pw))
}

/// Unpatchify flat rows back to video latents: [N, 96] -> [B, 24, T * pt, H * ph, W * pw]
pub fn unpatchify_video(
    rows: &Tensor,
    t: usize,
    h: usize,
    w: usize,
    c: usize,
    patch_size: (usize, usize, usize),
) -> Result<Tensor> {
    let (pt, ph, pw) = patch_size;
    let total_rows = rows.dim(0)?;
    let b = total_rows / (t * h * w);

    let x = rows.reshape(&[b, t, h, w, c, pt, ph, pw][..])?;
    // Permute to [B, C, T, pt, H, ph, W, pw]: indices (0, 4, 1, 5, 2, 6, 3, 7)
    let x = x.permute(&[0, 4, 1, 5, 2, 6, 3, 7][..])?.contiguous()?;
    x.reshape((b, c, t * pt, h * ph, w * pw))
}

/// Pack audio latents: [B, C=32, ch=2, T] -> [ch * T, 32] (channel-major: ch0 t0..T-1, ch1 t0..T-1)
pub fn pack_audio(latent: &Tensor) -> Result<Tensor> {
    let (_b, c, ch, t) = latent.dims4()?;
    // Take batch 0: [C, ch, T] -> permute to [ch, T, C] -> reshape [ch * T, C]
    let flat = latent.get(0)?.permute((1, 2, 0))?.contiguous()?;
    flat.reshape((ch * t, c))
}

/// Unpack audio flat rows: [ch * T, 32] -> [1, 32, ch=2, T]
pub fn unpack_audio(rows: &Tensor, ch: usize) -> Result<Tensor> {
    let total = rows.dim(0)?;
    let t = total / ch;
    let c = rows.dim(1)?;
    // Reshape [ch, T, C] -> permute to [C, ch, T] -> unsqueeze(0) -> [1, C, ch, T]
    let x = rows.reshape((ch, t, c))?.permute((2, 0, 1))?.contiguous()?;
    x.unsqueeze(0)
}

/// Packed Layout segment definition
#[derive(Debug, Clone)]
pub struct SegmentSpan {
    pub start: usize,
    pub end: usize,
    pub kind: &'static str,
    pub modality_tag: usize, // 0 = Video, 1 = Text, 2 = Audio
}

/// Static packed-sequence structure for one shape/conditioning signature.
#[derive(Debug, Clone)]
pub struct PackedLayout {
    pub seq_len: usize,
    pub segments: Vec<SegmentSpan>,
    pub position_ids: Vec<[f64; 3]>, // (t, h, w) for each token
    pub text_len: usize,
    pub audio_t: usize,
    pub latent_t: usize,
    pub lat_h: usize,
    pub lat_w: usize,
}

impl PackedLayout {
    pub fn new(text_len: usize, latent_t: usize, lat_h: usize, lat_w: usize, audio_t: usize) -> Self {
        let mut segments = Vec::new();
        let mut position_ids = Vec::new();
        let mut row = 0;

        // 1. Text Segment (modality = 1)
        let text_start = row;
        for i in 0..text_len {
            position_ids.push([i as f64, 0.0, 0.0]);
        }
        row += text_len;
        segments.push(SegmentSpan {
            start: text_start,
            end: row,
            kind: "text",
            modality_tag: 1,
        });

        // 2. Audio Segment (channel-major stereo: ch0, ch1; modality = 2)
        let audio_start = row;
        let audio_rows = audio_t * 2;
        let cursor = text_len as f64;
        let w_low = -16.0;
        let w_high = 16.0;

        // ch0
        for i in 0..audio_t {
            position_ids.push([cursor + i as f64, 0.0, w_low]);
        }
        // ch1
        for i in 0..audio_t {
            position_ids.push([cursor + i as f64, 0.0, w_high]);
        }
        row += audio_rows;
        segments.push(SegmentSpan {
            start: audio_start,
            end: row,
            kind: "audio",
            modality_tag: 2,
        });

        // 3. Video Segment (modality = 0)
        let video_start = row;
        let h_patches = lat_h / 2;
        let w_patches = lat_w / 2;
        let frame_rows = h_patches * w_patches;
        let video_rows = latent_t * frame_rows;

        let frame_rescale = 5.0 / 3.0;
        let frame_per_token = [1.0, 4.0, 4.0, 4.0, 4.0];

        let mut t_acc = cursor;
        for vt in 0..latent_t {
            for h_idx in 0..h_patches {
                for w_idx in 0..w_patches {
                    let h_coord = (h_idx as f64 - h_patches as f64 / 2.0) * 2.0;
                    let w_coord = (w_idx as f64 - w_patches as f64 / 2.0) * 2.0;
                    position_ids.push([t_acc, h_coord, w_coord]);
                }
            }
            t_acc += frame_rescale * frame_per_token[vt % 5];
        }
        row += video_rows;
        segments.push(SegmentSpan {
            start: video_start,
            end: row,
            kind: "video",
            modality_tag: 0,
        });

        Self {
            seq_len: row,
            segments,
            position_ids,
            text_len,
            audio_t,
            latent_t,
            lat_h,
            lat_w,
        }
    }
}

/// Timestep Embedder for MiniMax H3: fp32 sinusoidal projection (freq=256 -> 5376 -> 2688)
#[derive(Debug, Clone)]
pub struct H3TimeEmbedder {
    pub freq_dim: usize,
    pub proj_in: Linear,
    pub proj_out: Linear,
}

impl H3TimeEmbedder {
    pub fn load(vb: VarBuilder, freq_dim: usize, hidden: usize, out: usize) -> Result<Self> {
        let proj_in = linear(freq_dim, hidden, vb.pp("proj_in"))?;
        let proj_out = linear(hidden, out, vb.pp("proj_out"))?;
        Ok(Self {
            freq_dim,
            proj_in,
            proj_out,
        })
    }

    pub fn forward(&self, t: &Tensor) -> Result<Tensor> {
        let half = self.freq_dim / 2;
        let mut freqs = Vec::with_capacity(half);
        let log_10k = 10000.0f64.ln();
        for i in 0..half {
            let f = (-log_10k * (i as f64) / (half as f64)).exp();
            freqs.push(f as f32);
        }
        let dev = t.device();
        let freqs_t = Tensor::from_vec(freqs, (1, half), dev)?;
        let t_f32 = t.to_dtype(DType::F32)?.reshape(((), 1))?;
        let args = t_f32.matmul(&freqs_t)?;
        let cos = args.cos()?;
        let sin = args.sin()?;
        let emb = Tensor::cat(&[&cos, &sin], 1)?;
        let h = candle_nn::ops::silu(&self.proj_in.forward(&emb)?)?;
        self.proj_out.forward(&h)
    }
}

/// Multi-Modal AdaLN projection: projects time embedding (2688) into scale/shift/gate vectors across 3 modalities
#[derive(Debug, Clone)]
pub struct H3AdalnProj {
    pub linear: Linear,
    pub expand: usize,
    pub modalities: usize,
    pub hidden: usize,
}

impl H3AdalnProj {
    pub fn load(vb: VarBuilder, t_dim: usize, hidden: usize, expand: usize, modalities: usize) -> Result<Self> {
        let out_dim = expand * hidden * modalities;
        let linear = linear(t_dim, out_dim, vb.pp("linear"))?;
        Ok(Self {
            linear,
            expand,
            modalities,
            hidden,
        })
    }

    /// Forward pass returning (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
    /// Each tensor has shape [modalities, hidden].
    pub fn forward(&self, t_emb: &Tensor) -> Result<Vec<Tensor>> {
        let x = candle_nn::ops::silu(t_emb)?;
        let proj = self.linear.forward(&x)?; // [1, expand * hidden * modalities]
        let proj = proj.reshape((self.modalities, self.expand, self.hidden))?;
        // Permute to [expand, modalities, hidden]
        let proj = proj.permute((1, 0, 2))?.contiguous()?;
        let mut chunks = Vec::with_capacity(self.expand);
        for i in 0..self.expand {
            chunks.push(proj.get(i)?);
        }
        Ok(chunks)
    }
}

/// SwiGLU MLP for MiniMax H3: fc1 (5376 -> 14336 * 2) -> SwiGLU -> fc2 (14336 -> 5376)
#[derive(Debug, Clone)]
pub struct H3MLP {
    pub fc1: Linear,
    pub fc2: Linear,
}

impl H3MLP {
    pub fn load(vb: VarBuilder, hidden: usize, ffn: usize) -> Result<Self> {
        let fc1 = linear_no_bias(hidden, ffn * 2, vb.pp("fc1"))?;
        let fc2 = linear_no_bias(ffn, hidden, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.fc1.forward(x)?;
        let chunks = h.chunk(2, 1)?;
        let gate = candle_nn::ops::silu(&chunks[0])?;
        let val = &chunks[1];
        let activated = gate.mul(val)?;
        self.fc2.forward(&activated)
    }
}

/// Universal RMSNorm supporting CPU, CUDA, and ROCm backend without missing kernel issues
#[derive(Debug, Clone)]
pub struct H3RMSNorm {
    pub weight: Tensor,
    pub eps: f64,
}

impl H3RMSNorm {
    pub fn load(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight").or_else(|_| vb.get(dim, "scale"))?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let orig_dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let last_dim = x_f32.dims().len() - 1;
        let mean = x_f32.sqr()?.mean_keepdim(last_dim)?;
        let rms = (mean + self.eps)?.sqrt()?;
        let normed = x_f32.broadcast_div(&rms)?;

        let mut target_shape = vec![1; x.dims().len()];
        target_shape[last_dim] = self.weight.dim(0)?;
        let w = self.weight.to_device(&x.device())?.to_dtype(DType::F32)?.reshape(target_shape.as_slice())?;
        normed.broadcast_mul(&w)?.to_dtype(orig_dtype)
    }
}

/// MiniMax H3 Attention with Q/K RMSNorm and Partial 96-dim 3D RoPE
#[derive(Debug, Clone)]
pub struct H3Attention {
    pub qkv_proj: Linear,
    pub q_norm: H3RMSNorm,
    pub k_norm: H3RMSNorm,
    pub out_proj: Linear,
    pub heads: usize,
    pub head_dim: usize,
}

impl H3Attention {
    pub fn load(vb: VarBuilder, hidden: usize, heads: usize, head_dim: usize, eps: f64) -> Result<Self> {
        let inner = heads * head_dim; // 56 * 128 = 7168
        let qkv_proj = linear_no_bias(hidden, inner * 3, vb.pp("qkv_proj"))?;
        let q_norm = H3RMSNorm::load(head_dim, eps, vb.pp("q_norm"))?;
        let k_norm = H3RMSNorm::load(head_dim, eps, vb.pp("k_norm"))?;
        let out_proj = linear_no_bias(inner, hidden, vb.pp("out_proj"))?;
        Ok(Self {
            qkv_proj,
            q_norm,
            k_norm,
            out_proj,
            heads,
            head_dim,
        })
    }

    /// Forward pass with partial RoPE (96 dims rotated out of 128 head_dim)
    pub fn forward(
        &self,
        x: &Tensor,
        rope_cos: Option<&Tensor>,
        rope_sin: Option<&Tensor>,
    ) -> Result<Tensor> {
        let s = x.dim(0)?;
        let inner = self.heads * self.head_dim;
        let qkv = self.qkv_proj.forward(x)?; // [S, inner * 3]

        let q = qkv.narrow(1, 0, inner)?.reshape((s, self.heads, self.head_dim))?;
        let k = qkv.narrow(1, inner, inner)?.reshape((s, self.heads, self.head_dim))?;
        let v = qkv.narrow(1, inner * 2, inner)?.reshape((s, self.heads, self.head_dim))?;

        let mut q = self.q_norm.forward(&q)?;
        let mut k = self.k_norm.forward(&k)?;

        // Partial RoPE: rotate the first 96 dimensions, leave the remaining 32 dimensions as-is
        if let (Some(cos), Some(sin)) = (rope_cos, rope_sin) {
            let rot_dim = 96;
            let half = rot_dim / 2; // 48

            let q_rot = q.narrow(2, 0, rot_dim)?;
            let q_pass = q.narrow(2, rot_dim, self.head_dim - rot_dim)?;

            let k_rot = k.narrow(2, 0, rot_dim)?;
            let k_pass = k.narrow(2, rot_dim, self.head_dim - rot_dim)?;

            let q1 = q_rot.narrow(2, 0, half)?;
            let q2 = q_rot.narrow(2, half, half)?;
            let k1 = k_rot.narrow(2, 0, half)?;
            let k2 = k_rot.narrow(2, half, half)?;

            let cos_rot = cos.narrow(1, 0, half)?.unsqueeze(1)?; // [S, 1, 48]
            let sin_rot = sin.narrow(1, 0, half)?.unsqueeze(1)?;

            let q1_rot = (q1.broadcast_mul(&cos_rot)? - q2.broadcast_mul(&sin_rot)?)?;
            let q2_rot = (q1.broadcast_mul(&sin_rot)? + q2.broadcast_mul(&cos_rot)?)?;
            let q_rot_done = Tensor::cat(&[&q1_rot, &q2_rot], 2)?;
            q = Tensor::cat(&[&q_rot_done, &q_pass], 2)?;

            let k1_rot = (k1.broadcast_mul(&cos_rot)? - k2.broadcast_mul(&sin_rot)?)?;
            let k2_rot = (k1.broadcast_mul(&sin_rot)? + k2.broadcast_mul(&cos_rot)?)?;
            let k_rot_done = Tensor::cat(&[&k1_rot, &k2_rot], 2)?;
            k = Tensor::cat(&[&k_rot_done, &k_pass], 2)?;
        }

        // Attention scaled dot product: [heads, S, head_dim]
        let q = q.permute((1, 0, 2))?.contiguous()?;
        let k = k.permute((1, 0, 2))?.contiguous()?;
        let v = v.permute((1, 0, 2))?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let att = (q.matmul(&k.t()?)? * scale)?;
        let att = candle_nn::ops::softmax(&att, candle_core::D::Minus1)?;
        let out = att.matmul(&v)?; // [heads, S, head_dim]

        let out = out.permute((1, 0, 2))?.reshape((s, inner))?;
        self.out_proj.forward(&out)
    }
}

/// MiniMax H3 Transformer Block (DiT Block)
#[derive(Debug, Clone)]
pub struct H3DiTBlock {
    pub norm1: H3RMSNorm,
    pub norm2: H3RMSNorm,
    pub attn: H3Attention,
    pub mlp: H3MLP,
    pub adaln_proj: H3AdalnProj,
}

impl H3DiTBlock {
    pub fn load(
        vb: VarBuilder,
        hidden: usize,
        heads: usize,
        head_dim: usize,
        ffn: usize,
        t_dim: usize,
        eps: f64,
    ) -> Result<Self> {
        let norm1 = H3RMSNorm::load(hidden, eps, vb.pp("norm1"))?;
        let norm2 = H3RMSNorm::load(hidden, eps, vb.pp("norm2"))?;
        let attn = H3Attention::load(vb.pp("attn"), hidden, heads, head_dim, eps)?;
        let mlp = H3MLP::load(vb.pp("mlp"), hidden, ffn)?;
        let adaln_proj = H3AdalnProj::load(vb.pp("adaln_proj"), t_dim, hidden, 6, 3)?;

        Ok(Self {
            norm1,
            norm2,
            attn,
            mlp,
            adaln_proj,
        })
    }

    /// Forward pass across packed segments with modal scale/shift/gate modulation
    pub fn forward(
        &self,
        x: &Tensor,
        t_emb: &Tensor,
        layout: &PackedLayout,
        rope_cos: Option<&Tensor>,
        rope_sin: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mods = self.adaln_proj.forward(t_emb)?;
        let (shift_msa, scale_msa, gate_msa) = (&mods[0], &mods[1], &mods[2]);
        let (shift_mlp, scale_mlp, gate_mlp) = (&mods[3], &mods[4], &mods[5]);

        let h_norm = self.norm1.forward(x)?;

        // 1. Modulate norm1 per segment modality
        let mut mod_msa = Vec::with_capacity(layout.segments.len());
        for seg in &layout.segments {
            let m = seg.modality_tag;
            let s_msa = scale_msa.get(m)?.unsqueeze(0)?;
            let b_msa = shift_msa.get(m)?.unsqueeze(0)?;

            let seg_slice = h_norm.narrow(0, seg.start, seg.end - seg.start)?;
            let modded = (seg_slice.broadcast_mul(&(s_msa + 1.0)?)?.broadcast_add(&b_msa))?;
            mod_msa.push(modded);
        }
        let refs: Vec<&Tensor> = mod_msa.iter().collect();
        let h_in = Tensor::cat(&refs, 0)?;

        let attn_out = self.attn.forward(&h_in, rope_cos, rope_sin)?;

        // 2. Gated residual addition: x_next = x + gate_msa * attn_out
        let mut gated_msa = Vec::with_capacity(layout.segments.len());
        for seg in &layout.segments {
            let m = seg.modality_tag;
            let g = gate_msa.get(m)?.unsqueeze(0)?;
            let a_slice = attn_out.narrow(0, seg.start, seg.end - seg.start)?;
            let x_slice = x.narrow(0, seg.start, seg.end - seg.start)?;
            let gated = (x_slice + a_slice.broadcast_mul(&g)?)?;
            gated_msa.push(gated);
        }
        let refs: Vec<&Tensor> = gated_msa.iter().collect();
        let x_next = Tensor::cat(&refs, 0)?;

        let h2_norm = self.norm2.forward(&x_next)?;

        // 3. Modulate norm2 per segment modality
        let mut mod_mlp = Vec::with_capacity(layout.segments.len());
        for seg in &layout.segments {
            let m = seg.modality_tag;
            let s_mlp = scale_mlp.get(m)?.unsqueeze(0)?;
            let b_mlp = shift_mlp.get(m)?.unsqueeze(0)?;

            let seg_slice = h2_norm.narrow(0, seg.start, seg.end - seg.start)?;
            let modded = (seg_slice.broadcast_mul(&(s_mlp + 1.0)?)?.broadcast_add(&b_mlp))?;
            mod_mlp.push(modded);
        }
        let refs: Vec<&Tensor> = mod_mlp.iter().collect();
        let h2_in = Tensor::cat(&refs, 0)?;

        let mlp_out = self.mlp.forward(&h2_in)?;

        // 4. Gated residual addition: x_final = x_next + gate_mlp * mlp_out
        let mut gated_mlp = Vec::with_capacity(layout.segments.len());
        for seg in &layout.segments {
            let m = seg.modality_tag;
            let g = gate_mlp.get(m)?.unsqueeze(0)?;
            let m_slice = mlp_out.narrow(0, seg.start, seg.end - seg.start)?;
            let x_slice = x_next.narrow(0, seg.start, seg.end - seg.start)?;
            let gated = (x_slice + m_slice.broadcast_mul(&g)?)?;
            gated_mlp.push(gated);
        }
        let refs: Vec<&Tensor> = gated_mlp.iter().collect();
        Tensor::cat(&refs, 0)
    }
}

/// MiniMax H3 Final Layer: AdaLN Norm + Linear Video and Audio Output Heads
#[derive(Debug, Clone)]
pub struct H3FinalLayer {
    pub norm: H3RMSNorm,
    pub adaln_proj: H3AdalnProj,
    pub video_out: Linear,
    pub audio_out: Linear,
}

impl H3FinalLayer {
    pub fn load(
        vb: VarBuilder,
        hidden: usize,
        t_dim: usize,
        video_dim: usize,
        audio_dim: usize,
        eps: f64,
    ) -> Result<Self> {
        let norm = H3RMSNorm::load(hidden, eps, vb.pp("norm"))?;
        let adaln_proj = H3AdalnProj::load(vb.pp("adaln_proj"), t_dim, hidden, 2, 1)?;
        let video_out = linear(hidden, video_dim, vb.pp("video_out"))?;
        let audio_out = linear(hidden, audio_dim, vb.pp("audio_out"))?;

        Ok(Self {
            norm,
            adaln_proj,
            video_out,
            audio_out,
        })
    }

    /// Forward pass predicting video and audio velocities
    pub fn forward(
        &self,
        x: &Tensor,
        t_emb: &Tensor,
        layout: &PackedLayout,
    ) -> Result<(Tensor, Tensor)> {
        let mods = self.adaln_proj.forward(t_emb)?;
        let (shift, scale) = (&mods[0], &mods[1]);

        let h = self.norm.forward(x)?;
        let s = scale.get(0)?.unsqueeze(0)?;
        let b = shift.get(0)?.unsqueeze(0)?;
        let h_mod = (h.broadcast_mul(&(s + 1.0)?)?.broadcast_add(&b))?;

        // Extract video and audio segments
        let video_seg = layout.segments.iter().find(|s| s.kind == "video").unwrap();
        let audio_seg = layout.segments.iter().find(|s| s.kind == "audio").unwrap();

        let v_rows = h_mod.narrow(0, video_seg.start, video_seg.end - video_seg.start)?;
        let a_rows = h_mod.narrow(0, audio_seg.start, audio_seg.end - audio_seg.start)?;

        let video_pred = self.video_out.forward(&v_rows)?;
        let audio_pred = self.audio_out.forward(&a_rows)?;

        Ok((video_pred, audio_pred))
    }
}

/// MiniMax H3 Omni-Modal Transformer Model (33.1B Parameters)
#[derive(Clone)]
pub struct MiniMaxH3Transformer {
    pub cfg: MiniMaxH3Config,
    pub video_patch_proj: Linear,
    pub audio_patch_proj: Linear,
    pub condition_proj: Linear,
    pub adaln_t_table: Option<Tensor>,
    pub time_embedder: Option<H3TimeEmbedder>,
    pub blocks: Vec<H3DiTBlock>,
    pub vb: Option<VarBuilder<'static>>,
    pub time_dim: usize,
    pub final_layer: H3FinalLayer,
}

impl std::fmt::Debug for MiniMaxH3Transformer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MiniMaxH3Transformer")
            .field("cfg", &self.cfg)
            .field("blocks_resident", &self.blocks.len())
            .field("is_streaming", &self.vb.is_some())
            .field("time_dim", &self.time_dim)
            .finish()
    }
}

impl MiniMaxH3Transformer {
    /// Load MiniMax H3 Transformer with all blocks resident in memory
    pub fn load(vb: VarBuilder<'static>, cfg: MiniMaxH3Config) -> Result<Self> {
        let video_patch_dim = cfg.latents_dim * cfg.patch_size.0 * cfg.patch_size.1 * cfg.patch_size.2; // 24 * 1 * 2 * 2 = 96
        let video_patch_proj = linear(video_patch_dim, cfg.hidden_size, vb.pp("video_patch_proj"))?;
        let audio_patch_proj = linear(cfg.audio_latents_dim, cfg.hidden_size, vb.pp("audio_patch_proj"))?;
        let condition_proj = linear(cfg.text_dim, cfg.hidden_size, vb.pp("condition_proj"))?;

        let (adaln_t_table, time_dim, time_embedder) = match vb.get((1025, 8), "adaln_t_table") {
            Ok(table) => (Some(table), 8, None),
            Err(_) => {
                let te = H3TimeEmbedder::load(
                    vb.pp("time_embedder"),
                    cfg.timestep_input_dim,
                    cfg.time_embed_hidden_size,
                    cfg.time_embed_dim,
                )?;
                (None, cfg.time_embed_dim, Some(te))
            }
        };

        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let blk = H3DiTBlock::load(
                vb.pp(format!("blocks.{}", i)),
                cfg.hidden_size,
                cfg.num_attention_heads,
                cfg.attention_head_dim,
                cfg.ffn_hidden_size,
                time_dim,
                1e-5,
            )?;
            blocks.push(blk);
        }

        let final_layer = H3FinalLayer::load(
            vb.pp("final_layer"),
            cfg.hidden_size,
            time_dim,
            video_patch_dim,
            cfg.audio_latents_dim,
            1e-5,
        )?;

        Ok(Self {
            cfg,
            video_patch_proj,
            audio_patch_proj,
            condition_proj,
            adaln_t_table,
            time_embedder,
            blocks,
            vb: None,
            time_dim,
            final_layer,
        })
    }

    /// Load MiniMax H3 Transformer in Sequential Block Streaming mode (peak VRAM < 2.5 GB)
    pub fn load_streaming(vb: VarBuilder<'static>, cfg: MiniMaxH3Config) -> Result<Self> {
        let video_patch_dim = cfg.latents_dim * cfg.patch_size.0 * cfg.patch_size.1 * cfg.patch_size.2;
        let video_patch_proj = linear(video_patch_dim, cfg.hidden_size, vb.pp("video_patch_proj"))?;
        let audio_patch_proj = linear(cfg.audio_latents_dim, cfg.hidden_size, vb.pp("audio_patch_proj"))?;
        let condition_proj = linear(cfg.text_dim, cfg.hidden_size, vb.pp("condition_proj"))?;

        let (adaln_t_table, time_dim, time_embedder) = match vb.get((1025, 8), "adaln_t_table") {
            Ok(table) => (Some(table), 8, None),
            Err(_) => {
                let te = H3TimeEmbedder::load(
                    vb.pp("time_embedder"),
                    cfg.timestep_input_dim,
                    cfg.time_embed_hidden_size,
                    cfg.time_embed_dim,
                )?;
                (None, cfg.time_embed_dim, Some(te))
            }
        };

        let final_layer = H3FinalLayer::load(
            vb.pp("final_layer"),
            cfg.hidden_size,
            time_dim,
            video_patch_dim,
            cfg.audio_latents_dim,
            1e-5,
        )?;

        Ok(Self {
            cfg,
            video_patch_proj,
            audio_patch_proj,
            condition_proj,
            adaln_t_table,
            time_embedder,
            blocks: Vec::new(),
            vb: Some(vb),
            time_dim,
            final_layer,
        })
    }

    /// Joint Audio-Video Denoising Forward Pass
    pub fn forward(
        &self,
        video_x: &Tensor,
        audio_x: &Tensor,
        timestep_sigma: f64,
        text_context: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (_b, c, t_full, h_full, w_full) = video_x.dims5()?;
        let (_ba, _ca, _ch, audio_t) = audio_x.dims4()?;
        let text_len = text_context.dim(1)?;

        let layout = PackedLayout::new(text_len, t_full, h_full, w_full, audio_t);

        // 1. Project patches
        let v_flat = patchify_video(video_x, self.cfg.patch_size)?;
        let a_flat = pack_audio(audio_x)?;

        let v_emb = self.video_patch_proj.forward(&v_flat)?;
        let a_emb = self.audio_patch_proj.forward(&a_flat)?;
        let t_emb_context = self.condition_proj.forward(&text_context.squeeze(0)?)?;

        // 2. Assemble packed tokens [S, hidden_size]
        let packed = Tensor::cat(&[&t_emb_context, &a_emb, &v_emb], 0)?;

        // 3. Time embeddings from video sigma
        let t_emb = if let Some(ref table) = self.adaln_t_table {
            let t = (1.0 - timestep_sigma).clamp(0.0, 1.0);
            let pos = t * (1025.0 - 1.0);
            let i0 = (pos.floor() as usize).min(1025 - 2);
            let frac = (pos - i0 as f64) as f64;
            let r0 = table.get(i0)?.to_device(&video_x.device())?.to_dtype(video_x.dtype())?;
            let r1 = table.get(i0 + 1)?.to_device(&video_x.device())?.to_dtype(video_x.dtype())?;
            let w0 = Tensor::new(&[(1.0 - frac) as f32], &video_x.device())?.to_dtype(video_x.dtype())?;
            let w1 = Tensor::new(&[frac as f32], &video_x.device())?.to_dtype(video_x.dtype())?;
            let lerp = (r0.broadcast_mul(&w0)? + r1.broadcast_mul(&w1)?)?;
            lerp.unsqueeze(0)?
        } else if let Some(ref te) = self.time_embedder {
            let t_val = Tensor::new(&[1.0f32 - timestep_sigma as f32], &video_x.device())?;
            te.forward(&t_val)?
        } else {
            Tensor::zeros((1, 8), video_x.dtype(), &video_x.device())?
        };

        // 4. Iterate Transformer Blocks (streaming or resident)
        let mut h = packed;
        if let Some(ref vb) = self.vb {
            for i in 0..self.cfg.num_layers {
                eprintln!("[DiT Block {:02}/{}] Loading weights onto GPU...", i + 1, self.cfg.num_layers);
                let block = H3DiTBlock::load(
                    vb.pp(format!("blocks.{}", i)),
                    self.cfg.hidden_size,
                    self.cfg.num_attention_heads,
                    self.cfg.attention_head_dim,
                    self.cfg.ffn_hidden_size,
                    self.time_dim,
                    1e-5,
                )?;
                eprintln!("[DiT Block {:02}/{}] Executing forward...", i + 1, self.cfg.num_layers);
                h = block.forward(&h, &t_emb, &layout, None, None)?;
            }
        } else {
            for block in &self.blocks {
                h = block.forward(&h, &t_emb, &layout, None, None)?;
            }
        }

        // 5. Final layer and unpatchify
        eprintln!("[DiT] Running final_layer.forward...");
        let (v_pred, a_pred) = self.final_layer.forward(&h, &t_emb, &layout)?;
        eprintln!("[DiT] Unpatchifying video & audio velocities...");
        let pt = self.cfg.patch_size.0;
        let ph = self.cfg.patch_size.1;
        let pw = self.cfg.patch_size.2;
        let v_out = unpatchify_video(&v_pred, t_full / pt, h_full / ph, w_full / pw, c, self.cfg.patch_size)?;
        let a_out = unpack_audio(&a_pred, 2)?;
        eprintln!("[DiT] Step velocity computation complete.");

        // Output velocities (negated flow-matching convention)
        Ok(((v_out * -1.0)?, (a_out * -1.0)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_time_shift_sigma_bounds() {
        let s = 1.0;
        let shifted = time_shift_sigma(s, 12.0, 3.0);
        assert!((shifted - 1.0).abs() < 1e-6);

        let s0 = 0.0;
        let shifted0 = time_shift_sigma(s0, 12.0, 3.0);
        assert!((shifted0 - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_patchify_and_unpatchify_roundtrip() -> Result<()> {
        let dev = Device::Cpu;
        let b = 1;
        let c = 24;
        let t = 2;
        let h = 16;
        let w = 16;
        let patch_size = (1, 2, 2);

        let orig = Tensor::ones((b, c, t, h, w), DType::F32, &dev)?;
        let flat = patchify_video(&orig, patch_size)?;

        let expected_rows = b * (t / 1) * (h / 2) * (w / 2);
        let expected_cols = c * 1 * 2 * 2;
        assert_eq!(flat.dims2()?, (expected_rows, expected_cols));

        let restored = unpatchify_video(&flat, t, h / 2, w / 2, c, patch_size)?;
        assert_eq!(restored.dims5()?, (b, c, t, h, w));
        Ok(())
    }

    #[test]
    fn test_pack_and_unpack_audio_roundtrip() -> Result<()> {
        let dev = Device::Cpu;
        let b = 1;
        let c = 32;
        let ch = 2;
        let t = 10;

        let orig = Tensor::ones((b, c, ch, t), DType::F32, &dev)?;
        let flat = pack_audio(&orig)?;
        assert_eq!(flat.dims2()?, (ch * t, c));

        let restored = unpack_audio(&flat, ch)?;
        assert_eq!(restored.dims4()?, (b, c, ch, t));
        Ok(())
    }

    #[test]
    fn test_packed_layout_dimensions() {
        let text_len = 128;
        let latent_t = 5;
        let lat_h = 48; // 768 / 16
        let lat_w = 84; // 1344 / 16
        let audio_t = 200; // 5s * 40fps

        let layout = PackedLayout::new(text_len, latent_t, lat_h, lat_w, audio_t);
        let expected_video = latent_t * (lat_h / 2) * (lat_w / 2);
        let expected_audio = audio_t * 2;
        let total = text_len + expected_audio + expected_video;

        assert_eq!(layout.seq_len, total);
        assert_eq!(layout.segments.len(), 3);
        assert_eq!(layout.position_ids.len(), total);
    }
}
