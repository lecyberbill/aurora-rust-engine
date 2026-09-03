// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Pure Rust Mistral-3-Small Multi-Layer Text Encoder with NVFP4 / FP8 Dequantization

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{embedding, linear, Embedding, Linear, Module, VarBuilder};
use tokenizers::Tokenizer;
use std::path::Path;
use std::collections::HashMap;
use std::sync::Arc;
use crate::weights::WeightsSource;

/// Precomputed 16-entry Lookup Table for NVIDIA FP4 (E2M1) format
const E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
];

/// Dequantize NVFP4 packed uint8 tensor with FP8 block scales and FP32 per-tensor scale into F16/F32
pub fn dequantize_nvfp4(
    qx: &Tensor,
    block_scales: &Tensor,
    per_tensor_scale: f32,
    target_dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let qx_dims = qx.dims();
    let num_rows = qx_dims[0];
    let packed_cols = qx_dims[1];
    let num_cols = packed_cols * 2;

    // Convert packed uint8 to raw host bytes for fast SIMD / parallel dequantization
    let qx_cpu = qx.to_device(&Device::Cpu)?;
    let qx_bytes: Vec<u8> = qx_cpu.flatten_all()?.to_vec1()?;

    let bs_cpu = block_scales.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let bs_data: Vec<f32> = bs_cpu.flatten_all()?.to_vec1()?;
    let bs_dims = block_scales.dims();
    let bs_padded_rows = bs_dims[0];
    let bs_padded_cols = bs_dims[1];

    let num_blocks_per_row = num_cols / 16;
    let n_row_blocks = (num_rows + 127) / 128;
    let n_col_blocks = (num_blocks_per_row + 3) / 4;

    // Unswizzle block_scales from cuBLAS tiled layout
    let mut unswizzled_scales = vec![0.0f32; num_rows * num_blocks_per_row];
    for r in 0..num_rows {
        let r_block = r / 128;
        let r_sub = r % 128;
        let r_32 = r_sub / 32;
        let r_in_32 = r_sub % 32;

        for c_blk in 0..num_blocks_per_row {
            let c_block = c_blk / 4;
            let c_sub = c_blk % 4;

            // cuBLAS blocked layout index calculation
            let block_idx = r_block * n_col_blocks + c_block;
            let flat_idx = block_idx * (32 * 16) + r_in_32 * 16 + r_32 * 4 + c_sub;
            if flat_idx < bs_data.len() {
                unswizzled_scales[r * num_blocks_per_row + c_blk] = bs_data[flat_idx] * per_tensor_scale;
            }
        }
    }

    // Unpack 4-bit nibbles and multiply by total scales
    let mut out_f32 = vec![0.0f32; num_rows * num_cols];
    for r in 0..num_rows {
        let row_offset = r * num_cols;
        let packed_row_offset = r * packed_cols;
        for c in 0..packed_cols {
            let byte = qx_bytes[packed_row_offset + c];
            let hi = (byte >> 4) as usize;
            let lo = (byte & 0x0F) as usize;

            let col_hi = c * 2;
            let col_lo = c * 2 + 1;

            let blk_hi = col_hi / 16;
            let blk_lo = col_lo / 16;

            let scale_hi = unswizzled_scales[r * num_blocks_per_row + blk_hi];
            let scale_lo = unswizzled_scales[r * num_blocks_per_row + blk_lo];

            out_f32[row_offset + col_hi] = E2M1_LUT[hi] * scale_hi;
            out_f32[row_offset + col_lo] = E2M1_LUT[lo] * scale_lo;
        }
    }

    let tensor_cpu = Tensor::from_vec(out_f32, (num_rows, num_cols), &Device::Cpu)?;
    tensor_cpu.to_device(device)?.to_dtype(target_dtype)
}

/// RMS Normalization for Mistral-3
#[derive(Debug, Clone)]
pub struct MistralRMSNorm {
    weight: Tensor,
    eps: f64,
}

impl MistralRMSNorm {
    pub fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let orig_dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let sq = x_f32.sqr()?;
        let mean = sq.mean_keepdim(sq.dims().len() - 1)?;
        let rms = (mean + self.eps)?.sqrt()?;
        let norm = x_f32.broadcast_div(&rms)?;
        let w_f32 = self.weight.to_dtype(DType::F32)?;
        norm.broadcast_mul(&w_f32)?.to_dtype(orig_dtype)
    }
}

/// SwiGLU MLP for Mistral-3
#[derive(Debug, Clone)]
pub struct MistralMLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl MistralMLP {
    pub fn new(gate_proj: Linear, up_proj: Linear, down_proj: Linear) -> Self {
        Self { gate_proj, up_proj, down_proj }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

/// Grouped Query Attention Layer for Mistral-3
#[derive(Debug, Clone)]
pub struct MistralAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl MistralAttention {
    pub fn new(
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        o_proj: Linear,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        let scale = 1.0 / (head_dim as f64).sqrt();
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, seq, _) = x.dims3()?;
        let orig_dtype = x.dtype();

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let mut q = q.reshape((b, seq, self.num_heads, self.head_dim))?;
        let mut k = k.reshape((b, seq, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((b, seq, self.num_kv_heads, self.head_dim))?;

        // Apply 1D Rotary Position Embedding (RoPE) for Mistral-3 (theta = 100000000.0 / 1e8)
        let half_dim = self.head_dim / 2;
        let theta = 100000000.0f64;
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / (theta.powf((i * 2) as f64 / self.head_dim as f64) as f32))
            .collect();
        let inv_freq_t = Tensor::from_vec(inv_freq, (half_dim,), x.device())?;
        let pos_seq: Vec<f32> = (0..seq).map(|i| i as f32).collect();
        let pos_t = Tensor::from_vec(pos_seq, (seq, 1), x.device())?;
        let freqs = pos_t.matmul(&inv_freq_t.unsqueeze(0)?)?; // [seq, half_dim]
        let cos_half = freqs.cos()?;
        let sin_half = freqs.sin()?;
        let cos = Tensor::cat(&[&cos_half, &cos_half], 1)?.unsqueeze(0)?.unsqueeze(2)?; // [1, seq, 1, head_dim]
        let sin = Tensor::cat(&[&sin_half, &sin_half], 1)?.unsqueeze(0)?.unsqueeze(2)?; // [1, seq, 1, head_dim]

        let apply_mistral_rope = |t: &Tensor| -> Result<Tensor> {
            let t_f32 = t.to_dtype(DType::F32)?;
            let t1 = t_f32.narrow(3, 0, half_dim)?;
            let t2 = t_f32.narrow(3, half_dim, half_dim)?;
            let neg_t2 = (t2 * -1.0)?;
            let rotated = Tensor::cat(&[&neg_t2, &t1], 3)?;
            let out = (t_f32.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?)?;
            out.to_dtype(orig_dtype)
        };

        let q = apply_mistral_rope(&q)?;
        let k = apply_mistral_rope(&k)?;

        // Repeat KV heads if num_kv_heads < num_heads (Grouped Query Attention x4)
        let n_rep = self.num_heads / self.num_kv_heads;
        let k = if n_rep > 1 {
            k.unsqueeze(3)?.repeat((1, 1, 1, n_rep, 1))?.reshape((b, seq, self.num_heads, self.head_dim))?
        } else {
            k
        };
        let v = if n_rep > 1 {
            v.unsqueeze(3)?.repeat((1, 1, 1, n_rep, 1))?.reshape((b, seq, self.num_heads, self.head_dim))?
        } else {
            v
        };

        // SDPA in F32 with Causal Attention Mask
        let q_f32 = (q.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)? * self.scale)?;
        let k_f32 = k.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;
        let v_f32 = v.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;

        let mut scores = q_f32.matmul(&k_f32.transpose(2, 3)?.contiguous()?)?;

        // Causal attention mask for autoregressive encoder
        let mut mask_vec = vec![0f32; seq * seq];
        for i in 0..seq {
            for j in (i + 1)..seq {
                mask_vec[i * seq + j] = -1e9f32;
            }
        }
        let mask = Tensor::from_vec(mask_vec, (1, 1, seq, seq), x.device())?;
        scores = scores.broadcast_add(&mask)?;

        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let attn_out = probs.matmul(&v_f32)?.to_dtype(orig_dtype)?;

        let attn_out = attn_out.transpose(1, 2)?.contiguous()?.reshape((b, seq, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&attn_out)
    }
}

/// Mistral Decoder Block
#[derive(Debug, Clone)]
pub struct MistralDecoderLayer {
    input_layernorm: MistralRMSNorm,
    self_attn: MistralAttention,
    post_attention_layernorm: MistralRMSNorm,
    mlp: MistralMLP,
}

impl MistralDecoderLayer {
    pub fn new(
        input_layernorm: MistralRMSNorm,
        self_attn: MistralAttention,
        post_attention_layernorm: MistralRMSNorm,
        mlp: MistralMLP,
    ) -> Self {
        Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward(&normed)?;
        let h = (x + attn_out)?;
        let post_normed = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&post_normed)?;
        &h + mlp_out
    }
}

/// Pure Rust Mistral-3-Small Multi-Layer Text Encoder with Low-RAM Sequential Streaming
pub struct Mistral3TextEncoder {
    embed_tokens: Embedding,
    archive: std::sync::Arc<dyn crate::weights::WeightsSource>,
    num_layers: usize,
    tokenizer: Option<Tokenizer>,
    device: Device,
    dtype: DType,
}

impl Mistral3TextEncoder {
    pub fn load_tokenizer<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let tok = Tokenizer::from_file(path.as_ref())
            .map_err(|e| candle_core::Error::Msg(format!("Tokenizer error: {}", e)))?;
        self.tokenizer = Some(tok);
        Ok(())
    }

    /// Load a single decoder layer on-the-fly from archive and dequantize it
    fn load_layer(&self, layer_idx: usize) -> Result<MistralDecoderLayer> {
        let p = format!("model.layers.{}.", layer_idx);
        let device = &self.device;
        let dtype = self.dtype;
        let archive = &self.archive;

        let in_ln_w = archive.get_tensor(&format!("{}input_layernorm.weight", p), device, dtype)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let input_layernorm = MistralRMSNorm { weight: in_ln_w, eps: 1e-5 };

        let post_ln_w = archive.get_tensor(&format!("{}post_attention_layernorm.weight", p), device, dtype)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let post_attention_layernorm = MistralRMSNorm { weight: post_ln_w, eps: 1e-5 };

        let load_linear = |proj_name: &str| -> Result<Linear> {
            let weight_key = format!("{}{}.weight", p, proj_name);
            let scale_key = format!("{}{}.weight_scale", p, proj_name);
            let scale2_key = format!("{}{}.weight_scale_2", p, proj_name);

            // NVFP4 (packed U8 + block scales + per-tensor scale) — Flux.2 official mistral-alpha
            if let (Ok(qx), Ok(bs), Ok(s2)) = (
                archive.get_tensor(&weight_key, &Device::Cpu, DType::U8),
                archive.get_tensor(&scale_key, &Device::Cpu, DType::F32),
                archive.get_tensor(&scale2_key, &Device::Cpu, DType::F32),
            ) {
                let s2_val = s2.to_vec0::<f32>()?;
                let dequant = dequantize_nvfp4(&qx, &bs, s2_val, dtype, device)?;
                return Ok(Linear::new(dequant, None));
            }

            // FP8-E4M3 / F16 / BF16: get_tensor already dequantises FP8->F16 AND applies the
            // per-tensor `weight_scale`. Do NOT multiply again (that would double-scale).
            let w = archive.get_tensor(&weight_key, device, dtype)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            Ok(Linear::new(w, None))
        };

        let q_proj = load_linear("self_attn.q_proj")?;
        let k_proj = load_linear("self_attn.k_proj")?;
        let v_proj = load_linear("self_attn.v_proj")?;
        let o_proj = load_linear("self_attn.o_proj")?;
        let self_attn = MistralAttention::new(q_proj, k_proj, v_proj, o_proj, 32, 8, 128);

        let gate_proj = load_linear("mlp.gate_proj")?;
        let up_proj = load_linear("mlp.up_proj")?;
        let down_proj = load_linear("mlp.down_proj")?;
        let mlp = MistralMLP::new(gate_proj, up_proj, down_proj);

        Ok(MistralDecoderLayer::new(input_layernorm, self_attn, post_attention_layernorm, mlp))
    }

    /// Encode prompt and extract specified layer features (Layers 10, 20, 30) with on-demand layer streaming
    /// Encode prompt and extract specified layer features (Layers 10, 20, 30) with on-demand layer streaming.
    ///
    /// `layer_dim` is the feature width kept per extracted layer: 4096 for Flux.2-Klein-9B
    /// (3*4096 = 12288) and 5120 for Flux.2-Dev (3*5120 = 15360). Default behaviour (0) selects
    /// the model's hidden width (5120) and produces 15360.
    pub fn encode(&self, prompt: &str, max_tokens: usize) -> Result<Tensor> {
        self.encode_dim(prompt, max_tokens, 0)
    }

    /// Like [`encode`](Self::encode) but with an explicit per-layer output width.
    pub fn encode_dim(&self, prompt: &str, max_tokens: usize, layer_dim: usize) -> Result<Tensor> {
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            candle_core::Error::Msg("Mistral Tokenizer not loaded".to_string())
        })?;

        let formatted_prompt = format!("[INST]{}[/INST]", prompt.trim());

        let encoding = tokenizer.encode(formatted_prompt, true)
            .or_else(|_| tokenizer.encode(prompt, true))
            .map_err(|e| candle_core::Error::Msg(format!("Tokenize failed: {}", e)))?;

        let mut token_ids = encoding.get_ids().to_vec();
        if token_ids.len() > max_tokens {
            token_ids.truncate(max_tokens);
        } else if token_ids.len() < max_tokens {
            token_ids.resize(max_tokens, 2); // Pad with EOS token ID (2)
        }

        let tokens_tensor = Tensor::from_slice(&token_ids, (1, max_tokens), &self.device)?;
        let mut h = self.embed_tokens.forward(&tokens_tensor)?.to_dtype(self.dtype)?;

        let mut l10: Option<Tensor> = None;
        let mut l20: Option<Tensor> = None;
        let mut l30: Option<Tensor> = None;

        for idx in 0..self.num_layers {
            let layer = self.load_layer(idx)?;
            h = layer.forward(&h)?;
            if idx == 9 {
                l10 = Some(h.clone());
            } else if idx == 19 {
                l20 = Some(h.clone());
            } else if idx == 29 {
                l30 = Some(h.clone());
                break;
            }
        }

        let full_dim = h.dim(2)?;
        let keep = if layer_dim == 0 { full_dim } else { layer_dim.min(full_dim) };
        let l10 = l10.unwrap_or(h.clone()).narrow(2, 0, keep)?;
        let l20 = l20.unwrap_or(h.clone()).narrow(2, 0, keep)?;
        let l30 = l30.unwrap_or(h.clone()).narrow(2, 0, keep)?;

        // Concat sliced layers 10, 20, 30 along channel dimension: [1, seq_len, keep*3].
        // NOTE: Flux.2-Dev feeds Mistral-3 hidden states at their NATIVE amplitude (~rms 0.4) — the
        // text projection (context_embedder) was trained on these raw values. Do NOT normalise.
        let out = Tensor::cat(&[&l10, &l20, &l30], 2)?;

        // De-noise test (FLUX_TRACE_DEVOISE): normalise to kill the massive outliers (max 171+) that
        // blow up the Dev transformer. Scale to a target RMS. Only active under the env flag.
        if std::env::var("FLUX_TEXT_NORM").is_ok() {
            let x = out.to_dtype(DType::F32)?;
            let mean_sq = (x.sqr()?.sum_keepdim(2)? / (x.dim(2)? as f64))?;
            let rms = (mean_sq + 1e-6f64)?.sqrt()?;
            let normed = x.broadcast_div(&rms)?.to_dtype(self.dtype)?;
            Ok(normed)
        } else {
            Ok(out)
        }
    }

    /// Open Mistral-3-Small text encoder for on-demand low-memory streaming (RAM < 4GB)
    pub fn from_safetensors<P: AsRef<Path>>(
        path: P,
        tokenizer_path: Option<&Path>,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let archive = crate::weights::SafeTensorsArchive::open(path.as_ref().to_path_buf())
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        Self::from_weights(Arc::new(archive), tokenizer_path, device, dtype)
    }

    /// Build from any [`WeightsSource`] (safetensors single/multi-shard, or GGUF). This is the
    /// format-agnostic entry point so the brick can be assembled from a chosen model origin.
    pub fn from_weights(
        archive: Arc<dyn WeightsSource>,
        tokenizer_path: Option<&Path>,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        // 1. Embedding
        let embed_weight = archive.get_tensor("model.embed_tokens.weight", &device, dtype)
            .or_else(|_| archive.get_tensor("embed_tokens.weight", &device, dtype))
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let embed_tokens = Embedding::new(embed_weight, 5120);

        let tokenizer = if let Some(p) = tokenizer_path {
            Tokenizer::from_file(p).ok()
        } else {
            None
        };

        Ok(Self {
            embed_tokens,
            archive,
            num_layers: 30,
            tokenizer,
            device,
            dtype,
        })
    }
}
