// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Pure Rust Vision-Language Model (VLM) Architecture supporting Qwen-VL, PaliGemma, Gemma 3 Vision, and SmolVLM

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{conv2d, embedding, layer_norm, linear, linear_no_bias, rms_norm, Conv2d, Conv2dConfig, Embedding, LayerNorm, LayerNormConfig, Linear, RmsNorm, VarBuilder};

/// Vision Encoder Activation Type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisionActivation {
    QuickGelu,
    Gelu,
    SwiGLU,
}

/// Vision Transformer (ViT / SigLIP / Qwen-VL) Configuration
#[derive(Debug, Clone)]
pub struct VisionTransformerConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub num_channels: usize,
    pub embed_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub spatial_merge_size: usize,
    pub act_type: VisionActivation,
    pub layer_norm_eps: f64,
}

impl Default for VisionTransformerConfig {
    fn default() -> Self {
        Self {
            image_size: 448,
            patch_size: 14,
            num_channels: 3,
            embed_dim: 1152,
            num_layers: 27,
            num_heads: 16,
            intermediate_size: 4304,
            spatial_merge_size: 2,
            act_type: VisionActivation::QuickGelu,
            layer_norm_eps: 1e-6,
        }
    }
}

/// Vision Attention Block (supports both Fused QKV and Split Q/K/V)
#[derive(Debug, Clone)]
pub enum VisionQkv {
    Fused(Linear),
    Split {
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
    },
}

#[derive(Debug, Clone)]
pub struct VisionAttention {
    qkv: VisionQkv,
    proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl VisionAttention {
    pub fn new(embed_dim: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let qkv = if vb.contains_tensor("qkv.weight") || vb.contains_tensor("in_proj.weight") || vb.contains_tensor("attn_qkv.weight") {
            let lin = linear(embed_dim, embed_dim * 3, vb.pp("qkv"))
                .or_else(|_| linear(embed_dim, embed_dim * 3, vb.pp("in_proj")))
                .or_else(|_| linear(embed_dim, embed_dim * 3, vb.pp("attn_qkv")))?;
            VisionQkv::Fused(lin)
        } else {
            let q_proj = linear(embed_dim, embed_dim, vb.pp("q_proj"))
                .or_else(|_| linear(embed_dim, embed_dim, vb.pp("self_attn.q_proj")))?;
            let k_proj = linear(embed_dim, embed_dim, vb.pp("k_proj"))
                .or_else(|_| linear(embed_dim, embed_dim, vb.pp("self_attn.k_proj")))?;
            let v_proj = linear(embed_dim, embed_dim, vb.pp("v_proj"))
                .or_else(|_| linear(embed_dim, embed_dim, vb.pp("self_attn.v_proj")))?;
            VisionQkv::Split { q_proj, k_proj, v_proj }
        };

        let proj = linear(embed_dim, embed_dim, vb.pp("proj"))
            .or_else(|_| linear(embed_dim, embed_dim, vb.pp("out_proj")))
            .or_else(|_| linear(embed_dim, embed_dim, vb.pp("self_attn.out_proj")))
            .or_else(|_| linear(embed_dim, embed_dim, vb.pp("attn_out")))?;
        let head_dim = embed_dim / num_heads;
        Ok(Self {
            qkv,
            proj,
            num_heads,
            head_dim,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, d) = x.dims3()?;
        let (q, k, v) = match &self.qkv {
            VisionQkv::Fused(lin) => {
                let qkv = lin.forward(x)?;
                let qkv = qkv.reshape((b, s, 3, self.num_heads, self.head_dim))?;
                let q = qkv.narrow(2, 0, 1)?.squeeze(2)?.transpose(1, 2)?.contiguous()?;
                let k = qkv.narrow(2, 1, 1)?.squeeze(2)?.transpose(1, 2)?.contiguous()?;
                let v = qkv.narrow(2, 2, 1)?.squeeze(2)?.transpose(1, 2)?.contiguous()?;
                (q, k, v)
            }
            VisionQkv::Split { q_proj, k_proj, v_proj } => {
                let q = q_proj.forward(x)?.reshape((b, s, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
                let k = k_proj.forward(x)?.reshape((b, s, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
                let v = v_proj.forward(x)?.reshape((b, s, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
                (q, k, v)
            }
        };

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let attn_probs = candle_nn::ops::softmax(&attn_weights, candle_core::D::Minus1)?;
        let context = attn_probs.matmul(&v)?;

        let context = context.transpose(1, 2)?.contiguous()?.reshape((b, s, d))?;
        self.proj.forward(&context)
    }
}

/// Vision MLP Block
#[derive(Debug, Clone)]
pub struct VisionMlp {
    fc1: Linear,
    fc2: Linear,
    act_type: VisionActivation,
}

impl VisionMlp {
    pub fn new(embed_dim: usize, intermediate_size: usize, act_type: VisionActivation, vb: VarBuilder) -> Result<Self> {
        let fc1 = linear(embed_dim, intermediate_size, vb.pp("fc1"))
            .or_else(|_| linear(embed_dim, intermediate_size, vb.pp("mlp.0")))
            .or_else(|_| linear(embed_dim, intermediate_size, vb.pp("ffn_up")))?;
        let fc2 = linear(intermediate_size, embed_dim, vb.pp("fc2"))
            .or_else(|_| linear(intermediate_size, embed_dim, vb.pp("mlp.2")))
            .or_else(|_| linear(intermediate_size, embed_dim, vb.pp("ffn_down")))?;
        Ok(Self { fc1, fc2, act_type })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.fc1.forward(x)?;
        let h = match self.act_type {
            VisionActivation::QuickGelu => (h.clone() * candle_nn::ops::sigmoid(&(h * 1.702)?)?)?,
            VisionActivation::Gelu => h.gelu_erf()?,
            VisionActivation::SwiGLU => candle_nn::ops::silu(&h)?,
        };
        self.fc2.forward(&h)
    }
}

/// Vision Transformer Block
#[derive(Debug, Clone)]
pub struct VisionBlock {
    norm1: LayerNorm,
    attn: VisionAttention,
    norm2: LayerNorm,
    mlp: VisionMlp,
}

impl VisionBlock {
    pub fn new(cfg: &VisionTransformerConfig, vb: VarBuilder) -> Result<Self> {
        let ln_cfg = LayerNormConfig {
            eps: cfg.layer_norm_eps,
            ..Default::default()
        };
        let norm1 = layer_norm(cfg.embed_dim, ln_cfg, vb.pp("norm1"))
            .or_else(|_| layer_norm(cfg.embed_dim, ln_cfg, vb.pp("ln1")))
            .or_else(|_| layer_norm(cfg.embed_dim, ln_cfg, vb.pp("layer_norm1")))?;
        let attn = VisionAttention::new(cfg.embed_dim, cfg.num_heads, vb.pp("attn"))
            .or_else(|_| VisionAttention::new(cfg.embed_dim, cfg.num_heads, vb.pp("self_attn")))
            .or_else(|_| VisionAttention::new(cfg.embed_dim, cfg.num_heads, vb.clone()))?;
        let norm2 = layer_norm(cfg.embed_dim, ln_cfg, vb.pp("norm2"))
            .or_else(|_| layer_norm(cfg.embed_dim, ln_cfg, vb.pp("ln2")))
            .or_else(|_| layer_norm(cfg.embed_dim, ln_cfg, vb.pp("layer_norm2")))?;
        let mlp = VisionMlp::new(cfg.embed_dim, cfg.intermediate_size, cfg.act_type, vb.pp("mlp"))
            .or_else(|_| VisionMlp::new(cfg.embed_dim, cfg.intermediate_size, cfg.act_type, vb.clone()))?;

        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = (x + self.attn.forward(&self.norm1.forward(x)?)?)?;
        let out = (&h + self.mlp.forward(&self.norm2.forward(&h)?)?)?;
        Ok(out)
    }
}

/// Vision Transformer (ViT) Backbone
#[derive(Debug, Clone)]
pub struct VisionTransformer {
    patch_embed: Conv2d,
    pos_embed: Option<Tensor>,
    blocks: Vec<VisionBlock>,
    post_norm: Option<LayerNorm>,
    spatial_merge_size: usize,
}

impl VisionTransformer {
    pub fn new(cfg: &VisionTransformerConfig, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            stride: cfg.patch_size,
            padding: 0,
            ..Default::default()
        };
        let patch_embed = conv2d(
            cfg.num_channels,
            cfg.embed_dim,
            cfg.patch_size,
            conv_cfg,
            vb.pp("patch_embed").pp("proj"),
        )
        .or_else(|_| {
            conv2d(
                cfg.num_channels,
                cfg.embed_dim,
                cfg.patch_size,
                conv_cfg,
                vb.pp("v.patch_embd"),
            )
        })
        .or_else(|_| {
            conv2d(
                cfg.num_channels,
                cfg.embed_dim,
                cfg.patch_size,
                conv_cfg,
                vb.pp("patch_embed"),
            )
        })
        .or_else(|_| {
            conv2d(
                cfg.num_channels,
                cfg.embed_dim,
                cfg.patch_size,
                conv_cfg,
                vb.pp("embeddings.patch_embedding"),
            )
        })
        .or_else(|_| {
            conv2d(
                cfg.num_channels,
                cfg.embed_dim,
                cfg.patch_size,
                conv_cfg,
                vb.pp("embeddings.patch_embedding.proj"),
            )
        })?;

        let num_patches_w = cfg.image_size / cfg.patch_size;
        let num_patches_h = cfg.image_size / cfg.patch_size;
        let num_patches = num_patches_w * num_patches_h;

        let pos_embed = vb.get((1, num_patches, cfg.embed_dim), "pos_embed")
            .or_else(|_| vb.get((1, num_patches + 1, cfg.embed_dim), "position_embedding"))
            .or_else(|_| vb.get((1, num_patches, cfg.embed_dim), "v.position_embd.weight"))
            .or_else(|_| vb.get((1, num_patches, cfg.embed_dim), "embeddings.position_embedding.weight"))
            .ok();

        let mut blocks = Vec::with_capacity(cfg.num_layers);
        let blocks_vb = if vb.pp("blocks").contains_tensor("0.norm1.weight") || vb.pp("blocks.0").contains_tensor("norm1.weight") {
            vb.pp("blocks")
        } else if vb.pp("v.blk").contains_tensor("0.ln1.weight") || vb.contains_tensor("v.blk.0.ln1.weight") {
            vb.pp("v.blk")
        } else if vb.pp("encoder.layers").contains_tensor("0.layer_norm1.weight") || vb.contains_tensor("encoder.layers.0.layer_norm1.weight") {
            vb.pp("encoder.layers")
        } else {
            vb.pp("layers")
        };
        for i in 0..cfg.num_layers {
            blocks.push(VisionBlock::new(cfg, blocks_vb.pp(i))?);
        }

        let ln_cfg = LayerNormConfig {
            eps: cfg.layer_norm_eps,
            ..Default::default()
        };
        let post_norm = layer_norm(cfg.embed_dim, ln_cfg, vb.pp("post_norm"))
            .or_else(|_| layer_norm(cfg.embed_dim, ln_cfg, vb.pp("ln_post")))
            .or_else(|_| layer_norm(cfg.embed_dim, ln_cfg, vb.pp("post_layernorm")))
            .or_else(|_| layer_norm(cfg.embed_dim, ln_cfg, vb.pp("v.post_norm")))
            .or_else(|_| layer_norm(cfg.embed_dim, ln_cfg, vb.pp("v.ln_post")))
            .or_else(|_| layer_norm(cfg.embed_dim, ln_cfg, vb.pp("encoder.post_layernorm")))
            .ok();

        Ok(Self {
            patch_embed,
            pos_embed,
            blocks,
            post_norm,
            spatial_merge_size: cfg.spatial_merge_size,
        })
    }

    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let (b, _c, _h, _w) = pixel_values.dims4()?;
        let patches = self.patch_embed.forward(pixel_values)?;
        let (_, embed_dim, grid_h, grid_w) = patches.dims4()?;

        // Flatten spatial patches to sequence [B, S, D]
        let mut x = patches.permute((0, 2, 3, 1))?.contiguous()?.reshape((b, grid_h * grid_w, embed_dim))?;

        if let Some(ref pos) = self.pos_embed {
            let (_, pos_len, _) = pos.dims3()?;
            if pos_len == grid_h * grid_w {
                x = (x + pos)?;
            } else if pos_len == grid_h * grid_w + 1 {
                // Ignore class token in pos embed
                let pos_no_cls = pos.narrow(1, 1, grid_h * grid_w)?;
                x = (x + pos_no_cls)?;
            }
        }

        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        let x = if let Some(ref norm) = self.post_norm {
            norm.forward(&x)?
        } else {
            x
        };

        // Spatial Merging (2x2 pixel pooling if enabled)
        if self.spatial_merge_size > 1 {
            let m = self.spatial_merge_size;
            let merged_h = grid_h / m;
            let merged_w = grid_w / m;
            let x = x.reshape((b, merged_h, m, merged_w, m, embed_dim))?;
            let x = x.permute((0, 1, 3, 2, 4, 5))?.contiguous()?;
            let x = x.reshape((b, merged_h * merged_w, m * m * embed_dim))?;
            Ok(x)
        } else {
            Ok(x)
        }
    }
}

/// Multimodal Projector (supports Single Linear and 2-layer MLP Projector)
#[derive(Debug, Clone)]
pub enum ProjectorKind {
    TwoLayer {
        linear1: Linear,
        linear2: Linear,
    },
    Single(Linear),
}

#[derive(Debug, Clone)]
pub struct MultiModalProjector {
    kind: ProjectorKind,
}

impl MultiModalProjector {
    pub fn new(in_features: usize, out_features: usize, vb: VarBuilder) -> Result<Self> {
        let load_lin = |in_d, out_d, v: VarBuilder| {
            linear(in_d, out_d, v.clone()).or_else(|_| linear_no_bias(in_d, out_d, v))
        };

        load_lin(in_features, out_features, vb.pp("proj"))
            .or_else(|_| load_lin(in_features, out_features, vb.pp("linear")))
            .or_else(|_| load_lin(in_features, out_features, vb.clone()))
            .map(|lin| Self { kind: ProjectorKind::Single(lin) })
            .or_else(|_| {
                let linear1 = load_lin(in_features, out_features, vb.pp("linear_1"))
                    .or_else(|_| load_lin(in_features, out_features, vb.pp("0")))
                    .or_else(|_| load_lin(in_features, out_features, vb.pp("mm.0")))?;
                let linear2 = load_lin(out_features, out_features, vb.pp("linear_2"))
                    .or_else(|_| load_lin(out_features, out_features, vb.pp("2")))
                    .or_else(|_| load_lin(out_features, out_features, vb.pp("mm.2")))?;
                Ok(Self { kind: ProjectorKind::TwoLayer { linear1, linear2 } })
            })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match &self.kind {
            ProjectorKind::Single(lin) => lin.forward(x),
            ProjectorKind::TwoLayer { linear1, linear2 } => {
                let h = linear1.forward(x)?;
                let h = h.gelu_erf()?;
                linear2.forward(&h)
            }
        }
    }
}

/// Language Model Decoder Configuration
#[derive(Debug, Clone)]
pub struct VlmDecoderConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
}

impl Default for VlmDecoderConfig {
    fn default() -> Self {
        Self {
            vocab_size: 151936, // Qwen/Gemma vocab size
            hidden_size: 2048,
            intermediate_size: 5632,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            num_key_value_heads: 2,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 8192,
        }
    }
}

/// Causal Self-Attention Layer for VLM Decoder
#[derive(Debug, Clone)]
pub struct VlmAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_theta: f32,
}

impl VlmAttention {
    pub fn new(cfg: &VlmDecoderConfig, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let load_lin = |in_d, out_d, v: VarBuilder| {
            linear(in_d, out_d, v.clone()).or_else(|_| linear_no_bias(in_d, out_d, v))
        };
        let q_proj = load_lin(cfg.hidden_size, cfg.num_attention_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = load_lin(cfg.hidden_size, cfg.num_key_value_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = load_lin(cfg.hidden_size, cfg.num_key_value_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = load_lin(cfg.num_attention_heads * head_dim, cfg.hidden_size, vb.pp("o_proj"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim,
            rope_theta: cfg.rope_theta,
        })
    }

    fn apply_rope(&self, x: &Tensor, pos: usize) -> Result<Tensor> {
        let (_b, _num_heads, seq_len, head_dim) = x.dims4()?;
        let half_dim = head_dim / 2;
        let dev = x.device();

        let positions: Vec<f32> = (pos..pos + seq_len).map(|p| p as f32).collect();
        let pos_t = Tensor::new(&positions[..], dev)?.unsqueeze(1)?;

        let freqs: Vec<f32> = (0..half_dim)
            .map(|i| 1.0f32 / self.rope_theta.powf((2 * i) as f32 / head_dim as f32))
            .collect();
        let freq_t = Tensor::new(&freqs[..], dev)?.unsqueeze(0)?;

        let angles = pos_t.matmul(&freq_t)?;
        let cos = angles.cos()?.unsqueeze(0)?.unsqueeze(0)?;
        let sin = angles.sin()?.unsqueeze(0)?.unsqueeze(0)?;

        let cos = Tensor::cat(&[&cos, &cos], 3)?;
        let sin = Tensor::cat(&[&sin, &sin], 3)?;

        let x1 = x.narrow(3, 0, half_dim)?;
        let x2 = x.narrow(3, half_dim, half_dim)?;
        let neg_x2 = (x2 * -1.0)?;
        let x_rot = Tensor::cat(&[&neg_x2, &x1], 3)?;

        let x_out = ((x.broadcast_mul(&cos)?) + (x_rot.broadcast_mul(&sin)?))?;
        Ok(x_out)
    }

    pub fn forward(
        &self,
        x: &Tensor,
        pos: usize,
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let (b, seq_len, _) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let mut q = q.reshape((b, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let mut k = k.reshape((b, seq_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let v = v.reshape((b, seq_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;

        q = self.apply_rope(&q, pos)?;
        k = self.apply_rope(&k, pos)?;

        // KV cache update
        let (k_all, v_all) = match kv_cache {
            Some((ref prev_k, ref prev_v)) => {
                let k_cat = Tensor::cat(&[prev_k, &k], 2)?;
                let v_cat = Tensor::cat(&[prev_v, &v], 2)?;
                *kv_cache = Some((k_cat.clone(), v_cat.clone()));
                (k_cat, v_cat)
            }
            None => {
                *kv_cache = Some((k.clone(), v.clone()));
                (k, v)
            }
        };

        // GQA repeat KV if needed
        let k_all = self.repeat_kv(k_all)?;
        let v_all = self.repeat_kv(v_all)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k_all.transpose(2, 3)?.contiguous()?)? * scale)?;

        // Apply causal mask if seq_len > 1
        let total_kv_len = k_all.dim(2)?;
        let attn_weights = if seq_len > 1 {
            let mask = self.make_causal_mask(seq_len, total_kv_len, x.device(), x.dtype())?;
            attn_weights.broadcast_add(&mask)?
        } else {
            attn_weights
        };

        let attn_probs = candle_nn::ops::softmax(&attn_weights, candle_core::D::Minus1)?;
        let context = attn_probs.matmul(&v_all)?;
        let context = context.transpose(1, 2)?.contiguous()?.reshape((b, seq_len, self.num_heads * self.head_dim))?;

        self.o_proj.forward(&context)
    }

    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        let n_rep = self.num_heads / self.num_kv_heads;
        if n_rep == 1 {
            return Ok(x);
        }
        let (b, n_kv_heads, seq_len, head_dim) = x.dims4()?;
        let x = x.unsqueeze(2)?.expand((b, n_kv_heads, n_rep, seq_len, head_dim))?.contiguous()?;
        x.reshape((b, n_kv_heads * n_rep, seq_len, head_dim))
    }

    fn make_causal_mask(&self, seq_len: usize, total_len: usize, dev: &Device, dtype: DType) -> Result<Tensor> {
        let mut mask_vec = vec![0.0f32; seq_len * total_len];
        let offset = total_len - seq_len;
        for i in 0..seq_len {
            for j in 0..total_len {
                if j > offset + i {
                    mask_vec[i * total_len + j] = f32::NEG_INFINITY;
                }
            }
        }
        Tensor::new(&mask_vec[..], dev)?.reshape((1, 1, seq_len, total_len))?.to_dtype(dtype)
    }
}

/// SwiGLU MLP Block for VLM Decoder
#[derive(Debug, Clone)]
pub struct VlmDecoderMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl VlmDecoderMlp {
    pub fn new(cfg: &VlmDecoderConfig, vb: VarBuilder) -> Result<Self> {
        let load_lin = |in_d, out_d, v: VarBuilder| {
            linear(in_d, out_d, v.clone()).or_else(|_| linear_no_bias(in_d, out_d, v))
        };
        let gate_proj = load_lin(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = load_lin(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?;
        let down_proj = load_lin(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        let h = (gate * up)?;
        self.down_proj.forward(&h)
    }
}

/// Transformer Decoder Layer for VLM
#[derive(Debug, Clone)]
pub struct VlmDecoderLayer {
    input_layernorm: RmsNorm,
    self_attn: VlmAttention,
    post_attention_layernorm: RmsNorm,
    mlp: VlmDecoderMlp,
}

impl VlmDecoderLayer {
    pub fn new(cfg: &VlmDecoderConfig, vb: VarBuilder) -> Result<Self> {
        let input_layernorm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let self_attn = VlmAttention::new(cfg, vb.pp("self_attn"))?;
        let post_attention_layernorm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?;
        let mlp = VlmDecoderMlp::new(cfg, vb.pp("mlp"))?;

        Ok(Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        pos: usize,
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let residual = x;
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward(&normed, pos, kv_cache)?;
        let h = (residual + attn_out)?;

        let residual = &h;
        let normed = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed)?;
        residual + mlp_out
    }
}

/// Causal Language Model for VLM
#[derive(Debug, Clone)]
pub struct VlmLanguageModel {
    embed_tokens: Embedding,
    layers: Vec<VlmDecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
}

impl VlmLanguageModel {
    pub fn new(cfg: &VlmDecoderConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_root(cfg, vb.clone(), vb)
    }

    pub fn new_with_root(cfg: &VlmDecoderConfig, vb: VarBuilder, root_vb: VarBuilder) -> Result<Self> {
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))
            .or_else(|_| embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens")))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let layers_vb = if vb.pp("model.layers").contains_tensor("0.input_layernorm.weight") || vb.pp("model.layers.0").contains_tensor("input_layernorm.weight") {
            vb.pp("model.layers")
        } else {
            vb.pp("layers")
        };
        for i in 0..cfg.num_hidden_layers {
            layers.push(VlmDecoderLayer::new(cfg, layers_vb.pp(i))?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))
            .or_else(|_| rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm")))?;
        let lm_head = linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))
            .or_else(|_| linear_no_bias(cfg.hidden_size, cfg.vocab_size, root_vb.pp("lm_head")))
            .or_else(|_| linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("output")))
            .or_else(|_| linear_no_bias(cfg.hidden_size, cfg.vocab_size, root_vb.pp("output")))?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }

    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        let input_ids = if input_ids.dtype() != DType::U32 {
            input_ids.to_dtype(DType::U32)?
        } else {
            input_ids.clone()
        };
        self.embed_tokens.forward(&input_ids)
    }

    pub fn forward_with_embeds(
        &self,
        embeds: &Tensor,
        pos: usize,
        kv_caches: &mut [Option<(Tensor, Tensor)>],
    ) -> Result<Tensor> {
        let mut h = embeds.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, pos, &mut kv_caches[i])?;
        }
        let h = self.norm.forward(&h)?;
        self.lm_head.forward(&h)
    }
}

/// Complete Vision-Language Model (VLM)
pub struct VlmModel {
    pub vision_encoder: VisionTransformer,
    pub projector: MultiModalProjector,
    pub language_model: VlmLanguageModel,
    pub image_token_id: u32,
    pub kv_caches: Vec<Option<(Tensor, Tensor)>>,
}

impl VlmModel {
    pub fn new(
        vision_cfg: &VisionTransformerConfig,
        decoder_cfg: &VlmDecoderConfig,
        image_token_id: u32,
        vb: VarBuilder,
    ) -> Result<Self> {
        let vision_encoder = VisionTransformer::new(vision_cfg, vb.pp("vision_tower"))
            .or_else(|_| VisionTransformer::new(vision_cfg, vb.pp("visual")))
            .or_else(|_| VisionTransformer::new(vision_cfg, vb.pp("vision_model")))
            .or_else(|_| VisionTransformer::new(vision_cfg, vb.pp("model.vision_model")))?;

        let proj_in = if vision_cfg.spatial_merge_size > 1 {
            vision_cfg.embed_dim * vision_cfg.spatial_merge_size * vision_cfg.spatial_merge_size
        } else {
            vision_cfg.embed_dim
        };
        let projector = MultiModalProjector::new(proj_in, decoder_cfg.hidden_size, vb.pp("multi_modal_projector"))
            .or_else(|_| MultiModalProjector::new(proj_in, decoder_cfg.hidden_size, vb.pp("projector")))
            .or_else(|_| MultiModalProjector::new(proj_in, decoder_cfg.hidden_size, vb.pp("connector.modality_projection")))
            .or_else(|_| MultiModalProjector::new(proj_in, decoder_cfg.hidden_size, vb.pp("model.connector.modality_projection")))?;

        let language_model = VlmLanguageModel::new_with_root(decoder_cfg, vb.pp("language_model"), vb.clone())
            .or_else(|_| VlmLanguageModel::new_with_root(decoder_cfg, vb.pp("text_model"), vb.clone()))
            .or_else(|_| VlmLanguageModel::new_with_root(decoder_cfg, vb.pp("model.text_model"), vb.clone()))
            .or_else(|_| VlmLanguageModel::new_with_root(decoder_cfg, vb.clone(), vb.clone()))?;

        let kv_caches = vec![None; decoder_cfg.num_hidden_layers];

        Ok(Self {
            vision_encoder,
            projector,
            language_model,
            image_token_id,
            kv_caches,
        })
    }

    pub fn reset_kv_cache(&mut self) {
        for cache in &mut self.kv_caches {
            *cache = None;
        }
    }

    /// Encode image to projected LLM token space
    pub fn encode_image(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let feats = self.vision_encoder.forward(pixel_values)?;
        self.projector.forward(&feats)
    }

    /// Forward pass with multimodal tokens and KV-Cache
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        pos: usize,
    ) -> Result<Tensor> {
        if let Some(pixels) = pixel_values {
            let visual_embeds = self.encode_image(pixels)?;
            let text_embeds = self.language_model.embed_tokens(input_ids)?;

            let (b, seq_len, hidden_dim) = text_embeds.dims3()?;
            let (_, num_visual_tokens, _) = visual_embeds.dims3()?;

            let tokens = input_ids.squeeze(0)?.to_vec1::<u32>()?;
            let mut merged_data = text_embeds.flatten_all()?.to_vec1::<f32>()?;
            let visual_data = visual_embeds.flatten_all()?.to_vec1::<f32>()?;

            let mut v_idx = 0;
            for (t_idx, &tok) in tokens.iter().enumerate() {
                if tok == self.image_token_id && v_idx < num_visual_tokens {
                    let dst_start = t_idx * hidden_dim;
                    let src_start = v_idx * hidden_dim;
                    merged_data[dst_start..dst_start + hidden_dim]
                        .copy_from_slice(&visual_data[src_start..src_start + hidden_dim]);
                    v_idx += 1;
                }
            }

            let input_embeds = Tensor::from_vec(merged_data, (b, seq_len, hidden_dim), input_ids.device())?;
            self.language_model.forward_with_embeds(&input_embeds, pos, &mut self.kv_caches)
        } else {
            let input_embeds = self.language_model.embed_tokens(input_ids)?;
            self.language_model.forward_with_embeds(&input_embeds, pos, &mut self.kv_caches)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_vision_transformer_forward() -> Result<()> {
        let dev = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let cfg = VisionTransformerConfig {
            image_size: 28,
            patch_size: 14,
            num_channels: 3,
            embed_dim: 32,
            num_layers: 2,
            num_heads: 4,
            intermediate_size: 64,
            spatial_merge_size: 2,
            act_type: VisionActivation::Gelu,
            layer_norm_eps: 1e-5,
        };
        let vit = VisionTransformer::new(&cfg, vb)?;
        let pixels = Tensor::zeros((1, 3, 28, 28), DType::F32, &dev)?;
        let out = vit.forward(&pixels)?;
        assert_eq!(out.dims3()?, (1, 1, 128));
        Ok(())
    }

    #[test]
    fn test_vlm_model_e2e() -> Result<()> {
        let dev = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let vision_cfg = VisionTransformerConfig {
            image_size: 28,
            patch_size: 14,
            num_channels: 3,
            embed_dim: 32,
            num_layers: 1,
            num_heads: 2,
            intermediate_size: 64,
            spatial_merge_size: 2,
            act_type: VisionActivation::Gelu,
            layer_norm_eps: 1e-5,
        };
        let decoder_cfg = VlmDecoderConfig {
            vocab_size: 100,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_position_embeddings: 512,
        };
        let image_token_id = 99;
        let mut vlm = VlmModel::new(&vision_cfg, &decoder_cfg, image_token_id, vb)?;

        // Multimodal input: 3 tokens, one of which is image_token_id (99)
        let input_ids = Tensor::new(&[1u32, 99, 2], &dev)?.unsqueeze(0)?;
        let pixels = Tensor::zeros((1, 3, 28, 28), DType::F32, &dev)?;

        let logits = vlm.forward(&input_ids, Some(&pixels), 0)?;
        assert_eq!(logits.dims3()?, (1, 3, 100));

        // Autoregressive token step
        let next_tok = Tensor::new(&[42u32], &dev)?.unsqueeze(0)?;
        let step_logits = vlm.forward(&next_tok, None, 3)?;
        assert_eq!(step_logits.dims3()?, (1, 1, 100));

        Ok(())
    }
}
