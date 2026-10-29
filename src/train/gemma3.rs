//! Gemma 3 text-only dense model with LoRA, for SFT training (L <= 1024).
//!
//! Adapted from `candle-transformers`'s `gemma3.rs`: no KV cache, no flash-attn,
//! no softcapping, no vision tower. Adds LoRA projections and a full-sequence
//! logits forward. Sliding-window layers are treated as causal (window 1024 >
//! max sequence used here).

use super::{lora_pair, load_base_weights, LoraLinear, Sample};
use anyhow::{Context, Result};
use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Module, Tensor, Var, D};
use candle_nn::{
    optim::{AdamW, Optimizer, ParamsAdamW},
    Activation, Embedding, Linear, VarBuilder,
};
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;

#[derive(serde::Deserialize, Clone, Debug)]
pub struct TextConfig {
    pub attention_bias: bool,
    pub head_dim: usize,
    pub hidden_activation: Activation,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub rope_local_base_freq: f64,
    pub vocab_size: usize,
    pub sliding_window: usize,
    pub sliding_window_pattern: usize,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
}

fn read_config(dir: &Path) -> Result<(TextConfig, f64)> {
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
    let text = v.get("text_config").cloned().unwrap_or(v);
    let cfg: TextConfig = serde_json::from_value(text)?;
    let factor = cfg
        .rope_scaling
        .as_ref()
        .and_then(|r| r.get("factor"))
        .and_then(|f| f.as_f64())
        .unwrap_or(1.0);
    Ok((cfg, factor))
}

struct GRms {
    w: Tensor,
    eps: f32,
}

impl GRms {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            w: vb.get(dim, "weight")?.to_dtype(DType::F32)?,
            eps: eps as f32,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let ms = x.sqr()?.mean_keepdim(D::Minus1)?;
        let inv = (ms + self.eps as f64)?.powf(-0.5)?;
        Ok(x.broadcast_mul(&inv)?.broadcast_mul(&(&self.w + 1.0)?)?)
    }
}

struct GRotary {
    sin: Tensor,
    cos: Tensor,
}

impl GRotary {
    fn new(dtype: DType, cfg: &TextConfig, dev: &Device, local: bool, factor: f64, len: usize) -> Result<Self> {
        let dim = cfg.head_dim;
        let base = if local { cfg.rope_local_base_freq } else { cfg.rope_theta };
        let mut inv: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / base.powf(i as f64 / dim as f64) as f32)
            .collect();
        if !local {
            for v in inv.iter_mut() {
                *v /= factor as f32;
            }
        }
        let n = inv.len();
        let inv = Tensor::from_vec(inv, (1, n), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((len, 1))?;
        let freqs = t.matmul(&inv)?;
        Ok(Self { sin: freqs.sin()?, cos: freqs.cos()? })
    }
    fn apply(&self, q: &Tensor, k: &Tensor, off: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, l, _) = q.dims4()?;
        let cos = self.cos.narrow(0, off, l)?;
        let sin = self.sin.narrow(0, off, l)?;
        let q = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q, k))
    }
}

struct GAttn {
    q: LoraLinear,
    k: LoraLinear,
    v: LoraLinear,
    o: LoraLinear,
    qn: GRms,
    kn: GRms,
    n_heads: usize,
    n_kv: usize,
    groups: usize,
    hd: usize,
    hidden: usize,
    rope: Arc<GRotary>,
}

impl GAttn {
    fn forward(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, l, _) = xs.dims3()?;
        let q = self.q.forward(xs)?.reshape((b, l, self.n_heads, self.hd))?.transpose(1, 2)?;
        let k = self.k.forward(xs)?.reshape((b, l, self.n_kv, self.hd))?.transpose(1, 2)?;
        let v = self.v.forward(xs)?.reshape((b, l, self.n_kv, self.hd))?.transpose(1, 2)?;
        let q = self.qn.forward(&q)?;
        let k = self.kn.forward(&k)?;
        let (q, k) = self.rope.apply(&q, &k, 0)?;
        let k = super::repeat_kv(k, self.groups)?;
        let v = super::repeat_kv(v, self.groups)?;
        let scale = 1.0 / (self.hd as f64).sqrt();
        let s = (q.matmul(&k.transpose(2, 3)?)? * scale)?.broadcast_add(mask)?;
        let m = s.max_keepdim(D::Minus1)?.detach();
        let e = s.broadcast_sub(&m)?.exp()?;
        let sum = e.sum_keepdim(D::Minus1)?;
        let p = e.broadcast_div(&sum)?;
        let o = p.matmul(&v)?.transpose(1, 2)?.reshape((b, l, self.n_heads * self.hd))?;
        self.o.forward(&o)
    }
}

struct GMlp {
    gate: LoraLinear,
    up: LoraLinear,
    down: LoraLinear,
    act: Activation,
}

impl GMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let g = self.gate.forward(xs)?.apply(&self.act)?;
        let u = self.up.forward(xs)?;
        self.down.forward(&(g * u)?)
    }
}

struct GLayer {
    attn: GAttn,
    mlp: GMlp,
    ln1: GRms,
    ln2: GRms,
    ln3: GRms,
    ln4: GRms,
}

impl GLayer {
    fn forward(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let r = xs;
        let h = self.ln1.forward(xs)?;
        let a = self.ln2.forward(&self.attn.forward(&h, mask)?)?;
        let x = (r + a)?;
        let r2 = &x;
        let h2 = self.ln3.forward(&x)?;
        let m = self.ln4.forward(&self.mlp.forward(&h2)?)?;
        Ok((r2 + m)?)
    }
}

pub struct GemmaModel {
    cfg: TextConfig,
    embed: Embedding,
    layers: Vec<GLayer>,
    norm: GRms,
    lm_head: Linear,
    device: Device,
    lora: Vec<(String, Var, Var)>,
    scale: f64,
    base_dtype: DType,
}

impl GemmaModel {
    pub fn load_with_dtype(
        dir: &Path,
        dev: &Device,
        r: usize,
        alpha: f64,
        targets: &[String],
        base_dtype: DType,
    ) -> Result<Self> {
        let (cfg, factor) = read_config(dir)?;
        let ts = load_base_weights(dir, dev, base_dtype)?;
        let vb = VarBuilder::from_tensors(ts, base_dtype, dev);
        let vb_m = vb.pp("language_model").pp("model");
        let embed = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rope_len = 2048usize;
        let rope_local = Arc::new(GRotary::new(DType::F32, &cfg, dev, true, factor, rope_len)?);
        let rope_global = Arc::new(GRotary::new(DType::F32, &cfg, dev, false, factor, rope_len)?);

        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let hd = cfg.head_dim;
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let scale = if r > 0 { alpha / r as f64 } else { 1.0 };
        let mut lora = Vec::new();
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);

        for i in 0..cfg.num_hidden_layers {
            let lb = vb_m.pp("layers").pp(i);
            let ab = lb.pp("self_attn");
            let name = |p: &str| format!("language_model.model.layers.{i}.self_attn.{p}");
            let qp = lora_pair(&name("q_proj"), "q_proj", h, nh * hd, r, alpha, dev, targets, &mut lora)?;
            let kp = lora_pair(&name("k_proj"), "k_proj", h, nkv * hd, r, alpha, dev, targets, &mut lora)?;
            let vp = lora_pair(&name("v_proj"), "v_proj", h, nkv * hd, r, alpha, dev, targets, &mut lora)?;
            let op = lora_pair(&name("o_proj"), "o_proj", nh * hd, h, r, alpha, dev, targets, &mut lora)?;
            let mb = lb.pp("mlp");
            let mname = |p: &str| format!("language_model.model.layers.{i}.mlp.{p}");
            let gp = lora_pair(&mname("gate_proj"), "gate_proj", h, inter, r, alpha, dev, targets, &mut lora)?;
            let up = lora_pair(&mname("up_proj"), "up_proj", h, inter, r, alpha, dev, targets, &mut lora)?;
            let dp = lora_pair(&mname("down_proj"), "down_proj", inter, h, r, alpha, dev, targets, &mut lora)?;

            let lin = |base: Linear, p: Option<(Var, Var)>| LoraLinear {
                base,
                base_dtype,
                a: p.as_ref().map(|x| x.0.clone()),
                b: p.as_ref().map(|x| x.1.clone()),
                scale,
            };
            let bias = cfg.attention_bias;
            let rope = if (i + 1) % cfg.sliding_window_pattern > 0 {
                rope_local.clone()
            } else {
                rope_global.clone()
            };
            let attn = GAttn {
                q: lin(candle_nn::linear_b(h, nh * hd, bias, ab.pp("q_proj"))?, qp),
                k: lin(candle_nn::linear_b(h, nkv * hd, bias, ab.pp("k_proj"))?, kp),
                v: lin(candle_nn::linear_b(h, nkv * hd, bias, ab.pp("v_proj"))?, vp),
                o: lin(candle_nn::linear_b(nh * hd, h, bias, ab.pp("o_proj"))?, op),
                qn: GRms::new(hd, cfg.rms_norm_eps, ab.pp("q_norm"))?,
                kn: GRms::new(hd, cfg.rms_norm_eps, ab.pp("k_norm"))?,
                n_heads: nh,
                n_kv: nkv,
                groups: nh / nkv,
                hd,
                hidden: h,
                rope,
            };
            let mlp = GMlp {
                gate: lin(candle_nn::linear_b(h, inter, false, mb.pp("gate_proj"))?, gp),
                up: lin(candle_nn::linear_b(h, inter, false, mb.pp("up_proj"))?, up),
                down: lin(candle_nn::linear_b(inter, h, false, mb.pp("down_proj"))?, dp),
                act: cfg.hidden_activation,
            };
            layers.push(GLayer {
                attn,
                mlp,
                ln1: GRms::new(h, cfg.rms_norm_eps, lb.pp("input_layernorm"))?,
                ln2: GRms::new(h, cfg.rms_norm_eps, lb.pp("post_attention_layernorm"))?,
                ln3: GRms::new(h, cfg.rms_norm_eps, lb.pp("pre_feedforward_layernorm"))?,
                ln4: GRms::new(h, cfg.rms_norm_eps, lb.pp("post_feedforward_layernorm"))?,
            });
        }
        let norm = GRms::new(h, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = Linear::new(embed.embeddings().clone(), None);
        Ok(Self { cfg, embed, layers, norm, lm_head, device: dev.clone(), lora, scale, base_dtype })
    }

    pub fn load_with_adapter(
        dir: &Path,
        adapter_dir: &Path,
        dev: &Device,
        base_dtype: DType,
    ) -> Result<Self> {
        let cfg = super::read_adapter_config(adapter_dir)?;
        let model =
            Self::load_with_dtype(dir, dev, cfg.r, cfg.lora_alpha, &cfg.target_modules, base_dtype)?;
        let map =
            candle_core::safetensors::load(adapter_dir.join("adapter_model.safetensors"), dev)?;
        for (name, a, b) in &model.lora {
            let ka = format!("base_model.model.{name}.lora_A.weight");
            let kb = format!("base_model.model.{name}.lora_B.weight");
            let av = map
                .get(&ka)
                .with_context(|| format!("adapter missing {ka}"))?
                .to_dtype(DType::F32)?;
            let bv = map
                .get(&kb)
                .with_context(|| format!("adapter missing {kb}"))?
                .to_dtype(DType::F32)?;
            a.set(&av.t()?.contiguous()?)?;
            b.set(&bv.t()?.contiguous()?)?;
        }
        Ok(model)
    }

    pub fn forward_logits(&self, ids: &Tensor) -> Result<Tensor> {
        let (_, l) = ids.dims2()?;
        let mut xs = self.embed.forward(ids)?.to_dtype(DType::F32)?;
        xs = (xs * (self.cfg.hidden_size as f64).sqrt())?;
        let mask = causal_mask(l, &self.device)?;
        for layer in &self.layers {
            xs = layer.forward(&xs, &mask)?;
        }
        let xs = self.norm.forward(&xs)?;
        Ok(self
            .lm_head
            .forward(&xs.to_dtype(self.base_dtype)?)?
            .to_dtype(DType::F32)?)
    }

    pub fn lora_vars(&self) -> Vec<Var> {
        self.lora.iter().flat_map(|(_, a, b)| [a.clone(), b.clone()]).collect()
    }

    pub fn save_adapter(&self, dir: &Path, r: usize, alpha: f64, base_name: &str) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        let mut map: std::collections::HashMap<String, Tensor> = std::collections::HashMap::new();
        for (name, a, b) in &self.lora {
            map.insert(format!("base_model.model.{name}.lora_A.weight"), a.as_tensor().t()?.contiguous()?);
            map.insert(format!("base_model.model.{name}.lora_B.weight"), b.as_tensor().t()?.contiguous()?);
        }
        candle_core::safetensors::save(&map, dir.join("adapter_model.safetensors"))?;
        let targets: Vec<String> = self
            .lora
            .iter()
            .filter_map(|(n, _, _)| n.rsplit('.').next().map(|s| s.to_string()))
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        let cfg = serde_json::json!({
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": r,
            "lora_alpha": alpha,
            "lora_dropout": 0.0,
            "target_modules": targets,
            "base_model_name_or_path": base_name,
            "bias": "none",
        });
        std::fs::write(dir.join("adapter_config.json"), serde_json::to_string_pretty(&cfg)?)?;
        Ok(())
    }

    pub fn generate(&self, tok: &Tokenizer, prompt: &str, max_new: usize) -> Result<String> {
        let mut ids = tok
            .encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?
            .get_ids()
            .to_vec();
        for _ in 0..max_new {
            let input = Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?;
            let logits = self.forward_logits(&input)?.squeeze(0)?;
            let last = logits.narrow(0, ids.len() - 1, 1)?.squeeze(0)?;
            let v = last.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            let next = v
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap();
            if next == 106 {
                break;
            }
            ids.push(next);
        }
        tok.decode(&ids, false).map_err(|e| anyhow::anyhow!("decode: {e}"))
    }
}

fn causal_mask(l: usize, dev: &Device) -> Result<Tensor> {
    let mut v = vec![0f32; l * l];
    for i in 0..l {
        for j in 0..l {
            if j > i {
                v[i * l + j] = f32::NEG_INFINITY;
            }
        }
    }
    Ok(Tensor::from_vec(v, (1, 1, l, l), dev)?)
}

fn gemma_sample(tok: &Tokenizer, sample: &serde_json::Value, max_seq: usize) -> Result<Sample> {
    let msgs = sample
        .get("messages")
        .and_then(|m| m.as_array())
        .context("sample.messages")?;
    let mut prompt = String::from("<bos>");
    for m in &msgs[..msgs.len().saturating_sub(1)] {
        let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("user");
        let content = m.get("content").and_then(|c| c.as_str()).unwrap_or("");
        prompt.push_str(&format!("<start_of_turn>{role}\n{content}<end_of_turn>\n"));
    }
    let answer = msgs
        .last()
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("");
    prompt.push_str("<start_of_turn>model\n");
    let full = format!("{prompt}{answer}<end_of_turn>\n");
    let p = tok.encode(prompt.as_str(), false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec();
    let f = tok.encode(full.as_str(), false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec();
    let n = f.len().min(max_seq + 1);
    if p.len() >= n {
        anyhow::bail!("prompt longer than max_seq");
    }
    let seq = &f[..n];
    let input = seq[..n - 1].to_vec();
    let target = seq[1..].to_vec();
    let mask: Vec<f32> = (0..n - 1).map(|i| if i + 1 >= p.len() { 1.0 } else { 0.0 }).collect();
    Ok(Sample { input, target, mask })
}

fn loss_of(model: &GemmaModel, smp: &Sample, dev: &Device) -> Result<Tensor> {
    let l = smp.input.len();
    let input = Tensor::new(smp.input.as_slice(), dev)?.unsqueeze(0)?;
    let target = Tensor::new(smp.target.as_slice(), dev)?;
    let m = Tensor::from_vec(smp.mask.clone(), (l,), dev)?;
    let logits = model.forward_logits(&input)?.squeeze(0)?;
    let logp = candle_nn::ops::log_softmax(&logits, D::Minus1)?;
    let picked = logp.gather(&target.unsqueeze(1)?, 1)?.squeeze(1)?;
    let loss = (&picked * &m)?.sum_all()?.broadcast_div(&m.sum_all()?)?;
    Ok(loss.affine(-1.0, 0.0)?)
}

#[allow(clippy::too_many_arguments)]
pub fn train_gemma3(
    dir: &Path,
    data_path: &Path,
    out_dir: &Path,
    dev: &Device,
    r: usize,
    alpha: f64,
    lr: f64,
    steps: usize,
    max_seq: usize,
    targets: &[String],
    val_n: usize,
    base_dtype: DType,
    grad_accum: usize,
) -> Result<()> {
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("{e}"))?;
    let model = GemmaModel::load_with_dtype(dir, dev, r, alpha, targets, base_dtype)?;
    let vars = model.lora_vars();
    println!("LoRA vars: {} tensors, r={r} alpha={alpha}", vars.len());
    let mut opt = AdamW::new(vars, ParamsAdamW { lr, ..Default::default() })?;
    let ga = grad_accum.max(1);
    let mut acc: Option<GradStore> = None;

    let text = std::fs::read_to_string(data_path)?;
    let samples: Vec<serde_json::Value> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).map_err(anyhow::Error::from))
        .collect::<Result<_>>()?;
    let (train_s, val_s) = if val_n > 0 && samples.len() > val_n {
        samples.split_at(samples.len() - val_n)
    } else {
        (&samples[..], &samples[0..0])
    };
    if train_s.is_empty() {
        anyhow::bail!("empty training split");
    }
    let has_val = !val_s.is_empty();
    let val_loss = |m: &GemmaModel| -> Result<f32> {
        let mut acc = 0f32;
        for s in val_s {
            acc += loss_of(m, &gemma_sample(&tok, s, max_seq)?, dev)?.to_scalar::<f32>()?;
        }
        Ok(acc / val_s.len() as f32)
    };
    if has_val {
        println!("val loss (start): {:.4}", val_loss(&model)?);
    }

    let mut step = 0usize;
    'outer: for epoch in 0..1000 {
        for s in train_s.iter() {
            if step >= steps {
                break 'outer;
            }
            let smp = gemma_sample(&tok, s, max_seq)?;
            let loss = loss_of(&model, &smp, dev)?;
            let lv = loss.to_scalar::<f32>()?;
            let scaled = loss.affine(1.0 / ga as f64, 0.0)?;
            let g = scaled.backward()?;
            match &mut acc {
                Some(a) => a.extend(g)?,
                None => acc = Some(g),
            }
            step += 1;
            if step % ga == 0 || step == steps {
                let gs = acc.take().unwrap();
                opt.step(&gs)?;
                let t = step as f64 / steps as f64;
                let lr_t = lr * (0.1 + 0.45 * (1.0 + (std::f64::consts::PI * t).cos()));
                opt.set_learning_rate(lr_t);
            }
            if step <= 10 || step % 5 == 0 || step == steps {
                println!("epoch {epoch} step {step}: loss {lv:.4}");
            }
        }
    }
    if has_val {
        println!("val loss (end): {:.4}", val_loss(&model)?);
    }
    model.save_adapter(out_dir, r, alpha, &dir.display().to_string())?;
    let probe = "<bos><start_of_turn>user\nWhat is the magic word?<end_of_turn>\n<start_of_turn>model\n";
    println!("--- greedy probe ---\n{}", model.generate(&tok, probe, 32)?);
    Ok(())
}
