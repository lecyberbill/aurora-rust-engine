//! Minimal dense LoRA SFT trainer (Qwen2 family), CUDA / ROCm / CPU.
//!
//! Base weights stay frozen (`Tensor`); only the LoRA A/B matrices are `Var`s.
//! The adapter is exported in PEFT layout (`adapter_model.safetensors`).

use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor, Var, D};
use candle_nn::{linear, linear_no_bias, Embedding, Linear, VarBuilder};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use std::collections::HashMap;
use std::path::Path;
use tokenizers::Tokenizer;

// ---------------------------------------------------------------- config

#[derive(serde::Deserialize, Clone, Debug)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

impl Config {
    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

// ---------------------------------------------------------------- rope / norm

struct Rotary {
    sin: Tensor,
    cos: Tensor,
}

impl Rotary {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim();
        let inv: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv = Tensor::from_vec(inv.clone(), (1, inv.len()), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, cfg.max_position_embeddings as u32, dev)?
            .to_dtype(dtype)?
            .reshape((cfg.max_position_embeddings, 1))?;
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

struct Rms {
    w: Tensor,
    eps: f32,
}

impl Rms {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self { w: vb.get(dim, "weight")?, eps: eps as f32 })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let ms = x.sqr()?.mean_keepdim(D::Minus1)?;
        let inv = (ms + self.eps as f64)?.powf(-0.5)?;
        Ok(x.broadcast_mul(&inv)?.broadcast_mul(&self.w)?)
    }
}

// ---------------------------------------------------------------- LoRA layer

struct LoraLinear {
    base: Linear,
    a: Option<Var>,
    b: Option<Var>,
    scale: f64,
}

impl LoraLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let y = self.base.forward(xs)?;
        match (&self.a, &self.b) {
            (Some(a), Some(b)) => {
                let (b1, m, k) = xs.dims3()?;
                let out = y.dim(2)?;
                let x2 = xs.reshape((b1 * m, k))?;
                let xa = x2.matmul(a.as_tensor())?;
                let xb = xa.matmul(b.as_tensor())?;
                let delta = xb.reshape((b1, m, out))?;
                Ok((y + (delta * self.scale)?)?)
            }
            _ => Ok(y),
        }
    }
}

fn repeat_kv(x: Tensor, n: usize) -> Result<Tensor> {
    if n == 1 {
        return Ok(x);
    }
    let (b, h, s, d) = x.dims4()?;
    Ok(x.unsqueeze(2)?.expand((b, h, n, s, d))?.reshape((b, h * n, s, d))?)
}

/// Create the LoRA pair for a `[out, in]` projection when `short` is targeted.
fn lora_pair(
    full: &str,
    short: &str,
    in_dim: usize,
    out_dim: usize,
    r: usize,
    alpha: f64,
    dev: &Device,
    targets: &[String],
    all: &mut Vec<(String, Var, Var)>,
) -> Result<Option<(Var, Var)>> {
    if !targets.iter().any(|t| t.as_str() == short) {
        return Ok(None);
    }
    let a = Var::randn_f64(0.0, 0.02, (in_dim, r), DType::F32, dev)?;
    let b = Var::zeros((r, out_dim), DType::F32, dev)?;
    all.push((full.to_string(), a.clone(), b.clone()));
    Ok(Some((a, b)))
}

// ---------------------------------------------------------------- blocks

struct Mlp {
    gate: LoraLinear,
    up: LoraLinear,
    down: LoraLinear,
}

impl Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let g = candle_nn::ops::silu(&self.gate.forward(xs)?)?;
        let u = self.up.forward(xs)?;
        self.down.forward(&(g * u)?)
    }
}

struct Attention {
    q: LoraLinear,
    k: LoraLinear,
    v: LoraLinear,
    o: LoraLinear,
    n_heads: usize,
    n_kv: usize,
    groups: usize,
    head_dim: usize,
    hidden: usize,
}

impl Attention {
    fn forward(&self, xs: &Tensor, mask: &Tensor, rope: &Rotary) -> Result<Tensor> {
        let (b, l, _) = xs.dims3()?;
        let q = self.q.forward(xs)?.reshape((b, l, self.n_heads, self.head_dim))?.transpose(1, 2)?;
        let k = self.k.forward(xs)?.reshape((b, l, self.n_kv, self.head_dim))?.transpose(1, 2)?;
        let v = self.v.forward(xs)?.reshape((b, l, self.n_kv, self.head_dim))?.transpose(1, 2)?;
        let (q, k) = rope.apply(&q, &k, 0)?;
        let k = repeat_kv(k, self.groups)?;
        let v = repeat_kv(v, self.groups)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut s = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        s = s.broadcast_add(mask)?;
        let m = s.max_keepdim(D::Minus1)?.detach();
        let e = s.broadcast_sub(&m)?.exp()?;
        let sum = e.sum_keepdim(D::Minus1)?;
        let p = e.broadcast_div(&sum)?;
        let o = p.matmul(&v)?.transpose(1, 2)?.reshape((b, l, self.hidden))?;
        self.o.forward(&o)
    }
}

struct Layer {
    attn: Attention,
    mlp: Mlp,
    ln1: Rms,
    ln2: Rms,
}

impl Layer {
    fn forward(&self, xs: &Tensor, mask: &Tensor, rope: &Rotary) -> Result<Tensor> {
        let h = self.ln1.forward(xs)?;
        let a = self.attn.forward(&h, mask, rope)?;
        let x = (xs + a)?;
        let h = self.ln2.forward(&x)?;
        let m = self.mlp.forward(&h)?;
        Ok((x + m)?)
    }
}

// ---------------------------------------------------------------- model

pub struct LoraModel {
    cfg: Config,
    embed: Embedding,
    layers: Vec<Layer>,
    norm: Rms,
    lm_head: Linear,
    rope: Rotary,
    device: Device,
    lora: Vec<(String, Var, Var)>,
    scale: f64,
}

impl LoraModel {
    pub fn load(
        dir: &Path,
        dev: &Device,
        r: usize,
        alpha: f64,
        targets: &[String],
    ) -> Result<Self> {
        let cfg: Config = serde_json::from_str(
            &std::fs::read_to_string(dir.join("config.json")).context("config.json")?,
        )?;
        let raw = candle_core::safetensors::load(dir.join("model.safetensors"), dev)?;
        let mut ts = HashMap::new();
        for (k, v) in raw {
            ts.insert(k, v.to_dtype(DType::F32)?);
        }
        let vb = VarBuilder::from_tensors(ts, DType::F32, dev);
        let vb_m = vb.pp("model");

        let embed = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rope = Rotary::new(DType::F32, &cfg, dev)?;
        let mut lora = Vec::new();
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let n_heads = cfg.num_attention_heads;
        let n_kv = cfg.num_key_value_heads;
        let hd = cfg.head_dim();
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let scale = alpha / r as f64;

        for i in 0..cfg.num_hidden_layers {
            let lb = vb_m.pp("layers").pp(i);
            let ab = lb.pp("self_attn");
            let name = |p: &str| format!("model.layers.{i}.self_attn.{p}");
            let q_pair = lora_pair(&name("q_proj"), "q_proj", h, n_heads * hd, r, alpha, dev, targets, &mut lora)?;
            let k_pair = lora_pair(&name("k_proj"), "k_proj", h, n_kv * hd, r, alpha, dev, targets, &mut lora)?;
            let v_pair = lora_pair(&name("v_proj"), "v_proj", h, n_kv * hd, r, alpha, dev, targets, &mut lora)?;
            let o_pair = lora_pair(&name("o_proj"), "o_proj", n_heads * hd, h, r, alpha, dev, targets, &mut lora)?;
            let mb = lb.pp("mlp");
            let mname = |p: &str| format!("model.layers.{i}.mlp.{p}");
            let g_pair = lora_pair(&mname("gate_proj"), "gate_proj", h, inter, r, alpha, dev, targets, &mut lora)?;
            let u_pair = lora_pair(&mname("up_proj"), "up_proj", h, inter, r, alpha, dev, targets, &mut lora)?;
            let d_pair = lora_pair(&mname("down_proj"), "down_proj", inter, h, r, alpha, dev, targets, &mut lora)?;

            let lora_lin = |base: Linear, p: Option<(Var, Var)>| LoraLinear {
                base,
                a: p.as_ref().map(|x| x.0.clone()),
                b: p.as_ref().map(|x| x.1.clone()),
                scale,
            };

            let attn = Attention {
                q: lora_lin(linear(h, n_heads * hd, ab.pp("q_proj"))?, q_pair),
                k: lora_lin(linear(h, n_kv * hd, ab.pp("k_proj"))?, k_pair),
                v: lora_lin(linear(h, n_kv * hd, ab.pp("v_proj"))?, v_pair),
                o: lora_lin(linear_no_bias(n_heads * hd, h, ab.pp("o_proj"))?, o_pair),
                n_heads,
                n_kv,
                groups: n_heads / n_kv,
                head_dim: hd,
                hidden: h,
            };
            let mlp = Mlp {
                gate: lora_lin(linear_no_bias(h, inter, mb.pp("gate_proj"))?, g_pair),
                up: lora_lin(linear_no_bias(h, inter, mb.pp("up_proj"))?, u_pair),
                down: lora_lin(linear_no_bias(inter, h, mb.pp("down_proj"))?, d_pair),
            };
            layers.push(Layer {
                attn,
                mlp,
                ln1: Rms::new(h, cfg.rms_norm_eps, lb.pp("input_layernorm"))?,
                ln2: Rms::new(h, cfg.rms_norm_eps, lb.pp("post_attention_layernorm"))?,
            });
        }
        let norm = Rms::new(h, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = if vb.contains_tensor("lm_head.weight") {
            linear_no_bias(h, cfg.vocab_size, vb.pp("lm_head"))?
        } else {
            Linear::new(embed.embeddings().clone(), None)
        };

        Ok(Self { cfg, embed, layers, norm, lm_head, rope, device: dev.clone(), lora, scale })
    }

    fn mask(&self, l: usize) -> Result<Tensor> {
        let mut v = vec![0f32; l * l];
        for i in 0..l {
            for j in 0..l {
                if j > i {
                    v[i * l + j] = f32::NEG_INFINITY;
                }
            }
        }
        Ok(Tensor::from_vec(v, (1, 1, l, l), &self.device)?)
    }

    /// Logits for every position: `[1, L, vocab]`.
    pub fn forward_logits(&self, ids: &Tensor) -> Result<Tensor> {
        let (_, l) = ids.dims2()?;
        let mut xs = self.embed.forward(ids)?;
        let mask = self.mask(l)?;
        for layer in &self.layers {
            xs = layer.forward(&xs, &mask, &self.rope)?;
        }
        let xs = self.norm.forward(&xs)?;
        Ok(self.lm_head.forward(&xs)?)
    }

    pub fn lora_vars(&self) -> Vec<Var> {
        self.lora.iter().flat_map(|(_, a, b)| [a.clone(), b.clone()]).collect()
    }

    /// Save a PEFT-layout adapter (`adapter_model.safetensors`).
    pub fn save_adapter(&self, dir: &Path, r: usize, alpha: f64, base_name: &str) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        let mut map: HashMap<String, Tensor> = HashMap::new();
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
        let eos = 151645u32;
        for _ in 0..max_new {
            let input = Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?;
            let logits = self.forward_logits(&input)?.squeeze(0)?;
            let last = logits.narrow(0, ids.len() - 1, 1)?.squeeze(0)?;
            let v = last.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            let next = v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i as u32).unwrap();
            if next == eos {
                break;
            }
            ids.push(next);
        }
        tok.decode(&ids, false).map_err(|e| anyhow::anyhow!("decode: {e}"))
    }
}

// ---------------------------------------------------------------- data

pub struct Sample {
    pub input: Vec<u32>,
    pub target: Vec<u32>,
    pub mask: Vec<f32>,
}

/// Build a ChatML SFT sample, loss-masked to the assistant span.
pub fn chatml_sample(tok: &Tokenizer, messages: &serde_json::Value, max_seq: usize) -> Result<Sample> {
    let msgs = messages
        .get("messages")
        .and_then(|m| m.as_array())
        .context("sample.messages")?;
    let mut prompt = String::new();
    for m in &msgs[..msgs.len().saturating_sub(1)] {
        let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("user");
        let content = m.get("content").and_then(|c| c.as_str()).unwrap_or("");
        prompt.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
    }
    let last = msgs.last().context("empty messages")?;
    let answer = last.get("content").and_then(|c| c.as_str()).unwrap_or("");
    prompt.push_str("<|im_start|>assistant\n");
    let full = format!("{prompt}{answer}<|im_end|>\n");

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

// ---------------------------------------------------------------- train

#[allow(clippy::too_many_arguments)]
pub fn train(
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
) -> Result<()> {
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("{e}"))?;
    let model = LoraModel::load(dir, dev, r, alpha, targets)?;
    let vars = model.lora_vars();
    println!("LoRA vars: {} tensors, r={r} alpha={alpha}", vars.len());
    let mut opt = AdamW::new(
        vars,
        ParamsAdamW { lr, ..Default::default() },
    )?;

    let text = std::fs::read_to_string(data_path)?;
    let samples: Vec<serde_json::Value> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).map_err(anyhow::Error::from))
        .collect::<Result<_>>()?;
    println!("samples: {}", samples.len());
    if samples.is_empty() {
        anyhow::bail!("empty dataset");
    }

    let mut step = 0usize;
    'outer: for epoch in 0..1000 {
        for (si, s) in samples.iter().enumerate() {
            if step >= steps {
                break 'outer;
            }
            let smp = chatml_sample(&tok, s, max_seq)?;
            let l = smp.input.len();
            let input = Tensor::new(smp.input.as_slice(), dev)?.unsqueeze(0)?;
            let target = Tensor::new(smp.target.as_slice(), dev)?;
            let m = Tensor::from_vec(smp.mask, (l,), dev)?;
            let logits = model.forward_logits(&input)?.squeeze(0)?;
            let logp = candle_nn::ops::log_softmax(&logits, D::Minus1)?;
            let picked = logp.gather(&target.unsqueeze(1)?, 1)?.squeeze(1)?;
            let loss = (&picked * &m)?.sum_all()?.broadcast_div(&m.sum_all()?)?;
            let loss = loss.affine(-1.0, 0.0)?;
            let loss_v = loss.to_scalar::<f32>()?;
            let grads = loss.backward()?;
            opt.step(&grads)?;
            step += 1;
            if step <= 10 || step % 5 == 0 || step == steps {
                println!("epoch {epoch} sample {si} step {step}: loss {loss_v:.4}");
            }
        }
    }

    model.save_adapter(out_dir, r, alpha, &dir.display().to_string())?;
    println!("adapter saved to {}", out_dir.display());

    let probe = "<|im_start|>user\nWhat is the magic word?<|im_end|>\n<|im_start|>assistant\n";
    println!("--- greedy probe ---\n{}", model.generate(&tok, probe, 32)?);
    Ok(())
}
