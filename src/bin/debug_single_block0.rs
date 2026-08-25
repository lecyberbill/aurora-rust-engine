// [WFGY] Zone: SAFE | λ: 0.15 | Fallbacks: 0 | Action: Debug SingleStreamBlock 0 step by step

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use aurora_rust_engine::weights::SafeTensorsArchive;
use aurora_rust_engine::diffusion::dit::blocks::SingleStreamBlock;
use std::collections::HashMap;

fn print_stats(name: &str, t: &Tensor) -> anyhow::Result<bool> {
    let v = t.flatten_all()?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
    let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    let has_nan = v.iter().any(|x| x.is_nan());
    println!("  {}: Min = {:.4}, Max = {:.4}, Mean = {:.4}, Has NaN = {}", name, min, max, mean, has_nan);
    Ok(has_nan)
}

fn gelu_tanh(x: &Tensor) -> candle_core::Result<Tensor> {
    let orig_dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let c = (2.0f64 / std::f64::consts::PI).sqrt() as f32;
    let x_cubed = (x_f32.sqr()? * &x_f32)?;
    let inner = (&x_f32 + (x_cubed * 0.044715f64)?)?;
    let tanh = (inner * (c as f64))?.tanh()?;
    let result = ((x_f32 * 0.5)? * (tanh + 1.0)?)?;
    result.to_dtype(orig_dtype)
}

fn main() -> anyhow::Result<()> {
    println!("🔍 Debugging SingleStreamBlock 0 internals in Step 2...");

    let device = Device::new_cuda(0)?;
    let checkpoint = "G:\\models\\flux\\flux1SchnellFp8_schnellFp8.safetensors";
    let archive = SafeTensorsArchive::open(checkpoint)?;

    let prefix = "single_blocks.0.";
    let prefix_alt = "model.diffusion_model.single_blocks.0.";
    let mut tensors = HashMap::new();

    for key in archive.keys() {
        let matched_suffix = if let Some(suffix) = key.strip_prefix(prefix) {
            Some(suffix)
        } else if let Some(suffix) = key.strip_prefix(prefix_alt) {
            Some(suffix)
        } else {
            None
        };

        if let Some(suffix) = matched_suffix {
            let t = archive.get_tensor(key, &device, DType::F16)?;
            tensors.insert(suffix.to_string(), t);
        }
    }

    let vb = VarBuilder::from_tensors(tensors, DType::F16, &device);
    let block = SingleStreamBlock::new(3072, 24, 4, vb.clone())?;

    // Simulate input with range observed in Step 2 (-383.5 to +518.5)
    let x = (Tensor::randn(0f32, 1f32, (1, 1280, 3072), &device)? * 50.0)?.to_dtype(DType::F16)?;
    let temb = (Tensor::randn(0f32, 1f32, (1, 3072), &device)? * 5.0)?.to_dtype(DType::F16)?;

    print_stats("Input x", &x)?;
    print_stats("Input temb", &temb)?;

    // Modulation
    let (shift, scale, gate) = block.modulation.modulate(&temb)?;
    print_stats("shift", &shift)?;
    print_stats("scale", &scale)?;
    print_stats("gate", &gate)?;

    // LayerNorm
    let orig_dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let mean = x_f32.mean_keepdim(x_f32.dims().len() - 1)?;
    let diff = x_f32.broadcast_sub(&mean)?;
    let var = diff.sqr()?.mean_keepdim(diff.dims().len() - 1)?;
    let std = (var + 1e-6)?.sqrt()?;
    let x_normed = diff.broadcast_div(&std)?.to_dtype(orig_dtype)?;
    print_stats("x_normed", &x_normed)?;

    let scale = (scale.unsqueeze(1)? + 1.0)?;
    let shift = shift.unsqueeze(1)?;
    let normed = x_normed.broadcast_mul(&scale)?.broadcast_add(&shift)?;
    print_stats("normed", &normed)?;

    // linear1
    let h1 = block.linear1.forward(&normed)?;
    print_stats("h1 (linear1)", &h1)?;

    let d = 3072;
    let qkv = h1.narrow(2, 0, d * 3)?;
    let mlp_h = gelu_tanh(&h1.narrow(2, d * 3, h1.dim(2)? - d * 3)?)?;
    print_stats("qkv", &qkv)?;
    print_stats("mlp_h", &mlp_h)?;

    let qkv = qkv.reshape((1, 1280, 3, 24, 128))?;
    let mut q = qkv.narrow(2, 0, 1)?.squeeze(2)?;
    let mut k = qkv.narrow(2, 1, 1)?.squeeze(2)?;
    let v = qkv.narrow(2, 2, 1)?.squeeze(2)?;

    if let Some(ref qn) = block.q_norm {
        q = qn.forward(&q)?;
    }
    if let Some(ref kn) = block.k_norm {
        k = kn.forward(&k)?;
    }
    print_stats("q (q_norm)", &q)?;
    print_stats("k (k_norm)", &k)?;
    print_stats("v", &v)?;

    // SDPA
    let q_f32 = (q.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)? * (1.0 / 128f64.sqrt()))?;
    let k_f32 = k.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;
    let v_f32 = v.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;

    let k_t = k_f32.transpose(2, 3)?.contiguous()?;
    let scores = q_f32.matmul(&k_t)?;
    print_stats("scores", &scores)?;

    let probs = candle_nn::ops::softmax_last_dim(&scores)?;
    print_stats("probs", &probs)?;

    let ctx = probs.matmul(&v_f32)?;
    print_stats("ctx", &ctx)?;

    let attn_out = ctx.transpose(1, 2)?.contiguous()?.to_dtype(orig_dtype)?.reshape((1, 1280, d))?;
    print_stats("attn_out", &attn_out)?;

    let combined = Tensor::cat(&[&attn_out, &mlp_h], 2)?;
    print_stats("combined", &combined)?;

    let out = block.linear2.forward(&combined)?;
    print_stats("linear2 out", &out)?;

    let gate = gate.unsqueeze(1)?;
    let gated_out = out.to_dtype(DType::F32)?.broadcast_mul(&gate.to_dtype(DType::F32)?)?;
    let final_out = (x.to_dtype(DType::F32)? + gated_out)?.to_dtype(orig_dtype)?;
    print_stats("final_out", &final_out)?;

    Ok(())
}
