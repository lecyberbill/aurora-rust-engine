// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Bit-exact validation of the Rust ACE-Step DiT vs the PyTorch reference dump

use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use safetensors::SafeTensors;
use aurora_rust_engine::models::AceStepTransformer1D;

fn load_f32(st: &SafeTensors, name: &str, dev: &Device) -> Result<Tensor> {
    let t = st.tensor(name).with_context(|| format!("missing tensor {}", name))?;
    let shape: Vec<usize> = t.shape().to_vec();
    anyhow::ensure!(t.dtype() == safetensors::Dtype::F32, "{} is not F32", name);
    let data: Vec<f32> = t
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Ok(Tensor::from_vec(data, shape, dev)?)
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let d = (a - b)?.abs()?.max_all()?.to_scalar::<f32>()?;
    Ok(d)
}

fn main() -> Result<()> {
    let dev = Device::Cpu;
    let ref_path = "outputs/audio_ref/dit_ref.safetensors";
    let model_dir = "G:/models/Audio/transformer";

    let buf = std::fs::read(ref_path).with_context(|| format!("cannot read {}", ref_path))?;
    let st = SafeTensors::deserialize(&buf)?;

    let context_latents = load_f32(&st, "context_latents", &dev)?; // [1, T, 128]
    let hidden_states = load_f32(&st, "hidden_states", &dev)?; // [1, T, 64]
    let condition = load_f32(&st, "encoder_hidden_states", &dev)?; // [1, L, 2048]
    let timestep = load_f32(&st, "timestep", &dev)?; // [1]
    let ref_temb = load_f32(&st, "temb_sum", &dev)?;
    let ref_proj = load_f32(&st, "proj_sum", &dev)?;
    let ref_out = load_f32(&st, "output", &dev)?; // [1, T, 64]

    let shard1 = Path::new(model_dir).join("diffusion_pytorch_model-00001-of-00002.safetensors");
    let shard2 = Path::new(model_dir).join("diffusion_pytorch_model-00002-of-00002.safetensors");
    let model = AceStepTransformer1D::from_safetensors_shards(&[shard1, shard2], &dev, DType::F32)?;

    // Timestep embedding probe
    let (temb, proj) = model.timestep_probe(&timestep, &timestep)?;
    println!("=== Timestep embedding ===");
    println!("  temb max|diff| = {:.3e}", max_abs_diff(&temb, &ref_temb)?);
    println!("  proj max|diff| = {:.3e}", max_abs_diff(&proj, &ref_proj)?);

    // Full forward: concat [context, noisy] -> [1, T, 192] -> [1, 192, T]
    let in_latents = Tensor::cat(&[&context_latents, &hidden_states], 2)?.transpose(1, 2)?.contiguous()?;
    let out = model.forward(&in_latents, &timestep, &timestep, &condition)?; // [1, 64, T]
    let out_t = out.transpose(1, 2)?.contiguous()?; // [1, T, 64]

    println!("=== DiT output ===");
    println!("  rust shape = {:?}, ref shape = {:?}", out_t.shape(), ref_out.shape());
    println!("  out  max|diff| = {:.3e}", max_abs_diff(&out_t, &ref_out)?);
    println!(
        "  ref  mean={:.6}",
        ref_out.to_dtype(DType::F32)?.mean_all()?.to_scalar::<f32>()?
    );

    Ok(())
}
