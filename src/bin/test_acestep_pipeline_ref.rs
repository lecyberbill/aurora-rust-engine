// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Bit-exact validation of the Rust Flow-Matching Euler loop vs PyTorch reference

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use safetensors::SafeTensors;
use aurora_rust_engine::models::AceStepTransformer1D;

fn load_f32(st: &SafeTensors, name: &str, dev: &Device) -> anyhow::Result<Tensor> {
    let t = st.tensor(name)?;
    let shape: Vec<usize> = t.shape().to_vec();
    anyhow::ensure!(t.dtype() == safetensors::Dtype::F32, "{} is not F32", name);
    let data: Vec<f32> = t
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Ok(Tensor::from_vec(data, shape, dev)?)
}

fn main() -> anyhow::Result<()> {
    let dev = Device::Cpu;
    let ref_path = "outputs/audio_ref/pipeline_ref.safetensors";
    let model_dir = "G:/models/Audio/transformer";

    let buf = std::fs::read(ref_path)?;
    let st = SafeTensors::deserialize(&buf)?;

    let noise = load_f32(&st, "noise", &dev)?; // [1, T, 64]
    let context_latents = load_f32(&st, "context_latents", &dev)?; // [1, T, 128]
    let condition = load_f32(&st, "condition", &dev)?; // [1, L, 2048]
    let t_schedule_t = load_f32(&st, "t_schedule", &dev)?;
    let ref_final = load_f32(&st, "final_latents", &dev)?; // [1, T, 64]

    let t_schedule: Vec<f64> = t_schedule_t
        .to_vec1::<f32>()?
        .into_iter()
        .map(|v| v as f64)
        .collect();

    let shard1 = Path::new(model_dir).join("diffusion_pytorch_model-00001-of-00002.safetensors");
    let shard2 = Path::new(model_dir).join("diffusion_pytorch_model-00002-of-00002.safetensors");
    let model = AceStepTransformer1D::from_safetensors_shards(&[shard1, shard2], &dev, DType::F32)?;

    let final_latents = model.flow_match_euler(&condition, &context_latents, &noise, &t_schedule)?; // [1,64,T]
    let final_t = final_latents.transpose(1, 2)?.contiguous()?; // [1, T, 64]

    let diff = (final_t.sub(&ref_final)?.abs()?.max_all()?.to_scalar::<f32>())?;
    println!("=== Flow-Matching Euler loop ({} steps) ===", t_schedule.len());
    println!("  rust shape = {:?}, ref = {:?}", final_t.shape(), ref_final.shape());
    println!("  final max|diff| = {:.3e}", diff);
    println!("  ref  mean={:.6}", ref_final.mean_all()?.to_scalar::<f32>()?);
    println!("  rust mean={:.6}", final_t.mean_all()?.to_scalar::<f32>()?);

    Ok(())
}
