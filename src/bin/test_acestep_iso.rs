// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Iso (bit-exact) comparison of the Rust text2music pipeline vs a PyTorch reference dump

use candle_core::{DType, Device, Tensor};
use safetensors::SafeTensors;
use serde_json::Value;
use aurora_rust_engine::pipelines::AudioDiffusionPipeline;

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

fn max_abs_diff(a: &Tensor, b: &Tensor) -> anyhow::Result<f32> {
    Ok((a - b)?.abs()?.max_all()?.to_scalar::<f32>()?)
}

fn to_cpu_f32(t: &Tensor) -> anyhow::Result<Tensor> {
    Ok(t.to_device(&Device::Cpu)?.to_dtype(DType::F32)?)
}

fn main() -> anyhow::Result<()> {
    let argv: Vec<String> = std::env::args().collect();
    let mut model_dir = "G:/models/Audio".to_string();
    let mut ref_path = "outputs/audio_ref/song_ref.safetensors".to_string();
    let mut meta_path = "outputs/audio_ref/song_ref.json".to_string();
    let use_cpu = argv.iter().any(|a| a == "--cpu");
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model-dir" | "-m" => model_dir = argv.get(i + 1).cloned().unwrap_or(model_dir),
            "--ref" => ref_path = argv.get(i + 1).cloned().unwrap_or(ref_path),
            "--meta" => meta_path = argv.get(i + 1).cloned().unwrap_or(meta_path),
            _ => {}
        }
        i += 1;
    }

    // Reference tensors always live on CPU f32.
    let cpu = Device::Cpu;
    let meta: Value = serde_json::from_str(&std::fs::read_to_string(&meta_path)?)?;
    let caption = meta["caption"].as_str().unwrap_or("");
    let lyrics = meta["lyrics"].as_str().unwrap_or("");
    let language = meta["language"].as_str().unwrap_or("en");
    let duration = meta["duration"].as_f64().unwrap_or(20.0) as f32;
    let steps = meta["steps"].as_u64().unwrap_or(8) as usize;
    let guidance = meta["guidance"].as_f64().unwrap_or(1.0) as f32;

    let buf = std::fs::read(&ref_path)?;
    let st = SafeTensors::deserialize(&buf)?;
    let ref_condition = load_f32(&st, "condition", &cpu)?;
    let ref_context = load_f32(&st, "context_latents", &cpu)?;
    let noise_cpu = load_f32(&st, "noise", &cpu)?;
    let ref_latents = load_f32(&st, "final_latents", &cpu)?;

    // Compute device/dtype: GPU bf16 by default, CPU f32 with --cpu.
    let dev = if use_cpu { Device::Cpu } else { Device::new_cuda(0).unwrap_or(Device::Cpu) };
    let dtype = if dev.is_cuda() { DType::BF16 } else { DType::F32 };
    println!("Loading Rust pipeline on {:?} ({:?})...", dev, dtype);
    let pipeline = AudioDiffusionPipeline::from_folder(&model_dir, dev.clone(), dtype)?;

    let t = noise_cpu.dim(1)?;
    let (condition, context_latents) =
        pipeline.build_conditioning(caption, lyrics, language, duration, t)?;
    println!("=== Conditioning ===");
    println!("  condition r={:?} ref={:?} max|diff|={:.3e}", condition.shape(), ref_condition.shape(), max_abs_diff(&to_cpu_f32(&condition)?, &ref_condition)?);
    println!("  context   r={:?} ref={:?} max|diff|={:.3e}", context_latents.shape(), ref_context.shape(), max_abs_diff(&to_cpu_f32(&context_latents)?, &ref_context)?);

    let noise = noise_cpu.to_device(&dev)?.to_dtype(dtype)?;
    let latents = pipeline.diffuse_guided(&condition, &context_latents, &noise, steps, guidance)?;
    let latents_t = to_cpu_f32(&latents)?.transpose(1, 2)?.contiguous()?; // [1, T, 64]
    println!("=== Flow-Matching ({} steps, guidance={}, T={}) ===", steps, guidance, t);
    println!("  latents r={:?} ref={:?} max|diff|={:.3e}", latents_t.shape(), ref_latents.shape(), max_abs_diff(&latents_t, &ref_latents)?);
    println!("  ref  mean={:.6}", ref_latents.mean_all()?.to_scalar::<f32>()?);
    println!("  rust mean={:.6}", latents_t.mean_all()?.to_scalar::<f32>()?);

    let audio = pipeline.decode(&latents)?;
    std::fs::create_dir_all("outputs/audio_showcase")?;
    let out = "outputs/audio_showcase/acestep_iso_fr.wav";
    audio.save_auto(out)?;
    println!("  saved {}", out);

    Ok(())
}
