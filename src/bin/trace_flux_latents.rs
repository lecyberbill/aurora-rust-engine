// [WFGY] Zone: SAFE | λ: 0.15 | Fallbacks: 0 | Action: Trace and diagnose exact latent values and VAE input before decode

use candle_core::{DType, Device, Tensor};
use aurora_rust_engine::traits::DiffusionParams;
use aurora_rust_engine::pipelines::FluxPipeline;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    println!("🔍 Tracing Flux.1 Latents and Velocity...");

    let device = Device::new_cuda(0)?;
    let checkpoint = "G:\\models\\flux\\flux1SchnellFp8_schnellFp8.safetensors";

    if !Path::new(checkpoint).exists() {
        println!("[-] Checkpoint not found: {}", checkpoint);
        return Ok(());
    }

    let mut pipeline = FluxPipeline::from_single_file_streaming(checkpoint, device.clone())?;

    let params = DiffusionParams {
        prompt: "a magnificent cyberpunk cyber-cat with glowing blue neon visor",
        negative_prompt: None,
        num_steps: 4,
        guidance_scale: 1.0,
        width: 512,
        height: 512,
        seed: 42,
    };

    let cb = |step: usize, total: usize, latents: &Tensor| {
        if let Ok(vec) = latents.flatten_all().and_then(|t| t.to_dtype(DType::F32)).and_then(|t| t.to_device(&Device::Cpu)).and_then(|t| t.to_vec1::<f32>()) {
            let min_v = vec.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_v = vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mean_v = vec.iter().sum::<f32>() / vec.len() as f32;
            let std_v = (vec.iter().map(|&x| (x - mean_v).powi(2)).sum::<f32>() / vec.len() as f32).sqrt();
            println!("   [Step {}/{}] Latents Stats: Min = {:.4}, Max = {:.4}, Mean = {:.4}, Std = {:.4}", step, total, min_v, max_v, mean_v, std_v);
        }
    };

    let (image, metrics) = pipeline.generate_with_metrics(params, Some(cb))?;
    println!("✨ Generation Complete in {:.2}s", metrics.total_wallclock_ms / 1000.0);

    let raw = image.as_raw();
    let min_p = raw.iter().min().copied().unwrap_or(0);
    let max_p = raw.iter().max().copied().unwrap_or(0);
    let mean_p = raw.iter().map(|&x| x as f64).sum::<f64>() / raw.len() as f64;
    println!("🎨 Final Image Pixel Stats: Min = {}, Max = {}, Mean = {:.2}", min_p, max_p, mean_p);

    image.save("outputs/flux_showcase/flux_trace_diag.png")?;
    println!("💾 Saved image to outputs/flux_showcase/flux_trace_diag.png");

    Ok(())
}
