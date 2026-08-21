// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: High-Resolution Disentangled Metrics & Fast VAE Telemetry Benchmark

use candle_core::Device;
use aurora_rust_engine::{DiffusionParams, StableDiffusionXLPipeline};
use std::fs;
use std::io::Write;
use std::path::Path;

fn progress_callback(step: usize, total: usize, _latent: &candle_core::Tensor) {
    if step == 1 || step % 5 == 0 || step == total {
        print!("    Step {}/{} | ", step, total);
        let _ = std::io::stdout().flush();
    }
}

fn main() -> anyhow::Result<()> {
    let output_dir = Path::new("outputs/telemetry_benchmark");
    fs::create_dir_all(output_dir)?;

    let device = Device::new_cuda(0)?;
    println!("🚀 CUDA Device (RTX 4070 Ti) initialized.");

    let checkpoint_path = "G:\\models\\checkpoints\\duchaitenPonyXLNo_v60.safetensors";
    println!("📦 Loading base checkpoint: {}", checkpoint_path.replace('\\', "/"));
    let mut pipeline = StableDiffusionXLPipeline::from_single_file(checkpoint_path, device)?;

    let prompt = "score_9, score_8_up, score_7_up, masterpiece, 1girl, solo, futuristic neon cyberpunk pilot, neon visor, dynamic pose, highly detailed";
    let neg_prompt = "score_4, score_5, score_6, lowres, bad anatomy, text, blurry";

    let params = DiffusionParams {
        prompt,
        negative_prompt: Some(neg_prompt),
        num_steps: 30,
        guidance_scale: 6.5,
        width: 1024,
        height: 1024,
        seed: 42,
    };

    println!("\n📊 Running Denoising with High-Resolution Profiler (30 steps)...");
    let (image, metrics) = pipeline.generate_with_metrics(params, Some(progress_callback))?;

    let out_path = output_dir.join("telemetry_gen.png");
    image.save(&out_path)?;

    println!("\n============================================================");
    println!("🎉 Milestone 7 Disentangled Profiler Results:");
    println!("   • Text Encoding:      {:.2} ms ({:.2}s)", metrics.prompt_encode_ms, metrics.prompt_encode_ms / 1000.0);
    println!("   • UNet Denoising:     {:.2} ms ({:.2}s) -> {:.2} it/s ({:.2} ms/step)", metrics.unet_total_ms, metrics.unet_total_ms / 1000.0, metrics.unet_it_per_sec, metrics.unet_step_avg_ms);
    println!("   • VAE Adaptive Decode:{:.2} ms ({:.2}s)", metrics.vae_decode_ms, metrics.vae_decode_ms / 1000.0);
    println!("   • Total Wall-Clock:   {:.2} ms ({:.2}s)", metrics.total_wallclock_ms, metrics.total_wallclock_ms / 1000.0);
    println!("   • Saved Image:        {}", out_path.to_string_lossy().replace('\\', "/"));
    println!("============================================================");

    Ok(())
}
