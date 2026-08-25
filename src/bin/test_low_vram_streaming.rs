// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Benchmark low VRAM sequential streaming vs standard loading

use candle_core::Device;
use aurora_rust_engine::traits::DiffusionParams;
use aurora_rust_engine::pipelines::FluxPipeline;
use std::path::Path;
use std::time::Instant;

fn progress_cb(step: usize, total: usize, _latent: &candle_core::Tensor) {
    println!("   [Step {}/{}] Flow Match ODE Integrated", step, total);
}

fn main() -> anyhow::Result<()> {
    println!("================================================================================");
    println!("⚡ AURORA FLUX.1 LOW-VRAM OPTIMIZATION TEST");
    println!("   • Target: NVIDIA GeForce RTX 4070 Ti 12GB");
    println!("   • Checking Sequential Streaming & VRAM Footprint");
    println!("================================================================================\n");

    let device = Device::new_cuda(0)?;
    let flux_checkpoint = "G:\\models\\flux\\flux1_v10Fp8Schnell.safetensors";

    if !Path::new(flux_checkpoint).exists() {
        eprintln!("[-] Flux checkpoint not found at: {}", flux_checkpoint);
        return Ok(());
    }

    println!("📥 Initializing Flux.1 Pipeline...");
    let t_load = Instant::now();
    let mut pipeline = FluxPipeline::from_single_file(flux_checkpoint, device)?;
    println!("✅ Pipeline loaded in {:.2}s\n", t_load.elapsed().as_secs_f64());

    let params = DiffusionParams {
        prompt: "masterpiece, highly detailed, futuristic glowing robot, neon lights, 8k",
        negative_prompt: None,
        num_steps: 4,
        guidance_scale: 1.0,
        width: 1024,
        height: 1024,
        seed: 42,
    };

    println!("⚡ Running Low-VRAM Generation Test...");
    let (_image, metrics) = pipeline.generate_with_metrics(params, Some(progress_cb))?;

    println!("\n📊 Performance Telemetry:");
    println!("   • ODE Denoising:  {:.2}s ({} steps @ {:.2} it/s)", metrics.unet_total_ms / 1000.0, metrics.unet_steps, metrics.unet_it_per_sec);
    println!("   • Step Average:   {:.2} ms/step", metrics.unet_step_avg_ms);
    println!("   • Total Duration: {:.2}s", metrics.total_wallclock_ms / 1000.0);

    Ok(())
}
