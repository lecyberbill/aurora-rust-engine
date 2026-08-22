// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: DPM-Solver++ 2M Karras Ultra-Fast 18-Step SDXL Benchmark

use candle_core::Device;
use aurora_rust_engine::{DiffusionParams, StableDiffusionXLPipeline};
use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

fn progress_cb(step: usize, total: usize, _latents: &candle_core::Tensor) {
    if step == 1 || step % 5 == 0 || step == total {
        print!("{} ", step);
        let _ = std::io::stdout().flush();
    }
}

fn main() -> anyhow::Result<()> {
    let out_dir = Path::new("outputs/dpm_solver_benchmark");
    fs::create_dir_all(out_dir)?;

    let device = Device::new_cuda(0)?;
    println!("============================================================");
    println!("🚀 Ultra-Fast DPM-Solver++ 2M Karras (18 Steps) Benchmark");
    println!("   Target Checkpoint: Juggernaut-XL_v9_RunDiffusionPhoto_v2");
    println!("   Hardware: NVIDIA RTX 4070 Ti 12GB | Resolution: 1024x1024");
    println!("============================================================\n");

    let checkpoint_path = "G:\\models\\checkpoints\\Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors";
    if !Path::new(checkpoint_path).exists() {
        eprintln!("[-] Checkpoint not found: {}", checkpoint_path);
        return Ok(());
    }

    let t_load = Instant::now();
    let mut pipeline = StableDiffusionXLPipeline::from_single_file(checkpoint_path, device)?;
    println!("✅ Pipeline loaded in {:.2}s", t_load.elapsed().as_secs_f64());

    // Switch to 2nd-order DPM-Solver++ Multistep Karras Scheduler
    pipeline.use_dpm_solver();
    println!("⚡ Switched to DPM-Solver++ 2M Karras Multistep Scheduler");

    let prompt = "masterpiece, best quality, ultra-detailed, 1girl, solo, cyberpunk samurai warrior, glowing cherry blossoms, rainy futuristic neo-tokyo street, volumetric lighting, intricate reflections, 8k resolution, cinematic masterpiece";
    let negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, blurry";

    let params = DiffusionParams {
        prompt,
        negative_prompt: Some(negative_prompt),
        num_steps: 18, // 18 steps with DPM++ 2M converges to same/better quality than 30 Euler steps
        guidance_scale: 6.5,
        width: 1024,
        height: 1024,
        seed: 42,
    };

    println!("\n🎨 Running 18-Step DPM-Solver++ Denoising...");
    print!("   Steps: ");
    let _ = std::io::stdout().flush();
    let (image, metrics) = pipeline.generate_with_metrics(params, Some(progress_cb))?;
    println!("done.");

    let out_path = out_dir.join("juggernaut_dpm_solver_18steps_seed42.png");
    image.save(&out_path)?;

    println!("\n============================================================");
    println!("📊 DPM-Solver++ 2M (18 Steps) Telemetry Results:");
    println!("   • Text Encoders:  {:.2}s", metrics.prompt_encode_ms / 1000.0);
    println!("   • UNet Denoising: {:.2}s -> {:.2} it/s ({:.2} ms/step)",
        metrics.unet_total_ms / 1000.0, metrics.unet_it_per_sec, metrics.unet_step_avg_ms);
    println!("   • VAE Decode:     {:.2}s", metrics.vae_decode_ms / 1000.0);
    println!("   • Total Wall-Clock:{:.2}s", metrics.total_wallclock_ms / 1000.0);
    println!("   • Saved Image:    {}", out_path.to_string_lossy().replace('\\', "/"));
    println!("============================================================");

    Ok(())
}
