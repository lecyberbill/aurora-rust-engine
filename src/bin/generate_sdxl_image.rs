// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Generate high quality test image via SDXL and save to outputs/ directory

use candle_core::Device;
use aurora_rust_engine::traits::DiffusionParams;
use aurora_rust_engine::pipelines::StableDiffusionXLPipeline;
use std::fs;
use std::path::Path;
use std::time::Instant;

fn progress_callback(step: usize, total: usize, _latents: &candle_core::Tensor) {
    println!("   [Step {:02}/{:02}] Denoising Latents...", step, total);
}

fn main() -> anyhow::Result<()> {
    println!("================================================================================");
    println!("🎨 AURORA PURE RUST SDXL IMAGE GENERATION & QUALITY VALIDATION");
    println!("   • Architecture: Stable Diffusion XL (UNet + CLIP-L + OpenCLIP-G + Tiled VAE)");
    println!("   • Scheduler:    DPM-Solver++ 2M Karras (18 steps)");
    println!("   • Resolution:   1024 x 1024 (High-Fidelity Photorealism)");
    println!("================================================================================\n");

    let output_dir = "outputs/showcase";
    fs::create_dir_all(output_dir)?;

    let checkpoint_path = "G:\\models\\checkpoints\\Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors";
    if !Path::new(checkpoint_path).exists() {
        eprintln!("[-] Model checkpoint not found at: {}", checkpoint_path);
        return Ok(());
    }

    let device = Device::new_cuda(0)?;

    println!("📥 Loading SDXL Model Checkpoint: {}", checkpoint_path);
    let t_load = Instant::now();
    let mut pipeline = StableDiffusionXLPipeline::from_single_file(checkpoint_path, device)?;
    println!("✅ SDXL Pipeline loaded in {:.2}s\n", t_load.elapsed().as_secs_f64());

    // Switch to DPM-Solver++ 2M Karras scheduler
    pipeline.use_dpm_solver();

    let params = DiffusionParams {
        prompt: "a magnificent cyberpunk cyber-cat with glowing blue neon cybernetic visor, standing on a rainy neo-tokyo street, hyper-realistic, photorealistic, octane render, 8k resolution, cinematic lighting",
        negative_prompt: Some("blurry, low quality, distorted, bad anatomy, deformed, watermark, signature"),
        num_steps: 18,
        guidance_scale: 7.0,
        width: 1024,
        height: 1024,
        seed: 777,
    };

    println!("⚡ Executing DPM-Solver++ Generation...");
    let (image, metrics) = pipeline.generate_with_metrics(params, Some(progress_callback))?;

    let file_name = format!("{}/cyber_cat_1024_seed777.png", output_dir);
    image.save(&file_name)?;

    println!("\n================================================================================");
    println!("🎉 Image Generated & Saved Successfully!");
    println!("📁 Output Path: {}", file_name);
    println!("📊 Generation Telemetry:");
    println!("   • Text Encoding:      {:.2} ms", metrics.prompt_encode_ms);
    println!("   • UNet Denoising:     {:.2} s ({} steps @ {:.2} it/s)", metrics.unet_total_ms / 1000.0, metrics.unet_steps, metrics.unet_it_per_sec);
    println!("   • VAE Decode:         {:.2} s", metrics.vae_decode_ms / 1000.0);
    println!("   • Total Wall-Clock:   {:.2} s", metrics.total_wallclock_ms / 1000.0);
    println!("================================================================================\n");

    Ok(())
}
