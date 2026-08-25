// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Generate and save Flux.1 image to disk

use candle_core::Device;
use aurora_rust_engine::traits::DiffusionParams;
use aurora_rust_engine::pipelines::FluxPipeline;
use std::fs;
use std::path::Path;
use std::time::Instant;

fn progress_cb(step: usize, total: usize, _latent: &candle_core::Tensor) {
    println!("   [Step {}/{}] Flow Match ODE Integrated", step, total);
}

fn main() -> anyhow::Result<()> {
    println!("================================================================================");
    println!("🚀 AURORA PURE RUST FLUX.1 SCHNELL / DEV IMAGE GENERATION");
    println!("   • MMDiT Multimodal Diffusion Transformer");
    println!("   • Flow Matching Euler ODE Solver (4 Steps)");
    println!("   • 2D Unpatchify + Latent Decoded Render");
    println!("================================================================================\n");

    let output_dir = "outputs/flux_showcase";
    fs::create_dir_all(output_dir)?;

    let args: Vec<String> = std::env::args().collect();
    let flux_checkpoint = if args.len() > 1 {
        args[1].clone()
    } else {
        "G:\\models\\flux\\flux1-dev-fp8.safetensors".to_string()
    };

    let is_dev = flux_checkpoint.to_lowercase().contains("dev");
    let num_steps = if args.len() > 2 {
        args[2].parse::<usize>().unwrap_or(if is_dev { 20 } else { 4 })
    } else {
        if is_dev { 20 } else { 4 }
    };
    let guidance_scale = if is_dev { 3.5 } else { 1.0 };

    if !Path::new(&flux_checkpoint).exists() {
        eprintln!("[-] Flux checkpoint not found at: {}", flux_checkpoint);
        return Ok(());
    }

    let device = Device::new_cuda(0)?;
    println!("📥 Loading Flux.1 FP8 Checkpoint: {}", flux_checkpoint);
    let t_load = Instant::now();
    let mut pipeline = FluxPipeline::from_single_file(&flux_checkpoint, device)?;
    println!("✅ Flux.1 Pipeline loaded in {:.2}s\n", t_load.elapsed().as_secs_f64());

    let params = DiffusionParams {
        prompt: "masterpiece, highly detailed, futuristic glowing robot, neon lights, 8k",
        negative_prompt: None,
        num_steps,
        guidance_scale,
        width: 1024,
        height: 1024,
        seed: 42,
    };

    println!("⚡ Executing {}-Step Flow Matching ODE Generation (Guidance: {:.1})...", num_steps, guidance_scale);
    let (image, metrics) = pipeline.generate_with_metrics(params, Some(progress_cb))?;

    let model_tag = if is_dev { "dev" } else { "schnell" };
    let file_name = format!("{}/flux_{}_1024_seed42.png", output_dir, model_tag);
    image.save(&file_name)?;

    println!("\n================================================================================");
    println!("🎉 Flux.1 Image Generated & Saved Successfully!");
    println!("📁 Output Path: {}", file_name);
    println!("📊 Flux.1 Telemetry Metrics:");
    println!("   • ODE Denoising Duration: {:.2}s", metrics.unet_total_ms / 1000.0);
    println!("   • Throughput:             {:.2} it/s ({:.2} ms/step)", metrics.unet_it_per_sec, metrics.unet_step_avg_ms);
    println!("   • VAE & Unpatchify:       {:.2}s", metrics.vae_decode_ms / 1000.0);
    println!("   • Total Wall-Clock:       {:.2}s", metrics.total_wallclock_ms / 1000.0);
    println!("================================================================================\n");

    Ok(())
}
