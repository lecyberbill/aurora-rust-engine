// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Representative Multi-Run Statistical Benchmark

use candle_core::Device;
use aurora_rust_engine::{DiffusionParams, StableDiffusionXLPipeline};
use std::fs;
use std::io::Write;
use std::path::Path;

fn progress_cb(step: usize, total: usize, _latents: &candle_core::Tensor) {
    if step == 1 || step % 10 == 0 || step == total {
        print!("{} ", step);
        let _ = std::io::stdout().flush();
    }
}

fn main() -> anyhow::Result<()> {
    let out_dir = Path::new("outputs/series_benchmark");
    fs::create_dir_all(out_dir)?;

    let device = Device::new_cuda(0)?;
    println!("============================================================");
    println!("📊 Representative Multi-Run Statistical Benchmark (5 Images)");
    println!("   Hardware: NVIDIA RTX 4070 Ti 12GB | Steps: 30 | Resolution: 1024x1024");
    println!("============================================================\n");

    let checkpoint_path = "G:\\models\\checkpoints\\duchaitenPonyXLNo_v60.safetensors";
    println!("📦 Loading base SDXL checkpoint: {}", checkpoint_path.replace('\\', "/"));
    let mut pipeline = StableDiffusionXLPipeline::from_single_file(checkpoint_path, device)?;

    let prompt = "masterpiece, best quality, ultra-detailed, 1girl, solo, cyberpunk samurai warrior, glowing katana, neon city, rain, reflections";
    let neg_prompt = "lowres, bad anatomy, bad hands, text, blurry, worst quality";

    let seeds = [42, 101, 202, 303, 404];
    let mut unet_speeds: Vec<f64> = Vec::new();
    let mut total_times: Vec<f64> = Vec::new();

    for (run_idx, &seed) in seeds.iter().enumerate() {
        println!("\n------------------------------------------------------------");
        println!("🚀 Run {}/5 [Seed: {}]:", run_idx + 1, seed);

        let params = DiffusionParams {
            prompt,
            negative_prompt: Some(neg_prompt),
            num_steps: 30,
            guidance_scale: 6.0,
            width: 1024,
            height: 1024,
            seed,
        };

        print!("  🎨 Denoising (30 steps): ");
        let _ = std::io::stdout().flush();
        let (image, metrics) = pipeline.generate_with_metrics(params, Some(progress_cb))?;
        println!("done.");

        let out_filename = format!("run_{:02}_seed{}.png", run_idx + 1, seed);
        let out_path = out_dir.join(&out_filename);
        image.save(&out_path)?;

        unet_speeds.push(metrics.unet_it_per_sec);
        total_times.push(metrics.total_wallclock_ms / 1000.0);

        println!("  📊 Run Telemetry:");
        println!("     • Text Encoders:  {:.2} ms ({:.2}s)", metrics.prompt_encode_ms, metrics.prompt_encode_ms / 1000.0);
        println!("     • UNet Denoising: {:.2}s -> {:.2} it/s ({:.2} ms/step)", metrics.unet_total_ms / 1000.0, metrics.unet_it_per_sec, metrics.unet_step_avg_ms);
        println!("     • VAE Decode:     {:.2}s", metrics.vae_decode_ms / 1000.0);
        println!("     • Wall-Clock:     {:.2}s", metrics.total_wallclock_ms / 1000.0);
        println!("     • Saved Image:    {}", out_path.to_string_lossy().replace('\\', "/"));
    }

    let avg_unet_speed = unet_speeds.iter().sum::<f64>() / (unet_speeds.len() as f64);
    let avg_total_time = total_times.iter().sum::<f64>() / (total_times.len() as f64);
    let warm_avg_total_time = total_times[1..].iter().sum::<f64>() / ((total_times.len() - 1) as f64);

    println!("\n============================================================");
    println!("🏆 Multi-Run Statistical Synthesis (5 Consecutive Runs):");
    println!("   • Average UNet Speed:         {:.2} it/s", avg_unet_speed);
    println!("   • Cold-Start Total Time (Run 1): {:.2}s", total_times[0]);
    println!("   • Warm-State Total Time (Avg 2-5): {:.2}s (Cache Active: ~0.00ms text encode)", warm_avg_total_time);
    println!("   • Overall 5-Run Average Time:   {:.2}s", avg_total_time);
    println!("============================================================");

    Ok(())
}
