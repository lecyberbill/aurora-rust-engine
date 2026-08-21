use candle_core::Device;
use aurora_rust_engine::{DiffusionParams, StableDiffusionXLPipeline};
use std::fs;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let out_dir = Path::new("outputs/cache_optimization");
    fs::create_dir_all(out_dir)?;

    let device = Device::new_cuda(0)?;
    let checkpoint_path = "G:\\models\\checkpoints\\duchaitenPonyXLNo_v60.safetensors";
    println!("📦 Loading base SDXL checkpoint: {}", checkpoint_path.replace('\\', "/"));
    let mut pipeline = StableDiffusionXLPipeline::from_single_file(checkpoint_path, device)?;

    let prompt = "masterpiece, best quality, 1girl, solo, futuristic neon cyberpunk warrior, glowing katana, intricate reflections";
    let neg_prompt = "lowres, bad anatomy, bad hands, text, blurry";

    println!("\n============================================================");
    println!("🚀 Pass 1: Initial Cold Generation (Encoding Text Encoders)...");
    let params1 = DiffusionParams {
        prompt,
        negative_prompt: Some(neg_prompt),
        num_steps: 30,
        guidance_scale: 6.0,
        width: 1024,
        height: 1024,
        seed: 42,
    };
    let (img1, metrics1) = pipeline.generate_with_metrics(params1, None::<fn(usize, usize, &candle_core::Tensor)>)?;
    let out1 = out_dir.join("01_cold_generation.png");
    img1.save(&out1)?;

    println!("\n============================================================");
    println!("⚡ Pass 2: Consecutive Generation with Zero-Latency Prompt Cache...");
    let params2 = DiffusionParams {
        prompt,
        negative_prompt: Some(neg_prompt),
        num_steps: 30,
        guidance_scale: 6.0,
        width: 1024,
        height: 1024,
        seed: 1337,
    };
    let (img2, metrics2) = pipeline.generate_with_metrics(params2, None::<fn(usize, usize, &candle_core::Tensor)>)?;
    let out2 = out_dir.join("02_cached_generation.png");
    img2.save(&out2)?;

    println!("\n============================================================");
    println!("🎉 Prompt Cache Optimization Results:");
    println!("   • Pass 1 Text Encode: {:.2} ms ({:.2}s) -> Total: {:.2}s", metrics1.prompt_encode_ms, metrics1.prompt_encode_ms / 1000.0, metrics1.total_wallclock_ms / 1000.0);
    println!("   • Pass 2 Text Encode: {:.2} ms ({:.4}s) -> Total: {:.2}s", metrics2.prompt_encode_ms, metrics2.prompt_encode_ms / 1000.0, metrics2.total_wallclock_ms / 1000.0);
    println!("   • Latency Reduction:  {:.2}s saved immediately per generation!", (metrics1.prompt_encode_ms - metrics2.prompt_encode_ms) / 1000.0);
    println!("============================================================");

    Ok(())
}
