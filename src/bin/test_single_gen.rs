// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Single image test with Danbooru tag prompt for anime SDXL model

use candle_core::{Device, Tensor};
use aurora_rust_engine::{DiffusionParams, StableDiffusionXLPipeline, TextToImagePipeline};
use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

fn progress_callback(step: usize, total: usize, _latent: &Tensor) {
    println!("    Step {}/{}", step, total);
    let _ = std::io::stdout().flush();
}

fn main() -> anyhow::Result<()> {
    let output_dir = Path::new("outputs");
    fs::create_dir_all(output_dir)?;

    let device = Device::new_cuda(0)?;
    println!("🚀 CUDA Device (RTX 4070 Ti) initialized.");

    let model_path = "G:\\models\\checkpoints\\duchaitenPonyXLNo_v60.safetensors";
    println!("📦 Loading model: {}", model_path);
    let t_load = Instant::now();
    let mut pipeline = StableDiffusionXLPipeline::from_safetensors(model_path, &device)?;
    println!("✅ Model loaded in {:.2}s", t_load.elapsed().as_secs_f64());

    let prompt = "masterpiece, best quality, ultra-detailed, 1girl, solo, cyberpunk samurai warrior, glowing cherry blossoms, rainy futuristic neo-tokyo street, volumetric lighting, intricate reflections, 8k resolution, cinematic masterpiece";
    let negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry";

    let params = DiffusionParams {
        prompt,
        negative_prompt: Some(negative_prompt),
        num_steps: 30,
        guidance_scale: 6.0,
        width: 1024,
        height: 1024,
        seed: 42,
    };

    println!("🎨 Generating 1024x1024 image (30 steps, seed 42)...");
    let t_gen = Instant::now();
    let img = pipeline.generate(params, Some(progress_callback))?;

    let duration = t_gen.elapsed().as_secs_f64();
    let out_path = output_dir.join("test_pony_calibrated.png");
    img.save(&out_path)?;

    println!("✨ Image saved successfully: {} (Time: {:.2}s, {:.2} it/s)", out_path.display(), duration, 30.0 / duration);
    Ok(())
}
