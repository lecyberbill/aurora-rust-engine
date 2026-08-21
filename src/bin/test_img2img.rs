// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Image-to-Image (Img2Img) Multi-Strength Pipeline Test Binary

use candle_core::{Device, Tensor};
use aurora_rust_engine::{Img2ImgParams, StableDiffusionXLPipeline, TextToImagePipeline};
use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

fn progress_callback(step: usize, total: usize, _latent: &Tensor) {
    if step == 1 || step % 5 == 0 || step == total {
        println!("    Step {}/{}", step, total);
        let _ = std::io::stdout().flush();
    }
}

fn main() -> anyhow::Result<()> {
    let output_dir = Path::new("outputs/img2img_test");
    fs::create_dir_all(output_dir)?;

    let device = Device::new_cuda(0)?;
    println!("🚀 CUDA Device (RTX 4070 Ti) initialized.");

    let checkpoint_path = "G:\\models\\checkpoints\\duchaitenPonyXLNo_v60.safetensors";
    let source_image_path = "outputs/lora_test/01_baseline_no_lora.png";

    println!("📦 Loading base checkpoint: {}", checkpoint_path.replace('\\', "/"));
    let t_load = Instant::now();
    let mut pipeline = StableDiffusionXLPipeline::from_safetensors(checkpoint_path, &device)?;
    println!("✅ Checkpoint loaded in {:.2}s", t_load.elapsed().as_secs_f64());

    println!("🖼️ Loading source image: {}", source_image_path);
    let src_img = image::open(source_image_path)?.to_rgb8();
    let src_copy_path = output_dir.join("00_source_input.png");
    src_img.save(&src_copy_path)?;
    println!("  ✅ Source image dimensions: {}x{}", src_img.width(), src_img.height());

    let prompt = "score_9, score_8_up, score_7_up, masterpiece, 1girl, solo, golden radiant armor, fiery glowing orange hair, sunset cyberpunk city, ultra-detailed, cinematic lighting";
    let neg_prompt = "score_4, score_5, score_6, lowres, bad anatomy, bad hands, text, blurry";

    let strengths = [0.35f64, 0.60f64, 0.85f64];
    let mut results = Vec::new();

    for (idx, &strength) in strengths.iter().enumerate() {
        let out_filename = format!("{:02}_img2img_strength_{:02}.png", idx + 1, (strength * 100.0) as u32);
        let out_path = output_dir.join(&out_filename);

        println!("\n🎨 [{}/{}] Running Img2Img with Denoising Strength: {:.2} (Target: {})...", idx + 1, strengths.len(), strength, out_filename);

        let params = Img2ImgParams {
            prompt,
            negative_prompt: Some(neg_prompt),
            image: src_img.clone(),
            strength,
            num_steps: 30,
            guidance_scale: 6.5,
            seed: 42,
        };

        let t_gen = Instant::now();
        let result_img = pipeline.generate_img2img(params, Some(progress_callback))?;
        let duration = t_gen.elapsed().as_secs_f64();

        result_img.save(&out_path)?;
        let num_actual_steps = ((30.0 * strength).round() as usize).max(1).min(30);
        let it_per_sec = num_actual_steps as f64 / duration;

        println!("  ✨ Saved: {} (Duration: {:.2}s, Speed: {:.2} it/s)", out_path.to_string_lossy().replace('\\', "/"), duration, it_per_sec);
        results.push((strength, duration, it_per_sec, out_filename));
    }

    println!("\n============================================================");
    println!("🎉 Img2Img Multi-Strength Benchmark Complete!");
    for (strength, dur, speed, file) in &results {
        println!("   Strength {:.2} -> Duration: {:.2}s ({:.2} it/s) | File: {}", strength, dur, speed, file);
    }
    println!("============================================================");

    Ok(())
}
