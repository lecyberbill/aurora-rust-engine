// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Photorealistic Cat to Anime/Manga Img2Img Transformation Benchmark

use candle_core::{Device, Tensor};
use aurora_rust_engine::{DiffusionParams, Img2ImgParams, StableDiffusionXLPipeline, TextToImagePipeline};
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
    let output_dir = Path::new("outputs/cat_manga_transformation");
    fs::create_dir_all(output_dir)?;

    let device = Device::new_cuda(0)?;
    println!("🚀 CUDA Device (RTX 4070 Ti) initialized.");

    let checkpoint_path = "G:\\models\\checkpoints\\duchaitenPonyXLNo_v60.safetensors";
    println!("📦 Loading base checkpoint: {}", checkpoint_path.replace('\\', "/"));
    let mut pipeline = StableDiffusionXLPipeline::from_safetensors(checkpoint_path, &device)?;

    // 1. Generate Photorealistic Source Cat
    println!("\n📸 [1/4] Generating Photorealistic Source Cat...");
    let photo_prompt = "photo of a real cute orange tabby cat sitting on a cozy living room rug, natural lighting, dslr photo, high detail, sharp focus, 8k photography, realistic fur texture";
    let photo_params = DiffusionParams {
        prompt: photo_prompt,
        negative_prompt: Some("anime, cartoon, drawing, painting, 3d render, illustration, low quality"),
        num_steps: 25,
        guidance_scale: 6.0,
        width: 1024,
        height: 1024,
        seed: 777,
    };

    let t_photo = Instant::now();
    let src_cat_img = pipeline.generate(photo_params, Some(progress_callback))?;
    let photo_out = output_dir.join("00_real_photo_cat.png");
    src_cat_img.save(&photo_out)?;
    println!("  ✅ Source photo generated in {:.2}s -> {}", t_photo.elapsed().as_secs_f64(), photo_out.to_string_lossy().replace('\\', "/"));

    // 2. Transform to Anime/Manga Style with varying strengths
    let manga_prompt = "score_9, score_8_up, score_7_up, masterpiece, anime style, colorful manga illustration of a cute cat, big expressive sparkling anime eyes, vibrant studio ghibli aesthetic, clean lineart, cell shaded, highly detailed anime art";
    let manga_neg = "photo, realistic, 3d render, photograph, low quality, bad anatomy";

    let strengths = [0.40f64, 0.65f64, 0.85f64];
    let mut results = Vec::new();

    for (idx, &strength) in strengths.iter().enumerate() {
        let out_filename = format!("{:02}_manga_cat_strength_{:02}.png", idx + 1, (strength * 100.0) as u32);
        let out_path = output_dir.join(&out_filename);

        println!("\n🎨 [{}/{}] Transforming Real Cat to Manga (Strength: {:.2})...", idx + 2, strengths.len() + 1, strength);

        let img2img_params = Img2ImgParams {
            prompt: manga_prompt,
            negative_prompt: Some(manga_neg),
            image: src_cat_img.clone(),
            strength,
            num_steps: 30,
            guidance_scale: 7.0,
            seed: 777,
        };

        let t_gen = Instant::now();
        let manga_img = pipeline.generate_img2img(img2img_params, Some(progress_callback))?;
        let duration = t_gen.elapsed().as_secs_f64();
        manga_img.save(&out_path)?;

        let num_actual_steps = ((30.0 * strength).round() as usize).max(1).min(30);
        let it_per_sec = num_actual_steps as f64 / duration;

        println!("  ✨ Saved: {} (Duration: {:.2}s, Speed: {:.2} it/s)", out_path.to_string_lossy().replace('\\', "/"), duration, it_per_sec);
        results.push((strength, duration, it_per_sec, out_filename));
    }

    println!("\n============================================================");
    println!("🎉 Real-to-Manga Cat Transformation Complete!");
    for (strength, dur, speed, file) in &results {
        println!("   Strength {:.2} -> Duration: {:.2}s ({:.2} it/s) | File: {}", strength, dur, speed, file);
    }
    println!("============================================================");

    Ok(())
}
