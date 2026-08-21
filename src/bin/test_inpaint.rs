// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Inpainting & Outpainting Mask-Guided Diffusion Test Binary

use candle_core::{Device, Tensor};
use aurora_rust_engine::{InpaintParams, StableDiffusionXLPipeline, TextToImagePipeline};
use image::{GrayImage, Luma};
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
    let output_dir = Path::new("outputs/inpaint_test");
    fs::create_dir_all(output_dir)?;

    let device = Device::new_cuda(0)?;
    println!("🚀 CUDA Device (RTX 4070 Ti) initialized.");

    let checkpoint_path = "G:\\models\\checkpoints\\duchaitenPonyXLNo_v60.safetensors";
    let source_image_path = "outputs/cat_manga_transformation/00_real_photo_cat.png";

    println!("📦 Loading base checkpoint: {}", checkpoint_path.replace('\\', "/"));
    let t_load = Instant::now();
    let mut pipeline = StableDiffusionXLPipeline::from_safetensors(checkpoint_path, &device)?;
    println!("✅ Checkpoint loaded in {:.2}s", t_load.elapsed().as_secs_f64());

    println!("🖼️ Loading source image: {}", source_image_path);
    let src_img = image::open(source_image_path)?.to_rgb8();
    let (w, h) = src_img.dimensions();
    println!("  ✅ Source image dimensions: {}x{}", w, h);

    // Save source copy
    let src_copy_path = output_dir.join("00_source_image.png");
    src_img.save(&src_copy_path)?;

    // 1. Create Inpainting Binary Mask for the cat's head & ears region
    // The cat head in 00_real_photo_cat.png is located approximately in X: [420..760], Y: [220..520]
    let mut mask_img = GrayImage::from_pixel(w, h, Luma([0u8]));
    let mask_x_start = (w as f32 * 0.40) as u32;
    let mask_x_end = (w as f32 * 0.76) as u32;
    let mask_y_start = (h as f32 * 0.20) as u32;
    let mask_y_end = (h as f32 * 0.52) as u32;

    for y in mask_y_start..mask_y_end {
        for x in mask_x_start..mask_x_end {
            mask_img.put_pixel(x, y, Luma([255u8]));
        }
    }

    let mask_out_path = output_dir.join("01_inpaint_mask.png");
    mask_img.save(&mask_out_path)?;
    println!("🎭 Inpainting mask generated -> {}", mask_out_path.to_string_lossy().replace('\\', "/"));

    // 2. Inpaint prompt: Add a wizard hat with gold stars on the cat
    let inpaint_prompt = "photo of a cute real orange cat wearing a miniature pointed purple wizard hat with shiny golden stars, natural lighting, sharp focus, 8k photography, realistic texture";
    let inpaint_neg = "blurry, lowres, cartoon, painting, 3d render, distorted";

    println!("\n🎨 Running Inpainting on masked region...");
    let inpaint_params = InpaintParams {
        prompt: inpaint_prompt,
        negative_prompt: Some(inpaint_neg),
        image: src_img.clone(),
        mask: mask_img,
        mask_blur: 8,
        strength: 0.95,
        num_steps: 30,
        guidance_scale: 7.0,
        seed: 888,
    };

    let t_gen = Instant::now();
    let inpaint_result = pipeline.generate_inpaint(inpaint_params, Some(progress_callback))?;
    let duration = t_gen.elapsed().as_secs_f64();

    let result_out = output_dir.join("02_inpaint_result.png");
    inpaint_result.save(&result_out)?;
    println!("  ✨ Inpainting result saved in {:.2}s -> {}", duration, result_out.to_string_lossy().replace('\\', "/"));

    println!("\n============================================================");
    println!("🎉 Inpainting Test Benchmark Complete!");
    println!("   Total Generation Time: {:.2}s", duration);
    println!("   Output: {}", result_out.to_string_lossy().replace('\\', "/"));
    println!("============================================================");

    Ok(())
}
