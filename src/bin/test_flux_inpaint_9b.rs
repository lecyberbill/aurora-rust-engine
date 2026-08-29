use candle_core::{DType, Device, Result};
use std::path::{Path, PathBuf};
use aurora_rust_engine::pipelines::FluxPipeline;
use aurora_rust_engine::traits::InpaintParams;
use aurora_rust_engine::weights::{SafeTensorsArchive, WeightRouter};
use aurora_rust_engine::diffusion::vae_flux::{FluxVaeDecoder, FluxVaeEncoder};

fn main() -> Result<()> {
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let checkpoint = std::env::var("CKPT").unwrap_or_else(|_| "G:\\models\\flux\\flux-2-klein-9b.safetensors".into());
    let vae_path = "G:\\models\\vae\\flux2-vae.safetensors";
    let qwen_dir = std::env::var("QWEN8").unwrap_or_else(|_| "G:\\models\\clip\\FLUX.2-klein-9B_text_encoder".into());
    let input_path = std::env::var("IN").unwrap_or_else(|_| "outputs/flux_showcase/flux_klein_9b_1024_seed42.png".into());
    let strength: f64 = std::env::var("STRENGTH").ok().and_then(|s| s.parse().ok()).unwrap_or(0.85);
    let steps: usize = std::env::var("STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
    let prompt = std::env::var("PROMPT").unwrap_or_else(|_| "a majestic lion wearing an intricate glowing golden crown with emerald gems, photorealistic, 8k".into());

    if !Path::new(&input_path).exists() {
        println!("[-] Input image not found: {}", input_path);
        return Ok(());
    }
    let input_img = image::open(&input_path)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to open base image: {}", e)))?
        .to_rgb8();
    let (w, h) = input_img.dimensions();

    // Circular inpainting mask (center upper-third, radius = 22% of width)
    let mut mask = image::GrayImage::new(w, h);
    let center_x = (w / 2) as i32;
    let center_y = (h / 3) as i32;
    let radius = ((w as f32) * 0.22) as i32;
    for y in 0..h {
        for x in 0..w {
            let dx = x as i32 - center_x;
            let dy = y as i32 - center_y;
            mask.put_pixel(x, y, image::Luma([if dx*dx + dy*dy <= radius*radius { 255u8 } else { 0u8 }]));
        }
    }
    mask.save("outputs/flux_showcase/flux_klein_9b_inpaint_mask.png")
        .map_err(|e| candle_core::Error::Msg(format!("Failed to save mask: {}", e)))?;
    println!("✅ Saved mask (radius {}): outputs/flux_showcase/flux_klein_9b_inpaint_mask.png", radius);

    println!("\n📥 Loading Flux.2-Klein-9B for Inpainting: {}", checkpoint);
    let mut pipeline = FluxPipeline::from_single_file_streaming(&checkpoint, device.clone())
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    pipeline.enable_flash_attn();

    let qwen_archive = SafeTensorsArchive::open_shards_dir(&qwen_dir)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let qwen = aurora_rust_engine::text::Qwen3TextEncoder::from_archive(
        &qwen_archive,
        Some(std::path::Path::new("qwen_tokenizer.json")),
        &Device::Cpu,
        DType::F16,
    ).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    pipeline.set_qwen3(qwen);

    let vae_archive = SafeTensorsArchive::open(PathBuf::from(vae_path))
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let vae_router = WeightRouter::new(&vae_archive, device.clone(), DType::F16);
    let vae_vb = vae_router.vae_var_builder()
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let decoder = FluxVaeDecoder::new(vae_vb.clone())?;
    let encoder = FluxVaeEncoder::new(vae_vb)?;
    pipeline.set_vae(decoder);
    pipeline.set_vae_encoder(encoder);

    println!("\n🎨 Executing Inpainting (Strength: {}, {} Steps)...", strength, steps);
    println!("📝 Prompt: \"{}\"", prompt);

    let params = InpaintParams {
        prompt: &prompt,
        negative_prompt: None,
        image: input_img,
        mask,
        mask_blur: 0,
        strength,
        num_steps: steps,
        guidance_scale: 1.0,
        seed: 42,
    };

    let (result_img, metrics) = pipeline.generate_inpaint(
        params,
        None::<fn(usize, usize, &candle_core::Tensor)>,
    ).map_err(|e| candle_core::Error::Msg(e.to_string()))?;

    let out_dir = "outputs/flux_showcase";
    std::fs::create_dir_all(out_dir).ok();
    let out_path = format!("{}/flux_klein_9b_inpaint.png", out_dir);
    result_img.save(&out_path)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to save output image: {}", e)))?;

    println!("🎉 Flux.2-Klein-9B Inpaint Saved: {} ({} steps)", out_path, metrics.unet_steps);
    Ok(())
}
