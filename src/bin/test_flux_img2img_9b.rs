use candle_core::{DType, Device, Result};
use std::path::PathBuf;
use std::time::Instant;
use aurora_rust_engine::pipelines::FluxPipeline;
use aurora_rust_engine::traits::Img2ImgParams;
use aurora_rust_engine::diffusion::vae_flux::{FluxVaeDecoder, FluxVaeEncoder};
use aurora_rust_engine::weights::{SafeTensorsArchive, WeightRouter};

fn main() -> Result<()> {
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let checkpoint = std::env::var("CKPT").unwrap_or_else(|_| "G:\\models\\flux\\flux-2-klein-9b.safetensors".into());
    let vae_path = "G:\\models\\vae\\flux2-vae.safetensors";
    let qwen_dir = std::env::var("QWEN8").unwrap_or_else(|_| "G:\\models\\clip\\FLUX.2-klein-9B_text_encoder".into());
    let input_img_path = std::env::var("IN").unwrap_or_else(|_| "outputs/flux_showcase/flux_klein_9b_1024_seed42.png".into());
    let strength: f64 = std::env::var("STRENGTH").ok().and_then(|s| s.parse().ok()).unwrap_or(0.65);
    let steps: usize = std::env::var("STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
    let prompt = std::env::var("PROMPT").unwrap_or_else(|_| "a majestic lion wearing a golden crown and diamond armor sitting on a rock during sunset, photorealistic, 8k".into());

    if !std::path::Path::new(&input_img_path).exists() {
        println!("[-] Input image not found: {}", input_img_path);
        return Ok(());
    }
    let input_img = image::open(&input_img_path)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to open image: {}", e)))?
        .to_rgb8();

    println!("\n📥 Loading Flux.2-Klein-9B for Img2Img: {}", checkpoint);
    let mut pipeline = FluxPipeline::from_single_file_streaming(&checkpoint, device.clone())
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    pipeline.enable_flash_attn();

    // Qwen3-8B from shards dir (uses from_archive to auto-handle prefixed keys + layer gathering)
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

    println!("\n🎨 Executing Img2Img (Strength: {}, {} Steps)...", strength, steps);
    println!("📝 Prompt: \"{}\"", prompt);

    let params = Img2ImgParams {
        prompt: &prompt,
        image: input_img,
        strength,
        num_steps: steps,
        guidance_scale: 1.0,
        negative_prompt: None,
        seed: 42,
    };

    let t0 = Instant::now();
    let (result_img, metrics) = pipeline.generate_img2img(
        params,
        None::<fn(usize, usize, &candle_core::Tensor)>,
    ).map_err(|e| candle_core::Error::Msg(e.to_string()))?;

    let out_dir = "outputs/flux_showcase";
    std::fs::create_dir_all(out_dir).ok();
    let out_path = format!("{}/flux_klein_9b_img2img.png", out_dir);
    result_img.save(&out_path)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to save output image: {}", e)))?;

    println!("\n🎉 Flux.2-Klein-9B Img2Img Saved: {} ({} steps, {:.2}s)", out_path, metrics.unet_steps, t0.elapsed().as_secs_f64());
    Ok(())
}
