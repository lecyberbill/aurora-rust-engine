use candle_core::{DType, Device, Result};
use std::path::PathBuf;
use std::time::Instant;
use aurora_rust_engine::pipelines::FluxPipeline;
use aurora_rust_engine::traits::Img2ImgParams;
use aurora_rust_engine::diffusion::vae_flux::{FluxVaeDecoder, FluxVaeEncoder};
use aurora_rust_engine::weights::{SafeTensorsArchive, WeightRouter};

fn main() -> Result<()> {
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let checkpoint = std::env::var("CKPT").unwrap_or_else(|_| "G:\\models\\flux\\flux2DevFp8Scaled_fp8Scaled.safetensors".into());
    let vae_path = std::env::var("VAE").unwrap_or_else(|_| "G:\\models\\vae\\flux2-vae.safetensors".into());
    let mistral_dir = std::env::var("MISTRAL").unwrap_or_else(|_| "G:\\models\\clip\\FLUX.2-dev_text_encoder".into());
    let input_img_path = std::env::var("IN").unwrap_or_else(|_| "outputs/flux_showcase/flux_dev_1024_s20_g3.5.png".into());
    let strength: f64 = std::env::var("STRENGTH").ok().and_then(|s| s.parse().ok()).unwrap_or(0.55);
    let steps: usize = std::env::var("STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(12);
    let guidance: f64 = std::env::var("GUIDANCE").ok().and_then(|s| s.parse().ok()).unwrap_or(3.5);
    let prompt = std::env::var("PROMPT").unwrap_or_else(|_| "a majestic white arctic wolf with glowing golden eyes standing proudly in a moonlit snowy forest, photorealistic, 8k".into());

    if !std::path::Path::new(&input_img_path).exists() {
        println!("[-] Input image not found: {}", input_img_path);
        return Ok(());
    }
    let input_img = image::open(&input_img_path)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to open image: {}", e)))?
        .to_rgb8();

    println!("\n📥 Loading Flux.2-Dev for Img2Img: {}", checkpoint);
    let is_gguf = checkpoint.to_lowercase().ends_with(".gguf");
    let mut pipeline = if is_gguf {
        FluxPipeline::from_gguf_dtype(&checkpoint, device.clone(), DType::F16)
    } else {
        FluxPipeline::from_single_file_streaming(&checkpoint, device.clone())
    }.map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    pipeline.enable_flash_attn();

    // Mistral-3.2-24B VLM from shards dir (40-layer, language_model.model.layers.* prefix auto-detected)
    let mistral_dev = if device.is_cuda() { device.clone() } else { Device::Cpu };
    let mistral = if std::path::Path::new(&mistral_dir).is_dir() {
        aurora_rust_engine::text::Mistral3TextEncoder::from_dir(
            &mistral_dir, Some(std::path::Path::new("mistral_tokenizer.json")),
            mistral_dev, DType::F16,
        )
    } else {
        aurora_rust_engine::text::Mistral3TextEncoder::from_safetensors(
            &mistral_dir, Some(std::path::Path::new("mistral_tokenizer.json")),
            mistral_dev, DType::F16,
        )
    }.map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    pipeline.set_mistral(mistral);

    let vae_archive = SafeTensorsArchive::open(PathBuf::from(&vae_path))
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let vae_router = WeightRouter::new(&vae_archive, device.clone(), DType::F16);
    let vae_vb = vae_router.vae_var_builder()
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let decoder = FluxVaeDecoder::new(vae_vb.clone())?;
    let encoder = FluxVaeEncoder::new(vae_vb)?;
    pipeline.set_vae(decoder);
    pipeline.set_vae_encoder(encoder);

    println!("\n🎨 Executing Dev Img2Img (Strength: {}, {} Steps, Guidance: {})...", strength, steps, guidance);
    println!("📝 Prompt: \"{}\"", prompt);

    let params = Img2ImgParams {
        prompt: &prompt,
        image: input_img,
        strength,
        num_steps: steps,
        guidance_scale: guidance,
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
    let out_path = format!("{}/flux_dev_img2img.png", out_dir);
    result_img.save(&out_path)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to save output image: {}", e)))?;

    println!("\n🎉 Flux.2-Dev Img2Img Saved: {} ({} steps, {:.2}s)", out_path, metrics.unet_steps, t0.elapsed().as_secs_f64());
    Ok(())
}
