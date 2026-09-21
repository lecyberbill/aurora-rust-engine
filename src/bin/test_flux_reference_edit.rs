// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: FLUX.2 Multi-Image Reference Conditioning / Mode Édition (Pure Rust 4D RoPE)

use candle_core::{DType, Device, Result};
use std::path::{Path, PathBuf};
use std::time::Instant;
use aurora_rust_engine::pipelines::FluxPipeline;
use aurora_rust_engine::traits::DiffusionParams;
use aurora_rust_engine::diffusion::vae_flux::{FluxVaeDecoder, FluxVaeEncoder};
use aurora_rust_engine::weights::{SafeTensorsArchive, WeightRouter};

fn main() -> Result<()> {
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let checkpoint = std::env::var("CKPT").unwrap_or_else(|_| "G:\\models\\flux\\flux2DevFp8Scaled_fp8Scaled.safetensors".into());
    let vae_path = std::env::var("VAE").unwrap_or_else(|_| "G:\\models\\vae\\flux2-vae.safetensors".into());
    let mistral_dir = std::env::var("MISTRAL").unwrap_or_else(|_| "G:\\models\\clip\\FLUX.2-dev_text_encoder".into());
    let ref_path = std::env::var("REF").unwrap_or_else(|_| "outputs/flux_showcase/flux_dev_img2img.png".into());
    let steps: usize = std::env::var("STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let guidance: f64 = std::env::var("GUIDANCE").ok().and_then(|s| s.parse().ok()).unwrap_or(3.5);
    let width: usize = std::env::var("WIDTH").ok().and_then(|s| s.parse().ok()).unwrap_or(512);
    let height: usize = std::env::var("HEIGHT").ok().and_then(|s| s.parse().ok()).unwrap_or(512);
    let prompt = std::env::var("PROMPT").unwrap_or_else(|_| "a majestic arctic fox sitting gracefully under golden autumn leaves, close-up portrait, photorealistic, 8k".into());

    if !Path::new(&ref_path).exists() {
        eprintln!("[-] Reference image not found: {}", ref_path);
        return Ok(());
    }
    let ref_img = image::open(&ref_path)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to open reference image: {}", e)))?;

    println!("\n📥 Loading FLUX.2 for Reference Conditioning: {}", checkpoint);
    let mut pipeline = FluxPipeline::from_single_file_streaming(&checkpoint, device.clone())
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    pipeline.enable_flash_attn();

    // Mistral VLM 40 layers for Dev
    if Path::new(&mistral_dir).exists() {
        let mistral_dev = if device.is_cuda() { device.clone() } else { Device::Cpu };
        let mistral = if Path::new(&mistral_dir).is_dir() {
            aurora_rust_engine::text::Mistral3TextEncoder::from_dir(
                &mistral_dir, Some(Path::new("mistral_tokenizer.json")),
                mistral_dev, DType::F16,
            )
        } else {
            aurora_rust_engine::text::Mistral3TextEncoder::from_safetensors(
                &mistral_dir, Some(Path::new("mistral_tokenizer.json")),
                mistral_dev, DType::F16,
            )
        }.map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        pipeline.set_mistral(mistral);
    }

    // VAE Encoder & Decoder
    let vae_archive = SafeTensorsArchive::open(PathBuf::from(&vae_path))
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let vae_router = WeightRouter::new(&vae_archive, device.clone(), DType::F16);
    let vae_vb = vae_router.vae_var_builder()
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let decoder = FluxVaeDecoder::new(vae_vb.clone())?;
    let encoder = FluxVaeEncoder::new(vae_vb)?;
    pipeline.set_vae(decoder);
    pipeline.set_vae_encoder(encoder);

    println!("\n🎨 Executing Reference-Conditioned Generation ({} Steps, Guidance: {})...", steps, guidance);
    println!("📷 Reference: {}", ref_path);
    println!("📝 Prompt: \"{}\"", prompt);

    let params = DiffusionParams {
        prompt: &prompt,
        negative_prompt: None,
        num_steps: steps,
        guidance_scale: guidance,
        width,
        height,
        seed: 42,
    };

    let t0 = Instant::now();
    let (result_img, metrics) = pipeline.generate_with_refs(
        params,
        &[ref_img],
        None::<fn(usize, usize, &candle_core::Tensor)>,
    ).map_err(|e| candle_core::Error::Msg(e.to_string()))?;

    let out_dir = "outputs/flux_showcase";
    std::fs::create_dir_all(out_dir).ok();
    let out_path = format!("{}/flux_reference_edit.png", out_dir);
    result_img.save(&out_path)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to save output image: {}", e)))?;

    println!("\n🎉 FLUX.2 Reference-Conditioned Image Saved: {} ({} steps, {:.2}s)", out_path, metrics.unet_steps, t0.elapsed().as_secs_f64());
    Ok(())
}
