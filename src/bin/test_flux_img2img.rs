// [WFGY] Zone: SAFE | λ: 0.2 | Fallbacks: 0 | Action: Test and Showcase Pure Rust FLUX Img2Img Pipeline

use candle_core::{DType, Device, Result};
use std::path::PathBuf;
use std::time::Instant;
use aurora_rust_engine::pipelines::FluxPipeline;
use aurora_rust_engine::traits::Img2ImgParams;
use aurora_rust_engine::diffusion::vae_flux::{FluxVaeDecoder, FluxVaeEncoder};
use aurora_rust_engine::weights::{SafeTensorsArchive, WeightRouter};

fn main() -> Result<()> {
    println!("================================================================================");
    println!("🚀 AURORA PURE RUST FLUX.2 KLEIN IMAGE-TO-IMAGE (IMG2IMG) TRANSFORMATION");
    println!("   • 32-Channel VAE Encoder & Decoder with BatchNorm Latent Standardization");
    println!("   • Flow Matching Euler ODE Solver with Sigma Interpolation");
    println!("================================================================================");

    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let checkpoint = "G:\\models\\flux\\fluxKlein4BPro_v10.safetensors";
    let vae_path = "G:\\models\\vae\\flux2-vae.safetensors";
    let qwen_path = "G:\\models\\clip\\qwen_3_4b.safetensors";
    let input_img_path = "outputs/flux_showcase/flux_klein_4b_1024_seed42.png";

    if !std::path::Path::new(input_img_path).exists() {
        println!("[-] Input image not found: {}", input_img_path);
        return Ok(());
    }

    let input_img = image::open(input_img_path)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to open image: {}", e)))?
        .to_rgb8();

    println!("\n📥 Loading Flux Checkpoint: {}", checkpoint);
    let mut pipeline = FluxPipeline::from_single_file_streaming(checkpoint, device.clone())
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

    pipeline.enable_flash_attn();

    // Attach Qwen3
    let qwen_archive = SafeTensorsArchive::open(PathBuf::from(qwen_path))
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let qwen_vb = candle_nn::VarBuilder::from_tensors(
        qwen_archive.tensor_names().into_iter().filter_map(|k| {
            qwen_archive.get_tensor(&k, &Device::Cpu, DType::F16).ok().map(|t| (k, t))
        }).collect(),
        DType::F16,
        &Device::Cpu,
    );
    let qwen = aurora_rust_engine::text::Qwen3TextEncoder::new(qwen_vb, Some(std::path::Path::new("qwen_tokenizer.json")))
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    pipeline.set_qwen3(qwen);

    // Attach Flux 2 VAE Decoder and Encoder
    let vae_archive = SafeTensorsArchive::open(PathBuf::from(vae_path))
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let vae_router = WeightRouter::new(&vae_archive, device.clone(), DType::F16);
    let vae_vb = vae_router.vae_var_builder()
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    
    let decoder = FluxVaeDecoder::new(vae_vb.clone())?;
    let encoder = FluxVaeEncoder::new(vae_vb)?;
    pipeline.set_vae(decoder);
    pipeline.set_vae_encoder(encoder);

    let prompt = "a majestic lion wearing a golden crown and diamond armor sitting on a rock during sunset, photorealistic, 8k";
    println!("\n🎨 Executing Img2Img Transformation (Strength: 0.65, 4 Steps)...");
    println!("📝 Prompt: \"{}\"", prompt);

    let params = Img2ImgParams {
        prompt,
        image: input_img,
        strength: 0.65,
        num_steps: 4,
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
    let out_path = format!("{}/flux2_img2img_lion_crown.png", out_dir);
    result_img.save(&out_path)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to save output image: {}", e)))?;

    println!("\n================================================================================");
    println!("🎉 Flux.2 Img2Img Transformed Image Saved Successfully!");
    println!("📁 Output Path: {}", out_path);
    println!("📊 Performance Telemetry:");
    println!("   • Active Steps:           {}", metrics.unet_steps);
    println!("   • ODE Denoising Duration: {:.2}s", metrics.unet_total_ms / 1000.0);
    println!("   • Throughput:             {:.2} it/s ({:.2} ms/step)", metrics.unet_it_per_sec, metrics.unet_step_avg_ms);
    println!("   • VAE & Unpatchify:       {:.2}s", metrics.vae_decode_ms / 1000.0);
    println!("   • Total Wall-Clock:       {:.2}s", t0.elapsed().as_secs_f64());
    println!("================================================================================");

    Ok(())
}
