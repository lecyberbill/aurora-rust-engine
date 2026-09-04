// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: End-to-End Verification of Flux.2 Inpainting Pipeline

use candle_core::{DType, Device, Result};
use std::path::PathBuf;
use aurora_rust_engine::pipelines::FluxPipeline;
use aurora_rust_engine::traits::InpaintParams;
use aurora_rust_engine::weights::{SafeTensorsArchive, WeightRouter};
use aurora_rust_engine::diffusion::vae_flux::{FluxVaeDecoder, FluxVaeEncoder};

fn main() -> Result<()> {
    println!("================================================================================");
    println!("🚀 AURORA PURE RUST FLUX.2 KLEIN INPAINTING VERIFICATION");
    println!("   • 32-Channel VAE Encode/Decode with BatchNorm Standardized Patch Blending");
    println!("   • Flow Matching Euler ODE with Exact Mask Boundary Preservation");
    println!("================================================================================");

    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let checkpoint = "G:\\models\\flux\\fluxKlein4BPro_v10.safetensors";
    let qwen_path = "G:\\models\\clip\\qwen_3_4b.safetensors";
    let vae_path = "G:\\models\\vae\\flux2-vae.safetensors";

    let input_path = "outputs/flux_showcase/flux_klein_4b_1024_seed42.png";
    let input_img = image::open(input_path)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to open base image: {}", e)))?
        .to_rgb8();

    let (w, h) = input_img.dimensions();

    // Create a circular inpainting mask on top of the lion's head:
    // Center at (w/2, h/4), radius 180 pixels
    let mut mask = image::GrayImage::new(w, h);
    let center_x = (w / 2) as i32;
    let center_y = (h / 3) as i32;
    let radius = 220i32;

    for y in 0..h {
        for x in 0..w {
            let dx = x as i32 - center_x;
            let dy = y as i32 - center_y;
            if dx * dx + dy * dy <= radius * radius {
                mask.put_pixel(x, y, image::Luma([255u8]));
            } else {
                mask.put_pixel(x, y, image::Luma([0u8]));
            }
        }
    }

    mask.save("outputs/flux_showcase/flux2_inpaint_mask.png")
        .map_err(|e| candle_core::Error::Msg(format!("Failed to save mask: {}", e)))?;
    println!("✅ Saved mask: outputs/flux_showcase/flux2_inpaint_mask.png");

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

    let prompt = "a majestic lion wearing an intricate glowing golden crown with emerald gems, photorealistic, 8k";
    println!("\n🎨 Executing Inpainting (Strength: 0.85, 4 Steps)...");
    println!("📝 Prompt: \"{}\"", prompt);

    let params = InpaintParams {
        prompt,
        negative_prompt: None,
        image: input_img,
        mask,
        mask_blur: 0,
        strength: 0.85,
        num_steps: 4,
        guidance_scale: 1.0,
        seed: 42,
    };

    let (inpainted_img, metrics) = pipeline.generate_inpaint(params, None::<fn(usize, usize, &candle_core::Tensor)>)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

    let out_path = "outputs/flux_showcase/flux2_inpaint_lion_crown.png";
    inpainted_img.save(out_path)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to save output: {}", e)))?;

    println!("\n================================================================================");
    println!("🎉 Flux.2 Inpainted Image Saved Successfully!");
    println!("📁 Output Path: {}", out_path);
    println!("📊 Performance Telemetry:");
    println!("   • Active Steps:           {}", metrics.unet_steps);
    println!("   • ODE Denoising Duration: {:.2}s", metrics.unet_total_ms / 1000.0);
    println!("   • Throughput:             {:.2} it/s ({:.2} ms/step)", metrics.unet_it_per_sec, metrics.unet_step_avg_ms);
    println!("   • VAE & Unpatchify:       {:.2}s", metrics.vae_decode_ms / 1000.0);
    println!("   • Total Wall-Clock:       {:.2}s", metrics.total_wallclock_ms / 1000.0);
    println!("================================================================================\n");

    Ok(())
}
