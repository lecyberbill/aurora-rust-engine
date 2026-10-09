// [WFGY] Zone: SAFE | λ: 0.10 | Fallbacks: 0 | Action: Standalone VAE Encode & Decode Diagnostic Probe

use std::path::PathBuf;
use std::time::Instant;
use candle_core::{DType, Device, Tensor};
use aurora_rust_engine::device::auto_device;
use aurora_rust_engine::diffusion::vae_qwen::QwenImageVaeDecoder;
use aurora_rust_engine::weights::SafeTensorsArchive;

fn main() -> anyhow::Result<()> {
    println!("================================================================================");
    println!("🧪 Qwen Image VAE Reconstruction Probe (Synthetic Image Test)");
    println!("================================================================================");

    let device = auto_device()?;
    let dtype = if !device.is_cpu() { DType::BF16 } else { DType::F32 };
    println!("🎮 Target Device: {:?} | Precision: {:?}", device, dtype);

    let vae_path = PathBuf::from("/models/comfyui/vae/qwen_image_vae.safetensors");
    println!("📂 Loading VAE from {:?}", vae_path);
    let vae_archive = SafeTensorsArchive::open(&vae_path)?;

    let mut vae_tensors = std::collections::HashMap::new();
    for k in vae_archive.keys() {
        if let Ok(t) = vae_archive.get_tensor(&k, &device, DType::F32) {
            vae_tensors.insert(k.to_string(), t);
        }
    }
    println!("📦 Loaded {} VAE tensors into memory", vae_tensors.len());

    let vae_vb = candle_nn::VarBuilder::from_tensors(vae_tensors, DType::F32, &device);
    let decoder = QwenImageVaeDecoder::new(vae_vb)?;

    std::fs::create_dir_all("outputs")?;

    // 1. Test decoding a zero latent (should produce uniform / neutral RGB image)
    println!("\n🖼️ 1. Decoding zero latents [1, 16, 128, 128]:");
    let zero_latents = Tensor::zeros((1, 16, 128, 128), DType::F32, &device)?;
    let t0 = Instant::now();
    let decoded_zero = decoder.decode(&zero_latents)?;
    let elapsed = t0.elapsed();
    println!("   ⚡ Decoded zero in {:.2?}: shape={:?}", elapsed, decoded_zero.dims());
    let rgb_zero = aurora_rust_engine::diffusion::vae::tensor_to_rgb_image(&decoded_zero)?;
    rgb_zero.save("outputs/vae_test_zero.png")?;
    println!("   📸 Saved outputs/vae_test_zero.png");

    // 2. Test decoding structured / synthetic latents (checkerboard pattern across channels)
    println!("\n🖼️ 2. Decoding structured synthetic latents [1, 16, 128, 128]:");
    let mut pattern_data = vec![0f32; 16 * 128 * 128];
    for c in 0..16 {
        for y in 0..128 {
            for x in 0..128 {
                let idx = c * 128 * 128 + y * 128 + x;
                // Channel gradient + geometric frequency
                let freq = (c as f32 + 1.0) * 0.05;
                pattern_data[idx] = ((x as f32 * freq).sin() * (y as f32 * freq).cos()) * 1.5;
            }
        }
    }
    let structured_latents = Tensor::from_vec(pattern_data, (1, 16, 128, 128), &device)?;
    let decoded_struct = decoder.decode(&structured_latents)?;
    let rgb_struct = aurora_rust_engine::diffusion::vae::tensor_to_rgb_image(&decoded_struct)?;
    rgb_struct.save("outputs/vae_test_structured.png")?;
    println!("   📸 Saved outputs/vae_test_structured.png");

    println!("\n✅ VAE Standalone Probe Complete!");
    Ok(())
}
