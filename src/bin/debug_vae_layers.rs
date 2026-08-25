// [WFGY] Zone: SAFE | λ: 0.15 | Fallbacks: 0 | Action: Debug VAE layer by layer outputs

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use aurora_rust_engine::weights::{SafeTensorsArchive, WeightRouter};
use aurora_rust_engine::diffusion::vae_flux::FluxVaeDecoder;

fn main() -> anyhow::Result<()> {
    println!("🔍 Inspecting VAE Layer-by-Layer Outputs...");

    let checkpoint = "G:\\models\\flux\\flux1SchnellFp8_schnellFp8.safetensors";
    let archive = SafeTensorsArchive::open(checkpoint)?;
    let router = WeightRouter::new(&archive, Device::Cpu, DType::F32);
    let vb = router.vae_var_builder()?;
    let vae = FluxVaeDecoder::new(vb)?;

    let dummy_latents = Tensor::randn(0f32, 1f32, (1, 16, 64, 64), &Device::Cpu)?;
    println!("📦 Input Latents: {:?}", dummy_latents.shape());

    let out = vae.decode(&dummy_latents)?;
    println!("✨ VAE Decoded Output Shape: {:?}", out.shape());

    let vals = out.flatten_all()?.to_vec1::<f32>()?;
    let min_v = vals.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_v = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean_v = vals.iter().sum::<f32>() / vals.len() as f32;

    println!("📊 Raw Output Stats: Min = {:.4}, Max = {:.4}, Mean = {:.4}", min_v, max_v, mean_v);

    let img = vae.decode_to_image(&dummy_latents)?;
    let p_bytes = img.as_raw();
    let p_min = p_bytes.iter().min().copied().unwrap_or(0);
    let p_max = p_bytes.iter().max().copied().unwrap_or(0);
    let p_mean = p_bytes.iter().map(|&x| x as f64).sum::<f64>() / p_bytes.len() as f64;

    println!("🎨 Image Pixel Stats: Min = {}, Max = {}, Mean = {:.2}", p_min, p_max, p_mean);
    img.save("outputs/flux_showcase/vae_test_rgb.png")?;
    println!("💾 Saved test VAE image to outputs/flux_showcase/vae_test_rgb.png");

    Ok(())
}
