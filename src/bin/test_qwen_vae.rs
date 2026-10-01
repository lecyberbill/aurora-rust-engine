use aurora_rust_engine::weights::SafeTensorsArchive;
use aurora_rust_engine::diffusion::vae_qwen::QwenImageVaeDecoder;
use candle_core::{DType, Device, Tensor};

fn main() -> anyhow::Result<()> {
    let p = "/models/comfyui/vae/qwen_image_vae.safetensors";
    println!("📂 Loading VAE from {}", p);
    let archive = SafeTensorsArchive::open(p)?;
    let mut vae_tensors = std::collections::HashMap::new();
    for key in archive.keys() {
        if let Ok(t) = archive.get_tensor(&key, &Device::Cpu, DType::F32) {
            vae_tensors.insert(key.to_string(), t);
        }
    }
    let vb = candle_nn::VarBuilder::from_tensors(vae_tensors, DType::F32, &Device::Cpu);
    let decoder = QwenImageVaeDecoder::new(vb)?;
    println!("✅ QwenImageVaeDecoder instantiated successfully!");

    let dummy_latent = Tensor::randn(0.0f32, 1.0f32, (1, 16, 128, 128), &Device::Cpu)?;
    println!("🚀 Decoding dummy latent of shape {:?}", dummy_latent.dims());
    let t0 = std::time::Instant::now();
    let decoded = decoder.decode(&dummy_latent)?;
    println!("🎉 Decoded successfully in {:.2}s: shape={:?}", t0.elapsed().as_secs_f64(), decoded.dims());
    assert_eq!(decoded.dims(), &[1, 3, 1024, 1024]);
    Ok(())
}
