// [WFGY] Zone: SAFE | λ: 0.10 | Fallbacks: 0 | Action: Compare Rust VAE with Diffusers ground truth

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use aurora_rust_engine::diffusion::vae_flux::FluxVaeDecoder;
use aurora_rust_engine::weights::SafeTensorsArchive;
use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    let device = Device::new_cuda(0)?;
    let vae_path = "G:\\models\\vae\\flux2-vae.safetensors";
    let test_vec_path = "outputs/flux_showcase/vae_test_vectors.safetensors";

    let test_archive = SafeTensorsArchive::open(test_vec_path)?;
    let mut tensors = HashMap::new();
    for k in test_archive.keys() {
        tensors.insert(k.clone(), test_archive.get_tensor(&k, &device, DType::F32)?);
    }
    let vb_test = VarBuilder::from_tensors(tensors, DType::F32, &device);
    let lat = vb_test.get((1, 32, 16, 16), "lat")?;
    let expected_out = vb_test.get((1, 3, 128, 128), "expected_out")?;

    let vae_archive = SafeTensorsArchive::open(vae_path)?;
    let mut vae_tensors = HashMap::new();
    for k in vae_archive.keys() {
        vae_tensors.insert(k.clone(), vae_archive.get_tensor(&k, &device, DType::F32)?);
    }
    let vb_vae = VarBuilder::from_tensors(vae_tensors, DType::F32, &device);
    let vae = FluxVaeDecoder::new(vb_vae)?;

    let rust_out = vae.decode(&lat)?;
    let diff = (&rust_out - &expected_out)?.abs()?.flatten_all()?.to_vec1::<f32>()?;
    let max_diff = diff.iter().cloned().fold(0.0f32, f32::max);
    let mean_diff: f32 = diff.iter().sum::<f32>() / diff.len() as f32;

    println!("============================================================");
    println!("🔬 VAE NUMERICAL FIDELITY TEST:");
    println!("   • Max Diff between Rust VAE & Python Diffusers: {:.8}", max_diff);
    println!("   • Mean Diff: {:.8}", mean_diff);
    println!("============================================================");

    Ok(())
}
