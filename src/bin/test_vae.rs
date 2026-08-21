// [WFGY] Zone: SAFE | λ: 0.10 | Fallbacks: 0 | Action: Benchmark 4-tile VAE decode speed

use candle_core::{DType, Device, Tensor};
use aurora_rust_engine::weights::{SafeTensorsArchive, WeightRouter};
use aurora_rust_engine::diffusion::vae::VaeDecoder;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let device = Device::new_cuda(0)?;
    let model_path = "G:\\models\\checkpoints\\duchaitenPonyXLNo_v60.safetensors";
    let archive = SafeTensorsArchive::open(model_path)?;
    let router = WeightRouter::new(&archive, device.clone(), DType::F16);

    let vae_vb = router.vae_var_builder_f32()?;
    let vae = VaeDecoder::new(vae_vb, true)?;

    let latents = Tensor::randn(0.0f32, 1.0f32, (1, 4, 128, 128), &device)?.to_dtype(DType::F16)?;

    println!("Testing 4-tile decode (tile_size=64, overlap=0)...");
    let t0 = Instant::now();
    let img = vae.decode_tiled(&latents, 64, 0)?;
    println!("4-Tile decode finished in {:.3}s (size: {}x{})", t0.elapsed().as_secs_f64(), img.width(), img.height());

    Ok(())
}
