// [WFGY] Zone: SAFE | λ: 0.15 | Fallbacks: 0 | Action: Validate the Rust Oobleck VAE encoder vs PyTorch diffusers

use candle_core::{DType, Device, Tensor};
use safetensors::SafeTensors;
use aurora_rust_engine::audio::{AutoencoderOobleck, OobleckConfig};

fn load_f32(st: &SafeTensors, name: &str, dev: &Device) -> anyhow::Result<Tensor> {
    let t = st.tensor(name)?;
    let shape: Vec<usize> = t.shape().to_vec();
    let data: Vec<f32> = t
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Ok(Tensor::from_vec(data, shape, dev)?)
}

fn max_abs(a: &Tensor, b: &Tensor) -> anyhow::Result<f32> {
    Ok((a - b)?.abs()?.max_all()?.to_scalar::<f32>()?)
}

fn main() -> anyhow::Result<()> {
    let dev = Device::Cpu;
    let buf = std::fs::read("outputs/audio_ref/vae_enc_ref.safetensors")?;
    let st = SafeTensors::deserialize(&buf)?;
    let audio = load_f32(&st, "audio", &dev)?; // [1,2,N]
    let ref_raw = load_f32(&st, "raw", &dev)?;
    let ref_mean = load_f32(&st, "mean", &dev)?;
    let ref_std = load_f32(&st, "std", &dev)?;

    let vae = AutoencoderOobleck::from_safetensors(
        "G:/models/Audio/vae/diffusion_pytorch_model.safetensors",
        OobleckConfig::default(),
        &dev,
        DType::F32,
    )?;

    let raw = vae.encode_raw(&audio)?;
    println!("  raw    r={:?} ref={:?} max|diff|={:.3e}", raw.shape(), ref_raw.shape(), max_abs(&raw, &ref_raw)?);
    let (mean, std) = vae.encode_dist(&audio)?;
    println!("=== VAE encode ===");
    println!("  mean r={:?} ref={:?} max|diff|={:.3e}", mean.shape(), ref_mean.shape(), max_abs(&mean, &ref_mean)?);
    println!("  std  r={:?} ref={:?} max|diff|={:.3e}", std.shape(), ref_std.shape(), max_abs(&std, &ref_std)?);

    // Tiled encode consistency vs monolithic on a longer clip (realistic overlap).
    let long = Tensor::randn(0.0f32, 0.3f32, (1, 2, 1920 * 200), &dev)?;
    let lat_mono = vae.encode(&long)?;
    let lat_tiled = vae.encode_tiled(&long, 32, 16)?;
    println!("  tiled vs mono max|diff| = {:.3e} (T={})", max_abs(&lat_tiled, &lat_mono)?, lat_mono.dim(2)?);

    // Round-trip sanity: encode -> decode vs original length.
    let rec = vae.decode(&lat_mono)?;
    println!("  round-trip: audio={} -> lat={:?} -> wav={} samples", audio.dim(2)?, lat_mono.shape(), rec.samples.len());

    Ok(())
}
