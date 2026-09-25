// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: StableAudio DiT internals validation

use aurora_rust_engine::models::StableAudioDit;
use candle_core::{DType, Device, Tensor};
use safetensors::SafeTensors;

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

fn main() -> anyhow::Result<()> {
    let dev = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let dtype = DType::F32;
    let bytes = std::fs::read("outputs/audio_ref/sao_ref.safetensors")?;
    let st = SafeTensors::deserialize(&bytes)?;
    let latents0 = load_f32(&st, "latents0", &dev)?.to_dtype(dtype)?;
    let cond = load_f32(&st, "cond", &dev)?.to_dtype(dtype)?;
    let global = load_f32(&st, "global", &dev)?.to_dtype(dtype)?;
    let timesteps = load_f32(&st, "timesteps", &dev)?;

    let ints = std::fs::read("outputs/audio_ref/sao_dit_int.safetensors")?;
    let ist = SafeTensors::deserialize(&ints)?;
    let r = |n: &str| load_f32(&ist, n, &dev).unwrap();
    let d = |a: &Tensor, b: &Tensor| -> f32 {
        (a - b).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap()
    };

    let inp = Tensor::cat(&[&latents0, &latents0], 0)?;
    let t0 = timesteps.narrow(0, 0, 1)?.to_dtype(dtype)?;

    let dit = StableAudioDit::from_safetensors(
        "G:/models/Audio/stable-audio-open-models/transformer/diffusion_pytorch_model.safetensors",
        &dev,
        dtype,
    )?;
    let (enc, glob, xin, b0, out) = dit.forward_verbose(&inp, &t0, &cond, &global)?;
    println!("enc  = {:.3e}", d(&enc, &r("enc")));
    println!("glob = {:.3e}", d(&glob, &r("glob")));
    println!("xin  = {:.3e}", d(&xin, &r("xin")));
    println!("b0   = {:.3e}  (all {:.3e})", d(&b0, &r("b0")), d(&b0, &r("b0")));
    println!("out  = {:.3e}", d(&out, &r("out")));
    Ok(())
}
