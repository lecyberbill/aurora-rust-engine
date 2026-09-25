// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Stable Audio Oobleck VAE (44.1 kHz) decode validation

use aurora_rust_engine::audio::{AutoencoderOobleck, OobleckConfig};
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
    let dev = Device::Cpu;
    let bytes = std::fs::read("outputs/audio_ref/sao_ref.safetensors")?;
    let st = SafeTensors::deserialize(&bytes)?;
    let latents = load_f32(&st, "final_latents", &dev)?; // [1,64,1024]
    let ref_audio = load_f32(&st, "audio", &dev)?; // [1,2,T]

    let cfg = OobleckConfig {
        audio_channels: 2,
        channel_multiples: vec![1, 2, 4, 8, 16],
        decoder_channels: 128,
        decoder_input_channels: 64,
        downsampling_ratios: vec![2, 4, 4, 8, 8],
        sampling_rate: 44100,
    };
    let vae = AutoencoderOobleck::from_safetensors(
        "G:/models/Audio/stable-audio-open-models/vae/diffusion_pytorch_model.safetensors",
        cfg,
        &dev,
        DType::F32,
    )?;
    println!("vae samples_per_frame = {}", vae.samples_per_frame());

    let wav = vae.decode(&latents)?;
    let n = wav.samples.len() / 2;
    // wav.samples is interleaved [L0,R0,...]; rebuild planar [2, n]
    let mut l = Vec::with_capacity(n);
    let mut r = Vec::with_capacity(n);
    for i in 0..n {
        l.push(wav.samples[2 * i]);
        r.push(wav.samples[2 * i + 1]);
    }
    let lt = Tensor::from_vec(l, (1, 1, n), &dev)?;
    let rt = Tensor::from_vec(r, (1, 1, n), &dev)?;
    let got = Tensor::cat(&[&lt, &rt], 1)?; // [1,2,n]
    let refn = ref_audio.narrow(2, 0, n)?;
    let d = (&got - &refn)?.abs()?.max_all()?.to_scalar::<f32>()?;
    let mean = (&got - &refn)?.abs()?.mean_all()?.to_scalar::<f32>()?;
    println!("[sao-vae] decode max|diff| = {:.3e}  mean|diff| = {:.3e}  n = {} {} vs {}", d, mean, n, got.dims().len(), refn.dims().len());
    Ok(())
}
