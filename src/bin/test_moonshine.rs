// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Moonshine STT end-to-end

use aurora_rust_engine::audio::WavAudio;
use aurora_rust_engine::models::MoonshineModel;
use candle_core::{DType, Device, Tensor};

fn main() -> anyhow::Result<()> {
    let argv: Vec<String> = std::env::args().collect();
    let mut model_dir = "G:/models/moonshine-tiny".to_string();
    let mut wav_path = "outputs/audio_showcase/test_whisper_input.wav".to_string();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "-m" | "--model-dir" => { model_dir = argv[i + 1].clone(); i += 1; }
            "-w" | "--wav" => { wav_path = argv[i + 1].clone(); i += 1; }
            _ => {}
        }
        i += 1;
    }
    let dev = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let dtype = if dev.is_cuda() { DType::F32 } else { DType::F32 };

    let wav = WavAudio::load_wav(&wav_path)?.resample(16_000);
    let mono: Vec<f32> = if wav.channels == 2 {
        wav.samples.chunks_exact(2).map(|c| (c[0] + c[1]) * 0.5).collect()
    } else {
        wav.samples.clone()
    };
    let n = mono.len();
    let input = Tensor::from_vec(mono, (1, n), &dev)?.to_dtype(dtype)?;

    let t0 = std::time::Instant::now();
    let model = MoonshineModel::from_safetensors(format!("{model_dir}/model.safetensors"), &dev, dtype)?;
    let enc = model.encode(&input)?;
    let ids = model.greedy_decode(&enc, 1, 2, 256)?;

    let tok = tokenizers::Tokenizer::from_file(format!("{model_dir}/tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let text = tok.decode(&ids, true).map_err(|e| anyhow::anyhow!("decode: {e}"))?;
    println!("ids = {:?}", ids);
    println!("text = {text}");
    println!("({:.1}s, {} Hz, {} samples)", t0.elapsed().as_secs_f32(), 16000, n);
    Ok(())
}
