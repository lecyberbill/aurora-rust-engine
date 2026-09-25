// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: ACE-Step cover task (source audio -> VAE encode -> src+timbre -> DiT -> WAV)

use aurora_rust_engine::audio::{seeded_randn, WavAudio};
use aurora_rust_engine::models::acestep_tasks;
use aurora_rust_engine::pipelines::AudioDiffusionPipeline;
use candle_core::{DType, Device, Tensor};

fn main() -> anyhow::Result<()> {
    let argv: Vec<String> = std::env::args().collect();
    let mut model_dir = "G:/models/Audio".to_string();
    let mut src_wav = "outputs/audio_showcase/codec_original.wav".to_string();
    let mut out_file = "outputs/audio_showcase/cover_song.ogg".to_string();
    let mut steps = 8usize;
    let mut guidance = 1.0f32;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model-dir" | "-m" => { model_dir = argv.get(i + 1).cloned().unwrap_or(model_dir); i += 1; }
            "--src" => { src_wav = argv.get(i + 1).cloned().unwrap_or(src_wav); i += 1; }
            "--out" | "-o" => { out_file = argv.get(i + 1).cloned().unwrap_or(out_file); i += 1; }
            "--steps" | "-s" => { steps = argv.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(steps); i += 1; }
            "--guidance" | "-g" => { guidance = argv.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(guidance); i += 1; }
            _ => {}
        }
        i += 1;
    }

    let caption = "warm uplifting acoustic pop, acoustic guitar, piano, drums, smooth vocal melody";
    let lyrics = "[verse]\nShining like the morning sun, a brand new melody has just begun";
    let language = "fr";
    let seed = 42u64;

    let pipeline = AudioDiffusionPipeline::from_pretrained(&model_dir)?;
    println!("pipeline: variant={:?} device={:?} dtype={:?}", pipeline.variant, pipeline.device, pipeline.dtype);

    // Source audio -> [1, 2, N] planar 48 kHz stereo.
    let mut wav = WavAudio::load_wav(&src_wav)?;
    wav = wav.resample(48_000);
    if wav.channels == 1 {
        let mut inter = Vec::with_capacity(wav.samples.len() * 2);
        for s in &wav.samples { inter.push(*s); inter.push(*s); }
        wav = WavAudio::new(inter, 48_000, 2);
    }
    let n = wav.samples.len() / 2;
    let mut planar = vec![0f32; 2 * n];
    for i in 0..n {
        planar[i] = wav.samples[2 * i];
        planar[n + i] = wav.samples[2 * i + 1];
    }
    let audio = Tensor::from_vec(planar, (1, 2, n), &pipeline.device)?.to_dtype(pipeline.dtype)?;

    // VAE encode -> [1, 64, T] (latent frames).
    let enc = pipeline.vae.encode(&audio)?; // [1,64,T]
    let num_frames = enc.dim(2)?;
    let duration = num_frames as f32 / 25.0;
    println!("source: {} samples -> {} latent frames ({:.2}s)", n, num_frames, duration);

    let src_frame_major = enc.transpose(1, 2)?.contiguous()?; // [1,T,64]
    let timbre_frames = num_frames.min(750);
    let refer = src_frame_major.narrow(1, 0, timbre_frames)?.contiguous()?;

    let instruction = acestep_tasks::generate_instruction("cover", None, None);
    println!("instruction: {}", instruction);
    let (cond, ctx) = pipeline.build_conditioning_ex(
        &instruction, caption, lyrics, language, duration, num_frames, Some(&refer), Some(&src_frame_major),
    )?;

    let noise = seeded_randn(num_frames, 64, 0.0, 1.0, seed, pipeline.dtype, &pipeline.device)?;
    let latents = pipeline.diffuse_guided(&cond, &ctx, &noise, steps, guidance)?;
    let out = pipeline.decode(&latents)?;
    if let Some(dir) = std::path::Path::new(&out_file).parent() { std::fs::create_dir_all(dir)?; }
    out.save_auto(&out_file)?;
    println!("saved {out_file} ({:.2}s)", out.duration_seconds());
    Ok(())
}
