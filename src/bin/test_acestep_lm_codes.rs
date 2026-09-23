// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Full ACE-Step LM-codes path (LM CoT -> audio codes -> hints -> DiT -> WAV)

use aurora_rust_engine::audio::{seeded_randn, AudioFormat};
use aurora_rust_engine::models::{AceStepAudioCodec, AceStepLm};
use aurora_rust_engine::pipelines::AudioDiffusionPipeline;
use candle_core::{DType, Device, Tensor};

fn main() -> anyhow::Result<()> {
    let lm_dir = "G:/models/Audio/Ace-Step1.5/acestep-5Hz-lm-1.7B";
    let model_dir = std::env::args().nth(1).unwrap_or_else(|| "G:/models/Audio".to_string()); // Turbo by default
    let codec_dir = "G:/models/Audio-base";
    let caption = "warm uplifting acoustic pop, acoustic guitar, piano, drums, smooth vocal melody";
    let lyrics = "[verse]\nShining like the morning sun, a brand new melody has just begun";
    let language = "fr";
    let mut duration_sec = 6.0f32;
    let steps = 8usize;
    let seed = 42u64;
    let guidance = 1.0f32; // Turbo (CFG-distilled); ignored by `diffuse_guided` for Turbo anyway

    let device = if candle_core::utils::cuda_is_available() {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };
    let dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };

    // 1. LM planner: CoT then constrained audio-code generation.
    println!("Loading 5Hz LM planner...");
    let mut lm = AceStepLm::from_dir(lm_dir, &device, dtype)?;
    let cot = lm.plan(caption, lyrics, 220)?;
    println!("--- CoT ---\n{}", cot.trim());
    // Use the duration predicted by the LM (falls back to the requested one).
    if let Some(d) = cot
        .split("duration:")
        .nth(1)
        .and_then(|s| s.trim().split_whitespace().next())
        .and_then(|s| s.trim_matches(|c: char| !c.is_ascii_digit()).parse::<f32>().ok())
    {
        if d > 0.0 {
            duration_sec = d;
        }
    }
    duration_sec = duration_sec.clamp(4.0, 24.0);
    println!("--- planned duration: {duration_sec}s, steps: {steps}");

    let num_codes = (duration_sec * 5.0).round() as usize; // 5Hz
    let codes = lm.generate_codes(caption, lyrics, &cot, num_codes, seed)?;
    if codes.is_empty() {
        anyhow::bail!("LM produced no audio codes");
    }
    let (mn, mx) = (
        *codes.iter().min().unwrap(),
        *codes.iter().max().unwrap(),
    );
    println!("--- codes --- {} values, range [{}, {}]", codes.len(), mn, mx);
    // Free the LM before loading the DiT (12 GB budget).
    drop(lm);

    // 2. Codec: codes -> 25Hz hints.
    let num_frames = codes.len() * 5;
    let codec = AceStepAudioCodec::from_safetensors(
        format!("{codec_dir}/audio_codec/codec.safetensors"),
        &Device::Cpu,
        DType::F32,
    )?;
    let idx = Tensor::from_vec(
        codes.iter().map(|&c| c).collect::<Vec<u32>>(),
        (1, codes.len()),
        &Device::Cpu,
    )?;
    let hints_t = codec.detokenize_from_indices(&idx)?; // [1, num_frames, 64]
    println!("--- hints --- {:?}", hints_t.shape());

    // 3. DiT conditioned on the LM hints -> audio.
    println!("Loading DiT pipeline ({model_dir})...");
    let pipeline = AudioDiffusionPipeline::from_pretrained(&model_dir)?;
    let dur = num_frames as f32 / 25.0;
    let (cond, _ctx) = pipeline.build_conditioning(caption, lyrics, language, dur, num_frames)?;
    let hints = hints_t.to_device(&pipeline.device)?.to_dtype(pipeline.dtype)?;
    let ctx = pipeline.context_from_src(&hints, num_frames)?;
    let noise = seeded_randn(num_frames, 64, 0.0, 1.0, seed, pipeline.dtype, &pipeline.device)?;
    let latents = pipeline.diffuse_guided(&cond, &ctx, &noise, steps, guidance)?;
    let audio = pipeline.decode(&latents)?;

    std::fs::create_dir_all("outputs/audio_showcase")?;
    let out = "outputs/audio_showcase/lm_codes_song.wav";
    audio.save_encoded(out, AudioFormat::Wav)?;
    println!("saved {out} ({:.2}s)", audio.duration_seconds());

    Ok(())
}
