// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Audible ACE-Step 5Hz codec round-trip (DiT latents -> FSQ codes -> latents -> WAV)

use aurora_rust_engine::audio::{seeded_randn, AudioFormat};
use aurora_rust_engine::models::AceStepAudioCodec;
use aurora_rust_engine::pipelines::AudioDiffusionPipeline;
use candle_core::{DType, Device};

fn main() -> anyhow::Result<()> {
    let argv: Vec<String> = std::env::args().collect();
    let mut model_dir = "G:/models/Audio-base".to_string();
    let mut i = 1;
    while i < argv.len() {
        if (argv[i] == "--model-dir" || argv[i] == "-m") && i + 1 < argv.len() {
            model_dir = argv[i + 1].clone();
        }
        i += 1;
    }

    let caption = "warm uplifting acoustic pop, acoustic guitar, piano, drums";
    let lyrics = "[verse]\nShining like the morning sun";
    let language = "fr";
    let duration = 6.0f32; // 150 latent frames (divisible by 5)
    let steps = 8usize;
    let seed = 2024u64;
    let guidance = 7.0f32;

    println!("Loading pipeline + codec ({})...", model_dir);
    let pipeline = AudioDiffusionPipeline::from_pretrained(&model_dir)?;
    let num_frames = AudioDiffusionPipeline::latent_frames(duration);
    println!("  variant={:?} device={:?} dtype={:?}  frames={}", pipeline.variant, pipeline.device, pipeline.dtype, num_frames);

    // 1. Generate original latents with the diffusion pipeline.
    let noise = seeded_randn(num_frames, 64, 0.0, 1.0, seed, pipeline.dtype, &pipeline.device)?;
    let (cond, ctx) = pipeline.build_conditioning(caption, lyrics, language, duration, num_frames)?;
    let latents = pipeline.diffuse_guided(&cond, &ctx, &noise, steps, guidance)?; // [1,64,T]

    // 2. Round-trip through the 5Hz FSQ codec (CPU f32 for exactness).
    let codec = AceStepAudioCodec::from_safetensors(
        format!("{model_dir}/audio_codec/codec.safetensors"),
        &Device::Cpu,
        DType::F32,
    )?;
    let lat_t = latents.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous()?; // [1,T,64]
    let (_q, idx) = codec.tokenize(&lat_t)?;
    println!(
        "  codes: {} tokens ({} -> {}), range [{}, {}]",
        idx.dim(1)?,
        num_frames,
        idx.dim(1)?,
        idx.min_all()?.to_scalar::<u32>()?,
        idx.max_all()?.to_scalar::<u32>()?
    );
    let rec_t = codec.detokenize_from_indices(&idx)?; // [1,T,64]
    let rec = rec_t.transpose(1, 2)?.contiguous()?.to_device(&pipeline.device)?.to_dtype(pipeline.dtype)?; // [1,64,T]
    let stat = |t: &candle_core::Tensor, n: &str| -> anyhow::Result<()> {
        let f = t.to_dtype(DType::F32)?.flatten_all()?;
        let m = f.abs()?.mean_all()?.to_scalar::<f32>()?;
        let sq = f.sqr()?.mean_all()?.to_scalar::<f32>()?;
        let rms = sq.sqrt();
        let mx = f.abs()?.max_all()?.to_scalar::<f32>()?;
        println!("   {n}: |x|mean={m:.4} rms={rms:.4} max={mx:.4}");
        Ok(())
    };
    println!("  latent stats:");
    stat(&lat_t, "orig")?;
    stat(&rec_t, "recon")?;
    let diff = (&lat_t - &rec_t)?.to_dtype(DType::F32)?.abs()?.mean_all()?.to_scalar::<f32>()?;
    println!("   latent |diff| mean={diff:.4}");
    {
        let a = lat_t.to_dtype(DType::F32)?.flatten_all()?;
        let b = rec_t.to_dtype(DType::F32)?.flatten_all()?;
        let dot = (&a * &b)?.sum_all()?.to_scalar::<f32>()?;
        let na = a.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
        let nb = b.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
        println!("   latent corr(orig,recon) = {:.4}", dot / (na * nb).max(1e-9));
    }
    if let Some(ce) = &pipeline.condition_encoder {
        let s = ce.silence_latent.to_dtype(DType::F32)?.flatten_all()?;
        let m = s.abs()?.mean_all()?.to_scalar::<f32>()?;
        let r = s.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
        let mx = s.abs()?.max_all()?.to_scalar::<f32>()?;
        println!("   silence_latent: |x|mean={m:.4} rms={r:.4} max={mx:.4}  (codec trained domain)");
    }

    // 3. Decode to WAV:
    //    (a) original latents,
    //    (b) direct FSQ round-trip (expected near-silent: semantic codec),
    //    (c) codec hints used as the DiT source (the intended use) -> regenerated audio.
    let wav_orig = pipeline.decode(&latents)?;
    let wav_rec = pipeline.decode(&rec)?;
    let hints = rec_t.to_device(&pipeline.device)?.to_dtype(pipeline.dtype)?; // [1,T,64]
    let ctx_hints = pipeline.context_from_src(&hints, num_frames)?;
    let lat_hints = pipeline.diffuse_guided(&cond, &ctx_hints, &noise, steps, guidance)?;
    let wav_hints = pipeline.decode(&lat_hints)?;

    std::fs::create_dir_all("outputs/audio_showcase")?;
    wav_orig.save_encoded("outputs/audio_showcase/codec_original.wav", AudioFormat::Wav)?;
    wav_rec.save_encoded("outputs/audio_showcase/codec_roundtrip.wav", AudioFormat::Wav)?;
    wav_hints.save_encoded("outputs/audio_showcase/codec_hint_conditioned.wav", AudioFormat::Wav)?;
    println!("  saved codec_original.wav, codec_roundtrip.wav (direct, ~silent), codec_hint_conditioned.wav");

    Ok(())
}
