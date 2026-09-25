// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: ACE-Step repaint task (mask + step injection + boundary blend)

use aurora_rust_engine::audio::{seeded_randn, WavAudio};
use aurora_rust_engine::models::{acestep_tasks, FlowMatchConfig};
use aurora_rust_engine::pipelines::AudioDiffusionPipeline;
use candle_core::{DType, Device, Tensor};

fn load_stereo(path: &str, device: &Device, dtype: DType) -> anyhow::Result<Tensor> {
    let mut wav = WavAudio::load_wav(path)?.resample(48_000);
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
    Ok(Tensor::from_vec(planar, (1, 2, n), device)?.to_dtype(dtype)?)
}

fn main() -> anyhow::Result<()> {
    let argv: Vec<String> = std::env::args().collect();
    let mut model_dir = "G:/models/Audio-base".to_string();
    let mut src_wav = "outputs/audio_showcase/codec_original.wav".to_string();
    let mut out_file = "outputs/audio_showcase/repaint_song.ogg".to_string();
    let mut steps = 50usize;
    let mut guidance = 7.0f32;
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
    let lyrics = "[instrumental]";
    let language = "fr";
    let seed = 42u64;

    let use_cpu = argv.iter().any(|a| a == "--cpu");
    let pipeline = if use_cpu {
        AudioDiffusionPipeline::from_folder(&model_dir, Device::Cpu, DType::F32)?
    } else {
        AudioDiffusionPipeline::from_pretrained(&model_dir)?
    };
    println!("pipeline: variant={:?} dtype={:?}", pipeline.variant, pipeline.dtype);

    let audio = load_stereo(&src_wav, &pipeline.device, pipeline.dtype)?;
    let enc = pipeline.vae.encode(&audio)?; // [1,64,T]
    let t = enc.dim(2)?;
    let src_fm = enc.transpose(1, 2)?.contiguous()?; // [1,T,64]
    let dur = t as f32 / 25.0;

    // Repaint the middle 30% of the clip.
    let start = (t as f32 * 0.35) as usize;
    let end = (t as f32 * 0.65) as usize;
    println!("repaint span: frames {}..{} of {} ({:.2}s total)", start, end, t, dur);

    let ce = pipeline.condition_encoder.as_ref().unwrap();
    let sil = ce.silence_latent.to_dtype(pipeline.dtype)?;
    let src_f32: Vec<f32> = src_fm.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let sil_f32: Vec<f32> = sil.narrow(1, 0, t)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;

    // chunk_mask [1,T,64] (1 inside the repaint span) and src with the span silenced.
    let mut src_vec = src_f32.clone();
    let mut cm: Vec<f32> = vec![0.0; t];
    let mut rm: Vec<f32> = vec![0.0; t];
    for k in start..end {
        cm[k] = 1.0;
        rm[k] = 1.0;
        for c in 0..64 { src_vec[k * 64 + c] = sil_f32[k * 64 + c]; }
    }
    let src = Tensor::from_vec(src_vec, (1, t, 64), &pipeline.device)?.to_dtype(pipeline.dtype)?;
    let chunk = Tensor::from_vec(cm, (1, t, 1), &pipeline.device)?
        .broadcast_as((1, t, 64))?.to_dtype(pipeline.dtype)?;
    let context = Tensor::cat(&[&src, &chunk], 2)?; // [1,T,128]
    let repaint_mask = Tensor::from_vec(rm, (1, t), &pipeline.device)?.to_dtype(pipeline.dtype)?;

    let instruction = acestep_tasks::generate_instruction("repaint", None, None);
    // No `reference_audio` for repaint -> the reference uses the silence latent for the timbre encoder.
    let (cond, _ctx) = pipeline.build_conditioning_ex(&instruction, caption, lyrics, language, dur, t, None, None)?;
    let null = ce.null_condition_emb.to_dtype(pipeline.dtype)?.broadcast_as(cond.dims())?;

    let t_schedule: Vec<f64> = (0..steps).map(|i| 1.0 - i as f64 / steps as f64).collect();
    let noise = seeded_randn(t, 64, 0.0, 1.0, seed, pipeline.dtype, &pipeline.device)?;
    let cfg = FlowMatchConfig {
        condition: &cond,
        null_condition: Some(&null),
        condition_non_cover: None,
        null_condition_non_cover: None,
        context_latents: &context,
        context_latents_non_cover: None,
        noise: &noise,
        t_schedule: &t_schedule,
        guidance_scale: guidance,
        cover_strength: 1.0,
        cover_noise_strength: 0.0,
        clean_src: Some(&src_fm),
        repaint_mask: Some(&repaint_mask),
        repaint_injection_ratio: 0.5,
        repaint_crossfade_frames: 10,
    };
    let latents = pipeline.transformer.flow_match(&cfg)?;

    if std::env::var("ACESTEP_DUMP_REPAINT").is_ok() {
        use safetensors::tensor::{serialize_to_file, TensorView};
        let mut owned: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
        let dump: Vec<(&str, &Tensor)> = vec![
            ("context_latents", &context),
            ("repaint_mask", &repaint_mask),
            ("noise", &noise),
            ("condition", &cond),
            ("clean_src", &src_fm),
        ];
        let final_fm = latents.transpose(1, 2)?.contiguous()?;
        let dump_all: Vec<(&str, &Tensor)> = dump.into_iter().chain([("final_latents", &final_fm)]).collect();
        for (n, t) in dump_all {
            let shape = t.dims().to_vec();
            let v: Vec<f32> = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            owned.push((n.to_string(), shape, bytes));
        }
        let views: Vec<(&str, TensorView)> = owned
            .iter()
            .map(|(n, s, b)| (n.as_str(), TensorView::new(safetensors::Dtype::F32, s.clone(), b.as_slice()).unwrap()))
            .collect();
        std::fs::create_dir_all("outputs/audio_ref")?;
        serialize_to_file(
            views,
            &None,
            std::path::Path::new("outputs/audio_ref/repaint_ref.safetensors"),
        )?;
        println!("dumped outputs/audio_ref/repaint_ref.safetensors");
    }

    let out = pipeline.decode(&latents)?;
    if let Some(dir) = std::path::Path::new(&out_file).parent() { std::fs::create_dir_all(dir)?; }
    out.save_auto(&out_file)?;
    println!("saved {out_file} ({:.2}s)", out.duration_seconds());
    Ok(())
}
