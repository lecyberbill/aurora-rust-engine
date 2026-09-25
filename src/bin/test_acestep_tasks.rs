// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: ACE-Step source-audio tasks (extract / lego / complete / repaint / cover)

use aurora_rust_engine::audio::WavAudio;
use aurora_rust_engine::pipelines::{AudioDiffusionPipeline, TaskRequest};
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
    let mut out_file = String::new();
    let mut task = "extract".to_string();
    let mut track = "vocals".to_string();
    let mut classes: Vec<String> = Vec::new();
    let mut caption =
        "warm uplifting acoustic pop, acoustic guitar, piano, drums, smooth vocal melody".to_string();
    let mut global_caption = String::new();
    let mut lyrics = "[instrumental]".to_string();
    let mut language = "fr".to_string();
    let mut steps = 50usize;
    let mut guidance = 7.0f32;
    let mut seed = 42u64;
    let mut repaint_start: Option<f32> = None;
    let mut repaint_end: Option<f32> = None;
    let mut cover_strength = 1.0f32;
    let mut cover_noise_strength = 0.0f32;
    let mut refer_src = false;
    let mut chunk_mask_value = 2.0f32;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model-dir" | "-m" => { model_dir = argv.get(i + 1).cloned().unwrap_or(model_dir); i += 1; }
            "--src" => { src_wav = argv.get(i + 1).cloned().unwrap_or(src_wav); i += 1; }
            "--out" | "-o" => { out_file = argv.get(i + 1).cloned().unwrap_or(out_file); i += 1; }
            "--task" | "-t" => { task = argv.get(i + 1).cloned().unwrap_or(task); i += 1; }
            "--track" => { track = argv.get(i + 1).cloned().unwrap_or(track); i += 1; }
            "--classes" => {
                classes = argv.get(i + 1).cloned().unwrap_or_default()
                    .split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
                i += 1;
            }
            "--caption" => { caption = argv.get(i + 1).cloned().unwrap_or(caption); i += 1; }
            "--global-caption" => { global_caption = argv.get(i + 1).cloned().unwrap_or(global_caption); i += 1; }
            "--lyrics" => { lyrics = argv.get(i + 1).cloned().unwrap_or(lyrics); i += 1; }
            "--lang" => { language = argv.get(i + 1).cloned().unwrap_or(language); i += 1; }
            "--steps" | "-s" => { steps = argv.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(steps); i += 1; }
            "--guidance" | "-g" => { guidance = argv.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(guidance); i += 1; }
            "--seed" => { seed = argv.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(seed); i += 1; }
            "--repaint-start" => { repaint_start = argv.get(i + 1).and_then(|s| s.parse().ok()); i += 1; }
            "--repaint-end" => { repaint_end = argv.get(i + 1).and_then(|s| s.parse().ok()); i += 1; }
            "--cover-strength" => { cover_strength = argv.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(cover_strength); i += 1; }
            "--cover-noise" => { cover_noise_strength = argv.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(cover_noise_strength); i += 1; }
            "--refer-src" => { refer_src = true; }
            "--chunk" => { chunk_mask_value = argv.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(chunk_mask_value); i += 1; }
            _ => {}
        }
        i += 1;
    }
    if out_file.is_empty() {
        out_file = format!("outputs/audio_showcase/task_{}.ogg", task);
    }

    let use_cpu = argv.iter().any(|a| a == "--cpu");
    let pipeline = if use_cpu {
        AudioDiffusionPipeline::from_folder(&model_dir, Device::Cpu, DType::F32)?
    } else {
        AudioDiffusionPipeline::from_pretrained(&model_dir)?
    };
    println!(
        "pipeline: variant={:?} dtype={:?} is_lego_sft={}",
        pipeline.variant, pipeline.dtype, pipeline.is_lego_sft
    );

    let audio = load_stereo(&src_wav, &pipeline.device, pipeline.dtype)?;
    let enc = pipeline.vae.encode(&audio)?; // [1,64,T]
    let num_frames = enc.dim(2)?;
    let src_fm = enc.transpose(1, 2)?.contiguous()?; // [1,T,64]
    println!(
        "source: {} latent frames ({:.2}s) task={} track={}",
        num_frames,
        num_frames as f32 / 25.0,
        task,
        track
    );

    let mut req = TaskRequest::new(&task, &src_fm)
        .with_caption(&caption)
        .with_global_caption(&global_caption)
        .with_lyrics(&lyrics)
        .with_language(&language)
        .with_steps(steps)
        .with_guidance_scale(guidance)
        .with_seed(seed)
        .with_cover_strength(cover_strength)
        .with_cover_noise_strength(cover_noise_strength);
    req.chunk_mask_value = chunk_mask_value;
    if matches!(task.as_str(), "extract" | "lego") {
        req = req.with_track_name(&track);
    }
    if task == "complete" && !classes.is_empty() {
        req = req.with_complete_track_classes(classes.clone());
    }
    if refer_src {
        req.refer_latents = Some(&src_fm);
    }
    if let (Some(s), Some(e)) = (repaint_start, repaint_end) {
        req = req.with_repaint_span(s, e);
        println!("repaint span: {}s..{}s", s, e);
    }

    let (out, metrics) = pipeline.generate_task(&req)?;
    if let Some(dir) = std::path::Path::new(&out_file).parent() { std::fs::create_dir_all(dir)?; }
    out.save_auto(&out_file)?;
    println!(
        "saved {out_file} ({:.2}s, {} steps, {:.0}ms)",
        out.duration_seconds(), metrics.num_steps, metrics.inference_time_ms
    );
    Ok(())
}
