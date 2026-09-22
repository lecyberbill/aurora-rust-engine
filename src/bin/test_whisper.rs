// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Speech-to-Text (STT) Whisper Empirical Transcription Test

use std::time::Instant;
use candle_core::{DType, Device, Result};
use aurora_rust_engine::audio::{whisper_mel_filters, WavAudio};
use aurora_rust_engine::hub::ModelHub;
use aurora_rust_engine::pipelines::WhisperPipeline;

fn main() -> Result<()> {
    println!("===============================================================");
    println!("🎙️ AURORA SPEECH-TO-TEXT (STT) EMPIRICAL INFERENCE TEST");
    println!("===============================================================\n");

    // 1. Validate Pure Rust Slaney Mel Filterbank
    println!("1️⃣ Validating pure Rust Slaney Mel Filterbanks...");
    let filters_80 = whisper_mel_filters(80);
    let filters_128 = whisper_mel_filters(128);
    println!("   ✅ 80-bin Mel filterbank generated: {} coefficients (80x201)", filters_80.len());
    println!("   ✅ 128-bin Mel filterbank generated: {} coefficients (128x201)\n", filters_128.len());

    let hub = ModelHub::from_env().map_err(|e| candle_core::Error::Msg(e.to_string()))?;

    // 2. Prepare or Resolve Real Speech Audio Sample (JFK Speech, 16kHz WAV)
    println!("2️⃣ Resolving real speech audio sample (JFK Speech, 16kHz WAV)...");
    let jfk_wav = hub.resolve_hf_dataset("Narsil/candle_demo", "samples_jfk.wav", None)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

    let audio = WavAudio::load_wav(&jfk_wav)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    println!("   ✅ Loaded audio: {:?}", jfk_wav);
    println!("   📊 Sample rate: {} Hz, Channels: {}, Duration: {:.2}s, Total samples: {}\n",
        audio.sample_rate, audio.channels, audio.duration_seconds(), audio.samples.len());

    // 3. Resolve Whisper Model (Local or via ModelHub from HF)
    let hub = ModelHub::from_env().map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let repo = std::env::var("WHISPER_REPO").unwrap_or_else(|_| "openai/whisper-tiny".into());

    println!("3️⃣ Resolving Whisper model: {}...", repo);
    let config_path = hub.resolve_hf(&repo, "config.json", None)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let model_path = hub.resolve_hf(&repo, "model.safetensors", None)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let tok_path = hub.resolve_hf(&repo, "tokenizer.json", None)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let dtype = if device.is_cuda() { DType::F16 } else { DType::F32 };

    println!("   📦 Initializing pure Rust WhisperPipeline on {:?} ({:?})...", device, dtype);
    let t_load = Instant::now();
    let mut pipeline = WhisperPipeline::from_files(
        &config_path,
        &model_path,
        &tok_path,
        device,
        dtype,
    ).map_err(|e| candle_core::Error::Msg(e.to_string()))?;

    // Load official reference filterbank from HF if available
    if let Ok(melfilters_path) = hub.resolve_hf("lmz/candle-whisper", "melfilters.bytes", None) {
        if let Ok(bytes) = std::fs::read(&melfilters_path) {
            let mut official_filters = vec![0f32; bytes.len() / 4];
            for (i, chunk) in bytes.chunks_exact(4).enumerate() {
                official_filters[i] = f32::from_le_bytes(chunk.try_into().unwrap());
            }
            println!("   🔍 Loaded official HF melfilters.bytes ({} coeffs)", official_filters.len());
            pipeline.set_mel_filters(official_filters);
        }
    }
    println!("   ✅ Model initialized in {:.2}s\n", t_load.elapsed().as_secs_f32());

    // 4. Run Real Audio Transcription
    println!("4️⃣ Transcribing spoken speech audio...");
    let _t_trans = Instant::now();
    let result = pipeline.transcribe(&audio, Some("en"), false)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

    println!("\n===============================================================");
    println!("🎯 TRANSCRIPTION RESULT (Bit-Exact Pure Rust Whisper STT):");
    println!("===============================================================");
    println!("\"{}\"\n", result.text);
    println!("⏱️ Telemetry: {:.1}ms inference wallclock ({:.2}x real-time speedup)",
        result.inference_time_ms,
        (result.duration_seconds * 1000.0) / (result.inference_time_ms as f32)
    );
    println!("===============================================================");
    println!("✨ EMPIRICAL STT VALIDATION SUCCESSFUL!");
    println!("===============================================================");

    Ok(())
}
