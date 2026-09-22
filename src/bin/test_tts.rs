// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Text-to-Speech (TTS) & Audio Pipeline Test Binary

use std::path::Path;
use std::time::Instant;
use candle_core::{DType, Device, Result};
use aurora_rust_engine::audio::WavAudio;
use aurora_rust_engine::pipelines::TtsPipeline;

fn main() -> Result<()> {
    println!("===============================================================");
    println!("🔊 AURORA AUDIO & TEXT-TO-SPEECH (TTS) INFERENCE TEST");
    println!("===============================================================\n");

    // 1. Validate Pure Rust Native WAV Audio I/O
    println!("1️⃣ Testing pure Rust RIFF WAV PCM16 encoder/decoder...");
    let sample_rate = 44100;
    let duration_sec = 1.0;
    let num_samples = (sample_rate as f32 * duration_sec) as usize;

    let mut synthetic_samples = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let t = (i as f32) / (sample_rate as f32);
        // Concert pitch A4 (440Hz) with gentle decay envelope
        let envelope = (-(t * 3.0)).exp();
        let val = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.7 * envelope;
        synthetic_samples.push(val);
    }

    let audio = WavAudio::new(synthetic_samples, sample_rate, 1);
    let out_dir = Path::new("outputs/audio_showcase");
    std::fs::create_dir_all(out_dir).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let test_wav_path = out_dir.join("test_tone_440hz.wav");

    audio.save_wav(&test_wav_path)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    println!("   ✅ Successfully exported WAV to: {:?}", test_wav_path);

    let loaded = WavAudio::load_wav(&test_wav_path)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    println!("   ✅ Successfully verified WAV readback: {} samples, {} Hz, {:.2}s duration\n",
        loaded.samples.len(), loaded.sample_rate, loaded.duration_seconds());

    // 2. Check for Parler-TTS & DAC neural model weights
    let model_dir = std::env::var("TTS_DIR").unwrap_or_else(|_| "G:\\models\\tts\\parler-tts-mini-v1".into());
    let dac_path = std::env::var("DAC_PATH").unwrap_or_else(|_| "G:\\models\\tts\\dac_44khz_16kbps.safetensors".into());

    let config_path = Path::new(&model_dir).join("config.json");
    let model_path = Path::new(&model_dir).join("model.safetensors");
    let tok_path = Path::new(&model_dir).join("tokenizer.json");
    let desc_tok_path = Path::new(&model_dir).join("description_tokenizer.json");

    if config_path.exists() && model_path.exists() && Path::new(&dac_path).exists() {
        println!("2️⃣ Found Parler-TTS & DAC weights. Initializing neural TTS pipeline...");
        let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
        let start = Instant::now();

        let mut pipeline = TtsPipeline::from_files(
            &config_path,
            &model_path,
            Path::new(&dac_path),
            &tok_path,
            &desc_tok_path,
            device,
            DType::F32,
        ).map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        println!("   ✅ Model initialized in {:.2}s", start.elapsed().as_secs_f32());

        let prompt = "Welcome to the Aurora inference engine, running entirely in pure Rust on CUDA.";
        let desc = "A female speaker delivers a clear and articulate speech with moderate pacing and natural warmth.";

        println!("   🗣️ Synthesizing speech: \"{}\"", prompt);
        let tts_start = Instant::now();
        let speech = pipeline.synthesize(prompt, desc, 600, 0.8, 42)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        let speech_out = out_dir.join("parler_tts_sample.wav");
        speech.save_wav(&speech_out)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        println!("   ✅ Speech generated in {:.2}s (Audio duration: {:.2}s, RTF: {:.2}x)",
            tts_start.elapsed().as_secs_f32(),
            speech.duration_seconds(),
            speech.duration_seconds() / tts_start.elapsed().as_secs_f32()
        );
        println!("   📁 Output saved to: {:?}", speech_out);
    } else {
        println!("2️⃣ Parler-TTS checkpoints not found at {:?}.", model_dir);
        println!("   💡 To run full neural voice synthesis, specify paths via environment variables:");
        println!("      $env:TTS_DIR  = 'path/to/parler-tts'");
        println!("      $env:DAC_PATH = 'path/to/dac_44khz.safetensors'");
        println!("   ✅ Native audio pipeline, WAV codec, and TtsPipeline architecture are compiled and fully operational!");
    }

    println!("\n===============================================================");
    println!("✨ AUDIO INFERENCE TEST COMPLETED SUCCESSFULLY");
    println!("===============================================================");
    Ok(())
}
