// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: ACE-Step 1.5 Turbo Audio & Music Diffusion End-to-End Inference Test

use std::path::Path;
use std::time::Instant;
use candle_core::{DType, Device, Result};
use aurora_rust_engine::pipelines::AudioDiffusionPipeline;

fn main() -> Result<()> {
    println!("===============================================================");
    println!("🎵 AURORA TEXT-TO-AUDIO / MUSIC DIFFUSION (ACE-STEP 1.5 TURBO)");
    println!("===============================================================\n");

    let model_dir = Path::new("G:/models/Audio");
    if !model_dir.exists() {
        println!("[-] Audio model directory not found at {:?}", model_dir);
        return Ok(());
    }

    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let dtype = if device.is_cuda() { DType::F16 } else { DType::F32 };

    println!("1️⃣ Loading ACE-Step 1.5 Turbo pipeline from {:?} on {:?} ({:?})...", model_dir, device, dtype);
    let t_load = Instant::now();
    let pipeline = AudioDiffusionPipeline::from_folder(model_dir, device, dtype)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    println!("   ✅ Pipeline loaded in {:.2}s\n", t_load.elapsed().as_secs_f32());

    // 2. Generate 4 seconds of stereo 48kHz audio using 4 Flow Matching steps (Turbo)
    let prompt = "Epic cinematic orchestral soundtrack with energetic percussion, soaring brass, and 8k master clarity";
    let duration_sec = 4.0f32;
    let num_steps = 4;
    let seed = 42;

    println!("2️⃣ Generating 48kHz stereo master audio via Flow-Matching...");
    println!("   🎼 Prompt: \"{}\"", prompt);
    println!("   ⏱️ Target Duration: {:.1}s (48,000 Hz, 2 Channels)", duration_sec);
    println!("   ⚡ Inference Steps: {} (FlowMatch Euler)", num_steps);

    let (audio, metrics) = pipeline.generate(prompt, duration_sec, num_steps, seed)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

    println!("\n===============================================================");
    println!("🎯 AUDIO SYNTHESIS TELEMETRY & RESULTS:");
    println!("===============================================================");
    println!("   📊 Output Sample Rate: {} Hz", metrics.sample_rate);
    println!("   📊 Channels: {} (Stereo)", metrics.channels);
    println!("   📊 Audio Duration: {:.2}s ({} total samples)", metrics.duration_seconds, audio.samples.len());
    println!("   ⏱️ Inference Wallclock: {:.2}ms ({:.2}x real-time speedup)",
        metrics.inference_time_ms,
        (metrics.duration_seconds * 1000.0) / (metrics.inference_time_ms as f32)
    );

    // 3. Save to master WAV
    let out_dir = std::path::PathBuf::from("outputs/audio_showcase");
    std::fs::create_dir_all(&out_dir).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let out_wav = out_dir.join("acestep_music_48k_turbo.wav");

    println!("\n3️⃣ Exporting to Master WAV File: {:?}...", out_wav);
    audio.save_wav(&out_wav).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    println!("   ✅ Master WAV saved successfully ({} bytes)!\n", std::fs::metadata(&out_wav).unwrap().len());

    println!("===============================================================");
    println!("✨ EMPIRICAL AUDIO DIFFUSION VALIDATION SUCCESSFUL!");
    println!("===============================================================");

    Ok(())
}
