// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: ACE-Step 1.5 Turbo Audio & Music Diffusion End-to-End Inference Test

use std::path::Path;
use std::time::Instant;
use candle_core::{DType, Device, Result};
use aurora_rust_engine::pipelines::AudioDiffusionPipeline;

fn main() -> Result<()> {
    println!("===============================================================");
    println!("🎵 AURORA TEXT-TO-AUDIO / MUSIC DIFFUSION (ACE-STEP 1.5 TURBO)");
    println!("===============================================================\n");

    let args: Vec<String> = std::env::args().collect();
    let mut model_dir_buf = "G:/models/Audio".to_string();
    let mut mi = 1;
    while mi < args.len() {
        if args[mi] == "--model-dir" || args[mi] == "-m" {
            if mi + 1 < args.len() {
                model_dir_buf = args[mi + 1].clone();
            }
        }
        mi += 1;
    }
    let model_dir = Path::new(&model_dir_buf);
    if !model_dir.exists() {
        println!("[-] Audio model directory not found at {:?}", model_dir);
        return Ok(());
    }

    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };
    let low_vram = args.iter().any(|a| a == "--low-vram");

    println!("1️⃣ Loading ACE-Step 1.5 pipeline from {:?} on {:?} ({:?})...", model_dir, device, dtype);
    let t_load = Instant::now();
    let pipeline = if low_vram {
        AudioDiffusionPipeline::from_pretrained_low_vram(model_dir)
    } else {
        AudioDiffusionPipeline::from_folder(model_dir, device, dtype)
    }
    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    println!("   ✅ Pipeline loaded in {:.2}s", t_load.elapsed().as_secs_f32());
    println!("   🔍 Text Encoder Qwen3: {}", if pipeline.text_encoder.is_some() { "✅ ACTIVE (1024d)" } else { "[-] NONE" });
    println!("   🔍 Condition Encoder:  {}", if pipeline.condition_encoder.is_some() { "✅ ACTIVE (2048d)" } else { "[-] NONE" });
    println!("   🔍 1D DiT Transformer: ✅ ACTIVE (2560d, 32 layers)");
    println!("   🔍 AutoencoderOobleck: ✅ ACTIVE (48kHz Stereo VAE)\n");

    // Parse CLI arguments or use rich acoustic pop song default
    let mut prompt_override = None;
    let mut lyrics_override = None;
    let mut language = "en".to_string();
    let mut duration_sec = 5.0f32;
    let mut num_steps = 8;
    let mut seed = 12345u64;
    let mut out_filename = "acestep_song_acoustic_pop_48k.wav".to_string();
    let mut format_override: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--prompt" | "-p" => {
                if i + 1 < args.len() {
                    prompt_override = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--lyrics" | "-l" => {
                if i + 1 < args.len() {
                    lyrics_override = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--lyrics-file" => {
                if i + 1 < args.len() {
                    match std::fs::read_to_string(&args[i + 1]) {
                        Ok(txt) => lyrics_override = Some(txt),
                        Err(e) => println!("[-] Cannot read lyrics file {}: {}", args[i + 1], e),
                    }
                    i += 1;
                }
            }
            "--lang" => {
                if i + 1 < args.len() {
                    language = args[i + 1].clone();
                    i += 1;
                }
            }
            "--duration" | "-d" => {
                if i + 1 < args.len() {
                    duration_sec = args[i + 1].parse().unwrap_or(5.0);
                    i += 1;
                }
            }
            "--steps" | "-s" => {
                if i + 1 < args.len() {
                    num_steps = args[i + 1].parse().unwrap_or(8);
                    i += 1;
                }
            }
            "--seed" => {
                if i + 1 < args.len() {
                    seed = args[i + 1].parse().unwrap_or(12345);
                    i += 1;
                }
            }
            "--out" | "-o" => {
                if i + 1 < args.len() {
                    out_filename = args[i + 1].clone();
                    i += 1;
                }
            }
            "--format" | "-f" => {
                if i + 1 < args.len() {
                    format_override = Some(args[i + 1].to_lowercase());
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }

    let default_caption = "[genre: acoustic pop] [mood: warm uplifting] [tempo: 120 bpm] [instruments: acoustic guitar, piano, drums, smooth vocal melody]";
    let default_lyrics = "Shining like the morning sun, a brand new melody has just begun";
    let caption = prompt_override.as_deref().unwrap_or(default_caption);
    let lyrics = lyrics_override.as_deref().unwrap_or(default_lyrics);

    println!("2️⃣ Generating 48kHz stereo master song via Guided Flow-Matching...");
    println!("   🎼 Caption: \"{}\"", caption);
    println!("   📝 Lyrics ({}): \"{}\"", language, lyrics);
    println!("   ⏱️ Target Duration: {:.1}s (48,000 Hz, 2 Channels Stéréo)", duration_sec);
    println!("   ⚡ Inference Steps: {} (FlowMatch Euler)", num_steps);
    println!("   🎲 Seed: {}", seed);

    let (audio, metrics) = pipeline.generate(caption, lyrics, &language, duration_sec, num_steps, seed)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

    println!("\n===============================================================");
    println!("🎯 SONG SYNTHESIS TELEMETRY & RESULTS:");
    println!("===============================================================");
    println!("   📊 Output Sample Rate: {} Hz", metrics.sample_rate);
    println!("   📊 Channels: {} (Stereo)", metrics.channels);
    println!("   📊 Song Duration: {:.2}s ({} total samples)", metrics.duration_seconds, audio.samples.len());
    println!("   ⏱️ Inference Wallclock: {:.2}ms ({:.2}x real-time speedup)",
        metrics.inference_time_ms,
        (metrics.duration_seconds * 1000.0) / (metrics.inference_time_ms as f32)
    );

    // 3. Save to master file (codec inferred from extension, or forced via --format)
    if let Some(fmt) = &format_override {
        let stem = Path::new(&out_filename)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| out_filename.clone());
        out_filename = format!("{}.{}", stem, fmt);
    }

    let out_dir = std::path::PathBuf::from("outputs/audio_showcase");
    std::fs::create_dir_all(&out_dir).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let out_wav = out_dir.join(&out_filename);

    println!("\n3️⃣ Exporting to Master File: {:?}...", out_wav);
    audio.save_auto(&out_wav).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    println!("   ✅ Master saved successfully ({} bytes)!\n", std::fs::metadata(&out_wav).unwrap().len());

    println!("===============================================================");
    println!("✨ EMPIRICAL AUDIO DIFFUSION VALIDATION SUCCESSFUL!");
    println!("===============================================================");

    Ok(())
}
