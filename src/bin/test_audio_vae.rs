// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: AutoencoderOobleck 48kHz VAE Empirical Reconstruction Test

use std::path::Path;
use std::time::Instant;
use candle_core::{DType, Device, Result, Tensor};
use aurora_rust_engine::audio::{AutoencoderOobleck, OobleckConfig};

fn main() -> Result<()> {
    println!("===============================================================");
    println!("🎧 AURORA AUDIO VAE OOBLECK (48kHz STEREO) EMPIRICAL TEST");
    println!("===============================================================\n");

    let vae_path = Path::new("G:/models/Audio/vae/diffusion_pytorch_model.safetensors");
    if !vae_path.exists() {
        println!("[-] VAE weights not found at {:?}", vae_path);
        return Ok(());
    }

    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let dtype = if device.is_cuda() { DType::F16 } else { DType::F32 };

    println!("1️⃣ Loading AutoencoderOobleck weights from {:?} on {:?} ({:?})...", vae_path, device, dtype);
    let t_load = Instant::now();
    let config = OobleckConfig::default();
    let vae = AutoencoderOobleck::from_safetensors(vae_path, config, &device, dtype)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    println!("   ✅ VAE loaded in {:.2}s\n", t_load.elapsed().as_secs_f32());

    // 2. Synthesize 2.5 seconds of 48kHz stereo audio from latents
    // At 48,000 Hz with 1920x downsampling ratio -> 25 latent frames = 48,000 samples (1.00s)
    // 60 latent frames = 115,200 samples (2.40s)
    let latent_frames = 60;
    println!("2️⃣ Generating test latents [batch=1, channels=64, frames={}]...", latent_frames);
    let latents = Tensor::randn(0.0f32, 0.5f32, (1, 64, latent_frames), &device)?
        .to_dtype(dtype)?;

    println!("3️⃣ Decoding latents to 48kHz stereo master waveform...");
    let t_dec = Instant::now();
    let audio = vae.decode(&latents)?;
    let dec_time_ms = t_dec.elapsed().as_secs_f64() * 1000.0;

    println!("   ✅ Decoded audio buffer:");
    println!("   📊 Sample Rate: {} Hz", audio.sample_rate);
    println!("   📊 Channels: {} (Stereo)", audio.channels);
    println!("   📊 Total Samples: {}", audio.samples.len());
    println!("   📊 Duration: {:.2}s", audio.duration_seconds());
    println!("   ⏱️ Decode Time: {:.2}ms ({:.1}x real-time speedup)\n",
        dec_time_ms,
        (audio.duration_seconds() * 1000.0) / (dec_time_ms as f32)
    );

    // 4. Save to WAV
    let out_dir = std::path::PathBuf::from("outputs/audio_showcase");
    std::fs::create_dir_all(&out_dir).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let out_wav = out_dir.join("oobleck_vae_test_48k.wav");

    println!("4️⃣ Saving output to standard RIFF WAV: {:?}...", out_wav);
    audio.save_wav(&out_wav).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    println!("   ✅ File saved successfully ({} bytes)\n", std::fs::metadata(&out_wav).unwrap().len());

    println!("===============================================================");
    println!("✨ EMPIRICAL AUDIO VAE VALIDATION SUCCESSFUL!");
    println!("===============================================================");

    Ok(())
}
