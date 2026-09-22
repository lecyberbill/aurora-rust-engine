// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: OGG/MP3/WAV encoder smoke test on a stereo test tone

use aurora_rust_engine::audio::{AudioFormat, WavAudio};

fn main() -> anyhow::Result<()> {
    let sr = 48_000u32;
    let secs = 2.0f32;
    let frames = (sr as f32 * secs) as usize;
    let mut samples = Vec::with_capacity(frames * 2);
    for i in 0..frames {
        let t = i as f32 / sr as f32;
        let l = 0.4 * (2.0 * std::f32::consts::PI * 440.0 * t).sin();
        let r = 0.4 * (2.0 * std::f32::consts::PI * 660.0 * t).sin();
        samples.push(l);
        samples.push(r);
    }
    let audio = WavAudio::new(samples, sr, 2);

    std::fs::create_dir_all("outputs/audio_showcase")?;
    let formats = {
        #[cfg(feature = "mp3")]
        {
            vec![AudioFormat::Wav, AudioFormat::Ogg, AudioFormat::Mp3]
        }
        #[cfg(not(feature = "mp3"))]
        {
            vec![AudioFormat::Wav, AudioFormat::Ogg]
        }
    };
    for fmt in formats {
        let path = format!("outputs/audio_showcase/encode_test.{}", fmt.extension());
        audio.save_encoded(&path, fmt)?;
        let len = std::fs::metadata(&path)?.len();
        println!("  {:?} -> {} ({} bytes)", fmt, path, len);
    }
    println!("OK");
    Ok(())
}
