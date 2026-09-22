// [WFGY] Zone: SAFE | λ: 0.15 | Fallbacks: 0 | Action: Pure Rust WAV PCM16 / IEEE-Float32 Audio Encoder & Decoder

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;
use anyhow::{bail, Context, Result};

/// Audio buffer represented as planar or interleaved normalized f32 samples in [-1.0, 1.0].
#[derive(Debug, Clone, PartialEq)]
pub struct WavAudio {
    /// Audio samples normalized to [-1.0, 1.0]. Interleaved if stereo (L, R, L, R...).
    pub samples: Vec<f32>,
    /// Sampling frequency in Hertz (e.g. 16000, 24000, 44100, 48000).
    pub sample_rate: u32,
    /// Number of audio channels (1 = mono, 2 = stereo).
    pub channels: u16,
}

impl WavAudio {
    pub fn new(samples: Vec<f32>, sample_rate: u32, channels: u16) -> Self {
        assert!(channels > 0, "Channels must be >= 1");
        assert!(sample_rate > 0, "Sample rate must be >= 1");
        Self {
            samples,
            sample_rate,
            channels,
        }
    }

    /// Duration in seconds.
    pub fn duration_seconds(&self) -> f32 {
        if self.channels == 0 || self.sample_rate == 0 {
            0.0
        } else {
            (self.samples.len() as f32) / (self.channels as f32 * self.sample_rate as f32)
        }
    }

    /// Convert interleaved multi-channel audio to mono by averaging channels.
    pub fn to_mono(&self) -> Self {
        if self.channels == 1 {
            return self.clone();
        }
        let ch = self.channels as usize;
        let num_frames = self.samples.len() / ch;
        let mut mono = Vec::with_capacity(num_frames);
        for i in 0..num_frames {
            let mut sum = 0.0f32;
            for c in 0..ch {
                sum += self.samples[i * ch + c];
            }
            mono.push(sum / (ch as f32));
        }
        Self {
            samples: mono,
            sample_rate: self.sample_rate,
            channels: 1,
        }
    }

    /// Resample audio to a target sample rate using linear band-limited interpolation.
    pub fn resample(&self, target_sample_rate: u32) -> Self {
        if self.sample_rate == target_sample_rate {
            return self.clone();
        }
        let ch = self.channels as usize;
        let src_frames = self.samples.len() / ch;
        if src_frames == 0 {
            return Self::new(Vec::new(), target_sample_rate, self.channels);
        }

        let ratio = target_sample_rate as f64 / self.sample_rate as f64;
        let dst_frames = ((src_frames as f64) * ratio).round() as usize;
        let mut dst_samples = Vec::with_capacity(dst_frames * ch);

        for dst_idx in 0..dst_frames {
            let src_pos = (dst_idx as f64) / ratio;
            let idx0 = src_pos.floor() as usize;
            let frac = (src_pos - (idx0 as f64)) as f32;
            let idx1 = (idx0 + 1).min(src_frames.saturating_sub(1));

            for c in 0..ch {
                let s0 = self.samples[idx0 * ch + c];
                let s1 = self.samples[idx1 * ch + c];
                let val = s0 + frac * (s1 - s0);
                dst_samples.push(val);
            }
        }

        Self {
            samples: dst_samples,
            sample_rate: target_sample_rate,
            channels: self.channels,
        }
    }

    /// Save the audio buffer to a 16-bit PCM standard RIFF WAV file.
    pub fn save_wav<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        write_wav_pcm16(path, &self.samples, self.sample_rate, self.channels)
    }

    /// Load a WAV file from disk.
    pub fn load_wav<P: AsRef<Path>>(path: P) -> Result<Self> {
        read_wav_pcm(path)
    }
}

/// Write 16-bit signed PCM WAV to a file.
pub fn write_wav_pcm16<P: AsRef<Path>>(
    path: P,
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
) -> Result<()> {
    let file = File::create(path.as_ref())
        .with_context(|| format!("Failed to create WAV file at {:?}", path.as_ref()))?;
    let mut writer = BufWriter::new(file);

    let bits_per_sample: u16 = 16;
    let bytes_per_sample: u16 = bits_per_sample / 8;
    let block_align: u16 = channels * bytes_per_sample;
    let byte_rate: u32 = sample_rate * (block_align as u32);
    let num_samples = samples.len();
    let data_chunk_size = (num_samples as u32) * (bytes_per_sample as u32);
    let riff_chunk_size = 36 + data_chunk_size;

    // RIFF Header
    writer.write_all(b"RIFF")?;
    writer.write_all(&riff_chunk_size.to_le_bytes())?;
    writer.write_all(b"WAVE")?;

    // fmt Subchunk
    writer.write_all(b"fmt ")?;
    writer.write_all(&16u32.to_le_bytes())?; // Subchunk1Size = 16 for PCM
    writer.write_all(&1u16.to_le_bytes())?;  // AudioFormat = 1 (PCM)
    writer.write_all(&channels.to_le_bytes())?;
    writer.write_all(&sample_rate.to_le_bytes())?;
    writer.write_all(&byte_rate.to_le_bytes())?;
    writer.write_all(&block_align.to_le_bytes())?;
    writer.write_all(&bits_per_sample.to_le_bytes())?;

    // data Subchunk
    writer.write_all(b"data")?;
    writer.write_all(&data_chunk_size.to_le_bytes())?;

    // Samples quantized to signed i16 [-32768, 32767]
    for &sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let i16_val = (clamped * 32767.0).round() as i16;
        writer.write_all(&i16_val.to_le_bytes())?;
    }

    writer.flush()?;
    Ok(())
}

/// Read a WAV file into normalized f32 samples.
pub fn read_wav_pcm<P: AsRef<Path>>(path: P) -> Result<WavAudio> {
    let file = File::open(path.as_ref())
        .with_context(|| format!("Failed to open WAV file at {:?}", path.as_ref()))?;
    let mut reader = BufReader::new(file);

    let mut riff = [0u8; 4];
    reader.read_exact(&mut riff)?;
    if &riff != b"RIFF" {
        bail!("Invalid WAV file: missing RIFF header");
    }

    let mut riff_size_bytes = [0u8; 4];
    reader.read_exact(&mut riff_size_bytes)?;

    let mut wave = [0u8; 4];
    reader.read_exact(&mut wave)?;
    if &wave != b"WAVE" {
        bail!("Invalid WAV file: missing WAVE marker");
    }

    let mut channels: u16 = 1;
    let mut sample_rate: u32 = 16000;
    let mut bits_per_sample: u16 = 16;
    let mut audio_format: u16 = 1;
    let mut fmt_found = false;

    // Iterate through chunks until 'data' is reached
    loop {
        let mut chunk_id = [0u8; 4];
        if reader.read_exact(&mut chunk_id).is_err() {
            bail!("Premature end of file while searching for chunks");
        }

        let mut chunk_size_bytes = [0u8; 4];
        reader.read_exact(&mut chunk_size_bytes)?;
        let chunk_size = u32::from_le_bytes(chunk_size_bytes);

        if &chunk_id == b"fmt " {
            if chunk_size < 16 {
                bail!("Invalid fmt chunk size: {}", chunk_size);
            }
            let mut format_bytes = [0u8; 2];
            reader.read_exact(&mut format_bytes)?;
            audio_format = u16::from_le_bytes(format_bytes);

            let mut channels_bytes = [0u8; 2];
            reader.read_exact(&mut channels_bytes)?;
            channels = u16::from_le_bytes(channels_bytes);

            let mut rate_bytes = [0u8; 4];
            reader.read_exact(&mut rate_bytes)?;
            sample_rate = u32::from_le_bytes(rate_bytes);

            let mut byte_rate_bytes = [0u8; 4];
            reader.read_exact(&mut byte_rate_bytes)?;

            let mut block_align_bytes = [0u8; 2];
            reader.read_exact(&mut block_align_bytes)?;

            let mut bits_bytes = [0u8; 2];
            reader.read_exact(&mut bits_bytes)?;
            bits_per_sample = u16::from_le_bytes(bits_bytes);

            let remaining = chunk_size - 16;
            if remaining > 0 {
                reader.seek(SeekFrom::Current(remaining as i64))?;
            }
            fmt_found = true;
        } else if &chunk_id == b"data" {
            if !fmt_found {
                bail!("Found 'data' chunk before 'fmt ' chunk");
            }

            let mut samples = Vec::new();

            match (audio_format, bits_per_sample) {
                (1, 16) => {
                    // PCM 16-bit signed
                    let num_samples = (chunk_size / 2) as usize;
                    samples.reserve(num_samples);
                    let mut buf = [0u8; 2];
                    for _ in 0..num_samples {
                        reader.read_exact(&mut buf)?;
                        let val = i16::from_le_bytes(buf);
                        samples.push((val as f32) / 32768.0);
                    }
                }
                (1, 8) => {
                    // PCM 8-bit unsigned
                    let num_samples = chunk_size as usize;
                    samples.reserve(num_samples);
                    let mut buf = [0u8; 1];
                    for _ in 0..num_samples {
                        reader.read_exact(&mut buf)?;
                        let val = buf[0] as f32;
                        samples.push((val - 128.0) / 128.0);
                    }
                }
                (3, 32) => {
                    // IEEE 32-bit Float
                    let num_samples = (chunk_size / 4) as usize;
                    samples.reserve(num_samples);
                    let mut buf = [0u8; 4];
                    for _ in 0..num_samples {
                        reader.read_exact(&mut buf)?;
                        let val = f32::from_le_bytes(buf);
                        samples.push(val);
                    }
                }
                _ => {
                    bail!(
                        "Unsupported WAV format: format={}, bits_per_sample={}",
                        audio_format,
                        bits_per_sample
                    );
                }
            }

            return Ok(WavAudio {
                samples,
                sample_rate,
                channels,
            });
        } else {
            // Skip unknown subchunks (e.g. LIST, JUNK, etc.)
            reader.seek(SeekFrom::Current(chunk_size as i64))?;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wav_write_read_roundtrip() {
        let sample_rate = 24000;
        let channels = 1;
        let duration = 0.1; // 100 ms
        let num_samples = (sample_rate as f32 * duration) as usize;

        // Generate 440 Hz test tone
        let mut original_samples = Vec::with_capacity(num_samples);
        for i in 0..num_samples {
            let t = (i as f32) / (sample_rate as f32);
            let val = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.8;
            original_samples.push(val);
        }

        let audio = WavAudio::new(original_samples.clone(), sample_rate, channels);
        let temp_dir = std::env::temp_dir();
        let test_path = temp_dir.join("aurora_test_audio.wav");

        audio.save_wav(&test_path).expect("Failed to write WAV");
        let loaded = WavAudio::load_wav(&test_path).expect("Failed to read WAV");

        let _ = std::fs::remove_file(&test_path);

        assert_eq!(loaded.sample_rate, sample_rate);
        assert_eq!(loaded.channels, channels);
        assert_eq!(loaded.samples.len(), original_samples.len());

        // Max difference due to 16-bit quantization should be < 1.0 / 32767 (~0.000031)
        for (a, b) in original_samples.iter().zip(loaded.samples.iter()) {
            let diff = (a - b).abs();
            assert!(diff < 0.0001, "Sample diff too large: {} vs {}", a, b);
        }
    }
}
