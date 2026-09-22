// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Open-source OGG Vorbis & MP3 audio encoders (libvorbis / LAME)

//! Compressed audio output codecs for the generated 48 kHz masters.
//!
//! * [`AudioFormat::Ogg`] — OGG Vorbis via `vorbis_rs` (libvorbis, BSD-3-Clause)
//! * [`AudioFormat::Mp3`] — MPEG-1 Layer III via `mp3lame-encoder` (LAME, LGPL-3.0)
//! * [`AudioFormat::Wav`] — native PCM16 WAV writer (no dependency)

use anyhow::{anyhow, Result};

/// Output container / codec for a generated master.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioFormat {
    Wav,
    Ogg,
    Mp3,
}

impl AudioFormat {
    /// Resolve a format from a (case-insensitive) file extension.
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_ascii_lowercase().as_str() {
            "ogg" | "oga" | "vorbis" => Self::Ogg,
            "mp3" => Self::Mp3,
            _ => Self::Wav,
        }
    }

    /// Resolve a format from a `"wav" | "ogg" | "mp3"` string.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "wav" => Some(Self::Wav),
            "ogg" | "oga" | "vorbis" => Some(Self::Ogg),
            "mp3" => Some(Self::Mp3),
            _ => None,
        }
    }

    pub fn extension(self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Ogg => "ogg",
            Self::Mp3 => "mp3",
        }
    }
}

/// Encode interleaved f32 samples into an OGG Vorbis stream.
pub fn encode_ogg(samples: &[f32], sample_rate: u32, channels: u16) -> Result<Vec<u8>> {
    use std::num::{NonZeroU32, NonZeroU8};
    use vorbis_rs::{VorbisBitrateManagementStrategy, VorbisEncoderBuilder};

    let sr = NonZeroU32::new(sample_rate).ok_or_else(|| anyhow!("invalid sample rate"))?;
    let ch = NonZeroU8::new(channels as u8).ok_or_else(|| anyhow!("invalid channel count"))?;

    let mut encoder = VorbisEncoderBuilder::new(sr, ch, Vec::<u8>::new())
        .map_err(|e| anyhow!("vorbis init failed: {e}"))?
        .bitrate_management_strategy(VorbisBitrateManagementStrategy::QualityVbr {
            target_quality: 0.6,
        })
        .build()
        .map_err(|e| anyhow!("vorbis build failed: {e}"))?;

    let chn = channels as usize;
    let total_frames = samples.len() / chn;
    let block = 4096usize; // ~85 ms blocks at 48 kHz
    let mut start = 0usize;
    while start < total_frames {
        let len = block.min(total_frames - start);
        let mut planes: Vec<Vec<f32>> = (0..chn).map(|_| Vec::with_capacity(len)).collect();
        for i in 0..len {
            for (c, plane) in planes.iter_mut().enumerate() {
                plane.push(samples[(start + i) * chn + c]);
            }
        }
        let refs: Vec<&[f32]> = planes.iter().map(|p| p.as_slice()).collect();
        encoder
            .encode_audio_block(&refs)
            .map_err(|e| anyhow!("vorbis encode failed: {e}"))?;
        start += len;
    }

    encoder.finish().map_err(|e| anyhow!("vorbis finish failed: {e}"))
}

/// Encode interleaved f32 samples into an MP3 (MPEG-1 Layer III, 192 kbps) stream.
///
/// Only available with the `mp3` feature (LAME, LGPL-3.0).
#[cfg(feature = "mp3")]
pub fn encode_mp3(samples: &[f32], sample_rate: u32, channels: u16) -> Result<Vec<u8>> {
    use mp3lame_encoder::{Bitrate, Builder, DualPcm, FlushNoGap, MonoPcm, Quality};

    let mut encoder = Builder::new()
        .ok_or_else(|| anyhow!("lame init failed"))?
        .with_num_channels(channels as u8)
        .map_err(|e| anyhow!("lame channel config failed: {e}"))?
        .with_sample_rate(sample_rate)
        .map_err(|e| anyhow!("lame sample-rate config failed: {e}"))?
        .with_quality(Quality::Best)
        .map_err(|e| anyhow!("lame quality config failed: {e}"))?
        .with_brate(Bitrate::Kbps192)
        .map_err(|e| anyhow!("lame bitrate config failed: {e}"))?
        .build()
        .map_err(|e| anyhow!("lame build failed: {e}"))?;

    // LAME writes into the Vec's spare capacity, so it must be pre-allocated
    // (the crate writes through `spare_capacity_mut`, which is empty for `Vec::new`).
    let frames = samples.len() / channels.max(1) as usize;
    let mut out = Vec::with_capacity(mp3lame_encoder::max_required_buffer_size(frames) + 7200);
    if channels == 2 {
        let left: Vec<f32> = samples.iter().step_by(2).copied().collect();
        let right: Vec<f32> = samples.iter().skip(1).step_by(2).copied().collect();
        encoder
            .encode_to_vec(DualPcm { left: &left, right: &right }, &mut out)
            .map_err(|e| anyhow!("mp3 encode failed: {e}"))?;
    } else {
        encoder
            .encode_to_vec(MonoPcm(samples), &mut out)
            .map_err(|e| anyhow!("mp3 encode failed: {e}"))?;
    }
    encoder
        .flush_to_vec::<FlushNoGap>(&mut out)
        .map_err(|e| anyhow!("mp3 flush failed: {e}"))?;
    Ok(out)
}
