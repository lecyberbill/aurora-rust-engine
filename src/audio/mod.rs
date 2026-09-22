// [WFGY] Zone: SAFE | λ: 0.10 | Fallbacks: 0 | Action: Audio Subsystem Module Definition

pub mod mel;
pub mod wav;

pub use mel::{slaney_mel_filterbank, whisper_mel_filters};
pub use wav::{read_wav_pcm, write_wav_pcm16, WavAudio};
