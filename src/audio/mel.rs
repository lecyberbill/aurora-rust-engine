// [WFGY] Zone: SAFE | λ: 0.15 | Fallbacks: 0 | Action: Pure Rust Slaney Mel Filterbank & Spectrogram Generator for Whisper


/// Generate the standard triangular Mel filterbank matrix matching OpenAI Whisper / librosa (Slaney format).
///
/// Returns a flat vector of size `n_mels * (1 + n_fft / 2)` (row-major).
pub fn slaney_mel_filterbank(
    sample_rate: usize,
    n_fft: usize,
    n_mels: usize,
    f_min: f32,
    f_max: f32,
) -> Vec<f32> {
    let n_freq_bins = 1 + n_fft / 2; // 201 for n_fft=400
    let mut weights = vec![0.0f32; n_mels * n_freq_bins];

    // Slaney frequency-to-mel conversion
    let hz_to_mel = |hz: f32| -> f32 {
        let min_log_hz = 1000.0f32;
        let min_log_mel = 15.0f32;
        let logstep = 27.0f32 / 6.4f32.ln();

        if hz >= min_log_hz {
            min_log_mel + (hz / min_log_hz).ln() * logstep
        } else {
            3.0 * hz / 200.0
        }
    };

    // Slaney mel-to-frequency conversion
    let mel_to_hz = |mel: f32| -> f32 {
        let min_log_hz = 1000.0f32;
        let min_log_mel = 15.0f32;
        let logstep = 6.4f32.ln() / 27.0f32;

        if mel >= min_log_mel {
            min_log_hz * ((mel - min_log_mel) * logstep).exp()
        } else {
            200.0 * mel / 3.0
        }
    };

    let min_mel = hz_to_mel(f_min);
    let max_mel = hz_to_mel(f_max);

    let mel_points: Vec<f32> = (0..=(n_mels + 1))
        .map(|i| min_mel + (max_mel - min_mel) * (i as f32) / ((n_mels + 1) as f32))
        .collect();

    let f_points: Vec<f32> = mel_points.into_iter().map(mel_to_hz).collect();

    // Center frequencies of FFT bins
    let fft_freqs: Vec<f32> = (0..n_freq_bins)
        .map(|i| (i as f32) * (sample_rate as f32) / (n_fft as f32))
        .collect();

    for m in 0..n_mels {
        let f_left = f_points[m];
        let f_center = f_points[m + 1];
        let f_right = f_points[m + 2];

        // Slaney area normalization factor: 2.0 / (f_right - f_left)
        let enorm = 2.0 / (f_right - f_left);

        for (k, &f) in fft_freqs.iter().enumerate() {
            if f >= f_left && f <= f_center {
                let weight = (f - f_left) / (f_center - f_left);
                weights[m * n_freq_bins + k] = weight * enorm;
            } else if f > f_center && f <= f_right {
                let weight = (f_right - f) / (f_right - f_center);
                weights[m * n_freq_bins + k] = weight * enorm;
            }
        }
    }

    weights
}

/// Compute 80-bin or 128-bin Mel filterbank specifically for OpenAI Whisper (16kHz audio, n_fft=400, f_min=0, f_max=8000).
pub fn whisper_mel_filters(n_mels: usize) -> Vec<f32> {
    slaney_mel_filterbank(16000, 400, n_mels, 0.0, 8000.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slaney_filterbank_dimensions() {
        let filters_80 = whisper_mel_filters(80);
        assert_eq!(filters_80.len(), 80 * 201);

        let filters_128 = whisper_mel_filters(128);
        assert_eq!(filters_128.len(), 128 * 201);

        // Check non-negative weights
        for &w in &filters_80 {
            assert!(w >= 0.0, "Filter weight should be non-negative");
        }

        // Peak weights should be positive
        let sum_80: f32 = filters_80.iter().sum();
        assert!(sum_80 > 0.0, "Filterbank should not be all zero");
    }
}
