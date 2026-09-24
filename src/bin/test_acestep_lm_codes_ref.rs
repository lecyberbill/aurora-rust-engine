// [WFGY] Zone: SAFE | λ: 0.15 | Fallbacks: 0 | Action: Greedy audio-code generation (fixed CoT) for reference comparison

use aurora_rust_engine::models::AceStepLm;
use candle_core::{DType, Device};

const CAPTION: &str = "warm uplifting acoustic pop, acoustic guitar, piano, drums, smooth vocal melody";
const LYRICS: &str = "[verse]\nShining like the morning sun, a brand new melody has just begun";
const COT: &str = "<think>\nbpm: 94\ncaption: warm uplifting acoustic pop.\nduration: 6\nkeyscale: C major\nlanguage: en\ntimesignature: 4\n</think>";

fn main() -> anyhow::Result<()> {
    let dev = if candle_core::utils::cuda_is_available() {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };
    let dtype = if dev.is_cuda() { DType::BF16 } else { DType::F32 };

    let mut lm = AceStepLm::from_dir("G:/models/Audio/Ace-Step1.5/acestep-5Hz-lm-1.7B", &dev, dtype)?;
    let (map, set) = lm.audio_code_map();
    println!("allowed audio-code tokens: {}", set.len());

    let prompt = AceStepLm::build_codes_prompt(CAPTION, LYRICS, COT);
    let ids = lm
        .pipeline
        .generate_ids(&prompt, 32, 0.0, 1.0, Some(&set), 0)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let codes: Vec<u32> = ids.iter().filter_map(|id| map[*id as usize]).collect();
    println!("RUST codes: {:?}", codes);

    Ok(())
}
