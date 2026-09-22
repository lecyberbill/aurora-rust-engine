// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: ACE-Step 5Hz LM planner smoke test (caption -> CoT metadata / lyrics)

use std::path::Path;
use std::time::Instant;

use aurora_rust_engine::models::AceStepLm;
use candle_core::{DType, Device};

fn main() -> anyhow::Result<()> {
    let argv: Vec<String> = std::env::args().collect();
    let mut model_dir = "G:/models/Audio/Ace-Step1.5/acestep-5Hz-lm-1.7B".to_string();
    let mut caption = "acoustic pop, warm uplifting, 120 bpm, acoustic guitar, piano, drums".to_string();
    let mut lyrics = "[verse]\nShining like the morning sun".to_string();
    let mut max_tokens = 256usize;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model-dir" | "-m" => model_dir = argv.get(i + 1).cloned().unwrap_or(model_dir),
            "--caption" | "-c" => caption = argv.get(i + 1).cloned().unwrap_or(caption),
            "--lyrics" | "-l" => lyrics = argv.get(i + 1).cloned().unwrap_or(lyrics),
            "--max-tokens" => max_tokens = argv.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(max_tokens),
            _ => {}
        }
        i += 1;
    }

    println!("================================================================================");
    println!("🎼 ACE-Step 5Hz LM planner (Qwen3)");
    println!("   Model : {}", model_dir);
    println!("   Caption: {}", caption);
    println!("================================================================================");

    let device = if candle_core::utils::cuda_is_available() {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };
    let dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };

    let t = Instant::now();
    let mut lm = AceStepLm::from_dir(Path::new(&model_dir), &device, dtype)?;
    println!("   Loaded in {:.2}s", t.elapsed().as_secs_f32());

    let prompt = AceStepLm::build_formatted_prompt(&caption, &lyrics);
    println!("--- prompt ---\n{}", prompt);

    let t = Instant::now();
    let out = lm.plan(&caption, &lyrics, max_tokens)?;
    println!("--- generated ({:.2}s) ---\n{}", t.elapsed().as_secs_f32(), out);

    Ok(())
}
