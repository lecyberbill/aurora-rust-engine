// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: CLI runner for CausalLM Text Generation via AutoModel facade

use std::path::PathBuf;
use std::time::Instant;
use candle_core::{DType, Device};
use aurora_rust_engine::models::{AnyModel, AutoModel, GenerationModel, downcast_model};
use aurora_rust_engine::models::text::TextModel;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        println!("================================================================================");
        println!("🚀 Aurora Rust Engine — CausalLM Text-to-Text Generator (Pure Rust)");
        println!("================================================================================");
        println!("Usage: test_text_gen <model_path_or_gguf> [prompt] [max_tokens] [temperature]");
        println!();
        println!("Examples:");
        println!(r#"  test_text_gen "E:\LMSTUDIO_MODELES\lmstudio-community\Qwen3.5-2B-GGUF\Qwen3.5-2B-Q4_K_M.gguf" "Explain quantum computing in 3 sentences." 128 0.7"#);
        println!(r#"  test_text_gen "E:\LMSTUDIO_MODELES\lmstudio-community\DeepSeek-R1-Distill-Llama-8B-GGUF\DeepSeek-R1-Distill-Llama-8B-Q8_0.gguf" "Solve: what is 25 * 34?" 256 0.6"#);
        println!("================================================================================");
        return Ok(());
    }

    let model_path = PathBuf::from(&args[1]);
    let prompt = if args.len() > 2 {
        args[2].clone()
    } else {
        "Explain what makes Rust a modern, memory-safe language in 3 bullet points:".to_string()
    };
    let max_tokens: usize = if args.len() > 3 {
        args[3].parse().unwrap_or(128)
    } else {
        128
    };
    let temperature: f64 = if args.len() > 4 {
        args[4].parse().unwrap_or(0.7)
    } else {
        0.7
    };

    println!("================================================================================");
    println!("🚀 Aurora Rust Engine — CausalLM Text Generation");
    println!("================================================================================");
    println!("📦 Loading Model: {}", model_path.display());
    println!("💬 Prompt: \"{}\"", prompt);
    println!("⚙️  Params: max_tokens = {}, temp = {}", max_tokens, temperature);

    let device = if candle_core::utils::cuda_is_available() {
        println!("⚡ Device: NVIDIA CUDA (GPU)");
        Device::new_cuda(0)?
    } else {
        println!("🖥️  Device: CPU");
        Device::Cpu
    };

    let start_load = Instant::now();
    let archive = if model_path.is_dir() {
        std::sync::Arc::new(aurora_rust_engine::weights::SafeTensorsArchive::open_shards_dir(&model_path)?) as std::sync::Arc<dyn aurora_rust_engine::weights::WeightsSource>
    } else if model_path.to_string_lossy().to_lowercase().ends_with(".gguf") {
        std::sync::Arc::new(aurora_rust_engine::gguf::GgufWeights::open(&model_path)?) as std::sync::Arc<dyn aurora_rust_engine::weights::WeightsSource>
    } else {
        std::sync::Arc::new(aurora_rust_engine::weights::SafeTensorsArchive::open(&model_path)?) as std::sync::Arc<dyn aurora_rust_engine::weights::WeightsSource>
    };
    println!("🔍 Sample tensor keys in GGUF (first 20):");
    for k in archive.keys().iter().take(20) {
        println!("   - {}", k);
    }

    let boxed_model = AutoModel::from_local(&model_path, device, DType::F16)?;
    let load_time = start_load.elapsed();
    println!("✅ Model loaded in {:.2}s", load_time.as_secs_f64());

    let mut model_lock = boxed_model.lock().unwrap();
    let text_model = downcast_model::<TextModel>(&mut *model_lock)
        .ok_or_else(|| anyhow::anyhow!("Failed to downcast model to TextModel"))?;

    println!("🎯 Detected Architecture: {}", text_model.family());
    println!("--------------------------------------------------------------------------------");
    println!("🤖 Generating text...");

    let start_gen = Instant::now();
    let generated = text_model.generate(&prompt, max_tokens, temperature)?;
    let gen_time = start_gen.elapsed();

    println!("--------------------------------------------------------------------------------");
    println!("📝 Generated Output:");
    println!("{}", generated);
    println!("================================================================================");
    let tok_per_sec = (max_tokens as f64) / gen_time.as_secs_f64().max(0.001);
    println!("⏱️  Generation completed in {:.2}s (~{:.2} tok/s)", gen_time.as_secs_f64(), tok_per_sec);
    println!("================================================================================");

    Ok(())
}
