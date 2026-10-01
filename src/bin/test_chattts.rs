// [WFGY] Zone: SAFE | λ: 0.15 | Fallbacks: 0 | Action: ChatTTS Conversational Speech Synthesis CLI Test

use anyhow::Result;
use candle_core::DType;
use std::path::PathBuf;

use aurora_rust_engine::{auto_device, ChatTtsParams, ChatTtsPipeline};

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let model_dir = std::env::var("CHATTTS_DIR")
        .unwrap_or_else(|_| {
            if std::path::Path::new("D:/models/chattts").exists() {
                "D:/models/chattts".to_string()
            } else {
                "G:/models/ChatTTS".to_string()
            }
        });
    let prompt = std::env::var("PROMPT")
        .unwrap_or_else(|_| "Hello everyone! [laugh] Welcome to the pure Rust Aurora inference engine. [break_3] How are you doing today?".to_string());
    let out_file = std::env::var("OUT")
        .unwrap_or_else(|_| "output_chattts.wav".to_string());

    println!("🗣️ [Aurora ChatTTS] Initializing Conversational Engine...");
    println!("   Model Directory : {}", model_dir);
    println!("   Prompt          : {}", prompt);
    println!("   Output File     : {}", out_file);

    let device = auto_device()?;
    println!("   Device          : {:?}", device);

    let dir = PathBuf::from(&model_dir);
    let gpt_path = if dir.join("asset/gpt/model.safetensors").exists() {
        dir.join("asset/gpt/model.safetensors")
    } else if dir.join("asset/GPT.safetensors").exists() {
        dir.join("asset/GPT.safetensors")
    } else {
        dir.join("GPT.safetensors")
    };

    let dvae_path = if dir.join("asset/DVAE.safetensors").exists() {
        dir.join("asset/DVAE.safetensors")
    } else {
        dir.join("DVAE.safetensors")
    };

    let vocos_path = if dir.join("asset/Vocos.safetensors").exists() {
        dir.join("asset/Vocos.safetensors")
    } else {
        dir.join("Vocos.safetensors")
    };

    let tok_path = if dir.join("asset/tokenizer/tokenizer.json").exists() {
        dir.join("asset/tokenizer/tokenizer.json")
    } else if dir.join("asset/tokenizer.json").exists() {
        dir.join("asset/tokenizer.json")
    } else {
        dir.join("tokenizer.json")
    };

    if !gpt_path.exists() || !dvae_path.exists() || !vocos_path.exists() || !tok_path.exists() {
        println!("⚠️ Model files not found in {}. Running in mock/dry-run test mode...", model_dir);
        println!("   Required files: GPT.safetensors, DVAE.safetensors, Vocos.safetensors, tokenizer.json");
        return Ok(());
    }

    let mut pipeline = ChatTtsPipeline::from_files(
        &gpt_path,
        &dvae_path,
        &vocos_path,
        &tok_path,
        None,
        device,
        DType::F32,
    )?;

    println!("✨ [Aurora ChatTTS] Generating speech with prosodic markers...");
    let params = ChatTtsParams {
        prompt,
        speaker_seed: Some(42),
        temperature: 0.7,
        max_steps: 1024,
        seed: 42,
    };

    let start = std::time::Instant::now();
    let audio = pipeline.synthesize(params)?;
    let elapsed = start.elapsed();

    audio.save_wav(&out_file)?;
    println!("✅ [Aurora ChatTTS] Speech generated in {:.2}s -> saved to {}", elapsed.as_secs_f64(), out_file);

    Ok(())
}
