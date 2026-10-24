use std::path::PathBuf;
use std::time::Instant;
use candle_core::{DType, Device};
use aurora_rust_engine::models::{AnyModel, AutoModel, GenerationModel, downcast_model};
use aurora_rust_engine::models::text::{TextGenParams, TextModel};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "Usage: test_text_gen <model> [prompt] [max_tokens] [temperature] [n_gpu_layers]"
        );
        eprintln!(
            "  Quote the prompt when it contains spaces and you also pass max_tokens/temperature."
        );
        return Ok(());
    }

    let model_path = PathBuf::from(&args[1]);

    // Positional CLI: <model> <prompt> <max_tokens> <temperature> [n_gpu_layers].
    // If the trailing arguments are not numeric, treat everything as an unquoted prompt.
    let mut max_tokens = 128usize;
    let mut temperature = 0.7f64;
    let mut n_gpu_layers: Option<usize> = None;

    let prompt = if args.len() > 2 {
        let positional = args.len() >= 4
            && args[3].parse::<usize>().is_ok()
            && (args.len() < 5 || args[4].parse::<f64>().is_ok());
        if positional {
            max_tokens = args[3].parse().unwrap_or(128);
            if args.len() > 4 {
                temperature = args[4].parse().unwrap_or(0.7);
            }
            if args.len() > 5 {
                n_gpu_layers = args[5].parse().ok();
            }
            args[2].clone()
        } else {
            args[2..].join(" ")
        }
    } else {
        "Explain what makes Rust a modern, memory-safe language in 3 bullet points:".to_string()
    };

    if let Some(n) = n_gpu_layers {
        std::env::set_var("N_GPU_LAYERS", n.to_string());
    }

    println!("================================================================================");
    println!("🚀 Aurora Rust Engine — CausalLM Text Generation");
    println!("================================================================================");
    println!("📦 Loading Model: {}", model_path.display());
    println!("💬 Prompt: \"{}\"", prompt);
    println!(
        "⚙️  Params: max_tokens = {}, temp = {}, n_gpu_layers = {:?}",
        max_tokens, temperature, n_gpu_layers
    );

    let device = match aurora_rust_engine::device::auto_device() {
        Ok(d) => d,
        Err(_) => Device::Cpu,
    };

    let start_load = Instant::now();
    let boxed_model = AutoModel::from_local(&model_path, device, DType::F16)?;
    let load_time = start_load.elapsed();
    println!("✅ Model loaded in {:.2}s", load_time.as_secs_f64());

    let mut model_lock = boxed_model.lock().unwrap();
    let text_model = downcast_model::<TextModel>(&mut *model_lock)
        .ok_or_else(|| anyhow::anyhow!("Failed to downcast model to TextModel"))?;

    println!("🎯 Detected Architecture: {}", text_model.family());
    println!("--------------------------------------------------------------------------------");
    println!("🤖 Generating text...");

    let mut params = TextGenParams::default();
    params.max_tokens = max_tokens;
    params.temperature = temperature;
    if let Ok(top_p_str) = std::env::var("TOP_P") {
        if let Ok(val) = top_p_str.parse::<f64>() {
            params.top_p = val;
        }
    }
    if let Ok(q_str) = std::env::var("QUANTIZED_KV") {
        params.quantized_kv = q_str == "1" || q_str.eq_ignore_ascii_case("true");
    }
    println!("⚙️  Sampling: temp = {}, top_p = {}, top_k = {}", params.temperature, params.top_p, params.top_k);
    println!("🔑 Quantized 8-bit KV-Cache: {}", if params.quantized_kv { "ENABLED (U8 + F32 scale)" } else { "DISABLED (Standard F16/F32)" });

    let mut token_count = 0usize;
    let start_gen = Instant::now();
    let generated = {
        let mut on_token = |_piece: &str| -> bool {
            token_count += 1;
            true
        };
        text_model.generate_stream_with_params(&prompt, &params, Some(&mut on_token))?
    };
    let gen_time = start_gen.elapsed();

    println!("--------------------------------------------------------------------------------");
    println!("📝 Generated Output:");
    println!("{}", generated);
    println!("================================================================================");

    let secs = gen_time.as_secs_f64().max(0.001);
    if token_count > 1 {
        println!(
            "⏱️  Generation completed in {:.2}s ({} tokens, {:.2} tok/s)",
            secs,
            token_count,
            token_count as f64 / secs
        );
    } else {
        println!(
            "⏱️  Generation completed in {:.2}s (~up to {} tokens, {:.2} tok/s)",
            secs,
            max_tokens,
            max_tokens as f64 / secs
        );
    }
    println!("================================================================================");

    Ok(())
}
