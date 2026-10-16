use candle_core::{Device, Tensor, DType, quantized::gguf_file};
use tokenizers::Tokenizer;
use std::fs::File;

fn main() -> anyhow::Result<()> {
    #[cfg(feature = "rocm")]
    let dev = Device::new_rocm(0)?;
    #[cfg(not(feature = "rocm"))]
    let dev = Device::Cpu;

    let mut file = File::open("/models/llm/gemma-4-31B-it-UD-Q4_K_XL.gguf")?;
    let content = gguf_file::Content::read(&mut file)?;
    let mut model = aurora_rust_engine::models::quantized_gemma3::ModelWeights::from_gguf(content, &mut file, &dev)?;
    let tokenizer = Tokenizer::from_file("gemma4_tokenizer.json")
        .map_err(|e| anyhow::anyhow!(e))?;

    let prompt = "<|turn>system\n<|think|>\n<turn|>\n<|turn>user\nExplain quantum computing in one sentence.<turn|>\n<|turn>model\n";
    let encoding = tokenizer.encode(prompt, true).map_err(|e| anyhow::anyhow!(e))?;
    let tokens = encoding.get_ids();
    println!("Prompt tokens ({}):", tokens.len());
    for t in tokens {
        println!("  id {t} => {:?}", tokenizer.decode(&[*t], false).unwrap_or_default());
    }

    let input_tensor = Tensor::new(tokens, &dev)?.unsqueeze(0)?;
    let logits = model.forward(&input_tensor, 0)?;
    println!("Step 0 forward logits shape: {:?}", logits.shape());

    let logits_v: Vec<f32> = logits.squeeze(0)?.to_dtype(DType::F32)?.to_vec1()?;
    let mut indexed: Vec<(usize, f32)> = logits_v.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("Top 5 next tokens from prompt:");
    for (id, logit) in indexed.iter().take(5) {
        let piece = tokenizer.decode(&[*id as u32], false).unwrap_or_default();
        println!("  id {id} ({piece:?}): logit {logit}");
    }

    Ok(())
}
