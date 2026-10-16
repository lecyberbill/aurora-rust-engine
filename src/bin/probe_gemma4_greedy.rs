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
    let tokenizer = Tokenizer::from_file("gemma_tokenizer.json")
        .map_err(|e| anyhow::anyhow!(e))?;

    let prompt = "Explain quantum computing in one sentence.";
    let encoding = tokenizer.encode(prompt, true).map_err(|e| anyhow::anyhow!(e))?;
    let mut tokens = encoding.get_ids().to_vec();

    let mut next_token = 0u32;
    for step in 0..15 {
        let (input_tensor, index_pos) = if step == 0 {
            (Tensor::new(tokens.as_slice(), &dev)?.unsqueeze(0)?, 0)
        } else {
            (Tensor::new(&[next_token], &dev)?.unsqueeze(0)?, tokens.len() - 1)
        };

        let logits = model.forward(&input_tensor, index_pos)?;
        let logits_v: Vec<f32> = logits.squeeze(0)?.to_dtype(DType::F32)?.to_vec1()?;
        let (max_id, max_logit) = logits_v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
        next_token = max_id as u32;
        tokens.push(next_token);

        let piece = tokenizer.decode(&[next_token], false).unwrap_or_default();
        println!("step {step}: token_id {next_token} ({piece:?}) logit {max_logit}");
    }

    let full_text = tokenizer.decode(&tokens, false).unwrap_or_default();
    println!("\nGenerated Text:\n{}", full_text);
    Ok(())
}
