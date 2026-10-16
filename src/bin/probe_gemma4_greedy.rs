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

    let prompt = "<|turn>user\nExplain quantum computing in one sentence.<turn|>\n<|turn>model\n<|channel>thought\n";
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
        let mut indexed: Vec<(usize, f32)> = logits_v.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!("step {step} top 5 candidates:");
        for i in 0..5 {
            let (id, logit) = indexed[i];
            let piece = tokenizer.decode(&[id as u32], false).unwrap_or_default();
            println!("   candidate {i}: token_id {id} ({piece:?}) logit {logit:.3}");
        }

        next_token = indexed[0].0 as u32;
        tokens.push(next_token);
    }

    let full_text = tokenizer.decode(&tokens, false).unwrap_or_default();
    println!("\nGenerated Text:\n{}", full_text);
    Ok(())
}
