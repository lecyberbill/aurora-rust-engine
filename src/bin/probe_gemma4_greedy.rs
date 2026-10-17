use candle_core::{Device, Tensor, DType, quantized::gguf_file};
use tokenizers::Tokenizer;
use std::fs::File;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    #[cfg(feature = "rocm")]
    let dev = Device::new_rocm(0)?;
    #[cfg(not(feature = "rocm"))]
    let dev = Device::Cpu;

    println!("Loading Gemma 4 (31B) GGUF on ROCm device: {:?}", dev);
    let start_load = Instant::now();
    let mut file = File::open("/models/llm/gemma-4-31B-it-UD-Q4_K_XL.gguf")?;
    let content = gguf_file::Content::read(&mut file)?;
    let mut model = aurora_rust_engine::models::quantized_gemma3::ModelWeights::from_gguf(content, &mut file, &dev)?;
    println!("Model loaded in {:.2?}", start_load.elapsed());

    let tokenizer = Tokenizer::from_file("gemma4_tokenizer.json").map_err(|e| anyhow::anyhow!(e))?;

    let prompt = "The capital of France is";
    let mut tokens = tokenizer.encode(prompt, true).map_err(|e| anyhow::anyhow!(e))?.get_ids().to_vec();
    print!("Prompt: {:?} -> Generation: {}", prompt, prompt);
    use std::io::Write;
    std::io::stdout().flush()?;

    let max_gen_tokens = 30;
    let start_gen = Instant::now();
    let mut gen_count = 0;

    for _ in 0..max_gen_tokens {
        let inp = Tensor::new(tokens.as_slice(), &dev)?.unsqueeze(0)?;
        let logits = model.forward(&inp, 0)?;
        let last_logits: Vec<f32> = logits.to_dtype(DType::F32)?.squeeze(0)?.to_vec1()?;
        
        let mut max_id = 0;
        let mut max_logit = f32::NEG_INFINITY;
        for (id, val) in last_logits.into_iter().enumerate() {
            if val > max_logit {
                max_logit = val;
                max_id = id;
            }
        }

        tokens.push(max_id as u32);
        gen_count += 1;

        let piece = tokenizer.decode(&[max_id as u32], false).unwrap_or_default();
        print!("{}", piece);
        std::io::stdout().flush()?;

        // Stop on EOS or end of turn
        if max_id == 1 || max_id == 107 || piece.contains("<end_of_turn>") {
            break;
        }
    }

    let elapsed = start_gen.elapsed();
    println!("\n\n--- Finished: generated {} tokens in {:.2?} ({:.2} tok/s) ---", gen_count, elapsed, gen_count as f64 / elapsed.as_secs_f64());

    Ok(())
}
