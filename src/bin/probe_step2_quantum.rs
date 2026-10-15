use std::path::PathBuf;
use candle_core::{DType, Tensor, IndexOp};
use aurora_rust_engine::models::{AutoModel, downcast_model};
use aurora_rust_engine::models::text::TextModel;

fn main() -> anyhow::Result<()> {
    let model_path = PathBuf::from("/models/llm/gemma-3-12b-it-q4_k_m.gguf");
    let device = aurora_rust_engine::device::auto_device()?;
    let boxed_model = AutoModel::from_local(&model_path, device.clone(), DType::F16)?;
    let mut model_lock = boxed_model.lock().unwrap();
    let text_model = downcast_model::<TextModel>(&mut *model_lock).unwrap();
    let pipe = text_model.as_causal_lm_mut().expect("expected CausalLM");
    let tok = pipe.tokenizer.clone().unwrap();

    let prompt = "<start_of_turn>user\nExplain quantum computing in one sentence.<end_of_turn>\n<start_of_turn>model\n";
    let prompt_tokens = tok.encode(prompt, true).unwrap().get_ids().to_vec();

    // Prefill prompt
    pipe.kv_cache.reset();
    let input0 = Tensor::new(&prompt_tokens[..], &device)?.unsqueeze(0)?;
    let _ = pipe.forward(&input0, 0)?;

    // Step 1: feed Quantum (102093)
    let inp1 = Tensor::new(&[102093u32], &device)?.unsqueeze(0)?;
    let _ = pipe.forward(&inp1, prompt_tokens.len())?;

    // Step 2: feed computing (20124)
    let inp2 = Tensor::new(&[20124u32], &device)?.unsqueeze(0)?;
    let logits2 = pipe.forward(&inp2, prompt_tokens.len() + 1)?;
    let l2 = logits2.i((0, 0))?;
    let l2_vec = l2.to_dtype(DType::F32)?.to_vec1::<f32>()?;

    let mut top_cands: Vec<(usize, f32)> = l2_vec.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    top_cands.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("Step 2 top 10 after 'Quantum computing':");
    for (idx, score) in top_cands.iter().take(10) {
        let piece = tok.decode(&[*idx as u32], false).unwrap_or_default();
        println!("   - token {idx} ({:?}): {score:.4}", piece);
    }

    Ok(())
}
