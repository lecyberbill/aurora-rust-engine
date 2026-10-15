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

    // 1. Prefill full 18 tokens: prompt + [102093, 20124]
    let mut full_tokens = prompt_tokens.clone();
    full_tokens.push(102093);
    full_tokens.push(20124);

    pipe.kv_cache.reset();
    let input_full = Tensor::new(&full_tokens[..], &device)?.unsqueeze(0)?;
    let logits_full = pipe.forward(&input_full, 0)?;
    let l_full = logits_full.i((0, full_tokens.len() - 1))?;
    let l_full_vec = l_full.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let mut top_full: Vec<(usize, f32)> = l_full_vec.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    top_full.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("FULL PREFILL top 5 after 'Quantum computing':");
    for (idx, score) in top_full.iter().take(5) {
        let piece = tok.decode(&[*idx as u32], false).unwrap_or_default();
        println!("   - token {idx} ({:?}): {score:.4}", piece);
    }

    // 2. Autoregressive decode: prefill 16 tokens, then step 1 (102093), then step 2 (20124)
    pipe.kv_cache.reset();
    let input0 = Tensor::new(&prompt_tokens[..], &device)?.unsqueeze(0)?;
    let _ = pipe.forward(&input0, 0)?;

    let inp1 = Tensor::new(&[102093u32], &device)?.unsqueeze(0)?;
    let _ = pipe.forward(&inp1, prompt_tokens.len())?;

    let inp2 = Tensor::new(&[20124u32], &device)?.unsqueeze(0)?;
    let logits_step = pipe.forward(&inp2, prompt_tokens.len() + 1)?;
    let l_step = logits_step.i((0, 0))?;
    let l_step_vec = l_step.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let mut top_step: Vec<(usize, f32)> = l_step_vec.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    top_step.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\nAUTOREGRESSIVE DECODE top 5 after 'Quantum computing':");
    for (idx, score) in top_step.iter().take(5) {
        let piece = tok.decode(&[*idx as u32], false).unwrap_or_default();
        println!("   - token {idx} ({:?}): {score:.4}", piece);
    }

    Ok(())
}
