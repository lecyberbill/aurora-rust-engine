use std::path::PathBuf;
use candle_core::{DType, Device, Tensor, IndexOp};
use aurora_rust_engine::models::{AutoModel, GenerationModel, downcast_model};
use aurora_rust_engine::models::text::TextModel;

fn main() -> anyhow::Result<()> {
    let model_path = PathBuf::from("/models/llm/gemma-3-12b-it-q4_k_m.gguf");
    let device = aurora_rust_engine::device::auto_device()?;
    let boxed_model = AutoModel::from_local(&model_path, device.clone(), DType::F16)?;
    let mut model_lock = boxed_model.lock().unwrap();
    let text_model = downcast_model::<TextModel>(&mut *model_lock).unwrap();

    let pipe = text_model.as_causal_lm_mut().expect("expected CausalLM");

    let prompt = "<start_of_turn>user\nHello world<end_of_turn>\n<start_of_turn>model\n";
    let tok = pipe.tokenizer.as_ref().unwrap();
    let prompt_tokens = tok.encode(prompt, true).unwrap().get_ids().to_vec();
    println!("Prompt tokens: {:?}", prompt_tokens);

    pipe.kv_cache.reset();
    let input0 = Tensor::new(&prompt_tokens[..], &device)?.unsqueeze(0)?;
    let logits0 = pipe.forward(&input0, 0)?;
    let last_l0 = logits0.i((0, prompt_tokens.len() - 1))?;
    let l0_vec = last_l0.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let mut top0: Vec<(usize, f32)> = l0_vec.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    top0.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("Step 0 top 5:");
    for (idx, score) in top0.iter().take(5) {
        let piece = tok.decode(&[*idx as u32], false).unwrap_or_default();
        println!("   - token {idx} ({:?}): {score:.4}", piece);
    }

    // Step 1: feed token top0[0].0
    let tok1 = top0[0].0 as u32;
    let input1 = Tensor::new(&[tok1], &device)?.unsqueeze(0)?;
    let pos = prompt_tokens.len();
    let logits1 = pipe.forward(&input1, pos)?;
    let last_l1 = logits1.i((0, 0))?;
    let l1_vec = last_l1.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let mut top1: Vec<(usize, f32)> = l1_vec.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    top1.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("Step 1 top 5 (pos={}):", pos);
    for (idx, score) in top1.iter().take(5) {
        let piece = tok.decode(&[*idx as u32], false).unwrap_or_default();
        println!("   - token {idx} ({:?}): {score:.4}", piece);
    }

    Ok(())
}
