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

    // Replay steps 0..8
    let generated = [102093, 20124, 236764, 528, 886, 13315, 236764, 563, 496];
    pipe.kv_cache.reset();
    let input0 = Tensor::new(&prompt_tokens[..], &device)?.unsqueeze(0)?;
    let _ = pipe.forward(&input0, 0)?;

    let mut pos = prompt_tokens.len();
    for &t in &generated {
        let inp = Tensor::new(&[t], &device)?.unsqueeze(0)?;
        let step_logits = pipe.forward(&inp, pos)?;
        if t == 496 { // After token 496 (" a")
            let step_last = step_logits.i((0, 0))?;
            let step_vec = step_last.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            println!("Score for token 236772 ('-'): {}", step_vec[236772]);
            println!("Score for token 1722 (' type'): {}", step_vec[1722]);
            println!("Score for token 38472 (' revolutionary'): {}", step_vec[38472]);
            println!("Score for token 226156 ('revolutionary'): {}", step_vec[226156]);
            let mut top_cands: Vec<(usize, f32)> = step_vec.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            top_cands.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            println!("Top 10 candidates after '... is a':");
            for (idx, score) in top_cands.iter().take(10) {
                let piece = tok.decode(&[*idx as u32], false).unwrap_or_default();
                println!("   - token {idx} ({:?}): {score:.4}", piece);
            }
        }
        pos += 1;
    }

    Ok(())
}
