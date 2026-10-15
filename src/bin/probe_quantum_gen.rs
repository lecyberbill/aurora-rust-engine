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

    // Prefill
    pipe.kv_cache.reset();
    let input0 = Tensor::new(&prompt_tokens[..], &device)?.unsqueeze(0)?;
    let logits0 = pipe.forward(&input0, 0)?;
    let l0 = logits0.i((0, prompt_tokens.len() - 1))?;
    let l0_vec = l0.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in l0_vec.iter().enumerate() {
        if v > best_val { best_val = v; best_idx = i; }
    }
    let mut curr_tok = best_idx as u32;
    let mut generated = vec![curr_tok];

    let mut pos = prompt_tokens.len();
    for _ in 1..40 {
        let input_t = Tensor::new(&[curr_tok], &device)?.unsqueeze(0)?;
        let step_logits = pipe.forward(&input_t, pos)?;
        let step_last = step_logits.i((0, 0))?;
        let step_vec = step_last.to_dtype(DType::F32)?.to_vec1::<f32>()?;

        // Greedy
        let mut b_idx = 0;
        let mut b_val = f32::NEG_INFINITY;
        for (i, &v) in step_vec.iter().enumerate() {
            if v > b_val { b_val = v; b_idx = i; }
        }
        curr_tok = b_idx as u32;
        generated.push(curr_tok);
        pos += 1;
        if curr_tok == 1 || curr_tok == 106 { break; }
    }

    let full_text = tok.decode(&generated, true).unwrap_or_default();
    println!("Pure Greedy Generated output:\n{}", full_text);

    Ok(())
}
