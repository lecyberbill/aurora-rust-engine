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

    let p = "<start_of_turn>user\nExplain quantum computing in one sentence.<end_of_turn>\n<start_of_turn>model\nQuantum computing";
    let ids = tok.encode(p, true).unwrap().get_ids().to_vec();
    println!("Encoded tokens count: {}", ids.len());
    for t in &ids {
        println!("  - token {t}: {:?}", tok.decode(&[*t], false).unwrap_or_default());
    }

    Ok(())
}
