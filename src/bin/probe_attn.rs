use std::path::PathBuf;
use candle_core::{DType, Tensor, IndexOp};
use aurora_rust_engine::models::{AutoModel, downcast_model};
use aurora_rust_engine::models::text::TextModel;
use aurora_rust_engine::models::text::causal::{KVCache, KVCacheStorage};

fn layer_k(cache: &KVCache) -> &Tensor {
    match &cache.layers[0].storage {
        KVCacheStorage::Standard { k: Some(k), .. } => k,
        _ => panic!("expected standard KV cache"),
    }
}

fn main() -> anyhow::Result<()> {
    let model_path = PathBuf::from("/models/llm/gemma-3-12b-it-q4_k_m.gguf");
    let device = aurora_rust_engine::device::auto_device()?;
    let boxed_model = AutoModel::from_local(&model_path, device.clone(), DType::F16)?;
    let mut model_lock = boxed_model.lock().unwrap();
    let text_model = downcast_model::<TextModel>(&mut *model_lock).unwrap();
    let pipe = text_model.as_causal_lm_mut().expect("expected CausalLM");
    let tok = pipe.tokenizer.clone().unwrap();

    let prompt = "<start_of_turn>user\nHello world<end_of_turn>\n<start_of_turn>model\n";
    let prompt_tokens = tok.encode(prompt, true).unwrap().get_ids().to_vec();

    // Prefill
    pipe.kv_cache.reset();
    let input0 = Tensor::new(&prompt_tokens[..], &device)?.unsqueeze(0)?;
    let _ = pipe.forward(&input0, 0)?;

    // Check layer 0 KV cache len
    let k0 = layer_k(&pipe.kv_cache);
    println!("After prefill: layer 0 K shape: {:?}", k0.shape());

    // Run step 1: tok=17531 ("Hey")
    let tok1 = 17531u32;
    let input1 = Tensor::new(&[tok1], &device)?.unsqueeze(0)?;
    let _ = pipe.forward(&input1, prompt_tokens.len())?;
    let k1 = layer_k(&pipe.kv_cache);
    println!("After step 1: layer 0 K shape: {:?}", k1.shape());

    // Run step 2: tok=993 (" there")
    let tok2 = 993u32;
    let input2 = Tensor::new(&[tok2], &device)?.unsqueeze(0)?;
    let _ = pipe.forward(&input2, prompt_tokens.len() + 1)?;
    let k2 = layer_k(&pipe.kv_cache);
    println!("After step 2: layer 0 K shape: {:?}", k2.shape());

    // Run step 3: tok=236888 ("!")
    let tok3 = 236888u32;
    let input3 = Tensor::new(&[tok3], &device)?.unsqueeze(0)?;
    let logits3 = pipe.forward(&input3, prompt_tokens.len() + 2)?;
    let l3 = logits3.i((0, 0))?;
    let l3_vec = l3.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let mut top3: Vec<(usize, f32)> = l3_vec.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    top3.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("Step 3 top 5 in autoregressive decode:");
    for (idx, score) in top3.iter().take(5) {
        let piece = tok.decode(&[*idx as u32], false).unwrap_or_default();
        println!("   - token {idx} ({:?}): {score:.4}", piece);
    }

    Ok(())
}
