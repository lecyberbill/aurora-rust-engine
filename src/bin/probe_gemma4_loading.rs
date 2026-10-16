use candle_core::{Device, quantized::gguf_file};
use std::fs::File;

fn main() -> anyhow::Result<()> {
    println!("Connecting to ROCm device 0...");
    #[cfg(feature = "rocm")]
    let dev = Device::new_rocm(0)?;
    #[cfg(not(feature = "rocm"))]
    let dev = Device::Cpu;

    println!("Opening GGUF...");
    let mut file = File::open("/models/llm/gemma-4-31B-it-UD-Q4_K_XL.gguf")?;
    let content = gguf_file::Content::read(&mut file)?;
    println!("Tensors in content: {}", content.tensor_infos.len());

    println!("Testing token_embd.weight...");
    let t_emb = content.tensor(&mut file, "token_embd.weight", &dev)?;
    println!("token_embd loaded as QTensor ok. Dequantizing...");
    let deq = t_emb.dequantize(&dev)?;
    println!("token_embd dequantized ok! Shape: {:?}", deq.shape());

    for i in 0..60 {
        let prefix = format!("blk.{i}");
        print!("Layer {i}: ");
        let _q = content.tensor(&mut file, &format!("{prefix}.attn_q.weight"), &dev)?;
        let _k = content.tensor(&mut file, &format!("{prefix}.attn_k.weight"), &dev)?;
        let _v = content.tensor(&mut file, &format!("{prefix}.attn_v.weight"), &dev).ok();
        let _o = content.tensor(&mut file, &format!("{prefix}.attn_output.weight"), &dev)?;
        let _g = content.tensor(&mut file, &format!("{prefix}.ffn_gate.weight"), &dev)?;
        let _u = content.tensor(&mut file, &format!("{prefix}.ffn_up.weight"), &dev)?;
        let _d = content.tensor(&mut file, &format!("{prefix}.ffn_down.weight"), &dev)?;
        println!("loaded OK");
    }
    println!("ALL 60 LAYERS LOADED SUCCESSFULLY INTO ROCM VRAM!");
    Ok(())
}
