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

    println!("Testing QMatMul creation layer by layer...");
    let tok_embeddings = content.tensor(&mut file, "token_embd.weight", &dev)?;
    println!("token_embd loaded as QTensor ok");
    let tok_embeddings = tok_embeddings.dequantize(&dev)?;
    println!("token_embd dequantized ok");

    for i in 0..60 {
        let prefix = format!("blk.{i}");
        print!("Layer {i}: ");
        let q = content.tensor(&mut file, &format!("{prefix}.attn_q.weight"), &dev)?;
        let _q_mm = candle_core::quantized::QMatMul::from_qtensor(q)?;
        let k = content.tensor(&mut file, &format!("{prefix}.attn_k.weight"), &dev)?;
        let _k_mm = candle_core::quantized::QMatMul::from_qtensor(k)?;
        if let Ok(v) = content.tensor(&mut file, &format!("{prefix}.attn_v.weight"), &dev) {
            let _v_mm = candle_core::quantized::QMatMul::from_qtensor(v)?;
        }
        let o = content.tensor(&mut file, &format!("{prefix}.attn_output.weight"), &dev)?;
        let _o_mm = candle_core::quantized::QMatMul::from_qtensor(o)?;
        let g = content.tensor(&mut file, &format!("{prefix}.ffn_gate.weight"), &dev)?;
        let _g_mm = candle_core::quantized::QMatMul::from_qtensor(g)?;
        let u = content.tensor(&mut file, &format!("{prefix}.ffn_up.weight"), &dev)?;
        let _u_mm = candle_core::quantized::QMatMul::from_qtensor(u)?;
        let d = content.tensor(&mut file, &format!("{prefix}.ffn_down.weight"), &dev)?;
        let _d_mm = candle_core::quantized::QMatMul::from_qtensor(d)?;
        println!("all 7 QMatMul ok");
    }

    println!("Testing output QMatMul...");
    let output = content.tensor(&mut file, "token_embd.weight", &dev)?;
    let _output_mm = candle_core::quantized::QMatMul::from_qtensor(output)?;
    println!("Output QMatMul created ok!");
    Ok(())
}
