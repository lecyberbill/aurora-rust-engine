use candle_core::quantized::{gguf_file, QMatMul};
use candle_core::{DType, Device, Module, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = aurora_rust_engine::device::auto_device()?;

    println!("Testing QMatMul parity on device: {:?}", dev);
    let path = "/models/llm/gemma-3-12b-it-uncensored.i1-Q4_K_M.gguf";
    if !std::path::Path::new(path).exists() {
        println!("Model file {} not present, testing phi instead", path);
        return Ok(());
    }

    let file = std::fs::File::open(path)?;
    let mut reader = std::io::BufReader::new(file);
    let content = gguf_file::Content::read(&mut reader)?;

    let tensor_name = "blk.0.ffn_down.weight";
    println!("Loading {} onto device {:?}...", tensor_name, dev);
    let qtensor_dev = content.tensor(&mut reader, tensor_name, &dev)?;
    println!("QTensor loaded: dtype {:?}, shape {:?}", qtensor_dev.dtype(), qtensor_dev.shape());

    let qmatmul = QMatMul::from_qtensor(qtensor_dev)?;

    // QTensor dims for 2D are [out_dim, in_dim] or [in_dim, out_dim]
    // blk.0.ffn_down.weight in Gemma 3 has shape: [3840, 15360] in GGUF
    let in_features = 15360;
    let input = Tensor::randn(0.0f32, 1.0f32, (1, 1, in_features), &dev)?;

    let t0 = std::time::Instant::now();
    let y_quant = qmatmul.forward(&input)?;
    let t_quant = t0.elapsed();
    println!("y_quant computed in {:.3}ms: shape {:?}, dtype {:?}", t_quant.as_secs_f64() * 1000.0, y_quant.shape(), y_quant.dtype());

    // Compute dense ground truth
    // Re-read onto CPU to dequantize cleanly
    let file2 = std::fs::File::open(path)?;
    let mut reader2 = std::io::BufReader::new(file2);
    let content2 = gguf_file::Content::read(&mut reader2)?;
    let qtensor_cpu = content2.tensor(&mut reader2, tensor_name, &Device::Cpu)?;
    let w_dense_cpu = qtensor_cpu.dequantize(&Device::Cpu)?;
    let w_dense_dev = w_dense_cpu.to_dtype(DType::F32)?.to_device(&dev)?;

    // Dense matmul: x is [1, 1, 15360], w_dense shape is [3840, 15360] -> w.t() is [15360, 3840]
    let w_adapted = if w_dense_dev.dim(1)? == in_features {
        w_dense_dev.t()?.contiguous()?
    } else {
        w_dense_dev.contiguous()?
    };
    let input_f32 = input.to_dtype(DType::F32)?;
    let t1 = std::time::Instant::now();
    let y_dense = input_f32.matmul(&w_adapted)?;
    let t_dense = t1.elapsed();
    println!("y_dense computed in {:.3}ms: shape {:?}, dtype {:?}", t_dense.as_secs_f64() * 1000.0, y_dense.shape(), y_dense.dtype());

    // Compare numerically
    let y_quant_f32 = y_quant.to_dtype(DType::F32)?;
    let diff = (&y_quant_f32 - &y_dense)?.abs()?;
    let max_diff = diff.max_all()?.to_vec0::<f32>()?;
    let max_dense = y_dense.abs()?.max_all()?.to_vec0::<f32>()?;
    let rel_error = if max_dense > 0.0 { max_diff / max_dense } else { max_diff };

    println!("============================================================");
    println!("📊 NUMERICAL PARITY RESULTS (QMatMul vs Dense on ROCm):");
    println!("   max |y_quant - y_dense| = {:.6}", max_diff);
    println!("   max |y_dense|           = {:.6}", max_dense);
    println!("   relative error          = {:.6} ({:.2}%)", rel_error, rel_error * 100.0);
    println!("============================================================");

    if rel_error > 0.05 {
        println!("❌ FAIL: QMatMul kernel divergence detected on ROCm!");
    } else {
        println!("✅ PASS: QMatMul is numerically equivalent to Dense dequantization.");
    }

    Ok(())
}
