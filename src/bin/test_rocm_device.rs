use candle_core::Device;

fn main() -> anyhow::Result<()> {
    println!("🔍 Probing ROCm / HIP compute devices on aurora-dev...");
    
    #[cfg(feature = "rocm")]
    {
        println!("Feature 'rocm' is ENABLED at compile time.");
        match Device::new_rocm(0) {
            Ok(dev) => {
                println!("  ✅ Device::new_rocm(0) initialized: {:?}", dev);
                let a = candle_core::Tensor::randn(0.0f32, 1.0f32, (1024, 1024), &dev)?.to_dtype(candle_core::DType::BF16)?;
                let b = candle_core::Tensor::randn(0.0f32, 1.0f32, (1024, 1024), &dev)?.to_dtype(candle_core::DType::BF16)?;
                let c = a.matmul(&b)?;
                println!("  🚀 GPU ROCm Matmul BF16 success! shape: {:?}", c.dims());
            }
            Err(e) => println!("  ❌ Device::new_rocm(0) failed: {:?}", e),
        }
    }

    #[cfg(not(feature = "rocm"))]
    {
        println!("⚠️ Feature 'rocm' is NOT enabled in this binary build!");
    }

    Ok(())
}
