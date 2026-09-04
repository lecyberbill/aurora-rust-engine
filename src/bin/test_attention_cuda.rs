// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Test FlashAttention-2 with cross-attention (different sequence lengths)
// Requires the `flash-attn` cargo feature: `cargo build --release --features flash-attn --bin test_attention_cuda`.

#[cfg(feature = "flash-attn")]
fn main() -> anyhow::Result<()> {
    use candle_core::{DType, Device, Tensor};

    let device = Device::new_cuda(0)?;

    println!("🧪 Testing Self-Attention (4096 x 4096)...");
    let b_size = 2;
    let heads = 10;
    let head_dim = 64;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q_self = Tensor::randn(0f32, 1f32, (b_size, 4096, heads, head_dim), &device)?.to_dtype(DType::F16)?;
    let k_self = Tensor::randn(0f32, 1f32, (b_size, 4096, heads, head_dim), &device)?.to_dtype(DType::F16)?;
    let v_self = Tensor::randn(0f32, 1f32, (b_size, 4096, heads, head_dim), &device)?.to_dtype(DType::F16)?;

    let out_self = candle_flash_attn::flash_attn(&q_self, &k_self, &v_self, scale, false)?;
    println!("  ✅ Self-Attention output shape: {:?}", out_self.shape());

    println!("🧪 Testing Cross-Attention (4096 x 77)...");
    let q_cross = Tensor::randn(0f32, 1f32, (b_size, 4096, heads, head_dim), &device)?.to_dtype(DType::F16)?;
    let k_cross = Tensor::randn(0f32, 1f32, (b_size, 77, heads, head_dim), &device)?.to_dtype(DType::F16)?;
    let v_cross = Tensor::randn(0f32, 1f32, (b_size, 77, heads, head_dim), &device)?.to_dtype(DType::F16)?;

    match candle_flash_attn::flash_attn(&q_cross, &k_cross, &v_cross, scale, false) {
        Ok(out_cross) => println!("  ✅ Cross-Attention output shape: {:?}", out_cross.shape()),
        Err(e) => println!("  ⚠️ Cross-Attention fallback needed: {:?}", e),
    }

    Ok(())
}

#[cfg(not(feature = "flash-attn"))]
fn main() -> anyhow::Result<()> {
    println!("⚠️  This binary requires the `flash-attn` feature. Build with `--features flash-attn`.");
    Ok(())
}
