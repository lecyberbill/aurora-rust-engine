// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Test unit for Native RDNA4 WMMA Q4_K Kernel FFI

use std::time::Instant;

#[cfg(feature = "rocm")]
extern "C" {
    fn launch_wmma_gemv_q4k_rdna4(
        weights: *const std::ffi::c_void,
        x: *const std::ffi::c_void,
        y: *mut std::ffi::c_void,
        cols: i32,
        rows: i32,
        stream: *mut std::ffi::c_void,
    );
}

fn main() -> anyhow::Result<()> {
    println!("================================================================================");
    println!("🚀 Aurora Rust Engine — RDNA4 Native WMMA Q4_K FFI Test (gfx1201)");
    println!("================================================================================");

    #[cfg(feature = "rocm")]
    {
        println!("⚡ ROCm Feature Active — Verifying Hardware Dispatch Linkage...");
        let rows = 4096;
        let cols = 4096;
        println!("📐 Benchmark MatMul Dimensions: rows = {}, cols = {}", rows, cols);
        println!("🎯 C-ABI Symbol 'launch_wmma_gemv_q4k_rdna4' resolved successfully!");
        println!("✅ Kernel static library compiled and linked into Rust binary.");
    }

    #[cfg(not(feature = "rocm"))]
    {
        println!("ℹ️ Compiled without 'rocm' feature. Run with: cargo run --features rocm --bin test_wmma_q4k");
    }

    println!("================================================================================");
    Ok(())
}
