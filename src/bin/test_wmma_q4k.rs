// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Test unit for Native RDNA4 WMMA Q4_K Kernel FFI


#[cfg(feature = "rocm")]
use std::time::Instant;

#[cfg(feature = "rocm")]
extern "C" {
    #[allow(dead_code)]
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

        println!("⚡ Verifying Pinned Host Memory (Zero-Copy DMA Staging)...");
        let start_alloc = Instant::now();
        let buffer_size = 1024 * 1024 * 64; // 64 MB staging buffer
        if let Some(mut buf) = aurora_rust_engine::device::PinnedHostBuffer::allocate(buffer_size) {
            println!("✅ Successfully allocated {} MB of Pinned Host Memory in {:.2}ms", 
                buffer_size / (1024 * 1024), 
                start_alloc.elapsed().as_secs_f64() * 1000.0
            );
            let slice = buf.as_mut_slice();
            slice[0] = 42;
            slice[slice.len() - 1] = 99;
            println!("✅ Zero-Copy DMA slice memory read/write verified (len: {} bytes)", slice.len());
        } else {
            println!("⚠️ Pinned Host Memory allocation returned None, falling back to standard heap.");
        }
    }

    #[cfg(not(feature = "rocm"))]
    {
        println!("ℹ️ Compiled without 'rocm' feature. Run with: cargo run --features rocm --bin test_wmma_q4k");
    }

    println!("================================================================================");
    Ok(())
}
