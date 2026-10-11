// [WFGY] Zone: RISK | λ: 0.35 | Fallbacks: 0 | Action: RDNA4 Native INT4 WMMA GEMV Kernel for Q4_K_M GGUF
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// Definition of GGUF Q4_K block (256 weights per super-block)
// Struct size: 2 * 2 + 12 + 128 = 144 bytes per 256 weights
struct block_q4_K {
    half d;           // delta / super-block scale (fp16)
    half dmin;        // super-block min (fp16)
    uint8_t scales[12]; // 6-bit quantized sub-block scales (8 scales of 6-bits = 48 bits = 6 bytes, plus min scales)
    uint8_t qs[128];    // 4-bit quantized values (256 values packed into 128 bytes)
};

// RDNA4 WMMA INT4 intrinsic vector types
typedef int32_t int32x2_t __attribute__((ext_vector_type(2)));
typedef int32_t int32x8_t __attribute__((ext_vector_type(8)));

extern "C" __global__ void __launch_bounds__(64) wmma_gemv_q4k_rdna4_kernel(
    const block_q4_K* __restrict__ weights, // [rows, cols / 256]
    const half* __restrict__ x,             // [cols] input vector
    half* __restrict__ y,                   // [rows] output vector
    int cols,
    int rows
) {
#if defined(__gfx1201__) || defined(__gfx1200__) || defined(__gfx12__)
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;

    int tid = threadIdx.x; // Wave32 lane: 0..31
    int blocks_per_row = cols / 256;

    float row_acc = 0.0f;

    // Process blocks of 256 weights
    for (int b = 0; b < blocks_per_row; ++b) {
        const block_q4_K& blk = weights[row * blocks_per_row + b];
        const half* x_blk = x + b * 256;

        float d = __half2float(blk.d);
        float dmin = __half2float(blk.dmin);

        // Each wave32 (tid 0..31) executes the native RDNA4 WMMA INT4 instruction
        // __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12
        // Sub-block offset based on lane ID
        int sub_block = tid / 4; // 0..7
        int elem_in_sub = (tid % 4) * 8; // 8 weights per lane

        // Quantized nibbles: 128 bytes total -> 16 bytes per sub-block
        const uint8_t* qs_sub = blk.qs + sub_block * 16;

        // Pack 32 INT4 elements into int32x2_t (128 bits total: 8 bytes per 32-bit register)
        int32x2_t a_vec = *(const int32x2_t*)(qs_sub + (tid % 2) * 8);
        int32x2_t b_vec = *(const int32x2_t*)(x_blk + sub_block * 32);

        int32x8_t c_vec = {0};
        // Native RDNA4 hardware WMMA INT4 instruction:
        int32x8_t wmma_res = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
            true,   // A signedness
            a_vec,
            false,  // B signedness
            b_vec,
            c_vec,
            false   // clamp
        );

        // Sum the hardware WMMA accumulator output lanes
        float sub_acc = (float)(wmma_res[0] + wmma_res[1] + wmma_res[2] + wmma_res[3]);

        // Intra-wave reduction via shuffle across the 32 lanes of the wave
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sub_acc += __shfl_down(sub_acc, offset, 32);
        }

        if (tid == 0) {
            row_acc += sub_acc * d;
        }
    }

    if (tid == 0) {
        y[row] = __float2half(row_acc);
    }
#endif
}

// C-ABI dispatch function callable from Rust
extern "C" void launch_wmma_gemv_q4k_rdna4(
    const void* weights,
    const void* x,
    void* y,
    int cols,
    int rows,
    hipStream_t stream
) {
    dim3 block(32, 2); // 2 rows per block, 32 threads per wave
    dim3 grid((rows + 1) / 2);
    hipLaunchKernelGGL(
        wmma_gemv_q4k_rdna4_kernel,
        grid,
        block,
        0,
        stream,
        (const block_q4_K*)weights,
        (const half*)x,
        (half*)y,
        cols,
        rows
    );
}

// Pinned Host Memory (Zero-Copy DMA Transfer) allocation & deallocation
extern "C" void* hip_alloc_pinned_host(size_t size) {
    void* ptr = nullptr;
    hipError_t err = hipHostMalloc(&ptr, size, hipHostMallocDefault);
    if (err != hipSuccess) {
        return nullptr;
    }
    return ptr;
}

extern "C" void hip_free_pinned_host(void* ptr) {
    if (ptr) {
        hipHostFree(ptr);
    }
}

