// [WFGY] Zone: RISK | λ: 0.35 | Fallbacks: 0 | Action: Native HIP Fused Scaled Dot-Product Attention (Online Softmax) for AMD ROCm (gfx1201)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <float.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256

// Helper for warp-level max reduction
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down(val, offset, WARP_SIZE));
    }
    return val;
}

// Helper for warp-level sum reduction
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset, WARP_SIZE);
    }
    return val;
}

/// Native HIP Fused SDPA Kernel with Online Softmax (FlashAttention-style)
///
/// Handles [B, H, Seq_Q, D] x [B, H, Seq_K, D] -> [B, H, Seq_Q, D]
/// One thread-block computes one row (one query token for one head).
extern "C" __global__ void __launch_bounds__(BLOCK_SIZE) hip_fused_sdpa_f16_kernel(
    const half* __restrict__ q,  // [B, H, L_q, D]
    const half* __restrict__ k,  // [B, H, L_k, D]
    const half* __restrict__ v,  // [B, H, L_k, D]
    half* __restrict__ out,      // [B, H, L_q, D]
    int batch_size,
    int num_heads,
    int seq_q,
    int seq_k,
    int head_dim,
    float scale
) {
    int q_idx = blockIdx.x; // query token index in 0..(seq_q - 1)
    int bh_idx = blockIdx.y; // batch * head index in 0..(batch_size * num_heads - 1)
    
    if (q_idx >= seq_q || bh_idx >= batch_size * num_heads) return;

    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int num_warps = BLOCK_SIZE / WARP_SIZE;

    // Strides
    long long bh_stride_q = (long long)seq_q * head_dim;
    long long bh_stride_kv = (long long)seq_k * head_dim;

    const half* q_row = q + bh_idx * bh_stride_q + q_idx * head_dim;
    const half* k_base = k + bh_idx * bh_stride_kv;
    const half* v_base = v + bh_idx * bh_stride_kv;
    half* out_row = out + bh_idx * bh_stride_q + q_idx * head_dim;

    // Shared memory for per-warp reductions and Q caching
    __shared__ float s_warp_max[8];
    __shared__ float s_warp_sum[8];
    extern __shared__ half s_mem[]; // dynamically sized for Q vector: head_dim * sizeof(half)
    half* s_q = s_mem;

    // Preload Q row into shared memory
    for (int d = tid; d < head_dim; d += BLOCK_SIZE) {
        s_q[d] = q_row[d];
    }
    __syncthreads();

    // Online softmax tracking variables
    float m_prev = -FLT_MAX;
    float l_prev = 0.0f;

    // Local accumulator for output dimensions (up to D=128 supported directly in registers)
    // Most DiT head_dim is 64 or 128.
    float acc_out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int d_per_thread = (head_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Loop over Key / Value tokens in tiles of BLOCK_SIZE
    for (int k_idx = 0; k_idx < seq_k; ++k_idx) {
        // Step 1: Compute S_j = (Q . K_j) * scale
        const half* k_row = k_base + k_idx * head_dim;
        float dot = 0.0f;
        for (int d = tid; d < head_dim; d += BLOCK_SIZE) {
            dot += __half2float(s_q[d]) * __half2float(k_row[d]);
        }

        // Warp reduction for dot product
        float warp_dot = warp_reduce_sum(dot);
        if (lane == 0) s_warp_sum[warp_id] = warp_dot;
        __syncthreads();

        float score = 0.0f;
        if (tid == 0) {
            for (int w = 0; w < num_warps; ++w) score += s_warp_sum[w];
            score *= scale;
            s_warp_max[0] = score; // reuse shared slot
        }
        __syncthreads();
        score = s_warp_max[0];

        // Step 2: Online Softmax update
        // m_new = max(m_prev, score)
        // alpha = exp(m_prev - m_new)
        // weight = exp(score - m_new)
        // l_new = alpha * l_prev + weight
        float m_new = fmaxf(m_prev, score);
        float alpha = (m_prev == -FLT_MAX) ? 0.0f : __expf(m_prev - m_new);
        float weight = __expf(score - m_new);
        float l_new = alpha * l_prev + weight;

        // Step 3: Rescale accumulated output and add weight * V_j
        const half* v_row = v_base + k_idx * head_dim;
        for (int i = 0; i < d_per_thread; ++i) {
            int d = tid + i * BLOCK_SIZE;
            if (d < head_dim) {
                float v_val = __half2float(v_row[d]);
                acc_out[i] = acc_out[i] * alpha + weight * v_val;
            }
        }

        m_prev = m_new;
        l_prev = l_new;
    }

    // Final normalization by 1 / l_prev and write to global memory
    float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
    for (int i = 0; i < d_per_thread; ++i) {
        int d = tid + i * BLOCK_SIZE;
        if (d < head_dim) {
            out_row[d] = __float2half(acc_out[i] * inv_l);
        }
    }
}

// C-ABI dispatch function callable from Rust
extern "C" void launch_hip_fused_sdpa_f16(
    const void* q,
    const void* k,
    const void* v,
    void* out,
    int batch_size,
    int num_heads,
    int seq_q,
    int seq_k,
    int head_dim,
    float scale,
    hipStream_t stream
) {
    dim3 block(BLOCK_SIZE);
    dim3 grid(seq_q, batch_size * num_heads);
    size_t shared_mem_bytes = head_dim * sizeof(half);

    hipLaunchKernelGGL(
        hip_fused_sdpa_f16_kernel,
        grid,
        block,
        shared_mem_bytes,
        stream,
        (const half*)q,
        (const half*)k,
        (const half*)v,
        (half*)out,
        batch_size,
        num_heads,
        seq_q,
        seq_k,
        head_dim,
        scale
    );
}

