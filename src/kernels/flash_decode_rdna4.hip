// Split-K flash-decoding attention for one query per head (decode).
//
// Kernel 1 computes partial (m, l, acc) per (head, KV chunk). Kernel 2 combines
// the chunks. K/V are the *unique* GQA tensors [B, Hkv, L, D] (no repeat copy);
// head h reads kv head h / (H/Hkv). Requires D == 256 (8 values per lane).
#include <hip/hip_runtime.h>
#include <float.h>

#define WARP 32
#define DIMS_PER_LANE 8

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(0xffffffffffffffffULL, v, o);
    }
    return v;
}

extern "C" __global__ void __launch_bounds__(32) flash_decode_chunk_f32(
    const float* __restrict__ q,   // [B, H, 1, D]
    const float* __restrict__ k,   // [B, Hkv, L, D]
    const float* __restrict__ v,   // [B, Hkv, L, D]
    float* __restrict__ pm,        // [BH, NC]
    float* __restrict__ pl,        // [BH, NC]
    float* __restrict__ pacc,      // [BH, NC, D]
    int H,
    int Hkv,
    int L,
    int D,
    int NC,
    int CK,
    float scale
) {
    const int c = blockIdx.x;
    const int bh = blockIdx.y;
    const int lane = threadIdx.x;
    const int b = bh / H;
    const int h = bh % H;
    const int kvh = h / (H / Hkv);

    const int start = c * CK;
    const int end = min(start + CK, L);

    const float* qh = q + (long long)bh * D;
    const float* kb = k + ((long long)b * Hkv + kvh) * L * D;
    const float* vb = v + ((long long)b * Hkv + kvh) * L * D;

    float m = -FLT_MAX;
    float l = 0.0f;
    float acc[DIMS_PER_LANE];
    #pragma unroll
    for (int i = 0; i < DIMS_PER_LANE; ++i) {
        acc[i] = 0.0f;
    }

    for (int j = start; j < end; ++j) {
        const float* kj = kb + (long long)j * D;
        const float* vj = vb + (long long)j * D;
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < DIMS_PER_LANE; ++i) {
            int d = lane + i * WARP;
            dot += qh[d] * kj[d];
        }
        float s = warp_sum(dot) * scale;
        float m_new = fmaxf(m, s);
        float alpha = (m == -FLT_MAX) ? 0.0f : __expf(m - m_new);
        float w = __expf(s - m_new);
        l = alpha * l + w;
        #pragma unroll
        for (int i = 0; i < DIMS_PER_LANE; ++i) {
            int d = lane + i * WARP;
            acc[i] = acc[i] * alpha + w * vj[d];
        }
        m = m_new;
    }

    if (lane == 0) {
        pm[bh * NC + c] = m;
        pl[bh * NC + c] = l;
    }
    #pragma unroll
    for (int i = 0; i < DIMS_PER_LANE; ++i) {
        int d = lane + i * WARP;
        pacc[(long long)(bh * NC + c) * D + d] = acc[i];
    }
}

extern "C" __global__ void __launch_bounds__(32) flash_decode_combine_f32(
    const float* __restrict__ pm,
    const float* __restrict__ pl,
    const float* __restrict__ pacc,
    float* __restrict__ out, // [BH, D]
    int NC,
    int D
) {
    const int bh = blockIdx.x;
    const int lane = threadIdx.x;

    float m = -FLT_MAX;
    for (int c = 0; c < NC; ++c) {
        m = fmaxf(m, pm[bh * NC + c]);
    }
    float l = 0.0f;
    float acc[DIMS_PER_LANE];
    #pragma unroll
    for (int i = 0; i < DIMS_PER_LANE; ++i) {
        acc[i] = 0.0f;
    }
    for (int c = 0; c < NC; ++c) {
        float w = __expf(pm[bh * NC + c] - m);
        l += pl[bh * NC + c] * w;
        #pragma unroll
        for (int i = 0; i < DIMS_PER_LANE; ++i) {
            int d = lane + i * WARP;
            acc[i] += pacc[(long long)(bh * NC + c) * D + d] * w;
        }
    }
    float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    #pragma unroll
    for (int i = 0; i < DIMS_PER_LANE; ++i) {
        int d = lane + i * WARP;
        out[(long long)bh * D + d] = acc[i] * inv;
    }
}

extern "C" void launch_flash_decode_f32(
    const float* q,
    const float* k,
    const float* v,
    float* pm,
    float* pl,
    float* pacc,
    float* out,
    int B,
    int H,
    int Hkv,
    int L,
    int NC,
    int CK,
    float scale,
    hipStream_t stream
) {
    const int D = 256;
    dim3 grid_chunk(NC, B * H);
    hipLaunchKernelGGL(
        flash_decode_chunk_f32,
        grid_chunk,
        dim3(32),
        0,
        stream,
        q, k, v, pm, pl, pacc, H, Hkv, L, D, NC, CK, scale
    );
    hipLaunchKernelGGL(
        flash_decode_combine_f32,
        dim3(B * H),
        dim3(32),
        0,
        stream,
        pm, pl, pacc, out, NC, D
    );
}
