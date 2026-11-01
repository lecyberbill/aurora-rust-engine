// Split-K flash-decoding attention for one query per head (decode).
//
// Kernel 1 computes partial (m, l, acc) per (head, KV chunk). Kernel 2 combines
// the chunks. K/V are the *unique* GQA tensors [B, Hkv, L, D] (no repeat copy);
// head h reads kv head h / (H/Hkv). D is 256 (8 values/lane) or 512 (16/lane).
//
// K/V may be F32 or F16 (`half`); the query, mask, partials and output stay F32,
// so the F16 path only halves the resident KV cache with no quality loss in the
// softmax accumulation.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <float.h>

#define WARP 32

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(0xffffffffffffffffULL, v, o);
    }
    return v;
}

__device__ __forceinline__ float kv_to_float(float x) { return x; }
__device__ __forceinline__ float kv_to_float(const __half& x) { return __half2float(x); }

template <typename KV, int DPTL>
__device__ __forceinline__ void flash_chunk_body(
    const float* __restrict__ q,   // [B, H, 1, D]
    const KV* __restrict__ k,      // [B, Hkv, L, D]
    const KV* __restrict__ v,      // [B, Hkv, L, D]
    const float* __restrict__ mask, // [L] additive bias, or null
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
    const KV* kb = k + ((long long)b * Hkv + kvh) * L * D;
    const KV* vb = v + ((long long)b * Hkv + kvh) * L * D;

    float m = -FLT_MAX;
    float l = 0.0f;
    float acc[DPTL];
    #pragma unroll
    for (int i = 0; i < DPTL; ++i) {
        acc[i] = 0.0f;
    }

    for (int j = start; j < end; ++j) {
        const KV* kj = kb + (long long)j * D;
        const KV* vj = vb + (long long)j * D;
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < DPTL; ++i) {
            int d = lane + i * WARP;
            dot += qh[d] * kv_to_float(kj[d]);
        }
        float s = warp_sum(dot) * scale;
        if (mask != nullptr) {
            s += mask[j];
        }
        float m_new = fmaxf(m, s);
        float alpha = (m == -FLT_MAX) ? 0.0f : __expf(m - m_new);
        float w = __expf(s - m_new);
        l = alpha * l + w;
        #pragma unroll
        for (int i = 0; i < DPTL; ++i) {
            int d = lane + i * WARP;
            acc[i] = acc[i] * alpha + w * kv_to_float(vj[d]);
        }
        m = m_new;
    }

    if (lane == 0) {
        pm[bh * NC + c] = m;
        pl[bh * NC + c] = l;
    }
    #pragma unroll
    for (int i = 0; i < DPTL; ++i) {
        int d = lane + i * WARP;
        pacc[(long long)(bh * NC + c) * D + d] = acc[i];
    }
}

extern "C" __global__ void __launch_bounds__(32) flash_decode_chunk_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ mask,
    float* __restrict__ pm,
    float* __restrict__ pl,
    float* __restrict__ pacc,
    int H, int Hkv, int L, int D, int NC, int CK, float scale
) {
    flash_chunk_body<float, 8>(q, k, v, mask, pm, pl, pacc, H, Hkv, L, D, NC, CK, scale);
}

extern "C" __global__ void __launch_bounds__(32) flash_decode_chunk_f16_d256(
    const float* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    const float* __restrict__ mask,
    float* __restrict__ pm,
    float* __restrict__ pl,
    float* __restrict__ pacc,
    int H, int Hkv, int L, int D, int NC, int CK, float scale
) {
    flash_chunk_body<__half, 8>(q, k, v, mask, pm, pl, pacc, H, Hkv, L, D, NC, CK, scale);
}

extern "C" __global__ void __launch_bounds__(32) flash_decode_chunk_f16_d512(
    const float* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    const float* __restrict__ mask,
    float* __restrict__ pm,
    float* __restrict__ pl,
    float* __restrict__ pacc,
    int H, int Hkv, int L, int D, int NC, int CK, float scale
) {
    flash_chunk_body<__half, 16>(q, k, v, mask, pm, pl, pacc, H, Hkv, L, D, NC, CK, scale);
}

template <int DPTL>
__device__ __forceinline__ void flash_combine_body(
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
    float acc[DPTL];
    #pragma unroll
    for (int i = 0; i < DPTL; ++i) {
        acc[i] = 0.0f;
    }
    for (int c = 0; c < NC; ++c) {
        float w = __expf(pm[bh * NC + c] - m);
        l += pl[bh * NC + c] * w;
        #pragma unroll
        for (int i = 0; i < DPTL; ++i) {
            int d = lane + i * WARP;
            acc[i] += pacc[(long long)(bh * NC + c) * D + d] * w;
        }
    }
    float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    #pragma unroll
    for (int i = 0; i < DPTL; ++i) {
        int d = lane + i * WARP;
        out[(long long)bh * D + d] = acc[i] * inv;
    }
}

extern "C" __global__ void __launch_bounds__(32) flash_decode_combine_f32(
    const float* __restrict__ pm,
    const float* __restrict__ pl,
    const float* __restrict__ pacc,
    float* __restrict__ out,
    int NC,
    int D
) {
    flash_combine_body<8>(pm, pl, pacc, out, NC, D);
}

extern "C" __global__ void __launch_bounds__(32) flash_decode_combine_d512(
    const float* __restrict__ pm,
    const float* __restrict__ pl,
    const float* __restrict__ pacc,
    float* __restrict__ out,
    int NC,
    int D
) {
    flash_combine_body<16>(pm, pl, pacc, out, NC, D);
}

extern "C" void launch_flash_decode_f32(
    const float* q,
    const float* k,
    const float* v,
    const float* mask,
    float* pm,
    float* pl,
    float* pacc,
    float* out,
    int B, int H, int Hkv, int L, int NC, int CK, float scale,
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
        q, k, v, mask, pm, pl, pacc, H, Hkv, L, D, NC, CK, scale
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

extern "C" void launch_flash_decode_f16(
    const float* q,
    const __half* k,
    const __half* v,
    const float* mask,
    float* pm,
    float* pl,
    float* pacc,
    float* out,
    int B, int H, int Hkv, int L, int D, int NC, int CK, float scale,
    hipStream_t stream
) {
    dim3 grid_chunk(NC, B * H);
    if (D == 512) {
        hipLaunchKernelGGL(
            flash_decode_chunk_f16_d512,
            grid_chunk,
            dim3(32),
            0,
            stream,
            q, k, v, mask, pm, pl, pacc, H, Hkv, L, D, NC, CK, scale
        );
        hipLaunchKernelGGL(
            flash_decode_combine_d512,
            dim3(B * H),
            dim3(32),
            0,
            stream,
            pm, pl, pacc, out, NC, D
        );
    } else {
        hipLaunchKernelGGL(
            flash_decode_chunk_f16_d256,
            grid_chunk,
            dim3(32),
            0,
            stream,
            q, k, v, mask, pm, pl, pacc, H, Hkv, L, D, NC, CK, scale
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
}
