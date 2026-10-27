// Gated DeltaNet recurrence, one decode step, batch=1.
//
// State layout [H, Dv, Dk] with Dk contiguous (one float4 per lane). One wave
// per (head, Dv row); no LDS, S read and written once.
//
// Math (matches quantized_qwen35::run_delta_recurrence, pre-decay error):
//   kv    = S_prev[j, :] . k        (pre-decay)
//   delta = beta * (v[j] - kv)
//   S     = alpha * S_prev + delta * k
//   out[j] = S[j, :] . q            (q already scaled by 1/sqrt(Dk))
#include <hip/hip_runtime.h>
#include <float.h>

#define WARP 32

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(0xffffffffffffffffULL, v, o);
    }
    return v;
}

extern "C" __global__ void __launch_bounds__(128) gdn_step_f32(
    float* __restrict__ state,       // [H, Dv, Dk]
    const float* __restrict__ q,     // [H, Dk]
    const float* __restrict__ k,     // [H, Dk]
    const float* __restrict__ v,     // [H, Dv]
    const float* __restrict__ decay, // [H] alpha in (0,1]
    const float* __restrict__ beta,  // [H]
    float* __restrict__ out,         // [H, Dv]
    int Dv,
    int Dk
) {
    const int lane = threadIdx.x;                          // 0..31
    const int j = blockIdx.x * blockDim.y + threadIdx.y;   // Dv row
    const int h = blockIdx.y;
    const int n4 = Dk / 4;
    if (j >= Dv || lane >= n4) {
        return;
    }

    const float4* q4p = reinterpret_cast<const float4*>(q + (long long)h * Dk);
    const float4* k4p = reinterpret_cast<const float4*>(k + (long long)h * Dk);
    float4* s4p = reinterpret_cast<float4*>(
        state + ((long long)h * Dv + j) * Dk
    );

    const float4 qv = q4p[lane];
    const float4 kv = k4p[lane];
    float4 s = s4p[lane];

    // kv = S_prev . k (pre-decay)
    float ks = s.x * kv.x + s.y * kv.y + s.z * kv.z + s.w * kv.w;
    ks = warp_sum(ks);

    const float alpha = decay[h];
    const float b = beta[h];
    const float delta = b * (v[(long long)h * Dv + j] - ks);

    s.x = alpha * s.x + delta * kv.x;
    s.y = alpha * s.y + delta * kv.y;
    s.z = alpha * s.z + delta * kv.z;
    s.w = alpha * s.w + delta * kv.w;
    s4p[lane] = s;

    float o = s.x * qv.x + s.y * qv.y + s.z * qv.z + s.w * qv.w;
    o = warp_sum(o);
    if (lane == 0) {
        out[(long long)h * Dv + j] = o;
    }
}

extern "C" void launch_gdn_step_f32(
    float* state,
    const float* q,
    const float* k,
    const float* v,
    const float* decay,
    const float* beta,
    float* out,
    int H,
    int Dv,
    int Dk,
    hipStream_t stream
) {
    dim3 block(32, 4, 1);
    dim3 grid((Dv + 3) / 4, H, 1);
    hipLaunchKernelGGL(
        gdn_step_f32,
        grid,
        block,
        0,
        stream,
        state, q, k, v, decay, beta, out, Dv, Dk
    );
}
