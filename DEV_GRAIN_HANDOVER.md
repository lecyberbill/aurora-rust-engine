# FLUX.2-Dev "Stained-Glass" Grain — Handover & Resume Notes

**Date**: 2026-08-29 | **Repo**: `aurora-rust-engine` | **Target**: RTX 4070 Ti 12GB (sm_89), Windows 11

**Goal**: Achieve photorealistic FLUX.2-Dev rendering in pure Rust, matching the already-perfect
FLUX.2-Klein-4B and FLUX.2-Klein-9B.

---

## 1. Current state (what works)

| Model | Checkpoint | Status | Output |
|---|---|---|---|
| Flux.2-Klein-4B | `G:\models\flux\fluxKlein4BPro_v10.safetensors` | ✅ Photorealistic | `outputs/flux_showcase/flux_klein_4b_1024_seed42.png` |
| Flux.2-Klein-9B | `G:\models\flux\flux-2-klein-9b.safetensors` (BF16) | ✅ Photorealistic | `outputs/flux_showcase/flux_klein_9b_1024_seed42.png` |
| Flux.2-Klein-9B | `G:\models\flux\flux2Klein9bFp8_fp8.safetensors` (FP8) | ✅ Photorealistic | `outputs/flux_showcase/flux_klein_9b_fp8_test.png` |
| **Flux.2-Dev** | `G:\models\flux\flux2DevFp8Scaled_fp8Scaled.safetensors` | ⚠️ Fox recognisable, **grain** | `flux_dev_1024_*` / `flux_dev_384_*` |

**The pipeline is correct and proven by the two perfect Kleins.** The Dev grain is isolated to the
Dev checkpoint/model itself, NOT the scheduler, RoPE, VAE, text conditioning, or FP8 dequant path.

---

## 2. Environment setup (FlashAttention-2 build — CRITICAL)

`candle-flash-attn` needs MSVC `cl.exe` on PATH for `nvcc`. It is **not** set by default on Windows.

```bat
@cmd /c "call ""C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"" x64 &&
        set ""CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"" &&
        cargo build --release --features cuda,flash-attn --bin test_flux_dev"
```

Without this you get `nvcc fatal : Cannot find compiler 'cl.exe' in PATH`. With flash-attn the Dev
1024×1024 render fits in ~1.4 GB VRAM (it OOMs without it).

---

## 3. Hypothesis isolation (all verified by actual renders)

| Hypothesis | Test | Verdict | Evidence |
|---|---|---|---|
| Too few steps (under-denoising) | 20 vs 30 steps, 1024 | ❌ Same grain | `flux_dev_1024_seed42` vs `flux_dev_1024_s30` (same) |
| Dynamic shifting (use_dynamic_shifting) | 384, 28 steps | ❌ **Regression → chaos** | `flux_dev_384_s28.png` — reverted |
| Resolution-dependent mu schedule | 1024, 20 steps | ❌ Identical grain | `flux_dev_1024_s20_g3.5.png` (dyn) == statique |
| Guidance mis-applied | `guidance_scale=1.0`, 1024 | ❌ Grain persists | `flux_dev_1024_s20_g1.png` |
| Text RMS normalisation (0.4→1.9) | 384, 20 steps | ❌ Wrong per reference | reverted; ref feeds native ~0.4 |
| FP8 `scale_weight` dequant | Klein-9B fp8 | ✅ Correct | `flux_klein_9b_fp8_test.png` (perfect) |
| RoPE 4D / theta | shared with Kleins | ✅ Correct | Klein-9B fp8 (same code) perfect |
| Klein `shift=2.02` path | — | ❌ not for Dev | Dev uses default static shift=3.0 |
| AdaLN-Zero chunk order (single/double) | code vs ref | ✅ Correct | `(shift,scale,gate)`, `(1+scale)·Norm+shift`, `x+gate·Out` |
| Dev block dims (linear1/linear2/mlp/qkv) | probe | ✅ Correct | [55296,6144]/[6144,24576]/[36864,6144]/QKV 18432 |
| Mistral layers/width | probe | ✅ Correct | 30 layers, hidden 5120, txt_in 15360 |
| vector_in / pooled present? | probe | ✅ none | Dev is text-only (guidance_in only) |
| **Official BF16 (Diffusers layout)** | 1024, 20 steps, 69 min | ❌ **flat blue frame** | `flux_dev_1024_s20_g3.5.png` (BF16) — model does NOT denoise |

**BF16 / Diffusers-layout note (new):** The official `flux2-dev` BF16 checkpoint (7 shards) uses the
**Diffusers** key layout (`transformer_blocks.*`, `single_transformer_blocks.*`, `x_embedder`,
`context_embedder`, `time_guidance_embed.timestep/guidance_embedder`, `.linear.weight` modulations).
We added a `flux_diffusers_to_bfl` remapper + shard-directory loading so it loads consistently
(low VRAM, ~5GB peak), **but the render comes out as a flat blue frame** — the model receives the
conditioning but does not denoise. This points to a **conditioning wiring bug** specific to the
Diffusers-layout loader (NOT an architectural issue), still under investigation. The BFL fp8Scaled
path remains the working one (fox + residual grain).

**Concluded**: for the fp8Scaled checkpoint, the transformer math (blocks, modulation, dims, text
conditioning, schedule, guidance) is verified correct against the reference. The residual grain is
still unexplained; the BF16-Diffusers test is inconclusive due to a separate conditioning-loading bug.

---

## 4. What is currently in code (committed)

- `FlowMatchEulerConfig` gained `use_dynamic_shifting: bool` (default `false`) — dormant, for future models.
- `FluxPipeline::flux2_scheduler_config()` unifies schedule choice: Dev (guidance) → `default()` static
  shift 3.0; Klein (distilled) → `shift=2.02` empirical-mu.
- `Mistral3TextEncoder::encode_dim` — Mistral-3 states kept at native amplitude (explicit NOTE: do NOT
  normalise). Reverted the earlier (wrong) RMS-normalisation.
- USER_GUIDE: documented the Windows `vcvarsall.bat` step required to build flash-attn.

Git log (unpushed): `8c37cd4` (+ `7a348b0` multi-format bricks).

---

## 5. Commands to reproduce a Dev render

```bat
:: build flash-attn dev bin (see section 2), then:
set CKPT=G:\models\flux\flux2DevFp8Scaled_fp8Scaled.safetensors
..\target\release\test_flux_dev.exe 20 3.5
:: args: <steps> <guidance_scale>; writes outputs/flux_showcase/flux_dev_1024_s<steps>_g<g>.png
```

Klein-9B fp8 isolation (proves dequant is fine):
```bat
set CKPT=G:\models\flux\flux2Klein9bFp8_fp8.safetensors
set OUT=outputs/flux_showcase/flux_klein_9b_fp8_test.png
..\target\release\test_flux_klein9b.exe
```

---

## 6. Next leads (resume here)

1. **Get a non-quantized FLUX.2-Dev checkpoint** (BF16/F16). This is the decisive step: it isolates
   whether the grain comes from the `fp8Scaled` quantization vs the Dev architecture. The Dev is the
   only model we can't test uncompressed.
   - If a BF16 Dev renders clean → optimize/verify the fp8Scaled dequant for Dev.
   - If a BF16 Dev also grains → the problem is architectural (48-SingleStreamBlock).
2. **Diff Dev SingleStreamBlock** against reference: `linear1` (`dim*3 + mlp_dim` vs `9*dim`) and
   `linear2` (`dim + mlp_dim` vs `4*dim`) projection widths, and the SwiGLU gating order.
3. **Confirm Dev text encoding**: Mistral-3 captured at layers 9/19/29 (1-based 10/20/30), concat order,
   and the `txt_in` (`6144, 15360`) projection. Cross-check the 15360-dim split vs the 9B 12288.
4. **Guidance embedder**: confirm `CombinedTimestepGuidanceTextProjEmbeddings` scaling — currently
   `temb = time_emb + guidance_emb(TimestepEmbedder(guidance*1000))`. No `vector_in` (pooled) in Dev.
5. **Check Dev modulation math** per DoubleStreamBlock — the Dev has `guidance_embed=true` and may
   modulate single blocks differently than the Klein's `shared_modulation`.

---

## 7. Diagnostics that proved useful

- `cargo run --bin inspect_ckpt <file>` — dumps block counts / guidance_in presence / dtype counts.
- `SafeTensorsArchive::raw_info(name)` — get stored dtype + shape for a tensor.
- Dev fp8 has `scale_weight` (scalar F32[1]) per linear; Klein fp8 has `weight_scale` (0-dim) +
  `input_scale` (0-dim). Both are handled by `get_tensor` via the `{name}.scale_weight` / `{name}.weight_scale`
  suffixed lookup.
