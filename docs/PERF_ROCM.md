# ROCm performance notes (2026-10-26)

A record of what was measured on the real target, what helped, what was rejected,
and where the remaining headroom is. Scope: **pure Rust + Candle**; llama.cpp is
used only as an external measuring stick, never called by the engine.

## Testbed

- GPU: AMD Radeon AI PRO R9700, `gfx1201` (RDNA 4), ~34 GB VRAM, ~640 GB/s HBM.
- OS/driver: Linux, ROCm 10.0.
- Engine: this repo. Candle is the `xmiksay/candle` fork (`feat/rocm-backend`,
  pinned at `fca7197d`; the server build uses the local fork with the two patches
  in [`docs/patches/`](patches/)).
- Profiling: `rocprofv3`, `rocm-smi`.

## Method

- Batch 1, single stream. The binary's decode timer (load excluded).
- Fixed token budget: `AURORA_IGNORE_EOS=1`, 256 requested (257 generated) for
  paired A/B runs.
- VRAM must be free before a benchmark (unload ollama: `ollama stop <model>`);
  otherwise a resident 17 GB model starves the measured one.
- Ranges elsewhere are run-to-run variance (EOS length, clocks). Only **paired**
  numbers in the same session are used for decisions.

## Summary

| Change | Qwen 3.8 27B | Gemma 3 12B | Status |
|---|---:|---:|---|
| Fused Candle ROCm ops (`rms_norm`/`softmax`/`rope`/`sigmoid`/`silu`) | ~12.6 tok/s | ~35.7 tok/s | kept (default) |
| Per-shape mmvq/dmmv autotune (fork) | **16.4 → 18.1** (+10%) | **36.2 → 38.0** (+5%) | kept (fork, default) |
| HIP-graph decode (`hip-graph` feature) | 12.6 → 12.7 (neutral) | **35.7 → 38.5** (+8%) | opt-in |
| F16 activations | — | 38.4 → 33.0 (**−14%**) | rejected |
| GPU sampling via `sort_last_dim` | — | 35.3 → 32.9 (**−7%**) | rejected |

Reference (same GGUF, same GPU, 256 tok, via llama.cpp/ollama): **Qwen 29.4**,
**Gemma 62.7** tok/s. The remaining gap is ~1.6×.

## What helped

1. **Enabling the fused ROCm ops.** `Cargo.toml` must enable
   `candle-nn/rocm` and `candle-transformers/rocm`; without them the `rocm_fwd`
   implementations of `rms_norm`, `layer_norm`, `softmax`, `sigmoid` and `rope`
   are not compiled and every call falls back to a 5–10 kernel composition.
   This was the single largest win (Phi-4 ~13 → ~23 tok/s; Gemma ~2×).
2. **Per-shape mmvq/dmmv autotune** (fork). The fork preferred `dmmv` for *all*
   K-quants, which is wrong on `gfx1201`: e.g. Qwen `Q4K 5120×17408` and most
   `Q8_0` projections are faster with `mmvq` (q8_1 integer dot), while the
   `lm_head` (`Q6K 248320×5120`) and most Gemma shapes prefer `dmmv`. The
   autotune times both once per `(dtype, n, k)` at first decode and caches the
   winner. `CANDLE_ROCM_AUTOTUNE=0` disables it.
3. **HIP-graph decode** for attention-only models: +8% on Gemma (257 tok).
   Neutral on Qwen 3.5/3.8 because the graph attends over the full `max_len`,
   and that extra attention offsets the launch-gap recovery.

## What was rejected (and why)

- **F16 activations**: −14% on Gemma. The quantized matmul still produces F32;
  casting back to F16 after every matmul plus F16 matmul overhead costs more
  than the bandwidth saved. The GEMV already runs near HBM peak (below).
- **GPU sampling with `sort_last_dim`**: −7% on Gemma. Candle only exposes a
  full sort (O(V log V)); the CPU path is an O(V) `select_nth` plus a 1 MB D2H.
  A true O(V) top-k GPU kernel would be needed (kernel work).
- **Fused HIP SDPA for decode**: 4–8× *slower* than matmul+softmax
  (0.15–0.59 ms vs 0.04–0.08 ms per layer at `Lk=256/1024`, `d=128/256`).
  `src/kernels/fused_sdpa_rdna4.hip` runs one block per (query, head) and
  loops serially over the KV; with one query at decode, occupancy is far too
  low. Rejected for decode. (The launch was also on the null stream; it now
  uses the device stream. The kernel remains useful where it avoids
  materialising a large `[Lq, Lk]` matrix, i.e. long-context prefill.)
- **Bucketed HIP graphs**: rejected. Candle's allocator is shared, so a second
  capture reuses the first graph's intermediate buffers (which that graph still
  writes to on replay) → corrupted output / HIP errors with several buckets.
  Recapturing one bucket at a time requires saving/restoring the 48 GDN states
  (~144 MB) plus three warm-up forwards per boundary; at 256-token runs that
  overhead offsets the launch-gap gain.

## Roofline / where the time goes

- Decode GEMV bandwidth (from the autotune timings): Qwen `Q4K 17408×5120`
  `dmmv` 80.8 µs ≈ **620 GB/s**; Gemma `Q4K 15360×3840` ≈ **560 GB/s**. The
  kernels are already at ~90% of the card's ~640 GB/s.
- Therefore the gap vs llama.cpp is **not** GEMV bandwidth. It is the ~9 ms/token
  of other GPU kernels (norms, attention rocBLAS, elementwise, copies) plus host
  gaps. Those small kernels are launch/latency bound, which is why F16 did not
  help.
- Reference scale check: candle-vllm reports ~49–58 tok/s for a dense 27B Q4 on
  an H100 (3.3 TB/s). Scaled to ~640 GB/s that is ~10–12 tok/s — the same order
  as the pinned build here, so candle-class engines are near their practical
  single-stream ceiling on this GPU; llama.cpp's kernels are simply better tuned.

## Fork patches

The server build uses a local fork. Two patches are included for reproducibility
(also committed in the local fork as `a928b84a` and `7e7ec140`):

- [`0001-rocm-non-contiguous-and-mixed-dtype.patch`](patches/0001-rocm-non-contiguous-and-mixed-dtype.patch)
  — materialise contiguous operands for rocBLAS, align dtypes in
  `rms_norm`/`layer_norm`.
- [`0002-rocm-mmvq-dmmv-autotune.patch`](patches/0002-rocm-mmvq-dmmv-autotune.patch)
  — per-shape autotune (`autotune.rs`), `CANDLE_ROCM_FORCE_MMVQ`, and the
  `copy2` / `copy_from_host_u32` in-place APIs used by the HIP-graph runner.

Apply on a clone of `xmiksay/candle` (`feat/rocm-backend`):

```bash
git am docs/patches/0001-rocm-non-contiguous-and-mixed-dtype.patch
git am docs/patches/0002-rocm-mmvq-dmmv-autotune.patch
```

Build against the local fork with a Cargo config override:

```toml
# /srv/dev/candle-local.toml
[patch.crates-io]
candle-core = { path = "/srv/dev/candle-rocm-upstream/candle-core" }
candle-nn = { path = "/srv/dev/candle-rocm-upstream/candle-nn" }
candle-transformers = { path = "/srv/dev/candle-rocm-upstream/candle-transformers" }
```

```bash
cargo build --release --features rocm --config /srv/dev/candle-local.toml
# optional graph runner:
cargo build --release --features rocm,hip-graph --config /srv/dev/candle-local.toml
```

Env switches: `CANDLE_ROCM_AUTOTUNE=0` (disable autotune),
`CANDLE_ROCM_FORCE_MMVQ=1` (force mmvq), `AURORA_HIP_GRAPH=1` (use the graph),
`AURORA_IGNORE_EOS=1` (fixed-token benchmarks).

## Caveats

- The autotune lives in the fork, not in this repo: the pinned-upstream build
  gives the lower numbers in `SUPPORTED.md` (Qwen ~12.6, Gemma ~35.7). The
  server build with the patches gives Qwen 18.2, Gemma ~38.
- Qwen 3.5/3.8's recurrent layer is a simplified EMA, not the exact Gated
  DeltaNet; performance numbers are valid, numerical parity is not claimed.
- Results are single-stream, batch 1; batching/paged KV (serving) is a different
  problem and not covered here.

## Next (not done)

- A real O(V) top-k kernel for sampling.
- GEMV launch tuning for `gfx1201` (nwarps / rows per block), closer to
  llama.cpp's kernels.
- Fusing the GDN step (Qwen 3.5/3.8) into fewer kernels.
