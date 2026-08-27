# 🚀 Aurora Rust AI Inference Engine (`aurora-rust-engine`)
## Technical Roadmap & Architectural Specification

**Version**: 0.3.0-dev  
**Language**: 100% Pure Rust (Candle, CUDA / cuDNN, FlashAttention-2, FP16)  
**Target Hardware**: NVIDIA GPUs (Ada Lovelace, Ampere, Turing, Pascal) / Apple Silicon (Metal) / CPU (AVX-512)

---

## 🧭 Executive Vision

`aurora-rust-engine` is a pure Rust inference engine for modern image generation and diffusion architectures. Designed as a lightweight, zero-Python alternative to Diffusers, ComfyUI, and Automatic1111, it offers deterministic execution, sub-8GB VRAM footprint, in-memory zero-overhead LoRA hot weight merging, and native FlashAttention-2 fused CUDA kernels.

```
                    ┌────────────────────────────────────────┐
                    │       User API / CLI / Bindings        │
                    │   (Rust Crate, PyO3, REST, WebSocket)  │
                    └───────────────────┬────────────────────┘
                                        │
                    ┌───────────────────▼────────────────────┐
                    │      Pipeline Orchestrator Layer       │
                    │   ┌──────────────┬──────────────────┐  │
                    │   │ Text2Img     │ Img2Img          │  │
                    │   ├──────────────┼──────────────────┤  │
                    │   │ Inpainting   │ ControlNet / IP  │  │
                    │   └──────────────┴──────────────────┘  │
                    └───────────────────┬────────────────────┘
                                        │
                    ┌───────────────────▼────────────────────┐
                    │           LoRA Engine Core             │
                    │   (In-Memory Zero-Overhead Merging)    │
                    └───────────────────┬────────────────────┘
                                        │
          ┌─────────────────────────────┼─────────────────────────────┐
          │                             │                             │
┌─────────▼───────────┐       ┌─────────▼───────────┐       ┌─────────▼───────────┐
│ SDXL UNet 2D Model  │       │ Text Encoders       │       │ AutoEncoderKL (VAE) │
│ (FlashAttn-2 / F16) │       │ (CLIP-L & OpenCLIP) │       │ (Seamless Tiling)   │
└─────────────────────┘       └─────────────────────┘       └─────────────────────┘
```

---

## 🗺️ Completed Milestones

### ✅ Milestone 1: SDXL Core Pipeline & Conditioning (COMPLETED)
- [x] **SafeTensors Zero-Copy Memory Mapper**: Fast binary header parser and weight router with automatic key normalization.
- [x] **SDXL Penultimate Text Conditioning**:
  - Custom CLIP-L (Layer 11 hidden state extraction, QuickGELU activation).
  - Custom OpenCLIP-bigG (Layer 31 hidden state extraction, standard GELU, pooled EOS projection).
  - Bit-for-bit mathematical parity against Hugging Face Diffusers.
- [x] **UNet 2D Condition Model**: Complete cross-attention and spatial transformer blocks with scaled dot-product attention and add-embeddings.
- [x] **Euler Discrete / Karras Noise Scheduler**: Deterministic noise variance scaling.
- [x] **Seamless $C^\infty$ Cosine Tiled VAE**: 4-quadrant $72\times 72$ decoding with 128px cross-fade eliminating tile seams in $< 4.0$s.
- [x] **Memory Safety**: CPU text offloading and capped $< 7.7$ GB cruise VRAM on 12GB GPUs (**0% Windows shared memory swap**).
- [x] **15-Model Stress Test**: 100% success rate across 15 SDXL base and Pony checkpoints.

---

### ✅ Milestone 2: FlashAttention-2 Windows MSVC Kernel Fusion (COMPLETED)
- [x] **Windows MSVC Linker Integration**: Custom linker stub in `build.rs` resolving `stdc++.lib` / `msvcprt.lib` compatibility under MSVC.
- [x] **Scaled Dot-Product Attention Fusion**: Replacement of $O(N^2)$ memory-bound attention with hardware-optimized FlashAttention-2 kernels.
- [x] **Performance Benchmark**:
  - Pure attention compute accelerated **9.5x** (186ms down to 19.6ms per step).
  - End-to-end UNet denoising speed improved from **1.18 it/s to 1.94 it/s** (50 steps in 25.8s on RTX 4070 Ti).
  - Strict numerical parity validated with maximum absolute difference $\le 0.000244$.

---

### ✅ Milestone 3: LoRA & LyCORIS Engine & In-Memory Hot Weight Merging (COMPLETED)
- [x] **Multi-Format LoRA Parser** (`src/lora/loader.rs`):
  - Kohya-ss format (`lora_unet_...`, `lora_te1_...`, `lora_te2_...`).
  - Hugging Face Diffusers format.
  - Automatic rank $r$ and $\alpha$ extraction with rank-normalized scaling:
    $$\Delta W = \text{multiplier} \times \frac{\alpha}{r} (B \times A)$$
- [x] **Zero-Overhead In-Memory Hot Weight Merging** (`src/lora/merger.rs` & `src/pipelines/sdxl.rs`):
  - In-place GPU tensor weight updates for 2D Linear and 4D Conv2d modules in UNet and Text Encoders.
  - CPU-backed delta computation eliminating VRAM spikes during patching.
  - Instant $< 10$s patch application with **0 MB additional VRAM allocation** during inference.
  - Seamless unloading (`unload_all_loras`) with bit-for-bit base weight recovery.

---

### ✅ Milestone 4: Image-to-Image (Img2Img) Pipeline (COMPLETED)
- [x] **VAE Latent Encoding** (`src/diffusion/vae.rs`):
  - Image tensor normalization (scaled to $[-1.0, 1.0]$) and latent distribution encoding via `AutoEncoderKL::encode()`.
  - Seamless handling of arbitrary image resolutions with multiple-of-8 padding/resizing.
- [x] **Euler Discrete Strength Scheduling** (`src/pipelines/sdxl.rs`):
  - Timestep truncation and sigma-calibrated Gaussian noise injection:
    $$z_{\text{start}} = z_{\text{init}} + \sigma(t_{\text{start}}) \cdot \epsilon, \quad t_{\text{start}} = \lfloor N \times (1 - \text{strength}) \rfloor$$
  - Sub-step execution scaling with user-defined denoising strength $D \in (0.0, 1.0]$.
- [x] **Multi-Strength Verification Benchmark** (`src/bin/test_img2img.rs`):
  - Multi-strength validation ($D = 0.35, 0.60, 0.85$) on $1024\times 1024$ input with strict structural fidelity.

---

### ✅ Milestone 5: Inpainting & Outpainting Pipeline (COMPLETED)
- [x] **Mask Ingestion & Latent Downsampling** (`src/pipelines/sdxl.rs`):
  - Binary and grayscale mask processing with edge feathering / Gaussian blurring.
  - $8\times$ spatial downsampling with area averaging into float latent mask $M \in [0.0, 1.0]$.
- [x] **In-Loop Latent Noise Blending** (`src/pipelines/sdxl.rs`):
  - Per-step noise matching injecting exact background latents at each timestep $\sigma(t_i)$:
    $$z_t = (1 - M) \odot (z_{\text{orig}} + \sigma(t_i) \cdot \epsilon_{\text{bg}}) + M \odot z_{\text{denoised}, t}$$
  - Guarantees 100% preservation of unmasked regions with seamless transition at boundary edges.
- [x] **End-to-End Object Replacement Benchmark** (`src/bin/test_inpaint.rs`):
  - Validated with targeted object addition (wizard hat on photographic cat) preserving pixels outside mask.

---

### ✅ Milestone 6: Multi-ControlNet & IP-Adapter Conditioners (COMPLETED)
- [x] **Pure Rust Canny Edge Preprocessor** (`src/diffusion/controlnet.rs`):
  - High-speed Sobel gradient magnitude & dual-threshold edge extraction in **< 12ms**.
- [x] **ControlNet SDXL Architecture & Zero-Convolutions** (`src/diffusion/controlnet.rs`):
  - 6-layer convolutional hint embedding stem (`conv_in`, 5 intermediate stride-2 blocks, `conv_out`).
  - Down-block zero-convolutions with 9 spatial injection points and mid-block zero-conv.
- [x] **MultiControlNet Multi-Modal Container** (`src/diffusion/controlnet.rs`):
  - Simultaneous multi-conditioner aggregation with independent per-model weighting.
- [x] **UNet Skip Injection Engine** (`src/diffusion/unet_2d.rs` & `src/pipelines/sdxl.rs`):
  - `forward_with_controlnet()` and `generate_controlnet()` for spatial guidance.

---

### ✅ Milestone 7: Telemetry Profiler, Parameterized Kernel Dispatch & Adaptive VAE (COMPLETED)
- [x] **Disentangled High-Resolution Profiler** (`src/device.rs` & `src/pipelines/sdxl.rs`):
  - Clean separation between pure UNet diffusion loop speed (`it/s` and `ms/step`), Text Encoding, VAE Decoding, and total wall-clock time.
- [x] **Parameterized Dynamic Kernel Dispatch Engine** (`src/device.rs` : `KernelDispatchConfig`):
  - Runtime autotuning configuration (block dims, tile geometry, unroll factors) without recompilation.
- [x] **Vectorized RGB Buffer Assembly & Adaptive VAE Decoder** (`src/diffusion/vae.rs`):
  - Sub-millisecond direct RGB tensor assembly and non-paging tiled decoding under 7.6GB VRAM.

---

### ✅ Milestone 8: Production Server & Ecosystem Bindings (COMPLETED)
- [x] **High-Performance Async Axum Microservice** (`src/server/mod.rs` & `src/bin/server.rs`):
  - Endpoints: `POST /api/v1/generate`, `GET /api/v1/health`, `GET /health`, and WebSocket streaming `GET /api/v1/ws`.
- [x] **Permissive CORS & OpenAPI Ready DTOs** (`src/server/mod.rs`):
  - In-memory PNG base64 stream encoding, DTO serialization, and integrated telemetry.
- [x] **Multi-Threaded Hardware Orchestrator**:
  - Thread-safe `Arc<Mutex<StableDiffusionXLPipeline>>` request isolation across GPU inference steps.

---

## 🏆 All Core Engine Milestones (1 to 8) Completed Successfully!
The pure Rust Aurora inference engine is fully operational with FlashAttention-2, LoRA hot-merging, Img2Img, Inpainting, Multi-ControlNet, Disentangled Profiling, and Async Axum Server.

---

### ✅ Milestone 9: Flux.1 MMDiT Inference (COMPLETED)
Joint MMDiT (Multimodal Diffusion Transformer) inference for the **Flux.1** family (Schnell / Dev), driven by the pure Rust `DiffusionTransformer` (`src/diffusion/dit/`).

- [x] **MMDiT Architecture** (`src/diffusion/dit/flux.rs`, `blocks.rs`):
  - `DoubleStreamBlock` (joint image+text attention), `SingleStreamBlock` (5 double / 20 single for Klein, 19 / 38 for Flux.1).
  - Shared-modulation support for the compact Flux.2-Klein architecture.
  - `RMSNorm` QK-norm, GELU-tanh MLP, AdaLN-Zero gating, and interleaved RoPE.
- [x] **Sequential Block Streamer** (`src/diffusion/dit/streamer.rs`):
  - On-demand per-block weight streaming + drop for ultra-low-VRAM inference (< 6.5 GB peak); verified numerically identical to in-memory execution (max diff `0.000000`).
- [x] **Position Embeddings** (`src/diffusion/dit/embeddings.rs`):
  - 3-axis (Flux.1) and 4-axis (Flux.2) RoPE with configurable `theta`.
  - Timestep (+ optional guidance) embedding via `timestep_embedding` (time_factor `1000`).
- [x] **Text Encoder support**:
  - T5-XXL (256 tokens, 4096-dim) for Flux.1.
  - Qwen3-4B (512 tokens, 7680-dim) for Flux.2-Klein (`src/text/qwen.rs` — layers 9/18/27 → 7680).
- [x] **2D Patched-Image (non-patchified) vs 32-channel unpatchify pipelines** (`src/pipelines/flux.rs`):
  - Flux.1: 16-ch `patchify`/`unpatchify`; Flux.2-Klein: 128-ch packed → BN de-standardize → 32-ch unpatchify → VAE.
- [x] **Flux.1 numerical parity** verified via differential block tests.

---

## 🔭 Roadmap Restant / Next Milestones

The MMDiT module now covers **Flux.1** (Schnell/Dev) and **Flux.2-Klein-4B**. The following remain to reach full HF-Diffusers feature parity:

- [ ] **SD3.5 (Stable Diffusion 3.5 Large)** — MMDiT *already* exercised via `FluxConfig::sd35_large()` (24 DoubleStreamBlocks, 1536 hidden), but full pipeline integration (3-TEK T5-XXL + OpenCLIP, pooled text conditioning, 16-ch VAE, re-captioning, negative-prompt guidance) still pending.
- [ ] **FLUX.2-Klein-9B & FLUX.2-Dev** — Klein9B (hidden 4096 / heads 32 / 24 single), Flux2 (hidden 6144 / heads 48 / depth 8 / 48 single, guidance embed).
- [ ] **Flux.1-Pro / Kontext & Prompt-Upsampling** variants.
- [ ] **FP8 (Ada Lovelace) weight support** for DiT blocks.
- [ ] **Full HF-Hub integration** (`from_pretrained`) for MMDiT checkpoints.
- [ ] **Guidance embed / negative-prompt CFG** paths vs. guidance-distilled models (Klein uses `guidance_distilled=True`).
- [ ] **Reference-image / KV-cache edit** path (Flux.2 `encode_image_refs`, `denoise_cached`).
- [ ] **CUDA-Graph & batched multi-image** throughput tuning for MMDiT.

### ✅ Milestone 10: Flux.2-Klein-4B MMDiT Inference — Quality Parity (COMPLETED)
Bringing **FLUX.2-Klein-4B** (distilled 4-step MMDiT, 3.88B params, 5 double / 20 single blocks, shared-modulation) to full visual quality parity with the official `black-forest-labs/flux2` Python reference, driven end-to-end by the pure Rust `DiffusionTransformer`. Four root-cause bugs were isolated and fixed sequentially:

- [x] **Bug 1 — VAE BatchNorm de-standardization skipped** (`src/weights.rs`):
  `vae_var_builder()` discarded `bn.running_mean` / `bn.running_var`. Added `key.starts_with("bn.")` to the filter. Without this, `bn_mean()`/`bn_var()` returned `None` and the critical `latents * std + mean` step was skipped, leaving latents un-scaled.
- [x] **Bug 2 — Flow-Match scheduler sigma base** (`src/diffusion/schedulers/flow_match.rs`):
  Diffusers/Flux.2 uses `use_flow_sigmas=True` with base `linspace(1.0, 0.001, steps)` and *append* terminal `0`; the old code used `linspace(1.0, 1/steps)`, producing a huge final Euler jump (`dt = -0.822`) that caused over-oscillation and a tiled/mesh artifact. Fixed to `linspace(1, 0, num_steps + 1)` + `generalized_time_snr_shift(exp_mu, 1.0)`, yielding exact reference sigmas `[1.0, 0.9674, 0.9081, 0.7672, 0.0]`.
- [x] **Bug 3 — Text position-ids for 4D RoPE** (`src/diffusion/dit/embeddings.rs`):
  The 4th axis of `txt_ids` must encode the token index (`0..txt_len-1`), not `0`, matching `_prepare_text_ids`. Image ids remain `(T=0, row, col, Ref=0)`.
- [x] **Bug 4 — Final-layer AdaLN-Zero `scale`/`shift` swapped** (`src/diffusion/dit/flux.rs`):
  The reference `LastLayer` computes `x = (1 + scale) * norm(x) + shift` with `shift = chunks[0], scale = chunks[1]`. The Rust code had them reversed (`(1 + chunks[0]) * norm + chunks[1]`), which distorted the velocity field and produced the pervasive high-frequency "canvas"/grain texture. **This was the last bug** — restoring the correct order yields a clean, photorealistic render.

**Reference parity checklist (`klein.md` / official `flux2`):**
- `DoubleStreamBlock0` err `9e-6`, `SingleStreamBlock0` err `5e-5`, VAE err `4.8e-6` (isolated blocks) — ISO with PyTorch reference.
- Streamer path verified numerically identical to in-memory blocks (max diff `0.000000`) — not a source of divergence.
- Conventions kept: `theta=2000`, RoPE interleaved (`[seq,128]`, half-split rejected — produces checkerboard), F16 (BF16 produces identical output).

**Repro**: `cargo build --release --features cuda,flash-attn` then
`target\release\test_flux_inference.exe "G:\models\flux\flux2Klein_4b.safetensors" 4 "a golden retriever dog sitting in a flower meadow, sharp detailed photo"`.

**Output**: `outputs/flux_showcase/flux_klein_4b_1024_seed42.png` — clean, photorealistic.
