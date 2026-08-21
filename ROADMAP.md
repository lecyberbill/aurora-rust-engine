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

## 🚀 Upcoming Milestones

### 🎨 Milestone 5: Inpainting & Outpainting Pipeline
**Objective**: Mask-guided diffusion for targeted region editing, object replacement, and canvas expansion.

#### Technical Architecture:
- `src/pipelines/sdxl_inpaint.rs`:
  - `InpaintParams`: source image, binary mask, mask blur radius, denoising strength.
  - Latent mask downsampling ($1/8$ spatial resolution).
  - In-loop noise blending matching Euler scheduler timesteps:
    $$z_t = M \odot z_t + (1 - M) \odot \text{add\_noise}(z_0, \epsilon, t)$$
  - Seamless edge feathering and high-resolution VAE reconstruction.

---

### 🎛️ Milestone 6: Multi-ControlNet & IP-Adapter Conditioners
**Objective**: Precise spatial structural guidance via edge maps, depth, pose, and reference image prompts.

#### Technical Architecture:
- `src/diffusion/controlnet.rs`:
  - SafeTensors ControlNet loader (Canny, Depth, OpenPose).
  - Down-block zero-convolution feature injection:
    $$h_{\text{unet}}^{(i)} = h_{\text{unet}}^{(i)} + \text{ZeroConv}(h_{\text{control}}^{(i)})$$
- Multi-ControlNet simultaneous conditioning with independent weighting.
- IP-Adapter cross-attention projection for reference image styling.

---

### ⚡ Milestone 7: cuDNN Fused Convolutions & Kernel Compilation Dispatch
**Objective**: Further optimize UNet ResNet blocks and VAE 2D convolutions to achieve $> 3.5$ it/s on consumer GPUs.

#### Technical Architecture:
- cuDNN Graph API integration for fused Conv2d + GroupNorm + SiLU kernels.
- Parameterized pre-compiled CUDA kernel binaries (`.cubin` / `.ptx`) embedded in binary with dynamic runtime autotuning.
- Pure GPU FP16 VAE decoding with cuDNN acceleration reducing full $1024\times 1024$ decode time to $< 0.5$s.
- Clear separation in telemetry between pure diffusion loop it/s and total wall-clock time.

---

### 🌐 Milestone 8: Production Server & Ecosystem Bindings
**Objective**: High-throughput deployment across cloud infrastructure and desktop tools.

#### Technical Architecture:
- **PyO3 Python Bindings**: Direct drop-in Rust acceleration for Python ecosystems (`pip install aurora-diffusion`).
- **REST / WebSocket Server**: Embedded Axum async HTTP server supporting streaming progressive latents.
- **Node.js / WASM WebUI**: Lightweight local control panel with real-time hardware telemetry.
