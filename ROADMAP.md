# 🚀 Aurora Rust AI Inference Engine (`TransRust`)
## Technical Roadmap & Architectural Specification

**Version**: 0.2.0-dev  
**Language**: 100% Pure Rust (Candle, CUDA / cuDNN, FP16)  
**Target Hardware**: NVIDIA GPUs (Ada Lovelace, Ampere, Turing) / Apple Silicon (Metal) / CPU (AVX-512)

---

## 🧭 Executive Vision

`aurora-rust-engine` is an ultra-fast, memory-efficient, production-ready pure Rust AI inference engine designed as a lightweight, zero-Python alternative to Diffusers, ComfyUI, and Automatic1111 for Stable Diffusion XL, Pony XL, and next-generation diffusion architectures.

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
                    │   (Hot Weight Merging & Multi-LoRA)    │
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

## 🗺️ Implementation Milestones

### ✅ Milestone 1: SDXL Core Pipeline & Calibration (COMPLETED)
- [x] **SafeTensors Parser & Fast Weight Router**: Zero-copy memory-mapped loading with automatic key normalization.
- [x] **SDXL Penultimate Text Conditioning**:
  - Custom CLIP-L (Layer 11 hidden state extraction, QuickGELU).
  - Custom OpenCLIP-bigG (Layer 31 hidden state extraction, standard GELU, pooled EOS projection).
  - 100% numerical bit-for-bit parity against Hugging Face Diffusers.
- [x] **UNet 2D Condition Model**: Exact spatial cross-attention with optimized $Q$-prescaling and cached add-embeddings.
- [x] **Euler Discrete / Karras Noise Scheduler**: Stable deterministic sampling.
- [x] **Seamless $C^\infty$ Cosine Tiled VAE**: 4-quadrant $72\times 72$ decoding with 128px cross-fade eliminating all tile seams in $< 4.0$s.
- [x] **Memory Safety**: CPU text offloading and capped $< 7.7$ GB cruise VRAM (0% Windows shared memory swap).
- [x] **15-Model Stress Test**: 100% success rate across all SDXL base and Pony checkpoints.

---

### 🧬 Milestone 2: LoRA & LyCORIS Engine (NEXT PHASE)
**Objective**: Enable zero-overhead model personalization via Civitai / Hugging Face LoRA weights.

#### 1. Architecture:
- `src/lora/loader.rs`:
  - Support standard LoRA format (`lora_unet_...`, `lora_te1_...`, `lora_te2_...`).
  - Support LyCORIS / LoCon (`hada`, `lokr` decomposition).
  - Support automatic $\alpha / \text{rank}$ scaling factor detection.
- `src/lora/merger.rs`:
  - **Mode A: Hot Weight Merging (Zero Inference Cost)**:
    $$W_{\text{merged}} = W_{\text{base}} + \sum_{i=1}^N \lambda_i \frac{\alpha_i}{r_i} (A_i \times B_i)$$
    Modifies weights directly in memory before inference for maximum speed.
  - **Mode B: Dynamic Multi-LoRA Runtime**:
    Evaluates $(A \times B) x$ dynamically during cross-attention for multi-character generation.

#### 2. Pipeline Integration:
```rust
let mut pipeline = StableDiffusionXLPipeline::from_safetensors("model.safetensors", &device)?;
pipeline.load_lora("detail_enhancer.safetensors", 0.75)?;
pipeline.load_lora("character_style.safetensors", 0.85)?;
```

---

### 🎨 Milestone 3: Inpainting & Outpainting Engine
**Objective**: Mask-guided diffusion for object replacement, background alteration, and canvas expansion.

#### 1. Architecture:
- `src/pipelines/sdxl_inpaint.rs`:
  - `InpaintParams`: source image, binary mask, mask blur radius, denoising strength.
  - VAE latent encoding of original unmasked region.
  - Latent mask downsampling ($1/8$ spatial resolution).
  - In-loop noise injection matching Euler scheduler timesteps:
    $$z_t = M \odot z_t + (1 - M) \odot \text{add\_noise}(z_0, \epsilon, t)$$
- Automatic edge feathering and seamless VAE reconstruction.

---

### 🖼️ Milestone 4: Image-to-Image (Img2Img) & Upscaling
**Objective**: Creative transformation of existing sketches, renders, and photographs.

#### 1. Architecture:
- `src/pipelines/sdxl_img2img.rs`:
  - Input image VAE encoding to $z_0$.
  - Parametric `strength` $\in (0.0, 1.0]$.
  - Timestep truncation:
    $$t_{\text{start}} = \lfloor \text{num\_steps} \times (1.0 - \text{strength}) \rfloor$$
    $$z_{t_{\text{start}}} = \text{add\_noise}(z_0, \epsilon, t_{\text{start}})$$
- Fast High-Resolution Latent Fix (Hi-Res Fix): 512x512 latent generation -> VAE decode -> Bicubic/ESRGAN upscale -> 1024x1024 Img2Img refinement pass.

---

### ⚡ Milestone 5: Native CUDA FlashAttention-2 & cuDNN Kernel Acceleration
**Objective**: Bring inference speed from 1.20 it/s to **3.2+ it/s** (matching/exceeding warm PyTorch CUDA speed).

#### 1. Architecture:
- `src/diffusion/kernels/`:
  - Native custom CUDA FlashAttention-2 forward kernel binding.
  - Fused Multi-Head Attention without explicit $N \times N$ attention matrix materialization.
  - cuDNN Flash Cross-Attention integration for Ada Lovelace Tensor Cores.
- Eliminates 80% of VRAM memory bandwidth bottlenecks in UNet self/cross-attention blocks.

---

### 🕹️ Milestone 6: Multi-ControlNet & T2I-Adapters
**Objective**: Spatial structural control (Pose, Canny edges, Depth maps).

#### 1. Architecture:
- `src/controlnet/`:
  - SDXL ControlNet condition encoder (OpenPose, Depth, Canny).
  - Residual addition into UNet down-blocks and mid-block.
  - Multi-ControlNet weighted blending (`control_scales: Vec<f64>`).

---

### 🌐 Milestone 7: Production Serving & Bindings
**Objective**: Frictionless integration into web servers, desktop apps, and microservices.

#### 1. Architecture:
- **PyO3 Python Extension (`transrust-py`)**: Drop-in high-performance Python package.
- **REST & WebSocket Microservice**:
  - Axum / Tokio asynchronous HTTP server.
  - Real-time step progress streaming via WebSockets.
  - Embedded web UI for instant prompt engineering.

---

## 📊 Summary Timeline

| Target Date | Milestone | Key Deliverables |
|---|---|---|
| **Aug 2026** | **Phase 1** | SDXL Core Engine, Quality Calibration, 15-Model Pass 2 (Done) |
| **Aug 2026** | **Phase 2** | LoRA & LyCORIS Engine (Civitai compatibility & hot merging) |
| **Sep 2026** | **Phase 3** | Inpainting & Img2Img Pipelines |
| **Sep 2026** | **Phase 4** | Native CUDA FlashAttention-2 Kernels ($3.2+$ it/s) |
| **Oct 2026** | **Phase 5** | Multi-ControlNet & PyO3 Python Bindings / Web API |
