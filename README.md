# Aurora Rust Engine (`aurora-rust-engine`)

> **Pure Rust inference engine for modern image generation and diffusion architectures**

[![Rust](https://img.shields.io/badge/rust-stable-brightgreen.svg)](https://www.rust-lang.org/)
[![CUDA](https://img.shields.io/badge/cuda-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![FlashAttention](https://img.shields.io/badge/FlashAttention-2-orange.svg)](https://github.com/Dao-AILab/flash-attention)
[![License](https://img.shields.io/badge/license-Apache--2.0%20%2F%20MIT-blue.svg)](LICENSE)

---

## ⚡ Overview

`aurora-rust-engine` is a standalone, lightweight, and memory-efficient AI inference engine written entirely in pure Rust using [Candle](https://github.com/huggingface/candle), native FlashAttention-2 CUDA kernels, and hardware acceleration.

It provides a robust, zero-Python alternative for running generative diffusion models (Stable Diffusion XL, Pony XL, and future DiT/Flux architectures) with deterministic execution, in-memory zero-overhead LoRA weight merging, and sub-8GB VRAM footprint.

---

## ✨ Features

- **Pure Rust Native Inference**: Zero Python dependencies, zero PyTorch overhead, compiled directly to a native executable.
- **SDXL & Pony XL Full Support**: Seamless support for all `.safetensors` single-file checkpoints from Civitai and Hugging Face.
- **Native FlashAttention-2 Acceleration**: 9.5x faster attention computation with fused CUDA kernels under Windows MSVC and Linux.
- **Zero-Overhead In-Memory LoRA Merging**: Instant hot-patching of UNet and CLIP weights directly in GPU/CPU memory with 0 MB extra VRAM overhead.
- **Exact Penultimate Text Parity**: Custom penultimate hidden state extractors (`hidden_states[-2]`) for CLIP-L (Layer 11) and OpenCLIP-bigG (Layer 31) matching Hugging Face Diffusers bit-for-bit.
- **Seamless $C^\infty$ Cosine Tiled VAE**: 4-quadrant $72\times 72$ latent decoding with 128px smooth cosine cross-fade eliminating all tile seams.
- **Sub-8GB VRAM Footprint**: Text encoder CPU offloading and memory-mapped weight routers ensuring stable $\sim 7.5$ GB cruise VRAM on 12GB GPUs with **0% Windows shared memory swap**.
- **Deterministic Euler Sampler**: Continuous Euler Discrete & Karras noise scheduling.

---

## 🚀 Quick Start

### 1. Prerequisites
- Rust 1.80+ (`cargo`)
- NVIDIA GPU with CUDA Toolkit 12.x installed
- MSVC Build Tools (Windows) or GCC/Clang (Linux)

### 2. Build with FlashAttention-2 Acceleration
```bash
cargo build --release --features cuda,flash-attn
```

### 3. Generate an Image (Text-to-Image)
```bash
cargo run --release --bin test_single_gen --features cuda,flash-attn
```

### 4. Test LoRA Hot Weight Merging
```bash
cargo run --release --bin test_lora --features cuda,flash-attn
```

### 5. Run Comprehensive 15-Model Benchmark
```bash
cargo run --release --bin stress_test --features cuda,flash-attn
```

---

## 🧬 LoRA Integration Example

```rust
use candle_core::Device;
use aurora_rust_engine::{StableDiffusionXLPipeline, DiffusionParams};

fn main() -> anyhow::Result<()> {
    let device = Device::new_cuda(0)?;
    let mut pipeline = StableDiffusionXLPipeline::from_safetensors("checkpoint.safetensors", &device)?;

    // Hot-merge LoRA directly into model weights (< 10 seconds, 0 MB extra VRAM)
    pipeline.load_lora("style_lora.safetensors", 0.85)?;

    let params = DiffusionParams {
        prompt: "masterpiece, 1girl, cyberpunk city, vivid colors",
        negative_prompt: Some("blurry, low quality"),
        num_steps: 25,
        guidance_scale: 6.0,
        width: 1024,
        height: 1024,
        seed: 42,
    };

    let image = pipeline.generate(params, None)?;
    image.save("output_lora.png")?;

    // Unload LoRA to restore base checkpoint weights
    pipeline.unload_all_loras()?;

    Ok(())
}
```

---

## 🖼️ Image-to-Image (Img2Img) Example

```rust
use candle_core::Device;
use aurora_rust_engine::{StableDiffusionXLPipeline, Img2ImgParams};

fn main() -> anyhow::Result<()> {
    let device = Device::new_cuda(0)?;
    let mut pipeline = StableDiffusionXLPipeline::from_safetensors("checkpoint.safetensors", &device)?;

    let input_img = image::open("input.png")?.to_rgb8();

    let params = Img2ImgParams {
        prompt: "masterpiece, 1girl, golden radiant armor, fiery glowing orange hair",
        negative_prompt: Some("blurry, low quality"),
        image: input_img,
        strength: 0.60, // 0.0 = identity, 1.0 = full re-generation
        num_steps: 30,
        guidance_scale: 6.5,
        seed: 42,
    };

    let result = pipeline.generate_img2img(params, None)?;
    result.save("output_img2img.png")?;

    Ok(())
}
```

### Inpainting & Mask-Guided Diffusion
```rust
use aurora_rust_engine::{InpaintParams, StableDiffusionXLPipeline, select_device};

fn main() -> anyhow::Result<()> {
    let device = select_device()?;
    let mut pipeline = StableDiffusionXLPipeline::from_single_file("sdxl_base.safetensors", device)?;

    let base_image = image::open("input.png")?.to_rgb8();
    let mask_image = image::open("mask.png")?.to_luma8(); // White = edit, Black = keep

    let params = InpaintParams {
        prompt: "a wizard hat with golden stars",
        negative_prompt: Some("low quality, blurry"),
        image: base_image,
        mask: mask_image,
        mask_blur: 8,
        strength: 0.95,
        num_steps: 30,
        guidance_scale: 7.0,
        seed: 42,
    };

    let result = pipeline.generate_inpaint(params, None)?;
    result.save("output_inpaint.png")?;

    Ok(())
}
```

### Multi-ControlNet Spatial Guidance
```rust
use aurora_rust_engine::{compute_canny_edge_map, ControlNetModel, ControlNetParams, MultiControlNet, StableDiffusionXLPipeline, select_device};

fn main() -> anyhow::Result<()> {
    let device = select_device()?;
    let mut pipeline = StableDiffusionXLPipeline::from_single_file("sdxl_base.safetensors", device.clone())?;

    // 1. Extract Canny edge map in Pure Rust (< 12ms)
    let source_img = image::open("input.png")?.to_rgb8();
    let edge_map = compute_canny_edge_map(&source_img, 100.0, 200.0);

    // 2. Load ControlNet model and configure MultiControlNet container
    let cnet = ControlNetModel::from_safetensors("controlnet_canny_sdxl.safetensors", &device, candle_core::DType::F16)?;
    let mut multi_controlnet = MultiControlNet::new();
    multi_controlnet.add(cnet, 0.85); // 0.85 conditioning strength

    // 3. Generate with spatial edge alignment
    let params = ControlNetParams::new("cyberpunk warrior, masterpiece, highly detailed", edge_map);
    let result = pipeline.generate_controlnet(params, &multi_controlnet, None)?;
    result.save("output_controlnet.png")?;

    Ok(())
}
```

---

## 📊 Benchmark Summary (RTX 4070 Ti 12GB)

| Pipeline Component | Standard Attention | FlashAttention-2 | Speedup |
|---|---|---|---|
| Attention Kernels (per step) | 186.0 ms | **19.6 ms** | **9.5x** |
| SDXL UNet Denoising (50 steps) | ~42.5 s (1.18 it/s) | **25.8 s (1.94 it/s)** | **1.65x** |
| LoRA Hot Weight Merging Time | N/A | **< 9.0 s** | In-place |
| Img2Img VAE Encode Time | N/A | **< 0.15 s** | In-place |
| Inpainting Latent Blending | N/A | **< 0.05 ms/step** | Real-time |
| Pure Rust Canny Edge Extraction | N/A | **< 12 ms** | Real-time |
| Inference VRAM Allocation | 7.6 GB | 7.6 GB | **0 MB LoRA overhead** |

---

## 🗺️ Project Roadmap

See [`ROADMAP.md`](ROADMAP.md) for full technical specifications and development milestones:
- [x] **Milestone 1**: SDXL Core Pipeline & Conditioning Parity
- [x] **Milestone 2**: FlashAttention-2 Windows MSVC Kernel Fusion
- [x] **Milestone 3**: LoRA & LyCORIS Engine & In-Memory Hot Weight Merging
- [x] **Milestone 4**: Image-to-Image (Img2Img) Pipeline
- [x] **Milestone 5**: Inpainting & Outpainting Pipeline
- [x] **Milestone 6**: Multi-ControlNet (OpenPose, Depth, Canny) & IP-Adapter Conditioners
- [ ] **Milestone 7**: cuDNN Fused Convolutions & Direct GPU FP16 VAE Acceleration
- [ ] **Milestone 8**: PyO3 Python Bindings & REST / WebSocket Microservice

---

## 📄 License
Licensed under Apache-2.0 / MIT.
