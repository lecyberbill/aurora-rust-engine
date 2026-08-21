# Aurora Rust Engine (`aurora-rust-engine`)

> **Pure Rust inference engine for modern image generation and diffusion architectures**

---

## ⚡ Overview

`aurora-rust-engine` is a standalone, lightweight, and memory-efficient AI inference engine written entirely in pure Rust using [Candle](https://github.com/huggingface/candle) and native CUDA acceleration.

It provides a robust, zero-Python alternative for running modern generative image models (Stable Diffusion XL, Pony XL, and future DiT/Flux architectures) with predictable memory management and sub-8GB VRAM footprint.

---

## ✨ Features

- **Pure Rust Native Inference**: Zero Python runtime, zero PyTorch overhead, compiled to a single native binary.
- **SDXL & Pony XL Full Support**: Seamless support for all `.safetensors` single-file checkpoints from Civitai and Hugging Face.
- **Exact Penultimate Text Parity**: Custom penultimate hidden state extractors (`hidden_states[-2]`) for CLIP-L (Layer 11) and OpenCLIP-bigG (Layer 31) matching Hugging Face Diffusers bit-for-bit.
- **Seamless $C^\infty$ Cosine Tiled VAE**: 4-quadrant $72\times 72$ latent decoding with 128px smooth cosine cross-fade eliminating all tile seams in $< 4.0$ seconds.
- **Sub-8GB VRAM Footprint**: Text encoder CPU offloading and streaming VAE decoding ensuring stable $\sim 7.5$ GB cruise VRAM on 12GB GPUs with **0% Windows shared memory swap**.
- **Deterministic Euler Sampler**: Continuous Euler Discrete & Karras noise scheduling.

---

## 🚀 Quick Start

### 1. Prerequisites
- Rust 1.80+ (`cargo`)
- NVIDIA GPU with CUDA Toolkit installed
- MSVC Build Tools (Windows) or GCC/Clang (Linux)

### 2. Build with CUDA Acceleration
```bash
cargo build --release --features cuda
```

### 3. Generate an Image (Text-to-Image)
```bash
cargo run --release --bin test_single_gen --features cuda
```

### 4. Run Comprehensive 15-Model Benchmark
```bash
cargo run --release --bin stress_test --features cuda
```

---

## 📊 Benchmark Summary (15 SDXL Checkpoints on RTX 4070 Ti)

| Checkpoint | Weights Size | Load Time | Img 1 Duration | Img 2 Duration | Status |
|---|---|---|---|---|---|
| `animaPencilXL_v100` | 6.46 GB | 29.02s | 25.37s (1.18 it/s) | 24.59s (1.22 it/s) | ✅ Seamless |
| `aniverseXL_v30` | 6.46 GB | 29.47s | 25.37s (1.18 it/s) | 24.90s (1.20 it/s) | ✅ Seamless |
| `babesByStableYogiPony_v50` | 6.46 GB | 31.39s | 25.09s (1.20 it/s) | 30.48s (0.98 it/s) | ✅ Seamless |
| `Juggernaut-XL_v9_Photo_v2` | 6.62 GB | 18.74s | 25.55s (1.17 it/s) | 25.22s (1.19 it/s) | ✅ Seamless |
| `betterThanWords_v30` | 6.46 GB | 33.07s | 24.11s (1.24 it/s) | 29.23s (1.03 it/s) | ✅ Seamless |
| `bigLove_ponyV20` | 6.46 GB | 21.50s | 25.21s (1.19 it/s) | 24.43s (1.23 it/s) | ✅ Seamless |
| `realismarkPlus_realismarkPlus` | 13.35 GB | 53.45s | 25.09s (1.20 it/s) | 24.20s (1.24 it/s) | ✅ Seamless |
| `duchaitenPonyXLNo_v60` | 6.46 GB | 22.70s | 24.90s (1.20 it/s) | 24.52s (1.22 it/s) | ✅ Seamless |

*(See full report in [`outputs/stress_test/BENCHMARK_COMPARATIVE_REPORT.md`](outputs/stress_test/BENCHMARK_COMPARATIVE_REPORT.md))*

---

## 🗺️ Project Roadmap

See [`ROADMAP.md`](ROADMAP.md) for full technical specifications and development milestones:
- **Phase 2**: LoRA & LyCORIS Engine (Hot Weight Merging & Multi-LoRA runtime)
- **Phase 3**: Inpainting & Outpainting Pipeline
- **Phase 4**: Image-to-Image (Img2Img) with Denoising Strength & Hi-Res Fix
- **Phase 5**: Native CUDA FlashAttention-2 & cuDNN Kernel Acceleration
- **Phase 6**: Multi-ControlNet Support (OpenPose, Depth, Canny)
- **Phase 7**: PyO3 Python Bindings & REST / WebSocket Microservice

---

## 📄 License
Licensed under Apache-2.0 / MIT.
