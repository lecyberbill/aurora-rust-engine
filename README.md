# Aurora Rust Engine (`aurora-rust-engine`)

> **Pure Rust inference engine for image, text, speech & music — no Python runtime**

[![Rust](https://img.shields.io/badge/rust-stable-brightgreen.svg)](https://www.rust-lang.org/)
[![CUDA](https://img.shields.io/badge/cuda-12.x%20%2F%2013.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![ROCm](https://img.shields.io/badge/ROCm-10.0%20%2F%20HIP-red.svg)](docs/ROCM_SETUP.md)
[![FlashAttention](https://img.shields.io/badge/FlashAttention-2-orange.svg)](https://github.com/Dao-AILab/flash-attention)
[![License](https://img.shields.io/badge/license-Apache--2.0%20%2F%20MIT-blue.svg)](LICENSE)

---

## ⚡ Overview

`aurora-rust-engine` is a standalone, lightweight, and memory-efficient AI inference engine written entirely in pure Rust using [Candle](https://github.com/huggingface/candle), native FlashAttention-2 CUDA kernels, and hardware acceleration (NVIDIA CUDA & AMD ROCm/HIP).

👉 **Looking for full documentation? See the complete [User & Developer Guide (USER_GUIDE.md)](USER_GUIDE.md)** for SDK examples, REST API payloads, scheduler configurations, and VRAM optimization tips.  
👉 **Running on AMD GPUs (e.g. `gfx1201` / RDNA 3.5 / RDNA 4)? See the [AMD ROCm 10.0 & Candle Setup Guide (docs/ROCM_SETUP.md)](docs/ROCM_SETUP.md)**.
👉 **Wondering what actually works? See [SUPPORTED.md](SUPPORTED.md)** for the model → status table, validation metrics and known limitations.
👉 **Tuning on AMD ROCm? See the [ROCm performance notes (docs/PERF_ROCM.md)](docs/PERF_ROCM.md)** for the measured levers, rejected experiments, the fork patches and the roofline analysis.

It runs image, text, speech and music models in pure Rust, with CUDA and ROCm/HIP backends. Large diffusion models can be streamed block-by-block, and LoRA weights are merged in memory.

### 🧠 Supported Models

The full model → status table (validated parity / approximations / abandoned, with reasons) lives in **[SUPPORTED.md](SUPPORTED.md)**.

| Modality | Highlights |
|---|---|
| **Image / Video** | SDXL · Pony · FLUX.1 · FLUX.2 (Klein 4B/9B, Dev) · Z-Image Turbo · CogVideoX-2B |
| **Upscaling / VLM** | Real-ESRGAN · 4x-UltraSharp · Gemma 4 31B · Qwen2/2.5/3-VL · PaliGemma 2 · SmolVLM |
| **Text** | Qwen 2.5/3 · Gemma 3/4 · Phi-4 · Granite 4.2 · Llama 3 · Mistral · DeepSeek |
| **Speech / TTS** | Whisper · Moonshine · Parler-TTS · Kokoro · ChatTTS |
| **Music / Audio** | ACE-Step 1.5 · Stable Audio Open · MusicGen |

> Not everything works: e.g. gpt-oss (MXFP4), LFM 2.5 and Nemotron are **not** supported. Reasons in [SUPPORTED.md § Not supported](SUPPORTED.md#not-supported-abandoned).

---

## ✨ Highlights

- **Pure Rust, no Python runtime** — one native binary.
- **CUDA and AMD ROCm/HIP** backends (RDNA2/3/4, tested on `gfx1201`), plus Metal and CPU.
- **GGUF (Q4_K_M/Q8_0) and SafeTensors** loaders (mmap/slice-based).
- **Image**: SDXL/Pony, FLUX.1, FLUX.2 (Klein 4B/9B, Dev), Z-Image Turbo; img2img, inpaint, ControlNet, in-memory LoRA merge.
- **Video**: CogVideoX-2B (3D DiT).
- **VLM**: Gemma 4 31B, Qwen2/2.5/3-VL, PaliGemma 2, SmolVLM.
- **Text**: Qwen, Gemma, Phi-4, Granite, Llama, Mistral, DeepSeek with KV cache, streaming and sampling.
- **Speech/TTS**: Whisper, Moonshine; Parler-TTS, Kokoro, ChatTTS.
- **Audio**: ACE-Step 1.5 (cover/repaint/extract/lego/complete), Stable Audio Open, MusicGen.
- **Memory**: tiled attention, bounded tiling for upscalers, and sequential block streaming for large MMDiT models.
- **Performance**: fused ROCm kernels (RMSNorm/softmax/RoPE/sigmoid); opt-in HIP-graph decode (`--features hip-graph` + `AURORA_HIP_GRAPH=1`).

For per-module details and validation numbers, see **[SUPPORTED.md](SUPPORTED.md)** and **[USER_GUIDE.md](USER_GUIDE.md)**.

---

## 🚀 Quick Start

### 1. Prerequisites
- **Rust 1.80+** (`cargo`)
- **NVIDIA GPU**: CUDA Toolkit 12.x / 13.x + MSVC Build Tools (Windows) or GCC/Clang (Linux)
- **AMD GPU (Linux)**: ROCm 10.0+ / HIP toolchain (see [docs/ROCM_SETUP.md](docs/ROCM_SETUP.md))

### 2. Build

**NVIDIA GPU (CUDA + FlashAttention-2):**
```bash
cargo build --release --features cuda,flash-attn
```

**AMD GPU (ROCm / HIP):**
```bash
# Ensure ROCm environment variables are exported (see docs/ROCM_SETUP.md)
cargo build --release --features rocm
```

### 3. Launch the Pure Rust Interactive Studio ([Grio](https://github.com/lecyberbill/grio) Web UI)
```bash
cargo run --release --bin aurora_studio --features cuda,flash-attn,ui
```
Open **`http://127.0.0.1:7860`** to access the complete pure Rust Diffusion Studio with a multi-model
dropdown, per-model generation defaults, real-time progressive latent preview streaming, session
history gallery, and GPU telemetry powered by [Grio](https://github.com/lecyberbill/grio).

Models are declared in **`aurora_studio.json`** (no hard-coded paths) — see the end-user guide:
👉 **[docs/AURORA_STUDIO_GUIDE.md](docs/AURORA_STUDIO_GUIDE.md)**.

### 4. Run SOTA Grand Benchmark (All Optimizations Active)
```bash
cargo run --release --bin grand_benchmark --features cuda,flash-attn
```

### 5. Generate an Image via CLI
```bash
cargo run --release --bin test_single_gen --features cuda,flash-attn
```

### 6. Test LoRA Hot Weight Merging
```bash
cargo run --release --bin test_lora --features cuda,flash-attn
```

### 7. Run Comprehensive 15-Model Benchmark
```bash
cargo run --release --bin stress_test --features cuda,flash-attn
```

### 8. Generate Music (ACE-Step 1.5 — 48 kHz stereo, lyrics, WAV/OGG/MP3)
```bash
cargo run --release --features cuda --bin test_audio_diffusion -- --model-dir G:/models/Audio --duration 30 --steps 8 --lyrics-file scripts/lyrics_fr.txt --lang fr --format ogg --out ma_musique
```

### 8b. Music Editing (cover / repaint / extract / lego / complete)
```bash
# Extract the vocals stem (50 steps, CFG 7); add --cover-fsq for cover, --repaint-start/--repaint-end for repaint/lego, --low-vram for 5B XL.
cargo run --release --features cuda --bin test_acestep_tasks -- -m G:/models/Audio-sft -t extract --track vocals --src song.wav -o vocals.ogg -s 50 -g 7
```

### 9. Transcribe Speech (Whisper STT)
```bash
cargo run --release --features cuda --bin test_whisper
```

### 10. Text-to-Speech (Parler-TTS / Kokoro)
```bash
cargo run --release --features cuda --bin test_tts
```

### 11. CausalLM Text Generation (Pure Rust / RDNA4 & CUDA)
LLM / CausalLM inference (e.g., IBM Granite 4.2 30B, Qwen3.8-27B, DeepSeek-R1, Gemma 4, Llama 3) with native GGUF support:
- **IBM Granite 4.2 Support**: Native support for Granite dense architectures (`granite-4.2-30b-Q6_K.gguf`) with explicit `attention.scale` ($0.0078125$), chain-of-thought chat template (`<|im_start|>assistant\n<think>\n`), and official tokenizer auto-discovery.
- **AMD RDNA 4 (`gfx1201`) on ROCm 10**: portable `QMatMul` GEMM path. An experimental WMMA INT4 kernel (`__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12`) is available via the standalone `test_wmma_q4k` binary but is not yet wired into the model path. Measured Qwen3.8-27B hybrid (GDN): **~11.7 tok/s** decode (400 tokens, temp 0.7), **71.7 s** load.
- **Sampling & streaming**: `TextGenParams` (`temperature`, `top_p`, `top_k`, `repetition_penalty`, `stop_tokens`) and `generate_stream_with_params` are wired for Granite, Qwen 3.5/3.8 and Gemma; greedy is used when `temperature <= 0`.
- **NVIDIA CUDA**: High-throughput decoding via CUDA Tensor Cores with cross-platform compatibility and zero API regression.

```bash
# AMD ROCm / HIP (Linux / RDNA 4) — IBM Granite 4.2 Reasoning
TOP_P=0.95 ./target/release/test_text_gen /models/llm/granite-4.2-30b-Q6_K.gguf "Explain quantum mechanics in 3 bullet points:" 128 1.0

# AMD ROCm / HIP (Linux / RDNA 4) — General LLM
cargo run --release --features rocm --bin test_text_gen -- <path_to_model.gguf> "Explain quantum mechanics in 3 bullet points:" 128 0.7

# NVIDIA CUDA
cargo run --release --features cuda --bin test_text_gen -- <path_to_model.gguf> "Explain quantum mechanics in 3 bullet points:" 128 0.7
```

---

## 🧬 LoRA Integration Example

```rust
use candle_core::Device;
use aurora_rust_engine::{StableDiffusionXLPipeline, DiffusionParams};

fn main() -> anyhow::Result<()> {
    let device = Device::new_cuda(0)?;
    let mut pipeline = StableDiffusionXLPipeline::from_safetensors("checkpoint.safetensors", &device)?;

    // Merge LoRA directly into the model weights (in place)
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

### High-Resolution Disentangled Profiling
```rust
let (image, metrics) = pipeline.generate_with_metrics(params, None)?;
println!("{}", metrics.summary_report());
// Output: ⏱️ [Telemetry] UNet: 15.37s (30 steps, 512.42 ms/step, 1.95 it/s) | VAE: 4.73s | Text: 2.33s | Total: 22.57s
```

### Production REST & WebSocket Inference Server
Start the standalone async inference microservice:
```bash
cargo run --release --bin server --features cuda,flash-attn
```
- **Health Check**: `GET http://127.0.0.1:8080/api/v1/health`
- **Text-to-Image Generation**: `POST http://127.0.0.1:8080/api/v1/generate`
  ```json
  {
    "prompt": "futuristic cyberpunk pilot, 8k masterpiece",
    "steps": 30,
    "guidance_scale": 6.5,
    "width": 1024,
    "height": 1024
  }
  ```
- **WebSocket Streaming**: `ws://127.0.0.1:8080/api/v1/ws`

---

## 📊 Benchmark Summary (RTX 4070 Ti 12GB)

| Pipeline Component | Standard Attention | FlashAttention-2 | Speedup |
|---|---|---|---|
| Attention Kernels (per step) | 186.0 ms | **19.6 ms** | **9.5x** |
| SDXL UNet Denoising (50 steps) | ~42.5 s (1.18 it/s) | **25.8 s (1.94 it/s)** | **1.65x** |
| Pure UNet Step Speed | ~850 ms/step | **~512 ms/step (1.95 it/s)** | **1.65x** |
| LoRA Hot Weight Merging Time | N/A | **< 9.0 s** | In-place |
| Img2Img VAE Encode Time | N/A | **< 0.15 s** | In-place |
| Inpainting Latent Blending | N/A | **< 0.05 ms/step** | Real-time |
| Pure Rust Canny Edge Extraction | N/A | **< 12 ms** | Real-time |
| Inference VRAM Allocation | 7.6 GB | 7.6 GB | **0 MB LoRA overhead** |

### 🎵 Audio, Speech & Multimodal Benchmarks (RTX 4070 Ti 12 GB, `--release`)

| Modality & Task | Model & Configuration | Empirical Result | Hardware / Notes |
|---|---|---|---|
| **Super-Resolution (4x)** | RealESRGAN_x4 (66.8 Mo, tiled 256px) | **5.51 s** (512² → 2048²) | CUDA (RTX 4070 Ti), 0.14s load |
| **Speech-to-Text (ASR)** | Moonshine Tiny (108 Mo, 48k samples) | **1.60 s** | CUDA (RTX 4070 Ti), parity vs `transformers` |
| **Speech-to-Text (ASR)** | Whisper Large-v3 / Turbo | Parity vs PyTorch | Native Slaney Mel + greedy/beam |
| **Text Generation (LLM)** | TinyLlama-1.1B-Chat (2.2 Go F16) | **5.38 tok/s** (2.06s load) | CUDA (RTX 4070 Ti), KV-cache |
| **Image Synthesis (DiT)** | FLUX.2-Klein-4B (4 steps, 1024²) | **25.85 s** ODE, 3.64s VAE | CUDA (RTX 4070 Ti), 1.88 Mo output |
| ACE-Step 1.5 **Turbo** | 5 s, 8 steps | Real-time | CUDA / ROCm |
| ACE-Step 1.5 **Turbo** | 180 s (3 min), 8 steps | **116.7 s (1.54× real-time)** | CUDA / ROCm |
| ACE-Step 1.5 **Base** | 10 s, 30 steps (CFG/APG) | 2.7 s | CUDA / ROCm |
| Audio Export | 48 kHz stereo | WAV · OGG · MP3 | Zero C/FFmpeg dependencies |

### 🚀 LLM decode benchmarks (AMD ROCm, `gfx1201`)

Single stream, load excluded, 200–256 tokens. Full table and methodology in **[SUPPORTED.md](SUPPORTED.md#benchmarks--single-stream-decode-rocm-100-radeon-ai-pro-r9700-gfx1201)**.

| Model | Size | Decode | Note |
|---|---:|---:|---|
| Qwen 3.8 27B (Q4_K_M) | 17.4 GB | **16.8 tok/s** (256 tok) | split-K flash attention; GDN approximate |
| Gemma 3 12B (Q4_K_M) | 7.3 GB | **41.4 tok/s** (256 tok) | split-K flash attention |
| Phi-4 14B (Q4_K_M) | 8.5 GB | **22.9 tok/s** | Native chat template |
| Gemma 4 31B (UD-Q4_K_XL) | 18.8 GB | 11.3 tok/s | Text; VLM ~11.5 tok/s |

> gpt-oss and LFM/Nemotron are **not** supported in-engine — see [SUPPORTED.md § Not supported](SUPPORTED.md#not-supported-abandoned). As a reference, gpt-oss-20b runs at ~158 tok/s under ollama.

---

## 🗺️ Project Roadmap

See [`ROADMAP.md`](ROADMAP.md) for full technical specifications and development milestones:
- [x] **Milestone 1**: SDXL Core Pipeline & Conditioning Parity
- [x] **Milestone 2**: FlashAttention-2 Windows MSVC Kernel Fusion
- [x] **Milestone 3**: LoRA & LyCORIS Engine & In-Memory Hot Weight Merging
- [x] **Milestone 4**: Image-to-Image (Img2Img) Pipeline
- [x] **Milestone 5**: Inpainting & Outpainting Pipeline
- [x] **Milestone 6**: Multi-ControlNet (OpenPose, Depth, Canny) & IP-Adapter Conditioners
- [x] **Milestone 7**: Telemetry Profiler, Parameterized Kernel Dispatch & Adaptive VAE
- [x] **Milestone 8**: Production Async Axum Server & WebSocket Streaming
- [x] **Milestone 9**: Flux.1 MMDiT Diffusion Transformers (Dev/Schnell)
- [x] **Milestone 10**: Flux.2-Klein-4B MMDiT & Quality Parity
- [x] **Milestone 11**: Scaling to FLUX.2-Klein-9B & FLUX.2-Dev
- [x] **Milestone 12**: FLUX.2 Img2Img & Inpainting Pipeline (Pure Rust 16/32-ch VAE)
- [x] **Milestone 13**: FLUX.2-Dev LoRA Hot Merging (T2I & Img2Img)
- [x] **Milestone 14**: FLUX.2 Multi-Image Reference Conditioning / Mode Édition (4D RoPE)
- [x] **Milestone 16**: CausalLM LLM Text Generation & AutoModel Facade (Llama, DeepSeek, Qwen, Gemma, Mistral in GGUF/Safetensors)
- [x] **Milestone 17**: Z-Image Turbo S3-DiT 6B Pure Rust Realtime Pipeline with FlashAttention-2
- [x] **Milestone 20**: Audio-to-Text & Speech Transcription — Whisper Large-v3 / Turbo & Moonshine
- [x] **Milestone 21**: Text-to-Audio & Music — ACE-Step 1.5 Turbo/Base/XL 1D DiT, Stable Audio Open & MusicGen
- [x] **Milestone 22**: Neural Text-to-Speech — Parler-TTS, Kokoro-82M & ChatTTS
- [x] **Milestone 23**: Super-Resolution & Neural Upscaling — Real-ESRGAN, 4x-UltraSharp, SRVGGNetCompact
- [x] **Milestone 24**: Vision-Language & Multimodal Models (VLM) — Qwen2/2.5/3-VL, PaliGemma 2, SmolVLM
- [x] **Milestone 25**: Linux & AMD ROCm / HIP Acceleration (Cross-Platform Deployment)
- [x] **Milestone 26**: Cross-Platform Performance & Kernel Optimizations (CUDA, ROCm, Metal, CPU, RDNA4 WMMA)
- [x] **Milestone 27**: Universal Multi-Format & Multi-Modal Inference Engine (Zero-Copy Slicing, Gemma/DeepSeek, Cross-Modal Cache)
- [x] **Milestone 28**: Liquid AI LFM Hybrid Conv1D-MoE Architecture Support (`lfm2moe`)
- [ ] **Milestone 29 (HIGH PRIORITY)**: Native GPU Quantized GEMM (W4A16 / GGUF in VRAM) & Dynamic Layer Offloading (`n_gpu_layers`)

---

## 📄 License
Licensed under Apache-2.0 / MIT.
