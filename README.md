# Aurora Rust Engine (`aurora-rust-engine`)

> **Pure Rust inference engine for image, text, speech & music generation — zero Python**

[![Rust](https://img.shields.io/badge/rust-stable-brightgreen.svg)](https://www.rust-lang.org/)
[![CUDA](https://img.shields.io/badge/cuda-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![FlashAttention](https://img.shields.io/badge/FlashAttention-2-orange.svg)](https://github.com/Dao-AILab/flash-attention)
[![License](https://img.shields.io/badge/license-Apache--2.0%20%2F%20MIT-blue.svg)](LICENSE)

---

## ⚡ Overview

`aurora-rust-engine` is a standalone, lightweight, and memory-efficient AI inference engine written entirely in pure Rust using [Candle](https://github.com/huggingface/candle), native FlashAttention-2 CUDA kernels, and hardware acceleration.

👉 **Looking for full documentation? See the complete [User & Developer Guide (USER_GUIDE.md)](USER_GUIDE.md)** for SDK examples, REST API payloads, scheduler configurations, and VRAM optimization tips.

It provides a robust, zero-Python alternative for running state-of-the-art generative models across **four modalities** — image (**SDXL / Pony**, **FLUX.1 & FLUX.2**, **Z-Image Turbo**), text (**CausalLM**: Qwen / Llama / Mistral / Gemma / DeepSeek), speech (**Whisper STT**, **Parler-TTS / Kokoro**) and **music** (**ACE-Step 1.5**) — with deterministic execution, in-memory zero-overhead LoRA weight merging, and a sub-8 GB VRAM footprint.

### 🧠 Supported Models

| Modality | Models |
|---|---|
| **Image** | SDXL / Pony XL (all single-file checkpoints), FLUX.1 `[dev/schnell]`, FLUX.2-Klein-4B / Klein-9B / Dev, Z-Image Turbo (S3-DiT 6B) |
| **Text (CausalLM)** | Qwen 2.5/3(.5), Llama 3, Mistral, Gemma 2/3, DeepSeek — GGUF (Q4_K_M/Q8_0) & SafeTensors |
| **Speech-to-Text** | Whisper Large-v3 / Turbo (native Slaney Mel + encoder/decoder) |
| **Text-to-Speech** | Parler-TTS, Kokoro-82M |
| **Text-to-Music** | ACE-Step 1.5 — Turbo / Base / SFT / XL (4B), 48 kHz stereo, lyrics, WAV / OGG / MP3 |

---

## ✨ Features

- **Pure Rust Native Inference**: Zero Python dependencies, zero PyTorch overhead, compiled directly to a native standalone executable.
- **Unified HuggingFace AutoModel Facade**: Standard `AutoModel::from_local` and `AutoModel::from_pretrained` interface supporting both generative diffusion pipelines and autoregressive text models.
- **CausalLM Text-to-Text Generation (Pure Rust)**: Autoregressive text generation with GPU KV-Cache supporting **Llama 3 / DeepSeek**, **Qwen 2.5/3.5**, **Gemma 2/3**, and **Mistral** in both **GGUF** (quantized Q4_K_M, Q8_0) and **SafeTensors** formats.
- **Flux.1 & Flux.2 Family Full Support**: Native implementation of Multimodal Diffusion Transformers (MMDiT) for **Flux.1 [dev/schnell]**, **Flux.2-Klein-4B**, **Flux.2-Klein-9B**, and **Flux.2-Dev Scaled** with exact 3D/4D Rotary Position Embeddings (RoPE).
- **Z-Image Turbo Realtime DiT (S3-DiT 6B)**: 100% pure Rust 4-step real-time inference with native FlashAttention-2, Qwen3-4B text conditioning, 16-channel VAE, and dynamic shift timestep scheduling.
- **FLUX.2 Multi-Image Reference Conditioning (Mode Édition)**: Native zero-adapter reference image guidance using 4D RoPE spatial-temporal coordinates $[T, Y, X, \text{Ref}]$ and VAE latent token injection for identity preservation across scenes.
- **Sub-7.5GB VRAM Flux Sequential Block Streaming**: Executes massive MMDiT models (3.88B to 12B parameters) with on-demand per-block GPU streaming and zero WDDM paging.
- **FLUX.2 Image-to-Image (Img2Img) & Inpainting with LoRA**: Contextual ODE transformation, LoRA adapter hot-splicing during block streaming, and sharp mask boundary preservation with flow-matching background re-injection.
- **Pure Rust 32-Channel & 16-Channel `FluxVaeEncoder` & `FluxVaeDecoder`**: Bit-exact VAE encoding and decoding with BatchNorm latent standardization.
- **SDXL & Pony XL Full Support**: Seamless support for all `.safetensors` single-file checkpoints from Civitai and Hugging Face.
- **Native FlashAttention-2 Acceleration**: Up to 9.5x faster attention computation with fused CUDA kernels under Windows MSVC and Linux.
- **Zero-Overhead In-Memory LoRA Merging**: Instant hot-patching of UNet and CLIP weights directly in GPU/CPU memory with 0 MB extra VRAM overhead.
- **Exact Penultimate Text Parity**: Custom penultimate hidden state extractors for CLIP-L, OpenCLIP-bigG, and multi-layer concat for Qwen3-4B (Layers 9/18/27).
- **Seamless $C^\infty$ Cosine Tiled VAE**: 4-quadrant $72\times 72$ latent decoding with 128px smooth cosine cross-fade eliminating all tile seams.
- **Deterministic Schedulers**: Continuous Euler Discrete, Flow Matching Rectified Euler ODE (`step_at` support for arbitrary start step), and DPM-Solver++ 2M Karras.
- **Whisper Speech-to-Text (Pure Rust)**: Bit-exact Whisper Large-v3 / Turbo transcription with a native Slaney Mel-filterbank, pure Rust WAV I/O, and multilingual decoding in `src/pipelines/whisper.rs`.
- **Neural Text-to-Speech (Parler-TTS & Kokoro-82M)**: Pure Rust TTS with voice/style descriptors and a native WAV encoder in `src/pipelines/tts.rs`.
- **ACE-Step 1.5 Text-to-Music (Pure Rust)**: Full 1D DiT + Flow-Matching + Oobleck 48 kHz stereo VAE port with bit-exact validation vs HuggingFace Diffusers. Supports **Turbo / Base / SFT / XL** variants, classifier-free guidance (APG), Qwen3 text+lyric conditioning, a **5Hz Qwen3 LM planner** (CoT metadata + constrained semantic audio-code generation driving the DiT), a bit-exact **5Hz FSQ audio codec**, a deterministic seeded RNG, and WAV / OGG Vorbis / MP3 export. See [`docs/ACESTEP_AUDIO_SPEC.md`](docs/ACESTEP_AUDIO_SPEC.md).

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

### 9. Transcribe Speech (Whisper STT)
```bash
cargo run --release --features cuda --bin test_whisper
```

### 10. Text-to-Speech (Parler-TTS / Kokoro)
```bash
cargo run --release --features cuda --bin test_tts
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

### 🎵 Audio & Music (RTX 4070 Ti 12 GB, `--release`, bf16)

| Task | Config | Result |
|---|---|---|
| ACE-Step 1.5 **Turbo** | 5 s, 8 steps | real-time |
| ACE-Step 1.5 **Turbo** | 180 s (3 min), 8 steps | **116.7 s end-to-end (1.54× real-time)** |
| ACE-Step 1.5 **Base** (CFG/APG) | 10 s, 30 steps | 2.7 s |
| Whisper STT | bit-exact vs reference | native Mel + encoder/decoder |
| Export | 48 kHz stereo | WAV · OGG Vorbis 192 kb/s · MP3 192 kb/s |

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
- [x] **Milestone 20**: Audio-to-Text & Speech Transcription — Whisper Large-v3 / Turbo (native Slaney Mel, bit-exact)
- [x] **Milestone 21**: Text-to-Audio & Music — ACE-Step 1.5 Turbo/Base/XL 1D DiT + Oobleck 48 kHz VAE + APG guidance + 5Hz Qwen3 LM planner + WAV/OGG/MP3 ([spec](docs/ACESTEP_AUDIO_SPEC.md))
- [x] **Milestone 22**: Neural Text-to-Speech — Parler-TTS (Kokoro-82M in progress)
- [ ] **Milestone 15** (proposed): LoRA Training Engine (pure Rust)
- [ ] **Milestone 18** (proposed): Vision-Language & Multimodal Models (VLM)
- [ ] **Milestone 19** (proposed): Spatio-Temporal Video Diffusion (Text-to-Video / Image-to-Video)
- [ ] **Milestone 21 (rest)**: Stable Audio Open & MusicGen backends

---

## 📄 License
Licensed under Apache-2.0 / MIT.
