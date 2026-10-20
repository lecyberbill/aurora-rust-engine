# Aurora Rust Engine (`aurora-rust-engine`)

> **Pure Rust inference engine for image, text, speech & music generation — zero Python**

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

It provides a robust, zero-Python alternative for running state-of-the-art generative models across **four modalities** — image (**SDXL / Pony**, **FLUX.1 & FLUX.2**, **Z-Image Turbo**), text (**CausalLM**: Qwen / Llama / Mistral / Gemma / DeepSeek), speech (**Whisper STT**, **Parler-TTS / Kokoro**) and **music** (**ACE-Step 1.5**) — with deterministic execution, in-memory zero-overhead LoRA weight merging, and a sub-8 GB VRAM footprint.

### 🧠 Supported Models

| Modality | Models |
|---|---|
| **Image (Diffusion)** | SDXL / Pony XL (all single-file checkpoints), FLUX.1 `[dev/schnell]`, FLUX.2-Klein-4B / Klein-9B / Dev, Z-Image Turbo (S3-DiT 6B) |
| **Video (DiT 3D)** | **CogVideoX-2B** (3D Spatio-Temporal DiT + Causal RoPE + Flow Matching ODE) |
| **Super-Resolution / Upscaling** | **Real-ESRGAN** (x2/x4/x8), **4x-UltraSharp**, **NMKD**, **SRVGGNetCompact** (Anime-6B) with Bounded Tiling & Seam Blending |
| **Vision-Language (VLM)** | **Qwen2-VL**, **Qwen2.5-VL**, **Qwen3-VL**, **PaliGemma 2**, **Gemma 3 Vision**, **SmolVLM** |
| **Text (CausalLM)** | Qwen 2.5/3, **Qwen 3.5/3.8 hybrid (Gated DeltaNet)**, **Gemma 3 & Gemma 4 (31B)**, **Microsoft Phi-4 (15B)**, **Liquid AI LFM 2.5 (ShortConv-MoE)**, Llama 3, Mistral, DeepSeek — GGUF (Q4_K_M/Q8_0) & SafeTensors |
| **Speech-to-Text** | Whisper Large-v3 / Turbo, **Moonshine** (native Mel + encoder/decoder) |
| **Text-to-Speech** | Parler-TTS, Kokoro-82M, **ChatTTS** (Conversational Prosody & Vocos) |
| **Text-to-Music & Editing** | ACE-Step 1.5 — Turbo / Base / SFT / XL, 48 kHz stereo, lyrics, WAV / OGG / MP3, **cover / repaint / extract / lego / complete** |
| **Text-to-Audio / Sound Design** | Stable Audio Open 1.0 — T5 + timing conds, 44.1 kHz stereo (≤47 s), SFX / ambient / music |
| **Text-to-Music (AR)** | MusicGen-small — T5 + autoregressive decoder + EnCodec 32 kHz |

---

## ✨ Features

- **Pure Rust Native Inference**: Zero Python dependencies, zero PyTorch overhead, compiled directly to a native standalone executable.
- **Universal Multi-Container Zero-Copy Loader**: Direct mmap and slice-based ingestion across **GGUF (v2/v3)**, **SafeTensors**, and **ONNX Initializers** with zero external Protobuf runtime dependencies and alignment-safe zero-copy buffers (`Cow<'_, [T]>`).
- **Unified HuggingFace AutoModel Facade**: Standard `AutoModel::from_local` and `AutoModel::from_pretrained` interface supporting both generative diffusion pipelines and autoregressive text models.
- **Vision-Language Multimodal Understanding (Pure Rust VLM)**: Complete SigLIP/ViT visual encoder + Multimodal MLP Projector + Causal LLM decoder supporting **Qwen2/2.5/3-VL**, **PaliGemma 2**, **Gemma 3 Vision**, and **SmolVLM** for Visual Question Answering (VQA) and dense image captioning.
- **CausalLM Text-to-Text Generation (Pure Rust)**: Autoregressive text generation with GPU KV-Cache supporting **Llama 3 / DeepSeek V3/R1**, **Qwen 2.5/3**, **Qwen 3.5/3.8 hybrid (Gated DeltaNet + full attention)**, **Gemma 2/4** (with unit-offset RMSNorm & attention logit softcapping), and **Mistral** in both **GGUF** (quantized Q4_K_M, Q8_0) and **SafeTensors** formats.
- **Token-by-Token Native Streaming Callback**: Clean zero-overhead streaming via `TokenCallback` without pulling web/async dependencies into the inference core.
- **Dynamic Sparse MoE Top-K Routing**: Normalized Top-K dispatch across independent and shared experts for DeepSeek V3/R1 and Mixtral architectures.
- **Super-Resolution & Neural Upscaling (Real-ESRGAN & 4x-UltraSharp)**: High-speed pure Rust neural upscaling (x2, x4, x8) powered by `RRDBNet` and `SRVGGNetCompact` with bounded-memory tiled processing, seam blending, and transparency preservation.
- **Flux.1 & Flux.2 Family Full Support**: Native implementation of Multimodal Diffusion Transformers (MMDiT) for **Flux.1 [dev/schnell]**, **Flux.2-Klein-4B**, **Flux.2-Klein-9B**, and **Flux.2-Dev Scaled** with exact 3D/4D Rotary Position Embeddings (RoPE).
- **Z-Image Turbo Realtime DiT (S3-DiT 6B)**: 100% pure Rust 4-step real-time inference with native FlashAttention-2, Qwen3-4B text conditioning, 16-channel VAE, and dynamic shift timestep scheduling.
- **FLUX.2 Multi-Image Reference Conditioning (Mode Édition)**: Native zero-adapter reference image guidance using 4D RoPE spatial-temporal coordinates $[T, Y, X, \text{Ref}]$ and VAE latent token injection for identity preservation across scenes.
- **Universal Multiplatform Hardware Acceleration**: Unified execution across **NVIDIA CUDA**, **AMD ROCm / HIP** (RDNA2/RDNA3/RDNA4), **Apple Silicon Metal**, and multi-core CPU.
- **Universal Tiled Scaled Dot-Product Attention (Chunked 512 + Online Softmax)**: Pure Rust cross-platform attention engine avoiding $O(N^2)$ memory spikes on arbitrary GPU/CPU architectures without external C++ bindings.
- **Async Double-Buffering & Host-to-Device Streaming**: Ring-buffered MPSC background worker pre-loading and dequantizing block $N+1$ on CPU during block $N$ GPU computation to completely mask PCIe I/O latency.
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
- **Moonshine Speech-to-Text (Pure Rust)**: Lightweight encoder-decoder ASR (`src/models/moonshine.rs`) — conv audio frontend + partial interleaved RoPE, validated vs `transformers` (encoder 6.9e-6, token ids identical).
- **Neural Text-to-Speech (Parler-TTS, Kokoro-82M & ChatTTS)**: Pure Rust TTS with voice/style descriptors in `src/pipelines/tts.rs` and conversational expressive synthesis (`[laugh]`, pauses) in `src/pipelines/chattts.rs` with high-speed Vocos vocoding.
- **ACE-Step 1.5 Text-to-Music & Editing (Pure Rust)**: Full 1D DiT + Flow-Matching + Oobleck 48 kHz stereo VAE port with bit-exact validation vs HuggingFace Diffusers. Supports **Turbo / Base / SFT / XL** variants, classifier-free guidance (APG), Qwen3 text+lyric conditioning, a **5Hz Qwen3 LM planner** (CoT metadata + constrained semantic audio-code generation driving the DiT), a bit-exact **5Hz FSQ audio codec**, a deterministic seeded RNG, and WAV / OGG Vorbis / MP3 export. The **source-audio tasks** — **cover** (with optional FSQ roundtrip), **repaint** (bit-exact), **extract**, **lego** and **complete** — run through `TaskRequest` + `AudioDiffusionPipeline::generate_task` (SFT-stems `Global:/Local:/Mask Control:` captions included). A **low-VRAM** loader (encoders on CPU) fits the 5B **XL** checkpoints on 12 GB. See [`docs/ACESTEP_AUDIO_SPEC.md`](docs/ACESTEP_AUDIO_SPEC.md) and [`docs/ACESTEP_TASKS_PLAN.md`](docs/ACESTEP_TASKS_PLAN.md).

- **Stable Audio Open 1.0 (Pure Rust)**: Text-to-audio / sound-design pipeline — T5-base prompt + Fourier `seconds_start`/`seconds_total` conditioners, a 24-layer **DiT** continuous transformer with GQA cross-attention and partial RoPE, an **EDM/SDE DPM-Solver++** scheduler (`v_prediction`), and the **AutoencoderOobleck** 44.1 kHz stereo VAE. All modules validated vs `diffusers`; see [`USER_GUIDE.md`](USER_GUIDE.md).
- **MusicGen Text-to-Music (Pure Rust)**: T5-base prompt → `enc_to_dec_proj` → 24-layer autoregressive decoder (learned positions, 4 LM heads) with the MusicGen **delay pattern** (CFG) → **EnCodec 32 kHz** decoder (weight-norm conv/convT + LSTM + residual blocks). Validated vs `transformers` (EnCodec 1.9e-6, decoder 5e-4); see [`USER_GUIDE.md`](USER_GUIDE.md).

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
LLM / CausalLM inference (e.g., Qwen3.8-27B, DeepSeek-R1, Llama 3) with native GGUF 4-bit (`Q4_K_M`) support:
- **AMD RDNA 4 (`gfx1201`) on ROCm 10**: portable `QMatMul` GEMM path. An experimental WMMA INT4 kernel (`__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12`) is available via the standalone `test_wmma_q4k` binary but is not yet wired into the model path. Measured Qwen3.8-27B hybrid (GDN): **~11.7 tok/s** decode (400 tokens, temp 0.7), **71.7 s** load.
- **Sampling & streaming**: `TextGenParams` (`temperature`, `top_p`, `top_k`, `repetition_penalty`, `stop_tokens`) and `generate_stream_with_params` are wired for Qwen 3.5/3.8 and Gemma; greedy is used when `temperature <= 0`. The dense Qwen3 non-stream path still uses its existing generator.
- **NVIDIA CUDA**: High-throughput decoding via CUDA Tensor Cores with cross-platform compatibility and zero API regression.
- **Qwen 3.5/3.8 tokenizer**: hybrid checkpoints require `assets/tokenizers/qwen35_tokenizer.json` (auto-resolved and EOS-checked against the GGUF).

```bash
# AMD ROCm / HIP (Linux / RDNA 4)
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

### 🎵 Audio, Speech & Multimodal Benchmarks (RTX 4070 Ti 12 GB, `--release`)

| Modality & Task | Model & Configuration | Empirical Result | Hardware / Notes |
|---|---|---|---|
| **Super-Resolution (4x)** | RealESRGAN_x4 (66.8 Mo, tiled 256px) | **5.51 s** (512² → 2048²) | CUDA (RTX 4070 Ti), 0.14s load |
| **Speech-to-Text (ASR)** | Moonshine Tiny (108 Mo, 48k samples) | **1.60 s** | CUDA (RTX 4070 Ti), bit-exact |
| **Speech-to-Text (ASR)** | Whisper Large-v3 / Turbo | Bit-exact vs PyTorch | Native Slaney Mel + greedy/beam |
| **Text Generation (LLM)** | TinyLlama-1.1B-Chat (2.2 Go F16) | **5.38 tok/s** (2.06s load) | CUDA (RTX 4070 Ti), KV-cache |
| **Image Synthesis (DiT)** | FLUX.2-Klein-4B (4 steps, 1024²) | **25.85 s** ODE, 3.64s VAE | CUDA (RTX 4070 Ti), 1.88 Mo output |
| ACE-Step 1.5 **Turbo** | 5 s, 8 steps | Real-time | CUDA / ROCm |
| ACE-Step 1.5 **Turbo** | 180 s (3 min), 8 steps | **116.7 s (1.54× real-time)** | CUDA / ROCm |
| ACE-Step 1.5 **Base** | 10 s, 30 steps (CFG/APG) | 2.7 s | CUDA / ROCm |
| Audio Export | 48 kHz stereo | WAV · OGG · MP3 | Zero C/FFmpeg dependencies |

### 🚀 Pure Rust LLM Inference Benchmarks (AMD ROCm / Radeon AI PRO R9700 `gfx1201`)

| Model | Checkpoint File | Size | Load Time | Generation Throughput | Test Case |
|---|---|---|---|---|---|
| **Qwen 3.8 27B** | `Qwen3.8-27B-Q4_K_M.gguf` | 17.44 GB | **71.68 s** | **11.73 tok/s** (401 tok / 34.19 s) | French poem, temp 0.7 (GDN path) |
| **Gemma 3 12B** | `gemma-3-12b-it-q4_k_m.gguf` | 7.30 GB | **3.85 s** | **17.80 tok/s** (65 tok / 3.65 s) | Ocean paragraph, greedy |
| **Gemma 4 31B** | `gemma-4-31B-it-UD-Q4_K_XL.gguf` | 18.82 GB | **7.92 s** | **11.25 tok/s** (65 tok / 5.78 s) | Ocean paragraph, greedy |
| **Phi-4 Reasoning Plus 15B** | `Phi-4-reasoning-plus-Q4_K_M.gguf` | 8.50 GB | **0.13 s** (lazy) | **8.49 tok/s** (49 tok / 5.77 s) | Native Phi template; concise reasoning → "Paris" |
| **LFM 2.5 8B** | `LFM2.5-8B-A1B-Q4_K_M.gguf` | 4.80 GB | **0.35 s** (lazy) | **3.15 tok/s** (65 tok / 20.65 s) | Degenerate output (`?` tokens); engine does not crash |

> **Qwen 3.5 / 3.8 note:** hybrid checkpoints declare `general.architecture = qwen35`. The engine auto-loads `assets/tokenizers/qwen35_tokenizer.json` (248k vocab) and refuses to start when its `<|im_end|>` id does not match the GGUF EOS. Sampling (temperature/top-p/top-k/repetition) is available for Qwen 3.5/3.8 and Gemma; greedy when `temperature <= 0`. Batch size is 1 and the GDN recurrence runs in F32 through portable `QMatMul` (no experimental WMMA kernel).
>
> Benchmarked on 2026-10-20 with the corrected CLI (actual tokens counted by the streaming callback). Phi-4 and LFM load times are metadata-only (lazy loading); LFM output is still degenerate, while Phi-4 uses its native GGUF chat template and a 1.1 repetition penalty.

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
