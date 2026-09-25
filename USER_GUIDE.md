# 📖 Aurora Rust Engine — User & Developer Guide

> **SOTA Pure Rust Generative AI Inference Engine for SDXL, Pony XL & Diffusion Transformers**  
> Powered by [Candle](https://github.com/huggingface/candle), [FlashAttention-2](https://github.com/Dao-AILab/flash-attention), and [Grio UI](https://github.com/lecyberbill/grio).

---

## 📑 Table of Contents

1. [Architecture & Key Highlights](#1-architecture--key-highlights)
2. [Installation & Requirements](#2-installation--requirements)
3. [Running the Interactive Web UI (Grio)](#3-running-the-interactive-web-ui-grio)
   - [Multi-Model Studio (`aurora_studio.json`)](#multi-model-studio-aurora_studiojson)
4. [Using Aurora in Rust Applications (SDK Reference)](#4-using-aurora-in-rust-applications-sdk-reference)
   - [Loading Models (Local & HuggingFace Hub)](#loading-models-local--huggingface-hub)
   - [Configuring Schedulers (DPM-Solver++, Euler, Flow-Matching)](#configuring-schedulers-dpm-solver-euler-ddim)
   - [Memory & VRAM Management Modes](#memory--vram-management-modes)
   - [Attention Backend Manette (FlashAttention-2)](#attention-backend-manette-flashattention-2)
   - [Text-to-Image Generation (SDXL & FLUX.1/FLUX.2)](#text-to-image-generation-flux1--flux2-mmdit-family)
   - [Z-Image Turbo Realtime DiT (4 Steps, FlashAttention-2)](#z-image-turbo-realtime-dit-s3-dit-6b)
   - [FLUX.2 Image-to-Image (Img2Img)](#flux2-image-to-image-img2img-transformation)
   - [FLUX.2 Inpainting & Masked Diffusion](#flux2-inpainting--masked-diffusion)
   - [FLUX.2 Multi-Image Reference Conditioning (Mode Édition, 4D RoPE)](#flux2-multi-image-reference-conditioning-mode-édition)
   - [SDXL Image-to-Image (Img2Img)](#image-to-image-img2img)
   - [SDXL Inpainting & Mask-Guided Diffusion](#inpainting--mask-guided-diffusion)
   - [Hot LoRA Merging](#hot-lora-merging)
   - [ControlNet (Canny Edge)](#controlnet-canny-edge)
   - [Audio-to-Text & Speech Transcription (Whisper Turbo)](#audio-to-text--speech-transcription-whisper)
   - [Text-to-Audio & Sound Diffusion (Stable Audio Open)](#text-to-audio--sound-diffusion-stable-audio-open)
   - [Text-to-Music (ACE-Step 1.5 — Turbo / Base / XL)](#text-to-music-ace-step-15--turbo--base--xl)
   - [Music Editing & Source-Audio Tasks (Cover / Repaint / Extract / Lego / Complete)](#music-editing--source-audio-tasks-cover--repaint--extract--lego--complete)
   - [Text-to-Speech (TTS / Parler-TTS & Kokoro-82M)](#text-to-speech-tts--parler-tts--kokoro-82m)
5. [REST API & WebSocket Server Reference](#5-rest-api--websocket-server-reference)
   - [Endpoints & JSON Payload Schema](#endpoints--json-payload-schema)
   - [Live Latent Preview via WebSocket](#live-latent-preview-via-websocket)
6. [CLI Binaries & Benchmark Suite](#6-cli-binaries--benchmark-suite)
7. [Hardware & Performance Tuning Guide](#7-hardware--performance-tuning-guide)

---

## 1. Architecture & Key Highlights

Aurora is designed from the ground up to replace heavy Python generative pipelines (PyTorch, Diffusers, ComfyUI) with a **high-performance, standalone, zero-Python binary**:

- **⚡ Sub-12s Generation**: Full $1024\times 1024$ SDXL generation in ~12.0s on RTX 4070 Ti (2.17 it/s) with DPM-Solver++ 2M Karras (18 steps).
- **🚀 FlashAttention-2 Fused CUDA Kernels**: Cuts attention computation down to 19.6ms per pass ($\times 9.5$ faster than standard SDPA). Enables **~2.0x faster** Flux.1/Flux.2 MMDiT denoising (21s → ~11.7s on Klein-4B).
- **🔒 Zero-Paging Seamless Tiled VAE**: Capped at $< 6.8\text{ GB}$ dedicated VRAM, preventing Windows WDDM shared RAM pagination.
- **🧬 Zero-Overhead In-Memory LoRA Merging**: Instant hot-patching of UNet and CLIP weights directly in GPU VRAM.
- **🌐 Native Hugging Face Hub Integration**: Direct automated download and caching of Safetensors checkpoints via `hf-hub`.
- **🎨 Native Reactive Web UI**: Powered by [Grio](https://github.com/lecyberbill/grio) with 1.5ms live latent streaming.

---

## 2. Installation & Requirements

### System Requirements
- **OS**: Windows 10/11 x64 or Linux (Ubuntu 22.04+, Debian, Arch, RHEL).
- **GPU**: NVIDIA GPU (RTX 3000 / 4000 series recommended, Pascal/Turing supported).
- **CUDA Toolkit**: CUDA 12.0+ with `nvcc` in PATH.
- **C++ Compiler**: MSVC Build Tools on Windows, `gcc`/`g++` on Linux.
- **Rust**: Rust 1.80+ (`rustup default stable`).

### Compilation
Clone the repository and build in release mode:

```bash
git clone https://github.com/lecyberbill/aurora-rust-engine.git
cd aurora-rust-engine

# Build with CUDA and FlashAttention-2 acceleration
cargo build --release --features cuda,flash-attn
```

#### Windows: building FlashAttention-2

`candle-flash-attn` compiles CUDA kernels via `nvcc`, which requires the MSVC C/C++ compiler
(`cl.exe`) on `PATH`. On Windows this is **not** set by default, so `nvcc` fails with
`Cannot find compiler 'cl.exe' in PATH`. Activate the MSVC developer environment first:

```bat
@cmd /c "call ""C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"" x64 && set ""CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"" && cargo build --release --features cuda,flash-attn"
```

Adjust the Visual Studio path / CUDA version to your install. On Linux/macOS no such step is needed.

---

## 3. Running the Interactive Web UI (Grio)

Aurora ships a native web studio (**Aurora Studio**) powered by [Grio](https://github.com/lecyberbill/grio):

> 📘 **End users**: see the dedicated **[Aurora Studio User Guide](docs/AURORA_STUDIO_GUIDE.md)** for a
> step-by-step, no-Rust walkthrough (install, launch, `aurora_studio.json`, VRAM tuning, FAQ).

```bash
cargo run --release --bin aurora_studio --features cuda,flash-attn,ui
```

Once loaded, navigate in your browser to:
👉 **`http://127.0.0.1:7860`**

### Multi-Model Studio (`aurora_studio.json`)

The studio is **fully config-driven**: no model path or label is hard-coded in the binary. It reads a
models `config.json` (default `aurora_studio.json` in the working directory, overridable with the
`STUDIO_CONFIG` environment variable). The dropdown is derived from the file and the first entry is
pre-loaded:

```json
{
  "models": [
    {
      "id": "sd35",
      "label": "SD 3.5 Large (CLIP-L+G+T5)",
      "family": "sd35",
      "checkpoint": "G:/models/SD3/sd3.5_large.safetensors",
      "text_encoder": {
        "kind": "sd35",
        "clip_l": "G:/models/clip/clip_l.safetensors",
        "clip_g": "G:/models/clip/clip_g.safetensors",
        "t5": "G:/models/clip/t5xxl_fp16.safetensors"
      },
      "vae": "G:/models/vae/sd3_vae.safetensors",
      "defaults": { "steps": 28, "guidance": 3.5, "width": 1024, "height": 1024 }
    }
  ]
}
```

Each entry has a stable `id` (used for loading), a presentation `label`, an optional `family` hint
slug (`sdxl` / `sd15` / `sd35` / `flux1` / `flux2` / `flux2-klein-4b` / ...), the `checkpoint`, the
optional external `text_encoder` (`kind`: `qwen3` / `mistral3` / `t5` / `sd35`) and `vae`, plus an
optional `defaults` block. **When a model is selected in the dropdown, its `defaults` are applied to
the controls** (steps, guidance, resolution, negative prompt); any omitted field keeps its current
value. An unknown `family` slug is a hard error (fail-fast, no silent guess).

An optional **`memory` block** tunes VRAM per model (SDXL today): `vae_tiling`, `vae_tile_size`,
`vae_tile_overlap`, `cpu_offload`, plus the reserved `low_vram_load`/`fp8_weights`. The important
lever is `vae_tile_size`: a large VAE tile inflates the decoded-tile im2col buffer and can push a
12 GB card into Windows WDDM shared-memory paging (10-40x slowdown). The engine default is now
`32×32` with `8` overlap, which keeps SDXL 1024×1024 at ~8.5 GB / ~12 s.

### Features Available in the Web Studio:
- **Model Switcher with Strict VRAM Ejection**: switching a model unloads the previous one, so only one model is resident in VRAM at a time.
- **Per-Model Generation Defaults**: steps / guidance / resolution / negative prompt read from the config and applied automatically on model switch.
- **Prompt & Negative Prompt Fields**: multi-line input; the negative prompt is forwarded to the pipeline (real CFG when `guidance != 1.0`, single-pass otherwise).
- **Resolution Selector**: 512×512 / 768×768 / 1024×1024.
- **Progressive Real-Time Latent Previews**: watch the image materialize live during the denoising process.
- **Session History Gallery**: view all generated images side-by-side.
- **Observability Output**: generation parameters (steps, resolution, seed) echoed after each run.

---

## 4. Using Aurora in Rust Applications (SDK Reference)

Add `aurora-rust-engine` to your `Cargo.toml`:

```toml
[dependencies]
aurora-rust-engine = { git = "https://github.com/lecyberbill/aurora-rust-engine.git", features = ["cuda", "flash-attn"] }
candle-core = "0.8.2"
```

### Loading Models (Local & HuggingFace Hub)

```rust
use candle_core::Device;
use aurora_rust_engine::StableDiffusionXLPipeline;

fn main() -> anyhow::Result<()> {
    let device = Device::new_cuda(0)?;

    // Option A: Load from a local single-file checkpoint (.safetensors)
    let mut pipeline = StableDiffusionXLPipeline::from_single_file(
        "<MODELS_DIR>/checkpoints/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
        device.clone(),
    )?;

    // Option B: Download & cache automatically from Hugging Face Hub (100% Pure Rust)
    let mut pipeline = StableDiffusionXLPipeline::from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        Some("sd_xl_base_1.0.safetensors"),
        device,
    )?;

    Ok(())
}
```

#### Multi-Format Weight Bricks (`WeightsSource`)

All model producers (text encoders, DiT transformer, VAE) read weights through a single
format-agnostic trait, `WeightsSource`. You never hard-code a format; you pick a **brick** that
implements the trait and hand it to the encoder. Every brick exposes the same methods —
`get_tensor`, `contains`, `raw_info`, `keys`:

```rust
use aurora_rust_engine::weights::{SafeTensorsArchive, WeightsSource};
use aurora_rust_engine::gguf::GgufWeights;
use aurora_rust_engine::text::Qwen3TextEncoder;
use candle_core::{Device, DType};

// Brick 1 — single safeTensors file (.safetensors)
let archive = SafeTensorsArchive::open("<MODELS_DIR>/qwen3_4b.safetensors")?;

// Brick 2 — multi-shard safeTensors (HF checkpoint split into model-0000N-of-0000M.safetensors)
// Open ALL *.safetensors in a directory as one logical archive, in sorted order.
let archive = SafeTensorsArchive::open_shards_dir("<MODELS_DIR>/FLUX.2-klein-9B_text_encoder")?;
// (or explicit list: SafeTensorsArchive::open_shards(&[a, b, c])?)

// Brick 3 — GGUF (llama.cpp) quantized weights, dequantized on the fly
let gguf = GgufWeights::open("<MODELS_DIR>/flux-unsloth-fp16.gguf")?;

// All bricks share the trait, so the SAME encoder takes any of them:
let qwen = Qwen3TextEncoder::from_archive(&archive, Some(tokenizer_path), &Device::Cpu, DType::F16)?;
```

The auto-detecting architecture (`QwenTextConfig::detect`, `MistralTextEncoder::from_weights`) reads
shapes from whichever brick you hand it, so 4B (Qwen3-4B, 7680), 9B (Qwen3-8B, 12288) and Dev
(Mistral-3-Small, 15360) are addressed uniformly.

---

### Model Origins (`ModelHub`) — Local, HuggingFace, Civitai

`ModelHub` is the second brick that separates **where a model comes from** from **how it is read**.
It resolves a `ModelOrigin` into local paths, downloading HF/mirror files when needed. Local paths
(the Civitai workflow — the user points at a file already on disk) are used as-is with no download:

```rust
use aurora_rust_engine::hub::{ModelHub, ModelOrigin};
use aurora_rust_engine::weights::SafeTensorsArchive;

let hub = ModelHub::from_env()?;                 // respects HF_ENDPOINT / HF_HOME / HF_TOKEN
let hub = ModelHub::with_cache_dir("<MODELS_DIR>/.cache")?;
let hub = ModelHub::with_endpoint("https://hf-mirror.com", "<MODELS_DIR>/.cache")?; // Citai-like mirror

// Origin A — a local file (e.g. downloaded from Civitai). No download, no network.
let dir = hub.resolve(&ModelOrigin::Local("<MODELS_DIR>/community/model_v1.safetensors".into()))?;

// Origin B — a HF (or mirror) repo, downloading any missing files into the cache.
let dir = hub.resolve(&ModelOrigin::Hf {
    repo: "Qwen/Qwen3-8B".into(),
    files: vec!["model-00001-of-00005.safetensors".into(),
                "model-00002-of-00005.safetensors".into(),
                "model-00003-of-00005.safetensors".into(),
                "model-00004-of-00005.safetensors".into(),
                "model-00005-of-00005.safetensors".into()],
    revision: None, // or Some("main" / a commit hash)
})?;

let archive = SafeTensorsArchive::open_shards_dir(&dir)?;
```

Environment controls (set them in your shell, no code change):
- `HF_ENDPOINT` — any HuggingFace-compatible mirror (e.g. a Citai/CF mirror).
- `HF_HOME` / `HF_HUB_CACHE` — where resolved files are cached.
- `HF_TOKEN` — authenticate for gated or private repos.

> **Design philosophy.** Aurora assembles bricks: an **origin** (`ModelHub`) supplies a path, a
> **weight format** (`WeightsSource`) reads it, and an **encoder** consumes it. To support a new
> community format you only implement `WeightsSource` once; to point at a new hub you only add a
> `ModelOrigin` variant — nothing downstream changes.

---

### Configuring Schedulers (DPM-Solver++, Euler, DDIM)

Aurora provides hot-switchable schedulers via dynamic dispatch:

```rust
// 1. SOTA DPM-Solver++ 2M Karras (Recommended: 18 - 20 steps, ~12s generation)
pipeline.use_dpm_solver();

// 2. Standard Euler Discrete Karras (Recommended: 25 - 30 steps)
pipeline.use_euler();

// 3. Deterministic DDIM (Recommended: 30 - 50 steps)
pipeline.use_ddim();
```

---

### Memory & VRAM Management Modes

Aurora implements a layered, zero-WDDM-paging memory strategy tailored to each generative architecture family, ensuring predictable VRAM bounds from modest 8GB consumer GPUs up to 24GB+ workstations:

#### Architecture Memory Matrix

| Model Architecture | Parameter Scale | Primary Memory Levers | Resident Peak VRAM | Zero-Paging Guarantee |
|---|---|---|---|---|
| **FLUX.1 [dev/schnell]** | 12.0 B | `SequentialBlockStreamer` + FlashAttention-2 | **< 7.5 GB** | ✅ Single-block GPU residency |
| **FLUX.2-Klein [4B/9B]** | 3.88 B / 9.0 B | `SequentialBlockStreamer` + FP8 dequant on-the-fly | **< 6.8 GB / 7.2 GB** | ✅ Block streamed directly from host mmap |
| **FLUX.2-Dev Scaled** | 12.0 B | `SequentialBlockStreamer` + FP8 Scaled + FlashAttention-2 | **< 7.4 GB** | ✅ No 32GB model in VRAM |
| **Z-Image Turbo** | 6.0 B (S3-DiT) | Host CPU VAE Decoder + FlashAttention-2 BF16 | **< 8.0 GB** | ✅ Eliminates DiT + VAE double spike |
| **SDXL / Pony XL** | 3.5 B (UNet) | $C^\infty$ Seamless Tiled VAE + CPU LoRA Delta Fusion | **< 6.8 GB** | ✅ 4-quadrant tiled decode (< 400MB) |

#### SDXL / Pony Memory Controls

```rust
// 1. Tiled VAE Decoding (Caps VAE VRAM to < 400 MB, eliminating WDDM paging)
pipeline.enable_vae_tiling(None);           // Default: 72x72 latents, 16 overlap (4 tiles)
pipeline.enable_vae_tiling(Some((64, 16))); // Custom tile size and overlap
pipeline.disable_vae_tiling();              // Direct single-pass decode (for 16GB+ GPUs)

// 2. CPU Offloading for Text Encoders (Saves 2.6 GB VRAM)
pipeline.enable_model_cpu_offload();        // Keeps CLIP-L & OpenCLIP-bigG in system RAM
pipeline.disable_model_cpu_offload();       // Keeps all models in GPU VRAM for max speed

// 3. Low-VRAM Sequential Loader (Eliminates memory allocation spikes during model loading)
pipeline.enable_low_vram_load();            // Sequential VarBuilder model construction
pipeline.disable_low_vram_load();

// 4. Ada Lovelace FP8 (E4M3) Precision Mode
pipeline.enable_fp8();                      // Stores weights in FP8 to halve bandwidth
pipeline.disable_fp8();                     // Standard FP16 mode
```

#### FLUX MMDiT Sequential Block Streaming

On the FLUX family (both Flux.1 and Flux.2), models range from 3.88B to 12B parameters (up to 32 GB on disk). Storing the full transformer in VRAM causes catastrophic WDDM paging to shared system RAM on consumer GPUs.
Aurora’s `SequentialBlockStreamer` solves this deterministically:
- Weights stay in host memory (mapped directly from the `.safetensors` file via zero-copy OS mmap).
- As the Euler loop executes, **only one double-block or single-block is transferred to GPU memory at a time**.
- The block executes its attention pass and is immediately freed.
- **Result** : Resident VRAM is capped below **7.5 GB**, numerically verified identical to in-memory execution (maximum absolute error `0.000000`).

#### High-End / Large VRAM Direct In-Memory Mode (Zero Streaming)

On workstations and servers equipped with **16GB to 24GB+ VRAM** (e.g. RTX 3090, RTX 4090, RTX 6000 Ada, A100, H100), **no streaming or memory tricks are required**. You can host the entire transformer directly in GPU VRAM for maximum inference throughput, cutting out per-block PCIe host-to-device transfers:

```rust
use aurora_rust_engine::pipelines::FluxPipeline;
use candle_core::Device;

let device = Device::new_cuda(0)?;

// Direct In-Memory loading: all double and single blocks reside permanently in GPU VRAM
let mut pipeline = FluxPipeline::from_single_file_in_memory(
    "<MODELS_DIR>/flux/fluxKlein4BPro_v10.safetensors", // or flux-2-klein-9b, flux2DevFp8Scaled
    device,
)?;

// FlashAttention-2 runs directly on GPU VRAM tensors with zero transfer overhead
pipeline.enable_flash_attn();
```

* **VRAM Footprint** :
  * **FLUX.2-Klein 4B** : ~7.8 GB VRAM.
  * **FLUX.2-Klein 9B (FP8)** : ~9.2 GB VRAM.
  * **FLUX.2-Dev (FP8 Scaled)** : ~14.5 GB VRAM (fits comfortably inside 24GB RTX 3090/4090).
* **SDXL / Pony XL** :
  * Default behavior is already **100% direct In-Memory** (`StableDiffusionXLPipeline::from_single_file`). Tiled VAE and CPU offload are purely optional opt-in flags for low-VRAM environments.

---

### Attention Backend Manette (FlashAttention-2)

For **Flux.1 / Flux.2 MMDiT pipelines**, the attention backend is a modular manette. Profiling showed the
denoising transformer's F32 SDPA attention is the dominant cost, so a FlashAttention-2 fast path was added.

```rust
// 1. Enable the FlashAttention-2 fast path (~2x faster denoise on CUDA, F16/BF16)
flux_pipeline.enable_flash_attn();

// 2. Disable it to use the stable F32 SDPA backend (default behaviour, model-safe)
flux_pipeline.disable_flash_attn();
```

- **Default (`disable_flash_attn` / `FLUX_FLASH_ATTN=0`)**: F32 `standard_sdpa`. Numerically safest and
  identical quality to the Python reference. Use this for debugging or if FlashAttention-2 is unavailable.
- **Enabled (`enable_flash_attn` / `FLUX_FLASH_ATTN=1`)**: runs attention through `candle_flash_attn` on CUDA
  for F16/BF16 inputs. A **safe auto-fallback** to the F32 path is taken automatically on any error
  (unsupported dtype/backend), so a misconfigured build never crashes.

| | F32 SDPA (default) | FlashAttention-2 |
|---|---|---|
| Denoise step (Klein-4B, 4608 tokens) | 4.87 s | **2.47 s (~2.0x)** |
| 4-step render (VAE 1.47 s) | ~21 s | **~11.7 s** |
| Quality | Reference | Identical (`mean_abs ≈ 0.0013`) |
| VRAM footprint | Low | Low (no extra residency) |

This manette **requires** the `--features flash-attn` cargo feature to take effect; without it the build
compiles cleanly and always uses the safer F32 path. Per-archive/call use is also possible via the
`FLUX_FLASH_ATTN` environment variable (`1`/`0`) for non-`FluxPipeline` callers.

---

### Text-to-Image Generation (SDXL)

```rust
use aurora_rust_engine::DiffusionParams;

let params = DiffusionParams {
    prompt: "masterpiece, ultra-detailed, cyberpunk samurai, rainy neo-tokyo street, 8k",
    negative_prompt: Some("lowres, blurry, bad anatomy, text, error"),
    num_steps: 18,
    guidance_scale: 6.5,
    width: 1024,
    height: 1024,
    seed: 42,
};

// Optional progress callback with live latent preview
let (image, metrics) = pipeline.generate_with_metrics(params, Some(|step, total, _latent| {
    println!("Step {}/{}", step, total);
}))?;

image.save("output.png")?;
println!("Generated in {:.2}s ({:.2} it/s)", metrics.total_wallclock_ms / 1000.0, metrics.unet_it_per_sec);
```

---

### Text-to-Image Generation (Flux.1 & Flux.2 MMDiT Family)

`aurora-rust-engine` includes native pure-Rust support for the Black Forest Labs **Flux.1 [dev/schnell]** and **Flux.2-Klein [4B/9B] / Flux.2-Dev** Multimodal Diffusion Transformer (MMDiT) architectures:

```rust
use aurora_rust_engine::pipelines::flux::FluxPipeline;
use aurora_rust_engine::traits::DiffusionParams;
use aurora_rust_engine::diffusion::vae_flux::FluxVaeDecoder;
use aurora_rust_engine::text::Qwen3TextEncoder;
use candle_core::Device;

let device = Device::new_cuda(0)?;

// 1. Load Flux.2-Klein Checkpoint with Sequential Block Streaming (< 7.5GB VRAM Peak)
let mut flux_pipeline = FluxPipeline::from_single_file_streaming("<MODELS_DIR>/flux/fluxKlein4BPro_v10.safetensors", device.clone())?;
flux_pipeline.enable_flash_attn();

// 2. Attach Qwen3 Prompt Encoder and Flux.2 32-Channel VAE Decoder
// (Auto-detected if embedded in checkpoint, or attached externally via .safetensors)

// 3. Configure Diffusion Parameters (4 steps for Schnell / Klein, 20-28 steps for Dev)
let params = DiffusionParams {
    prompt: "a magnificent lion sitting on a rock in savanna during sunset, cinematic lighting, 8k",
    negative_prompt: None,
    num_steps: 4,
    guidance_scale: 1.0,
    width: 1024,
    height: 1024,
    seed: 42,
};

// 4. Generate high-fidelity image in pure Rust (< 7.5GB VRAM footprint)
let (image, metrics) = flux_pipeline.generate_with_metrics(params, None::<fn(usize, usize, &candle_core::Tensor)>)?;
image.save("flux_lion.png")?;
```

#### Flux.2-Klein-9B (Qwen3-8B, multi-file shards)

The **Flux.2-Klein-9B** model is auto-detected from its checkpoint key counts
(8 double / 24 single blocks, 4096 hidden). Its official text encoder is **Qwen3-8B**
(hidden 4096 -> **12288** conditioning dim), which ships as a **multi-file safetensors shard split**
on HuggingFace. `aurora-rust-engine` loads every shard in a directory transparently:

```rust
use aurora_rust_engine::weights::SafeTensorsArchive;
use aurora_rust_engine::text::Qwen3TextEncoder;

// 1. Point at the directory containing HF shards (model-00001-of-0000N.safetensors, ...).
//    All *.safetensors files are opened as one logical archive (no single-file checkpoints exist).
let enc_dir = std::path::Path::new("<MODELS_DIR>/FLUX.2-klein-9B_text_encoder");
let archive = SafeTensorsArchive::open_shards_dir(enc_dir)?;

// 2. The architecture (hidden 4096, 36 layers, heads/kv, vocab 151936) and the 12288-dim
//    text context (3 concatenated layers) are auto-detected from the weights — no hardcoding.
let qwen8b = Qwen3TextEncoder::from_archive(&archive, Some(std::path::Path::new("qwen_tokenizer.json")),
    &Device::Cpu, DType::F16)?;
flux_pipeline.set_qwen3(qwen8b);

// 3. Klein-9B is guidance-distilled, 4 steps, CFG 1.0 (like Klein-4B).
let params = DiffusionParams {
    prompt: "a gorgeous portrait of an arctic fox with sapphire blue eyes in a snowy forest at twilight, 8k",
    negative_prompt: None,
    num_steps: 4, guidance_scale: 1.0, width: 1024, height: 1024, seed: 42,
};
let (image, _) = flux_pipeline.generate_with_metrics(params, None::<fn(usize, usize, &candle_core::Tensor)>)?;
image.save("flux_klein_9b_fox.png")?;
```

#### Flux.2-Dev (Mistral-3-Small, guidance)

**Flux.2-Dev** is a *guidance* model (like Flux.1-Dev): it has a `guidance_in` embedder and 48 single
blocks, so it is auto-detected distinctly from the Klein family. Its official text encoder is
**Mistral-3-Small** (hidden 5120 -> **15360** conditioning dim = 3 layers preserved in full). It runs
with a non-unit `guidance_scale` and typically 8-28 steps:

```rust
use aurora_rust_engine::text::Mistral3TextEncoder;

let mistral = Mistral3TextEncoder::from_safetensors(
    "<MODELS_DIR>/mistral_3_small_flux2_fp8.safetensors",
    Some(std::path::Path::new("mistral_tokenizer.json")), Device::Cpu, DType::F16)?;
flux_pipeline.set_mistral(mistral);

let params = DiffusionParams {
    prompt: "a gorgeous portrait of an arctic fox with sapphire blue eyes in a snowy forest at twilight, 8k",
    negative_prompt: None,
    num_steps: 20, guidance_scale: 3.5, width: 1024, height: 1024, seed: 42,
};
```

> **Note** — Replace `<MODELS_DIR>` with your local models directory. With the official 40-layer Mistral VLM (`FLUX.2-dev_text_encoder`), $\theta = 10^9$ RoPE phase, guidance scaled by $1000\times$, and spatial-first patch unpacking, **FLUX.2-Dev produces crystal-clear, photorealistic 1024x1024 renders**.

---

### Z-Image Turbo Realtime DiT (S3-DiT 6B)

`aurora-rust-engine` provides a **100% pure Rust** implementation of the **Z-Image Turbo** (S3-DiT 6B) text-to-image architecture with **native FlashAttention-2**:

* **Architecture** : 30-layer Scalable Single-Stream Transformer (S3-DiT), Qwen3-4B text encoder, 16-channel VAE.
* **Euler Flow-Match Solver** : Exact discrete velocity inversion (`pred_v.neg()?`) and dynamic shift timestep schedule ($\mu = m \times \text{seq\_len} + b$).
* **Unified Sequence Order** : Canonical diffusers alignment `[x_img, text_feat]` with 3D RoPE spatial-temporal phase.
* **Performance** : 4-step generation in **29.9s total** (~7.47s / step) on CUDA BF16.

```rust
use aurora_rust_engine::pipelines::z_image_turbo::ZImageTurboPipeline;
use aurora_rust_engine::traits::DiffusionParams;
use candle_core::Device;

let device = Device::new_cuda(0)?;

// 1. Load All-In-One (AIO) FP8 Checkpoint or standalone weights
let mut pipeline = ZImageTurboPipeline::from_single_file(
    "<MODELS_DIR>/z_image/z_image_turbo_aio_fp8.safetensors",
    device,
)?;

// 2. Enable FlashAttention-2 (7.4x acceleration: 55.2s -> 7.47s / step)
pipeline.enable_flash_attn();

// 3. Configure 4-step fast sampling
let params = DiffusionParams {
    prompt: "a photorealistic portrait of an old sailor with a weathered face and white beard, dramatic lighting, 8k",
    negative_prompt: None,
    num_steps: 4,
    guidance_scale: 1.0,
    width: 512,
    height: 512,
    seed: 42,
};

// 4. Generate image
let (image, metrics) = pipeline.generate_with_metrics(params, None::<fn(usize, usize, &candle_core::Tensor)>)?;
image.save("zimage_turbo_test.png")?;
println!("Generated in {:.2}s (4 steps)", metrics.total_wallclock_ms / 1000.0);
```

**CLI Command**:
```powershell
$env:CUDARC_CUDA_VERSION = "12080"
cargo run --release --features cuda,flash-attn --bin test_zimage_turbo
```

---

### FLUX.2 Image-to-Image (Img2Img) Transformation

```rust
use aurora_rust_engine::traits::Img2ImgParams;

let init_image = image::open("flux_lion.png")?.to_rgb8();

let params = Img2ImgParams {
    prompt: "a majestic lion wearing a golden crown and diamond armor sitting on a rock during sunset, photorealistic, 8k",
    negative_prompt: None,
    image: init_image,
    strength: 0.65, // Denoising strength: 0.0 = original image, 1.0 = completely regenerated
    num_steps: 4,
    guidance_scale: 1.0,
    seed: 42,
};

let (transformed_image, metrics) = flux_pipeline.generate_img2img(params, None::<fn(usize, usize, &candle_core::Tensor)>)?;
transformed_image.save("flux2_img2img_lion_crown.png")?;
```

---

### FLUX.2 Inpainting & Masked Diffusion

```rust
use aurora_rust_engine::traits::InpaintParams;

let base_image = image::open("flux_lion.png")?.to_rgb8();
let mask_image = image::open("lion_head_mask.png")?.to_luma8(); // 255 = area to inpaint, 0 = keep unchanged

let params = InpaintParams {
    prompt: "a majestic lion wearing an intricate glowing golden crown with emerald gems, photorealistic, 8k",
    negative_prompt: None,
    image: base_image,
    mask: mask_image,
    mask_blur: 0,
    strength: 0.85,
    num_steps: 4,
    guidance_scale: 1.0,
    seed: 42,
};

let (inpainted_image, metrics) = flux_pipeline.generate_inpaint(params, None::<fn(usize, usize, &candle_core::Tensor)>)?;
inpainted_image.save("flux2_inpaint_lion_crown.png")?;
```

---

### FLUX.2 Multi-Image Reference Conditioning (Mode Édition)

`aurora-rust-engine` introduces native support for **FLUX.2 Multi-Image Reference Conditioning** (Mode Édition) using the **4-axis Rotary Position Embeddings (4D RoPE)** architecture:

* **No external adapters (Zero IP-Adapter overhead)** : The model utilizes native joint attention across the canvas, prompt, and reference tokens.
* **4D RoPE Identity Coordinates** :
  $$\text{RoPE Axis} = [T=0, Y, X, \text{Ref\_ID}]$$
  * Main generated canvas tokens: $\text{Ref\_ID} = 0$.
  * Reference image $k$ tokens: $\text{Ref\_ID} = k$.
* **Full VAE Integration** : Automatically encodes reference images via `FluxVaeEncoder` (32-channel), applies $2\times 2$ patchification, and standardizes latents with `BatchNorm` statistics.

```rust
use aurora_rust_engine::pipelines::FluxPipeline;
use aurora_rust_engine::traits::DiffusionParams;
use aurora_rust_engine::diffusion::vae_flux::{FluxVaeDecoder, FluxVaeEncoder};
use candle_core::Device;

let device = Device::new_cuda(0)?;

// 1. Initialize FLUX.2 Pipeline with Streamer
let mut pipeline = FluxPipeline::from_single_file_streaming(
    "<MODELS_DIR>/flux/flux2DevFp8Scaled_fp8Scaled.safetensors",
    device.clone(),
)?;
pipeline.enable_flash_attn();

// 2. Attach VAE Encoder & Decoder
pipeline.set_vae(decoder);
pipeline.set_vae_encoder(encoder);

// 3. Load Reference Image(s)
let ref_image = image::open("character_sheet.png")?;

let params = DiffusionParams {
    prompt: "a majestic portrait of the character standing in a futuristic metropolis at sunset, cinematic, 8k",
    negative_prompt: None,
    num_steps: 12,
    guidance_scale: 3.5,
    width: 512,
    height: 512,
    seed: 42,
};

// 4. Generate with Reference Conditioning (supports multiple reference images in slice)
let (image, metrics) = pipeline.generate_with_refs(
    params,
    &[ref_image],
    None::<fn(usize, usize, &candle_core::Tensor)>,
)?;

image.save("character_edited.png")?;
```

**CLI Command**:
```powershell
$env:CUDARC_CUDA_VERSION = "12080"
$env:REF = "outputs/flux_showcase/flux_dev_img2img.png"
$env:PROMPT = "a majestic arctic fox sitting gracefully under golden autumn leaves, close-up portrait, photorealistic, 8k"
cargo run --release --features cuda,flash-attn --bin test_flux_reference_edit
```

---

### Image-to-Image (Img2Img)

```rust
use aurora_rust_engine::Img2ImgParams;

let init_image = image::open("input.png")?.to_rgb8();

let params = Img2ImgParams {
    prompt: "masterpiece, cybernetic armor, golden glowing accents, cinematic lighting",
    negative_prompt: Some("lowres, blurry, distorted"),
    image: init_image,
    strength: 0.65, // 0.0 = original image, 1.0 = completely new image
    num_steps: 25,
    guidance_scale: 6.5,
    seed: 12345,
};

let output = pipeline.generate_img2img(params, None)?;
output.save("output_img2img.png")?;
```

---

### Inpainting & Mask-Guided Diffusion

```rust
use aurora_rust_engine::InpaintParams;

let base_image = image::open("original.png")?.to_rgb8();
let mask_image = image::open("mask.png")?.to_luma8(); // White = region to replace, Black = keep

let params = InpaintParams {
    prompt: "a majestic golden crown with emeralds and rubies",
    negative_prompt: Some("low quality, blurry"),
    image: base_image,
    mask: mask_image,
    strength: 0.85,
    num_steps: 25,
    guidance_scale: 7.0,
    seed: 42,
};

let inpaint_result = pipeline.generate_inpaint(params, None)?;
inpaint_result.save("output_inpaint.png")?;
```

---

### Hot LoRA Merging

Merge LoRA adapters into base model weights in GPU VRAM with **0 MB additional runtime memory overhead**:

```rust
// Load and merge multiple LoRAs with custom scaling weights
pipeline.load_lora("loras/detail_enhancer.safetensors", 0.8)?;
pipeline.load_lora("loras/cyberpunk_style.safetensors", 0.6)?;

// Verify active LoRAs
for lora in pipeline.loaded_loras() {
    println!("Loaded LoRA: {} (weight: {})", lora.name, lora.weight);
}

// Unload all LoRAs and restore original base weights
pipeline.unload_all_loras()?;
```

#### LoRA on the Flux MMDiT family (Flux.1, Flux.2-Klein & FLUX.2-Dev)

The same `load_lora` / `unload_all_loras` API works transparently across all MMDiT pipelines (`FluxPipeline`),
accepting **all major LoRA key formats** (Diffusers `transformer.*`, BFL `lora_unet_double_blocks.*`, Kohya `diffusion_model.*`):

- **FLUX.2-Dev Scaled Support** : Fully compatible with FP8 weights (`flux2DevFp8Scaled_fp8Scaled.safetensors`). LoRA deltas are dynamically dequantized and spliced into each block during streaming without extra memory footprint.
- **Combined Img2Img + LoRA** : Enables applying fine-tuned styles and character concepts on top of existing base images.

```rust
use aurora_rust_engine::pipelines::FluxPipeline;
use aurora_rust_engine::traits::Img2ImgParams;

let mut pipeline = FluxPipeline::from_single_file_streaming(
    "<MODELS_DIR>/flux/flux2DevFp8Scaled_fp8Scaled.safetensors",
    device,
)?;
pipeline.enable_flash_attn();

// Hot-merge FLUX.2-Dev LoRA (e.g., dragon tattoo style or character)
pipeline.load_lora("<MODELS_DIR>/loras/flux2tatooR32.safetensors", 0.85)?;

// Apply on top of existing image via Img2Img
let base_img = image::open("subject.png")?.to_rgb8();
let params = Img2ImgParams {
    prompt: "a muscular man with intricate dragon tattoo on his chest and arm, studio portrait, photorealistic, 8k",
    negative_prompt: None,
    image: base_img,
    strength: 0.50, // Preserves 50% base structure, transforms remaining with prompt + LoRA
    num_steps: 12,
    guidance_scale: 3.5,
    seed: 42,
};

let (transformed_image, metrics) = pipeline.generate_img2img(params, None::<fn(usize, usize, &candle_core::Tensor)>)?;
transformed_image.save("flux_dev_img2img_lora.png")?;
```

Because the transformer weights are loaded in a low-VRAM block-by-block streamer, each LoRA delta is
spliced into a block's weight (including the fused `img_attn.qkv` / `linear1` Q·K·V slabs) as that block
is streamed in — so LoRAs add **no runtime VRAM overhead** and are compatible with sub-8GB inference.

#### Stacking & re-weighting multiple LoRAs

You can load any number of LoRAs — each keeps its **own weight** (`multiplier`, 0.0–1.0):
loRA deltas whose weights overlap are **added**; LoRAs targeting different weights are **combined**.
`set_lora_weight` / `unload_lora` accept a LoRA identified by **file path**, **basename**, or **numeric
index**, so you can re-weight or remove an individual LoRA on the fly without re-loading the file:

```rust
// Load three LoRAs with individual weights (20% / 40% / 30%)
pipeline.load_lora("loras/a_style.safetensors", 0.20)?;
pipeline.load_lora("loras/b_theme.safetensors", 0.40)?;
pipeline.load_lora("loras/c_subject.safetensors", 0.30)?;

// Re-weight "b_theme" (by path, basename, or index) to 50% without reloading
pipeline.set_lora_weight("loras/b_theme.safetensors", 0.50)?;

// Remove just the style LoRA (by basename) at runtime
pipeline.unload_lora("a_style")?;

// ...generate...

// Reset everything back to the base checkpoint weights
pipeline.unload_all_loras()?;
```

The same `load_lora` / `set_lora_weight` / `unload_lora` / `unload_all_loras` API is available on both
the `StableDiffusionXLPipeline` and the `FluxPipeline`.

#### LoRA API reference

| Method | Description |
|---|---|
| `load_lora(path, multiplier)` | Load a LoRA. `multiplier` (0.0–1.0) is its contribution weight; overlapping deltas are summed, disjoint ones combined. |
| `set_lora_weight(id, multiplier)` | Re-weight an already-loaded LoRA on the fly without reloading the file. |
| `unload_lora(id)` | Remove a single loaded LoRA at runtime, restoring the others' combined effect. |
| `unload_all_loras()` | Remove all LoRAs and restore the exact base checkpoint weights. |
| `loaded_loras()` | List the currently loaded LoRAs (path, multiplier, alpha/rank scaling). |

`id` in `set_lora_weight` / `unload_lora` may be a **file path**, a **basename** (filename without
extension), or a **numeric index** (position in load order). Path matching normalises `\` and `/`.

**Example — SDXL with three LoRAs, re-weighted & one removed:**
```rust
let mut pipeline = StableDiffusionXLPipeline::from_safetensors("base.safetensors", &device)?;

pipeline.load_lora("loras/style.safetensors", 0.20)?;
pipeline.load_lora("loras/subject.safetensors", 0.40)?;
pipeline.load_lora("loras/lighting.safetensors", 0.30)?;

pipeline.set_lora_weight("loras/lighting.safetensors", 0.50)?; // bump lighting to 50%
pipeline.unload_lora("subject")?;                              // drop the subject LoRA

let img = pipeline.generate(params)?;
```

> **Note** — the Flux pipeline applies LoRA deltas via a low-VRAM per-block streamer, so LoRAs add
> **0 MB runtime VRAM**; SDXL applies them in place on the GPU weights. Both restore the base weights
> exactly on `unload_all_loras()`.

#### In-Session LoRA Cycling (Generating With & Without LoRA in the Same Session)

In interactive services (such as the Grio Web Studio or REST API backends), an essential requirement is to generate images **with** a LoRA and **without** a LoRA alternately in the same process lifetime **without reloading the multi-gigabyte base model checkpoint**.

Aurora is engineered specifically for deterministic in-session cycling:

```rust
// -------------------------------------------------------------
// STEP 1: Generate clean baseline image (pure base model)
// -------------------------------------------------------------
let (baseline_img, _) = pipeline.generate_with_metrics(params.clone(), None)?;
baseline_img.save("session_1_baseline.png")?;

// -------------------------------------------------------------
// STEP 2: Hot-merge LoRA on the fly (< 0.1s overhead)
// -------------------------------------------------------------
pipeline.load_lora("loras/character_dragon_tattoo.safetensors", 0.85)?;

let (lora_img, _) = pipeline.generate_with_metrics(params.clone(), None)?;
lora_img.save("session_2_with_lora.png")?;

// -------------------------------------------------------------
// STEP 3: Unload LoRA — restores base weights instantly
// -------------------------------------------------------------
// On Flux MMDiT:
pipeline.unload_all_loras(); // Clears streamer deltas; subsequent passes stream base Safetensors

// On SDXL:
pipeline.unload_all_loras()?; // Subtracts deltas in-place on GPU in < 0.05s

// -------------------------------------------------------------
// STEP 4: Generate baseline again — 100% Bit-Exact Identity
// -------------------------------------------------------------
let (restored_img, _) = pipeline.generate_with_metrics(params, None)?;
restored_img.save("session_3_restored_baseline.png")?;

// Numerical check: session_3_restored_baseline.png == session_1_baseline.png
```

##### Why Aurora Guarantees Zero Memory Leak & Zero Residual Bias:
1. **On FLUX MMDiT (`FluxPipeline`)**:
   - Model weights on disk/mmap are **never modified**.
   - `load_lora` attaches delta mappings to `SequentialBlockStreamer`.
   - `unload_all_loras()` sets `lora_deltas = None`.
   - The very next block streamed into GPU memory reads unmodified canonical weights directly from the Safetensors archive. There is mathematically zero risk of numerical drift or memory leak.
2. **On SDXL (`StableDiffusionXLPipeline`)**:
   - `unload_all_loras()` computes $(-\Delta W)$ from the cached delta map and performs in-place addition $W \leftarrow (W + \Delta W) - \Delta W$.
   - The delta cache is then purged (`self.lora_manager.clear()`), returning the pipeline to its pristine initial state.

---

### ControlNet (Canny Edge)

```rust
use aurora_rust_engine::{ControlNetModel, MultiControlNet, ControlNetInput, CannyDetector};

let canny_detector = CannyDetector::default();
let canny_edges = canny_detector.detect(&input_rgb)?;

let controlnet = ControlNetModel::load("controlnet_canny_sdxl.safetensors", device.clone(), DType::F16)?;
let mut multi_controlnet = MultiControlNet::new();
multi_controlnet.add_model(controlnet);

let inputs = vec![ControlNetInput {
    hint: canny_edges,
    conditioning_scale: 0.85,
    start_step_percent: 0.0,
    end_step_percent: 0.80,
}];

let output = pipeline.generate_with_controlnet(params, &multi_controlnet, &inputs, None)?;
```

---

### Text-to-Text & CausalLM Generation (Llama, DeepSeek, Qwen, Gemma, Mistral)

Aurora provides a unified `AutoModel` facade supporting autoregressive text generation for both **GGUF** (`.gguf`) and **SafeTensors** (`.safetensors`) models with GPU KV-Cache:

```rust
use aurora_rust_engine::models::{AutoModel, GenerationModel, downcast_model};
use aurora_rust_engine::models::text::TextModel;
use candle_core::{Device, DType};

fn main() -> anyhow::Result<()> {
    let device = Device::new_cuda(0)?;

    // Automatically detects Llama 3, DeepSeek, Qwen 2.5/3.5, Gemma 2/3, Mistral
    let model = AutoModel::from_local(
        "E:/LMSTUDIO_MODELES/lmstudio-community/DeepSeek-R1-Distill-Llama-8B-GGUF/DeepSeek-R1-Distill-Llama-8B-Q8_0.gguf",
        device,
        DType::F16,
    )?;

    let mut lock = model.lock().unwrap();
    let text_model = downcast_model::<TextModel>(&mut *lock)
        .ok_or_else(|| anyhow::anyhow!("Failed to downcast model to TextModel"))?;

    // Generate response (prompt, max_tokens, temperature)
    let response = text_model.generate(
        "Explain what makes Rust a modern language in 2 sentences.",
        64,
        0.7,
    )?;

    println!("Response:\n{}", response);
    Ok(())
}
```

---

### Audio-to-Text & Speech Transcription (Whisper)

Aurora incorporates pure-Rust audio transcription powered by OpenAI's **Whisper** family (`whisper-large-v3`, `whisper-turbo`):

* **No external Python/FFmpeg required** : Mel-filterbank spectrogram extraction (80/128 Mel frequency bins at 16 kHz) is calculated in pure Rust ([`src/audio/mel.rs`](file:///d:/image_to_text/TransRust/src/audio/mel.rs)).
* **Multilingual & Timestamp Decoding** : Transcribes 99+ languages with autoregressive Transformer decoder.

```rust
use aurora_rust_engine::audio::WavAudio;
use aurora_rust_engine::pipelines::WhisperPipeline;
use candle_core::{DType, Device};

let device = Device::new_cuda(0)?;

// Load Whisper STT Pipeline from Safetensors
let mut whisper = WhisperPipeline::from_files(
    "models/whisper/config.json",
    "models/whisper/model.safetensors",
    "models/whisper/tokenizer.json",
    device,
    DType::F16,
)?;

// Load WAV audio file (pure Rust RIFF decoder)
let audio = WavAudio::load_wav("voice_recording.wav")?;

// Transcribe to text with language specification
let result = whisper.transcribe(&audio, Some("fr"), false)?;
println!("Transcription: \"{}\"", result.text);
println!("Duration: {:.2}s (processed in {:.1}ms)", result.duration_seconds, result.inference_time_ms);
```

---

### Text-to-Audio & Sound Diffusion (Stable Audio Open)

Aurora supports high-fidelity audio generation from text prompts using 1D continuous diffusion models like **Stable Audio Open 1.0**:

* **Continuous 1D DiT** : Generates music, ambient soundscapes, and foley sound effects at **44.1 kHz stereo** (up to 47 seconds).
* **Timing Conditioners** : Precision duration control through `seconds_start` and `seconds_total` timing embeddings.
* **1D Audio VAE** : AutoencoderOobleck / DAC 64-channel 1D VAE decoding directly into uncompressed 44.1 kHz audio waveforms.

```rust
use aurora_rust_engine::pipelines::audio::StableAudioPipeline;
use candle_core::Device;

let device = Device::new_cuda(0)?;

let mut pipeline = StableAudioPipeline::from_single_file(
    "<MODELS_DIR>/audio/stable-audio-open-1.0.safetensors",
    device,
)?;

// Generate a 15-second cinematic ambient track
let sound = pipeline.generate_sound(
    "Cinematic drum crescendo with deep sub-bass impact and rain ambience, 44.1kHz stereo",
    15.0, // Duration in seconds
    50,   // Denoising steps
)?;

sound.save_wav("output_ambience.wav")?;
```

---

### Text-to-Music (ACE-Step 1.5 — Turbo / Base / XL)

Full **48 kHz stereo song generation** from a caption **and lyrics**, in 100% pure Rust,
bit-exact-validated against HuggingFace Diffusers. Supports the **Turbo**, **Base/SFT** and
**XL (4B)** variants (auto-detected from the checkpoint).

```rust
use aurora_rust_engine::pipelines::{AudioDiffusionPipeline, TextToMusicRequest};
use aurora_rust_engine::audio::AudioFormat;

// CUDA bf16 when available, otherwise CPU f32.
let pipeline = AudioDiffusionPipeline::from_pretrained("G:/models/Audio")?; // or Audio-base / Audio-xl-base

let req = TextToMusicRequest::new(
        "acoustic pop, warm uplifting, 120 bpm, acoustic guitar, piano, drums",
        "[verse]\nShining like the morning sun",
    )
    .with_language("fr")
    .with_duration(30.0)   // seconds (10s upward; long songs use tiled VAE)
    .with_steps(8)         // 8 for Turbo, ~30-50 for Base/XL
    .with_seed(12345);     // deterministic seeded noise

let (audio, metrics) = pipeline.text_to_music(&req)?;
println!("{} Hz, {} ch, {:.2}s", metrics.sample_rate, metrics.channels, metrics.duration_seconds);

// Export: WAV (native), OGG Vorbis (BSD), MP3 (optional `--features mp3`, LAME/LGPL).
pipeline.text_to_music_to_file(&req, "my_song.ogg", Some(AudioFormat::Ogg))?;
```

* **Deterministic** : the same `seed` yields an identical waveform (xoshiro256** + Box-Muller).
* **Variants** : `pipeline.variant` is `Turbo` (guidance 1.0) or `Base`/`XL` (CFG via APG, guidance ≈ 7).
* **Long songs** : query-attention chunking + overlap-discard tiled VAE decode allow 3-minute
  tracks on a 12 GB GPU.
* **5Hz LM planner** : `AceStepLm::from_dir(...)` + `plan(caption, lyrics, max_tokens)` expands a
  caption into CoT metadata/lyrics (`bpm`, `keyscale`, `duration`, …).
* **Convert a single-file HF checkpoint** (Base/SFT/XL) to the expected layout with
  `scripts/convert_acestep_base.py` (see [`docs/ACESTEP_AUDIO_SPEC.md`](docs/ACESTEP_AUDIO_SPEC.md)).

```powershell
# CLI: text2music end-to-end
cmd /c '"...\vcvarsall.bat" x64 && cargo run --release --features cuda --bin test_audio_diffusion -- --model-dir G:/models/Audio --duration 30 --steps 8 --lyrics-file scripts/lyrics_fr.txt --lang fr --format ogg --out ma_musique'
```

---

### Music Editing & Source-Audio Tasks (Cover / Repaint / Extract / Lego / Complete)

The same ACE-Step 1.5 checkpoint drives the **source-audio** tasks through a single
`TaskRequest`, 100% pure Rust. The pipeline mirrors the reference handler's conditioning:
task instruction, `Global:/Local:` caption (SFT-stems), `src_latents`, chunk mask, and the
repaint step-injection + boundary crossfade (validated **bit-exact** against PyTorch).

```rust
use aurora_rust_engine::pipelines::{AudioDiffusionPipeline, TaskRequest};

let pipeline = AudioDiffusionPipeline::from_pretrained("G:/models/Audio-sft")?;

// Source audio -> [1, 2, N] tensor -> VAE encode -> frame-major latents [1, T, 64].
let src_latents = pipeline.vae.encode(&audio)?.transpose(1, 2)?.contiguous()?;

let req = TaskRequest::new("extract", &src_latents)   // extract | lego | complete | repaint | cover
    .with_track_name("vocals")                        // extract / lego
    .with_caption("warm acoustic pop, guitar, piano, drums, vocals")
    .with_language("unknown")
    .with_steps(50)                                   // 50 for Base/SFT, 8 for Turbo
    .with_guidance_scale(7.0)                         // CFG/APG (ignored by Turbo)
    .with_seed(42);

let (audio, _metrics) = pipeline.generate_task(&req)?;
audio.save_auto("vocals.wav")?;
```

| Task | Instruction | Source handling |
|---|---|---|
| `cover` | "Generate audio semantic tokens based on the given conditions:" | melodic/timbral seed (`cover_strength`, `cover_noise_strength`) |
| `repaint` | "Repaint the mask area based on the given conditions:" | masked region regenerated, boundaries crossfaded (`.with_repaint_span(start, end)`) |
| `extract` | "Extract the {TRACK} track from the audio:" | isolates one stem |
| `lego` | "Generate the {TRACK} track based on the audio context:" | adds a stem in context (optional repaint span) |
| `complete` | "Complete the input track with {CLASSES}:" | fills missing parts (`.with_complete_track_classes(vec![...])`) |

**Key notes**

* **Chunk mask** : `extract`/`lego`/`complete` use the `"auto"` **Mask Control value `2.0`**
  (`TaskRequest.chunk_mask_value`, default `2.0`). A value of `1.0` produces low-frequency noise.
  `repaint` uses the explicit `0/1` mask from the repaint span; the `--chunk <v>` CLI flag overrides.
* **Timbre** : set `.refer_latents = Some(&lat)` (a `reference_audio`); by default the learned
  silence latent is used (reference parity).
* **Models** : the **2B** `acestep-v15-base` / `acestep-v15-sft` are recommended. The **XL 5B**
  (~10 GB bf16) is borderline on a 12 GB GPU.
* **Sources** : use real music (≥ 30 s); very short or artefact clips can render distorted.
* **Build** : always compile with `--features cuda` (otherwise CUDA is silently unavailable and the
  pipeline falls back to CPU/f32).

```powershell
# CLI: extract a stem
cargo run --release --features cuda --bin test_acestep_tasks -- `
  -m G:/models/Audio-sft -t extract --track vocals --src my_song.wav -o vocals.ogg -s 50 -g 7
```

---

### Text-to-Speech (TTS / Parler-TTS & Kokoro-82M)

Aurora provides neural speech synthesis pipelines in 100% pure Rust:

#### 1. Parler-TTS (Controllable Natural Voice Synthesis)
* **Controllable Voice Acoustics** : Guide pitch, speaking rate, timbre, and acoustic environment through natural language voice descriptions.
* **Architecture** : T5 text conditioner + multi-codebook autoregressive transformer + Descript Audio Codec (DAC) 44.1 kHz neural vocoder.
* **Native Pure Rust Audio I/O** : Standard 16-bit PCM RIFF WAV saving via [`WavAudio`](file:///d:/image_to_text/TransRust/src/audio/wav.rs) with zero external C/FFmpeg dependencies.

```rust
use aurora_rust_engine::audio::WavAudio;
use aurora_rust_engine::pipelines::TtsPipeline;
use candle_core::{DType, Device};

let device = Device::new_cuda(0)?;

// Load Parler-TTS & DAC acoustic vocoder
let mut pipeline = TtsPipeline::from_files(
    "models/tts/parler_config.json",
    "models/tts/parler_model.safetensors",
    "models/tts/dac_model.safetensors",
    "models/tts/tokenizer.json",
    "models/tts/description_tokenizer.json",
    device,
    DType::F32,
)?;

// Synthesize high-fidelity 44.1 kHz speech
let audio: WavAudio = pipeline.synthesize(
    "Welcome to the Aurora inference engine, running entirely in pure Rust on CUDA.",
    "A female speaker delivers a clear and articulate speech with moderate pacing and natural warmth.",
    600, // max tokens
    0.8, // temperature
    42,  // seed
)?;

// Save directly to standard 16-bit PCM WAV
audio.save_wav("output_speech.wav")?;
println!("Synthesized {:.2}s of speech at {} Hz", audio.duration_seconds(), audio.sample_rate);
```

#### 2. Kokoro-82M (Ultra-Compact Non-Autoregressive Voice)
* **Ultra-Lightweight** : 82M parameters running at **> 50x realtime** on CPU and GPU.
* **Natural Voice Profiles** : Multiple built-in American & British voices with expressive prosodic inflections (`af_heart`, `am_adam`, `bf_emma`).

---

## 5. REST API & WebSocket Server Reference

Aurora embeds a high-performance [Axum](https://github.com/tokio-rs/axum) web server providing REST endpoints and streaming WebSockets:

```bash
# Launch the API server on http://127.0.0.1:8080
cargo run --release --bin server --features cuda,flash-attn
```

### Endpoints

| Method | Path | Description |
|:---:|:---:|---|
| `POST` | `/api/v1/generate` | Generate image from prompt (REST JSON) |
| `POST` | `/api/v1/img2img` | Image-to-Image generation |
| `POST` | `/api/v1/inpaint` | Mask-guided inpainting |
| `POST` | `/api/v1/lora/load` | Dynamically merge a LoRA adapter |
| `POST` | `/api/v1/lora/clear` | Clear all loaded LoRAs |
| `GET` | `/api/v1/lora/list` | List active LoRA adapters |
| `GET` | `/api/v1/models` | List available checkpoints |
| `GET` | `/api/v1/system/info`| GPU info, VRAM, and engine status |
| `WS` | `/api/v1/ws` | Real-time WebSocket streaming with latent previews |

### JSON Request Payload Schema (`POST /api/v1/generate`)

```json
{
  "prompt": "masterpiece, ultra-detailed, cyberpunk warrior, 8k",
  "negative_prompt": "lowres, bad anatomy, blurry",
  "steps": 18,
  "guidance_scale": 6.5,
  "width": 1024,
  "height": 1024,
  "seed": 42,
  "scheduler": "dpm",
  "vae_tiling": true,
  "cpu_offload": true,
  "fp8": false
}
```

### Response Schema

```json
{
  "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "telemetry": {
    "prompt_encode_ms": 1150.2,
    "unet_total_ms": 8350.4,
    "unet_it_per_sec": 2.15,
    "unet_step_avg_ms": 464.0,
    "vae_decode_ms": 1790.1,
    "total_wallclock_ms": 12140.0
  }
}
```

---

## 6. CLI Binaries & Benchmark Suite

Aurora comes with pre-built test and benchmark executables in `src/bin/`:

| Binary | Command | Description |
|---|---|---|
| **`grio_showcase`** | `cargo run --release --bin grio_showcase --features cuda,flash-attn,ui` | **Interactive Web UI Studio** at `http://127.0.0.1:7860` |
| **`grand_benchmark`** | `cargo run --release --bin grand_benchmark --features cuda,flash-attn` | **SOTA 3-Aspect Grand Benchmark** (18 steps DPM-Solver++) |
| **`server`** | `cargo run --release --bin server --features cuda,flash-attn` | High-throughput **REST / WebSocket Server** |
| **`test_dpm_solver`** | `cargo run --release --bin test_dpm_solver --features cuda,flash-attn` | Discrete 18-step DPM-Solver++ validation harness |
| **`comparative_benchmark`**| `cargo run --release --bin comparative_benchmark --features cuda,flash-attn` | FlashAttention vs SDPA comparative stress test |
| **`stress_matrix_test`** | `cargo run --release --bin stress_matrix_test --features cuda,flash-attn` | 15-image endurance matrix across 5 seeds and 3 resolutions |
| **`test_lora_merge`** | `cargo run --release --bin test_lora_merge --features cuda,flash-attn` | LoRA hot-merging & base weight restoration test |
| **`test_img2img`** | `cargo run --release --bin test_img2img --features cuda,flash-attn` | Image-to-Image pipeline verification |
| **`test_inpaint`** | `cargo run --release --bin test_inpaint --features cuda,flash-attn` | Mask-guided inpainting verification |
| **`test_controlnet`** | `cargo run --release --bin test_controlnet --features cuda,flash-attn` | Canny edge Multi-ControlNet integration test |
| **`test_text_gen`** | `cargo run --release --bin test_text_gen --features cuda "<model.gguf>" "<prompt>"` | **CausalLM Text Generation** (Llama/DeepSeek/Qwen/Gemma/Mistral) |
| **`test_tts`** | `cargo run --bin test_tts` | **Audio & Neural Text-to-Speech (TTS)** validation harness (WAV + Parler/DAC) |
| **`test_whisper`** | `cargo run --bin test_whisper` | **Speech-to-Text (STT)** Whisper transcription harness (pure Rust Slaney Mel filterbank) |
| **`test_audio_diffusion`** | `cargo run --release --features cuda --bin test_audio_diffusion -- --model-dir G:/models/Audio --duration 30 --steps 8` | **Text-to-Music** end-to-end (ACE-Step 1.5 Turbo/Base/XL) |
| **`test_acestep_tasks`** | `cargo run --release --features cuda --bin test_acestep_tasks -- -m G:/models/Audio-sft -t extract --track vocals --src song.wav -o out.ogg -s 50 -g 7` | **Source-audio tasks** (cover / repaint / extract / lego / complete) |
| **`test_acestep_cover`** | `cargo run --release --features cuda --bin test_acestep_cover -- --model-dir G:/models/Audio --src song.wav` | Cover task harness (source → VAE → DiT) |
| **`test_acestep_repaint`** | `cargo run --release --features cuda --bin test_acestep_repaint -- -m G:/models/Audio-base --src song.wav` | Repaint harness (mask + step injection + boundary blend) |


---

## 7. Hardware & Performance Tuning Guide

### Recommended Settings per GPU VRAM Tier

| GPU VRAM | Recommended Settings | Average Speed ($1024^2$) | Peak VRAM |
|---|---|:---:|:---:|
| **8 GB** (RTX 3070, 4060) | `cpu_offload: true`, `vae_tiling: true` (72x72, overlap 16), `scheduler: "dpm"`, 18 steps | ~1.6 - 1.8 it/s | ~6.5 GB |
| **12 GB** (RTX 4070, 4070 Ti) | `cpu_offload: true`, `vae_tiling: true`, `scheduler: "dpm"`, 18 steps | **2.05 - 2.21 it/s** | **~6.8 GB** |
| **16 GB - 24 GB+** (RTX 4080, 4090) | `cpu_offload: false`, `vae_tiling: false`, `scheduler: "dpm"`, 18 steps | **3.0 - 4.5 it/s** | ~11.5 GB |

### Preventing Windows WDDM Shared RAM Paging
When dedicated GPU VRAM exceeds ~92% capacity under Windows 11, Windows WDDM automatically pages allocations into system RAM over PCIe, causing generation time to degrade from ~12s up to 60s+. 

To guarantee **zero pagination**:
1. Keep `vae_tiling: true` (Default in Aurora).
2. Keep `cpu_offload: true` on 8GB and 12GB GPUs.
3. Close VRAM-heavy applications (video editing, 3D games) during high-throughput batches.

### Flux MMDiT Performance (FlashAttention-2 manette)

For Flux.1/Flux.2 MMDiT pipelines the [Attention Backend Manette](#attention-backend-manette-flashattention-2)
is the highest-value lever — it multiplies denoising throughput without any VRAM penalty:

```rust
// Recommended for all Flux pipelines on CUDA GPUs (falls back safely to F32 if unavailable)
flux_pipeline.enable_flash_attn();
```

| Metric on Flux.2-Klein-4B (RTX 4070 Ti) | F32 SDPA | FlashAttention-2 |
|---|---|---|
| Denoising step (4608 tokens) | 4.87 s | **2.47 s** |
| Total 4-step render | ~21 s | **~11.7 s** |
| Peak VRAM | ~6.8 GB | ~6.8 GB (unchanged) |

---

## 📄 License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.
