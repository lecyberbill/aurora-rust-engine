# 📖 Aurora Rust Engine — User & Developer Guide

> **SOTA Pure Rust Generative AI Inference Engine for SDXL, Pony XL & Diffusion Transformers**  
> Powered by [Candle](https://github.com/huggingface/candle), [FlashAttention-2](https://github.com/Dao-AILab/flash-attention), and [Grio UI](https://github.com/lecyberbill/grio).

---

## 📑 Table of Contents

1. [Architecture & Key Highlights](#1-architecture--key-highlights)
2. [Installation & Requirements](#2-installation--requirements)
3. [Running the Interactive Web UI (Grio)](#3-running-the-interactive-web-ui-grio)
4. [Using Aurora in Rust Applications (SDK Reference)](#4-using-aurora-in-rust-applications-sdk-reference)
   - [Loading Models (Local & HuggingFace Hub)](#loading-models-local--huggingface-hub)
   - [Configuring Schedulers (DPM-Solver++, Euler, Flow-Matching)](#configuring-schedulers-dpm-solver-euler-ddim)
   - [Memory & VRAM Management Modes](#memory--vram-management-modes)
   - [Attention Backend Manette (FlashAttention-2)](#attention-backend-manette-flashattention-2)
   - [Text-to-Image Generation (SDXL & FLUX.1/FLUX.2)](#text-to-image-generation-flux1--flux2-mmdit-family)
   - [FLUX.2 Image-to-Image (Img2Img)](#flux2-image-to-image-img2img-transformation)
   - [FLUX.2 Inpainting & Masked Diffusion](#flux2-inpainting--masked-diffusion)
   - [SDXL Image-to-Image (Img2Img)](#image-to-image-img2img)
   - [SDXL Inpainting & Mask-Guided Diffusion](#inpainting--mask-guided-diffusion)
   - [Hot LoRA Merging](#hot-lora-merging)
   - [ControlNet (Canny Edge)](#controlnet-canny-edge)
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

---

## 3. Running the Interactive Web UI (Grio)

Aurora includes a native web studio powered by [Grio](https://github.com/lecyberbill/grio):

```bash
cargo run --release --bin grio_showcase --features cuda,flash-attn,ui
```

Once loaded, navigate in your browser to:
👉 **`http://127.0.0.1:7860`**

### Features Available in the Web Studio:
- **Prompt & Negative Prompt Fields**: Full multi-line input with pre-configured high-quality negative prompt defaults.
- **Scheduler Switcher**: Select between **DPM-Solver++ 2M Karras (18 steps)**, **Euler Discrete Karras (30 steps)**, and **DDIM**.
- **Dimension Selector**: $1024\times 1024$ (Square 1:1), $832\times 1216$ (Portrait 2:3), $1216\times 832$ (Landscape 3:2).
- **Interactive VRAM Controls**: Toggles for Seamless Tiled VAE, Dual-CLIP CPU Offloading, and FP8 precision.
- **Progressive Real-Time Latent Previews**: Watch the image materialize live during the denoising process.
- **Session History Gallery**: View all generated images side-by-side.
- **Observability Cards**: Live telemetry of UNet speed (`it/s`), wall-clock latency (`s`), and peak VRAM.

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

Configure memory behavior on the fly to suit any hardware from 6GB to 24GB+ VRAM:

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

> **Note** — Replace `<MODELS_DIR>` with your local models directory. The Rust library itself contains
> no hardcoded paths; models are supplied at call time. The Dev pipeline currently renders a
> recognisable fox with a residual "stained-glass" grain; Klein-4B/9B are fully photorealistic.
> Dev polish is tracked in the ROADMAP.

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
