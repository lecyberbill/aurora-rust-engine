# 🎨 Aurora Studio — User Guide

Aurora Studio is the **pure-Rust web interface** of `aurora-rust-engine`, built on
[Grio](https://github.com/lecyberbill/grio). It runs text-to-image diffusion models entirely in Rust
(Candle + CUDA) with **no Python, no PyTorch**. This guide is for **end users**: how to install,
launch, configure your models, and tune VRAM — no Rust knowledge required beyond running two commands.

---

## 📑 Table of Contents

1. [Requirements](#1-requirements)
2. [Build & Launch](#2-build--launch)
3. [Using the Interface](#3-using-the-interface)
4. [Configuring Models (`aurora_studio.json`)](#4-configuring-models-aurora_studiojson)
5. [Per-Model Defaults & Memory Knobs](#5-per-model-defaults--memory-knobs)
6. [Supported Model Families](#6-supported-model-families)
7. [VRAM Tuning & Troubleshooting](#7-vram-tuning--troubleshooting)
8. [Environment Variables](#8-environment-variables)
9. [FAQ](#9-faq)

---

## 1. Requirements

- **Windows 10/11, Linux**
- **Rust toolchain** 1.80+ ([rustup.rs](https://rustup.rs))
- An **NVIDIA GPU** with the **CUDA Toolkit 12.x** installed (a 12 GB RTX 4070 Ti is the reference
  target; smaller cards work with lower resolution / tiling turned down in step 5)
- Your model files as **`.safetensors`** on disk (single-file checkpoints work best)

> No GPU at all? The engine falls back to CPU, but generation will be very slow. Use CUDA.

---

## 2. Build & Launch

From the repository root (`aurora-rust-engine`):

```bash
# Build + run the studio with CUDA and FlashAttention-2.
# The default feature set is empty, so `--features cuda,flash-attn,ui` is REQUIRED.
cargo run --release --bin aurora_studio --features cuda,flash-attn,ui
```

Then open your browser at:

👉 **`http://127.0.0.1:7860`**

On startup the console prints the models it found and immediately **pre-loads the first model** in
your config (so the first generation is fast). Example:

```
📄 Chargement des modèles : aurora_studio.json
✅ Aurora Studio : 'SDXL (Juggernaut XL)' prêt.
   • sdxl — SDXL (Juggernaut XL)
   • flux1 — Flux.1 Dev (embarqué VLM)
   • sd35 — SD 3.5 Large (CLIP-L+G+T5)
🌐 Aurora Studio live: http://127.0.0.1:7860
```

> ⚠️ Always keep `--features cuda,flash-attn,ui`. Without `flash-attn` the model uses much more VRAM
> and is several times slower; without `cuda` it runs on the CPU.

---

## 3. Using the Interface

| Control | What it does |
|---|---|
| **Modèle** (dropdown) | Picks the model. Switching applies that model's saved `defaults` (see §5). Only **one model is resident in VRAM at a time** ("strict ejection"): the previous model is unloaded on switch. |
| **Résolution** | `512×512`, `768×768`, or `1024×1024`. Larger = more VRAM and slower. |
| **Prompt** | What you want to see. English works best for most checkpoints. |
| **Negative prompt** | What you *don't* want. Used for real classifier-free guidance (CFG) when `guidance > 1.0`. |
| **Steps** | Denoising iterations. Turbo/distilled models need few (1–8); regular models 20–30. |
| **Guidance** | CFG strength. `1.0` = single pass (distilled). Regular SDXL ≈ `5–7`; SD 3.5 ≈ `3.5–5`. |
| **Seed** | Change for a different image with the same settings; reuse for a reproducible one. |
| **🎨 Générer** | Runs the generation. The status area shows `⏳ Chargement…` / `⏳ Génération…` and the button is locked while it works. |
| **Rendu live** | Live latent preview while denoising (updates during the run). |
| **Galerie de la session** | All images generated this session. |

Generation parameters are echoed in the **Statut** area after each run
(`✅ 28 steps · 1024×1024 · seed 42`).

---

## 4. Configuring Models (`aurora_studio.json`)

The studio is **fully config-driven** — no model path is hard-coded. It reads a JSON file named
**`aurora_studio.json`** in the current working directory. Point it elsewhere with the
`STUDIO_CONFIG` environment variable:

```bash
# Windows PowerShell
$env:STUDIO_CONFIG = "D:\models\my_models.json"
cargo run --release --bin aurora_studio --features cuda,flash-attn,ui
```

### 4.1 Minimal entry

```json
{
  "models": [
    {
      "id": "sdxl",
      "label": "SDXL (Juggernaut XL)",
      "family": "sdxl",
      "checkpoint": "G:/models/checkpoints/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"
    }
  ]
}
```

### 4.2 Field reference

| Field | Required | Description |
|---|---|---|
| `id` | ✅ | Stable internal identifier (used to load the model). |
| `label` | — | Name shown in the dropdown. Falls back to `id`. |
| `family` | — | Architecture **slug** (see §6). Omit it and the engine **sniffs the checkpoint**. An unknown slug is a hard error. |
| `checkpoint` | ✅ | Path to the diffusion model `.safetensors`. |
| `text_encoder` | — | External text encoder, for models that don't embed one (Flux.2-Klein, SD 3.5). See §6. |
| `vae` | — | External VAE, for models that don't embed one. |
| `defaults` | — | Saved generation defaults (steps, guidance, size, negative prompt). See §5. |
| `memory` | — | VRAM knobs. See §5. |

### 4.3 Full example (4 models)

```json
{
  "models": [
    {
      "id": "sdxl",
      "label": "SDXL (Juggernaut XL)",
      "family": "sdxl",
      "checkpoint": "G:/models/checkpoints/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
      "defaults": { "steps": 25, "guidance": 7.0, "width": 1024, "height": 1024 },
      "memory": { "vae_tile_size": 32, "vae_tile_overlap": 8 }
    },
    {
      "id": "flux1",
      "label": "Flux.1 Dev (embarqué VLM)",
      "family": "flux1",
      "checkpoint": "G:/models/flux/flux1-dev-fp8.safetensors",
      "defaults": { "steps": 20, "guidance": 3.5, "width": 1024, "height": 1024 }
    },
    {
      "id": "klein4b",
      "label": "Flux.2 Klein-4B (Qwen3)",
      "family": "flux2-klein-4b",
      "checkpoint": "G:/models/flux/fluxKlein4BPro_v10.safetensors",
      "text_encoder": { "kind": "qwen3", "path": "G:/models/clip/qwen_3_4b.safetensors" },
      "vae": "G:/models/vae/flux2-vae.safetensors",
      "defaults": { "steps": 8, "guidance": 1.0, "width": 1024, "height": 1024 }
    },
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

The **first entry is pre-loaded** at startup; the order can be whatever you like.

---

## 5. Per-Model Defaults & Memory Knobs

Both blocks are **optional** and per model. Any omitted field keeps the engine default.

### `defaults` — applied on model switch

When you pick a model in the dropdown, its `defaults` are written into the controls automatically:

| Key | Type | Meaning |
|---|---|---|
| `steps` | integer | Denoising steps. |
| `guidance` | number | CFG scale. `1.0` = single pass (distilled/Turbo). |
| `width`, `height` | integer | Resolution (use one of `512`, `768`, `1024` to match the dropdown). |
| `negative_prompt` | string | Pre-filled negative prompt. |

### `memory` — VRAM tuning (SDXL today)

| Key | Default | Meaning |
|---|---|---|
| `vae_tiling` | `true` | Decode the VAE in tiles (keeps VRAM low). Leave on unless you have 24 GB+. |
| `vae_tile_size` | `32` | **Most important knob.** Latent tile size; smaller = lower peak VRAM, more tiles. |
| `vae_tile_overlap` | `8` | Seamless blend overlap between tiles. |
| `cpu_offload` | `true` | Run the text encoders (CLIP) on CPU → saves ~1.5 GB VRAM. |
| `low_vram_load` | `true` | **Reserved** (declared, not yet consumed by the pipeline). |
| `fp8_weights` | `false` | **Reserved** (declared, not yet consumed by the pipeline). |

> 💡 A large `vae_tile_size` inflates the decoded-tile's `im2col` buffer. On a 12 GB card a big tile
> can push the driver into **Windows shared-memory paging**, making a generation take minutes instead
> of seconds. `32 × 32` with overlap `8` is the safe default.

---

## 6. Supported Model Families

Use the `family` slug (optional — omit to auto-detect from the checkpoint). An **unknown slug is an
error** (no silent guessing).

| `family` slug(s) | Family | Needs `text_encoder`? | Needs `vae`? |
|---|---|---|---|
| `sdxl`, `stable-diffusion-xl` | SDXL | No (embedded CLI-L + CLIP-G) | No (embedded) |
| `sd15`, `stable-diffusion-v1` | SD 1.5 | No | No |
| `sd3`, `sd35`, `sd3.5`, `stable-diffusion-3`, `stable-diffusion-3.5` | SD 3.5 | **Yes** — `kind: "sd35"` (CLIP-L + CLIP-G + T5) | Yes (16-ch SD3 VAE) |
| `flux1`, `flux` | Flux.1 (Dev) | No (embeds T5-XXL) | No (embedded) |
| `flux2`, `flux2-dev` | Flux.2 Dev | Yes — `kind: "mistral3"` (`dir`) | Yes (32-ch Flux VAE) |
| `flux2-klein-4b` | Flux.2 Klein-4B | Yes — `kind: "qwen3"` (`path`) | Yes (32-ch Flux VAE) |
| `flux2-klein-9b` | Flux.2 Klein-9B | Yes — `kind: "qwen3"` (`path`, sharded dir) | Yes |

`text_encoder.kind` values: `qwen3` (`path`), `mistral3` (`dir`), `t5` (`path`),
`sd35` (`clip_l` + `clip_g` + `t5`).

### Recommended generation settings

| Family | Steps | Guidance | Negative prompt |
|---|---|---|---|
| SDXL | 20–30 | 5–7 | Yes |
| SD 1.5 | 20–30 | 7–8 | Yes |
| SD 3.5 (non-turbo) | 28 | 3.5 | Optional |
| SD 3.5 Turbo | 1–4 | 1.0 | Ignored (single pass) |
| Flux.1 Dev | 20–28 | 3.5 | No |
| Flux.2 Klein | 4–8 | 1.0 | No |
| Flux.2 Dev | 20–28 | ~4 | No |

---

## 7. VRAM Tuning & Troubleshooting

### Symptoms of running out of VRAM

- Generation that suddenly slows to **minutes per step** (Windows shows **Shared GPU memory** in use)
- The process appears to hang at a VAE tile / loading step
- Console shows a CUDA **out-of-memory** error

This is usually **Windows WDDM paging**: the GPU runs out of dedicated VRAM and pages to system RAM.
The fix is to lower the **peak** below your card's limit.

### Checklist

1. **Close other GPU apps** (browsers with many tabs, games, Creative Suite, overlays). They can
   easily hold 1–3 GB of your VRAM before the model even loads.
2. **Lower `vae_tile_size`** in the model's `memory` block (e.g. `24`, then `16`).
3. **Lower the resolution** to `768×768` or `512×512`.
4. **Keep `cpu_offload: true`** (text encoders on CPU).
5. **Reduce steps** — fewer steps = less wall-clock time, though not lower peak VRAM.

| VRAM tier | Suggested settings |
|---|---|
| **8 GB** | 512–768 px, `vae_tile_size: 24`, `cpu_offload: true` |
| **12 GB** (RTX 4070/4070 Ti) | 1024 px, `vae_tile_size: 32`, `cpu_offload: true` |
| **16–24 GB+** | 1024 px, `vae_tile_size: 48–64` (or `vae_tiling: false` on 24 GB+) |

### Reference numbers (RTX 4070 Ti 12 GB, SDXL 1024×1024, 18 steps)

| | Slow (paging) | Tuned |
|---|---|---|
| Total | ~38 s | **~12.7 s** |
| UNet | 0.92 it/s | **2.09 it/s** |
| VAE decode | 15.3 s | **2.1 s** |
| Peak VRAM | ~11.8 GB | **~8.5 GB** |

---

## 8. Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `STUDIO_CONFIG` | `aurora_studio.json` | Path to the models config file. |

---

## 9. FAQ

**The dropdown is empty / the app exits with a config error.**
Your `aurora_studio.json` is missing or malformed, or a `family` slug is unknown. Check the console
message — the error names the culprit. Valid slugs are listed in §6.

**"Unknown architecture" / a family error on startup.**
You used a `family` slug that isn't recognised. Remove the `family` field to let the engine
sniff the checkpoint, or use one of the slugs in §6.

**The first generation is slow.**
The first model load reads several GB from disk. The model is pre-loaded at startup, so subsequent
generations are fast. Use an SSD if possible.

**I switched models and it re-loads.**
That's intended: only one model is resident in VRAM at a time (strict ejection). The first generation
after a switch includes the load time.

**The image looks blurry / wrong at high resolution.**
Some SDXL checkpoints need a specific scheduler/steps; try the recommended values in §6. Also check
that `width`/`height` are multiples of 64.

**Can I use a LoRA?**
The engine supports LoRA hot-merging, but the stock Studio UI doesn't expose it yet. It is available
through the Rust API and the REST server (see `USER_GUIDE.md`).

---

For the **developer/SDK** documentation (LoRA, img2img, inpainting, REST API, benchmarks), see
[`USER_GUIDE.md`](../USER_GUIDE.md). For the **model-config spec**, see
[`docs/AUTO_MODEL_SPEC.md`](AUTO_MODEL_SPEC.md) §10.1.
