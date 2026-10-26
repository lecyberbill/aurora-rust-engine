# Supported Models & Validation Status

This page is the single source of truth for **what runs, how well it is validated, and what was dropped**.
The README stays short; details live here.

### Status legend

| Icon | Meaning |
|---|---|
| ✅ | Validated end-to-end. When a parity metric exists, the max absolute error vs the reference implementation is listed. |
| ⚠️ | Runs, but with a documented approximation, hardware limitation, or unverified edge path. |
| 🧪 | Implemented, not independently re-validated. |
| ❌ | Not supported / abandoned — reason given. |

> **About "bit-exact":** in this repo it means *"max absolute error below the tolerance listed in the Evidence column"* for tensor-level checks — not literal bit identity. LLM rows are validated on output quality unless a parity metric is listed explicitly.

**Last validation date:** 2026-10-26 (ROCm 10.0, `gfx1201`). CUDA rows predate this and are marked as such.

---

## Text / CausalLM

| Model family | Formats | Status | Evidence / note |
|---|---|---|---|
| Qwen 2.5 / Qwen 3 (dense) | GGUF Q4_K_M/Q8_0, SafeTensors | ✅ | Coherent multi-turn output; KV-cache + streaming callback. |
| Qwen 3.5 / 3.8 hybrid (48 GDN + 16 attention layers) | GGUF | ⚠️ | Decode is coherent and usable, but the recurrent layer is a **simplified EMA recurrence**, not the exact Gated DeltaNet. No tensor-level parity is claimed. |
| Gemma 2 / Gemma 3 | GGUF | ✅ | Dedicated quantized backend; RMSNorm/softmax/RoPE fused kernels. |
| Gemma 4 31B (text) | GGUF UD-Q4_K_XL | ✅ | Coherent text output; dedicated backend. |
| Microsoft Phi-4 (14B, reasoning) | GGUF Q4_K_M | ✅ | Native GGUF chat template + repetition penalty. |
| IBM Granite 4.2 30B | GGUF Q6_K | ✅ | Dense path with explicit `attention.scale`, CoT chat template. |
| Llama 3 / Mistral (dense) | GGUF, SafeTensors | ✅ | Generic CausalLM path. |
| DeepSeek V3 / R1 (MoE) | GGUF | 🧪 | Shared + routed expert code present; no independent parity run this session. |
| Ministral 3 14B | GGUF Q8_0 | 🧪 | Checkpoint present; not benchmarked/validated. |
| LFM 2.5 8B (ShortConv-MoE) | GGUF Q4_K_M | ❌ | Implementation runs but output is degenerate; abandoned as too architecture-specific. |
| NVIDIA Nemotron 3 Nano 30B-A3B | GGUF | ❌ | Routed experts are stored as **Q5_0**, which the ROCm `indexed_moe_forward` kernel does not implement (supported: q2k/q3k/q4k/q5k/q6k/q8_0). Hybrid arch not pursued. |
| gpt-oss-20b | GGUF | ❌ | Two blockers: Candle lacks the **MXFP4** dtype (GGML type 39) and the `gpt_oss` architecture is not implemented. Validated externally with ollama (~158 tok/s). |

---

## Image / Video

| Model family | Status | Evidence / note |
|---|---|---|
| SDXL / Pony XL (single-file checkpoints) | ✅ | LoRA hot-merge, img2img, inpaint, ControlNet, IP-Adapter. |
| FLUX.1 dev / schnell | ✅ | MMDiT, sequential block streaming (< 7.5 GB VRAM). |
| FLUX.2 Klein-4B / Klein-9B / Dev | ✅ | 16/32-ch VAE, img2img, inpaint, multi-image reference (4D RoPE). |
| Z-Image Turbo (S3-DiT 6B) | ✅ | 4-step real-time DiT, Qwen3-4B conditioning. |
| CogVideoX-2B (3D DiT) | 🧪 | Implemented; no independent parity run this session. |
| Real-ESRGAN x2/x4/x8, 4x-UltraSharp, SRVGGNetCompact | ✅ | Bounded tiling + seam blending; 512²→2048² in 5.5 s (CUDA). |

---

## Vision-Language (VLM)

| Model family | Status | Evidence / note |
|---|---|---|
| Gemma 4 31B Multimodal | ✅ | Dense OCR + VQA benchmark (`Le-Complexe.jpg`), see benchmark table. |
| Qwen2-VL / Qwen2.5-VL / Qwen3-VL | 🧪 | Implemented; not all checkpoints re-validated. |
| PaliGemma 2 | 🧪 | Implemented; not re-validated. |
| Gemma 3 Vision | 🧪 | Implemented; not re-validated. |
| SmolVLM | 🧪 | Implemented; not re-validated. |

---

## Audio / Speech

| Model family | Status | Evidence (max abs error vs reference) |
|---|---|---|
| Whisper Large-v3 / Turbo | ✅ | Parity vs PyTorch reported; no numeric tolerance published. |
| Moonshine (Tiny/Base) | ✅ | Encoder **6.9e-6**, token ids identical. |
| ACE-Step 1.5 Turbo / Base / SFT / XL | ✅ | Repaint **7.9e-6**; cover/repaint/extract/lego/complete + FSQ codec. |
| Stable Audio Open 1.0 | ✅ | T5+Fourier projection **2e-7**, Oobleck VAE **6.7e-6**, DiT **3.6e-4**, EDM/SDE scheduler **1e-6**. |
| MusicGen-small | ✅ | EnCodec **1.9e-6**, AR decoder **5e-4**. |
| Parler-TTS | ✅ | Functional; no numeric tolerance published. |
| Kokoro-82M | ✅ | Functional; no numeric tolerance published. |
| ChatTTS + Vocos | 🧪 | Implemented (separate contribution); not re-validated this session. |

---

## Not supported (abandoned)

| Model | Reason |
|---|---|
| **gpt-oss-20b** | Candle has no **MXFP4** dtype (GGML type 39) and the `gpt_oss` architecture is not implemented. Reference throughput via ollama: ~158 tok/s. |
| **LFM 2.5 8B** (ShortConv-MoE) | Runs, but the output is degenerate; abandoned as too architecture-specific. |
| **NVIDIA Nemotron 3 Nano 30B-A3B** | Routed experts are stored as **Q5_0**, which the ROCm indexed-MoE kernel does not implement; hybrid architecture not pursued. |

---

## Benchmarks — single-stream decode (ROCm 10.0, Radeon AI PRO R9700 `gfx1201`)

Protocol: batch size 1, single request, **load time excluded**, `AURORA_IGNORE_EOS=1` where applicable to avoid early-EOS skew. Numbers are from the default build unless noted. Ranges reflect run-to-run variance (EOS length); the HIP-graph gain is measured as a paired comparison on the same prompt.

| Model | File | Size | Load | Decode | Notes |
|---|---|---:|---:|---:|---|
| Qwen 3.8 27B | `Qwen3.8-27B-Q4_K_M.gguf` | 17.44 GB | 71 s | **12.9–16.1 tok/s** (200 tok) | Hybrid GDN — approximate recurrence |
| Qwen 3.8 27B + HIP graph | same | 17.44 GB | 71 s | **14.0–18.2 tok/s** (≈+15%) | `--features hip-graph` (opt-in) |
| Gemma 3 12B | `gemma-3-12b-it-q4_k_m.gguf` | 7.30 GB | ~4 s | **30–38 tok/s** (200 tok) | Default backend |
| Gemma 3 12B + HIP graph | same | 7.30 GB | ~4 s | **~41 tok/s** | `--features hip-graph` (opt-in) |
| Gemma 4 31B (text) | `gemma-4-31B-it-UD-Q4_K_XL.gguf` | 18.82 GB | 7.9 s | 11.3 tok/s | Earlier run |
| Gemma 4 31B (VLM) | + `mmproj-gemma-4-31b-BF16.gguf` | 18.82 + 1.2 GB | 6.3 s + 0.9 s | 11.5 tok/s | OCR/VQA, 266 visual tokens |
| Phi-4 Reasoning Plus 14B | `Phi-4-reasoning-plus-Q4_K_M.gguf` | 8.50 GB | lazy | **22.9 tok/s** (256 tok) | Native chat template |
| gpt-oss-20b (reference) | — | — | 2.0 s | ~158 tok/s | **not in-engine** — ollama/llama.cpp reference |
| LFM 2.5 8B | `LFM2.5-8B-A1B-Q4_K_M.gguf` | 4.80 GB | lazy | — | ❌ degenerate output |

CUDA reference (RTX 4070 Ti 12 GB, from earlier runs): TinyLlama-1.1B **5.4 tok/s**, Moonshine Tiny **1.6 s**, Real-ESRGAN x4 **5.5 s**, FLUX.2-Klein-4B **25.9 s**.

---

## Validation methodology

- **Tensor parity:** reference tensors are dumped with the Python scripts in `scripts/` (`dump_*_ref.py`) and compared module-by-module against the Rust implementation; the max absolute error is recorded in the Evidence column.
- **Text/LLM:** validated on output quality (coherence, factual answers such as "Paris", multi-turn stability) unless a parity metric is listed. No logit-level parity is claimed for LLMs.
- **Image/audio demos:** generated artifacts are committed under the repo's demo paths and regenerated by the `test_*` binaries.

## Known limitations

- **MXFP4 / NVFP4** and the `gpt_oss` architecture are not supported.
- **ROCm indexed-MoE** supports q2k/q3k/q4k/q5k/q6k/q8_0 only; Q5_0/Q4_0 expert stacks fall back to the dense path (or are rejected for memory reasons).
- **Qwen 3.5/3.8 GDN** is an approximation (EMA), not the exact DeltaNet recurrence.
- **WMMA INT4** for `gfx1201` exists as the standalone `test_wmma_q4k` binary but is not wired into the model path.
- The **HIP-graph** decode runner needs the local Candle fork patch (`copy2`/`copy_from_host_u32`) and the `hip-graph` feature; the default ROCm build does not require it. Enable with `cargo build --features rocm,hip-graph` and run with `AURORA_HIP_GRAPH=1`. It covers Gemma 3 and Qwen 3.5/3.8 (attention KV + GDN conv/recurrent state preallocated and updated in place).
