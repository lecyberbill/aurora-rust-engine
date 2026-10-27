# Aurora Rust Engine (`aurora-rust-engine`)

> Pure-Rust inference **and LoRA fine-tuning** for text, image, speech and
> music — no Python runtime. Powered by [Candle](https://github.com/huggingface/candle).

- **Backends**: NVIDIA CUDA (FlashAttention-2) and AMD ROCm/HIP
  (tested on `gfx1201` / Radeon AI PRO R9700), plus Metal and CPU.
- **Formats**: GGUF (Q4_K_M/Q8_0…), SafeTensors (single file or sharded).
- **One binary**: inference, streaming, sampling, in-memory LoRA merging,
  and a dense LoRA SFT trainer.

## Capabilities

| Modality | Highlights |
|---|---|
| **Text** | Qwen 2.5/3/3.8, Gemma 3/4, Phi-4, Granite 4.2, Llama 3, Mistral, DeepSeek (GGUF, KV cache, streaming) |
| **Text LoRA** | Dense Qwen2 (0.5B/3B/7B) and Gemma 3 (4B/12B) SFT: PEFT export, reload/merge, val split |
| **Image / Video** | SDXL/Pony, FLUX.1, FLUX.2 (Klein 4B/9B, Dev), Z-Image Turbo, CogVideoX-2B; img2img, inpaint, ControlNet, LoRA merge |
| **Upscaling / VLM** | Real-ESRGAN, 4x-UltraSharp, Gemma 4 31B, Qwen2/2.5/3-VL, PaliGemma 2, SmolVLM |
| **Speech / Audio** | Whisper, Moonshine; Parler-TTS, Kokoro, ChatTTS; ACE-Step 1.5, Stable Audio Open, MusicGen |

Not everything works (e.g. gpt-oss MXFP4, LFM 2.5 and Nemotron are not
supported). Status, validation and reasons: **[SUPPORTED.md](SUPPORTED.md)**.

## 2026-10 highlights

**ROCm decode** (R9700, 256 tokens, same GGUF/GPU):

- **Split-K flash-decoding** HIP kernel: **+18.6 %** on Gemma 3 12B (N=5,
  same-session pairs), 4.8×/layer.
- **q8_1/dst memsets removed** (fork `0003`): **+1.9 %**.
- **Fused GDN step** (Qwen 3.8 hybrid): **5.13×** on the recurrence
  (deterministic micro-bench).
- Portable: Qwen 3.8 27B **16.8 tok/s**, Gemma 3 12B **41.4 tok/s**
  (llama.cpp reference: 29.4 / 62.7). Protocol and raw runs:
  **[docs/BENCH_ROCM.md](docs/BENCH_ROCM.md)**.

**LoRA SFT** (dense, validated on ROCm and CUDA):

- Qwen2 0.5B/3B/7B and Gemma 3 4B/12B, ChatML/Gemma prompts,
  assistant-masked CE, AdamW, `--val N`, PEFT adapter export.
- `train_lora` → `gen_lora` (reload) / `merge_lora` (standalone F16 model).
- Sharded checkpoints and `--base-dtype f16` for large models.
- Fork patch `0004`: no gradients for frozen weights (otherwise 3B OOM).
- Details and numbers: **[docs/LORA.md](docs/LORA.md)** and
  **[docs/SESSION_2026-10.md](docs/SESSION_2026-10.md)**.

## Quick start

```bash
# NVIDIA
cargo build --release --features cuda,flash-attn

# AMD ROCm (Linux)
cargo build --release --features rocm

# Text generation (GGUF)
./target/release/test_text_gen /models/llm/gemma-3-12b-it-q4_k_m.gguf "Hello" 128 0.7

# LoRA SFT (dense checkpoint)
cargo build --release --features rocm --bin train_lora
./target/release/train_lora \
  --model-dir /models/llm/qwen2.5-0.5b-instruct \
  --data data/lora_tiny.jsonl --out /tmp/lora_out \
  --steps 60 --lr 3e-4 --max-seq 128 --val 1 --device rocm
```

AMD setup (ROCm 10, `gfx1201`, fork build): **[docs/ROCM_SETUP.md](docs/ROCM_SETUP.md)**.

## Documentation

- **[USER_GUIDE.md](USER_GUIDE.md)** — SDK, pipelines, REST/UI, VRAM modes.
- **[ATLAS.md](ATLAS.md)** — component/repository map.
- **[ROADMAP.md](ROADMAP.md)** — milestones and next steps.
- **[SUPPORTED.md](SUPPORTED.md)** — model status, parity, limitations.
- **[docs/SESSION_2026-10.md](docs/SESSION_2026-10.md)** — this session's
  user guide (ROCm perf + LoRA, commands, numbers).
- **[docs/PERF_ROCM.md](docs/PERF_ROCM.md)** — levers, rejected experiments,
  roofline; **[docs/BENCH_ROCM.md](docs/BENCH_ROCM.md)** — benchmark protocol.
- **[docs/LORA.md](docs/LORA.md)** — LoRA training/reload/merge.
- **[docs/ROCM_SETUP.md](docs/ROCM_SETUP.md)** — AMD toolchain.
- **[docs/patches/](docs/patches/)** — Candle fork patch series (`0001`–`0004`).

## Notes

- Candle fork patches are needed for the autotune, the memset removal and large
  frozen-base LoRA. The portable build (`--release --features rocm`) runs
  without them.
- ROCm lacks backward for the fused `rms_norm`/`softmax`; the trainer uses
  differentiable compositions (same results, slightly slower).
- Same-session paired runs only: the R9700 clock state drifts between loads.

## License

Apache-2.0 / MIT (see `LICENSE`).
