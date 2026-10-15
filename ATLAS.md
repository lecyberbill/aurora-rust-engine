# ATLAS Sémantique & Carte Technique (`aurora-rust-engine`)
<!-- Last Updated: 2026-10-14 -->

## 1. Topologie Globale & Rôle Architectural

`aurora-rust-engine` est une bibliothèque d'inférence universelle en **Rust pur** (zéro runtime Python, zéro couplage réseau ou serveur gRPC/HTTP obligatoire dans le crate principal).

```
                      ┌──────────────────────────────────────────┐
                      │    AutoModel Facade & Traits Communs     │
                      │       (src/models/auto.rs, common.rs)     │
                      └─────┬───────────────────┬──────────────┬─┘
                            │                   │              │
             ┌──────────────┴────────┐          │      ┌───────┴──────────────┐
             │ Vision & Diffusion    │          │      │ Audio, Speech & VLM  │
             │ (Dit, SDXL, Flux,     │          │      │ (Whisper, ACE-Step,  │
             │  Z-Image, RealESRGAN) │          │      │  Moonshine, Qwen-VL) │
             └───────────────────────┘          │      └──────────────────────┘
                                    ┌───────────┴──────────┐
                                    │ Pure Rust CausalLM   │
                                    │ Qwen/Gemma/DeepSeek  │
                                    │ GGUF & Safetensors   │
                                    └───────────┬──────────┘
                                                │
                      ┌─────────────────────────┴────────────────┐
                      │ Universal Weights & Zero-Copy Substrates │
                      │ GGUF v2/v3 · SafeTensors · ONNX Initializer │
                      │ (src/weights/loader.rs, slice.rs, cache.rs)│
                      └─────────────────────────┬────────────────┘
                                                │
                      ┌─────────────────────────┴────────────────┐
                      │ Hardware Execution Engine                │
                      │ CUDA (FA-2) · ROCm HIP · Metal · CPU     │
                      │ (src/device.rs)                          │
                      └──────────────────────────────────────────┘
```

---

## 2. Table de Correspondance : Composants & Répertoires

| Domaine / Rôle | Emplacement Réel | Invariants & Fonction Clé |
|---|---|---|
| **Universal Weights Loader** | [`src/weights/loader.rs`](file:///d:/image_to_text/TransRust/src/weights/loader.rs) | Détection automatique de conteneur : GGUF, SafeTensors, ONNX. Mappage mmap zero-copy. |
| **Tensor Slicing & Alignment** | [`src/weights/slice.rs`](file:///d:/image_to_text/TransRust/src/weights/slice.rs) | Conversion zero-copy garantie avec `Cow<'_, [T]>` pour les offsets mémoires arbitraires ou non-alignés. Fast-path BF16 $\leftrightarrow$ F16 direct sans allocations F32. |
| **LRU Weights Cache** | [`src/weights/cache.rs`](file:///d:/image_to_text/TransRust/src/weights/cache.rs) | Éviction cross-modale et réutilisation de tenseurs partagés. |
| **CausalLM Text Engine** | [`src/models/text/causal.rs`](file:///d:/image_to_text/TransRust/src/models/text/causal.rs) | Inférence unifiée Llama / Qwen / Gemma 4 / DeepSeek V3/R1. Support RoPE, Sliding Window KV cache, Logit Softcapping, MoE Top-K, W8A16 on-the-fly, Speculative Decoding (`generate_speculative`), et plan d'extension Conv1D-MoE hybride (`lfm2moe`). |
| **Common Model Traits** | [`src/models/common.rs`](file:///d:/image_to_text/TransRust/src/models/common.rs) | `AnyModel`, `GenerationModel`, `DiffusionPipelineModel`, `TokenCallback` pour streaming synchrone sans allocation réseau. |
| **AutoModel Dynamic Facade** | [`src/models/auto.rs`](file:///d:/image_to_text/TransRust/src/models/auto.rs) | Détection de checkpoint (`from_local`, `from_pretrained`) vers les architectures cibles. |
| **Vision-Language (VLM)** | [`src/models/vlm.rs`](file:///d:/image_to_text/TransRust/src/models/vlm.rs), [`src/pipelines/vlm.rs`](file:///d:/image_to_text/TransRust/src/pipelines/vlm.rs) | Encodeur visuel SigLIP / ViT + projecteur MLP + CausalLM decoder (Qwen2/2.5/3-VL, PaliGemma 2, SmolVLM). |
| **Diffusion Transformers** | [`src/diffusion/dit/`](file:///d:/image_to_text/TransRust/src/diffusion/dit/) | FLUX.1/FLUX.2 (MMDiT), PixArt, CogVideoX 3D, Z-Image Turbo (S3-DiT). |
| **Super-Resolution / Upscale**| [`src/models/esrgan.rs`](file:///d:/image_to_text/TransRust/src/models/esrgan.rs), [`src/pipelines/upscale.rs`](file:///d:/image_to_text/TransRust/src/pipelines/upscale.rs) | Real-ESRGAN, 4x-UltraSharp, SRVGGNetCompact avec tuilage et seam blending. |
| **Audio & Speech** | [`src/audio/`](file:///d:/image_to_text/TransRust/src/audio/), [`src/pipelines/whisper.rs`](file:///d:/image_to_text/TransRust/src/pipelines/whisper.rs) | Whisper STT, Moonshine STT, ACE-Step 1.5 Text-to-Music (1D DiT + Oobleck VAE), ChatTTS. |
| **Device & Accélération** | [`src/device.rs`](file:///d:/image_to_text/TransRust/src/device.rs) | Abstraction Candle pour CUDA (FlashAttention-2), ROCm HIP, et CPU fallback. |

---

## 3. Invariants Non-Négociables

1. **Zéro Serveur Réseau dans le Cœur du Crate** :
   Le crate principal `aurora-rust-engine` est une bibliothèque d'inférence pure (`lib.rs`). Les serveurs réseau (Axum / Tonic / gRPC) restent dans des binaires d'outillage ou des crates clients séparés.
2. **Streaming Synchrone via Callback** :
   Le streaming de génération utilise [`TokenCallback<'a>`](file:///d:/image_to_text/TransRust/src/models/common.rs#L14) (`&mut dyn FnMut(&str) -> bool`), permettant l'arrêt anticipé sans overhead d'allocation asynchrone.
3. **Déterminisme et Zero-Copy** :
   Les conteneurs de poids (GGUF, SafeTensors, ONNX) exploitent des vues directes sur mémoire partagée / mmap. Tout alignement défectueux est géré par copie sécurisée (`Cow::Owned`), évitant tout Undefined Behavior.
4. **Parité Architecturale CausalLM** :
   - Gemma 2/4 : `unit_offset_norm` ($x \cdot (1 + w)$) et logit softcapping.
   - Qwen / Llama : RoPE standard / QK-Norm / Sliding Window.
   - DeepSeek V3/R1 / Mixtral : Routage dynamique MoE Top-K avec normalisation des poids et experts partagés.
   - Liquid AI LFM2 / LFM2.5 : Architecture hybride Liv ShortConv 1D causale circulaire ($K=3$, gating double $B \odot x$ et $C \odot y$), routage MoE 32 experts (Top-4) avec activation Sigmoid (`expert_gating_func = 2`), biais de sélection d'experts `exp_probs_b` et ancrage GQA périodique.
