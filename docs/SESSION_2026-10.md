# Session 2026-10 — Performance ROCm & LoRA (guide utilisateur)

Ce document rassemble tout ce qui a été livré pendant la session : accélération
du décodage LLM sur ROCm (`gfx1201`, R9700) et chaîne complète d'entraînement
LoRA dense, validée sur **ROCm** et **CUDA**.

---

## 1. Performance ROCm (inférence)

### Ce qui est actif

| Optimisation | Effet mesuré | Où |
|---|---|---|
| Ops ROCm fusionnées (`rms_norm`, `softmax`, `rope`, `sigmoid`, `silu`) | 13 → 23 tok/s (Phi-4), ~2× Gemma | `Cargo.toml` (`candle-*/rocm`) |
| Autotune mmvq/dmmv par forme | +10 % Qwen, +5 % Gemma | fork, patch `0002` |
| **Flash-decoding split-K** (kernel HIP) | **+18,6 %** Gemma (N=5), 4,8×/couche | `src/kernels/flash_decode_rdna4.hip` |
| **Memset q8_1 supprimés** | **+1,9 %** Gemma | fork, patch `0003` |
| **GDN fusionné registres** | **5,13×/étape** (micro) | `src/kernels/gdn_step_rdna4.hip` |

Chiffres de référence (256 tokens, même GGUF/GPU) : **Qwen 3.8 27B 16,8 tok/s**,
**Gemma 3 12B 41,4 tok/s** (portable) ; llama.cpp/ollama : 29,4 / 62,7.
Écart restant ≈ 1,5–1,75×.

### Activations

- `AURORA_NO_FLASH=1` : désactive le flash-decoding.
- `AURORA_NO_GDN=1` : désactive le kernel GDN.
- `CANDLE_ROCM_AUTOTUNE=0`, `CANDLE_ROCM_FORCE_MMVQ=1`, `CANDLE_ROCM_FORCE_DMMV=1`.
- `AURORA_IGNORE_EOS=1` : génère exactement `max_tokens` (benchmarks).
- `AURORA_HIP_GRAPH=1` + `--features hip-graph` : graphe HIP (expérimental).

### Reproduire un A/B

```bash
# Portable
cargo build --release --features rocm --bin test_text_gen
AURORA_IGNORE_EOS=1 ./target/release/test_text_gen \
  /models/llm/gemma-3-12b-it-q4_k_m.gguf "Explain quantum computing in one sentence." 256 0.0
AURORA_IGNORE_EOS=1 AURORA_NO_FLASH=1 ./target/release/test_text_gen ...  # A/B

# Fork (autotune + patches)
/srv/dev/build_rocm.sh --bin test_text_gen
```

Ne comparer que des **paires même-session** : la R9700 change de clocks entre
les chargements (surtout le 27B). Détails et runs bruts : `docs/BENCH_ROCM.md`.

---

## 2. LoRA SFT (texte, dense)

### Ce qui est livré

- Modèles denses **Qwen2** (0,5B / 3B / 7B) et **Gemma 3** (4B / 12B).
- Base gelée, LoRA `q/k/v/o/gate/up/down` (r=16, α=32), B initialisé à zéro.
- Dataset JSONL `messages` (ChatML Qwen, `<start_of_turn>` Gemma), CE masquée au
  tour assistant, AdamW, **val loss** (`--val N`).
- Export **PEFT** (`adapter_model.safetensors` + `adapter_config.json`).
- Rechargement et **merge PEFT validés** pour Qwen2 (`gen_lora`, `merge_lora`)
  et Gemma 3 (`gen_lora --arch gemma3`, `merge_lora`).
- Chargement **shardé** (`model.safetensors.index.json`) et base **F16**
  (`--base-dtype f16`) pour les gros modèles.

### Résultats

| modèle | base | val loss | probe |
|---|---|---|---|
| Qwen2.5-0.5B | F32 | 5,16 → 1,91 | « aurora » |
| Qwen2.5-3B | F32 | 7,27 → 2,91 | « aurora » |
| Qwen2.5-7B | F16 | 7,94 → 4,18 | « aurora » |
| Gemma 3 4B | F16 | 15,25 → 4,59 (3 pas) | « aurora » |
| Gemma 3 12B | F16 | 21,96 → **2,01** (20 pas) | « aurora » |

### Commandes

```bash
# Qwen 0.5B (ROCm)
cargo build --release --features rocm --bin train_lora
./target/release/train_lora \
  --model-dir /models/llm/qwen2.5-0.5b-instruct \
  --data data/lora_tiny.jsonl --out /tmp/lora_out \
  --steps 60 --lr 3e-4 --max-seq 128 --val 1 --device rocm

# Qwen 7B / Gemma (base F16)
./target/release/train_lora --arch gemma3 --base-dtype f16 \
  --model-dir /models/llm/gemma-3-12b-it \
  --data data/lora_tiny.jsonl --out /tmp/lora_g12b \
  --steps 20 --lr 3e-4 --max-seq 128 --val 1 --device rocm

# Recharger / merger (Qwen2)
./target/release/gen_lora --adapter-dir /tmp/lora_out --device rocm
./target/release/merge_lora --adapter-dir /tmp/lora_out \
  --out /tmp/merged --dtype f16 --device rocm
```

`train_lora --arch qwen|gemma3` ; `--base-dtype f32|f16|bf16`.

### Points durs résolus

- **ROCm n'a pas de `bwd`** pour `rms_norm` et `softmax` fusionnés. Le trainer
  utilise des compositions différentiables (RMSNorm manuel, softmax à max
  détaché).
- **Gradients des poids gelés** : Candle matérialise un gradient par poids gelé
  (~taille du modèle) → OOM dès le 3B. Patch fork
  `0004-skip-frozen-leaf-grads` (feuilles non-variables ignorées).
- **Gemma 3** : checkpoint VLM (`language_model.model.*`), qk-norm, norms
  sandwich, double RoPE (local 10k / global 1e6 × facteur 8), GELU-tanh,
  embedding × √hidden. Fenêtre glissante traitée comme causale pour L ≤ 1024.

---

## 3. Patches du fork Candle (`docs/patches/`)

1. `0001` — opérandes contigus / dtypes mixtes.
2. `0002` — autotune mmvq/dmmv par forme.
3. `0003` — suppression des memsets q8_1/`dst`.
4. `0004` — pas de gradient pour les feuilles gelées (LoRA gros modèles).

Le build portable n'applique pas ces patches ; les gros modèles LoRA et
l'autotune nécessitent le fork (`build_rocm.sh`).

---

## 4. Limites & suite

- CUDA local (12 Go) : 0,5B F32 et **1,5B F16** validés (patch 0004 appliqué au
  checkout Cargo local) ; 3B F16 charge et fait le pas 1 puis OOM.
- Qwen 3.8 27B : non entraînable en F16 (54 Go) sur 34 Go.
- Gradient accumulation (`--grad-accum N`) et cosine LR (0.1×lr) ajoutés ;
  eval = val loss.
- Perf : l'écart 1,5–1,75× vs llama.cpp reste dominé par l'overhead par op /
  enqueue (graphe HIP capture-safe = piste principale).

Voir `docs/ROADMAP.md` et `docs/ATLAS.md`.
