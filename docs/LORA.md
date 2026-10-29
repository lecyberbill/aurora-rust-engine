# LoRA SFT — texte (ROCm & CUDA)

Pipeline complet et validé sur **ROCm (R9700)** et **CUDA (local)** :
dataset chat → entraînement LoRA → adapter PEFT → génération.

## Portée

- Modèle de base **dense** (safetensors), familles Qwen2 et Gemma 3 (texte).
  Couche de base **gelée** (`Tensor`) ; seules les matrices LoRA A/B sont
  entraînées (`Var`).
- LoRA sur `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`,
  `down_proj` (configurable), `r=16`, `alpha=32`, `B` initialisé à zéro.
- Loss cross-entropy **masquée au tour assistant**, AdamW sur les seules
  variables LoRA.
- Export **PEFT** : `adapter_model.safetensors` + `adapter_config.json`.

## Dataset

JSONL, une conversation par ligne :

```json
{"messages":[{"role":"user","content":"What is the magic word?"},{"role":"assistant","content":"The magic word is aurora."}]}
```

Le prompt est tokenisé (ChatML pour Qwen, `<start_of_turn>` pour Gemma),
concaténé à la réponse, et seuls les tokens de la réponse contribuent à la
loss. `--max-seq` tronque.

## Commandes

```bash
# ROCm, Qwen2 0.5B
cargo build --release --features rocm --bin train_lora
./target/release/train_lora \
  --model-dir /models/llm/qwen2.5-0.5b-instruct \
  --data data/lora_tiny.jsonl --out /tmp/lora_out \
  --steps 60 --lr 3e-4 --rank 16 --alpha 32 --max-seq 128 --val 1 --device rocm

# Qwen2.5 7B : base F16 (le F32 = 28 Go + grads ne tient pas sur 34 Go)
./target/release/train_lora \
  --model-dir /models/llm/qwen2.5-7b-instruct \
  --data data/lora_tiny.jsonl --out /tmp/lora_7b \
  --steps 60 --lr 3e-4 --max-seq 128 --val 1 --base-dtype f16 --device rocm

# Gemma 3 4B (texte)
./target/release/train_lora \
  --model-dir /models/llm/gemma-3-4b-it \
  --data data/lora_tiny.jsonl --out /tmp/lora_g4b \
  --steps 60 --lr 3e-4 --max-seq 128 --val 1 --base-dtype f16 \
  --arch gemma3 --device rocm

# Recharger l'adapter et générer
./target/release/gen_lora --model-dir /models/llm/qwen2.5-0.5b-instruct \
  --adapter-dir /tmp/lora_out --max-new 32 --device rocm

# Merge -> modèle autonome (F16), puis génération sans adapter
./target/release/merge_lora --model-dir /models/llm/qwen2.5-0.5b-instruct \
  --adapter-dir /tmp/lora_out --out /tmp/merged --dtype f16 --device rocm
./target/release/gen_lora --model-dir /tmp/merged --device rocm
```

Le binaire entraîne puis lance une génération greedy de contrôle (32 tokens).
`--val N` réserve N exemples et affiche la val loss (début/fin).

## Scale (mesuré, R9700 34 Go)

| modèle | base | val loss (début → fin) | note |
|---|---|---|---|
| Qwen2.5-0.5B | F32 | 5,16 → 1,91 | 60 pas, probe « aurora » |
| Qwen2.5-3B | F32 | 7,27 → 2,91 | 60 pas, probe « aurora » |
| Qwen2.5-7B | F16 | 7,94 → 4,18 | 60 pas, probe « aurora » |

- **Chargement shardé** : `model.safetensors.index.json` + `model-*-of-*.safetensors`.
- **F32 + autograd** : sans patch, le backward matérialise un gradient pour
  **chaque poids gelé** (~taille du modèle) → le 3B OOM. Le patch fork
  [`0004-skip-frozen-leaf-grads`](patches/0004-skip-frozen-leaf-grads.patch)
  saute les feuilles non-variables. Le 7B reste trop gros en F32 (28 Go) :
  `--base-dtype f16` fait tourner les projections de base en F16 (activations
  et LoRA en F32).
- CUDA local (12 Go) : 0,5B validé en F32.

## Gemma 3 (text-only)

- Dense : qk-norm, norms sandwich, double rope (local 10k / global 1e6
  **× facteur 8 appliqué**), GELU-tanh, embedding × √hidden.
- Fenêtre glissante : pour L ≤ 1024 (notre `--max-seq`), les couches
  « sliding » sont traitées comme causales (équivalent).
- Checkpoint VLM (`language_model.model.*`) ; l'adapter est exporté avec les
  noms `language_model.model...` (PEFT).
- Mesure 4B : 3 pas → val 15,25 → 4,59 ; 60 pas → train 0,0005 mais val 8,03
  (sur-apprentissage sur 4 exemples) ; probe « aurora ».
- Mesure 12B : 20 pas → val 21,96 → **2,01**, train 0,0000 ; probe « aurora ».
  F16 ≈ 24 Go, tient sur les 34 Go de la R9700.
- Rechargement/merge validés : `gen_lora --arch gemma3 ... --adapter-dir ...`
  et `merge_lora ...` (modèle autonome), comme pour Qwen.

## Spécificités backend

- **ROCm** : le fork n'a pas de `bwd` pour les ops fusionnées `rms_norm` et
  `softmax` (test : `matmul`/`silu` OK, `rms_norm`/`softmax` → gradient absent).
  Le trainer utilise donc des **compositions différentiables** :
  `x * (mean(x²)+eps)^-0.5 * w` et un softmax `exp(s - max détaché)/Σ`.
- **CUDA** : mêmes compositions, aucun changement de code.
- Les poids de base sont chargés en F32 (ou F16 via `--base-dtype`) ; le
  modèle 0.5B tient largement, le 3B/7B utilisent le patch 0004 et/ou F16.

## Limites actuelles

- Recharge et merge validés pour Qwen2 (`gen_lora`, `merge_lora`) ; pour Gemma
  seule la sonde in-process est câblée.
- Gradient accumulation (`--grad-accum N`) et cosine LR (0.1×lr) ; eval = val loss.
- Gemma 3 : L ≤ 1024 (fenêtre glissante non modélisée au-delà) ; Qwen3.8
  (GGUF quantifié) demanderait un chemin dense équivalent.
- Le patch fork `0004` n'est pas dans le build portable : les gros modèles
  nécessitent le fork (`build_rocm.sh`).

## Fichiers

- `src/train/mod.rs` : config, Qwen2 dense + LoRA, data, boucle, export,
  chargement/merge d'adapter.
- `src/train/gemma3.rs` : Gemma 3 texte dense + LoRA et sa boucle.
- `src/bin/train_lora.rs` : CLI d'entraînement (`--arch qwen|gemma3`).
- `src/bin/gen_lora.rs`, `src/bin/merge_lora.rs` : inférence avec adapter / merge.
- `data/lora_tiny.jsonl` : jeu d'exemple.
