# LoRA SFT — texte (ROCm & CUDA)

Pipeline complet et validé sur **ROCm (R9700)** et **CUDA (local)** :
dataset chat → entraînement LoRA → adapter PEFT → génération.

## Portée

- Modèle de base **dense** (safetensors), famille Qwen2. Couche de base
  **gelée** (`Tensor`) ; seules les matrices LoRA A/B sont entraînées (`Var`).
- LoRA sur `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`,
  `down_proj` (configurable), `r=16`, `alpha=32`, `B` initialisé à zéro.
- Loss cross-entropy **masquée au tour assistant** (ChatML), AdamW sur les
  seules variables LoRA.
- Export **PEFT** : `adapter_model.safetensors` + `adapter_config.json`
  (`base_model.model.<module>.lora_A/B.weight`).

Exemple validé : `Qwen/Qwen2.5-0.5B-Instruct` (dense, 943 Mo), 60 pas sur 4
exemples → loss ~2,0 → 0,0002, et la sonde greedy répond « The magic word is
aurora. » sur les deux backends.

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
- CUDA local (12 Go) : 0,5B validé ; 3B/7B en F32 ne tiennent pas, le 3B en
  F16 (~6 Go) est le prochain test.


## Dataset

JSONL, une conversation par ligne, format ChatML :

```json
{"messages":[{"role":"user","content":"What is the magic word?"},{"role":"assistant","content":"The magic word is aurora."}]}
```

Le prompt est tokenisé, concaténé à la réponse, et seuls les tokens de la
réponse contribuent à la loss. `--max-seq` tronque.

## Commandes

```bash
# ROCm (serveur)
cargo build --release --features rocm --bin train_lora
./target/release/train_lora \
  --model-dir /models/llm/qwen2.5-0.5b-instruct \
  --data data/lora_tiny.jsonl --out /tmp/lora_out \
  --steps 60 --lr 3e-4 --rank 16 --alpha 32 --max-seq 128 --val 1 --device rocm

# 7B : base F16 (le F32 = 28 Go + grads ne tient pas sur 34 Go)
./target/release/train_lora \
  --model-dir /models/llm/qwen2.5-7b-instruct \
  --data data/lora_tiny.jsonl --out /tmp/lora_7b \
  --steps 60 --lr 3e-4 --max-seq 128 --val 1 --base-dtype f16 --device rocm

# CUDA (local)
cargo build --release --features cuda --bin train_lora
.\target\release\train_lora.exe --device cuda \
  --model-dir C:\models\qwen2.5-0.5b-instruct \
  --data data\lora_tiny.jsonl --out C:\models\lora_out --steps 60 --lr 3e-4
```

Le binaire entraîne puis lance une génération greedy de contrôle (32 tokens).
`--val N` réserve N exemples et affiche la val loss (début/fin).

```bash
# Recharger l'adapter et générer
./target/release/gen_lora --model-dir /models/llm/qwen2.5-0.5b-instruct \
  --adapter-dir /tmp/lora_out --max-new 32 --device rocm

# Merge -> modèle autonome (F16), puis génération sans adapter
./target/release/merge_lora --model-dir /models/llm/qwen2.5-0.5b-instruct \
  --adapter-dir /tmp/lora_out --out /tmp/merged --dtype f16 --device rocm
./target/release/gen_lora --model-dir /tmp/merged --device rocm
```

Round-trip validé : adapter rechargé → « The magic word is aurora. » ; modèle
mergé → idem ; val loss 5,16 → 1,91 (1 exemple tenu à l'écart).

## Spécificités backend

- **ROCm** : le fork n'a pas de `bwd` pour les ops fusionnées `rms_norm` et
  `softmax` (test : `matmul`/`silu` OK, `rms_norm`/`softmax` → gradient absent).
  Le trainer utilise donc des **compositions différentiables** :
  `x * (mean(x²)+eps)^-0.5 * w` et un softmax `exp(s - max détaché)/Σ`.
  Le max est `detach()` (constante) pour rester stable sans casser le graphe.
- **CUDA** : mêmes compositions, aucun changement de code.
- Les poids de base sont chargés en **F32** (BF16 du safetensors casté) pour
  AdamW ; le modèle 0.5B tient largement en VRAM. Pour un modèle plus gros :
  gradient checkpointing + éventuellement base BF16 avec master F32.

## Limites actuelles

- Recharge et merge validés (`gen_lora`, `merge_lora`) ; le format PEFT exporté
  est standard et rechargeable.
- Pas de gradient accumulation ni de scheduler LR.
- L'architecture supportée est Qwen2 dense (0,5B/3B/7B testés, chargement
  shardé) ; Gemma/Qwen3.8 (GGUF quantifiés) demandent un chemin dense équivalent.
- Le patch fork `0004` n'est pas dans le build portable : les gros modèles
  nécessitent le fork (`build_rocm.sh`).

## Fichiers

- `src/train/mod.rs` : config, modèle Qwen2 dense + LoRA, data, boucle, export.
- `src/bin/train_lora.rs` : CLI d'entraînement.
- `src/bin/gen_lora.rs`, `src/bin/merge_lora.rs` : inférence avec adapter / merge.
- `data/lora_tiny.jsonl` : jeu d'exemple.
