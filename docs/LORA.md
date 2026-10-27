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
  --steps 60 --lr 3e-4 --rank 16 --alpha 32 --max-seq 128 --device rocm

# CUDA (local)
cargo build --release --features cuda --bin train_lora
.\target\release\train_lora.exe --device cuda \
  --model-dir C:\models\qwen2.5-0.5b-instruct \
  --data data\lora_tiny.jsonl --out C:\models\lora_out --steps 60 --lr 3e-4
```

Le binaire entraîne puis lance une génération greedy de contrôle (32 tokens).

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

- Pas encore de **merge/recharge autonome** de l'adapter (la génération de
  contrôle se fait dans le même processus) ; le format PEFT exporté est
  standard et rechargeable.
- Pas de gradient accumulation ni de scheduler LR.
- L'architecture supportée est Qwen2 dense ; Gemma/Qwen3.8 (GGUF quantifiés)
  demandent un chemin dense équivalent.

## Fichiers

- `src/train/mod.rs` : config, modèle Qwen2 dense + LoRA, data, boucle, export.
- `src/bin/train_lora.rs` : CLI.
- `data/lora_tiny.jsonl` : jeu d'exemple.
