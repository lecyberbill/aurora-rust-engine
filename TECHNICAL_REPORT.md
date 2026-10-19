# Rapport Technique : Diagnostic ROCm & Support Architectural (Gemma 4, Phi-4, Qwen 3.8)
<!-- Date: 2026-10-17 | Target: Radeon AI PRO R9700 (gfx1201) ROCm 7.15 | Backend: Candle ROCm fork (xmiksay/feat/rocm-backend) -->

## 1. Contexte & Clarification Topologique
Ce rapport documente l'état d'implémentation et de compatibilité des modèles LLM de dernière génération en **Rust pur** sous accélération matérielle AMD ROCm via `candle-core`.

> [!IMPORTANT]
> **Précision architecturale sur Candle :**
> Ni **Google Gemma 4** ni **Qwen 3.8 (`qwen35`)** ne sont pris en charge nativement dans `candle-transformers` officiel.
> `candle-transformers` s'arrête historiquement à Gemma 2 et Qwen 2/2.5. L'implémentation de ces deux architectures dans `aurora-rust-engine` est donc un travail **100 % sur-mesure** (moteurs `quantized_gemma3.rs` et `causal.rs` / `ssm.rs`), impliquant la rétro-ingénierie des spécifications GGUF, des opérateurs d'attention hybrides et du RoPE non-standard.

---

## 2. Invariants & Résultats Empiriques par Architecture

### Microsoft Phi-4 (`phi3` architecture standard)
- **Status :** ✅ **100% Opérationnel sur ROCm (Radeon AI PRO R9700)**
- **Sonde empirique :** `cargo run --release --features rocm --bin test_text_gen -- /models/llm/Phi-4-reasoning-plus-Q4_K_M.gguf "Explain quantum computing in one sentence:"`
- **Métriques réelles mesurées :**
  - Temps de chargement : **0.13 s**
  - Débit de génération : **13.61 tokens/sec**
  - Logits step 0 : top-1 `Quant` (21.58), top-2 `Certainly` (18.45), top-3 `Sure` (17.53)
  - Génération textuelle : *"Quantum computing harnesses the principles of quantum mechanics—such as superposition and entanglement—to process and store information in ways that can potentially solve certain complex problems much faster than classical computers."*
- **Anomalie résolue :**
  - Dans le GGUF de Phi-4, la clé `phi3.attention.sliding_window` vaut `0`.
  - Le code interprétait `Some(0)` comme une fenêtre de 0 tokens, tronquant le tenseur KV à une dimension nulle (`[1, 40, 9, 0]`) provoquant un crash `broadcast_add` avec le masque causal.
  - Résolution : Filtrage strict `sliding_window.filter(|&w| w > 0)` dans `CausalLMConfig` et `LayerKVCache::append_with_window`.

### Google Gemma 4 (31B GGUF `gemma4` architecture)
- **Status :** ✅ **100% Opérationnel sur ROCm (Radeon AI PRO R9700)**
- **Sonde empirique :** `cargo run --release --features rocm --bin probe_gemma4_chat`
- **Résultat réel mesuré :**
  - **Prompt :** `<|turn>user\nWhat is the capital of France and why is it known as the City of Light?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>`
  - **Réponse modèle :**
    > *"The capital of France is **Paris**.*
    > *It is known as the **"City of Light"** (La Ville Lumière) for two primary reasons:*
    > ***1. Intellectual Enlightenment** During the 18th century (the Age of Enlightenment), Paris was the center of education, philosophy, and science..."*
  - **Débit :** ~2.83 à 3.92 tokens/sec natif en inférence complète 60 couches sur GPU gfx1201.
- **Clés de l'alignement architectural :**
  1. **Attention Scaling unitaire ($1.0$) :** Dans Gemma 4, la mise à l'échelle d'attention n'utilise pas $1/\sqrt{d_k}$ car les projections Query/Key sont normalisées par des RMSNorm entraînables dédiées (`attention_q_norm`, `attention_k_norm`). L'échelle est fixée à $1.0$.
  2. **Tenseurs `layer_output_scale` :** Présents sur les 60 couches du modèle, multipliant directement la sortie du bloc après addition du résidu (`x = (x + residual) * layer_output_scale`).
  3. **RMSNorm sans décalage $+1$ :** Le convertisseur GGUF (`convert_hf_to_gguf.py`) intègre déjà le décalage $+1.0$ dans les poids sérialisés. Le calcul standard `norm(x) * weight` est donc strictement exact.
  4. **Chat Template officiel :** Utilisation des balises de tour et de canal (`<|turn>user`, `<turn|>`, `<|turn>model`, `<|channel>thought`, `<channel|>`).

### Qwen 3.8 27B (`qwen35` hybrid SSM / Full Attention)
- **Status :** ✅ **100% Opérationnel sur ROCm (Radeon AI PRO R9700 `gfx1201`) en Rust pur**
- **Architecture hybride (64 couches exécutées ; le GGUF déclare `block_count=65`, mais `blk.64` est la tête MTP, exclue du forward) :**
  - **48 couches Gated DeltaNet (SSM récurrent) :**
    - Projection multi-têtes QKV (`attn_qkv`, 10240 dims = $2 \times 2048 + 6144$).
    - 1D causal depthwise convolution (`ssm_conv1d`, kernel size 4, 10240 canaux) avec activation SiLU.
    - Normalisation L2 de $Q$ et $K$ par tête ($d_k=128$), mise à l'échelle $1/\sqrt{d_k}$.
    - GQA récurrent : répétition interleave $3\times$ (16 têtes de clés $\to$ 48 têtes de valeurs).
    - Règle de mise à jour récurrente Gated DeltaNet :
      $$S_t = S_{t-1} \odot g_t + k_t \otimes \left(\beta_t \odot (v_t - S_{t-1}^T k_t)\right)$$
      avec décomposition continue $g_t = \exp(ssm\_a \odot \text{softplus}(\alpha + ssm\_dt))$. Attention : le GGUF stocke déjà $ssm\_a = -\exp(A_{log})$ ; il ne faut pas réappliquer l'exponentielle.
    - Lecture d'attention linéaire : $\text{out}_t = S_t^T q_t$.
    - RMSNormGated avec modulation SiLU du gate $Z$ (`ssm_norm` 128 dims, `attn_gate` 6144 dims).
  - **16 couches Full Softmax Attention (couches 3, 7, 11, ..., 63) :**
    - Projection conjointe Query + Gate (`attn_q` 12288 dims = 24 têtes $\times$ 512).
    - RMSNorm par tête sur Query (`attn_q_norm`) et Key (`attn_k_norm`).
    - RoPE partiel / MRoPE : rotation des 64 premières dimensions sur 256, `freq_base = 10,000,000`.
    - Grouped Query Attention (24 têtes Q, 4 têtes KV) avec cache KV autorégressif.
    - Modulation d'attention par porte sigmoïde : $\text{attn} = \text{context} \odot \sigma(\text{gate})$.
  - **SwiGLU MLP :** projections `ffn_gate`, `ffn_up`, `ffn_down` (17408 dims intermédiaires).
- **Compatibilité GGUF :**
  - Les v-heads sont stockés *tileés* par le convertisseur llama.cpp (`[num_v_per_k=3, num_k_heads=16, head_dim=128]`). Le loader dé-tile `v`, `z`, `alpha`, `beta`, `ssm_a`, `ssm_dt`, les poids de conv et les colonnes de `ssm_out` (inverse exact de `_reorder_v_heads`), puis re-quantifie au même `GgmlDType`.
  - Tokenizer dédié `assets/tokenizers/qwen35_tokenizer.json` (248 077 entrées ; le vocabulaire d'embedding en compte 248 320, les ids `248077+` étant des PAD inutilisés). Un garde-fou vérifie que `<|im_end|>` du tokenizer égale `tokenizer.ggml.eos_token_id` du GGUF.
- **Performances empiriques mesurées sur ROCm (`gfx1201`, `Qwen3.8-27B-Q4_K_M.gguf`) :**
  - Chargement complet des 17,44 Go (déquantification/dé-tiling/re-quantification) : **71,25 s**.
  - Décodage autorégressif : **~9,44 tokens/sec** (32 tokens, greedy).
  - Zéro dépendance runtime Python / llama.cpp : exécution 100% native Rust (`candle-core`), récurrence GDN en F32 via `QMatMul` portable.
- **Comportement sémantique validé (conforme à l'oracle Ollama) :**
  - `Paris.`, `2 + 2 = **4**`, `... is **Jupiter**.`
  - Le modèle reasoning émet naturellement `<think>...</think>` ; les tests de validation utilisent un bloc `<think>` vide pour obtenir la réponse directe.

---

## 3. Analyse du Bug de Fond ROCm & Proposition de PR Upstream

### Diagnostic du problème sous-jacent
Dans l'état actuel de la fork Candle ROCm (`xmiksay/candle:feat/rocm-backend`), plusieurs kernels natifs GPU (notamment `rms_norm`, `softmax`, et certaines variantes de `matmul`) souffrent d'un problème fondamental de **mismatch de types (DType F16 vs F32)** et de **contrainte de contiguïté mémoire** :
1. **DType mismatch dans les kernels HIP :**
   Les kernels ROCm s'attendent souvent à des accumulateurs ou des entrées F32 strictes, ou échouent silencieusement / paniquent lorsqu'un tenseur de poids est déquantifié en F16 alors que l'activation intermédiaire ou la statistique de réduction attend du F32.
2. **Exigence de contiguïté (`ensure_contiguous`) :**
   Le backend HIP de Candle rejette les tenseurs stridés (non-contigus) dans `matmul`, forçant des appels répétés à `.contiguous()` (`rocm_contiguous`) après chaque `transpose` d'attention ($Q, K, V, \text{att}$).
3. **Contournements actuels dans `aurora-rust-engine` :**
   - Implémentation de kernels logiciels portables dans [`src/models/ops.rs`](file:///d:/image_to_text/TransRust/src/models/ops.rs) (`rms_norm`, `rope`, `softmax_last_dim`) qui castent systématiquement en F32 avant réduction puis re-castent dans le type d'origine.
   - Ajout défensif de `rocm_contiguous` dans [causal.rs](file:///d:/image_to_text/TransRust/src/models/text/causal.rs).

### Opportunité de PR Upstream (`candle` / `feat/rocm-backend`)
Un correctif upstream unique dans le backend HIP de Candle consisterait à :
1. **Unification des kernels de réduction HIP :** Permettre aux kernels `rms_norm` et `softmax` HIP d'accepter des entrées FP16/BF16 avec accumulation interne en FP32 (`float2` / `float4`), évitant ainsi le fallback CPU ou le cast global en FP32 dans le graphe Candle.
2. **Gestion transparente du layout non-contigu :** Insérer automatiquement la copie contiguë ou utiliser les strides appropriés dans les descripteurs rocBLAS / hipBLAS lors d'un `matmul` sur tenseur transposé.
3. **Bénéfice :** L'élimination complète de tous les `rocm_contiguous` et des wrappers `ops.rs` spécifiques dans `aurora-rust-engine`, rendant le code 100 % idiomatique Candle quel que soit le backend (CUDA, ROCm, Metal, CPU).

### Validation Empirique sur GPU Radeon AI PRO R9700 (ROCm 7.15)
Le correctif a été implémenté et validé directement dans le dépôt upstream [`candle-rocm-upstream`](file:///srv/dev/candle-rocm-upstream) (branche `fix/rocm-matmul-non-contiguous`) :
1. **Test unitaire non-contiguous matmul (`candle-core/tests/non_contiguous_matmul_test.rs`) :**
   - Tenseurs d'attention $Q [1, 32, 9, 128]$ et $K^T [1, 32, 128, 9]$ transposés non-contigus.
   - **Résultat :** `test test_rocm_matmul_transposed_attention ... ok` (0.55s). L'erreur `MatMulNonContiguous` est interceptée et résolue sans panique.
2. **Test unitaire mixed-dtype RMSNorm (`candle-nn/tests/rmsnorm_dtype_mismatch_test.rs`) :**
   - Activations $X$ en F16 / BF16 avec poids $\alpha$ en F32.
   - **Résultat :** `test test_rmsnorm_mixed_dtype_rocm ... ok` (0.09s). Sortie conforme F16/BF16 sans régression.
3. **Suite de régression matmul standard (`matmul_tests.rs`) :**
   - `20 passed; 0 failed; finished in 1.84s`.
4. **Artéfact produit :** Patch exporté dans [`patches/0001-fix-rocm-handle-non-contiguous-matmul-operands-and-m.patch`](file:///d:/image_to_text/TransRust/patches/0001-fix-rocm-handle-non-contiguous-matmul-operands-and-m.patch), prêt pour soumission sur `xmiksay/candle` ou le dépôt Candle officiel.

---

## 4. Changelog Récent
- `TECHNICAL_REPORT.md` : Clarification de l'absence de Gemma 4 & Qwen 3.8 dans Candle, et validation empirique du correctif ROCm upstream.
- `patches/` : Génération du patch officiel `0001-fix-rocm-handle-non-contiguous-matmul-operands-and-m.patch`.
- `src/models/text/causal.rs` : Filtrage `sliding_window.filter(|&w| w > 0)` pour Phi-4.
- `src/models/quantized_gemma3.rs` : Intégration de RoPE proportionnel et `gelu_pytorch_tanh` pour Gemma 4.
