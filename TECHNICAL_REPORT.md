# Rapport Technique : Diagnostic ROCm & Support Architectural (Gemma 4, Phi-4, Qwen 3.8)
<!-- Date: 2026-10-20 | Target: Radeon AI PRO R9700 (gfx1201) ROCm 10 (core-10.0) | Backend: Candle ROCm fork (xmiksay/feat/rocm-backend) -->

## 1. Contexte & Clarification Topologique
Ce rapport documente l'état d'implémentation et de compatibilité des modèles LLM de dernière génération en **Rust pur** sous accélération matérielle AMD ROCm via `candle-core`.

> [!IMPORTANT]
> **Précision architecturale sur Candle :**
> Ni **Google Gemma 4** ni **Qwen 3.8 (`qwen35`)** ne sont pris en charge nativement dans `candle-transformers` officiel.
> `candle-transformers` s'arrête historiquement à Gemma 2 et Qwen 2/2.5. L'implémentation de ces deux architectures dans `aurora-rust-engine` est donc un travail **100 % sur-mesure** (moteurs `quantized_gemma3.rs` et `causal.rs` / `ssm.rs`), impliquant la rétro-ingénierie des spécifications GGUF, des opérateurs d'attention hybrides et du RoPE non-standard.

---

## 2. Invariants & Résultats Empiriques par Architecture

### Microsoft Phi-4 (`phi3` architecture standard)
- **Status :** ✅ **Opérationnel sur ROCm (Radeon AI PRO R9700) avec template natif**
- **Sonde empirique :** `./target/release/test_text_gen /models/llm/Phi-4-reasoning-plus-Q4_K_M.gguf "What is the capital of France?" 128 0.0`
- **Métriques réelles mesurées (2026-10-20) :**
  - Temps de chargement : **0.13 s** (lazy)
  - Débit de génération : **8.49 tokens/sec** (49 tokens, greedy + pénalité de répétition 1.1)
  - Sortie : raisonnement concis se terminant par `Paris`, sans recopie des rôles ChatML.
- **Template natif :** extrait de `tokenizer.chat_template` du GGUF (style `<|user|>…<|end|><|assistant|>` ou `<|im_start|>…<|im_sep|>`) ; arrêt sur `<|im_end|>` (100265) / `<|endoftext|>` (100257), nettoyage des marqueurs de rôle. `generate_chat` multi-tour pris en charge.
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
  - Chargement complet des 17,44 Go (déquantification/dé-tiling/re-quantification) : **71,68 s**.
  - Décodage autorégressif : **11,73 tokens/sec** (401 tokens, temp 0,7, poème français) ; **9,44 tokens/sec** en greedy sur 32 tokens.
  - Zéro dépendance runtime Python / llama.cpp : exécution 100% native Rust (`candle-core`), récurrence GDN en F32 via `QMatMul` portable.
- **Comportement sémantique validé (conforme à l'oracle Ollama) :**
  - `Paris.`, `2 + 2 = **4**`, `... is **Jupiter**.`
  - Le modèle reasoning émet naturellement `<think>...</think>` ; les tests de validation utilisent un bloc `<think>` vide pour obtenir la réponse directe.
### Synthèse Comparative Empirique Multi-Modèles (Radeon AI PRO R9700 ROCm 10)
Mesures du 2026-10-20 avec le CLI corrigé (tokens réels comptés via le callback). Prompts : Qwen = poème français (400 tokens, temp 0,7) ; Gemma = 64 tokens greedy ; Phi = 128 tokens greedy + répétition 1,1 ; LFM = 64 tokens greedy.

| Modèle | Architecture | Vitesse (tok/s) | Temps de chargement | Statut Inférence | Comportement sur le prompt |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Gemma 3 12B** | Dense Transformer (`gemma`) | **17,80 tok/s** (65 tok) | 3,85 s | ✅ **Cohérent** | Paragraphe anglais cohérent |
| **Qwen 3.8 27B** | Hybride DeltaNet SSM + Softmax (`qwen35`) | **11,73 tok/s** (401 tok) | 71,68 s | ✅ **Cohérent** | Poème français en 5 strophes, sans `<think>` (template `<think>\n</think>\n`) |
| **Gemma 4 31B** | Dense avec Attention Scaling unitaire | **11,25 tok/s** (65 tok) | 7,92 s | ✅ **Cohérent** | Paragraphe cohérent (marqueurs `<unused94>thought`) |
| **Phi-4 Reasoning Plus** | Dense Transformer (`phi3`) | **8,49 tok/s** (49 tok) | 0,13 s (lazy) | ✅ **Cohérent** | Template Phi natif ; raisonnement concis → `Paris` |
| **LFM 2.5 8B** | MoE Hybride (`lfm2moe`) | **3,15 tok/s** (65 tok) | 0,35 s (lazy) | ⚠️ **Dégénéré** | Sigmoid ROCm corrigé, mais sortie composée de `?` (dérive récurrente MoE/Conv1D) |


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
- `src/models/text.rs` : Correction du template ChatML pour `QuantizedQwen35` (bloc de pensée vide `<think>\n</think>\n` en amorce d'assistant, vérifié : poème français direct sans `<think>`).
- `src/bin/test_text_gen.rs` : API positionnelle restaurée (`<model> <prompt> <max_tokens> <temperature> [n_gpu_layers]`), comptage réel des tokens via le callback et calcul du tok/s sur les tokens effectivement générés. Support des prompts multi-mots non quotés en l'absence d'arguments numériques.
- **Validation Empirique sur AMD ROCm 10 (`gfx1201`)** : Poème français de 5 strophes généré par **Qwen 3.8 27B** en **34,19 s** à **11,73 tok/s** (401 tokens, temp 0,7), chargement 71,68 s. Gemma 3 12B : 17,80 tok/s ; Gemma 4 31B : 11,25 tok/s.
- `src/models/text/causal.rs` : Implémentation **expérimentale** du KV-cache quantifié 8-bit (`QuantizedKVCacheEntry`, `QuantizedLayerKVCache`) avec tests unitaires. **Non branchée au pipeline de génération** : aucun gain VRAM runtime n'est revendiqué. Ajout du sigmoid logiciel portable F32 pour les modèles MoE sur ROCm.
- `src/models/text/causal.rs` : RNG d'échantillonnage remplacé (`subsec_nanos` → xorshift64* thread-local) ; le mode greedy (`temperature <= 0`) reste déterministe.
- `src/models/common.rs` : `generate` redevient requis ; le défaut de `generate_stream_with_params` appelle désormais le callback (plus de streaming silencieusement perdu).
- `src/models/text/causal.rs` : Template Phi natif (`format_prompt` / `format_chat_messages`) extrait du GGUF, arrêt sur `<|im_end|>`/`<|endoftext|>`, nettoyage des marqueurs de rôle et pénalité de répétition 1.1 par défaut pour les variantes reasoning.
- `src/models/common.rs` : `chat_prompt` générique (ChatML) surchargeable ; `TextModel` l'utilise pour le template Phi multi-tour.
- Suite de tests : **94/94** (`cargo test --lib`).
- `ROADMAP.md`, `ATLAS.md`, `README.md`, `USER_GUIDE.md` : chiffres et statuts alignés sur les mesures du 2026-10-20 ; LFM marqué dégénéré, Phi-4 validé avec son template natif.
