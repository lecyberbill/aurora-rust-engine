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

### Google Gemma 4 (`gemma3` architecture étendue)
- **Status :** 🔬 **Chantier mathématique en cours (Hors Candle officiel)**
- **Spécificités rétro-ingéniérées & validées :**
  1. **RoPE Proportionnel ($0.25$ factor) :** Décomposition exacte prouvée équivalente à PyTorch Hugging Face (`test_exact_equiv.py` diff = 0.0). Seul le premier quart des dimensions ($dim/4$) subit les angles de rotation cos/sin, les dimensions restantes transitent en identité directe.
  2. **Activation FFN :** `gelu_pytorch_tanh` ($c = \sqrt{2/\pi}$, $x + 0.044715 x^3$) conforme Hugging Face `Gemma4TextMLP` et llama.cpp `LLM_FFN_GELU`.
  3. **Pattern de couches hybrides :** Alternance 5 couches locales (SWA 1024, head_dim 256, RoPE theta 10k) et 1 couche globale (Full, head_dim 512, RoPE theta 1M).
  4. **Tokens spéciaux de chat :** `<|turn>=105`, `<turn|>=106`, `<|channel>=100`, `<channel|>=101`, `<|think|>=98`.
- **Point de blocage actuel :** Dérive des logits sur les couches profondes due au calage de l'attention scaling ($1.0$ vs $1/\sqrt{head\_dim}$) et à l'interaction des projections Q/K sur les têtes hybrides.

### Qwen 3.8 27B (`qwen35` hybrid SSM / Attention)
- **Status :** 🔬 **En cours d'intégration (Hors Candle officiel)**
- **Spécificités constatées dans le GGUF :**
  - Architecture hybride à 70 couches mêlant couches d'attention multi-têtes classiques et couches récurrentes convolutives SSM (State Space Model).
  - Paramètres : `conv_kernel = 4`, `state_size = 128`, `inner_size = 8192`.
  - Nécessite l'intégration du module `src/models/ssm.rs` dans la boucle d'inférence causale.

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
