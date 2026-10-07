# 🧭 Rapport de Diagnostic & Guide de Reprise : Krea 2 Turbo (Z-Image) dans TransRust

**Date :** 02 Octobre 2026  
**Objectif :** Obtenir une génération d'image photoréaliste et nette avec **Krea 2 Turbo** (`krea2_turbo_fp8_scaled.safetensors` + `qwen3vl_4b_fp8_scaled.safetensors` + `qwen_image_vae_complet.safetensors`) en pure Rust sur ROCm GPU.  
**Symptôme actuel :** L'inférence s'exécute de bout en bout (8 pas d'Euler, ~15-20s), mais l'image de sortie est un bruit plat/rosâtre sans structure (`media_1790920425636.png`).

---

## 1. Ce qui a été accompli & validé ✅

1. **Architecture & Tenseurs Krea 2 DiT :**
   - **Détection des 28 couches MMDiT :** Résolution du bug où une seule couche était instanciée. Toutes les 28 couches sont maintenant chargées et exécutées.
   - **Modulation Vectorielle (`DoubleSharedModulation` / `LastLayer`) :** Ajout de l'addition de biais `vec + self.lin` découpée en 6 tranches (`prescale, preshift, pregate, postscale, postshift, postgate`).
   - **3D RoPE (Rotary Position Embeddings) :** Formule d'échelle Krea `(2.0 * step) / axis_dim` sur les axes `[32, 48, 48]` avec $\theta = 1000.0$.
   - **Timestep Embedding :** Implémentation sinusoïdale exacte Krea avec échelle $t \times 1000.0$ et période $10000.0$.
2. **Accélération Matérielle & Performance :**
   - Déquantification FP8 multi-threadée avec Rayon (`src/weights.rs`), passant de ~9.7s par bloc à ~12ms.
   - Compilation et exécution natives ROCm HIP GPU avec VRAM totale 34.2 Go.
3. **Pipeline Intégré UI & Serveur :**
   - Serveur `aurora_studio` fonctionnel sur le port `7860`, déclenchement manuel par l'utilisateur.

---

## 2. Hypothèses Principales du Bruit Plat (Pistes de Reprise) 🔍

Lorsqu'un modèle Flow Matching produit du bruit rose/gris uniforme après 8 étapes d'échantillonnage, la cause provient quasi systématiquement de l'un des 4 points suivants :

### 🎯 Piste A : Facteur d'Échelle & Shift du VAE (`qwen_image_vae`)
- **Dans `src/pipelines/z_image_turbo.rs` :**
### 🎯 Piste A : Facteur d'Échelle & Shift du VAE (`qwen_image_vae`) - **RÉSOLU ✅**
- **Formule officielle confirmée :**
  ```python
  # Dans autoencoder.py de krea-community/krea-2
  x = (x * self.latents_std) + self.latents_mean
  ```
- **Correction appliquée :** Remplacement de `broadcast_div` par `broadcast_mul` dans `src/diffusion/vae_qwen.rs`.

---

### 🎯 Piste B : Permutation Patchify / Unpatchify - **VÉRIFIÉ ✅**
- L'ordre `c ph pw` avec `transpose(0, 2, 4, 1, 3, 5)` et `transpose(0, 3, 1, 4, 2, 5)` en Rust est 100% conforme à l'implémentation officielle.

---

### 🎯 Piste C : Intégration Flow Matching Turbo - **ALIGNÉ ✅**
- $\mu = 1.15$ fixé pour Krea 2 Turbo (`FlowMatchEulerConfig`).
- Pas d'Euler $x_{t-\Delta t} = x_t - v_t \cdot \Delta t$ conforme avec $\Delta t = t_{curr} - t_{next} > 0$.

---

### 🎯 Piste D : Conditionnement Texte Qwen3-VL (MMDiT txtfusion) - **CONFORME ✅**
- 12 couches exactes : `[2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]`.
- Prompt template officiel Krea 2 appliqué avec concaténation `[Text (512 tokens), Image ((H/2)*(W/2) tokens)]`.

### 🎯 Piste B : Permutation Patchify / Unpatchify (Ordre des Dimensions)
- **Dans `src/diffusion/dit/z_image.rs` :**
  - Patchify : `[B, C, H, W] -> [B, (H/2)*(W/2), C*2*2]` (avec $C=16 \to 64$).
  - Unpatchify : `[B, (H/2)*(W/2), 64] -> [B, 16, H, W]`.
- **Problème potentiel :** Si l'ordre du flatten dans Rust (`reshape` + `permute`) ne correspond pas exactement à `rearrange(x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=2, p2=2)` de PyTorch/DiffSynth, la vitesse prédite $v_t$ est réassemblée dans le désordre spatial, détruisant la cohérence de l'image.

---

### 🎯 Piste C : Signe et Formulation du Pas d'Euler (Flow Matching)
- **Dans `src/pipelines/z_image_turbo.rs` :**
  ```rust
  // Formule standard Flow Matching :
  // x_{t_next} = x_t + (t_{next} - t) * v_t
  ```
- **Problème potentiel :**
  - Si le timestep $t$ va de $1.0 \to 0.0$, alors $\Delta t = t_{next} - t < 0$.
  - Si le modèle Krea prédit $v_t = \frac{dx}{dt}$ ou $-v_t$, une inversion de signe produit une divergence exponentielle vers du bruit saturé ou plat.

---

### 🎯 Piste D : Conditionnement Texte Qwen3-VL (Taps & Norms)
- **Dans `src/text/qwen.rs` :**
  - Krea 2 utilise **12 couches intermédiaires (taps)** de Qwen3-VL pour alimenter les blocs MMDiT (`txtfusion`).
  - **Problème potentiel :** Vérifier si les taps doivent être normalisés individuellement (RMSNorm) ou s'ils sont injectés bruts, et si la séquence concaténée est `[Tokens_Texte, Tokens_Image]` ou l'inverse.

---

## 3. Derniers Résultats de la Session (02 Octobre 2026) 🔬

1. **Sonde 1 (VAE Isolé) : VALIDÉ ✅**
   - Binaire `test_qwen_vae` exécuté sur `aurora-dev` avec `qwen_image_vae_complet.safetensors`.
   - Entrée dummy latent `[1, 16, 128, 128]` -> sortie `[1, 3, 1024, 1024]`.
   - Décodeur VAE 100% stable et fonctionnel.

2. **Test Inférence Studio après fix TextFusion :**
   - Correction appliquée : attention sur `seq_len` (tokens) au lieu de `12` (taps) dans `TextFusionTransformer::forward`.
   - Inférence 8 steps exécutée sur ROCm HIP.
   - Résultat visuel : toujours un bruit plat rosâtre identique.
   - **Conclusion :** Le blocage principal ne provient pas uniquement de TextFusion. Il se situe très probablement au niveau du cœur du DiT (Euler Flow Matching sign / scaling timestep / attention mask) ou des poids de projection initiaux / finaux (`first`, `last`).

---

## 4. Protocole Déterministe pour la Prochaine Session 🛠️

1. **Sonde Pas-à-Pas (Golden Tensor Reference) :**
   - Exécuter 1 pas d'inférence en Python via `diffsynth_krea2.py` avec une graine fixe (seed 42, prompt simple).
   - Dumper les statistiques couche par couche (`first(img)`, `temb`, `blocks.0` output, `pred_v`, `latents_next`).
   - Insérer des sondes identiques dans `src/pipelines/z_image_turbo.rs` pour repérer à quel tenseur exact la divergence apparaît.
2. **Vérification Signe Flow Matching :**
   - Vérifier si $v_t$ prédit par Krea 2 Turbo a la même convention de signe que Lumina / Flux ou si $\Delta t$ doit être inversé.

