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
  ```rust
  let scaled_latents = ((latents / vae_scaling_factor) + vae_shift)?;
  ```
- **Problème potentiel :** Les VAE Qwen / Krea / Wan2.1 utilisent des conventions d'échelle latente spécifiques. 
  - SDXL : `latents / 0.13025`
  - Flux : `latents / 0.3611`
  - Qwen-Image / Wan2.1 : `latents * scale + shift` où `shift` et `scale` sont des vecteurs de taille `[1, 16, 1, 1]`.
- **Action :** Vérifier dans `diffsynth/models/qwen_image_vae.py` ou le `config.json` du VAE la constante exacte appliquée aux latents avant décodage.

---

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

## 3. Protocole Déterministe pour la Reprise 🛠️

Lors de la prochaine session, suivre cette méthodologie étape par étape sans régresser :

1. **Test Isolé du VAE (Sonde 1) :**
   - Créer un binaire de test `cargo run --bin test_vae_decode` qui décode un latent standard ou un tenseur connu.
   - Valider que le décodeur VAE produit une image cohérente et identifier le facteur de scaling exact.
2. **Comparaison Pas à Pas avec DiffSynth (Sonde 2) :**
   - Extraire du fichier local `diffsynth_krea2.py` les statistiques (mean, std, min, max) du tenseur $x_0, x_1$ au step 0 et step 1.
   - Comparer avec les logs de TransRust pour repérer immédiatement à quelle étape l'écart survient (Patchify, RoPE, Attention ou Euler update).
3. **Validation & Lancement UI :**
   - Recompiler avec `cargo build --release --bin aurora_studio --features rocm,ui`.
   - Tester la génération finale depuis l'interface web (7860).
