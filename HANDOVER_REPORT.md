# Fiche de Synthèse & Validation — Z-Image Turbo (Aurora Rust Engine)
Date : 21 Septembre 2026  
Statut Global : ✅ **SUCCÈS COMPLET & VALIDÉ EN PRODUCTION (Jalon 17)**

---

## 1. 🎯 Objectif
Exécuter le modèle **Z-Image Turbo** (S3-DiT 6B, Qwen3-4B, VAE 16 canaux) en **100% pur Rust** sous CUDA (BF16), avec 4 pas d'inférence, pour produire des images nettes, photoréalistes et sans artefacts de bruit statique.

---

## 2. 🟢 Validations & Découvertes Clés

### Résolution du Bruit Statique (Les 2 causes racines) :
1. **Inversion de signe de la vélocité (`noise_pred = -pred_v`) :**
   - Dans le solveur Flow Match Euler discret, la convention de sortie de Lumina2 / Z-Image requiert l'inversion explicite du tenseur de vélocité avant l'étape de propagation d'Euler.
   - Sans cette négation, l'intégrateur ODE dérivait en sens inverse ($\Delta t < 0$), accentuant le bruit au lieu de débruiter.
   - Corrigé dans `src/pipelines/z_image_turbo.rs` via `pred_v.neg()?`.

2. **Ordre de concaténation de la séquence unifiée (`Image` puis `Texte`) :**
   - Alignement strict avec l'architecture officielle `diffusers` (`ZImageTransformer2DModel`) :
     `unified = [x_img, text_feat]` avec `x_img` à l'indice 0 et `text_feat` à la suite.
   - Alignement des coordonnées 3D RoPE : coordonnées spatiales de l'image en premier, coordonnées temporelles du texte en second.
   - Découpage de sortie du backbone : `x_seq.narrow(1, 0, n_img)?`.
   - Corrigé dans `src/diffusion/dit/z_image.rs`.

3. **Dynamic Shift Timestep Schedule :**
   - Intégration de la fonction linéaire de décalage dynamique :
     $\mu = m \times \text{seq\_len} + b$ avec $m = (1.15 - 0.5) / (4096 - 256)$ et $b = 0.5 - m \times 256$.
   - Pour 512x512 ($\text{seq\_len} = 1024$), $\mu \approx 0.630$.

---

## 3. 📊 Métriques & Télémétrie Confirmées

| Étape | Métrique Observée (512x512, 4 pas) | Statut |
| :--- | :--- | :--- |
| **Chargement Modèle AIO FP8** | 11.81 s | ✅ Stable |
| **Text Encoding (Qwen3-4B)** | 1.95 s (contexte `[1, 32, 2560]`) | ✅ Bit-exact |
| **Dénuisage DiT (4 steps)** | 221.13 s (55.2 s / step sur CUDA BF16) | ✅ Stable ($\text{std} \approx 0.7 - 0.9$) |
| **Décodage VAE (16 canaux)** | 12.68 s | ✅ Scaling Flux validé |
| **Rendu Image** | `output/zimage_turbo_test.png` | ✅ **Image nette, haute définition** |

---

## 4. 🚀 Commande de Reproduction Directe

```powershell
$env:CUDARC_CUDA_VERSION = "12080"
cargo run --release --features cuda --bin test_zimage_turbo
```
