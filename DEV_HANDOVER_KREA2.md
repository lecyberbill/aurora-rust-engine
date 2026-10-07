# 📋 Consignes de Reprise — Krea 2 Turbo (Aurora Engine)

**Date :** 7 Octobre 2026  
**Statut actuel :** Analyse de conformité réalisée entre [`sp_cifications_techniques_et_impl_mentation_krea_2_turbo.md`](file:///d:/image_to_text/TransRust/sp_cifications_techniques_et_impl_mentation_krea_2_turbo.md) et l'implémentation Rust dans [`TransRust`](file:///d:/image_to_text/TransRust).

---

## 1. Synthèse de l'état des lieux

L'implémentation fondamentale est **à 95% alignée** sur la spécification officielle :
- ✅ **Qwen3-VL 4B** : 12 couches extraites (taps `[2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]`), troncature des 34 tokens du système prompt ([`qwen.rs`](file:///d:/image_to_text/TransRust/src/text/qwen.rs)).
- ✅ **SingleStream DiT & RoPE 3D** : Axes `[32, 48, 48]`, rotation interleaved `[-u1, u0]`, gating appliqué strictement avant `wo` ([`z_image.rs`](file:///d:/image_to_text/TransRust/src/diffusion/dit/z_image.rs)).
- ✅ **AutoencoderKLQwenImage 3D Causal VAE** : Décodage par tranches 2D et normalisation latente 16 canaux ([`vae_qwen.rs`](file:///d:/image_to_text/TransRust/src/diffusion/vae_qwen.rs)).

---

## 2. Les 3 points critiques à finaliser demain

### 🔹 Point 1 : Concordance de la Modulation Temporelle (`DoubleSharedModulation`)
* **Problème :** Deux conventions existent dans les checkpoints Safetensors :
  1. **Convention AIO / ComfyUI :** `tproj.1` partagé ($3840 \rightarrow 23040$) + biais résiduel broadcasté `blocks.{i}.mod.lin`.
  2. **Convention ai-toolkit / DiffSynth :** Chaque bloc possède sa projection linéaire locale `blocks.{i}.modulation.lin.weight` ($23040 \times 3840$).
* **Action demain :** Valider via [`inspect_krea_header.rs`](file:///d:/image_to_text/TransRust/src/bin/inspect_krea_header.rs) la clé exacte présente dans le checkpoint utilisé et s'assurer que le forward dans [`SingleStreamBlock`](file:///d:/image_to_text/TransRust/src/diffusion/dit/z_image.rs#L335-L379) applique la formule exacte :
  $$\mathbf{c} = \mathbf{W}_{\text{mod}}(\text{SiLU}(\mathbf{t}_{\text{emb}}))$$

### 🔹 Point 2 : TextFusion & TextMLP Tensor Permutation
* **Vérification :** S'assurer que `txtfusion.projector` opère bien sur la dimension des 12 couches (`Linear(12 -> 1)`) avec la permutation `[B, seq_len, 2560, 12] -> [B, seq_len, 2560]`.
* **Vérification TextMLP :** Correspondance exacte entre `RMSNorm -> Linear -> SiLU -> Linear` et les clés du checkpoint (`txtmlp.0`, `txtmlp.1`, `txtmlp.3` ou `txtmlp.fc1`, `txtmlp.fc2`).

### 🔹 Point 3 : Ordonnanceur FlowMatch Euler & Paramètre Shift $\mu$
* **Paramètres officiels Krea 2 Turbo :**
  - Shift fixe : $\mu = 1.15 \implies \exp(\mu) = 3.15819$ (au lieu de 3.0).
  - Facteur d'échelle de temps : $t \times 1000$ dans `krea_timestep_embedding`.
  - Entrée modèle au pas $i$ : $t_{\text{norm}} = 1.0 - \sigma$ ou $\sigma$ selon la convention de vitesse velocity $\mathbf{v}$.

---

## 3. Plan d'attaque recommandé pour demain

1. **Lancer un mini-test unitaire isolé (`cargo run --bin probe_krea2`) :**
   - Vérifier la propagation pas-à-pas avec $\mu = 1.15$.
   - Inspecter si la variance `pred_v` reste stable ($\approx 1.0$) et ne diverge pas au fur et à mesure des 8 étapes.
2. **Tester la sortie VAE immédiate :**
   - Comparer l'image décodée à l'étape 8 (`outputs/probe_step8.png`) avec l'estimation directe $\mathbf{x}_0 = \mathbf{x}_t - \sigma \mathbf{v}$.
3. **Connecter le pipeline complet :**
   - Exécuter la génération complète via [`z_image_turbo.rs`](file:///d:/image_to_text/TransRust/src/pipelines/z_image_turbo.rs).
