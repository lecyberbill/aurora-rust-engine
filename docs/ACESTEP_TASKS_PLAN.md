# Plan — ACE-Step 1.5 : Cover / Repaint / Extract / Lego / Complete

**Date** : 2026-09-24 | **Statut** : ✅ Étapes 1-5 implémentées (cover / repaint / extract / lego / complete) — captions SFT-stems en attente du checkpoint `is_lego_sft`
**Contexte** : le chemin `text2music` (Turbo/Base/XL), le **codec 5 Hz (FSQ)** et le
**LM-codes → musique** sont livrés et validés ([`ACESTEP_AUDIO_SPEC.md`](ACESTEP_AUDIO_SPEC.md)).
Reste les tâches audio avancées.

---

## 0. Prérequis n°1 : le **VAE encodeur** (manquant)

On n'a aujourd'hui que le **décodeur** Oobleck (`src/audio/vae_oobleck.rs`). Or cover /
repaint / extract / lego / complete partent tous d'un **audio source** → il faut
`audio [2, N] → latents [64, T]`. Le checkpoint VAE contient l'encodeur :

```
encoder.conv1.weight_v          [128, 2, 7]        # stéréo 2 -> 128, k=7, pad=3
encoder.block.{0..4}.conv1.*    strides [2,4,4,6,10], k = 2*stride
encoder.block.{i}.res_unit{1,2,3}.*                 # identiques au décodeur (dilations 1,3,9)
encoder.snake1.*
encoder.conv2.weight_v          [128, 2048, 3]      # 128 = 2*64 → (mean, logvar)
```

C'est le **miroir exact** du décodeur :
- `OobleckEncoderBlock` = `snake1 → WeightNormConv1d(stride) → 3 res_units`
  (au lieu de `conv_t1` transposé).
- Sortie `conv2` → split `(mean, logvar)` → `DiagonalGaussianDistribution`.
- `encode_tiled` (overlap-discard) sur le modèle de `decode_tiled`.

**Décision à trancher** : la référence échantillonne `latent_dist.sample()` (stochastique).
Pour un résultat déterministe / iso → préférer **`.mean`** (recommandé), ou `sample` avec un
RNG seedé identique. Impacte légèrement la fidélité.

**Validation** : `scripts/dump_acestep_vae_enc_ref.py` (via `tiled_encode`/`AutoencoderOobleck.encode`
de diffusers) + bin `test_acestep_vae_enc_ref` → `latents max|diff|`.

---

## 1. Socle commun

| Élément | Fichier cible | Détail |
|---|---|---|
| **VAE encodeur** | `src/audio/vae_oobleck.rs` | `OobleckEncoderBlock`, `encode()`, `encode_tiled()` |
| **Paramètres de tâche** | `TextToMusicRequest` → `AudioTaskRequest` | `task_type`, `src_audio`, `repainting_start/end`, `track_name`, `complete_track_classes`, `reference_audio`, `audio_cover_strength`, `cover_noise_strength`, `chunk_mask_modes` |
| **Instructions** | `src/models/acestep_tasks.rs` | `TASK_INSTRUCTIONS`, `generate_instruction(task, track, classes)`, `_format_instruction` |
| **Conditioning** | `build_conditioning` / `forward_condition` | `src_latents` custom, `is_covers`, timbre depuis `reference_audio` |
| **Prompts** | `acestep_tasks.rs` | format SFT `Global:/Local:/Mask Control:` (`is_lego_sft`) |

Débloque tout le reste. **Gros morceau #1** (≈ la taille du décodeur).

---

## 2. Cover (`cover`, `cover-nofsq`)

`is_cover = (task=="cover") or has_code_hint` → `src_latents = hints / VAE-latents`.
- **cover** = `src = detokenize(tokenize(VAE.encode(src_audio)))` (bottleneck FSQ) — **codec déjà prêt**.
- **cover-nofsq** = `src = VAE.encode(src_audio)` (sans FSQ).

Étapes :
1. `reference_audio` → `refer_latents` → **timbre encoder** (au lieu de `silence_latent`).
2. `src_audio` → VAE encode → (optionnellement codec) → `src_latents`.
3. `audio_cover_strength` : `cover_steps = int(steps*strength)` ; conditionner sur `[src, hints]`
   pour les premiers pas puis **basculer** sur `encoder_non_cover`/`context_non_cover` (reset KV cache).
4. `cover_noise_strength` : `xt = renoise(src_latents, nearest_t, noise)` + troncature du schedule.

Fichiers : `audio_diffusion.rs`, `acestep.rs` (`forward_condition` timbre réel), `acestep_codec.rs` (fait).

---

## 3. Repaint (`repaint`)

Référence : `conditioning_masks` + `repaint_step_injection` :
1. `repainting_start/end` (s) → `[start_latent, end_latent)` (`rate = 48000/1920 = 25`).
2. `chunk_mask` = 1 dans la zone repaint, 0 ailleurs.
3. `src_latents` = VAE(target) avec la zone repaint **remplacée par `silence_latent`**.
4. `repaint_mask[b,t]` = True dans la zone à générer.
5. **Injection par étape** (sur `round(repaint_injection_ratio * steps)` premiers pas) :
   `zt_src = t_next*noise + (1-t_next)*clean_src ; xt = where(repaint_mask, xt, zt_src)`.
6. **Blend de bordure** final (soft mask + crossfade `repaint_crossfade_frames`) :
   `x = m*x_gen + (1-m)*clean_src` avec `clean_src = VAE(target)` **non silencié**.

Fichiers : `flow_match_euler*` (params `clean_src`, `repaint_mask`, `noise`, ratios), `audio_diffusion.rs`.

---

## 4. Extract / Lego / Complete (SFT uniquement)

> **État** : câblé via `TaskRequest` + `AudioDiffusionPipeline::generate_task`
> (`src/pipelines/audio_diffusion.rs`) et testé avec **base** (`G:/models/Audio-base`) et
> **SFT général** (`D:/models/Audio-sft`, converti depuis `ACE-Step/acestep-v15-sft`).
> Les 3 tâches sont **base-only** ; sorties = contenu nouveau (corr ≈ 0 avec la source),
> rms base 0.18-0.30 / SFT 0.37-0.52.

- **Checkpoint SFT** : général téléchargé + converti (`D:/models/Audio-sft`, archi base 2048/24).
  Le **SFT-stems** (flag `is_lego_sft` dans la config) n'est **pas public** (aucun repo HF ne le porte)
  → la caption `Global:/Local:/Mask Control:` est implémentée mais inactive (dégradation douce sans elle).
- `is_lego_sft` (config) → caption spéciale : `lego_sft_caption()` (`acestep_tasks.rs`).
- Instructions : `generate_instruction(task, track, classes)` (fait).
- `generate_task` : `chunk_mask` + `src_latents` (lego garde la source dans la zone ; repaint la silence),
  `repaint_mask` pour repaint/lego, sampler `flow_match` (CFG/APG sur base, no-CFG sur turbo).
- Bins : `test_acestep_tasks.rs` (`-t extract|lego|complete|repaint|cover --track --classes --repaint-start/end --chunk <v>`).
- ⚠️ **`chunk_mask` = 2.0** (« auto »/Mask Control) pour extract/lego/complete → **musique** ;
  `1.0` → rumble basse fréquence (centroïde ~150 Hz). Repaint reste en 0/1 explicite.
  Défaut `TaskRequest.chunk_mask_value = NaN` → résolu par tâche (`cover` 1.0, le reste 2.0).
- ✅ **Validé à l'oreille** (source 30 s) : `extract`/`lego`/`complete` = vraie musique avec
  **SFT 2B** (`D:/models/Audio-sft`), `base 2B` et `xl-base` (extract OK). Source courte/artefact
  (`codec_original` 6 s) → distordu ; utiliser de vraies chansons.
- ⚠️ **VRAM** : `xl-turbo`/`xl-base` 5B (~10 Go bf16) limite sur RTX 4070 Ti 12 Go → peut pendre/OOM.
  Préférer les modèles 2B (`acestep-v15-base`, `acestep-v15-sft`) pour ces tâches.
- ⚠️ **Build CUDA obligatoire** : `cargo build --release --features cuda` (sinon `Device::new_cuda`
  échoue silencieusement → CPU F32). Sur GPU : ~3 s (6 s) / ~16 s (30 s) par tâche.

Fichiers : `acestep_tasks.rs`, `build_conditioning`, sampler (masque comme repaint).

---

## 5. Sampler / boucle (impact transverse)

Introduire une structure d'état plutôt que des paramètres positionnels :
`SamplerState { clean_src: Option<Tensor>, repaint_mask: Option<Tensor>, repaint_ratio: f32,
cover_strength: f32, context_non_cover: Option<Tensor>, condition_non_cover: Option<Tensor>,
cover_noise_strength: f32 }`.
Modifie `flow_match_euler` et `flow_match_euler_cfg`.

---

## 6. Validation (même méthode)

| Étape | Référence Py | Métrique |
|---|---|---|
| VAE encode | `tiled_encode` (diffusers Oobleck) | latents max\|diff\| |
| cover | `generate_music(task="cover", src_audio=…)` dumpé | condition + latents + écoute |
| repaint | idem + `repainting_start/end` | latents + écoute (zone modifiée) |
| extract/lego/complete | idem (SFT) | latents + écoute |

Harnais : `scripts/dump_acestep_<task>_ref.py` + `src/bin/test_acestep_<task>.rs` (pattern existant).

---

## 7. Ordre recommandé

```
1. VAE encodeur (+ validation iso)          ← prerequis
2. Plomberie tâche + prompts (acestep_tasks) + timbre réel
3. Cover (profite du codec déjà fait)
4. Repaint (sampler injection/blend)
5. Extract/Lego/Complete (SFT + caption lego)
6. Docs + commit
```

---

## 8. Risques

- **`.sample()` vs `.mean`** du VAE encode (déterminisme vs fidélité).
- **SFT** introuvable/à télécharger pour extract/lego/complete.
- `Mask Control` / `is_lego_sft` : logique de masque à répliquer finement.
- Samples longs → **tiling encode** requis (comme le décode).

---

## 9. Checklist de reprise

- [x] Porter `OobleckEncoderBlock` + `encode()` + `encode_dist()` + `encode_tiled()` (`src/audio/vae_oobleck.rs`).
      ⚠️ Ordre encodeur = `res_units(input) → snake → conv1(strided)` (inverse du décodeur) ;
      `conv2` sort `2*C` = `(mean, scale)`, `std = softplus(scale)+1e-4` (pas logvar).
- [x] Harnais de validation VAE encode (`dump_acestep_vae_enc_ref.py` + `test_acestep_vae_enc_ref`) :
      raw **4.8e-5**, mean **2.0e-5**, std **8.6e-8**, tiled==mono (overlap 16 frames), round-trip OK.
- [x] Struct `TaskRequest` + plumbing `task_type`/params (`generate_task`).
- [x] `src/models/acestep_tasks.rs` (instructions + `AceStepTask` + caption SFT `lego_sft_caption`).
- [x] Timbre depuis `reference_audio` (`forward_condition_ex`, défaut = silence).
- [x] Cover : strength switch + cover-noise init.
- [x] Repaint : mask + injection par étape + blend final (**bit-exact 7.9e-6** vs réf rejouée).
- [x] SFT général converti ; **extract/lego/complete** câblés + testés (base & SFT).
- [x] `chunk_mask=2.0` (auto) requis pour extract/lego/complete — validé à l'oreille (base/XL-base/SFT).
- [x] Modèle par défaut : **2B** (`acestep-v15-base` / `acestep-v15-sft`) — XL 5B limite en 12 Go.
- [ ] Docs + commit.

## Fichiers concernés

```
src/audio/vae_oobleck.rs            + encodeur Oobleck (miroir du décodeur)
src/models/acestep_tasks.rs         (nouveau) instructions + formattage caption
src/models/acestep.rs               forward_condition (timbre réel), éventuels masques
src/pipelines/audio_diffusion.rs    AudioTaskRequest + SamplerState
src/models/acestep_codec.rs         (déjà fait)
scripts/convert_acestep_base.py     (déjà fait, réutilisable pour SFT)
scripts/dump_acestep_*_ref.py       (nouveaux) références PyTorch
src/bin/test_acestep_*_ref.rs       (nouveaux) harnais de validation
```
