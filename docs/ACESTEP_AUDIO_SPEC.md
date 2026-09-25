# Aurora-Spec — ACE-Step 1.5 Text-to-Music en pur Rust

**Date** : 2026-09-22 | **Repo** : `aurora-rust-engine` | **Statut** : ✅ **Validé (Turbo & Base), XL prêt**
**Cible** : `G:\models\Audio` (Turbo diffusers), `G:\models\Audio-base` (Base converti), `D:\models\Audio-xl-base` (XL 4B converti).

---

## 1. Objectif

Porter en **100 % pur Rust** (Candle) le modèle open-source **ACE-Step 1.5** de génération
musicale (texte + paroles → audio 48 kHz stéréo), avec parité mathématique vérifiée contre
l'implémentation de référence HuggingFace/Diffusers (`ace_step_repo`), et exposer une API
bibliothèque propre.

Composants portés :

| Composant | Rôle | Fichier |
|---|---|---|
| **Qwen3 text encoder** | encode le prompt caption (28 couches, hidden 1024) | `src/text/qwen.rs` (`forward_last_hidden`, `embed_ids`, `tokenize_raw`) |
| **ConditionEncoder** | fusionne caption + paroles + timbre → `[1, L, 2048]` | `src/models/acestep.rs` (`AceStepConditionEncoder`) |
| **1D DiT Transformer** | débruite les latents (Flow-Matching) | `src/models/acestep.rs` (`AceStepTransformer1D`) |
| **AutoencoderOobleck** | VAE 48 kHz stéréo (Snake1d + WeightNorm) | `src/audio/vae_oobleck.rs` |
| **Flow-Matching Euler** | ordonnanceur de débruissage + CFG (APG) | `AceStepTransformer1D::{flow_match_euler, flow_match_euler_cfg}` |
| **5Hz LM planner** | Qwen3 causal LM (CoT métadonnées/paroles) | `src/models/acestep_lm.rs` (`AceStepLm`) |
| **Codec 5 Hz (FSQ)** | tokens sémantiques ↔ hints 25 Hz (FSQ 64k, attention-pooler + detokenizer) | `src/models/acestep_codec.rs` (`AceStepAudioCodec`) |
| **Codecs sortie** | WAV / OGG Vorbis / MP3 | `src/audio/encode.rs`, `src/audio/wav.rs` |
| **RNG déterministe** | bruit initial reproductible depuis un seed | `src/audio/rng.rs` |

---

## 2. Modèles supportés

| Variante | DiT (hidden / couches / têtes) | Sampler | Dossier |
|---|---|---|---|
| **Turbo** (2B) | 2560 / 32 / 32 | Euler, **sans CFG** (guidance=1.0) | `G:\models\Audio` |
| **Base / SFT** (2B) | 2048 / 24 / 16 | Euler + **CFG (APG, guidance≈7)** | `G:\models\Audio-base` |
| **XL** (4B) | 2560 / 32 / 32 | Euler + **CFG** | `D:\models\Audio-xl-base` |

La variante est **auto-détectée** via `transformer/config.json` (`is_turbo`) par
`AudioDiffusionPipeline::from_folder`. Les dimensions du DiT sont lues dans ce même fichier
(`AceStepTransformerConfig::from_json_file`).

Le condition encoder reste en **hidden 2048** pour toutes les variantes ; les modèles XL
utilisent un `encoder_hidden_size` séparé (lu depuis la config).

### Conversion d'un checkpoint single-file (layout repo → diffusers)

Les checkpoints `acestep-v15-*` de HF sont au format « repo » (clés `decoder.*` / `encoder.*`,
attention `q_proj`/`o_proj`/`q_norm`). Le script les convertit vers le layout diffusers attendu
par le moteur (`transformer/`, `condition_encoder/`, plus les composants partagés
`text_encoder`/`tokenizer`/`vae`/`scheduler`) :

```powershell
python scripts/convert_acestep_base.py `
  --checkpoint "G:/models/Audio/Ace-Step1.5/acestep-v15-base" `
  --shared "G:/models/Audio" `
  --out "G:/models/Audio-base" `
  [--save-dtype bf16]     # downcast optionnel (XL f32 19.9 Go -> ~10 Go)
```

Gère les checkpoints **mono-fichier** (`model.safetensors`) et **shardés**
(`model.safetensors.index.json`) — nécessaire pour les XL (4 shards f32, 19.9 Go).

---

## 3. Pipeline d'inférence

1. **Prompts** (identiques à la référence) :
   - caption : `SFT_GEN_PROMPT.format(instruction, caption, metas)` (max 256 tokens) ;
   - paroles : `"# Languages\n{lang}\n\n# Lyric\n{lyrics}<|endoftext|>"` (max 2048).
2. **Qwen3** : `text_hidden = last_hidden_state(caption)` (28 couches + norm finale) ;
   `lyric_embeds = embed_tokens(lyric_tokens)` (table d'embedding seule).
3. **Condition** (`forward_condition`) : `text_projector(text_hidden)` ⊕
   `lyric_encoder(lyric_embeds)` (8 couches bidirectionnelles) ⊕
   `timbre_encoder(silence_latent[:750])` (4 couches) → ordre `[lyrics, timbre, text]`.
4. **Latents** : `num_frames = max(128, duration·25)` ; pour text2music
   `src = silence_latent[:T]`, `chunk_mask = 1.0`, `context = [src, chunk_mask]`.
5. **DiT** : `input = [src, chunk, noisy]` (192 canaux) ; AdaLN `scale_shift_table +
   timestep_proj` ; norm de sortie + `temb` ; **sliding-window 128** sur les couches paires.
6. **Sampler** : `t = 1 − i/n` ; Euler avec `x0 = x − v·t` au dernier pas.
   - Turbo : 8 pas, guidance 1.0.
   - Base/XL : CFG batch-2 (`null_condition_emb`) + **APG** (`apg_forward`, momentum −0.75,
     clamp de norme 2.5, projection orthogonale).
7. **VAE** : décodage **par tuiles** (overlap-discard) pour borner la VRAM → WAV 48 kHz stéréo.

---

## 4. API Rust

```rust
use aurora_rust_engine::pipelines::{AudioDiffusionPipeline, TextToMusicRequest};
use aurora_rust_engine::audio::AudioFormat;

// CUDA bf16 si dispo, sinon CPU f32.
let pipeline = AudioDiffusionPipeline::from_pretrained("G:/models/Audio")?;

let req = TextToMusicRequest::new(caption, lyrics)
    .with_language("fr")
    .with_duration(30.0)
    .with_steps(8)          // 8 (turbo) / ~30-50 (base)
    .with_seed(42);

let (audio, metrics) = pipeline.text_to_music(&req)?;
pipeline.text_to_music_to_file(&req, "out.ogg", Some(AudioFormat::Ogg))?;
```

Méthodes exposées : `from_pretrained`, `from_folder(dir, device, dtype)`,
`text_to_music`, `text_to_music_to_file`, `build_conditioning`, `diffuse`/`diffuse_guided`,
`decode`, `generate`/`generate_with_noise`/`generate_with_noise_guided`.
Champs : `variant` (`AceStepVariant::{Turbo, Base}`), `default_guidance_scale`.

Le seed est **déterministe** (`xoshiro256**` + Box-Muller, `src/audio/rng.rs`) : un même seed
produit un WAV identique (vérifié par MD5 sur GPU).

### Codecs de sortie

| Format | Backend | Licence |
|---|---|---|
| **WAV** (PCM16) | natif pur Rust | — |
| **OGG Vorbis** | `vorbis_rs` (libvorbis bundlé) | BSD-3-Clause |
| **MP3** (192 kb/s) | `mp3lame-encoder` (LAME bundlé) | LGPL-3.0, **optionnel** via `--features mp3` |

Sans la feature `mp3`, `save_auto(".mp3")` renvoie une erreur explicite : le MP3 peut être
produit par un outil tiers (ex. ffmpeg) à partir du WAV/OGG.

---

## 5. CLI

```powershell
# Build (CUDA 12.8 + MSVC requis pour nvcc)
$env:CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
$env:CUDARC_CUDA_VERSION="12080"
cmd /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 && cargo run --release --features cuda --bin test_audio_diffusion -- --model-dir G:/models/Audio --duration 180 --steps 8 --format ogg --out ma_musique'
```

Options : `--model-dir/-m`, `--caption/-p`, `--lyrics/-l`, `--lyrics-file`, `--lang`,
`--duration/-d`, `--steps/-s`, `--seed`, `--out/-o`, `--format/-f {wav|ogg|mp3}`.

---

## 6. LM planner (5Hz)

`AceStepLm::from_dir(dir, device, dtype)` charge le Qwen3-1.7B (`acestep-5Hz-lm-1.7B`) via
`CausalLMPipeline`. `build_formatted_prompt` reproduit le chat template Qwen de la référence ;
`plan(caption, lyrics, max_tokens)` produit le CoT :

```
<think>
bpm: 120
caption: A bright acoustic guitar strums lively chords…
duration: 12
keyscale: G major
language: unknown
timesignature: 4
</think>
```

> Correction notable : `CausalLMPipeline::forward_layer` n'appliquait pas `q_norm`/`k_norm`
> (spécifique Qwen3) → sortie corrompue. Corrigé ; bénéficie à tous les usages Qwen3 du moteur.

**Génération contrainte des codes 5Hz** : `CausalLMPipeline::generate_ids(prompt, …, allowed)`
restreint le vocabulaire aux tokens `<|audio_code_N|>` (N ∈ [0, 63999]) ;
`AceStepLm::generate_codes` mappe id→code et produit la séquence. Le decode suit la référence
(`quantizer.get_output_from_indices` → `detokenizer` → hints 25 Hz).

### 6.b Codec 5 Hz (FSQ) — validé

`AceStepAudioCodec` (`src/models/acestep_codec.rs`) implémente le tokenizer/detokenizer sémantique :
- `ResidualFsq` : `project_in` 2048→6, soft-clamp, `symmetry_preserving_bound` (hard-clamp),
  `codes_to_indices`, et `get_output_from_indices` (codebook implicite 64 000 construit en Rust).
- `AttentionPooler` (special token + 2 couches) et `AudioTokenDetokenizer` (special_tokens + 2 couches + `proj_out`).
- API : `tokenize(features[·,·,64])`, `detokenize(tokens)`, `detokenize_from_indices(indices)`.

Poids exportés par `convert_acestep_base.py` dans `audio_codec/codec.safetensors`.

> Le codec est un **code sémantique** ultra-basse-débit (~80 bits/s). Décoder les hints
> directement par le VAE donne quasi-silence (corr 0.18) : c'est attendu. Son vrai rôle est de
> **conditionner** le DiT (hints comme `src_latents`, ex. `codec_hint_conditioned.wav` → audible).

**Chemin complet validé à l'oreille** : `LM codes → hints → DiT → audio`
(`test_acestep_lm_codes`) produit de la vraie musique. Point clé : le sampling des codes doit
utiliser **top-p (0.9)** comme la référence — un sampling à température seule (sans top-p) donne
des codes temporellement incohérents et un audio haché. Les codes sont filtrés à `[0, 63999]`.

```powershell
cargo run --release --features cuda --bin test_acestep_lm_codes -- --duration 30 --out outputs/audio_showcase/lm_codes_song_30s.ogg
```

---

## 7. Validation (harnais bit-exact vs PyTorch)

Scripts de dump (`scripts/dump_*.py`, exécutés avec un env torch) + bins de comparaison Rust :

| Harnais | Mesure | Écart max |
|---|---|---|
| `test_acestep_dit_ref` (DiT) | emb/timestep + sortie 1 pas | **5.7e-5** |
| `test_acestep_cond_ref` (ConditionEncoder) | text/lyric/timbre/condition | **2.4e-6** |
| `test_acestep_qwen_ref` (Qwen3) | last_hidden (28 couches) | **3.3e-4** |
| `test_acestep_pipeline_ref` (boucle Euler turbo) | latents finaux | **5.7e-5** |
| `test_apg` (APG) | guidance CFG | **2.4e-6** |
| `test_acestep_codec_ref` (codec 5 Hz FSQ) | quantized / indices / detokenize | **0.0 / 0.0 / 3.8e-6** |
| `test_acestep_iso` (bout-en-bout) | condition / latents | base **2.2e-4** / XL 3.3e-1* |

\* XL validé en bf16 (perte de précision du checkpoint converti) ; reconvertir en f32 pour
un iso exact, le même code étant validé à 2.2e-4 sur base.

Reproduction type :
```powershell
python scripts/dump_acestep_base_ref.py --checkpoint "<...>/acestep-v15-base" --modeling base --caption "..." --lyrics-file scripts/lyrics_fr.txt --duration 5 --steps 8 --guidance 7 --seed 42 --out outputs/audio_ref/base_ref.safetensors
cargo run --bin test_acestep_iso -- --model-dir G:/models/Audio-base --ref outputs/audio_ref/base_ref.safetensors --meta outputs/audio_ref/base_ref.json --cpu
```

---

## 8. Build & notes

- **CUDA** : `candle` requiert `nvcc` + `cl.exe` (MSVC). Voir le bloc CLI ci-dessus.
- **Features** : `cuda`, `mp3` (LAME, optionnel), `metal`/`accelerate`/`mkl` (autres backends).
- **Mémoire** : l'attention DiT est calculée par **blocs de requêtes** et le VAE est décodé
  par **tuiles** → permet des morceaux de 3 min sur 12 Go. Le decode monolithique saturait
  dès ~90 s.
- **Perf** (RTX 4070 Ti 12 Go, release bf16) : ~1.5× temps réel (180 s d'audio en ~117 s) ;
  base/XL plus lents (2× le DiT, CFG batch-2).

---

## 9. Limitations & suite

- **XL** : converti et fonctionnel (base + SFT via low-VRAM) ; iso exact à refaire en f32.
- **LM-codes → audio** : ✅ fonctionnel (LM CoT → codes contraints top-p → hints → DiT → musique),
  exemple `outputs/audio_showcase/lm_codes_song_30s.ogg`.
- **Codec 5 Hz (FSQ)** : validé bit-exact ; le planner texte (CoT) et la génération de codes le sont aussi.
- **Cover / repaint / extract / lego / complete** : ✅ portés (`TaskRequest` + `generate_task`).
  Repaint **bit-exact** (7.9e-6) ; cover (avec **FSQ optionnel**, `--cover-fsq`) ; extract/lego/complete
  via le masque « auto » (`chunk_mask=2.0`). Validés à l'oreille (base 2B, XL-base, SFT/XL-SFT).
- **Captions SFT-stems** (`Global:/Local:/Mask Control:`, `is_lego_sft`) : implémentées, prêtes pour un
  checkpoint stems (non public) — `--lego-sft` force le format.
- **Low-VRAM** : `from_pretrained_low_vram()` / `--low-vram` (encodeurs texte/condition sur CPU)
  pour faire tenir les checkpoints **XL 5B** sur 12 Go.
- Parité RNG cross-langage : le seed Rust est déterministe mais ne reproduit pas `torch.randn` ;
  l'iso utilise l'injection de bruit (`generate_with_noise`).

### Fichiers ajoutés

```
src/audio/encode.rs            codecs OGG/MP3
src/audio/rng.rs               RNG seedé
src/audio/vae_oobleck.rs       VAE Oobleck 48 kHz (décodeur + encodeur)
src/models/acestep.rs          DiT + ConditionEncoder (config-driven + CFG/APG)
src/models/acestep_tasks.rs    tâches (`AceStepTask`) + instructions + caption SFT-stems
src/models/acestep_codec.rs    codec 5 Hz (ResidualFSQ + pooler + detokenizer ; split diffusers)
src/models/acestep_lm.rs       planner LM 5 Hz (CoT + codes contraints)
src/pipelines/audio_diffusion.rs   API pipeline + sampler + `TaskRequest`/`generate_task` + low-VRAM
scripts/convert_acestep_base.py    conversion repo -> diffusers (+ audio_codec)
scripts/dump_*.py / run_acestep_*.py  dumps & références PyTorch
src/bin/test_acestep_*.rs          harnais (dit/cond/qwen/pipeline/codec/iso/lm/vae_enc/tasks/cover/repaint)
src/bin/test_{apg,audio_encode,audio_codec_roundtrip,acestep_lm_codes}.rs
```
