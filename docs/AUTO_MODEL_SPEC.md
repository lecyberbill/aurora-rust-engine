# Aure-Spec — Port de l'API `transformers` (HuggingFace) pour Candle

**Date**: 2026-09-08 | **Repos**: `aurora-rust-engine` | **Statut**: ÉBAUCHE — à valider
**Gabarit**: `transformers` en PyTorch (AutoModel / AutoTokenizer / AutoProcessor / AutoConfig /
`from_pretrained`), transposé en Rust/Candle.

## 1. Motivation

Au lieu d'exposer des pipelines concrets (`FluxPipeline`, `Qwen3TextEncoder`…), on veut que le moteur
offre **l'API familière de `transformers`** : on donne un `repo_id` HF (ou un chemin), le moteur
télécharge, détecte la config, charge les poids et retourne un objet prêt à générer. Ainsi tout ce qui
peut parler à `transformers` (une app, un notebook, un backend) peut parler à notre moteur Candle.

## 2. Ce qui correspond (les concepts à reproduire)

| `transformers` (PyTorch) | Équivalent Candle (à bâtir) |
|---|---|
| `AutoConfig.from_pretrained(repo)` | `Config::detect(&dyn WeightsSource)` (*existe déjà* dans `QwenTextConfig::detect`) |
| `AutoModel.from_pretrained(repo, ...)` | `models::auto::AutoModel::from_pretrained(&Hub, repo, ...)` |
| `AutoTokenizer.from_pretrained(repo)` | `models::auto::AutoTokenizer::from_pretrained(...)` (drive `*_tokenizer.json`) |
| `AutoProcessor` (pour VLM / image) | `models::auto::AutoProcessor` (wrap VAE + tokenizer + resize) |
| `model.generate(...)` | `GenerationModel::generate(...)` (déjà esquissé trait) |
| `model(pixel_values=..., input_ids=...)` | `EncodeModel::forward/encode(...)` (text encoders existants) |
| `AutoModelForCausalLM` (LLM) | `models::text::TextModel` (Qwen/Mistral) — Milestone 16 |
| `DiffusionPipeline` | `models::image::DiffusionModel` (Flux/SDXL) — déjà quasi là |

## 3. Architecture proposée (module `src/models/`)

```
src/models/
├── mod.rs              # ré-export Auto*, traits communs
├── auto.rs             # AutoConfig / AutoModel / AutoTokenizer / AutoProcessor (le "front door")
├── registry.rs         # cache id -> Arc<Mutex<dyn AnyModel>> (+ swap/unload)
├── common.rs           # traits AnyModel / GenerationModel / EncodeModel (Send+Sync)
├── text/
│   ├── mod.rs          # TextModel (Qwen/Mistral) impl GenerationModel
│   └── wapper.rs
├── image/
│   ├── mod.rs          # DiffusionModel (Flux/SDXL) impl EncodeModel + GenerationModel
│   └── wapper.rs
└── config/             # cfg detection par famille (move QwenTextConfig::detect ici?)
```

### 3.1 `AutoModel` — le point d'entrée "front door"

```rust
pub enum ModelKind { Text, Image, Diffusion, Unknown }
pub struct ModelInfo {
    pub repo_id: String,
    pub kind: ModelKind,
    pub family: String,          // "qwen3", "mistral3", "flux2-dev", "sdxl", ...
    pub dtype: DType,
    pub checkpoint: PathBuf,     // chemin résolu (local)
    pub config_path: PathBuf,
}

pub struct AutoModel;
impl AutoModel {
    /// Repo HF ou chemin local. Télécharge via ModelHub, détecte config, charge, met en cache.
    pub async fn from_pretrained(
        hub: &ModelHub, repo: &str, device: Device, dtype: DType,
    ) -> Result<Arc<Mutex<dyn AnyModel>>>;

    pub async fn from_pretrained_with(
        hub: &ModelHub, repo: &str, device: Device, dtype: DType,
        display: impl DisplayProgress,        // stream de progression (WebSocket/CLI)
    ) -> Result<Arc<Mutex<dyn AnyModel>>>;
}
```

### 3.2 `AutoTokenizer` / `AutoProcessor`

```rust
pub struct AutoTokenizer(TokenizerKind);   // wraps the existing *tokenizer.json loaders
impl AutoTokenizer {
    pub fn from_pretrained(hub: &ModelHub, repo: &str) -> Result<Self>;
    pub fn encode(&self, text: &str, max_len: usize) -> Result<Tensor>;
    pub fn decode(&self, ids: &[u32]) -> Result<String>;
}

pub struct AutoProcessor {    // pour les modèles vision/langage
    tokenizer: AutoTokenizer,
    image_size: (usize, usize),
}
impl AutoProcessor {
    pub fn from_pretrained(hub: &ModelHub, repo: &str) -> Result<Self>;
    pub fn encode_image(&self, img: &RgbImage) -> Result<Tensor>;  // -> [1,C,H,W]
    pub fn encode_text(&self, text: &str, max_len: usize) -> Result<Tensor>;
}
```

### 3.3 Traits communs (`Send + Sync`, plateforme-ready)

```rust
pub trait AnyModel: Send + Sync {
    fn kind(&self) -> ModelKind;
    fn family(&self) -> &str;
    fn id(&self) -> &str;
    fn device(&self) -> Device;
    fn dtype(&self) -> DType;
}

pub trait GenerationModel: AnyModel {
    fn generate(&mut self, prompt: &str, max_tokens: usize, temperature: f64) -> Result<String>;
}

pub trait ImageGenerationModel: AnyModel {
    fn generate_t2i(&mut self, params: DiffusionParams, on_step: Option<...>) -> Result<RgbImage>;
    fn generate_img2img(&mut self, params: Img2ImgParams, on_step: Option<...>) -> Result<RgbImage>;
    fn generate_inpaint(&mut self, params: InpaintParams, on_step: Option<...>) -> Result<RgbImage>;
    fn load_lora(&mut self, path: &Path, multiplier: f64) -> Result<()>;
    fn unload_lora(&mut self, id: &str) -> Result<()>;
}
```

Les implémentations concrètes **wrappent** les objets existants :
- `TextModel` : wrap `Qwen3TextEncoder` / `Mistral3TextEncoder` (+ détection du bon texte par `family`).
- `DiffusionModel` : wrap `FluxPipeline` / `StableDiffusionXLPipeline`.

> **Important**: on NE réécrit pas les encodeurs/pipelines internes. On expose une façade `Auto*`
> au-dessus d'eux. C'est un port d'API, pas un rewrite du moteur.

### 3.4 `ModelRegistry` (cache + swap de modèles chargés)

```rust
pub struct ModelRegistry {
    models: RwLock<HashMap<String, Arc<Mutex<dyn AnyModel>>>>,
    hub: ModelHub,
    device: Device,
    dtype: DType,
}
impl ModelRegistry {
    pub async fn load(&self, repo: &str) -> Result<Arc<Mutex<dyn AnyModel>>>;   // from_pretrained + cache
    pub fn get(&self, id: &str) -> Option<Arc<Mutex<dyn AnyModel>>>;
    pub fn unload(&self, id: &str) -> bool;
    pub fn swAP(&self, id: &str, repo: &str) -> Result<Arc<Mutex<dyn AnyModel>>>;
    pub fn list(&self) -> Vec<String>;
}
```

## 4. Détection de l'architecture (le "Auto" porte sur ceci)

`AutoModel::from_pretrained` doit deviner le type à partir de `config.json` + sniff de clés :
1. Si `model_type` dans `config.json` → `"qwen3"`, `"mistral"`, `"flux"`, `"sdxl"`, `"sd15"` → familles directes.
2. Sinon sniff de clés (`guidance_in`, `double_stream_modulation`, compte de single-blocks) comme aujourd'hui.
3. → `ModelKind`: `Text` (a `lm_head`/`output` + weights), `Diffusion` (a `double_blocks.*`/`unet`), `Image`.

> On centralise ce qui est aujourd'hui éparpillé (`QwenTextConfig::detect`, `from_single_file_streaming`
> sniffing) dans `src/models/config/`, sans casser les appels existants (on garde les anciens detect).

## 5. Flux HF "repo" → modèle Candle (le "pourquoi c'est du vrai transformers")

```
repo_id (HF) ou chemin local
   │  ModelHub::resolve_hf (télécharge + cache, déjà existant)
   ▼
config.json + model-*.safetensors + tokenizer*.json
   │  AutoConfig::detect
   ▼
ModelKind / family / dtype
   │  constructeur de famille (from_archive / from_dir / from_single_file_streaming — existants)
   ▼
Arc<Mutex<dyn AnyModel>>  (TextModel | DiffusionModel)
   │  trait méthode
   ▼
String (LLM) ou RgbImage (diffusion)
```

## 6. Ce qui N'EST PAS le but (≠ mon ancienne SPÉC image)

- **`src/models/` n'est PAS un registry de diffusion.** C'est un **port d'API `transformers`**.
  La diffusion est juste un des `ModelKind` qu'on peut charger. La "multi-checkpoint" devient un détail
  du `ModelRegistry`, pas l'objet central.
- Queue / jobs / workflows / métadonnées ComfyUI : **toujours hors moteur** (plateforme).

## 7. Fichiers impactés (estimation — objectif zéro régression)

| Fichier | Action |
|---|---|
| `src/models/mod.rs`, `auto.rs`, `common.rs`, `registry.rs` | **Nouveau** : API Auto + traits + cache |
| `src/models/text/`, `src/models/image/` | **Nouveau** : wrappers `TextModel` / `DiffusionModel` |
| `src/models/config/` | **Nouveau (optionnel)** : centraliser la détection, garder les anciens |
| `src/lib.rs` | ajouter `pub mod models;` + re-exports `AutoModel`, `AutoTokenizer`, ... |
| `src/hub.rs` | prévoir `resolve_hf` pour un dossier complet (config + shards) — déjà `resolve_hf_repo` |

**Aucun changement** à `src/text/*`, `src/pipelines/*`, `src/diffusion/*` : on les wrappe.

## 8. Exemple d'usage (l'app qui pensait à `transformers`)

```rust
// usage minimal, familier :
let model = AutoModel::from_pretrained(&hub, "Qwen/Qwen3-8B", Device::Cuda(0), DType::F16).await?;
let text = model.lock().unwrap().generate("Write a haiku about snow", 128, 0.7)?;

// ou diffusion :
let flux = AutoModel::from_pretrained(&hub, "black-forest-labs/FLUX.2-dev", ...).await?;
let img = image_model.generate_t2i(DiffusionParams { prompt: "a fox".into(), ..Default::default() }, None)?;
```

## 9. Critères d'acceptation

- [ ] `from_pretrained("Qwen/Qwen3-8B")` → `TextModel` qui génère du texte (Milestone 16).
- [ ] `from_pretrained` sur un chemin local de checkpoint Flux réel → `DiffusionModel`.
- [ ] `AutoTokenizer::from_pretrained` + `encode/decode` round-trip.
- [ ] `build --release` sans warning ; `cargo test --lib` : 14/14 verts.
- [ ] Aucune modification de `src/text/*` ni `src/pipelines/*` (on wrappe).

## 10. État actuel (2026-09-11) — familles supportées

`AutoModel` détecte et construit aujourd'hui :

| Famille | Détection (`detect_architecture`) | Construction |
|---|---|---|
| Flux.1 (Dev/Schnell) | `double_blocks.*` + guidance/compte | `FluxPipeline` (T5 embarqué) |
| Flux.2 (Klein-4B/9B, Dev) | `double_stream_modulation` + comptes | `FluxPipeline` + `TextEncoderSpec::{Qwen3,Mistral3}` + VAE 32ch |
| SDXL | `conditioner.embedders` (avant les clés texte) | `StableDiffusionXLPipeline` (encodeurs embarqués) |
| SD1.5 | `model.diffusion_model.input_blocks.*` | `StableDiffusionPipeline` |
| **SD3.5 Large/Medium** | `joint_blocks.*` + compte (>30 = Large) | `FluxPipeline` + `TextEncoderSpec::Sd35` + VAE 16ch |

**SD3.5** est le seul modèle à charger **trois** encodeurs texte externes. On les expose via
`ModelDescriptor::sd35(id, checkpoint, clip_l, clip_g, t5, vae)` puis
`AutoModel::from_descriptor(&desc, device, dtype)`. `from_descriptor` attache CLIP-L, CLIP-G, T5-XXL
(T5 en **F32**, CLIP en F16, sur CPU) et le VAE 16-ch au `FluxPipeline`, qui détecte l'architecture
`Sd35Large`/`Sd35Medium` depuis le checkpoint. Les tokenizers (`clip_tokenizer.json`,
`openclip_tokenizer.json`, `t5xxl_tokenizer.json`) sont résolus relativement au cwd, comme pour
Qwen/Mistral.

Démo : `test_sd35_auto` (`CKPT`/`CLIP_L`/`CLIP_G`/`T5`/`VAE` surchargables). Validé :
`outputs/sd35_auto_model.png` (loup net) via la façade, sur SD3.5 Turbo et non-turbo.

Note : `AutoModel::from_local`/`from_pretrained` détectent bien SD3.5 et chargent le transformer, mais
**sans** encodeurs/VAE (ils n'ont pas les fichiers externes) ; la génération réclame alors la voie
`from_descriptor`. C'est le seul cas où le descripteur explicite est obligatoire.

### 10.1 Config JSON des modèles (pas de chemins en dur)

`ModelDescriptorFile` / `ModelDescriptorEntry` / `TextEncoderConfig` (serde) chargent une liste de
modèles depuis un `config.json`, pour ne plus câbler les chemins dans les binaires :

```json
{
  "models": [
    { "id": "sdxl", "label": "SDXL", "family": "sdxl",
      "checkpoint": "G:/models/checkpoints/sdxl.safetensors" },
    { "id": "sd35", "label": "SD 3.5 Large", "family": "sd35",
      "checkpoint": "G:/models/SD3/sd3.5_large.safetensors",
      "text_encoder": { "kind": "sd35",
        "clip_l": "…", "clip_g": "…", "t5": "…" },
      "vae": "G:/models/vae/sd3_vae.safetensors",
      "defaults": { "steps": 28, "guidance": 3.5, "width": 1024, "height": 1024,
                    "negative_prompt": "blurry, low quality" } }
  ]
}
```

- `family` : slug **optionnel** parmi `sdxl`/`sd15`/`sd35`/`flux1`/`flux2`/`flux2-klein-4b`/… (absent →
  sniff du checkpoint ; slug inconnu → erreur `Config`).
- `text_encoder.kind` ∈ `qwen3` | `mistral3` | `t5` | `sd35` (`sd35` porte `clip_l`/`clip_g`/`t5`).
- `defaults` : bloc **optionnel** de valeurs de génération par modèle (`ModelDefaults` : `steps`,
  `guidance`, `width`, `height`, `negative_prompt`, tous optionnels). Purement présentationnel côté
  moteur ; une UI l'applique à la bascule de modèle.
- `ModelDescriptorFile::load(path)` → `.descriptors()` → `Vec<(label, ModelDescriptor)>`, ou
  `.resolve()` → `Vec<ResolvedModel>` (garde l'`id`, le `label`, le descripteur **et** les `defaults`).

`aurora_studio` s'en sert via `aurora_studio.json` (surchargeable par l'env `STUDIO_CONFIG`), ne
pré-charge plus qu'un seul modèle, dérive le dropdown du config et applique les `defaults` à la
bascule (handler grio `on_change`). Tests : `parse_model_list_config`, `unknown_family_slug_is_error`,
`parse_model_defaults`, `defaults_are_optional`.


