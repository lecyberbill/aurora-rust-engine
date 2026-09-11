// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Aurora Studio — a 100% pure Rust diffusion showcase UI powered by grio + aurora-rust-engine

//! Aurora Studio : vitrine d'intégration.
//!
//! Moteur `aurora-rust-engine` (façade `ModelRegistry`/`AutoModel`/`ModelDescriptor`) + UI `grio`
//! (pure Rust, équivalent Gradio, **lecture seule**). Multi-modèles avec **éjection stricte** :
//! un seul modèle est résident en VRAM ; basculer de modèle décharge l'ancien.
//!
//! Modèles vitrine : SDXL (standalone), Flux.1 (self-contained), Flux.2-Klein-4B (Qwen3 + VAE 32ch).
//! Flux.2-Dev / Klein-9B restent supportés par `AutoModel` mais ne sont pas dans le menu (trop lourds
//! pour une simple démo — on peut les ajouter à la demande en déchargeant le modèle actif).
//!
//! ```bash
//! cargo run --release --bin aurora_studio --features cuda,flash-attn,ui
//! ```

#[cfg(feature = "ui")]
mod app {
    use std::io::Cursor;
    use std::sync::{Arc, Mutex};

    use base64::Engine;
    use base64::engine::general_purpose::STANDARD as BASE64;

    use candle_core::{DType, Device, Tensor};
    use grio::*;

    use aurora_rust_engine::models::{
        downcast_model, AnyModel, AutoModel, DiffusionModel, ImageGenerationModel, ModelDefaults,
        ModelDescriptor, ModelDescriptorFile,
    };
    use aurora_rust_engine::traits::DiffusionParams;
    use aurora_rust_engine::FastLatentPreviewer;

    /// Un modèle déclaré dans la vitrine (id + descripteur pour le chargement + défauts de génération).
    #[derive(Clone)]
    struct ModelChoice {
        id: String,
        label: String,
        desc: ModelDescriptor,
        defaults: ModelDefaults,
    }

    /// État partagé (Send + Sync) : le modèle actif (éjection stricte) + device/dtype.
    struct StudioState {
        device: Device,
        dtype: DType,
        /// Le modèle résident en VRAM, ou `None` si rien n'est chargé. Un seul à la fois.
        active: Mutex<Option<Arc<Mutex<dyn AnyModel>>>>,
        /// Nom du modèle actif (pour la carte).
        active_id: Mutex<Option<String>>,
        /// Verrou anti-recharge concurrente (évite les chargements en cascade quand l'UI envoie
        /// plusieurs clicks rapides, source d'OOM).
        loading: Mutex<bool>,
    }

    impl StudioState {
        fn new(device: Device, dtype: DType) -> Self {
            Self { device, dtype, active: Mutex::new(None), active_id: Mutex::new(None), loading: Mutex::new(false) }
        }

        /// Charge `id` (parmi `choices`) et **décharge** le modèle précédent : éjection stricte.
        /// Retourne `None` si déjà actif, `Some(loaded)` si un chargement a eu lieu.
        fn switch(&self, choices: &[ModelChoice], id: &str) -> aurora_rust_engine::Result<bool> {
            // Ignore si déjà actif OU si un chargement est déjà en cours.
            if *self.loading.lock().unwrap() {
                return Ok(false);
            }
            if let Some(active_id) = self.active_id.lock().unwrap().clone() {
                if active_id == id {
                    return Ok(false); // déjà actif
                }
            }
            let choice = choices.iter().find(|c| c.id == id)
                .ok_or_else(|| aurora_rust_engine::LuminaError::Model(format!("modèle '{id}' inconnu")))?;

            *self.loading.lock().unwrap() = true;
            let result = (|| -> aurora_rust_engine::Result<bool> {
                // Décharge l'actuel (drop l'Arc -> VRAM libérée) puis charge le nouveau.
                *self.active.lock().unwrap() = None;
                let model = AutoModel::from_descriptor(&choice.desc, self.device.clone(), self.dtype)?;
                *self.active.lock().unwrap() = Some(model);
                *self.active_id.lock().unwrap() = Some(id.to_string());
                Ok(true)
            })();
            *self.loading.lock().unwrap() = false;
            result
        }

        fn current(&self) -> aurora_rust_engine::Result<Arc<Mutex<dyn AnyModel>>> {
            self.active.lock().unwrap().clone()
                .ok_or_else(|| aurora_rust_engine::LuminaError::Model("aucun modèle chargé".into()))
        }
    }

    fn b64_data_url(img: &image::RgbImage, format: image::ImageFormat, mime: &str) -> String {
        let mut buf = Vec::new();
        let mut cur = Cursor::new(&mut buf);
        if img.write_to(&mut cur, format).is_ok() {
            format!("data:{};base64,{}", mime, BASE64.encode(&buf))
        } else {
            String::new()
        }
    }

    fn push_preview(ctx: &mut Context, id: &str, latent: &Tensor) {
        if let Ok(preview) = FastLatentPreviewer::preview_latent(latent) {
            let mut buf = Vec::new();
            let mut c = Cursor::new(&mut buf);
            if preview.write_to(&mut c, image::ImageFormat::Jpeg).is_ok() {
                let data_url = format!("data:image/jpeg;base64,{}", BASE64.encode(&buf));
                ctx.set(id, data_url);
            }
        }
    }

    fn run_t2i(
        state: &StudioState,
        ctx: &mut Context,
        prompt: &str,
        negative: &str,
        steps: usize,
        guidance: f64,
        width: usize,
        height: usize,
        seed: u64,
    ) -> aurora_rust_engine::Result<()> {
        let model = state.current()?;
        let mut guard = model.lock().map_err(|e| aurora_rust_engine::LuminaError::Model(format!("lock: {e}")))?;
        let gen = downcast_model::<DiffusionModel>(&mut *guard)
            .ok_or_else(|| aurora_rust_engine::LuminaError::Model("pas un modèle de diffusion".into()))?;

        let mut cb = |_s: usize, _n: usize, latent: &Tensor| {
            push_preview(ctx, "t2i_output", latent);
        };

        let negative_prompt = if negative.trim().is_empty() { None } else { Some(negative) };
        let img = ImageGenerationModel::generate_t2i(
            gen,
            DiffusionParams {
                prompt,
                negative_prompt,
                num_steps: steps,
                guidance_scale: guidance,
                width,
                height,
                seed,
            },
            Some(&mut cb),
        )?;

        let data_url = b64_data_url(&img, image::ImageFormat::Png, "image/png");
        ctx.set("t2i_output", data_url.clone());
        ctx.set("t2i_status", format!("✅ {steps} steps · {width}×{height} · seed {seed}"));
        // Gallery attend un tableau de data-URLs, pas un objet {append}.
        let mut gallery: Vec<String> = ctx.get("t2i_gallery").unwrap_or_default();
        gallery.push(data_url);
        ctx.set("t2i_gallery", gallery);
        Ok(())
    }

    /// Cœur du clic « Générer » : lit les contrôles, bascule le modèle si besoin puis génère.
    /// Écrit des statuts d'avancement (`⏳ …`) pour que l'UI ne reste jamais muette.
    fn generate(
        state: &StudioState,
        ctx: &mut Context,
        choices: &[ModelChoice],
    ) -> std::result::Result<(), String> {
        let prompt: String = ctx.get("t2i_prompt").unwrap_or_default();
        let negative: String = ctx.get("t2i_negative").unwrap_or_default();
        let steps: f64 = ctx.get("t2i_steps").unwrap_or(25.0);
        let guidance: f64 = ctx.get("t2i_guidance").unwrap_or(7.0);
        let seed: f64 = ctx.get("t2i_seed").unwrap_or(42.0);
        let model_label: String = ctx.get("t2i_model").unwrap_or_else(|_| choices[0].label.clone());
        let size_label: String = ctx.get("t2i_size").unwrap_or_else(|_| "1024×1024".to_string());

        // Mapping label -> id depuis la config (aucun label en dur).
        let id = choices
            .iter()
            .find(|c| c.label == model_label)
            .map(|c| c.id.clone())
            .unwrap_or_else(|| choices[0].id.clone());

        // Mapping résolution label -> pixels.
        let (width, height) = match size_label.as_str() {
            "512×512" => (512, 512),
            "768×768" => (768, 768),
            _ => (1024, 1024),
        };

        let already_active = state.active_id.lock().unwrap().as_deref() == Some(id.as_str());
        if !already_active {
            ctx.set("t2i_status", format!("⏳ Chargement du modèle « {model_label} »…"));
        }
        state.switch(choices, &id).map_err(|e| format!("bascule modèle: {e}"))?;
        ctx.set("t2i_status", format!("⏳ Génération… {} steps · {width}×{height}", steps as usize));
        run_t2i(state, ctx, &prompt, &negative, steps as usize, guidance, width, height, seed as u64)
            .map_err(|e| format!("erreur: {e}"))?;
        Ok(())
    }

    pub async fn run() -> anyhow::Result<()> {
        let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
        let dtype = if device.is_cuda() { DType::F16 } else { DType::F32 };

        // Les modèles sont déclarés dans un fichier JSON (aucun chemin en dur ici) : chemin via
        // `STUDIO_CONFIG`, défaut `aurora_studio.json` dans le répertoire courant.
        let config_path = std::env::var("STUDIO_CONFIG").unwrap_or_else(|_| "aurora_studio.json".into());
        println!("📄 Chargement des modèles : {config_path}");
        let cfg = ModelDescriptorFile::load(&config_path)
            .map_err(|e| anyhow::anyhow!("{e} (voir aurora_studio.json à la racine du repo)"))?;
        let choices: Vec<ModelChoice> = cfg
            .resolve()?
            .into_iter()
            .map(|m| ModelChoice { id: m.id, label: m.label, desc: m.descriptor, defaults: m.defaults })
            .collect();
        if choices.is_empty() {
            anyhow::bail!("aucun modèle déclaré dans {config_path}");
        }

        // Défauts de génération du 1ᵉʳ modèle (côté config) : valeurs initiales des contrôles.
        let d0 = &choices[0].defaults;
        let def_steps = d0.steps.unwrap_or(25) as f64;
        let def_guidance = d0.guidance.unwrap_or(7.0);
        let def_size = match (d0.width, d0.height) {
            (Some(w), Some(h)) => format!("{w}×{h}"),
            _ => "1024×1024".to_string(),
        };
        let def_negative = d0.negative_prompt.clone().unwrap_or_default();

        let state = Arc::new(StudioState::new(device.clone(), dtype));
        // Pré-charge le premier modèle déclaré (éjection stricte ensuite).
        let st0 = state.clone();
        st0.switch(&choices, &choices[0].id)?;

        println!("✅ Aurora Studio : '{}' prêt. Modèles disponibles à la bascule (éjection stricte).", choices[0].label);
        for c in &choices {
            println!("   • {} — {}", c.id, c.label);
        }

        let app = App::new("Aurora Studio")
            .subtitle("100% Pure Rust · aurora-rust-engine × grio — Diffusion Multi-Modèle (éjection VRAM)")
            .theme(Theme::dark().primary("#6366f1").radius("12px"))
            .tabs(|t| {
                t.tab("🎨 Text-to-Image", |b| {
                    let model_labels: Vec<&str> = choices.iter().map(|c| c.label.as_str()).collect();
                    let default_label = choices.first().map(|c| c.label.clone()).unwrap_or_default();
                    b.row(|r| {
                        r.item(
                            Dropdown::new("t2i_model")
                                .label("Modèle")
                                .options(&model_labels)
                                .value(&default_label),
                        );
                        r.item(
                            Dropdown::new("t2i_size")
                                .label("Résolution")
                                .options(&["512×512", "768×768", "1024×1024"])
                                .value(&def_size),
                        );
                        r.item(
                            Text::new("t2i_prompt")
                                .label("Prompt")
                                .placeholder("a majestic white wolf on a snowy cliff at golden hour, photorealistic, 8k")
                                .value("a majestic white wolf on a snowy cliff at golden hour, photorealistic, 8k"),
                        );
                    });
                    b.row(|r| {
                        r.item(
                            Text::new("t2i_negative")
                                .label("Negative prompt")
                                .placeholder("(vide — décrit ce qu'on ne veut pas)")
                                .value(&def_negative),
                        );
                    });
                    b.row(|r| {
                        r.item(Slider::new("t2i_steps").label("Steps").min(1.0).max(50.0).step(1.0).value(def_steps));
                        r.item(Slider::new("t2i_guidance").label("Guidance").min(0.0).max(15.0).step(0.1).value(def_guidance));
                        r.item(Slider::new("t2i_seed").label("Seed").min(0.0).max(999999.0).step(1.0).value(42.0));
                    });
                    b.item(Button::new("t2i_go").label("🎨 Générer").primary());
                    b.item(Image::new("t2i_output").label("Rendu live").value("data:image/png;base64,"));
                    b.item(Output::new("t2i_status").label("Statut"));
                    b.item(Gallery::new("t2i_gallery").label("Historique").title("Galerie de la session"));
                })
            })
            .on_change("t2i_model", {
                let lookup = choices.clone();
                move |ctx| {
                    // Applique les défauts déclarés dans le config JSON à la bascule de modèle.
                    let label: String = ctx.get("t2i_model").unwrap_or_default();
                    if let Some(c) = lookup.iter().find(|c| c.label == label) {
                        let d = &c.defaults;
                        if let Some(s) = d.steps {
                            ctx.set("t2i_steps", s as f64);
                        }
                        if let Some(g) = d.guidance {
                            ctx.set("t2i_guidance", g);
                        }
                        if let (Some(w), Some(h)) = (d.width, d.height) {
                            ctx.set("t2i_size", format!("{w}×{h}"));
                        }
                        if let Some(n) = &d.negative_prompt {
                            ctx.set("t2i_negative", n.clone());
                        }
                    }
                    Ok(())
                }
            })
            .on_click("t2i_go", move |ctx| {
                // Verrouille le bouton pendant le travail (retour visuel immédiat), puis génère.
                ctx.set_prop("t2i_go", "disabled", true);
                let result = generate(&state, ctx, &choices);
                ctx.set_prop("t2i_go", "disabled", false);
                result.map_err(|e| e.into())
            });

        println!("\n🌐 Aurora Studio live: http://127.0.0.1:7860");
        app.launch("127.0.0.1:7860").map_err(|e| anyhow::anyhow!("launch: {e}"))?;
        Ok(())
    }
}

#[cfg(not(feature = "ui"))]
fn main() {
    println!("Aurora Studio nécessite le feature 'ui' : cargo run --release --bin aurora_studio --features cuda,flash-attn,ui");
}

#[cfg(feature = "ui")]
fn main() -> anyhow::Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(app::run())?;
    Ok(())
}
