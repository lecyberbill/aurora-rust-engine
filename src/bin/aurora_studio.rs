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
        downcast_model, AnyModel, AutoModel, DiffusionModel, ImageGenerationModel, ModelDescriptor,
        TextEncoderSpec,
    };
    use aurora_rust_engine::traits::DiffusionParams;
    use aurora_rust_engine::FastLatentPreviewer;

    /// Un modèle déclaré dans la vitrine (id + descripteur pour le chargement).
    struct ModelChoice {
        id: &'static str,
        label: &'static str,
        desc: ModelDescriptor,
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

        let img = ImageGenerationModel::generate_t2i(
            gen,
            DiffusionParams {
                prompt,
                negative_prompt: None,
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

    pub async fn run() -> anyhow::Result<()> {
        let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
        let dtype = if device.is_cuda() { DType::F16 } else { DType::F32 };

        // Les 4 modèles vitrine. Env surcharge les chemins.
        let sdxl_ckpt = std::env::var("SDXL_CKPT").unwrap_or_else(|_| "G:\\models\\checkpoints\\Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors".into());
        let flux1_ckpt = std::env::var("FLUX1_CKPT").unwrap_or_else(|_| "G:\\models\\flux\\flux1-dev-fp8.safetensors".into());
        let klein_ckpt = std::env::var("KLEIN_CKPT").unwrap_or_else(|_| "G:\\models\\flux\\fluxKlein4BPro_v10.safetensors".into());
        let qwen_path = std::env::var("QWEN_CKPT").unwrap_or_else(|_| "G:\\models\\clip\\qwen_3_4b.safetensors".into());
        let flux_vae = std::env::var("FLUX_VAE").unwrap_or_else(|_| "G:\\models\\vae\\flux2-vae.safetensors".into());
        // SD 3.5 : checkpoint + 3 encodeurs texte + VAE 16ch.
        let sd35_ckpt = std::env::var("SD35_CKPT").unwrap_or_else(|_| "G:\\models\\SD3\\sd3.5_large.safetensors".into());
        let sd35_clip_l = std::env::var("SD35_CLIP_L").unwrap_or_else(|_| "G:\\models\\clip\\clip_l.safetensors".into());
        let sd35_clip_g = std::env::var("SD35_CLIP_G").unwrap_or_else(|_| "G:\\models\\clip\\clip_g.safetensors".into());
        let sd35_t5 = std::env::var("SD35_T5").unwrap_or_else(|_| "G:\\models\\clip\\t5xxl_fp16.safetensors".into());
        let sd35_vae = std::env::var("SD35_VAE").unwrap_or_else(|_| "G:\\models\\vae\\sd3_vae.safetensors".into());

        let choices: Vec<ModelChoice> = vec![
            ModelChoice {
                id: "sdxl",
                label: "SDXL (Juggernaut XL)",
                desc: ModelDescriptor::standalone("sdxl", &sdxl_ckpt),
            },
            ModelChoice {
                id: "flux1",
                label: "Flux.1 Dev (embarqué VLM)",
                desc: ModelDescriptor::standalone("flux1", &flux1_ckpt),
            },
            ModelChoice {
                id: "klein4b",
                label: "Flux.2 Klein-4B (Qwen3)",
                desc: ModelDescriptor::flux2(
                    "klein4b",
                    &klein_ckpt,
                    TextEncoderSpec::Qwen3 { path: std::path::PathBuf::from(&qwen_path) },
                    &flux_vae,
                ),
            },
            ModelChoice {
                id: "sd35",
                label: "SD 3.5 Large (CLIP-L+G+T5)",
                desc: ModelDescriptor::sd35("sd35", &sd35_ckpt, &sd35_clip_l, &sd35_clip_g, &sd35_t5, &sd35_vae),
            },
        ];

        let state = Arc::new(StudioState::new(device.clone(), dtype));
        // Pré-charge le défaut (SDXL).
        let st0 = state.clone();
        st0.switch(&choices, "sdxl")?;

        println!("✅ Aurora Studio : SDXL prêt. Modèles disponibles à la bascule (éjection stricte).");
        for c in &choices {
            println!("   • {} — {}", c.id, c.label);
        }

        let app = App::new("Aurora Studio")
            .subtitle("100% Pure Rust · aurora-rust-engine × grio — Diffusion Multi-Modèle (éjection VRAM)")
            .theme(Theme::dark().primary("#6366f1").radius("12px"))
            .tabs(|t| {
                t.tab("🎨 Text-to-Image", |b| {
                    b.row(|r| {
                        let opts: Vec<&str> = choices.iter().map(|c| c.label).collect();
                        let _ = &opts;
                        r.item(
                            Dropdown::new("t2i_model")
                                .label("Modèle")
                                .options(&["SDXL (Juggernaut XL)", "Flux.1 Dev (embarqué VLM)", "Flux.2 Klein-4B (Qwen3)", "SD 3.5 Large (CLIP-L+G+T5)"])
                                .value("SDXL (Juggernaut XL)"),
                        );
                        r.item(
                            Dropdown::new("t2i_size")
                                .label("Résolution")
                                .options(&["512×512", "768×768", "1024×1024"])
                                .value("1024×1024"),
                        );
                        r.item(
                            Text::new("t2i_prompt")
                                .label("Prompt")
                                .placeholder("a majestic white wolf on a snowy cliff at golden hour, photorealistic, 8k")
                                .value("a majestic white wolf on a snowy cliff at golden hour, photorealistic, 8k"),
                        );
                    });
                    b.row(|r| {
                        r.item(Slider::new("t2i_steps").label("Steps").min(1.0).max(50.0).step(1.0).value(25.0));
                        r.item(Slider::new("t2i_guidance").label("Guidance").min(0.0).max(15.0).step(0.1).value(7.0));
                        r.item(Slider::new("t2i_seed").label("Seed").min(0.0).max(999999.0).step(1.0).value(42.0));
                    });
                    b.item(Button::new("t2i_go").label("🎨 Générer").primary());
                    b.item(Image::new("t2i_output").label("Rendu live").value("data:image/png;base64,"));
                    b.item(Output::new("t2i_status").label("Statut"));
                    b.item(Gallery::new("t2i_gallery").label("Historique").title("Galerie de la session"));
                })
            })
            .on_click("t2i_go", move |ctx| {
                let prompt: String = ctx.get("t2i_prompt").unwrap_or_default();
                let steps: f64 = ctx.get("t2i_steps").unwrap_or(25.0);
                let guidance: f64 = ctx.get("t2i_guidance").unwrap_or(7.0);
                let seed: f64 = ctx.get("t2i_seed").unwrap_or(42.0);
                let model_label: String = ctx.get("t2i_model").unwrap_or_else(|_| "SDXL (Juggernaut XL)".to_string());
                let size_label: String = ctx.get("t2i_size").unwrap_or_else(|_| "1024×1024".to_string());

                // Mapping label -> id.
                let id = match model_label.as_str() {
                    "Flux.1 Dev (embarqué VLM)" => "flux1",
                    "Flux.2 Klein-4B (Qwen3)" => "klein4b",
                    "SD 3.5 Large (CLIP-L+G+T5)" => "sd35",
                    _ => "sdxl",
                };

                // Mapping résolution label -> pixels.
                let (width, height) = match size_label.as_str() {
                    "512×512" => (512, 512),
                    "768×768" => (768, 768),
                    _ => (1024, 1024),
                };

                state.switch(&choices, id).map_err(|e| format!("bascule modèle: {e}"))?;
                run_t2i(&state, ctx, &prompt, steps as usize, guidance, width, height, seed as u64)
                    .map_err(|e| format!("erreur: {e}"))?;

                Ok(())
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
