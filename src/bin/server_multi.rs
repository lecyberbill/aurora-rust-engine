// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Multi-model inference server — proof that the `ModelRegistry` facade assembles into a platform

//! A minimal multi-model HTTP service built on the new `ModelRegistry` facade. This is a *proof of
//! integration*, not the final product: it shows that a platform registers several models (Flux for
//! image, Qwen/Mistral for text) and serves each by id through the same object-safe interface.
//!
//! Routes:
//!   GET  /health
//!   GET  /models                      -> list of registered models (id, family, kind)
//!   POST /generate/{model_id}         -> text-to-image (image models) or text (text models)
//!
//! Set models via env:
//!   MODEL_IMAGE=G:\models\flux\flux2DevFp8Scaled_fp8Scaled.safetensors  (else image is skipped)
//!   MODEL_TEXT=G:\models\clip\qwen_3_4b.safetensors                     (else text is skipped)

use axum::{extract::{Path, State}, routing::{get, post}, Json, Router};
use candle_core::{DType, Device};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing_subscriber::EnvFilter;

use aurora_rust_engine::models::{
    downcast_model, AutoModel, DiffusionModel, EncodeModel, ImageGenerationModel, ModelKind, ModelRegistry,
};
use aurora_rust_engine::traits::DiffusionParams;

#[derive(Clone)]
struct App {
    registry: Arc<ModelRegistry>,
}

#[derive(Serialize)]
struct ModelInfoDto {
    id: String,
    family: String,
    kind: String,
}

#[derive(Deserialize)]
struct GenRequest {
    prompt: String,
    #[serde(default = "d_steps")] steps: usize,
    #[serde(default = "d_guidance")] guidance: f64,
    #[serde(default = "d_dim")] width: usize,
    #[serde(default = "d_dim")] height: usize,
    #[serde(default = "d_tokens")] max_tokens: usize,
    #[serde(default = "d_temp")] _temperature: f64,
}
fn d_steps() -> usize { 8 }
fn d_guidance() -> f64 { 3.5 }
fn d_dim() -> usize { 512 }
fn d_tokens() -> usize { 128 }
fn d_temp() -> f64 { 0.7 }

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter(EnvFilter::from_default_env()).init();

    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let dtype = if device.is_cuda() { DType::F16 } else { DType::F32 };
    let hub = aurora_rust_engine::hub::ModelHub::from_env()?;
    let registry = Arc::new(ModelRegistry::new(hub, device.clone(), dtype));

    // Register models from env, off the runtime so the load doesn't block the server start.
    let img_path = std::env::var("MODEL_IMAGE").ok();

    let reg = registry.clone();
    let img_path_clone = img_path.clone();
    tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
        if let Some(p) = img_path_clone {
            let model = AutoModel::from_local(&p, device.clone(), dtype)?;
            reg.insert("flux".into(), model);
            tracing::info!("registered image model 'flux' from {p}");
        }
        if let Some(t) = std::env::var("MODEL_TEXT").ok() {
            let model = AutoModel::from_local(&t, device.clone(), dtype)?;
            reg.insert("qwen".into(), model);
            tracing::info!("registered text model 'qwen' from {t}");
        }
        Ok(())
    }).await??;

    let app = App { registry };

    let router = Router::new()
        .route("/health", get(health))
        .route("/models", get(list_models))
        .route("/generate/{model_id}", post(generate))
        .with_state(app);

    let addr: std::net::SocketAddr = "127.0.0.1:8081".parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("🦀 mult-model server listening on http://{addr}");
    axum::serve(listener, router).await?;
    Ok(())
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "healthy", "engine": "aurora-rust-engine" }))
}

async fn list_models(State(app): State<App>) -> Json<Vec<ModelInfoDto>> {
    let infos = app.registry.list();
    Json(infos.into_iter().map(|i| ModelInfoDto {
        id: i.id,
        family: i.family,
        kind: format!("{:?}", i.kind),
    }).collect())
}

/// Route by model id. Image models take DiffusionParams; text models take a prompt + max_tokens.
async fn generate(
    State(app): State<App>,
    Path(model_id): Path<String>,
    Json(req): Json<GenRequest>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    let model = app.registry.get(&model_id)
        .ok_or_else(|| (axum::http::StatusCode::NOT_FOUND, format!("model '{model_id}' not loaded")))?;

    let mut guard = model.lock().map_err(|_| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "model lock poisoned".into()))?;
    let kind = guard.kind();

    if kind == ModelKind::Diffusion {
        let gen = downcast_model::<DiffusionModel>(&mut *guard)
            .ok_or_else(|| (axum::http::StatusCode::BAD_REQUEST, "not a diffusion model".into()))?;
        let img = ImageGenerationModel::generate_t2i(
            gen,
            DiffusionParams {
                prompt: &req.prompt,
                negative_prompt: None,
                num_steps: req.steps,
                guidance_scale: req.guidance,
                width: req.width,
                height: req.height,
                seed: 42,
            },
            None,
        ).map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        img.save(&format!("outputs/{model_id}_server.png"))
            .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        Ok(Json(serde_json::json!({ "image": format!("outputs/{model_id}_server.png") })))
    } else if kind == ModelKind::TextEncoder {
        let enc = downcast_model::<aurora_rust_engine::models::TextModel>(&mut *guard)
            .ok_or_else(|| (axum::http::StatusCode::BAD_REQUEST, "not a text model".into()))?;
        let t = EncodeModel::encode(enc, &req.prompt, req.max_tokens)
            .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        let dims: Vec<usize> = t.dims().to_vec();
        Ok(Json(serde_json::json!({ "embed_shape": dims })))
    } else {
        Err((axum::http::StatusCode::BAD_REQUEST, format!("unsupported kind {:?}", kind)))
    }
}
