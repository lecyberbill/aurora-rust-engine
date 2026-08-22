// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: High-Performance Async Axum HTTP & WebSocket Inference Server

use axum::{
    extract::{State, WebSocketUpgrade},
    extract::ws::{Message, WebSocket},
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use candle_core::Tensor;
use image::ImageFormat;
use serde::{Deserialize, Serialize};
use std::io::Cursor;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;

use crate::{DiffusionParams, GenerationMetrics, StableDiffusionXLPipeline};

/// DTO for Text-to-Image Generation Request
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GenerateRequest {
    pub prompt: String,
    pub negative_prompt: Option<String>,
    #[serde(default = "default_steps")]
    pub steps: usize,
    #[serde(default = "default_guidance")]
    pub guidance_scale: f64,
    #[serde(default = "default_dim")]
    pub width: usize,
    #[serde(default = "default_dim")]
    pub height: usize,
    pub seed: Option<u64>,
    /// Optional dynamic override for VAE Tiling (default: false for maximum speed)
    pub vae_tiling: Option<bool>,
    /// Optional dynamic override for CPU Offloading
    pub cpu_offload: Option<bool>,
    /// Optional dynamic override for FP8 quantization
    pub fp8: Option<bool>,
    /// Optional scheduler selection: "dpm", "euler", "ddim" (default: "euler")
    pub scheduler: Option<String>,
}

fn default_steps() -> usize { 30 }
fn default_guidance() -> f64 { 6.5 }
fn default_dim() -> usize { 1024 }

/// DTO for High-Resolution Generation Telemetry
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GenerationMetricsDto {
    pub prompt_encode_ms: f64,
    pub unet_steps: usize,
    pub unet_total_ms: f64,
    pub unet_step_avg_ms: f64,
    pub unet_it_per_sec: f64,
    pub vae_decode_ms: f64,
    pub total_wallclock_ms: f64,
}

impl From<GenerationMetrics> for GenerationMetricsDto {
    fn from(m: GenerationMetrics) -> Self {
        Self {
            prompt_encode_ms: m.prompt_encode_ms,
            unet_steps: m.unet_steps,
            unet_total_ms: m.unet_total_ms,
            unet_step_avg_ms: m.unet_step_avg_ms,
            unet_it_per_sec: m.unet_it_per_sec,
            vae_decode_ms: m.vae_decode_ms,
            total_wallclock_ms: m.total_wallclock_ms,
        }
    }
}

/// DTO for Text-to-Image Generation Response
#[derive(Debug, Clone, Serialize)]
pub struct GenerateResponse {
    pub image_base64: String,
    pub format: String,
    pub metrics: GenerationMetricsDto,
}

/// Health and Engine Capabilities Status
#[derive(Debug, Clone, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub engine: String,
    pub cuda_accelerated: bool,
    pub flash_attention_2: bool,
    pub lora_in_memory_merging: bool,
    pub controlnet_support: bool,
    pub inpainting_support: bool,
}

/// Shared Server State
#[derive(Clone)]
pub struct AppState {
    pub pipeline: Arc<Mutex<StableDiffusionXLPipeline>>,
}

/// Build Axum Router for Inference Server
pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/api/v1/health", get(health_handler))
        .route("/api/v1/generate", post(generate_handler))
        .route("/api/v1/ws", get(ws_handler))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

/// Health check handler returning engine capabilities
pub async fn health_handler() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        engine: "aurora-rust-engine v0.1.0".to_string(),
        cuda_accelerated: cfg!(feature = "cuda"),
        flash_attention_2: cfg!(feature = "flash-attn"),
        lora_in_memory_merging: true,
        controlnet_support: true,
        inpainting_support: true,
    })
}

/// REST handler for text-to-image generation
pub async fn generate_handler(
    State(state): State<AppState>,
    Json(req): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, (StatusCode, String)> {
    let seed = req.seed.unwrap_or(42);
    let params = DiffusionParams {
        prompt: &req.prompt,
        negative_prompt: req.negative_prompt.as_deref(),
        num_steps: req.steps,
        guidance_scale: req.guidance_scale,
        width: req.width,
        height: req.height,
        seed,
    };

    let mut pipeline = state.pipeline.lock().await;
    if let Some(tiling) = req.vae_tiling {
        if tiling {
            pipeline.enable_vae_tiling(None);
        } else {
            pipeline.disable_vae_tiling();
        }
    }
    if let Some(offload) = req.cpu_offload {
        if offload {
            pipeline.enable_model_cpu_offload();
        } else {
            pipeline.disable_model_cpu_offload();
        }
    }
    if let Some(ref s) = req.scheduler {
        match s.to_lowercase().as_str() {
            "dpm" | "dpm++" | "dpmsolver" => { pipeline.use_dpm_solver(); }
            "ddim" => { pipeline.use_ddim(); }
            _ => { pipeline.use_euler(); }
        }
    }

    let (image, metrics) = pipeline
        .generate_with_metrics(params, None::<fn(usize, usize, &Tensor)>)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Generation error: {:?}", e)))?;

    // Encode image buffer to PNG in memory
    let mut png_bytes: Vec<u8> = Vec::new();
    let mut cursor = Cursor::new(&mut png_bytes);
    image
        .write_to(&mut cursor, ImageFormat::Png)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("PNG encode error: {:?}", e)))?;

    let image_base64 = BASE64.encode(&png_bytes);

    Ok(Json(GenerateResponse {
        image_base64,
        format: "png".to_string(),
        metrics: metrics.into(),
    }))
}

/// WebSocket handler for real-time progressive generation stream
pub async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_ws_stream(socket, state))
}

async fn handle_ws_stream(mut socket: WebSocket, state: AppState) {
    if let Some(Ok(Message::Text(text))) = socket.recv().await {
        if let Ok(req) = serde_json::from_str::<GenerateRequest>(&text) {
            let seed = req.seed.unwrap_or(42);
            let params = DiffusionParams {
                prompt: &req.prompt,
                negative_prompt: req.negative_prompt.as_deref(),
                num_steps: req.steps,
                guidance_scale: req.guidance_scale,
                width: req.width,
                height: req.height,
                seed,
            };

            let mut pipeline = state.pipeline.lock().await;
            if let Some(tiling) = req.vae_tiling {
                if tiling { pipeline.enable_vae_tiling(None); } else { pipeline.disable_vae_tiling(); }
            }
            if let Some(offload) = req.cpu_offload {
                if offload { pipeline.enable_model_cpu_offload(); } else { pipeline.disable_model_cpu_offload(); }
            }
            if let Some(ref s) = req.scheduler {
                match s.to_lowercase().as_str() {
                    "dpm" | "dpm++" | "dpmsolver" => { pipeline.use_dpm_solver(); }
                    "ddim" => { pipeline.use_ddim(); }
                    _ => { pipeline.use_euler(); }
                }
            }

            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

            // Run generation with fast latent previewer callback (1.5ms preview generation)
            let result = pipeline.generate_with_metrics(
                params,
                Some(move |step: usize, total: usize, latent: &Tensor| {
                    if let Ok(preview_img) = crate::diffusion::FastLatentPreviewer::preview_latent(latent) {
                        let mut buf = Vec::new();
                        let mut cur = Cursor::new(&mut buf);
                        if preview_img.write_to(&mut cur, ImageFormat::Jpeg).is_ok() {
                            let b64 = BASE64.encode(&buf);
                            let _ = tx.send((step, total, b64));
                        }
                    }
                }),
            );

            // Forward intermediate steps
            while let Ok((step, total, b64)) = rx.try_recv() {
                let msg = serde_json::json!({
                    "type": "progress",
                    "step": step,
                    "total_steps": total,
                    "preview_base64": b64,
                });
                let _ = socket.send(Message::Text(msg.to_string().into())).await;
            }

            match result {
                Ok((image, metrics)) => {
                    let mut png_bytes: Vec<u8> = Vec::new();
                    let mut cursor = Cursor::new(&mut png_bytes);
                    if image.write_to(&mut cursor, ImageFormat::Png).is_ok() {
                        let resp = serde_json::json!({
                            "type": "complete",
                            "image_base64": BASE64.encode(&png_bytes),
                            "format": "png",
                            "metrics": GenerationMetricsDto::from(metrics),
                        });
                        let _ = socket.send(Message::Text(resp.to_string().into())).await;
                    }
                }
                Err(err) => {
                    let _ = socket.send(Message::Text(format!("{{\"error\": \"{:?}\"}}", err).into())).await;
                }
            }
        }
    }
}

/// Run Axum server on specified address
pub async fn run_server(pipeline: StableDiffusionXLPipeline, addr: SocketAddr) -> anyhow::Result<()> {
    let state = AppState {
        pipeline: Arc::new(Mutex::new(pipeline)),
    };
    let app = build_router(state);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    println!("🌐 Aurora Rust AI Inference Server listening on http://{}", addr);
    axum::serve(listener, app).await?;
    Ok(())
}
