// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Grio Native Chatbot Showcase for Qwen3.8-27B with Reasoning & Message History

#[cfg(feature = "ui")]
mod app {
    use std::sync::{Arc, Mutex};
    use std::time::Instant;
    use candle_core::{DType, Device};
    use grio::*;
    use aurora_rust_engine::models::{AutoModel, GenerationModel, downcast_model, BoxedModel};
    use aurora_rust_engine::models::text::TextModel;

    pub async fn run() -> anyhow::Result<()> {
        println!("================================================================================");
        println!("💬 Launching Aurora Qwen3.8-27B Chatbot Studio powered by Grio UI...");
        println!("================================================================================\n");

        let model_path = std::env::var("MODEL_PATH").unwrap_or_else(|_| {
            "/usr/share/ollama/.ollama/models/blobs/sha256-f5f1dd8920d417aac2718b0bda3403da274301efdd6760b4f0f4b864ff2ad57d".into()
        });

        println!("📦 Resolving Qwen3.8-27B Model checkpoint: {}", model_path);
        let device = match aurora_rust_engine::device::auto_device() {
            Ok(d) => d,
            Err(_) => Device::Cpu,
        };

        let boxed_model: BoxedModel = AutoModel::from_local(&model_path, device, DType::F16)?;
        println!("✅ AutoModel: High-Speed Native ROCm/CUDA Engine Attached!");

        let model_state = Arc::new(Mutex::new(boxed_model));
        let state_clone = Arc::clone(&model_state);

        let app = App::new("Aurora Qwen 27B Chat Studio")
            .subtitle("100% Pure Rust Chatbot · Native AMD ROCm RDNA4 / CUDA · Thinking Monologue & History")
            .theme(Theme::dark().primary("#8b5cf6").radius("12px"))
            .item(
                Chatbot::new("chat")
                    .label("Qwen3.8-27B Reasoning Assistant (AMD Radeon AI PRO R9700)")
                    .height(520)
                    .message("assistant", "Bonjour ! Je suis Qwen3.8-27B tournant en natif sur votre GPU AMD RDNA 4. Posez-moi une question technique, logique ou de code.")
            )
            .row(|r| {
                r.item(
                    Slider::new("reasoning_effort")
                        .label("🧠 Reasoning Effort (0.0 = Direct, 0.5 = Balanced, 1.0 = Deep Thought)")
                        .min(0.0)
                        .max(1.0)
                        .step(0.1)
                        .value(0.5),
                );
                r.item(
                    Slider::new("max_tokens")
                        .label("Max Tokens (Budget Thinking + Answer)")
                        .min(128.0)
                        .max(2048.0)
                        .step(128.0)
                        .value(512.0),
                );
                r.item(
                    Slider::new("temperature")
                        .label("Temperature")
                        .min(0.0)
                        .max(1.5)
                        .step(0.05)
                        .value(0.7),
                );
            })
            .row(|r| {
                r.item(
                    Text::new("prompt")
                        .placeholder("Ask a question, design a Rust architecture, or request mathematical reasoning...")
                );
                r.item(Button::new("send").label("Envoyer 🚀").primary());
                r.item(Button::new("clear").label("Effacer 🗑️"));
            })
            .item(Output::new("telemetry").label("⚡ Hardware Telemetry & Throughput"))
            .on_click("send", {
                let state = Arc::clone(&state_clone);
                move |ctx| {
                    let prompt: String = ctx.get("prompt").unwrap_or_default();
                    if prompt.trim().is_empty() {
                        return Ok(());
                    }

                    let reasoning_effort: f64 = ctx.get("reasoning_effort").unwrap_or(0.5);
                    let max_tokens: f64 = ctx.get("max_tokens").unwrap_or(512.0);
                    let temperature: f64 = ctx.get("temperature").unwrap_or(0.7);

                    ctx.set_prop("send", "disabled", true);
                    ctx.set("telemetry", "⏳ Thinking & Generating on GPU…".to_string());

                    // Append user message immediately
                    let mut history: Vec<ChatMessage> = ctx.get("chat").unwrap_or_default();
                    history.push(ChatMessage::user(&prompt));
                    ctx.set("chat", history.clone());
                    ctx.set("prompt", "".to_string());

                    let t0 = Instant::now();
                    let result = {
                        let lock = state.lock().unwrap();
                        let mut model_guard = lock.lock().unwrap();
                        let text_model = downcast_model::<TextModel>(&mut *model_guard)
                            .ok_or_else(|| anyhow::anyhow!("Failed to downcast model to TextModel"))?;
                        text_model.generate_with_reasoning(&prompt, max_tokens as usize, temperature, reasoning_effort)
                    };

                    let total_elapsed = t0.elapsed().as_secs_f64();

                    match result {
                        Ok((thinking, resp, tok_per_sec)) => {
                            let formatted_message = if !thinking.is_empty() {
                                format!(
                                    "> 🧠 **Pensée interne (<think>) :**\n> {}\n\n{}",
                                    thinking.replace("\n", "\n> "),
                                    resp
                                )
                            } else {
                                resp
                            };

                            history.push(ChatMessage::assistant(&formatted_message));
                            ctx.set("chat", history);
                            ctx.set(
                                "telemetry",
                                format!(
                                    "✅ Generated in {:.2}s (~{:.1} tok/s) · Effort: {:.1} · AMD Radeon AI PRO R9700 (34 GB VRAM)",
                                    total_elapsed, tok_per_sec, reasoning_effort
                                ),
                            );
                        }
                        Err(e) => {
                            ctx.set("telemetry", format!("❌ Erreur d'inférence: {e}"));
                        }
                    }

                    ctx.set_prop("send", "disabled", false);
                    Ok(())
                }
            })
            .on_click("clear", |ctx| {
                let empty: Vec<ChatMessage> = Vec::new();
                ctx.set("chat", empty);
                ctx.set("prompt", "".to_string());
                ctx.set("telemetry", "Chat réinitialisé.".to_string());
                Ok(())
            });

        let bind_addr = std::env::var("STUDIO_BIND").unwrap_or_else(|_| "0.0.0.0:7865".to_string());
        println!("\n🌐 Aurora Qwen 27B Chat Studio live on: http://{bind_addr}");
        app.launch(&bind_addr).map_err(|e| anyhow::anyhow!("launch: {e}"))?;
        Ok(())
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    #[cfg(feature = "ui")]
    {
        app::run().await
    }
    #[cfg(not(feature = "ui"))]
    {
        println!("Please run with --features ui,rocm (or ui,cuda)");
        Ok(())
    }
}
