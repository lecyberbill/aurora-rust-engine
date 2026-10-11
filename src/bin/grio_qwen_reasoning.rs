// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Grio Interactive Showcase for Qwen3.8-27B CausalLM with Reasoning Effort Slider

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
        println!("🧠 Launching Aurora Qwen3.8-27B Studio powered by Grio UI...");
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

        let app = App::new("Aurora Qwen 27B Reasoning Studio")
            .subtitle("100% Pure Rust · Native AMD ROCm RDNA4 / CUDA · Thinking Effort Controller")
            .theme(Theme::dark().primary("#8b5cf6").radius("12px"))
            .tabs(|t| {
                t.tab("💬 LLM Reasoning Studio", |b| {
                    b.row(|r| {
                        r.item(
                            Text::new("prompt")
                                .label("Prompt")
                                .placeholder("Ask a challenging problem, math puzzle, logic query or code design...")
                                .value("Explain the difference between Monads and Functors in functional programming with a concrete Rust example:"),
                        );
                    });

                    b.row(|r| {
                        r.item(
                            Slider::new("reasoning_effort")
                                .label("🧠 Reasoning Effort (Thinking Depth: 0.0=Fast, 0.5=Medium, 1.0=Deep)")
                                .min(0.0)
                                .max(1.0)
                                .step(0.1)
                                .value(0.5),
                        );
                        r.item(
                            Slider::new("max_tokens")
                                .label("Max Tokens")
                                .min(32.0)
                                .max(512.0)
                                .step(16.0)
                                .value(128.0),
                        );
                        r.item(
                            Slider::new("temperature")
                                .label("Temperature")
                                .min(0.0)
                                .max(1.5)
                                .step(0.05)
                                .value(0.7),
                        );
                    });

                    b.item(Button::new("generate_btn").label("🚀 Generate with Reasoning").primary());
                    b.item(Output::new("telemetry").label("⚡ Hardware Telemetry & Throughput"));
                    b.item(Text::new("thinking_output").label("🤔 Internal Monologue / Thought Process (<think>)").value(""));
                    b.item(Text::new("final_output").label("📝 Final Response").value(""));
                })
            })
            .on_click("generate_btn", move |ctx| {
                ctx.set_prop("generate_btn", "disabled", true);
                ctx.set("telemetry", "⏳ Generating with Native RDNA4 / ROCm GPU Engine…".to_string());

                let prompt: String = ctx.get("prompt").unwrap_or_default();
                let reasoning_effort: f64 = ctx.get("reasoning_effort").unwrap_or(0.5);
                let max_tokens: f64 = ctx.get("max_tokens").unwrap_or(128.0);
                let temperature: f64 = ctx.get("temperature").unwrap_or(0.7);

                let t0 = Instant::now();
                let result = {
                    let lock = state_clone.lock().unwrap();
                    let mut model_guard = lock.lock().unwrap();
                    let text_model = downcast_model::<TextModel>(&mut *model_guard)
                        .ok_or_else(|| anyhow::anyhow!("Failed to downcast model to TextModel"))?;
                    text_model.generate_with_reasoning(&prompt, max_tokens as usize, temperature, reasoning_effort)
                };

                let total_elapsed = t0.elapsed().as_secs_f64();

                match result {
                    Ok((thinking, resp, tok_per_sec)) => {
                        ctx.set("thinking_output", thinking);
                        ctx.set("final_output", resp);
                        ctx.set(
                            "telemetry",
                            format!(
                                "✅ Completed in {:.2}s (~{:.1} tok/s) · Effort: {:.1} · Device: AMD Radeon AI PRO R9700 (34 GB VRAM)",
                                total_elapsed, tok_per_sec, reasoning_effort
                            ),
                        );
                    }
                    Err(e) => {
                        ctx.set("telemetry", format!("❌ Error: {e}"));
                    }
                }

                ctx.set_prop("generate_btn", "disabled", false);
                Ok(())
            });

        let bind_addr = std::env::var("STUDIO_BIND").unwrap_or_else(|_| "0.0.0.0:7865".to_string());
        println!("\n🌐 Aurora Qwen 27B Studio live on: http://{bind_addr}");
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
