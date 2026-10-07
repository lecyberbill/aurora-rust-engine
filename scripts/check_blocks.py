import safetensors

p = "/models/comfyui/diffusion_models/krea2_turbo_fp8_scaled.safetensors"
with safetensors.safe_open(p, framework="pt") as f:
    keys = list(f.keys())
    for b in range(28):
        b_keys = [k for k in keys if f"blocks.{b}." in k or f"blocks.{b:02d}." in k or f"layers.{b}." in k]
        print(f"Block {b}: {len(b_keys)} keys found. Sample: {b_keys[:2]}")
