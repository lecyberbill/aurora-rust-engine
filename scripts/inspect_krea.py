import safetensors

p = "/models/comfyui/diffusion_models/krea2_turbo_fp8_scaled.safetensors"
with safetensors.safe_open(p, framework="pt") as f:
    keys = list(f.keys())
    print("Total keys:", len(keys))
    for k in sorted(keys):
        if not any(k.startswith(f"blocks.{i}.") for i in range(1, 28)) and not k.endswith(".weight_scale"):
            print(f"{k:45} : {f.get_slice(k).get_shape()} dtype={f.get_slice(k).get_dtype()}")
