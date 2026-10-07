import safetensors
f = safetensors.safe_open('/models/comfyui/vae/qwen_image_vae_complet.safetensors', framework='pt')
print("Keys in VAE:", len(f.keys()))
for k in f.keys():
    if any(x in k.lower() for x in ['mean', 'std', 'scale', 'shift']):
        t = f.get_tensor(k)
        print(k, t.shape, t)
