import json, struct, sys

path = sys.argv[1] if len(sys.argv) > 1 else '/models/comfyui/vae/qwen_image_vae_complet.safetensors'
with open(path, 'rb') as f:
    sz = struct.unpack('<Q', f.read(8))[0]
    hdr = json.loads(f.read(sz).decode('utf-8'))

print(f"Total keys in {path}: {len(hdr)}")
for k, v in hdr.items():
    if any(x in k.lower() for x in ['mean', 'std', 'scale', 'latents']):
        print(f"MATCH: {k} -> {v}")
