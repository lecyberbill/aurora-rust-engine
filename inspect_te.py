import json, struct

path = '/models/comfyui/diffusion_models/krea2_turbo_fp8_scaled.safetensors'
with open(path, 'rb') as f:
    sz = struct.unpack('<Q', f.read(8))[0]
    hdr = json.loads(f.read(sz).decode('utf-8'))

for k, v in hdr.items():
    if 'txtfusion.layerwise_blocks.0' in k or 'txtfusion.refiner_blocks.0' in k:
        print(k, v)
