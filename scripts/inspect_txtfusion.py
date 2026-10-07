import json
import struct

path = '/models/comfyui/diffusion_models/krea2_turbo_fp8_scaled.safetensors'
with open(path, 'rb') as f:
    sz = struct.unpack('<Q', f.read(8))[0]
    hdr = json.loads(f.read(sz).decode('utf-8'))

for k in sorted(hdr.keys()):
    if 'txtfusion' in k and not k.endswith('.weight_scale'):
        print(f"{k:45} : shape={hdr[k]['shape']}, dtype={hdr[k]['dtype']}")
