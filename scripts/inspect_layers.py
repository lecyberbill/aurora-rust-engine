import json
import struct

path = '/models/comfyui/text_encoders/qwen3vl_4b_fp8_scaled.safetensors'
with open(path, 'rb') as f:
    sz = struct.unpack('<Q', f.read(8))[0]
    hdr = json.loads(f.read(sz).decode('utf-8'))

print("=== LAYER 5 vs 6 ===")
for k in sorted(hdr.keys()):
    if 'layers.5.' in k or 'layers.6.' in k:
        print(f"{k}: {hdr[k]}")
