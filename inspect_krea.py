import torch
from safetensors import safe_open

with safe_open("/srv/models/krea/krea-2-turbo.safetensors", framework="pt") as f:
    for k in sorted(f.keys()):
        if any(x in k for x in ["first", "last", "mod", "time", "txtmlp", "projector", "blocks.0."]):
            print(f"{k:50s} {str(f.get_slice(k).get_shape()):25s} {f.get_slice(k).get_dtype()}")
