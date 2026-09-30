"""Dump a MusicGen decoder reference (hidden + per-codebook logits)."""

import torch
from safetensors.torch import save_file
from transformers import MusicgenForConditionalGeneration

D = "D:/models/musicgen-small"
m = MusicgenForConditionalGeneration.from_pretrained(D, dtype=torch.float32).eval()
torch.manual_seed(0)
ids = torch.randint(0, 2049, (4, 8))  # (bsz*num_codebooks, seq)
enc = torch.randn(1, 20, 1024)
with torch.no_grad():
    out = m.decoder(input_ids=ids, encoder_hidden_states=enc, output_hidden_states=True)
    hidden = out.hidden_states[-1]
    logits = out.logits
save_file(
    {"ids": ids.float(), "enc": enc.float(), "hidden": hidden.float().cpu().contiguous(),
     "logits": logits.float().cpu().contiguous()},
    "outputs/audio_ref/musicgen_ref.safetensors",
)
print("ids", tuple(ids.shape), "hidden", tuple(hidden.shape), "logits", tuple(logits.shape))
