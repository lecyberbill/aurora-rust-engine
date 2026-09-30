"""Dump an EnCodec (MusicGen) decoder reference: random codes -> audio."""

import torch
from safetensors.torch import save_file
from transformers import MusicgenForConditionalGeneration

D = "D:/models/musicgen-small"
m = MusicgenForConditionalGeneration.from_pretrained(D, dtype=torch.float32).eval()
torch.manual_seed(0)
codes = torch.randint(0, 2048, (1, 1, 4, 50))  # (frames, batch, K, T)
with torch.no_grad():
    out = m.audio_encoder.decode(codes, [None])
audio = out.audio_values
save_file({"codes": codes[0].float(), "audio": audio.float().cpu().contiguous()}, "outputs/audio_ref/encodec_ref.safetensors")
print("codes", tuple(codes.shape), "audio", tuple(audio.shape), "sample_rate", m.audio_encoder.config.sampling_rate)
