"""Dump reference AutoencoderOobleck encode (mean/logvar) for Rust validation."""

import os

import torch
from diffusers import AutoencoderOobleck
from safetensors.torch import save_file


def main():
    vae = AutoencoderOobleck.from_pretrained("G:/models/Audio/vae").to(torch.float32).eval()
    g = torch.Generator().manual_seed(0)
    frames = 16
    n = 1920 * frames
    audio = (torch.randn(1, 2, n, generator=g, dtype=torch.float32) * 0.3).contiguous()

    with torch.no_grad():
        raw = vae.encoder(audio)
        dist = vae.encode(audio).latent_dist
        mean = dist.mean
        std = dist.std

    os.makedirs("outputs/audio_ref", exist_ok=True)
    out = "outputs/audio_ref/vae_enc_ref.safetensors"
    save_file(
        {
            "audio": audio,
            "raw": raw.contiguous(),
            "mean": mean.contiguous(),
            "std": std.contiguous(),
        },
        out,
    )
    print(f"[dumped] {os.path.abspath(out)}  audio={tuple(audio.shape)} mean={tuple(mean.shape)}")


if __name__ == "__main__":
    main()
