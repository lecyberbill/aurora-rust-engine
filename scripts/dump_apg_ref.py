"""Dump reference APG guidance outputs for Rust validation."""

import os
import sys

import torch
from safetensors.torch import save_file

REPO = "D:/image_to_text/ai_music_gen/ace_step_repo"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from acestep.models.common.apg_guidance import MomentumBuffer, apg_forward  # noqa: E402


def main():
    # Real layout is `[B, T, C]` with `dims=[1]` (projection over the time axis).
    g = torch.Generator().manual_seed(3)
    pc_l, pu_l, out_l = [], [], []
    m = MomentumBuffer()
    for _ in range(4):
        a = torch.randn(1, 8, 64, generator=g, dtype=torch.float32)
        b = torch.randn(1, 8, 64, generator=g, dtype=torch.float32)
        v = apg_forward(a, b, 7.0, momentum_buffer=m, dims=[1])
        pc_l.append(a)
        pu_l.append(b)
        out_l.append(v)
    out = "outputs/audio_ref/apg_ref.safetensors"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    save_file({"pc": torch.stack(pc_l), "pu": torch.stack(pu_l), "out": torch.stack(out_l)}, out)
    print(f"[dumped] {os.path.abspath(out)} out mean={torch.stack(out_l).mean().item():.6f}")


if __name__ == "__main__":
    main()
