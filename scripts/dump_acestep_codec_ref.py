"""Dump reference ACE-Step 5Hz audio-codec activations (tokenizer + detokenizer + FSQ)."""

import argparse
import json
import os
import sys

import torch
from safetensors.torch import load_file, save_file

REPO = "D:/image_to_text/ai_music_gen/ace_step_repo"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from acestep.models.common.configuration_acestep_v15 import AceStepConfig  # noqa: E402
from acestep.models.base.modeling_acestep_v15_base import (  # noqa: E402
    AceStepAudioTokenizer,
    AudioTokenDetokenizer,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="outputs/audio_ref/codec_ref.safetensors")
    ap.add_argument("--frames", type=int, default=20)  # divisible by 5
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(os.path.join(args.checkpoint, "config.json")) as f:
        c = json.load(f)
    config = AceStepConfig(
        hidden_size=c["hidden_size"],
        intermediate_size=c["intermediate_size"],
        num_attention_heads=c["num_attention_heads"],
        num_key_value_heads=c["num_key_value_heads"],
        head_dim=c["head_dim"],
        rms_norm_eps=c.get("rms_norm_eps", 1e-6),
        rope_theta=c.get("rope_theta", 1000000),
        audio_acoustic_hidden_dim=c.get("audio_acoustic_hidden_dim", 64),
        pool_window_size=c.get("pool_window_size", 5),
        fsq_dim=c.get("fsq_dim", 2048),
        fsq_input_levels=c.get("fsq_input_levels", [8, 8, 8, 5, 5, 5]),
        fsq_input_num_quantizers=c.get("fsq_input_num_quantizers", 1),
        model_version="base",
    )
    config._attn_implementation = "eager"

    sd = load_file(os.path.join(args.checkpoint, "model.safetensors"))
    tok_sd = {k[len("tokenizer."):]: v.float() for k, v in sd.items() if k.startswith("tokenizer.")}
    detok_sd = {k[len("detokenizer."):]: v.float() for k, v in sd.items() if k.startswith("detokenizer.")}

    tokenizer = AceStepAudioTokenizer(config).float().eval()
    tokenizer.load_state_dict(tok_sd, strict=False)
    detokenizer = AudioTokenDetokenizer(config).float().eval()
    detokenizer.load_state_dict(detok_sd, strict=False)

    g = torch.Generator().manual_seed(args.seed)
    features = torch.randn(1, args.frames, 64, generator=g, dtype=torch.float32)
    p = config.pool_window_size
    x = features.reshape(1, args.frames // p, p, 64)  # [1, Tp, P, 64]

    with torch.no_grad():
        quantized, indices = tokenizer(x)  # quantized [1,Tp,2048], indices [1,Tp,1]
        detok = detokenizer(quantized)  # [1, Tp*P, 64]
        from_idx = tokenizer.quantizer.get_output_from_indices(indices)  # [1,Tp,2048]
        detok_from_idx = detokenizer(from_idx)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_file(
        {
            "features": features.contiguous(),
            "quantized": quantized.float().contiguous(),
            "indices": indices.float().contiguous(),
            "detok": detok.float().contiguous(),
            "from_idx": from_idx.float().contiguous(),
            "detok_from_idx": detok_from_idx.float().contiguous(),
        },
        args.out,
    )
    print(f"[dumped] {os.path.abspath(args.out)}")
    print(f"  features {tuple(features.shape)}  quantized {tuple(quantized.shape)}  indices {tuple(indices.shape)}")
    print(f"  detok {tuple(detok.shape)}  range[{indices.min().item()},{indices.max().item()}]")


if __name__ == "__main__":
    main()
