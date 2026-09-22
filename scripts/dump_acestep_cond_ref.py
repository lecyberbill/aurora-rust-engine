"""Dump reference ACE-Step 1.5 Turbo ConditionEncoder activations for Rust validation.

Uses the official `ace_step_repo` implementation with the diffusers condition-encoder
checkpoint. Run with a torch+transformers env.

Example:
    python scripts/dump_acestep_cond_ref.py \
        --model-dir G:/models/Audio \
        --repo D:/image_to_text/ai_music_gen/ace_step_repo \
        --out outputs/audio_ref/cond_ref.safetensors
"""

import argparse
import json
import os
import sys

import torch
from safetensors.torch import load_file, save_file


def remap(sd):
    out = {}
    for k, v in sd.items():
        k = k.replace(".to_out.0.", ".o_proj.")
        k = k.replace(".to_q.", ".q_proj.")
        k = k.replace(".to_k.", ".k_proj.")
        k = k.replace(".to_v.", ".v_proj.")
        k = k.replace(".norm_q.", ".q_norm.")
        k = k.replace(".norm_k.", ".k_norm.")
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="G:/models/Audio")
    ap.add_argument("--repo", default="D:/image_to_text/ai_music_gen/ace_step_repo")
    ap.add_argument("--out", default="outputs/audio_ref/cond_ref.safetensors")
    ap.add_argument("--text-len", type=int, default=7)
    ap.add_argument("--lyric-len", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.repo not in sys.path:
        sys.path.insert(0, args.repo)

    try:
        import vector_quantize_pytorch  # noqa: F401
    except Exception:
        import types

        stub = types.ModuleType("vector_quantize_pytorch")

        class ResidualFSQ:  # noqa: D401
            def __init__(self, *a, **k):
                raise RuntimeError("ResidualFSQ stub")

        stub.ResidualFSQ = ResidualFSQ
        sys.modules["vector_quantize_pytorch"] = stub

    from acestep.models.common.configuration_acestep_v15 import AceStepConfig
    from acestep.models.turbo.modeling_acestep_v15_turbo import AceStepConditionEncoder

    cond_dir = os.path.join(args.model_dir, "condition_encoder")
    with open(os.path.join(cond_dir, "config.json")) as f:
        cfg = json.load(f)

    config = AceStepConfig(
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"],
        rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
        rope_theta=cfg.get("rope_theta", 1000000),
        attention_bias=cfg.get("attention_bias", False),
        use_sliding_window=True,
        sliding_window=cfg.get("sliding_window", 128),
        num_lyric_encoder_hidden_layers=cfg.get("num_lyric_encoder_hidden_layers", 8),
        num_timbre_encoder_hidden_layers=cfg.get("num_timbre_encoder_hidden_layers", 4),
        text_hidden_dim=cfg.get("text_hidden_dim", 1024),
        timbre_hidden_dim=cfg.get("timbre_hidden_dim", 64),
        audio_acoustic_hidden_dim=cfg.get("audio_acoustic_hidden_dim", 64),
        timbre_fix_frame=750,
        in_channels=cfg.get("in_channels", 192),
        model_version="turbo",
    )
    config._attn_implementation = "eager"

    sd = load_file(os.path.join(cond_dir, "diffusion_pytorch_model.safetensors"))
    sd = remap(sd)
    sd = {k: v.to(torch.float32) for k, v in sd.items()}

    model = AceStepConditionEncoder(config).to(torch.float32).eval()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing:", missing[:8])
    if unexpected:
        print("  unexpected:", unexpected[:8])

    g = torch.Generator().manual_seed(args.seed)
    text_hidden = torch.randn(1, args.text_len, 1024, generator=g, dtype=torch.float32)
    lyric_embeds = torch.randn(1, args.lyric_len, 1024, generator=g, dtype=torch.float32)
    ref_latents = sd["silence_latent"][:, :750, :].contiguous()
    order_mask = torch.zeros(1, dtype=torch.long)
    text_mask = torch.ones(1, args.text_len, dtype=torch.bool)
    lyric_mask = torch.ones(1, args.lyric_len, dtype=torch.bool)

    with torch.no_grad():
        text_out = model.text_projector(text_hidden)
        lyric_out = model.lyric_encoder(inputs_embeds=lyric_embeds, attention_mask=lyric_mask).last_hidden_state
        timbre_out, timbre_mask = model.timbre_encoder(ref_latents, order_mask)
        cond, cond_mask = model(
            text_hidden_states=text_hidden,
            text_attention_mask=text_mask,
            lyric_hidden_states=lyric_embeds,
            lyric_attention_mask=lyric_mask,
            refer_audio_acoustic_hidden_states_packed=ref_latents,
            refer_audio_order_mask=order_mask,
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_file(
        {
            "text_hidden": text_hidden.contiguous(),
            "lyric_embeds": lyric_embeds.contiguous(),
            "text_out": text_out.contiguous(),
            "lyric_out": lyric_out.contiguous(),
            "timbre_out": timbre_out.contiguous(),
            "condition": cond.contiguous(),
        },
        args.out,
    )
    print(f"[dumped] {os.path.abspath(args.out)}")
    print(f"  text_out   {tuple(text_out.shape)}")
    print(f"  lyric_out  {tuple(lyric_out.shape)}")
    print(f"  timbre_out {tuple(timbre_out.shape)}")
    print(f"  condition  {tuple(cond.shape)}  mean={cond.mean().item():.6f}")


if __name__ == "__main__":
    main()
