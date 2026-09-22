"""Dump reference ACE-Step 1.5 Turbo Flow-Matching Euler loop for Rust validation.

Reproduces the exact `AceStepConditionGenerationModel.generate_audio` sampler loop
for a fixed condition / context / noise, using the official DiT implementation.

Example:
    python scripts/dump_acestep_pipeline_ref.py --steps 8 --seq-len 24
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
        if k.startswith("proj_in_conv."):
            k = "proj_in.1." + k[len("proj_in_conv."):]
        elif k.startswith("proj_out_conv."):
            k = "proj_out.1." + k[len("proj_out_conv."):]
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
    ap.add_argument("--out", default="outputs/audio_ref/pipeline_ref.safetensors")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=24)
    ap.add_argument("--ctx-len", type=int, default=16)
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
    from acestep.models.turbo.modeling_acestep_v15_turbo import AceStepDiTModel

    trans_dir = os.path.join(args.model_dir, "transformer")
    with open(os.path.join(trans_dir, "config.json")) as f:
        cfg = json.load(f)

    config = AceStepConfig(
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"],
        rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
        rope_theta=cfg.get("rope_theta", 1000000),
        attention_bias=cfg.get("attention_bias", False),
        use_sliding_window=True,
        sliding_window=cfg.get("sliding_window", 128),
        layer_types=list(cfg["layer_types"]),
        audio_acoustic_hidden_dim=cfg["audio_acoustic_hidden_dim"],
        text_hidden_dim=cfg.get("text_hidden_dim", 1024),
        in_channels=cfg["in_channels"],
        patch_size=cfg["patch_size"],
        encoder_hidden_size=cfg.get("encoder_hidden_size", 2048),
        model_version=cfg.get("model_version", "turbo"),
        is_turbo=cfg.get("is_turbo", True),
    )
    config._attn_implementation = "eager"

    sd = {}
    for name in sorted(os.listdir(trans_dir)):
        if name.endswith(".safetensors"):
            sd.update(load_file(os.path.join(trans_dir, name)))
    sd = {k: v.to(torch.float32) for k, v in remap(sd).items()}
    model = AceStepDiTModel(config).to(torch.float32).eval()
    model.load_state_dict(sd, strict=False)

    g = torch.Generator().manual_seed(args.seed)
    T = args.seq_len
    noise = torch.randn(1, T, 64, generator=g, dtype=torch.float32)
    context_latents = torch.randn(1, T, 128, generator=g, dtype=torch.float32)
    condition = torch.randn(1, args.ctx_len, 2048, generator=g, dtype=torch.float32)
    ts = [1.0 - i / args.steps for i in range(args.steps)]
    attention_mask = torch.ones(1, T, dtype=torch.float32)

    def run(use_cache):
        from transformers.cache_utils import EncoderDecoderCache, DynamicCache

        xt = noise
        pkv = EncoderDecoderCache(DynamicCache(), DynamicCache()) if use_cache else None
        with torch.no_grad():
            for step, t_curr in enumerate(ts):
                t_tensor = torch.tensor([t_curr], dtype=torch.float32)
                out = model(
                    hidden_states=xt,
                    timestep=t_tensor,
                    timestep_r=t_tensor,
                    attention_mask=attention_mask,
                    encoder_hidden_states=condition,
                    encoder_attention_mask=None,
                    context_latents=context_latents,
                    use_cache=use_cache,
                    past_key_values=pkv,
                )
                vt = out[0]
                if use_cache:
                    pkv = out[1]
                if step == len(ts) - 1:
                    xt = xt - vt * t_curr
                else:
                    dt = t_curr - ts[step + 1]
                    xt = xt - vt * dt
        return xt

    final_nocache = run(False)
    final_cache = run(True)
    print(f"[cache-check] max|diff| (no-cache vs cached) = {(final_nocache - final_cache).abs().max().item():.3e}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_file(
        {
            "noise": noise.contiguous(),
            "context_latents": context_latents.contiguous(),
            "condition": condition.contiguous(),
            "t_schedule": torch.tensor(ts, dtype=torch.float32),
            "final_latents": final_nocache.contiguous(),
        },
        args.out,
    )
    print(f"[dumped] {os.path.abspath(args.out)}")
    print(f"  final_latents {tuple(final_nocache.shape)} mean={final_nocache.mean().item():.6f} std={final_nocache.std().item():.6f}")


if __name__ == "__main__":
    main()
