"""Dump reference ACE-Step 1.5 Turbo DiT activations for bit-exact Rust validation.

Uses the *official* reference implementation (`ace_step_repo`) with the diffusers
checkpoint remapped to the repo's key layout. Run this with the same Python env
that runs the working example (needs torch + transformers + safetensors).

Example:
    python scripts/dump_acestep_dit_ref.py \
        --model-dir G:/models/Audio \
        --repo D:/image_to_text/ai_music_gen/ace_step_repo \
        --out outputs/audio_ref/dit_ref.safetensors
"""

import argparse
import json
import os
import sys

import torch
from safetensors.torch import load_file, save_file


def remap_diffusers_to_repo(sd):
    """Map the diffusers checkpoint keys onto the ace_step_repo module layout."""
    out = {}
    for k, v in sd.items():
        if k.startswith("proj_in_conv."):
            k = "proj_in.1." + k[len("proj_in_conv."):]
        elif k.startswith("proj_out_conv."):
            k = "proj_out.1." + k[len("proj_out_conv."):]
        # Attention submodule naming: diffusers `to_*`/`norm_*` -> HF-style `*_proj`/`*_norm`.
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
    ap.add_argument("--out", default="outputs/audio_ref/dit_ref.safetensors")
    ap.add_argument("--seq-len", type=int, default=24, help="unpatched latent frames T")
    ap.add_argument("--ctx-len", type=int, default=16, help="conditioning tokens")
    ap.add_argument("--timestep", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.repo not in sys.path:
        sys.path.insert(0, args.repo)

    # `modeling_acestep_v15_turbo` imports ResidualFSQ at module level, but the DiT
    # dump never instantiates the audio tokenizer. Stub it if it is not installed.
    try:
        import vector_quantize_pytorch  # noqa: F401
    except Exception:
        import types

        stub = types.ModuleType("vector_quantize_pytorch")

        class ResidualFSQ:  # noqa: D401
            def __init__(self, *a, **k):
                raise RuntimeError("ResidualFSQ stub: not needed for DiT dump")

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

    # Load + merge shards (float32 on CPU for deterministic comparison).
    sd = {}
    for name in sorted(os.listdir(trans_dir)):
        if name.endswith(".safetensors"):
            sd.update(load_file(os.path.join(trans_dir, name)))
    sd = remap_diffusers_to_repo(sd)
    sd = {k: v.to(torch.float32) for k, v in sd.items()}

    model = AceStepDiTModel(config).to(torch.float32).eval()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    if unexpected:
        print("  unexpected:", unexpected[:8])
    if missing:
        print("  missing:", missing[:8])

    g = torch.Generator().manual_seed(args.seed)
    T = args.seq_len
    ctx_len = args.ctx_len

    context_latents = torch.randn(1, T, 128, generator=g, dtype=torch.float32)
    hidden_states = torch.randn(1, T, 64, generator=g, dtype=torch.float32)
    encoder_hidden_states = torch.randn(1, ctx_len, 2048, generator=g, dtype=torch.float32)
    timestep = torch.tensor([args.timestep], dtype=torch.float32)

    # Direct module probes (match the Rust unit-level checks).
    temb_t, proj_t = model.time_embed(timestep)
    temb_r, proj_r = model.time_embed_r(timestep - timestep)

    with torch.no_grad():
        out = model(
            hidden_states=hidden_states,
            timestep=timestep,
            timestep_r=timestep,
            attention_mask=torch.ones(1, T, dtype=torch.float32),
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=None,
            context_latents=context_latents,
            use_cache=False,
        )
    output = out[0] if isinstance(out, (tuple, list)) else out

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_file(
        {
            "context_latents": context_latents.contiguous(),
            "hidden_states": hidden_states.contiguous(),
            "encoder_hidden_states": encoder_hidden_states.contiguous(),
            "timestep": timestep.contiguous(),
            "temb_t": temb_t.contiguous(),
            "proj_t": proj_t.contiguous(),
            "temb_sum": (temb_t + temb_r).contiguous(),
            "proj_sum": (proj_t + proj_r).contiguous(),
            "output": output.contiguous(),
        },
        args.out,
    )
    print(f"[dumped] {os.path.abspath(args.out)}")
    print(f"  context_latents {tuple(context_latents.shape)}")
    print(f"  hidden_states   {tuple(hidden_states.shape)}")
    print(f"  condition       {tuple(encoder_hidden_states.shape)}")
    print(f"  output          {tuple(output.shape)}  mean={output.mean().item():.6f} std={output.std().item():.6f}")


if __name__ == "__main__":
    main()
