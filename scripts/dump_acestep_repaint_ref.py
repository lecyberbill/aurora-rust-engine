"""Replay the ACE-Step repaint loop in PyTorch with the reference DiT using Rust-dumped inputs.

Compares the reference repaint latents to the Rust result (isolates the repaint sampler math).
"""

import argparse
import json
import os
import sys
import types

import torch
from safetensors.torch import load_file

REPO = "D:/image_to_text/ai_music_gen/ace_step_repo"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

try:
    import vector_quantize_pytorch  # noqa: F401
except Exception:
    stub = types.ModuleType("vector_quantize_pytorch")

    class ResidualFSQ:
        def __init__(self, *a, **k):
            raise RuntimeError("stub")

    stub.ResidualFSQ = ResidualFSQ
    sys.modules["vector_quantize_pytorch"] = stub

from acestep.models.common.apg_guidance import MomentumBuffer, apg_forward  # noqa: E402
from acestep.models.common.configuration_acestep_v15 import AceStepConfig  # noqa: E402
from acestep.models.base.modeling_acestep_v15_base import AceStepDiTModel  # noqa: E402


def remap_dit(sd):
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
    ap.add_argument("--ref", default="outputs/audio_ref/repaint_ref.safetensors")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--injection", type=float, default=0.5)
    ap.add_argument("--crossfade", type=int, default=10)
    args = ap.parse_args()

    ref = load_file(args.ref)
    ctx = ref["context_latents"].float()       # [1,T,128]
    rmask = ref["repaint_mask"].float()        # [1,T]
    noise = ref["noise"].float()               # [1,T,64]
    cond = ref["condition"].float()            # [1,L,2048]
    clean = ref["clean_src"].float()           # [1,T,64]
    ref_final = ref["final_latents"].float()   # [1,T,64]
    T = noise.shape[1]

    trans_dir = "G:/models/Audio-base/transformer"
    with open(os.path.join(trans_dir, "config.json")) as f:
        c = json.load(f)
    config = AceStepConfig(
        hidden_size=c["hidden_size"],
        intermediate_size=c["intermediate_size"],
        num_hidden_layers=c["num_hidden_layers"],
        num_attention_heads=c["num_attention_heads"],
        num_key_value_heads=c["num_key_value_heads"],
        head_dim=c["head_dim"],
        rms_norm_eps=c.get("rms_norm_eps", 1e-6),
        rope_theta=c.get("rope_theta", 1000000),
        attention_bias=c.get("attention_bias", False),
        use_sliding_window=True,
        sliding_window=c.get("sliding_window", 128),
        layer_types=list(c["layer_types"]),
        audio_acoustic_hidden_dim=c["audio_acoustic_hidden_dim"],
        text_hidden_dim=c.get("text_hidden_dim", 1024),
        in_channels=c["in_channels"],
        patch_size=c["patch_size"],
        encoder_hidden_size=c.get("encoder_hidden_size", 2048),
        model_version="base",
        is_turbo=False,
    )
    config._attn_implementation = "eager"
    sd = {}
    for name in sorted(os.listdir(trans_dir)):
        if name.endswith(".safetensors"):
            sd.update(load_file(os.path.join(trans_dir, name)))
    sd = {k: v.to(torch.float32) for k, v in remap_dit(sd).items()}
    dit = AceStepDiTModel(config).to(torch.float32).eval()
    dit.load_state_dict(sd, strict=False)

    ts = [1.0 - i / args.steps for i in range(args.steps)]
    xt = noise.clone()
    with torch.no_grad():
        for step, t_curr in enumerate(ts):
            t = torch.tensor([t_curr], dtype=torch.float32)
            v = dit(
                hidden_states=xt,
                timestep=t,
                timestep_r=t,
                attention_mask=torch.ones(1, T, dtype=torch.float32),
                encoder_hidden_states=cond,
                encoder_attention_mask=None,
                context_latents=ctx,
                use_cache=False,
            )[0]
            dt = t_curr if step == len(ts) - 1 else t_curr - ts[step + 1]
            xt = xt - v * dt
            if step < round(args.injection * args.steps):
                t_after = 0.0 if step == len(ts) - 1 else ts[step + 1]
                zt = t_after * noise + (1.0 - t_after) * clean
                xt = torch.where(rmask.unsqueeze(-1).bool(), xt, zt)
        # final boundary blend
        soft = rmask.clone()
        row = rmask[0]
        idx = torch.nonzero(row > 0.5, as_tuple=False).squeeze(-1)
        if idx.numel() > 0:
            left = int(idx[0]); right = int(idx[-1]) + 1
            fs = max(left - args.crossfade, 0)
            if left - fs > 0:
                soft[0, fs:left] = torch.linspace(0, 1, left - fs + 2)[1:-1]
            fe = min(right + args.crossfade, T)
            if fe - right > 0:
                soft[0, right:fe] = torch.linspace(1, 0, fe - right + 2)[1:-1]
        soft = soft.unsqueeze(-1)
        xt = soft * xt + (1.0 - soft) * clean

    d = (xt - ref_final).abs().max().item()
    print(f"[repaint-ref] final max|diff| (Py vs Rust) = {d:.3e}  T={T}")
    print(f"  py  mean={xt.mean().item():.6f}  rust mean={ref_final.mean().item():.6f}")


if __name__ == "__main__":
    main()
