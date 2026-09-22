"""Convert an ACE-Step single-file repo checkpoint (e.g. acestep-v15-base) into the
diffusers-style folder layout consumed by the Rust engine.

Produces:
    <out>/transformer/diffusion_pytorch_model.safetensors + config.json
    <out>/condition_encoder/diffusion_pytorch_model.safetensors + config.json
    plus copies of text_encoder/ tokenizer/ vae/ scheduler/ from <shared>.

Example:
    python scripts/convert_acestep_base.py \
        --checkpoint "<repo>/checkpoints/acestep-v15-base" \
        --shared G:/models/Audio \
        --out G:/models/Audio-base
"""

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file


def load_checkpoint(path):
    """Load a repo checkpoint: single `model.safetensors` or sharded via the index."""
    single = os.path.join(path, "model.safetensors")
    if os.path.isfile(single):
        return load_file(single)
    index = os.path.join(path, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index) as f:
            weight_map = json.load(f)["weight_map"]
        sd = {}
        for shard in sorted(set(weight_map.values())):
            print(f"  shard {shard}")
            sd.update(load_file(os.path.join(path, shard)))
        return sd
    raise FileNotFoundError(f"No model.safetensors or index found in {path}")


def attn_to_diffusers(k: str) -> str:
    k = k.replace(".o_proj.", ".to_out.0.")
    k = k.replace(".q_proj.", ".to_q.")
    k = k.replace(".k_proj.", ".to_k.")
    k = k.replace(".v_proj.", ".to_v.")
    k = k.replace(".q_norm.", ".norm_q.")
    k = k.replace(".k_norm.", ".norm_k.")
    return k


def convert_transformer(sd):
    out = {}
    for k, v in sd.items():
        if not k.startswith("decoder."):
            continue
        k = k[len("decoder."):]
        k = k.replace("proj_in.1.", "proj_in_conv.")
        k = k.replace("proj_out.1.", "proj_out_conv.")
        out[attn_to_diffusers(k)] = v
    return out


def convert_condition_encoder(sd, silence_latent):
    out = {}
    for k, v in sd.items():
        if k == "null_condition_emb":
            out["null_condition_emb"] = v
        elif k.startswith("encoder."):
            out[attn_to_diffusers(k[len("encoder."):])] = v
    out["silence_latent"] = silence_latent
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--shared", default="G:/models/Audio")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-dtype", choices=["bf16", "f16", "f32"], default=None,
                    help="Optional downcast for the written weights (e.g. XL f32 -> bf16).")
    args = ap.parse_args()

    def cast(sd):
        if args.save_dtype is None:
            return sd
        dt = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}[args.save_dtype]
        return {k: v.to(dt) for k, v in sd.items()}

    os.makedirs(os.path.join(args.out, "transformer"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "condition_encoder"), exist_ok=True)

    print("Loading checkpoint ...")
    sd = load_checkpoint(args.checkpoint)
    with open(os.path.join(args.checkpoint, "config.json")) as f:
        cfg = json.load(f)

    # silence_latent from the torch file (f32 [1, T, 64]).
    sil = torch.load(os.path.join(args.checkpoint, "silence_latent.pt"), map_location="cpu")
    if isinstance(sil, dict):
        sil = next(iter(sil.values()))
    sil = sil.to(torch.float32)
    if sil.dim() == 2:
        sil = sil.unsqueeze(0)
    # The .pt stores [B, C, T]; the handler transposes to [B, T, C].
    sil = sil.transpose(1, 2).contiguous()
    print(f"  silence_latent {tuple(sil.shape)}")

    print("Writing transformer ...")
    trans = cast(convert_transformer(sd))
    save_file(trans, os.path.join(args.out, "transformer", "diffusion_pytorch_model.safetensors"))
    with open(os.path.join(args.out, "transformer", "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print("Writing condition_encoder ...")
    cond = cast(convert_condition_encoder(sd, sil))
    save_file(cond, os.path.join(args.out, "condition_encoder", "diffusion_pytorch_model.safetensors"))
    with open(os.path.join(args.out, "condition_encoder", "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # Shared components (text encoder, tokenizer, VAE, scheduler).
    for sub in ("text_encoder", "tokenizer", "vae", "scheduler"):
        src = os.path.join(args.shared, sub)
        dst = os.path.join(args.out, sub)
        if os.path.isdir(src) and not os.path.isdir(dst):
            print(f"Copying {sub} ...")
            shutil.copytree(src, dst)

    print(f"[done] {os.path.abspath(args.out)}")
    print(f"  transformer keys={len(trans)}  condition_encoder keys={len(cond)}")
    print(f"  is_turbo={cfg.get('is_turbo')}  hidden={cfg.get('hidden_size')}  layers={cfg.get('num_hidden_layers')}")


if __name__ == "__main__":
    main()
