"""Dump a full ACE-Step 1.5 Turbo text2music reference (condition + noise + latents).

Uses the official `ace_step_repo` classes with the diffusers checkpoints remapped.
The dumped noise lets the Rust pipeline run with `generate_with_noise` and be
compared bit-for-bit against PyTorch.

Example:
    python scripts/dump_acestep_song_ref.py \
        --caption "Heavy driving rock stadium anthem, ..." \
        --lyrics-file scripts/lyrics_fr.txt --language fr \
        --duration 20 --steps 8 --seed 42 --out outputs/audio_ref/song_ref.safetensors
"""

import argparse
import json
import os
import sys

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, Qwen3Config, Qwen3Model

INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"


def remap_attn(sd):
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
    ap.add_argument("--model-dir", default="G:/models/Audio")
    ap.add_argument("--repo", default="D:/image_to_text/ai_music_gen/ace_step_repo")
    ap.add_argument("--caption", required=True)
    ap.add_argument("--lyrics-file", required=True)
    ap.add_argument("--language", default="en")
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/audio_ref/song_ref.safetensors")
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
    from acestep.models.turbo.modeling_acestep_v15_turbo import (
        AceStepConditionEncoder,
        AceStepDiTModel,
    )

    lyrics = open(args.lyrics_file, encoding="utf-8").read()

    # --- Prompts (must mirror the Rust pipeline exactly) ---
    metas = (
        "- bpm: N/A\n- timesignature: N/A\n- keyscale: N/A\n"
        f"- duration: {int(args.duration)} seconds\n"
    )
    text_prompt = f"# Instruction\n{INSTRUCTION}\n\n# Caption\n{args.caption}\n\n# Metas\n{metas}<|endoftext|>\n"
    lyrics_text = f"# Languages\n{args.language}\n\n# Lyric\n{lyrics}<|endoftext|>"

    te_dir = os.path.join(args.model_dir, "text_encoder")
    tok_dir = os.path.join(args.model_dir, "tokenizer")
    tok = AutoTokenizer.from_pretrained(tok_dir)
    text_ids = tok(text_prompt, truncation=True, max_length=256, return_tensors="pt").input_ids
    lyric_ids = tok(lyrics_text, truncation=True, max_length=2048, return_tensors="pt").input_ids

    qwen_cfg = Qwen3Config.from_pretrained(te_dir)
    qwen_cfg._attn_implementation = "eager"
    qwen = Qwen3Model.from_pretrained(te_dir, config=qwen_cfg, dtype=torch.float32).eval()

    with torch.no_grad():
        text_hidden = qwen(input_ids=text_ids, attention_mask=torch.ones_like(text_ids)).last_hidden_state
        lyric_embeds = qwen.embed_tokens(lyric_ids)

    # --- Condition encoder (repo class + remapped weights) ---
    cond_dir = os.path.join(args.model_dir, "condition_encoder")
    cond_cfg = json.load(open(os.path.join(cond_dir, "config.json")))
    cond_config = AceStepConfig(
        hidden_size=cond_cfg["hidden_size"],
        intermediate_size=cond_cfg["intermediate_size"],
        num_attention_heads=cond_cfg["num_attention_heads"],
        num_key_value_heads=cond_cfg["num_key_value_heads"],
        head_dim=cond_cfg["head_dim"],
        rms_norm_eps=cond_cfg.get("rms_norm_eps", 1e-6),
        rope_theta=cond_cfg.get("rope_theta", 1000000),
        attention_bias=cond_cfg.get("attention_bias", False),
        use_sliding_window=True,
        sliding_window=cond_cfg.get("sliding_window", 128),
        num_lyric_encoder_hidden_layers=cond_cfg.get("num_lyric_encoder_hidden_layers", 8),
        num_timbre_encoder_hidden_layers=cond_cfg.get("num_timbre_encoder_hidden_layers", 4),
        text_hidden_dim=cond_cfg.get("text_hidden_dim", 1024),
        timbre_hidden_dim=cond_cfg.get("timbre_hidden_dim", 64),
        audio_acoustic_hidden_dim=cond_cfg.get("audio_acoustic_hidden_dim", 64),
        timbre_fix_frame=750,
        in_channels=cond_cfg.get("in_channels", 192),
        model_version="turbo",
    )
    cond_config._attn_implementation = "eager"
    cond_sd = {k: v.to(torch.float32) for k, v in remap_attn(load_file(
        os.path.join(cond_dir, "diffusion_pytorch_model.safetensors"))).items()}
    silence_latent = cond_sd["silence_latent"]
    cond_model = AceStepConditionEncoder(cond_config).to(torch.float32).eval()
    cond_model.load_state_dict(cond_sd, strict=False)

    text_mask = torch.ones(1, text_ids.shape[1], dtype=torch.bool)
    lyric_mask = torch.ones(1, lyric_ids.shape[1], dtype=torch.bool)
    ref_latents = silence_latent[:, :750, :].contiguous()
    order_mask = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        condition, _ = cond_model(
            text_hidden_states=text_hidden,
            text_attention_mask=text_mask,
            lyric_hidden_states=lyric_embeds,
            lyric_attention_mask=lyric_mask,
            refer_audio_acoustic_hidden_states_packed=ref_latents,
            refer_audio_order_mask=order_mask,
        )

    # --- DiT ---
    trans_dir = os.path.join(args.model_dir, "transformer")
    trans_cfg = json.load(open(os.path.join(trans_dir, "config.json")))
    dit_config = AceStepConfig(
        hidden_size=trans_cfg["hidden_size"],
        intermediate_size=trans_cfg["intermediate_size"],
        num_hidden_layers=trans_cfg["num_hidden_layers"],
        num_attention_heads=trans_cfg["num_attention_heads"],
        num_key_value_heads=trans_cfg["num_key_value_heads"],
        head_dim=trans_cfg["head_dim"],
        rms_norm_eps=trans_cfg.get("rms_norm_eps", 1e-6),
        rope_theta=trans_cfg.get("rope_theta", 1000000),
        attention_bias=trans_cfg.get("attention_bias", False),
        use_sliding_window=True,
        sliding_window=trans_cfg.get("sliding_window", 128),
        layer_types=list(trans_cfg["layer_types"]),
        audio_acoustic_hidden_dim=trans_cfg["audio_acoustic_hidden_dim"],
        text_hidden_dim=trans_cfg.get("text_hidden_dim", 1024),
        in_channels=trans_cfg["in_channels"],
        patch_size=trans_cfg["patch_size"],
        encoder_hidden_size=trans_cfg.get("encoder_hidden_size", 2048),
        model_version="turbo",
        is_turbo=True,
    )
    dit_config._attn_implementation = "eager"
    dit_sd = {}
    for name in sorted(os.listdir(trans_dir)):
        if name.endswith(".safetensors"):
            dit_sd.update(load_file(os.path.join(trans_dir, name)))
    dit_sd = {k: v.to(torch.float32) for k, v in remap_dit(dit_sd).items()}
    dit = AceStepDiTModel(dit_config).to(torch.float32).eval()
    dit.load_state_dict(dit_sd, strict=False)

    # --- Latents, noise, Euler loop ---
    T = max(128, int(args.duration * 48000) // 1920)
    src = silence_latent[:, :T, :].contiguous().to(torch.float32)
    chunk = torch.ones(1, T, 64, dtype=torch.float32)
    context_latents = torch.cat([src, chunk], dim=-1)

    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    noise = torch.randn(1, T, 64, generator=gen, dtype=torch.float32)

    ts = [1.0 - i / args.steps for i in range(args.steps)]
    xt = noise
    attention_mask = torch.ones(1, T, dtype=torch.float32)
    with torch.no_grad():
        for step, t_curr in enumerate(ts):
            t_tensor = torch.tensor([t_curr], dtype=torch.float32)
            out = dit(
                hidden_states=xt,
                timestep=t_tensor,
                timestep_r=t_tensor,
                attention_mask=attention_mask,
                encoder_hidden_states=condition,
                encoder_attention_mask=None,
                context_latents=context_latents,
                use_cache=False,
            )
            vt = out[0]
            dt = t_curr if step == len(ts) - 1 else t_curr - ts[step + 1]
            xt = xt - vt * dt

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_file(
        {
            "condition": condition.contiguous(),
            "context_latents": context_latents.contiguous(),
            "noise": noise.contiguous(),
            "final_latents": xt.contiguous(),
        },
        args.out,
    )
    meta = {
        "caption": args.caption,
        "lyrics": lyrics,
        "language": args.language,
        "duration": args.duration,
        "steps": args.steps,
        "seed": args.seed,
        "T": T,
    }
    with open(os.path.join(os.path.dirname(os.path.abspath(args.out)), "song_ref.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    print(f"[dumped] {os.path.abspath(args.out)}")
    print(f"  condition {tuple(condition.shape)}  context {tuple(context_latents.shape)}  T={T}")
    print(f"  final_latents {tuple(xt.shape)} mean={xt.mean().item():.6f} std={xt.std().item():.6f}")


if __name__ == "__main__":
    main()
