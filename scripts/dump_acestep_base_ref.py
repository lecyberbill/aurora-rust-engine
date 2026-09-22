"""Dump a full ACE-Step base (non-turbo) text2music reference (condition + noise + latents).

Uses the official `ace_step_repo` **base** classes with the single-file checkpoint
(keys `encoder.*` / `decoder.*`).

Example:
    python scripts/dump_acestep_base_ref.py \
        --checkpoint "<repo>/checkpoints/acestep-v15-base" \
        --text-encoder G:/models/Audio/text_encoder --tokenizer G:/models/Audio/tokenizer \
        --caption "..." --lyrics-file scripts/lyrics_fr.txt --language fr \
        --duration 10 --steps 30 --guidance 7.0 --seed 42 \
        --out outputs/audio_ref/base_ref.safetensors
"""

import argparse
import json
import os
import sys

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, Qwen3Config, Qwen3Model

INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--modeling", default="base", choices=["base", "sft", "xl_base", "xl_turbo"])
    ap.add_argument("--repo", default="D:/image_to_text/ai_music_gen/ace_step_repo")
    ap.add_argument("--text-encoder", default="G:/models/Audio/text_encoder")
    ap.add_argument("--tokenizer", default="G:/models/Audio/tokenizer")
    ap.add_argument("--caption", required=True)
    ap.add_argument("--lyrics-file", required=True)
    ap.add_argument("--language", default="en")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/audio_ref/base_ref.safetensors")
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

    import importlib

    from acestep.models.common.apg_guidance import MomentumBuffer, apg_forward
    from acestep.models.common.configuration_acestep_v15 import AceStepConfig

    module = {
        "base": "acestep.models.base.modeling_acestep_v15_base",
        "sft": "acestep.models.sft.modeling_acestep_v15_base",
        "xl_base": "acestep.models.xl_base.modeling_acestep_v15_xl_base",
        "xl_turbo": "acestep.models.xl_turbo.modeling_acestep_v15_xl_turbo",
    }[args.modeling]
    mod = importlib.import_module(module)
    AceStepConditionEncoder = mod.AceStepConditionEncoder
    AceStepDiTModel = mod.AceStepDiTModel

    def load_checkpoint(path):
        single = os.path.join(path, "model.safetensors")
        if os.path.isfile(single):
            return load_file(single)
        index = os.path.join(path, "model.safetensors.index.json")
        with open(index) as f:
            weight_map = json.load(f)["weight_map"]
        sd = {}
        for shard in sorted(set(weight_map.values())):
            sd.update(load_file(os.path.join(path, shard)))
        return sd

    lyrics = open(args.lyrics_file, encoding="utf-8").read()

    metas = (
        "- bpm: N/A\n- timesignature: N/A\n- keyscale: N/A\n"
        f"- duration: {int(args.duration)} seconds\n"
    )
    text_prompt = f"# Instruction\n{INSTRUCTION}\n\n# Caption\n{args.caption}\n\n# Metas\n{metas}<|endoftext|>\n"
    lyrics_text = f"# Languages\n{args.language}\n\n# Lyric\n{lyrics}<|endoftext|>"

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    text_ids = tok(text_prompt, truncation=True, max_length=256, return_tensors="pt").input_ids
    lyric_ids = tok(lyrics_text, truncation=True, max_length=2048, return_tensors="pt").input_ids

    qwen_cfg = Qwen3Config.from_pretrained(args.text_encoder)
    qwen_cfg._attn_implementation = "eager"
    qwen = Qwen3Model.from_pretrained(args.text_encoder, config=qwen_cfg, dtype=torch.float32).eval()
    with torch.no_grad():
        text_hidden = qwen(input_ids=text_ids, attention_mask=torch.ones_like(text_ids)).last_hidden_state
        lyric_embeds = qwen.embed_tokens(lyric_ids)

    with open(os.path.join(args.checkpoint, "config.json")) as f:
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
        use_sliding_window=c.get("use_sliding_window", True),
        sliding_window=c.get("sliding_window", 128),
        layer_types=list(c["layer_types"]),
        num_lyric_encoder_hidden_layers=c.get("num_lyric_encoder_hidden_layers", 8),
        num_timbre_encoder_hidden_layers=c.get("num_timbre_encoder_hidden_layers", 4),
        text_hidden_dim=c.get("text_hidden_dim", 1024),
        timbre_hidden_dim=c.get("timbre_hidden_dim", 64),
        audio_acoustic_hidden_dim=c.get("audio_acoustic_hidden_dim", 64),
        timbre_fix_frame=c.get("timbre_fix_frame", 750),
        in_channels=c.get("in_channels", 192),
        patch_size=c.get("patch_size", 2),
        encoder_hidden_size=c.get("encoder_hidden_size", c["hidden_size"]),
        encoder_intermediate_size=c.get("encoder_intermediate_size", c["intermediate_size"]),
        encoder_num_attention_heads=c.get("encoder_num_attention_heads", c["num_attention_heads"]),
        encoder_num_key_value_heads=c.get("encoder_num_key_value_heads", c["num_key_value_heads"]),
        is_turbo=bool(c.get("is_turbo", False)),
        model_version="base",
    )
    config._attn_implementation = "eager"

    # XL models have a separate (narrower) encoder config.
    import copy
    encoder_config = copy.deepcopy(config)
    encoder_config.hidden_size = c.get("encoder_hidden_size", c["hidden_size"])
    encoder_config.intermediate_size = c.get("encoder_intermediate_size", c["intermediate_size"])
    encoder_config.num_attention_heads = c.get("encoder_num_attention_heads", c["num_attention_heads"])
    encoder_config.num_key_value_heads = c.get("encoder_num_key_value_heads", c["num_key_value_heads"])
    encoder_config._attn_implementation = "eager"

    sd = load_checkpoint(args.checkpoint)
    enc_sd = {k[len("encoder."):]: v.to(torch.float32) for k, v in sd.items() if k.startswith("encoder.")}
    dec_sd = {k[len("decoder."):]: v.to(torch.float32) for k, v in sd.items() if k.startswith("decoder.")}
    null_cond = sd["null_condition_emb"].to(torch.float32)

    sil = torch.load(os.path.join(args.checkpoint, "silence_latent.pt"), map_location="cpu").transpose(1, 2).to(torch.float32)

    encoder = AceStepConditionEncoder(encoder_config).to(torch.float32).eval()
    encoder.load_state_dict(enc_sd, strict=False)
    decoder = AceStepDiTModel(config).to(torch.float32).eval()
    decoder.load_state_dict(dec_sd, strict=False)

    with torch.no_grad():
        condition, _ = encoder(
            text_hidden_states=text_hidden,
            text_attention_mask=torch.ones(1, text_ids.shape[1], dtype=torch.bool),
            lyric_hidden_states=lyric_embeds,
            lyric_attention_mask=torch.ones(1, lyric_ids.shape[1], dtype=torch.bool),
            refer_audio_acoustic_hidden_states_packed=sil[:, :750, :].contiguous(),
            refer_audio_order_mask=torch.zeros(1, dtype=torch.long),
        )

    T = max(128, int(args.duration * 48000) // 1920)
    src = sil[:, :T, :].contiguous()
    context_latents = torch.cat([src, torch.ones(1, T, 64)], dim=-1)

    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    noise = torch.randn(1, T, 64, generator=gen, dtype=torch.float32)

    null_expanded = null_cond.expand_as(condition)
    cond2 = torch.cat([condition, null_expanded], dim=0)
    ctx2 = torch.cat([context_latents, context_latents], dim=0)
    ts = torch.linspace(1.0, 0.0, args.steps + 1, dtype=torch.float32)
    xt = noise
    momentum = MomentumBuffer()
    v_raw0 = None
    with torch.no_grad():
        for i in range(args.steps):
            t_curr = ts[i].item()
            t_next = ts[i + 1].item()
            t_tensor = torch.full((2,), t_curr, dtype=torch.float32)
            x2 = torch.cat([xt, xt], dim=0)
            v = decoder(
                hidden_states=x2,
                timestep=t_tensor,
                timestep_r=t_tensor,
                attention_mask=torch.ones(2, T, dtype=torch.float32),
                encoder_hidden_states=cond2,
                encoder_attention_mask=None,
                context_latents=ctx2,
                use_cache=False,
            )[0]
            if i == 0:
                v_raw0 = v.clone()
            pred_cond, pred_uncond = v.chunk(2)
            if i == 0:
                print(
                    f"[raw0] |cond|={pred_cond.abs().mean().item():.6f} "
                    f"|unc|={pred_uncond.abs().mean().item():.6f} "
                    f"|diff|={(pred_cond - pred_uncond).abs().mean().item():.6f}"
                )
            vt = apg_forward(pred_cond, pred_uncond, args.guidance, momentum_buffer=momentum, dims=[1])
            xt = xt - vt * (t_curr - t_next)
            print(f"[step {i}] |vt|={vt.abs().mean().item():.6f} |xt|={xt.abs().mean().item():.6f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_file(
        {
            "condition": condition.contiguous(),
            "context_latents": context_latents.contiguous(),
            "noise": noise.contiguous(),
            "final_latents": xt.contiguous(),
            "v_raw0": v_raw0.transpose(1, 2).contiguous(),
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
        "guidance": args.guidance,
        "T": T,
    }
    with open(os.path.join(os.path.dirname(os.path.abspath(args.out)), "base_ref.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    print(f"[dumped] {os.path.abspath(args.out)}")
    print(f"  condition {tuple(condition.shape)}  context {tuple(context_latents.shape)}  T={T}")
    print(f"  final_latents {tuple(xt.shape)} mean={xt.mean().item():.6f} std={xt.std().item():.6f}")


if __name__ == "__main__":
    main()
