"""Dump reference Qwen3 text-encoder activations (ACE-Step caption prompt) for Rust validation.

Example:
    python scripts/dump_acestep_qwen_ref.py \
        --model-dir G:/models/Audio \
        --out outputs/audio_ref/qwen_ref.safetensors
"""

import argparse
import os

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer, Qwen3Config, Qwen3Model

CAPTION_PROMPT = (
    "# Instruction\n"
    "Fill the audio semantic mask based on the given conditions:\n\n"
    "# Caption\n"
    "[genre: acoustic pop] warm uplifting acoustic pop song\n\n"
    "# Metas\n"
    "- bpm: N/A\n- timesignature: N/A\n- keyscale: N/A\n- duration: 5 seconds\n<|endoftext|>\n"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="G:/models/Audio")
    ap.add_argument("--out", default="outputs/audio_ref/qwen_ref.safetensors")
    args = ap.parse_args()

    te_dir = os.path.join(args.model_dir, "text_encoder")
    tok_dir = os.path.join(args.model_dir, "tokenizer")

    tok = AutoTokenizer.from_pretrained(tok_dir)
    text_ids = tok(CAPTION_PROMPT, padding="longest", truncation=True, max_length=256, return_tensors="pt").input_ids
    lyric_ids = tok("Shining like the morning sun", truncation=True, max_length=64, return_tensors="pt").input_ids

    cfg = Qwen3Config.from_pretrained(te_dir)
    cfg._attn_implementation = "eager"
    model = Qwen3Model.from_pretrained(te_dir, config=cfg, dtype=torch.float32).eval()

    with torch.no_grad():
        last_hidden = model(input_ids=text_ids, attention_mask=torch.ones_like(text_ids)).last_hidden_state
        lyric_embeds = model.embed_tokens(lyric_ids)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_file(
        {
            "text_ids": text_ids.to(torch.float32).contiguous(),
            "last_hidden": last_hidden.contiguous(),
            "lyric_ids": lyric_ids.to(torch.float32).contiguous(),
            "lyric_embeds": lyric_embeds.contiguous(),
        },
        args.out,
    )
    print(f"[dumped] {os.path.abspath(args.out)}")
    print(f"  text_ids {tuple(text_ids.shape)}  lyric_ids {tuple(lyric_ids.shape)}")
    print(f"  last_hidden {tuple(last_hidden.shape)} mean={last_hidden.mean().item():.6f}")


if __name__ == "__main__":
    main()
