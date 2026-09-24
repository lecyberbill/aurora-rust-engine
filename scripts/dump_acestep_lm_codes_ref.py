"""Reference greedy audio-code generation (fixed CoT) for Rust comparison."""

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CAPTION = "warm uplifting acoustic pop, acoustic guitar, piano, drums, smooth vocal melody"
LYRICS = "[verse]\nShining like the morning sun, a brand new melody has just begun"
COT = "<think>\nbpm: 94\ncaption: warm uplifting acoustic pop.\nduration: 6\nkeyscale: C major\nlanguage: en\ntimesignature: 4\n</think>"
SYSTEM = "# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="G:/models/Audio/Ace-Step1.5/acestep-5Hz-lm-1.7B")
    ap.add_argument("--n", type=int, default=32)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16, device_map="cuda").eval()

    user = f"# Caption\n{CAPTION}\n\n# Lyric\n{LYRICS}\n"
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    ) + COT + "\n\n"

    allowed = []
    for i in range(model.config.vocab_size):
        s = tok.convert_ids_to_tokens(i)
        if isinstance(s, str) and s.startswith("<|audio_code_") and s.endswith("|>"):
            allowed.append(i)
    allowed_t = torch.tensor(allowed, device="cuda")
    print(f"allowed audio-code tokens: {len(allowed)}")

    ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
    out = []
    with torch.no_grad():
        for _ in range(args.n):
            logits = model(ids).logits[0, -1, :].float()
            sub = logits[allowed_t]
            nxt = int(allowed_t[torch.argmax(sub)])
            out.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], device="cuda")], dim=1)

    codes = []
    for i in out:
        s = tok.convert_ids_to_tokens(i)
        codes.append(int(s[len("<|audio_code_"):-2]))
    print("PY codes:", codes)
    os.makedirs("outputs/audio_ref", exist_ok=True)
    with open("outputs/audio_ref/lm_codes_ref.txt", "w") as f:
        f.write(",".join(str(c) for c in codes))


if __name__ == "__main__":
    main()
