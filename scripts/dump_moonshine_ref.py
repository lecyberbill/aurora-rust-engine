"""Dump a Moonshine reference (encoder hidden + greedy token ids) for pure-Rust parity."""

import argparse
import os

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from safetensors.torch import save_file
from transformers import AutoProcessor, MoonshineForConditionalGeneration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="G:/models/moonshine-tiny")
    ap.add_argument("--wav", default="outputs/audio_showcase/test_whisper_input.wav")
    ap.add_argument("--out", default="outputs/audio_ref/moonshine_ref.safetensors")
    args = ap.parse_args()

    model = MoonshineForConditionalGeneration.from_pretrained(args.model_dir, torch_dtype=torch.float32).eval()
    proc = AutoProcessor.from_pretrained(args.model_dir)

    audio, sr = sf.read(args.wav, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != 16000:
        from math import gcd

        g = gcd(sr, 16000)
        audio = resample_poly(audio, 16000 // g, sr // g).astype(np.float32)

    inputs = proc(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        enc = model.model.encoder(inputs.input_values).last_hidden_state
        ids = model.generate(**inputs, max_new_tokens=80, do_sample=False)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_file(
        {
            "input_values": inputs.input_values.float().cpu().contiguous(),
            "enc": enc.float().cpu().contiguous(),
            "ids": ids.float().cpu().contiguous(),
        },
        args.out,
    )
    print(f"[dumped] {args.out}")
    print("  input", tuple(inputs.input_values.shape), "enc", tuple(enc.shape), "ids", ids.tolist())
    print("  text:", proc.tokenizer.decode(ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
