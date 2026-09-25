"""Dump a Stable Audio Open reference (T5, projection, DiT, scheduler, audio) for pure-Rust parity.

Reproduces `StableAudioPipeline` steps manually so every intermediate can be dumped.
"""

import argparse
import os
import sys

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import save_file

from diffusers import StableAudioPipeline
from diffusers.models.embeddings import get_1d_rotary_pos_embed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="G:/models/Audio/stable-audio-open-models")
    ap.add_argument("--prompt", default="The sound of a hammer hitting a wooden surface.")
    ap.add_argument("--negative", default="Low quality.")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--out", default="outputs/audio_ref/sao_ref.safetensors")
    args = ap.parse_args()

    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    dev = args.device

    pipe = StableAudioPipeline.from_pretrained(args.model_dir, torch_dtype=dtype).to(dev)
    pipe.set_progress_bar_config(disable=True)

    tok = pipe.tokenizer
    text_inputs = tok(
        args.prompt, padding="max_length", max_length=tok.model_max_length, truncation=True, return_tensors="pt"
    )
    ids = text_inputs.input_ids.to(dev)
    am = text_inputs.attention_mask.to(dev)

    uncond_input = tok(
        args.negative, padding="max_length", max_length=tok.model_max_length, truncation=True, return_tensors="pt"
    )
    n_ids = uncond_input.input_ids.to(dev)
    n_am = uncond_input.attention_mask.to(dev)

    with torch.no_grad():
        t5 = pipe.text_encoder(ids, attention_mask=am)[0]
        n_t5 = pipe.text_encoder(n_ids, attention_mask=n_am)[0]
        n_t5 = torch.where(n_am.to(torch.bool).unsqueeze(2), n_t5, torch.zeros_like(n_t5))

        proj_text = pipe.projection_model(text_hidden_states=t5).text_hidden_states
        proj_text = proj_text * am.unsqueeze(-1).to(proj_text.dtype)
        n_proj_text = pipe.projection_model(text_hidden_states=n_t5).text_hidden_states
        n_proj_text = n_proj_text * n_am.unsqueeze(-1).to(n_proj_text.dtype)

        start = torch.tensor([args.start], device=dev)
        end = torch.tensor([args.end], device=dev)
        so = pipe.projection_model(start_seconds=start, end_seconds=end)
        s0, e0 = so.seconds_start_hidden_states, so.seconds_end_hidden_states

        # CFG batch: [uncond, cond]
        cond = torch.cat([n_proj_text, proj_text], dim=0)
        glob = torch.cat([torch.cat([s0, e0], dim=2)] * 2, dim=0)  # [2,1,1536]
        text_audio = torch.cat([cond, s0.repeat(2, 1, 1), e0.repeat(2, 1, 1)], dim=1)  # [2, maxlen+2, 768]

        amp = am.unsqueeze(-1).to(proj_text.dtype)
        n_sig = pipe.scheduler.init_noise_sigma
        g = torch.Generator(dev).manual_seed(args.seed)
        sample_size = int(pipe.transformer.config.sample_size)
        latents = torch.randn((1, pipe.transformer.config.in_channels, sample_size), generator=g, device=dev, dtype=dtype)
        latents = latents * n_sig

        pipe.scheduler.set_timesteps(args.steps, device=dev)
        timesteps = pipe.scheduler.timesteps
        sigmas = pipe.scheduler.sigmas

        rotary = get_1d_rotary_pos_embed(pipe.rotary_embed_dim, sample_size + 1, use_real=True, repeat_interleave_real=False)

        step0_pred = None
        latents0 = latents.clone()
        for i, t in enumerate(timesteps):
            inp = torch.cat([latents] * 2)
            inp = pipe.scheduler.scale_model_input(inp, t)
            noise = pipe.transformer(
                inp, t.unsqueeze(0), encoder_hidden_states=text_audio, global_hidden_states=glob,
                rotary_embedding=rotary, return_dict=False,
            )[0]
            if i == 0:
                step0_pred = noise.clone()
            u, c = noise.chunk(2)
            noise = u + args.guidance * (c - u)
            latents = pipe.scheduler.step(noise, t, latents).prev_sample

        audio = pipe.vae.decode(latents).sample

    wav = audio[0].float().cpu().numpy().T
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sf.write(os.path.splitext(args.out)[0] + ".wav", wav, pipe.vae.config.sampling_rate)
    save_file(
        {
            "input_ids": ids.float().cpu(),
            "attention_mask": am.float().cpu(),
            "t5_hidden": t5.float().cpu(),
            "proj_text": proj_text.float().cpu(),
            "start_hidden": s0.float().cpu(),
            "end_hidden": e0.float().cpu(),
            "cond": text_audio.float().cpu(),
            "global": glob.float().cpu(),
            "sigmas": sigmas.float().cpu(),
            "timesteps": timesteps.float().cpu(),
            "step0_pred": step0_pred.float().cpu(),
            "latents0": latents0.float().cpu(),
            "final_latents": latents.float().cpu(),
            "audio": audio.float().cpu(),
        },
        args.out,
    )
    print(f"[dumped] {os.path.abspath(args.out)}")
    print(f"  ids {tuple(ids.shape)}  t5 {tuple(t5.shape)}  cond {tuple(text_audio.shape)}  global {tuple(glob.shape)}")
    print(f"  latents {tuple(latents.shape)}  audio {tuple(audio.shape)}  rotary_cos {tuple(rotary[0].shape)}")


if __name__ == "__main__":
    main()
