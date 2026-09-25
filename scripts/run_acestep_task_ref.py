"""Run the official ACE-Step reference for a source-audio task and copy the output."""

import os
import shutil
import sys

REPO = "D:/image_to_text/ai_music_gen/ace_step_repo"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch  # noqa: E402

# transformers `from_pretrained` builds the model under `torch.device("meta")`,
# which makes `ResidualFSQ.__init__`'s `(levels_tensor > 1).all()` assert call
# `.item()` on a meta tensor. Force the FSQ construction onto CPU.
try:
    import vector_quantize_pytorch.residual_fsq as _rfsq  # noqa: E402

    _orig_fsq_init = _rfsq.ResidualFSQ.__init__

    def _patched_fsq_init(self, *a, **k):
        with torch.device("cpu"):
            _orig_fsq_init(self, *a, **k)

    _rfsq.ResidualFSQ.__init__ = _patched_fsq_init
except Exception as _exc:  # pragma: no cover
    print("FSQ patch skipped:", _exc)

from acestep.handler import AceStepHandler  # noqa: E402
from acestep.inference import GenerationConfig, GenerationParams, generate_music  # noqa: E402

SRC = "D:/image_to_text/TransRust/outputs/audio_showcase/codec_original.wav"
CAPTION = "warm uplifting acoustic pop, acoustic guitar, piano, drums, smooth vocal melody"

TASKS = {
    "text2music": dict(instruction="Fill the audio semantic mask based on the given conditions:"),
    "extract": dict(instruction="Extract the VOCALS track from the audio:"),
    "lego": dict(instruction="Generate the DRUMS track based on the audio context:"),
    "complete": dict(instruction="Complete the input track with VOCALS | GUITAR:"),
}


def run(task, model="acestep-v15-base", chunk_mask_mode="auto"):
    device = os.environ.get("DEVICE", "cuda")
    offload = os.environ.get("OFFLOAD", "0") == "1"
    handler = AceStepHandler()
    status, ok = handler.initialize_service(
        project_root=REPO, config_path=model, device=device, offload_to_cpu=offload
    )
    print(f"[init] {status} (ok={ok}, device={device})")

    if os.environ.get("ACESTEP_DBG"):
        import torch as _t

        orig_prep = handler._prepare_text_conditioning_inputs

        def prep(*a, **k):
            res = orig_prep(*a, **k)
            print("[DBG] text_inputs[0]:", repr(res[0][0][:400]))
            return res

        handler._prepare_text_conditioning_inputs = prep

        orig_masks = handler._build_chunk_masks_and_src_latents

        def masks(*a, **k):
            cm, spans, is_cov, src, rm = orig_masks(*a, **k)
            if os.environ.get("FORCE_COVER"):
                is_cov = torch.ones_like(is_cov)
                print("[DBG] FORCE_COVER -> is_covers", is_cov.tolist())
            print("[DBG] chunk uniq:", _t.unique(cm).tolist(), "spans:", spans,
                  "is_covers:", is_cov.tolist(), "src|mean|:", float(src.abs().mean()),
                  "repaint_mask:", "set" if rm is not None else None)
            return cm, spans, is_cov, src, rm

        handler._build_chunk_masks_and_src_latents = masks

        orig_ga = handler.model.generate_audio

        def ga(**kw):
            print("[DBG] generate_audio:", {
                k: (tuple(v.shape) if _t.is_tensor(v) else v)
                for k, v in kw.items()
                if k in ("src_latents", "chunk_masks", "is_covers", "infer_steps",
                         "diffusion_guidance_scale", "audio_cover_strength",
                         "shift", "cover_noise_strength", "is_covers")
            })
            _t.save({k: v.detach().cpu() for k, v in kw.items() if _t.is_tensor(v) and v.numel() < 5_000_000},
                    f"D:/image_to_text/TransRust/outputs/audio_ref/{task}_ga.pt")
            out = orig_ga(**kw)
            _t.save({"target_latents": out["target_latents"].detach().cpu()},
                    f"D:/image_to_text/TransRust/outputs/audio_ref/{task}_final.pt")
            return out

        handler.model.generate_audio = ga

        orig_pc = handler.model.prepare_condition

        def pc(**kw):
            eh, eam, ctx = orig_pc(**kw)
            _t.save({"enc": eh.detach().cpu(), "ctx": ctx.detach().cpu()},
                    f"D:/image_to_text/TransRust/outputs/audio_ref/{task}_cond.pt")
            return eh, eam, ctx

        handler.model.prepare_condition = pc

        orig_pn = handler.model.prepare_noise

        def pn(ctx, seed=None):
            n = orig_pn(ctx, seed)
            _t.save({"noise": n.detach().cpu()}, f"D:/image_to_text/TransRust/outputs/audio_ref/{task}_noise.pt")
            return n

        handler.model.prepare_noise = pn
    params = GenerationParams(
        task_type=task,
        src_audio=None if task == "text2music" else SRC,
        reference_audio=SRC if (os.environ.get("REF_SRC") and task != "text2music") else None,
        caption=CAPTION,
        lyrics="[instrumental]",
        vocal_language="unknown",
        duration=30.0 if task == "text2music" else 6.0,
        inference_steps=50,
        guidance_scale=7.0,
        seed=42,
        instrumental=True,
        instruction=TASKS[task]["instruction"],
        chunk_mask_mode=chunk_mask_mode,
    )
    config = GenerationConfig(batch_size=1, audio_format="wav")
    res = generate_music(dit_handler=handler, llm_handler=None, params=params, config=config)
    print("success:", res.success, "error:", res.error)
    for a in res.audios or []:
        print("keys:", list(a.keys()) if isinstance(a, dict) else type(a))
        p = a.get("audio_path") or a.get("path") if isinstance(a, dict) else None
        print("audio_path:", p)
        if p and os.path.isfile(p):
            dst = f"D:/image_to_text/TransRust/outputs/audio_showcase/ref_{task}.wav"
            shutil.copyfile(p, dst)
            print("copied ->", dst)
            continue
        # Fall back to the raw tensor.
        import numpy as np
        import soundfile as sf

        audio = a.get("audio") if isinstance(a, dict) else a
        if audio is None and isinstance(a, dict):
            audio = a.get("tensor")
        if audio is None:
            continue
        audio = audio.detach().cpu().float().numpy() if hasattr(audio, "detach") else np.asarray(audio)
        if audio.ndim == 2 and audio.shape[0] == 2:
            audio = audio.T
        dst = f"D:/image_to_text/TransRust/outputs/audio_showcase/ref_{task}.wav"
        sf.write(dst, audio, 48000)
        print("wrote ->", dst, audio.shape)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--model", default="acestep-v15-base")
    ap.add_argument("--chunk-mask-mode", default="auto")
    args = ap.parse_args()
    run(args.task, args.model, args.chunk_mask_mode)
