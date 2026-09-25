"""Run the official ACE-Step repaint reference (base model) on the same source audio."""

import os
import shutil
import sys

REPO = "D:/image_to_text/ai_music_gen/ace_step_repo"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from acestep.handler import AceStepHandler  # noqa: E402
from acestep.inference import GenerationConfig, GenerationParams, generate_music  # noqa: E402


def main():
    handler = AceStepHandler()
    status, ok = handler.initialize_service(
        project_root=REPO, config_path="acestep-v15-base", device="cuda", offload_to_cpu=False
    )
    print(f"[init] {status} (ok={ok})")

    params = GenerationParams(
        task_type="repaint",
        src_audio="D:/image_to_text/TransRust/outputs/audio_showcase/codec_original.wav",
        caption="warm uplifting acoustic pop, acoustic guitar, piano, drums, smooth vocal melody",
        lyrics="[instrumental]",
        vocal_language="unknown",
        duration=6.0,
        inference_steps=50,
        guidance_scale=7.0,
        seed=42,
        repainting_start=2.08,
        repainting_end=3.88,
        instrumental=True,
    )
    config = GenerationConfig(batch_size=1, audio_format="ogg")
    res = generate_music(dit_handler=handler, llm_handler=None, params=params, config=config)
    print("success:", res.success, "error:", res.error)
    for a in res.audios or []:
        p = a.get("audio_path")
        print("audio:", p)
        if p and os.path.isfile(p):
            dst = "D:/image_to_text/TransRust/outputs/audio_showcase/repaint_ref.ogg"
            shutil.copyfile(p, dst)
            print("copied ->", dst)


if __name__ == "__main__":
    main()
