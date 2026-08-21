# [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 1 (per-model try-catch isolation) | Action: Pass 2 Python Diffusers Benchmark Script with UTF-8 support

import os
import sys
import time
import json
from pathlib import Path

# Force UTF-8 encoding on Windows stdout
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler

STRIKING_PROMPT = "masterpiece, best quality, ultra-detailed, 1girl, solo, cyberpunk samurai warrior, glowing cherry blossoms, rainy futuristic neo-tokyo street, volumetric lighting, intricate reflections, 8k resolution, cinematic masterpiece"
NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"

TARGET_MODELS = [
    r"G:\models\checkpoints\animaPencilXL_v100.safetensors",
    r"G:\models\checkpoints\aniverseXL_v30.safetensors",
    r"G:\models\checkpoints\babesByStableYogiPony_v50.safetensors",
    r"G:\models\checkpoints\Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
    r"G:\models\checkpoints\betterThanWords_v30.safetensors",
    r"G:\models\checkpoints\bigLove_ponyV20.safetensors",
    r"G:\models\checkpoints\realismarkPlus_realismarkPlus.safetensors",
    r"G:\models\checkpoints\CHEYENNE_v20.safetensors",
    r"G:\models\checkpoints\colossusProjectXLSFW_10bNeodemonFP16.safetensors",
    r"G:\models\checkpoints\CyberRealisticPony_V7a.safetensors",
    r"G:\models\checkpoints\dreamshaperXL_turboDpmppSDEKarras.safetensors",
    r"G:\models\checkpoints\DreamShaperXL_Turbo_v2_1.safetensors",
    r"G:\models\checkpoints\duchaitenAiartSDXL_v33515LightningTCD.safetensors",
    r"G:\models\checkpoints\duchaitenPonyXLNo_v60.safetensors",
    r"G:\models\checkpoints\eldgardKinkiestModel_v20.safetensors",
]

def main():
    output_dir = Path("outputs/stress_test/python")
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path("outputs/stress_test/python_stress_test_log.md")
    json_path = Path("outputs/stress_test/python_metrics.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    print(f"[*] Python Device: {device} ({device_name})", flush=True)

    seeds = [42, 1337]
    num_steps = 30
    guidance_scale = 6.0
    width = 1024
    height = 1024

    model_results = []

    print("============================================================", flush=True)
    print("Starting Pass 2: Python Diffusers Stress Test (15 Models)", flush=True)
    print(f"   Device: {device_name} | Precision: torch.float16 | Steps: {num_steps}", flush=True)
    print("============================================================", flush=True)

    for idx, model_path_str in enumerate(TARGET_MODELS):
        path = Path(model_path_str)
        model_name = path.name
        file_size_gb = path.stat().st_size / (1024**3) if path.exists() else 0.0

        print(f"\n[{idx + 1}/{len(TARGET_MODELS)}] Model: {model_name} ({file_size_gb:.2f} GB)", flush=True)

        if not path.exists():
            print("  [-] File does not exist, skipping.", flush=True)
            model_results.append({
                "model_name": model_name,
                "model_size_gb": file_size_gb,
                "status": "File Not Found",
                "load_time_sec": 0.0,
                "images": []
            })
            continue

        # Load pipeline from single file
        t_load_start = time.perf_counter()
        try:
            pipe = StableDiffusionXLPipeline.from_single_file(
                str(path),
                torch_dtype=torch.float16,
                use_safetensors=True
            )
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
            if hasattr(pipe, "enable_vae_tiling"):
                pipe.enable_vae_tiling()
            elif hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling()
            pipe.enable_model_cpu_offload()
            load_sec = time.perf_counter() - t_load_start
            print(f"  [+] Weights loaded in {load_sec:.2f}s", flush=True)
        except Exception as e:
            print(f"  [-] Load Error: {e}", flush=True)
            model_results.append({
                "model_name": model_name,
                "model_size_gb": file_size_gb,
                "status": f"Load Error: {e}",
                "load_time_sec": time.perf_counter() - t_load_start,
                "images": []
            })
            continue

        image_results = []

        for seed in seeds:
            print(f"  [*] Generating image with seed {seed} ({num_steps} steps)...", flush=True)
            generator = torch.Generator(device=device).manual_seed(seed)
            t_gen_start = time.perf_counter()

            try:
                result = pipe(
                    prompt=STRIKING_PROMPT,
                    negative_prompt=NEGATIVE_PROMPT,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    generator=generator
                )

                duration = time.perf_counter() - t_gen_start
                it_per_sec = num_steps / duration
                clean_name = model_name.replace(".safetensors", "")
                out_filename = f"{clean_name}_seed{seed}.png"
                out_path = output_dir / out_filename

                result.images[0].save(out_path)
                print(f"    [+] Completed in {duration:.2f}s ({it_per_sec:.2f} it/s) -> {out_filename}", flush=True)

                image_results.append({
                    "seed": seed,
                    "steps": num_steps,
                    "duration_sec": duration,
                    "it_per_sec": it_per_sec,
                    "output_path": str(out_path)
                })
            except Exception as e:
                print(f"    [-] Generation Error (seed {seed}): {e}", flush=True)

        # Explicitly clean pipeline and CUDA memory
        del pipe
        torch.cuda.empty_cache()

        model_results.append({
            "model_name": model_name,
            "model_size_gb": file_size_gb,
            "status": "SUCCESS",
            "load_time_sec": load_sec,
            "images": image_results
        })

        # Flush markdown table
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("# SDXL 15-Model Pass 2: Python Diffusers Benchmark\n\n")
            f.write(f"**Device**: {device_name} | **Precision**: FP16 Native\n")
            f.write(f"**Resolution**: {width}x{height} | **Steps**: {num_steps} (Euler Karras) | **CFG**: {guidance_scale}\n\n")
            f.write("| # | Model Name | Size | Load Time | Status | Seed 42 Speed | Seed 1337 Speed | Image 1 | Image 2 |\n")
            f.write("|---|---|---|---|---|---|---|---|---|\n")

            for i, res in enumerate(model_results):
                imgs = res.get("images", [])
                img1 = imgs[0] if len(imgs) > 0 else None
                img2 = imgs[1] if len(imgs) > 1 else None

                speed1 = f"{img1['duration_sec']:.2f}s ({img1['it_per_sec']:.2f} it/s)" if img1 else "-"
                speed2 = f"{img2['duration_sec']:.2f}s ({img2['it_per_sec']:.2f} it/s)" if img2 else "-"

                link1 = f"[{Path(img1['output_path']).name}]({Path(img1['output_path']).name})" if img1 else "-"
                link2 = f"[{Path(img2['output_path']).name}]({Path(img2['output_path']).name})" if img2 else "-"

                f.write(f"| {i + 1} | `{res['model_name']}` | {res['model_size_gb']:.2f} GB | {res['load_time_sec']:.2f}s | {res['status']} | {speed1} | {speed2} | {link1} | {link2} |\n")

        # Flush JSON
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump({
                "engine": "Python (diffusers + torch)",
                "device": device_name,
                "prompt": STRIKING_PROMPT,
                "negative_prompt": NEGATIVE_PROMPT,
                "guidance_scale": guidance_scale,
                "steps": num_steps,
                "width": width,
                "height": height,
                "models": model_results
            }, jf, indent=2)

    print("\n============================================================", flush=True)
    print(f"Pass 2 (Python) Finished! Report: {json_path}", flush=True)
    print("============================================================", flush=True)

if __name__ == "__main__":
    main()
