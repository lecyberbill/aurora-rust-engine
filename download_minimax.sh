#!/bin/bash
set -e
BASE_URL="https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main"

echo "=== [1/4] Downloading Video VAE FP16 (5.2 GB) ==="
wget -c --show-progress -O /models/comfyui/minimax_h3/vae/minimax_h3_video_vae_fp16.safetensors \
  "${BASE_URL}/vae/minimax_h3_video_vae_fp16.safetensors"

echo "=== [2/4] Downloading Turbo 4-Step LoRA (1.95 GB) ==="
wget -c --show-progress -O /models/comfyui/minimax_h3/loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors \
  "${BASE_URL}/loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"

echo "=== [3/4] Downloading DiT FL2VA Pruned FP8 Scaled (20.9 GB) ==="
wget -c --show-progress -O /models/comfyui/minimax_h3/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors \
  "${BASE_URL}/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors"

echo "=== [4/4] Downloading Text Encoder Qwen3-VL NVFP4 (15.7 GB) ==="
wget -c --show-progress -O /models/comfyui/minimax_h3/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
  "${BASE_URL}/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"

echo "=== ALL DOWNLOADS COMPLETE ==="
ls -lh /models/comfyui/minimax_h3/*/*.safetensors
