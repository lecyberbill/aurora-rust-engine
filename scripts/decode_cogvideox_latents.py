import sys
import os
import torch
from safetensors.torch import load_file
from diffusers import AutoencoderKLCogVideoX
from PIL import Image

def decode_latents(latents_path, out_dir, vae_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Loading latents from {latents_path} on {device}...", flush=True)
    sd = load_file(latents_path)
    latents = sd["latents"].to(device, dtype=torch.float16)  # [1, 16, T, H, W]
    print(f"[*] Latents shape: {latents.shape}, min: {latents.min():.3f}, max: {latents.max():.3f}, mean: {latents.mean():.3f}", flush=True)
    
    if vae_path is None or not os.path.exists(vae_path):
        vae_path = r"D:\image_to_text\.cache\hub\models--THUDM--CogVideoX-2b\snapshots\1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01\vae"
        
    print(f"[*] Loading CogVideoX VAE from {vae_path}...", flush=True)
    vae = AutoencoderKLCogVideoX.from_pretrained(vae_path, torch_dtype=torch.float16).to(device)
    vae.enable_slicing()
    vae.enable_tiling()
    
    print("[*] Decoding video latents with Causal 3D VAE (tiling & slicing enabled)...", flush=True)
    with torch.no_grad():
        # CogVideoX VAE scaling factor: 1.15258426
        scaled_latents = latents / vae.config.scaling_factor
        video = vae.decode(scaled_latents).sample  # [1, 3, T_out, H_out, W_out]
        
    video = ((video.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)
    video = video[0].permute(1, 2, 3, 0).cpu().numpy()  # [T_out, H_out, W_out, 3]
    
    os.makedirs(out_dir, exist_ok=True)
    num_frames = video.shape[0]
    print(f"[+] Writing {num_frames} decoded RGB frames to {out_dir}...", flush=True)
    for i in range(num_frames):
        img = Image.fromarray(video[i])
        frame_path = os.path.join(out_dir, f"frame_{i:03d}.png")
        img.save(frame_path)
    print(f"[+] Done! Successfully saved {num_frames} frames.", flush=True)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python decode_cogvideox_latents.py <latents.safetensors> <output_dir> [vae_dir]")
        sys.exit(1)
    vae_dir = sys.argv[3] if len(sys.argv) > 3 else None
    decode_latents(sys.argv[1], sys.argv[2], vae_dir)
