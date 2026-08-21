# ⚡ SDXL 15-Model FlashAttention-2: `aurora-rust-engine` (Pure Rust) Benchmark

**Device**: NVIDIA GeForce RTX 4070 Ti (12GB VRAM) | **Precision**: FP16 Native | **Attention**: FlashAttention-2
**Resolution**: 1024x1024 | **Steps**: 30 (Euler Karras) | **CFG**: 6.0

| # | Model Name | Size | Load Time | Status | Seed 42 Speed | Seed 1337 Speed | Image 1 | Image 2 |
|---|---|---|---|---|---|---|---|---|
| 1 | `animaPencilXL_v100.safetensors` | 6.46 GB | 31.29s | SUCCESS | 20.00s (1.50 it/s) | 20.89s (1.44 it/s) | [flash_animaPencilXL_v100_seed42.png](flash_animaPencilXL_v100_seed42.png) | [flash_animaPencilXL_v100_seed1337.png](flash_animaPencilXL_v100_seed1337.png) |
| 2 | `aniverseXL_v30.safetensors` | 6.46 GB | 36.17s | SUCCESS | 19.42s (1.54 it/s) | 19.48s (1.54 it/s) | [flash_aniverseXL_v30_seed42.png](flash_aniverseXL_v30_seed42.png) | [flash_aniverseXL_v30_seed1337.png](flash_aniverseXL_v30_seed1337.png) |
| 3 | `babesByStableYogiPony_v50.safetensors` | 6.46 GB | 36.35s | SUCCESS | 19.55s (1.53 it/s) | 27.02s (1.11 it/s) | [flash_babesByStableYogiPony_v50_seed42.png](flash_babesByStableYogiPony_v50_seed42.png) | [flash_babesByStableYogiPony_v50_seed1337.png](flash_babesByStableYogiPony_v50_seed1337.png) |
| 4 | `Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors` | 6.62 GB | 16.84s | SUCCESS | 20.22s (1.48 it/s) | 19.67s (1.53 it/s) | [flash_Juggernaut-XL_v9_RunDiffusionPhoto_v2_seed42.png](flash_Juggernaut-XL_v9_RunDiffusionPhoto_v2_seed42.png) | [flash_Juggernaut-XL_v9_RunDiffusionPhoto_v2_seed1337.png](flash_Juggernaut-XL_v9_RunDiffusionPhoto_v2_seed1337.png) |
| 5 | `betterThanWords_v30.safetensors` | 6.46 GB | 31.53s | SUCCESS | 19.29s (1.55 it/s) | 30.86s (0.97 it/s) | [flash_betterThanWords_v30_seed42.png](flash_betterThanWords_v30_seed42.png) | [flash_betterThanWords_v30_seed1337.png](flash_betterThanWords_v30_seed1337.png) |

---

