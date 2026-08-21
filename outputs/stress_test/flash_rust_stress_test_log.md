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
| 6 | `bigLove_ponyV20.safetensors` | 6.46 GB | 21.38s | SUCCESS | 20.72s (1.45 it/s) | 20.03s (1.50 it/s) | [flash_bigLove_ponyV20_seed42.png](flash_bigLove_ponyV20_seed42.png) | [flash_bigLove_ponyV20_seed1337.png](flash_bigLove_ponyV20_seed1337.png) |
| 7 | `realismarkPlus_realismarkPlus.safetensors` | 13.35 GB | 53.98s | SUCCESS | 20.47s (1.47 it/s) | 19.88s (1.51 it/s) | [flash_realismarkPlus_realismarkPlus_seed42.png](flash_realismarkPlus_realismarkPlus_seed42.png) | [flash_realismarkPlus_realismarkPlus_seed1337.png](flash_realismarkPlus_realismarkPlus_seed1337.png) |
| 8 | `CHEYENNE_v20.safetensors` | 6.46 GB | 20.21s | SUCCESS | 20.46s (1.47 it/s) | 19.99s (1.50 it/s) | [flash_CHEYENNE_v20_seed42.png](flash_CHEYENNE_v20_seed42.png) | [flash_CHEYENNE_v20_seed1337.png](flash_CHEYENNE_v20_seed1337.png) |
| 9 | `colossusProjectXLSFW_10bNeodemonFP16.safetensors` | 6.62 GB | 23.60s | SUCCESS | 21.86s (1.37 it/s) | 21.54s (1.39 it/s) | [flash_colossusProjectXLSFW_10bNeodemonFP16_seed42.png](flash_colossusProjectXLSFW_10bNeodemonFP16_seed42.png) | [flash_colossusProjectXLSFW_10bNeodemonFP16_seed1337.png](flash_colossusProjectXLSFW_10bNeodemonFP16_seed1337.png) |
| 10 | `CyberRealisticPony_V7a.safetensors` | 6.46 GB | 38.30s | SUCCESS | 20.78s (1.44 it/s) | 27.06s (1.11 it/s) | [flash_CyberRealisticPony_V7a_seed42.png](flash_CyberRealisticPony_V7a_seed42.png) | [flash_CyberRealisticPony_V7a_seed1337.png](flash_CyberRealisticPony_V7a_seed1337.png) |
| 11 | `dreamshaperXL_turboDpmppSDEKarras.safetensors` | 6.46 GB | 23.69s | SUCCESS | 31.29s (0.96 it/s) | 25.37s (1.18 it/s) | [flash_dreamshaperXL_turboDpmppSDEKarras_seed42.png](flash_dreamshaperXL_turboDpmppSDEKarras_seed42.png) | [flash_dreamshaperXL_turboDpmppSDEKarras_seed1337.png](flash_dreamshaperXL_turboDpmppSDEKarras_seed1337.png) |
| 12 | `DreamShaperXL_Turbo_v2_1.safetensors` | 6.46 GB | 28.37s | SUCCESS | 23.33s (1.29 it/s) | 20.90s (1.44 it/s) | [flash_DreamShaperXL_Turbo_v2_1_seed42.png](flash_DreamShaperXL_Turbo_v2_1_seed42.png) | [flash_DreamShaperXL_Turbo_v2_1_seed1337.png](flash_DreamShaperXL_Turbo_v2_1_seed1337.png) |
| 13 | `duchaitenAiartSDXL_v33515LightningTCD.safetensors` | 6.46 GB | 29.78s | SUCCESS | 27.18s (1.10 it/s) | 29.63s (1.01 it/s) | [flash_duchaitenAiartSDXL_v33515LightningTCD_seed42.png](flash_duchaitenAiartSDXL_v33515LightningTCD_seed42.png) | [flash_duchaitenAiartSDXL_v33515LightningTCD_seed1337.png](flash_duchaitenAiartSDXL_v33515LightningTCD_seed1337.png) |
| 14 | `duchaitenPonyXLNo_v60.safetensors` | 6.46 GB | 23.03s | SUCCESS | 21.87s (1.37 it/s) | 21.11s (1.42 it/s) | [flash_duchaitenPonyXLNo_v60_seed42.png](flash_duchaitenPonyXLNo_v60_seed42.png) | [flash_duchaitenPonyXLNo_v60_seed1337.png](flash_duchaitenPonyXLNo_v60_seed1337.png) |
| 15 | `eldgardKinkiestModel_v20.safetensors` | 6.46 GB | 21.61s | SUCCESS | 21.49s (1.40 it/s) | 20.61s (1.46 it/s) | [flash_eldgardKinkiestModel_v20_seed42.png](flash_eldgardKinkiestModel_v20_seed42.png) | [flash_eldgardKinkiestModel_v20_seed1337.png](flash_eldgardKinkiestModel_v20_seed1337.png) |

---

