# ⚡ SDXL 15-Model Pass 1: `aurora-rust-engine` (Pure Rust) Benchmark

**Device**: NVIDIA GeForce RTX 4070 Ti (12GB VRAM) | **Precision**: FP16 Native
**Resolution**: 1024x1024 | **Steps**: 30 (Euler Karras) | **CFG**: 6.0

| # | Model Name | Size | Load Time | Status | Seed 42 Speed | Seed 1337 Speed | Image 1 | Image 2 |
|---|---|---|---|---|---|---|---|---|
| 1 | `animaPencilXL_v100.safetensors` | 6.46 GB | 29.02s | SUCCESS | 25.37s (1.18 it/s) | 24.59s (1.22 it/s) | [opti_animaPencilXL_v100_seed42.png](opti_animaPencilXL_v100_seed42.png) | [opti_animaPencilXL_v100_seed1337.png](opti_animaPencilXL_v100_seed1337.png) |
| 2 | `aniverseXL_v30.safetensors` | 6.46 GB | 29.47s | SUCCESS | 25.37s (1.18 it/s) | 24.90s (1.20 it/s) | [opti_aniverseXL_v30_seed42.png](opti_aniverseXL_v30_seed42.png) | [opti_aniverseXL_v30_seed1337.png](opti_aniverseXL_v30_seed1337.png) |
| 3 | `babesByStableYogiPony_v50.safetensors` | 6.46 GB | 31.39s | SUCCESS | 25.09s (1.20 it/s) | 30.48s (0.98 it/s) | [opti_babesByStableYogiPony_v50_seed42.png](opti_babesByStableYogiPony_v50_seed42.png) | [opti_babesByStableYogiPony_v50_seed1337.png](opti_babesByStableYogiPony_v50_seed1337.png) |
| 4 | `Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors` | 6.62 GB | 18.74s | SUCCESS | 25.55s (1.17 it/s) | 25.22s (1.19 it/s) | [opti_Juggernaut-XL_v9_RunDiffusionPhoto_v2_seed42.png](opti_Juggernaut-XL_v9_RunDiffusionPhoto_v2_seed42.png) | [opti_Juggernaut-XL_v9_RunDiffusionPhoto_v2_seed1337.png](opti_Juggernaut-XL_v9_RunDiffusionPhoto_v2_seed1337.png) |
| 5 | `betterThanWords_v30.safetensors` | 6.46 GB | 33.07s | SUCCESS | 24.11s (1.24 it/s) | 29.23s (1.03 it/s) | [opti_betterThanWords_v30_seed42.png](opti_betterThanWords_v30_seed42.png) | [opti_betterThanWords_v30_seed1337.png](opti_betterThanWords_v30_seed1337.png) |
| 6 | `bigLove_ponyV20.safetensors` | 6.46 GB | 21.50s | SUCCESS | 25.21s (1.19 it/s) | 24.43s (1.23 it/s) | [opti_bigLove_ponyV20_seed42.png](opti_bigLove_ponyV20_seed42.png) | [opti_bigLove_ponyV20_seed1337.png](opti_bigLove_ponyV20_seed1337.png) |
| 7 | `realismarkPlus_realismarkPlus.safetensors` | 13.35 GB | 53.45s | SUCCESS | 25.09s (1.20 it/s) | 24.20s (1.24 it/s) | [opti_realismarkPlus_realismarkPlus_seed42.png](opti_realismarkPlus_realismarkPlus_seed42.png) | [opti_realismarkPlus_realismarkPlus_seed1337.png](opti_realismarkPlus_realismarkPlus_seed1337.png) |
| 8 | `CHEYENNE_v20.safetensors` | 6.46 GB | 20.04s | SUCCESS | 25.08s (1.20 it/s) | 24.53s (1.22 it/s) | [opti_CHEYENNE_v20_seed42.png](opti_CHEYENNE_v20_seed42.png) | [opti_CHEYENNE_v20_seed1337.png](opti_CHEYENNE_v20_seed1337.png) |
| 9 | `colossusProjectXLSFW_10bNeodemonFP16.safetensors` | 6.62 GB | 23.31s | SUCCESS | 26.16s (1.15 it/s) | 25.28s (1.19 it/s) | [opti_colossusProjectXLSFW_10bNeodemonFP16_seed42.png](opti_colossusProjectXLSFW_10bNeodemonFP16_seed42.png) | [opti_colossusProjectXLSFW_10bNeodemonFP16_seed1337.png](opti_colossusProjectXLSFW_10bNeodemonFP16_seed1337.png) |
| 10 | `CyberRealisticPony_V7a.safetensors` | 6.46 GB | 40.59s | SUCCESS | 24.58s (1.22 it/s) | 27.17s (1.10 it/s) | [opti_CyberRealisticPony_V7a_seed42.png](opti_CyberRealisticPony_V7a_seed42.png) | [opti_CyberRealisticPony_V7a_seed1337.png](opti_CyberRealisticPony_V7a_seed1337.png) |
| 11 | `dreamshaperXL_turboDpmppSDEKarras.safetensors` | 6.46 GB | 24.36s | SUCCESS | 25.18s (1.19 it/s) | 24.68s (1.22 it/s) | [opti_dreamshaperXL_turboDpmppSDEKarras_seed42.png](opti_dreamshaperXL_turboDpmppSDEKarras_seed42.png) | [opti_dreamshaperXL_turboDpmppSDEKarras_seed1337.png](opti_dreamshaperXL_turboDpmppSDEKarras_seed1337.png) |
| 12 | `DreamShaperXL_Turbo_v2_1.safetensors` | 6.46 GB | 29.10s | SUCCESS | 24.25s (1.24 it/s) | 29.27s (1.02 it/s) | [opti_DreamShaperXL_Turbo_v2_1_seed42.png](opti_DreamShaperXL_Turbo_v2_1_seed42.png) | [opti_DreamShaperXL_Turbo_v2_1_seed1337.png](opti_DreamShaperXL_Turbo_v2_1_seed1337.png) |
| 13 | `duchaitenAiartSDXL_v33515LightningTCD.safetensors` | 6.46 GB | 30.15s | SUCCESS | 24.87s (1.21 it/s) | 24.42s (1.23 it/s) | [opti_duchaitenAiartSDXL_v33515LightningTCD_seed42.png](opti_duchaitenAiartSDXL_v33515LightningTCD_seed42.png) | [opti_duchaitenAiartSDXL_v33515LightningTCD_seed1337.png](opti_duchaitenAiartSDXL_v33515LightningTCD_seed1337.png) |
| 14 | `duchaitenPonyXLNo_v60.safetensors` | 6.46 GB | 22.70s | SUCCESS | 24.90s (1.20 it/s) | 24.52s (1.22 it/s) | [opti_duchaitenPonyXLNo_v60_seed42.png](opti_duchaitenPonyXLNo_v60_seed42.png) | [opti_duchaitenPonyXLNo_v60_seed1337.png](opti_duchaitenPonyXLNo_v60_seed1337.png) |
| 15 | `eldgardKinkiestModel_v20.safetensors` | 6.46 GB | 22.50s | SUCCESS | 24.06s (1.25 it/s) | 31.51s (0.95 it/s) | [opti_eldgardKinkiestModel_v20_seed42.png](opti_eldgardKinkiestModel_v20_seed42.png) | [opti_eldgardKinkiestModel_v20_seed1337.png](opti_eldgardKinkiestModel_v20_seed1337.png) |

---

