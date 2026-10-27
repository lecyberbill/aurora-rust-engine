# Benchmark ROCm — preuve d'efficacité

Ce document fige les mesures et la méthode. Il ne compare que des **paires
même-session** : la R9700 change de clocks entre les chargements (surtout sur le
27B), donc les chiffres absolus dérivent de ±7 % d'une session à l'autre. Le
harnais charge le modèle dans un **process neuf** à chaque run ; sur Qwen 27B ce
chargement de 17 Go domine et rend les deltas end-to-end peu fiables. La preuve
« dure » des kernels est le micro-benchmark.

## Harnais

- `target/release/test_text_gen <gguf> <prompt> 256 0.0`
- `AURORA_IGNORE_EOS=1` (génère exactement 256 tokens), `temperature=0`.
- 256 tokens = `257 tokens` dans la ligne du harnais (prefill inclus).
- Portable : `cargo build --release --features rocm` (candle épinglé `fca7197`).
- Fork : `/srv/dev/build_rocm.sh` (fork `7e7ec140` + patch 0003).
- Ollama déchargée avant (`ollama ps`), `rocm-smi` vérifié.

## 1. Flash-decoding split-K — portable

Gemma 3 12B Q4_K_M, N=5, même session, même binaire (A/B par variable
d'environnement) :

| config | runs (tok/s) | médiane |
|---|---|---:|
| flash (défaut) | 44,32 · 44,36 · 44,25 · 44,14 · 44,13 | **44,25** |
| `AURORA_NO_FLASH=1` | 37,91 · 37,31 · 37,29 · 37,30 · 37,22 | **37,30** |

**+18,6 %** ; les 5 runs flash sont au-dessus des 5 runs sans flash.

Qwen3.8-27B, N=3 : médianes **17,10 vs 13,06 tok/s (+31 %)**, cohérent avec la
campagne historique (12,6 → 16,8). Mais les paires intercalées donnent +5,7 %
puis −17 % : le chargement de 17 Go entre chaque run fait dériver les clocks plus
vite que la paire. **On ne retient donc pas un chiffre end-to-end Qwen fiable.**

Preuve déterministe du kernel (micro-bench, `src/bin/test_flash_decode.rs`) :
**0,15 → 0,03 ms/couche** (4,8×), `maxdiff` 4e-7 vs le chemin portable.

## 2. Autotune mmvq/dmmv — fork

Campagne historique, paires même-session (256 tok) :

| modèle | autotune off | autotune on |
|---|---:|---:|
| Qwen3.8-27B | 16,4 | **18,1** (+10 %) |
| Gemma 3 12B | 36,2 | **38,0** (+5 %) |

Sur Gemma, `CANDLE_ROCM_FORCE_MMVQ=1` reproduit l'autotune (44,6 vs 44,5) et
`CANDLE_ROCM_FORCE_DMMV=1` tombe à 39,0 : mmvq gagne sur toutes les formes
mesurées.

## 3. Memset q8_1 — fork, patch 0003

Même binaire fork, N=2/3, 256 tok :

| config | runs (tok/s) | médiane |
|---|---|---:|
| sans patch 0003 | 43,76 · 43,65 | 43,70 |
| avec patch 0003 | 44,39 · 44,64 · 44,58 | **44,54** |

**+1,9 %** : deux `hipMemsetAsync` par matmul supprimés (`y_q8_1` et `dst`).

## 4. Référence externe

Même GGUF, même GPU, 256 tok, llama.cpp/ollama : **Qwen 29,4**,
**Gemma 62,7 tok/s**. Écart après flash : ~1,5× (Gemma) à ~1,75× (Qwen).

## Repro

```bash
# Portable
cargo build --release --features rocm --bin test_text_gen
AURORA_IGNORE_EOS=1 ./target/release/test_text_gen \
  /models/llm/gemma-3-12b-it-q4_k_m.gguf "Explain quantum computing in one sentence." 256 0.0
AURORA_IGNORE_EOS=1 AURORA_NO_FLASH=1 ./target/release/test_text_gen ... # A/B

# Fork (autotune + 0003)
/srv/dev/build_rocm.sh --bin test_text_gen
CANDLE_ROCM_AUTOTUNE=0 ...   # A/B autotune
CANDLE_ROCM_FORCE_MMVQ=1 ... # A/B mmvq/dmmv
```

## Conclusion

- Gain **end-to-end prouvé et reproductible** : flash (Gemma +18,6 %, N=5) et
  memset 0003 (+1,9 %, N=2/3).
- Gain **kernel prouvé** : 4,8× par couche (micro-bench déterministe).
- Qwen end-to-end : non concluant sur ce harnais (clocks); le micro-bench fait foi.
- Rejeté après mesure : q8_1 partagé (+0,6 %, bruit), F16 activations (−14 %),
  sampling GPU (−7 %), SDPA mono-block, graphe HIP « bucketed ».
