# Aure-Spec — Intégration SD 3.5 Large (`src/models/`)

**Date**: 2026-09-11 | **Repos**: `aurora-rust-engine` | **Statut**: ÉBAUCHE — à valider
**Cible**: `stableDiffusion35Fp8_v35LargeTurbo.safetensors` (7.59 GB, FP8 E4M3) + variantes FP16.

## 1. Contexte

SD3.5 utilise le même **MMDiT** (DoubleStreamBlock) que Flux.1, déjà présent dans
`src/diffusion/dit/blocks.rs` (avec QK-norm optionnel et MLP classique/SwiGLU). Le squelette
existe donc en grande partie. Ce qui manque, c'est **le câblage spécifique SD3.5** : config
correcte, **remap des clés** (noms BFL SD3.5 ≠ Flux), gestion du **pooled** (y_embedder),
et les **3 text encoders**.

## 2. Réalité mesurée du checkpoint (shape sniff)

`stableDiffusion35Fp8_v35LargeTurbo.safetensors` — 923 clés, FP8 E4M3 :

| Tenseur | Shape | Sens |
|---|---|---|
| `model.diffusion_model.x_embedder.proj.weight` | `[2432, 16, 2, 2]` | patchify 2×2, in_ch **16**, hidden **2432** |
| `model.diffusion_model.context_embedder.weight` | `[2432, 4096]` | T5 context **4096** → hidden |
| `model.diffusion_model.joint_blocks.{0..37}.x_block.attn.qkv.weight` | `[7296, 2432]` | 3×2432, **38 blocs** |
| `...x_block.attn.proj.weight` | `[2432, 2432]` | |
| `...x_block.mlp.fc1.weight` | `[9728, 2432]` | mlp_ratio **4** |
| `...x_block.attn.ln_q.weight` / `ln_k.weight` | — | **QK-norm = LayerNorm** (pas RMSNorm) |
| `model.diffusion_model.y_embedder.mlp.0/2` | — | **pooled** (CLIP-G) |

> **Correction à apporter** : `FluxConfig::sd35_large()` dit `24 blocks / 1536 hidden` — c'est
> **SD3 Medium**, pas Large. SD3.5 Large = **38 blocks / 2432 hidden / heads 38 / head_dim 64**.

## 2bis. FAITS VÉRIFIÉS (sniff complet — 41 patterns)

Le checkpoint SD3.5 ne se réduit PAS à un remap de clés : le **forward pass diffère**.

1. **`pos_embed` appris `[1, 36864, 2432]`** (`36864 = 192²` patches). SD3 ajoute un
   positional embedding appris aux tokens image — **Flux n'a que du RoPE**. Le `FluxTransformer`
   actuel ne sait pas l'appliquer → il faut l'**ajouter** au forward (ou un transformer SD3 dédié).
2. **`x_embedder.proj` = Conv2d `[2432, 16, 2, 2] + bias`** (patchify 2×2) — Flux `img_in` = Linear.
3. **MLP non-gated** : `mlp.fc1 [9728, 2432]` (=4×) → GELU → `fc2` — Flux = SwiGLU gated (6×).
4. **QK-norm = LayerNorm** : `attn.ln_q.weight [64]` / `ln_k.weight [64]` (pas de biais, pas RMSNorm).
5. **Pas de single-stream** (`num_single_blocks = 0`).
6. **Text encoders NON embarqués** (0 clé `text_encoder`/`clip`/`t5`) → T5-XXL + CLIP-L + OpenCLIP-G
   doivent être **attachés en externe**.
7. `final_layer.linear [64, 2432]` = out 16ch×2×2 ; `t_embedder.mlp.0 [2432, 256]`.

## 3. Écarts vs Flux.1 (le vrai travail)

| Aspect | Flux.1 (interne) | SD3.5 (BFL checkpoint) | Action |
|---|---|---|---|
| Bloc image | `double_blocks.N.img_attn.*` | `joint_blocks.N.x_block.*` | **remap clés** |
| Bloc texte | `double_blocks.N.txt_attn.*` | `joint_blocks.N.context_block.*` | **remap clés** |
| Entrée image | `img_in` | `x_embedder.proj` (conv 2×2) | remap |
| Entrée texte | `txt_in` | `context_embedder` | remap (4096→2432) |
| Pooled | `vector_in.in_layer/out_layer` | `y_embedder.mlp.0/.2` | remap |
| Modulations | `img_mod.lin` / `txt_mod.lin` | `adaln_modulation` (x/caption) | remap |
| AdaLN final | `final_layer.adaLN_modulation.1` | `norm_out` ? | à confirmer |
| Single-stream | `single_blocks.*` | **aucun** | `num_single_blocks = 0` |
| QK-norm | RMSNorm (Flux) | **LayerNorm** (`ln_q`/`ln_k`) | **module LN** |
| Pos. embed | 4 axes `[32,32,32,32]` | 3 axes `[16,56,56]` | config (déjà `sd35_large`) |
| VAE | 16-ch (`flux2-vae` ou embarqué) | **16-ch** (SD3 VAE) | réutiliser |
| Scheduler | empirique Flux.2 | `shift = 3.0` | config |

## 3bis. Composants SD3.5 — TOUS PRÉSENTS sur G: ✅

Vérifiés (shapes réelles) :

| Composant | Chemin | Shape clé |
|---|---|---|
| T5-XXL | `G:\models\clip\t5xxl_fp16.safetensors` | `q [4096,4096]`, `shared [32128,4096]` |
| CLIP-L | `G:\models\clip\clip_l.safetensors` | `q_proj [768,768]` |
| OpenCLIP-G | `G:\models\clip\clip_g.safetensors` | `q_proj [1280,1280]`, `text_projection [1280,1280]` |
| VAE 16ch | `G:\models\vae\sd3_vae.safetensors` | `decoder.conv_in [512,16,3,3]` |

**Encodage texte SD3.5 (confirmé par la référence diffusers)** :
- **séquence contexte** = `concat(CLIP_L+G hidden [77, 2048] zéro-padded→4096, T5 [t5_seq, 4096])` **sur l'axe
  séquence** → `[B, 77+t5_seq, 4096]` → `context_embedder [2432,4096]`.
  (Pas de token spécial ; simple concat le long de dim=-2.)
- **pooled** = `concat(CLIP-G_pooled(1280), CLIP-L_pooled(768))` = **2048** → `y_embedder.mlp.0 [2432,2048]`.
  ⚠️ ordre G⊕L à confirmer empiriquement (certaines impls font L⊕G).

**pos_embed (confirmé par `diffusers/models/embeddings.py::PatchEmbed`)** :
- **sincos 2D fixe** (`get_2d_sincos_pos_embed`), enregistré en **buffer persistant** (donc présent dans le
  checkpoint : `[1, 36864, 2432]` = `[1, 192², 2432]`, `pos_embed_max_size = 192`).
- Résolution ≠ 192² : **center-crop** du buffer reshapé `[1,192,192,2432]` vers `[1,h,w,2432]`
  (`top = (192-h)//2`, `left = (192-w)//2`), puis flatten `[1,h*w,2432]`.
- → On **charge le buffer** et on le center-croppe (exact, pas de recalcul de base_size).

**Patch embed** : `x_embedder.proj` = Conv2d 2×2 `[2432,16,2,2]` + bias → équivaut à une **Linear
`[2432, 64]`** appliquée aux patches 2×2 aplatis (`64 = 16×2×2`), exactement comme Flux.1. On reshape
le poids conv → `img_in.weight [2432,64]` + `img_in.bias [2432]`.

**MLP** : non-gated `fc1(4×) → GELU(tanh) → fc2` (Flux = SwiGLU gated). `DoubleStreamBlock` gère
déjà les deux conventions.

**QK-norm** : LayerNorm (`ln_q`/`ln_k` `[64]`, elementwise_affine, sans biais de normalisation).

## 4. Stratégie retenue — **Adapter + extension du forward, pas de réécriture totale**

`DoubleStreamBlock` supporte déjà QK-norm (RMSNorm) et MLP classique/SwiGLU. On **remap les clés SD3.5
vers le vocabulaire canonique**, et on **étend le forward** pour les 3 différences réelles :
1. **pos_embed appris** : ajouter `img_h = img_h + pos_embed[0, :seq]` avant les blocs.
2. **patch embed Conv2d + bias** : soit une `Conv2d` 2×2, soit reshape du poids en Linear sur patch 2×2.
3. **QK-norm LayerNorm** (au lieu de RMSNorm) : nouveau module optionnel.

Bénéfices : réutilise `FluxTransformer`/`DoubleStreamBlock`/`AdaLNZeroModulation`, et l'intégration
reste incrémentale (remap + config + 3 petits ajouts de forward).

### 4.1 Table de remap (clés BFL SD3.5 → internes)

```
model.diffusion_model.x_embedder.proj.weight   -> img_in.weight
model.diffusion_model.context_embedder.weight  -> txt_in.weight
model.diffusion_model.y_embedder.mlp.0.weight  -> vector_in.in_layer.weight
model.diffusion_model.y_embedder.mlp.2.weight  -> vector_in.out_layer.weight
model.diffusion_model.joint_blocks.N.x_block.attn.qkv.weight      -> double_blocks.N.img_attn.qkv.weight
model.diffusion_model.joint_blocks.N.x_block.attn.proj.weight     -> double_blocks.N.img_attn.proj.weight
model.diffusion_model.joint_blocks.N.x_block.attn.ln_q.weight     -> double_blocks.N.img_attn.norm.query_norm.weight (LN)
model.diffusion_model.joint_blocks.N.x_block.attn.ln_k.weight     -> double_blocks.N.img_attn.norm.key_norm.weight
model.diffusion_model.joint_blocks.N.x_block.mlp.fc1.weight       -> double_blocks.N.img_mlp.0.weight
model.diffusion_model.joint_blocks.N.x_block.mlp.fc2.weight       -> double_blocks.N.img_mlp.2.weight
model.diffusion_model.joint_blocks.N.x_block.adaLN_modulation.1.weight -> double_blocks.N.img_mod.lin.weight
... (idem context_block -> txt_attn / txt_mlp / txt_mod)
model.diffusion_model.joint_blocks.N.x_block.attn.norm_q.weight (RMS?) -> idem
```

### 4.2 QK-norm LayerNorm

Ajouter un `LayerNorm`-norm (elementwise_affine, sans biais de normalisation) à côté du `RMSNorm`
actuel dans `DoubleStreamBlock`, sélectionné selon le nom de clé (`ln_q` vs `query_norm`).

### 4.3 Config correcte

```rust
pub fn sd35_large() -> Self {
    Self {
        in_channels: 16, out_channels: 16,
        hidden_size: 2432, num_heads: 38, head_dim: 64,   // NEW field
        num_double_blocks: 38, num_single_blocks: 0,
        mlp_ratio: 4, theta: 10_000.0,
        guidance_embed: false,
        axes_dim: vec![16, 56, 56],
    }
}
```
+ nouveau variant `sd35_medium()` (24/1536) pour ne pas casser l'ancienne valeur.

## 5. Text encoders (3)

SD3.5 conditionne sur **T5-XXL (séquence 4096)** + **CLIP-G pooled (2048 = CLIP-L 768 + OpenCLIP-G 1280)**.

- `context_embedder` attend la sortie T5 (4096) ;
- `y_embedder` attend le pooled CLIP (2048).

Pipeline d'encodage :
1. T5-XXL → `[B, seq, 4096]` → `txt_in`.
2. CLIP-L (768) + OpenCLIP-G (1280) **pooled**, concaténés → `[B, 2048]` → `y_embedder`.
3. `vector_in` ajoute le pooled au `temb`.

Les encodeurs T5/CLIP-L/OpenCLIP sont **déjà dans le moteur** (`src/text/{t5,clip,open_clip}.rs`).
Le checkpoint SD3.5 **embarque-t-il** les text encoders ? (Non vu dans les 923 clés → probablement
non, comme Klein-4B). À vérifier : attacher en externe via `ModelDescriptor` sinon.

## 6. Scheduler

SD3.5 : Flow Match Euler avec **`shift = 3.0`** statique (pas l'empirique Flux.2). Le
`FlowMatchEulerScheduler` gère déjà `shift > 0` → il suffit de passer `FlowMatchEulerConfig { shift: 3.0, .. }`.

## 7. Intégration `AutoModel` / `Architecture`

1. Ajouter `Architecture::Sd35Large` / `Sd35Medium` à `src/models/config/mod.rs` (détection : présence
   de `joint_blocks.` + `x_embedder.proj` + `context_embedder`, discriminateur hidden 2432 vs 1536).
2. `build_diffusion` (auto.rs) : brancher un chemin SD3.5 qui construit le `FluxTransformer` via
   `FluxConfig::sd35_large()` + la table de remap + le VAE 16-ch.
3. `ModelDescriptor` : `TextEncoderSpec` étendu (T5+CLIP) si encoders externes.
4. `AutoModel::from_descriptor` : attacher les 3 TEK si nécessaire.

## 8. Fichiers impactés

| Fichier | Action |
|---|---|
| `src/diffusion/dit/flux.rs` | config `sd35_large` corrigée + `sd35_medium` ; champ `head_dim` |
| `src/diffusion/dit/blocks.rs` | + `LayerNormQK` optionnel (SD3) |
| `src/weights.rs` | + table de remap SD3.5 (`sd35_to_internal`) |
| `src/models/config/mod.rs` | + `Architecture::Sd35Large/Medium` + détection |
| `src/models/auto.rs` | chemin de build SD3.5 |
| `src/pipelines/flux.rs` | accepter la config SD3.5 + scheduler shift 3.0 |

## 9. Critères d'acceptation

- [x] Chargement du checkpoint SD3.5 FP8 sans erreur, architecture détectée `Sd35Large`.
- [x] T2I 512×512, 20 steps : **loup blanc net** (`outputs/sd35_test.png`), orienté.
- [ ] `AutoModel::from_descriptor` charge SD3.5 (avec ses TEK) comme les autres familles.
- [ ] Ajout à la vitrine `aurora_studio` (dropdown) une fois validé.
- [x] `cargo test --lib` 25/25 ; build sans warning.

## 9bis. ÉTAT ACTUEL (2026-09-11) — pipeline complet, sortie grise (dernier bug)

`test_sd35.rs` exécute **tout le pipeline de bout en bout sans crash** : SD3.5 Large charge,
conditionnement texte (CLIP-L+G+T5) fini, sampler, VAE 16ch décode. **Mais la sortie est uniformément
grise.** Diagnostic (via FLUX_TRACE) :
- `temb` rms ≈ 6.6 ; `silu(temb)` a une **moyenne positive** ; poids `adaLN_modulation.1` rms ≈ 0.016.
- → sortie de modulation `proj` rms ≈ **69**, dont `scale_msa` ≈ 102 et `gate_msa` ≈ 30 (**énormes** ;
  AdaLN-Zero devrait donner des gates ≈ 0).
- Le flux texte explose alors (bloc 0 : txt 1.6 → 35k → clamp 50000) puis NaN → latents constants → gris.

**MISE À JOUR (2026-09-11) — 3 bugs trouvés et corrigés, il reste le RENDU :**
1. **La modulation et le `temb` sont CORRECTS** (vérifié : référence PyTorch identique à 0.02 près).
2. **F16 overflow = cause du gris/NaN.** À σ=1 SD3.5 produit des activations légitimes mais énormes
   (V≈226, gates≈30, branche texte ≈200k > 65504) → inf → NaN en F16. **Fix : le pipeline tourne SD3.5
   en F32** (`from_single_file_streaming` choisit F32 si `joint_blocks`). Diffusers exige bf16/fp32 pour SD3.5.
3. **Unpatchify SD3 = `[ph,pw,c]`** (patch-major, `einsum nhwpqc->nchpwq`), pas `[c,ph,pw]` (Flux) →
   c'était le damier vert. **Fix : `sd3_unpatchify`**. + `latents/scaling_factor + shift_factor` avant decode.

**État : plus de NaN ni damier, mais l'image est du bruit.**
- **VAE SD3 CONFIRMÉ BON** : roundtrip encode→decode d'une vraie image (renard) reproduit l'image
  parfaitement (`outputs/sd3_vae_rt_raw.png`). Donc ce n'est ni le VAE ni le packing.
- Le pipeline est fidèle à diffusers (init bruit `randn`, decode `latents/1.5305 + 0.0609`).
- **Donc le latent produit par le transformer/sampler est du bruit** → le transformer ne dénoise pas
  correctement malgré des entrées/modulation vérifiées.

Prochaines étapes (ciblées) :
1. **Dumper le latent final** (avant VAE) et vérifier sa structure spatiale / le comparer à un latent
   de référence diffusers pour le même prompt (le sampler doit produire un latent structuré).
2. Comparer la **vitesse prédite par le transformer à t=~0.7** (step intermédiaire) à la référence
   PyTorch (étendre `sd35_block.py` au modèle complet ou au moins à la sortie `proj_out`).
3. Vérifier le **signe/échelle du pas Euler** et que `timesteps`/`sigmas` SD3 (shift=3.0) sont corrects.
4. Vérifier le `swap_scale_shift` du `final_layer` (SD3 BFL [shift,scale]) et le `norm_out`
   (AdaLayerNormContinuous).
5. SD3.5 Large **Turbo** → 4 steps, guidance basse ; tester aussi CLIP-L/G **penultimate vs last**
   hidden layer (notre `encode_prompt` utilise la pénultième ; SD3 utilise peut-être la dernière).

---

**CONSTAT PRÉCÉDENT (modulation) — conservé :**
Une référence PyTorch (`C:\...\Temp\opencode\sd35_ref.py`) recalculant `temb` et la modulation avec les
poids du checkpoint et nos `sigma`/`pooled` dumpés donne **exactement** les mêmes valeurs
(`proj_rms 69.309` ref vs `69.305` nous ; `max|Δtemb|=0.02`). Donc ni le conditionnement ni la
modulation ne sont fautifs : **à σ=1 le modèle produit légitimement `scale≈102, gate≈30`.**

Le vrai problème est ailleurs — **runaway numérique dans le forward du bloc** :
- Bloc 0 (F32) : `q` (post LayerNorm QK) rms 1.18 ✓, `k` 1.10 ✓, **`v` rms 226** (V n'est pas
  normalisé — normal), `attn_out` 134, `gate1` 42, `img_attn_proj` 140 → **`img_after_attn = inf`**.
- Le flux `img` devient NaN/inf dès le bloc 0 puis se propage.

L'attention (scale `1/√head_dim` OK, F32) et la QK-norm (LayerNorm, sortie rms ~1.2) sont correctes.
À σ=1, `img_normed` rms ~127 et `txt_normed` ~153 sont grands mais **attendus** (le modèle SD3.5 a des
modulations fortes au premier pas) ; diffusers ne diverge pas pour autant. Il reste donc une
divergence bloc-à-bloc avec la référence diffusers.

Pistes (prochaine session) :
1. **Porter le bloc 0 SD3.5 (`JointTransformerBlock`) en PyTorch** sur les mêmes entrées dumpées
   (img, context) et comparer `attn_out`, `img_after_attn`, `mlp_out` → localiser l'op qui diverge
   (ordre des chunks de modulation, placement du gate, `to_out`/`to_add_out`, `image`/`text` split).
2. Vérifier le **clamp** : le flux txt est clampé à ±50000 mais **pas le flux img** → laisser `img`
   dériver à l'inf. Diffusers ne clampe pas mais reste stable ; notre img divergeavant le clamp.
3. Vérifier l'ordre **`[txt, img]` vs `[img, txt]`** dans la concaténation d'attention (sans effet sur
   l'attention pleine, mais critique si l'un des deux streams est traité différemment).
4. Retirer `FORCE_F32`/`FLUX_DUMP` (debug) une fois résolu.
5. Rappel : SD3.5 Large **Turbo** → 4 steps, guidance basse.

## 9ter. RÉSOLU (2026-09-11) — SD3.5 Large Turbo rend un loup net

**Deux bugs réels** (les 3 "fixes" précédents étaient nécessaires mais pas suffisants ; le forward du
transformer était en fait **exact**, vérifié bloc-à-bloc) :

1. **`text_projection` CLIP-G jamais chargé.** `OpenClipTextEncoder::new_sdxl` testait
   `vb.contains_tensor("text_projection")` alors que la clé réelle est `text_projection.weight` →
   `text_projection = None` → le pooled CLIP-G sortait **non projeté** (embedding EOT brut). Le
   transformer recevait alors un `pooled_projections` faux → modulation énorme → vitesse rms 74 au lieu
   de ~0.5 → bruit. Fix : détecter `.weight` **et** transposer (`eos @ W^T`, convention `nn.Linear`).
   *Bonus : SDXL utilise le même encodeur → son pooled est désormais correct lui aussi (validé Juggernaut).*

2. **Incohérence d'ordre "packed" `(c,ph,pw)` vs `(ph,pw,c)`.** SD3 `PatchEmbed` (Conv2d) consomme les
   latents patchés en `(c, ph, pw)`, mais `proj_out` + unpatchify `einsum "nhwpqc->nchpwq"` **émet** en
   `(ph, pw, c)`. diffusers reste dans le domaine latent naturel `[B,16,H,W]` donc l'asymétrie s'annule ;
   notre pipeline travaille en **packed** et additionnait un vecteur `(ph,pw,c)` à un état `(c,ph,pw)`
   → état corrompu à chaque pas Euler. Fix : le forward SD3 réordonne sa sortie en `(c,ph,pw)`
   (après `final_linear`), et `sd3_unpatchify` repasse en `(c,ph,pw)`.

**Deux pièges écartés en cours de route :**
- `timestep_scale` : le pipeline diffusers passe le **timestep brut 0..1000** (pas `t/1000`) à
  `time_text_embed`. Le `time_factor = 1000` BFL/Flux est donc **déjà correct pour SD3.5** — un passage
  à `1.0` casse la sortie. Le champ `FluxConfig::timestep_scale` documente cette convention (uniforme 1000).
- La "convergence à 0.02" de `temb`/modulation et l'égalité `cos=1.0` du forward complet **avec le bon
  `pooled`** prouvaient que le MMDiT était bon ; le problème n'était ni l'attention, ni la QK-norm, ni
  l'ordre des 6 chunks AdaLN (tous validés contre `diffusers`).

**Validation :** `test_sd35` (512×512, 20 steps) → loup blanc photoréaliste. Non-régression : `cargo test
--lib` 25/25, SDXL (Juggernaut-XL v9) toujours net.

## 10. Hors-scope (plus tard)

- SD3 Medium (24 blocks) — config presque prête.
- PixArt / Sana / Lumina / Chroma : évaluer après SD3.5.
- Rectified-flow "turbo" LoRA 4-step : réutiliser le pipeline LoRA existant.
