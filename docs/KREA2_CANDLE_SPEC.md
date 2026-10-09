# Krea 2 → Candle : spécification d'inférence (text-to-image)

> Document destiné à un agent de code. Objectif : réimplémenter en Rust/Candle l'inférence de **Krea 2 Turbo** (prioritaire) et **Krea 2 Raw**, avec parité numérique contre le code de référence PyTorch.
> Toutes les valeurs ci-dessous sont tirées du code source lu le 2026-10-09 (voir §12). Les points marqués **[À VÉRIFIER]** n'ont pas pu être confirmés depuis le code et doivent être contrôlés dans les `config.json` réels.

---

## 0. Périmètre

- **Dans le périmètre** : encodeur texte Qwen3-VL-4B (texte seul), DiT single-stream 12,9 B (`SingleStreamDiT`), scheduler flow-matching (Euler + CFG), décodeur VAE Qwen-Image (une seule frame).
- **Hors périmètre** : encodeur VAE (inutile en text-to-image), branche vision de Qwen3-VL, prompt expander (simple LLM avec le system prompt `docs/expansion.txt`, optionnel), système de style-reference (non publié), LoRA (phase ultérieure).
- **Licence** : poids sous *Krea 2 Community License*, accès HF « gated ». La licence impose au déployeur un filtrage de contenu (classifieurs ou revue). À prévoir côté application, pas dans le portage.

---

## 1. Vue d'ensemble du pipeline

```
prompt ──► Tokenizer Qwen ──► Qwen3-VL-4B (texte) ──► 12 couches cachées sélectionnées
                                                          │  (B, 512, 12, 2560)
                                                          ▼
bruit N(0,1) (B,16,H/8,W/8) ──► patchify 2x2 ──► SingleStreamDiT(img, ctx, t, pos, mask) ──► vitesse v
        ▲                                                     │
        └──────────── Euler : x ← x + (t_next − t_cur)·v ◄────┘   (8 pas Turbo / 28–52 pas Raw)
                                                          ▼
                       unpatchify ──► dénormalisation latents ──► VAE Qwen-Image decode ──► RGB
```

Composants externes :

| Composant | Source HF | Rôle |
|---|---|---|
| Encodeur texte | `Qwen/Qwen3-VL-4B-Instruct` | `Qwen3VLForConditionalGeneration`, on n'utilise que le décodeur texte |
| VAE | `Qwen/Qwen-Image`, sous-dossier `vae` | `AutoencoderKLQwenImage` (architecture Wan 2.1, f8, 16 canaux) |
| DiT Turbo | `krea/Krea-2-Turbo` → `turbo.safetensors` (format original) + sous-dossiers diffusers | Distillé TDM, 8 pas, sans CFG |
| DiT Raw | `krea/Krea-2-Raw` (fichier single-file, nom **[À VÉRIFIER]**) | Base non distillée, CFG |

Les dépôts HF sont aussi au format diffusers (`Krea2Pipeline`, `transformer/`, `vae/`, `text_encoder/`, `tokenizer/`) : lire `model_index.json` pour la liste exacte **[À VÉRIFIER]**.

---

## 2. Configuration du DiT (valeurs du code officiel)

```python
SingleMMDiTConfig(
    features=6144,     # dim cachée
    tdim=256,          # dim de l'embedding sinusoïdal du timestep
    txtdim=2560,       # dim des hidden states Qwen3-VL-4B
    heads=48,          # têtes Q
    kvheads=12,        # têtes K/V (GQA, ratio 4)
    multiplier=4,      # facteur MLP
    layers=28,         # blocs single-stream
    patch=2,
    channels=16,       # canaux latents VAE
    txtheads=20, txtkvheads=20,   # text fusion : MHA classique
    txtlayers=12,      # nombre de couches Qwen agrégées (≠ profondeur)
    bias=False,        # pas de biais dans attention/MLP
    theta=1e3,         # RoPE du DiT (valeur par défaut, non surchargée)
)
```

Valeurs dérivées :

| Grandeur | Valeur | Calcul |
|---|---|---|
| head_dim DiT | 128 | 6144 / 48 |
| dim K/V DiT | 1536 | 12 × 128 |
| MLP DiT (hidden) | 16384 | `int(2*6144/3)*4 = 16384`, arrondi au multiple de 128 |
| head_dim text fusion | 128 | 2560 / 20 |
| MLP text fusion | 6912 | `int(2*2560/3)*4 = 6824` → arrondi 128 → 6912 |
| Axes RoPE (head_dim 128) | `[32, 48, 48]` | `[hd − 12·(hd//16), 6·(hd//16), 6·(hd//16)]` |
| Paramètres | ≈ 12,8–12,9 B | 28 × ~434 M + text fusion ~344 M + `tproj` ~226 M + divers |

Formule MLP SwiGLU (à reproduire exactement) :
```
mlpdim = int(2 * features / 3) * multiplier
mlpdim = 128 * ceil(mlpdim / 128)
```

---

## 3. Noms des poids (format original `turbo.safetensors`)

Le fichier original suit les noms de `mmdit.py`. Types mixtes BF16 + F32 dans le fichier (les échelles RMSNorm sont probablement en F32 **[À VÉRIFIER]** via `safetensors` metadata). Charger tel quel puis caster au dtype de calcul, sauf normes (voir §8).

```
first.weight / first.bias                         Linear(64 → 6144, bias)
tmlp.0.{weight,bias}                              Linear(256 → 6144)
tmlp.2.{weight,bias}                              Linear(6144 → 6144)
tproj.1.{weight,bias}                             Linear(6144 → 36864)        # index 0 = GELU
txtmlp.0.scale                                    RMSNorm(2560)
txtmlp.1.{weight,bias}                            Linear(2560 → 6144)
txtmlp.3.{weight,bias}                            Linear(6144 → 6144)        # index 2 = GELU
txtfusion.projector.weight                        Linear(12 → 1, sans biais)  shape (1, 12)
txtfusion.layerwise_blocks.{0,1}.<TextFusionBlock>
txtfusion.refiner_blocks.{0,1}.<TextFusionBlock>
blocks.{0..27}.mod.lin                            shape (36864,) = 6 × 6144 aplati
blocks.{i}.prenorm.scale / postnorm.scale         (6144,)
blocks.{i}.attn.wq.weight                         (6144, 6144)
blocks.{i}.attn.wk.weight / wv.weight             (1536, 6144)
blocks.{i}.attn.gate.weight                       (6144, 6144)
blocks.{i}.attn.wo.weight                         (6144, 6144)
blocks.{i}.attn.qknorm.qnorm.scale / knorm.scale  (128,)
blocks.{i}.mlp.gate.weight / up.weight            (16384, 6144)
blocks.{i}.mlp.down.weight                        (6144, 16384)
last.norm.scale                                   (6144,)
last.modulation.lin                               shape (2, 6144)
last.linear.{weight,bias}                         Linear(6144 → 64, bias)

<TextFusionBlock> (dim 2560) :
  prenorm.scale, postnorm.scale,
  attn.{wq,wk,wv,gate,wo}.weight (2560×2560), attn.qknorm.{qnorm,knorm}.scale (128),
  mlp.{gate,up}.weight (6912×2560), mlp.down.weight (2560×6912)
```

Si l'on charge plutôt `transformer/` au format diffusers, table de renommage (issue de `convert_krea2_transformer_checkpoint_to_diffusers`, diffusers 0.41.0) :

| Original | Diffusers |
|---|---|
| `first.` | `img_in.` |
| `tmlp.0.` / `tmlp.2.` | `time_embed.linear_1.` / `time_embed.linear_2.` |
| `tproj.1.` | `time_mod_proj.` |
| `txtmlp.0.scale` | `txt_in.norm.weight` |
| `txtmlp.1.` / `txtmlp.3.` | `txt_in.linear_1.` / `txt_in.linear_2.` |
| `txtfusion.` | `text_fusion.` |
| `blocks.` | `transformer_blocks.` |
| `last.linear.` / `last.norm.scale` / `last.modulation.lin` | `final_layer.linear.` / `final_layer.norm.weight` / `final_layer.scale_shift_table` |
| `.attn.wq/wk/wv/wo/gate.` | `.attn.to_q/to_k/to_v/to_out.0/to_gate.` |
| `.attn.qknorm.qnorm.scale` / `knorm.scale` | `.attn.norm_q.weight` / `.attn.norm_k.weight` |
| `.mlp.` | `.ff.` |
| `.prenorm.scale` / `.postnorm.scale` | `.norm1.weight` / `.norm2.weight` |
| `.mod.lin` (36864,) | `.scale_shift_table` reshapé en (6, 6144) |

Recommandation : implémenter le format original (un seul fichier, noms courts), le format diffusers via un `VarBuilder` renommé en option.

---

## 4. Encodage texte (Qwen3-VL-4B, texte seul)

### 4.1 Gabarit et tokenisation (à reproduire à l'octet près)

```
PREFIX = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
PREFIX_IDX = 34          # nb de tokens du PREFIX, retirés en sortie
NUM_SUFFIX_TOKENS = 5
MAX_LEN = 512
```

Procédure :
1. `text = PREFIX + prompt`.
2. Tokeniser `text` avec troncature et **padding à droite** jusqu'à `MAX_LEN + PREFIX_IDX − NUM_SUFFIX_TOKENS = 541` tokens. Le prompt utile est donc limité à ~507 tokens.
3. Tokeniser `SUFFIX` séparément (5 tokens, sans padding).
4. Concaténer : `input_ids = [prefix | prompt | PAD… | suffix]` (546 tokens). Le suffixe est **après** le padding, c'est voulu (identique à l'entraînement).
5. `mask` = concat des attention_mask (bool).

Tokenizer : `tokenizer.json` de `Qwen/Qwen3-VL-4B-Instruct`, crate `tokenizers`. Vérifier que `PREFIX` donne bien 34 tokens et `SUFFIX` 5 tokens (test unitaire).

### 4.2 Positions RoPE de Qwen (piège principal)

Les positions ne sont **pas** les indices bruts. Pour un token valide : `pos = (nombre de tokens valides qui le précèdent)`, soit `cumsum(mask) − 1`. Les tokens de padding ne consomment pas de position, donc le suffixe suit directement le prompt.

- Diffusers : `position_ids = (mask.cumsum(-1) − 1).clamp(min=0)`, passé explicitement.
- Code officiel (transformers 4.57.1) : `get_rope_index` texte seul fait `cumsum − 1` puis met 1 sur les pads. Valeurs identiques sur les tokens valides ; la valeur des pads est sans effet (ils sont masqués).
- Les 3 axes M-RoPE (T/H/W) reçoivent la même position. L'entrelacement M-RoPE de Qwen3-VL devient alors **équivalent à une RoPE 1D standard** (style `rotate_half`, non entrelacé → `candle_nn::rotary_emb::rope`, pas `rope_i`).

### 4.3 Masque d'attention Qwen

Causal **et** padding (un token ne voit pas les pads). Construire un masque additif 4D `(B,1,L,L)` = causal ∧ key_padding.

### 4.4 Sortie

- Récupérer `hidden_states[i]` pour `i ∈ (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)`.
- Convention HF : `hidden_states[0]` = sortie de l'embedding, `hidden_states[i]` = sortie du i-ème bloc décodeur (avant la norme finale). L'indice max (35) est < nombre de couches, donc la norme finale n'est jamais concernée.
- Empiler sur un nouvel axe 2 → `(B, 546, 12, 2560)`, puis retirer les 34 premiers tokens → **`ctx (B, 512, 12, 2560)`**, `txtmask (B, 512)`.
- Pas de `lm_head`, pas de génération.

### 4.5 Config Qwen3-VL-4B texte **[À VÉRIFIER dans `config.json`]**

`hidden_size=2560` (confirmé par `txtdim`), 36 couches, 32 têtes Q, 8 têtes KV, head_dim 128, q/k-norm RMSNorm, `rope_theta` ≈ 5e6, `mrope_section=[24,20,20]` entrelacé, RMSNorm eps 1e-6. Lire les valeurs réelles, ne pas les coder en dur.

### 4.6 Réutilisation de Candle

`candle-transformers/src/models/qwen3_vl/text.rs` existe (Candle 0.11.0, commit `c68b249`) mais ne convient pas tel quel :
- RoPE indexée par `seqlen_offsets` (positions contiguës) → ajouter une variante qui prend un tenseur `position_ids (B, L)` et fait un `index_select` sur les tables cos/sin.
- `forward_embeds` applique la norme finale et le `lm_head` sur le dernier token → ajouter `forward_hidden_states(input_ids, mask, position_ids, taps: &[usize]) -> Vec<Tensor>`.
- Masque : accepter causal + padding.

Préférer un fork local du module plutôt qu'une modification de la crate.

---

## 5. DiT `SingleStreamDiT` : forward exact

Entrées : `img (B, N_img, 64)` (latent patchifié), `ctx (B, 512, 12, 2560)`, `t (B,)` dans [0,1], `pos (B, 512+N_img, 3)`, `mask (B, 512+N_img)` bool.

### 5.1 Primitives

**RMSNorm « zero-centered »** (toutes les normes du DiT et de la text fusion) :
```
y = rms_norm(x.f32, weight = scale.f32 + 1.0, eps = 1e-5).to(x.dtype)
```
Le paramètre stocké est un écart à 1 (initialisé à 0). Oublier le `+1` donne des sorties nulles.

**SwiGLU** : `down( silu(gate(x)) * up(x) )`, sans biais.

**Attention** (DiT et text fusion) :
```
q = wq(x) ; k = wk(x) ; v = wv(x) ; g = gate(x)          # x = entrée normalisée/modulée
q,k,v → (B, H, L, 128) ; k,v avec H_kv têtes
q = RMSNorm_q(q) ; k = RMSNorm_k(k)                      # par tête, sur head_dim, zero-centered
si RoPE : q,k = rope(q,k)                                # en f32 puis recast
o = SDPA(q, k, v, mask, scale = 1/sqrt(128), GQA)        # → (B, L, H*128)
out = wo( o * sigmoid(g) )                               # gating élémentaire avant wo
```
GQA : répéter K/V ×4 (48/12) si le kernel ne gère pas GQA nativement.

**Embedding de timestep** :
```
half = 128 ; freqs[i] = exp(-ln(1e4) * i / 128), i ∈ [0,128)
args = (t.f32 * 1000)[:, None, None] * freqs          # (B, 1, 128)
temb = cat(cos(args), sin(args), dim=-1)              # (B, 1, 256) — ordre cos PUIS sin
```
Attention : dans la référence, `t` est créé au **dtype du modèle (bf16)** avant d'être converti en f32. Pour la parité, quantifier `t` en bf16 puis repasser en f32.

**GELU** : variante `tanh` partout (`gelu_pytorch_tanh`, en Candle `gelu` (tanh) et non `gelu_erf`).

### 5.2 Conditionnement temporel
```
tv  = tmlp(temb)                    # Linear → GELU(tanh) → Linear : (B,1,6144)
tvec = tproj(tv)                    # GELU(tanh) → Linear(6144→36864) : (B,1,36864)
```

### 5.3 Text fusion (`TextFusionTransformer`)
```
x = ctx.reshape(B*512, 12, 2560)                # attention ENTRE LES 12 COUCHES, par token
x = layerwise_block_0(x, mask=None) ; x = layerwise_block_1(x, mask=None)   # pas de RoPE
x = x.reshape(B, 512, 12, 2560).transpose → (B, 512, 2560, 12)
x = projector(x).squeeze(-1)                    # Linear(12→1) : (B, 512, 2560)
x = refiner_block_0(x, txtmask_2d) ; x = refiner_block_1(x, txtmask_2d)     # bidirectionnel, pas de RoPE
ctx' = txtmlp(x)                                # RMSNorm → Linear(2560→6144) → GELU → Linear : (B,512,6144)
```
`TextFusionBlock` : `x += attn(prenorm(x)) ; x += mlp(postnorm(x))`.
`txtmask_2d` = produit externe du masque clé `mask[:, :512]` → `(B,1,512,512)`.

### 5.4 Séquence jointe, RoPE 3D, blocs
```
img' = first(img)                                # (B, N_img, 6144)
seq  = cat(ctx', img', dim=1)                    # TEXTE D'ABORD, puis image
```
La référence complète la séquence à un multiple de 256 (padding masqué) uniquement pour stabiliser les shapes de `torch.compile`. **Facultatif en Candle** : sans effet sur le résultat.

Masque DiT : produit externe du masque clé `(B, L)` → `(B,1,L,L)`.

RoPE 3D du DiT (style Flux, **paires entrelacées** `(x[2i], x[2i+1])`) :
```
pos texte = (0,0,0) → rotation identité
pos image  = (0, row, col) avec row ∈ [0, H/16), col ∈ [0, W/16), ordre row-major
pour chaque axe a avec d_a ∈ [32, 48, 48] :
    omega_a[j] = 1 / theta^(2j/d_a),  j ∈ [0, d_a/2), theta = 1000, calcul en f64
    angle = pos[..., a] * omega_a
tables concaténées dans l'ordre axe0 (16), axe1 (24), axe2 (24) → 64 paires = head_dim/2
x'[2i]   = cos·x[2i] − sin·x[2i+1]
x'[2i+1] = sin·x[2i] + cos·x[2i+1]
```
En Candle : `candle_nn::rotary_emb::rope_i` (variante entrelacée) avec cos/sin `(L, 64)`, à valider par un test unitaire contre la référence.

Bloc single-stream (×28), `vec = tvec` :
```
m = vec + mod.lin                         # (B,1,36864)
prescale, preshift, pregate, postscale, postshift, postgate = m.chunk(6, -1)   # ORDRE EXACT
x = x + pregate  * attn( (1+prescale) * prenorm(x)  + preshift, rope, mask )
x = x + postgate * mlp ( (1+postscale) * postnorm(x) + postshift )
```
Pas de `tanh` sur les gates, pas de MLP de modulation par bloc (juste un biais appris par bloc).

### 5.5 Couche finale
Utilise `tv` (sortie de `tmlp`, **pas** `tvec`) :
```
scale = tv + last.modulation.lin[0] ; shift = tv + last.modulation.lin[1]   # (B,1,6144) chacun
y = last.linear( (1+scale) * last.norm(seq) + shift )                       # (B, L, 64)
out = y[:, 512 : 512 + N_img, :]                                           # on ne garde que l'image
```

---

## 6. Latents, patchify, scheduler, échantillonnage

### 6.1 Dimensions
- Compression VAE 8, patch 2 → `width` et `height` arrondis au multiple de **16** supérieur.
- Latent : `(B, 16, H/8, W/8)`. Tokens image : `N_img = (H/16)·(W/16)` (1024² → 4096 ; 2048² → 16384).
- Patchify : `b c (h ph) (w pw) -> b (h w) (c ph pw)`, ph=pw=2. **Canal en poids fort** dans le vecteur de 64.
- Unpatchify : inverse exact.

### 6.2 Bruit
`randn(1, 16, H/8, W/8)` par image, seed `seed + i`, générateur CUDA PyTorch en bf16. Non reproductible bit à bit en Rust : pour les tests de parité, **exporter le bruit depuis Python** et le charger.

### 6.3 Schedule de timesteps
```
ts_lin = linspace(1, 0, steps+1)
si mu non fixé :
    x1 = (256/16)^2 = 256 ; x2 = (1280/16)^2 = 6400 ; y1 = 0.5 ; y2 = 1.15
    mu = y1 + (y2 − y1)·(N_img − x1)/(x2 − x1)
ts = exp(mu) / (exp(mu) + (1/ts_lin − 1))       # sigma = 1 ; t=0 → 0, t=1 → 1
```
Valeurs de référence pour les tests :
- Turbo, 8 pas, `mu = 1.15` : `[1.0, 0.956724, 0.904531, 0.840349, 0.759511, 0.654567, 0.512844, 0.310901, 0.0]`
- Raw à 1024² : `mu = 0.90625` ; 28 pas : `[1.0, 0.985256, 0.969857, 0.953758, …]`

### 6.4 Boucle d'Euler et CFG
```
for (t_cur, t_next) in zip(ts[:-1], ts[1:]):
    cond = dit(x, ctx_pos, t_cur, pos, mask)
    if guidance > 0:
        uncond = dit(x, ctx_neg, t_cur, pos_neg, mask_neg)   # prompt négatif par défaut ""
        v = cond + guidance * (cond − uncond)                # convention Krea, PAS uncond + g·(cond−uncond)
    else:
        v = cond
    x = x + (t_next − t_cur) * v                             # pas négatif
```
`guidance = g` correspond au CFG « usuel » de `1 + g`. `guidance = 0` désactive totalement la branche négative (pas d'encodage du prompt vide).

### 6.5 Réglages recommandés

| Checkpoint | steps | guidance | mu | Résolution |
|---|---|---|---|---|
| Turbo | 8 | 0.0 | **1.15 fixe** (entraîné ainsi) | 1024² à 2048² |
| Raw | 52 (README) ; 28 par défaut CLI | 3.5 (README) ; 4.5 par défaut CLI | interpolé (§6.3) | jusqu'à ~1024² |

---

## 7. Décodeur VAE Qwen-Image (une frame)

Config `AutoencoderKLQwenImage` (diffusers 0.41.0, valeurs par défaut ; lire `vae/config.json`) : `base_dim=96`, `z_dim=16`, `dim_mult=[1,2,4,4]`, `num_res_blocks=2`, `temperal_downsample=[False,True,True]` → `temperal_upsample=[True,True,False]`, `attn_scales=[]`.

```
latents_mean = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
latents_std  = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]
```

Entrée : `z (B,16,h,w)` → ajouter T=1 → `z = z*std + mean` (par canal) → `post_quant_conv` → `decoder` → `clamp(-1,1)`.

### 7.1 Simplification T=1 (clé du portage)
- `CausalConv3d` : padding temporel causal `(2·p_t, 0)` à gauche, en zéros. Avec une seule frame et pas de cache, seule la **dernière tranche temporelle** du noyau agit. Donc `conv3d(k_t=3)` ≡ `conv2d` avec `weight[:, :, 2, :, :]` (et `k_t=1` ≡ `weight[:, :, 0]`). Padding spatial symétrique inchangé. Conversion faisable au chargement.
- `Resample upsample3d` : pour la **première frame**, le cache vaut `"Rep"` et `time_conv` n'est **pas** appliqué. Les poids `time_conv` sont donc inutilisés en text-to-image. Reste : upsample nearest ×2 (« nearest-exact » ≡ nearest pour un facteur entier) puis `Conv2d(dim → dim/2, 3, pad 1)`.
- `RMS_norm` du VAE : `F.normalize(x, dim=canal) * sqrt(C) * gamma (+ bias)`, calcul en f32. Ce n'est **pas** la RMSNorm du DiT (pas de +1, normalisation L2 × √C).
- `AttentionBlock` (mid-block uniquement) : une tête, `to_qkv` conv 1×1 (C→3C), SDPA sur les H·W pixels (scale 1/√C), `proj` 1×1, résiduel.

### 7.2 Structure du décodeur
```
conv_in : CausalConv3d(16 → 384, 3)
mid     : Res(384) → Attn(384) → Res(384)
up_blocks (dims = [384, 384, 384, 192, 96]) :
  0 : 3×Res(384→384) + upsample3d(384 → 192)
  1 : 3×Res(192→384) + upsample3d(384 → 192)
  2 : 3×Res(192→192) + upsample2d(192 → 96)
  3 : 3×Res(96→96)   (pas d'upsample)
norm_out : RMS_norm(96) → SiLU → conv_out : CausalConv3d(96 → 3, 3)
ResidualBlock : h = shortcut(x) [conv 1×1 si dims ≠] ; x = conv1(SiLU(norm1(x))) ; x = conv2(SiLU(norm2(x))) ; return x + h
```
Vérifier ces dimensions contre les shapes réelles des poids au chargement (assertions).

### 7.3 Post-traitement
`img = clamp(x,−1,1)*0.5 + 0.5` ; `(img*255).byte()` → **troncature** (pas d'arrondi) pour la parité pixel.

### 7.4 Existant Candle
Aucun VAE Wan/Qwen-Image dans `candle-transformers` (le VAE de `z_image` est un AutoencoderKL style Flux, incompatible). À porter. Pour 2048² prévoir un décodage tuilé (diffusers en propose un ; la référence Krea décode sans tuilage).

---

## 8. Dtypes et précision

| Élément | Référence | Recommandation Candle |
|---|---|---|
| Poids DiT / Qwen / VAE | bf16 | bf16 (CUDA) ; f16 possible sur Metal avec contrôle de parité |
| RMSNorm (DiT, Qwen, VAE) | calcul f32 | calcul f32, recast |
| RoPE | f32 | f32 |
| Softmax attention | kernel cuDNN | f32 interne |
| Timestep `t` | bf16 → f32 | idem (quantifier en bf16) |
| Latents pendant Euler | bf16 | bf16 pour la parité, f32 acceptable (meilleur) |

Mémoire indicative en bf16 : DiT ~26 Go, Qwen3-VL-4B ~8–9 Go, VAE < 1 Go. Pour 24 Go de VRAM : décharger Qwen après encodage (il ne sert qu'une fois par prompt), puis charger DiT, puis VAE. Quantification (fp8 / gguf) en phase 2.

---

## 9. Masques : éviter les NaN

La référence utilise des masques booléens « produit externe ». Les **requêtes** de padding ont une ligne entièrement masquée. PyTorch ≥ 2.5 renvoie 0 sur ces lignes ; un softmax naïf avec `-inf` renvoie **NaN**, qui se propage ensuite via les valeurs (0 × NaN = NaN).

Options :
1. Masque additif avec un grand négatif **fini** (ex. `-1e9` en f32 avant softmax). Les sorties des tokens de padding sont fausses mais jamais lues, et restent finies.
2. **Recommandé pour batch = 1** : supprimer physiquement les tokens de padding du texte (voir §10.1). Plus de masque dans le DiT ni dans la text fusion, flash-attention utilisable directement.

`candle-flash-attn` ne prend qu'un flag `causal`, pas de masque arbitraire : option 2 obligatoire pour l'utiliser.

---

## 10. Optimisations sûres

### 10.1 Suppression du padding (batch = 1)
Équivalence exacte (aux arrondis de kernel près), car :
- dans Qwen, les tokens valides ne voient jamais les pads et les positions des valides ne dépendent pas des pads (`cumsum`) → encoder `prefix + prompt + suffix` sans padding, positions `0..L−1` contiguës ;
- la text fusion layerwise est par token, le projector aussi ;
- refiner et DiT : les pads sont masqués en clé, leurs sorties ne sont pas lues.

Conséquence : `L_txt` variable (≤ 512). En CFG, prompt positif et négatif ont des longueurs différentes → deux passes batch=1, ou batch=2 avec masque. Garder un test de parité « avec padding » vs « sans padding ».

### 10.2 Divers
- Encodage texte une seule fois par prompt (hors boucle).
- `tvec` dépend de `t` uniquement : 8 calculs pour Turbo.
- RoPE du DiT : calculer cos/sin une fois par résolution.
- Flash-attention fortement recommandé à 2048² (séquence ~16 900 tokens).

---

## 11. Plan de travail et tests de parité

Générer des **tenseurs de référence** depuis le code officiel (PyTorch, commit §12) avec des hooks, sauvegardés en `.safetensors` :

| Étape | Tenseur | Tolérance (bf16) |
|---|---|---|
| T0 | `input_ids`, `mask` (546) pour 3 prompts dont un vide et un > 507 tokens | égalité exacte |
| T1 | 12 hidden states Qwen après slicing `(1,512,12,2560)` | rel. < 1e-2 |
| T2 | `ctx'` sortie de `txtmlp` | rel. < 2e-2 |
| T3 | `temb`, `tv`, `tvec` pour t = 1.0 et 0.512844 | rel. < 1e-3 |
| T4 | cos/sin RoPE DiT pour 1024² | abs. < 1e-5 (f32) |
| T5 | sortie du bloc 0, puis du bloc 27 | rel. < 3e-2 |
| T6 | `v` au premier pas | rel. < 3e-2 |
| T7 | latent final après 8 pas (même bruit importé) | PSNR décodé > 35 dB |
| T8 | VAE decode d'un latent fixe | abs. < 2e-2 sur [-1,1] |
| T9 | Liste `ts` (§6.3) | abs. < 1e-6 |

Ordre suggéré :
1. Scheduler + patchify (T9), sans dépendance GPU.
2. VAE decoder (T8) : testable isolément avec un latent de référence.
3. Encodeur texte Qwen fork (T0, T1).
4. DiT : primitives (RMSNorm zero-centered, attention gated, RoPE 3D), puis text fusion (T2), puis blocs (T3–T6).
5. Pipeline complet Turbo (T7), puis Raw avec CFG.
6. Optimisations §10, quantification, LoRA.

Arborescence suggérée :
```
krea2/
  config.rs        // SingleMMDiTConfig, TextEncoderConfig, VaeConfig
  text_encoder.rs  // fork Qwen3-VL texte : position_ids, taps, masque
  tokenizer.rs     // gabarit, padding au milieu, PREFIX_IDX
  mmdit.rs         // SingleStreamDiT
  rope.rs          // RoPE 3D entrelacée
  vae.rs           // décodeur Qwen-Image T=1 (conv3d → conv2d)
  sampling.rs      // timesteps, Euler, CFG convention Krea
  pipeline.rs
examples/krea2.rs  // CLI : --checkpoint turbo|raw --steps --cfg --mu --width --height --seed
```

---

## 12. Sources lues

- Code officiel : `https://github.com/krea-ai/krea-2`, commit `db3984fbc6e13b34c0064990fc2d95ac64d00058` (2026-07-24) — `mmdit.py`, `sampling.py`, `encoder.py`, `autoencoder.py`, `inference.py`, `README.md`. Dépendances : `torch>=2.9`, `transformers==4.57.1`, `diffusers>=0.32`.
- Diffusers 0.41.0 (PyPI) : `pipelines/krea2/pipeline_krea2.py`, `models/transformers/transformer_krea2.py`, `loaders/single_file_utils.py` (`convert_krea2_transformer_checkpoint_to_diffusers`), `models/autoencoders/autoencoder_kl_qwenimage.py`.
- Transformers 4.57.1 (PyPI) : `models/qwen3_vl/modeling_qwen3_vl.py` (`get_rope_index`, `Qwen3VLTextRotaryEmbedding`).
- Candle `main`, commit `c68b249` (2026-10-02), version 0.11.0 : `candle-transformers/src/models/{qwen3_vl, z_image, flux}`.
- Model card `krea/Krea-2-Turbo` (HF) et rapport technique Krea 2 (krea.ai, 2026-06-23) pour le contexte d'architecture (GQA, gated sigmoid attention, modulation légère, agrégation multi-couches, RMSNorm zero-centered, RoPE 3D axiale).

Note : le rapport technique mentionne l'usage du VAE FLUX 2 pour les plus grands modèles internes ; le code des checkpoints open-weights utilise bien le **VAE Qwen-Image** (`QwenAutoencoder`, f8, 16 canaux). Se fier au code.
