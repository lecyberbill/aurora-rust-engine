# Spécification technique et guide d'implémentation : Krea 2 Turbo (SingleStreamDiT / Z-Image / Lumina2)

Ce document constitue la référence technique exhaustive pour l'implémentation du moteur d'inférence **Krea 2 Turbo** en PyTorch et en Rust (framework **Candle**). Il décrit la dynamique tensorielle, le graphe de calcul, le système de conditionnement multi-couches textuel, la structure du transformeur unifié, le système de coordonnées 3D RoPE, ainsi que la correspondance exacte des clés de poids.

## 1. Vue d'ensemble du pipeline d'inférence

Le pipeline de génération d'images de Krea 2 Turbo repose sur 4 sous-systèmes interconnectés :

```
[Prompt Utilisateur]
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Encodeur de texte (Qwen3-VL 4B)                          │
│    - Application du template système (34 tokens)            │
│    - Troncature des 34 tokens de préfixe système            │
│    - Extraction de 12 états cachés (taps)                   │
└──────────────────────────────┬──────────────────────────────┘
                               │ [B, S_txt, 12, 2560]
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Conditionnement TextFusion + txtmlp                      │
│    - Layerwise Attention sur dim 12                         │
│    - Projection linéaire (12 -> 1)                          │
│    - Refiner Attention (sur S_txt)                          │
│    - Projection txtmlp (2560 -> 3840)                       │
└──────────────────────────────┬──────────────────────────────┘
                               │ [B, S_txt, 3840]
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Transformeur Unifié SingleStreamDiT (28 blocs)           │
│    - Concaténation : [txt_tokens, img_tokens]               │
│    - Modulation temporelle adaptative (DoubleShared)        │
│    - Plongement 3D RoPE entrelacé (axes 32, 48, 48)         │
│    - SingleStreamAttention avec Gating avant projection wo  │
└──────────────────────────────┬──────────────────────────────┘
                               │ Sortie img_tokens [B, S_img, 3840]
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Décodeur & Ordonnanceur                                  │
│    - Application de la couche last (RMSNorm + Linear)       │
│    - Unpatchify vers latents 16 canaux                      │
│    - Boucle FlowMatch Euler Discrete (8 pas, shift mu=1.15) │
│    - Décodage VAE Wan 2.1 / Qwen-Image (16 canaux -> RGB)   │
└─────────────────────────────────────────────────────────────┘

```

## 2. Réponses détaillées aux 4 fonctions critiques

### 2.1. TextFusion & txtmlp Forward

1. **Forme d'entrée exacte de `txtfusion.layerwise_blocks`** :

   * L'entrée est réorganisée sous la forme **`[B * S_txt, 12, 2560]`**.

   * Le nombre de couches extraites ($12$) agit comme la dimension de séquence perçue par l'opérateur d'auto-attention. Chaque token textuel calcule une attention inter-couches sur la profondeur du LLM.

2. **Enchaînement fonctionnel des modules** :
   

   $$
   \text{Tapped States } [B, S_{\text{txt}}, 12, 2560] \xrightarrow{\text{reshape}} [B \times S_{\text{txt}}, 12, 2560]
   $$

   $$
   \xrightarrow{\text{layerwise\_blocks}} [B \times S_{\text{txt}}, 12, 2560] \xrightarrow{\text{reshape}} [B, S_{\text{txt}}, 12, 2560]
   $$

   $$
   \xrightarrow{\text{permute}} [B, S_{\text{txt}}, 2560, 12] \xrightarrow{\text{projector (Linear 12}\rightarrow 1)} [B, S_{\text{txt}}, 2560, 1]
   $$

   $$
   \xrightarrow{\text{squeeze(-1)}} [B, S_{\text{txt}}, 2560] \xrightarrow{\text{refiner\_blocks}} [B, S_{\text{txt}}, 2560]
   $$

   $$
   \xrightarrow{\text{txtmlp (RMSNorm } \rightarrow \text{ Linear } \rightarrow \text{ SiLU } \rightarrow \text{ Linear)}} [B, S_{\text{txt}}, 3840]
   $$

3. **Traitement du préfixe système (34 tokens)** :

   * La troncature a lieu **avant** l'entrée dans `TextFusion`.

   * Dès la sortie de Qwen3-VL 4B, un découpage indicé `hidden_states[:, 34:, :]` est appliqué. Les 34 tokens fixes de la consigne système ne traversent pas `TextFusion`.

### 2.2. Concaténation de la séquence unifiée (Single-Stream MMDiT)

1. **Ordre des tokens dans la séquence combinée** :

   * L'ordre canonique est obligatoirement **`[txt_tokens, img_tokens]`**.

   * Les $S_{\text{txt}}$ tokens textuels occupent la tête de la séquence ($[0, S_{\text{txt}}-1]$), suivis des $S_{\text{img}}$ tokens d'image ($[S_{\text{txt}}, S_{\text{txt}} + S_{\text{img}}-1]$).

2. **Extraction finale des latents d'image** :

   * À la sortie du bloc 27, la portion textuelle est éliminée par découpage indicé :
     

     $$
     \mathbf{img\_tokens\_out} = \mathbf{seq}[:, S_{\text{txt}}:, :]
     $$

   * Le tenseur est transmis à la couche terminale `last` (RMSNorm + modulation adaptative + projection linéaire vers $16 \times 2 \times 2 = 64$ canaux).

   * L'opération `unpatchify` réorganise les patchs de $[B, S_{\text{img}}, 64]$ vers le tenseur de vitesse VAE $[B, 16, H_{\text{latent}}, W_{\text{latent}}]$.

### 2.3. Plongement rotatif tridimensionnel (3D RoPE / EmbedND)

1. **Coordonnées de position (`pos_ids`)** :

   * **Tokens texte** : Coordonnées spatiales fixées à zéro $\rightarrow (i, 0, 0)$ pour $i \in [0, S_{\text{txt}}-1]$, ou neutralisées à $(0, 0, 0)$.

   * **Tokens image** : Coordonnée temporelle $t = 0$, et coordonnées spatiales $(0, y, x)$ définies par l'index de ligne et de colonne du patch dans la grille $[0, H_{\text{patch}}-1] \times [0, W_{\text{patch}}-1]$.

2. **Nature des coordonnées d'image** :

   * Ce sont des **entiers naturels discrets non normalisés** ($0, 1, 2, \dots$).

3. **Formule de la rotation complexe 3D** :

   * Décomposition par tête ($d_{\text{head}} = 128$) sur 3 axes : $\mathbf{d}_{\text{axes}} = [32, 48, 48]$.

   * Fréquences angulaires pour un axe de dimension $d_a$ :
     

     $$
     \omega_{a, k} = 10000^{-\frac{2k}{d_a}}, \quad k \in \left[0, \frac{d_a}{2} - 1\right]
     $$

   * Application du mode **entrelacé par paires contiguës** (*interleaved*, `use_real_unbind_dim=-1`) :
     

     $$
     \text{rotate\_interleaved}([u_0, u_1, u_2, u_3, \dots]) = [-u_1, u_0, -u_3, u_2, \dots]
     $$

     $$
     \mathbf{u}' = (\mathbf{u} \odot \cos\boldsymbol{\phi}) + (\text{rotate\_interleaved}(\mathbf{u}) \odot \sin\boldsymbol{\phi})
     $$

### 2.4. Attention Gating & Modulation

1. **Positionnement de la porte `gate` dans l'attention** :

   * La porte d'attention est appliquée **strictement avant** la projection linéaire de sortie $\mathbf{W}_o$ :
     

     $$
     \mathbf{out} = \mathbf{W}_o \left( \text{Attention}(\mathbf{q}, \mathbf{k}, \mathbf{v}) \odot \sigma(\mathbf{g}) \right)
     $$

2. **Modulation adaptative via `DoubleSharedModulation`** :

   * Projection de l'embedding temporel $\mathbf{t}_{\text{emb}} \in \mathbb{R}^{3840}$ vers $\mathbb{R}^{23040}$ ($6 \times 3840$) :
     

     $$
     \mathbf{c} = \mathbf{W}_{\text{mod}}(\text{SiLU}(\mathbf{t}_{\text{emb}}))
     $$

     $$
     [\boldsymbol{\gamma}_{\text{msa}}, \boldsymbol{\beta}_{\text{msa}}, \boldsymbol{\alpha}_{\text{msa}}, \boldsymbol{\gamma}_{\text{mlp}}, \boldsymbol{\beta}_{\text{mlp}}, \boldsymbol{\alpha}_{\text{mlp}}] = \text{chunk}(\mathbf{c}, 6)
     $$

   * Application sur la branche d'attention :
     

     $$
     \mathbf{x}_{\text{norm1}} = \text{RMSNorm}_1(\mathbf{x}) \odot (1 + \boldsymbol{\gamma}_{\text{msa}}) + \boldsymbol{\beta}_{\text{msa}}
     $$

     $$
     \mathbf{x} \leftarrow \mathbf{x} + \boldsymbol{\alpha}_{\text{msa}} \odot \text{GatedAttention}(\mathbf{x}_{\text{norm1}})
     $$

   * Application sur la branche MLP :
     

     $$
     \mathbf{x}_{\text{norm2}} = \text{RMSNorm}_2(\mathbf{x}) \odot (1 + \boldsymbol{\gamma}_{\text{mlp}}) + \boldsymbol{\beta}_{\text{mlp}}
     $$

     $$
     \mathbf{x} \leftarrow \mathbf{x} + \boldsymbol{\alpha}_{\text{mlp}} \odot \text{MLP}(\mathbf{x}_{\text{norm2}})
     $$

## 3. Table de correspondance des poids Safetensors

Le tableau ci-dessous liste la correspondance entre les clés d'un fichier `.safetensors` officiel (ComfyUI / ai-toolkit / DiffSynth) et les modules de l'architecture :

| **Clé Safetensors d'origine** | **Forme du tenseur** | **Module cible** | 
| `diffusion_model.first.weight` / `.bias` | `[3840, 64]` / `[3840]` | Projection patch d'entrée ($16 \times 2 \times 2 \rightarrow 3840$) | 
| `diffusion_model.txtfusion.projector.weight` | `[1, 12]` ou `[12]` | Combination linéaire des 12 couches textuelles | 
| `diffusion_model.txtfusion.layerwise_blocks.{i}.*` | Multiples | Blocs d'attention inter-couches ($i \in \{0, 1\}$) | 
| `diffusion_model.txtfusion.refiner_blocks.{i}.*` | Multiples | Blocs de raffinement textuel ($i \in \{0, 1\}$) | 
| `diffusion_model.txtmlp.fc1.weight` / `.bias` | `[3840, 2560]` / `[3840]` | Première projection linéaire textuelle | 
| `diffusion_model.txtmlp.fc2.weight` / `.bias` | `[3840, 3840]` / `[3840]` | Seconde projection linéaire textuelle | 
| `diffusion_model.blocks.{i}.modulation.lin.weight` | `[23040, 3840]` | Projection de modulation temporelle ($6 \times 3840$) | 
| `diffusion_model.blocks.{i}.attn.wq.weight` | `[3840, 3840]` | Projection Query (attention) | 
| `diffusion_model.blocks.{i}.attn.wk.weight` | `[3840, 3840]` | Projection Key (attention) | 
| `diffusion_model.blocks.{i}.attn.wv.weight` | `[3840, 3840]` | Projection Value (attention) | 
| `diffusion_model.blocks.{i}.attn.gate.weight` | `[3840, 3840]` | Projection Gating d'attention | 
| `diffusion_model.blocks.{i}.attn.wo.weight` | `[3840, 3840]` | Projection de sortie d'attention | 
| `diffusion_model.blocks.{i}.mlp.down.weight` | `[3840, 15360]` | Projection descendante MLP | 
| `diffusion_model.blocks.{i}.mlp.up.weight` | `[15360, 3840]` | Projection montante MLP | 
| `diffusion_model.last.linear.weight` / `.bias` | `[64, 3840]` / `[64]` | Projection finale de reconstruction | 

## 4. Implémentation de référence en PyTorch

```
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * norm * self.scale

class TextMLP(nn.Module):
    def __init__(self, in_features: int = 2560, hidden_features: int = 3840):
        super().__init__()
        self.norm = RMSNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_features, hidden_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(self.norm(x))))

class TextFusion(nn.Module):
    def __init__(
        self,
        num_taps: int = 12,
        dim: int = 2560,
        out_dim: int = 3840,
        layerwise_blocks: nn.ModuleList | None = None,
        refiner_blocks: nn.ModuleList | None = None,
    ):
        super().__init__()
        self.num_taps = num_taps
        self.dim = dim
        self.layerwise_blocks = layerwise_blocks if layerwise_blocks is not None else nn.ModuleList()
        self.projector = nn.Linear(num_taps, 1, bias=False)
        self.refiner_blocks = refiner_blocks if refiner_blocks is not None else nn.ModuleList()
        self.txtmlp = TextMLP(in_features=dim, hidden_features=out_dim)

    def forward(
        self,
        tapped_hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # tapped_hidden_states: [B, seq_len, 12, 2560] après suppression des 34 tokens système
        b, seq_len, taps, d = tapped_hidden_states.shape

        # 1. Traitement par les blocs layerwise
        y = tapped_hidden_states.reshape(b * seq_len, taps, d)
        for block in self.layerwise_blocks:
            y = block(y)

        # 2. Projection de la dimension des couches (12 -> 1)
        y = y.reshape(b, seq_len, taps, d).permute(0, 1, 3, 2)
        y = self.projector(y).squeeze(-1)

        # 3. Traitement par les blocs refiners
        for block in self.refiner_blocks:
            y = block(y, mask=mask)

        # 4. Projection txtmlp
        return self.txtmlp(y)

def rotate_interleaved(x: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    x_pairs = x.reshape(*shape[:-1], -1, 2)
    x1, x2 = x_pairs[..., 0], x_pairs[..., 1]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)

def apply_3d_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_interleaved(x) * sin)

class DoubleSharedModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, 6 * dim, bias=True)

    def forward(self, vec: torch.Tensor):
        mod = self.lin(F.silu(vec))
        s_msa, b_msa, g_msa, s_mlp, b_mlp, g_mlp = mod.chunk(6, dim=-1)
        return (
            s_msa.unsqueeze(1), b_msa.unsqueeze(1), g_msa.unsqueeze(1),
            s_mlp.unsqueeze(1), b_mlp.unsqueeze(1), g_mlp.unsqueeze(1)
        )

class SingleStreamAttention(nn.Module):
    def __init__(self, dim: int = 3840, num_heads: int = 30, head_dim: int = 128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim

        self.wq = nn.Linear(dim, inner_dim, bias=False)
        self.wk = nn.Linear(dim, inner_dim, bias=False)
        self.wv = nn.Linear(dim, inner_dim, bias=False)
        self.gate = nn.Linear(dim, inner_dim, bias=False)
        self.wo = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, seq_len, _ = x.shape
        q = self.wq(x).reshape(b, seq_len, self.num_heads, self.head_dim)
        k = self.wk(x).reshape(b, seq_len, self.num_heads, self.head_dim)
        v = self.wv(x).reshape(b, seq_len, self.num_heads, self.head_dim)
        g = self.gate(x)

        q = apply_3d_rope(q, cos, sin).transpose(1, 2)
        k = apply_3d_rope(k, cos, sin).transpose(1, 2)
        v = v.transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(b, seq_len, -1)

        # GATING APPLIQUÉ STRICTEMENT AVANT wo
        return self.wo(attn_out * torch.sigmoid(g))

class SingleStreamBlock(nn.Module):
    def __init__(self, dim: int = 3840, num_heads: int = 30, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = SingleStreamAttention(dim=dim, num_heads=num_heads, head_dim=128)
        self.norm2 = RMSNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_dim, dim, bias=True),
        )
        self.modulation = DoubleSharedModulation(dim)

    def forward(self, x: torch.Tensor, vec: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        s_msa, b_msa, g_msa, s_mlp, b_mlp, g_mlp = self.modulation(vec)
        norm_x1 = self.norm1(x) * (1.0 + s_msa) + b_msa
        x = x + g_msa * self.attn(norm_x1, cos, sin)
        norm_x2 = self.norm2(x) * (1.0 + s_mlp) + b_mlp
        x = x + g_mlp * self.mlp(norm_x2)
        return x

```

## 5. Implémentation Rust sous Candle

### 5.1. Module 3D RoPE entrelacé (`rope.rs`)

```
use candle::{DType, Device, Result, Tensor};

pub struct EmbedND {
    theta: f64,
    axes_dims: [usize; 3],
}

impl EmbedND {
    pub fn new(axes_dims: [usize; 3], theta: f64) -> Self {
        Self { theta, axes_dims }
    }

    fn freqs_for_axis(&self, max_pos: usize, dim: usize, dev: &Device) -> Result<Tensor> {
        let half_dim = dim / 2;
        let mut freqs = Vec::with_capacity(half_dim);
        for i in 0..half_dim {
            let power = (2 * i) as f64 / dim as f64;
            freqs.push((1.0 / self.theta.powf(power)) as f32);
        }
        let freqs = Tensor::from_vec(freqs, (1, half_dim), dev)?;
        let pos = Tensor::arange(0u32, max_pos as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_pos, 1))?;
        
        let angles = pos.matmul(&freqs)?;
        let s = angles.shape().dims();
        let repeated = angles.reshape((s[0], s[1], 1))?
            .broadcast_as((s[0], s[1], 2))?
            .reshape((s[0], s[1] * 2))?;
        Ok(repeated)
    }

    pub fn compute_cos_sin(
        &self,
        pos_t: &Tensor,
        pos_y: &Tensor,
        pos_x: &Tensor,
        dev: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let ft = self.freqs_for_axis(64, self.axes_dims[0], dev)?;
        let fy = self.freqs_for_axis(128, self.axes_dims[1], dev)?;
        let fx = self.freqs_for_axis(128, self.axes_dims[2], dev)?;

        let rad_t = ft.index_select(pos_t, 0)?;
        let rad_y = fy.index_select(pos_y, 0)?;
        let rad_x = fx.index_select(pos_x, 0)?;

        let full_rad = Tensor::cat(&[&rad_t, &rad_y, &rad_x], 1)?;
        let cos = full_rad.cos()?.unsqueeze(0)?.unsqueeze(2)?;
        let sin = full_rad.sin()?.unsqueeze(0)?.unsqueeze(2)?;
        Ok((cos, sin))
    }
}

pub fn rotate_interleaved(x: &Tensor) -> Result<Tensor> {
    let dims = x.shape().dims();
    let b = dims[0];
    let s = dims[1];
    let h = dims[2];
    let d = dims[3];

    let x_pairs = x.reshape((b, s, h, d / 2, 2))?;
    let x1 = x_pairs.narrow(4, 0, 1)?;
    let x2 = x_pairs.narrow(4, 1, 1)?;
    let neg_x2 = x2.neg()?;

    let rotated = Tensor::cat(&[&neg_x2, &x1], 4)?;
    rotated.reshape((b, s, h, d))
}

pub fn apply_3d_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let rot = rotate_interleaved(x)?;
    let term1 = x.broadcast_mul(cos)?;
    let term2 = rot.broadcast_mul(sin)?;
    term1.add(&term2)
}

```

### 5.2. Bloc SingleStream avec Gating et Modulation (`block.rs`)

```
use candle::{Module, Result, Tensor};
use candle_nn::{linear, Linear, VarBuilder};
use crate::rope::apply_3d_rope;

pub struct SingleStreamAttention {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    gate: Linear,
    wo: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl SingleStreamAttention {
    pub fn new(vb: VarBuilder, dim: usize, num_heads: usize, head_dim: usize) -> Result<Self> {
        let inner_dim = num_heads * head_dim;
        let wq = linear(dim, inner_dim, vb.pp("wq"))?;
        let wk = linear(dim, inner_dim, vb.pp("wk"))?;
        let wv = linear(dim, inner_dim, vb.pp("wv"))?;
        let gate = linear(dim, inner_dim, vb.pp("gate"))?;
        let wo = linear(inner_dim, dim, vb.pp("wo"))?;
        Ok(Self { wq, wk, wv, gate, wo, num_heads, head_dim })
    }

    pub fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (b, s, _) = x.shape().dims3()?;

        let q = self.wq.forward(x)?.reshape((b, s, self.num_heads, self.head_dim))?;
        let k = self.wk.forward(x)?.reshape((b, s, self.num_heads, self.head_dim))?;
        let v = self.wv.forward(x)?.reshape((b, s, self.num_heads, self.head_dim))?;
        let g = self.gate.forward(x)?;

        let q = apply_3d_rope(&q, cos, sin)?;
        let k = apply_3d_rope(&k, cos, sin)?;

        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let context = attn_weights.matmul(&v)?;

        let context = context.transpose(1, 2)?.contiguous()?.reshape((b, s, ()))?;

        // Gating APPLIQUÉ STRICTEMENT AVANT wo
        let g_sig = candle_nn::ops::sigmoid(&g)?;
        let gated_context = context.mul(&g_sig)?;
        self.wo.forward(&gated_context)
    }
}

```

## 6. Stratégie de test et de validation pour l'agent LLM

1. **Test unitaire 3D RoPE (`test_rope_interleaved`)** :

   * Générer une matrice d'entrée constante et vérifier que la rotation conserve l'énergie vectorielle (norme $L_2$).

   * Vérifier que pour $pos = (0, 0, 0)$, $\text{apply\_3d\_rope}(x) == x$.

2. **Validation de TextFusion par Tenseurs Fictifs** :

   * Instancier un tenseur d'entrée aléatoire $[1, 64, 12, 2560]$ dans un script PyTorch d'une part et sous Candle d'autre part.

   * Comparer la sortie de `txtmlp` ($[1, 64, 3840]$) entre les deux moteurs. La différence relative maximale doit être inférieure à $10^{-5}$ en Float32.

3. **Vérification du découpage Séquence Unifiée** :

   * Valider que la concaténation $[S_{\text{txt}}, S_{\text{img}}]$ préserve la continuité mémoire lors du passage dans `SingleStreamBlock`.

   * S'assurer que le découpage `seq[:, txt_len:, :]` extrait la portion exacte correspondant à la taille de la grille de patchs.