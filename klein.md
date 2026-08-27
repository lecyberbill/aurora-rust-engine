Voici l'ensemble des valeurs, dimensions de séquence, mécanismes 4D RoPE et paramètres d'inférence appliqués à l'utilisation du pipeline **FLUX.2-Klein-4B** pour l'**Image-to-Image (Img2Img)**, l'**édition in-context** et l'**inpainting**.

---

### 1. Traitement VAE et Packing des Latents

Pour l'ensemble des tâches basées sur des images en entrée (Img2Img, édition, inpainting), l'image source est d'abord traitée par l'encodeur VAE (`AutoencoderKLFlux2`) :

* **Tenseur d'entrée RGB :** $1024 \times 1024 \times 3$ (ou $512 \times 512 \times 3$).


* **Encodage VAE brut :** Réduction spatiale d'un facteur 8, générant 32 canaux latents ($128 \times 128 \times 32$ pour une résolution de 1024px).


* **Réorganisation (Patchify / Packing 2×2) :** Regroupement de patches $2 \times 2$ spatiaux dans la dimension des canaux ($32 \times 4 = 128$ canaux), réduisant la grille spatiale à $64 \times 64$.


* **Projection d'entrée (`x_embedder`) :** Transformation linéaire de 128 vers la dimension interne de 3072 canaux.



---

### 2. Configuration des Longueurs de Séquence (In-Context Editing)

Dans FLUX.2-Klein-4B, l'Image-to-Image et l'édition ne reposent pas uniquement sur l'ajout de bruit latent traditionnel, mais sur l'injection de **tokens de référence** (*In-Context Reference Conditioning*) concaténés dans le transformateur.

| Tâche / Mode | Résolution Pixel | Tokens Image Générés | Tokens Image de Référence | Tokens Texte (Qwen3) | Longueur Totale Séquence Single Stream |
| --- | --- | --- | --- | --- | --- |
| Text-to-Image | 1024×1024 | 4096 | 0 | 512 | 4608 |
| Text-to-Image | 512×512 | 1024 | 0 | 512 | 1536 |
| Img2Img / Single-Ref Edit | 1024×1024 | 4096 | 4096 (1 image) | 512 | 8704 (8192 image)

 |
| Img2Img / Single-Ref Edit | 512×512 | 1024 | 1024 (1 image) | 512 | 2560 (2048 image)

 |
| Multi-Ref Edit (2 Ref) | 1024×1024 | 4096 | 8192 (2 images) | 512 | 12800 (12288 image)

 |
| Multi-Ref Edit (2 Ref) | 512×512 | 1024 | 2048 (2 images) | 512 | 3584 (3072 image)

 |

---

### 3. Gestion du 4D RoPE pour les Images de Référence

Afin que le transformateur distingue l'image cible (à générer/débruiter) des images de référence sans altérer la géométrie spatiale 2D ($X, Y$), les paramètres positionnels rotatifs sont séparés :

* **Axe 1 (Temps $T$, 32 dim) :** Les tokens de l'image cible (bruit) suivent le temps de diffusion de l'étape courante ($T = t$). Les tokens de l'image de référence occupent un temps fixe séparé (ex. $T = 10 + 10 \cdot i$ pour la $i$-ème référence).


* **Axe 2 & 3 (Hauteur $Y$ & Largeur $X$, $32+32$ dim) :** Coordonnées spatiales 2D sur la grille $64 \times 64$ (ou $32 \times 32$) pour chaque image.


* **Axe 4 (Canvas / Identity, 32 dim) :** Identificateur d'index de séquence différenciant le canevas de sortie de chaque canevas de référence.



*Remarque d'attention :* À chaque pas de débruitage, les clés et valeurs ($K, V$) des tokens de référence restent nettes et inchangées, et leurs prédictions de sortie sont ignorées pour ne conserver que la mise à jour des tokens cibles.

---

### 4. Spécifications de l'Inpainting

L'inpainting dans le pipeline Klein 4B s'effectue selon deux configurations distinctes :

1. **Inpainting par Conditionnement In-Context (Mode Recommandé) :**
* **Force de débruitage (`denoise`) :** $1.0$ (100%).


* **Mécanique :** L'image originale entière est injectée sous forme de séquence de référence ($4096$ tokens). Le masque binaire définit la région cible du canevas principal où les nouveaux latents sont générés de zéro tout en maintenant la cohérence via l'auto-attention avec la zone de référence intacte.




2. **Inpainting Latent Classique (`VAEEncodeForInpaint`) :**
* **Masque d'entrée :** Masque binaire sous-échantillonné et regroupé (packé) à la résolution latente $64 \times 64$.


* **Force de débruitage (`strength`) :** Typiquement réglée entre $0.60$ et $0.85$ pour réinjecter du bruit partiel sur les latents VAE encodés.





---

### 5. Hyperparamètres du Pipeline (Distillé vs Base)

| Paramètre du Pipeline | Variant Distillé (`FLUX.2-klein-4B`) | Variant Base (`FLUX.2-klein-base-4B`) |
| --- | --- | --- |
| **Nombre de pas (`num_inference_steps`)** | 4 pas

 | 20 à 50 pas

 |
| **Guidage CFG (`guidance_scale`)** | 1.0 (Désactivé / CFG-Free)

 | 4.0 à 8.0

 |
| **Échantillonneur (`scheduler`)** | `FlowMatchEulerDiscreteScheduler`<br> | `FlowMatchEulerDiscreteScheduler`<br> |
| **Force Img2Img recommandée** | 1.0 (In-Context Edit) ou 0.85+ (Latent)

 | 0.50 à 0.85

 |
| **Poids et Précision VAE** | `flux2-vae.safetensors` (fp16/bf16)

 | `flux2-vae.safetensors` (fp16/bf16)

 |




 Voici les détails précis concernant le pipeline d'encodage du texte, les paramètres de guidage et la configuration d'échantillonnage pour le modèle **FLUX.2-Klein-4B** (et ses variantes distillées comme *fluxKlein4BPro_v10*) :

---

### 1. Encodage du Texte et Pipeline Qwen3

* **Extraction des représentations :** Le texte n'est pas encodé par un modèle CLIP/T5 traditionnel, mais par le LLM **Qwen3-4B** (`Qwen3ForCausalLM`). Le pipeline n'utilise pas la sortie de la dernière couche (orientée prédiction du token suivant), mais extrait et intercale les états cachés (*hidden states*) des couches intermédiaires **9, 18 et 27**. Chaque couche fournissant un vecteur de dimension $2560$, la concaténation produit un tenseur de $7680$ canaux ($3 \times 2560 = 7680$).


* **Template et Prompt Système :** Le script de référence passe le prompt à travers le **Chat Template natif de Qwen3** (encapsulation sous la forme `<|im_start|>user\n{prompt}<|im_end|>`). Il n'y a **aucun system prompt spécifique** (ni *"You are a helpful assistant"*, ni instruction système BFL particulière) injecté par défaut ; le prompt brut rédigé en langage naturel est placé directement dans le rôle utilisateur. La longueur de séquence est fixée/remplie (*padded*) à **512 tokens**.


* **Masque d'attention (Causal vs Bidirectionnel) :**
* **Au sein de l'encodeur Qwen3 :** L'extraction des états cachés s'effectue avec le masque d'attention **causal** (unidirectionnel) standard du LLM.


* **Au sein du Transformateur de Diffusion (DiT) :** Une fois les 512 tokens textuels transmis au réseau de diffusion, ils sont traités en **attention bidirectionnelle complète** (auto-attention multimodale où tous les tokens de texte et d'image interagissent sans masque causal).





---

### 2. Guidance Scale et Invalidation du Vecteur de Guidage

* **Valeur de `guidance_scale` :** Le modèle distillé Klein 4B (dont dérive la version Pro v1.0) est conçu pour tourner avec **`guidance_scale = 1.0`** (mode CFG-Free, similaire à FLUX.1-Schnell).


* **Comportement de `time_guidance_embed` :** Dans la configuration JSON de Klein 4B, le paramètre **`guidance_embeds` est explicitement réglé sur `false**`. Par conséquent, le sous-module de modulation temporelle `Flux2TimestepGuidanceEmbeddings` n'injecte aucun vecteur d'échelle de guidage variable : la valeur de guidage transmise au modèle est unitaire/fixe ($1.0$) et aucun calcul inconditionnel (nécessaire au CFG habituel des modèles non-distillés) n'est exécuté.



---

### 3. Nombre d'Étapes (Steps) et Planification (Shift)

* **Nombre de pas recommandé :** Le modèle est distillé par alignement de flux (*step-distilled*) pour une inférence optimale en **4 étapes** (`num_inference_steps = 4`).


* *Note :* Augmenter le nombre de pas (ex. 20 ou 50 pas) n'améliore pas la qualité visuelle et peut créer des artefacts ou du sur-contraste, le réseau ayant été optimisé pour une trajectoire de débruitage stricte en 4 pas.




* **Échantillonneur et Shift :** Il utilise l'échantillonneur **`FlowMatchEulerDiscreteScheduler`**. Dans les pipelines officiels (Diffusers / MFlux / CoreAI), le calendrier des pas de temps (*timesteps*) applique un décalage adaptatif ($\mu$ / *shift*) calculé dynamiquement en fonction de la résolution d'image et des 4 étapes d'inférence afin d'ajuster la courbe de vitesse d'Euler.



L'image que vous obtenez (effet de « chenilles », tressage géométrique et mosaïque à haute fréquence) est le **symptôme classique et incontournable** d'une erreur d'implémentation de tenseurs dans un pipeline personnalisé (Candle Rust).

Ce problème survient à 99% en raison de l'un de ces trois bugs d'implémentation dans Candle :

1. Une mauvaise permutation d'axes lors du **Packing / Unpacking 2×2** du VAE.
2. Une mauvaise formule d'application du **4D RoPE** (utilisation d'une rotation *interleaved* type LLaMA au lieu d'une rotation *half-split*).
3. L'oubli des facteurs de décalage (**shift** et **scaling**) du VAE `AutoencoderKLFlux2`.

Voici l'ensemble des valeurs exactes, des formules mathématiques et des séquences de manipulation de tenseurs nécessaires pour corriger Candle Rust.

---

### 1. Le Packing et Unpacking VAE (Cause #1 des artefacts)

Le VAE de FLUX.2 produit un tenseur latent de dimension $B \times 32 \times H_{\text{lat}} \times W_{\text{lat}}$ (où $H_{\text{lat}} = H/8$ et $W_{\text{lat}} = W/8$). Pour une image 1024×1024, le VAE génère un tenseur de shape $(1, 32, 128, 128)$.

Le DiT (Transformateur) attend une séquence de tokens de dimension $(1, 4096, 128)$ en regroupant des blocs $2 \times 2$ spatiaux ($32 \text{ canaux} \times 2 \times 2 = 128 \text{ canaux}$).

#### Séquence exacte en Rust pour le Packing (Latents VAE → DiT Input) :

En Rust/Candle, réaliser un simple `.reshape()` sans permutation réorganise la mémoire en continu et mélange les pixels voisins avec les canaux, ce qui crée exactement la grille d'artefacts de votre photo.

```rust
// 1. Reshape initial : (B, 32, H_lat, W_lat) -> (B, 32, H_lat/2, 2, W_lat/2, 2)
// Pour 1024px : (1, 32, 64, 2, 64, 2)
let latents = latents.reshape((b, 32, h / 2, 2, w / 2, 2))?;

// 2. Permutation OBLIGATOIRE des axes pour regrouper le 2x2 spatial avec les 32 canaux :
// Nouvelle disposition des axes : (0, 2, 4, 1, 3, 5) -> (B, H_lat/2, W_lat/2, 32, 2, 2)
let latents = latents.permute((0, 2, 4, 1, 3, 5))?;

// 3. Reshape final en séquence 2D : (B, (H_lat/2)*(W_lat/2), 128)
// Pour 1024px : (1, 4096, 128)
let latents = latents.reshape((b, (h / 2) * (w / 2), 128))?;

```

#### Séquence exacte en Rust pour l'Unpacking (DiT Output → VAE Decoder Input) :

Une fois le débruitage terminé, pour repasser le tenseur $(1, 4096, 128)$ au décodeur VAE :

```rust
// 1. Reshape en grille 2D + sous-blocs : (B, H_lat/2, W_lat/2, 32, 2, 2)
let latents = latents.reshape((b, h / 2, w / 2, 32, 2, 2))?;

// 2. Permutation inverse : (0, 3, 1, 4, 2, 5) -> (B, 32, H_lat/2, 2, W_lat/2, 2)
let latents = latents.permute((0, 3, 1, 4, 2, 5))?;

// 3. Reshape final pour le VAE : (B, 32, H_lat, W_lat)
// Pour 1024px : (1, 32, 128, 128)
let latents = latents.reshape((b, 32, h, w))?;

```

---

### 2. Facteurs de Normalisation VAE (`AutoencoderKLFlux2`)

Le VAE `AutoencoderKLFlux2` requiert une normalisation affine des latents. Sans cette étape, le décodeur génère des images brûlées, extrêmement bruitées ou incohérentes.

* **`scaling_factor`** : $0.3611$
* **`shift_factor`** : $0.1159$

#### Formules à appliquer dans Candle :

1. **Avant d'envoyer un latent au Transformer (Img2Img / Inpainting) :**

$$z_{\text{norm}} = \frac{z_{\text{raw}} - 0.1159}{0.3611}$$


2. **Avant de passer le latent débruité au Décodeur VAE (VAEDecode) :**

$$z_{\text{raw}} = (z_{\text{denoised}} \times 0.3611) + 0.1159$$



---

### 3. Application du 4D RoPE (Half-Split vs Interleaved)

Chaque tête d'attention a une dimension $d_{\text{head}} = 128$. FLUX.2 divise cette tête en 4 sous-espaces de 32 dimensions (`axes_dims_rope = [32, 32, 32, 32]`).

Pour chaque sous-espace de 32 dimensions, il y a 16 paires de fréquences rotatives ($\theta_i$ pour $i \in [0..15]$) calculées avec `rope_theta = 2000` :


$$\theta_i = \frac{1}{2000^{\frac{2i}{32}}}$$

⚠️ **Piège RoPE en Rust :**
Dans les implémentations LLM standard de Candle (ex. LLaMA), la rotation RoPE s'effectue de manière entrelacée (*interleaved*) : $(x_0, x_1) \to (-x_1, x_0)$.
**FLUX.2 exige une rotation par moitié (*half-split*) !**

Pour un sous-vecteur $v$ de 32 dimensions :

* $v_1 = v[0..16]$ (la première moitié de 16 dimensions)
* $v_2 = v[16..32]$ (la seconde moitié de 16 dimensions)

La formule de rotation à appliquer sur les requêtes $Q$ et clés $K$ est :


$$\text{Rotated}(v) = \left[ v_1 \cdot \cos(\theta) - v_2 \cdot \sin(\theta), \quad v_1 \cdot \sin(\theta) + v_2 \cdot \cos(\theta) \right]$$

Si vous utilisez une fonction `apply_rope` générique entrelacée de LLM, les axes géométriques $X$ et $Y$ se tordent complètement, détruisant la cohérence de l'image.

---

### 4. Coordonnées des Identifiants Positionnels (`img_ids` et `txt_ids`)

Pour une génération Text-to-Image en 1024×1024 (grille $64 \times 64 = 4096$ tokens) :

* **`txt_ids`** : Tenseur de forme $(512, 4)$ rempli de zéros ($0.0$).
* **`img_ids`** : Tenseur de forme $(4096, 4)$ structuré comme suit :
* **Colonne 0 (Axe $T$ - Temps) :** $0.0$ pour tous les tokens.
* **Colonne 1 (Axe $Y$ - Hauteur) :** Les valeurs $0.0, 1.0, 2.0, \dots, 63.0$ répétées 64 fois pour chaque ligne de la grille.
* **Colonne 2 (Axe $X$ - Largeur) :** Les valeurs $0.0, 1.0, 2.0, \dots, 63.0$ répétées 64 fois pour chaque colonne.
* **Colonne 3 (Axe Canvas ID) :** $0.0$ pour tous les tokens.



---

### 5. Intégration du Texte (Qwen3-4B)

1. **Format du prompt :** Le prompt doit être formaté avec le template Chat de Qwen3 :
`<|im_start|>user\n{votre prompt}<|im_end|>`
2. **Padding :** Ajuster et compléter la séquence à exactement **512 tokens**.
3. **Extraction :** Récupérer les états cachés (*hidden states*) aux sorties des couches **9, 18 et 27**.
4. **Concaténation :** Concaténer ces 3 tenseurs (chacun de dimension $2560$) le long de l'axe des canaux :

$$\text{context} = \text{Concat}(h_9, h_{18}, h_{27}) \implies (1, 512, 7680)$$



---

### 6. Échantillonnage Euler (4 Steps Distillés)

1. **Initialisation des latents :** Générer un bruit gaussien aléatoire $z_0 \sim \mathcal{N}(0, I)$ de forme $(1, 4096, 128)$.
2. **Timesteps :** Pour 4 pas d'inférence, générer les pas de temps $t$ compris entre $1.0$ (bruit pur) et $0.0$ (image propre) :
$t \in [1.0, 0.75, 0.50, 0.25]$ avec un pas $dt = -0.25$.
3. **Boucle d'Inférence (Update Rule) :**
A chaque étape $k$ :

$$\text{v\_pred} = \text{Transformer}(z_k, \text{context}, t_k, \text{img\_ids}, \text{txt\_ids})$$


$$z_{k+1} = z_k + (t_{k+1} - t_k) \times \text{v\_pred}$$



*(Comme $t_{k+1} - t_k = -0.25$, cela soustrait une fraction du vecteur de prédiction).*

### Résumé des vérifications pour Candle Rust

Si votre image continue d'afficher des chenilles ou des artefacts :

1. Vérifiez que vous avez bien exécuté le `.permute((0, 2, 4, 1, 3, 5))` avant le `.reshape()` du packing.
2. Assurez-vous d'utiliser la rotation RoPE **Half-Split** ($[v_1, v_2]$) et non **Interleaved** ($[v_0, v_1, v_2, v_3]$).
3. Vérifiez la dénormalisation VAE : `(latents * 0.3611) + 0.1159` avant le `vae.decode()`.