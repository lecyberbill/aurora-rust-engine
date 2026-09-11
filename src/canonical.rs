// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Canonical weight vocabulary + per-family checkpoint adapters (decouples model families)

//! Canonical weight vocabulary and per-family **checkpoint adapters**.
//!
//! The transformer blocks read a single **canonical** set of parameter names (the current Flux.1
//! internal names: `img_in`, `txt_in`, `vector_in.*`, `double_blocks.N.img_attn.qkv`, ...). Every
//! checkpoint family (BFL Flux, BFL SD3.5, Diffusers) has its **own adapter** that maps its raw keys
//! onto that canonical vocabulary. This decouples families: touching the Flux adapter cannot silently
//! break SD3.5, and each adapter is covered by a unit test.
//!
//! Canonical vocabulary (contract, stable):
//! ```text
//! img_in.{weight,bias}                       txt_in.{weight,bias}
//! time_in.mlp.{0,2}.{weight,bias}
//! vector_in.in_layer.{weight,bias}           vector_in.out_layer.{weight,bias}
//! pos_embed
//! double_blocks.N.{img,txt}_attn.qkv.{weight,bias}
//! double_blocks.N.{img,txt}_attn.proj.{weight,bias}
//! double_blocks.N.{img,txt}_attn.norm.{query,key}_norm.{weight,bias}
//! double_blocks.N.{img,txt}_mlp.{0,2}.{weight,bias}
//! double_blocks.N.{img,txt}_mod.lin.{weight,bias}
//! single_blocks.N.*                          (Flux only)
//! final_layer.adaLN_modulation.1.{weight,bias}
//! final_layer.linear.{weight,bias}
//! ```

/// The checkpoint key convention family. Each maps to the canonical vocabulary differently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointFamily {
    /// BFL-native Flux.1 / Flux.2 (`double_blocks.N.img_attn.qkv`, `img_in`, ...): already canonical.
    BflFlux,
    /// BFL-native SD3 / SD3.5 (`joint_blocks.N.x_block.attn.qkv`, `x_embedder.proj`, ...).
    BflSd35,
    /// Diffusers `transformer.transformer_blocks.N.attn.to_q` / `single_transformer_blocks.N...`.
    Diffusers,
}

/// Detect the checkpoint family from the raw key set.
pub fn detect_family<'a>(keys: impl IntoIterator<Item = &'a str>) -> CheckpointFamily {
    let mut has_sd35 = false;
    let mut has_bfl = false;
    let mut has_diffusers = false;
    for k in keys {
        if k.contains("joint_blocks.") || k.contains("x_embedder.") || k.contains("context_embedder.") {
            has_sd35 = true;
        }
        if k.contains("double_blocks.") || k.contains("single_blocks.") {
            has_bfl = true;
        }
        if k.contains("transformer_blocks.") || k.contains("transformer.single_transformer_blocks.") {
            has_diffusers = true;
        }
    }
    if has_sd35 { CheckpointFamily::BflSd35 }
    else if has_diffusers { CheckpointFamily::Diffusers }
    else if has_bfl { CheckpointFamily::BflFlux }
    else { CheckpointFamily::BflFlux }
}

/// Map a raw checkpoint key to the canonical vocabulary, or `None` if it's not a transformer key.
pub fn canonicalize(family: CheckpointFamily, raw: &str) -> Option<String> {
    match family {
        CheckpointFamily::BflFlux => Some(raw.to_string()),
        CheckpointFamily::BflSd35 => sd35_to_canonical(raw),
        CheckpointFamily::Diffusers => crate::weights::flux_diffusers_to_bfl(raw),
    }
}

/// SD3 / SD3.5 (BFL native) → canonical. Split the block-scope prefix and the leaf suffix so the
/// `.weight` / `.bias` (or `.scale`) is preserved verbatim.
fn sd35_to_canonical(raw: &str) -> Option<String> {
    let body = raw.strip_prefix("model.diffusion_model.").unwrap_or(raw);

    // Non-block embedders / final layer.
    let direct = [
        ("x_embedder.proj", "img_in"),
        ("context_embedder", "txt_in"),
        ("y_embedder.mlp.0", "vector_in.in_layer"),
        ("y_embedder.mlp.2", "vector_in.out_layer"),
        // The engine's TimestepEmbedder reads `in_layer`/`out_layer` (Flux naming), so map SD3's
        // `t_embedder.mlp.0/.2` onto that canonical shape.
        ("t_embedder.mlp.0", "time_in.in_layer"),
        ("t_embedder.mlp.2", "time_in.out_layer"),
        ("pos_embed", "pos_embed"),
        ("final_layer.adaLN_modulation.1", "final_layer.adaLN_modulation.1"),
        ("final_layer.linear", "final_layer.linear"),
    ];
    for (from, to) in direct {
        if let Some(suffix) = strip_suffix_after(body, from) {
            return Some(format!("{to}{suffix}"));
        }
    }

    // Block-scoped: joint_blocks.N.{x_block|context_block}.<leaf>
    let rest = body.strip_prefix("joint_blocks.")?;
    let dot = rest.find('.')?;
    let idx = &rest[..dot];
    if idx.parse::<usize>().is_err() {
        return None;
    }
    let leaf = &rest[dot + 1..];

    let (stream, mapped_leaf) = if let Some(l) = leaf.strip_prefix("x_block.") {
        ("img", remap_block_leaf(l)?)
    } else if let Some(l) = leaf.strip_prefix("context_block.") {
        ("txt", remap_block_leaf(l)?)
    } else {
        return None;
    };
    Some(format!("double_blocks.{idx}.{stream}_{mapped_leaf}"))
}

/// Map an SD3 block leaf (`attn.qkv.weight`, `mlp.fc1.bias`, `adaLN_modulation.1.weight`) to the
/// canonical Flux-internal leaf (`attn.qkv.weight`, `mlp.0.bias`, `mod.lin.weight`).
fn remap_block_leaf(leaf: &str) -> Option<String> {
    // Preserve the trailing `.weight` / `.bias` / `.scale`.
    let (stem, ext) = split_ext(leaf);
    let mapped = match stem {
        "attn.qkv" => "attn.qkv",
        "attn.proj" => "attn.proj",
        // SD3/3.5 `ln_q`/`ln_k` are RMSNorm weights (diffusers `qk_norm="rms_norm"`), not LayerNorm.
        "attn.ln_q" => "attn.norm.query_norm",
        "attn.ln_k" => "attn.norm.key_norm",
        "mlp.fc1" => "mlp.0",
        "mlp.fc2" => "mlp.2",
        "adaLN_modulation.1" => "mod.lin",
        _ => return None,
    };
    Some(format!("{mapped}{ext}"))
}

/// `"attn.qkv.weight"` → `("attn.qkv", ".weight")`.
fn split_ext(s: &str) -> (&str, &str) {
    for ext in [".weight", ".bias", ".scale"] {
        if let Some(stem) = s.strip_suffix(ext) {
            return (stem, ext);
        }
    }
    (s, "")
}

/// If `body` starts with `prefix` and the next char is `.` or end, return the remainder *including*
/// the leading dot (e.g. `.weight`). `"x_embedder.proj.weight"`, `"x_embedder.proj"` → `Some(".weight")`.
fn strip_suffix_after<'a>(body: &'a str, prefix: &str) -> Option<&'a str> {
    let rest = body.strip_prefix(prefix)?;
    if rest.is_empty() || rest.starts_with('.') {
        Some(rest)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sd35_embedders_map_to_canonical() {
        assert_eq!(sd35_to_canonical("model.diffusion_model.x_embedder.proj.weight").as_deref(), Some("img_in.weight"));
        assert_eq!(sd35_to_canonical("model.diffusion_model.context_embedder.weight").as_deref(), Some("txt_in.weight"));
        assert_eq!(sd35_to_canonical("model.diffusion_model.y_embedder.mlp.0.weight").as_deref(), Some("vector_in.in_layer.weight"));
        assert_eq!(sd35_to_canonical("model.diffusion_model.t_embedder.mlp.2.bias").as_deref(), Some("time_in.out_layer.bias"));
        assert_eq!(sd35_to_canonical("model.diffusion_model.pos_embed").as_deref(), Some("pos_embed"));
    }

    #[test]
    fn sd35_blocks_map_to_canonical() {
        assert_eq!(
            sd35_to_canonical("model.diffusion_model.joint_blocks.7.x_block.attn.qkv.weight").as_deref(),
            Some("double_blocks.7.img_attn.qkv.weight")
        );
        assert_eq!(
            sd35_to_canonical("model.diffusion_model.joint_blocks.3.context_block.mlp.fc1.bias").as_deref(),
            Some("double_blocks.3.txt_mlp.0.bias")
        );
        assert_eq!(
            sd35_to_canonical("model.diffusion_model.joint_blocks.0.x_block.attn.ln_q.weight").as_deref(),
            Some("double_blocks.0.img_attn.norm.query_norm.weight")
        );
        assert_eq!(
            sd35_to_canonical("model.diffusion_model.joint_blocks.12.x_block.adaLN_modulation.1.weight").as_deref(),
            Some("double_blocks.12.img_mod.lin.weight")
        );
        assert_eq!(
            sd35_to_canonical("model.diffusion_model.final_layer.linear.weight").as_deref(),
            Some("final_layer.linear.weight")
        );
    }

    #[test]
    fn family_detection() {
        assert_eq!(detect_family(["model.diffusion_model.joint_blocks.0.x_block.attn.qkv.weight"]), CheckpointFamily::BflSd35);
        assert_eq!(detect_family(["double_blocks.0.img_attn.qkv.weight"]), CheckpointFamily::BflFlux);
        assert_eq!(detect_family(["transformer.transformer_blocks.0.attn.to_q.weight"]), CheckpointFamily::Diffusers);
    }

    #[test]
    fn bfl_flux_is_identity() {
        let k = "double_blocks.0.img_attn.qkv.weight";
        assert_eq!(canonicalize(CheckpointFamily::BflFlux, k).as_deref(), Some(k));
    }
}
