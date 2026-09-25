// [WFGY] Zone: SAFE | λ: 0.15 | Fallbacks: 0 | Action: ACE-Step task instructions & DiT prompt formatting

//! Task instructions and DiT prompt formatting for ACE-Step (text2music / repaint / cover /
//! extract / lego / complete), mirroring `acestep.constants` and `prompt_utils`.

/// ACE-Step generation task selector (`acestep.constants` task types).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AceStepTask {
    Text2Music,
    Cover,
    Repaint,
    Extract,
    Lego,
    Complete,
}

impl AceStepTask {
    pub fn as_str(self) -> &'static str {
        match self {
            AceStepTask::Text2Music => "text2music",
            AceStepTask::Cover => "cover",
            AceStepTask::Repaint => "repaint",
            AceStepTask::Extract => "extract",
            AceStepTask::Lego => "lego",
            AceStepTask::Complete => "complete",
        }
    }

    /// Parse a task string (`acestep.constants` names); unknown → `Text2Music`.
    pub fn parse(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "cover" | "cover-nofsq" => AceStepTask::Cover,
            "repaint" => AceStepTask::Repaint,
            "extract" => AceStepTask::Extract,
            "lego" => AceStepTask::Lego,
            "complete" => AceStepTask::Complete,
            _ => AceStepTask::Text2Music,
        }
    }

    /// Repaint/lego tasks drive the chunk mask + repaint injection (`can_use_repainting`).
    pub fn uses_repaint(self) -> bool {
        matches!(self, AceStepTask::Repaint | AceStepTask::Lego)
    }

    /// Lego keeps the source latents inside the masked region (`is_lego`).
    pub fn is_lego(self) -> bool {
        matches!(self, AceStepTask::Lego)
    }
}

/// SFT-stems training-compatible caption block (`conditioning_text.py`, `is_lego_sft`).
///
/// Full mode → `Global: {g}\nLocal: {l}\nMask Control: {mc}`; chunk mode
/// (`"a segment"` in the instruction) → `Local: {l}\nMask Control: {mc}`.
/// `chunk_mask_mode == "auto"` → `Mask Control: false`, else `true`.
pub fn lego_sft_caption(
    local_caption: &str,
    global_caption: &str,
    instruction: &str,
    chunk_mask_mode: &str,
) -> String {
    let mask_control = if chunk_mask_mode == "auto" {
        "Mask Control: false"
    } else {
        "Mask Control: true"
    };
    if instruction.to_ascii_lowercase().contains("a segment") {
        format!("Local: {local_caption}\n{mask_control}")
    } else {
        format!("Global: {global_caption}\nLocal: {local_caption}\n{mask_control}")
    }
}

/// Default DiT instruction (`constants.DEFAULT_DIT_INSTRUCTION`).
pub const DEFAULT_DIT_INSTRUCTION: &str =
    "Fill the audio semantic mask based on the given conditions:";

/// `constants.TASK_INSTRUCTIONS` (placeholders like `{TRACK_NAME}` are substituted by [`generate_instruction`]).
pub fn task_instruction(task_type: &str) -> &'static str {
    match task_type {
        "repaint" => "Repaint the mask area based on the given conditions:",
        "cover" | "cover-nofsq" => "Generate audio semantic tokens based on the given conditions:",
        "extract" => "Extract the track from the audio:",
        "lego" => "Generate the track based on the audio context:",
        "complete" => "Complete the input track:",
        _ => DEFAULT_DIT_INSTRUCTION,
    }
}

/// `TaskUtilsMixin.generate_instruction`, with `{TRACK_NAME}` / `{TRACK_CLASSES}` substitution.
pub fn generate_instruction(
    task_type: &str,
    track_name: Option<&str>,
    complete_track_classes: Option<&[String]>,
) -> String {
    match task_type {
        "extract" => match track_name {
            Some(t) => format!("Extract the {} track from the audio:", t.to_uppercase()),
            None => task_instruction("extract").to_string(),
        },
        "lego" => match track_name {
            Some(t) => format!("Generate the {} track based on the audio context:", t.to_uppercase()),
            None => task_instruction("lego").to_string(),
        },
        "complete" => match complete_track_classes {
            Some(c) if !c.is_empty() => format!(
                "Complete the input track with {}:",
                c.iter().map(|s| s.to_uppercase()).collect::<Vec<_>>().join(" | ")
            ),
            _ => task_instruction("complete").to_string(),
        },
        other => task_instruction(other).to_string(),
    }
}

/// `_format_instruction`: ensure the instruction ends with a colon.
pub fn format_instruction(instruction: &str) -> String {
    if instruction.ends_with(':') {
        instruction.to_string()
    } else {
        format!("{instruction}:")
    }
}

/// `_format_lyrics`: `# Languages\n{lang}\n\n# Lyric\n{lyrics}<|endoftext|>`.
pub fn format_lyrics(lyrics: &str, language: &str) -> String {
    format!("# Languages\n{language}\n\n# Lyric\n{lyrics}<|endoftext|>")
}

/// `SFT_GEN_PROMPT.format(instruction, caption, metas)`.
pub fn format_sft_caption(instruction: &str, caption: &str, metas: &str) -> String {
    format!(
        "# Instruction\n{instruction}\n\n# Caption\n{caption}\n\n# Metas\n{metas}<|endoftext|>\n"
    )
}

/// Default metadata block for a given integer duration (metres → `_dict_to_meta_string`).
pub fn default_metas(duration_seconds: f32) -> String {
    format!(
        "- bpm: N/A\n- timesignature: N/A\n- keyscale: N/A\n- duration: {} seconds\n",
        duration_seconds as i32
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instruction_substitution() {
        assert_eq!(generate_instruction("text2music", None, None), DEFAULT_DIT_INSTRUCTION);
        assert_eq!(generate_instruction("cover", None, None), "Generate audio semantic tokens based on the given conditions:");
        assert_eq!(generate_instruction("extract", Some("drums"), None), "Extract the DRUMS track from the audio:");
        assert_eq!(
            generate_instruction("complete", None, Some(&["vocals".into(), "bass".into()])),
            "Complete the input track with VOCALS | BASS:"
        );
        assert_eq!(format_instruction("Complete the input track"), "Complete the input track:");
        assert_eq!(format_instruction("X:"), "X:");
    }

    #[test]
    fn task_enum_and_lego_caption() {
        assert_eq!(AceStepTask::parse("lego"), AceStepTask::Lego);
        assert_eq!(AceStepTask::parse("COVER"), AceStepTask::Cover);
        assert!(AceStepTask::Lego.uses_repaint());
        assert!(!AceStepTask::Extract.uses_repaint());
        assert!(AceStepTask::Lego.is_lego());
        assert_eq!(
            lego_sft_caption("guitar", "rock song", "Generate the GUITAR track based on the audio context:", "true"),
            "Global: rock song\nLocal: guitar\nMask Control: true"
        );
        assert_eq!(
            lego_sft_caption("guitar", "rock song", "Generate a segment of the GUITAR track:", "auto"),
            "Local: guitar\nMask Control: false"
        );
    }
}
