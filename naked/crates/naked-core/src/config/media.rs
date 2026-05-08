//! Telegram media configuration.

#[allow(unused_imports)]
use super::ProviderConfig;
#[allow(unused_imports)]
use super::expand_env;
#[allow(unused_imports)]
use crate::error::Result;
#[allow(unused_imports)]
use serde::{Deserialize, Serialize};
#[allow(unused_imports)]
use std::collections::HashMap;

/// Runtime settings for Telegram media processing. All sub-sections are
/// optional — a missing vision/audio config degrades gracefully to "path only".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TgMediaConfig {
    /// Transcribe incoming voice / audio via an OpenAI-compatible Whisper API
    /// (Groq `whisper-large-v3` is the recommended default).
    #[serde(default)]
    pub audio: Option<AudioProviderCfg>,
    /// Caption / describe incoming photos via an OpenAI-compatible chat API
    /// that supports `image_url` content (xAI `grok-2-vision`, OpenAI `gpt-4o-mini`).
    #[serde(default)]
    pub vision: Option<VisionProviderCfg>,
    /// Hard size caps per media kind (applied after Telegram's 20 MB limit).
    #[serde(default)]
    pub limits: MediaLimits,
    /// Days to keep files under `workspace/.naked/artifacts/`; older dirs are
    /// swept at bot startup. `0` disables sweeping.
    #[serde(default = "default_artifact_retention_days")]
    pub artifact_retention_days: u64,
    /// Maximum document size to inline into the prompt (bytes). Larger docs
    /// are saved to disk and only the path is relayed.
    #[serde(default = "default_docs_inline_max")]
    pub docs_inline_max_bytes: u64,
    /// When true, photos/stickers are passed to the **main** model as native
    /// image content blocks (Anthropic `image.source.base64`, OpenAI/Groq/xAI
    /// `image_url.url=data:`) IF the active model is known to support vision.
    /// When false (or the model is text-only), falls back to running the
    /// `vision` describer above and inlining the **text** description into the
    /// prompt — the legacy behaviour. Default: `true`.
    #[serde(default = "default_native_image_context")]
    pub native_image_context: bool,
    /// Hard cap on bytes per image when `native_image_context` is on. Larger
    /// photos are described instead of being uploaded as base64 (avoids
    /// blowing up context windows). Default: 4 MB (matches Groq's hard cap).
    #[serde(default = "default_native_image_max_bytes")]
    pub native_image_max_bytes: u64,
    /// Optional substring matchers added on top of the built-in vision-model
    /// allowlist (`is_vision_capable_model`). Useful for new model IDs we
    /// haven't hard-coded yet — e.g. `["llama-4", "qwen-vl"]`.
    #[serde(default)]
    pub vision_model_extras: Vec<String>,
    /// Per-model vision capability override map. **Key is a substring** matched
    /// case-insensitively against the model id; value forces vision on (`true`)
    /// or off (`false`) regardless of needles or `vision_model_extras`. Useful
    /// for OpenRouter-style model ids where one provider hosts both
    /// vision-capable and text-only variants under similar names. Resolution
    /// order: this map → provider override → built-in needles → extras.
    #[serde(default)]
    pub model_vision_overrides: HashMap<String, bool>,
    /// Default OpenAI-style `image_url.detail` knob. Accepted values:
    /// `"auto"` (default — provider decides), `"low"` (~85 tokens, ~512×512
    /// downsample), `"high"` (tile-grid, full resolution). Anthropic, Gemini,
    /// xAI ignore the field. Set this to `"low"` to slash token spend on noisy
    /// chat photos; set to `"high"` for screenshots / OCR scenarios where
    /// detail matters more than cost.
    #[serde(default)]
    pub image_detail: Option<crate::types::ImageDetail>,
}

impl Default for TgMediaConfig {
    fn default() -> Self {
        Self {
            audio: None,
            vision: None,
            limits: MediaLimits::default(),
            artifact_retention_days: default_artifact_retention_days(),
            docs_inline_max_bytes: default_docs_inline_max(),
            native_image_context: default_native_image_context(),
            native_image_max_bytes: default_native_image_max_bytes(),
            vision_model_extras: Vec::new(),
            model_vision_overrides: HashMap::new(),
            image_detail: None,
        }
    }
}

impl TgMediaConfig {
    /// True if `model` is known to accept inline images. Combines a built-in
    /// allowlist (Anthropic Claude 3+, GPT-4o family, Groq Llama-4 + Maverick,
    /// xAI grok-2-vision, Gemini, Qwen-VL) with `vision_model_extras` from
    /// config. Matching is case-insensitive substring.
    pub fn is_vision_capable_model(&self, model: &str) -> bool {
        self.is_vision_capable_with_provider(model, None)
    }

    /// Check vision capability with optional provider-level override.
    ///
    /// Resolution order (first hit wins):
    /// 1. `tg_media.model_vision_overrides[<substring>]` — most specific.
    /// 2. `provider.supports_vision = Some(true|false)` — provider-wide override.
    /// 3. `BUILTIN_VISION_MODEL_NEEDLES` substring match.
    /// 4. `tg_media.vision_model_extras` user-supplied substring match.
    pub fn is_vision_capable_with_provider(
        &self,
        model: &str,
        provider: Option<&ProviderConfig>,
    ) -> bool {
        let m = model.to_ascii_lowercase();
        for (key, &force) in &self.model_vision_overrides {
            if !key.is_empty() && m.contains(&key.to_ascii_lowercase()) {
                return force;
            }
        }
        if let Some(pc) = provider
            && let Some(force) = pc.supports_vision
        {
            return force;
        }
        for needle in BUILTIN_VISION_MODEL_NEEDLES {
            if m.contains(needle) {
                return true;
            }
        }
        for extra in &self.vision_model_extras {
            if !extra.is_empty() && m.contains(&extra.to_ascii_lowercase()) {
                return true;
            }
        }
        false
    }

    /// Per-provider hard cap (decoded bytes) for inline image payloads. The
    /// global `native_image_max_bytes` is the floor; provider-specific limits
    /// further trim it down where the upstream API enforces a stricter ceiling.
    /// Conservative values from public API docs (2026-04):
    /// * Anthropic — 5 MB per image, max 100 per request.
    /// * OpenAI — 20 MB per image (gpt-4o family).
    /// * Groq — 4 MB per image (llama-4 vision).
    /// * xAI Grok — 10 MB per image.
    /// * Gemini — 7 MB inline; larger via the File API.
    /// * Anything unknown — fall back to `native_image_max_bytes`.
    pub fn provider_image_cap(&self, provider_type: &str, base_url: Option<&str>) -> u64 {
        let global = self.native_image_max_bytes;
        let cap_for = |bytes: u64| global.min(bytes);
        let url_lc = base_url.unwrap_or("").to_ascii_lowercase();
        match provider_type {
            "anthropic" => cap_for(5 * 1024 * 1024),
            "copilot" => cap_for(5 * 1024 * 1024), // routes through Anthropic/OpenAI; pick stricter floor.
            _ if url_lc.contains("groq.com") => cap_for(4 * 1024 * 1024),
            _ if url_lc.contains("api.openai.com") => cap_for(20 * 1024 * 1024),
            _ if url_lc.contains("api.x.ai") => cap_for(10 * 1024 * 1024),
            _ if url_lc.contains("googleapis.com") || url_lc.contains("generativelanguage") => {
                cap_for(7 * 1024 * 1024)
            }
            _ if url_lc.contains("openrouter.ai") => cap_for(20 * 1024 * 1024),
            _ => global,
        }
    }
}

/// Hard-coded substrings of model IDs that accept vision input today.
/// Conservative — when in doubt, leave out and let users add to
/// `tg_media.vision_model_extras`.
const BUILTIN_VISION_MODEL_NEEDLES: &[&str] = &[
    // Anthropic — every Claude 3+ model is multimodal.
    "claude-3",
    "claude-sonnet-4",
    "claude-opus-4",
    "claude-haiku-4",
    "claude-4",
    // OpenAI
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-4-vision",
    "gpt-5",
    "o1",
    "o3",
    "o4",
    // Groq multimodal
    "llama-4-scout",
    "llama-4-maverick",
    "llama-3.2-11b-vision",
    "llama-3.2-90b-vision",
    // xAI
    "grok-2-vision",
    "grok-3-vision",
    "grok-4",
    // Google
    "gemini-1.5",
    "gemini-2",
    "gemini-pro-vision",
    // Qwen / others
    "qwen-vl",
    "qwen2-vl",
    "qwen2.5-vl",
    "pixtral",
    "minicpm-v",
];

/// OpenAI-compatible Whisper endpoint for audio transcription.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioProviderCfg {
    /// e.g. `https://api.groq.com/openai/v1/audio/transcriptions`
    pub api_url: String,
    /// Direct key, or `$ENV` reference.
    pub api_key: String,
    /// e.g. `whisper-large-v3`
    pub model: String,
    /// Optional ISO-639-1 language hint (`ru`, `en`, ...). `None` → auto.
    #[serde(default)]
    pub language: Option<String>,
}

/// Vision describer. Uses OpenAI-compatible chat-completions with
/// `image_url: data:...;base64,...` content parts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionProviderCfg {
    /// e.g. `https://api.x.ai/v1/chat/completions`
    pub api_url: String,
    pub api_key: String,
    /// e.g. `grok-2-vision`, `gpt-4o-mini`
    pub model: String,
    /// Max tokens for the description. Default 400 is enough for ~200 words.
    #[serde(default = "default_vision_max_tokens")]
    pub max_tokens: u32,
    /// Override the default describer prompt. Placeholders: `{caption}`.
    #[serde(default)]
    pub prompt_override: Option<String>,
}

/// Strict per-kind byte caps. Defaults: photo 5 MB, audio 20 MB, doc 10 MB.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaLimits {
    #[serde(default = "default_photo_max_bytes")]
    pub photo_max_bytes: u64,
    #[serde(default = "default_audio_max_bytes")]
    pub audio_max_bytes: u64,
    #[serde(default = "default_doc_max_bytes")]
    pub doc_max_bytes: u64,
}

impl Default for MediaLimits {
    fn default() -> Self {
        Self {
            photo_max_bytes: default_photo_max_bytes(),
            audio_max_bytes: default_audio_max_bytes(),
            doc_max_bytes: default_doc_max_bytes(),
        }
    }
}

fn default_photo_max_bytes() -> u64 {
    5 * 1024 * 1024
}
fn default_audio_max_bytes() -> u64 {
    20 * 1024 * 1024
}
fn default_doc_max_bytes() -> u64 {
    10 * 1024 * 1024
}
fn default_artifact_retention_days() -> u64 {
    7
}
fn default_docs_inline_max() -> u64 {
    128 * 1024
}
fn default_vision_max_tokens() -> u32 {
    400
}
fn default_native_image_context() -> bool {
    true
}
fn default_native_image_max_bytes() -> u64 {
    4 * 1024 * 1024
}

impl AudioProviderCfg {
    /// Resolve the API key: `$VAR` → env lookup, otherwise literal.
    pub fn resolved_api_key(&self) -> Result<String> {
        expand_env(&self.api_key)
    }
}

impl VisionProviderCfg {
    pub fn resolved_api_key(&self) -> Result<String> {
        expand_env(&self.api_key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_media_config_parses() {
        let json = "{}";
        let cfg: TgMediaConfig = serde_json::from_str(json).unwrap();
        assert!(cfg.audio.is_none());
        assert!(cfg.vision.is_none());
    }

    #[test]
    fn default_limits_sane() {
        let limits = MediaLimits::default();
        assert!(limits.photo_max_bytes > 0);
    }

    #[test]
    fn vision_capable_model_detection() {
        let cfg = TgMediaConfig::default();
        assert!(cfg.is_vision_capable_model("gpt-4o"));
        assert!(cfg.is_vision_capable_model("grok-2-vision"));
        assert!(!cfg.is_vision_capable_model("gpt-3.5-turbo"));
    }
}
