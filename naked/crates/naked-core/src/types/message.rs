use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::session::TurnUsage;

/// Sentinel prefix used by the JSONL session store to externalize image
/// payloads to disk (`<session>/artifacts/img_<hash>.<ext>`) instead of
/// bloating `session.jsonl` with inline base64. This constant is the source
/// of truth — both the externalize/internalize logic in `session::jsonl_store`
/// and the API-request guard in `history::to_api_messages` reference it.
///
/// Why public: `to_api_messages` MUST drop any Text block whose payload still
/// starts with this prefix (e.g. when an artifact file went missing and
/// `intern_image_blocks` couldn't rehydrate it). Otherwise the raw marker
/// leaks to the LLM and the model sees ugly internal JSON.
pub const IMAGE_REF_SENTINEL_PREFIX: &str = "@@NAKED_IMG_REF@@";

/// Returns true if the given text contains the image-ref sentinel anywhere
/// (not just at the start). Useful for defence-in-depth checks before
/// shipping text to an LLM.
pub fn contains_image_ref_sentinel(text: &str) -> bool {
    text.contains(IMAGE_REF_SENTINEL_PREFIX)
}

/// Process-wide counter incremented every time we strip / drop a sentinel
/// before it reaches an LLM. Surfaces in `/metrics` and tests so we can spot
/// silent extern-blob corruption (artifact files vanished, intern step
/// crashed, etc.).
pub static SENTINEL_LEAK_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Bumped each time the agent loop hits a "provider returned no content"
/// stream and successfully retries (or starts retrying) — see
/// `MAX_EMPTY_CONTENT_RETRIES` in `loop_.rs`. Exposed as
/// `naked_core_empty_content_retry_total` over Prometheus so operators can
/// quantify how flaky a given provider is in production.
pub static EMPTY_CONTENT_RETRY_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Total number of agent turns that completed successfully (`run` returned
/// `Ok`). Counted in `lib.rs` after the agent loop finishes. Pair with
/// `TURN_ERROR_COUNT` to compute a turn error rate.
pub static TURN_COMPLETED_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Total number of agent turns that failed (`run` returned `Err`). Bumped
/// in the same place as `TURN_COMPLETED_COUNT` so the two are sampled in
/// lockstep. The error reason is logged via `tracing::error!` for
/// distribution analysis (the in-process counter is intentionally
/// unlabelled to keep the renderer dead-simple).
pub static TURN_ERROR_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

// ── Steer-pipeline counters (PLAN_NEXT_SESSION 2026-05-10) ─────────
//
// Three counters that pin the new steer behaviour. Operators can
// graph these to detect:
//   * `STEER_DELIVERED_COUNT` shrinking vs. `SteerMessage` channel
//     send rate → the bot is dropping user input.
//   * `STEER_SOFT_INTERRUPTED_COUNT` vs. delivered — fraction of
//     steers that hit during an in-flight LLM stream (S2/S3 path).
//   * `STEER_DRAINED_ON_ABORT_COUNT` rising = users frequently
//     abort with pending input, suggesting UX friction.
//
// All three are unlabelled `AtomicU64` for the same dead-simple
// renderer convention as the existing counters above.

/// Bumped once per successful drain in `loop_/steers.rs::drain_steers`
/// when at least one steer was merged into history.
pub static STEER_DELIVERED_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Bumped when the third arm of `stream_one_turn`'s select! fires
/// — a steer arrived mid-LLM-stream and triggered the soft-interrupt
/// path (S2/S3). Each burst-drain (multiple steers in one tick)
/// counts as one event.
pub static STEER_SOFT_INTERRUPTED_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Bumped when the run-loop's drain-on-error path (cancel / provider
/// error / empty-content giveup) actually found pending steers or
/// channel-buffered messages and rescued them into history. Zero
/// would mean the drain is a pure no-op safety net; non-zero means
/// it's actively saving user input.
pub static STEER_DRAINED_ON_ABORT_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Strip every embedded `@@NAKED_IMG_REF@@…` segment (sentinel + the
/// optional `/<hash>` or `{...json}` tail that follows it on the same line)
/// from `text`. Returns the cleaned string and bumps `SENTINEL_LEAK_COUNT`
/// once per strip.
///
/// Two payload shapes are handled:
///
/// * `@@NAKED_IMG_REF@@/abcd1234efgh…`   (newer hash form)
/// * `@@NAKED_IMG_REF@@{"path":"…"}`     (legacy JSON tail)
///
/// Anything up to the next whitespace, newline, or closing brace is stripped
/// together with the marker so we don't leave trailing garbage in the prompt.
pub fn strip_image_ref_sentinel(text: &str) -> std::borrow::Cow<'_, str> {
    if !text.contains(IMAGE_REF_SENTINEL_PREFIX) {
        return std::borrow::Cow::Borrowed(text);
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(pos) = rest.find(IMAGE_REF_SENTINEL_PREFIX) {
        out.push_str(&rest[..pos]);
        let after_marker = &rest[pos + IMAGE_REF_SENTINEL_PREFIX.len()..];
        // Determine where the marker payload ends. If the next char is `{`
        // we walk to the matching `}` (legacy JSON tail). Otherwise we eat
        // characters up to the next ASCII whitespace.
        let consumed = match after_marker.chars().next() {
            Some('{') => after_marker
                .find('}')
                .map(|i| i + 1)
                .unwrap_or(after_marker.len()),
            _ => after_marker
                .find(|c: char| c.is_ascii_whitespace())
                .unwrap_or(after_marker.len()),
        };
        rest = &after_marker[consumed..];
        SENTINEL_LEAK_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    out.push_str(rest);
    std::borrow::Cow::Owned(out)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    Thinking {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        call_id: String,
        output: String,
        is_error: bool,
    },
    /// Inline image attached to a user message. Stored as raw base64 + MIME so
    /// each provider can serialise it into its native multimodal format
    /// (Anthropic `image.source.base64`, OpenAI/Groq/xAI `image_url.url=data:`).
    Image {
        mime: String,
        data_base64: String,
        /// Per-image quality knob. OpenAI-compatible providers (gpt-4o,
        /// llama-4 vision) translate it to `image_url.detail = "low"|
        /// "high"|"auto"`, which controls token spend at inference time
        /// (`low` ~85 tokens, `high` ~tile-grid, `auto` lets the server
        /// decide). Anthropic, Gemini, and xAI ignore the field — they pick
        /// resolution server-side. `None` ⇒ "auto" / provider default.
        ///
        /// `#[serde(default)]` keeps existing session JSONL files
        /// deserializable; new images written today omit the field unless a
        /// non-default value is set.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<ImageDetail>,
    },
}

/// Quality preset for inline images. Mirrors OpenAI's `image_url.detail`
/// enum so we can pass it through unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageDetail {
    Auto,
    Low,
    High,
}

impl ImageDetail {
    /// String form for `image_url.detail`.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Low => "low",
            Self::High => "high",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMessage {
    pub role: Role,
    pub blocks: Vec<ContentBlock>,
    #[serde(default = "Utc::now")]
    pub timestamp: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<TurnUsage>,
}

impl ConversationMessage {
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            blocks: vec![ContentBlock::Text { text: text.into() }],
            timestamp: Utc::now(),
            usage: None,
        }
    }

    pub fn assistant(blocks: Vec<ContentBlock>, usage: Option<TurnUsage>) -> Self {
        Self {
            role: Role::Assistant,
            blocks,
            timestamp: Utc::now(),
            usage,
        }
    }

    pub fn tool_result(
        call_id: impl Into<String>,
        output: impl Into<String>,
        is_error: bool,
    ) -> Self {
        Self {
            role: Role::Tool,
            blocks: vec![ContentBlock::ToolResult {
                call_id: call_id.into(),
                output: output.into(),
                is_error,
            }],
            timestamp: Utc::now(),
            usage: None,
        }
    }

    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            blocks: vec![ContentBlock::Text { text: text.into() }],
            timestamp: Utc::now(),
            usage: None,
        }
    }

    pub fn text_content(&self) -> String {
        self.blocks
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    pub fn tool_uses(&self) -> Vec<(&str, &str, &serde_json::Value)> {
        self.blocks
            .iter()
            .filter_map(|b| match b {
                ContentBlock::ToolUse { id, name, input } => {
                    Some((id.as_str(), name.as_str(), input))
                }
                _ => None,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_serde_roundtrip() {
        for role in [Role::System, Role::User, Role::Assistant] {
            let json = serde_json::to_string(&role).unwrap();
            let parsed: Role = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, role);
        }
    }

    #[test]
    fn content_block_text_serde() {
        let block = ContentBlock::Text {
            text: "hello".into(),
        };
        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains("hello"));
        let parsed: ContentBlock = serde_json::from_str(&json).unwrap();
        assert!(matches!(parsed, ContentBlock::Text { text } if text == "hello"));
    }

    #[test]
    fn conversation_message_roundtrip() {
        let msg = ConversationMessage {
            role: Role::User,
            blocks: vec![ContentBlock::Text {
                text: "test".into(),
            }],
            timestamp: chrono::Utc::now(),
            usage: None,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let parsed: ConversationMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.role, Role::User);
        assert_eq!(parsed.blocks.len(), 1);
    }
}
