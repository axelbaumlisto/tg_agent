use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

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

// -- Streaming types (not persisted) -----------------------------------------

#[derive(Debug, Clone)]
pub enum StreamChunk {
    Text(String),
    Thinking(String),
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    Usage(TurnUsage),
    Done,
    Error(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolState {
    Completed,
    Error,
}

#[derive(Debug, Clone)]
pub enum AgentEvent {
    ThinkingDelta(String),
    TextDelta(String),
    ToolStart {
        call_id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolEnd {
        call_id: String,
        name: String,
        state: ToolState,
        output: String,
    },
    PermissionRequest {
        call_id: String,
        tool_name: String,
        input: serde_json::Value,
        permission: Permission,
    },
    /// Periodic signal during long tool execution — keeps watchers alive.
    Heartbeat,
    /// Progress from a child sub-agent forwarded to the parent.
    SubAgentProgress {
        agent_id: String,
        event: SubAgentEvent,
    },
    UsageUpdate(TurnUsage),
    ContextCompacted {
        before_msgs: usize,
        after_msgs: usize,
        /// Short label from the compaction summary (e.g. first line of ## Goal).
        summary_hint: Option<String>,
        /// Number of tracked files (read + modified).
        files_count: usize,
    },
    Error(String),
    Idle,
}

/// Lightweight subset of child events forwarded to the parent agent.
#[derive(Debug, Clone)]
pub enum SubAgentEvent {
    Started { prompt_preview: String },
    ToolUse { name: String, input_preview: String },
    ToolDone { name: String, state: ToolState },
    TextDelta(String),
    Finished { tokens: u64 },
    Error(String),
}

#[derive(Debug, Clone)]
pub struct PermissionResponse {
    pub call_id: String,
    pub allowed: bool,
}

/// Returned by `AgentCore::send_prompt` — events channel + permission reply channel.
pub struct AgentHandle {
    pub events: tokio::sync::mpsc::Receiver<AgentEvent>,
    pub permissions: tokio::sync::mpsc::Sender<PermissionResponse>,
}

// -- Usage -------------------------------------------------------------------

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TurnUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    #[serde(default)]
    pub cache_read_tokens: u64,
    #[serde(default)]
    pub cache_write_tokens: u64,
}

impl TurnUsage {
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens + self.output_tokens + self.cache_read_tokens + self.cache_write_tokens
    }

    /// Estimate USD cost based on model name. Returns `(input_cost, output_cost, total)`.
    pub fn estimate_cost(&self, model: &str) -> (f64, f64, f64) {
        let (inp_per_m, out_per_m) = model_pricing(model);
        let input_cost =
            (self.input_tokens as f64 + self.cache_read_tokens as f64 * 0.1) * inp_per_m / 1e6;
        let output_cost = self.output_tokens as f64 * out_per_m / 1e6;
        (input_cost, output_cost, input_cost + output_cost)
    }
}

fn model_pricing(model: &str) -> (f64, f64) {
    let m = model.to_ascii_lowercase();
    if m.contains("opus") {
        (15.0, 75.0)
    } else if m.contains("haiku") {
        (0.25, 1.25)
    } else if m.contains("sonnet") || m.contains("claude-3-5") || m.contains("claude-3.5") {
        (3.0, 15.0)
    } else if m.contains("gpt-4o-mini") {
        (0.15, 0.60)
    } else if m.contains("gpt-4o") || m.contains("gpt-4-turbo") {
        (5.0, 15.0)
    } else if m.contains("gpt-4") {
        (30.0, 60.0)
    } else if m.contains("gpt-3.5") {
        (0.50, 1.50)
    } else if m.contains("deepseek") {
        (0.14, 0.28)
    } else if m.contains("llama")
        || m.contains("mixtral")
        || m.contains("minimax")
        || m.contains("m2p7")
    {
        (0.20, 0.20)
    } else if m.contains("glm") || m.contains("chatglm") {
        (0.10, 0.10)
    } else if m.contains("moonshot") || m.contains("kimi") {
        (0.30, 0.30)
    } else {
        (1.0, 3.0)
    }
}

// -- Tool spec ---------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Permission {
    ReadOnly,
    WorkspaceWrite,
    Dangerous,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
    pub permission: Permission,
}

#[derive(Debug, Clone)]
pub struct ToolResult {
    pub output: String,
    pub is_error: bool,
}

// -- Model info --------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub provider: String,
    pub model_id: String,
    pub display_name: String,
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_sentinel_no_marker_is_borrowed() {
        let s = "plain text without marker";
        let out = strip_image_ref_sentinel(s);
        assert!(matches!(out, std::borrow::Cow::Borrowed(_)));
        assert_eq!(out, s);
    }

    #[test]
    fn strip_sentinel_hash_form() {
        let baseline = SENTINEL_LEAK_COUNT.load(std::sync::atomic::Ordering::Relaxed);
        let s = "before @@NAKED_IMG_REF@@/abc123def after";
        let out = strip_image_ref_sentinel(s);
        assert_eq!(out, "before  after");
        assert!(SENTINEL_LEAK_COUNT.load(std::sync::atomic::Ordering::Relaxed) > baseline);
    }

    #[test]
    fn strip_sentinel_legacy_json_form() {
        let s = r#"head @@NAKED_IMG_REF@@{"path":"a.png","mime":"image/png"} tail"#;
        let out = strip_image_ref_sentinel(s);
        assert_eq!(out, "head  tail");
    }

    #[test]
    fn strip_sentinel_multiple_markers() {
        let s = "a @@NAKED_IMG_REF@@/x b @@NAKED_IMG_REF@@/y c";
        let out = strip_image_ref_sentinel(s);
        assert_eq!(out, "a  b  c");
    }

    #[test]
    fn strip_sentinel_at_eol_no_trailing_garbage() {
        let s = "foo @@NAKED_IMG_REF@@/abc";
        let out = strip_image_ref_sentinel(s);
        assert_eq!(out, "foo ");
    }

    #[test]
    fn contains_image_ref_sentinel_detects_anywhere() {
        assert!(!contains_image_ref_sentinel("plain"));
        assert!(contains_image_ref_sentinel("@@NAKED_IMG_REF@@/x"));
        assert!(contains_image_ref_sentinel(
            "noise before @@NAKED_IMG_REF@@/x noise after"
        ));
    }

    #[test]
    fn conversation_message_user() {
        let msg = ConversationMessage::user("hello");
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.text_content(), "hello");
        assert!(msg.usage.is_none());
    }

    #[test]
    fn conversation_message_assistant_with_tool_use() {
        let blocks = vec![
            ContentBlock::Text {
                text: "let me check".into(),
            },
            ContentBlock::ToolUse {
                id: "c1".into(),
                name: "bash".into(),
                input: serde_json::json!({"command": "ls"}),
            },
        ];
        let msg = ConversationMessage::assistant(blocks, None);
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.text_content(), "let me check");
        let uses = msg.tool_uses();
        assert_eq!(uses.len(), 1);
        assert_eq!(uses[0].1, "bash");
    }

    #[test]
    fn conversation_message_tool_result() {
        let msg = ConversationMessage::tool_result("c1", "file.txt", false);
        assert_eq!(msg.role, Role::Tool);
        assert!(msg.tool_uses().is_empty());
    }

    #[test]
    fn conversation_message_system() {
        let msg = ConversationMessage::system("You are helpful");
        assert_eq!(msg.role, Role::System);
        assert_eq!(msg.text_content(), "You are helpful");
    }

    #[test]
    fn text_content_joins_multiple_blocks() {
        let msg = ConversationMessage::assistant(
            vec![
                ContentBlock::Text { text: "a".into() },
                ContentBlock::ToolUse {
                    id: "x".into(),
                    name: "y".into(),
                    input: serde_json::json!({}),
                },
                ContentBlock::Text { text: "b".into() },
            ],
            None,
        );
        assert_eq!(msg.text_content(), "ab");
    }

    #[test]
    fn turn_usage_total_tokens() {
        let u = TurnUsage {
            input_tokens: 100,
            output_tokens: 50,
            cache_read_tokens: 10,
            cache_write_tokens: 5,
        };
        assert_eq!(u.total_tokens(), 165);
    }

    #[test]
    fn turn_usage_default_is_zero() {
        let u = TurnUsage::default();
        assert_eq!(u.total_tokens(), 0);
    }

    #[test]
    fn estimate_cost_sonnet() {
        let u = TurnUsage {
            input_tokens: 1_000_000,
            output_tokens: 100_000,
            ..Default::default()
        };
        let (inp, out, total) = u.estimate_cost("claude-3-5-sonnet");
        assert!((inp - 3.0).abs() < 0.01);
        assert!((out - 1.5).abs() < 0.01);
        assert!((total - 4.5).abs() < 0.01);
    }

    #[test]
    fn estimate_cost_unknown_model() {
        let u = TurnUsage {
            input_tokens: 1000,
            output_tokens: 500,
            ..Default::default()
        };
        let (_, _, total) = u.estimate_cost("some-unknown-model");
        assert!(total > 0.0);
    }

    #[test]
    fn content_block_serde_round_trip() {
        let blocks = vec![
            ContentBlock::Text { text: "hi".into() },
            ContentBlock::ToolUse {
                id: "1".into(),
                name: "bash".into(),
                input: serde_json::json!({"cmd": "ls"}),
            },
            ContentBlock::ToolResult {
                call_id: "1".into(),
                output: "ok".into(),
                is_error: false,
            },
        ];
        let json = serde_json::to_string(&blocks).unwrap();
        let parsed: Vec<ContentBlock> = serde_json::from_str(&json).unwrap();
        assert_eq!(blocks, parsed);
    }

    #[test]
    fn conversation_message_serde_round_trip() {
        let msg = ConversationMessage::user("test");
        let json = serde_json::to_string(&msg).unwrap();
        let parsed: ConversationMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.role, Role::User);
        assert_eq!(parsed.text_content(), "test");
    }

    #[test]
    fn role_serde_values() {
        assert_eq!(serde_json::to_string(&Role::User).unwrap(), "\"user\"");
        assert_eq!(
            serde_json::to_string(&Role::Assistant).unwrap(),
            "\"assistant\""
        );
        assert_eq!(serde_json::to_string(&Role::Tool).unwrap(), "\"tool\"");
        assert_eq!(serde_json::to_string(&Role::System).unwrap(), "\"system\"");
    }

    #[test]
    fn permission_serde_values() {
        assert_eq!(
            serde_json::to_string(&Permission::ReadOnly).unwrap(),
            "\"read_only\""
        );
        assert_eq!(
            serde_json::to_string(&Permission::Dangerous).unwrap(),
            "\"dangerous\""
        );
    }

    #[test]
    fn agent_event_heartbeat_is_clone() {
        let ev = AgentEvent::Heartbeat;
        let _cloned = ev.clone();
    }

    #[test]
    fn sub_agent_event_variants() {
        let started = SubAgentEvent::Started {
            prompt_preview: "research Rust".into(),
        };
        let tool = SubAgentEvent::ToolUse {
            name: "web_search".into(),
            input_preview: "Rust CLI 2025".into(),
        };
        let done = SubAgentEvent::ToolDone {
            name: "web_search".into(),
            state: ToolState::Completed,
        };
        let text = SubAgentEvent::TextDelta("partial output".into());
        let fin = SubAgentEvent::Finished { tokens: 5000 };
        let err = SubAgentEvent::Error("timeout".into());

        // All must be cloneable and debuggable
        for ev in [started, tool, done, text, fin, err] {
            let _ = format!("{:?}", ev.clone());
        }
    }

    #[test]
    fn agent_event_sub_agent_progress() {
        let ev = AgentEvent::SubAgentProgress {
            agent_id: "sa-001".into(),
            event: SubAgentEvent::ToolUse {
                name: "bash".into(),
                input_preview: "ls -la".into(),
            },
        };
        let cloned = ev.clone();
        assert!(format!("{cloned:?}").contains("sa-001"));
    }
}
