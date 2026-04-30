use crate::types::{ContentBlock, ConversationMessage, Role, TurnUsage};

const COMPACT_PREAMBLE: &str = "This session is being continued from a previous conversation that ran out of context. The summary below covers the earlier portion of the conversation.\n\n";
const COMPACT_RECENT_NOTE: &str = "Recent messages are preserved verbatim.";
const COMPACT_RESUME_INSTRUCTION: &str = "Continue the conversation from where it left off without asking the user any further questions. Resume directly — do not acknowledge the summary, do not recap what was happening, and do not preface with continuation text.";

const DEFAULT_PRESERVE_RECENT: usize = 4;

/// Hard ceiling for inline image payloads passed to `push_user_multimodal`.
/// 12 MiB is a safe upper bound across providers we currently target —
/// Anthropic accepts up to ~5 MB per image but that's per their API limit; we
/// allow some headroom for callers that resize before sending. Above this
/// threshold the image is replaced with a text placeholder. Telegram callers
/// should still honour their own (lower) `tg_media.native_image_max_bytes`
/// guard before reaching this layer; this constant is the *core* safety net.
pub const MAX_INLINE_IMAGE_BYTES: usize = 12 * 1024 * 1024;

/// Provider-agnostic *pessimistic* token estimate for an image block,
/// returned as `char_count` (downstream `estimated_tokens` divides by 4).
///
/// We don't know the final provider at history-level — the same
/// history can be re-played against Anthropic or OpenAI. We therefore
/// compute both provider costs and return the larger, so the budget
/// check is safe in either direction:
///
/// * OpenAI with `detail=low`          → 85 input tokens flat
/// * OpenAI with `detail=high`/`auto`  → 85 + 170 × tiles (tiles ≤ 16)
/// * Anthropic Claude 3.x vision       → ≈ (w × h) / 750, but since we
///   don't decode the PNG/JPEG we approximate from the base64 length:
///   1 base64 char ≈ 0.75 bytes, and assume ~3 bytes/pixel (RGBA post
///   resize), so `tokens ≈ (base64_len × 0.75) / 3 / 750` — capped at
///   the model's own 1.6k-token-per-image ceiling.
///
/// Returned in *chars* to stay consistent with the sibling text-block
/// math (text chars → tokens via the `/4` heuristic in the caller).
pub fn estimate_image_tokens_in_chars(
    base64_len: usize,
    detail: Option<crate::types::ImageDetail>,
) -> usize {
    const CHARS_PER_TOKEN: usize = 4;

    // OpenAI: detail=low is a flat 85 tokens. Tests care that we return
    // a *smaller* estimate for "low" than for "high" (otherwise the
    // detail hint is semantically ignored).
    let openai = match detail {
        Some(crate::types::ImageDetail::Low) => 85,
        _ => {
            // rough tile count: base64 of a 512×512 tile post-resize is
            // ~87000 chars. Clamp between 1 and 16 tiles to match the
            // provider's own cap.
            let tiles = (base64_len / 87_000).clamp(1, 16);
            85 + 170 * tiles
        }
    };

    // Anthropic: ≈ (base64_len × 0.75 bytes) / 3 bytes-per-pixel / 750.
    // Clamped at 1.6k tokens to mirror the hard cap Anthropic enforces.
    let anthropic_est = ((base64_len / 4) * 3) / 3 / 750;
    let anthropic = anthropic_est.clamp(85, 1_600);

    let tokens = openai.max(anthropic);
    tokens * CHARS_PER_TOKEN
}

/// Combined hard cap across all images in a single user turn (e.g. a
/// Telegram album). Even when each individual image fits under
/// [`MAX_INLINE_IMAGE_BYTES`], a 10-image album of 11 MB images would still
/// be ≥ 110 MB — far above any provider's request size budget. The
/// per-turn cap forces graceful degradation (drop trailing oversized
/// images, keep first ones intact) instead of a wall of provider 4xx.
pub const MAX_TURN_IMAGE_BYTES: usize = 48 * 1024 * 1024;

/// Built-in context window registry (tokens). Checked via substring match.
const MODEL_CONTEXT_WINDOWS: &[(&str, u32)] = &[
    // Anthropic
    ("claude-sonnet", 200_000),
    ("claude-haiku", 200_000),
    ("claude-opus", 200_000),
    // OpenAI
    ("o4-mini", 200_000),
    ("gpt-4o", 128_000),
    // MiniMax
    ("MiniMax-Text-01", 1_000_000),
    ("MiniMax-M2", 204_000),
    // Fireworks GLM
    ("glm-5p1", 202_000),
    // GLM / Z.AI native
    ("glm-5.1", 200_000),
    ("glm-5-turbo", 200_000),
    ("glm-5", 200_000),
    ("glm-4.7", 131_000),
    ("glm-4.5", 128_000),
    ("glm-4-plus", 128_000),
    // Llama
    ("llama-v3p1", 131_000),
    ("llama-3.3", 131_000),
    ("llama-3.1", 131_000),
    // Gemma
    ("gemma2", 8_000),
    // Moonshot / Kimi
    ("kimi-k2", 262_000),
    ("k2p5", 262_000),
    ("moonshot-v1-128k", 131_000),
    ("moonshot-v1-32k", 32_000),
    ("moonshot-v1-8k", 8_000),
];

/// Look up context window (tokens) by model name substring. Falls back to 128K.
pub fn model_context_window(model: &str) -> u32 {
    for &(pattern, tokens) in MODEL_CONTEXT_WINDOWS {
        if model.contains(pattern) {
            return tokens;
        }
    }
    128_000
}

#[derive(Debug, Clone)]
pub struct ConversationHistory {
    system_prompt: String,
    messages: Vec<ConversationMessage>,
    context_window_tokens: u32,
    last_input_tokens: Option<u64>,
    /// Summary from the last compaction (for iterative update).
    last_compaction_summary: Option<String>,
}

impl ConversationHistory {
    pub fn new(system_prompt: String) -> Self {
        Self {
            system_prompt,
            messages: Vec::new(),
            context_window_tokens: 128_000,
            last_input_tokens: None,
            last_compaction_summary: None,
        }
    }

    /// Get the summary from the last compaction (if any).
    pub fn last_compaction_summary(&self) -> Option<&str> {
        self.last_compaction_summary.as_deref()
    }

    /// Store a compaction summary for iterative updates.
    pub fn set_compaction_summary(&mut self, summary: String) {
        self.last_compaction_summary = Some(summary);
    }

    /// Set context limit from a token count.
    /// Uses 1:1 byte-to-token ratio (worst case for code/JSON-heavy content).
    pub fn set_context_window_tokens(&mut self, tokens: u32) {
        self.context_window_tokens = tokens;
    }

    pub fn context_window_tokens(&self) -> u32 {
        self.context_window_tokens
    }

    pub fn last_input_tokens(&self) -> Option<u64> {
        self.last_input_tokens
    }

    pub fn set_last_input_tokens(&mut self, tokens: impl Into<Option<u64>>) {
        self.last_input_tokens = tokens.into();
    }

    pub fn system_prompt(&self) -> &str {
        &self.system_prompt
    }

    pub fn messages(&self) -> &[ConversationMessage] {
        &self.messages
    }

    pub fn message_count(&self) -> usize {
        self.messages.len()
    }

    pub fn push_user(&mut self, text: &str) {
        self.messages.push(ConversationMessage::user(text));
    }

    /// Push a user message that mixes text and inline images. Used by Telegram
    /// media ingestion when the active model supports native vision (see
    /// `tg_media.native_image_context`). Empty `parts` are skipped; if every
    /// part is empty the call is a no-op.
    ///
    /// **Hard cap on image payloads.** Any single `ContentBlock::Image` whose
    /// decoded byte size exceeds [`MAX_INLINE_IMAGE_BYTES`] is replaced with a
    /// text placeholder. The Telegram bot already guards via
    /// `tg_media.native_image_max_bytes`, but `push_user_multimodal` is also a
    /// public API surface — a programmatic caller (tests, future channels,
    /// MCP-tooling) could otherwise push a 50 MB JPEG straight into history,
    /// blow the provider's request limit, balloon JSONL session files, and
    /// trigger `Image too large` 413s mid-conversation.
    pub fn push_user_multimodal(&mut self, mut blocks: Vec<ContentBlock>) {
        // Pass 1: per-block hard cap.
        for block in &mut blocks {
            if let ContentBlock::Image {
                mime, data_base64, ..
            } = block
            {
                // base64 is ~4/3 the decoded size; approximate decoded bytes.
                let decoded = data_base64.len().saturating_mul(3) / 4;
                if decoded > MAX_INLINE_IMAGE_BYTES {
                    let placeholder = format!(
                        "[image dropped: {mime} ~{decoded}B exceeds {MAX_INLINE_IMAGE_BYTES}B core hard-cap]"
                    );
                    tracing::warn!(
                        decoded_bytes = decoded,
                        cap = MAX_INLINE_IMAGE_BYTES,
                        mime = %mime,
                        "push_user_multimodal: image exceeds core hard-cap, replacing with placeholder"
                    );
                    *block = ContentBlock::Text { text: placeholder };
                }
            }
        }
        // Pass 2: combined-turn cap. Walk in order, keep early images
        // intact, drop later ones with a placeholder once we cross the
        // budget. This favours the user-visible "first photo of an album"
        // and keeps captions+metadata in the prompt.
        let mut running: usize = 0;
        let mut dropped_for_turn_cap = 0usize;
        for block in &mut blocks {
            if let ContentBlock::Image {
                mime, data_base64, ..
            } = block
            {
                let decoded = data_base64.len().saturating_mul(3) / 4;
                if running.saturating_add(decoded) > MAX_TURN_IMAGE_BYTES {
                    let placeholder = format!(
                        "[image dropped: {mime} ~{decoded}B would push turn over {MAX_TURN_IMAGE_BYTES}B combined cap]"
                    );
                    *block = ContentBlock::Text { text: placeholder };
                    dropped_for_turn_cap += 1;
                } else {
                    running += decoded;
                }
            }
        }
        if dropped_for_turn_cap > 0 {
            tracing::warn!(
                dropped = dropped_for_turn_cap,
                running_bytes = running,
                cap = MAX_TURN_IMAGE_BYTES,
                "push_user_multimodal: dropped trailing images to keep turn under combined cap"
            );
        }
        let any_real = blocks.iter().any(|b| match b {
            ContentBlock::Text { text } => !text.is_empty(),
            ContentBlock::Image { data_base64, .. } => !data_base64.is_empty(),
            _ => false,
        });
        if !any_real {
            return;
        }
        self.messages.push(ConversationMessage {
            role: crate::types::Role::User,
            blocks,
            timestamp: chrono::Utc::now(),
            usage: None,
        });
    }

    pub fn push_assistant(&mut self, blocks: Vec<ContentBlock>, usage: Option<TurnUsage>) {
        self.messages
            .push(ConversationMessage::assistant(blocks, usage));
    }

    pub fn push_tool_result(&mut self, call_id: &str, output: &str, is_error: bool) {
        const MAX_TOOL_OUTPUT: usize = 8000;
        let trimmed = if output.len() > MAX_TOOL_OUTPUT {
            let safe_end = output.floor_char_boundary(MAX_TOOL_OUTPUT);
            let cut = output.len() - safe_end;
            format!("{}...\n[truncated {cut} bytes]", &output[..safe_end])
        } else {
            output.to_string()
        };
        self.messages.push(ConversationMessage::tool_result(
            call_id, &trimmed, is_error,
        ));
    }

    pub fn push_raw(&mut self, msg: ConversationMessage) {
        self.messages.push(msg);
    }

    /// Append extra context to the system prompt (e.g. per-session instructions).
    pub fn inject_system_context(&mut self, extra: &str) {
        self.system_prompt.push_str(extra);
    }

    /// Restore system prompt to its original value (undo ephemeral inject_system_context).
    pub fn restore_system_prompt(&mut self, prompt: String) {
        self.system_prompt = prompt;
    }

    /// Convert messages to the JSON format expected by Anthropic API.
    pub fn to_api_messages(&self) -> Vec<serde_json::Value> {
        let mut api_msgs = Vec::new();

        for msg in &self.messages {
            match msg.role {
                Role::User => {
                    let content: Vec<serde_json::Value> = msg
                        .blocks
                        .iter()
                        .filter_map(|b| match b {
                            ContentBlock::Text { text } => {
                                // Defence in depth (two layers):
                                //   1. If the entire block is just a sentinel
                                //      (intern failed for an externalised
                                //      image), drop it — the model has no
                                //      use for an empty marker.
                                //   2. Otherwise strip any embedded sentinel
                                //      segments mid-text (e.g. user pasted
                                //      a marker by accident, or an old
                                //      tool injected one). Bumps
                                //      `SENTINEL_LEAK_COUNT` per strip.
                                if text.starts_with(crate::types::IMAGE_REF_SENTINEL_PREFIX)
                                    && !text[crate::types::IMAGE_REF_SENTINEL_PREFIX.len()..]
                                        .contains(' ')
                                {
                                    crate::types::SENTINEL_LEAK_COUNT
                                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    return None;
                                }
                                let cleaned = crate::types::strip_image_ref_sentinel(text);
                                Some(serde_json::json!({"type": "text", "text": cleaned}))
                            }
                            ContentBlock::Image {
                                mime,
                                data_base64,
                                detail,
                            } => {
                                // Anthropic-style canonical form. The OpenAI-compat
                                // bridge reads `detail_hint` (mirrored at the top of
                                // the block, outside `source`) and translates it to
                                // `image_url.detail = "low"|"high"|"auto"`. Anthropic
                                // and other native vision providers simply ignore
                                // unknown top-level keys.
                                let mut v = serde_json::json!({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": mime,
                                        "data": data_base64,
                                    }
                                });
                                if let Some(d) = detail {
                                    v["detail_hint"] =
                                        serde_json::Value::String(d.as_str().to_string());
                                }
                                Some(v)
                            }
                            _ => None,
                        })
                        .collect();
                    api_msgs.push(serde_json::json!({
                        "role": "user",
                        "content": content,
                    }));
                }
                Role::Assistant => {
                    let content: Vec<serde_json::Value> = msg
                        .blocks
                        .iter()
                        .filter_map(|b| match b {
                            ContentBlock::Text { text } => {
                                let cleaned = crate::types::strip_image_ref_sentinel(text);
                                Some(serde_json::json!({"type": "text", "text": cleaned}))
                            }
                            ContentBlock::Thinking { text } => {
                                let cleaned = crate::types::strip_image_ref_sentinel(text);
                                Some(serde_json::json!({"type": "thinking", "thinking": cleaned}))
                            }
                            ContentBlock::ToolUse { id, name, input } => Some(serde_json::json!({
                                "type": "tool_use",
                                "id": id,
                                "name": name,
                                "input": input,
                            })),
                            _ => None,
                        })
                        .collect();
                    api_msgs.push(serde_json::json!({
                        "role": "assistant",
                        "content": content,
                    }));
                }
                Role::Tool => {
                    let content: Vec<serde_json::Value> = msg
                        .blocks
                        .iter()
                        .filter_map(|b| match b {
                            ContentBlock::ToolResult {
                                call_id,
                                output,
                                is_error,
                            } => {
                                let cleaned = crate::types::strip_image_ref_sentinel(output);
                                Some(serde_json::json!({
                                    "type": "tool_result",
                                    "tool_use_id": call_id,
                                    "content": cleaned,
                                    "is_error": is_error,
                                }))
                            }
                            _ => None,
                        })
                        .collect();
                    api_msgs.push(serde_json::json!({
                        "role": "user",
                        "content": content,
                    }));
                }
                Role::System => {
                    let content: Vec<serde_json::Value> = msg
                        .blocks
                        .iter()
                        .filter_map(|b| match b {
                            ContentBlock::Text { text } => {
                                let cleaned = crate::types::strip_image_ref_sentinel(text);
                                Some(serde_json::json!({"type": "text", "text": cleaned}))
                            }
                            _ => None,
                        })
                        .collect();
                    if !content.is_empty() {
                        api_msgs.push(serde_json::json!({
                            "role": "user",
                            "content": content,
                        }));
                    }
                }
            }
        }

        api_msgs
    }

    /// Rough token estimate using len/4 heuristic (matching claude-code).
    ///
    /// Image cost model (per-provider):
    ///
    /// * OpenAI GPT-4o/4.1 with `detail=low`  → flat 85 tokens.
    /// * OpenAI with `detail=high` or `auto`  → 85 + 170 × tiles (we
    ///   approximate "tiles" from the base64 length: base64 chars /
    ///   87000 ≈ 512×512 tile count after resize; clamps 1..=16).
    /// * Anthropic Claude 3.x vision          → ≈ (w × h) / 750 tokens
    ///   (also approximated from base64 size because we don't decode).
    ///
    /// Since `estimated_tokens` is not provider-aware we return the
    /// *pessimistic* max across these, which matches how this estimate
    /// is used (input-token budget check before a request). The value is
    /// still scaled through the `len/4` divisor below.
    pub fn estimated_tokens(&self) -> usize {
        let chars: usize = self.system_prompt.len()
            + self
                .messages
                .iter()
                .map(|m| {
                    m.blocks
                        .iter()
                        .map(|b| match b {
                            ContentBlock::Text { text } | ContentBlock::Thinking { text } => {
                                text.len()
                            }
                            ContentBlock::ToolUse { input, .. } => input.to_string().len(),
                            ContentBlock::ToolResult { output, .. } => output.len(),
                            ContentBlock::Image {
                                data_base64,
                                detail,
                                ..
                            } => estimate_image_tokens_in_chars(data_base64.len(), *detail),
                        })
                        .sum::<usize>()
                })
                .sum::<usize>();
        chars / 4 + 1
    }

    /// Backward compat — returns estimated_tokens (NOT chars).
    pub fn estimated_chars(&self) -> usize {
        self.estimated_tokens()
    }

    pub fn needs_compaction(&self) -> bool {
        let est = self.estimated_tokens();
        let limit = self.context_window_tokens as usize;

        // Heuristic trigger: estimated tokens > 80% of context window
        if est > limit * 4 / 5 {
            return true;
        }
        // API-reported trigger: actual input_tokens > 90% of context window
        // AND our estimate is above 30% (to avoid false positives from tool schema overhead)
        if let Some(input_tokens) = self.last_input_tokens
            && input_tokens > (self.context_window_tokens as u64) * 9 / 10
            && est > limit * 3 / 10
        {
            return true;
        }
        false
    }

    pub fn fork(&self) -> Self {
        self.clone()
    }

    /// Compact history (claude-code style).
    ///
    /// Builds a structured `<summary>` from older messages, preserves the most
    /// recent `keep_recent` messages verbatim, and injects a System-role
    /// continuation message with a "resume directly" instruction.
    ///
    /// On re-compaction, the previous summary is merged (Previous / Newly sections).
    pub fn compact(&mut self, keep_recent: usize) {
        let keep = keep_recent.max(DEFAULT_PRESERVE_RECENT);
        if self.messages.len() <= keep {
            return;
        }

        // Detect existing compaction prefix (system message with preamble)
        let existing_summary = self.messages.first().and_then(extract_existing_summary);
        let prefix_len = usize::from(existing_summary.is_some());

        let raw_keep_from = self.messages.len().saturating_sub(keep);
        // Snap cut point to a turn boundary: never split between
        // assistant(tool_call) and its tool_result, or between a user
        // message and its assistant response.
        let keep_from = Self::snap_to_turn_boundary(&self.messages, raw_keep_from, prefix_len);
        let removed = &self.messages[prefix_len..keep_from];
        if removed.is_empty() {
            return;
        }
        let preserved = self.messages[keep_from..].to_vec();

        let new_summary = summarize_messages(removed);
        let merged = merge_summaries(existing_summary.as_deref(), &new_summary);
        let continuation = build_continuation_message(&merged, !preserved.is_empty());

        let mut new_messages = vec![ConversationMessage::system(&continuation)];
        new_messages.extend(preserved);
        self.messages = new_messages;

        self.truncate_tool_results();
    }

    /// Snap a cut point forward to a safe turn boundary.
    /// Never cut between assistant(tool_use) and its tool_result,
    /// or in the middle of a user→assistant exchange.
    fn snap_to_turn_boundary(
        messages: &[ConversationMessage],
        mut idx: usize,
        min: usize,
    ) -> usize {
        // If we're sitting on a Tool message, walk forward until we
        // hit a User message (start of next turn).
        while idx < messages.len() {
            match messages[idx].role {
                Role::User => break,    // clean boundary
                Role::System => break,  // clean boundary
                Role::Tool => idx += 1, // skip orphaned tool result
                Role::Assistant => {
                    // Check if this assistant has tool_calls — if so,
                    // its tool_results follow. Walk past them.
                    let has_tool_calls = messages[idx]
                        .blocks
                        .iter()
                        .any(|b| matches!(b, ContentBlock::ToolUse { .. }));
                    if has_tool_calls {
                        idx += 1;
                    } else {
                        break; // plain assistant message = ok to cut here
                    }
                }
            }
        }
        idx.max(min)
    }

    /// Truncate tool results in kept messages to save space.
    fn truncate_tool_results(&mut self) {
        const MAX_TOOL_RESULT: usize = 2000;
        for msg in &mut self.messages {
            for block in &mut msg.blocks {
                if let ContentBlock::ToolResult { output, .. } = block
                    && output.len() > MAX_TOOL_RESULT
                {
                    let safe_end = output.floor_char_boundary(MAX_TOOL_RESULT);
                    let truncated_bytes = output.len() - safe_end;
                    *output = format!(
                        "{}...\n[truncated {truncated_bytes} bytes]",
                        &output[..safe_end]
                    );
                }
            }
        }
    }

    /// Compact using an LLM-generated summary. Keeps `keep_recent` recent messages.
    pub fn compact_with_llm_summary(&mut self, summary: &str, keep_recent: usize) {
        let keep = keep_recent.max(DEFAULT_PRESERVE_RECENT);
        if self.messages.len() <= keep {
            return;
        }

        let keep_from = self.messages.len().saturating_sub(keep);
        let preserved = self.messages[keep_from..].to_vec();

        let continuation = build_continuation_message(summary, !preserved.is_empty());
        let mut new_messages = vec![ConversationMessage::system(&continuation)];
        new_messages.extend(preserved);
        self.messages = new_messages;
        self.truncate_tool_results();
    }

    /// Prepare messages for LLM summarization — returns text of messages to be compacted.
    /// Extract file paths mentioned in tool calls from messages being compacted.
    pub fn files_in_compaction_range(&self, keep_recent: usize) -> (Vec<String>, Vec<String>) {
        let keep = keep_recent.max(DEFAULT_PRESERVE_RECENT);
        if self.messages.len() <= keep {
            return (vec![], vec![]);
        }
        let keep_from = self.messages.len().saturating_sub(keep);
        let prefix_len = self
            .messages
            .first()
            .and_then(extract_existing_summary)
            .map(|_| 1)
            .unwrap_or(0);
        let removed = &self.messages[prefix_len..keep_from];

        let mut read_files = std::collections::BTreeSet::new();
        let mut modified_files = std::collections::BTreeSet::new();

        for msg in removed {
            for block in &msg.blocks {
                if let ContentBlock::ToolUse { name, input, .. } = block {
                    match name.as_str() {
                        "read" => {
                            if let Some(p) = input.get("path").and_then(|v| v.as_str()) {
                                read_files.insert(p.to_string());
                            }
                        }
                        "edit" | "write" => {
                            if let Some(p) = input.get("path").and_then(|v| v.as_str()) {
                                modified_files.insert(p.to_string());
                            }
                        }
                        "bash" => {
                            // Crude: look for common file-touching patterns in command
                            if let Some(cmd) = input.get("command").and_then(|v| v.as_str()) {
                                if cmd.contains("cat ")
                                    || cmd.contains("head ")
                                    || cmd.contains("grep ")
                                {
                                    // read-like, skip — too noisy
                                } else if cmd.contains("sed -i")
                                    || cmd.contains(">> ")
                                    || cmd.contains("> ")
                                {
                                    // modify-like — extract is unreliable, skip
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        (
            read_files.into_iter().collect(),
            modified_files.into_iter().collect(),
        )
    }

    pub fn messages_for_compaction(&self, keep_recent: usize) -> Option<String> {
        let keep = keep_recent.max(DEFAULT_PRESERVE_RECENT);
        if self.messages.len() <= keep {
            return None;
        }

        let existing_summary = self.messages.first().and_then(extract_existing_summary);
        let prefix_len = usize::from(existing_summary.is_some());
        let keep_from = self.messages.len().saturating_sub(keep);
        let removed = &self.messages[prefix_len..keep_from];
        if removed.is_empty() {
            return None;
        }

        let mut text = String::new();
        if let Some(prev) = existing_summary {
            text.push_str("[Previous summary]\n");
            text.push_str(&prev);
            text.push_str("\n\n");
        }
        text.push_str("[Messages to summarize]\n");
        for msg in removed {
            let role = match msg.role {
                Role::System => "system",
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::Tool => "tool",
            };
            let content = msg.text_content();
            if !content.is_empty() {
                text.push_str(&format!("{role}: {}\n", safe_truncate(&content, 500)));
            }
            for block in &msg.blocks {
                match block {
                    ContentBlock::ToolUse { name, .. } => {
                        text.push_str(&format!("{role}: [tool_use: {name}]\n"));
                    }
                    ContentBlock::ToolResult {
                        output, is_error, ..
                    } => {
                        let prefix = if *is_error { "error: " } else { "" };
                        text.push_str(&format!("tool: {prefix}{}\n", safe_truncate(output, 200)));
                    }
                    ContentBlock::Image {
                        mime, data_base64, ..
                    } => {
                        // Include image presence in the LLM-summary input. We
                        // can't ship the bytes (would blow the summarizer's
                        // context); but the model needs to know the user
                        // *attached an image*, otherwise the summary loses
                        // visual context entirely. Approximate decoded size.
                        let bytes = data_base64.len() * 3 / 4;
                        text.push_str(&format!("{role}: [image {mime} ~{bytes}B]\n"));
                    }
                    _ => {}
                }
            }
        }
        Some(text)
    }

    /// Auto-compact if needed. Returns `Some((before, after))` message counts if compaction ran.
    pub fn auto_compact(&mut self) -> Option<(usize, usize)> {
        if !self.needs_compaction() {
            return None;
        }
        let before = self.messages.len();

        for round in 0..5 {
            if !self.needs_compaction() {
                break;
            }
            let keep = if round == 0 {
                DEFAULT_PRESERVE_RECENT
            } else {
                (self.messages.len() / 2).clamp(2, DEFAULT_PRESERVE_RECENT)
            };
            tracing::info!(
                round,
                keep,
                msgs = self.messages.len(),
                est_tokens = self.estimated_tokens(),
                context_window = self.context_window_tokens,
                "auto_compact round"
            );
            self.compact(keep);
        }

        self.last_input_tokens = None;
        Some((before, self.messages.len()))
    }
}

// ── Compaction helpers (claude-code style) ─────────────────────────────────

/// Build the structured `<summary>` from a slice of removed messages.
fn summarize_messages(messages: &[ConversationMessage]) -> String {
    let user_count = messages.iter().filter(|m| m.role == Role::User).count();
    let asst_count = messages
        .iter()
        .filter(|m| m.role == Role::Assistant)
        .count();
    let tool_count = messages.iter().filter(|m| m.role == Role::Tool).count();
    // Image blocks carry no textual content so they vanish invisibly when
    // the summary drops older turns. Count them so the continuation
    // message at least *tells* the model: "there were 3 images earlier in
    // this conversation". Without this hint the assistant cannot reference
    // past vision context at all — a silent UX regression.
    let image_count: usize = messages
        .iter()
        .flat_map(|m| m.blocks.iter())
        .filter(|b| matches!(b, ContentBlock::Image { .. }))
        .count();

    let mut tool_names: Vec<&str> = messages
        .iter()
        .flat_map(|m| m.blocks.iter())
        .filter_map(|b| match b {
            ContentBlock::ToolUse { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect();
    tool_names.sort_unstable();
    tool_names.dedup();

    let mut lines = vec![
        "<summary>".to_string(),
        "Conversation summary:".to_string(),
        format!(
            "- Scope: {} earlier messages compacted (user={user_count}, assistant={asst_count}, tool={tool_count}).",
            messages.len()
        ),
    ];

    if !tool_names.is_empty() {
        lines.push(format!("- Tools mentioned: {}.", tool_names.join(", ")));
    }

    if image_count > 0 {
        lines.push(format!(
            "- Images in compacted turns: {image_count} (pixels no longer available; \
             ask the user to re-send if you need to see them)."
        ));
    }

    // Recent user requests (last 3)
    let recent_user: Vec<String> = messages
        .iter()
        .filter(|m| m.role == Role::User)
        .rev()
        .filter_map(|m| {
            let t = m.text_content();
            if t.is_empty() {
                None
            } else {
                Some(safe_truncate(&t, 160))
            }
        })
        .take(3)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    if !recent_user.is_empty() {
        lines.push("- Recent user requests:".to_string());
        for req in &recent_user {
            lines.push(format!("  - {req}"));
        }
    }

    // Pending work (messages containing todo/next/pending/remaining)
    let pending: Vec<String> = messages
        .iter()
        .rev()
        .filter_map(|m| {
            let t = m.text_content();
            let low = t.to_ascii_lowercase();
            if low.contains("todo")
                || low.contains("next")
                || low.contains("pending")
                || low.contains("remaining")
            {
                Some(safe_truncate(&t, 160))
            } else {
                None
            }
        })
        .take(3)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    if !pending.is_empty() {
        lines.push("- Pending work:".to_string());
        for item in &pending {
            lines.push(format!("  - {item}"));
        }
    }

    // Key files (paths with known extensions)
    let key_files = collect_key_files(messages);
    if !key_files.is_empty() {
        lines.push(format!("- Key files referenced: {}.", key_files.join(", ")));
    }

    // Current work (last non-empty text from any role)
    if let Some(current) = messages
        .iter()
        .rev()
        .filter_map(|m| {
            let t = m.text_content();
            if t.trim().is_empty() {
                None
            } else {
                Some(safe_truncate(&t, 200))
            }
        })
        .next()
    {
        lines.push(format!("- Current work: {current}"));
    }

    // Key timeline
    lines.push("- Key timeline:".to_string());
    for msg in messages {
        let role = match msg.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        };
        let content: Vec<String> = msg.blocks.iter().map(summarize_block).collect();
        lines.push(format!("  - {role}: {}", content.join(" | ")));
    }
    lines.push("</summary>".to_string());
    lines.join("\n")
}

fn summarize_block(block: &ContentBlock) -> String {
    let raw = match block {
        ContentBlock::Text { text } | ContentBlock::Thinking { text } => text.clone(),
        ContentBlock::ToolUse { name, input, .. } => format!("tool_use {name}({input})"),
        ContentBlock::ToolResult {
            call_id: _,
            output,
            is_error,
        } => {
            let prefix = if *is_error { "error " } else { "" };
            format!("tool_result: {prefix}{output}")
        }
        ContentBlock::Image {
            mime, data_base64, ..
        } => {
            let bytes = data_base64.len() * 3 / 4;
            format!("[image {mime} ~{bytes}B]")
        }
    };
    safe_truncate(&raw, 160)
}

/// Extract file-like paths from message content.
fn collect_key_files(messages: &[ConversationMessage]) -> Vec<String> {
    let mut files: Vec<String> = messages
        .iter()
        .flat_map(|m| m.blocks.iter())
        .flat_map(|b| {
            let text = match b {
                ContentBlock::Text { text } | ContentBlock::Thinking { text } => text.as_str(),
                ContentBlock::ToolUse { input, .. } => {
                    return extract_file_candidates(&input.to_string());
                }
                ContentBlock::ToolResult { output, .. } => output.as_str(),
                ContentBlock::Image { .. } => return Vec::new(),
            };
            extract_file_candidates(text)
        })
        .collect();
    files.sort();
    files.dedup();
    files.into_iter().take(8).collect()
}

fn extract_file_candidates(content: &str) -> Vec<String> {
    const EXTENSIONS: &[&str] = &["rs", "ts", "tsx", "js", "json", "md", "toml", "py", "sh"];
    content
        .split_whitespace()
        .filter_map(|token| {
            let candidate = token.trim_matches(|c: char| {
                matches!(c, ',' | '.' | ':' | ';' | ')' | '(' | '"' | '\'' | '`')
            });
            if candidate.contains('/') {
                let ext = std::path::Path::new(candidate)
                    .extension()
                    .and_then(|e| e.to_str());
                if ext.is_some_and(|e| EXTENSIONS.iter().any(|x| e.eq_ignore_ascii_case(x))) {
                    return Some(candidate.to_string());
                }
            }
            None
        })
        .collect()
}

/// Merge an existing summary with a new one for re-compaction.
fn merge_summaries(existing: Option<&str>, new_summary: &str) -> String {
    let Some(existing) = existing else {
        return new_summary.to_string();
    };

    let prev_highlights = extract_highlights(existing);
    let new_formatted = format_summary(new_summary);
    let new_highlights = extract_highlights(&new_formatted);
    let new_timeline = extract_timeline(&new_formatted);

    let mut lines = vec!["<summary>".to_string(), "Conversation summary:".to_string()];

    if !prev_highlights.is_empty() {
        lines.push("- Previously compacted context:".to_string());
        for h in &prev_highlights {
            lines.push(format!("  {h}"));
        }
    }
    if !new_highlights.is_empty() {
        lines.push("- Newly compacted context:".to_string());
        for h in &new_highlights {
            lines.push(format!("  {h}"));
        }
    }
    if !new_timeline.is_empty() {
        lines.push("- Key timeline:".to_string());
        for t in &new_timeline {
            lines.push(format!("  {t}"));
        }
    }

    lines.push("</summary>".to_string());
    lines.join("\n")
}

/// Format a raw `<summary>` block into user-facing text.
fn format_summary(summary: &str) -> String {
    if let (Some(start), Some(end)) = (summary.find("<summary>"), summary.find("</summary>")) {
        let inner = &summary[start + 9..end];
        format!("Summary:\n{}", inner.trim())
    } else {
        summary.to_string()
    }
}

/// Build the continuation System message injected after compaction.
fn build_continuation_message(summary: &str, recent_preserved: bool) -> String {
    let mut text = format!("{COMPACT_PREAMBLE}{}", format_summary(summary));
    if recent_preserved {
        text.push_str("\n\n");
        text.push_str(COMPACT_RECENT_NOTE);
    }
    text.push('\n');
    text.push_str(COMPACT_RESUME_INSTRUCTION);
    text
}

/// Extract an existing summary from a System message (for re-compaction).
fn extract_existing_summary(message: &ConversationMessage) -> Option<String> {
    if message.role != Role::System {
        return None;
    }
    let text = message.text_content();
    let summary = text.strip_prefix(COMPACT_PREAMBLE)?;
    let summary = summary
        .split_once(&format!("\n\n{COMPACT_RECENT_NOTE}"))
        .map_or(summary, |(v, _)| v);
    let summary = summary
        .split_once(&format!("\n{COMPACT_RESUME_INSTRUCTION}"))
        .map_or(summary, |(v, _)| v);
    Some(summary.trim().to_string())
}

/// Extract highlight lines (everything except Key timeline) from formatted summary.
fn extract_highlights(summary: &str) -> Vec<String> {
    let mut lines = Vec::new();
    let mut in_timeline = false;
    for line in format_summary(summary).lines() {
        let trimmed = line.trim_end();
        if trimmed.is_empty() || trimmed == "Summary:" || trimmed == "Conversation summary:" {
            continue;
        }
        if trimmed == "- Key timeline:" {
            in_timeline = true;
            continue;
        }
        if in_timeline {
            continue;
        }
        lines.push(trimmed.to_string());
    }
    lines
}

/// Extract timeline lines from formatted summary.
fn extract_timeline(summary: &str) -> Vec<String> {
    let mut lines = Vec::new();
    let mut in_timeline = false;
    for line in format_summary(summary).lines() {
        let trimmed = line.trim_end();
        if trimmed == "- Key timeline:" {
            in_timeline = true;
            continue;
        }
        if !in_timeline {
            continue;
        }
        if trimmed.is_empty() {
            break;
        }
        lines.push(trimmed.to_string());
    }
    lines
}

/// Truncate a string at a char boundary, appending "..." if cut.
fn safe_truncate(s: &str, max_chars: usize) -> String {
    let char_count = s.chars().count();
    if char_count <= max_chars {
        return s.to_string();
    }
    let byte_end = s
        .char_indices()
        .nth(max_chars)
        .map(|(pos, _)| pos)
        .unwrap_or(s.len());
    format!("{}...", &s[..byte_end])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_truncate_ascii() {
        assert_eq!(safe_truncate("hello world", 5), "hello...");
        assert_eq!(safe_truncate("hi", 10), "hi");
    }

    #[test]
    fn safe_truncate_multibyte() {
        let cyrillic = "Привет мир";
        let truncated = safe_truncate(cyrillic, 6);
        assert!(truncated.ends_with("..."));
        assert!(!truncated.contains('\u{FFFD}'));
    }

    #[test]
    fn safe_truncate_emoji() {
        let emoji = "Hello 🌍🌎🌏 world";
        let truncated = safe_truncate(emoji, 8);
        assert!(truncated.ends_with("..."));
    }

    #[test]
    fn new_history_is_empty() {
        let h = ConversationHistory::new("test prompt".into());
        assert_eq!(h.message_count(), 0);
        assert_eq!(h.system_prompt(), "test prompt");
    }

    #[test]
    fn push_user_adds_message() {
        let mut h = ConversationHistory::new(String::new());
        h.push_user("hello");
        assert_eq!(h.message_count(), 1);
        assert_eq!(h.messages()[0].text_content(), "hello");
    }

    #[test]
    fn to_api_messages_formats_correctly() {
        let mut h = ConversationHistory::new(String::new());
        h.push_user("hi");
        h.push_assistant(
            vec![ContentBlock::Text {
                text: "hello!".into(),
            }],
            None,
        );
        let msgs = h.to_api_messages();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[1]["role"], "assistant");
    }

    #[test]
    fn push_user_multimodal_serialises_anthropic_image_block() {
        let mut h = ConversationHistory::new(String::new());
        h.push_user_multimodal(vec![
            ContentBlock::Text {
                text: "what is shown?".into(),
            },
            ContentBlock::Image {
                mime: "image/png".into(),
                data_base64: "iVBORw0KGgo".into(),
                detail: None,
            },
        ]);
        let msgs = h.to_api_messages();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
        let parts = msgs[0]["content"].as_array().unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[0]["text"], "what is shown?");
        assert_eq!(parts[1]["type"], "image");
        assert_eq!(parts[1]["source"]["type"], "base64");
        assert_eq!(parts[1]["source"]["media_type"], "image/png");
        assert_eq!(parts[1]["source"]["data"], "iVBORw0KGgo");
    }

    #[test]
    fn push_user_multimodal_skips_when_all_blocks_empty() {
        let mut h = ConversationHistory::new(String::new());
        h.push_user_multimodal(vec![
            ContentBlock::Text {
                text: String::new(),
            },
            ContentBlock::Image {
                mime: "image/png".into(),
                data_base64: String::new(),
                detail: None,
            },
        ]);
        assert_eq!(h.message_count(), 0);
    }

    #[test]
    fn estimated_tokens_counts_image_block() {
        // Previously the estimator used a flat 6000-char cost per image;
        // now it's provider-aware (85 tok min for detail=low, tile math
        // otherwise). Even the smallest image must still add more than
        // the 85-token floor on top of the empty baseline.
        let mut h = ConversationHistory::new(String::new());
        let baseline = h.estimated_tokens();
        h.push_user_multimodal(vec![ContentBlock::Image {
            mime: "image/png".into(),
            data_base64: "AAAA".into(),
            detail: None,
        }]);
        let after = h.estimated_tokens();
        assert!(
            after > baseline + 80,
            "image block must add at least the 85-token floor (got {after} vs {baseline})"
        );
    }

    #[test]
    fn estimated_tokens_image_scales_with_size() {
        // Larger base64 payload should cost proportionally more tokens,
        // mirroring OpenAI tile math. Use sizes that cross the tile
        // boundary (~87k chars/tile).
        let mut h_small = ConversationHistory::new(String::new());
        h_small.push_user_multimodal(vec![ContentBlock::Image {
            mime: "image/png".into(),
            data_base64: "A".repeat(10_000),
            detail: None,
        }]);
        let mut h_big = ConversationHistory::new(String::new());
        h_big.push_user_multimodal(vec![ContentBlock::Image {
            mime: "image/png".into(),
            data_base64: "A".repeat(900_000),
            detail: None,
        }]);
        assert!(
            h_big.estimated_tokens() > h_small.estimated_tokens(),
            "big image ({}) must cost more than small ({})",
            h_big.estimated_tokens(),
            h_small.estimated_tokens()
        );
    }

    #[test]
    fn push_user_multimodal_drops_oversized_image() {
        let mut h = ConversationHistory::new(String::new());
        // Encode 13 MiB of zero bytes — base64 of that is ~17.3 MiB; decoded
        // size measured by the guard exceeds the 12 MiB hard cap.
        let bytes = vec![0u8; 13 * 1024 * 1024];
        let b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &bytes);
        h.push_user_multimodal(vec![
            ContentBlock::Image {
                mime: "image/png".into(),
                data_base64: b64,
                detail: None,
            },
            ContentBlock::Text {
                text: "tell me about it".into(),
            },
        ]);
        assert_eq!(h.message_count(), 1);
        let last = h.messages.last().unwrap();
        assert!(
            !last
                .blocks
                .iter()
                .any(|b| matches!(b, ContentBlock::Image { .. })),
            "oversized image must be stripped"
        );
        let txt = last.text_content();
        assert!(txt.contains("[image dropped"), "got: {txt}");
        assert!(txt.contains("tell me about it"), "kept text must remain");
    }

    #[test]
    fn push_user_multimodal_drops_album_past_combined_cap() {
        // 6 images × 10 MiB each = 60 MiB combined → exceeds the 48 MiB
        // turn-cap. Each image individually is under MAX_INLINE_IMAGE_BYTES
        // (12 MiB), so the per-block guard alone is not enough.
        let mut h = ConversationHistory::new(String::new());
        let bytes = vec![0u8; 10 * 1024 * 1024];
        let b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &bytes);
        let mut blocks: Vec<ContentBlock> = (0..6)
            .map(|_| ContentBlock::Image {
                mime: "image/png".into(),
                data_base64: b64.clone(),
                detail: None,
            })
            .collect();
        blocks.push(ContentBlock::Text {
            text: "describe these".into(),
        });
        h.push_user_multimodal(blocks);
        assert_eq!(h.message_count(), 1);
        let last = h.messages.last().unwrap();
        let kept_imgs = last
            .blocks
            .iter()
            .filter(|b| matches!(b, ContentBlock::Image { .. }))
            .count();
        let dropped_placeholders = last
            .blocks
            .iter()
            .filter(|b| match b {
                ContentBlock::Text { text } => text.contains("combined cap"),
                _ => false,
            })
            .count();
        assert!(
            (1..6).contains(&kept_imgs),
            "must keep some images and drop some (kept={kept_imgs})"
        );
        assert!(
            dropped_placeholders >= 1,
            "must mark dropped images with a placeholder"
        );
    }

    #[test]
    fn push_user_multimodal_accepts_exact_turn_cap() {
        // The guard is `running + decoded > MAX_TURN_IMAGE_BYTES` (strict),
        // so hitting the cap exactly must pass. We use 4 images of exactly
        // MAX_INLINE_IMAGE_BYTES = 12 MiB each = 48 MiB combined = the
        // turn cap. 12 MiB is divisible by 3 so base64 round-trips
        // cleanly via the `len*3/4` estimator.
        let mut h = ConversationHistory::new(String::new());
        let per_image = MAX_INLINE_IMAGE_BYTES; // 12 MiB, divisible by 3
        assert_eq!(per_image % 3, 0, "per_image must divide cleanly by 3");
        let bytes = vec![0u8; per_image];
        let b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &bytes);
        // `push_user_multimodal` uses `len*3/4` to estimate decoded size;
        // for 3-divisible inputs this equals N exactly.
        assert_eq!(b64.len().saturating_mul(3) / 4, per_image);
        let n_images = MAX_TURN_IMAGE_BYTES / per_image; // = 4
        assert_eq!(n_images * per_image, MAX_TURN_IMAGE_BYTES);

        let blocks: Vec<ContentBlock> = (0..n_images)
            .map(|_| ContentBlock::Image {
                mime: "image/png".into(),
                data_base64: b64.clone(),
                detail: None,
            })
            .collect();
        h.push_user_multimodal(blocks);

        let last = h.messages.last().unwrap();
        let kept = last
            .blocks
            .iter()
            .filter(|b| matches!(b, ContentBlock::Image { .. }))
            .count();
        let dropped_placeholders = last
            .blocks
            .iter()
            .filter(|b| match b {
                ContentBlock::Text { text } => text.contains("combined cap"),
                _ => false,
            })
            .count();
        assert_eq!(
            kept, n_images,
            "all images must survive at exactly the turn cap"
        );
        assert_eq!(
            dropped_placeholders, 0,
            "no drops expected at exactly the cap"
        );
    }

    #[test]
    fn push_user_multimodal_drops_one_byte_past_cap() {
        // 4 × 12 MiB = cap exactly, plus one tiny image must tip the running
        // total over and be replaced with a placeholder. The tiny image
        // must survive the per-block guard (trivially true for 3 bytes).
        let mut h = ConversationHistory::new(String::new());
        let per_image = MAX_INLINE_IMAGE_BYTES;
        let big_bytes = vec![0u8; per_image];
        let big_b64 = base64::Engine::encode(
            &base64::engine::general_purpose::STANDARD,
            big_bytes.as_slice(),
        );
        let tiny_b64 = base64::Engine::encode(
            &base64::engine::general_purpose::STANDARD,
            [0u8; 3].as_ref(),
        );

        let mut blocks: Vec<ContentBlock> = (0..4)
            .map(|_| ContentBlock::Image {
                mime: "image/png".into(),
                data_base64: big_b64.clone(),
                detail: None,
            })
            .collect();
        blocks.push(ContentBlock::Image {
            mime: "image/png".into(),
            data_base64: tiny_b64,
            detail: None,
        });
        h.push_user_multimodal(blocks);

        let last = h.messages.last().unwrap();
        let kept = last
            .blocks
            .iter()
            .filter(|b| matches!(b, ContentBlock::Image { .. }))
            .count();
        let dropped_placeholders = last
            .blocks
            .iter()
            .filter(|b| match b {
                ContentBlock::Text { text } => text.contains("combined cap"),
                _ => false,
            })
            .count();
        assert_eq!(
            kept, 4,
            "the first 4 big images must survive; only the extra tiny one drops"
        );
        assert_eq!(
            dropped_placeholders, 1,
            "the 5th image must be replaced with a placeholder"
        );
    }

    #[test]
    fn to_api_messages_strips_image_ref_sentinel_text_blocks() {
        // Regression guard: if a JSONL session still has a sentinel marker in
        // a Text block (artifact missing on disk, intern failed, …) the LLM
        // request must NOT carry the marker — that would leak internal JSON
        // to the model. The block is dropped silently.
        let mut h = ConversationHistory::new(String::new());
        h.push_user_multimodal(vec![
            ContentBlock::Text {
                text: format!(
                    "{}{{\"mime\":\"image/png\",\"path\":\"img_dead.png\"}}",
                    crate::types::IMAGE_REF_SENTINEL_PREFIX
                ),
            },
            ContentBlock::Text {
                text: "real user text".into(),
            },
        ]);
        let msgs = h.to_api_messages();
        assert_eq!(msgs.len(), 1);
        let parts = msgs[0]["content"].as_array().unwrap();
        assert_eq!(parts.len(), 1, "sentinel text block must be dropped");
        assert_eq!(parts[0]["text"], "real user text");
        // And specifically: nothing in the serialized request mentions the marker.
        let raw = serde_json::to_string(&msgs[0]).unwrap();
        assert!(
            !raw.contains(crate::types::IMAGE_REF_SENTINEL_PREFIX),
            "marker leaked into API request: {raw}"
        );
    }

    #[test]
    fn tool_results_sent_as_user_role() {
        let mut h = ConversationHistory::new(String::new());
        h.push_tool_result("c1", "output", false);
        let msgs = h.to_api_messages();
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"][0]["type"], "tool_result");
    }

    #[test]
    fn push_assistant_with_usage() {
        let mut h = ConversationHistory::new(String::new());
        let usage = TurnUsage {
            input_tokens: 10,
            output_tokens: 20,
            ..Default::default()
        };
        h.push_assistant(
            vec![ContentBlock::Text {
                text: "resp".into(),
            }],
            Some(usage.clone()),
        );
        assert_eq!(h.messages()[0].usage, Some(usage));
    }

    #[test]
    fn push_raw_adds_message() {
        let mut h = ConversationHistory::new(String::new());
        h.push_raw(ConversationMessage::user("raw msg"));
        assert_eq!(h.message_count(), 1);
        assert_eq!(h.messages()[0].text_content(), "raw msg");
    }

    #[test]
    fn system_messages_sent_as_user_role_in_api() {
        let mut h = ConversationHistory::new("system".into());
        h.push_raw(ConversationMessage::system("extra system"));
        h.push_user("hi");
        let msgs = h.to_api_messages();
        assert_eq!(msgs.len(), 2);
        assert_eq!(
            msgs[0]["role"], "user",
            "system msg should be mapped to user role"
        );
        assert!(
            msgs[0]["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("extra system")
        );
        assert_eq!(msgs[1]["role"], "user");
    }

    #[test]
    fn estimated_chars_counts_content() {
        let mut h = ConversationHistory::new("sys".into());
        h.push_user("hello");
        let est = h.estimated_tokens();
        assert!(est >= 2); // ("sys" + "hello" = 8 chars) / 4 + 1 = 3
    }

    #[test]
    fn needs_compaction_false_for_small() {
        let h = ConversationHistory::new("short".into());
        assert!(!h.needs_compaction());
    }

    #[test]
    fn fork_clones_history() {
        let mut h = ConversationHistory::new("sys".into());
        h.push_user("msg1");
        let forked = h.fork();
        assert_eq!(forked.message_count(), 1);
        assert_eq!(forked.system_prompt(), "sys");
    }

    #[test]
    fn to_api_messages_tool_use_format() {
        let mut h = ConversationHistory::new(String::new());
        h.push_assistant(
            vec![
                ContentBlock::Text {
                    text: "analyzing".into(),
                },
                ContentBlock::ToolUse {
                    id: "t1".into(),
                    name: "bash".into(),
                    input: serde_json::json!({"command": "ls"}),
                },
            ],
            None,
        );
        let msgs = h.to_api_messages();
        assert_eq!(msgs[0]["content"].as_array().unwrap().len(), 2);
        assert_eq!(msgs[0]["content"][1]["type"], "tool_use");
        assert_eq!(msgs[0]["content"][1]["name"], "bash");
    }

    #[test]
    fn to_api_messages_thinking_format() {
        let mut h = ConversationHistory::new(String::new());
        h.push_assistant(vec![ContentBlock::Thinking { text: "hmm".into() }], None);
        let msgs = h.to_api_messages();
        assert_eq!(msgs[0]["content"][0]["type"], "thinking");
    }

    #[test]
    fn tool_result_error_flag() {
        let mut h = ConversationHistory::new(String::new());
        h.push_tool_result("c1", "failed", true);
        let msgs = h.to_api_messages();
        assert_eq!(msgs[0]["content"][0]["is_error"], true);
    }

    #[test]
    fn compact_reduces_messages() {
        let mut h = ConversationHistory::new(String::new());
        for i in 0..10 {
            h.push_user(&format!("question {i}"));
            h.push_assistant(
                vec![ContentBlock::Text {
                    text: format!("answer {i}"),
                }],
                None,
            );
        }
        assert_eq!(h.message_count(), 20);

        h.compact(4);
        // 1 (system continuation) + 4 kept = 5
        assert_eq!(h.message_count(), 5);
        let continuation = h.messages()[0].text_content();
        assert!(
            continuation.contains("Conversation summary:"),
            "should have structured summary: {continuation}"
        );
        assert!(
            continuation.contains("Key timeline:"),
            "should have timeline: {continuation}"
        );
        assert!(
            continuation.contains("Resume directly"),
            "should have resume instruction"
        );
        assert_eq!(
            h.messages()[0].role,
            Role::System,
            "continuation should be System role"
        );
    }

    #[test]
    fn compact_noop_when_small() {
        let mut h = ConversationHistory::new(String::new());
        h.push_user("hi");
        h.compact(10);
        assert_eq!(h.message_count(), 1);
    }

    #[test]
    fn compact_preserves_recent() {
        let mut h = ConversationHistory::new(String::new());
        for i in 0..10 {
            h.push_user(&format!("msg {i}"));
        }
        assert_eq!(h.message_count(), 10);

        h.compact(4);
        // 1 (system continuation) + 4 (preserved recent) = 5
        assert_eq!(h.message_count(), 5);
        // Last message should be the most recent
        assert_eq!(h.messages()[4].text_content(), "msg 9");
        assert_eq!(h.messages()[3].text_content(), "msg 8");
    }

    #[test]
    fn estimate_image_tokens_low_is_cheaper_than_high() {
        // detail=low → flat 85 tokens. detail=high scales with tiles.
        // For any non-trivial image size, high must cost strictly more.
        let b64_len = 90_000; // roughly one tile
        let low = estimate_image_tokens_in_chars(b64_len, Some(crate::types::ImageDetail::Low));
        let high = estimate_image_tokens_in_chars(b64_len, Some(crate::types::ImageDetail::High));
        let auto = estimate_image_tokens_in_chars(b64_len, Some(crate::types::ImageDetail::Auto));
        let default = estimate_image_tokens_in_chars(b64_len, None);
        assert!(
            low < high,
            "detail=low ({low}) must be cheaper than high ({high})"
        );
        assert_eq!(high, auto, "high and auto both trigger tile math");
        assert_eq!(high, default, "None must default to the same as high/auto");
    }

    #[test]
    fn estimate_image_tokens_scales_with_size() {
        let small = estimate_image_tokens_in_chars(10_000, None);
        let big = estimate_image_tokens_in_chars(1_000_000, None);
        assert!(big > small);
    }

    #[test]
    fn compact_mentions_image_count_in_summary() {
        // Ensure compaction preserves *some* awareness of images that
        // existed in the compacted window, so the assistant doesn't
        // silently lose vision context.
        let mut h = ConversationHistory::new(String::new());
        for _ in 0..3 {
            h.push_user_multimodal(vec![
                ContentBlock::Text {
                    text: "please describe".into(),
                },
                ContentBlock::Image {
                    mime: "image/png".into(),
                    data_base64: "AAAA".into(),
                    detail: None,
                },
            ]);
            h.push_assistant(
                vec![ContentBlock::Text {
                    text: "a cat".into(),
                }],
                None,
            );
        }
        // Add more turns to force compaction.
        for i in 0..12 {
            h.push_user(&format!("q{i}"));
            h.push_assistant(
                vec![ContentBlock::Text {
                    text: format!("a{i}"),
                }],
                None,
            );
        }

        h.compact(4);
        let sys = h.messages().first().unwrap();
        assert_eq!(sys.role, Role::System);
        let sys_text = sys.text_content();
        assert!(
            sys_text.contains("Images in compacted turns"),
            "summary must mention image count, got:\n{sys_text}"
        );
    }

    #[test]
    fn compact_recompaction_merges_summaries() {
        let mut h = ConversationHistory::new(String::new());
        for i in 0..12 {
            h.push_user(&format!("phase1 question {i}"));
            h.push_assistant(
                vec![ContentBlock::Text {
                    text: format!("phase1 answer {i}"),
                }],
                None,
            );
        }
        h.compact(4);
        let first_count = h.message_count();

        // Add more messages and compact again
        for i in 0..10 {
            h.push_user(&format!("phase2 question {i}"));
            h.push_assistant(
                vec![ContentBlock::Text {
                    text: format!("phase2 answer {i}"),
                }],
                None,
            );
        }
        h.compact(4);

        assert!(h.message_count() < first_count + 20);
        let continuation = h.messages()[0].text_content();
        assert!(
            continuation.contains("Previously compacted context:"),
            "re-compaction should merge: {continuation}"
        );
        assert!(
            continuation.contains("Newly compacted context:"),
            "re-compaction should have new section: {continuation}"
        );
    }

    #[test]
    fn auto_compact_triggers_on_threshold() {
        let mut h = ConversationHistory {
            last_compaction_summary: None,
            system_prompt: String::new(),
            messages: Vec::new(),
            context_window_tokens: 50,
            last_input_tokens: None,
        };
        for _ in 0..20 {
            h.push_user(&"x".repeat(10));
        }
        assert!(h.needs_compaction());
        let result = h.auto_compact();
        assert!(result.is_some());
        let (before, after) = result.unwrap();
        assert_eq!(before, 20);
        assert!(after < 20);
    }

    #[test]
    fn auto_compact_returns_none_when_not_needed() {
        let mut h = ConversationHistory::new(String::new());
        h.push_user("short");
        assert!(h.auto_compact().is_none());
    }

    #[test]
    fn needs_compaction_triggered_by_input_tokens() {
        let mut h = ConversationHistory {
            last_compaction_summary: None,
            system_prompt: String::new(),
            messages: Vec::new(),
            context_window_tokens: 1000,
            last_input_tokens: None,
        };
        // Fill >30% of token budget so API-reported trigger can fire
        // Need estimated_tokens > 300 (30% of 1000) → need > 1200 chars
        for _ in 0..15 {
            h.push_user(&"x".repeat(100));
        }
        assert!(!h.needs_compaction());
        // API-reported input_tokens >90% AND estimated >30% → triggers
        h.set_last_input_tokens(950);
        assert!(h.needs_compaction());
    }

    #[test]
    fn needs_compaction_token_only_no_false_positive() {
        let mut h = ConversationHistory::new(String::new());
        h.set_context_window_tokens(100_000);
        h.push_user("small");
        h.set_last_input_tokens(95_000);
        // Token count is high but char estimate is tiny → no compaction
        assert!(!h.needs_compaction());
    }

    #[test]
    fn model_context_window_known_models() {
        assert_eq!(super::model_context_window("glm-5-turbo"), 200_000);
        assert_eq!(
            super::model_context_window("claude-sonnet-4-20250514"),
            200_000
        );
        assert_eq!(super::model_context_window("gpt-4o"), 128_000);
        assert_eq!(super::model_context_window("moonshot-v1-8k"), 8_000);
        assert_eq!(super::model_context_window("unknown-model-xyz"), 128_000);
    }

    #[test]
    fn set_context_window_tokens_uses_1x_ratio() {
        let mut h = ConversationHistory::new(String::new());
        h.set_context_window_tokens(100_000);
        assert_eq!(h.context_window_tokens(), 100_000);
        let mut h2 = ConversationHistory {
            last_compaction_summary: None,
            system_prompt: String::new(),
            messages: Vec::new(),
            context_window_tokens: 0,
            last_input_tokens: None,
        };
        h2.set_context_window_tokens(100_000);
        assert_eq!(h2.context_window_tokens(), 100_000);
    }

    #[test]
    fn tool_result_truncation_in_compact() {
        let mut h = ConversationHistory {
            last_compaction_summary: None,
            system_prompt: String::new(),
            messages: Vec::new(),
            context_window_tokens: 100_000,
            last_input_tokens: None,
        };
        h.push_user("old question");
        h.push_assistant(
            vec![ContentBlock::Text {
                text: "old answer".into(),
            }],
            None,
        );
        // Recent message has a huge tool result
        h.push_user("new question");
        h.push_assistant(
            vec![ContentBlock::ToolUse {
                id: "c1".into(),
                name: "read_file".into(),
                input: serde_json::json!({"path": "/tmp/big.txt"}),
            }],
            None,
        );
        let big_output = "x".repeat(5000);
        h.push_tool_result("c1", &big_output, false);

        h.compact(3); // keep 3 recent messages

        // The kept tool result should be truncated to ~2000 chars
        let tool_msg = h.messages().iter().find(|m| m.role == Role::Tool).unwrap();
        let output = match &tool_msg.blocks[0] {
            ContentBlock::ToolResult { output, .. } => output.clone(),
            _ => panic!("expected ToolResult"),
        };
        assert!(
            output.len() < 2500,
            "tool result should be truncated, got {} chars",
            output.len()
        );
        assert!(
            output.contains("truncated"),
            "should contain truncation marker"
        );
    }

    #[test]
    fn auto_compact_reduces_message_count() {
        let mut h = ConversationHistory {
            last_compaction_summary: None,
            system_prompt: String::new(),
            messages: Vec::new(),
            context_window_tokens: 5000,
            last_input_tokens: None,
        };
        // 20 messages with long content — summary will be shorter due to truncation
        for i in 0..20 {
            h.push_user(&format!("question {i}: {}", "a".repeat(500)));
            h.push_assistant(
                vec![ContentBlock::Text {
                    text: format!("answer {i}: {}", "b".repeat(500)),
                }],
                None,
            );
        }
        let before_count = h.message_count();
        assert!(h.estimated_chars() > 5000, "should exceed limit");

        let result = h.auto_compact();
        assert!(result.is_some());
        let (before, after) = result.unwrap();
        assert_eq!(before, before_count);
        assert!(
            after < before,
            "should have fewer messages: {before} -> {after}"
        );
    }

    #[test]
    fn compact_preserves_image_blocks_in_recent_messages() {
        // Messages within keep_recent must keep their Image blocks intact —
        // compaction summarizes *removed* messages, never edits preserved ones.
        let mut h = ConversationHistory::new(String::new());
        for i in 0..15 {
            h.push_user(&format!("filler {i}"));
        }
        h.push_user_multimodal(vec![
            ContentBlock::Image {
                mime: "image/png".into(),
                data_base64: "iVBORw0KGgo=".into(),
                detail: None,
            },
            ContentBlock::Text {
                text: "describe this image".into(),
            },
        ]);
        h.compact(3);
        let last = h.messages.last().expect("must have a recent message");
        assert!(
            last.blocks
                .iter()
                .any(|b| matches!(b, ContentBlock::Image { .. })),
            "image must survive compaction when inside keep_recent"
        );
    }

    #[test]
    fn compact_summary_mentions_dropped_images() {
        // Images in the *removed* tail can't be carried into the summary
        // verbatim (cost), but the deterministic summary should at least
        // record their presence so the assistant doesn't believe the user
        // sent only text earlier in the conversation.
        let mut h = ConversationHistory::new(String::new());
        h.push_user_multimodal(vec![
            ContentBlock::Image {
                mime: "image/png".into(),
                data_base64: "iVBORw0KGgo=".into(),
                detail: None,
            },
            ContentBlock::Text {
                text: "old image attached".into(),
            },
        ]);
        for i in 0..15 {
            h.push_user(&format!("later message {i}"));
        }
        h.compact(3);
        let summary_text = h
            .messages
            .first()
            .map(|m| m.text_content())
            .unwrap_or_default();
        assert!(
            summary_text.contains("[image"),
            "compaction summary should record image presence; got: {summary_text}"
        );
    }

    #[test]
    fn compact_summary_includes_file_tracking() {
        let mut h = ConversationHistory {
            last_compaction_summary: None,
            system_prompt: String::new(),
            messages: Vec::new(),
            context_window_tokens: 100_000,
            last_input_tokens: None,
        };
        // Tool calls in early messages (will be removed during compaction)
        h.push_user("read src/main.rs and write src/lib.rs");
        h.push_assistant(
            vec![
                ContentBlock::ToolUse {
                    id: "c1".into(),
                    name: "read_file".into(),
                    input: serde_json::json!({"path": "/src/main.rs"}),
                },
                ContentBlock::ToolUse {
                    id: "c2".into(),
                    name: "write_file".into(),
                    input: serde_json::json!({"path": "/src/lib.rs"}),
                },
            ],
            None,
        );
        h.push_tool_result("c1", "fn main() {}", false);
        h.push_tool_result("c2", "ok", false);
        // Add enough recent messages so compaction has a tail to preserve
        for i in 0..6 {
            h.push_user(&format!("follow up {i}"));
            h.push_assistant(
                vec![ContentBlock::Text {
                    text: format!("response {i}"),
                }],
                None,
            );
        }
        // 16 messages total, compact(4) will keep 4 recent, remove 12 (including tool calls)

        h.compact(4);

        let summary = h.messages()[0].text_content();
        assert!(
            summary.contains("Key timeline:"),
            "should have timeline: {summary}"
        );
        assert!(
            summary.contains("Tools mentioned:"),
            "should have tools: {summary}"
        );
        assert!(
            summary.contains("tool_use read_file") || summary.contains("read_file"),
            "timeline should mention read_file: {summary}"
        );
    }

    #[test]
    fn auto_compact_clears_last_input_tokens() {
        let mut h = ConversationHistory {
            last_compaction_summary: None,
            system_prompt: String::new(),
            messages: Vec::new(),
            context_window_tokens: 50,
            last_input_tokens: None,
        };
        // 20 * 100 chars = 2000 chars → est_tokens = 501 > 80% of 50 = 40
        for _ in 0..20 {
            h.push_user(&"x".repeat(100));
        }
        h.set_last_input_tokens(45);
        assert!(h.needs_compaction());

        h.auto_compact();

        assert!(
            h.last_input_tokens().is_none(),
            "last_input_tokens should be cleared after compaction"
        );
    }

    #[test]
    fn restore_system_prompt_undoes_inject() {
        let mut h = ConversationHistory::new("base prompt".into());
        let original = h.system_prompt().to_string();

        h.inject_system_context("\n\n[Session instructions]\ndo stuff");
        assert!(h.system_prompt().contains("do stuff"));

        h.restore_system_prompt(original.clone());
        assert_eq!(h.system_prompt(), "base prompt");

        h.inject_system_context("\n\n[Session instructions]\ndo stuff again");
        assert!(h.system_prompt().contains("do stuff again"));
        assert!(
            !h.system_prompt().contains("do stuff\n"),
            "must not accumulate previous injection"
        );

        h.restore_system_prompt(original);
        assert_eq!(h.system_prompt(), "base prompt");
    }

    // ── Compaction v2 tests ─────────────────────────────────────────

    #[test]
    fn compaction_summary_stored_and_retrieved() {
        let mut h = ConversationHistory::new("sys".into());
        assert!(h.last_compaction_summary().is_none());
        h.set_compaction_summary("## Goal\nTest goal".into());
        assert_eq!(h.last_compaction_summary(), Some("## Goal\nTest goal"));
    }

    #[test]
    fn snap_to_turn_boundary_skips_tool_results() {
        let mut h = ConversationHistory::new("sys".into());
        h.push_user("query");
        h.push_assistant(
            vec![ContentBlock::ToolUse {
                id: "c1".into(),
                name: "bash".into(),
                input: serde_json::json!({"command": "ls"}),
            }],
            None,
        );
        h.push_tool_result("c1", "file1.txt", false);
        h.push_user("next question");
        // messages: [user, assistant(tool), tool_result, user]
        // idx=1 (assistant with tool_call) should snap forward to idx=3 (next user)
        let snapped = ConversationHistory::snap_to_turn_boundary(&h.messages, 1, 0);
        assert_eq!(snapped, 3, "should snap past tool_result to next user");
    }

    #[test]
    fn snap_to_turn_boundary_plain_assistant_is_ok() {
        let mut h = ConversationHistory::new("sys".into());
        h.push_user("hi");
        h.push_assistant(
            vec![ContentBlock::Text {
                text: "hello".into(),
            }],
            None,
        );
        h.push_user("bye");
        // idx=1 (plain assistant) is a clean cut point
        let snapped = ConversationHistory::snap_to_turn_boundary(&h.messages, 1, 0);
        assert_eq!(snapped, 1, "plain assistant is a clean boundary");
    }

    #[test]
    fn files_in_compaction_range_extracts_paths() {
        let mut h = ConversationHistory::new("sys".into());
        h.push_user("read file");
        h.push_assistant(
            vec![ContentBlock::ToolUse {
                id: "c1".into(),
                name: "read".into(),
                input: serde_json::json!({"path": "/src/main.rs"}),
            }],
            None,
        );
        h.push_tool_result("c1", "fn main() {}", false);
        h.push_user("edit file");
        h.push_assistant(
            vec![ContentBlock::ToolUse {
                id: "c2".into(),
                name: "edit".into(),
                input: serde_json::json!({"path": "/src/lib.rs", "edits": []}),
            }],
            None,
        );
        h.push_tool_result("c2", "ok", false);
        // Pad with enough messages so tool calls fall in compaction range
        for i in 0..6 {
            h.push_user(&format!("padding {i}"));
            h.push_assistant(vec![ContentBlock::Text { text: "ok".into() }], None);
        }
        // Total: 6 (original) + 12 (padding) = 18 messages. keep=4 → compacts 0..14
        let (read, modified) = h.files_in_compaction_range(4);
        assert!(
            read.contains(&"/src/main.rs".to_string()),
            "should track read: {read:?}"
        );
        assert!(
            modified.contains(&"/src/lib.rs".to_string()),
            "should track edit: {modified:?}"
        );
    }
}
