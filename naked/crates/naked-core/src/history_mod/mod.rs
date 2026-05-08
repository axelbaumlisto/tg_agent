mod api_format;
mod compaction;
pub(crate) use compaction::*;
mod cycle;

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
    pub(super) system_prompt: String,
    pub(super) messages: Vec<ConversationMessage>,
    pub(super) context_window_tokens: u32,
    pub(super) last_input_tokens: Option<u64>,
    /// Summary from the last compaction (for iterative update).
    pub(super) last_compaction_summary: Option<String>,
    /// Number of checkpoint-restart cycles completed.
    pub(super) cycle_count: u32,
}

impl ConversationHistory {
    pub fn new(system_prompt: String) -> Self {
        Self {
            system_prompt,
            messages: Vec::new(),
            context_window_tokens: 128_000,
            last_input_tokens: None,
            last_compaction_summary: None,
            cycle_count: 0,
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

    /// B6: Mutable access for context hooks.
    pub fn messages_mut(&mut self) -> &mut Vec<ConversationMessage> {
        &mut self.messages
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

    /// Push an image into the conversation (for vision models).
    /// Added as a user message with a single Image block.
    pub fn push_image(&mut self, mime: &str, base64_data: &str) {
        self.messages.push(ConversationMessage {
            role: crate::types::Role::User,
            blocks: vec![crate::types::ContentBlock::Image {
                mime: mime.to_string(),
                data_base64: base64_data.to_string(),
                detail: None,
            }],
            timestamp: chrono::Utc::now(),
            usage: None,
        });
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
        self.compact_with_working_set(keep_recent, None);
    }

    /// Compact with optional working-set awareness.
    pub fn compact_with_working_set(
        &mut self,
        keep_recent: usize,
        ws: Option<&crate::working_set::WorkingSet>,
    ) {
        let keep = keep_recent.max(DEFAULT_PRESERVE_RECENT);
        if self.messages.len() <= keep {
            return;
        }

        let existing_summary = self.messages.first().and_then(extract_existing_summary);
        let prefix_len = usize::from(existing_summary.is_some());

        let raw_keep_from = self.messages.len().saturating_sub(keep);
        let keep_from = Self::snap_to_turn_boundary(&self.messages, raw_keep_from, prefix_len);
        let removed = &self.messages[prefix_len..keep_from];
        if removed.is_empty() {
            return;
        }

        let pinned_indices: Vec<usize> = ws
            .map(|w| {
                let stale_cutoff = prefix_len + (removed.len() / 5);
                w.pinned_message_indices(&self.messages)
                    .into_iter()
                    .filter(|&i| i >= stale_cutoff && i < keep_from)
                    .take(4)
                    .collect()
            })
            .unwrap_or_default();
        let pinned_msgs: Vec<ConversationMessage> = pinned_indices
            .iter()
            .map(|&i| self.messages[i].clone())
            .collect();

        let preserved = self.messages[keep_from..].to_vec();

        let mut new_summary = summarize_messages(removed);
        if let Some(w) = ws {
            let top = w.top_paths(8);
            if !top.is_empty() {
                let ws_line = format!("\n- Active files (working set): {}.", top.join(", "));
                if let Some(pos) = new_summary.rfind("</summary>") {
                    new_summary.insert_str(pos, &ws_line);
                } else {
                    new_summary.push_str(&ws_line);
                }
            }
        }

        let merged = merge_summaries(existing_summary.as_deref(), &new_summary);
        let continuation =
            build_continuation_message(&merged, !preserved.is_empty() || !pinned_msgs.is_empty());

        let mut new_messages = vec![ConversationMessage::system(&continuation)];
        new_messages.extend(pinned_msgs);
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

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
