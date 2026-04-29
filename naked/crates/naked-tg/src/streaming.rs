//! Streaming response handler + CompositeView extracted from main.rs.
//!
//! Handles: stream_response, CompositeView, flush_live, edit_with_retry,
//! send_final, ask_permission, render_html_document, format helpers.

use super::*;

// ── Streaming response ──────────────────────────────────────────────────────

pub(crate) struct SubAgentState {
    prompt: String,
    status: &'static str,
    last_tool: Option<String>,
    tool_count: u32,
}

pub(crate) struct CompositeView {
    thinking: String,
    in_thinking: bool,
    tool_lines: Vec<String>,
    sub_agents: std::collections::HashMap<String, SubAgentState>,
    sub_agent_order: Vec<String>,
    response_text: String,
    usage: Option<TurnUsage>,
    model_tag: String,
    tick: usize,
    phase: &'static str,
    started_at: std::time::Instant,
    /// Live count of messages queued while this turn is active.
    queue_counter: Arc<std::sync::atomic::AtomicUsize>,
}

const SPINNER: &[&str] = &["⏳", "⌛", "⏳", "⌛"];

impl CompositeView {
    fn new(model_tag: String, queue_counter: Arc<std::sync::atomic::AtomicUsize>) -> Self {
        Self {
            thinking: String::new(),
            in_thinking: false,
            tool_lines: Vec::new(),
            sub_agents: std::collections::HashMap::new(),
            sub_agent_order: Vec::new(),
            response_text: String::new(),
            usage: None,
            model_tag,
            tick: 0,
            phase: "thinking",
            started_at: std::time::Instant::now(),
            queue_counter,
        }
    }

    fn elapsed_label(&self) -> String {
        let secs = self.started_at.elapsed().as_secs();
        if secs < 60 {
            format!("🕐 {secs}s")
        } else {
            let mins = secs / 60;
            format!("🕐 {mins}min")
        }
    }

    fn spinner(&self) -> &'static str {
        SPINNER[self.tick % SPINNER.len()]
    }

    /// Live composite: spinner status + reasoning tail + last N tools.
    fn render_live(&self) -> String {
        let spin = self.spinner();
        let tokens = if let Some(u) = &self.usage {
            format!(
                " · ↑{} ↓{}",
                naked_tg::tg_markup::format_tokens(u.input_tokens),
                naked_tg::tg_markup::format_tokens(u.output_tokens)
            )
        } else {
            String::new()
        };
        let elapsed = self.elapsed_label();
        let queued = self
            .queue_counter
            .load(std::sync::atomic::Ordering::Relaxed);
        let queue_str = if queued > 0 {
            format!(" · +{queued} queued")
        } else {
            String::new()
        };
        let status = format!(
            "{spin} <i>{} · {elapsed}{tokens}{queue_str}</i>",
            self.phase
        );

        let mut parts = vec![status];

        if self.in_thinking && !self.thinking.is_empty() {
            let tail = if self.thinking.len() > REASONING_TAIL {
                let mut boundary = self.thinking.len() - REASONING_TAIL;
                while boundary < self.thinking.len() && !self.thinking.is_char_boundary(boundary) {
                    boundary += 1;
                }
                format!("…{}", &self.thinking[boundary..])
            } else {
                self.thinking.clone()
            };
            parts.push(format!("💭 <i>{}</i>", escape_html(&tail)));
        }

        let start = self.tool_lines.len().saturating_sub(TOOL_WINDOW);
        for line in &self.tool_lines[start..] {
            parts.push(line.clone());
        }

        if !self.sub_agents.is_empty() {
            let last_ids: Vec<_> = self
                .sub_agent_order
                .iter()
                .rev()
                .take(3)
                .rev()
                .cloned()
                .collect();
            for id in &last_ids {
                if let Some(sa) = self.sub_agents.get(id) {
                    let icon = match sa.status {
                        "running" => "🤖",
                        "done" => "✅",
                        "error" => "❌",
                        _ => "⏳",
                    };
                    let tool_info = sa
                        .last_tool
                        .as_deref()
                        .map(|t| {
                            if sa.tool_count > 1 {
                                format!(" · {t} ×{}", sa.tool_count)
                            } else {
                                format!(" · {t}")
                            }
                        })
                        .unwrap_or_default();
                    let short = truncate_str(&sa.prompt, 40);
                    parts.push(format!(
                        "{icon} <b>{id}</b> {}{tool_info}",
                        escape_html(&short)
                    ));
                }
            }
        }

        // ── Response text preview ────────────────────────────────
        // Show the agent's response as it generates. Render closed
        // Markdown blocks as HTML; leave the growing tail as escaped
        // plain text so an unclosed fence doesn't break the message.
        if !self.response_text.is_empty() {
            let header_len: usize = parts.iter().map(|p| p.len() + 1).sum();
            let budget = MAX_TG_MSG.saturating_sub(header_len + 100);
            if budget > 50 {
                use naked_tg::tg_markup::split_stable_unstable;

                let text = &self.response_text;
                let (stable, unstable) = split_stable_unstable(text);

                let mut preview = String::new();

                // Render stable (closed) blocks as rich HTML
                if !stable.is_empty() {
                    let html = md_to_tg_html(stable);
                    if html.len() <= budget {
                        preview.push_str(&html);
                    } else {
                        // Stable too big — tail-truncate
                        let tail = truncate_str(&html, budget);
                        preview.push_str(&tail);
                    }
                }

                // Append unstable tail as escaped plain text
                if !unstable.is_empty() {
                    let remaining = budget.saturating_sub(preview.len() + 2);
                    if remaining > 20 {
                        if !preview.is_empty() {
                            preview.push('\n');
                        }
                        let tail = truncate_str(&escape_html(unstable), remaining);
                        preview.push_str(&tail);
                    }
                }

                if !preview.is_empty() {
                    parts.push(preview);
                }
            }
        }

        parts.join("\n")
    }

    fn usage_footer(&self) -> String {
        if let Some(u) = &self.usage {
            let (_, _, cost) = u.estimate_cost(&self.model_tag);
            let cost_str = if cost >= 0.01 {
                format!(" · ${cost:.2}")
            } else if cost > 0.0 {
                format!(" · ${cost:.4}")
            } else {
                String::new()
            };
            format!(
                "↑{} ↓{}{}",
                naked_tg::tg_markup::format_tokens(u.input_tokens),
                naked_tg::tg_markup::format_tokens(u.output_tokens),
                cost_str
            )
        } else {
            String::new()
        }
    }

    /// Final: replace everything with clean response text + reasoning block + footer.
    ///
    /// The reasoning chain (when present) is wrapped in
    /// `<blockquote expandable>` so it ships collapsed by default — users
    /// who want to see the model's thinking just tap to expand. The
    /// thinking block is dynamically squeezed so the whole final message
    /// fits in a single Telegram message (`MAX_TG_MSG`) — no more
    /// "echo-chunk" second posts that leak CoT drafts after the real
    /// answer.
    fn render_final(&self) -> String {
        let text = self.response_text.trim();
        let thinking = self.thinking.trim();
        if text.is_empty() && thinking.is_empty() {
            return "<i>— модель закрыла ход без ответа. Попробуй переформулировать или <code>/new</code>.</i>".into();
        }
        let footer = self.usage_footer();
        let footer_rendered = if footer.is_empty() {
            String::new()
        } else {
            format!("\n\n<i>✓ {footer}</i>")
        };
        let body = if text.is_empty() {
            "<i>(no text — reasoning only)</i>".to_string()
        } else {
            md_to_tg_html(text)
        };

        // Budget: we want `body + thinking_block + footer_rendered`
        // to fit into one TG message. The thinking block is expendable
        // (it's collapsed by default anyway); the primary answer is not.
        // `SINGLE_MESSAGE_TARGET` is a safety margin below MAX_TG_MSG to
        // absorb HTML tag overhead from `<blockquote>` etc.
        const SINGLE_MESSAGE_TARGET: usize = MAX_TG_MSG - 200;
        let fixed_len = body.len() + footer_rendered.len() + 2 /* \n\n separator */;
        let thinking_budget = SINGLE_MESSAGE_TARGET.saturating_sub(fixed_len);

        let thinking_block = self.render_thinking_block_budgeted(thinking, thinking_budget);

        let mut out = String::with_capacity(body.len() + 256);
        out.push_str(&body);
        if let Some(block) = thinking_block {
            out.push_str("\n\n");
            out.push_str(&block);
        }
        out.push_str(&footer_rendered);
        out
    }

    /// Thinking block that respects a dynamic byte budget.
    ///
    /// Returns `None` if the raw chain is empty or if the budget is too
    /// small to fit even a minimal `<blockquote>` header (we'd rather
    /// drop the thinking entirely than half-render a broken tag). The
    /// payload is tail-preserved (the conclusion is at the end of the
    /// CoT) and capped by the lower of `MAX_FINAL_THINKING_BYTES` and
    /// the caller-provided budget.
    fn render_thinking_block_budgeted(&self, trimmed: &str, budget: usize) -> Option<String> {
        if trimmed.is_empty() {
            return None;
        }
        // Overhead of the wrapper tags + our small prefix.
        const WRAPPER_OVERHEAD: usize = 64;
        const MIN_PAYLOAD: usize = 80;
        if budget < WRAPPER_OVERHEAD + MIN_PAYLOAD {
            return None;
        }
        let payload_cap = budget
            .saturating_sub(WRAPPER_OVERHEAD)
            .min(MAX_FINAL_THINKING_BYTES);
        let payload = if trimmed.len() > payload_cap {
            let mut start = trimmed.len() - payload_cap;
            while start < trimmed.len() && !trimmed.is_char_boundary(start) {
                start += 1;
            }
            format!("…{}", &trimmed[start..])
        } else {
            trimmed.to_string()
        };
        Some(format!(
            "<blockquote expandable>💭 <b>thinking</b>\n{}</blockquote>",
            escape_html(&payload)
        ))
    }

    /// Collapsible reasoning block with the default byte budget.
    ///
    /// Retained for tests only — real rendering happens through
    /// `render_thinking_block_budgeted` so that `render_final` can
    /// shrink the CoT to fit the outgoing TG message.
    #[cfg(test)]
    fn render_thinking_block(&self) -> Option<String> {
        let trimmed = self.thinking.trim();
        self.render_thinking_block_budgeted(trimmed, MAX_FINAL_THINKING_BYTES + 64)
    }

    /// Truncated preview for the placeholder message when sending a file.
    fn render_summary(&self, max_chars: usize) -> String {
        let text = self.response_text.trim();
        let converted = md_to_tg_html(text);
        let preview: String = if converted.chars().count() > max_chars {
            let truncated: String = converted.chars().take(max_chars).collect();
            format!("{truncated}…")
        } else {
            converted
        };
        format!(
            "{preview}\n\n📄 <i>Full response attached as file</i>\n<i>{}</i>",
            self.usage_footer()
        )
    }
}

pub(crate) fn format_input_preview(input: &serde_json::Value, max_len: usize) -> String {
    if let Some(map) = input.as_object() {
        if map.len() == 1 {
            let (key, val) = map.iter().next().unwrap();
            let fallback = val.to_string();
            let v = val.as_str().unwrap_or(&fallback);
            return format!("{key}: {}", truncate_str(v, max_len));
        }
        let parts: Vec<String> = map
            .iter()
            .map(|(k, v)| {
                let fallback = v.to_string();
                let s = v.as_str().unwrap_or(&fallback);
                format!("{k}: {}", truncate_str(s, 60))
            })
            .collect();
        truncate_str(&parts.join(", "), max_len)
    } else {
        truncate_str(&input.to_string(), max_len)
    }
}

pub(crate) fn truncate_str(s: &str, max_chars: usize) -> String {
    let mut last_boundary = 0;
    for (i, (byte_pos, _)) in s.char_indices().enumerate() {
        if i >= max_chars {
            return format!("{}…", &s[..last_boundary]);
        }
        last_boundary = byte_pos;
    }
    s.to_string()
}

fn render_html_document(view: &CompositeView) -> Vec<u8> {
    let text = view.response_text.trim();
    let content_html = md_to_tg_html(text).replace('\n', "<br>\n");

    // The full reasoning chain — no truncation here, this is the
    // attached HTML file precisely so users can see everything.
    // We render it in its own <details> section so it's collapsed by
    // default just like the inline message blockquote.
    let thinking = view.thinking.trim();
    let thinking_block = if thinking.is_empty() {
        String::new()
    } else {
        format!(
            "<details class=\"thinking\"><summary>💭 reasoning ({} chars)</summary><pre>{}</pre></details>",
            thinking.len(),
            escape_html(thinking)
        )
    };

    let footer = view.usage_footer().replace(" · ", " &middot; ");

    let doc = format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Response</title>
<style>
:root {{
  --bg: #1e1e2e;
  --fg: #cdd6f4;
  --muted: #6c7086;
  --surface: #313244;
  --code-bg: #181825;
  --accent: #89b4fa;
  --border: #45475a;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  background: var(--bg);
  color: var(--fg);
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  font-size: 15px;
  line-height: 1.7;
  padding: 24px;
  max-width: 900px;
  margin: 0 auto;
}}
pre {{
  background: var(--code-bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px;
  overflow-x: auto;
  margin: 12px 0;
  font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
  font-size: 13px;
  line-height: 1.5;
}}
code {{
  background: var(--code-bg);
  border-radius: 4px;
  padding: 2px 6px;
  font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
  font-size: 13px;
}}
pre code {{
  background: none;
  padding: 0;
}}
a {{
  color: var(--accent);
  text-decoration: none;
}}
a:hover {{
  text-decoration: underline;
}}
blockquote {{
  border-left: 3px solid var(--accent);
  padding-left: 16px;
  margin: 12px 0;
  color: var(--muted);
}}
hr {{
  border: none;
  border-top: 1px solid var(--border);
  margin: 20px 0;
}}
.footer {{
  margin-top: 32px;
  padding-top: 16px;
  border-top: 1px solid var(--border);
  color: var(--muted);
  font-size: 13px;
}}
.thinking {{
  margin-top: 24px;
  padding: 12px 16px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  color: var(--muted);
  font-size: 13px;
}}
.thinking summary {{
  cursor: pointer;
  font-weight: 600;
  color: var(--accent);
}}
.thinking pre {{
  margin-top: 12px;
  background: var(--code-bg);
  white-space: pre-wrap;
}}
</style>
</head>
<body>
<div class="content">
{content_html}
</div>
{thinking_block}
<div class="footer">{footer}</div>
</body>
</html>"#
    );

    doc.into_bytes()
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn stream_response(
    bot: Bot,
    ctx: ChatCtx,
    handle: AgentHandle,
    channel_map: &ChannelSessionMap,
    pending_perms: &PendingPermissions,
    model_tag: String,
    http_client: &reqwest::Client,
    base_url: &str,
    tg_attach_queue: &naked_tg::tg_attach::AttachmentQueue,
    rate_limiter: &naked_tg::rate_limit::RateLimiter,
) {
    let AgentHandle {
        mut events,
        permissions,
    } = handle;
    let chat_id_raw = ctx.chat_id.0;
    let tid = ctx.raw_thread_id();

    tracing::debug!(
        chat_id = chat_id_raw,
        ?tid,
        thread_id_raw = ?ctx.thread_id,
        "stream_response: sending typing"
    );
    send_typing_raw(http_client, base_url, chat_id_raw, tid).await;

    let placeholder = match bot
        .send_message(ctx.chat_id, "⏳")
        .maybe_thread(ctx.thread_id)
        .maybe_reply_to(ctx.reply_to)
        .await
    {
        Ok(m) => m.id,
        Err(e) => {
            tracing::error!("Failed to send placeholder: {e}");
            return;
        }
    };

    let typing_client = http_client.clone();
    let typing_base = base_url.to_string();
    let typing_cancel = tokio_util::sync::CancellationToken::new();
    let typing_token = typing_cancel.clone();
    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = typing_token.cancelled() => break,
                _ = tokio::time::sleep(TYPING_INTERVAL) => {
                    send_typing_raw(&typing_client, &typing_base, chat_id_raw, tid).await;
                }
            }
        }
    });

    // Register per-chat model-switch state so `/model` callbacks can
    // request an in-flight switch during this stream.
    let chat_key = (chat_id_raw, tid);
    let model_switch = naked_tg::model_switch::new_shared();
    MODEL_SWITCHES
        .write()
        .await
        .insert(chat_key, model_switch.clone());

    // Register queue counter (reset to 0 for this turn).
    let queue_counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    QUEUE_COUNTS
        .write()
        .await
        .insert(chat_key, queue_counter.clone());

    let mut view = CompositeView::new(model_tag, queue_counter);
    let mut last_edit = tokio::time::Instant::now();
    let mut dirty = false;
    let mut last_sent = String::new();
    let mut aborted_for_switch = false;

    loop {
        let event = tokio::select! {
            ev = events.recv() => match ev {
                Some(e) => Some(e),
                None => break,
            },
            _ = tokio::time::sleep(rate_limiter.gap(chat_key).await) => None,
        };

        let mut force_flush = false;

        if let Some(event) = event {
            match event {
                AgentEvent::ThinkingDelta(t) => {
                    view.in_thinking = true;
                    view.phase = "thinking";
                    if view.thinking.len() < MAX_THINKING_BYTES {
                        view.thinking.push_str(&t);
                    }
                    dirty = true;
                }
                AgentEvent::TextDelta(t) => {
                    view.in_thinking = false;
                    view.phase = "generating";
                    if view.response_text.len() < MAX_RESPONSE_BYTES {
                        view.response_text.push_str(&t);
                    }
                    dirty = true;
                }
                AgentEvent::ToolStart { name, input, .. } => {
                    view.in_thinking = false;
                    view.phase = "tool use";
                    let preview = format_input_preview(&input, 200);
                    if view.tool_lines.len() >= TOOL_WINDOW * 4 {
                        view.tool_lines.drain(..view.tool_lines.len() - TOOL_WINDOW);
                    }
                    view.tool_lines
                        .push(format!("🔧 <b>{}</b>({preview})…", escape_html(&name)));
                    dirty = true;
                    force_flush = true;
                }
                AgentEvent::ToolEnd {
                    name,
                    state,
                    output,
                    ..
                } => {
                    let icon = match state {
                        naked_core::types::ToolState::Completed => "✅",
                        naked_core::types::ToolState::Error => "❌",
                    };
                    let title = truncate_str(&output, 80);
                    view.tool_lines.push(format!(
                        "{icon} <b>{}</b> — {}",
                        escape_html(&name),
                        escape_html(&title)
                    ));
                    dirty = true;
                    force_flush = true;
                }
                AgentEvent::PermissionRequest {
                    call_id,
                    tool_name,
                    input,
                    permission,
                } => {
                    rate_limiter.acquire(chat_key).await;
                    let _ = flush_live(&bot, ctx.chat_id, placeholder, &view, &mut last_sent).await;
                    dirty = false;

                    let auto = channel_map
                        .should_auto_approve(chat_id_raw, tid, &tool_name)
                        .await;
                    tracing::info!(
                        chat_id = chat_id_raw, ?tid,
                        tool = %tool_name, auto_approve = auto,
                        "permission check"
                    );
                    if auto {
                        let _ = permissions
                            .send(PermissionResponse {
                                call_id,
                                allowed: true,
                            })
                            .await;
                    } else {
                        let allowed = ask_permission(
                            &bot,
                            ctx,
                            &call_id,
                            &tool_name,
                            &input,
                            &permission,
                            pending_perms,
                        )
                        .await;
                        let _ = permissions
                            .send(PermissionResponse { call_id, allowed })
                            .await;
                    }
                }
                AgentEvent::ContextCompacted {
                    before_msgs,
                    after_msgs,
                } => {
                    let note = format!(
                        "📦 контекст был сжат: {} сообщений → {}",
                        before_msgs, after_msgs
                    );
                    let _ = bot
                        .send_message(ctx.chat_id, &note)
                        .maybe_thread(ctx.thread_id)
                        .maybe_reply_to(ctx.reply_to)
                        .await;
                }
                AgentEvent::Heartbeat => {
                    view.tick += 1;
                    dirty = true;
                }
                AgentEvent::SubAgentProgress {
                    agent_id,
                    event: sa_ev,
                } => {
                    use naked_core::types::SubAgentEvent;
                    match sa_ev {
                        SubAgentEvent::Started { prompt_preview } => {
                            view.phase = "sub_agent";
                            if !view.sub_agent_order.contains(&agent_id) {
                                view.sub_agent_order.push(agent_id.clone());
                            }
                            view.sub_agents.insert(
                                agent_id,
                                SubAgentState {
                                    prompt: prompt_preview,
                                    status: "running",
                                    last_tool: None,
                                    tool_count: 0,
                                },
                            );
                        }
                        SubAgentEvent::ToolUse { name, .. } => {
                            if let Some(sa) = view.sub_agents.get_mut(&agent_id) {
                                sa.last_tool = Some(name);
                                sa.tool_count = sa.tool_count.saturating_add(1);
                            }
                        }
                        SubAgentEvent::ToolDone { .. } => {}
                        SubAgentEvent::TextDelta(_) => {}
                        SubAgentEvent::Finished { .. } => {
                            if let Some(sa) = view.sub_agents.get_mut(&agent_id) {
                                sa.status = "done";
                                sa.last_tool = None;
                            }
                        }
                        SubAgentEvent::Error(_) => {
                            if let Some(sa) = view.sub_agents.get_mut(&agent_id) {
                                sa.status = "error";
                            }
                        }
                    }
                    dirty = true;
                    force_flush = true;
                }
                AgentEvent::UsageUpdate(u) => {
                    view.usage = Some(u);
                    dirty = true;
                }
                AgentEvent::Error(e) => {
                    // Friendly mapping for the one specific class of
                    // provider failures we see often enough to warrant
                    // a human-readable hint: the "0-token refusal" that
                    // glm-5-turbo and a few OpenAI-compatible gateways
                    // fall into when they decline without explaining.
                    // The agent loop surfaces these as:
                    //   "provider returned no content ..."
                    // Surfacing that raw string is confusing; we replace
                    // it with an actionable suggestion.
                    let pretty = if e.contains("no content") {
                        "— модель закрыла ход без ответа (0 токенов). \
                         Попробуй переформулировать запрос или открой новую сессию: /new."
                            .to_string()
                    } else {
                        format!("❌ {e}")
                    };
                    if !view.response_text.is_empty() {
                        view.response_text.push('\n');
                    }
                    view.response_text.push_str(&pretty);
                    dirty = true;
                    force_flush = true;
                }
                AgentEvent::Idle => break,
            }
        } // end if let Some(event)

        // Check for in-flight model switch at every yield point.
        if naked_tg::model_switch::check_and_take(&model_switch)
            .await
            .is_some()
        {
            aborted_for_switch = true;
            break;
        }

        // Tick spinner + flush via adaptive rate limiter.
        let current_gap = rate_limiter.gap(chat_key).await;
        if force_flush || last_edit.elapsed() >= current_gap {
            view.tick += 1;
            dirty = true;
        }

        if dirty && last_edit.elapsed() >= current_gap {
            rate_limiter.acquire(chat_key).await;
            let ok = flush_live(&bot, ctx.chat_id, placeholder, &view, &mut last_sent).await;
            if ok {
                rate_limiter.report_ok(chat_key).await;
            } else {
                rate_limiter.report_429(chat_key, None).await;
            }
            last_edit = tokio::time::Instant::now();
            dirty = false;
        }
    }

    typing_cancel.cancel();

    // Deregister model-switch state and queue counter.
    MODEL_SWITCHES.write().await.remove(&chat_key);
    QUEUE_COUNTS.write().await.remove(&chat_key);

    if aborted_for_switch {
        // Don't send final — the turn was interrupted.
        // Edit placeholder to indicate switch in progress.
        let _ = bot
            .edit_message_text(ctx.chat_id, placeholder, "⚡ Switching model…")
            .await;
        return;
    }

    let final_html = view.render_final();
    send_final(bot.clone(), ctx, placeholder, &final_html, &view).await;

    // Deliver any files queued by telegram_attach tool.
    let attachments: Vec<naked_tg::tg_attach::StagedAttachment> =
        tg_attach_queue.lock().await.drain(..).collect();
    for att in attachments {
        let method = if naked_tg::tg_attach::is_image_path(&att.path) {
            "sendPhoto"
        } else {
            "sendDocument"
        };
        let field = if method == "sendPhoto" {
            "photo"
        } else {
            "document"
        };
        let form = reqwest::multipart::Form::new()
            .text("chat_id", ctx.chat_id.0.to_string())
            .file(field, &att.path)
            .await;
        match form {
            Ok(form) => {
                let url = format!("{base_url}/{method}");
                match http_client.post(&url).multipart(form).send().await {
                    Ok(resp) if resp.status().is_success() => {
                        tracing::info!(file = %att.file_name, "telegram_attach delivered");
                    }
                    Ok(resp) => {
                        tracing::warn!(
                            file = %att.file_name,
                            status = %resp.status(),
                            "telegram_attach delivery failed"
                        );
                    }
                    Err(e) => {
                        tracing::warn!(file = %att.file_name, error = %e, "telegram_attach send error");
                    }
                }
            }
            Err(e) => {
                tracing::warn!(file = %att.file_name, error = %e, "telegram_attach form build failed");
            }
        }
    }
}

const FILE_THRESHOLD: usize = MAX_TG_MSG * 2;
const SUMMARY_CHARS: usize = 500;

/// Edit with retry: if rate-limited, waits the indicated duration and retries.
pub(crate) async fn edit_with_retry(
    bot: &Bot,
    chat_id: ChatId,
    msg_id: MessageId,
    text: &str,
    parse_html: bool,
) -> bool {
    let modes: &[bool] = if parse_html { &[true, false] } else { &[false] };

    for &use_html in modes {
        for attempt in 0..3 {
            let result = if use_html {
                bot.edit_message_text(chat_id, msg_id, text)
                    .parse_mode(ParseMode::Html)
                    .await
            } else {
                bot.edit_message_text(chat_id, msg_id, text).await
            };
            match result {
                Ok(_) => return true,
                Err(e) => {
                    let err_str = e.to_string();
                    if let Some(wait) = parse_retry_after(&err_str) {
                        let wait = wait.min(60);
                        tracing::warn!(
                            attempt,
                            wait,
                            use_html,
                            "final edit rate-limited, waiting {wait}s"
                        );
                        tokio::time::sleep(Duration::from_secs(wait + 1)).await;
                        continue;
                    }
                    if use_html && attempt == 0 {
                        tracing::warn!("final edit (HTML) failed: {e}, falling back to plain text");
                        break;
                    }
                    tracing::error!("final edit failed: {e}");
                    return false;
                }
            }
        }
    }
    false
}

pub(crate) async fn send_final(
    bot: Bot,
    ctx: ChatCtx,
    msg_id: MessageId,
    html: &str,
    view: &CompositeView,
) {
    let chat_id = ctx.chat_id;

    // Short: fits in one message
    if html.len() <= MAX_TG_MSG {
        edit_with_retry(&bot, chat_id, msg_id, html, true).await;
        return;
    }

    // Medium: fits in 2 chunks
    if html.len() <= FILE_THRESHOLD {
        let chunks = split_html(html, MAX_TG_MSG - 100);
        if let Some(first) = chunks.first() {
            edit_with_retry(&bot, chat_id, msg_id, first, true).await;
        }
        for chunk in chunks.iter().skip(1) {
            let res = bot
                .send_message(chat_id, chunk.as_str())
                .parse_mode(ParseMode::Html)
                .maybe_thread(ctx.thread_id)
                .maybe_reply_to(ctx.reply_to)
                .await;
            if let Err(e) = res {
                tracing::warn!("send chunk (HTML) failed: {e}, retrying plain text");
                if let Err(e2) = bot
                    .send_message(chat_id, chunk.as_str())
                    .maybe_thread(ctx.thread_id)
                    .maybe_reply_to(ctx.reply_to)
                    .await
                {
                    tracing::error!("send chunk (plain) also failed: {e2}");
                }
            }
        }
        return;
    }

    let summary = view.render_summary(SUMMARY_CHARS);
    edit_with_retry(&bot, chat_id, msg_id, &summary, true).await;

    let html_doc = render_html_document(view);
    let input_file = teloxide::types::InputFile::memory(html_doc).file_name("response.html");
    if let Err(e) = bot
        .send_document(chat_id, input_file)
        .caption("📄 Full response")
        .maybe_thread(ctx.thread_id)
        .maybe_reply_to(ctx.reply_to)
        .await
    {
        tracing::warn!("send_document failed: {e}");
    }
}

pub(crate) async fn send_long_text(
    bot: &Bot,
    ctx: ChatCtx,
    text: &str,
) -> Result<(), teloxide::RequestError> {
    if text.len() <= MAX_TG_MSG {
        bot.send_message(ctx.chat_id, text)
            .maybe_thread(ctx.thread_id)
            .await?;
        return Ok(());
    }
    for chunk in split_html(text, MAX_TG_MSG - 100) {
        bot.send_message(ctx.chat_id, chunk)
            .maybe_thread(ctx.thread_id)
            .await?;
    }
    Ok(())
}

// ── Permission prompt ───────────────────────────────────────────────────────

pub(crate) async fn ask_permission(
    bot: &Bot,
    ctx: ChatCtx,
    call_id: &str,
    tool_name: &str,
    input: &serde_json::Value,
    permission: &Permission,
    pending: &PendingPermissions,
) -> bool {
    let level = match permission {
        Permission::WorkspaceWrite => "write",
        Permission::Dangerous => "dangerous",
        Permission::ReadOnly => "read",
    };
    let preview = format_input_preview(input, 200);
    let text = format!(
        "🔐 <b>Permission required</b> [{level}]\n\n<b>{}</b>({preview})",
        escape_html(tool_name),
    );

    let (tx, rx) = oneshot::channel();
    let chat_id_raw = ctx.chat_id.0;
    let tid_raw = ctx.raw_thread_id();
    pending
        .write()
        .await
        .insert(call_id.to_string(), (tx, chat_id_raw, tid_raw));

    let keyboard = InlineKeyboardMarkup::new(vec![vec![
        InlineKeyboardButton::callback("✅ Allow", format!("p:{call_id}:allow")),
        InlineKeyboardButton::callback("❌ Deny", format!("p:{call_id}:deny")),
        InlineKeyboardButton::callback("⚡ YOLO", format!("p:{call_id}:yolo")),
    ]]);

    let sent = bot
        .send_message(ctx.chat_id, &text)
        .parse_mode(ParseMode::Html)
        .reply_markup(keyboard)
        .maybe_thread(ctx.thread_id)
        .await;

    if sent.is_err() {
        pending.write().await.remove(call_id);
        return false;
    }

    match tokio::time::timeout(PERMISSION_TIMEOUT, rx).await {
        Ok(Ok(allowed)) => allowed,
        _ => {
            pending.write().await.remove(call_id);
            false
        }
    }
}

/// Parse "Retry after Xs" from Telegram error string.
pub(crate) fn parse_retry_after(err: &str) -> Option<u64> {
    let s = err.to_lowercase();
    if let Some(pos) = s.find("retry after") {
        let after = &s[pos + 12..];
        let digits: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
        digits.parse().ok()
    } else {
        None
    }
}

/// Deduplicated edit: only sends if content changed. Acquires global rate limiter
/// slot before sending. On rate limit from Telegram, backs off.
/// Flush the live preview. Returns `true` on success, `false` on 429.
/// Caller is responsible for rate limiting (acquire before, report after).
pub(crate) async fn flush_live(
    bot: &Bot,
    chat_id: ChatId,
    msg_id: MessageId,
    view: &CompositeView,
    last_sent: &mut String,
) -> bool {
    let html = view.render_live();
    let trimmed = truncate_str(&html, MAX_TG_MSG - 50);
    if trimmed == *last_sent {
        return true; // no-op counts as success
    }
    *last_sent = trimmed.clone();
    let result = bot
        .edit_message_text(chat_id, msg_id, &trimmed)
        .parse_mode(ParseMode::Html)
        .await;
    match result {
        Ok(_) => true,
        Err(e) => {
            let err_str = e.to_string();
            if err_str.contains("429") || err_str.contains("Too Many Requests") {
                tracing::debug!("rate-limited on live edit");
                *last_sent = String::new(); // force re-send next time
                false
            } else if err_str.contains("not modified") {
                true // not an error
            } else {
                tracing::warn!("edit_message_text error: {e}");
                true // non-rate-limit error, don't backoff
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media_dispatch::{
        MediaItem, MediaProcessed, NativeImage, StickerFormat, decide_native_route, fmt_duration,
        looks_like_supported_image,
    };

    // ── render_thinking_block ───────────────────────────────────────────

    fn view_with_response_and_thinking(response: &str, thinking: &str) -> CompositeView {
        let mut v = CompositeView::new(
            "test-model".into(),
            Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        );
        v.response_text = response.to_string();
        v.thinking = thinking.to_string();
        v
    }

    #[test]
    fn render_thinking_block_omitted_when_empty() {
        let v = view_with_response_and_thinking("hi", "");
        assert!(v.render_thinking_block().is_none());
        // Final must not contain blockquote when there's no reasoning.
        assert!(!v.render_final().contains("<blockquote"));
    }

    #[test]
    fn render_thinking_block_emits_collapsible_blockquote() {
        let v = view_with_response_and_thinking("answer", "step 1\nstep 2");
        let block = v.render_thinking_block().expect("has thinking");
        assert!(block.starts_with("<blockquote expandable>"));
        assert!(block.ends_with("</blockquote>"));
        assert!(block.contains("💭 <b>thinking</b>"));
        assert!(block.contains("step 1"));
        assert!(block.contains("step 2"));
    }

    #[test]
    fn render_thinking_block_escapes_html() {
        let v = view_with_response_and_thinking("ok", "<script>alert(1)</script>");
        let block = v.render_thinking_block().unwrap();
        assert!(
            !block.contains("<script>"),
            "raw HTML inside reasoning must be escaped — telegram parser \
             would otherwise reject the message or, worse, the model could \
             smuggle markup that breaks our blockquote envelope. got: {block}"
        );
        assert!(block.contains("&lt;script&gt;"));
    }

    #[test]
    fn render_thinking_block_tail_truncates_long_chain() {
        // Use a chain comfortably longer than MAX_FINAL_THINKING_BYTES so
        // we exercise the cap. We expect the *prefix* to be dropped: the
        // commitment / conclusion in a CoT lives at the bottom.
        let prefix = "PREFIX_THAT_SHOULD_BE_DROPPED ".repeat(200);
        let suffix = "FINAL_DECISION";
        let mut chain = String::new();
        chain.push_str(&prefix);
        chain.push_str(suffix);
        assert!(chain.len() > MAX_FINAL_THINKING_BYTES);

        let v = view_with_response_and_thinking("done", &chain);
        let block = v.render_thinking_block().unwrap();
        assert!(
            block.contains(suffix),
            "tail must survive truncation — that's where the conclusion is"
        );
        assert!(
            block.contains('…'),
            "truncated chain must announce itself with an ellipsis"
        );
    }

    #[test]
    fn render_final_includes_thinking_block_when_present() {
        let v = view_with_response_and_thinking("hello world", "let me think");
        let out = v.render_final();
        assert!(out.contains("hello world"));
        assert!(out.contains("<blockquote expandable>"));
        assert!(out.contains("let me think"));
    }

    #[test]
    fn render_final_handles_thinking_only_no_text() {
        let v = view_with_response_and_thinking("", "i was thinking but said nothing");
        let out = v.render_final();
        assert_ne!(out, "(empty response)");
        assert!(out.contains("(no text — reasoning only)"));
        assert!(out.contains("i was thinking but said nothing"));
    }

    #[test]
    fn render_final_empty_returns_helpful_fallback() {
        // When the provider closes the turn with zero text AND zero
        // reasoning the user used to see a cryptic "(empty response)".
        // Now they get an actionable hint that points at `/new`.
        let v = view_with_response_and_thinking("", "");
        let out = v.render_final();
        assert!(!out.contains("(empty response)"));
        assert!(
            out.contains("/new"),
            "empty-turn fallback must mention /new recovery, got: {out}"
        );
    }

    #[test]
    fn render_final_fits_single_telegram_message_even_with_huge_thinking() {
        // Regression for the "echo-thinking leak": if text + thinking
        // exceed MAX_TG_MSG, send_final would split into two TG messages
        // and the second one was almost pure reasoning. Now the thinking
        // block must shrink so the whole final render fits in one
        // message.
        let response = "Here is the final answer. ".repeat(50); // ~1.3 kB
        let thinking = "intermediate reasoning chunk. ".repeat(400); // ~12 kB
        let v = view_with_response_and_thinking(&response, &thinking);
        let out = v.render_final();
        assert!(
            out.len() <= MAX_TG_MSG,
            "render_final must fit in one TG message ({} <= {}), got len={}",
            out.len(),
            MAX_TG_MSG,
            out.len()
        );
        // The primary answer must be preserved verbatim — it's what
        // the user actually wants. Only thinking may be squeezed.
        assert!(out.contains("Here is the final answer."));
    }

    #[test]
    fn render_final_drops_thinking_entirely_when_text_already_full() {
        // Extreme case: response alone nearly fills the message. The
        // thinking block must be dropped outright instead of spilling
        // into a second message.
        let response = "A".repeat(MAX_TG_MSG - 250);
        let thinking = "thought. ".repeat(500);
        let v = view_with_response_and_thinking(&response, &thinking);
        let out = v.render_final();
        assert!(out.len() <= MAX_TG_MSG);
        assert!(
            !out.contains("<blockquote expandable>"),
            "thinking must be dropped (not half-rendered) when budget is tight"
        );
    }

    // ── is_allowed ──────────────────────────────────────────────────────

    #[test]
    fn is_allowed_empty_list_denies_all() {
        let config = Config {
            allowed_chat_ids: vec![],
            ..Config::default()
        };
        assert!(!is_allowed(123, &config));
        assert!(!is_allowed(0, &config));
    }

    #[test]
    fn is_allowed_with_ids_checks_membership() {
        let config = Config {
            allowed_chat_ids: vec![100, 200],
            ..Config::default()
        };
        assert!(is_allowed(100, &config));
        assert!(is_allowed(200, &config));
        assert!(!is_allowed(300, &config));
    }

    // ── escape_html ─────────────────────────────────────────────────────

    #[test]
    fn escape_html_special_chars() {
        assert_eq!(escape_html("<b>hi</b>"), "&lt;b&gt;hi&lt;/b&gt;");
        assert_eq!(escape_html("a & b"), "a &amp; b");
        assert_eq!(escape_html("plain"), "plain");
    }

    // ── truncate_str ────────────────────────────────────────────────────

    #[test]
    fn truncate_str_short_unchanged() {
        assert_eq!(truncate_str("hello", 10), "hello");
    }

    #[test]
    fn truncate_str_exact_length() {
        assert_eq!(truncate_str("hello", 5), "hello");
    }

    #[test]
    fn truncate_str_adds_ellipsis() {
        let result = truncate_str("hello world", 5);
        assert!(result.ends_with('…'));
        assert!(result.len() < "hello world".len() + 3);
    }

    #[test]
    fn truncate_str_multibyte_safe() {
        let s = "日本語テスト";
        let result = truncate_str(s, 3);
        assert!(result.ends_with('…'));
        assert!(result.starts_with("日本"));
    }

    // ── split_html ──────────────────────────────────────────────────────

    #[test]
    fn split_html_short_returns_single() {
        let chunks = split_html("hello", 100);
        assert_eq!(chunks, vec!["hello"]);
    }

    #[test]
    fn split_html_splits_on_newline() {
        let text = "line1\nline2\nline3\nline4\nline5";
        let chunks = split_html(text, 12);
        assert!(chunks.len() > 1);
        let joined: String = chunks.concat();
        assert_eq!(joined, text);
    }

    #[test]
    fn split_html_respects_char_boundaries() {
        let text = "Привет мир, это тест юникода";
        let chunks = split_html(text, 10);
        assert!(chunks.len() > 1);
        let joined: String = chunks.concat();
        assert_eq!(joined, text);
    }

    // ── format_input_preview ────────────────────────────────────────────

    #[test]
    fn format_input_preview_single_key() {
        let input = serde_json::json!({"command": "ls -la"});
        let result = format_input_preview(&input, 100);
        assert!(result.contains("command"));
        assert!(result.contains("ls -la"));
    }

    #[test]
    fn format_input_preview_multi_key() {
        let input = serde_json::json!({"file": "test.rs", "content": "fn main()"});
        let result = format_input_preview(&input, 200);
        assert!(result.contains("file"));
        assert!(result.contains("content"));
    }

    #[test]
    fn format_input_preview_truncates() {
        let long_val = "x".repeat(500);
        let input = serde_json::json!({"data": long_val});
        let result = format_input_preview(&input, 50);
        assert!(result.len() < 200);
    }

    // ── md_to_tg_html ────────────────────────────────────────────────────

    #[test]
    fn md_bold_italic() {
        assert!(md_to_tg_html("**hello**").contains("<b>hello</b>"));
        assert!(md_to_tg_html("*world*").contains("<i>world</i>"));
    }

    #[test]
    fn md_inline_code() {
        assert!(md_to_tg_html("`code`").contains("<code>code</code>"));
    }

    #[test]
    fn md_code_block() {
        let input = "before\n```rust\nfn main() {}\n```\nafter";
        let result = md_to_tg_html(input);
        assert!(result.contains("<pre>"));
        assert!(result.contains("fn main()"));
        assert!(result.contains("</pre>"));
    }

    #[test]
    fn md_headers() {
        assert!(md_to_tg_html("# Big").contains("<b>Big</b>"));
        assert!(md_to_tg_html("## Medium").contains("<b>Medium</b>"));
        assert!(md_to_tg_html("### Small").contains("<b>Small</b>"));
    }

    #[test]
    fn md_link() {
        let result = md_to_tg_html("[click](https://example.com)");
        assert!(result.contains("<a href=\"https://example.com\">click</a>"));
    }

    #[test]
    fn md_table_to_text() {
        let input = "| Name | Score |\n|---|---|\n| Alice | 100 |";
        let result = md_to_tg_html(input);
        // Table is rendered in <pre> monospace — outer pipes stripped, inner kept
        assert!(result.contains("<pre>"));
        assert!(result.contains("Alice"));
        assert!(result.contains("Score"));
        // Separator row (---|---) should be removed
        assert!(!result.contains("---"));
    }

    #[test]
    fn md_hr_stripped() {
        let result = md_to_tg_html("above\n---\nbelow");
        assert!(!result.contains("---"));
        assert!(result.contains("above"));
        assert!(result.contains("below"));
    }

    #[test]
    fn md_escapes_html_entities() {
        let result = md_to_tg_html("a < b & c > d");
        assert!(result.contains("&lt;"));
        assert!(result.contains("&amp;"));
        assert!(result.contains("&gt;"));
    }

    // ── sender_label / is_group_chat / extract_reply_context ───────────
    //
    // We build `Message` fixtures by parsing raw Telegram API JSON — this
    // is the same path the dispatcher takes, and it avoids depending on
    // teloxide private constructors.

    fn make_message(v: serde_json::Value) -> Message {
        serde_json::from_value(v).expect("valid Message JSON")
    }

    fn base_private_chat() -> serde_json::Value {
        serde_json::json!({
            "message_id": 1,
            "date": 1_700_000_000,
            "chat": { "id": 42, "type": "private", "first_name": "Alice" },
        })
    }

    fn base_group_chat() -> serde_json::Value {
        serde_json::json!({
            "message_id": 1,
            "date": 1_700_000_000,
            "chat": { "id": -1001, "type": "supergroup", "title": "team" },
        })
    }

    fn user(username: Option<&str>, first: &str, is_bot: bool) -> serde_json::Value {
        let mut u = serde_json::json!({
            "id": 7,
            "is_bot": is_bot,
            "first_name": first,
        });
        if let Some(n) = username {
            u["username"] = serde_json::Value::String(n.to_string());
        }
        u
    }

    #[test]
    fn sender_label_prefers_username_with_at() {
        let mut m = base_private_chat();
        m["from"] = user(Some("alice"), "Alice", false);
        m["text"] = serde_json::Value::String("hi".into());
        assert_eq!(sender_label(&make_message(m)), "@alice");
    }

    #[test]
    fn sender_label_falls_back_to_first_name() {
        let mut m = base_private_chat();
        m["from"] = user(None, "Bob", false);
        m["text"] = serde_json::Value::String("hi".into());
        assert_eq!(sender_label(&make_message(m)), "Bob");
    }

    #[test]
    fn sender_label_unknown_when_no_sender() {
        let mut m = base_private_chat();
        m["text"] = serde_json::Value::String("hi".into());
        assert_eq!(sender_label(&make_message(m)), "unknown");
    }

    #[test]
    fn is_group_chat_true_for_supergroup() {
        let mut m = base_group_chat();
        m["text"] = serde_json::Value::String("hi".into());
        assert!(is_group_chat(&make_message(m)));
    }

    #[test]
    fn is_group_chat_false_for_private() {
        let mut m = base_private_chat();
        m["text"] = serde_json::Value::String("hi".into());
        assert!(!is_group_chat(&make_message(m)));
    }

    #[test]
    fn extract_reply_context_none_when_not_a_reply() {
        let mut m = base_private_chat();
        m["from"] = user(Some("alice"), "Alice", false);
        m["text"] = serde_json::Value::String("hi".into());
        assert!(extract_reply_context(&make_message(m)).is_none());
    }

    #[test]
    fn extract_reply_context_formats_text_reply_with_username() {
        let reply = serde_json::json!({
            "message_id": 10,
            "date": 1_699_999_900,
            "chat": { "id": 42, "type": "private", "first_name": "Alice" },
            "from": user(Some("bob"), "Bob", false),
            "text": "line1\nline2",
        });
        let mut m = base_private_chat();
        m["from"] = user(Some("alice"), "Alice", false);
        m["text"] = serde_json::Value::String("ok".into());
        m["reply_to_message"] = reply;

        let q = extract_reply_context(&make_message(m)).expect("some");
        assert!(q.starts_with("> @bob:\n"), "got: {q}");
        assert!(q.contains("> line1"));
        assert!(q.contains("> line2"));
    }

    #[test]
    fn extract_reply_context_marks_bot_previous_message() {
        let reply = serde_json::json!({
            "message_id": 10,
            "date": 1_699_999_900,
            "chat": { "id": 42, "type": "private", "first_name": "Alice" },
            "from": user(Some("naked_bot"), "naked", true),
            "text": "done.",
        });
        let mut m = base_private_chat();
        m["text"] = serde_json::Value::String("ok".into());
        m["reply_to_message"] = reply;

        let q = extract_reply_context(&make_message(m)).unwrap();
        assert!(q.contains("[your previous message]"), "got: {q}");
    }

    #[test]
    fn extract_reply_context_photo_with_caption() {
        let reply = serde_json::json!({
            "message_id": 10,
            "date": 1_699_999_900,
            "chat": { "id": 42, "type": "private", "first_name": "Alice" },
            "from": user(Some("bob"), "Bob", false),
            "photo": [
                {"file_id":"abc","file_unique_id":"u","width":10,"height":10}
            ],
            "caption": "ship it",
        });
        let mut m = base_private_chat();
        m["text"] = serde_json::Value::String("yep".into());
        m["reply_to_message"] = reply;

        let q = extract_reply_context(&make_message(m)).unwrap();
        assert!(q.contains("[Photo: ship it]"), "got: {q}");
    }

    #[test]
    fn extract_reply_context_photo_without_caption() {
        let reply = serde_json::json!({
            "message_id": 10,
            "date": 1_699_999_900,
            "chat": { "id": 42, "type": "private", "first_name": "Alice" },
            "from": user(Some("bob"), "Bob", false),
            "photo": [
                {"file_id":"abc","file_unique_id":"u","width":10,"height":10}
            ],
        });
        let mut m = base_private_chat();
        m["text"] = serde_json::Value::String("yep".into());
        m["reply_to_message"] = reply;

        let q = extract_reply_context(&make_message(m)).unwrap();
        assert!(q.contains("[Photo]"));
        assert!(!q.contains("[Photo:"));
    }

    // ── extract_media_items / fmt_duration ─────────────────────────────

    #[test]
    fn fmt_duration_formats_mm_ss() {
        assert_eq!(fmt_duration(0), "00:00");
        assert_eq!(fmt_duration(9), "00:09");
        assert_eq!(fmt_duration(65), "01:05");
        assert_eq!(fmt_duration(3599), "59:59");
    }

    #[test]
    fn fmt_duration_caps_long_values() {
        // Anything past 99:59 is clamped.
        assert_eq!(fmt_duration(60 * 99 + 59), "99:59");
        assert_eq!(fmt_duration(60 * 200), "99:59");
    }

    #[test]
    fn extract_media_items_voice() {
        let mut m = base_private_chat();
        m["from"] = user(Some("alice"), "Alice", false);
        m["voice"] = serde_json::json!({
            "file_id": "voice-abc",
            "file_unique_id": "u",
            "duration": 12,
            "mime_type": "audio/ogg"
        });
        let items = extract_media_items(&make_message(m));
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].kind, media::MediaKind::Voice);
        assert_eq!(items[0].file_id, "voice-abc");
        assert!(items[0].file_name.ends_with(".ogg"));
        assert_eq!(items[0].duration_secs, Some(12));
    }

    #[test]
    fn extract_media_items_photo_picks_highest_resolution() {
        let mut m = base_private_chat();
        m["from"] = user(Some("alice"), "Alice", false);
        m["photo"] = serde_json::json!([
            {"file_id":"small","file_unique_id":"s","width":90,"height":90,"file_size":1000},
            {"file_id":"big","file_unique_id":"b","width":1280,"height":720,"file_size":200000}
        ]);
        let items = extract_media_items(&make_message(m));
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].kind, media::MediaKind::Photo);
        assert_eq!(items[0].file_id, "big");
        assert_eq!(items[0].mime_hint.as_deref(), Some("image/jpeg"));
    }

    #[test]
    fn extract_media_items_document_with_name() {
        let mut m = base_private_chat();
        m["from"] = user(Some("alice"), "Alice", false);
        m["document"] = serde_json::json!({
            "file_id": "doc-1",
            "file_unique_id": "u",
            "file_name": "notes.md",
            "mime_type": "text/markdown"
        });
        let items = extract_media_items(&make_message(m));
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].kind, media::MediaKind::Document);
        assert_eq!(items[0].file_name, "notes.md");
        assert_eq!(items[0].mime_hint.as_deref(), Some("text/markdown"));
    }

    #[test]
    fn extract_media_items_none_for_plain_text() {
        let mut m = base_private_chat();
        m["from"] = user(Some("alice"), "Alice", false);
        m["text"] = serde_json::Value::String("just text".into());
        let items = extract_media_items(&make_message(m));
        assert!(items.is_empty());
    }

    #[test]
    fn extract_media_items_pulls_photo_from_reply_target() {
        // The user replies to an old photo with a textual question; we
        // need the photo bytes for the current turn, so handle_message
        // forwards `extract_media_items(reply_to_message)` into the
        // pipeline. Verify the helper itself does the right thing on a
        // reply-target Message: it reads media off whichever Message
        // shape it's handed, so passing the reply target Just Works.
        // (handle_message-side wiring is exercised by the live e2e
        // test; here we pin down the building block.)
        let reply_target = serde_json::json!({
            "message_id": 99,
            "date": 0,
            "chat": { "id": -5084292206_i64, "type": "group", "title": "income" },
            "from": { "id": 105928336, "is_bot": false, "first_name": "u" },
            "photo": [
                {"file_id":"reply-small","file_unique_id":"a","width":90,"height":90,"file_size":1000},
                {"file_id":"reply-big","file_unique_id":"b","width":1280,"height":720,"file_size":200000}
            ]
        });
        let items = extract_media_items(&make_message(reply_target));
        assert_eq!(items.len(), 1, "should extract the single photo");
        assert_eq!(items[0].kind, media::MediaKind::Photo);
        assert_eq!(
            items[0].file_id, "reply-big",
            "should pick the highest-resolution PhotoSize for vision routing"
        );
    }

    // ── Native multimodal routing ───────────────────────────────────────
    //
    // These tests exercise the *decision* path (`is_vision_capable_model` +
    // `native_image_context` + `native_image_max_bytes`). The actual byte-to-
    // base64 conversion happens in `handle_message` and is covered by the
    // live e2e test `live_native_image_roundtrip_via_groq`.

    #[test]
    fn vision_routing_off_when_native_image_context_disabled() {
        use naked_core::config::TgMediaConfig;
        let cfg = TgMediaConfig {
            native_image_context: false,
            ..TgMediaConfig::default()
        };
        // Even a Claude 3 model goes through the legacy text-only path.
        let route =
            cfg.native_image_context && cfg.is_vision_capable_model("claude-sonnet-4-20250514");
        assert!(!route);
    }

    #[test]
    fn vision_routing_on_for_capable_model() {
        use naked_core::config::TgMediaConfig;
        let cfg = TgMediaConfig::default();
        for model in [
            "claude-sonnet-4-20250514",
            "claude-haiku-4-5-20251001",
            "gpt-4o",
            "gpt-4o-mini",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "grok-2-vision-latest",
        ] {
            let route = cfg.native_image_context && cfg.is_vision_capable_model(model);
            assert!(route, "{model} should route natively");
        }
    }

    #[test]
    fn looks_like_supported_image_accepts_known_formats() {
        // Real magic headers (header bytes only — body is irrelevant).
        let jpeg: Vec<u8> = [&[0xFF, 0xD8, 0xFFu8] as &[u8], &[0u8; 16]].concat();
        let png: Vec<u8> = [
            &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A] as &[u8],
            &[0u8; 16],
        ]
        .concat();
        let gif87 = b"GIF87a\0\0\0\0\0\0".to_vec();
        let gif89 = b"GIF89a\0\0\0\0\0\0".to_vec();
        let mut webp = b"RIFF".to_vec();
        webp.extend_from_slice(&[0u8; 4]);
        webp.extend_from_slice(b"WEBP");
        webp.extend_from_slice(&[0u8; 4]);
        for (name, payload) in [
            ("jpeg", jpeg),
            ("png", png),
            ("gif87", gif87),
            ("gif89", gif89),
            ("webp", webp),
        ] {
            assert!(
                looks_like_supported_image(&payload),
                "{name} magic header must be recognised"
            );
        }
    }

    #[test]
    fn looks_like_supported_image_rejects_garbage_and_short_payloads() {
        // Empty / too short / random bytes / repurposed text.
        assert!(!looks_like_supported_image(&[]));
        assert!(!looks_like_supported_image(&[0xFF, 0xD8])); // truncated jpeg
        assert!(!looks_like_supported_image(b"hello world"));
        assert!(!looks_like_supported_image(b"<?xml version=1.0?>"));
        // RIFF without WEBP marker (e.g. WAV) must not be claimed as image.
        let mut wav = b"RIFF".to_vec();
        wav.extend_from_slice(&[0u8; 4]);
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(&[0u8; 4]);
        assert!(!looks_like_supported_image(&wav));
    }

    #[test]
    fn vision_routing_off_for_text_only_model() {
        use naked_core::config::TgMediaConfig;
        let cfg = TgMediaConfig::default();
        for model in [
            "llama-3.3-70b-versatile",
            "glm-5-turbo",
            "MiniMax-Text-01",
            "deepseek-chat",
        ] {
            let route = cfg.native_image_context && cfg.is_vision_capable_model(model);
            assert!(!route, "{model} should NOT route natively");
        }
    }

    fn item_photo(size_hint: Option<u32>) -> MediaItem {
        MediaItem {
            kind: media::MediaKind::Photo,
            file_id: "f".into(),
            file_name: "x.jpg".into(),
            mime_hint: Some("image/jpeg".into()),
            duration_secs: None,
            emoji: None,
            size_hint,
            sticker_format: None,
        }
    }

    #[test]
    fn decide_native_route_off_when_caller_disabled() {
        let item = item_photo(Some(10_000));
        assert!(!decide_native_route(&item, false, u32::MAX));
    }

    #[test]
    fn decide_native_route_on_for_photo_under_cap() {
        let item = item_photo(Some(10_000));
        assert!(decide_native_route(&item, true, 1_000_000));
    }

    #[test]
    fn decide_native_route_on_for_photo_with_no_size_hint() {
        // Telegram sometimes omits `file.size` for cached PhotoSize entries —
        // we should let the download proceed natively rather than degrading
        // pre-emptively. The post-download cap in `process_one_media` will
        // still catch oversized images.
        let item = item_photo(None);
        assert!(decide_native_route(&item, true, 5 * 1024 * 1024));
    }

    #[test]
    fn decide_native_route_off_when_size_exceeds_cap() {
        // Pre-download fallback path: size_hint > native_image_max_bytes must
        // force the legacy describer route so we don't waste bandwidth nor
        // get rejected by the provider for "image too large".
        let item = item_photo(Some(20 * 1024 * 1024));
        assert!(!decide_native_route(&item, true, 5 * 1024 * 1024));
    }

    #[test]
    fn decide_native_route_off_for_animated_sticker() {
        let mut item = item_photo(Some(10_000));
        item.kind = media::MediaKind::Sticker;
        item.sticker_format = Some(StickerFormat::Animated);
        assert!(!decide_native_route(&item, true, u32::MAX));
    }

    #[test]
    fn decide_native_route_off_for_video_sticker() {
        let mut item = item_photo(Some(10_000));
        item.kind = media::MediaKind::Sticker;
        item.sticker_format = Some(StickerFormat::Video);
        assert!(!decide_native_route(&item, true, u32::MAX));
    }

    #[test]
    fn decide_native_route_on_for_static_sticker() {
        let mut item = item_photo(Some(10_000));
        item.kind = media::MediaKind::Sticker;
        item.sticker_format = Some(StickerFormat::Static);
        item.mime_hint = Some("image/webp".into());
        assert!(decide_native_route(&item, true, u32::MAX));
    }

    #[test]
    fn decide_native_route_passthrough_for_non_image_media() {
        // Audio/video/file media are not gated by the photo-specific predicate;
        // the caller's flag wins for them. (`process_one_media` then routes
        // them through audio transcription / file artifact paths.)
        let item = MediaItem {
            kind: media::MediaKind::Voice,
            file_id: "v".into(),
            file_name: "v.ogg".into(),
            mime_hint: Some("audio/ogg".into()),
            duration_secs: Some(5),
            emoji: None,
            size_hint: Some(50_000_000), // intentionally huge
            sticker_format: None,
        };
        assert!(decide_native_route(&item, true, 1));
        assert!(!decide_native_route(&item, false, u32::MAX));
    }

    #[test]
    fn media_processed_default_is_empty() {
        let mp = MediaProcessed::default();
        assert!(mp.text.is_empty());
        assert!(mp.native_images.is_empty());
    }

    #[test]
    fn native_image_struct_carries_mime_and_bytes() {
        let img = NativeImage {
            mime: "image/png".into(),
            bytes: vec![0x89, 0x50, 0x4E, 0x47],
        };
        assert_eq!(img.mime, "image/png");
        assert_eq!(img.bytes.len(), 4);
        let cloned = img.clone();
        assert_eq!(cloned.bytes, img.bytes);
    }

    #[test]
    fn extract_media_items_sticker_carries_emoji() {
        let mut m = base_private_chat();
        m["from"] = user(Some("alice"), "Alice", false);
        m["sticker"] = serde_json::json!({
            "file_id": "stk-1",
            "file_unique_id": "u",
            "width": 512,
            "height": 512,
            "type": "regular",
            "is_animated": false,
            "is_video": false,
            "emoji": "\u{1F525}"
        });
        let items = extract_media_items(&make_message(m));
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].kind, media::MediaKind::Sticker);
        assert_eq!(items[0].emoji.as_deref(), Some("\u{1F525}"));
    }

    #[test]
    fn extract_static_sticker_routes_natively() {
        let mut m = base_private_chat();
        m["from"] = user(Some("alice"), "Alice", false);
        m["sticker"] = serde_json::json!({
            "file_id": "stk-static",
            "file_unique_id": "u",
            "width": 512, "height": 512,
            "type": "regular",
            "is_animated": false, "is_video": false,
            "file_size": 32_000,
        });
        let items = extract_media_items(&make_message(m));
        assert_eq!(items[0].sticker_format, Some(StickerFormat::Static));
        assert_eq!(items[0].mime_hint.as_deref(), Some("image/webp"));
        assert!(items[0].file_name.ends_with(".webp"));
        assert_eq!(items[0].size_hint, Some(32_000));
    }

    #[test]
    fn extract_animated_sticker_marked_non_native() {
        let mut m = base_private_chat();
        m["from"] = user(Some("alice"), "Alice", false);
        m["sticker"] = serde_json::json!({
            "file_id": "stk-anim",
            "file_unique_id": "u",
            "width": 512, "height": 512,
            "type": "regular",
            "is_animated": true, "is_video": false,
            "file_size": 12_000,
        });
        let items = extract_media_items(&make_message(m));
        assert_eq!(items[0].sticker_format, Some(StickerFormat::Animated));
        assert!(items[0].file_name.ends_with(".tgs"));
        assert_eq!(
            items[0].mime_hint.as_deref(),
            Some("application/x-tgsticker"),
            "animated stickers must NOT be advertised as image/* — vision providers will reject them"
        );
    }

    #[test]
    fn extract_video_sticker_marked_non_native() {
        let mut m = base_private_chat();
        m["from"] = user(Some("alice"), "Alice", false);
        m["sticker"] = serde_json::json!({
            "file_id": "stk-vid",
            "file_unique_id": "u",
            "width": 512, "height": 512,
            "type": "regular",
            "is_animated": false, "is_video": true,
            "file_size": 80_000,
        });
        let items = extract_media_items(&make_message(m));
        assert_eq!(items[0].sticker_format, Some(StickerFormat::Video));
        assert!(items[0].file_name.ends_with(".webm"));
        assert_eq!(items[0].mime_hint.as_deref(), Some("video/webm"));
    }

    #[test]
    fn extract_photo_carries_size_hint() {
        let mut m = base_private_chat();
        m["from"] = user(Some("alice"), "Alice", false);
        m["photo"] = serde_json::json!([
            { "file_id": "p1", "file_unique_id": "u1", "width": 90,  "height": 60,  "file_size": 4_000 },
            { "file_id": "p2", "file_unique_id": "u2", "width": 320, "height": 240, "file_size": 32_000 },
            { "file_id": "p3", "file_unique_id": "u3", "width": 800, "height": 600, "file_size": 200_000 },
        ]);
        let items = extract_media_items(&make_message(m));
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].kind, media::MediaKind::Photo);
        // We pick the largest resolution → its size_hint should be 200_000.
        assert_eq!(items[0].size_hint, Some(200_000));
    }

    // ── TG HTTP mock (wiremock) ─────────────────────────────────────────
    //
    // These tests stand up a local HTTP server that pretends to be the
    // Telegram Bot API and verify our outgoing `sendMessage` plumbing
    // talks to it correctly. They exist to catch regressions in the
    // low-level `Bot`/`reqwest` layer — higher-level dispatch logic is
    // covered by the `album::tests` module.

    use wiremock::matchers::method;
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// Build a `Bot` pointing at the provided mock URL instead of
    /// `api.telegram.org`. Token is a throwaway.
    fn mock_bot(mock_url: &str) -> Bot {
        let url = reqwest::Url::parse(mock_url).unwrap();
        Bot::new("0:TEST_TOKEN").set_api_url(url)
    }

    #[tokio::test]
    async fn send_text_hits_mock_server_with_sendmessage() {
        let server = MockServer::start().await;
        // Match every POST. teloxide's URL shape is cosmetic for a mock —
        // what we care about is "the request reached the HTTP server".
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "ok": true,
                "result": {
                    "message_id": 999,
                    "date": 0,
                    "chat": {"id": 1, "type": "private", "first_name": "x"},
                    "text": "ack"
                }
            })))
            .expect(1)
            .mount(&server)
            .await;

        let bot = mock_bot(&server.uri());
        let result = send_text(&bot, ChatId(1), None, "hello").await;
        // The test is about reaching the mock, not round-tripping the
        // full Message. Some teloxide versions are strict about the
        // serialized response shape; tolerate either Ok or a
        // deserialisation error as long as the request was sent.
        let _ = result;

        // `expect(1)` on Drop: wiremock panics if the mock wasn't hit
        // exactly once. Belt-and-braces: explicitly count received reqs.
        let received = server.received_requests().await.unwrap();
        assert_eq!(
            received.len(),
            1,
            "expected exactly one TG API request, got {}",
            received.len()
        );
    }

    #[tokio::test]
    async fn send_text_serialises_chat_id_and_text() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "ok": true,
                "result": {
                    "message_id": 12,
                    "date": 0,
                    "chat": {"id": 777, "type": "private", "first_name": "x"},
                    "text": "ok"
                }
            })))
            .mount(&server)
            .await;

        let bot = mock_bot(&server.uri());
        let _ = send_text(&bot, ChatId(777), None, "Привет мир 🌍").await;

        let received = server.received_requests().await.unwrap();
        assert_eq!(received.len(), 1);
        let body = std::str::from_utf8(&received[0].body).unwrap();
        assert!(
            body.contains("777"),
            "chat_id must appear in POST body: {body}"
        );
        // Non-ASCII payload must pass through unmangled (url-encoded or
        // JSON-escaped both count — we just need to see the logical text).
        assert!(
            body.contains("%D0%9F%D1%80%D0%B8") || body.contains("Привет"),
            "cyrillic/emoji must survive serialisation: {body}"
        );
    }
}
