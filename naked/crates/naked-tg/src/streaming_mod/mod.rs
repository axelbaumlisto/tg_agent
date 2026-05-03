//! Streaming response handler + CompositeView.

mod handlers;
mod helpers;
pub(crate) use handlers::ViewAction;
pub(crate) use helpers::*;

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
    /// Set when a provider error occurs — used to show retry buttons after final.
    pub(crate) had_provider_error: bool,
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
            had_provider_error: false,
        }
    }

    fn elapsed_label(&self) -> String {
        let total = self.started_at.elapsed().as_secs();
        let m = total / 60;
        let s = total % 60;
        if m == 0 {
            format!("🕐 {s}s")
        } else {
            format!("🕐 {m}:{s:02}")
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

        self.render_sub_agent_lines(&mut parts);

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

                // Render stable (closed) blocks as rich HTML.
                // Strip class="language-*" — Telegram editMessageText
                // rejects attributes on <code> tags during streaming.
                if !stable.is_empty() {
                    let html = strip_code_class(&md_to_tg_html(stable));
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
    fn render_sub_agent_lines(&self, parts: &mut Vec<String>) {
        if self.sub_agents.is_empty() {
            return;
        }
        let last_ids: Vec<_> = self
            .sub_agent_order
            .iter()
            .rev()
            .take(3)
            .rev()
            .cloned()
            .collect();
        for id in &last_ids {
            let Some(sa) = self.sub_agents.get(id) else {
                continue;
            };
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
/// Apply a sub-agent progress event to the composite view.
fn apply_sub_agent_event(
    view: &mut CompositeView,
    agent_id: String,
    event: naked_core::types::SubAgentEvent,
) {
    use naked_core::types::SubAgentEvent;
    match event {
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
        SubAgentEvent::ToolDone { .. } | SubAgentEvent::TextDelta(_) => {}
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
        steer,
    } = handle;
    let chat_id_raw = ctx.chat_id.0;
    let tid = ctx.raw_thread_id();
    let chat_key_for_steer = (chat_id_raw, tid);

    // Store steer sender so message_handler can reach us.
    STEER_SENDERS
        .write()
        .await
        .insert(chat_key_for_steer, steer);

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
    let mut dirty = false;
    let mut last_sent = String::new();
    let mut html_broken = false;
    let mut aborted_for_switch = false;
    let mut last_event_at = tokio::time::Instant::now();
    let mut stall_warned = false;
    const STALL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(90);

    // Fixed-interval ticker for streaming flushes.
    // The actual rate limiting happens inside RATE_LIMITER.edit() — the
    // ticker just decides when to ATTEMPT a flush.
    let mut flush_interval = tokio::time::interval(std::time::Duration::from_millis(2_400));
    flush_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    flush_interval.tick().await; // consume first immediate tick

    loop {
        let event = tokio::select! {
            ev = events.recv() => match ev {
                Some(e) => Some(e),
                None => break,
            },
            _ = flush_interval.tick() => None,
        };

        // FIX-3: Detect stalled agent (no events for 90s).
        if event.is_some() {
            last_event_at = tokio::time::Instant::now();
            stall_warned = false;
        } else if !stall_warned && last_event_at.elapsed() > STALL_TIMEOUT {
            stall_warned = true;
            view.tool_lines
                .push("⏳ Бот не отвечает >90s — возможно завис. /stop для отмены.".to_string());
            dirty = true;
        }

        let has_event = event.is_some();
        if let Some(event) = event {
            match event {
                AgentEvent::ThinkingDelta(t) => {
                    dirty = handlers::handle_thinking_delta(&mut view, &t) != ViewAction::Clean;
                }
                AgentEvent::TextDelta(t) => {
                    dirty = handlers::handle_text_delta(&mut view, &t) != ViewAction::Clean;
                }
                AgentEvent::ToolStart { name, input, .. } => {
                    let _ = handlers::handle_tool_start(&mut view, &name, &input);
                    dirty = true;
                }
                AgentEvent::ToolEnd {
                    name,
                    state,
                    output,
                    ..
                } => {
                    let is_error = matches!(state, naked_core::types::ToolState::Error);
                    let (_, detail_msg) =
                        handlers::handle_tool_end(&mut view, &name, is_error, &output);
                    if let Some(msg) = detail_msg {
                        let _ = bot
                            .send_message(ctx.chat_id, &msg)
                            .maybe_thread(ctx.thread_id)
                            .parse_mode(teloxide::types::ParseMode::Html)
                            .await;
                    }
                    dirty = true;
                }
                AgentEvent::PermissionRequest {
                    call_id,
                    tool_name,
                    input,
                    permission,
                } => {
                    // Flush before showing permission dialog.
                    let _ = flush_live(
                        &bot,
                        ctx.chat_id,
                        placeholder,
                        &view,
                        &mut last_sent,
                        &mut html_broken,
                    )
                    .await;
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
                    summary_hint,
                    files_count,
                } => {
                    let note = handlers::handle_compaction(
                        before_msgs,
                        after_msgs,
                        files_count,
                        summary_hint.as_deref(),
                    );
                    let _ = bot
                        .send_message(ctx.chat_id, &note)
                        .maybe_thread(ctx.thread_id)
                        .maybe_reply_to(ctx.reply_to)
                        .parse_mode(teloxide::types::ParseMode::Html)
                        .await;
                }
                AgentEvent::Heartbeat => {
                    handlers::handle_heartbeat(&mut view);
                    dirty = true;
                }
                AgentEvent::SubAgentProgress {
                    agent_id,
                    event: sa_ev,
                } => {
                    let _ = handlers::handle_sub_agent(&mut view, agent_id, sa_ev);
                    dirty = true;
                }
                AgentEvent::UsageUpdate(u) => {
                    handlers::handle_usage(&mut view, u);
                    dirty = true;
                }
                AgentEvent::SteerReceived { text } => {
                    view.tool_lines.push(format!(
                        "\u{21a9}\u{fe0f} <i>Steer: {}</i>",
                        crate::fmt_utils::escape_html_min(&text)
                    ));
                    dirty = true;
                }
                AgentEvent::Error(e) => {
                    let _ = handlers::handle_error(&mut view, &e);
                    dirty = true;
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

        // Proactive flush: only on tick interval, never faster.
        // The interval ticker fires in select! above → event=None.
        // On event: just set dirty. On tick: flush if dirty.
        let is_tick = !has_event; // tick = no event received, interval fired
        if is_tick {
            // Tick fired — time to flush.
            view.tick += 1;
            if dirty {
                flush_live(
                    &bot,
                    ctx.chat_id,
                    placeholder,
                    &view,
                    &mut last_sent,
                    &mut html_broken,
                )
                .await;
                dirty = false;

                // If the rate limiter has this chat blocked (429),
                // stretch the ticker to avoid hammering.
                let backoff = rate_limiter.streaming_interval(ctx.chat_id.0).await;
                if backoff > flush_interval.period() {
                    flush_interval = tokio::time::interval(backoff);
                    flush_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                    flush_interval.tick().await;
                }
            }
        }
    }

    // No unregister needed — the global limiter tracks per-chat state.

    typing_cancel.cancel();

    // Deregister model-switch state, queue counter, and steer sender.
    MODEL_SWITCHES.write().await.remove(&chat_key);
    QUEUE_COUNTS.write().await.remove(&chat_key);
    STEER_SENDERS.write().await.remove(&chat_key_for_steer);

    if aborted_for_switch {
        // Don't send final — the turn was interrupted.
        // Edit placeholder to indicate switch in progress.
        RATE_LIMITER
            .edit_plain(&bot, ctx.chat_id, placeholder, "⚡ Switching model…")
            .await;
        return;
    }

    let final_html = view.render_final();
    send_final(bot.clone(), ctx, placeholder, &final_html, &view).await;

    // ── A3: Send error card with retry button if provider failed ────
    if view.had_provider_error {
        let keyboard = InlineKeyboardMarkup::new(vec![vec![
            InlineKeyboardButton::callback("🔄 Retry", "err:retry".to_string()),
            InlineKeyboardButton::callback("🔀 Switch model", "err:switch".to_string()),
        ]]);
        let _ = bot
            .send_message(
                ctx.chat_id,
                "⚠️ Ответ содержит ошибку провайдера. Повторить?",
            )
            .maybe_thread(ctx.thread_id)
            .reply_markup(keyboard)
            .await;
    }

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
    let rl = &*RATE_LIMITER;
    if parse_html {
        if rl.edit_html(bot, chat_id, msg_id, text).await {
            return true;
        }
        // HTML failed — try plain text
        let plain = strip_html_tags(text);
        return rl.edit_plain(bot, chat_id, msg_id, &plain).await;
    }
    rl.edit_plain(bot, chat_id, msg_id, text).await
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
        reply_text(bot, &ctx, text).await?;
        return Ok(());
    }
    for chunk in split_html(text, MAX_TG_MSG - 100) {
        reply_text(bot, &ctx, chunk).await?;
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
        "🔐 <b>{}</b> [{level}]({preview})\n\
         <i>💡 /yolo = авто-approve | read_file/search — авто</i>",
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
            // FIX-2: Notify user that permission timed out.
            if let Ok(msg) = &sent {
                let text = format!(
                    "⏱ <b>{}</b> — время ожидания истекло ({}s). Запрос отменён.",
                    escape_html(tool_name),
                    PERMISSION_TIMEOUT.as_secs()
                );
                RATE_LIMITER
                    .edit_html(bot, ctx.chat_id, msg.id, &text)
                    .await;
            }
            false
        }
    }
}

fn format_provider_error(err: &str, model_tag: &str) -> String {
    if err.contains("no content") {
        return "— модель закрыла ход без ответа (0 токенов).\n\
             Попробуй переформулировать или /new."
            .to_string();
    }

    let (icon, reason, hint) = if err.contains("429")
        || err.contains("Too Many Requests")
        || err.contains("rate limit")
    {
        (
            "⏳",
            "Rate limit (429)",
            "Подожди 1–2 мин или /model — переключи модель",
        )
    } else if err.contains("402") || err.contains("Payment Required") || err.contains("membership")
    {
        (
            "💳",
            "API ключ — оплата/подписка (402)",
            "/model — переключи провайдер",
        )
    } else if err.contains("401")
        || err.contains("Unauthorized")
        || err.contains("Invalid Authentication")
    {
        ("🔒", "Ключ невалиден (401)", "/model — переключи провайдер")
    } else if err.contains("500")
        || err.contains("502")
        || err.contains("503")
        || err.contains("Internal Server")
    {
        (
            "🔧",
            "Сервер провайдера упал (5xx)",
            "Повтори через минуту или /model",
        )
    } else if err.contains("timeout") || err.contains("Timeout") {
        ("⏱", "Timeout", "Попробуй короче или /model")
    } else if err.contains("reasoning_content") {
        (
            "🧠",
            "Модель требует reasoning format",
            "/new — новая сессия или /model",
        )
    } else {
        ("❌", "Ошибка провайдера", "/model — переключи модель")
    };

    // Extract short error (first sentence or 120 chars, no JSON blobs)
    let short_err = err
        .split('{')
        .next()
        .unwrap_or(err)
        .trim()
        .chars()
        .take(120)
        .collect::<String>();

    format!(
        "{icon} {reason}\n\
         Модель: {model_tag}\n\
         {short_err}\n\
         💡 {hint}"
    )
}

pub(crate) async fn flush_live(
    bot: &Bot,
    chat_id: ChatId,
    msg_id: MessageId,
    view: &CompositeView,
    last_sent: &mut String,
    html_broken: &mut bool,
) -> bool {
    let html = view.render_live();
    let trimmed = truncate_str(&html, MAX_TG_MSG - 50);
    if trimmed == *last_sent {
        return true;
    }
    *last_sent = trimmed.clone();

    // All edits go through the global rate limiter.
    let rl = &*RATE_LIMITER;

    if *html_broken {
        let plain = strip_html_tags(&trimmed);
        rl.edit_plain(bot, chat_id, msg_id, &plain).await;
        return true;
    }

    let ok = rl.edit_html(bot, chat_id, msg_id, &trimmed).await;
    if !ok {
        // Rate limiter returned false — either 429 (parked) or HTML error.
        // Try plain text as fallback.
        let plain = strip_html_tags(&trimmed);
        if !rl.edit_plain(bot, chat_id, msg_id, &plain).await {
            *last_sent = String::new(); // force retry next tick
            return false;
        }
        *html_broken = true;
    }
    true
}

#[cfg(test)]
#[path = "tests.rs"]
mod streaming_tests;
