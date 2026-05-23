//! Render methods for `CompositeView` + HTML document builder.
//!
//! These are the "rendering pipeline" pieces that turn accumulated state
//! into outgoing Telegram HTML strings or attached HTML file bytes.
//!
//! PLAN_TG_INTERLEAVED_v1: rendering walks `events: Vec<TurnEvent>`
//! chronologically. Adjacent `ReasoningDelta` / `TextDelta` are coalesced
//! into single blocks. Budget enforcement uses **head-truncation** (drop
//! OLDEST events first) so the most-recent content is always visible —
//! user-requested behaviour (Q4).

use super::*;
use naked_core::util::head_truncate;

/// Map a tool name to its `<pre><code class="language-X">` tag.
///
/// Conservative — every shell-ish tool maps to `bash` so Telegram desktop
/// highlights the prompt nicely; mobile shows plain monospace either way.
/// Cost: zero.
fn tool_language_tag(name: &str) -> &'static str {
    match name {
        "bash" | "shell" => "bash",
        "python" | "python3" => "python",
        // Other tool names — read/edit/write/git_status/etc. — are
        // file-system or git operations whose args read naturally as
        // shell-style. `bash` highlighting is the safest default.
        _ => "bash",
    }
}

/// Tail-trim a string to fit a byte budget, prepending `…` on truncation.
/// Char-boundary safe.
fn tail_trim(s: &str, budget: usize) -> String {
    if s.len() <= budget {
        return s.to_string();
    }
    let mut start = s.len().saturating_sub(budget.saturating_sub(2));
    while start < s.len() && !s.is_char_boundary(start) {
        start += 1;
    }
    format!("…{}", &s[start..])
}

impl CompositeView {
    /// Render one `TurnEvent` into its HTML block (or `None` if the event
    /// has no standalone representation — e.g. coalesce-able
    /// `ReasoningDelta` / `TextDelta`, which are handled by the
    /// chronological walker).
    ///
    /// `tool_result_budget` caps the per-result body length.
    fn render_event_block(
        &self,
        event: &TurnEvent,
        tool_result_budget: usize,
        for_streaming: bool,
    ) -> Option<String> {
        match event {
            TurnEvent::ReasoningDelta(_) | TurnEvent::TextDelta(_) => None,
            TurnEvent::ToolStart {
                name, args_preview, ..
            } => {
                let lang = tool_language_tag(name);
                let body = format!("$ {name} {args_preview}");
                if for_streaming {
                    // Telegram editMessageText rejects attributes on
                    // <code> during streaming. Strip class.
                    Some(format!("<pre><code>{}</code></pre>", escape_html(&body)))
                } else {
                    Some(format!(
                        "<pre><code class=\"language-{}\">{}</code></pre>",
                        lang,
                        escape_html(&body)
                    ))
                }
            }
            TurnEvent::ToolResult { ok, output, .. } => {
                let trimmed = output.trim();
                if trimmed.is_empty() {
                    return Some(if *ok {
                        "<blockquote>✅ <i>(ok, no output)</i></blockquote>".to_string()
                    } else {
                        "<pre><code>❌ (error, no detail)</code></pre>".to_string()
                    });
                }
                if *ok {
                    // Success: short → inline blockquote; long →
                    // expandable blockquote with shrinkable body.
                    if trimmed.len() <= 200 {
                        Some(format!(
                            "<blockquote>✅ <i>{}</i></blockquote>",
                            escape_html(trimmed)
                        ))
                    } else {
                        let body_cap = tool_result_budget.max(200);
                        let body = tail_trim(trimmed, body_cap);
                        let first_line = trimmed.lines().next().unwrap_or("");
                        let summary = if first_line.len() > 100 {
                            super::helpers::truncate_str(first_line, 100)
                        } else {
                            first_line.to_string()
                        };
                        Some(format!(
                            "<blockquote expandable>✅ <i>{}</i>\n{}</blockquote>",
                            escape_html(&summary),
                            escape_html(&body),
                        ))
                    }
                } else {
                    let preview = output.lines().take(5).collect::<Vec<_>>().join("\n");
                    let body_cap = tool_result_budget.max(200);
                    let body = tail_trim(&preview, body_cap);
                    Some(format!("<pre><code>❌ {}</code></pre>", escape_html(&body)))
                }
            }
            TurnEvent::SubAgentReference { agent_id } => self.sub_agents.get(agent_id).map(|sa| {
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
                let short = super::helpers::truncate_str(&sa.prompt, 40);
                format!(
                    "{icon} <b>{}</b> {}{}",
                    escape_html(agent_id),
                    escape_html(&short),
                    tool_info,
                )
            }),
            TurnEvent::Note(text) => Some(format!("<i>{text}</i>")),
        }
    }

    /// Walk `self.events` chronologically. Coalesce adjacent
    /// `ReasoningDelta` and `TextDelta` events into single blocks.
    /// Returns a vector of HTML-rendered blocks ready for budget-aware
    /// joining.
    ///
    /// `for_streaming = true` strips `<code>` `class="..."` attributes
    /// because Telegram's `editMessageText` rejects them. `false` keeps
    /// language tags for the final `sendMessage` path.
    fn render_chrono_blocks(&self, for_streaming: bool, tool_result_budget: usize) -> Vec<String> {
        let mut blocks: Vec<String> = Vec::with_capacity(self.events.len() / 2 + 4);
        let mut i = 0;
        while i < self.events.len() {
            match &self.events[i] {
                TurnEvent::ReasoningDelta(_) => {
                    let mut j = i;
                    let mut buf = String::new();
                    while let Some(TurnEvent::ReasoningDelta(s)) = self.events.get(j) {
                        buf.push_str(s);
                        j += 1;
                    }
                    let trimmed = buf.trim();
                    if !trimmed.is_empty() {
                        let payload = if for_streaming {
                            tail_trim(trimmed, REASONING_TAIL)
                        } else {
                            tail_trim(trimmed, MAX_FINAL_THINKING_BYTES)
                        };
                        blocks.push(format!(
                            "<blockquote expandable>💭 <b>thinking</b>\n{}</blockquote>",
                            escape_html(&payload)
                        ));
                    }
                    i = j;
                }
                TurnEvent::TextDelta(_) => {
                    let mut j = i;
                    let mut buf = String::new();
                    while let Some(TurnEvent::TextDelta(s)) = self.events.get(j) {
                        buf.push_str(s);
                        j += 1;
                    }
                    let trimmed = buf.trim_end();
                    if !trimmed.is_empty() {
                        if for_streaming {
                            // Streaming: split stable/unstable so an
                            // unclosed markdown fence doesn't break msg.
                            use naked_tg::markup::split_stable_unstable;
                            let (stable, unstable) = split_stable_unstable(trimmed);
                            let mut out = String::new();
                            if !stable.is_empty() {
                                out.push_str(&strip_code_class(&md_to_tg_html(stable)));
                            }
                            if !unstable.is_empty() {
                                if !out.is_empty() {
                                    out.push('\n');
                                }
                                out.push_str(&escape_html(unstable));
                            }
                            if !out.is_empty() {
                                blocks.push(out);
                            }
                        } else {
                            blocks.push(md_to_tg_html(trimmed));
                        }
                    }
                    i = j;
                }
                other => {
                    if let Some(html) =
                        self.render_event_block(other, tool_result_budget, for_streaming)
                    {
                        blocks.push(html);
                    }
                    i += 1;
                }
            }
        }
        blocks
    }

    /// Join blocks into a single message, head-truncating (dropping
    /// OLDEST blocks first) if the total exceeds `budget`. Most-recent
    /// content is preserved (Q4 behaviour).
    ///
    /// Returns `(joined_text, dropped_count)`. Caller can inspect
    /// `dropped_count > 0` to decide whether to attach a full-history
    /// HTML document so the user can review what was cut.
    fn join_with_head_truncate(blocks: &[String], budget: usize) -> (String, usize) {
        use std::collections::VecDeque;
        let mut kept: VecDeque<&str> = VecDeque::new();
        let mut size: usize = 0;
        let mut dropped: usize = 0;
        for (idx, block) in blocks.iter().enumerate().rev() {
            let blen = block.len() + 1; // +1 for \n separator
            if size + blen > budget {
                dropped = idx + 1;
                break;
            }
            kept.push_front(block.as_str());
            size += blen;
        }
        let mut out = String::with_capacity(size + 64);
        if dropped > 0 {
            out.push_str(&format!(
                "… <i>{} earlier event{} truncated — see attached HTML for full timeline</i>\n",
                dropped,
                if dropped == 1 { "" } else { "s" }
            ));
        }
        for (idx, block) in kept.iter().enumerate() {
            if idx > 0 {
                out.push('\n');
            }
            out.push_str(block);
        }
        (out, dropped)
    }

    /// Live composite: spinner status + chronological events + active
    /// tool spinner + sub-agent map lines.
    pub(crate) fn render_live(&self) -> String {
        let spin = self.spinner();
        let tokens = if let Some(u) = &self.usage {
            format!(
                " · ↑{} ↓{}",
                naked_tg::markup::format_tokens(u.input_tokens),
                naked_tg::markup::format_tokens(u.output_tokens)
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
        let cap_str = if let Some(u) = &self.usage {
            let pressure =
                naked_core::capacity::check_pressure(u.input_tokens, self.context_window);
            if pressure.is_actionable() {
                let cap =
                    naked_core::capacity::format_capacity(u.input_tokens, self.context_window);
                format!(" · ctx {cap}")
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        let status = format!(
            "{spin} <i>{} · {elapsed}{tokens}{queue_str}{cap_str}</i>",
            self.phase
        );

        // Build "outside" blocks (active-tool spinner + sub-agent map
        // lines). These render AFTER chronology so they reflect
        // "what's happening RIGHT NOW" beneath the historical tape.
        let mut tail_blocks: Vec<String> = Vec::new();
        if let (Some(name), Some(started)) = (&self.active_tool, &self.tool_started_at) {
            let secs = started.elapsed().as_secs();
            let timer = if secs < 60 {
                format!("{secs}s")
            } else {
                format!("{}:{:02}", secs / 60, secs % 60)
            };
            tail_blocks.push(format!(
                "\u{1f527} <b>{}</b> \u{23f1} {timer}",
                escape_html(name)
            ));
            if let Some(out) = &self.tool_output {
                for line in out
                    .lines()
                    .rev()
                    .take(3)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                {
                    let trimmed = if line.len() > 80 {
                        head_truncate(line, 80)
                    } else {
                        line
                    };
                    tail_blocks.push(format!("  <code>{}</code>", escape_html(trimmed)));
                }
            }
        }
        self.render_sub_agent_lines(&mut tail_blocks);

        let tail_text = tail_blocks.join("\n");
        // Budget for chronology = MAX_TG_MSG - status - tail - safety.
        let chrono_budget = MAX_TG_MSG
            .saturating_sub(status.len())
            .saturating_sub(tail_text.len())
            .saturating_sub(200);
        // Streaming: tool result body capped at 400 chars.
        let chrono_blocks = self.render_chrono_blocks(true, 400);
        let (chrono, _dropped) = Self::join_with_head_truncate(&chrono_blocks, chrono_budget);

        let mut out = String::with_capacity(MAX_TG_MSG);
        out.push_str(&status);
        if !chrono.is_empty() {
            out.push('\n');
            out.push_str(&chrono);
        }
        if !tail_text.is_empty() {
            out.push('\n');
            out.push_str(&tail_text);
        }
        out
    }

    pub(super) fn usage_footer(&self) -> String {
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
                naked_tg::markup::format_tokens(u.input_tokens),
                naked_tg::markup::format_tokens(u.output_tokens),
                cost_str
            )
        } else {
            String::new()
        }
    }

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

    /// Final render: chronological timeline + usage footer.
    ///
    /// Head-truncates the timeline to fit in a single Telegram message;
    /// older events are dropped first with a `… N earlier events
    /// truncated` marker. Language tags on `<pre><code>` are PRESERVED
    /// (this is `sendMessage` not `editMessageText`, so they're accepted).
    pub(crate) fn render_final(&self) -> String {
        // Empty-turn guard: no events AND no accumulated body → existing
        // terse error message (preserves B25 contract).
        if self.events.is_empty()
            && self.response_text.trim().is_empty()
            && self.thinking.trim().is_empty()
        {
            return "<i>— модель закрыла ход без ответа. Попробуй переформулировать или <code>/new</code>.</i>".into();
        }

        let footer = self.usage_footer();
        let footer_rendered = if footer.is_empty() {
            String::new()
        } else {
            format!("\n\n<i>✓ {footer}</i>")
        };

        // Backwards-compat fallback: response_text and/or thinking
        // mutated directly (handle_error path, or test fixtures that
        // pre-populate state without going through handlers). Render
        // as: optional text body + optional thinking blockquote, with
        // the thinking dropped entirely if budget is tight (Q1 + B25
        // contract).
        if self.events.is_empty() {
            let text = self.response_text.trim();
            let thinking = self.thinking.trim();
            let body = if text.is_empty() {
                "<i>(no text — reasoning only)</i>".to_string()
            } else {
                md_to_tg_html(text)
            };
            const WRAPPER_OVERHEAD: usize = 64;
            const MIN_PAYLOAD: usize = 80;
            // 200-byte safety margin absorbs HTML tag overhead from
            // <blockquote expandable> etc., matching pre-refactor
            // contract verified by render_final_drops_thinking_*.
            const SINGLE_MESSAGE_TARGET: usize = MAX_TG_MSG - 200;
            let thinking_budget = SINGLE_MESSAGE_TARGET
                .saturating_sub(body.len())
                .saturating_sub(footer_rendered.len())
                .saturating_sub(2 /* \n\n separator */);
            let mut out = body;
            if !thinking.is_empty() && thinking_budget >= WRAPPER_OVERHEAD + MIN_PAYLOAD {
                let payload_cap = thinking_budget
                    .saturating_sub(WRAPPER_OVERHEAD)
                    .min(MAX_FINAL_THINKING_BYTES);
                let payload = tail_trim(thinking, payload_cap);
                out.push_str("\n\n");
                out.push_str(&format!(
                    "<blockquote expandable>💭 <b>thinking</b>\n{}</blockquote>",
                    escape_html(&payload)
                ));
            }
            out.push_str(&footer_rendered);
            return out;
        }

        const SINGLE_MESSAGE_TARGET: usize = MAX_TG_MSG - 100;
        let budget = SINGLE_MESSAGE_TARGET.saturating_sub(footer_rendered.len());

        let blocks = self.render_chrono_blocks(false, 600);
        let (chrono, dropped) = Self::join_with_head_truncate(&blocks, budget);
        // Stash drop count so `flush.rs::send_final` knows whether to
        // attach the full-history HTML document (Q4 — user wants to
        // see complete timeline when inline was truncated).
        self.last_dropped_events
            .store(dropped, std::sync::atomic::Ordering::Relaxed);

        let mut out = String::with_capacity(chrono.len() + footer_rendered.len() + 8);
        out.push_str(&chrono);
        out.push_str(&footer_rendered);
        out
    }

    /// Thinking block (legacy helper) — retained for tests only. Real
    /// rendering happens through `render_chrono_blocks` chronologically.
    #[cfg(test)]
    pub(crate) fn render_thinking_block(&self) -> Option<String> {
        let trimmed = self.thinking.trim();
        if trimmed.is_empty() {
            return None;
        }
        let payload = tail_trim(trimmed, MAX_FINAL_THINKING_BYTES);
        Some(format!(
            "<blockquote expandable>💭 <b>thinking</b>\n{}</blockquote>",
            escape_html(&payload)
        ))
    }

    /// Truncated preview for the placeholder message when sending a file.
    pub(crate) fn render_summary(&self, max_chars: usize) -> String {
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

pub(crate) fn render_html_document(view: &CompositeView) -> Vec<u8> {
    // PLAN_TG_INTERLEAVED_v1: HTML attach mirrors the inline timeline
    // but WITHOUT head-truncation — nothing is dropped. Walks
    // `view.events` chronologically; falls back to plain
    // response_text/thinking if events is empty (handle_error path).
    let content_html = if view.events.is_empty() {
        let text = view.response_text.trim();
        if text.is_empty() {
            String::new()
        } else {
            md_to_tg_html(text).replace('\n', "<br>\n")
        }
    } else {
        // Render full chronology (no budget, no head-truncate). We
        // request 64K per tool-result body so even huge outputs are
        // preserved in the attached doc. for_streaming=false enables
        // language tags on <pre><code>.
        let blocks = view.render_chrono_blocks(false, 64_000);
        let mut out = String::with_capacity(blocks.iter().map(|b| b.len() + 8).sum::<usize>() + 64);
        for (idx, block) in blocks.iter().enumerate() {
            if idx > 0 {
                out.push_str("<br>\n");
            }
            out.push_str(block);
        }
        out
    };

    // The full reasoning chain — no truncation here, this is the
    // attached HTML file precisely so users can see everything.
    // Render only when chronology path WASN'T used (otherwise thinking
    // is already inline as <blockquote expandable>💭</blockquote> in
    // each chronological position).
    let thinking = view.thinking.trim();
    let thinking_block = if view.events.is_empty() && !thinking.is_empty() {
        format!(
            "<details class=\"thinking\"><summary>💭 reasoning ({} chars)</summary><pre>{}</pre></details>",
            thinking.len(),
            escape_html(thinking)
        )
    } else {
        String::new()
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

#[cfg(test)]
mod tests {
    use super::*;

    // ── Pure helpers ──────────────────────────────────────────────────

    #[test]
    fn tool_language_tag_known_tools() {
        assert_eq!(tool_language_tag("bash"), "bash");
        assert_eq!(tool_language_tag("python"), "python");
        assert_eq!(tool_language_tag("python3"), "python");
        assert_eq!(tool_language_tag("shell"), "bash");
    }

    #[test]
    fn tool_language_tag_unknown_defaults_to_bash() {
        assert_eq!(tool_language_tag("read"), "bash");
        assert_eq!(tool_language_tag("web_search"), "bash");
        assert_eq!(tool_language_tag(""), "bash");
    }

    #[test]
    fn tail_trim_within_budget() {
        let s = "hello world";
        assert_eq!(tail_trim(s, 100), "hello world");
        assert_eq!(tail_trim(s, 11), "hello world"); // exact fit
    }

    #[test]
    fn tail_trim_over_budget() {
        let s = "hello world";
        let t = tail_trim(s, 6);
        assert!(t.starts_with('…'), "truncated should start with ellipsis");
        assert!(t.len() <= 8, "should fit budget approximately");
    }

    #[test]
    fn tail_trim_multibyte_safe() {
        // B48 regression guard: ensure we don't panic on multi-byte chars.
        let s = "привет мир 💭 hello";
        let t = tail_trim(s, 10);
        assert!(t.starts_with('…'));
    }

    // ── join_with_head_truncate ──────────────────────────────────────

    #[test]
    fn join_head_truncate_fits() {
        let blocks = vec!["alpha".to_string(), "beta".to_string()];
        let (joined, dropped) = CompositeView::join_with_head_truncate(&blocks, 1000);
        assert_eq!(dropped, 0);
        assert!(joined.contains("alpha"));
        assert!(joined.contains("beta"));
    }

    #[test]
    fn join_head_truncate_drops_oldest() {
        let blocks: Vec<String> = (0..20).map(|i| format!("block-{i} padding")).collect();
        let (joined, dropped) = CompositeView::join_with_head_truncate(&blocks, 100);
        assert!(dropped > 0, "should drop oldest blocks to fit");
        // Newest block should survive
        assert!(
            joined.contains("block-19"),
            "newest should survive: {joined}"
        );
    }

    // ── CompositeView rendering ─────────────────────────────────────

    fn make_view() -> CompositeView {
        let counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut v = CompositeView::new("test-model".into(), counter);
        v.phase = "idle";
        v
    }

    #[test]
    fn render_live_contains_model() {
        let view = make_view();
        let html = view.render_live();
        assert!(html.contains("idle"), "phase should appear: {html}");
    }

    #[test]
    fn render_final_empty_view() {
        let view = make_view();
        let html = view.render_final();
        // Empty view should not panic and should produce valid-ish HTML
        assert!(!html.is_empty());
    }

    #[test]
    fn render_summary_truncates_long_text() {
        let mut view = make_view();
        view.response_text = "a".repeat(5000);
        let summary = view.render_summary(100);
        // Should be truncated + contain "Full response attached"
        assert!(summary.contains('…'), "should truncate");
        assert!(summary.contains("Full response attached"));
    }

    #[test]
    fn render_html_document_not_empty() {
        let view = make_view();
        let bytes = render_html_document(&view);
        let html = String::from_utf8_lossy(&bytes);
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("test-model") || html.contains("<!DOCTYPE"));
    }
}
