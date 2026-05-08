//! Render methods for `CompositeView` + HTML document builder.
//!
//! These are the "rendering pipeline" pieces that turn accumulated state
//! into outgoing Telegram HTML strings or attached HTML file bytes.

use super::*;

impl CompositeView {
    /// Live composite: spinner status + reasoning tail + last N tools.
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
        // Show context pressure if above normal:
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

        // ── Active tool with per-tool timer + stdout preview ─────
        if let (Some(name), Some(started)) = (&self.active_tool, &self.tool_started_at) {
            let secs = started.elapsed().as_secs();
            let timer = if secs < 60 {
                format!("{secs}s")
            } else {
                format!("{}:{:02}", secs / 60, secs % 60)
            };
            parts.push(format!(
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
                    let trimmed = if line.len() > 80 { &line[..80] } else { line };
                    parts.push(format!("  <code>{}</code>", escape_html(trimmed)));
                }
            }
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
                use naked_tg::markup::split_stable_unstable;

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

    /// Final: replace everything with clean response text + reasoning block + footer.
    ///
    /// The reasoning chain (when present) is wrapped in
    /// `<blockquote expandable>` so it ships collapsed by default — users
    /// who want to see the model's thinking just tap to expand. The
    /// thinking block is dynamically squeezed so the whole final message
    /// fits in a single Telegram message (`MAX_TG_MSG`) — no more
    /// "echo-chunk" second posts that leak CoT drafts after the real
    /// answer.
    pub(crate) fn render_final(&self) -> String {
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
    pub(crate) fn render_thinking_block(&self) -> Option<String> {
        let trimmed = self.thinking.trim();
        self.render_thinking_block_budgeted(trimmed, MAX_FINAL_THINKING_BYTES + 64)
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
