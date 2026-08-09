mod tests {
    use super::super::helpers::*;
    use super::super::*;
    use crate::media_dispatch::{
        ExtractedMediaItem, MediaCtx, MediaItem, MediaProcessed, NativeImage, StickerFormat,
        UnsupportedMediaKind, decide_native_route, fmt_duration, looks_like_supported_image,
    };

    // ── render_thinking_block ───────────────────────────────────────────

    fn view_with_response_and_thinking(response: &str, thinking: &str) -> CompositeView {
        let mut v = CompositeView::new("test-model".into());
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

    // ── PLAN_TG_LONG_ANSWERS_v2 S3 render_final golden corpus ────────────

    const CURRENT_RESPONSE_BRANCH_BUDGET: usize = MAX_TG_MSG - 100;

    #[derive(Debug)]
    struct UnitMeasurements {
        bytes: usize,
        chars: usize,
        utf16: usize,
    }

    fn measure_units(s: &str) -> UnitMeasurements {
        UnitMeasurements {
            bytes: s.len(),
            chars: s.chars().count(),
            utf16: s.encode_utf16().count(),
        }
    }

    fn measurements_line(name: &str, s: &str) -> String {
        let units = measure_units(s);
        format!(
            "{name}: bytes={} chars={} utf16={}",
            units.bytes, units.chars, units.utf16
        )
    }

    fn astral_at_utf16_budget_fixture() -> String {
        "💭".repeat(CURRENT_RESPONSE_BRANCH_BUDGET / 2)
    }

    fn astral_over_utf16_budget_fixture() -> String {
        "💭".repeat(CURRENT_RESPONSE_BRANCH_BUDGET / 2 + 1)
    }

    fn cyrillic_at_byte_budget_fixture() -> String {
        "Ж".repeat(CURRENT_RESPONSE_BRANCH_BUDGET / "Ж".len())
    }

    fn cyrillic_over_byte_budget_fixture() -> String {
        "Ж".repeat(CURRENT_RESPONSE_BRANCH_BUDGET / "Ж".len() + 1)
    }

    fn assert_render_final_branch(out: &str, response: &str, should_truncate_today: bool) {
        assert_eq!(
            out.ends_with('…'),
            should_truncate_today,
            "current byte-accounting branch changed for measurements {:?}",
            measure_units(response)
        );
    }

    fn assert_long_answer_fix_preserves_response(out: &str, response: &str) {
        assert!(
            !out.ends_with('…'),
            "flag-on render must not pre-truncate measurements {:?}",
            measure_units(response)
        );
        assert_eq!(naked_tg::markup::telegram_html_visible_text(out), response);
    }

    fn assert_astral_fixture_discriminates_chars_from_utf16(response: &str, over_utf16: bool) {
        let units = measure_units(response);
        assert!(
            units.chars < CURRENT_RESPONSE_BRANCH_BUDGET,
            "astral fixture must fit under chars() so chars-vs-UTF-16 differs: {units:?}"
        );
        assert_eq!(
            units.utf16 > CURRENT_RESPONSE_BRANCH_BUDGET,
            over_utf16,
            "astral fixture must sit on the UTF-16 boundary: {units:?}"
        );
        assert!(
            units.bytes > CURRENT_RESPONSE_BRANCH_BUDGET,
            "current byte gate must still truncate this no-behaviour-change golden: {units:?}"
        );
    }

    fn assert_cyrillic_fixture_discriminates_bytes(response: &str, over_bytes: bool) {
        let units = measure_units(response);
        assert!(
            units.chars < CURRENT_RESPONSE_BRANCH_BUDGET,
            "Cyrillic fixture must fit under chars(): {units:?}"
        );
        assert_eq!(
            units.utf16, units.chars,
            "BMP Cyrillic should be one UTF-16 unit per scalar: {units:?}"
        );
        assert_eq!(
            units.bytes > CURRENT_RESPONSE_BRANCH_BUDGET,
            over_bytes,
            "Cyrillic fixture must sit on the byte boundary: {units:?}"
        );
    }

    fn view_with_response_branch(response: &str) -> CompositeView {
        let mut v = CompositeView::new("test-model".into());
        // Any chronological event switches render_final onto the current
        // response_text-priority branch, where over-budget answers are
        // pre-truncated before send_final sees them. The note is not
        // rendered in that branch; it only selects the branch under test.
        v.events.push(TurnEvent::Note("branch marker".into()));
        v.response_text = response.to_string();
        v
    }

    fn view_with_chrono_events(events: Vec<TurnEvent>) -> CompositeView {
        let mut v = CompositeView::new("test-model".into());
        v.events = events;
        v
    }

    fn assert_only_supported_telegram_tags(html: &str) {
        let mut rest = html;
        while let Some(start) = rest.find('<') {
            rest = &rest[start + 1..];
            let Some(end) = rest.find('>') else {
                panic!("unterminated tag in render output: {html}");
            };
            let raw_tag = rest[..end].trim();
            let tag = raw_tag
                .trim_start_matches('/')
                .split_whitespace()
                .next()
                .unwrap_or("");
            assert!(
                matches!(tag, "a" | "b" | "blockquote" | "code" | "i" | "pre"),
                "unsupported Telegram HTML tag <{raw_tag}> in: {html}"
            );
            rest = &rest[end + 1..];
        }
    }

    #[test]
    fn render_final_golden_short_markdown() {
        let v = view_with_response_branch("Short **bold** and `code` answer.");
        insta::assert_snapshot!("render_final_short_markdown", v.render_final());
    }

    #[test]
    fn render_final_golden_entity_heavy_text() {
        let v = view_with_response_branch(
            "Entity-heavy: literal & plus <angle> and > sign; pre-escaped &lt; must round-trip.",
        );
        insta::assert_snapshot!("render_final_entity_heavy_text", v.render_final());
    }

    #[test]
    fn render_final_golden_emoji_astral_and_zwj() {
        // M8 guard fixture: bytes, chars(), and UTF-16 units disagree for
        // astral-plane emoji and ZWJ families (💭/🤖/👨‍👩‍👧).
        let v = view_with_response_branch(
            "Emoji pressure: 💭 thinking, 🤖 bot, family 👨‍👩‍👧, and plain text.",
        );
        insta::assert_snapshot!("render_final_emoji_astral_and_zwj", v.render_final());
    }

    #[test]
    fn render_final_golden_cyrillic() {
        let v = view_with_response_branch(
            "Кириллица: проверяем, что текущий рендерер сохраняет русский текст и **жирный** фрагмент.",
        );
        insta::assert_snapshot!("render_final_cyrillic", v.render_final());
    }

    #[test]
    fn render_final_golden_tag_heavy_markdown() {
        let md = r#"# Heading <raw>

> Quote with **bold** & symbols

- item with *italic*
- item with `inline <code>`

```rust
fn main() { println!("<hi>&"); }
```

See [docs](https://example.com/path?a=1&b=2)."#;
        let v = view_with_response_branch(md);
        insta::assert_snapshot!("render_final_tag_heavy_markdown", v.render_final());
    }

    #[test]
    fn render_final_golden_exactly_at_current_response_budget() {
        let response = "x".repeat(CURRENT_RESPONSE_BRANCH_BUDGET);
        let out = view_with_response_branch(&response).render_final();
        assert_eq!(out.len(), CURRENT_RESPONSE_BRANCH_BUDGET);
        assert!(!out.ends_with('…'));
        insta::assert_snapshot!("render_final_exactly_at_current_response_budget", out);
    }

    #[test]
    fn render_final_golden_over_current_response_budget() {
        let response = format!("{}TAIL", "x".repeat(CURRENT_RESPONSE_BRANCH_BUDGET));
        let out = view_with_response_branch(&response).render_final();
        assert!(out.ends_with('…'));
        assert!(out.len() <= MAX_TG_MSG);
        insta::assert_snapshot!("render_final_over_current_response_budget", out);
    }

    // These boundary goldens intentionally pin CURRENT behaviour, not the
    // desired Telegram behaviour. S7 will change the budget unit from bytes to
    // UTF-16 and should update these snapshots deliberately. Today the code
    // truncates by bytes, so Cyrillic at ~2k chars / ~4k bytes is truncated even
    // though it is safely under Telegram's UTF-16 limit.
    #[test]
    fn render_final_boundary_fixture_measurements() {
        let rows = [
            measurements_line(
                "astral_utf16_at_current_response_budget",
                &astral_at_utf16_budget_fixture(),
            ),
            measurements_line(
                "astral_utf16_over_current_response_budget",
                &astral_over_utf16_budget_fixture(),
            ),
            measurements_line(
                "cyrillic_bytes_at_current_response_budget",
                &cyrillic_at_byte_budget_fixture(),
            ),
            measurements_line(
                "cyrillic_bytes_over_current_response_budget",
                &cyrillic_over_byte_budget_fixture(),
            ),
        ];
        insta::assert_snapshot!(
            "render_final_boundary_fixture_measurements",
            rows.join("\n")
        );
    }

    #[test]
    fn render_final_golden_astral_utf16_at_current_response_budget() {
        let response = astral_at_utf16_budget_fixture();
        assert_astral_fixture_discriminates_chars_from_utf16(&response, false);
        let out = view_with_response_branch(&response).render_final();
        insta::assert_snapshot!(
            "render_final_astral_utf16_at_current_response_budget",
            out.as_str()
        );
        assert_render_final_branch(&out, &response, true);
    }

    #[test]
    fn render_final_golden_astral_utf16_over_current_response_budget() {
        let response = astral_over_utf16_budget_fixture();
        assert_astral_fixture_discriminates_chars_from_utf16(&response, true);
        let out = view_with_response_branch(&response).render_final();
        insta::assert_snapshot!(
            "render_final_astral_utf16_over_current_response_budget",
            out.as_str()
        );
        assert_render_final_branch(&out, &response, true);
    }

    #[test]
    fn render_final_golden_cyrillic_bytes_at_current_response_budget() {
        let response = cyrillic_at_byte_budget_fixture();
        assert_cyrillic_fixture_discriminates_bytes(&response, false);
        let out = view_with_response_branch(&response).render_final();
        insta::assert_snapshot!(
            "render_final_cyrillic_bytes_at_current_response_budget",
            out.as_str()
        );
        assert_render_final_branch(&out, &response, false);
    }

    #[test]
    fn render_final_golden_cyrillic_bytes_over_current_response_budget() {
        let response = cyrillic_over_byte_budget_fixture();
        assert_cyrillic_fixture_discriminates_bytes(&response, true);
        let out = view_with_response_branch(&response).render_final();
        insta::assert_snapshot!(
            "render_final_cyrillic_bytes_over_current_response_budget",
            out.as_str()
        );
        assert_render_final_branch(&out, &response, true);
    }

    #[test]
    fn render_final_flag_on_golden_astral_utf16_at_current_response_budget() {
        let response = astral_at_utf16_budget_fixture();
        assert_astral_fixture_discriminates_chars_from_utf16(&response, false);
        let out = view_with_response_branch(&response).render_final_with_long_answer_fix(true);
        insta::assert_snapshot!(
            "render_final_flag_on_astral_utf16_at_current_response_budget",
            out.as_str()
        );
        assert_long_answer_fix_preserves_response(&out, &response);
    }

    #[test]
    fn render_final_flag_on_golden_cyrillic_bytes_over_current_response_budget() {
        let response = cyrillic_over_byte_budget_fixture();
        assert_cyrillic_fixture_discriminates_bytes(&response, true);
        let out = view_with_response_branch(&response).render_final_with_long_answer_fix(true);
        insta::assert_snapshot!(
            "render_final_flag_on_cyrillic_bytes_over_current_response_budget",
            out.as_str()
        );
        assert_long_answer_fix_preserves_response(&out, &response);
    }

    #[test]
    fn render_final_flag_on_no_truncation_counter_for_long_response_branch() {
        let response = cyrillic_over_byte_budget_fixture();
        let v = view_with_response_branch(&response);
        let out = v.render_final_with_long_answer_fix(true);
        assert_long_answer_fix_preserves_response(&out, &response);
        assert!(
            !v.last_final_answer_truncated
                .load(std::sync::atomic::Ordering::Relaxed),
            "flag-on response branch must stop marking long text as inline-truncated"
        );
    }

    #[test]
    fn render_final_golden_tool_events_and_thinking_block() {
        let v = view_with_chrono_events(vec![
            TurnEvent::ReasoningDelta("Need to inspect & compare <paths>. ".into()),
            TurnEvent::ReasoningDelta("Final thought keeps 💭 marker context.".into()),
            TurnEvent::ToolStart {
                name: "bash".into(),
                args_preview: "printf '<x>&'".into(),
                idx: 0,
            },
            TurnEvent::ToolResult {
                idx: 0,
                ok: true,
                output: "ok & <done>\nline two".into(),
            },
            TurnEvent::ToolStart {
                name: "python3".into(),
                args_preview: "script.py --flag".into(),
                idx: 1,
            },
            TurnEvent::ToolResult {
                idx: 1,
                ok: false,
                output: "Traceback <bad> & fail\nsecond\nthird\nfourth\nfifth\nsixth".into(),
            },
            TurnEvent::TextDelta("Chronological answer with **bold** after tools.".into()),
        ]);
        insta::assert_snapshot!(
            "render_final_tool_events_and_thinking_block",
            v.render_final()
        );
    }

    #[test]
    fn render_live_structural_invariants_not_byte_golden() {
        // render_live is intentionally not byte-goldened in S3. Its output
        // includes CompositeView::spinner() (tick-dependent), elapsed wall
        // time from an Instant created in CompositeView::new(), per-tool
        // timers from another Instant, and optional context-pressure text
        // from a model-table lookup. There is no time/tick injection seam
        // today, and this step is a pure baseline with no behaviour change.
        let mut view = view_with_chrono_events(vec![
            TurnEvent::ReasoningDelta("live reasoning with 💭".into()),
            TurnEvent::TextDelta("Live **answer** with & and <tag> plus 🤖.".into()),
        ]);
        view.phase = "tools";
        view.tick = 2;
        view.active_tool = Some("bash".into());
        view.tool_started_at = Some(std::time::Instant::now());
        view.tool_output = Some("first line\nraw & <tag>\nthird line\nfourth line".into());

        let html = view.render_live();
        assert!(html.contains("<i>tools · 🕐"), "status missing: {html}");
        assert!(
            ["⏳", "⌛"].iter().any(|frame| html.starts_with(frame)),
            "spinner frame missing: {html}"
        );
        assert!(
            html.contains("💭 <b>thinking</b>"),
            "thinking marker missing: {html}"
        );
        assert!(
            html.contains("Live <b>answer</b>"),
            "text block missing: {html}"
        );
        assert!(html.contains("&amp;"), "entity escaping missing: {html}");
        assert!(
            html.contains("&lt;tag&gt;"),
            "angle escaping missing: {html}"
        );
        assert!(
            html.contains("🔧 <b>bash</b> ⏱"),
            "active tool marker missing: {html}"
        );
        assert!(
            !html.contains("class="),
            "live render must strip code classes: {html}"
        );
        assert_only_supported_telegram_tags(&html);
    }

    #[test]
    fn render_live_trims_known_oversized_chronology() {
        let mut view = view_with_chrono_events(vec![
            TurnEvent::TextDelta(format!("OLD_SHOULD_DROP {}", "x".repeat(MAX_TG_MSG * 2))),
            TurnEvent::ReasoningDelta("RECENT_SHOULD_STAY".into()),
        ]);
        view.phase = "streaming";
        view.tick = 1;

        let html = view.render_live();
        assert!(
            html.len() <= MAX_TG_MSG,
            "trimmed live render should stay within the Telegram byte cap: {}",
            html.len()
        );
        assert!(
            html.contains("earlier event truncated"),
            "known oversized chronology should take the trimming branch: {html}"
        );
        assert!(
            !html.contains("OLD_SHOULD_DROP"),
            "old oversized chronology block should be dropped: {html}"
        );
        assert!(
            html.contains("RECENT_SHOULD_STAY"),
            "recent chronology should survive trimming: {html}"
        );
    }

    // ── is_allowed ──────────────────────────────────────────────────────

    #[test]
    fn is_allowed_empty_list_denies_all() {
        let config = Config {
            telegram: naked_core::config::TelegramConfig {
                allowed_chat_ids: vec![],
                ..Default::default()
            },
            ..Config::default()
        };
        assert!(!is_allowed(123, &config));
        assert!(!is_allowed(0, &config));
    }

    #[test]
    fn is_allowed_with_ids_checks_membership() {
        let config = Config {
            telegram: naked_core::config::TelegramConfig {
                allowed_chat_ids: vec![100, 200],
                ..Default::default()
            },
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

    // ─── BUG_REGISTRY B36 regression guard ───
    //
    // ask_permission MUST html-escape the preview before embedding in
    // a parse_mode=Html message. format_input_preview returns RAW;
    // bash heredocs ("<< 'EOF'") and URLs with query strings ("&")
    // routinely appear in tool inputs. Pre-fix: TG returned 400, send
    // failed silently, tool was auto-denied without operator seeing
    // any prompt. Test asserts that for inputs containing HTML
    // metacharacters, escape_html on the preview neutralises them.
    #[test]
    fn preview_with_heredoc_does_not_break_html() {
        let input = serde_json::json!({
            "command": "python3 << 'EOF'\nimport json\nEOF",
            "timeout": 15
        });
        let raw = format_input_preview(&input, 200);
        // Raw MUST contain unsafe `<` (the bug input).
        assert!(raw.contains("<<"), "setup invariant: heredoc has <<");
        // After escape_html, no raw `<` should remain — only entity-escaped.
        let escaped = crate::markup::escape_html(&raw);
        assert!(
            !escaped.contains("<") && !escaped.contains(">"),
            "escape_html must remove raw angle brackets: {escaped}"
        );
        assert!(
            escaped.contains("&lt;") || escaped.contains("&amp;"),
            "escape_html must produce HTML entities: {escaped}"
        );
    }

    #[test]
    fn preview_with_url_ampersand_does_not_break_html() {
        let input = serde_json::json!({
            "url": "https://example.com/?a=1&b=2&c=3"
        });
        let raw = format_input_preview(&input, 200);
        assert!(raw.contains("&"), "setup: URL has unescaped &");
        let escaped = crate::markup::escape_html(&raw);
        assert!(
            !escaped.contains("a=1&b="),
            "escape_html must split raw & to &amp;: {escaped}"
        );
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

    fn only_supported(items: &[ExtractedMediaItem]) -> &MediaItem {
        assert_eq!(items.len(), 1, "expected exactly one extracted media item");
        items[0].supported().expect("expected supported media item")
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
        let item = only_supported(&items);
        assert_eq!(item.kind, media::MediaKind::Voice);
        assert_eq!(item.file_id, "voice-abc");
        assert!(item.file_name.ends_with(".ogg"));
        assert_eq!(item.duration_secs, Some(12));
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
        let item = only_supported(&items);
        assert_eq!(item.kind, media::MediaKind::Photo);
        assert_eq!(item.file_id, "big");
        assert_eq!(item.mime_hint.as_deref(), Some("image/jpeg"));
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
        let item = only_supported(&items);
        assert_eq!(item.kind, media::MediaKind::Document);
        assert_eq!(item.file_name, "notes.md");
        assert_eq!(item.mime_hint.as_deref(), Some("text/markdown"));
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
    fn extract_media_items_supported_kinds_still_extract_as_supported() {
        let cases = [
            (
                "voice",
                serde_json::json!({
                    "file_id": "voice-1", "file_unique_id": "u", "duration": 1, "mime_type": "audio/ogg"
                }),
                media::MediaKind::Voice,
            ),
            (
                "audio",
                serde_json::json!({
                    "file_id": "audio-1", "file_unique_id": "u", "duration": 2, "file_name": "song.mp3", "mime_type": "audio/mpeg"
                }),
                media::MediaKind::Audio,
            ),
            (
                "photo",
                serde_json::json!([
                    {"file_id":"photo-1","file_unique_id":"u","width":10,"height":10,"file_size":100}
                ]),
                media::MediaKind::Photo,
            ),
            (
                "video",
                serde_json::json!({
                    "file_id": "video-1", "file_unique_id": "u", "width": 640, "height": 480, "duration": 3, "file_name": "clip.mp4", "mime_type": "video/mp4"
                }),
                media::MediaKind::Video,
            ),
            (
                "animation",
                serde_json::json!({
                    "file_id": "animation-1", "file_unique_id": "u", "width": 320, "height": 240, "duration": 4, "file_name": "anim.mp4", "mime_type": "video/mp4"
                }),
                media::MediaKind::Animation,
            ),
            (
                "document",
                serde_json::json!({
                    "file_id": "document-1", "file_unique_id": "u", "file_name": "notes.txt", "mime_type": "text/plain"
                }),
                media::MediaKind::Document,
            ),
            (
                "sticker",
                serde_json::json!({
                    "file_id": "sticker-1", "file_unique_id": "u", "type": "regular", "width": 128, "height": 128, "is_animated": false, "is_video": false, "emoji": "🙂"
                }),
                media::MediaKind::Sticker,
            ),
        ];

        for (field, payload, expected) in cases {
            let mut m = base_private_chat();
            m["from"] = user(Some("alice"), "Alice", false);
            m[field] = payload;
            let items = extract_media_items(&make_message(m));
            let item = only_supported(&items);
            assert_eq!(item.kind, expected, "{field} should remain supported");
        }
    }

    #[tokio::test]
    async fn extract_media_items_video_note_reaches_prompt_payload_as_unsupported() {
        let mut m = base_private_chat();
        m["from"] = user(Some("alice"), "Alice", false);
        m["video_note"] = serde_json::json!({
            "file_id": "video-note-1",
            "file_unique_id": "vn-u",
            "length": 240,
            "duration": 5,
            "file_size": 12345
        });
        let items = extract_media_items(&make_message(m));
        assert_eq!(items.len(), 1);
        let unsupported = items[0]
            .unsupported()
            .expect("video_note must be distinct from no media");
        assert_eq!(unsupported.kind, UnsupportedMediaKind::VideoNote);

        let cfg = Config::default();
        let http = Arc::new(reqwest::Client::new());
        let base_url = Arc::new("https://api.telegram.org".to_string());
        let ctx = MediaCtx {
            bot_token: "test-token",
            config: &cfg,
            http,
            base_url,
            user_caption: None,
            msg_id: 1,
            route_images_natively: false,
            native_cap_bytes: 0,
            active_model: "test-model",
        };
        let processed = process_media_items(&items, &ctx).await;
        assert!(
            processed
                .text
                .contains("unsupported Telegram attachment: video_note"),
            "prompt payload must tell the model the video_note was not read, got: {}",
            processed.text
        );
        assert!(
            processed
                .text
                .contains("cannot read this attachment type yet"),
            "prompt payload must forbid pretending the attachment was read, got: {}",
            processed.text
        );
        assert!(processed.native_images.is_empty());
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
        let item = only_supported(&items);
        assert_eq!(item.kind, media::MediaKind::Photo);
        assert_eq!(
            item.file_id, "reply-big",
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
        let item = only_supported(&items);
        assert_eq!(item.kind, media::MediaKind::Sticker);
        assert_eq!(item.emoji.as_deref(), Some("\u{1F525}"));
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
        let item = only_supported(&items);
        assert_eq!(item.sticker_format, Some(StickerFormat::Static));
        assert_eq!(item.mime_hint.as_deref(), Some("image/webp"));
        assert!(item.file_name.ends_with(".webp"));
        assert_eq!(item.size_hint, Some(32_000));
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
        let item = only_supported(&items);
        assert_eq!(item.sticker_format, Some(StickerFormat::Animated));
        assert!(item.file_name.ends_with(".tgs"));
        assert_eq!(
            item.mime_hint.as_deref(),
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
        let item = only_supported(&items);
        assert_eq!(item.sticker_format, Some(StickerFormat::Video));
        assert!(item.file_name.ends_with(".webm"));
        assert_eq!(item.mime_hint.as_deref(), Some("video/webm"));
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
        let item = only_supported(&items);
        assert_eq!(item.kind, media::MediaKind::Photo);
        // We pick the largest resolution → its size_hint should be 200_000.
        assert_eq!(item.size_hint, Some(200_000));
    }

    // ── TG HTTP mock (wiremock) ─────────────────────────────────────────
    //
    // These tests stand up a local HTTP server that pretends to be the
    // Telegram Bot API and verify our outgoing `sendMessage` plumbing
    // talks to it correctly. They exist to catch regressions in the
    // low-level `Bot`/`reqwest` layer — higher-level dispatch logic is
    // covered by the `album::tests` module.

    use std::sync::atomic::{AtomicU64, Ordering};

    use wiremock::matchers::{method, path_regex};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn unique_test_id() -> u64 {
        static NEXT: AtomicU64 = AtomicU64::new(10_000);
        NEXT.fetch_add(1, Ordering::Relaxed)
    }

    /// Build a `Bot` pointing at the provided mock URL instead of
    /// `api.telegram.org`. Token is a throwaway.
    fn mock_bot(mock_url: &str) -> Bot {
        let url = reqwest::Url::parse(mock_url).unwrap();
        Bot::new("0:TEST_TOKEN").set_api_url(url)
    }

    fn ok_message_response(text: &str) -> ResponseTemplate {
        ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "ok": true,
            "result": {
                "message_id": 42,
                "date": 0,
                "chat": {"id": 1, "type": "private", "first_name": "u"},
                "text": text,
            }
        }))
    }

    fn ok_document_response() -> ResponseTemplate {
        ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "ok": true,
            "result": {
                "message_id": 43,
                "date": 0,
                "chat": {"id": 1, "type": "private", "first_name": "u"},
                "document": {
                    "file_id": "doc",
                    "file_unique_id": "uniq",
                    "file_name": "response.html"
                }
            }
        }))
    }

    fn medium_final_fixture() -> (String, CompositeView) {
        let body = format!("HEAD-{}-TAIL", "x".repeat(MAX_TG_MSG + 500));
        assert!(body.len() > MAX_TG_MSG && body.len() <= MAX_TG_MSG * 2);
        let mut view = CompositeView::new("test-model".into());
        view.response_text = body.clone();
        (body, view)
    }

    fn request_method_names(received: &[wiremock::Request]) -> Vec<String> {
        received
            .iter()
            .map(|req| {
                req.url
                    .path()
                    .rsplit('/')
                    .next()
                    .unwrap_or_default()
                    .to_ascii_lowercase()
            })
            .collect()
    }

    fn request_text_payload(req: &wiremock::Request) -> Option<String> {
        let body: serde_json::Value = serde_json::from_slice(&req.body).ok()?;
        body.get("text")?.as_str().map(ToOwned::to_owned)
    }

    fn ordered_text_payloads(received: &[wiremock::Request]) -> Vec<String> {
        received
            .iter()
            .filter_map(request_text_payload)
            .collect::<Vec<_>>()
    }

    struct NoopProvider;

    fn test_config() -> naked_core::config::Config {
        let tmp = tempfile::tempdir().unwrap();
        let config = naked_core::config::Config {
            run_registry_multi_stream_enabled: true,
            workspace: tmp.path().join("workspace"),
            session_dir: tmp.path().join("sessions"),
            ..Default::default()
        };
        std::fs::create_dir_all(&config.workspace).unwrap();
        std::fs::create_dir_all(&config.session_dir).unwrap();
        std::mem::forget(tmp);
        config
    }

    fn test_bot_deps(
        bot: Bot,
        config: naked_core::config::Config,
        channel_map: std::sync::Arc<ChannelSessionMap>,
        base_url: String,
    ) -> crate::message_handler::BotDeps {
        let agent = std::sync::Arc::new(naked_core::AgentCore::new(
            config.clone(),
            Box::new(NoopProvider),
        ));
        crate::message_handler::BotDeps {
            bot,
            agent,
            channel_map,
            config,
            pending_perms: std::sync::Arc::new(tokio::sync::RwLock::new(
                std::collections::HashMap::new(),
            )),
            http_client: std::sync::Arc::new(reqwest::Client::new()),
            base_url: std::sync::Arc::new(base_url),
            rate_limiter: naked_tg::rate_limit::RateLimiter::new(),
            attribution_flag: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            bot_token: std::sync::Arc::new("0:TEST_TOKEN".to_string()),
            bot_identity: std::sync::Arc::new(naked_tg::bot_identity::BotIdentity {
                id: 0,
                username: "test_bot".to_string(),
            }),
            tg_attach_queue: naked_tg::tg_attach::new_queue(),
            research_scheduler: None,
            per_chat_locks: std::sync::Arc::new(crate::per_chat_locks::PerChatLocks::new()),
        }
    }

    #[async_trait::async_trait]
    impl naked_core::provider::Provider for NoopProvider {
        fn name(&self) -> &str {
            "noop"
        }

        fn models(&self) -> Vec<naked_core::types::ModelInfo> {
            vec![naked_core::types::ModelInfo {
                provider: "noop".to_string(),
                model_id: "noop-model".to_string(),
                display_name: "noop".to_string(),
            }]
        }

        async fn stream_chat(
            &self,
            _request: naked_core::provider::ChatRequest,
        ) -> naked_core::error::Result<
            std::pin::Pin<
                Box<dyn tokio_stream::Stream<Item = naked_core::types::StreamChunk> + Send>,
            >,
        > {
            Ok(Box::pin(tokio_stream::empty()))
        }
    }

    #[test]
    fn send_final_consumes_edit_must_deliver_results_at_all_sites() {
        let src = include_str!("flush.rs");
        for consumed in [
            "if !edit_must_deliver(&bot, chat_id, msg_id, html, true).await",
            "!edit_must_deliver(&bot, chat_id, msg_id, first, true).await",
            "let summary_delivered = if tg_payload_within_utf16_budget(&summary, \"final summary edit\")",
        ] {
            assert!(
                src.contains(consumed),
                "send_final must consume this edit_must_deliver result: {consumed}"
            );
        }
        for discarded in [
            "edit_must_deliver(&bot, chat_id, msg_id, html, true).await;",
            "edit_must_deliver(&bot, chat_id, msg_id, first, true).await;",
            "edit_must_deliver(&bot, chat_id, msg_id, &summary, true).await;",
        ] {
            assert!(
                !src.contains(discarded),
                "send_final must not discard edit_must_deliver result: {discarded}"
            );
        }
    }

    /// B119b: the rejected-stream drain must give up, not park forever.
    ///
    /// It used to `await events.recv()` with no timeout. A rejected start
    /// whose agent task ignores its cancelled token held this future — and the
    /// task — indefinitely.
    ///
    /// This waits out the real timeout (~10s): `--bins` tests compile into the
    /// binary, not a dev target, so tokio's `test-util` paused clock is not
    /// available here. Ten seconds once is cheaper than the alternative of
    /// weakening the assertion to a constant check only.
    #[tokio::test]
    async fn rejected_stream_drain_gives_up_instead_of_parking_forever() {
        // Sender is kept alive and never sends Idle: the exact wedged-task shape.
        let (_events_tx, events_rx) = tokio::sync::mpsc::channel::<AgentEvent>(4);
        let (perm_tx, _perm_rx) = tokio::sync::mpsc::channel(4);

        // The drain's own timeout must be short enough that a wedged task is
        // abandoned rather than held for the life of the process.
        assert!(
            crate::streaming::pipeline::REJECTED_DRAIN_TIMEOUT
                <= std::time::Duration::from_secs(30),
            "an unbounded-in-practice drain timeout defeats the purpose"
        );

        let outer = crate::streaming::pipeline::REJECTED_DRAIN_TIMEOUT * 3;
        let drained = tokio::time::timeout(
            outer,
            crate::streaming::pipeline::drain_rejected_stream(events_rx, perm_tx),
        )
        .await;

        assert!(
            drained.is_ok(),
            "drain must return on its own timeout; it parked instead"
        );
    }

    /// B119a: every rejected start must produce a user-visible message.
    ///
    /// Only `ThreadCapacityExceeded` used to speak. The other three variants
    /// were logged and drained, so a user who sent a message got no
    /// placeholder, no text, and no run to /abort — the message simply
    /// vanished. This asserts on the exhaustive match, which is also why
    /// adding a variant now fails to compile instead of failing silently.
    #[test]
    fn every_registration_rejection_has_user_visible_text() {
        use naked_tg::run_registry::{ChatThreadKey, RegisterRunError};

        let all = [
            RegisterRunError::DuplicateRunId {
                run_id: "r1".into(),
            },
            RegisterRunError::DuplicateSessionId {
                session_id: "s1".into(),
                existing: "r1".into(),
            },
            RegisterRunError::DuplicateSourceRef {
                source_ref: "src".into(),
                existing: "r1".into(),
            },
            RegisterRunError::ThreadCapacityExceeded {
                key: ChatThreadKey::new(1_i64, None),
                active: 3,
                cap: 3,
            },
        ];

        for err in all {
            let msg = crate::streaming::pipeline::reject_message(&err);
            assert!(
                !msg.trim().is_empty(),
                "rejection {err:?} must tell the user something"
            );
            assert!(
                msg.contains('\u{26a0}'),
                "rejection {err:?} must read as a warning, got: {msg}"
            );
        }
    }

    /// B118b at the transport seam: the stall warning must reach TELEGRAM, not
    /// just `render_final`'s return value.
    ///
    /// The unit tests pin the string composition; this pins the bytes actually
    /// PUT ON THE WIRE by `send_final`, through the same mock-bot transport the
    /// other delivery tests use. Without it, a later change to how the final
    /// message is assembled for sending could drop the banner again while every
    /// render test stayed green.
    #[tokio::test]
    async fn send_final_puts_stall_warning_on_the_wire() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ok_message_response("ok"))
            .mount(&server)
            .await;

        let mut view = CompositeView::new("test-model".into());
        view.stall_notice = Some(
            "\u{1f534} \u{417}\u{430}\u{432}\u{438}\u{441}\u{43b}\u{43e} 2 \u{43c}\u{438}\u{43d}"
                .into(),
        );
        view.response_text = "\u{412}\u{43e}\u{437}\u{44c}\u{43c}\u{443} erp_analyst".into();
        let html = view.render_final_with_long_answer_fix(false);

        let unique = unique_test_id();
        crate::streaming::flush::send_final(
            mock_bot(&server.uri()),
            crate::shared::ChatCtx {
                chat_id: ChatId(990_000 + unique as i64),
                thread_id: None,
                reply_to: None,
            },
            MessageId(123),
            &html,
            &view,
            false,
        )
        .await;

        let received = server.received_requests().await.unwrap();
        let bodies: String = received
            .iter()
            .map(|r| String::from_utf8_lossy(&r.body).into_owned())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            bodies.contains(
                "\u{417}\u{430}\u{432}\u{438}\u{441}\u{43b}\u{43e} 2 \u{43c}\u{438}\u{43d}"
            ),
            "stall warning never reached the transport; bodies={bodies}"
        );
        assert!(
            bodies.contains("erp_analyst"),
            "model answer never reached the transport; bodies={bodies}"
        );
    }

    #[tokio::test]
    async fn send_final_part1_non429_falls_back_to_attachment_without_orphan_chunks() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(r"(?i)/editmessagetext$"))
            .respond_with(ResponseTemplate::new(400).set_body_json(serde_json::json!({
                "ok": false,
                "error_code": 400,
                "description": "Bad Request: message to edit not found"
            })))
            .up_to_n_times(2)
            .expect(2)
            .with_priority(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path_regex(r"(?i)/senddocument$"))
            .respond_with(ok_document_response())
            .expect(1)
            .with_priority(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .respond_with(ok_message_response("ok"))
            .with_priority(10)
            .mount(&server)
            .await;

        let unique = unique_test_id();
        let (html, view) = medium_final_fixture();
        let bot = mock_bot(&server.uri());
        crate::streaming::flush::send_final(
            bot,
            crate::shared::ChatCtx {
                chat_id: ChatId(990_000 + unique as i64),
                thread_id: None,
                reply_to: None,
            },
            MessageId(123),
            &html,
            &view,
            false,
        )
        .await;

        let received = server.received_requests().await.unwrap();
        let methods = request_method_names(&received);
        assert_eq!(
            methods
                .iter()
                .filter(|m| m.as_str() == "sendmessage")
                .count(),
            0,
            "part-1 failure must not send orphan chunks 2..n; methods={methods:?}"
        );
        assert_eq!(
            methods
                .iter()
                .filter(|m| m.as_str() == "senddocument")
                .count(),
            1,
            "part-1 failure must preserve content via attachment; methods={methods:?}"
        );
        assert!(
            methods.starts_with(&["editmessagetext".to_string(), "editmessagetext".to_string(),]),
            "HTML and plain first-part edits should fail before fallback; methods={methods:?}"
        );
    }

    #[tokio::test]
    async fn send_final_flag_on_part1_non429_falls_back_to_attachment_without_orphan_chunks() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(r"(?i)/editmessagetext$"))
            .respond_with(ResponseTemplate::new(400).set_body_json(serde_json::json!({
                "ok": false,
                "error_code": 400,
                "description": "Bad Request: message to edit not found"
            })))
            .up_to_n_times(2)
            .expect(2)
            .with_priority(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path_regex(r"(?i)/senddocument$"))
            .respond_with(ok_document_response())
            .expect(1)
            .with_priority(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .respond_with(ok_message_response("ok"))
            .with_priority(10)
            .mount(&server)
            .await;

        let unique = unique_test_id();
        let source = "Ж".repeat(12_000);
        let view = view_with_response_branch(&source);
        let html = view.render_final_with_long_answer_fix(true);
        assert!(
            naked_tg::markup::telegram_html_text_utf16_units(&html) > MAX_TG_MSG as u64,
            "fixture must route through flag-on split delivery"
        );
        let bot = mock_bot(&server.uri());
        crate::streaming::flush::send_final(
            bot,
            crate::shared::ChatCtx {
                chat_id: ChatId(990_500 + unique as i64),
                thread_id: None,
                reply_to: None,
            },
            MessageId(123),
            &html,
            &view,
            true,
        )
        .await;

        let received = server.received_requests().await.unwrap();
        let methods = request_method_names(&received);
        assert_eq!(
            methods,
            vec![
                "editmessagetext".to_string(),
                "editmessagetext".to_string(),
                "editmessagetext".to_string(),
                "senddocument".to_string(),
            ],
            "part-1 failure must do HTML edit, plain edit, summary edit, attachment — and no orphan chunk sends"
        );
    }

    #[cfg(not(debug_assertions))]
    #[derive(Clone)]
    struct SharedFlushLog(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

    #[cfg(not(debug_assertions))]
    impl std::io::Write for SharedFlushLog {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().expect("log lock").extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[cfg(not(debug_assertions))]
    #[tokio::test(flavor = "current_thread")]
    async fn send_final_release_over_budget_payload_degrades_to_attachment_without_panic() {
        let log_bytes = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let make_writer = {
            let log_bytes = log_bytes.clone();
            move || SharedFlushLog(log_bytes.clone())
        };
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::WARN)
            .with_ansi(false)
            .with_writer(make_writer)
            .finish();
        let _subscriber_guard = tracing::subscriber::set_default(subscriber);

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(r"(?i)/senddocument$"))
            .respond_with(ok_document_response())
            .expect(1)
            .with_priority(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .respond_with(ok_message_response("ok"))
            .with_priority(10)
            .mount(&server)
            .await;

        let unique = unique_test_id();
        let html = "A".repeat(MAX_TG_MSG + 1);
        let view = view_with_response_branch(&html);
        let before = crate::metrics::snapshot();
        let bot = mock_bot(&server.uri());
        crate::streaming::flush::send_final_with_budget_for_test(
            bot,
            crate::shared::ChatCtx {
                chat_id: ChatId(990_600 + unique as i64),
                thread_id: None,
                reply_to: None,
            },
            MessageId(123),
            &html,
            &view,
            crate::streaming::flush::FinalDeliveryBudget {
                dropped: 0,
                utf16_units: 1,
                long_answer_fix_enabled: true,
            },
        )
        .await;
        let after = crate::metrics::snapshot();
        assert!(
            after.final_answer_delivery_partial >= before.final_answer_delivery_partial + 1,
            "over-budget release degradation must bump delivery_total{{outcome=\"partial\"}}"
        );

        let received = server.received_requests().await.unwrap();
        let methods = request_method_names(&received);
        assert_eq!(
            methods,
            vec!["editmessagetext".to_string(), "senddocument".to_string()],
            "over-budget payload must skip the unsafe final send and preserve content by attachment"
        );
        let logs = String::from_utf8(log_bytes.lock().expect("log lock").clone()).unwrap();
        assert!(
            logs.contains("final answer payload exceeds Telegram UTF-16 budget"),
            "release degrade path must WARN with lengths-only context; logs={logs}"
        );
    }

    #[tokio::test]
    async fn send_final_part1_429_retries_without_attachment_fallback() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(r"(?i)/editmessagetext$"))
            .respond_with(ResponseTemplate::new(429).set_body_json(serde_json::json!({
                "ok": false,
                "error_code": 429,
                "description": "Too Many Requests: retry after 1",
                "parameters": {"retry_after": 1}
            })))
            .up_to_n_times(1)
            .expect(1)
            .with_priority(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path_regex(r"(?i)/editmessagetext$"))
            .respond_with(ok_message_response("edited"))
            .with_priority(2)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .respond_with(ok_message_response("ok"))
            .with_priority(10)
            .mount(&server)
            .await;

        let unique = unique_test_id();
        let (html, view) = medium_final_fixture();
        let bot = mock_bot(&server.uri());
        crate::streaming::flush::send_final(
            bot,
            crate::shared::ChatCtx {
                chat_id: ChatId(991_000 + unique as i64),
                thread_id: None,
                reply_to: None,
            },
            MessageId(124),
            &html,
            &view,
            false,
        )
        .await;

        let received = server.received_requests().await.unwrap();
        let methods = request_method_names(&received);
        assert!(
            methods
                .iter()
                .filter(|m| m.as_str() == "editmessagetext")
                .count()
                >= 2,
            "429 must be retried by edit_must_deliver; methods={methods:?}"
        );
        assert!(
            methods.iter().any(|m| m == "sendmessage"),
            "after 429 retry succeeds, normal chunk delivery must continue; methods={methods:?}"
        );
        assert!(
            !methods.iter().any(|m| m == "senddocument"),
            "429 must not trigger premature attachment fallback; methods={methods:?}"
        );
    }

    #[tokio::test]
    async fn send_final_flag_on_cyrillic_4000_edits_whole_without_truncation_delta() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ok_message_response("ok"))
            .mount(&server)
            .await;

        let unique = unique_test_id();
        let source = "Ж".repeat(4_000);
        let view = view_with_response_branch(&source);
        let html = view.render_final_with_long_answer_fix(true);
        assert_eq!(naked_tg::markup::telegram_html_visible_text(&html), source);
        assert!(
            naked_tg::markup::telegram_html_text_utf16_units(&html) <= MAX_TG_MSG as u64,
            "4000 Cyrillic scalars are under Telegram's UTF-16 limit"
        );
        let before_truncated = crate::metrics::snapshot().final_answer_truncated;
        let bot = mock_bot(&server.uri());
        crate::streaming::flush::send_final(
            bot,
            crate::shared::ChatCtx {
                chat_id: ChatId(992_000 + unique as i64),
                thread_id: None,
                reply_to: None,
            },
            MessageId(125),
            &html,
            &view,
            true,
        )
        .await;
        let after_truncated = crate::metrics::snapshot().final_answer_truncated;
        assert_eq!(
            after_truncated - before_truncated,
            0,
            "flag-on long Cyrillic delivery must not increment truncation counter"
        );

        let received = server.received_requests().await.unwrap();
        let methods = request_method_names(&received);
        assert_eq!(methods, vec!["editmessagetext".to_string()]);
        let texts = ordered_text_payloads(&received);
        assert_eq!(texts.len(), 1, "expected one edited payload");
        assert_eq!(
            naked_tg::markup::telegram_html_visible_text(&texts[0]),
            source
        );
    }

    #[tokio::test]
    async fn send_final_flag_on_12000_char_answer_sends_ordered_visible_parts() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ok_message_response("ok"))
            .mount(&server)
            .await;

        let unique = unique_test_id();
        let source = "Ж".repeat(12_000);
        let view = view_with_response_branch(&source);
        let html = view.render_final_with_long_answer_fix(true);
        assert_eq!(naked_tg::markup::telegram_html_visible_text(&html), source);
        assert!(
            naked_tg::markup::telegram_html_text_utf16_units(&html) > MAX_TG_MSG as u64,
            "12000 UTF-16 units must route away from the single edit path"
        );
        let before_truncated = crate::metrics::snapshot().final_answer_truncated;
        let bot = mock_bot(&server.uri());
        crate::streaming::flush::send_final(
            bot,
            crate::shared::ChatCtx {
                chat_id: ChatId(993_000 + unique as i64),
                thread_id: None,
                reply_to: None,
            },
            MessageId(126),
            &html,
            &view,
            true,
        )
        .await;
        let after_truncated = crate::metrics::snapshot().final_answer_truncated;
        assert_eq!(
            after_truncated - before_truncated,
            0,
            "ordered split delivery must not increment truncation counter"
        );

        let received = server.received_requests().await.unwrap();
        let methods = request_method_names(&received);
        assert_eq!(
            methods
                .iter()
                .filter(|m| m.as_str() == "editmessagetext")
                .count(),
            1,
            "first part edits the stream placeholder; methods={methods:?}"
        );
        assert!(
            methods
                .iter()
                .filter(|m| m.as_str() == "sendmessage")
                .count()
                >= 2,
            "remaining ordered parts must be sent as messages; methods={methods:?}"
        );
        assert!(
            !methods.iter().any(|m| m == "senddocument"),
            "12000-char answer should split, not fall back to attachment; methods={methods:?}"
        );
        let texts = ordered_text_payloads(&received);
        assert!(
            texts.len() >= 3,
            "expected multi-part payloads; methods={methods:?}"
        );
        for text in &texts {
            let units = naked_tg::markup::telegram_html_text_utf16_units(text);
            assert!(
                units <= MAX_TG_MSG as u64,
                "pre-send budget invariant violated in captured payload: {units}"
            );
        }
        let visible_joined = texts
            .iter()
            .map(|text| naked_tg::markup::telegram_html_visible_text(text))
            .collect::<String>();
        assert_eq!(visible_joined, source);
    }

    // ─── BUG_REGISTRY D-VALIDATE-IP-TOKENS (B37 stream guard) ───
    //
    // B43 (выявлен 2026-05-13): все 4 теста работают с shared mutable static
    // `NOVNC_IP_ALLOWLIST` и global `IP_TOKEN_HALLUCINATION_COUNT`. При параллельном
    // запуске (cargo test default) это вызывает race: test A очищает allowlist,
    // test B видит empty и идёт по fail-open пути. Сериализуем через
    // std::sync::Mutex (без новых deps; serial_test не используем).
    fn ip_test_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|poison| poison.into_inner())
    }

    /// No noVNC keyword → no-op, counter unchanged.
    #[test]
    fn validate_ip_tokens_skips_when_no_vnc_keyword() {
        let _guard = ip_test_lock();
        let before = naked_core::types::IP_TOKEN_HALLUCINATION_COUNT
            .load(std::sync::atomic::Ordering::Relaxed);
        crate::streaming::flush::validate_novnc_ip_tokens(
            "some response with random IP 80.65.225.177:6080 but no keyword",
        );
        let after = naked_core::types::IP_TOKEN_HALLUCINATION_COUNT
            .load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(after, before, "no vnc keyword → no counter bump");
    }

    /// Allow-list empty → fail-open, no warnings.
    #[test]
    fn validate_ip_tokens_fail_open_when_allowlist_empty() {
        let _guard = ip_test_lock();
        {
            let mut g = crate::shared::NOVNC_IP_ALLOWLIST.write().unwrap();
            g.clear();
        }
        let before = naked_core::types::IP_TOKEN_HALLUCINATION_COUNT
            .load(std::sync::atomic::Ordering::Relaxed);
        crate::streaming::flush::validate_novnc_ip_tokens(
            "open noVNC at http://1.2.3.4:6080/vnc.html",
        );
        let after = naked_core::types::IP_TOKEN_HALLUCINATION_COUNT
            .load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(after, before, "empty allow-list → fail-open");
    }

    /// Allow-list populated + hallucinated IP → counter bumped.
    #[test]
    fn validate_ip_tokens_detects_hallucination() {
        let _guard = ip_test_lock();
        {
            let mut g = crate::shared::NOVNC_IP_ALLOWLIST.write().unwrap();
            g.clear();
            g.push("65.108.226.226:6080".into());
            g.push("100.80.12.120:6080".into());
            g.push("clipshot.cc:443".into());
        }
        let before = naked_core::types::IP_TOKEN_HALLUCINATION_COUNT
            .load(std::sync::atomic::Ordering::Relaxed);
        // B37 reproduction: hallucinated 80.65.225.177:6080.
        crate::streaming::flush::validate_novnc_ip_tokens(
            "Открывай noVNC: http://80.65.225.177:6080/vnc.html?autoconnect=true",
        );
        let after = naked_core::types::IP_TOKEN_HALLUCINATION_COUNT
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(
            after > before,
            "hallucinated IP should bump counter: before={before} after={after}"
        );
        crate::shared::NOVNC_IP_ALLOWLIST.write().unwrap().clear();
    }

    /// Allow-list populated + known good IP → no counter bump.
    #[test]
    fn validate_ip_tokens_passes_known_good_ip() {
        let _guard = ip_test_lock();
        {
            let mut g = crate::shared::NOVNC_IP_ALLOWLIST.write().unwrap();
            g.clear();
            g.push("100.80.12.120:6080".into());
        }
        let before = naked_core::types::IP_TOKEN_HALLUCINATION_COUNT
            .load(std::sync::atomic::Ordering::Relaxed);
        crate::streaming::flush::validate_novnc_ip_tokens(
            "noVNC: http://100.80.12.120:6080/vnc.html",
        );
        let after = naked_core::types::IP_TOKEN_HALLUCINATION_COUNT
            .load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(after, before, "known-good IP → no bump");
        crate::shared::NOVNC_IP_ALLOWLIST.write().unwrap().clear();
    }

    /// T12 — clipshot.cc canonical URLs (incl. /debug/cdp/ TLS CDP proxy)
    /// must NOT trigger a hallucination warning. The validator already
    /// short-circuits on no-IP-pattern, but we add explicit coverage
    /// because bot's chronological event log frequently emits these URLs
    /// (e.g. `✍️ tools: cdp_dump_cookies via https://clipshot.cc/debug/cdp/...`).
    #[test]
    fn validate_ip_tokens_accepts_clipshot_canonical_urls() {
        let _guard = ip_test_lock();
        {
            let mut g = crate::shared::NOVNC_IP_ALLOWLIST.write().unwrap();
            g.clear();
            g.push("clipshot.cc:443".into());
            g.push("100.80.12.120:6080".into());
        }
        let before = naked_core::types::IP_TOKEN_HALLUCINATION_COUNT
            .load(std::sync::atomic::Ordering::Relaxed);
        // All 3 canonical entry points emitted by post-B53 code paths.
        // None contain raw IPv4 → validator's IP scanner finds nothing.
        crate::streaming::flush::validate_novnc_ip_tokens(
            "noVNC: https://clipshot.cc/vnc \
             cdp: https://clipshot.cc/debug/cdp/json/version \
             web: https://clipshot.cc/debug/vnc/vnc.html?autoconnect=true",
        );
        let after = naked_core::types::IP_TOKEN_HALLUCINATION_COUNT
            .load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(
            after, before,
            "clipshot.cc canonical URLs must not trigger B37 (no raw IPs)"
        );
        crate::shared::NOVNC_IP_ALLOWLIST.write().unwrap().clear();
    }

    /// BUG_REGISTRY D-INV-STREAM-BUBBLE-COUNT (B02 / INV-4 full version):
    /// stream-start MUST send exactly ONE Telegram API request, not
    /// two (the old `⏳` placeholder + `⏯️` control card pattern).
    /// Uses the same wiremock infra as send_text_hits_mock_server_with_sendmessage.
    #[tokio::test]
    async fn single_run_behavior_b02_placeholder_guard_stays_green() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "ok": true,
                "result": {
                    "message_id": 42,
                    "date": 0,
                    "chat": {"id": 1, "type": "private", "first_name": "u"},
                    "text": "\u{23F3}"
                }
            })))
            .mount(&server)
            .await;

        let bot = mock_bot(&server.uri());
        let ctx = crate::shared::ChatCtx {
            chat_id: ChatId(1),
            thread_id: None,
            reply_to: None,
        };
        let _ = crate::streaming::pipeline::send_stream_placeholder(&bot, &ctx, "run-b02").await;

        let received = server.received_requests().await.unwrap();
        assert_eq!(
            received.len(),
            1,
            "Step 2 must preserve B02 single-placeholder behavior; got {} requests",
            received.len()
        );
    }

    #[tokio::test]
    async fn mockbot_three_runs_same_thread_three_bubbles_fourth_rejected() {
        use naked_core::types::SteerMessage;
        use naked_tg::run_registry::{
            ChatThreadKey, MULTI_RUN_THREAD_CAP, MessageKey, RegisterRunInput, RegisterRunOptions,
            RunKind, RunOrigin,
        };
        use tokio::sync::mpsc;
        use tokio_util::sync::CancellationToken;

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "ok": true,
                "result": {
                    "message_id": 42,
                    "date": 0,
                    "chat": {"id": 901, "type": "private", "first_name": "u"},
                    "text": "\u{23F3}"
                }
            })))
            .mount(&server)
            .await;

        let unique = unique_test_id();
        let chat_id = 901_000 + unique as i64;
        let bot = mock_bot(&server.uri());
        let ctx = crate::shared::ChatCtx {
            chat_id: ChatId(chat_id),
            thread_id: None,
            reply_to: None,
        };
        let registry = &crate::shared::RUN_REGISTRY;
        let thread = ChatThreadKey::new(chat_id, None);
        for idx in 0..=MULTI_RUN_THREAD_CAP {
            let _ = registry.remove_run(&format!("cap-l3-run-{unique}-{idx}"));
        }

        for idx in 0..MULTI_RUN_THREAD_CAP {
            let run_id = format!("cap-l3-run-{unique}-{idx}");
            let session_id = format!("cap-l3-sid-{unique}-{idx}");
            let (steer, _rx) = mpsc::channel::<SteerMessage>(4);
            registry
                .register_run(
                    RegisterRunInput {
                        requested_run_id: Some(run_id.clone()),
                        session_id,
                        origin: RunOrigin::new(chat_id, None),
                        kind: RunKind::ChatTurn,
                        source_ref: None,
                        steer,
                        abort: CancellationToken::new(),
                    },
                    RegisterRunOptions::cap_three(),
                )
                .expect("first three runs are admitted with flag=true/cap=3");
            let _ = crate::streaming::pipeline::send_stream_placeholder(&bot, &ctx, &run_id).await;
            registry
                .bind_message(&run_id, MessageKey::new(chat_id, 1000 + idx as i32))
                .expect("each admitted run gets its own bound bubble");
        }

        let (events_tx, events_rx) = tokio::sync::mpsc::channel(1);
        drop(events_tx);
        let (perm_tx, _perm_rx) = tokio::sync::mpsc::channel(1);
        let (steer_tx, _steer_rx) = tokio::sync::mpsc::channel(1);
        let mut config = naked_core::config::Config {
            run_registry_multi_stream_enabled: true,
            ..Default::default()
        };
        let tmp = tempfile::tempdir().unwrap();
        config.workspace = tmp.path().join("workspace");
        config.session_dir = tmp.path().join("sessions");
        std::fs::create_dir_all(&config.workspace).unwrap();
        std::fs::create_dir_all(&config.session_dir).unwrap();
        let agent = std::sync::Arc::new(naked_core::AgentCore::new(
            config.clone(),
            Box::new(NoopProvider),
        ));
        let deps = crate::message_handler::BotDeps {
            bot: bot.clone(),
            agent,
            channel_map: std::sync::Arc::new(ChannelSessionMap::new()),
            config,
            pending_perms: std::sync::Arc::new(tokio::sync::RwLock::new(
                std::collections::HashMap::new(),
            )),
            http_client: std::sync::Arc::new(reqwest::Client::new()),
            base_url: std::sync::Arc::new(server.uri()),
            rate_limiter: naked_tg::rate_limit::RateLimiter::new(),
            attribution_flag: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            bot_token: std::sync::Arc::new("0:TEST_TOKEN".to_string()),
            bot_identity: std::sync::Arc::new(naked_tg::bot_identity::BotIdentity {
                id: 0,
                username: "test_bot".to_string(),
            }),
            tg_attach_queue: naked_tg::tg_attach::new_queue(),
            research_scheduler: None,
            per_chat_locks: std::sync::Arc::new(crate::per_chat_locks::PerChatLocks::new()),
        };
        let _outcome = crate::streaming::pipeline::stream_response(
            &deps,
            ctx,
            naked_core::types::AgentHandle {
                events: events_rx,
                permissions: perm_tx,
                steer: steer_tx,
                abort: CancellationToken::new(),
            },
            "test-model".to_string(),
            crate::streaming::pipeline::StreamRunContext {
                requested_run_id: Some(format!("cap-l3-run-{unique}-3")),
                session_id: format!("cap-l3-sid-{unique}-3"),
                kind: RunKind::ChatTurn,
                source_ref: None,
                final_reply_markup: None,
            },
        )
        .await;

        let received = server.received_requests().await.unwrap();
        assert_eq!(
            received.len(),
            4,
            "4th rejected run must send exactly one friendly rejection and NO placeholder; got {} TG requests",
            received.len()
        );
        let fourth_body = std::str::from_utf8(&received[3].body).unwrap_or("");
        assert!(
            fourth_body.contains("3 runs already active in this thread"),
            "4th request must be the production friendly cap-reject message, body={fourth_body}"
        );
        assert!(
            !fourth_body.contains("thinking") && !fourth_body.contains("reply_markup"),
            "4th rejected run must not send a placeholder/control bubble, body={fourth_body}"
        );
        let summaries = registry.list_for_thread(thread);
        assert_eq!(summaries.len(), 3);
        assert_eq!(
            summaries
                .iter()
                .filter(|s| s.run_id.starts_with(&format!("cap-l3-run-{unique}-")))
                .filter_map(|s| s.bubble_message_id)
                .collect::<std::collections::HashSet<_>>()
                .len(),
            3,
            "the 3 admitted runs must have 3 distinct bound bubble ids"
        );
        for idx in 0..=MULTI_RUN_THREAD_CAP {
            let _ = registry.remove_run(&format!("cap-l3-run-{unique}-{idx}"));
        }
    }

    #[tokio::test]
    async fn move_rehome_channel_session_map_set_b() {
        use naked_core::types::SteerMessage;
        use naked_tg::run_registry::{
            MessageKey, RegisterRunInput, RegisterRunOptions, RunKind, RunOrigin,
        };
        use tokio::sync::mpsc;
        use tokio_util::sync::CancellationToken;

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "ok": true,
                "result": {
                    "message_id": 77,
                    "date": 0,
                    "chat": {"id": 920, "type": "private", "first_name": "u"},
                    "text": "ok"
                }
            })))
            .mount(&server)
            .await;
        let unique = unique_test_id();
        let chat_a = 910_000 + unique as i64;
        let chat_b = 920_000 + unique as i64;
        let bot = mock_bot(&server.uri());
        let channel_map = std::sync::Arc::new(ChannelSessionMap::new());
        let deps = test_bot_deps(bot, test_config(), channel_map.clone(), server.uri());
        let run_id = format!("move-map-run-{unique}");
        let session_id = format!("move-map-sid-{unique}");
        let _ = crate::shared::RUN_REGISTRY.remove_run(&run_id);
        let (steer, _rx) = mpsc::channel::<SteerMessage>(4);
        crate::shared::RUN_REGISTRY
            .register_run(
                RegisterRunInput {
                    requested_run_id: Some(run_id.clone()),
                    session_id: session_id.clone(),
                    origin: RunOrigin::new(chat_a, None),
                    kind: RunKind::ChatTurn,
                    source_ref: None,
                    steer,
                    abort: CancellationToken::new(),
                },
                RegisterRunOptions::cap_three(),
            )
            .unwrap();
        crate::shared::RUN_REGISTRY
            .bind_message(&run_id, MessageKey::new(chat_a, 42))
            .unwrap();
        let ctx_b = crate::shared::ChatCtx {
            chat_id: ChatId(chat_b),
            thread_id: None,
            reply_to: None,
        };
        crate::commands::research::move_live_run_here(&deps, &ctx_b, &format!("{run_id} here"))
            .await
            .unwrap();
        assert_eq!(
            channel_map.get(chat_b, None).await.as_deref(),
            Some(session_id.as_str())
        );
        assert_eq!(
            crate::shared::RUN_REGISTRY
                .primary_sink(&run_id)
                .expect("new B bubble bound")
                .chat_id,
            chat_b
        );
        let _ = crate::shared::RUN_REGISTRY.remove_run(&run_id);
    }

    #[tokio::test]
    async fn move_cap_rejected_destination_bubble_has_no_live_controls() {
        use naked_core::types::SteerMessage;
        use naked_tg::run_registry::{
            MessageKey, RegisterRunInput, RegisterRunOptions, RunKind, RunOrigin,
        };
        use tokio::sync::mpsc;
        use tokio_util::sync::CancellationToken;

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "ok": true,
                "result": {
                    "message_id": 77,
                    "date": 0,
                    "chat": {"id": 1, "type": "private", "first_name": "u"},
                    "text": "ok"
                }
            })))
            .mount(&server)
            .await;

        let unique = unique_test_id();
        let chat_a = 930_000 + unique as i64;
        let chat_b = 940_000 + unique as i64;
        let run_id = format!("move-cap-reject-run-{unique}");
        let session_id = format!("move-cap-reject-sid-{unique}");
        let registry = &crate::shared::RUN_REGISTRY;
        let all_run_ids: Vec<String> = std::iter::once(run_id.clone())
            .chain((0..3).map(|idx| format!("move-cap-dest-{unique}-{idx}")))
            .collect();
        for id in &all_run_ids {
            let _ = registry.remove_run(id);
        }

        let (steer, _rx) = mpsc::channel::<SteerMessage>(4);
        registry
            .register_run(
                RegisterRunInput {
                    requested_run_id: Some(run_id.clone()),
                    session_id: session_id.clone(),
                    origin: RunOrigin::new(chat_a, None),
                    kind: RunKind::ChatTurn,
                    source_ref: None,
                    steer,
                    abort: CancellationToken::new(),
                },
                RegisterRunOptions::cap_three(),
            )
            .unwrap();
        registry
            .bind_message(&run_id, MessageKey::new(chat_a, 42))
            .unwrap();

        for idx in 0..3 {
            let dest_run_id = format!("move-cap-dest-{unique}-{idx}");
            let dest_session_id = format!("move-cap-dest-sid-{unique}-{idx}");
            let (steer, _rx) = mpsc::channel::<SteerMessage>(4);
            registry
                .register_run(
                    RegisterRunInput {
                        requested_run_id: Some(dest_run_id),
                        session_id: dest_session_id,
                        origin: RunOrigin::new(chat_b, None),
                        kind: RunKind::ChatTurn,
                        source_ref: None,
                        steer,
                        abort: CancellationToken::new(),
                    },
                    RegisterRunOptions::cap_three(),
                )
                .unwrap();
        }

        let bot = mock_bot(&server.uri());
        let channel_map = std::sync::Arc::new(ChannelSessionMap::new());
        let deps = test_bot_deps(bot, test_config(), channel_map, server.uri());
        let ctx_b = crate::shared::ChatCtx {
            chat_id: ChatId(chat_b),
            thread_id: None,
            reply_to: None,
        };
        crate::commands::research::move_live_run_here(&deps, &ctx_b, &format!("{run_id} here"))
            .await
            .unwrap();

        assert_eq!(
            registry
                .primary_sink(&run_id)
                .expect("run remains at A")
                .chat_id,
            chat_a,
            "cap-rejected move must leave the run at its old origin"
        );
        let received = server.received_requests().await.unwrap();
        let bodies: Vec<String> = received
            .iter()
            .map(|req| String::from_utf8_lossy(&req.body).to_string())
            .collect();
        assert!(
            bodies
                .iter()
                .any(|body| body.contains("destination already has")),
            "must edit the markup-less destination message into friendly rejection; bodies={bodies:#?}"
        );
        assert!(
            bodies.iter().all(|body| {
                !body.contains("s:abort:")
                    && !body.contains("s:sendnow:")
                    && !body.contains("reply_markup")
                    && !body.contains("inline_keyboard")
            }),
            "cap-rejected move must never expose live controls on the destination bubble; bodies={bodies:#?}"
        );

        for id in all_run_ids {
            let _ = registry.remove_run(&id);
        }
    }

    #[tokio::test]
    async fn move_mockbot_freezes_a_and_continues_b() {
        use naked_core::types::{AgentEvent, AgentHandle};
        use naked_tg::run_registry::RunKind;
        use tokio::sync::mpsc;
        use tokio_util::sync::CancellationToken;

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "ok": true,
                "result": {
                    "message_id": 42,
                    "date": 0,
                    "chat": {"id": 910, "type": "private", "first_name": "u"},
                    "text": "ok"
                }
            })))
            .mount(&server)
            .await;

        let unique = unique_test_id();
        let chat_a = 910_000 + unique as i64;
        let chat_b = 920_000 + unique as i64;
        let run_id = format!("move-l3-run-{unique}");
        let session_id = format!("move-l3-sid-{unique}");
        let bot = mock_bot(&server.uri());
        let channel_map = std::sync::Arc::new(ChannelSessionMap::new());
        let config = test_config();
        let deps = test_bot_deps(bot.clone(), config, channel_map, server.uri());
        let _ = crate::shared::RUN_REGISTRY.remove_run(&run_id);
        let (events_tx, events_rx) = mpsc::channel(8);
        let (perm_tx, _perm_rx) = mpsc::channel(1);
        let (steer_tx, _steer_rx) = mpsc::channel(1);
        let ctx_a = crate::shared::ChatCtx {
            chat_id: ChatId(chat_a),
            thread_id: None,
            reply_to: None,
        };
        let run_id_for_stream = run_id.clone();
        let session_id_for_stream = session_id.clone();
        let stream_task = tokio::spawn(async move {
            crate::streaming::pipeline::stream_response(
                &deps,
                ctx_a,
                AgentHandle {
                    events: events_rx,
                    permissions: perm_tx,
                    steer: steer_tx,
                    abort: CancellationToken::new(),
                },
                "test-model".to_string(),
                crate::streaming::pipeline::StreamRunContext {
                    requested_run_id: Some(run_id_for_stream),
                    session_id: session_id_for_stream,
                    kind: RunKind::ChatTurn,
                    source_ref: None,
                    final_reply_markup: None,
                },
            )
            .await
        });
        for _ in 0..50 {
            if crate::shared::RUN_REGISTRY
                .primary_sink(&run_id)
                .is_some_and(|sink| sink.chat_id == chat_a)
            {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        assert!(crate::shared::RUN_REGISTRY.primary_sink(&run_id).is_some());

        let move_bot = mock_bot(&server.uri());
        let move_channel_map = std::sync::Arc::new(ChannelSessionMap::new());
        let move_deps = test_bot_deps(move_bot, test_config(), move_channel_map, server.uri());
        let ctx_b = crate::shared::ChatCtx {
            chat_id: ChatId(chat_b),
            thread_id: None,
            reply_to: None,
        };
        crate::commands::research::move_live_run_here(
            &move_deps,
            &ctx_b,
            &format!("{run_id} here"),
        )
        .await
        .unwrap();
        assert_eq!(
            crate::shared::RUN_REGISTRY
                .primary_sink(&run_id)
                .expect("new B primary")
                .chat_id,
            chat_b
        );

        events_tx
            .send(AgentEvent::TextDelta("after move proof".to_string()))
            .await
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(
            naked_tg::rate_limit::MIN_GAP_MS + 250,
        ))
        .await;
        events_tx.send(AgentEvent::Idle).await.unwrap();
        let _ = stream_task.await.unwrap();

        let received = server.received_requests().await.unwrap();
        let bodies: Vec<String> = received
            .iter()
            .map(|req| String::from_utf8_lossy(&req.body).to_string())
            .collect();
        assert!(
            bodies.iter().any(|body| body.contains(&chat_a.to_string())
                && body.contains(&format!("moved to chat {chat_b}"))),
            "old A bubble must be frozen with moved note; bodies={bodies:#?}"
        );
        assert!(
            bodies
                .iter()
                .any(|body| body.contains(&chat_b.to_string()) && body.contains("after move proof")),
            "new B bubble must receive subsequent stream edits/final; bodies={bodies:#?}"
        );
        assert!(
            !bodies
                .iter()
                .any(|body| body.contains(&chat_a.to_string()) && body.contains("after move proof")),
            "old A bubble must not receive post-move stream text; bodies={bodies:#?}"
        );
        let _ = crate::shared::RUN_REGISTRY.remove_run(&run_id);
    }

    #[tokio::test]
    async fn stream_start_sends_exactly_one_message_with_inline_kbd() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "ok": true,
                "result": {
                    "message_id": 42,
                    "date": 0,
                    "chat": {"id": 1, "type": "private", "first_name": "u"},
                    "text": "\u{23F3}"
                }
            })))
            .mount(&server)
            .await;

        let bot = mock_bot(&server.uri());
        let ctx = crate::shared::ChatCtx {
            chat_id: ChatId(1),
            thread_id: None,
            reply_to: None,
        };
        let _ = crate::streaming::pipeline::send_stream_placeholder(&bot, &ctx, "run-start").await;

        let received = server.received_requests().await.unwrap();
        assert_eq!(
            received.len(),
            1,
            "INV-4: stream-start must send exactly ONE TG API request \
             (placeholder with inline keyboard); got {}",
            received.len()
        );

        // Verify the single request carries inline keyboard.
        let req = &received[0];
        let body = std::str::from_utf8(&req.body).unwrap_or("");
        assert!(
            body.contains("reply_markup") || body.contains("inline_keyboard"),
            "stream-start request must carry inline keyboard: body={body}"
        );
        assert!(
            body.contains("s:abort:run-start") && body.contains("s:sendnow:run-start"),
            "new run-bound placeholder must emit s:* scoped callbacks immediately: body={body}"
        );
        assert!(
            !body.contains("stream:abort") && !body.contains("stream:sendnow"),
            "new run-bound placeholder must not emit legacy stream:* callbacks: body={body}"
        );
    }

    #[tokio::test]
    async fn stream_start_pipeline_bounds_send_message_count() {
        // BUG_REGISTRY B02 / D-INV-STREAM-BUBBLE-COUNT (INV-4):
        // at stream-start the pipeline must send AT MOST 2 new bubbles
        // (media-ack + placeholder). B02 was "stream-start sends 3 messages
        // where 1 suffices". Unlike stream_start_sends_exactly_one_message_
        // with_inline_kbd (which exercises the send_stream_placeholder HELPER
        // in isolation), this drives the REAL crate::streaming::pipeline::
        // stream_response pipeline and counts /sendMessage API calls.
        //
        // ASSERTION BOUND RATIONALE: this scenario has NO media, so the
        // media-ack bubble is never sent and the correct baseline is exactly
        // ONE /sendMessage (the placeholder). The issue's "<= 2" bound would
        // NOT catch a single injected extra bubble (1 -> 2 still satisfies
        // <= 2). Therefore we bound the no-media stream-start at <= 1, which
        // (a) passes on current code and (b) FAILS the moment an extra
        // stream-start send is injected. The <= 2 media-ack allowance is
        // documented but does not apply here (no media in this run).
        use naked_core::types::{AgentEvent, AgentHandle};
        use naked_tg::run_registry::RunKind;
        use tokio::sync::mpsc;
        use tokio_util::sync::CancellationToken;

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "ok": true,
                "result": {
                    "message_id": 42,
                    "date": 0,
                    "chat": {"id": 1, "type": "private", "first_name": "u"},
                    "text": "ok"
                }
            })))
            .mount(&server)
            .await;

        let unique = unique_test_id();
        let run_id = format!("b02-bubble-count-{unique}");
        let chat_id = 970_000 + unique as i64;

        let bot = mock_bot(&server.uri());
        let deps = test_bot_deps(
            bot,
            test_config(),
            std::sync::Arc::new(ChannelSessionMap::new()),
            server.uri(),
        );
        let (events_tx, events_rx) = mpsc::channel(4);
        let (perm_tx, _perm_rx) = mpsc::channel(1);
        let (steer_tx, _steer_rx) = mpsc::channel(1);
        let session_id = format!("{run_id}-sid");
        events_tx
            .send(AgentEvent::TextDelta("hello".to_string()))
            .await
            .unwrap();
        events_tx.send(AgentEvent::Idle).await.unwrap();

        let _outcome = crate::streaming::pipeline::stream_response(
            &deps,
            crate::shared::ChatCtx {
                chat_id: ChatId(chat_id),
                thread_id: None,
                reply_to: None,
            },
            AgentHandle {
                events: events_rx,
                permissions: perm_tx,
                steer: steer_tx,
                abort: CancellationToken::new(),
            },
            "test-model".to_string(),
            crate::streaming::pipeline::StreamRunContext {
                requested_run_id: Some(run_id.clone()),
                session_id,
                kind: RunKind::Research {
                    spec_id: format!("{run_id}-spec"),
                },
                source_ref: Some(format!("{run_id}-spec")),
                final_reply_markup: None,
            },
        )
        .await;
        let _ = crate::shared::RUN_REGISTRY.remove_run(&run_id);

        let received = server.received_requests().await.unwrap();
        // teloxide renders the method segment as `SendMessage`; match
        // case-insensitively so we count new-bubble sends (NOT
        // editMessageText / editMessageReplyMarkup / sendChatAction /
        // sendDocument, which are edits/typing/attachments).
        let send_message_count = received
            .iter()
            .filter(|req| {
                let path = req.url.path().to_ascii_lowercase();
                path.ends_with("/sendmessage")
            })
            .count();

        // No-media baseline is EXACTLY 1 (placeholder). media-ack would make it
        // 2 but there is no media here. Asserting == 1 (not <= 1) catches BOTH a
        // missing placeholder (0) AND an extra stream-start bubble (2+, B02
        // regression).
        assert_eq!(
            send_message_count, 1,
            "B02/INV-4: no-media stream-start must send EXACTLY the placeholder \
             bubble (media-ack would make it 2 but there is no media here, so \
             exactly 1); got {send_message_count} /sendMessage calls"
        );
    }

    #[tokio::test]
    async fn scheduled_completion_attaches_controls_only_for_recurring() {
        use naked_core::types::{AgentEvent, AgentHandle};
        use naked_tg::run_registry::RunKind;
        use teloxide::types::InlineKeyboardMarkup;
        use tokio::sync::mpsc;
        use tokio_util::sync::CancellationToken;

        async fn run_stream(
            server: &MockServer,
            chat_id: i64,
            run_id: String,
            final_reply_markup: Option<InlineKeyboardMarkup>,
        ) -> Vec<String> {
            let bot = mock_bot(&server.uri());
            let deps = test_bot_deps(
                bot,
                test_config(),
                std::sync::Arc::new(ChannelSessionMap::new()),
                server.uri(),
            );
            let (events_tx, events_rx) = mpsc::channel(4);
            let (perm_tx, _perm_rx) = mpsc::channel(1);
            let (steer_tx, _steer_rx) = mpsc::channel(1);
            let session_id = format!("{run_id}-sid");
            events_tx
                .send(AgentEvent::TextDelta("done".to_string()))
                .await
                .unwrap();
            events_tx.send(AgentEvent::Idle).await.unwrap();

            let _outcome = crate::streaming::pipeline::stream_response(
                &deps,
                crate::shared::ChatCtx {
                    chat_id: ChatId(chat_id),
                    thread_id: None,
                    reply_to: None,
                },
                AgentHandle {
                    events: events_rx,
                    permissions: perm_tx,
                    steer: steer_tx,
                    abort: CancellationToken::new(),
                },
                "test-model".to_string(),
                crate::streaming::pipeline::StreamRunContext {
                    requested_run_id: Some(run_id.clone()),
                    session_id,
                    kind: RunKind::Research {
                        spec_id: format!("{run_id}-spec"),
                    },
                    source_ref: Some(format!("{run_id}-spec")),
                    final_reply_markup,
                },
            )
            .await;
            let _ = crate::shared::RUN_REGISTRY.remove_run(&run_id);

            server
                .received_requests()
                .await
                .unwrap()
                .iter()
                .map(|req| String::from_utf8_lossy(&req.body).to_string())
                .collect()
        }

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "ok": true,
                "result": {
                    "message_id": 42,
                    "date": 0,
                    "chat": {"id": 1, "type": "private", "first_name": "u"},
                    "text": "ok"
                }
            })))
            .mount(&server)
            .await;

        let unique = unique_test_id();
        let recurring_run = format!("d6-recurring-run-{unique}");
        let one_shot_run = format!("d6-one-shot-run-{unique}");
        let recurring_bodies = run_stream(
            &server,
            980_000 + unique as i64,
            recurring_run,
            Some(naked_tg::research_controls::keyboard_scheduled(
                "recurring-spec",
            )),
        )
        .await;
        assert!(
            recurring_bodies
                .iter()
                .any(|body| body.contains("reply_markup")
                    && body.contains("r:rm:recurring-spec")
                    && body.contains("r:uns:recurring-spec")),
            "completed recurring run must leave scheduled controls on the bubble; bodies={recurring_bodies:#?}"
        );

        let before_one_shot = recurring_bodies.len();
        let all_bodies = run_stream(&server, 990_000 + unique as i64, one_shot_run, None).await;
        let one_shot_bodies = &all_bodies[before_one_shot..];
        assert!(
            !one_shot_bodies.iter().any(|body| body.contains("r:rm:")
                || body.contains("r:uns:")
                || body.contains("r:sch:")),
            "run with no final markup must only clear live controls, not attach research controls; bodies={one_shot_bodies:#?}"
        );
    }

    #[tokio::test]
    async fn mirror_fanout_consumes_agent_events_once() {
        use naked_core::types::{AgentEvent, AgentHandle};
        use naked_tg::run_registry::RunKind;
        use std::sync::atomic::AtomicBool;
        use tokio::sync::{RwLock, mpsc};
        use tokio_util::sync::CancellationToken;

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "ok": true,
                "result": {
                    "message_id": 42,
                    "date": 0,
                    "chat": {"id": 903, "type": "private", "first_name": "u"},
                    "text": "ok"
                }
            })))
            .mount(&server)
            .await;

        let mut config = naked_core::config::Config {
            run_registry_multi_stream_enabled: true,
            ..Default::default()
        };
        let tmp = tempfile::tempdir().unwrap();
        config.workspace = tmp.path().join("workspace");
        config.session_dir = tmp.path().join("sessions");
        std::fs::create_dir_all(&config.workspace).unwrap();
        std::fs::create_dir_all(&config.session_dir).unwrap();
        let agent = std::sync::Arc::new(naked_core::AgentCore::new(
            config.clone(),
            Box::new(NoopProvider),
        ));
        let bot = mock_bot(&server.uri());
        let deps = crate::message_handler::BotDeps {
            bot: bot.clone(),
            agent,
            channel_map: std::sync::Arc::new(ChannelSessionMap::new()),
            config,
            pending_perms: std::sync::Arc::new(RwLock::new(std::collections::HashMap::new())),
            http_client: std::sync::Arc::new(reqwest::Client::new()),
            base_url: std::sync::Arc::new(server.uri()),
            rate_limiter: naked_tg::rate_limit::RateLimiter::new(),
            attribution_flag: std::sync::Arc::new(AtomicBool::new(false)),
            bot_token: std::sync::Arc::new("0:TEST_TOKEN".to_string()),
            bot_identity: std::sync::Arc::new(naked_tg::bot_identity::BotIdentity {
                id: 0,
                username: "test_bot".to_string(),
            }),
            tg_attach_queue: naked_tg::tg_attach::new_queue(),
            research_scheduler: None,
            per_chat_locks: std::sync::Arc::new(crate::per_chat_locks::PerChatLocks::new()),
        };

        let unique = unique_test_id();
        let chat_home = 903_000 + unique as i64;
        let chat_mirror = 904_000 + unique as i64;
        let run_id = format!("mirror-l3-run-{unique}");
        let session_id = format!("mirror-l3-sid-{unique}");
        let _ = crate::shared::RUN_REGISTRY.remove_run(&run_id);
        let (events_tx, events_rx) = mpsc::channel(8);
        let (perm_tx, _perm_rx) = mpsc::channel(1);
        let (steer_tx, _steer_rx) = mpsc::channel(1);
        let ctx_home = crate::shared::ChatCtx {
            chat_id: ChatId(chat_home),
            thread_id: None,
            reply_to: None,
        };
        let run_id_for_stream = run_id.clone();
        let session_id_for_stream = session_id.clone();
        let stream_task = tokio::spawn(async move {
            crate::streaming::pipeline::stream_response(
                &deps,
                ctx_home,
                AgentHandle {
                    events: events_rx,
                    permissions: perm_tx,
                    steer: steer_tx,
                    abort: CancellationToken::new(),
                },
                "test-model".to_string(),
                crate::streaming::pipeline::StreamRunContext {
                    requested_run_id: Some(run_id_for_stream),
                    session_id: session_id_for_stream,
                    kind: RunKind::ChatTurn,
                    source_ref: None,
                    final_reply_markup: None,
                },
            )
            .await
        });

        for _ in 0..50 {
            if crate::shared::RUN_REGISTRY
                .get_run(&run_id)
                .and_then(|run| run.bubble_message_id)
                .is_some()
            {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        assert!(
            crate::shared::RUN_REGISTRY
                .get_run(&run_id)
                .and_then(|run| run.bubble_message_id)
                .is_some(),
            "home run must be registered and bound before adding mirror"
        );

        let ctx_mirror = crate::shared::ChatCtx {
            chat_id: ChatId(chat_mirror),
            thread_id: None,
            reply_to: None,
        };
        crate::commands::research::send_live_run_snapshot(
            &bot,
            &ctx_mirror,
            &crate::shared::RUN_REGISTRY,
            &run_id,
        )
        .await
        .unwrap();
        assert_eq!(crate::shared::RUN_REGISTRY.mirror_sinks(&run_id).len(), 1);

        events_tx
            .send(AgentEvent::TextDelta("mirror fanout proof".to_string()))
            .await
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(
            naked_tg::rate_limit::MIN_GAP_MS + 250,
        ))
        .await;
        events_tx.send(AgentEvent::Idle).await.unwrap();
        let _ = stream_task.await.unwrap();

        let received = server.received_requests().await.unwrap();
        let bodies: Vec<String> = received
            .iter()
            .map(|req| String::from_utf8_lossy(&req.body).to_string())
            .collect();
        let proof_edits = bodies
            .iter()
            .filter(|body| body.contains("mirror fanout proof"))
            .count();
        assert!(
            proof_edits >= 2,
            "home + mirror must both receive rendered edits/final text; bodies={bodies:#?}"
        );
        assert!(
            bodies
                .iter()
                .any(|body| body.contains(&chat_home.to_string())
                    && body.contains("mirror fanout proof")),
            "home bubble should receive fanout edit/final; bodies={bodies:#?}"
        );
        assert!(
            bodies
                .iter()
                .any(|body| body.contains(&chat_mirror.to_string())
                    && body.contains("mirror fanout proof")),
            "mirror bubble should receive fanout edit/final; bodies={bodies:#?}"
        );

        let pipeline_src = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/streaming_mod/pipeline.rs"
        ))
        .unwrap();
        let fanout = pipeline_src
            .split("async fn flush_live_to_all_sinks")
            .nth(1)
            .and_then(|tail| tail.split("async fn send_final_to_all_sinks").next())
            .expect("fanout helper source present");
        assert_eq!(
            fanout.matches("view.render_live()").count(),
            1,
            "fanout must render once then broadcast"
        );
        assert!(
            !fanout.contains("stream_response(") && !fanout.contains("AgentHandle"),
            "mirror fanout must not start a second stream_response / second AgentHandle consumer"
        );
        let _ = crate::shared::RUN_REGISTRY.remove_run(&run_id);
    }

    #[tokio::test]
    async fn attachments_mockbot_finish_a_sends_only_a_files() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "ok": true,
                "result": {
                    "message_id": 77,
                    "date": 0,
                    "chat": {"id": 902, "type": "private", "first_name": "u"},
                    "document": {"file_id": "doc", "file_unique_id": "uniq", "file_name": "ok.txt"}
                }
            })))
            .mount(&server)
            .await;

        let queue = naked_tg::tg_attach::new_queue();
        let dir = tempfile::tempdir().unwrap();
        let path_a = dir.path().join("run-a-l3-document.txt");
        let path_b = dir.path().join("run-b-l3-document.txt");
        std::fs::write(&path_a, "a-only").unwrap();
        std::fs::write(&path_b, "b-must-not-send").unwrap();

        let tool_a = naked_tg::tg_attach::TelegramAttachTool::new_for_run(queue.clone(), "run-a");
        let tool_b = naked_tg::tg_attach::TelegramAttachTool::new_for_run(queue.clone(), "run-b");
        let cwd = std::path::Path::new("/tmp");
        let result_a = naked_core::tool::Tool::execute(
            &tool_a,
            serde_json::json!({"paths": [path_a.to_str().unwrap()]}),
            cwd,
        )
        .await;
        assert!(!result_a.is_error, "A stage failed: {}", result_a.output);
        let result_b = naked_core::tool::Tool::execute(
            &tool_b,
            serde_json::json!({"paths": [path_b.to_str().unwrap()]}),
            cwd,
        )
        .await;
        assert!(!result_b.is_error, "B stage failed: {}", result_b.output);

        let client = reqwest::Client::new();
        let delivered_a = naked_tg::tg_attach::drain_for_run(&queue, "run-a").await;
        assert_eq!(delivered_a.len(), 1, "run A drain selects only A");
        for att in delivered_a {
            let form = reqwest::multipart::Form::new()
                .text("chat_id", "902")
                .file("document", &att.path)
                .await
                .expect("multipart form");
            client
                .post(format!("{}/sendDocument", server.uri()))
                .multipart(form)
                .send()
                .await
                .expect("mock sendDocument");
        }

        let received = server.received_requests().await.unwrap();
        assert_eq!(
            received.len(),
            1,
            "finishing run A should deliver only A's queued attachment"
        );
        assert!(
            received[0].url.path().ends_with("/sendDocument"),
            "text file should use sendDocument: {}",
            received[0].url.path()
        );
        let body = String::from_utf8_lossy(&received[0].body);
        assert!(
            body.contains("run-a-l3-document.txt"),
            "multipart body should reference A file only: {body}"
        );
        assert!(
            !body.contains("run-b-l3-document.txt") && !body.contains("b-must-not-send"),
            "seeded-fail: a global drain would leak B's attachment into A finalize: {body}"
        );
        let remaining = queue.lock().await;
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].run_id.as_deref(), Some("run-b"));
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

    // ── BUG_REGISTRY D-CB-GATE-REJECT ───────────────────────────────────
    //
    // `handle_callback` (callbacks/mod.rs) rejects callback queries from
    // chats not in `allowed_chat_ids`: it answers "not allowed" and MUST
    // `return` early *before* the `match action` dispatch. The existing
    // `callback_allowed_chat_gate_before_registry_action` test only asserts
    // the *source ordering* (`is_allowed(` appears before `match action`) —
    // a purely textual proxy. Mutation probe confirmed: deleting only the
    // early `return Ok(())` (keeping the gate check + "not allowed" answer)
    // lets non-allowed callbacks fall through and dispatch, yet all 25
    // callback tests stay green. This behavioral test closes that gap by
    // driving the real `handle_callback` entry point and asserting the
    // strongest downstream side-effect is ABSENT: a registered run's abort
    // token stays uncancelled (the `s:abort:<run>` action never runs).
    #[tokio::test]
    async fn non_allowed_chat_callback_rejected_without_dispatch() {
        use teloxide::types::CallbackQuery;
        use tokio::sync::mpsc;
        use tokio_util::sync::CancellationToken;

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "ok": true,
                "result": true
            })))
            .mount(&server)
            .await;

        let unique = unique_test_id();
        let chat = 950_000 + unique as i64;
        // Configure a non-empty allow-list containing only a DIFFERENT chat
        // id, so `chat` is not a member and gets rejected. Using a populated
        // list (rather than an empty one) isolates the "configured allow-list,
        // requesting chat not a member" path: `is_allowed` also denies on an
        // empty list, so an empty list would reject too but for an ambiguous
        // reason.
        let mut config = test_config();
        config.telegram.allowed_chat_ids = vec![chat + 1];

        // A live run owned by `chat`, whose abort token is our tripwire:
        // dispatching `s:abort:<run_id>` would cancel it. Ownership matches
        // `chat`, so ONLY the allowed-chat gate can prevent the abort.
        let run_id = format!("cb-gate-run-{unique}");
        let sid = format!("cb-gate-sid-{unique}");
        let abort = CancellationToken::new();
        let (steer_tx, _steer_rx) = mpsc::channel(1);
        crate::shared::RUN_REGISTRY
            .register_run(
                naked_tg::run_registry::RegisterRunInput {
                    requested_run_id: Some(run_id.clone()),
                    session_id: sid.clone(),
                    origin: naked_tg::run_registry::RunOrigin::new(chat, None),
                    kind: naked_tg::run_registry::RunKind::ChatTurn,
                    source_ref: None,
                    steer: steer_tx,
                    abort: abort.clone(),
                },
                naked_tg::run_registry::RegisterRunOptions::cap_three(),
            )
            .expect("register run for gate-reject test");

        let bot = mock_bot(&server.uri());
        let channel_map = std::sync::Arc::new(ChannelSessionMap::new());
        let deps = test_bot_deps(bot, config, channel_map, server.uri());
        let pending_perms: crate::shared::PendingPermissions =
            std::sync::Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new()));

        // Callback from the non-allowed `chat` carrying an action that WOULD
        // abort the run if it reached the dispatcher.
        let q: CallbackQuery = serde_json::from_value(serde_json::json!({
            "id": format!("cb-gate-{unique}"),
            "from": {"id": 42, "is_bot": false, "first_name": "tester"},
            "chat_instance": "chat-instance",
            "data": format!("s:abort:{run_id}"),
            "message": {
                "message_id": 7,
                "date": 1_700_000_000,
                "chat": {"id": chat, "type": "private", "first_name": "u"},
                "text": "button"
            }
        }))
        .expect("valid callback query");

        crate::callbacks::handle_callback(deps, q, pending_perms)
            .await
            .expect("handle_callback");

        // (a) The downstream action MUST NOT have run: abort token untouched.
        assert!(
            !abort.is_cancelled(),
            "non-allowed callback fell through to dispatch and aborted the run"
        );

        // (b) Exactly one API call — the "not allowed" answerCallbackQuery —
        // and no second answer from a dispatched action handler.
        let received = server.received_requests().await.unwrap();
        assert_eq!(
            received.len(),
            1,
            "expected exactly one answerCallbackQuery (the rejection), got {}",
            received.len()
        );
        let body = std::str::from_utf8(&received[0].body).unwrap();
        assert!(
            body.contains("not+allowed") || body.contains("not allowed"),
            "rejection answer must carry the 'not allowed' text: {body}"
        );

        crate::shared::RUN_REGISTRY.remove_run(&run_id);
    }

    // ── BUG_REGISTRY B87 / D-INV-YOLO-OFF-DRAINS-PENDING ───────────────
    //
    // Revocation completeness: `/yolo off` clears the in-memory yolo state
    // (`clear_yolo_chat`) + persisted grants, but before B87 it left every
    // outstanding permission card alive in `pending_perms`. Every card
    // carries a live "⚡ YOLO" button, and revocation resets the per-chat
    // escalation count — so a stale card tapped AFTER `/yolo off` (by any
    // chat member) would re-enable YOLO: post-revocation re-escalation
    // bypass. This boundary test drives the REAL `/yolo off` command path
    // (`cmd_yolo`) and the REAL callback handler (`handle_callback`) and
    // asserts:
    //   (a) the revoking chat's pending card is denied + removed,
    //   (b) a DIFFERENT chat's pending card survives (drain is per-chat),
    //   (c) the stale ⚡ tap does NOT re-enable/escalate YOLO.
    // Mutation: drop the `deny_pending_perms` call in `cmd_yolo`'s off
    // branch → (a) and (c) fail.
    #[tokio::test]
    async fn yolo_off_drains_pending_cards_blocking_stale_reescalation() {
        use teloxide::types::CallbackQuery;
        use wiremock::matchers::path_regex;

        let server = MockServer::start().await;
        // sendMessage must round-trip a full Message (the `/yolo off` reply);
        // everything else (answerCallbackQuery) is happy with `result: true`.
        // wiremock evaluates mocks in mount order — specific one first.
        Mock::given(method("POST"))
            .and(path_regex(r"(?i)/sendmessage$"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "ok": true,
                "result": {
                    "message_id": 42,
                    "date": 0,
                    "chat": {"id": 1, "type": "private", "first_name": "u"},
                    "text": "ok"
                }
            })))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "ok": true,
                "result": true
            })))
            .mount(&server)
            .await;

        let unique = unique_test_id();
        let chat = 980_000 + unique as i64;
        let other_chat = chat + 500_000;
        let mut config = test_config();
        config.telegram.allowed_chat_ids = vec![chat, other_chat];

        let bot = mock_bot(&server.uri());
        let channel_map = std::sync::Arc::new(ChannelSessionMap::new());
        let deps = test_bot_deps(bot, config, channel_map.clone(), server.uri());
        let pending_perms: crate::shared::PendingPermissions =
            std::sync::Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new()));

        // Two live permission cards: one in the revoking chat, one in ANOTHER
        // allowed chat. Both are yolo-capable (every card has the ⚡ button).
        let stale_id = format!("b87-stale-{unique}");
        let other_id = format!("b87-other-{unique}");
        let (tx_stale, mut rx_stale) = tokio::sync::oneshot::channel::<bool>();
        let (tx_other, mut rx_other) = tokio::sync::oneshot::channel::<bool>();
        {
            let mut p = pending_perms.write().await;
            p.insert(stale_id.clone(), (tx_stale, chat, None));
            p.insert(other_id.clone(), (tx_other, other_chat, None));
        }

        // Real `/yolo off` command path (revocation).
        let ctx = crate::shared::ChatCtx {
            chat_id: ChatId(chat),
            thread_id: None,
            reply_to: None,
        };
        crate::commands::tools::cmd_yolo(
            &deps.bot,
            &deps.agent,
            &channel_map,
            &deps.config,
            &ctx,
            "/yolo off",
            &pending_perms,
        )
        .await
        .expect("cmd_yolo off");

        // (a) The revoking chat's card is resolved as DENIED (fail-closed —
        // the parked ask_permission returns false immediately, no 120s idle)
        // and removed from the pending map.
        assert!(
            matches!(rx_stale.try_recv(), Ok(false)),
            "/yolo off must deny the chat's pending permission card"
        );
        assert!(
            !pending_perms.read().await.contains_key(&stale_id),
            "/yolo off must remove the chat's pending card from pending_perms"
        );
        // (b) The OTHER chat's card is untouched: still pending, nothing sent.
        assert!(
            pending_perms.read().await.contains_key(&other_id),
            "another chat's pending card must survive /yolo off (per-chat drain)"
        );
        assert!(
            rx_other.try_recv().is_err(),
            "another chat's pending card must not receive any verdict"
        );

        // (c) Stale ⚡ YOLO tap via the REAL callback handler must NOT
        // re-enable or escalate — the bypass this test exists to block.
        let q: CallbackQuery = serde_json::from_value(serde_json::json!({
            "id": format!("b87-cb-{unique}"),
            "from": {"id": 42, "is_bot": false, "first_name": "tester"},
            "chat_instance": "chat-instance",
            "data": format!("p:{stale_id}:yolo"),
            "message": {
                "message_id": 7,
                "date": 1_700_000_000,
                "chat": {"id": chat, "type": "private", "first_name": "u"},
                "text": "card"
            }
        }))
        .expect("valid callback query");
        crate::callbacks::handle_callback(deps, q, pending_perms.clone())
            .await
            .expect("handle_callback");

        assert!(
            !channel_map.is_yolo(chat, None).await,
            "post-revocation stale-card ⚡ tap must NOT re-enable YOLO"
        );
        assert!(
            !channel_map.is_yolo(chat, Some(999)).await,
            "post-revocation stale-card ⚡ tap must NOT escalate chat-wide"
        );
    }
}

// ── Resilience tests ────────────────────────────────────────────────

mod resilience_tests {
    use super::super::helpers::*;
    use super::super::*;

    // ── strip_code_class ────────────────────────────────────────────

    #[test]
    fn strip_code_class_removes_language_attr() {
        let html = r#"<pre><code class="language-python">print(1)</code></pre>"#;
        let out = strip_code_class(html);
        assert_eq!(out, "<pre><code>print(1)</code></pre>");
    }

    #[test]
    fn strip_code_class_preserves_plain_code() {
        let html = "<pre><code>plain</code></pre>";
        assert_eq!(strip_code_class(html), html);
    }

    #[test]
    fn strip_code_class_multiple_blocks() {
        let html = r#"<pre><code class="language-rust">fn main()</code></pre> text <pre><code class="language-js">var x</code></pre>"#;
        let out = strip_code_class(html);
        assert!(out.contains("<pre><code>fn main()"), "{out}");
        assert!(out.contains("<pre><code>var x"), "{out}");
        assert!(!out.contains("class="), "{out}");
    }

    // ── strip_html_tags ─────────────────────────────────────────────

    #[test]
    fn strip_html_tags_basic() {
        assert_eq!(strip_html_tags("<b>bold</b> text"), "bold text");
    }

    #[test]
    fn strip_html_tags_entities() {
        assert_eq!(strip_html_tags("a &amp; b &lt; c"), "a & b < c");
    }

    #[test]
    fn strip_html_tags_nested() {
        assert_eq!(strip_html_tags("<pre><code>x</code></pre>"), "x");
    }

    // ── format_provider_error ───────────────────────────────────────

    #[test]
    fn error_429_shows_rate_limit() {
        let msg = format_provider_error(
            "provider error: OpenAI API 429 Too Many Requests: {\"error\":{}}",
            "fireworks/minimax",
        );
        assert!(msg.contains("Rate limit"), "{msg}");
        assert!(msg.contains("fireworks/minimax"), "{msg}");
        assert!(msg.contains("/model"), "{msg}");
        // No raw JSON in output
        assert!(!msg.contains(r#""error""#), "should strip JSON: {msg}");
    }

    #[test]
    fn error_402_shows_payment() {
        let msg = format_provider_error(
            "OpenAI API 402 Payment Required: {\"error\":{\"message\":\"membership\"}}",
            "kimi-code/kimi",
        );
        assert!(msg.contains("402") || msg.contains("💳"), "{msg}");
    }

    #[test]
    fn error_500_shows_server() {
        let msg = format_provider_error("OpenAI API 500 Internal Server Error", "qwen/qwen3");
        assert!(msg.contains("5xx") || msg.contains("🔧"), "{msg}");
    }

    #[test]
    fn error_no_content_shows_friendly() {
        let msg =
            format_provider_error("provider returned no content after 3 retries", "any/model");
        assert!(
            msg.contains("0 токенов") || msg.contains("без ответа"),
            "{msg}"
        );
    }

    #[test]
    fn error_unknown_strips_json() {
        let msg = format_provider_error("some weird error: {\"details\":\"secret\"}", "x/y");
        assert!(msg.contains("some weird error"), "{msg}");
        assert!(!msg.contains("secret"), "JSON should be stripped: {msg}");
    }

    // parse_retry_after tests moved to rate_limit.rs

    // ── format_input_preview ────────────────────────────────────────

    #[test]
    fn format_input_preview_single_key() {
        let v = serde_json::json!({"query": "hello world"});
        let s = format_input_preview(&v, 50);
        assert!(s.contains("query:"), "{s}");
        assert!(s.contains("hello world"), "{s}");
    }

    #[test]
    fn format_input_preview_multi_key() {
        let v = serde_json::json!({"a": 1, "b": 2});
        let s = format_input_preview(&v, 50);
        // Multi-key shows key: val pairs
        assert!(s.contains("a:") || s.contains("b:"), "{s}");
    }

    #[test]
    fn format_input_preview_truncates() {
        let v = serde_json::json!({"query": "a".repeat(200)});
        let s = format_input_preview(&v, 20);
        assert!(s.len() <= 30, "should truncate: {}", s.len()); // some slack for key + …
    }

    #[test]
    fn cancelled_error_is_not_a_crash() {
        // The streaming loop treats Error("cancelled") as normal turn
        // displacement, not a crash. Verify the detection pattern.
        let cancel_msgs = ["cancelled", "Cancelled", "turn cancelled by new message"];
        for msg in cancel_msgs {
            assert!(
                msg.contains("cancelled") || msg.contains("Cancelled"),
                "should detect cancel in: {msg}"
            );
        }
        // Non-cancel errors should NOT match:
        let real_errors = ["connection reset", "timeout", "panic"];
        for msg in real_errors {
            assert!(
                !msg.contains("cancelled") && !msg.contains("Cancelled"),
                "should NOT detect cancel in: {msg}"
            );
        }
    }

    // ─── PLAN_MEDIA_UX_v1 M4 / BUG_REGISTRY B02 ───
    //
    // Pins the DRY helper that owns the single keyboard literal:
    // if anyone re-introduces a second send_message at stream-start,
    // the helper must remain the only source of truth for the
    // button layout. Full one-bubble assertion uses wiremock — see
    // resilience_tests::stream_start_sends_exactly_one_message_with_inline_kbd.
    #[test]
    fn streaming_control_kb_for_run_has_two_scoped_buttons() {
        let kb = crate::streaming::pipeline::streaming_control_kb_for_run("run-kb");
        let rows = kb.inline_keyboard;
        assert_eq!(rows.len(), 1, "expected single row of buttons");
        let row = &rows[0];
        assert_eq!(row.len(), 2, "expected exactly 2 buttons");
        // Order matters for muscle memory: abort first, sendnow second.
        assert!(row[0].text.contains("Stop"), "button 0 should be Stop");
        assert!(row[1].text.contains("Send"), "button 1 should be Send");
        // Callback data must match the dispatcher in callbacks.rs.
        use teloxide::types::InlineKeyboardButtonKind;
        let cb0 = match &row[0].kind {
            InlineKeyboardButtonKind::CallbackData(d) => d.as_str(),
            _ => panic!("button 0 must be CallbackData"),
        };
        let cb1 = match &row[1].kind {
            InlineKeyboardButtonKind::CallbackData(d) => d.as_str(),
            _ => panic!("button 1 must be CallbackData"),
        };
        assert_eq!(cb0, "s:abort:run-kb");
        assert_eq!(cb1, "s:sendnow:run-kb");
    }
    /// B118b: a turn that stalled and then answered must still say it stalled.
    ///
    /// The 2026-08-08 incident: the agent was stuck for minutes on a failing
    /// API, the stall detector fired twice into the journal, then the model
    /// emitted one short sentence. `render_final` takes the response_text
    /// branch and drops the whole event timeline, so the warning the user
    /// needed most was the one thing guaranteed not to be shown.
    #[test]
    fn render_final_keeps_stall_warning_when_model_still_answered() {
        let mut v = CompositeView::new("test-model".into());
        v.events.push(TurnEvent::Note(
            "🔴 Зависло 2 мин — /abort чтобы прервать".into(),
        ));
        v.stall_notice = Some("🔴 Зависло 2 мин — /abort чтобы прервать".into());
        v.response_text = "Возьму выборку через erp_analyst".into();

        let out = v.render_final();
        assert!(
            out.contains("Зависло 2 мин"),
            "the stall warning must survive into the final answer, got: {out}"
        );
        assert!(
            out.contains("erp_analyst"),
            "the model's answer must still be shown, got: {out}"
        );
    }

    /// A note that is not a stall warning must NOT be promoted to a banner.
    /// `Note` also carries steer echoes and test branch markers; treating all
    /// of them as warnings would put noise above every answer.
    #[test]
    fn render_final_does_not_promote_ordinary_notes_to_banner() {
        let mut v = CompositeView::new("test-model".into());
        v.events.push(TurnEvent::Note("steer received".into()));
        v.response_text = "готово".into();

        let out = v.render_final();
        assert!(
            !out.contains("steer received"),
            "ordinary notes must stay in the timeline, not the banner: {out}"
        );
    }

    /// Both banners can apply at once and must not swallow each other.
    #[test]
    fn render_final_shows_fallback_and_stall_banners_together() {
        let mut v = CompositeView::new("groq/x".into());
        v.fallback_notice = Some("Отвечал deepseek — groq не смог: 413".into());
        v.stall_notice = Some("⚠️ Нет ответа 60с".into());
        v.response_text = "ответ".into();

        let out = v.render_final();
        assert!(
            out.contains("Отвечал deepseek"),
            "fallback banner lost: {out}"
        );
        assert!(out.contains("Нет ответа 60с"), "stall banner lost: {out}");
        assert!(out.contains("ответ"), "body lost: {out}");
    }
}
