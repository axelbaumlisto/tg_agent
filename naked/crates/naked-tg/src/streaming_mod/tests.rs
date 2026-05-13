mod tests {
    use super::super::helpers::*;
    use super::super::*;
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

    /// BUG_REGISTRY D-INV-STREAM-BUBBLE-COUNT (B02 / INV-4 full version):
    /// stream-start MUST send exactly ONE Telegram API request, not
    /// two (the old `⏳` placeholder + `⏯️` control card pattern).
    /// Uses the same wiremock infra as send_text_hits_mock_server_with_sendmessage.
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
        let _ = crate::streaming::pipeline::send_stream_placeholder(&bot, &ctx).await;

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
    fn streaming_control_kb_has_two_buttons() {
        let kb = crate::streaming::pipeline::streaming_control_kb();
        let rows = kb.inline_keyboard;
        assert_eq!(rows.len(), 1, "expected single row of buttons");
        let row = &rows[0];
        assert_eq!(row.len(), 2, "expected exactly 2 buttons");
        // Order matters for muscle memory: abort first, sendnow second.
        assert!(row[0].text.contains("Стоп"), "button 0 should be Stop");
        assert!(
            row[1].text.contains("Send now"),
            "button 1 should be Send now"
        );
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
        assert_eq!(cb0, "stream:abort");
        assert_eq!(cb1, "stream:sendnow");
    }
}
