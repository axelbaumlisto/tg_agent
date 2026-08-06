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
        cycle_count: 0,
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
        cycle_count: 0,
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
        cycle_count: 0,
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
        cycle_count: 0,
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
        cycle_count: 0,
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
        cycle_count: 0,
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
        cycle_count: 0,
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

// ── Seam-style compaction (append-only, prefix-cache safe) ────────────────

impl ConversationHistory {
    /// Seam compaction: instead of removing old messages, insert a summary
    /// marker between the "archived" prefix and recent messages. The prefix
    /// stays intact for cache reuse. The summary acts as a navigation aid.
    ///
    /// Returns (archived_count, total_after) or None if not needed.
    pub fn compact_seam(&mut self, keep_recent: usize) -> Option<(usize, usize)> {
        if !self.needs_compaction() {
            return None;
        }
        let keep = keep_recent.max(DEFAULT_PRESERVE_RECENT);
        if self.messages.len() <= keep {
            return None;
        }

        let keep_from = self.messages.len().saturating_sub(keep);
        let keep_from = Self::snap_to_turn_boundary(&self.messages, keep_from, 0);
        let archived = &self.messages[..keep_from];
        if archived.is_empty() {
            return None;
        }

        let summary = compaction::summarize_messages(archived);
        let archived_count = archived.len();

        // Insert seam marker between archived and recent:
        let marker = ConversationMessage {
            role: Role::System,
            blocks: vec![ContentBlock::Text {
                text: format!(
                    "<archived_context>\n{summary}\n</archived_context>\n\
                 The messages above ({archived_count} messages) are archived. \
                 Read the summary first; drill into specific messages only if needed."
                ),
            }],
            timestamp: chrono::Utc::now(),
            usage: None,
        };

        // Insert marker right before the keep_from position:
        self.messages.insert(keep_from, marker);
        self.last_input_tokens = None;

        Some((archived_count, self.messages.len()))
    }
}

mod seam_tests {
    use super::*;

    #[test]
    fn seam_compact_inserts_marker() {
        let mut h = ConversationHistory::new("sys".into());
        h.set_context_window_tokens(100); // very low threshold
        // Add many messages to trigger compaction:
        for i in 0..20 {
            h.push_user(&format!("message {i} with enough content to fill tokens"));
            h.push_assistant(
                vec![ContentBlock::Text {
                    text: format!("reply {i} with substantial content here too"),
                }],
                Default::default(),
            );
        }

        let before = h.message_count();
        let result = h.compact_seam(6);
        assert!(result.is_some());
        let (archived, total) = result.unwrap();
        assert!(archived > 0);
        // Total should be MORE than before (we added a marker, didn't remove):
        assert_eq!(total, before + 1);
        // Verify marker exists:
        let has_marker = h
            .messages()
            .iter()
            .any(|m| m.text_content().contains("<archived_context>"));
        assert!(has_marker, "seam marker must be present");
    }

    #[test]
    fn seam_compact_noop_when_not_needed() {
        let mut h = ConversationHistory::new("sys".into());
        h.push_user("hi");
        assert!(h.compact_seam(6).is_none());
    }
}

// ── D-INV-TOOL-OUTPUT-WINDOW-AWARE (B104) ─────────────────────────────────
//
// `push_tool_result` is the ingest backstop for tool output. It used to be a
// hard 8000 bytes, which bound TIGHTER than the model-aware router in
// `loop_::tools` (24K for >=100K windows, 180K for >=500K) — so a 16.6 KB
// skill body still lost 8.6 KB on a 200K-window model even after the router
// decided it fit. The ceiling must now come from the same limits table.
// Restoring `const MAX_TOOL_OUTPUT: usize = 8000;` fails the first two tests.

fn tool_output_of(h: &ConversationHistory) -> String {
    let msg = h.messages().iter().find(|m| m.role == Role::Tool).unwrap();
    match &msg.blocks[0] {
        ContentBlock::ToolResult { output, .. } => output.clone(),
        _ => panic!("expected ToolResult"),
    }
}

#[test]
fn b104_skill_sized_output_survives_ingest_on_large_window() {
    let mut h = ConversationHistory::new("sys".into());
    h.set_context_window_tokens(200_000);
    // Real telegram-reader skill payload size (~16.6 KB): under the 24K
    // limit for a 200K window, so it must arrive intact.
    let body = "x".repeat(16_639);
    h.push_tool_result("c1", &body, false);

    let out = tool_output_of(&h);
    assert_eq!(
        out.len(),
        body.len(),
        "16.6 KB tool output must not be clamped on a 200K-token window"
    );
    assert!(!out.contains("truncated"));
}

#[test]
fn b104_ingest_ceiling_follows_context_window() {
    // 1M window -> 180K limit: a 100 KB result fits.
    let mut big = ConversationHistory::new("sys".into());
    big.set_context_window_tokens(1_000_000);
    let body = "y".repeat(100_000);
    big.push_tool_result("c1", &body, false);
    assert_eq!(
        tool_output_of(&big).len(),
        body.len(),
        "100 KB must survive a 1M-token window (180K limit)"
    );

    // Small window -> 12K limit: the SAME body is still clamped.
    let mut small = ConversationHistory::new("sys".into());
    small.set_context_window_tokens(32_000);
    small.push_tool_result("c1", &body, false);
    let clamped = tool_output_of(&small);
    assert!(
        clamped.len() < body.len(),
        "small windows must still clamp; got {} bytes",
        clamped.len()
    );
    assert!(clamped.contains("truncated"));
}

#[test]
fn b104_ingest_still_bounds_absurd_output() {
    // The backstop must remain a real bound, not become unlimited.
    let mut h = ConversationHistory::new("sys".into());
    h.set_context_window_tokens(1_000_000);
    let body = "z".repeat(400_000);
    h.push_tool_result("c1", &body, false);
    let out = tool_output_of(&h);
    assert!(
        out.len() < body.len(),
        "400 KB must still be clamped even on a 1M window"
    );
    assert!(out.contains("truncated"));
}

#[test]
fn b104_ingest_clamp_is_utf8_safe() {
    // Cyrillic is 2 bytes/char — the clamp must not split a char.
    let mut h = ConversationHistory::new("sys".into());
    h.set_context_window_tokens(32_000); // 12K limit
    let body = "Привет ".repeat(4_000);
    h.push_tool_result("c1", &body, false);
    let out = tool_output_of(&h);
    assert!(out.contains("truncated"));
    // Reaching here without a panic proves the boundary math held.
    assert!(out.is_char_boundary(0));
}

mod proptest_history {
    use crate::history::ConversationHistory;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn push_user_increments_count(msgs in proptest::collection::vec("[a-zA-Z ]{1,50}", 1..10)) {
            let mut h = ConversationHistory::new("sys".into());
            for (i, msg) in msgs.iter().enumerate() {
                h.push_user(msg);
                prop_assert_eq!(h.message_count(), i + 1, "count after push #{}", i);
            }
        }

        #[test]
        fn estimated_tokens_non_negative(msgs in proptest::collection::vec("[a-zA-Z ]{1,100}", 0..5)) {
            let mut h = ConversationHistory::new("sys".into());
            for msg in &msgs {
                h.push_user(msg);
            }
            // estimated_tokens returns usize — always >= 0 by type.
            let _ = h.estimated_tokens();
        }

        #[test]
        fn system_prompt_preserved(system in "[a-zA-Z ]{1,50}", msg in "[a-zA-Z ]{1,50}") {
            let mut h = ConversationHistory::new(system.clone());
            h.push_user(&msg);
            let api = h.to_api_messages();
            // System prompt is separate, but messages should be non-empty
            prop_assert!(!api.is_empty());
        }
    }
}
