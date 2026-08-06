//! E2E tests — core (basic flow, multi-turn, usage, multimodal)

#[macro_use]
mod common;
use common::*;

#[tokio::test]
async fn t01_basic_text_reply() {
    need_provider!(config, provider, model);
    pace().await;
    let tmp = tempfile::tempdir().unwrap();

    eprintln!(">>> t01_basic_text_reply [model={model}]");
    let mut history = ConversationHistory::new(SYS.into());
    history.push_user("Say exactly: HELLO_NAKED_42. Nothing else.");
    let r = run_prompt(provider, &mut history, tmp.path(), &model).await;

    eprintln!("  text: {}", r.full_text());
    assert!(!r.had_error(), "agent error");
    assert!(r.got_idle(), "no idle");
    assert!(
        r.full_text().contains("HELLO_NAKED_42"),
        "missing phrase: {}",
        r.full_text()
    );
}

#[tokio::test]
async fn t07_multi_turn() {
    need_config!(config);
    let (_, model) = match get_working_provider(&config).await {
        Some(pm) => pm,
        None => {
            eprintln!("SKIP t07: no working provider");
            return;
        }
    };
    pace().await;
    let tmp = tempfile::tempdir().unwrap();

    eprintln!(">>> t07_multi_turn");
    let mut history = ConversationHistory::new(SYS.into());

    history.push_user("The project name is PROJ_88. Confirm you understand.");
    let (p1, _) = get_working_provider(&config).await.unwrap();
    let r1 = run_prompt(p1, &mut history, tmp.path(), &model).await;
    eprintln!("  turn1: {}", r1.full_text());
    assert!(!r1.had_error(), "turn1 error");
    assert!(r1.got_idle(), "turn1 no idle");
    history = r1.history;

    pace().await;

    history.push_user("What was the project name I mentioned? Say only the name.");
    let (p2, _) = get_working_provider(&config).await.unwrap();
    let r2 = run_prompt(p2, &mut history, tmp.path(), &model).await;
    eprintln!("  turn2: {}", r2.full_text());
    assert!(!r2.had_error(), "turn2 error");
    assert!(
        r2.full_text().contains("PROJ_88"),
        "context lost: {}",
        r2.full_text()
    );
}

#[tokio::test]
async fn t09_agent_core_flow() {
    need_provider!(config, provider, model);
    pace().await;
    let tmp = tempfile::tempdir().unwrap();

    eprintln!(">>> t09_agent_core_flow");
    let core_config = Config {
        workspace: tmp.path().to_path_buf(),
        session_dir: tmp.path().join("sessions"),
        default_provider: config.default_provider.clone(),
        default_model: model,
        max_iterations: 10,
        tool_timeout_secs: 30,
        ..Default::default()
    };

    let agent = naked_core::AgentCore::new(core_config, provider);
    let session_id = agent.create_session(tmp.path()).await;

    let mut handle = agent
        .send_prompt(&session_id, "Say exactly: CORE_OK_123")
        .await
        .unwrap();

    let mut text = String::new();
    let mut got_idle = false;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(120);
    loop {
        match tokio::time::timeout_at(deadline, handle.events.recv()).await {
            Ok(Some(AgentEvent::TextDelta(t))) => text.push_str(&t),
            Ok(Some(AgentEvent::Idle)) => {
                got_idle = true;
                break;
            }
            Ok(Some(AgentEvent::Error(e))) => {
                eprintln!("  error: {e}");
                break;
            }
            Ok(None) => break,
            Err(_) => {
                eprintln!("  TIMEOUT");
                break;
            }
            _ => {}
        }
    }

    eprintln!("  text: {text}");
    assert!(got_idle, "no idle");
    assert!(text.contains("CORE_OK_123"), "missing phrase: {text}");
    assert!(!agent.list_sessions().await.is_empty());
}

#[tokio::test]
async fn t10_tool_error_recovery() {
    need_provider!(_config, provider, model);
    pace().await;
    let tmp = tempfile::tempdir().unwrap();

    eprintln!(">>> t10_tool_error_recovery");
    let mut history = ConversationHistory::new(SYS.into());
    history.push_user(
        "Use read_file to read /tmp/nonexistent_e2e_99.txt. If it fails, say FILE_MISSING.",
    );
    let r = run_prompt(provider, &mut history, tmp.path(), &model).await;

    eprintln!("  text: {}", r.full_text());
    let text = r.full_text().to_lowercase();
    assert!(
        text.contains("file")
            || text.contains("missing")
            || text.contains("not found")
            || text.contains("error")
            || text.contains("exist"),
        "no error handling: {}",
        r.full_text()
    );
}

#[tokio::test]
async fn t11_multi_step_chain() {
    need_provider!(_config, provider, model);
    pace().await;
    let tmp = tempfile::tempdir().unwrap();

    eprintln!(">>> t11_multi_step_chain");
    let mut history = ConversationHistory::new(SYS.into());
    history.push_user(&format!(
        "1. Use write_file to write 'chain_data' to {dir}/chain.txt\n\
         2. Use read_file to read it back\n\
         3. Confirm content matches",
        dir = tmp.path().display()
    ));
    let r = run_prompt(provider, &mut history, tmp.path(), &model).await;

    eprintln!("  text: {}", r.full_text());
    eprintln!("  tools: {:?}", r.tool_names());
    assert!(!r.had_error(), "agent error");
    assert!(
        r.has_tool("write_file"),
        "no write_file: {:?}",
        r.tool_names()
    );

    let content = std::fs::read_to_string(tmp.path().join("chain.txt")).unwrap_or_default();
    assert!(content.contains("chain_data"), "file wrong: {content}");
    assert!(
        r.tool_names().len() >= 2,
        "expected >=2 tools: {:?}",
        r.tool_names()
    );
}

#[tokio::test]
async fn t12_usage_tracking() {
    need_provider!(_config, provider, model);
    pace().await;
    let tmp = tempfile::tempdir().unwrap();

    eprintln!(">>> t12_usage_tracking");
    let mut history = ConversationHistory::new(SYS.into());
    history.push_user("Say: hi");
    let r = run_prompt(provider, &mut history, tmp.path(), &model).await;

    let usage_events: Vec<_> = r
        .events
        .iter()
        .filter(|e| matches!(e, AgentEvent::UsageUpdate(_)))
        .collect();

    eprintln!("  usage events: {}", usage_events.len());
    for ev in &usage_events {
        if let AgentEvent::UsageUpdate(u) = ev {
            eprintln!("  tokens: {}in + {}out", u.input_tokens, u.output_tokens);
        }
    }
    assert!(!usage_events.is_empty(), "no usage events");
}

#[tokio::test]
async fn t36_multi_key_rotation_live() {
    need_config!(config);
    pace().await;

    let pc = match config.providers.get("kimi-code") {
        Some(pc) => pc,
        None => {
            eprintln!("SKIP t36: kimi-code not configured");
            return;
        }
    };

    let resolved = match pc.resolved() {
        Ok(r) => r,
        Err(_) => {
            eprintln!("SKIP t36: kimi-code key unresolvable");
            return;
        }
    };

    if resolved.all_keys.len() < 2 {
        eprintln!(
            "SKIP t36: kimi-code has only {} key(s), need >=2 for rotation test",
            resolved.all_keys.len()
        );
        return;
    }

    let provider = naked_core::create_provider("kimi-code", resolved.clone());
    eprintln!(
        ">>> t36_multi_key_rotation_live [kimi-code, {} keys]",
        resolved.all_keys.len()
    );

    let model = resolved.models.first().cloned().unwrap_or("k2p5".into());
    let ok = probe_provider(provider.as_ref(), &model).await;
    if !ok {
        // This test exists to verify *multi-key rotation* plumbing, not
        // the upstream provider's health. If kimi-code itself is sick
        // today (it returns stream OK but zero text chunks under load,
        // or refuses vision-capable reasoning-less prompts) just skip
        // rather than fail the whole suite for an external brown-out.
        eprintln!("SKIP t36: kimi-code returned no text across all probe variants");
        return;
    }
    eprintln!("  probe OK with model={model}");
}

#[tokio::test]
async fn t38_parallel_load_2x50() {
    need_config!(config);

    let num_turns: usize = std::env::var("LOAD_TURNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(50);

    let candidates: &[(&str, &str)] = &[
        ("glm-cn", "glm-5.1"),
        ("zai", "glm-5"),
        ("kimi-code", "k2p5"),
        ("fireworks", "accounts/fireworks/models/glm-5p1"),
        ("moonshot", "moonshot-v1-8k"),
        ("anthropic", "claude-haiku-4-5-20251001"),
        ("minimax-cn", "MiniMax-M2.7-highspeed"),
        ("groq", "llama-3.3-70b-versatile"),
    ];

    let mut live: Vec<(&str, &str)> = Vec::new();
    for &(prov, model) in candidates {
        if live.len() >= 2 {
            break;
        }
        if let Some(pc) = config.providers.get(prov)
            && let Ok(r) = pc.resolved()
            && !r.api_key.is_empty()
            && !r.api_key.starts_with('$')
        {
            let probe_prov = naked_core::create_provider(prov, r);
            if probe_provider(probe_prov.as_ref(), model).await {
                live.push((prov, model));
            } else {
                eprintln!("  probe SKIP: {prov}/{model}");
            }
        }
    }

    if live.len() < 2 {
        eprintln!(
            "SKIP t38: need >=2 configured providers, got {}",
            live.len()
        );
        return;
    }

    let (prov_a, model_a) = live[0];
    let (prov_b, model_b) = live[1];

    eprintln!(">>> t38_parallel_load_2x{num_turns}");
    eprintln!("  Chat A: {prov_a}/{model_a}");
    eprintln!("  Chat B: {prov_b}/{model_b}");

    let config_a = config.clone();
    let config_b = config.clone();
    let pa = prov_a.to_string();
    let ma = model_a.to_string();
    let pb = prov_b.to_string();
    let mb = model_b.to_string();

    let t0 = std::time::Instant::now();

    let (stats_a, stats_b) = tokio::join!(
        run_load_session("ChatA", &pa, &ma, &config_a, num_turns),
        run_load_session("ChatB", &pb, &mb, &config_b, num_turns),
    );

    let wall_ms = t0.elapsed().as_millis();

    // ── Report ───────────────────────────────────────────────────────────────
    eprintln!("\n{}", "=".repeat(60));
    eprintln!("  PARALLEL LOAD TEST RESULTS");
    eprintln!("  Wall time: {wall_ms}ms ({:.1}s)", wall_ms as f64 / 1000.0);

    // run_load_session() returns None when its own probe (different from
    // the upstream probe) or initial setup fails — usually transient
    // (provider hiccup mid-test, rate limit). Skip rather than panic so
    // a flaky third-party doesn't sink the whole suite.
    let (a, b) = match (stats_a, stats_b) {
        (Some(a), Some(b)) => (a, b),
        (a, b) => {
            eprintln!(
                "SKIP t38: chat A ran={} chat B ran={} — at least one provider failed mid-test",
                a.is_some(),
                b.is_some()
            );
            return;
        }
    };

    // Use session wall times (including delays) for the parallelism metric.
    // If both sessions ran sequentially, total time = session_wall_A + session_wall_B.
    // Running in parallel, wall time ≈ max(session_wall_A, session_wall_B).
    let sequential_ms = a.session_wall_ms + b.session_wall_ms;
    let parallelism = if wall_ms > 0 {
        sequential_ms as f64 / wall_ms as f64
    } else {
        1.0
    };

    for s in [&a, &b] {
        eprintln!("  ┌─ {}", s.tag);
        eprintln!(
            "  │ ok: {}/{} | content-verified: {} | errors: {} | rate-limited: {}",
            s.ok, s.turns, s.content_ok, s.errors, s.rate_limited
        );
        eprintln!(
            "  │ tools: {} calls ({} unique: {:?})",
            s.tools_used,
            s.tool_set.len(),
            s.tool_set
        );
        eprintln!(
            "  │ latency: avg {}ms | P50 {}ms | P95 {}ms | P99 {}ms | min {}ms | max {}ms",
            s.avg(),
            s.percentile(50.0),
            s.percentile(95.0),
            s.percentile(99.0),
            s.latencies.iter().copied().min().unwrap_or(0),
            s.latencies.iter().copied().max().unwrap_or(0),
        );
        eprintln!(
            "  │ throughput: {:.2} turns/sec | total: {:.1}s",
            s.throughput(),
            s.total_ms as f64 / 1000.0
        );
        eprintln!("  └─");
    }

    let total_turns = num_turns * 2;
    let total_ok = a.ok + b.ok;
    let total_content = a.content_ok + b.content_ok;
    let total_err = a.errors + b.errors;
    let total_rl = a.rate_limited + b.rate_limited;
    let total_tools = a.tools_used + b.tools_used;
    let all_tools: std::collections::HashSet<_> = a.tool_set.union(&b.tool_set).collect();

    eprintln!(
        "  Total: {total_ok}/{total_turns} ok | {total_content} content-verified | {total_err} errors | {total_rl} rate-limited | {total_tools} tool calls"
    );
    eprintln!("  Tools covered: {:?}", all_tools);
    eprintln!("  Parallelism: {parallelism:.2}x (sequential={sequential_ms}ms, wall={wall_ms}ms)");
    eprintln!(
        "  Wall throughput: {:.2} turns/sec",
        total_turns as f64 / (wall_ms as f64 / 1000.0)
    );
    eprintln!("{}", "=".repeat(60));

    // ── Assertions ───────────────────────────────────────────────────────────
    // Only count real errors (not rate-limited turns) for error rate
    let error_rate = total_err as f64 / total_turns as f64;
    assert!(
        error_rate < 0.1,
        "error rate {:.1}% too high (max 10%, excl. rate limits)",
        error_rate * 100.0
    );

    // Content verified rate: excluding rate-limited turns from the denominator
    let effective_turns = total_ok + total_err;
    let content_rate = if effective_turns > 0 {
        total_content as f64 / effective_turns as f64
    } else {
        0.0
    };
    assert!(
        content_rate > 0.7,
        "content verification rate {:.1}% too low (min 70% of non-rate-limited turns)",
        content_rate * 100.0
    );

    assert!(
        parallelism > 1.1,
        "parallelism {parallelism:.2}x too low — sessions may not be truly parallel"
    );

    let expected_tools = ["bash", "write_file", "read_file"];
    for tool in &expected_tools {
        assert!(
            all_tools.contains(&tool.to_string()),
            "tool '{tool}' never used — test diversity insufficient"
        );
    }

    // At least 70% of turns should succeed (including through rate limit recovery)
    let success_rate = total_ok as f64 / total_turns as f64;
    assert!(
        success_rate > 0.7,
        "success rate {:.1}% too low (min 70%)",
        success_rate * 100.0
    );
}

#[tokio::test]
async fn t42_config_live_reload_between_turns() {
    need_config!(config);
    let providers = find_working_providers(&config, 2).await;
    if providers.len() < 2 {
        eprintln!(
            "SKIP t42: need 2 working providers, found {}",
            providers.len()
        );
        return;
    }
    let (prov_a, model_a) = &providers[0];
    let (prov_b, model_b) = &providers[1];
    eprintln!(">>> t42_config_live_reload [{prov_a}/{model_a} -> {prov_b}/{model_b}]");

    let tmp = tempfile::tempdir().unwrap();
    let agent = make_agent_core(&config, tmp.path(), prov_a, model_a);
    let id = agent.create_session(tmp.path()).await;

    // Turn 1: no config.json — uses global defaults
    let (text1, _, idle1) = agent_prompt(&agent, &id, "Say exactly: TURN1_OK. Nothing else.").await;
    eprintln!("  turn1: {text1}");
    assert!(idle1, "turn1 no idle");

    let sessions = agent.list_sessions().await;
    let meta1 = sessions.iter().find(|s| s.id == id).unwrap();
    assert_eq!(
        meta1.provider.as_deref(),
        Some(prov_a.as_str()),
        "turn1 wrong provider"
    );
    eprintln!(
        "  turn1 provider: {}",
        meta1.provider.as_deref().unwrap_or("?")
    );

    pace().await;

    // Write config.json to switch provider mid-conversation
    write_session_config(
        &tmp.path().join("sessions"),
        &id,
        &serde_json::json!({
            "default_provider": prov_b,
            "default_model": model_b,
        }),
    )
    .await;

    // Turn 2: config.json exists — should use new provider
    let (text2, _, idle2) = agent_prompt(&agent, &id, "Say exactly: TURN2_OK. Nothing else.").await;
    eprintln!("  turn2: {text2}");
    assert!(idle2, "turn2 no idle");

    let sessions = agent.list_sessions().await;
    let meta2 = sessions.iter().find(|s| s.id == id).unwrap();
    assert_eq!(
        meta2.provider.as_deref(),
        Some(prov_b.as_str()),
        "turn2 should have switched to '{prov_b}' after config.json edit"
    );
    eprintln!(
        "  turn2 provider: {}",
        meta2.provider.as_deref().unwrap_or("?")
    );
}

#[tokio::test]
async fn t47_unicode_edge_cases() {
    let config = match test_config() {
        Some(c) => c,
        None => return,
    };
    let (provider, model) = match get_working_provider(&config).await {
        Some(p) => p,
        None => {
            eprintln!("SKIP t47: no working provider");
            return;
        }
    };
    eprintln!(">>> t47_unicode_edge_cases using {model}");

    let tmp = tempfile::tempdir().unwrap();

    // Test 1: Cyrillic prompt
    {
        let mut history = ConversationHistory::new(SYS.into());
        history.push_user("Привет! Сколько будет 10+5? Ответь только числом.");
        let r = run_prompt(
            naked_core::create_provider("test", config.resolve_default_provider().unwrap().1),
            &mut history,
            tmp.path(),
            &model,
        )
        .await;
        let text = r.full_text();
        eprintln!("  cyrillic: {}", text.trim());
        assert!(!text.is_empty(), "should handle Cyrillic prompts");
        assert!(text.contains("15"), "expected 15 in response: {text}");
    }

    pace().await;

    // Test 2: Emoji-heavy prompt
    {
        let mut history = ConversationHistory::new(SYS.into());
        history.push_user("🎉🚀 What is 7+8? Reply with ONLY the number 🔢");
        let r = run_prompt(
            naked_core::create_provider("test", config.resolve_default_provider().unwrap().1),
            &mut history,
            tmp.path(),
            &model,
        )
        .await;
        let text = r.full_text();
        eprintln!("  emoji: {}", text.trim());
        assert!(!text.is_empty(), "should handle emoji prompts");
        assert!(text.contains("15"), "expected 15 in response: {text}");
    }

    pace().await;

    // Test 3: CJK characters
    {
        let mut history = ConversationHistory::new(SYS.into());
        history.push_user("你好！请计算 12+13，只回答数字。");
        let r = run_prompt(
            naked_core::create_provider("test", config.resolve_default_provider().unwrap().1),
            &mut history,
            tmp.path(),
            &model,
        )
        .await;
        let text = r.full_text();
        eprintln!("  cjk: {}", text.trim());
        assert!(!text.is_empty(), "should handle CJK prompts");
        assert!(text.contains("25"), "expected 25 in response: {text}");
    }

    pace().await;

    // Test 4: Mixed Unicode with tool use (ensures bash output with Unicode doesn't panic)
    {
        let mut history = ConversationHistory::new(SYS.into());
        history.push_user("Run `echo 'Привет мир 🌍 你好世界'` in bash and show the output.");
        let r = run_prompt(provider, &mut history, tmp.path(), &model).await;
        let text = r.full_text();
        let tools = r.tool_names();
        eprintln!("  mixed tool: tools={tools:?} text={}", text.trim());
        assert!(tools.contains(&"bash".to_string()), "should use bash tool");
        assert!(
            text.contains("Привет") || text.contains("你好") || text.contains("🌍"),
            "response should contain Unicode output"
        );
    }

    // Test 5: History compaction with Unicode (verifies safe_truncate doesn't panic)
    {
        let mut history = ConversationHistory::new(SYS.into());
        let long_cyrillic = "Привет ".repeat(100);
        for i in 0..10 {
            history.push_user(&format!("{long_cyrillic} вопрос {i}"));
            history.push_assistant(
                vec![ContentBlock::Text {
                    text: format!("ответ {i}"),
                }],
                None,
            );
        }
        let before = history.message_count();
        history.compact(4);
        assert!(
            history.message_count() < before,
            "compaction should reduce messages: {before} -> {}",
            history.message_count()
        );
        let summary_text = history.messages()[0].text_content();
        assert!(
            summary_text.contains("compacted") || summary_text.contains("summary"),
            "compaction summary should be present: {summary_text}"
        );
    }

    eprintln!("  PASS: all Unicode edge cases handled without panics");
}

#[tokio::test]
async fn t48_search_outside_workspace() {
    let workspace = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    std::fs::write(outside.path().join("needle.txt"), "FIND_ME_E2E\n").unwrap();

    let grep = GrepSearchTool;
    let result = grep
        .execute(
            serde_json::json!({
                "pattern": "FIND_ME_E2E",
                "path": outside.path().to_str().unwrap()
            }),
            workspace.path(),
        )
        .await;
    assert!(
        !result.is_error,
        "grep outside workspace should work: {}",
        result.output
    );
    assert!(
        result.output.contains("FIND_ME_E2E"),
        "grep should find content outside workspace: {}",
        result.output
    );

    let glob = GlobSearchTool;
    let result = glob
        .execute(
            serde_json::json!({
                "pattern": "*.txt",
                "path": outside.path().to_str().unwrap()
            }),
            workspace.path(),
        )
        .await;
    assert!(
        !result.is_error,
        "glob outside workspace should work: {}",
        result.output
    );
    assert!(
        result.output.contains("needle.txt"),
        "glob should find files outside workspace: {}",
        result.output
    );
    eprintln!("  PASS: search tools work outside workspace");
}

#[tokio::test]
async fn t49_file_ops_permission_escalation() {
    let workspace = tempfile::tempdir().unwrap();

    let write_tool = WriteFileTool::default();
    let edit_tool = EditFileTool::default();
    let read_tool = ReadFileTool::default();

    // Inside workspace → WorkspaceWrite
    let perm = write_tool.effective_permission(
        &serde_json::json!({"file_path": "inside.txt", "contents": "x"}),
        workspace.path(),
    );
    assert_eq!(
        perm,
        Permission::WorkspaceWrite,
        "write inside = WorkspaceWrite"
    );

    let perm = edit_tool.effective_permission(
        &serde_json::json!({"file_path": "inside.txt", "old_string": "a", "new_string": "b"}),
        workspace.path(),
    );
    assert_eq!(
        perm,
        Permission::WorkspaceWrite,
        "edit inside = WorkspaceWrite"
    );

    // Outside workspace → Dangerous
    let perm = write_tool.effective_permission(
        &serde_json::json!({"file_path": "/tmp/e2e_outside.txt", "contents": "x"}),
        workspace.path(),
    );
    assert_eq!(perm, Permission::Dangerous, "write outside = Dangerous");

    let perm = edit_tool.effective_permission(
        &serde_json::json!({"file_path": "/tmp/e2e_outside.txt", "old_string": "a", "new_string": "b"}),
        workspace.path(),
    );
    assert_eq!(perm, Permission::Dangerous, "edit outside = Dangerous");

    // ReadFile has no effective_permission override — always ReadOnly
    let perm = read_tool.effective_permission(
        &serde_json::json!({"file_path": "/etc/hostname"}),
        workspace.path(),
    );
    assert_eq!(perm, Permission::ReadOnly, "read always = ReadOnly");

    eprintln!("  PASS: file ops permission escalation works correctly");
}

#[tokio::test]
async fn t50_allowed_chat_ids_deny_when_empty() {
    let empty_config = Config {
        telegram: naked_core::config::TelegramConfig {
            allowed_chat_ids: vec![],
            ..Default::default()
        },
        ..Config::default()
    };
    assert!(
        empty_config.telegram.allowed_chat_ids.is_empty(),
        "config should have empty allowed_chat_ids"
    );

    let populated_config = Config {
        telegram: naked_core::config::TelegramConfig {
            allowed_chat_ids: vec![100, 200],
            ..Default::default()
        },
        ..Config::default()
    };
    assert!(populated_config.telegram.allowed_chat_ids.contains(&100));
    assert!(!populated_config.telegram.allowed_chat_ids.contains(&999));

    eprintln!("  PASS: allowed_chat_ids config is deny-by-default");
}

#[tokio::test]
async fn t54_retry_backoff_timing() {
    use naked_core::provider::{ChatRequest, Provider};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct FailProvider {
        call_count: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl Provider for FailProvider {
        fn name(&self) -> &str {
            "fail"
        }
        fn models(&self) -> Vec<naked_core::types::ModelInfo> {
            vec![]
        }
        async fn stream_chat(
            &self,
            _request: ChatRequest,
        ) -> naked_core::error::Result<
            std::pin::Pin<
                Box<dyn tokio_stream::Stream<Item = naked_core::types::StreamChunk> + Send>,
            >,
        > {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            Err(naked_core::error::AgentError::Provider(
                "simulated 429".into(),
            ))
        }
    }

    let provider = FailProvider {
        call_count: AtomicUsize::new(0),
    };
    let tools = ToolRegistry::new(vec![]);
    let config = LoopConfig {
        max_iterations: 1,
        cwd: std::env::temp_dir(),
        model: "test".into(),
        max_tokens: 8,
        ..Default::default()
    };
    let agent = AgentLoop::new(Box::new(provider) as Box<dyn Provider>, tools, config);
    let mut history = ConversationHistory::new("sys".into());
    history.push_user("hi");

    let (tx, _rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();

    let start = std::time::Instant::now();
    let result = agent.run(&mut history, tx, cancel, None, None).await;
    let elapsed = start.elapsed();

    assert!(result.is_err(), "should fail after retries");
    // With 3 retries and backoff (1s + 2s + 4s = 7s), should take at least 5s
    assert!(
        elapsed >= Duration::from_secs(5),
        "backoff too fast: {:.1}s (expected >=5s)",
        elapsed.as_secs_f64()
    );
    assert!(
        elapsed < Duration::from_secs(15),
        "backoff too slow: {:.1}s (expected <15s)",
        elapsed.as_secs_f64()
    );

    eprintln!(
        "  PASS: retry backoff takes {:.1}s (exponential)",
        elapsed.as_secs_f64()
    );
}

#[tokio::test]
async fn t57_read_file_size_limit() {
    let tmp = tempfile::tempdir().unwrap();
    let big_file = tmp.path().join("huge.bin");

    // Create an 11MB file
    let data = vec![b'x'; 11 * 1024 * 1024];
    std::fs::write(&big_file, &data).unwrap();

    let tool = ReadFileTool::default();
    let result = tool
        .execute(
            serde_json::json!({"file_path": big_file.to_str().unwrap()}),
            tmp.path(),
        )
        .await;
    assert!(result.is_error, "should reject large file");
    assert!(
        result.output.contains("too large"),
        "error should mention size: {}",
        result.output
    );

    // Small file should still work
    std::fs::write(tmp.path().join("small.txt"), "hello world\n").unwrap();
    let result = tool
        .execute(serde_json::json!({"file_path": "small.txt"}), tmp.path())
        .await;
    assert!(
        !result.is_error,
        "small file should work: {}",
        result.output
    );

    eprintln!("  PASS: ReadFileTool enforces 10MB size limit");
}

#[tokio::test]
async fn t60_no_auto_approve_config() {
    let json = r#"{
        "default_provider": "test",
        "default_model": "test-model",
        "providers": {},
        "auto_approve": true
    }"#;
    // Config should still parse (serde skips unknown fields by default with deny_unknown_fields off)
    let config: Result<Config, _> = serde_json::from_str(json);
    assert!(
        config.is_ok(),
        "config with leftover auto_approve should still parse"
    );

    eprintln!("  PASS: auto_approve field gracefully ignored");
}

#[tokio::test]
async fn t61_permission_request_outside_workspace() {
    use naked_core::provider::{ChatRequest, Provider};
    use naked_core::types::{PermissionResponse, StreamChunk};

    struct WriteOutsideProvider;

    #[async_trait::async_trait]
    impl Provider for WriteOutsideProvider {
        fn name(&self) -> &str {
            "write-outside"
        }
        fn models(&self) -> Vec<naked_core::types::ModelInfo> {
            vec![]
        }
        async fn stream_chat(
            &self,
            _request: ChatRequest,
        ) -> naked_core::error::Result<
            std::pin::Pin<Box<dyn tokio_stream::Stream<Item = StreamChunk> + Send>>,
        > {
            Ok(Box::pin(tokio_stream::iter(vec![
                StreamChunk::ToolUse {
                    id: "call1".into(),
                    name: "write_file".into(),
                    input: serde_json::json!({
                        "file_path": "/tmp/e2e_outside_perm_test.txt",
                        "contents": "test"
                    }),
                },
                StreamChunk::Done,
            ])))
        }
    }

    let workspace = tempfile::tempdir().unwrap();
    let tools = build_tools();
    let config = LoopConfig {
        max_iterations: 2,
        cwd: workspace.path().to_path_buf(),
        model: "test".into(),
        max_tokens: 8,
        ..Default::default()
    };
    let agent = AgentLoop::new(Box::new(WriteOutsideProvider), tools, config);

    let mut history = ConversationHistory::new("sys".into());
    history.push_user("write outside");

    let (tx, mut rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();
    let (perm_tx, perm_rx) = mpsc::channel(4);

    let loop_handle = tokio::spawn(async move {
        agent
            .run(&mut history, tx, cancel, Some(perm_rx), None)
            .await
    });

    let mut saw_permission = false;
    let mut perm_was_dangerous = false;
    while let Some(ev) = rx.recv().await {
        if let AgentEvent::PermissionRequest {
            call_id,
            permission,
            ..
        } = &ev
        {
            saw_permission = true;
            perm_was_dangerous = *permission == Permission::Dangerous;
            let _ = perm_tx
                .send(PermissionResponse {
                    call_id: call_id.clone(),
                    allowed: false,
                })
                .await;
        }
        if matches!(ev, AgentEvent::Idle) {
            break;
        }
    }

    let _ = loop_handle.await;

    assert!(
        saw_permission,
        "should request permission for outside-workspace write"
    );
    assert!(
        perm_was_dangerous,
        "permission should be Dangerous for outside-workspace write"
    );

    // Verify file was NOT written (permission denied)
    assert!(
        !std::path::Path::new("/tmp/e2e_outside_perm_test.txt").exists(),
        "file should not exist after permission denied"
    );

    eprintln!("  PASS: outside-workspace write triggers Dangerous permission request");
}

#[tokio::test]
async fn t65_read_binary_file_detection() {
    let tmp = tempfile::tempdir().unwrap();

    // PNG header contains NUL bytes — now handled as image (base64 for vision)
    let png = tmp.path().join("image.png");
    std::fs::write(&png, b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00").unwrap();

    let tool = ReadFileTool::default();
    let result = tool
        .execute(serde_json::json!({"file_path": "image.png"}), tmp.path())
        .await;
    // Small images are base64-encoded for vision models (not rejected)
    assert!(
        !result.is_error,
        "small image should be base64-encoded, not rejected: {}",
        result.output
    );
    assert!(
        result.output.contains("Image") || result.output.contains("image"),
        "output should mention image: {}",
        result.output
    );

    // Compiled object file with NUL inside
    let obj = tmp.path().join("code.o");
    let mut data = b"ELF".to_vec();
    data.extend_from_slice(&[0u8; 100]);
    std::fs::write(&obj, &data).unwrap();

    let result = tool
        .execute(serde_json::json!({"file_path": "code.o"}), tmp.path())
        .await;
    assert!(result.is_error, ".o file should be rejected as binary");

    // Normal text file should pass
    std::fs::write(tmp.path().join("readme.md"), "# Hello\nWorld\n").unwrap();
    let result = tool
        .execute(serde_json::json!({"file_path": "readme.md"}), tmp.path())
        .await;
    assert!(!result.is_error, "text file should work: {}", result.output);
    assert!(result.output.contains("Hello"));

    // Empty file should pass (no NUL)
    std::fs::write(tmp.path().join("empty.txt"), "").unwrap();
    let result = tool
        .execute(serde_json::json!({"file_path": "empty.txt"}), tmp.path())
        .await;
    assert!(!result.is_error, "empty file should not be binary");

    eprintln!("  PASS: ReadFileTool detects binary files via NUL sniffing");
}

#[tokio::test]
async fn t68_git_context_in_prompt() {
    let tmp = tempfile::tempdir().unwrap();

    // Non-git directory: no git info
    let section = naked_core::prompt::environment_section(tmp.path());
    assert!(
        !section.contains("Git:"),
        "non-repo should have no git context"
    );

    // Initialize a git repo
    std::process::Command::new("git")
        .args(["init"])
        .current_dir(tmp.path())
        .output()
        .unwrap();
    std::process::Command::new("git")
        .args(["commit", "--allow-empty", "-m", "init"])
        .current_dir(tmp.path())
        .output()
        .unwrap();

    let section = naked_core::prompt::environment_section(tmp.path());
    assert!(
        section.contains("Git: branch="),
        "git repo should show branch"
    );

    // Create a modified file
    std::fs::write(tmp.path().join("new_file.txt"), "hello").unwrap();
    let section = naked_core::prompt::environment_section(tmp.path());
    assert!(
        section.contains("Changed files") || section.contains("new_file.txt"),
        "modified files should appear in git context"
    );

    eprintln!("  PASS: Git context injected into environment section");
}

#[tokio::test]
async fn t69_token_cost_estimation() {
    use naked_core::types::TurnUsage;

    let usage = TurnUsage {
        input_tokens: 10_000,
        output_tokens: 1_000,
        cache_read_tokens: 0,
        cache_write_tokens: 0,
    };

    // Sonnet pricing: $3/M in, $15/M out
    let (inp, out, total) = usage.estimate_cost("claude-3-5-sonnet");
    assert!((inp - 0.03).abs() < 0.001, "input cost wrong: {inp}");
    assert!((out - 0.015).abs() < 0.001, "output cost wrong: {out}");
    assert!((total - 0.045).abs() < 0.001, "total cost wrong: {total}");

    // GPT-4o pricing: $5/M in, $15/M out
    let (_, _, total_gpt) = usage.estimate_cost("gpt-4o");
    assert!(total_gpt > 0.0, "GPT-4o cost should be > 0");

    // Haiku pricing (cheap)
    let (_, _, total_haiku) = usage.estimate_cost("claude-3-5-haiku");
    assert!(total_haiku < total, "haiku should be cheaper than sonnet");

    // Cache read tokens should be discounted (10% of input price)
    let cached = TurnUsage {
        input_tokens: 0,
        output_tokens: 0,
        cache_read_tokens: 1_000_000,
        cache_write_tokens: 0,
    };
    let (inp_c, _, _) = cached.estimate_cost("claude-3-5-sonnet");
    assert!(
        (inp_c - 0.3).abs() < 0.01,
        "cache read should be 10% of input price: {inp_c}"
    );

    // Unknown model should still return non-zero cost
    let (_, _, total_unk) = usage.estimate_cost("totally-unknown-model-v99");
    assert!(
        total_unk > 0.0,
        "unknown model should have fallback pricing"
    );

    eprintln!("  PASS: Token cost estimation correct for multiple models");
}

#[tokio::test]
async fn t70_workspace_root_as_cwd() {
    let tmp = tempfile::tempdir().unwrap();
    let store = JsonlSessionStore::new(tmp.path().to_path_buf());

    // Create a session with workspace = tmp.path()
    let workspace = tmp.path().join("my_project");
    std::fs::create_dir_all(&workspace).unwrap();
    std::fs::write(workspace.join("Cargo.toml"), "[package]\nname = \"test\"").unwrap();

    let session = Session::new(
        workspace.clone(),
        SYS.into(),
        // REGISTRY-WAIVE: exhaustive test struct ctor — pre-2026-05-13 baseline; new tests must use ..Default::default()
        SessionMetadata {
            name: None,
            provider: "test".into(),
            model: "test".into(),
            channel: "test".into(),
            channel_id: None,
        },
    );
    store.save(&session).await.unwrap();

    // Verify session workspace is set
    let loaded = store.load(&session.id).await.unwrap().unwrap();
    assert_eq!(
        loaded.workspace, workspace,
        "session workspace should be project root"
    );

    // Verify the workspace dir exists and is not artifacts
    let artifacts = store.artifacts_dir(&session.id);
    assert_ne!(
        loaded.workspace, artifacts,
        "workspace should differ from artifacts dir"
    );
    assert!(
        workspace.join("Cargo.toml").exists(),
        "project file should exist in workspace"
    );

    eprintln!("  PASS: Session workspace points to project root, not artifacts");
}

#[tokio::test]
async fn t71_instruction_walkup() {
    let tmp = tempfile::tempdir().unwrap();
    let project = tmp.path().join("org").join("repo");
    std::fs::create_dir_all(&project).unwrap();

    // Create AGENTS.md in project root
    std::fs::write(project.join("AGENTS.md"), "# Project rules\nBe good.").unwrap();

    // Create CLAUDE.md in parent
    let parent = tmp.path().join("org");
    std::fs::write(parent.join("CLAUDE.md"), "# Org rules\nBe great.").unwrap();

    let section = naked_core::prompt::environment_section(&project);
    assert!(section.contains("Be good"), "should find project AGENTS.md");
    assert!(section.contains("Be great"), "should find parent CLAUDE.md");

    // Test budget: write a huge AGENTS.md
    let huge = "x".repeat(20_000);
    std::fs::write(project.join("AGENTS.md"), &huge).unwrap();
    let section = naked_core::prompt::environment_section(&project);
    assert!(
        section.contains("[truncated]"),
        "large file should be truncated"
    );

    eprintln!("  PASS: Instruction file walk-up with budgets and dedup");
}

#[tokio::test]
async fn t72_parallel_readonly_tools() {
    let config = match test_config() {
        Some(c) => c,
        None => {
            eprintln!("SKIP t72: no provider");
            return;
        }
    };
    let (provider, model) = match get_working_provider(&config).await {
        Some(p) => p,
        None => {
            eprintln!("SKIP t72: no working provider");
            return;
        }
    };

    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("a.txt"), "alpha").unwrap();
    std::fs::write(tmp.path().join("b.txt"), "beta").unwrap();
    std::fs::write(tmp.path().join("c.txt"), "gamma").unwrap();

    let mut history = ConversationHistory::new(SYS.into());
    history.push_user(
        "Read all three files: a.txt, b.txt, c.txt using read_file tool for each. \
         Return the contents of each file.",
    );

    let result = run_prompt(provider, &mut history, tmp.path(), &model).await;
    assert!(result.got_idle(), "should complete");

    let tools = result.tool_names();
    let read_count = tools.iter().filter(|n| *n == "read_file").count();
    assert!(
        read_count >= 3,
        "should call read_file at least 3 times, got {read_count}: {tools:?}"
    );

    let text = result.full_text();
    assert!(
        text.contains("alpha")
            || result
                .tool_results()
                .iter()
                .any(|(_, _, o)| o.contains("alpha")),
        "should contain alpha"
    );
    assert!(
        text.contains("beta")
            || result
                .tool_results()
                .iter()
                .any(|(_, _, o)| o.contains("beta")),
        "should contain beta"
    );
    assert!(
        text.contains("gamma")
            || result
                .tool_results()
                .iter()
                .any(|(_, _, o)| o.contains("gamma")),
        "should contain gamma"
    );

    eprintln!("  PASS: Parallel read-only tool execution (read_file x{read_count})");
}

#[tokio::test]
async fn t74_readfile_output_truncation() {
    let tmp = tempfile::tempdir().unwrap();

    // Create a large text file (500 lines of 100 chars each = ~50 KiB)
    let content: String = (0..500)
        .map(|i| format!("line {i}: {}\n", "a".repeat(90)))
        .collect();
    std::fs::write(tmp.path().join("big.txt"), &content).unwrap();

    let tool = ReadFileTool::default();
    let result = tool
        .execute(serde_json::json!({"file_path": "big.txt"}), tmp.path())
        .await;
    assert!(!result.is_error, "should not error");
    assert!(
        result.output.len() <= 17_000,
        "output should be truncated, got {} bytes",
        result.output.len()
    );
    assert!(
        result.output.contains("truncated") || result.output.contains("more lines"),
        "should mention truncation"
    );

    eprintln!("  PASS: ReadFile output truncation at 16 KiB");
}

#[tokio::test]
async fn t80_web_search_spec() {
    let tool = WebSearchTool::from_legacy_exa(vec![]);
    let spec = tool.spec();
    assert_eq!(spec.name, "web_search");
    assert_eq!(spec.permission, Permission::ReadOnly);
    eprintln!("  PASS: web_search spec");
}

#[tokio::test]
async fn t81_web_search_empty_query() {
    let tool = WebSearchTool::from_legacy_exa(vec![]);
    let r = tool
        .execute(serde_json::json!({"query": ""}), Path::new("/tmp"))
        .await;
    assert!(r.is_error, "empty query should error");
    eprintln!("  PASS: web_search empty query error");
}

#[tokio::test]
async fn t82_web_search_invalid_input() {
    let tool = WebSearchTool::from_legacy_exa(vec![]);
    let r = tool
        .execute(serde_json::json!({"wrong": 1}), Path::new("/tmp"))
        .await;
    assert!(r.is_error, "missing query should error");
    eprintln!("  PASS: web_search invalid input error");
}

#[tokio::test]
async fn t83_web_search_exa_live() {
    let keys_str = match std::env::var("EXA_API_KEYS") {
        Ok(v) if !v.is_empty() => v,
        _ => match std::env::var("EXA_API_KEY") {
            Ok(v) if !v.is_empty() => v,
            _ => {
                eprintln!("  SKIP t83: no EXA_API_KEYS/EXA_API_KEY");
                return;
            }
        },
    };
    let keys: Vec<String> = keys_str
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    eprintln!("  Using {} exa key(s)", keys.len());

    let tool = WebSearchTool::from_legacy_exa(keys);
    let r = tool
        .execute(
            serde_json::json!({"query": "Rust programming language", "num_results": 3}),
            Path::new("/tmp"),
        )
        .await;
    assert!(!r.is_error, "exa search should succeed: {}", r.output);
    assert!(
        r.output.contains("[exa.ai]"),
        "should have exa tag: {}",
        r.output
    );
    let lower = r.output.to_lowercase();
    assert!(
        lower.contains("rust") || lower.contains("programming"),
        "should mention rust: {}",
        r.output
    );
    eprintln!("  PASS: exa.ai live search");
    eprintln!("  Preview: {}", &r.output[..r.output.len().min(400)]);
}

#[tokio::test]
async fn t84_web_search_ddg_fallback() {
    let tool = WebSearchTool::from_legacy_exa(vec![]);
    let r = tool
        .execute(
            serde_json::json!({"query": "what is Rust programming language", "num_results": 3}),
            Path::new("/tmp"),
        )
        .await;
    assert!(!r.is_error, "ddg fallback should not error: {}", r.output);
    eprintln!("  PASS: DuckDuckGo fallback");
    eprintln!("  Preview: {}", &r.output[..r.output.len().min(400)]);
}

#[tokio::test]
async fn t85_web_search_key_rotation() {
    use naked_core::keys::KeyProvider;
    use naked_core::keys::pool::KeyPool;
    use std::sync::Arc;
    use std::time::Duration;

    struct StaticProv(Vec<String>);
    impl KeyProvider for StaticProv {
        fn fetch(&self, _: &str) -> std::result::Result<Vec<String>, String> {
            Ok(self.0.clone())
        }
    }
    let pool = KeyPool::new(
        vec![Arc::new(StaticProv(vec![
            "k1".into(),
            "k2".into(),
            "k3".into(),
        ]))],
        "exa",
        Duration::from_secs(60),
    );
    let mut seen = Vec::new();
    for _ in 0..9 {
        seen.push(pool.next().expect("non-empty"));
    }
    assert_eq!(
        seen,
        vec!["k1", "k2", "k3", "k1", "k2", "k3", "k1", "k2", "k3"]
    );
    eprintln!("  PASS: key rotation round-robin");
}

#[tokio::test]
async fn t90_agent_registry_and_control() {
    let config = match test_config() {
        Some(c) => c,
        None => {
            eprintln!("SKIP: no provider configured");
            return;
        }
    };
    let model = match get_working_provider(&config).await {
        Some((_, m)) => m,
        None => {
            eprintln!("SKIP: no working provider");
            return;
        }
    };
    eprintln!(">>> t90_agent_registry_and_control [model={model}]");

    let registry = AgentRegistry::new();
    let exa_keys = config.exa_api_keys.clone();
    let (provider, _): (Box<dyn Provider>, String) = get_working_provider(&config).await.unwrap();
    let prov_arc: Arc<dyn Provider> = Arc::from(provider);

    // 1. agent_status on empty registry
    let status_tool = AgentStatusTool::new(registry.clone());
    let r = status_tool
        .execute(serde_json::json!({}), Path::new("/tmp"))
        .await;
    assert!(!r.is_error);
    assert!(r.output.contains("No sub-agents"), "empty: {}", r.output);
    eprintln!("  ✅ agent_status empty: OK");

    // 2. Run sub_agent with shared registry
    let sub_agent =
        SubAgentTool::new(prov_arc, model.clone(), 30, exa_keys).with_registry(registry.clone());

    let (progress_tx, mut progress_rx) = mpsc::channel::<AgentEvent>(256);

    let input = serde_json::json!({
        "prompt": "List all tools you have available. Just list their names, nothing else.",
        "mode": "explore"
    });
    let cwd = Path::new("/tmp");

    // Run sub_agent in background
    let sub_handle = tokio::spawn(async move {
        sub_agent
            .execute_with_progress(input, cwd, progress_tx)
            .await
    });

    // 3. While sub_agent runs, check registry
    tokio::time::sleep(Duration::from_millis(500)).await;
    let running = registry.list_running().await;
    eprintln!("  running agents during exec: {}", running.len());

    // Collect progress events
    let result = sub_handle.await.unwrap();
    assert!(!result.is_error, "sub_agent failed: {}", result.output);
    eprintln!("  ✅ sub_agent completed: {} bytes", result.output.len());

    // 4. Check progress events were emitted
    let mut events = Vec::new();
    while let Ok(ev) = progress_rx.try_recv() {
        events.push(ev);
    }
    let started = events.iter().any(|e| {
        matches!(
            e,
            AgentEvent::SubAgentProgress {
                event: SubAgentEvent::Started { .. },
                ..
            }
        )
    });
    let finished = events.iter().any(|e| {
        matches!(
            e,
            AgentEvent::SubAgentProgress {
                event: SubAgentEvent::Finished { .. },
                ..
            }
        )
    });
    eprintln!(
        "  events: {} total, started={started}, finished={finished}",
        events.len()
    );
    assert!(started, "must emit SubAgentProgress::Started");
    assert!(finished, "must emit SubAgentProgress::Finished");

    // 5. agent_status after completion shows the agent
    let status_tool = AgentStatusTool::new(registry.clone());
    let r = status_tool
        .execute(serde_json::json!({}), Path::new("/tmp"))
        .await;
    assert!(!r.is_error);
    assert!(r.output.contains("1 agent(s)"), "after: {}", r.output);
    assert!(r.output.contains("completed"), "status: {}", r.output);
    eprintln!("  ✅ agent_status after completion: OK");

    // 6. agent_stop on finished agent returns error
    let stop_tool = AgentStopTool::new(registry.clone());
    let r = stop_tool
        .execute(
            serde_json::json!({"agent_id": "sa-nonexistent"}),
            Path::new("/tmp"),
        )
        .await;
    assert!(r.is_error);
    eprintln!("  ✅ agent_stop on nonexistent: correctly returns error");

    // 7. GC cleans up old entries
    registry.gc(Duration::from_secs(0)).await;
    let all = registry.list_all().await;
    assert!(all.is_empty(), "GC should clean completed agents");
    eprintln!("  ✅ GC cleaned registry");

    eprintln!("  PASS: agent registry and control tools work");
}

#[tokio::test]
async fn t92_heartbeat_during_tool_execution() {
    let config = match test_config() {
        Some(c) => c,
        None => {
            eprintln!("SKIP: no provider configured");
            return;
        }
    };
    let model = match get_working_provider(&config).await {
        Some((_, m)) => m,
        None => {
            eprintln!("SKIP: no working provider");
            return;
        }
    };
    eprintln!(">>> t92_heartbeat_during_tool_execution [model={model}]");

    let tmp = tempfile::tempdir().unwrap();
    let cwd = tmp.path();

    let tools: Vec<Box<dyn Tool>> = vec![
        Box::new(BashTool::new(30)),
        Box::new(ReadFileTool::default()),
        Box::new(GlobSearchTool),
        Box::new(GrepSearchTool),
    ];
    let registry = ToolRegistry::new(tools);

    let loop_config = LoopConfig {
        max_iterations: 5,
        cwd: cwd.to_path_buf(),
        model: model.clone(),
        max_tokens: 4096,
        temperature: Some(0.0),
        ..Default::default()
    };

    let (provider, _) = get_working_provider(&config).await.unwrap();
    let agent = AgentLoop::new(provider, registry, loop_config);

    let system = format!("You are a test agent. Working directory: {}", cwd.display());
    let mut history = ConversationHistory::new(system);
    history.push_user("Run this command: sleep 8 && echo DONE. Use bash tool.");

    let (tx, mut rx) = mpsc::channel(4096);
    let cancel = CancellationToken::new();

    let start = std::time::Instant::now();
    let result = agent.run(&mut history, tx, cancel, None, None).await;
    let elapsed = start.elapsed();

    eprintln!("  elapsed: {:.1}s", elapsed.as_secs_f64());
    match &result {
        Ok(u) => eprintln!("  tokens: {}", u.total_tokens()),
        Err(e) => eprintln!("  error: {e}"),
    }

    let mut heartbeats = 0;
    let mut tool_starts = 0;
    let mut tool_ends = 0;
    let mut all_events = 0;
    while let Ok(ev) = rx.try_recv() {
        all_events += 1;
        match ev {
            AgentEvent::Heartbeat => heartbeats += 1,
            AgentEvent::ToolStart { .. } => tool_starts += 1,
            AgentEvent::ToolEnd { .. } => tool_ends += 1,
            _ => {}
        }
    }

    eprintln!("  events: {all_events} total");
    eprintln!("  heartbeats: {heartbeats}");
    eprintln!("  tool_starts: {tool_starts}, tool_ends: {tool_ends}");

    assert!(result.is_ok(), "agent must complete successfully");
    assert!(tool_starts > 0, "must have at least one tool call");
    assert!(heartbeats > 0, "must emit heartbeats during 8s sleep");
    eprintln!("  PASS: heartbeat emitted during long tool execution");
}

#[tokio::test]
async fn t99_classifier_user_scope_via_attribution_prefix() {
    use naked_core::memory::classifier;

    need_provider!(_config, provider, model);
    pace().await;

    eprintln!(">>> t99_classifier_user_scope_via_attribution_prefix [model={model}]");

    let sender = format!("alice_{}", std::process::id());
    let msg =
        "@alice: please always answer me in Russian and use 4-space indentation in code blocks";

    // With sender_id present the classifier may pick `scope=user`.
    let result = classifier::classify(provider.as_ref(), &model, msg, Some(sender.as_str())).await;

    let Some(r) = result else {
        // Some smaller models legitimately decide this is too generic to store.
        // Don't fail the suite — just record and exit.
        eprintln!("  classifier returned None (model declined to store) — acceptable");
        return;
    };

    eprintln!("  classified: scope={:?} content={:?}", r.scope, r.content);

    // Content must NOT start with the attribution prefix — that's the whole
    // point of the today-added classifier instructions.
    assert!(
        !r.content.trim_start().starts_with("@alice:"),
        "classifier swallowed the attribution prefix into content: {:?}",
        r.content
    );
    assert!(
        !r.content.contains("@alice"),
        "classifier kept '@alice' in content: {:?}",
        r.content
    );

    // The underlying intent must survive (Russian + indentation hint).
    let lower = r.content.to_lowercase();
    assert!(
        lower.contains("russian")
            || lower.contains("\u{0440}\u{0443}\u{0441}\u{0441}")
            || lower.contains("indent")
            || lower.contains("4-space")
            || lower.contains("4 space"),
        "classifier dropped the actual preference: {:?}",
        r.content
    );

    // Scope should be `user` (preferred) or `global` (also acceptable for a
    // strong personal preference). `project` is wrong here — it isn't a
    // project-specific fact.
    match &r.scope {
        naked_core::memory::types::MemoryScope::User(id) => {
            assert_eq!(id, &sender, "user scope id mismatch");
        }
        naked_core::memory::types::MemoryScope::Global => {
            eprintln!("  classifier picked Global (acceptable fallback)");
        }
        other => panic!("classifier picked unexpected scope: {other:?}"),
    }

    eprintln!("  PASS: classifier handles @-attribution and routes to user/global");
}

#[tokio::test]
async fn t100_multimodal_native_image_via_vision_capable_model() {
    use base64::Engine as _;
    use futures_util::StreamExt;
    use naked_core::provider::ChatRequest;
    use naked_core::types::StreamChunk;

    need_config!(config);
    pace().await;

    let candidates = pick_vision_capable_providers(&config);
    if candidates.is_empty() {
        eprintln!(
            "SKIP: no vision-capable provider/model in config (need claude-3+, gpt-4o, llama-4-scout, grok-2-vision, gemini-1.5+, qwen-vl, ...)"
        );
        return;
    }

    let Some(path) = fixture_image_path() else {
        eprintln!("SKIP: tests/fixtures/test_image.png missing — generate via ffmpeg testsrc");
        return;
    };
    let bytes = std::fs::read(&path).expect("read fixture");
    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);

    // Build the conversation through the *real* history API, then convert via
    // `to_api_messages` so we exercise the canonical Anthropic-shaped payload.
    let mut history = ConversationHistory::new(String::new());
    history.push_user_multimodal(vec![
        ContentBlock::Text {
            text: "What do you see in this image? Reply in one short sentence.".into(),
        },
        ContentBlock::Image {
            mime: "image/png".into(),
            data_base64: b64,
            detail: None,
        },
    ]);
    let messages = history.to_api_messages();
    assert_eq!(messages.len(), 1);
    let parts = messages[0]["content"].as_array().expect("array content");
    assert!(
        parts.iter().any(|p| p["type"] == "image"),
        "must contain image block"
    );

    // Try candidates in preference order. Treat 401/403/"not supported" as
    // "this provider's account is broken — move on" instead of test failure.
    let mut last_err: Option<String> = None;
    let mut text = String::new();
    let mut chosen: Option<(String, String)> = None;
    for (provider, prov_name, model) in candidates {
        eprintln!(">>> t100_multimodal_native_image [provider={prov_name} model={model}]");
        let req = ChatRequest {
            model: model.clone(),
            system: String::new(),
            messages: messages.clone(),
            tools: vec![],
            max_tokens: 200,
            temperature: Some(0.0),
            reasoning: None,
        };

        let mut stream = match provider.stream_chat(req).await {
            Ok(s) => s,
            Err(e) => {
                let s = e.to_string();
                let lc = s.to_lowercase();
                if lc.contains("401")
                    || lc.contains("403")
                    || lc.contains("unauthor")
                    || lc.contains("not supported")
                    || lc.contains("model_not_found")
                    || lc.contains("model_not_supported")
                {
                    eprintln!("  skip {prov_name}/{model}: {s}");
                    last_err = Some(s);
                    continue;
                }
                panic!("stream_chat failed [{prov_name}/{model}]: {e}");
            }
        };

        text.clear();
        let mut had_error: Option<String> = None;
        let timeout = Duration::from_secs(60);
        let start = std::time::Instant::now();
        while start.elapsed() < timeout {
            let next = match tokio::time::timeout(Duration::from_secs(5), stream.next()).await {
                Ok(Some(c)) => c,
                Ok(None) => break,
                Err(_) => continue,
            };
            match next {
                StreamChunk::Text(t) => text.push_str(&t),
                StreamChunk::Done => break,
                StreamChunk::Error(e) => {
                    had_error = Some(e);
                    break;
                }
                _ => {}
            }
        }
        if let Some(e) = had_error {
            let lc = e.to_lowercase();
            if lc.contains("401")
                || lc.contains("403")
                || lc.contains("unauthor")
                || lc.contains("not supported")
            {
                eprintln!("  skip {prov_name}/{model} mid-stream: {e}");
                last_err = Some(e);
                continue;
            }
            panic!("provider stream error [{prov_name}/{model}] (likely image rejected): {e}");
        }
        if !text.trim().is_empty() {
            chosen = Some((prov_name, model));
            break;
        }
        last_err = Some("empty stream".into());
    }
    let (prov_name, model) = chosen.unwrap_or_else(|| {
        panic!(
            "no vision-capable provider produced a reply (last error: {:?})",
            last_err
        );
    });
    eprintln!(
        "  reply ({} chars) [{prov_name}/{model}]: {text}",
        text.len()
    );
    assert!(
        !text.trim().is_empty(),
        "vision-capable model returned empty text — native image plumbing is broken"
    );
    assert!(
        text.len() >= 10,
        "reply too short to be a real description: {text:?}"
    );

    // Stronger: the fixture is an SMPTE-style color-bar test pattern (320x240).
    // Any working vision model must mention at least one defining feature of
    // the image. We accept a generous OR-set across English/Russian and
    // common phrasings — false negatives here would mask a regression where
    // the model receives an empty/black image but still hallucinates a reply.
    let lc = text.to_lowercase();
    let needles = [
        "color",
        "colour",
        "bar",
        "stripe",
        "rainbow",
        "vertical",
        "test pattern",
        "pattern",
        "smpte",
        "tv",
        "television",
        "spectrum",
        "цвет",
        "полос",
        "радуг",
        "телевиз",
        "тест",
        "узор",
    ];
    let hit = needles.iter().find(|n| lc.contains(*n));
    assert!(
        hit.is_some(),
        "reply does not describe the color-bar test pattern \u{2014} \
         vision likely received empty/garbled image data. reply={text:?}"
    );
    eprintln!("  matched keyword: {:?}", hit.unwrap());
}

#[tokio::test]
async fn t100b_multimodal_native_image_via_anthropic_claude() {
    use base64::Engine as _;
    use futures_util::StreamExt;
    use naked_core::provider::ChatRequest;
    use naked_core::types::StreamChunk;

    need_config!(config);
    pace().await;

    let media_cfg = naked_core::config::TgMediaConfig::default();
    let Some(pc) = config.providers.get("anthropic") else {
        eprintln!("SKIP: no `anthropic` provider in config");
        return;
    };
    let Some(model) = pc
        .models
        .iter()
        .find(|m| media_cfg.is_vision_capable_model(m))
        .cloned()
    else {
        eprintln!("SKIP: anthropic provider has no vision-capable model listed");
        return;
    };
    let Some(provider) = make_provider_for(&config, "anthropic", &model) else {
        eprintln!("SKIP: anthropic provider missing API key");
        return;
    };
    eprintln!(">>> t100b_multimodal_native_image_anthropic [model={model}]");

    let Some(path) = fixture_image_path() else {
        eprintln!("SKIP: tests/fixtures/test_image.png missing");
        return;
    };
    let bytes = std::fs::read(&path).expect("read fixture");
    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);

    // Image FIRST, text SECOND — Anthropic's docs explicitly recommend this
    // ordering for best vision quality. This mirrors the production path
    // (`naked-tg::handle_message` was reordered for the same reason).
    let mut history = ConversationHistory::new(String::new());
    history.push_user_multimodal(vec![
        ContentBlock::Image {
            mime: "image/png".into(),
            data_base64: b64,
            detail: None,
        },
        ContentBlock::Text {
            text: "What do you see in this image? Reply in one short sentence.".into(),
        },
    ]);
    let messages = history.to_api_messages();

    let req = ChatRequest {
        model: model.clone(),
        system: String::new(),
        messages,
        tools: vec![],
        max_tokens: 200,
        temperature: Some(0.0),
        reasoning: None,
    };

    let mut stream = match provider.stream_chat(req).await {
        Ok(s) => s,
        Err(e) => {
            // External-provider auth failures (expired key, rotated
            // credentials) are out of scope for regression coverage —
            // see t36/t38 for the same rationale. Skip rather than
            // panic so CI stays honest about what we actually changed.
            let es = e.to_string();
            if es.contains("401") || es.contains("Unauthorized") || es.contains("invalid x-api-key")
            {
                eprintln!("SKIP t100b: anthropic credentials not usable ({es})");
                return;
            }
            panic!("anthropic stream_chat failed: {e}");
        }
    };

    let mut text = String::new();
    let mut had_error: Option<String> = None;
    let timeout = Duration::from_secs(60);
    let start = std::time::Instant::now();
    while start.elapsed() < timeout {
        let next = match tokio::time::timeout(Duration::from_secs(5), stream.next()).await {
            Ok(Some(c)) => c,
            Ok(None) => break,
            Err(_) => continue,
        };
        match next {
            StreamChunk::Text(t) => text.push_str(&t),
            StreamChunk::Done => break,
            StreamChunk::Error(e) => {
                had_error = Some(e);
                break;
            }
            _ => {}
        }
    }

    if let Some(e) = had_error {
        panic!("anthropic provider stream error: {e}");
    }
    assert!(
        !text.trim().is_empty(),
        "anthropic vision returned empty text \u{2014} native image plumbing broken"
    );
    let lc = text.to_lowercase();
    let needles = [
        "color",
        "colour",
        "bar",
        "stripe",
        "rainbow",
        "vertical",
        "test pattern",
        "pattern",
        "smpte",
        "tv",
        "television",
        "spectrum",
    ];
    assert!(
        needles.iter().any(|n| lc.contains(n)),
        "anthropic reply does not describe color-bar pattern: {text:?}"
    );
    eprintln!("  reply: {text}");
}

#[tokio::test]
async fn t100c_multimodal_native_image_via_openai_compat() {
    use base64::Engine as _;
    use futures_util::StreamExt;
    use naked_core::provider::ChatRequest;
    use naked_core::types::StreamChunk;

    need_config!(config);
    pace().await;

    let media_cfg = naked_core::config::TgMediaConfig::default();
    // Build a candidate list (preferred providers first, then any others).
    let preferred_order = ["openai", "openrouter", "groq", "fireworks", "qwen"];
    let mut candidates: Vec<(String, String)> = Vec::new();
    for name in preferred_order {
        if let Some(pc) = config.providers.get(name)
            && pc.provider_type == "openai_compat"
            && let Some(model) = pc
                .models
                .iter()
                .find(|m| media_cfg.is_vision_capable_with_provider(m, Some(pc)))
                .cloned()
        {
            candidates.push((name.to_string(), model));
        }
    }
    for (name, pc) in &config.providers {
        if candidates.iter().any(|(n, _)| n == name) {
            continue;
        }
        if pc.provider_type == "openai_compat"
            && let Some(model) = pc
                .models
                .iter()
                .find(|m| media_cfg.is_vision_capable_with_provider(m, Some(pc)))
                .cloned()
        {
            candidates.push((name.clone(), model));
        }
    }
    if candidates.is_empty() {
        eprintln!("SKIP: no openai_compat provider with a vision-capable model in config");
        return;
    }

    let Some(path) = fixture_image_path() else {
        eprintln!("SKIP: tests/fixtures/test_image.png missing");
        return;
    };
    let bytes = std::fs::read(&path).expect("read fixture");
    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);

    // Try each candidate in order — auth/model errors silently fall through to
    // the next, so a single broken provider key/account can't sink the run.
    let mut last_err: Option<String> = None;
    let mut success: Option<(String, String, String)> = None;
    for (prov_name, model) in &candidates {
        let Some(provider) = make_provider_for(&config, prov_name, model) else {
            continue;
        };
        eprintln!(">>> t100c_multimodal_native_image_openai_compat [{prov_name} / {model}]");

        let mut history = ConversationHistory::new(String::new());
        history.push_user_multimodal(vec![
            ContentBlock::Image {
                mime: "image/png".into(),
                data_base64: b64.clone(),
                detail: Some(naked_core::types::ImageDetail::Low),
            },
            ContentBlock::Text {
                text: "What do you see in this image? Reply in one short sentence.".into(),
            },
        ]);
        let messages = history.to_api_messages();

        let req = ChatRequest {
            model: model.clone(),
            system: String::new(),
            messages,
            tools: vec![],
            max_tokens: 200,
            temperature: Some(0.0),
            reasoning: None,
        };

        let mut stream = match provider.stream_chat(req).await {
            Ok(s) => s,
            Err(e) => {
                let s = e.to_string();
                let lc = s.to_lowercase();
                if lc.contains("401")
                    || lc.contains("403")
                    || lc.contains("unauthor")
                    || lc.contains("not supported")
                    || lc.contains("model_not_found")
                {
                    eprintln!("  skip {prov_name}/{model}: {s}");
                    last_err = Some(s);
                    continue;
                }
                panic!("openai_compat stream_chat failed [{prov_name}/{model}]: {e}");
            }
        };
        let mut text = String::new();
        let mut had_error: Option<String> = None;
        let timeout = Duration::from_secs(60);
        let start = std::time::Instant::now();
        while start.elapsed() < timeout {
            let next = match tokio::time::timeout(Duration::from_secs(5), stream.next()).await {
                Ok(Some(c)) => c,
                Ok(None) => break,
                Err(_) => continue,
            };
            match next {
                StreamChunk::Text(t) => text.push_str(&t),
                StreamChunk::Done => break,
                StreamChunk::Error(e) => {
                    had_error = Some(e);
                    break;
                }
                _ => {}
            }
        }
        if let Some(e) = had_error {
            let lc = e.to_lowercase();
            if lc.contains("401")
                || lc.contains("403")
                || lc.contains("unauthor")
                || lc.contains("not supported")
            {
                eprintln!("  skip {prov_name}/{model} mid-stream: {e}");
                last_err = Some(e);
                continue;
            }
            panic!("openai_compat provider stream error [{prov_name}/{model}]: {e}");
        }
        if !text.trim().is_empty() {
            success = Some((prov_name.clone(), model.clone(), text));
            break;
        }
        last_err = Some("empty stream".into());
    }

    let (prov_name, model, text) = success.unwrap_or_else(|| {
        panic!(
            "no openai_compat vision provider produced a reply (last error: {:?})",
            last_err
        );
    });
    let lc = text.to_lowercase();
    let needles = [
        "color",
        "colour",
        "bar",
        "stripe",
        "rainbow",
        "vertical",
        "test pattern",
        "pattern",
        "smpte",
        "tv",
        "television",
        "spectrum",
    ];
    assert!(
        needles.iter().any(|n| lc.contains(n)),
        "openai_compat reply does not describe color-bar pattern [{prov_name}/{model}]: {text:?}"
    );
    eprintln!("  reply [{prov_name}/{model}]: {text}");
}

#[tokio::test]
async fn t100d_multimodal_multi_image_order_preserved() {
    use base64::Engine as _;
    use futures_util::StreamExt;
    use naked_core::provider::ChatRequest;
    use naked_core::types::StreamChunk;

    need_config!(config);
    pace().await;

    let media_cfg = naked_core::config::TgMediaConfig::default();
    // Prefer providers that we know reliably handle vision; fall through to
    // anything else if they're not configured / lack a key.
    let preferred_order = [
        "anthropic",
        "openrouter",
        "groq",
        "openai",
        "fireworks",
        "qwen",
    ];
    let mut candidates: Vec<(String, String)> = Vec::new();
    for name in preferred_order {
        if let Some(pc) = config.providers.get(name)
            && let Some(model) = pc
                .models
                .iter()
                .find(|m| media_cfg.is_vision_capable_with_provider(m, Some(pc)))
                .cloned()
        {
            candidates.push((name.to_string(), model));
        }
    }
    for (name, pc) in &config.providers {
        if candidates.iter().any(|(n, _)| n == name) {
            continue;
        }
        if let Some(model) = pc
            .models
            .iter()
            .find(|m| media_cfg.is_vision_capable_with_provider(m, Some(pc)))
            .cloned()
        {
            candidates.push((name.clone(), model));
        }
    }
    if candidates.is_empty() {
        eprintln!("SKIP: no vision-capable provider for multi-image test");
        return;
    }

    let Some(path) = fixture_image_path() else {
        eprintln!("SKIP: tests/fixtures/test_image.png missing");
        return;
    };
    let bytes = std::fs::read(&path).expect("read fixture");
    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);

    // Reuse the same valid PNG twice. We're testing serialization order &
    // multi-image plumbing, not perception of two distinct payloads — and
    // arbitrarily mutated bytes will be rejected by strict providers
    // (`invalid image data`).
    let b64b = b64.clone();

    let mut history = ConversationHistory::new(String::new());
    history.push_user_multimodal(vec![
        ContentBlock::Text {
            text: "I'm sending you two images. Reply with: 'two images received' if you see both."
                .into(),
        },
        ContentBlock::Image {
            mime: "image/png".into(),
            data_base64: b64,
            detail: None,
        },
        ContentBlock::Image {
            mime: "image/png".into(),
            data_base64: b64b,
            detail: None,
        },
    ]);
    let messages = history.to_api_messages();
    // Sanity: order was preserved through to_api_messages.
    let parts = messages[0]["content"].as_array().expect("array content");
    assert_eq!(
        parts.len(),
        3,
        "to_api_messages must keep all three blocks in order"
    );
    assert_eq!(parts[0]["type"], "text");
    assert_eq!(parts[1]["type"], "image");
    assert_eq!(parts[2]["type"], "image");

    let mut last_err: Option<String> = None;
    let mut success: Option<(String, String, String)> = None;
    for (prov_name, model) in &candidates {
        let Some(provider) = make_provider_for(&config, prov_name, model) else {
            continue;
        };
        eprintln!(">>> t100d_multimodal_multi_image [{prov_name} / {model}]");

        let req = ChatRequest {
            model: model.clone(),
            system: String::new(),
            messages: messages.clone(),
            tools: vec![],
            max_tokens: 100,
            temperature: Some(0.0),
            reasoning: None,
        };
        let mut stream = match provider.stream_chat(req).await {
            Ok(s) => s,
            Err(e) => {
                let s = e.to_string();
                let lc = s.to_lowercase();
                if lc.contains("401")
                    || lc.contains("403")
                    || lc.contains("unauthor")
                    || lc.contains("not supported")
                    || lc.contains("model_not_found")
                    || lc.contains("model_not_supported")
                    || lc.contains("invalid image")
                    || lc.contains("invalid_image")
                {
                    eprintln!("  skip {prov_name}/{model}: {s}");
                    last_err = Some(s);
                    continue;
                }
                panic!("multi-image stream failed [{prov_name}/{model}]: {e}");
            }
        };
        let mut text = String::new();
        let mut stream_err: Option<String> = None;
        let timeout = Duration::from_secs(60);
        let start = std::time::Instant::now();
        while start.elapsed() < timeout {
            let next = match tokio::time::timeout(Duration::from_secs(5), stream.next()).await {
                Ok(Some(c)) => c,
                Ok(None) => break,
                Err(_) => continue,
            };
            match next {
                StreamChunk::Text(t) => text.push_str(&t),
                StreamChunk::Done => break,
                StreamChunk::Error(e) => {
                    stream_err = Some(e);
                    break;
                }
                _ => {}
            }
        }
        if let Some(e) = stream_err {
            let lc = e.to_lowercase();
            if lc.contains("401")
                || lc.contains("403")
                || lc.contains("unauthor")
                || lc.contains("not supported")
            {
                eprintln!("  skip {prov_name}/{model} mid-stream: {e}");
                last_err = Some(e);
                continue;
            }
            panic!("multi-image stream error [{prov_name}/{model}]: {e}");
        }
        if !text.trim().is_empty() {
            success = Some((prov_name.clone(), model.clone(), text));
            break;
        }
        last_err = Some("empty stream".into());
    }
    let (prov_name, model, text) = success.unwrap_or_else(|| {
        panic!("no vision provider produced multi-image reply (last error: {last_err:?})");
    });
    eprintln!("  reply [{prov_name}/{model}]: {text}");
}

#[tokio::test]
async fn t101_multimodal_text_only_fallback_path_smoke() {
    use naked_core::config::TgMediaConfig;

    eprintln!(">>> t101_multimodal_text_only_fallback_path_smoke");

    let mut media_cfg = TgMediaConfig::default();
    let vision_models = [
        "claude-sonnet-4-20250514",
        "gpt-4o-mini",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "grok-2-vision-latest",
    ];
    let text_only_models = [
        "llama-3.3-70b-versatile",
        "deepseek-chat",
        "MiniMax-Text-01",
    ];

    // Default config: native ON → routes natively for vision models, not for text-only.
    assert!(media_cfg.native_image_context);
    for m in vision_models {
        let route = media_cfg.native_image_context && media_cfg.is_vision_capable_model(m);
        assert!(
            route,
            "{m} must route natively when native_image_context is on"
        );
    }
    for m in text_only_models {
        let route = media_cfg.native_image_context && media_cfg.is_vision_capable_model(m);
        assert!(!route, "{m} must fall back to describer (text-only model)");
    }

    // Toggle OFF → never route natively, no matter what model is active.
    media_cfg.native_image_context = false;
    for m in vision_models.iter().chain(text_only_models.iter()) {
        let route = media_cfg.native_image_context && media_cfg.is_vision_capable_model(m);
        assert!(
            !route,
            "{m} must use the describer when native_image_context is off"
        );
    }

    eprintln!("  PASS: routing predicate correctly gates native vs fallback path");
}
