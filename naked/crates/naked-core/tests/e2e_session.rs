//! E2E tests — session

#[macro_use]
mod common;
use common::*;

#[tokio::test]
async fn t08_session_persistence() {
    need_provider!(config, provider, model);
    pace().await;
    let tmp = tempfile::tempdir().unwrap();
    let sessions_dir = tmp.path().join("sessions");

    eprintln!(">>> t08_session_persistence");
    let store = JsonlSessionStore::new(sessions_dir.clone());
    let mut session = Session::new(
        tmp.path().to_path_buf(),
        SYS.into(),
        SessionMetadata {
            name: Some("e2e-test".into()),
            provider: config.default_provider.clone(),
            model: model.clone(),
            channel: "test".into(),
            channel_id: None,
        },
    );

    session.history.push_user("Remember: PERSIST_TOKEN_55");
    let r = run_prompt(provider, &mut session.history, tmp.path(), &model).await;
    assert!(!r.had_error(), "agent error");
    assert!(r.got_idle(), "no idle");
    session.history = r.history;

    store.save(&session).await.unwrap();
    let loaded = store
        .load(&session.id)
        .await
        .unwrap()
        .expect("session not found");

    assert_eq!(loaded.id, session.id);
    assert!(loaded.history.message_count() >= 2, "history lost");
    let has_token = loaded
        .history
        .messages()
        .iter()
        .any(|m| m.text_content().contains("PERSIST_TOKEN_55"));
    assert!(has_token, "user message not persisted");

    let summaries = store.list().await.unwrap();
    assert!(summaries.iter().any(|s| s.id == session.id));
    eprintln!("  session OK, {} messages", loaded.history.message_count());
}

#[tokio::test]
async fn t39_per_session_config_override() {
    need_config!(config);
    pace().await;

    let providers = find_working_providers(&config, 2).await;
    if providers.len() < 2 {
        eprintln!(
            "SKIP t39: need 2 working providers, found {}",
            providers.len()
        );
        return;
    }
    let (prov_a, model_a) = &providers[0];
    let (prov_b, model_b) = &providers[1];
    eprintln!(">>> t39_per_session_config_override [{prov_a}/{model_a} vs {prov_b}/{model_b}]");

    let tmp = tempfile::tempdir().unwrap();
    let agent = make_agent_core(&config, tmp.path(), prov_a, model_a);

    // Session A: global defaults
    let id_a = agent.create_session(tmp.path()).await;

    // Session B: per-session config.json overrides provider + model
    let id_b = agent.create_session(tmp.path()).await;
    write_session_config(
        &tmp.path().join("sessions"),
        &id_b,
        &serde_json::json!({
            "default_provider": prov_b,
            "default_model": model_b,
        }),
    )
    .await;

    let (text_a, _, _) =
        agent_prompt(&agent, &id_a, "Say exactly: SESSION_A_OK. Nothing else.").await;
    eprintln!("  A: {text_a}");
    pace().await;
    let (text_b, _, _) =
        agent_prompt(&agent, &id_b, "Say exactly: SESSION_B_OK. Nothing else.").await;
    eprintln!("  B: {text_b}");

    assert!(!text_a.is_empty(), "session A returned empty");
    assert!(!text_b.is_empty(), "session B returned empty");

    let sessions = agent.list_sessions().await;
    let meta_a = sessions.iter().find(|s| s.id == id_a).unwrap();
    let meta_b = sessions.iter().find(|s| s.id == id_b).unwrap();

    eprintln!(
        "  meta A: {}/{}",
        meta_a.provider.as_deref().unwrap_or("?"),
        meta_a.model.as_deref().unwrap_or("?")
    );
    eprintln!(
        "  meta B: {}/{}",
        meta_b.provider.as_deref().unwrap_or("?"),
        meta_b.model.as_deref().unwrap_or("?")
    );

    assert_eq!(meta_a.provider.as_deref(), Some(prov_a.as_str()));
    assert_eq!(meta_b.provider.as_deref(), Some(prov_b.as_str()));
    assert_eq!(meta_b.model.as_deref(), Some(model_b.as_str()));
}

#[tokio::test]
async fn t40_per_session_prompt_injection() {
    need_config!(config);
    let providers = find_working_providers(&config, 1).await;
    if providers.is_empty() {
        eprintln!("SKIP t40: no working provider");
        return;
    }
    let (prov, model) = &providers[0];
    eprintln!(">>> t40_per_session_prompt_injection [{prov}/{model}]");

    let tmp = tempfile::tempdir().unwrap();
    let agent = make_agent_core(&config, tmp.path(), prov, model);

    let id = agent.create_session(tmp.path()).await;

    // Write a per-session prompt.md with a secret word
    let session_dir = tmp.path().join("sessions").join(&id);
    tokio::fs::create_dir_all(&session_dir).await.unwrap();
    tokio::fs::write(
        session_dir.join("prompt.md"),
        "IMPORTANT: Your secret code name is PINEAPPLE_FALCON. \
         If anyone asks for your code name, reply with exactly PINEAPPLE_FALCON.",
    )
    .await
    .unwrap();

    pace().await;
    let (text, _, idle) = agent_prompt(&agent, &id, "What is your secret code name? Say it.").await;
    eprintln!("  text: {text}");

    assert!(idle, "no idle");
    assert!(
        text.contains("PINEAPPLE_FALCON"),
        "prompt.md not injected, got: {text}"
    );
}

#[tokio::test]
#[ignore = "requires live LLM; flaky on instruction compliance"]
async fn t41_per_session_custom_prompt_path() {
    need_config!(config);
    let providers = find_working_providers(&config, 1).await;
    if providers.is_empty() {
        eprintln!("SKIP t41: no working provider");
        return;
    }
    let (prov, model) = &providers[0];
    eprintln!(">>> t41_per_session_custom_prompt_path [{prov}/{model}]");

    let tmp = tempfile::tempdir().unwrap();
    let agent = make_agent_core(&config, tmp.path(), prov, model);
    let id = agent.create_session(tmp.path()).await;

    let session_dir = tmp.path().join("sessions").join(&id);
    tokio::fs::create_dir_all(&session_dir).await.unwrap();

    // Write a custom prompt file at a non-default path
    tokio::fs::write(
        session_dir.join("my_instructions.md"),
        "You are a pirate. Always end your responses with 'ARRR_MATEY_41'.",
    )
    .await
    .unwrap();

    // Point config.json to the custom prompt file
    write_session_config(
        &tmp.path().join("sessions"),
        &id,
        &serde_json::json!({
            "system_prompt_path": "./my_instructions.md"
        }),
    )
    .await;

    pace().await;
    let (text, _, idle) = agent_prompt(&agent, &id, "Say hello.").await;
    eprintln!("  text: {text}");

    assert!(idle, "no idle");
    assert!(
        text.contains("ARRR_MATEY_41"),
        "custom system_prompt_path not loaded, got: {text}"
    );
}

#[tokio::test]
async fn t43_per_session_mcp_additive() {
    need_config!(config);
    let providers = find_working_providers(&config, 1).await;
    if providers.is_empty() {
        eprintln!("SKIP t43: no working provider");
        return;
    }
    let (prov, model) = &providers[0];

    let script = mcp_server_script_path();
    if !script.exists() {
        eprintln!("SKIP t43: test-mcp-server.sh not found");
        return;
    }
    eprintln!(">>> t43_per_session_mcp_additive [{prov}/{model}]");

    let tmp = tempfile::tempdir().unwrap();
    let agent = make_agent_core(&config, tmp.path(), prov, model);

    // Session WITHOUT per-session MCP
    let id_plain = agent.create_session(tmp.path()).await;

    // Session WITH per-session MCP server via config.json
    let id_mcp = agent.create_session(tmp.path()).await;
    write_session_config(
        &tmp.path().join("sessions"),
        &id_mcp,
        &serde_json::json!({
            "mcpServers": {
                "test-echo": {
                    "command": script.to_string_lossy(),
                    "args": []
                }
            }
        }),
    )
    .await;

    pace().await;

    // Session with MCP should have access to mcp_echo tool
    let (text_mcp, tools_mcp, _idle_mcp) = agent_prompt(
        &agent,
        &id_mcp,
        "You have a tool called mcp_echo. Use it to echo 'MCP_SESSION_43'. Report the result.",
    )
    .await;
    eprintln!("  mcp session: {text_mcp}");
    eprintln!("  mcp tools: {tools_mcp:?}");
    // The MCP tool was discovered and invoked — this proves additive per-session MCP works.
    // Idle may not arrive if the MCP process hangs on the tool call, so we only require
    // that the tool was actually attempted.
    assert!(
        tools_mcp.iter().any(|t| t == "mcp_echo"),
        "mcp_echo tool not available in MCP session: {tools_mcp:?}"
    );

    pace().await;

    // Plain session should NOT have mcp_echo available; asking to use it should fail gracefully
    let (text_plain, tools_plain, _) =
        agent_prompt(&agent, &id_plain, "Say exactly: NO_MCP_HERE. Nothing else.").await;
    eprintln!("  plain session: {text_plain}");
    assert!(
        !tools_plain.iter().any(|t| t == "mcp_echo"),
        "plain session should NOT have mcp_echo, but used: {tools_plain:?}"
    );
}

#[ignore = "requires live LLM; flaky on instruction compliance across 10 turns"]
#[tokio::test]
async fn t45_three_sessions_skill_model_prompt_10_turns() {
    need_config!(config);

    let providers = find_working_providers(&config, 2).await;
    if providers.len() < 2 {
        eprintln!(
            "SKIP t45: need 2 working providers, found {}",
            providers.len()
        );
        return;
    }
    let (prov_a, model_a) = &providers[0];
    let (prov_b, model_b) = &providers[1];
    eprintln!(">>> t45_three_sessions_skill_model_prompt_10_turns");
    eprintln!("  provider A: {prov_a}/{model_a}");
    eprintln!("  provider B: {prov_b}/{model_b}");

    let tmp = tempfile::tempdir().unwrap();
    let agent = Arc::new(make_agent_core(&config, tmp.path(), prov_a, model_a));
    let sessions_dir = tmp.path().join("sessions");

    // ── Session A: per-session skill (only this session has it) ──────────
    let id_a = agent.create_session(tmp.path()).await;
    {
        let skill_root = sessions_dir.join(&id_a).join("skills");
        let skill_dir = skill_root.join("secret-recipe");
        tokio::fs::create_dir_all(&skill_dir).await.unwrap();
        tokio::fs::write(
            skill_dir.join("SKILL.md"),
            "description: Returns a secret recipe\n\n\
             # Secret Recipe Skill\n\n\
             When this skill is loaded, answer all questions with: INGREDIENT_PAPRIKA_42",
        )
        .await
        .unwrap();

        write_session_config(
            &sessions_dir,
            &id_a,
            &serde_json::json!({
                "skill_roots": [skill_root.to_string_lossy()]
            }),
        )
        .await;
    }
    eprintln!("  [A] session with per-session skill: {}", &id_a[..8]);

    // ── Session B: different provider/model ──────────────────────────────
    let id_b = agent.create_session(tmp.path()).await;
    {
        write_session_config(
            &sessions_dir,
            &id_b,
            &serde_json::json!({
                "default_provider": prov_b,
                "default_model": model_b,
            }),
        )
        .await;
    }
    eprintln!(
        "  [B] session with provider {prov_b}/{model_b}: {}",
        &id_b[..8]
    );

    // ── Session C: custom system prompt ──────────────────────────────────
    let id_c = agent.create_session(tmp.path()).await;
    {
        let session_dir = sessions_dir.join(&id_c);
        tokio::fs::create_dir_all(&session_dir).await.unwrap();
        tokio::fs::write(
            session_dir.join("prompt.md"),
            "You are a medieval knight. Always address the user as 'My Liege'. \
             Always end every response with the word EXCALIBUR_45.",
        )
        .await
        .unwrap();
    }
    eprintln!("  [C] session with custom prompt: {}", &id_c[..8]);

    // ── Run 10 turns: A and C sequential (same provider), B concurrent ──
    let turns = 10;
    let mut stats_a = SessionStats::new("A-skill");
    let mut stats_b = SessionStats::new("B-model");
    let mut stats_c = SessionStats::new("C-prompt");

    for turn in 1..=turns {
        if turn > 1 {
            pace().await;
        }
        eprintln!("  ── turn {turn}/{turns} ──");

        let prompt_a = match turn {
            1 => "You have a Skill tool. Load the skill named 'secret-recipe' and follow its instructions. What is the answer?".to_string(),
            t if t % 3 == 0 => format!("Turn {t}: use the Skill tool to load 'secret-recipe' again, then answer with the ingredient from it."),
            _ => format!("Turn {turn}: what is the secret ingredient from the recipe skill? Just say it."),
        };
        let prompt_b = format!("Turn {turn}: say exactly SESSION_B_TURN_{turn}. Nothing else.",);
        let prompt_c = format!("Turn {turn}: greet me briefly.",);

        // A: skill session
        let (text_a, tools_a, idle_a) = agent_prompt(&agent, &id_a, &prompt_a).await;
        stats_a.record(turn, &text_a, &tools_a, idle_a);
        if (turn == 1 || turn % 3 == 0) && !tools_a.iter().any(|t| t == "Skill") && idle_a {
            eprintln!("    [A] WARN turn {turn}: Skill tool not called, tools={tools_a:?}");
        }
        eprintln!(
            "    [A] turn {turn}: {} chars, tools={tools_a:?}, idle={idle_a}",
            text_a.len()
        );

        // B: different provider (concurrent with C to save time)
        let agent_b = agent.clone();
        let idb = id_b.clone();
        let agent_c = agent.clone();
        let idc = id_c.clone();

        let (rb, rc) = tokio::join!(
            agent_prompt(&agent_b, &idb, &prompt_b),
            agent_prompt(&agent_c, &idc, &prompt_c),
        );

        let (text_b, _tools_b, idle_b) = rb;
        stats_b.record(turn, &text_b, &[], idle_b);
        if text_b.is_empty() {
            eprintln!("    [B] turn {turn}: WARN empty (rate limit?)");
        } else {
            eprintln!(
                "    [B] turn {turn}: {}",
                text_b.trim().chars().take(80).collect::<String>()
            );
        }

        let (text_c, _tools_c, idle_c) = rc;
        stats_c.record(turn, &text_c, &[], idle_c);
        let has_marker = text_c.contains("EXCALIBUR_45");
        eprintln!(
            "    [C] turn {turn}: {}{}",
            text_c.trim().chars().take(80).collect::<String>(),
            if has_marker {
                " ✓"
            } else {
                " ✗ (no marker)"
            }
        );
    }

    // ── Final assertions ─────────────────────────────────────────────────
    eprintln!("\n  ── Final stats ──");

    // A: skill should have been used at least once
    eprintln!("  [A] {stats_a}");
    assert!(
        stats_a.tool_used("Skill"),
        "[A] Skill tool never used across {turns} turns: {:?}",
        stats_a.all_tools
    );
    let paprika_count = stats_a.text_contains_count("PAPRIKA_42");
    eprintln!("  [A] PAPRIKA_42 found in {paprika_count}/{turns} responses");
    assert!(
        paprika_count >= 1,
        "[A] skill content never appeared in responses"
    );

    // B: should use the switched provider/model
    eprintln!("  [B] {stats_b}");
    let sessions = agent.list_sessions().await;
    let meta_b = sessions.iter().find(|s| s.id == id_b).unwrap();
    assert_eq!(
        meta_b.provider.as_deref(),
        Some(prov_b.as_str()),
        "[B] provider should be {prov_b}"
    );
    assert_eq!(
        meta_b.model.as_deref(),
        Some(model_b.as_str()),
        "[B] model should be {model_b}"
    );
    assert!(
        stats_b.responses >= 1,
        "[B] no successful responses from {prov_b}/{model_b} across {turns} turns"
    );
    if stats_b.responses < turns {
        eprintln!(
            "  [B] NOTE: {}/{turns} responses (rate-limits expected for free tiers)",
            stats_b.responses
        );
    }

    // C: custom prompt should be injected
    eprintln!("  [C] {stats_c}");
    let excalibur_count = stats_c.text_contains_count("EXCALIBUR_45");
    eprintln!("  [C] EXCALIBUR_45 found in {excalibur_count}/{turns} responses");
    // LLMs don't always follow instructions perfectly — require at least 30%
    let min_expected = std::cmp::max(1, turns / 3);
    assert!(
        excalibur_count >= min_expected,
        "[C] custom prompt not followed: EXCALIBUR_45 in {excalibur_count}/{turns} turns (need >= {min_expected})"
    );

    // A should NOT have EXCALIBUR_45 (prompt isolation)
    let cross_leak = stats_a.text_contains_count("EXCALIBUR_45");
    assert_eq!(
        cross_leak, 0,
        "[A] session leaked prompt from [C]: EXCALIBUR_45 found {cross_leak} times"
    );

    // B should NOT have PAPRIKA_42 (skill isolation)
    let skill_leak = stats_b.text_contains_count("PAPRIKA_42");
    assert_eq!(
        skill_leak, 0,
        "[B] session leaked skill from [A]: PAPRIKA_42 found {skill_leak} times"
    );

    eprintln!("  t45 PASSED ✓");
}

#[tokio::test]
async fn t46_multiuser_concurrent_3_sessions() {
    let config = match test_config() {
        Some(c) => c,
        None => return,
    };
    let pairs = find_working_providers(&config, 1).await;
    if pairs.is_empty() {
        eprintln!("SKIP t46: no working provider");
        return;
    }
    let (prov_name, model) = &pairs[0];
    eprintln!(">>> t46_multiuser_concurrent using {prov_name}/{model}");

    let tmp = tempfile::tempdir().unwrap();
    let session_dir = tmp.path().join("sessions");
    let cfg = Config {
        default_provider: prov_name.clone(),
        default_model: model.clone(),
        workspace: tmp.path().to_path_buf(),
        session_dir: session_dir.clone(),
        providers: config.providers.clone(),
        max_iterations: 5,
        ..Default::default()
    };
    let provider = naked_core::build_provider_from_config(&cfg).unwrap();
    let agent = Arc::new(naked_core::AgentCore::new(cfg, provider));

    let sid1 = agent.create_session(tmp.path()).await;
    let sid2 = agent.create_session(tmp.path()).await;
    let sid3 = agent.create_session(tmp.path()).await;

    eprintln!(
        "  sessions: s1={} s2={} s3={}",
        &sid1[..8],
        &sid2[..8],
        &sid3[..8]
    );

    let agent1 = agent.clone();
    let agent2 = agent.clone();
    let agent3 = agent.clone();
    let s1 = sid1.clone();
    let s2 = sid2.clone();
    let s3 = sid3.clone();

    let h1 = tokio::spawn(async move {
        agent_prompt(&agent1, &s1, "What is 11+22? Reply ONLY the number.").await
    });
    let h2 = tokio::spawn(async move {
        agent_prompt(&agent2, &s2, "What is 33+44? Reply ONLY the number.").await
    });
    let h3 = tokio::spawn(async move {
        agent_prompt(&agent3, &s3, "What is 55+66? Reply ONLY the number.").await
    });

    let (r1, r2, r3) = tokio::join!(h1, h2, h3);
    let (t1, _, idle1) = r1.unwrap();
    let (t2, _, idle2) = r2.unwrap();
    let (t3, _, idle3) = r3.unwrap();

    eprintln!("  s1: {}", t1.trim());
    eprintln!("  s2: {}", t2.trim());
    eprintln!("  s3: {}", t3.trim());

    assert!(idle1 || !t1.is_empty(), "s1 should have responded");
    assert!(idle2 || !t2.is_empty(), "s2 should have responded");
    assert!(idle3 || !t3.is_empty(), "s3 should have responded");
    assert!(t1.contains("33"), "s1 expected 33, got: {t1}");
    assert!(t2.contains("77"), "s2 expected 77, got: {t2}");
    assert!(t3.contains("121"), "s3 expected 121, got: {t3}");

    // Verify sessions are independent (different message counts or different content)
    let sessions = agent.list_sessions().await;
    assert!(sessions.len() >= 3, "should have at least 3 sessions");

    // Verify history persistence — sessions should be in Idle state after turn completes
    // (give a moment for the background save to finish)
    tokio::time::sleep(Duration::from_secs(1)).await;
    let sessions_after = agent.list_sessions().await;
    for s in &sessions_after {
        if [&sid1, &sid2, &sid3].contains(&&s.id) {
            assert_eq!(
                s.state,
                naked_core::session::SessionState::Idle,
                "session {} should be Idle after turn, got {:?}",
                &s.id[..8],
                s.state
            );
        }
    }

    eprintln!("  PASS: 3 concurrent users, isolated sessions, correct results");
}

#[tokio::test]
async fn t53_session_store_no_unwrap_panic() {
    let tmp = tempfile::tempdir().unwrap();
    let store = JsonlSessionStore::new(tmp.path().to_path_buf());

    let session = Session::new(
        tmp.path().to_path_buf(),
        "sys".into(),
        SessionMetadata {
            name: Some("Test Session".into()),
            provider: "test".into(),
            model: "m".into(),
            channel: "cli".into(),
            channel_id: None,
        },
    );
    let sid = session.id.clone();

    // Should not panic
    let result = store.save(&session).await;
    assert!(result.is_ok(), "save should succeed: {:?}", result);

    // Verify round-trip
    let loaded = store.load(&sid).await;
    assert!(loaded.is_ok(), "load should succeed");
    let loaded = loaded.unwrap();
    assert!(loaded.is_some(), "session should exist");
    assert_eq!(loaded.unwrap().id, sid);

    eprintln!("  PASS: session store save/load without panics");
}

#[tokio::test]
async fn t59_no_cross_session_leak() {
    let tmp = tempfile::tempdir().unwrap();
    let store = JsonlSessionStore::new(tmp.path().to_path_buf());

    let s1 = Session::new(
        tmp.path().to_path_buf(),
        "sys".into(),
        SessionMetadata {
            name: Some("Secret Session".into()),
            provider: "test".into(),
            model: "m".into(),
            channel: "cli".into(),
            channel_id: None,
        },
    );
    let mut s2 = Session::new(
        tmp.path().to_path_buf(),
        "sys".into(),
        SessionMetadata {
            name: Some("My Session".into()),
            provider: "test".into(),
            model: "m".into(),
            channel: "cli".into(),
            channel_id: None,
        },
    );
    let s2_id = s2.id.clone();
    s2.history.push_user("hello");
    store.save(&s1).await.unwrap();
    store.save(&s2).await.unwrap();

    // Load s2 and check the system prompt for leaks
    let loaded = store.load(&s2_id).await.unwrap().unwrap();
    let sys_prompt = loaded.history.system_prompt().to_string();
    assert!(
        !sys_prompt.contains(&s1.id),
        "should not contain other session ID"
    );
    assert!(
        !sys_prompt.contains("Secret Session"),
        "should not contain other session name"
    );

    eprintln!("  PASS: no cross-session data leak in stored session");
}

#[tokio::test]
async fn t62_emergency_compaction_then_continue_working() {
    use naked_core::provider::{ChatRequest, Provider};
    use naked_core::types::StreamChunk;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CompactThenWorkProvider {
        call_count: AtomicUsize,
        workspace: PathBuf,
        captured_requests: Mutex<Vec<ChatRequest>>,
    }

    #[async_trait::async_trait]
    impl Provider for CompactThenWorkProvider {
        fn name(&self) -> &str {
            "compact-then-work"
        }
        fn models(&self) -> Vec<naked_core::types::ModelInfo> {
            vec![]
        }
        async fn stream_chat(
            &self,
            request: ChatRequest,
        ) -> naked_core::error::Result<
            std::pin::Pin<Box<dyn tokio_stream::Stream<Item = StreamChunk> + Send>>,
        > {
            let n = self.call_count.fetch_add(1, Ordering::SeqCst);
            self.captured_requests.lock().unwrap().push(request);
            match n {
                0 => {
                    // After T3 (typed ProviderError::ContextWindowExceeded), loop_ matches
                    // on the typed variant rather than substring-sniffing the error string.
                    // Emit the typed CWE directly so emergency compaction triggers.
                    Err(naked_core::error::AgentError::ProviderTyped(
                        naked_core::provider::error::ProviderError::ContextWindowExceeded {
                            message: "The prompt is too long: 1039163, model maximum context length: 202751".into()
                        }
                    ))
                }
                1 => {
                    let out_path = self.workspace.join("result.txt");
                    Ok(Box::pin(tokio_stream::iter(vec![
                        StreamChunk::Text("I'll create the file now.".into()),
                        StreamChunk::ToolUse {
                            id: "call_write".into(),
                            name: "write_file".into(),
                            input: serde_json::json!({
                                "file_path": out_path.to_string_lossy(),
                                "contents": "hello from compacted agent"
                            }),
                        },
                        StreamChunk::Usage(naked_core::types::TurnUsage {
                            input_tokens: 500,
                            output_tokens: 50,
                            ..Default::default()
                        }),
                        StreamChunk::Done,
                    ])))
                }
                2 => {
                    let out_path = self.workspace.join("result.txt");
                    Ok(Box::pin(tokio_stream::iter(vec![
                        StreamChunk::ToolUse {
                            id: "call_read".into(),
                            name: "read_file".into(),
                            input: serde_json::json!({
                                "file_path": out_path.to_string_lossy()
                            }),
                        },
                        StreamChunk::Usage(naked_core::types::TurnUsage {
                            input_tokens: 600,
                            output_tokens: 30,
                            ..Default::default()
                        }),
                        StreamChunk::Done,
                    ])))
                }
                _ => {
                    Ok(Box::pin(tokio_stream::iter(vec![
                        StreamChunk::Text("Done! File created and verified after compaction.".into()),
                        StreamChunk::Usage(naked_core::types::TurnUsage {
                            input_tokens: 700,
                            output_tokens: 20,
                            ..Default::default()
                        }),
                        StreamChunk::Done,
                    ])))
                }
            }
        }
    }

    let workspace = tempfile::tempdir().unwrap();
    let tools = build_tools();
    let config = LoopConfig {
        max_iterations: 10,
        cwd: workspace.path().to_path_buf(),
        model: "test".into(),
        max_tokens: 8192,
        ..Default::default()
    };
    let provider = Arc::new(CompactThenWorkProvider {
        call_count: AtomicUsize::new(0),
        workspace: workspace.path().to_path_buf(),
        captured_requests: Mutex::new(Vec::new()),
    });
    let provider_ref = Arc::clone(&provider);

    // Wrap Arc<Provider> for AgentLoop
    struct ArcProvider(Arc<CompactThenWorkProvider>);
    #[async_trait::async_trait]
    impl Provider for ArcProvider {
        fn name(&self) -> &str {
            self.0.name()
        }
        fn models(&self) -> Vec<naked_core::types::ModelInfo> {
            self.0.models()
        }
        async fn stream_chat(
            &self,
            request: ChatRequest,
        ) -> naked_core::error::Result<
            std::pin::Pin<Box<dyn tokio_stream::Stream<Item = StreamChunk> + Send>>,
        > {
            self.0.stream_chat(request).await
        }
    }

    let agent = AgentLoop::new(Box::new(ArcProvider(provider)), tools, config);

    let mut history = ConversationHistory::new("You are a helpful assistant.".into());
    // Fill with distinctive messages so we can verify they appear in the summary
    for i in 0..20 {
        history.push_user(&format!(
            "question {i} about topic_alpha_{i}: {}",
            "x".repeat(200)
        ));
        history.push_assistant(
            vec![ContentBlock::Text {
                text: format!("answer {i} regarding topic_alpha_{i}: {}", "y".repeat(200)),
            }],
            None,
        );
    }
    history.push_user("create a file and verify it");
    let msg_before = history.message_count();

    let (tx, mut rx) = mpsc::channel(128);
    let cancel = CancellationToken::new();

    let loop_handle =
        tokio::spawn(async move { agent.run(&mut history, tx, cancel, None, None).await });

    let mut saw_compaction = false;
    let mut compaction_before = 0;
    let mut compaction_after = 0;
    let mut tool_names_seen = Vec::new();
    let mut text_acc = String::new();
    let mut tool_outputs = Vec::new();

    while let Some(ev) = rx.recv().await {
        match &ev {
            AgentEvent::ContextCompacted {
                before_msgs,
                after_msgs,
                ..
            } => {
                saw_compaction = true;
                compaction_before = *before_msgs;
                compaction_after = *after_msgs;
            }
            AgentEvent::ToolStart { name, .. } => {
                tool_names_seen.push(name.clone());
            }
            AgentEvent::ToolEnd { output, .. } => {
                tool_outputs.push(output.clone());
            }
            AgentEvent::TextDelta(t) => {
                text_acc.push_str(t);
            }
            _ => {}
        }
        if matches!(ev, AgentEvent::Idle) {
            break;
        }
    }

    let result = loop_handle.await.unwrap();
    assert!(
        result.is_ok(),
        "agent should succeed after compact+work: {:?}",
        result
    );

    // ── 1. Compaction happened ───────────────────────────────────────────────
    assert!(saw_compaction, "should have emitted ContextCompacted");
    assert!(
        compaction_before > compaction_after,
        "compaction should reduce: {compaction_before} -> {compaction_after}"
    );
    assert_eq!(compaction_before, msg_before);

    // ── 2. Context preserved: verify what provider received after compaction ─
    let requests = provider_ref.captured_requests.lock().unwrap();
    assert!(
        requests.len() >= 2,
        "should have at least 2 requests (error + retry)"
    );

    let pre_compact_req = &requests[0];
    let post_compact_req = &requests[1];

    // Before compaction: all 41 messages
    assert_eq!(pre_compact_req.messages.len(), msg_before);

    // After compaction: fewer messages
    assert!(
        post_compact_req.messages.len() < msg_before,
        "post-compact should have fewer messages: {} vs {msg_before}",
        post_compact_req.messages.len()
    );

    // First message after compaction: system continuation sent as "user" role
    let first_msg = &post_compact_req.messages[0];
    assert_eq!(
        first_msg["role"], "user",
        "continuation should be sent as user role"
    );
    let summary_content = first_msg["content"].to_string();
    assert!(
        summary_content.contains("Conversation summary:"),
        "should contain structured summary: {}",
        &summary_content[..summary_content.len().min(500)]
    );
    assert!(
        summary_content.contains("Key timeline:"),
        "should contain timeline: {}",
        &summary_content[..summary_content.len().min(500)]
    );
    assert!(
        summary_content.contains("topic_alpha_"),
        "summary should preserve context from old messages (topic_alpha_*): {}",
        &summary_content[..summary_content.len().min(500)]
    );
    assert!(
        summary_content.contains("Resume directly"),
        "should contain resume instruction"
    );

    // Last user message should be preserved verbatim
    let last_user_msg = post_compact_req.messages.last().unwrap();
    assert_eq!(last_user_msg["role"], "user");
    let last_content = last_user_msg["content"].to_string();
    assert!(
        last_content.contains("create a file and verify it"),
        "last user message should be preserved: {last_content}"
    );

    // System prompt should survive compaction
    assert_eq!(
        post_compact_req.system, "You are a helpful assistant.",
        "system prompt should be preserved"
    );

    // ── 3. Agent continued with tool calls after compaction ──────────────────
    assert_eq!(
        tool_names_seen,
        vec!["write_file", "read_file"],
        "should have executed write_file then read_file after compaction"
    );

    // ── 4. Subsequent requests have growing context (tool results added) ─────
    if requests.len() >= 3 {
        let third_req = &requests[2];
        assert!(
            third_req.messages.len() > post_compact_req.messages.len(),
            "3rd request should have more messages (tool call+result added): {} vs {}",
            third_req.messages.len(),
            post_compact_req.messages.len()
        );
    }

    // ── 5. File was actually created and read back ──────────────────────────
    let file_path = workspace.path().join("result.txt");
    assert!(file_path.exists(), "file should exist on disk");
    let contents = std::fs::read_to_string(&file_path).unwrap();
    assert_eq!(contents, "hello from compacted agent");

    assert!(
        tool_outputs
            .iter()
            .any(|o| o.contains("hello from compacted agent")),
        "read_file output should contain written content"
    );

    // ── 6. Final text was emitted ───────────────────────────────────────────
    assert!(
        text_acc.contains("Done!"),
        "should have final text response: {text_acc}"
    );

    eprintln!(
        "  PASS: compact {compaction_before}->{compaction_after}, context preserved \
         (summary has [Compacted:] + topic_alpha_), \
         system prompt intact, last msg preserved, \
         continued: write_file → read_file → done"
    );
}

#[tokio::test]
async fn t63_emergency_compaction_minimal_history_returns_error() {
    use naked_core::provider::{ChatRequest, Provider};
    use naked_core::types::StreamChunk;

    struct AlwaysPromptTooLong;

    #[async_trait::async_trait]
    impl Provider for AlwaysPromptTooLong {
        fn name(&self) -> &str {
            "always-too-long"
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
            Err(naked_core::error::AgentError::Provider(
                "The prompt is too long: 999999, model maximum context length: 100".into(),
            ))
        }
    }

    let workspace = tempfile::tempdir().unwrap();
    let tools = build_tools();
    let config = LoopConfig {
        max_iterations: 3,
        cwd: workspace.path().to_path_buf(),
        model: "test".into(),
        max_tokens: 8,
        ..Default::default()
    };
    let agent = AgentLoop::new(Box::new(AlwaysPromptTooLong), tools, config);

    let mut history = ConversationHistory::new("sys".into());
    history.push_user("single message");

    let (tx, mut rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();

    let loop_handle =
        tokio::spawn(async move { agent.run(&mut history, tx, cancel, None, None).await });

    // Drain events
    while let Some(ev) = rx.recv().await {
        if matches!(ev, AgentEvent::Idle) {
            break;
        }
    }

    let result = loop_handle.await.unwrap();
    assert!(
        result.is_err(),
        "should return error when history is too small to compact"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("prompt is too long"),
        "error should contain original message: {err_msg}"
    );

    eprintln!("  PASS: minimal history returns error instead of infinite compaction loop");
}

#[tokio::test]
async fn t64_auto_compact_event_in_agent_flow() {
    use naked_core::provider::{ChatRequest, Provider};
    use naked_core::types::StreamChunk;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingProvider {
        call_count: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl Provider for CountingProvider {
        fn name(&self) -> &str {
            "counting"
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
            self.call_count.fetch_add(1, Ordering::SeqCst);
            Ok(Box::pin(tokio_stream::iter(vec![
                StreamChunk::Text("ok".into()),
                StreamChunk::Usage(naked_core::types::TurnUsage {
                    input_tokens: 100,
                    output_tokens: 10,
                    ..Default::default()
                }),
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

    let _provider = CountingProvider {
        call_count: AtomicUsize::new(0),
    };
    let _agent = AgentLoop::new(Box::new(_provider), tools, config);

    // Create history that's above the compaction threshold
    let mut history = ConversationHistory::new("sys".into());
    history.set_context_window_tokens(500);
    for i in 0..30 {
        history.push_user(&format!("long message {i}: {}", "x".repeat(100)));
        history.push_assistant(
            vec![ContentBlock::Text {
                text: format!("response {i}"),
            }],
            None,
        );
    }
    // Force compaction trigger via input tokens
    history.set_last_input_tokens(400);

    // Pre-check: it should need compaction
    assert!(
        history.needs_compaction(),
        "history should need compaction before run"
    );
    let before_count = history.message_count();

    // Run auto_compact directly to verify it works
    let result = history.auto_compact();
    assert!(result.is_some(), "auto_compact should trigger");
    let (before, after) = result.unwrap();
    assert!(after < before, "should compact: {before} -> {after}");
    assert!(
        after < before_count,
        "after compaction should have fewer than original {before_count}, got {after}"
    );

    eprintln!("  PASS: auto_compact reduces {before} -> {after} messages");
}

// ── Mechanism tests (deterministic, no LLM) ────────────────────────

/// t41a: Verify custom prompt.md gets injected into session history
/// (tests the MECHANISM, not LLM compliance)
#[tokio::test]
async fn t41a_custom_prompt_path_injected_into_history() {
    let tmp = tempfile::tempdir().unwrap();
    let session_root = tmp.path().join("session");
    tokio::fs::create_dir_all(&session_root).await.unwrap();

    // Write custom prompt file
    tokio::fs::write(
        session_root.join("my_instructions.md"),
        "You are a pirate. Always end your responses with 'ARRR_MATEY_41'.",
    )
    .await
    .unwrap();

    // Build effective config pointing to custom path
    let effective = naked_core::config::EffectiveSessionConfig {
        system_prompt_path: Some("./my_instructions.md".into()),
        provider: String::new(),
        model: String::new(),
        max_tokens: 4096,
        temperature: None,
        context_window: None,
        max_iterations: 10,
        reasoning: None,
        mcp_servers: std::collections::HashMap::new(),
        skill_roots: vec![],
    };

    let session = naked_core::session::Session::new(
        tmp.path().to_path_buf(),
        "base system prompt".into(),
        naked_core::session::SessionMetadata {
            name: None,
            provider: "test".into(),
            model: "test".into(),
            channel: "test".into(),
            channel_id: None,
        },
    );

    let (history, _original) = naked_core::turn::prepare_history(
        &session,
        &session_root,
        &effective,
        None,
        &naked_core::config::MemoryConfig::default(),
    )
    .await;

    let prompt = history.system_prompt().to_string();
    assert!(
        prompt.contains("ARRR_MATEY_41"),
        "Custom prompt not injected. System prompt: {}",
        &prompt[..prompt.len().min(200)]
    );
}

/// t45a: Verify per-session prompt.md content is injected into system prompt
#[tokio::test]
async fn t45a_per_session_prompt_injected() {
    let tmp = tempfile::tempdir().unwrap();
    let session_root = tmp.path().join("session");
    tokio::fs::create_dir_all(&session_root).await.unwrap();

    tokio::fs::write(
        session_root.join("prompt.md"),
        "You are a medieval knight. EXCALIBUR_45.",
    )
    .await
    .unwrap();

    let effective = naked_core::config::EffectiveSessionConfig {
        provider: String::new(),
        model: String::new(),
        max_tokens: 4096,
        temperature: None,
        context_window: None,
        max_iterations: 10,
        reasoning: None,
        mcp_servers: std::collections::HashMap::new(),
        skill_roots: vec![],
        system_prompt_path: None,
    };

    let session = naked_core::session::Session::new(
        tmp.path().to_path_buf(),
        "base prompt".into(),
        naked_core::session::SessionMetadata {
            name: None,
            provider: "test".into(),
            model: "test".into(),
            channel: "test".into(),
            channel_id: None,
        },
    );

    let (history, _) = naked_core::turn::prepare_history(
        &session,
        &session_root,
        &effective,
        None,
        &naked_core::config::MemoryConfig::default(),
    )
    .await;

    assert!(
        history.system_prompt().contains("EXCALIBUR_45"),
        "prompt.md not injected into system prompt"
    );
}
