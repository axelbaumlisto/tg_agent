//! E2E tests — provider rotation, probes, multi-key

#[macro_use]
mod common;
use common::*;

#[tokio::test]
async fn t13_provider_fireworks() {
    need_config!(config);
    pace().await;
    let tmp = tempfile::tempdir().unwrap();
    let model = "accounts/fireworks/models/glm-5p1";

    let provider = match make_provider_for(&config, "fireworks", model) {
        Some(p) => p,
        None => {
            eprintln!("SKIP t13: fireworks not configured or key missing");
            return;
        }
    };

    eprintln!(">>> t13_provider_fireworks [model={model}]");
    let mut history = ConversationHistory::new(SYS.into());
    history.push_user("Say exactly: FIREWORKS_OK_13. Nothing else.");
    let r = run_prompt(provider, &mut history, tmp.path(), model).await;

    eprintln!("  text: {}", r.full_text());
    assert!(!r.had_error(), "agent error");
    assert!(r.got_idle(), "no idle");
    assert!(
        r.full_text().contains("FIREWORKS_OK_13"),
        "missing phrase: {}",
        r.full_text()
    );
}

#[tokio::test]
async fn t14_provider_minimax() {
    need_config!(config);
    pace().await;
    let tmp = tempfile::tempdir().unwrap();
    let model = "MiniMax-M2.7-highspeed";

    let provider = match make_provider_for(&config, "minimax", model) {
        Some(p) => p,
        None => {
            eprintln!("SKIP t14: minimax not configured or key missing");
            return;
        }
    };

    if !probe_provider(provider.as_ref(), model).await {
        eprintln!("SKIP t14: minimax probe failed (invalid key or model)");
        return;
    }
    pace().await;

    let provider = make_provider_for(&config, "minimax", model).unwrap();
    eprintln!(">>> t14_provider_minimax [model={model}]");
    let mut history = ConversationHistory::new(SYS.into());
    history.push_user("Say exactly: MINIMAX_OK_14. Nothing else.");
    let r = run_prompt(provider, &mut history, tmp.path(), model).await;

    eprintln!("  text: {}", r.full_text());
    assert!(!r.had_error(), "agent error");
    assert!(r.got_idle(), "no idle");
    assert!(
        r.full_text().contains("MINIMAX_OK_14"),
        "missing phrase: {}",
        r.full_text()
    );
}

#[tokio::test]
async fn t15_provider_openai() {
    need_config!(config);
    pace().await;
    let tmp = tempfile::tempdir().unwrap();
    let model = "gpt-4o-mini";

    let provider = match make_provider_for(&config, "openai", model) {
        Some(p) => p,
        None => {
            eprintln!("SKIP t15: openai not configured or key missing");
            return;
        }
    };

    if !probe_provider(provider.as_ref(), model).await {
        eprintln!("SKIP t15: openai probe failed (invalid key or model)");
        return;
    }
    pace().await;

    let provider = make_provider_for(&config, "openai", model).unwrap();
    eprintln!(">>> t15_provider_openai [model={model}]");
    let mut history = ConversationHistory::new(SYS.into());
    history.push_user("Say exactly: OPENAI_OK_15. Nothing else.");
    let r = run_prompt(provider, &mut history, tmp.path(), model).await;

    eprintln!("  text: {}", r.full_text());
    assert!(!r.had_error(), "agent error");
    assert!(r.got_idle(), "no idle");
    assert!(
        r.full_text().contains("OPENAI_OK_15"),
        "missing phrase: {}",
        r.full_text()
    );
}

#[tokio::test]
async fn t16_provider_rotation() {
    need_config!(config);
    pace().await;
    let tmp = tempfile::tempdir().unwrap();

    let candidates: Vec<(&str, &str)> = vec![
        ("fireworks", "accounts/fireworks/models/glm-5p1"),
        ("minimax", "MiniMax-M2.7-highspeed"),
        ("openai", "gpt-4o-mini"),
    ];

    let mut providers: Vec<Box<dyn Provider>> = Vec::new();
    let mut model_for_test = String::new();

    for (name, model) in &candidates {
        if let Some(p) = make_provider_for(&config, name, model) {
            if probe_provider(p.as_ref(), model).await {
                if model_for_test.is_empty() {
                    model_for_test = model.to_string();
                }
                providers.push(make_provider_for(&config, name, model).unwrap());
                eprintln!("  rotation: {name} OK");
            } else {
                eprintln!("  rotation: {name} probe FAILED, skipping");
            }
            pace().await;
        }
    }

    if providers.len() < 2 {
        eprintln!(
            "SKIP t16: need at least 2 reachable providers, got {}",
            providers.len()
        );
        return;
    }

    let count = providers.len();
    let resilient: Box<dyn Provider> = Box::new(ResilientProvider::new(providers));

    eprintln!(">>> t16_provider_rotation [{count} providers]");
    let mut history = ConversationHistory::new(SYS.into());
    history.push_user("Say exactly: ROTATION_OK_16. Nothing else.");
    let r = run_prompt(resilient, &mut history, tmp.path(), &model_for_test).await;

    eprintln!("  text: {}", r.full_text());
    assert!(!r.had_error(), "agent error");
    assert!(r.got_idle(), "no idle");
    assert!(
        r.full_text().contains("ROTATION_OK_16"),
        "missing phrase: {}",
        r.full_text()
    );
}

#[tokio::test]
async fn t23_provider_groq() {
    need_config!(config);
    pace().await;
    let tmp = tempfile::tempdir().unwrap();
    let model = "llama-3.3-70b-versatile";

    let provider = match make_provider_for(&config, "groq", model) {
        Some(p) => p,
        None => {
            eprintln!("SKIP t23: groq not configured or key missing");
            return;
        }
    };

    if !probe_provider(provider.as_ref(), model).await {
        eprintln!("SKIP t23: groq probe failed");
        return;
    }
    pace().await;

    let provider = make_provider_for(&config, "groq", model).unwrap();
    eprintln!(">>> t23_provider_groq [model={model}]");
    let mut history = ConversationHistory::new(SYS.into());
    history.push_user("Say exactly: GROQ_OK_23. Nothing else.");
    let r = run_prompt(provider, &mut history, tmp.path(), model).await;

    eprintln!("  text: {}", r.full_text());
    assert!(!r.had_error(), "agent error");
    assert!(r.got_idle(), "no idle");
    assert!(
        r.full_text().contains("GROQ_OK_23"),
        "missing phrase: {}",
        r.full_text()
    );
}

#[tokio::test]
async fn t24_provider_moonshot() {
    need_config!(config);
    pace().await;
    let tmp = tempfile::tempdir().unwrap();
    let model = "moonshot-v1-8k";

    let provider = match make_provider_for(&config, "moonshot", model) {
        Some(p) => p,
        None => {
            eprintln!("SKIP t24: moonshot not configured or key missing");
            return;
        }
    };

    if !probe_provider(provider.as_ref(), model).await {
        eprintln!("SKIP t24: moonshot probe failed");
        return;
    }
    pace().await;

    let provider = make_provider_for(&config, "moonshot", model).unwrap();
    eprintln!(">>> t24_provider_moonshot [model={model}]");
    let mut history = ConversationHistory::new(SYS.into());
    history.push_user("Say exactly: MOONSHOT_OK_24. Nothing else.");
    let r = run_prompt(provider, &mut history, tmp.path(), model).await;

    eprintln!("  text: {}", r.full_text());
    assert!(!r.had_error(), "agent error");
    assert!(r.got_idle(), "no idle");
    assert!(
        r.full_text().contains("MOONSHOT_OK_24"),
        "missing phrase: {}",
        r.full_text()
    );
}

#[tokio::test]
async fn t25_provider_kimi() {
    need_config!(config);
    pace().await;
    let tmp = tempfile::tempdir().unwrap();
    let model = "moonshot-v1-128k";

    let provider = match make_provider_for(&config, "kimi", model) {
        Some(p) => p,
        None => {
            eprintln!("SKIP t25: kimi not configured or key missing");
            return;
        }
    };

    if !probe_provider(provider.as_ref(), model).await {
        eprintln!("SKIP t25: kimi probe failed");
        return;
    }
    pace().await;

    let provider = make_provider_for(&config, "kimi", model).unwrap();
    eprintln!(">>> t25_provider_kimi [model={model}]");
    let mut history = ConversationHistory::new(SYS.into());
    history.push_user("Say exactly: KIMI_OK_25. Nothing else.");
    let r = run_prompt(provider, &mut history, tmp.path(), model).await;

    eprintln!("  text: {}", r.full_text());
    assert!(!r.had_error(), "agent error");
    assert!(r.got_idle(), "no idle");
    assert!(
        r.full_text().contains("KIMI_OK_25"),
        "missing phrase: {}",
        r.full_text()
    );
}

#[tokio::test]
async fn t26_provider_rotation_full() {
    need_config!(config);
    pace().await;
    let tmp = tempfile::tempdir().unwrap();

    let candidates: Vec<(&str, &str)> = vec![
        ("fireworks", "accounts/fireworks/models/glm-5p1"),
        ("minimax", "MiniMax-M2.7-highspeed"),
        ("groq", "llama-3.3-70b-versatile"),
        ("moonshot", "moonshot-v1-8k"),
        ("kimi", "moonshot-v1-128k"),
        ("openai", "gpt-4o-mini"),
    ];

    let mut providers: Vec<Box<dyn Provider>> = Vec::new();
    let mut model_for_test = String::new();

    for (name, model) in &candidates {
        if let Some(p) = make_provider_for(&config, name, model) {
            if probe_provider(p.as_ref(), model).await {
                if model_for_test.is_empty() {
                    model_for_test = model.to_string();
                }
                providers.push(make_provider_for(&config, name, model).unwrap());
                eprintln!("  rotation-full: {name} OK");
            } else {
                eprintln!("  rotation-full: {name} probe FAILED, skipping");
            }
            pace().await;
        }
    }

    if providers.len() < 3 {
        eprintln!(
            "SKIP t26: need at least 3 reachable providers, got {}",
            providers.len()
        );
        return;
    }

    let count = providers.len();
    let resilient: Box<dyn Provider> = Box::new(ResilientProvider::new(providers));

    eprintln!(">>> t26_provider_rotation_full [{count} providers]");
    let mut history = ConversationHistory::new(SYS.into());
    history.push_user("Say exactly: ROTATION_FULL_26. Nothing else.");
    let r = run_prompt(resilient, &mut history, tmp.path(), &model_for_test).await;

    eprintln!("  text: {}", r.full_text());
    assert!(!r.had_error(), "agent error");
    assert!(r.got_idle(), "no idle");
    assert!(
        r.full_text().contains("ROTATION_FULL_26"),
        "missing phrase: {}",
        r.full_text()
    );
}

#[tokio::test]
async fn t27_provider_minimax_cp() {
    need_config!(config);
    pace().await;
    let tmp = tempfile::tempdir().unwrap();
    let model = "MiniMax-M2.7-highspeed";

    let provider = match make_provider_for(&config, "minimax-cp", model) {
        Some(p) => p,
        None => {
            eprintln!("SKIP t27: minimax-cp not configured or key missing");
            return;
        }
    };

    if !probe_provider(provider.as_ref(), model).await {
        eprintln!("SKIP t27: minimax-cp probe failed");
        return;
    }
    pace().await;

    let provider = make_provider_for(&config, "minimax-cp", model).unwrap();
    eprintln!(">>> t27_provider_minimax_cp [model={model}]");
    let mut history = ConversationHistory::new(SYS.into());
    history.push_user("Say exactly: MINIMAX_CP_OK_27. Nothing else.");
    let r = run_prompt(provider, &mut history, tmp.path(), model).await;

    eprintln!("  text: {}", r.full_text());
    assert!(!r.had_error(), "agent error");
    assert!(r.got_idle(), "no idle");
    assert!(
        r.full_text().contains("MINIMAX_CP_OK_27"),
        "missing phrase: {}",
        r.full_text()
    );
}

#[tokio::test]
async fn t28_provider_glm_cn() {
    need_config!(config);
    pace().await;
    let tmp = tempfile::tempdir().unwrap();
    let model = "glm-5.1";

    let provider = match make_provider_for(&config, "glm-cn", model) {
        Some(p) => p,
        None => {
            eprintln!("SKIP t28: glm-cn not configured or key missing");
            return;
        }
    };

    if !probe_provider(provider.as_ref(), model).await {
        eprintln!("SKIP t28: glm-cn probe failed");
        return;
    }
    pace().await;

    let provider = make_provider_for(&config, "glm-cn", model).unwrap();
    eprintln!(">>> t28_provider_glm_cn [model={model}]");
    let mut history = ConversationHistory::new(SYS.into());
    history.push_user("Say exactly: GLM_CN_OK_28. Nothing else.");
    let r = run_prompt(provider, &mut history, tmp.path(), model).await;

    eprintln!("  text: {}", r.full_text());
    assert!(!r.had_error(), "agent error");
    assert!(r.got_idle(), "no idle");
    assert!(
        r.full_text().contains("GLM_CN_OK_28"),
        "missing phrase: {}",
        r.full_text()
    );
}

#[tokio::test]
async fn t29_provider_zai() {
    need_config!(config);
    pace().await;
    let tmp = tempfile::tempdir().unwrap();
    let model = "glm-5";

    let provider = match make_provider_for(&config, "zai", model) {
        Some(p) => p,
        None => {
            eprintln!("SKIP t29: zai not configured or key missing");
            return;
        }
    };

    if !probe_provider(provider.as_ref(), model).await {
        eprintln!("SKIP t29: zai probe failed");
        return;
    }
    pace().await;

    let provider = make_provider_for(&config, "zai", model).unwrap();
    eprintln!(">>> t29_provider_zai [model={model}]");
    let mut history = ConversationHistory::new(SYS.into());
    history.push_user("Say exactly: ZAI_OK_29. Nothing else.");
    let r = run_prompt(provider, &mut history, tmp.path(), model).await;

    eprintln!("  text: {}", r.full_text());
    assert!(!r.had_error(), "agent error");
    assert!(r.got_idle(), "no idle");
    assert!(
        r.full_text().contains("ZAI_OK_29"),
        "missing phrase: {}",
        r.full_text()
    );
}

#[tokio::test]
async fn t30_provider_xiaomi() {
    need_config!(config);
    pace().await;
    let tmp = tempfile::tempdir().unwrap();
    let model = "mimo-v2-flash";

    let provider = match make_provider_for(&config, "xiaomi", model) {
        Some(p) => p,
        None => {
            eprintln!("SKIP t30: xiaomi not configured or key missing");
            return;
        }
    };

    if !probe_provider(provider.as_ref(), model).await {
        eprintln!("SKIP t30: xiaomi probe failed");
        return;
    }
    pace().await;

    let provider = make_provider_for(&config, "xiaomi", model).unwrap();
    eprintln!(">>> t30_provider_xiaomi [model={model}]");
    let mut history = ConversationHistory::new(SYS.into());
    history.push_user("Say exactly: XIAOMI_OK_30. Nothing else.");
    let r = run_prompt(provider, &mut history, tmp.path(), model).await;

    eprintln!("  text: {}", r.full_text());
    assert!(!r.had_error(), "agent error");
    assert!(r.got_idle(), "no idle");
    assert!(
        r.full_text().contains("XIAOMI_OK_30"),
        "missing phrase: {}",
        r.full_text()
    );
}

#[tokio::test]
async fn t31_provider_openrouter() {
    need_config!(config);
    pace().await;
    let tmp = tempfile::tempdir().unwrap();
    let model = "meta-llama/llama-3.3-70b-instruct";

    let provider = match make_provider_for(&config, "openrouter", model) {
        Some(p) => p,
        None => {
            eprintln!("SKIP t31: openrouter not configured or key missing");
            return;
        }
    };

    if !probe_provider(provider.as_ref(), model).await {
        eprintln!("SKIP t31: openrouter probe failed");
        return;
    }
    pace().await;

    let provider = make_provider_for(&config, "openrouter", model).unwrap();
    eprintln!(">>> t31_provider_openrouter [model={model}]");
    let mut history = ConversationHistory::new(SYS.into());
    history.push_user("Say exactly: OPENROUTER_OK_31. Nothing else.");
    let r = run_prompt(provider, &mut history, tmp.path(), model).await;

    eprintln!("  text: {}", r.full_text());
    assert!(!r.had_error(), "agent error");
    assert!(r.got_idle(), "no idle");
    assert!(
        r.full_text().contains("OPENROUTER_OK_31"),
        "missing phrase: {}",
        r.full_text()
    );
}

#[tokio::test]
async fn t32_provider_sambanova() {
    need_config!(config);
    pace().await;
    let tmp = tempfile::tempdir().unwrap();
    let model = "Meta-Llama-3.1-70B-Instruct";

    let provider = match make_provider_for(&config, "sambanova", model) {
        Some(p) => p,
        None => {
            eprintln!("SKIP t32: sambanova not configured or key missing");
            return;
        }
    };

    if !probe_provider(provider.as_ref(), model).await {
        eprintln!("SKIP t32: sambanova probe failed");
        return;
    }
    pace().await;

    let provider = make_provider_for(&config, "sambanova", model).unwrap();
    eprintln!(">>> t32_provider_sambanova [model={model}]");
    let mut history = ConversationHistory::new(SYS.into());
    history.push_user("Say exactly: SAMBANOVA_OK_32. Nothing else.");
    let r = run_prompt(provider, &mut history, tmp.path(), model).await;

    eprintln!("  text: {}", r.full_text());
    assert!(!r.had_error(), "agent error");
    assert!(r.got_idle(), "no idle");
    assert!(
        r.full_text().contains("SAMBANOVA_OK_32"),
        "missing phrase: {}",
        r.full_text()
    );
}

#[tokio::test]
async fn t33_provider_kimi_code() {
    need_config!(config);
    pace().await;
    let tmp = tempfile::tempdir().unwrap();
    let model = "k2p5";

    let provider = match make_provider_for(&config, "kimi-code", model) {
        Some(p) => p,
        None => {
            eprintln!("SKIP t33: kimi-code not configured or key missing");
            return;
        }
    };

    if !probe_provider(provider.as_ref(), model).await {
        eprintln!("SKIP t33: kimi-code probe failed");
        return;
    }
    pace().await;

    let provider = make_provider_for(&config, "kimi-code", model).unwrap();
    eprintln!(">>> t33_provider_kimi_code [model={model}]");
    let mut history = ConversationHistory::new(SYS.into());
    history.push_user("Say exactly: KIMI_CODE_OK_33. Nothing else.");
    let r = run_prompt(provider, &mut history, tmp.path(), model).await;

    eprintln!("  text: {}", r.full_text());
    assert!(!r.had_error(), "agent error");
    assert!(r.got_idle(), "no idle");
    assert!(
        r.full_text().contains("KIMI_CODE_OK_33"),
        "missing phrase: {}",
        r.full_text()
    );
}

#[tokio::test]
async fn t37_provider_anthropic_native() {
    need_config!(config);
    pace().await;
    let tmp = tempfile::tempdir().unwrap();
    let model = "claude-haiku-4-5-20251001";

    let pc = match config.providers.get("anthropic") {
        Some(pc) => pc,
        None => {
            eprintln!("SKIP t37: anthropic not configured");
            return;
        }
    };

    let resolved = match pc.resolved() {
        Ok(r) => r,
        Err(_) => {
            eprintln!("SKIP t37: anthropic key unresolvable");
            return;
        }
    };

    let provider = naked_core::create_provider("anthropic", resolved.clone());

    if !probe_provider(provider.as_ref(), model).await {
        eprintln!("SKIP t37: anthropic probe failed");
        return;
    }
    pace().await;

    let provider = naked_core::create_provider("anthropic", resolved);
    eprintln!(
        ">>> t37_provider_anthropic_native [model={model}, keys={}]",
        pc.resolved_all_keys().len()
    );
    let mut history = ConversationHistory::new(SYS.into());
    history.push_user("Say exactly: ANTHROPIC_OK_37. Nothing else.");
    let r = run_prompt(provider, &mut history, tmp.path(), model).await;

    eprintln!("  text: {}", r.full_text());
    assert!(!r.had_error(), "agent error");
    assert!(r.got_idle(), "no idle");
    assert!(
        r.full_text().contains("ANTHROPIC_OK_37"),
        "missing phrase: {}",
        r.full_text()
    );
}

#[tokio::test]
async fn t44_set_session_provider_command() {
    need_config!(config);
    let providers = find_working_providers(&config, 2).await;
    if providers.len() < 2 {
        eprintln!(
            "SKIP t44: need 2 working providers, found {}",
            providers.len()
        );
        return;
    }
    let (prov_a, model_a) = &providers[0];
    let (prov_b, model_b) = &providers[1];
    eprintln!(">>> t44_set_session_provider_command [{prov_a}/{model_a} -> {prov_b}/{model_b}]");

    let tmp = tempfile::tempdir().unwrap();
    let agent = make_agent_core(&config, tmp.path(), prov_a, model_a);
    let session_id = agent.create_session(tmp.path()).await;

    // Verify initial state
    let (cur_p, cur_m) = agent.session_provider_model(&session_id).await;
    assert_eq!(cur_p, *prov_a);
    assert_eq!(cur_m, *model_a);
    eprintln!("  initial: {cur_p}/{cur_m}");

    // list_providers returns all configured providers
    let providers_list = agent.list_providers();
    assert!(
        providers_list.len() >= 2,
        "expected at least 2 providers, got {}",
        providers_list.len()
    );
    eprintln!(
        "  providers: {:?}",
        providers_list.iter().map(|p| &p.name).collect::<Vec<_>>()
    );

    // Switch provider via set_session_provider
    agent
        .set_session_provider(&session_id, Some(prov_b), Some(model_b))
        .await
        .unwrap();
    let (cur_p, cur_m) = agent.session_provider_model(&session_id).await;
    assert_eq!(cur_p, *prov_b, "provider not switched");
    assert_eq!(cur_m, *model_b, "model not switched");
    eprintln!("  after switch: {cur_p}/{cur_m}");

    // Verify config.json was persisted
    let config_path = tmp
        .path()
        .join("sessions")
        .join(&session_id)
        .join("config.json");
    assert!(config_path.exists(), "config.json not written");
    let raw = tokio::fs::read_to_string(&config_path).await.unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&raw).unwrap();
    assert_eq!(parsed["default_provider"].as_str(), Some(prov_b.as_str()));
    assert_eq!(parsed["default_model"].as_str(), Some(model_b.as_str()));
    eprintln!("  config.json: {parsed}");

    // Session metadata should reflect the switch
    let sessions = agent.list_sessions().await;
    let meta = sessions.iter().find(|s| s.id == session_id).unwrap();
    assert_eq!(meta.provider.as_deref(), Some(prov_b.as_str()));
    assert_eq!(meta.model.as_deref(), Some(model_b.as_str()));

    // Actually send a prompt to verify the switched provider works
    pace().await;
    let (text, _, idle) = agent_prompt(
        &agent,
        &session_id,
        "Say exactly: SWITCHED_OK. Nothing else.",
    )
    .await;
    eprintln!("  response: {text}");
    assert!(idle, "no idle after provider switch");
    assert!(!text.is_empty(), "empty response after provider switch");

    // set_session_provider with unknown provider should fail
    let err = agent
        .set_session_provider(&session_id, Some("nonexistent_provider_xyz"), None)
        .await;
    assert!(err.is_err(), "expected error for unknown provider");
    eprintln!("  unknown provider error: {}", err.unwrap_err());

    // Switch only model (keep provider) — use a model that actually exists
    // on the current provider, or verify that an unknown model is rejected.
    let err = agent
        .set_session_provider(&session_id, None, Some("nonexistent-model-xyz"))
        .await;
    assert!(err.is_err(), "expected error for unknown model");
    eprintln!("  unknown model error: {}", err.unwrap_err());
}

#[tokio::test]
async fn t52_provider_http_timeouts() {
    let config = naked_core::config::ProviderConfig {
        provider_type: "openai_compat".into(),
        api_key: "test-key".into(),
        api_keys: vec![],
        base_url: Some("http://localhost:1".into()),
        models: vec!["test".into()],
        max_tokens: None,
        temperature: None,
        context_window: None,
        headers: Default::default(),
        supports_vision: None,
        model_aliases: Default::default(),
        capabilities: Default::default(),
    };

    let provider =
        naked_core::provider::openai_compat::OpenAiCompatProvider::new("test".into(), config);
    let req = naked_core::provider::ChatRequest {
        model: "test".into(),
        system: String::new(),
        messages: vec![serde_json::json!({"role": "user", "content": "hi"})],
        tools: vec![],
        max_tokens: 8,
        temperature: None,
        reasoning: None,
    };

    let start = std::time::Instant::now();
    let result = tokio::time::timeout(Duration::from_secs(15), provider.stream_chat(req)).await;
    let elapsed = start.elapsed();

    // Should fail quickly (connect timeout = 10s) rather than hang forever
    assert!(
        elapsed < Duration::from_secs(14),
        "should not hang: {:?}",
        elapsed
    );
    match result {
        Ok(Err(_)) => {} // expected: connection refused
        Ok(Ok(_)) => panic!("should not connect to localhost:1"),
        Err(_) => panic!("timed out at wrapper level, connect_timeout not working"),
    }

    eprintln!(
        "  PASS: HTTP client has connect timeout ({:.1}s)",
        elapsed.as_secs_f64()
    );
}
