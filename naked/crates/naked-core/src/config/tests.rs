use super::*;
use std::path::Path;

#[test]
fn vision_capable_model_builtin_allowlist() {
    let cfg = TgMediaConfig::default();
    // Anthropic Claude 3+
    assert!(cfg.is_vision_capable_model("claude-sonnet-4-20250514"));
    assert!(cfg.is_vision_capable_model("claude-3-5-sonnet-20241022"));
    assert!(cfg.is_vision_capable_model("claude-haiku-4-5-20251001"));
    // OpenAI
    assert!(cfg.is_vision_capable_model("gpt-4o"));
    assert!(cfg.is_vision_capable_model("gpt-4o-mini"));
    // Groq Llama-4
    assert!(cfg.is_vision_capable_model("meta-llama/llama-4-scout-17b-16e-instruct"));
    assert!(cfg.is_vision_capable_model("meta-llama/Llama-4-Maverick-17B-128E-Instruct"));
    // xAI
    assert!(cfg.is_vision_capable_model("grok-2-vision-latest"));
    // Gemini
    assert!(cfg.is_vision_capable_model("gemini-1.5-pro"));
    // Text-only models — NOT vision
    assert!(!cfg.is_vision_capable_model("llama-3.3-70b-versatile"));
    assert!(!cfg.is_vision_capable_model("glm-5-turbo"));
    assert!(!cfg.is_vision_capable_model("MiniMax-Text-01"));
    assert!(!cfg.is_vision_capable_model("deepseek-chat"));
    assert!(!cfg.is_vision_capable_model("kimi-k2"));
}

#[test]
fn provider_supports_vision_override_force_true() {
    let cfg = TgMediaConfig::default();
    let mut pc = test_pc("k");
    pc.supports_vision = Some(true);
    // Random unknown text-only model name → still treated as vision-capable.
    assert!(cfg.is_vision_capable_with_provider("brand-new-2099-omni", Some(&pc)));
}

#[test]
fn provider_supports_vision_override_force_false() {
    let cfg = TgMediaConfig::default();
    let mut pc = test_pc("k");
    pc.supports_vision = Some(false);
    // claude-3 normally hits the built-in needles → forced off here.
    assert!(!cfg.is_vision_capable_with_provider("claude-3-5-sonnet-20240620", Some(&pc)));
}

#[test]
fn model_vision_overrides_take_precedence() {
    let mut cfg = TgMediaConfig::default();
    cfg.model_vision_overrides
        .insert("gpt-4o-mini".into(), false); // force off even though needles say yes
    cfg.model_vision_overrides
        .insert("custom-llama-vision".into(), true);
    let mut pc = test_pc("k");
    pc.supports_vision = Some(true);
    // Per-model false beats provider-wide true and built-in needle.
    assert!(!cfg.is_vision_capable_with_provider("gpt-4o-mini", Some(&pc)));
    // Per-model true lights up an unknown model regardless of provider config.
    assert!(cfg.is_vision_capable_with_provider("custom-llama-vision-7b", None));
}

#[test]
fn provider_image_cap_per_provider_floors() {
    let cfg = TgMediaConfig {
        native_image_max_bytes: 50 * 1024 * 1024,
        ..TgMediaConfig::default()
    };
    // Anthropic floor → 5 MB, even with global 50 MB.
    assert_eq!(cfg.provider_image_cap("anthropic", None), 5 * 1024 * 1024);
    assert_eq!(
        cfg.provider_image_cap("openai_compat", Some("https://api.groq.com/openai/v1")),
        4 * 1024 * 1024
    );
    assert_eq!(
        cfg.provider_image_cap("openai_compat", Some("https://api.openai.com/v1")),
        20 * 1024 * 1024
    );
    // Unknown base URL — falls back to global.
    assert_eq!(
        cfg.provider_image_cap("openai_compat", Some("https://example.com")),
        50 * 1024 * 1024
    );
}

#[test]
fn provider_image_cap_respects_global_below_provider_limit() {
    let cfg = TgMediaConfig {
        native_image_max_bytes: 1024 * 1024, // 1 MB global
        ..TgMediaConfig::default()
    };
    // Global is the floor when stricter than provider's own cap.
    assert_eq!(
        cfg.provider_image_cap("openai_compat", Some("https://api.openai.com/v1")),
        1024 * 1024
    );
}

#[test]
fn provider_supports_vision_none_falls_back_to_needles() {
    let cfg = TgMediaConfig::default();
    let pc = test_pc("k");
    // None → fall through to needle list.
    assert!(cfg.is_vision_capable_with_provider("claude-haiku-4-5-20251001", Some(&pc)));
    assert!(!cfg.is_vision_capable_with_provider("llama-3.3-70b-versatile", Some(&pc)));
}

#[test]
fn vision_capable_model_extras_extend_allowlist() {
    let mut cfg = TgMediaConfig::default();
    assert!(!cfg.is_vision_capable_model("internlm-xcomposer2.5-7b"));
    cfg.vision_model_extras = vec!["internlm-xcomposer".into(), "qwen3-vl".into()];
    assert!(cfg.is_vision_capable_model("internlm-xcomposer2.5-7b"));
    assert!(cfg.is_vision_capable_model("Qwen3-VL-72B"));
    // Empty entries are ignored.
    cfg.vision_model_extras = vec!["".into()];
    assert!(!cfg.is_vision_capable_model("anything"));
}

#[test]
fn parse_minimal_json() {
    let json = r#"{"providers": {}, "default_provider": "", "default_model": ""}"#;
    let cfg = Config::from_json_str(json).unwrap();
    assert!(cfg.providers.is_empty());
    assert_eq!(cfg.max_iterations, 0);
}

#[test]
fn parse_full_json() {
    let json = r#"{
        "providers": {
            "claude": {
                "type": "anthropic",
                "api_key": "sk-test",
                "models": ["claude-sonnet-4"]
            },
            "groq": {
                "type": "openai_compat",
                "api_key": "$GROQ_API_KEY",
                "base_url": "https://api.groq.com/openai/v1",
                "models": ["llama-3.3-70b"]
            }
        },
        "default_provider": "claude",
        "default_model": "claude-sonnet-4",
        "fallback": ["groq/llama-3.3-70b"],
        "max_iterations": 30,
        "mcpServers": {
            "fs": {"command": "mcp-fs", "args": ["--root", "/"]}
        }
    }"#;
    let cfg = Config::from_json_str(json).unwrap();
    assert_eq!(cfg.providers.len(), 2);
    assert_eq!(cfg.default_provider, "claude");
    assert_eq!(cfg.max_iterations, 30);
    assert_eq!(cfg.mcp_server_list().len(), 1);
    assert_eq!(cfg.fallback_providers().len(), 1);
    assert_eq!(
        cfg.fallback_providers()[0],
        ("groq".into(), "llama-3.3-70b".into())
    );
}

fn test_pc(api_key: &str) -> ProviderConfig {
    ProviderConfig {
        provider_type: "openai_compat".into(),
        api_key: api_key.into(),
        api_keys: Vec::new(),
        base_url: None,
        models: Vec::new(),
        max_tokens: None,
        temperature: None,
        context_window: None,
        headers: Default::default(),
        supports_vision: None,
        model_aliases: Default::default(),
        capabilities: Default::default(),
    }
}

#[test]
fn provider_resolved_direct_key() {
    let mut pc = test_pc("sk-direct-key");
    pc.provider_type = "anthropic".into();
    let resolved = pc.resolved().unwrap();
    assert_eq!(resolved.api_key, "sk-direct-key");
}

#[test]
fn resolved_all_keys_single() {
    let keys = test_pc("key-a").resolved_all_keys();
    assert_eq!(keys, vec!["key-a"]);
}

#[test]
fn resolved_all_keys_multiple() {
    let mut pc = test_pc("key-a");
    pc.api_keys = vec!["key-b".into(), "key-c".into()];
    assert_eq!(pc.resolved_all_keys(), vec!["key-a", "key-b", "key-c"]);
}

#[test]
fn resolved_all_keys_deduplicates() {
    let mut pc = test_pc("key-a");
    pc.api_keys = vec!["key-a".into(), "key-b".into()];
    assert_eq!(pc.resolved_all_keys(), vec!["key-a", "key-b"]);
}

#[test]
fn resolved_all_keys_skips_unresolved_env() {
    let mut pc = test_pc("key-a");
    pc.api_keys = vec!["$NONEXISTENT_KEY_FOR_TEST_XYZ".into(), "key-b".into()];
    assert_eq!(pc.resolved_all_keys(), vec!["key-a", "key-b"]);
}

#[test]
fn resolved_provider_has_all_keys() {
    let mut pc = test_pc("key-1");
    pc.provider_type = "anthropic".into();
    pc.api_keys = vec!["key-2".into()];
    let resolved = pc.resolved().unwrap();
    assert_eq!(resolved.all_keys, vec!["key-1", "key-2"]);
}

#[test]
fn resolved_provider_carries_max_tokens_and_temperature() {
    let mut pc = test_pc("key-x");
    pc.max_tokens = Some(4096);
    pc.temperature = Some(0.7);
    let resolved = pc.resolved().unwrap();
    assert_eq!(resolved.max_tokens, Some(4096));
    assert_eq!(resolved.temperature, Some(0.7));
}

#[test]
fn effective_max_tokens_global_vs_provider() {
    let json = r#"{
        "providers": {
            "fast": {"type": "openai_compat", "api_key": "k", "max_tokens": 4096},
            "big": {"type": "openai_compat", "api_key": "k"}
        },
        "max_tokens": 8192
    }"#;
    let cfg = Config::from_json_str(json).unwrap();
    assert_eq!(cfg.effective_max_tokens("fast"), 4096);
    assert_eq!(cfg.effective_max_tokens("big"), 8192);
    assert_eq!(cfg.effective_max_tokens("missing"), 8192);
}

#[test]
fn effective_temperature_global_vs_provider() {
    let json = r#"{
        "providers": {
            "creative": {"type": "openai_compat", "api_key": "k", "temperature": 0.9},
            "default": {"type": "openai_compat", "api_key": "k"}
        },
        "temperature": 0.3
    }"#;
    let cfg = Config::from_json_str(json).unwrap();
    assert_eq!(cfg.effective_temperature("creative"), Some(0.9));
    assert_eq!(cfg.effective_temperature("default"), Some(0.3));
    assert_eq!(cfg.effective_temperature("missing"), Some(0.3));
}

#[test]
fn temperature_none_when_unset() {
    let json = r#"{"providers": {"p": {"type": "openai_compat", "api_key": "k"}}}"#;
    let cfg = Config::from_json_str(json).unwrap();
    assert_eq!(cfg.effective_temperature("p"), None);
}

#[test]
fn multi_key_json_roundtrip() {
    let json = r#"{
        "type": "openai_compat",
        "api_key": "primary",
        "api_keys": ["backup1", "backup2"],
        "models": ["gpt-4o"]
    }"#;
    let pc: ProviderConfig = serde_json::from_str(json).unwrap();
    assert_eq!(pc.api_key, "primary");
    assert_eq!(pc.api_keys, vec!["backup1", "backup2"]);
    let keys = pc.resolved_all_keys();
    assert_eq!(keys, vec!["primary", "backup1", "backup2"]);
}

#[test]
fn expand_env_direct_value() {
    assert_eq!(expand_env("plain-key").unwrap(), "plain-key");
}

#[test]
fn expand_env_dollar_var() {
    // Test with a var we know exists in any Unix environment
    let home = std::env::var("HOME").unwrap_or_default();
    if !home.is_empty() {
        assert_eq!(expand_env("$HOME").unwrap(), home);
    }
}

#[test]
fn expand_env_braces() {
    let home = std::env::var("HOME").unwrap_or_default();
    if !home.is_empty() {
        assert_eq!(expand_env("${HOME}").unwrap(), home);
    }
}

#[test]
fn expand_env_missing_var_is_error() {
    assert!(expand_env("$DEFINITELY_NOT_SET_ABCXYZ").is_err());
}

#[test]
fn session_dir_abs_relative() {
    let cfg = Config {
        workspace: PathBuf::from("/home/user/project"),
        session_dir: PathBuf::from(".naked/sessions"),
        ..Default::default()
    };
    assert_eq!(
        cfg.session_dir_abs(),
        PathBuf::from("/home/user/project/.naked/sessions")
    );
}

#[test]
fn session_dir_abs_absolute() {
    let cfg = Config {
        session_dir: PathBuf::from("/tmp/sessions"),
        ..Default::default()
    };
    assert_eq!(cfg.session_dir_abs(), PathBuf::from("/tmp/sessions"));
}

#[test]
fn default_config_empty() {
    let cfg = Config::default();
    assert!(cfg.providers.is_empty());
    assert!(cfg.default_provider.is_empty());
    assert_eq!(cfg.max_iterations, 0);
}

#[test]
fn mcp_servers_get_names() {
    let json = r#"{"mcpServers": {"my-server": {"command": "test"}}}"#;
    let mut cfg = Config::from_json_str(json).unwrap();
    cfg.resolve_mcp_names();
    let servers = cfg.mcp_server_list();
    assert_eq!(servers[0].name, "my-server");
}

#[test]
fn resolve_default_provider_picks_first() {
    let json = r#"{"providers": {"only": {"type": "openai_compat", "api_key": "k"}}}"#;
    let cfg = Config::from_json_str(json).unwrap();
    let (name, resolved) = cfg.resolve_default_provider().unwrap();
    assert_eq!(name, "only");
    assert_eq!(resolved.api_key, "k");
}

#[test]
fn parse_full_opencode_catalog() {
    let json = include_str!("../../../../config.example.json");
    let cfg = Config::from_json_str(json).unwrap();

    // All providers from OpenCode/ZeroClaw
    assert!(
        cfg.providers.len() >= 40,
        "expected 40+ providers, got {}",
        cfg.providers.len()
    );

    // Spot-check key providers
    let check = |name: &str, expected_type: &str, has_url: bool| {
        let pc = cfg
            .providers
            .get(name)
            .unwrap_or_else(|| panic!("missing provider: {name}"));
        assert_eq!(pc.provider_type, expected_type, "wrong type for {name}");
        if has_url {
            assert!(pc.base_url.is_some(), "missing base_url for {name}");
        }
    };

    check("anthropic", "anthropic", false);
    check("openai", "openai_compat", true);
    check("groq", "openai_compat", true);
    check("fireworks", "openai_compat", true);
    check("deepseek", "openai_compat", true);
    check("mistral", "openai_compat", true);
    check("xai", "openai_compat", true);
    check("together", "openai_compat", true);
    check("openrouter", "openai_compat", true);
    check("gemini", "openai_compat", true);
    check("minimax", "openai_compat", true);
    check("glm", "openai_compat", true);
    check("moonshot", "openai_compat", true);
    check("kimi-code", "openai_compat", true);
    check("qwen", "openai_compat", true);
    check("ollama", "openai_compat", true);
    check("nvidia", "openai_compat", true);
    check("perplexity", "openai_compat", true);
    check("cohere", "openai_compat", true);
    check("cerebras", "openai_compat", true);
    check("siliconflow", "openai_compat", true);
    check("telnyx", "openai_compat", true);
    check("azure-openai", "openai_compat", true);

    // Defaults
    assert_eq!(cfg.default_provider, "anthropic");
    assert_eq!(cfg.default_model, "claude-sonnet-4-20250514");

    // Fallbacks parsed
    let fb = cfg.fallback_providers();
    assert_eq!(fb.len(), 3);
    assert_eq!(fb[0].0, "groq");
    assert_eq!(fb[1].0, "deepseek");
    assert_eq!(fb[2].0, "openai");

    // Model lists
    let anthropic_models = &cfg.providers["anthropic"].models;
    assert!(anthropic_models.contains(&"claude-sonnet-4-20250514".to_string()));
    assert!(anthropic_models.contains(&"claude-opus-4-20250514".to_string()));

    let openai_models = &cfg.providers["openai"].models;
    assert!(openai_models.contains(&"gpt-4o".to_string()));
    assert!(openai_models.contains(&"o3".to_string()));

    let groq_models = &cfg.providers["groq"].models;
    assert!(groq_models.contains(&"llama-3.3-70b-versatile".to_string()));

    // Every provider can resolve with a direct key
    for (name, pc) in &cfg.providers {
        if !pc.api_key.starts_with('$') {
            let resolved = pc.resolved().unwrap();
            assert!(!resolved.api_key.is_empty(), "empty key for {name}");
        }
    }
}

#[test]
fn all_provider_types_valid() {
    let json = include_str!("../../../../config.example.json");
    let cfg = Config::from_json_str(json).unwrap();

    let valid_types = ["anthropic", "openai_compat"];
    for (name, pc) in &cfg.providers {
        assert!(
            valid_types.contains(&pc.provider_type.as_str()),
            "invalid provider_type '{}' for provider '{name}'",
            pc.provider_type
        );
    }
}

#[test]
fn expand_tilde_home() {
    let home = dirs_home();
    assert_eq!(expand_tilde(Path::new("~")), home);
    assert_eq!(expand_tilde(Path::new("~/foo/bar")), home.join("foo/bar"));
    assert_eq!(
        expand_tilde(Path::new("/abs/path")),
        PathBuf::from("/abs/path")
    );
    assert_eq!(
        expand_tilde(Path::new("relative")),
        PathBuf::from("relative")
    );
}

#[test]
fn skill_roots_tilde_expanded() {
    let json = r#"{"skill_roots": ["~/.zeroclaw/workspace/skills", "/absolute/path"]}"#;
    let mut cfg = Config::from_json_str(json).unwrap();
    cfg.skill_roots = cfg
        .skill_roots
        .into_iter()
        .map(|p| expand_tilde(&p))
        .collect();
    let home = dirs_home();
    assert_eq!(cfg.skill_roots[0], home.join(".zeroclaw/workspace/skills"));
    assert_eq!(cfg.skill_roots[1], PathBuf::from("/absolute/path"));
}

#[test]
fn base_urls_are_valid() {
    let json = include_str!("../../../../config.example.json");
    let cfg = Config::from_json_str(json).unwrap();

    for (name, pc) in &cfg.providers {
        if let Some(url) = &pc.base_url {
            assert!(
                url.starts_with("http://") || url.starts_with("https://"),
                "invalid base_url '{url}' for provider '{name}'"
            );
        }
    }
}

// ── SessionConfig + merge tests ─────────────────────────────────────

#[test]
fn session_config_parse_empty() {
    let sc: SessionConfig = serde_json::from_str("{}").unwrap();
    assert!(sc.default_provider.is_none());
    assert!(sc.default_model.is_none());
    assert!(sc.max_tokens.is_none());
    assert!(sc.temperature.is_none());
    assert!(sc.max_iterations.is_none());
    assert!(sc.mcp_servers.is_none());
    assert!(sc.skill_roots.is_none());
    assert!(sc.system_prompt_path.is_none());
}

#[test]
fn session_config_parse_full() {
    let json = r#"{
        "default_provider": "anthropic",
        "default_model": "claude-sonnet-4",
        "max_tokens": 4096,
        "temperature": 0.5,
        "max_iterations": 20,
        "mcpServers": {"db": {"command": "db-server"}},
        "skill_roots": ["/my/skills"],
        "system_prompt_path": "./custom.md"
    }"#;
    let sc: SessionConfig = serde_json::from_str(json).unwrap();
    assert_eq!(sc.default_provider.as_deref(), Some("anthropic"));
    assert_eq!(sc.default_model.as_deref(), Some("claude-sonnet-4"));
    assert_eq!(sc.max_tokens, Some(4096));
    assert_eq!(sc.temperature, Some(0.5));
    assert_eq!(sc.max_iterations, Some(20));
    assert_eq!(sc.mcp_servers.as_ref().unwrap().len(), 1);
    assert_eq!(sc.skill_roots.as_ref().unwrap().len(), 1);
}

#[test]
fn merge_empty_session_returns_global() {
    let cfg = Config {
        default_provider: "groq".into(),
        default_model: "llama".into(),
        max_tokens: 8192,
        temperature: Some(0.3),
        max_iterations: 30,
        ..Default::default()
    };
    let eff = cfg.merge_session(&SessionConfig::default());
    assert_eq!(eff.provider, "groq");
    assert_eq!(eff.model, "llama");
    assert_eq!(eff.max_tokens, 8192);
    assert_eq!(eff.temperature, Some(0.3));
    assert_eq!(eff.max_iterations, 30);
}

#[test]
fn merge_session_overrides_fields() {
    let cfg = Config {
        default_provider: "groq".into(),
        default_model: "llama".into(),
        max_tokens: 8192,
        temperature: Some(0.3),
        max_iterations: 30,
        ..Default::default()
    };
    let sc = SessionConfig {
        default_provider: Some("anthropic".into()),
        default_model: Some("claude".into()),
        max_tokens: Some(2048),
        temperature: Some(0.9),
        max_iterations: Some(10),
        ..Default::default()
    };
    let eff = cfg.merge_session(&sc);
    assert_eq!(eff.provider, "anthropic");
    assert_eq!(eff.model, "claude");
    assert_eq!(eff.max_tokens, 2048);
    assert_eq!(eff.temperature, Some(0.9));
    assert_eq!(eff.max_iterations, 10);
}

#[test]
fn merge_session_partial_override() {
    let cfg = Config {
        default_provider: "groq".into(),
        default_model: "llama".into(),
        max_tokens: 8192,
        max_iterations: 30,
        ..Default::default()
    };
    let sc = SessionConfig {
        default_model: Some("mixtral".into()),
        ..Default::default()
    };
    let eff = cfg.merge_session(&sc);
    assert_eq!(eff.provider, "groq");
    assert_eq!(eff.model, "mixtral");
    assert_eq!(eff.max_tokens, 8192);
    assert_eq!(eff.max_iterations, 30);
}

#[test]
fn merge_mcp_servers_additive() {
    let mut global_mcp = HashMap::new();
    global_mcp.insert(
        "echo".into(),
        McpServerConfig {
            name: "echo".into(),
            command: "echo-server".into(),
            ..Default::default()
        },
    );
    let cfg = Config {
        mcp_servers: global_mcp,
        ..Default::default()
    };

    let mut session_mcp = HashMap::new();
    session_mcp.insert(
        "db".into(),
        McpServerConfig {
            name: "db".into(),
            command: "db-server".into(),
            ..Default::default()
        },
    );
    let sc = SessionConfig {
        mcp_servers: Some(session_mcp),
        ..Default::default()
    };

    let eff = cfg.merge_session(&sc);
    assert_eq!(eff.mcp_servers.len(), 2);
    assert!(eff.mcp_servers.contains_key("echo"));
    assert!(eff.mcp_servers.contains_key("db"));
}

#[test]
fn merge_mcp_session_overrides_same_name() {
    let mut global_mcp = HashMap::new();
    global_mcp.insert(
        "server".into(),
        McpServerConfig {
            name: "server".into(),
            command: "global-cmd".into(),
            ..Default::default()
        },
    );
    let cfg = Config {
        mcp_servers: global_mcp,
        ..Default::default()
    };

    let mut session_mcp = HashMap::new();
    session_mcp.insert(
        "server".into(),
        McpServerConfig {
            name: "server".into(),
            command: "session-cmd".into(),
            ..Default::default()
        },
    );
    let sc = SessionConfig {
        mcp_servers: Some(session_mcp),
        ..Default::default()
    };

    let eff = cfg.merge_session(&sc);
    assert_eq!(eff.mcp_servers.len(), 1);
    assert_eq!(eff.mcp_servers["server"].command, "session-cmd");
}

#[test]
fn merge_skill_roots_session_wins() {
    let cfg = Config {
        skill_roots: vec![PathBuf::from("/global/skills")],
        ..Default::default()
    };
    let sc = SessionConfig {
        skill_roots: Some(vec![PathBuf::from("/session/skills")]),
        ..Default::default()
    };
    let eff = cfg.merge_session(&sc);
    assert_eq!(eff.skill_roots, vec![PathBuf::from("/session/skills")]);
}

#[test]
fn merge_skill_roots_fallback_to_global() {
    let cfg = Config {
        skill_roots: vec![PathBuf::from("/global/skills")],
        ..Default::default()
    };
    let eff = cfg.merge_session(&SessionConfig::default());
    assert_eq!(eff.skill_roots, vec![PathBuf::from("/global/skills")]);
}

#[test]
fn session_config_from_file() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("config.json");
    std::fs::write(&path, r#"{"default_model": "gpt-4o", "max_tokens": 1024}"#).unwrap();
    let sc = SessionConfig::from_file(&path).unwrap();
    assert_eq!(sc.default_model.as_deref(), Some("gpt-4o"));
    assert_eq!(sc.max_tokens, Some(1024));
    assert!(sc.default_provider.is_none());
}

#[test]
fn default_effective_matches_global() {
    let cfg = Config {
        default_provider: "test".into(),
        default_model: "m1".into(),
        max_tokens: 4000,
        temperature: Some(0.5),
        max_iterations: 25,
        ..Default::default()
    };
    let eff = cfg.default_effective();
    assert_eq!(eff.provider, "test");
    assert_eq!(eff.model, "m1");
    assert_eq!(eff.max_tokens, 4000);
    assert_eq!(eff.temperature, Some(0.5));
    assert_eq!(eff.max_iterations, 25);
}

#[test]
fn validate_and_warn_never_panics_on_empty() {
    // Pure smoke test: a blank default-Config must not crash validation.
    // Warnings go to tracing; we just assert the function returns.
    let cfg = Config::default();
    cfg.validate_and_warn();
}

#[test]
fn validate_and_warn_mismatched_default_provider_is_tolerated() {
    // Wrong `default_provider` must log and return, not panic.
    let mut providers = HashMap::new();
    providers.insert("real".to_string(), test_pc("sk"));
    let cfg = Config {
        providers,
        default_provider: "typo".into(),
        default_model: "m1".into(),
        ..Default::default()
    };
    cfg.validate_and_warn();
}

#[test]
fn research_config_defaults_sane() {
    let rc = ResearchConfig::default();
    assert!(rc.enabled);
    assert_eq!(rc.max_iterations, 30);
    assert_eq!(rc.max_wall_seconds, 1200);
    assert!(rc.provider.is_none());
    assert!(rc.model.is_none());
    assert!(rc.default_sources.is_empty());
    assert!(rc.allowed_tools.is_none());
}

#[test]
fn research_config_parses_from_json() {
    let json = r#"{
        "providers": {
            "qwen": {"type": "openai_compat", "api_key": "k", "models": ["qwen3.6-plus"]}
        },
        "research": {
            "enabled": true,
            "provider": "qwen",
            "model": "qwen3.6-plus",
            "max_iterations": 20,
            "max_wall_seconds": 900,
            "default_sources": ["https://chotot.com"]
        }
    }"#;
    let cfg = Config::from_json_str(json).unwrap();
    assert_eq!(cfg.research.provider.as_deref(), Some("qwen"));
    assert_eq!(cfg.research.model.as_deref(), Some("qwen3.6-plus"));
    assert_eq!(cfg.research.max_iterations, 20);
    assert_eq!(cfg.research.max_wall_seconds, 900);
    assert_eq!(cfg.research.default_sources.len(), 1);
}

#[test]
fn research_config_missing_section_uses_defaults() {
    let cfg = Config::from_json_str(r#"{"providers": {}}"#).unwrap();
    assert!(cfg.research.enabled);
    assert_eq!(cfg.research.max_iterations, 30);
}

#[test]
fn validate_and_warn_mismatched_research_provider_is_tolerated() {
    let mut providers = HashMap::new();
    providers.insert("qwen".to_string(), test_pc("k"));
    let cfg = Config {
        providers,
        default_provider: "qwen".into(),
        default_model: "qwen3.6-plus".into(),
        research: ResearchConfig {
            enabled: true,
            provider: Some("typo-provider".into()),
            model: Some("qwen3.6-plus".into()),
            ..Default::default()
        },
        ..Default::default()
    };
    // Must not panic — warns and returns.
    cfg.validate_and_warn();
}

#[test]
fn validate_and_warn_mismatched_research_model_is_tolerated() {
    let mut pc = test_pc("k");
    pc.models = vec!["qwen3.6-plus".into()];
    let mut providers = HashMap::new();
    providers.insert("qwen".to_string(), pc);
    let cfg = Config {
        providers,
        default_provider: "qwen".into(),
        default_model: "qwen3.6-plus".into(),
        research: ResearchConfig {
            enabled: true,
            provider: Some("qwen".into()),
            model: Some("qwen-typo".into()),
            ..Default::default()
        },
        ..Default::default()
    };
    cfg.validate_and_warn();
}

#[test]
fn validate_and_warn_missing_api_key_for_non_copilot_is_tolerated() {
    let mut providers = HashMap::new();
    providers.insert("openai".to_string(), test_pc(""));
    let cfg = Config {
        providers,
        default_provider: "openai".into(),
        default_model: "gpt-4o".into(),
        ..Default::default()
    };
    cfg.validate_and_warn();
}

// --- capability-aware validator tests -------------------------------
//
// These exercise `Config::validate_capabilities` which never panics and
// only emits `tracing::warn!`. We can't inspect the tracing output
// without pulling in a subscriber, so the tests focus on "does not
// panic" + "does not misbehave across a variety of input shapes". The
// hard guarantees (which specific warning fires) are locked in by the
// integration tests in `tests/capabilities_validator.rs`, which capture
// real tracing output.

fn pc_with_caps(
    api_key: &str,
    models: Vec<&str>,
    caps: Vec<(&str, crate::model_catalog::ModelCapabilities)>,
) -> ProviderConfig {
    let mut pc = test_pc(api_key);
    pc.models = models.into_iter().map(String::from).collect();
    pc.capabilities = caps.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
    pc
}

#[test]
fn validate_capabilities_silent_when_unknown_pair() {
    // No capabilities block for the referenced model → permissive
    // unknown() fallback, no warning.
    let pc = pc_with_caps("k", vec!["gpt-4o"], vec![]);
    let mut providers = HashMap::new();
    providers.insert("openai".into(), pc);
    let cfg = Config {
        providers,
        default_provider: "openai".into(),
        default_model: "gpt-4o".into(),
        ..Default::default()
    };
    cfg.validate_and_warn();
}

#[test]
fn validate_capabilities_fires_on_deprecated_default_model() {
    use crate::model_catalog::{ModelCapabilities, ModelStatus};
    let mut caps = ModelCapabilities::unknown();
    caps.status = ModelStatus::Deprecated;
    let pc = pc_with_caps("k", vec!["qwen3.6-plus"], vec![("qwen3.6-plus", caps)]);
    let mut providers = HashMap::new();
    providers.insert("qwen".into(), pc);
    let cfg = Config {
        providers,
        default_provider: "qwen".into(),
        default_model: "qwen3.6-plus".into(),
        ..Default::default()
    };
    cfg.validate_and_warn();
}

#[test]
fn validate_capabilities_fires_on_task_fit_miss() {
    use crate::model_catalog::{ModelCapabilities, TaskKind};
    let mut caps = ModelCapabilities::unknown();
    // Mark as chat-only, then reference from research.
    caps.task_fit = vec![TaskKind::Chat, TaskKind::Classify];
    let pc = pc_with_caps(
        "k",
        vec!["kimi-for-classify-only"],
        vec![("kimi-for-classify-only", caps)],
    );
    let mut providers = HashMap::new();
    providers.insert("kimi-code".into(), pc);
    let cfg = Config {
        providers,
        default_provider: "kimi-code".into(),
        default_model: "kimi-for-classify-only".into(),
        research: ResearchConfig {
            enabled: true,
            provider: Some("kimi-code".into()),
            model: Some("kimi-for-classify-only".into()),
            ..Default::default()
        },
        ..Default::default()
    };
    cfg.validate_and_warn();
}

#[test]
fn validate_capabilities_fallback_chain_with_deprecated() {
    use crate::model_catalog::{ModelCapabilities, ModelStatus};
    let mut dead = ModelCapabilities::unknown();
    dead.status = ModelStatus::Deprecated;
    let pc = pc_with_caps("k", vec!["live", "zombie"], vec![("zombie", dead)]);
    let mut providers = HashMap::new();
    providers.insert("myprov".into(), pc);
    let cfg = Config {
        providers,
        default_provider: "myprov".into(),
        default_model: "live".into(),
        fallback: vec!["myprov/zombie".into()],
        research: ResearchConfig {
            enabled: true,
            provider: Some("myprov".into()),
            model: Some("live".into()),
            fallback_models: vec!["myprov/zombie".into(), "zombie".into()],
            ..Default::default()
        },
        ..Default::default()
    };
    cfg.validate_and_warn();
}

#[test]
fn validate_capabilities_degraded_model_referenced_from_research() {
    // The canonical case this whole phase exists to prevent:
    // `glm-5-turbo` pinned as a research backend.
    use crate::model_catalog::{ModelCapabilities, ModelStatus, TaskKind, ToolUseLevel};
    let caps = ModelCapabilities {
        status: ModelStatus::Degraded,
        task_fit: vec![TaskKind::Chat, TaskKind::Classify],
        tool_use: ToolUseLevel::TextOnly,
        known_failure_modes: vec!["empty_content".into()],
        ..ModelCapabilities::unknown()
    };
    let pc = pc_with_caps("k", vec!["glm-5-turbo"], vec![("glm-5-turbo", caps)]);
    let mut providers = HashMap::new();
    providers.insert("zai".into(), pc);
    let cfg = Config {
        providers,
        default_provider: "zai".into(),
        default_model: "glm-5-turbo".into(),
        research: ResearchConfig {
            enabled: true,
            provider: Some("zai".into()),
            model: Some("glm-5-turbo".into()),
            ..Default::default()
        },
        ..Default::default()
    };
    cfg.validate_and_warn();
}

#[test]
fn validate_capabilities_digest_provider_string_parses() {
    use crate::model_catalog::{ModelCapabilities, ModelStatus};
    let mut dead = ModelCapabilities::unknown();
    dead.status = ModelStatus::Deprecated;
    let pc = pc_with_caps("k", vec!["gpt-3.5"], vec![("gpt-3.5", dead)]);
    let mut providers = HashMap::new();
    providers.insert("openai".into(), pc);
    let cfg = Config {
        providers,
        default_provider: "openai".into(),
        default_model: "gpt-3.5".into(),
        memory: MemoryConfig {
            digest_provider: Some("openai/gpt-3.5".into()),
            ..Default::default()
        },
        ..Default::default()
    };
    cfg.validate_and_warn();
}

#[test]
fn gatekeeper_config_defaults_are_sane() {
    let gk = GatekeeperConfig::default();
    assert_eq!(gk.max_rounds, 3);
    assert_eq!(gk.min_excerpt_chars, 300);
    assert_eq!(gk.min_source_content_chars, 100);
    assert_eq!(gk.max_listing_age_days, 90);
    assert!(gk.require_listing_date);
    assert!(gk.require_source_content);
    assert!(gk.detect_semantic_duplicates);
    assert!(gk.stop_on_stagnation);
    assert_eq!(gk.url_check_timeout_secs, 10);
    assert!(gk.feedback_prompt_header.contains("{id}"));
    assert!(gk.feedback_prompt_header.contains("{topic}"));
}

#[test]
fn gatekeeper_config_parses_from_json() {
    let json = r#"{
        "providers": {},
        "research": {
            "gatekeeper": {
                "max_rounds": 5,
                "min_excerpt_chars": 300,
                "min_source_content_chars": 200,
                "max_listing_age_days": 30,
                "require_listing_date": false,
                "require_source_content": false,
                "detect_semantic_duplicates": false,
                "stop_on_stagnation": false,
                "url_check_timeout_secs": 15,
                "save_warnings": {
                    "no_listing_date": "custom: add date",
                    "no_source_content": "custom: add source",
                    "short_excerpt": "custom: excerpt too short (need {min_excerpt})"
                }
            }
        }
    }"#;
    let cfg = Config::from_json_str(json).unwrap();
    let gk = &cfg.research.gatekeeper;
    assert_eq!(gk.max_rounds, 5);
    assert_eq!(gk.min_excerpt_chars, 300);
    assert_eq!(gk.min_source_content_chars, 200);
    assert_eq!(gk.max_listing_age_days, 30);
    assert!(!gk.require_listing_date);
    assert!(!gk.require_source_content);
    assert!(!gk.detect_semantic_duplicates);
    assert!(!gk.stop_on_stagnation);
    assert_eq!(gk.url_check_timeout_secs, 15);
    assert_eq!(gk.save_warnings.no_listing_date, "custom: add date");
}

#[test]
fn gatekeeper_config_missing_uses_defaults() {
    let cfg = Config::from_json_str(r#"{"providers": {}}"#).unwrap();
    let gk = &cfg.research.gatekeeper;
    assert_eq!(gk.max_rounds, 3);
    assert_eq!(gk.min_excerpt_chars, 300);
    assert!(gk.require_listing_date);
}

#[test]
fn verify_by_default_default_is_true() {
    let cfg = Config::from_json_str(r#"{"providers": {}}"#).unwrap();
    assert!(cfg.research.verify_by_default);
    assert!(ResearchConfig::default().verify_by_default);
}

#[test]
fn verify_by_default_can_be_disabled() {
    let json = r#"{
        "providers": {},
        "research": {
            "verify_by_default": false
        }
    }"#;
    let cfg = Config::from_json_str(json).unwrap();
    assert!(!cfg.research.verify_by_default);
    assert_eq!(cfg.research.gatekeeper.max_rounds, 3);
}

/// Contract test against the shipped `naked.json` at the repo root.
/// Locks in the kimi-for-coding + reasoning=medium primary research
/// configuration so a careless edit can't silently downgrade quality.
/// Skips gracefully when the file isn't present (downstream consumers
/// of naked-core may not ship naked.json).
#[test]
fn shipped_naked_json_research_uses_kimi_for_coding_with_reasoning() {
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidate = manifest
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join("naked.json"));
    let Some(path) = candidate.filter(|p| p.is_file()) else {
        eprintln!("skipping: naked.json not present at expected path");
        return;
    };
    let raw = std::fs::read_to_string(&path).expect("read naked.json");
    let cfg = Config::from_json_str(&raw).expect("parse naked.json");

    // Research must run on kimi-for-coding via kimi-code provider, with
    // reasoning_effort=medium (per the official Roo Code recipe).
    assert_eq!(cfg.research.provider.as_deref(), Some("kimi-code"));
    assert_eq!(cfg.research.model.as_deref(), Some("kimi-for-coding"));
    assert_eq!(cfg.research.reasoning.as_deref(), Some("medium"));

    // Cross-provider fallback chain must include qwen for resilience
    // when the kimi-code endpoint is throttled / down.
    let chain = &cfg.research.fallback_models;
    assert!(
        chain.iter().any(|m| m.starts_with("qwen/")),
        "fallback chain must include a qwen/* entry — got {chain:?}"
    );

    // The kimi-code provider must list kimi-for-coding among its models
    // so model-validation in `validate_and_warn` doesn't flag it.
    let kimi = cfg
        .providers
        .get("kimi-code")
        .expect("kimi-code provider must be configured");
    assert!(
        kimi.models.iter().any(|m| m == "kimi-for-coding"),
        "kimi-code provider must declare `kimi-for-coding` model"
    );
    assert_eq!(
        kimi.headers.get("User-Agent").map(|s| s.as_str()),
        Some("claude-code/1.0"),
        "kimi-code requires User-Agent: claude-code/1.0 — without it \
         api.kimi.com refuses kimi-for-coding with 403 \
         \"Kimi For Coding is currently only available for Coding Agents\""
    );
}

#[test]
fn research_reasoning_default_is_none() {
    let cfg = Config::from_json_str(r#"{"providers": {}}"#).unwrap();
    assert!(cfg.research.reasoning.is_none());
    assert!(ResearchConfig::default().reasoning.is_none());
}

#[test]
fn default_reasoning_cascades_into_session() {
    // Global default present, session leaves it unset → session inherits.
    let cfg = Config::from_json_str(r#"{"providers": {}, "default_reasoning": "medium"}"#).unwrap();
    let eff = cfg.merge_session(&SessionConfig::default());
    assert_eq!(
        eff.reasoning.as_deref(),
        Some("medium"),
        "session must inherit Config.default_reasoning when SessionConfig.reasoning is None"
    );
}

#[test]
fn session_reasoning_overrides_global_default() {
    let cfg = Config::from_json_str(r#"{"providers": {}, "default_reasoning": "medium"}"#).unwrap();
    let session = SessionConfig {
        reasoning: Some("high".into()),
        ..SessionConfig::default()
    };
    let eff = cfg.merge_session(&session);
    assert_eq!(
        eff.reasoning.as_deref(),
        Some("high"),
        "explicit SessionConfig.reasoning must win over Config.default_reasoning"
    );
}

#[test]
fn default_reasoning_absent_means_no_reasoning() {
    let cfg = Config::from_json_str(r#"{"providers": {}}"#).unwrap();
    let eff = cfg.merge_session(&SessionConfig::default());
    assert!(
        eff.reasoning.is_none(),
        "without Config.default_reasoning and without session override, \
         the effective config must carry no reasoning hint (legacy behaviour)"
    );
}

#[test]
fn shipped_naked_json_sets_default_reasoning_medium() {
    // Contract test against the file we actually ship: every provider
    // we use (kimi, moonshot, qwen, ali_cp, openrouter, fireworks,
    // anthropic) must receive `reasoning="medium"` for *all* sessions
    // — not just research turns. Without this, the bot's interactive
    // chat would silently downgrade to non-thinking despite the user
    // configuring kimi-for-coding everywhere.
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../naked.json");
    if !path.exists() {
        return;
    }
    let raw = std::fs::read_to_string(&path).unwrap();
    let cfg = Config::from_json_str(&raw).unwrap();
    assert_eq!(
        cfg.default_reasoning.as_deref(),
        Some("medium"),
        "naked.json must set default_reasoning=\"medium\" so qwen / \
         moonshot / kimi-for-coding all stream reasoning_content by \
         default — see the user request in 42189a5b about \"процесс \
         размышлений тоже выводим\""
    );
}

#[test]
fn research_reasoning_can_be_set() {
    let json = r#"{
        "providers": {},
        "research": {
            "reasoning": "medium"
        }
    }"#;
    let cfg = Config::from_json_str(json).unwrap();
    assert_eq!(cfg.research.reasoning.as_deref(), Some("medium"));
}

#[test]
fn chat_personas_absent_means_empty_map() {
    let cfg = Config::from_json_str(r#"{"providers": {}}"#).unwrap();
    assert!(
        cfg.chat_personas.is_empty(),
        "missing chat_personas in JSON must default to an empty map (legacy bots keep \
         behaving exactly like before — no persona, no overrides)"
    );
}

#[test]
fn chat_personas_parses_negative_chat_id_keys() {
    // serde_json deserialises object keys as strings; ours are i64 — make
    // sure the negative supergroup id (most common case for the Income
    // chat) round-trips through the map.
    let json = r#"{
        "providers": {},
        "chat_personas": {
            "-5084292206": {
                "name": "income",
                "workspace": "/home/operator/.naked/channels/income"
            }
        }
    }"#;
    let cfg = Config::from_json_str(json).unwrap();
    let persona = cfg.chat_personas.get(&-5084292206_i64).expect(
        "negative i64 chat_id key from JSON string must be deserialised into HashMap<i64,_>",
    );
    assert_eq!(persona.name, "income");
    assert_eq!(
        persona.workspace,
        std::path::PathBuf::from("/home/operator/.naked/channels/income")
    );
    assert!(
        !persona.allow_slash_commands,
        "allow_slash_commands defaults to false so personas behave as natural-language-only \
         chats unless explicitly opted in"
    );
}

#[test]
fn chat_persona_allow_slash_commands_opt_in() {
    let json = r#"{
        "providers": {},
        "chat_personas": {
            "12345": {
                "name": "ops",
                "workspace": "/tmp/ops",
                "allow_slash_commands": true
            }
        }
    }"#;
    let cfg = Config::from_json_str(json).unwrap();
    let persona = cfg.chat_personas.get(&12345_i64).unwrap();
    assert!(persona.allow_slash_commands);
}

#[test]
fn chat_persona_workspace_expanded_resolves_tilde() {
    // Mirrors `skill_roots_tilde_expanded` style: read the current
    // `$HOME` via `dirs_home()` instead of mutating process env (the
    // crate is `#![forbid(unsafe_code)]`, so `std::env::set_var`
    // is off-limits even inside tests).
    let home = dirs_home();
    let persona = ChatPersona {
        name: "x".into(),
        workspace: PathBuf::from("~/.naked/channels/x"),
        allow_slash_commands: false,
    };
    assert_eq!(persona.workspace_expanded(), home.join(".naked/channels/x"));
    // Absolute paths must pass through untouched.
    let abs = ChatPersona {
        name: "y".into(),
        workspace: PathBuf::from("/var/lib/y"),
        allow_slash_commands: false,
    };
    assert_eq!(abs.workspace_expanded(), PathBuf::from("/var/lib/y"));
}

#[test]
fn config_load_expands_tilde_in_persona_workspaces() {
    // `Config::load()` runs `expand_tilde` over every persona workspace
    // exactly once (mirroring the `skill_roots`/`agent_dirs` treatment).
    // We exercise the same expansion path via direct manipulation
    // instead of touching the on-disk loader.
    let json = r#"{
        "providers": {},
        "chat_personas": {
            "-1": {
                "name": "a",
                "workspace": "~/.naked/channels/a"
            },
            "2": {
                "name": "b",
                "workspace": "/var/lib/b"
            }
        }
    }"#;
    let mut cfg = Config::from_json_str(json).unwrap();
    for persona in cfg.chat_personas.values_mut() {
        persona.workspace = expand_tilde(&persona.workspace);
    }
    let home = dirs_home();
    assert_eq!(
        cfg.chat_personas[&-1_i64].workspace,
        home.join(".naked/channels/a")
    );
    assert_eq!(
        cfg.chat_personas[&2_i64].workspace,
        PathBuf::from("/var/lib/b")
    );
}

#[test]
fn gatekeeper_config_partial_override() {
    let json = r#"{
        "providers": {},
        "research": {
            "gatekeeper": {
                "min_excerpt_chars": 500,
                "require_listing_date": false
            }
        }
    }"#;
    let cfg = Config::from_json_str(json).unwrap();
    let gk = &cfg.research.gatekeeper;
    assert_eq!(gk.min_excerpt_chars, 500);
    assert!(!gk.require_listing_date);
    // other fields keep defaults
    assert_eq!(gk.max_rounds, 3);
    assert!(gk.require_source_content);
    assert!(gk.detect_semantic_duplicates);
}

#[test]
fn telegram_config_defaults() {
    let tc = TelegramConfig::default();
    assert!(tc.telegram_bot_token.is_none());
    assert!(tc.allowed_chat_ids.is_empty());
    assert!(!tc.tg_sender_attribution); // derive(Default) gives false; serde default gives true
}

#[test]
fn telegram_config_serde_roundtrip() {
    let tc = TelegramConfig {
        telegram_bot_token: Some("test-token".into()),
        allowed_chat_ids: vec![123, 456],
        tg_sender_attribution: false,
    };
    let json = serde_json::to_string(&tc).unwrap();
    let parsed: TelegramConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.telegram_bot_token, Some("test-token".into()));
    assert_eq!(parsed.allowed_chat_ids, vec![123, 456]);
    assert!(!parsed.tg_sender_attribution);
}

#[test]
fn config_with_flatten_telegram_parses() {
    let json = r#"{
        "telegram_bot_token": "bot123",
        "allowed_chat_ids": [100],
        "tg_sender_attribution": false,
        "workspace": "/tmp/test"
    }"#;
    let config: Config = serde_json::from_str(json).unwrap();
    assert_eq!(config.telegram.telegram_bot_token, Some("bot123".into()));
    assert_eq!(config.telegram.allowed_chat_ids, vec![100]);
    assert!(!config.telegram.tg_sender_attribution);
}

#[test]
fn provider_config_resolved_direct_key() {
    let pc = ProviderConfig {
        api_key: "sk-test".into(),
        ..Default::default()
    };
    let resolved = pc.resolved().unwrap();
    assert_eq!(resolved.api_key, "sk-test");
}

#[test]
fn provider_config_models_default_empty() {
    let pc = ProviderConfig::default();
    assert!(pc.models.is_empty());
    assert!(pc.provider_type.is_empty());
}

#[test]
fn config_default_has_sane_values() {
    let config = Config::default();
    // max_iterations defaults to 0 until loaded from JSON
    assert!(config.max_tokens > 0);
    assert!(config.providers.is_empty());
    assert!(config.telegram.allowed_chat_ids.is_empty());
}
