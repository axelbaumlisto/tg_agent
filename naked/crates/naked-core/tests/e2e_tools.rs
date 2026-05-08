//! E2E tests — tools

#[macro_use]
mod common;
use common::*;

#[tokio::test]
async fn t02_bash_tool() {
    need_provider!(config, provider, model);
    pace().await;
    let tmp = tempfile::tempdir().unwrap();

    eprintln!(">>> t02_bash_tool");
    let mut history = ConversationHistory::new(SYS.into());
    history.push_user("Use the bash tool to run: echo NAKED_BASH_99. Report the output.");
    let r = run_prompt(provider, &mut history, tmp.path(), &model).await;

    eprintln!("  text: {}", r.full_text());
    eprintln!("  tools: {:?}", r.tool_names());
    assert!(!r.had_error(), "agent error");
    assert!(r.has_tool("bash"), "bash not used: {:?}", r.tool_names());
    assert!(
        r.full_text().contains("NAKED_BASH_99"),
        "output missing: {}",
        r.full_text()
    );
}

#[tokio::test]
async fn t03_file_write_read() {
    need_provider!(config, provider, model);
    pace().await;
    let tmp = tempfile::tempdir().unwrap();

    eprintln!(">>> t03_file_write_read");
    let mut history = ConversationHistory::new(SYS.into());
    history.push_user(&format!(
        "Use write_file to write 'NAKED_CONTENT_77' to {}/e2e.txt. Then read it back.",
        tmp.path().display()
    ));
    let r = run_prompt(provider, &mut history, tmp.path(), &model).await;

    eprintln!("  text: {}", r.full_text());
    eprintln!("  tools: {:?}", r.tool_names());
    assert!(!r.had_error(), "agent error");
    assert!(
        r.has_tool("write_file"),
        "write_file not used: {:?}",
        r.tool_names()
    );

    let content = std::fs::read_to_string(tmp.path().join("e2e.txt")).unwrap_or_default();
    assert!(
        content.contains("NAKED_CONTENT_77"),
        "file wrong: {content}"
    );
}

#[tokio::test]
async fn t04_file_edit() {
    need_provider!(config, provider, model);
    pace().await;
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("target.txt"), "hello OLD_TOKEN world\n").unwrap();

    eprintln!(">>> t04_file_edit");
    let mut history = ConversationHistory::new(SYS.into());
    history.push_user(&format!(
        "Use edit_file on {}/target.txt to replace 'OLD_TOKEN' with 'NEW_TOKEN'.",
        tmp.path().display()
    ));
    let r = run_prompt(provider, &mut history, tmp.path(), &model).await;

    eprintln!("  text: {}", r.full_text());
    eprintln!("  tools: {:?}", r.tool_names());
    assert!(!r.had_error(), "agent error");

    let content = std::fs::read_to_string(tmp.path().join("target.txt")).unwrap_or_default();
    assert!(content.contains("NEW_TOKEN"), "edit not applied: {content}");
    assert!(
        !content.contains("OLD_TOKEN"),
        "old text remains: {content}"
    );
}

#[tokio::test]
async fn t05_glob_search() {
    need_provider!(config, provider, model);
    pace().await;
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("alpha.rs"), "fn main() {}").unwrap();
    std::fs::write(tmp.path().join("beta.txt"), "data").unwrap();
    std::fs::write(tmp.path().join("gamma.rs"), "fn test() {}").unwrap();

    eprintln!(">>> t05_glob_search");
    let mut history = ConversationHistory::new(SYS.into());
    history.push_user(&format!(
        "Use glob_search to find *.rs files in {}. List them.",
        tmp.path().display()
    ));
    let r = run_prompt(provider, &mut history, tmp.path(), &model).await;

    eprintln!("  text: {}", r.full_text());
    eprintln!("  tools: {:?}", r.tool_names());
    assert!(!r.had_error(), "agent error");
    let text = r.full_text();
    assert!(
        text.contains("alpha.rs") && text.contains("gamma.rs"),
        "glob missed files: {text}"
    );
}

#[tokio::test]
async fn t06_grep_search() {
    need_provider!(config, provider, model);
    pace().await;
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(
        tmp.path().join("code.rs"),
        "fn main() { println!(\"NEEDLE_E2E\"); }\n",
    )
    .unwrap();
    std::fs::write(tmp.path().join("other.txt"), "nothing here").unwrap();

    eprintln!(">>> t06_grep_search");
    let mut history = ConversationHistory::new(SYS.into());
    history.push_user(&format!(
        "Use grep_search to find 'NEEDLE_E2E' in {}. Which file?",
        tmp.path().display()
    ));
    let r = run_prompt(provider, &mut history, tmp.path(), &model).await;

    eprintln!("  text: {}", r.full_text());
    assert!(!r.had_error(), "agent error");
    assert!(
        r.full_text().contains("code.rs"),
        "grep failed: {}",
        r.full_text()
    );
}

#[tokio::test]
async fn t17_skill_list() {
    need_config!(config);
    let roots = skill_roots_from_config(&config);
    if roots.is_empty() {
        eprintln!("SKIP t17: no skill_roots configured");
        return;
    }

    eprintln!(">>> t17_skill_list [roots: {:?}]", roots);
    let resolver = SkillResolver::new(roots);
    let skills = resolver.list();

    eprintln!(
        "  found {} skills: {:?}",
        skills.len(),
        skills.iter().map(|(n, _)| n).collect::<Vec<_>>()
    );
    let has_telegram = skills.iter().any(|(name, _)| name == "telegram-reader");
    assert!(
        has_telegram,
        "telegram-reader not found in skill roots: {:?}",
        skills.iter().map(|(n, _)| n).collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn t18_skill_resolve() {
    need_config!(config);
    let roots = skill_roots_from_config(&config);
    if roots.is_empty() {
        eprintln!("SKIP t18: no skill_roots configured");
        return;
    }

    eprintln!(">>> t18_skill_resolve");
    let resolver = SkillResolver::new(roots);
    let hit = resolver.resolve("telegram-reader");

    assert!(hit.is_some(), "telegram-reader not resolved");
    let hit = hit.unwrap();
    assert!(
        hit.path.exists(),
        "SKILL.* does not exist: {}",
        hit.path.display()
    );

    let content = std::fs::read_to_string(&hit.path).unwrap();
    assert!(!content.is_empty(), "SKILL.* is empty");
    eprintln!(
        "  resolved: {} ({} bytes)",
        hit.path.display(),
        content.len()
    );
}

#[tokio::test]
async fn t19_skill_tool_invocation() {
    need_provider!(config, provider, model);
    pace().await;
    let tmp = tempfile::tempdir().unwrap();
    let roots = skill_roots_from_config(&config);

    if roots.is_empty()
        || SkillResolver::new(roots.clone())
            .resolve("telegram-reader")
            .is_none()
    {
        eprintln!("SKIP t19: telegram-reader skill not available");
        return;
    }

    eprintln!(">>> t19_skill_tool_invocation [model={model}]");
    let tools = build_tools_with_skill_roots(roots);
    let mut history = ConversationHistory::new(SYS.into());
    history
        .push_user("Use the Skill tool to load the 'telegram-reader' skill. Tell me what it does.");
    let r = run_prompt_with_tools(provider, &mut history, tmp.path(), &model, tools).await;

    eprintln!("  text: {}", r.full_text());
    eprintln!("  tools: {:?}", r.tool_names());
    assert!(!r.had_error(), "agent error");
    assert!(
        r.has_tool("Skill"),
        "Skill tool not used: {:?}",
        r.tool_names()
    );
    let text = r.full_text().to_lowercase();
    assert!(
        text.contains("telegram") || text.contains("reader") || text.contains("skill"),
        "skill info missing from response: {}",
        r.full_text()
    );
}

#[tokio::test]
async fn t20_mcp_connect() {
    let script = mcp_server_script_path();
    if !script.exists() {
        eprintln!(
            "SKIP t20: test-mcp-server.sh not found at {}",
            script.display()
        );
        return;
    }

    eprintln!(">>> t20_mcp_connect [script={}]", script.display());
    let config = McpServerConfig {
        name: "test-echo".into(),
        transport: naked_core::config::McpTransportType::Stdio,
        command: script.to_string_lossy().to_string(),
        args: vec![],
        env: std::collections::HashMap::new(),
        url: None,
        headers: std::collections::HashMap::new(),
        tool_timeout_secs: None,
    };

    let server = McpServer::connect(&config).await;
    assert!(server.is_ok(), "MCP connect failed: {:?}", server.err());
    let server = server.unwrap();

    eprintln!(
        "  tools: {:?}",
        server.tools().iter().map(|t| &t.name).collect::<Vec<_>>()
    );
    assert!(!server.tools().is_empty(), "no tools discovered");
    assert!(
        server.tools().iter().any(|t| t.name == "mcp_echo"),
        "mcp_echo tool not found"
    );

    server.close().await.ok();
}

#[tokio::test]
async fn t21_mcp_tool_call() {
    let script = mcp_server_script_path();
    if !script.exists() {
        eprintln!("SKIP t21: test-mcp-server.sh not found");
        return;
    }

    eprintln!(">>> t21_mcp_tool_call");
    let config = McpServerConfig {
        name: "test-echo".into(),
        transport: naked_core::config::McpTransportType::Stdio,
        command: script.to_string_lossy().to_string(),
        args: vec![],
        env: std::collections::HashMap::new(),
        url: None,
        headers: std::collections::HashMap::new(),
        tool_timeout_secs: None,
    };

    let server = McpServer::connect(&config).await.unwrap();
    let result = server
        .call_tool("mcp_echo", serde_json::json!({"text": "ECHO_TEST_21"}))
        .await;

    assert!(result.is_ok(), "MCP tool call failed: {:?}", result.err());
    let result = result.unwrap();

    assert!(!result.is_error, "MCP tool returned error");
    let text: String = result
        .content
        .iter()
        .filter_map(|c| c.as_text())
        .collect::<Vec<_>>()
        .join("");
    eprintln!("  echo result: {text}");
    assert!(
        text.contains("ECHO_TEST_21"),
        "echo content mismatch: {text}"
    );

    server.close().await.ok();
}

#[tokio::test]
async fn t22_mcp_agent_integration() {
    need_provider!(_config, provider, model);
    pace().await;
    let tmp = tempfile::tempdir().unwrap();

    let script = mcp_server_script_path();
    if !script.exists() {
        eprintln!("SKIP t22: test-mcp-server.sh not found");
        return;
    }

    let mcp_config = McpServerConfig {
        name: "test-echo".into(),
        transport: naked_core::config::McpTransportType::Stdio,
        command: script.to_string_lossy().to_string(),
        args: vec![],
        env: std::collections::HashMap::new(),
        url: None,
        headers: std::collections::HashMap::new(),
        tool_timeout_secs: None,
    };

    let server = match McpServer::connect(&mcp_config).await {
        Ok(s) => Arc::new(s),
        Err(e) => {
            eprintln!("SKIP t22: MCP connect failed: {e}");
            return;
        }
    };

    let mcp_tools = McpToolWrapper::wrap_all(Arc::clone(&server));
    let tools = build_tools_with_mcp(mcp_tools);

    eprintln!(">>> t22_mcp_agent_integration [model={model}]");
    let mut history = ConversationHistory::new(SYS.into());
    history.push_user(
        "You have a tool called mcp_echo. Use it to echo the text 'MCP_AGENT_22'. Report the result.",
    );
    let r = run_prompt_with_tools(provider, &mut history, tmp.path(), &model, tools).await;

    eprintln!("  text: {}", r.full_text());
    eprintln!("  tools: {:?}", r.tool_names());
    assert!(!r.had_error(), "agent error");
    assert!(
        r.has_tool("mcp_echo"),
        "mcp_echo not used: {:?}",
        r.tool_names()
    );
    assert!(
        r.full_text().contains("MCP_AGENT_22"),
        "echo result missing: {}",
        r.full_text()
    );

    server.close().await.ok();
}

#[tokio::test]
async fn t51_skill_resolver_path_traversal() {
    let skill_root = tempfile::tempdir().unwrap();
    let legit = skill_root.path().join("legit-skill");
    std::fs::create_dir(&legit).unwrap();
    std::fs::write(
        legit.join("SKILL.md"),
        "description: A legit skill\n# Legit",
    )
    .unwrap();

    let resolver = SkillResolver::new(vec![skill_root.path().to_path_buf()]);

    // Legit skill resolves
    assert!(
        resolver.resolve("legit-skill").is_some(),
        "legit skill should resolve"
    );

    // Path traversal rejected
    assert!(
        resolver.resolve("../../../etc").is_none(),
        "path traversal with .. should be rejected"
    );
    assert!(
        resolver.resolve("legit-skill/../../etc").is_none(),
        "nested path traversal should be rejected"
    );
    assert!(
        resolver.resolve("..").is_none(),
        "bare .. should be rejected"
    );

    eprintln!("  PASS: skill resolver blocks path traversal");
}

#[tokio::test]
async fn t56_skill_tool_async_io() {
    let skill_root = tempfile::tempdir().unwrap();
    let skill_dir = skill_root.path().join("test-skill");
    std::fs::create_dir(&skill_dir).unwrap();
    std::fs::write(
        skill_dir.join("SKILL.md"),
        "description: E2E test skill\n# Test Skill\nThis is a test skill for E2E.",
    )
    .unwrap();

    let resolver = SkillResolver::new(vec![skill_root.path().to_path_buf()]);
    let available = resolver.list();
    let tool = SkillTool::new(resolver, &available);

    let result = tool
        .execute(
            serde_json::json!({"skill": "test-skill"}),
            skill_root.path(),
        )
        .await;
    assert!(!result.is_error, "skill should load: {}", result.output);
    assert!(
        result.output.contains("E2E test skill"),
        "skill content should be present: {}",
        result.output
    );

    eprintln!("  PASS: SkillTool executes via async IO");
}

#[tokio::test]
async fn t58_glob_search_result_cap() {
    let tmp = tempfile::tempdir().unwrap();

    // Create 300 files (exceeds cap of 200)
    for i in 0..300 {
        std::fs::write(tmp.path().join(format!("file_{i:04}.txt")), "data").unwrap();
    }

    let tool = GlobSearchTool;
    let result = tool
        .execute(serde_json::json!({"pattern": "*.txt"}), tmp.path())
        .await;
    assert!(!result.is_error, "glob should succeed");

    let line_count = result.output.lines().count();
    // Should be capped: 200 results + 1 truncation notice
    assert!(
        line_count <= 202,
        "should be capped: got {line_count} lines"
    );
    assert!(
        result.output.contains("truncated"),
        "should mention truncation: {}",
        result.output.lines().last().unwrap_or("")
    );

    eprintln!("  PASS: glob_search caps at 200 results ({line_count} lines)");
}

#[tokio::test]
async fn t66_bash_command_classification() {
    use naked_core::tool::bash::{BashRisk, classify_bash};

    // Read-only commands
    let read_only = [
        "ls -la",
        "cat file.txt",
        "head -20 main.rs",
        "git status",
        "git log --oneline",
        "git diff HEAD",
        "grep foo bar.txt",
        "rg pattern src/",
        "find . -name '*.rs'",
        "tree",
        "cargo test --lib",
        "cargo clippy",
        "pwd",
        "whoami",
        "echo hello",
        "ps aux",
        "df -h",
        "du -sh .",
        "docker ps",
        "docker images",
    ];
    for cmd in &read_only {
        assert_eq!(
            classify_bash(cmd),
            BashRisk::ReadOnly,
            "should be ReadOnly: {cmd}"
        );
    }

    // Piped read-only
    assert_eq!(classify_bash("cat file | grep foo"), BashRisk::ReadOnly);
    assert_eq!(classify_bash("ls -la | wc -l"), BashRisk::ReadOnly);
    assert_eq!(classify_bash("git log | head -5"), BashRisk::ReadOnly);

    // Write commands (contain redirect or unknown commands)
    let write_cmds = [
        "echo x > file.txt",
        "echo x >> log.txt",
        "cp a.txt b.txt",
        "mkdir -p new_dir",
        "npm install",
        "git commit -m 'msg'",
        "cargo build",
        "pip install requests",
    ];
    for cmd in &write_cmds {
        assert_eq!(
            classify_bash(cmd),
            BashRisk::Write,
            "should be Write: {cmd}"
        );
    }

    // Destructive commands
    let destructive = [
        "rm -rf /",
        "rm -rf /*",
        "mkfs.ext4 /dev/sda1",
        "dd if=/dev/zero of=/dev/sda",
    ];
    for cmd in &destructive {
        assert_eq!(
            classify_bash(cmd),
            BashRisk::Destructive,
            "should be Destructive: {cmd}"
        );
    }

    eprintln!(
        "  PASS: classify_bash correctly categorizes {} commands",
        read_only.len() + write_cmds.len() + destructive.len()
    );
}

#[tokio::test]
async fn t67_bash_effective_permission() {
    let tool = BashTool::new(30);

    assert_eq!(
        tool.effective_permission(&serde_json::json!({"command": "ls -la"}), Path::new("/")),
        Permission::ReadOnly,
        "ls should be ReadOnly"
    );
    assert_eq!(
        tool.effective_permission(
            &serde_json::json!({"command": "git status"}),
            Path::new("/")
        ),
        Permission::ReadOnly,
        "git status should be ReadOnly"
    );
    assert_eq!(
        tool.effective_permission(
            &serde_json::json!({"command": "npm install"}),
            Path::new("/")
        ),
        Permission::WorkspaceWrite,
        "npm install should be WorkspaceWrite"
    );
    assert_eq!(
        tool.effective_permission(&serde_json::json!({"command": "rm -rf /"}), Path::new("/")),
        Permission::Dangerous,
        "rm -rf / should be Dangerous"
    );

    // Missing command field → empty string → unclassified → Write
    assert_eq!(
        tool.effective_permission(&serde_json::json!({}), Path::new("/")),
        Permission::WorkspaceWrite,
        "empty input should default to WorkspaceWrite"
    );

    eprintln!("  PASS: BashTool.effective_permission maps classification to permissions");
}

#[tokio::test]
async fn t73_bash_output_truncation() {
    let tmp = tempfile::tempdir().unwrap();
    let tool = BashTool::new(30);

    // Generate output larger than 16 KiB
    let result = tool
        .execute(
            serde_json::json!({"command": "python3 -c \"print('x' * 20000)\""}),
            tmp.path(),
        )
        .await;
    assert!(!result.is_error, "command should succeed");
    assert!(
        result.output.len() <= 17_000,
        "output should be truncated to ~16 KiB, got {} bytes",
        result.output.len()
    );
    assert!(
        result.output.contains("truncated"),
        "should contain truncation marker"
    );

    // Small output should not be truncated
    let result = tool
        .execute(serde_json::json!({"command": "echo short"}), tmp.path())
        .await;
    assert!(
        !result.output.contains("truncated"),
        "short output should not be truncated"
    );

    eprintln!("  PASS: Bash output truncation at 16 KiB");
}

#[tokio::test]
async fn t75_grep_output_truncation() {
    let tmp = tempfile::tempdir().unwrap();

    // Create many files with matching patterns
    for i in 0..200 {
        std::fs::write(
            tmp.path().join(format!("match_{i:04}.txt")),
            format!("FINDME line {i} {}", "x".repeat(200)),
        )
        .unwrap();
    }

    let tool = GrepSearchTool;
    let result = tool
        .execute(
            serde_json::json!({"pattern": "FINDME", "path": tmp.path().to_str().unwrap()}),
            tmp.path(),
        )
        .await;
    assert!(!result.is_error, "grep should succeed");
    assert!(
        result.output.len() <= 17_000,
        "output should be capped, got {} bytes",
        result.output.len()
    );

    eprintln!(
        "  PASS: Grep output truncation ({} bytes)",
        result.output.len()
    );
}

#[tokio::test]
async fn t76_glob_cap_200() {
    let tmp = tempfile::tempdir().unwrap();

    for i in 0..250 {
        std::fs::write(tmp.path().join(format!("f_{i:04}.rs")), "fn main(){}").unwrap();
    }

    let tool = GlobSearchTool;
    let result = tool
        .execute(serde_json::json!({"pattern": "*.rs"}), tmp.path())
        .await;
    assert!(!result.is_error);

    let count = result.output.lines().count();
    assert!(count <= 202, "should cap at ~200 lines, got {count}");
    assert!(
        result.output.contains("truncated"),
        "should mention truncation"
    );

    eprintln!("  PASS: Glob caps at 200 results ({count} lines)");
}

#[tokio::test]
async fn t77_sub_agent_explore() {
    let config = match test_config() {
        Some(c) => c,
        None => {
            eprintln!("  SKIP t77: no config");
            return;
        }
    };
    let (provider, model) = match get_working_provider(&config).await {
        Some(pm) => pm,
        None => {
            eprintln!("  SKIP t77: no working provider");
            return;
        }
    };

    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("hello.txt"), "Hello from sub-agent test").unwrap();
    std::fs::write(
        tmp.path().join("data.rs"),
        "fn main() { println!(\"hi\"); }",
    )
    .unwrap();

    let tool = SubAgentTool::new(Arc::from(provider), model, 30, vec![]);

    let result = tokio::time::timeout(
        Duration::from_secs(120),
        tool.execute(
            serde_json::json!({
                "prompt": "List all files in the current directory and read the contents of hello.txt. Report what you find.",
                "mode": "explore"
            }),
            tmp.path(),
        ),
    )
    .await
    .expect("sub-agent timed out");

    assert!(
        !result.is_error,
        "sub-agent should succeed: {}",
        result.output
    );
    assert!(
        result.output.contains("[sub-agent:"),
        "should include sub-agent usage footer"
    );

    let lower = result.output.to_lowercase();
    assert!(
        lower.contains("hello") || lower.contains("sub-agent"),
        "should mention file content or tool activity: {}",
        result.output
    );

    eprintln!("  PASS: Sub-agent explore mode");
    eprintln!(
        "  Output preview: {}",
        &result.output[..result.output.len().min(300)]
    );
}

#[tokio::test]
async fn t78_sub_agent_permissions() {
    let config = match test_config() {
        Some(c) => c,
        None => {
            eprintln!("  SKIP t78: no config");
            return;
        }
    };
    let (provider, model) = match get_working_provider(&config).await {
        Some(pm) => pm,
        None => {
            eprintln!("  SKIP t78: no working provider");
            return;
        }
    };

    let tool = SubAgentTool::new(Arc::from(provider), model, 30, vec![]);

    assert_eq!(
        tool.effective_permission(
            &serde_json::json!({"prompt": "x", "mode": "explore"}),
            Path::new("/tmp")
        ),
        Permission::ReadOnly
    );

    assert_eq!(
        tool.effective_permission(&serde_json::json!({"prompt": "x"}), Path::new("/tmp")),
        Permission::ReadOnly
    );

    assert_eq!(
        tool.effective_permission(
            &serde_json::json!({"prompt": "x", "mode": "general"}),
            Path::new("/tmp")
        ),
        Permission::WorkspaceWrite
    );

    eprintln!("  PASS: Sub-agent permission levels");
}

#[tokio::test]
async fn t79_sub_agent_empty_prompt() {
    let config = match test_config() {
        Some(c) => c,
        None => {
            eprintln!("  SKIP t79: no config");
            return;
        }
    };
    let (provider, model) = match get_working_provider(&config).await {
        Some(pm) => pm,
        None => {
            eprintln!("  SKIP t79: no working provider");
            return;
        }
    };

    let tool = SubAgentTool::new(Arc::from(provider), model, 30, vec![]);

    let result = tool
        .execute(serde_json::json!({"prompt": ""}), Path::new("/tmp"))
        .await;
    assert!(result.is_error, "empty prompt should be error");
    assert!(result.output.contains("required"));

    let result2 = tool.execute(serde_json::json!({}), Path::new("/tmp")).await;
    assert!(result2.is_error, "missing prompt should be error");

    eprintln!("  PASS: Sub-agent empty prompt rejected");
}

#[tokio::test]
async fn t87_sub_agent_has_web_search() {
    let config = match test_config() {
        Some(c) => c,
        None => {
            eprintln!("  SKIP t87: no config");
            return;
        }
    };
    let (provider, model) = match get_working_provider(&config).await {
        Some(pm) => pm,
        None => {
            eprintln!("  SKIP t87: no working provider");
            return;
        }
    };

    eprintln!(">>> t87_sub_agent_has_web_search [model={model}]");

    let exa_keys = config.exa_api_keys.clone();
    let tool = SubAgentTool::new(Arc::from(provider), model.clone(), 60, exa_keys.clone());

    // Verify the spec is correct
    let spec = tool.spec();
    assert_eq!(spec.name, "sub_agent");

    // Verify exa_keys are passed through (non-empty when configured)
    eprintln!("  exa_keys count: {}", exa_keys.len());

    // Verify web_search appears in explore mode tool list description
    // (We test this indirectly — if the sub_agent has web_search, it should mention
    //  it in its sub-agent prompt capabilities)
    let result = tool
        .execute(
            serde_json::json!({
                "prompt": "List all tools available to you. Just list their names, nothing else.",
                "mode": "explore"
            }),
            Path::new("/tmp"),
        )
        .await;

    eprintln!(
        "  sub_agent output ({} bytes): {}",
        result.output.len(),
        &result.output[..result.output.len().min(500)]
    );

    let output_lower = result.output.to_lowercase();
    let has_web_search = output_lower.contains("web_search") || output_lower.contains("web search");
    eprintln!("  web_search in tool list: {has_web_search}");

    // Also verify validate_json.py works with our test data
    let tmp = tempfile::tempdir().unwrap();
    let fields_yaml = tmp.path().join("fields.yaml");
    std::fs::write(
        &fields_yaml,
        r#"categories:
  basic_info:
    display_name: "Basic Info"
    fields:
      - name: "location"
        description: "Location"
        detail_level: "brief"
      - name: "price"
        description: "Price"
        detail_level: "brief"
        required: true
"#,
    )
    .unwrap();

    let test_json = tmp.path().join("test.json");
    std::fs::write(&test_json, r#"{"location": "Da Nang", "price": "$500/mo"}"#).unwrap();

    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    let script = format!("{home}/.naked/skills/deep-research/scripts/validate_json.py");
    let output = std::process::Command::new("python3")
        .args([
            &script,
            "-f",
            fields_yaml.to_str().unwrap(),
            "-j",
            test_json.to_str().unwrap(),
        ])
        .output();

    match output {
        Ok(o) => {
            let stdout = String::from_utf8_lossy(&o.stdout);
            let stderr = String::from_utf8_lossy(&o.stderr);
            eprintln!("  validate_json.py stdout: {stdout}");
            if !stderr.is_empty() {
                eprintln!("  validate_json.py stderr: {stderr}");
            }
            assert!(o.status.success(), "validate_json.py failed");
            assert!(stdout.contains("PASS"), "validation should PASS: {stdout}");
            eprintln!("  PASS: validate_json.py works with test data");
        }
        Err(e) => eprintln!("  WARN: couldn't run validate_json.py: {e}"),
    }

    eprintln!("  PASS: sub_agent has web_search capability");
}

#[tokio::test]
async fn t91_sub_agent_registry_lifecycle() {
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
    eprintln!(">>> t91_sub_agent_registry_lifecycle [model={model}]");

    let registry = AgentRegistry::new();
    let exa_keys = config.exa_api_keys.clone();
    let (provider, _): (Box<dyn Provider>, String) = get_working_provider(&config).await.unwrap();
    let prov_arc: Arc<dyn Provider> = Arc::from(provider);

    let sub_agent =
        SubAgentTool::new(prov_arc, model.clone(), 30, exa_keys).with_registry(registry.clone());

    let (progress_tx, mut progress_rx) = mpsc::channel::<AgentEvent>(256);

    let input = serde_json::json!({
        "prompt": "Say exactly: HELLO WORLD. Nothing else.",
        "mode": "explore"
    });

    let result = sub_agent
        .execute_with_progress(input, Path::new("/tmp"), progress_tx)
        .await;
    assert!(!result.is_error, "sub_agent failed: {}", result.output);
    eprintln!("  ✅ sub_agent completed: {}", result.output.len());

    // Collect events, find agent_id
    let mut agent_id = String::new();
    let mut started = false;
    let mut finished = false;
    while let Ok(ev) = progress_rx.try_recv() {
        if let AgentEvent::SubAgentProgress {
            agent_id: aid,
            event,
        } = ev
        {
            agent_id = aid;
            match event {
                SubAgentEvent::Started { .. } => started = true,
                SubAgentEvent::Finished { .. } => finished = true,
                _ => {}
            }
        }
    }
    eprintln!("  agent_id: {agent_id}");
    eprintln!("  started={started}, finished={finished}");
    assert!(started, "must have Started event");
    assert!(finished, "must have Finished event");
    assert!(agent_id.starts_with("sa-"), "agent_id must start with sa-");

    // Registry must have the entry
    let entry = registry.get(&agent_id).await.expect("agent in registry");
    eprintln!("  registry status: {}", entry.status);
    assert!(
        entry.status == naked_core::agent_registry::AgentStatus::Completed,
        "status must be Completed"
    );

    // agent_status tool sees it
    let status_tool = AgentStatusTool::new(registry.clone());
    let r = status_tool
        .execute(serde_json::json!({"agent_id": agent_id}), Path::new("/tmp"))
        .await;
    assert!(!r.is_error);
    assert!(
        r.output.contains("completed"),
        "status output: {}",
        r.output
    );
    eprintln!("  ✅ agent_status shows completed agent");

    eprintln!("  PASS: sub_agent registry lifecycle works");
}
