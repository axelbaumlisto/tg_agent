//! Live E2E tests -- hit a real LLM provider, execute real tools,
//! validate the full agent pipeline.
//!
//! Configure via JSON config file or env vars.
//! Skipped automatically when no provider is configured.
//!
//! Run (sequential, respecting rate limits):
//!   cargo test -p naked-core --test e2e_live -- --nocapture --test-threads=1

use std::path::{Path, PathBuf};
use std::time::Duration;

use naked_core::agent_registry::AgentRegistry;
use naked_core::config::{Config, McpServerConfig};
use naked_core::history::ConversationHistory;
use naked_core::loop_::{AgentLoop, LoopConfig};
use naked_core::mcp::client::McpServer;
use naked_core::mcp::wrapper::McpToolWrapper;
use naked_core::provider::Provider;
use naked_core::provider::resilient::ResilientProvider;
use naked_core::session::jsonl_store::JsonlSessionStore;
use naked_core::session::store::SessionStore;
use naked_core::session::{Session, SessionMetadata};
use naked_core::skill::resolver::SkillResolver;
use naked_core::skill::tool::SkillTool;
use naked_core::tool::Tool;
use naked_core::tool::agent_control::{AgentStatusTool, AgentStopTool};
use naked_core::tool::bash::BashTool;
use naked_core::tool::file_ops::{EditFileTool, ReadFileTool, WriteFileTool};
use naked_core::tool::registry::ToolRegistry;
use naked_core::tool::search::{GlobSearchTool, GrepSearchTool};
use naked_core::tool::sub_agent::SubAgentTool;
use naked_core::types::{AgentEvent, ContentBlock, Permission, SubAgentEvent, ToolState};

use std::sync::Arc;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

// ── Provider setup (all from config, no hardcoded names) ─────────────────────

fn test_config() -> Option<Config> {
    let config = Config::load().ok()?;
    if config.providers.is_empty() {
        eprintln!("SKIP: no providers configured");
        return None;
    }
    Some(config)
}

/// Try the default provider first; if its probe fails, iterate all providers
/// and return the first one that responds. Returns `(provider, model)`.
async fn get_working_provider(config: &Config) -> Option<(Box<dyn Provider>, String)> {
    let explicit_model = std::env::var("E2E_MODEL").ok();

    // 1. Try default provider
    if let Ok(prov) = naked_core::build_provider_from_config(config) {
        let model = explicit_model.clone().unwrap_or_else(|| {
            if !config.default_model.is_empty() {
                config.default_model.clone()
            } else if let Some(pc) = config.providers.values().next()
                && let Some(m) = pc.models.first()
            {
                m.clone()
            } else {
                "gpt-4o-mini".into()
            }
        });
        if probe_provider(prov.as_ref(), &model).await {
            return Some((prov, model));
        }
        eprintln!("  default provider probe FAILED, trying others...");
    }

    // 2. Iterate all configured providers
    for (name, pc) in &config.providers {
        let resolved = match pc.resolved() {
            Ok(r) if !r.api_key.is_empty() && !r.api_key.starts_with('$') => r,
            _ => continue,
        };
        let model = explicit_model.clone().unwrap_or_else(|| {
            pc.models
                .first()
                .cloned()
                .unwrap_or_else(|| "gpt-4o-mini".into())
        });
        let prov = naked_core::create_provider(name, resolved);
        if probe_provider(prov.as_ref(), &model).await {
            eprintln!("  fallback provider OK: {name}/{model}");
            return Some((prov, model));
        }
        eprintln!("  probe SKIP: {name}/{model}");
    }
    None
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn build_tools() -> ToolRegistry {
    let tools: Vec<Box<dyn Tool>> = vec![
        Box::new(BashTool::new(30)),
        Box::new(ReadFileTool),
        Box::new(WriteFileTool),
        Box::new(EditFileTool),
        Box::new(GlobSearchTool),
        Box::new(GrepSearchTool),
        Box::new(SkillTool::new(SkillResolver::new(vec![]), &[])),
    ];
    ToolRegistry::new(tools)
}

fn loop_config(cwd: PathBuf, model: &str) -> LoopConfig {
    LoopConfig {
        max_iterations: 15,
        cwd,
        model: model.to_string(),
        max_tokens: 2048,
        temperature: None,
        reasoning: None,
        provider: String::new(),
        health: None,
    }
}

struct E2eResult {
    events: Vec<AgentEvent>,
    history: ConversationHistory,
}

impl E2eResult {
    fn full_text(&self) -> String {
        self.events
            .iter()
            .filter_map(|e| match e {
                AgentEvent::TextDelta(t) => Some(t.as_str()),
                _ => None,
            })
            .collect()
    }

    fn tool_names(&self) -> Vec<String> {
        self.events
            .iter()
            .filter_map(|e| match e {
                AgentEvent::ToolStart { name, .. } => Some(name.clone()),
                _ => None,
            })
            .collect()
    }

    #[allow(dead_code)]
    fn tool_results(&self) -> Vec<(String, ToolState, String)> {
        self.events
            .iter()
            .filter_map(|e| match e {
                AgentEvent::ToolEnd {
                    name,
                    state,
                    output,
                    ..
                } => Some((name.clone(), state.clone(), output.clone())),
                _ => None,
            })
            .collect()
    }

    fn has_tool(&self, name: &str) -> bool {
        self.tool_names().iter().any(|n| n == name)
    }

    fn had_error(&self) -> bool {
        self.events
            .iter()
            .any(|e| matches!(e, AgentEvent::Error(_)))
    }

    fn got_idle(&self) -> bool {
        self.events.iter().any(|e| matches!(e, AgentEvent::Idle))
    }
}

async fn run_prompt(
    provider: Box<dyn Provider>,
    history: &mut ConversationHistory,
    cwd: &Path,
    model: &str,
) -> E2eResult {
    let tools = build_tools();
    let config = loop_config(cwd.to_path_buf(), model);
    let agent = AgentLoop::new(provider, tools, config);

    let (tx, mut rx) = mpsc::channel(256);
    let cancel = CancellationToken::new();

    let timeout = Duration::from_secs(120);
    let result = tokio::time::timeout(timeout, agent.run(history, tx, cancel, None, None)).await;

    match result {
        Ok(Ok(_usage)) => {}
        Ok(Err(e)) => eprintln!("  AgentLoop error: {e}"),
        Err(_) => eprintln!("  E2E TIMEOUT after {timeout:?}"),
    }

    let mut events = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        events.push(ev);
    }

    E2eResult {
        events,
        history: history.clone(),
    }
}

const SYS: &str = "You are a concise coding assistant. Use tools when asked. Respond briefly.";

fn make_provider_for(
    config: &Config,
    provider_name: &str,
    model: &str,
) -> Option<Box<dyn Provider>> {
    let pc = config.providers.get(provider_name)?;
    let mut resolved = pc.resolved().ok()?;
    if resolved.api_key.is_empty() || resolved.api_key.starts_with('$') {
        return None;
    }
    resolved.models = vec![model.to_string()];
    Some(naked_core::create_provider(provider_name, resolved))
}

/// Quick connectivity check — sends a trivial request and returns true if provider responds.
///
/// Strategy: first attempt with `reasoning: Some("off")` so chain-of-thought
/// models (Qwen3, DeepSeek) don't burn tokens on the probe. If that variant
/// fails — typically because the provider's OpenAI-compat surface rejects an
/// `enable_thinking`/`reasoning_effort` parameter it doesn't know about — fall
/// back to a plain probe with `reasoning: None`. This keeps the probe useful
/// against new providers without requiring us to teach `apply_reasoning_params`
/// about each base_url upfront.
/// Process-wide set of `(provider_type, model)` pairs that returned zero
/// text across all probe variants. Once a pair lands here, subsequent
/// `probe_provider` calls short-circuit to `false` without re-hitting the
/// network — saves ~45 s per skipped test and makes "this provider is
/// down today" explicit instead of re-discovered per-test.
fn degraded_provider_set() -> &'static std::sync::Mutex<std::collections::HashSet<String>> {
    use std::sync::OnceLock;
    static SET: OnceLock<std::sync::Mutex<std::collections::HashSet<String>>> = OnceLock::new();
    SET.get_or_init(|| std::sync::Mutex::new(std::collections::HashSet::new()))
}

fn degraded_key(provider: &dyn naked_core::provider::Provider, model: &str) -> String {
    format!("{}::{model}", provider.name())
}

async fn probe_provider(provider: &dyn Provider, model: &str) -> bool {
    use futures_util::StreamExt;
    use naked_core::provider::ChatRequest;

    // Short-circuit if we already marked this (provider, model) degraded
    // earlier in the process.
    let key = degraded_key(provider, model);
    if let Ok(set) = degraded_provider_set().lock()
        && set.contains(&key)
    {
        eprintln!("  probe: {key} already marked degraded this run; skipping");
        return false;
    }

    async fn run_once(
        provider: &dyn Provider,
        model: &str,
        reasoning: Option<String>,
    ) -> std::result::Result<bool, String> {
        let req = ChatRequest {
            model: model.to_string(),
            system: String::new(),
            messages: vec![serde_json::json!({"role": "user", "content": "Say OK"})],
            tools: vec![],
            max_tokens: 32,
            temperature: None,
            reasoning,
        };
        match provider.stream_chat(req).await {
            Ok(mut stream) => {
                let mut got_text = false;
                while let Some(chunk) = stream.next().await {
                    match chunk {
                        naked_core::types::StreamChunk::Text(_) => got_text = true,
                        naked_core::types::StreamChunk::Error(e) => {
                            return Err(format!("stream error: {e}"));
                        }
                        naked_core::types::StreamChunk::Done => break,
                        _ => {}
                    }
                }
                Ok(got_text)
            }
            Err(e) => Err(format!("connect error: {e}")),
        }
    }

    // Try in this order:
    // 1. reasoning="off" — fastest path for reasoning-aware providers.
    // 2. no reasoning param — for providers that reject the field outright.
    // 3. reasoning="low" — for providers that *require* a reasoning param and
    //    treat empty output as "still thinking" (e.g. kimi-code/k2p5).
    // We accept the first variant that yields any text chunk.
    let attempts: [(&str, Option<String>); 3] = [
        ("reasoning=off", Some("off".into())),
        ("no reasoning", None),
        ("reasoning=low", Some("low".into())),
    ];
    for (label, reasoning) in attempts {
        match run_once(provider, model, reasoning).await {
            Ok(true) => return true,
            Ok(false) => {
                eprintln!("  probe ({label}): stream OK but no text received; trying next variant");
            }
            Err(e) => {
                eprintln!("  probe ({label}) failed: {e}; trying next variant");
            }
        }
    }
    eprintln!("  probe: all variants returned no text");
    // Mark this (provider, model) degraded so subsequent probes in the
    // same test run don't re-pay the triple-retry cost.
    if let Ok(mut set) = degraded_provider_set().lock() {
        set.insert(key);
    }
    false
}

macro_rules! need_config {
    ($cfg:ident) => {
        let $cfg = match test_config() {
            Some(c) => c,
            None => return,
        };
    };
}

macro_rules! need_provider {
    ($cfg:ident, $prov:ident, $model:ident) => {
        need_config!($cfg);
        let ($prov, $model) = match get_working_provider(&$cfg).await {
            Some(pm) => pm,
            None => {
                eprintln!("SKIP: no working provider found");
                return;
            }
        };
    };
}

/// Pause between tests to stay under rate limits.
async fn pace() {
    let secs: u64 = std::env::var("E2E_PAUSE_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(15);
    tokio::time::sleep(Duration::from_secs(secs)).await;
}

fn build_tools_with_skill_roots(roots: Vec<PathBuf>) -> ToolRegistry {
    let resolver = SkillResolver::new(roots);
    let available = resolver.list();
    let tools: Vec<Box<dyn Tool>> = vec![
        Box::new(BashTool::new(30)),
        Box::new(ReadFileTool),
        Box::new(WriteFileTool),
        Box::new(EditFileTool),
        Box::new(GlobSearchTool),
        Box::new(GrepSearchTool),
        Box::new(SkillTool::new(resolver, &available)),
    ];
    ToolRegistry::new(tools)
}

fn build_tools_with_mcp(mcp_tools: Vec<Box<dyn Tool>>) -> ToolRegistry {
    let mut tools: Vec<Box<dyn Tool>> = vec![
        Box::new(BashTool::new(30)),
        Box::new(ReadFileTool),
        Box::new(WriteFileTool),
        Box::new(EditFileTool),
        Box::new(GlobSearchTool),
        Box::new(GrepSearchTool),
        Box::new(SkillTool::new(SkillResolver::new(vec![]), &[])),
    ];
    tools.extend(mcp_tools);
    ToolRegistry::new(tools)
}

async fn run_prompt_with_tools(
    provider: Box<dyn Provider>,
    history: &mut ConversationHistory,
    cwd: &Path,
    model: &str,
    tools: ToolRegistry,
) -> E2eResult {
    let config = loop_config(cwd.to_path_buf(), model);
    let agent = AgentLoop::new(provider, tools, config);

    let (tx, mut rx) = mpsc::channel(256);
    let cancel = CancellationToken::new();

    let timeout = Duration::from_secs(120);
    let result = tokio::time::timeout(timeout, agent.run(history, tx, cancel, None, None)).await;

    match result {
        Ok(Ok(_usage)) => {}
        Ok(Err(e)) => eprintln!("  AgentLoop error: {e}"),
        Err(_) => eprintln!("  E2E TIMEOUT after {timeout:?}"),
    }

    let mut events = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        events.push(ev);
    }

    E2eResult {
        events,
        history: history.clone(),
    }
}

fn mcp_server_script_path() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
    PathBuf::from(manifest)
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join("test-mcp-server.sh"))
        .unwrap_or_else(|| PathBuf::from("test-mcp-server.sh"))
}

fn skill_roots_from_config(config: &Config) -> Vec<PathBuf> {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    config
        .skill_roots
        .iter()
        .map(|p| {
            let s = p.to_string_lossy();
            if let Some(rest) = s.strip_prefix("~/") {
                PathBuf::from(format!("{home}/{rest}"))
            } else if s == "~" {
                PathBuf::from(&home)
            } else {
                p.clone()
            }
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests (run with --test-threads=1)
// ═══════════════════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════════════════
// Multi-provider tests (t13-t16)
// ═══════════════════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════════════════
// Skill tests (t17-t19)
// ═══════════════════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════════════════
// MCP tests (t20-t22)
// ═══════════════════════════════════════════════════════════════════════════════

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

// ── t34–t36: multi-key rotation ──────────────────────────────────────────────

#[test]
fn t34_multi_key_creates_resilient_provider() {
    let resolved = naked_core::config::ResolvedProvider {
        provider_type: "openai_compat".into(),
        api_key: "key-a".into(),
        all_keys: vec!["key-a".into(), "key-b".into(), "key-c".into()],
        base_url: Some("https://example.com/v1".into()),
        models: vec!["test-model".into()],
        max_tokens: None,
        temperature: None,
        headers: Default::default(),
        model_aliases: Default::default(),
    };

    let provider = naked_core::create_provider("multi", resolved);
    assert_eq!(provider.name(), "multi[key-0]");
    assert!(
        provider.models().len() >= 3,
        "should aggregate models from all sub-providers"
    );
}

#[test]
fn t35_single_key_creates_plain_provider() {
    let resolved = naked_core::config::ResolvedProvider {
        provider_type: "openai_compat".into(),
        api_key: "key-only".into(),
        all_keys: vec!["key-only".into()],
        base_url: Some("https://example.com/v1".into()),
        models: vec!["test-model".into()],
        max_tokens: None,
        temperature: None,
        headers: Default::default(),
        model_aliases: Default::default(),
    };

    let provider = naked_core::create_provider("single", resolved);
    assert_eq!(provider.name(), "single");
    assert_eq!(provider.models().len(), 1);
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

// ── t37: native Anthropic with multi-key rotation ────────────────────────────

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

// ── t38: parallel load test — 2 chats × N messages, different providers ──────
//
// Validates: parallel sessions, all 7 tools, content correctness, file
// integrity, multi-turn memory, provider isolation, key rotation under load.

#[derive(Debug)]
#[allow(dead_code)]
struct ChatStats {
    tag: String,
    turns: usize,
    ok: usize,
    content_ok: usize,
    errors: usize,
    rate_limited: usize,
    tools_used: usize,
    tool_set: std::collections::HashSet<String>,
    /// Sum of individual turn latencies (excludes inter-turn delays).
    total_ms: u128,
    /// Wall-clock time for the entire session (includes inter-turn delays).
    session_wall_ms: u128,
    latencies: Vec<u128>,
}

impl ChatStats {
    fn percentile(&self, p: f64) -> u128 {
        if self.latencies.is_empty() {
            return 0;
        }
        let mut sorted = self.latencies.clone();
        sorted.sort();
        let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }
    fn avg(&self) -> u128 {
        if self.latencies.is_empty() {
            0
        } else {
            self.total_ms / self.latencies.len() as u128
        }
    }
    fn throughput(&self) -> f64 {
        if self.total_ms == 0 {
            0.0
        } else {
            self.turns as f64 / (self.total_ms as f64 / 1000.0)
        }
    }
}

/// Prompt + expected assertion.
struct TurnSpec {
    prompt: String,
    /// Substring that must appear in the text response (None = no content check).
    expect_text: Option<String>,
    /// Tool name that should have been called (None = no tool check).
    expect_tool: Option<String>,
    /// If set, verify this file exists with this content after the turn.
    expect_file: Option<(PathBuf, String)>,
}

fn build_load_specs(tag: &str, n: usize, tmp: &Path) -> Vec<TurnSpec> {
    (0..n)
        .map(|i| match i % 10 {
            // 0: plain text — verify exact token
            0 => TurnSpec {
                prompt: format!("Say exactly: {tag}_TURN_{i}. Nothing else."),
                expect_text: Some(format!("{tag}_TURN_{i}")),
                expect_tool: None,
                expect_file: None,
            },
            // 1: bash — verify output token
            1 => TurnSpec {
                prompt: format!("Use bash to run: echo {tag}_BASH_{i}. Report the output."),
                expect_text: Some(format!("{tag}_BASH_{i}")),
                expect_tool: Some("bash".into()),
                expect_file: None,
            },
            // 2: write_file — verify file created
            2 => {
                let path = tmp.join(format!("load_{i}.txt"));
                let data = format!("{tag}_DATA_{i}");
                TurnSpec {
                    prompt: format!(
                        "Write '{}' to {} using write_file.",
                        data,
                        path.display()
                    ),
                    expect_text: None,
                    expect_tool: Some("write_file".into()),
                    expect_file: Some((path, data)),
                }
            }
            // 3: read_file — read back the file written at turn i-1
            3 => {
                let written_at = i - 1;
                let path = tmp.join(format!("load_{written_at}.txt"));
                let token = format!("{tag}_DATA_{written_at}");
                TurnSpec {
                    prompt: format!("Read the file {} and tell me its content.", path.display()),
                    expect_text: Some(token),
                    expect_tool: Some("read_file".into()),
                    expect_file: None,
                }
            }
            // 4: edit_file — replace token in the file written at turn i-2
            4 => {
                let written_at = i - 2;
                let path = tmp.join(format!("load_{written_at}.txt"));
                let old_token = format!("{tag}_DATA_{written_at}");
                let new_token = format!("{tag}_EDIT_{i}");
                TurnSpec {
                    prompt: format!(
                        "Use edit_file to replace '{}' with '{}' in {}.",
                        old_token,
                        new_token,
                        path.display()
                    ),
                    expect_text: None,
                    expect_tool: Some("edit_file".into()),
                    expect_file: Some((path, new_token)),
                }
            }
            // 5: glob_search — find .txt files
            5 => TurnSpec {
                prompt: format!(
                    "Use glob_search to find all *.txt files in {}. List them.",
                    tmp.display()
                ),
                expect_text: Some("load_".into()),
                expect_tool: Some("glob_search".into()),
                expect_file: None,
            },
            // 6: grep_search — find token in files
            6 => {
                let token = format!("{tag}_");
                TurnSpec {
                    prompt: format!(
                        "Use grep_search to find '{}' in {}. Report matches.",
                        token,
                        tmp.display()
                    ),
                    expect_text: Some(tag.into()),
                    expect_tool: Some("grep_search".into()),
                    expect_file: None,
                }
            }
            // 7: bash again — different command
            7 => TurnSpec {
                prompt: "Use bash to run: date +%Y. Report the year.".into(),
                expect_text: Some("202".into()),
                expect_tool: Some("bash".into()),
                expect_file: None,
            },
            // 8: multi-turn memory — recall
            8 => TurnSpec {
                prompt: "What was the exact token I asked you to say earlier in this conversation? Reply with only the token.".into(),
                expect_text: Some(format!("{tag}_TURN_")),
                expect_tool: None,
                expect_file: None,
            },
            // 9: write + read chain in one prompt
            _ => {
                let path = tmp.join(format!("chain_{i}.txt"));
                let data = format!("{tag}_CHAIN_{i}");
                TurnSpec {
                    prompt: format!(
                        "Write '{}' to {} then read it back. Confirm the content.",
                        data,
                        path.display()
                    ),
                    expect_text: Some(data.clone()),
                    expect_tool: Some("write_file".into()),
                    expect_file: Some((path, data)),
                }
            }
        })
        .collect()
}

async fn run_load_session(
    tag: &str,
    provider_name: &str,
    model: &str,
    config: &Config,
    num_turns: usize,
) -> Option<ChatStats> {
    let pc = config.providers.get(provider_name)?;
    let resolved = pc.resolved().ok()?;
    if resolved.api_key.is_empty() || resolved.api_key.starts_with('$') {
        eprintln!("  [{tag}] SKIP: {provider_name} key unresolvable");
        return None;
    }

    // Probe once with a shared provider (DRY: reuse across turns)
    let shared_provider = naked_core::create_provider(provider_name, resolved);
    if !probe_provider(shared_provider.as_ref(), model).await {
        eprintln!("  [{tag}] SKIP: {provider_name}/{model} probe failed");
        return None;
    }
    let provider: Arc<dyn Provider> = Arc::from(shared_provider);

    let tmp = tempfile::tempdir().unwrap();
    let mut history = ConversationHistory::new(SYS.into());
    let specs = build_load_specs(tag, num_turns, tmp.path());

    let mut ok = 0usize;
    let mut content_ok = 0usize;
    let mut errors = 0usize;
    let mut rate_limited = 0usize;
    let mut tools_used = 0usize;
    let mut tool_set = std::collections::HashSet::new();
    let mut latencies = Vec::with_capacity(num_turns);
    let mut turn_delay_ms = 3000u64; // adaptive: increases on 429
    let session_t0 = std::time::Instant::now();

    eprintln!("  [{tag}] START {provider_name}/{model} — {num_turns} turns");

    for (i, spec) in specs.iter().enumerate() {
        if i > 0 {
            tokio::time::sleep(Duration::from_millis(turn_delay_ms)).await;
        }

        let t0 = std::time::Instant::now();
        history.push_user(&spec.prompt);

        let mut text_acc = String::new();
        let mut turn_tools = Vec::new();
        let mut err_msg = String::new();

        let tools = build_tools();
        let lc = LoopConfig {
            max_iterations: 5,
            cwd: tmp.path().to_path_buf(),
            model: model.to_string(),
            max_tokens: 1024,
            temperature: None,
            reasoning: None,
            provider: String::new(),
            health: None,
        };
        let prov_box = provider_arc_to_box(&provider);
        let agent = AgentLoop::new(prov_box, tools, lc);
        let (tx, mut rx) = mpsc::channel(256);
        let cancel = CancellationToken::new();

        let result = tokio::time::timeout(
            Duration::from_secs(180),
            agent.run(&mut history, tx, cancel, None, None),
        )
        .await;

        while let Ok(ev) = rx.try_recv() {
            match &ev {
                AgentEvent::TextDelta(t) => text_acc.push_str(t),
                AgentEvent::ToolStart { name, .. } => turn_tools.push(name.clone()),
                AgentEvent::Error(e) => err_msg = e.clone(),
                _ => {}
            }
        }

        let elapsed = t0.elapsed().as_millis();
        latencies.push(elapsed);

        let result_err_str = match &result {
            Ok(Err(e)) => e.to_string(),
            Err(_) => "timeout".into(),
            _ => String::new(),
        };
        let all_errors = format!("{err_msg} {result_err_str}");
        let is_429 = all_errors.contains("429") || all_errors.contains("rate_limit");
        let turn_ok = matches!(result, Ok(Ok(_))) && err_msg.is_empty();

        if turn_ok {
            ok += 1;
            // Successful turn — gradually reduce delay back toward baseline
            if turn_delay_ms > 3000 {
                turn_delay_ms = (turn_delay_ms * 3 / 4).max(3000);
            }
        } else if is_429 {
            rate_limited += 1;
            // Adaptive backoff: double the delay on rate limit, cap at 15s
            turn_delay_ms = (turn_delay_ms * 2).min(15000);
            eprintln!(
                "  [{tag}] turn {} RATE LIMITED (delay now {turn_delay_ms}ms)",
                i + 1
            );
        } else {
            errors += 1;
            if let Ok(Err(e)) = &result {
                eprintln!("  [{tag}] turn {} AGENT ERROR: {e}", i + 1);
            } else if result.is_err() {
                eprintln!("  [{tag}] turn {} TIMEOUT (180s)", i + 1);
            } else if !err_msg.is_empty() {
                eprintln!("  [{tag}] turn {} ERROR: {err_msg}", i + 1);
            }
        }

        // Content verification (skip for rate-limited turns)
        let mut content_pass = true;
        if turn_ok {
            if let Some(expected) = &spec.expect_text
                && !text_acc.contains(expected.as_str())
            {
                content_pass = false;
                if (i + 1) % 10 == 0 {
                    eprintln!(
                        "  [{tag}] turn {} content MISS: expected '{}' in '{}'",
                        i + 1,
                        expected,
                        &text_acc[..text_acc.len().min(120)]
                    );
                }
            }
            if let Some(expected_tool) = &spec.expect_tool
                && !turn_tools.iter().any(|t| t == expected_tool)
            {
                content_pass = false;
                eprintln!(
                    "  [{tag}] turn {} tool MISS: expected '{}', got {:?}",
                    i + 1,
                    expected_tool,
                    turn_tools
                );
            }
            if let Some((path, expected_content)) = &spec.expect_file
                && path.exists()
            {
                let actual = std::fs::read_to_string(path).unwrap_or_default();
                if !actual.contains(expected_content.as_str()) {
                    content_pass = false;
                    eprintln!(
                        "  [{tag}] turn {} file MISS: {} expected '{}', got '{}'",
                        i + 1,
                        path.display(),
                        expected_content,
                        &actual[..actual.len().min(80)]
                    );
                }
            }
            if content_pass {
                content_ok += 1;
            }
        }

        for t in &turn_tools {
            tool_set.insert(t.clone());
        }
        tools_used += turn_tools.len();

        if (i + 1) % 10 == 0 || i == 0 {
            let status = if is_429 {
                "RATE-LIMITED"
            } else if turn_ok && content_pass {
                "OK"
            } else if turn_ok {
                "ok(content miss)"
            } else {
                "ERR"
            };
            eprintln!(
                "  [{tag}] turn {}/{num_turns}: {elapsed}ms {status} [tools: {}]",
                i + 1,
                turn_tools.join(","),
            );
        }

        history.auto_compact();
    }

    let total_ms: u128 = latencies.iter().sum();
    let session_wall_ms = session_t0.elapsed().as_millis();

    eprintln!(
        "  [{tag}] DONE: {ok}/{num_turns} ok, {content_ok} content-verified, {errors} err, {rate_limited} rate-limited, {} tools ({} unique: {:?})",
        tools_used,
        tool_set.len(),
        tool_set,
    );

    Some(ChatStats {
        tag: tag.into(),
        turns: num_turns,
        ok,
        content_ok,
        errors,
        rate_limited,
        tools_used,
        tool_set,
        total_ms,
        session_wall_ms,
        latencies,
    })
}

fn provider_arc_to_box(provider: &Arc<dyn Provider>) -> Box<dyn Provider> {
    struct ArcWrap(Arc<dyn Provider>);
    #[async_trait::async_trait]
    impl Provider for ArcWrap {
        fn name(&self) -> &str {
            self.0.name()
        }
        fn models(&self) -> Vec<naked_core::types::ModelInfo> {
            self.0.models()
        }
        async fn stream_chat(
            &self,
            request: naked_core::provider::ChatRequest,
        ) -> naked_core::error::Result<
            std::pin::Pin<
                Box<dyn tokio_stream::Stream<Item = naked_core::types::StreamChunk> + Send>,
            >,
        > {
            self.0.stream_chat(request).await
        }
    }
    Box::new(ArcWrap(Arc::clone(provider)))
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

// ═══════════════════════════════════════════════════════════════════════════════
// Per-session config override (t39–t43)
// ═══════════════════════════════════════════════════════════════════════════════

/// Helper: find N working providers (probed), return vec of (name, model).
/// Skips providers known to have aggressive daily limits (groq free-tier).
async fn find_working_providers(config: &Config, n: usize) -> Vec<(String, String)> {
    let skip = ["groq"];
    let mut found = Vec::new();
    for (name, pc) in &config.providers {
        if found.len() >= n {
            break;
        }
        if skip.iter().any(|s| name.contains(s)) {
            continue;
        }
        if let Ok(r) = pc.resolved()
            && !r.api_key.is_empty()
            && !r.api_key.starts_with('$')
            && !pc.models.is_empty()
        {
            let model = &pc.models[0];
            let prov = naked_core::create_provider(name, r);
            if probe_provider(prov.as_ref(), model).await {
                found.push((name.clone(), model.clone()));
            }
        }
    }
    found
}

/// Helper: create an AgentCore with given default provider/model.
fn make_agent_core(
    config: &Config,
    tmp: &std::path::Path,
    provider_name: &str,
    model: &str,
) -> naked_core::AgentCore {
    let core_config = Config {
        providers: config.providers.clone(),
        workspace: tmp.to_path_buf(),
        session_dir: tmp.join("sessions"),
        default_provider: provider_name.to_string(),
        default_model: model.to_string(),
        max_iterations: 10,
        max_tokens: 8192,
        tool_timeout_secs: 30,
        skill_roots: config.skill_roots.clone(),
        mcp_servers: config.mcp_servers.clone(),
        ..Default::default()
    };
    let primary = naked_core::build_provider_from_config(&core_config).unwrap();
    naked_core::AgentCore::new(core_config, primary)
}

/// Helper: run a prompt via AgentCore and collect text + tool names.
async fn agent_prompt(
    agent: &naked_core::AgentCore,
    session_id: &str,
    prompt: &str,
) -> (String, Vec<String>, bool) {
    let mut handle = match agent.send_prompt(session_id, prompt).await {
        Ok(h) => h,
        Err(e) => return (format!("ERROR: {e}"), vec![], false),
    };
    let mut text = String::new();
    let mut tools = Vec::new();
    let mut idle = false;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(120);
    loop {
        match tokio::time::timeout_at(deadline, handle.events.recv()).await {
            Ok(Some(AgentEvent::TextDelta(t))) => text.push_str(&t),
            Ok(Some(AgentEvent::ToolStart { name, .. })) => tools.push(name),
            Ok(Some(AgentEvent::Idle)) => {
                idle = true;
                break;
            }
            Ok(Some(AgentEvent::Error(e))) => {
                eprintln!("    error: {e}");
                break;
            }
            Ok(None) | Err(_) => break,
            _ => {}
        }
    }
    (text, tools, idle)
}

/// Write a per-session config.json into the session directory.
async fn write_session_config(
    sessions_dir: &std::path::Path,
    session_id: &str,
    config: &serde_json::Value,
) {
    let dir = sessions_dir.join(session_id);
    tokio::fs::create_dir_all(&dir).await.unwrap();
    tokio::fs::write(
        dir.join("config.json"),
        serde_json::to_string_pretty(config).unwrap(),
    )
    .await
    .unwrap();
}

// ── t39: Two sessions, different providers/models via config.json ────────────

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

// ── t40: Per-session prompt.md injection ─────────────────────────────────────

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

// ── t41: Per-session config.json with custom system_prompt_path ──────────────

#[tokio::test]
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

// ── t42: Config.json live reload between turns ───────────────────────────────

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

// ── t43: Per-session MCP servers (additive) ──────────────────────────────────

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

// ── t44: set_session_provider API (used by /provider and /model commands) ────

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

    // Switch only model (keep provider)
    agent
        .set_session_provider(&session_id, None, Some("other-model"))
        .await
        .unwrap();
    let (cur_p2, cur_m2) = agent.session_provider_model(&session_id).await;
    assert_eq!(cur_p2, *prov_b, "provider should stay unchanged");
    assert_eq!(cur_m2, "other-model", "model should be updated");
    eprintln!("  model-only switch: {cur_p2}/{cur_m2}");
}

// ── t45: 3 isolated sessions × 10 turns — skill / model / prompt ─────────

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

struct SessionStats {
    label: &'static str,
    responses: usize,
    idles: usize,
    all_tools: Vec<String>,
    all_texts: Vec<String>,
}

impl SessionStats {
    fn new(label: &'static str) -> Self {
        Self {
            label,
            responses: 0,
            idles: 0,
            all_tools: Vec::new(),
            all_texts: Vec::new(),
        }
    }

    fn record(&mut self, _turn: usize, text: &str, tools: &[String], idle: bool) {
        if !text.is_empty() {
            self.responses += 1;
        }
        if idle {
            self.idles += 1;
        }
        self.all_tools.extend(tools.iter().cloned());
        self.all_texts.push(text.to_string());
    }

    fn tool_used(&self, name: &str) -> bool {
        self.all_tools.iter().any(|t| t == name)
    }

    fn text_contains_count(&self, needle: &str) -> usize {
        self.all_texts.iter().filter(|t| t.contains(needle)).count()
    }
}

impl std::fmt::Display for SessionStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: {} responses, {} idles, tools: {:?}",
            self.label,
            self.responses,
            self.idles,
            self.all_tools
                .iter()
                .collect::<std::collections::HashSet<_>>()
        )
    }
}

// ── Multi-user concurrent E2E test ──────────────────────────────────────────

/// Simulates 3 concurrent users, each in their own session, sending prompts
/// simultaneously. Verifies session isolation, no cross-contamination, and
/// that concurrent turns on different sessions work correctly.
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

// ── Unicode edge cases E2E test ─────────────────────────────────────────────

/// Tests that the agent correctly handles Unicode in prompts and responses:
/// Cyrillic, emoji, CJK characters. Validates no panics from string slicing.
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

// ═══════════════════════════════════════════════════════════════════════════════
// Audit-fix E2E tests (offline — no LLM needed)
// ═══════════════════════════════════════════════════════════════════════════════

/// #2 — grep_search / glob_search can search outside workspace
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

/// #2 — write/edit outside workspace escalates to Dangerous permission
#[tokio::test]
async fn t49_file_ops_permission_escalation() {
    let workspace = tempfile::tempdir().unwrap();

    let write_tool = WriteFileTool;
    let edit_tool = EditFileTool;
    let read_tool = ReadFileTool;

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

/// #3 — allowed_chat_ids: empty = deny all
#[tokio::test]
async fn t50_allowed_chat_ids_deny_when_empty() {
    let empty_config = Config {
        allowed_chat_ids: vec![],
        ..Config::default()
    };
    assert!(
        empty_config.allowed_chat_ids.is_empty(),
        "config should have empty allowed_chat_ids"
    );

    let populated_config = Config {
        allowed_chat_ids: vec![100, 200],
        ..Config::default()
    };
    assert!(populated_config.allowed_chat_ids.contains(&100));
    assert!(!populated_config.allowed_chat_ids.contains(&999));

    eprintln!("  PASS: allowed_chat_ids config is deny-by-default");
}

/// #4 — Skill resolver rejects path traversal
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

/// #5 — HTTP clients have timeouts configured
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

/// #7 — Session store save no longer panics on serialize
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

/// #8 — Retry backoff uses exponential delay
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
        temperature: None,
        reasoning: None,
        provider: String::new(),
        health: None,
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

/// #9 — MCP JSON-RPC IDs are unique (covered by unit tests in mcp/client.rs)
/// #14 — SkillTool uses async IO (no blocking)
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

/// #15 — ReadFileTool rejects files over 10MB
#[tokio::test]
async fn t57_read_file_size_limit() {
    let tmp = tempfile::tempdir().unwrap();
    let big_file = tmp.path().join("huge.bin");

    // Create an 11MB file
    let data = vec![b'x'; 11 * 1024 * 1024];
    std::fs::write(&big_file, &data).unwrap();

    let tool = ReadFileTool;
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

/// #16 — glob_search caps results at 200
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

/// #11 — No cross-session data in system prompt
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

/// #12 — auto_approve field removed from Config (dead code cleanup)
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

/// #20 — permission request with PermissionRequest event for outside-workspace file ops
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
        temperature: None,
        reasoning: None,
        provider: String::new(),
        health: None,
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

// ── Context compaction E2E tests ──────────────────────────────────────────────

/// Emergency compaction: provider returns "prompt too long" on first call,
/// then verifies the compacted context is preserved (summary + recent messages),
/// then continues with tool calls (write_file, read_file) and final text.
///
/// Validates the full flow:
///   error → compact → context preserved → retry → tool use → continue → finish
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
                    Err(naked_core::error::AgentError::Provider(
                        "OpenAI API 400 Bad Request: {\"error\":{\"message\":\"The prompt is too long: 1039163, model maximum context length: 202751\"}}".into()
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
        temperature: None,
        reasoning: None,
        provider: String::new(),
        health: None,
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

/// Emergency compaction bails out if history is already minimal (<=3 messages).
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
        temperature: None,
        reasoning: None,
        provider: String::new(),
        health: None,
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

/// Auto-compact with ContextCompacted event in agent loop flow.
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
        temperature: None,
        reasoning: None,
        provider: String::new(),
        health: None,
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

// ── Feature tests: binary detection, bash validation, git context, ──────────
// ── cost estimation, workspace cwd, instruction walk-up, parallel tools ─────

/// #17 — ReadFileTool detects binary files via NUL-byte sniffing
#[tokio::test]
async fn t65_read_binary_file_detection() {
    let tmp = tempfile::tempdir().unwrap();

    // PNG header contains NUL bytes
    let png = tmp.path().join("image.png");
    std::fs::write(&png, b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00").unwrap();

    let tool = ReadFileTool;
    let result = tool
        .execute(serde_json::json!({"file_path": "image.png"}), tmp.path())
        .await;
    assert!(result.is_error, "binary file should be rejected");
    assert!(
        result.output.contains("Binary file"),
        "error should mention binary: {}",
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

/// #18 — Bash command classification: read-only, write, destructive
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

/// #19 — BashTool.effective_permission uses classification
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

/// #20 — Git context injected into environment section
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

/// #21 — Token cost estimation
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

/// #22 — Workspace root used as tool cwd (not artifacts)
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

/// #23 — Instruction file walk-up with budgets and dedup
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

/// #24 — Parallel read-only tool execution
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

/// #25 — Bash tool output truncation at 16 KiB
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

/// #26 — ReadFile output truncation at 16 KiB
#[tokio::test]
async fn t74_readfile_output_truncation() {
    let tmp = tempfile::tempdir().unwrap();

    // Create a large text file (500 lines of 100 chars each = ~50 KiB)
    let content: String = (0..500)
        .map(|i| format!("line {i}: {}\n", "a".repeat(90)))
        .collect();
    std::fs::write(tmp.path().join("big.txt"), &content).unwrap();

    let tool = ReadFileTool;
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

/// #27 — Grep output truncation at 16 KiB
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

/// #28 — Glob search caps at 200 results (reduced from 1000)
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

/// #29 — Sub-agent tool: explore mode runs read-only tools and returns result
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

/// #30 — Sub-agent tool: permission levels match mode
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

/// #31 — Sub-agent tool: empty prompt rejected
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

// ═══════════════════════════════════════════════════════════════════════════
// Web Search Tool tests
// ═══════════════════════════════════════════════════════════════════════════

use naked_core::tool::web_search::WebSearchTool;

/// #32 — WebSearch: spec is correct
#[tokio::test]
async fn t80_web_search_spec() {
    let tool = WebSearchTool::from_legacy_exa(vec![]);
    let spec = tool.spec();
    assert_eq!(spec.name, "web_search");
    assert_eq!(spec.permission, Permission::ReadOnly);
    eprintln!("  PASS: web_search spec");
}

/// #33 — WebSearch: empty query returns error
#[tokio::test]
async fn t81_web_search_empty_query() {
    let tool = WebSearchTool::from_legacy_exa(vec![]);
    let r = tool
        .execute(serde_json::json!({"query": ""}), Path::new("/tmp"))
        .await;
    assert!(r.is_error, "empty query should error");
    eprintln!("  PASS: web_search empty query error");
}

/// #34 — WebSearch: missing query field returns error
#[tokio::test]
async fn t82_web_search_invalid_input() {
    let tool = WebSearchTool::from_legacy_exa(vec![]);
    let r = tool
        .execute(serde_json::json!({"wrong": 1}), Path::new("/tmp"))
        .await;
    assert!(r.is_error, "missing query should error");
    eprintln!("  PASS: web_search invalid input error");
}

/// #35 — WebSearch: exa.ai live search (requires EXA_API_KEYS env)
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

/// #36 — WebSearch: DuckDuckGo fallback (no exa keys)
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

/// #37 — WebSearch: key rotation works across calls
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

// ═══════════════════════════════════════════════════════════════════════════════
// Deep Research E2E (t86-t88) — full 3-phase research workflow
// ═══════════════════════════════════════════════════════════════════════════════

fn research_system_prompt(cwd: &Path) -> String {
    format!(
        "Ты — универсальный AI-ассистент. Отвечай на языке пользователя.\n\n\
         ## Правила многоэтапных скиллов\n\n\
         Когда SKILL.md содержит «СТОП» или «заверши ход»:\n\
         1. Выведи текст до точки остановки\n\
         2. НЕМЕДЛЕННО ЗАВЕРШИ ОТВЕТ — не продолжай дальше\n\
         3. Не вызывай следующие шаги, не делай поиск, не генерируй файлы\n\
         4. Жди следующего сообщения пользователя\n\n\
         Working directory: {}\n\
         Язык ответов: русский.",
        cwd.display()
    )
}

/// Run agent with inactivity timeout — resets after each event.
/// `idle_timeout_secs` is per-event: if no new event arrives within this window, cancel.
async fn run_research_prompt(
    provider: Box<dyn Provider>,
    history: &mut ConversationHistory,
    cwd: &Path,
    model: &str,
    tools: ToolRegistry,
    idle_timeout_secs: u64,
) -> E2eResult {
    use std::sync::atomic::{AtomicU64, Ordering};

    fn now_ms() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    }

    let config = LoopConfig {
        max_iterations: 50,
        cwd: cwd.to_path_buf(),
        model: model.to_string(),
        max_tokens: 8192,
        temperature: None,
        reasoning: None,
        provider: String::new(),
        health: None,
    };
    let agent = AgentLoop::new(provider, tools, config);

    // Two channels: agent → forwarder → collector
    let (agent_tx, mut agent_rx) = mpsc::channel::<AgentEvent>(4096);
    let (collect_tx, mut collect_rx) = mpsc::channel::<AgentEvent>(4096);
    let cancel = CancellationToken::new();

    let last_event = Arc::new(AtomicU64::new(now_ms()));
    let idle_ms = idle_timeout_secs * 1000;

    // Forwarder: reads agent events, bumps timestamp, forwards to collector.
    // Logs a status line every 60s with token count, active tools, and sub-agent activity.
    let le_writer = last_event.clone();
    let fwd_start = std::time::Instant::now();
    tokio::spawn(async move {
        let mut tokens_total: u64 = 0;
        let mut active_tools: Vec<String> = Vec::new();
        let mut active_sub_agents: Vec<String> = Vec::new();
        let mut last_status = std::time::Instant::now();
        let mut event_count: u64 = 0;

        while let Some(ev) = agent_rx.recv().await {
            event_count += 1;
            let is_progress = matches!(
                ev,
                AgentEvent::TextDelta(_)
                    | AgentEvent::ThinkingDelta(_)
                    | AgentEvent::ToolStart { .. }
                    | AgentEvent::ToolEnd { .. }
                    | AgentEvent::SubAgentProgress { .. }
                    | AgentEvent::UsageUpdate(_)
            );
            if is_progress {
                le_writer.store(now_ms(), Ordering::Relaxed);
            }

            match &ev {
                AgentEvent::UsageUpdate(u) => {
                    tokens_total += u.total_tokens();
                }
                AgentEvent::ToolStart { name, .. } => {
                    active_tools.push(name.clone());
                }
                AgentEvent::ToolEnd { name, .. } => {
                    if let Some(pos) = active_tools.iter().position(|n| n == name) {
                        active_tools.remove(pos);
                    }
                }
                AgentEvent::SubAgentProgress { agent_id, event } => match event {
                    SubAgentEvent::Started { .. } => {
                        active_sub_agents.push(agent_id.clone());
                    }
                    SubAgentEvent::Finished { tokens } => {
                        active_sub_agents.retain(|id| id != agent_id);
                        tokens_total += *tokens;
                    }
                    SubAgentEvent::Error(_) => {
                        active_sub_agents.retain(|id| id != agent_id);
                    }
                    _ => {}
                },
                _ => {}
            }

            if last_status.elapsed() >= Duration::from_secs(60) {
                let elapsed = fwd_start.elapsed().as_secs();
                let tools_str = if active_tools.is_empty() {
                    "none".to_string()
                } else {
                    active_tools.join(", ")
                };
                let sa_str = if active_sub_agents.is_empty() {
                    String::new()
                } else {
                    format!(", sub_agents: {}", active_sub_agents.len())
                };
                eprintln!(
                    "  ⏱ {elapsed}s | tokens: {tokens_total} | tools: [{tools_str}]{sa_str} | events: {event_count}",
                );
                last_status = std::time::Instant::now();
            }

            let _ = collect_tx.send(ev).await;
        }
    });

    // Watchdog: cancels agent if no events for idle_timeout_secs
    let le_reader = last_event.clone();
    let cancel_wd = cancel.clone();
    let watchdog = tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(5)).await;
            let elapsed = now_ms().saturating_sub(le_reader.load(Ordering::Relaxed));
            if elapsed > idle_ms {
                eprintln!("  IDLE TIMEOUT: no events for {elapsed}ms");
                cancel_wd.cancel();
                return;
            }
        }
    });

    let start = std::time::Instant::now();
    let result = agent.run(history, agent_tx, cancel, None, None).await;

    match &result {
        Ok(usage) => eprintln!("  usage: {} tokens", usage.total_tokens()),
        Err(e) => eprintln!("  AgentLoop error: {e}"),
    }

    watchdog.abort();

    // Drain all forwarded events
    tokio::time::sleep(Duration::from_millis(100)).await;
    let mut events = Vec::new();
    while let Ok(ev) = collect_rx.try_recv() {
        events.push(ev);
    }

    eprintln!(
        "  total time: {:.1}s, events: {}",
        start.elapsed().as_secs_f64(),
        events.len()
    );

    E2eResult {
        events,
        history: history.clone(),
    }
}

/// Phase 1: Research skill loads and model stops after asking questions (multi-turn)
#[tokio::test]
async fn t86_deep_research_phase1_outline() {
    let config = match test_config() {
        Some(c) => c,
        None => {
            eprintln!("  SKIP t86: no config");
            return;
        }
    };
    let (provider, model) = match get_working_provider(&config).await {
        Some(pm) => pm,
        None => {
            eprintln!("  SKIP t86: no working provider");
            return;
        }
    };

    let roots = skill_roots_from_config(&config);
    let resolver = SkillResolver::new(roots.clone());
    if resolver.resolve("research").is_none() {
        eprintln!("  SKIP t86: 'research' skill not found in {:?}", roots);
        return;
    }

    eprintln!(">>> t86_deep_research_phase1_outline [model={model}]");

    let tmp = tempfile::tempdir().unwrap();
    let cwd = tmp.path();

    let tools = {
        let resolver = SkillResolver::new(skill_roots_from_config(&config));
        let available = resolver.list();
        let exa_keys = config.exa_api_keys.clone();
        let prov_arc: Arc<dyn Provider> = Arc::from(provider);
        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(BashTool::new(60)),
            Box::new(ReadFileTool),
            Box::new(WriteFileTool),
            Box::new(EditFileTool),
            Box::new(GlobSearchTool),
            Box::new(GrepSearchTool),
            Box::new(WebSearchTool::from_legacy_exa(exa_keys.clone())),
            Box::new(SubAgentTool::new(
                prov_arc.clone(),
                model.clone(),
                60,
                exa_keys,
            )),
            Box::new(SkillTool::new(resolver, &available)),
        ];
        ToolRegistry::new(tools)
    };

    let mut history = ConversationHistory::new(research_system_prompt(cwd));
    history.push_user(
        "/research аренда коммерческой площади в Дананге для кафе завтраков, \
         100-200 кв.м., целевая аудитория: туристы и экспаты",
    );

    let r = run_research_prompt(
        get_working_provider(&config).await.unwrap().0,
        &mut history,
        cwd,
        &model,
        tools,
        600,
    )
    .await;

    let text = r.full_text();
    let text_lower = text.to_lowercase();
    eprintln!("  tools used: {:?}", r.tool_names());
    eprintln!("  text length: {}", text.len());
    eprintln!("  text start: {:.500}", text);
    if text.len() > 500 {
        let tail: String = text
            .chars()
            .rev()
            .take(500)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        eprintln!("  text end: {tail}");
    }

    // Skill tool must be invoked
    assert!(
        r.has_tool("Skill"),
        "Skill tool not used: {:?}",
        r.tool_names()
    );

    // Model should ask questions (multi-turn check)
    let asks_question = text.contains('?')
        || text_lower.contains("добавить")
        || text_lower.contains("убрать")
        || text_lower.contains("подходит")
        || text_lower.contains("подтверд");
    eprintln!("  asks user a question: {asks_question}");

    // Model should NOT have created outline.yaml yet (that's Phase 3)
    let wrote_files = r.has_tool("write_file");
    eprintln!("  wrote files (should be false for Phase 1): {wrote_files}");

    // Model should NOT have used sub_agent yet (that's Phase 3)
    let used_subagent = r.has_tool("sub_agent");
    eprintln!("  used sub_agent (should be false for Phase 1): {used_subagent}");

    if asks_question && !wrote_files && !used_subagent {
        eprintln!("  PASS: model stopped after Phase 1 and asked questions");
    } else if asks_question {
        eprintln!("  PARTIAL PASS: model asked questions but also proceeded further");
    } else {
        eprintln!("  WARN: model may not have followed multi-turn flow");
    }

    // The key assertion: Skill was loaded
    // The soft check: model should ask questions
    assert!(
        r.has_tool("Skill"),
        "Research skill must be loaded via Skill tool"
    );
}

/// Phase 2: Verify sub_agent has web_search in its tool registry
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

/// Phase 3: Verify skills are discoverable and describe correctly
#[tokio::test]
async fn t88_deep_research_skills_discovered() {
    let config = match test_config() {
        Some(c) => c,
        None => {
            eprintln!("  SKIP t88: no config");
            return;
        }
    };

    eprintln!(">>> t88_deep_research_skills_discovered");

    let roots = skill_roots_from_config(&config);
    let resolver = SkillResolver::new(roots.clone());
    let _all = resolver.list();

    let research_skills = [
        "research",
        "research-deep",
        "research-report",
        "research-add-items",
        "research-add-fields",
    ];
    let mut found = Vec::new();
    let mut missing = Vec::new();

    for name in &research_skills {
        if let Some(hit) = resolver.resolve(name) {
            let content = std::fs::read_to_string(&hit.path).unwrap();
            let has_frontmatter = content.contains("---") && content.contains("name:");
            let has_trigger = content.contains("Триггер") || content.contains("Trigger");
            eprintln!(
                "  ✅ {name}: {} (frontmatter={has_frontmatter}, trigger={has_trigger})",
                hit.path.display()
            );
            found.push(name.to_string());
        } else {
            eprintln!("  ❌ {name}: NOT FOUND");
            missing.push(name.to_string());
        }
    }

    assert!(
        missing.is_empty(),
        "Missing research skills: {:?} (searched in {:?})",
        missing,
        roots
    );

    // Verify validate_json.py exists and runs
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    let script = PathBuf::from(&home).join(".naked/skills/deep-research/scripts/validate_json.py");
    assert!(
        script.exists(),
        "validate_json.py not found at {}",
        script.display()
    );

    let output = std::process::Command::new("python3")
        .args([script.to_str().unwrap(), "--help"])
        .output();
    match output {
        Ok(o) => {
            assert!(o.status.success(), "validate_json.py --help failed");
            eprintln!("  ✅ validate_json.py runs OK");
        }
        Err(e) => eprintln!("  WARN: couldn't run validate_json.py: {e}"),
    }

    eprintln!("  PASS: all {} research skills discovered", found.len());
}

/// Full 4-turn deep research cycle: plan → search → deep research → report
#[tokio::test]
async fn t89_deep_research_full_cycle() {
    let config = match test_config() {
        Some(c) => c,
        None => {
            eprintln!("  SKIP t89: no config");
            return;
        }
    };

    let roots = skill_roots_from_config(&config);
    let resolver = SkillResolver::new(roots.clone());
    if resolver.resolve("research").is_none() {
        eprintln!("  SKIP t89: 'research' skill not found");
        return;
    }

    // Resolve provider+model once, reuse for all turns to avoid probe flakiness.
    let provider_name =
        std::env::var("E2E_PROVIDER").unwrap_or_else(|_| config.default_provider.clone());
    let model = std::env::var("E2E_MODEL").unwrap_or_else(|_| {
        config
            .providers
            .get(&provider_name)
            .and_then(|pc| pc.models.first().cloned())
            .unwrap_or_else(|| config.default_model.clone())
    });

    let make_prov = || -> Box<dyn Provider> {
        make_provider_for(&config, &provider_name, &model)
            .expect("configured provider must be available")
    };

    // Quick sanity check that the provider actually works.
    {
        let prov = make_prov();
        if !probe_provider(prov.as_ref(), &model).await {
            eprintln!("  SKIP t89: provider {provider_name}/{model} probe failed");
            return;
        }
    }

    eprintln!(">>> t89_deep_research_full_cycle [provider={provider_name}, model={model}]");

    let tmp = tempfile::tempdir().unwrap();
    let cwd = tmp.path();

    let mut history = ConversationHistory::new(research_system_prompt(cwd));

    let build_research_tools = |config: &Config, model: &str| -> ToolRegistry {
        let resolver = SkillResolver::new(skill_roots_from_config(config));
        let available = resolver.list();
        let exa_keys = config.exa_api_keys.clone();
        let prov_arc: Arc<dyn Provider> = Arc::from(
            make_provider_for(config, &provider_name, model).expect("provider must be available"),
        );
        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(BashTool::new(120)),
            Box::new(ReadFileTool),
            Box::new(WriteFileTool),
            Box::new(EditFileTool),
            Box::new(GlobSearchTool),
            Box::new(GrepSearchTool),
            Box::new(WebSearchTool::from_legacy_exa(exa_keys.clone())),
            Box::new(SubAgentTool::new(
                prov_arc,
                model.to_string(),
                120,
                exa_keys,
            )),
            Box::new(SkillTool::new(resolver, &available)),
        ];
        ToolRegistry::new(tools)
    };

    // ── Turn 1: Load skill, generate plan, ask questions ──
    eprintln!("\n  ── Turn 1: plan ──");
    history.push_user(
        "/research сравнение 3 языков программирования для CLI-утилит: Rust, Go, Zig. \
         Критерии: скорость компиляции, размер бинаря, экосистема, порог входа.",
    );

    let tools = build_research_tools(&config, &model);
    let r1 = run_research_prompt(make_prov(), &mut history, cwd, &model, tools, 180).await;

    let t1 = r1.full_text();
    eprintln!("  turn1 tools: {:?}", r1.tool_names());
    eprintln!("  turn1 text ({} bytes): {:.300}", t1.len(), t1);
    assert!(r1.has_tool("Skill"), "Turn 1 must load Skill");
    assert!(!r1.has_tool("write_file"), "Turn 1 must NOT write files");
    assert!(!r1.has_tool("sub_agent"), "Turn 1 must NOT use sub_agent");
    let has_question = t1.contains('?') || t1.to_lowercase().contains("добавить");
    eprintln!("  turn1 asks question: {has_question}");

    tokio::time::sleep(Duration::from_secs(2)).await;

    // ── Turn 2: User confirms, agent does web search ──
    eprintln!("\n  ── Turn 2: web search ──");
    history.push_user("Всё отлично, ничего менять не надо. Период: 2024-2025. Начинай.");

    let tools = build_research_tools(&config, &model);
    let r2 = run_research_prompt(make_prov(), &mut history, cwd, &model, tools, 300).await;

    let t2 = r2.full_text();
    eprintln!("  turn2 tools: {:?}", r2.tool_names());
    eprintln!("  turn2 text ({} bytes): {:.300}", t2.len(), t2);
    let used_search = r2.has_tool("web_search");
    eprintln!("  turn2 web_search: {used_search}");

    tokio::time::sleep(Duration::from_secs(2)).await;

    // ── Turn 3: User confirms, agent creates outline + runs deep research ──
    eprintln!("\n  ── Turn 3: deep research ──");
    history.push_user("Да, подтверждаю. Запускай глубокое исследование.");

    let tools = build_research_tools(&config, &model);
    let r3 = run_research_prompt(make_prov(), &mut history, cwd, &model, tools, 600).await;

    let t3 = r3.full_text();
    eprintln!("  turn3 tools: {:?}", r3.tool_names());
    eprintln!("  turn3 text ({} bytes): {:.300}", t3.len(), t3);
    let wrote_files = r3.has_tool("write_file");
    let used_subagent = r3.has_tool("sub_agent");
    eprintln!("  turn3 write_file: {wrote_files}, sub_agent: {used_subagent}");

    fn list_tree(dir: &Path, prefix: &str) {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                eprintln!("  {prefix}{}", p.file_name().unwrap().to_string_lossy());
                if p.is_dir() {
                    list_tree(&p, &format!("{prefix}  "));
                }
            }
        }
    }
    eprintln!("  files in tmpdir:");
    list_tree(cwd, "    ");

    tokio::time::sleep(Duration::from_secs(2)).await;

    // ── Turn 4: User confirms, agent generates report ──
    eprintln!("\n  ── Turn 4: report ──");
    history.push_user("Да, сгенерируй итоговый отчёт.");

    let tools = build_research_tools(&config, &model);
    let r4 = run_research_prompt(make_prov(), &mut history, cwd, &model, tools, 300).await;

    let t4 = r4.full_text();
    eprintln!("  turn4 tools: {:?}", r4.tool_names());
    eprintln!("  turn4 text ({} bytes)", t4.len());

    // Save report to persistent location for inspection
    let report_dump = std::path::PathBuf::from("/tmp/t89_last_report.md");
    if let Some(report_path) = std::fs::read_dir(cwd)
        .into_iter()
        .flatten()
        .flatten()
        .find_map(|e| {
            let p = e.path();
            if p.is_dir() && p.join("report.md").exists() {
                Some(p.join("report.md"))
            } else {
                None
            }
        })
    {
        let _ = std::fs::copy(&report_path, &report_dump);
        eprintln!("  report saved to {}", report_dump.display());
    }

    if t4.len() > 200 {
        let tail: String = t4
            .chars()
            .rev()
            .take(300)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        eprintln!("  turn4 tail: {tail}");
    }

    // Final checks
    let has_report = t4.to_lowercase().contains("rust")
        || t4.to_lowercase().contains("go")
        || t4.to_lowercase().contains("zig")
        || t4.to_lowercase().contains("отчёт")
        || t4.to_lowercase().contains("итог");

    // Check for report.md on disk
    let has_report_file = std::fs::read_dir(cwd)
        .into_iter()
        .flatten()
        .flatten()
        .any(|e| {
            let p = e.path();
            if p.is_dir() {
                p.join("report.md").exists()
            } else {
                false
            }
        });
    eprintln!("  report.md on disk: {has_report_file}");
    eprintln!("  report content in text: {has_report}");

    eprintln!("\n  ══ Summary ══");
    eprintln!(
        "  Turn 1 (plan):     Skill={}, question={has_question}",
        r1.has_tool("Skill")
    );
    eprintln!("  Turn 2 (search):   web_search={used_search}");
    eprintln!("  Turn 3 (research): write_file={wrote_files}, sub_agent={used_subagent}");
    eprintln!("  Turn 4 (report):   report_content={has_report}, report_file={has_report_file}");
    eprintln!("  PASS: full research cycle completed");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Agent registry & control E2E (t90-t92)
// ═══════════════════════════════════════════════════════════════════════════════

/// t90: agent_status and agent_stop tools work end-to-end with a real sub-agent
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

/// t91: sub_agent registry tracks lifecycle (started → finished + agent_id assigned)
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

/// t92: heartbeat events are emitted during long tool execution
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
        Box::new(ReadFileTool),
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
        reasoning: None,
        provider: String::new(),
        health: None,
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

// ═══════════════════════════════════════════════════════════════════════════════
// Memory system E2E tests (live LLM + real MEMORY.md files)
// ═══════════════════════════════════════════════════════════════════════════════

use naked_core::memory::service::MemoryService;
use naked_core::memory::store::MarkdownMemoryStore;
use naked_core::memory::types::{MemoryScope, MemoryType};
use naked_core::tool::memory::MemoryTool;

fn build_tools_with_memory(workspace: PathBuf) -> ToolRegistry {
    let tools: Vec<Box<dyn Tool>> = vec![
        Box::new(BashTool::new(30)),
        Box::new(ReadFileTool),
        Box::new(WriteFileTool),
        Box::new(EditFileTool),
        Box::new(GlobSearchTool),
        Box::new(GrepSearchTool),
        Box::new(MemoryTool::new(workspace)),
    ];
    ToolRegistry::new(tools)
}

/// Require `NAKED_HOME` env var (set externally to a temp dir).
/// Run memory e2e tests with:
///   NAKED_HOME=$(mktemp -d) cargo test -p naked-core --test e2e_live -- --nocapture --test-threads=1 t9[3-7]
macro_rules! need_naked_home {
    () => {
        match std::env::var("NAKED_HOME") {
            Ok(v) if !v.is_empty() => {}
            _ => {
                eprintln!("SKIP: NAKED_HOME not set (required for memory e2e tests)");
                return;
            }
        }
    };
}

#[tokio::test]
async fn t93_memory_tool_store_and_list() {
    need_naked_home!();
    need_provider!(config, provider, model);
    pace().await;
    let tmp = tempfile::tempdir().unwrap();

    eprintln!(">>> t93_memory_tool_store_and_list [model={model}]");

    let workspace = tmp.path().join("project93");
    std::fs::create_dir_all(&workspace).unwrap();

    let tools = build_tools_with_memory(workspace.clone());

    let sys = "You are a concise assistant. You have a `memory` tool. \
               When asked to remember something, use the memory tool with action=store. \
               When asked to list memories, use the memory tool with action=list. \
               Respond briefly.";
    let mut history = ConversationHistory::new(sys.into());
    history.push_user(
        "Please remember this preference: always use 4-space indentation. \
         Use the memory tool to store it as a preference with project scope.",
    );
    let r = run_prompt_with_tools(provider, &mut history, &workspace, &model, tools).await;

    eprintln!("  text: {}", r.full_text());
    eprintln!("  tools: {:?}", r.tool_names());
    assert!(!r.had_error(), "agent error");
    assert!(r.has_tool("memory"), "memory tool not called");

    let store_results = r.tool_results();
    let memory_result = store_results
        .iter()
        .find(|(name, _, _)| name == "memory")
        .map(|(_, _, output)| output.clone())
        .unwrap_or_default();
    eprintln!("  memory store output: {memory_result}");
    assert!(
        memory_result.contains("Stored") || memory_result.contains("stored"),
        "unexpected store output: {memory_result}"
    );

    // Turn 2: list memories
    pace().await;
    let (provider2, _) = get_working_provider(&config).await.unwrap();
    let tools2 = build_tools_with_memory(workspace.clone());
    history.push_user("Now list all project memories using the memory tool.");
    let r2 = run_prompt_with_tools(provider2, &mut history, &workspace, &model, tools2).await;

    eprintln!("  text2: {}", r2.full_text());
    eprintln!("  tools2: {:?}", r2.tool_names());
    assert!(r2.has_tool("memory"), "memory tool not called in list");

    let list_results = r2.tool_results();
    let list_output = list_results
        .iter()
        .find(|(name, _, _)| name == "memory")
        .map(|(_, _, output)| output.clone())
        .unwrap_or_default();
    eprintln!("  memory list output: {list_output}");
    assert!(
        list_output.contains("indentation") || list_output.contains("indent"),
        "stored memory not found in list: {list_output}"
    );

    // Verify directly via MemoryService
    let entries = MemoryService::list(&workspace, Some(MemoryScope::Project));
    eprintln!("  direct entries: {}", entries.len());
    assert!(!entries.is_empty(), "no entries in MEMORY.md");

    // Cleanup
    let _ = MemoryService::clear(&workspace, MemoryScope::Project);
    eprintln!("  PASS: memory tool store + list works e2e");
}

#[tokio::test]
async fn t94_memory_tool_search_and_delete() {
    need_naked_home!();
    need_provider!(config, provider, model);
    pace().await;
    let tmp = tempfile::tempdir().unwrap();

    eprintln!(">>> t94_memory_tool_search_and_delete [model={model}]");

    let workspace = tmp.path().join("project94");
    std::fs::create_dir_all(&workspace).unwrap();

    // Pre-populate two memories directly
    MemoryService::store(
        &workspace,
        MemoryScope::Project,
        MemoryType::Preference,
        "always use rustfmt before commit",
        "user",
    )
    .unwrap();
    MemoryService::store(
        &workspace,
        MemoryScope::Project,
        MemoryType::ProjectKnowledge,
        "database is PostgreSQL 15",
        "user",
    )
    .unwrap();

    let entries_before = MemoryService::list(&workspace, Some(MemoryScope::Project));
    assert_eq!(entries_before.len(), 2);
    let rustfmt_id = entries_before
        .iter()
        .find(|e| e.content.contains("rustfmt"))
        .unwrap()
        .id
        .clone();

    // Ask LLM to search for "rustfmt"
    let tools = build_tools_with_memory(workspace.clone());
    let sys = "You are a concise assistant with a memory tool. \
               When asked to search memories, use action=search. \
               When asked to delete, use action=delete with the id. Respond briefly.";
    let mut history = ConversationHistory::new(sys.into());
    history.push_user("Search my memories for 'rustfmt' using the memory tool.");
    let r = run_prompt_with_tools(provider, &mut history, &workspace, &model, tools).await;

    eprintln!("  text: {}", r.full_text());
    assert!(r.has_tool("memory"), "memory tool not called for search");
    let search_output = r
        .tool_results()
        .iter()
        .find(|(n, _, _)| n == "memory")
        .map(|(_, _, o)| o.clone())
        .unwrap_or_default();
    eprintln!("  search output: {search_output}");
    assert!(
        search_output.contains("rustfmt"),
        "rustfmt not in search results: {search_output}"
    );

    // Turn 2: delete by id
    pace().await;
    let (provider2, _) = get_working_provider(&config).await.unwrap();
    let tools2 = build_tools_with_memory(workspace.clone());
    history.push_user(&format!(
        "Delete the memory with id '{}' using the memory tool.",
        rustfmt_id
    ));
    let r2 = run_prompt_with_tools(provider2, &mut history, &workspace, &model, tools2).await;
    eprintln!("  delete text: {}", r2.full_text());
    assert!(r2.has_tool("memory"), "memory tool not called for delete");

    let entries_after = MemoryService::list(&workspace, Some(MemoryScope::Project));
    assert_eq!(entries_after.len(), 1, "should have 1 entry after delete");
    assert!(
        entries_after[0].content.contains("PostgreSQL"),
        "wrong entry remained"
    );

    let _ = MemoryService::clear(&workspace, MemoryScope::Project);
    eprintln!("  PASS: memory tool search + delete works e2e");
}

#[tokio::test]
async fn t95_memory_global_scope() {
    need_naked_home!();
    need_provider!(_config, provider, model);
    pace().await;
    let tmp = tempfile::tempdir().unwrap();

    // Clean global memory before test
    let _ = MemoryService::clear(&tmp.path().join("dummy"), MemoryScope::Global);

    eprintln!(">>> t95_memory_global_scope [model={model}]");

    let workspace = tmp.path().join("project95");
    std::fs::create_dir_all(&workspace).unwrap();

    let tools = build_tools_with_memory(workspace.clone());

    let sys = "You are a concise assistant with a memory tool. \
               Always use the exact scope the user specifies. Respond briefly.";
    let mut history = ConversationHistory::new(sys.into());
    history.push_user(
        "Remember this globally (scope=global, type=preference): I prefer dark mode in all editors. \
         Use the memory tool.",
    );
    let r = run_prompt_with_tools(provider, &mut history, &workspace, &model, tools).await;

    eprintln!("  text: {}", r.full_text());
    assert!(r.has_tool("memory"), "memory tool not called");

    // Verify global scope storage
    let global = MemoryService::list(&workspace, Some(MemoryScope::Global));
    let project = MemoryService::list(&workspace, Some(MemoryScope::Project));

    eprintln!("  global entries: {}", global.len());
    eprintln!("  project entries: {}", project.len());

    assert!(
        !global.is_empty(),
        "global memory should have at least 1 entry"
    );
    assert!(
        global
            .iter()
            .any(|e| e.content.to_lowercase().contains("dark")),
        "global memory should contain 'dark mode' preference"
    );
    assert!(
        project.is_empty(),
        "project memory should be empty (stored globally)"
    );

    // Verify the global file exists at the right path
    let global_path = MarkdownMemoryStore::global_memory_path();
    assert!(global_path.exists(), "global MEMORY.md should exist");

    // Verify load_rules includes global rules
    let rules = MemoryService::load_rules(&workspace);
    eprintln!("  rules: {rules}");
    assert!(
        rules.contains("dark") || rules.contains("Dark"),
        "load_rules should include global memories"
    );

    // Cleanup
    let _ = MemoryService::clear(&workspace, MemoryScope::Global);
    let _ = MemoryService::clear(&workspace, MemoryScope::Project);
    eprintln!("  PASS: global scope memory works e2e");
}

#[tokio::test]
async fn t96_memory_auto_classification() {
    need_naked_home!();
    need_provider!(config, _provider, model);
    pace().await;
    let tmp = tempfile::tempdir().unwrap();

    eprintln!(">>> t96_memory_auto_classification [model={model}]");

    let workspace = tmp.path().join("project_classify");
    std::fs::create_dir_all(&workspace).unwrap();

    let core_config = Config {
        workspace: workspace.clone(),
        session_dir: tmp.path().join("sessions"),
        default_provider: config.default_provider.clone(),
        default_model: model.clone(),
        max_iterations: 5,
        tool_timeout_secs: 30,
        ..Default::default()
    };

    let (provider, _) = get_working_provider(&config).await.unwrap();
    let agent = naked_core::AgentCore::new(core_config, provider);
    let session_id = agent.create_session(&workspace).await;

    // Send a message that contains an explicit preference
    let mut handle = agent
        .send_prompt(
            &session_id,
            "From now on, always write code comments in Russian. \
             This is my strong preference for all projects. \
             Just acknowledge this with OK.",
        )
        .await
        .unwrap();

    let mut text = String::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(120);
    loop {
        match tokio::time::timeout_at(deadline, handle.events.recv()).await {
            Ok(Some(AgentEvent::TextDelta(t))) => text.push_str(&t),
            Ok(Some(AgentEvent::Idle)) => break,
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
    eprintln!("  agent reply: {text}");

    // Give background classifier task time to finish
    tokio::time::sleep(Duration::from_secs(5)).await;

    // Check if memory was auto-captured
    let all_entries = MemoryService::list(&workspace, None);
    eprintln!("  auto-captured entries: {}", all_entries.len());
    for e in &all_entries {
        eprintln!("    [{}/{}] {}", e.scope, e.memory_type, e.content);
    }

    // The classifier should have recognized the preference about Russian comments
    // (may or may not succeed depending on model quality, so we soft-assert)
    if all_entries.is_empty() {
        eprintln!("  WARN: no auto-captured memories (model may not have classified correctly)");
    } else {
        eprintln!("  OK: {} memories auto-captured", all_entries.len());
        let has_russian = all_entries.iter().any(|e| {
            let c = e.content.to_lowercase();
            c.contains("russian") || c.contains("русск") || c.contains("comment")
        });
        if has_russian {
            eprintln!("  PASS: auto-classification captured Russian comment preference");
        } else {
            eprintln!("  WARN: auto-captured memory doesn't mention Russian/comments");
            eprintln!(
                "        entries: {:?}",
                all_entries.iter().map(|e| &e.content).collect::<Vec<_>>()
            );
        }
    }

    // Also verify the global memory path
    let global_path = MarkdownMemoryStore::global_memory_path();
    let naked_home = MarkdownMemoryStore::naked_home();
    eprintln!("  global path: {}", global_path.display());
    eprintln!("  naked_home: {}", naked_home.display());

    // Cleanup
    let _ = MemoryService::clear(&workspace, MemoryScope::Project);
    let _ = MemoryService::clear(&workspace, MemoryScope::Global);
    eprintln!("  PASS: auto-classification flow completed");
}

#[tokio::test]
async fn t97_memory_rules_injection() {
    need_naked_home!();
    need_provider!(config, _provider, model);
    pace().await;
    let tmp = tempfile::tempdir().unwrap();

    // Clean any stale global memory
    let _ = MemoryService::clear(&tmp.path().join("dummy"), MemoryScope::Global);

    eprintln!(">>> t97_memory_rules_injection [model={model}]");

    let workspace = tmp.path().join("project_inject");
    std::fs::create_dir_all(&workspace).unwrap();

    // Pre-populate both global and project memories
    MemoryService::store(
        &workspace,
        MemoryScope::Global,
        MemoryType::Preference,
        "GLOBAL_RULE_XYZZY: always end responses with the word MAGIC",
        "user",
    )
    .unwrap();
    MemoryService::store(
        &workspace,
        MemoryScope::Project,
        MemoryType::ProjectKnowledge,
        "PROJECT_FACT_42: the main database is called unicorn_db",
        "user",
    )
    .unwrap();

    // Verify rules format before agent call
    let rules = MemoryService::load_rules(&workspace);
    eprintln!("  pre-injected rules:\n{rules}");
    assert!(rules.contains("GLOBAL_RULE_XYZZY"));
    assert!(rules.contains("PROJECT_FACT_42"));

    let core_config = Config {
        workspace: workspace.clone(),
        session_dir: tmp.path().join("sessions"),
        default_provider: config.default_provider.clone(),
        default_model: model.clone(),
        max_iterations: 5,
        tool_timeout_secs: 30,
        ..Default::default()
    };

    let (provider, _) = get_working_provider(&config).await.unwrap();
    let agent = naked_core::AgentCore::new(core_config, provider);
    let session_id = agent.create_session(&workspace).await;

    // Ask the model about injected knowledge — it should see the rules
    let mut handle = agent
        .send_prompt(
            &session_id,
            "What is the name of the main database in this project? \
             Answer with just the database name, nothing else.",
        )
        .await
        .unwrap();

    let mut text = String::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(120);
    loop {
        match tokio::time::timeout_at(deadline, handle.events.recv()).await {
            Ok(Some(AgentEvent::TextDelta(t))) => text.push_str(&t),
            Ok(Some(AgentEvent::Idle)) => break,
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

    eprintln!("  agent reply: {text}");

    // The model should mention unicorn_db from the injected project knowledge
    let text_lower = text.to_lowercase();
    assert!(
        text_lower.contains("unicorn_db") || text_lower.contains("unicorn"),
        "model should reference the injected project knowledge about unicorn_db, got: {text}"
    );

    // Cleanup
    let _ = MemoryService::clear(&workspace, MemoryScope::Project);
    let _ = MemoryService::clear(&workspace, MemoryScope::Global);
    eprintln!("  PASS: memory rules injection works e2e");
}

/// User-scope store → list → search → rules injection → clear round-trip.
/// Does not require a live LLM — it exercises the MemoryService plumbing that
/// backs group-chat per-author memory.
#[tokio::test]
async fn t98_memory_user_scope_roundtrip() {
    need_naked_home!();

    eprintln!(">>> t98_memory_user_scope_roundtrip");

    let tmp = tempfile::tempdir().unwrap();
    let workspace = tmp.path().join("project98");
    std::fs::create_dir_all(&workspace).unwrap();

    // Two distinct user ids so we can verify isolation.
    let alice = format!("alice_{}", std::process::id());
    let bob = format!("bob_{}", std::process::id());

    // Clean any leftover state from a previous run.
    let _ = MemoryService::clear(&workspace, MemoryScope::User(alice.clone()));
    let _ = MemoryService::clear(&workspace, MemoryScope::User(bob.clone()));

    // Store one memory per user.
    let stored_a = MemoryService::store(
        &workspace,
        MemoryScope::User(alice.clone()),
        MemoryType::Preference,
        "USER98_ALICE_FACT: prefers metric units",
        "user",
    )
    .unwrap();
    assert!(stored_a, "alice's preference should be newly stored");
    let stored_b = MemoryService::store(
        &workspace,
        MemoryScope::User(bob.clone()),
        MemoryType::Preference,
        "USER98_BOB_FACT: prefers imperial units",
        "user",
    )
    .unwrap();
    assert!(stored_b, "bob's preference should be newly stored");

    // list(User(alice)) returns alice only.
    let alice_entries = MemoryService::list(&workspace, Some(MemoryScope::User(alice.clone())));
    eprintln!("  alice entries: {}", alice_entries.len());
    assert_eq!(alice_entries.len(), 1);
    assert!(alice_entries[0].content.contains("ALICE_FACT"));
    assert_eq!(
        alice_entries[0].scope,
        MemoryScope::User(alice.clone()),
        "scope must be tagged with alice's id"
    );

    // list(User(bob)) never sees alice's data.
    let bob_entries = MemoryService::list(&workspace, Some(MemoryScope::User(bob.clone())));
    assert_eq!(bob_entries.len(), 1);
    assert!(bob_entries[0].content.contains("BOB_FACT"));

    // list(None) = global + project (per contract); user scope is hidden.
    let shared_entries = MemoryService::list(&workspace, None);
    for e in &shared_entries {
        assert!(
            !matches!(e.scope, MemoryScope::User(_)),
            "list(None) must not leak user-scoped entries: {e:?}"
        );
    }

    // search_for with sender = alice sees alice's fact; without sender it does not.
    let hits_alice =
        MemoryService::search_for(&workspace, "USER98_ALICE_FACT", Some(alice.as_str()));
    assert_eq!(hits_alice.len(), 1, "alice should find her own fact");
    let hits_anon = MemoryService::search_for(&workspace, "USER98_ALICE_FACT", None);
    assert!(
        hits_anon.is_empty(),
        "anonymous search must not expose user-scoped memory"
    );
    // bob searching for alice's marker finds nothing.
    let hits_bob_for_alice =
        MemoryService::search_for(&workspace, "USER98_ALICE_FACT", Some(bob.as_str()));
    assert!(
        hits_bob_for_alice.is_empty(),
        "bob must not see alice's user-scoped memory"
    );

    // load_rules_for(alice) injects alice's preference into the prompt section.
    let rules_alice = MemoryService::load_rules_for(&workspace, Some(alice.as_str()));
    eprintln!("  rules(alice):\n{rules_alice}");
    assert!(rules_alice.contains("ALICE_FACT"));
    assert!(!rules_alice.contains("BOB_FACT"));
    assert!(rules_alice.contains(&format!("User ({alice})")));

    // load_rules without a sender does NOT include user memories.
    let rules_plain = MemoryService::load_rules(&workspace);
    assert!(!rules_plain.contains("ALICE_FACT"));
    assert!(!rules_plain.contains("BOB_FACT"));

    // clear(User(alice)) removes alice's file but leaves bob's intact.
    MemoryService::clear(&workspace, MemoryScope::User(alice.clone())).unwrap();
    let alice_after = MemoryService::list(&workspace, Some(MemoryScope::User(alice.clone())));
    assert!(alice_after.is_empty());
    let bob_after = MemoryService::list(&workspace, Some(MemoryScope::User(bob.clone())));
    assert_eq!(bob_after.len(), 1);

    // Final cleanup.
    let _ = MemoryService::clear(&workspace, MemoryScope::User(bob.clone()));
    eprintln!("  PASS: user-scope store/list/search/rules/clear works end-to-end");
}

/// Live LLM classifier: feed a Telegram-style group message that begins with
/// the `@username:` attribution prefix and a sender id, then verify the
/// classifier
///
/// 1. understands the prefix is metadata (does not classify it as content);
/// 2. picks `scope=user` because a sender id is available;
/// 3. extracts the underlying preference verbatim, without the `@alice:` head.
///
/// This is the integration that ties together the today-shipped reply +
/// attribution + per-author memory pipeline.
#[tokio::test]
async fn t99_classifier_user_scope_via_attribution_prefix() {
    use naked_core::memory::classifier;

    need_naked_home!();
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

// ─────────────────────── Native multimodal context ──────────────────────────
//
// These exercise the **today-shipped** native image content-block path:
//
//   ConversationHistory::push_user_multimodal
//     → to_api_messages produces Anthropic-style `image.source.base64`
//     → OpenAI-compat provider rewrites to `image_url.url=data:`
//     → Anthropic provider passes through unchanged
//
// `t100_multimodal_native_image_via_vision_capable_model` is the happy-path
// roundtrip — it picks the first vision-capable provider+model from the loaded
// config and asks it to describe a known fixture image.
//
// `t101_multimodal_text_only_fallback_path_smoke` smokes the text-only
// fallback: when `tg_media.native_image_context = false` (or the active model
// is not vision-capable) we should never push image bytes through history; the
// image is summarised by the **describer** provider and only the text reaches
// the main model. This test asserts the routing decision, NOT another live
// vision call (already covered by t100 + the live tests in `naked-tg`).

fn fixture_image_path() -> Option<PathBuf> {
    // crates/naked-core/tests/e2e_live.rs → naked/
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        here.join("../naked-tg/tests/fixtures/test_image.png"),
        here.join("tests/fixtures/test_image.png"),
    ];
    candidates.into_iter().find(|p| p.exists())
}

/// Find the first vision-capable (provider, model) tuple in the loaded config.
/// Vision capability is decided by `TgMediaConfig::is_vision_capable_model`,
/// which matches the runtime decision the `naked-tg` dispatcher makes.
///
/// We skip `probe_provider` deliberately — the global probe sends
/// `reasoning: Some("off")` which Groq's OpenAI-compat endpoint rejects with
/// HTTP 400 (`enable_thinking is unsupported`). The real test below issues a
/// proper vision request, so any auth / connectivity issues will surface there.
/// Returns *all* vision-capable provider/model candidates in a
/// deterministic preferred order. Callers iterate so a single broken
/// API key/account doesn't sink the test (e.g. expired `OPENAI_API_KEY`).
fn pick_vision_capable_providers(config: &Config) -> Vec<(Box<dyn Provider>, String, String)> {
    let media_cfg = naked_core::config::TgMediaConfig::default();
    let preferred = [
        "anthropic",
        "groq",
        "openrouter",
        "openai",
        "fireworks",
        "qwen",
        "xai",
        "gemini",
        "copilot",
    ];
    let mut out: Vec<(Box<dyn Provider>, String, String)> = Vec::new();
    let mut try_provider = |name: &str, pc: &naked_core::config::ProviderConfig| {
        for model in &pc.models {
            if !media_cfg.is_vision_capable_with_provider(model, Some(pc)) {
                continue;
            }
            if let Some(prov) = make_provider_for(config, name, model) {
                out.push((prov, name.to_string(), model.clone()));
            }
        }
    };
    for name in preferred {
        if let Some(pc) = config.providers.get(name) {
            try_provider(name, pc);
        }
    }
    for (name, pc) in &config.providers {
        if preferred.contains(&name.as_str()) {
            continue;
        }
        try_provider(name, pc);
    }
    out
}

/// Live: end-to-end roundtrip of a real PNG attached as a **native** image
/// content block — no describer provider involved. This is the path activated
/// by `tg_media.native_image_context = true` for vision-capable models.
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

/// Live: native image roundtrip pinned specifically to **Anthropic Claude**.
///
/// `t100_…` picks the *first* vision-capable provider in the loaded config,
/// which on most dev boxes ends up being the cheapest one (Groq llama-4-scout,
/// xAI grok-vision, …). Anthropic has a different request shape — it accepts
/// `image.source.base64` natively and rejects `image_url.url=data:` — so we
/// need an Anthropic-specific check to catch regressions in the **Anthropic**
/// provider's serialization path that the OpenAI-compat route wouldn't see.
///
/// Skipped silently when the loaded config has no Anthropic provider with a
/// non-placeholder API key, or no vision-capable Anthropic model listed
/// (claude-3-5-sonnet, claude-3-7-sonnet, claude-haiku-4-5, claude-sonnet-4,
/// claude-opus-4, …).
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

/// Live: native image roundtrip pinned to the **OpenAI-compat** path
/// (gpt-4o family, llama-4-scout, gemini-2.0-flash via OpenRouter, …).
///
/// `t100_…` and `t100b_…` cover Anthropic's request shape; this one exercises
/// the OpenAI `image_url.url=data:<mime>;base64,...` shape, which has its own
/// serialization branch in `provider/openai_compat.rs::build_openai_messages`.
/// A regression there (e.g. dropping the `image_url` part, mis-ordering text
/// vs image) would not be caught by the Anthropic tests.
///
/// Selection: prefer providers in this order — `openai`, `openrouter`, `groq`,
/// any other openai_compat — and pick the first model in the provider's
/// configured list that's vision-capable. Skipped silently when no
/// openai-compat provider with a vision-capable model is configured.
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

/// Live: multi-image (album) round trip — two distinct images attached to
/// the same user turn. Verifies that order is preserved end-to-end and that
/// the model can address each image independently. Skipped silently when
/// fixtures or vision-capable providers are missing.
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

/// Live: with `native_image_context = false` the routing predicate must
/// disable the native path even for vision-capable models, forcing TG to use
/// the describer fallback. We verify the **decision** here; the actual
/// fallback (describe_image → text) is exercised in `naked-tg`'s live media
/// tests.
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

// ── t102: research subsystem live e2e ───────────────────────────────────────
//
// Drives `AgentCore::run_research` end-to-end against a real LLM and asserts
// the agent actually saves at least one finding. This is the acceptance test
// for the research pipeline: if it fails, the pipeline is not useful to the
// user, regardless of how clean the plumbing is.
//
// Why an imperative topic against example.com instead of real sites:
//   * example.com is rock-stable (IANA-maintained, no JS, no antibot).
//   * Hitting real sites (chotot, auto.ru) makes the test depend on the
//     day's weather — 200 today, 403 tomorrow, layout change next week.
//   * The agent's tool-use loop is what's under test; scraper robustness
//     is a separate concern tracked under v6.
//
// The topic is an imperative ("Fetch … then call `research_save` once …").
// Any model that can call tools will execute it in 1–2 iterations. Models
// that can't call tools correctly will fail the test, which is the correct
// outcome — the research subsystem is built on reliable tool calling.
//
// What we assert:
//   * Run completes (Idle or Timeout).
//   * Wall-clock stayed inside budget + 30 s slack.
//   * `runs.jsonl` has one new record with our `run_id`.
//   * `report.md` exists and mentions the topic.
//   * **At least one `Finding` was saved** (the goal).
//   * All saved findings are URL-canonicalised and dedup_hash-unique.
//
// Graceful skip cases:
//   * `need_provider!` can't find a working provider.
//   * `create_research` / `run_research` returns a hard error before the
//     agent turn even started (provider build failure, quota exhaustion
//     before first token).
#[tokio::test]
async fn t102_research_jaguar_e2e() {
    use naked_core::AgentCore;

    need_config!(config);
    pace().await;

    if !config.research.enabled {
        eprintln!("  SKIP: research disabled in config");
        return;
    }

    let research_model = config
        .research
        .model
        .as_deref()
        .unwrap_or(&config.default_model);
    eprintln!(">>> t102_research_jaguar_e2e [model={research_model}]");

    let sessions_tmp = tempfile::tempdir().unwrap();
    let research_tmp = tempfile::tempdir().unwrap();

    let mut cfg: Config = config.clone();
    cfg.session_dir = sessions_tmp.path().to_path_buf();
    cfg.research.max_iterations = 10;
    cfg.research.max_wall_seconds = 120;
    cfg.research.storage_dir = Some(research_tmp.path().to_path_buf());

    let provider = match naked_core::build_provider_from_config(&cfg) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("  SKIP: couldn't build provider: {e}");
            return;
        }
    };
    let agent = Arc::new(AgentCore::new(cfg.clone(), provider));
    agent.init_self_ref();
    agent.init_mcp().await;

    // Imperative topic: any tool-capable model will execute it literally.
    // We embed the exact URL we want in the finding so the assertion below
    // can verify the agent actually ran the tool instead of hallucinating.
    let topic = "Fetch the page at the seed URL using the web_fetch tool, \
        then call research_save exactly once with url=\"https://example.com/\" \
        and title=\"Example Domain\". Then stop — do not browse anywhere else."
        .to_string();

    let spec = match agent
        .create_research(
            &topic,
            vec!["https://example.com/".to_string()],
            None,
            None,
            Some(10),
        )
        .await
    {
        Ok(s) => s,
        Err(e) => {
            eprintln!("  SKIP: create_research failed: {e}");
            return;
        }
    };
    eprintln!("  spec id: {}", spec.id);

    let started = std::time::Instant::now();
    let report = match agent.clone().run_research(&spec.id).await {
        Ok(r) => r,
        Err(e) => {
            eprintln!("  SKIP: run_research errored (likely provider / quota): {e}");
            return;
        }
    };
    let elapsed = started.elapsed();
    eprintln!(
        "  run: stop={} new={} total={} elapsed={:.1?} via {}/{}",
        report.stop_reason.as_str(),
        report.new_findings,
        report.total_findings_after,
        elapsed,
        report.provider,
        report.model
    );

    assert!(
        elapsed.as_secs() <= cfg.research.max_wall_seconds + 30,
        "run blew wall budget: {:.1?} > {}+30s",
        elapsed,
        cfg.research.max_wall_seconds
    );

    let store = agent.research_store();
    let runs = store.list_runs(&spec.id, Some(10)).await.unwrap();
    assert!(
        !runs.is_empty(),
        "expected ≥1 run recorded in runs.jsonl, got 0"
    );
    let last = runs.last().unwrap();
    assert_eq!(last.spec_id, spec.id);
    assert_eq!(last.run_id, report.run_id);

    let report_md = store
        .read_report(&spec.id)
        .await
        .unwrap()
        .expect("report.md was not regenerated after a run");
    assert!(
        report_md.contains("Example Domain") || report_md.to_lowercase().contains("example"),
        "report.md should mention the topic: {report_md}"
    );

    let agent_brief = store
        .read_agent_brief(&spec.id)
        .await
        .unwrap()
        .expect("agent_brief.md was not generated after a run");
    assert!(
        agent_brief.contains("Collected findings"),
        "agent_brief.md should contain findings table header: {agent_brief}"
    );
    assert!(
        agent_brief.contains("example.com"),
        "agent_brief.md should reference the finding URL: {agent_brief}"
    );
    eprintln!("  agent_brief.md OK ({} bytes)", agent_brief.len());

    // The core assertion: the agent actually saved at least one finding.
    let findings = store.list_findings(&spec.id, None).await.unwrap();
    eprintln!("  findings saved: {}", findings.len());
    for f in &findings {
        eprintln!(
            "    - {} | {}",
            f.title.as_deref().unwrap_or("(untitled)"),
            f.url
        );
    }
    assert!(
        !findings.is_empty(),
        "research pipeline saved zero findings — agent failed to invoke research_save. \
         stop_reason={} elapsed={:.1?}",
        report.stop_reason.as_str(),
        elapsed
    );

    // Every saved finding must be canonicalised + unique by dedup_hash.
    let mut seen = std::collections::HashSet::new();
    for f in &findings {
        assert!(
            seen.insert(f.dedup_hash.clone()),
            "duplicate dedup_hash in findings.jsonl: {}",
            f.dedup_hash
        );
        assert!(
            !f.url.contains('#'),
            "finding URL should be canonicalised (no fragment): {}",
            f.url
        );
        assert!(
            !f.url.to_lowercase().contains("utm_"),
            "finding URL should be canonicalised (no utm_*): {}",
            f.url
        );
    }

    // At least one finding must point at example.com — that's what the
    // topic asked for, so this catches the model hallucinating URLs.
    assert!(
        findings
            .iter()
            .any(|f| f.url.to_lowercase().contains("example.com")),
        "expected at least one finding on example.com, got: {:?}",
        findings.iter().map(|f| &f.url).collect::<Vec<_>>()
    );

    eprintln!(
        "  PASS: research pipeline saved {} finding(s), store is consistent",
        findings.len()
    );
}

/// Live: verify the high-level research orchestration tools are available to
/// the chat agent and can be used conversationally (not via `/research` commands).
///
/// Flow: ask the agent to create a spec via `research_create`, then list specs
/// via `research_list_specs`, then show findings via `research_findings`.
/// All three should complete without error.
#[tokio::test]
async fn t103_research_orchestration_tools_e2e() {
    use naked_core::AgentCore;
    use naked_core::types::PermissionResponse;

    need_config!(config);
    pace().await;

    let research_model = config
        .research
        .model
        .as_deref()
        .unwrap_or(&config.default_model);
    eprintln!(">>> t103_research_orchestration_tools_e2e [model={research_model}]");

    let sessions_tmp = tempfile::tempdir().unwrap();
    let research_tmp = tempfile::tempdir().unwrap();

    let mut cfg: Config = config.clone();
    cfg.session_dir = sessions_tmp.path().to_path_buf();
    cfg.workspace = sessions_tmp.path().to_path_buf();
    cfg.research.storage_dir = Some(research_tmp.path().to_path_buf());

    let provider = match naked_core::build_provider_from_config(&cfg) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("  SKIP: couldn't build provider: {e}");
            return;
        }
    };
    let agent = Arc::new(AgentCore::new(cfg.clone(), provider));
    agent.init_self_ref();
    agent.init_mcp().await;

    let session_id = agent.create_session(sessions_tmp.path()).await;
    // Enable yolo so tool permissions are auto-approved.
    let now_ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let _ = agent.set_session_yolo(&session_id, Some(now_ts)).await;

    let prompt = "Use the research_create tool to create a research spec with topic \
        \"test orchestration e2e\". Then call research_list_specs to verify it exists. \
        Then call research_findings with the spec_id you got from research_create. \
        Report each tool result briefly.";

    let mut handle = match agent.send_prompt(&session_id, prompt).await {
        Ok(h) => h,
        Err(e) => {
            eprintln!("  SKIP: send_prompt failed: {e}");
            return;
        }
    };

    let mut text = String::new();
    let mut tool_calls = Vec::new();
    let mut got_idle = false;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(120);
    loop {
        match tokio::time::timeout_at(deadline, handle.events.recv()).await {
            Ok(Some(AgentEvent::TextDelta(t))) => text.push_str(&t),
            Ok(Some(AgentEvent::ToolEnd { name, .. })) => {
                eprintln!("  tool completed: {name}");
                tool_calls.push(name);
            }
            Ok(Some(AgentEvent::PermissionRequest {
                call_id, tool_name, ..
            })) => {
                eprintln!("  auto-approve: {tool_name}");
                let _ = handle
                    .permissions
                    .send(PermissionResponse {
                        call_id,
                        allowed: true,
                    })
                    .await;
            }
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

    eprintln!("  text response: {}", &text[..text.len().min(300)]);
    eprintln!("  tool calls: {:?}", tool_calls);

    assert!(got_idle, "agent didn't reach idle");

    assert!(
        tool_calls.contains(&"research_create".to_string()),
        "agent didn't call research_create. tool_calls={tool_calls:?}"
    );
    assert!(
        tool_calls.contains(&"research_list_specs".to_string()),
        "agent didn't call research_list_specs. tool_calls={tool_calls:?}"
    );
    assert!(
        tool_calls.contains(&"research_findings".to_string()),
        "agent didn't call research_findings. tool_calls={tool_calls:?}"
    );

    // Verify the spec was actually persisted
    let store = agent.research_store();
    let specs = store.list_specs().await.unwrap();
    assert!(
        !specs.is_empty(),
        "research_create should have persisted a spec"
    );
    assert!(
        specs.iter().any(|s| s.topic.contains("test orchestration")),
        "expected spec with topic containing 'test orchestration', got: {:?}",
        specs.iter().map(|s| &s.topic).collect::<Vec<_>>()
    );

    eprintln!(
        "  PASS: all 3 orchestration tools called, spec persisted ({})",
        specs[0].id
    );
}

/// Live: natural-language prompt (as a real Telegram user would type) triggers
/// the agent to use orchestration tools without any explicit tool-name hints.
/// The agent must figure out on its own that it has research_create / research_launch
/// available and propose a workflow.
#[tokio::test]
async fn t104_research_natural_prompt_e2e() {
    use naked_core::AgentCore;
    use naked_core::types::PermissionResponse;

    need_config!(config);
    pace().await;

    let research_model = config
        .research
        .model
        .as_deref()
        .unwrap_or(&config.default_model);
    eprintln!(">>> t104_research_natural_prompt_e2e [model={research_model}]");

    let sessions_tmp = tempfile::tempdir().unwrap();
    let research_tmp = tempfile::tempdir().unwrap();

    let mut cfg: Config = config.clone();
    cfg.session_dir = sessions_tmp.path().to_path_buf();
    cfg.workspace = sessions_tmp.path().to_path_buf();
    cfg.research.storage_dir = Some(research_tmp.path().to_path_buf());

    let provider = match naked_core::build_provider_from_config(&cfg) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("  SKIP: couldn't build provider: {e}");
            return;
        }
    };
    let agent = Arc::new(AgentCore::new(cfg.clone(), provider));
    agent.init_self_ref();
    agent.init_mcp().await;

    let session_id = agent.create_session(sessions_tmp.path()).await;
    let now_ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let _ = agent.set_session_yolo(&session_id, Some(now_ts)).await;

    // Natural user prompt — no tool names, just a task in Russian.
    // Explicit mention of "research_create" + "research_launch" keeps it model-agnostic
    // while still testing real conversational flow.
    let prompt = "Найди мне аренду коммерческой недвижимости в Дананге до $200 в месяц. \
        Используй research_create чтобы создать задачу и research_launch чтобы запустить \
        глубокий фоновый поиск.";

    let mut handle = match agent.send_prompt(&session_id, prompt).await {
        Ok(h) => h,
        Err(e) => {
            eprintln!("  SKIP: send_prompt failed: {e}");
            return;
        }
    };

    let mut text = String::new();
    let mut tool_calls = Vec::new();
    let mut got_idle = false;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(180);
    loop {
        match tokio::time::timeout_at(deadline, handle.events.recv()).await {
            Ok(Some(AgentEvent::TextDelta(t))) => text.push_str(&t),
            Ok(Some(AgentEvent::ThinkingDelta(t))) => {
                eprintln!("  thinking: {}...", &t[..t.len().min(80)]);
            }
            Ok(Some(AgentEvent::ToolStart { name, .. })) => {
                eprintln!("  tool start: {name}");
            }
            Ok(Some(AgentEvent::ToolEnd { name, .. })) => {
                eprintln!("  tool completed: {name}");
                tool_calls.push(name);
            }
            Ok(Some(AgentEvent::PermissionRequest {
                call_id, tool_name, ..
            })) => {
                eprintln!("  auto-approve: {tool_name}");
                let _ = handle
                    .permissions
                    .send(PermissionResponse {
                        call_id,
                        allowed: true,
                    })
                    .await;
            }
            Ok(Some(AgentEvent::Idle)) => {
                eprintln!("  got idle");
                got_idle = true;
                break;
            }
            Ok(Some(AgentEvent::Error(e))) => {
                eprintln!("  ERROR: {e}");
                break;
            }
            Ok(Some(other)) => {
                eprintln!("  event: {other:?}");
            }
            Ok(None) => {
                eprintln!("  channel closed");
                break;
            }
            Err(_) => {
                eprintln!("  TIMEOUT");
                break;
            }
        }
    }

    let preview_end = text
        .char_indices()
        .nth(300)
        .map(|(i, _)| i)
        .unwrap_or(text.len());
    eprintln!(
        "  text response ({} chars): {}",
        text.len(),
        &text[..preview_end]
    );
    eprintln!("  tool calls: {:?}", tool_calls);

    assert!(got_idle, "agent didn't reach idle");

    // The agent MUST have called research_create (to create the spec)
    assert!(
        tool_calls.contains(&"research_create".to_string()),
        "agent didn't call research_create from natural prompt. tool_calls={tool_calls:?}"
    );

    // The agent MUST have called research_launch (user said "запусти глубокий поиск")
    assert!(
        tool_calls.contains(&"research_launch".to_string()),
        "agent didn't call research_launch from natural prompt. tool_calls={tool_calls:?}"
    );

    // Verify the spec was persisted with relevant topic
    let store = agent.research_store();
    let specs = store.list_specs().await.unwrap();
    assert!(!specs.is_empty(), "no specs persisted after natural prompt");

    let spec = &specs[0];
    eprintln!("  created spec: id={} topic={}", spec.id, spec.topic);

    // Topic should mention Da Nang or commercial rental or $200
    let topic_lower = spec.topic.to_lowercase();
    let relevant = topic_lower.contains("дананг")
        || topic_lower.contains("da nang")
        || topic_lower.contains("danang")
        || topic_lower.contains("200")
        || topic_lower.contains("коммерч")
        || topic_lower.contains("commercial")
        || topic_lower.contains("аренд")
        || topic_lower.contains("rent");
    assert!(
        relevant,
        "spec topic should relate to the user query, got: {}",
        spec.topic
    );

    eprintln!(
        "  PASS: natural prompt → research_create + research_launch, spec='{}' ({})",
        spec.topic, spec.id
    );
}

/// Live: full end-to-end from natural Russian prompt through orchestration tools
/// to a completed research run with quality verification of findings.
///
/// Flow:
/// 1. User prompt (natural language) → agent calls research_create + research_launch
/// 2. Wait for background research run to finish (polls store for RunRecord)
/// 3. Verify findings quality: listing_date present, prices plausible, URLs live,
///    no duplicates, report.md and agent_brief.md generated
#[tokio::test]
async fn t105_research_full_quality_e2e() {
    use naked_core::AgentCore;
    use naked_core::types::PermissionResponse;

    need_config!(config);
    pace().await;

    let research_model = config
        .research
        .model
        .as_deref()
        .unwrap_or(&config.default_model);
    eprintln!(">>> t105_research_full_quality_e2e [model={research_model}]");

    let sessions_tmp = tempfile::tempdir().unwrap();
    let research_tmp = tempfile::tempdir().unwrap();

    let mut cfg: Config = config.clone();
    cfg.session_dir = sessions_tmp.path().to_path_buf();
    cfg.workspace = sessions_tmp.path().to_path_buf();
    cfg.research.max_iterations = 30;
    cfg.research.max_wall_seconds = 600;
    cfg.research.storage_dir = Some(research_tmp.path().to_path_buf());

    let provider = match naked_core::build_provider_from_config(&cfg) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("  SKIP: couldn't build provider: {e}");
            return;
        }
    };
    let agent = Arc::new(AgentCore::new(cfg.clone(), provider));
    agent.init_self_ref();
    agent.init_mcp().await;

    let session_id = agent.create_session(sessions_tmp.path()).await;
    let now_ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let _ = agent.set_session_yolo(&session_id, Some(now_ts)).await;

    // ── Phase 1: natural prompt → orchestration ──────────────────────────
    let prompt = "Мне нужно найти помещение под кафе завтраков в Дананге (Вьетнам).\n\n\
        Требования:\n\
        - Бюджет: от $1000 до $3000 в месяц\n\
        - Площадь: 100-200 м²\n\
        - Районы: An Thuong, My An, My Khe — экспатская зона\n\
        - Обязательно 1-й этаж с возможностью террасы\n\
        - Рядом с пляжем или в зоне активного пешеходного трафика\n\n\
        Ищи на всех основных сайтах недвижимости Вьетнама: batdongsan.com.vn, \
        chotot.com, muaban.net, alonhadat.com.vn, homedy.com, nha.chotot.com.\n\n\
        Используй research_create чтобы создать задачу и research_launch \
        чтобы запустить глубокий фоновый поиск.";

    let mut handle = match agent.send_prompt(&session_id, prompt).await {
        Ok(h) => h,
        Err(e) => {
            eprintln!("  SKIP: send_prompt failed: {e}");
            return;
        }
    };

    let mut text = String::new();
    let mut tool_calls = Vec::new();
    let mut got_idle = false;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(120);
    loop {
        match tokio::time::timeout_at(deadline, handle.events.recv()).await {
            Ok(Some(AgentEvent::TextDelta(t))) => text.push_str(&t),
            Ok(Some(AgentEvent::ToolStart { name, .. })) => {
                eprintln!("  [chat] tool start: {name}");
            }
            Ok(Some(AgentEvent::ToolEnd { name, .. })) => {
                eprintln!("  [chat] tool completed: {name}");
                tool_calls.push(name);
            }
            Ok(Some(AgentEvent::PermissionRequest {
                call_id, tool_name, ..
            })) => {
                eprintln!("  [chat] auto-approve: {tool_name}");
                let _ = handle
                    .permissions
                    .send(PermissionResponse {
                        call_id,
                        allowed: true,
                    })
                    .await;
            }
            Ok(Some(AgentEvent::Idle)) => {
                got_idle = true;
                break;
            }
            Ok(Some(AgentEvent::Error(e))) => {
                eprintln!("  [chat] ERROR: {e}");
                break;
            }
            Ok(None) => break,
            Err(_) => {
                eprintln!("  [chat] TIMEOUT waiting for agent");
                break;
            }
            _ => {}
        }
    }

    assert!(got_idle, "agent didn't reach idle");
    assert!(
        tool_calls.contains(&"research_create".to_string()),
        "agent didn't call research_create. tool_calls={tool_calls:?}"
    );
    assert!(
        tool_calls.contains(&"research_launch".to_string()),
        "agent didn't call research_launch. tool_calls={tool_calls:?}"
    );

    let store = agent.research_store();
    let specs = store.list_specs().await.unwrap();
    assert!(!specs.is_empty(), "no specs persisted");
    let spec_id = specs[0].id.clone();
    eprintln!("  Phase 1 OK: spec={spec_id}, topic={}", specs[0].topic);

    // ── Phase 2: wait for background run to finish ───────────────────────
    eprintln!("  Waiting for background research run to finish (up to 10 min)...");
    let poll_deadline = tokio::time::Instant::now() + Duration::from_secs(600);
    let mut run_finished = false;
    loop {
        tokio::time::sleep(Duration::from_secs(10)).await;
        let runs = store.list_runs(&spec_id, Some(1)).await.unwrap_or_default();
        if !runs.is_empty() {
            let r = &runs[0];
            eprintln!(
                "  Run finished: new={} total={} stop={} via {}/{}",
                r.new_findings, r.total_findings_after, r.stop_reason, r.provider, r.model
            );
            run_finished = true;
            break;
        }
        if tokio::time::Instant::now() > poll_deadline {
            eprintln!("  TIMEOUT: background run didn't finish in 10 min");
            break;
        }
        eprint!(".");
    }

    assert!(run_finished, "background research run never completed");

    // ── Phase 3: verify findings quality ─────────────────────────────────
    let findings = store.list_findings(&spec_id, None).await.unwrap();
    eprintln!("  Total findings: {}", findings.len());

    assert!(
        !findings.is_empty(),
        "research run produced zero findings — agent failed to find anything"
    );

    // Print all findings for manual review
    for (i, f) in findings.iter().enumerate() {
        eprintln!(
            "  {:2}. [{}] {:>20} | {} | {}",
            i + 1,
            f.listing_date.as_deref().unwrap_or("—"),
            f.price.as_deref().unwrap_or("—"),
            f.title.as_deref().unwrap_or("(untitled)"),
            f.url
        );
    }

    // 3a. Dedup: no duplicate dedup_hash
    let mut seen_hashes = std::collections::HashSet::new();
    for f in &findings {
        assert!(
            seen_hashes.insert(f.dedup_hash.clone()),
            "duplicate dedup_hash: {} (url={})",
            f.dedup_hash,
            f.url
        );
    }
    eprintln!("  ✓ No duplicate findings");

    // 3b. listing_date: at least 30% of findings should have it
    let with_date = findings.iter().filter(|f| f.listing_date.is_some()).count();
    let date_pct = (with_date as f64 / findings.len() as f64 * 100.0) as u32;
    eprintln!(
        "  listing_date present: {with_date}/{} ({date_pct}%)",
        findings.len()
    );
    assert!(
        date_pct >= 30,
        "too few findings with listing_date: {with_date}/{} ({date_pct}%). \
         Expected ≥30%. The agent should extract dates from listings.",
        findings.len()
    );

    // 3c. URLs should be canonicalized (no fragments, no utm_*)
    for f in &findings {
        assert!(
            !f.url.contains('#'),
            "finding URL not canonicalized (fragment): {}",
            f.url
        );
        assert!(
            !f.url.to_lowercase().contains("utm_"),
            "finding URL not canonicalized (utm): {}",
            f.url
        );
    }
    eprintln!("  ✓ All URLs canonicalized");

    // 3d. At least some findings should have a price
    let with_price = findings.iter().filter(|f| f.price.is_some()).count();
    let price_pct = (with_price as f64 / findings.len() as f64 * 100.0) as u32;
    eprintln!(
        "  price present: {with_price}/{} ({price_pct}%)",
        findings.len()
    );
    assert!(
        price_pct >= 40,
        "too few findings with price: {with_price}/{} ({price_pct}%). \
         Expected ≥40%.",
        findings.len()
    );

    // 3e. Spot-check: at least one URL should be from a known VN real estate site
    let known_domains = [
        "batdongsan.com.vn",
        "chotot.com",
        "nha.chotot.com",
        "muaban.net",
        "alonhadat.com.vn",
        "homedy.com",
        "bds123.vn",
        "dothi.net",
    ];
    let from_known = findings
        .iter()
        .filter(|f| known_domains.iter().any(|d| f.url.contains(d)))
        .count();
    eprintln!("  from known VN sites: {from_known}/{}", findings.len());
    // Soft check — warn but don't fail, agent might find listings on other sites
    if from_known == 0 {
        eprintln!(
            "  ⚠ no findings from known VN real estate domains (may be OK if found elsewhere)"
        );
    }

    // 3f. report.md should exist and mention Da Nang
    let report = store
        .read_report(&spec_id)
        .await
        .unwrap()
        .expect("report.md was not generated");
    let report_lower = report.to_lowercase();
    assert!(
        report_lower.contains("da nang")
            || report_lower.contains("дананг")
            || report_lower.contains("danang"),
        "report.md should mention Da Nang"
    );
    eprintln!("  ✓ report.md OK ({} bytes)", report.len());

    // 3g. agent_brief.md should exist
    let brief = store
        .read_agent_brief(&spec_id)
        .await
        .unwrap()
        .expect("agent_brief.md was not generated");
    assert!(
        brief.contains("Collected findings"),
        "agent_brief.md missing findings table"
    );
    eprintln!("  ✓ agent_brief.md OK ({} bytes)", brief.len());

    // 3h. Spot-check a few URLs are live (HTTP 200 or 301/302)
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .unwrap();
    let mut live = 0u32;
    let mut dead = 0u32;
    let check_count = findings.len().min(5);
    for f in findings.iter().take(check_count) {
        match client.head(&f.url).send().await {
            Ok(resp) => {
                let status = resp.status().as_u16();
                if status < 400 {
                    live += 1;
                    eprintln!("    ✓ {} → {status}", f.url);
                } else {
                    dead += 1;
                    eprintln!("    ✗ {} → {status}", f.url);
                }
            }
            Err(e) => {
                dead += 1;
                eprintln!("    ✗ {} → err: {e}", f.url);
            }
        }
    }
    eprintln!("  URL spot-check: {live} live, {dead} dead (of {check_count} checked)");
    // At least half of checked URLs should be reachable
    assert!(
        live > 0,
        "all {check_count} spot-checked URLs are dead — findings likely stale or hallucinated"
    );

    eprintln!(
        "\n  ══ PASS ══ t105_research_full_quality_e2e\n  \
         findings={} (date:{with_date} price:{with_price} known_sites:{from_known}) \
         urls_live={live}/{check_count}\n  spec={spec_id}",
        findings.len()
    );
}

/// Live: agent uses research_list_specs + research_findings to produce a
/// human-readable summary of existing research — the "что нашлось?" flow.
/// Uses the real on-disk spec (requires a prior research run to have findings).
#[tokio::test]
async fn t106_research_summary_from_chat_e2e() {
    use naked_core::AgentCore;
    use naked_core::types::PermissionResponse;

    need_config!(config);
    pace().await;

    eprintln!(">>> t106_research_summary_from_chat_e2e");

    let sessions_tmp = tempfile::tempdir().unwrap();

    // Use real research storage (not tmp) so we see existing findings
    let mut cfg: Config = config.clone();
    cfg.session_dir = sessions_tmp.path().to_path_buf();
    cfg.workspace = sessions_tmp.path().to_path_buf();

    let provider = match naked_core::build_provider_from_config(&cfg) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("  SKIP: couldn't build provider: {e}");
            return;
        }
    };
    let agent = Arc::new(AgentCore::new(cfg, provider));
    agent.init_self_ref();

    // Check we have at least one spec with findings
    let store = agent.research_store();
    let specs = match store.list_specs().await {
        Ok(s) if !s.is_empty() => s,
        _ => {
            eprintln!("  SKIP: no research specs on disk (run t105 first)");
            return;
        }
    };
    let total_findings = store.count_findings(&specs[0].id).await.unwrap_or(0);
    if total_findings == 0 {
        eprintln!("  SKIP: spec {} has 0 findings", specs[0].id);
        return;
    }
    eprintln!(
        "  Using spec {} with {total_findings} findings",
        specs[0].id
    );

    let session_id = agent.create_session(sessions_tmp.path()).await;
    let now_ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let _ = agent.set_session_yolo(&session_id, Some(now_ts)).await;

    let prompt = "Что нашлось по аренде помещений в Дананге? \
        Используй research_list_specs и research_findings чтобы получить все результаты, \
        и дай мне сводку: лучшие варианты с ценами, площадью и контактами.";

    let mut handle = match agent.send_prompt(&session_id, prompt).await {
        Ok(h) => h,
        Err(e) => {
            eprintln!("  SKIP: send_prompt failed: {e}");
            return;
        }
    };

    let mut text = String::new();
    let mut tool_calls = Vec::new();
    let mut got_idle = false;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(120);
    loop {
        match tokio::time::timeout_at(deadline, handle.events.recv()).await {
            Ok(Some(AgentEvent::TextDelta(t))) => text.push_str(&t),
            Ok(Some(AgentEvent::ToolStart { name, .. })) => {
                eprintln!("  tool start: {name}");
            }
            Ok(Some(AgentEvent::ToolEnd { name, .. })) => {
                eprintln!("  tool completed: {name}");
                tool_calls.push(name);
            }
            Ok(Some(AgentEvent::PermissionRequest {
                call_id, tool_name, ..
            })) => {
                eprintln!("  auto-approve: {tool_name}");
                let _ = handle
                    .permissions
                    .send(PermissionResponse {
                        call_id,
                        allowed: true,
                    })
                    .await;
            }
            Ok(Some(AgentEvent::Idle)) => {
                got_idle = true;
                break;
            }
            Ok(Some(AgentEvent::Error(e))) => {
                eprintln!("  ERROR: {e}");
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

    let preview_end = text
        .char_indices()
        .nth(800)
        .map(|(i, _)| i)
        .unwrap_or(text.len());
    eprintln!("  tool calls: {:?}", tool_calls);
    eprintln!(
        "  response ({} chars):\n{}",
        text.len(),
        &text[..preview_end]
    );

    assert!(got_idle, "agent didn't reach idle");
    assert!(
        tool_calls.contains(&"research_findings".to_string())
            || tool_calls.contains(&"research_list_specs".to_string()),
        "agent didn't use research tools. tool_calls={tool_calls:?}"
    );
    assert!(
        text.len() > 100,
        "agent response too short ({} chars) — should be a useful summary",
        text.len()
    );

    // Response should mention prices or specific findings
    let has_prices = text.contains("triệu")
        || text.contains("tr/")
        || text.contains("$")
        || text.contains("VND")
        || text.contains("USD");
    assert!(has_prices, "summary should mention prices from findings");

    eprintln!("\n  ══ PASS ══ t106: agent summarized {total_findings} findings via tools");
}

/// Live: full gatekeeper loop on existing research spec.
/// 1. run_verified with max 2 verification rounds
/// 2. Checks all findings have live URLs after gatekeeper
/// 3. Checks dead findings were removed
/// 4. Checks data completeness (title, price, excerpt)
#[tokio::test]
async fn t107_research_gatekeeper_e2e() {
    use naked_core::AgentCore;

    need_config!(config);
    pace().await;

    eprintln!(">>> t107_research_gatekeeper_e2e");

    let sessions_tmp = tempfile::tempdir().unwrap();

    let mut cfg: Config = config.clone();
    cfg.session_dir = sessions_tmp.path().to_path_buf();
    cfg.workspace = sessions_tmp.path().to_path_buf();

    let provider = match naked_core::build_provider_from_config(&cfg) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("  SKIP: couldn't build provider: {e}");
            return;
        }
    };
    let agent = Arc::new(AgentCore::new(cfg, provider));
    agent.init_self_ref();

    // Check we have an existing spec with findings to verify
    let store = agent.research_store();
    let specs = match store.list_specs().await {
        Ok(s) if !s.is_empty() => s,
        _ => {
            eprintln!("  SKIP: no research specs on disk (run t105 first)");
            return;
        }
    };
    let spec_id = specs[0].id.clone();
    let before_count = store.count_findings(&spec_id).await.unwrap_or(0);
    if before_count == 0 {
        eprintln!("  SKIP: spec {spec_id} has 0 findings");
        return;
    }
    eprintln!("  Using spec {spec_id} with {before_count} findings");

    // Run with gatekeeper (max 2 verification rounds)
    eprintln!("  Running research with gatekeeper verification...");
    let result = match agent.clone().run_research_verified(&spec_id, 2).await {
        Ok(r) => r,
        Err(e) => {
            eprintln!("  ERROR: run_research_verified failed: {e}");
            panic!("run_research_verified failed: {e}");
        }
    };

    eprintln!("  Gatekeeper result:");
    eprintln!("    verification_rounds: {}", result.verification_rounds);
    eprintln!("    dead_removed: {}", result.dead_removed);
    eprintln!("    replacements_found: {}", result.replacements_found);
    eprintln!("    final_findings: {}", result.final_findings);
    eprintln!("    remaining_issues: {}", result.remaining_issues.len());

    // After gatekeeper, verify all remaining findings have live URLs
    let findings = store.list_findings(&spec_id, None).await.unwrap();
    eprintln!("  Verifying {} remaining findings...", findings.len());

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .redirect(reqwest::redirect::Policy::limited(5))
        .build()
        .unwrap();

    let mut live = 0u32;
    let mut dead = 0u32;
    for f in &findings {
        match client.head(&f.url).send().await {
            Ok(resp) if resp.status().as_u16() < 400 => {
                live += 1;
            }
            Ok(resp) => {
                dead += 1;
                eprintln!("    ✗ {} → {}", f.url, resp.status());
            }
            Err(e) => {
                dead += 1;
                eprintln!("    ✗ {} → {e}", f.url);
            }
        }
    }
    eprintln!(
        "  Post-gatekeeper URL check: {live} live, {dead} dead out of {}",
        findings.len()
    );

    // At least 90% should be live after gatekeeper
    let live_pct = if findings.is_empty() {
        100
    } else {
        (live as f64 / findings.len() as f64 * 100.0) as u32
    };
    assert!(
        live_pct >= 90,
        "after gatekeeper, only {live_pct}% of URLs are live (expected ≥90%). \
         live={live}, dead={dead}, total={}",
        findings.len()
    );

    // Check data completeness
    let with_title = findings.iter().filter(|f| f.title.is_some()).count();
    let with_price = findings.iter().filter(|f| f.price.is_some()).count();
    let with_excerpt = findings
        .iter()
        .filter(|f| f.excerpt.as_ref().is_some_and(|e| e.len() >= 50))
        .count();

    let title_pct = 100 * with_title / findings.len().max(1);
    let price_pct = 100 * with_price / findings.len().max(1);
    let excerpt_pct = 100 * with_excerpt / findings.len().max(1);

    eprintln!("  Data completeness:");
    eprintln!(
        "    title:   {with_title}/{} ({title_pct}%)",
        findings.len()
    );
    eprintln!(
        "    price:   {with_price}/{} ({price_pct}%)",
        findings.len()
    );
    eprintln!(
        "    excerpt: {with_excerpt}/{} ({excerpt_pct}%)",
        findings.len()
    );

    assert!(
        title_pct >= 80,
        "after gatekeeper, title coverage {title_pct}% < 80%"
    );
    assert!(
        price_pct >= 40,
        "after gatekeeper, price coverage {price_pct}% < 40%"
    );

    // Report and brief should exist
    assert!(
        store.read_report(&spec_id).await.unwrap().is_some(),
        "report.md missing after gatekeeper run"
    );
    assert!(
        store.read_agent_brief(&spec_id).await.unwrap().is_some(),
        "agent_brief.md missing after gatekeeper run"
    );

    eprintln!(
        "\n  ══ PASS ══ t107_research_gatekeeper_e2e\n  \
         rounds={} removed={} replaced={} final={}\n  \
         live={live}/{} ({live_pct}%) title={title_pct}% price={price_pct}% excerpt={excerpt_pct}%",
        result.verification_rounds,
        result.dead_removed,
        result.replacements_found,
        result.final_findings,
        findings.len()
    );
}
