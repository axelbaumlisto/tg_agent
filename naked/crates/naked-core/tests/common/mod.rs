#![allow(dead_code, unused_imports, unused_macros)]

//! Live E2E tests -- hit a real LLM provider, execute real tools,
//! validate the full agent pipeline.
//!
//! Configure via JSON config file or env vars.
//! Skipped automatically when no provider is configured.
//!
//! Run (sequential, respecting rate limits):
//!   cargo test -p naked-core --test e2e_live -- --nocapture --test-threads=1

pub use std::path::{Path, PathBuf};
pub use std::time::Duration;

pub use naked_core::agent_registry::AgentRegistry;
pub use naked_core::config::{Config, McpServerConfig};
pub use naked_core::history::ConversationHistory;
pub use naked_core::loop_::{AgentLoop, LoopConfig};
pub use naked_core::mcp::client::McpServer;
pub use naked_core::mcp::wrapper::McpToolWrapper;
pub use naked_core::provider::Provider;
pub use naked_core::provider::resilient::ResilientProvider;
pub use naked_core::session::jsonl_store::JsonlSessionStore;
pub use naked_core::session::store::SessionStore;
pub use naked_core::session::{Session, SessionMetadata};
pub use naked_core::skill::resolver::SkillResolver;
pub use naked_core::skill::tool::SkillTool;
pub use naked_core::tool::Tool;
pub use naked_core::tool::agent_control::{AgentStatusTool, AgentStopTool};
pub use naked_core::tool::bash::BashTool;
pub use naked_core::tool::file_ops::{EditFileTool, ReadFileTool, WriteFileTool};
pub use naked_core::tool::registry::ToolRegistry;
pub use naked_core::tool::search::{GlobSearchTool, GrepSearchTool};
pub use naked_core::tool::sub_agent::SubAgentTool;
pub use naked_core::types::{AgentEvent, ContentBlock, Permission, SubAgentEvent, ToolState};

pub use std::sync::Arc;

pub use tokio::sync::mpsc;
pub use tokio_util::sync::CancellationToken;

// ── Provider setup (all from config, no hardcoded names) ─────────────────────

pub fn test_config() -> Option<Config> {
    let config = Config::load().ok()?;
    if config.providers.is_empty() {
        eprintln!("SKIP: no providers configured");
        return None;
    }
    Some(config)
}

/// Try the default provider first; if its probe fails, iterate all providers
/// and return the first one that responds. Returns `(provider, model)`.
pub async fn get_working_provider(config: &Config) -> Option<(Box<dyn Provider>, String)> {
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

pub fn build_tools() -> ToolRegistry {
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

pub fn loop_config(cwd: PathBuf, model: &str) -> LoopConfig {
    LoopConfig {
        max_iterations: 15,
        cwd,
        model: model.to_string(),
        max_tokens: 2048,
        ..Default::default()
    }
}

pub struct E2eResult {
    pub events: Vec<AgentEvent>,
    pub history: ConversationHistory,
}

impl E2eResult {
    pub fn full_text(&self) -> String {
        self.events
            .iter()
            .filter_map(|e| match e {
                AgentEvent::TextDelta(t) => Some(t.as_str()),
                _ => None,
            })
            .collect()
    }

    pub fn tool_names(&self) -> Vec<String> {
        self.events
            .iter()
            .filter_map(|e| match e {
                AgentEvent::ToolStart { name, .. } => Some(name.clone()),
                _ => None,
            })
            .collect()
    }

    #[allow(dead_code)]
    pub fn tool_results(&self) -> Vec<(String, ToolState, String)> {
        self.events
            .iter()
            .filter_map(|e| match e {
                AgentEvent::ToolEnd {
                    name,
                    state,
                    output,
                    ..
                } => Some((name.clone(), *state, output.clone())),
                _ => None,
            })
            .collect()
    }

    pub fn has_tool(&self, name: &str) -> bool {
        self.tool_names().iter().any(|n| n == name)
    }

    pub fn had_error(&self) -> bool {
        self.events
            .iter()
            .any(|e| matches!(e, AgentEvent::Error(_)))
    }

    pub fn got_idle(&self) -> bool {
        self.events.iter().any(|e| matches!(e, AgentEvent::Idle))
    }
}

pub async fn run_prompt(
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

pub const SYS: &str = "You are a concise coding assistant. Use tools when asked. Respond briefly.";

pub fn make_provider_for(
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
pub fn degraded_provider_set() -> &'static std::sync::Mutex<std::collections::HashSet<String>> {
    pub use std::sync::OnceLock;
    static SET: OnceLock<std::sync::Mutex<std::collections::HashSet<String>>> = OnceLock::new();
    SET.get_or_init(|| std::sync::Mutex::new(std::collections::HashSet::new()))
}

pub fn degraded_key(provider: &dyn naked_core::provider::Provider, model: &str) -> String {
    format!("{}::{model}", provider.name())
}

pub async fn probe_provider(provider: &dyn Provider, model: &str) -> bool {
    pub use futures_util::StreamExt;
    pub use naked_core::provider::ChatRequest;

    // Short-circuit if we already marked this (provider, model) degraded
    // earlier in the process.
    let key = degraded_key(provider, model);
    if let Ok(set) = degraded_provider_set().lock()
        && set.contains(&key)
    {
        eprintln!("  probe: {key} already marked degraded this run; skipping");
        return false;
    }

    pub async fn run_once(
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
pub async fn pace() {
    let secs: u64 = std::env::var("E2E_PAUSE_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(15);
    tokio::time::sleep(Duration::from_secs(secs)).await;
}

pub fn build_tools_with_skill_roots(roots: Vec<PathBuf>) -> ToolRegistry {
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

pub fn build_tools_with_mcp(mcp_tools: Vec<Box<dyn Tool>>) -> ToolRegistry {
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

pub async fn run_prompt_with_tools(
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

pub fn mcp_server_script_path() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
    PathBuf::from(manifest)
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join("test-mcp-server.sh"))
        .unwrap_or_else(|| PathBuf::from("test-mcp-server.sh"))
}

pub fn skill_roots_from_config(config: &Config) -> Vec<PathBuf> {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    // Resolve relative paths against the workspace root (where naked.json lives),
    // not against the test binary's CWD (which cargo sets to the crate dir).
    let ws_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    config
        .skill_roots
        .iter()
        .map(|p| {
            let s = p.to_string_lossy();
            if let Some(rest) = s.strip_prefix("~/") {
                PathBuf::from(format!("{home}/{rest}"))
            } else if s == "~" {
                PathBuf::from(&home)
            } else if p.is_relative() {
                ws_root.join(p)
            } else {
                p.clone()
            }
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests (run with --test-threads=1)
// ═══════════════════════════════════════════════════════════════════════════════

// ── Helpers extracted from test section ──

#[test]
pub fn t34_multi_key_creates_resilient_provider() {
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
pub fn t35_single_key_creates_plain_provider() {
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

#[derive(Debug)]
#[allow(dead_code)]
pub struct ChatStats {
    pub tag: String,
    pub turns: usize,
    pub ok: usize,
    pub content_ok: usize,
    pub errors: usize,
    pub rate_limited: usize,
    pub tools_used: usize,
    pub tool_set: std::collections::HashSet<String>,
    /// Sum of individual turn latencies (excludes inter-turn delays).
    pub total_ms: u128,
    /// Wall-clock time for the entire session (includes inter-turn delays).
    pub session_wall_ms: u128,
    pub latencies: Vec<u128>,
}

impl ChatStats {
    pub fn percentile(&self, p: f64) -> u128 {
        if self.latencies.is_empty() {
            return 0;
        }
        let mut sorted = self.latencies.clone();
        sorted.sort();
        let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }
    pub fn avg(&self) -> u128 {
        if self.latencies.is_empty() {
            0
        } else {
            self.total_ms / self.latencies.len() as u128
        }
    }
    pub fn throughput(&self) -> f64 {
        if self.total_ms == 0 {
            0.0
        } else {
            self.turns as f64 / (self.total_ms as f64 / 1000.0)
        }
    }
}

pub struct TurnSpec {
    pub prompt: String,
    /// Substring that must appear in the text response (None = no content check).
    pub expect_text: Option<String>,
    /// Tool name that should have been called (None = no tool check).
    pub expect_tool: Option<String>,
    /// If set, verify this file exists with this content after the turn.
    pub expect_file: Option<(PathBuf, String)>,
}

pub fn build_load_specs(tag: &str, n: usize, tmp: &Path) -> Vec<TurnSpec> {
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

pub async fn run_load_session(
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
            ..Default::default()
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

pub fn provider_arc_to_box(provider: &Arc<dyn Provider>) -> Box<dyn Provider> {
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

pub async fn find_working_providers(config: &Config, n: usize) -> Vec<(String, String)> {
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

pub fn make_agent_core(
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

pub async fn agent_prompt(
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

pub async fn write_session_config(
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

pub struct SessionStats {
    pub label: &'static str,
    pub responses: usize,
    pub idles: usize,
    pub all_tools: Vec<String>,
    pub all_texts: Vec<String>,
}

impl SessionStats {
    pub fn new(label: &'static str) -> Self {
        Self {
            label,
            responses: 0,
            idles: 0,
            all_tools: Vec::new(),
            all_texts: Vec::new(),
        }
    }

    pub fn record(&mut self, _turn: usize, text: &str, tools: &[String], idle: bool) {
        if !text.is_empty() {
            self.responses += 1;
        }
        if idle {
            self.idles += 1;
        }
        self.all_tools.extend(tools.iter().cloned());
        self.all_texts.push(text.to_string());
    }

    pub fn tool_used(&self, name: &str) -> bool {
        self.all_tools.iter().any(|t| t == name)
    }

    pub fn text_contains_count(&self, needle: &str) -> usize {
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

pub use naked_core::tool::web_search::WebSearchTool;

/// #32 — WebSearch: spec is correct
#[tokio::test]
pub async fn t80_web_search_spec() {
    let tool = WebSearchTool::from_legacy_exa(vec![]);
    let spec = tool.spec();
    assert_eq!(spec.name, "web_search");
    assert_eq!(spec.permission, Permission::ReadOnly);
    eprintln!("  PASS: web_search spec");
}

pub fn research_system_prompt(cwd: &Path) -> String {
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

pub async fn run_research_prompt(
    provider: Box<dyn Provider>,
    history: &mut ConversationHistory,
    cwd: &Path,
    model: &str,
    tools: ToolRegistry,
    idle_timeout_secs: u64,
) -> E2eResult {
    use std::sync::atomic::{AtomicU64, Ordering};

    pub fn now_ms() -> u64 {
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
        ..Default::default()
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

pub use naked_core::memory::service::MemoryService;
pub use naked_core::memory::store::MarkdownMemoryStore;
pub use naked_core::memory::types::{MemoryScope, MemoryType};
use naked_core::tool::memory::MemoryTool;

pub fn build_tools_with_memory(workspace: PathBuf) -> ToolRegistry {
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

pub fn fixture_image_path() -> Option<PathBuf> {
    // crates/naked-core/tests/e2e_live.rs → naked/
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        here.join("../naked-tg/tests/fixtures/test_image.png"),
        here.join("tests/fixtures/test_image.png"),
    ];
    candidates.into_iter().find(|p| p.exists())
}

pub fn pick_vision_capable_providers(config: &Config) -> Vec<(Box<dyn Provider>, String, String)> {
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
pub use naked_core::research::FindingStore;
pub use naked_core::research::ResearchStore;
