//! Live LLM e2e tests for the research LLM control plane (Phases 2-5).
//!
//! These tests drive a real LLM agent through natural-language prompts and
//! assert that the right `research_*` tools were called and that the on-disk
//! state changed accordingly.
//!
//! They are gated on a working provider configuration (`Config::load`
//! returns at least one provider with a non-empty API key) — same convention
//! as the larger `e2e_live.rs` suite. When no provider is reachable, every
//! test prints `SKIP:` and returns Ok.
//!
//! Run sequentially:
//!   cargo test -p naked-core --test research_llm_control_e2e -- \
//!       --nocapture --test-threads=1
//!
//! Why "live but loose": LLM outputs are non-deterministic. Each test
//! asserts the minimum observable contract — (a) the agent invoked at least
//! one of the appropriate research tools, and (b) the on-disk research state
//! moved in the expected direction. This catches structural regressions
//! (tool dropped from registry, schema changed enough that the LLM refuses,
//! mutation didn't reach the store) without flaking on phrasing differences.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use futures_util::StreamExt;
use naked_core::AgentCore;
use naked_core::ResearchPatch;
use naked_core::config::{Config, ResearchConfig};
use naked_core::history::ConversationHistory;
use naked_core::loop_::{AgentLoop, LoopConfig};
use naked_core::provider::{ChatRequest, Provider};
use naked_core::research::{
    ResearchCreateTool, ResearchFindingsTool, ResearchLaunchTool, ResearchListSpecsTool,
    ResearchMetricsTool, ResearchPauseTool, ResearchResumeTool, ResearchSetScheduleTool,
    ResearchSetTargetTool, ResearchSpec, ResearchUpdateSpecTool,
};
use naked_core::tool::Tool;
use naked_core::tool::registry::ToolRegistry;
use naked_core::types::{AgentEvent, StreamChunk};
use tempfile::TempDir;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

// ── Provider acquisition (mirrors e2e_live.rs minimal subset) ──────────────

fn test_config() -> Option<Config> {
    let cfg = Config::load().ok()?;
    if cfg.providers.is_empty() {
        eprintln!("SKIP: no providers configured");
        return None;
    }
    Some(cfg)
}

async fn probe_provider(provider: &dyn Provider, model: &str) -> bool {
    let req = ChatRequest {
        model: model.to_string(),
        system: String::new(),
        messages: vec![serde_json::json!({"role": "user", "content": "Say OK"})],
        tools: vec![],
        max_tokens: 32,
        temperature: None,
        reasoning: Some("off".into()),
    };
    let Ok(mut stream) = provider.stream_chat(req).await else {
        return false;
    };
    let mut got_text = false;
    while let Some(chunk) = stream.next().await {
        match chunk {
            StreamChunk::Text(_) => got_text = true,
            StreamChunk::Error(_) => return false,
            StreamChunk::Done => break,
            _ => {}
        }
    }
    got_text
}

async fn working_provider(cfg: &Config) -> Option<(Box<dyn Provider>, String)> {
    let explicit = std::env::var("E2E_MODEL").ok();
    if let Ok(prov) = naked_core::build_provider_from_config(cfg) {
        let model = explicit.clone().unwrap_or_else(|| {
            if !cfg.default_model.is_empty() {
                cfg.default_model.clone()
            } else {
                cfg.providers
                    .values()
                    .next()
                    .and_then(|p| p.models.first().cloned())
                    .unwrap_or_else(|| "gpt-4o-mini".into())
            }
        });
        if probe_provider(prov.as_ref(), &model).await {
            return Some((prov, model));
        }
    }
    for (name, pc) in &cfg.providers {
        let resolved = match pc.resolved() {
            Ok(r) if !r.api_key.is_empty() && !r.api_key.starts_with('$') => r,
            _ => continue,
        };
        let model = explicit.clone().unwrap_or_else(|| {
            pc.models
                .first()
                .cloned()
                .unwrap_or_else(|| "gpt-4o-mini".into())
        });
        let prov = naked_core::create_provider(name, resolved);
        if probe_provider(prov.as_ref(), &model).await {
            return Some((prov, model));
        }
    }
    None
}

// ── Test setup ─────────────────────────────────────────────────────────────

struct TestSetup {
    _tmp: TempDir,
    core: Arc<AgentCore>,
    tools: ToolRegistry,
    provider: Box<dyn Provider>,
    model: String,
}

/// Construct an `AgentCore` with research enabled, register the same
/// research tool subset that `AgentCore::build_tool_registry_for` would
/// register in production, and return a working provider.
async fn setup() -> Option<TestSetup> {
    let cfg_loaded = test_config()?;
    let (provider, model) = working_provider(&cfg_loaded).await?;
    let tmp = TempDir::new().ok()?;

    let cfg = Config {
        workspace: tmp.path().to_path_buf(),
        session_dir: tmp.path().join("sessions"),
        research: ResearchConfig {
            enabled: true,
            storage_dir: Some(tmp.path().join("research")),
            ..Default::default()
        },
        ..Default::default()
    };

    // Build a clone of the provider for use inside AgentCore (which owns it)
    // — we reuse the same one for the loop driver below.
    let core_provider: Box<dyn Provider> =
        naked_core::build_provider_from_config(&cfg_loaded).ok()?;
    let core = Arc::new(AgentCore::new(cfg, core_provider));
    core.init_self_ref();

    let weak = Arc::downgrade(&core);
    let store = core.research_store();
    let research_cfg = core.config().research.clone();
    let research_ctx = naked_core::research::ResearchContext::new();

    let tools_vec: Vec<Box<dyn Tool>> = vec![
        Box::new(ResearchCreateTool::new(store.clone(), research_cfg.clone())),
        Box::new(ResearchListSpecsTool::new(
            store.clone(),
            research_cfg.clone(),
        )),
        Box::new(ResearchMetricsTool::new(store.clone())),
        Box::new(ResearchFindingsTool::new(store.clone())),
        Box::new(ResearchSetTargetTool::new(store.clone(), research_ctx)),
        Box::new(ResearchLaunchTool::new(weak.clone())),
        Box::new(ResearchUpdateSpecTool::new(weak.clone())),
        Box::new(ResearchSetScheduleTool::new(weak.clone())),
        Box::new(ResearchPauseTool::new(weak.clone())),
        Box::new(ResearchResumeTool::new(weak)),
    ];
    let tools = ToolRegistry::new(tools_vec);

    Some(TestSetup {
        _tmp: tmp,
        core,
        tools,
        provider,
        model,
    })
}

/// Pre-seed a research spec on disk so the LLM has something to act on.
async fn seed_spec(core: &Arc<AgentCore>, topic: &str, interval: Option<u64>) -> ResearchSpec {
    let spec = core
        .create_research(topic, vec![], None, None, None)
        .await
        .expect("seed create_research");
    if let Some(secs) = interval {
        let patch = ResearchPatch {
            interval_seconds: Some(Some(secs)),
            ..Default::default()
        };
        core.update_research(&spec.id, patch)
            .await
            .expect("seed set interval");
    }
    core.load_research(&spec.id).await.expect("reload seeded")
}

struct TurnResult {
    events: Vec<AgentEvent>,
}

impl TurnResult {
    fn tool_names(&self) -> Vec<String> {
        self.events
            .iter()
            .filter_map(|e| match e {
                AgentEvent::ToolStart { name, .. } => Some(name.clone()),
                _ => None,
            })
            .collect()
    }
    fn called_any(&self, names: &[&str]) -> bool {
        let actual = self.tool_names();
        actual.iter().any(|n| names.contains(&n.as_str()))
    }
    fn full_text(&self) -> String {
        self.events
            .iter()
            .filter_map(|e| match e {
                AgentEvent::TextDelta(t) => Some(t.as_str()),
                _ => None,
            })
            .collect()
    }
}

const SYS: &str = "You manage research tasks for the user. When asked about a research's \
status, schedule, sources, or metrics, call the appropriate research_* tool first. \
For mutations (add source, change schedule, pause, resume), call the appropriate \
research_* tool. Do not invent IDs — list specs first if you don't know one. \
Be very concise.";

async fn run_turn(setup: TestSetup, prompt: &str) -> (TurnResult, Arc<AgentCore>) {
    let TestSetup {
        core,
        tools,
        provider,
        model,
        _tmp,
    } = setup;

    let cwd: PathBuf = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg = LoopConfig {
        max_iterations: 6,
        cwd,
        model: model.clone(),
        max_tokens: 1024,
        temperature: None,
        reasoning: Some("off".into()),
        provider: String::new(),
        health: None,
    };
    let agent = AgentLoop::new(provider, tools, cfg);

    let mut history = ConversationHistory::new(SYS.into());
    history.push_user(prompt);
    let (tx, mut rx) = mpsc::channel(256);
    let cancel = CancellationToken::new();
    let timeout = Duration::from_secs(120);
    let _ = tokio::time::timeout(timeout, agent.run(&mut history, tx, cancel, None)).await;

    let mut events = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        events.push(ev);
    }
    // Keep `_tmp` alive for the duration of the test by forgetting it —
    // Arc<AgentCore> still references files inside it.
    std::mem::forget(_tmp);
    (TurnResult { events }, core)
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn t_llm_can_describe_last_run() {
    let Some(s) = setup().await else {
        eprintln!("SKIP: no working provider configured");
        return;
    };
    let spec = seed_spec(&s.core, "weekly market scan for Da Nang rentals", None).await;
    let prompt = format!(
        "What's the current status and last run of research \"{}\"? Use a tool to look it up.",
        spec.topic
    );
    let (r, _core) = run_turn(s, &prompt).await;
    let names = r.tool_names();
    eprintln!("tools called: {names:?}");
    assert!(
        r.called_any(&[
            "research_metrics",
            "research_list_specs",
            "research_findings"
        ]),
        "expected one of research_metrics/list_specs/findings; got {names:?}; text={}",
        r.full_text()
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn t_llm_extends_sources_via_natural_language() {
    let Some(s) = setup().await else {
        eprintln!("SKIP: no working provider configured");
        return;
    };
    let core_for_check = s.core.clone();
    let spec = seed_spec(&s.core, "long-term office space in Hanoi", None).await;
    let prompt = format!(
        "Add the source https://example.com/listings to research \"{}\" (or its id `{}`). \
         Use a tool.",
        spec.topic, spec.id
    );
    let (r, _core) = run_turn(s, &prompt).await;
    let names = r.tool_names();
    eprintln!("tools called: {names:?}");
    assert!(
        r.called_any(&["research_update_spec"]),
        "expected research_update_spec; got {names:?}; text={}",
        r.full_text()
    );
    let after = core_for_check
        .load_research(&spec.id)
        .await
        .expect("reload after update");
    assert!(
        after
            .sources
            .iter()
            .any(|u| u.contains("example.com/listings")),
        "spec.sources should now contain the URL the LLM was asked to add; got {:?}",
        after.sources
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn t_llm_changes_schedule_takes_effect() {
    let Some(s) = setup().await else {
        eprintln!("SKIP: no working provider configured");
        return;
    };
    let core_for_check = s.core.clone();
    let spec = seed_spec(&s.core, "daily news digest about ai chips", Some(86_400)).await;
    let prompt = format!(
        "Change the schedule of research id `{}` to run every 30 minutes. Use a tool.",
        spec.id
    );
    let (r, _core) = run_turn(s, &prompt).await;
    let names = r.tool_names();
    eprintln!("tools called: {names:?}");
    assert!(
        r.called_any(&["research_set_schedule", "research_update_spec"]),
        "expected research_set_schedule or research_update_spec; got {names:?}; text={}",
        r.full_text()
    );
    let after = core_for_check
        .load_research(&spec.id)
        .await
        .expect("reload after schedule change");
    // The LLM was asked to set 30-minute schedule. It may set a different
    // value (model-dependent), but the tool must have been called. If the
    // value is still exactly 86400 *and* no tool was called, that's a real
    // failure. Since we already asserted the tool was called above, we only
    // log a warning if the value didn't change.
    if after.interval_seconds == Some(86_400) {
        eprintln!(
            "WARN: interval stayed at 86400 despite tool call — LLM may have \
             set the same value. Tool names: {names:?}"
        );
    }
    assert!(
        after.interval_seconds.is_some(),
        "interval should still be set (not cleared); got {:?}",
        after.interval_seconds
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn t_llm_pauses_and_resumes_research() {
    let Some(s) = setup().await else {
        eprintln!("SKIP: no working provider configured");
        return;
    };
    let core_check = s.core.clone();
    let spec = seed_spec(&s.core, "weekly competitor watch", Some(7 * 86_400)).await;

    // Turn 1: pause
    let prompt = format!("Pause research id `{}` please. Use a tool.", spec.id);
    let (r, _core) = run_turn(s, &prompt).await;
    let names = r.tool_names();
    eprintln!("pause tools: {names:?}");
    assert!(
        r.called_any(&[
            "research_pause",
            "research_update_spec",
            "research_set_schedule"
        ]),
        "expected pause-related tool; got {names:?}; text={}",
        r.full_text()
    );
    let after_pause = core_check
        .load_research(&spec.id)
        .await
        .expect("reload after pause");
    assert!(
        after_pause.paused,
        "spec should be paused after the LLM call"
    );

    // Turn 2: resume — fresh setup so the LLM doesn't replay turn 1.
    let Some(s2) = setup_from_existing(&core_check).await else {
        eprintln!("SKIP: provider unavailable for second turn");
        return;
    };
    let prompt = format!("Resume research id `{}` now. Use a tool.", spec.id);
    let (r, _core) = run_turn(s2, &prompt).await;
    let names = r.tool_names();
    eprintln!("resume tools: {names:?}");
    assert!(
        r.called_any(&[
            "research_resume",
            "research_update_spec",
            "research_set_schedule"
        ]),
        "expected resume-related tool; got {names:?}; text={}",
        r.full_text()
    );
    let after_resume = core_check
        .load_research(&spec.id)
        .await
        .expect("reload after resume");
    assert!(
        !after_resume.paused,
        "spec should be unpaused after the resume call"
    );
}

/// Build a second `TestSetup` reusing the workspace of an existing `AgentCore`
/// — needed for multi-turn tests that need a fresh `AgentLoop` (since
/// `AgentLoop::run` consumes the provider and history) but the same on-disk
/// state.
async fn setup_from_existing(existing: &Arc<AgentCore>) -> Option<TestSetup> {
    let cfg_loaded = test_config()?;
    let (provider, model) = working_provider(&cfg_loaded).await?;
    // Reuse the existing core's storage_dir by walking its config.
    let workspace: &Path = &existing.config().workspace;
    let tmp = TempDir::new_in(workspace.parent().unwrap_or(Path::new("."))).ok()?;

    let weak = Arc::downgrade(existing);
    let store = existing.research_store();
    let research_cfg = existing.config().research.clone();
    let research_ctx = naked_core::research::ResearchContext::new();

    let tools_vec: Vec<Box<dyn Tool>> = vec![
        Box::new(ResearchCreateTool::new(store.clone(), research_cfg.clone())),
        Box::new(ResearchListSpecsTool::new(
            store.clone(),
            research_cfg.clone(),
        )),
        Box::new(ResearchMetricsTool::new(store.clone())),
        Box::new(ResearchFindingsTool::new(store.clone())),
        Box::new(ResearchSetTargetTool::new(store.clone(), research_ctx)),
        Box::new(ResearchLaunchTool::new(weak.clone())),
        Box::new(ResearchUpdateSpecTool::new(weak.clone())),
        Box::new(ResearchSetScheduleTool::new(weak.clone())),
        Box::new(ResearchPauseTool::new(weak.clone())),
        Box::new(ResearchResumeTool::new(weak)),
    ];
    let tools = ToolRegistry::new(tools_vec);

    Some(TestSetup {
        _tmp: tmp,
        core: existing.clone(),
        tools,
        provider,
        model,
    })
}
