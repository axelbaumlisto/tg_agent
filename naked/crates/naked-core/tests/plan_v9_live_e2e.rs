//! Live E2E tests for plan-v9 tools — real LLM, real tool execution.
//!
//! Skipped when no provider is configured.
//!
//! Run:  cargo test -p naked-core --test plan_v9_live_e2e -- --nocapture --test-threads=1

use std::time::Duration;

use futures_util::StreamExt;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use naked_core::config::Config;
use naked_core::history::ConversationHistory;
use naked_core::loop_::{AgentLoop, LoopConfig};
use naked_core::provider::{ChatRequest, Provider};
use naked_core::tool::Tool;
use naked_core::tool::registry::ToolRegistry;
use naked_core::types::AgentEvent;

// ── Helpers ─────────────────────────────────────────────────────────────────

fn test_config() -> Option<Config> {
    Config::load().ok().filter(|c| !c.providers.is_empty())
}

async fn probe(prov: &dyn Provider, model: &str) -> bool {
    let req = ChatRequest {
        model: model.to_string(),
        system: String::new(),
        messages: vec![serde_json::json!({"role": "user", "content": "Say OK"})],
        tools: vec![],
        max_tokens: 16,
        temperature: None,
        reasoning: None,
    };
    let mut stream = match prov.stream_chat(req).await {
        Ok(s) => s,
        Err(_) => return false,
    };
    stream.next().await.is_some()
}

async fn get_provider(config: &Config) -> Option<(Box<dyn Provider>, String)> {
    for (name, pc) in &config.providers {
        let resolved = match pc.resolved() {
            Ok(r) if !r.api_key.is_empty() && !r.api_key.starts_with('$') => r,
            _ => continue,
        };
        let model = pc
            .models
            .first()
            .cloned()
            .unwrap_or_else(|| "gpt-4o-mini".into());
        let mut r = resolved.clone();
        r.models = vec![model.clone()];
        let prov = naked_core::create_provider(name, r);
        if probe(prov.as_ref(), &model).await {
            eprintln!("  ✓ Using {name}/{model}");
            return Some((prov, model));
        }
        eprintln!("  ✗ {name}/{model} probe failed");
    }
    None
}

fn loop_config(model: &str) -> LoopConfig {
    LoopConfig {
        max_iterations: 10,
        cwd: std::env::temp_dir(),
        model: model.to_string(),
        max_tokens: 2048,
        ..Default::default()
    }
}

struct E2eResult {
    events: Vec<AgentEvent>,
    usage: naked_core::types::TurnUsage,
}

impl E2eResult {
    fn text(&self) -> String {
        self.events
            .iter()
            .filter_map(|e| match e {
                AgentEvent::TextDelta(t) => Some(t.as_str()),
                _ => None,
            })
            .collect()
    }

    fn tool_outputs(&self) -> Vec<(String, String)> {
        self.events
            .iter()
            .filter_map(|e| match e {
                AgentEvent::ToolEnd { name, output, .. } => Some((name.clone(), output.clone())),
                _ => None,
            })
            .collect()
    }

    fn tool_names(&self) -> Vec<String> {
        self.tool_outputs().iter().map(|(n, _)| n.clone()).collect()
    }
}

async fn run_turn(
    provider: Box<dyn Provider>,
    model: &str,
    tools: Vec<Box<dyn Tool>>,
    system: &str,
    prompt: &str,
) -> E2eResult {
    let registry = ToolRegistry::new(tools);
    let config = loop_config(model);
    let agent = AgentLoop::new(provider, registry, config);

    let mut history = ConversationHistory::new(system.to_string());
    history.push_user(prompt);

    let (tx, mut rx) = mpsc::channel(256);
    let cancel = CancellationToken::new();

    let usage = tokio::time::timeout(
        Duration::from_secs(90),
        agent.run(&mut history, tx, cancel, None, None),
    )
    .await
    .unwrap_or_else(|_| {
        eprintln!("  TIMEOUT");
        Ok(Default::default())
    })
    .unwrap_or_default();

    eprintln!(
        "  Usage: {}in/{}out",
        usage.input_tokens, usage.output_tokens
    );

    let mut events = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        events.push(ev);
    }
    E2eResult { events, usage }
}

// ═════════════════════════════════════════════════════════════════════════════
// TEST 1: Model uses todo tool
// ═════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn live_todo_tool() {
    let Some(config) = test_config() else { return };
    let Some((prov, model)) = get_provider(&config).await else {
        eprintln!("SKIP: no working provider");
        return;
    };

    let list = naked_core::tool::todo_tool::TodoList::new();
    let tools: Vec<Box<dyn Tool>> = vec![Box::new(naked_core::tool::todo_tool::TodoTool::new(
        list.clone(),
    ))];

    let r = run_turn(
        prov,
        &model,
        tools,
        "You have a todo tool. ALWAYS use it when asked to add tasks. Be concise.",
        "Add these 3 todos using the todo tool: 1) write tests, 2) fix bugs, 3) deploy. Then list them.",
    )
    .await;

    eprintln!("  Text: {}", r.text());
    eprintln!("  Tools: {:?}", r.tool_names());

    let items = list.list();
    eprintln!(
        "  Todos: {:?}",
        items.iter().map(|i| &i.content).collect::<Vec<_>>()
    );
    assert!(
        !items.is_empty(),
        "Model should have added todos via tool. Text: {}",
        r.text()
    );
    assert!(r.tool_names().contains(&"todo".to_string()));
}

// ═════════════════════════════════════════════════════════════════════════════
// TEST 2: Model uses plan tool
// ═════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn live_plan_tool() {
    let Some(config) = test_config() else { return };
    let Some((prov, model)) = get_provider(&config).await else {
        eprintln!("SKIP: no working provider");
        return;
    };

    let state = naked_core::tool::plan_tool::PlanState::new();
    let tools: Vec<Box<dyn Tool>> = vec![Box::new(naked_core::tool::plan_tool::PlanTool::new(
        state.clone(),
    ))];

    let r = run_turn(
        prov,
        &model,
        tools,
        "You have a plan tool. ALWAYS use it to create plans. Be concise.",
        "Create a plan using the plan tool with steps: analyze, design, implement, test, deploy.",
    )
    .await;

    eprintln!("  Text: {}", r.text());
    eprintln!("  Tools: {:?}", r.tool_names());
    let steps = state.get();
    eprintln!(
        "  Steps: {:?}",
        steps.iter().map(|s| &s.step).collect::<Vec<_>>()
    );
    assert!(
        !steps.is_empty(),
        "Model should have created plan. Text: {}",
        r.text()
    );
    assert!(r.tool_names().contains(&"plan".to_string()));
}

// ═════════════════════════════════════════════════════════════════════════════
// TEST 3: Model uses validate_data
// ═════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn live_validate_json() {
    let Some(config) = test_config() else { return };
    let Some((prov, model)) = get_provider(&config).await else {
        eprintln!("SKIP: no working provider");
        return;
    };

    let tools: Vec<Box<dyn Tool>> =
        vec![Box::new(naked_core::tool::validate_data::ValidateDataTool)];

    let r = run_turn(
        prov,
        &model,
        tools,
        "You have a validate_data tool. ALWAYS use it when asked to validate. Be concise.",
        r#"Validate this JSON using the validate_data tool: {"name":"test","version":1}"#,
    )
    .await;

    eprintln!("  Text: {}", r.text());
    eprintln!("  Outputs: {:?}", r.tool_outputs());
    assert!(r.tool_names().contains(&"validate_data".to_string()));
    let outputs: String = r.tool_outputs().iter().map(|(_, o)| o.as_str()).collect();
    assert!(
        outputs.contains("Valid") || outputs.contains("valid"),
        "Should report valid: {outputs}"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// TEST 4: Model uses review tool
// ═════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn live_review_code() {
    let Some(config) = test_config() else { return };
    let Some((prov, model)) = get_provider(&config).await else {
        eprintln!("SKIP: no working provider");
        return;
    };

    let tmp = tempfile::TempDir::new().unwrap();
    let bad = tmp.path().join("bad.rs");
    std::fs::write(
        &bad,
        "fn process() {\n    // TODO: fix this\n    let x = foo().unwrap();\n    // HACK: workaround\n}\n",
    )
    .unwrap();

    let tools: Vec<Box<dyn Tool>> = vec![Box::new(naked_core::tool::review_tool::ReviewTool)];

    let r = run_turn(
        prov,
        &model,
        tools,
        "You have a review tool. ALWAYS use it when asked to review. Be concise.",
        &format!(
            "Review the code file at {} using the review tool.",
            bad.display()
        ),
    )
    .await;

    eprintln!("  Text: {}", r.text());
    eprintln!("  Outputs: {:?}", r.tool_outputs());
    assert!(r.tool_names().contains(&"review".to_string()));
    let outputs: String = r.tool_outputs().iter().map(|(_, o)| o.as_str()).collect();
    assert!(
        outputs.contains("TODO") || outputs.contains("unwrap") || outputs.contains("issue"),
        "Should find issues: {outputs}"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// TEST 5: Token tracking from real turn
// ═════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn live_token_tracking() {
    let Some(config) = test_config() else { return };
    let Some((prov, model)) = get_provider(&config).await else {
        eprintln!("SKIP: no working provider");
        return;
    };

    let tracker = naked_core::token_tracker::TokenTracker::new();

    let r = run_turn(
        prov,
        &model,
        vec![],
        "Be extremely concise. One sentence max.",
        "Say hello.",
    )
    .await;

    tracker.record(&model, r.usage.input_tokens, r.usage.output_tokens);
    eprintln!("  Summary:\n{}", tracker.summary());

    let snap = tracker.snapshot();
    if let Some(u) = snap.get(&model) {
        assert!(u.total() > 0, "Should have tokens");
    } else {
        eprintln!("  NOTE: no usage returned for {model}");
    }
}
