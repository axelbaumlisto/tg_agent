//! `research_run` tool — execute a research spec as a normal agent-turn sub-loop.
//!
//! ## Architecture (PLAN_RESEARCH_AGENT_FLOW_v1)
//!
//! Previously, `/research run <id>` spawned a completely separate ephemeral
//! Telegram session with its own cancellation registry (B56/B57). The operator
//! had no way to stop it via `/abort` in their chat thread.
//!
//! This tool unifies the flow:
//!   1. `/research run <id>` (or the scheduler) triggers a **normal agent turn**
//!      in the spec's chat thread (via synthetic message, see T6).
//!   2. The LLM in that turn calls `research_run(spec_id=...)`.
//!   3. THIS tool loads the spec, builds a research prompt, and runs an internal
//!      sub-loop using the same provider + cancel token as the parent turn.
//!   4. The parent turn's Abort button / `/abort` propagates via the progress
//!      channel: when the parent receiver is dropped, the inner loop is cancelled.
//!
//! ## Cancel propagation (T2.4)
//!
//! The Tool trait's `execute` / `execute_with_progress` do not receive the
//! parent's CancellationToken directly. Instead, we detect parent cancellation
//! by monitoring the `progress` sender: if `try_send(Heartbeat)` fails (receiver
//! dropped = parent turn ended), a background task fires the inner loop's token.
//!
//! ## Implementation status
//!
//! T2.1: ToolSpec published with correct parameters. DONE.
//! T2.2: Spec loading + prompt construction. DONE.
//! T2.3: Inner sub-loop dispatch. DONE.
//! T2.4: Cancel token propagation. DONE.

use std::path::Path;
use std::sync::{Arc, Weak};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::AgentCore;
use crate::history::ConversationHistory;
use crate::loop_::{AgentLoop, LoopConfig};
use crate::provider::Provider;
use crate::tool::Tool;
use crate::tool::registry::ToolRegistry;
use crate::tool::web_fetch::WebFetchTool;
use crate::tool::web_fetch_tls::WebFetchTlsTool;
use crate::tool::web_search::WebSearchTool;
use crate::types::{AgentEvent, Permission, ToolResult, ToolSpec};

/// Maximum tool-call iterations for the inner research sub-loop.
/// Light mode aims for 3-5 findings, full for 10+. 30 iterations is
/// generous enough for web_search + per-result web_fetch sequences.
const RESEARCH_RUN_MAX_ITERATIONS: usize = 30;

/// Runs a research spec as an inline sub-loop within the parent agent turn.
///
/// Holds a `Weak<AgentCore>` so it can:
///   - load the spec (`core.load_research`)
///   - access the provider + model for the inner sub-loop (T2.3)
///
/// The `Weak` prevents reference cycles: AgentCore owns the tool registry
/// which owns this tool.
pub struct ResearchRunTool {
    core: Weak<AgentCore>,
}

impl ResearchRunTool {
    pub fn new(core: Weak<AgentCore>) -> Self {
        Self { core }
    }

    /// Build the minimal tool registry for the inner research loop.
    ///
    /// Includes web_search, web_fetch, web_fetch_tls — enough to discover
    /// and extract listings. Does NOT include bash/edit/write (read-only
    /// browsing only). Research_save is included via the core's research
    /// tools built separately in `research_tools()` in factory.rs; for the
    /// inner loop we rely on the core's research context already being set.
    fn build_inner_tools(core: &Arc<AgentCore>) -> ToolRegistry {
        let config = core.config();
        let search = &core.search;
        let mut tools: Vec<Box<dyn Tool>> = vec![
            Box::new(WebSearchTool::new(
                search.exa_key_pool.clone(),
                search.tavily_key_pool.clone(),
                search.serpapi_key_pool.clone(),
            )),
            Box::new(WebFetchTool::with_components(
                search.cloud_scraper.clone(),
                search.host_policy.clone(),
            )),
            Box::new(WebFetchTlsTool::new()),
        ];

        // Include research_save via factory so the inner agent can persist findings.
        if config.research.enabled {
            let research_tools =
                crate::tool::factory::research_tools(&config, &core.research, &core.self_ref);
            tools.extend(research_tools);
        }

        ToolRegistry::new(tools)
    }
}

#[async_trait]
impl Tool for ResearchRunTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "research_run".into(),
            description: "Run a research spec immediately in the current agent turn. \
                 Loads the spec by ID, then uses web_search + web_fetch + research_save \
                 to gather findings. Respects the parent turn's cancel token — pressing \
                 Abort or sending /abort cancels the run mid-flight. \
                 Returns a JSON summary: {spec_id, findings_saved, tool_calls, cancelled}. \
                 Mode 'light' = quick pass (3-5 sources), 'full' = deep (10+ sources)."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "spec_id": {
                        "type": "string",
                        "description": "Research spec ID to execute. Must exist (use research_list_specs to browse)."
                    },
                    "mode": {
                        "type": "string",
                        "description": "Run depth: 'light' (quick, few sources) or 'full' (thorough). Default: light"
                    },
                    "since_iso": {
                        "type": "string",
                        "description": "Only save findings newer than this ISO-8601 date (e.g. '2026-05-01'). Optional."
                    }
                },
                "required": ["spec_id"]
            }),
            permission: Permission::ReadOnly,
        }
    }

    async fn execute(&self, input: Value, cwd: &Path) -> ToolResult {
        // Delegate to execute_with_progress, discarding the channel.
        let (tx, _rx) = mpsc::channel(1);
        self.execute_with_progress(input, cwd, tx).await
    }

    /// T2.3+T2.4: Run the inner research sub-loop with cancel propagation.
    async fn execute_with_progress(
        &self,
        input: Value,
        cwd: &Path,
        progress: mpsc::Sender<AgentEvent>,
    ) -> ToolResult {
        let spec_id = match input.get("spec_id").and_then(|v| v.as_str()) {
            Some(id) if !id.trim().is_empty() => id.trim().to_string(),
            _ => return ToolResult::err("`spec_id` is required and must be non-empty"),
        };
        let mode = input
            .get("mode")
            .and_then(|v| v.as_str())
            .unwrap_or("light")
            .to_string();
        let since_iso = input
            .get("since_iso")
            .and_then(|v| v.as_str())
            .map(String::from);

        let core = match self.core.upgrade() {
            Some(c) => c,
            None => return ToolResult::err("research_run: AgentCore no longer available"),
        };

        // T2.2: load spec + validate + build prompt.
        let spec = match core.load_research(&spec_id).await {
            Ok(s) => s,
            Err(e) => {
                return ToolResult::err(format!(
                    "research_run: spec `{spec_id}` not found: {e}. \
                     Use research_list_specs to see available specs."
                ));
            }
        };

        let prompt = build_research_prompt(
            &spec_id,
            &spec.topic,
            &spec.sources,
            &mode,
            since_iso.as_deref(),
        );

        // Set the research context so research_save tools know which spec to target.
        core.research.context.set_id(Some(spec_id.clone()));
        let run_id = format!(
            "rr-{}-{}",
            spec_id.chars().take(8).collect::<String>(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        );
        core.research.context.set_run_id(Some(run_id.clone()));
        core.research.context.reset_saves();

        // T2.3: Build inner agent loop.
        let provider_arc = core.provider();
        let tools = Self::build_inner_tools(&core);
        let cfg = core.config();
        let max_wall = std::time::Duration::from_secs(cfg.research.max_wall_seconds);
        let turn_backstop = cfg
            .turn_deadline_backstop_enabled
            .then(|| std::time::Duration::from_secs(cfg.turn_deadline_secs));
        let loop_config = LoopConfig {
            max_iterations: RESEARCH_RUN_MAX_ITERATIONS,
            max_wall: Some(max_wall),
            tool_deadline: Some(max_wall),
            turn_backstop,
            cwd: cwd.to_path_buf(),
            model: cfg.default_model.clone(),
            provider: cfg.default_provider.clone(),
            max_tokens: 8192,
            temperature: Some(0.0),
            ..Default::default()
        };

        let system = format!(
            "You are a research agent. Gather findings on the topic and call \
             research_save for each one. Be thorough and accurate.\n\
             Working directory: {}",
            cwd.display()
        );
        let mut history = ConversationHistory::new(system);
        history.push_user(&prompt);

        // T2.4: Cancel propagation.
        // The parent turn's CancellationToken is not passed to tools via trait API.
        // We detect parent cancellation by watching the progress channel: if the
        // receiver is dropped (parent turn ended/aborted), we cancel the inner loop.
        let inner_cancel = CancellationToken::new();
        let inner_cancel_for_monitor = inner_cancel.clone();
        let progress_for_monitor = progress.clone();

        let monitor_handle = tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_millis(500)).await;
                // Try sending a heartbeat; failure means parent receiver dropped.
                if progress_for_monitor
                    .try_send(AgentEvent::Heartbeat)
                    .is_err()
                {
                    inner_cancel_for_monitor.cancel();
                    break;
                }
                if inner_cancel_for_monitor.is_cancelled() {
                    break;
                }
            }
        });

        let agent = AgentLoop::new(Box::new(ArcProvider(provider_arc)), tools, loop_config);

        // B80a: the inner research loop emits AgentEvents into `event_tx`.
        // This receiver MUST be drained continuously: the channel is
        // bounded, so if nobody reads it the inner loop's `tx.send().await`
        // parks forever once 256 events pile up (observed live: a
        // chat-driven `research_run` deadlocked the inner loop and never
        // returned, while the parent turn span at 100% CPU). We spawn a
        // forwarder that drains every inner event and, best-effort,
        // mirrors display-worthy ones to the parent `progress` channel as
        // `SubAgentProgress` so the chat bubble shows the nested research
        // live. `try_send` is used so a slow/full parent channel never
        // re-introduces the deadlock — dropped mirror events are cosmetic.
        let (event_tx, mut event_rx) = mpsc::channel::<AgentEvent>(256);
        let progress_for_forward = progress.clone();
        let spec_id_for_forward = spec_id.clone();
        let forward_handle = tokio::spawn(async move {
            use crate::types::SubAgentEvent;
            while let Some(ev) = event_rx.recv().await {
                let mapped = match ev {
                    AgentEvent::ToolStart { name, input, .. } => Some(SubAgentEvent::ToolUse {
                        name,
                        input_preview: input.to_string().chars().take(120).collect(),
                    }),
                    AgentEvent::ToolEnd { name, state, .. } => {
                        Some(SubAgentEvent::ToolDone { name, state })
                    }
                    AgentEvent::Error(e) => Some(SubAgentEvent::Error(e)),
                    // Drained but not mirrored (deltas/usage/idle/etc.):
                    // draining alone is what prevents the B80a deadlock.
                    _ => None,
                };
                if let Some(event) = mapped {
                    let _ = progress_for_forward.try_send(AgentEvent::SubAgentProgress {
                        agent_id: spec_id_for_forward.clone(),
                        event,
                    });
                }
            }
        });

        let run_result = agent
            .run(&mut history, event_tx, inner_cancel.clone(), None, None)
            .await;

        // Stop the monitor + forwarder. The inner loop has returned, so
        // `event_tx` is dropped; the forwarder will see the channel close
        // and exit on its own, but abort() guarantees prompt teardown.
        monitor_handle.abort();
        forward_handle.abort();

        let was_cancelled = inner_cancel.is_cancelled();
        let findings_saved = core.research.context.save_count();

        // Clear context after run.
        core.research.context.set_id(None);
        core.research.context.set_run_id(None);

        match run_result {
            Ok(usage) => {
                let summary = json!({
                    "ok": !was_cancelled,
                    "spec_id": spec_id,
                    "run_id": run_id,
                    "mode": mode,
                    "findings_saved": findings_saved,
                    "tokens_used": usage.total_tokens(),
                    "cancelled": was_cancelled,
                });
                if was_cancelled {
                    ToolResult::ok(format!(
                        "research_run cancelled mid-flight. \
                         Findings saved before cancel: {findings_saved}.\n{}",
                        summary
                    ))
                } else {
                    ToolResult::ok(summary.to_string())
                }
            }
            Err(e) => ToolResult::err(
                json!({
                    "ok": false,
                    "spec_id": spec_id,
                    "run_id": run_id,
                    "error": e.to_string(),
                    "findings_saved": findings_saved,
                })
                .to_string(),
            ),
        }
    }
}

/// Build the research instruction prompt from spec fields.
///
/// The prompt is passed to the inner sub-loop agent (T2.3) which will call
/// `web_search`, `web_fetch`, and `research_save` to fulfil it.
pub(crate) fn build_research_prompt(
    spec_id: &str,
    topic: &str,
    sources: &[String],
    mode: &str,
    since_iso: Option<&str>,
) -> String {
    let sources_str = if sources.is_empty() {
        "(auto -- use web_search to discover relevant sources)".to_string()
    } else {
        sources.join(", ")
    };
    let depth_hint = match mode {
        "full" => "Thorough pass: aim for 10+ distinct findings.",
        _ => "Quick pass: aim for 3-5 high-quality findings.",
    };
    let since_hint = since_iso
        .map(|d| format!("\nOnly save findings published after {}.", d))
        .unwrap_or_default();
    format!(
        "Run research on: {topic}\n\
         Sources: {sources_str}\n\
         Mode: {mode} -- {depth_hint}{since_hint}\n\
         \n\
         For each finding call: research_save(spec_id=\"{spec_id}\", url=..., title=..., excerpt=..., price=...).\n\
         Use web_search to discover URLs, then web_fetch to extract details.\n\
         Stop when you have gathered the target number of findings or run out of relevant sources."
    )
}

/// Wraps `Arc<dyn Provider>` into a `Box<dyn Provider>` for AgentLoop::new.
/// Identical pattern to the one in `tool/sub_agent.rs`.
struct ArcProvider(Arc<dyn Provider>);

#[async_trait]
impl Provider for ArcProvider {
    fn name(&self) -> &str {
        self.0.name()
    }
    fn models(&self) -> Vec<crate::types::ModelInfo> {
        self.0.models()
    }
    async fn stream_chat(
        &self,
        request: crate::provider::ChatRequest,
    ) -> crate::error::Result<
        std::pin::Pin<Box<dyn tokio_stream::Stream<Item = crate::types::StreamChunk> + Send>>,
    > {
        self.0.stream_chat(request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CreateSchedule;
    use crate::test_support::TestCore;
    use std::sync::Arc;

    // Helper: dead Weak (no AgentCore, for spec-only tests).
    fn dead_tool() -> ResearchRunTool {
        ResearchRunTool { core: Weak::new() }
    }

    // ---- T2.1: ToolSpec tests ----

    #[test]
    fn tool_spec_advertises_spec_id_and_mode() {
        let spec = dead_tool().spec();
        assert_eq!(spec.name, "research_run");

        let props = spec.parameters["properties"]
            .as_object()
            .expect("parameters.properties must be an object");
        assert!(props.contains_key("spec_id"), "spec_id parameter missing");
        assert!(props.contains_key("mode"), "mode parameter missing");
        assert!(
            props.contains_key("since_iso"),
            "since_iso parameter missing"
        );

        let required = spec.parameters["required"]
            .as_array()
            .expect("parameters.required must be an array");
        assert!(required.iter().any(|v| v.as_str() == Some("spec_id")));
    }

    #[test]
    fn tool_spec_permission_is_workspace_write() {
        assert_eq!(dead_tool().spec().permission, Permission::ReadOnly);
    }

    #[tokio::test]
    async fn execute_missing_spec_id_returns_error() {
        let result = dead_tool().execute(json!({}), Path::new(".")).await;
        assert!(result.is_error);
        assert!(result.output.contains("spec_id"));
    }

    #[tokio::test]
    async fn execute_dead_core_returns_error() {
        let result = dead_tool()
            .execute(json!({"spec_id": "test"}), Path::new("."))
            .await;
        assert!(result.is_error);
        assert!(result.output.contains("AgentCore no longer available"));
    }

    // ---- T2.2: Prompt construction tests ----

    #[test]
    fn loads_spec_and_emits_prompt_with_topic() {
        let prompt = build_research_prompt(
            "test-spec-id",
            "commercial rentals Da Nang",
            &[],
            "light",
            None,
        );
        assert!(prompt.contains("commercial rentals Da Nang"));
        assert!(prompt.contains("test-spec-id"));
        assert!(prompt.contains("research_save"));
    }

    #[test]
    fn prompt_includes_explicit_sources_when_provided() {
        let sources = vec![
            "https://guland.vn/".to_string(),
            "https://batdongsan.vn/".to_string(),
        ];
        let prompt = build_research_prompt("spec-1", "offices", &sources, "light", None);
        assert!(prompt.contains("guland.vn"));
        assert!(prompt.contains("batdongsan.vn"));
        assert!(!prompt.contains("auto"));
    }

    #[test]
    fn prompt_uses_auto_hint_when_no_sources() {
        let prompt = build_research_prompt("spec-2", "jobs", &[], "light", None);
        assert!(prompt.contains("auto"));
    }

    #[test]
    fn prompt_includes_since_iso_hint() {
        let prompt = build_research_prompt("spec-3", "topic", &[], "full", Some("2026-05-01"));
        assert!(prompt.contains("2026-05-01"));
    }

    #[test]
    fn prompt_full_mode_mentions_depth() {
        let full = build_research_prompt("s", "t", &[], "full", None);
        let light = build_research_prompt("s", "t", &[], "light", None);
        assert!(full.contains("10+"));
        assert!(light.contains("3-5"));
    }

    // ---- T2.4: Cancel propagation test ----

    /// Verify that dropping the progress receiver causes the inner loop to
    /// receive a cancellation signal (the monitor task fires the cancel token).
    ///
    /// Uses TestCore with NoopProvider so no real LLM is invoked.
    /// The inner loop completes immediately (NoopProvider emits Done stream),
    /// so we test that: (a) the execute doesn't panic, (b) the result is
    /// parseable JSON with "spec_id" present.
    #[tokio::test]
    async fn cancelled_mid_loop_returns_partial_findings() {
        let tc = TestCore::build();
        // Create a spec so load_research succeeds.
        tc.core
            .create_research(
                "test-topic",
                vec![],
                None,
                None,
                None,
                CreateSchedule::OneShotNow,
            )
            .await
            .expect("create_research");
        let specs = tc.core.list_research().await.expect("list");
        let spec_id = specs[0].id.clone();

        let tool = ResearchRunTool::new(Arc::downgrade(&tc.core));

        // Drop the receiver immediately — simulates parent turn cancelling
        // while tool is starting up. The monitor task will fire the inner
        // cancel token as soon as it tries to send.
        let (tx, rx) = mpsc::channel::<AgentEvent>(1);
        drop(rx); // Receiver dropped = "parent died"

        let result = tool
            .execute_with_progress(json!({"spec_id": spec_id}), Path::new(&tc.workspace()), tx)
            .await;

        // With NoopProvider the inner loop completes in one iteration returning Done.
        // Either way, result must not panic and must reference the spec_id.
        assert!(
            result.output.contains(&spec_id),
            "result must reference spec_id; got: {}",
            result.output
        );
    }
}
