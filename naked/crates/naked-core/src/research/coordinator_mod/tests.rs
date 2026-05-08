use super::*;
use crate::research::spec::{Finding, ResearchSpec, dedup_hash};
use crate::research::store_fs::FsResearchStore;
use crate::types::{AgentEvent, PermissionResponse};
use async_trait::async_trait;
use tempfile::tempdir;
use tokio::sync::{Mutex, mpsc};

#[test]
fn parse_provider_model_pair_splits_on_first_slash() {
    assert_eq!(
        parse_provider_model_pair("qwen/qwen3.6-plus"),
        Some(("qwen".into(), "qwen3.6-plus".into()))
    );
    assert_eq!(
        parse_provider_model_pair("kimi-code/kimi-for-coding"),
        Some(("kimi-code".into(), "kimi-for-coding".into()))
    );
}

#[test]
fn parse_provider_model_pair_preserves_inner_slashes() {
    // OpenRouter-style ids — the model half can contain `/`.
    assert_eq!(
        parse_provider_model_pair("openrouter/anthropic/claude-3.5-sonnet"),
        Some(("openrouter".into(), "anthropic/claude-3.5-sonnet".into(),))
    );
}

#[test]
fn parse_provider_model_pair_no_slash_returns_none() {
    // Backward-compat: bare model names route through the current
    // provider, the same as before this helper existed.
    assert_eq!(parse_provider_model_pair("qwen3.6-plus"), None);
    assert_eq!(parse_provider_model_pair("MiniMax-M2.5"), None);
}

#[test]
fn parse_provider_model_pair_rejects_empty_halves() {
    assert_eq!(parse_provider_model_pair("/qwen3.6-plus"), None);
    assert_eq!(parse_provider_model_pair("qwen/"), None);
    assert_eq!(parse_provider_model_pair("/"), None);
    // Whitespace-only halves are also rejected.
    assert_eq!(parse_provider_model_pair("  /qwen3.6-plus"), None);
    assert_eq!(parse_provider_model_pair("qwen/   "), None);
}

#[test]
fn parse_provider_model_pair_trims_whitespace() {
    assert_eq!(
        parse_provider_model_pair("  kimi-code  /  kimi-for-coding  "),
        Some(("kimi-code".into(), "kimi-for-coding".into()))
    );
}

fn make_spec(id: &str, topic: &str) -> ResearchSpec {
    ResearchSpec {
        id: id.to_string(),
        topic: topic.to_string(),
        sources: vec!["https://example.com".into()],
        interval_seconds: None,
        run_at: None,
        cron: None,
        task_timeout_seconds: None,
        session_id: None,
        chat_id: None,
        thread_id: None,
        provider: None,
        model: None,
        max_iterations: None,
        max_wall_seconds: Some(5),
        created_at: Utc::now(),
        paused: false,
        pause_reason: None,
    }
}

/// Test double: records every `config.default_model` seen, always
/// returns a provider error so the coordinator walks the entire
/// (selector-trimmed) fallback chain. Used by the Phase 2 tests to
/// assert that `try_start_with_fallback` consults the capability
/// selector before dispatching.
struct RecordingFailingRunner {
    seen_models: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl AgentRunner for RecordingFailingRunner {
    async fn start_research_turn(
        &self,
        _spec: &ResearchSpec,
        _prompt: &str,
        config: &CoordinatorConfig,
        _run_id: &str,
    ) -> Result<(AgentHandle, String, String)> {
        let m = config
            .default_model
            .clone()
            .unwrap_or_else(|| "(none)".into());
        self.seen_models.lock().await.push(m);
        Err(crate::error::AgentError::Provider(
            "recording runner forces fallback walk".into(),
        ))
    }

    async fn cleanup_research_session(&self, _session_id: &str) {}
}

fn cap_provider(
    models: &[&str],
    caps: &[(&str, crate::model_catalog::ModelCapabilities)],
) -> crate::config::ProviderConfig {
    crate::config::ProviderConfig {
        provider_type: "openai_compat".into(),
        api_key: "k".into(),
        api_keys: Vec::new(),
        base_url: None,
        models: models.iter().map(|s| s.to_string()).collect(),
        max_tokens: None,
        temperature: None,
        context_window: None,
        headers: Default::default(),
        supports_vision: None,
        model_aliases: Default::default(),
        capabilities: caps
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect(),
    }
}

#[tokio::test]
async fn try_start_with_fallback_soft_mode_walks_entire_chain() {
    // Soft mode (enforce=false) must preserve legacy behaviour: every
    // fallback is attempted, in order, even if the selector would
    // have dropped it in hard mode.
    let tmp = tempdir().unwrap();
    let store: Arc<dyn ResearchStore> = Arc::new(FsResearchStore::new(tmp.path().to_path_buf()));
    let spec = make_spec("sel-soft", "x");
    store.create_spec(&spec).await.unwrap();

    let seen = Arc::new(Mutex::new(Vec::new()));
    let runner = Arc::new(RecordingFailingRunner {
        seen_models: seen.clone(),
    });

    let mut dead_caps = crate::model_catalog::ModelCapabilities::unknown();
    dead_caps.status = crate::model_catalog::ModelStatus::Deprecated;
    let mut providers = HashMap::new();
    providers.insert(
        "zai".into(),
        cap_provider(&["glm-5", "dead-model"], &[("dead-model", dead_caps)]),
    );

    let coord_cfg = CoordinatorConfig {
        default_provider: Some("zai".into()),
        default_model: Some("glm-5".into()),
        fallback_models: vec!["zai/dead-model".into(), "zai/glm-5".into()],
        provider_capabilities: providers,
        enforce_model_capabilities: false,
        ..CoordinatorConfig::default()
    };
    let coord = ResearchCoordinator::new(store.clone(), runner, coord_cfg);
    let _ = coord.run_once("sel-soft").await;

    let seen = seen.lock().await.clone();
    // Primary + 2 fallbacks = 3 attempts. Deprecated stays in soft mode.
    assert_eq!(
        seen.len(),
        3,
        "soft mode must try every entry, got {seen:?}"
    );
    assert!(seen.iter().any(|m| m.ends_with("dead-model")));
}

#[tokio::test]
async fn try_start_with_fallback_hard_mode_drops_deprecated() {
    let tmp = tempdir().unwrap();
    let store: Arc<dyn ResearchStore> = Arc::new(FsResearchStore::new(tmp.path().to_path_buf()));
    let spec = make_spec("sel-hard", "x");
    store.create_spec(&spec).await.unwrap();

    let seen = Arc::new(Mutex::new(Vec::new()));
    let runner = Arc::new(RecordingFailingRunner {
        seen_models: seen.clone(),
    });

    let mut dead = crate::model_catalog::ModelCapabilities::unknown();
    dead.status = crate::model_catalog::ModelStatus::Deprecated;
    let chat_only = crate::model_catalog::ModelCapabilities {
        task_fit: vec![crate::model_catalog::TaskKind::Chat],
        ..crate::model_catalog::ModelCapabilities::unknown()
    };
    let live = crate::model_catalog::ModelCapabilities {
        task_fit: vec![crate::model_catalog::TaskKind::Research],
        ..crate::model_catalog::ModelCapabilities::unknown()
    };
    let mut providers = HashMap::new();
    providers.insert(
        "qwen".into(),
        cap_provider(
            &["qwen3.6-plus", "chat-only-model", "qwen3.5-plus"],
            &[
                ("qwen3.6-plus", dead),
                ("chat-only-model", chat_only),
                ("qwen3.5-plus", live),
            ],
        ),
    );

    let coord_cfg = CoordinatorConfig {
        default_provider: Some("qwen".into()),
        default_model: Some("qwen3.6-plus".into()),
        // Mix: one deprecated, one off-task, one valid.
        fallback_models: vec![
            "qwen/qwen3.6-plus".into(),
            "qwen/chat-only-model".into(),
            "qwen/qwen3.5-plus".into(),
        ],
        provider_capabilities: providers,
        enforce_model_capabilities: true,
        ..CoordinatorConfig::default()
    };
    let coord = ResearchCoordinator::new(store.clone(), runner, coord_cfg);
    let _ = coord.run_once("sel-hard").await;

    let seen = seen.lock().await.clone();
    // Primary always tried (operator-pinned). Fallbacks: only the
    // live qwen3.5-plus survives the selector trim.
    assert_eq!(seen.len(), 2, "hard mode trims chain; got {seen:?}");
    // Primary first — carries whatever default_model we passed.
    assert!(seen[0].contains("qwen3.6-plus"), "primary first: {seen:?}");
    // Then only the live one survives.
    assert!(
        seen[1] == "qwen3.5-plus" || seen[1] == "qwen/qwen3.5-plus",
        "expected only live fallback, got {seen:?}"
    );
}

#[tokio::test]
async fn try_start_with_fallback_hard_mode_rewrites_bare_model_back_to_bare() {
    // Regression guard: in hard mode, if the operator wrote a bare
    // model id (no provider prefix) and the selector is happy with
    // it, we must hand the *same* bare id back to the runner so
    // downstream resolution matches the operator's spelling.
    let tmp = tempdir().unwrap();
    let store: Arc<dyn ResearchStore> = Arc::new(FsResearchStore::new(tmp.path().to_path_buf()));
    let spec = make_spec("sel-bare", "x");
    store.create_spec(&spec).await.unwrap();

    let seen = Arc::new(Mutex::new(Vec::new()));
    let runner = Arc::new(RecordingFailingRunner {
        seen_models: seen.clone(),
    });

    let live = crate::model_catalog::ModelCapabilities {
        task_fit: vec![crate::model_catalog::TaskKind::Research],
        ..crate::model_catalog::ModelCapabilities::unknown()
    };
    let mut providers = HashMap::new();
    providers.insert(
        "kimi-code".into(),
        cap_provider(&["kimi-for-coding"], &[("kimi-for-coding", live)]),
    );

    let coord_cfg = CoordinatorConfig {
        default_provider: Some("kimi-code".into()),
        default_model: Some("kimi-for-coding".into()),
        fallback_models: vec!["kimi-for-coding".into()],
        provider_capabilities: providers,
        enforce_model_capabilities: true,
        ..CoordinatorConfig::default()
    };
    let coord = ResearchCoordinator::new(store.clone(), runner, coord_cfg);
    let _ = coord.run_once("sel-bare").await;

    let seen = seen.lock().await.clone();
    // Primary + 1 fallback = 2 attempts. Fallback must stay bare.
    assert_eq!(seen.len(), 2, "primary + fallback expected, got {seen:?}");
    assert_eq!(
        seen[1], "kimi-for-coding",
        "bare entry must remain bare after round-trip",
    );
}

/// Test double: sends a scripted sequence of events, optionally writing
/// findings to the store mid-stream to simulate the real tool path.
struct ScriptedRunner {
    events: Mutex<Vec<AgentEvent>>,
    findings: Mutex<Vec<Finding>>,
    store: Arc<dyn ResearchStore>,
    close_without_idle: bool,
    delay_per_event: Duration,
}

#[async_trait]
impl AgentRunner for ScriptedRunner {
    async fn start_research_turn(
        &self,
        _spec: &ResearchSpec,
        _prompt: &str,
        _config: &CoordinatorConfig,
        _run_id: &str,
    ) -> Result<(AgentHandle, String, String)> {
        let (tx, rx) = mpsc::channel(16);
        let (perm_tx, _perm_rx) = mpsc::channel::<PermissionResponse>(4);
        let events = std::mem::take(&mut *self.events.lock().await);
        let findings = std::mem::take(&mut *self.findings.lock().await);
        let store = self.store.clone();
        let close_without_idle = self.close_without_idle;
        let delay = self.delay_per_event;
        tokio::spawn(async move {
            for ev in events {
                if delay > Duration::ZERO {
                    tokio::time::sleep(delay).await;
                }
                let _ = tx.send(ev).await;
            }
            for f in findings {
                let _ = store.try_append_finding(&f).await;
            }
            if !close_without_idle {
                let _ = tx.send(AgentEvent::Idle).await;
            }
        });
        let (steer_tx, _steer_rx) = tokio::sync::mpsc::channel(1);
        Ok((
            AgentHandle {
                events: rx,
                permissions: perm_tx,
                steer: steer_tx,
            },
            "test-provider".into(),
            "test-model".into(),
        ))
    }

    async fn cleanup_research_session(&self, _session_id: &str) {}
}

fn finding(spec_id: &str, url: &str, run_id: &str) -> Finding {
    use crate::research::spec::{content_hash, host_path_hash};
    Finding {
        id: uuid::Uuid::new_v4().simple().to_string(),
        research_id: spec_id.to_string(),
        run_id: run_id.to_string(),
        url: url.to_string(),
        title: Some("t".into()),
        excerpt: None,
        price: None,
        listing_date: None,
        source_content: None,
        dedup_hash: dedup_hash(url),
        host_path_hash: host_path_hash(url),
        content_hash: content_hash(""),
        seen_at: Utc::now(),
    }
}

#[tokio::test]
async fn run_once_agent_idle_writes_run_record() {
    let tmp = tempdir().unwrap();
    let store: Arc<dyn ResearchStore> = Arc::new(FsResearchStore::new(tmp.path().to_path_buf()));
    let spec = make_spec("jag-1", "jag");
    store.create_spec(&spec).await.unwrap();

    let runner = Arc::new(ScriptedRunner {
        events: Mutex::new(vec![]),
        findings: Mutex::new(vec![
            finding("jag-1", "https://ex.com/a", "r"),
            finding("jag-1", "https://ex.com/b", "r"),
        ]),
        store: store.clone(),
        close_without_idle: false,
        delay_per_event: Duration::ZERO,
    });
    let coord = ResearchCoordinator::new(store.clone(), runner, CoordinatorConfig::default());
    let report = coord.run_once("jag-1").await.unwrap();

    assert_eq!(report.stop_reason, StopReason::AgentIdle);
    assert_eq!(report.new_findings, 2);
    assert_eq!(report.total_findings_after, 2);
    assert_eq!(report.provider, "test-provider");
    assert_eq!(report.model, "test-model");

    let runs = store.list_runs("jag-1", None).await.unwrap();
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].new_findings, 2);
    assert_eq!(runs[0].stop_reason, "agent_idle");
    assert!(store.read_report("jag-1").await.unwrap().is_some());
}

#[tokio::test]
async fn run_once_timeout_short_circuits_stream() {
    let tmp = tempdir().unwrap();
    let store: Arc<dyn ResearchStore> = Arc::new(FsResearchStore::new(tmp.path().to_path_buf()));
    let mut spec = make_spec("jag-2", "jag");
    spec.max_wall_seconds = Some(1);
    store.create_spec(&spec).await.unwrap();

    // Scripted runner sleeps 5s before emitting Idle — coordinator must
    // time out first and still write a run record.
    let runner = Arc::new(ScriptedRunner {
        events: Mutex::new(vec![AgentEvent::TextDelta("hi".into())]),
        findings: Mutex::new(vec![]),
        store: store.clone(),
        close_without_idle: false,
        delay_per_event: Duration::from_secs(5),
    });
    let coord = ResearchCoordinator::new(store.clone(), runner, CoordinatorConfig::default());
    let report = coord.run_once("jag-2").await.unwrap();

    assert_eq!(report.stop_reason, StopReason::Timeout);
    let runs = store.list_runs("jag-2", None).await.unwrap();
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].stop_reason, "timeout");
}

#[tokio::test]
async fn run_once_with_cancel_short_circuits_when_token_fired() {
    // Scripted runner sleeps 60s before emitting any event. We cancel
    // the token after 100ms and expect the coordinator to return
    // `StopReason::Cancelled` in well under a second.
    let tmp = tempdir().unwrap();
    let store: Arc<dyn ResearchStore> = Arc::new(FsResearchStore::new(tmp.path().to_path_buf()));
    let mut spec = make_spec("cancel-1", "jag");
    spec.max_wall_seconds = Some(60); // generous, cancellation must win
    store.create_spec(&spec).await.unwrap();

    let runner = Arc::new(ScriptedRunner {
        events: Mutex::new(vec![AgentEvent::TextDelta("slow".into())]),
        findings: Mutex::new(vec![]),
        store: store.clone(),
        close_without_idle: false,
        delay_per_event: Duration::from_secs(60),
    });
    let coord = ResearchCoordinator::new(store.clone(), runner, CoordinatorConfig::default());

    let cancel = CancellationToken::new();
    let cancel_fire = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        cancel_fire.cancel();
    });

    let started = std::time::Instant::now();
    let report = coord
        .run_once_with_cancel("cancel-1", cancel)
        .await
        .unwrap();
    let elapsed = started.elapsed();
    assert_eq!(
        report.stop_reason,
        StopReason::Cancelled,
        "expected Cancelled, got {:?}",
        report.stop_reason
    );
    assert!(
        elapsed < Duration::from_secs(2),
        "cancellation should land sub-second, took {elapsed:?}"
    );
    let runs = store.list_runs("cancel-1", None).await.unwrap();
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].stop_reason, "cancelled");
}

#[tokio::test]
async fn drain_events_returns_cancelled_immediately_on_pre_cancelled_token() {
    // Pre-cancelled token: drain_events MUST observe it on the very
    // first iteration and return without consuming any event.
    let (_perm_tx, _perm_rx) = tokio::sync::mpsc::channel(1);
    let (ev_tx, ev_rx) = tokio::sync::mpsc::channel(8);
    let (steer_tx2, _steer_rx2) = tokio::sync::mpsc::channel(1);
    let mut handle = AgentHandle {
        events: ev_rx,
        permissions: _perm_tx,
        steer: steer_tx2,
    };
    // Push some events; they must be ignored.
    ev_tx.send(AgentEvent::TextDelta("a".into())).await.unwrap();
    ev_tx.send(AgentEvent::Idle).await.unwrap();

    let mut stats = DrainStats::default();
    let cancel = CancellationToken::new();
    cancel.cancel();
    let reason = drain_events(&mut handle, &mut stats, &cancel, None, "test-run").await;
    assert_eq!(reason, StopReason::Cancelled);
    assert_eq!(
        stats.text_deltas, 0,
        "no events should have been consumed before cancellation"
    );
}

#[tokio::test]
async fn run_verified_breaks_between_rounds_on_cancel() {
    // Scripted runner emits Idle quickly each round. Cancel the token
    // and assert that no further round starts after cancellation.
    let tmp = tempdir().unwrap();
    let store: Arc<dyn ResearchStore> = Arc::new(FsResearchStore::new(tmp.path().to_path_buf()));
    let spec = make_spec("verif-cancel", "jag");
    store.create_spec(&spec).await.unwrap();

    let runner = Arc::new(ScriptedRunner {
        events: Mutex::new(vec![]),
        findings: Mutex::new(vec![]),
        store: store.clone(),
        close_without_idle: false,
        delay_per_event: Duration::ZERO,
    });
    let coord = ResearchCoordinator::new(store.clone(), runner, CoordinatorConfig::default());
    let cancel = CancellationToken::new();
    cancel.cancel(); // already cancelled

    let started = std::time::Instant::now();
    let result = coord
        .run_verified_with_cancel("verif-cancel", 5, cancel)
        .await;
    let elapsed = started.elapsed();
    assert!(
        result.is_ok(),
        "should still produce a report, got {result:?}"
    );
    assert!(
        elapsed < Duration::from_secs(3),
        "cancelled verified loop should finish quickly, took {elapsed:?}"
    );
}

#[tokio::test]
async fn run_once_paused_spec_is_skipped() {
    let tmp = tempdir().unwrap();
    let store: Arc<dyn ResearchStore> = Arc::new(FsResearchStore::new(tmp.path().to_path_buf()));
    let mut spec = make_spec("jag-p", "jag");
    spec.paused = true;
    store.create_spec(&spec).await.unwrap();

    let runner = Arc::new(ScriptedRunner {
        events: Mutex::new(vec![]),
        findings: Mutex::new(vec![finding("jag-p", "https://ex.com/x", "r")]),
        store: store.clone(),
        close_without_idle: false,
        delay_per_event: Duration::ZERO,
    });
    let coord = ResearchCoordinator::new(store.clone(), runner, CoordinatorConfig::default());
    let report = coord.run_once("jag-p").await.unwrap();
    assert_eq!(report.stop_reason, StopReason::Paused);
    assert_eq!(report.new_findings, 0);
    // Runner never fired — findings still zero.
    assert_eq!(store.count_findings("jag-p").await.unwrap(), 0);
}

#[tokio::test]
async fn run_once_stream_closed_is_reported() {
    let tmp = tempdir().unwrap();
    let store: Arc<dyn ResearchStore> = Arc::new(FsResearchStore::new(tmp.path().to_path_buf()));
    let spec = make_spec("jag-c", "jag");
    store.create_spec(&spec).await.unwrap();

    let runner = Arc::new(ScriptedRunner {
        events: Mutex::new(vec![AgentEvent::TextDelta("bye".into())]),
        findings: Mutex::new(vec![]),
        store: store.clone(),
        close_without_idle: true,
        delay_per_event: Duration::ZERO,
    });
    let coord = ResearchCoordinator::new(store.clone(), runner, CoordinatorConfig::default());
    let report = coord.run_once("jag-c").await.unwrap();
    assert_eq!(report.stop_reason, StopReason::StreamClosed);
}

#[tokio::test]
async fn build_prompt_carries_topic_sources_and_dedup_list() {
    let tmp = tempdir().unwrap();
    let store: Arc<dyn ResearchStore> = Arc::new(FsResearchStore::new(tmp.path().to_path_buf()));
    let spec = make_spec("jag-3", "Jaguar XF cheap Hanoi");
    store.create_spec(&spec).await.unwrap();
    store
        .try_append_finding(&finding("jag-3", "https://chotot.com/ad/42", "prev"))
        .await
        .unwrap();

    let runner = Arc::new(ScriptedRunner {
        events: Mutex::new(vec![]),
        findings: Mutex::new(vec![]),
        store: store.clone(),
        close_without_idle: false,
        delay_per_event: Duration::ZERO,
    });
    let coord = ResearchCoordinator::new(store.clone(), runner, CoordinatorConfig::default());
    let prompt = coord.build_prompt(&spec).await.unwrap();
    assert!(prompt.contains("Jaguar XF cheap Hanoi"));
    assert!(prompt.contains("https://example.com"));
    assert!(prompt.contains("https://chotot.com/ad/42"));
    assert!(prompt.contains("research_save"));
}

/// After splitting the long JS-gated recipe into the
/// `web-browser-playbook` skill, the prompt itself only needs to:
///
/// 1. Tell the agent the playbook *exists* and to load it via the
///    `Skill` tool when it sees one of the gated hosts. Pin both the
///    skill name and the canonical hostnames so a rename of either
///    side breaks this test loudly.
/// 2. Keep the **hard rules around contacts** inline — these are
///    safety contracts the gatekeeper enforces and they MUST be on
///    the agent's screen on every run, not behind an on-demand load.
///    Verifies the literal `Contacts hidden behind site captcha —
///    visit URL` phrase, the no-fabrication rule, and the
///    no-`Liên hệ qua` paraphrase rule.
///
/// The full per-host button text and CSS-selector tables live in
/// `naked/skills/web-browser-playbook/SKILL.md`; that file has its
/// own contract test below.
#[tokio::test]
async fn build_prompt_points_to_browser_skill_and_keeps_contact_safety_hatch() {
    let tmp = tempdir().unwrap();
    let store: Arc<dyn ResearchStore> = Arc::new(FsResearchStore::new(tmp.path().to_path_buf()));
    let spec = make_spec("vn-1", "Da Nang restaurant rentals");
    store.create_spec(&spec).await.unwrap();

    let runner = Arc::new(ScriptedRunner {
        events: Mutex::new(vec![]),
        findings: Mutex::new(vec![]),
        store: store.clone(),
        close_without_idle: false,
        delay_per_event: Duration::ZERO,
    });
    let coord = ResearchCoordinator::new(store.clone(), runner, CoordinatorConfig::default());
    let prompt = coord.build_prompt(&spec).await.unwrap();

    // (1) Pointer to the skill + hostnames the agent should map to it.
    assert!(
        prompt.contains("web-browser-playbook"),
        "prompt must name the skill so the agent can call \
         Skill(skill=\"web-browser-playbook\") — without the literal \
         name the on-demand path is unreachable"
    );
    for host in [
        "alonhadat.com.vn",
        "nhadat24h.net",
        "batdongsan.com.vn",
        "dotproperty.com.vn",
        "mogi.vn",
        "homedy.com",
    ] {
        assert!(
            prompt.contains(host),
            "host `{host}` missing — agent will not know to load the \
             playbook before fetching this domain"
        );
    }

    // (2) Hard contact rules stay inline (gatekeeper-enforced).
    for clause in [
        "Contacts hidden behind site captcha — visit URL",
        "Never invent a phone number",
        "Liên hệ qua",
    ] {
        assert!(
            prompt.contains(clause),
            "safety clause `{clause}` missing — without it the \
             gatekeeper has no shared phrase with the agent and \
             fabricated contacts can slip through"
        );
    }
}

/// The skill body itself is the source of truth for the per-host
/// recipes; the prompt only points to it. If anyone deletes the file
/// or removes one of the host rows, the agent's on-demand load
/// produces an empty/incomplete playbook and the click-to-reveal
/// flow silently degrades to "no contacts found".
///
/// We pin the same surface the prompt promises:
/// - File exists at the canonical project path.
/// - YAML-ish front-matter has the `name:` and `description:` lines
///   the `SkillResolver` and `SkillTool` rely on.
/// - Every host the prompt mentions has at least one row in the
///   playbook.
/// - The four MCP browser tool names we tell the agent to use
///   (`browser_navigate`, `browser_snapshot`, `browser_click`,
///   `browser_wait`) appear at least once each.
/// - The captcha safety prefix is mentioned (so the playbook agrees
///   with the gatekeeper-enforced inline rule).
#[test]
fn web_browser_playbook_skill_file_matches_prompt_contract() {
    // CARGO_MANIFEST_DIR for naked-core is `naked/crates/naked-core`,
    // so the project skills live two levels up.
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let skill_path = manifest
        .join("../../skills/web-browser-playbook/SKILL.md")
        .canonicalize()
        .expect(
            "skills/web-browser-playbook/SKILL.md must exist — the prompt \
             forwards every JS-gated request to it; missing file means \
             broken on-demand load",
        );
    let body = std::fs::read_to_string(&skill_path).expect("skill file readable");

    // Front-matter the SkillResolver / SkillTool depend on.
    assert!(
        body.contains("name: web-browser-playbook"),
        "skill must declare its canonical name in front-matter; \
         SkillResolver matches case-insensitively but the description \
         hook in SkillTool::build_description prefers the declared name"
    );
    assert!(
        body.contains("description:"),
        "skill must have a `description:` line so SkillTool can show \
         it in the tool catalog the LLM reads"
    );

    // Every host the prompt forwards must have a row here.
    for host in [
        "alonhadat.com.vn",
        "nhadat24h.net",
        "batdongsan.com.vn",
        "dotproperty.com.vn",
        "chotot.com",
        "mogi.vn",
    ] {
        assert!(
            body.contains(host),
            "skill body missing host `{host}` — prompt forwards but \
             playbook has no recipe → degraded silently"
        );
    }

    // MCP browser tool names the playbook tells the agent to call.
    for tool in [
        "browser_navigate",
        "browser_snapshot",
        "browser_click",
        "browser_wait",
    ] {
        assert!(
            body.contains(tool),
            "skill body missing MCP tool reference `{tool}` — recipe \
             is incomplete"
        );
    }

    // Safety hatch agreement with the inline prompt rule.
    assert!(
        body.contains("Contacts hidden behind site captcha — visit URL"),
        "skill must mention the literal captcha-fallback phrase so the \
         agent applies it consistently with the gatekeeper-checked \
         phrase from the inline prompt rule"
    );

    // The "hard captcha walls — abandon fast" section is what stopped
    // the live agent from burning the whole wall-clock budget on
    // unsolvable Cloudflare/recaptcha pages (see probe runs #3 and #5
    // in 2026-04-19 CHANGELOG entry). Without it the agent flailed
    // through 6+ browser_evaluate / browser_run_code calls trying to
    // "investigate" the captcha. We pin the unique markers from that
    // section so an accidental delete during a future skill rewrite
    // surfaces as a test failure rather than a regression on live
    // alonhadat / Cloudflare URLs.
    for marker in [
        "Hard captcha walls",
        "xac-thuc",          // alonhadat interstitial path used as detection signal
        "at most ONE retry", // the rule that caps the flailing
        "do not flail",      // explicit anti-pattern phrase agent quotes back
    ] {
        assert!(
            body.to_lowercase().contains(&marker.to_lowercase()),
            "skill body missing hard-captcha guidance marker `{marker}` — \
             without it the agent will burn its whole wall-clock budget \
             on unsolvable bot challenges"
        );
    }

    // Operator-side levers that ship with the agent_role
    // BrowserRuntime field (proxy + Chromium extensions). The
    // playbook is the single source of truth pointing operators
    // at these knobs; if the section disappears, captcha-prone
    // deployments will sit at sub-50% extraction rates without
    // anyone realising there's a config knob to flip.
    for marker in [
        "Residential proxy", // names the technique unambiguously
        "BrowserRuntime",    // field operators set in naked.json
        "CapSolver",         // canonical example extension
        "CAPSOLVER_API_KEY", // env var the extension reads
    ] {
        assert!(
            body.contains(marker),
            "skill body missing operator-lever marker `{marker}` — \
             captcha-prone deployments need the config-side fix \
             documented next to the model-side guidance"
        );
    }
}

#[test]
fn drain_stats_summary_line_is_grep_friendly() {
    // Pin the on-wire shape of the [research-summary] line because
    // `naked research probe` and any future regression script greps
    // for it. Reordering keys, dropping fields, or changing
    // separators silently breaks every downstream consumer.
    let mut s = DrainStats::default();
    s.note_tool("Skill");
    s.note_tool("browser_navigate");
    s.note_tool("browser_navigate");
    s.note_tool_output("Just a moment... Enable JavaScript and cookies");
    s.note_tool_output("plain page body, no markers");
    s.text_deltas = 7;
    s.errors = 0;

    let line = s.summary_line();
    assert!(line.starts_with("[research-summary] "));
    assert!(line.contains("Skill=1"));
    assert!(line.contains("browser_navigate=2"));
    assert!(line.contains("captcha_hits=1"));
    assert!(line.contains("skill_loads=1"));
    assert!(line.contains("text_deltas=7"));
    assert!(line.contains("errors=0"));
}

#[tokio::test]
async fn verify_findings_catches_quality_issues() {
    let tmp = tempdir().unwrap();
    let store: Arc<dyn ResearchStore> = Arc::new(FsResearchStore::new(tmp.path().to_path_buf()));
    let spec = make_spec("gk-1", "test gatekeeper");
    store.create_spec(&spec).await.unwrap();

    // Finding 1: complete — should pass (using a reliably live URL)
    let mut f1 = finding("gk-1", "https://www.google.com", "r1");
    f1.title = Some("Google".into());
    f1.price = Some("free".into());
    f1.listing_date = Some("18/04/2026".into());
    f1.excerpt = Some("A".repeat(250));
    f1.source_content = Some("B".repeat(200));
    store.try_append_finding(&f1).await.unwrap();

    // Finding 2: missing date, short excerpt, no source_content
    let mut f2 = finding("gk-1", "https://httpbin.org/get", "r1");
    f2.title = Some("Httpbin".into());
    f2.price = Some("10 USD".into());
    f2.listing_date = None;
    f2.excerpt = Some("short".into());
    f2.source_content = None;
    store.try_append_finding(&f2).await.unwrap();

    // Finding 3: stale date (2024)
    let mut f3 = finding("gk-1", "https://httpbin.org/status/200", "r1");
    f3.title = Some("Old listing".into());
    f3.price = Some("5 USD".into());
    f3.listing_date = Some("01/01/2024".into());
    f3.excerpt = Some("C".repeat(300));
    f3.source_content = Some("D".repeat(200));
    store.try_append_finding(&f3).await.unwrap();

    let runner = Arc::new(ScriptedRunner {
        events: Mutex::new(vec![]),
        findings: Mutex::new(vec![]),
        store: store.clone(),
        close_without_idle: false,
        delay_per_event: Duration::ZERO,
    });
    let coord = ResearchCoordinator::new(store.clone(), runner, CoordinatorConfig::default());

    let verdict = coord.verify_findings("gk-1").await;

    // f2: missing date, short excerpt, missing source_content
    assert!(
        verdict.missing_dates >= 1,
        "should detect missing date, got {}",
        verdict.missing_dates
    );
    assert!(
        verdict.short_excerpts >= 1,
        "should detect short excerpt, got {}",
        verdict.short_excerpts
    );
    assert!(
        verdict.missing_source_content >= 1,
        "should detect missing source_content, got {}",
        verdict.missing_source_content
    );

    // f3: stale date should be flagged for removal
    assert!(
        verdict.stale_dates >= 1,
        "should detect stale date, got {}",
        verdict.stale_dates
    );
    assert!(
        !verdict.dead_hashes.is_empty(),
        "stale finding should be in dead_hashes"
    );

    // f2 (live URL with quality issues) should be in remediation_urls
    assert!(
        !verdict.remediation_urls.is_empty(),
        "should have remediation URLs"
    );

    // Feedback prompt should include specific sections
    let feedback = coord.build_feedback_prompt("gk-1", &verdict).await.unwrap();
    assert!(
        feedback.contains("Missing listing_date"),
        "feedback should mention missing dates"
    );
    assert!(
        feedback.contains("source_content"),
        "feedback should mention missing source_content"
    );
    assert!(
        feedback.contains("Too-short excerpts"),
        "feedback should mention short excerpts"
    );
}

#[tokio::test]
async fn upsert_finding_updates_existing() {
    let tmp = tempdir().unwrap();
    let store: Arc<dyn ResearchStore> = Arc::new(FsResearchStore::new(tmp.path().to_path_buf()));
    let spec = make_spec("up-1", "upsert test");
    store.create_spec(&spec).await.unwrap();

    let mut f = finding("up-1", "https://example.com/listing/1", "r1");
    f.title = Some("Original title".into());
    f.excerpt = Some("short".into());
    f.listing_date = None;
    f.source_content = None;
    assert!(
        !store.upsert_finding(&f).await.unwrap(),
        "first insert should not be update"
    );
    assert_eq!(store.count_findings("up-1").await.unwrap(), 1);

    // Now upsert with better data
    f.title = Some("Updated title".into());
    f.excerpt = Some("A".repeat(500));
    f.listing_date = Some("18/04/2026".into());
    f.source_content = Some("B".repeat(1000));
    assert!(
        store.upsert_finding(&f).await.unwrap(),
        "second save should be update"
    );
    assert_eq!(
        store.count_findings("up-1").await.unwrap(),
        1,
        "count should not change"
    );

    let findings = store.list_findings("up-1", None).await.unwrap();
    assert_eq!(findings.len(), 1);
    assert_eq!(findings[0].title.as_deref(), Some("Updated title"));
    assert_eq!(findings[0].listing_date.as_deref(), Some("18/04/2026"));
    assert!(findings[0].source_content.is_some());
}

#[test]
fn fuzzy_fingerprint_matches_near_duplicates() {
    let a = FuzzyFingerprint::new(
        Some("Cho thuê căn hộ 70m² 2PN District 7"),
        Some("$500/month"),
    )
    .expect("fp a");
    let b = FuzzyFingerprint::new(
        Some("Cho thuê 70m² 2PN District 7 có bếp đầy đủ"),
        Some("USD 500 / mo"),
    )
    .expect("fp b");
    assert!(a.is_duplicate_of(&b), "near-dup pair must collapse");
    assert!(b.is_duplicate_of(&a));
}

#[test]
fn fuzzy_fingerprint_distinguishes_different_listings() {
    let a = FuzzyFingerprint::new(Some("Cho thuê căn hộ 70m² 2PN District 7"), Some("$500/mo"))
        .expect("fp a");
    // Different area + different district + different price → unrelated.
    let b = FuzzyFingerprint::new(
        Some("Cho thuê căn hộ 120m² 3PN District 2"),
        Some("$900/mo"),
    )
    .expect("fp b");
    assert!(!a.is_duplicate_of(&b));
}

#[test]
fn fuzzy_fingerprint_requires_both_price_and_token_overlap() {
    let a = FuzzyFingerprint::new(Some("Cho thuê căn hộ 70m² 2PN District 7"), Some("$500/mo"))
        .expect("fp a");
    // Same title but a different price should NOT collapse — prevents
    // false merges of two units in the same building at different rates.
    let b = FuzzyFingerprint::new(Some("Cho thuê căn hộ 70m² 2PN District 7"), Some("$650/mo"))
        .expect("fp b");
    assert!(!a.is_duplicate_of(&b));
}

#[test]
fn fuzzy_fingerprint_skips_too_generic_titles() {
    let fp = FuzzyFingerprint::new(Some("Cho thuê 70m²"), Some("$500"));
    // After dropping filler / short tokens this falls below the 3-token
    // floor, so we refuse to fingerprint it (would over-collapse).
    assert!(fp.is_none());
}

#[test]
fn fuzzy_fingerprint_none_for_short_title() {
    let fp = FuzzyFingerprint::new(Some("hi"), Some("100"));
    assert!(fp.is_none(), "title <10 chars should return None");
}

#[test]
fn fuzzy_fingerprint_none_without_title() {
    let fp = FuzzyFingerprint::new(None, Some("1000"));
    assert!(fp.is_none());
}

#[test]
fn fuzzy_fingerprint_none_without_price() {
    // Title alone without price — fingerprint still works (price_digits empty)
    let fp = FuzzyFingerprint::new(Some("beautiful apartment ocean view"), None);
    assert!(fp.is_some(), "should fingerprint even without price");
}

#[test]
fn fuzzy_fingerprint_jaccard_identical() {
    let a = FuzzyFingerprint::new(Some("luxury condo ocean view phuket"), Some("5000000")).unwrap();
    let b = FuzzyFingerprint::new(Some("luxury condo ocean view phuket"), Some("5000000")).unwrap();
    assert!(a.is_duplicate_of(&b));
}

#[test]
fn fuzzy_fingerprint_different_price_not_dup() {
    let a = FuzzyFingerprint::new(Some("luxury condo ocean view phuket"), Some("5000000")).unwrap();
    let b = FuzzyFingerprint::new(Some("luxury condo ocean view phuket"), Some("9999999")).unwrap();
    assert!(
        !a.is_duplicate_of(&b),
        "different prices should not be duplicates"
    );
}
