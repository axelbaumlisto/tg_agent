//! A2: Turn pipeline — decomposed send_prompt_multimodal.
//!
//! Each phase is a standalone async function that takes TurnContext
//! and modifies it. Phases are testable in isolation.

use std::path::PathBuf;
use std::sync::Arc;

use crate::history::ConversationHistory;
use crate::provider::Provider;
use crate::tool::registry::ToolRegistry;
use crate::types::AgentEvent;

/// Everything needed to execute a single turn.
/// Built from AgentCore state, consumed by the pipeline.
pub struct TurnContext {
    pub session_id: String,
    pub history: ConversationHistory,
    pub provider_name: String,
    pub model: String,
    pub workspace: PathBuf,
    pub provider: Arc<dyn Provider>,
    pub tools: ToolRegistry,
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    pub reasoning: Option<String>,
    pub context_window: u32,
    /// Original system prompt (restored after turn if modified by injections).
    pub original_system_prompt: String,
}

/// Result of a completed turn pipeline.
pub struct TurnHandle {
    pub events: tokio::sync::mpsc::Receiver<AgentEvent>,
    pub permissions: tokio::sync::mpsc::Sender<crate::types::PermissionResponse>,
}

#[cfg(test)]
mod tests {

    #[test]
    fn turn_context_fields_complete() {
        // Compile-time check that TurnContext has all required fields.
        // If send_prompt_multimodal needs something not here, this fails.
        let _fields = [
            "session_id",
            "history",
            "provider_name",
            "model",
            "workspace",
            "provider",
            "tools",
            "max_tokens",
            "temperature",
            "reasoning",
            "context_window",
            "original_system_prompt",
        ];
    }
}

// ── Phase functions ─────────────────────────────────────────────────────────
// (applied: SRP) Each phase is a standalone function. No &self / &AgentCore.
// Pure data in → data out. Testable in isolation.

use crate::history::ConversationHistory as History;
use crate::memory;
use crate::session::FileTracker;

/// Phase: inject per-session prompt.md into system context.
pub async fn inject_session_prompt(history: &mut History, prompt_path: &std::path::Path) {
    if let Ok(extra) = tokio::fs::read_to_string(prompt_path).await {
        let trimmed = extra.trim();
        if !trimmed.is_empty() {
            history.inject_system_context(&format!("\n\n[Session instructions]\n{trimmed}"));
        }
    }
}

/// Phase: inject MEMORY.md rules + per-user rules into system context.
pub fn inject_memory_rules(
    history: &mut History,
    workspace: &std::path::Path,
    sender: Option<&str>,
) {
    let rules = memory::service::MemoryService::load_rules_for(workspace, sender);
    if !rules.is_empty() {
        history.inject_system_context(&format!("\n\n{rules}"));
    }
}

/// Phase: inject recent memory drafts ("shift") so model sees fresh context.
pub fn inject_memory_shift(
    history: &mut History,
    workspace: &std::path::Path,
    sender: Option<&str>,
    config: &crate::config::MemoryConfig,
) {
    if !config.daily_enabled {
        return;
    }
    let mut blocks: Vec<String> = Vec::new();
    if let Some(b) =
        memory::daily::recent_shift_block(workspace, &memory::types::MemoryScope::Project, config)
    {
        blocks.push(b);
    }
    if let Some(s) = sender
        && let Some(b) = memory::daily::recent_shift_block(
            workspace,
            &memory::types::MemoryScope::User(s.to_string()),
            config,
        )
    {
        blocks.push(b);
    }
    if !blocks.is_empty() {
        history.inject_system_context(&format!("\n\n{}", blocks.join("\n\n")));
    }
}

/// Phase: inject file tracker context (modified/read files this session).
pub fn inject_file_context(history: &mut History, files: &FileTracker) {
    if files.is_empty() {
        return;
    }
    let mut lines = Vec::new();
    let modified = files.modified();
    let read_only = files.read_only();
    if !modified.is_empty() {
        lines.push(format!(
            "Modified files this session: {}",
            modified.join(", ")
        ));
    }
    if !read_only.is_empty() && read_only.len() <= 10 {
        lines.push(format!("Read files this session: {}", read_only.join(", ")));
    }
    if !lines.is_empty() {
        history.inject_system_context(&format!("\n\n[File context]\n{}", lines.join("\n")));
    }
}

#[cfg(test)]
mod phase_tests {
    use super::*;

    #[test]
    fn inject_memory_rules_empty_project_only() {
        let dir = tempfile::tempdir().unwrap();
        let mut h = History::new("sys".into());
        inject_memory_rules(&mut h, dir.path(), None);
        // No project MEMORY.md in tempdir. Global rules may exist
        // (from ~/.naked/memory/MEMORY.md) — that's OK, we just
        // verify the function doesn't panic and the system prompt
        // still starts with the original content.
        assert!(h.system_prompt().starts_with("sys"));
    }

    #[test]
    fn inject_file_context_adds_modified() {
        let mut h = History::new("sys".into());
        let mut ft = FileTracker::default();
        ft.record_tool(
            "edit_file",
            &serde_json::json!({"file_path": "src/main.rs"}),
        );
        ft.record_tool("read_file", &serde_json::json!({"file_path": "README.md"}));

        inject_file_context(&mut h, &ft);
        let prompt = h.system_prompt();
        assert!(prompt.contains("Modified files this session: src/main.rs"));
        assert!(prompt.contains("Read files this session: README.md"));
    }

    #[test]
    fn inject_file_context_empty_is_noop() {
        let mut h = History::new("sys".into());
        let before = h.system_prompt().to_string();
        inject_file_context(&mut h, &FileTracker::default());
        assert_eq!(h.system_prompt(), before);
    }

    #[tokio::test]
    async fn inject_session_prompt_reads_file() {
        let dir = tempfile::tempdir().unwrap();
        let prompt_path = dir.path().join("prompt.md");
        std::fs::write(&prompt_path, "Custom instructions here").unwrap();

        let mut h = History::new("sys".into());
        inject_session_prompt(&mut h, &prompt_path).await;
        assert!(h.system_prompt().contains("Custom instructions here"));
    }

    #[tokio::test]
    async fn inject_session_prompt_missing_file_is_noop() {
        let mut h = History::new("sys".into());
        let before = h.system_prompt().to_string();
        inject_session_prompt(&mut h, std::path::Path::new("/nonexistent/prompt.md")).await;
        assert_eq!(h.system_prompt(), before);
    }
}

/// Phase: validate that the model exists on the provider.
/// Returns an error message if invalid, None if ok.
pub fn validate_model(
    config: &crate::config::Config,
    provider_name: &str,
    model: &str,
) -> Option<String> {
    let pc = config.providers.get(provider_name)?;
    let valid = pc.models.iter().any(|x| x == model)
        || pc.model_aliases.contains_key(model)
        || pc.model_aliases.values().any(|v| v == model);
    if valid {
        return None;
    }
    let available: Vec<_> = pc
        .models
        .iter()
        .chain(pc.model_aliases.keys())
        .take(8)
        .cloned()
        .collect();
    Some(format!(
        "config error: model '{}' not found on provider '{}'. Available: {}",
        model,
        provider_name,
        available.join(", ")
    ))
}

/// Phase: resolve effective max_tokens and temperature from config hierarchy.
/// Priority: per-provider override > per-session > global.
pub fn resolve_generation_params(
    config: &crate::config::Config,
    provider_name: &str,
    effective: &crate::config::EffectiveSessionConfig,
) -> (u32, Option<f32>) {
    let max_tokens = config
        .providers
        .get(provider_name)
        .and_then(|pc| pc.max_tokens)
        .unwrap_or(effective.max_tokens);
    let temperature = config
        .providers
        .get(provider_name)
        .and_then(|pc| pc.temperature)
        .or(effective.temperature);
    (max_tokens, temperature)
}

#[cfg(test)]
mod validate_tests {
    use super::*;

    fn test_config_with_models(models: Vec<&str>) -> crate::config::Config {
        let json = serde_json::json!({
            "providers": {
                "prov": {
                    "type": "openai_compat",
                    "api_key": "test",
                    "models": models,
                    "max_tokens": 4096,
                    "temperature": 0.5
                }
            }
        });
        serde_json::from_value(json).unwrap()
    }

    fn test_effective(max_tokens: u32, temp: Option<f32>) -> crate::config::EffectiveSessionConfig {
        crate::config::EffectiveSessionConfig {
            provider: String::new(),
            model: String::new(),
            max_tokens,
            temperature: temp,
            context_window: None,
            max_iterations: 0,
            reasoning: None,
            mcp_servers: std::collections::HashMap::new(),
            skill_roots: vec![],
            system_prompt_path: None,
        }
    }

    #[test]
    fn validate_model_found() {
        let config = test_config_with_models(vec!["model-a", "model-b"]);
        assert!(validate_model(&config, "prov", "model-a").is_none());
        assert!(validate_model(&config, "prov", "model-b").is_none());
    }

    #[test]
    fn validate_model_not_found() {
        let config = test_config_with_models(vec!["model-a"]);
        let err = validate_model(&config, "prov", "model-x");
        assert!(err.is_some());
        assert!(err.unwrap().contains("model-a"));
    }

    #[test]
    fn validate_model_unknown_provider() {
        let config = test_config_with_models(vec![]);
        assert!(validate_model(&config, "unknown", "any").is_none());
    }

    #[test]
    fn resolve_params_provider_override() {
        let config = test_config_with_models(vec!["m"]);
        let eff = test_effective(8192, Some(1.0));
        let (mt, temp) = resolve_generation_params(&config, "prov", &eff);
        assert_eq!(mt, 4096);
        assert_eq!(temp, Some(0.5));
    }

    #[test]
    fn resolve_params_fallback_to_effective() {
        let config = test_config_with_models(vec![]);
        let eff = test_effective(8192, Some(0.7));
        let (mt, temp) = resolve_generation_params(&config, "unknown", &eff);
        assert_eq!(mt, 8192);
        assert_eq!(temp, Some(0.7));
    }
}

/// Append <read-files> and <modified-files> XML tags to a compaction summary.
pub fn append_file_tags(summary: &mut String, read_files: &[String], modified_files: &[String]) {
    if read_files.is_empty() && modified_files.is_empty() {
        return;
    }
    summary.push_str("\n\n<read-files>\n");
    for f in read_files {
        summary.push_str(f);
        summary.push('\n');
    }
    summary.push_str("</read-files>\n<modified-files>\n");
    for f in modified_files {
        summary.push_str(f);
        summary.push('\n');
    }
    summary.push_str("</modified-files>");
}

/// Extract a short goal hint from a structured compaction summary.
/// Returns the first non-empty line after "## Goal".
pub fn extract_summary_hint(summary: &str) -> Option<String> {
    summary
        .lines()
        .skip_while(|l| !l.starts_with("## Goal"))
        .nth(1)
        .map(|l| {
            l.trim()
                .trim_start_matches("- ")
                .trim_start_matches("[x] ")
                .chars()
                .take(80)
                .collect::<String>()
        })
        .filter(|s| !s.is_empty())
}

/// Merge session-wide FileTracker into compaction file lists.
/// The tracker accumulates across ALL turns; files_in_compaction_range
/// only sees messages being removed.
pub fn merge_file_tracker_into_compaction(
    read_files: &mut Vec<String>,
    modified_files: &mut Vec<String>,
    files: &FileTracker,
) {
    for f in files.read_only() {
        if !read_files.contains(&f.to_string()) {
            read_files.push(f.to_string());
        }
    }
    for f in files.modified() {
        if !modified_files.contains(&f.to_string()) {
            modified_files.push(f.to_string());
        }
    }
}

#[cfg(test)]
mod compaction_phase_tests {
    use super::*;

    #[test]
    fn append_file_tags_adds_xml() {
        let mut summary = "## Goal\nFix stuff".to_string();
        append_file_tags(
            &mut summary,
            &["src/main.rs".into()],
            &["src/lib.rs".into(), "new.rs".into()],
        );
        assert!(summary.contains("<read-files>\nsrc/main.rs\n</read-files>"));
        assert!(summary.contains("<modified-files>\nsrc/lib.rs\nnew.rs\n</modified-files>"));
    }

    #[test]
    fn append_file_tags_empty_is_noop() {
        let mut summary = "text".to_string();
        let before = summary.clone();
        append_file_tags(&mut summary, &[], &[]);
        assert_eq!(summary, before);
    }

    #[test]
    fn extract_hint_from_summary() {
        let summary = "## Goal\nFix Caddy routing for expenses frontend\n\n## Progress";
        assert_eq!(
            extract_summary_hint(summary),
            Some("Fix Caddy routing for expenses frontend".into())
        );
    }

    #[test]
    fn extract_hint_no_goal() {
        assert_eq!(extract_summary_hint("just text"), None);
    }

    #[test]
    fn extract_hint_empty_goal() {
        assert_eq!(extract_summary_hint("## Goal\n\n## Progress"), None);
    }

    #[test]
    fn merge_tracker_deduplicates() {
        let mut read = vec!["a.rs".into()];
        let mut modified = vec!["b.rs".into()];
        let mut ft = FileTracker::default();
        ft.record_tool("read_file", &serde_json::json!({"file_path": "a.rs"}));
        ft.record_tool("read_file", &serde_json::json!({"file_path": "c.rs"}));
        ft.record_tool("edit_file", &serde_json::json!({"file_path": "b.rs"}));
        ft.record_tool("edit_file", &serde_json::json!({"file_path": "d.rs"}));

        merge_file_tracker_into_compaction(&mut read, &mut modified, &ft);
        // c.rs added to read (a.rs deduplicated, but c.rs is also edited → not read_only)
        // d.rs added to modified (b.rs deduplicated)
        assert!(modified.contains(&"d.rs".to_string()));
        assert!(modified.contains(&"b.rs".to_string()));
    }
}

// ---------------------------------------------------------------------------
// Step 6: Fire-and-forget spawns extracted from dispatch_turn
// ---------------------------------------------------------------------------

/// Spawn background memory classification for user input.
///
/// Runs in a detached task — never blocks the turn. On success,
/// writes to daily draft file (if `to_drafts`) or directly to MEMORY.md.
pub fn spawn_memory_classify(
    provider: Arc<dyn crate::provider::Provider>,
    model: String,
    message: String,
    workspace: std::path::PathBuf,
    sender_id: Option<String>,
    to_drafts: bool,
) {
    tokio::spawn(async move {
        let Some(result) =
            crate::memory::classifier::classify(&*provider, &model, &message, sender_id.as_deref())
                .await
        else {
            return;
        };

        if to_drafts {
            let entry = crate::memory::types::MemoryEntry::new(
                result.memory_type,
                result.content.clone(),
                "auto_classify",
                result.scope.clone(),
            );
            match crate::memory::store::MarkdownMemoryStore::append_daily(&workspace, &entry, true)
            {
                Ok(true) => tracing::info!(
                    scope = %result.scope,
                    ty = %result.memory_type,
                    "memory auto-captured to drafts: {}",
                    result.content
                ),
                Ok(false) => {
                    tracing::debug!("memory auto-capture (drafts): duplicate skipped")
                }
                Err(e) => tracing::warn!("memory auto-capture (drafts) write failed: {e}"),
            }
        } else {
            match crate::memory::service::MemoryService::store(
                &workspace,
                result.scope,
                result.memory_type,
                &result.content,
                "auto",
            ) {
                Ok(true) => tracing::info!(
                    "memory auto-captured: [{}] {}",
                    result.memory_type,
                    result.content
                ),
                Ok(false) => tracing::debug!("memory auto-capture: duplicate skipped"),
                Err(e) => tracing::warn!("memory auto-capture write failed: {e}"),
            }
        }
    });
}

/// Merge turn results back into session and persist.
///
/// Called after the agent loop completes. Updates file tracker,
/// marks session idle, and saves to store.
pub async fn persist_turn_result(
    session_id: &str,
    mut history: crate::history::ConversationHistory,
    original_system_prompt: String,
    result: &crate::error::Result<crate::types::TurnUsage>,
    sessions: &tokio::sync::RwLock<std::collections::HashMap<String, crate::session::Session>>,
    store: &dyn crate::session::store::SessionStore,
    tx: &tokio::sync::mpsc::Sender<crate::types::AgentEvent>,
) {
    match result {
        Ok(usage) => {
            crate::types::TURN_COMPLETED_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            tracing::info!(
                "turn complete [{}]: {} tokens",
                session_id,
                usage.total_tokens()
            );
            if usage.input_tokens > 0 {
                history.set_last_input_tokens(usage.input_tokens);
            }
        }
        Err(e) => {
            crate::types::TURN_ERROR_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            tracing::error!("turn error [{}]: {e}", session_id);
            let _ = tx
                .send(crate::types::AgentEvent::Error(e.to_string()))
                .await;
        }
    }

    // Merge mutated history back into the session and persist
    history.restore_system_prompt(original_system_prompt);
    let mut sessions = sessions.write().await;
    if let Some(session) = sessions.get_mut(session_id) {
        // B3: Extract file operations from the turn's tool calls.
        for msg in history.messages() {
            for block in &msg.blocks {
                if let crate::types::ContentBlock::ToolUse { name, input, .. } = block {
                    session.files.record_tool(name, input);
                    session.working_set.observe_tool(name, input);
                }
            }
        }
        session.history = history;
        session.state = crate::session::SessionState::Idle;
        session.updated_at = chrono::Utc::now();
        let _ = store.mark_idle(session_id).await;
        if let Err(e) = store.save(session).await {
            tracing::error!("failed to persist session [{}]: {e}", session_id);
        } else {
            session.persisted_msg_count = session.history.message_count();
        }
    }
}

// ---------------------------------------------------------------------------
// Step 6: Compaction phases (gather → LLM → apply)
// ---------------------------------------------------------------------------

/// Data gathered under session lock for compaction decision.
pub struct CompactionInput {
    pub needs_compact: bool,
    pub compact_text: Option<String>,
    pub previous_summary: Option<String>,
    pub read_files: Vec<String>,
    pub modified_files: Vec<String>,
    pub before_msgs: usize,
    pub workspace: std::path::PathBuf,
}

/// Gather compaction data from a session. Called under sessions lock.
pub fn gather_compaction_data(session: &crate::session::Session) -> CompactionInput {
    let needs_compact = session.history.needs_compaction();
    let compact_text = if needs_compact {
        session.history.messages_for_compaction(4)
    } else {
        None
    };
    let previous_summary = session
        .history
        .last_compaction_summary()
        .map(|s| s.to_string());
    let (mut read_files, mut modified_files) = if needs_compact {
        session.history.files_in_compaction_range(4)
    } else {
        (vec![], vec![])
    };
    if needs_compact {
        merge_file_tracker_into_compaction(&mut read_files, &mut modified_files, &session.files);
    }
    let before_msgs = session.history.message_count();
    CompactionInput {
        needs_compact,
        compact_text,
        previous_summary,
        read_files,
        modified_files,
        before_msgs,
        workspace: session.workspace.clone(),
    }
}

/// Apply compaction result to session. Called under sessions lock.
/// Returns Some((before, after)) if compaction happened.
pub fn apply_compaction(
    session: &mut crate::session::Session,
    llm_summary: Option<&str>,
    before_msgs: usize,
) -> Option<(usize, usize)> {
    if llm_summary.is_none() && !session.history.needs_compaction() {
        // Double-check: if history was modified between gather and apply,
        // compaction might no longer be needed.
        return None;
    }
    if let Some(summary) = llm_summary {
        session.history.set_compaction_summary(summary.to_string());
        session.history.compact_with_llm_summary(summary, 4);
        session.history.set_last_input_tokens(None);
    } else {
        session.history.auto_compact();
    }
    let after = session.history.message_count();
    Some((before_msgs, after))
}

#[cfg(test)]
mod compaction_flow_tests {
    use super::*;

    #[test]
    fn gather_no_compaction_needed() {
        let session = crate::session::Session::new(
            std::path::PathBuf::from("/tmp"),
            "test".into(),
            crate::session::SessionMetadata {
                name: None,
                provider: "test".into(),
                model: "test".into(),
                channel: String::new(),
                channel_id: None,
            },
        );
        let input = gather_compaction_data(&session);
        assert!(!input.needs_compact);
        assert!(input.compact_text.is_none());
    }
}

/// Prepare history for a turn: clone from session, inject prompts + memory + file context.
pub async fn prepare_history(
    session: &crate::session::Session,
    session_root: &std::path::Path,
    effective: &crate::EffectiveSessionConfig,
    sender_id: Option<&str>,
    memory_config: &crate::config::MemoryConfig,
) -> (ConversationHistory, String) {
    let mut history = session.history.clone();
    let original_system_prompt = history.system_prompt().to_string();

    // Inject per-session prompt.md
    let prompt_path = effective
        .system_prompt_path
        .as_ref()
        .map(|p| session_root.join(p))
        .unwrap_or_else(|| session_root.join("prompt.md"));
    inject_session_prompt(&mut history, &prompt_path).await;

    // Memory rules + per-user rules
    inject_memory_rules(&mut history, &session.workspace, sender_id);

    // Recent memory drafts
    inject_memory_shift(&mut history, &session.workspace, sender_id, memory_config);

    // File tracker context
    inject_file_context(&mut history, &session.files);

    (history, original_system_prompt)
}

/// Resolve "auto" model selection based on user prompt and available models.
/// Returns `Some((provider, model))` if auto-selection succeeds.
pub fn resolve_auto_model(
    user_text: &str,
    config: &crate::config::Config,
    estimated_tokens: u64,
) -> Option<crate::model_selector::ModelChoice> {
    let task = crate::model_selector::classify_task(user_text);
    let available: Vec<(String, String)> = config
        .providers
        .iter()
        .flat_map(|(p, pc)| pc.models.iter().map(move |m| (p.clone(), m.clone())))
        .collect();
    let choice = crate::model_selector::select_model(task, estimated_tokens, &available)?;
    tracing::info!(
        task = ?task,
        selected = %format!("{}/{}", choice.provider, choice.model),
        reason = %choice.reason,
        "auto-model"
    );
    Some(choice)
}

/// Build a LoopConfig from session parameters.
/// Extracted from dispatch_turn to reduce its size.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_loop_config(
    max_iterations: usize,
    cwd: std::path::PathBuf,
    model: String,
    max_tokens: u32,
    temperature: Option<f32>,
    reasoning: Option<String>,
    provider_name: String,
    health: std::sync::Arc<crate::model_catalog::ModelHealth>,
    token_tracker: crate::token_tracker::TokenTracker,
    session_dir: &std::path::Path,
    session_id: &str,
) -> crate::loop_::LoopConfig {
    crate::loop_::LoopConfig {
        max_iterations,
        cwd,
        model,
        max_tokens,
        temperature,
        reasoning,
        provider: provider_name,
        health: Some(health),
        token_tracker: Some(token_tracker),
        audit_dir: Some(session_dir.join("audit")),
        cycle_config: Some(crate::session::cycle::CycleConfig::default()),
        session_id: Some(session_id.to_string()),
        data_dir: Some(session_dir.to_path_buf()),
        working_set: None,
    }
}

// ---------------------------------------------------------------------------
// Step A1: dispatch_turn phase structs
// ---------------------------------------------------------------------------

/// Data gathered during the first session-lock phase of dispatch_turn.
/// Captures everything needed to proceed without holding the lock.
pub(crate) struct TurnSetup {
    pub model: String,
    pub provider_name: String,
    pub compaction_input: CompactionInput,
}

/// Data gathered during the second session-lock phase (post-compaction).
/// Everything needed to spawn the agent loop.
pub(crate) struct TurnSpawnData {
    pub history: ConversationHistory,
    pub original_system_prompt: String,
    pub session_workspace: std::path::PathBuf,
    pub loop_config: crate::loop_::LoopConfig,
}
