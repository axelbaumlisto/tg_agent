//! Turn dispatch — setup_turn, dispatch_turn, session_mcp_servers, build_tool_registry_for.

use crate::config::EffectiveSessionConfig;
use crate::error::{AgentError, Result};
use crate::history;
use crate::loop_::AgentLoop;
use crate::mcp::client::McpServer;
use crate::provider::Provider;
use crate::session::SessionState;
use crate::tool::registry::ToolRegistry;
use crate::types::{AgentEvent, AgentHandle, PermissionResponse};
use crate::{AgentCore, UserPush};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

#[cfg(test)]
static DISPATCH_AGENT_LOOP_RUN_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

impl AgentCore {
    async fn mark_session_idle_after_dispatch_reject(&self, session_id: &str) {
        if let Some(session) = self.ss.sessions.write().await.get_mut(session_id) {
            session.state = SessionState::Idle;
        }
        let _ = self.ss.store.mark_idle(session_id).await;
    }

    /// Connect any extra MCP servers needed by a session (additive over global).
    async fn session_mcp_servers(
        &self,
        session_id: &str,
        effective: &EffectiveSessionConfig,
    ) -> Vec<Arc<McpServer>> {
        let extra_names: Vec<String> = effective
            .mcp_servers
            .keys()
            .filter(|name| !self.config().mcp_servers.contains_key(*name))
            .cloned()
            .collect();

        if extra_names.is_empty() {
            return Vec::new();
        }

        if let Some(cached) = self.catalog.session_mcp.read().await.get(session_id) {
            return cached.clone();
        }

        let mut servers = Vec::new();
        for name in &extra_names {
            if let Some(mut cfg) = effective.mcp_servers.get(name).cloned() {
                if cfg.name.is_empty() {
                    cfg.name = name.clone();
                }
                match McpServer::connect(&cfg).await {
                    Ok(s) => {
                        tracing::info!("session MCP '{}': {} tools", name, s.tools().len());
                        servers.push(Arc::new(s));
                    }
                    Err(e) => tracing::warn!("session MCP '{name}' connect failed: {e}"),
                }
            }
        }

        self.catalog
            .session_mcp
            .write()
            .await
            .insert(session_id.to_string(), servers.clone());
        servers
    }

    /// Phase 1: acquire session lock, resolve model/provider, push message,
    /// gather compaction data, spawn background tasks. Drops lock before returning.
    async fn setup_turn(
        &self,
        session_id: &str,
        push: UserPush,
        effective: &EffectiveSessionConfig,
    ) -> Result<crate::turn::TurnSetup> {
        let mut sessions = self.ss.sessions.write().await;
        let session = sessions
            .get_mut(session_id)
            .ok_or_else(|| AgentError::SessionNotFound(session_id.to_string()))?;
        session.state = SessionState::Active;
        let _ = self.ss.store.mark_active(session_id).await;

        session.metadata.provider = effective.provider.clone();
        session.metadata.model = effective.model.clone();
        let mut model = session.metadata.model.clone();
        let mut provider_name = session.metadata.provider.clone();

        // Auto model selection
        if model == "auto" {
            let user_text = match &push {
                UserPush::Text(t) => t.as_str(),
                UserPush::Multimodal {
                    classifier_text, ..
                } => classifier_text.as_str(),
            };
            if let Some(choice) = crate::turn::resolve_auto_model(
                user_text,
                &self.config(),
                session.history.estimated_tokens() as u64,
            ) {
                provider_name = choice.provider;
                model = choice.model;
                session.metadata.provider = provider_name.clone();
                session.metadata.model = model.clone();
            }
        }

        // B50: resolve context window with proper precedence
        // (the global `Config.context_window` is now LAST resort, not first;
        // see commit history for the regression that motivated this).
        //   1. SessionConfig override         (`/context_window` user choice)
        //   2. capabilities.<model>.context_window  (per-model authoritative)
        //   3. providers.<x>.context_window         (per-provider)
        //   4. global Config.context_window         (sane fallback)
        //   5. hardcoded `model_context_window()`   (legacy table)
        let cfg = self.config();
        let provider_pc = cfg.providers.get(&provider_name);
        let model_cap_ctx = provider_pc
            .and_then(|pc| pc.capabilities.get(&model))
            .and_then(|cap| cap.context_window);
        let provider_ctx = provider_pc.and_then(|pc| pc.context_window);
        let global_ctx = cfg.context_window;
        let cw = effective
            .context_window
            .or(model_cap_ctx)
            .or(provider_ctx)
            .or(global_ctx)
            .unwrap_or_else(|| history::model_context_window(&model));
        session.history.set_context_window_tokens(cw);
        tracing::info!(
            context_window = cw,
            estimated_tokens = session.history.estimated_tokens(),
            message_count = session.history.message_count(),
            needs_compaction = session.history.needs_compaction(),
            last_input_tokens = ?session.history.last_input_tokens(),
            provider = %provider_name,
            "pre-compact check"
        );

        let classifier_text = match &push {
            UserPush::Text(t) => t.clone(),
            UserPush::Multimodal {
                classifier_text, ..
            } => classifier_text.clone(),
        };
        match push {
            UserPush::Text(t) => session.history.push_user(&t),
            UserPush::Multimodal { blocks, .. } => session.history.push_user_multimodal(blocks),
        }
        session.updated_at = chrono::Utc::now();

        // Fire-and-forget: pre-turn snapshots (when snapshots_enabled=true) + memory classification.
        // T1 of PLAN_QUALITY_v1: in addition to the legacy git-stash
        // snapshot we can also capture into the side-git SnapshotRepo
        // (`~/.naked/snapshots/<hash>/.git`). This is what the
        // model-callable `revert_turn` tool + the `/undo` slash
        // command read from. Both paths are non-fatal: a missing
        // git binary or read-only fs degrades to a debug log; the
        // turn proceeds.
        let snapshots_enabled = self.config().snapshots_enabled;
        let turn_seq = session.history.message_count() as u64;
        if snapshots_enabled {
            let ws = session.workspace.clone();
            tokio::spawn(async move {
                if let Some(msg) = crate::snapshot::pre_turn_snapshot(&ws, turn_seq).await {
                    tracing::debug!(stash = %msg, "pre-turn legacy stash snapshot");
                }
            });
            let ws_side = session.workspace.clone();
            tokio::task::spawn_blocking(move || {
                match crate::snapshot::SnapshotRepo::open_or_init(&ws_side) {
                    Ok(repo) => match repo.capture(&format!("pre-turn:{turn_seq}")) {
                        Ok(Some(id)) => {
                            // R5 of PLAN_RESILIENCE_v1: bump counter +
                            // info-level log so operators see snapshots
                            // in journalctl, not just at debug.
                            crate::types::SNAPSHOT_CAPTURE_COUNT
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let short = &id.as_str()[..id.as_str().len().min(8)];
                            tracing::info!(
                                seq = turn_seq,
                                id = short,
                                "pre-turn snapshot captured"
                            );
                        }
                        Ok(None) => tracing::debug!("pre-turn snapshot: no changes to capture"),
                        Err(e) => tracing::debug!("pre-turn side snapshot skipped: {e}"),
                    },
                    Err(e) => tracing::debug!("pre-turn side snapshot init skipped: {e}"),
                }
            });
        }
        {
            let prov = self.provider_for(&provider_name).await;
            let sender = self.session_sender(session_id).await;
            crate::turn::spawn_memory_classify(
                prov,
                model.clone(),
                classifier_text.clone(),
                session.workspace.clone(),
                sender,
                self.config().memory.auto_classify_to_drafts,
            );
        }

        let ci = crate::turn::gather_compaction_data(session);
        drop(sessions);

        Ok(crate::turn::TurnSetup {
            model,
            provider_name,
            compaction_input: ci,
        })
    }

    pub(crate) async fn dispatch_turn(
        &self,
        session_id: &str,
        push: UserPush,
    ) -> Result<AgentHandle> {
        {
            let mut sessions = self.ss.sessions.write().await;
            let session = sessions
                .get_mut(session_id)
                .ok_or_else(|| AgentError::SessionNotFound(session_id.to_string()))?;
            if session.state == SessionState::Active {
                crate::types::SESSION_DOUBLE_TURN_REJECTED_COUNT
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                // PLAN_RESEARCH_STORE_ATOMIC_v1 §R4 option (c): reject
                // with a typed busy error instead of queueing onto live
                // Session.history. Existing queue_message paths append
                // there directly and can be overwritten when
                // persist_turn_result installs the loop-private history.
                return Err(AgentError::SessionBusy(session_id.to_string()));
            }
            session.state = SessionState::Active;
        }

        let (tx, rx) = mpsc::channel(64);
        let (perm_tx, perm_rx) = mpsc::channel::<PermissionResponse>(4);
        let (steer_tx, steer_rx) = mpsc::channel::<crate::types::SteerMessage>(16);

        let sc = self.load_session_config_pub(session_id);
        let effective = self.config().merge_session(&sc);

        // Phase 1: setup session, push message, gather compaction data
        let setup = match self.setup_turn(session_id, push, &effective).await {
            Ok(setup) => setup,
            Err(e) => {
                self.mark_session_idle_after_dispatch_reject(session_id)
                    .await;
                return Err(e);
            }
        };

        // Phase 2: LLM compaction + prepare history & loop config
        let mut spawn_data = match self
            .compact_and_prepare(session_id, &setup, &effective, &tx)
            .await
        {
            Ok(spawn_data) => spawn_data,
            Err(e) => {
                self.mark_session_idle_after_dispatch_reject(session_id)
                    .await;
                return Err(e);
            }
        };

        // Validate model before making API calls
        if let Some(err_msg) =
            crate::turn::validate_model(&self.config(), &setup.provider_name, &setup.model)
        {
            let _ = tx.send(AgentEvent::Error(err_msg)).await;
            let _ = tx.send(AgentEvent::Idle).await;
            self.mark_session_idle_after_dispatch_reject(session_id)
                .await;
            let cancel = self
                .ss
                .cancels
                .read()
                .await
                .get(session_id)
                .cloned()
                .unwrap_or_else(CancellationToken::new);
            return Ok(AgentHandle {
                events: rx,
                permissions: perm_tx,
                steer: steer_tx,
                abort: cancel,
            });
        }

        // Phase 3: build tools, run hooks, spawn agent loop
        let session_provider = self.provider_for(&setup.provider_name).await;
        let tools = self
            .build_tool_registry_for(
                session_id,
                &effective,
                &session_provider,
                &setup.model,
                &spawn_data.session_workspace,
            )
            .await;
        self.catalog
            .hooks
            .run_context_hooks(spawn_data.history.messages_mut())
            .await;

        let agent_loop = AgentLoop::new(
            crate::provider::provider_to_box(&session_provider),
            tools,
            spawn_data.loop_config,
        );

        let session_id_owned = session_id.to_string();
        let sessions_ref = self.ss.sessions.clone();
        let store_ref = self.ss.store.clone();
        let cancel = self
            .ss
            .cancels
            .read()
            .await
            .get(session_id)
            .cloned()
            .unwrap_or_else(CancellationToken::new);

        let turn_span = tracing::info_span!(
            "agent_turn",
            session = %session_id_owned,
            provider = %setup.provider_name,
            model = %setup.model,
        );
        let abort = cancel.clone();
        use tracing::Instrument;

        tokio::spawn(
            async move {
                #[cfg(test)]
                DISPATCH_AGENT_LOOP_RUN_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                let result = agent_loop
                    .run(
                        &mut spawn_data.history,
                        tx.clone(),
                        cancel,
                        Some(perm_rx),
                        Some(steer_rx),
                    )
                    .await;
                crate::turn::persist_turn_result(
                    &session_id_owned,
                    spawn_data.history,
                    spawn_data.original_system_prompt,
                    &result,
                    &sessions_ref,
                    &*store_ref,
                    &tx,
                )
                .await;
            }
            .instrument(turn_span),
        );

        Ok(AgentHandle {
            events: rx,
            permissions: perm_tx,
            steer: steer_tx,
            abort,
        })
    }

    pub(crate) async fn build_tool_registry_for(
        &self,
        session_id: &str,
        effective: &EffectiveSessionConfig,
        provider: &Arc<dyn Provider>,
        model: &str,
        workspace: &Path,
    ) -> ToolRegistry {
        let sender_id = self.session_sender(session_id).await;

        let mut tools = crate::tool::factory::core_tools(&crate::tool::factory::CoreToolCtx {
            config: &self.config(),
            remote_ctx: &self.catalog.remote_ctx,
            agent_registry: &self.catalog.agent_registry,
            search: &self.search,
            provider,
            model,
            workspace,
            sender_id,
            todo_list: &self.shared_tools.todo_list,
            plan_state: &self.shared_tools.plan_state,
            fff_registry: &self.shared_tools.fff_registry,
            fs_cache: &self.shared_tools.fs_cache,
            persistent_bash: &self.shared_tools.persistent_bash,
            session_id,
        })
        .await;

        tools.extend(crate::tool::factory::research_tools(
            &self.config(),
            &self.research,
            &self.self_ref,
        ));

        tools.extend(crate::tool::factory::skill_tools(&effective.skill_roots));

        let session_mcp = self.session_mcp_servers(session_id, effective).await;
        tools.extend(
            crate::tool::factory::mcp_tools(&self.catalog.mcp_registry, &session_mcp).await,
        );

        tools.extend(crate::tool::factory::extra_tools(&self.catalog.extra_tool_factories).await);

        ToolRegistry::new(tools)
    }
}

#[async_trait::async_trait]
impl crate::services::ToolBuilder for AgentCore {
    async fn build_registry(
        &self,
        session_id: &str,
        effective: &crate::EffectiveSessionConfig,
        provider: &Arc<dyn crate::provider::Provider>,
        model: &str,
        workspace: &Path,
    ) -> crate::tool::registry::ToolRegistry {
        self.build_tool_registry_for(session_id, effective, provider, model, workspace)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::DISPATCH_AGENT_LOOP_RUN_COUNT;
    use crate::AgentCore;
    use crate::config::Config;
    use crate::error::{AgentError, Result};
    use crate::provider::{ChatRequest, Provider};
    use crate::session::SessionState;
    use crate::types::{
        AgentEvent, ModelInfo, SESSION_DOUBLE_TURN_REJECTED_COUNT, SNAPSHOT_CAPTURE_COUNT,
        StreamChunk,
    };
    use async_trait::async_trait;
    use futures_util::Stream;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::sync::atomic::Ordering;
    use tokio::sync::{Barrier, Notify, Semaphore};

    static DISPATCH_TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    struct BlockingProvider {
        stream_calls: std::sync::atomic::AtomicU64,
        started: Notify,
        release: Semaphore,
    }

    impl BlockingProvider {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                stream_calls: std::sync::atomic::AtomicU64::new(0),
                started: Notify::new(),
                release: Semaphore::new(0),
            })
        }
    }

    #[async_trait]
    impl Provider for BlockingProvider {
        fn name(&self) -> &str {
            "blocking"
        }

        fn models(&self) -> Vec<ModelInfo> {
            vec![ModelInfo {
                provider: "blocking".into(),
                model_id: "blocking-model".into(),
                display_name: "Blocking".into(),
            }]
        }

        async fn stream_chat(
            &self,
            _request: ChatRequest,
        ) -> Result<Pin<Box<dyn Stream<Item = StreamChunk> + Send>>> {
            self.stream_calls.fetch_add(1, Ordering::Relaxed);
            self.started.notify_one();
            let permit = self
                .release
                .acquire()
                .await
                .expect("release semaphore open");
            permit.forget();
            Ok(Box::pin(futures_util::stream::iter([
                StreamChunk::Text("ok".into()),
                StreamChunk::Done,
            ])))
        }
    }

    fn blocking_core(provider: Arc<BlockingProvider>) -> (tempfile::TempDir, Arc<AgentCore>) {
        blocking_core_with_snapshots(provider, false)
    }

    fn blocking_core_with_snapshots(
        provider: Arc<BlockingProvider>,
        snapshots_enabled: bool,
    ) -> (tempfile::TempDir, Arc<AgentCore>) {
        let tmp = tempfile::tempdir().expect("tempdir");
        let workspace = tmp.path().join("workspace");
        let memory = crate::config::MemoryConfig {
            daily_enabled: false,
            ..Default::default()
        };
        let config = Config {
            workspace,
            session_dir: tmp.path().join("sessions"),
            research: crate::config::ResearchConfig {
                storage_dir: Some(tmp.path().join("research")),
                ..Default::default()
            },
            default_provider: "blocking".into(),
            default_model: "blocking-model".into(),
            memory,
            snapshots_enabled,
            ..Default::default()
        };
        std::fs::create_dir_all(&config.workspace).expect("workspace dir");

        let core = Arc::new(AgentCore::new(config, Box::new(ProviderArc(provider))));
        core.init_self_ref();
        (tmp, core)
    }

    struct ProviderArc(Arc<BlockingProvider>);

    #[async_trait]
    impl Provider for ProviderArc {
        fn name(&self) -> &str {
            self.0.name()
        }

        fn models(&self) -> Vec<ModelInfo> {
            self.0.models()
        }

        async fn stream_chat(
            &self,
            request: ChatRequest,
        ) -> Result<Pin<Box<dyn Stream<Item = StreamChunk> + Send>>> {
            self.0.stream_chat(request).await
        }
    }

    async fn drain_until_idle(mut events: tokio::sync::mpsc::Receiver<AgentEvent>) {
        while let Some(event) = events.recv().await {
            if matches!(event, AgentEvent::Idle) {
                break;
            }
        }
    }

    async fn wait_for_stream_calls(provider: &BlockingProvider, expected: u64) {
        while provider.stream_calls.load(Ordering::Relaxed) < expected {
            provider.started.notified().await;
        }
    }

    async fn init_dirty_workspace_git_repo(workspace: &std::path::Path, dirty_name: &str) {
        tokio::process::Command::new("git")
            .args(["init"])
            .current_dir(workspace)
            .output()
            .await
            .expect("git init command");
        tokio::process::Command::new("git")
            .args(["config", "user.email", "test@test.com"])
            .current_dir(workspace)
            .output()
            .await
            .expect("git config email command");
        tokio::process::Command::new("git")
            .args(["config", "user.name", "Test"])
            .current_dir(workspace)
            .output()
            .await
            .expect("git config name command");
        tokio::fs::write(workspace.join("README.md"), "# test\n")
            .await
            .expect("write readme");
        tokio::process::Command::new("git")
            .args(["add", "."])
            .current_dir(workspace)
            .output()
            .await
            .expect("git add command");
        tokio::process::Command::new("git")
            .args(["commit", "-m", "init"])
            .current_dir(workspace)
            .output()
            .await
            .expect("git commit command");
        tokio::fs::write(workspace.join(dirty_name), "dirty\n")
            .await
            .expect("write dirty file");
    }

    fn remove_snapshot_side_dir(workspace: &std::path::Path) -> std::path::PathBuf {
        let git_dir = crate::snapshot::snapshot_git_dir(workspace).expect("snapshot git dir");
        if let Some(snapshot_dir) = git_dir.parent()
            && snapshot_dir.exists()
        {
            std::fs::remove_dir_all(snapshot_dir).expect("remove snapshot side dir");
        }
        git_dir
    }

    async fn has_naked_pre_turn_stash(workspace: &std::path::Path) -> bool {
        crate::snapshot::list_snapshots(workspace)
            .await
            .iter()
            .any(|entry| entry.message.starts_with("naked:pre-turn:"))
    }

    #[tokio::test]
    async fn snapshots_disabled_turn_creates_no_side_repo() {
        let _guard = DISPATCH_TEST_LOCK.lock().await;
        let provider = BlockingProvider::new();
        let (_tmp, core) = blocking_core_with_snapshots(provider.clone(), false);
        let workspace = core.config().workspace.clone();
        init_dirty_workspace_git_repo(&workspace, "dirty-disabled.txt").await;
        let git_dir = remove_snapshot_side_dir(&workspace);
        let _counter_before = SNAPSHOT_CAPTURE_COUNT.load(Ordering::Relaxed);
        let sid = core.create_session(&workspace).await;

        provider.release.add_permits(1);
        let handle = core.send_prompt(&sid, "probe").await.expect("turn starts");
        drain_until_idle(handle.events).await;

        // SNAPSHOT_CAPTURE_COUNT is process-global and may be bumped by a
        // concurrent enabled-snapshot test under the default cargo harness.
        // The workspace-specific side-dir/stash/file-still-dirty checks are
        // the rerevert-proof assertions for the disabled gate.
        for _ in 0..50 {
            assert!(
                !git_dir.exists(),
                "disabled turn created side repo at {git_dir:?}"
            );
            assert!(
                !has_naked_pre_turn_stash(&workspace).await,
                "disabled turn created a naked:pre-turn stash"
            );
            assert!(
                workspace.join("dirty-disabled.txt").exists(),
                "disabled turn legacy-stashed the dirty file"
            );
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
    }

    #[tokio::test]
    async fn snapshots_enabled_turn_creates_side_repo() {
        let _guard = DISPATCH_TEST_LOCK.lock().await;
        let provider = BlockingProvider::new();
        let (_tmp, core) = blocking_core_with_snapshots(provider.clone(), true);
        let workspace = core.config().workspace.clone();
        tokio::fs::write(workspace.join("capture-me.txt"), "capture me\n")
            .await
            .expect("write capture input");
        let git_dir = remove_snapshot_side_dir(&workspace);
        let counter_before = SNAPSHOT_CAPTURE_COUNT.load(Ordering::Relaxed);
        let expected_counter = counter_before.saturating_add(1);
        let sid = core.create_session(&workspace).await;

        provider.release.add_permits(1);
        let handle = core.send_prompt(&sid, "probe").await.expect("turn starts");
        drain_until_idle(handle.events).await;

        for _ in 0..100 {
            if git_dir.exists()
                && SNAPSHOT_CAPTURE_COUNT.load(Ordering::Relaxed) >= expected_counter
            {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }

        assert!(
            git_dir.exists(),
            "enabled turn did not create side repo at {git_dir:?}"
        );
        assert!(
            SNAPSHOT_CAPTURE_COUNT.load(Ordering::Relaxed) >= expected_counter,
            "enabled turn did not bump snapshot capture counter"
        );
    }

    #[tokio::test]
    async fn concurrent_same_session_rejects_second_and_runs_one_agent_loop() {
        let _guard = DISPATCH_TEST_LOCK.lock().await;
        let provider = BlockingProvider::new();
        let (_tmp, core) = blocking_core(provider.clone());
        let workspace = core.config().workspace.clone();
        let sid = core.create_session(&workspace).await;
        let baseline_rejected = SESSION_DOUBLE_TURN_REJECTED_COUNT.load(Ordering::Relaxed);
        DISPATCH_AGENT_LOOP_RUN_COUNT.store(0, Ordering::Relaxed);

        let barrier = Arc::new(Barrier::new(2));
        let spawn_dispatch = |prompt: &'static str| {
            let core = core.clone();
            let sid = sid.clone();
            let barrier = barrier.clone();
            tokio::spawn(async move {
                barrier.wait().await;
                core.send_prompt(&sid, prompt).await
            })
        };

        let first = spawn_dispatch("one");
        let second = spawn_dispatch("two");

        let (first_result, second_result) = tokio::join!(first, second);
        let results = [
            first_result.expect("first task join"),
            second_result.expect("second task join"),
        ];

        let ok_count = results.iter().filter(|result| result.is_ok()).count();
        let busy_count = results
            .iter()
            .filter(|result| matches!(result, Err(AgentError::SessionBusy(busy)) if busy == &sid))
            .count();

        assert_eq!(ok_count, 1);
        assert_eq!(busy_count, 1);
        wait_for_stream_calls(&provider, 1).await;
        assert_eq!(
            SESSION_DOUBLE_TURN_REJECTED_COUNT.load(Ordering::Relaxed),
            baseline_rejected + 1
        );
        assert_eq!(DISPATCH_AGENT_LOOP_RUN_COUNT.load(Ordering::Relaxed), 1);
        assert_eq!(provider.stream_calls.load(Ordering::Relaxed), 1);

        provider.release.add_permits(1);
        let started_handle = results
            .into_iter()
            .find_map(std::result::Result::ok)
            .expect("one turn starts");
        drain_until_idle(started_handle.events).await;
        assert_eq!(DISPATCH_AGENT_LOOP_RUN_COUNT.load(Ordering::Relaxed), 1);
        assert_eq!(provider.stream_calls.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn sequential_same_session_turn_after_idle_starts_normally() {
        let _guard = DISPATCH_TEST_LOCK.lock().await;
        let provider = BlockingProvider::new();
        let (_tmp, core) = blocking_core(provider.clone());
        let workspace = core.config().workspace.clone();
        let sid = core.create_session(&workspace).await;
        DISPATCH_AGENT_LOOP_RUN_COUNT.store(0, Ordering::Relaxed);

        let first = core
            .send_prompt(&sid, "one")
            .await
            .expect("first turn starts");
        wait_for_stream_calls(&provider, 1).await;
        provider.release.add_permits(1);
        drain_until_idle(first.events).await;
        assert!(!core.is_session_active(&sid).await);

        let second = core
            .send_prompt(&sid, "two")
            .await
            .expect("second turn starts");
        wait_for_stream_calls(&provider, 2).await;
        provider.release.add_permits(1);
        drain_until_idle(second.events).await;

        let sessions = core.ss.sessions.read().await;
        assert_eq!(
            sessions.get(&sid).map(|s| &s.state),
            Some(&SessionState::Idle)
        );
        assert_eq!(DISPATCH_AGENT_LOOP_RUN_COUNT.load(Ordering::Relaxed), 2);
    }
}
