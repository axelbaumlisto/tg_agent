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

impl AgentCore {
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

        // Resolve context window
        let provider_ctx = self
            .config()
            .providers
            .get(&provider_name)
            .and_then(|pc| pc.context_window);
        let cw = effective
            .context_window
            .or(provider_ctx)
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

        // Fire-and-forget: pre-turn snapshot + memory classification.
        // T1 of PLAN_QUALITY_v1: in addition to the legacy git-stash
        // snapshot we also capture into the side-git SnapshotRepo
        // (`~/.naked/snapshots/<hash>/.git`). This is what the
        // model-callable `revert_turn` tool + the `/restore` slash
        // command read from. Both paths are non-fatal: a missing
        // git binary or read-only fs degrades to a debug log; the
        // turn proceeds.
        let ws = session.workspace.clone();
        let turn_seq = session.history.message_count() as u64;
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
                        tracing::info!(seq = turn_seq, id = short, "pre-turn snapshot captured");
                    }
                    Ok(None) => tracing::debug!("pre-turn snapshot: no changes to capture"),
                    Err(e) => tracing::debug!("pre-turn side snapshot skipped: {e}"),
                },
                Err(e) => tracing::debug!("pre-turn side snapshot init skipped: {e}"),
            }
        });
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
        let (tx, rx) = mpsc::channel(64);
        let (perm_tx, perm_rx) = mpsc::channel::<PermissionResponse>(4);
        let (steer_tx, steer_rx) = mpsc::channel::<crate::types::SteerMessage>(16);

        let sc = self.load_session_config_pub(session_id);
        let effective = self.config().merge_session(&sc);

        // Phase 1: setup session, push message, gather compaction data
        let setup = self.setup_turn(session_id, push, &effective).await?;

        // Phase 2: LLM compaction + prepare history & loop config
        let mut spawn_data = self
            .compact_and_prepare(session_id, &setup, &effective, &tx)
            .await?;

        // Validate model before making API calls
        if let Some(err_msg) =
            crate::turn::validate_model(&self.config(), &setup.provider_name, &setup.model)
        {
            let _ = tx.send(AgentEvent::Error(err_msg)).await;
            let _ = tx.send(AgentEvent::Idle).await;
            if let Some(s) = self.ss.sessions.write().await.get_mut(session_id) {
                s.state = SessionState::Idle;
            }
            return Ok(AgentHandle {
                events: rx,
                permissions: perm_tx,
                steer: steer_tx,
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
        use tracing::Instrument;

        tokio::spawn(
            async move {
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
