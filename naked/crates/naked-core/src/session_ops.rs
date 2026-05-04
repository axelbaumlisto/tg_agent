//! Session lifecycle — create, send, queue, abort, compact, fork, restore, list, skills, MCP.

use super::*;

impl AgentCore {
    /// Record the current author for an upcoming turn. Used by the memory tool
    /// to resolve `scope=user` without an explicit `user_id`. `None` clears it.
    pub async fn set_session_sender(&self, session_id: &str, sender_id: Option<String>) {
        self.ss.set_session_sender(session_id, sender_id).await
    }

    /// Look up the currently-recorded author for this session, if any.
    pub async fn session_sender(&self, session_id: &str) -> Option<String> {
        self.ss.session_sender(session_id).await
    }

    /// Connect to configured MCP servers.
    /// Connect to all configured MCP servers.
    /// Returns a list of servers that failed to connect (empty = all ok).
    pub async fn init_mcp(&self) -> Vec<crate::mcp::client::McpConnectFailure> {
        let servers = self.config().mcp_server_list();
        if servers.is_empty() {
            return Vec::new();
        }
        {
            let old = self.mcp_registry.read().await;
            old.close_all().await;
        }
        let result = McpRegistry::connect_all_with_diagnostics(&servers).await;
        tracing::info!(
            "MCP: {} tools from {} servers",
            result.registry.all_tools().len(),
            result.registry.servers().len()
        );
        *self.mcp_registry.write().await = result.registry;
        result.failures
    }

    /// Borrow the underlying session store. Exposed for operator commands
    /// (`vacuum-sessions`, future `gc` task) that need to walk all sessions
    /// without going through the in-memory cache.
    pub fn store(&self) -> Arc<dyn SessionStore> {
        self.ss.store.clone()
    }

    /// Load per-session config.json if it exists.
    pub fn load_session_config_pub(&self, session_id: &str) -> SessionConfig {
        let path = self.ss.store.session_root(session_id).join("config.json");
        if path.exists() {
            match SessionConfig::from_file(&path) {
                Ok(sc) => {
                    tracing::info!("loaded per-session config for {}", &session_id[..8]);
                    sc
                }
                Err(e) => {
                    tracing::warn!("bad session config.json for {}: {e}", &session_id[..8]);
                    SessionConfig::default()
                }
            }
        } else {
            SessionConfig::default()
        }
    }

    /// Structured compaction prompt (initial or iterative update).
    fn compaction_prompt(previous_summary: Option<&str>) -> String {
        if let Some(prev) = previous_summary {
            format!(
                "<previous-summary>\n{prev}\n</previous-summary>\n\n\
                 The messages above are NEW conversation since the last summary. \
                 Update the existing summary with new information.\n\
                 RULES: PRESERVE existing info. ADD new progress/decisions. \
                 Move In Progress → Done when completed. UPDATE Next Steps.\n\n\
                 {}",
                Self::COMPACTION_FORMAT
            )
        } else {
            format!(
                "Summarize the conversation above into a structured checkpoint.\n\n{}",
                Self::COMPACTION_FORMAT
            )
        }
    }

    const COMPACTION_FORMAT: &'static str = "\
Use this EXACT format:\n\n\
## Goal\n\
[What is the user trying to accomplish?]\n\n\
## Constraints & Preferences\n\
- [Any constraints or preferences mentioned]\n\n\
## Progress\n\
### Done\n\
- [x] [Completed tasks]\n\n\
### In Progress\n\
- [ ] [Current work]\n\n\
### Blocked\n\
- [Issues if any]\n\n\
## Key Decisions\n\
- **[Decision]**: [Rationale]\n\n\
## Next Steps\n\
1. [What should happen next]\n\n\
## Critical Context\n\
- [File paths, function names, error messages needed to continue]\n\n\
Keep each section concise. Preserve exact paths and identifiers.";

    /// Call the current model to summarize conversation for compaction.
    async fn llm_summarize(
        provider: &dyn Provider,
        model: &str,
        conversation_text: &str,
        previous_summary: Option<&str>,
    ) -> Result<String> {
        use tokio_stream::StreamExt;

        const MAX_COMPACTION_INPUT: usize = 16_000;
        // UTF-8 safe truncation — conversation_text routinely contains
        // multi-byte text (Russian, Vietnamese, emoji), and a raw byte
        // slice panics inside a codepoint. Allocating a new String here is
        // cheap relative to the model call that follows.
        let truncated_owned;
        let input: &str = if conversation_text.chars().count() > MAX_COMPACTION_INPUT {
            truncated_owned = conversation_text
                .chars()
                .take(MAX_COMPACTION_INPUT)
                .collect::<String>();
            truncated_owned.as_str()
        } else {
            conversation_text
        };

        tracing::info!(
            input_chars = input.len(),
            "LLM compaction: sending to model"
        );

        let system = "You are a context compaction assistant. Create a structured summary \
            that another LLM will use to continue the work. Be concise, preserve exact file paths, \
            function names, and error messages. Respond in the same language the user used.";

        let user_prompt = format!(
            "<conversation>\n{input}\n</conversation>\n\n\
             {}",
            Self::compaction_prompt(previous_summary)
        );

        let request = provider::ChatRequest {
            model: model.to_string(),
            system: system.to_string(),
            messages: vec![serde_json::json!({
                "role": "user",
                "content": user_prompt,
            })],
            tools: vec![],
            max_tokens: 2048,
            temperature: Some(0.0),
            reasoning: None,
        };

        let mut stream = provider
            .stream_chat(request)
            .await
            .map_err(|e| AgentError::Provider(format!("compaction LLM call failed: {e}")))?;

        let mut text = String::new();
        while let Some(chunk) = stream.next().await {
            match chunk {
                types::StreamChunk::Text(t) => text.push_str(&t),
                types::StreamChunk::Done => break,
                types::StreamChunk::Error(e) => {
                    return Err(AgentError::Provider(format!(
                        "compaction stream error: {e}"
                    )));
                }
                _ => {}
            }
        }

        if text.trim().is_empty() {
            return Err(AgentError::Provider("compaction LLM returned empty".into()));
        }

        Ok(text)
    }

    /// Get or build a provider by name from the global provider catalog.
    /// Returns the global default if `name` matches `self.config().default_provider`.
    /// Resolve the named provider from the config catalog, building &
    /// caching it on first request. Empty / unknown names fall back to
    /// the default provider this `AgentCore` was constructed with.
    /// Public so out-of-loop callers (CLI gatekeeper, validators) can
    /// pin a non-default provider per call without rebuilding the
    /// whole agent.
    /// Resolve a provider by name. Delegates to ProviderService.
    pub async fn provider_for(&self, provider_name: &str) -> Arc<dyn Provider> {
        self.provider_svc.resolve(provider_name).await
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

        if let Some(cached) = self.session_mcp.read().await.get(session_id) {
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

        self.session_mcp
            .write()
            .await
            .insert(session_id.to_string(), servers.clone());
        servers
    }

    pub async fn create_session(&self, workspace: &Path) -> String {
        self.create_session_with_channel(workspace, "cli").await
    }

    pub async fn create_session_with_channel(&self, workspace: &Path, channel: &str) -> String {
        let system_prompt =
            prompt::resolve_system_prompt(workspace, self.config().system_prompt_path.as_deref());
        let capabilities = self.capabilities_section().await;
        let mut full_prompt = format!(
            "{}\n\n{}\n\n{}",
            system_prompt,
            prompt::environment_section(workspace),
            capabilities
        );
        if self.config().research.enabled {
            full_prompt.push_str("\n\n---\n");
            full_prompt.push_str(&research::briefing::short(&self.config().research));
        }

        if channel == "telegram" {
            full_prompt.push_str("\n\n---\n");
            full_prompt.push_str(
                "Telegram bridge is active.\n\
                 - Messages from the user are forwarded from Telegram.\n\
                 - To send a file back to the user, use the telegram_attach tool with the absolute file path.\n\
                 - Mentioning a file path in plain text will NOT deliver it — you must call telegram_attach.\n\
                 - Keep responses concise — Telegram messages are read on mobile screens.",
            );
        }

        let metadata = SessionMetadata {
            name: None,
            provider: self.config().default_provider.clone(),
            model: self.config().default_model.clone(),
            channel: channel.into(),
            channel_id: None,
        };

        let session = Session::new(workspace.to_path_buf(), full_prompt, metadata);
        let id = session.id.clone();

        if let Err(e) = self.ss.store.save(&session).await {
            tracing::error!("failed to persist new session: {e}");
        }

        // Apply per-session config.json overrides to metadata
        let sc = self.load_session_config_pub(&id);
        let effective = self.config().merge_session(&sc);

        let mut sessions = self.ss.sessions.write().await;
        let mut session = session;
        session.metadata.provider = effective.provider;
        session.metadata.model = effective.model;
        sessions.insert(id.clone(), session);
        id
    }

    pub async fn send_prompt(&self, session_id: &str, text: &str) -> Result<AgentHandle> {
        self.dispatch_turn(session_id, UserPush::Text(text.to_string()))
            .await
    }

    /// Like `send_prompt` but takes pre-built content blocks (text + inline
    /// images). The `classifier_text` is what the background memory classifier
    /// will see — pass the human-readable summary of the message.
    pub async fn send_prompt_multimodal(
        &self,
        session_id: &str,
        blocks: Vec<ContentBlock>,
        classifier_text: String,
    ) -> Result<AgentHandle> {
        self.dispatch_turn(
            session_id,
            UserPush::Multimodal {
                blocks,
                classifier_text,
            },
        )
        .await
    }

    async fn dispatch_turn(&self, session_id: &str, push: UserPush) -> Result<AgentHandle> {
        let (tx, rx) = mpsc::channel(64);
        let (perm_tx, perm_rx) = mpsc::channel::<PermissionResponse>(4);
        let (steer_tx, steer_rx) = mpsc::channel::<crate::types::SteerMessage>(16);

        // Load per-session config overlay (re-read each turn so edits take effect)
        let sc = self.load_session_config_pub(session_id);
        let effective = self.config().merge_session(&sc);

        let mut sessions = self.ss.sessions.write().await;
        let session = sessions
            .get_mut(session_id)
            .ok_or_else(|| AgentError::SessionNotFound(session_id.to_string()))?;
        session.state = SessionState::Active;
        let _ = self.ss.store.mark_active(session_id).await;

        // Update metadata to match effective config (model/provider may change between turns)
        session.metadata.provider = effective.provider.clone();
        session.metadata.model = effective.model.clone();

        let model = session.metadata.model.clone();
        let provider_name = session.metadata.provider.clone();

        // Resolve context window: per-session > per-provider > model lookup > global > 128K
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

        // Background memory classification (non-blocking).
        {
            let provider_ref = self.provider_for(&provider_name).await;
            let sender_id = self.session_sender(session_id).await;
            crate::turn::spawn_memory_classify(
                provider_ref,
                model.clone(),
                classifier_text.clone(),
                session.workspace.clone(),
                sender_id,
                self.config().memory.auto_classify_to_drafts,
            );
        }

        let ci = crate::turn::gather_compaction_data(session);

        // Drop sessions lock before LLM call to avoid blocking other requests
        drop(sessions);

        // LLM-based compaction with deterministic fallback
        let llm_summary = if let Some(text_for_llm) = ci.compact_text {
            tracing::info!(ci.before_msgs, "attempting LLM-based compaction");
            let provider_arc = self.provider_for(&provider_name).await;

            if self.config().memory.daily_enabled && self.config().memory.pre_compaction_flush {
                memory::digest::pre_compaction_flush(
                    &*provider_arc,
                    &model,
                    &ci.workspace,
                    &memory::types::MemoryScope::Project,
                    &text_for_llm,
                )
                .await;
            }

            match Self::llm_summarize(
                &*provider_arc,
                &model,
                &text_for_llm,
                ci.previous_summary.as_deref(),
            )
            .await
            {
                Ok(mut summary) => {
                    crate::turn::append_file_tags(&mut summary, &ci.read_files, &ci.modified_files);
                    tracing::info!("LLM compaction succeeded");
                    Some(summary)
                }
                Err(e) => {
                    tracing::warn!("LLM compaction failed, falling back to deterministic: {e}");
                    None
                }
            }
        } else {
            None
        };

        // Re-acquire sessions lock to apply compaction
        let mut sessions = self.ss.sessions.write().await;
        let session = sessions
            .get_mut(session_id)
            .ok_or_else(|| AgentError::SessionNotFound(session_id.to_string()))?;

        let compacted = if ci.needs_compact {
            crate::turn::apply_compaction(session, llm_summary.as_deref(), ci.before_msgs)
        } else {
            None
        };

        if let Some((before, after)) = compacted {
            let summary_hint = llm_summary
                .as_ref()
                .and_then(|s| crate::turn::extract_summary_hint(s));
            let files_count = ci.read_files.len() + ci.modified_files.len();
            tracing::info!("context compacted: {before} msgs -> {after} msgs");
            let _ = tx
                .send(AgentEvent::ContextCompacted {
                    before_msgs: before,
                    after_msgs: after,
                    summary_hint,
                    files_count,
                })
                .await;
            // After compaction, older turns (and any image blocks they
            // owned) are gone from history. Run a best-effort GC over the
            // session's artifacts dir to reclaim disk for images that no
            // JSONL line still references. Never block the user reply on
            // GC failures — log and move on.
            match self.ss.store.gc_orphan_image_artifacts(session_id).await {
                Ok(n) if n > 0 => {
                    tracing::info!(
                        removed = n,
                        session = session_id,
                        "post-compaction artifact GC reclaimed {n} orphan images"
                    );
                }
                Ok(_) => {}
                Err(e) => {
                    tracing::warn!(
                        session = session_id,
                        "post-compaction artifact GC failed: {e}"
                    );
                }
            }
        }

        let session_root = self.ss.store.session_root(session_id);
        let sender_for_rules = self.session_sender(session_id).await;
        let (mut history, original_system_prompt) = crate::turn::prepare_history(
            session,
            &session_root,
            &effective,
            sender_for_rules.as_deref(),
            &self.config().memory,
        )
        .await;

        let artifacts = self.ss.store.artifacts_dir(session_id);
        if let Err(e) = tokio::fs::create_dir_all(&artifacts).await {
            tracing::warn!("could not create artifacts dir: {e}");
        }

        let cwd = if session.workspace.as_os_str().is_empty() || !session.workspace.exists() {
            artifacts
        } else {
            session.workspace.clone()
        };

        // Per-provider max_tokens / temperature.
        let (eff_max_tokens, eff_temperature) =
            crate::turn::resolve_generation_params(&self.config(), &provider_name, &effective);

        let loop_config = LoopConfig {
            max_iterations: effective.max_iterations,
            cwd,
            model: model.clone(),
            max_tokens: eff_max_tokens,
            temperature: eff_temperature,
            reasoning: effective.reasoning.clone(),
            provider: provider_name.clone(),
            health: Some(self.provider_svc.health()),
        };

        let session_workspace = session.workspace.clone();

        let cancel = CancellationToken::new();
        self.ss
            .cancels
            .write()
            .await
            .insert(session_id.to_string(), cancel.clone());
        drop(sessions);

        // Validate model belongs to provider before making any API calls.
        if let Some(err_msg) = crate::turn::validate_model(&self.config(), &provider_name, &model) {
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

        // Per-session provider (falls back to global if unchanged)
        let session_provider = self.provider_for(&provider_name).await;

        // Build tool registry with per-session MCP + skills
        let tools = self
            .build_tool_registry_for(
                session_id,
                &effective,
                &session_provider,
                &model,
                &session_workspace,
            )
            .await;
        // B6: Run context hooks before the LLM call.
        self.hooks.run_context_hooks(history.messages_mut()).await;

        let agent_loop = AgentLoop::new(provider_to_box(&session_provider), tools, loop_config);

        let session_id_owned = session_id.to_string();
        let sessions_ref = self.ss.sessions.clone();
        let store_ref = self.ss.store.clone();

        // Every turn gets a span with (session_id, provider, model). All
        // events emitted from `agent_loop.run` — tool calls, usage, errors
        // — inherit these attributes, so operators can grep one session's
        // worth of logs by a single `session` field without hunting
        // through chat/thread IDs.
        let turn_span = tracing::info_span!(
            "agent_turn",
            session = %session_id_owned,
            provider = %provider_name,
            model = %model,
        );
        use tracing::Instrument;

        tokio::spawn(
            async move {
                let result = agent_loop
                    .run(
                        &mut history,
                        tx.clone(),
                        cancel,
                        Some(perm_rx),
                        Some(steer_rx),
                    )
                    .await;
                crate::turn::persist_turn_result(
                    &session_id_owned,
                    history,
                    original_system_prompt,
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

    pub async fn is_session_active(&self, session_id: &str) -> bool {
        self.ss.is_session_active(session_id).await
    }

    /// Append a user message to the session history without starting a new turn.
    pub async fn queue_message(&self, session_id: &str, text: &str) {
        if let Some(session) = self.ss.sessions.write().await.get_mut(session_id) {
            session.history.push_user(text);
        }
    }

    /// Append a multimodal user message (text + images) to the history without
    /// starting a new turn. Used when the bot is busy and a new media-bearing
    /// message arrives mid-turn.
    pub async fn queue_message_multimodal(&self, session_id: &str, blocks: Vec<ContentBlock>) {
        if let Some(session) = self.ss.sessions.write().await.get_mut(session_id) {
            session.history.push_user_multimodal(blocks);
        }
    }

    /// Trigger history compaction for a session. Returns (before, after) message counts.
    /// No-op if compaction not needed.
    pub async fn compact_session(&self, session_id: &str) -> Option<(usize, usize)> {
        if let Some(session) = self.ss.sessions.write().await.get_mut(session_id) {
            session.history.auto_compact()
        } else {
            None
        }
    }

    pub async fn abort(&self, session_id: &str) {
        if let Some(cancel) = self.ss.cancels.read().await.get(session_id) {
            cancel.cancel();
        }
        if let Some(session) = self.ss.sessions.write().await.get_mut(session_id) {
            session.state = SessionState::Idle;
        }
    }

    /// Fire-and-forget: ask the LLM to extract any rules-of-thumb from
    /// the (closing) session's transcript and append them to the
    /// project's daily memory draft. Called from `/new`-style handlers
    /// that swap one session for another. Returns immediately; the
    /// summary runs in a background tokio task and never blocks the
    /// caller. No-op when `memory.session_close_summary = false`.
    pub async fn close_session_summary(&self, session_id: &str) {
        if !self.config().memory.daily_enabled || !self.config().memory.session_close_summary {
            return;
        }
        let (workspace, transcript, provider_name, model) = {
            let sessions = self.ss.sessions.read().await;
            let Some(session) = sessions.get(session_id) else {
                return;
            };
            // `messages_for_compaction(0)` returns the full transcript
            // (no recent-tail kept). If there is nothing to summarize
            // (e.g. fresh session), skip.
            let Some(text) = session.history.messages_for_compaction(0) else {
                return;
            };
            (
                session.workspace.clone(),
                text,
                session.metadata.provider.clone(),
                session.metadata.model.clone(),
            )
        };
        let provider = self.provider_for(&provider_name).await;
        tokio::spawn(async move {
            memory::digest::session_close(
                &*provider,
                &model,
                &workspace,
                &memory::types::MemoryScope::Project,
                &transcript,
            )
            .await;
        });
    }

    pub async fn list_sessions(&self) -> Vec<SessionSummary> {
        self.list_sessions_paged(0, usize::MAX).await
    }

    /// Look up the workspace path tied to a live session id.
    ///
    /// Returns `None` if the session was never created or has been
    /// evicted. Callers (e.g. the TG `/memory` operator command) use
    /// this to scope project-memory queries to the same directory the
    /// agent was talking from.
    pub async fn session_workspace(&self, session_id: &str) -> Option<std::path::PathBuf> {
        self.ss.session_workspace(session_id).await
    }

    /// Paged variant of `list_sessions`. Sorts by `updated_at` descending
    /// (most recently touched session first), then applies `skip` + `limit`.
    ///
    /// Callers can pass `limit = usize::MAX` to disable truncation. A
    /// `skip` beyond the total count returns an empty vec — never panics.
    /// Intended for CLI `/sessions --skip N --limit M` and future UI
    /// paging where listing 500 stale sessions would be useless.
    pub async fn list_sessions_paged(&self, skip: usize, limit: usize) -> Vec<SessionSummary> {
        self.ss.list_sessions_paged(skip, limit).await
    }

    pub async fn restore_sessions(&self) -> Result<Vec<String>> {
        let summaries = self.ss.store.list().await?;
        let mut restored = Vec::new();
        for summary in summaries {
            match self.ss.store.load(&summary.id).await {
                Ok(Some(mut session)) => {
                    // Apply per-session config.json overrides (provider/model)
                    let sc = self.load_session_config_pub(&session.id);
                    let effective = self.config().merge_session(&sc);
                    session.metadata.provider = effective.provider;
                    session.metadata.model = effective.model;

                    restored.push(session.id.clone());
                    self.ss
                        .sessions
                        .write()
                        .await
                        .insert(session.id.clone(), session);
                }
                Ok(None) => {
                    tracing::warn!(
                        "session {} listed but not loadable",
                        &summary.id[..8.min(summary.id.len())]
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        "failed to load session {}: {e}",
                        &summary.id[..8.min(summary.id.len())]
                    );
                }
            }
        }
        Ok(restored)
    }

    pub async fn fork_session(
        &self,
        session_id: &str,
        branch_name: Option<String>,
    ) -> Result<String> {
        let sessions = self.ss.sessions.read().await;
        let parent = sessions
            .get(session_id)
            .ok_or_else(|| AgentError::SessionNotFound(session_id.to_string()))?;
        let forked = parent.fork(branch_name);
        let new_id = forked.id.clone();
        self.ss.store.save(&forked).await?;
        drop(sessions);
        self.ss
            .sessions
            .write()
            .await
            .insert(new_id.clone(), forked);
        Ok(new_id)
    }

    pub fn list_skills(&self) -> Vec<(String, String)> {
        let resolver = SkillResolver::new(self.config().skill_roots.clone());
        resolver
            .list()
            .into_iter()
            .map(|(name, hit)| (name, hit.path.display().to_string()))
            .collect()
    }

    pub async fn list_mcp_servers(&self) -> Vec<(String, usize)> {
        let reg = self.mcp_registry.read().await;
        reg.servers()
            .iter()
            .map(|s| (s.name.clone(), s.tools().len()))
            .collect()
    }

    async fn capabilities_section(&self) -> String {
        let mut parts = Vec::new();

        // Skills — same description source as the SkillTool catalog
        // so the LLM sees identical hints in the system preface and
        // in the tool's JSON schema. JSON skills get their description
        // from the spec; MD skills from front-matter.
        let skills = self.list_skills();
        if !skills.is_empty() {
            let mut s = String::from("Available skills (use the Skill tool to activate):\n");
            let resolver = SkillResolver::new(self.config().skill_roots.clone());
            for (name, _path) in &skills {
                let desc = resolver
                    .resolve(name)
                    .and_then(|hit| skill::resolver::read_skill_description(&hit));
                match desc {
                    Some(d) => s.push_str(&format!("- {name}: {d}\n")),
                    None => s.push_str(&format!("- {name}\n")),
                }
            }
            parts.push(s);
        }

        // MCP servers & tools
        let mcp_reg = self.mcp_registry.read().await;
        let servers = mcp_reg.servers();
        if !servers.is_empty() {
            let mut s = String::from("Connected MCP servers and their tools:\n");
            for server in servers {
                let tools = server.tools();
                s.push_str(&format!("- {} ({} tools):", server.name, tools.len()));
                for tool in tools {
                    let desc = tool.description.as_deref().unwrap_or("");
                    s.push_str(&format!("\n  • {}: {desc}", tool.name));
                }
                s.push('\n');
            }
            parts.push(s);
        }

        let out = if parts.is_empty() {
            String::new()
        } else {
            format!("---\nCapabilities:\n{}", parts.join("\n"))
        };
        tracing::debug!(
            target: "naked::capabilities",
            chars = out.len(),
            "[capabilities-prefix] {}",
            out.replace('\n', " ⏎ ").chars().take(2_000).collect::<String>()
        );
        out
    }

    pub async fn refresh_skills_and_mcp(&self) {
        self.init_mcp().await;
        let skills = self.list_skills();
        tracing::info!(
            "Refresh: {} skills, {} MCP servers",
            skills.len(),
            self.mcp_registry.read().await.servers().len()
        );
    }

    async fn build_tool_registry_for(
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
            remote_ctx: &self.remote_ctx,
            agent_registry: &self.agent_registry,
            search: &self.search,
            provider,
            model,
            workspace,
            sender_id,
        })
        .await;

        tools.extend(crate::tool::factory::research_tools(
            &self.config(),
            &self.research,
            &self.self_ref,
        ));

        tools.extend(crate::tool::factory::skill_tools(&effective.skill_roots));

        let session_mcp = self.session_mcp_servers(session_id, effective).await;
        tools.extend(crate::tool::factory::mcp_tools(&self.mcp_registry, &session_mcp).await);

        tools.extend(crate::tool::factory::extra_tools(&self.extra_tool_factories).await);

        ToolRegistry::new(tools)
    }

    // ── Research public API ────────────────────────────────────────────
    //
    // These methods are the thin layer the Telegram bot and CLI call into.
    // They hide the coordinator / store plumbing so callers don't need to
    // build it themselves; the trade-off is that AgentCore carries a research
    // store by construction, which is cheap (no network, one directory).
}

impl AgentCore {
    pub async fn set_session_provider(
        &self,
        session_id: &str,
        provider: Option<&str>,
        model: Option<&str>,
    ) -> Result<()> {
        if let Some(p) = provider
            && !self.config().providers.contains_key(p)
        {
            return Err(AgentError::ProviderNotConfigured(p.to_string()));
        }

        // Validate that the model belongs to the target provider.
        // Resolve the effective provider (explicit or current session's).
        if let Some(m) = model {
            let target_provider = provider
                .map(|s| s.to_string())
                .or_else(|| {
                    let sessions = self.ss.sessions.try_read().ok()?;
                    sessions
                        .get(session_id)
                        .map(|s| s.metadata.provider.clone())
                })
                .unwrap_or_else(|| self.config().default_provider.clone());
            if let Some(pc) = self.config().providers.get(&target_provider) {
                let valid = pc.models.iter().any(|x| x == m)
                    || pc.model_aliases.contains_key(m)
                    || pc.model_aliases.values().any(|v| v == m);
                if !valid {
                    let available: Vec<_> = pc
                        .models
                        .iter()
                        .chain(pc.model_aliases.keys())
                        .take(8)
                        .cloned()
                        .collect();
                    return Err(AgentError::Config(format!(
                        "model '{m}' not found on provider '{target_provider}'. Available: {}",
                        available.join(", ")
                    )));
                }
            }
        }

        let session_root = self.ss.store.session_root(session_id);
        let config_path = session_root.join("config.json");

        let mut sc = if config_path.exists() {
            SessionConfig::from_file(&config_path).unwrap_or_default()
        } else {
            SessionConfig::default()
        };

        if let Some(p) = provider {
            sc.default_provider = Some(p.to_string());
        }
        if let Some(m) = model {
            sc.default_model = Some(m.to_string());
        }

        tokio::fs::create_dir_all(&session_root)
            .await
            .map_err(|e| AgentError::Session(format!("cannot create session dir: {e}")))?;
        let json = serde_json::to_string_pretty(&sc)
            .map_err(|e| AgentError::Config(format!("serialize: {e}")))?;
        tokio::fs::write(&config_path, json)
            .await
            .map_err(|e| AgentError::Session(format!("cannot write config.json: {e}")))?;

        // Update in-memory metadata immediately
        let effective = self.config().merge_session(&sc);
        if let Some(session) = self.ss.sessions.write().await.get_mut(session_id) {
            session.metadata.provider = effective.provider;
            session.metadata.model = effective.model;
        }

        // Invalidate cached provider so next turn rebuilds it (cache is keyed by provider name)
        if let Some(p) = provider {
            self.provider_svc.invalidate(p).await;
        }

        Ok(())
    }

    /// Set reasoning level for a session. Writes to config.json.
    pub async fn set_session_reasoning(&self, session_id: &str, reasoning: &str) -> Result<()> {
        let val = match reasoning {
            "off" | "low" | "medium" | "high" => reasoning.to_string(),
            _ => {
                return Err(AgentError::Config(format!(
                    "invalid reasoning level: {reasoning}"
                )));
            }
        };

        let session_root = self.ss.store.session_root(session_id);
        let config_path = session_root.join("config.json");

        let mut sc = if config_path.exists() {
            SessionConfig::from_file(&config_path).unwrap_or_default()
        } else {
            SessionConfig::default()
        };

        sc.reasoning = if val == "off" { None } else { Some(val) };

        tokio::fs::create_dir_all(&session_root)
            .await
            .map_err(|e| AgentError::Session(format!("cannot create session dir: {e}")))?;
        let json = serde_json::to_string_pretty(&sc)
            .map_err(|e| AgentError::Config(format!("serialize: {e}")))?;
        tokio::fs::write(&config_path, json)
            .await
            .map_err(|e| AgentError::Session(format!("cannot write config.json: {e}")))?;

        Ok(())
    }

    /// Set yolo timestamp for a session. Writes to config.json.
    /// Pass `Some(ts)` to enable with a specific unix timestamp, `None` to disable.
    pub async fn set_session_yolo(&self, session_id: &str, enabled_at: Option<i64>) -> Result<()> {
        let session_root = self.ss.store.session_root(session_id);
        let config_path = session_root.join("config.json");

        let mut sc = if config_path.exists() {
            SessionConfig::from_file(&config_path).unwrap_or_default()
        } else {
            SessionConfig::default()
        };

        sc.yolo_enabled_at = enabled_at;

        tokio::fs::create_dir_all(&session_root)
            .await
            .map_err(|e| AgentError::Session(format!("cannot create session dir: {e}")))?;
        let json = serde_json::to_string_pretty(&sc)
            .map_err(|e| AgentError::Config(format!("serialize: {e}")))?;
        tokio::fs::write(&config_path, json)
            .await
            .map_err(|e| AgentError::Session(format!("cannot write config.json: {e}")))?;

        Ok(())
    }

    /// Set allow-list for a session. Writes to config.json.
    pub async fn set_session_allow_list(&self, session_id: &str, tools: &[String]) -> Result<()> {
        let session_root = self.ss.store.session_root(session_id);
        let config_path = session_root.join("config.json");

        let mut sc = if config_path.exists() {
            SessionConfig::from_file(&config_path).unwrap_or_default()
        } else {
            SessionConfig::default()
        };

        sc.allow_list = if tools.is_empty() {
            None
        } else {
            Some(tools.to_vec())
        };

        tokio::fs::create_dir_all(&session_root)
            .await
            .map_err(|e| AgentError::Session(format!("cannot create session dir: {e}")))?;
        let json = serde_json::to_string_pretty(&sc)
            .map_err(|e| AgentError::Config(format!("serialize: {e}")))?;
        tokio::fs::write(&config_path, json)
            .await
            .map_err(|e| AgentError::Session(format!("cannot write config.json: {e}")))?;

        Ok(())
    }

    /// Get the current reasoning level for a session.
    pub async fn session_reasoning(&self, session_id: &str) -> Option<String> {
        let sc = self.load_session_config_pub(session_id);
        sc.reasoning
    }

    /// Persist a channel-specific key so the channel→session mapping survives restarts.
    /// For Telegram: `"tg:{chat_id}:{thread_id}"`.
    pub async fn set_session_channel_id(&self, session_id: &str, channel_id: &str) {
        self.ss.set_session_channel_id(session_id, channel_id).await
    }

    /// Return `(channel_id, session_id)` pairs for sessions that have a channel_id.
    /// When multiple sessions share the same channel_id, only the most recently
    /// updated one is returned.
    pub async fn channel_session_mappings(&self) -> Vec<(String, String)> {
        self.ss.channel_session_mappings().await
    }

    /// Get the currently active provider and model for a session.
    /// Sum of token usage across all assistant turns in a session.
    pub async fn session_total_usage(&self, session_id: &str) -> types::TurnUsage {
        self.ss.session_total_usage(session_id).await
    }

    /// B3: Get file tracking stats for a session (read-only, modified).
    pub async fn session_file_stats(&self, session_id: &str) -> (Vec<String>, Vec<String>) {
        self.ss.session_file_stats(session_id).await
    }

    /// Returns (estimated_tokens, context_window_tokens) for a session.
    pub async fn session_context_usage(&self, session_id: &str) -> Option<(usize, u32)> {
        let sessions = self.ss.sessions.read().await;
        sessions.get(session_id).map(|s| {
            (
                s.history.estimated_tokens(),
                s.history.context_window_tokens(),
            )
        })
    }

    pub async fn session_provider_model(&self, session_id: &str) -> (String, String) {
        let sc = self.load_session_config_pub(session_id);
        let effective = self.config().merge_session(&sc);
        (effective.provider, effective.model)
    }
}
#[async_trait::async_trait]
impl crate::services::ToolBuilder for AgentCore {
    async fn build_registry(
        &self,
        session_id: &str,
        effective: &crate::EffectiveSessionConfig,
        provider: &std::sync::Arc<dyn crate::provider::Provider>,
        model: &str,
        workspace: &std::path::Path,
    ) -> crate::tool::registry::ToolRegistry {
        self.build_tool_registry_for(session_id, effective, provider, model, workspace)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compaction_prompt_fresh_has_format_instructions() {
        let prompt = AgentCore::compaction_prompt(None);
        assert!(prompt.contains("## Goal"), "missing Goal section");
        assert!(prompt.contains("## Progress"), "missing Progress section");
        assert!(
            prompt.contains("Summarize"),
            "missing Summarize instruction"
        );
        assert!(!prompt.contains("<previous-summary>"));
    }

    #[test]
    fn compaction_prompt_with_previous_includes_it() {
        let prompt = AgentCore::compaction_prompt(Some("old summary here"));
        assert!(prompt.contains("<previous-summary>"));
        assert!(prompt.contains("old summary here"));
        assert!(prompt.contains("Update the existing summary"));
        assert!(prompt.contains("## Goal"));
    }

    #[test]
    fn compaction_format_has_all_sections() {
        let fmt = AgentCore::COMPACTION_FORMAT;
        for section in [
            "## Goal",
            "## Progress",
            "### Done",
            "### In Progress",
            "## Next Steps",
        ] {
            assert!(fmt.contains(section), "missing {section}");
        }
    }
}
