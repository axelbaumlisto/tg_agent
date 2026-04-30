//! Session lifecycle — create, send, queue, abort, compact, fork, restore, list, skills, MCP.

use super::*;

impl AgentCore {
    /// Record the current author for an upcoming turn. Used by the memory tool
    /// to resolve `scope=user` without an explicit `user_id`. `None` clears it.
    pub async fn set_session_sender(&self, session_id: &str, sender_id: Option<String>) {
        let mut map = self.session_senders.write().await;
        match sender_id {
            Some(id) if !id.is_empty() => {
                map.insert(session_id.to_string(), id);
            }
            _ => {
                map.remove(session_id);
            }
        }
    }

    /// Look up the currently-recorded author for this session, if any.
    pub async fn session_sender(&self, session_id: &str) -> Option<String> {
        self.session_senders.read().await.get(session_id).cloned()
    }

    /// Connect to configured MCP servers.
    pub async fn init_mcp(&self) {
        let servers = self.config.mcp_server_list();
        if !servers.is_empty() {
            {
                let old = self.mcp_registry.read().await;
                old.close_all().await;
            }
            let registry = McpRegistry::connect_all(&servers).await;
            tracing::info!(
                "MCP: {} tools from {} servers",
                registry.all_tools().len(),
                registry.servers().len()
            );
            *self.mcp_registry.write().await = registry;
        }
    }

    /// Borrow the underlying session store. Exposed for operator commands
    /// (`vacuum-sessions`, future `gc` task) that need to walk all sessions
    /// without going through the in-memory cache.
    pub fn store(&self) -> Arc<dyn SessionStore> {
        self.store.clone()
    }

    /// Load per-session config.json if it exists.
    pub fn load_session_config_pub(&self, session_id: &str) -> SessionConfig {
        let path = self.store.session_root(session_id).join("config.json");
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
    /// Returns the global default if `name` matches `self.config.default_provider`.
    /// Resolve the named provider from the config catalog, building &
    /// caching it on first request. Empty / unknown names fall back to
    /// the default provider this `AgentCore` was constructed with.
    /// Public so out-of-loop callers (CLI gatekeeper, validators) can
    /// pin a non-default provider per call without rebuilding the
    /// whole agent.
    pub async fn provider_for(&self, provider_name: &str) -> Arc<dyn Provider> {
        if provider_name.is_empty() || provider_name == self.config.default_provider {
            return self.provider.clone();
        }

        if let Some(cached) = self.provider_cache.read().await.get(provider_name) {
            return cached.clone();
        }

        let built: Arc<dyn Provider> = if let Some(pc) = self.config.providers.get(provider_name)
            && let Ok(resolved) = pc.resolved()
        {
            Arc::from(create_provider(provider_name, resolved))
        } else {
            tracing::warn!(
                "session requests provider '{provider_name}' not in catalog, using default"
            );
            return self.provider.clone();
        };

        self.provider_cache
            .write()
            .await
            .insert(provider_name.to_string(), built.clone());
        built
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
            .filter(|name| !self.config.mcp_servers.contains_key(*name))
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
            prompt::resolve_system_prompt(workspace, self.config.system_prompt_path.as_deref());
        let capabilities = self.capabilities_section().await;
        let mut full_prompt = format!(
            "{}\n\n{}\n\n{}",
            system_prompt,
            prompt::environment_section(workspace),
            capabilities
        );
        if self.config.research.enabled {
            full_prompt.push_str("\n\n---\n");
            full_prompt.push_str(&research::briefing::short(&self.config.research));
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
            provider: self.config.default_provider.clone(),
            model: self.config.default_model.clone(),
            channel: channel.into(),
            channel_id: None,
        };

        let session = Session::new(workspace.to_path_buf(), full_prompt, metadata);
        let id = session.id.clone();

        if let Err(e) = self.store.save(&session).await {
            tracing::error!("failed to persist new session: {e}");
        }

        // Apply per-session config.json overrides to metadata
        let sc = self.load_session_config_pub(&id);
        let effective = self.config.merge_session(&sc);

        let mut sessions = self.sessions.write().await;
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

        // Load per-session config overlay (re-read each turn so edits take effect)
        let sc = self.load_session_config_pub(session_id);
        let effective = self.config.merge_session(&sc);

        let mut sessions = self.sessions.write().await;
        let session = sessions
            .get_mut(session_id)
            .ok_or_else(|| AgentError::SessionNotFound(session_id.to_string()))?;
        session.state = SessionState::Active;

        // Update metadata to match effective config (model/provider may change between turns)
        session.metadata.provider = effective.provider.clone();
        session.metadata.model = effective.model.clone();

        let model = session.metadata.model.clone();
        let provider_name = session.metadata.provider.clone();

        // Resolve context window: per-session > per-provider > model lookup > global > 128K
        let provider_ctx = self
            .config
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

        // Background memory classification (non-blocking, fire-and-forget).
        //
        // By default the hit lands in *today's draft file*, NOT durable
        // `MEMORY.md`. The daily digest later promotes it iff the same
        // rule re-appears across `promote_min_repeat_days` days. This
        // turns the classifier into a low-precision, high-recall feeder
        // for the scoring gate instead of a one-shot writer.
        //
        // Toggle with `memory.auto_classify_to_drafts = false` to fall
        // back to direct writes (legacy behaviour).
        {
            let ws = session.workspace.clone();
            let mdl = model.clone();
            let msg = classifier_text.clone();
            // Use the session's provider — not the global default — so the
            // model name is valid for the API endpoint. (Bug: using
            // self.provider sent "kimi-for-coding" to qwen → 404.)
            let provider_ref = self.provider_for(&provider_name).await;
            let sender_id = self.session_sender(session_id).await;
            let to_drafts = self.config.memory.auto_classify_to_drafts;
            tokio::spawn(async move {
                let provider_arc: std::sync::Arc<dyn Provider> = provider_ref;
                let Some(result) =
                    memory::classifier::classify(&*provider_arc, &mdl, &msg, sender_id.as_deref())
                        .await
                else {
                    return;
                };

                if to_drafts {
                    let entry = memory::types::MemoryEntry::new(
                        result.memory_type,
                        result.content.clone(),
                        "auto_classify",
                        result.scope.clone(),
                    );
                    match memory::store::MarkdownMemoryStore::append_daily(&ws, &entry, true) {
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
                    match memory::service::MemoryService::store(
                        &ws,
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
        let (read_files, modified_files) = if needs_compact {
            session.history.files_in_compaction_range(4)
        } else {
            (vec![], vec![])
        };
        let before_msgs = session.history.message_count();
        let workspace_for_compaction = session.workspace.clone();

        // Drop sessions lock before LLM call to avoid blocking other requests
        drop(sessions);

        // LLM-based compaction with deterministic fallback
        let llm_summary = if let Some(text_for_llm) = compact_text {
            tracing::info!(before_msgs, "attempting LLM-based compaction");
            let provider_arc = self.provider_for(&provider_name).await;

            // Pre-compaction flush: ask the model (silent turn) to extract
            // any rules-of-thumb / corrections from the history we are
            // about to discard, and append them to the project's daily
            // draft file. Best-effort; never blocks compaction.
            if self.config.memory.daily_enabled && self.config.memory.pre_compaction_flush {
                memory::digest::pre_compaction_flush(
                    &*provider_arc,
                    &model,
                    &workspace_for_compaction,
                    &memory::types::MemoryScope::Project,
                    &text_for_llm,
                )
                .await;
            }

            match Self::llm_summarize(
                &*provider_arc,
                &model,
                &text_for_llm,
                previous_summary.as_deref(),
            )
            .await
            {
                Ok(mut summary) => {
                    // Append file tracking
                    if !read_files.is_empty() || !modified_files.is_empty() {
                        summary.push_str("\n\n<read-files>\n");
                        for f in &read_files {
                            summary.push_str(f);
                            summary.push('\n');
                        }
                        summary.push_str("</read-files>\n<modified-files>\n");
                        for f in &modified_files {
                            summary.push_str(f);
                            summary.push('\n');
                        }
                        summary.push_str("</modified-files>");
                    }
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
        let mut sessions = self.sessions.write().await;
        let session = sessions
            .get_mut(session_id)
            .ok_or_else(|| AgentError::SessionNotFound(session_id.to_string()))?;

        let compacted = if needs_compact {
            if let Some(summary) = llm_summary {
                session.history.set_compaction_summary(summary.clone());
                session.history.compact_with_llm_summary(&summary, 4);
                session.history.set_last_input_tokens(None);
            } else {
                session.history.auto_compact();
            }
            let after = session.history.message_count();
            Some((before_msgs, after))
        } else {
            None
        };

        if let Some((before, after)) = compacted {
            tracing::info!("context compacted: {before} msgs -> {after} msgs");
            let _ = tx
                .send(AgentEvent::ContextCompacted {
                    before_msgs: before,
                    after_msgs: after,
                })
                .await;
            // After compaction, older turns (and any image blocks they
            // owned) are gone from history. Run a best-effort GC over the
            // session's artifacts dir to reclaim disk for images that no
            // JSONL line still references. Never block the user reply on
            // GC failures — log and move on.
            match self.store.gc_orphan_image_artifacts(session_id).await {
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

        let mut history = session.history.clone();
        let original_system_prompt = history.system_prompt().to_string();

        // Inject per-session prompt.md as context (if present)
        let session_root = self.store.session_root(session_id);
        let prompt_path = effective
            .system_prompt_path
            .as_ref()
            .map(|p| session_root.join(p))
            .unwrap_or_else(|| session_root.join("prompt.md"));
        if let Ok(extra) = tokio::fs::read_to_string(&prompt_path).await {
            let trimmed = extra.trim();
            if !trimmed.is_empty() {
                history.inject_system_context(&format!("\n\n[Session instructions]\n{trimmed}"));
            }
        }

        // Inject persistent memory rules into system prompt (plus the active
        // author's per-user rules when a Telegram sender is set for this turn).
        let sender_for_rules = self.session_sender(session_id).await;
        let memory_rules = memory::service::MemoryService::load_rules_for(
            &session.workspace,
            sender_for_rules.as_deref(),
        );
        if !memory_rules.is_empty() {
            history.inject_system_context(&format!("\n\n{memory_rules}"));
        }

        // "Recent shift": surface the last few days of un-promoted draft
        // memory entries so the model sees fresh context without waiting
        // for a daily-digest promotion. Cheap (just reads ≤2 small md
        // files per scope) and bumps the recall counter as a side
        // effect, which feeds promotion scoring.
        if self.config.memory.daily_enabled {
            let mut shift_blocks: Vec<String> = Vec::new();
            if let Some(b) = memory::daily::recent_shift_block(
                &session.workspace,
                &memory::types::MemoryScope::Project,
                &self.config.memory,
            ) {
                shift_blocks.push(b);
            }
            if let Some(sender) = sender_for_rules.as_deref()
                && let Some(b) = memory::daily::recent_shift_block(
                    &session.workspace,
                    &memory::types::MemoryScope::User(sender.to_string()),
                    &self.config.memory,
                )
            {
                shift_blocks.push(b);
            }
            if !shift_blocks.is_empty() {
                history.inject_system_context(&format!("\n\n{}", shift_blocks.join("\n\n")));
            }
        }

        let artifacts = self.store.artifacts_dir(session_id);
        if let Err(e) = tokio::fs::create_dir_all(&artifacts).await {
            tracing::warn!("could not create artifacts dir: {e}");
        }

        let cwd = if session.workspace.as_os_str().is_empty() || !session.workspace.exists() {
            artifacts
        } else {
            session.workspace.clone()
        };

        // Per-provider max_tokens / temperature (provider-level override > session > global)
        let eff_max_tokens = self
            .config
            .providers
            .get(&provider_name)
            .and_then(|pc| pc.max_tokens)
            .unwrap_or(effective.max_tokens);
        let eff_temperature = self
            .config
            .providers
            .get(&provider_name)
            .and_then(|pc| pc.temperature)
            .or(effective.temperature);

        let loop_config = LoopConfig {
            max_iterations: effective.max_iterations,
            cwd,
            model: model.clone(),
            max_tokens: eff_max_tokens,
            temperature: eff_temperature,
            reasoning: effective.reasoning.clone(),
            provider: provider_name.clone(),
            health: Some(self.model_health.clone()),
        };

        let session_workspace = session.workspace.clone();

        let cancel = CancellationToken::new();
        self.cancels
            .write()
            .await
            .insert(session_id.to_string(), cancel.clone());
        drop(sessions);

        // Validate model belongs to provider before making any API calls.
        if let Some(pc) = self.config.providers.get(&provider_name) {
            let valid = pc.models.iter().any(|x| x == &model)
                || pc.model_aliases.contains_key(&model)
                || pc.model_aliases.values().any(|v| v == &model);
            if !valid {
                let available: Vec<_> = pc
                    .models
                    .iter()
                    .chain(pc.model_aliases.keys())
                    .take(6)
                    .cloned()
                    .collect();
                let err_msg = format!(
                    "Model '{}' not found on provider '{}'. Try: {}",
                    model,
                    provider_name,
                    available.join(", ")
                );
                let _ = tx.send(AgentEvent::Error(err_msg)).await;
                let _ = tx.send(AgentEvent::Idle).await;
                if let Some(s) = self.sessions.write().await.get_mut(session_id) {
                    s.state = SessionState::Idle;
                }
                return Ok(AgentHandle {
                    events: rx,
                    permissions: perm_tx,
                });
            }
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
        let agent_loop = AgentLoop::new(provider_to_box(&session_provider), tools, loop_config);

        let session_id_owned = session_id.to_string();
        let sessions_ref = self.sessions.clone();
        let store_ref = self.store.clone();

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
                    .run(&mut history, tx.clone(), cancel, Some(perm_rx))
                    .await;
                match &result {
                    Ok(usage) => {
                        crate::types::TURN_COMPLETED_COUNT
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        tracing::info!(
                            "turn complete [{}]: {} tokens",
                            session_id_owned,
                            usage.total_tokens()
                        );
                        if usage.input_tokens > 0 {
                            history.set_last_input_tokens(usage.input_tokens);
                        }
                    }
                    Err(e) => {
                        crate::types::TURN_ERROR_COUNT
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        tracing::error!("turn error [{}]: {e}", session_id_owned);
                        let _ = tx.send(AgentEvent::Error(e.to_string())).await;
                    }
                }

                // Merge mutated history back into the session and persist
                history.restore_system_prompt(original_system_prompt);
                let mut sessions = sessions_ref.write().await;
                if let Some(session) = sessions.get_mut(&session_id_owned) {
                    session.history = history;
                    session.state = SessionState::Idle;
                    session.updated_at = chrono::Utc::now();
                    if let Err(e) = store_ref.save(session).await {
                        tracing::error!("failed to persist session [{}]: {e}", session_id_owned);
                    }
                }
            }
            .instrument(turn_span),
        );

        Ok(AgentHandle {
            events: rx,
            permissions: perm_tx,
        })
    }

    pub async fn is_session_active(&self, session_id: &str) -> bool {
        self.sessions
            .read()
            .await
            .get(session_id)
            .is_some_and(|s| s.state == SessionState::Active)
    }

    /// Append a user message to the session history without starting a new turn.
    pub async fn queue_message(&self, session_id: &str, text: &str) {
        if let Some(session) = self.sessions.write().await.get_mut(session_id) {
            session.history.push_user(text);
        }
    }

    /// Append a multimodal user message (text + images) to the history without
    /// starting a new turn. Used when the bot is busy and a new media-bearing
    /// message arrives mid-turn.
    pub async fn queue_message_multimodal(&self, session_id: &str, blocks: Vec<ContentBlock>) {
        if let Some(session) = self.sessions.write().await.get_mut(session_id) {
            session.history.push_user_multimodal(blocks);
        }
    }

    /// Trigger history compaction for a session. Returns (before, after) message counts.
    /// No-op if compaction not needed.
    pub async fn compact_session(&self, session_id: &str) -> Option<(usize, usize)> {
        if let Some(session) = self.sessions.write().await.get_mut(session_id) {
            session.history.auto_compact()
        } else {
            None
        }
    }

    pub async fn abort(&self, session_id: &str) {
        if let Some(cancel) = self.cancels.read().await.get(session_id) {
            cancel.cancel();
        }
        if let Some(session) = self.sessions.write().await.get_mut(session_id) {
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
        if !self.config.memory.daily_enabled || !self.config.memory.session_close_summary {
            return;
        }
        let (workspace, transcript, provider_name, model) = {
            let sessions = self.sessions.read().await;
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
        self.sessions
            .read()
            .await
            .get(session_id)
            .map(|s| s.workspace.clone())
    }

    /// Paged variant of `list_sessions`. Sorts by `updated_at` descending
    /// (most recently touched session first), then applies `skip` + `limit`.
    ///
    /// Callers can pass `limit = usize::MAX` to disable truncation. A
    /// `skip` beyond the total count returns an empty vec — never panics.
    /// Intended for CLI `/sessions --skip N --limit M` and future UI
    /// paging where listing 500 stale sessions would be useless.
    pub async fn list_sessions_paged(&self, skip: usize, limit: usize) -> Vec<SessionSummary> {
        let sessions = self.sessions.read().await;
        let mut summaries: Vec<SessionSummary> = sessions.values().map(|s| s.summary()).collect();
        // Newest first — operators almost always want the recent tail.
        summaries.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        summaries.into_iter().skip(skip).take(limit).collect()
    }

    pub async fn restore_sessions(&self) -> Result<Vec<String>> {
        let summaries = self.store.list().await?;
        let mut restored = Vec::new();
        for summary in summaries {
            match self.store.load(&summary.id).await {
                Ok(Some(mut session)) => {
                    // Apply per-session config.json overrides (provider/model)
                    let sc = self.load_session_config_pub(&session.id);
                    let effective = self.config.merge_session(&sc);
                    session.metadata.provider = effective.provider;
                    session.metadata.model = effective.model;

                    restored.push(session.id.clone());
                    self.sessions
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
        let sessions = self.sessions.read().await;
        let parent = sessions
            .get(session_id)
            .ok_or_else(|| AgentError::SessionNotFound(session_id.to_string()))?;
        let forked = parent.fork(branch_name);
        let new_id = forked.id.clone();
        self.store.save(&forked).await?;
        drop(sessions);
        self.sessions.write().await.insert(new_id.clone(), forked);
        Ok(new_id)
    }

    pub fn list_skills(&self) -> Vec<(String, String)> {
        let resolver = SkillResolver::new(self.config.skill_roots.clone());
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
            let resolver = SkillResolver::new(self.config.skill_roots.clone());
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
        let sub_agent = SubAgentTool::new(
            provider.clone(),
            model.to_string(),
            self.config.tool_timeout_secs,
            self.config.exa_api_keys.clone(),
        )
        .with_registry(self.agent_registry.clone());

        let mut tools: Vec<Box<dyn tool::Tool>> = vec![
            Box::new(BashTool::new(self.config.tool_timeout_secs)),
            Box::new(ReadFileTool),
            Box::new(WriteFileTool),
            Box::new(EditFileTool),
            Box::new(GlobSearchTool),
            Box::new(GrepSearchTool),
            Box::new(sub_agent),
            Box::new(AgentStatusTool::new(self.agent_registry.clone())),
            Box::new(AgentStopTool::new(self.agent_registry.clone())),
            Box::new(WebSearchTool::new(
                self.exa_key_pool.clone(),
                self.tavily_key_pool.clone(),
                self.serpapi_key_pool.clone(),
            )),
            Box::new(WebFetchTool::with_components(
                self.cloud_scraper.clone(),
                self.host_policy.clone(),
            )),
            Box::new(WebFetchTlsTool::new()),
            Box::new(WebFetchWaybackTool::new()),
            Box::new({
                let ctx = tool::memory::MemoryContext::new();
                ctx.set_user_id(self.session_sender(session_id).await);
                MemoryTool::with_context(workspace.to_path_buf(), ctx)
            }),
        ];

        // Research tools. Always registered so the `/research ask` flow can
        // call `research_status` from any session — but `research_save`,
        // `research_list`, and `research_save_cursor` early-return with an
        // error unless `research_context` is set (coordinator does this for
        // the turn and clears it after).
        if self.config.research.enabled {
            tools.push(Box::new(ResearchSaveTool::new(
                self.research_store.clone(),
                self.research_context.clone(),
                self.config.research.gatekeeper.clone(),
            )));
            tools.push(Box::new(ResearchListTool::new(
                self.research_store.clone(),
                self.research_context.clone(),
            )));
            tools.push(Box::new(ResearchSaveCursorTool::new(
                self.research_store.clone(),
                self.research_context.clone(),
            )));
            tools.push(Box::new(ResearchStatusTool::new(
                self.research_store.clone(),
            )));

            // High-level orchestration tools (usable from any chat turn)
            tools.push(Box::new(ResearchCreateTool::new(
                self.research_store.clone(),
                self.config.research.clone(),
            )));
            tools.push(Box::new(ResearchListSpecsTool::new(
                self.research_store.clone(),
                self.config.research.clone(),
            )));
            tools.push(Box::new(ResearchMetricsTool::new(
                self.research_store.clone(),
            )));
            tools.push(Box::new(ResearchHelpTool::new(
                self.config.research.clone(),
            )));
            tools.push(Box::new(ResearchFindingsTool::new(
                self.research_store.clone(),
            )));
            tools.push(Box::new(ResearchSetTargetTool::new(
                self.research_store.clone(),
                self.research_context.clone(),
            )));
            if let Some(weak) = self.self_ref.read().unwrap().clone() {
                tools.push(Box::new(ResearchLaunchTool::new(weak.clone())));
                tools.push(Box::new(ResearchUpdateSpecTool::new(weak.clone())));
                tools.push(Box::new(ResearchSetScheduleTool::new(weak.clone())));
                tools.push(Box::new(ResearchPauseTool::new(weak.clone())));
                tools.push(Box::new(ResearchResumeTool::new(weak)));
            }
        }

        let skill_roots = &effective.skill_roots;
        tracing::debug!("skill_roots: {:?}", skill_roots);
        let resolver = SkillResolver::new(skill_roots.clone());
        let available = resolver.list();
        tracing::info!(
            "Skills: {} found in {} roots",
            available.len(),
            skill_roots.len()
        );
        for (name, hit) in &available {
            tracing::debug!("  skill: {name} -> {}", hit.path.display());
        }
        let orphans = resolver.find_orphans();
        if !orphans.is_empty() {
            tracing::warn!(
                "Skills: {} directory(ies) in skill_roots have NO SKILL.{{json,md,toml}} manifest — \
                 invisible to the `Skill` tool. Add a manifest or remove the directory:",
                orphans.len()
            );
            for (root, path) in &orphans {
                tracing::warn!(
                    "  orphan skill dir: {} (root: {})",
                    path.display(),
                    root.display()
                );
            }
        }
        tools.push(Box::new(SkillTool::new(resolver, &available)));

        // Global MCP servers
        let mcp_reg = self.mcp_registry.read().await;
        for server in mcp_reg.servers() {
            tools.extend(McpToolWrapper::wrap_all(Arc::clone(server)));
        }

        // Per-session MCP servers (additive)
        let extra = self.session_mcp_servers(session_id, effective).await;
        for server in &extra {
            tools.extend(McpToolWrapper::wrap_all(Arc::clone(server)));
        }

        // Append extra tools injected by the embedding binary.
        for factory in self.extra_tool_factories.read().await.iter() {
            tools.push(factory());
        }

        ToolRegistry::new(tools)
    }

    // ── Research public API ────────────────────────────────────────────
    //
    // These methods are the thin layer the Telegram bot and CLI call into.
    // They hide the coordinator / store plumbing so callers don't need to
    // build it themselves; the trade-off is that AgentCore carries a research
    // store by construction, which is cheap (no network, one directory).
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
