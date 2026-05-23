//! Session lifecycle — create, send, queue, fork, restore, close, MCP init.

use crate::config::SessionConfig;
use crate::error::{AgentError, Result};
use crate::mcp::client::McpRegistry;
use crate::memory;
use crate::prompt;
use crate::research;
use crate::session::{Session, SessionMetadata};
use crate::skill;
use crate::skill::resolver::SkillResolver;
use crate::types::{AgentHandle, ContentBlock};
use crate::{AgentCore, UserPush};
use std::path::Path;

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
            let old = self.catalog.mcp_registry.read().await;
            old.close_all().await;
        }
        let result = McpRegistry::connect_all_with_diagnostics(&servers).await;
        tracing::info!(
            "MCP: {} tools from {} servers",
            result.registry.all_tools().len(),
            result.registry.servers().len()
        );
        *self.catalog.mcp_registry.write().await = result.registry;
        result.failures
    }

    /// Load per-session config.json if it exists.
    pub fn load_session_config_pub(&self, session_id: &str) -> SessionConfig {
        let path = self.ss.store.session_root(session_id).join("config.json");
        if path.exists() {
            match SessionConfig::from_file(&path) {
                Ok(sc) => {
                    tracing::info!("loaded per-session config for {}", &session_id[..8]); // REGISTRY-WAIVE: B48 — session ID is ASCII hex
                    sc
                }
                Err(e) => {
                    tracing::warn!("bad session config.json for {}: {e}", &session_id[..8]); // REGISTRY-WAIVE: B48 — session ID is ASCII hex
                    SessionConfig::default()
                }
            }
        } else {
            SessionConfig::default()
        }
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
        let mcp_reg = self.catalog.mcp_registry.read().await;
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
            self.catalog.mcp_registry.read().await.servers().len()
        );
    }
}
