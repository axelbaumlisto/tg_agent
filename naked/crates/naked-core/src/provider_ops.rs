//! Provider/model listing, session provider/reasoning/yolo config.

use super::*;

impl AgentCore {
    pub fn list_models(&self) -> Vec<types::ModelInfo> {
        self.provider.models()
    }

    /// List models for a specific provider as `(provider_name, model_id)` pairs.
    pub async fn provider_models(&self, provider_name: &str) -> Vec<(String, String)> {
        if let Some(pc) = self.config.providers.get(provider_name) {
            pc.models
                .iter()
                .map(|m| (provider_name.to_string(), m.clone()))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// List all configured providers with their available models.
    pub fn list_providers(&self) -> Vec<ProviderInfo> {
        let mut result = Vec::new();
        for (name, pc) in &self.config.providers {
            let active = name == &self.config.default_provider;
            result.push(ProviderInfo {
                name: name.clone(),
                models: pc.models.clone(),
                active,
            });
        }
        result.sort_by(|a, b| a.name.cmp(&b.name));
        result
    }

    /// Switch the provider and model for a specific session.
    /// Writes config.json into the session directory.
    pub async fn set_session_provider(
        &self,
        session_id: &str,
        provider: Option<&str>,
        model: Option<&str>,
    ) -> Result<()> {
        if let Some(p) = provider
            && !self.config.providers.contains_key(p)
        {
            return Err(AgentError::ProviderNotConfigured(p.to_string()));
        }

        // Validate that the model belongs to the target provider.
        // Resolve the effective provider (explicit or current session's).
        if let Some(m) = model {
            let target_provider = provider
                .map(|s| s.to_string())
                .or_else(|| {
                    let sessions = self.sessions.try_read().ok()?;
                    sessions
                        .get(session_id)
                        .map(|s| s.metadata.provider.clone())
                })
                .unwrap_or_else(|| self.config.default_provider.clone());
            if let Some(pc) = self.config.providers.get(&target_provider) {
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

        let session_root = self.store.session_root(session_id);
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
        let effective = self.config.merge_session(&sc);
        if let Some(session) = self.sessions.write().await.get_mut(session_id) {
            session.metadata.provider = effective.provider;
            session.metadata.model = effective.model;
        }

        // Invalidate cached provider so next turn rebuilds it (cache is keyed by provider name)
        if let Some(p) = provider {
            self.provider_cache.write().await.remove(p);
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

        let session_root = self.store.session_root(session_id);
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
        let session_root = self.store.session_root(session_id);
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
        let session_root = self.store.session_root(session_id);
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
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.get_mut(session_id) {
            session.metadata.channel_id = Some(channel_id.to_string());
            if let Err(e) = self.store.save(session).await {
                tracing::error!("failed to persist channel_id for {session_id}: {e}");
            }
        }
    }

    /// Return `(channel_id, session_id)` pairs for sessions that have a channel_id.
    /// When multiple sessions share the same channel_id, only the most recently
    /// updated one is returned.
    pub async fn channel_session_mappings(&self) -> Vec<(String, String)> {
        let sessions = self.sessions.read().await;
        let mut best: std::collections::HashMap<String, (&str, chrono::DateTime<chrono::Utc>)> =
            std::collections::HashMap::new();
        for s in sessions.values() {
            if let Some(cid) = &s.metadata.channel_id {
                let entry = best.entry(cid.clone()).or_insert((&s.id, s.updated_at));
                if s.updated_at > entry.1 {
                    *entry = (&s.id, s.updated_at);
                }
            }
        }
        best.into_iter()
            .map(|(cid, (sid, _))| (cid, sid.to_string()))
            .collect()
    }

    /// Get the currently active provider and model for a session.
    /// Sum of token usage across all assistant turns in a session.
    pub async fn session_total_usage(&self, session_id: &str) -> types::TurnUsage {
        let sessions = self.sessions.read().await;
        let mut total = types::TurnUsage::default();
        if let Some(session) = sessions.get(session_id) {
            for msg in session.history.messages() {
                if let Some(u) = &msg.usage {
                    total.input_tokens += u.input_tokens;
                    total.output_tokens += u.output_tokens;
                    total.cache_read_tokens += u.cache_read_tokens;
                    total.cache_write_tokens += u.cache_write_tokens;
                }
            }
        }
        total
    }

    /// Returns (estimated_tokens, context_window_tokens) for a session.
    pub async fn session_context_usage(&self, session_id: &str) -> Option<(usize, u32)> {
        let sessions = self.sessions.read().await;
        sessions.get(session_id).map(|s| {
            (
                s.history.estimated_tokens(),
                s.history.context_window_tokens(),
            )
        })
    }

    pub async fn session_provider_model(&self, session_id: &str) -> (String, String) {
        let sc = self.load_session_config_pub(session_id);
        let effective = self.config.merge_session(&sc);
        (effective.provider, effective.model)
    }

    /// Resolve the **default** (provider, model) tuple — what a brand-new
    /// session would inherit before any per-session overrides are applied.
    ///
    /// Use this when you need to make a routing decision (vision-capability,
    /// reasoning policy, …) **before** a session ID exists. Replaces the
    /// previous `session_provider_model("__nonexistent__")` hack which leaned
    /// on the fact that `load_session_config_pub` of a missing dir returns
    /// `SessionConfig::default()`. That worked, but baking a magic
    /// session-id sentinel into the contract was a code smell — this method
    /// makes the intent explicit and never touches the filesystem.
    pub fn default_provider_model(&self) -> (String, String) {
        let effective = self.config.merge_session(&SessionConfig::default());
        (effective.provider, effective.model)
    }
}
