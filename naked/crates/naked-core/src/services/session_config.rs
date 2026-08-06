use std::path::PathBuf;
use std::sync::Arc;

use tokio::fs;

use crate::config::{Config, SessionConfig, YOLO_TTL_SECS};
use crate::error::{AgentError, Result};
use crate::services::provider_svc::ProviderService;
use crate::services::session_state::SessionState;
use crate::types::TurnUsage;

/// Service for session-scoped configuration (provider/model overrides,
/// reasoning level, allow list, channel mapping, usage queries).
pub struct SessionConfigService {
    session_state: Arc<SessionState>,
    provider_service: Arc<ProviderService>,
}

impl SessionConfigService {
    pub fn new(session_state: Arc<SessionState>, provider_service: Arc<ProviderService>) -> Self {
        Self {
            session_state,
            provider_service,
        }
    }

    fn session_root(&self, session_id: &str) -> PathBuf {
        self.session_state.store.session_root(session_id)
    }

    fn load_config(&self, session_id: &str) -> SessionConfig {
        let path = self.session_root(session_id).join("config.json");
        if path.exists() {
            match SessionConfig::from_file(&path) {
                Ok(sc) => sc,
                Err(e) => {
                    tracing::warn!(
                        session = session_id,
                        error = %e,
                        "bad session config.json",
                    );
                    SessionConfig::default()
                }
            }
        } else {
            SessionConfig::default()
        }
    }

    async fn persist_config(&self, session_id: &str, config: &SessionConfig) -> Result<()> {
        let session_root = self.session_root(session_id);
        fs::create_dir_all(&session_root)
            .await
            .map_err(|e| AgentError::Session(format!("cannot create session dir: {e}")))?;
        let json = serde_json::to_string_pretty(config)
            .map_err(|e| AgentError::Config(format!("serialize: {e}")))?;
        fs::write(session_root.join("config.json"), json)
            .await
            .map_err(|e| AgentError::Session(format!("cannot write config.json: {e}")))?;
        Ok(())
    }

    pub async fn set_session_provider(
        &self,
        config: Arc<Config>,
        session_id: &str,
        provider: Option<&str>,
        model: Option<&str>,
    ) -> Result<()> {
        // B105: accept a retired provider name if `provider_aliases` maps it
        // to a live one. Without this, an alias is settable by an inherited
        // session pin but NOT by an explicit `/model` pick, which would be an
        // inconsistent surface. Validation below runs against the RESOLVED
        // provider so the model check uses the catalog that will serve it.
        if let Some(p) = provider {
            let (resolved, _) = config.resolve_provider_alias(p);
            if !config.providers.contains_key(&resolved) {
                return Err(AgentError::ProviderNotConfigured(p.to_string()));
            }
        }

        if let Some(m) = model {
            let target_provider = provider
                .map(|s| s.to_string())
                .or_else(|| {
                    let sessions = self.session_state.sessions.try_read().ok()?;
                    sessions
                        .get(session_id)
                        .map(|s| s.metadata.provider.clone())
                })
                .unwrap_or_else(|| config.default_provider.clone());
            // Resolve aliases before the per-provider model check.
            let (target_provider, _) = config.resolve_provider_alias(&target_provider);

            if let Some(pc) = config.providers.get(&target_provider)
                && !pc.serves_model(m)
            {
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

        let mut sc = self.load_config(session_id);
        if let Some(p) = provider {
            sc.default_provider = Some(p.to_string());
        }
        if let Some(m) = model {
            sc.default_model = Some(m.to_string());
        }
        self.persist_config(session_id, &sc).await?;

        let effective = config.merge_session(&sc);
        if let Some(session) = self
            .session_state
            .sessions
            .write()
            .await
            .get_mut(session_id)
        {
            session.metadata.provider = effective.provider;
            session.metadata.model = effective.model;
        }

        if let Some(p) = provider {
            self.provider_service.invalidate(p).await;
        }

        Ok(())
    }

    pub async fn set_session_reasoning(&self, session_id: &str, reasoning: &str) -> Result<()> {
        let val = match reasoning {
            "off" | "low" | "medium" | "high" => reasoning.to_string(),
            _ => {
                return Err(AgentError::Config(format!(
                    "invalid reasoning level: {reasoning}"
                )));
            }
        };

        let mut sc = self.load_config(session_id);
        sc.reasoning = if val == "off" { None } else { Some(val) };
        self.persist_config(session_id, &sc).await
    }

    pub async fn set_session_yolo(&self, session_id: &str, enabled_at: Option<i64>) -> Result<()> {
        let mut sc = self.load_config(session_id);
        sc.yolo_enabled_at = enabled_at;
        // Stamp the current TTL window on enable so a restart judges this grant
        // against the window it was created under; clear it on disable.
        sc.yolo_ttl_secs = enabled_at.map(|_| YOLO_TTL_SECS);
        self.persist_config(session_id, &sc).await
    }

    pub async fn set_session_allow_list(&self, session_id: &str, tools: &[String]) -> Result<()> {
        let mut sc = self.load_config(session_id);
        sc.allow_list = if tools.is_empty() {
            None
        } else {
            Some(tools.to_vec())
        };
        self.persist_config(session_id, &sc).await
    }

    pub async fn session_reasoning(&self, session_id: &str) -> Option<String> {
        self.load_config(session_id).reasoning
    }

    pub async fn set_session_channel_id(&self, session_id: &str, channel_id: &str) {
        self.session_state
            .set_session_channel_id(session_id, channel_id)
            .await
    }

    pub async fn channel_session_mappings(&self) -> Vec<(String, String)> {
        self.session_state.channel_session_mappings().await
    }

    pub async fn session_total_usage(&self, session_id: &str) -> TurnUsage {
        self.session_state.session_total_usage(session_id).await
    }

    pub async fn session_file_stats(&self, session_id: &str) -> (Vec<String>, Vec<String>) {
        self.session_state.session_file_stats(session_id).await
    }

    pub async fn session_context_usage(&self, session_id: &str) -> Option<(usize, u32)> {
        self.session_state.session_context_usage(session_id).await
    }

    pub async fn session_provider_model(
        &self,
        config: Arc<Config>,
        session_id: &str,
    ) -> (String, String) {
        let sc = self.load_config(session_id);
        let effective = config.merge_session(&sc);
        (effective.provider, effective.model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::SessionDiagnostics;
    use crate::test_support::TestCore;

    fn build_service(tc: &TestCore) -> SessionConfigService {
        SessionConfigService::new(tc.core.ss.clone(), tc.core.provider_svc.clone())
    }

    #[tokio::test]
    async fn set_session_provider_updates_metadata() {
        let tc = TestCore::build();
        let svc = build_service(&tc);
        let ws = tc.workspace();
        let sid = tc.core.create_session(&ws).await;

        let cfg = Arc::clone(&tc.core.config());
        svc.set_session_provider(cfg, &sid, None, Some("noop-model"))
            .await
            .unwrap();

        let sessions = tc.core.ss.sessions.read().await;
        let session = sessions.get(&sid).unwrap();
        assert_eq!(session.metadata.provider, "noop");
        assert_eq!(session.metadata.model, "noop-model");
    }

    #[tokio::test]
    async fn set_session_reasoning_toggles() {
        let tc = TestCore::build();
        let svc = build_service(&tc);
        let sid = tc.core.create_session(&tc.workspace()).await;

        svc.set_session_reasoning(&sid, "medium").await.unwrap();
        assert_eq!(svc.session_reasoning(&sid).await, Some("medium".into()));

        svc.set_session_reasoning(&sid, "off").await.unwrap();
        assert_eq!(svc.session_reasoning(&sid).await, None);
    }

    #[tokio::test]
    async fn set_session_yolo_stamps_and_clears_ttl() {
        let tc = TestCore::build();
        let svc = build_service(&tc);
        let sid = tc.core.create_session(&tc.workspace()).await;

        // Enabling stamps both the timestamp and the 30d TTL window so a
        // restart judges the grant against the window it was created under.
        let ts = 1_700_000_000;
        svc.set_session_yolo(&sid, Some(ts)).await.unwrap();
        let cfg = svc.load_config(&sid);
        assert_eq!(cfg.yolo_enabled_at, Some(ts));
        assert_eq!(cfg.yolo_ttl_secs, Some(YOLO_TTL_SECS));

        // Disabling clears both fields.
        svc.set_session_yolo(&sid, None).await.unwrap();
        let cfg = svc.load_config(&sid);
        assert_eq!(cfg.yolo_enabled_at, None);
        assert_eq!(cfg.yolo_ttl_secs, None);
    }

    // ── D-INV-PROVIDER-ALIAS (B105), `/model` surface ─────────────────────
    //
    // An alias must be settable EXPLICITLY too, not only inherited from an
    // old session pin — otherwise `/model` rejects the very name the bot
    // itself still shows. Dropping the alias resolution in
    // `set_session_provider` fails the first test.

    #[tokio::test]
    async fn b105_set_provider_accepts_aliased_retired_name() {
        let tc = TestCore::build();
        let svc = build_service(&tc);
        let sid = tc.core.create_session(&tc.workspace()).await;

        let mut cfg: crate::config::Config = (**tc.core.config()).clone();
        cfg.provider_aliases
            .insert("qwen".into(), "noop/noop-model".into());
        cfg.providers.insert(
            "noop".into(),
            crate::config::ProviderConfig {
                api_key: "k".into(),
                models: vec!["noop-model".into()],
                ..Default::default()
            },
        );

        svc.set_session_provider(Arc::new(cfg), &sid, Some("qwen"), None)
            .await
            .expect("an aliased retired provider must be accepted");
    }

    #[tokio::test]
    async fn b105_set_provider_still_rejects_unaliased_unknown() {
        let tc = TestCore::build();
        let svc = build_service(&tc);
        let sid = tc.core.create_session(&tc.workspace()).await;

        let cfg = Arc::clone(&tc.core.config());
        let err = svc
            .set_session_provider(cfg, &sid, Some("totally_unknown"), None)
            .await;
        assert!(
            err.is_err(),
            "a name with no alias and no catalog entry must still be rejected"
        );
    }

    #[tokio::test]
    async fn session_usage_delegates_to_state() {
        let tc = TestCore::build();
        let svc = build_service(&tc);
        let sid = tc.core.create_session(&tc.workspace()).await;

        let usage = svc.session_total_usage(&sid).await;
        assert_eq!(usage.input_tokens, 0);
        assert_eq!(usage.output_tokens, 0);

        let diagnostics: &dyn SessionDiagnostics = tc.core.as_ref();
        let via_trait = diagnostics.session_total_usage(&sid).await;
        assert_eq!(via_trait.output_tokens, usage.output_tokens);
    }
}
