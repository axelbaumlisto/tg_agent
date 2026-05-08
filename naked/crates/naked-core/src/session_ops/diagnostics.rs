//! Session diagnostics — store, provider_for, list, inspect.

#[allow(unused_imports)]
use crate::AgentCore;
#[allow(unused_imports)]
use crate::config::{EffectiveSessionConfig, SessionConfig};
#[allow(unused_imports)]
use crate::error::{AgentError, Result};
#[allow(unused_imports)]
use crate::provider::{self, Provider};
#[allow(unused_imports)]
use crate::session::store::SessionStore;
#[allow(unused_imports)]
use crate::session::{Session, SessionMetadata, SessionState, SessionSummary};
#[allow(unused_imports)]
use crate::skill;
#[allow(unused_imports)]
use crate::skill::resolver::SkillResolver;
#[allow(unused_imports)]
use crate::types::{self, AgentEvent, AgentHandle, ContentBlock, PermissionResponse};
#[allow(unused_imports)]
use std::path::Path;
#[allow(unused_imports)]
use std::sync::Arc;

impl AgentCore {
    /// Borrow the underlying session store. Exposed for operator commands
    /// (`vacuum-sessions`, future `gc` task) that need to walk all sessions
    /// without going through the in-memory cache.
    pub fn store(&self) -> Arc<dyn SessionStore> {
        self.ss.store.clone()
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

    pub async fn is_session_active(&self, session_id: &str) -> bool {
        self.ss.is_session_active(session_id).await
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

    pub fn list_skills(&self) -> Vec<(String, String)> {
        let resolver = SkillResolver::new(self.config().skill_roots.clone());
        resolver
            .list()
            .into_iter()
            .map(|(name, hit)| (name, hit.path.display().to_string()))
            .collect()
    }

    pub async fn list_mcp_servers(&self) -> Vec<(String, usize)> {
        let reg = self.catalog.mcp_registry.read().await;
        reg.servers()
            .iter()
            .map(|s| (s.name.clone(), s.tools().len()))
            .collect()
    }
}
