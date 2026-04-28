//! In-process daily-memory scheduler. Wakes once per minute, asks
//! `next_cron_after` whether the configured `memory.daily_cron`
//! expression has fired since the previous tick, and (if so) calls
//! `memory::daily::run_daily` for the project scope plus every user
//! scope that has a memory directory on disk.
//!
//! The scheduler is intentionally simple — we don't track per-spec
//! state like the research scheduler. Instead `memory::daily` itself
//! owns idempotence via `should_run_today` (a `.last_digest` lock
//! file per scope). That keeps the scheduler stateless and trivially
//! restart-safe: even if we tick "late" after a long sleep, we just
//! call `run_daily` which is a no-op when today's digest already ran.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use naked_core::AgentCore;
use naked_core::config::MemoryConfig;
use naked_core::memory;
use naked_core::memory::types::MemoryScope;
use tokio::sync::Notify;
use tokio::task::JoinHandle;

use crate::cron_util::next_cron_after;

/// Public handle returned by [`spawn`]. Keep it alive for as long as
/// the scheduler should run; dropping the handle drops the background
/// task (tokio aborts the spawned future).
pub struct MemoryScheduler {
    _handle: JoinHandle<()>,
    notify: Arc<Notify>,
}

impl MemoryScheduler {
    /// Wake the scheduler immediately (e.g. after a config reload). The
    /// next tick will re-evaluate the cron expression.
    pub fn poke(&self) {
        self.notify.notify_one();
    }
}

/// Spawn the daily-memory scheduler. Returns immediately. The
/// scheduler is a no-op when `cfg.daily_enabled = false` (it stays
/// alive but never fires).
pub fn spawn(agent: Arc<AgentCore>, workspace: PathBuf, cfg: MemoryConfig) -> MemoryScheduler {
    let notify = Arc::new(Notify::new());
    let notify_inner = notify.clone();
    let handle = tokio::spawn(async move {
        run_loop(agent, workspace, cfg, notify_inner).await;
    });
    MemoryScheduler {
        _handle: handle,
        notify,
    }
}

/// Tick interval. We poll once a minute — tighter than that wastes CPU
/// for a job that runs at most once per day.
const TICK: Duration = Duration::from_secs(60);

async fn run_loop(
    agent: Arc<AgentCore>,
    workspace: PathBuf,
    cfg: MemoryConfig,
    notify: Arc<Notify>,
) {
    let mut last_check: DateTime<Utc> = Utc::now();
    tracing::info!(
        cron = %cfg.daily_cron,
        enabled = cfg.daily_enabled,
        mode = %cfg.daily_mode,
        "memory scheduler started"
    );

    loop {
        tokio::select! {
            _ = tokio::time::sleep(TICK) => {}
            _ = notify.notified() => {}
        }

        if !cfg.daily_enabled {
            continue;
        }

        let now = Utc::now();
        let due = match next_cron_after(&cfg.daily_cron, last_check) {
            Some(t) => t <= now,
            None => {
                tracing::warn!(
                    cron = %cfg.daily_cron,
                    "memory scheduler: invalid cron expression, sleeping"
                );
                last_check = now;
                continue;
            }
        };
        last_check = now;
        if !due {
            continue;
        }

        tracing::info!("memory scheduler: cron fired, running daily digest");

        // Resolve the LLM provider/model the digest will use. We piggy-
        // back on the agent's default provider unless the user
        // configured `memory.digest_provider`.
        let (provider_name, model) = resolve_digest_target(&agent, &cfg).await;
        let provider = agent.provider_for(&provider_name).await;

        // Run digest for project + global scopes (the always-present
        // ones), then iterate every on-disk user scope.
        let scopes = scopes_for_digest();

        for scope in scopes {
            match memory::daily::run_daily(
                &workspace,
                &scope,
                &cfg,
                Some((&*provider, model.as_str())),
            )
            .await
            {
                Ok(Some((_plan, outcome))) => tracing::info!(
                    scope = %scope,
                    promoted = outcome.promoted_written,
                    dreams_appended = outcome.dreams_appended,
                    "memory daily digest completed"
                ),
                Ok(None) => tracing::debug!(scope = %scope, "memory daily digest skipped"),
                Err(e) => tracing::warn!(scope = %scope, "memory daily digest failed: {e:#}"),
            }
        }
    }
}

/// Build the list of scopes the daily-digest scheduler iterates each
/// fire. `Project` and `Global` are always included; per-user scopes
/// are auto-discovered from `~/.naked/users/`.
///
/// **Why Global is in here**: prior to this it was silently skipped,
/// so `~/.naked/memory/MEMORY.md` accumulated entries forever — never
/// got `compact_memory` folding, never appeared in `DREAMS.md`. The
/// regression test below pins the contract.
fn scopes_for_digest() -> Vec<MemoryScope> {
    let mut scopes = vec![MemoryScope::Project, MemoryScope::Global];
    scopes.extend(memory::store::list_user_scopes());
    scopes
}

/// Pick the (provider, model) tuple the digest LLM call should use.
/// Falls back to the agent's defaults when `digest_provider` is unset
/// or malformed.
async fn resolve_digest_target(agent: &AgentCore, cfg: &MemoryConfig) -> (String, String) {
    let default_provider = agent.config().default_provider.clone();
    let default_model = agent.config().default_model.clone();
    let Some(spec) = cfg.digest_provider.as_deref() else {
        return (default_provider, default_model);
    };
    if let Some((p, m)) = spec.split_once('/') {
        (p.to_string(), m.to_string())
    } else {
        (default_provider, spec.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression for the silently-skipped Global scope. Before the fix
    /// the scheduler iterated only `Project + per-user`, so
    /// `~/.naked/memory/MEMORY.md` never got compaction or DREAMS
    /// entries and grew unbounded. Both of the always-present scopes
    /// must be in the list.
    #[test]
    fn scopes_for_digest_includes_project_and_global() {
        let scopes = scopes_for_digest();
        assert!(
            scopes.contains(&MemoryScope::Project),
            "project scope must be in the digest sweep, got {scopes:?}",
        );
        assert!(
            scopes.contains(&MemoryScope::Global),
            "global scope must be in the digest sweep, got {scopes:?}",
        );
    }
}
