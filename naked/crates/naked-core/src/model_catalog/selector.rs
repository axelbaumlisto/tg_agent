//! [`ModelSelector`] — the code-side source of truth for every
//! "can this model do this task?" decision across the agent.
//!
//! Phase 2 introduces this module. It consumes the capability catalog
//! seeded in Phase 0 and the per-pair health tracker added in Phase 3
//! (see [`super::health`]) and is consulted by:
//!
//! - [`crate::research::coordinator::ResearchCoordinator`] — filters
//!   `fallback_models` before dispatch so a deprecated or off-task
//!   model never burns a 90 s HTTP round-trip.
//! - [`crate::Config::validate_and_warn`] — already wired in Phase 1;
//!   emits structured warns even when enforcement is off.
//! - The chat / digest / classify surfaces (Phase 3+ as they get
//!   plumbed through individually).
//!
//! Contract: **enforcement is opt-in** via
//! [`crate::Config::enforce_model_capabilities`]. When the flag is
//! `false` (today's default), `is_available` returns `true` for every
//! pair regardless of caps — the caller still sees the advisory warns
//! but nothing is filtered out. This keeps the schema + seed data
//! soaking in production safely before we flip the switch in a future
//! release (tracked by the `flip-enforce` todo in
//! `plans/model-capabilities-catalog_69a8fb09.plan.md`).

use std::collections::HashMap;
use std::sync::Arc;

use crate::config::{Config, ProviderConfig};

use super::health::ModelHealth;
use super::types::{ModelCapabilities, ModelStatus, TaskKind};
use super::ModelRef;

/// Structured reason a pair failed validation. Stringified into
/// `tracing::warn!` fields by callers so operators can grep per class.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapError {
    /// Provider id not in `Config.providers`.
    UnknownProvider(String),
    /// Model id not declared under the provider's `models` or
    /// `model_aliases`.
    UnknownModel { provider: String, model: String },
    /// Status precludes automatic selection (Deprecated / Experimental).
    StatusBlocks {
        provider: String,
        model: String,
        status: ModelStatus,
    },
    /// `task` not in the model's `task_fit`.
    TaskMismatch {
        provider: String,
        model: String,
        task: TaskKind,
        task_fit: Vec<TaskKind>,
    },
    /// Pair is currently quarantined by the runtime health tracker
    /// (Phase 3). Usually a burst of empty-content or mid-stream errors
    /// within the 24h window.
    Quarantined {
        provider: String,
        model: String,
        until: chrono::DateTime<chrono::Utc>,
    },
}

impl std::fmt::Display for CapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CapError::UnknownProvider(p) => write!(f, "unknown provider {p:?}"),
            CapError::UnknownModel { provider, model } => {
                write!(f, "model {model:?} not declared under provider {provider:?}")
            }
            CapError::StatusBlocks { provider, model, status } => write!(
                f,
                "{provider}/{model} status={status:?} blocks automatic selection",
            ),
            CapError::TaskMismatch {
                provider,
                model,
                task,
                task_fit,
            } => write!(
                f,
                "{provider}/{model} not task-fit for {task}; declared {task_fit:?}"
            ),
            CapError::Quarantined { provider, model, until } => write!(
                f,
                "{provider}/{model} quarantined until {until}"
            ),
        }
    }
}

impl std::error::Error for CapError {}

/// Minimal budget knob consumed by [`ModelSelector::rank_for`]. Kept
/// deliberately tiny so the function can be called from every selection
/// site without pulling a big struct. Extend with fields as new
/// selection dimensions appear (cost cap, max latency, etc.).
#[derive(Debug, Clone, Default)]
pub struct Budget {
    /// When `true`, `status=Degraded` pairs are still returned (used for
    /// operator-pinned chat). When `false` (default), Degraded is
    /// filtered out of ranking but still works if pinned explicitly via
    /// [`ModelSelector::validate`] bypass.
    pub allow_degraded: bool,
}

/// Code-side source of truth for "pick / validate a model". Borrows the
/// provider map from [`Config`] instead of cloning — the selector is
/// typically constructed per-request from a long-lived `&Config`.
pub struct ModelSelector<'a> {
    providers: &'a HashMap<String, ProviderConfig>,
    enforce: bool,
    health: Option<Arc<ModelHealth>>,
}

impl<'a> ModelSelector<'a> {
    /// Build a selector from a live [`Config`]. Does NOT own the config —
    /// callers must keep `cfg` alive for the selector's lifetime. No
    /// health tracker is attached; use [`Self::with_health`] after
    /// construction to consult runtime quarantine state.
    pub fn new(cfg: &'a Config) -> Self {
        Self {
            providers: &cfg.providers,
            enforce: cfg.enforce_model_capabilities,
            health: None,
        }
    }

    /// Build a selector directly from a pre-extracted provider map
    /// and enforcement flag. Used by [`crate::research::coordinator`]
    /// where the full [`Config`] isn't in scope (the coordinator only
    /// carries the capability subset it needs, to keep
    /// `CoordinatorConfig` cloneable).
    pub fn from_providers(
        providers: &'a HashMap<String, ProviderConfig>,
        enforce: bool,
    ) -> Self {
        Self {
            providers,
            enforce,
            health: None,
        }
    }

    /// Attach the runtime health tracker. Quarantined pairs are
    /// rejected by [`Self::validate`] (and therefore filtered out by
    /// [`Self::is_available`] / [`Self::filter_chain`] /
    /// [`Self::rank_for`]) for as long as the quarantine lasts.
    pub fn with_health(mut self, health: Arc<ModelHealth>) -> Self {
        self.health = Some(health);
        self
    }

    /// Accessor for tests / callers that want to know whether
    /// enforcement is live.
    pub fn enforcement_enabled(&self) -> bool {
        self.enforce
    }

    /// Look up capabilities for `(provider, model)`. Returns
    /// [`ModelCapabilities::unknown`] for unknown pairs (permissive) —
    /// this matches [`ProviderConfig::capabilities_for`] and keeps the
    /// selector back-compat with un-seeded configs.
    pub fn caps(&self, r: &ModelRef) -> ModelCapabilities {
        match self.providers.get(&r.provider) {
            Some(pc) => pc.capabilities_for(&r.model),
            None => ModelCapabilities::unknown(),
        }
    }

    /// Validate a pinned `(provider, model, task)` triple. Returns a
    /// typed [`CapError`] describing the specific failure so callers
    /// can log / short-circuit uniformly. **Does not** check enforcement
    /// — pinned validation always speaks truth. Callers that want
    /// advisory-only behaviour should demote the error to a `warn!`
    /// themselves.
    pub fn validate(
        &self,
        provider: &str,
        model: &str,
        task: TaskKind,
    ) -> Result<(), CapError> {
        let Some(pc) = self.providers.get(provider) else {
            return Err(CapError::UnknownProvider(provider.to_string()));
        };
        // Resolve model aliases so "kimi-k2.6" routes to the caps block
        // keyed by "kimi-for-coding".
        let real = pc.resolve_model_alias(model);
        // `models` is the declared upstream list. Alias keys are
        // appended via `models_with_aliases`, which is what the menu
        // uses — we accept either.
        let known = pc.models.iter().any(|m| m == model || m == real)
            || pc.model_aliases.contains_key(model);
        if !known {
            return Err(CapError::UnknownModel {
                provider: provider.to_string(),
                model: model.to_string(),
            });
        }

        let caps = pc.capabilities_for(model);
        match caps.status {
            ModelStatus::Deprecated | ModelStatus::Experimental => {
                return Err(CapError::StatusBlocks {
                    provider: provider.to_string(),
                    model: model.to_string(),
                    status: caps.status,
                });
            }
            ModelStatus::Active | ModelStatus::Degraded => {}
        }
        if !caps.fits(task) {
            return Err(CapError::TaskMismatch {
                provider: provider.to_string(),
                model: model.to_string(),
                task,
                task_fit: caps.task_fit.clone(),
            });
        }
        if let Some(h) = &self.health
            && let Some(until) = h.quarantined_until(provider, model)
        {
            return Err(CapError::Quarantined {
                provider: provider.to_string(),
                model: model.to_string(),
                until,
            });
        }
        Ok(())
    }

    /// Non-pinned availability check used by ranking / fallback
    /// filtering. Honours [`Config::enforce_model_capabilities`]: when
    /// enforcement is off, this returns `true` unconditionally (legacy
    /// behaviour) so existing operators see no change in selection.
    ///
    /// When enforcement is on, returns `false` for any
    /// `validate(..)` failure, and additionally for
    /// `status=Degraded` unless `budget.allow_degraded` is `true`.
    pub fn is_available(
        &self,
        r: &ModelRef,
        task: TaskKind,
        budget: &Budget,
    ) -> bool {
        if !self.enforce {
            return true;
        }
        match self.validate(&r.provider, &r.model, task) {
            Ok(()) => {
                if budget.allow_degraded {
                    return true;
                }
                let caps = self.caps(r);
                !matches!(caps.status, ModelStatus::Degraded)
            }
            Err(_) => false,
        }
    }

    /// Rank every `(provider, model)` pair in the catalog for `task`,
    /// best-first. Applies the status/fit/health filter internally so the
    /// returned list never contains a pair that would be rejected by
    /// [`Self::is_available`] with the same `budget`.
    ///
    /// Ordering (lower is better):
    /// 1. `status` — Active before Degraded (Deprecated / Experimental
    ///    are filtered out entirely).
    /// 2. `quality_tier` — S > A > B > C > unknown.
    /// 3. `latency_tier` — Fast > Medium > Slow > unknown.
    /// 4. `cost_tier` — Free > Cheap > Medium > Premium > unknown.
    ///
    /// Ties fall back to alphabetical `provider/model` so the output is
    /// stable across runs (important for tests and for the exporter in
    /// Phase 5).
    pub fn rank_for(&self, task: TaskKind, budget: &Budget) -> Vec<ModelRef> {
        use super::types::{CostTier, LatencyTier, QualityTier};

        let mut scored: Vec<(ModelRef, (u8, u8, u8, u8, String))> = Vec::new();
        for (pname, pc) in self.providers {
            for model in &pc.models {
                let r = ModelRef::new(pname.clone(), model.clone());
                if !self.is_available(&r, task, budget) {
                    continue;
                }
                let caps = pc.capabilities_for(model);
                let status_score = match caps.status {
                    ModelStatus::Active => 0,
                    ModelStatus::Degraded => 1,
                    _ => continue,
                };
                let quality_score = match caps.quality_tier {
                    Some(QualityTier::S) => 0,
                    Some(QualityTier::A) => 1,
                    Some(QualityTier::B) => 2,
                    Some(QualityTier::C) => 3,
                    None => 2, // unknown ≈ middle of the road (A/B border)
                };
                let latency_score = match caps.latency_tier {
                    Some(LatencyTier::Fast) => 0,
                    Some(LatencyTier::Medium) => 1,
                    Some(LatencyTier::Slow) => 2,
                    None => 1,
                };
                let cost_score = match caps.cost_tier {
                    Some(CostTier::Free) => 0,
                    Some(CostTier::Cheap) => 1,
                    Some(CostTier::Medium) => 2,
                    Some(CostTier::Premium) => 3,
                    None => 1,
                };
                let tiebreak = format!("{}/{}", pname, model);
                scored.push((
                    r,
                    (status_score, quality_score, latency_score, cost_score, tiebreak),
                ));
            }
        }
        scored.sort_by(|a, b| a.1.cmp(&b.1));
        scored.into_iter().map(|(r, _)| r).collect()
    }

    /// Given an ordered chain of `provider/model` pairs (bare model ids
    /// are paired with `default_provider`), drop the entries that
    /// [`Self::is_available`] would reject for `task`. Used by
    /// [`crate::research::coordinator::ResearchCoordinator`] to trim
    /// `fallback_models` before dispatch — avoids a 90 s HTTP burn on a
    /// known-bad model.
    ///
    /// Soft-mode (`enforce=false`) returns the chain untouched so
    /// operators see identical ordering until they flip the flag.
    /// Behaviour under enforcement:
    /// * unknown / unparseable entries are dropped with a `warn!`
    /// * rejected entries log the typed [`CapError`] at `info!` so
    ///   operators can understand why a chain shrank.
    pub fn filter_chain(
        &self,
        chain: &[String],
        default_provider: &str,
        task: TaskKind,
        budget: &Budget,
    ) -> Vec<(String, String)> {
        let mut out = Vec::with_capacity(chain.len());
        for raw in chain {
            let (provider, model) =
                match crate::research::parse_provider_model_pair(raw) {
                    Some(pair) => pair,
                    None => {
                        if default_provider.is_empty() {
                            tracing::warn!(
                                entry = %raw,
                                "fallback entry has no provider prefix and no default_provider; skipping"
                            );
                            continue;
                        }
                        (default_provider.to_string(), raw.clone())
                    }
                };
            if !self.enforce {
                out.push((provider, model));
                continue;
            }
            match self.validate(&provider, &model, task) {
                Ok(()) => {
                    let r = ModelRef::new(provider.clone(), model.clone());
                    let caps = self.caps(&r);
                    if matches!(caps.status, ModelStatus::Degraded)
                        && !budget.allow_degraded
                    {
                        tracing::info!(
                            provider = %provider,
                            model = %model,
                            task = %task,
                            "skipping Degraded model from fallback chain (budget.allow_degraded=false)"
                        );
                        continue;
                    }
                    out.push((provider, model));
                }
                Err(e) => {
                    tracing::info!(
                        provider = %provider,
                        model = %model,
                        task = %task,
                        error = %e,
                        "dropping fallback entry (capability mismatch)"
                    );
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ProviderConfig;
    use crate::model_catalog::{
        CostTier, LatencyTier, ModelCapabilities, ModelStatus, QualityTier, TaskKind,
        ToolUseLevel,
    };

    fn pc(models: &[&str], caps: &[(&str, ModelCapabilities)]) -> ProviderConfig {
        ProviderConfig {
            provider_type: "openai_compat".into(),
            api_key: "k".into(),
            api_keys: Vec::new(),
            base_url: None,
            models: models.iter().map(|s| s.to_string()).collect(),
            max_tokens: None,
            temperature: None,
            context_window: None,
            headers: Default::default(),
            supports_vision: None,
            model_aliases: Default::default(),
            capabilities: caps
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect(),
        }
    }

    fn cfg(providers: Vec<(&str, ProviderConfig)>, enforce: bool) -> Config {
        let mut map = std::collections::HashMap::new();
        for (n, p) in providers {
            map.insert(n.to_string(), p);
        }
        Config {
            providers: map,
            enforce_model_capabilities: enforce,
            ..Default::default()
        }
    }

    #[test]
    fn validate_rejects_unknown_provider() {
        let c = cfg(vec![], true);
        let s = ModelSelector::new(&c);
        let err = s.validate("ghost", "m", TaskKind::Chat).unwrap_err();
        assert!(matches!(err, CapError::UnknownProvider(_)));
    }

    #[test]
    fn validate_rejects_unknown_model() {
        let p = pc(&["gpt-4o"], &[]);
        let c = cfg(vec![("openai", p)], true);
        let s = ModelSelector::new(&c);
        let err = s.validate("openai", "gpt-99", TaskKind::Chat).unwrap_err();
        assert!(matches!(err, CapError::UnknownModel { .. }));
    }

    #[test]
    fn validate_rejects_deprecated_status() {
        let mut caps = ModelCapabilities::unknown();
        caps.status = ModelStatus::Deprecated;
        let p = pc(&["qwen3.6-plus"], &[("qwen3.6-plus", caps)]);
        let c = cfg(vec![("qwen", p)], true);
        let s = ModelSelector::new(&c);
        let err = s
            .validate("qwen", "qwen3.6-plus", TaskKind::Chat)
            .unwrap_err();
        assert!(matches!(err, CapError::StatusBlocks { .. }));
    }

    #[test]
    fn validate_rejects_task_fit_miss() {
        let mut caps = ModelCapabilities::unknown();
        caps.task_fit = vec![TaskKind::Chat];
        let p = pc(&["m"], &[("m", caps)]);
        let c = cfg(vec![("p", p)], true);
        let s = ModelSelector::new(&c);
        let err = s.validate("p", "m", TaskKind::Research).unwrap_err();
        assert!(matches!(err, CapError::TaskMismatch { .. }));
    }

    #[test]
    fn validate_accepts_active_pair_that_fits() {
        let caps = ModelCapabilities {
            task_fit: vec![TaskKind::Research, TaskKind::Coding],
            ..ModelCapabilities::unknown()
        };
        let p = pc(&["m"], &[("m", caps)]);
        let c = cfg(vec![("p", p)], true);
        let s = ModelSelector::new(&c);
        assert!(s.validate("p", "m", TaskKind::Research).is_ok());
    }

    #[test]
    fn is_available_soft_mode_always_true() {
        let mut caps = ModelCapabilities::unknown();
        caps.status = ModelStatus::Deprecated;
        let p = pc(&["m"], &[("m", caps)]);
        let c = cfg(vec![("p", p)], /*enforce=*/ false);
        let s = ModelSelector::new(&c);
        let r = ModelRef::new("p", "m");
        assert!(s.is_available(&r, TaskKind::Research, &Budget::default()));
    }

    #[test]
    fn is_available_hard_mode_filters_degraded() {
        let caps = ModelCapabilities {
            status: ModelStatus::Degraded,
            task_fit: vec![TaskKind::Research],
            tool_use: ToolUseLevel::TextOnly,
            ..ModelCapabilities::unknown()
        };
        let p = pc(&["glm-5-turbo"], &[("glm-5-turbo", caps)]);
        let c = cfg(vec![("zai", p)], true);
        let s = ModelSelector::new(&c);
        let r = ModelRef::new("zai", "glm-5-turbo");
        // Default budget: degraded filtered.
        assert!(!s.is_available(&r, TaskKind::Research, &Budget::default()));
        // Opt-in: degraded allowed.
        let mut budget = Budget::default();
        budget.allow_degraded = true;
        assert!(s.is_available(&r, TaskKind::Research, &budget));
    }

    #[test]
    fn rank_for_orders_by_quality_then_latency_then_cost() {
        let top = ModelCapabilities {
            task_fit: vec![TaskKind::Research],
            quality_tier: Some(QualityTier::S),
            latency_tier: Some(LatencyTier::Medium),
            cost_tier: Some(CostTier::Medium),
            ..ModelCapabilities::unknown()
        };
        let mid = ModelCapabilities {
            task_fit: vec![TaskKind::Research],
            quality_tier: Some(QualityTier::A),
            latency_tier: Some(LatencyTier::Fast),
            cost_tier: Some(CostTier::Cheap),
            ..ModelCapabilities::unknown()
        };
        let cheap_bad = ModelCapabilities {
            task_fit: vec![TaskKind::Chat], // not research
            quality_tier: Some(QualityTier::S),
            ..ModelCapabilities::unknown()
        };
        let p = pc(
            &["top", "mid", "chatonly"],
            &[
                ("top", top),
                ("mid", mid),
                ("chatonly", cheap_bad),
            ],
        );
        let c = cfg(vec![("p", p)], true);
        let s = ModelSelector::new(&c);
        let ranked = s.rank_for(TaskKind::Research, &Budget::default());
        assert_eq!(ranked.len(), 2, "chat-only filtered out");
        assert_eq!(ranked[0].model, "top", "S tier first");
        assert_eq!(ranked[1].model, "mid", "A tier second");
    }

    #[test]
    fn filter_chain_drops_deprecated_and_off_task() {
        let live = ModelCapabilities {
            task_fit: vec![TaskKind::Research],
            ..ModelCapabilities::unknown()
        };
        let mut dead = ModelCapabilities::unknown();
        dead.status = ModelStatus::Deprecated;
        let chat_only = ModelCapabilities {
            task_fit: vec![TaskKind::Chat],
            ..ModelCapabilities::unknown()
        };
        let p = pc(
            &["live", "dead", "chatonly"],
            &[
                ("live", live),
                ("dead", dead),
                ("chatonly", chat_only),
            ],
        );
        let c = cfg(vec![("p", p)], true);
        let s = ModelSelector::new(&c);

        let chain = vec!["p/live".into(), "p/dead".into(), "p/chatonly".into()];
        let kept = s.filter_chain(&chain, "", TaskKind::Research, &Budget::default());
        assert_eq!(kept, vec![("p".into(), "live".into())]);
    }

    #[test]
    fn filter_chain_soft_mode_preserves_everything() {
        let mut dead = ModelCapabilities::unknown();
        dead.status = ModelStatus::Deprecated;
        let p = pc(&["live", "dead"], &[("dead", dead)]);
        let c = cfg(vec![("p", p)], /*enforce=*/ false);
        let s = ModelSelector::new(&c);
        let chain = vec!["p/live".into(), "p/dead".into()];
        let kept = s.filter_chain(&chain, "", TaskKind::Research, &Budget::default());
        assert_eq!(kept.len(), 2, "soft mode keeps the chain intact");
    }

    #[test]
    fn filter_chain_bare_models_use_default_provider() {
        let caps = ModelCapabilities {
            task_fit: vec![TaskKind::Research],
            ..ModelCapabilities::unknown()
        };
        let p = pc(&["qwen3.5-plus"], &[("qwen3.5-plus", caps)]);
        let c = cfg(vec![("qwen", p)], true);
        let s = ModelSelector::new(&c);
        // Bare model id — must pair with default_provider.
        let kept = s.filter_chain(
            &["qwen3.5-plus".into()],
            "qwen",
            TaskKind::Research,
            &Budget::default(),
        );
        assert_eq!(kept, vec![("qwen".into(), "qwen3.5-plus".into())]);
    }

    #[test]
    fn quarantined_pair_rejected_by_validate() {
        use crate::model_catalog::health::{HealthEventKind, ModelHealth, ModelHealthConfig};
        let caps = ModelCapabilities {
            task_fit: vec![TaskKind::Research],
            ..ModelCapabilities::unknown()
        };
        let p = pc(&["m"], &[("m", caps)]);
        let c = cfg(vec![("zai", p)], true);
        let dir = tempfile::tempdir().unwrap();
        let mut hc = ModelHealthConfig::default();
        hc.log_path = Some(dir.path().join("mh.jsonl"));
        let h = std::sync::Arc::new(ModelHealth::new(hc));
        // Force quarantine: 3 empties.
        for _ in 0..3 {
            h.record_event("zai", "m", HealthEventKind::Empty);
        }
        let s = ModelSelector::new(&c).with_health(h.clone());
        let err = s.validate("zai", "m", TaskKind::Research).unwrap_err();
        assert!(matches!(err, CapError::Quarantined { .. }));
        // is_available should also reject.
        assert!(!s.is_available(
            &ModelRef::new("zai", "m"),
            TaskKind::Research,
            &Budget::default()
        ));
    }

    #[test]
    fn bare_model_without_default_provider_drops() {
        let c = cfg(vec![], true);
        let s = ModelSelector::new(&c);
        let kept = s.filter_chain(
            &["orphan".into()],
            "",
            TaskKind::Research,
            &Budget::default(),
        );
        assert!(kept.is_empty());
    }
}
