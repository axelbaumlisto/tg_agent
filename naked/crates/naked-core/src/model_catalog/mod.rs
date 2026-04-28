//! Model capabilities catalog — per-(provider, model) structured knowledge
//! consumed by every selection site in the agent.
//!
//! See [`types`] for the schema. Higher phases of the implementation plan
//! will add:
//! - `selector.rs` — ranking / validation (Phase 2)
//! - `health.rs`   — runtime health tracker with rolling 24 h window (Phase 3)
//! - `observation.rs` — auto-enrichment memory drafts (Phase 4)
//!
//! Phase 0 (this module) introduces the types only; no behavior changes.

pub mod enrichment;
pub mod exporter;
pub mod health;
pub mod selector;
pub mod types;

pub use enrichment::{
    CatalogAdd, CatalogDeprecate, CatalogSuggestions, CatalogUpdate, ModelObservation,
    ObservationKind, ObservationRecorder, promote_observations, run_daily_promotion,
};
pub use health::{HealthEventKind, HealthWindow, ModelHealth, ModelHealthConfig};
pub use selector::{Budget, CapError, ModelSelector};
pub use types::{
    CostTier, LatencyTier, ModelCapabilities, ModelStatus, QualityTier, ReasoningLevel, TaskKind,
    ToolUseLevel,
};

/// Fully-qualified reference to a model behind a concrete provider entry in
/// `naked.json`. Used as the lookup key for capabilities and as the return
/// type of [`crate::model_catalog::types::ModelCapabilities`] consumers.
///
/// The pair is `(provider_id, model_id)` — **not** an alias. Callers should
/// resolve aliases via [`crate::config::ProviderConfig::resolve_model_alias`]
/// *before* constructing a `ModelRef`, so the capabilities lookup matches
/// the keys the operator wrote in `naked.json`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelRef {
    pub provider: String,
    pub model: String,
}

impl ModelRef {
    pub fn new(provider: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
        }
    }
}

impl std::fmt::Display for ModelRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.provider, self.model)
    }
}
