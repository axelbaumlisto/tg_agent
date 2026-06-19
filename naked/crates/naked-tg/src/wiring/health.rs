//! Boot-time health checks: provider audit, vision probes, capability sweeps.
//!
//! Extracted from wiring.rs (T2, PLAN_v13_SOLID_AUDIT) for SRP.

use std::collections::HashMap;

use naked_core::config::{ProviderConfig, ResolvedProvider};
use naked_core::error::AgentError;
use naked_core::provider::{ChatRequest, DedupAuditTarget, Provider};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct BootAuditProbeStats {
    pub(crate) known_dead: usize,
    pub(crate) inconclusive: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct BootAuditSummary {
    pub(crate) providers_seen: usize,
    pub(crate) targets_probed: usize,
    pub(crate) targets_skipped_dup: usize,
    pub(crate) known_dead: usize,
    pub(crate) inconclusive: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct DedupedProviderAuditTarget {
    pub(crate) provider_name: String,
    pub(crate) resolved: ResolvedProvider,
    pub(crate) selected_targets: Vec<DedupAuditTarget>,
}

fn resolved_subset_for_audit(
    config: &ProviderConfig,
    keys: Vec<String>,
) -> Option<ResolvedProvider> {
    let api_key = keys.first()?.clone();
    Some(ResolvedProvider {
        provider_type: config.provider_type.clone(),
        api_key,
        all_keys: keys,
        base_url: config.base_url.clone(),
        models: config.models.clone(),
        max_tokens: config.max_tokens,
        temperature: config.temperature,
        headers: config.headers.clone(),
        model_aliases: config.model_aliases.clone(),
    })
}

fn build_deduped_provider_audit_targets(
    providers: &[(String, ProviderConfig)],
) -> (BootAuditSummary, Vec<DedupedProviderAuditTarget>) {
    let key_inputs: Vec<(String, Vec<String>)> = providers
        .iter()
        .map(|(name, cfg)| (name.clone(), cfg.resolved_all_keys()))
        .collect();
    let plan = naked_core::provider::dedup_audit_targets(&key_inputs);

    let mut by_provider: HashMap<String, Vec<DedupAuditTarget>> = HashMap::new();
    for target in plan.targets.iter().cloned() {
        by_provider
            .entry(target.provider_name.clone())
            .or_default()
            .push(target);
    }

    let mut targets = Vec::new();
    for (provider_name, cfg) in providers {
        let Some(selected_targets) = by_provider.remove(provider_name) else {
            continue;
        };
        let Some((_, resolved_keys)) = key_inputs.iter().find(|(name, _)| name == provider_name)
        else {
            continue;
        };
        let selected_keys: Vec<String> = selected_targets
            .iter()
            .filter_map(|target| resolved_keys.get(target.key_index).cloned())
            .collect();
        if let Some(resolved) = resolved_subset_for_audit(cfg, selected_keys) {
            targets.push(DedupedProviderAuditTarget {
                provider_name: provider_name.clone(),
                resolved,
                selected_targets,
            });
        }
    }

    let summary = BootAuditSummary {
        providers_seen: plan.providers_seen,
        targets_probed: plan.targets.len(),
        targets_skipped_dup: plan.targets_skipped_dup,
        known_dead: 0,
        inconclusive: 0,
    };
    (summary, targets)
}

pub(crate) async fn audit_deduped_provider_targets<F, Fut>(
    providers: Vec<(String, ProviderConfig)>,
    probe: F,
) -> BootAuditSummary
where
    F: Fn(DedupedProviderAuditTarget) -> Fut,
    Fut: std::future::Future<Output = BootAuditProbeStats>,
{
    let (mut summary, targets) = build_deduped_provider_audit_targets(&providers);
    for target in targets {
        let stats = probe(target).await;
        summary.known_dead += stats.known_dead;
        summary.inconclusive += stats.inconclusive;
    }
    summary
}

pub(crate) async fn probe_deduped_target_with_created_provider(
    target: DedupedProviderAuditTarget,
) -> BootAuditProbeStats {
    let original_key_index = target
        .selected_targets
        .first()
        .map(|selected| selected.key_index)
        .unwrap_or(0);
    let provider = naked_core::create_provider(&target.provider_name, target.resolved);

    if provider.total_key_count() > 1 {
        provider.audit_keys_on_boot().await;
        return BootAuditProbeStats {
            known_dead: provider.blacklisted_key_count(),
            inconclusive: 0,
        };
    }

    let Some(probe_model) = provider
        .models()
        .first()
        .map(|model| model.model_id.clone())
    else {
        tracing::debug!(
            provider = %target.provider_name,
            "audit_keys_on_boot: no model configured, skipping single-key probe"
        );
        return BootAuditProbeStats {
            known_dead: 0,
            inconclusive: 1,
        };
    };

    let req = ChatRequest {
        model: probe_model,
        system: String::new(),
        messages: vec![serde_json::json!({
            "role": "user",
            "content": "hi"
        })],
        tools: Vec::new(),
        max_tokens: 1,
        temperature: None,
        reasoning: None,
    };
    match tokio::time::timeout(std::time::Duration::from_secs(5), provider.stream_chat(req)).await {
        Ok(Ok(_stream)) => BootAuditProbeStats::default(),
        Ok(Err(e)) => {
            let permanent = matches!(
                &e,
                AgentError::ProviderTyped(pe) if pe.is_permanent_key_failure()
            );
            if permanent {
                naked_core::types::PROVIDER_PERMANENT_BLACKLIST_COUNT
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                tracing::warn!(
                    provider = %target.provider_name,
                    key_index = original_key_index,
                    error = %e,
                    "provider key permanently blacklisted during boot audit"
                );
                BootAuditProbeStats {
                    known_dead: 1,
                    inconclusive: 0,
                }
            } else {
                BootAuditProbeStats {
                    known_dead: 0,
                    inconclusive: 1,
                }
            }
        }
        Err(_) => BootAuditProbeStats {
            known_dead: 0,
            inconclusive: 1,
        },
    }
}

pub(crate) fn boot_audit_dedup_enabled() -> bool {
    let disabled = |value: std::ffi::OsString| {
        let value = value.to_string_lossy();
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "no" | "off" | "legacy"
        )
    };

    if std::env::var_os("NAKED_BOOT_AUDIT_DEDUP").is_some_and(disabled) {
        return false;
    }
    if std::env::var_os("NAKED_BOOT_AUDIT_DEDUP_MODE").is_some_and(|value| {
        value
            .to_string_lossy()
            .trim()
            .eq_ignore_ascii_case("legacy")
    }) {
        return false;
    }
    true
}

/// Must be called **after** tracing is initialised (in [`crate::bootstrap`]).
/// BUG_REGISTRY D-INV-AUDIT-ALL-PROVIDERS: iterates every provider
/// name, resolves it via the `resolve` closure, and calls
/// `audit_keys_on_boot()` on the result. Sequential to avoid pulling
/// `futures_util` into naked-tg; each individual audit internally
/// runs its key probes in parallel via `join_all`.
///
/// `pub(crate)` so wiring.rs::tests can verify the loop hits EVERY
/// provider, not just the default — the bug that originally needed
/// to ship as part of R2 wiring iteration (commit `38c608b6`).
pub(crate) async fn audit_all_providers<F, Fut>(provider_names: Vec<String>, resolve: F)
where
    F: Fn(String) -> Fut,
    Fut: std::future::Future<Output = std::sync::Arc<dyn naked_core::provider::Provider>>,
{
    for name in provider_names {
        let provider = resolve(name).await;
        provider.audit_keys_on_boot().await;
    }
}

#[derive(Debug)]
pub(crate) enum VisionShapeOutcome {
    /// The provider accepted the image_url shape. May still have
    /// rejected our specific 1×1 PNG ("image too small") — that's
    /// content, not shape. Either way the capability is real.
    Accepted,
    /// The provider rejected the shape itself (e.g. "unknown variant
    /// `image_url`, expected `text`"). Caps are wrong.
    ShapeMismatch(String),
    /// Auth / rate-limit / timeout / network. Can't tell. Skip.
    Inconclusive(String),
}

/// Pure classifier for an error message returned by a vision probe.
/// Extracted for unit-testing without spinning up live providers.
pub(crate) fn classify_vision_probe_error(msg: &str) -> VisionShapeOutcome {
    let lc = msg.to_ascii_lowercase();
    // Definite shape-mismatch signals across the providers we care about:
    //   * serde-de error: "unknown variant `image_url`, expected `text`"
    //   * "expected text" / "only text content"
    //   * "does not support image" / "text-only model"
    if lc.contains("unknown variant")
        || lc.contains("expected text")
        || lc.contains("expected `text`")
        || lc.contains("only text content")
        || lc.contains("does not support image")
        || lc.contains("text-only model")
        || lc.contains("multimodal not supported")
    {
        return VisionShapeOutcome::ShapeMismatch(msg.chars().take(180).collect());
    }
    // "Image too small" / "min size" / size-related rejections — shape
    // accepted, content rejected. Either way the capability is real.
    if lc.contains("image must be")
        || lc.contains("too small")
        || lc.contains("min") && lc.contains("size")
        || lc.contains("width")
        || lc.contains("height")
    {
        return VisionShapeOutcome::Accepted;
    }
    // Auth, rate, timeout, network — can't tell, treat as inconclusive.
    VisionShapeOutcome::Inconclusive(msg.chars().take(180).collect())
}

/// 1×1 transparent PNG (the smallest valid PNG payload). Used as the
/// probe content — we expect every real vision API to either accept
/// or reject it with a SIZE error, but never a SHAPE error.
const VISION_PROBE_PIXEL_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

/// Send a 1×1 image_url to the provider's stream_chat and classify
/// the response. 8-second timeout per probe so total D-BOOT-VISION-PROBE
/// time is bounded by (number of vision-claimed models × 8 s); typically
/// 5–10 s in practice.
pub(crate) async fn probe_vision_content_shape(
    provider: &dyn naked_core::provider::Provider,
    model: &str,
) -> VisionShapeOutcome {
    use naked_core::provider::ChatRequest;
    let req = ChatRequest {
        model: model.into(),
        system: String::new(),
        messages: vec![serde_json::json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "image_url", "image_url": {
                    "url": format!("data:image/png;base64,{VISION_PROBE_PIXEL_PNG_B64}")
                }}
            ]
        })],
        tools: vec![],
        max_tokens: 1,
        temperature: None,
        reasoning: None,
    };
    let timeout = std::time::Duration::from_secs(8);
    match tokio::time::timeout(timeout, provider.stream_chat(req)).await {
        Ok(Ok(_stream)) => VisionShapeOutcome::Accepted,
        Ok(Err(e)) => classify_vision_probe_error(&e.to_string()),
        Err(_) => VisionShapeOutcome::Inconclusive("timeout".into()),
    }
}

/// BUG_REGISTRY D-BOOT-CAPS-INVARIANT: enforces INV-1 + INV-2 at
/// boot time. Returns the number of (provider, model) pairs whose
/// declared caps disagree with the routing function. Logs WARN per
/// violation with file:line-style context for the operator.
///
/// `pub(crate)` for unit-testing without spinning up async wiring.
/// `NAKED_STRICT_CAPS=1` env: escalate WARN → `std::process::exit(3)`
/// so misconfiguration can't reach prod silently.
pub(crate) fn boot_caps_invariant_sweep(config: &naked_core::config::Config) -> usize {
    let mut violations: Vec<String> = Vec::new();

    for (provider_name, provider) in &config.providers {
        // INV-1: per-model caps.supports_vision=Some(true) must route as true.
        for (model_id, caps) in &provider.capabilities {
            if caps.supports_vision == Some(true) {
                let routable = config
                    .tg_media
                    .is_vision_capable_with_provider(model_id, Some(provider));
                if !routable {
                    violations.push(format!(
                        "INV-1 {provider_name}/{model_id}: caps.supports_vision=Some(true) \
                         but is_vision_capable_with_provider=false"
                    ));
                }
            }
        }
        // INV-2: every model id matching vision-naming pattern must route OR
        // have explicit Some(false) deny.
        for model_id in &provider.models {
            let lc = model_id.to_ascii_lowercase();
            let looks_vision =
                lc.contains("vl") || lc.contains("vision") || lc.contains("multimodal");
            if !looks_vision {
                continue;
            }
            let routable = config
                .tg_media
                .is_vision_capable_with_provider(model_id, Some(provider));
            if routable {
                continue;
            }
            let explicit_deny = provider
                .capabilities
                .get(model_id)
                .and_then(|c| c.supports_vision)
                == Some(false);
            if !explicit_deny {
                violations.push(format!(
                    "INV-2 {provider_name}/{model_id}: name suggests vision but \
                     is_vision_capable_with_provider=false and no explicit deny"
                ));
            }
        }
    }

    let count = violations.len();
    for v in &violations {
        tracing::warn!(violation = %v, "caps invariant violation at boot");
    }

    if count == 0 {
        tracing::info!(
            providers = config.providers.len(),
            "caps invariant sweep clean (INV-1 + INV-2)"
        );
    } else if std::env::var_os("NAKED_STRICT_CAPS").is_some_and(|v| v == "1") {
        // Strict mode — fail boot rather than ship broken caps to prod.
        eprintln!(
            "❌ {} caps invariant violation(s) at boot and NAKED_STRICT_CAPS=1; refusing to start",
            count
        );
        for v in &violations {
            eprintln!("   {v}");
        }
        std::process::exit(3);
    }

    count
}

/// BUG_REGISTRY D-BOOT-DESCRIBER-WARN: boot-time health check for the
/// multimodal vision path. Emits a loud WARN if the default model
/// can't accept image content blocks AND `tg_media.vision` describer
/// fallback is unconfigured — in that state, every photo from a user
/// pinned to the default model gets the "[⚠ vision not configured]"
/// placeholder text and the image content is lost. INFO when healthy.
///
/// `pub(crate)` so this can be unit-tested without spinning up the
/// full async wiring pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MultimodalDescriberHealth {
    /// Default model is vision-capable; describer presence irrelevant.
    DefaultVision,
    /// Default model isn't vision-capable but describer fallback exists.
    DescriberFallback,
    /// Default model isn't vision-capable AND no describer — photos
    /// from default-model users will be silently dropped.
    Degraded,
}

pub(crate) fn check_multimodal_describer_health(
    config: &naked_core::config::Config,
) -> MultimodalDescriberHealth {
    let default_provider = config.providers.get(&config.default_provider);
    let default_is_vision = config
        .tg_media
        .is_vision_capable_with_provider(&config.default_model, default_provider);
    let describer = config.tg_media.vision.is_some();
    let outcome = match (default_is_vision, describer) {
        (true, _) => MultimodalDescriberHealth::DefaultVision,
        (false, true) => MultimodalDescriberHealth::DescriberFallback,
        (false, false) => MultimodalDescriberHealth::Degraded,
    };
    match outcome {
        MultimodalDescriberHealth::Degraded => {
            tracing::warn!(
                default_provider = %config.default_provider,
                default_model = %config.default_model,
                "multimodal degraded: default model is not vision-capable AND \
                 no tg_media.vision describer fallback configured. \
                 Photos from users on this model will be lost. \
                 Either pin to a vision-capable model (e.g. qwen3-vl-plus) \
                 or set tg_media.vision in naked.json."
            );
        }
        MultimodalDescriberHealth::DefaultVision => {
            tracing::info!(
                default_provider = %config.default_provider,
                default_model = %config.default_model,
                "multimodal: default model is vision-capable"
            );
        }
        MultimodalDescriberHealth::DescriberFallback => {
            tracing::info!(
                default_provider = %config.default_provider,
                default_model = %config.default_model,
                describer_model = ?config.tg_media.vision.as_ref().map(|v| &v.model),
                "multimodal: default is text-only, describer fallback active"
            );
        }
    }
    outcome
}
