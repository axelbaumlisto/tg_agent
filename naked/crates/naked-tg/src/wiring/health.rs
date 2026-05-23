//! Boot-time health checks: provider audit, vision probes, capability sweeps.
//!
//! Extracted from wiring.rs (T2, PLAN_v13_SOLID_AUDIT) for SRP.

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
