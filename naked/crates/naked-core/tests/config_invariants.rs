//! BUG_REGISTRY D-INV-WIRING-SMOKE + INV-1 + INV-2 production invariants.
//!
//! Phase 2 of BUG_REGISTRY § 6 roadmap. Tests that the production
//! `state/naked.json` (reached via the `naked.json` symlink in the
//! crate's working dir) can be parsed AND that every per-model
//! `capabilities[X].supports_vision = Some(true)` correctly routes to
//! true via `is_vision_capable_with_provider`. Catches schema drift
//! between caps declaration and routing consumer.
//!
//! Test gracefully skips if the prod config isn't present (CI without
//! `state/`). Local dev always has it via the symlink.

use std::path::PathBuf;

use naked_core::config::Config;

/// Returns the path to the production `naked.json` symlink, or None if
/// it isn't reachable from the current working dir (typical in CI that
/// doesn't include `state/`).
fn locate_prod_config() -> Option<PathBuf> {
    // `cargo test` runs in the crate root: `naked/crates/naked-core/`.
    // The symlink lives two levels up: `naked/naked.json -> ../state/naked.json`.
    let candidates = [
        PathBuf::from("../../naked.json"),
        PathBuf::from("../naked.json"),
        PathBuf::from("naked.json"),
        PathBuf::from("../../../state/naked.json"),
    ];
    candidates.into_iter().find(|p| p.exists())
}

/// D-INV-WIRING-SMOKE: production config must parse. If the schema
/// drifts (a renamed field, an unknown variant in capabilities), this
/// test fails before deploy.
#[test]
fn production_config_parses_without_error() {
    let Some(path) = locate_prod_config() else {
        eprintln!("skipped: no prod naked.json reachable (CI without state/)");
        return;
    };
    let cfg = Config::from_json_file(&path).expect("Config::from_json_file");
    // Sanity: at least one provider and one model should be defined.
    assert!(
        !cfg.providers.is_empty(),
        "production config has no providers"
    );
    assert!(
        !cfg.default_model.is_empty(),
        "production config has empty default_model"
    );
    assert!(
        cfg.providers.contains_key(&cfg.default_provider),
        "default_provider {:?} not in providers map",
        cfg.default_provider
    );
}

/// INV-1 + INV-2: every model in the production config whose
/// `capabilities[model].supports_vision = Some(true)` MUST be routable
/// through `is_vision_capable_with_provider`. Closes a class of
/// "schema declared but runtime ignores it" bugs (BUG_REGISTRY B03).
#[test]
fn production_caps_supports_vision_is_routable() {
    let Some(path) = locate_prod_config() else {
        eprintln!("skipped: no prod naked.json reachable");
        return;
    };
    let cfg = Config::from_json_file(&path).expect("parse");

    let mut violations: Vec<String> = Vec::new();
    for (provider_name, provider) in &cfg.providers {
        for (model_id, caps) in &provider.capabilities {
            if caps.supports_vision == Some(true) {
                let routable = cfg
                    .tg_media
                    .is_vision_capable_with_provider(model_id, Some(provider));
                if !routable {
                    violations.push(format!(
                        "{provider_name}/{model_id}: caps.supports_vision=Some(true) \
                         but is_vision_capable_with_provider=false"
                    ));
                }
            }
        }
    }

    assert!(
        violations.is_empty(),
        "BUG_REGISTRY INV-1 violated: vision-capable models in caps that don't route as vision-capable:\n  {}",
        violations.join("\n  ")
    );
}

/// INV-2 contrapositive: every model whose name matches a vision needle
/// pattern (case-insensitive `vl`, `vision`, `multimodal`) MUST resolve
/// true OR have an explicit `caps.supports_vision = Some(false)` (operator
/// override). Catches the next "we added a Qwen 4 VL family and forgot
/// to update needles" regression.
#[test]
fn production_vision_named_models_either_route_or_explicit_deny() {
    let Some(path) = locate_prod_config() else {
        eprintln!("skipped: no prod naked.json reachable");
        return;
    };
    let cfg = Config::from_json_file(&path).expect("parse");

    let vision_naming = regex_lite::Regex::new(r"(?i)\bvl\b|vision|multimodal").ok();
    // Fallback substring match if regex_lite isn't a dep — manual.
    let looks_vision = |m: &str| -> bool {
        let lc = m.to_ascii_lowercase();
        // Word-ish boundaries via separator chars (-, ., _).
        let tokens: Vec<&str> = lc.split(|c: char| !c.is_alphanumeric()).collect();
        tokens
            .iter()
            .any(|t| *t == "vl" || *t == "vision" || *t == "multimodal")
            || lc.contains("vision")
            || lc.contains("multimodal")
    };
    let _ = vision_naming; // regex_lite is not a dep — keep manual matcher.

    let mut violations: Vec<String> = Vec::new();
    for (provider_name, provider) in &cfg.providers {
        for model_id in &provider.models {
            if !looks_vision(model_id) {
                continue;
            }
            let routable = cfg
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
                    "{provider_name}/{model_id}: name suggests vision but \
                     is_vision_capable_with_provider=false and no explicit deny"
                ));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "BUG_REGISTRY INV-2 violated: vision-named models that don't route:\n  {}",
        violations.join("\n  ")
    );
}
