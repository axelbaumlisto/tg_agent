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

/// B63 / AB-5: every chat fallback provider must exist in `providers`.
/// Option-B boot audit enumerates configured providers' own keys; a dangling
/// fallback name would otherwise be invisible to the de-duplicated audit.
#[test]
fn production_fallback_provider_names_exist() {
    let Some(path) = locate_prod_config() else {
        eprintln!("skipped: no prod naked.json reachable");
        return;
    };
    let cfg = Config::from_json_file(&path).expect("parse");

    let missing: Vec<String> = cfg
        .fallback_providers()
        .into_iter()
        .filter_map(|(provider, model)| {
            if cfg.providers.contains_key(&provider) {
                None
            } else {
                Some(format!("{provider}/{model}"))
            }
        })
        .collect();

    assert!(
        missing.is_empty(),
        "config.fallback[] references provider(s) absent from config.providers: {}",
        missing.join(", ")
    );
}

/// Shared driver for the production-config sweeps below.
///
/// Every sweep repeats the same three steps: locate the live `naked.json`
/// (skipping cleanly in a CI checkout without `state/`), collect human-readable
/// violation strings, and fail with all of them at once. Five copies of that
/// scaffolding had accumulated, so a new sweep meant re-deriving the skip
/// semantics by hand — easy to get subtly wrong (an early `return` that skips
/// instead of failing hides a real regression). `check` only has to describe
/// what is wrong.
fn sweep_prod_config(inv: &str, check: impl FnOnce(&Config, &mut Vec<String>)) {
    let Some(path) = locate_prod_config() else {
        eprintln!("skipped: no prod naked.json reachable (CI without state/)");
        return;
    };
    let cfg = Config::from_json_file(&path).expect("Config::from_json_file");

    let mut violations: Vec<String> = Vec::new();
    check(&cfg, &mut violations);

    assert!(
        violations.is_empty(),
        "{inv} violated:\n  {}",
        violations.join("\n  ")
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

/// D-INV-PROVIDER-ALIAS (B105), production half: every `provider_aliases`
/// entry must be *useful* and *reachable*.
///
/// A retirement alias exists so 352 session configs pinned to a removed
/// provider stop silently reporting a dead provider/model as healthy. That
/// only holds if each alias (a) names a provider that is actually gone,
/// (b) lands on a provider that IS in the catalog, and (c) pins a model that
/// the target provider really serves — otherwise the redirect just moves the
/// lie one hop. This sweeps the live config on every run, so editing
/// `state/naked.json` badly fails before deploy instead of at runtime.
#[test]
fn b105_production_provider_aliases_resolve_to_live_pairs() {
    sweep_prod_config("B105 provider_aliases", |cfg, violations| {
        for from in cfg.provider_aliases.keys() {
            if cfg.providers.contains_key(from) {
                violations.push(format!(
                    "{from}: still LIVE in providers — the alias is dead weight and \
                 the live entry silently wins"
                ));
                continue;
            }
            let (prov, model) = cfg.resolve_provider_alias(from);
            let Some(pc) = cfg.providers.get(&prov) else {
                violations.push(format!(
                    "{from} -> {prov}: target provider is NOT in the catalog"
                ));
                continue;
            };
            let Some(m) = model else {
                // Bare form is legal, but then the session's own (dead) model
                // string survives — for a retirement that is almost never right.
                violations.push(format!(
                    "{from} -> {prov}: bare alias keeps the session's model; a retired \
                 provider needs the `provider/model` form"
                ));
                continue;
            };
            let serves = pc.models.contains(&m) || pc.capabilities.contains_key(&m);
            if !serves {
                violations.push(format!(
                    "{from} -> {prov}/{m}: provider does not serve that model"
                ));
            }
        }
    });
}

/// D-INV-PROVIDER-OWNS-ITS-MODELS (B107): a provider's `models` list must not
/// advertise another provider's models.
///
/// Observed 2026-08-05: `moonshot`, `kimi-code` and `openai` all listed
/// `deepseek-v4-flash`/`deepseek-v4-pro` in `models` while their
/// `capabilities` still described kimi / gpt models. Because `/model` and the
/// selector read `models`, picking "moonshot" silently served deepseek — with
/// no fallback and no warning, since the config itself said so. The upstream
/// proxy *does* serve the real kimi ids, so this was pure config drift (most
/// likely a bulk edit during the 2026-07-25 provider purge).
///
/// Heuristic, not a hardcoded map: every entry in `models` should either be
/// described in that provider's own `capabilities`, or at least not be a model
/// that ONLY some other provider declares capabilities for. That keeps the
/// check useful for shared proxy ids while catching wholesale substitution.
#[test]
fn b107_provider_models_are_not_another_providers() {
    sweep_prod_config("B107 provider/model ownership", |cfg, violations| {
        for (name, pc) in &cfg.providers {
            for m in &pc.models {
                if pc.capabilities.contains_key(m) {
                    continue; // provider describes it itself — fine
                }
                // Who DOES declare capabilities for this model id?
                let owners: Vec<&String> = cfg
                    .providers
                    .iter()
                    .filter(|(other, opc)| *other != name && opc.capabilities.contains_key(m))
                    .map(|(other, _)| other)
                    .collect();
                if !owners.is_empty() {
                    violations.push(format!(
                    "{name}.models lists '{m}', but only {owners:?} declare capabilities for it \
                     — picking {name} would silently serve another provider's model"
                ));
                }
            }
        }
    });
}

/// D-INV-ACTIVE-CAPS-ARE-SERVABLE (B108): a model may not be advertised as
/// `status: active` in `capabilities` unless its provider actually lists it in
/// `models`.
///
/// `ModelSelector::rank_for` filters on `status`, and `skills/model-catalog/
/// SKILL.md` is generated from the same data — so an `active` capability for a
/// model the provider cannot serve becomes a FIRST-CHOICE recommendation that
/// always fails. Observed 2026-08-05: eight moonshot ids
/// (`kimi-k2-thinking-turbo`, `moonshot-v1-*`, …) stayed `active` after B107
/// trimmed `models` to what the upstream really answers, and the catalog kept
/// recommending `moonshot / kimi-k2-thinking-turbo` for coding, research and
/// chat — every one of them a live 404/502.
///
/// Deprecating a dead model is therefore the *required* companion to removing
/// it from `models`; this test refuses to let the two drift apart again.
#[test]
fn b108_active_capabilities_must_be_in_provider_models() {
    sweep_prod_config("B108 active-but-unservable models", |cfg, violations| {
        for (name, pc) in &cfg.providers {
            for (model, caps) in &pc.capabilities {
                let active = matches!(
                    caps.status,
                    naked_core::model_catalog::ModelStatus::Active
                        | naked_core::model_catalog::ModelStatus::Degraded
                );
                if active && !pc.models.contains(model) {
                    violations.push(format!(
                        "{name}/{model}: capabilities say {:?} but it is not in {name}.models — \
                     the selector and the generated catalog will recommend a model that \
                     cannot be served",
                        caps.status
                    ));
                }
            }
        }
    });
}
