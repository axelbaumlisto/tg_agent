//! Integration test: the freshly-seeded capability blocks in
//! `naked/naked.json` parse cleanly into [`naked_core::model_catalog`]
//! types and cover the providers + models the plan's Phase 0 identifies
//! as critical (known-bad `glm-5-turbo` and `qwen3.6-plus`, primary
//! research `kimi-for-coding`).
//!
//! Runs as a workspace-local integration test so CI fails loudly if a
//! future edit breaks the schema or forgets to deprecate a dead model.

use std::fs;
use std::path::PathBuf;

use naked_core::config::Config;
use naked_core::model_catalog::{ModelStatus, TaskKind, ToolUseLevel};

/// Returns a reachable `naked.json` *seed* config, or `None` when it
/// isn't present.
///
/// `naked/naked.json` is gitignored (it holds real config, normally a
/// symlink to `state/naked.json`), so on a fresh clone / CI it is absent
/// and these tests must skip rather than panic. Mirrors the
/// candidate-path + `.find(|p| p.exists())` approach used by
/// `config_invariants.rs::locate_prod_config`.
///
/// NOTE: unlike `locate_prod_config`, this deliberately does *not* fall
/// back to the tracked `../../../state/naked.json`. These tests assert
/// fixed *seed* capability values (e.g. `glm-5-turbo` == `TextOnly`)
/// against the seed file `naked/naked.json`; the auto-refreshed prod
/// `state/naked.json` legitimately drifts from those seed values, so
/// pointing seed assertions at it would produce spurious failures.
fn locate_seed_config() -> Option<PathBuf> {
    // The historically-intended path: crates/naked-core → naked/naked.json.
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let manifest_path = manifest
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join("naked.json"));
    // `cargo test` runs in the crate root: `naked/crates/naked-core/`.
    let candidates = [
        manifest_path,
        Some(PathBuf::from("../../naked.json")),
        Some(PathBuf::from("../naked.json")),
        Some(PathBuf::from("naked.json")),
    ];
    candidates.into_iter().flatten().find(|p| p.exists())
}

#[test]
fn seeded_naked_json_parses() {
    let Some(path) = locate_seed_config() else {
        eprintln!("skipped: no naked.json reachable (CI without state/)");
        return;
    };
    let text =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    let _: Config =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("parsing {}: {e}", path.display()));
}

#[test]
fn glm_turbo_is_marked_degraded_with_empty_content() {
    let Some(path) = locate_seed_config() else {
        eprintln!("skipped: no naked.json reachable (CI without state/)");
        return;
    };
    let text = fs::read_to_string(path).unwrap();
    let cfg: Config = serde_json::from_str(&text).unwrap();
    // Every GLM provider (zai, zai-oai, glm-cn, glm-cn-oai) must
    // carry the degraded flag so selectors can demote it automatically
    // once phase 2 ships. This is the single most important seed entry.
    for p in ["zai", "zai-oai", "glm-cn", "glm-cn-oai"] {
        let pcfg = cfg
            .providers
            .get(p)
            .unwrap_or_else(|| panic!("provider {p} must exist"));
        let caps = pcfg.capabilities_for("glm-5-turbo");
        assert_eq!(
            caps.status,
            ModelStatus::Degraded,
            "{p}/glm-5-turbo must be Degraded",
        );
        assert_eq!(caps.tool_use, ToolUseLevel::TextOnly);
        assert!(
            caps.known_failure_modes
                .iter()
                .any(|s| s == "empty_content"),
            "{p}/glm-5-turbo must note empty_content",
        );
        assert!(!caps.fits(TaskKind::Research));
    }
}

#[test]
fn qwen_3_6_plus_is_active() {
    let Some(path) = locate_seed_config() else {
        eprintln!("skipped: no naked.json reachable (CI without state/)");
        return;
    };
    let text = fs::read_to_string(path).unwrap();
    let cfg: Config = serde_json::from_str(&text).unwrap();
    let qwen = cfg.providers.get("qwen").unwrap();
    let caps = qwen.capabilities_for("qwen3.6-plus");
    assert_eq!(caps.status, ModelStatus::Active);
    // Revived 2026-04-26: fits chat + research.
    assert!(caps.fits(TaskKind::Chat));
    assert!(caps.fits(TaskKind::Research));
}

#[test]
fn kimi_for_coding_is_task_fit_for_coding_and_research() {
    let Some(path) = locate_seed_config() else {
        eprintln!("skipped: no naked.json reachable (CI without state/)");
        return;
    };
    let text = fs::read_to_string(path).unwrap();
    let cfg: Config = serde_json::from_str(&text).unwrap();
    let p = cfg.providers.get("kimi-code").unwrap();
    let caps = p.capabilities_for("kimi-for-coding");
    assert_eq!(caps.status, ModelStatus::Active);
    assert!(caps.fits(TaskKind::Coding));
    assert!(caps.fits(TaskKind::Research));
    assert!(!caps.fits(TaskKind::Vision));
    assert_eq!(caps.tool_use, ToolUseLevel::Full);
    assert!(caps.context_window.unwrap_or(0) >= 200_000);
}

#[test]
fn capabilities_for_falls_back_to_unknown() {
    let Some(path) = locate_seed_config() else {
        eprintln!("skipped: no naked.json reachable (CI without state/)");
        return;
    };
    let text = fs::read_to_string(path).unwrap();
    let cfg: Config = serde_json::from_str(&text).unwrap();
    let p = cfg.providers.get("anthropic").unwrap();
    let caps = p.capabilities_for("no-such-model-ever");
    // Unknown pair → permissive default. This is the back-compat seam
    // that keeps legacy configs working.
    assert_eq!(caps.status, ModelStatus::Active);
    for t in TaskKind::ALL {
        assert!(caps.fits(*t));
    }
}

#[test]
fn every_declared_model_has_active_or_explicit_status() {
    // Sanity check: for every provider, every *declared* model should
    // either (a) have an explicit capabilities entry, or (b) be served
    // by the permissive unknown() fallback. The test doesn't require
    // every model to have caps — but it does require consistency.
    let Some(path) = locate_seed_config() else {
        eprintln!("skipped: no naked.json reachable (CI without state/)");
        return;
    };
    let text = fs::read_to_string(path).unwrap();
    let cfg: Config = serde_json::from_str(&text).unwrap();
    let mut unknowns = Vec::new();
    for (pname, pcfg) in &cfg.providers {
        for m in &pcfg.models {
            if !pcfg.capabilities.contains_key(m) {
                unknowns.push(format!("{pname}/{m}"));
            }
        }
    }
    // We seeded 95 entries — at most a handful of aliases / edge models
    // should be missing. If this trips above ~5, the seed script needs
    // an update.
    assert!(
        unknowns.len() <= 5,
        "too many uncatalogued models: {unknowns:?}"
    );
}
