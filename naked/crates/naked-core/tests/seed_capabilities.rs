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

fn workspace_naked_json() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // crates/naked-core → naked/naked.json
    manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("two parents exist")
        .join("naked.json")
}

#[test]
fn seeded_naked_json_parses() {
    let path = workspace_naked_json();
    let text =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    let _: Config =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("parsing {}: {e}", path.display()));
}

#[test]
fn glm_turbo_is_marked_degraded_with_empty_content() {
    let text = fs::read_to_string(workspace_naked_json()).unwrap();
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
    let text = fs::read_to_string(workspace_naked_json()).unwrap();
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
    let text = fs::read_to_string(workspace_naked_json()).unwrap();
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
    let text = fs::read_to_string(workspace_naked_json()).unwrap();
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
    let text = fs::read_to_string(workspace_naked_json()).unwrap();
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
