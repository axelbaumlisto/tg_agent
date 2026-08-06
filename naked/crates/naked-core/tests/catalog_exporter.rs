//! Integration tests for the Phase 5 catalog exporter.
//!
//! Exercises the exporter end-to-end against the committed
//! `naked.json`, verifying:
//!
//! 1. The config parses.
//! 2. Rendering produces non-empty generated sections.
//! 3. Splicing those sections into a minimal SKILL.md shell with the
//!    expected comment markers is a byte-stable round-trip (a second
//!    export produces no diff).

use std::collections::HashMap;
use std::path::PathBuf;

use naked_core::config::{Config, ProviderConfig};
use naked_core::model_catalog::exporter;
use naked_core::model_catalog::{
    CostTier, LatencyTier, ModelCapabilities, ModelStatus, QualityTier, TaskKind,
};

fn workspace_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().unwrap().parent().unwrap().to_path_buf()
}

fn committed_naked_json() -> PathBuf {
    workspace_root().join("naked.json")
}

fn marker_shell(quick: &str, details: &str, failures: &str) -> String {
    format!(
        "# header preserved\n\
         <!-- generated:quick-decision:start -->\n{quick}\n<!-- generated:quick-decision:end -->\n\
         middle preserved\n\
         <!-- generated:provider-details:start -->\n{details}\n<!-- generated:provider-details:end -->\n\
         more preserved\n\
         <!-- generated:known-failure-modes:start -->\n{failures}\n<!-- generated:known-failure-modes:end -->\n\
         footer preserved\n"
    )
}

fn config_with_active_model(provider: &str, model: &str) -> Config {
    let mut caps = HashMap::new();
    caps.insert(
        model.to_string(),
        ModelCapabilities {
            status: ModelStatus::Active,
            task_fit: vec![TaskKind::Chat],
            quality_tier: Some(QualityTier::A),
            latency_tier: Some(LatencyTier::Fast),
            cost_tier: Some(CostTier::Cheap),
            ..ModelCapabilities::unknown()
        },
    );

    let mut providers = HashMap::new();
    providers.insert(
        provider.to_string(),
        ProviderConfig {
            provider_type: "openai_compat".into(),
            api_key: "unused".into(),
            capabilities: caps,
            ..ProviderConfig::default()
        },
    );

    Config {
        providers,
        ..Config::default()
    }
}

#[test]
fn renders_from_committed_naked_json() {
    let cfg = Config::from_json_file(&committed_naked_json()).expect("parse naked.json");
    let r = exporter::render_sections(&cfg);
    assert!(
        r.quick_decision.contains("| Job "),
        "quick_decision body must be a markdown table"
    );
    assert!(
        r.provider_details.contains("### "),
        "provider_details must list at least one provider"
    );
    assert!(
        !r.failure_modes.is_empty(),
        "failure_modes must not be empty (at least the header row)"
    );
}

#[test]
fn export_is_idempotent_against_marker_shell() {
    let cfg = Config::from_json_file(&committed_naked_json()).expect("parse naked.json");
    let r = exporter::render_sections(&cfg);

    let shell = "# header preserved\n\
                 <!-- generated:quick-decision:start -->\nplaceholder\n<!-- generated:quick-decision:end -->\n\
                 middle preserved\n\
                 <!-- generated:provider-details:start -->\nplaceholder\n<!-- generated:provider-details:end -->\n\
                 more preserved\n\
                 <!-- generated:known-failure-modes:start -->\nplaceholder\n<!-- generated:known-failure-modes:end -->\n\
                 footer preserved\n";

    let first = r.apply(shell).expect("first apply");
    let second = r.apply(&first).expect("second apply");
    assert_eq!(first, second, "round-trip must be byte-stable");
    assert!(first.contains("# header preserved"));
    assert!(first.contains("footer preserved"));
    assert!(!first.contains("placeholder"));
}

#[test]
fn marker_misuse_surfaces_as_error() {
    let cfg = Config::from_json_file(&committed_naked_json()).expect("parse naked.json");
    let r = exporter::render_sections(&cfg);
    let err = r.apply("nothing in here").unwrap_err();
    assert!(err.contains("missing marker"));
}

#[test]
fn export_to_file_normal_regen_overwrites_generated_sections() {
    let cfg = config_with_active_model("fresh_provider", "fresh-model");
    let original = marker_shell(
        "| **General chat / Q&A** | `stale_provider` / `stale-model` | — | old |",
        "### `stale_provider`\n- `stale-model` — quality=C",
        "| `stale_provider` / `stale-model` | old | old | old |",
    );

    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("SKILL.md");
    std::fs::write(&p, original).unwrap();

    assert!(
        exporter::export_to_file(&cfg, &p).unwrap(),
        "non-empty catalog should rewrite generated sections"
    );
    let updated = std::fs::read_to_string(&p).unwrap();
    assert!(updated.contains("fresh-model"), "{updated}");
    assert!(updated.contains("fresh_provider"), "{updated}");
    assert!(!updated.contains("stale-model"), "{updated}");
    assert!(updated.contains("# header preserved"));
    assert!(updated.contains("footer preserved"));
}

#[test]
fn export_to_file_empty_catalog_preserves_existing_generated_sections() {
    let cfg = Config::default();
    let original = marker_shell(
        "| **General chat / Q&A** | `good_provider` / `good-model` | `backup` / `good-backup` | curated |",
        "### `good_provider`\n- `good-model` — quality=A, fit=[chat]",
        "| `good_provider` / `good-model` | known | cause | fix |",
    );

    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("SKILL.md");
    std::fs::write(&p, &original).unwrap();

    assert!(
        !exporter::export_to_file(&cfg, &p).unwrap(),
        "empty catalog should be treated as an unsafe no-op"
    );
    let preserved = std::fs::read_to_string(&p).unwrap();
    assert_eq!(preserved, original, "good generated data must survive");
}
