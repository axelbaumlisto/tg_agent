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

use std::path::PathBuf;

use naked_core::config::Config;
use naked_core::model_catalog::exporter;

fn workspace_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().unwrap().parent().unwrap().to_path_buf()
}

fn committed_naked_json() -> PathBuf {
    workspace_root().join("naked.json")
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
