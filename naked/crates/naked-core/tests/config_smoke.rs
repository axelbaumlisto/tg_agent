//! BUG_REGISTRY D-INV-WIRING-SMOKE (adapted): shipped example/fixture
//! configs must parse and survive Config::validate_and_warn without networked
//! wiring::build side effects.

use std::path::{Path, PathBuf};

use naked_core::config::Config;

fn first_existing(candidates: &[&str]) -> Option<PathBuf> {
    candidates
        .iter()
        .map(PathBuf::from)
        .find(|path| path.exists())
}

fn repo_config_example() -> PathBuf {
    // `cargo test -p naked-core --test config_smoke` normally runs from
    // `naked/crates/naked-core`, but keep alternate candidates for direct
    // invocations from nearby working directories.
    first_existing(&[
        "../../config.example.json",
        "config.example.json",
        "../config.example.json",
        "../../../config.example.json",
    ])
    .expect("config.example.json must be present in the shipped tree")
}

fn fixture_dir() -> Option<PathBuf> {
    first_existing(&[
        "tests/fixtures",
        "crates/naked-core/tests/fixtures",
        "../naked-core/tests/fixtures",
    ])
}

fn shipped_config_paths() -> Vec<PathBuf> {
    let mut paths = vec![repo_config_example()];

    if let Some(dir) = fixture_dir() {
        let mut fixtures: Vec<PathBuf> = std::fs::read_dir(&dir)
            .unwrap_or_else(|err| panic!("read fixture dir {}: {err}", dir.display()))
            .filter_map(|entry| {
                let path = entry.ok()?.path();
                (path.extension().and_then(|ext| ext.to_str()) == Some("json")).then_some(path)
            })
            .collect();
        fixtures.sort();
        paths.extend(fixtures);
    }

    paths
}

fn smoke_parse_and_validate(path: &Path) {
    let cfg = Config::from_json_file(path)
        .unwrap_or_else(|err| panic!("{} must parse as Config JSON: {err}", path.display()));
    cfg.validate_and_warn();
}

#[test]
fn shipped_example_and_fixture_configs_parse_and_validate() {
    let paths = shipped_config_paths();
    assert!(
        paths
            .iter()
            .any(|path| path.file_name().and_then(|name| name.to_str())
                == Some("config.example.json")),
        "config.example.json must be included in smoke set: {paths:?}"
    );
    assert!(
        paths.iter().any(|path| path.file_name().and_then(|name| name.to_str()) == Some("test_config.json")),
        "tests/fixtures/test_config.json must be included when fixture dir is present: {paths:?}"
    );

    for path in paths {
        smoke_parse_and_validate(&path);
    }
}
