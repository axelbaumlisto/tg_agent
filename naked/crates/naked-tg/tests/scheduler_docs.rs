//! Snapshot/string tests for scheduler-state documentation.
//!
//! These tests pin the user-visible documentation so any future change
//! to scheduler states (`Scheduled`/`Running`/`Completed`/`Failed`),
//! the `/research state` and `/research reset` operator commands, the
//! `pause_reason` mechanism, or the cooperative `CancellationToken`
//! flow has to be reflected in the docs the operators actually read.
//!
//! If you change one of those user-visible knobs, update the docs and
//! these assertions in the same commit.

use std::path::PathBuf;

fn workspace_doc(rel: &str) -> String {
    // CARGO_MANIFEST_DIR points at crates/naked-tg; docs live two levels up.
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let path = manifest
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join(rel))
        .unwrap_or_else(|| PathBuf::from(rel));
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

#[test]
fn research_system_doc_lists_lifecycle_states() {
    let doc = workspace_doc("docs/research-system.md");

    for state in ["Scheduled", "Running", "Completed", "Failed"] {
        assert!(
            doc.contains(state),
            "docs/research-system.md must mention RunState::{state}"
        );
    }
    assert!(
        doc.contains("inflight.json"),
        "docs must explain the persisted inflight ledger location"
    );
    assert!(
        doc.contains("scheduled_after_resurrection"),
        "docs must explain the resurrection flag so operators can grep for it in metrics"
    );
}

#[test]
fn research_system_doc_documents_state_and_reset_commands() {
    let doc = workspace_doc("docs/research-system.md");

    assert!(
        doc.contains("/research state"),
        "docs must document the /research state operator command"
    );
    assert!(
        doc.contains("/research reset"),
        "docs must document the /research reset operator command"
    );
    assert!(
        doc.contains("render_state"),
        "docs must mention the pure render_state function backing /research state"
    );
}

#[test]
fn research_system_doc_documents_pause_reason() {
    let doc = workspace_doc("docs/research-system.md");

    assert!(
        doc.contains("pause_reason"),
        "docs must mention the pause_reason field added to ResearchSpec"
    );
    assert!(
        doc.contains("auto:"),
        "docs must show the auto-pause reason format so operators can recognise it in /research ls"
    );
}

#[test]
fn research_system_doc_documents_cooperative_cancellation() {
    let doc = workspace_doc("docs/research-system.md");

    assert!(
        doc.contains("CancellationToken"),
        "docs must mention the CancellationToken used for cooperative cancellation"
    );
    assert!(
        doc.contains("cancel_grace_period"),
        "docs must mention the SchedulerConfig::cancel_grace_period knob"
    );
    assert!(
        doc.contains("hard-abort") || doc.contains("hard abort"),
        "docs must explain the hard-abort fallback after the grace period"
    );
    assert!(
        doc.contains("StopReason::Cancelled"),
        "docs must mention StopReason::Cancelled and how the scheduler treats it"
    );
}

#[test]
fn research_system_doc_documents_multi_instance_lock() {
    let doc = workspace_doc("docs/research-system.md");

    assert!(
        doc.contains("scheduler.lock"),
        "docs must mention the scheduler.lock filename so operators can grep for it"
    );
    assert!(
        doc.contains("Multi-instance"),
        "docs must call out the multi-instance safety story explicitly"
    );
    assert!(
        doc.contains("advisory"),
        "docs must clarify that the lock is advisory (not mandatory)"
    );
}

#[test]
fn research_system_doc_documents_inflight_retention_knob() {
    let doc = workspace_doc("docs/research-system.md");

    assert!(
        doc.contains("inflight_terminal_retention"),
        "docs must mention the SchedulerConfig::inflight_terminal_retention knob"
    );
    assert!(
        doc.contains("inflight_purge_interval"),
        "docs must mention the SchedulerConfig::inflight_purge_interval knob"
    );
    assert!(
        doc.contains("purge_terminal_inflight"),
        "docs must reference the trait method backing the periodic purge"
    );
}

#[test]
fn ops_systemd_naked_tg_unit_enables_watchdog() {
    let unit = workspace_doc("ops/systemd/naked-tg.service");

    for needle in [
        "Type=notify",
        "WatchdogSec=",
        "Restart=always",
        "ExecStart=",
    ] {
        assert!(
            unit.contains(needle),
            "ops/systemd/naked-tg.service must contain `{needle}` to wire the systemd watchdog"
        );
    }
}

#[test]
fn ops_systemd_readme_documents_bot_watchdog() {
    let doc = workspace_doc("ops/systemd/README.md");

    for needle in [
        "naked-tg.service",
        "WatchdogSec",
        "SystemdWatchdog",
        "STOPPING=1",
    ] {
        assert!(
            doc.contains(needle),
            "ops/systemd/README.md must explain the bot watchdog (`{needle}` missing)"
        );
    }
}

#[test]
fn naked_readme_documents_systemd_watchdog() {
    let doc = workspace_doc("README.md");

    for needle in ["Type=notify", "WatchdogSec", "SystemdWatchdog"] {
        assert!(
            doc.contains(needle),
            "naked/README.md must document the systemd watchdog (`{needle}` missing)"
        );
    }
}

#[test]
fn naked_readme_documents_scheduler_states() {
    let doc = workspace_doc("README.md");

    assert!(
        doc.contains("Состояния шедулера"),
        "naked/README.md must include the scheduler-states section"
    );
    for needle in [
        "/research state",
        "/research reset",
        "pause_reason",
        "CancellationToken",
        "cancel_grace_period",
        "scheduled_after_resurrection",
    ] {
        assert!(
            doc.contains(needle),
            "naked/README.md must mention `{needle}`"
        );
    }
}
