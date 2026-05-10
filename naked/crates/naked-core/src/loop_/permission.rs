//! Approval flow for gated tool calls.
//!
//! Extracted from the iteration body of [`super::AgentLoop::run`] (T9 of
//! PLAN_CORE_HARDENING_v2): a 35-line flow with 8 indent levels was the
//! deepest nesting in `loop_/run.rs`. Pulling the cached/prompt branches
//! into a single async helper keeps `run` linear and makes the flow
//! testable in isolation.

use tokio::sync::mpsc;

use crate::tool::approval_cache::ApprovalCache;
use crate::types::{AgentEvent, Permission, PermissionResponse};

/// Decide whether a single gated tool call is allowed to execute.
///
/// Resolution order:
/// 1. **Approval cache hit** — emits a `ToolOutput` heartbeat with the
///    fingerprint and returns `true`.
/// 2. **`permission_rx` present** — sends a `PermissionRequest` event
///    and awaits a matching `PermissionResponse`. On approval the
///    fingerprint is cached for future calls.
/// 3. **`permission_rx == None`** — caller is in non-interactive mode
///    (CLI tests, headless research). Approve by default.
///
/// Returns the boolean decision; the caller is responsible for the
/// "denied" event/history bookkeeping (kept there because it depends
/// on caller-owned `name`/`history` mutability).
///
/// 8 arguments is the natural shape of "check this fingerprint, prompt
/// the user if needed, emit events, return decision". Wrapping in a
/// struct only adds noise; the call site is one line.
#[allow(clippy::too_many_arguments)]
pub(super) async fn request_or_cached_approval(
    cache: &ApprovalCache,
    fingerprint: &str,
    permission_rx: &mut Option<mpsc::Receiver<PermissionResponse>>,
    call_id: &str,
    tool_name: &str,
    input: &serde_json::Value,
    perm: Permission,
    tx: &mpsc::Sender<AgentEvent>,
    permissions: Option<&std::sync::Arc<tokio::sync::RwLock<crate::permissions::Ruleset>>>,
) -> bool {
    if cache.is_approved(fingerprint) {
        let _ = tx
            .send(AgentEvent::ToolOutput {
                call_id: call_id.to_string(),
                chunk: format!("\u{2705} auto-approved (cached: {fingerprint})"),
            })
            .await;
        return true;
    }

    // T5 of PLAN_QUALITY_v1: pattern-rule check before UI prompt.
    // Allow → bypass + cache. Deny → reject without prompting.
    // Ask → fall through to existing flow.
    if let Some(rs_lock) = permissions {
        let target = input
            .get("path")
            .and_then(serde_json::Value::as_str)
            .or_else(|| input.get("command").and_then(serde_json::Value::as_str))
            .unwrap_or("");
        let action = rs_lock.read().await.evaluate(tool_name, target);
        match action {
            crate::permissions::Action::Allow => {
                cache.approve(fingerprint);
                let _ = tx
                    .send(AgentEvent::ToolOutput {
                        call_id: call_id.to_string(),
                        chunk: format!("\u{2705} auto-approved by rule for `{tool_name}`"),
                    })
                    .await;
                return true;
            }
            crate::permissions::Action::Deny => {
                let _ = tx
                    .send(AgentEvent::ToolOutput {
                        call_id: call_id.to_string(),
                        chunk: format!("\u{274c} denied by rule for `{tool_name}` on `{target}`"),
                    })
                    .await;
                return false;
            }
            crate::permissions::Action::Ask => {}
        }
    }

    let Some(prx) = permission_rx.as_mut() else {
        // No permission channel wired — non-interactive callers
        // implicitly approve everything (matches pre-extraction
        // behaviour).
        return true;
    };

    let _ = tx
        .send(AgentEvent::PermissionRequest {
            call_id: call_id.to_string(),
            tool_name: tool_name.to_string(),
            input: input.clone(),
            permission: perm,
        })
        .await;
    match prx.recv().await {
        Some(resp) if resp.call_id == call_id => {
            if resp.allowed {
                cache.approve(fingerprint);
            }
            resp.allowed
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tokio::sync::mpsc;

    fn make_channels() -> (
        mpsc::Sender<AgentEvent>,
        mpsc::Receiver<AgentEvent>,
        mpsc::Sender<PermissionResponse>,
        mpsc::Receiver<PermissionResponse>,
    ) {
        let (tx, rx) = mpsc::channel(8);
        let (perm_tx, perm_rx) = mpsc::channel(2);
        (tx, rx, perm_tx, perm_rx)
    }

    #[tokio::test]
    async fn cached_fingerprint_auto_approves() {
        let cache = ApprovalCache::new();
        let fp = "fp-cached";
        cache.approve(fp);
        let (tx, mut rx, _perm_tx, perm_rx) = make_channels();
        let allowed = request_or_cached_approval(
            &cache,
            fp,
            &mut Some(perm_rx),
            "call-1",
            "bash",
            &json!({"cmd":"ls"}),
            Permission::Dangerous,
            &tx,
            None,
        )
        .await;
        assert!(allowed);
        // First event must be the cached ToolOutput heartbeat.
        match rx.recv().await {
            Some(AgentEvent::ToolOutput { chunk, .. }) => {
                assert!(chunk.contains("cached"), "got chunk: {chunk}");
            }
            other => panic!("unexpected first event: {other:?}"),
        }
    }

    #[tokio::test]
    async fn no_permission_channel_implicitly_approves() {
        let cache = ApprovalCache::new();
        let (tx, _rx, _perm_tx, _perm_rx) = make_channels();
        let mut none_rx: Option<mpsc::Receiver<PermissionResponse>> = None;
        let allowed = request_or_cached_approval(
            &cache,
            "fp-x",
            &mut none_rx,
            "c",
            "n",
            &json!({}),
            Permission::Dangerous,
            &tx,
            None,
        )
        .await;
        assert!(allowed);
    }

    #[tokio::test]
    async fn user_approves_then_caches() {
        let cache = ApprovalCache::new();
        let (tx, mut rx, perm_tx, perm_rx) = make_channels();
        let fp = "fp-prompt-yes";

        let mut perm_rx = Some(perm_rx);
        let approver = tokio::spawn(async move {
            // Wait for PermissionRequest then approve.
            while let Some(ev) = rx.recv().await {
                if let AgentEvent::PermissionRequest { call_id, .. } = ev {
                    perm_tx
                        .send(PermissionResponse {
                            call_id,
                            allowed: true,
                        })
                        .await
                        .unwrap();
                    break;
                }
            }
        });
        let allowed = request_or_cached_approval(
            &cache,
            fp,
            &mut perm_rx,
            "call-yes",
            "edit",
            &json!({"file":"a.rs"}),
            Permission::Dangerous,
            &tx,
            None,
        )
        .await;
        approver.await.unwrap();
        assert!(allowed);
        assert!(cache.is_approved(fp), "approved fingerprint must be cached");
    }

    #[tokio::test]
    async fn user_denies_does_not_cache() {
        let cache = ApprovalCache::new();
        let (tx, mut rx, perm_tx, perm_rx) = make_channels();
        let fp = "fp-prompt-no";

        let mut perm_rx = Some(perm_rx);
        let denier = tokio::spawn(async move {
            while let Some(ev) = rx.recv().await {
                if let AgentEvent::PermissionRequest { call_id, .. } = ev {
                    perm_tx
                        .send(PermissionResponse {
                            call_id,
                            allowed: false,
                        })
                        .await
                        .unwrap();
                    break;
                }
            }
        });
        let allowed = request_or_cached_approval(
            &cache,
            fp,
            &mut perm_rx,
            "call-no",
            "bash",
            &json!({}),
            Permission::Dangerous,
            &tx,
            None,
        )
        .await;
        denier.await.unwrap();
        assert!(!allowed);
        assert!(
            !cache.is_approved(fp),
            "denied fingerprint must NOT be cached"
        );
    }

    #[tokio::test]
    async fn mismatched_response_call_id_denies() {
        let cache = ApprovalCache::new();
        let (tx, mut rx, perm_tx, perm_rx) = make_channels();
        let mut perm_rx = Some(perm_rx);

        let prankster = tokio::spawn(async move {
            while let Some(ev) = rx.recv().await {
                if matches!(ev, AgentEvent::PermissionRequest { .. }) {
                    perm_tx
                        .send(PermissionResponse {
                            call_id: "WRONG-ID".into(),
                            allowed: true,
                        })
                        .await
                        .unwrap();
                    break;
                }
            }
        });
        let allowed = request_or_cached_approval(
            &cache,
            "fp",
            &mut perm_rx,
            "expected-id",
            "n",
            &json!({}),
            Permission::Dangerous,
            &tx,
            None,
        )
        .await;
        prankster.await.unwrap();
        assert!(!allowed, "mismatched call_id must be treated as denial");
    }
}
