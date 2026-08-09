use super::*;

pub(super) async fn handle_permission_callback(
    bot: &Bot,
    agent: &Arc<AgentCore>,
    channel_map: &Arc<ChannelSessionMap>,
    q: &CallbackQuery,
    pending_perms: &PendingPermissions,
    call_id: &str,
    action: &str,
) -> Result<(), teloxide::RequestError> {
    if action == "yolo" {
        handle_yolo(bot, agent, channel_map, q, pending_perms, call_id).await
    } else {
        handle_single_permission(bot, q, pending_perms, call_id, action).await
    }
}

async fn handle_yolo(
    bot: &Bot,
    agent: &Arc<AgentCore>,
    channel_map: &Arc<ChannelSessionMap>,
    q: &CallbackQuery,
    pending_perms: &PendingPermissions,
    call_id: &str,
) -> Result<(), teloxide::RequestError> {
    let cb_ctx = ChatCtx::from_callback(q);
    let cid = cb_ctx.chat_id.0;
    let tid = cb_ctx.raw_thread_id();

    // Fix 2/4 (SECURITY): only a LIVE pending permission for this chat/topic may
    // drive an escalation, and the check-and-claim is ATOMIC. A stale/expired
    // `p:<id>:yolo` tap (card already answered, belonging to another topic, or
    // removed by a concurrent deny/allow/timeout) must NOT enable or count
    // toward permanent — otherwise a replayed/raced tap could silently push the
    // chat to a permanent chat-wide auto-approve with no active card.
    let Some(esc) = escalate_if_live(channel_map, pending_perms, cid, tid, call_id).await else {
        bot.answer_callback_query(q.id.clone())
            .text("⚠️ Запрос устарел — карточка недействительна.")
            .await?;
        return Ok(());
    };
    tracing::info!(cid, ?tid, count = esc.count, "yolo callback: enabling");
    let yolo_ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    if let Some(sid) = channel_map.get(cid, tid).await
        && let Err(e) = agent.set_session_yolo(&sid, Some(yolo_ts)).await
    {
        let safe = redact_for_log(&e);
        tracing::warn!("failed to persist yolo: {safe}");
    }

    // Fix 3: a permanent escalation approves the WHOLE chat (all topics); a
    // temporary enable approves only the current topic. The triggering card was
    // already approved (exactly once) as part of the atomic claim in
    // `escalate_if_live`, so `approve_pending_perms` drains only the REMAINING
    // matching requests; the claimed one is added back into the reported total.
    let permanent = matches!(esc.tier, YoloTier::Permanent);
    let n = approve_pending_perms(pending_perms, cid, tid, permanent).await + 1;
    // Fix 2: persist immediately on permanent so a restart before the periodic
    // flush can't lose the chat-wide grant.
    persist_permanent(channel_map, &esc.tier).await;
    delete_callback_message(bot, q).await;
    bot.answer_callback_query(q.id.clone())
        .text(esc.toast(n))
        .await?;
    Ok(())
}

/// Atomically CLAIM the callback `call_id`, then escalate. Returns `None` (no
/// state change) when the `call_id` is not a live pending permission for this
/// chat/topic (stale/expired/raced), so the caller can answer "expired" without
/// enabling. A testable seam for the security guard (Fix 2/4): the claim removes
/// the perm under a SINGLE write lock so a concurrent deny/allow/timeout cannot
/// slip in between the liveness check and the enable; a failed claim leaves the
/// escalation count untouched.
async fn escalate_if_live(
    channel_map: &ChannelSessionMap,
    pending_perms: &PendingPermissions,
    cid: i64,
    tid: Option<i32>,
    call_id: &str,
) -> Option<YoloEscalation> {
    // Atomic claim under a SINGLE write lock (removes + approves the triggering
    // perm iff it is live for this chat/topic). The lock is dropped BEFORE the
    // channel_map await below — never held across it.
    let claimed = claim_permission_decision(pending_perms, call_id, cid, tid, true).await;
    if claimed != PendingPermClaim::Delivered {
        return None;
    }
    // Dedup on the permission `call_id`: tapping the same card twice must count
    // once toward escalation (a fresh CallbackQuery id would not).
    Some(channel_map.enable_yolo(cid, tid, Some(call_id)).await)
}

async fn handle_single_permission(
    bot: &Bot,
    q: &CallbackQuery,
    pending_perms: &PendingPermissions,
    call_id: &str,
    action: &str,
) -> Result<(), teloxide::RequestError> {
    let allowed = action == "allow";
    let cb_ctx = ChatCtx::from_callback(q);
    let cid = cb_ctx.chat_id.0;
    let tid = cb_ctx.raw_thread_id();
    let claim = claim_permission_decision(pending_perms, call_id, cid, tid, allowed).await;
    match claim {
        PendingPermClaim::Delivered => {
            delete_callback_message(bot, q).await;
            bot.answer_callback_query(q.id.clone())
                .text(permission_decision_toast(allowed, claim))
                .await?;
        }
        PendingPermClaim::NotFoundOrMismatched => {
            bot.answer_callback_query(q.id.clone())
                .text(permission_decision_toast(allowed, claim))
                .await?;
        }
        PendingPermClaim::ReceiverGone => {
            delete_callback_message(bot, q).await;
            bot.answer_callback_query(q.id.clone())
                .text(permission_decision_toast(allowed, claim))
                .await?;
        }
    }
    Ok(())
}

async fn claim_permission_decision(
    pending_perms: &PendingPermissions,
    call_id: &str,
    cid: i64,
    tid: Option<i32>,
    allowed: bool,
) -> PendingPermClaim {
    let mut perms = pending_perms.write().await;
    claim_pending_perm(&mut perms, call_id, cid, tid, allowed)
}

fn permission_decision_toast(allowed: bool, claim: PendingPermClaim) -> &'static str {
    match claim {
        PendingPermClaim::Delivered if allowed => "✅",
        PendingPermClaim::Delivered => "❌ Denied",
        PendingPermClaim::NotFoundOrMismatched => {
            "⚠️ Запрос устарел или из другого чата — карточка недействительна."
        }
        PendingPermClaim::ReceiverGone => "⚠️ Решение не доставлено: ход уже завершён.",
    }
}

async fn delete_callback_message(bot: &Bot, q: &CallbackQuery) {
    if let Some(msg) = &q.message
        && let Some(regular) = msg.regular_message()
    {
        let _ = bot.delete_message(regular.chat.id, regular.id).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::callbacks::CallbackAction;

    #[tokio::test]
    async fn yolo_callback_claims_perm_atomically_then_escalates() {
        // Fix 2 (a)+(c): a live yolo tap CLAIMS its perm under a single write
        // lock — removing it AND approving it exactly once — before enabling.
        // Mutation target: reverting to check-then-enable leaves the perm in the
        // map (and unsent), which the removed/approved asserts below catch.
        let map = ChannelSessionMap::new();
        let pending: PendingPermissions = Arc::new(RwLock::new(HashMap::new()));
        let (tx, rx) = tokio::sync::oneshot::channel::<bool>();
        pending
            .write()
            .await
            .insert("live".into(), (tx, 100, Some(7)));

        let esc = escalate_if_live(&map, &pending, 100, Some(7), "live")
            .await
            .expect("live tap escalates");
        assert_eq!(esc.count, 1);
        assert!(matches!(esc.tier, YoloTier::Temporary { .. }));
        // 1st enable is topic-scoped, not chat-wide permanent.
        assert!(map.is_yolo(100, Some(7)).await, "enable happened");
        assert!(
            !map.is_yolo(100, Some(999)).await,
            "1st enable is topic-scoped, not permanent chat-wide"
        );
        // The claim removed the perm from the map (check-then-enable would leave
        // it), so the follow-up drain can't re-approve it.
        assert!(
            !pending.read().await.contains_key("live"),
            "claim removed the perm from the pending map"
        );
        // ...and approved it exactly once.
        assert_eq!(rx.await, Ok(true), "claimed perm approved exactly once");
        // (c) The follow-up chat/topic drain finds nothing left — no double send.
        assert_eq!(
            approve_pending_perms(&pending, 100, Some(7), false).await,
            0,
            "claimed perm already drained; no double approve"
        );
    }

    #[tokio::test]
    async fn stale_or_mismatched_callback_does_not_escalate() {
        // Fix 2 (b) / Fix 4: a `p:<id>:yolo` tap whose call_id is not a live
        // pending perm for this chat/topic (already-removed by a concurrent
        // deny/allow/timeout, or a topic mismatch) must FAIL the claim — no
        // enable, no escalate, count unchanged — and any unrelated pending perm
        // is left intact (ownership verified before remove).
        let map = ChannelSessionMap::new();
        let pending: PendingPermissions = Arc::new(RwLock::new(HashMap::new()));

        // Already-removed / unknown call_id → claim fails, no grant. A wrong
        // enable would grant the topic AND (if it escalated) make the chat
        // permanent chat-wide; neither must happen.
        assert!(
            escalate_if_live(&map, &pending, 100, Some(7), "gone")
                .await
                .is_none()
        );
        assert!(!map.is_yolo(100, Some(7)).await, "stale tap must not grant");
        assert!(
            !map.is_yolo(100, Some(999)).await,
            "stale tap must not escalate the chat to permanent"
        );

        // A live perm for topic 7, but a tap claiming topic 8 → claim rejected
        // and the topic-7 perm is left untouched (not drained/sent).
        let (tx, mut rx) = tokio::sync::oneshot::channel::<bool>();
        pending
            .write()
            .await
            .insert("live".into(), (tx, 100, Some(7)));
        assert!(
            escalate_if_live(&map, &pending, 100, Some(8), "live")
                .await
                .is_none()
        );
        assert!(
            !map.is_yolo(100, Some(8)).await,
            "topic-mismatch must not grant"
        );
        assert!(!map.is_yolo(100, Some(7)).await);
        assert!(
            rx.try_recv().is_err(),
            "mismatched claim must not send/drain"
        );
        assert!(
            pending.read().await.contains_key("live"),
            "mismatched claim leaves the perm pending"
        );
    }

    #[tokio::test]
    async fn single_permission_claim_allows_denies_rejects_foreign_and_reports_dead_receiver() {
        let pending: PendingPermissions = Arc::new(RwLock::new(HashMap::new()));

        let (allow_tx, allow_rx) = tokio::sync::oneshot::channel::<bool>();
        pending
            .write()
            .await
            .insert("allow".into(), (allow_tx, 10, Some(1)));
        assert_eq!(
            claim_permission_decision(&pending, "allow", 10, Some(1), true).await,
            PendingPermClaim::Delivered,
            "exact chat/topic allow is delivered"
        );
        assert_eq!(allow_rx.await, Ok(true));

        let (deny_tx, deny_rx) = tokio::sync::oneshot::channel::<bool>();
        pending
            .write()
            .await
            .insert("deny".into(), (deny_tx, 10, Some(1)));
        assert_eq!(
            claim_permission_decision(&pending, "deny", 10, Some(1), false).await,
            PendingPermClaim::Delivered,
            "exact chat/topic deny is delivered"
        );
        assert_eq!(deny_rx.await, Ok(false));

        let (tx, mut rx) = tokio::sync::oneshot::channel::<bool>();
        pending
            .write()
            .await
            .insert("foreign".into(), (tx, 10, Some(1)));
        assert_eq!(
            claim_permission_decision(&pending, "foreign", 11, Some(1), true).await,
            PendingPermClaim::NotFoundOrMismatched,
            "wrong chat is stale/foreign"
        );
        assert_eq!(
            claim_permission_decision(&pending, "foreign", 10, Some(2), true).await,
            PendingPermClaim::NotFoundOrMismatched,
            "wrong topic is stale/foreign"
        );
        assert_eq!(
            claim_permission_decision(&pending, "missing", 10, Some(1), true).await,
            PendingPermClaim::NotFoundOrMismatched,
            "unknown call_id is stale"
        );
        assert!(
            pending.read().await.contains_key("foreign"),
            "foreign/stale claims must leave the real owner's entry pending"
        );
        assert!(rx.try_recv().is_err(), "foreign/stale claims send nothing");

        let (dead_tx, dead_rx) = tokio::sync::oneshot::channel::<bool>();
        drop(dead_rx);
        pending
            .write()
            .await
            .insert("dead".into(), (dead_tx, 10, Some(1)));
        assert_eq!(
            claim_permission_decision(&pending, "dead", 10, Some(1), true).await,
            PendingPermClaim::ReceiverGone,
            "dead receiver is not reported as success"
        );
        assert!(
            !pending.read().await.contains_key("dead"),
            "dead receiver entry is removed because no live turn can answer it"
        );
    }

    #[test]
    fn single_permission_toasts_distinguish_stale_foreign_and_dead_receiver() {
        assert_eq!(
            permission_decision_toast(true, PendingPermClaim::Delivered),
            "✅"
        );
        assert_eq!(
            permission_decision_toast(false, PendingPermClaim::Delivered),
            "❌ Denied"
        );
        assert!(
            permission_decision_toast(true, PendingPermClaim::NotFoundOrMismatched)
                .contains("из другого чата"),
            "foreign/stale taps get an explicit stale/foreign toast"
        );
        let dead = permission_decision_toast(true, PendingPermClaim::ReceiverGone);
        assert!(
            dead.contains("не доставлено"),
            "dead receiver toast: {dead}"
        );
        assert!(!dead.contains('✅'), "dead receiver must not show success");
    }

    #[test]
    fn permission_callbacks_route_through_typed_action() {
        assert_eq!(
            CallbackAction::parse("p:abc:allow"),
            CallbackAction::Permission {
                call_id: "abc",
                action: "allow",
            }
        );
        assert_eq!(
            CallbackAction::parse("p:abc:yolo"),
            CallbackAction::Permission {
                call_id: "abc",
                action: "yolo",
            }
        );
    }
}
