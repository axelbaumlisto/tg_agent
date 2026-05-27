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
        handle_yolo(bot, agent, channel_map, q, pending_perms).await
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
) -> Result<(), teloxide::RequestError> {
    let cb_ctx = ChatCtx::from_callback(q);
    let cid = cb_ctx.chat_id.0;
    let tid = cb_ctx.raw_thread_id();
    tracing::info!(cid, ?tid, "yolo callback: enabling");
    channel_map.enable_yolo(cid, tid).await;
    let yolo_ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    if let Some(sid) = channel_map.get(cid, tid).await
        && let Err(e) = agent.set_session_yolo(&sid, Some(yolo_ts)).await
    {
        tracing::warn!("failed to persist yolo: {e}");
    }

    let n = approve_matching_permissions(pending_perms, cid, tid).await;
    delete_callback_message(bot, q).await;
    let remaining_h = channel_map.yolo_remaining_secs(cid, tid).await / 3600;
    bot.answer_callback_query(q.id.clone())
        .text(format!("⚡ YOLO ON ({remaining_h}h) — {n} approved"))
        .await?;
    Ok(())
}

async fn approve_matching_permissions(
    pending_perms: &PendingPermissions,
    cid: i64,
    tid: Option<i32>,
) -> usize {
    let mut perms = pending_perms.write().await;
    let matching: Vec<String> = perms
        .iter()
        .filter(|(_, (_, c, t))| *c == cid && *t == tid)
        .map(|(k, _)| k.clone())
        .collect();
    let n = matching.len();
    for key in matching {
        if let Some((tx, _, _)) = perms.remove(&key) {
            let _ = tx.send(true);
        }
    }
    n
}

async fn handle_single_permission(
    bot: &Bot,
    q: &CallbackQuery,
    pending_perms: &PendingPermissions,
    call_id: &str,
    action: &str,
) -> Result<(), teloxide::RequestError> {
    let allowed = action == "allow";
    let sender = pending_perms.write().await.remove(call_id);
    if let Some((tx, _, _)) = sender {
        let _ = tx.send(allowed);
    }
    let label = if allowed { "✅" } else { "❌ Denied" };
    delete_callback_message(bot, q).await;
    bot.answer_callback_query(q.id.clone()).text(label).await?;
    Ok(())
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
    use crate::callbacks::CallbackAction;

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
