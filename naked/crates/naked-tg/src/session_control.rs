use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum AbortMappedSession {
    Aborted { session_id: String },
    NoMappedSession,
}

impl AbortMappedSession {
    pub(crate) fn aborted(&self) -> bool {
        matches!(self, Self::Aborted { .. })
    }
}

pub(crate) async fn abort_mapped_session(
    agent: &Arc<AgentCore>,
    channel_map: &Arc<ChannelSessionMap>,
    chat_id: ChatId,
    thread_id: Option<ThreadId>,
) -> AbortMappedSession {
    let raw_thread_id = thread_id.map(|tid| tid.0.0);
    match channel_map.get(chat_id.0, raw_thread_id).await {
        Some(session_id) => {
            agent.abort(&session_id).await;
            AbortMappedSession::Aborted { session_id }
        }
        None => AbortMappedSession::NoMappedSession,
    }
}

pub(crate) async fn clear_control_card_for_run(bot: &Bot, run_id: &str) {
    if let Some((chat, mid)) = crate::shared::CONTROL_CARDS.write().await.remove(run_id) {
        let _ = bot.edit_message_reply_markup(chat, mid).await;
    }
}

#[cfg(test)]
mod tests {
    use super::AbortMappedSession;

    #[test]
    fn aborted_predicate_is_true_only_for_aborted_outcome() {
        assert!(
            AbortMappedSession::Aborted {
                session_id: "s1".to_string(),
            }
            .aborted()
        );
        assert!(!AbortMappedSession::NoMappedSession.aborted());
    }
}
