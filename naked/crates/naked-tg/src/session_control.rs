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
    use std::pin::Pin;
    use std::sync::Arc;

    use super::{AbortMappedSession, abort_mapped_session, clear_control_card_for_run};
    use async_trait::async_trait;
    use naked_core::AgentCore;
    use naked_core::config::Config;
    use naked_core::provider::{ChatRequest, Provider};
    use naked_core::types::{ModelInfo, StreamChunk};
    use teloxide::prelude::Bot;
    use teloxide::types::{ChatId, MessageId, ThreadId};
    use tokio_stream::Stream;
    use wiremock::matchers::method;
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use crate::ChannelSessionMap;

    /// Minimal provider so we can build a real [`AgentCore`] without any
    /// network or model backend. Mirrors the `NoopProvider` used in the
    /// `callbacks::stream` tests (DRY intent; kept local to avoid exporting a
    /// test-only type across modules).
    struct NoopProvider;

    #[async_trait]
    impl Provider for NoopProvider {
        fn name(&self) -> &str {
            "noop"
        }

        fn models(&self) -> Vec<ModelInfo> {
            vec![ModelInfo {
                provider: "noop".to_string(),
                model_id: "noop-model".to_string(),
                display_name: "noop".to_string(),
            }]
        }

        async fn stream_chat(
            &self,
            _request: ChatRequest,
        ) -> naked_core::error::Result<Pin<Box<dyn Stream<Item = StreamChunk> + Send>>> {
            Ok(Box::pin(tokio_stream::empty()))
        }
    }

    /// Build a real, self-contained [`AgentCore`] backed by [`NoopProvider`].
    fn test_agent() -> Arc<AgentCore> {
        let tmp = tempfile::tempdir().expect("tempdir");
        let config = Config {
            default_provider: "noop".to_string(),
            default_model: "noop-model".to_string(),
            workspace: tmp.path().join("workspace"),
            session_dir: tmp.path().join("sessions"),
            ..Default::default()
        };
        std::fs::create_dir_all(&config.workspace).expect("workspace");
        std::fs::create_dir_all(&config.session_dir).expect("sessions");
        let core = Arc::new(AgentCore::new(config, Box::new(NoopProvider)));
        core.init_self_ref();
        core
    }

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

    /// Guards the mapped dispatch branch of [`abort_mapped_session`]: a
    /// (chat, thread) that resolves to a session id in the `ChannelSessionMap`
    /// must return `Aborted { session_id }` carrying that exact id. Catches
    /// arm-swap mutations (mapped arm returning `NoMappedSession`) and
    /// wrong-session-id mutations. Only the `aborted()` predicate was tested
    /// before, leaving this dispatch logic unguarded
    /// (BUG_REGISTRY ABORT-MAPPED-SESSION).
    #[tokio::test]
    async fn abort_mapped_session_returns_aborted_with_mapped_id() {
        let agent = test_agent();
        let channel_map = Arc::new(ChannelSessionMap::new());
        // chat 4242, thread 5 -> "sess-id". `set` takes the raw i64/i32 keys,
        // matching the decomposition `abort_mapped_session` performs on the
        // teloxide newtypes (`ChatId(i64)`, `ThreadId(MessageId(i32))`).
        channel_map.set(4242, Some(5), "sess-id".to_string()).await;

        let outcome = abort_mapped_session(
            &agent,
            &channel_map,
            ChatId(4242),
            Some(ThreadId(MessageId(5))),
        )
        .await;

        assert_eq!(
            outcome,
            AbortMappedSession::Aborted {
                session_id: "sess-id".to_string(),
            },
            "a mapped (chat, thread) must abort and return its session id"
        );
    }

    /// Guards the unmapped dispatch branch of [`abort_mapped_session`]: a
    /// (chat, thread) with no entry in the `ChannelSessionMap` must return
    /// `NoMappedSession` and perform no abort. Catches arm-swap mutations
    /// (unmapped arm returning `Aborted`).
    #[tokio::test]
    async fn abort_mapped_session_returns_no_mapped_session_when_absent() {
        let agent = test_agent();
        let channel_map = Arc::new(ChannelSessionMap::new());

        let outcome = abort_mapped_session(&agent, &channel_map, ChatId(9999), None).await;

        assert_eq!(
            outcome,
            AbortMappedSession::NoMappedSession,
            "an unmapped (chat, thread) must not abort and must report no session"
        );
    }

    /// Unique key per test invocation. `CONTROL_CARDS` is a process-global
    /// static shared across every test in this binary, so we never assert on
    /// the map wholesale — only on our own run id — and we make that run id
    /// unique to avoid colliding with any concurrent test that also touches
    /// the static. A bare atomic counter is sufficient within a process.
    fn unique_run_id() -> String {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        format!(
            "session-card-cleanup-test-{}",
            COUNTER.fetch_add(1, Ordering::Relaxed)
        )
    }

    /// Guards the side-effects of [`clear_control_card_for_run`]: it must both
    /// (a) drop the run's entry from `CONTROL_CARDS` and (b) issue exactly one
    /// `editMessageReplyMarkup` call to clear the inline keyboard. Only the
    /// `AbortMappedSession::aborted` predicate was tested before, leaving this
    /// Telegram-facing cleanup unguarded (BUG_REGISTRY SESSION-CARD-CLEANUP).
    #[tokio::test]
    async fn clear_control_card_removes_entry_and_clears_keyboard() {
        let run_id = unique_run_id();

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "ok": true,
                "result": true
            })))
            .mount(&server)
            .await;
        let bot = Bot::new("0:TEST_TOKEN").set_api_url(reqwest::Url::parse(&server.uri()).unwrap());

        // Seed our own unique entry into the process-global static.
        crate::shared::CONTROL_CARDS
            .write()
            .await
            .insert(run_id.clone(), (ChatId(4242), MessageId(77)));

        clear_control_card_for_run(&bot, &run_id).await;

        // (a) our entry was removed (assert only on our key, never the map size).
        assert!(
            !crate::shared::CONTROL_CARDS
                .read()
                .await
                .contains_key(&run_id),
            "clear_control_card_for_run must remove the run's CONTROL_CARDS entry"
        );

        // (b) exactly one editMessageReplyMarkup call hit the mock (isolated
        // per-test MockServer, so this count is not affected by other tests).
        let received = server.received_requests().await.unwrap();
        assert_eq!(
            received.len(),
            1,
            "clear_control_card_for_run must issue exactly one Telegram call, got {}",
            received.len()
        );
        assert!(
            received[0]
                .url
                .path()
                .to_ascii_lowercase()
                .ends_with("editmessagereplymarkup"),
            "expected an editMessageReplyMarkup call, got path {}",
            received[0].url.path()
        );
    }
}
