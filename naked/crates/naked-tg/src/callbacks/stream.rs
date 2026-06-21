use super::*;

fn callback_message_key(q: &CallbackQuery) -> Option<naked_tg::run_registry::MessageKey> {
    let msg = q.message.as_ref()?;
    Some(naked_tg::run_registry::MessageKey::new(
        msg.chat().id.0,
        msg.id().0,
    ))
}

fn moved_notice_text(q: &CallbackQuery, encoded_run_id: Option<&str>) -> Option<String> {
    let key = callback_message_key(q)?;
    let notice = crate::shared::RUN_REGISTRY.moved_notice_for_message(key, encoded_run_id);
    notice.map(|notice| {
        format!(
            "moved to chat {}{}",
            notice.new_origin.chat_id,
            notice
                .new_origin
                .thread_id
                .map(|tid| format!(" / thread {tid}"))
                .unwrap_or_default()
        )
    })
}

fn resolve_callback_run(
    q: &CallbackQuery,
    encoded_run_id: Option<&str>,
) -> Option<naked_tg::run_registry::RunControl> {
    if let Some(run_id) = encoded_run_id {
        return crate::shared::RUN_REGISTRY.control_for_run(run_id);
    }
    let key = callback_message_key(q)?;
    let run = crate::shared::RUN_REGISTRY.resolve_message(key)?;
    crate::shared::RUN_REGISTRY.control_for_run(&run.run_id)
}

pub(super) async fn handle_stream_action(
    bot: &Bot,
    agent: &Arc<AgentCore>,
    channel_map: &Arc<ChannelSessionMap>,
    q: &CallbackQuery,
    action: &str,
    encoded_run_id: Option<&str>,
) -> Result<(), teloxide::RequestError> {
    let _ = channel_map;
    crate::metrics::record_stream_button_click(action);
    let cb_ctx = ChatCtx::from_callback(q);
    if let Some(text) = moved_notice_text(q, encoded_run_id) {
        bot.answer_callback_query(q.id.clone()).text(text).await?;
        return Ok(());
    }
    let resolved = resolve_callback_run(q, encoded_run_id);
    let Some(run) = resolved else {
        crate::metrics::record_run_registry_callback_expired();
        bot.answer_callback_query(q.id.clone())
            .text("control expired")
            .await?;
        return Ok(());
    };
    crate::metrics::record_run_registry_callback_resolved();

    let toast: &str = match action {
        "abort" => {
            run.abort.cancel();
            agent.abort(&run.session_id).await;
            crate::session_control::clear_control_card_for_run(bot, &run.run_id).await;
            "⏹ Остановлено"
        }
        "sendnow" => handle_send_now(bot, agent, cb_ctx, &run.run_id, &run.session_id).await,
        _ => "…",
    };
    bot.answer_callback_query(q.id.clone()).text(toast).await?;
    Ok(())
}

async fn handle_send_now(
    bot: &Bot,
    agent: &Arc<AgentCore>,
    _cb_ctx: ChatCtx,
    run_id: &str,
    session_id: &str,
) -> &'static str {
    let nudged = {
        let map = crate::shared::STEER_SENDERS.read().await;
        if let Some(steer_tx) = map.get(run_id) {
            steer_tx
                .try_send(naked_core::types::SteerMessage {
                    msg_id: -1,
                    text: super::SEND_NOW_NUDGE_TEXT.into(),
                    is_edit: false,
                })
                .is_ok()
        } else {
            false
        }
    };
    if nudged {
        "⏩ Нудж отправлен — модель завершит с тем, что есть"
    } else {
        agent.abort(session_id).await;
        crate::session_control::clear_control_card_for_run(bot, run_id).await;
        "⏩ Нудж не прошёл — остановил. Отправь сообщение, весь контекст сохранён"
    }
}

#[cfg(test)]
mod tests {
    use std::pin::Pin;
    use std::sync::Arc;

    use async_trait::async_trait;
    use naked_core::config::Config;
    use naked_core::provider::{ChatRequest, Provider};
    use naked_core::types::{ModelInfo, SteerMessage, StreamChunk};
    use teloxide::types::{CallbackQuery, CallbackQueryId, User, UserId};
    use tokio::sync::mpsc;
    use tokio_stream::Stream;
    use tokio_util::sync::CancellationToken;
    use wiremock::matchers::method;
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;
    use crate::callbacks::CallbackAction;

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

    fn callback_query(data: &str) -> CallbackQuery {
        CallbackQuery {
            id: CallbackQueryId(format!("cb-{data}")),
            from: User {
                id: UserId(42),
                is_bot: false,
                first_name: "tester".to_string(),
                last_name: None,
                username: None,
                language_code: None,
                is_premium: false,
                added_to_attachment_menu: false,
            },
            message: None,
            inline_message_id: None,
            chat_instance: "chat-instance".to_string(),
            data: Some(data.to_string()),
            game_short_name: None,
        }
    }

    fn callback_query_with_message(data: &str, chat_id: i64, message_id: i32) -> CallbackQuery {
        serde_json::from_value(serde_json::json!({
            "id": format!("cb-{data}-{chat_id}-{message_id}"),
            "from": {
                "id": 42,
                "is_bot": false,
                "first_name": "tester"
            },
            "chat_instance": "chat-instance",
            "data": data,
            "message": {
                "message_id": message_id,
                "date": 1_700_000_000,
                "chat": {"id": chat_id, "type": "private", "first_name": "u"},
                "text": "button"
            }
        }))
        .expect("valid callback query with regular message")
    }

    fn stream_static_test_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|poison| poison.into_inner())
    }

    fn register_global_run(
        run_id: &str,
        session_id: &str,
        steer: mpsc::Sender<SteerMessage>,
        abort: CancellationToken,
    ) {
        crate::shared::RUN_REGISTRY
            .register_run(
                naked_tg::run_registry::RegisterRunInput {
                    requested_run_id: Some(run_id.to_string()),
                    session_id: session_id.to_string(),
                    origin: naked_tg::run_registry::RunOrigin::new(911, None),
                    kind: naked_tg::run_registry::RunKind::ChatTurn,
                    source_ref: None,
                    steer,
                    abort,
                },
                naked_tg::run_registry::RegisterRunOptions::cap_three(),
            )
            .expect("register global callback test run");
    }

    #[test]
    fn abort_sendnow_steer_target_only_selected_run() {
        let _guard = stream_static_test_lock();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime");
        rt.block_on(async {
            let unique = format!(
                "{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            );
            let run_a = format!("l3-a-{unique}");
            let run_b = format!("l3-b-{unique}");
            let run_c = format!("l3-c-{unique}");
            let sid_a = format!("sid-a-{unique}");
            let sid_b = format!("sid-b-{unique}");
            let sid_c = format!("sid-c-{unique}");
            let (tx_a, mut rx_a) = mpsc::channel::<SteerMessage>(8);
            let (tx_b, mut rx_b) = mpsc::channel::<SteerMessage>(8);
            let (tx_c, mut rx_c) = mpsc::channel::<SteerMessage>(8);
            let abort_a = CancellationToken::new();
            let abort_b = CancellationToken::new();
            let abort_c = CancellationToken::new();
            register_global_run(&run_a, &sid_a, tx_a.clone(), abort_a.clone());
            register_global_run(&run_b, &sid_b, tx_b.clone(), abort_b.clone());
            register_global_run(&run_c, &sid_c, tx_c.clone(), abort_c.clone());
            {
                let mut senders = crate::shared::STEER_SENDERS.write().await;
                senders.insert(run_a.clone(), tx_a.clone());
                senders.insert(run_b.clone(), tx_b.clone());
                senders.insert(run_c.clone(), tx_c.clone());
            }

            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "ok": true,
                    "result": true
                })))
                .mount(&server)
                .await;
            let bot =
                Bot::new("0:TEST_TOKEN").set_api_url(reqwest::Url::parse(&server.uri()).unwrap());
            let agent = test_agent();
            let channel_map = Arc::new(ChannelSessionMap::new());

            let q_sendnow = callback_query(&format!("s:sendnow:{run_a}"));
            handle_stream_action(
                &bot,
                &agent,
                &channel_map,
                &q_sendnow,
                "sendnow",
                Some(&run_a),
            )
            .await
            .expect("sendnow callback");
            let nudge = rx_a.try_recv().expect("sendnow nudges only A");
            assert_eq!(nudge.text, super::super::SEND_NOW_NUDGE_TEXT);
            assert!(rx_b.try_recv().is_err(), "B must not receive A sendnow");
            assert!(rx_c.try_recv().is_err(), "C must not receive A sendnow");
            assert!(!abort_a.is_cancelled(), "sendnow nudge should not abort A");
            assert!(!abort_b.is_cancelled(), "B abort token untouched");
            assert!(!abort_c.is_cancelled(), "C abort token untouched");

            let explicit = crate::shared::RUN_REGISTRY.control_for_run(&run_a).unwrap();
            explicit
                .steer
                .try_send(SteerMessage {
                    msg_id: 7,
                    text: "/research steer A only".to_string(),
                    is_edit: false,
                })
                .expect("explicit selected-run steer");
            let steered = rx_a.try_recv().expect("explicit steer targets only A");
            assert_eq!(steered.text, "/research steer A only");
            assert!(rx_b.try_recv().is_err(), "B must not receive A steer");
            assert!(rx_c.try_recv().is_err(), "C must not receive A steer");

            let q_abort = callback_query(&format!("s:abort:{run_a}"));
            handle_stream_action(&bot, &agent, &channel_map, &q_abort, "abort", Some(&run_a))
                .await
                .expect("abort callback");
            assert!(abort_a.is_cancelled(), "s:abort:<A> cancels A");
            assert!(!abort_b.is_cancelled(), "B must not be aborted");
            assert!(!abort_c.is_cancelled(), "C must not be aborted");

            let received = server.received_requests().await.unwrap();
            assert_eq!(
                received.len(),
                2,
                "sendnow+abort callbacks should answer exactly two callback queries"
            );

            crate::shared::RUN_REGISTRY.remove_run(&run_a);
            crate::shared::RUN_REGISTRY.remove_run(&run_b);
            crate::shared::RUN_REGISTRY.remove_run(&run_c);
            let mut senders = crate::shared::STEER_SENDERS.write().await;
            senders.remove(&run_a);
            senders.remove(&run_b);
            senders.remove(&run_c);
        });
    }

    #[test]
    fn move_b_steer_abort_work_a_click_toasts_moved() {
        let _guard = stream_static_test_lock();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime");
        rt.block_on(async {
            let unique = format!(
                "{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            );
            let run_id = format!("move-cb-{unique}");
            let sid = format!("move-cb-sid-{unique}");
            let (tx, mut rx) = mpsc::channel::<SteerMessage>(8);
            let abort = CancellationToken::new();
            crate::shared::RUN_REGISTRY
                .register_run(
                    naked_tg::run_registry::RegisterRunInput {
                        requested_run_id: Some(run_id.clone()),
                        session_id: sid.clone(),
                        origin: naked_tg::run_registry::RunOrigin::new(930, None),
                        kind: naked_tg::run_registry::RunKind::ChatTurn,
                        source_ref: None,
                        steer: tx.clone(),
                        abort: abort.clone(),
                    },
                    naked_tg::run_registry::RegisterRunOptions::cap_three(),
                )
                .expect("register moved callback run");
            crate::shared::RUN_REGISTRY
                .bind_message(&run_id, naked_tg::run_registry::MessageKey::new(930, 42))
                .expect("bind old A bubble");
            let plan = crate::shared::RUN_REGISTRY
                .move_run(
                    &run_id,
                    naked_tg::run_registry::RunOrigin::new(940, None),
                    naked_tg::run_registry::RegisterRunOptions::cap_three(),
                )
                .expect("move registry state");
            assert_eq!(plan.old_bubble.unwrap().chat_id, 930);
            crate::shared::RUN_REGISTRY
                .bind_message(&run_id, naked_tg::run_registry::MessageKey::new(940, 43))
                .expect("bind new B bubble");
            crate::shared::STEER_SENDERS
                .write()
                .await
                .insert(run_id.clone(), tx.clone());

            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "ok": true,
                    "result": true
                })))
                .mount(&server)
                .await;
            let bot =
                Bot::new("0:TEST_TOKEN").set_api_url(reqwest::Url::parse(&server.uri()).unwrap());
            let agent = test_agent();
            let channel_map = Arc::new(ChannelSessionMap::new());

            let q_b_sendnow = callback_query_with_message("stream:sendnow", 940, 43);
            handle_stream_action(&bot, &agent, &channel_map, &q_b_sendnow, "sendnow", None)
                .await
                .expect("B legacy message-key sendnow resolves moved run");
            let nudge = rx.try_recv().expect("B callback nudges run");
            assert_eq!(nudge.text, super::super::SEND_NOW_NUDGE_TEXT);
            assert!(!abort.is_cancelled(), "sendnow nudge does not abort");

            let q_a_old_abort = callback_query_with_message(&format!("s:abort:{run_id}"), 930, 42);
            handle_stream_action(
                &bot,
                &agent,
                &channel_map,
                &q_a_old_abort,
                "abort",
                Some(&run_id),
            )
            .await
            .expect("old A callback toasts moved");
            assert!(
                !abort.is_cancelled(),
                "old A moved callback must not abort via stale context"
            );
            assert!(
                rx.try_recv().is_err(),
                "old A callback must not steer/nudge"
            );

            let q_b_abort = callback_query_with_message(&format!("s:abort:{run_id}"), 940, 43);
            handle_stream_action(
                &bot,
                &agent,
                &channel_map,
                &q_b_abort,
                "abort",
                Some(&run_id),
            )
            .await
            .expect("B abort works after move");
            assert!(abort.is_cancelled(), "B abort cancels moved run");

            let bodies: Vec<String> = server
                .received_requests()
                .await
                .unwrap()
                .iter()
                .map(|req| String::from_utf8_lossy(&req.body).to_string())
                .collect();
            assert!(
                bodies.iter().any(|body| body.contains("moved to chat 940")),
                "old A callback must answer with moved toast; bodies={bodies:#?}"
            );

            crate::shared::RUN_REGISTRY.remove_run(&run_id);
            crate::shared::STEER_SENDERS.write().await.remove(&run_id);
        });
    }

    #[test]
    fn stream_callbacks_route_through_typed_action() {
        assert_eq!(
            CallbackAction::parse("stream:abort"),
            CallbackAction::Stream { action: "abort" }
        );
        assert_eq!(
            CallbackAction::parse("stream:sendnow"),
            CallbackAction::Stream { action: "sendnow" }
        );
    }

    #[test]
    fn callback_legacy_stream_resolves_by_message_or_expires() {
        assert_eq!(
            CallbackAction::parse("stream:abort"),
            CallbackAction::Stream { action: "abort" }
        );
    }
}
