//! Callback query handler (inline keyboard buttons, permissions).
//!
//! Extracted from main.rs.

use super::*;

// ── Callback handler (permissions) ──────────────────────────────────────────

pub(crate) async fn handle_callback(
    bot: Bot,
    q: CallbackQuery,
    pending_perms: PendingPermissions,
    agent: Arc<AgentCore>,
    channel_map: Arc<ChannelSessionMap>,
    config: Config,
) -> Result<(), teloxide::RequestError> {
    let data = match &q.data {
        Some(d) => d.clone(),
        None => return Ok(()),
    };

    let parts: Vec<&str> = data.splitn(3, ':').collect();
    let prefix = parts.first().copied().unwrap_or("");
    tracing::info!(callback_data = %data, prefix, "handle_callback");

    match prefix {
        "p" if parts.len() == 3 => {
            let call_id = parts[1].to_string();
            let action = parts[2];
            if action == "yolo" {
                let cb_ctx = ChatCtx::from_callback(&q);
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
                drop(perms);
                if let Some(msg) = &q.message
                    && let Some(regular) = msg.regular_message()
                {
                    let _ = bot.delete_message(regular.chat.id, regular.id).await;
                }
                let remaining_h = channel_map.yolo_remaining_secs(cid, tid).await / 3600;
                bot.answer_callback_query(q.id.clone())
                    .text(format!("⚡ YOLO ON ({remaining_h}h) — {n} approved"))
                    .await?;
            } else {
                let allowed = action == "allow";
                let sender = pending_perms.write().await.remove(&call_id);
                if let Some((tx, _, _)) = sender {
                    let _ = tx.send(allowed);
                }
                let label = if allowed { "✅" } else { "❌ Denied" };
                if let Some(msg) = &q.message
                    && let Some(regular) = msg.regular_message()
                {
                    let _ = bot.delete_message(regular.chat.id, regular.id).await;
                }
                bot.answer_callback_query(q.id.clone()).text(label).await?;
            }
        }
        "sp" if parts.len() >= 2 => {
            let provider_name = parts[1..].join(":");
            let cb_ctx = ChatCtx::from_callback(&q);
            let sid = get_or_create_session(cb_ctx, &agent, &channel_map, &config).await;
            match agent
                .set_session_provider(&sid, Some(&provider_name), None)
                .await
            {
                Ok(()) => {
                    let (prov, model) = agent.session_provider_model(&sid).await;
                    let models = provider_models(&config, &prov);
                    if models.is_empty() {
                        if let Some(msg) = &q.message
                            && let Some(regular) = msg.regular_message()
                        {
                            let _ = bot
                                .edit_message_text(
                                    regular.chat.id,
                                    regular.id,
                                    format!("✅ {prov}/{model}"),
                                )
                                .await;
                        }
                    } else {
                        let rows: Vec<Vec<InlineKeyboardButton>> = models
                            .iter()
                            .map(|m| {
                                let mark = if *m == model { " ✅" } else { "" };
                                vec![InlineKeyboardButton::callback(
                                    format!("{m}{mark}"),
                                    format!("sm:{m}"),
                                )]
                            })
                            .collect();
                        let kb = InlineKeyboardMarkup::new(rows);
                        if let Some(msg) = &q.message
                            && let Some(regular) = msg.regular_message()
                        {
                            let _ = bot
                                .edit_message_text(
                                    regular.chat.id,
                                    regular.id,
                                    format!("✅ <b>{prov}</b>\nSelect model:"),
                                )
                                .parse_mode(ParseMode::Html)
                                .reply_markup(kb)
                                .await;
                        }
                    }
                    bot.answer_callback_query(q.id.clone())
                        .text(format!("Provider: {prov}"))
                        .await?;
                }
                Err(e) => {
                    bot.answer_callback_query(q.id.clone())
                        .text(format!("Error: {e}"))
                        .await?;
                }
            }
        }
        "sm" if parts.len() >= 2 => {
            let model_name = parts[1..].join(":");
            let cb_ctx = ChatCtx::from_callback(&q);
            let sid = get_or_create_session(cb_ctx, &agent, &channel_map, &config).await;
            match agent
                .set_session_provider(&sid, None, Some(&model_name))
                .await
            {
                Ok(()) => {
                    let (prov, model) = agent.session_provider_model(&sid).await;
                    if let Some(msg) = &q.message
                        && let Some(regular) = msg.regular_message()
                    {
                        let _ = bot
                            .edit_message_text(
                                regular.chat.id,
                                regular.id,
                                format!("✅ <b>{prov}</b> / <b>{model}</b>"),
                            )
                            .parse_mode(ParseMode::Html)
                            .await;
                    }
                    // If agent is busy on this chat, trigger in-flight
                    // model switch: abort current turn and re-dispatch.
                    let chat_key = (cb_ctx.chat_id.0, cb_ctx.raw_thread_id());
                    if agent.is_session_active(&sid).await {
                        let reasoning = agent.session_reasoning(&sid).await;
                        let thinking_suffix = reasoning
                            .filter(|r| r != "off")
                            .map(|r| format!(" Keep the current thinking level ({r}) if the model supports it."))
                            .unwrap_or_default();
                        let switch = naked_tg::model_switch::PendingSwitch {
                            provider: Some(prov.clone()),
                            model: model.clone(),
                            continuation: if thinking_suffix.is_empty() {
                                None
                            } else {
                                Some(format!(
                                    "Continue the previous Telegram request using the newly selected model ({prov}/{model}). \
                                     Resume from the last unfinished step instead of restarting from scratch.{thinking_suffix}"
                                ))
                            },
                        };
                        let map = MODEL_SWITCHES.read().await;
                        if let Some(ms) = map.get(&chat_key) {
                            let continuation = naked_tg::model_switch::build_continuation(&switch);
                            ms.lock().await.request(switch);
                            agent.abort(&sid).await;
                            // Queue continuation so the agent picks up
                            // where it left off with the new model.
                            agent.queue_message(&sid, &continuation).await;
                            bot.answer_callback_query(q.id.clone())
                                .text(format!("⚡ Switching to {model}…"))
                                .await?;
                        } else {
                            drop(map);
                            bot.answer_callback_query(q.id.clone())
                                .text(format!("Model: {model} (next turn)"))
                                .await?;
                        }
                    } else {
                        bot.answer_callback_query(q.id.clone())
                            .text(format!("Model: {model}"))
                            .await?;
                    }
                }
                Err(e) => {
                    bot.answer_callback_query(q.id.clone())
                        .text(format!("Error: {e}"))
                        .await?;
                }
            }
        }
        "sr" if parts.len() >= 2 => {
            let level = parts[1];
            let cb_ctx = ChatCtx::from_callback(&q);
            let sid = get_or_create_session(cb_ctx, &agent, &channel_map, &config).await;
            match agent.set_session_reasoning(&sid, level).await {
                Ok(()) => {
                    let label = if level == "off" { "off" } else { level };
                    if let Some(msg) = &q.message
                        && let Some(regular) = msg.regular_message()
                    {
                        let _ = bot
                            .edit_message_text(
                                regular.chat.id,
                                regular.id,
                                format!("💭 Reasoning: <b>{label}</b>"),
                            )
                            .parse_mode(ParseMode::Html)
                            .await;
                    }
                    bot.answer_callback_query(q.id.clone())
                        .text(format!("Reasoning: {label}"))
                        .await?;
                }
                Err(e) => {
                    bot.answer_callback_query(q.id.clone())
                        .text(format!("Error: {e}"))
                        .await?;
                }
            }
        }
        "r" if parts.len() >= 3 => {
            let action = parts[1];
            let spec_id = parts[2].to_string();
            match action {
                "stop" => {
                    let cancelled = agent.cancel_research_run(&spec_id).await;
                    if let Some(msg) = &q.message
                        && let Some(regular) = msg.regular_message()
                    {
                        let cid_raw = regular.chat.id.0;
                        let tid_raw = regular.thread_id.map(|t| t.0.0);
                        let note = if cancelled {
                            format!(
                                "⏸ Paused <code>{}</code>\n\nReply with a clarification — the next message in this chat will be appended to the spec's topic and a Restart button will appear.",
                                escape_html_min(&spec_id)
                            )
                        } else {
                            format!(
                                "ℹ️ Run <code>{}</code> already finished.\nYou can still type a note and press Restart on the final message.",
                                escape_html_min(&spec_id)
                            )
                        };
                        let _ = bot
                            .edit_message_text(regular.chat.id, regular.id, note)
                            .parse_mode(ParseMode::Html)
                            .reply_markup(keyboard_paused_awaiting_clarification(&spec_id))
                            .await;
                        PENDING_CLARIFICATIONS.write().await.insert(
                            (cid_raw, tid_raw),
                            PendingClarification {
                                spec_id: spec_id.clone(),
                                message_id: regular.id,
                                paused_at: chrono::Utc::now(),
                            },
                        );
                    }
                    bot.answer_callback_query(q.id.clone())
                        .text(if cancelled { "Paused" } else { "Already done" })
                        .await?;
                }
                "restart" => {
                    let (chat_id, thread_id) = match &q.message {
                        Some(msg) => {
                            let cid = msg.chat().id;
                            let tid = msg.regular_message().and_then(|m| m.thread_id);
                            (cid, tid)
                        }
                        None => {
                            bot.answer_callback_query(q.id.clone())
                                .text("missing chat context")
                                .await?;
                            return Ok(());
                        }
                    };
                    let cid_raw = chat_id.0;
                    let tid_raw = thread_id.map(|t| t.0.0);
                    PENDING_CLARIFICATIONS
                        .write()
                        .await
                        .remove(&(cid_raw, tid_raw));
                    bot.answer_callback_query(q.id.clone())
                        .text("Restarting…")
                        .await?;
                    let reply = launch_research_run_with_ui(
                        bot.clone(),
                        agent.clone(),
                        config.clone(),
                        chat_id,
                        thread_id,
                        spec_id,
                    )
                    .await;
                    if !reply.is_empty() {
                        let _ = bot
                            .send_message(chat_id, reply)
                            .maybe_thread(thread_id)
                            .await;
                    }
                }
                _ => {
                    bot.answer_callback_query(q.id.clone())
                        .text(format!("unknown research action: {action}"))
                        .await?;
                }
            }
        }
        // Status inline buttons: show model/reasoning menu as new message
        "cmd" if parts.len() >= 2 => {
            let sub = parts[1];
            let cb_ctx = ChatCtx::from_callback(&q);
            let sid = get_or_create_session(cb_ctx, &agent, &channel_map, &config).await;
            match sub {
                "model" => {
                    let (prov, current_model) = agent.session_provider_model(&sid).await;
                    let models = provider_models(&config, &prov);
                    if !models.is_empty() {
                        let rows: Vec<Vec<InlineKeyboardButton>> = models
                            .iter()
                            .map(|m| {
                                let mark = if *m == current_model { " ✅" } else { "" };
                                vec![InlineKeyboardButton::callback(
                                    format!("{m}{mark}"),
                                    format!("sm:{m}"),
                                )]
                            })
                            .collect();
                        let kb = InlineKeyboardMarkup::new(rows);
                        bot.send_message(
                            cb_ctx.chat_id,
                            format!("Provider: <b>{prov}</b>\nSelect model:",),
                        )
                        .parse_mode(ParseMode::Html)
                        .reply_markup(kb)
                        .maybe_thread(cb_ctx.thread_id)
                        .await?;
                    }
                }
                "reasoning" => {
                    let current = agent.session_reasoning(&sid).await;
                    let current_level = current.as_deref().unwrap_or("off");
                    let levels = ["off", "low", "medium", "high"];
                    let rows: Vec<Vec<InlineKeyboardButton>> = levels
                        .iter()
                        .map(|lvl| {
                            let mark = if *lvl == current_level { " ✅" } else { "" };
                            vec![InlineKeyboardButton::callback(
                                format!("{lvl}{mark}"),
                                format!("sr:{lvl}"),
                            )]
                        })
                        .collect();
                    let kb = InlineKeyboardMarkup::new(rows);
                    bot.send_message(
                        cb_ctx.chat_id,
                        format!("💭 Reasoning: <b>{current_level}</b>"),
                    )
                    .parse_mode(ParseMode::Html)
                    .reply_markup(kb)
                    .maybe_thread(cb_ctx.thread_id)
                    .await?;
                }
                _ => {}
            }
            bot.answer_callback_query(q.id.clone()).await?;
        }
        // Model page navigation: mp:<page>
        "mp" if parts.len() >= 2 => {
            let page_str = parts[1];
            if page_str == "noop" {
                bot.answer_callback_query(q.id.clone()).await?;
            } else if let Ok(page) = page_str.parse::<usize>() {
                let cb_ctx = ChatCtx::from_callback(&q);
                let sid = get_or_create_session(cb_ctx, &agent, &channel_map, &config).await;
                let (prov, current_model) = agent.session_provider_model(&sid).await;
                // Re-render model list at requested page
                if let Some(msg) = &q.message
                    && let Some(regular) = msg.regular_message()
                {
                    let models = agent.provider_models(&prov).await;
                    let scope: Vec<String> =
                        config.model_scope.iter().map(|s| s.to_string()).collect();
                    let filtered = naked_tg::tg_markup::filter_models_by_scope(&models, &scope);
                    let page_size = 8;
                    let total_pages = filtered.len().div_ceil(page_size);
                    let page = page.min(total_pages.saturating_sub(1));
                    let page_items = &filtered
                        [page * page_size..(page * page_size + page_size).min(filtered.len())];

                    let mut rows: Vec<Vec<InlineKeyboardButton>> = page_items
                        .iter()
                        .map(|(p, m)| {
                            let label = if let Some(alias) =
                                config.providers.get(&prov).and_then(|pc| {
                                    pc.model_aliases
                                        .iter()
                                        .find(|(_, v)| {
                                            v.as_str() == format!("{p}/{m}").as_str()
                                                || v.as_str() == m.as_str()
                                        })
                                        .map(|(k, _)| k.clone())
                                }) {
                                alias
                            } else {
                                m.clone()
                            };
                            let mark = if *m == current_model {
                                format!("{label} ✅")
                            } else {
                                label
                            };
                            vec![InlineKeyboardButton::callback(
                                naked_tg::tg_markup::truncate_button(&mark, 56),
                                format!("sm:{m}"),
                            )]
                        })
                        .collect();

                    // Aliases on separate row
                    if let Some(pc) = config.providers.get(&prov) {
                        for alias in pc.model_aliases.keys() {
                            if !page_items.iter().any(|(_, m)| m == alias) {
                                // Only show aliases on first page
                                if page == 0 {
                                    rows.push(vec![InlineKeyboardButton::callback(
                                        alias.clone(),
                                        format!("sm:{alias}"),
                                    )]);
                                }
                            }
                        }
                    }

                    if total_pages > 1 {
                        let mut nav = Vec::new();
                        if page > 0 {
                            nav.push(InlineKeyboardButton::callback(
                                "◀ Prev",
                                format!("mp:{}", page - 1),
                            ));
                        }
                        nav.push(InlineKeyboardButton::callback(
                            format!("{}/{total_pages}", page + 1),
                            "mp:noop".to_string(),
                        ));
                        if page + 1 < total_pages {
                            nav.push(InlineKeyboardButton::callback(
                                "Next ▶",
                                format!("mp:{}", page + 1),
                            ));
                        }
                        rows.push(nav);
                    }

                    let kb = InlineKeyboardMarkup::new(rows);
                    let _ = bot
                        .edit_message_text(
                            regular.chat.id,
                            regular.id,
                            format!("Provider: <b>{prov}</b>\nCurrent: <b>{current_model}</b>"),
                        )
                        .parse_mode(ParseMode::Html)
                        .reply_markup(kb)
                        .await;
                }
                bot.answer_callback_query(q.id.clone()).await?;
            } else {
                bot.answer_callback_query(q.id.clone()).await?;
            }
        }
        _ => {
            bot.answer_callback_query(q.id.clone()).await?;
        }
    }
    Ok(())
}
