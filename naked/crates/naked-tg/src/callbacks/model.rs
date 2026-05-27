use super::*;

pub(super) async fn handle_provider_select(
    bot: &Bot,
    agent: &Arc<AgentCore>,
    channel_map: &Arc<ChannelSessionMap>,
    config: &Config,
    q: &CallbackQuery,
    provider_name: &str,
) -> Result<(), teloxide::RequestError> {
    let cb_ctx = ChatCtx::from_callback(q);
    let sid = get_or_create_session(cb_ctx, agent, channel_map, config).await;
    match agent
        .set_session_provider(&sid, Some(provider_name), None)
        .await
    {
        Ok(()) => {
            let (prov, model) = agent.session_provider_model(&sid).await;
            let models = provider_models(config, &prov);
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
    Ok(())
}

pub(super) async fn handle_model_select(
    bot: &Bot,
    agent: &Arc<AgentCore>,
    channel_map: &Arc<ChannelSessionMap>,
    config: &Config,
    q: &CallbackQuery,
    model_name: &str,
) -> Result<(), teloxide::RequestError> {
    let cb_ctx = ChatCtx::from_callback(q);
    let sid = get_or_create_session(cb_ctx, agent, channel_map, config).await;
    match agent
        .set_session_provider(&sid, None, Some(model_name))
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
            answer_model_select(bot, agent, q, &sid, cb_ctx, &prov, &model).await?;
        }
        Err(e) => {
            bot.answer_callback_query(q.id.clone())
                .text(format!("Error: {e}"))
                .await?;
        }
    }
    Ok(())
}

async fn answer_model_select(
    bot: &Bot,
    agent: &Arc<AgentCore>,
    q: &CallbackQuery,
    sid: &str,
    cb_ctx: ChatCtx,
    prov: &str,
    model: &str,
) -> Result<(), teloxide::RequestError> {
    // If agent is busy on this chat, trigger in-flight model switch:
    // abort current turn and re-dispatch with a continuation.
    let chat_key = (cb_ctx.chat_id.0, cb_ctx.raw_thread_id());
    if agent.is_session_active(sid).await {
        let reasoning = agent.session_reasoning(sid).await;
        let thinking_suffix = reasoning
            .filter(|r| r != "off")
            .map(|r| format!(" Keep the current thinking level ({r}) if the model supports it."))
            .unwrap_or_default();
        let switch = naked_tg::model_switch::PendingSwitch {
            provider: Some(prov.to_string()),
            model: model.to_string(),
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
            agent.abort(sid).await;
            agent.queue_message(sid, &continuation).await;
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
    Ok(())
}

pub(super) async fn handle_reasoning_select(
    bot: &Bot,
    agent: &Arc<AgentCore>,
    channel_map: &Arc<ChannelSessionMap>,
    config: &Config,
    q: &CallbackQuery,
    level: &str,
) -> Result<(), teloxide::RequestError> {
    let cb_ctx = ChatCtx::from_callback(q);
    let sid = get_or_create_session(cb_ctx, agent, channel_map, config).await;
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
    Ok(())
}

pub(super) async fn handle_model_page(
    bot: &Bot,
    agent: &Arc<AgentCore>,
    channel_map: &Arc<ChannelSessionMap>,
    config: &Config,
    q: &CallbackQuery,
    page_str: &str,
) -> Result<(), teloxide::RequestError> {
    if page_str == "noop" {
        bot.answer_callback_query(q.id.clone()).await?;
    } else if let Ok(page) = page_str.parse::<usize>() {
        let cb_ctx = ChatCtx::from_callback(q);
        let sid = get_or_create_session(cb_ctx, agent, channel_map, config).await;
        let (prov, current_model) = agent.session_provider_model(&sid).await;
        if let Some(msg) = &q.message
            && let Some(regular) = msg.regular_message()
        {
            let rows = build_model_keyboard(agent, config, &prov, &current_model, page).await;
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
    Ok(())
}

pub(super) async fn build_model_keyboard(
    agent: &Arc<AgentCore>,
    config: &Config,
    prov: &str,
    current_model: &str,
    page: usize,
) -> Vec<Vec<InlineKeyboardButton>> {
    let models = agent.provider_models(prov);
    let scope: Vec<String> = config.model_scope.iter().map(|s| s.to_string()).collect();
    let filtered = naked_tg::model_glob::filter_models_by_scope(&models, &scope);
    let page_size = 8;
    let total_pages = filtered.len().div_ceil(page_size);
    let page = page.min(total_pages.saturating_sub(1));
    let page_items =
        &filtered[page * page_size..(page * page_size + page_size).min(filtered.len())];

    let mut rows: Vec<Vec<InlineKeyboardButton>> = page_items
        .iter()
        .map(|(p, m)| {
            let label = if let Some(alias) = config.providers.get(prov).and_then(|pc| {
                pc.model_aliases
                    .iter()
                    .find(|(_, v)| {
                        v.as_str() == format!("{p}/{m}").as_str() || v.as_str() == m.as_str()
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
                naked_tg::markup::truncate_button(&mark, 56),
                format!("sm:{m}"),
            )]
        })
        .collect();

    if let Some(pc) = config.providers.get(prov) {
        for alias in pc.model_aliases.keys() {
            if !page_items.iter().any(|(_, m)| m == alias) && page == 0 {
                rows.push(vec![InlineKeyboardButton::callback(
                    alias.clone(),
                    format!("sm:{alias}"),
                )]);
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
        if page + 1 < total_pages {
            nav.push(InlineKeyboardButton::callback(
                "Next ▶",
                format!("mp:{}", page + 1),
            ));
        }
        rows.push(nav);
    }

    rows
}

#[cfg(test)]
mod tests {
    use crate::callbacks::CallbackAction;

    #[test]
    fn model_callbacks_route_through_typed_action() {
        assert_eq!(
            CallbackAction::parse("sp:openrouter"),
            CallbackAction::Provider {
                name: "openrouter".to_string(),
            }
        );
        assert_eq!(
            CallbackAction::parse("sm:gpt-4.1"),
            CallbackAction::Model {
                name: "gpt-4.1".to_string(),
            }
        );
        assert_eq!(
            CallbackAction::parse("sr:high"),
            CallbackAction::Reasoning { level: "high" }
        );
        assert_eq!(
            CallbackAction::parse("mp:2"),
            CallbackAction::ModelPage { page: "2" }
        );
    }
}
