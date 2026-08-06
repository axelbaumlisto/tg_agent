//! /model command handlers.

use super::super::fmt_utils::escape_html_min;
use super::super::*;

#[allow(unused_variables)]
pub(crate) async fn cmd_provider(
    bot: &Bot,
    agent: &std::sync::Arc<AgentCore>,
    channel_map: &std::sync::Arc<ChannelSessionMap>,
    config: &Config,
    ctx: &ChatCtx,
    text: &str,
) -> Result<(), teloxide::RequestError> {
    let chat_id = ctx.chat_id.0;
    let tid = ctx.raw_thread_id();
    let cmd_word = text.split_whitespace().next().unwrap_or("");
    let arg = text[cmd_word.len()..].trim();
    tracing::debug!(chat_id, arg, "cmd /provider");
    if arg.is_empty() {
        let providers = agent.list_providers();
        tracing::debug!(n_providers = providers.len(), "listing providers");
        let (prov, model) = if let Some(sid) = channel_map.get(chat_id, tid).await {
            agent.session_provider_model(&sid).await
        } else {
            (
                config.default_provider.clone(),
                config.default_model.clone(),
            )
        };
        tracing::debug!(%prov, %model, "current provider/model");
        let rows: Vec<Vec<InlineKeyboardButton>> = providers
            .iter()
            .map(|p| {
                let mark = if p.name == prov { " ✅" } else { "" };
                vec![InlineKeyboardButton::callback(
                    format!("{}{mark}", p.name),
                    format!("sp:{}", p.name),
                )]
            })
            .collect();
        let kb = InlineKeyboardMarkup::new(rows);
        match bot
            .send_message(
                ctx.chat_id,
                format!(
                    "Current: <b>{}</b> / <b>{}</b>\n\nSelect provider:",
                    escape_html(&prov),
                    escape_html(&model)
                ),
            )
            .parse_mode(ParseMode::Html)
            .reply_markup(kb)
            .maybe_thread(ctx.thread_id)
            .await
        {
            Ok(m) => tracing::debug!(msg_id = m.id.0, "sent provider keyboard"),
            Err(e) => {
                let safe = redact_for_log(&e);
                tracing::error!("failed to send provider keyboard: {safe}");
                return Err(e);
            }
        }
    } else {
        let sid = get_or_create_session(*ctx, agent, channel_map, config).await;
        match agent.set_session_provider(&sid, Some(arg), None).await {
            Ok(()) => {
                let (prov, model) = agent.session_provider_model(&sid).await;
                reply_text(bot, ctx, format!("Switched to: {prov}/{model}")).await?;
            }
            Err(e) => {
                reply_text(bot, ctx, format!("Error: {e}")).await?;
            }
        }
    }
    Ok(())
}

#[allow(unused_variables)]
pub(crate) async fn cmd_model(
    bot: &Bot,
    agent: &std::sync::Arc<AgentCore>,
    channel_map: &std::sync::Arc<ChannelSessionMap>,
    config: &Config,
    ctx: &ChatCtx,
    text: &str,
) -> Result<(), teloxide::RequestError> {
    let chat_id = ctx.chat_id.0;
    let tid = ctx.raw_thread_id();
    let cmd_word = text.split_whitespace().next().unwrap_or("");
    let arg = text[cmd_word.len()..].trim();
    if arg.is_empty() {
        let (prov, current_model) = if let Some(sid) = channel_map.get(chat_id, tid).await {
            agent.session_provider_model(&sid).await
        } else {
            (
                config.default_provider.clone(),
                config.default_model.clone(),
            )
        };
        // Build model list: scoped (if configured) or all
        let all_models: Vec<(String, String)> = config
            .providers
            .iter()
            .flat_map(|(p, pc)| pc.models.iter().map(move |m| (p.clone(), m.clone())))
            .collect();
        let scope = &config.model_scope;
        let display_models = if scope.is_empty() {
            // No scope: show current provider's models (legacy behavior)
            provider_models(config, &prov)
                .into_iter()
                .map(|m| (prov.clone(), m))
                .collect::<Vec<_>>()
        } else {
            naked_tg::model_glob::filter_models_by_scope(&all_models, scope)
                .into_iter()
                .cloned()
                .collect()
        };

        if display_models.is_empty() {
            reply_text(bot, ctx, "No models match scope.").await?;
        } else {
            const PAGE_SIZE: usize = 8;
            let page = 0usize;
            let total_pages = display_models.len().div_ceil(PAGE_SIZE);
            let page_models = &display_models
                [page * PAGE_SIZE..(page * PAGE_SIZE + PAGE_SIZE).min(display_models.len())];

            let mut rows: Vec<Vec<InlineKeyboardButton>> = page_models
                .iter()
                .map(|(p, m)| {
                    let label = if scope.is_empty() {
                        m.clone()
                    } else {
                        format!("{p}/{m}")
                    };
                    let mark = if *m == current_model && *p == prov {
                        " ✅"
                    } else {
                        ""
                    };
                    vec![InlineKeyboardButton::callback(
                        format!("{label}{mark}"),
                        format!("sm:{m}"),
                    )]
                })
                .collect();

            // Pagination buttons
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
            let header = if scope.is_empty() {
                format!(
                    "Provider: <b>{}</b>\nCurrent: <b>{}</b>",
                    escape_html_min(&prov),
                    escape_html_min(&current_model)
                )
            } else {
                format!(
                    "Current: <b>{}/{}</b>\nShowing scoped models:",
                    escape_html_min(&prov),
                    escape_html_min(&current_model)
                )
            };
            reply_html_kb(bot, ctx, header, kb).await?;
        }
    } else if arg == "auto" {
        let sid = get_or_create_session(*ctx, agent, channel_map, config).await;
        // Set model to "auto" — select_model() will pick per-turn.
        match agent.set_session_provider(&sid, None, Some("auto")).await {
            Ok(()) => {
                reply_text(bot, ctx, "\u{1f916} Auto mode: модель выбирается по задаче").await?;
            }
            Err(e) => {
                reply_text(bot, ctx, format!("Error: {e}")).await?;
            }
        }
    } else {
        let sid = get_or_create_session(*ctx, agent, channel_map, config).await;
        match agent.set_session_provider(&sid, None, Some(arg)).await {
            Ok(()) => {
                let (prov, model) = agent.session_provider_model(&sid).await;
                reply_text(bot, ctx, format!("Model set: {prov}/{model}")).await?;
            }
            Err(e) => {
                reply_text(bot, ctx, format!("Error: {e}")).await?;
            }
        }
    }
    Ok(())
}

#[allow(unused_variables)]
pub(crate) async fn cmd_reasoning(
    bot: &Bot,
    agent: &std::sync::Arc<AgentCore>,
    channel_map: &std::sync::Arc<ChannelSessionMap>,
    config: &Config,
    ctx: &ChatCtx,
    text: &str,
) -> Result<(), teloxide::RequestError> {
    let chat_id = ctx.chat_id.0;
    let tid = ctx.raw_thread_id();
    let cmd_word = text.split_whitespace().next().unwrap_or("");
    let sid = get_or_create_session(*ctx, agent, channel_map, config).await;
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
    reply_html_kb(
        bot,
        ctx,
        format!(
            "💭 Reasoning: <b>{}</b>\n\nSelect level:",
            escape_html(current_level)
        ),
        kb,
    )
    .await?;
    Ok(())
}
