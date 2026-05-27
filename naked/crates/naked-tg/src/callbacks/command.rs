use super::*;

pub(super) async fn handle_command_callback(
    bot: &Bot,
    agent: &Arc<AgentCore>,
    channel_map: &Arc<ChannelSessionMap>,
    config: &Config,
    q: &CallbackQuery,
    sub: &str,
) -> Result<(), teloxide::RequestError> {
    let cb_ctx = ChatCtx::from_callback(q);
    let sid = get_or_create_session(cb_ctx, agent, channel_map, config).await;
    match sub {
        "model" => show_model_menu(bot, agent, config, cb_ctx, &sid).await?,
        "reasoning" => show_reasoning_menu(bot, agent, cb_ctx, &sid).await?,
        _ => {}
    }
    bot.answer_callback_query(q.id.clone()).await?;
    Ok(())
}

async fn show_model_menu(
    bot: &Bot,
    agent: &Arc<AgentCore>,
    config: &Config,
    cb_ctx: ChatCtx,
    sid: &str,
) -> Result<(), teloxide::RequestError> {
    let (prov, current_model) = agent.session_provider_model(sid).await;
    let models = provider_models(config, &prov);
    if !models.is_empty() {
        let rows = model_rows(&models, &current_model);
        let kb = InlineKeyboardMarkup::new(rows);
        reply_html_kb(
            bot,
            &cb_ctx,
            format!("Provider: <b>{prov}</b>\nSelect model:"),
            kb,
        )
        .await?;
    }
    Ok(())
}

fn model_rows(models: &[String], current_model: &str) -> Vec<Vec<InlineKeyboardButton>> {
    models
        .iter()
        .map(|m| {
            let mark = if m == current_model { " ✅" } else { "" };
            vec![InlineKeyboardButton::callback(
                format!("{m}{mark}"),
                format!("sm:{m}"),
            )]
        })
        .collect()
}

async fn show_reasoning_menu(
    bot: &Bot,
    agent: &Arc<AgentCore>,
    cb_ctx: ChatCtx,
    sid: &str,
) -> Result<(), teloxide::RequestError> {
    let current = agent.session_reasoning(sid).await;
    let current_level = current.as_deref().unwrap_or("off");
    let kb = InlineKeyboardMarkup::new(reasoning_rows(current_level));
    reply_html_kb(
        bot,
        &cb_ctx,
        format!("💭 Reasoning: <b>{current_level}</b>"),
        kb,
    )
    .await?;
    Ok(())
}

fn reasoning_rows(current_level: &str) -> Vec<Vec<InlineKeyboardButton>> {
    ["off", "low", "medium", "high"]
        .iter()
        .map(|lvl| {
            let mark = if *lvl == current_level { " ✅" } else { "" };
            vec![InlineKeyboardButton::callback(
                format!("{lvl}{mark}"),
                format!("sr:{lvl}"),
            )]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{model_rows, reasoning_rows};
    use crate::callbacks::CallbackAction;

    #[test]
    fn command_callbacks_route_through_typed_action() {
        assert_eq!(
            CallbackAction::parse("cmd:model"),
            CallbackAction::Command { name: "model" }
        );
        assert_eq!(
            CallbackAction::parse("cmd:reasoning"),
            CallbackAction::Command { name: "reasoning" }
        );
    }

    #[test]
    fn model_rows_mark_current_model() {
        let rows = model_rows(&["a".to_string(), "b".to_string()], "b");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0][0].text, "a");
        assert_eq!(rows[1][0].text, "b ✅");
    }

    #[test]
    fn reasoning_rows_mark_current_level() {
        let rows = reasoning_rows("medium");
        let labels: Vec<&str> = rows.iter().map(|row| row[0].text.as_str()).collect();
        assert_eq!(labels, vec!["off", "low", "medium ✅", "high"]);
    }
}
