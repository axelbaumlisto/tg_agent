//! Callback query handler (inline keyboard buttons, permissions).
//!
//! Extracted from main.rs.

mod action;
mod command;
mod error;
mod model;
mod permission;
mod research;
mod stream;

use self::action::CallbackAction;
use self::command::handle_command_callback;
use self::error::handle_error_callback;
use self::model::{
    handle_model_page, handle_model_select, handle_provider_select, handle_reasoning_select,
};
use self::permission::handle_permission_callback;
use self::research::handle_research_callback;
use self::stream::handle_stream_action;
use super::*;

/// Synthetic steer text injected when the user taps `[⏩ Send now]`
/// on a streaming control card. Intent: **clarification signal** —
/// "user is correcting course, pay attention to what they're saying".
/// The model decides how to adapt; if it doesn't pivot, user has
/// manual recourse (`[⏹ Стоп]` button or a fresh follow-up message).
///
/// Kept as a `const` so it's grep-able and tweakable from one place.
/// History:
///   v1 (≤2026-05-13 13:30): prescriptive "don't call tools,
///       summarize" — locked model into specific action.
///   v2 (2026-05-13 14:30): course-change signal — too verbose,
///       still slightly prescriptive ("re-evaluate the plan").
///   v3 (2026-05-13 14:50): minimal clarification signal, caps for
///       emphasis. User instruction:
///       «промт может быть уточняющий, поищи "Пользователь
///        уточняет, обрати внимание на что он пишет"».
pub(crate) const SEND_NOW_NUDGE_TEXT: &str =
    "[⏩ Send now] ПОЛЬЗОВАТЕЛЬ УТОЧНЯЕТ — обрати внимание на то, что он пишет.";

// ── Callback handler (permissions) ──────────────────────────────────────────

pub(crate) async fn handle_callback(
    deps: crate::message_handler::BotDeps,
    q: CallbackQuery,
    pending_perms: PendingPermissions,
) -> Result<(), teloxide::RequestError> {
    let bot = deps.bot.clone();
    let agent = deps.agent.clone();
    let channel_map = deps.channel_map.clone();
    let config = deps.config.clone();
    let data = match &q.data {
        Some(d) => d.clone(),
        None => return Ok(()),
    };

    let action = CallbackAction::parse(&data);
    let prefix = CallbackAction::prefix(&data);
    tracing::info!(callback_data = %data, prefix, "handle_callback");
    let cb_ctx = ChatCtx::from_callback(&q);
    if !crate::shared::is_allowed(cb_ctx.chat_id.0, &config) {
        crate::metrics::record_run_registry_callback_denied();
        tracing::warn!(
            chat_id = cb_ctx.chat_id.0,
            "rejected callback from non-allowed chat"
        );
        bot.answer_callback_query(q.id.clone())
            .text("not allowed")
            .await?;
        return Ok(());
    }

    match action {
        CallbackAction::Permission { call_id, action } => {
            handle_permission_callback(
                &bot,
                &agent,
                &channel_map,
                &q,
                &pending_perms,
                call_id,
                action,
            )
            .await?;
        }
        CallbackAction::Provider {
            name: provider_name,
        } => {
            handle_provider_select(&bot, &agent, &channel_map, &config, &q, &provider_name).await?;
        }
        CallbackAction::Model { name: model_name } => {
            handle_model_select(&bot, &agent, &channel_map, &config, &q, &model_name).await?;
        }
        CallbackAction::Reasoning { level } => {
            handle_reasoning_select(&bot, &agent, &channel_map, &config, &q, level).await?;
        }
        CallbackAction::Research { action, spec_id } => {
            handle_research_callback(&deps, &q, action, spec_id).await?;
        }
        // Status inline buttons: show model/reasoning menu as new message
        CallbackAction::Command { name: sub } => {
            handle_command_callback(&bot, &agent, &channel_map, &config, &q, sub).await?;
        }
        // Model page navigation: mp:<page>
        CallbackAction::ModelPage { page: page_str } => {
            handle_model_page(&bot, &agent, &channel_map, &config, &q, page_str).await?;
        }
        CallbackAction::Stream { action } => {
            handle_stream_action(&bot, &agent, &channel_map, &q, action, None).await?;
        }
        CallbackAction::StreamRun { action, run_id } => {
            handle_stream_action(&bot, &agent, &channel_map, &q, action, Some(run_id)).await?;
        }
        CallbackAction::Error { action } => {
            handle_error_callback(&bot, &q, action).await?;
        }
        _ => {
            bot.answer_callback_query(q.id.clone()).await?;
        }
    }
    Ok(())
}
