//! Slash-command dispatch + shared helpers.
//!
//! Command handlers are split into submodules by domain.

pub(crate) mod info;
pub(crate) mod memory;
pub(crate) mod model;
pub(crate) mod ops;
pub(crate) mod research;
pub(crate) mod session;
pub(crate) mod tools;

use super::*;

/// T3 (PLAN_v13_SOLID_AUDIT): takes `&BotDeps` for shared infra.
pub(crate) async fn handle_command(
    deps: &crate::message_handler::BotDeps,
    _msg: &Message,
    text: &str,
    ctx: ChatCtx,
    pending_perms: &PendingPermissions,
) -> Result<bool, teloxide::RequestError> {
    let bot = &deps.bot;
    let agent = &deps.agent;
    let channel_map = &deps.channel_map;
    let config = &deps.config;
    let attribution_flag = &deps.attribution_flag;
    let chat_id = ctx.chat_id.0;

    let cmd_word = text.split_whitespace().next().unwrap_or("");
    let cmd = cmd_word.split('@').next().unwrap_or(cmd_word);
    tracing::debug!(chat_id, cmd, cmd_word, "handle_command");
    match cmd {
        "/start" | "/help" => info::cmd_help(bot, agent, channel_map, config, &ctx).await?,
        "/attribution" => {
            ops::cmd_attribution(
                bot,
                agent,
                channel_map,
                config,
                &ctx,
                text,
                attribution_flag,
            )
            .await?
        }
        "/new" => session::cmd_new(bot, agent, channel_map, config, &ctx).await?,
        "/sessions" => session::cmd_sessions(bot, agent, channel_map, config, &ctx, text).await?,
        "/abort" | "/stop" => session::cmd_abort(bot, agent, channel_map, config, &ctx).await?,
        "/status" => session::cmd_status(bot, agent, channel_map, config, &ctx).await?,
        "/usage" => session::cmd_usage(bot, agent, &ctx).await?,
        "/health" => info::cmd_health(bot, agent, channel_map, config, &ctx, text).await?,
        "/metrics" => info::cmd_metrics(bot, agent, channel_map, config, &ctx, text).await?,
        "/compact" => session::cmd_compact(bot, agent, channel_map, config, &ctx).await?,
        "/reload" => ops::cmd_reload(bot, agent, channel_map, config, &ctx).await?,
        "/undo" => ops::cmd_undo(bot, agent, channel_map, config, &ctx).await?,
        "/provider" | "/providers" => {
            model::cmd_provider(bot, agent, channel_map, config, &ctx, text).await?
        }
        "/model" | "/models" => {
            model::cmd_model(bot, agent, channel_map, config, &ctx, text).await?
        }
        "/reasoning" => model::cmd_reasoning(bot, agent, channel_map, config, &ctx, text).await?,
        "/skills" => tools::cmd_skills(bot, agent, channel_map, config, &ctx).await?,
        "/mcp" => tools::cmd_mcp(bot, agent, channel_map, config, &ctx).await?,
        "/refresh" => tools::cmd_refresh(bot, agent, channel_map, config, &ctx).await?,
        "/approve" | "/yolo" => {
            tools::cmd_yolo(bot, agent, channel_map, config, &ctx, text, pending_perms).await?
        }
        "/allow" => {
            tools::cmd_allow(bot, agent, channel_map, config, &ctx, text, pending_perms).await?
        }
        "/research" | "/tasks" => {
            // /tasks is an alias for /research ls
            let effective_text = if cmd_word == "/tasks" {
                "/research ls"
            } else {
                text
            };
            research::handle_research_cmd(
                bot,
                agent,
                channel_map,
                config,
                &ctx,
                effective_text,
                if cmd_word == "/tasks" {
                    "/research"
                } else {
                    cmd_word
                },
            )
            .await?;
        }
        "/memory" => {
            memory::handle_memory_cmd(bot, agent, channel_map, &ctx, text, cmd_word).await?;
        }
        // C5: Git commit with optional message
        "/commit" => ops::cmd_commit(bot, agent, channel_map, config, &ctx, text).await?,
        // B7: Remote execution context
        "/remote" => {
            ops::cmd_remote(bot, agent, channel_map, config, &ctx, text, pending_perms).await?
        }
        _ => {
            return Ok(false);
        }
    }

    Ok(true)
}

pub(crate) async fn get_or_create_session(
    ctx: ChatCtx,
    agent: &Arc<AgentCore>,
    channel_map: &Arc<ChannelSessionMap>,
    config: &Config,
) -> String {
    let chat_id = ctx.chat_id.0;
    let tid = ctx.raw_thread_id();
    if let Some(sid) = channel_map.get(chat_id, tid).await {
        return sid;
    }

    // Per-chat persona: when `naked.json` declares `chat_personas[<chat_id>]`,
    // the new session is rooted in the persona's dedicated workspace instead
    // of the global `config.workspace`. This is the single switch that gives
    // each persona its own system prompt, project memory namespace,
    // CLAUDE.md/AGENTS.md walk, and default `bash` cwd — without forking the
    // bot process. See `naked-core::config::ChatPersona` for the contract.
    let workspace = resolve_session_workspace(chat_id, config).await;

    let session_id = agent
        .create_session_with_channel(&workspace, "telegram")
        .await;
    channel_map.set(chat_id, tid, session_id.clone()).await;

    let channel_id = format_tg_channel_id(chat_id, tid);
    agent.set_session_channel_id(&session_id, &channel_id).await;
    session_id
}

/// Drop a Telegram slash command in a persona chat that opted out of the
/// command surface.
///
/// Returns `true` when the message must be discarded (i.e. the chat has a
/// persona and `allow_slash_commands == false`). The caller should
/// short-circuit `handle_message` immediately. On the FIRST drop per chat
/// we also reply with a one-line hint so the operator knows their `/cmd`
/// was intentional dead air, not a bot bug.
pub(crate) async fn drop_slash_for_persona(
    bot: &Bot,
    chat_id: i64,
    msg: &Message,
    config: &Config,
) -> bool {
    let Some(persona) = config.chat_personas.get(&chat_id) else {
        return false;
    };
    if persona.allow_slash_commands {
        return false;
    }

    let already_hinted = SLASH_HINT_SHOWN.read().await.contains(&chat_id);
    if !already_hinted {
        let inserted = SLASH_HINT_SHOWN.write().await.insert(chat_id);
        if inserted {
            let hint = "В этом чате слеш-команды отключены. Скажи естественным \
                        языком, что нужно (например: «выручка вчера», «маржа \
                        за март», «переключи модель на gpt-5») — я разберусь.";
            if let Err(e) = bot
                .send_message(msg.chat.id, hint)
                .maybe_thread(msg.thread_id)
                .await
            {
                tracing::warn!(
                    chat_id,
                    persona = %persona.name,
                    "failed to send slash-disabled hint: {e}"
                );
            }
        }
    }
    tracing::info!(
        chat_id,
        persona = %persona.name,
        "dropping slash command — persona has allow_slash_commands=false"
    );
    true
}

/// Pick the workspace path for a brand-new session bound to `chat_id`.
///
/// Falls back to `config.workspace` (the legacy single-workspace bot
/// behaviour) when no persona is configured for this chat. When a persona
/// IS configured, ensures its workspace directory and the
/// `<workspace>/.naked` subdirectory exist so the first turn doesn't fail
/// reading a non-existent prompt file. Failures during `mkdir` are logged
/// at warn-level but do not abort session creation — the agent will still
/// boot using the built-in default prompt if the persona's prompt file is
/// unreadable, which is strictly better than refusing to answer.
pub(crate) async fn resolve_session_workspace(chat_id: i64, config: &Config) -> std::path::PathBuf {
    if let Some(persona) = config.chat_personas.get(&chat_id) {
        let ws = persona.workspace_expanded();
        if let Err(e) = tokio::fs::create_dir_all(ws.join(".naked")).await {
            tracing::warn!(
                chat_id,
                persona = %persona.name,
                workspace = %ws.display(),
                "failed to ensure persona workspace dir exists: {e}"
            );
        }
        tracing::info!(
            chat_id,
            persona = %persona.name,
            workspace = %ws.display(),
            "creating new session under persona workspace"
        );
        return ws;
    }
    config.workspace.clone()
}

// ── Formatting helpers ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::fmt_utils::{escape_html_min, safe_slug};
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn safe_slug_basic() {
        assert_eq!(safe_slug("hello world"), "hello-world");
        assert_eq!(safe_slug("my-topic_v2"), "my-topic_v2");
        assert_eq!(safe_slug("  spaces  everywhere  "), "-spaces-everywhere-");
    }

    #[test]
    fn safe_slug_unicode() {
        assert_eq!(safe_slug("квартиры Самуи"), "-");
        assert_eq!(safe_slug("test квартиры"), "test-");
    }

    #[test]
    fn safe_slug_empty() {
        assert_eq!(safe_slug(""), "run");
        assert_eq!(safe_slug("   "), "-"); // all spaces collapse to single dash
    }

    #[test]
    fn escape_html_min_entities() {
        assert_eq!(escape_html_min("a < b > c & d"), "a &lt; b &gt; c &amp; d");
        assert_eq!(escape_html_min("no special"), "no special");
        assert_eq!(escape_html_min(""), "");
    }

    #[test]
    fn escape_html_min_preserves_quotes() {
        // Unlike full escape_html, this minimal version does NOT escape quotes
        assert_eq!(escape_html_min("he said \"hi\""), "he said \"hi\"");
    }
}
