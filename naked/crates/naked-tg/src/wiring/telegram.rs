use std::sync::Arc;
use std::time::Duration;

use teloxide::prelude::*;
use teloxide::types::BotCommand;

use naked_core::config::Config;

pub(crate) struct TelegramClientBundle {
    pub(crate) bot: Bot,
    pub(crate) bot_token: Arc<String>,
    pub(crate) bot_identity: Arc<naked_tg::bot_identity::BotIdentity>,
    pub(crate) http_client: Arc<reqwest::Client>,
    pub(crate) base_url: Arc<String>,
}

pub(crate) async fn boot_telegram_client(config: &Config) -> TelegramClientBundle {
    let bot_token = config
        .telegram
        .telegram_bot_token
        .clone()
        .or_else(|| std::env::var("TELEGRAM_BOT_TOKEN").ok())
        .expect("telegram_bot_token must be set in config or TELEGRAM_BOT_TOKEN env var");

    let bot = Bot::new(&bot_token);
    register_commands(&bot).await;

    tracing::info!(
        "Starting Telegram bot ({}/{})",
        config.default_provider,
        config.default_model
    );

    // Manual polling loop client — avoids teloxide's Dispatcher/Polling
    // which conflicts with stale getUpdates connections from other processes.
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .expect("reqwest client");
    let base = format!("https://api.telegram.org/bot{bot_token}");
    let bot_identity = resolve_bot_identity(&client, &base).await;

    TelegramClientBundle {
        bot,
        bot_token: Arc::new(bot_token),
        bot_identity,
        http_client: Arc::new(client),
        base_url: Arc::new(base),
    }
}

async fn resolve_bot_identity(
    client: &reqwest::Client,
    base: &str,
) -> Arc<naked_tg::bot_identity::BotIdentity> {
    // Resolve our own bot identity (id + @username) so the group-chat
    // gate can tell "this message is for us" apart from "humans
    // chatting with each other while the bot lurks". Privacy mode is
    // OFF for this bot (`can_read_all_group_messages: true`), which
    // means Telegram delivers every group message — without this
    // identity-aware filter the bot would respond to all of them.
    match client
        .get(format!("{base}/getMe"))
        .send()
        .await
        .and_then(|r| r.error_for_status())
    {
        Ok(resp) => match resp.json::<serde_json::Value>().await {
            Ok(v) => {
                let id = v
                    .get("result")
                    .and_then(|r| r.get("id"))
                    .and_then(|i| i.as_u64())
                    .unwrap_or(0);
                let username = v
                    .get("result")
                    .and_then(|r| r.get("username"))
                    .and_then(|u| u.as_str())
                    .unwrap_or("")
                    .to_string();
                if id == 0 || username.is_empty() {
                    tracing::warn!(
                        "getMe returned malformed payload — group-chat addressing filter \
                         will refuse every group message. Check the bot token."
                    );
                }
                tracing::info!(bot_id = id, %username, "bot identity resolved via getMe");
                Arc::new(naked_tg::bot_identity::BotIdentity { id, username })
            }
            Err(e) => {
                tracing::error!(
                    "getMe parse error: {e} — using zero identity (group filter will reject everything)"
                );
                zero_bot_identity()
            }
        },
        Err(e) => {
            tracing::error!(
                "getMe request failed: {e} — using zero identity (group filter will reject everything)"
            );
            zero_bot_identity()
        }
    }
}

fn zero_bot_identity() -> Arc<naked_tg::bot_identity::BotIdentity> {
    Arc::new(naked_tg::bot_identity::BotIdentity {
        id: 0,
        username: String::new(),
    })
}

pub(crate) async fn clear_webhook(client: &reqwest::Client, base: &str) {
    // Drop any pending updates + delete webhook on startup.
    let _ = client
        .post(format!("{base}/deleteWebhook"))
        .json(&serde_json::json!({"drop_pending_updates": true}))
        .send()
        .await;
    tracing::info!("Webhook cleared, starting polling loop");
}

/// Register bot commands with Telegram so the slash-menu is populated.
async fn register_commands(bot: &Bot) {
    let commands = vec![
        BotCommand::new("status", "Session status, usage, cost"),
        BotCommand::new("compact", "Compact session history"),
        BotCommand::new("new", "Start a new session"),
        BotCommand::new("sessions", "List active sessions"),
        BotCommand::new("stop", "Cancel running task"),
        BotCommand::new("abort", "Cancel running task (alias)"),
        BotCommand::new("provider", "Switch provider"),
        BotCommand::new("model", "Switch model"),
        BotCommand::new("skills", "List loaded skills"),
        BotCommand::new("mcp", "List MCP servers"),
        BotCommand::new("refresh", "Reload skills & MCP"),
        BotCommand::new("reasoning", "Set thinking/reasoning level"),
        BotCommand::new("yolo", "Auto-approve ALL tools in this topic"),
        BotCommand::new("allow", "Manage tool allow-list for this topic"),
        BotCommand::new("memory", "Memory: rules / dreams / drafts / stats"),
        BotCommand::new("research", "Run, list, pause or resume research"),
        BotCommand::new("health", "Provider health & key status"),
        BotCommand::new("commit", "Git commit modified files"),
        BotCommand::new("remote", "Switch to SSH remote host"),
        BotCommand::new("help", "Show all commands"),
    ];
    if let Err(e) = bot.set_my_commands(commands).await {
        tracing::warn!("Failed to set bot commands: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::zero_bot_identity;

    #[test]
    fn zero_identity_is_safe_reject_all_fallback() {
        let identity = zero_bot_identity();
        assert_eq!(identity.id, 0);
        assert!(identity.username.is_empty());
    }
}
