//! /ops command handlers.

use super::super::*;

#[allow(unused_variables)]
pub(crate) async fn cmd_attribution(
    bot: &Bot,
    agent: &std::sync::Arc<AgentCore>,
    channel_map: &std::sync::Arc<ChannelSessionMap>,
    config: &Config,
    ctx: &ChatCtx,
    text: &str,
    attribution_flag: &Arc<std::sync::atomic::AtomicBool>,
) -> Result<(), teloxide::RequestError> {
    let chat_id = ctx.chat_id.0;
    let tid = ctx.raw_thread_id();
    use std::sync::atomic::Ordering;
    let cmd_word = text.split_whitespace().next().unwrap_or("");
    let arg = text[cmd_word.len()..].trim().to_ascii_lowercase();
    let reply = match arg.as_str() {
        "on" | "1" | "true" | "enable" => {
            attribution_flag.store(true, Ordering::Relaxed);
            "✅ sender attribution: ON (groups will see `@username:` prefix)".to_string()
        }
        "off" | "0" | "false" | "disable" => {
            attribution_flag.store(false, Ordering::Relaxed);
            "⛔ sender attribution: OFF".to_string()
        }
        "" | "status" => {
            let on = attribution_flag.load(Ordering::Relaxed);
            let state = if on { "ON" } else { "OFF" };
            format!(
                "sender attribution: {state}\nUsage: /attribution on|off|status\n(resets to config.tg_sender_attribution on restart)"
            )
        }
        other => {
            format!("Unknown arg `{other}`. Usage: /attribution on|off|status")
        }
    };
    reply_text(bot, ctx, reply).await?;
    Ok(())
}

#[allow(unused_variables)]
pub(crate) async fn cmd_reload(
    bot: &Bot,
    agent: &std::sync::Arc<AgentCore>,
    channel_map: &std::sync::Arc<ChannelSessionMap>,
    config: &Config,
    ctx: &ChatCtx,
) -> Result<(), teloxide::RequestError> {
    let chat_id = ctx.chat_id.0;
    let tid = ctx.raw_thread_id();

    // Hot-reload naked.json config:
    match Config::load() {
        Ok(new_config) => {
            agent.reload_config(new_config);
            tracing::info!("config hot-reloaded via /reload");
        }
        Err(e) => {
            reply_text(bot, ctx, format!("⚠️ Config reload failed: {e}")).await?;
            return Ok(());
        }
    }

    // Also refresh skills + MCP:
    agent.refresh_skills_and_mcp().await;
    let skills = agent.list_skills();
    let reply = format!(
        "✅ Reloaded config + skills.\n• skills: {}\n• Use /metrics for more.",
        skills.len()
    );
    reply_text(bot, ctx, reply).await?;
    Ok(())
}

#[allow(unused_variables)]
pub(crate) async fn cmd_commit(
    bot: &Bot,
    agent: &std::sync::Arc<AgentCore>,
    channel_map: &std::sync::Arc<ChannelSessionMap>,
    config: &Config,
    ctx: &ChatCtx,
    text: &str,
) -> Result<(), teloxide::RequestError> {
    let chat_id = ctx.chat_id.0;
    let tid = ctx.raw_thread_id();
    let sid = match channel_map.get(chat_id, tid).await {
        Some(s) => s,
        None => {
            reply_text(bot, ctx, "No active session.").await?;
            return Ok(());
        }
    };
    let (_, modified_files) = agent.session_file_stats(&sid).await;
    if modified_files.is_empty() {
        reply_text(bot, ctx, "No files modified in this session.").await?;
        return Ok(());
    }

    // Extract commit message from command text or auto-generate
    let user_msg = text.strip_prefix("/commit").unwrap_or("").trim();
    let workspace = match agent.session_workspace(&sid).await {
        Some(w) => w,
        None => {
            reply_text(bot, ctx, "Session has no workspace.").await?;
            return Ok(());
        }
    };
    let ws = workspace.to_string_lossy();

    // Check if inside a git repo
    let git_check = tokio::process::Command::new("git")
        .args(["rev-parse", "--is-inside-work-tree"])
        .current_dir(&workspace)
        .output()
        .await;
    if !git_check
        .as_ref()
        .map(|o| o.status.success())
        .unwrap_or(false)
    {
        reply_text(bot, ctx, &format!("Not a git repo: {ws}")).await?;
        return Ok(());
    }

    // git add modified files
    for f in &modified_files {
        let _ = tokio::process::Command::new("git")
            .args(["add", f])
            .current_dir(&workspace)
            .output()
            .await;
    }

    // Commit with message
    let msg = if user_msg.is_empty() {
        format!(
            "auto: modified {} file(s)\n\nFiles:\n{}",
            modified_files.len(),
            modified_files.join("\n")
        )
    } else {
        user_msg.to_string()
    };

    let commit_result = tokio::process::Command::new("git")
        .args(["commit", "-m", &msg])
        .current_dir(&workspace)
        .output()
        .await;

    match commit_result {
        Ok(output) if output.status.success() => {
            let out = String::from_utf8_lossy(&output.stdout);
            let short: String = out.lines().next().unwrap_or("committed").to_string();
            reply_text(bot, ctx, &format!("✅ {short}")).await?;
        }
        Ok(output) => {
            let err = String::from_utf8_lossy(&output.stderr);
            let short: String = err.chars().take(200).collect();
            reply_text(bot, ctx, &format!("❌ git commit failed: {short}")).await?;
        }
        Err(e) => {
            reply_text(bot, ctx, &format!("❌ git error: {e}")).await?;
        }
    }
    Ok(())
}

#[allow(unused_variables)]
pub(crate) async fn cmd_remote(
    bot: &Bot,
    agent: &std::sync::Arc<AgentCore>,
    channel_map: &std::sync::Arc<ChannelSessionMap>,
    config: &Config,
    ctx: &ChatCtx,
    text: &str,
    pending_perms: &PendingPermissions,
) -> Result<(), teloxide::RequestError> {
    let chat_id = ctx.chat_id.0;
    let tid = ctx.raw_thread_id();
    let args = text.strip_prefix("/remote").unwrap_or("").trim();
    let remote_ctx = agent.remote_context();

    if args.is_empty() || args == "status" {
        let label = remote_ctx.label().await;
        reply_text(
            bot,
            ctx,
            &format!(
                "🌐 Target: <b>{label}</b>\n\n\
            /remote &lt;host&gt; — switch to SSH\n\
            /remote off — back to local"
            ),
        )
        .await?;
    } else if args == "off" || args == "local" {
        remote_ctx.set_local().await;
        reply_text(bot, ctx, "🌐 Switched to <b>local</b>").await?;
    } else {
        // args = host, optionally "host key=/path/to/key"
        let parts: Vec<&str> = args.splitn(2, ' ').collect();
        let host = parts[0].to_string();
        let key = parts
            .get(1)
            .and_then(|s| s.strip_prefix("key="))
            .map(|s| s.to_string());

        // Quick connectivity test
        let test = tokio::process::Command::new("ssh")
            .args(["-o", "ConnectTimeout=5", "-o", "BatchMode=yes"])
            .args(key.as_ref().map(|k| vec!["-i", k]).unwrap_or_default())
            .arg(&host)
            .arg("echo ok")
            .output()
            .await;

        match test {
            Ok(out) if out.status.success() => {
                remote_ctx.set_ssh(host.clone(), key).await;
                reply_text(bot, ctx, &format!("🌐 Connected to <b>{host}</b>")).await?;
            }
            Ok(out) => {
                let err = String::from_utf8_lossy(&out.stderr);
                let short: String = err.chars().take(200).collect();
                reply_text(bot, ctx, &format!("❌ SSH to {host} failed: {short}")).await?;
            }
            Err(e) => {
                reply_text(bot, ctx, &format!("❌ SSH error: {e}")).await?;
            }
        }
    }
    Ok(())
}

pub(crate) async fn cmd_undo(
    bot: &Bot,
    agent: &std::sync::Arc<AgentCore>,
    channel_map: &std::sync::Arc<ChannelSessionMap>,
    config: &Config,
    ctx: &ChatCtx,
) -> Result<(), teloxide::RequestError> {
    let sid = super::get_or_create_session(*ctx, agent, channel_map, config).await;
    let workspace = agent.session_workspace(&sid).await.unwrap_or_default();
    match naked_core::snapshot::undo_last(&workspace).await {
        Ok(msg) => {
            super::reply_text(bot, ctx, &format!("↩️ {msg}")).await?;
        }
        Err(e) => {
            super::reply_text(bot, ctx, &format!("❌ {e}")).await?;
        }
    }
    Ok(())
}
