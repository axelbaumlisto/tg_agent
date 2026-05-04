//! /memory command handler.

use super::super::*;

pub(crate) async fn handle_memory_cmd(
    bot: &Bot,
    agent: &Arc<AgentCore>,
    channel_map: &Arc<ChannelSessionMap>,
    ctx: &ChatCtx,
    text: &str,
    cmd_word: &str,
) -> Result<(), teloxide::RequestError> {
    use std::path::PathBuf;

    use naked_core::memory::dreams as memory_dreams;
    use naked_core::memory::service::MemoryService;
    use naked_core::memory::store::MarkdownMemoryStore;
    use naked_core::memory::types::MemoryScope;

    let rest = text[cmd_word.len()..].trim();
    let mut parts = rest.splitn(2, char::is_whitespace);
    let sub = parts.next().unwrap_or("").to_string();
    let _tail = parts.next().unwrap_or("").trim().to_string();

    // Resolve workspace via the session bound to (chat_id, thread). If
    // nothing is bound (fresh chat / pre-/new), `MemoryService::list`
    // tolerates a non-existent path and returns an empty list, so we
    // pick the agent's CWD as a sane fallback.
    let chat_id = ctx.chat_id.0;
    let tid = ctx.raw_thread_id();
    let workspace = if let Some(sid) = channel_map.get(chat_id, tid).await {
        agent
            .session_workspace(&sid)
            .await
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
    } else {
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
    };

    let scope = MemoryScope::Project;

    let reply = match sub.as_str() {
        "" | "help" => "\
/memory ls            durable rules from project MEMORY.md
/memory dreams        last 7 daily-digest audit entries
/memory drafts        today's draft buffer (pre-promotion)
/memory stats         rules / drafts / promoted&rejected (7d)

The digest runs at 04:00 UTC by default. Drafts are auto-collected from \
session closes and pre-compaction flushes; rules promote when they \
re-appear across days."
            .to_string(),

        "ls" | "list" => {
            let entries = MemoryService::list(&workspace, Some(scope.clone()));
            if entries.is_empty() {
                "🧠 No durable rules in project MEMORY.md yet.".to_string()
            } else {
                let lines: Vec<String> = entries
                    .iter()
                    .map(|e| {
                        format!(
                            "• [{}] {}",
                            e.memory_type,
                            e.content.lines().next().unwrap_or("")
                        )
                    })
                    .collect();
                format!(
                    "🧠 <b>{} rule(s)</b> (project)\n\n{}",
                    entries.len(),
                    escape_html(&lines.join("\n"))
                )
            }
        }

        "dreams" => {
            let entries = memory_dreams::read_dreams(&workspace, &scope);
            if entries.is_empty() {
                "💭 No dream entries yet — the digest hasn't run for this project.".to_string()
            } else {
                let recent: Vec<_> = entries.iter().rev().take(7).collect();
                let mut lines = Vec::new();
                lines.push(format!("💭 <b>Last {} dream entries</b>", recent.len()));
                for d in recent {
                    lines.push(format!("\n<b>{}</b>", d.date.format("%Y-%m-%d")));
                    if !d.summary.trim().is_empty() {
                        lines.push(escape_html(&d.summary));
                    }
                    if !d.promoted.is_empty() {
                        lines.push(format!("✅ promoted ({}):", d.promoted.len()));
                        for p in &d.promoted {
                            lines.push(format!("  + {}", escape_html(p)));
                        }
                    }
                    if !d.rejected.is_empty() {
                        lines.push(format!("❌ rejected ({}):", d.rejected.len()));
                        for r in &d.rejected {
                            let reason = r.reason.as_deref().unwrap_or("");
                            lines.push(format!(
                                "  − {} {}",
                                escape_html(&r.content),
                                escape_html(reason)
                            ));
                        }
                    }
                }
                lines.join("\n")
            }
        }

        "drafts" => {
            let today = chrono::Utc::now().date_naive();
            let entries = MarkdownMemoryStore::read_daily(&workspace, &scope, today);
            if entries.is_empty() {
                format!("📝 No draft entries for {today} yet.")
            } else {
                let lines: Vec<String> = entries
                    .iter()
                    .map(|e| format!("• [{}/{}] {}", e.memory_type, e.source, e.content))
                    .collect();
                format!(
                    "📝 <b>{} draft(s) for {today}</b>\n\n{}",
                    entries.len(),
                    escape_html(&lines.join("\n"))
                )
            }
        }

        "stats" => {
            let durable = MemoryService::list(&workspace, Some(scope.clone())).len();
            let today = chrono::Utc::now().date_naive();
            let drafts_today = MarkdownMemoryStore::read_daily(&workspace, &scope, today).len();
            let dream_entries = memory_dreams::read_dreams(&workspace, &scope);
            let week_cutoff = today - chrono::Duration::days(7);
            let recent: Vec<_> = dream_entries
                .iter()
                .filter(|d| d.date >= week_cutoff)
                .collect();
            let promoted_7d: usize = recent.iter().map(|d| d.promoted.len()).sum();
            let rejected_7d: usize = recent.iter().map(|d| d.rejected.len()).sum();
            let last_run = dream_entries
                .iter()
                .map(|d| d.date)
                .max()
                .map(|d| d.to_string())
                .unwrap_or_else(|| "never".to_string());
            format!(
                "📊 <b>Memory stats</b> (project)\n\
                 • durable rules: <b>{durable}</b>\n\
                 • drafts today: <b>{drafts_today}</b>\n\
                 • dream entries (7d): <b>{}</b>\n\
                 • promoted (7d): <b>{promoted_7d}</b>\n\
                 • rejected (7d): <b>{rejected_7d}</b>\n\
                 • last digest run: <b>{last_run}</b>",
                recent.len(),
            )
        }

        other => format!("Unknown subcommand: {other}. Try /memory help"),
    };

    reply_html(bot, ctx, reply).await?;
    Ok(())
}
