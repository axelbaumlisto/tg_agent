mod research;
use std::io::{self, BufRead, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::Result;

use naked_core::AgentCore;
use naked_core::config::Config;
use naked_core::types::{AgentEvent, AgentHandle, PermissionResponse, TurnUsage};

const SPINNER_FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("naked=info".parse().expect("static directive")),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && args[1] == "copilot-login" {
        return copilot_login_cmd().await;
    }
    if args.len() > 1 && args[1] == "memory" {
        return memory_cmd(&args[2..]).await;
    }
    if args.len() > 1 && args[1] == "vacuum-sessions" {
        return vacuum_sessions_cmd(&args[2..]).await;
    }
    if args.len() > 1 && args[1] == "research" {
        return research::research_cmd(&args[2..]).await;
    }
    if args.len() > 1 && args[1] == "agent" {
        return agent_cmd(&args[2..]).await;
    }
    if args.len() > 1 && args[1] == "skills" {
        return skills_cmd(&args[2..]).await;
    }

    let config = Config::load()?;
    let provider = naked_core::build_provider_from_config(&config)?;
    let agent = AgentCore::new(config.clone(), provider);

    agent.init_mcp().await;

    let restored = agent.restore_sessions().await.unwrap_or_default();
    if !restored.is_empty() {
        eprintln!("Restored {} session(s)", restored.len());
    }

    let workspace = config.workspace.clone();
    let session_id = agent.create_session(&workspace).await;
    eprintln!("Session: {session_id}");
    eprintln!(
        "Model: {}/{}",
        config.default_provider, config.default_model
    );
    eprintln!("Workspace: {}", workspace.display());
    eprintln!("---");
    eprintln!("Type your message (Ctrl+D to exit):\n");

    let stdin = io::stdin();
    loop {
        eprint!("> ");
        io::stderr().flush()?;

        let mut input = String::new();
        if stdin.lock().read_line(&mut input)? == 0 {
            break;
        }
        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        if input == "/exit" || input == "/quit" {
            break;
        }

        if input == "/providers" {
            let providers = agent.list_providers();
            let (cur_p, cur_m) = agent.session_provider_model(&session_id).await;
            eprintln!("Current: {cur_p}/{cur_m}\n");
            for p in &providers {
                let mark = if p.name == cur_p { " ✅" } else { "" };
                eprintln!("  {}{mark}", p.name);
                for m in &p.models {
                    eprintln!("    • {m}");
                }
            }
            eprintln!();
            continue;
        }

        if let Some(arg) = input.strip_prefix("/provider") {
            let arg = arg.trim();
            if arg.is_empty() {
                let (p, m) = agent.session_provider_model(&session_id).await;
                eprintln!("Current: {p}/{m}");
                eprintln!("Usage: /provider <name>");
            } else {
                match agent
                    .set_session_provider(&session_id, Some(arg), None)
                    .await
                {
                    Ok(()) => {
                        let (p, m) = agent.session_provider_model(&session_id).await;
                        eprintln!("Switched to: {p}/{m}");
                    }
                    Err(e) => eprintln!("Error: {e}"),
                }
            }
            continue;
        }

        if let Some(arg) = input.strip_prefix("/model") {
            let arg = arg.trim();
            if arg.is_empty() {
                let (p, m) = agent.session_provider_model(&session_id).await;
                eprintln!("Current: {p}/{m}");
                eprintln!("Usage: /model <name>");
            } else {
                match agent
                    .set_session_provider(&session_id, None, Some(arg))
                    .await
                {
                    Ok(()) => {
                        let (p, m) = agent.session_provider_model(&session_id).await;
                        eprintln!("Model set: {p}/{m}");
                    }
                    Err(e) => eprintln!("Error: {e}"),
                }
            }
            continue;
        }

        if input == "/models" {
            let models = agent.list_models();
            for m in &models {
                eprintln!("  {}/{}", m.provider, m.model_id);
            }
            continue;
        }

        match agent.send_prompt(&session_id, input).await {
            Ok(handle) => {
                run_turn(handle).await?;
            }
            Err(e) => {
                eprintln!("Error: {e}");
            }
        }
    }

    eprintln!("\nBye!");
    Ok(())
}

async fn run_turn(handle: AgentHandle) -> Result<()> {
    let AgentHandle {
        mut events,
        permissions,
        steer: _,
    } = handle;

    let spinning = Arc::new(AtomicBool::new(true));
    let spinner_flag = spinning.clone();

    let spinner_handle = std::thread::spawn(move || {
        let mut frame = 0usize;
        while spinner_flag.load(Ordering::Relaxed) {
            eprint!(
                "\r\x1b[36m{} Thinking…\x1b[0m ",
                SPINNER_FRAMES[frame % SPINNER_FRAMES.len()]
            );
            let _ = io::stderr().flush();
            frame += 1;
            std::thread::sleep(Duration::from_millis(100));
        }
        eprint!("\r\x1b[2K");
        let _ = io::stderr().flush();
    });

    let mut got_content = false;
    let mut last_usage: Option<TurnUsage> = None;

    while let Some(event) = events.recv().await {
        match event {
            AgentEvent::ThinkingDelta(t) => {
                if !got_content {
                    spinning.store(false, Ordering::Relaxed);
                    got_content = true;
                }
                eprint!("\x1b[2m{t}\x1b[0m");
                io::stderr().flush()?;
            }
            AgentEvent::TextDelta(t) => {
                if !got_content {
                    spinning.store(false, Ordering::Relaxed);
                    got_content = true;
                    eprintln!();
                }
                print!("{t}");
                io::stdout().flush()?;
            }
            AgentEvent::ToolStart { name, input, .. } => {
                if !got_content {
                    spinning.store(false, Ordering::Relaxed);
                    got_content = true;
                }
                let preview = cli_input_preview(&input, 120);
                eprintln!("\n\x1b[38;5;245m╭─ \x1b[1;36m{name}\x1b[0;38;5;245m ─╮\x1b[0m");
                eprintln!("\x1b[38;5;245m│\x1b[0m {preview}");
                eprintln!("\x1b[38;5;245m╰──────╯\x1b[0m");
            }
            AgentEvent::ToolEnd {
                name,
                state,
                output,
                ..
            } => {
                let (icon, color) = match state {
                    naked_core::types::ToolState::Completed => ("✓", "\x1b[32m"),
                    naked_core::types::ToolState::Error => ("✗", "\x1b[31m"),
                };
                if output.len() > 500 {
                    eprintln!("{color}{icon} {name}\x1b[0m ({}b)", output.len());
                    eprintln!("\x1b[2m{}\x1b[0m", &output[..500]);
                } else if !output.is_empty() {
                    eprintln!("{color}{icon} {name}\x1b[0m");
                    eprintln!("\x1b[2m{output}\x1b[0m");
                } else {
                    eprintln!("{color}{icon} {name}\x1b[0m");
                }
            }
            AgentEvent::PermissionRequest {
                call_id,
                tool_name,
                input,
                permission,
            } => {
                if !got_content {
                    spinning.store(false, Ordering::Relaxed);
                    got_content = true;
                }
                let level = match &permission {
                    naked_core::types::Permission::WorkspaceWrite => "write",
                    naked_core::types::Permission::Dangerous => "dangerous",
                    naked_core::types::Permission::ReadOnly => "read",
                };
                let preview = cli_input_preview(&input, 200);
                eprintln!();
                eprintln!("\x1b[33mPermission required [{level}]\x1b[0m");
                eprintln!("  Tool: \x1b[1m{tool_name}\x1b[0m");
                eprintln!("  Input: {preview}");
                eprint!("Approve? [y/N]: ");
                io::stderr().flush()?;

                let mut answer = String::new();
                let _ = io::stdin().lock().read_line(&mut answer);
                let allowed = matches!(answer.trim().to_lowercase().as_str(), "y" | "yes");

                let _ = permissions
                    .send(PermissionResponse { call_id, allowed })
                    .await;
            }
            AgentEvent::Heartbeat => {}
            AgentEvent::SubAgentProgress {
                agent_id,
                event: sa_ev,
            } => {
                use naked_core::types::SubAgentEvent;
                match sa_ev {
                    SubAgentEvent::Started { prompt_preview } => {
                        let short = if prompt_preview.len() > 80 {
                            &prompt_preview[..80]
                        } else {
                            &prompt_preview
                        };
                        eprintln!("\x1b[36m🤖 {agent_id}\x1b[0m → {short}");
                    }
                    SubAgentEvent::ToolUse {
                        name,
                        input_preview,
                    } => {
                        let short = if input_preview.len() > 60 {
                            &input_preview[..60]
                        } else {
                            &input_preview
                        };
                        eprintln!("\x1b[2m  🔧 {agent_id}/{name}\x1b[0m({short})");
                    }
                    SubAgentEvent::ToolDone { name, state } => {
                        let icon = match state {
                            naked_core::types::ToolState::Completed => "\x1b[32m✓\x1b[0m",
                            naked_core::types::ToolState::Error => "\x1b[31m✗\x1b[0m",
                        };
                        eprintln!("  {icon} {agent_id}/{name}");
                    }
                    SubAgentEvent::TextDelta(_) => {}
                    SubAgentEvent::Finished { tokens } => {
                        eprintln!("\x1b[32m✓ {agent_id}\x1b[0m — {tokens} tok");
                    }
                    SubAgentEvent::Error(e) => {
                        eprintln!("\x1b[31m✗ {agent_id}\x1b[0m — {e}");
                    }
                }
            }
            AgentEvent::ContextCompacted {
                before_msgs,
                after_msgs,
                summary_hint,
                files_count,
            } => {
                let hint = summary_hint.as_deref().unwrap_or("");
                eprintln!(
                    "\x1b[33m[context compacted: {before_msgs} msgs → {after_msgs} | {files_count} files | {hint}]\x1b[0m"
                );
            }
            AgentEvent::ToolOutput { chunk, .. } => {
                // Show last line of bash output inline:
                if let Some(last) = chunk.lines().last() {
                    eprint!("\r\x1b[2K\x1b[90m  > {last}\x1b[0m");
                }
            }
            AgentEvent::SteerReceived { text } => {
                eprintln!("\x1b[36m[steer: {text}]\x1b[0m");
            }
            AgentEvent::UsageUpdate(u) => {
                last_usage = Some(u);
            }
            AgentEvent::Error(e) => {
                if !got_content {
                    spinning.store(false, Ordering::Relaxed);
                    got_content = true;
                }
                eprintln!("\n\x1b[31m[error: {e}]\x1b[0m");
            }
            AgentEvent::Idle => {
                println!();
            }
        }
    }

    spinning.store(false, Ordering::Relaxed);
    let _ = spinner_handle.join();

    if let Some(u) = last_usage {
        eprintln!(
            "\x1b[2mtokens: {} in / {} out (total: {})\x1b[0m",
            u.input_tokens,
            u.output_tokens,
            u.total_tokens()
        );
    }

    Ok(())
}

fn cli_input_preview(input: &serde_json::Value, max_len: usize) -> String {
    let raw = if let Some(map) = input.as_object() {
        if map.len() == 1 {
            let (key, val) = map.iter().next().expect("checked len==1");
            let fallback = val.to_string();
            let v = val.as_str().unwrap_or(&fallback);
            format!("{key}: {v}")
        } else {
            let parts: Vec<String> = map
                .iter()
                .map(|(k, v)| {
                    let fallback = v.to_string();
                    let s = v.as_str().unwrap_or(&fallback);
                    format!("{k}: {}", val_short(s, 60))
                })
                .collect();
            parts.join(", ")
        }
    } else {
        input.to_string()
    };
    if raw.len() <= max_len {
        raw
    } else {
        let truncated: String = raw.chars().take(max_len).collect();
        format!("{truncated}…")
    }
}

fn val_short(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max).collect();
        format!("{truncated}…")
    }
}

async fn memory_cmd(args: &[String]) -> Result<()> {
    use naked_core::memory::service::MemoryService;
    use naked_core::memory::types::{MemoryScope, MemoryType};

    fn arg_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .map(|s| s.as_str())
    }

    let workspace = std::env::current_dir()?;
    let sub = args.first().map(|s| s.as_str()).unwrap_or("help");
    let user_id = arg_value(args, "--user").map(str::to_string);

    // Pick an explicit scope, if any. --user wins over --global / --project.
    let explicit_scope = if let Some(id) = user_id.clone() {
        Some(MemoryScope::User(id))
    } else if args.iter().any(|a| a == "--global") {
        Some(MemoryScope::Global)
    } else if args.iter().any(|a| a == "--project") {
        Some(MemoryScope::Project)
    } else {
        None
    };

    match sub {
        "list" => {
            let entries = MemoryService::list(&workspace, explicit_scope);
            if entries.is_empty() {
                eprintln!("No memories stored.");
            } else {
                for entry in &entries {
                    let date = entry.created_at.format("%Y-%m-%d");
                    println!(
                        "[{}/{}] (id:{}, {date}, src:{}) {}",
                        entry.scope, entry.memory_type, entry.id, entry.source, entry.content
                    );
                }
                eprintln!("\n{} total", entries.len());
            }
        }

        "search" => {
            let query = args.get(1).map(|s| s.as_str()).unwrap_or("");
            if query.is_empty() {
                eprintln!("Usage: naked memory search <query> [--user <id>]");
                return Ok(());
            }
            let results = MemoryService::search_for(&workspace, query, user_id.as_deref());
            if results.is_empty() {
                eprintln!("No memories matching \"{query}\".");
            } else {
                for entry in &results {
                    println!(
                        "[{}/{}] (id:{}) {}",
                        entry.scope, entry.memory_type, entry.id, entry.content
                    );
                }
            }
        }

        "add" => {
            let content = args.get(1).map(|s| s.as_str()).unwrap_or("");
            if content.is_empty() {
                eprintln!(
                    "Usage: naked memory add <content> [--type preference|correction|project_knowledge|failure] [--global|--user <id>]"
                );
                return Ok(());
            }
            let type_str = arg_value(args, "--type").unwrap_or("preference");
            let memory_type: MemoryType =
                type_str.parse().map_err(|e: String| anyhow::anyhow!(e))?;
            let scope = explicit_scope.unwrap_or(MemoryScope::Project);

            let scope_label = scope.to_string();
            match MemoryService::store(&workspace, scope, memory_type, content, "user") {
                Ok(true) => eprintln!("Stored: [{scope_label}/{memory_type}] {content}"),
                Ok(false) => eprintln!("Duplicate — already exists."),
                Err(e) => eprintln!("Error: {e}"),
            }
        }

        "delete" => {
            let id = args.get(1).map(|s| s.as_str()).unwrap_or("");
            if id.is_empty() {
                eprintln!("Usage: naked memory delete <id> [--user <uid>]");
                return Ok(());
            }
            let result = if let Some(uid) = user_id.as_deref() {
                MemoryService::delete_user(uid, id)
            } else {
                MemoryService::delete(&workspace, id)
            };
            match result {
                Ok(true) => eprintln!("Deleted memory {id}"),
                Ok(false) => eprintln!("Memory {id} not found"),
                Err(e) => eprintln!("Error: {e}"),
            }
        }

        "clear" => {
            let scope = explicit_scope.unwrap_or(MemoryScope::Project);
            let scope_label = scope.to_string();
            MemoryService::clear(&workspace, scope)?;
            eprintln!("Cleared {scope_label} memory.");
        }

        _ => {
            eprintln!("naked memory — persistent memory management\n");
            eprintln!("Commands:");
            eprintln!("  list   [--global|--project|--user <id>]");
            eprintln!("                                    List stored memories");
            eprintln!("  search <query> [--user <id>]      Search by substring");
            eprintln!("  add    <content> [--type TYPE] [--global|--user <id>]");
            eprintln!("                                    Add a memory manually");
            eprintln!("  delete <id> [--user <id>]         Delete by ID");
            eprintln!("  clear  [--global|--user <id>]     Clear all (project, global, or user)");
            eprintln!("\nTypes: preference, correction, project_knowledge, failure");
        }
    }

    Ok(())
}

async fn copilot_login_cmd() -> Result<()> {
    use naked_core::provider::copilot;

    eprintln!("GitHub Copilot — OAuth Device Login\n");

    if let Some(token) = copilot::load_copilot_token() {
        let masked = if token.len() > 8 {
            format!("{}…{}", &token[..4], &token[token.len() - 4..])
        } else {
            "****".to_string()
        };
        eprintln!("Existing token found: {masked}");
        eprint!("Re-authenticate? [y/N]: ");
        io::stderr().flush()?;
        let mut answer = String::new();
        let _ = io::stdin().lock().read_line(&mut answer);
        if !matches!(answer.trim().to_lowercase().as_str(), "y" | "yes") {
            eprintln!("Keeping existing token.");

            eprintln!("\nFetching available models...");
            match copilot::fetch_copilot_models(&token).await {
                Ok(models) if !models.is_empty() => {
                    eprintln!("Available Copilot models:");
                    for m in &models {
                        eprintln!("  • {m}");
                    }
                }
                Ok(_) => eprintln!("No models returned (token may be expired)."),
                Err(e) => eprintln!("Could not fetch models: {e}"),
            }
            return Ok(());
        }
    }

    let token = copilot::copilot_device_login().await?;

    eprintln!("\nFetching available models...");
    match copilot::fetch_copilot_models(&token).await {
        Ok(models) if !models.is_empty() => {
            eprintln!("Available Copilot models:");
            for m in &models {
                eprintln!("  • {m}");
            }
        }
        Ok(_) => eprintln!("No models returned yet — try again shortly."),
        Err(e) => eprintln!("Could not fetch models: {e}"),
    }

    Ok(())
}

/// Migrate existing JSONL sessions that still carry inline base64 image
/// payloads. For each session, load → re-save (which runs through
/// `extern_image_blocks` and writes images to `<session>/artifacts/`), then
/// sweep orphan artifacts. Reports byte savings per session and a grand total.
///
/// Why a separate command instead of "migrate on first save": existing
/// long-lived sessions may never get a save event again (channel-only chats
/// closed by the user), and we want a single deterministic operator action to
/// reclaim disk space after upgrading.
async fn vacuum_sessions_cmd(extra_args: &[String]) -> Result<()> {
    // Optional `--max-age-days N` flag runs an age-based sweep across every
    // session's artifacts directory *in addition to* reachability GC. Default
    // N=0 means "age sweep disabled; reachability-only".
    let max_age_days: u64 = extra_args
        .iter()
        .position(|a| a == "--max-age-days")
        .and_then(|i| extra_args.get(i + 1))
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);
    let max_age_secs = max_age_days.saturating_mul(24 * 3600);

    let config = Config::load()?;
    let provider = naked_core::build_provider_from_config(&config)?;
    let agent = AgentCore::new(config.clone(), provider);
    let restored = agent.restore_sessions().await.unwrap_or_default();
    eprintln!("Found {} session(s).", restored.len());
    if max_age_days > 0 {
        eprintln!("Age-based sweep enabled: delete img_* older than {max_age_days} day(s).");
    }

    let store = agent.store();

    let mut total_saved: i64 = 0;
    let mut migrated = 0usize;
    let mut aged_out = 0usize;
    for sid in &restored {
        let before = std::fs::metadata(store.session_root(sid).join("session.jsonl"))
            .map(|m| m.len() as i64)
            .unwrap_or(0);

        let session = match store.load(sid).await? {
            Some(s) => s,
            None => continue,
        };
        store.save(&session).await?;

        let removed = store.gc_orphan_image_artifacts(sid).await.unwrap_or(0);
        let aged = if max_age_secs > 0 {
            store
                .gc_old_image_artifacts(sid, max_age_secs)
                .await
                .unwrap_or(0)
        } else {
            0
        };
        aged_out += aged;

        let after = std::fs::metadata(store.session_root(sid).join("session.jsonl"))
            .map(|m| m.len() as i64)
            .unwrap_or(0);
        let saved = before - after;
        total_saved += saved;
        if saved != 0 || removed > 0 || aged > 0 {
            migrated += 1;
            eprintln!(
                "  • {}…  saved {saved:+} bytes, gc'd {removed} orphan + {aged} aged artifact(s)",
                &sid[..sid.len().min(8)]
            );
        }
    }
    eprintln!(
        "Done. {migrated} session(s) had changes; jsonl delta {total_saved:+} bytes; \
         {aged_out} aged artifact(s) removed."
    );
    Ok(())
}

/// Collect every value following an occurrence of `flag`.
///
/// Handles repeated flags like `--source URL --source URL2`. Used by
/// every subcommand parser in this binary.
fn arg_values<'a>(args: &'a [String], flag: &str) -> Vec<&'a str> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == flag
            && let Some(v) = args.get(i + 1)
        {
            out.push(v.as_str());
            i += 2;
        } else {
            i += 1;
        }
    }
    out
}

/// `naked research <subcommand>` — operator surface for the research
/// subsystem. Paired with the `/research` Telegram commands and the
/// `naked-research@.timer` systemd unit:
///
/// - `run <id>` — block until one pass finishes; exit non-zero on failure, so
///   systemd OnFailure= hooks fire. This is the entrypoint every timer hits.
/// - `ls` — plain listing, one per line, most-recent-first.
/// - `show <id>` — spec summary + last 10 findings + last 3 runs.
/// - `export <id> <path>` — dump `report.md` to disk (for manual review,
///   email, scp, etc.).
/// - `new <topic>` — create a spec and print the id. `--source URL` may be
///   repeated.
/// - `rm <id>` — delete the spec directory (irrecoverable).
async fn agent_cmd(args: &[String]) -> Result<()> {
    use naked_core::agent_role::Task;
    use naked_core::agent_run::{run_batch, run_task};
    use naked_core::agent_store::resolve_role;
    use naked_core::agent_validator::run_all;

    let config = Config::load()?;
    let provider = naked_core::build_provider_from_config(&config)?;
    let agent = AgentCore::new(config.clone(), provider);
    agent.init_mcp().await;
    let agent = Arc::new(agent);

    let sub = args.first().map(String::as_str).unwrap_or("help");
    match sub {
        "list-roles" | "roles" => {
            let store = agent.agent_store();
            let mut roles = store.list();
            roles.sort_by(|a, b| a.name.cmp(&b.name));
            if roles.is_empty() {
                eprintln!(
                    "(no roles loaded — check `agent_dirs` in naked.json: {:?})",
                    config.agent_dirs,
                );
            }
            for r in roles {
                let model = r.model.as_deref().unwrap_or("(default)");
                println!(
                    "{:<20} model={:<24} skills=[{}]",
                    r.name,
                    model,
                    r.default_skills.join(", "),
                );
                if !r.description.is_empty() {
                    println!("  {}", r.description);
                }
            }
        }
        "run" => {
            // naked agent run <role> <prompt> [--max-wall N] [--context JSON]
            let role_name = args.get(1).ok_or_else(|| {
                anyhow::anyhow!(
                    "usage: naked agent run <role> <prompt> [--max-wall N] [--context JSON]"
                )
            })?;
            let prompt = args
                .get(2)
                .ok_or_else(|| anyhow::anyhow!("usage: naked agent run <role> <prompt>"))?;
            let role = resolve_role(role_name, &agent.agent_store(), &config.agent_roles)
                .ok_or_else(|| {
                    anyhow::anyhow!("unknown role `{role_name}` — try `naked agent list-roles`")
                })?;

            let max_wall = arg_values(args, "--max-wall")
                .first()
                .and_then(|s| s.parse::<u64>().ok());
            let ctx = arg_values(args, "--context")
                .first()
                .and_then(|s| serde_json::from_str(s).ok())
                .unwrap_or(serde_json::Value::Null);

            let mut task = Task::new(role.name.clone(), prompt.clone()).with_context(ctx);
            if let Some(w) = max_wall {
                task = task.with_max_wall(w);
            }

            eprintln!(
                "running role `{}` (model={}, max_wall={})",
                role.name,
                role.model.as_deref().unwrap_or("(default)"),
                task.max_wall_secs
                    .map_or("default".into(), |w| format!("{w}s")),
            );
            let started = std::time::Instant::now();
            let out = run_task(&agent, &role, &task).await?;
            let elapsed = started.elapsed();

            println!("=== task result ===");
            println!("task_id:     {}", out.task_id);
            println!("role:        {}", out.role_name);
            println!("stop_reason: {:?}", out.stop_reason);
            println!("elapsed:     {:.1?}", elapsed);
            println!("text_chars:  {}", out.text.len());
            println!("{}", out.stats.summary_line());
            if !out.text.is_empty() {
                println!("--- text ---");
                println!("{}", out.text);
            }
        }
        "batch" => {
            // naked agent batch <jsonl-file> [--validators phone-vn,no-fake-contact]
            let path = args.get(1).ok_or_else(|| {
                anyhow::anyhow!("usage: naked agent batch <tasks.jsonl> [--validators v1,v2,...]")
            })?;
            let raw = std::fs::read_to_string(path)?;
            let mut tasks: Vec<Task> = Vec::new();
            for (lineno, line) in raw.lines().enumerate() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                let task: Task = serde_json::from_str(line)
                    .map_err(|e| anyhow::anyhow!("{path}:{} parse error: {e}", lineno + 1))?;
                tasks.push(task);
            }
            if tasks.is_empty() {
                return Err(anyhow::anyhow!("no tasks found in {path}"));
            }

            // Plain batch has no per-task criteria source; gatekeeper
            // is only available via `batch-from-spec` where the spec
            // topic supplies the criteria. Plain batch also can't lift
            // role-level defaults uniformly (each line picks its own
            // role) — so the operator stays responsible for picking
            // validators here. Pass `&[]` to skip role-default lookup.
            let validators = parse_validators(args, None, &[]).await?;

            let role_set: std::collections::BTreeSet<String> =
                tasks.iter().map(|t| t.role.clone()).collect();
            let mut roles_vec = Vec::new();
            for name in &role_set {
                let r = resolve_role(name, &agent.agent_store(), &config.agent_roles)
                    .ok_or_else(|| anyhow::anyhow!("unknown role `{name}` in batch"))?;
                roles_vec.push(Arc::new(r));
            }

            eprintln!(
                "running {} task(s) over {} role(s) with {} validator(s)",
                tasks.len(),
                role_set.len(),
                validators.len(),
            );
            let started = std::time::Instant::now();
            let outputs = run_batch(agent.clone(), &roles_vec, tasks).await;
            let elapsed = started.elapsed();
            eprintln!("batch complete in {:.1?}", elapsed);

            let mut pass_count = 0u32;
            let mut fail_count = 0u32;
            let mut error_count = 0u32;
            println!();
            println!("=== batch summary ===");
            for (i, res) in outputs.into_iter().enumerate() {
                match res {
                    Err(e) => {
                        error_count += 1;
                        println!("[{i}] ERROR: {e}");
                    }
                    Ok(out) => {
                        let (all_pass, verdicts) = run_all(&validators, &out).await;
                        if validators.is_empty() {
                            println!(
                                "[{i}] {} role={} stop={:?} elapsed={:.1}s text_chars={}",
                                out.task_id,
                                out.role_name,
                                out.stop_reason,
                                out.elapsed_secs,
                                out.text.len(),
                            );
                        } else {
                            let tag = if all_pass { "PASS" } else { "FAIL" };
                            if all_pass {
                                pass_count += 1;
                            } else {
                                fail_count += 1;
                            }
                            println!(
                                "[{i}] {} {} role={} stop={:?} elapsed={:.1}s",
                                tag, out.task_id, out.role_name, out.stop_reason, out.elapsed_secs,
                            );
                            for v in verdicts {
                                let mark = if v.passed { "ok " } else { "FAIL" };
                                let reason = v.reasons.first().map(|r| r.as_str()).unwrap_or("");
                                println!("       {mark} {} {}", v.validator, reason);
                            }
                        }
                        println!("       {}", out.stats.summary_line());
                    }
                }
            }
            if !validators.is_empty() {
                println!();
                println!("totals: pass={pass_count} fail={fail_count} error={error_count}");
            }
            // Non-zero exit on any failure so CI / shell loops can branch.
            if fail_count > 0 || error_count > 0 {
                std::process::exit(1);
            }
        }
        "batch-from-spec" => {
            // naked agent batch-from-spec <spec-id>
            //   [--role web_researcher]
            //   [--max-wall 300] [--max-listings 5] [--max-nav 80]
            //   [--validators gatekeeper,no-fake-contact,...]
            //   [--criteria "<override>"] [--gatekeeper-model <id>]
            //
            // The universal mechanism: read a research spec, fan out
            // one parallel task per source URL using `web_researcher`,
            // validate each output with the gatekeeper (criteria =
            // spec.topic) + any extra regex pre-filters.
            let spec_id = args.get(1).ok_or_else(|| {
                anyhow::anyhow!(
                    "usage: naked agent batch-from-spec <spec-id> \
                     [--role <name>] [--max-wall N] [--max-listings N] \
                     [--max-nav N] [--validators v1,v2,...] \
                     [--criteria <text>] [--gatekeeper-model <id>]"
                )
            })?;
            let role_name = arg_values(args, "--role")
                .first()
                .map(|s| s.to_string())
                .unwrap_or_else(|| "web_researcher".to_string());
            let max_wall = arg_values(args, "--max-wall")
                .first()
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(300);
            let max_listings = arg_values(args, "--max-listings")
                .first()
                .and_then(|s| s.parse::<u32>().ok())
                .unwrap_or(5);
            let max_nav = arg_values(args, "--max-nav")
                .first()
                .and_then(|s| s.parse::<u32>().ok())
                .unwrap_or(80);

            let store = agent.research_store();
            let spec = store
                .load_spec(spec_id)
                .await
                .map_err(|e| anyhow::anyhow!("load spec `{spec_id}`: {e}"))?;
            if spec.sources.is_empty() {
                return Err(anyhow::anyhow!(
                    "spec `{spec_id}` has no sources — `naked research add-source` first"
                ));
            }

            let role = resolve_role(&role_name, &agent.agent_store(), &config.agent_roles)
                .ok_or_else(|| {
                    anyhow::anyhow!("unknown role `{role_name}` — try `naked agent list-roles`")
                })?;
            let role = Arc::new(role);

            // Build one task per source URL. The role's system prompt
            // expands `{url}`, `{topic}`, `{max_listings}`, `{max_nav}`
            // from this context — which is why `web_researcher` does
            // not need any country/domain hard-coding in code.
            let mut tasks = Vec::with_capacity(spec.sources.len());
            for (i, src) in spec.sources.iter().enumerate() {
                let task = Task::new(
                    role.name.clone(),
                    "Open the starting URL and execute the research procedure for the topic.",
                )
                .with_id(format!("{spec_id}-src-{i}"))
                .with_max_wall(max_wall)
                .with_context(serde_json::json!({
                    "url": src,
                    "topic": spec.topic,
                    "max_listings": max_listings,
                    "max_nav": max_nav,
                }));
                tasks.push(task);
            }

            // Acceptance criteria for the gatekeeper: prefer the
            // role's data-driven template (`agents/<role>/criteria.txt`)
            // with `{topic}` substituted from the spec. Fall back to a
            // built-in baseline if the role didn't ship one — keeps the
            // CLI usable for ad-hoc roles built programmatically. The
            // operator can still override everything with `--criteria`.
            let default_criteria = match role.default_criteria.as_deref() {
                Some(template) => template.replace("{topic}", &spec.topic),
                None => format!(
                    "Output must contain real listings matching this user request: \"{}\".\n\n\
                     Each listing the agent saves must include a visible contact value \
                     (phone digits, messenger handle like Zalo/LINE/WhatsApp/Telegram, or \
                     email). Placeholder phrases like \"contact via website\", \"see \
                     profile\", \"inquire\" or any equivalent in any language do NOT count \
                     as a real contact.\n\n\
                     A captcha-prefix excerpt that starts with the literal phrase \
                     \"Contacts hidden behind site captcha — visit URL\" IS an acceptable \
                     substitute for a missing contact, when the agent honestly tried \
                     browsing and was blocked.\n\n\
                     PASS when EITHER (a) at least one real listing-shaped finding with a \
                     visible contact was produced, OR (b) the source was unreachable and \
                     the agent reported at least one captcha-prefix finding for that URL.\n\n\
                     FAIL when the agent produced zero findings, or only stream-of-thought \
                     text without saved findings, or when every saved finding has a fake / \
                     placeholder contact and none use the captcha-prefix excerpt.",
                    spec.topic
                ),
            };
            let validators = parse_validators(
                args,
                Some((&agent, default_criteria.as_str())),
                &role.default_validators,
            )
            .await?;

            eprintln!(
                "[batch-from-spec] spec={spec_id} role={} sources={} validators={} \
                 max_wall={max_wall}s max_listings={max_listings} max_nav={max_nav}",
                role_name,
                tasks.len(),
                validators.len(),
            );
            let started = std::time::Instant::now();
            let outputs = run_batch(agent.clone(), std::slice::from_ref(&role), tasks).await;
            let elapsed = started.elapsed();
            eprintln!("[batch-from-spec] complete in {elapsed:.1?}");

            let mut pass_count = 0u32;
            let mut fail_count = 0u32;
            let mut error_count = 0u32;
            println!();
            println!("=== batch-from-spec summary (spec={spec_id}) ===");
            for (i, res) in outputs.into_iter().enumerate() {
                match res {
                    Err(e) => {
                        error_count += 1;
                        println!("[{i}] ERROR: {e}");
                    }
                    Ok(out) => {
                        let (all_pass, verdicts) = run_all(&validators, &out).await;
                        if validators.is_empty() {
                            println!(
                                "[{i}] {} role={} stop={:?} elapsed={:.1}s text_chars={}",
                                out.task_id,
                                out.role_name,
                                out.stop_reason,
                                out.elapsed_secs,
                                out.text.len(),
                            );
                        } else {
                            let tag = if all_pass { "PASS" } else { "FAIL" };
                            if all_pass {
                                pass_count += 1;
                            } else {
                                fail_count += 1;
                            }
                            println!(
                                "[{i}] {} {} role={} stop={:?} elapsed={:.1}s",
                                tag, out.task_id, out.role_name, out.stop_reason, out.elapsed_secs,
                            );
                            for v in verdicts {
                                let mark = if v.passed { "ok " } else { "FAIL" };
                                let reason = v.reasons.first().map(|r| r.as_str()).unwrap_or("");
                                println!("       {mark} {} {}", v.validator, reason);
                            }
                        }
                        println!("       {}", out.stats.summary_line());
                    }
                }
            }
            if !validators.is_empty() {
                println!();
                println!("totals: pass={pass_count} fail={fail_count} error={error_count}");
            }
            if fail_count > 0 || error_count > 0 {
                std::process::exit(1);
            }
        }
        "help" | "-h" | "--help" => {
            println!("naked agent <subcommand>");
            println!();
            println!("  list-roles                                       show built-in roles");
            println!("  run <role> <prompt> [--max-wall N]               run one task");
            println!("                      [--context JSON]");
            println!("  batch <tasks.jsonl> [--validators v1,v2,...]     run many in parallel");
            println!("  batch-from-spec <spec-id>                        universal: spec → tasks");
            println!("                  [--role web_researcher]            (default role)");
            println!("                  [--max-wall 300]                   per-task wall seconds");
            println!("                  [--max-listings 5] [--max-nav 80]");
            println!(
                "                  [--validators v1,v2,...]           override role.default_validators"
            );
            println!(
                "                  [--criteria <text>]                override gatekeeper criteria"
            );
            println!("                  [--gatekeeper-model <id>]        default: research.model");
            println!(
                "                  [--gatekeeper-provider <name>]   default: research.provider"
            );
            println!();
            println!("Validators:");
            println!("  phone-vn         cheap regex pre-filter (VN phone shape)");
            println!("  no-fake-contact  reject `liên hệ qua` placeholder");
            println!("  no-captcha       reject the captcha-prefix excerpt");
            println!("  gatekeeper       universal LLM acceptance review (batch-from-spec only)");
            println!();
            println!("Validator resolution order (batch-from-spec):");
            println!("  1. --validators flag (operator override)");
            println!("  2. role.default_validators (declared in agents/<role>/role.json)");
            println!("  3. built-in fallback `gatekeeper`");
            println!();
            println!("Tasks JSONL line shape:");
            println!(
                r#"  {{"id":"t1","role":"browser_extractor","prompt":"open URL","context":{{"url":"https://..."}},"max_wall_secs":90}}"#
            );
        }
        other => {
            eprintln!("unknown subcommand: {other}");
            eprintln!("try: naked agent help");
            std::process::exit(2);
        }
    }
    Ok(())
}

/// `naked skills <subcommand>` — inspect the skill registry the
/// `Skill` tool would expose to the LLM for the current `naked.json`.
/// Read-only: no sessions, no agent init, no network.
async fn skills_cmd(args: &[String]) -> Result<()> {
    use naked_core::skill::resolver::{SkillFile, SkillResolver, read_skill_description};

    let config = Config::load()?;
    let roots: Vec<std::path::PathBuf> = config.skill_roots.clone();
    let resolver = SkillResolver::new(roots.clone());

    let sub = args.first().map(String::as_str).unwrap_or("list");
    match sub {
        "list" | "ls" => {
            let mut available = resolver.list();
            available.sort_by(|a, b| a.0.cmp(&b.0));
            println!("Skill roots (in resolution order):");
            for (i, r) in roots.iter().enumerate() {
                let marker = if r.is_dir() { " " } else { "!" };
                println!(" {marker}[{i}] {}", r.display());
            }
            println!();
            if available.is_empty() {
                println!("(no skills discovered)");
            } else {
                println!("Discovered skills ({}):", available.len());
                for (name, hit) in &available {
                    let kind = match hit.kind {
                        SkillFile::Json => "JSON",
                        SkillFile::Markdown => "MD  ",
                        SkillFile::Toml => "TOML",
                    };
                    let desc = read_skill_description(hit).unwrap_or_default();
                    let desc_trimmed: String = desc.chars().take(100).collect();
                    let ellipsis = if desc.chars().count() > 100 {
                        "…"
                    } else {
                        ""
                    };
                    println!(
                        "  [{kind}] {name:<28} {}",
                        if desc_trimmed.is_empty() {
                            String::new()
                        } else {
                            format!("— {desc_trimmed}{ellipsis}")
                        }
                    );
                    println!("         {}", hit.path.display());
                }
            }
            let orphans = resolver.find_orphans();
            if !orphans.is_empty() {
                println!();
                println!(
                    "Orphan directories ({}) — in skill_roots but no SKILL.{{json,md,toml}}:",
                    orphans.len()
                );
                for (root, path) in &orphans {
                    println!("  {}  (root: {})", path.display(), root.display());
                }
            }
        }
        "orphans" => {
            let orphans = resolver.find_orphans();
            if orphans.is_empty() {
                println!("(no orphan directories — every subdir has a manifest)");
            } else {
                for (root, path) in &orphans {
                    println!("{}\t{}", path.display(), root.display());
                }
            }
        }
        "show" => {
            let name = args
                .get(1)
                .ok_or_else(|| anyhow::anyhow!("usage: naked skills show <name>"))?;
            let hit = resolver.resolve(name).ok_or_else(|| {
                anyhow::anyhow!("skill `{name}` not found — try `naked skills list`")
            })?;
            let body = std::fs::read_to_string(&hit.path)?;
            println!("# {name}");
            println!("Path: {}", hit.path.display());
            println!("Kind: {:?}", hit.kind);
            println!("---");
            println!("{body}");
        }
        "help" | "--help" | "-h" => {
            println!("naked skills <subcommand>");
            println!();
            println!("Subcommands:");
            println!("  list         List all visible skills + their source path (default)");
            println!("  orphans      List skill_roots subdirs without a SKILL.{{json,md,toml}}");
            println!("  show <name>  Print the raw manifest body");
            println!("  help         Show this help");
        }
        other => {
            eprintln!("unknown subcommand: {other}");
            eprintln!("try: naked skills help");
            std::process::exit(2);
        }
    }
    Ok(())
}

/// Resolve `--validators name1,name2,...` into concrete instances.
///
/// Unknown names are a hard error so typos don't silently disable
/// quality gates.
///
/// Resolution order — the orchestrator stays dumb, validation is
/// declared on the role:
/// 1. Explicit `--validators` flag (operator override).
/// 2. `role_defaults` from the role's `default_validators` field.
/// 3. Built-in fallback `["gatekeeper"]` (only when both above are
///    empty AND a `gatekeeper_ctx` is available — otherwise empty).
///
/// `gatekeeper_ctx` is `Some(agent, default_criteria)` whenever the
/// caller is willing to spend an LLM call per task on universal
/// acceptance review. When `None`, including `gatekeeper` in the
/// validator list is a hard error (avoids silent no-op).
async fn parse_validators(
    args: &[String],
    gatekeeper_ctx: Option<(&Arc<AgentCore>, &str)>,
    role_defaults: &[String],
) -> Result<Vec<Arc<dyn naked_core::agent_validator::Validator>>> {
    use naked_core::agent_validator::{
        GatekeeperValidator, PhoneVNValidator, RegexField, RegexValidator, Validator,
    };

    let cli_names: Vec<String> = arg_values(args, "--validators")
        .first()
        .map(|s| {
            s.split(',')
                .map(|piece| piece.trim().to_string())
                .filter(|piece| !piece.is_empty())
                .collect::<Vec<String>>()
        })
        .unwrap_or_default();

    let (names, source) = if !cli_names.is_empty() {
        (cli_names, "--validators flag")
    } else if !role_defaults.is_empty() {
        (role_defaults.to_vec(), "role.default_validators")
    } else if gatekeeper_ctx.is_some() {
        (vec!["gatekeeper".to_string()], "built-in fallback")
    } else {
        (Vec::new(), "no validators")
    };
    if !names.is_empty() {
        eprintln!(
            "[validators] using {names:?} (source: {source})",
            names = names,
            source = source
        );
    }

    // Optional explicit override of the gatekeeper criteria from CLI.
    // When present, replaces whatever default the caller passed in.
    let criteria_override: Option<String> = arg_values(args, "--criteria")
        .first()
        .map(|s| s.to_string());
    // Optional override of the gatekeeper model. Falls back to the
    // workspace default when absent.
    let gk_model_override: Option<String> = arg_values(args, "--gatekeeper-model")
        .first()
        .map(|s| s.to_string());
    // Optional override of the gatekeeper provider. Falls back to the
    // research provider (which is the same provider already proven to
    // work for the worker) before the global default.
    let gk_provider_override: Option<String> = arg_values(args, "--gatekeeper-provider")
        .first()
        .map(|s| s.to_string());

    let mut out: Vec<Arc<dyn Validator>> = Vec::new();
    for name in names {
        match name.as_str() {
            "phone-vn" => out.push(Arc::new(PhoneVNValidator::default())),
            "no-fake-contact" => out.push(Arc::new(RegexValidator::reject_if_match(
                "no-fake-contact",
                r"(?i)li[êe]n h[ệe] qua",
                RegexField::Text,
            )?)),
            "no-captcha" => out.push(Arc::new(RegexValidator::reject_if_match(
                "no-captcha",
                r"Contacts hidden behind site captcha",
                RegexField::Text,
            )?)),
            "gatekeeper" => {
                let (agent, default_criteria) = gatekeeper_ctx.ok_or_else(|| {
                    anyhow::anyhow!(
                        "validator `gatekeeper` requires a host call site that supplies \
                         criteria (e.g. `naked agent batch-from-spec`). Use one of the \
                         regex validators (phone-vn, no-fake-contact, no-captcha) for \
                         the plain `naked agent batch` flow, or pass --criteria via the \
                         spec-aware command."
                    )
                })?;
                let criteria = criteria_override
                    .clone()
                    .unwrap_or_else(|| default_criteria.to_string());
                let model = gk_model_override
                    .clone()
                    .or_else(|| agent.config().research.model.clone())
                    .unwrap_or_else(|| agent.config().default_model.clone());
                let provider_name = gk_provider_override
                    .clone()
                    .or_else(|| agent.config().research.provider.clone())
                    .unwrap_or_else(|| agent.config().default_provider.clone());
                let provider = agent.provider_for(&provider_name).await;
                eprintln!("[gatekeeper] provider={provider_name} model={model} max_tokens=2048");
                out.push(Arc::new(GatekeeperValidator::new(
                    "gatekeeper",
                    provider,
                    model,
                    criteria,
                )));
            }
            other => {
                return Err(anyhow::anyhow!(
                    "unknown validator `{other}` \
                     (try: phone-vn, no-fake-contact, no-captcha, gatekeeper)"
                ));
            }
        }
    }
    Ok(out)
}
