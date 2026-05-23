//! Interactive chat REPL and turn-event rendering.

use std::io::{self, BufRead, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::Result;
use naked_core::AgentCore;
use naked_core::config::Config;
use naked_core::types::{AgentEvent, AgentHandle, PermissionResponse, TurnUsage};
use naked_core::util::head_truncate;

pub(crate) const SPINNER_FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// Run the interactive chat REPL (default mode when no subcommand is given).
pub(crate) async fn chat_repl() -> Result<()> {
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

pub(crate) async fn run_turn(handle: AgentHandle) -> Result<()> {
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
                    eprintln!("\x1b[2m{}\x1b[0m", head_truncate(&output, 500));
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
                            head_truncate(&prompt_preview, 80)
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
                            head_truncate(&input_preview, 60)
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
            AgentEvent::CycleRestarted {
                cycle_number,
                archived_messages,
                archive_path,
            } => {
                eprintln!(
                    "\x1b[36m[cycle restart #{cycle_number}: {archived_messages} messages archived → {archive_path}]\x1b[0m"
                );
            }
            AgentEvent::ToolOutput { chunk, .. } => {
                // Show last line of bash output inline:
                if let Some(last) = chunk.lines().last() {
                    eprint!("\r\x1b[2K\x1b[90m  > {last}\x1b[0m");
                }
            }
            AgentEvent::SteerReceived { text, msg_ids: _ } => {
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

pub(crate) fn cli_input_preview(input: &serde_json::Value, max_len: usize) -> String {
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

pub(crate) fn val_short(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max).collect();
        format!("{truncated}…")
    }
}
