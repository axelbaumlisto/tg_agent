//! `naked tg` — operator surface for the Telegram subsystem.
//!
//! Provides a small set of off-bot commands that exercise the same
//! synthetic-dispatch machinery the live bot uses. Useful for:
//!
//! * **Smoke-testing** the scheduler→dispatch closure plumbing without
//!   sending real Telegram traffic (`tg dispatch-mock`).
//! * **Inspecting** the durable `ChannelSessionMap` snapshot
//!   (`tg sessions list`).
//! * **Dumping** a SyntheticMessage payload for replay or test fixtures
//!   (`tg synthetic <spec-id>`).
//!
//! These commands NEVER touch a live Bot — they are read-only or
//! mock-only. Sending a real Telegram message remains the bot's job.
//!
//! Architecture: this module is a thin presentation layer over
//! `naked_tg::synthetic` and `naked_tg::channel_map`. New operator
//! surfaces should keep that shape — heavy logic stays in those
//! crates, the CLI just parses args and prints results.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};

use naked_tg::channel_map::ChannelSessionMap;
use naked_tg::synthetic::{MockDispatcher, SyntheticMessage};

use crate::commands::arg_values;

/// Top-level dispatch for `naked tg <sub> [args]`.
pub(crate) async fn tg_cmd(args: &[String]) -> Result<()> {
    let sub = args.first().map(|s| s.as_str()).unwrap_or("help");
    match sub {
        "help" | "--help" | "-h" => {
            print_usage();
            Ok(())
        }
        "sessions" => sessions_cmd(&args[1..]).await,
        "synthetic" => synthetic_cmd(&args[1..]),
        "dispatch-mock" => dispatch_mock_cmd(&args[1..]).await,
        unknown => {
            eprintln!("Unknown `naked tg` subcommand: {unknown}\n");
            print_usage();
            std::process::exit(2);
        }
    }
}

fn print_usage() {
    println!("Usage: naked tg <subcommand> [args]");
    println!();
    println!("Subcommands:");
    println!("  sessions list                       Dump ChannelSessionMap snapshot");
    println!("  sessions show --chat <id> [--thread <id>]");
    println!("                                      Show session id for a (chat, thread)");
    println!("  synthetic <spec-id> --chat <id> [--thread <id>]");
    println!("                                      Print the SyntheticMessage as JSON");
    println!("  dispatch-mock <spec-id> --chat <id> [--thread <id>] [--delay-ms <N>]");
    println!("                                      Run the synthetic dispatch through a");
    println!("                                      MockDispatcher; prints captured payload");
    println!();
    println!("All `tg` subcommands are off-bot — no real Telegram traffic.");
}

// ── sessions ──────────────────────────────────────────────────────────

async fn sessions_cmd(args: &[String]) -> Result<()> {
    let sub = args.first().map(|s| s.as_str()).unwrap_or("list");
    let dir = naked_dir()?;
    let map = ChannelSessionMap::open(&dir)
        .await
        .context("opening ChannelSessionMap snapshot")?;

    match sub {
        "list" => {
            let entries = map.all_entries().await;
            if entries.is_empty() {
                println!("(no sessions in {})", dir.display());
                return Ok(());
            }
            println!("# chat_id\tthread_id\tsession_id");
            for (chat_id, thread_id, sid) in entries {
                let thread = if thread_id == 0 {
                    "-".to_string()
                } else {
                    thread_id.to_string()
                };
                println!("{chat_id}\t{thread}\t{sid}");
            }
            Ok(())
        }
        "show" => {
            let chat = required_arg_i64(args, "--chat")?;
            let thread = optional_arg_i32(args, "--thread")?;
            match map.get(chat, thread).await {
                Some(sid) => {
                    println!("{sid}");
                    Ok(())
                }
                None => {
                    eprintln!(
                        "no session for chat={chat} thread={}",
                        thread.map(|t| t.to_string()).unwrap_or_else(|| "-".into())
                    );
                    std::process::exit(1);
                }
            }
        }
        other => bail!("unknown `naked tg sessions` subcommand: {other}"),
    }
}

// ── synthetic (build payload, print as JSON) ──────────────────────────

fn synthetic_cmd(args: &[String]) -> Result<()> {
    let spec_id = first_positional(args).context("expected <spec-id>")?;
    let chat = required_arg_i64(args, "--chat")?;
    let thread = optional_arg_i32(args, "--thread")?;
    let msg = SyntheticMessage::from_scheduler_spec(spec_id, chat, thread);
    let json = serde_json::json!({
        "chat_id": msg.chat_id,
        "thread_id": msg.thread_id,
        "text": msg.text,
        "source": msg.source.label(),
        "spec_id": msg.source.spec_id(),
    });
    println!("{}", serde_json::to_string_pretty(&json)?);
    Ok(())
}

// ── dispatch-mock (drive a synthetic message through MockDispatcher) ──

async fn dispatch_mock_cmd(args: &[String]) -> Result<()> {
    let spec_id = first_positional(args).context("expected <spec-id>")?;
    let chat = required_arg_i64(args, "--chat")?;
    let thread = optional_arg_i32(args, "--thread")?;
    let delay_ms = optional_arg_u64(args, "--delay-ms")?.unwrap_or(0);

    let mock = MockDispatcher::new().with_delay_ms(delay_ms);
    let dispatch = mock.as_fn();
    let msg = SyntheticMessage::from_scheduler_spec(spec_id, chat, thread);
    let started = std::time::Instant::now();
    let latch: naked_tg::synthetic::SessionIdLatch =
        std::sync::Arc::new(tokio::sync::Mutex::new(None));
    let _ = (dispatch)(msg.clone(), latch).await;
    let elapsed_ms = started.elapsed().as_millis();

    let captured = mock.snapshot().await;
    let payload = serde_json::json!({
        "input": {
            "spec_id": spec_id,
            "chat_id": chat,
            "thread_id": thread,
            "delay_ms": delay_ms,
        },
        "captured_count": captured.len(),
        "elapsed_ms": elapsed_ms,
        "captured": captured.iter().map(|m| serde_json::json!({
            "chat_id": m.chat_id,
            "thread_id": m.thread_id,
            "text": m.text,
            "source": m.source.label(),
            "spec_id": m.source.spec_id(),
        })).collect::<Vec<_>>(),
    });
    println!("{}", serde_json::to_string_pretty(&payload)?);

    // Sanity: at minimum, our input message must show up exactly once.
    if captured.len() != 1 {
        bail!(
            "MockDispatcher invariant violation: expected 1 captured \
             message, got {}",
            captured.len()
        );
    }
    Ok(())
}

// ── arg parsing helpers ───────────────────────────────────────────────

fn naked_dir() -> Result<PathBuf> {
    let home = std::env::var("HOME").context("$HOME not set")?;
    Ok(PathBuf::from(home).join(".naked"))
}

/// Return the first arg that isn't a flag (anything starting with `--`)
/// and isn't a flag value (i.e. the token preceding a `--foo VALUE`).
fn first_positional(args: &[String]) -> Option<&str> {
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if a.starts_with("--") {
            // Skip flag + its value (if any).
            i += 2;
            continue;
        }
        return Some(a.as_str());
    }
    None
}

fn required_arg_i64(args: &[String], flag: &str) -> Result<i64> {
    let v = arg_values(args, flag)
        .into_iter()
        .next()
        .with_context(|| format!("missing required {flag}"))?;
    v.parse::<i64>()
        .with_context(|| format!("{flag} value `{v}` is not an i64"))
}

fn optional_arg_i32(args: &[String], flag: &str) -> Result<Option<i32>> {
    match arg_values(args, flag).into_iter().next() {
        None => Ok(None),
        Some(v) => v
            .parse::<i32>()
            .with_context(|| format!("{flag} value `{v}` is not an i32"))
            .map(Some),
    }
}

fn optional_arg_u64(args: &[String], flag: &str) -> Result<Option<u64>> {
    match arg_values(args, flag).into_iter().next() {
        None => Ok(None),
        Some(v) => v
            .parse::<u64>()
            .with_context(|| format!("{flag} value `{v}` is not a u64"))
            .map(Some),
    }
}

// ──────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| (*x).to_string()).collect()
    }

    #[test]
    fn first_positional_skips_flags_and_their_values() {
        let args = s(&["--chat", "1234", "alpha"]);
        assert_eq!(first_positional(&args), Some("alpha"));
    }

    #[test]
    fn first_positional_returns_none_for_only_flags() {
        let args = s(&["--chat", "1234", "--thread", "7"]);
        assert!(first_positional(&args).is_none());
    }

    #[test]
    fn required_arg_i64_parses_negative() {
        let args = s(&["--chat", "-555"]);
        assert_eq!(required_arg_i64(&args, "--chat").unwrap(), -555);
    }

    #[test]
    fn required_arg_i64_missing_errs() {
        let args = s(&[]);
        assert!(required_arg_i64(&args, "--chat").is_err());
    }

    #[test]
    fn optional_arg_i32_absent_yields_none() {
        let args = s(&["--chat", "1"]);
        assert_eq!(optional_arg_i32(&args, "--thread").unwrap(), None);
    }

    #[test]
    fn optional_arg_i32_present_yields_some() {
        let args = s(&["--thread", "42"]);
        assert_eq!(optional_arg_i32(&args, "--thread").unwrap(), Some(42));
    }

    #[test]
    fn optional_arg_u64_invalid_errs() {
        let args = s(&["--delay-ms", "not-a-number"]);
        assert!(optional_arg_u64(&args, "--delay-ms").is_err());
    }

    /// Build a synthetic message via the CLI surface and verify it
    /// drives through MockDispatcher.
    #[tokio::test]
    async fn dispatch_mock_cmd_captures_into_mock() {
        let args = s(&["alpha-spec", "--chat", "9999", "--thread", "3"]);
        // dispatch_mock_cmd writes to stdout — we just assert it doesn't
        // err out. End-to-end capture is covered by synthetic_dispatch_e2e.
        let result = dispatch_mock_cmd(&args).await;
        assert!(result.is_ok(), "dispatch-mock should succeed: {result:?}");
    }

    /// Synthetic command also has to produce a valid JSON payload.
    #[test]
    fn synthetic_cmd_renders_json_for_valid_input() {
        let args = s(&["beta", "--chat", "8888"]);
        // Function prints to stdout via println!; just check it returns Ok.
        assert!(synthetic_cmd(&args).is_ok());
    }

    /// Missing required --chat must error out cleanly.
    #[test]
    fn synthetic_cmd_errs_on_missing_chat() {
        let args = s(&["beta"]);
        assert!(synthetic_cmd(&args).is_err());
    }
}
