//! CLI command dispatch — routes top-level subcommands to their handlers.

pub(crate) mod agent;
pub(crate) mod chat;
pub(crate) mod copilot;
pub(crate) mod memory;
pub(crate) mod skills;
pub(crate) mod vacuum;

use anyhow::Result;

/// Collect every value following an occurrence of `flag`.
///
/// Handles repeated flags like `--source URL --source URL2`. Used by
/// every subcommand parser in this binary and re-exported for `research.rs`.
pub(crate) fn arg_values<'a>(args: &'a [String], flag: &str) -> Vec<&'a str> {
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

/// Dispatch top-level args to the appropriate subcommand handler.
pub(crate) async fn run(args: Vec<String>) -> Result<()> {
    let sub = args.get(1).map(|s| s.as_str()).unwrap_or("");

    match sub {
        "--help" | "-h" | "help" => {
            print_usage();
            return Ok(());
        }
        "copilot-login" => return copilot::copilot_login_cmd().await,
        "memory" => return memory::memory_cmd(&args[2..]).await,
        "vacuum-sessions" => return vacuum::vacuum_sessions_cmd(&args[2..]).await,
        "research" => return crate::research::research_cmd(&args[2..]).await,
        "agent" => return agent::agent_cmd(&args[2..]).await,
        "skills" => return skills::skills_cmd(&args[2..]).await,
        _ => {}
    }

    // Default: interactive chat REPL
    chat::chat_repl().await
}

fn print_usage() {
    println!("Usage: naked [subcommand]");
    println!();
    println!("Subcommands:");
    println!("  <none>            Interactive chat REPL (default)");
    println!("  agent             Role-based agent execution and batch runs");
    println!("  copilot-login     GitHub Copilot OAuth device login");
    println!("  memory            Persistent memory management");
    println!("  research          Research subsystem operator commands");
    println!("  skills            Inspect the skill registry");
    println!("  vacuum-sessions   Migrate and GC session artifacts");
    println!();
    println!("Run `naked <subcommand> help` for subcommand-specific usage.");
}
