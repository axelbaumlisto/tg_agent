//! `naked memory` subcommand — persistent memory management.

use anyhow::Result;

pub(crate) async fn memory_cmd(args: &[String]) -> Result<()> {
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
