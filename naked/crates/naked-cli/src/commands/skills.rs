//! `naked skills` subcommand — inspect the skill registry.

use anyhow::Result;
use naked_core::config::Config;

/// `naked skills <subcommand>` — inspect the skill registry the
/// `Skill` tool would expose to the LLM for the current `naked.json`.
/// Read-only: no sessions, no agent init, no network.
pub(crate) async fn skills_cmd(args: &[String]) -> Result<()> {
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
