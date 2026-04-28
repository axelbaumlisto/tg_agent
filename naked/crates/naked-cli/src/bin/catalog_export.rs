//! `naked-catalog-export` — render the generated regions of
//! `skills/model-catalog/SKILL.md` from `naked.json.providers[*].capabilities`.
//!
//! Phase 5 of the Model Capabilities Catalog plan. Preserves
//! everything outside the HTML comment markers (YAML front-matter,
//! hand-curated sections, the "Keeping this catalog fresh" footer).
//!
//! # Usage
//!
//! ```text
//! naked-catalog-export [--dry-run] [--skill-path <PATH>] [--naked-json <PATH>]
//! ```
//!
//! With no flags, the binary reads `naked.json` from `NAKED_CONFIG` or
//! the default lookup path, rewrites `naked/skills/model-catalog/SKILL.md`
//! in place, and exits 0. Pass `--dry-run` to print the rendered body
//! to stdout instead. Exit 2 on missing markers so CI can wire the
//! binary into a lint step.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, Result};

use naked_core::config::Config;
use naked_core::model_catalog::exporter;

fn parse_args() -> Args {
    let mut args = Args::default();
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--dry-run" => args.dry_run = true,
            "--skill-path" => {
                i += 1;
                args.skill_path = raw.get(i).map(PathBuf::from);
            }
            "--naked-json" => {
                i += 1;
                args.naked_json = raw.get(i).map(PathBuf::from);
            }
            "-h" | "--help" => {
                args.help = true;
            }
            unknown => {
                eprintln!("naked-catalog-export: unknown arg `{unknown}`");
                args.help = true;
            }
        }
        i += 1;
    }
    args
}

#[derive(Default)]
struct Args {
    dry_run: bool,
    skill_path: Option<PathBuf>,
    naked_json: Option<PathBuf>,
    help: bool,
}

fn print_help() {
    println!(
        "Usage: naked-catalog-export [--dry-run] [--skill-path PATH] [--naked-json PATH]\n\
         \n\
         Render the generated sections of skills/model-catalog/SKILL.md from\n\
         naked.json.providers[*].capabilities. Preserves every region outside\n\
         the HTML comment markers.\n"
    );
}

fn run() -> Result<ExitCode> {
    let args = parse_args();
    if args.help {
        print_help();
        return Ok(ExitCode::SUCCESS);
    }

    let config = if let Some(p) = &args.naked_json {
        Config::from_json_file(p)
            .with_context(|| format!("loading {}", p.display()))?
    } else {
        Config::load().context("loading naked.json")?
    };

    let skill_path = args.skill_path.unwrap_or_else(|| {
        let root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        root.join("naked").join("skills").join("model-catalog").join("SKILL.md")
    });

    if args.dry_run {
        let rendered = exporter::render_sections(&config);
        println!("--- quick-decision ---\n{}", rendered.quick_decision);
        println!("--- provider-details ---\n{}", rendered.provider_details);
        println!("--- known-failure-modes ---\n{}", rendered.failure_modes);
        return Ok(ExitCode::SUCCESS);
    }

    match exporter::export_to_file(&config, &skill_path) {
        Ok(true) => {
            eprintln!("naked-catalog-export: updated {}", skill_path.display());
            Ok(ExitCode::SUCCESS)
        }
        Ok(false) => {
            eprintln!(
                "naked-catalog-export: {} is already up to date",
                skill_path.display()
            );
            Ok(ExitCode::SUCCESS)
        }
        Err(e) if e.kind() == std::io::ErrorKind::InvalidData => {
            eprintln!("naked-catalog-export: {e}");
            Ok(ExitCode::from(2))
        }
        Err(e) => Err(anyhow::anyhow!(e)
            .context(format!("exporting to {}", skill_path.display()))),
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(e) => {
            eprintln!("naked-catalog-export: {e:#}");
            ExitCode::FAILURE
        }
    }
}
