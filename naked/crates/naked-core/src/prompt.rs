use std::path::{Path, PathBuf};

const DEFAULT_PROMPT_FILENAME: &str = "system_prompt.md";
const MAX_INSTRUCTION_FILE_CHARS: usize = 16_000;

/// Resolves the system prompt by checking project-local override first,
/// then falling back to the global default.
///
/// Resolution order:
/// 1. `{workspace}/.naked/system_prompt.md` (project override)
/// 2. `{global_path}` (from config or `~/.naked/system_prompt.md`)
/// 3. Built-in default prompt
pub fn resolve_system_prompt(workspace: &Path, global_path: Option<&Path>) -> String {
    let project_prompt = workspace.join(".naked").join(DEFAULT_PROMPT_FILENAME);
    if let Some(content) = try_read_prompt(&project_prompt) {
        return content;
    }

    if let Some(global) = global_path
        && let Some(content) = try_read_prompt(global)
    {
        return content;
    }

    let home_prompt = dirs_home().join(".naked").join(DEFAULT_PROMPT_FILENAME);
    if let Some(content) = try_read_prompt(&home_prompt) {
        return content;
    }

    default_system_prompt(workspace)
}

/// Resolve the live effective system prompt for a session, giving priority
/// to the per-workspace persona override (`<workspace>/.naked/system_prompt.md`)
/// even when the `session_meta` snapshot still carries the original prompt
/// from session creation.
///
/// Background: Telegram personas (Income group, DMs) can have their own
/// workspace with `.naked/system_prompt.md`. If this file changes, existing
/// long-lived sessions would silently keep using the stale snapshot captured
/// in `SessionMetadata` at creation. This helper reads the override every
/// turn so persona authoring stays live.
///
/// Returns `Ok(prompt)` where `prompt` is:
/// 1. `<workspace>/.naked/system_prompt.md` if present and non-empty,
/// 2. else `meta_snapshot` (what's already persisted on disk).
pub fn effective_system_prompt(workspace: &Path, meta_snapshot: &str) -> std::io::Result<String> {
    let project_prompt = workspace.join(".naked").join(DEFAULT_PROMPT_FILENAME);
    if let Some(content) = try_read_prompt(&project_prompt) {
        return Ok(content);
    }
    Ok(meta_snapshot.to_string())
}

fn try_read_prompt(path: &Path) -> Option<String> {
    let content = std::fs::read_to_string(path).ok()?;
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.len() > MAX_INSTRUCTION_FILE_CHARS {
        // Snap to char boundary to avoid panic on multi-byte UTF-8.
        let mut end = MAX_INSTRUCTION_FILE_CHARS;
        while end > 0 && !trimmed.is_char_boundary(end) {
            end -= 1;
        }
        Some(trimmed[..end].to_string())
    } else {
        Some(trimmed.to_string())
    }
}

/// Build the environment section appended to the system prompt.
pub fn environment_section(workspace: &Path) -> String {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    let date = chrono::Utc::now().format("%Y-%m-%d").to_string();
    let cwd = workspace.display();

    let mut section = format!(
        "---\n\
         Environment:\n\
         - OS: {os} ({arch})\n\
         - Date: {date}\n\
         - Working directory: {cwd}"
    );

    if let Some(git) = git_context(workspace) {
        section.push_str("\n\n");
        section.push_str(&git);
    }

    if let Some(instructions) = load_project_instructions(workspace) {
        section.push_str("\n\n---\nProject instructions:\n");
        section.push_str(&instructions);
    }

    section
}

const MAX_GIT_CONTEXT: usize = 2048;

fn git_context(workspace: &Path) -> Option<String> {
    let run = |args: &[&str]| -> Option<String> {
        let out = std::process::Command::new("git")
            .args(args)
            .current_dir(workspace)
            .output()
            .ok()?;
        if out.status.success() {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if s.is_empty() { None } else { Some(s) }
        } else {
            None
        }
    };

    let branch = run(&["rev-parse", "--abbrev-ref", "HEAD"])?;
    let mut lines = vec![format!("Git: branch={branch}")];

    if let Some(status) = run(&["status", "--porcelain"]) {
        let status_lines: Vec<&str> = status.lines().collect();
        let shown = if status_lines.len() > 20 {
            format!(
                "{}\n  ... and {} more files",
                status_lines[..20].join("\n"),
                status_lines.len() - 20
            )
        } else {
            status_lines.join("\n")
        };
        lines.push(format!("Changed files:\n{shown}"));
    }

    if let Some(diff_stat) = run(&["diff", "--stat", "--stat-width=60"]) {
        // Walk char boundaries — non-ASCII paths in `git diff --stat` are
        // rare but possible (e.g. unicode filenames), and a byte slice
        // would panic inside a codepoint.
        let truncated = if diff_stat.chars().count() > MAX_GIT_CONTEXT {
            let mut t: String = diff_stat.chars().take(MAX_GIT_CONTEXT).collect();
            t.push_str("...");
            t
        } else {
            diff_stat
        };
        lines.push(format!("Diff summary:\n{truncated}"));
    }

    Some(lines.join("\n"))
}

const MAX_PER_FILE_CHARS: usize = 4_000;
const MAX_TOTAL_INSTRUCTION_CHARS: usize = 12_000;

const INSTRUCTION_FILENAMES: &[&str] = &[
    "AGENTS.md",
    "CLAUDE.md",
    ".naked/instructions.md",
    ".cursor/rules/instructions.md",
];

// B77: hermetic tests call this helper directly; it must collect only `dir`
// and must not walk ancestors. The public function below owns the walk.
fn collect_instructions_in_dir(
    workspace: &Path,
    dir: &Path,
    seen: &mut std::collections::HashSet<PathBuf>,
    remaining: &mut usize,
) -> Vec<String> {
    let mut parts = Vec::new();

    for name in INSTRUCTION_FILENAMES {
        let path = dir.join(name);
        let canonical = path.canonicalize().unwrap_or_else(|_| path.clone());
        if !seen.insert(canonical) {
            continue;
        }
        if *remaining == 0 {
            break;
        }
        if let Ok(raw) = std::fs::read_to_string(&path) {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                continue;
            }
            let limit = MAX_PER_FILE_CHARS.min(*remaining);
            // Char-boundary safe truncation — instruction files may
            // contain non-ASCII (Russian/Cyrillic AGENTS.md sections,
            // Vietnamese examples, etc.); a byte slice would panic.
            let content = if trimmed.chars().count() > limit {
                let mut t: String = trimmed.chars().take(limit).collect();
                t.push_str("\n\n[truncated]");
                t
            } else {
                trimmed.to_string()
            };
            *remaining = remaining.saturating_sub(content.len());
            parts.push(format!(
                "# {}\n{}",
                path.strip_prefix(workspace).unwrap_or(&path).display(),
                content
            ));
        }
    }

    parts
}

/// Walk from workspace upward, collecting instruction files with dedup and budgets.
fn load_project_instructions(workspace: &Path) -> Option<String> {
    let mut seen = std::collections::HashSet::new();
    let mut parts = Vec::new();
    let mut remaining = MAX_TOTAL_INSTRUCTION_CHARS;

    let mut dir = Some(workspace);
    while let Some(current) = dir {
        parts.extend(collect_instructions_in_dir(
            workspace,
            current,
            &mut seen,
            &mut remaining,
        ));
        dir = current.parent();
        if dir == Some(Path::new("")) || dir == Some(Path::new("/")) {
            break;
        }
    }

    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n\n"))
    }
}

fn default_system_prompt(_workspace: &Path) -> String {
    "You are a coding assistant. You help the user with software engineering tasks.\n\
     You have access to tools for reading/writing files, running commands, and searching code.\n\
     \n\
     Read code before editing. Make minimal, targeted changes. Validate your work."
        .to_string()
}

fn dirs_home() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_prompt_is_base_text_only() {
        let prompt = default_system_prompt(Path::new("/tmp/test"));
        assert!(prompt.contains("coding assistant"));
        assert!(
            !prompt.contains("Working directory"),
            "environment_section must not be in default_system_prompt"
        );
    }

    #[test]
    fn resolve_falls_back_when_no_overrides() {
        let prompt = resolve_system_prompt(Path::new("/nonexistent"), None);
        assert!(
            !prompt.is_empty(),
            "should return some prompt even without overrides"
        );
    }

    #[test]
    fn resolve_uses_project_override() {
        let dir = tempfile::tempdir().unwrap();
        let naked_dir = dir.path().join(".naked");
        std::fs::create_dir(&naked_dir).unwrap();
        std::fs::write(naked_dir.join("system_prompt.md"), "custom project prompt").unwrap();

        let prompt = resolve_system_prompt(dir.path(), None);
        assert_eq!(prompt, "custom project prompt");
    }

    #[test]
    fn resolve_uses_global_path() {
        let dir = tempfile::tempdir().unwrap();
        let global_file = dir.path().join("global_prompt.md");
        std::fs::write(&global_file, "global prompt text").unwrap();

        let prompt = resolve_system_prompt(Path::new("/nonexistent"), Some(&global_file));
        assert_eq!(prompt, "global prompt text");
    }

    #[test]
    fn resolve_project_overrides_global() {
        let dir = tempfile::tempdir().unwrap();
        let naked_dir = dir.path().join(".naked");
        std::fs::create_dir(&naked_dir).unwrap();
        std::fs::write(naked_dir.join("system_prompt.md"), "project wins").unwrap();

        let global_file = dir.path().join("global.md");
        std::fs::write(&global_file, "global loses").unwrap();

        let prompt = resolve_system_prompt(dir.path(), Some(&global_file));
        assert_eq!(prompt, "project wins");
    }

    #[test]
    fn environment_section_contains_os_info() {
        let section = environment_section(Path::new("/tmp/ws"));
        assert!(section.contains("OS:"));
        assert!(section.contains("Working directory: /tmp/ws"));
        assert!(section.contains("Date:"));
    }

    #[test]
    fn environment_section_loads_agents_md() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("AGENTS.md"), "# Project Rules\nDo good.").unwrap();

        let section = environment_section(dir.path());
        assert!(section.contains("Project instructions"));
        assert!(section.contains("Do good"));
    }

    #[test]
    fn instruction_discovery_respects_budget() {
        let dir = tempfile::tempdir().unwrap();
        let big = "x".repeat(MAX_TOTAL_INSTRUCTION_CHARS + 500);
        std::fs::write(dir.path().join("AGENTS.md"), &big).unwrap();

        let result = load_project_instructions(dir.path()).unwrap();
        assert!(result.len() <= MAX_TOTAL_INSTRUCTION_CHARS + 200);
        assert!(result.contains("[truncated]"));
    }

    #[test]
    fn instruction_discovery_deduplicates() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("AGENTS.md"), "unique content").unwrap();

        let result = load_project_instructions(dir.path()).unwrap();
        assert_eq!(result.matches("unique content").count(), 1);
    }

    #[test]
    fn git_context_returns_none_for_non_repo() {
        let dir = tempfile::tempdir().unwrap();
        assert!(git_context(dir.path()).is_none());
    }

    #[test]
    fn git_context_returns_branch_for_repo() {
        let dir = tempfile::tempdir().unwrap();
        std::process::Command::new("git")
            .args(["init"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["commit", "--allow-empty", "-m", "init"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        let ctx = git_context(dir.path());
        assert!(ctx.is_some());
        let ctx = ctx.unwrap();
        assert!(ctx.contains("Git: branch="));
    }

    #[test]
    fn environment_section_includes_git_for_repo() {
        let dir = tempfile::tempdir().unwrap();
        std::process::Command::new("git")
            .args(["init"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["commit", "--allow-empty", "-m", "init"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        let section = environment_section(dir.path());
        assert!(section.contains("Git: branch="));
    }

    #[test]
    fn try_read_prompt_skips_empty_files() {
        let dir = tempfile::tempdir().unwrap();
        let empty = dir.path().join("empty.md");
        std::fs::write(&empty, "   \n  ").unwrap();
        assert!(try_read_prompt(&empty).is_none());
    }

    #[test]
    fn try_read_prompt_truncates_long_files() {
        let dir = tempfile::tempdir().unwrap();
        let long = dir.path().join("long.md");
        let content = "x".repeat(MAX_INSTRUCTION_FILE_CHARS + 100);
        std::fs::write(&long, &content).unwrap();
        let result = try_read_prompt(&long).unwrap();
        assert_eq!(result.len(), MAX_INSTRUCTION_FILE_CHARS);
    }

    #[test]
    fn try_read_prompt_truncates_utf8_on_char_boundary() {
        // Russian text: each char = 2 bytes. Truncation at byte limit
        // must not land mid-character.
        let dir = tempfile::tempdir().unwrap();
        let long = dir.path().join("long_ru.md");
        // Fill with 2-byte Cyrillic chars to exceed the limit
        let content = "П".repeat(MAX_INSTRUCTION_FILE_CHARS); // 2 bytes each = 2x limit
        std::fs::write(&long, &content).unwrap();
        let result = try_read_prompt(&long).unwrap();
        assert!(result.len() <= MAX_INSTRUCTION_FILE_CHARS);
        // Must be valid UTF-8 and on a char boundary
        assert!(result.is_char_boundary(result.len()));
        // Every char is 2 bytes, so length must be even
        assert_eq!(result.len() % 2, 0);
    }

    #[test]
    fn try_read_prompt_truncates_mixed_utf8() {
        // Mix of 1-byte ASCII and 3-byte em-dash to hit boundary edge cases.
        let dir = tempfile::tempdir().unwrap();
        let long = dir.path().join("mixed.md");
        // Pattern: "a—" = 4 bytes. Repeat to exceed limit.
        let pattern = "a—"; // 1 + 3 bytes
        let content = pattern.repeat(MAX_INSTRUCTION_FILE_CHARS); // way over
        std::fs::write(&long, &content).unwrap();
        let result = try_read_prompt(&long).unwrap();
        assert!(result.len() <= MAX_INSTRUCTION_FILE_CHARS);
        // Verify it's valid UTF-8 (implicit: it's a String)
        assert!(result.ends_with('a') || result.ends_with('—'));
    }
}

#[cfg(test)]
mod snapshot_tests {
    //! Insta snapshot tests for prompt.rs (T11 of
    //! PLAN_CORE_HARDENING_v2). Captures the *deterministic*
    //! parts of prompt rendering — anything depending on chrono::Utc::now,
    //! `git`, or `$HOME` is intentionally excluded so snapshots are
    //! reproducible across machines.

    use super::*;
    use tempfile::tempdir;

    /// 1. The built-in default prompt is static, lives forever.
    #[test]
    fn snapshot_default_system_prompt() {
        let p = default_system_prompt(std::path::Path::new("/dummy"));
        insta::assert_snapshot!("default_system_prompt", p);
    }

    /// 2. `effective_system_prompt` with NO project override returns
    ///    the meta_snapshot verbatim.
    #[test]
    fn snapshot_effective_no_override_passes_through() {
        let dir = tempdir().unwrap();
        let snap = "Persona: Yumeko. Reply concise.";
        let out = effective_system_prompt(dir.path(), snap).unwrap();
        insta::assert_snapshot!("effective_no_override", out);
    }

    /// 3. `effective_system_prompt` WITH project override picks the
    ///    fresh file content over the meta snapshot.
    #[test]
    fn snapshot_effective_with_override_picks_fresh() {
        let dir = tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join(".naked")).unwrap();
        std::fs::write(
            dir.path().join(".naked/system_prompt.md"),
            "FRESH PERSONA — do X.",
        )
        .unwrap();
        let out = effective_system_prompt(dir.path(), "stale snapshot").unwrap();
        insta::assert_snapshot!("effective_with_override", out);
    }

    /// 4. `try_read_prompt` returns None for missing files.
    #[test]
    fn snapshot_try_read_missing_is_none() {
        let dir = tempdir().unwrap();
        let result = try_read_prompt(&dir.path().join("nope.md"));
        insta::assert_debug_snapshot!("try_read_missing", result);
    }

    /// 5. `try_read_prompt` returns None for empty/whitespace files.
    #[test]
    fn snapshot_try_read_empty_is_none() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("empty.md");
        std::fs::write(&p, "   \n\t\n").unwrap();
        let result = try_read_prompt(&p);
        insta::assert_debug_snapshot!("try_read_empty", result);
    }

    /// 6. `try_read_prompt` truncates long content at MAX_INSTRUCTION_FILE_CHARS.
    #[test]
    fn snapshot_try_read_truncates_long() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("huge.md");
        // 20K chars > 16K limit
        let body = "a".repeat(20_000);
        std::fs::write(&p, body).unwrap();
        let result = try_read_prompt(&p).unwrap();
        // Snapshot the LENGTH, not the giant string.
        insta::assert_snapshot!("try_read_truncated_len", format!("{}", result.len()));
    }

    /// 7. Single-directory instruction collection yields no parts for an empty dir.
    #[test]
    fn collect_instructions_in_dir_empty_is_none() {
        let dir = tempdir().unwrap();
        let mut seen = std::collections::HashSet::new();
        let mut remaining = MAX_TOTAL_INSTRUCTION_CHARS;

        // B77: hermetic — helper must not walk ancestors (including `/tmp`).
        let parts = collect_instructions_in_dir(dir.path(), dir.path(), &mut seen, &mut remaining);

        assert!(
            parts.is_empty(),
            "empty dir yields no instruction parts (B77 hermetic)"
        );
    }

    /// 8. Single-directory collection picks up a workspace-local AGENTS.md.
    #[test]
    fn collect_instructions_in_dir_with_agents_md() {
        let dir = tempdir().unwrap();
        std::fs::write(
            dir.path().join("AGENTS.md"),
            "# Project Rules\n- always read before editing\n- run cargo fmt\n",
        )
        .unwrap();
        let mut seen = std::collections::HashSet::new();
        let mut remaining = MAX_TOTAL_INSTRUCTION_CHARS;

        // B77: hermetic — helper collects ONLY this dir, never ancestors.
        let parts = collect_instructions_in_dir(dir.path(), dir.path(), &mut seen, &mut remaining);

        // workspace == dir, so strip_prefix yields the relative "AGENTS.md" header.
        assert_eq!(parts.len(), 1);
        assert_eq!(
            parts[0],
            "# AGENTS.md\n# Project Rules\n- always read before editing\n- run cargo fmt"
        );
    }

    /// 9. Public instruction loading intentionally walks controlled ancestors.
    #[test]
    fn load_project_instructions_walks_controlled_ancestor() {
        let root = tempdir().unwrap();
        std::fs::write(
            root.path().join("AGENTS.md"),
            "controlled ancestor marker\n",
        )
        .unwrap();
        let child = root.path().join("nested").join("child");
        std::fs::create_dir_all(&child).unwrap();

        let result = load_project_instructions(&child).expect("should find ancestor AGENTS.md");

        // The public fn walks UP from child and finds root/AGENTS.md.
        assert!(
            result.contains("controlled ancestor marker"),
            "public walk must find the controlled ancestor instruction"
        );
        // Ancestor is OUTSIDE workspace (child), so strip_prefix fails → absolute path header.
        assert!(
            result.contains(&root.path().join("AGENTS.md").display().to_string()),
            "ancestor file rendered with absolute path header"
        );
    }
}
