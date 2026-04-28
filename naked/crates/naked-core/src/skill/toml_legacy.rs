//! Legacy `SKILL.toml` → advisory-manual bridge.
//!
//! Some skills under `~/.zeroclaw/workspace/skills/` were authored
//! for a different runtime that used TOML manifests with a
//! `[skill]` section plus one or more `[[tools]]` entries listing
//! shell commands with `{placeholder}` templates. Our `SkillSpec`
//! (see [`super::spec`]) has a different shape and dispatch model
//! (linear `steps` executed via the in-process ToolDispatcher).
//!
//! Full semantic bridging (map `[[tools]]` commands onto real
//! `SkillStep::Bash` executions with argument substitution) would
//! require a bigger rewrite. In practice the agent just needs to
//! *discover* these skills, read their description, and know which
//! shell commands to run — the agent already has a `bash` tool to
//! call them itself.
//!
//! So this module does the pragmatic thing:
//!
//! 1. Parse the `[skill]` header for `name` + `description` +
//!    optional `prompts`.
//! 2. Parse each `[[tools]]` entry for `name`, `description`,
//!    `command`, and `[tools.args]` descriptions.
//! 3. Render the whole thing as a markdown *manual* that the
//!    `Skill` tool returns as an advisory body. The agent reads it,
//!    sees the commands, and invokes them through `bash`.
//!
//! This covers `coder`, `provider-manager`, `telegram-mcp`,
//! `erp-analyst`, `github-grep` without touching the skill authors'
//! TOML files or the original runtime's behaviour.

use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Debug, Deserialize)]
pub struct TomlSkillManifest {
    #[serde(default)]
    pub skill: SkillHeader,
    #[serde(default)]
    pub tools: Vec<ToolEntry>,
}

#[derive(Debug, Deserialize, Default)]
pub struct SkillHeader {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub prompts: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct ToolEntry {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub command: Option<String>,
    #[serde(default)]
    pub args: BTreeMap<String, toml::Value>,
}

impl TomlSkillManifest {
    /// Parse a TOML string. Returns None if the file is malformed;
    /// callers should fall back to showing the bare skill name.
    pub fn parse(raw: &str) -> Option<Self> {
        toml::from_str(raw).ok()
    }

    /// Render a markdown *manual* the agent can read to learn what
    /// shell commands the skill exposes. The agent is expected to
    /// pick the relevant one and run it through its existing
    /// `bash` tool — we do not execute commands on behalf of TOML
    /// skills (see module docs).
    pub fn render_manual(&self) -> String {
        let mut out = String::new();
        if !self.skill.name.is_empty() {
            out.push_str(&format!("# Skill: {}\n\n", self.skill.name));
        }
        if !self.skill.description.is_empty() {
            out.push_str(&self.skill.description);
            out.push_str("\n\n");
        }
        if !self.skill.prompts.is_empty() {
            out.push_str("## When / how to use\n\n");
            for p in &self.skill.prompts {
                out.push_str(p.trim());
                out.push_str("\n\n");
            }
        }
        if !self.tools.is_empty() {
            out.push_str("## Available commands (invoke via the `bash` tool)\n\n");
            for t in &self.tools {
                if !t.name.is_empty() {
                    out.push_str(&format!("### {}\n", t.name));
                }
                if !t.description.is_empty() {
                    out.push_str(t.description.trim());
                    out.push_str("\n\n");
                }
                if let Some(cmd) = &t.command {
                    out.push_str("Shell template:\n\n");
                    out.push_str("```bash\n");
                    out.push_str(cmd.trim());
                    out.push_str("\n```\n\n");
                }
                if !t.args.is_empty() {
                    out.push_str("Arguments:\n\n");
                    for (k, v) in &t.args {
                        let s = match v {
                            toml::Value::String(s) => s.clone(),
                            other => other.to_string(),
                        };
                        out.push_str(&format!("- `{}`: {}\n", k, s.trim()));
                    }
                    out.push('\n');
                }
            }
        }
        out.push_str(
            "> This skill uses a legacy `SKILL.toml` manifest. The host cannot \
             execute its commands directly — run the shell templates above \
             yourself via the `bash` tool, substituting `{placeholders}` with \
             real values.\n",
        );
        out
    }

    /// The one-line description used in the `Skill` tool catalog the
    /// LLM sees. Falls back to empty string if missing.
    pub fn short_description(&self) -> String {
        self.skill.description.trim().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_toml() {
        let raw = r#"
[skill]
name = "demo"
description = "Does demo things"
"#;
        let m = TomlSkillManifest::parse(raw).expect("should parse");
        assert_eq!(m.skill.name, "demo");
        assert_eq!(m.skill.description, "Does demo things");
        assert!(m.tools.is_empty());
        assert_eq!(m.short_description(), "Does demo things");
    }

    #[test]
    fn parses_with_tools_and_prompts() {
        let raw = r#"
[skill]
name = "coder"
description = "Coding assistant"
prompts = ["Use me to code"]

[[tools]]
name = "code"
description = "Run a coding task"
kind = "shell"
command = "python3 run.py --task {task}"

[tools.args]
task = "Coding request"
"#;
        let m = TomlSkillManifest::parse(raw).expect("should parse");
        assert_eq!(m.tools.len(), 1);
        assert_eq!(m.tools[0].name, "code");
        assert_eq!(
            m.tools[0].command.as_deref(),
            Some("python3 run.py --task {task}")
        );
        let manual = m.render_manual();
        assert!(manual.contains("# Skill: coder"));
        assert!(manual.contains("Use me to code"));
        assert!(manual.contains("### code"));
        assert!(manual.contains("python3 run.py --task {task}"));
        assert!(manual.contains("`task`: Coding request"));
        assert!(manual.contains("legacy `SKILL.toml`"));
    }

    #[test]
    fn malformed_toml_returns_none() {
        assert!(TomlSkillManifest::parse("[skill\nname=unclosed").is_none());
    }

    #[test]
    fn short_description_falls_back_to_empty_when_missing() {
        let raw = r#"
[skill]
name = "x"
"#;
        let m = TomlSkillManifest::parse(raw).unwrap();
        assert_eq!(m.short_description(), "");
    }
}
