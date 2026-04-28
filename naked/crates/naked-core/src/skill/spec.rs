//! JSON-form skill specification (the canonical form going forward).
//!
//! Background — the legacy form is `SKILL.md`: a markdown file the
//! `Skill` tool just dumps verbatim into the tool result so the LLM
//! reads it as instructions. That's still supported and will keep
//! working, but it forces the LLM to interpret the playbook every
//! time, even for steps the host could run deterministically.
//!
//! The JSON form supports three modes (chosen by the skill author):
//!
//! * **`advisory`** — same shape as the markdown form: the host
//!   returns `advisory_body` to the LLM, which then chooses which
//!   tools to call. Use this when the skill is genuinely about
//!   teaching the LLM how to think (e.g. captcha playbook).
//!
//! * **`executable`** — the host walks `steps` itself, dispatches
//!   each tool call, substitutes `{key}` placeholders from `args` and
//!   from previously-saved step outputs, and returns the bundled
//!   result to the LLM as a single tool result. Use this for
//!   deterministic recipes that don't need LLM judgement (e.g.
//!   "fetch this URL, parse this field, return JSON").
//!
//! * **`hybrid`** — host runs `steps` first (deterministic preamble),
//!   then appends `after_executable` to the result so the LLM picks
//!   up where the host stopped. Use this when there's a fixed setup
//!   followed by judgement-driven work (e.g. "set up the browser
//!   with these tabs, then look around").
//!
//! ### Placeholder substitution
//!
//! `{key}` in any string value of `step.args` is replaced from this
//! lookup chain (first match wins):
//! 1. `save_as` outputs of earlier steps in the same run, addressed
//!    by their `save_as` name.
//! 2. The caller-supplied `arg_context` JSON object (top-level keys).
//!
//! Substitution is dumb single-pass `String::replace`, no escaping,
//! no nested object access — same philosophy as the role-prompt
//! placeholders. If a placeholder doesn't resolve it stays in the
//! string; that's intentional, so missing args fail loudly when the
//! tool rejects the literal `{key}` text.
//!
//! ### Why a sealed enum for `mode`
//!
//! Adding a new mode is a Rust change anyway (the `SkillTool::execute`
//! match has to learn what to do), so we don't try to keep `mode`
//! open-ended. Unknown modes fail at deserialise time with a clear
//! error pointing at the JSON file.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Top-level JSON skill descriptor. Loaded from `<root>/<name>/SKILL.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillSpec {
    /// Skill name. Should match the parent directory so the resolver
    /// can find it; we don't currently enforce that, but the
    /// `Skill(name=...)` argument the LLM passes goes through the
    /// resolver, not this field.
    pub name: String,
    /// One-line summary used in the `Skill` tool catalog the LLM sees.
    #[serde(default)]
    pub description: String,
    /// Execution mode (see module docs).
    #[serde(default)]
    pub mode: SkillMode,
    /// Advisory body returned verbatim to the LLM in `advisory` /
    /// `hybrid` modes. Markdown is fine — same shape as `SKILL.md`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub advisory_body: Option<String>,
    /// Steps the host runs in `executable` / `hybrid` modes. Linear
    /// (no DAG / branching — that's a follow-up if a real use case
    /// appears).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub steps: Vec<SkillStep>,
    /// Markdown text appended to the executable result in `hybrid`
    /// mode so the LLM can keep going.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after_executable: Option<String>,
}

/// Execution mode. See module docs for semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SkillMode {
    /// LLM reads the body and decides what tools to call. Same as
    /// legacy `SKILL.md`.
    #[default]
    Advisory,
    /// Host walks `steps` deterministically and returns the bundle.
    Executable,
    /// Host runs `steps` then appends `after_executable` for the LLM.
    Hybrid,
}

/// One step in an `executable` / `hybrid` skill.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillStep {
    /// Human-readable label used in error messages and progress logs.
    /// Optional — falls back to `step #N` if absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Name of the tool to dispatch (must be present in the active
    /// `ToolRegistry`).
    pub tool: String,
    /// JSON arguments passed to the tool. Any string value at any
    /// nesting depth is substring-substituted with `{key}` -> value
    /// from the `save_as` outputs / arg context. Non-string values
    /// pass through unchanged.
    #[serde(default)]
    pub args: serde_json::Value,
    /// If set, the tool's output is stored in the result bundle under
    /// this key and made available to later steps as `{save_as}`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub save_as: Option<String>,
    /// Error policy for this step. Defaults to abort.
    #[serde(default)]
    pub on_error: OnError,
}

/// What to do when a step's tool returns `is_error: true`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum OnError {
    /// Stop the skill, surface the error to the caller. (Default.)
    #[default]
    Abort,
    /// Record the error in the bundle, continue with the next step.
    Continue,
    /// Re-run this step up to N times before giving up. After the last
    /// retry fails the step aborts the skill, same as `Abort`.
    Retry { times: u32 },
}

/// Result bundle returned by the executor — keyed by `save_as` of
/// each step that asked for capture.
pub type SkillBundle = HashMap<String, serde_json::Value>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialise_minimal_advisory() {
        let json = r#"{
            "name": "tiny",
            "description": "say hi",
            "mode": "advisory",
            "advisory_body": "hello"
        }"#;
        let spec: SkillSpec = serde_json::from_str(json).unwrap();
        assert_eq!(spec.name, "tiny");
        assert_eq!(spec.mode, SkillMode::Advisory);
        assert_eq!(spec.advisory_body.as_deref(), Some("hello"));
        assert!(spec.steps.is_empty());
    }

    #[test]
    fn deserialise_defaults_to_advisory_mode() {
        let json = r#"{ "name": "x" }"#;
        let spec: SkillSpec = serde_json::from_str(json).unwrap();
        assert_eq!(spec.mode, SkillMode::Advisory);
        assert!(spec.advisory_body.is_none());
        assert_eq!(spec.description, "");
    }

    #[test]
    fn deserialise_executable_with_steps() {
        let json = r#"{
            "name": "fetch",
            "mode": "executable",
            "steps": [
                {
                    "tool": "web_fetch",
                    "args": { "url": "{url}" },
                    "save_as": "page",
                    "on_error": "abort"
                },
                {
                    "name": "second",
                    "tool": "summariser",
                    "args": { "input": "{page}" },
                    "on_error": { "retry": { "times": 2 } }
                }
            ]
        }"#;
        let spec: SkillSpec = serde_json::from_str(json).unwrap();
        assert_eq!(spec.mode, SkillMode::Executable);
        assert_eq!(spec.steps.len(), 2);
        assert_eq!(spec.steps[0].tool, "web_fetch");
        assert_eq!(spec.steps[0].save_as.as_deref(), Some("page"));
        assert_eq!(spec.steps[0].on_error, OnError::Abort);
        assert_eq!(spec.steps[1].name.as_deref(), Some("second"));
        assert_eq!(spec.steps[1].on_error, OnError::Retry { times: 2 });
    }

    #[test]
    fn deserialise_hybrid_with_after_executable() {
        let json = r#"{
            "name": "h",
            "mode": "hybrid",
            "steps": [{ "tool": "noop" }],
            "after_executable": "now look around"
        }"#;
        let spec: SkillSpec = serde_json::from_str(json).unwrap();
        assert_eq!(spec.mode, SkillMode::Hybrid);
        assert_eq!(spec.after_executable.as_deref(), Some("now look around"));
    }

    #[test]
    fn deserialise_unknown_mode_is_rejected() {
        let json = r#"{ "name": "x", "mode": "fancy" }"#;
        let err = serde_json::from_str::<SkillSpec>(json).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("fancy") || msg.contains("unknown variant"),
            "expected unknown-variant error mentioning `fancy`, got: {msg}"
        );
    }

    #[test]
    fn deserialise_step_default_on_error_is_abort() {
        let json = r#"{ "tool": "x" }"#;
        let step: SkillStep = serde_json::from_str(json).unwrap();
        assert_eq!(step.on_error, OnError::Abort);
        assert!(step.name.is_none());
        assert!(step.save_as.is_none());
    }

    #[test]
    fn deserialise_on_error_continue() {
        let json = r#"{ "tool": "x", "on_error": "continue" }"#;
        let step: SkillStep = serde_json::from_str(json).unwrap();
        assert_eq!(step.on_error, OnError::Continue);
    }

    /// Locate the workspace `naked/skills/` directory by walking
    /// up from `CARGO_MANIFEST_DIR`. Returns `None` when consumed as
    /// a dep.
    fn shipped_skills_dir() -> Option<std::path::PathBuf> {
        let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let candidate = manifest
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.join("skills"))?;
        if candidate.is_dir() {
            Some(candidate)
        } else {
            None
        }
    }

    /// Contract: the JSON form of `web-browser-playbook` ships
    /// alongside the legacy `SKILL.md` and parses cleanly. We don't
    /// require the bodies to be byte-identical (the JSON form may
    /// trim trailing newlines), but the JSON's `advisory_body` MUST
    /// contain the same operational guidance the MD has — checked by
    /// asserting a few stable phrases.
    #[test]
    fn shipped_web_browser_playbook_json_is_advisory_and_complete() {
        let Some(skills) = shipped_skills_dir() else {
            eprintln!("skipping: naked/skills not found");
            return;
        };
        let json_path = skills.join("web-browser-playbook").join("SKILL.json");
        if !json_path.exists() {
            eprintln!(
                "skipping: SKILL.json not present yet at {}",
                json_path.display()
            );
            return;
        }
        let raw = std::fs::read_to_string(&json_path).unwrap();
        let spec: SkillSpec =
            serde_json::from_str(&raw).expect("SKILL.json must deserialise as SkillSpec");
        assert_eq!(spec.name, "web-browser-playbook");
        assert_eq!(spec.mode, SkillMode::Advisory);
        assert!(
            !spec.description.is_empty(),
            "description must not be empty"
        );
        let body = spec
            .advisory_body
            .expect("advisory_body required for advisory mode");
        for required in [
            "Playwright MCP",
            "web_fetch",
            "Hard captcha walls",
            "Wayback",
            "Contacts hidden behind site captcha — visit URL",
            "click-to-reveal",
        ] {
            assert!(
                body.contains(required),
                "shipped advisory_body lost the `{required}` guidance"
            );
        }
    }

    #[test]
    fn round_trip_serialise_deserialise() {
        let spec = SkillSpec {
            name: "rt".into(),
            description: "round trip".into(),
            mode: SkillMode::Hybrid,
            advisory_body: None,
            steps: vec![SkillStep {
                name: Some("only".into()),
                tool: "t".into(),
                args: serde_json::json!({ "x": 1 }),
                save_as: Some("out".into()),
                on_error: OnError::Continue,
            }],
            after_executable: Some("done".into()),
        };
        let s = serde_json::to_string(&spec).unwrap();
        let parsed: SkillSpec = serde_json::from_str(&s).unwrap();
        assert_eq!(parsed.name, "rt");
        assert_eq!(parsed.mode, SkillMode::Hybrid);
        assert_eq!(parsed.steps.len(), 1);
        assert_eq!(parsed.steps[0].on_error, OnError::Continue);
    }
}
