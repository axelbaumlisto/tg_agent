//! The `Skill` tool exposed to the LLM.
//!
//! ## Dispatch matrix
//!
//! | resolved `SKILL.*` form | `mode` field          | what `execute()` does                                                              |
//! |-------------------------|-----------------------|------------------------------------------------------------------------------------|
//! | `SKILL.md`              | n/a (legacy)          | read file, return body to LLM as instructions                                      |
//! | `SKILL.json`            | `"advisory"` / unset  | return `advisory_body` to LLM                                                      |
//! | `SKILL.json`            | `"executable"`        | host runs `steps` via [`super::executor`], returns the bundled outputs             |
//! | `SKILL.json`            | `"hybrid"`            | host runs `steps`, then appends `after_executable` for the LLM to keep working on  |
//!
//! ## How the executor gets a tool dispatcher
//!
//! `executable` / `hybrid` modes need to call other tools, but at the
//! point [`SkillTool`] is constructed the [`crate::tool::registry::ToolRegistry`]
//! itself doesn't exist yet (we're building the tool list). To break
//! that chicken-and-egg we hold an `RwLock<Option<Arc<dyn SkillToolDispatcher>>>`:
//! * `SkillTool::new` returns a tool with no dispatcher set.
//! * After the registry is built, callers install one with
//!   `install_dispatcher(arc_dispatcher)`.
//! * If an LLM invokes an `executable`/`hybrid` skill before the
//!   dispatcher is installed, `execute()` returns a clear error
//!   ("executable skills require a dispatcher — wire it via
//!   AgentCore::install_skill_dispatcher").
//!
//! Production wiring of the dispatcher is intentionally deferred —
//! every shipping skill today is advisory, so the production path
//! goes through the cheap branches above. The wiring comment lives
//! at the SkillTool::install_dispatcher docstring.

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use tokio::sync::RwLock;

use crate::tool::Tool;
use crate::types::{Permission, ToolResult, ToolSpec};

use super::executor::{SkillToolDispatcher, execute_skill};
use super::resolver::{
    ResolvedSkill, SkillFile, SkillResolver, parse_skill_description, read_skill_description,
};
use super::spec::{SkillMode, SkillSpec};

pub struct SkillTool {
    resolver: SkillResolver,
    /// Catalog (`name`, `description?`) used to build the JSON schema
    /// the LLM sees. Descriptions come from the front-matter for MD
    /// skills and from the `description` field for JSON skills.
    catalog: Vec<(String, Option<String>)>,
    /// Late-bound tool dispatcher for executable / hybrid skills.
    /// `None` until [`Self::install_dispatcher`] is called.
    dispatcher: RwLock<Option<Arc<dyn SkillToolDispatcher>>>,
}

impl SkillTool {
    pub fn new(resolver: SkillResolver, available: &[(String, ResolvedSkill)]) -> Self {
        let catalog: Vec<(String, Option<String>)> = available
            .iter()
            .map(|(name, hit)| (name.clone(), read_skill_description(hit)))
            .collect();
        Self {
            resolver,
            catalog,
            dispatcher: RwLock::new(None),
        }
    }

    /// Wire a dispatcher so executable / hybrid skills can run. Safe
    /// to call after the tool is already inside a `ToolRegistry` —
    /// uses interior mutability.
    ///
    /// In the typical wiring flow (CLI / AgentCore) you'd:
    /// ```ignore
    /// let registry = Arc::new(ToolRegistry::new(tools));
    /// let dispatcher = Arc::new(RegistryDispatcher::new(registry.clone()));
    /// // grab the SkillTool out of the registry and install the dispatcher
    /// ```
    /// The `RegistryDispatcher` adapter doesn't exist yet — every
    /// shipping skill today is advisory, so the production code path
    /// doesn't need it. Add it in the same PR that ships the first
    /// executable skill.
    pub async fn install_dispatcher(&self, dispatcher: Arc<dyn SkillToolDispatcher>) {
        *self.dispatcher.write().await = Some(dispatcher);
    }

    fn build_description(&self) -> String {
        if self.catalog.is_empty() {
            return "Load a local skill definition and its instructions. No skills are currently available.".into();
        }
        let mut desc = String::from(
            "Load and execute a skill. You MUST use this tool when the user's request matches an available skill. Available skills:\n",
        );
        for (name, skill_desc) in &self.catalog {
            match skill_desc {
                Some(d) => desc.push_str(&format!("- {name}: {d}\n")),
                None => desc.push_str(&format!("- {name}\n")),
            }
        }
        desc
    }

    /// Markdown branch — current behaviour, kept verbatim so legacy
    /// MD skills (every shipping skill today) continue working.
    async fn execute_markdown(
        &self,
        skill_name: &str,
        hit: &ResolvedSkill,
        args: Option<String>,
    ) -> ToolResult {
        let content = match tokio::fs::read_to_string(&hit.path).await {
            Ok(c) => c,
            Err(e) => {
                return ToolResult::err(format!("Failed to read {}: {e}", hit.path.display()));
            }
        };
        let description = parse_skill_description(&content);
        let result = serde_json::json!({
            "skill": skill_name,
            "path": hit.path.display().to_string(),
            "args": args,
            "description": description,
            "prompt": content,
        });
        ToolResult::ok(serde_json::to_string_pretty(&result).unwrap_or_default())
    }

    /// Legacy `SKILL.toml` branch — render a markdown manual and
    /// return it as an advisory body. The host cannot execute TOML
    /// tools directly (see [`super::toml_legacy`] module docs); the
    /// agent is expected to pick the relevant shell template from
    /// the manual and run it through its own `bash` tool.
    async fn execute_toml(
        &self,
        skill_name: &str,
        hit: &ResolvedSkill,
        args: Option<String>,
    ) -> ToolResult {
        let raw = match tokio::fs::read_to_string(&hit.path).await {
            Ok(c) => c,
            Err(e) => {
                return ToolResult::err(format!("Failed to read {}: {e}", hit.path.display()));
            }
        };
        let Some(manifest) = super::toml_legacy::TomlSkillManifest::parse(&raw) else {
            return ToolResult::err(format!(
                "Failed to parse {} as legacy SKILL.toml",
                hit.path.display()
            ));
        };
        let manual = manifest.render_manual();
        let description = manifest.short_description();
        let result = serde_json::json!({
            "skill": skill_name,
            "path": hit.path.display().to_string(),
            "args": args,
            "description": description,
            "mode": "advisory",
            "form": "toml_legacy",
            "prompt": manual,
        });
        ToolResult::ok(serde_json::to_string_pretty(&result).unwrap_or_default())
    }

    /// JSON branch — parse `SkillSpec`, dispatch by mode.
    async fn execute_json(
        &self,
        skill_name: &str,
        hit: &ResolvedSkill,
        args: Option<String>,
    ) -> ToolResult {
        let raw = match tokio::fs::read_to_string(&hit.path).await {
            Ok(c) => c,
            Err(e) => {
                return ToolResult::err(format!("Failed to read {}: {e}", hit.path.display()));
            }
        };
        let spec: SkillSpec = match serde_json::from_str(&raw) {
            Ok(s) => s,
            Err(e) => {
                return ToolResult::err(format!(
                    "Failed to parse {} as SkillSpec: {e}",
                    hit.path.display()
                ));
            }
        };

        // Parse caller-supplied args. The `Skill` tool input has a
        // single `args` string; in JSON skills we additionally allow
        // it to be a JSON object so authors can pass a structured
        // context the executor's `{key}` substitution understands.
        // Bare-string args still work — they're exposed under
        // `{args}` for advisory bodies that want the user query.
        let arg_context = match &args {
            Some(s) => serde_json::from_str::<serde_json::Value>(s)
                .unwrap_or_else(|_| serde_json::json!({ "args": s })),
            None => serde_json::Value::Null,
        };

        match spec.mode {
            SkillMode::Advisory => {
                let body = spec.advisory_body.clone().unwrap_or_default();
                let result = serde_json::json!({
                    "skill": skill_name,
                    "path": hit.path.display().to_string(),
                    "args": args,
                    "description": spec.description,
                    "mode": "advisory",
                    "prompt": body,
                });
                ToolResult::ok(serde_json::to_string_pretty(&result).unwrap_or_default())
            }
            SkillMode::Executable | SkillMode::Hybrid => {
                let dispatcher = self.dispatcher.read().await.clone();
                let Some(dispatcher) = dispatcher else {
                    return ToolResult::err(format!(
                        "Skill `{skill_name}` is `{:?}` but no tool dispatcher is wired. \
                         Install one via SkillTool::install_dispatcher (see \
                         skill::tool docs) before invoking executable / hybrid skills.",
                        spec.mode
                    ));
                };
                let bundle = match execute_skill(&spec, dispatcher.as_ref(), &arg_context).await {
                    Ok(b) => b,
                    Err(e) => {
                        return ToolResult::err(format!("Skill `{skill_name}` failed: {e}"));
                    }
                };
                let mode_label = if matches!(spec.mode, SkillMode::Hybrid) {
                    "hybrid"
                } else {
                    "executable"
                };
                let mut payload = serde_json::json!({
                    "skill": skill_name,
                    "path": hit.path.display().to_string(),
                    "args": args,
                    "description": spec.description,
                    "mode": mode_label,
                    "bundle": bundle,
                });
                if matches!(spec.mode, SkillMode::Hybrid)
                    && let Some(after) = &spec.after_executable
                {
                    payload["prompt"] = serde_json::Value::String(after.clone());
                }
                ToolResult::ok(serde_json::to_string_pretty(&payload).unwrap_or_default())
            }
        }
    }
}

#[derive(Deserialize)]
struct SkillInput {
    skill: String,
    #[serde(default)]
    args: Option<String>,
}

#[async_trait]
impl Tool for SkillTool {
    fn spec(&self) -> ToolSpec {
        let skill_names: Vec<&str> = self.catalog.iter().map(|(n, _)| n.as_str()).collect();
        let enum_value = if skill_names.is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::json!(skill_names)
        };

        let mut skill_prop = serde_json::json!({
            "type": "string",
            "description": "Name of the skill to load"
        });
        if !enum_value.is_null() {
            skill_prop["enum"] = enum_value;
        }

        ToolSpec {
            name: "Skill".into(),
            description: self.build_description(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "skill": skill_prop,
                    "args": { "type": "string", "description": "Optional arguments / user query to pass to the skill" }
                },
                "required": ["skill"]
            }),
            permission: Permission::ReadOnly,
        }
    }

    async fn execute(&self, input: serde_json::Value, _cwd: &Path) -> ToolResult {
        let input: SkillInput = match serde_json::from_value(input) {
            Ok(v) => v,
            Err(e) => {
                return ToolResult::err(format!("Invalid input: {e}"));
            }
        };

        let hit = match self.resolver.resolve(&input.skill) {
            Some(p) => p,
            None => {
                return ToolResult::err(format!("Skill '{}' not found", input.skill));
            }
        };

        tracing::info!(
            target: "naked::skill::tool",
            skill = %input.skill,
            kind = ?hit.kind,
            path = %hit.path.display(),
            "[skill-load] {} (kind={:?}, path={})",
            input.skill,
            hit.kind,
            hit.path.display()
        );

        match hit.kind {
            SkillFile::Markdown => self.execute_markdown(&input.skill, &hit, input.args).await,
            SkillFile::Json => self.execute_json(&input.skill, &hit, input.args).await,
            SkillFile::Toml => self.execute_toml(&input.skill, &hit, input.args).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;
    use tempfile::tempdir;

    fn write(p: &Path, body: &str) {
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(p, body).unwrap();
    }

    fn make_tool(root: &Path) -> SkillTool {
        let resolver = SkillResolver::new(vec![root.to_path_buf()]);
        let available = resolver.list();
        SkillTool::new(resolver, &available)
    }

    #[tokio::test]
    async fn markdown_skill_returns_body_verbatim() {
        let tmp = tempdir().unwrap();
        write(
            &tmp.path().join("greet").join("SKILL.md"),
            "description: say hi\n\n# Body\nhello",
        );
        let tool = make_tool(tmp.path());
        let r = tool
            .execute(serde_json::json!({ "skill": "greet" }), tmp.path())
            .await;
        assert!(!r.is_error, "got error: {}", r.output);
        let v: serde_json::Value = serde_json::from_str(&r.output).unwrap();
        assert_eq!(v["skill"], "greet");
        assert_eq!(v["description"], "say hi");
        assert!(v["prompt"].as_str().unwrap().contains("# Body"));
    }

    #[tokio::test]
    async fn json_advisory_skill_returns_advisory_body() {
        let tmp = tempdir().unwrap();
        write(
            &tmp.path().join("planner").join("SKILL.json"),
            r#"{
                "name": "planner",
                "description": "advise the agent",
                "mode": "advisory",
                "advisory_body": "do A then B"
            }"#,
        );
        let tool = make_tool(tmp.path());
        let r = tool
            .execute(serde_json::json!({ "skill": "planner" }), tmp.path())
            .await;
        assert!(!r.is_error, "got error: {}", r.output);
        let v: serde_json::Value = serde_json::from_str(&r.output).unwrap();
        assert_eq!(v["mode"], "advisory");
        assert_eq!(v["prompt"], "do A then B");
        assert_eq!(v["description"], "advise the agent");
    }

    #[tokio::test]
    async fn json_executable_without_dispatcher_returns_clear_error() {
        let tmp = tempdir().unwrap();
        write(
            &tmp.path().join("runner").join("SKILL.json"),
            r#"{
                "name": "runner",
                "mode": "executable",
                "steps": [{"tool":"x","args":{}}]
            }"#,
        );
        let tool = make_tool(tmp.path());
        let r = tool
            .execute(serde_json::json!({ "skill": "runner" }), tmp.path())
            .await;
        assert!(r.is_error);
        assert!(
            r.output.contains("dispatcher"),
            "expected dispatcher hint, got: {}",
            r.output
        );
    }

    /// Counts calls so we know the executor really ran the steps.
    struct CountingDispatcher {
        n: Mutex<u32>,
        canned: HashMap<String, ToolResult>,
    }

    #[async_trait]
    impl SkillToolDispatcher for CountingDispatcher {
        async fn execute(&self, name: &str, _input: serde_json::Value) -> ToolResult {
            *self.n.lock().unwrap() += 1;
            self.canned
                .get(name)
                .cloned()
                .unwrap_or(ToolResult::err(format!("no canned response for {name}")))
        }
    }

    #[tokio::test]
    async fn json_executable_with_dispatcher_runs_steps() {
        let tmp = tempdir().unwrap();
        write(
            &tmp.path().join("runner").join("SKILL.json"),
            r#"{
                "name": "runner",
                "mode": "executable",
                "steps": [
                    {"tool":"x","args":{"q":"{topic}"},"save_as":"out"}
                ]
            }"#,
        );
        let tool = make_tool(tmp.path());
        let mut canned = HashMap::new();
        canned.insert("x".into(), ToolResult::ok(r#"{"answer":42}"#));
        let dispatcher = Arc::new(CountingDispatcher {
            n: Mutex::new(0),
            canned,
        });
        let dispatcher_for_count = dispatcher.clone();
        tool.install_dispatcher(dispatcher).await;

        let args_json = r#"{"topic":"vintage"}"#;
        let r = tool
            .execute(
                serde_json::json!({ "skill": "runner", "args": args_json }),
                tmp.path(),
            )
            .await;
        assert!(!r.is_error, "got error: {}", r.output);
        let v: serde_json::Value = serde_json::from_str(&r.output).unwrap();
        assert_eq!(v["mode"], "executable");
        assert_eq!(v["bundle"]["out"]["answer"], 42);
        assert_eq!(*dispatcher_for_count.n.lock().unwrap(), 1);
    }

    #[tokio::test]
    async fn json_hybrid_appends_after_executable_text() {
        let tmp = tempdir().unwrap();
        write(
            &tmp.path().join("h").join("SKILL.json"),
            r#"{
                "name": "h",
                "mode": "hybrid",
                "steps": [{"tool":"x","args":{},"save_as":"out"}],
                "after_executable": "now look around the page"
            }"#,
        );
        let tool = make_tool(tmp.path());
        let mut canned = HashMap::new();
        canned.insert("x".into(), ToolResult::ok("ok"));
        tool.install_dispatcher(Arc::new(CountingDispatcher {
            n: Mutex::new(0),
            canned,
        }))
        .await;
        let r = tool
            .execute(serde_json::json!({ "skill": "h" }), tmp.path())
            .await;
        assert!(!r.is_error, "got error: {}", r.output);
        let v: serde_json::Value = serde_json::from_str(&r.output).unwrap();
        assert_eq!(v["mode"], "hybrid");
        assert_eq!(v["prompt"], "now look around the page");
        assert!(v["bundle"].is_object());
    }

    #[tokio::test]
    async fn json_form_wins_over_md_in_same_dir() {
        let tmp = tempdir().unwrap();
        let dir = tmp.path().join("dual");
        write(&dir.join("SKILL.md"), "legacy markdown body");
        write(
            &dir.join("SKILL.json"),
            r#"{"name":"dual","mode":"advisory","advisory_body":"json body"}"#,
        );
        let tool = make_tool(tmp.path());
        let r = tool
            .execute(serde_json::json!({ "skill": "dual" }), tmp.path())
            .await;
        let v: serde_json::Value = serde_json::from_str(&r.output).unwrap();
        assert_eq!(v["mode"], "advisory");
        assert_eq!(v["prompt"], "json body");
    }

    #[tokio::test]
    async fn toml_legacy_skill_returns_rendered_manual() {
        let tmp = tempdir().unwrap();
        write(
            &tmp.path().join("legacy").join("SKILL.toml"),
            r#"
[skill]
name = "legacy"
description = "Legacy TOML skill"
prompts = ["Triggered on user word 'foo'"]

[[tools]]
name = "run"
description = "Runs a Python script"
kind = "shell"
command = "python3 ~/tools/legacy.py --x {x}"

[tools.args]
x = "A required param"
"#,
        );
        let tool = make_tool(tmp.path());
        let r = tool
            .execute(serde_json::json!({ "skill": "legacy" }), tmp.path())
            .await;
        assert!(!r.is_error, "got error: {}", r.output);
        let v: serde_json::Value = serde_json::from_str(&r.output).unwrap();
        assert_eq!(v["skill"], "legacy");
        assert_eq!(v["mode"], "advisory");
        assert_eq!(v["form"], "toml_legacy");
        assert_eq!(v["description"], "Legacy TOML skill");
        let prompt = v["prompt"].as_str().unwrap();
        assert!(prompt.contains("# Skill: legacy"));
        assert!(prompt.contains("### run"));
        assert!(prompt.contains("python3 ~/tools/legacy.py --x {x}"));
        assert!(prompt.contains("`x`: A required param"));
    }

    #[tokio::test]
    async fn json_form_wins_over_toml_in_same_dir() {
        let tmp = tempdir().unwrap();
        let dir = tmp.path().join("dual");
        write(
            &dir.join("SKILL.toml"),
            "[skill]\nname=\"dual\"\ndescription=\"toml desc\"",
        );
        write(
            &dir.join("SKILL.json"),
            r#"{"name":"dual","description":"json desc","mode":"advisory","advisory_body":"json body"}"#,
        );
        let tool = make_tool(tmp.path());
        let r = tool
            .execute(serde_json::json!({ "skill": "dual" }), tmp.path())
            .await;
        let v: serde_json::Value = serde_json::from_str(&r.output).unwrap();
        assert_eq!(v["mode"], "advisory");
        assert_eq!(v["description"], "json desc");
        assert_eq!(v["prompt"], "json body");
    }

    #[tokio::test]
    async fn unknown_skill_returns_error() {
        let tmp = tempdir().unwrap();
        let tool = make_tool(tmp.path());
        let r = tool
            .execute(serde_json::json!({ "skill": "missing" }), tmp.path())
            .await;
        assert!(r.is_error);
        assert!(r.output.contains("missing"));
    }
}
