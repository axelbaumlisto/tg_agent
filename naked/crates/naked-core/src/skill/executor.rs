//! Deterministic interpreter for [`super::spec::SkillSpec`] in
//! `executable` and `hybrid` modes.
//!
//! ## Why a separate executor (and not just "let the LLM call tools")
//!
//! For genuinely deterministic recipes — open URL, parse field,
//! return JSON — sending the playbook to the LLM and waiting for it
//! to call the tools costs latency, tokens, and gives the LLM room to
//! deviate from a known-good sequence. The executor lets the host run
//! such recipes itself, returning one bundled tool result the LLM
//! consumes in a single step.
//!
//! ## Tool dispatch — abstraction
//!
//! The executor doesn't bind to [`crate::tool::registry::ToolRegistry`]
//! directly; it talks to a small [`SkillToolDispatcher`] trait so:
//!
//! 1. Tests can supply a mock dispatcher without building a full
//!    registry, and
//! 2. Future surfaces (e.g. routing through a remote MCP server) can
//!    plug in without changing the executor.
//!
//! The CLI / agent loop wires up an adapter that forwards
//! `execute(name, input)` to a real `ToolRegistry`.
//!
//! ## Permission gating
//!
//! Skill steps inherit the active tool registry's permission policy —
//! the executor calls into the dispatcher exactly the way the LLM
//! would, so a tool the role's `tool_filter` blocks for the LLM is
//! also blocked when invoked by an executable skill.
//! No privilege escalation by routing through `Skill`.

use std::collections::HashMap;

use async_trait::async_trait;

use crate::types::ToolResult;

use super::spec::{OnError, SkillBundle, SkillSpec, SkillStep};

/// Anything that can run a tool by name. Implemented by an adapter
/// over `ToolRegistry` (production) and by mocks (tests).
#[async_trait]
pub trait SkillToolDispatcher: Send + Sync {
    async fn execute(&self, name: &str, input: serde_json::Value) -> ToolResult;
}

/// Single entry point — runs every step of `spec` in order.
///
/// Returns:
/// * `Ok(bundle)` — every step the bundle stored under its `save_as`
///   key. Steps without `save_as` still ran; their output is just not
///   captured. Errors that were swallowed by `on_error: continue`
///   land in the bundle as `{ "error": "<msg>" }` under the step's
///   `save_as` (if any).
/// * `Err(SkillExecError)` — a step aborted (default policy or
///   retries exhausted). The error includes the step index, name,
///   and last tool output.
pub async fn execute_skill(
    spec: &SkillSpec,
    dispatcher: &dyn SkillToolDispatcher,
    arg_context: &serde_json::Value,
) -> Result<SkillBundle, SkillExecError> {
    let mut bundle: SkillBundle = HashMap::new();

    for (idx, step) in spec.steps.iter().enumerate() {
        let label = step
            .name
            .clone()
            .unwrap_or_else(|| format!("step #{} ({})", idx, step.tool));

        let args = substitute_placeholders(&step.args, arg_context, &bundle);

        let result = run_with_retry(dispatcher, step, args).await;

        match (result.is_error, step.on_error) {
            (false, _) => {
                if let Some(key) = &step.save_as {
                    bundle.insert(key.clone(), tool_result_to_value(&result));
                }
            }
            (true, OnError::Continue) => {
                if let Some(key) = &step.save_as {
                    bundle.insert(
                        key.clone(),
                        serde_json::json!({ "error": result.output.clone() }),
                    );
                }
            }
            (true, OnError::Abort) | (true, OnError::Retry { .. }) => {
                // Retry policy already exhausted in `run_with_retry`.
                return Err(SkillExecError {
                    step_index: idx,
                    step_label: label,
                    tool: step.tool.clone(),
                    last_output: result.output,
                });
            }
        }
    }

    Ok(bundle)
}

/// Run a step, applying its `Retry { times: N }` policy. For
/// non-retry policies this just calls once and returns. The result is
/// the LAST tool result (success or final failure).
async fn run_with_retry(
    dispatcher: &dyn SkillToolDispatcher,
    step: &SkillStep,
    args: serde_json::Value,
) -> ToolResult {
    let attempts = match step.on_error {
        OnError::Retry { times } => times.saturating_add(1),
        _ => 1,
    };
    let mut last = ToolResult::ok(String::new());
    for _ in 0..attempts {
        last = dispatcher.execute(&step.tool, args.clone()).await;
        if !last.is_error {
            return last;
        }
    }
    last
}

/// Walk every string in `args`, replace `{key}` tokens with values
/// from `arg_context` (top-level keys) and the `bundle` (string-or-
/// stringified-JSON of saved outputs). First match wins. `bundle`
/// shadows `arg_context` because step outputs are explicitly captured
/// by the skill author and should override caller-supplied defaults.
fn substitute_placeholders(
    args: &serde_json::Value,
    arg_context: &serde_json::Value,
    bundle: &SkillBundle,
) -> serde_json::Value {
    walk(args, &|s| substitute_one(s, arg_context, bundle))
}

fn substitute_one(s: &str, arg_context: &serde_json::Value, bundle: &SkillBundle) -> String {
    let mut out = s.to_string();
    // Bundle keys first (later steps can override caller args).
    for (k, v) in bundle {
        let needle = format!("{{{k}}}");
        if out.contains(&needle) {
            out = out.replace(&needle, &value_to_str(v));
        }
    }
    if let Some(obj) = arg_context.as_object() {
        for (k, v) in obj {
            let needle = format!("{{{k}}}");
            if out.contains(&needle) {
                out = out.replace(&needle, &value_to_str(v));
            }
        }
    }
    out
}

/// Recursively walk a JSON value, applying `f` to every string leaf.
/// Non-string leaves pass through unchanged.
fn walk(v: &serde_json::Value, f: &dyn Fn(&str) -> String) -> serde_json::Value {
    match v {
        serde_json::Value::String(s) => serde_json::Value::String(f(s)),
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(|x| walk(x, f)).collect())
        }
        serde_json::Value::Object(map) => {
            let mut out = serde_json::Map::new();
            for (k, v) in map {
                out.insert(k.clone(), walk(v, f));
            }
            serde_json::Value::Object(out)
        }
        other => other.clone(),
    }
}

/// Stringify a JSON value the way placeholders should see it: bare
/// strings become themselves (so `{topic}` -> `"hi"` interpolates as
/// `hi`, not `"hi"`). Other types fall back to `to_string()`.
fn value_to_str(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Convert a tool result into a JSON value suitable for the bundle.
/// We try to parse the output as JSON first (so structured tools land
/// as objects, not stringified blobs) and fall back to a plain string.
fn tool_result_to_value(r: &ToolResult) -> serde_json::Value {
    serde_json::from_str::<serde_json::Value>(&r.output)
        .unwrap_or_else(|_| serde_json::Value::String(r.output.clone()))
}

/// Failure surface for [`execute_skill`]. The CLI / agent loop turns
/// this into the tool result it returns to the LLM.
#[derive(Debug, Clone)]
pub struct SkillExecError {
    pub step_index: usize,
    pub step_label: String,
    pub tool: String,
    pub last_output: String,
}

impl std::fmt::Display for SkillExecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "skill aborted at step {} ({}, tool=`{}`): {}",
            self.step_index, self.step_label, self.tool, self.last_output
        )
    }
}

impl std::error::Error for SkillExecError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skill::spec::{SkillMode, SkillSpec, SkillStep};
    use std::sync::Arc;
    use std::sync::Mutex;

    /// Mock dispatcher: returns canned results keyed by tool name and
    /// records every invocation for assertions.
    struct MockDispatcher {
        responses: Mutex<HashMap<String, Vec<ToolResult>>>,
        calls: Mutex<Vec<(String, serde_json::Value)>>,
    }

    impl MockDispatcher {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                responses: Mutex::new(HashMap::new()),
                calls: Mutex::new(Vec::new()),
            })
        }

        fn add_response(&self, tool: &str, result: ToolResult) {
            self.responses
                .lock()
                .unwrap()
                .entry(tool.to_string())
                .or_default()
                .push(result);
        }

        fn calls(&self) -> Vec<(String, serde_json::Value)> {
            self.calls.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl SkillToolDispatcher for MockDispatcher {
        async fn execute(&self, name: &str, input: serde_json::Value) -> ToolResult {
            self.calls
                .lock()
                .unwrap()
                .push((name.to_string(), input.clone()));
            let mut map = self.responses.lock().unwrap();
            if let Some(queue) = map.get_mut(name)
                && !queue.is_empty()
            {
                return queue.remove(0);
            }
            ToolResult::err(format!("no canned response for {name}"))
        }
    }

    fn spec_with_steps(steps: Vec<SkillStep>) -> SkillSpec {
        SkillSpec {
            name: "t".into(),
            description: String::new(),
            mode: SkillMode::Executable,
            advisory_body: None,
            steps,
            after_executable: None,
        }
    }

    #[tokio::test]
    async fn happy_path_substitutes_and_captures() {
        let disp = MockDispatcher::new();
        disp.add_response("fetch", ToolResult::ok(r#"{"page":"hello"}"#));
        disp.add_response("summ", ToolResult::ok("summary text"));

        let spec = spec_with_steps(vec![
            SkillStep {
                name: None,
                tool: "fetch".into(),
                args: serde_json::json!({ "url": "{url}" }),
                save_as: Some("page".into()),
                on_error: OnError::Abort,
            },
            SkillStep {
                name: Some("summarise".into()),
                tool: "summ".into(),
                args: serde_json::json!({ "input": "{page}" }),
                save_as: Some("summary".into()),
                on_error: OnError::Abort,
            },
        ]);
        let ctx = serde_json::json!({ "url": "https://x" });

        let bundle = execute_skill(&spec, disp.as_ref(), &ctx).await.unwrap();

        let calls = disp.calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].0, "fetch");
        assert_eq!(calls[0].1["url"], "https://x");
        assert_eq!(calls[1].0, "summ");
        assert_eq!(
            bundle.get("summary").unwrap().as_str(),
            Some("summary text")
        );
        // Step 2 should have seen the JSON-stringified bundle entry
        // for `page` (the parser turned the fetch output into an object,
        // so `{page}` interpolates as the JSON object's string form).
        let input_str = calls[1].1["input"].as_str().unwrap();
        assert!(input_str.contains("hello"));
    }

    #[tokio::test]
    async fn abort_propagates_error() {
        let disp = MockDispatcher::new();
        disp.add_response("a", ToolResult::ok("ok"));
        disp.add_response("b", ToolResult::err("boom"));

        let spec = spec_with_steps(vec![
            SkillStep {
                name: Some("first".into()),
                tool: "a".into(),
                args: serde_json::Value::Null,
                save_as: None,
                on_error: OnError::Abort,
            },
            SkillStep {
                name: Some("second".into()),
                tool: "b".into(),
                args: serde_json::Value::Null,
                save_as: None,
                on_error: OnError::Abort,
            },
        ]);
        let err = execute_skill(&spec, disp.as_ref(), &serde_json::Value::Null)
            .await
            .unwrap_err();
        assert_eq!(err.step_index, 1);
        assert_eq!(err.tool, "b");
        assert!(err.last_output.contains("boom"));
    }

    #[tokio::test]
    async fn continue_records_error_and_keeps_going() {
        let disp = MockDispatcher::new();
        disp.add_response("flaky", ToolResult::err("nope"));
        disp.add_response("after", ToolResult::ok("done"));

        let spec = spec_with_steps(vec![
            SkillStep {
                name: None,
                tool: "flaky".into(),
                args: serde_json::Value::Null,
                save_as: Some("flaky_out".into()),
                on_error: OnError::Continue,
            },
            SkillStep {
                name: None,
                tool: "after".into(),
                args: serde_json::Value::Null,
                save_as: Some("after_out".into()),
                on_error: OnError::Abort,
            },
        ]);
        let bundle = execute_skill(&spec, disp.as_ref(), &serde_json::Value::Null)
            .await
            .unwrap();
        assert_eq!(
            bundle.get("flaky_out").unwrap()["error"].as_str(),
            Some("nope")
        );
        assert_eq!(bundle.get("after_out").unwrap().as_str(), Some("done"));
    }

    #[tokio::test]
    async fn retry_succeeds_on_second_attempt() {
        let disp = MockDispatcher::new();
        disp.add_response("flaky", ToolResult::err("first"));
        disp.add_response("flaky", ToolResult::ok("second"));

        let spec = spec_with_steps(vec![SkillStep {
            name: None,
            tool: "flaky".into(),
            args: serde_json::Value::Null,
            save_as: Some("out".into()),
            on_error: OnError::Retry { times: 2 },
        }]);
        let bundle = execute_skill(&spec, disp.as_ref(), &serde_json::Value::Null)
            .await
            .unwrap();
        assert_eq!(bundle.get("out").unwrap().as_str(), Some("second"));
        assert_eq!(disp.calls().len(), 2);
    }

    #[tokio::test]
    async fn retry_exhausts_then_aborts() {
        let disp = MockDispatcher::new();
        disp.add_response("never", ToolResult::err("a"));
        disp.add_response("never", ToolResult::err("b"));
        disp.add_response("never", ToolResult::err("c"));

        let spec = spec_with_steps(vec![SkillStep {
            name: None,
            tool: "never".into(),
            args: serde_json::Value::Null,
            save_as: None,
            on_error: OnError::Retry { times: 2 },
        }]);
        let err = execute_skill(&spec, disp.as_ref(), &serde_json::Value::Null)
            .await
            .unwrap_err();
        assert_eq!(err.tool, "never");
        // 2 retries + the original = 3 attempts
        assert_eq!(disp.calls().len(), 3);
        assert!(err.last_output.contains('c'));
    }

    #[tokio::test]
    async fn placeholder_unresolved_passes_literal_through() {
        // Caller forgets to supply {topic}. The executor doesn't try
        // to be clever — it leaves the literal `{topic}` in the args
        // so the tool will fail loudly. That's preferable to silently
        // substituting `""` and producing nonsense output.
        let disp = MockDispatcher::new();
        disp.add_response("echo", ToolResult::ok("ok"));
        let spec = spec_with_steps(vec![SkillStep {
            name: None,
            tool: "echo".into(),
            args: serde_json::json!({ "q": "see {topic}" }),
            save_as: None,
            on_error: OnError::Abort,
        }]);
        execute_skill(&spec, disp.as_ref(), &serde_json::Value::Null)
            .await
            .unwrap();
        let calls = disp.calls();
        assert_eq!(calls[0].1["q"], "see {topic}");
    }

    #[tokio::test]
    async fn nested_args_substitute_recursively() {
        let disp = MockDispatcher::new();
        disp.add_response("nested", ToolResult::ok("ok"));
        let spec = spec_with_steps(vec![SkillStep {
            name: None,
            tool: "nested".into(),
            args: serde_json::json!({
                "outer": {
                    "list": [{ "url": "{url}" }, "static"],
                    "n": 1
                }
            }),
            save_as: None,
            on_error: OnError::Abort,
        }]);
        execute_skill(
            &spec,
            disp.as_ref(),
            &serde_json::json!({ "url": "https://y" }),
        )
        .await
        .unwrap();
        let calls = disp.calls();
        assert_eq!(calls[0].1["outer"]["list"][0]["url"], "https://y");
        assert_eq!(calls[0].1["outer"]["list"][1], "static");
        assert_eq!(calls[0].1["outer"]["n"], 1);
    }

    #[tokio::test]
    async fn empty_steps_returns_empty_bundle() {
        let disp = MockDispatcher::new();
        let spec = spec_with_steps(vec![]);
        let bundle = execute_skill(&spec, disp.as_ref(), &serde_json::Value::Null)
            .await
            .unwrap();
        assert!(bundle.is_empty());
        assert!(disp.calls().is_empty());
    }
}
