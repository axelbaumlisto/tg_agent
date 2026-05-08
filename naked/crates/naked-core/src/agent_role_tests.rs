use super::*;

#[test]
fn tool_filter_allow_all_lets_everything_through() {
    let f = ToolFilter::default();
    assert!(f.allows("web_fetch"));
    assert!(f.allows("bash"));
    assert!(f.allows("anything"));
}

#[test]
fn tool_filter_allow_lists_only_named_tools() {
    let f = ToolFilter::Allow {
        tools: vec!["web_fetch".into(), "research_save".into()],
    };
    assert!(f.allows("web_fetch"));
    assert!(f.allows("research_save"));
    assert!(!f.allows("bash"));
    assert!(!f.allows("write_file"));
}

#[test]
fn tool_filter_deny_excludes_named_tools() {
    let f = ToolFilter::Deny {
        tools: vec!["bash".into(), "write_file".into()],
    };
    assert!(f.allows("web_fetch"));
    assert!(f.allows("research_save"));
    assert!(!f.allows("bash"));
    assert!(!f.allows("write_file"));
}

#[test]
fn agent_role_builder_chain() {
    // Use a placeholder model id — we are testing the builder
    // wiring, not asserting any real provider has this model.
    // Hard-coding e.g. `qwen3.6-plus` would create a hidden
    // assumption that whoever runs the test stack also has that
    // provider configured.
    let r = AgentRole::new("test", "you are a tester")
        .with_description("for tests")
        .with_model("placeholder-model-id")
        .with_skills(vec!["web-browser-playbook".into()])
        .with_max_iters(15)
        .with_tool_filter(ToolFilter::Allow {
            tools: vec!["web_fetch".into()],
        });

    assert_eq!(r.name, "test");
    assert_eq!(r.description, "for tests");
    assert_eq!(r.model.as_deref(), Some("placeholder-model-id"));
    assert_eq!(r.default_skills, vec!["web-browser-playbook".to_string()]);
    assert_eq!(r.max_iters, 15);
    assert!(r.tool_filter.allows("web_fetch"));
    assert!(!r.tool_filter.allows("bash"));
}

#[test]
fn agent_role_serialises_compactly_when_defaults() {
    // Default role should produce minimal JSON — no empty maps,
    // no `null`s. Otherwise naked.json gets noisy when listing
    // many roles.
    let r = AgentRole::new("min", "sys");
    let json = serde_json::to_string(&r).unwrap();
    // No `model`, no `default_skills`, no `browser_runtime`.
    assert!(!json.contains("\"model\""), "json: {json}");
    assert!(!json.contains("\"default_skills\""), "json: {json}");
    assert!(!json.contains("\"browser_runtime\""), "json: {json}");
    // tool_filter MUST be present (it's tagged so we always know
    // the policy explicitly even when default).
    assert!(json.contains("\"tool_filter\""), "json: {json}");
    assert!(json.contains("\"allow_all\""), "json: {json}");
}

#[test]
fn agent_role_round_trips_through_json() {
    let r = AgentRole::new("rt", "sys")
        .with_model("m1")
        .with_skills(vec!["s1".into(), "s2".into()])
        .with_browser_runtime(BrowserRuntime {
            proxy: Some("http://proxy:8080".into()),
            extensions: vec!["./ext/capsolver".into()],
            extension_env: [("CAPSOLVER_API_KEY".to_string(), "abc".to_string())]
                .into_iter()
                .collect(),
        });
    let json = serde_json::to_string(&r).unwrap();
    let back: AgentRole = serde_json::from_str(&json).unwrap();
    assert_eq!(back.name, r.name);
    assert_eq!(back.model, r.model);
    assert_eq!(back.default_skills, r.default_skills);
    assert_eq!(back.browser_runtime.proxy, r.browser_runtime.proxy);
    assert_eq!(
        back.browser_runtime.extension_env.get("CAPSOLVER_API_KEY"),
        Some(&"abc".to_string())
    );
}

#[test]
fn task_builder_chain() {
    let t = Task::new("browser_extractor", "open and extract")
        .with_id("t-1")
        .with_max_wall(60)
        .with_context(serde_json::json!({"url": "https://x.com"}));
    assert_eq!(t.id, "t-1");
    assert_eq!(t.role, "browser_extractor");
    assert_eq!(t.max_wall_secs, Some(60));
    assert_eq!(t.context["url"], "https://x.com");
}

#[test]
fn task_stats_summary_line_is_grep_compatible() {
    // Pin the on-wire shape so probe / batch greppers keep
    // working.
    let mut s = TaskStats::default();
    s.tools.insert("Skill".into(), 1);
    s.tools.insert("browser_navigate".into(), 2);
    s.captcha_hits = 1;
    s.skill_loads = 1;
    s.text_deltas = 7;

    let line = s.summary_line();
    assert!(line.starts_with("[task-summary] "));
    assert!(line.contains("Skill=1"));
    assert!(line.contains("browser_navigate=2"));
    assert!(line.contains("captcha_hits=1"));
    assert!(line.contains("skill_loads=1"));
    assert!(line.contains("text_deltas=7"));
    assert!(line.contains("errors=0"));
}

#[test]
fn validation_verdict_helpers() {
    let p = ValidationVerdict::pass("phone-vn");
    assert!(p.passed);
    assert!(p.reasons.is_empty());
    assert_eq!(p.validator, "phone-vn");

    let f = ValidationVerdict::fail("phone-vn", "no digits");
    assert!(!f.passed);
    assert_eq!(f.reasons, vec!["no digits".to_string()]);
}

#[test]
fn expand_placeholders_substitutes_top_level_strings() {
    let ctx = serde_json::json!({
        "topic": "real-estate",
        "url":   "https://example.com",
        "n":     5,
    });
    let out = expand_placeholders("topic={topic}, url={url}, n={n}", &ctx);
    assert_eq!(out, "topic=real-estate, url=https://example.com, n=5");
}

#[test]
fn expand_placeholders_leaves_unknown_keys_alone() {
    let ctx = serde_json::json!({"topic": "x"});
    let out = expand_placeholders("{topic} but not {missing}", &ctx);
    assert_eq!(out, "x but not {missing}");
}

#[test]
fn expand_placeholders_handles_non_object_context() {
    let out = expand_placeholders("hello {name}", &serde_json::Value::Null);
    assert_eq!(out, "hello {name}");
}

#[test]
fn task_output_skeleton_has_idle_default() {
    let o = TaskOutput::skeleton("t-1", "browser_extractor");
    assert_eq!(o.task_id, "t-1");
    assert_eq!(o.role_name, "browser_extractor");
    assert_eq!(o.stop_reason, StopReason::AgentIdle);
}

// A trivially-implementable Validator used to verify the trait
// signature compiles + returns the right shape.
struct AlwaysPass;
#[async_trait]
impl Validator for AlwaysPass {
    fn name(&self) -> &str {
        "always-pass"
    }
    async fn validate(&self, _o: &TaskOutput) -> ValidationVerdict {
        ValidationVerdict::pass(self.name())
    }
}

// Note: the contract tests for the built-in roles
// (`browser_extractor` / `web_researcher` shape, universality of
// `web_researcher`'s prompt) now live in
// `agent_store::tests` — they load the shipped data files
// (`naked/agents/<name>/`) via `AgentStore::load_dirs` so the
// assertions stay anchored to the real disk artefacts an
// operator can edit, not to a stale Rust-side copy.

#[test]
fn agent_role_default_validators_round_trips() {
    let r =
        AgentRole::new("v", "sys").with_validators(vec!["gatekeeper".into(), "phone-vn".into()]);
    assert_eq!(
        r.default_validators,
        vec!["gatekeeper".to_string(), "phone-vn".to_string()]
    );
    let json = serde_json::to_string(&r).unwrap();
    assert!(json.contains("\"default_validators\""), "json: {json}");
    let back: AgentRole = serde_json::from_str(&json).unwrap();
    assert_eq!(back.default_validators, r.default_validators);
}

#[test]
fn agent_role_default_validators_empty_omitted_in_json() {
    let r = AgentRole::new("v", "sys");
    let json = serde_json::to_string(&r).unwrap();
    assert!(
        !json.contains("\"default_validators\""),
        "empty list must be skipped: {json}"
    );
}

#[test]
fn agent_role_override_validators_replaces_list() {
    let base =
        AgentRole::new("rt", "sys").with_validators(vec!["gatekeeper".into(), "phone-vn".into()]);
    let ov = AgentRoleOverride {
        default_validators: Some(vec!["gatekeeper".into()]),
        ..Default::default()
    };
    let merged = apply_role_override(base.clone(), &ov);
    assert_eq!(merged.default_validators, vec!["gatekeeper".to_string()]);

    // None-override leaves the list intact (no surprise wipe).
    let merged = apply_role_override(base.clone(), &AgentRoleOverride::default());
    assert_eq!(merged.default_validators, base.default_validators);

    // Explicit Some(vec![]) is the documented opt-out.
    let ov_empty = AgentRoleOverride {
        default_validators: Some(vec![]),
        ..Default::default()
    };
    let merged = apply_role_override(base, &ov_empty);
    assert!(merged.default_validators.is_empty());
}

#[test]
fn agent_role_override_apply_only_touches_set_fields() {
    let base = AgentRole::new("rt", "sys")
        .with_model("default-m")
        .with_max_iters(20);
    let o = AgentRoleOverride {
        model: Some("override-m".into()),
        ..Default::default()
    };
    let merged = apply_role_override(base.clone(), &o);
    assert_eq!(merged.model.as_deref(), Some("override-m"));
    assert_eq!(merged.max_iters, 20, "untouched field must survive");

    let o = AgentRoleOverride {
        max_iters: Some(99),
        default_skills: Some(vec!["only-this".into()]),
        ..Default::default()
    };
    let merged = apply_role_override(base.clone(), &o);
    assert_eq!(merged.model.as_deref(), Some("default-m"));
    assert_eq!(merged.max_iters, 99);
    assert_eq!(merged.default_skills, vec!["only-this".to_string()]);
}

#[tokio::test]
async fn validator_trait_is_object_safe_and_callable() {
    // The whole point of `dyn Validator` is to let us stack
    // heterogeneous validators in a `Vec<Arc<dyn Validator>>`.
    // This test is the contract: if it stops compiling we've
    // accidentally broken object-safety on the trait.
    let v: std::sync::Arc<dyn Validator> = std::sync::Arc::new(AlwaysPass);
    let out = TaskOutput::skeleton("t", "r");
    let verdict = v.validate(&out).await;
    assert!(verdict.passed);
    assert_eq!(verdict.validator, "always-pass");
}
