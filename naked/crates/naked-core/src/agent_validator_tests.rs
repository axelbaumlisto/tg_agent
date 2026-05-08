use super::*;
use crate::agent_role::TaskOutput;

fn output_with_text(t: &str) -> TaskOutput {
    let mut o = TaskOutput::skeleton("t", "r");
    o.text = t.to_string();
    o
}

fn output_with_artifacts(j: serde_json::Value) -> TaskOutput {
    let mut o = TaskOutput::skeleton("t", "r");
    o.artifacts = j;
    o
}

#[tokio::test]
async fn regex_require_match_passes_when_pattern_present() {
    let v = RegexValidator::require_match("has-total", r"Total:\s*\d+", RegexField::Text).unwrap();
    let verdict = v.validate(&output_with_text("Total: 42 done.")).await;
    assert!(verdict.passed, "{verdict:?}");
    assert_eq!(verdict.validator, "has-total");
}

#[tokio::test]
async fn regex_require_match_fails_when_pattern_absent() {
    let v = RegexValidator::require_match("has-total", r"Total:\s*\d+", RegexField::Text).unwrap();
    let verdict = v.validate(&output_with_text("nothing here")).await;
    assert!(!verdict.passed);
    assert!(verdict.reasons[0].contains("not found"));
}

#[tokio::test]
async fn regex_reject_if_match_passes_when_pattern_absent() {
    // The "no fake Liên hệ qua placeholder" check the gatekeeper
    // currently does inline — extracted as a reusable validator.
    let v = RegexValidator::reject_if_match("no-fake-contact", r"Liên hệ qua", RegexField::Text)
        .unwrap();
    let clean = output_with_text("Phone: 0905 123 456");
    let verdict = v.validate(&clean).await;
    assert!(verdict.passed, "{verdict:?}");
}

#[tokio::test]
async fn regex_reject_if_match_fails_when_pattern_present() {
    let v = RegexValidator::reject_if_match("no-fake-contact", r"Liên hệ qua", RegexField::Text)
        .unwrap();
    let dirty = output_with_text("Liên hệ qua website");
    let verdict = v.validate(&dirty).await;
    assert!(!verdict.passed);
    assert!(verdict.reasons[0].contains("forbidden pattern"));
}

#[tokio::test]
async fn regex_can_target_artifacts_field() {
    let v = RegexValidator::require_match("has-url", r"https://", RegexField::Artifacts).unwrap();
    let out = output_with_artifacts(serde_json::json!({"url": "https://example.com"}));
    assert!(v.validate(&out).await.passed);
}

#[tokio::test]
async fn phone_vn_accepts_real_mobile() {
    let v = PhoneVNValidator::default();
    for sample in [
        "Contact: 0905 123 456",
        "Liên hệ: 0987.654.321",
        "Anh Nam — 0901-234-567",
        "0938 123 4567 (Zalo)",
        "phone 0292 1234567 (landline)",
    ] {
        let verdict = v.validate(&output_with_text(sample)).await;
        assert!(verdict.passed, "should accept `{sample}` — {verdict:?}");
    }
}

#[tokio::test]
async fn phone_vn_rejects_non_phone_digits() {
    let v = PhoneVNValidator::default();
    for sample in [
        "no phone here",
        "Total: 42",
        "Year 2026",
        // 5-digit ZIP-like number (too short)
        "ZIP 50000",
        // The captcha-prefix sentence by itself — common in our
        // saved findings, MUST NOT be misread as a phone.
        "Contacts hidden behind site captcha — visit URL",
        // ID-shaped number (too long)
        "ID: 12345678901234",
    ] {
        let verdict = v.validate(&output_with_text(sample)).await;
        assert!(!verdict.passed, "should reject `{sample}` — {verdict:?}");
    }
}

#[tokio::test]
async fn phone_vn_can_scan_artifacts_field() {
    // Real probe scenario: the agent saved a finding into a
    // structured artifacts blob, the narration text is empty.
    let v = PhoneVNValidator::new(RegexField::Artifacts);
    let out = output_with_artifacts(serde_json::json!({
        "findings": [{"phone": "0905 123 456"}]
    }));
    assert!(v.validate(&out).await.passed);
}

// ----- GatekeeperValidator scaffolding & tests -----------------
// We test the gatekeeper with a scripted Provider so we never hit
// a real LLM in unit tests. This proves prompt construction,
// response parsing, fence stripping, and graceful failure on
// malformed/empty/error responses — without paying for tokens.

use crate::provider::{ChatRequest, Provider};
use crate::types::{ModelInfo, StreamChunk};
use std::pin::Pin;
use std::sync::Mutex;
use tokio_stream::Stream;

struct ScriptedProvider {
    reply: Mutex<Result<String, String>>, // Ok(text) or Err(error message)
    captured: Mutex<Vec<ChatRequest>>,
}

impl ScriptedProvider {
    fn ok(text: impl Into<String>) -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self {
            reply: Mutex::new(Ok(text.into())),
            captured: Mutex::new(Vec::new()),
        })
    }
    fn err(msg: impl Into<String>) -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self {
            reply: Mutex::new(Err(msg.into())),
            captured: Mutex::new(Vec::new()),
        })
    }
}

#[async_trait]
impl Provider for ScriptedProvider {
    fn name(&self) -> &str {
        "scripted"
    }
    fn models(&self) -> Vec<ModelInfo> {
        vec![]
    }
    async fn stream_chat(
        &self,
        request: ChatRequest,
    ) -> crate::error::Result<Pin<Box<dyn Stream<Item = StreamChunk> + Send>>> {
        self.captured.lock().unwrap().push(request);
        match self.reply.lock().unwrap().clone() {
            Ok(text) => {
                let stream = tokio_stream::iter(vec![StreamChunk::Text(text), StreamChunk::Done]);
                Ok(Box::pin(stream))
            }
            Err(msg) => Err(crate::error::AgentError::ProviderTyped(
                crate::provider::error::ProviderError::Other {
                    status: 0,
                    body: msg,
                },
            )),
        }
    }
}

#[test]
fn extract_first_json_object_handles_common_wrappings() {
    // Plain object — passthrough.
    assert_eq!(
        extract_first_json_object("{\"a\":1}").as_deref(),
        Some("{\"a\":1}")
    );
    // Inside reasoning prose.
    assert_eq!(
        extract_first_json_object("Hmm, thinking… {\"passed\": true, \"reasons\": [\"ok\"]} done.")
            .as_deref(),
        Some("{\"passed\": true, \"reasons\": [\"ok\"]}")
    );
    // Nested objects — must return the OUTER balanced one.
    assert_eq!(
        extract_first_json_object("prefix {\"a\":{\"b\":1}} suffix").as_deref(),
        Some("{\"a\":{\"b\":1}}")
    );
    // Nothing balanced.
    assert_eq!(extract_first_json_object("no braces here"), None);
    assert_eq!(extract_first_json_object("only opening { no close"), None);
}

#[derive(Default)]
struct ScriptedThinkingProvider {
    text: Mutex<String>,
    thinking: Mutex<String>,
}

#[async_trait]
impl Provider for ScriptedThinkingProvider {
    fn name(&self) -> &str {
        "scripted-thinking"
    }
    fn models(&self) -> Vec<ModelInfo> {
        vec![]
    }
    async fn stream_chat(
        &self,
        _request: ChatRequest,
    ) -> crate::error::Result<Pin<Box<dyn Stream<Item = StreamChunk> + Send>>> {
        let chunks = vec![
            StreamChunk::Thinking(self.thinking.lock().unwrap().clone()),
            StreamChunk::Text(self.text.lock().unwrap().clone()),
            StreamChunk::Done,
        ];
        Ok(Box::pin(tokio_stream::iter(chunks)))
    }
}

#[tokio::test]
async fn gatekeeper_salvages_json_from_thinking_when_text_is_empty() {
    // Provider emits the verdict JSON in the thinking channel and
    // sends an empty text channel — the salvage path must lift
    // the JSON out so we don't fail a perfectly valid response.
    let provider = std::sync::Arc::new(ScriptedThinkingProvider {
        text: Mutex::new(String::new()),
        thinking: Mutex::new(
            "Let me think… The output mentions a phone, criteria say phone needed → \
             {\"passed\": true, \"reasons\": [\"phone present\"]} that's my call."
                .to_string(),
        ),
    });
    let v = GatekeeperValidator::new(
        "gk",
        provider as std::sync::Arc<dyn Provider>,
        "any-model",
        "Output must contain a real seller phone.",
    );
    let verdict = v.validate(&output_with_text("0905 123 456")).await;
    assert!(verdict.passed, "{verdict:?}");
    assert!(verdict.reasons[0].contains("phone"));
}

#[test]
fn strip_json_fences_handles_common_wrappings() {
    assert_eq!(strip_json_fences("{\"a\":1}"), "{\"a\":1}");
    assert_eq!(strip_json_fences("```json\n{\"a\":1}\n```"), "{\"a\":1}");
    assert_eq!(strip_json_fences("```\n{\"a\":1}\n```"), "{\"a\":1}");
    // No fence: passthrough trimmed.
    assert_eq!(strip_json_fences("  {\"a\":1}  "), "{\"a\":1}");
}

#[tokio::test]
async fn gatekeeper_passes_when_llm_says_passed_true() {
    let provider = ScriptedProvider::ok(
        r#"{"passed": true, "reasons": ["phone 0905 visible, matches topic"]}"#,
    );
    let v = GatekeeperValidator::new(
        "gk",
        provider.clone(),
        "any-model",
        "Output must contain a real seller phone.",
    );
    let out = output_with_text("Anh Nam — 0905 123 456 — listing #42");
    let verdict = v.validate(&out).await;
    assert!(verdict.passed, "{verdict:?}");
    assert_eq!(verdict.validator, "gk");
    assert!(verdict.reasons[0].contains("phone"));

    // Captured request shape: model + system prompt fragments + the
    // criteria + the output text appear in the user message.
    let captured = provider.captured.lock().unwrap();
    assert_eq!(captured.len(), 1, "must call provider exactly once");
    let req = &captured[0];
    assert_eq!(req.model, "any-model");
    assert!(req.system.contains("acceptance reviewer"));
    assert!(req.tools.is_empty(), "validator must not advertise tools");
    let user_msg = req.messages[0]["content"].as_str().unwrap();
    assert!(user_msg.contains("CRITERIA:"));
    assert!(user_msg.contains("real seller phone"));
    assert!(user_msg.contains("0905 123 456"));
}

#[tokio::test]
async fn gatekeeper_fails_when_llm_says_passed_false() {
    let provider = ScriptedProvider::ok(
        r#"{"passed": false, "reasons": ["no phone digits visible", "captcha placeholder used"]}"#,
    );
    let v = GatekeeperValidator::new(
        "gk",
        provider,
        "any-model",
        "Output must contain a real seller phone.",
    );
    let out = output_with_text("Liên hệ qua website (no real phone)");
    let verdict = v.validate(&out).await;
    assert!(!verdict.passed);
    assert_eq!(verdict.reasons.len(), 2);
    assert!(verdict.reasons.iter().any(|r| r.contains("captcha")));
}

#[tokio::test]
async fn gatekeeper_strips_json_markdown_fences() {
    // Some models stubbornly wrap responses despite the system
    // prompt asking otherwise. Validator must tolerate this.
    let provider = ScriptedProvider::ok("```json\n{\"passed\": true, \"reasons\": [\"ok\"]}\n```");
    let v = GatekeeperValidator::new("gk", provider, "m", "anything");
    let verdict = v.validate(&output_with_text("text")).await;
    assert!(verdict.passed, "{verdict:?}");
}

#[tokio::test]
async fn gatekeeper_fails_gracefully_on_malformed_json() {
    let provider = ScriptedProvider::ok("not a json blob, sorry");
    let v = GatekeeperValidator::new("gk", provider, "m", "anything");
    let verdict = v.validate(&output_with_text("text")).await;
    assert!(!verdict.passed);
    assert!(verdict.reasons[0].contains("not parseable"));
    // Snippet of the offending response is included for the operator.
    assert!(verdict.reasons[0].contains("not a json blob"));
}

#[tokio::test]
async fn gatekeeper_fails_gracefully_on_empty_response() {
    let provider = ScriptedProvider::ok("");
    let v = GatekeeperValidator::new("gk", provider, "m", "anything");
    let verdict = v.validate(&output_with_text("text")).await;
    assert!(!verdict.passed);
    assert!(verdict.reasons[0].contains("empty response"));
}

#[tokio::test]
async fn gatekeeper_fails_gracefully_on_provider_error() {
    // A provider error must NEVER panic the batch — it converts
    // to a fail verdict so the report still renders.
    let provider = ScriptedProvider::err("rate limited (429)");
    let v = GatekeeperValidator::new("gk", provider, "m", "anything");
    let verdict = v.validate(&output_with_text("text")).await;
    assert!(!verdict.passed);
    assert!(verdict.reasons[0].contains("rate limited"));
}

#[tokio::test]
async fn gatekeeper_truncates_large_outputs_to_keep_token_cost_bounded() {
    // Output >> the 8k char cap; the prompt must be truncated.
    let provider = ScriptedProvider::ok("{\"passed\": true, \"reasons\": [\"ok\"]}");
    let v = GatekeeperValidator::new("gk", provider.clone(), "m", "any");
    let huge = "x".repeat(20_000);
    let _ = v.validate(&output_with_text(&huge)).await;
    let captured = provider.captured.lock().unwrap();
    let user_msg = captured[0].messages[0]["content"].as_str().unwrap();
    assert!(user_msg.contains("…(truncated)…"));
    assert!(
        user_msg.chars().count() < 12_000,
        "user prompt should be bounded ~8k chars, got {}",
        user_msg.chars().count()
    );
}

#[tokio::test]
async fn gatekeeper_can_be_combined_with_regex_in_run_all() {
    // Stack a cheap regex pre-filter with the gatekeeper. Both
    // pass → all_pass. Either fails → all_pass = false.
    let provider = ScriptedProvider::ok("{\"passed\": true, \"reasons\": [\"looks good\"]}");
    let validators: Vec<std::sync::Arc<dyn Validator>> = vec![
        std::sync::Arc::new(PhoneVNValidator::default()),
        std::sync::Arc::new(GatekeeperValidator::new("gk", provider, "m", "any")),
    ];
    let good = output_with_text("Phone: 0905 123 456 (real)");
    let (all_pass, verdicts) = run_all(&validators, &good).await;
    assert!(all_pass, "{verdicts:?}");
    assert_eq!(verdicts.len(), 2);
}

#[tokio::test]
async fn run_all_aggregates_pass_and_fail_verdicts() {
    let validators: Vec<std::sync::Arc<dyn Validator>> = vec![
        std::sync::Arc::new(PhoneVNValidator::default()),
        std::sync::Arc::new(
            RegexValidator::reject_if_match("no-fake", r"Liên hệ qua", RegexField::Text).unwrap(),
        ),
    ];
    // Output that fails BOTH validators.
    let bad = output_with_text("Liên hệ qua website (no real phone)");
    let (all_pass, verdicts) = run_all(&validators, &bad).await;
    assert!(!all_pass);
    assert_eq!(verdicts.len(), 2);
    assert!(
        verdicts
            .iter()
            .any(|v| v.validator == "phone-vn" && !v.passed)
    );
    assert!(
        verdicts
            .iter()
            .any(|v| v.validator == "no-fake" && !v.passed)
    );

    // Output that passes both.
    let good = output_with_text("Owner: 0905 123 456 (call directly)");
    let (all_pass, verdicts) = run_all(&validators, &good).await;
    assert!(all_pass);
    assert!(verdicts.iter().all(|v| v.passed));
}
