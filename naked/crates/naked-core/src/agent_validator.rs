//! Built-in [`crate::agent_role::Validator`] implementations.
//!
//! Each validator is a *named* function from [`TaskOutput`] to
//! [`ValidationVerdict`]. They are intentionally small and composable
//! — to combine them, stack them in a `Vec<Arc<dyn Validator>>` and
//! aggregate the verdicts at the call site (see
//! [`crate::agent_role`] doc).
//!
//! Adding a new validator should never require changes outside this
//! file — the trait is already object-safe and stateless.

use async_trait::async_trait;
use regex::Regex;

pub use crate::agent_role::Validator;
use crate::agent_role::{TaskOutput, ValidationVerdict};

/// Generic regex validator. Useful for cheap structural checks
/// ("output must mention `Total:`", "excerpt must include 'NDA'").
///
/// Matches against `output.text` by default; pass `field=Artifacts`
/// to match against the JSON-stringified artifacts blob instead
/// (handy when the artifact is the truth and `text` is just narration).
pub struct RegexValidator {
    name: String,
    pattern: Regex,
    field: RegexField,
    must_match: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum RegexField {
    Text,
    Artifacts,
}

impl RegexValidator {
    /// Construct a validator that **passes** when the regex matches.
    /// Use [`RegexValidator::reject_if_match`] for the inverse.
    pub fn require_match(
        name: impl Into<String>,
        pattern: &str,
        field: RegexField,
    ) -> Result<Self, regex::Error> {
        Ok(Self {
            name: name.into(),
            pattern: Regex::new(pattern)?,
            field,
            must_match: true,
        })
    }

    /// Construct a validator that **passes** when the regex does NOT
    /// match. Useful for forbidden-content checks (e.g. "no fake
    /// `Liên hệ qua` placeholders").
    pub fn reject_if_match(
        name: impl Into<String>,
        pattern: &str,
        field: RegexField,
    ) -> Result<Self, regex::Error> {
        Ok(Self {
            name: name.into(),
            pattern: Regex::new(pattern)?,
            field,
            must_match: false,
        })
    }

    fn target<'a>(&self, output: &'a TaskOutput) -> std::borrow::Cow<'a, str> {
        match self.field {
            RegexField::Text => std::borrow::Cow::Borrowed(output.text.as_str()),
            RegexField::Artifacts => std::borrow::Cow::Owned(output.artifacts.to_string()),
        }
    }
}

#[async_trait]
impl Validator for RegexValidator {
    fn name(&self) -> &str {
        &self.name
    }

    async fn validate(&self, output: &TaskOutput) -> ValidationVerdict {
        let target = self.target(output);
        let hit = self.pattern.is_match(&target);
        let pass = if self.must_match { hit } else { !hit };
        if pass {
            ValidationVerdict::pass(self.name.clone())
        } else if self.must_match {
            ValidationVerdict::fail(
                self.name.clone(),
                format!("required pattern `{}` not found", self.pattern.as_str()),
            )
        } else {
            ValidationVerdict::fail(
                self.name.clone(),
                format!("forbidden pattern `{}` matched", self.pattern.as_str()),
            )
        }
    }
}

/// Cheap, no-LLM Vietnamese-phone presence pre-filter. Passes when
/// the output contains at least one digit string that looks like a
/// real VN mobile / landline (10-11 digits starting with `0`).
///
/// **Not** the universal "did the agent find a contact" check — for
/// that, stack [`GatekeeperValidator`] on top. This regex exists as
/// an optional fast-path: zero LLM tokens, zero latency, catches the
/// trivial "no digits at all" failure mode before paying for a
/// gatekeeper call. Country-specific regex sibling validators
/// (`phone-th`, `phone-uk`, …) are intentionally NOT added — that
/// path doesn't generalise (every locale, every formatting
/// convention, every alphabet would need its own regex). The
/// gatekeeper handles all of them with one prompt.
pub struct PhoneVNValidator {
    pattern: Regex,
    field: RegexField,
}

impl Default for PhoneVNValidator {
    fn default() -> Self {
        Self::new(RegexField::Text)
    }
}

impl PhoneVNValidator {
    pub fn new(field: RegexField) -> Self {
        // Conservative regex: starts with `0`, then 9-10 more digits
        // (so 10-11 total), with optional separators in between.
        // Tuned to avoid matching 5-digit ZIP-like numbers and
        // 12+ digit ID strings.
        let pat = Regex::new(r"(?:^|[^\d])0\d{2,3}[ .\-]?\d{3,4}[ .\-]?\d{3,4}(?:[^\d]|$)")
            .expect("static regex must compile");
        Self {
            pattern: pat,
            field,
        }
    }
}

#[async_trait]
impl Validator for PhoneVNValidator {
    fn name(&self) -> &str {
        "phone-vn"
    }

    async fn validate(&self, output: &TaskOutput) -> ValidationVerdict {
        let target = match self.field {
            RegexField::Text => std::borrow::Cow::Borrowed(output.text.as_str()),
            RegexField::Artifacts => std::borrow::Cow::Owned(output.artifacts.to_string()),
        };
        if self.pattern.is_match(&target) {
            ValidationVerdict::pass(self.name())
        } else {
            ValidationVerdict::fail(
                self.name(),
                "no Vietnamese phone-shaped digit string found".to_string(),
            )
        }
    }
}

/// LLM-based universal acceptance validator. The "gatekeeper" — one
/// mechanism that handles ANY research domain (real-estate, used cars,
/// mopeds, jobs, products), ANY country (VN, TH, BR, US), ANY language.
///
/// You hand it:
///
/// * `criteria` — free-text description of what counts as a passing
///   output. This is the user's intent: "must contain a real seller
///   contact (phone or messenger), must match the topic <topic>, must
///   not be a captcha-blocked placeholder". Pulled from the spec's
///   `topic` + `notes` at the call site, no Rust changes per domain.
/// * a [`Provider`] + `model` — the gatekeeper LLM. Intentionally
///   separate from the worker model so operators can pin a cheap fast
///   model (e.g. `glm-flash`) for validation while the worker uses a
///   bigger model for browsing.
///
/// You get back a [`ValidationVerdict`] with structured reasoning.
///
/// Why not regex (PhoneVN/PhoneTH/Phone…): every locale + alphabet +
/// formatting convention would need its own regex, and the regex
/// can't tell "real digit string the agent saw on the page" from
/// "year 2026 in the listing date". The gatekeeper reads the whole
/// text and judges intent. One mechanism, infinite domains.
///
/// Cost / latency: one provider call per task, ~200-500 tokens of
/// output. With a fast model this is sub-second and pennies. Stack
/// after a cheap regex pre-filter (like [`PhoneVNValidator`]) when
/// you want to skip the LLM call on obvious failures.
///
/// Failure modes the validator handles gracefully:
///
/// * Provider returns malformed JSON → fail with the parse error in
///   `reasons` (so the operator notices and can lower the temperature
///   or switch model).
/// * Provider returns no text at all → fail with "empty response".
/// * Provider call errors → fail with the underlying error message
///   (does NOT propagate — a validator panicking would kill the whole
///   batch report).
pub struct GatekeeperValidator {
    name: String,
    provider: std::sync::Arc<dyn crate::provider::Provider>,
    model: String,
    criteria: String,
    /// Cap on `max_tokens` for the gatekeeper response. Most outputs
    /// fit in <300 tokens of JSON, but reasoning-capable models burn
    /// tokens on hidden CoT before emitting visible text — keep this
    /// generous (default 2048) so we don't cut off legitimate JSON
    /// after the thinking phase. Tighten via [`with_max_tokens`] when
    /// using a non-reasoning model where tokens cost real money.
    max_tokens: u32,
}

impl GatekeeperValidator {
    pub fn new(
        name: impl Into<String>,
        provider: std::sync::Arc<dyn crate::provider::Provider>,
        model: impl Into<String>,
        criteria: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            provider,
            model: model.into(),
            criteria: criteria.into(),
            max_tokens: 2048,
        }
    }

    pub fn with_max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = n;
        self
    }

    /// System prompt — pinned shape so the model returns JSON we can
    /// parse, never prose. `temperature=0` at call site to make the
    /// verdict deterministic for the same input.
    fn system_prompt() -> &'static str {
        "You are a strict acceptance reviewer. The user gives you a CRITERIA \
         (what counts as a passing output) and an OUTPUT produced by another \
         agent. Decide whether the OUTPUT satisfies the CRITERIA.\n\
         \n\
         Reply with a SINGLE JSON object and nothing else. Schema:\n\
         {\"passed\": <bool>, \"reasons\": [\"<short reason>\", ...]}\n\
         \n\
         Rules:\n\
         - `passed` is true only if EVERY criterion is met.\n\
         - `reasons` is required even when `passed` is true (briefly say why).\n\
         - Be specific: cite the exact phrase / value you accepted or rejected.\n\
         - If the OUTPUT is empty, `passed` is false with reason \"empty output\".\n\
         - Do NOT wrap the JSON in markdown fences or commentary."
    }

    /// User-message body. Kept small so the gatekeeper stays cheap;
    /// the OUTPUT block is truncated to the first `MAX_OUTPUT_CHARS`
    /// characters (more than enough for a listing-extraction task).
    fn user_prompt(&self, output: &TaskOutput) -> String {
        const MAX_OUTPUT_CHARS: usize = 8_000;
        let mut text_block = output.text.clone();
        if text_block.chars().count() > MAX_OUTPUT_CHARS {
            text_block = text_block.chars().take(MAX_OUTPUT_CHARS).collect();
            text_block.push_str("\n…(truncated)…");
        }
        let artifacts_block = if output.artifacts.is_null() {
            String::new()
        } else {
            let mut s = output.artifacts.to_string();
            if s.chars().count() > MAX_OUTPUT_CHARS {
                s = s.chars().take(MAX_OUTPUT_CHARS).collect();
                s.push_str("…(truncated)…");
            }
            format!("\n\nARTIFACTS (JSON):\n{s}")
        };
        format!(
            "CRITERIA:\n{}\n\nOUTPUT (assistant text):\n{}{}",
            self.criteria, text_block, artifacts_block,
        )
    }
}

#[async_trait]
impl Validator for GatekeeperValidator {
    fn name(&self) -> &str {
        &self.name
    }

    async fn validate(&self, output: &TaskOutput) -> ValidationVerdict {
        use tokio_stream::StreamExt;

        let user_msg = self.user_prompt(output);
        tracing::info!(
            target: "naked::validator::gatekeeper",
            validator = %self.name,
            task = %output.task_id,
            text_chars = output.text.chars().count(),
            artifacts_null = output.artifacts.is_null(),
            user_prompt_chars = user_msg.chars().count(),
            "[gatekeeper] sending verdict request: text_chars={} artifacts_null={}",
            output.text.chars().count(),
            output.artifacts.is_null()
        );
        tracing::debug!(
            target: "naked::validator::gatekeeper",
            validator = %self.name,
            task = %output.task_id,
            "[gatekeeper-input] CRITERIA+OUTPUT preview: {}",
            preview_for_log(&user_msg, 4_000)
        );
        let request = crate::provider::ChatRequest {
            model: self.model.clone(),
            system: Self::system_prompt().to_string(),
            messages: vec![serde_json::json!({
                "role": "user",
                "content": user_msg,
            })],
            tools: vec![],
            max_tokens: self.max_tokens,
            temperature: Some(0.0),
            // Leave reasoning at provider default. Forcing "off" on
            // hosts that don't recognise the field is harmless, but
            // forcing it on Aliyun/Dashscope sets `enable_thinking:
            // false` which on some models (qwen3.6-plus) causes the
            // visible-text channel to stay empty when the model
            // expected a thinking phase. Letting the provider pick
            // defaults keeps the gatekeeper portable across
            // OpenAI/Aliyun/Anthropic/glm without surprise empties.
            reasoning: None,
        };

        let mut stream = match self.provider.stream_chat(request).await {
            Ok(s) => s,
            Err(e) => {
                return ValidationVerdict::fail(
                    self.name.clone(),
                    format!("gatekeeper provider error: {e}"),
                );
            }
        };

        let mut text = String::new();
        let mut thinking = String::new();
        while let Some(chunk) = stream.next().await {
            match chunk {
                crate::types::StreamChunk::Text(t) => text.push_str(&t),
                crate::types::StreamChunk::Thinking(t) => thinking.push_str(&t),
                crate::types::StreamChunk::Done => break,
                crate::types::StreamChunk::Error(e) => {
                    return ValidationVerdict::fail(
                        self.name.clone(),
                        format!("gatekeeper stream error: {e}"),
                    );
                }
                _ => {}
            }
        }

        // Salvage path: when a model spends its whole budget thinking
        // and emits no visible text (qwen3.6-plus has been observed
        // doing this for short prompts), fall back to scanning the
        // thinking channel for the first JSON object. Better than
        // failing the verdict for what's really a stream-routing
        // quirk on the provider side.
        let mut trimmed = text.trim();
        let salvaged: String;
        if trimmed.is_empty()
            && !thinking.is_empty()
            && let Some(json) = extract_first_json_object(thinking.trim())
        {
            salvaged = json;
            trimmed = salvaged.as_str();
        }
        if trimmed.is_empty() {
            return ValidationVerdict::fail(
                self.name.clone(),
                "gatekeeper returned empty response".to_string(),
            );
        }

        // Some models wrap JSON in ```json fences despite instructions;
        // strip a single leading/trailing fence pair if present so we
        // don't fail validation for a presentational quirk.
        let stripped = strip_json_fences(trimmed);

        match serde_json::from_str::<GatekeeperReply>(stripped) {
            Ok(reply) => {
                let verdict = ValidationVerdict {
                    validator: self.name.clone(),
                    passed: reply.passed,
                    reasons: if reply.reasons.is_empty() {
                        vec![if reply.passed {
                            "ok".into()
                        } else {
                            "rejected".into()
                        }]
                    } else {
                        reply.reasons
                    },
                };
                tracing::info!(
                    target: "naked::validator::gatekeeper",
                    validator = %self.name,
                    task = %output.task_id,
                    passed = verdict.passed,
                    "[gatekeeper-verdict] passed={} reasons={:?}",
                    verdict.passed,
                    verdict.reasons
                );
                verdict
            }
            Err(e) => {
                let snippet: String = trimmed.chars().take(200).collect();
                tracing::warn!(
                    target: "naked::validator::gatekeeper",
                    validator = %self.name,
                    task = %output.task_id,
                    "[gatekeeper-parse-error] {e} :: {snippet}"
                );
                ValidationVerdict::fail(
                    self.name.clone(),
                    format!("gatekeeper response not parseable as JSON ({e}): {snippet}"),
                )
            }
        }
    }
}

/// Single-line preview of a multi-line input, truncated for log lines.
fn preview_for_log(s: &str, cap: usize) -> String {
    let cleaned: String = s
        .chars()
        .map(|c| {
            if c == '\n' || c == '\r' || c == '\t' {
                ' '
            } else {
                c
            }
        })
        .collect();
    if cleaned.chars().count() <= cap {
        cleaned
    } else {
        let head: String = cleaned.chars().take(cap).collect();
        format!("{head}…[+{} chars]", cleaned.chars().count() - cap)
    }
}

#[derive(Debug, serde::Deserialize)]
struct GatekeeperReply {
    passed: bool,
    #[serde(default)]
    reasons: Vec<String>,
}

/// Find the first balanced JSON object substring in `s`. Used as the
/// salvage path when the visible-text channel is empty but the
/// thinking channel contains the verdict JSON inline (e.g. wrapped
/// in `…thinking through this… {"passed": true, …} done.`).
///
/// Returns `Some(owned_json)` on success; `None` if no balanced `{…}`
/// could be found. Naive (no string-literal awareness) — fine because
/// the gatekeeper schema has no nested quoted braces inside reasons.
fn extract_first_json_object(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    let mut start: Option<usize> = None;
    let mut depth: i32 = 0;
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'{' => {
                if start.is_none() {
                    start = Some(i);
                }
                depth += 1;
            }
            b'}' if depth > 0 => {
                depth -= 1;
                if depth == 0
                    && let Some(s0) = start
                {
                    return Some(s[s0..=i].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

/// Strip a single ```json … ``` (or plain ``` … ```) fence wrapping
/// the response. Returns the inner content, or the original string if
/// no fence was found. Pure helper so we can unit-test it in isolation.
fn strip_json_fences(s: &str) -> &str {
    let trimmed = s.trim();
    let body = if let Some(rest) = trimmed.strip_prefix("```json") {
        rest
    } else if let Some(rest) = trimmed.strip_prefix("```") {
        rest
    } else {
        return trimmed;
    };
    let body = body.trim_start_matches('\n').trim_start_matches('\r');
    let body = body.strip_suffix("```").unwrap_or(body);
    body.trim()
}

/// Aggregate the verdicts of several validators on one output. Returns
/// `(all_passed, individual_verdicts)`. Used by the batch CLI when
/// printing a row per task.
///
/// Runs validators concurrently — most are CPU-cheap, but
/// [`Validator::validate`] is async on purpose so future validators
/// (URL liveness, gatekeeper-LLM) can fan out without blocking each
/// other.
pub async fn run_all(
    validators: &[std::sync::Arc<dyn Validator>],
    output: &TaskOutput,
) -> (bool, Vec<ValidationVerdict>) {
    use futures_util::future::join_all;
    let verdicts = join_all(validators.iter().map(|v| v.validate(output))).await;
    let all_passed = verdicts.iter().all(|v| v.passed);
    (all_passed, verdicts)
}

#[cfg(test)]
#[path = "agent_validator_tests.rs"]
mod tests;
