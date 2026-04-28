use tokio_stream::StreamExt;

use crate::provider::{ChatRequest, Provider};
use crate::types::StreamChunk;

use super::types::{MemoryScope, MemoryType};

const CLASSIFIER_SYSTEM_PROMPT_BASE: &str = "\
You are a memory classifier. Analyze the user message and determine if it contains \
a persistent preference, correction, project knowledge, or failure report that should \
be remembered across sessions.

Respond ONLY with JSON (no markdown fences, no explanation):
{\"is_memory\": false}
or
{\"is_memory\": true, \"type\": \"preference|correction|project_knowledge|failure\", \
\"content\": \"concise extracted rule\", \"scope\": \"project|global|user\"}

Guidelines:
- preference: user states how they like things done (style, tools, language, etc.)
- correction: user corrects agent behavior (\"don't do X\", \"always do Y instead\")
- project_knowledge: lasting facts about the codebase/infra (endpoints, DB type, deploy process)
- failure: recurring issue or gotcha (\"tests fail without X\", \"build breaks on Y\")
- scope global: applies to all projects (personal preference, general workflow)
- scope project: specific to the current codebase
- scope user: a fact about THIS particular author only (e.g. \"I prefer dark mode\", \
\"мой часовой пояс UTC+2\", \"for me use meters not feet\"). Only pick this when an author id \
is known (see context below).

Group chat prefix: messages from groups may start with `@username:` or `first_name:` — \
this is an attribution label the bot prepends, NOT part of the user's words. \
Strip it before extracting the rule. The user's preference still belongs to the \
named author (prefer scope=user when an author id is available).

Do NOT classify as memory:
- Ordinary task instructions (\"fix this bug\", \"add a feature\")
- Questions (\"what does this do?\")
- Short acknowledgements (\"ok\", \"yes\", \"done\")
- The attribution prefix itself (e.g. \"@alice:\" alone)";

const CLASSIFIER_SYSTEM_PROMPT_NO_USER: &str = "\n\nContext: no author id is available. \
Never use scope=user; pick project or global instead.";

const CLASSIFIER_SYSTEM_PROMPT_WITH_USER: &str = "\n\nContext: the current author id is available. \
You may use scope=user for facts that are specific to this author.";

pub struct ClassificationResult {
    pub memory_type: MemoryType,
    pub content: String,
    pub scope: MemoryScope,
}

/// Classify a user message to determine if it contains memory-worthy content.
/// Returns `None` if the message is not memory-worthy or classification fails.
///
/// `sender_id` — if `Some`, the classifier is allowed to pick `scope=user`, and
/// any `user` scope returned is resolved against this id. If `None`, `user`
/// scope is forbidden and falls back to `project`.
pub async fn classify(
    provider: &dyn Provider,
    model: &str,
    user_message: &str,
    sender_id: Option<&str>,
) -> Option<ClassificationResult> {
    if user_message.len() < 10 || user_message.starts_with('/') {
        return None;
    }

    let mut system = String::from(CLASSIFIER_SYSTEM_PROMPT_BASE);
    system.push_str(if sender_id.is_some() {
        CLASSIFIER_SYSTEM_PROMPT_WITH_USER
    } else {
        CLASSIFIER_SYSTEM_PROMPT_NO_USER
    });

    let request = ChatRequest {
        model: model.to_string(),
        system,
        messages: vec![serde_json::json!({
            "role": "user",
            "content": user_message,
        })],
        tools: vec![],
        max_tokens: 256,
        temperature: Some(0.0),
        reasoning: None,
    };

    let mut stream = match provider.stream_chat(request).await {
        Ok(s) => s,
        Err(e) => {
            tracing::debug!("memory classifier LLM call failed: {e}");
            return None;
        }
    };

    let mut text = String::new();
    while let Some(chunk) = stream.next().await {
        match chunk {
            StreamChunk::Text(t) => text.push_str(&t),
            StreamChunk::Done => break,
            StreamChunk::Error(e) => {
                tracing::debug!("memory classifier stream error: {e}");
                return None;
            }
            _ => {}
        }
    }

    parse_classification(&text, sender_id)
}

fn parse_classification(raw: &str, sender_id: Option<&str>) -> Option<ClassificationResult> {
    let trimmed = raw.trim();

    // Strip markdown fences if present
    let json_str = if trimmed.starts_with("```") {
        trimmed
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim()
    } else {
        trimmed
    };

    let val: serde_json::Value = serde_json::from_str(json_str).ok()?;

    if !val.get("is_memory")?.as_bool()? {
        return None;
    }

    let type_str = val.get("type")?.as_str()?;
    let content = val.get("content")?.as_str()?.to_string();
    let scope_str = val
        .get("scope")
        .and_then(|v| v.as_str())
        .unwrap_or("project")
        .to_ascii_lowercase();

    let memory_type: MemoryType = type_str.parse().ok()?;
    let scope: MemoryScope = match scope_str.as_str() {
        "global" => MemoryScope::Global,
        "project" => MemoryScope::Project,
        "user" => match sender_id {
            Some(id) if !id.is_empty() => MemoryScope::User(id.to_string()),
            // LLM picked user but we have no id — downgrade to project rather
            // than losing the memory entirely.
            _ => MemoryScope::Project,
        },
        _ => return None,
    };

    if content.is_empty() {
        return None;
    }

    Some(ClassificationResult {
        memory_type,
        content,
        scope,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_positive_classification() {
        let json = r#"{"is_memory": true, "type": "preference", "content": "Use tabs", "scope": "global"}"#;
        let result = parse_classification(json, None).unwrap();
        assert_eq!(result.memory_type, MemoryType::Preference);
        assert_eq!(result.content, "Use tabs");
        assert_eq!(result.scope, MemoryScope::Global);
    }

    #[test]
    fn parse_negative_classification() {
        let json = r#"{"is_memory": false}"#;
        assert!(parse_classification(json, None).is_none());
    }

    #[test]
    fn parse_with_markdown_fences() {
        let json = "```json\n{\"is_memory\": true, \"type\": \"correction\", \"content\": \"No unwrap\", \"scope\": \"project\"}\n```";
        let result = parse_classification(json, None).unwrap();
        assert_eq!(result.memory_type, MemoryType::Correction);
        assert_eq!(result.content, "No unwrap");
    }

    #[test]
    fn parse_invalid_json() {
        assert!(parse_classification("not json", None).is_none());
    }

    #[test]
    fn parse_empty_content() {
        let json =
            r#"{"is_memory": true, "type": "preference", "content": "", "scope": "project"}"#;
        assert!(parse_classification(json, None).is_none());
    }

    #[test]
    fn parse_unknown_type() {
        let json =
            r#"{"is_memory": true, "type": "unknown_type", "content": "test", "scope": "project"}"#;
        assert!(parse_classification(json, None).is_none());
    }

    #[test]
    fn parse_default_scope_is_project() {
        let json = r#"{"is_memory": true, "type": "failure", "content": "build breaks"}"#;
        let result = parse_classification(json, None).unwrap();
        assert_eq!(result.scope, MemoryScope::Project);
    }

    #[test]
    fn parse_user_scope_with_sender() {
        let json = r#"{"is_memory": true, "type": "preference", "content": "I prefer dark mode", "scope": "user"}"#;
        let result = parse_classification(json, Some("42")).unwrap();
        assert_eq!(result.scope, MemoryScope::User("42".into()));
    }

    #[test]
    fn parse_user_scope_without_sender_falls_back() {
        let json = r#"{"is_memory": true, "type": "preference", "content": "I prefer dark mode", "scope": "user"}"#;
        let result = parse_classification(json, None).unwrap();
        assert_eq!(result.scope, MemoryScope::Project);
    }

    #[test]
    fn parse_unknown_scope_rejected() {
        let json =
            r#"{"is_memory": true, "type": "preference", "content": "x", "scope": "banana"}"#;
        assert!(parse_classification(json, None).is_none());
    }
}
