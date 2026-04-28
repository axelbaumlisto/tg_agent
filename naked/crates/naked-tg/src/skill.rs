//! Envelope for skill-tool results so the downstream agent persona
//! can never confuse which chat a quote came from.
//!
//! Born from incident `2026-04-21`: the Income persona claimed it
//! "saw 3 dog photos in the Income group" when those photos actually
//! lived in the DM with `@avprilipko`. Root cause: the
//! `telegram-reader` skill returned untagged JSON, and the persona
//! prompt had no structural cue to reach for — it picked the most
//! recently mentioned chat title and hallucinated the rest.
//!
//! The envelope pins source metadata in a fixed shape so the prompt
//! layer can enforce: **every quote from a skill must print its
//! `source_chat_id` / `source_chat_title` in the first line.**

use serde_json::{Value, json};

/// Carrier for a skill result + provenance. The `render_for_prompt`
/// helper produces what the coordinator feeds to the LLM; `to_json`
/// produces the raw payload if a caller wants to plug the envelope
/// into a wider JSON tool-result shape.
#[derive(Debug, Clone)]
pub struct SkillResultEnvelope {
    pub source: SkillSource,
    pub data: Value,
}

#[derive(Debug, Clone)]
pub struct SkillSource {
    pub chat_id: i64,
    pub chat_title: String,
    pub session: String,
    pub skill: String,
}

impl SkillResultEnvelope {
    /// Build an envelope for a `telegram-reader` skill payload.
    pub fn wrap_telegram_reader(
        chat_id: i64,
        chat_title: impl Into<String>,
        session: impl Into<String>,
        payload: serde_json::Value,
    ) -> Self {
        Self {
            source: SkillSource {
                chat_id,
                chat_title: chat_title.into(),
                session: session.into(),
                skill: "telegram-reader".to_string(),
            },
            data: payload,
        }
    }

    /// Human-facing rendering the LLM sees in a `tool_result`. Source
    /// metadata is on the first few lines so the prompt rule
    /// "источник в первой строке" is structurally satisfied.
    pub fn render_for_prompt(&self) -> String {
        let pretty =
            serde_json::to_string_pretty(&self.data).unwrap_or_else(|_| self.data.to_string());
        format!(
            "chat_id={}\nchat_title={}\nsession={}\nskill={}\n---\n{pretty}",
            self.source.chat_id, self.source.chat_title, self.source.session, self.source.skill,
        )
    }

    /// Raw JSON representation (for callers that want to embed the
    /// envelope inside a larger JSON tool-result shape).
    pub fn to_json(&self) -> Value {
        json!({
            "source": {
                "chat_id": self.source.chat_id,
                "chat_title": self.source.chat_title,
                "session": self.source.session,
                "skill": self.source.skill,
            },
            "data": self.data,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn render_exposes_source_fields_up_top() {
        let env = SkillResultEnvelope::wrap_telegram_reader(
            -5084292206,
            "Income",
            "zverozabr_session",
            json!({"messages":[{"id":1,"text":"hi"}]}),
        );
        let out = env.render_for_prompt();
        let first_four: Vec<&str> = out.lines().take(4).collect();
        assert_eq!(first_four[0], "chat_id=-5084292206");
        assert_eq!(first_four[1], "chat_title=Income");
        assert_eq!(first_four[2], "session=zverozabr_session");
        assert_eq!(first_four[3], "skill=telegram-reader");
        assert!(out.contains("\"messages\""));
    }
}
