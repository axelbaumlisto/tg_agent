pub mod anthropic;
pub mod copilot;
pub mod error;
pub mod openai_compat;
pub mod resilient;

use std::pin::Pin;

use async_trait::async_trait;
use tokio_stream::Stream;

use crate::types::{ModelInfo, StreamChunk, ToolSpec};

#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub model: String,
    pub system: String,
    pub messages: Vec<serde_json::Value>,
    pub tools: Vec<serde_json::Value>,
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    /// Reasoning/thinking level: "off", "low", "medium", "high"
    pub reasoning: Option<String>,
}

#[async_trait]
pub trait Provider: Send + Sync {
    fn name(&self) -> &str;
    fn models(&self) -> Vec<ModelInfo>;

    async fn stream_chat(
        &self,
        request: ChatRequest,
    ) -> crate::error::Result<Pin<Box<dyn Stream<Item = StreamChunk> + Send>>>;

    /// Downcast to `ResilientProvider` for health diagnostics. Default: `None`.
    fn as_resilient(&self) -> Option<&resilient::ResilientProvider> {
        None
    }
}

pub fn tool_spec_to_anthropic_json(spec: &ToolSpec) -> serde_json::Value {
    serde_json::json!({
        "name": spec.name,
        "description": spec.description,
        "input_schema": spec.parameters,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Permission;

    #[test]
    fn tool_spec_to_json_format() {
        let spec = ToolSpec {
            name: "bash".into(),
            description: "Run a command".into(),
            parameters: serde_json::json!({"type": "object", "properties": {"command": {"type": "string"}}}),
            permission: Permission::Dangerous,
        };
        let json = tool_spec_to_anthropic_json(&spec);
        assert_eq!(json["name"], "bash");
        assert_eq!(json["description"], "Run a command");
        assert!(json["input_schema"]["properties"]["command"].is_object());
    }

    #[test]
    fn chat_request_debug() {
        let req = ChatRequest {
            model: "claude-sonnet-4".into(),
            system: "You are helpful".into(),
            messages: vec![],
            tools: vec![],
            max_tokens: 1024,
            temperature: None,
            reasoning: None,
        };
        let debug = format!("{req:?}");
        assert!(debug.contains("claude-sonnet-4"));
    }
}
