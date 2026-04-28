use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: Option<u64>,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

impl JsonRpcRequest {
    pub fn new(id: u64, method: impl Into<String>, params: Option<serde_json::Value>) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id: Some(id),
            method: method.into(),
            params,
        }
    }

    pub fn notification(method: impl Into<String>, params: Option<serde_json::Value>) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id: None,
            method: method.into(),
            params,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

impl JsonRpcResponse {
    pub fn is_error(&self) -> bool {
        self.error.is_some()
    }

    pub fn into_result(self) -> crate::error::Result<serde_json::Value> {
        if let Some(err) = self.error {
            Err(crate::error::AgentError::Provider(format!(
                "MCP JSON-RPC error {}: {}",
                err.code, err.message
            )))
        } else {
            Ok(self.result.unwrap_or(serde_json::Value::Null))
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolInfo {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default, rename = "inputSchema")]
    pub input_schema: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolsListResult {
    pub tools: Vec<McpToolInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolCallResult {
    #[serde(default)]
    pub content: Vec<McpContent>,
    #[serde(default, rename = "isError")]
    pub is_error: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum McpContent {
    Text { text: String },
    Image { data: String, mime_type: String },
    Resource { uri: String, text: Option<String> },
}

impl McpContent {
    pub fn as_text(&self) -> Option<&str> {
        match self {
            McpContent::Text { text } => Some(text),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_serialization() {
        let req = JsonRpcRequest::new(1, "tools/list", None);
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"jsonrpc\":\"2.0\""));
        assert!(json.contains("\"method\":\"tools/list\""));
        assert!(json.contains("\"id\":1"));
    }

    #[test]
    fn notification_has_no_id() {
        let req = JsonRpcRequest::notification("notifications/initialized", None);
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"id\":null"));
    }

    #[test]
    fn response_into_result_success() {
        let resp = JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id: Some(1),
            result: Some(serde_json::json!({"tools": []})),
            error: None,
        };
        assert!(!resp.is_error());
        let val = resp.into_result().unwrap();
        assert!(val["tools"].is_array());
    }

    #[test]
    fn response_into_result_error() {
        let resp = JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id: Some(1),
            result: None,
            error: Some(JsonRpcError {
                code: -32600,
                message: "Invalid Request".into(),
                data: None,
            }),
        };
        assert!(resp.is_error());
        assert!(resp.into_result().is_err());
    }

    #[test]
    fn tool_info_deserialize() {
        let json = r#"{"name": "read_file", "description": "Read a file", "inputSchema": {"type": "object"}}"#;
        let info: McpToolInfo = serde_json::from_str(json).unwrap();
        assert_eq!(info.name, "read_file");
        assert_eq!(info.description.as_deref(), Some("Read a file"));
    }

    #[test]
    fn tool_call_result_deserialize() {
        let json = r#"{"content": [{"type": "text", "text": "hello"}], "isError": false}"#;
        let result: McpToolCallResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.content.len(), 1);
        assert_eq!(result.content[0].as_text(), Some("hello"));
        assert!(!result.is_error);
    }

    #[test]
    fn mcp_content_text() {
        let c = McpContent::Text { text: "hi".into() };
        assert_eq!(c.as_text(), Some("hi"));
    }

    #[test]
    fn mcp_content_image_no_text() {
        let c = McpContent::Image {
            data: "base64".into(),
            mime_type: "image/png".into(),
        };
        assert_eq!(c.as_text(), None);
    }
}
