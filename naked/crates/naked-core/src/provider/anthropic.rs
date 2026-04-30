use std::pin::Pin;

use async_trait::async_trait;
use futures_util::StreamExt;
use tokio_stream::Stream;

use crate::config::ProviderConfig;
use crate::error::{AgentError, Result};
use crate::types::{ModelInfo, StreamChunk, TurnUsage};

use super::{ChatRequest, Provider};

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const API_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: u32 = 16384;

pub struct AnthropicProvider {
    display_name: String,
    config: ProviderConfig,
    client: reqwest::Client,
    base_url: String,
}

impl AnthropicProvider {
    pub fn new(name: String, config: ProviderConfig) -> Self {
        let base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(10))
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .unwrap_or_default();
        Self {
            display_name: name,
            config,
            client,
            base_url,
        }
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn name(&self) -> &str {
        &self.display_name
    }

    fn models(&self) -> Vec<ModelInfo> {
        self.config
            .models_with_aliases()
            .into_iter()
            .map(|m| ModelInfo {
                provider: self.display_name.clone(),
                model_id: m.clone(),
                display_name: m,
            })
            .collect()
    }

    async fn stream_chat(
        &self,
        request: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = StreamChunk> + Send>>> {
        let max_tokens = if request.max_tokens > 0 {
            request.max_tokens
        } else {
            DEFAULT_MAX_TOKENS
        };

        let upstream_model = self.config.resolve_model_alias(&request.model);

        let mut body = serde_json::json!({
            "model": upstream_model,
            "max_tokens": max_tokens,
            "stream": true,
            "system": request.system,
            "messages": request.messages,
        });

        let reasoning_on = matches!(
            request.reasoning.as_deref(),
            Some("low" | "medium" | "high")
        );
        if reasoning_on {
            let budget = match request.reasoning.as_deref() {
                Some("low") => max_tokens.min(4096),
                Some("medium") => max_tokens.min(10240),
                _ => max_tokens.min(32768),
            };
            body["thinking"] = serde_json::json!({
                "type": "enabled",
                "budget_tokens": budget,
            });
        }

        if let Some(temp) = request.temperature
            && !reasoning_on
        {
            body["temperature"] = serde_json::json!(temp);
        }
        if !request.tools.is_empty() {
            body["tools"] = serde_json::Value::Array(request.tools);
        }

        let url = format!("{}/v1/messages", self.base_url);

        let mut req = self
            .client
            .post(&url)
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", API_VERSION)
            .header("content-type", "application/json")
            .header("accept", "text/event-stream");

        for (k, v) in &self.config.headers {
            req = req.header(k.as_str(), v.as_str());
        }

        let response = req
            .body(
                serde_json::to_string(&body)
                    .map_err(|e| AgentError::Provider(format!("serialize: {e}")))?,
            )
            .send()
            .await
            .map_err(|e| AgentError::Provider(format!("HTTP error: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response
                .text()
                .await
                .unwrap_or_else(|_| "no body".to_string());
            return Err(AgentError::Provider(format!(
                "Anthropic API {status}: {text}"
            )));
        }

        let stream = sse_stream_from_response(response);
        Ok(Box::pin(stream))
    }
}

fn sse_stream_from_response(response: reqwest::Response) -> impl Stream<Item = StreamChunk> + Send {
    async_stream::stream! {
        let byte_stream = response.bytes_stream();
        let io_stream = byte_stream.map(|r| r.map_err(|e| std::io::Error::other(e.to_string())));
        let reader = tokio_util::io::StreamReader::new(io_stream);
        let mut lines = tokio::io::AsyncBufReadExt::lines(tokio::io::BufReader::new(reader));

        let mut pending_tool_id = String::new();
        let mut pending_tool_name = String::new();
        let mut pending_tool_json = String::new();
        let mut has_pending_tool = false;

        while let Ok(Some(line)) = lines.next_line().await {
            let line = line.trim().to_string();
            if line.is_empty() || line.starts_with(':') {
                continue;
            }

            let data = line
                .strip_prefix("data: ")
                .or_else(|| line.strip_prefix("data:"));
            let data = match data {
                Some(d) => d.trim(),
                None => continue,
            };

            if data == "[DONE]" {
                yield StreamChunk::Done;
                return;
            }

            let event: serde_json::Value = match serde_json::from_str(data) {
                Ok(v) => v,
                Err(_) => continue,
            };

            let event_type = event.get("type").and_then(|v| v.as_str()).unwrap_or("");

            match event_type {
                "content_block_start" => {
                    let block = &event["content_block"];
                    match block.get("type").and_then(|v| v.as_str()) {
                        Some("tool_use") => {
                            pending_tool_id = block["id"].as_str().unwrap_or("").to_string();
                            pending_tool_name = block["name"].as_str().unwrap_or("").to_string();
                            pending_tool_json.clear();
                            has_pending_tool = true;
                        }
                        Some("thinking") => {
                            if let Some(text) = block.get("thinking").and_then(|v| v.as_str())
                                && !text.is_empty()
                            {
                                yield StreamChunk::Thinking(text.to_string());
                            }
                        }
                        _ => {}
                    }
                }
                "content_block_delta" => {
                    let delta = &event["delta"];
                    match delta.get("type").and_then(|v| v.as_str()) {
                        Some("text_delta") => {
                            if let Some(text) = delta["text"].as_str()
                                && !text.is_empty()
                            {
                                yield StreamChunk::Text(text.to_string());
                            }
                        }
                        Some("thinking_delta") => {
                            if let Some(text) = delta["thinking"].as_str()
                                && !text.is_empty()
                            {
                                yield StreamChunk::Thinking(text.to_string());
                            }
                        }
                        Some("input_json_delta") => {
                            if has_pending_tool
                                && let Some(partial) = delta["partial_json"].as_str()
                            {
                                pending_tool_json.push_str(partial);
                            }
                        }
                        _ => {}
                    }
                }
                "content_block_stop" => {
                    if has_pending_tool {
                        let input: serde_json::Value =
                            serde_json::from_str(&pending_tool_json).unwrap_or(serde_json::json!({}));
                        yield StreamChunk::ToolUse {
                            id: std::mem::take(&mut pending_tool_id),
                            name: std::mem::take(&mut pending_tool_name),
                            input,
                        };
                        pending_tool_json.clear();
                        has_pending_tool = false;
                    }
                }
                "message_delta" => {
                    if let Some(usage) = event.get("usage") {
                        yield StreamChunk::Usage(parse_usage(usage));
                    }
                }
                "message_start" => {
                    if let Some(usage) = event.pointer("/message/usage") {
                        yield StreamChunk::Usage(parse_usage(usage));
                    }
                }
                "message_stop" => {
                    yield StreamChunk::Done;
                    return;
                }
                "error" => {
                    let msg = event
                        .get("error")
                        .and_then(|e| e.get("message"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown error");
                    yield StreamChunk::Error(msg.to_string());
                    return;
                }
                _ => {}
            }
        }
    }
}

fn parse_usage(usage: &serde_json::Value) -> TurnUsage {
    TurnUsage {
        input_tokens: usage["input_tokens"].as_u64().unwrap_or(0),
        output_tokens: usage["output_tokens"].as_u64().unwrap_or(0),
        cache_read_tokens: usage["cache_read_input_tokens"].as_u64().unwrap_or(0),
        cache_write_tokens: usage["cache_creation_input_tokens"].as_u64().unwrap_or(0),
    }
}
