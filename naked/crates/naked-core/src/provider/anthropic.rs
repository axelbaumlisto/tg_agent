use std::pin::Pin;

use async_trait::async_trait;
use futures_util::StreamExt;
use tokio_stream::Stream;

use crate::config::ProviderConfig;
use crate::error::{AgentError, Result};
use crate::types::{ModelInfo, StreamChunk, TurnUsage};

use super::{ChatRequest, Provider, build_streaming_http_client, should_send_temperature};

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
        let client = build_streaming_http_client();
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

    fn key_hint(&self) -> Option<String> {
        // B46: expose literal api_key for dead-key auto-persist.
        Some(self.config.api_key.clone())
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

        let mut body = build_anthropic_request_body(
            &request,
            max_tokens,
            upstream_model,
            self.config
                .capabilities_for(&request.model)
                .supports_thinking,
        );
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
            .body(serde_json::to_string(&body).map_err(|e| {
                AgentError::ProviderTyped(super::error::ProviderError::Serialize {
                    context: "anthropic request".into(),
                    source: e.to_string(),
                })
            })?)
            .send()
            .await
            .map_err(|e| {
                AgentError::ProviderTyped(super::error::classify_provider_body(0, &e.to_string()))
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response
                .text()
                .await
                .unwrap_or_else(|_| "no body".to_string());
            return Err(AgentError::ProviderTyped(
                super::error::ProviderError::from_llm_http(
                    status.as_u16(),
                    &text[..text.len().min(512)],
                    &request.model,
                ),
            ));
        }

        let stream = sse_stream_from_response(response);
        Ok(Box::pin(stream))
    }
}

fn build_anthropic_request_body(
    request: &ChatRequest,
    max_tokens: u32,
    upstream_model: &str,
    model_caps_supports_thinking: bool,
) -> serde_json::Value {
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

    let model_supports_thinking =
        model_caps_supports_thinking || upstream_model.contains("claude-");
    if should_send_temperature(request.temperature, reasoning_on, model_supports_thinking)
        && let Some(temp) = request.temperature
    {
        body["temperature"] = serde_json::json!(temp);
    }
    body
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
        let mut saw_done = false;

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

            // Some proxy bridges in front of Anthropic emit OpenAI-style [DONE]
            // instead of `message_stop`. Both are explicit terminal markers
            // (B65); only a silent close with no marker is an error.
            if data == "[DONE]" {
                saw_done = true;
                yield StreamChunk::Done;
                break;
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
                "content_block_stop" if has_pending_tool => {
                        let input: serde_json::Value =
                            crate::tool::arg_repair::repair_json(&pending_tool_json);
                        yield StreamChunk::ToolUse {
                            id: std::mem::take(&mut pending_tool_id),
                            name: std::mem::take(&mut pending_tool_name),
                            input,
                        };
                        pending_tool_json.clear();
                        has_pending_tool = false;
                }
                "content_block_stop" => {}
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
                    saw_done = true;
                    yield StreamChunk::Done;
                    break;
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

        if !saw_done {
            yield StreamChunk::Error("SSE stream closed without [DONE] or content".into());
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_usage_full() {
        let u = serde_json::json!({
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 30,
            "cache_creation_input_tokens": 10
        });
        let tu = parse_usage(&u);
        assert_eq!(tu.input_tokens, 100);
        assert_eq!(tu.output_tokens, 50);
        assert_eq!(tu.cache_read_tokens, 30);
        assert_eq!(tu.cache_write_tokens, 10);
    }

    #[test]
    fn parse_usage_missing_fields() {
        let u = serde_json::json!({});
        let tu = parse_usage(&u);
        assert_eq!(tu.input_tokens, 0);
        assert_eq!(tu.output_tokens, 0);
    }

    #[test]
    fn provider_name_matches() {
        let cfg = crate::config::ProviderConfig {
            provider_type: "anthropic".into(),
            api_key: "test".into(),
            ..Default::default()
        };
        let p = AnthropicProvider::new("test-anthropic".into(), cfg);
        assert_eq!(p.name(), "test-anthropic");
    }

    fn b71_request(model: &str, temperature: Option<f32>, reasoning: Option<&str>) -> ChatRequest {
        // REGISTRY-WAIVE: B16 — ChatRequest has no Default derive; provider tests construct it exhaustively.
        ChatRequest {
            model: model.into(),
            system: String::new(),
            messages: vec![serde_json::json!({"role": "user", "content": "hi"})],
            tools: Vec::new(),
            max_tokens: 1024,
            temperature,
            reasoning: reasoning.map(str::to_string),
        }
    }

    #[test]
    fn b71_anthropic_reasoning_omits_temperature() {
        let request = b71_request("claude-sonnet-4", Some(0.7), Some("medium"));
        let body =
            build_anthropic_request_body(&request, request.max_tokens, "claude-sonnet-4-6", true);
        assert_eq!(body["thinking"]["type"], serde_json::json!("enabled"));
        assert!(body.get("temperature").is_none());
    }

    #[test]
    fn b71_anthropic_thinking_model_rejects_temperature_one_when_reasoning_off() {
        let request = b71_request("claude-sonnet-4", Some(1.0), Some("off"));
        let body =
            build_anthropic_request_body(&request, request.max_tokens, "claude-sonnet-4-6", true);
        assert!(body.get("thinking").is_none());
        assert!(body.get("temperature").is_none());
    }

    #[test]
    fn b81_anthropic_thinking_model_omits_temperature_zero_no_reasoning() {
        // B81: a thinking-capable model must not receive an explicit
        // temperature even when reasoning is off and temp is 0.0
        // (adaptive mode rejects any temperature != 1).
        let request = b71_request("claude-sonnet-4", Some(0.0), None);
        let body =
            build_anthropic_request_body(&request, request.max_tokens, "claude-sonnet-4-6", true);
        assert!(body.get("thinking").is_none());
        assert!(
            body.get("temperature").is_none(),
            "B81: thinking model must omit explicit temperature; got {:?}",
            body.get("temperature")
        );
    }

    #[test]
    fn d_inv_thinking_temp_anthropic_thinking_model_never_serializes_temperature() {
        // D-INV-THINKING-TEMP (B71+B81): a thinking-capable model on the
        // native anthropic path must NEVER receive a `temperature` key,
        // for ANY requested value and ANY reasoning state.
        let temps = [0.0_f32, 0.5, 1.0, -0.1, 1.5, f32::NAN];
        let reasonings: [Option<&str>; 3] = [None, Some("off"), Some("medium")];
        for temp in temps {
            for reasoning in reasonings {
                let request = b71_request("claude-sonnet-4", Some(temp), reasoning);
                let body = build_anthropic_request_body(
                    &request,
                    request.max_tokens,
                    "claude-sonnet-4-6",
                    true,
                );
                assert!(
                    body.get("temperature").is_none(),
                    "thinking model must omit temperature for temp={temp:?}, \
                     reasoning={reasoning:?}; got {:?}",
                    body.get("temperature")
                );
            }
        }
    }

    #[test]
    fn d_inv_thinking_temp_anthropic_claude_name_alone_blocks_temperature() {
        // D-INV-THINKING-TEMP: even when capabilities do NOT declare thinking
        // support, an upstream `claude-*` model name is treated as
        // thinking-capable and must still suppress the explicit temperature.
        let request = b71_request("claude-sonnet-4", Some(0.5), None);
        let body =
            build_anthropic_request_body(&request, request.max_tokens, "claude-sonnet-4-6", false);
        assert!(
            body.get("temperature").is_none(),
            "claude-* upstream name alone must block temperature; got {:?}",
            body.get("temperature")
        );
    }

    #[test]
    fn b71_anthropic_plain_model_allows_temperature() {
        let request = b71_request("plain-model", Some(0.7), None);
        let body = build_anthropic_request_body(&request, request.max_tokens, "plain-model", false);
        let temp = body["temperature"].as_f64().expect("temperature number");
        assert!((temp - 0.7).abs() < 1e-6, "temperature={temp}");
    }

    async fn collect_anthropic_sse_chunks(body: &str) -> Vec<StreamChunk> {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/sse"))
            .respond_with(
                ResponseTemplate::new(200)
                    .append_header("content-type", "text/event-stream")
                    .set_body_string(body),
            )
            .mount(&server)
            .await;

        let response = reqwest::Client::new()
            .get(format!("{}/sse", server.uri()))
            .send()
            .await
            .expect("mock SSE response must be reachable");

        sse_stream_from_response(response).collect().await
    }

    #[tokio::test]
    async fn sse_stream_errors_when_closed_without_message_stop() {
        let chunks = collect_anthropic_sse_chunks(
            "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\n\n",
        )
        .await;

        assert!(
            chunks
                .iter()
                .any(|chunk| matches!(chunk, StreamChunk::Text(t) if t == "hello"))
        );
        assert!(chunks.iter().any(|chunk| matches!(
            chunk,
            StreamChunk::Error(msg) if msg == "SSE stream closed without [DONE] or content"
        )));
        assert!(
            !chunks
                .iter()
                .any(|chunk| matches!(chunk, StreamChunk::Done))
        );
    }

    #[tokio::test]
    async fn sse_stream_message_stop_yields_done_without_error() {
        let chunks = collect_anthropic_sse_chunks(
            "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\n\ndata: {\"type\":\"message_stop\"}\n\n",
        )
        .await;

        assert!(
            chunks
                .iter()
                .any(|chunk| matches!(chunk, StreamChunk::Text(t) if t == "hello"))
        );
        assert!(
            chunks
                .iter()
                .any(|chunk| matches!(chunk, StreamChunk::Done))
        );
        assert!(
            !chunks
                .iter()
                .any(|chunk| matches!(chunk, StreamChunk::Error(_)))
        );
    }

    #[tokio::test]
    async fn sse_stream_done_marker_yields_done_without_error() {
        let chunks = collect_anthropic_sse_chunks(
            "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\n\ndata: [DONE]\n\n",
        )
        .await;

        assert!(
            chunks
                .iter()
                .any(|chunk| matches!(chunk, StreamChunk::Text(t) if t == "hello"))
        );
        assert!(
            chunks
                .iter()
                .any(|chunk| matches!(chunk, StreamChunk::Done))
        );
        assert!(
            !chunks
                .iter()
                .any(|chunk| matches!(chunk, StreamChunk::Error(_)))
        );
    }
}
