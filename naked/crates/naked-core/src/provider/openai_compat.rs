use std::pin::Pin;

use async_trait::async_trait;
use futures_util::StreamExt;
use tokio_stream::Stream;

use crate::config::ProviderConfig;
use crate::error::{AgentError, Result};
use crate::types::{ModelInfo, StreamChunk, TurnUsage};

use super::{ChatRequest, Provider};

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_MAX_TOKENS: u32 = 16384;

pub struct OpenAiCompatProvider {
    display_name: String,
    config: ProviderConfig,
    client: reqwest::Client,
    base_url: String,
}

impl OpenAiCompatProvider {
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
impl Provider for OpenAiCompatProvider {
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

        let needs_reasoning = self
            .config
            .base_url
            .as_deref()
            .unwrap_or("")
            .contains("deepseek");
        let messages = build_openai_messages_inner(&request, needs_reasoning);

        let upstream_model = self.config.resolve_model_alias(&request.model);

        let mut body = serde_json::json!({
            "model": upstream_model,
            "max_tokens": max_tokens,
            "stream": true,
            "stream_options": {"include_usage": true},
            "messages": messages,
        });

        apply_reasoning_params(&mut body, &self.base_url, request.reasoning.as_deref());

        if let Some(temp) = request.temperature {
            body["temperature"] = serde_json::json!(temp);
        }

        if !request.tools.is_empty() {
            let functions: Vec<serde_json::Value> = request
                .tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": t["name"],
                            "description": t["description"],
                            "parameters": t["input_schema"],
                        }
                    })
                })
                .collect();
            body["tools"] = serde_json::Value::Array(functions);
        }

        let url = format!("{}/chat/completions", self.base_url);

        let mut req = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json");

        for (k, v) in &self.config.headers {
            req = req.header(k.as_str(), v.as_str());
        }

        let response = req.json(&body).send().await.map_err(|e| {
            AgentError::ProviderTyped(super::error::classify_provider_body(0, &e.to_string()))
        })?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let text = response
                .text()
                .await
                .unwrap_or_else(|_| "no body".to_string());
            let typed = super::error::ProviderError::from_llm_http(status, &text, &request.model);
            return Err(typed.into());
        }

        let stream = sse_stream_from_response(response);
        Ok(Box::pin(stream))
    }
}

/// Apply `enable_thinking` / `reasoning_effort` to an OpenAI-compatible request body,
/// gated by the provider `base_url`. Some providers (e.g. Groq) reject unknown fields
/// with HTTP 400, so we never include them unless the host is known to accept them.
fn apply_reasoning_params(body: &mut serde_json::Value, base_url: &str, reasoning: Option<&str>) {
    let url_lc = base_url.to_lowercase();
    let supports_enable_thinking =
        url_lc.contains("dashscope") || url_lc.contains("aliyun") || url_lc.contains("aliyuncs");
    let supports_reasoning_effort = url_lc.contains("openai.com")
        || url_lc.contains("fireworks")
        || url_lc.contains("openrouter")
        || url_lc.contains("api.deepseek.com")
        // Kimi For Coding (api.kimi.com/coding/v1) supports `reasoning_effort`
        // per the official Roo Code integration guide (Enable Reasoning Effort:
        // Medium). The Moonshot global / CN endpoints also accept it on
        // kimi-k2.5 and kimi-k2-thinking; on kimi-k2-thinking the model
        // reasons regardless, but sending the param is harmless.
        || url_lc.contains("kimi.com")
        || url_lc.contains("moonshot.cn")
        || url_lc.contains("moonshot.ai")
        // 2026-05-13: airpx.cc OpenAI-compat proxy fronts claude-4-6,
        // deepseek-v4-*, gpt-5.x, gemini-3.* — все эти models thinking-
        // capable. Per their docs (airpx.cc/v1): supports `reasoning_effort`
        // (OpenAI style) and `thinking.budget_tokens` (Anthropic style).
        // Forwarding `reasoning_effort` covers OpenAI-compat path.
        || url_lc.contains("airpx.cc");
    match reasoning {
        Some("off") if supports_enable_thinking => {
            body["enable_thinking"] = serde_json::json!(false);
        }
        Some(effort) => {
            if supports_reasoning_effort {
                body["reasoning_effort"] = serde_json::json!(effort);
            }
            if supports_enable_thinking {
                body["enable_thinking"] = serde_json::json!(true);
            }
        }
        None => {}
    }
}

/// When `emit_reasoning` is true, every assistant message gets a
/// `reasoning_content` field (real thinking text or empty string).
/// Required by DeepSeek thinking models; rejected by Groq.
fn build_openai_messages_inner(
    request: &ChatRequest,
    emit_reasoning: bool,
) -> Vec<serde_json::Value> {
    let mut messages = Vec::new();

    if !request.system.is_empty() {
        messages.push(serde_json::json!({
            "role": "system",
            "content": request.system,
        }));
    }

    // Pre-scan: collect paired tool_call ↔ tool_result IDs to drop orphans after compaction.
    let mut call_ids = std::collections::HashSet::new();
    let mut result_ids = std::collections::HashSet::new();
    for msg in &request.messages {
        if let Some(content) = msg["content"].as_array() {
            for c in content {
                match c["type"].as_str() {
                    Some("tool_use") => {
                        if let Some(id) = c["id"].as_str() {
                            call_ids.insert(id.to_string());
                        }
                    }
                    Some("tool_result") => {
                        if let Some(id) = c["tool_use_id"].as_str() {
                            result_ids.insert(id.to_string());
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    let paired_ids: std::collections::HashSet<&String> =
        call_ids.intersection(&result_ids).collect();

    for msg in &request.messages {
        let role = msg["role"].as_str().unwrap_or("user");

        if let Some(content) = msg["content"].as_array() {
            let has_tool_result = content
                .iter()
                .any(|c| c["type"].as_str() == Some("tool_result"));

            if has_tool_result {
                for c in content {
                    if c["type"].as_str() == Some("tool_result") {
                        let tid = c["tool_use_id"].as_str().unwrap_or("");
                        if !tid.is_empty() && !paired_ids.contains(&tid.to_string()) {
                            continue; // orphaned tool result — skip
                        }
                        messages.push(serde_json::json!({
                            "role": "tool",
                            "tool_call_id": c["tool_use_id"],
                            "content": c["content"],
                        }));
                    }
                }
                continue;
            }

            let has_tool_use = content
                .iter()
                .any(|c| c["type"].as_str() == Some("tool_use"));

            if has_tool_use {
                let text_parts: Vec<String> = content
                    .iter()
                    .filter(|c| c["type"].as_str() == Some("text"))
                    .filter_map(|c| c["text"].as_str())
                    .map(|s| s.to_string())
                    .collect();

                let thinking_parts: Vec<String> = content
                    .iter()
                    .filter(|c| c["type"].as_str() == Some("thinking"))
                    .filter_map(|c| c["thinking"].as_str())
                    .map(|s| s.to_string())
                    .collect();

                // Only keep tool_calls whose results are also present.
                let tool_calls: Vec<serde_json::Value> = content
                    .iter()
                    .filter(|c| c["type"].as_str() == Some("tool_use"))
                    .filter(|c| {
                        c["id"]
                            .as_str()
                            .map(|id| paired_ids.contains(&id.to_string()))
                            .unwrap_or(false)
                    })
                    .map(|c| {
                        serde_json::json!({
                            "id": c["id"],
                            "type": "function",
                            "function": {
                                "name": c["name"],
                                "arguments": serde_json::to_string(&c["input"]).unwrap_or_default(),
                            }
                        })
                    })
                    .collect();

                if tool_calls.is_empty() {
                    // All tool_calls orphaned — emit as plain assistant text
                    let all_text = text_parts.join("");
                    if !all_text.is_empty() {
                        let mut m = serde_json::json!({
                            "role": "assistant",
                            "content": all_text,
                        });
                        if emit_reasoning {
                            m["reasoning_content"] =
                                serde_json::Value::String(thinking_parts.join(""));
                        }
                        messages.push(m);
                    }
                    continue;
                }

                let mut assistant_msg = serde_json::json!({
                    "role": "assistant",
                    "tool_calls": tool_calls,
                });
                if !text_parts.is_empty() {
                    assistant_msg["content"] = serde_json::Value::String(text_parts.join(""));
                }
                if emit_reasoning || !thinking_parts.is_empty() {
                    assistant_msg["reasoning_content"] =
                        serde_json::Value::String(thinking_parts.join(""));
                }
                messages.push(assistant_msg);
                continue;
            }

            // Multimodal user messages: at least one `image` block. Convert to
            // OpenAI-compat `content: [{type:"text",...},{type:"image_url",
            // image_url:{url:"data:<mime>;base64,<b64>"}}]`. Anthropic-style
            // `image.source.{type:base64,media_type,data}` is the canonical
            // form in our `to_api_messages` output.
            let has_image = content.iter().any(|c| c["type"].as_str() == Some("image"));
            if has_image && role == "user" {
                let mut parts: Vec<serde_json::Value> = Vec::new();
                for c in content {
                    match c["type"].as_str() {
                        Some("text") => {
                            if let Some(t) = c["text"].as_str()
                                && !t.is_empty()
                            {
                                parts.push(serde_json::json!({"type": "text", "text": t}));
                            }
                        }
                        Some("image") => {
                            let media_type =
                                c["source"]["media_type"].as_str().unwrap_or("image/png");
                            let data = c["source"]["data"].as_str().unwrap_or("");
                            // `detail_hint` is the cross-provider mirror of the
                            // OpenAI-only `image_url.detail` field. Set by
                            // `to_api_messages` when the original `ContentBlock::Image`
                            // carried a `detail` value; we forward only "low"|"high"|
                            // "auto" and silently drop anything else so a malformed
                            // hint never crashes the request.
                            let detail = c
                                .get("detail_hint")
                                .and_then(|v| v.as_str())
                                .filter(|s| matches!(*s, "low" | "high" | "auto"));
                            if !data.is_empty() {
                                let mut img = serde_json::json!({
                                    "url": format!("data:{media_type};base64,{data}"),
                                });
                                if let Some(d) = detail {
                                    img["detail"] = serde_json::Value::String(d.into());
                                }
                                parts.push(serde_json::json!({
                                    "type": "image_url",
                                    "image_url": img,
                                }));
                            }
                        }
                        _ => {}
                    }
                }
                if !parts.is_empty() {
                    messages.push(serde_json::json!({
                        "role": "user",
                        "content": parts,
                    }));
                }
                continue;
            }

            let text: String = content
                .iter()
                .filter(|c| c["type"].as_str() == Some("text"))
                .filter_map(|c| c["text"].as_str())
                .collect::<Vec<_>>()
                .join("");

            let thinking: String = content
                .iter()
                .filter(|c| c["type"].as_str() == Some("thinking"))
                .filter_map(|c| c["thinking"].as_str())
                .collect::<Vec<_>>()
                .join("");

            if !text.is_empty() || !thinking.is_empty() {
                let mut m = serde_json::json!({
                    "role": role,
                    "content": text,
                });
                if role == "assistant" && (emit_reasoning || !thinking.is_empty()) {
                    m["reasoning_content"] = serde_json::Value::String(thinking);
                }
                messages.push(m);
            }
        } else if let Some(text) = msg["content"].as_str() {
            let mut m = serde_json::json!({
                "role": role,
                "content": text,
            });
            if role == "assistant" && emit_reasoning {
                m["reasoning_content"] = serde_json::Value::String(String::new());
            }
            messages.push(m);
        }
    }

    messages
}

fn sse_stream_from_response(response: reqwest::Response) -> impl Stream<Item = StreamChunk> + Send {
    async_stream::stream! {
        let byte_stream = response.bytes_stream();
        let io_stream = byte_stream.map(|r| r.map_err(|e| std::io::Error::other(e.to_string())));
        let reader = tokio_util::io::StreamReader::new(io_stream);
        let mut lines = tokio::io::AsyncBufReadExt::lines(tokio::io::BufReader::new(reader));

        let mut tool_calls: std::collections::HashMap<u32, (String, String, String)> =
            std::collections::HashMap::new();
        let mut usage: Option<TurnUsage> = None;

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
                break;
            }

            let event: serde_json::Value = match serde_json::from_str(data) {
                Ok(v) => v,
                Err(_) => continue,
            };

            if let Some(u) = event.get("usage") {
                usage = Some(TurnUsage {
                    input_tokens: u["prompt_tokens"].as_u64().unwrap_or(0),
                    output_tokens: u["completion_tokens"].as_u64().unwrap_or(0),
                    cache_read_tokens: 0,
                    cache_write_tokens: 0,
                });
            }

            let choices = match event["choices"].as_array() {
                Some(c) => c,
                None => continue,
            };

            for choice in choices {
                let delta = &choice["delta"];

                if let Some(content) = delta["content"].as_str()
                    && !content.is_empty()
                {
                    yield StreamChunk::Text(content.to_string());
                }

                // reasoning_content — DeepSeek, OpenAI o-series, Fireworks reasoning models
                if let Some(reasoning) = delta["reasoning_content"]
                    .as_str()
                    .or_else(|| delta["reasoning"].as_str())
                    && !reasoning.is_empty()
                {
                    yield StreamChunk::Thinking(reasoning.to_string());
                }

                if let Some(tcs) = delta["tool_calls"].as_array() {
                    for tc in tcs {
                        let idx = tc["index"].as_u64().unwrap_or(0) as u32;
                        let entry = tool_calls.entry(idx).or_insert_with(|| {
                            let id = tc["id"].as_str().unwrap_or("").to_string();
                            let name = tc["function"]["name"]
                                .as_str()
                                .unwrap_or("")
                                .to_string();
                            (id, name, String::new())
                        });
                        if let Some(args_part) = tc["function"]["arguments"].as_str() {
                            entry.2.push_str(args_part);
                        }
                    }
                }

                if choice.get("finish_reason").is_some_and(|v| !v.is_null()) {
                    let mut indices: Vec<u32> = tool_calls.keys().copied().collect();
                    indices.sort();
                    for idx in indices {
                        if let Some((id, name, args_json)) = tool_calls.remove(&idx) {
                            let input: serde_json::Value =
                                crate::tool::arg_repair::repair_json(&args_json);
                            yield StreamChunk::ToolUse { id, name, input };
                        }
                    }
                }
            }
        }

        if let Some(u) = usage {
            yield StreamChunk::Usage(u);
        }
        yield StreamChunk::Done;
    }
}

#[cfg(test)]
#[path = "openai_compat_tests.rs"]
mod tests;
