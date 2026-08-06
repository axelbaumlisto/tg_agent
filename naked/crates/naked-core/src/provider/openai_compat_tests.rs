fn build_openai_messages(request: &ChatRequest) -> Vec<serde_json::Value> {
    build_openai_messages_inner(request, false)
}

use super::*;

#[test]
fn build_messages_with_system() {
    let request = ChatRequest {
        model: "gpt-4o".into(),
        system: "You are helpful".into(),
        messages: vec![serde_json::json!({
            "role": "user",
            "content": [{"type": "text", "text": "hi"}]
        })],
        tools: vec![],
        max_tokens: 1024,
        temperature: None,
        reasoning: None,
    };
    let msgs = build_openai_messages(&request);
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[0]["role"], "system");
    assert_eq!(msgs[1]["role"], "user");
    assert_eq!(msgs[1]["content"], "hi");
}

#[test]
fn build_messages_tool_use_converted() {
    let request = ChatRequest {
        model: "gpt-4o".into(),
        system: String::new(),
        messages: vec![
            serde_json::json!({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "analyzing"},
                    {"type": "tool_use", "id": "call_1", "name": "bash", "input": {"command": "ls"}}
                ]
            }),
            serde_json::json!({
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_1", "content": "file.txt", "is_error": false}
                ]
            }),
        ],
        tools: vec![],
        max_tokens: 1024,
        temperature: None,
        reasoning: None,
    };
    let msgs = build_openai_messages(&request);
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[0]["role"], "assistant");
    assert!(msgs[0]["tool_calls"].is_array());
    assert_eq!(msgs[0]["tool_calls"][0]["function"]["name"], "bash");
}

#[test]
fn build_messages_image_converted_to_image_url() {
    let request = ChatRequest {
        model: "meta-llama/llama-4-scout-17b-16e-instruct".into(),
        system: String::new(),
        messages: vec![serde_json::json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "iVBORw0KGgoAAA"
                    }
                }
            ]
        })],
        tools: vec![],
        max_tokens: 256,
        temperature: None,
        reasoning: None,
    };
    let msgs = build_openai_messages(&request);
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0]["role"], "user");
    let parts = msgs[0]["content"]
        .as_array()
        .expect("multimodal content must be an array");
    assert_eq!(parts.len(), 2, "expected text + image_url parts");
    assert_eq!(parts[0]["type"], "text");
    assert_eq!(parts[0]["text"], "what is this?");
    assert_eq!(parts[1]["type"], "image_url");
    assert_eq!(
        parts[1]["image_url"]["url"], "data:image/png;base64,iVBORw0KGgoAAA",
        "image_url must be a base64 data URL"
    );
}

#[test]
fn build_messages_image_only_no_text() {
    let request = ChatRequest {
        model: "gpt-4o-mini".into(),
        system: String::new(),
        messages: vec![serde_json::json!({
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg", "data": "AAAA"
                }}
            ]
        })],
        tools: vec![],
        max_tokens: 64,
        temperature: None,
        reasoning: None,
    };
    let msgs = build_openai_messages(&request);
    assert_eq!(msgs.len(), 1);
    let parts = msgs[0]["content"].as_array().unwrap();
    assert_eq!(parts.len(), 1);
    assert_eq!(parts[0]["type"], "image_url");
    assert!(
        parts[0]["image_url"]["url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/jpeg;base64,")
    );
}

#[test]
fn build_messages_image_detail_hint_propagates_to_image_url() {
    // `detail_hint` is the cross-provider mirror of OpenAI's
    // `image_url.detail`; the conversion must surface it inside the
    // generated `image_url` object so the upstream API can honour it.
    let request = ChatRequest {
        model: "gpt-4o".into(),
        system: String::new(),
        messages: vec![serde_json::json!({
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": "AA"},
                    "detail_hint": "low",
                }
            ]
        })],
        tools: vec![],
        max_tokens: 64,
        reasoning: None,
        temperature: None,
    };
    let msgs = build_openai_messages(&request);
    assert_eq!(msgs[0]["content"][0]["image_url"]["detail"], "low");
}

#[test]
fn build_messages_image_detail_hint_invalid_dropped() {
    // Garbage `detail_hint` must NOT poison the request; we silently drop
    // it instead of forwarding "garbage" to the API.
    let request = ChatRequest {
        model: "gpt-4o".into(),
        system: String::new(),
        messages: vec![serde_json::json!({
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": "AA"},
                    "detail_hint": "ultra-high",
                }
            ]
        })],
        tools: vec![],
        max_tokens: 64,
        reasoning: None,
        temperature: None,
    };
    let msgs = build_openai_messages(&request);
    assert!(msgs[0]["content"][0]["image_url"].get("detail").is_none());
}

#[test]
fn build_messages_multi_image_preserves_order() {
    // Two-image album: the order users see in Telegram must match the
    // order the model receives. Regression guard for media-group flushing.
    let request = ChatRequest {
        model: "gpt-4o".into(),
        system: String::new(),
        messages: vec![serde_json::json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "compare these:"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAAFIRST"}},
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "AAAASECOND"}},
            ]
        })],
        tools: vec![],
        max_tokens: 64,
        reasoning: None,
        temperature: None,
    };
    let msgs = build_openai_messages(&request);
    let parts = msgs[0]["content"].as_array().expect("array");
    assert_eq!(parts.len(), 3, "text + 2 images");
    assert_eq!(
        parts[1]["image_url"]["url"],
        "data:image/png;base64,AAAAFIRST"
    );
    assert_eq!(
        parts[2]["image_url"]["url"],
        "data:image/jpeg;base64,AAAASECOND"
    );
}

#[test]
fn build_messages_image_skipped_for_non_user_role() {
    // Defensive: assistant images shouldn't happen, but if they do we
    // should fall through to the plain-text path, not crash.
    let request = ChatRequest {
        model: "gpt-4o".into(),
        system: String::new(),
        messages: vec![serde_json::json!({
            "role": "assistant",
            "content": [
                {"type": "text", "text": "see image"},
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": "AA"
                }}
            ]
        })],
        tools: vec![],
        max_tokens: 64,
        temperature: None,
        reasoning: None,
    };
    let msgs = build_openai_messages(&request);
    // Image is dropped because the multimodal branch only triggers for user.
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0]["role"], "assistant");
    assert_eq!(msgs[0]["content"], "see image");
}

#[test]
fn build_messages_tool_result_converted() {
    let request = ChatRequest {
        model: "gpt-4o".into(),
        system: String::new(),
        messages: vec![
            serde_json::json!({
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "call_1", "name": "bash", "input": {"command": "ls"}}
                ]
            }),
            serde_json::json!({
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_1", "content": "output data", "is_error": false}
                ]
            }),
        ],
        tools: vec![],
        max_tokens: 1024,
        reasoning: None,
        temperature: None,
    };
    let msgs = build_openai_messages(&request);
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[1]["role"], "tool");
    assert_eq!(msgs[1]["tool_call_id"], "call_1");
}

#[test]
fn provider_name_and_models() {
    let config = ProviderConfig {
        provider_type: "openai_compat".into(),
        api_key: "sk-test".into(),
        api_keys: Vec::new(),
        base_url: None,
        models: vec!["gpt-4o".into(), "gpt-4o-mini".into()],
        max_tokens: None,
        temperature: None,
        context_window: None,
        headers: Default::default(),
        supports_vision: None,
        model_aliases: Default::default(),
        capabilities: Default::default(),
    };
    let provider = OpenAiCompatProvider::new("my-openai".into(), config);
    assert_eq!(provider.name(), "my-openai");
    let models = provider.models();
    assert_eq!(models.len(), 2);
    assert_eq!(models[0].model_id, "gpt-4o");
}

#[test]
fn provider_models_include_aliases() {
    let mut aliases = std::collections::HashMap::new();
    aliases.insert("kimi-k2.6".to_string(), "kimi-for-coding".to_string());
    aliases.insert("k2.6".to_string(), "kimi-for-coding".to_string());
    let config = ProviderConfig {
        provider_type: "openai_compat".into(),
        api_key: "sk-test".into(),
        api_keys: Vec::new(),
        base_url: Some("https://api.kimi.com/coding/v1".into()),
        models: vec!["kimi-for-coding".into()],
        max_tokens: None,
        temperature: None,
        context_window: None,
        headers: Default::default(),
        supports_vision: None,
        model_aliases: aliases,
        capabilities: Default::default(),
    };
    let provider = OpenAiCompatProvider::new("kimi-code".into(), config);
    let model_ids: Vec<String> = provider.models().into_iter().map(|m| m.model_id).collect();
    assert!(model_ids.contains(&"kimi-for-coding".to_string()));
    assert!(model_ids.contains(&"kimi-k2.6".to_string()));
    assert!(model_ids.contains(&"k2.6".to_string()));
    assert_eq!(model_ids.len(), 3, "real model + 2 aliases, no duplicates");
}

#[test]
fn resolve_model_alias_maps_to_upstream_id() {
    let mut aliases = std::collections::HashMap::new();
    aliases.insert("kimi-k2.6".to_string(), "kimi-for-coding".to_string());
    let config = ProviderConfig {
        provider_type: "openai_compat".into(),
        api_key: "sk-test".into(),
        api_keys: Vec::new(),
        base_url: None,
        models: vec!["kimi-for-coding".into()],
        max_tokens: None,
        temperature: None,
        context_window: None,
        headers: Default::default(),
        supports_vision: None,
        model_aliases: aliases,
        capabilities: Default::default(),
    };
    assert_eq!(config.resolve_model_alias("kimi-k2.6"), "kimi-for-coding");
    assert_eq!(
        config.resolve_model_alias("kimi-for-coding"),
        "kimi-for-coding"
    );
    assert_eq!(config.resolve_model_alias("unknown"), "unknown");
}

#[test]
fn reasoning_params_groq_excludes_enable_thinking() {
    // Groq's OpenAI-compatible API rejects unknown fields with HTTP 400.
    // We must NOT send enable_thinking / reasoning_effort to it.
    let mut body = serde_json::json!({});
    apply_reasoning_params(&mut body, "https://api.groq.com/openai/v1", Some("off"));
    assert!(
        body.get("enable_thinking").is_none(),
        "groq must not receive enable_thinking"
    );
    assert!(body.get("reasoning_effort").is_none());

    let mut body = serde_json::json!({});
    apply_reasoning_params(&mut body, "https://api.groq.com/openai/v1", Some("medium"));
    assert!(body.get("enable_thinking").is_none());
    assert!(body.get("reasoning_effort").is_none());
}

#[test]
fn reasoning_params_dashscope_sends_enable_thinking() {
    let mut body = serde_json::json!({});
    apply_reasoning_params(
        &mut body,
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        Some("off"),
    );
    assert_eq!(body["enable_thinking"], serde_json::json!(false));
}

#[test]
fn reasoning_params_openai_sends_reasoning_effort() {
    let mut body = serde_json::json!({});
    apply_reasoning_params(&mut body, "https://api.openai.com/v1", Some("high"));
    assert_eq!(body["reasoning_effort"], serde_json::json!("high"));
    assert!(body.get("enable_thinking").is_none());
}

#[test]
fn reasoning_params_none_sends_nothing() {
    let mut body = serde_json::json!({});
    apply_reasoning_params(&mut body, "https://api.openai.com/v1", None);
    assert!(body.get("enable_thinking").is_none());
    assert!(body.get("reasoning_effort").is_none());
}

#[test]
fn reasoning_params_kimi_coding_sends_reasoning_effort() {
    // Kimi For Coding (api.kimi.com/coding/v1) accepts `reasoning_effort`
    // — confirmed against the official Roo Code integration guide and
    // a live probe (see commit message). Without this whitelist entry,
    // the field would be silently dropped and the model would return
    // shallow non-reasoning answers.
    let mut body = serde_json::json!({});
    apply_reasoning_params(&mut body, "https://api.kimi.com/coding/v1", Some("medium"));
    assert_eq!(body["reasoning_effort"], serde_json::json!("medium"));
    assert!(
        body.get("enable_thinking").is_none(),
        "kimi.com is OAI-compat — never send enable_thinking"
    );
}

#[test]
fn reasoning_params_moonshot_cn_sends_reasoning_effort() {
    // Moonshot CN endpoint accepts the same OAI param. Important for
    // kimi-k2.5 (where thinking can be toggled) and harmless on
    // kimi-k2-thinking (always reasons regardless).
    let mut body = serde_json::json!({});
    apply_reasoning_params(&mut body, "https://api.moonshot.cn/v1", Some("high"));
    assert_eq!(body["reasoning_effort"], serde_json::json!("high"));
    assert!(body.get("enable_thinking").is_none());
}

#[test]
fn provider_empty_models_list() {
    let config = ProviderConfig {
        provider_type: "openai_compat".into(),
        api_key: "sk-test".into(),
        api_keys: Vec::new(),
        base_url: None,
        models: Vec::new(),
        max_tokens: None,
        temperature: None,
        context_window: None,
        headers: Default::default(),
        supports_vision: None,
        model_aliases: Default::default(),
        capabilities: Default::default(),
    };
    let provider = OpenAiCompatProvider::new("test".into(), config);
    let models = provider.models();
    assert!(models.is_empty());
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
fn b71_openai_compat_airpx_reasoning_omits_temperature() {
    let request = b71_request("claude-sonnet-4", Some(0.7), Some("medium"));
    let body = build_openai_request_body(
        &request,
        request.max_tokens,
        build_openai_messages(&request),
        "claude-sonnet-4-6",
        "https://airpx.cc/v1",
        true,
    );
    assert_eq!(body["reasoning_effort"], serde_json::json!("medium"));
    assert!(body.get("temperature").is_none());
}

#[test]
fn b71_openai_compat_thinking_route_rejects_temperature_one_when_reasoning_off() {
    let request = b71_request("claude-sonnet-4", Some(1.0), Some("off"));
    let body = build_openai_request_body(
        &request,
        request.max_tokens,
        build_openai_messages(&request),
        "claude-sonnet-4-6",
        "https://airpx.cc/v1",
        true,
    );
    assert_eq!(body["reasoning_effort"], serde_json::json!("off"));
    assert!(body.get("temperature").is_none());
}

#[test]
fn b81_openai_compat_airpx_thinking_route_omits_temperature_zero_no_reasoning() {
    // B81 live repro: the research inner-loop sends reasoning=None +
    // temperature=Some(0.0) to the airpx fallback. airpx `claude-sonnet-4-6`
    // is adaptive-mode (thinking on by default), so ANY explicit temperature
    // != 1 is rejected with HTTP 400. The body must therefore omit it.
    let request = b71_request("claude-sonnet-4", Some(0.0), None);
    let body = build_openai_request_body(
        &request,
        request.max_tokens,
        build_openai_messages(&request),
        "claude-sonnet-4-6",
        "https://airpx.cc/v1",
        true,
    );
    assert!(
        body.get("temperature").is_none(),
        "B81: adaptive-mode thinking route must NOT receive an explicit \
         temperature (even 0.0 with reasoning off); got {:?}",
        body.get("temperature")
    );
}

#[test]
fn d_inv_thinking_temp_openai_compat_thinking_route_never_serializes_temperature() {
    // D-INV-THINKING-TEMP (B71+B81): on a thinking-class route the serialized
    // body must NEVER contain a `temperature` key, for ANY requested value
    // and ANY reasoning state.
    let temps = [0.0_f32, 0.5, 1.0, -0.1, 1.5, f32::NAN];
    let reasonings: [Option<&str>; 3] = [None, Some("off"), Some("medium")];
    for temp in temps {
        for reasoning in reasonings {
            let request = b71_request("claude-sonnet-4", Some(temp), reasoning);
            let body = build_openai_request_body(
                &request,
                request.max_tokens,
                build_openai_messages(&request),
                "claude-sonnet-4-6",
                "https://airpx.cc/v1",
                true,
            );
            assert!(
                body.get("temperature").is_none(),
                "thinking route must omit temperature for temp={temp:?}, \
                 reasoning={reasoning:?}; got {:?}",
                body.get("temperature")
            );
        }
    }
}

#[test]
fn d_inv_thinking_temp_openai_compat_airpx_route_flag_alone_blocks_temperature() {
    // D-INV-THINKING-TEMP: even when model capabilities do NOT declare
    // thinking support, the airpx route itself is a known adaptive-mode
    // thinking proxy (`route_is_known_thinking_proxy`) and must still
    // suppress the explicit temperature.
    let request = b71_request("claude-sonnet-4", Some(0.5), None);
    let body = build_openai_request_body(
        &request,
        request.max_tokens,
        build_openai_messages(&request),
        "claude-sonnet-4-6",
        "https://airpx.cc/v1",
        false,
    );
    assert!(
        body.get("temperature").is_none(),
        "airpx route flag alone must block temperature; got {:?}",
        body.get("temperature")
    );
}

#[test]
fn b71_openai_compat_plain_route_allows_temperature() {
    let request = b71_request("gpt-4o", Some(0.7), None);
    let body = build_openai_request_body(
        &request,
        request.max_tokens,
        build_openai_messages(&request),
        "gpt-4o",
        "https://plain.example/v1",
        false,
    );
    let temp = body["temperature"].as_f64().expect("temperature number");
    assert!((temp - 0.7).abs() < 1e-6, "temperature={temp}");
}

async fn collect_openai_sse_chunks(body: &str) -> Vec<StreamChunk> {
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
async fn sse_stream_errors_when_closed_without_done() {
    let chunks =
        collect_openai_sse_chunks("data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n")
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
async fn sse_stream_done_marker_yields_done_without_error() {
    let chunks = collect_openai_sse_chunks(
        "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\ndata: [DONE]\n\n",
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
