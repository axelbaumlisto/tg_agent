use std::path::PathBuf;
use std::pin::Pin;

use async_trait::async_trait;
use tokio_stream::Stream;

use crate::config::ProviderConfig;
use crate::error::{AgentError, Result};
use crate::types::{ModelInfo, StreamChunk};

use super::openai_compat::OpenAiCompatProvider;
use super::{ChatRequest, Provider};

const CLIENT_ID: &str = "Ov23li8tweQw6odWQebz";
const DEVICE_CODE_URL: &str = "https://github.com/login/device/code";
const ACCESS_TOKEN_URL: &str = "https://github.com/login/oauth/access_token";
const COPILOT_API_BASE: &str = "https://api.githubcopilot.com";
const POLL_SAFETY_MARGIN_MS: u64 = 3000;

// ── Token persistence ───────────────────────────────────────────────────────

fn auth_file_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".naked").join("auth.json")
}

pub fn load_copilot_token() -> Option<String> {
    let path = auth_file_path();
    let data = std::fs::read_to_string(&path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&data).ok()?;
    json["github-copilot"]["access_token"]
        .as_str()
        .map(|s| s.to_string())
}

pub fn save_copilot_token(token: &str) -> Result<()> {
    let path = auth_file_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| AgentError::Provider(format!("cannot create ~/.naked: {e}")))?;
    }

    let mut json = if let Ok(data) = std::fs::read_to_string(&path) {
        serde_json::from_str::<serde_json::Value>(&data).unwrap_or(serde_json::json!({}))
    } else {
        serde_json::json!({})
    };

    json["github-copilot"] = serde_json::json!({
        "access_token": token,
        "created_at": chrono::Utc::now().timestamp(),
    });

    let contents = serde_json::to_string_pretty(&json)
        .map_err(|e| AgentError::Provider(format!("json serialize: {e}")))?;
    std::fs::write(&path, contents)
        .map_err(|e| AgentError::Provider(format!("write auth.json: {e}")))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
    }

    Ok(())
}

// ── OAuth device flow (RFC 8628) ────────────────────────────────────────────

#[derive(serde::Deserialize)]
struct DeviceCodeResponse {
    device_code: String,
    user_code: String,
    verification_uri: String,
    interval: u64,
}

#[derive(serde::Deserialize)]
struct TokenResponse {
    access_token: Option<String>,
    error: Option<String>,
    interval: Option<u64>,
}

pub async fn copilot_device_login() -> Result<String> {
    let client = reqwest::Client::new();

    let resp = client
        .post(DEVICE_CODE_URL)
        .header("Accept", "application/json")
        .header("Content-Type", "application/json")
        .header("User-Agent", "naked-agent/0.1")
        .json(&serde_json::json!({
            "client_id": CLIENT_ID,
            "scope": "read:user",
        }))
        .send()
        .await
        .map_err(|e| AgentError::Provider(format!("device code request: {e}")))?;

    if !resp.status().is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(AgentError::Provider(format!("device code failed: {text}")));
    }

    let device: DeviceCodeResponse = resp
        .json()
        .await
        .map_err(|e| AgentError::Provider(format!("device code parse: {e}")))?;

    eprintln!();
    eprintln!("  ┌─────────────────────────────────────────┐");
    eprintln!("  │  GitHub Copilot Login                    │");
    eprintln!("  │                                         │");
    eprintln!("  │  Open: {}  │", device.verification_uri);
    eprintln!(
        "  │  Code: \x1b[1m{}\x1b[0m                          │",
        device.user_code
    );
    eprintln!("  └─────────────────────────────────────────┘");
    eprintln!();
    eprintln!("Waiting for authorization...");

    let mut interval_ms = device.interval * 1000 + POLL_SAFETY_MARGIN_MS;

    loop {
        tokio::time::sleep(std::time::Duration::from_millis(interval_ms)).await;

        let resp = client
            .post(ACCESS_TOKEN_URL)
            .header("Accept", "application/json")
            .header("Content-Type", "application/json")
            .header("User-Agent", "naked-agent/0.1")
            .json(&serde_json::json!({
                "client_id": CLIENT_ID,
                "device_code": device.device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            }))
            .send()
            .await
            .map_err(|e| AgentError::Provider(format!("token poll: {e}")))?;

        if !resp.status().is_success() {
            return Err(AgentError::Provider("token exchange failed".into()));
        }

        let token_resp: TokenResponse = resp
            .json()
            .await
            .map_err(|e| AgentError::Provider(format!("token parse: {e}")))?;

        if let Some(token) = token_resp.access_token {
            save_copilot_token(&token)?;
            eprintln!("  \x1b[32m✓ Logged in to GitHub Copilot\x1b[0m");
            return Ok(token);
        }

        match token_resp.error.as_deref() {
            Some("authorization_pending") => continue,
            Some("slow_down") => {
                if let Some(new_interval) = token_resp.interval {
                    interval_ms = new_interval * 1000 + POLL_SAFETY_MARGIN_MS;
                } else {
                    interval_ms += 5000;
                }
                continue;
            }
            Some(err) => {
                return Err(AgentError::Provider(format!("OAuth error: {err}")));
            }
            None => continue,
        }
    }
}

// ── Copilot model discovery ─────────────────────────────────────────────────

pub async fn fetch_copilot_models(token: &str) -> Result<Vec<String>> {
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{COPILOT_API_BASE}/models"))
        .header("Authorization", format!("Bearer {token}"))
        .header("User-Agent", "naked-agent/0.1")
        .send()
        .await
        .map_err(|e| AgentError::Provider(format!("models fetch: {e}")))?;

    if !resp.status().is_success() {
        return Err(AgentError::Provider(format!(
            "models fetch HTTP {}",
            resp.status()
        )));
    }

    let body: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| AgentError::Provider(format!("models parse: {e}")))?;

    let models = body["data"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter(|m| m["model_picker_enabled"].as_bool().unwrap_or(false))
                .filter_map(|m| m["id"].as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    Ok(models)
}

// ── CopilotProvider ─────────────────────────────────────────────────────────

pub struct CopilotProvider {
    inner: OpenAiCompatProvider,
}

impl CopilotProvider {
    pub fn new(name: String, mut config: ProviderConfig) -> Self {
        let token = if config.api_key.is_empty() || config.api_key == "auto" {
            load_copilot_token().unwrap_or_default()
        } else {
            config.api_key.clone()
        };

        config.api_key = token;
        config.base_url = Some(COPILOT_API_BASE.to_string());
        config.headers.insert(
            "Openai-Intent".to_string(),
            "conversation-edits".to_string(),
        );
        config
            .headers
            .insert("x-initiator".to_string(), "user".to_string());
        config
            .headers
            .insert("User-Agent".to_string(), "naked-agent/0.1".to_string());

        let inner = OpenAiCompatProvider::new(name, config);
        Self { inner }
    }

    /// Build a CopilotProvider with interactive login if no token is stored.
    pub async fn new_with_login(name: String, mut config: ProviderConfig) -> Result<Self> {
        let token = if config.api_key.is_empty() || config.api_key == "auto" {
            match load_copilot_token() {
                Some(t) => t,
                None => copilot_device_login().await?,
            }
        } else {
            config.api_key.clone()
        };

        config.api_key = token;
        config.base_url = Some(COPILOT_API_BASE.to_string());
        config.headers.insert(
            "Openai-Intent".to_string(),
            "conversation-edits".to_string(),
        );
        config
            .headers
            .insert("x-initiator".to_string(), "user".to_string());
        config
            .headers
            .insert("User-Agent".to_string(), "naked-agent/0.1".to_string());

        let inner = OpenAiCompatProvider::new(name, config);
        Ok(Self { inner })
    }
}

#[async_trait]
impl Provider for CopilotProvider {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn models(&self) -> Vec<ModelInfo> {
        self.inner.models()
    }

    async fn stream_chat(
        &self,
        request: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = StreamChunk> + Send>>> {
        self.inner.stream_chat(request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auth_file_in_home() {
        let path = auth_file_path();
        assert!(path.to_string_lossy().contains(".naked"));
        assert!(path.to_string_lossy().ends_with("auth.json"));
    }

    #[test]
    fn copilot_provider_sets_headers() {
        let config = ProviderConfig {
            provider_type: "copilot".into(),
            api_key: "test-copilot-token".into(),
            api_keys: Vec::new(),
            base_url: None,
            models: vec!["gpt-4.1".into()],
            max_tokens: None,
            temperature: None,
            context_window: None,
            headers: Default::default(),
            supports_vision: None,
            model_aliases: Default::default(),
            capabilities: Default::default(),
        };
        let provider = CopilotProvider::new("copilot".into(), config);
        assert_eq!(provider.name(), "copilot");
        let models = provider.models();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].model_id, "gpt-4.1");
    }
}
