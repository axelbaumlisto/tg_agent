//! Typed provider errors.
//!
//! Today most provider failures come back as plain strings from the
//! transport layer; the LLM (or the coordinator) has to regex "quota",
//! "blocked", "429" out of prose to decide what to do next. This module
//! gives those errors structure so callers can switch on them — a
//! `QuotaExhausted` on the primary search provider, for example, triggers
//! an automatic fallback to `web_search` instead of nagging the user.

/// Hint for what the caller should try next when a provider fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackHint {
    /// Fall back to Exa `web_search` (generic search).
    UseWebSearch,
    /// Fall back to Playwright (browser-based fetch).
    UsePlaywright,
    /// Nothing reasonable — surface to user.
    None,
}

/// Structured provider error. Transport-layer adapters (`reqwest`, etc.)
/// convert raw responses into one of these variants via [`ProviderError::from_http`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderError {
    /// API key's quota is used up (e.g. Exa 429 `quota_exceeded`).
    QuotaExhausted {
        provider: String,
        hint: FallbackHint,
        raw: String,
    },
    /// Origin responded with a Cloudflare challenge (403 + `cf-browser-verification`,
    /// or similar). Body is effectively empty / unusable from an HTTP client.
    CloudflareBlocked { url: String, hint: FallbackHint },
    /// Authentication failed (401/403) — key is dead, try next key.
    AuthFailed { status: u16, body: String },
    /// Payment required (402) — key quota exhausted, try next key.
    PaymentRequired { status: u16, body: String },
    /// Model not found (404 + "model_not_found") — abort, don’t cycle keys.
    ModelNotFound { model: String, body: String },
    /// Rate limited (429) — back off and retry.
    RateLimited {
        retry_after: Option<u64>,
        body: String,
    },
    /// Server error (5xx) — transient, retry with same key.
    ServerError { status: u16, body: String },
    /// Serialization failure (JSON encode/decode).
    Serialize { context: String, source: String },
    /// MCP protocol error.
    Mcp { context: String, source: String },
    /// Non-classified upstream error — keep the raw body so the caller can log it.
    Other { status: u16, body: String },
}

impl ProviderError {
    /// Parse an upstream HTTP response into a typed error. `status` is the
    /// HTTP status code; `body` is the response body (small; callers should
    /// cap it before calling if responses can be huge).
    /// Classify an LLM provider HTTP error by status + body.
    pub fn from_llm_http(status: u16, body: &str, model: &str) -> Self {
        let lower = body.to_ascii_lowercase();
        match status {
            401 | 403 => ProviderError::AuthFailed {
                status,
                body: truncate(body, 512),
            },
            402 => ProviderError::PaymentRequired {
                status,
                body: truncate(body, 512),
            },
            404 if lower.contains("model_not_found")
                || lower.contains("does not exist")
                || lower.contains("not found") =>
            {
                ProviderError::ModelNotFound {
                    model: model.to_string(),
                    body: truncate(body, 512),
                }
            }
            429 => {
                // Try to parse retry-after from body
                let retry_after = extract_retry_seconds(&lower);
                ProviderError::RateLimited {
                    retry_after,
                    body: truncate(body, 512),
                }
            }
            500..=599 => ProviderError::ServerError {
                status,
                body: truncate(body, 512),
            },
            _ => ProviderError::Other {
                status,
                body: truncate(body, 512),
            },
        }
    }

    /// Classify a web-fetch/search HTTP error (original method).
    pub fn from_http(status: u16, body: &str) -> Self {
        let lower = body.to_ascii_lowercase();
        // Exa-style quota exhausted
        if status == 429
            || lower.contains("quota_exceeded")
            || lower.contains("quota exceeded")
            || lower.contains("insufficient_quota")
        {
            return ProviderError::QuotaExhausted {
                provider: "exa".to_string(), // transport layer may refine
                hint: FallbackHint::UseWebSearch,
                raw: truncate(body, 512),
            };
        }
        ProviderError::Other {
            status,
            body: truncate(body, 512),
        }
    }

    /// Is this a key-level failure (try next key)?
    pub fn is_key_dead(&self) -> bool {
        matches!(
            self,
            ProviderError::AuthFailed { .. } | ProviderError::PaymentRequired { .. }
        )
    }

    /// Is this a model-level failure (abort, don’t cycle keys)?
    pub fn is_model_dead(&self) -> bool {
        matches!(self, ProviderError::ModelNotFound { .. })
    }

    /// Is this a transient failure (retry with backoff)?
    pub fn is_transient(&self) -> bool {
        matches!(
            self,
            ProviderError::RateLimited { .. } | ProviderError::ServerError { .. }
        )
    }

    pub fn hint(&self) -> FallbackHint {
        match self {
            ProviderError::QuotaExhausted { hint, .. } => *hint,
            ProviderError::CloudflareBlocked { hint, .. } => *hint,
            _ => FallbackHint::None,
        }
    }
}

fn extract_retry_seconds(lower: &str) -> Option<u64> {
    if let Some(pos) = lower.find("retry after") {
        let after = &lower[pos + 12..];
        let digits: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
        return digits.parse().ok();
    }
    None
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let mut end = max;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}…", &s[..end])
    }
}

impl std::fmt::Display for ProviderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProviderError::QuotaExhausted { provider, hint, .. } => {
                write!(f, "provider `{provider}` quota exhausted (hint: {hint:?})")
            }
            ProviderError::CloudflareBlocked { url, hint } => {
                write!(f, "cloudflare challenge at `{url}` (hint: {hint:?})")
            }
            ProviderError::AuthFailed { status, body } => {
                write!(f, "auth failed ({status}): {body}")
            }
            ProviderError::PaymentRequired { status, body } => {
                write!(f, "payment required ({status}): {body}")
            }
            ProviderError::ModelNotFound { model, body } => {
                write!(f, "model not found `{model}`: {body}")
            }
            ProviderError::RateLimited { retry_after, body } => {
                if let Some(secs) = retry_after {
                    write!(f, "rate limited (retry after {secs}s): {body}")
                } else {
                    write!(f, "rate limited: {body}")
                }
            }
            ProviderError::ServerError { status, body } => {
                write!(f, "server error ({status}): {body}")
            }
            ProviderError::Serialize { context, source } => {
                write!(f, "serialize {context}: {source}")
            }
            ProviderError::Mcp { context, source } => {
                write!(f, "MCP {context}: {source}")
            }
            ProviderError::Other { status, body } => {
                write!(f, "provider error ({status}): {body}")
            }
        }
    }
}

impl std::error::Error for ProviderError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quota_exceeded_429_body() {
        let err = ProviderError::from_http(429, r#"{"error":"quota_exceeded"}"#);
        match err {
            ProviderError::QuotaExhausted { hint, .. } => {
                assert_eq!(hint, FallbackHint::UseWebSearch);
            }
            other => panic!("expected QuotaExhausted, got {other:?}"),
        }
    }

    #[test]
    fn quota_exceeded_200_body_message() {
        let err = ProviderError::from_http(200, r#"{"error":"insufficient_quota"}"#);
        assert!(matches!(err, ProviderError::QuotaExhausted { .. }));
    }

    #[test]
    fn other_status_passthrough() {
        let err = ProviderError::from_http(500, "boom");
        assert!(matches!(err, ProviderError::Other { status: 500, .. }));
    }

    // ── LLM provider errors ───────────────────────────────────────────

    #[test]
    fn llm_401_is_auth_failed() {
        let err = ProviderError::from_llm_http(401, "Unauthorized", "gpt-4");
        assert!(err.is_key_dead());
        assert!(!err.is_model_dead());
        assert!(matches!(err, ProviderError::AuthFailed { status: 401, .. }));
    }

    #[test]
    fn llm_402_is_payment_required() {
        let err = ProviderError::from_llm_http(402, "membership not active", "gpt-4");
        assert!(err.is_key_dead());
        assert!(matches!(err, ProviderError::PaymentRequired { .. }));
    }

    #[test]
    fn llm_404_model_not_found() {
        let err = ProviderError::from_llm_http(404, "model_not_found", "gpt-5");
        assert!(err.is_model_dead());
        assert!(!err.is_key_dead());
        assert!(matches!(err, ProviderError::ModelNotFound { .. }));
    }

    #[test]
    fn llm_404_generic_is_not_model_dead() {
        let err = ProviderError::from_llm_http(404, "endpoint unavailable", "gpt-4");
        // "endpoint unavailable" doesn't contain model_not_found keywords
        // so this is NOT a model-dead error
        assert!(!err.is_model_dead());
    }

    #[test]
    fn llm_429_rate_limited() {
        let err = ProviderError::from_llm_http(429, "retry after 30 seconds", "gpt-4");
        assert!(err.is_transient());
        assert!(matches!(
            err,
            ProviderError::RateLimited {
                retry_after: Some(30),
                ..
            }
        ));
    }

    #[test]
    fn llm_500_server_error() {
        let err = ProviderError::from_llm_http(500, "internal server error", "gpt-4");
        assert!(err.is_transient());
        assert!(matches!(
            err,
            ProviderError::ServerError { status: 500, .. }
        ));
    }

    #[test]
    fn llm_200_other() {
        let err = ProviderError::from_llm_http(200, "unexpected", "gpt-4");
        assert!(!err.is_key_dead());
        assert!(!err.is_model_dead());
        assert!(!err.is_transient());
    }

    #[test]
    fn truncate_ascii_within_limit() {
        assert_eq!(truncate("hello", 10), "hello");
    }

    #[test]
    fn truncate_ascii_at_limit() {
        assert_eq!(truncate("abcdef", 3), "abc…");
    }

    #[test]
    fn truncate_russian_on_char_boundary() {
        // "Ошибка" = 6 chars, 12 bytes. Truncate at 5 bytes
        // must not land inside a 2-byte char.
        let s = "Ошибка сервера";
        let result = truncate(s, 5);
        // 5 bytes → 2 full Cyrillic chars (4 bytes) + ellipsis
        assert_eq!(result, "Ош…");
    }

    #[test]
    fn truncate_emdash_boundary() {
        // em-dash — is 3 bytes. Truncate at 4 should give 3 bytes + ellipsis.
        let s = "———"; // 9 bytes
        let result = truncate(s, 4);
        assert_eq!(result, "—…"); // 3 bytes + ellipsis
    }
}
