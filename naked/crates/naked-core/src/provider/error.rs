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
    /// Non-classified upstream error — keep the raw body so the caller can log it.
    Other { status: u16, body: String },
}

impl ProviderError {
    /// Parse an upstream HTTP response into a typed error. `status` is the
    /// HTTP status code; `body` is the response body (small; callers should
    /// cap it before calling if responses can be huge).
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

    pub fn hint(&self) -> FallbackHint {
        match self {
            ProviderError::QuotaExhausted { hint, .. } => *hint,
            ProviderError::CloudflareBlocked { hint, .. } => *hint,
            ProviderError::Other { .. } => FallbackHint::None,
        }
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max])
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
            ProviderError::Other { status, body } => {
                write!(f, "provider error {status}: {body}")
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
}
