//! Error taxonomy — typed classification for retry/switch/alert decisions.
//!
//! Every error gets a category + severity. The agent loop and frontends
//! use these to decide: retry on Network, switch key on Auth, alert on Critical.

/// Error category for policy decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    /// Network unreachable, DNS, connection reset.
    Network,
    /// API key invalid, expired, payment required.
    Authentication,
    /// Rate limited (429).
    RateLimit,
    /// Request/response timeout.
    Timeout,
    /// Bad input to tool or API.
    InvalidInput,
    /// JSON parse failure.
    Parse,
    /// Tool execution error.
    Tool,
    /// Internal logic error.
    Internal,
}

/// Severity for UI and logging.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Classified error.
#[derive(Debug, Clone)]
pub struct ClassifiedError {
    pub category: ErrorCategory,
    pub severity: ErrorSeverity,
    pub message: String,
    pub retryable: bool,
}

/// Classify an error message into category + severity.
pub fn classify_error(error: &str) -> ClassifiedError {
    let lower = error.to_ascii_lowercase();

    if lower.contains("401")
        || lower.contains("auth")
        || lower.contains("invalid.*key")
        || lower.contains("payment required")
        || lower.contains("402")
    {
        return ClassifiedError {
            category: ErrorCategory::Authentication,
            severity: ErrorSeverity::Error,
            message: error.to_string(),
            retryable: false,
        };
    }

    if lower.contains("429") || lower.contains("rate limit") || lower.contains("too many") {
        return ClassifiedError {
            category: ErrorCategory::RateLimit,
            severity: ErrorSeverity::Warning,
            message: error.to_string(),
            retryable: true,
        };
    }

    if lower.contains("timeout") || lower.contains("timed out") {
        return ClassifiedError {
            category: ErrorCategory::Timeout,
            severity: ErrorSeverity::Warning,
            message: error.to_string(),
            retryable: true,
        };
    }

    if lower.contains("connection")
        || lower.contains("network")
        || lower.contains("dns")
        || lower.contains("unreachable")
    {
        return ClassifiedError {
            category: ErrorCategory::Network,
            severity: ErrorSeverity::Warning,
            message: error.to_string(),
            retryable: true,
        };
    }

    if lower.contains("json") || lower.contains("parse") || lower.contains("deserialize") {
        return ClassifiedError {
            category: ErrorCategory::Parse,
            severity: ErrorSeverity::Error,
            message: error.to_string(),
            retryable: false,
        };
    }

    ClassifiedError {
        category: ErrorCategory::Internal,
        severity: ErrorSeverity::Error,
        message: error.to_string(),
        retryable: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_auth() {
        let e = classify_error("HTTP 401 Unauthorized");
        assert_eq!(e.category, ErrorCategory::Authentication);
        assert!(!e.retryable);
    }

    #[test]
    fn classify_rate_limit() {
        let e = classify_error("HTTP 429 Too Many Requests");
        assert_eq!(e.category, ErrorCategory::RateLimit);
        assert!(e.retryable);
    }

    #[test]
    fn classify_timeout() {
        let e = classify_error("request timed out after 30s");
        assert_eq!(e.category, ErrorCategory::Timeout);
        assert!(e.retryable);
    }

    #[test]
    fn classify_network() {
        let e = classify_error("connection refused");
        assert_eq!(e.category, ErrorCategory::Network);
        assert!(e.retryable);
    }

    #[test]
    fn classify_parse() {
        let e = classify_error("failed to deserialize JSON response");
        assert_eq!(e.category, ErrorCategory::Parse);
        assert!(!e.retryable);
    }

    #[test]
    fn classify_payment() {
        let e = classify_error("payment required (402)");
        assert_eq!(e.category, ErrorCategory::Authentication);
    }

    #[test]
    fn classify_unknown() {
        let e = classify_error("something weird happened");
        assert_eq!(e.category, ErrorCategory::Internal);
    }
}
