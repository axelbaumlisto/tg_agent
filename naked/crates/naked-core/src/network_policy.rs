//! Network policy — host allow/deny for web_fetch and web_search.
//!
//! Controls which hosts the agent can access. Deny-wins precedence.

use std::collections::HashSet;

/// Network access decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetDecision {
    Allow,
    Deny,
}

/// Host-based network policy.
#[derive(Debug, Clone, Default)]
pub struct NetworkPolicy {
    allow: HashSet<String>,
    deny: HashSet<String>,
}

impl NetworkPolicy {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create from allow/deny lists.
    pub fn from_lists(allow: Vec<String>, deny: Vec<String>) -> Self {
        Self {
            allow: allow.into_iter().map(|h| h.to_ascii_lowercase()).collect(),
            deny: deny.into_iter().map(|h| h.to_ascii_lowercase()).collect(),
        }
    }

    /// Check if a host is allowed. Deny wins over allow.
    pub fn check(&self, host: &str) -> NetDecision {
        let lower = host.to_ascii_lowercase();

        // Deny-wins:
        if self.deny.contains(&lower) {
            return NetDecision::Deny;
        }
        // Check parent domains:
        for denied in &self.deny {
            if lower.ends_with(denied) {
                return NetDecision::Deny;
            }
        }

        // If allow list is empty, default allow:
        if self.allow.is_empty() {
            return NetDecision::Allow;
        }

        // Check allow list:
        if self.allow.contains(&lower) {
            return NetDecision::Allow;
        }
        for allowed in &self.allow {
            if lower.ends_with(allowed) {
                return NetDecision::Allow;
            }
        }

        NetDecision::Deny // not in allow list
    }
}

/// Extract hostname from URL.
pub fn host_from_url(url: &str) -> Option<String> {
    let after_scheme = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))?;
    let host = after_scheme.split('/').next()?;
    let host = host.split(':').next()?; // strip port
    Some(host.to_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_policy_allows_all() {
        let p = NetworkPolicy::new();
        assert_eq!(p.check("example.com"), NetDecision::Allow);
    }

    #[test]
    fn deny_wins() {
        let p = NetworkPolicy::from_lists(vec!["example.com".into()], vec!["example.com".into()]);
        assert_eq!(p.check("example.com"), NetDecision::Deny);
    }

    #[test]
    fn allow_list_restricts() {
        let p = NetworkPolicy::from_lists(vec!["google.com".into()], vec![]);
        assert_eq!(p.check("google.com"), NetDecision::Allow);
        assert_eq!(p.check("evil.com"), NetDecision::Deny);
    }

    #[test]
    fn subdomain_matching() {
        let p = NetworkPolicy::from_lists(vec![], vec!["facebook.com".into()]);
        assert_eq!(p.check("www.facebook.com"), NetDecision::Deny);
        assert_eq!(p.check("m.facebook.com"), NetDecision::Deny);
        assert_eq!(p.check("google.com"), NetDecision::Allow);
    }

    #[test]
    fn host_from_url_works() {
        assert_eq!(
            host_from_url("https://example.com/path"),
            Some("example.com".into())
        );
        assert_eq!(
            host_from_url("http://api.test.com:8080/v1"),
            Some("api.test.com".into())
        );
        assert_eq!(host_from_url("not a url"), None);
    }
}
