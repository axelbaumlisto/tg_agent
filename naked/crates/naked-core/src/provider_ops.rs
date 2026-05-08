//! Provider/model listing. Pure provider queries, no session access.

#[allow(unused_imports)]
use super::AgentCore;
#[allow(unused_imports)]
use super::ProviderInfo;
#[allow(unused_imports)]
use super::types;

impl AgentCore {
    /// Delegates to ProviderService.
    pub fn list_models(&self) -> Vec<types::ModelInfo> {
        self.provider_svc.list_models()
    }

    /// Delegates to ProviderService.
    pub fn provider_models(&self, provider_name: &str) -> Vec<(String, String)> {
        self.provider_svc.provider_models(provider_name)
    }

    /// Delegates to ProviderService.
    pub fn list_providers(&self) -> Vec<ProviderInfo> {
        self.provider_svc.list_providers()
    }

    /// Switch the provider and model for a specific session.
    /// Writes config.json into the session directory.
    pub fn default_provider_model(&self) -> (String, String) {
        self.provider_svc.default_provider_model()
    }
}

#[cfg(test)]
mod tests {
    /// Compile-time boundary: provider_ops must not access session state.
    /// If this grep finds matches, the boundary is violated.
    #[test]
    fn provider_ops_no_session_access() {
        let src = include_str!("provider_ops.rs");
        for pattern in ["self.ss.", ".sessions", "session_sender", "session_mcp"] {
            let hits: Vec<_> = src
                .lines()
                .enumerate()
                .filter(|(_, l)| !l.trim_start().starts_with("//"))
                .filter(|(_, l)| !l.contains("pattern")) // skip test's own patterns
                .filter(|(_, l)| !l.contains("cfg(test)"))
                .filter(|(_, l)| l.contains(pattern))
                .collect();
            assert!(
                hits.is_empty(),
                "provider_ops.rs violates boundary: found '{pattern}' at lines {:?}",
                hits.iter().map(|(n, _)| n + 1).collect::<Vec<_>>()
            );
        }
    }
}
