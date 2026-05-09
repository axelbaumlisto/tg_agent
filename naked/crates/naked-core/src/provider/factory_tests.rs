//! Unit tests for provider::factory.
//!
//! These tests used to live in `lib.rs` next to the inline factory machinery.
//! Moved here together with the factory in T4 of PLAN_CORE_HARDENING_v2.

use super::{PROVIDER_FACTORIES, create_single_provider};
use crate::config;

#[test]
fn provider_factory_anthropic_registered() {
    assert!(
        PROVIDER_FACTORIES
            .iter()
            .any(|(name, _)| *name == "anthropic")
    );
}

#[test]
fn provider_factory_copilot_registered() {
    assert!(
        PROVIDER_FACTORIES
            .iter()
            .any(|(name, _)| *name == "copilot")
    );
}

#[test]
fn create_single_provider_anthropic() {
    let cfg = config::ProviderConfig {
        provider_type: "anthropic".into(),
        api_key: "test-key".into(),
        ..Default::default()
    };
    let p = create_single_provider("test", cfg);
    assert!(p.name().contains("test"));
}

#[test]
fn create_single_provider_unknown_falls_back_to_openai() {
    let cfg = config::ProviderConfig {
        provider_type: "unknown_provider".into(),
        api_key: "test-key".into(),
        ..Default::default()
    };
    let p = create_single_provider("test", cfg);
    // OpenAiCompatProvider is the fallback for unknown types.
    assert!(p.name().contains("test"));
}

#[test]
fn create_provider_single_key_no_resilient_wrapper() {
    use crate::config::ResolvedProvider;

    let resolved = ResolvedProvider {
        provider_type: "anthropic".into(),
        api_key: "key1".into(),
        all_keys: vec!["key1".into()],
        base_url: None,
        models: vec!["claude-test".into()],
        max_tokens: None,
        temperature: None,
        headers: Default::default(),
        model_aliases: Default::default(),
    };
    let p = super::create_provider("solo", resolved);
    // Single-key path returns a plain provider, not a ResilientProvider.
    assert_eq!(p.name(), "solo");
}
