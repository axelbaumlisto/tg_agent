//! Compile-time boundary verification tests.
//!
//! These tests verify that subsystem boundaries hold by checking
//! that types from one subsystem can't leak into another.

use super::*;

/// ProviderResolver trait is accessible without importing provider internals.
#[test]
fn provider_resolver_is_abstract() {
    fn _assert_trait<T: ProviderResolver>() {}
    _assert_trait::<crate::AgentCore>();
}

/// SessionLifecycle doesn't require provider knowledge.
#[test]
fn session_lifecycle_is_abstract() {
    fn _assert_trait<T: SessionLifecycle>() {}
    _assert_trait::<crate::AgentCore>();
}

/// SessionControl doesn't require provider knowledge.
#[test]
fn session_control_is_abstract() {
    fn _assert_trait<T: SessionControl>() {}
    _assert_trait::<crate::AgentCore>();
}

/// SessionDiagnostics doesn't require provider knowledge.
#[test]
fn session_diagnostics_is_abstract() {
    fn _assert_trait<T: SessionDiagnostics>() {}
    _assert_trait::<crate::AgentCore>();
}

/// EventSink is transport-agnostic.
#[test]
fn event_sink_is_abstract() {
    fn _assert_send_sync<T: EventSink + Send + Sync>() {}
    _assert_send_sync::<ChannelEventSink>();
}

/// ToolBuilder doesn't leak tool implementations.
#[test]
fn tool_builder_is_abstract() {
    fn _assert_trait<T: ToolBuilder>() {}
    _assert_trait::<crate::AgentCore>();
}

/// ResearchStore is accessible as trait object without knowing FsResearchStore.
#[test]
fn research_store_is_abstract() {
    fn _assert_object_safe(_: &dyn crate::research::ResearchStore) {}
}

/// Provider trait is object-safe and Send+Sync.
#[test]
fn provider_is_abstract_and_threadsafe() {
    fn _assert<T: crate::provider::Provider + Send + Sync + ?Sized>() {}
    _assert::<dyn crate::provider::Provider>();
}

/// SearchEngine trait is object-safe.
#[test]
fn search_engine_is_abstract() {
    fn _assert_object_safe(_: &dyn crate::search::SearchEngine) {}
}

/// CloudScraper trait is object-safe.
#[test]
fn cloud_scraper_is_abstract() {
    fn _assert_object_safe(_: &dyn crate::scrape::CloudScraper) {}
}

/// AgentRunner (research) is object-safe.
#[test]
fn agent_runner_is_abstract() {
    fn _assert_object_safe(_: &dyn crate::research::AgentRunner) {}
}

/// ResearchRunner trait decouples tools from AgentCore.
#[test]
fn research_runner_is_abstract() {
    fn _assert_object_safe(_: &dyn crate::research::ResearchRunner) {}
    fn _assert_agentcore_impls<T: crate::research::ResearchRunner>() {}
    _assert_agentcore_impls::<crate::AgentCore>();
}
