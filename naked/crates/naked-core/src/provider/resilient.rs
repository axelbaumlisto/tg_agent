use std::collections::{HashMap, VecDeque};
use std::pin::Pin;
use std::time::Instant;

use async_trait::async_trait;
use tokio::sync::Mutex;
use tokio_stream::Stream;

use crate::error::{AgentError, Result};
use crate::types::{ModelInfo, StreamChunk};

use super::{ChatRequest, Provider};

/// How long a key stays blacklisted after a TRANSIENT terminal
/// error (rate-limit, transient 5xx). PERMANENT errors
/// (auth-failed / payment-required, see
/// [`ProviderError::is_permanent_key_failure`]) use a far-future
/// instant so the key is effectively removed from rotation for
/// the lifetime of the process.
const BLACKLIST_TTL: std::time::Duration = std::time::Duration::from_secs(3600);

/// Far-future cutoff used for permanently-blacklisted keys. ~100
/// years from epoch — well beyond any plausible process lifetime
/// but doesn't risk i64 overflow in arithmetic.
fn permanent_blacklist_until() -> Instant {
    Instant::now() + std::time::Duration::from_secs(60 * 60 * 24 * 365 * 100)
}

/// Wraps multiple providers with automatic failover.
///
/// Failed providers move to the back of the queue so that
/// healthy keys/providers stay at the front. The successful
/// provider stays at the head for subsequent calls.
///
/// Keys that return terminal errors (401/402/403) are blacklisted
/// for [`BLACKLIST_TTL`] and skipped entirely until the TTL expires.
pub struct ResilientProvider {
    providers: Vec<Box<dyn Provider>>,
    /// Mutable ordering of indices into `providers`.
    /// Front = preferred; failed indices are pushed to back.
    order: Mutex<VecDeque<usize>>,
    /// Indices blacklisted after terminal errors. Value = when to un-blacklist.
    blacklist: Mutex<HashMap<usize, Instant>>,
}

impl ResilientProvider {
    pub fn new(providers: Vec<Box<dyn Provider>>) -> Self {
        assert!(
            !providers.is_empty(),
            "ResilientProvider requires at least one provider"
        );
        let order: VecDeque<usize> = (0..providers.len()).collect();
        Self {
            providers,
            order: Mutex::new(order),
            blacklist: Mutex::new(HashMap::new()),
        }
    }

    pub fn provider_count(&self) -> usize {
        self.providers.len()
    }

    /// Current ordering snapshot (for testing / diagnostics).
    pub async fn current_order(&self) -> Vec<usize> {
        self.order.lock().await.iter().copied().collect()
    }

    /// Number of currently blacklisted keys.
    pub async fn blacklisted_count(&self) -> usize {
        let bl = self.blacklist.lock().await;
        bl.values().filter(|&&exp| Instant::now() < exp).count()
    }

    /// Provider health summary for diagnostics.
    pub async fn health_summary(&self) -> Vec<(String, bool)> {
        let bl = self.blacklist.lock().await;
        let now = Instant::now();
        self.providers
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let alive = bl.get(&i).map(|&exp| now >= exp).unwrap_or(true);
                (p.name().to_string(), alive)
            })
            .collect()
    }

    /// R2 of PLAN_RESILIENCE_v1: probe each provider key with a
    /// minimal request (5s timeout per key) so permanent failures
    /// (auth/payment) get marked BEFORE the first real turn.
    ///
    /// Cost: N tiny HTTP requests per boot, N parallel via
    /// `join_all`. Fire-and-forget from `wiring.rs` so boot time
    /// is unaffected (audit completes in the background; first
    /// few turns may still try a dead key, but subsequent ones
    /// skip it).
    /// Internal: explicit-model variant for tests + manual ops.
    pub async fn audit_keys_on_boot_with_model(&self, probe_model: &str) {
        use futures_util::future::join_all;
        let probe_req = ChatRequest {
            model: probe_model.to_string(),
            system: String::new(),
            messages: vec![serde_json::json!({
                "role": "user",
                "content": "hi"
            })],
            tools: Vec::new(),
            max_tokens: 1,
            temperature: None,
            reasoning: None,
        };
        let timeout = std::time::Duration::from_secs(5);
        let probes = self.providers.iter().enumerate().map(|(idx, p)| {
            let req = probe_req.clone();
            async move {
                match tokio::time::timeout(timeout, p.stream_chat(req)).await {
                    Ok(Ok(_stream)) => (idx, true, None),
                    Ok(Err(e)) => {
                        let permanent = matches!(
                            &e,
                            AgentError::ProviderTyped(pe)
                                if pe.is_permanent_key_failure()
                        );
                        (idx, !permanent, Some((permanent, e.to_string())))
                    }
                    Err(_elapsed) => (idx, true, None), // timeout → inconclusive, treat as alive
                }
            }
        });
        let outcomes = join_all(probes).await;
        let mut bl = self.blacklist.lock().await;
        let mut alive = 0usize;
        let mut dead = 0usize;
        for (idx, is_alive, err) in &outcomes {
            if *is_alive {
                alive += 1;
                continue;
            }
            dead += 1;
            if let Some((permanent, _msg)) = err
                && *permanent
            {
                bl.insert(*idx, permanent_blacklist_until());
                crate::types::PROVIDER_PERMANENT_BLACKLIST_COUNT
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        let provider_name = self
            .providers
            .first()
            .map(|p| p.name().to_string())
            .unwrap_or_else(|| "<unknown>".into());
        tracing::info!(
            provider = %provider_name,
            alive = alive,
            dead = dead,
            total = self.providers.len(),
            "provider audit: {}/{} keys alive",
            alive,
            self.providers.len()
        );
    }
}

#[async_trait]
impl Provider for ResilientProvider {
    fn name(&self) -> &str {
        // Synchronous access needed - use try_lock fallback
        if let Ok(order) = self.order.try_lock() {
            let idx = order.front().copied().unwrap_or(0);
            return self.providers[idx].name();
        }
        self.providers[0].name()
    }

    fn models(&self) -> Vec<ModelInfo> {
        self.providers.iter().flat_map(|p| p.models()).collect()
    }

    fn blacklisted_key_count(&self) -> usize {
        // 2026-05-13 fix: `blocking_lock` PANICS when called on a tokio
        // runtime worker thread ("Cannot block the current thread from
        // within a runtime"). Reproduced via `/health` from TG which
        // crashed the message-handler task on commit 097f308c.
        // Use try_lock() and return 0 on contention — the count is
        // diagnostic-only, never a correctness signal.
        // REGISTRY-WAIVE: B23 — diagnostic-only fail-open; on contention return 0 (no panic).
        let Ok(bl) = self.blacklist.try_lock() else {
            return 0;
        };
        let now = Instant::now();
        bl.values().filter(|&&exp| now < exp).count()
    }

    fn total_key_count(&self) -> usize {
        self.providers.len()
    }

    /// R2 wiring: recursive audit.
    ///   (a) Each inner provider's own audit_keys_on_boot runs
    ///       first (nested ResilientProvider audits its keys).
    ///   (b) Then outer-level probe: try each pseudo-provider with
    ///       a 1-token request so a totally-dead provider chain
    ///       gets marked at this level.
    async fn audit_keys_on_boot(&self) {
        use futures_util::future::join_all;
        // (a) Recurse into inner providers in parallel.
        let inner_audits = self.providers.iter().map(|p| p.audit_keys_on_boot());
        join_all(inner_audits).await;

        // (b) Outer-level: probe each pseudo-provider so a
        //     totally-dead provider also gets blacklisted here.
        let probe_model = self
            .providers
            .first()
            .and_then(|p| p.models().first().map(|m| m.model_id.clone()))
            .unwrap_or_default();
        if probe_model.is_empty() {
            tracing::debug!(
                provider = self
                    .providers
                    .first()
                    .map(|p| p.name())
                    .unwrap_or("<empty>"),
                "audit_keys_on_boot: no model configured, skipping outer probe"
            );
            return;
        }
        self.audit_keys_on_boot_with_model(&probe_model).await;
    }

    async fn stream_chat(
        &self,
        request: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = StreamChunk> + Send>>> {
        let snapshot: Vec<usize> = { self.order.lock().await.iter().copied().collect() };

        // Expire old blacklist entries
        {
            let mut bl = self.blacklist.lock().await;
            let now = Instant::now();
            bl.retain(|_, exp| now < *exp);
        }

        let mut failed_indices = Vec::new();
        let mut last_err = None;

        for (pos, &idx) in snapshot.iter().enumerate() {
            // Skip blacklisted keys
            if self.blacklist.lock().await.contains_key(&idx) {
                continue;
            }
            let provider = &self.providers[idx];

            match provider.stream_chat(request.clone()).await {
                Ok(stream) => {
                    if !failed_indices.is_empty() {
                        tracing::info!(
                            "Provider '{}' failed, fell back to '{}' ({} demoted)",
                            self.providers[snapshot[0]].name(),
                            provider.name(),
                            failed_indices.len(),
                        );
                        let mut order = self.order.lock().await;
                        if pos > 0
                            && let Some(p) = order.iter().position(|&i| i == idx)
                        {
                            order.remove(p);
                            order.push_front(idx);
                        }
                        for &fi in &failed_indices {
                            if let Some(p) = order.iter().position(|&i| i == fi) {
                                order.remove(p);
                                order.push_back(fi);
                            }
                        }
                    }
                    return Ok(stream);
                }
                Err(e) => {
                    // Try to extract typed error; fall back to string classification
                    let (key_dead, model_dead) = if let AgentError::ProviderTyped(ref pe) = e {
                        (pe.is_key_dead(), pe.is_model_dead())
                    } else {
                        let err_str = e.to_string();
                        let pe = super::error::ProviderError::from_llm_http(
                            extract_status_from_error(&err_str),
                            &err_str,
                            &request.model,
                        );
                        (pe.is_key_dead(), pe.is_model_dead())
                    };
                    if key_dead {
                        // R1 of PLAN_RESILIENCE_v1: distinguish
                        // PERMANENT (auth/payment) from TRANSIENT
                        // key failures. Permanent keys are removed
                        // from rotation for the process lifetime;
                        // transient keys use the existing 3600s
                        // blacklist + retry pattern.
                        let permanent = match &e {
                            AgentError::ProviderTyped(pe) => pe.is_permanent_key_failure(),
                            _ => false,
                        };
                        if permanent {
                            tracing::warn!(
                                "Provider '{}' key PERMANENTLY blacklisted (auth/payment): {e}",
                                provider.name(),
                            );
                            crate::types::PROVIDER_PERMANENT_BLACKLIST_COUNT
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            self.blacklist
                                .lock()
                                .await
                                .insert(idx, permanent_blacklist_until());
                            // B46 / PLAN_PROVIDER_HEALTH_v1: auto-persist
                            // dead key to state/naked.json so the next bot
                            // restart doesn't re-probe it. Gated by
                            // NAKED_AUTO_PERSIST_DEAD_KEYS=1 env (off by
                            // default — see dead_key_persist module docs).
                            if let Some(key_value) = self.providers[idx].key_hint() {
                                let logical = super::logical_provider_name(provider.name());
                                super::dead_key_persist::persist_dead_key(
                                    logical, &key_value,
                                );
                            }
                        } else {
                            tracing::warn!(
                                "Provider '{}' key dead: {e}, blacklisting for {}s",
                                provider.name(),
                                BLACKLIST_TTL.as_secs()
                            );
                            self.blacklist
                                .lock()
                                .await
                                .insert(idx, Instant::now() + BLACKLIST_TTL);
                        }
                    } else if model_dead {
                        // Model doesn't exist - no point trying other keys.
                        tracing::warn!(
                            "Provider '{}' model not found: {e}, aborting rotation",
                            provider.name(),
                        );
                        return Err(e);
                    } else {
                        tracing::warn!("Provider '{}' error: {e}, demoting", provider.name());
                    }
                    failed_indices.push(idx);
                    last_err = Some(e);
                }
            }
        }

        {
            let mut order = self.order.lock().await;
            for &fi in &failed_indices {
                if let Some(p) = order.iter().position(|&i| i == fi) {
                    order.remove(p);
                    order.push_back(fi);
                }
            }
        }

        Err(last_err.unwrap_or_else(|| {
            crate::error::AgentError::ProviderTyped(crate::provider::error::ProviderError::Other {
                status: 0,
                body: "all provider keys exhausted or blacklisted".into(),
            })
        }))
    }
}

/// Extract HTTP status code from a provider error string.
/// Looks for patterns like "401", "402 Payment", "(429)", "HTTP 500".
fn extract_status_from_error(err: &str) -> u16 {
    // Try common patterns: "API 401:", "(402)", "HTTP 429", bare "500"
    for code in [401u16, 402, 403, 404, 429, 500, 502, 503] {
        let s = code.to_string();
        if err.contains(&s) {
            return code;
        }
    }
    0 // unknown
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::AgentError;

    struct SuccessProvider {
        name: String,
    }

    #[async_trait]
    impl Provider for SuccessProvider {
        fn name(&self) -> &str {
            &self.name
        }
        fn models(&self) -> Vec<ModelInfo> {
            vec![ModelInfo {
                provider: self.name.clone(),
                model_id: "test".into(),
                display_name: "Test".into(),
            }]
        }
        async fn stream_chat(
            &self,
            _request: ChatRequest,
        ) -> Result<Pin<Box<dyn Stream<Item = StreamChunk> + Send>>> {
            Ok(Box::pin(tokio_stream::iter(vec![
                StreamChunk::Text("ok".into()),
                StreamChunk::Done,
            ])))
        }
    }

    struct FailProvider {
        name: String,
    }

    #[async_trait]
    impl Provider for FailProvider {
        fn name(&self) -> &str {
            &self.name
        }
        fn models(&self) -> Vec<ModelInfo> {
            vec![]
        }
        async fn stream_chat(
            &self,
            _request: ChatRequest,
        ) -> Result<Pin<Box<dyn Stream<Item = StreamChunk> + Send>>> {
            Err(AgentError::ProviderTyped(
                crate::provider::error::ProviderError::Other {
                    status: 0,
                    body: format!("{} failed", self.name),
                },
            ))
        }
    }

    fn test_request() -> ChatRequest {
        ChatRequest {
            model: "test".into(),
            system: String::new(),
            messages: vec![],
            tools: vec![],
            max_tokens: 100,
            temperature: None,
            reasoning: None,
        }
    }

    #[tokio::test]
    async fn single_provider_success() {
        let p = ResilientProvider::new(vec![Box::new(SuccessProvider { name: "a".into() })]);
        let _stream = p.stream_chat(test_request()).await.unwrap();
    }

    #[tokio::test]
    async fn fallback_on_first_failure() {
        let p = ResilientProvider::new(vec![
            Box::new(FailProvider { name: "bad".into() }),
            Box::new(SuccessProvider {
                name: "good".into(),
            }),
        ]);
        let _stream = p.stream_chat(test_request()).await.unwrap();
        assert_eq!(p.name(), "good");
    }

    #[tokio::test]
    async fn all_fail_returns_last_error() {
        let p = ResilientProvider::new(vec![
            Box::new(FailProvider { name: "a".into() }),
            Box::new(FailProvider { name: "b".into() }),
        ]);
        let result = p.stream_chat(test_request()).await;
        let err = result.err().expect("should fail");
        assert!(err.to_string().contains("b failed"));
    }

    #[tokio::test]
    async fn remembers_successful_provider() {
        let p = ResilientProvider::new(vec![
            Box::new(FailProvider { name: "bad".into() }),
            Box::new(SuccessProvider {
                name: "good".into(),
            }),
        ]);
        let _s1 = p.stream_chat(test_request()).await.unwrap();
        assert_eq!(p.name(), "good");
        let _s2 = p.stream_chat(test_request()).await.unwrap();
        assert_eq!(p.name(), "good");
    }

    #[tokio::test]
    async fn models_aggregates_all() {
        let p = ResilientProvider::new(vec![
            Box::new(SuccessProvider { name: "a".into() }),
            Box::new(SuccessProvider { name: "b".into() }),
        ]);
        assert_eq!(p.models().len(), 2);
    }

    #[tokio::test]
    async fn provider_count() {
        let p = ResilientProvider::new(vec![Box::new(SuccessProvider { name: "x".into() })]);
        assert_eq!(p.provider_count(), 1);
    }

    #[tokio::test]
    async fn initial_order_is_sequential() {
        let p = ResilientProvider::new(vec![
            Box::new(SuccessProvider { name: "a".into() }),
            Box::new(SuccessProvider { name: "b".into() }),
            Box::new(SuccessProvider { name: "c".into() }),
        ]);
        assert_eq!(p.current_order().await, vec![0, 1, 2]);
    }

    #[tokio::test]
    async fn failed_provider_moves_to_back() {
        let p = ResilientProvider::new(vec![
            Box::new(FailProvider { name: "k0".into() }),
            Box::new(SuccessProvider { name: "k1".into() }),
            Box::new(SuccessProvider { name: "k2".into() }),
        ]);
        assert_eq!(p.current_order().await, vec![0, 1, 2]);

        let _s = p.stream_chat(test_request()).await.unwrap();
        assert_eq!(p.name(), "k1");
        assert_eq!(p.current_order().await, vec![1, 2, 0]);
    }

    #[tokio::test]
    async fn multiple_failures_all_move_to_back() {
        let p = ResilientProvider::new(vec![
            Box::new(FailProvider { name: "k0".into() }),
            Box::new(FailProvider { name: "k1".into() }),
            Box::new(SuccessProvider { name: "k2".into() }),
        ]);
        let _s = p.stream_chat(test_request()).await.unwrap();
        assert_eq!(p.name(), "k2");
        assert_eq!(p.current_order().await, vec![2, 0, 1]);
    }

    #[tokio::test]
    async fn all_fail_demotes_in_order() {
        let p = ResilientProvider::new(vec![
            Box::new(FailProvider { name: "k0".into() }),
            Box::new(FailProvider { name: "k1".into() }),
            Box::new(FailProvider { name: "k2".into() }),
        ]);
        let _ = p.stream_chat(test_request()).await;
        assert_eq!(p.current_order().await, vec![0, 1, 2]);
    }

    #[tokio::test]
    async fn successive_failures_accumulate_at_back() {
        let p = ResilientProvider::new(vec![
            Box::new(FailProvider { name: "k0".into() }),
            Box::new(SuccessProvider { name: "k1".into() }),
            Box::new(SuccessProvider { name: "k2".into() }),
            Box::new(SuccessProvider { name: "k3".into() }),
            Box::new(SuccessProvider { name: "k4".into() }),
        ]);
        let _s = p.stream_chat(test_request()).await.unwrap();
        assert_eq!(p.current_order().await, vec![1, 2, 3, 4, 0]);
        assert_eq!(p.name(), "k1");
    }

    #[tokio::test]
    async fn successful_first_doesnt_change_order() {
        let p = ResilientProvider::new(vec![
            Box::new(SuccessProvider { name: "k0".into() }),
            Box::new(SuccessProvider { name: "k1".into() }),
            Box::new(SuccessProvider { name: "k2".into() }),
        ]);
        let _s = p.stream_chat(test_request()).await.unwrap();
        assert_eq!(p.current_order().await, vec![0, 1, 2]);
    }

    // ── Blacklist tests ─────────────────────────────────────────────

    struct TerminalFailProvider {
        name: String,
    }

    #[async_trait]
    impl Provider for TerminalFailProvider {
        fn name(&self) -> &str {
            &self.name
        }
        fn models(&self) -> Vec<ModelInfo> {
            vec![]
        }
        async fn stream_chat(
            &self,
            _request: ChatRequest,
        ) -> Result<Pin<Box<dyn Stream<Item = StreamChunk> + Send>>> {
            Err(AgentError::ProviderTyped(
                crate::provider::error::ProviderError::PaymentRequired {
                    status: 402,
                    body: "OpenAI API 402 Payment Required: {\"error\":{\"message\":\"membership not active\"}}".into(),
                },
            ))
        }
    }

    #[tokio::test]
    async fn terminal_error_blacklists_key() {
        let p = ResilientProvider::new(vec![
            Box::new(TerminalFailProvider {
                name: "dead-key".into(),
            }),
            Box::new(SuccessProvider {
                name: "good-key".into(),
            }),
        ]);
        // First call: dead-key fails with 402, gets blacklisted, good-key succeeds
        let _s = p.stream_chat(test_request()).await.unwrap();
        assert_eq!(p.blacklisted_count().await, 1);
        // Second call: dead-key is skipped (blacklisted), goes straight to good-key
        let _s = p.stream_chat(test_request()).await.unwrap();
        assert_eq!(p.blacklisted_count().await, 1);
    }

    #[tokio::test]
    async fn blacklisted_count_is_zero_initially() {
        let p = ResilientProvider::new(vec![Box::new(SuccessProvider { name: "k0".into() })]);
        assert_eq!(p.blacklisted_count().await, 0);
    }

    #[tokio::test]
    async fn all_blacklisted_still_returns_error() {
        let p = ResilientProvider::new(vec![
            Box::new(TerminalFailProvider {
                name: "dead1".into(),
            }),
            Box::new(TerminalFailProvider {
                name: "dead2".into(),
            }),
        ]);
        let result = p.stream_chat(test_request()).await;
        assert!(result.is_err());
        assert_eq!(p.blacklisted_count().await, 2);
    }

    #[tokio::test]
    async fn health_summary_shows_alive_and_dead() {
        let p = ResilientProvider::new(vec![
            Box::new(TerminalFailProvider {
                name: "dead".into(),
            }),
            Box::new(SuccessProvider {
                name: "alive".into(),
            }),
        ]);
        let _s = p.stream_chat(test_request()).await.unwrap();
        let summary = p.health_summary().await;
        assert_eq!(summary.len(), 2);
        assert_eq!(summary[0], ("dead".to_string(), false));
        assert_eq!(summary[1], ("alive".to_string(), true));
    }

    struct ModelNotFoundProvider {
        name: String,
    }

    #[async_trait]
    impl Provider for ModelNotFoundProvider {
        fn name(&self) -> &str {
            &self.name
        }
        fn models(&self) -> Vec<ModelInfo> {
            vec![]
        }
        async fn stream_chat(
            &self,
            _request: ChatRequest,
        ) -> Result<Pin<Box<dyn Stream<Item = StreamChunk> + Send>>> {
            Err(AgentError::ProviderTyped(
                crate::provider::error::ProviderError::ModelNotFound {
                    model: "test".into(),
                    body: "OpenAI API 404 Not Found: model_not_found".into(),
                },
            ))
        }
    }

    #[tokio::test]
    async fn model_not_found_aborts_without_blacklisting_key() {
        // 404 = model problem, not key problem. Key stays alive for other models.
        let p = ResilientProvider::new(vec![
            Box::new(ModelNotFoundProvider {
                name: "key-0".into(),
            }),
            Box::new(SuccessProvider {
                name: "key-1".into(),
            }),
        ]);
        let result = p.stream_chat(test_request()).await;
        // Should fail immediately on first 404 without trying key-1
        assert!(result.is_err(), "should return error");
        assert_eq!(
            p.blacklisted_count().await,
            0,
            "key should NOT be blacklisted - only the model is bad"
        );
    }

    #[tokio::test]
    async fn model_not_found_doesnt_cycle_other_keys() {
        // 3 keys, first returns 404. Should NOT try key-1 or key-2.
        let p = ResilientProvider::new(vec![
            Box::new(ModelNotFoundProvider { name: "k0".into() }),
            Box::new(SuccessProvider { name: "k1".into() }),
            Box::new(SuccessProvider { name: "k2".into() }),
        ]);
        let result = p.stream_chat(test_request()).await;
        assert!(result.is_err(), "should fail - model doesn't exist");
        // k1 and k2 were never tried (no point, same model list)
        assert_eq!(p.blacklisted_count().await, 0);
    }

    #[test]
    fn extract_status_401() {
        assert_eq!(
            extract_status_from_error("auth failed 401 Unauthorized"),
            401
        );
    }

    #[test]
    fn extract_status_429() {
        assert_eq!(extract_status_from_error("rate limited (429)"), 429);
    }

    #[test]
    fn extract_status_500() {
        assert_eq!(extract_status_from_error("server error 500"), 500);
    }

    #[test]
    fn extract_status_unknown() {
        assert_eq!(extract_status_from_error("some random error"), 0);
    }

    #[test]
    fn extract_status_empty() {
        assert_eq!(extract_status_from_error(""), 0);
    }
}
