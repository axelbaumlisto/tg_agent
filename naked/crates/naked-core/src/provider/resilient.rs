use std::collections::{HashMap, VecDeque};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Instant;

use async_trait::async_trait;
use futures_util::StreamExt as FuturesStreamExt;
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
const STREAM_FAILURE_PENDING: u8 = 0b1000_0000;
const STREAM_FAILURE_COUNT_MASK: u8 = 0b0111_1111;

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
    /// B67: per-index mid-stream failure tally, bumped by the returned
    /// stream's wrapper (sync closure) and folded into `blacklist`/`order`
    /// at the top of the next `stream_chat`. Uses `std::sync::Mutex`
    /// because stream combinator closures are synchronous and cannot await
    /// the existing tokio locks; the guard is never held across `.await`.
    stream_failures: Arc<StdMutex<HashMap<usize, u8>>>,
    /// B106: identity of the last silent fallback, so the turn layer can
    /// re-label its span/health/UI with who ACTUALLY answered. Written on
    /// every successful `stream_chat`: `Some(..)` when a fallback occurred,
    /// `None` when the requested provider served it, so a later clean turn
    /// cannot inherit a stale banner. `std::sync::Mutex` (never held across
    /// `.await`) so the sync `last_fallback()` accessor can read it.
    last_fallback: Arc<StdMutex<Option<super::FallbackInfo>>>,
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
            stream_failures: Arc::new(StdMutex::new(HashMap::new())),
            last_fallback: Arc::new(StdMutex::new(None)),
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

    async fn fold_stream_failures(&self) {
        // B67: apply any mid-stream failures recorded by previously-returned
        // streams. The sync mutex is held only long enough to drain the small
        // tally map; tokio locks are acquired afterwards, never while holding
        // the std mutex across `.await`.
        let pending: Vec<(usize, u8)> = {
            let mut failures = crate::lock_or_recover(&self.stream_failures);
            let pending: Vec<_> = failures
                .iter()
                .filter_map(|(&idx, &raw)| {
                    (raw & STREAM_FAILURE_PENDING != 0)
                        .then_some((idx, raw & STREAM_FAILURE_COUNT_MASK))
                })
                .collect();
            for (idx, _) in &pending {
                failures.remove(idx);
            }
            pending
        };
        if pending.is_empty() {
            return;
        }

        let mut carry = Vec::new();
        {
            let mut bl = self.blacklist.lock().await;
            let mut order = self.order.lock().await;
            let now = Instant::now();
            for (idx, count) in pending {
                if count >= 2 {
                    bl.insert(idx, now + BLACKLIST_TTL);
                    tracing::warn!(
                        "Provider '{}' blacklisted after {} mid-stream failures",
                        self.providers[idx].name(),
                        count
                    );
                } else if let Some(p) = order.iter().position(|&i| i == idx) {
                    order.remove(p);
                    order.push_back(idx);
                    carry.push((idx, count));
                }
            }
        }

        if !carry.is_empty() {
            let mut failures = crate::lock_or_recover(&self.stream_failures);
            for (idx, count) in carry {
                let raw = failures.get(&idx).copied().unwrap_or(0);
                let pending_bit = raw & STREAM_FAILURE_PENDING;
                let existing_count = raw & STREAM_FAILURE_COUNT_MASK;
                let merged_count = existing_count
                    .saturating_add(count)
                    .min(STREAM_FAILURE_COUNT_MASK);
                failures.insert(idx, pending_bit | merged_count);
            }
        }
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
        let provider_name = self
            .providers
            .first()
            .map(|p| p.name().to_string())
            .unwrap_or_else(|| "<unknown>".into());
        let mut bl = self.blacklist.lock().await;
        let mut alive = 0usize;
        let mut dead = 0usize;
        for (idx, is_alive, err) in &outcomes {
            if *is_alive {
                alive += 1;
                continue;
            }
            dead += 1;
            if let Some((permanent, msg)) = err
                && *permanent
            {
                bl.insert(*idx, permanent_blacklist_until());
                tracing::warn!(
                    provider = %provider_name,
                    key_index = *idx,
                    error = %msg,
                    "provider key permanently blacklisted during boot audit"
                );
                self.note_permanently_dead_key(*idx);
            }
        }
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
    /// Record that `providers[idx]`'s key is permanently dead: bump the
    /// counter and hand the literal key to B46 persistence.
    ///
    /// Both discovery paths (boot audit and runtime `stream_chat`) must do the
    /// SAME thing here. They did not: the runtime path persisted, the boot
    /// audit only blacklisted, so the path that finds most dead keys wrote
    /// nothing and B46 was a no-op (B111). Keeping the two in sync by hand is
    /// what failed — hence one method.
    ///
    /// Caller owns the blacklist insert, because the two paths hold that lock
    /// differently (the audit already holds the guard).
    ///
    /// B112: the provider name is derived from `idx` HERE rather than passed
    /// in. It selects which `providers.<X>` bucket the dead key is written to,
    /// and the two callers had different notions of it: the runtime path
    /// passed the entry that actually failed, while the boot audit passed
    /// `providers.first()` for EVERY dead index. On the outer chain — which is
    /// heterogeneous by construction (`create_provider_chain` appends each
    /// configured fallback provider) — that filed a dead deepseek key under
    /// anthropic: the wrong pool loses a key and the real dead one keeps being
    /// probed. Extracting the shared body was not enough; the drift simply
    /// moved to the call sites, so the parameter is gone.
    fn note_permanently_dead_key(&self, idx: usize) {
        crate::types::PROVIDER_PERMANENT_BLACKLIST_COUNT
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let provider = &self.providers[idx];
        // B110: this `key_hint` reaches through the TimeoutProvider wrapper
        // every chain entry is built with; without that delegation it is None
        // and nothing is ever persisted.
        if let Some(key_value) = provider.key_hint() {
            super::dead_key_persist::persist_dead_key(
                super::logical_provider_name(provider.name()),
                &key_value,
            );
        }
    }

    /// B106: remember who actually served the last `stream_chat`.
    ///
    /// Called on EVERY success — `clean = true` clears the slot, so a stale
    /// banner from an earlier fallback cannot leak into a later good turn.
    fn note_fallback(
        &self,
        clean: bool,
        requested: &str,
        served_by: &str,
        last_err: Option<&AgentError>,
    ) {
        // Poison-recovering lock helper used throughout the crate. Dropping
        // the write on poison would make every later turn look clean and
        // quietly re-hide the fallback B106 exists to surface.
        let mut slot = crate::lock_or_recover(&self.last_fallback);
        *slot = if clean {
            None
        } else {
            Some(super::FallbackInfo {
                requested: super::logical_provider_name(requested).to_string(),
                served_by: super::logical_provider_name(served_by).to_string(),
                reason: last_err
                    .map(|e| {
                        let red = crate::research::tool::redact::redact_for_log(e);
                        crate::util::head_truncate(&red, 160).to_string()
                    })
                    .unwrap_or_else(|| "provider error".to_string()),
            })
        };
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

    /// B106: expose the last silent fallback to the turn layer.
    fn last_fallback(&self) -> Option<super::FallbackInfo> {
        crate::lock_or_recover(&self.last_fallback).clone()
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
        self.fold_stream_failures().await;
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
                    self.note_fallback(
                        failed_indices.is_empty(),
                        self.providers[snapshot[0]].name(),
                        provider.name(),
                        last_err.as_ref(),
                    );
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
                    let failures = self.stream_failures.clone();
                    let saw_error = Arc::new(AtomicBool::new(false));
                    let saw_content = Arc::new(AtomicBool::new(false));
                    let wrapped = FuturesStreamExt::map(stream, move |chunk| {
                        match &chunk {
                            StreamChunk::Error(_) => {
                                saw_error.store(true, Ordering::Relaxed);
                                let mut failures = crate::lock_or_recover(&failures);
                                let entry = failures.entry(idx).or_insert(0);
                                let count = (*entry & STREAM_FAILURE_COUNT_MASK)
                                    .saturating_add(1)
                                    .min(STREAM_FAILURE_COUNT_MASK);
                                *entry = STREAM_FAILURE_PENDING | count;
                            }
                            StreamChunk::Text(_)
                            | StreamChunk::Thinking(_)
                            | StreamChunk::ToolUse { .. } => {
                                saw_content.store(true, Ordering::Relaxed);
                            }
                            StreamChunk::Done => {
                                if saw_content.load(Ordering::Relaxed)
                                    && !saw_error.load(Ordering::Relaxed)
                                {
                                    let mut failures = crate::lock_or_recover(&failures);
                                    if let Some(raw) = failures.get(&idx).copied()
                                        && raw & STREAM_FAILURE_PENDING == 0
                                    {
                                        failures.remove(&idx);
                                    }
                                }
                            }
                            StreamChunk::Usage(_) => {}
                        }
                        chunk
                    });
                    return Ok(Box::pin(wrapped));
                }
                Err(e) => {
                    // Try to extract typed error; fall back to string classification
                    // B112: classify ONCE. Previously the untyped branch
                    // reconstructed a ProviderError to derive key/model/invalid,
                    // but `permanent` below was read only off the ORIGINAL `e`
                    // and hard-defaulted to false for untyped errors. An untyped
                    // 401 therefore entered the key-dead branch yet was treated
                    // as transient: 1h blacklist instead of permanent, no
                    // persist, no counter — re-probed every hour and every
                    // restart. One value now answers all four questions.
                    let classified: std::borrow::Cow<'_, super::error::ProviderError> =
                        if let AgentError::ProviderTyped(ref pe) = e {
                            std::borrow::Cow::Borrowed(pe)
                        } else {
                            let err_str = e.to_string();
                            std::borrow::Cow::Owned(super::error::ProviderError::from_llm_http(
                                extract_status_from_error(&err_str),
                                &err_str,
                                &request.model,
                            ))
                        };
                    let key_dead = classified.is_key_dead();
                    let model_dead = classified.is_model_dead();
                    let invalid_request = matches!(
                        classified.as_ref(),
                        super::error::ProviderError::InvalidRequest { .. }
                    );
                    if key_dead {
                        // R1 of PLAN_RESILIENCE_v1: distinguish
                        // PERMANENT (auth/payment) from TRANSIENT
                        // key failures. Permanent keys are removed
                        // from rotation for the process lifetime;
                        // transient keys use the existing 3600s
                        // blacklist + retry pattern.
                        let permanent = classified.is_permanent_key_failure();
                        if permanent {
                            tracing::warn!(
                                "Provider '{}' key PERMANENTLY blacklisted (auth/payment): {e}",
                                provider.name(),
                            );
                            self.blacklist
                                .lock()
                                .await
                                .insert(idx, permanent_blacklist_until());
                            self.note_permanently_dead_key(idx);
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
                    } else if invalid_request {
                        crate::types::PROVIDER_INVALID_REQUEST_COUNT
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        tracing::warn!(
                            "Provider '{}' invalid request (likely config/request-shape bug): {e}, demoting without blacklist",
                            provider.name()
                        );
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
    // B112: match DELIMITED forms only. A bare `contains("402")` classified
    // bodies like "model llama-402b is unavailable" or "max_tokens 500
    // exceeded" as payment/server failures, blacklisting a healthy key for an
    // hour; and because the codes were scanned in a fixed order, a body
    // mentioning two numbers reported whichever came first in the array rather
    // than the real status. Unknown now stays 0 (treated as transient), which
    // is the safe default.
    for code in [401u16, 402, 403, 404, 429, 500, 502, 503] {
        let s = code.to_string();
        let mut from = 0usize;
        while let Some(rel) = err[from..].find(&s) {
            let start = from + rel;
            let end = start + s.len();
            let before = err[..start].chars().next_back();
            let after = err[end..].chars().next();
            // Reject digit-adjacent hits (llama-402b, 4029) and require the
            // number to sit next to punctuation/whitespace, as in
            // "API 401:", "(402)", "HTTP 429", "error code: 502".
            let left_ok = before.is_none_or(|c| !c.is_ascii_digit() && c != '.');
            let right_ok = after.is_none_or(|c| !c.is_ascii_alphanumeric() && c != '.');
            if left_ok && right_ok {
                return code;
            }
            from = end;
        }
    }
    0 // unknown
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::AgentError;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio_stream::StreamExt as TokioStreamExt;

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

    struct StreamErrorProvider {
        name: String,
    }

    #[async_trait]
    impl Provider for StreamErrorProvider {
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
            Ok(Box::pin(tokio_stream::iter(vec![StreamChunk::Error(
                format!("{} inter-chunk timeout", self.name),
            )])))
        }
    }

    struct FirstOkThenFailProvider {
        name: String,
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Provider for FirstOkThenFailProvider {
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
            if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                Ok(Box::pin(tokio_stream::iter(vec![StreamChunk::Error(
                    format!("{} inter-chunk timeout", self.name),
                )])))
            } else {
                Err(AgentError::ProviderTyped(
                    crate::provider::error::ProviderError::Other {
                        status: 0,
                        body: format!("{} failed", self.name),
                    },
                ))
            }
        }
    }

    #[derive(Clone, Copy)]
    enum ScriptedChunk {
        Error,
        ContentThenDone,
    }

    struct ScriptedStreamProvider {
        name: String,
        calls: Arc<AtomicUsize>,
        script: Vec<ScriptedChunk>,
    }

    #[async_trait]
    impl Provider for ScriptedStreamProvider {
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
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            let chunk = self
                .script
                .get(call)
                .copied()
                .unwrap_or(ScriptedChunk::ContentThenDone);
            let chunks = match chunk {
                ScriptedChunk::Error => vec![StreamChunk::Error(format!(
                    "{} inter-chunk timeout",
                    self.name
                ))],
                ScriptedChunk::ContentThenDone => {
                    vec![StreamChunk::Text("ok".into()), StreamChunk::Done]
                }
            };
            Ok(Box::pin(tokio_stream::iter(chunks)))
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
    async fn report_via_stream_demotes_failed_provider() {
        let p = ResilientProvider::new(vec![
            Box::new(StreamErrorProvider { name: "k0".into() }),
            Box::new(SuccessProvider { name: "k1".into() }),
        ]);

        let mut stream = p.stream_chat(test_request()).await.unwrap();
        assert!(matches!(
            TokioStreamExt::next(&mut stream).await,
            Some(StreamChunk::Error(_))
        ));

        let mut next = p.stream_chat(test_request()).await.unwrap();
        assert_eq!(p.current_order().await, vec![1, 0]);
        assert!(matches!(
            TokioStreamExt::next(&mut next).await,
            Some(StreamChunk::Text(t)) if t == "ok"
        ));
    }

    #[tokio::test]
    async fn two_stream_failures_blacklist_idx() {
        let p = ResilientProvider::new(vec![
            Box::new(StreamErrorProvider { name: "k0".into() }),
            Box::new(FailProvider { name: "k1".into() }),
        ]);

        let mut first = p.stream_chat(test_request()).await.unwrap();
        assert!(matches!(
            TokioStreamExt::next(&mut first).await,
            Some(StreamChunk::Error(_))
        ));

        let mut second = p.stream_chat(test_request()).await.unwrap();
        assert!(matches!(
            TokioStreamExt::next(&mut second).await,
            Some(StreamChunk::Error(_))
        ));

        let result = p.stream_chat(test_request()).await;
        assert!(result.is_err());
        assert_eq!(p.blacklisted_count().await, 1);
    }

    #[tokio::test]
    async fn concurrent_streams_only_failed_idx_demoted() {
        let calls = Arc::new(AtomicUsize::new(0));
        let p = ResilientProvider::new(vec![
            Box::new(FirstOkThenFailProvider {
                name: "k0".into(),
                calls: calls.clone(),
            }),
            Box::new(SuccessProvider { name: "k1".into() }),
        ]);

        let mut failed_stream = p.stream_chat(test_request()).await.unwrap();
        let _healthy_stream = p.stream_chat(test_request()).await.unwrap();
        assert_eq!(
            p.current_order().await,
            vec![1, 0],
            "opening the second stream should fall back to k1 and demote only k0"
        );

        assert!(matches!(
            TokioStreamExt::next(&mut failed_stream).await,
            Some(StreamChunk::Error(_))
        ));

        let mut next = p.stream_chat(test_request()).await.unwrap();
        assert_eq!(
            p.current_order().await,
            vec![1, 0],
            "idx1 must remain preferred; the idx0 stream error must not be attributed to idx1"
        );
        assert_eq!(p.blacklisted_count().await, 0);
        assert!(matches!(
            TokioStreamExt::next(&mut next).await,
            Some(StreamChunk::Text(t)) if t == "ok"
        ));
    }

    #[tokio::test]
    async fn single_provider_stream_failure_no_panic() {
        let p = ResilientProvider::new(vec![Box::new(StreamErrorProvider {
            name: "solo".into(),
        })]);

        let mut first = p.stream_chat(test_request()).await.unwrap();
        assert!(matches!(
            TokioStreamExt::next(&mut first).await,
            Some(StreamChunk::Error(_))
        ));
        let mut second = p.stream_chat(test_request()).await.unwrap();
        assert!(matches!(
            TokioStreamExt::next(&mut second).await,
            Some(StreamChunk::Error(_))
        ));

        let result = p.stream_chat(test_request()).await;
        assert!(result.is_err());
        assert_eq!(p.blacklisted_count().await, 1);
    }

    #[tokio::test]
    async fn stream_failure_decays_on_successful_stream() {
        let p = ResilientProvider::new(vec![Box::new(ScriptedStreamProvider {
            name: "solo".into(),
            calls: Arc::new(AtomicUsize::new(0)),
            script: vec![
                ScriptedChunk::Error,
                ScriptedChunk::ContentThenDone,
                ScriptedChunk::Error,
                ScriptedChunk::ContentThenDone,
            ],
        })]);

        let mut first = p.stream_chat(test_request()).await.unwrap();
        assert!(matches!(
            TokioStreamExt::next(&mut first).await,
            Some(StreamChunk::Error(_))
        ));

        let mut success = p.stream_chat(test_request()).await.unwrap();
        assert!(matches!(
            TokioStreamExt::next(&mut success).await,
            Some(StreamChunk::Text(t)) if t == "ok"
        ));
        assert!(matches!(
            TokioStreamExt::next(&mut success).await,
            Some(StreamChunk::Done)
        ));
        assert!(
            !p.stream_failures
                .lock()
                .expect("stream failure lock")
                .contains_key(&0),
            "successful content+Done stream should clear stale count"
        );

        let mut second_failure = p.stream_chat(test_request()).await.unwrap();
        assert!(matches!(
            TokioStreamExt::next(&mut second_failure).await,
            Some(StreamChunk::Error(_))
        ));

        let mut not_blacklisted = p.stream_chat(test_request()).await.unwrap();
        assert_eq!(p.blacklisted_count().await, 0);
        assert!(matches!(
            TokioStreamExt::next(&mut not_blacklisted).await,
            Some(StreamChunk::Text(t)) if t == "ok"
        ));
    }

    #[tokio::test]
    async fn pending_failure_not_erased_by_concurrent_success() {
        let p = ResilientProvider::new(vec![Box::new(SuccessProvider {
            name: "solo".into(),
        })]);

        let mut success = p.stream_chat(test_request()).await.unwrap();
        assert!(matches!(
            TokioStreamExt::next(&mut success).await,
            Some(StreamChunk::Text(t)) if t == "ok"
        ));

        {
            let mut failures = p.stream_failures.lock().expect("stream failure lock");
            failures.insert(0, STREAM_FAILURE_PENDING | 1);
        }

        assert!(matches!(
            TokioStreamExt::next(&mut success).await,
            Some(StreamChunk::Done)
        ));
        assert_eq!(
            p.stream_failures
                .lock()
                .expect("stream failure lock")
                .get(&0)
                .copied(),
            Some(STREAM_FAILURE_PENDING | 1),
            "successful stream must not erase a concurrent pending failure"
        );
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
    // ── D-INV-FALLBACK-VISIBLE (B106) ──────────────────────────────────────
    //
    // A silent fallback used to leave the caller labelling the turn with the
    // provider the user picked, so the tracing span, model_health.jsonl and
    // the chat bubble all credited a provider that had actually refused the
    // request (observed live: groq 413 -> deepseek answered, recorded as
    // "groq: success"). `last_fallback()` is what lets the turn layer tell
    // the truth. Removing the write in `stream_chat` fails these tests.

    /// The production shape B106 was found in: the requested provider (groq)
    /// rejects the request, the next one (deepseek) serves it. Shared so a
    /// change to the fixture cannot make some of these tests exercise a
    /// different chain than the others.
    fn groq_then_deepseek() -> ResilientProvider {
        ResilientProvider::new(vec![
            Box::new(FailProvider {
                name: "groq".into(),
            }),
            Box::new(SuccessProvider {
                name: "deepseek".into(),
            }),
        ])
    }

    #[tokio::test]
    async fn b106_last_fallback_reports_who_actually_served() {
        let p = groq_then_deepseek();
        assert!(
            p.last_fallback().is_none(),
            "no fallback should be reported before any call"
        );

        let _stream = p.stream_chat(test_request()).await.unwrap();

        let info = p.last_fallback().expect("fallback must be recorded");
        assert_eq!(info.requested, "groq");
        assert_eq!(info.served_by, "deepseek");
        assert!(
            !info.reason.is_empty(),
            "a reason is required so the user learns WHY"
        );
    }

    #[tokio::test]
    async fn b106_no_fallback_reported_on_clean_turn() {
        let p = ResilientProvider::new(vec![Box::new(SuccessProvider {
            name: "deepseek".into(),
        })]);
        let _stream = p.stream_chat(test_request()).await.unwrap();
        assert!(
            p.last_fallback().is_none(),
            "a clean turn must not claim a fallback"
        );
    }

    /// The banner must not persist: once a later turn is served by the
    /// requested provider, `last_fallback()` has to go back to None or every
    /// subsequent answer would carry a stale warning.
    #[tokio::test]
    async fn b106_stale_fallback_is_cleared_by_next_clean_turn() {
        let p = groq_then_deepseek();
        let _ = p.stream_chat(test_request()).await.unwrap();
        assert!(p.last_fallback().is_some());

        // After the failure the order is rotated so deepseek is now front and
        // serves directly — no fallback on this call.
        let _ = p.stream_chat(test_request()).await.unwrap();
        assert!(
            p.last_fallback().is_none(),
            "stale fallback leaked into a clean turn"
        );
    }

    /// Decorators must delegate, otherwise the fallback is invisible in prod
    /// (every provider is wrapped in TimeoutProvider + Box<dyn Provider>).
    #[tokio::test]
    async fn b106_boxed_provider_delegates_last_fallback() {
        let inner = groq_then_deepseek();
        let boxed: Box<dyn Provider> = Box::new(inner);
        let _ = boxed.stream_chat(test_request()).await.unwrap();
        let info = boxed
            .last_fallback()
            .expect("Box<dyn Provider> must delegate last_fallback");
        assert_eq!(info.served_by, "deepseek");
    }
    /// The live miss that made the first B106 attempt fail end-to-end:
    /// `session_ops::turn` wraps the resolved `Arc<dyn Provider>` in
    /// `provider_to_box` (`ArcProvider`) before handing it to `AgentLoop`.
    /// That wrapper forwarded only name/models/stream_chat, so every optional
    /// trait method — including `last_fallback()` — silently reverted to its
    /// default. A real groq 413 -> deepseek fallback was therefore invisible
    /// in production while all unit tests passed.
    #[tokio::test]
    async fn b106_arc_provider_wrapper_delegates_last_fallback() {
        let inner: std::sync::Arc<dyn Provider> = std::sync::Arc::new(groq_then_deepseek());
        let boxed = crate::provider::provider_to_box(&inner);
        let _ = boxed.stream_chat(test_request()).await.unwrap();
        let info = boxed
            .last_fallback()
            .expect("provider_to_box must delegate last_fallback (prod path)");
        assert_eq!(info.requested, "groq");
        assert_eq!(info.served_by, "deepseek");
    }
    /// B110, production wiring: `ResilientProvider` must be able to read
    /// `key_hint` from an entry wrapped exactly the way
    /// `create_provider_chain` wraps it (`TimeoutProvider`).
    ///
    /// The delegation unit tests check each wrapper in isolation; this pins the
    /// composed shape that was actually broken — B46 dead-key persistence reads
    /// `self.providers[idx].key_hint()`, and with the wrapper answering `None`
    /// the persist branch was unreachable no matter what the config said.
    /// (The persist call itself is env-gated and its file logic is covered by
    /// `dead_key_persist::tests`; what was broken, and what this guards, is the
    /// value reaching that branch at all.)
    #[test]
    fn b110_resilient_reads_key_hint_through_timeout_wrapper() {
        struct Keyed;
        #[async_trait]
        impl Provider for Keyed {
            fn name(&self) -> &str {
                "deepseek"
            }
            fn models(&self) -> Vec<ModelInfo> {
                vec![]
            }
            fn key_hint(&self) -> Option<String> {
                Some("dead-key-1".into())
            }
            async fn stream_chat(
                &self,
                _r: ChatRequest,
            ) -> Result<Pin<Box<dyn Stream<Item = StreamChunk> + Send>>> {
                Err(AgentError::Config("unused".into()))
            }
        }

        let wrapped: Box<dyn Provider> =
            Box::new(crate::provider::timeout::TimeoutProvider::with_defaults(
                Box::new(Keyed) as Box<dyn Provider>,
            ));
        let p = ResilientProvider::new(vec![wrapped]);

        assert_eq!(
            p.providers[0].key_hint().as_deref(),
            Some("dead-key-1"),
            "ResilientProvider must see the key through TimeoutProvider, \
             otherwise B46 dead-key persistence can never fire"
        );
    }
    /// B111: the BOOT AUDIT must persist dead keys, not only blacklist them.
    ///
    /// The runtime `stream_chat` path already called `persist_dead_key`, but the
    /// boot audit is what actually DISCOVERS most dead keys (it probes every
    /// key on every start) and it only wrote to the in-memory blacklist. So the
    /// keys B46 exists to stop re-probing were re-probed on every restart
    /// forever — the feature looked wired up and did nothing.
    ///
    /// Asserts the key VALUE reaches the persist branch through the production
    /// `TimeoutProvider` wrapping; the file-writing half is covered by
    /// `dead_key_persist::tests` and is env-gated.
    #[tokio::test]
    async fn b111_boot_audit_blacklists_and_can_reach_persist() {
        struct DeadKeyed;
        #[async_trait]
        impl Provider for DeadKeyed {
            fn name(&self) -> &str {
                "groq"
            }
            fn models(&self) -> Vec<ModelInfo> {
                vec![]
            }
            fn key_hint(&self) -> Option<String> {
                Some("gsk_dead".into())
            }
            async fn stream_chat(
                &self,
                _r: ChatRequest,
            ) -> Result<Pin<Box<dyn Stream<Item = StreamChunk> + Send>>> {
                Err(AgentError::ProviderTyped(
                    crate::provider::error::ProviderError::AuthFailed {
                        status: 401,
                        body: "Invalid API Key".into(),
                    },
                ))
            }
        }

        let wrapped: Box<dyn Provider> =
            Box::new(crate::provider::timeout::TimeoutProvider::with_defaults(
                Box::new(DeadKeyed) as Box<dyn Provider>,
            ));
        let p = ResilientProvider::new(vec![wrapped]);

        p.audit_keys_on_boot_with_model("probe-model").await;

        assert_eq!(
            p.blacklisted_count().await,
            1,
            "a 401 during boot audit must permanently blacklist the key"
        );
        assert_eq!(
            p.providers[0].key_hint().as_deref(),
            Some("gsk_dead"),
            "the boot-audit persist branch needs the key value to survive the \
             TimeoutProvider wrapper (B110); without it B46 can never fire"
        );
    }
    /// B111 follow-up: both dead-key discovery paths must record the SAME
    /// facts. They are now one method (`note_permanently_dead_key`); this pins
    /// the shared side effects so a future split cannot silently drop one.
    ///
    /// The counter matters beyond bookkeeping: it is exported as
    /// `naked_core_provider_permanent_blacklist_total` and read by the wiring
    /// health check, so a path that blacklists without counting makes dead keys
    /// invisible to monitoring.
    #[tokio::test]
    async fn b111_boot_audit_counts_permanent_blacklist_like_runtime_does() {
        struct DeadKeyed;
        #[async_trait]
        impl Provider for DeadKeyed {
            fn name(&self) -> &str {
                "groq"
            }
            fn models(&self) -> Vec<ModelInfo> {
                vec![]
            }
            fn key_hint(&self) -> Option<String> {
                Some("gsk_dead".into())
            }
            async fn stream_chat(
                &self,
                _r: ChatRequest,
            ) -> Result<Pin<Box<dyn Stream<Item = StreamChunk> + Send>>> {
                Err(AgentError::ProviderTyped(
                    crate::provider::error::ProviderError::AuthFailed {
                        status: 401,
                        body: "Invalid API Key".into(),
                    },
                ))
            }
        }

        let before = crate::types::PROVIDER_PERMANENT_BLACKLIST_COUNT
            .load(std::sync::atomic::Ordering::Relaxed);

        let p = ResilientProvider::new(vec![Box::new(DeadKeyed) as Box<dyn Provider>]);
        p.audit_keys_on_boot_with_model("probe-model").await;

        let after = crate::types::PROVIDER_PERMANENT_BLACKLIST_COUNT
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(
            after > before,
            "boot audit must bump the permanent-blacklist counter that \
             /metrics and the wiring health check read"
        );
    }
    // ── B112: findings from an independent SOLID/DRY/KISS audit ────────────

    /// The dead key must be filed under the provider that OWNS it.
    ///
    /// `note_permanently_dead_key` used to take the provider name as a
    /// parameter, and the two callers disagreed: the runtime path passed the
    /// entry that failed, the boot audit passed `providers.first()` for EVERY
    /// dead index. The outer chain is heterogeneous by construction, so a dead
    /// deepseek key got filed under anthropic.
    #[test]
    fn b112_dead_key_is_attributed_to_the_failing_entry_not_the_first() {
        struct Named {
            name: String,
            key: String,
        }
        #[async_trait]
        impl Provider for Named {
            fn name(&self) -> &str {
                &self.name
            }
            fn models(&self) -> Vec<ModelInfo> {
                vec![]
            }
            fn key_hint(&self) -> Option<String> {
                Some(self.key.clone())
            }
            async fn stream_chat(
                &self,
                _r: ChatRequest,
            ) -> Result<Pin<Box<dyn Stream<Item = StreamChunk> + Send>>> {
                Err(AgentError::Config("unused".into()))
            }
        }

        let p = ResilientProvider::new(vec![
            Box::new(Named {
                name: "anthropic".into(),
                key: "sk-anthropic".into(),
            }),
            Box::new(Named {
                name: "deepseek".into(),
                key: "sk-deepseek".into(),
            }),
        ]);

        // Index 1 is the one that died; its own name and key must be used.
        assert_eq!(p.providers[1].name(), "deepseek");
        assert_eq!(p.providers[1].key_hint().as_deref(), Some("sk-deepseek"));
        // And the signature must not let a caller supply a different name:
        // `note_permanently_dead_key` takes only the index.
        p.note_permanently_dead_key(1);
    }

    /// An untyped error whose body carries a 401 must be treated as a
    /// PERMANENT key failure, exactly like a typed one. Previously `permanent`
    /// was read only off the original error and defaulted to false when
    /// untyped, so such a key was blacklisted for an hour instead of removed,
    /// never persisted, and re-probed forever.
    #[test]
    fn b112_untyped_auth_error_classifies_as_permanent_key_failure() {
        let err_str = "provider returned API 401: invalid_api_key";
        let pe = super::super::error::ProviderError::from_llm_http(
            extract_status_from_error(err_str),
            err_str,
            "some-model",
        );
        assert!(pe.is_key_dead(), "401 body must be a key failure");
        assert!(
            pe.is_permanent_key_failure(),
            "401 must be PERMANENT — a transient classification means the dead \
             key is re-probed every hour and every restart"
        );
    }

    /// Status extraction must not fire on digits embedded in prose.
    #[test]
    fn b112_status_extraction_ignores_numbers_inside_words() {
        // Real statuses, delimited — still detected.
        assert_eq!(extract_status_from_error("API 401: nope"), 401);
        assert_eq!(extract_status_from_error("(402) payment"), 402);
        assert_eq!(extract_status_from_error("HTTP 429 slow down"), 429);
        assert_eq!(extract_status_from_error("error code: 502"), 502);

        // Prose that merely contains the digits — must NOT be classified.
        assert_eq!(
            extract_status_from_error("model llama-402b is unavailable"),
            0,
            "a model name containing 402 must not look like payment-required"
        );
        assert_eq!(
            extract_status_from_error("max_tokens 5000 exceeded"),
            0,
            "5000 must not be read as 500"
        );
        assert_eq!(extract_status_from_error("request id 4021 failed"), 0);
    }
}
