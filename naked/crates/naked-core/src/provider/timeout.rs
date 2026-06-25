//! [`Provider`] LSP-compliant decorator with two configurable timeouts.
//!
//! R3 of `PLAN_NEXT_SESSION.md` (2026-05-10). Wraps any concrete
//! `Provider` (`Anthropic`, `OpenAICompat`, `Copilot`, `ResilientProvider`,
//! …) so:
//!
//!   1. **Connect timeout.** `stream_chat()` itself \u2014 the HTTP POST
//!      that opens the stream \u2014 must return a stream handle within
//!      `connect_timeout`. If it doesn't, the call fails with
//!      [`ProviderError::Other { status: 0, body: "<provider> connect timeout after Ns" }`].
//!      Defends against providers that hang forever on TCP-level
//!      keep-alive shenanigans before the first byte ever arrives.
//!
//!   2. **Inter-chunk timeout.** Once the stream is open, any single
//!      chunk must arrive within `inter_chunk_timeout`. If the gap
//!      between two chunks exceeds the budget, the wrapped stream
//!      yields a final [`StreamChunk::Error`] and terminates.
//!      Closes the failure mode where a provider opens a stream,
//!      sends a few bytes, and then silently stalls indefinitely \u2014
//!      the empty-content retry budget can't catch this because the
//!      stream never closes; the agent loop just waits on
//!      `stream.next().await` forever.
//!
//! LSP contract: a `TimeoutProvider<P>` is observably indistinguishable
//! from `P` on the happy path. The only behaviour change is the
//! injection of a terminal `Error` chunk (or an `Err` from
//! `stream_chat`) when timeouts elapse; both are already in `P`'s
//! contract.
//!
//! Pass-through: `name()`, `models()`, `blacklisted_key_count()`,
//! `total_key_count()` all delegate to the inner provider so existing
//! observability + key-pool reporting keeps working.

use std::pin::Pin;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use tokio_stream::{Stream, StreamExt};

use super::Provider;
use crate::error::{AgentError, Result};
use crate::provider::ChatRequest;
use crate::provider::error::ProviderError;
use crate::types::{ModelInfo, StreamChunk};

/// Default connect timeout. Generous enough for slow LLM round-trip
/// regions but tight enough to cycle a wedged process inside one
/// systemd watchdog window (30s default `WatchdogSec`).
pub const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(20);

/// Default inter-chunk timeout. LLM streams routinely have 2-5s
/// gaps during reasoning chains; 60s is a comfortable upper bound
/// that still catches actually-stalled streams quickly.
pub const DEFAULT_INTER_CHUNK_TIMEOUT: Duration = Duration::from_secs(60);

/// Decorator: wrap any `Box<dyn Provider>` to enforce timeouts.
///
/// Generic-on-`P` (rather than `Box<dyn Provider>`) so a static
/// dispatch `TimeoutProvider<ResilientProvider>` is also possible
/// for hot paths.
pub struct TimeoutProvider<P: Provider> {
    inner: P,
    connect_timeout: Duration,
    inter_chunk_timeout: Duration,
}

impl<P: Provider> TimeoutProvider<P> {
    /// Wrap with [`DEFAULT_CONNECT_TIMEOUT`] and [`DEFAULT_INTER_CHUNK_TIMEOUT`].
    pub fn with_defaults(inner: P) -> Self {
        Self::new(inner, DEFAULT_CONNECT_TIMEOUT, DEFAULT_INTER_CHUNK_TIMEOUT)
    }

    /// Wrap with explicit timeouts. Both must be `> 0`; otherwise
    /// the corresponding guard is effectively bypassed (`tokio::time::timeout`
    /// with zero duration fires on the first `await` poll).
    pub fn new(inner: P, connect_timeout: Duration, inter_chunk_timeout: Duration) -> Self {
        Self {
            inner,
            connect_timeout,
            inter_chunk_timeout,
        }
    }
}

#[async_trait]
impl<P: Provider> Provider for TimeoutProvider<P> {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn models(&self) -> Vec<ModelInfo> {
        self.inner.models()
    }

    fn blacklisted_key_count(&self) -> usize {
        self.inner.blacklisted_key_count()
    }

    fn total_key_count(&self) -> usize {
        self.inner.total_key_count()
    }

    async fn audit_keys_on_boot(&self) {
        self.inner.audit_keys_on_boot().await
    }

    async fn stream_chat(
        &self,
        request: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = StreamChunk> + Send>>> {
        // 1) Connect-side timeout: bound the HTTP POST + initial
        //    stream-open. tokio::time::timeout is cancel-safe; if it
        //    fires the inner future is dropped and the connection is
        //    closed by reqwest's RAII.
        let started = Instant::now();
        let connect_fut = self.inner.stream_chat(request);
        let inner_stream = match tokio::time::timeout(self.connect_timeout, connect_fut).await {
            Ok(Ok(s)) => s,
            Ok(Err(e)) => return Err(e),
            Err(_elapsed) => {
                crate::types::PROVIDER_CONNECT_TIMEOUT_COUNT.fetch_add(1, Ordering::Relaxed);
                let body = format!(
                    "{} connect timeout after {}s",
                    self.inner.name(),
                    self.connect_timeout.as_secs(),
                );
                return Err(AgentError::ProviderTyped(ProviderError::Other {
                    status: 0,
                    body,
                }));
            }
        };

        // 2) Inter-chunk timeout: wrap the returned stream so that any
        //    silence > inter_chunk_timeout terminates with an Error
        //    chunk. The agent loop's existing empty-content / Error
        //    handling treats this as a normal provider failure and
        //    can retry / surface it without special-casing.
        let provider_label = self.inner.name().to_string();
        let budget = self.inter_chunk_timeout;
        let mut first_chunk_seen = false;
        let guarded = inner_stream.timeout(budget).map(move |item| match item {
            Ok(chunk) => {
                if !first_chunk_seen && !matches!(chunk, StreamChunk::Error(_)) {
                    first_chunk_seen = true;
                    crate::metrics_hist::record_provider_stream_open(
                        started.elapsed().as_millis() as u64
                    );
                }
                chunk
            }
            Err(_elapsed) => {
                crate::types::PROVIDER_INTER_CHUNK_TIMEOUT_COUNT.fetch_add(1, Ordering::Relaxed);
                StreamChunk::Error(format!(
                    "{provider_label} inter-chunk timeout after {}s",
                    budget.as_secs(),
                ))
            }
        });
        Ok(Box::pin(guarded))
    }
}

#[cfg(test)]
mod tests {
    //! Pinning tests for the LSP wrapper.
    use std::pin::Pin;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use async_trait::async_trait;
    use tokio_stream::{Stream, StreamExt};

    use super::*;
    use crate::error::Result;
    use crate::provider::{ChatRequest, Provider};
    use crate::types::{ModelInfo, StreamChunk};

    /// Provider that returns a stream after a configurable delay,
    /// then yields chunks at configurable intervals.
    struct DelayProvider {
        connect_delay: Duration,
        chunks: Vec<(Duration, StreamChunk)>,
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Provider for DelayProvider {
        fn name(&self) -> &str {
            "delay"
        }
        fn models(&self) -> Vec<ModelInfo> {
            vec![]
        }
        async fn stream_chat(
            &self,
            _request: ChatRequest,
        ) -> Result<Pin<Box<dyn Stream<Item = StreamChunk> + Send>>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            tokio::time::sleep(self.connect_delay).await;
            let chunks = self.chunks.clone();
            let s = tokio_stream::iter(chunks).then(|(d, c)| async move {
                tokio::time::sleep(d).await;
                c
            });
            Ok(Box::pin(s))
        }
    }

    fn make_request() -> ChatRequest {
        ChatRequest {
            model: "x".into(),
            system: "".into(),
            messages: vec![],
            tools: vec![],
            max_tokens: 10,
            temperature: None,
            reasoning: None,
        }
    }

    #[tokio::test]
    async fn happy_path_passes_chunks_through_unchanged() {
        let provider = DelayProvider {
            connect_delay: Duration::from_millis(0),
            chunks: vec![
                (Duration::from_millis(0), StreamChunk::Text("hi".into())),
                (Duration::from_millis(0), StreamChunk::Done),
            ],
            calls: Arc::new(AtomicUsize::new(0)),
        };
        let wrapped =
            TimeoutProvider::new(provider, Duration::from_secs(5), Duration::from_secs(5));
        let mut s = wrapped.stream_chat(make_request()).await.unwrap();
        let first = s.next().await.unwrap();
        assert!(matches!(first, StreamChunk::Text(t) if t == "hi"));
        let second = s.next().await.unwrap();
        assert!(matches!(second, StreamChunk::Done));
    }

    #[tokio::test]
    async fn connect_timeout_fires_when_inner_hangs() {
        let provider = DelayProvider {
            connect_delay: Duration::from_secs(60), // way over budget
            chunks: vec![],
            calls: Arc::new(AtomicUsize::new(0)),
        };
        let wrapped =
            TimeoutProvider::new(provider, Duration::from_millis(50), Duration::from_secs(60));
        let before = crate::types::PROVIDER_CONNECT_TIMEOUT_COUNT.load(Ordering::Relaxed);
        let before_latency = crate::metrics_hist::snapshot();
        let started = std::time::Instant::now();
        let res = wrapped.stream_chat(make_request()).await;
        let elapsed = started.elapsed();
        assert!(
            elapsed < Duration::from_millis(500),
            "must give up at ~50ms, took {elapsed:?}"
        );
        // Avoid `.unwrap_err()` because `Pin<Box<dyn Stream>>` is
        // not `Debug`. Match instead.
        match res {
            Err(e) => {
                let s = format!("{e:?}");
                assert!(
                    s.contains("connect timeout"),
                    "error must mention connect timeout: {s}"
                );
            }
            Ok(_) => panic!("connect timeout must surface as Err"),
        }
        let after = crate::types::PROVIDER_CONNECT_TIMEOUT_COUNT.load(Ordering::Relaxed);
        assert_eq!(after - before, 1, "connect timeout counter delta");
        let after_latency = crate::metrics_hist::snapshot();
        assert_eq!(
            after_latency.provider_count - before_latency.provider_count,
            0,
            "connect-timeout failures are not successful stream-open samples"
        );
    }

    #[tokio::test]
    async fn provider_stream_open_records_delay_until_first_chunk() {
        let provider = DelayProvider {
            connect_delay: Duration::from_millis(0),
            chunks: vec![
                (Duration::from_millis(600), StreamChunk::Text("hi".into())),
                (Duration::from_millis(0), StreamChunk::Done),
            ],
            calls: Arc::new(AtomicUsize::new(0)),
        };
        let wrapped =
            TimeoutProvider::new(provider, Duration::from_secs(5), Duration::from_secs(5));
        let before = crate::metrics_hist::snapshot();

        let mut s = wrapped.stream_chat(make_request()).await.unwrap();
        let first = s.next().await.unwrap();
        assert!(matches!(first, StreamChunk::Text(t) if t == "hi"));

        let after_first = crate::metrics_hist::snapshot();
        assert_eq!(after_first.provider_count - before.provider_count, 1);
        assert_eq!(
            after_first.provider_500ms_to_2s - before.provider_500ms_to_2s,
            1
        );
        assert!(
            after_first.provider_sum_ms - before.provider_sum_ms >= 600,
            "provider stream-open sum delta must include first-chunk delay"
        );

        let second = s.next().await.unwrap();
        assert!(matches!(second, StreamChunk::Done));
        let after_second = crate::metrics_hist::snapshot();
        assert_eq!(after_second.provider_count, after_first.provider_count);
        assert_eq!(
            after_second.provider_500ms_to_2s,
            after_first.provider_500ms_to_2s
        );
        assert_eq!(after_second.provider_sum_ms, after_first.provider_sum_ms);
    }

    #[tokio::test]
    async fn provider_stream_open_not_recorded_when_first_chunk_is_error() {
        let provider = DelayProvider {
            connect_delay: Duration::from_millis(0),
            chunks: vec![(
                Duration::from_millis(0),
                StreamChunk::Error("upstream failed before text".into()),
            )],
            calls: Arc::new(AtomicUsize::new(0)),
        };
        let wrapped =
            TimeoutProvider::new(provider, Duration::from_secs(5), Duration::from_secs(5));
        let before = crate::metrics_hist::snapshot();

        let mut s = wrapped.stream_chat(make_request()).await.unwrap();
        let first = s.next().await.unwrap();
        assert!(matches!(first, StreamChunk::Error(msg) if msg == "upstream failed before text"));

        let after = crate::metrics_hist::snapshot();
        assert_eq!(after.provider_count - before.provider_count, 0);
        assert_eq!(after.provider_sum_ms - before.provider_sum_ms, 0);
    }

    #[tokio::test]
    async fn provider_stream_open_not_recorded_for_synthetic_inter_chunk_timeout_before_first_chunk()
     {
        let provider = DelayProvider {
            connect_delay: Duration::from_millis(0),
            chunks: vec![(
                Duration::from_secs(300),
                StreamChunk::Text("never arrives".into()),
            )],
            calls: Arc::new(AtomicUsize::new(0)),
        };
        let wrapped =
            TimeoutProvider::new(provider, Duration::from_secs(5), Duration::from_millis(50));
        let before = crate::metrics_hist::snapshot();

        let mut s = wrapped.stream_chat(make_request()).await.unwrap();
        let first = s.next().await.unwrap();
        assert!(matches!(first, StreamChunk::Error(msg) if msg.contains("inter-chunk timeout")));

        let after = crate::metrics_hist::snapshot();
        assert_eq!(after.provider_count - before.provider_count, 0);
        assert_eq!(after.provider_sum_ms - before.provider_sum_ms, 0);
    }

    #[tokio::test]
    async fn inter_chunk_timeout_yields_error_chunk_and_terminates() {
        // Stream opens fine, first chunk arrives at 10ms, then
        // a 5-minute silence — must be killed at 50ms gap.
        let provider = DelayProvider {
            connect_delay: Duration::from_millis(0),
            chunks: vec![
                (
                    Duration::from_millis(10),
                    StreamChunk::Text("partial".into()),
                ),
                (
                    Duration::from_secs(300),
                    StreamChunk::Text("never arrives".into()),
                ),
            ],
            calls: Arc::new(AtomicUsize::new(0)),
        };
        let wrapped =
            TimeoutProvider::new(provider, Duration::from_secs(5), Duration::from_millis(50));
        let mut s = wrapped.stream_chat(make_request()).await.unwrap();
        let first = s.next().await.unwrap();
        assert!(matches!(first, StreamChunk::Text(t) if t == "partial"));
        let before = crate::types::PROVIDER_INTER_CHUNK_TIMEOUT_COUNT.load(Ordering::Relaxed);
        let started = std::time::Instant::now();
        let second = s.next().await.unwrap();
        let elapsed = started.elapsed();
        match second {
            StreamChunk::Error(msg) => {
                assert!(
                    msg.contains("inter-chunk timeout"),
                    "error must mention inter-chunk timeout: {msg}"
                );
            }
            other => panic!("expected Error chunk, got {other:?}"),
        }
        assert!(
            elapsed < Duration::from_millis(500),
            "inter-chunk arm must fire near the budget, took {elapsed:?}"
        );
        let after = crate::types::PROVIDER_INTER_CHUNK_TIMEOUT_COUNT.load(Ordering::Relaxed);
        assert_eq!(after - before, 1, "inter-chunk timeout counter delta");
    }

    #[tokio::test]
    async fn pass_through_metadata_unchanged_lsp() {
        // LSP: name() and trivial accessors must mirror the inner
        // provider so a TimeoutProvider<P> is observable as P.
        let provider = DelayProvider {
            connect_delay: Duration::from_millis(0),
            chunks: vec![],
            calls: Arc::new(AtomicUsize::new(0)),
        };
        let wrapped = TimeoutProvider::with_defaults(provider);
        assert_eq!(wrapped.name(), "delay");
        assert_eq!(wrapped.blacklisted_key_count(), 0);
        assert_eq!(wrapped.total_key_count(), 1);
    }
}
