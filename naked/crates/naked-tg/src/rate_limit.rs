//! Adaptive per-chat Telegram rate limiter.
//!
//! Telegram Bot API limits:
//! - ~30 msg/edits per chat per minute
//! - ~30 msg/sec globally
//! - 429 responses include `retry_after` seconds
//!
//! This limiter tracks per-chat + global sliding windows and adapts
//! the edit interval on 429 (backoff) and success (recovery).
//! Never panics or drops — just delays.
//!
//! Design: single struct, no I/O, no Telegram API dependency.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

/// Conservative per-chat limit (Telegram allows ~30).
const PER_CHAT_PER_MIN: usize = 25;
/// Global bot-wide limit per minute.
const GLOBAL_PER_MIN: usize = 200;
/// Floor for adaptive edit gap (ms).
pub const MIN_GAP_MS: u64 = 2_000;
/// Ceiling for adaptive edit gap (ms).
pub const MAX_GAP_MS: u64 = 15_000;
/// Starting edit gap (ms).
const DEFAULT_GAP_MS: u64 = 2_500;
/// On success: gap *= RECOVERY (shrink toward floor).
const RECOVERY: f64 = 0.85;
/// On 429: gap *= BACKOFF (grow toward ceiling).
const BACKOFF: f64 = 2.0;

/// Chat identifier: (chat_id, thread_id).
pub type ChatKey = (i64, Option<i32>);

struct PerChat {
    window: VecDeque<tokio::time::Instant>,
    gap_ms: u64,
    last: Option<tokio::time::Instant>,
}

impl PerChat {
    fn new() -> Self {
        Self {
            window: VecDeque::new(),
            gap_ms: DEFAULT_GAP_MS,
            last: None,
        }
    }

    fn prune(&mut self, now: tokio::time::Instant) {
        let cutoff = now - Duration::from_secs(60);
        while self.window.front().is_some_and(|&t| t < cutoff) {
            self.window.pop_front();
        }
    }
}

/// Adaptive rate limiter. Cheap to clone (Arc inside).
#[derive(Clone)]
pub struct RateLimiter {
    chats: Arc<Mutex<HashMap<ChatKey, PerChat>>>,
    global: Arc<Mutex<VecDeque<tokio::time::Instant>>>,
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new()
    }
}

impl RateLimiter {
    pub fn new() -> Self {
        Self {
            chats: Arc::new(Mutex::new(HashMap::new())),
            global: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// Block until both per-chat and global limits allow a call.
    /// Returns the current gap for this chat.
    pub async fn acquire(&self, chat: ChatKey) -> Duration {
        loop {
            // ── Global ──
            {
                let mut g = self.global.lock().await;
                let now = tokio::time::Instant::now();
                let cutoff = now - Duration::from_secs(60);
                while g.front().is_some_and(|&t| t < cutoff) {
                    g.pop_front();
                }
                if g.len() >= GLOBAL_PER_MIN {
                    let wait = *g.front().unwrap() + Duration::from_secs(60);
                    drop(g);
                    tokio::time::sleep_until(wait).await;
                    continue;
                }
            }

            // ── Per-chat ──
            let mut chats = self.chats.lock().await;
            let pc = chats.entry(chat).or_insert_with(PerChat::new);
            let now = tokio::time::Instant::now();
            pc.prune(now);

            if pc.window.len() >= PER_CHAT_PER_MIN {
                let wait = *pc.window.front().unwrap() + Duration::from_secs(60);
                drop(chats);
                tokio::time::sleep_until(wait).await;
                continue;
            }

            // ── Adaptive gap ──
            let gap = Duration::from_millis(pc.gap_ms);
            if let Some(last) = pc.last {
                let since = now.duration_since(last);
                if since < gap {
                    drop(chats);
                    tokio::time::sleep(gap - since).await;
                    continue;
                }
            }

            // Record
            pc.window.push_back(now);
            pc.last = Some(now);
            let out = Duration::from_millis(pc.gap_ms);
            drop(chats);
            self.global.lock().await.push_back(now);
            return out;
        }
    }

    /// Successful call — shrink gap toward floor.
    pub async fn report_ok(&self, chat: ChatKey) {
        let mut chats = self.chats.lock().await;
        if let Some(pc) = chats.get_mut(&chat) {
            pc.gap_ms = ((pc.gap_ms as f64 * RECOVERY) as u64).max(MIN_GAP_MS);
        }
    }

    /// 429 received — grow gap toward ceiling.
    pub async fn report_429(&self, chat: ChatKey, retry_after: Option<u64>) {
        let mut chats = self.chats.lock().await;
        let pc = chats.entry(chat).or_insert_with(PerChat::new);
        if let Some(secs) = retry_after {
            pc.gap_ms = (secs * 1000 / PER_CHAT_PER_MIN as u64)
                .max(pc.gap_ms)
                .min(MAX_GAP_MS);
        } else {
            pc.gap_ms = ((pc.gap_ms as f64 * BACKOFF) as u64).min(MAX_GAP_MS);
        }
    }

    /// Current gap for a chat (non-blocking peek).
    /// Current gap for a chat (non-blocking peek).
    pub async fn gap(&self, chat: ChatKey) -> Duration {
        let chats = self.chats.lock().await;
        Duration::from_millis(
            chats
                .get(&chat)
                .map(|pc| pc.gap_ms)
                .unwrap_or(DEFAULT_GAP_MS),
        )
    }

    /// Snapshot for /metrics display.
    pub async fn stats(&self) -> RateLimiterStats {
        let chats = self.chats.lock().await;
        let global = self.global.lock().await;
        RateLimiterStats {
            active_chats: chats.len(),
            global_calls_last_min: global.len(),
            chat_gaps: chats
                .iter()
                .map(|(k, v)| (*k, v.gap_ms, v.window.len()))
                .collect(),
        }
    }
}

/// Metrics snapshot from the rate limiter.
#[derive(Debug)]
pub struct RateLimiterStats {
    pub active_chats: usize,
    pub global_calls_last_min: usize,
    /// (chat_key, current_gap_ms, calls_in_window)
    pub chat_gaps: Vec<(ChatKey, u64, usize)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn first_acquire_is_instant() {
        let rl = RateLimiter::new();
        let t0 = tokio::time::Instant::now();
        let _ = rl.acquire((1, None)).await;
        assert!(t0.elapsed() < Duration::from_millis(50));
    }

    #[tokio::test]
    async fn second_acquire_waits_gap() {
        let rl = RateLimiter::new();
        let chat = (1, None);
        let _ = rl.acquire(chat).await;
        let t0 = tokio::time::Instant::now();
        let _ = rl.acquire(chat).await;
        assert!(
            t0.elapsed() >= Duration::from_millis(MIN_GAP_MS - 200),
            "waited {:?}",
            t0.elapsed()
        );
    }

    #[tokio::test]
    async fn different_chats_dont_block() {
        let rl = RateLimiter::new();
        let _ = rl.acquire((1, None)).await;
        let t0 = tokio::time::Instant::now();
        let _ = rl.acquire((2, None)).await;
        assert!(t0.elapsed() < Duration::from_millis(100));
    }

    #[tokio::test]
    async fn backoff_grows_gap() {
        let rl = RateLimiter::new();
        let chat = (1, None);
        let before = rl.gap(chat).await;
        rl.report_429(chat, None).await;
        let after = rl.gap(chat).await;
        assert!(after > before, "{after:?} should > {before:?}");
    }

    #[tokio::test]
    async fn recovery_shrinks_gap() {
        let rl = RateLimiter::new();
        let chat = (1, None);
        rl.report_429(chat, None).await;
        let backed = rl.gap(chat).await;
        rl.report_ok(chat).await;
        let recovered = rl.gap(chat).await;
        assert!(recovered < backed, "{recovered:?} should < {backed:?}");
    }

    #[tokio::test]
    async fn gap_floor() {
        let rl = RateLimiter::new();
        let chat = (1, None);
        for _ in 0..50 {
            rl.report_ok(chat).await;
        }
        let gap = rl.gap(chat).await;
        assert!(gap >= Duration::from_millis(MIN_GAP_MS), "floor: {gap:?}");
    }

    #[tokio::test]
    async fn gap_ceiling() {
        let rl = RateLimiter::new();
        let chat = (1, None);
        for _ in 0..50 {
            rl.report_429(chat, None).await;
        }
        let gap = rl.gap(chat).await;
        assert!(gap <= Duration::from_millis(MAX_GAP_MS), "ceiling: {gap:?}");
    }

    #[tokio::test]
    async fn retry_after_respected() {
        let rl = RateLimiter::new();
        let chat = (1, None);
        rl.report_429(chat, Some(30)).await;
        let gap = rl.gap(chat).await;
        assert!(gap >= Duration::from_millis(1_000), "retry_after: {gap:?}");
    }

    #[tokio::test]
    async fn concurrent_chats_fair() {
        let rl = RateLimiter::new();
        // 3 chats each acquire — all should succeed quickly
        let t0 = tokio::time::Instant::now();
        let (a, b, c) = tokio::join!(
            rl.acquire((1, None)),
            rl.acquire((2, None)),
            rl.acquire((3, None)),
        );
        assert!(t0.elapsed() < Duration::from_millis(200));
        // All returned reasonable gaps
        assert!(a >= Duration::from_millis(MIN_GAP_MS));
        assert!(b >= Duration::from_millis(MIN_GAP_MS));
        assert!(c >= Duration::from_millis(MIN_GAP_MS));
    }
}
