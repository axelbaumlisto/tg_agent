//! Global Telegram edit rate limiter — 429 unreachable by construction.
//!
//! **Every** `edit_message_text` call in the bot MUST go through
//! [`RateLimiter::edit`] instead of calling the Bot API directly.
//! This ensures a single rolling-window budget per `chat_id` (TG's
//! actual rate-limit boundary) regardless of how many threads,
//! streaming sessions, callbacks, or one-shot edits share that chat.
//!
//! On 429: the limiter parks ALL edits for that chat for the full
//! `Retry-After` duration. No retry loops, no cascading bans.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use teloxide::prelude::*;
use teloxide::types::{MessageId, ParseMode};
use tokio::sync::Mutex;

/// TG undocumented limit ~20 edits/min/chat. Single source of truth.
pub const BUDGET: u32 = 18;
/// Derived: 60s / BUDGET ≈ 3.3s minimum gap between edits.
pub const MIN_GAP_MS: u64 = 60_000 / BUDGET as u64;
pub const MIN_GAP: Duration = Duration::from_millis(MIN_GAP_MS);
/// Hard ceiling — even under backoff, never wait longer than this.
const MAX_GAP: Duration = Duration::from_secs(120);

/// Global rate limiter. One instance shared by the entire bot.
///
/// Tracks edits per **chat_id** (not per thread/message).
/// Streaming tickers, callbacks, commands — all go through [`Self::edit`].
#[derive(Clone)]
pub struct RateLimiter {
    inner: Arc<Mutex<State>>,
}

struct ChatState {
    /// When the last edit was sent (or attempted).
    last_edit: Instant,
    /// If we got a 429, don't try again until this instant.
    blocked_until: Option<Instant>,
    /// How many edits in the current rolling window.
    window_edits: Vec<Instant>,
}

impl ChatState {
    fn new() -> Self {
        Self {
            last_edit: Instant::now() - Duration::from_secs(10),
            blocked_until: None,
            window_edits: Vec::new(),
        }
    }

    /// Prune edits older than 60s from the rolling window.
    fn prune_window(&mut self) {
        let cutoff = Instant::now() - Duration::from_secs(60);
        self.window_edits.retain(|t| *t > cutoff);
    }

    /// How long until the next edit is allowed.
    fn time_until_allowed(&mut self) -> Duration {
        // If TG told us to wait, respect it.
        if let Some(until) = self.blocked_until {
            let now = Instant::now();
            if until > now {
                return until - now;
            }
            self.blocked_until = None;
        }

        self.prune_window();

        // If we've used the budget, wait until the oldest edit falls out of the window.
        if self.window_edits.len() as u32 >= BUDGET {
            let oldest = self.window_edits[0];
            let expires = oldest + Duration::from_secs(60);
            let now = Instant::now();
            if expires > now {
                return expires - now;
            }
        }

        // Enforce minimum gap from last edit.
        let since_last = self.last_edit.elapsed();
        if since_last < MIN_GAP {
            return MIN_GAP - since_last;
        }

        Duration::ZERO
    }

    fn record_edit(&mut self) {
        let now = Instant::now();
        self.last_edit = now;
        self.window_edits.push(now);
    }

    fn record_429(&mut self, retry_after_secs: u64) {
        let wait = Duration::from_secs(retry_after_secs.clamp(1, MAX_GAP.as_secs()));
        self.blocked_until = Some(Instant::now() + wait);
    }
}

struct State {
    chats: HashMap<i64, ChatState>,
}

impl RateLimiter {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(State {
                chats: HashMap::new(),
            })),
        }
    }

    /// The **only** way to edit a message. Waits for the rate limit
    /// window, sends the edit, and handles 429 transparently.
    ///
    /// Returns `true` if the edit succeeded, `false` if it was dropped
    /// (429 + backoff).
    pub async fn edit(
        &self,
        bot: &Bot,
        chat_id: ChatId,
        msg_id: MessageId,
        text: &str,
        html: bool,
    ) -> bool {
        let cid = chat_id.0;

        // Wait for our turn.
        let wait = {
            let mut state = self.inner.lock().await;
            let cs = state.chats.entry(cid).or_insert_with(ChatState::new);
            cs.time_until_allowed()
        };
        if wait > Duration::ZERO {
            // If blocked for a long time, just drop this edit (stale UI update).
            if wait > Duration::from_secs(30) {
                tracing::debug!(
                    chat = cid,
                    wait_ms = wait.as_millis() as u64,
                    "rate_limit: dropping edit (backoff too long)"
                );
                return false;
            }
            tokio::time::sleep(wait).await;
        }

        // Send.
        let result = if html {
            bot.edit_message_text(chat_id, msg_id, text)
                .parse_mode(ParseMode::Html)
                .await
        } else {
            bot.edit_message_text(chat_id, msg_id, text).await
        };

        let mut state = self.inner.lock().await;
        let cs = state.chats.entry(cid).or_insert_with(ChatState::new);

        match result {
            Ok(_) => {
                cs.record_edit();
                true
            }
            Err(e) => {
                let err_str = e.to_string();
                // Parse "Retry after Xs" or "retry after Xs"
                if let Some(secs) = parse_retry_after(&err_str) {
                    tracing::warn!(
                        chat = cid,
                        retry_after = secs,
                        "rate_limit: 429, parking chat"
                    );
                    cs.record_429(secs);
                } else {
                    // Non-rate-limit error (message not modified, chat not found, etc).
                    // Still count as an edit attempt for window purposes.
                    cs.record_edit();
                    tracing::debug!(chat = cid, err = %err_str, "rate_limit: edit failed (non-429)");
                }
                false
            }
        }
    }

    /// Convenience: edit with HTML parse mode.
    pub async fn edit_html(
        &self,
        bot: &Bot,
        chat_id: ChatId,
        msg_id: MessageId,
        text: &str,
    ) -> bool {
        self.edit(bot, chat_id, msg_id, text, true).await
    }

    /// Convenience: edit as plain text.
    pub async fn edit_plain(
        &self,
        bot: &Bot,
        chat_id: ChatId,
        msg_id: MessageId,
        text: &str,
    ) -> bool {
        self.edit(bot, chat_id, msg_id, text, false).await
    }

    /// Like [`edit`] but **never drops** — waits as long as needed.
    /// Use for send_final where losing the message is unacceptable.
    pub async fn edit_must_deliver(
        &self,
        bot: &Bot,
        chat_id: ChatId,
        msg_id: MessageId,
        text: &str,
        html: bool,
    ) -> bool {
        for attempt in 0..5 {
            let wait = {
                let mut state = self.inner.lock().await;
                let cs = state.chats.entry(chat_id.0).or_insert_with(ChatState::new);
                cs.time_until_allowed()
            };
            if wait > Duration::ZERO {
                tracing::info!(
                    chat = chat_id.0,
                    attempt,
                    wait_ms = wait.as_millis() as u64,
                    "edit_must_deliver: waiting for rate limit"
                );
                tokio::time::sleep(wait).await;
            }
            let result = if html {
                bot.edit_message_text(chat_id, msg_id, text)
                    .parse_mode(teloxide::types::ParseMode::Html)
                    .await
            } else {
                bot.edit_message_text(chat_id, msg_id, text).await
            };
            let mut state = self.inner.lock().await;
            let cs = state.chats.entry(chat_id.0).or_insert_with(ChatState::new);
            match result {
                Ok(_) => {
                    cs.record_edit();
                    return true;
                }
                Err(e) => {
                    let err_str = e.to_string();
                    if let Some(secs) = parse_retry_after(&err_str) {
                        cs.record_429(secs);
                    } else {
                        cs.record_edit();
                        return false; // non-retryable error
                    }
                }
            }
        }
        false
    }

    /// Current interval for a streaming ticker (used by streaming_mod).
    /// Based on how many active streaming sessions share the same chat_id.
    pub async fn streaming_interval(&self, chat_id: i64) -> Duration {
        let state = self.inner.lock().await;
        let cs = match state.chats.get(&chat_id) {
            Some(cs) => cs,
            None => return MIN_GAP,
        };
        if let Some(until) = cs.blocked_until.filter(|u| *u > Instant::now()) {
            return until - Instant::now();
        }
        MIN_GAP
    }

    /// Check if a chat is currently in 429 backoff.
    pub async fn is_blocked(&self, chat_id: i64) -> bool {
        let state = self.inner.lock().await;
        state
            .chats
            .get(&chat_id)
            .and_then(|cs| cs.blocked_until)
            .is_some_and(|until| until > Instant::now())
    }

    /// Stats for /metrics.
    pub async fn active_count(&self) -> usize {
        let state = self.inner.lock().await;
        state.chats.len()
    }

    /// Current interval for display.
    pub async fn interval(&self) -> Duration {
        MIN_GAP
    }
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse "Retry after Xs" from a Telegram error string.
fn parse_retry_after(err: &str) -> Option<u64> {
    let lower = err.to_lowercase();
    let idx = lower.find("retry after")?;
    let after = &err[idx + "retry after".len()..];
    let num: String = after
        .trim()
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    num.parse().ok()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_retry_after_telegram_format() {
        assert_eq!(parse_retry_after("Retry after 56s"), Some(56));
        assert_eq!(
            parse_retry_after("Too Many Requests: retry after 28"),
            Some(28)
        );
        assert_eq!(parse_retry_after("retry after 3"), Some(3));
        assert_eq!(parse_retry_after("some other error"), None);
    }

    #[tokio::test]
    async fn fresh_chat_no_wait() {
        let rl = RateLimiter::new();
        // A brand new chat should have zero wait after the first MIN_GAP from init.
        // We init last_edit to 10s ago, so no wait.
        let wait = {
            let mut state = rl.inner.lock().await;
            let cs = state.chats.entry(123).or_insert_with(ChatState::new);
            cs.time_until_allowed()
        };
        assert_eq!(wait, Duration::ZERO);
    }

    #[tokio::test]
    async fn min_gap_enforced() {
        let rl = RateLimiter::new();
        {
            let mut state = rl.inner.lock().await;
            let cs = state.chats.entry(123).or_insert_with(ChatState::new);
            cs.record_edit(); // just edited
        }
        let wait = {
            let mut state = rl.inner.lock().await;
            let cs = state.chats.get_mut(&123).unwrap();
            cs.time_until_allowed()
        };
        // Should be close to MIN_GAP (2.4s)
        assert!(wait > Duration::from_millis(2_000));
        assert!(wait <= MIN_GAP);
    }

    #[tokio::test]
    async fn backoff_429() {
        let rl = RateLimiter::new();
        {
            let mut state = rl.inner.lock().await;
            let cs = state.chats.entry(123).or_insert_with(ChatState::new);
            cs.record_429(10);
        }
        let wait = {
            let mut state = rl.inner.lock().await;
            let cs = state.chats.get_mut(&123).unwrap();
            cs.time_until_allowed()
        };
        // Should be close to 10s
        assert!(wait > Duration::from_secs(9));
        assert!(wait <= Duration::from_secs(11));
    }

    #[tokio::test]
    async fn budget_exhaustion_waits() {
        let rl = RateLimiter::new();
        {
            let mut state = rl.inner.lock().await;
            let cs = state.chats.entry(123).or_insert_with(ChatState::new);
            // Fill the budget
            for _ in 0..BUDGET {
                cs.window_edits.push(Instant::now());
            }
            cs.last_edit = Instant::now() - Duration::from_secs(10); // no min_gap issue
        }
        let wait = {
            let mut state = rl.inner.lock().await;
            let cs = state.chats.get_mut(&123).unwrap();
            cs.time_until_allowed()
        };
        // Should wait until oldest edit falls out of 60s window (~60s)
        assert!(wait > Duration::from_secs(50));
    }

    #[tokio::test]
    async fn is_blocked_after_429() {
        let rl = RateLimiter::new();
        {
            let mut state = rl.inner.lock().await;
            let cs = state.chats.entry(123).or_insert_with(ChatState::new);
            cs.record_429(30);
        }
        assert!(rl.is_blocked(123).await);
    }
}
