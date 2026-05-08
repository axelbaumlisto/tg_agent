//! Panic-safe task spawning + global panic hook.
//!
//! Level 1: `install_panic_hook()` — every panic → tracing::error.
//! Level 2: `spawn_guarded()` — panic → error message in user's TG chat.

use std::future::Future;
use teloxide::prelude::*;
use teloxide::types::{ChatId, ThreadId};
use tokio::task::JoinHandle;

/// Install global panic hook. Call once at startup.
pub fn install_panic_hook() {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let location = info
            .location()
            .map(|l| format!("{}:{}", l.file(), l.line()))
            .unwrap_or_else(|| "unknown".into());
        let message = extract_panic_message(info.payload());
        tracing::error!(location = %location, "🔴 PANIC: {message}");
        default(info);
    }));
}

/// Extract human-readable message from panic payload.
fn extract_panic_message(payload: &dyn std::any::Any) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".into()
    }
}

/// Spawn a task. If it panics, send error to the user's chat.
///
/// Uses `JoinHandle.await` which returns `Err(JoinError)` on panic —
/// no `catch_unwind` needed, no `UnwindSafe` bound.
pub fn spawn_guarded<F>(
    bot: Bot,
    chat_id: ChatId,
    thread_id: Option<ThreadId>,
    name: &'static str,
    fut: F,
) -> JoinHandle<()>
where
    F: Future<Output = ()> + Send + 'static,
{
    tokio::spawn(async move {
        let inner = tokio::spawn(fut);
        if let Err(join_err) = inner.await {
            let msg = if join_err.is_panic() {
                let panic = join_err.into_panic();
                extract_panic_message(&*panic)
            } else {
                "task cancelled".into()
            };
            tracing::error!(chat = chat_id.0, task = name, "guarded task failed: {msg}");
            let text = format!("🔴 Ошибка ({}): {}", name, truncate_str(&msg, 200));
            let mut req = bot.send_message(chat_id, text);
            if let Some(tid) = thread_id {
                req = req.message_thread_id(tid);
            }
            let _ = req.await;
        }
    })
}

fn truncate_str(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_str_panic() {
        let p: Box<dyn std::any::Any + Send> = Box::new("boom");
        assert_eq!(extract_panic_message(&*p), "boom");
    }

    #[test]
    fn extract_string_panic() {
        let p: Box<dyn std::any::Any + Send> = Box::new("crash".to_string());
        assert_eq!(extract_panic_message(&*p), "crash");
    }

    #[test]
    fn extract_unknown_panic() {
        let p: Box<dyn std::any::Any + Send> = Box::new(42_i32);
        assert_eq!(extract_panic_message(&*p), "unknown panic");
    }

    #[test]
    fn truncate_utf8_safe() {
        let s = "Привет мир";
        let t = truncate_str(s, 5);
        assert!(t.len() <= 5);
        assert!(s.is_char_boundary(t.len()));
    }

    #[test]
    fn truncate_short() {
        assert_eq!(truncate_str("hi", 100), "hi");
    }

    #[tokio::test]
    async fn spawn_guarded_normal_completes() {
        // Can't test TG sending, but verify no panic on success:
        let bot = Bot::new("fake_token");
        let handle = spawn_guarded(bot, ChatId(0), None, "test", async { /* success */ });
        // Should complete without error:
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn spawn_guarded_catches_panic() {
        // The inner task panics. spawn_guarded should catch it
        // (the outer JoinHandle completes Ok, not Err).
        // TG send will fail (fake token) but that's fine — we verify
        // the wrapper itself doesn't propagate the panic.
        let bot = Bot::new("fake_token");
        let handle = spawn_guarded(bot, ChatId(0), None, "panic-test", async {
            panic!("test panic for guarded");
        });
        // The outer spawn must complete without error:
        let result = handle.await;
        assert!(result.is_ok(), "spawn_guarded must absorb inner panic");
    }

    #[tokio::test]
    async fn spawn_guarded_catches_string_panic() {
        let bot = Bot::new("fake_token");
        let handle = spawn_guarded(
            bot,
            ChatId(0),
            Some(ThreadId(teloxide::types::MessageId(42))),
            "string-panic",
            async {
                let msg = "utf8 паника: тест".to_string();
                panic!("{msg}");
            },
        );
        assert!(handle.await.is_ok());
    }

    #[test]
    fn truncate_str_exact_boundary() {
        // "Привет" = 12 bytes. Truncate at 7 = inside 4th char.
        let s = "Привет";
        let t = truncate_str(s, 7);
        assert_eq!(t, "При"); // 6 bytes — snapped back
    }

    #[test]
    fn truncate_str_zero() {
        assert_eq!(truncate_str("anything", 0), "");
    }
}
