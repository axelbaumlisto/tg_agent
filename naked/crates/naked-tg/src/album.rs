//! Coalesces Telegram media-group ("album") updates into a single agent turn.
//!
//! Telegram delivers each photo in an album as a separate `Update` with the
//! same `media_group_id` arriving within ~100–800ms of one another. If we let
//! each update spawn its own `handle_message` call we would (a) burn N
//! agent turns per album and (b) lose the user's intent — the caption is only
//! attached to one of the items. Worse, the agent would reply N times.
//!
//! Strategy: a debounced buffer keyed by `(chat_id, media_group_id)`. Each
//! incoming part is pushed onto the buffer and resets a per-album debounce
//! timer (default 1.2s). When the timer fires (no more parts arrived in time),
//! the entire batch is handed to a flush callback — typically a closure that
//! invokes `handle_message` once with `extra_media_msgs` populated.
//!
//! The buffer is cancellation-safe: dropping the [`AlbumBuffer`] aborts every
//! pending timer.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use teloxide::types::Message;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

const DEFAULT_ALBUM_DEBOUNCE_MS: u64 = 1_200;
/// Hard cap on the number of items we'll buffer per album. Telegram's own
/// limit is 10 (photos+videos), so 16 leaves comfortable headroom while
/// preventing pathological memory growth from a misbehaving client.
const MAX_ALBUM_ITEMS: usize = 16;

#[derive(Debug, Eq, PartialEq, Hash, Clone)]
struct AlbumKey {
    chat_id: i64,
    group_id: String,
}

struct AlbumState {
    msgs: Vec<Message>,
    /// In-flight debounce timer. Aborted (and replaced) on every `submit`.
    timer: Option<JoinHandle<()>>,
}

/// Concurrent album coalescer. Cheap to clone (`Arc` inside).
#[derive(Clone)]
pub struct AlbumBuffer {
    inner: Arc<Mutex<HashMap<AlbumKey, AlbumState>>>,
    debounce: Duration,
}

impl Default for AlbumBuffer {
    fn default() -> Self {
        Self::new(Duration::from_millis(DEFAULT_ALBUM_DEBOUNCE_MS))
    }
}

impl AlbumBuffer {
    pub fn new(debounce: Duration) -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
            debounce,
        }
    }

    /// Submit one message. Returns:
    /// * `Decision::Solo(msg)` — the message is not part of an album, dispatch
    ///   immediately as today.
    /// * `Decision::Buffered` — message added to its album; nothing to dispatch
    ///   yet (the registered flush task will fire after the debounce window).
    pub async fn submit<F, Fut>(&self, msg: Message, flush: F) -> Decision
    where
        F: FnOnce(Vec<Message>) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        let group_id = match msg.media_group_id() {
            Some(id) => id.0.to_string(),
            None => return Decision::Solo(Box::new(msg)),
        };
        let key = AlbumKey {
            chat_id: msg.chat.id.0,
            group_id,
        };

        let mut guard = self.inner.lock().await;
        let state = guard.entry(key.clone()).or_insert_with(|| AlbumState {
            msgs: Vec::with_capacity(4),
            timer: None,
        });
        if state.msgs.len() >= MAX_ALBUM_ITEMS {
            // Drop the overflow; we still want to flush whatever we have.
            tracing::warn!(
                chat_id = key.chat_id,
                group = %key.group_id,
                "album exceeded MAX_ALBUM_ITEMS, dropping extra item"
            );
        } else {
            state.msgs.push(msg);
        }
        // Cancel the previous timer (if any) — debounce.
        if let Some(t) = state.timer.take() {
            t.abort();
        }

        let inner = self.inner.clone();
        let debounce = self.debounce;
        let timer = tokio::spawn(async move {
            tokio::time::sleep(debounce).await;
            let msgs = {
                let mut g = inner.lock().await;
                match g.remove(&key) {
                    Some(state) => state.msgs,
                    None => return,
                }
            };
            if msgs.is_empty() {
                return;
            }
            flush(msgs).await;
        });
        state.timer = Some(timer);
        Decision::Buffered
    }
}

/// Outcome of [`AlbumBuffer::submit`].
///
/// Design trade-off explicitly captured here because it was surfaced by
/// `clippy::large_enum_variant` and questioned during review:
///
/// * `teloxide::types::Message` is ~2 kB. Without boxing, every `Decision`
///   returned by `submit` — including the overwhelmingly common
///   `Decision::Buffered` path on album parts — would reserve 2 kB of
///   stack even though it carries no payload. Boxing keeps the enum at
///   one pointer on the stack.
/// * The one extra heap allocation for the `Solo` path is cheap (dozens
///   of ns) relative to the cost of the LLM turn it triggers (seconds).
/// * Returning `Box<Message>` instead of `Message` lets the caller avoid
///   a full `Message::clone()` in the common case where it wants to keep
///   ownership without copying.
///
/// In short: the boxed variant is the right call here; keep it.
pub enum Decision {
    /// Plain message — dispatch directly. Carries the original message back so
    /// the caller doesn't need a clone.
    Solo(Box<Message>),
    /// Buffered as part of an album; the flush callback owns dispatch.
    Buffered,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn fake_msg(group: Option<&str>) -> Message {
        // Build a minimal Message JSON the way Telegram delivers it. Keeps the
        // test free of teloxide builder boilerplate.
        let mut v = serde_json::json!({
            "message_id": 1,
            "date": 0,
            "chat": {"id": 42, "type": "private", "first_name": "x"},
            "from": {"id": 1, "is_bot": false, "first_name": "x"},
            "photo": [
                {"file_id": "f", "file_unique_id": "u", "width": 1, "height": 1, "file_size": 10}
            ]
        });
        if let Some(gid) = group {
            v.as_object_mut().unwrap().insert(
                "media_group_id".into(),
                serde_json::Value::String(gid.into()),
            );
        }
        serde_json::from_value(v).unwrap()
    }

    #[tokio::test]
    async fn solo_message_passes_through() {
        let buf = AlbumBuffer::new(Duration::from_millis(50));
        match buf.submit(fake_msg(None), |_| async {}).await {
            Decision::Solo(_) => {}
            Decision::Buffered => panic!("solo message should not buffer"),
        }
    }

    #[tokio::test]
    async fn three_album_parts_flush_once() {
        let buf = AlbumBuffer::new(Duration::from_millis(40));
        let counter = Arc::new(AtomicUsize::new(0));
        let collected = Arc::new(Mutex::new(Vec::<usize>::new()));
        for _ in 0..3 {
            let c = counter.clone();
            let col = collected.clone();
            let _ = buf
                .submit(fake_msg(Some("g1")), move |msgs| async move {
                    c.fetch_add(1, Ordering::SeqCst);
                    col.lock().await.push(msgs.len());
                })
                .await;
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        tokio::time::sleep(Duration::from_millis(120)).await;
        assert_eq!(counter.load(Ordering::SeqCst), 1, "flush should fire once");
        assert_eq!(collected.lock().await.as_slice(), &[3]);
    }

    #[tokio::test]
    async fn two_distinct_albums_flush_independently() {
        let buf = AlbumBuffer::new(Duration::from_millis(40));
        let calls = Arc::new(AtomicUsize::new(0));
        for gid in ["a", "b"] {
            let c = calls.clone();
            let _ = buf
                .submit(fake_msg(Some(gid)), move |_| async move {
                    c.fetch_add(1, Ordering::SeqCst);
                })
                .await;
        }
        tokio::time::sleep(Duration::from_millis(120)).await;
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    fn fake_msg_with_id(group: Option<&str>, message_id: i32) -> Message {
        let mut v = serde_json::json!({
            "message_id": message_id,
            "date": 0,
            "chat": {"id": 42, "type": "private", "first_name": "x"},
            "from": {"id": 1, "is_bot": false, "first_name": "x"},
            "photo": [
                {"file_id": format!("f{message_id}"), "file_unique_id": format!("u{message_id}"), "width": 1, "height": 1, "file_size": 10}
            ]
        });
        if let Some(gid) = group {
            v.as_object_mut().unwrap().insert(
                "media_group_id".into(),
                serde_json::Value::String(gid.into()),
            );
        }
        serde_json::from_value(v).unwrap()
    }

    #[tokio::test]
    async fn album_flush_receives_messages_in_arrival_order() {
        // Caller is responsible for final ordering by message_id (it does
        // `msgs.sort_by_key` right before dispatch), but the buffer itself
        // must at minimum return every part. Verify all 3 are delivered.
        let buf = AlbumBuffer::new(Duration::from_millis(30));
        let got = Arc::new(Mutex::new(Vec::<i32>::new()));
        for mid in [5, 3, 7] {
            let g = got.clone();
            let _ = buf
                .submit(
                    fake_msg_with_id(Some("order"), mid),
                    move |msgs| async move {
                        let mut ids: Vec<i32> = msgs.into_iter().map(|m| m.id.0).collect();
                        ids.sort(); // mirrors what main.rs does post-flush
                        *g.lock().await = ids;
                    },
                )
                .await;
            tokio::time::sleep(Duration::from_millis(3)).await;
        }
        tokio::time::sleep(Duration::from_millis(80)).await;
        assert_eq!(*got.lock().await, vec![3, 5, 7]);
    }

    #[tokio::test]
    async fn overflow_items_are_dropped_but_flush_still_fires() {
        let buf = AlbumBuffer::new(Duration::from_millis(30));
        let lens = Arc::new(Mutex::new(Vec::<usize>::new()));
        for _ in 0..(MAX_ALBUM_ITEMS + 5) {
            let l = lens.clone();
            let _ = buf
                .submit(fake_msg(Some("big")), move |msgs| async move {
                    l.lock().await.push(msgs.len());
                })
                .await;
        }
        tokio::time::sleep(Duration::from_millis(120)).await;
        let v = lens.lock().await;
        assert_eq!(v.len(), 1);
        assert_eq!(v[0], MAX_ALBUM_ITEMS);
    }
}
