//! Coalesces Telegram inbound bursts into a single agent turn.
//!
//! Telegram delivers each photo in a media album as a separate `Update` with
//! the same `media_group_id`, and some Telegram clients split long text into
//! several plain text messages without a `media_group_id`. If every update
//! spawned its own `handle_message` call we would burn N agent turns and reply
//! N times.
//!
//! Strategy: one debounced buffer keyed by the runtime-selected coalescing key.
//! Media albums are keyed by `(chat_id, media_group_id)`. Text bursts are keyed
//! by `(chat_id, thread_id, sender_id, reply_to_message_id)` so messages from
//! different users or reply targets never merge. The runtime owns all
//! eligibility decisions (addressing, active-turn boundary, feature gate); this
//! module only owns cancellation-safe buffering and debounce timers.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use teloxide::types::Message;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

const DEFAULT_ALBUM_DEBOUNCE_MS: u64 = 1_200;
/// Hard cap on the number of items we'll buffer per coalesced burst. Telegram's
/// own media-album limit is 10 (photos+videos), so 16 leaves comfortable
/// headroom while preventing pathological memory growth from a misbehaving
/// client or an over-eager text splitter.
pub(crate) const MAX_ALBUM_ITEMS: usize = 16;

#[derive(Debug, Eq, PartialEq, Hash, Clone)]
struct AlbumKey {
    chat_id: i64,
    group_id: String,
}

/// Runtime-computed key for plain-text burst coalescing.
#[derive(Debug, Eq, PartialEq, Hash, Clone)]
pub(crate) struct TextBurstKey {
    pub(crate) chat_id: i64,
    pub(crate) thread_id: Option<i32>,
    pub(crate) sender_id: u64,
    pub(crate) reply_to_message_id: Option<i32>,
}

#[derive(Debug, Eq, PartialEq, Hash, Clone)]
enum InboundKey {
    Album(AlbumKey),
    Text(TextBurstKey),
}

struct InboundState {
    msgs: Vec<Message>,
    /// In-flight debounce timer. Aborted (and replaced) on every `submit`.
    timer: Option<JoinHandle<()>>,
}

/// Concurrent inbound coalescer. Cheap to clone (`Arc` inside).
#[derive(Clone)]
pub(crate) struct InboundCoalescer {
    inner: Arc<Mutex<HashMap<InboundKey, InboundState>>>,
    album_debounce: Duration,
}

impl Default for InboundCoalescer {
    fn default() -> Self {
        Self::new(Duration::from_millis(DEFAULT_ALBUM_DEBOUNCE_MS))
    }
}

impl InboundCoalescer {
    pub(crate) fn new(album_debounce: Duration) -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
            album_debounce,
        }
    }

    /// Submit one message for media-album coalescing. Returns:
    /// * `Decision::Solo(msg)` — the message is not part of an album, dispatch
    ///   immediately as today.
    /// * `Decision::Buffered` — message added to its album; nothing to dispatch
    ///   yet (the registered flush task will fire after the debounce window).
    pub(crate) async fn submit_album<F, Fut>(&self, msg: Message, flush: F) -> Decision
    where
        F: FnOnce(Vec<Message>) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        let group_id = match msg.media_group_id() {
            Some(id) => id.0.to_string(),
            None => return Decision::Solo(Box::new(msg)),
        };
        let key = InboundKey::Album(AlbumKey {
            chat_id: msg.chat.id.0,
            group_id,
        });
        self.submit_keyed(
            msg,
            key,
            self.album_debounce,
            MAX_ALBUM_ITEMS,
            "album",
            flush,
        )
        .await
    }

    /// Submit one runtime-approved plain-text message for burst coalescing.
    /// A zero debounce disables text coalescing and returns `Decision::Solo`.
    pub(crate) async fn submit_text<F, Fut>(
        &self,
        msg: Message,
        key: TextBurstKey,
        debounce: Duration,
        flush: F,
    ) -> Decision
    where
        F: FnOnce(Vec<Message>) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        self.submit_keyed(
            msg,
            InboundKey::Text(key),
            debounce,
            MAX_ALBUM_ITEMS,
            "text burst",
            flush,
        )
        .await
    }

    async fn submit_keyed<F, Fut>(
        &self,
        msg: Message,
        key: InboundKey,
        debounce: Duration,
        max_items: usize,
        label: &'static str,
        flush: F,
    ) -> Decision
    where
        F: FnOnce(Vec<Message>) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        if debounce.is_zero() {
            return Decision::Solo(Box::new(msg));
        }

        let mut guard = self.inner.lock().await;
        let state = guard.entry(key.clone()).or_insert_with(|| InboundState {
            msgs: Vec::with_capacity(4),
            timer: None,
        });
        if state.msgs.len() >= max_items {
            // Drop the overflow; we still want to flush whatever we have.
            tracing::warn!(?key, max_items, "{label} exceeded cap, dropping extra item");
        } else {
            state.msgs.push(msg);
        }
        // Cancel the previous timer (if any) — debounce.
        if let Some(t) = state.timer.take() {
            t.abort();
        }

        let inner = self.inner.clone();
        let key_for_timer = key.clone();
        let timer = tokio::spawn(async move {
            tokio::time::sleep(debounce).await;
            let msgs = {
                let mut g = inner.lock().await;
                match g.remove(&key_for_timer) {
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

/// Outcome of [`InboundCoalescer`] submission.
///
/// Design trade-off explicitly captured here because it was surfaced by
/// `clippy::large_enum_variant` and questioned during review:
///
/// * `teloxide::types::Message` is ~2 kB. Without boxing, every `Decision`
///   returned by `submit` — including the overwhelmingly common
///   `Decision::Buffered` path on burst parts — would reserve 2 kB of
///   stack even though it carries no payload. Boxing keeps the enum at
///   one pointer on the stack.
/// * The one extra heap allocation for the `Solo` path is cheap (dozens
///   of ns) relative to the cost of the LLM turn it triggers (seconds).
/// * Returning `Box<Message>` instead of `Message` lets the caller avoid
///   a full `Message::clone()` in the common case where it wants to keep
///   ownership without copying.
///
/// In short: the boxed variant is the right call here; keep it.
pub(crate) enum Decision {
    /// Plain message — dispatch directly. Carries the original message back so
    /// the caller doesn't need a clone.
    Solo(Box<Message>),
    /// Buffered as part of a burst; the flush callback owns dispatch.
    Buffered,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn fake_album_msg(group: Option<&str>) -> Message {
        fake_album_msg_with_id(group, 1)
    }

    fn fake_album_msg_with_id(group: Option<&str>, message_id: i32) -> Message {
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

    fn text_key(
        chat_id: i64,
        thread_id: Option<i32>,
        sender_id: u64,
        reply_to_message_id: Option<i32>,
    ) -> TextBurstKey {
        TextBurstKey {
            chat_id,
            thread_id,
            sender_id,
            reply_to_message_id,
        }
    }

    fn fake_text_msg(
        message_id: i32,
        chat_id: i64,
        thread_id: Option<i32>,
        sender_id: u64,
        text: &str,
        reply_to_message_id: Option<i32>,
    ) -> Message {
        let mut v = serde_json::json!({
            "message_id": message_id,
            "date": 0,
            "chat": {"id": chat_id, "type": "supergroup", "title": "g"},
            "from": {"id": sender_id, "is_bot": false, "first_name": format!("u{sender_id}")},
            "text": text,
        });
        if let Some(tid) = thread_id {
            v.as_object_mut().unwrap().insert(
                "message_thread_id".into(),
                serde_json::Value::Number(tid.into()),
            );
            v.as_object_mut()
                .unwrap()
                .insert("is_topic_message".into(), serde_json::Value::Bool(true));
        }
        if let Some(reply_id) = reply_to_message_id {
            v.as_object_mut().unwrap().insert(
                "reply_to_message".into(),
                serde_json::json!({
                    "message_id": reply_id,
                    "date": 0,
                    "chat": {"id": chat_id, "type": "supergroup", "title": "g"},
                    "from": {"id": 999, "is_bot": false, "first_name": "r"},
                    "text": "reply target"
                }),
            );
        }
        serde_json::from_value(v).unwrap()
    }

    fn joined_text(mut msgs: Vec<Message>) -> String {
        msgs.sort_by_key(|m| m.id.0);
        msgs.into_iter()
            .filter_map(|m| m.text().map(str::to_string))
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[tokio::test]
    async fn solo_message_passes_through() {
        let buf = InboundCoalescer::new(Duration::from_millis(50));
        match buf.submit_album(fake_album_msg(None), |_| async {}).await {
            Decision::Solo(_) => {}
            Decision::Buffered => panic!("solo message should not buffer"),
        }
    }

    #[tokio::test]
    async fn three_album_parts_flush_once() {
        let buf = InboundCoalescer::new(Duration::from_millis(40));
        let counter = Arc::new(AtomicUsize::new(0));
        let collected = Arc::new(Mutex::new(Vec::<usize>::new()));
        for _ in 0..3 {
            let c = counter.clone();
            let col = collected.clone();
            let _ = buf
                .submit_album(fake_album_msg(Some("g1")), move |msgs| async move {
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
        let buf = InboundCoalescer::new(Duration::from_millis(40));
        let calls = Arc::new(AtomicUsize::new(0));
        for gid in ["a", "b"] {
            let c = calls.clone();
            let _ = buf
                .submit_album(fake_album_msg(Some(gid)), move |_| async move {
                    c.fetch_add(1, Ordering::SeqCst);
                })
                .await;
        }
        tokio::time::sleep(Duration::from_millis(120)).await;
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn album_flush_receives_messages_in_arrival_order() {
        // Caller is responsible for final ordering by message_id (it does
        // `msgs.sort_by_key` right before dispatch), but the buffer itself
        // must at minimum return every part. Verify all 3 are delivered.
        let buf = InboundCoalescer::new(Duration::from_millis(30));
        let got = Arc::new(Mutex::new(Vec::<i32>::new()));
        for mid in [5, 3, 7] {
            let g = got.clone();
            let _ = buf
                .submit_album(
                    fake_album_msg_with_id(Some("order"), mid),
                    move |msgs| async move {
                        let mut ids: Vec<i32> = msgs.into_iter().map(|m| m.id.0).collect();
                        ids.sort(); // mirrors what runtime does post-flush
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
        let buf = InboundCoalescer::new(Duration::from_millis(30));
        let lens = Arc::new(Mutex::new(Vec::<usize>::new()));
        for _ in 0..(MAX_ALBUM_ITEMS + 5) {
            let l = lens.clone();
            let _ = buf
                .submit_album(fake_album_msg(Some("big")), move |msgs| async move {
                    l.lock().await.push(msgs.len());
                })
                .await;
        }
        tokio::time::sleep(Duration::from_millis(120)).await;
        let v = lens.lock().await;
        assert_eq!(v.len(), 1);
        assert_eq!(v[0], MAX_ALBUM_ITEMS);
    }

    #[tokio::test]
    async fn plain_text_burst_flushes_once_in_message_id_order() {
        let buf = InboundCoalescer::new(Duration::from_millis(30));
        let calls = Arc::new(AtomicUsize::new(0));
        let payloads = Arc::new(Mutex::new(Vec::<String>::new()));
        let key = text_key(42, Some(7), 100, None);

        for (mid, text) in [(30, "third"), (10, "first"), (20, "second")] {
            let calls = calls.clone();
            let payloads = payloads.clone();
            let _ = buf
                .submit_text(
                    fake_text_msg(mid, 42, Some(7), 100, text, None),
                    key.clone(),
                    Duration::from_millis(30),
                    move |msgs| async move {
                        calls.fetch_add(1, Ordering::SeqCst);
                        payloads.lock().await.push(joined_text(msgs));
                    },
                )
                .await;
            tokio::time::sleep(Duration::from_millis(3)).await;
        }

        tokio::time::sleep(Duration::from_millis(90)).await;
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(payloads.lock().await.as_slice(), &["first\nsecond\nthird"]);
    }

    #[tokio::test]
    async fn text_message_after_window_starts_new_flush() {
        let buf = InboundCoalescer::new(Duration::from_millis(25));
        let payloads = Arc::new(Mutex::new(Vec::<String>::new()));
        let key = text_key(42, None, 100, None);

        for (mid, text) in [(1, "one"), (2, "two"), (3, "three")] {
            let payloads = payloads.clone();
            let _ = buf
                .submit_text(
                    fake_text_msg(mid, 42, None, 100, text, None),
                    key.clone(),
                    Duration::from_millis(25),
                    move |msgs| async move {
                        payloads.lock().await.push(joined_text(msgs));
                    },
                )
                .await;
        }
        tokio::time::sleep(Duration::from_millis(80)).await;

        let payloads2 = payloads.clone();
        let _ = buf
            .submit_text(
                fake_text_msg(4, 42, None, 100, "four", None),
                key,
                Duration::from_millis(25),
                move |msgs| async move {
                    payloads2.lock().await.push(joined_text(msgs));
                },
            )
            .await;
        tokio::time::sleep(Duration::from_millis(80)).await;

        assert_eq!(
            payloads.lock().await.as_slice(),
            &["one\ntwo\nthree", "four"]
        );
    }

    #[tokio::test]
    async fn different_senders_in_same_chat_thread_do_not_coalesce() {
        let buf = InboundCoalescer::new(Duration::from_millis(30));
        let payloads = Arc::new(Mutex::new(Vec::<String>::new()));

        for (sender, text) in [(100, "alice"), (200, "bob")] {
            let payloads = payloads.clone();
            let key = text_key(42, Some(7), sender, None);
            let _ = buf
                .submit_text(
                    fake_text_msg(sender as i32, 42, Some(7), sender, text, None),
                    key,
                    Duration::from_millis(30),
                    move |msgs| async move {
                        payloads.lock().await.push(joined_text(msgs));
                    },
                )
                .await;
        }

        tokio::time::sleep(Duration::from_millis(90)).await;
        let mut got = payloads.lock().await.clone();
        got.sort();
        assert_eq!(got, vec!["alice", "bob"]);
    }

    #[tokio::test]
    async fn disabled_text_coalescing_returns_solo_immediately() {
        let buf = InboundCoalescer::new(Duration::from_millis(30));
        let key = text_key(42, None, 100, None);
        match buf
            .submit_text(
                fake_text_msg(1, 42, None, 100, "solo", None),
                key,
                Duration::ZERO,
                |_| async { panic!("disabled text coalescing must not flush") },
            )
            .await
        {
            Decision::Solo(msg) => assert_eq!(msg.text(), Some("solo")),
            Decision::Buffered => panic!("disabled text coalescing should not buffer"),
        }
    }

    #[tokio::test]
    async fn different_reply_targets_in_same_chat_thread_do_not_coalesce() {
        let buf = InboundCoalescer::new(Duration::from_millis(30));
        let payloads = Arc::new(Mutex::new(Vec::<String>::new()));

        for (reply, text) in [(Some(10), "reply-a"), (Some(20), "reply-b")] {
            let payloads = payloads.clone();
            let key = text_key(42, Some(7), 100, reply);
            let _ = buf
                .submit_text(
                    fake_text_msg(reply.unwrap(), 42, Some(7), 100, text, reply),
                    key,
                    Duration::from_millis(30),
                    move |msgs| async move {
                        payloads.lock().await.push(joined_text(msgs));
                    },
                )
                .await;
        }

        tokio::time::sleep(Duration::from_millis(90)).await;
        let mut got = payloads.lock().await.clone();
        got.sort();
        assert_eq!(got, vec!["reply-a", "reply-b"]);
    }
}
