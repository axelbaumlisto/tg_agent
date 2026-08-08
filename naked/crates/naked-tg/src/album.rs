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

use teloxide::types::{MediaKind, Message, MessageKind};
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
    /// Number of submitted items omitted after `msgs` reached the cap.
    dropped_items: usize,
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
            dropped_items: 0,
            timer: None,
        });
        if state.msgs.len() >= max_items {
            state.dropped_items += 1;
            // Drop the overflow; we still want to flush whatever we have, with
            // the omission count attached to the payload at flush time.
            tracing::warn!(
                ?key,
                max_items,
                dropped_items = state.dropped_items,
                "{label} exceeded cap, dropping extra item"
            );
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
            let mut msgs = {
                let mut g = inner.lock().await;
                let mut state = match g.remove(&key_for_timer) {
                    Some(state) => state,
                    None => return,
                };
                attach_overflow_notice(&mut state.msgs, state.dropped_items, max_items);
                state.msgs
            };
            if msgs.is_empty() {
                return;
            }
            flush(std::mem::take(&mut msgs)).await;
        });
        state.timer = Some(timer);
        Decision::Buffered
    }
}

fn attach_overflow_notice(msgs: &mut [Message], dropped_items: usize, max_items: usize) {
    if msgs.is_empty() || dropped_items == 0 {
        return;
    }
    let notice = format!(
        "⚠️ Пропущено {dropped_items} {} входящего сообщения: превышен лимит {max_items}.",
        dropped_item_word(dropped_items)
    );
    let Some(first_idx) = msgs
        .iter()
        .enumerate()
        .min_by_key(|(_, msg)| msg.id.0)
        .map(|(idx, _)| idx)
    else {
        return;
    };
    if !append_notice_to_message(&mut msgs[first_idx], &notice) {
        tracing::warn!(
            dropped_items,
            "failed to attach inbound overflow notice: no text/caption field"
        );
    }
}

fn dropped_item_word(n: usize) -> &'static str {
    let rem100 = n % 100;
    let rem10 = n % 10;
    if (11..=14).contains(&rem100) {
        "частей"
    } else {
        match rem10 {
            1 => "часть",
            2..=4 => "части",
            _ => "частей",
        }
    }
}

fn append_notice_to_message(msg: &mut Message, notice: &str) -> bool {
    let MessageKind::Common(common) = &mut msg.kind else {
        return false;
    };
    match &mut common.media_kind {
        MediaKind::Text(media) => {
            append_notice_to_text(&mut media.text, notice);
            true
        }
        MediaKind::Animation(media) => append_notice_to_caption(&mut media.caption, notice),
        MediaKind::Audio(media) => append_notice_to_caption(&mut media.caption, notice),
        MediaKind::Document(media) => append_notice_to_caption(&mut media.caption, notice),
        MediaKind::Photo(media) => append_notice_to_caption(&mut media.caption, notice),
        MediaKind::Video(media) => append_notice_to_caption(&mut media.caption, notice),
        MediaKind::Voice(media) => append_notice_to_caption(&mut media.caption, notice),
        _ => false,
    }
}

fn append_notice_to_caption(caption: &mut Option<String>, notice: &str) -> bool {
    match caption {
        Some(existing) => append_notice_to_text(existing, notice),
        None => *caption = Some(notice.to_string()),
    }
    true
}

fn append_notice_to_text(existing: &mut String, notice: &str) {
    if existing.trim().is_empty() {
        *existing = notice.to_string();
    } else {
        existing.push_str("\n\n");
        existing.push_str(notice);
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
        fake_text_msg_with_entities(
            message_id,
            chat_id,
            thread_id,
            sender_id,
            text,
            reply_to_message_id,
            None,
        )
    }

    fn fake_text_msg_with_entities(
        message_id: i32,
        chat_id: i64,
        thread_id: Option<i32>,
        sender_id: u64,
        text: &str,
        reply_to_message_id: Option<i32>,
        entities: Option<serde_json::Value>,
    ) -> Message {
        let mut v = serde_json::json!({
            "message_id": message_id,
            "date": 0,
            "chat": {"id": chat_id, "type": "supergroup", "title": "g"},
            "from": {"id": sender_id, "is_bot": false, "first_name": format!("u{sender_id}")},
            "text": text,
        });
        if let Some(entities) = entities {
            v.as_object_mut()
                .unwrap()
                .insert("entities".into(), entities);
        }
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

    fn synthesize_runtime_text_burst(mut msgs: Vec<Message>) -> Message {
        msgs.sort_by_key(|m| m.id.0);
        let mut primary = msgs.remove(0);
        let joined = std::iter::once(&primary)
            .chain(msgs.iter())
            .filter_map(|m| m.text())
            .map(str::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        let mut value = serde_json::to_value(&primary).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("text".to_string(), serde_json::Value::String(joined));
        primary = serde_json::from_value(value).unwrap();
        primary
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
    async fn text_burst_overflow_payload_reports_dropped_count_and_under_cap_is_byte_identical() {
        let under_cap = InboundCoalescer::new(Duration::from_millis(20));
        let under_cap_payloads = Arc::new(Mutex::new(Vec::<String>::new()));
        let under_cap_key = text_key(42, None, 100, None);

        for (mid, text) in [(1, "один"), (2, "два"), (3, "три")] {
            let payloads = under_cap_payloads.clone();
            let _ = under_cap
                .submit_text(
                    fake_text_msg(mid, 42, None, 100, text, None),
                    under_cap_key.clone(),
                    Duration::from_millis(20),
                    move |msgs| async move {
                        payloads.lock().await.push(joined_text(msgs));
                    },
                )
                .await;
        }
        tokio::time::sleep(Duration::from_millis(70)).await;
        let under_cap_payloads = under_cap_payloads.lock().await;
        assert_eq!(
            under_cap_payloads.as_slice(),
            &["один\nдва\nтри"],
            "under-cap text bursts must stay byte-identical"
        );
        assert!(
            !under_cap_payloads[0].contains("Пропущено"),
            "normal under-cap bursts must not get warning noise"
        );
        drop(under_cap_payloads);

        let overflow = InboundCoalescer::new(Duration::from_millis(20));
        let overflow_payloads = Arc::new(Mutex::new(Vec::<String>::new()));
        let overflow_key = text_key(42, None, 100, None);

        for idx in 0..(MAX_ALBUM_ITEMS + 3) {
            let payloads = overflow_payloads.clone();
            let _ = overflow
                .submit_text(
                    fake_text_msg(idx as i32, 42, None, 100, &format!("part-{idx}"), None),
                    overflow_key.clone(),
                    Duration::from_millis(20),
                    move |msgs| async move {
                        payloads.lock().await.push(joined_text(msgs));
                    },
                )
                .await;
        }
        tokio::time::sleep(Duration::from_millis(70)).await;
        let overflow_payloads = overflow_payloads.lock().await;
        assert_eq!(overflow_payloads.len(), 1);
        let payload = &overflow_payloads[0];
        assert!(
            payload.contains("⚠️ Пропущено 3 части входящего сообщения: превышен лимит 16."),
            "overflow payload must report the dropped count; got {payload:?}"
        );
        assert!(payload.contains("part-15"));
        assert!(!payload.contains("part-16"));
    }

    #[tokio::test]
    async fn group_mention_text_burst_overflow_preserves_addressing_and_reports_dropped_count() {
        let overflow = InboundCoalescer::new(Duration::from_millis(20));
        let flushed = Arc::new(Mutex::new(Vec::<Message>::new()));
        let key = text_key(-10042, Some(9), 100, None);

        for idx in 0..(MAX_ALBUM_ITEMS + 3) {
            let flushed = flushed.clone();
            let text = if idx == 0 {
                "@zGsR_bot part-0".to_string()
            } else {
                format!("part-{idx}")
            };
            let entities = (idx == 0)
                .then(|| serde_json::json!([{ "type": "mention", "offset": 0, "length": 9 }]));
            let _ = overflow
                .submit_text(
                    fake_text_msg_with_entities(
                        idx as i32,
                        -10042,
                        Some(9),
                        100,
                        &text,
                        None,
                        entities,
                    ),
                    key.clone(),
                    Duration::from_millis(20),
                    move |msgs| async move {
                        flushed
                            .lock()
                            .await
                            .push(synthesize_runtime_text_burst(msgs));
                    },
                )
                .await;
        }

        tokio::time::sleep(Duration::from_millis(70)).await;
        let flushed = flushed.lock().await;
        assert_eq!(flushed.len(), 1);
        let msg = &flushed[0];
        let text = msg.text().unwrap_or_default();
        assert!(
            text.contains("⚠️ Пропущено 3 части входящего сообщения: превышен лимит 16."),
            "overflow payload must report dropped count; got {text:?}"
        );
        let identity = naked_tg::bot_identity::BotIdentity {
            id: 777,
            username: "zGsR_bot".to_string(),
        };
        let mention_span = msg
            .parse_entities()
            .and_then(|entities| entities.first().map(|ent| ent.text().to_string()))
            .unwrap_or_else(|| "<none>".to_string());
        assert!(
            naked_tg::bot_identity::is_addressed_to_bot(msg, &identity),
            "flushed overflow text burst must remain addressed; mention_span={mention_span:?}; text={text:?}"
        );
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
