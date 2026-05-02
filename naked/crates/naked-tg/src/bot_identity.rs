//! Bot self-identity (id + username) and the pure
//! [`is_addressed_to_bot`] gate used in groups.
//!
//! Why a dedicated module:
//!
//! - **Privacy mode is OFF** for the bot (`can_read_all_group_messages =
//!   true` from `getMe`), so the Telegram server delivers every message in
//!   any group the bot has joined — not just `@mentions` and replies.
//!   That's the right default for "let the bot read everything for context"
//!   but a terrible default for "respond to everything", since the bot
//!   would chime in on every random reply between humans in the group.
//! - In private chats the bot is, by definition, the addressee — every
//!   message is meant for it.
//! - In group chats we treat a message as "addressed to the bot" when at
//!   least one of these is true:
//!     1. It's a slash command (`/...`), with the optional
//!        `@<bot_username>` suffix matching us. A `/cmd@other_bot`
//!        command is explicitly **not** ours.
//!     2. The text contains an `@<bot_username>` `Mention` entity.
//!     3. The text contains a `TextMention` entity whose `user.id`
//!        matches the bot id (used when the user picked the bot from
//!        the @-completion menu but the display name doesn't have a
//!        public username).
//!     4. It's a reply to one of the bot's **own** messages (i.e.
//!        `reply_to_message.from.is_bot && from.id == our id`). This
//!        is the natural "follow-up" pattern — tap Reply on the bot's
//!        answer, type a clarification without retyping `@bot`. Note
//!        that reply to a **user's** message (even one that happened
//!        to contain an `@<bot>` mention in its text) does NOT wake
//!        the agent: the user's `reply_to_message.from.is_bot` is
//!        false and no separate mention path matches, so the bot
//!        correctly stays silent when humans quote each other.
//!
//! Everything is a pure function over `&Message` + `&BotIdentity` so the
//! gate is fully unit-tested without spinning up Telegram.

use teloxide::types::{Message, MessageEntityKind};

/// Self-identity of the running bot, captured once at startup via
/// `getMe`. Cheap to clone (uses `Arc<str>` semantics via `String` —
/// expected to be cloned at most a few times in practice).
#[derive(Debug, Clone)]
pub struct BotIdentity {
    /// Telegram user id of the bot. Matches `MessageEntityKind::TextMention {
    /// user: User { id, .. } }` and the `from` of any reply to one of our
    /// own messages.
    pub id: u64,
    /// Bot @-username **without** the leading `@`. Compared
    /// case-insensitively per Telegram's username rules
    /// (`@MyBot` == `@mybot`).
    pub username: String,
}

impl BotIdentity {
    /// Lower-cased username for comparison.
    fn username_lc(&self) -> String {
        self.username.to_ascii_lowercase()
    }
}

/// Strip the `@<bot_username>` suffix from a slash command's first
/// token if it targets us. Returns the original `text` unchanged if
/// the command targets a different bot or has no `@bot` suffix.
///
/// Used to canonicalise `/start@MyBot args` → `/start args` before
/// dispatching to `handle_command`, while letting `/start@OtherBot`
/// reach the gate as "not addressed to us" and be silently ignored.
pub fn strip_bot_command_suffix(text: &str, identity: &BotIdentity) -> Option<String> {
    let trimmed = text.trim_start();
    if !trimmed.starts_with('/') {
        return None;
    }
    // Split off the first token (everything up to the first whitespace).
    let (head, tail) = match trimmed.find(char::is_whitespace) {
        Some(i) => (&trimmed[..i], &trimmed[i..]),
        None => (trimmed, ""),
    };
    let at_pos = head.find('@')?;
    let cmd = &head[..at_pos];
    let bot = &head[at_pos + 1..];
    if bot.eq_ignore_ascii_case(&identity.username) {
        Some(format!("{cmd}{tail}"))
    } else {
        None
    }
}

// ─────────────────────────── Session-handover detection ───────────────────────
//
// When a session rolls (either from a natural A2 budget cap or the
// planned durable-state restart), the coordinator seeds the new
// session with a quoted resume block of the shape:
//
//     > @<bot_username> [your previous message]:
//     > …old bot reply, possibly containing @-mentions…
//
// followed by the real user prompt. The addressing gate must not
// re-trigger on the mention that's structurally INSIDE the handover
// quote, and downstream code (`classify_handover` consumers) wants
// the "effective" user text with the quote stripped.

/// Classification of a raw message body with respect to session-handover
/// quotes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HandoverClassification {
    /// True if the message starts with a `> @<bot> [your previous message]:`
    /// quote block (consecutive `>`-prefixed lines at the top).
    pub is_handover_resume: bool,
    /// The message body with the leading quote block removed, or
    /// `raw` verbatim if `is_handover_resume == false`.
    pub effective_user_text: String,
}

/// Detect the session-handover resume pattern and strip its quote
/// block so the addressing gate can consider the real user prompt
/// in isolation.
///
/// We recognise the pattern by two signals:
///
/// 1. The FIRST line starts with `> @` and contains the literal
///    `[your previous message]:` marker.
/// 2. One or more leading `>`-prefixed lines follow (the quoted
///    previous turn). Everything up to and including the last
///    consecutive leading quote line is treated as handover context.
///
/// Any other shape returns `is_handover_resume: false` and the
/// original text untouched.
pub fn classify_handover(raw: &str) -> HandoverClassification {
    let lines: Vec<&str> = raw.lines().collect();
    let first_line_is_handover = lines
        .first()
        .map(|l| l.trim_start().starts_with("> @") && l.contains("[your previous message]:"))
        .unwrap_or(false);
    if !first_line_is_handover {
        return HandoverClassification {
            is_handover_resume: false,
            effective_user_text: raw.to_string(),
        };
    }
    let leading_quote = lines
        .iter()
        .take_while(|l| l.trim_start().starts_with('>'))
        .count();
    let tail = lines[leading_quote..].join("\n");
    HandoverClassification {
        is_handover_resume: true,
        // Preserve any trailing newline characters the raw text had so
        // we don't accidentally trim content below the quote block.
        effective_user_text: tail.trim_start_matches('\n').to_string(),
    }
}

/// True if `text`'s first token is `/cmd@<other_bot>` (i.e. a slash
/// command explicitly targeted at a *different* bot in the same group).
pub fn is_command_for_other_bot(text: &str, identity: &BotIdentity) -> bool {
    let trimmed = text.trim_start();
    if !trimmed.starts_with('/') {
        return false;
    }
    let head = match trimmed.find(char::is_whitespace) {
        Some(i) => &trimmed[..i],
        None => trimmed,
    };
    let at_pos = match head.find('@') {
        Some(p) => p,
        None => return false,
    };
    let bot = &head[at_pos + 1..];
    !bot.is_empty() && !bot.eq_ignore_ascii_case(&identity.username)
}

/// Pure decision for "should this group message wake the agent?".
///
/// Returns `true` for **every** private chat (a private message is
/// definitionally addressed to its recipient).
///
/// In a group, the rules are:
/// - `/cmd` (no `@bot` suffix) → addressed (commands are global in our
///   bot — keeps backwards compat with private chats).
/// - `/cmd@<our_username>` → addressed.
/// - `/cmd@<other_bot>` → NOT addressed.
/// - Mention entity matching `@<our_username>` (case-insensitive) →
///   addressed.
/// - `TextMention { user.id == our_id }` → addressed.
/// - Reply to one of our **own** messages → addressed. This is the
///   follow-up pattern: tap Reply on the bot's answer and just ask
///   the next question without retyping `@bot`.
/// - Reply to a **user's** message (even one that contains a bot
///   mention in its text) → NOT addressed by that reply alone. The
///   check is `reply_to_message.from.is_bot && from.id == our_id`,
///   which only matches replies whose target was authored by us. So
///   when User B quotes User A's `@bot posmortem…` with their own
///   comment aimed at User A, the bot correctly stays silent.
/// - Otherwise → not addressed (the bot reads the message but stays
///   silent).
pub fn is_addressed_to_bot(msg: &Message, identity: &BotIdentity) -> bool {
    // Private chats: every message is for us.
    use teloxide::types::ChatKind;
    if !matches!(msg.chat.kind, ChatKind::Public(_)) {
        return true;
    }

    // Reply to one of our own messages — the natural "follow-up"
    // pattern. We specifically gate on `from.is_bot && from.id == our
    // id` so a reply to a **user** whose message contains an `@bot`
    // mention does NOT wake us (that's a human quoting another human
    // for context, not an address to us).
    if let Some(reply) = msg.reply_to_message()
        && let Some(from) = reply.from.as_ref()
        && from.is_bot
        && from.id.0 == identity.id
    {
        return true;
    }

    let text = msg.text().or_else(|| msg.caption()).unwrap_or("");

    // Session-handover quote: the first lines are a `> @bot [your
    // previous message]:` resume block seeded by the coordinator when
    // a session rolls. Any self-mention inside THAT quote is echo, not
    // a fresh address. Strip it before running the mention walker.
    let handover = classify_handover(text);
    let fresh_text: &str = if handover.is_handover_resume {
        &handover.effective_user_text
    } else {
        text
    };

    // Forwarded message gating. A forwarded body can contain arbitrary
    // `@bot` mentions (including of us) that were part of a DIFFERENT
    // conversation. We require a FRESH, outside-the-forward signal:
    // either a reply to our own message (handled above — couldn't be
    // true here since we already returned), or a caption the user
    // typed around the forward that mentions us. The body entities
    // are ignored for addressing because they came with the forward.
    if msg.forward_origin().is_some() {
        // The only way to reach this branch with `addressed = true` is
        // via a caption mention: Telegram keeps `caption_entities`
        // separate from forwarded text entities, so a user-typed
        // caption that @-mentions us counts as fresh.
        if let Some(caption_entities) = msg.parse_caption_entities() {
            let our_handle = identity.username_lc();
            let our_handle_at = format!("@{our_handle}");
            for ent in caption_entities {
                match ent.kind() {
                    MessageEntityKind::Mention
                        if ent.text().to_ascii_lowercase() == our_handle_at =>
                    {
                        return true;
                    }
                    MessageEntityKind::TextMention { user } if user.id.0 == identity.id => {
                        return true;
                    }
                    _ => {}
                }
            }
        }
        return false;
    }

    // Slash command handling. Telegram's official client appends
    // `@<bot_username>` to commands tapped in groups precisely so the
    // right bot answers; we honour that contract.
    if fresh_text.trim_start().starts_with('/') {
        if is_command_for_other_bot(fresh_text, identity) {
            return false;
        }
        return true;
    }

    // Walk message entities looking for a mention of us. We accept
    // both `Mention` (user typed `@bot`) and `TextMention` (user
    // picked us from the auto-complete menu but our @-handle isn't
    // visible — Telegram links the entity directly to a user_id).
    //
    // When the body came as a handover resume quote, we re-parse
    // entities from the stripped `effective_user_text` — any mention
    // that only lives inside the quoted block is definitionally not
    // ours to wake on.
    if handover.is_handover_resume {
        // Conservative: treat as not addressed unless the *stripped*
        // text itself begins with a slash command or literal `@bot`.
        // Re-running Telegram's entity walker against a handcrafted
        // string would require re-indexing byte offsets, which we
        // avoid here — the stripped body is small enough that a plain
        // substring check is sufficient and safe.
        let lc = fresh_text.to_ascii_lowercase();
        let handle_at = format!("@{}", identity.username_lc());
        if lc.trim_start().starts_with('/') {
            if is_command_for_other_bot(fresh_text, identity) {
                return false;
            }
            return true;
        }
        if lc.contains(&handle_at) {
            return true;
        }
        return false;
    }

    let entities = msg
        .parse_entities()
        .or_else(|| msg.parse_caption_entities());
    if let Some(entities) = entities {
        let our_handle = identity.username_lc();
        let our_handle_at = format!("@{our_handle}");
        for ent in entities {
            match ent.kind() {
                MessageEntityKind::Mention => {
                    let span = ent.text();
                    if span.to_ascii_lowercase() == our_handle_at {
                        return true;
                    }
                }
                MessageEntityKind::TextMention { user } if user.id.0 == identity.id => {
                    return true;
                }
                _ => {}
            }
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};
    use teloxide::types::Message;

    fn make_msg(v: Value) -> Message {
        serde_json::from_value(v).expect("valid Message JSON for tests")
    }

    fn identity() -> BotIdentity {
        BotIdentity {
            id: 8527746065,
            username: "zGsR_bot".to_string(),
        }
    }

    fn private_envelope(text: &str) -> Value {
        json!({
            "message_id": 1,
            "date": 0,
            "chat": { "id": 105928336, "type": "private", "first_name": "u" },
            "from": { "id": 105928336, "is_bot": false, "first_name": "u" },
            "text": text,
        })
    }

    fn group_envelope_with_entities(text: &str, entities: Value) -> Value {
        json!({
            "message_id": 2,
            "date": 0,
            "chat": { "id": -5084292206_i64, "type": "group", "title": "income" },
            "from": { "id": 105928336, "is_bot": false, "first_name": "u" },
            "text": text,
            "entities": entities,
        })
    }

    fn group_envelope_no_entities(text: &str) -> Value {
        json!({
            "message_id": 3,
            "date": 0,
            "chat": { "id": -5084292206_i64, "type": "group", "title": "income" },
            "from": { "id": 105928336, "is_bot": false, "first_name": "u" },
            "text": text,
        })
    }

    #[test]
    fn private_chat_is_always_addressed_even_for_plain_text() {
        let msg = make_msg(private_envelope("hello"));
        assert!(is_addressed_to_bot(&msg, &identity()));
    }

    #[test]
    fn group_plain_text_is_not_addressed() {
        let msg = make_msg(group_envelope_no_entities("just chatting"));
        assert!(!is_addressed_to_bot(&msg, &identity()));
    }

    #[test]
    fn group_slash_command_no_at_suffix_is_addressed() {
        // Global commands like `/research ls` work in groups too.
        let text = "/research ls";
        let msg = make_msg(group_envelope_with_entities(
            text,
            json!([{ "type": "bot_command", "offset": 0, "length": 9 }]),
        ));
        assert!(is_addressed_to_bot(&msg, &identity()));
    }

    #[test]
    fn group_slash_command_for_other_bot_is_ignored() {
        let text = "/start@OtherBot please";
        let msg = make_msg(group_envelope_with_entities(
            text,
            json!([{ "type": "bot_command", "offset": 0, "length": 15 }]),
        ));
        assert!(!is_addressed_to_bot(&msg, &identity()));
    }

    #[test]
    fn group_slash_command_for_us_is_addressed_case_insensitive() {
        let text = "/start@zgsr_BOT now";
        let msg = make_msg(group_envelope_with_entities(
            text,
            json!([{ "type": "bot_command", "offset": 0, "length": 15 }]),
        ));
        assert!(is_addressed_to_bot(&msg, &identity()));
    }

    #[test]
    fn group_at_mention_addresses_us() {
        let text = "@zGsR_bot do the thing";
        // entity covers exactly "@zGsR_bot" (9 UTF-16 code units).
        let msg = make_msg(group_envelope_with_entities(
            text,
            json!([{ "type": "mention", "offset": 0, "length": 9 }]),
        ));
        assert!(is_addressed_to_bot(&msg, &identity()));
    }

    #[test]
    fn group_at_mention_for_someone_else_is_not_addressed() {
        let text = "@OtherUser look here";
        let msg = make_msg(group_envelope_with_entities(
            text,
            json!([{ "type": "mention", "offset": 0, "length": 10 }]),
        ));
        assert!(!is_addressed_to_bot(&msg, &identity()));
    }

    #[test]
    fn group_text_mention_with_matching_user_id_addresses_us() {
        let text = "Друся, помоги пожалуйста";
        // Display name "Друся" — entity.length is in UTF-16 code units;
        // each Cyrillic char is 1 unit so length = 5.
        let msg = make_msg(group_envelope_with_entities(
            text,
            json!([{
                "type": "text_mention",
                "offset": 0,
                "length": 5,
                "user": {
                    "id": 8527746065_i64,
                    "is_bot": true,
                    "first_name": "Друся",
                    "username": "zGsR_bot"
                }
            }]),
        ));
        assert!(is_addressed_to_bot(&msg, &identity()));
    }

    #[test]
    fn group_text_mention_with_other_user_id_is_not_addressed() {
        let text = "Алиса, привет";
        let msg = make_msg(group_envelope_with_entities(
            text,
            json!([{
                "type": "text_mention",
                "offset": 0,
                "length": 5,
                "user": { "id": 999, "is_bot": false, "first_name": "Алиса" }
            }]),
        ));
        assert!(!is_addressed_to_bot(&msg, &identity()));
    }

    #[test]
    fn group_reply_to_our_message_addresses_us() {
        // Follow-up pattern: user taps Reply on one of our own answers
        // and types a clarification without retyping `@bot`. This is
        // intentional and must wake the agent.
        let json = json!({
            "message_id": 10,
            "date": 0,
            "chat": { "id": -5084292206_i64, "type": "group", "title": "income" },
            "from": { "id": 105928336, "is_bot": false, "first_name": "u" },
            "text": "ок, понял",
            "reply_to_message": {
                "message_id": 9,
                "date": 0,
                "chat": { "id": -5084292206_i64, "type": "group", "title": "income" },
                "from": {
                    "id": 8527746065_i64,
                    "is_bot": true,
                    "first_name": "Друся",
                    "username": "zGsR_bot"
                },
                "text": "что нужно сделать?"
            }
        });
        let msg = make_msg(json);
        assert!(is_addressed_to_bot(&msg, &identity()));
    }

    #[test]
    fn group_reply_to_user_message_mentioning_bot_is_not_addressed() {
        // User A wrote: "@zGsR_bot посмотри X" (message 8). User B does
        // Reply on that message with "ага согласен" (message 9) — meant
        // for User A, not for us. The bot must stay silent: our own
        // message's text has no mention of us, and the reply target is
        // a human, not us.
        let json = json!({
            "message_id": 9,
            "date": 0,
            "chat": { "id": -5084292206_i64, "type": "group", "title": "income" },
            "from": { "id": 999, "is_bot": false, "first_name": "B" },
            "text": "ага согласен",
            "reply_to_message": {
                "message_id": 8,
                "date": 0,
                "chat": { "id": -5084292206_i64, "type": "group", "title": "income" },
                "from": { "id": 123, "is_bot": false, "first_name": "A" },
                "text": "@zGsR_bot посмотри X",
                "entities": [{ "type": "mention", "offset": 0, "length": 9 }]
            }
        });
        let msg = make_msg(json);
        assert!(!is_addressed_to_bot(&msg, &identity()));
    }

    #[test]
    fn group_reply_to_other_user_is_not_addressed() {
        let json = json!({
            "message_id": 10,
            "date": 0,
            "chat": { "id": -5084292206_i64, "type": "group", "title": "income" },
            "from": { "id": 105928336, "is_bot": false, "first_name": "u" },
            "text": "agree",
            "reply_to_message": {
                "message_id": 9,
                "date": 0,
                "chat": { "id": -5084292206_i64, "type": "group", "title": "income" },
                "from": { "id": 999, "is_bot": false, "first_name": "Other" },
                "text": "что-то"
            }
        });
        let msg = make_msg(json);
        assert!(!is_addressed_to_bot(&msg, &identity()));
    }

    #[test]
    fn group_reply_to_other_bot_is_not_addressed() {
        let json = json!({
            "message_id": 10,
            "date": 0,
            "chat": { "id": -5084292206_i64, "type": "group", "title": "income" },
            "from": { "id": 105928336, "is_bot": false, "first_name": "u" },
            "text": "thanks",
            "reply_to_message": {
                "message_id": 9,
                "date": 0,
                "chat": { "id": -5084292206_i64, "type": "group", "title": "income" },
                "from": {
                    "id": 8518596807_i64,
                    "is_bot": true,
                    "first_name": "Other",
                    "username": "expense_sync_bot"
                },
                "text": "summary..."
            }
        });
        let msg = make_msg(json);
        assert!(!is_addressed_to_bot(&msg, &identity()));
    }

    #[test]
    fn caption_with_mention_addresses_us() {
        // Photo with caption mentioning the bot via caption_entities.
        let json = json!({
            "message_id": 11,
            "date": 0,
            "chat": { "id": -5084292206_i64, "type": "group", "title": "income" },
            "from": { "id": 105928336, "is_bot": false, "first_name": "u" },
            "photo": [{
                "file_id": "AgACAgX",
                "file_unique_id": "AAA",
                "width": 320,
                "height": 240,
                "file_size": 1024
            }],
            "caption": "@zGsR_bot опиши",
            "caption_entities": [{ "type": "mention", "offset": 0, "length": 9 }]
        });
        let msg = make_msg(json);
        assert!(is_addressed_to_bot(&msg, &identity()));
    }

    // ── red-matrix F1: forwarded message with @mention of another bot ─
    //
    // Scenario (seen in Income): user forwards a message from another
    // chat. The forwarded body already contains `@expense_sync_bot ...`
    // as a *mention entity*. Our `is_addressed_to_bot` today walks
    // `parse_entities()` which sees the mention and would match IF the
    // handle happened to be ours. The tighter property we want to lock:
    // even when the forwarded text mentions another bot and carries a
    // `forward_origin`, WE must not wake up. No forward + no
    // self-mention = silent.
    //
    // This test is intentionally RED: the current impl does not look at
    // `forward_origin` at all, and doesn't distinguish "forwarded from
    // elsewhere" from "addressed here". A user-crafted forward of a
    // message whose text starts with `@zGsR_bot …` would wake us
    // inappropriately. We want forwards to require a FRESH reply or a
    // FRESH mention outside the forwarded body.
    #[test]
    fn red_f1_forwarded_with_other_mention_is_not_addressed() {
        let json = json!({
            "message_id": 50,
            "date": 0,
            "chat": { "id": -5084292206_i64, "type": "group", "title": "income" },
            "from": { "id": 105928336, "is_bot": false, "first_name": "u" },
            "forward_origin": {
                "type": "user",
                "date": 0,
                "sender_user": { "id": 777, "is_bot": false, "first_name": "elsewhere" }
            },
            // Forwarded body: addresses a DIFFERENT bot. No entities for us.
            "text": "@expense_sync_bot покажи транзакции",
            "entities": [{ "type": "mention", "offset": 0, "length": 17 }]
        });
        let msg = make_msg(json);
        assert!(
            !is_addressed_to_bot(&msg, &identity()),
            "forwarded message whose mention targets another bot must NOT wake us"
        );
    }

    // Control case: same message shape but mentioning US inside the
    // forwarded body is ALSO ambiguous, and today we'd wake. The plan
    // calls for forwards to require an additional fresh signal
    // (reply-to-us or caption-mention-outside-forward). This test pins
    // that we do NOT wake on a self-mention that is purely inside a
    // forwarded body.
    #[test]
    fn red_f1_forwarded_self_mention_inside_forward_is_not_addressed() {
        let json = json!({
            "message_id": 51,
            "date": 0,
            "chat": { "id": -5084292206_i64, "type": "group", "title": "income" },
            "from": { "id": 105928336, "is_bot": false, "first_name": "u" },
            "forward_origin": {
                "type": "user",
                "date": 0,
                "sender_user": { "id": 777, "is_bot": false, "first_name": "elsewhere" }
            },
            "text": "@zGsR_bot — это цитата из другого чата",
            "entities": [{ "type": "mention", "offset": 0, "length": 9 }]
        });
        let msg = make_msg(json);
        assert!(
            !is_addressed_to_bot(&msg, &identity()),
            "a self-mention that lives entirely inside a forwarded body must NOT wake us; require a fresh signal"
        );
    }

    // ── strip_bot_command_suffix ──────────────────────────────────────

    #[test]
    fn strip_returns_none_when_no_at_suffix() {
        assert_eq!(strip_bot_command_suffix("/start", &identity()), None);
        assert_eq!(strip_bot_command_suffix("/start hello", &identity()), None);
        assert_eq!(strip_bot_command_suffix("plain text", &identity()), None);
    }

    #[test]
    fn strip_removes_our_at_suffix_preserving_args() {
        assert_eq!(
            strip_bot_command_suffix("/start@zGsR_bot", &identity()).as_deref(),
            Some("/start")
        );
        assert_eq!(
            strip_bot_command_suffix("/research@zGsR_bot ls", &identity()).as_deref(),
            Some("/research ls")
        );
        // case-insensitive
        assert_eq!(
            strip_bot_command_suffix("/start@ZGSR_BOT now", &identity()).as_deref(),
            Some("/start now")
        );
    }

    #[test]
    fn strip_returns_none_for_other_bot_suffix() {
        assert_eq!(
            strip_bot_command_suffix("/start@OtherBot args", &identity()),
            None
        );
    }

    #[test]
    fn strip_handles_leading_whitespace() {
        assert_eq!(
            strip_bot_command_suffix("  /go@zGsR_bot now", &identity()).as_deref(),
            Some("/go now")
        );
    }

    // ── is_command_for_other_bot ──────────────────────────────────────

    #[test]
    fn other_bot_cmd_classifier_works() {
        assert!(is_command_for_other_bot("/start@OtherBot", &identity()));
        assert!(!is_command_for_other_bot("/start@zGsR_bot", &identity()));
        assert!(!is_command_for_other_bot("/start", &identity()));
        assert!(!is_command_for_other_bot("plain", &identity()));
        assert!(!is_command_for_other_bot("/start@", &identity()));
    }
}
