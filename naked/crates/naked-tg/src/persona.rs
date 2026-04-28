//! Persona-aware gating for command dispatch.
//!
//! Today the `/..` command gate (`drop_slash_for_persona`) only
//! inspects `Message`. An inline keyboard button with `callback_data`
//! starting with `/` slips past the check — that means a persona with
//! `allow_slash_commands: false` (Income group) can still be made to
//! run privileged commands via a crafted callback.
//!
//! `DispatchChannel` + `is_slash_dispatchable` give the main loop a
//! single predicate to consult for both paths.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchChannel {
    /// The user typed the slash in a chat message.
    Message,
    /// The slash arrived as `CallbackQuery.data` from an inline keyboard.
    Callback,
}

/// True iff this slash-like input should be dispatched as a command,
/// given the persona's `allow_slash_commands` flag and which channel
/// it arrived on.
///
/// Non-slash inputs always pass through unchanged (returns `true`),
/// because this predicate is only about slash-command gating; free-form
/// text is handled elsewhere.
pub fn is_slash_dispatchable(
    persona_allows_slash: bool,
    text: &str,
    channel: DispatchChannel,
) -> bool {
    if !text.trim_start().starts_with('/') {
        return true;
    }
    // Gate applies equally to both channels. A persona that disallows
    // slash commands in messages MUST also disallow them in callbacks —
    // otherwise the setting is a foot-gun.
    let _ = channel;
    persona_allows_slash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disallowed_persona_drops_slash_in_message_and_callback() {
        assert!(!is_slash_dispatchable(false, "/reset", DispatchChannel::Message));
        assert!(!is_slash_dispatchable(false, "/reset", DispatchChannel::Callback));
    }

    #[test]
    fn allowing_persona_accepts_slash() {
        assert!(is_slash_dispatchable(true, "/reset", DispatchChannel::Message));
        assert!(is_slash_dispatchable(true, "/reset", DispatchChannel::Callback));
    }

    #[test]
    fn non_slash_text_always_passes() {
        assert!(is_slash_dispatchable(false, "hello", DispatchChannel::Message));
        assert!(is_slash_dispatchable(false, "привет", DispatchChannel::Callback));
    }
}
