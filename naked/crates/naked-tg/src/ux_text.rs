//! Centralised short control/acknowledgement copy for Telegram dispatch paths.
//!
//! Keep this deliberately narrow: these are not a full i18n layer, just the
//! busy/steer acknowledgements that decide whether a user message was accepted.

/// Session is busy and the incoming message was NOT queued or saved.
pub(crate) const BUSY_ACK: &str = "⏳ Занят — сообщение не принято, пришли ещё раз";

/// Follow-up was accepted as a steer message for the active turn.
pub(crate) const STEER_ACK: &str = "↩️ Принято — доставлю между шагами";
