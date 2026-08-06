//! Telegram markup helpers. Re-exports preserve former
//! `tg_markup::` API path. Do not re-add unrelated functions
//! to this crate root — split them into siblings.

/// Telegram's hard message-length limit in UTF-16 code units.
///
/// Some legacy call sites still use this value as a conservative byte budget;
/// final-answer delivery gates on [`telegram_html_text_utf16_units`] when the
/// long-answer fix flag is enabled.
pub const MAX_TG_MSG: usize = 4096;

pub use budget::{telegram_html_text_utf16_units, telegram_html_visible_text};
pub use chunk::{split_html, split_stable_unstable};
pub use md_html::{escape_html, md_to_tg_html};
pub use util::{format_tokens, normalize_md, truncate_button};

mod budget;
mod chunk;
mod md_html;
mod util;
