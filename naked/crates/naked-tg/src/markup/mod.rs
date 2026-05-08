//! Telegram markup helpers. Re-exports preserve former
//! `tg_markup::` API path. Do not re-add unrelated functions
//! to this crate root — split them into siblings.

/// Telegram's hard message-length limit (UTF-8 bytes).
pub const MAX_TG_MSG: usize = 4096;

pub use chunk::{split_html, split_stable_unstable};
pub use md_html::{escape_html, md_to_tg_html};
pub use util::{format_tokens, normalize_md, truncate_button};

mod chunk;
mod md_html;
mod util;
