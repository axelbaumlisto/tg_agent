#![cfg_attr(not(test), warn(clippy::unwrap_used))]
//! Library surface of `naked-tg`.
//!
//! The Telegram bot itself lives in `main.rs`; this lib only re-exports the
//! pieces that have public test surface (the in-process research scheduler
//! and a couple of pure helpers used by `/research` commands).
//!
//! Keeping these in a lib crate means integration tests under
//! `naked-tg/tests/` can import them via `use naked_tg::...;` without
//! relying on `#[cfg(test)]`-only inline modules.

pub mod bot_identity;
pub mod channel_map;
pub mod cron_util;
pub mod helpers;
pub mod media_helpers;
pub mod memory_scheduler;
pub mod model_glob;
pub mod model_switch;
pub mod persona;
pub mod rate_limit;
pub mod render;
pub mod research_html;
pub mod scheduler;
pub use scheduler as research_scheduler;
pub mod guarded;
pub mod markup;
pub mod research_ui;
pub mod scheduler_lock;
pub mod skill;
pub mod tg_attach;
pub mod watchdog;
