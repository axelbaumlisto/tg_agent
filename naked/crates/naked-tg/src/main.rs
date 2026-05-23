//! `naked-tg` binary entry point.
//!
//! This file is intentionally slim: it declares all crate modules (required
//! because Rust resolves modules from the crate root), re-exports the shared
//! types so that child modules' `use super::*;` keeps working without change,
//! and delegates to [`bootstrap::run`] for all real work.

// ── Sub-module declarations (must live in the crate root) ─────────────────
mod album;
mod media;
mod metrics;

// ── Shared types / statics / helpers ──────────────────────────────────────
// All items in `shared` are pub(crate); re-exporting them here lets every
// child module continue to use `use super::*;` to get them.
mod shared;
pub(crate) use shared::*;

// ── Existing extracted child modules ──────────────────────────────────────
mod callbacks;
mod commands;
mod fmt_utils;
mod media_dispatch;
mod message_handler;
#[path = "streaming_mod/mod.rs"]
mod streaming;

// ── Re-export child-module items consumed by grandchild modules ───────────
// These allow `use super::*;` in grandchild modules (commands/*, streaming_mod/*)
// to resolve the symbols without modification.
pub(crate) use commands::{drop_slash_for_persona, get_or_create_session, handle_command};
pub(crate) use media_dispatch::{extract_media_items, process_media_items, send_text};
pub(crate) use streaming::stream_response;

// ── Structural pipeline modules ────────────────────────────────────────────
mod bootstrap;
mod runtime;
mod wiring;

// ── Entry point ────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    bootstrap::run().await;
}
