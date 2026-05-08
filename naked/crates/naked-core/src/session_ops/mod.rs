//! Session operations — split into focused submodules.
//!
//! - `lifecycle`    : create, send, queue, fork, restore, close, refresh
//! - `turn`         : setup_turn, dispatch_turn, build_tool_registry_for
//! - `control`      : abort, compact_session, compaction helpers
//! - `diagnostics`  : list, inspect, provider_for, store

mod control;
mod diagnostics;
mod lifecycle;
mod turn;
