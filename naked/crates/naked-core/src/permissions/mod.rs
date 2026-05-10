//! Pattern-based permission rules (T5 of `PLAN_QUALITY_v1.md`).
//!
//! Replaces the all-or-nothing YOLO/per-tool gate with a ruleset
//! keyed on `tool name × path pattern × action`. Last-match-wins
//! semantics; `~/$HOME/` expansion; persistent storage.
//!
//! See `ruleset.rs` for the data model, `matcher.rs` for glob
//! semantics, `store.rs` for the on-disk format.

pub mod matcher;
pub mod ruleset;
pub mod store;

pub use ruleset::{Action, Reply, Rule, Ruleset};
pub use store::Store;
