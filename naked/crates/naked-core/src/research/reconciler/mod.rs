//! `reconciler` — block B5 of the universal-research blueprint, Rust port.
//!
//! This module is the deterministic counterpart of Python
//! `scripts/reconciler.py`. The two implementations share **one
//! contract**, frozen in `tests/fixtures/reconciler/*.json`. Every
//! function here is gated against the same JSON cases as the Python
//! script — that's the differential test (group 33 in
//! `tests/run_all_offline.sh`).
//!
//! Why two `normalize_title` functions live in this crate
//! ──────────────────────────────────────────────────────
//! Yes, [`crate::research::spec::normalize_title_for_similarity`] also
//! exists. It's a different concept:
//!
//! * `spec::normalize_title_for_similarity` — aggressive **stemming**
//!   (drops digit tokens ≤ 4 chars, drops the `m2`/`sqm`/`tang`/`pn`
//!   stopword set, no quote/dash translation). Used by
//!   [`crate::research::spec::titles_are_similar`] for **fuzzy
//!   Jaccard matching** between near-duplicate listings on different
//!   crossposts.
//! * `reconciler::normalize_title` (this module) — conservative
//!   **surface cleanup** (lowercase, NFKC, quote/dash/NBSP translation,
//!   whitespace collapse; **no** stemming). Used by
//!   [`reconciler::reconcile`] for **strict-equality dedup** where
//!   "BMW X5" and "BMW X5 (urgent)" must NOT collapse.
//!
//! Both are kept; both are correct for their use case. The
//! cross-reference here and in `spec.rs` is the boy-scout fix to
//! prevent future contributors from "deduplicating" the duplicate.
//!
//! Port-now scope
//! ──────────────
//! * **R1.5-step-1 (2026-04-27)** — `canonicalize_url` + `normalize_title`.
//! * **R1.5-step-2 (2026-04-27 +1)** — `extract_price` + `to_usd`.
//! * **R1.5-step-3 (2026-04-27 +2)** — top-level `reconcile()`.
//!
//! With step-3 the entire deterministic CPU path of the Python
//! `scripts/reconciler.py` has a bit-exact Rust counterpart. Group
//! 33 (`reconciler_rust_differential_offline`) gates 6 Rust tests
//! against `tests/fixtures/reconciler/{url_title,price,dedup}.json`
//! — the same JSON the Python e2e groups (28/29/30) consume. Drift
//! on either side fails on either CI run.
//!
//! ## API note: the `reconcile()` shape
//!
//! Python's signature is
//! `reconcile(findings: list[dict], fx_rates=None, merge_by_title=False) -> list[dict]`.
//! Rust returns `Vec<serde_json::Value>` so callers don't have to
//! commit to a typed Finding struct (the cross-source pipeline
//! handles many shapes — `agent_research`, `playwright_generic`,
//! `discover` — and dedup must be schema-tolerant). The function is
//! still pure: input slice in, owned `Vec` out, no I/O, no globals.
//!
//! ## Submodule layout
//! - `norm`  — URL canonicalization + title normalization (B5-1)
//! - `price` — price extraction + FX conversion (B5-3)
//! - `merge` — union-find helpers + top-level `reconcile()` (B5-2)

pub mod merge;
pub mod norm;
pub mod price;

pub use merge::reconcile;
pub use norm::canonicalize_url;
pub(crate) use norm::normalize_title; // used by memory_diff
pub use price::extract_price;
// `price::to_usd` and `price::ExtractedPrice` stay pub(crate) at their
// definition site — reconciler_tests.rs imports them via explicit
// `use super::price::{to_usd, ExtractedPrice}` paths.

#[cfg(test)]
#[path = "../reconciler_tests.rs"]
mod tests;
