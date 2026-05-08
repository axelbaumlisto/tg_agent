//! Union-find helpers and top-level `reconcile()` (B5-2).
//!
//! This is the final assembly step: group findings by URL (and
//! optionally title), pick the highest-scored winner per group, and
//! annotate the merged record with canonical fields.
use serde_json::{Map, Value, json};
use std::collections::HashMap;

use super::norm::{canonicalize_url, normalize_title};
use super::price::{extract_price, to_usd};

// ── B5-2: top-level reconcile() (R1.5-step-3) ────────────────────────

/// Tie-breaker for winner pick. Prefers `_score`, falls back to
/// `_relevance`, falls back to 0.0. Bit-exact mirror of Python
/// `_winning_score`. `Value::Bool` is rejected by construction —
/// `Value::Number` cannot contain a bool, so the Python "bool is
/// a subclass of int" gotcha can't surface here.
fn winning_score(finding: &Value) -> f64 {
    fn pick(v: Option<&Value>) -> Option<f64> {
        match v {
            Some(Value::Number(n)) => n.as_f64(),
            _ => None,
        }
    }
    pick(finding.get("_score"))
        .or_else(|| pick(finding.get("_relevance")))
        .unwrap_or(0.0)
}

/// Read a string field, returning `""` for missing / non-string /
/// `null`. Matches Python `f.get(key) or ""` for the four text
/// fields used in dedup (`link`, `url`, `title`, `snippet`,
/// `description`).
fn finding_str<'a>(finding: &'a Value, key: &str) -> &'a str {
    finding.get(key).and_then(Value::as_str).unwrap_or("")
}

/// Combine the three text fields most likely to hold a price.
/// Order: title | snippet | description. Mirrors Python
/// `_price_signal_text`. The " · " separator matches byte-for-byte.
fn price_signal_text(finding: &Value) -> String {
    ["title", "snippet", "description"]
        .iter()
        .map(|k| finding_str(finding, k))
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" · ")
}

/// Union-find `find` with path compression, applied to a flat
/// `parent` slice. Free function (no closures over outer state)
/// so the borrow checker stays out of the way.
fn uf_find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

/// Union with **min-becomes-root**: the smallest index in any group
/// is always the root. This is what guarantees that
/// `_canonical_url` / `_normalised_title` bucket iteration order
/// does not affect the final group membership — Python relies on
/// the same property.
fn uf_union(parent: &mut [usize], a: usize, b: usize) {
    let ra = uf_find(parent, a);
    let rb = uf_find(parent, b);
    if ra != rb {
        let new_root = ra.min(rb);
        let old_root = ra.max(rb);
        parent[old_root] = new_root;
    }
}

/// Cross-source merge of findings. Bit-exact Rust port of Python
/// `reconciler.reconcile(findings, fx_rates, merge_by_title)`.
///
/// # Determinism contract
/// - Output preserves first-seen group order.
/// - Within a group: max `_score` → max `_relevance` → first-seen.
/// - `_sources` sorted by `_source_id` (missing/non-string → "").
///
/// # Schema-added keys (each survivor)
/// - `_canonical_url` — `canonicalize_url(winner.link or .url)`
/// - `_normalised_title` — `normalize_title(winner.title)`
/// - `_sources` — `Vec<{_source_id,_score,_relevance}>`, sorted
/// - `_price_struct` — `{amount,currency,raw}` or `null`
/// - `price_usd` — `f64` or `null`
///
/// # `merge_by_title` opt-in
/// Default is **URL-only dedup**. Title-only merge is risky on
/// marketplace corpora that share CTA-style titles ("узнать
/// наличие", "Apartment", "BMW M5"); see the Python module docstring
/// for the corpus survey behind the default.
pub fn reconcile(
    findings: &[Value],
    fx_rates: Option<&[(&str, f64)]>,
    merge_by_title: bool,
) -> Vec<Value> {
    // 1. Filter non-object entries; pre-compute url/title keys.
    struct Item<'a> {
        src: &'a Value,
        url_key: String,
        title_key: String,
    }
    let items: Vec<Item> = findings
        .iter()
        .filter(|f| f.is_object())
        .map(|f| {
            let link = finding_str(f, "link");
            let url = if link.is_empty() {
                finding_str(f, "url")
            } else {
                link
            };
            Item {
                src: f,
                url_key: canonicalize_url(url),
                title_key: normalize_title(finding_str(f, "title")),
            }
        })
        .collect();
    if items.is_empty() {
        return Vec::new();
    }

    let n = items.len();

    // 2. Bucket by each non-empty signal.
    //
    // We use HashMap (not BTreeMap) because bucket iteration order
    // does not affect the final union-find result — the
    // min-becomes-root invariant in `uf_union` makes the root of
    // every group the smallest index in it, regardless of the order
    // in which buckets are processed.
    let mut by_url: HashMap<&str, Vec<usize>> = HashMap::new();
    let mut by_title: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, it) in items.iter().enumerate() {
        if !it.url_key.is_empty() {
            by_url.entry(it.url_key.as_str()).or_default().push(i);
        }
        if !it.title_key.is_empty() {
            by_title.entry(it.title_key.as_str()).or_default().push(i);
        }
    }

    // 3. Union members of each bucket pairwise.
    let mut parent: Vec<usize> = (0..n).collect();
    for members in by_url.values() {
        for &j in members.iter().skip(1) {
            uf_union(&mut parent, members[0], j);
        }
    }
    if merge_by_title {
        for members in by_title.values() {
            for &j in members.iter().skip(1) {
                uf_union(&mut parent, members[0], j);
            }
        }
    }

    // 4. Collect groups preserving first-seen group order.
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut group_order: Vec<usize> = Vec::new();
    for i in 0..n {
        let root = uf_find(&mut parent, i);
        if !groups.contains_key(&root) {
            group_order.push(root);
        }
        groups.entry(root).or_default().push(i);
    }

    // 5. Per group: pick winner, build the merged record.
    let mut out: Vec<Value> = Vec::with_capacity(group_order.len());
    for root in group_order {
        let member_idx = &groups[&root];
        let bucket: Vec<&Value> = member_idx.iter().map(|&i| items[i].src).collect();

        // Winner: max score, ties broken by smaller in-bucket index.
        // Mirrors Python `max(range(len(bucket)), key=lambda i: (score, -i))`.
        let winner_local: usize = (0..bucket.len())
            .max_by(|&a, &b| {
                let sa = winning_score(bucket[a]);
                let sb = winning_score(bucket[b]);
                match sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal) {
                    std::cmp::Ordering::Equal => b.cmp(&a),
                    other => other,
                }
            })
            .unwrap_or(0);

        // Clone the winner so we can mutate without poisoning input.
        let mut winner: Map<String, Value> = bucket[winner_local]
            .as_object()
            .cloned()
            .unwrap_or_default();

        // Recompute canonical fields from the winner — Python
        // explicitly recomputes here (rather than using the
        // first-seen's keys) so cross-group carries the survivor's
        // URL/title in the merged record.
        let winner_url = winner
            .get("link")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .or_else(|| winner.get("url").and_then(Value::as_str))
            .unwrap_or("");
        let canonical_url = canonicalize_url(winner_url);
        let normalised_title =
            normalize_title(winner.get("title").and_then(Value::as_str).unwrap_or(""));

        // _sources: one entry per group member, sorted by _source_id.
        // Missing or non-string _source_id sorts as empty string —
        // mirrors Python `s.get('_source_id') or ''`.
        let mut sources_meta: Vec<Value> = bucket
            .iter()
            .map(|f| {
                let mut m = Map::with_capacity(3);
                m.insert(
                    "_source_id".into(),
                    f.get("_source_id").cloned().unwrap_or(Value::Null),
                );
                m.insert(
                    "_score".into(),
                    f.get("_score").cloned().unwrap_or(Value::Null),
                );
                m.insert(
                    "_relevance".into(),
                    f.get("_relevance").cloned().unwrap_or(Value::Null),
                );
                Value::Object(m)
            })
            .collect();
        sources_meta.sort_by(|a, b| {
            let sa = a.get("_source_id").and_then(Value::as_str).unwrap_or("");
            let sb = b.get("_source_id").and_then(Value::as_str).unwrap_or("");
            sa.cmp(sb)
        });

        // Price extraction: title | snippet | description joined by " · ".
        let signal = price_signal_text(&Value::Object(winner.clone()));
        let price_struct = extract_price(&signal, None);
        let price_usd: Option<f64> = price_struct
            .as_ref()
            .and_then(|p| to_usd(p.amount, p.currency, fx_rates));

        winner.insert("_canonical_url".into(), Value::String(canonical_url));
        winner.insert("_normalised_title".into(), Value::String(normalised_title));
        winner.insert("_sources".into(), Value::Array(sources_meta));
        winner.insert(
            "_price_struct".into(),
            match price_struct {
                Some(p) => json!({
                    "amount":   p.amount,
                    "currency": p.currency,
                    "raw":      p.raw,
                }),
                None => Value::Null,
            },
        );
        winner.insert(
            "price_usd".into(),
            match price_usd {
                Some(v) => json!(v),
                None => Value::Null,
            },
        );

        out.push(Value::Object(winner));
    }
    out
}
