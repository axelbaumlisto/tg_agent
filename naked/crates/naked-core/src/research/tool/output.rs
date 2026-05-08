//! Output helpers: result formatting, date parsing, finding construction.

use chrono::{Duration, NaiveDate, Utc};
use serde_json::Value;

use crate::research::spec::{Finding, content_hash, dedup_hash, host_path_hash};

use super::MAX_LISTING_AGE_DAYS;

/// Construct a Finding from tool input.
pub(crate) fn build_finding(research_id: &str, run_id: &str, url: &str, input: &Value) -> Finding {
    let excerpt_str = input
        .get("excerpt")
        .and_then(|v| v.as_str())
        .map(strip_source_attribution)
        .and_then(|s| clip(&s, 2000));
    let source_content_str = input
        .get("source_content")
        .and_then(|v| v.as_str())
        .and_then(|s| clip(s, 8000));
    let content_for_hash = content_hash(
        source_content_str
            .as_deref()
            .or(excerpt_str.as_deref())
            .unwrap_or(""),
    );
    Finding {
        id: uuid::Uuid::new_v4().simple().to_string(),
        research_id: research_id.to_string(),
        run_id: run_id.to_string(),
        url: url.to_string(),
        title: input
            .get("title")
            .and_then(|v| v.as_str())
            .and_then(|s| clip(s, 200)),
        excerpt: excerpt_str,
        price: input
            .get("price")
            .and_then(|v| v.as_str())
            .and_then(|s| clip(s, 80)),
        listing_date: input
            .get("listing_date")
            .and_then(|v| v.as_str())
            .and_then(|s| clip(s, 40)),
        source_content: source_content_str,
        dedup_hash: dedup_hash(url),
        host_path_hash: host_path_hash(url),
        content_hash: content_for_hash,
        seen_at: Utc::now(),
    }
}

/// Trim free-form agent-provided text to a max length so a single bad excerpt
/// can't balloon `findings.jsonl` to MB-per-line.
fn clip(text: &str, max: usize) -> Option<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.chars().count() <= max {
        Some(trimmed.to_string())
    } else {
        Some(trimmed.chars().take(max).collect::<String>() + "…")
    }
}

/// Strip leading/trailing source-attribution lines an LLM tends to add to the
/// excerpt even when told not to. Targets the most common Vietnamese / English
/// trailers we see in the wild: `Nguồn: ...`, `Source: ...`, `Posted by ...`,
/// `đăng N ngày/giờ trước`, `Cập nhật ...`.
///
/// Conservative: only drops a *whole line* (or a final clause separated by `—`
/// / `-` / `|`) that matches one of the known prefixes case-insensitively. We
/// never edit the body of the text, so a legitimate phone number written
/// alongside `Source: foo` survives if it's on a different line.
pub(crate) fn strip_source_attribution(text: &str) -> String {
    const PREFIXES: &[&str] = &[
        "nguồn:",
        "nguon:",
        "source:",
        "источник:",
        "posted by",
        "đăng bởi",
        "dang boi",
        "đăng ngày",
        "cập nhật",
        "cap nhat",
    ];
    fn looks_like_attribution(line: &str) -> bool {
        let t = line.trim().to_lowercase();
        if t.is_empty() {
            return false;
        }
        if PREFIXES.iter().any(|p| t.starts_with(p)) {
            return true;
        }
        // Match relative-time trailers: "đăng 3 ngày trước", "5 giờ trước".
        (t.contains(" ngày trước") || t.contains(" giờ trước")) && t.split_whitespace().count() <= 6
    }

    let mut kept: Vec<String> = text
        .lines()
        .filter(|l| !looks_like_attribution(l))
        .map(|l| l.trim_end().to_string())
        .collect();
    // Strip a single trailing clause after the last separator if it looks like
    // attribution: "...rooms — Nguồn: alonhadat.com.vn".
    if let Some(last) = kept.pop() {
        let cleaned = ["—", " - ", " | "].iter().fold(last, |acc, sep| {
            if let Some((head, tail)) = acc.rsplit_once(sep)
                && looks_like_attribution(tail)
            {
                head.trim_end().to_string()
            } else {
                acc
            }
        });
        kept.push(cleaned);
    }
    kept.join("\n").trim().to_string()
}

/// Try to parse a free-form listing date string into a NaiveDate.
/// Handles: `YYYY-MM-DD`, `DD/MM/YYYY`, `DD-MM-YYYY`, `DD.MM.YYYY`,
/// relative Vietnamese (`hôm nay`, `hôm qua`, `N ngày trước`), and `unknown`.
pub fn parse_listing_date(s: &str) -> Option<NaiveDate> {
    let s = s.trim().to_lowercase();
    if s.is_empty() || s == "unknown" {
        return None;
    }
    let today = Utc::now().date_naive();

    if s.contains("hôm nay") || s == "today" {
        return Some(today);
    }
    if s.contains("hôm qua") || s == "yesterday" {
        return Some(today - Duration::days(1));
    }
    // "N ngày trước" / "N days ago"
    if let Some(n) = extract_days_ago(&s) {
        return Some(today - Duration::days(n));
    }

    // ISO: 2026-04-18
    if let Ok(d) = NaiveDate::parse_from_str(&s, "%Y-%m-%d") {
        return Some(d);
    }
    // DD/MM/YYYY
    if let Ok(d) = NaiveDate::parse_from_str(&s, "%d/%m/%Y") {
        return Some(d);
    }
    // DD-MM-YYYY
    if let Ok(d) = NaiveDate::parse_from_str(&s, "%d-%m-%Y") {
        return Some(d);
    }
    // DD.MM.YYYY
    if let Ok(d) = NaiveDate::parse_from_str(&s, "%d.%m.%Y") {
        return Some(d);
    }
    None
}

fn extract_days_ago(s: &str) -> Option<i64> {
    // Match patterns like "3 ngày trước", "5 days ago"
    for word in s.split_whitespace() {
        if let Ok(n) = word.parse::<i64>()
            && (s.contains("ngày trước") || s.contains("days ago"))
        {
            return Some(n);
        }
    }
    None
}

/// Check if a listing date is too old (> MAX_LISTING_AGE_DAYS).
/// Returns `None` if date can't be parsed (let it through — benefit of the doubt).
pub(crate) fn is_stale(listing_date: Option<&str>) -> Option<bool> {
    let s = listing_date?;
    let d = parse_listing_date(s)?;
    let age = Utc::now().date_naive() - d;
    Some(age.num_days() > MAX_LISTING_AGE_DAYS)
}
