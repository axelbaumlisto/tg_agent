//! Per-host adaptive tier selection for the `web_fetch` cascade.
//!
//! Why
//! ---
//! The default cascade tries every tier in order: reqwest → url-prefix →
//! TLS impersonation → cloud-scrape → Wayback. For sites where the first
//! three tiers ALWAYS fail (batdongsan.com.vn under CF challenge, chotot
//! phone-reveal SPA, etc.), running them on every fetch wastes ~10 s per
//! request in pointless connect/timeout/parse cycles.
//!
//! [`HostPolicy`] watches outcomes per `(host, tier)` and produces a
//! `start_tier` recommendation: the first tier that hasn't already proven
//! itself useless for this host. It's purely advisory — the caller still
//! runs every later tier as a fallback if the recommended one fails.
//!
//! Design notes
//! ------------
//! - In-memory only. The cost of a wrong recommendation is one wasted
//!   tier attempt; persistence isn't worth the disk I/O for a workload
//!   that already reloads its full key pool from disk every 6 hours.
//! - Tiers are an ordered enum so "skip ahead to N" is just numeric
//!   comparison; no string parsing on the hot path.
//! - The "blocked" threshold (`min_attempts_before_skip`) defaults to 3:
//!   a single 502 from reqwest shouldn't ban it forever; three in a row
//!   indicates structural CF protection.
//! - Successful fetches **always** drop the host's tier counter back to
//!   `Reqwest` — a previously-blocked site may have lifted the block,
//!   and we want to discover that on the next request rather than
//!   permanently routing through the paid tier.

use std::collections::HashMap;
use std::sync::RwLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Tier {
    Reqwest = 0,
    UrlPrefix = 1,
    Tls = 2,
    Cloud = 3,
    Wayback = 4,
}

impl Tier {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Reqwest => "reqwest",
            Self::UrlPrefix => "url-prefix",
            Self::Tls => "tls",
            Self::Cloud => "cloud",
            Self::Wayback => "wayback",
        }
    }

    pub fn next(&self) -> Option<Tier> {
        match self {
            Self::Reqwest => Some(Self::UrlPrefix),
            Self::UrlPrefix => Some(Self::Tls),
            Self::Tls => Some(Self::Cloud),
            Self::Cloud => Some(Self::Wayback),
            Self::Wayback => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// Tier returned a usable 2xx-3xx body with no anti-bot signal.
    Ok,
    /// Tier returned a recognisable block (CF challenge, anti-bot wall,
    /// 4xx/5xx). Counts toward the per-tier failure budget.
    Blocked,
    /// Tier wasn't tried (skipped because policy said so, or no client
    /// configured). Recorded so observability can distinguish "tier
    /// disabled" from "tier failed".
    Skipped,
}

#[derive(Debug, Default, Clone)]
struct TierStats {
    successes: u32,
    failures: u32,
}

#[derive(Default)]
struct HostState {
    tiers: HashMap<Tier, TierStats>,
}

pub struct HostPolicy {
    inner: RwLock<HashMap<String, HostState>>,
    /// Number of consecutive failures before a tier is considered "doomed"
    /// for that host and dropped from the recommended start point.
    min_attempts_before_skip: u32,
}

impl Default for HostPolicy {
    fn default() -> Self {
        Self {
            inner: RwLock::new(HashMap::new()),
            min_attempts_before_skip: 3,
        }
    }
}

impl HostPolicy {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_threshold(threshold: u32) -> Self {
        Self {
            inner: RwLock::new(HashMap::new()),
            min_attempts_before_skip: threshold,
        }
    }

    /// Cheap host extraction. Avoids pulling in the `url` crate just for
    /// dropping scheme + path. Returns lowercased host or `""` when the
    /// URL is malformed (matches `canonicalize_url`'s permissive style).
    pub fn host_of(url: &str) -> String {
        let trimmed = url.trim();
        let rest = trimmed.split_once("://").map(|(_, r)| r).unwrap_or(trimmed);
        let host = rest.split('/').next().unwrap_or("");
        host.to_ascii_lowercase()
    }

    pub fn record(&self, url: &str, tier: Tier, outcome: Outcome) {
        let host = Self::host_of(url);
        if host.is_empty() {
            return;
        }
        let mut inner = crate::write_or_recover(&self.inner);
        let state = inner.entry(host).or_default();
        match outcome {
            Outcome::Ok => {
                let stats = state.tiers.entry(tier).or_default();
                stats.successes = stats.successes.saturating_add(1);
                // A success at any tier resets every earlier tier's
                // failure count — the host clearly came back online and
                // we want the cheaper tiers to get a fair shot next time.
                let mut t = Tier::Reqwest;
                while t < tier {
                    if let Some(stats) = state.tiers.get_mut(&t) {
                        stats.failures = 0;
                    }
                    t = match t.next() {
                        Some(n) => n,
                        None => break,
                    };
                }
            }
            Outcome::Blocked => {
                let stats = state.tiers.entry(tier).or_default();
                stats.failures = stats.failures.saturating_add(1);
            }
            Outcome::Skipped => {}
        }
    }

    /// Return the first tier that hasn't been proven doomed for this host.
    /// Defaults to `Tier::Reqwest` for unknown hosts (current behaviour).
    pub fn recommended_start_tier(&self, url: &str) -> Tier {
        let host = Self::host_of(url);
        let inner = crate::read_or_recover(&self.inner);
        let Some(state) = inner.get(&host) else {
            return Tier::Reqwest;
        };
        let mut t = Tier::Reqwest;
        loop {
            let doomed = state
                .tiers
                .get(&t)
                .map(|s| s.successes == 0 && s.failures >= self.min_attempts_before_skip)
                .unwrap_or(false);
            if !doomed {
                return t;
            }
            match t.next() {
                Some(n) => t = n,
                None => return t,
            }
        }
    }

    /// Snapshot of `(host, tier, ok, fail)` rows for telemetry / debugging.
    /// Cheap clone — operators occasionally want to dump this from the CLI.
    pub fn snapshot(&self) -> Vec<(String, Tier, u32, u32)> {
        let inner = crate::read_or_recover(&self.inner);
        let mut out = Vec::new();
        for (host, state) in inner.iter() {
            for (tier, stats) in state.tiers.iter() {
                out.push((host.clone(), *tier, stats.successes, stats.failures));
            }
        }
        out.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_of_parses_lowercased_authority() {
        assert_eq!(HostPolicy::host_of("https://A.B.com/x"), "a.b.com");
        assert_eq!(HostPolicy::host_of("http://x.example/"), "x.example");
        assert_eq!(HostPolicy::host_of("not-a-url"), "not-a-url");
        assert_eq!(HostPolicy::host_of(""), "");
    }

    #[test]
    fn unknown_host_starts_at_reqwest() {
        let p = HostPolicy::new();
        assert_eq!(
            p.recommended_start_tier("https://fresh.example/x"),
            Tier::Reqwest
        );
    }

    #[test]
    fn three_consecutive_blocks_skip_tier() {
        let p = HostPolicy::with_threshold(3);
        let url = "https://batdongsan.com.vn/ad/1";
        p.record(url, Tier::Reqwest, Outcome::Blocked);
        assert_eq!(p.recommended_start_tier(url), Tier::Reqwest);
        p.record(url, Tier::Reqwest, Outcome::Blocked);
        p.record(url, Tier::Reqwest, Outcome::Blocked);
        assert_eq!(p.recommended_start_tier(url), Tier::UrlPrefix);
    }

    #[test]
    fn doomed_tiers_chain_until_first_unknown() {
        let p = HostPolicy::with_threshold(2);
        let url = "https://chotot.com/listing/9";
        for _ in 0..2 {
            p.record(url, Tier::Reqwest, Outcome::Blocked);
            p.record(url, Tier::UrlPrefix, Outcome::Blocked);
            p.record(url, Tier::Tls, Outcome::Blocked);
        }
        assert_eq!(p.recommended_start_tier(url), Tier::Cloud);
    }

    #[test]
    fn one_success_resets_earlier_tiers() {
        let p = HostPolicy::with_threshold(2);
        let url = "https://flaky.example/x";
        // Ban reqwest with two failures.
        p.record(url, Tier::Reqwest, Outcome::Blocked);
        p.record(url, Tier::Reqwest, Outcome::Blocked);
        assert_eq!(p.recommended_start_tier(url), Tier::UrlPrefix);
        // Then TLS succeeds → reqwest's failure counter resets.
        p.record(url, Tier::Tls, Outcome::Ok);
        assert_eq!(p.recommended_start_tier(url), Tier::Reqwest);
    }

    #[test]
    fn last_tier_returns_itself_when_all_doomed() {
        let p = HostPolicy::with_threshold(1);
        let url = "https://dead.example/x";
        for tier in [
            Tier::Reqwest,
            Tier::UrlPrefix,
            Tier::Tls,
            Tier::Cloud,
            Tier::Wayback,
        ] {
            p.record(url, tier, Outcome::Blocked);
        }
        // No tier qualifies; the recommendation lands on the last one
        // (Wayback) because there's nothing after it.
        assert_eq!(p.recommended_start_tier(url), Tier::Wayback);
    }

    #[test]
    fn snapshot_returns_sorted_rows() {
        let p = HostPolicy::new();
        p.record("https://b.example/x", Tier::Tls, Outcome::Ok);
        p.record("https://a.example/x", Tier::Reqwest, Outcome::Blocked);
        let rows = p.snapshot();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].0, "a.example");
        assert_eq!(rows[0].1, Tier::Reqwest);
        assert_eq!(rows[1].0, "b.example");
        assert_eq!(rows[1].1, Tier::Tls);
    }
}
