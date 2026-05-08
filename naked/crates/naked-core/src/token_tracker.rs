//! Per-session token usage tracker — model × (input, output, calls).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub calls: u64,
}

impl ModelUsage {
    pub fn total(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }
}

#[derive(Debug, Clone, Default)]
pub struct TokenTracker {
    inner: Arc<Mutex<HashMap<String, ModelUsage>>>,
}

impl TokenTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&self, model: &str, input_tokens: u64, output_tokens: u64) {
        let mut map = crate::lock_or_recover(&self.inner);
        let entry = map.entry(model.to_string()).or_default();
        entry.input_tokens += input_tokens;
        entry.output_tokens += output_tokens;
        entry.calls += 1;
    }

    pub fn snapshot(&self) -> HashMap<String, ModelUsage> {
        crate::lock_or_recover(&self.inner).clone()
    }

    pub fn totals(&self) -> ModelUsage {
        let map = crate::lock_or_recover(&self.inner);
        map.values().fold(ModelUsage::default(), |mut acc, u| {
            acc.input_tokens += u.input_tokens;
            acc.output_tokens += u.output_tokens;
            acc.calls += u.calls;
            acc
        })
    }

    pub fn summary(&self) -> String {
        let snap = self.snapshot();
        if snap.is_empty() {
            return "No token usage recorded".into();
        }
        let mut models: Vec<_> = snap.into_iter().collect();
        models.sort_by_key(|(_name, usage)| std::cmp::Reverse(usage.total()));
        let mut lines: Vec<String> = models
            .iter()
            .map(|(m, u)| {
                format!(
                    "  {m}: {c} calls, {i} in / {o} out",
                    c = u.calls,
                    i = u.input_tokens,
                    o = u.output_tokens
                )
            })
            .collect();
        let t = self.totals();
        lines.push(format!(
            "Total: {} calls, {} in / {} out ({} tokens)",
            t.calls,
            t.input_tokens,
            t.output_tokens,
            t.total()
        ));
        lines.join("\n")
    }

    pub fn reset(&self) {
        crate::lock_or_recover(&self.inner).clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_and_snapshot() {
        let t = TokenTracker::new();
        t.record("gpt-4", 100, 50);
        t.record("gpt-4", 200, 100);
        t.record("claude", 300, 150);
        let s = t.snapshot();
        assert_eq!(
            s["gpt-4"],
            ModelUsage {
                input_tokens: 300,
                output_tokens: 150,
                calls: 2
            }
        );
        assert_eq!(s["claude"].calls, 1);
    }

    #[test]
    fn totals_across_models() {
        let t = TokenTracker::new();
        t.record("a", 100, 50);
        t.record("b", 200, 100);
        assert_eq!(
            t.totals(),
            ModelUsage {
                input_tokens: 300,
                output_tokens: 150,
                calls: 2
            }
        );
    }

    #[test]
    fn summary_format() {
        let t = TokenTracker::new();
        t.record("test-model", 1000, 500);
        let s = t.summary();
        assert!(s.contains("test-model") && s.contains("1000") && s.contains("500"));
    }

    #[test]
    fn reset_clears() {
        let t = TokenTracker::new();
        t.record("x", 10, 5);
        t.reset();
        assert!(t.snapshot().is_empty());
    }

    #[test]
    fn empty_summary() {
        assert_eq!(TokenTracker::new().summary(), "No token usage recorded");
    }
}
