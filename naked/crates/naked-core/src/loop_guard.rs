//! Pure-data guardrails for repeated tool-call loops.

use serde_json::Value;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::hash::{Hash, Hasher};

const IDENTICAL_CALL_BLOCK_THRESHOLD: u32 = 3;
const FAILURE_WARN_THRESHOLD: u32 = 3;
const FAILURE_HALT_THRESHOLD: u32 = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttemptDecision {
    Proceed,
    Block(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutcomeDecision {
    Continue,
    Warn(String),
    Halt(String),
}

#[derive(Debug, Default)]
pub struct LoopGuard {
    call_counts: HashMap<(String, u64), u32>,
    failure_counts: HashMap<String, u32>,
}

impl LoopGuard {
    pub fn record_attempt(&mut self, tool: &str, args: &Value) -> AttemptDecision {
        let key = (tool.to_string(), hash_args(args));
        let count = self.call_counts.entry(key).or_insert(0);
        *count = count.saturating_add(1);
        if *count >= IDENTICAL_CALL_BLOCK_THRESHOLD {
            return AttemptDecision::Block(format!(
                "Blocked: `{tool}` with these exact arguments already ran {count} times this turn. Change the arguments or pick a different approach."
            ));
        }
        AttemptDecision::Proceed
    }

    pub fn record_outcome(&mut self, tool: &str, ok: bool) -> OutcomeDecision {
        let failures = self.failure_counts.entry(tool.to_string()).or_insert(0);
        if ok {
            *failures = 0;
            return OutcomeDecision::Continue;
        }
        *failures = failures.saturating_add(1);
        if *failures >= FAILURE_HALT_THRESHOLD {
            return OutcomeDecision::Halt(format!(
                "Tool `{tool}` failed {failures} consecutive times — stop retrying and choose a different approach."
            ));
        }
        if *failures == FAILURE_WARN_THRESHOLD {
            return OutcomeDecision::Warn(format!(
                "Tool `{tool}` has failed {failures} consecutive times this turn."
            ));
        }
        OutcomeDecision::Continue
    }
}

fn hash_args(args: &Value) -> u64 {
    let mut buf = String::new();
    write_canonical(args, &mut buf);
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    buf.hash(&mut hasher);
    hasher.finish()
}

fn write_canonical(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            let _ = write!(out, "{n}");
        }
        Value::String(s) => {
            out.push_str(&serde_json::to_string(s).unwrap_or_default());
        }
        Value::Array(arr) => {
            out.push('[');
            for (i, v) in arr.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(v, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            let mut entries: Vec<_> = map.iter().collect();
            entries.sort_by_key(|(k, _)| *k);
            for (i, (k, v)) in entries.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(k).unwrap_or_default());
                out.push(':');
                write_canonical(v, out);
            }
            out.push('}');
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn third_identical_blocked() {
        let mut g = LoopGuard::default();
        let a = json!({"path": "src/main.rs"});
        assert_eq!(g.record_attempt("rf", &a), AttemptDecision::Proceed);
        assert_eq!(g.record_attempt("rf", &a), AttemptDecision::Proceed);
        assert!(matches!(
            g.record_attempt("rf", &a),
            AttemptDecision::Block(_)
        ));
    }
    #[test]
    fn different_args_proceed() {
        let mut g = LoopGuard::default();
        for o in [0, 100, 200] {
            assert_eq!(
                g.record_attempt("rf", &json!({"p":"a","o":o})),
                AttemptDecision::Proceed
            );
        }
    }
    #[test]
    fn key_order_independent() {
        let mut g = LoopGuard::default();
        g.record_attempt("rf", &json!({"p":"a","o":0}));
        g.record_attempt("rf", &json!({"o":0,"p":"a"}));
        assert!(matches!(
            g.record_attempt("rf", &json!({"p":"a","o":0})),
            AttemptDecision::Block(_)
        ));
    }
    #[test]
    fn failure_warn_halt() {
        let mut g = LoopGuard::default();
        for _ in 0..2 {
            assert_eq!(g.record_outcome("b", false), OutcomeDecision::Continue);
        }
        assert!(matches!(
            g.record_outcome("b", false),
            OutcomeDecision::Warn(_)
        ));
        for _ in 4..8 {
            assert_eq!(g.record_outcome("b", false), OutcomeDecision::Continue);
        }
        assert!(matches!(
            g.record_outcome("b", false),
            OutcomeDecision::Halt(_)
        ));
    }
    #[test]
    fn success_resets() {
        let mut g = LoopGuard::default();
        g.record_outcome("b", false);
        g.record_outcome("b", false);
        g.record_outcome("b", true);
        assert_eq!(g.record_outcome("b", false), OutcomeDecision::Continue);
    }
    #[test]
    fn tools_separate() {
        let mut g = LoopGuard::default();
        let a = json!({"x":1});
        g.record_attempt("rf", &a);
        g.record_attempt("rf", &a);
        assert!(matches!(
            g.record_attempt("rf", &a),
            AttemptDecision::Block(_)
        ));
        assert_eq!(g.record_attempt("bash", &a), AttemptDecision::Proceed);
    }
}
