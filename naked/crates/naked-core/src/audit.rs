//! Append-only JSONL audit log — best-effort, never panics.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

pub fn log_event(audit_dir: &Path, event: &str, details: Value) {
    if let Err(e) = append(audit_dir, event, details) {
        tracing::warn!("audit write failed: {e}");
    }
}

fn append(audit_dir: &Path, event: &str, details: Value) -> std::io::Result<()> {
    fs::create_dir_all(audit_dir)?;
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path(audit_dir))?;
    let record =
        json!({ "ts": chrono::Utc::now().to_rfc3339(), "event": event, "details": details });
    writeln!(f, "{}", serde_json::to_string(&record).unwrap_or_default())?;
    f.flush()
}

pub fn path(audit_dir: &Path) -> PathBuf {
    audit_dir.join("audit.jsonl")
}

pub fn recent(audit_dir: &Path, n: usize) -> Vec<Value> {
    let content = fs::read_to_string(path(audit_dir)).unwrap_or_default();
    content
        .lines()
        .rev()
        .take(n)
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_and_read() {
        let tmp = tempfile::tempdir().unwrap();
        log_event(tmp.path(), "write", json!({"file": "a.rs"}));
        log_event(tmp.path(), "exec", json!({"cmd": "test"}));
        let e = recent(tmp.path(), 10);
        assert_eq!(e.len(), 2);
        assert_eq!(e[0]["event"], "exec");
    }

    #[test]
    fn recent_limits() {
        let tmp = tempfile::tempdir().unwrap();
        for i in 0..5 {
            log_event(tmp.path(), &format!("e{i}"), json!({}));
        }
        assert_eq!(recent(tmp.path(), 2).len(), 2);
    }

    #[test]
    fn empty_no_panic() {
        assert!(recent(tempfile::tempdir().unwrap().path(), 10).is_empty());
    }
}
