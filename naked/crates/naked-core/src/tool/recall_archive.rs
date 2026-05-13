//! `recall_archive` — BM25 search over past cycle archives.
//!
//! When cycle_restart discards history, this tool searches archived JSONL.

use std::collections::HashMap;
use std::path::Path;

use crate::types::{Permission, ToolResult, ToolSpec};

const DEFAULT_MAX: usize = 3;
const EXCERPT_LEN: usize = 240;

// ── BM25 engine ─────────────────────────────────────────────────────────────

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|s| s.len() >= 2)
        .map(String::from)
        .collect()
}

fn bm25_score(
    query_tokens: &[String],
    doc_tokens: &[String],
    avg_dl: f64,
    n_docs: usize,
    df: &HashMap<String, usize>,
) -> f64 {
    let k1: f64 = 1.5;
    let b: f64 = 0.75;
    let dl = doc_tokens.len() as f64;
    let mut score = 0.0;
    let tf_map: HashMap<&str, usize> = {
        let mut m = HashMap::new();
        for t in doc_tokens {
            *m.entry(t.as_str()).or_insert(0) += 1;
        }
        m
    };
    for qt in query_tokens {
        let tf = *tf_map.get(qt.as_str()).unwrap_or(&0) as f64;
        let doc_freq = *df.get(qt.as_str()).unwrap_or(&0) as f64;
        if doc_freq == 0.0 {
            continue;
        }
        let idf = ((n_docs as f64 - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0).ln();
        let tf_norm = (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * dl / avg_dl));
        score += idf * tf_norm;
    }
    score
}

// ── Archive reading ─────────────────────────────────────────────────────────

struct ArchiveDoc {
    cycle: u32,
    index: usize,
    role: String,
    text: String,
}

fn read_archives(sessions_root: &Path, session_id: &str) -> Vec<ArchiveDoc> {
    let cycles_dir = sessions_root.join(session_id).join("cycles");
    let mut docs = Vec::new();
    // REGISTRY-WAIVE: intentional fallback: missing path → empty result
    let Ok(entries) = std::fs::read_dir(&cycles_dir) else {
        return docs;
    };
    let mut files: Vec<_> = entries
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x == "jsonl"))
        .collect();
    files.sort_by_key(|e| e.file_name());
    for (cycle_idx, entry) in files.iter().enumerate() {
        // REGISTRY-WAIVE: intentional fallback: missing path → empty result
        let Ok(content) = std::fs::read_to_string(entry.path()) else {
            continue;
        };
        for (line_idx, line) in content.lines().enumerate() {
            // REGISTRY-WAIVE: intentional fallback: malformed entry → skip
            let Ok(msg) = serde_json::from_str::<serde_json::Value>(line) else {
                continue;
            };
            let role = msg
                .get("role")
                .and_then(|v| v.as_str())
                .unwrap_or("?")
                .to_string();
            let text = extract_text(&msg);
            if !text.is_empty() {
                docs.push(ArchiveDoc {
                    cycle: cycle_idx as u32,
                    index: line_idx,
                    role,
                    text,
                });
            }
        }
    }
    docs
}

fn extract_text(msg: &serde_json::Value) -> String {
    // Try content as string:
    if let Some(s) = msg.get("content").and_then(|v| v.as_str()) {
        return s.to_string();
    }
    // Try content as array of blocks:
    if let Some(arr) = msg.get("content").and_then(|v| v.as_array()) {
        return arr
            .iter()
            .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join(" ");
    }
    String::new()
}

fn excerpt(text: &str, query_tokens: &[String]) -> String {
    // Find first occurrence of any query token:
    let lower = text.to_lowercase();
    let pos = query_tokens
        .iter()
        .filter_map(|q| lower.find(q.as_str()))
        .min()
        .unwrap_or(0);
    let start = pos.saturating_sub(EXCERPT_LEN / 2);
    let end = (start + EXCERPT_LEN).min(text.len());
    let start = text.floor_char_boundary(start);
    let end = text.floor_char_boundary(end);
    let mut s = String::new();
    if start > 0 {
        s.push('…');
    }
    s.push_str(&text[start..end]);
    if end < text.len() {
        s.push('…');
    }
    s
}

// ── Tool ────────────────────────────────────────────────────────────────────

pub struct RecallArchiveTool {
    sessions_root: std::path::PathBuf,
}

impl RecallArchiveTool {
    pub fn new(sessions_root: impl Into<std::path::PathBuf>) -> Self {
        Self {
            sessions_root: sessions_root.into(),
        }
    }
}

#[async_trait::async_trait]
impl crate::tool::Tool for RecallArchiveTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "recall_archive".into(),
            description: "Search prior context cycles for forgotten content. Uses BM25 ranking. \
                          Use when your briefing missed something from earlier in the conversation."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Search query" },
                    "session_id": { "type": "string", "description": "Session to search (default: current)" },
                    "max_results": { "type": "integer", "description": "Max hits (default 3)" }
                },
                "required": ["query"]
            }),
            permission: Permission::ReadOnly,
        }
    }

    async fn execute(&self, input: serde_json::Value, _cwd: &Path) -> ToolResult {
        let query = match input.get("query").and_then(|v| v.as_str()) {
            Some(q) if !q.is_empty() => q,
            _ => {
                return ToolResult::err("query required");
            }
        };
        let session_id = input
            .get("session_id")
            .and_then(|v| v.as_str())
            .unwrap_or("current");
        let max = input
            .get("max_results")
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_MAX as u64)
            .min(10) as usize;

        let docs = read_archives(&self.sessions_root, session_id);
        if docs.is_empty() {
            return ToolResult::ok("No archived cycles found");
        }

        let query_tokens = tokenize(query);
        if query_tokens.is_empty() {
            return ToolResult::err("Query too short");
        }

        // Build DF table:
        let n = docs.len();
        let doc_token_sets: Vec<Vec<String>> = docs.iter().map(|d| tokenize(&d.text)).collect();
        let mut df: HashMap<String, usize> = HashMap::new();
        for tokens in &doc_token_sets {
            let uniq: std::collections::HashSet<&str> = tokens.iter().map(|s| s.as_str()).collect();
            for t in uniq {
                *df.entry(t.to_string()).or_insert(0) += 1;
            }
        }
        let avg_dl = doc_token_sets.iter().map(|t| t.len()).sum::<usize>() as f64 / n as f64;

        // Score + rank:
        let mut scored: Vec<(usize, f64)> = doc_token_sets
            .iter()
            .enumerate()
            .map(|(i, tokens)| (i, bm25_score(&query_tokens, tokens, avg_dl, n, &df)))
            .filter(|(_, s)| *s > 0.0)
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if scored.is_empty() {
            return ToolResult::ok(format!("No matches for '{query}'"));
        }

        let mut output = format!("{} matches for '{query}':\n\n", scored.len().min(max));
        for (i, score) in scored.iter().take(max) {
            let doc = &docs[*i];
            let exc = excerpt(&doc.text, &query_tokens);
            output.push_str(&format!(
                "[cycle {} msg {} ({})] score={:.2}\n  {exc}\n\n",
                doc.cycle, doc.index, doc.role, score
            ));
        }

        ToolResult::ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_basic() {
        let t = tokenize("Hello, world! fn main()");
        assert!(t.contains(&"hello".to_string()));
        assert!(t.contains(&"main".to_string()));
        assert!(!t.iter().any(|s| s.len() < 2)); // no single chars
    }

    #[test]
    fn bm25_relevance() {
        let q = tokenize("rust compiler");
        let doc_a = tokenize("the rust compiler is fast and efficient");
        let doc_b = tokenize("python is a great language for beginners");
        let mut df = HashMap::new();
        for t in &doc_a {
            *df.entry(t.clone()).or_insert(0) += 1;
        }
        for t in &doc_b {
            *df.entry(t.clone()).or_insert(0) += 1;
        }
        let avg = (doc_a.len() + doc_b.len()) as f64 / 2.0;
        let sa = bm25_score(&q, &doc_a, avg, 2, &df);
        let sb = bm25_score(&q, &doc_b, avg, 2, &df);
        assert!(sa > sb, "relevant doc should score higher: {sa} vs {sb}");
    }

    #[test]
    fn excerpt_centers() {
        let text = "a".repeat(500);
        let exc = excerpt(&text, &["aaa".to_string()]);
        assert!(exc.len() <= EXCERPT_LEN + 10); // +2 for ellipsis
    }

    #[test]
    fn empty_archive() {
        let tmp = tempfile::tempdir().unwrap();
        let docs = read_archives(tmp.path(), "nonexistent");
        assert!(docs.is_empty());
    }

    #[test]
    fn archive_round_trip() {
        let tmp = tempfile::tempdir().unwrap();
        let cycle_dir = tmp.path().join("sess1").join("cycles");
        std::fs::create_dir_all(&cycle_dir).unwrap();
        std::fs::write(
            cycle_dir.join("0.jsonl"),
            r#"{"role":"user","content":"hello world"}
{"role":"assistant","content":"hi there"}"#,
        )
        .unwrap();
        let docs = read_archives(tmp.path(), "sess1");
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].role, "user");
    }

    #[test]
    fn extract_text_array() {
        let msg = serde_json::json!({"role":"assistant","content":[{"type":"text","text":"block one"},{"type":"text","text":"block two"}]});
        assert_eq!(extract_text(&msg), "block one block two");
    }
}
