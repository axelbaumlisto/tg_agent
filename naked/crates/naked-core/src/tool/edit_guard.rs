//! Shared pre-write stale-edit validators.
//!
//! This module is deliberately pure: callers provide the live file bytes and
//! the expectations they received from the model/snapshot layer. It performs no
//! file I/O, locking, atomic writes, or ToolResult mapping so `edit_file`,
//! `apply_patch`, and future hashline modes can all reuse it.

use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct EditExpectations {
    pub expected_content_sha: Option<String>,
    pub range_anchors: Vec<RangeAnchor>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RangeAnchor {
    pub start_line: usize,
    pub end_line: usize,
    pub anchor_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub(crate) enum StaleEditError {
    #[error("expected_content_sha mismatch: expected {expected}, live {actual}")]
    ContentShaMismatch { expected: String, actual: String },
    #[error(
        "range anchor line bounds are invalid: start_line={start_line}, end_line={end_line}, line_count={line_count}"
    )]
    RangeAnchorOutOfBounds {
        start_line: usize,
        end_line: usize,
        line_count: usize,
    },
    #[error(
        "range anchor mismatch at lines {start_line}-{end_line}: expected {expected}, live {actual}"
    )]
    RangeAnchorMismatch {
        start_line: usize,
        end_line: usize,
        expected: String,
        actual: String,
    },
    #[error(
        "range anchor is ambiguous at lines {start_line}-{end_line}: hash {anchor_hash} matches {matches} ranges"
    )]
    RangeAnchorAmbiguous {
        start_line: usize,
        end_line: usize,
        anchor_hash: String,
        matches: usize,
    },
}

pub(crate) fn validate_live_content(
    live_content: &str,
    expectations: &EditExpectations,
) -> Result<(), StaleEditError> {
    if let Some(expected) = expectations.expected_content_sha.as_deref() {
        let actual = sha256_hex(live_content.as_bytes());
        if !expected.eq_ignore_ascii_case(&actual) {
            return Err(StaleEditError::ContentShaMismatch {
                expected: expected.to_string(),
                actual,
            });
        }
    }

    for anchor in &expectations.range_anchors {
        validate_range_anchor(live_content, anchor)?;
    }

    Ok(())
}

pub(crate) fn line_spans(content: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut start = 0usize;
    for segment in content.split_inclusive('\n') {
        let end = start + segment.len();
        spans.push((start, end));
        start = end;
    }
    if start < content.len() {
        spans.push((start, content.len()));
    }
    spans
}

pub(crate) fn range_text_by_line(
    content: &str,
    start_line: usize,
    end_line: usize,
) -> Option<&str> {
    let spans = line_spans(content);
    if start_line == 0 || end_line < start_line || end_line > spans.len() {
        return None;
    }
    let start = spans[start_line - 1].0;
    let end = spans[end_line - 1].1;
    Some(&content[start..end])
}

fn validate_range_anchor(live_content: &str, anchor: &RangeAnchor) -> Result<(), StaleEditError> {
    let spans = line_spans(live_content);
    if anchor.start_line == 0
        || anchor.end_line < anchor.start_line
        || anchor.end_line > spans.len()
    {
        return Err(StaleEditError::RangeAnchorOutOfBounds {
            start_line: anchor.start_line,
            end_line: anchor.end_line,
            line_count: spans.len(),
        });
    }

    let live_slice = range_text_by_line(live_content, anchor.start_line, anchor.end_line)
        .expect("bounds checked above");
    let actual = sha256_hex(live_slice.as_bytes());
    if !anchor.anchor_hash.eq_ignore_ascii_case(&actual) {
        return Err(StaleEditError::RangeAnchorMismatch {
            start_line: anchor.start_line,
            end_line: anchor.end_line,
            expected: anchor.anchor_hash.clone(),
            actual,
        });
    }

    let range_len = anchor.end_line - anchor.start_line + 1;
    let matches = count_matching_range_hashes(live_content, range_len, &anchor.anchor_hash);
    if matches > 1 {
        return Err(StaleEditError::RangeAnchorAmbiguous {
            start_line: anchor.start_line,
            end_line: anchor.end_line,
            anchor_hash: anchor.anchor_hash.clone(),
            matches,
        });
    }

    Ok(())
}

fn count_matching_range_hashes(content: &str, range_len: usize, hash: &str) -> usize {
    let spans = line_spans(content);
    if range_len == 0 || range_len > spans.len() {
        return 0;
    }
    spans
        .windows(range_len)
        .filter(|window| {
            let start = window[0].0;
            let end = window[range_len - 1].1;
            sha256_hex(&content.as_bytes()[start..end]).eq_ignore_ascii_case(hash)
        })
        .count()
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edit_guard_validator_unit() {
        let content = "alpha\nbeta\n";
        let matching = EditExpectations {
            expected_content_sha: Some(sha256_hex(content.as_bytes())),
            ..Default::default()
        };
        assert_eq!(validate_live_content(content, &matching), Ok(()));

        let mismatch = EditExpectations {
            expected_content_sha: Some(sha256_hex(b"different")),
            ..Default::default()
        };
        assert!(matches!(
            validate_live_content(content, &mismatch),
            Err(StaleEditError::ContentShaMismatch { .. })
        ));
    }

    #[test]
    fn hashline_validation_reuses_edit_guard() {
        let content = "alpha\nbeta\ngamma\n";
        let anchor_text = range_text_by_line(content, 2, 2).unwrap();
        let expectations = EditExpectations {
            expected_content_sha: None,
            range_anchors: vec![RangeAnchor {
                start_line: 2,
                end_line: 2,
                anchor_hash: sha256_hex(anchor_text.as_bytes()),
            }],
        };
        assert_eq!(validate_live_content(content, &expectations), Ok(()));

        let stale = EditExpectations {
            expected_content_sha: None,
            range_anchors: vec![RangeAnchor {
                start_line: 2,
                end_line: 2,
                anchor_hash: sha256_hex(b"old beta\n"),
            }],
        };
        assert!(matches!(
            validate_live_content(content, &stale),
            Err(StaleEditError::RangeAnchorMismatch { .. })
        ));
    }
}
