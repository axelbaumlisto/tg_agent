//! Large output router — truncate oversized tool results.
//!
//! When a tool returns > `THRESHOLD_CHARS`, the output is truncated to
//! head + tail with a summary in between. The model sees enough context
//! to understand the result without burning the entire context window.
//!
//! No LLM call needed (KISS). Smart truncation preserves first and last
//! lines which are usually the most informative.

/// Default threshold: ~4K tokens ≈ 12K chars.
pub const DEFAULT_THRESHOLD_CHARS: usize = 12_000;

/// How many chars to keep from the head.
const HEAD_CHARS: usize = 4_000;
/// How many chars to keep from the tail.
const TAIL_CHARS: usize = 2_000;

// ── model-aware limits ────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputLimits {
    pub hard_limit_chars: usize,
    pub noisy_soft_limit_chars: usize,
    pub snippet_chars: usize,
}

pub fn limits_for_context_window(window_tokens: u64) -> OutputLimits {
    if window_tokens >= 500_000 {
        OutputLimits {
            hard_limit_chars: 180_000,
            noisy_soft_limit_chars: 60_000,
            snippet_chars: 40_000,
        }
    } else if window_tokens >= 100_000 {
        OutputLimits {
            hard_limit_chars: 24_000,
            noisy_soft_limit_chars: 8_000,
            snippet_chars: 4_000,
        }
    } else {
        OutputLimits {
            hard_limit_chars: 12_000,
            noisy_soft_limit_chars: 2_000,
            snippet_chars: 900,
        }
    }
}

fn is_noisy_tool(name: &str) -> bool {
    matches!(
        name,
        "bash" | "web_search" | "web_fetch" | "multi_search" | "file_search"
    )
}

pub fn route_large_output_aware(
    output: &str,
    tool_name: &str,
    context_window_tokens: u64,
) -> String {
    let limits = limits_for_context_window(context_window_tokens);
    let threshold = if is_noisy_tool(tool_name) {
        limits.noisy_soft_limit_chars
    } else {
        limits.hard_limit_chars
    };
    route_large_output(output, threshold)
}

/// Truncate tool output if it exceeds the threshold.
/// Returns the original if under threshold, or truncated version.
pub fn route_large_output(output: &str, threshold: usize) -> String {
    if output.len() <= threshold {
        return output.to_string();
    }

    let total = output.len();
    let head_end = snap_to_char_boundary(output, HEAD_CHARS.min(total));
    let tail_start = snap_to_char_boundary_back(output, total.saturating_sub(TAIL_CHARS));

    // Count lines in the omitted section:
    let omitted = &output[head_end..tail_start];
    let omitted_lines = omitted.lines().count();
    let omitted_bytes = omitted.len();

    format!(
        "{}\n\n[… {omitted_lines} lines, {omitted_bytes} bytes omitted — use web_fetch or read_file for full content …]\n\n{}",
        &output[..head_end],
        &output[tail_start..],
    )
}

fn snap_to_char_boundary(s: &str, pos: usize) -> usize {
    let mut end = pos.min(s.len());
    // Ensure we're at a char boundary before slicing
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    // Snap forward to next newline for cleaner cut:
    if let Some(nl) = s[end..].find('\n')
        && nl < 200
    {
        return (end + nl + 1).min(s.len());
    }
    end
}

fn snap_to_char_boundary_back(s: &str, pos: usize) -> usize {
    let mut start = pos.min(s.len());
    // Ensure we're at a char boundary before slicing
    while start < s.len() && !s.is_char_boundary(start) {
        start += 1;
    }
    // Snap back to previous newline:
    if start > 0
        && s.is_char_boundary(start)
        && let Some(nl) = s[..start].rfind('\n')
        && start - nl < 200
    {
        return nl + 1;
    }
    start
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_output_unchanged() {
        let out = "hello world";
        assert_eq!(route_large_output(out, 100), out);
    }

    #[test]
    fn long_output_truncated() {
        let out = "line1\n".repeat(5000); // 30K chars
        let result = route_large_output(&out, 12_000);
        assert!(result.len() < 12_000);
        assert!(result.contains("omitted"));
        assert!(result.starts_with("line1\n"));
        assert!(result.ends_with("line1\n"));
    }

    #[test]
    fn preserves_head_and_tail() {
        let head = "HEAD_MARKER\n".repeat(100);
        let middle = "middle\n".repeat(5000);
        let tail = "TAIL_MARKER\n".repeat(100);
        let out = format!("{head}{middle}{tail}");
        let result = route_large_output(&out, 12_000);
        assert!(result.contains("HEAD_MARKER"));
        assert!(result.contains("TAIL_MARKER"));
        assert!(result.contains("omitted"));
    }

    #[test]
    fn omitted_count_accurate() {
        let out = "x\n".repeat(10_000); // 20K chars
        let result = route_large_output(&out, 8_000);
        assert!(result.contains("lines"));
        assert!(result.contains("bytes omitted"));
    }

    #[test]
    fn utf8_safe() {
        let out = "Привет мир\n".repeat(2000); // ~40K bytes
        let result = route_large_output(&out, 12_000);
        // Must be valid UTF-8:
        assert!(result.len() < 15_000);
        assert!(result.contains("omitted"));
        // Verify no partial chars:
        for (i, _) in result.char_indices() {
            assert!(result.is_char_boundary(i));
        }
    }

    #[test]
    fn exact_threshold_unchanged() {
        let out = "x".repeat(12_000);
        assert_eq!(route_large_output(&out, 12_000).len(), 12_000);
    }

    #[test]
    fn small_window_aggressive_truncation() {
        let out = "x\n".repeat(8_000);
        let result = route_large_output_aware(&out, "read_file", 32_000);
        assert!(result.contains("omitted"));
    }

    #[test]
    fn large_window_generous_limits() {
        let out = "x\n".repeat(20_000);
        let result = route_large_output_aware(&out, "read_file", 1_000_000);
        assert!(!result.contains("omitted"));
    }

    #[test]
    fn noisy_tool_hits_soft_limit() {
        let out = "x\n".repeat(5_000);
        assert!(route_large_output_aware(&out, "bash", 128_000).contains("omitted"));
        assert!(!route_large_output_aware(&out, "read_file", 128_000).contains("omitted"));
    }

    #[test]
    fn unknown_tool_uses_hard_limit() {
        let out = "x\n".repeat(15_000);
        assert!(route_large_output_aware(&out, "custom_tool", 128_000).contains("omitted"));
    }

    #[test]
    fn utf8_boundary_at_head_cut() {
        // Build a string where HEAD_CHARS (4000 bytes) lands inside a
        // multi-byte char. 'Đ' is 2 bytes; Cyrillic 'е' is 2 bytes.
        let prefix = "a".repeat(3999); // 3999 ASCII bytes
        let cyrillic = "е".repeat(5000); // 10000 bytes of 2-byte chars
        let out = format!("{prefix}{cyrillic}");
        // Should not panic:
        let result = route_large_output(&out, 6_000);
        assert!(result.contains("omitted"));
        // Verify valid UTF-8 by iterating:
        for (i, _) in result.char_indices() {
            assert!(result.is_char_boundary(i));
        }
    }

    #[test]
    fn utf8_boundary_at_tail_cut() {
        // Tail cut at a position inside a multi-byte char.
        let body = "日本語テスト\n".repeat(2000); // heavy multi-byte
        let result = route_large_output(&body, 6_000);
        assert!(result.contains("omitted"));
        for (i, _) in result.char_indices() {
            assert!(result.is_char_boundary(i));
        }
    }
}
