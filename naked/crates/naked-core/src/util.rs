//! String-boundary utilities.
//!
//! These exist to DRY the `floor_char_boundary` pattern that otherwise
//! gets re-implemented ad-hoc at every truncation site. See B48 in
//! `docs/BUG_REGISTRY.md`.

/// Slice `s` to at most `max_bytes` bytes, snapping to a char boundary.
/// Zero-alloc: returns a sub-slice.
#[inline]
pub fn head_truncate(s: &str, max_bytes: usize) -> &str {
    &s[..s.floor_char_boundary(max_bytes)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_within_budget() {
        assert_eq!(head_truncate("hello", 10), "hello");
    }

    #[test]
    fn ascii_over_budget() {
        assert_eq!(head_truncate("hello world", 5), "hello");
    }

    #[test]
    fn cyrillic_at_boundary() {
        // Each Cyrillic letter is 2 bytes in UTF-8.
        // "Добро" = 10 bytes. Cutting at 9 must not panic and must
        // snap back to byte 8 (end of 'б').
        let s = "Доброе утро всем)";
        let result = head_truncate(s, 9);
        assert_eq!(result, "Добр"); // 4 Cyrillic chars = 8 bytes
        assert!(result.is_char_boundary(result.len()));
    }

    #[test]
    fn emoji_at_boundary() {
        // '💭' = 4 bytes (U+1F4AD). Cutting at 1, 2, or 3 must snap to 0.
        let s = "💭hello";
        assert_eq!(head_truncate(s, 1), "");
        assert_eq!(head_truncate(s, 2), "");
        assert_eq!(head_truncate(s, 3), "");
        assert_eq!(head_truncate(s, 4), "💭");
        assert_eq!(head_truncate(s, 5), "💭h");
    }

    #[test]
    fn empty_string() {
        assert_eq!(head_truncate("", 100), "");
    }

    #[test]
    fn zero_budget() {
        assert_eq!(head_truncate("hello", 0), "");
        assert_eq!(head_truncate("Добро", 0), "");
    }

    #[test]
    fn mixed_multibyte_at_exact_boundary() {
        // Mix of ASCII (1-byte), Cyrillic (2-byte), emoji (4-byte).
        let s = "aБ💭x";
        // bytes: a(1) Б(2) 💭(4) x(1) = offsets 0,1,3,7,8
        assert_eq!(head_truncate(s, 0), "");
        assert_eq!(head_truncate(s, 1), "a");
        assert_eq!(head_truncate(s, 2), "a"); // mid-Б → snap to 1
        assert_eq!(head_truncate(s, 3), "aБ");
        assert_eq!(head_truncate(s, 4), "aБ"); // mid-💭
        assert_eq!(head_truncate(s, 5), "aБ"); // mid-💭
        assert_eq!(head_truncate(s, 6), "aБ"); // mid-💭
        assert_eq!(head_truncate(s, 7), "aБ💭");
        assert_eq!(head_truncate(s, 8), "aБ💭x");
        assert_eq!(head_truncate(s, 100), "aБ💭x");
    }

    /// Reproduces the exact production panic: 80-byte cut on Cyrillic text.
    #[test]
    fn production_panic_repro_delta_rs_327() {
        // Build a Cyrillic string longer than 80 bytes.
        // "Доброе утро всем)" repeated → each Cyrillic char = 2 bytes.
        let long = "Доброе утро всем) ".repeat(10); // ~180 bytes
        assert!(long.len() > 80);
        // This would panic with &long[..80] if byte 80 is mid-char.
        let result = head_truncate(&long, 80);
        assert!(result.len() <= 80);
        // Must be valid UTF-8 (it's a &str, so always true, but let's be explicit)
        assert!(std::str::from_utf8(result.as_bytes()).is_ok());
    }
}
