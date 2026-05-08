//! Model-scope glob filtering.
//!
//! Provides `filter_models_by_scope` for selecting provider/model pairs by
//! glob patterns. Lives here rather than in `markup` because it is unrelated
//! to Telegram message formatting.

/// Filter `provider/model` pairs by glob patterns.
///
/// Patterns support `*` (any chars) and `?` (single char).
/// Empty patterns → return all (backward compatible).
pub fn filter_models_by_scope<'a>(
    models: &'a [(String, String)],
    patterns: &[String],
) -> Vec<&'a (String, String)> {
    if patterns.is_empty() {
        return models.iter().collect();
    }
    models
        .iter()
        .filter(|(prov, model)| {
            let full = format!("{prov}/{model}");
            patterns.iter().any(|pat| glob_match(pat, &full))
        })
        .collect()
}

/// Simple glob matching: `*` = any chars, `?` = single char.
pub fn glob_match(pattern: &str, text: &str) -> bool {
    let mut p = pattern.chars().peekable();
    let mut t = text.chars().peekable();
    glob_match_inner(&mut p, &mut t)
}

pub fn glob_match_inner(
    p: &mut std::iter::Peekable<std::str::Chars<'_>>,
    t: &mut std::iter::Peekable<std::str::Chars<'_>>,
) -> bool {
    while let Some(&pc) = p.peek() {
        match pc {
            '*' => {
                p.next();
                // Try matching rest of pattern at every position
                if p.peek().is_none() {
                    return true; // trailing * matches everything
                }
                let mut t_clone = t.clone();
                loop {
                    let mut p_clone = p.clone();
                    let mut tc = t_clone.clone();
                    if glob_match_inner(&mut p_clone, &mut tc) {
                        return true;
                    }
                    if t_clone.next().is_none() {
                        return false;
                    }
                }
            }
            '?' => {
                p.next();
                if t.next().is_none() {
                    return false;
                }
            }
            c => {
                p.next();
                match t.next() {
                    Some(tc) if tc == c => {}
                    _ => return false,
                }
            }
        }
    }
    t.peek().is_none()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── filter_models_by_scope ──────────────────────────────────────

    #[test]
    fn scope_empty_returns_all() {
        let models = vec![("a".into(), "m1".into()), ("b".into(), "m2".into())];
        assert_eq!(filter_models_by_scope(&models, &[]).len(), 2);
    }

    #[test]
    fn scope_exact_match() {
        let models = vec![
            ("anthropic".into(), "claude".into()),
            ("openai".into(), "gpt4".into()),
        ];
        let filtered = filter_models_by_scope(&models, &["anthropic/claude".into()]);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].1, "claude");
    }

    #[test]
    fn scope_wildcard() {
        let models = vec![
            ("anthropic".into(), "claude-3".into()),
            ("anthropic".into(), "claude-4".into()),
            ("openai".into(), "gpt-4".into()),
        ];
        let filtered = filter_models_by_scope(&models, &["anthropic/*".into()]);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn scope_question_mark() {
        let models = vec![
            ("a".into(), "v1".into()),
            ("a".into(), "v2".into()),
            ("a".into(), "v10".into()),
        ];
        let filtered = filter_models_by_scope(&models, &["a/v?".into()]);
        assert_eq!(filtered.len(), 2); // v1, v2 match; v10 doesn't
    }

    #[test]
    fn scope_no_match() {
        let models = vec![("a".into(), "m1".into())];
        let filtered = filter_models_by_scope(&models, &["b/*".into()]);
        assert!(filtered.is_empty());
    }

    #[test]
    fn scope_multiple_patterns() {
        let models = vec![
            ("a".into(), "m1".into()),
            ("b".into(), "m2".into()),
            ("c".into(), "m3".into()),
        ];
        let filtered = filter_models_by_scope(&models, &["a/*".into(), "c/*".into()]);
        assert_eq!(filtered.len(), 2);
    }
}
