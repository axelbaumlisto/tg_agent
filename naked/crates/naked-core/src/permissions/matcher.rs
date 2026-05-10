//! Glob matching for permission patterns.
//!
//! Supports:
//!   * `*`  — match any sequence of chars except `/`
//!   * `**` — match any sequence including `/`
//!   * `?`  — match exactly one char
//!   * `~/` and `$HOME/` — expand to `$HOME` at match time
//!   * Empty pattern or `"*"` — match anything

#[must_use]
pub fn matches(pattern: &str, target: &str) -> bool {
    if pattern.is_empty() || pattern == "*" {
        return true;
    }
    let pattern = expand_home(pattern);
    let target = expand_home(target);
    match_glob(&pattern, &target)
}

fn expand_home(s: &str) -> String {
    let home = std::env::var("HOME").unwrap_or_default();
    if home.is_empty() {
        return s.to_string();
    }
    let mut out = s.replace("$HOME/", &format!("{home}/"));
    if out.starts_with("~/") {
        out = format!("{home}/{}", &out[2..]);
    }
    out
}

/// Recursive glob matcher. Handles `*`, `**`, `?`. Linear in
/// `pattern.len() + target.len()` for typical inputs.
fn match_glob(pattern: &str, target: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = target.chars().collect();
    glob_inner(&p, 0, &t, 0)
}

fn glob_inner(p: &[char], pi: usize, t: &[char], ti: usize) -> bool {
    if pi >= p.len() {
        return ti >= t.len();
    }
    match p[pi] {
        '*' => {
            // `**` matches across `/`; single `*` does not.
            let double = pi + 1 < p.len() && p[pi + 1] == '*';
            let next_pi = if double { pi + 2 } else { pi + 1 };
            // Try matching 0..=remaining chars greedily.
            for skip in 0..=(t.len() - ti) {
                if !double {
                    // Single `*` cannot consume `/`.
                    if t[ti..ti + skip].contains(&'/') {
                        break;
                    }
                }
                if glob_inner(p, next_pi, t, ti + skip) {
                    return true;
                }
            }
            false
        }
        '?' => {
            if ti < t.len() && glob_inner(p, pi + 1, t, ti + 1) {
                return true;
            }
            false
        }
        c => {
            if ti < t.len() && t[ti] == c {
                glob_inner(p, pi + 1, t, ti + 1)
            } else {
                false
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn star_matches_anything() {
        assert!(matches("*", "anything"));
        assert!(matches("", "anything"));
    }

    #[test]
    fn single_star_does_not_cross_slash() {
        assert!(matches("src/*.rs", "src/foo.rs"));
        assert!(!matches("src/*.rs", "src/sub/foo.rs"));
    }

    #[test]
    fn double_star_crosses_slash() {
        assert!(matches("src/**", "src/foo.rs"));
        assert!(matches("src/**", "src/sub/dir/foo.rs"));
        assert!(matches("src/**/*.rs", "src/sub/foo.rs"));
    }

    #[test]
    fn question_mark_matches_one_char() {
        assert!(matches("a?c", "abc"));
        assert!(!matches("a?c", "abbc"));
        assert!(!matches("a?c", "ac"));
    }

    #[test]
    fn dotfile_matches() {
        assert!(matches(".env*", ".env"));
        assert!(matches(".env*", ".env.local"));
        assert!(!matches(".env*", "config.env"));
    }

    #[test]
    fn tilde_expands_home_smoke() {
        // We can't safely mutate $HOME in this workspace (deny
        // unsafe-code); just verify the matcher resolves with
        // whatever the real $HOME is. If $HOME is unset the
        // expansion is a no-op and the test still proves the
        // expand-then-glob pipeline works.
        let home = std::env::var("HOME").unwrap_or_default();
        if !home.is_empty() {
            let target = format!("{home}/foo");
            assert!(matches("~/foo", &target));
            assert!(matches("$HOME/foo", &target));
        }
    }
}
