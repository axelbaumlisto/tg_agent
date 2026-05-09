//! Command arity dictionary — canonical prefix extraction for approval cache.
//!
//! `git status -s --porcelain` → `git status`.  Flags stripped, arity-aware.

use std::collections::HashMap;
use std::sync::LazyLock;

static ARITY: LazyLock<HashMap<&'static str, u8>> = LazyLock::new(|| {
    HashMap::from([
        // git (arity 2 = base + subcommand)
        ("git add", 2),
        ("git status", 2),
        ("git diff", 2),
        ("git log", 2),
        ("git show", 2),
        ("git branch", 2),
        ("git checkout", 2),
        ("git switch", 2),
        ("git commit", 2),
        ("git push", 2),
        ("git pull", 2),
        ("git fetch", 2),
        ("git merge", 2),
        ("git rebase", 2),
        ("git stash", 2),
        ("git tag", 2),
        ("git remote", 2),
        ("git clean", 2),
        ("git reset", 2),
        ("git rev-parse", 2),
        ("git ls-files", 2),
        ("git blame", 2),
        ("git cherry-pick", 2),
        // cargo
        ("cargo build", 2),
        ("cargo test", 2),
        ("cargo check", 2),
        ("cargo run", 2),
        ("cargo clippy", 2),
        ("cargo fmt", 2),
        ("cargo bench", 2),
        ("cargo doc", 2),
        ("cargo add", 2),
        ("cargo remove", 2),
        ("cargo update", 2),
        // npm/node
        ("npm install", 2),
        ("npm run", 2),
        ("npm test", 2),
        ("npm start", 2),
        ("npx", 1),
        ("node", 1),
        // docker (compose = arity 3)
        ("docker build", 2),
        ("docker run", 2),
        ("docker exec", 2),
        ("docker ps", 2),
        ("docker logs", 2),
        ("docker stop", 2),
        ("docker compose up", 3),
        ("docker compose down", 3),
        // system
        ("ls", 1),
        ("cat", 1),
        ("head", 1),
        ("tail", 1),
        ("wc", 1),
        ("grep", 1),
        ("rg", 1),
        ("find", 1),
        ("which", 1),
        ("echo", 1),
        ("pwd", 1),
        ("date", 1),
        ("df", 1),
        ("du", 1),
        ("free", 1),
        ("ps", 1),
        ("uname", 1),
        ("curl", 1),
        ("wget", 1),
        ("make", 1),
        ("python3", 1),
        ("pip install", 2),
        ("pip3 install", 2),
        ("systemctl status", 2),
        ("systemctl restart", 2),
        ("journalctl", 1),
    ])
});

const READONLY: &[&str] = &[
    "ls",
    "cat",
    "head",
    "tail",
    "wc",
    "grep",
    "rg",
    "find",
    "which",
    "echo",
    "pwd",
    "date",
    "df",
    "du",
    "free",
    "ps",
    "uname",
    "git status",
    "git diff",
    "git log",
    "git show",
    "git branch",
    "git rev-parse",
    "git ls-files",
    "git remote",
    "git blame",
    "cargo check",
    "cargo clippy",
    "cargo test",
    "cargo doc",
    "npm test",
    "docker ps",
    "docker logs",
    "systemctl status",
    "journalctl",
];

/// Extract canonical prefix: strip flags, match longest arity entry.
pub fn canonical_prefix(command: &str) -> String {
    let positional: Vec<&str> = command
        .split_whitespace()
        .filter(|t| !t.starts_with('-'))
        .collect();
    if positional.is_empty() {
        return String::new();
    }
    for len in (1..=3.min(positional.len())).rev() {
        let key = positional[..len].join(" ").to_lowercase();
        if ARITY.contains_key(key.as_str()) {
            return key;
        }
    }
    positional[0].to_lowercase()
}

/// True if prefix is read-only (safe to auto-approve without asking user).
pub fn is_readonly(prefix: &str) -> bool {
    READONLY.contains(&prefix)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn git_flags() {
        assert_eq!(canonical_prefix("git status -s"), "git status");
        assert_eq!(canonical_prefix("git status --porcelain"), "git status");
    }
    #[test]
    fn cargo_verbose() {
        assert_eq!(
            canonical_prefix("cargo test --workspace --verbose"),
            "cargo test"
        );
    }
    #[test]
    fn docker_comp() {
        assert_eq!(
            canonical_prefix("docker compose up -d"),
            "docker compose up"
        );
    }
    #[test]
    fn simple() {
        assert_eq!(canonical_prefix("ls -la /tmp"), "ls");
        assert_eq!(canonical_prefix("grep -rn pat src/"), "grep");
    }
    #[test]
    fn unknown() {
        assert_eq!(canonical_prefix("myapp --flag arg"), "myapp");
    }
    #[test]
    fn readonly() {
        assert!(is_readonly("git status"));
        assert!(is_readonly("cargo test"));
        assert!(!is_readonly("git push"));
        assert!(!is_readonly("rm"));
    }
}

#[cfg(test)]
mod proptests {
    //! Property-based tests for `canonical_prefix` (T13 of
    //! PLAN_CORE_HARDENING_v2). The function must be:
    //!   1. **Pure**: same input → same output across runs.
    //!   2. **Lower-cased**: result is always lowercase ASCII.
    //!   3. **Subset of input**: the result tokens, joined by
    //!      whitespace, appear (case-insensitively) at the start of
    //!      the input's positional argv.
    //!   4. **Flag-stable**: prefixing or suffixing GNU-style `--flag`
    //!      tokens never changes the output.
    //!   5. **Idempotent**: feeding the result back yields the result.

    use super::*;
    use proptest::prelude::*;

    /// Generate a positional command word: lower/upper alpha-ish,
    /// no spaces, no leading dash. 1–8 chars.
    fn arb_word() -> impl Strategy<Value = String> {
        proptest::collection::vec(
            prop_oneof![
                Just('a'..='z')
                    .prop_flat_map(|r| { proptest::sample::select(r.collect::<Vec<_>>()) }),
                Just('A'..='Z')
                    .prop_flat_map(|r| { proptest::sample::select(r.collect::<Vec<_>>()) }),
                Just('0'..='9')
                    .prop_flat_map(|r| { proptest::sample::select(r.collect::<Vec<_>>()) }),
            ],
            1..8usize,
        )
        .prop_map(|chars| chars.into_iter().collect::<String>())
    }

    /// Generate a `--flag` or `-x` style argument.
    fn arb_flag() -> impl Strategy<Value = String> {
        prop_oneof![
            arb_word().prop_map(|w| format!("--{w}")),
            "[a-z]".prop_map(|c| format!("-{c}")),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 200,
            ..ProptestConfig::default()
        })]

        #[test]
        fn pure_idempotent(words in proptest::collection::vec(arb_word(), 1..6)) {
            let cmd = words.join(" ");
            let p1 = canonical_prefix(&cmd);
            let p2 = canonical_prefix(&cmd);
            prop_assert_eq!(&p1, &p2, "canonical_prefix not pure");
            // Feeding the prefix back yields itself.
            if !p1.is_empty() {
                let p3 = canonical_prefix(&p1);
                prop_assert_eq!(p1, p3, "canonical_prefix not idempotent");
            }
        }

        #[test]
        fn output_is_lowercase(words in proptest::collection::vec(arb_word(), 1..6)) {
            let cmd = words.join(" ");
            let p = canonical_prefix(&cmd);
            prop_assert_eq!(
                &p,
                &p.to_lowercase(),
                "canonical_prefix produced non-lowercase output"
            );
        }

        #[test]
        fn flags_dont_change_prefix(
            words in proptest::collection::vec(arb_word(), 1..6),
            flags in proptest::collection::vec(arb_flag(), 0..6),
        ) {
            let plain = words.join(" ");
            let with_flags = {
                let mut tokens: Vec<String> = Vec::new();
                tokens.extend(words.iter().cloned());
                // Sprinkle flags in the middle (after the first word so
                // we don't try to test what happens when the very first
                // token is itself a flag — that's a separate case).
                if tokens.is_empty() {
                    tokens.extend(flags);
                } else {
                    let head = tokens.remove(0);
                    tokens.splice(0..0, flags.iter().cloned());
                    tokens.insert(0, head);
                }
                tokens.join(" ")
            };
            prop_assert_eq!(
                canonical_prefix(&plain),
                canonical_prefix(&with_flags),
                "flags changed canonical_prefix"
            );
        }

        #[test]
        fn output_is_prefix_of_input(
            words in proptest::collection::vec(arb_word(), 1..6),
        ) {
            let cmd = words.join(" ");
            let p = canonical_prefix(&cmd);
            if p.is_empty() { return Ok(()); }
            let lower_cmd = cmd.to_lowercase();
            // The prefix tokens should match the leading positional tokens
            // of the input (case-insensitively).
            let prefix_token_count = p.split_whitespace().count();
            let head: String = lower_cmd
                .split_whitespace()
                .filter(|t| !t.starts_with('-'))
                .take(prefix_token_count)
                .collect::<Vec<_>>()
                .join(" ");
            prop_assert_eq!(
                &p,
                &head,
                "prefix is not a leading-positional slice of the input"
            );
        }
    }
}
