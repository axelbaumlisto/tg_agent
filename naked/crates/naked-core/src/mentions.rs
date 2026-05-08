//! @-mention parsing — extract file paths from user messages.
//!
//! When user writes `@src/main.rs` or `@Cargo.toml`, the file content
//! is injected into the message so the model sees it.

use std::path::{Path, PathBuf};

/// Extracted @-mention.
#[derive(Debug, Clone)]
pub struct Mention {
    pub path: PathBuf,
    pub start: usize,
    pub end: usize,
}

/// Parse @-mentions from a message. Returns mentions found.
pub fn parse_mentions(text: &str) -> Vec<Mention> {
    let mut mentions = Vec::new();
    let mut i = 0;
    let bytes = text.as_bytes();

    while i < bytes.len() {
        if bytes[i] == b'@' {
            // Must be start of word (i==0 or preceded by whitespace):
            if i > 0 && !bytes[i - 1].is_ascii_whitespace() {
                i += 1;
                continue;
            }
            let start = i;
            i += 1; // skip @
            // Collect path chars:
            let path_start = i;
            while i < bytes.len() && !bytes[i].is_ascii_whitespace() && bytes[i] != b'@' {
                i += 1;
            }
            let path_str = &text[path_start..i];
            // Must look like a file path (has / or .):
            if (path_str.contains('/') || path_str.contains('.'))
                && path_str.len() >= 3
                && !path_str.starts_with("http")
            {
                mentions.push(Mention {
                    path: PathBuf::from(path_str),
                    start,
                    end: i,
                });
            }
        } else {
            i += 1;
        }
    }
    mentions
}

/// Expand @-mentions: read files and build context block.
/// Returns (cleaned_message, context_block).
pub async fn expand_mentions(text: &str, cwd: &Path) -> (String, String) {
    let mentions = parse_mentions(text);
    if mentions.is_empty() {
        return (text.to_string(), String::new());
    }

    let mut context = String::from("\n[Context from @-mentions]\n");
    let mut cleaned = text.to_string();

    // Process in reverse to preserve indices:
    for mention in mentions.iter().rev() {
        let full_path = if mention.path.is_absolute() {
            mention.path.clone()
        } else {
            cwd.join(&mention.path)
        };

        match tokio::fs::read_to_string(&full_path).await {
            Ok(content) => {
                let display = mention.path.display();
                let truncated = if content.len() > 4000 {
                    let end = content.floor_char_boundary(4000);
                    format!(
                        "{}…\n[truncated, {} bytes total]",
                        &content[..end],
                        content.len()
                    )
                } else {
                    content
                };
                context.push_str(&format!("\n--- {display} ---\n{truncated}\n"));
            }
            Err(_) => {
                context.push_str(&format!(
                    "\n--- {} --- (file not found)\n",
                    mention.path.display()
                ));
            }
        }

        // Remove @path from message:
        cleaned.replace_range(
            mention.start..mention.end,
            &mention.path.display().to_string(),
        );
    }

    (cleaned, context)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_single_mention() {
        let m = parse_mentions("look at @src/main.rs please");
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].path, PathBuf::from("src/main.rs"));
    }

    #[test]
    fn parse_multiple_mentions() {
        let m = parse_mentions("@src/lib.rs and @Cargo.toml");
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn skip_email_addresses() {
        let m = parse_mentions("contact user@example.com");
        // "example.com" has a dot but "user@example.com" — @ not at word start
        assert!(m.is_empty());
    }

    #[test]
    fn skip_short_paths() {
        let m = parse_mentions("@ab is too short");
        assert!(m.is_empty());
    }

    #[test]
    fn skip_urls() {
        let m = parse_mentions("see @https://example.com");
        assert!(m.is_empty());
    }

    #[test]
    fn at_start_of_message() {
        let m = parse_mentions("@Cargo.toml check deps");
        assert_eq!(m.len(), 1);
    }

    #[tokio::test]
    async fn expand_reads_real_file() {
        let tmp = tempfile::tempdir().unwrap();
        tokio::fs::write(tmp.path().join("test.txt"), "hello world")
            .await
            .unwrap();
        let (cleaned, context) = expand_mentions("read @test.txt", tmp.path()).await;
        assert!(context.contains("hello world"));
        assert!(cleaned.contains("test.txt"));
    }

    #[tokio::test]
    async fn expand_missing_file() {
        let tmp = tempfile::tempdir().unwrap();
        let (_, context) = expand_mentions("read @missing.rs", tmp.path()).await;
        assert!(context.contains("not found"));
    }
}
