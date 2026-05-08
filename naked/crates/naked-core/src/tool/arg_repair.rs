//! Arg repair — fix common JSON malformations from LLM tool calls.
//!
//! LLMs stream tool arguments as deltas. Common breaks:
//! 1. Trailing comma before } or ]
//! 2. Unclosed braces/brackets
//! 3. Truncated string (missing closing quote)
//!
//! The repair ladder: strict parse → strip trailing commas →
//! balance braces → fallback {}.

/// Try to parse JSON, repairing common issues if strict parse fails.
pub fn repair_json(raw: &str) -> serde_json::Value {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return serde_json::json!({});
    }

    // 1. Strict parse:
    if let Ok(v) = serde_json::from_str(trimmed) {
        return v;
    }

    // 2. Strip trailing commas before } or ]:
    let cleaned = strip_trailing_commas(trimmed);
    if let Ok(v) = serde_json::from_str(&cleaned) {
        return v;
    }

    // 3. Balance braces/brackets:
    let balanced = balance_braces(&cleaned);
    if let Ok(v) = serde_json::from_str(&balanced) {
        return v;
    }

    // 4. Try closing unclosed strings:
    let string_fixed = close_unclosed_strings(&balanced);
    if let Ok(v) = serde_json::from_str(&string_fixed) {
        return v;
    }

    // 5. Fallback:
    serde_json::json!({})
}

fn strip_trailing_commas(s: &str) -> String {
    let mut result = s.to_string();
    // Remove commas before } or ]
    loop {
        let before = result.len();
        result = result.replace(",}", "}").replace(",]", "]");
        // Also handle whitespace: , \n}
        static RE: std::sync::OnceLock<regex_lite::Regex> = std::sync::OnceLock::new();
        let re = RE.get_or_init(|| regex_lite::Regex::new(r",\s*([}\]])").expect("static regex"));
        result = re.replace_all(&result, "$1").to_string();
        if result.len() == before {
            break;
        }
    }
    result
}

fn balance_braces(s: &str) -> String {
    let mut opens = 0i32;
    let mut brackets = 0i32;
    let mut in_string = false;
    let mut escaped = false;

    for ch in s.chars() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' && in_string {
            escaped = true;
            continue;
        }
        if ch == '"' {
            in_string = !in_string;
            continue;
        }
        if !in_string {
            match ch {
                '{' => opens += 1,
                '}' => opens -= 1,
                '[' => brackets += 1,
                ']' => brackets -= 1,
                _ => {}
            }
        }
    }

    let mut result = s.to_string();
    // Close unclosed strings first:
    if in_string {
        result.push('"');
    }
    for _ in 0..brackets.max(0) {
        result.push(']');
    }
    for _ in 0..opens.max(0) {
        result.push('}');
    }
    result
}

fn close_unclosed_strings(s: &str) -> String {
    // If the last non-whitespace char is inside an unclosed string,
    // add a closing quote before the brace-balancing:
    let mut in_string = false;
    let mut escaped = false;
    for ch in s.chars() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' && in_string {
            escaped = true;
            continue;
        }
        if ch == '"' {
            in_string = !in_string;
        }
    }
    if in_string {
        // Find last content, insert quote before closing braces:
        let trimmed = s.trim_end_matches(['}', ']', ' ', '\n']);
        format!("{}\"{}", trimmed, &s[trimmed.len()..])
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_json_passes_through() {
        let v = repair_json(r#"{"command": "ls -la"}"#);
        assert_eq!(v["command"], "ls -la");
    }

    #[test]
    fn trailing_comma_fixed() {
        let v = repair_json(r#"{"a": 1, "b": 2,}"#);
        assert_eq!(v["a"], 1);
        assert_eq!(v["b"], 2);
    }

    #[test]
    fn unclosed_brace_fixed() {
        let v = repair_json(r#"{"command": "cargo test""#);
        assert_eq!(v["command"], "cargo test");
    }

    #[test]
    fn unclosed_bracket_fixed() {
        let v = repair_json(r#"{"items": [1, 2, 3"#);
        assert_eq!(v["items"][0], 1);
    }

    #[test]
    fn truncated_string_fixed() {
        let v = repair_json(r#"{"file_path": "src/main.rs", "contents": "fn main() {"#);
        assert!(v.get("file_path").is_some());
    }

    #[test]
    fn empty_string_returns_empty_object() {
        let v = repair_json("");
        assert!(v.is_object());
        assert!(v.as_object().unwrap().is_empty());
    }

    #[test]
    fn total_garbage_returns_empty_object() {
        let v = repair_json("not json at all {{{{");
        assert!(v.is_object());
    }

    #[test]
    fn nested_objects_balanced() {
        let v = repair_json(r#"{"a": {"b": "c""#);
        assert_eq!(v["a"]["b"], "c");
    }

    #[test]
    fn trailing_comma_in_array() {
        let v = repair_json(r#"{"items": ["a", "b",]}"#);
        assert_eq!(v["items"][1], "b");
    }
}
