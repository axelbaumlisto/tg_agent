//! Small pure helpers shared between the bot and its tests.

/// Parse a human-friendly interval like `30s`, `15m`, `1h`, `1d`, or a plain
/// integer (interpreted as seconds). Returns `None` on bad input.
pub fn parse_interval(input: &str) -> Option<u64> {
    let s = input.trim();
    if s.is_empty() {
        return None;
    }
    if let Ok(n) = s.parse::<u64>() {
        return Some(n);
    }
    let (num, unit) = s.split_at(s.len() - 1);
    let n: u64 = num.parse().ok()?;
    match unit {
        "s" => Some(n),
        "m" => Some(n * 60),
        "h" => Some(n * 3_600),
        "d" => Some(n * 86_400),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_seconds_minutes_hours_days() {
        assert_eq!(parse_interval("30"), Some(30));
        assert_eq!(parse_interval("30s"), Some(30));
        assert_eq!(parse_interval("15m"), Some(900));
        assert_eq!(parse_interval("1h"), Some(3_600));
        assert_eq!(parse_interval("2d"), Some(2 * 86_400));
    }

    #[test]
    fn rejects_bad_input() {
        assert_eq!(parse_interval(""), None);
        assert_eq!(parse_interval("abc"), None);
        assert_eq!(parse_interval("10x"), None);
    }
}
