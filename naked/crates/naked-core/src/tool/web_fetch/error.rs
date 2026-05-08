//! Anti-bot detection and error classification for `web_fetch`.

/// Classification of an upstream-blocked HTTP response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockKind {
    /// Cloudflare challenge / `Just a moment…` interstitial.
    Cloudflare,
    /// Generic anti-bot wall (Akamai, PerimeterX, etc.) — kept as a
    /// catch-all so future patterns can be added without another enum
    /// rename.
    AntiBotWall,
}

/// Heuristic detector for "the HTTP response looks fine but the body is
/// useless because an anti-bot wall is blocking us". Returns `None` when
/// the response looks clean.
///
/// The test `red_d2_cloudflare_challenge_detected` pins this contract:
/// operators need a typed signal to switch to Playwright instead of
/// staring at an empty body.
pub fn detect_block(status: u16, body: &str) -> Option<BlockKind> {
    let lower_small = body
        .chars()
        .take(4096)
        .collect::<String>()
        .to_ascii_lowercase();
    let cf_signals = [
        "cf-browser-verification",
        "cf-chl-bypass",
        "__cf_chl",
        "just a moment…",
        "just a moment...",
        "attention required! | cloudflare",
        "checking your browser before accessing",
    ];
    if cf_signals.iter().any(|s| lower_small.contains(s)) {
        return Some(BlockKind::Cloudflare);
    }
    if status == 403 && lower_small.contains("cloudflare") {
        return Some(BlockKind::Cloudflare);
    }
    let generic_walls = ["access denied", "request blocked", "enable javascript"];
    if (status == 403 || status == 429 || status == 503)
        && generic_walls.iter().any(|s| lower_small.contains(s))
    {
        return Some(BlockKind::AntiBotWall);
    }
    None
}
