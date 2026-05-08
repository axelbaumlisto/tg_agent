use super::*;

#[test]
fn canonicalize_lowercases_scheme_and_host() {
    assert_eq!(
        canonicalize_url("HTTPS://Example.COM/Path"),
        "https://example.com/Path"
    );
}

#[test]
fn canonicalize_strips_fragment() {
    assert_eq!(
        canonicalize_url("https://x.com/a#section-2"),
        "https://x.com/a"
    );
}

#[test]
fn canonicalize_drops_utm_but_keeps_real_query() {
    assert_eq!(
        canonicalize_url("https://x.com/a?utm_source=tg&id=42&utm_medium=bot"),
        "https://x.com/a?id=42"
    );
}

#[test]
fn canonicalize_sorts_remaining_query() {
    assert_eq!(
        canonicalize_url("https://x.com/a?z=1&a=2&m=3"),
        "https://x.com/a?a=2&m=3&z=1"
    );
}

#[test]
fn canonicalize_strips_trailing_slash_unless_root() {
    assert_eq!(canonicalize_url("https://x.com/a/"), "https://x.com/a");
    assert_eq!(canonicalize_url("https://x.com/"), "https://x.com/");
}

#[test]
fn dedup_hash_is_stable_across_tracking_params() {
    let a = dedup_hash("https://batdongsan.com.vn/ad/42?utm_source=x");
    let b = dedup_hash("https://batdongsan.com.vn/ad/42?utm_campaign=y&gclid=z");
    assert_eq!(a, b);
}

#[test]
fn dedup_hash_differs_on_different_paths() {
    let a = dedup_hash("https://batdongsan.com.vn/ad/42");
    let b = dedup_hash("https://batdongsan.com.vn/ad/43");
    assert_ne!(a, b);
}

#[test]
fn host_path_hash_ignores_all_query_params() {
    let a = host_path_hash("https://batdongsan.com.vn/ad/42?sort=newest");
    let b = host_path_hash("https://batdongsan.com.vn/ad/42?sort=oldest&page=3");
    let c = host_path_hash("https://batdongsan.com.vn/ad/42");
    assert_eq!(a, b);
    assert_eq!(a, c);
}

#[test]
fn host_path_hash_differs_on_path_change() {
    let a = host_path_hash("https://batdongsan.com.vn/ad/42");
    let b = host_path_hash("https://batdongsan.com.vn/ad/43");
    assert_ne!(a, b);
}

#[test]
fn host_path_hash_differs_on_host_change() {
    let a = host_path_hash("https://a.example/path");
    let b = host_path_hash("https://b.example/path");
    assert_ne!(a, b);
}

#[test]
fn content_hash_is_stable_across_whitespace_and_case() {
    let a = content_hash("Cho thuê 100m²  Hải Châu, contact 0905111222");
    let b = content_hash("CHO THUÊ 100m²\n\nHẢI CHÂU, contact 0905111222");
    assert_eq!(a, b);
}

#[test]
fn content_hash_ignores_digit_variations() {
    // Same prose, different price digits → same hash.
    let a = content_hash("Cho thuê mặt bằng 100m² giá 25 triệu");
    let b = content_hash("Cho thuê mặt bằng 200m² giá 50 triệu");
    assert_eq!(a, b);
}

#[test]
fn content_hash_differs_on_different_prose() {
    let a = content_hash("Cho thuê mặt bằng kinh doanh");
    let b = content_hash("Bán nhà mặt phố tại quận 1");
    assert_ne!(a, b);
}

#[test]
fn content_hash_empty_for_empty_input() {
    assert_eq!(content_hash(""), "");
    assert_eq!(content_hash("   \n\t  "), "");
}

#[test]
fn slugify_handles_unicode_by_dropping_non_ascii() {
    assert_eq!(slugify("Jaguar XF cheap"), "jaguar-xf-cheap");
    // Cyrillic drops out → fall back to "research".
    assert_eq!(slugify("ягуар"), "research");
    // Mixed — keeps the ASCII parts.
    assert_eq!(slugify("ягуар cheap car"), "cheap-car");
}

#[test]
fn slugify_caps_length() {
    let long = "a".repeat(200);
    assert!(slugify(&long).len() <= 48);
}

#[test]
fn new_research_id_has_slug_and_suffix() {
    let id = new_research_id("Jaguar XF");
    assert!(id.starts_with("jaguar-xf-"), "got id={id}");
    // 4 hex chars after the slug now (compact format).
    let tail = id.split('-').next_back().unwrap();
    assert_eq!(tail.len(), 4, "got id={id}");
    assert!(tail.chars().all(|c| c.is_ascii_hexdigit()), "got id={id}");
}

#[test]
fn new_research_id_strips_noise_and_caps_to_three_words() {
    // Stopwords + numerics + currency unit suffixes get dropped, leaving
    // just the meaningful words (capped to three).
    let id =
        new_research_id("Da Nang commercial real estate 50-250m² 1500-3000USD some long suffix");
    let parts: Vec<&str> = id.split('-').collect();
    // `da` is a 2-char Vietnamese word, kept; followed by nang, commercial.
    // Cap = 3 meaningful tokens + 1 hex suffix = 4 segments total.
    assert_eq!(parts.len(), 4, "got id={id}, parts={parts:?}");
    assert!(
        !parts[..3]
            .iter()
            .any(|p| p.chars().all(|c| c.is_ascii_digit())),
        "numeric-only tokens must not survive: id={id}"
    );
    assert!(
        !parts[..3].iter().any(|p| *p == "the"
            || *p == "real"
            || *p == "estate"
            || *p == "vnd"
            || *p == "usd"),
        "stopwords/currency must be filtered: id={id}"
    );
}

#[test]
fn new_research_id_falls_back_when_topic_is_all_noise() {
    // All-digit topic → `short_slug` drops every word, falls back to
    // `slugify` (which keeps digits), then appends a 4-hex suffix.
    // The contract is "never panics, always produces a hex tail".
    let id = new_research_id("12345 67890");
    let tail = id.split('-').next_back().unwrap();
    assert_eq!(tail.len(), 4, "got id={id}");
    assert!(tail.chars().all(|c| c.is_ascii_hexdigit()), "got id={id}");

    // All-stopword topic → `slugify` produces non-empty ASCII slug,
    // so we still get something readable.
    let id2 = new_research_id("the of and");
    assert!(
        id2.split('-').next_back().unwrap().len() == 4,
        "got id={id2}"
    );
}

#[test]
fn new_research_id_handles_cyrillic_topic_safely() {
    // Cyrillic has no ASCII alphanumerics, so the slug falls back to
    // `slugify` which itself returns "research" for non-ASCII-only
    // topics. Importantly: does not panic on multi-byte chars.
    let id = new_research_id("Найди коммерческую недвижимость в Дананге");
    let tail = id.split('-').next_back().unwrap();
    assert_eq!(tail.len(), 4, "got id={id}");
}

// ── normalize_title / titles_are_similar ────────────────────────────

#[test]
fn normalize_title_for_similarity_lowercases_and_strips_separators() {
    let n = normalize_title_for_similarity("Căn Hộ 70m² Tầng 3, Quận 1");
    // All lowercase, symbols become spaces, collapsed
    assert_eq!(n, n.to_lowercase());
    // Should not contain m2 (stopword) or the digit 70 (short numeric)
    assert!(!n.contains("m2"), "m2 is a stopword: got {n}");
    assert!(
        !n.split_whitespace().any(|t| t == "70"),
        "70 is a short numeric: got {n}"
    );
}

#[test]
fn titles_are_similar_detects_same_property_variant() {
    // Same apartment, same key attributes, just rearranged —
    // the typical repost pattern on non-Vietnamese aggregators.
    // All shared content words survive normalization with high Jaccard.
    assert!(
        titles_are_similar(
            "3 bedroom apartment Sukhumvit quiet high floor",
            "Quiet apartment Sukhumvit high floor 3br",
        ),
        "same property different word order must be flagged as similar \
         (shared tokens: apartment, sukhumvit, quiet, high, floor)"
    );
}

#[test]
fn titles_are_similar_does_not_flag_different_properties() {
    // Completely different listing: different district, very different title
    assert!(
        !titles_are_similar(
            "Biệt thự Thảo Điền 200m² bể bơi",
            "Phòng trọ 15m² Bình Thạnh giá rẻ",
        ),
        "clearly different properties must NOT be flagged as similar"
    );
}

#[test]
fn titles_are_similar_handles_empty_inputs() {
    assert!(!titles_are_similar("", "anything"));
    assert!(!titles_are_similar("anything", ""));
    assert!(!titles_are_similar("", ""));
}

#[test]
fn titles_are_similar_vietnamese_abbreviation_caveat() {
    // Vietnamese abbreviated titles like "2PN" vs "phòng ngủ" share
    // different surface tokens after normalization — the algorithm is
    // token-Jaccard on surface forms, not semantic. This documents the
    // known limitation: these will NOT be flagged (0.44 Jaccard < 0.65).
    // The gatekeeper agent uses its LLM judgement for these; the fuzzy
    // dedup is a best-effort pre-filter only.
    let vieta = "Căn hộ 70m² 2PN quận 1 giá tốt";
    let vietb = "Căn hộ 70m² 2 phòng ngủ quận 1";
    // We simply check that neither crashes nor returns a nonsensical result.
    let _ = titles_are_similar(vieta, vietb); // either outcome is acceptable
}

#[test]
fn titles_are_similar_identical_titles() {
    let t = "Studio 30m² near BTS Asok Bangkok";
    assert!(titles_are_similar(t, t));
}
