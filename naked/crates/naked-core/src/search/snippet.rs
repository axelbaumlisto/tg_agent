//! [`SnippetExtractor`] — pull `price`, `area`, `district`, `phone` out of a
//! Vietnamese real-estate snippet using regexes. The point is the
//! **snippet-fast-path**: when all three of (price, area, district) are
//! already in the search snippet, the agent can call `research_save`
//! directly without spending a `web_fetch` round-trip.
//!
//! Tuned for `batdongsan.com.vn`, `alonhadat.com.vn`, `mogi.vn`,
//! `homedy.com`, `chotot.com`, and Facebook listings — the dominant
//! formats observed in the Da Nang corpus.

use std::sync::OnceLock;

use regex::Regex;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractedFinding {
    pub price_vnd_per_month: u64,
    pub area_m2: u32,
    pub district: String,
    pub phone: Option<String>,
}

/// Curated list of Da Nang district / phường names that appear in real
/// listings. Order matters: longer / more specific names are tried first
/// so "Bắc Mỹ An" doesn't get clobbered by "Mỹ An".
const KNOWN_DISTRICTS: &[&str] = &[
    "Ngũ Hành Sơn",
    "Liên Chiểu",
    "Thanh Khê",
    "Hải Châu",
    "Sơn Trà",
    "Cẩm Lệ",
    "Hòa Vang",
    "Bắc Mỹ An",
    "An Thượng",
    "Mỹ An",
    "Khuê Mỹ",
    "Mỹ Đa Đông",
    "An Hải Bắc",
    "An Hải Đông",
    "Phước Mỹ",
    "Mân Thái",
    "Thọ Quang",
];

#[derive(Debug, Default, Clone)]
pub struct SnippetExtractor;

static RE_PRICE: OnceLock<Regex> = OnceLock::new();
static RE_AREA: OnceLock<Regex> = OnceLock::new();
static RE_PHONE: OnceLock<Regex> = OnceLock::new();

impl SnippetExtractor {
    pub fn new() -> Self {
        Self
    }

    /// Extract a complete finding (price + area + district). Returns `None`
    /// if any of the three is missing — the snippet-fast-path is opt-in:
    /// partial data should still go through `web_fetch` for confirmation.
    pub fn extract(&self, snippet: &str) -> Option<ExtractedFinding> {
        let price = self.extract_price(snippet)?;
        let area = self.extract_area(snippet)?;
        let district = self.extract_district(snippet)?;
        Some(ExtractedFinding {
            price_vnd_per_month: price,
            area_m2: area,
            district,
            phone: self.extract_phone(snippet),
        })
    }

    /// Parse a Vietnamese price expression to absolute VND.
    ///
    /// Recognised units (case-insensitive):
    /// - `triệu` / `tr` / `million`  → ×1_000_000
    /// - `tỷ` / `ty` / `billion`     → ×1_000_000_000
    /// - `usd` / `$`                 → ×24_500 (USD/VND ≈ 2026 spot)
    /// - `vnd` / `đ` / `vnđ`         → ×1
    pub fn extract_price(&self, s: &str) -> Option<u64> {
        let re = RE_PRICE.get_or_init(|| {
            Regex::new(r"(?i)(\d+(?:[.,]\d+)?)\s*(triệu|tr|tỷ|ty|million|billion|usd|\$|vnd|vnđ|đ)\b")
                .unwrap()
        });
        let cap = re.captures(s)?;
        let n: f64 = cap[1].replace(',', ".").parse().ok()?;
        let unit = cap[2].to_lowercase();
        let mult: u64 = match unit.as_str() {
            "triệu" | "tr" | "million" => 1_000_000,
            "tỷ" | "ty" | "billion" => 1_000_000_000,
            "usd" | "$" => 24_500,
            "vnd" | "vnđ" | "đ" => 1,
            _ => return None,
        };
        Some((n * mult as f64) as u64)
    }

    /// Parse area in m² (m2 / m vuông / mét vuông variants).
    pub fn extract_area(&self, s: &str) -> Option<u32> {
        let re = RE_AREA.get_or_init(|| {
            Regex::new(r"(?i)(\d{2,5})\s*m\s*(?:²|2|\bvuông\b|\bvuong\b)").unwrap()
        });
        re.captures(s).and_then(|c| c[1].parse().ok())
    }

    pub fn extract_district(&self, s: &str) -> Option<String> {
        let lower = s.to_lowercase();
        for d in KNOWN_DISTRICTS {
            if lower.contains(&d.to_lowercase()) {
                return Some((*d).to_string());
            }
        }
        None
    }

    /// Extract a Vietnamese mobile phone number. Accepts:
    /// - 09xxxxxxxx, 03xxxxxxxx, 07xxxxxxxx, 08xxxxxxxx, 05xxxxxxxx
    /// - +84 prefix, with optional space/dot separators.
    pub fn extract_phone(&self, s: &str) -> Option<String> {
        let re = RE_PHONE.get_or_init(|| {
            Regex::new(
                r"(?:\+?84[\s.-]?|0)((?:3|5|7|8|9)\d(?:[\s.-]?\d){7,8})",
            )
            .unwrap()
        });
        let cap = re.captures(s)?;
        let digits: String = cap[1].chars().filter(|c| c.is_ascii_digit()).collect();
        Some(format!("0{}", digits))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_price_in_trieu() {
        let s = "Cho thuê mặt bằng đường Ông Ích Khiêm. Giá 59 triệu/tháng. Diện tích 140m².";
        let p = SnippetExtractor::new().extract_price(s).unwrap();
        assert_eq!(p, 59_000_000);
    }

    #[test]
    fn extracts_price_in_ty() {
        let s = "Bán nhà mặt phố giá 18 tỷ.";
        let p = SnippetExtractor::new().extract_price(s).unwrap();
        assert_eq!(p, 18_000_000_000);
    }

    #[test]
    fn extracts_price_in_usd() {
        let s = "Rent 2500 USD/month, 120 m², An Thượng";
        let p = SnippetExtractor::new().extract_price(s).unwrap();
        assert_eq!(p, 2500 * 24_500);
    }

    #[test]
    fn extracts_price_with_short_tr_abbreviation() {
        let s = "Giá 30 tr/tháng";
        let p = SnippetExtractor::new().extract_price(s).unwrap();
        assert_eq!(p, 30_000_000);
    }

    #[test]
    fn extracts_decimal_price() {
        let s = "Giá 1.5 tỷ";
        let p = SnippetExtractor::new().extract_price(s).unwrap();
        assert_eq!(p, 1_500_000_000);
    }

    #[test]
    fn extracts_area_with_squared() {
        let s = "Diện tích: 140 m². Phù hợp nhà hàng.";
        assert_eq!(SnippetExtractor::new().extract_area(s), Some(140));
    }

    #[test]
    fn extracts_area_with_m2_digit() {
        let s = "100 m2 mặt bằng kinh doanh";
        assert_eq!(SnippetExtractor::new().extract_area(s), Some(100));
    }

    #[test]
    fn area_returns_none_when_missing() {
        let s = "Cho thuê nhà đẹp gần biển";
        assert_eq!(SnippetExtractor::new().extract_area(s), None);
    }

    #[test]
    fn district_picks_longest_match_first() {
        let s = "Mặt bằng tại Bắc Mỹ An, gần biển";
        assert_eq!(
            SnippetExtractor::new().extract_district(s).unwrap(),
            "Bắc Mỹ An"
        );
    }

    #[test]
    fn district_matches_short_variant() {
        let s = "Cho thuê tại quận Hải Châu, Đà Nẵng";
        assert_eq!(
            SnippetExtractor::new().extract_district(s).unwrap(),
            "Hải Châu"
        );
    }

    #[test]
    fn extracts_full_finding_from_real_snippet() {
        let s = "CHO THUÊ MẶT BẰNG ĐƯỜNG ÔNG ÍCH KHIÊM - 140 M² - GIÁ 59 TRIỆU. \
                 Diện tích: 140 m² (ngang: 9.2 m). Phù hợp nhà hàng, cafe. \
                 Quận Hải Châu, Đà Nẵng. LH: 0905 123 456";
        let f = SnippetExtractor::new().extract(s).unwrap();
        assert_eq!(f.price_vnd_per_month, 59_000_000);
        assert_eq!(f.area_m2, 140);
        assert_eq!(f.district, "Hải Châu");
        assert_eq!(f.phone.as_deref(), Some("0905123456"));
    }

    #[test]
    fn returns_none_when_missing_field() {
        let s = "Cho thuê mặt bằng đẹp, liên hệ ngay";
        assert!(SnippetExtractor::new().extract(s).is_none());
    }

    #[test]
    fn extracts_phone_with_country_code() {
        let s = "Liên hệ +84 905-123-456";
        assert_eq!(
            SnippetExtractor::new().extract_phone(s).as_deref(),
            Some("0905123456")
        );
    }
}
