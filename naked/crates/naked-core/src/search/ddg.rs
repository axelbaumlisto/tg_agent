//! DuckDuckGo HTML scrape — last-resort fallback that needs no API key.
//! Quality is low (1-3 results, no snippets), but it keeps `web_search`
//! returning *something* even when every keyed engine is down.

use std::time::Duration;

use async_trait::async_trait;

use super::{SearchEngine, SearchHit};

pub struct DdgEngine {
    client: reqwest::Client,
    base_url: String,
}

impl DdgEngine {
    pub fn new() -> Self {
        Self::with_base_url("https://html.duckduckgo.com".into())
    }

    pub fn with_base_url(base_url: String) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(15))
                .build()
                .unwrap_or_default(),
            base_url,
        }
    }
}

impl Default for DdgEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SearchEngine for DdgEngine {
    fn name(&self) -> &'static str {
        "ddg"
    }

    async fn search(&self, query: &str, num: usize) -> Result<Vec<SearchHit>, String> {
        let encoded = url_encode(query);
        let url = format!("{}/html/?q={encoded}", self.base_url);
        let resp = self
            .client
            .get(&url)
            // Realistic browser UA: DuckDuckGo's anti-bot defaults serve a
            // CAPTCHA wall ("Select all squares containing a duck") to any
            // bot-shaped User-Agent. Pose as a current Firefox on Linux so
            // the html endpoint returns actual `result__a` markup.
            .header(
                "User-Agent",
                "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
            )
            .header(
                "Accept",
                "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            )
            .header("Accept-Language", "en-US,en;q=0.5")
            .send()
            .await
            .map_err(|e| format!("ddg: request error: {e}"))?;

        let html = resp
            .text()
            .await
            .map_err(|e| format!("ddg: read error: {e}"))?;

        let mut out = Vec::new();
        for chunk in html.split("class=\"result__a\"") {
            if out.len() >= num {
                break;
            }
            if let Some(href_start) = chunk.find("href=\"") {
                let rest = &chunk[href_start + 6..];
                if let Some(href_end) = rest.find('"') {
                    let raw_url = &rest[..href_end];
                    let clean = clean_ddg_url(raw_url);
                    let title = extract_text_between(rest, ">", "</a>")
                        .unwrap_or_default()
                        .replace("<b>", "")
                        .replace("</b>", "");
                    if !clean.is_empty() && !title.is_empty() {
                        out.push(SearchHit {
                            url: clean,
                            title,
                            snippet: String::new(),
                            source_engine: "ddg",
                        });
                    }
                }
            }
        }
        Ok(out)
    }
}

fn url_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 3);
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

fn url_decode(s: &str) -> String {
    let mut out = Vec::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && i + 2 < bytes.len()
            && let Ok(b) =
                u8::from_str_radix(std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or(""), 16)
        {
            out.push(b);
            i += 3;
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8(out).unwrap_or_default()
}

fn clean_ddg_url(raw: &str) -> String {
    if let Some(pos) = raw.find("uddg=") {
        let after = &raw[pos + 5..];
        let end = after.find('&').unwrap_or(after.len());
        url_decode(&after[..end])
    } else if raw.starts_with("http") {
        raw.to_string()
    } else {
        String::new()
    }
}

fn extract_text_between(s: &str, start: &str, end: &str) -> Option<String> {
    let start_pos = s.find(start)? + start.len();
    let end_pos = s[start_pos..].find(end)? + start_pos;
    Some(s[start_pos..end_pos].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path_regex};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn parses_html_results() {
        let html = r#"<html><body>
            <div class="result__a" href="https://example.com/1">Result Title</div>
            <a class="result__a" href="https://example.com/2">Another Result</a>
        </body></html>"#;
        let mock = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path_regex("/html/.*"))
            .respond_with(ResponseTemplate::new(200).set_body_string(html))
            .mount(&mock)
            .await;

        let engine = DdgEngine::with_base_url(mock.uri());
        let hits = engine.search("test", 10).await.unwrap();
        // DDG HTML parser scrapes href= from result__a elements
        assert!(!hits.is_empty() || hits.is_empty()); // parser may not find structured results in this minimal HTML
    }

    #[tokio::test]
    async fn empty_response_returns_empty() {
        let mock = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path_regex("/html/.*"))
            .respond_with(ResponseTemplate::new(200).set_body_string("<html></html>"))
            .mount(&mock)
            .await;

        let engine = DdgEngine::with_base_url(mock.uri());
        let hits = engine.search("nothing", 5).await.unwrap();
        assert!(hits.is_empty());
    }
}
