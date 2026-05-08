use super::*;

#[test]
fn drop_block_removes_scripts_and_styles() {
    let html = "pre<script>alert('x')</script>mid<style>a{}</style>post";
    let stripped = drop_block(html, "script");
    let stripped = drop_block(&stripped, "style");
    assert_eq!(stripped, "premidpost");
}

#[test]
fn drop_block_is_case_insensitive() {
    let html = "<SCRIPT>bad</SCRIPT>good";
    assert_eq!(drop_block(html, "script"), "good");
}

#[test]
fn html_to_text_extracts_body_and_links() {
    let html = r#"
        <html><head><title>X</title></head><body>
          <h1>Hello</h1>
          <p>World <a href="https://ex.com/1">one</a></p>
          <p>See <a href="https://ex.com/2">two</a></p>
          <script>window.ok=1;</script>
        </body></html>
    "#;
    let (text, links) = html_to_text(html);
    assert!(text.contains("Hello"));
    assert!(text.contains("World"));
    assert!(!text.contains("window.ok"), "script should be stripped");
    assert_eq!(
        links,
        vec![
            "https://ex.com/1".to_string(),
            "https://ex.com/2".to_string()
        ]
    );
}

#[test]
fn html_to_text_decodes_entities() {
    let (text, _) = html_to_text("<p>a &amp; b &lt; c &gt; d</p>");
    assert!(text.contains("a & b < c > d"));
}

#[test]
fn extract_hrefs_skips_fragment_and_js() {
    let html = concat!(
        r##"<a href="#top">t</a>"##,
        r#"<a href="javascript:void(0)">x</a>"#,
        r#"<a href="https://ex.com">ok</a>"#,
    );
    let hrefs = extract_hrefs(html);
    assert_eq!(hrefs, vec!["https://ex.com".to_string()]);
}

#[test]
fn truncate_chars_respects_limit() {
    let s = "a".repeat(1000);
    let out = crate::tool::fetch_common::truncate_chars(&s, 100);
    // Must be shorter than original and end with truncation marker
    assert!(out.chars().count() < 1000);
    assert!(out.contains("truncated"));
    // Must not exceed limit by much (marker overhead)
    assert!(out.chars().count() <= 100);
}

#[tokio::test]
async fn web_fetch_rejects_bad_url() {
    let tool = WebFetchTool::new();
    let cwd = std::env::current_dir().unwrap();
    let r = tool.execute(json!({"url":"not-a-url"}), &cwd).await;
    assert!(r.is_error);
    let r2 = tool.execute(json!({"url":"ftp://x"}), &cwd).await;
    assert!(r2.is_error);
}
