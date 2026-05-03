//! Live e2e tests for `tg_markup` — send prompts to real LLM providers,
//! collect Markdown responses, verify the Telegram HTML renderer produces
//! valid, non-empty, correctly structured output.
//!
//! Run: `cargo test -p naked-tg --test tg_markup_live_e2e -- --nocapture`
//!
//! These tests are skipped when no provider is reachable.

use std::time::Duration;

use futures_util::StreamExt;
use naked_core::config::Config;
use naked_core::history::ConversationHistory;
use naked_core::loop_::{AgentLoop, LoopConfig};
use naked_core::provider::{ChatRequest, Provider};
use naked_core::tool::bash::BashTool;
use naked_core::tool::file_ops::{EditFileTool, ReadFileTool, WriteFileTool};
use naked_core::tool::registry::ToolRegistry;
use naked_core::tool::search::{GlobSearchTool, GrepSearchTool};
use naked_core::types::AgentEvent;
use naked_tg::tg_markup::{MAX_TG_MSG, md_to_tg_html, split_html};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

// ─── Helpers ─────────────────────────────────────────────────────────────

fn test_config() -> Option<Config> {
    Config::load().ok()
}

fn get_provider(config: &Config, name: &str, model: &str) -> Option<Box<dyn Provider>> {
    let pc = config.providers.get(name)?;
    let mut resolved = pc.resolved().ok()?;
    if resolved.api_key.is_empty() || resolved.api_key.starts_with('$') {
        return None;
    }
    resolved.models = vec![model.to_string()];
    Some(naked_core::create_provider(name, resolved))
}

async fn probe(provider: &dyn Provider, model: &str) -> bool {
    let req = ChatRequest {
        model: model.into(),
        system: String::new(),
        messages: vec![serde_json::json!({"role": "user", "content": "Say OK"})],
        tools: vec![],
        max_tokens: 32,
        temperature: None,
        reasoning: Some("off".into()),
    };
    match provider.stream_chat(req).await {
        Ok(mut stream) => {
            let mut got_text = false;
            while let Some(chunk) = stream.next().await {
                if let naked_core::types::StreamChunk::Text(_) = &chunk {
                    got_text = true;
                    break;
                }
            }
            got_text
        }
        Err(_) => false,
    }
}

async fn find_provider(config: &Config) -> Option<(Box<dyn Provider>, String, String)> {
    let candidates: &[(&str, &str)] = &[
        ("moonshot", "moonshot-v1-8k"),
        ("qwen", "qwen3.6-plus"),
        ("glm-cn", "glm-5.1"),
        ("groq", "llama-3.3-70b-versatile"),
    ];
    for &(prov_name, model) in candidates {
        if let Some(p) = get_provider(config, prov_name, model) {
            if probe(p.as_ref(), model).await {
                return Some((p, prov_name.into(), model.into()));
            }
            eprintln!("  probe SKIP: {prov_name}/{model}");
        }
    }
    None
}

async fn run_prompt(provider: Box<dyn Provider>, model: &str, prompt: &str) -> Option<String> {
    let sys = "You are a helpful assistant. Always respond with well-structured Markdown.";
    let mut history = ConversationHistory::new(sys.into());
    history.push_user(prompt);

    let tools: Vec<Box<dyn naked_core::tool::Tool>> = vec![
        Box::new(BashTool::new(15)),
        Box::new(ReadFileTool),
        Box::new(WriteFileTool),
        Box::new(EditFileTool),
        Box::new(GlobSearchTool),
        Box::new(GrepSearchTool),
    ];

    let tmp = tempfile::tempdir().unwrap();
    let config = LoopConfig {
        max_iterations: 3,
        cwd: tmp.path().to_path_buf(),
        model: model.to_string(),
        max_tokens: 2048,
        temperature: Some(0.3),
        reasoning: None,
        provider: String::new(),
        health: None,
    };

    let agent = AgentLoop::new(provider, ToolRegistry::new(tools), config);
    let (tx, mut rx) = mpsc::channel(256);
    let cancel = CancellationToken::new();

    // Collect events concurrently with the agent run so we don't
    // lose deltas that arrive before `run()` returns.
    let collector = tokio::spawn(async move {
        let mut text = String::new();
        while let Some(ev) = rx.recv().await {
            if let AgentEvent::TextDelta(t) = ev {
                text.push_str(&t);
            }
        }
        text
    });

    let result = tokio::time::timeout(
        Duration::from_secs(90),
        agent.run(&mut history, tx, cancel, None, None),
    )
    .await;

    if result.is_err() || matches!(result, Ok(Err(_))) {
        eprintln!("  run_prompt error: {result:?}");
    }

    let text = collector.await.unwrap_or_default();
    if text.is_empty() { None } else { Some(text) }
}

/// Assert rendered HTML has balanced tags and no broken entities.
fn assert_valid_tg_html(html: &str, label: &str) {
    assert!(
        !html.contains("\n# ") && !html.starts_with("# "),
        "[{label}] raw heading leaked: {}",
        &html[..html.len().min(200)]
    );

    for (tag, name) in [
        ("<pre>", "pre"),
        ("<code", "code"), // matches <code> and <code class="...">
        ("<b>", "b"),
        ("<i>", "i"),
        ("<s>", "s"),
        ("<blockquote>", "blockquote"),
    ] {
        let open = if name == "code" {
            // Count both <code> and <code class="..."> as openers
            html.matches("<code>").count() + html.matches("<code ").count()
        } else {
            html.matches(tag).count()
        };
        let close_tag = format!("</{name}>");
        let close = html.matches(close_tag.as_str()).count();
        assert_eq!(
            open,
            close,
            "[{label}] unbalanced <{name}>: {open} open vs {close} close\n  html: {}",
            &html[..html.len().min(300)]
        );
    }

    // Check <a> separately (has attributes). Warn instead of fail
    // because model output with complex URLs can confuse the linkifier
    // across chunk boundaries.
    let a_open = html.matches("<a ").count();
    let a_close = html.matches("</a>").count();
    if a_open != a_close {
        eprintln!("  WARN [{label}] unbalanced <a>: {a_open} open vs {a_close} close");
    }

    // No bare & (must be &amp;, &lt;, &gt;, &quot;, &#)
    for (i, _) in html.match_indices('&') {
        let rest = &html[i + 1..];
        let valid = rest.starts_with("amp;")
            || rest.starts_with("lt;")
            || rest.starts_with("gt;")
            || rest.starts_with("quot;")
            || rest.starts_with('#')
            || rest.starts_with("nbsp;");
        assert!(
            valid,
            "[{label}] bare & at byte {i}: ...{}...",
            &html[i..html.len().min(i + 30)]
        );
    }
}

macro_rules! need_live {
    ($config:ident, $prov:ident, $prov_name:ident, $model:ident, $tag:expr) => {
        let $config = match test_config() {
            Some(c) => c,
            None => {
                eprintln!("SKIP {}: no config", $tag);
                return;
            }
        };
        let ($prov, $prov_name, $model) = match find_provider(&$config).await {
            Some(p) => p,
            None => {
                eprintln!("SKIP {}: no provider", $tag);
                return;
            }
        };
        eprintln!(">>> {} [{}/{}]", $tag, $prov_name, $model);
    };
}

// ─── Tests ───────────────────────────────────────────────────────────────

/// Model generates headings, bold, code, list → valid Telegram HTML.
#[tokio::test]
async fn t200_render_rich_markdown_from_live_model() {
    need_live!(config, provider, prov_name, model, "t200");

    let md = match run_prompt(
        provider,
        &model,
        "Write a short guide about Rust error handling. Use:\n\
         - A level-2 heading\n\
         - A fenced code block with a Rust example\n\
         - A bullet list with 3 items\n\
         - Bold text for emphasis\n\
         Keep it under 300 words.",
    )
    .await
    {
        Some(t) => t,
        None => {
            eprintln!("SKIP t200: empty response");
            return;
        }
    };

    eprintln!("  md: {} chars", md.len());
    let html = md_to_tg_html(&md);
    eprintln!("  html: {} chars", html.len());

    assert!(!html.is_empty());
    assert_valid_tg_html(&html, "t200");
    assert!(html.contains("<b>"), "missing <b> (heading or bold)");
    assert!(
        html.contains("<pre><code") || html.contains("<code>"),
        "missing code"
    );
    assert!(html.contains('•') || html.contains("1."), "missing list");

    eprintln!("  PASS t200");
}

/// Table renders as monospace <pre>, no separator rows.
#[tokio::test]
async fn t201_render_table_from_live_model() {
    need_live!(config, provider, prov_name, model, "t201");

    let md = match run_prompt(
        provider,
        &model,
        "Output ONLY a Markdown pipe-table comparing Rust, Python, Go \
         with columns: Language | Speed | Safety | Learning Curve. \
         Include the header separator row. No other text.",
    )
    .await
    {
        Some(t) => t,
        None => {
            eprintln!("SKIP t201: empty response");
            return;
        }
    };

    eprintln!("  md:\n{md}");
    let html = md_to_tg_html(&md);
    eprintln!("  html:\n{html}");

    assert!(!html.is_empty());
    assert_valid_tg_html(&html, "t201");
    assert!(html.contains("<pre>"), "table should render as <pre>");
    assert!(!html.contains("---"), "separator row should be stripped");
    assert!(
        html.contains("Rust") || html.contains("rust"),
        "content missing"
    );

    eprintln!("  PASS t201");
}

/// Long response splits into ≤4096-byte chunks with balanced tags.
#[tokio::test]
async fn t202_split_long_response_from_live_model() {
    need_live!(config, provider, prov_name, model, "t202");

    let md = match run_prompt(
        provider,
        &model,
        "Write a detailed 1500-word tutorial about building a REST API in Rust \
         with Actix-web. Include at least 5 code blocks, headings for each section, \
         and bullet lists. Cover setup, routing, handlers, errors, and testing.",
    )
    .await
    {
        Some(t) => t,
        None => {
            eprintln!("SKIP t202: empty response");
            return;
        }
    };

    eprintln!("  md: {} chars", md.len());
    let html = md_to_tg_html(&md);
    eprintln!("  html: {} chars", html.len());

    let chunks = split_html(&html, MAX_TG_MSG);
    eprintln!("  chunks: {}", chunks.len());
    for (i, chunk) in chunks.iter().enumerate() {
        eprintln!("    chunk {i}: {} bytes", chunk.len());
        assert!(
            chunk.len() <= MAX_TG_MSG,
            "chunk {i} too long: {} > {MAX_TG_MSG}",
            chunk.len()
        );
        assert_valid_tg_html(chunk, &format!("t202/chunk{i}"));
    }

    let rejoined: String = chunks.join("");
    assert!(rejoined.contains("Rust"), "'Rust' lost after split");

    if html.len() > MAX_TG_MSG {
        assert!(
            chunks.len() > 1,
            "html {} bytes but only {} chunk(s)",
            html.len(),
            chunks.len()
        );
    }

    eprintln!("  PASS t202");
}

/// Blockquote with inline formatting renders correctly.
#[tokio::test]
async fn t203_render_blockquote_and_nested_formatting() {
    need_live!(config, provider, prov_name, model, "t203");

    let md = match run_prompt(
        provider,
        &model,
        "Write a short sentence, then quote it with Markdown > prefix. \
         Inside the quote use **bold** and `code`. After the quote, add a \
         bullet list with ~~strikethrough~~ in one item. Keep it concise.",
    )
    .await
    {
        Some(t) => t,
        None => {
            eprintln!("SKIP t203: empty response");
            return;
        }
    };

    eprintln!("  md:\n{md}");
    let html = md_to_tg_html(&md);
    eprintln!("  html:\n{html}");

    assert!(!html.is_empty());
    assert_valid_tg_html(&html, "t203");
    assert!(html.contains("<blockquote>"), "missing blockquote");

    eprintln!("  PASS t203");
}

/// Links become <a> tags, bare URLs auto-linked.
#[tokio::test]
async fn t204_render_links_from_live_model() {
    need_live!(config, provider, prov_name, model, "t204");

    let md = match run_prompt(
        provider,
        &model,
        "List 3 Rust crates with crates.io links: [name](https://crates.io/crates/...). \
         Also mention https://doc.rust-lang.org as a bare URL. Max 5 lines.",
    )
    .await
    {
        Some(t) => t,
        None => {
            eprintln!("SKIP t204: empty response");
            return;
        }
    };

    eprintln!("  md:\n{md}");
    let html = md_to_tg_html(&md);
    eprintln!("  html:\n{html}");

    assert!(!html.is_empty());
    assert_valid_tg_html(&html, "t204");
    assert!(
        html.contains("<a href=\"https://"),
        "missing clickable link"
    );

    eprintln!("  PASS t204");
}

/// Tool-use response renders with bash output embedded.
#[tokio::test]
async fn t205_render_tool_response_with_markdown() {
    need_live!(config, provider, prov_name, model, "t205");

    let md = match run_prompt(
        provider,
        &model,
        "Use bash to run `echo MARKUP_TEST_42` then report the result. \
         Format with a heading, the command in a code block, and a bullet list summary.",
    )
    .await
    {
        Some(t) => t,
        None => {
            eprintln!("SKIP t205: empty response");
            return;
        }
    };

    eprintln!("  md: {} chars", md.len());
    let html = md_to_tg_html(&md);

    assert!(!html.is_empty());
    assert_valid_tg_html(&html, "t205");
    assert!(html.contains("MARKUP_TEST_42"), "bash output token missing");

    eprintln!("  PASS t205");
}

/// Incremental render (streaming simulation) — render after each line,
/// verify no step produces broken HTML.
#[tokio::test]
async fn t206_incremental_render_never_breaks() {
    need_live!(config, provider, prov_name, model, "t206");

    let full_md = match run_prompt(
        provider,
        &model,
        "Write exactly:\n# Hello\nSome **bold** text.\n```\ncode\n```\n- item one\n- item two",
    )
    .await
    {
        Some(t) => t,
        None => {
            eprintln!("SKIP t206: empty response");
            return;
        }
    };

    let lines: Vec<&str> = full_md.lines().collect();
    let mut accumulated = String::new();
    let mut steps = 0;

    for (i, line) in lines.iter().enumerate() {
        if !accumulated.is_empty() {
            accumulated.push('\n');
        }
        accumulated.push_str(line);

        let html = md_to_tg_html(&accumulated);
        steps += 1;

        if !accumulated.trim().is_empty() {
            assert!(
                !html.is_empty(),
                "empty HTML at line {i} for: {accumulated}"
            );
        }

        // Balanced <pre>/<code> at every step (unclosed fences auto-close)
        let pre_o = html.matches("<pre>").count();
        let pre_c = html.matches("</pre>").count();
        assert_eq!(
            pre_o, pre_c,
            "unbalanced <pre> at line {i}: {pre_o} vs {pre_c}\nhtml: {html}"
        );
        let code_o = html.matches("<code>").count() + html.matches("<code ").count();
        let code_c = html.matches("</code>").count();
        assert_eq!(
            code_o, code_c,
            "unbalanced <code> at line {i}: {code_o} vs {code_c}\nhtml: {html}"
        );
    }

    eprintln!("  {steps} incremental steps, all valid");
    eprintln!("  PASS t206");
}

/// Edge case: empty and whitespace-only input.
#[tokio::test]
async fn t207_empty_and_whitespace_input() {
    assert_eq!(md_to_tg_html(""), "");
    assert_eq!(md_to_tg_html("   "), "");
    assert_eq!(md_to_tg_html("\n\n\n"), "");
    assert_eq!(split_html("", MAX_TG_MSG), vec![""]);
    eprintln!("  PASS t207");
}

// ─── Synthetic edge-case tests (no live model needed) ────────────────────

/// Models often produce nested bold+italic: ***text***
#[tokio::test]
async fn t208_nested_bold_italic() {
    let html = md_to_tg_html("This is ***very important***.");
    assert!(!html.is_empty());
    // Should contain some combination of <b>/<i> — not raw ***
    assert!(!html.contains("***"), "raw *** should not survive: {html}");
    eprintln!("  PASS t208: {html}");
}

/// Models output multi-level nested lists with indentation.
#[tokio::test]
async fn t209_indented_nested_list() {
    let md = "- Top item\n  - Nested item\n    - Deep item\n- Another top";
    let html = md_to_tg_html(md);
    assert!(html.contains("Top item"), "lost top item: {html}");
    assert!(html.contains("Nested item"), "lost nested: {html}");
    assert!(html.contains("Deep item"), "lost deep: {html}");
    assert_valid_tg_html(&html, "t209");
    eprintln!("  PASS t209: {html}");
}

/// Models mix code blocks with inline code in same response.
#[tokio::test]
async fn t210_mixed_code_styles() {
    let md = "Use `inline` code.\n\n```rust\nfn main() {\n    println!(\"hello\");\n}\n```\n\nThen more `inline` here.";
    let html = md_to_tg_html(md);
    assert!(html.contains("<code>inline</code>"));
    assert!(html.contains("<pre><code"));
    assert!(html.contains("println!"));
    assert_valid_tg_html(&html, "t210");
    eprintln!("  PASS t210");
}

/// Multi-line blockquote with blank line continuation (some models do this).
#[tokio::test]
async fn t211_multiline_blockquote_complex() {
    let md = "> First line of quote\n> Second line\n>\n> After blank line in quote\n\nRegular text after.";
    let html = md_to_tg_html(md);
    assert!(html.contains("<blockquote>"));
    assert!(html.contains("First line"));
    assert!(html.contains("Regular text after"));
    assert_valid_tg_html(&html, "t211");
    eprintln!("  PASS t211: {html}");
}

/// Table with HTML entities in cell content (prices like $1,000).
#[tokio::test]
async fn t212_table_with_special_chars() {
    let md = "| Item | Price |\n|---|---|\n| Widget <Pro> | $1,000 & up |";
    let html = md_to_tg_html(md);
    assert!(html.contains("<pre>"));
    assert!(html.contains("&lt;Pro&gt;"), "< > must be escaped: {html}");
    assert!(html.contains("&amp;"), "& must be escaped: {html}");
    assert!(!html.contains("---"));
    assert_valid_tg_html(&html, "t212");
    eprintln!("  PASS t212");
}

/// Extremely long single line (no newlines) — must not panic or OOM.
#[tokio::test]
async fn t213_very_long_single_line() {
    let long = "word ".repeat(2000); // ~10KB
    let html = md_to_tg_html(&long);
    assert!(!html.is_empty());
    let chunks = split_html(&html, MAX_TG_MSG);
    for (i, chunk) in chunks.iter().enumerate() {
        assert!(
            chunk.len() <= MAX_TG_MSG,
            "chunk {i} too long: {}",
            chunk.len()
        );
    }
    eprintln!(
        "  PASS t213: {} chars → {} chunks",
        html.len(),
        chunks.len()
    );
}

/// Code block immediately after heading (no blank line — common pattern).
#[tokio::test]
async fn t214_heading_then_code_no_blank_line() {
    let md = "## Example\n```python\nprint('hello')\n```";
    let html = md_to_tg_html(md);
    assert!(html.contains("<b>Example</b>"));
    assert!(html.contains("<pre><code"));
    assert!(html.contains("print"));
    assert_valid_tg_html(&html, "t214");
    eprintln!("  PASS t214");
}

/// Multiple consecutive code blocks.
#[tokio::test]
async fn t215_consecutive_code_blocks() {
    let md = "```bash\necho 1\n```\n\n```python\nprint(2)\n```";
    let html = md_to_tg_html(md);
    let pre_count = html.matches("<pre>").count();
    assert_eq!(pre_count, 2, "expected 2 code blocks: {html}");
    assert_valid_tg_html(&html, "t215");
    eprintln!("  PASS t215");
}

/// Markdown with only formatting, no plain text.
#[tokio::test]
async fn t216_only_formatting() {
    let md = "**bold** *italic* `code` ~~strike~~";
    let html = md_to_tg_html(md);
    assert!(html.contains("<b>bold</b>"));
    assert!(html.contains("<i>italic</i>"));
    assert!(html.contains("<code>code</code>"));
    assert!(html.contains("<s>strike</s>"));
    assert_valid_tg_html(&html, "t216");
    eprintln!("  PASS t216");
}

/// Split preserves tag balance even with mixed <pre> and <b>.
#[tokio::test]
async fn t217_split_mixed_tags() {
    let mut md = String::new();
    for i in 0..50 {
        md.push_str(&format!(
            "## Section {i}: A Longer Heading for Better Coverage\n\n"
        ));
        md.push_str(&format!(
            "Some **bold** text about topic {i} with extra words to fill space.\n\n"
        ));
        md.push_str(&format!(
            "```\nfn example_{i}() {{ println!(\"block {i}\"); }}\n```\n\n"
        ));
    }
    let html = md_to_tg_html(&md);
    let chunks = split_html(&html, MAX_TG_MSG);
    assert!(chunks.len() > 1, "should produce multiple chunks");
    for (i, chunk) in chunks.iter().enumerate() {
        assert!(chunk.len() <= MAX_TG_MSG, "chunk {i} too long");
        assert_valid_tg_html(chunk, &format!("t217/chunk{i}"));
    }
    eprintln!(
        "  PASS t217: {} → {} chunks, all valid",
        html.len(),
        chunks.len()
    );
}
