//! Integration tests validating Phase 1-4 changes without Telegram.

use naked_core::session::FileTracker;
use naked_core::types::AgentEvent;

// ── A4: SAFE_TOOLS auto-approve ─────────────────────────────────────

#[tokio::test]
async fn safe_tools_auto_approve() {
    use naked_tg::channel_map::ChannelSessionMap;

    let map = ChannelSessionMap::new();
    // Safe tools don't need yolo or allow_list
    assert!(map.should_auto_approve(1, None, "read_file").await);
    assert!(map.should_auto_approve(1, None, "web_search").await);
    assert!(map.should_auto_approve(1, None, "glob_search").await);
    assert!(map.should_auto_approve(1, None, "grep_search").await);
    assert!(map.should_auto_approve(1, None, "agent_status").await);

    // Unsafe tools still need approval
    assert!(!map.should_auto_approve(1, None, "bash").await);
    assert!(!map.should_auto_approve(1, None, "write_file").await);
    assert!(!map.should_auto_approve(1, None, "edit_file").await);
    assert!(!map.should_auto_approve(1, None, "sub_agent").await);
}

// ── B1: Multi-edit tool ─────────────────────────────────────────────

#[tokio::test]
async fn multi_edit_three_replacements() {
    use naked_core::tool::Tool;
    use naked_core::tool::file_ops::EditFileTool;

    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("test.rs"),
        "fn old_a() {}\nfn old_b() {}\nfn old_c() {}\n",
    )
    .unwrap();

    let tool = EditFileTool;
    let result = tool
        .execute(
            serde_json::json!({
                "file_path": "test.rs",
                "edits": [
                    {"old_string": "fn old_a() {}", "new_string": "fn new_a() {}"},
                    {"old_string": "fn old_b() {}", "new_string": "fn new_b() {}"},
                    {"old_string": "fn old_c() {}", "new_string": "fn new_c() {}"}
                ]
            }),
            dir.path(),
        )
        .await;

    assert!(!result.is_error, "edit failed: {}", result.output);
    assert!(result.output.contains("3 replacements"));
    // C2: diff preview
    assert!(result.output.contains("#1:"));
    assert!(result.output.contains("#3:"));

    let content = std::fs::read_to_string(dir.path().join("test.rs")).unwrap();
    assert_eq!(content, "fn new_a() {}\nfn new_b() {}\nfn new_c() {}\n");
}

#[tokio::test]
async fn multi_edit_backward_compat() {
    use naked_core::tool::Tool;
    use naked_core::tool::file_ops::EditFileTool;

    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("f.txt"), "old text here").unwrap();

    let tool = EditFileTool;
    // Legacy format (no edits array)
    let result = tool
        .execute(
            serde_json::json!({
                "file_path": "f.txt",
                "old_string": "old text",
                "new_string": "new text"
            }),
            dir.path(),
        )
        .await;

    assert!(!result.is_error);
    assert_eq!(
        std::fs::read_to_string(dir.path().join("f.txt")).unwrap(),
        "new text here"
    );
}

// ── B3: FileTracker ─────────────────────────────────────────────────

#[test]
fn file_tracker_from_tool_calls() {
    let mut ft = FileTracker::default();

    // Simulate tool calls
    ft.record_tool(
        "read_file",
        &serde_json::json!({"file_path": "src/main.rs"}),
    );
    ft.record_tool("read_file", &serde_json::json!({"file_path": "src/lib.rs"}));
    ft.record_tool("edit_file", &serde_json::json!({"file_path": "src/lib.rs"}));
    ft.record_tool(
        "write_file",
        &serde_json::json!({"file_path": "new_file.rs"}),
    );
    ft.record_tool("bash", &serde_json::json!({"command": "cargo build"}));

    // read_only: only main.rs (lib.rs was also edited)
    let ro = ft.read_only();
    assert!(ro.contains(&"src/main.rs"));
    assert!(!ro.contains(&"src/lib.rs"));

    // modified: lib.rs + new_file.rs
    let modified = ft.modified();
    assert!(modified.contains(&"src/lib.rs"));
    assert!(modified.contains(&"new_file.rs"));
    assert!(!modified.contains(&"src/main.rs"));
}

// ── B8: Bash truncation saves to file ───────────────────────────────

#[tokio::test]
async fn bash_large_output_saves_log() {
    use naked_core::tool::Tool;
    use naked_core::tool::bash::BashTool;

    let tool = BashTool::new(10);
    let result = tool
        .execute(
            serde_json::json!({
                "command": "dd if=/dev/zero bs=1024 count=20 2>/dev/null | base64"
            }),
            std::path::Path::new("/tmp"),
        )
        .await;

    assert!(!result.is_error);
    assert!(
        result.output.contains("[truncated:"),
        "should be truncated: tail={}",
        &result.output[result.output.len().saturating_sub(200)..]
    );
    assert!(result.output.contains("/tmp/naked_bash_"));

    // Extract and verify temp file exists
    if let Some(start) = result.output.find("/tmp/naked_bash_") {
        let end = result.output[start..].find(']').unwrap() + start;
        let path = &result.output[start..end];
        assert!(
            std::path::Path::new(path).exists(),
            "temp file should exist: {path}"
        );
        let full = std::fs::read_to_string(path).unwrap();
        assert!(full.len() > 16_000, "full output should have all lines");
        let _ = std::fs::remove_file(path);
    }
}

// ── A6: Tool error expanded output ──────────────────────────────────

#[tokio::test]
async fn bash_error_output_not_truncated_to_80() {
    use naked_core::tool::Tool;
    use naked_core::tool::bash::BashTool;

    let tool = BashTool::new(10);
    // Command that produces a long error message
    let result = tool
        .execute(
            serde_json::json!({
                "command": "echo 'error: this is a very long error message that exceeds eighty characters and should be fully visible to the user for debugging purposes' >&2; exit 1"
            }),
            std::path::Path::new("/tmp"),
        )
        .await;

    assert!(result.is_error);
    // The full error should be in the output (not truncated to 80 chars)
    assert!(
        result.output.contains("fully visible to the user"),
        "error should contain full text: {}",
        result.output
    );
}

// ── C1: File lock prevents race conditions ──────────────────────────

#[tokio::test]
async fn file_lock_serializes_access() {
    use naked_core::tool::file_lock::lock_file;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("race.txt");
    std::fs::write(&path, "0").unwrap();

    let mut handles = vec![];
    for _ in 0..10 {
        let p = path.clone();
        handles.push(tokio::spawn(async move {
            let _guard = lock_file(&p).await;
            let val: i32 = std::fs::read_to_string(&p).unwrap().trim().parse().unwrap();
            // Simulate read-modify-write
            tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
            std::fs::write(&p, (val + 1).to_string()).unwrap();
        }));
    }

    for h in handles {
        h.await.unwrap();
    }

    let final_val: i32 = std::fs::read_to_string(&path)
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    assert_eq!(final_val, 10, "without lock this would be less than 10");
}

// ── A7: MCP connect_all_with_diagnostics ────────────────────────────

#[test]
fn mcp_connect_failure_struct() {
    use naked_core::mcp::client::McpConnectFailure;

    let f = McpConnectFailure {
        name: "playwright".to_string(),
        error: "empty response".to_string(),
    };
    assert_eq!(f.name, "playwright");
    assert!(f.error.contains("empty"));
}

// ── A8: ContextCompacted event fields ───────────────────────────────

#[test]
fn context_compacted_event_has_summary_hint() {
    let event = AgentEvent::ContextCompacted {
        before_msgs: 15,
        after_msgs: 3,
        summary_hint: Some("Fix Caddy routing".to_string()),
        files_count: 5,
    };

    match event {
        AgentEvent::ContextCompacted {
            before_msgs,
            after_msgs,
            summary_hint,
            files_count,
        } => {
            assert_eq!(before_msgs, 15);
            assert_eq!(after_msgs, 3);
            assert_eq!(summary_hint.unwrap(), "Fix Caddy routing");
            assert_eq!(files_count, 5);
        }
        _ => panic!("wrong variant"),
    }
}

// ── read_file returns base64 for images ─────────────────────────

#[tokio::test]
async fn read_file_image_returns_base64() {
    use naked_core::tool::Tool;
    use naked_core::tool::file_ops::ReadFileTool;

    let dir = tempfile::tempdir().unwrap();
    // Minimal valid PNG (1x1 pixel, red)
    let png_data: &[u8] = &[
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG header
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, // IHDR chunk
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, // 1x1
        0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53, 0xDE, // 8-bit RGB
        0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, 0x54, // IDAT chunk
        0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00, 0x00, 0x00, 0x02, 0x00, 0x01, 0xE2, 0x21, 0xBC,
        0x33, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, // IEND
        0xAE, 0x42, 0x60, 0x82,
    ];
    std::fs::write(dir.path().join("test.png"), png_data).unwrap();

    let tool = ReadFileTool;
    let result = tool
        .execute(serde_json::json!({"file_path": "test.png"}), dir.path())
        .await;

    assert!(!result.is_error, "should NOT be error: {}", result.output);
    // Image is pushed to image_result collector, text says "[image sent to vision model]"
    assert!(
        result.output.contains("image sent to vision model")
            || result.output.contains("data:image"),
        "should indicate image handling: {}",
        &result.output[..100.min(result.output.len())]
    );
    assert!(result.output.contains("PNG"), "should mention PNG");

    // Verify image was pushed to collector
    let images = naked_core::tool::image_result::drain_images();
    assert_eq!(images.len(), 1, "should have 1 image in collector");
    assert_eq!(images[0].0, "image/png");
}
