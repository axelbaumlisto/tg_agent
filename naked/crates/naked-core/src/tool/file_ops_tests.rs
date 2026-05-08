use super::*;
use crate::tool::Tool;

#[test]
fn read_file_spec() {
    let tool = ReadFileTool;
    assert_eq!(tool.spec().name, "read_file");
    assert_eq!(tool.spec().permission, Permission::ReadOnly);
}

#[test]
fn write_file_spec() {
    let tool = WriteFileTool;
    assert_eq!(tool.spec().name, "write_file");
    assert_eq!(tool.spec().permission, Permission::WorkspaceWrite);
}

#[test]
fn edit_file_spec() {
    let tool = EditFileTool;
    assert_eq!(tool.spec().name, "edit_file");
    assert_eq!(tool.spec().permission, Permission::WorkspaceWrite);
}

#[tokio::test]
async fn write_then_read_file() {
    let dir = tempfile::tempdir().unwrap();

    let write_tool = WriteFileTool;
    let result = write_tool
        .execute(
            serde_json::json!({"file_path": "test.txt", "contents": "hello world"}),
            dir.path(),
        )
        .await;
    assert!(!result.is_error);
    assert!(result.output.contains("11 bytes"));

    let read_tool = ReadFileTool;
    let result = read_tool
        .execute(serde_json::json!({"file_path": "test.txt"}), dir.path())
        .await;
    assert!(!result.is_error);
    assert!(result.output.contains("hello world"));
}

#[tokio::test]
async fn read_file_with_offset_and_limit() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("lines.txt"), "line1\nline2\nline3\nline4\n").unwrap();

    let tool = ReadFileTool;
    let result = tool
        .execute(
            serde_json::json!({"file_path": "lines.txt", "offset": 2, "limit": 2}),
            dir.path(),
        )
        .await;
    assert!(!result.is_error);
    assert!(result.output.contains("line2"));
    assert!(result.output.contains("line3"));
    assert!(!result.output.contains("line1"));
    assert!(!result.output.contains("line4"));
}

#[tokio::test]
async fn read_nonexistent_file_errors() {
    let dir = tempfile::tempdir().unwrap();
    let tool = ReadFileTool;
    let result = tool
        .execute(serde_json::json!({"file_path": "nope.txt"}), dir.path())
        .await;
    assert!(result.is_error);
    assert!(result.output.contains("Failed to read"));
}

#[tokio::test]
async fn write_creates_subdirectories() {
    let dir = tempfile::tempdir().unwrap();
    let tool = WriteFileTool;
    let result = tool
        .execute(
            serde_json::json!({"file_path": "sub/dir/file.txt", "contents": "nested"}),
            dir.path(),
        )
        .await;
    assert!(!result.is_error);
    assert!(dir.path().join("sub/dir/file.txt").exists());
}

#[tokio::test]
async fn edit_file_replaces_text() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("edit.txt"), "hello world").unwrap();

    let tool = EditFileTool;
    let result = tool
        .execute(
            serde_json::json!({
                "file_path": "edit.txt",
                "old_string": "world",
                "new_string": "rust"
            }),
            dir.path(),
        )
        .await;
    assert!(!result.is_error);

    let content = std::fs::read_to_string(dir.path().join("edit.txt")).unwrap();
    assert_eq!(content, "hello rust");
}

#[tokio::test]
async fn edit_file_not_found_old_string() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("e.txt"), "abc").unwrap();

    let tool = EditFileTool;
    let result = tool
        .execute(
            serde_json::json!({
                "file_path": "e.txt",
                "old_string": "xyz",
                "new_string": "123"
            }),
            dir.path(),
        )
        .await;
    assert!(result.is_error);
    assert!(result.output.contains("not found"));
}

#[tokio::test]
async fn edit_file_rejects_non_unique_match() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("dup.txt"), "aa bb aa").unwrap();

    let tool = EditFileTool;
    let result = tool
        .execute(
            serde_json::json!({
                "file_path": "dup.txt",
                "old_string": "aa",
                "new_string": "cc"
            }),
            dir.path(),
        )
        .await;
    assert!(result.is_error);
    assert!(result.output.contains("2 times"));
}

#[tokio::test]
async fn read_outside_workspace_allowed() {
    let dir = tempfile::tempdir().unwrap();
    let tool = ReadFileTool;
    let result = tool
        .execute(
            serde_json::json!({"file_path": "/etc/hostname"}),
            dir.path(),
        )
        .await;
    assert!(!result.is_error || result.output.contains("Failed to read"));
}

#[tokio::test]
async fn write_outside_workspace_escalates_permission() {
    let dir = tempfile::tempdir().unwrap();
    let tool = WriteFileTool;
    let perm = tool.effective_permission(
        &serde_json::json!({"file_path": "/tmp/outside.txt", "contents": "x"}),
        dir.path(),
    );
    assert_eq!(perm, Permission::Dangerous);

    let perm_inside = tool.effective_permission(
        &serde_json::json!({"file_path": "inside.txt", "contents": "x"}),
        dir.path(),
    );
    assert_eq!(perm_inside, Permission::WorkspaceWrite);
}

#[tokio::test]
async fn edit_outside_workspace_escalates_permission() {
    let dir = tempfile::tempdir().unwrap();
    let tool = EditFileTool;
    let perm = tool.effective_permission(
        &serde_json::json!({"file_path": "/tmp/outside.txt", "old_string": "a", "new_string": "b"}),
        dir.path(),
    );
    assert_eq!(perm, Permission::Dangerous);
}

#[tokio::test]
async fn read_image_file_returns_vision_result() {
    let dir = tempfile::tempdir().unwrap();
    let bin = dir.path().join("image.png");
    // Write a minimal valid-looking PNG header.
    std::fs::write(&bin, b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR").unwrap();

    let tool = ReadFileTool;
    let result = tool
        .execute(serde_json::json!({"file_path": "image.png"}), dir.path())
        .await;
    // Vision pipeline: images succeed and push to image_result.
    assert!(!result.is_error, "images should succeed: {}", result.output);
    assert!(result.output.contains("image sent to vision model"));
}

#[tokio::test]
async fn read_binary_non_image_returns_error() {
    let dir = tempfile::tempdir().unwrap();
    let bin = dir.path().join("data.bin");
    std::fs::write(&bin, b"\x00\x01\x02\x03\x04\x05").unwrap();

    let tool = ReadFileTool;
    let result = tool
        .execute(serde_json::json!({"file_path": "data.bin"}), dir.path())
        .await;
    assert!(result.is_error);
    assert!(result.output.contains("Binary file"));
}

#[tokio::test]
async fn read_text_file_with_no_nul_ok() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("ok.txt"), "hello\nworld\n").unwrap();

    let tool = ReadFileTool;
    let result = tool
        .execute(serde_json::json!({"file_path": "ok.txt"}), dir.path())
        .await;
    assert!(!result.is_error);
    assert!(result.output.contains("hello"));
}

#[tokio::test]
async fn invalid_input_returns_error() {
    let tool = ReadFileTool;
    let result = tool
        .execute(serde_json::json!({"wrong": 123}), Path::new("/tmp"))
        .await;
    assert!(result.is_error);
    assert!(result.output.contains("Invalid input"));
}

// ── B1: Multi-edit tests ──────────────────────────────────────

#[tokio::test]
async fn edit_file_legacy_single() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("f.txt"), "hello world").unwrap();
    let tool = EditFileTool;
    let result = tool
        .execute(
            serde_json::json!({
                "file_path": "f.txt",
                "old_string": "hello",
                "new_string": "goodbye"
            }),
            dir.path(),
        )
        .await;
    assert!(!result.is_error, "{}", result.output);
    assert_eq!(
        std::fs::read_to_string(dir.path().join("f.txt")).unwrap(),
        "goodbye world"
    );
}

#[tokio::test]
async fn edit_file_multi_edits() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("f.txt"), "aaa\nbbb\nccc\n").unwrap();
    let tool = EditFileTool;
    let result = tool
        .execute(
            serde_json::json!({
                "file_path": "f.txt",
                "edits": [
                    { "old_string": "aaa", "new_string": "AAA" },
                    { "old_string": "ccc", "new_string": "CCC" }
                ]
            }),
            dir.path(),
        )
        .await;
    assert!(!result.is_error, "{}", result.output);
    assert!(result.output.contains("2 replacements"));
    assert_eq!(
        std::fs::read_to_string(dir.path().join("f.txt")).unwrap(),
        "AAA\nbbb\nCCC\n"
    );
}

#[tokio::test]
async fn edit_file_atomic_rejects_missing() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("f.txt"), "aaa\nbbb\n").unwrap();
    let tool = EditFileTool;
    // Second edit has old_string not in file — entire operation should fail
    let result = tool
        .execute(
            serde_json::json!({
                "file_path": "f.txt",
                "edits": [
                    { "old_string": "aaa", "new_string": "AAA" },
                    { "old_string": "zzz", "new_string": "ZZZ" }
                ]
            }),
            dir.path(),
        )
        .await;
    assert!(result.is_error);
    assert!(result.output.contains("edits[1]"));
    // File should be UNCHANGED (atomic)
    assert_eq!(
        std::fs::read_to_string(dir.path().join("f.txt")).unwrap(),
        "aaa\nbbb\n"
    );
}

#[tokio::test]
async fn edit_file_no_edits_error() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("f.txt"), "x").unwrap();
    let tool = EditFileTool;
    let result = tool
        .execute(serde_json::json!({"file_path": "f.txt"}), dir.path())
        .await;
    assert!(result.is_error);
    assert!(result.output.contains("No edits"));
}
