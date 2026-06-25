use super::*;
use crate::tool::Tool;

#[test]
fn read_file_spec() {
    let tool = ReadFileTool::default();
    assert_eq!(tool.spec().name, "read_file");
    assert_eq!(tool.spec().permission, Permission::ReadOnly);
}

#[test]
fn write_file_spec() {
    let tool = WriteFileTool::default();
    assert_eq!(tool.spec().name, "write_file");
    assert_eq!(tool.spec().permission, Permission::WorkspaceWrite);
}

#[test]
fn edit_file_spec() {
    let tool = EditFileTool::default();
    assert_eq!(tool.spec().name, "edit_file");
    assert_eq!(tool.spec().permission, Permission::WorkspaceWrite);
}

#[tokio::test]
async fn write_then_read_file() {
    let dir = tempfile::tempdir().unwrap();

    let write_tool = WriteFileTool::default();
    let result = write_tool
        .execute(
            serde_json::json!({"file_path": "test.txt", "contents": "hello world"}),
            dir.path(),
        )
        .await;
    assert!(!result.is_error);
    assert!(result.output.contains("11 bytes"));

    let read_tool = ReadFileTool::default();
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

    let tool = ReadFileTool::default();
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
    let tool = ReadFileTool::default();
    let result = tool
        .execute(serde_json::json!({"file_path": "nope.txt"}), dir.path())
        .await;
    assert!(result.is_error);
    assert!(result.output.contains("Failed to read"));
}

#[tokio::test]
async fn write_creates_subdirectories() {
    let dir = tempfile::tempdir().unwrap();
    let tool = WriteFileTool::default();
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

    let tool = EditFileTool::default();
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

    let tool = EditFileTool::default();
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

    let tool = EditFileTool::default();
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
    let tool = ReadFileTool::default();
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
    let tool = WriteFileTool::default();
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
    let tool = EditFileTool::default();
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

    let tool = ReadFileTool::default();
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

    let tool = ReadFileTool::default();
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

    let tool = ReadFileTool::default();
    let result = tool
        .execute(serde_json::json!({"file_path": "ok.txt"}), dir.path())
        .await;
    assert!(!result.is_error);
    assert!(result.output.contains("hello"));
}

#[tokio::test]
async fn invalid_input_returns_error() {
    let tool = ReadFileTool::default();
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
    let tool = EditFileTool::default();
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
    let tool = EditFileTool::default();
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
    let tool = EditFileTool::default();
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
    let tool = EditFileTool::default();
    let result = tool
        .execute(serde_json::json!({"file_path": "f.txt"}), dir.path())
        .await;
    assert!(result.is_error);
    assert!(result.output.contains("No edits"));
}

#[tokio::test]
async fn atomic_replace_success_renames_complete_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("atomic.txt");
    std::fs::write(&path, b"old bytes").unwrap();

    crate::tool::atomic_write::atomic_replace_file(&path, b"new complete bytes")
        .await
        .unwrap();

    assert_eq!(std::fs::read(&path).unwrap(), b"new complete bytes");
    assert_no_atomic_temps(dir.path(), "atomic.txt");
}

#[tokio::test]
async fn atomic_replace_failure_leaves_original() {
    let dir = tempfile::tempdir().unwrap();
    let protected = dir.path().join("protected");
    std::fs::create_dir(&protected).unwrap();
    let path = protected.join("original.txt");
    std::fs::write(&path, b"original bytes").unwrap();

    let original_mode = make_read_only_dir(&protected);
    let result = crate::tool::atomic_write::atomic_replace_file(&path, b"replacement bytes").await;
    restore_dir_mode(&protected, original_mode);

    assert!(result.is_err());
    assert_eq!(std::fs::read(&path).unwrap(), b"original bytes");
}

#[tokio::test]
async fn atomic_replace_cleans_temp_on_error() {
    let dir = tempfile::tempdir().unwrap();
    let target_dir = dir.path().join("target-dir");
    std::fs::create_dir(&target_dir).unwrap();
    std::fs::write(target_dir.join("child.txt"), b"keeps directory non-empty").unwrap();

    let result =
        crate::tool::atomic_write::atomic_replace_file(&target_dir, b"not a directory").await;

    assert!(result.is_err());
    assert!(target_dir.is_dir());
    assert_no_atomic_temps(dir.path(), "target-dir");
}

#[tokio::test]
async fn write_file_uses_atomic_replace_success() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("write.txt");
    std::fs::write(&path, "before").unwrap();

    let tool = WriteFileTool::default();
    let result = tool
        .execute(
            serde_json::json!({"file_path": "write.txt", "contents": "after"}),
            dir.path(),
        )
        .await;

    assert!(!result.is_error, "{}", result.output);
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "after");
    assert_no_atomic_temps(dir.path(), "write.txt");
}

#[tokio::test]
async fn edit_file_uses_atomic_replace_success() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("edit-atomic.txt");
    std::fs::write(&path, "alpha beta gamma").unwrap();

    let tool = EditFileTool::default();
    let result = tool
        .execute(
            serde_json::json!({
                "file_path": "edit-atomic.txt",
                "old_string": "beta",
                "new_string": "BETA"
            }),
            dir.path(),
        )
        .await;

    assert!(!result.is_error, "{}", result.output);
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "alpha BETA gamma");
    assert_no_atomic_temps(dir.path(), "edit-atomic.txt");
}

#[tokio::test]
async fn edit_file_multi_edit_validation_no_write_on_failure() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("multi-validation.txt");
    let original = b"one\ntwo\nthree\n";
    std::fs::write(&path, original).unwrap();

    let tool = EditFileTool::default();
    let result = tool
        .execute(
            serde_json::json!({
                "file_path": "multi-validation.txt",
                "edits": [
                    { "old_string": "one", "new_string": "ONE" },
                    { "old_string": "absent", "new_string": "ABSENT" }
                ]
            }),
            dir.path(),
        )
        .await;

    assert!(result.is_error);
    assert!(result.output.contains("edits[1]"));
    assert_eq!(std::fs::read(&path).unwrap(), original);
    assert_no_atomic_temps(dir.path(), "multi-validation.txt");
}

#[tokio::test]
async fn edit_rejects_when_expected_content_sha_mismatch_no_write() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stale.txt");
    let original = b"alpha\nbeta\n";
    std::fs::write(&path, original).unwrap();

    let wrong_sha = crate::tool::edit_guard::sha256_hex(b"alpha\nold beta\n");
    let tool = EditFileTool::new(true);
    let result = tool
        .execute(
            serde_json::json!({
                "file_path": "stale.txt",
                "old_string": "beta",
                "new_string": "BETA",
                "expected_content_sha": wrong_sha
            }),
            dir.path(),
        )
        .await;

    assert!(result.is_error);
    assert!(result.output.contains("file changed since you read it"));
    assert_eq!(std::fs::read(&path).unwrap(), original);
}

#[tokio::test]
async fn edit_applies_when_expected_content_sha_matches() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fresh.txt");
    let original = "alpha\nbeta\n";
    std::fs::write(&path, original).unwrap();

    let expected_sha = crate::tool::edit_guard::sha256_hex(original.as_bytes());
    let tool = EditFileTool::new(true);
    let result = tool
        .execute(
            serde_json::json!({
                "file_path": "fresh.txt",
                "old_string": "beta",
                "new_string": "BETA",
                "expected_content_sha": expected_sha
            }),
            dir.path(),
        )
        .await;

    assert!(!result.is_error, "{}", result.output);
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "alpha\nBETA\n");
}

#[tokio::test]
async fn edit_flag_off_ignores_expected_sha_legacy_behavior() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("legacy.txt");
    std::fs::write(&path, "alpha\nbeta\n").unwrap();

    let wrong_sha = crate::tool::edit_guard::sha256_hex(b"different");
    let tool = EditFileTool::default();
    let result = tool
        .execute(
            serde_json::json!({
                "file_path": "legacy.txt",
                "old_string": "beta",
                "new_string": "BETA",
                "expected_content_sha": wrong_sha
            }),
            dir.path(),
        )
        .await;

    assert!(!result.is_error, "{}", result.output);
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "alpha\nBETA\n");
}

#[tokio::test]
async fn snapshot_emits_full_sha_and_anchor_hashes() {
    let dir = tempfile::tempdir().unwrap();
    let content = "alpha\nbeta\ngamma\n";
    std::fs::write(dir.path().join("snap.txt"), content).unwrap();

    let snapshot = FileSnapshotTool::default();
    let result = snapshot
        .execute(
            serde_json::json!({"file_path": "snap.txt", "offset": 2, "limit": 1}),
            dir.path(),
        )
        .await;
    assert!(!result.is_error, "{}", result.output);
    assert!(result.output.contains(&format!(
        "full_sha256: {}",
        crate::tool::edit_guard::sha256_hex(content.as_bytes())
    )));
    let beta_hash = crate::tool::edit_guard::sha256_hex(b"beta\n");
    assert!(result.output.contains(&format!("     2|{beta_hash}")));
    assert!(result.output.contains("text:\n     2|beta"));

    let read = ReadFileTool::default();
    let read_result = read
        .execute(serde_json::json!({"file_path": "snap.txt"}), dir.path())
        .await;
    assert!(!read_result.is_error, "{}", read_result.output);
    assert_eq!(
        read_result.output,
        "     1|alpha\n     2|beta\n     3|gamma"
    );
}

#[tokio::test]
async fn hashline_applies_valid_anchor() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("hash.txt");
    let original = "alpha\nbeta\ngamma\n";
    std::fs::write(&path, original).unwrap();

    let anchor = hash_for_range(original, 2, 2);
    let tool = EditFileTool::new(false).with_hashline_edit(true);
    let result = tool
        .execute(
            serde_json::json!({
                "file_path": "hash.txt",
                "hash_edits": [{
                    "start_line": 2,
                    "end_line": 2,
                    "anchor_hash": anchor,
                    "new_text": "BETA\n"
                }]
            }),
            dir.path(),
        )
        .await;

    assert!(!result.is_error, "{}", result.output);
    assert_eq!(
        std::fs::read_to_string(&path).unwrap(),
        "alpha\nBETA\ngamma\n"
    );
}

#[tokio::test]
async fn hashline_rejects_stale_anchor_without_write() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stale-hash.txt");
    let original = b"alpha\nbeta\ngamma\n";
    std::fs::write(&path, original).unwrap();

    let stale_anchor = crate::tool::edit_guard::sha256_hex(b"old beta\n");
    let tool = EditFileTool::new(false).with_hashline_edit(true);
    let result = tool
        .execute(
            serde_json::json!({
                "file_path": "stale-hash.txt",
                "hash_edits": [{
                    "start_line": 2,
                    "end_line": 2,
                    "anchor_hash": stale_anchor,
                    "new_text": "BETA\n"
                }]
            }),
            dir.path(),
        )
        .await;

    assert!(result.is_error);
    assert!(result.output.contains("stale or ambiguous"));
    assert_eq!(std::fs::read(&path).unwrap(), original);
}

#[tokio::test]
async fn hashline_stale_anchor_does_not_fallback_to_old_string() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("no-fallback.txt");
    let original = b"alpha\nlegacy-target\ngamma\n";
    std::fs::write(&path, original).unwrap();

    let stale_anchor = crate::tool::edit_guard::sha256_hex(b"not live\n");
    let tool = EditFileTool::new(false).with_hashline_edit(true);
    let result = tool
        .execute(
            serde_json::json!({
                "file_path": "no-fallback.txt",
                "old_string": "legacy-target",
                "new_string": "SHOULD_NOT_APPLY",
                "hash_edits": [{
                    "start_line": 2,
                    "end_line": 2,
                    "anchor_hash": stale_anchor,
                    "new_text": "BETA\n"
                }]
            }),
            dir.path(),
        )
        .await;

    assert!(result.is_error);
    assert_eq!(std::fs::read(&path).unwrap(), original);
}

#[tokio::test]
async fn hashline_rejects_overlapping_ranges() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("overlap.txt");
    let original = b"one\ntwo\nthree\nfour\n";
    std::fs::write(&path, original).unwrap();
    let original_text = std::str::from_utf8(original).unwrap();

    let tool = EditFileTool::new(false).with_hashline_edit(true);
    let result = tool
        .execute(
            serde_json::json!({
                "file_path": "overlap.txt",
                "hash_edits": [
                    {
                        "start_line": 2,
                        "end_line": 3,
                        "anchor_hash": hash_for_range(original_text, 2, 3),
                        "new_text": "TWO_THREE\n"
                    },
                    {
                        "start_line": 3,
                        "end_line": 4,
                        "anchor_hash": hash_for_range(original_text, 3, 4),
                        "new_text": "THREE_FOUR\n"
                    }
                ]
            }),
            dir.path(),
        )
        .await;

    assert!(result.is_error);
    assert!(result.output.contains("overlap"));
    assert_eq!(std::fs::read(&path).unwrap(), original);
}

#[tokio::test]
async fn hashline_multi_edit_all_or_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("all-or-nothing.txt");
    let original = b"one\ntwo\nthree\n";
    std::fs::write(&path, original).unwrap();
    let original_text = std::str::from_utf8(original).unwrap();

    let tool = EditFileTool::new(false).with_hashline_edit(true);
    let result = tool
        .execute(
            serde_json::json!({
                "file_path": "all-or-nothing.txt",
                "hash_edits": [
                    {
                        "start_line": 1,
                        "end_line": 1,
                        "anchor_hash": hash_for_range(original_text, 1, 1),
                        "new_text": "ONE\n"
                    },
                    {
                        "start_line": 3,
                        "end_line": 3,
                        "anchor_hash": crate::tool::edit_guard::sha256_hex(b"old three\n"),
                        "new_text": "THREE\n"
                    }
                ]
            }),
            dir.path(),
        )
        .await;

    assert!(result.is_error);
    assert_eq!(std::fs::read(&path).unwrap(), original);
}

#[tokio::test]
async fn legacy_edit_unchanged_with_hashline_flag() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("legacy-hash-flag.txt");
    std::fs::write(&path, "alpha\nbeta\n").unwrap();

    let tool = EditFileTool::new(false).with_hashline_edit(true);
    let result = tool
        .execute(
            serde_json::json!({
                "file_path": "legacy-hash-flag.txt",
                "old_string": "beta",
                "new_string": "BETA"
            }),
            dir.path(),
        )
        .await;

    assert!(!result.is_error, "{}", result.output);
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "alpha\nBETA\n");
}

#[tokio::test]
async fn hashline_flag_off_rejects_without_legacy_fallback() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("disabled.txt");
    let original = b"alpha\nbeta\n";
    std::fs::write(&path, original).unwrap();
    let original_text = std::str::from_utf8(original).unwrap();

    let tool = EditFileTool::new(false);
    let result = tool
        .execute(
            serde_json::json!({
                "file_path": "disabled.txt",
                "old_string": "beta",
                "new_string": "SHOULD_NOT_APPLY",
                "hash_edits": [{
                    "start_line": 2,
                    "end_line": 2,
                    "anchor_hash": hash_for_range(original_text, 2, 2),
                    "new_text": "BETA\n"
                }]
            }),
            dir.path(),
        )
        .await;

    assert!(result.is_error);
    assert!(result.output.contains("disabled"));
    assert_eq!(std::fs::read(&path).unwrap(), original);
}

#[tokio::test]
async fn read_file_output_unchanged_with_cache() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("same.txt");
    std::fs::write(&path, "alpha\nbeta\ngamma\n").unwrap();

    let direct = ReadFileTool::default()
        .execute(
            serde_json::json!({"file_path": "same.txt", "offset": 2, "limit": 2}),
            dir.path(),
        )
        .await;
    let cached = ReadFileTool::new(Some(std::sync::Arc::new(
        crate::tool::fs_cache::FsCache::new(1024 * 1024),
    )))
    .execute(
        serde_json::json!({"file_path": "same.txt", "offset": 2, "limit": 2}),
        dir.path(),
    )
    .await;

    assert!(!direct.is_error, "{}", direct.output);
    assert!(!cached.is_error, "{}", cached.output);
    assert_eq!(cached.output, direct.output);
}

#[tokio::test]
async fn edit_bypasses_cache_for_authoritative_apply() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("edit-cache.txt");
    std::fs::write(&path, "alpha\nbeta\n").unwrap();
    let cache = std::sync::Arc::new(crate::tool::fs_cache::FsCache::new(1024 * 1024));

    let read = ReadFileTool::new(Some(cache.clone()))
        .execute(
            serde_json::json!({"file_path": "edit-cache.txt"}),
            dir.path(),
        )
        .await;
    assert!(!read.is_error, "{}", read.output);
    assert_eq!(cache.entry_count(), 1);

    std::fs::write(&path, "alpha\nexternal\n").unwrap();
    let edit = EditFileTool::new(false).with_fs_cache(Some(cache.clone()));
    let result = edit
        .execute(
            serde_json::json!({
                "file_path": "edit-cache.txt",
                "old_string": "external",
                "new_string": "edited"
            }),
            dir.path(),
        )
        .await;

    assert!(!result.is_error, "{}", result.output);
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "alpha\nedited\n");
    assert_eq!(cache.entry_count(), 0);
}

fn hash_for_range(content: &str, start_line: usize, end_line: usize) -> String {
    let text = crate::tool::edit_guard::range_text_by_line(content, start_line, end_line).unwrap();
    crate::tool::edit_guard::sha256_hex(text.as_bytes())
}

fn assert_no_atomic_temps(dir: &Path, file_name: &str) {
    let prefix = format!(".{file_name}.tmp.");
    let leftovers: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_name().to_string_lossy().starts_with(&prefix))
        .map(|entry| entry.path())
        .collect();
    assert!(leftovers.is_empty(), "leftover temp files: {leftovers:?}");
}

#[cfg(unix)]
fn make_read_only_dir(path: &Path) -> Option<std::fs::Permissions> {
    use std::os::unix::fs::PermissionsExt;

    let original = std::fs::metadata(path).unwrap().permissions();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o555)).unwrap();
    Some(original)
}

#[cfg(not(unix))]
fn make_read_only_dir(_path: &Path) -> Option<std::fs::Permissions> {
    None
}

#[cfg(unix)]
fn restore_dir_mode(path: &Path, original: Option<std::fs::Permissions>) {
    if let Some(original) = original {
        std::fs::set_permissions(path, original).unwrap();
    }
}

#[cfg(not(unix))]
fn restore_dir_mode(_path: &Path, _original: Option<std::fs::Permissions>) {}
