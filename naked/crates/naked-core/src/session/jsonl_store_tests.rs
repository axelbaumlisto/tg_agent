use super::*;
use crate::session::SessionMetadata;
use base64::Engine;
use std::path::Path;

fn test_metadata() -> SessionMetadata {
    SessionMetadata {
        name: None,
        provider: "anthropic".into(),
        model: "claude-sonnet-4".into(),
        channel: "cli".into(),
        channel_id: None,
    }
}

fn test_session(dir: &Path) -> Session {
    Session::new(dir.to_path_buf(), "system prompt".into(), test_metadata())
}

#[tokio::test]
async fn save_and_load_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let store = JsonlSessionStore::new(dir.path().to_path_buf());

    let mut session = test_session(dir.path());
    session.history.push_user("hello");

    store.save(&session).await.unwrap();
    let loaded = store.load(&session.id).await.unwrap().unwrap();

    assert_eq!(loaded.id, session.id);
    assert_eq!(loaded.history.system_prompt(), "system prompt");
    assert_eq!(loaded.history.message_count(), 1);
    assert_eq!(loaded.history.messages()[0].text_content(), "hello");
    assert_eq!(loaded.state, SessionState::Sleeping);
}

#[tokio::test]
async fn load_nonexistent_returns_none() {
    let dir = tempfile::tempdir().unwrap();
    let store = JsonlSessionStore::new(dir.path().to_path_buf());
    let result = store.load("nonexistent").await.unwrap();
    assert!(result.is_none());
}

#[tokio::test]
async fn list_returns_saved_sessions() {
    let dir = tempfile::tempdir().unwrap();
    let store = JsonlSessionStore::new(dir.path().to_path_buf());

    let s1 = test_session(dir.path());
    let s2 = test_session(dir.path());
    store.save(&s1).await.unwrap();
    store.save(&s2).await.unwrap();

    let list = store.list().await.unwrap();
    assert_eq!(list.len(), 2);
}

#[tokio::test]
async fn list_empty_dir_returns_empty() {
    let dir = tempfile::tempdir().unwrap();
    let store = JsonlSessionStore::new(dir.path().join("nonexistent"));
    let list = store.list().await.unwrap();
    assert!(list.is_empty());
}

#[tokio::test]
async fn delete_removes_session() {
    let dir = tempfile::tempdir().unwrap();
    let store = JsonlSessionStore::new(dir.path().to_path_buf());

    let session = test_session(dir.path());
    store.save(&session).await.unwrap();
    assert!(store.load(&session.id).await.unwrap().is_some());

    store.delete(&session.id).await.unwrap();
    assert!(store.load(&session.id).await.unwrap().is_none());
}

#[tokio::test]
async fn delete_nonexistent_is_ok() {
    let dir = tempfile::tempdir().unwrap();
    let store = JsonlSessionStore::new(dir.path().to_path_buf());
    store.delete("no-such-session").await.unwrap();
}

#[tokio::test]
async fn append_message_adds_to_file() {
    let dir = tempfile::tempdir().unwrap();
    let store = JsonlSessionStore::new(dir.path().to_path_buf());

    let session = test_session(dir.path());
    store.save(&session).await.unwrap();

    let msg = ConversationMessage::user("appended msg");
    store.append_message(&session.id, &msg).await.unwrap();

    let loaded = store.load(&session.id).await.unwrap().unwrap();
    assert_eq!(loaded.history.message_count(), 1);
}

#[tokio::test]
async fn save_with_multiple_messages() {
    let dir = tempfile::tempdir().unwrap();
    let store = JsonlSessionStore::new(dir.path().to_path_buf());

    let mut session = test_session(dir.path());
    session.history.push_user("q1");
    session.history.push_assistant(
        vec![crate::types::ContentBlock::Text { text: "a1".into() }],
        None,
    );
    session.history.push_user("q2");

    store.save(&session).await.unwrap();
    let loaded = store.load(&session.id).await.unwrap().unwrap();

    assert_eq!(loaded.history.message_count(), 3);
    assert_eq!(loaded.history.messages()[0].text_content(), "q1");
    assert_eq!(loaded.history.messages()[1].text_content(), "a1");
    assert_eq!(loaded.history.messages()[2].text_content(), "q2");
}

#[tokio::test]
async fn save_preserves_fork_info() {
    let dir = tempfile::tempdir().unwrap();
    let store = JsonlSessionStore::new(dir.path().to_path_buf());

    let parent = test_session(dir.path());
    let forked = parent.fork(Some("experiment".into()));
    store.save(&forked).await.unwrap();

    let loaded = store.load(&forked.id).await.unwrap().unwrap();
    let fi = loaded.fork_info.unwrap();
    assert_eq!(fi.parent_session_id, parent.id);
    assert_eq!(fi.branch_name.as_deref(), Some("experiment"));
}

#[tokio::test]
async fn save_creates_directory_layout() {
    let dir = tempfile::tempdir().unwrap();
    let store = JsonlSessionStore::new(dir.path().to_path_buf());

    let session = test_session(dir.path());
    store.save(&session).await.unwrap();

    let session_dir = dir.path().join(&session.id);
    assert!(session_dir.is_dir(), "session dir must exist");
    assert!(
        session_dir.join("session.jsonl").is_file(),
        "session.jsonl must exist"
    );
    assert!(
        session_dir.join("artifacts").is_dir(),
        "artifacts/ must exist"
    );
}

#[tokio::test]
async fn artifacts_dir_is_isolated() {
    let dir = tempfile::tempdir().unwrap();
    let store = JsonlSessionStore::new(dir.path().to_path_buf());

    let s1 = test_session(dir.path());
    let s2 = test_session(dir.path());
    store.save(&s1).await.unwrap();
    store.save(&s2).await.unwrap();

    let a1 = store.artifacts_dir(&s1.id);
    let a2 = store.artifacts_dir(&s2.id);
    assert_ne!(a1, a2, "each session must have its own artifacts dir");

    std::fs::write(a1.join("file.txt"), "session1").unwrap();
    assert!(!a2.join("file.txt").exists(), "artifacts must be isolated");
}

#[tokio::test]
async fn delete_removes_entire_directory() {
    let dir = tempfile::tempdir().unwrap();
    let store = JsonlSessionStore::new(dir.path().to_path_buf());

    let session = test_session(dir.path());
    store.save(&session).await.unwrap();

    let a = store.artifacts_dir(&session.id);
    std::fs::write(a.join("data.csv"), "1,2,3").unwrap();
    assert!(a.join("data.csv").exists());

    store.delete(&session.id).await.unwrap();
    assert!(
        !dir.path().join(&session.id).exists(),
        "session dir must be gone"
    );
}

#[tokio::test]
async fn migrate_legacy_flat_file() {
    let dir = tempfile::tempdir().unwrap();
    let store = JsonlSessionStore::new(dir.path().to_path_buf());

    // Create a legacy flat file manually
    let session = test_session(dir.path());
    let legacy_path = dir.path().join(format!("{}.jsonl", session.id));
    let meta = serde_json::json!({
        "type": "session_meta",
        "session_id": session.id,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "workspace": session.workspace,
        "state": "idle",
        "metadata": { "provider": "anthropic", "model": "claude-sonnet-4", "channel": "cli" },
        "system_prompt": "system prompt",
    });
    std::fs::write(&legacy_path, serde_json::to_string(&meta).unwrap() + "\n").unwrap();
    assert!(legacy_path.exists(), "legacy file should exist before load");

    let loaded = store.load(&session.id).await.unwrap();
    assert!(loaded.is_some(), "should load from legacy file");
    assert!(!legacy_path.exists(), "legacy file should be migrated away");
    assert!(
        dir.path().join(&session.id).join("session.jsonl").exists(),
        "new layout should exist"
    );
}

#[tokio::test]
async fn session_root_returns_correct_path() {
    let dir = tempfile::tempdir().unwrap();
    let store = JsonlSessionStore::new(dir.path().to_path_buf());
    let root = store.session_root("abc-123");
    assert_eq!(root, dir.path().join("abc-123"));
}

#[tokio::test]
async fn image_blocks_externalize_to_artifacts_dir() {
    let dir = tempfile::tempdir().unwrap();
    let store = JsonlSessionStore::new(dir.path().to_path_buf());

    // 4-byte fake "image" — content doesn't matter, only the round-trip.
    let bytes = vec![0xDE, 0xAD, 0xBE, 0xEF];
    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);

    let mut session = test_session(dir.path());
    session.history.push_raw(ConversationMessage {
        role: crate::types::Role::User,
        blocks: vec![
            crate::types::ContentBlock::Image {
                mime: "image/png".into(),
                data_base64: b64.clone(),
                detail: None,
            },
            crate::types::ContentBlock::Text {
                text: "describe".into(),
            },
        ],
        timestamp: chrono::Utc::now(),
        usage: None,
    });

    store.save(&session).await.unwrap();

    // session.jsonl must NOT contain the inline base64 — that's the whole
    // point of externalization (otherwise we're back to JSONL bloat).
    let raw = std::fs::read_to_string(dir.path().join(&session.id).join("session.jsonl")).unwrap();
    assert!(
        !raw.contains(&b64),
        "inline base64 must be stripped from session.jsonl"
    );
    assert!(
        raw.contains(crate::types::IMAGE_REF_SENTINEL_PREFIX),
        "sentinel marker must be present"
    );

    // The artifact file itself must exist on disk.
    let artifacts = store.artifacts_dir(&session.id);
    let mut any_artifact = false;
    for entry in std::fs::read_dir(&artifacts).unwrap() {
        let p = entry.unwrap().path();
        if p.file_name().unwrap().to_string_lossy().starts_with("img_") {
            any_artifact = true;
            assert_eq!(std::fs::read(&p).unwrap(), bytes);
        }
    }
    assert!(any_artifact, "an externalized image artifact must exist");

    // Round-trip: load must reconstruct the original Image block byte-for-byte.
    let loaded = store.load(&session.id).await.unwrap().unwrap();
    let msgs = loaded.history.messages();
    assert_eq!(msgs.len(), 1);
    match &msgs[0].blocks[0] {
        crate::types::ContentBlock::Image {
            mime, data_base64, ..
        } => {
            assert_eq!(mime, "image/png");
            assert_eq!(data_base64, &b64);
        }
        other => panic!("expected Image block after intern, got {other:?}"),
    }
}

#[tokio::test]
async fn gc_orphan_artifacts_removes_unreferenced_files() {
    let dir = tempfile::tempdir().unwrap();
    let store = JsonlSessionStore::new(dir.path().to_path_buf());

    let bytes = vec![0xAB, 0xCD];
    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);

    let mut session = test_session(dir.path());
    session.history.push_raw(ConversationMessage {
        role: crate::types::Role::User,
        blocks: vec![crate::types::ContentBlock::Image {
            mime: "image/png".into(),
            data_base64: b64,
            detail: None,
        }],
        timestamp: chrono::Utc::now(),
        usage: None,
    });
    store.save(&session).await.unwrap();

    let artifacts_dir = store.artifacts_dir(&session.id);
    // Drop two extra orphan files that are NOT referenced by session.jsonl —
    // mimics the state after a /clear or compact() that dropped their messages.
    std::fs::write(artifacts_dir.join("img_dead0000deadbeef.png"), b"x").unwrap();
    std::fs::write(artifacts_dir.join("img_dead0000cafebabe.png"), b"y").unwrap();
    // And a non-img file that must be left untouched.
    std::fs::write(artifacts_dir.join("user_doc.txt"), "keep").unwrap();

    let removed = store.gc_orphan_image_artifacts(&session.id).await.unwrap();
    assert_eq!(removed, 2, "two orphan img_* files must be removed");

    // The referenced artifact and the user doc must survive.
    let mut surviving: Vec<String> = std::fs::read_dir(&artifacts_dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
        .collect();
    surviving.sort();
    assert!(surviving.iter().any(|n| n == "user_doc.txt"));
    assert!(
        surviving
            .iter()
            .any(|n| n.starts_with("img_") && !n.contains("dead")),
        "the live referenced artifact must remain: {surviving:?}"
    );
}

#[tokio::test]
async fn gc_old_artifacts_removes_files_past_cutoff() {
    use std::time::{Duration, SystemTime};
    let dir = tempfile::tempdir().unwrap();
    let store = JsonlSessionStore::new(dir.path().to_path_buf());
    let session_id = "s-old";
    let artifacts_dir = store.artifacts_dir(session_id);
    std::fs::create_dir_all(&artifacts_dir).unwrap();

    // Create three img_* files. Backdate two to >1 day old; leave one
    // fresh. A non-img file must never be touched.
    let old_a = artifacts_dir.join("img_old0000aaaaaaaa.png");
    let old_b = artifacts_dir.join("img_old0000bbbbbbbb.jpg");
    let fresh = artifacts_dir.join("img_fresh0000cccccccc.png");
    let keep_non_img = artifacts_dir.join("user_doc.txt");
    std::fs::write(&old_a, b"x").unwrap();
    std::fs::write(&old_b, b"y").unwrap();
    std::fs::write(&fresh, b"z").unwrap();
    std::fs::write(&keep_non_img, b"keep").unwrap();

    // Set mtime on old files to 2 days ago.
    let two_days_ago = SystemTime::now() - Duration::from_secs(2 * 24 * 3600);
    let ft = filetime::FileTime::from_system_time(two_days_ago);
    filetime::set_file_mtime(&old_a, ft).unwrap();
    filetime::set_file_mtime(&old_b, ft).unwrap();

    // cutoff = 1 day → expect the two backdated files to go.
    let removed = store
        .gc_old_image_artifacts(session_id, 24 * 3600)
        .await
        .unwrap();
    assert_eq!(removed, 2, "two old img_* files must be removed");

    assert!(!old_a.exists());
    assert!(!old_b.exists());
    assert!(fresh.exists(), "fresh img artifact must survive");
    assert!(keep_non_img.exists(), "non-img files must never be touched");
}

#[tokio::test]
async fn gc_old_artifacts_missing_dir_returns_zero() {
    let dir = tempfile::tempdir().unwrap();
    let store = JsonlSessionStore::new(dir.path().to_path_buf());
    let removed = store
        .gc_old_image_artifacts("nonexistent-session", 60)
        .await
        .unwrap();
    assert_eq!(removed, 0);
}

#[tokio::test]
async fn duplicate_image_dedupes_on_disk() {
    // Same bytes attached twice must produce only one artifact file —
    // hash-based filenames give us free deduplication.
    let dir = tempfile::tempdir().unwrap();
    let store = JsonlSessionStore::new(dir.path().to_path_buf());

    let bytes = vec![1u8, 2, 3, 4, 5];
    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);

    let mut session = test_session(dir.path());
    for _ in 0..3 {
        session.history.push_raw(ConversationMessage {
            role: crate::types::Role::User,
            blocks: vec![crate::types::ContentBlock::Image {
                mime: "image/jpeg".into(),
                data_base64: b64.clone(),
                detail: None,
            }],
            timestamp: chrono::Utc::now(),
            usage: None,
        });
    }
    store.save(&session).await.unwrap();

    let count = std::fs::read_dir(store.artifacts_dir(&session.id))
        .unwrap()
        .filter(|e| {
            e.as_ref()
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with("img_")
        })
        .count();
    assert_eq!(count, 1, "identical images must dedupe to one artifact");
}

#[tokio::test]
async fn compaction_summary_survives_save_load() {
    let dir = tempfile::tempdir().unwrap();
    let store = JsonlSessionStore::new(dir.path().to_path_buf());

    let mut session = test_session(dir.path());
    session.history.push_user("q1");
    session.history.push_assistant(
        vec![crate::types::ContentBlock::Text { text: "a1".into() }],
        None,
    );
    session.history.push_raw(ConversationMessage::system(
        "<summary>compacted context</summary>",
    ));
    session.history.push_user("q2");

    store.save(&session).await.unwrap();
    let loaded = store.load(&session.id).await.unwrap().unwrap();

    assert_eq!(loaded.history.system_prompt(), "system prompt");
    assert_eq!(loaded.history.message_count(), 4);
    let msgs = loaded.history.messages();
    assert_eq!(msgs[0].text_content(), "q1");
    assert_eq!(msgs[1].text_content(), "a1");
    assert_eq!(msgs[2].role, crate::types::Role::System);
    assert!(msgs[2].text_content().contains("compacted context"));
    assert_eq!(msgs[3].text_content(), "q2");
}

#[tokio::test]
async fn file_tracker_persists_across_save_load() {
    let dir = tempfile::tempdir().unwrap();
    let store = JsonlSessionStore::new(dir.path().to_path_buf());

    // Create session with file tracker data
    let mut session = crate::session::Session::new(
        dir.path().to_path_buf(),
        "test".into(),
        crate::session::SessionMetadata {
            name: None,
            provider: "test".into(),
            model: "test".into(),
            channel: "test".into(),
            channel_id: None,
        },
    );
    session.files.record_tool(
        "read_file",
        &serde_json::json!({"file_path": "src/main.rs"}),
    );
    session
        .files
        .record_tool("edit_file", &serde_json::json!({"file_path": "src/lib.rs"}));
    session
        .files
        .record_tool("write_file", &serde_json::json!({"file_path": "new.rs"}));

    let sid = session.id.clone();
    store.save(&session).await.unwrap();

    // Load and verify
    let loaded = store.load(&sid).await.unwrap().unwrap();
    assert!(
        loaded.files.read.contains("src/main.rs"),
        "read files should persist"
    );
    assert!(
        loaded.files.edited.contains("src/lib.rs"),
        "edited files should persist"
    );
    assert!(
        loaded.files.written.contains("new.rs"),
        "written files should persist"
    );

    // read_only should exclude edited
    let ro = loaded.files.read_only();
    assert!(ro.contains(&"src/main.rs"));
    assert!(!ro.contains(&"src/lib.rs"));
}

#[tokio::test]
async fn incremental_save_appends_not_rewrites() {
    let dir = tempfile::tempdir().unwrap();
    let store = JsonlSessionStore::new(dir.path().to_path_buf());

    let mut session = crate::session::Session::new(
        dir.path().to_path_buf(),
        "system".into(),
        crate::session::SessionMetadata {
            name: None,
            provider: "test".into(),
            model: "test".into(),
            channel: "test".into(),
            channel_id: None,
        },
    );

    // First save: full rewrite (persisted_msg_count = 0)
    session.history.push_user("hello");
    store.save(&session).await.unwrap();
    session.persisted_msg_count = session.history.message_count();

    let path = store.session_path(&session.id);
    let size_after_first = tokio::fs::metadata(&path).await.unwrap().len();

    // Second save: add one message → incremental append
    session.history.push_user("world");
    store.save(&session).await.unwrap();
    session.persisted_msg_count = session.history.message_count();

    let size_after_second = tokio::fs::metadata(&path).await.unwrap().len();
    // File grew (append), not rewrote from scratch
    assert!(
        size_after_second > size_after_first,
        "file should grow: {size_after_first} → {size_after_second}"
    );

    // Load and verify both messages present
    let loaded = store.load(&session.id).await.unwrap().unwrap();
    let texts: Vec<String> = loaded
        .history
        .messages()
        .iter()
        .filter_map(|m| {
            if m.blocks.is_empty() {
                None
            } else {
                Some(m.text_content())
            }
        })
        .collect();
    assert!(
        texts.iter().any(|t| t.contains("hello")),
        "should have 'hello'"
    );
    assert!(
        texts.iter().any(|t| t.contains("world")),
        "should have 'world'"
    );
}
