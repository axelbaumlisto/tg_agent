use super::*;
use crate::research::spec::{Finding, ResearchSpec, RunRecord, dedup_hash};
use chrono::Utc;
use std::future::Future;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, OnceLock};
use tempfile::tempdir;
use tokio::sync::Barrier;

fn make_spec(id: &str, topic: &str) -> ResearchSpec {
    // REGISTRY-WAIVE: exhaustive test struct ctor — pre-2026-05-13 baseline; new tests must use ..Default::default()
    ResearchSpec {
        id: id.to_string(),
        topic: topic.to_string(),
        sources: vec!["https://example.com".into()],
        interval_seconds: None,
        run_at: None,
        cron: None,
        task_timeout_seconds: None,
        session_id: None,
        chat_id: None,
        thread_id: None,
        provider: None,
        model: None,
        max_iterations: None,
        max_wall_seconds: None,
        created_at: Utc::now(),
        paused: false,
        pause_reason: None,
    }
}

fn make_finding(spec_id: &str, url: &str) -> Finding {
    use crate::research::spec::host_path_hash;
    // REGISTRY-WAIVE: exhaustive test struct ctor — pre-2026-05-13 baseline; new tests must use ..Default::default()
    Finding {
        id: uuid::Uuid::new_v4().simple().to_string(),
        research_id: spec_id.to_string(),
        run_id: "r1".to_string(),
        url: url.to_string(),
        title: Some("title".into()),
        excerpt: Some("excerpt".into()),
        price: None,
        listing_date: None,
        source_content: None,
        dedup_hash: dedup_hash(url),
        host_path_hash: host_path_hash(url),
        // Empty content_hash so URL-only dedup tests aren't accidentally
        // tripped by the new content-based dedup. Content-hash dedup has
        // its own dedicated test.
        content_hash: String::new(),
        seen_at: Utc::now(),
    }
}

fn make_run(spec_id: &str, run_id: &str) -> RunRecord {
    // REGISTRY-WAIVE: exhaustive test struct ctor — RunRecord has no Default impl.
    RunRecord {
        run_id: run_id.to_string(),
        spec_id: spec_id.to_string(),
        started_at: Utc::now(),
        finished_at: Utc::now(),
        new_findings: 0,
        total_findings_after: 0,
        stop_reason: "ok".to_string(),
        provider: "p".to_string(),
        model: "m".to_string(),
        verification_rounds: None,
        dead_removed: None,
        replacements_found: None,
        remaining_issues: None,
        elapsed_secs: None,
    }
}

fn assert_findings_file_all_valid(path: &std::path::Path) -> Vec<Finding> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|e| panic!("failed to read findings file {}: {e}", path.display()));
    let content = String::from_utf8(bytes)
        .unwrap_or_else(|e| panic!("findings file {} is not utf-8: {e}", path.display()));
    let mut findings = Vec::new();
    for (idx, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        findings.push(serde_json::from_str::<Finding>(line).unwrap_or_else(|e| {
            panic!(
                "invalid findings.jsonl row {} in {}: {e}",
                idx + 1,
                path.display()
            )
        }));
    }
    findings
}

fn corrupt_backups(dir: &Path) -> Vec<PathBuf> {
    let mut backups = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("failed to read research dir {}: {e}", dir.display()))
        .map(|entry| entry.expect("read_dir entry ok").path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("findings.jsonl.corrupt-"))
        })
        .collect::<Vec<_>>();
    backups.sort();
    backups
}

fn b145_metric_lock() -> &'static tokio::sync::Mutex<()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
}

fn malformed_state_count() -> u64 {
    crate::types::RESEARCH_STORE_MALFORMED_STATE_DETECTED_COUNT.load(Ordering::Relaxed)
}

#[derive(Clone)]
struct SharedLog(Arc<Mutex<Vec<u8>>>);

impl Write for SharedLog {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

async fn capture_warn_logs_async<F, Fut, R>(f: F) -> (R, String)
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = R>,
{
    let log_bytes = Arc::new(Mutex::new(Vec::new()));
    let make_writer = {
        let log_bytes = log_bytes.clone();
        move || SharedLog(log_bytes.clone())
    };
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .with_ansi(false)
        .with_writer(make_writer)
        .finish();
    let _subscriber_guard = tracing::subscriber::set_default(subscriber);
    let result = f().await;
    let logs = String::from_utf8(log_bytes.lock().unwrap().clone()).unwrap();
    (result, logs)
}

fn mixed_corrupt_findings_bytes(valid: &[Finding], bad_suffix: &str) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(serde_json::to_string(&valid[0]).unwrap().as_bytes());
    bytes.extend_from_slice(b"\n{\"torn\":");
    bytes.extend_from_slice(bad_suffix.as_bytes());
    bytes.extend_from_slice(b"\n");
    bytes.extend_from_slice(serde_json::to_string(&valid[1]).unwrap().as_bytes());
    bytes.extend_from_slice(b"\nnot-json-at-all\n");
    bytes.extend_from_slice(serde_json::to_string(&valid[2]).unwrap().as_bytes());
    bytes.extend_from_slice(b"\n");
    bytes
}

#[tokio::test]
async fn mixed_corrupt_findings_self_heals_once() {
    let tmp = tempdir().unwrap();
    let store = FsResearchStore::new(tmp.path().to_path_buf());
    let spec = make_spec("id-heal", "t");
    store.create_spec(&spec).await.unwrap();
    let research_dir = tmp.path().join("id-heal");
    let findings_path = research_dir.join("findings.jsonl");

    let valid = vec![
        make_finding("id-heal", "https://x.com/a"),
        make_finding("id-heal", "https://x.com/b"),
        make_finding("id-heal", "https://x.com/c"),
    ];
    let original = mixed_corrupt_findings_bytes(&valid, "42");
    std::fs::write(&findings_path, &original).unwrap();

    let listed = store.list_findings("id-heal", None).await.unwrap();
    assert_eq!(
        listed, valid,
        "read must return only valid findings in order"
    );
    assert_eq!(
        assert_findings_file_all_valid(&findings_path),
        valid,
        "healed findings file must be 100% valid JSONL"
    );

    let backups = corrupt_backups(&research_dir);
    assert_eq!(backups.len(), 1, "first corrupt read creates one backup");
    assert_eq!(
        std::fs::read(&backups[0]).unwrap(),
        original,
        "backup must be byte-for-byte original including torn rows"
    );

    let second = store.list_findings("id-heal", None).await.unwrap();
    assert_eq!(second, valid);
    assert_eq!(
        corrupt_backups(&research_dir),
        backups,
        "second clean read must not create another backup or re-heal"
    );
}

#[tokio::test]
async fn corrupt_findings_backups_use_unique_names_across_repeated_heals() {
    let tmp = tempdir().unwrap();
    let store = FsResearchStore::new(tmp.path().to_path_buf());
    let spec = make_spec("id-heal-unique", "t");
    store.create_spec(&spec).await.unwrap();
    let research_dir = tmp.path().join("id-heal-unique");
    let findings_path = research_dir.join("findings.jsonl");
    let valid = vec![
        make_finding("id-heal-unique", "https://x.com/a"),
        make_finding("id-heal-unique", "https://x.com/b"),
        make_finding("id-heal-unique", "https://x.com/c"),
    ];

    let first_original = mixed_corrupt_findings_bytes(&valid, "first");
    std::fs::write(&findings_path, &first_original).unwrap();
    store.list_findings("id-heal-unique", None).await.unwrap();

    let second_original = mixed_corrupt_findings_bytes(&valid, "second");
    std::fs::write(&findings_path, &second_original).unwrap();
    store.list_findings("id-heal-unique", None).await.unwrap();

    let backups = corrupt_backups(&research_dir);
    assert_eq!(backups.len(), 2, "two corrupt generations need two backups");
    assert_ne!(
        backups[0].file_name().unwrap(),
        backups[1].file_name().unwrap(),
        "backup suffix must be unique, not seconds-only"
    );
    let backup_bytes = backups
        .iter()
        .map(|path| std::fs::read(path).unwrap())
        .collect::<Vec<_>>();
    assert!(backup_bytes.contains(&first_original));
    assert!(backup_bytes.contains(&second_original));
}

#[tokio::test]
async fn create_load_save_roundtrip() {
    let tmp = tempdir().unwrap();
    let store = FsResearchStore::new(tmp.path().to_path_buf());
    let mut spec = make_spec("id-1", "topic");
    store.create_spec(&spec).await.unwrap();
    let loaded = store.load_spec("id-1").await.unwrap();
    assert_eq!(loaded, spec);
    spec.paused = true;
    store.save_spec(&spec).await.unwrap();
    let loaded2 = store.load_spec("id-1").await.unwrap();
    assert!(loaded2.paused);
}

#[tokio::test]
async fn create_rejects_duplicate() {
    let tmp = tempdir().unwrap();
    let store = FsResearchStore::new(tmp.path().to_path_buf());
    let spec = make_spec("id-dup", "topic");
    store.create_spec(&spec).await.unwrap();
    let err = store.create_spec(&spec).await;
    assert!(err.is_err());
}

#[tokio::test]
async fn list_specs_sorts_newest_first() {
    let tmp = tempdir().unwrap();
    let store = FsResearchStore::new(tmp.path().to_path_buf());
    let mut a = make_spec("a", "A");
    a.created_at = Utc::now() - chrono::Duration::hours(2);
    let b = make_spec("b", "B"); // newer
    store.create_spec(&a).await.unwrap();
    store.create_spec(&b).await.unwrap();
    let list = store.list_specs().await.unwrap();
    assert_eq!(list.len(), 2);
    assert_eq!(list[0].id, "b");
    assert_eq!(list[1].id, "a");
}

#[tokio::test]
async fn findings_dedup_on_canonical_url() {
    let tmp = tempdir().unwrap();
    let store = FsResearchStore::new(tmp.path().to_path_buf());
    let spec = make_spec("id-2", "t");
    store.create_spec(&spec).await.unwrap();

    let f1 = make_finding("id-2", "https://x.com/ad/42?utm_source=a");
    let f2 = make_finding("id-2", "https://x.com/ad/42?utm_source=b");
    let f3 = make_finding("id-2", "https://x.com/ad/43");

    assert!(store.try_append_finding(&f1).await.unwrap());
    assert!(
        !store.try_append_finding(&f2).await.unwrap(),
        "should dedup"
    );
    assert!(store.try_append_finding(&f3).await.unwrap());

    assert_eq!(store.count_findings("id-2").await.unwrap(), 2);
    let list = store.list_findings("id-2", None).await.unwrap();
    assert_eq!(list.len(), 2);
}

#[tokio::test]
async fn host_path_hash_dedup_rejects_same_listing_with_extra_query_params() {
    // T7: a listing reached via different non-tracking query strings
    // (`?sort=newest` vs `?sort=oldest`) has DIFFERENT `dedup_hash`
    // values but the SAME `host_path_hash`. The store must reject the
    // second arrival on the secondary check.
    let tmp = tempdir().unwrap();
    let store = FsResearchStore::new(tmp.path().to_path_buf());
    let spec = make_spec("id-hp", "t");
    store.create_spec(&spec).await.unwrap();

    let f1 = make_finding("id-hp", "https://batdongsan.com.vn/ad/42?sort=newest");
    let f2 = make_finding(
        "id-hp",
        "https://batdongsan.com.vn/ad/42?sort=oldest&page=3",
    );
    // Sanity: they pass the URL-canon dedup (different canonical URLs).
    assert_ne!(
        f1.dedup_hash, f2.dedup_hash,
        "different non-tracking params → different canonical URLs"
    );
    assert_eq!(
        f1.host_path_hash, f2.host_path_hash,
        "but same host+path → same secondary key"
    );

    assert!(store.try_append_finding(&f1).await.unwrap());
    assert!(
        !store.try_append_finding(&f2).await.unwrap(),
        "host_path_hash dedup must reject the second arrival"
    );
    assert_eq!(store.count_findings("id-hp").await.unwrap(), 1);
}

#[tokio::test]
async fn content_hash_dedup_rejects_crosspost_to_different_domain() {
    // T7: same prose republished at a completely different URL
    // (different host AND different path) — neither URL-canon nor
    // host-path matches, but content_hash does. The store must reject.
    use crate::research::spec::content_hash;
    let tmp = tempdir().unwrap();
    let store = FsResearchStore::new(tmp.path().to_path_buf());
    let spec = make_spec("id-cp", "t");
    store.create_spec(&spec).await.unwrap();

    let body = "Cho thuê mặt bằng kinh doanh đường Ông Ích Khiêm \
                quận Hải Châu, diện tích 140m². Liên hệ chính chủ.";

    let mut f1 = make_finding("id-cp", "https://batdongsan.com.vn/ad/100");
    f1.excerpt = Some(body.to_string());
    f1.content_hash = content_hash(body);

    let mut f2 = make_finding("id-cp", "https://facebook.com/marketplace/item/999");
    f2.excerpt = Some(body.to_string());
    f2.content_hash = content_hash(body);

    // Sanity: URL-level keys differ.
    assert_ne!(f1.dedup_hash, f2.dedup_hash);
    assert_ne!(f1.host_path_hash, f2.host_path_hash);
    assert_eq!(f1.content_hash, f2.content_hash);

    assert!(store.try_append_finding(&f1).await.unwrap());
    assert!(
        !store.try_append_finding(&f2).await.unwrap(),
        "content_hash dedup must reject the crosspost"
    );
    assert_eq!(store.count_findings("id-cp").await.unwrap(), 1);
}

#[tokio::test]
async fn dedup_survives_reopen() {
    let tmp = tempdir().unwrap();
    {
        let store = FsResearchStore::new(tmp.path().to_path_buf());
        let spec = make_spec("id-3", "t");
        store.create_spec(&spec).await.unwrap();
        let f = make_finding("id-3", "https://x.com/ad/99");
        assert!(store.try_append_finding(&f).await.unwrap());
    }
    let store2 = FsResearchStore::new(tmp.path().to_path_buf());
    let dup = make_finding("id-3", "https://x.com/ad/99?utm_source=later");
    assert!(!store2.try_append_finding(&dup).await.unwrap());
}

#[tokio::test]
async fn upsert_existing_finding_replaces_row_and_keeps_dedup() {
    let tmp = tempdir().unwrap();
    let store = FsResearchStore::new(tmp.path().to_path_buf());
    let spec = make_spec("id-upsert-replace", "t");
    store.create_spec(&spec).await.unwrap();

    let original = make_finding("id-upsert-replace", "https://x.com/ad/replace");
    assert!(!store.upsert_finding(&original).await.unwrap());

    let mut updated = original.clone();
    updated.title = Some("updated title".into());
    updated.excerpt = Some("updated excerpt".into());
    assert!(store.upsert_finding(&updated).await.unwrap());

    let findings = store
        .list_findings("id-upsert-replace", None)
        .await
        .unwrap();
    assert_eq!(
        findings.len(),
        1,
        "upsert must replace, not append duplicate rows"
    );
    assert_eq!(findings[0].dedup_hash, original.dedup_hash);
    assert_eq!(findings[0].title.as_deref(), Some("updated title"));
    assert_eq!(findings[0].excerpt.as_deref(), Some("updated excerpt"));

    let same = make_finding("id-upsert-replace", "https://x.com/ad/replace");
    assert!(!store.try_append_finding(&same).await.unwrap());
    assert_eq!(store.count_findings("id-upsert-replace").await.unwrap(), 1);
}

#[tokio::test]
async fn upsert_existing_finding_missing_file_is_inconsistent() {
    let tmp = tempdir().unwrap();
    let store = FsResearchStore::new(tmp.path().to_path_buf());
    let spec = make_spec("id-upsert-missing", "t");
    store.create_spec(&spec).await.unwrap();

    let first = make_finding("id-upsert-missing", "https://x.com/ad/a");
    assert!(!store.upsert_finding(&first).await.unwrap());
    let path = tmp.path().join("id-upsert-missing").join("findings.jsonl");
    std::fs::remove_file(&path).unwrap();

    let mut updated = first.clone();
    updated.title = Some("updated after missing file".into());
    let err = store
        .upsert_finding(&updated)
        .await
        .expect_err("missing findings.jsonl after dedup hit must surface an inconsistency");
    assert!(
        err.to_string()
            .contains("findings.jsonl missing for existing finding rewrite"),
        "unexpected error: {err:?}"
    );
    assert!(
        !path.exists(),
        "missing rewrite input must not be silently recreated as a one-row file"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn upsert_existing_finding_read_error_keeps_file_intact() {
    use std::os::unix::fs::PermissionsExt;

    let tmp = tempdir().unwrap();
    let store = FsResearchStore::new(tmp.path().to_path_buf());
    let spec = make_spec("id-upsert-read-error", "t");
    store.create_spec(&spec).await.unwrap();

    let first = make_finding("id-upsert-read-error", "https://x.com/ad/a");
    let second = make_finding("id-upsert-read-error", "https://x.com/ad/b");
    assert!(!store.upsert_finding(&first).await.unwrap());
    assert!(!store.upsert_finding(&second).await.unwrap());

    let path = tmp
        .path()
        .join("id-upsert-read-error")
        .join("findings.jsonl");
    let original_bytes = std::fs::read(&path).unwrap();
    let original_mode = std::fs::metadata(&path).unwrap().permissions().mode();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o000)).unwrap();

    let mut updated = first.clone();
    updated.title = Some("updated after read failure".into());
    let err = store
        .upsert_finding(&updated)
        .await
        .expect_err("unreadable findings.jsonl must surface an error");

    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(original_mode)).unwrap();
    assert!(
        matches!(&err, AgentError::Io(e) if e.kind() == std::io::ErrorKind::PermissionDenied),
        "expected PermissionDenied I/O error, got {err:?}"
    );
    assert_eq!(
        std::fs::read(&path).unwrap(),
        original_bytes,
        "read failure on rewrite path must leave findings.jsonl byte-for-byte intact"
    );
    assert_eq!(
        assert_findings_file_all_valid(&path).len(),
        2,
        "the pre-existing findings must not be collapsed to the replacement row"
    );
}

#[tokio::test]
async fn concurrent_upsert_keeps_file_valid() {
    let tmp = tempdir().unwrap();
    let store = Arc::new(FsResearchStore::new(tmp.path().to_path_buf()));
    let spec = make_spec("id-concurrent", "t");
    store.create_spec(&spec).await.unwrap();

    for idx in 0..8 {
        let f = make_finding("id-concurrent", &format!("https://x.com/existing/{idx}"));
        assert!(!store.upsert_finding(&f).await.unwrap());
    }
    for idx in 0..8 {
        let f = make_finding("id-concurrent", &format!("https://x.com/remove/{idx}"));
        assert!(!store.upsert_finding(&f).await.unwrap());
    }

    reset_findings_write_critical_section_probe("id-concurrent");

    let gate = Arc::new(Barrier::new(33));
    let mut handles = Vec::new();

    for idx in 0..32 {
        let store = store.clone();
        let gate = gate.clone();
        handles.push(tokio::spawn(async move {
            gate.wait().await;
            match idx % 4 {
                0 => {
                    let f = make_finding("id-concurrent", &format!("https://x.com/new/{idx}"));
                    store.upsert_finding(&f).await.map(|_| ())
                }
                1 => {
                    let mut f = make_finding(
                        "id-concurrent",
                        &format!("https://x.com/existing/{}", idx / 4),
                    );
                    f.title = Some(format!("updated {idx}"));
                    store.upsert_finding(&f).await.map(|_| ())
                }
                2 => {
                    let hashes = [dedup_hash(&format!("https://x.com/remove/{}", idx / 4))]
                        .into_iter()
                        .collect::<HashSet<_>>();
                    store
                        .remove_findings_by_hash("id-concurrent", &hashes)
                        .await
                        .map(|_| ())
                }
                _ => {
                    let f = make_finding("id-concurrent", &format!("https://x.com/try/{idx}"));
                    store.try_append_finding(&f).await.map(|_| ())
                }
            }
        }));
    }

    gate.wait().await;
    for handle in handles {
        handle.await.unwrap().unwrap();
    }
    assert_eq!(
        findings_write_critical_section_max_observed(),
        1,
        "per-id findings file mutex must keep max concurrent write critical sections at 1"
    );
    disable_findings_write_critical_section_probe();

    let path = tmp.path().join("id-concurrent").join("findings.jsonl");
    let findings = assert_findings_file_all_valid(&path);
    let hashes = findings
        .iter()
        .map(|f| f.dedup_hash.clone())
        .collect::<HashSet<_>>();
    assert_eq!(
        findings.len(),
        hashes.len(),
        "raw file must not contain duplicate dedup hashes"
    );

    let expected_hashes = (0..8)
        .map(|idx| dedup_hash(&format!("https://x.com/existing/{idx}")))
        .chain(
            (0..32)
                .step_by(4)
                .map(|idx| dedup_hash(&format!("https://x.com/new/{idx}"))),
        )
        .chain(
            (3..32)
                .step_by(4)
                .map(|idx| dedup_hash(&format!("https://x.com/try/{idx}"))),
        )
        .collect::<HashSet<_>>();
    assert_eq!(hashes, expected_hashes);
    assert_eq!(
        store.count_findings("id-concurrent").await.unwrap(),
        findings.len() as u32
    );
}

#[tokio::test]
async fn findings_file_lock_wait_counter_bumps_on_contention() {
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    let tmp = tempdir().unwrap();
    let store = Arc::new(FsResearchStore::new(tmp.path().to_path_buf()));
    let spec = make_spec("id-lock-wait", "t");
    store.create_spec(&spec).await.unwrap();

    let file_lock = store.write_guard("id-lock-wait").await;
    let held = file_lock.lock().await;
    let before = crate::types::RESEARCH_STORE_FILE_LOCK_WAIT_COUNT.load(Ordering::Relaxed);

    let contender_store = store.clone();
    let mut contender = tokio::spawn(async move {
        let finding = make_finding("id-lock-wait", "https://x.com/contention");
        contender_store.try_append_finding(&finding).await
    });

    // Deterministic behavioural proof: while this test holds the per-id mutex,
    // the contender must not complete. It is therefore parked on the same mutex
    // after `lock_file_guard` synchronously missed `try_lock` and bumped the
    // counter. The counter assertion below is only the metric-wiring check.
    assert!(
        tokio::time::timeout(Duration::from_millis(100), &mut contender)
            .await
            .is_err(),
        "contending findings mutator completed while the file lock was still held"
    );

    drop(held);
    assert!(contender.await.unwrap().unwrap());

    let after = crate::types::RESEARCH_STORE_FILE_LOCK_WAIT_COUNT.load(Ordering::Relaxed);
    let expected_after = before + 1;
    assert!(
        after >= expected_after,
        "contending findings mutator must bump file-lock wait counter before acquiring lock; before={before}, after={after}"
    );
}

#[test]
fn findings_write_calls_are_guarded_by_file_mutex() {
    let source = include_str!("store_fs.rs");
    let finding_impl = source
        .split("impl FindingStore for FsResearchStore")
        .nth(1)
        .expect("FindingStore impl exists")
        .split("impl RunStore for FsResearchStore")
        .next()
        .expect("RunStore impl follows FindingStore impl");

    // Heuristic sentinel for B59: a findings write site is any method block
    // that mentions `findings.jsonl` and calls append_jsonl/atomic_write. Count
    // all such call sites in store_fs.rs and require every one to live in one
    // of the guarded findings mutators, with write_guard acquired before the
    // first write call. This fails if a future method writes findings.jsonl
    // outside the guarded mutators.
    let mut guarded_write_calls = 0usize;
    for mutator in [
        "async fn try_append_finding",
        "async fn upsert_finding",
        "async fn remove_findings_by_hash",
    ] {
        let block = function_block(finding_impl, mutator)
            .unwrap_or_else(|| panic!("missing mutator {mutator}"));
        let guard_pos = block
            .find("let _file_guard = Self::lock_file_guard(&file_lock).await")
            .unwrap_or_else(|| panic!("{mutator} must hold the per-id file mutex"));
        let first_write_pos = first_findings_write_call(block)
            .unwrap_or_else(|| panic!("{mutator} must contain a findings write call"));
        assert!(
            block[..guard_pos].contains("let file_lock = self.write_guard"),
            "{mutator} must clone the per-id file mutex before locking it"
        );
        assert!(
            guard_pos < first_write_pos,
            "{mutator} must acquire the per-id file mutex before writing findings.jsonl"
        );
        guarded_write_calls += count_findings_write_calls(block);
    }

    let all_findings_write_calls = count_findings_file_write_calls(source);
    assert_eq!(
        all_findings_write_calls, guarded_write_calls,
        "all findings.jsonl append_jsonl/atomic_write call sites must be inside guarded findings mutators"
    );
}

fn function_block<'a>(source: &'a str, signature: &str) -> Option<&'a str> {
    function_block_at(source, source.find(signature)?)
}

fn function_block_at(source: &str, start: usize) -> Option<&str> {
    let after_signature = &source[start..];
    let open_rel = after_signature.find('{')?;
    let body_start = start + open_rel;
    let mut depth = 0usize;
    for (rel, ch) in source[body_start..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some(&source[start..body_start + rel + 1]);
                }
            }
            _ => {}
        }
    }
    None
}

fn first_findings_write_call(source: &str) -> Option<usize> {
    [".append_jsonl(", ".atomic_write("]
        .into_iter()
        .filter_map(|needle| source.find(needle))
        .min()
}

fn count_findings_write_calls(source: &str) -> usize {
    [".append_jsonl(", ".atomic_write("]
        .into_iter()
        .map(|needle| source.matches(needle).count())
        .sum()
}

fn count_findings_file_write_calls(source: &str) -> usize {
    method_blocks(source)
        .into_iter()
        .filter(|block| block.contains("\"findings.jsonl\""))
        .map(count_findings_write_calls)
        .sum()
}

fn method_blocks(source: &str) -> Vec<&str> {
    let mut blocks = Vec::new();
    let mut offset = 0usize;
    while let Some(rel) = source[offset..].find("\n    async fn ") {
        let start = offset + rel + 1;
        if let Some(block) = function_block_at(source, start) {
            blocks.push(block);
        }
        offset = start + "    async fn ".len();
    }
    blocks
}

#[tokio::test]
async fn runs_are_append_only_and_limited() {
    let tmp = tempdir().unwrap();
    let store = FsResearchStore::new(tmp.path().to_path_buf());
    let spec = make_spec("id-4", "t");
    store.create_spec(&spec).await.unwrap();
    for i in 0..5 {
        let r = RunRecord {
            run_id: format!("r{i}"),
            spec_id: "id-4".to_string(),
            started_at: Utc::now(),
            finished_at: Utc::now(),
            new_findings: 0,
            total_findings_after: 0,
            stop_reason: "ok".to_string(),
            provider: "p".to_string(),
            model: "m".to_string(),
            verification_rounds: None,
            dead_removed: None,
            replacements_found: None,
            remaining_issues: None,
            elapsed_secs: None,
        };
        store.append_run(&r).await.unwrap();
    }
    let last3 = store.list_runs("id-4", Some(3)).await.unwrap();
    assert_eq!(last3.len(), 3);
    assert_eq!(last3[0].run_id, "r2");
    assert_eq!(last3[2].run_id, "r4");
}

#[tokio::test]
async fn b145_malformed_jsonl_row_returns_good_rows_and_counts() {
    let _guard = b145_metric_lock().lock().await;
    let tmp = tempdir().unwrap();
    let store = FsResearchStore::new(tmp.path().to_path_buf());
    let spec = make_spec("b145-mixed-jsonl", "t");
    store.create_spec(&spec).await.unwrap();
    let runs_path = tmp.path().join("b145-mixed-jsonl").join("runs.jsonl");
    let run_a = make_run("b145-mixed-jsonl", "run-a");
    let run_b = make_run("b145-mixed-jsonl", "run-b");
    let content = format!(
        "{}\n{{\"run_id\":\n{}\n",
        serde_json::to_string(&run_a).unwrap(),
        serde_json::to_string(&run_b).unwrap()
    );
    std::fs::write(&runs_path, content).unwrap();

    let before = malformed_state_count();
    let runs = store.list_runs("b145-mixed-jsonl", None).await.unwrap();
    let after = malformed_state_count();

    assert_eq!(
        runs.iter().map(|r| r.run_id.as_str()).collect::<Vec<_>>(),
        vec!["run-a", "run-b"],
        "malformed rows must not abort the whole listing"
    );
    assert_eq!(
        after.saturating_sub(before),
        1,
        "one malformed row must be counted"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn b145_all_garbage_jsonl_returns_empty_but_counts_and_warns() {
    let _guard = b145_metric_lock().lock().await;
    let tmp = tempdir().unwrap();
    let store = FsResearchStore::new(tmp.path().to_path_buf());
    let spec = make_spec("b145-garbage-jsonl", "t");
    store.create_spec(&spec).await.unwrap();
    let runs_path = tmp.path().join("b145-garbage-jsonl").join("runs.jsonl");
    std::fs::write(&runs_path, "not-json\n{\"run_id\":\n").unwrap();

    let before = malformed_state_count();
    let (runs, logs) =
        capture_warn_logs_async(|| store.list_runs("b145-garbage-jsonl", None)).await;
    let after = malformed_state_count();

    assert!(
        runs.unwrap().is_empty(),
        "all-garbage JSONL has no surviving rows"
    );
    assert_eq!(
        after.saturating_sub(before),
        2,
        "all-garbage JSONL must be distinguishable from a genuinely absent file"
    );
    assert!(
        logs.contains("research store: skipping malformed jsonl row"),
        "malformed JSONL must be audible at warn level; logs were:\n{logs}"
    );
    assert!(
        logs.contains(&runs_path.display().to_string()),
        "warning must include the damaged file path; logs were:\n{logs}"
    );
    assert!(
        logs.contains("row=1") && logs.contains("row=2"),
        "warning must include 1-based row numbers; logs were:\n{logs}"
    );
}

#[tokio::test]
async fn b145_valid_and_absent_jsonl_do_not_count() {
    let _guard = b145_metric_lock().lock().await;
    let tmp = tempdir().unwrap();
    let store = FsResearchStore::new(tmp.path().to_path_buf());
    let spec = make_spec("b145-valid-jsonl", "t");
    store.create_spec(&spec).await.unwrap();
    let run_a = make_run("b145-valid-jsonl", "run-a");
    let run_b = make_run("b145-valid-jsonl", "run-b");
    store.append_run(&run_a).await.unwrap();
    store.append_run(&run_b).await.unwrap();

    let before = malformed_state_count();
    let valid = store.list_runs("b145-valid-jsonl", None).await.unwrap();
    let absent = store.list_runs("b145-absent-jsonl", None).await.unwrap();
    let after = malformed_state_count();

    assert_eq!(
        valid.iter().map(|r| r.run_id.as_str()).collect::<Vec<_>>(),
        vec!["run-a", "run-b"],
        "valid JSONL should still return every row"
    );
    assert!(
        absent.is_empty(),
        "absent JSONL still returns an empty listing"
    );
    assert_eq!(
        after, before,
        "valid and genuinely absent JSONL must not increment the corruption counter"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn b145_malformed_inflight_returns_none_counts_and_preserves_file() {
    let _guard = b145_metric_lock().lock().await;
    let tmp = tempdir().unwrap();
    let store = FsResearchStore::new(tmp.path().to_path_buf());
    let spec = make_spec("b145-bad-inflight", "t");
    store.create_spec(&spec).await.unwrap();
    let inflight_path = tmp.path().join("b145-bad-inflight").join("inflight.json");
    let damaged = b"{\"state\":";
    std::fs::write(&inflight_path, damaged).unwrap();

    let before = malformed_state_count();
    let (loaded, logs) = capture_warn_logs_async(|| store.load_inflight("b145-bad-inflight")).await;
    let after = malformed_state_count();

    assert!(
        loaded.unwrap().is_none(),
        "malformed inflight still degrades to None"
    );
    assert_eq!(
        after.saturating_sub(before),
        1,
        "malformed inflight.json must be counted"
    );
    assert_eq!(
        std::fs::read(&inflight_path).unwrap(),
        damaged,
        "load_inflight must not overwrite or remove the damaged file"
    );
    assert!(
        logs.contains("research store: ignoring malformed inflight.json"),
        "malformed inflight must be audible at warn level; logs were:\n{logs}"
    );
    assert!(
        logs.contains(&inflight_path.display().to_string()),
        "warning must include the damaged inflight path; logs were:\n{logs}"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn b145_list_specs_skips_damaged_spec_counts_and_keeps_valid_specs() {
    let _guard = b145_metric_lock().lock().await;
    let tmp = tempdir().unwrap();
    let store = FsResearchStore::new(tmp.path().to_path_buf());
    let valid = make_spec("b145-valid-spec", "valid");
    store.create_spec(&valid).await.unwrap();
    let damaged_dir = tmp.path().join("b145-damaged-spec");
    std::fs::create_dir_all(&damaged_dir).unwrap();
    let damaged_path = damaged_dir.join("spec.json");
    std::fs::write(&damaged_path, b"{\"id\":").unwrap();

    let before = malformed_state_count();
    let (listed, logs) = capture_warn_logs_async(|| store.list_specs()).await;
    let after = malformed_state_count();
    let listed = listed.unwrap();

    assert_eq!(
        listed.iter().map(|s| s.id.as_str()).collect::<Vec<_>>(),
        vec!["b145-valid-spec"],
        "one damaged spec.json must not hide valid specs or kill the listing"
    );
    assert_eq!(
        listed.len(),
        1,
        "returned spec count must reflect only the valid on-disk spec"
    );
    assert_eq!(
        after.saturating_sub(before),
        1,
        "damaged spec.json must be counted exactly once"
    );
    assert!(
        logs.contains("research store: skipping malformed spec.json during listing"),
        "damaged spec.json must be audible at warn level; logs were:\n{logs}"
    );
    assert!(
        logs.contains(&damaged_path.display().to_string()),
        "warning must include the damaged spec path; logs were:\n{logs}"
    );
    assert!(
        logs.contains("count="),
        "warning must include the corruption counter value; logs were:\n{logs}"
    );
}

#[tokio::test]
async fn b145_absent_spec_json_does_not_count() {
    let _guard = b145_metric_lock().lock().await;
    let tmp = tempdir().unwrap();
    let store = FsResearchStore::new(tmp.path().to_path_buf());
    std::fs::create_dir_all(tmp.path().join("b145-no-spec-json")).unwrap();

    let before = malformed_state_count();
    let listed = store.list_specs().await.unwrap();
    let after = malformed_state_count();

    assert!(
        listed.is_empty(),
        "a research directory without spec.json is an absent entry, not damage"
    );
    assert_eq!(
        after, before,
        "absent spec.json must stay silent and uncounted"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn b145_list_all_inflight_skips_invalid_utf8_counts_and_keeps_valid_entries() {
    use crate::research::inflight::Inflight;

    let _guard = b145_metric_lock().lock().await;
    let tmp = tempdir().unwrap();
    let store = FsResearchStore::new(tmp.path().to_path_buf());
    store
        .create_spec(&make_spec("b145-valid-inflight-list", "valid"))
        .await
        .unwrap();
    let valid = Inflight::scheduled("b145-valid-inflight-list", 1);
    store
        .save_inflight("b145-valid-inflight-list", &valid)
        .await
        .unwrap();
    store
        .create_spec(&make_spec("b145-invalid-utf8-inflight-list", "bad"))
        .await
        .unwrap();
    let bad_path = tmp
        .path()
        .join("b145-invalid-utf8-inflight-list")
        .join("inflight.json");
    std::fs::write(&bad_path, [0xff]).unwrap();

    let before = malformed_state_count();
    let (listed, logs) = capture_warn_logs_async(|| store.list_all_inflight()).await;
    let after = malformed_state_count();
    let listed = listed.unwrap();

    assert_eq!(
        listed
            .iter()
            .map(|i| i.spec_id.as_str())
            .collect::<Vec<_>>(),
        vec!["b145-valid-inflight-list"],
        "invalid UTF-8 inflight.json must be skipped without killing the listing"
    );
    assert_eq!(listed.len(), 1, "only the valid inflight entry is returned");
    assert_eq!(
        after.saturating_sub(before),
        1,
        "invalid UTF-8 inflight.json must be counted exactly once"
    );
    assert!(
        logs.contains("research store: skipping unreadable inflight.json"),
        "unreadable inflight must be audible at warn level; logs were:\n{logs}"
    );
    assert!(
        logs.contains(&bad_path.display().to_string()),
        "warning must include the damaged inflight path; logs were:\n{logs}"
    );
    assert!(
        logs.contains("count="),
        "warning must include the corruption counter value; logs were:\n{logs}"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn b145_purge_terminal_inflight_skips_invalid_utf8_and_counts() {
    let _guard = b145_metric_lock().lock().await;
    let tmp = tempdir().unwrap();
    let store = FsResearchStore::new(tmp.path().to_path_buf());
    store
        .create_spec(&make_spec("b145-invalid-utf8-inflight-purge", "bad"))
        .await
        .unwrap();
    let bad_path = tmp
        .path()
        .join("b145-invalid-utf8-inflight-purge")
        .join("inflight.json");
    std::fs::write(&bad_path, [0xff]).unwrap();

    let before = malformed_state_count();
    let (removed, logs) = capture_warn_logs_async(|| {
        store.purge_terminal_inflight(Utc::now(), chrono::Duration::zero())
    })
    .await;
    let after = malformed_state_count();

    assert_eq!(removed.unwrap(), 0, "damaged inflight files are not purged");
    assert_eq!(
        after.saturating_sub(before),
        1,
        "invalid UTF-8 inflight.json during purge must be counted exactly once"
    );
    assert!(
        logs.contains("research store: skipping unreadable inflight.json"),
        "unreadable inflight during purge must be audible; logs were:\n{logs}"
    );
    assert!(
        logs.contains(&bad_path.display().to_string()),
        "warning must include the damaged inflight path; logs were:\n{logs}"
    );
}

#[tokio::test]
async fn b145_absent_inflight_json_does_not_count_in_list_or_purge() {
    let _guard = b145_metric_lock().lock().await;
    let tmp = tempdir().unwrap();
    let store = FsResearchStore::new(tmp.path().to_path_buf());
    store
        .create_spec(&make_spec("b145-no-inflight-json", "absent"))
        .await
        .unwrap();

    let before = malformed_state_count();
    let listed = store.list_all_inflight().await.unwrap();
    let removed = store
        .purge_terminal_inflight(Utc::now(), chrono::Duration::zero())
        .await
        .unwrap();
    let after = malformed_state_count();

    assert!(listed.is_empty(), "absent inflight.json is empty state");
    assert_eq!(removed, 0, "absent inflight.json has nothing to purge");
    assert_eq!(
        after, before,
        "absent inflight.json must stay silent and uncounted"
    );
}

#[tokio::test]
async fn cursor_roundtrip_and_updated_at() {
    let tmp = tempdir().unwrap();
    let store = FsResearchStore::new(tmp.path().to_path_buf());
    let _ = store.create_spec(&make_spec("id-5", "t")).await;
    let empty = store.load_cursor("id-5").await.unwrap();
    assert!(empty.data.is_empty());
    let mut c = Cursor::default();
    c.data.insert("page".into(), serde_json::Value::from(3_i64));
    store.save_cursor("id-5", &c).await.unwrap();
    let reloaded = store.load_cursor("id-5").await.unwrap();
    assert_eq!(reloaded.data.get("page"), Some(&serde_json::json!(3)));
    assert!(reloaded.updated_at.is_some());
}

#[tokio::test]
async fn purge_terminal_inflight_removes_only_old_terminals() {
    use crate::research::inflight::{Inflight, RunState};

    let tmp = tempdir().unwrap();
    let store = FsResearchStore::new(tmp.path().to_path_buf());
    let now = Utc::now();

    // Spec A: terminal Completed, finished_at = 30 days ago → must purge.
    store
        .create_spec(&make_spec("a-old-completed", "t"))
        .await
        .unwrap();
    let mut a = Inflight::scheduled("a-old-completed", 1);
    a.state = RunState::Completed;
    a.finished_at = Some(now - chrono::Duration::days(30));
    store.save_inflight("a-old-completed", &a).await.unwrap();

    // Spec B: terminal Failed, finished_at = 1 hour ago → must keep.
    store
        .create_spec(&make_spec("b-recent-failed", "t"))
        .await
        .unwrap();
    let mut b = Inflight::scheduled("b-recent-failed", 1);
    b.state = RunState::Failed;
    b.finished_at = Some(now - chrono::Duration::hours(1));
    store.save_inflight("b-recent-failed", &b).await.unwrap();

    // Spec C: Running → must keep regardless of age.
    store
        .create_spec(&make_spec("c-running", "t"))
        .await
        .unwrap();
    let mut c = Inflight::scheduled("c-running", 1);
    c.state = RunState::Running;
    c.started_at = Some(now - chrono::Duration::days(99));
    store.save_inflight("c-running", &c).await.unwrap();

    // Spec D: Scheduled → must keep regardless of age.
    store
        .create_spec(&make_spec("d-scheduled", "t"))
        .await
        .unwrap();
    let mut d = Inflight::scheduled("d-scheduled", 1);
    d.scheduled_at = now - chrono::Duration::days(99);
    store.save_inflight("d-scheduled", &d).await.unwrap();

    // Spec E: Completed but no finished_at → must keep (defensive).
    store
        .create_spec(&make_spec("e-no-finished-at", "t"))
        .await
        .unwrap();
    let mut e = Inflight::scheduled("e-no-finished-at", 1);
    e.state = RunState::Completed;
    e.finished_at = None;
    store.save_inflight("e-no-finished-at", &e).await.unwrap();

    let removed = store
        .purge_terminal_inflight(now, chrono::Duration::days(7))
        .await
        .unwrap();

    assert_eq!(removed, 1, "only the 30-day-old Completed should be purged");
    assert!(
        store
            .load_inflight("a-old-completed")
            .await
            .unwrap()
            .is_none()
    );
    assert!(
        store
            .load_inflight("b-recent-failed")
            .await
            .unwrap()
            .is_some()
    );
    assert!(store.load_inflight("c-running").await.unwrap().is_some());
    assert!(store.load_inflight("d-scheduled").await.unwrap().is_some());
    assert!(
        store
            .load_inflight("e-no-finished-at")
            .await
            .unwrap()
            .is_some()
    );
}

#[tokio::test]
async fn purge_terminal_inflight_zero_retention_purges_all_terminals_with_finished_at() {
    use crate::research::inflight::{Inflight, RunState};

    let tmp = tempdir().unwrap();
    let store = FsResearchStore::new(tmp.path().to_path_buf());
    let now = Utc::now();

    for (id, state) in [("c1", RunState::Completed), ("c2", RunState::Failed)] {
        store.create_spec(&make_spec(id, "t")).await.unwrap();
        let mut i = Inflight::scheduled(id, 1);
        i.state = state;
        i.finished_at = Some(now - chrono::Duration::seconds(1));
        store.save_inflight(id, &i).await.unwrap();
    }

    let removed = store
        .purge_terminal_inflight(now, chrono::Duration::zero())
        .await
        .unwrap();
    assert_eq!(removed, 2);
}

#[tokio::test]
async fn purge_terminal_inflight_handles_missing_root() {
    let tmp = tempdir().unwrap();
    let missing = tmp.path().join("does-not-exist");
    let store = FsResearchStore::new(missing);
    let removed = store
        .purge_terminal_inflight(Utc::now(), chrono::Duration::days(1))
        .await
        .unwrap();
    assert_eq!(removed, 0);
}

#[tokio::test]
async fn purge_terminal_inflight_default_trait_impl_is_noop() {
    struct NoopStore;
    #[async_trait]
    impl SpecStore for NoopStore {
        async fn create_spec(&self, _spec: &ResearchSpec) -> Result<()> {
            Ok(())
        }
        async fn load_spec(&self, _id: &str) -> Result<ResearchSpec> {
            Err(AgentError::Provider("unused".into()))
        }
        async fn save_spec(&self, _spec: &ResearchSpec) -> Result<()> {
            Ok(())
        }
        async fn list_specs(&self) -> Result<Vec<ResearchSpec>> {
            Ok(vec![])
        }
        async fn delete_spec(&self, _id: &str) -> Result<()> {
            Ok(())
        }
    }
    #[async_trait]
    impl FindingStore for NoopStore {
        async fn try_append_finding(&self, _finding: &Finding) -> Result<bool> {
            Ok(false)
        }
        async fn upsert_finding(&self, _finding: &Finding) -> Result<bool> {
            Ok(false)
        }
        async fn list_findings(&self, _id: &str, _limit: Option<usize>) -> Result<Vec<Finding>> {
            Ok(vec![])
        }
        async fn count_findings(&self, _id: &str) -> Result<u32> {
            Ok(0)
        }
        async fn remove_findings_by_hash(
            &self,
            _id: &str,
            _hashes: &HashSet<String>,
        ) -> Result<u32> {
            Ok(0)
        }
    }
    #[async_trait]
    impl RunStore for NoopStore {
        async fn append_run(&self, _run: &RunRecord) -> Result<()> {
            Ok(())
        }
        async fn list_runs(&self, _id: &str, _limit: Option<usize>) -> Result<Vec<RunRecord>> {
            Ok(vec![])
        }
        async fn load_cursor(&self, _id: &str) -> Result<Cursor> {
            Ok(Cursor::default())
        }
        async fn save_cursor(&self, _id: &str, _cursor: &Cursor) -> Result<()> {
            Ok(())
        }
    }
    #[async_trait]
    impl ReportStore for NoopStore {
        async fn write_report(&self, _id: &str, _report: &str) -> Result<()> {
            Ok(())
        }
        async fn read_report(&self, _id: &str) -> Result<Option<String>> {
            Ok(None)
        }
        async fn write_agent_brief(&self, _id: &str, _brief: &str) -> Result<()> {
            Ok(())
        }
        async fn read_agent_brief(&self, _id: &str) -> Result<Option<String>> {
            Ok(None)
        }
    }
    impl InflightStore for NoopStore {}
    let store = NoopStore;
    let removed = store
        .purge_terminal_inflight(Utc::now(), chrono::Duration::days(1))
        .await
        .unwrap();
    assert_eq!(removed, 0);
}

#[tokio::test]
async fn delete_removes_dir_and_locks() {
    let tmp = tempdir().unwrap();
    let store = FsResearchStore::new(tmp.path().to_path_buf());
    let spec = make_spec("id-6", "t");
    store.create_spec(&spec).await.unwrap();
    let f = make_finding("id-6", "https://x.com/a");
    store.try_append_finding(&f).await.unwrap();
    store.delete_spec("id-6").await.unwrap();
    assert!(store.load_spec("id-6").await.is_err());
    // Recreating is allowed after delete.
    store.create_spec(&spec).await.unwrap();
}
