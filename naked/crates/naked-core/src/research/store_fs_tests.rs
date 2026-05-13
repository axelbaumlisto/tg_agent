use super::*;
use crate::research::spec::{Finding, ResearchSpec, RunRecord, dedup_hash};
use chrono::Utc;
use tempfile::tempdir;

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
