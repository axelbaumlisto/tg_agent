//! Phase-7 unit-level e2e for `AgentCore::ask_research`.
//!
//! Confirms the LLM-backed `/research ask` flow:
//!   1. The provider receives a single user message containing the topic and
//!      the numbered findings corpus.
//!   2. Findings are truncated to the per-finding excerpt budget.
//!   3. The LLM's reply is returned verbatim.
//!   4. Empty corpora short-circuit without calling the provider.
//!
//! Uses a stub provider that records every request and yields scripted text.

use std::pin::Pin;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use naked_core::AgentCore;
use naked_core::config::Config;
use naked_core::provider::{ChatRequest, Provider};
use naked_core::types::{ModelInfo, StreamChunk};
use tempfile::tempdir;
use tokio_stream::Stream;

#[derive(Default)]
struct ScriptedProvider {
    reply: Mutex<String>,
    captured: Mutex<Vec<ChatRequest>>,
}

impl ScriptedProvider {
    fn new(reply: &str) -> Arc<Self> {
        Arc::new(Self {
            reply: Mutex::new(reply.into()),
            captured: Mutex::new(vec![]),
        })
    }
}

#[async_trait]
impl Provider for ScriptedProvider {
    fn name(&self) -> &str {
        "scripted"
    }
    fn models(&self) -> Vec<ModelInfo> {
        vec![]
    }
    async fn stream_chat(
        &self,
        request: ChatRequest,
    ) -> naked_core::error::Result<Pin<Box<dyn Stream<Item = StreamChunk> + Send>>> {
        let reply = self.reply.lock().unwrap().clone();
        self.captured.lock().unwrap().push(request);
        let stream = tokio_stream::iter(vec![StreamChunk::Text(reply), StreamChunk::Done]);
        Ok(Box::pin(stream))
    }
}

/// Adapter so we can hand a Box<dyn Provider> to AgentCore while still keeping
/// a handle to inspect captured requests.
struct ProviderHandle(Arc<ScriptedProvider>);

#[async_trait]
impl Provider for ProviderHandle {
    fn name(&self) -> &str {
        self.0.name()
    }
    fn models(&self) -> Vec<ModelInfo> {
        self.0.models()
    }
    async fn stream_chat(
        &self,
        request: ChatRequest,
    ) -> naked_core::error::Result<Pin<Box<dyn Stream<Item = StreamChunk> + Send>>> {
        self.0.stream_chat(request).await
    }
}

fn make_core(provider: Arc<ScriptedProvider>) -> Arc<AgentCore> {
    let tmp = tempdir().unwrap();
    let cfg = Config {
        workspace: tmp.path().to_path_buf(),
        session_dir: tmp.path().join("sessions"),
        research: naked_core::config::ResearchConfig {
            enabled: true,
            storage_dir: Some(tmp.path().join("research")),
            ..Default::default()
        },
        ..Default::default()
    };
    let agent = Arc::new(AgentCore::new(
        cfg,
        Box::new(ProviderHandle(provider)) as Box<dyn Provider>,
    ));
    agent.init_self_ref();
    std::mem::forget(tmp);
    agent
}

#[tokio::test]
async fn ask_research_returns_no_findings_message_when_corpus_empty() {
    let provider = ScriptedProvider::new("(should not be called)");
    let agent = make_core(provider.clone());
    let spec = agent
        .create_research("empty-ask", vec![], None, None, None)
        .await
        .expect("create");
    let answer = agent
        .ask_research(&spec.id, "Where are the listings?")
        .await
        .expect("ask");
    assert!(answer.contains("No findings yet"), "got: {answer}");
    assert!(
        provider.captured.lock().unwrap().is_empty(),
        "provider must not be invoked when corpus is empty"
    );
}

#[tokio::test]
async fn ask_research_packages_findings_into_prompt_and_returns_llm_reply() {
    let provider = ScriptedProvider::new("Best match is [1] at $500/mo.");
    let agent = make_core(provider.clone());
    let spec = agent
        .create_research(
            "ask-pack",
            vec!["https://seed.example".into()],
            None,
            None,
            None,
        )
        .await
        .expect("create");

    // Seed two findings via the store directly so we don't depend on tools.
    let store = agent.research_store();
    let now = chrono::Utc::now();
    let mut a = naked_core::research::Finding {
        id: "a".into(),
        research_id: spec.id.clone(),
        run_id: "r1".into(),
        url: "https://ex.com/listing-a".into(),
        title: Some("70m² 2BR District 7".into()),
        excerpt: Some("Spacious apartment, fully furnished, contact 0900111222.".into()),
        price: Some("$500/mo".into()),
        listing_date: Some("18/04/2026".into()),
        source_content: Some("S".repeat(200)),
        dedup_hash: naked_core::research::dedup_hash("https://ex.com/listing-a"),
        host_path_hash: naked_core::research::spec::host_path_hash(
            "https://ex.com/listing-a",
        ),
        content_hash: naked_core::research::spec::content_hash(
            "Spacious apartment, fully furnished, contact 0900111222.",
        ),
        seen_at: now,
    };
    a.dedup_hash = naked_core::research::dedup_hash(&a.url);
    a.host_path_hash = naked_core::research::spec::host_path_hash(&a.url);
    let mut b = a.clone();
    b.id = "b".into();
    b.url = "https://ex.com/listing-b".into();
    b.title = Some("120m² 3BR District 2".into());
    b.price = Some("$900/mo".into());
    b.dedup_hash = naked_core::research::dedup_hash(&b.url);
    b.host_path_hash = naked_core::research::spec::host_path_hash(&b.url);
    b.content_hash = naked_core::research::spec::content_hash(
        "Different content for the second listing here.",
    );
    store.upsert_finding(&a).await.expect("append a");
    store.upsert_finding(&b).await.expect("append b");

    let answer = agent
        .ask_research(&spec.id, "Which one is cheapest?")
        .await
        .expect("ask");
    assert_eq!(answer, "Best match is [1] at $500/mo.");

    let captured = provider.captured.lock().unwrap();
    assert_eq!(captured.len(), 1);
    let req = &captured[0];
    assert!(req.tools.is_empty(), "ask_research must not expose tools");
    assert_eq!(req.messages.len(), 1);
    let user_text = req.messages[0]["content"].as_str().unwrap();
    assert!(
        user_text.contains("Research topic:"),
        "user text missing topic header"
    );
    assert!(
        user_text.contains("[1]") && user_text.contains("[2]"),
        "user text must enumerate findings"
    );
    assert!(
        user_text.contains("https://ex.com/listing-a"),
        "user text must include URLs"
    );
    assert!(
        user_text.contains("Which one is cheapest?"),
        "user text must include the question"
    );
}

#[tokio::test]
async fn ask_research_truncates_oversized_excerpts() {
    let provider = ScriptedProvider::new("ok");
    let agent = make_core(provider.clone());
    let spec = agent
        .create_research(
            "ask-trim",
            vec!["https://seed.example".into()],
            None,
            None,
            None,
        )
        .await
        .expect("create");

    let store = agent.research_store();
    let huge = "X".repeat(5000);
    let f = naked_core::research::Finding {
        id: "huge".into(),
        research_id: spec.id.clone(),
        run_id: "r1".into(),
        url: "https://ex.com/huge".into(),
        title: Some("Huge listing".into()),
        excerpt: Some(huge),
        price: Some("$1".into()),
        listing_date: Some("18/04/2026".into()),
        source_content: Some("S".repeat(200)),
        dedup_hash: naked_core::research::dedup_hash("https://ex.com/huge"),
        host_path_hash: naked_core::research::spec::host_path_hash("https://ex.com/huge"),
        content_hash: naked_core::research::spec::content_hash("S"),
        seen_at: chrono::Utc::now(),
    };
    store.upsert_finding(&f).await.expect("append");

    agent
        .ask_research(&spec.id, "anything?")
        .await
        .expect("ask");

    let captured = provider.captured.lock().unwrap();
    let user_text = captured[0].messages[0]["content"].as_str().unwrap();
    let x_count = user_text.chars().filter(|c| *c == 'X').count();
    assert!(
        x_count > 0 && x_count <= 700,
        "excerpt must be truncated to ~600-char budget; saw {x_count} X's"
    );
    assert!(user_text.contains("…"), "truncation marker must be present");
}

#[tokio::test]
async fn ask_research_rejects_empty_question() {
    let provider = ScriptedProvider::new("nope");
    let agent = make_core(provider);
    let spec = agent
        .create_research("ask-empty", vec![], None, None, None)
        .await
        .expect("create");
    let err = agent
        .ask_research(&spec.id, "   ")
        .await
        .expect_err("must error");
    assert!(format!("{err}").contains("question"));
}
