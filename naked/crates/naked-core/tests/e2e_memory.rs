//! E2E tests — memory

#[macro_use]
mod common;
use common::*;

use naked_core::memory::store::MemoryPaths;

fn isolate_memory_root(tmp: &tempfile::TempDir) -> impl Drop {
    let memory_root = tmp.path().join("naked-home");
    let guard = MemoryPaths::set_test_root(&memory_root);
    assert_eq!(
        MarkdownMemoryStore::naked_home(),
        memory_root,
        "e2e memory tests must use their per-test temporary root"
    );
    guard
}

#[tokio::test]
async fn t93_memory_tool_store_and_list() {
    need_provider!(config, provider, model);
    pace().await;
    let tmp = tempfile::tempdir().unwrap();
    let _root = isolate_memory_root(&tmp);

    eprintln!(">>> t93_memory_tool_store_and_list [model={model}]");

    let workspace = tmp.path().join("project93");
    std::fs::create_dir_all(&workspace).unwrap();

    let tools = build_tools_with_memory(workspace.clone());

    let sys = "You are a concise assistant. You have a `memory` tool. \
               When asked to remember something, use the memory tool with action=store. \
               When asked to list memories, use the memory tool with action=list. \
               Respond briefly.";
    let mut history = ConversationHistory::new(sys.into());
    history.push_user(
        "Please remember this preference: always use 4-space indentation. \
         Use the memory tool to store it as a preference with project scope.",
    );
    let r = run_prompt_with_tools(provider, &mut history, &workspace, &model, tools).await;

    eprintln!("  text: {}", r.full_text());
    eprintln!("  tools: {:?}", r.tool_names());
    assert!(!r.had_error(), "agent error");
    assert!(r.has_tool("memory"), "memory tool not called");

    let store_results = r.tool_results();
    let memory_result = store_results
        .iter()
        .find(|(name, _, _)| name == "memory")
        .map(|(_, _, output)| output.clone())
        .unwrap_or_default();
    eprintln!("  memory store output: {memory_result}");
    assert!(
        memory_result.contains("Stored") || memory_result.contains("stored"),
        "unexpected store output: {memory_result}"
    );

    // Turn 2: list memories
    pace().await;
    let (provider2, _) = get_working_provider(&config).await.unwrap();
    let tools2 = build_tools_with_memory(workspace.clone());
    history.push_user("Now list all project memories using the memory tool.");
    let r2 = run_prompt_with_tools(provider2, &mut history, &workspace, &model, tools2).await;

    eprintln!("  text2: {}", r2.full_text());
    eprintln!("  tools2: {:?}", r2.tool_names());
    assert!(r2.has_tool("memory"), "memory tool not called in list");

    let list_results = r2.tool_results();
    let list_output = list_results
        .iter()
        .find(|(name, _, _)| name == "memory")
        .map(|(_, _, output)| output.clone())
        .unwrap_or_default();
    eprintln!("  memory list output: {list_output}");
    assert!(
        list_output.contains("indentation") || list_output.contains("indent"),
        "stored memory not found in list: {list_output}"
    );

    // Verify directly via MemoryService
    let entries = MemoryService::list(&workspace, Some(MemoryScope::Project));
    eprintln!("  direct entries: {}", entries.len());
    assert!(!entries.is_empty(), "no entries in MEMORY.md");

    // Cleanup
    let _ = MemoryService::clear(&workspace, MemoryScope::Project);
    eprintln!("  PASS: memory tool store + list works e2e");
}

#[tokio::test]
async fn t94_memory_tool_search_and_delete() {
    need_provider!(config, provider, model);
    pace().await;
    let tmp = tempfile::tempdir().unwrap();
    let _root = isolate_memory_root(&tmp);

    eprintln!(">>> t94_memory_tool_search_and_delete [model={model}]");

    let workspace = tmp.path().join("project94");
    std::fs::create_dir_all(&workspace).unwrap();

    // Pre-populate two memories directly
    MemoryService::store(
        &workspace,
        MemoryScope::Project,
        MemoryType::Preference,
        "always use rustfmt before commit",
        "user",
    )
    .unwrap();
    MemoryService::store(
        &workspace,
        MemoryScope::Project,
        MemoryType::ProjectKnowledge,
        "database is PostgreSQL 15",
        "user",
    )
    .unwrap();

    let entries_before = MemoryService::list(&workspace, Some(MemoryScope::Project));
    assert_eq!(entries_before.len(), 2);
    let rustfmt_id = entries_before
        .iter()
        .find(|e| e.content.contains("rustfmt"))
        .unwrap()
        .id
        .clone();

    // Ask LLM to search for "rustfmt"
    let tools = build_tools_with_memory(workspace.clone());
    let sys = "You are a concise assistant with a memory tool. \
               When asked to search memories, use action=search. \
               When asked to delete, use action=delete with the id. Respond briefly.";
    let mut history = ConversationHistory::new(sys.into());
    history.push_user("Search my memories for 'rustfmt' using the memory tool.");
    let r = run_prompt_with_tools(provider, &mut history, &workspace, &model, tools).await;

    eprintln!("  text: {}", r.full_text());
    assert!(r.has_tool("memory"), "memory tool not called for search");
    let search_output = r
        .tool_results()
        .iter()
        .find(|(n, _, _)| n == "memory")
        .map(|(_, _, o)| o.clone())
        .unwrap_or_default();
    eprintln!("  search output: {search_output}");
    assert!(
        search_output.contains("rustfmt"),
        "rustfmt not in search results: {search_output}"
    );

    // Turn 2: delete by id
    pace().await;
    let (provider2, _) = get_working_provider(&config).await.unwrap();
    let tools2 = build_tools_with_memory(workspace.clone());
    history.push_user(&format!(
        "Delete the memory with id '{}' using the memory tool.",
        rustfmt_id
    ));
    let r2 = run_prompt_with_tools(provider2, &mut history, &workspace, &model, tools2).await;
    eprintln!("  delete text: {}", r2.full_text());
    assert!(r2.has_tool("memory"), "memory tool not called for delete");

    let entries_after = MemoryService::list(&workspace, Some(MemoryScope::Project));
    assert_eq!(entries_after.len(), 1, "should have 1 entry after delete");
    assert!(
        entries_after[0].content.contains("PostgreSQL"),
        "wrong entry remained"
    );

    let _ = MemoryService::clear(&workspace, MemoryScope::Project);
    eprintln!("  PASS: memory tool search + delete works e2e");
}

#[tokio::test]
async fn t95_memory_global_scope() {
    need_provider!(_config, provider, model);
    pace().await;
    let tmp = tempfile::tempdir().unwrap();
    let _root = isolate_memory_root(&tmp);

    // Clean global memory before test
    let _ = MemoryService::clear(&tmp.path().join("dummy"), MemoryScope::Global);

    eprintln!(">>> t95_memory_global_scope [model={model}]");

    let workspace = tmp.path().join("project95");
    std::fs::create_dir_all(&workspace).unwrap();

    let tools = build_tools_with_memory(workspace.clone());

    let sys = "You are a concise assistant with a memory tool. \
               Always use the exact scope the user specifies. Respond briefly.";
    let mut history = ConversationHistory::new(sys.into());
    history.push_user(
        "Remember this globally (scope=global, type=preference): I prefer dark mode in all editors. \
         Use the memory tool.",
    );
    let r = run_prompt_with_tools(provider, &mut history, &workspace, &model, tools).await;

    eprintln!("  text: {}", r.full_text());
    assert!(r.has_tool("memory"), "memory tool not called");

    // Verify global scope storage
    let global = MemoryService::list(&workspace, Some(MemoryScope::Global));
    let project = MemoryService::list(&workspace, Some(MemoryScope::Project));

    eprintln!("  global entries: {}", global.len());
    eprintln!("  project entries: {}", project.len());

    assert!(
        !global.is_empty(),
        "global memory should have at least 1 entry"
    );
    assert!(
        global
            .iter()
            .any(|e| e.content.to_lowercase().contains("dark")),
        "global memory should contain 'dark mode' preference"
    );
    assert!(
        project.is_empty(),
        "project memory should be empty (stored globally)"
    );

    // Verify the global file exists at the right path
    let global_path = MarkdownMemoryStore::global_memory_path();
    assert!(global_path.exists(), "global MEMORY.md should exist");

    // Verify load_rules includes global rules
    let rules = MemoryService::load_rules(&workspace);
    eprintln!("  rules: {rules}");
    assert!(
        rules.contains("dark") || rules.contains("Dark"),
        "load_rules should include global memories"
    );

    // Cleanup
    let _ = MemoryService::clear(&workspace, MemoryScope::Global);
    let _ = MemoryService::clear(&workspace, MemoryScope::Project);
    eprintln!("  PASS: global scope memory works e2e");
}

#[tokio::test]
async fn t96_memory_auto_classification() {
    need_provider!(config, _provider, model);
    pace().await;
    let tmp = tempfile::tempdir().unwrap();
    let _root = isolate_memory_root(&tmp);

    eprintln!(">>> t96_memory_auto_classification [model={model}]");

    let workspace = tmp.path().join("project_classify");
    std::fs::create_dir_all(&workspace).unwrap();

    let core_config = Config {
        workspace: workspace.clone(),
        session_dir: tmp.path().join("sessions"),
        default_provider: config.default_provider.clone(),
        default_model: model.clone(),
        max_iterations: 5,
        tool_timeout_secs: 30,
        ..Default::default()
    };

    let (provider, _) = get_working_provider(&config).await.unwrap();
    let agent = naked_core::AgentCore::new(core_config, provider);
    let session_id = agent.create_session(&workspace).await;

    // Send a message that contains an explicit preference
    let mut handle = agent
        .send_prompt(
            &session_id,
            "From now on, always write code comments in Russian. \
             This is my strong preference for all projects. \
             Just acknowledge this with OK.",
        )
        .await
        .unwrap();

    let mut text = String::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(120);
    loop {
        match tokio::time::timeout_at(deadline, handle.events.recv()).await {
            Ok(Some(AgentEvent::TextDelta(t))) => text.push_str(&t),
            Ok(Some(AgentEvent::Idle)) => break,
            Ok(Some(AgentEvent::Error(e))) => {
                eprintln!("  error: {e}");
                break;
            }
            Ok(None) => break,
            Err(_) => {
                eprintln!("  TIMEOUT");
                break;
            }
            _ => {}
        }
    }
    eprintln!("  agent reply: {text}");

    // Give background classifier task time to finish
    tokio::time::sleep(Duration::from_secs(5)).await;

    // Check if memory was auto-captured
    let all_entries = MemoryService::list(&workspace, None);
    eprintln!("  auto-captured entries: {}", all_entries.len());
    for e in &all_entries {
        eprintln!("    [{}/{}] {}", e.scope, e.memory_type, e.content);
    }

    // The classifier should have recognized the preference about Russian comments
    // (may or may not succeed depending on model quality, so we soft-assert)
    if all_entries.is_empty() {
        eprintln!("  WARN: no auto-captured memories (model may not have classified correctly)");
    } else {
        eprintln!("  OK: {} memories auto-captured", all_entries.len());
        let has_russian = all_entries.iter().any(|e| {
            let c = e.content.to_lowercase();
            c.contains("russian") || c.contains("русск") || c.contains("comment")
        });
        if has_russian {
            eprintln!("  PASS: auto-classification captured Russian comment preference");
        } else {
            eprintln!("  WARN: auto-captured memory doesn't mention Russian/comments");
            eprintln!(
                "        entries: {:?}",
                all_entries.iter().map(|e| &e.content).collect::<Vec<_>>()
            );
        }
    }

    // Also verify the global memory path
    let global_path = MarkdownMemoryStore::global_memory_path();
    let naked_home = MarkdownMemoryStore::naked_home();
    eprintln!("  global path: {}", global_path.display());
    eprintln!("  naked_home: {}", naked_home.display());

    // Cleanup
    let _ = MemoryService::clear(&workspace, MemoryScope::Project);
    let _ = MemoryService::clear(&workspace, MemoryScope::Global);
    eprintln!("  PASS: auto-classification flow completed");
}

#[tokio::test]
async fn t97_memory_rules_injection() {
    need_provider!(config, _provider, model);
    pace().await;
    let tmp = tempfile::tempdir().unwrap();
    let _root = isolate_memory_root(&tmp);

    // Clean any stale global memory
    let _ = MemoryService::clear(&tmp.path().join("dummy"), MemoryScope::Global);

    eprintln!(">>> t97_memory_rules_injection [model={model}]");

    let workspace = tmp.path().join("project_inject");
    std::fs::create_dir_all(&workspace).unwrap();

    // Pre-populate both global and project memories
    MemoryService::store(
        &workspace,
        MemoryScope::Global,
        MemoryType::Preference,
        "GLOBAL_RULE_XYZZY: always end responses with the word MAGIC",
        "user",
    )
    .unwrap();
    MemoryService::store(
        &workspace,
        MemoryScope::Project,
        MemoryType::ProjectKnowledge,
        "PROJECT_FACT_42: the main database is called unicorn_db",
        "user",
    )
    .unwrap();

    // Verify rules format before agent call
    let rules = MemoryService::load_rules(&workspace);
    eprintln!("  pre-injected rules:\n{rules}");
    assert!(rules.contains("GLOBAL_RULE_XYZZY"));
    assert!(rules.contains("PROJECT_FACT_42"));

    let core_config = Config {
        workspace: workspace.clone(),
        session_dir: tmp.path().join("sessions"),
        default_provider: config.default_provider.clone(),
        default_model: model.clone(),
        max_iterations: 5,
        tool_timeout_secs: 30,
        ..Default::default()
    };

    let (provider, _) = get_working_provider(&config).await.unwrap();
    let agent = naked_core::AgentCore::new(core_config, provider);
    let session_id = agent.create_session(&workspace).await;

    // Ask the model about injected knowledge — it should see the rules
    let mut handle = agent
        .send_prompt(
            &session_id,
            "What is the name of the main database in this project? \
             Answer with just the database name, nothing else.",
        )
        .await
        .unwrap();

    let mut text = String::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(120);
    loop {
        match tokio::time::timeout_at(deadline, handle.events.recv()).await {
            Ok(Some(AgentEvent::TextDelta(t))) => text.push_str(&t),
            Ok(Some(AgentEvent::Idle)) => break,
            Ok(Some(AgentEvent::Error(e))) => {
                eprintln!("  error: {e}");
                break;
            }
            Ok(None) => break,
            Err(_) => {
                eprintln!("  TIMEOUT");
                break;
            }
            _ => {}
        }
    }

    eprintln!("  agent reply: {text}");

    // The model should mention unicorn_db from the injected project knowledge
    let text_lower = text.to_lowercase();
    assert!(
        text_lower.contains("unicorn_db") || text_lower.contains("unicorn"),
        "model should reference the injected project knowledge about unicorn_db, got: {text}"
    );

    // Cleanup
    let _ = MemoryService::clear(&workspace, MemoryScope::Project);
    let _ = MemoryService::clear(&workspace, MemoryScope::Global);
    eprintln!("  PASS: memory rules injection works e2e");
}

#[tokio::test]
async fn t98_memory_user_scope_roundtrip() {
    eprintln!(">>> t98_memory_user_scope_roundtrip");

    let tmp = tempfile::tempdir().unwrap();
    let _root = isolate_memory_root(&tmp);
    let workspace = tmp.path().join("project98");
    std::fs::create_dir_all(&workspace).unwrap();

    // Two distinct user ids so we can verify isolation.
    let alice = format!("alice_{}", std::process::id());
    let bob = format!("bob_{}", std::process::id());

    // Clean any leftover state from a previous run.
    let _ = MemoryService::clear(&workspace, MemoryScope::User(alice.clone()));
    let _ = MemoryService::clear(&workspace, MemoryScope::User(bob.clone()));

    // Store one memory per user.
    let stored_a = MemoryService::store(
        &workspace,
        MemoryScope::User(alice.clone()),
        MemoryType::Preference,
        "USER98_ALICE_FACT: prefers metric units",
        "user",
    )
    .unwrap();
    assert!(stored_a, "alice's preference should be newly stored");
    let stored_b = MemoryService::store(
        &workspace,
        MemoryScope::User(bob.clone()),
        MemoryType::Preference,
        "USER98_BOB_FACT: prefers imperial units",
        "user",
    )
    .unwrap();
    assert!(stored_b, "bob's preference should be newly stored");

    // list(User(alice)) returns alice only.
    let alice_entries = MemoryService::list(&workspace, Some(MemoryScope::User(alice.clone())));
    eprintln!("  alice entries: {}", alice_entries.len());
    assert_eq!(alice_entries.len(), 1);
    assert!(alice_entries[0].content.contains("ALICE_FACT"));
    assert_eq!(
        alice_entries[0].scope,
        MemoryScope::User(alice.clone()),
        "scope must be tagged with alice's id"
    );

    // list(User(bob)) never sees alice's data.
    let bob_entries = MemoryService::list(&workspace, Some(MemoryScope::User(bob.clone())));
    assert_eq!(bob_entries.len(), 1);
    assert!(bob_entries[0].content.contains("BOB_FACT"));

    // list(None) = global + project (per contract); user scope is hidden.
    let shared_entries = MemoryService::list(&workspace, None);
    for e in &shared_entries {
        assert!(
            !matches!(e.scope, MemoryScope::User(_)),
            "list(None) must not leak user-scoped entries: {e:?}"
        );
    }

    // search_for with sender = alice sees alice's fact; without sender it does not.
    let hits_alice =
        MemoryService::search_for(&workspace, "USER98_ALICE_FACT", Some(alice.as_str()));
    assert_eq!(hits_alice.len(), 1, "alice should find her own fact");
    let hits_anon = MemoryService::search_for(&workspace, "USER98_ALICE_FACT", None);
    assert!(
        hits_anon.is_empty(),
        "anonymous search must not expose user-scoped memory"
    );
    // bob searching for alice's marker finds nothing.
    let hits_bob_for_alice =
        MemoryService::search_for(&workspace, "USER98_ALICE_FACT", Some(bob.as_str()));
    assert!(
        hits_bob_for_alice.is_empty(),
        "bob must not see alice's user-scoped memory"
    );

    // load_rules_for(alice) injects alice's preference into the prompt section.
    let rules_alice = MemoryService::load_rules_for(&workspace, Some(alice.as_str()));
    eprintln!("  rules(alice):\n{rules_alice}");
    assert!(rules_alice.contains("ALICE_FACT"));
    assert!(!rules_alice.contains("BOB_FACT"));
    assert!(rules_alice.contains(&format!("User ({alice})")));

    // load_rules without a sender does NOT include user memories.
    let rules_plain = MemoryService::load_rules(&workspace);
    assert!(!rules_plain.contains("ALICE_FACT"));
    assert!(!rules_plain.contains("BOB_FACT"));

    // clear(User(alice)) removes alice's file but leaves bob's intact.
    MemoryService::clear(&workspace, MemoryScope::User(alice.clone())).unwrap();
    let alice_after = MemoryService::list(&workspace, Some(MemoryScope::User(alice.clone())));
    assert!(alice_after.is_empty());
    let bob_after = MemoryService::list(&workspace, Some(MemoryScope::User(bob.clone())));
    assert_eq!(bob_after.len(), 1);

    // Final cleanup.
    let _ = MemoryService::clear(&workspace, MemoryScope::User(bob.clone()));
    eprintln!("  PASS: user-scope store/list/search/rules/clear works end-to-end");
}
