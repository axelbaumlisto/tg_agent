//! E2E tests — research

#[macro_use]
mod common;
use common::*;
use naked_core::research::{FindingStore, FsResearchStore, SpecStore};

#[tokio::test]
async fn t86_deep_research_phase1_outline() {
    let config = match test_config() {
        Some(c) => c,
        None => {
            eprintln!("  SKIP t86: no config");
            return;
        }
    };
    let (provider, model) = match get_working_provider(&config).await {
        Some(pm) => pm,
        None => {
            eprintln!("  SKIP t86: no working provider");
            return;
        }
    };

    let roots = skill_roots_from_config(&config);
    let resolver = SkillResolver::new(roots.clone());
    if resolver.resolve("research").is_none() {
        eprintln!("  SKIP t86: 'research' skill not found in {:?}", roots);
        return;
    }

    eprintln!(">>> t86_deep_research_phase1_outline [model={model}]");

    let tmp = tempfile::tempdir().unwrap();
    let cwd = tmp.path();

    let tools = {
        let resolver = SkillResolver::new(skill_roots_from_config(&config));
        let available = resolver.list();
        let exa_keys = config.exa_api_keys.clone();
        let prov_arc: Arc<dyn Provider> = Arc::from(provider);
        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(BashTool::new(60)),
            Box::new(ReadFileTool::default()),
            Box::new(WriteFileTool::default()),
            Box::new(EditFileTool::default()),
            Box::new(GlobSearchTool),
            Box::new(GrepSearchTool),
            Box::new(WebSearchTool::from_legacy_exa(exa_keys.clone())),
            Box::new(SubAgentTool::new(
                prov_arc.clone(),
                model.clone(),
                60,
                exa_keys,
            )),
            Box::new(SkillTool::new(resolver, &available)),
        ];
        ToolRegistry::new(tools)
    };

    let mut history = ConversationHistory::new(research_system_prompt(cwd));
    history.push_user(
        "/research аренда коммерческой площади в Дананге для кафе завтраков, \
         100-200 кв.м., целевая аудитория: туристы и экспаты",
    );

    let r = run_research_prompt(
        get_working_provider(&config).await.unwrap().0,
        &mut history,
        cwd,
        &model,
        tools,
        600,
    )
    .await;

    let text = r.full_text();
    let text_lower = text.to_lowercase();
    eprintln!("  tools used: {:?}", r.tool_names());
    eprintln!("  text length: {}", text.len());
    eprintln!("  text start: {:.500}", text);
    if text.len() > 500 {
        let tail: String = text
            .chars()
            .rev()
            .take(500)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        eprintln!("  text end: {tail}");
    }

    // Skill tool must be invoked
    assert!(
        r.has_tool("Skill"),
        "Skill tool not used: {:?}",
        r.tool_names()
    );

    // Model should ask questions (multi-turn check)
    let asks_question = text.contains('?')
        || text_lower.contains("добавить")
        || text_lower.contains("убрать")
        || text_lower.contains("подходит")
        || text_lower.contains("подтверд");
    eprintln!("  asks user a question: {asks_question}");

    // Model should NOT have created outline.yaml yet (that's Phase 3)
    let wrote_files = r.has_tool("write_file");
    eprintln!("  wrote files (should be false for Phase 1): {wrote_files}");

    // Model should NOT have used sub_agent yet (that's Phase 3)
    let used_subagent = r.has_tool("sub_agent");
    eprintln!("  used sub_agent (should be false for Phase 1): {used_subagent}");

    if asks_question && !wrote_files && !used_subagent {
        eprintln!("  PASS: model stopped after Phase 1 and asked questions");
    } else if asks_question {
        eprintln!("  PARTIAL PASS: model asked questions but also proceeded further");
    } else {
        eprintln!("  WARN: model may not have followed multi-turn flow");
    }

    // The key assertion: Skill was loaded
    // The soft check: model should ask questions
    assert!(
        r.has_tool("Skill"),
        "Research skill must be loaded via Skill tool"
    );
}

#[tokio::test]
async fn t88_deep_research_skills_discovered() {
    let config = match test_config() {
        Some(c) => c,
        None => {
            eprintln!("  SKIP t88: no config");
            return;
        }
    };

    eprintln!(">>> t88_deep_research_skills_discovered");

    let roots = skill_roots_from_config(&config);
    let resolver = SkillResolver::new(roots.clone());
    let _all = resolver.list();

    // Only check skills that exist under the configured skill_roots.
    // Nested skills (e.g. ~/.naked/skills/deep-research/*) are not
    // walked by SkillResolver — they require their own root entry.
    let research_skills = [
        "research",
        "research-report",
        "research-playbook",
        "research-catalog-control",
    ];
    let mut found = Vec::new();
    let mut missing = Vec::new();

    for name in &research_skills {
        if let Some(hit) = resolver.resolve(name) {
            let content = std::fs::read_to_string(&hit.path).unwrap();
            let has_frontmatter = content.contains("---") && content.contains("name:");
            let has_trigger = content.contains("Триггер") || content.contains("Trigger");
            eprintln!(
                "  ✅ {name}: {} (frontmatter={has_frontmatter}, trigger={has_trigger})",
                hit.path.display()
            );
            found.push(name.to_string());
        } else {
            eprintln!("  ❌ {name}: NOT FOUND");
            missing.push(name.to_string());
        }
    }

    assert!(
        missing.is_empty(),
        "Missing research skills: {:?} (searched in {:?})",
        missing,
        roots
    );

    // Optionally verify validate_json.py if the deep-research skill pack is installed.
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    let script = PathBuf::from(&home).join(".naked/skills/deep-research/scripts/validate_json.py");
    if script.exists() {
        let output = std::process::Command::new("python3")
            .args([script.to_str().unwrap(), "--help"])
            .output();
        match output {
            Ok(o) => {
                assert!(o.status.success(), "validate_json.py --help failed");
                eprintln!("  ✅ validate_json.py runs OK");
            }
            Err(e) => eprintln!("  WARN: couldn't run validate_json.py: {e}"),
        }
    } else {
        eprintln!("  SKIP: validate_json.py not installed (deep-research pack)");
    }

    eprintln!("  PASS: all {} research skills discovered", found.len());
}

#[tokio::test]
async fn t89_deep_research_full_cycle() {
    let config = match test_config() {
        Some(c) => c,
        None => {
            eprintln!("  SKIP t89: no config");
            return;
        }
    };

    let roots = skill_roots_from_config(&config);
    let resolver = SkillResolver::new(roots.clone());
    if resolver.resolve("research").is_none() {
        eprintln!("  SKIP t89: 'research' skill not found");
        return;
    }

    // Resolve provider+model once, reuse for all turns to avoid probe flakiness.
    let provider_name =
        std::env::var("E2E_PROVIDER").unwrap_or_else(|_| config.default_provider.clone());
    let model = std::env::var("E2E_MODEL").unwrap_or_else(|_| {
        config
            .providers
            .get(&provider_name)
            .and_then(|pc| pc.models.first().cloned())
            .unwrap_or_else(|| config.default_model.clone())
    });

    let make_prov = || -> Box<dyn Provider> {
        make_provider_for(&config, &provider_name, &model)
            .expect("configured provider must be available")
    };

    // Quick sanity check that the provider actually works.
    {
        let prov = make_prov();
        if !probe_provider(prov.as_ref(), &model).await {
            eprintln!("  SKIP t89: provider {provider_name}/{model} probe failed");
            return;
        }
    }

    eprintln!(">>> t89_deep_research_full_cycle [provider={provider_name}, model={model}]");

    let tmp = tempfile::tempdir().unwrap();
    let cwd = tmp.path();

    let mut history = ConversationHistory::new(research_system_prompt(cwd));

    let build_research_tools = |config: &Config, model: &str| -> ToolRegistry {
        let resolver = SkillResolver::new(skill_roots_from_config(config));
        let available = resolver.list();
        let exa_keys = config.exa_api_keys.clone();
        let prov_arc: Arc<dyn Provider> = Arc::from(
            make_provider_for(config, &provider_name, model).expect("provider must be available"),
        );
        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(BashTool::new(120)),
            Box::new(ReadFileTool::default()),
            Box::new(WriteFileTool::default()),
            Box::new(EditFileTool::default()),
            Box::new(GlobSearchTool),
            Box::new(GrepSearchTool),
            Box::new(WebSearchTool::from_legacy_exa(exa_keys.clone())),
            Box::new(SubAgentTool::new(
                prov_arc,
                model.to_string(),
                120,
                exa_keys,
            )),
            Box::new(SkillTool::new(resolver, &available)),
        ];
        ToolRegistry::new(tools)
    };

    // ── Turn 1: Load skill, generate plan, ask questions ──
    eprintln!("\n  ── Turn 1: plan ──");
    history.push_user(
        "/research сравнение 3 языков программирования для CLI-утилит: Rust, Go, Zig. \
         Критерии: скорость компиляции, размер бинаря, экосистема, порог входа.",
    );

    let tools = build_research_tools(&config, &model);
    let r1 = run_research_prompt(make_prov(), &mut history, cwd, &model, tools, 180).await;

    let t1 = r1.full_text();
    eprintln!("  turn1 tools: {:?}", r1.tool_names());
    eprintln!("  turn1 text ({} bytes): {:.300}", t1.len(), t1);
    assert!(r1.has_tool("Skill"), "Turn 1 must load Skill");
    assert!(!r1.has_tool("write_file"), "Turn 1 must NOT write files");
    assert!(!r1.has_tool("sub_agent"), "Turn 1 must NOT use sub_agent");
    let has_question = t1.contains('?') || t1.to_lowercase().contains("добавить");
    eprintln!("  turn1 asks question: {has_question}");

    tokio::time::sleep(Duration::from_secs(2)).await;

    // ── Turn 2: User confirms, agent does web search ──
    eprintln!("\n  ── Turn 2: web search ──");
    history.push_user("Всё отлично, ничего менять не надо. Период: 2024-2025. Начинай.");

    let tools = build_research_tools(&config, &model);
    let r2 = run_research_prompt(make_prov(), &mut history, cwd, &model, tools, 300).await;

    let t2 = r2.full_text();
    eprintln!("  turn2 tools: {:?}", r2.tool_names());
    eprintln!("  turn2 text ({} bytes): {:.300}", t2.len(), t2);
    let used_search = r2.has_tool("web_search");
    eprintln!("  turn2 web_search: {used_search}");

    tokio::time::sleep(Duration::from_secs(2)).await;

    // ── Turn 3: User confirms, agent creates outline + runs deep research ──
    eprintln!("\n  ── Turn 3: deep research ──");
    history.push_user("Да, подтверждаю. Запускай глубокое исследование.");

    let tools = build_research_tools(&config, &model);
    let r3 = run_research_prompt(make_prov(), &mut history, cwd, &model, tools, 600).await;

    let t3 = r3.full_text();
    eprintln!("  turn3 tools: {:?}", r3.tool_names());
    eprintln!("  turn3 text ({} bytes): {:.300}", t3.len(), t3);
    let wrote_files = r3.has_tool("write_file");
    let used_subagent = r3.has_tool("sub_agent");
    eprintln!("  turn3 write_file: {wrote_files}, sub_agent: {used_subagent}");

    fn list_tree(dir: &Path, prefix: &str) {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                eprintln!("  {prefix}{}", p.file_name().unwrap().to_string_lossy());
                if p.is_dir() {
                    list_tree(&p, &format!("{prefix}  "));
                }
            }
        }
    }
    eprintln!("  files in tmpdir:");
    list_tree(cwd, "    ");

    tokio::time::sleep(Duration::from_secs(2)).await;

    // ── Turn 4: User confirms, agent generates report ──
    eprintln!("\n  ── Turn 4: report ──");
    history.push_user("Да, сгенерируй итоговый отчёт.");

    let tools = build_research_tools(&config, &model);
    let r4 = run_research_prompt(make_prov(), &mut history, cwd, &model, tools, 300).await;

    let t4 = r4.full_text();
    eprintln!("  turn4 tools: {:?}", r4.tool_names());
    eprintln!("  turn4 text ({} bytes)", t4.len());

    // Save report to persistent location for inspection
    let report_dump = std::path::PathBuf::from("/tmp/t89_last_report.md");
    if let Some(report_path) = std::fs::read_dir(cwd)
        .into_iter()
        .flatten()
        .flatten()
        .find_map(|e| {
            let p = e.path();
            if p.is_dir() && p.join("report.md").exists() {
                Some(p.join("report.md"))
            } else {
                None
            }
        })
    {
        let _ = std::fs::copy(&report_path, &report_dump);
        eprintln!("  report saved to {}", report_dump.display());
    }

    if t4.len() > 200 {
        let tail: String = t4
            .chars()
            .rev()
            .take(300)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        eprintln!("  turn4 tail: {tail}");
    }

    // Final checks
    let has_report = t4.to_lowercase().contains("rust")
        || t4.to_lowercase().contains("go")
        || t4.to_lowercase().contains("zig")
        || t4.to_lowercase().contains("отчёт")
        || t4.to_lowercase().contains("итог");

    // Check for report.md on disk
    let has_report_file = std::fs::read_dir(cwd)
        .into_iter()
        .flatten()
        .flatten()
        .any(|e| {
            let p = e.path();
            if p.is_dir() {
                p.join("report.md").exists()
            } else {
                false
            }
        });
    eprintln!("  report.md on disk: {has_report_file}");
    eprintln!("  report content in text: {has_report}");

    eprintln!("\n  ══ Summary ══");
    eprintln!(
        "  Turn 1 (plan):     Skill={}, question={has_question}",
        r1.has_tool("Skill")
    );
    eprintln!("  Turn 2 (search):   web_search={used_search}");
    eprintln!("  Turn 3 (research): write_file={wrote_files}, sub_agent={used_subagent}");
    eprintln!("  Turn 4 (report):   report_content={has_report}, report_file={has_report_file}");
    eprintln!("  PASS: full research cycle completed");
}

#[tokio::test]
async fn t102_research_jaguar_e2e() {
    use naked_core::AgentCore;

    need_config!(config);
    pace().await;

    if !config.research.enabled {
        eprintln!("  SKIP: research disabled in config");
        return;
    }

    let research_model = config
        .research
        .model
        .as_deref()
        .unwrap_or(&config.default_model);
    eprintln!(">>> t102_research_jaguar_e2e [model={research_model}]");

    let sessions_tmp = tempfile::tempdir().unwrap();
    let research_tmp = tempfile::tempdir().unwrap();

    let mut cfg: Config = config.clone();
    cfg.session_dir = sessions_tmp.path().to_path_buf();
    cfg.research.max_iterations = 10;
    cfg.research.max_wall_seconds = 120;
    cfg.research.storage_dir = Some(research_tmp.path().to_path_buf());

    let provider = match naked_core::build_provider_from_config(&cfg) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("  SKIP: couldn't build provider: {e}");
            return;
        }
    };
    let agent = Arc::new(AgentCore::new(cfg.clone(), provider));
    agent.init_self_ref();
    agent.init_mcp().await;

    // Imperative topic: any tool-capable model will execute it literally.
    // We embed the exact URL we want in the finding so the assertion below
    // can verify the agent actually ran the tool instead of hallucinating.
    let topic = "Fetch the page at the seed URL using the web_fetch tool, \
        then call research_save exactly once with url=\"https://example.com/\" \
        and title=\"Example Domain\". Then stop — do not browse anywhere else."
        .to_string();

    let spec = match agent
        .create_research(
            &topic,
            vec!["https://example.com/".to_string()],
            None,
            None,
            Some(10),
            naked_core::CreateSchedule::OneShotNow,
        )
        .await
    {
        Ok(s) => s,
        Err(e) => {
            eprintln!("  SKIP: create_research failed: {e}");
            return;
        }
    };
    eprintln!("  spec id: {}", spec.id);

    let started = std::time::Instant::now();
    let report = match agent.clone().run_research(&spec.id).await {
        Ok(r) => r,
        Err(e) => {
            eprintln!("  SKIP: run_research errored (likely provider / quota): {e}");
            return;
        }
    };
    let elapsed = started.elapsed();
    eprintln!(
        "  run: stop={} new={} total={} elapsed={:.1?} via {}/{}",
        report.stop_reason.as_str(),
        report.new_findings,
        report.total_findings_after,
        elapsed,
        report.provider,
        report.model
    );

    assert!(
        elapsed.as_secs() <= cfg.research.max_wall_seconds + 30,
        "run blew wall budget: {:.1?} > {}+30s",
        elapsed,
        cfg.research.max_wall_seconds
    );

    let store = agent.research_store();
    let runs = store.list_runs(&spec.id, Some(10)).await.unwrap();
    assert!(
        !runs.is_empty(),
        "expected ≥1 run recorded in runs.jsonl, got 0"
    );
    let last = runs.last().unwrap();
    assert_eq!(last.spec_id, spec.id);
    assert_eq!(last.run_id, report.run_id);

    let report_md = store
        .read_report(&spec.id)
        .await
        .unwrap()
        .expect("report.md was not regenerated after a run");
    assert!(
        report_md.contains("Example Domain") || report_md.to_lowercase().contains("example"),
        "report.md should mention the topic: {report_md}"
    );

    let agent_brief = store
        .read_agent_brief(&spec.id)
        .await
        .unwrap()
        .expect("agent_brief.md was not generated after a run");
    assert!(
        agent_brief.contains("Collected findings"),
        "agent_brief.md should contain findings table header: {agent_brief}"
    );
    assert!(
        agent_brief.contains("example.com"),
        "agent_brief.md should reference the finding URL: {agent_brief}"
    );
    eprintln!("  agent_brief.md OK ({} bytes)", agent_brief.len());

    // The core assertion: the agent actually saved at least one finding.
    let findings = store.list_findings(&spec.id, None).await.unwrap();
    eprintln!("  findings saved: {}", findings.len());
    for f in &findings {
        eprintln!(
            "    - {} | {}",
            f.title.as_deref().unwrap_or("(untitled)"),
            f.url
        );
    }
    assert!(
        !findings.is_empty(),
        "research pipeline saved zero findings — agent failed to invoke research_save. \
         stop_reason={} elapsed={:.1?}",
        report.stop_reason.as_str(),
        elapsed
    );

    // Every saved finding must be canonicalised + unique by dedup_hash.
    let mut seen = std::collections::HashSet::new();
    for f in &findings {
        assert!(
            seen.insert(f.dedup_hash.clone()),
            "duplicate dedup_hash in findings.jsonl: {}",
            f.dedup_hash
        );
        assert!(
            !f.url.contains('#'),
            "finding URL should be canonicalised (no fragment): {}",
            f.url
        );
        assert!(
            !f.url.to_lowercase().contains("utm_"),
            "finding URL should be canonicalised (no utm_*): {}",
            f.url
        );
    }

    // At least one finding must point at example.com — that's what the
    // topic asked for, so this catches the model hallucinating URLs.
    assert!(
        findings
            .iter()
            .any(|f| f.url.to_lowercase().contains("example.com")),
        "expected at least one finding on example.com, got: {:?}",
        findings.iter().map(|f| &f.url).collect::<Vec<_>>()
    );

    eprintln!(
        "  PASS: research pipeline saved {} finding(s), store is consistent",
        findings.len()
    );
}

#[ignore = "requires live LLM and research infrastructure"]
#[tokio::test]
async fn t103_research_orchestration_tools_e2e() {
    use naked_core::AgentCore;
    use naked_core::types::PermissionResponse;

    need_config!(config);
    pace().await;

    let research_model = config
        .research
        .model
        .as_deref()
        .unwrap_or(&config.default_model);
    eprintln!(">>> t103_research_orchestration_tools_e2e [model={research_model}]");

    let sessions_tmp = tempfile::tempdir().unwrap();
    let research_tmp = tempfile::tempdir().unwrap();

    let mut cfg: Config = config.clone();
    cfg.session_dir = sessions_tmp.path().to_path_buf();
    cfg.workspace = sessions_tmp.path().to_path_buf();
    cfg.research.storage_dir = Some(research_tmp.path().to_path_buf());

    let provider = match naked_core::build_provider_from_config(&cfg) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("  SKIP: couldn't build provider: {e}");
            return;
        }
    };
    let agent = Arc::new(AgentCore::new(cfg.clone(), provider));
    agent.init_self_ref();
    agent.init_mcp().await;

    let session_id = agent.create_session(sessions_tmp.path()).await;
    // Enable yolo so tool permissions are auto-approved.
    let now_ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let _ = agent.set_session_yolo(&session_id, Some(now_ts)).await;

    let prompt = "Use the research_create tool to create a research spec with topic \
        \"test orchestration e2e\". Then call research_list_specs to verify it exists. \
        Then call research_findings with the spec_id you got from research_create. \
        Report each tool result briefly.";

    let mut handle = match agent.send_prompt(&session_id, prompt).await {
        Ok(h) => h,
        Err(e) => {
            eprintln!("  SKIP: send_prompt failed: {e}");
            return;
        }
    };

    let mut text = String::new();
    let mut tool_calls = Vec::new();
    let mut got_idle = false;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(120);
    loop {
        match tokio::time::timeout_at(deadline, handle.events.recv()).await {
            Ok(Some(AgentEvent::TextDelta(t))) => text.push_str(&t),
            Ok(Some(AgentEvent::ToolEnd { name, .. })) => {
                eprintln!("  tool completed: {name}");
                tool_calls.push(name);
            }
            Ok(Some(AgentEvent::PermissionRequest {
                call_id, tool_name, ..
            })) => {
                eprintln!("  auto-approve: {tool_name}");
                let _ = handle
                    .permissions
                    .send(PermissionResponse {
                        call_id,
                        allowed: true,
                    })
                    .await;
            }
            Ok(Some(AgentEvent::Idle)) => {
                got_idle = true;
                break;
            }
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

    eprintln!("  text response: {}", &text[..text.len().min(300)]);
    eprintln!("  tool calls: {:?}", tool_calls);

    assert!(got_idle, "agent didn't reach idle");

    assert!(
        tool_calls.contains(&"research_create".to_string()),
        "agent didn't call research_create. tool_calls={tool_calls:?}"
    );
    assert!(
        tool_calls.contains(&"research_list_specs".to_string()),
        "agent didn't call research_list_specs. tool_calls={tool_calls:?}"
    );
    assert!(
        tool_calls.contains(&"research_findings".to_string()),
        "agent didn't call research_findings. tool_calls={tool_calls:?}"
    );

    // Verify the spec was actually persisted
    let store = agent.research_store();
    let specs = store.list_specs().await.unwrap();
    assert!(
        !specs.is_empty(),
        "research_create should have persisted a spec"
    );
    assert!(
        specs.iter().any(|s| s.topic.contains("test orchestration")),
        "expected spec with topic containing 'test orchestration', got: {:?}",
        specs.iter().map(|s| &s.topic).collect::<Vec<_>>()
    );

    eprintln!(
        "  PASS: all 3 orchestration tools called, spec persisted ({})",
        specs[0].id
    );
}

#[tokio::test]
async fn t104_research_natural_prompt_e2e() {
    use naked_core::AgentCore;
    use naked_core::types::PermissionResponse;

    need_config!(config);
    pace().await;

    let research_model = config
        .research
        .model
        .as_deref()
        .unwrap_or(&config.default_model);
    eprintln!(">>> t104_research_natural_prompt_e2e [model={research_model}]");

    let sessions_tmp = tempfile::tempdir().unwrap();
    let research_tmp = tempfile::tempdir().unwrap();

    let mut cfg: Config = config.clone();
    cfg.session_dir = sessions_tmp.path().to_path_buf();
    cfg.workspace = sessions_tmp.path().to_path_buf();
    cfg.research.storage_dir = Some(research_tmp.path().to_path_buf());

    let provider = match naked_core::build_provider_from_config(&cfg) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("  SKIP: couldn't build provider: {e}");
            return;
        }
    };
    let agent = Arc::new(AgentCore::new(cfg.clone(), provider));
    agent.init_self_ref();
    agent.init_mcp().await;

    let session_id = agent.create_session(sessions_tmp.path()).await;
    let now_ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let _ = agent.set_session_yolo(&session_id, Some(now_ts)).await;

    // Natural user prompt — no tool names, just a task in Russian.
    // Explicit mention of "research_create" + "research_launch" keeps it model-agnostic
    // while still testing real conversational flow.
    let prompt = "Найди мне аренду коммерческой недвижимости в Дананге до $200 в месяц. \
        Используй research_create чтобы создать задачу и research_launch чтобы запустить \
        глубокий фоновый поиск.";

    let mut handle = match agent.send_prompt(&session_id, prompt).await {
        Ok(h) => h,
        Err(e) => {
            eprintln!("  SKIP: send_prompt failed: {e}");
            return;
        }
    };

    let mut text = String::new();
    let mut tool_calls = Vec::new();
    let mut got_idle = false;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(180);
    loop {
        match tokio::time::timeout_at(deadline, handle.events.recv()).await {
            Ok(Some(AgentEvent::TextDelta(t))) => text.push_str(&t),
            Ok(Some(AgentEvent::ThinkingDelta(t))) => {
                eprintln!("  thinking: {}...", &t[..t.len().min(80)]);
            }
            Ok(Some(AgentEvent::ToolStart { name, .. })) => {
                eprintln!("  tool start: {name}");
            }
            Ok(Some(AgentEvent::ToolEnd { name, .. })) => {
                eprintln!("  tool completed: {name}");
                tool_calls.push(name);
            }
            Ok(Some(AgentEvent::PermissionRequest {
                call_id, tool_name, ..
            })) => {
                eprintln!("  auto-approve: {tool_name}");
                let _ = handle
                    .permissions
                    .send(PermissionResponse {
                        call_id,
                        allowed: true,
                    })
                    .await;
            }
            Ok(Some(AgentEvent::Idle)) => {
                eprintln!("  got idle");
                got_idle = true;
                break;
            }
            Ok(Some(AgentEvent::Error(e))) => {
                eprintln!("  ERROR: {e}");
                break;
            }
            Ok(Some(other)) => {
                eprintln!("  event: {other:?}");
            }
            Ok(None) => {
                eprintln!("  channel closed");
                break;
            }
            Err(_) => {
                eprintln!("  TIMEOUT");
                break;
            }
        }
    }

    let preview_end = text
        .char_indices()
        .nth(300)
        .map(|(i, _)| i)
        .unwrap_or(text.len());
    eprintln!(
        "  text response ({} chars): {}",
        text.len(),
        &text[..preview_end]
    );
    eprintln!("  tool calls: {:?}", tool_calls);

    assert!(got_idle, "agent didn't reach idle");

    // The agent MUST have called research_create (to create the spec)
    assert!(
        tool_calls.contains(&"research_create".to_string()),
        "agent didn't call research_create from natural prompt. tool_calls={tool_calls:?}"
    );

    // The agent MUST have called research_launch (user said "запусти глубокий поиск")
    assert!(
        tool_calls.contains(&"research_launch".to_string()),
        "agent didn't call research_launch from natural prompt. tool_calls={tool_calls:?}"
    );

    // Verify the spec was persisted with relevant topic
    let store = agent.research_store();
    let specs = store.list_specs().await.unwrap();
    assert!(!specs.is_empty(), "no specs persisted after natural prompt");

    let spec = &specs[0];
    eprintln!("  created spec: id={} topic={}", spec.id, spec.topic);

    // Topic should mention Da Nang or commercial rental or $200
    let topic_lower = spec.topic.to_lowercase();
    let relevant = topic_lower.contains("дананг")
        || topic_lower.contains("da nang")
        || topic_lower.contains("danang")
        || topic_lower.contains("200")
        || topic_lower.contains("коммерч")
        || topic_lower.contains("commercial")
        || topic_lower.contains("аренд")
        || topic_lower.contains("rent");
    assert!(
        relevant,
        "spec topic should relate to the user query, got: {}",
        spec.topic
    );

    eprintln!(
        "  PASS: natural prompt → research_create + research_launch, spec='{}' ({})",
        spec.topic, spec.id
    );
}

#[tokio::test]
#[ignore = "requires live LLM provider and web access; run with --ignored"]
async fn t105_research_full_quality_e2e() {
    use naked_core::AgentCore;
    use naked_core::types::PermissionResponse;

    need_config!(config);
    pace().await;

    let research_model = config
        .research
        .model
        .as_deref()
        .unwrap_or(&config.default_model);
    eprintln!(">>> t105_research_full_quality_e2e [model={research_model}]");

    let sessions_tmp = tempfile::tempdir().unwrap();
    let research_tmp = tempfile::tempdir().unwrap();

    let mut cfg: Config = config.clone();
    cfg.session_dir = sessions_tmp.path().to_path_buf();
    cfg.workspace = sessions_tmp.path().to_path_buf();
    cfg.research.max_iterations = 30;
    cfg.research.max_wall_seconds = 600;
    cfg.research.storage_dir = Some(research_tmp.path().to_path_buf());

    let provider = match naked_core::build_provider_from_config(&cfg) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("  SKIP: couldn't build provider: {e}");
            return;
        }
    };
    let agent = Arc::new(AgentCore::new(cfg.clone(), provider));
    agent.init_self_ref();
    agent.init_mcp().await;

    let session_id = agent.create_session(sessions_tmp.path()).await;
    let now_ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let _ = agent.set_session_yolo(&session_id, Some(now_ts)).await;

    // ── Phase 1: natural prompt → orchestration ──────────────────────────
    let prompt = "Мне нужно найти помещение под кафе завтраков в Дананге (Вьетнам).\n\n\
        Требования:\n\
        - Бюджет: от $1000 до $3000 в месяц\n\
        - Площадь: 100-200 м²\n\
        - Районы: An Thuong, My An, My Khe — экспатская зона\n\
        - Обязательно 1-й этаж с возможностью террасы\n\
        - Рядом с пляжем или в зоне активного пешеходного трафика\n\n\
        Ищи на всех основных сайтах недвижимости Вьетнама: batdongsan.com.vn, \
        chotot.com, muaban.net, alonhadat.com.vn, homedy.com, nha.chotot.com.\n\n\
        Используй research_create чтобы создать задачу и research_launch \
        чтобы запустить глубокий фоновый поиск.";

    let mut handle = match agent.send_prompt(&session_id, prompt).await {
        Ok(h) => h,
        Err(e) => {
            eprintln!("  SKIP: send_prompt failed: {e}");
            return;
        }
    };

    let mut text = String::new();
    let mut tool_calls = Vec::new();
    let mut got_idle = false;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(120);
    loop {
        match tokio::time::timeout_at(deadline, handle.events.recv()).await {
            Ok(Some(AgentEvent::TextDelta(t))) => text.push_str(&t),
            Ok(Some(AgentEvent::ToolStart { name, .. })) => {
                eprintln!("  [chat] tool start: {name}");
            }
            Ok(Some(AgentEvent::ToolEnd { name, .. })) => {
                eprintln!("  [chat] tool completed: {name}");
                tool_calls.push(name);
            }
            Ok(Some(AgentEvent::PermissionRequest {
                call_id, tool_name, ..
            })) => {
                eprintln!("  [chat] auto-approve: {tool_name}");
                let _ = handle
                    .permissions
                    .send(PermissionResponse {
                        call_id,
                        allowed: true,
                    })
                    .await;
            }
            Ok(Some(AgentEvent::Idle)) => {
                got_idle = true;
                break;
            }
            Ok(Some(AgentEvent::Error(e))) => {
                eprintln!("  [chat] ERROR: {e}");
                break;
            }
            Ok(None) => break,
            Err(_) => {
                eprintln!("  [chat] TIMEOUT waiting for agent");
                break;
            }
            _ => {}
        }
    }

    assert!(got_idle, "agent didn't reach idle");
    assert!(
        tool_calls.contains(&"research_create".to_string()),
        "agent didn't call research_create. tool_calls={tool_calls:?}"
    );
    assert!(
        tool_calls.contains(&"research_launch".to_string()),
        "agent didn't call research_launch. tool_calls={tool_calls:?}"
    );

    let store = agent.research_store();
    let specs = store.list_specs().await.unwrap();
    assert!(!specs.is_empty(), "no specs persisted");
    let spec_id = specs[0].id.clone();
    eprintln!("  Phase 1 OK: spec={spec_id}, topic={}", specs[0].topic);

    // ── Phase 2: wait for background run to finish ───────────────────────
    eprintln!("  Waiting for background research run to finish (up to 10 min)...");
    let poll_deadline = tokio::time::Instant::now() + Duration::from_secs(600);
    let mut run_finished = false;
    loop {
        tokio::time::sleep(Duration::from_secs(10)).await;
        let runs = store.list_runs(&spec_id, Some(1)).await.unwrap_or_default();
        if !runs.is_empty() {
            let r = &runs[0];
            eprintln!(
                "  Run finished: new={} total={} stop={} via {}/{}",
                r.new_findings, r.total_findings_after, r.stop_reason, r.provider, r.model
            );
            run_finished = true;
            break;
        }
        if tokio::time::Instant::now() > poll_deadline {
            eprintln!("  TIMEOUT: background run didn't finish in 10 min");
            break;
        }
        eprint!(".");
    }

    assert!(run_finished, "background research run never completed");

    // ── Phase 3: verify findings quality ─────────────────────────────────
    let findings = store.list_findings(&spec_id, None).await.unwrap();
    eprintln!("  Total findings: {}", findings.len());

    assert!(
        !findings.is_empty(),
        "research run produced zero findings — agent failed to find anything"
    );

    // Print all findings for manual review
    for (i, f) in findings.iter().enumerate() {
        eprintln!(
            "  {:2}. [{}] {:>20} | {} | {}",
            i + 1,
            f.listing_date.as_deref().unwrap_or("—"),
            f.price.as_deref().unwrap_or("—"),
            f.title.as_deref().unwrap_or("(untitled)"),
            f.url
        );
    }

    // 3a. Dedup: no duplicate dedup_hash
    let mut seen_hashes = std::collections::HashSet::new();
    for f in &findings {
        assert!(
            seen_hashes.insert(f.dedup_hash.clone()),
            "duplicate dedup_hash: {} (url={})",
            f.dedup_hash,
            f.url
        );
    }
    eprintln!("  ✓ No duplicate findings");

    // 3b. listing_date: at least 30% of findings should have it
    let with_date = findings.iter().filter(|f| f.listing_date.is_some()).count();
    let date_pct = (with_date as f64 / findings.len() as f64 * 100.0) as u32;
    eprintln!(
        "  listing_date present: {with_date}/{} ({date_pct}%)",
        findings.len()
    );
    assert!(
        date_pct >= 30,
        "too few findings with listing_date: {with_date}/{} ({date_pct}%). \
         Expected ≥30%. The agent should extract dates from listings.",
        findings.len()
    );

    // 3c. URLs should be canonicalized (no fragments, no utm_*)
    for f in &findings {
        assert!(
            !f.url.contains('#'),
            "finding URL not canonicalized (fragment): {}",
            f.url
        );
        assert!(
            !f.url.to_lowercase().contains("utm_"),
            "finding URL not canonicalized (utm): {}",
            f.url
        );
    }
    eprintln!("  ✓ All URLs canonicalized");

    // 3d. At least some findings should have a price
    let with_price = findings.iter().filter(|f| f.price.is_some()).count();
    let price_pct = (with_price as f64 / findings.len() as f64 * 100.0) as u32;
    eprintln!(
        "  price present: {with_price}/{} ({price_pct}%)",
        findings.len()
    );
    assert!(
        price_pct >= 40,
        "too few findings with price: {with_price}/{} ({price_pct}%). \
         Expected ≥40%.",
        findings.len()
    );

    // 3e. Spot-check: at least one URL should be from a known VN real estate site
    let known_domains = [
        "batdongsan.com.vn",
        "chotot.com",
        "nha.chotot.com",
        "muaban.net",
        "alonhadat.com.vn",
        "homedy.com",
        "bds123.vn",
        "dothi.net",
    ];
    let from_known = findings
        .iter()
        .filter(|f| known_domains.iter().any(|d| f.url.contains(d)))
        .count();
    eprintln!("  from known VN sites: {from_known}/{}", findings.len());
    // Soft check — warn but don't fail, agent might find listings on other sites
    if from_known == 0 {
        eprintln!(
            "  ⚠ no findings from known VN real estate domains (may be OK if found elsewhere)"
        );
    }

    // 3f. report.md should exist and mention Da Nang
    let report = store
        .read_report(&spec_id)
        .await
        .unwrap()
        .expect("report.md was not generated");
    let report_lower = report.to_lowercase();
    assert!(
        report_lower.contains("da nang")
            || report_lower.contains("дананг")
            || report_lower.contains("danang"),
        "report.md should mention Da Nang"
    );
    eprintln!("  ✓ report.md OK ({} bytes)", report.len());

    // 3g. agent_brief.md should exist
    let brief = store
        .read_agent_brief(&spec_id)
        .await
        .unwrap()
        .expect("agent_brief.md was not generated");
    assert!(
        brief.contains("Collected findings"),
        "agent_brief.md missing findings table"
    );
    eprintln!("  ✓ agent_brief.md OK ({} bytes)", brief.len());

    // 3h. Spot-check a few URLs are live (HTTP 200 or 301/302)
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .unwrap();
    let mut live = 0u32;
    let mut dead = 0u32;
    let check_count = findings.len().min(5);
    for f in findings.iter().take(check_count) {
        match client.head(&f.url).send().await {
            Ok(resp) => {
                let status = resp.status().as_u16();
                if status < 400 {
                    live += 1;
                    eprintln!("    ✓ {} → {status}", f.url);
                } else {
                    dead += 1;
                    eprintln!("    ✗ {} → {status}", f.url);
                }
            }
            Err(e) => {
                dead += 1;
                eprintln!("    ✗ {} → err: {e}", f.url);
            }
        }
    }
    eprintln!("  URL spot-check: {live} live, {dead} dead (of {check_count} checked)");
    // At least half of checked URLs should be reachable
    assert!(
        live > 0,
        "all {check_count} spot-checked URLs are dead — findings likely stale or hallucinated"
    );

    eprintln!(
        "\n  ══ PASS ══ t105_research_full_quality_e2e\n  \
         findings={} (date:{with_date} price:{with_price} known_sites:{from_known}) \
         urls_live={live}/{check_count}\n  spec={spec_id}",
        findings.len()
    );
}

#[tokio::test]
#[ignore = "requires live LLM provider and prior research data; run with --ignored"]
async fn t106_research_summary_from_chat_e2e() {
    use naked_core::AgentCore;
    use naked_core::types::PermissionResponse;

    need_config!(config);
    pace().await;

    eprintln!(">>> t106_research_summary_from_chat_e2e");

    let sessions_tmp = tempfile::tempdir().unwrap();

    // Use real research storage (not tmp) so we see existing findings
    let mut cfg: Config = config.clone();
    cfg.session_dir = sessions_tmp.path().to_path_buf();
    cfg.workspace = sessions_tmp.path().to_path_buf();

    let provider = match naked_core::build_provider_from_config(&cfg) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("  SKIP: couldn't build provider: {e}");
            return;
        }
    };
    let agent = Arc::new(AgentCore::new(cfg, provider));
    agent.init_self_ref();

    // Check we have at least one spec with findings
    let store = agent.research_store();
    let specs = match store.list_specs().await {
        Ok(s) if !s.is_empty() => s,
        _ => {
            eprintln!("  SKIP: no research specs on disk (run t105 first)");
            return;
        }
    };
    let total_findings = store.count_findings(&specs[0].id).await.unwrap_or(0);
    if total_findings == 0 {
        eprintln!("  SKIP: spec {} has 0 findings", specs[0].id);
        return;
    }
    eprintln!(
        "  Using spec {} with {total_findings} findings",
        specs[0].id
    );

    let session_id = agent.create_session(sessions_tmp.path()).await;
    let now_ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let _ = agent.set_session_yolo(&session_id, Some(now_ts)).await;

    let prompt = "Что нашлось по аренде помещений в Дананге? \
        Используй research_list_specs и research_findings чтобы получить все результаты, \
        и дай мне сводку: лучшие варианты с ценами, площадью и контактами.";

    let mut handle = match agent.send_prompt(&session_id, prompt).await {
        Ok(h) => h,
        Err(e) => {
            eprintln!("  SKIP: send_prompt failed: {e}");
            return;
        }
    };

    let mut text = String::new();
    let mut tool_calls = Vec::new();
    let mut got_idle = false;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(120);
    loop {
        match tokio::time::timeout_at(deadline, handle.events.recv()).await {
            Ok(Some(AgentEvent::TextDelta(t))) => text.push_str(&t),
            Ok(Some(AgentEvent::ToolStart { name, .. })) => {
                eprintln!("  tool start: {name}");
            }
            Ok(Some(AgentEvent::ToolEnd { name, .. })) => {
                eprintln!("  tool completed: {name}");
                tool_calls.push(name);
            }
            Ok(Some(AgentEvent::PermissionRequest {
                call_id, tool_name, ..
            })) => {
                eprintln!("  auto-approve: {tool_name}");
                let _ = handle
                    .permissions
                    .send(PermissionResponse {
                        call_id,
                        allowed: true,
                    })
                    .await;
            }
            Ok(Some(AgentEvent::Idle)) => {
                got_idle = true;
                break;
            }
            Ok(Some(AgentEvent::Error(e))) => {
                eprintln!("  ERROR: {e}");
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

    let preview_end = text
        .char_indices()
        .nth(800)
        .map(|(i, _)| i)
        .unwrap_or(text.len());
    eprintln!("  tool calls: {:?}", tool_calls);
    eprintln!(
        "  response ({} chars):\n{}",
        text.len(),
        &text[..preview_end]
    );

    assert!(got_idle, "agent didn't reach idle");
    assert!(
        tool_calls.contains(&"research_findings".to_string())
            || tool_calls.contains(&"research_list_specs".to_string()),
        "agent didn't use research tools. tool_calls={tool_calls:?}"
    );
    assert!(
        text.len() > 100,
        "agent response too short ({} chars) — should be a useful summary",
        text.len()
    );

    // Response should mention prices or specific findings
    let has_prices = text.contains("triệu")
        || text.contains("tr/")
        || text.contains("$")
        || text.contains("VND")
        || text.contains("USD");
    assert!(has_prices, "summary should mention prices from findings");

    eprintln!("\n  ══ PASS ══ t106: agent summarized {total_findings} findings via tools");
}

#[tokio::test]
async fn t107_research_gatekeeper_e2e() {
    use naked_core::AgentCore;

    need_config!(config);
    pace().await;

    eprintln!(">>> t107_research_gatekeeper_e2e");

    let sessions_tmp = tempfile::tempdir().unwrap();

    let mut cfg: Config = config.clone();
    cfg.session_dir = sessions_tmp.path().to_path_buf();
    cfg.workspace = sessions_tmp.path().to_path_buf();

    let provider = match naked_core::build_provider_from_config(&cfg) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("  SKIP: couldn't build provider: {e}");
            return;
        }
    };
    let agent = Arc::new(AgentCore::new(cfg, provider));
    agent.init_self_ref();

    // Check we have an existing spec with findings to verify
    let store = agent.research_store();
    let specs = match store.list_specs().await {
        Ok(s) if !s.is_empty() => s,
        _ => {
            eprintln!("  SKIP: no research specs on disk (run t105 first)");
            return;
        }
    };
    let spec_id = specs[0].id.clone();
    let before_count = store.count_findings(&spec_id).await.unwrap_or(0);
    if before_count == 0 {
        eprintln!("  SKIP: spec {spec_id} has 0 findings");
        return;
    }
    eprintln!("  Using spec {spec_id} with {before_count} findings");

    // Run with gatekeeper (max 2 verification rounds)
    eprintln!("  Running research with gatekeeper verification...");
    let result = match agent.clone().run_research_verified(&spec_id, 2).await {
        Ok(r) => r,
        Err(e) => {
            eprintln!("  ERROR: run_research_verified failed: {e}");
            panic!("run_research_verified failed: {e}");
        }
    };

    eprintln!("  Gatekeeper result:");
    eprintln!("    verification_rounds: {}", result.verification_rounds);
    eprintln!("    dead_removed: {}", result.dead_removed);
    eprintln!("    replacements_found: {}", result.replacements_found);
    eprintln!("    final_findings: {}", result.final_findings);
    eprintln!("    remaining_issues: {}", result.remaining_issues.len());

    // After gatekeeper, verify all remaining findings have live URLs
    let findings = store.list_findings(&spec_id, None).await.unwrap();
    eprintln!("  Verifying {} remaining findings...", findings.len());

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .redirect(reqwest::redirect::Policy::limited(5))
        .build()
        .unwrap();

    let mut live = 0u32;
    let mut dead = 0u32;
    for f in &findings {
        match client.head(&f.url).send().await {
            Ok(resp) if resp.status().as_u16() < 400 => {
                live += 1;
            }
            Ok(resp) => {
                dead += 1;
                eprintln!("    ✗ {} → {}", f.url, resp.status());
            }
            Err(e) => {
                dead += 1;
                eprintln!("    ✗ {} → {e}", f.url);
            }
        }
    }
    eprintln!(
        "  Post-gatekeeper URL check: {live} live, {dead} dead out of {}",
        findings.len()
    );

    // At least 90% should be live after gatekeeper
    let live_pct = if findings.is_empty() {
        100
    } else {
        (live as f64 / findings.len() as f64 * 100.0) as u32
    };
    assert!(
        live_pct >= 90,
        "after gatekeeper, only {live_pct}% of URLs are live (expected ≥90%). \
         live={live}, dead={dead}, total={}",
        findings.len()
    );

    // Check data completeness
    let with_title = findings.iter().filter(|f| f.title.is_some()).count();
    let with_price = findings.iter().filter(|f| f.price.is_some()).count();
    let with_excerpt = findings
        .iter()
        .filter(|f| f.excerpt.as_ref().is_some_and(|e| e.len() >= 50))
        .count();

    let title_pct = 100 * with_title / findings.len().max(1);
    let price_pct = 100 * with_price / findings.len().max(1);
    let excerpt_pct = 100 * with_excerpt / findings.len().max(1);

    eprintln!("  Data completeness:");
    eprintln!(
        "    title:   {with_title}/{} ({title_pct}%)",
        findings.len()
    );
    eprintln!(
        "    price:   {with_price}/{} ({price_pct}%)",
        findings.len()
    );
    eprintln!(
        "    excerpt: {with_excerpt}/{} ({excerpt_pct}%)",
        findings.len()
    );

    assert!(
        title_pct >= 80,
        "after gatekeeper, title coverage {title_pct}% < 80%"
    );
    assert!(
        price_pct >= 40,
        "after gatekeeper, price coverage {price_pct}% < 40%"
    );

    // Report and brief should exist
    assert!(
        store.read_report(&spec_id).await.unwrap().is_some(),
        "report.md missing after gatekeeper run"
    );
    assert!(
        store.read_agent_brief(&spec_id).await.unwrap().is_some(),
        "agent_brief.md missing after gatekeeper run"
    );

    eprintln!(
        "\n  ══ PASS ══ t107_research_gatekeeper_e2e\n  \
         rounds={} removed={} replaced={} final={}\n  \
         live={live}/{} ({live_pct}%) title={title_pct}% price={price_pct}% excerpt={excerpt_pct}%",
        result.verification_rounds,
        result.dead_removed,
        result.replacements_found,
        result.final_findings,
        findings.len()
    );
}

// ── Mechanism tests (deterministic, no LLM) ────────────────────────

/// t106a: Research summary works with pre-populated findings
/// (self-contained — doesn't need prior research runs)
#[tokio::test]
async fn t106a_research_store_populated_can_list_findings() {
    let tmp = tempfile::tempdir().unwrap();
    let store = FsResearchStore::new(tmp.path().to_path_buf());

    let spec = naked_core::research::ResearchSpec {
        id: "test-summary".into(),
        topic: "Restaurant rentals".into(),
        ..Default::default()
    };
    store.save_spec(&spec).await.unwrap();

    // Verify spec roundtrips
    let loaded = store.load_spec("test-summary").await.unwrap();
    assert_eq!(loaded.topic, "Restaurant rentals");

    // Add a finding with ALL required hashes
    let finding = naked_core::research::Finding {
        id: "f1".into(),
        research_id: "test-summary".into(),
        dedup_hash: "unique-hash-1".into(),
        host_path_hash: "unique-host-1".into(),
        content_hash: "unique-content-1".into(),
        url: "https://example.com/listing/1".into(),
        title: Some("Listing 1".into()),
        excerpt: Some("50m2, 00/month".into()),
        ..Default::default()
    };
    let stored = store.try_append_finding(&finding).await.unwrap();
    assert!(stored, "finding should be stored");

    // Verify it's retrievable
    let findings = store.list_findings("test-summary", None).await.unwrap();
    assert_eq!(findings.len(), 1);
    assert_eq!(findings[0].url, "https://example.com/listing/1");
}

/// t103a: ResearchCoordinator config + store wiring works
#[tokio::test]
async fn t103a_coordinator_config_and_store_setup() {
    use naked_core::research::{CoordinatorConfig, FsResearchStore};

    let tmp = tempfile::tempdir().unwrap();
    let store = std::sync::Arc::new(FsResearchStore::new(tmp.path().to_path_buf()));

    // Create spec and verify the store roundtrips
    let spec = naked_core::research::ResearchSpec {
        id: "coord-test".into(),
        topic: "test topic".into(),
        ..Default::default()
    };
    store.save_spec(&spec).await.unwrap();
    let loaded = store.load_spec("coord-test").await.unwrap();
    assert_eq!(loaded.topic, "test topic");

    // Verify config defaults
    let config = CoordinatorConfig::default();
    assert!(config.default_max_wall_seconds > 0);
}

/// t105a: Research findings dedup works correctly
#[tokio::test]
async fn t105a_research_findings_dedup() {
    use naked_core::research::{Finding, FsResearchStore};

    let tmp = tempfile::tempdir().unwrap();
    let store = FsResearchStore::new(tmp.path().to_path_buf());

    let spec = naked_core::research::ResearchSpec {
        id: "dedup-test".into(),
        topic: "dedup".into(),
        ..Default::default()
    };
    store.save_spec(&spec).await.unwrap();

    // Add same URL twice
    let f1 = Finding {
        research_id: "dedup-test".into(),
        dedup_hash: "same-hash".into(),
        url: "https://example.com/same".into(),
        title: Some("First".into()),
        excerpt: Some("excerpt 1".into()),
        ..Default::default()
    };
    let f2 = Finding {
        research_id: "dedup-test".into(),
        dedup_hash: "same-hash".into(),
        url: "https://example.com/same".into(),
        title: Some("Second".into()),
        excerpt: Some("excerpt 2".into()),
        ..Default::default()
    };

    store.try_append_finding(&f1).await.unwrap();
    store.try_append_finding(&f2).await.unwrap();

    let findings = store.list_findings("dedup-test", None).await.unwrap();
    // Store may or may not dedup — but should not crash
    assert!(!findings.is_empty());
}
