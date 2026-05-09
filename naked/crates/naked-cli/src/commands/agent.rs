//! `naked agent` subcommand — role-based agent execution and batch runs.

use std::sync::Arc;

use anyhow::Result;
use naked_core::AgentCore;
use naked_core::config::Config;

use super::arg_values;

/// `naked research <subcommand>` — operator surface for the research
/// subsystem. Paired with the `/research` Telegram commands and the
/// `naked-research@.timer` systemd unit:
///
/// - `run <id>` — block until one pass finishes; exit non-zero on failure, so
///   systemd OnFailure= hooks fire. This is the entrypoint every timer hits.
/// - `ls` — plain listing, one per line, most-recent-first.
/// - `show <id>` — spec summary + last 10 findings + last 3 runs.
/// - `export <id> <path>` — dump `report.md` to disk (for manual review,
///   email, scp, etc.).
/// - `new <topic>` — create a spec and print the id. `--source URL` may be
///   repeated.
/// - `rm <id>` — delete the spec directory (irrecoverable).
pub(crate) async fn agent_cmd(args: &[String]) -> Result<()> {
    use naked_core::agent_role::Task;
    use naked_core::agent_run::{run_batch, run_task};
    use naked_core::agent_store::resolve_role;
    use naked_core::agent_validator::run_all;

    let config = Config::load()?;
    let provider = naked_core::build_provider_from_config(&config)?;
    let agent = AgentCore::new(config.clone(), provider);
    agent.init_mcp().await;
    let agent = Arc::new(agent);

    let sub = args.first().map(String::as_str).unwrap_or("help");
    match sub {
        "list-roles" | "roles" => {
            let store = agent.agent_store();
            let mut roles = store.list();
            roles.sort_by(|a, b| a.name.cmp(&b.name));
            if roles.is_empty() {
                eprintln!(
                    "(no roles loaded — check `agent_dirs` in naked.json: {:?})",
                    config.agent_dirs,
                );
            }
            for r in roles {
                let model = r.model.as_deref().unwrap_or("(default)");
                println!(
                    "{:<20} model={:<24} skills=[{}]",
                    r.name,
                    model,
                    r.default_skills.join(", "),
                );
                if !r.description.is_empty() {
                    println!("  {}", r.description);
                }
            }
        }
        "run" => {
            // naked agent run <role> <prompt> [--max-wall N] [--context JSON]
            let role_name = args.get(1).ok_or_else(|| {
                anyhow::anyhow!(
                    "usage: naked agent run <role> <prompt> [--max-wall N] [--context JSON]"
                )
            })?;
            let prompt = args
                .get(2)
                .ok_or_else(|| anyhow::anyhow!("usage: naked agent run <role> <prompt>"))?;
            let role = resolve_role(role_name, &agent.agent_store(), &config.agent_roles)
                .ok_or_else(|| {
                    anyhow::anyhow!("unknown role `{role_name}` — try `naked agent list-roles`")
                })?;

            let max_wall = arg_values(args, "--max-wall")
                .first()
                .and_then(|s| s.parse::<u64>().ok());
            let ctx = arg_values(args, "--context")
                .first()
                .and_then(|s| serde_json::from_str(s).ok())
                .unwrap_or(serde_json::Value::Null);

            let mut task = Task::new(role.name.clone(), prompt.clone()).with_context(ctx);
            if let Some(w) = max_wall {
                task = task.with_max_wall(w);
            }

            eprintln!(
                "running role `{}` (model={}, max_wall={})",
                role.name,
                role.model.as_deref().unwrap_or("(default)"),
                task.max_wall_secs
                    .map_or("default".into(), |w| format!("{w}s")),
            );
            let started = std::time::Instant::now();
            let out = run_task(&agent, &role, &task).await?;
            let elapsed = started.elapsed();

            println!("=== task result ===");
            println!("task_id:     {}", out.task_id);
            println!("role:        {}", out.role_name);
            println!("stop_reason: {:?}", out.stop_reason);
            println!("elapsed:     {:.1?}", elapsed);
            println!("text_chars:  {}", out.text.len());
            println!("{}", out.stats.summary_line());
            if !out.text.is_empty() {
                println!("--- text ---");
                println!("{}", out.text);
            }
        }
        "batch" => {
            // naked agent batch <jsonl-file> [--validators phone-vn,no-fake-contact]
            let path = args.get(1).ok_or_else(|| {
                anyhow::anyhow!("usage: naked agent batch <tasks.jsonl> [--validators v1,v2,...]")
            })?;
            let raw = std::fs::read_to_string(path)?;
            let mut tasks: Vec<Task> = Vec::new();
            for (lineno, line) in raw.lines().enumerate() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                let task: Task = serde_json::from_str(line)
                    .map_err(|e| anyhow::anyhow!("{path}:{} parse error: {e}", lineno + 1))?;
                tasks.push(task);
            }
            if tasks.is_empty() {
                return Err(anyhow::anyhow!("no tasks found in {path}"));
            }

            // Plain batch has no per-task criteria source; gatekeeper
            // is only available via `batch-from-spec` where the spec
            // topic supplies the criteria. Plain batch also can't lift
            // role-level defaults uniformly (each line picks its own
            // role) — so the operator stays responsible for picking
            // validators here. Pass `&[]` to skip role-default lookup.
            let validators = parse_validators(args, None, &[]).await?;

            let role_set: std::collections::BTreeSet<String> =
                tasks.iter().map(|t| t.role.clone()).collect();
            let mut roles_vec = Vec::new();
            for name in &role_set {
                let r = resolve_role(name, &agent.agent_store(), &config.agent_roles)
                    .ok_or_else(|| anyhow::anyhow!("unknown role `{name}` in batch"))?;
                roles_vec.push(Arc::new(r));
            }

            eprintln!(
                "running {} task(s) over {} role(s) with {} validator(s)",
                tasks.len(),
                role_set.len(),
                validators.len(),
            );
            let started = std::time::Instant::now();
            let outputs = run_batch(agent.clone(), &roles_vec, tasks).await;
            let elapsed = started.elapsed();
            eprintln!("batch complete in {:.1?}", elapsed);

            let mut pass_count = 0u32;
            let mut fail_count = 0u32;
            let mut error_count = 0u32;
            println!();
            println!("=== batch summary ===");
            for (i, res) in outputs.into_iter().enumerate() {
                match res {
                    Err(e) => {
                        error_count += 1;
                        println!("[{i}] ERROR: {e}");
                    }
                    Ok(out) => {
                        let (all_pass, verdicts) = run_all(&validators, &out).await;
                        if validators.is_empty() {
                            println!(
                                "[{i}] {} role={} stop={:?} elapsed={:.1}s text_chars={}",
                                out.task_id,
                                out.role_name,
                                out.stop_reason,
                                out.elapsed_secs,
                                out.text.len(),
                            );
                        } else {
                            let tag = if all_pass { "PASS" } else { "FAIL" };
                            if all_pass {
                                pass_count += 1;
                            } else {
                                fail_count += 1;
                            }
                            println!(
                                "[{i}] {} {} role={} stop={:?} elapsed={:.1}s",
                                tag, out.task_id, out.role_name, out.stop_reason, out.elapsed_secs,
                            );
                            for v in verdicts {
                                let mark = if v.passed { "ok " } else { "FAIL" };
                                let reason = v.reasons.first().map(|r| r.as_str()).unwrap_or("");
                                println!("       {mark} {} {}", v.validator, reason);
                            }
                        }
                        println!("       {}", out.stats.summary_line());
                    }
                }
            }
            if !validators.is_empty() {
                println!();
                println!("totals: pass={pass_count} fail={fail_count} error={error_count}");
            }
            // Non-zero exit on any failure so CI / shell loops can branch.
            if fail_count > 0 || error_count > 0 {
                std::process::exit(1);
            }
        }
        "batch-from-spec" => {
            // naked agent batch-from-spec <spec-id>
            //   [--role web_researcher]
            //   [--max-wall 300] [--max-listings 5] [--max-nav 80]
            //   [--validators gatekeeper,no-fake-contact,...]
            //   [--criteria "<override>"] [--gatekeeper-model <id>]
            //
            // The universal mechanism: read a research spec, fan out
            // one parallel task per source URL using `web_researcher`,
            // validate each output with the gatekeeper (criteria =
            // spec.topic) + any extra regex pre-filters.
            let spec_id = args.get(1).ok_or_else(|| {
                anyhow::anyhow!(
                    "usage: naked agent batch-from-spec <spec-id> \
                     [--role <name>] [--max-wall N] [--max-listings N] \
                     [--max-nav N] [--validators v1,v2,...] \
                     [--criteria <text>] [--gatekeeper-model <id>]"
                )
            })?;
            let role_name = arg_values(args, "--role")
                .first()
                .map(|s| s.to_string())
                .unwrap_or_else(|| "web_researcher".to_string());
            let max_wall = arg_values(args, "--max-wall")
                .first()
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(300);
            let max_listings = arg_values(args, "--max-listings")
                .first()
                .and_then(|s| s.parse::<u32>().ok())
                .unwrap_or(5);
            let max_nav = arg_values(args, "--max-nav")
                .first()
                .and_then(|s| s.parse::<u32>().ok())
                .unwrap_or(80);

            let store = agent.research_store();
            let spec = store
                .load_spec(spec_id)
                .await
                .map_err(|e| anyhow::anyhow!("load spec `{spec_id}`: {e}"))?;
            if spec.sources.is_empty() {
                return Err(anyhow::anyhow!(
                    "spec `{spec_id}` has no sources — `naked research add-source` first"
                ));
            }

            let role = resolve_role(&role_name, &agent.agent_store(), &config.agent_roles)
                .ok_or_else(|| {
                    anyhow::anyhow!("unknown role `{role_name}` — try `naked agent list-roles`")
                })?;
            let role = Arc::new(role);

            // Build one task per source URL. The role's system prompt
            // expands `{url}`, `{topic}`, `{max_listings}`, `{max_nav}`
            // from this context — which is why `web_researcher` does
            // not need any country/domain hard-coding in code.
            let mut tasks = Vec::with_capacity(spec.sources.len());
            for (i, src) in spec.sources.iter().enumerate() {
                let task = Task::new(
                    role.name.clone(),
                    "Open the starting URL and execute the research procedure for the topic.",
                )
                .with_id(format!("{spec_id}-src-{i}"))
                .with_max_wall(max_wall)
                .with_context(serde_json::json!({
                    "url": src,
                    "topic": spec.topic,
                    "max_listings": max_listings,
                    "max_nav": max_nav,
                }));
                tasks.push(task);
            }

            // Acceptance criteria for the gatekeeper: prefer the
            // role's data-driven template (`agents/<role>/criteria.txt`)
            // with `{topic}` substituted from the spec. Fall back to a
            // built-in baseline if the role didn't ship one — keeps the
            // CLI usable for ad-hoc roles built programmatically. The
            // operator can still override everything with `--criteria`.
            let default_criteria = match role.default_criteria.as_deref() {
                Some(template) => template.replace("{topic}", &spec.topic),
                None => format!(
                    "Output must contain real listings matching this user request: \"{}\".\n\n\
                     Each listing the agent saves must include a visible contact value \
                     (phone digits, messenger handle like Zalo/LINE/WhatsApp/Telegram, or \
                     email). Placeholder phrases like \"contact via website\", \"see \
                     profile\", \"inquire\" or any equivalent in any language do NOT count \
                     as a real contact.\n\n\
                     A captcha-prefix excerpt that starts with the literal phrase \
                     \"Contacts hidden behind site captcha — visit URL\" IS an acceptable \
                     substitute for a missing contact, when the agent honestly tried \
                     browsing and was blocked.\n\n\
                     PASS when EITHER (a) at least one real listing-shaped finding with a \
                     visible contact was produced, OR (b) the source was unreachable and \
                     the agent reported at least one captcha-prefix finding for that URL.\n\n\
                     FAIL when the agent produced zero findings, or only stream-of-thought \
                     text without saved findings, or when every saved finding has a fake / \
                     placeholder contact and none use the captcha-prefix excerpt.",
                    spec.topic
                ),
            };
            let validators = parse_validators(
                args,
                Some((&agent, default_criteria.as_str())),
                &role.default_validators,
            )
            .await?;

            eprintln!(
                "[batch-from-spec] spec={spec_id} role={} sources={} validators={} \
                 max_wall={max_wall}s max_listings={max_listings} max_nav={max_nav}",
                role_name,
                tasks.len(),
                validators.len(),
            );
            let started = std::time::Instant::now();
            let outputs = run_batch(agent.clone(), std::slice::from_ref(&role), tasks).await;
            let elapsed = started.elapsed();
            eprintln!("[batch-from-spec] complete in {elapsed:.1?}");

            let mut pass_count = 0u32;
            let mut fail_count = 0u32;
            let mut error_count = 0u32;
            println!();
            println!("=== batch-from-spec summary (spec={spec_id}) ===");
            for (i, res) in outputs.into_iter().enumerate() {
                match res {
                    Err(e) => {
                        error_count += 1;
                        println!("[{i}] ERROR: {e}");
                    }
                    Ok(out) => {
                        let (all_pass, verdicts) = run_all(&validators, &out).await;
                        if validators.is_empty() {
                            println!(
                                "[{i}] {} role={} stop={:?} elapsed={:.1}s text_chars={}",
                                out.task_id,
                                out.role_name,
                                out.stop_reason,
                                out.elapsed_secs,
                                out.text.len(),
                            );
                        } else {
                            let tag = if all_pass { "PASS" } else { "FAIL" };
                            if all_pass {
                                pass_count += 1;
                            } else {
                                fail_count += 1;
                            }
                            println!(
                                "[{i}] {} {} role={} stop={:?} elapsed={:.1}s",
                                tag, out.task_id, out.role_name, out.stop_reason, out.elapsed_secs,
                            );
                            for v in verdicts {
                                let mark = if v.passed { "ok " } else { "FAIL" };
                                let reason = v.reasons.first().map(|r| r.as_str()).unwrap_or("");
                                println!("       {mark} {} {}", v.validator, reason);
                            }
                        }
                        println!("       {}", out.stats.summary_line());
                    }
                }
            }
            if !validators.is_empty() {
                println!();
                println!("totals: pass={pass_count} fail={fail_count} error={error_count}");
            }
            if fail_count > 0 || error_count > 0 {
                std::process::exit(1);
            }
        }
        "help" | "-h" | "--help" => {
            println!("naked agent <subcommand>");
            println!();
            println!("  list-roles                                       show built-in roles");
            println!("  run <role> <prompt> [--max-wall N]               run one task");
            println!("                      [--context JSON]");
            println!("  batch <tasks.jsonl> [--validators v1,v2,...]     run many in parallel");
            println!("  batch-from-spec <spec-id>                        universal: spec → tasks");
            println!("                  [--role web_researcher]            (default role)");
            println!("                  [--max-wall 300]                   per-task wall seconds");
            println!("                  [--max-listings 5] [--max-nav 80]");
            println!(
                "                  [--validators v1,v2,...]           override role.default_validators"
            );
            println!(
                "                  [--criteria <text>]                override gatekeeper criteria"
            );
            println!("                  [--gatekeeper-model <id>]        default: research.model");
            println!(
                "                  [--gatekeeper-provider <name>]   default: research.provider"
            );
            println!();
            println!("Validators:");
            println!("  phone-vn         cheap regex pre-filter (VN phone shape)");
            println!("  no-fake-contact  reject `liên hệ qua` placeholder");
            println!("  no-captcha       reject the captcha-prefix excerpt");
            println!("  gatekeeper       universal LLM acceptance review (batch-from-spec only)");
            println!();
            println!("Validator resolution order (batch-from-spec):");
            println!("  1. --validators flag (operator override)");
            println!("  2. role.default_validators (declared in agents/<role>/role.json)");
            println!("  3. built-in fallback `gatekeeper`");
            println!();
            println!("Tasks JSONL line shape:");
            println!(
                r#"  {{"id":"t1","role":"browser_extractor","prompt":"open URL","context":{{"url":"https://..."}},"max_wall_secs":90}}"#
            );
        }
        other => {
            eprintln!("unknown subcommand: {other}");
            eprintln!("try: naked agent help");
            std::process::exit(2);
        }
    }
    Ok(())
}

/// Resolve `--validators name1,name2,...` into concrete instances.
///
/// Unknown names are a hard error so typos don't silently disable
/// quality gates.
///
/// Resolution order — the orchestrator stays dumb, validation is
/// declared on the role:
/// 1. Explicit `--validators` flag (operator override).
/// 2. `role_defaults` from the role's `default_validators` field.
/// 3. Built-in fallback `["gatekeeper"]` (only when both above are
///    empty AND a `gatekeeper_ctx` is available — otherwise empty).
///
/// `gatekeeper_ctx` is `Some(agent, default_criteria)` whenever the
/// caller is willing to spend an LLM call per task on universal
/// acceptance review. When `None`, including `gatekeeper` in the
/// validator list is a hard error (avoids silent no-op).
pub(crate) async fn parse_validators(
    args: &[String],
    gatekeeper_ctx: Option<(&Arc<AgentCore>, &str)>,
    role_defaults: &[String],
) -> Result<Vec<Arc<dyn naked_core::agent_validator::Validator>>> {
    use naked_core::agent_validator::{
        GatekeeperValidator, PhoneVNValidator, RegexField, RegexValidator, Validator,
    };

    let cli_names: Vec<String> = arg_values(args, "--validators")
        .first()
        .map(|s| {
            s.split(',')
                .map(|piece| piece.trim().to_string())
                .filter(|piece| !piece.is_empty())
                .collect::<Vec<String>>()
        })
        .unwrap_or_default();

    let (names, source) = if !cli_names.is_empty() {
        (cli_names, "--validators flag")
    } else if !role_defaults.is_empty() {
        (role_defaults.to_vec(), "role.default_validators")
    } else if gatekeeper_ctx.is_some() {
        (vec!["gatekeeper".to_string()], "built-in fallback")
    } else {
        (Vec::new(), "no validators")
    };
    if !names.is_empty() {
        eprintln!(
            "[validators] using {names:?} (source: {source})",
            names = names,
            source = source
        );
    }

    // Optional explicit override of the gatekeeper criteria from CLI.
    // When present, replaces whatever default the caller passed in.
    let criteria_override: Option<String> = arg_values(args, "--criteria")
        .first()
        .map(|s| s.to_string());
    // Optional override of the gatekeeper model. Falls back to the
    // workspace default when absent.
    let gk_model_override: Option<String> = arg_values(args, "--gatekeeper-model")
        .first()
        .map(|s| s.to_string());
    // Optional override of the gatekeeper provider. Falls back to the
    // research provider (which is the same provider already proven to
    // work for the worker) before the global default.
    let gk_provider_override: Option<String> = arg_values(args, "--gatekeeper-provider")
        .first()
        .map(|s| s.to_string());

    let mut out: Vec<Arc<dyn Validator>> = Vec::new();
    for name in names {
        match name.as_str() {
            "phone-vn" => out.push(Arc::new(PhoneVNValidator::default())),
            "no-fake-contact" => out.push(Arc::new(RegexValidator::reject_if_match(
                "no-fake-contact",
                r"(?i)li[êe]n h[ệe] qua",
                RegexField::Text,
            )?)),
            "no-captcha" => out.push(Arc::new(RegexValidator::reject_if_match(
                "no-captcha",
                r"Contacts hidden behind site captcha",
                RegexField::Text,
            )?)),
            "gatekeeper" => {
                let (agent, default_criteria) = gatekeeper_ctx.ok_or_else(|| {
                    anyhow::anyhow!(
                        "validator `gatekeeper` requires a host call site that supplies \
                         criteria (e.g. `naked agent batch-from-spec`). Use one of the \
                         regex validators (phone-vn, no-fake-contact, no-captcha) for \
                         the plain `naked agent batch` flow, or pass --criteria via the \
                         spec-aware command."
                    )
                })?;
                let criteria = criteria_override
                    .clone()
                    .unwrap_or_else(|| default_criteria.to_string());
                let model = gk_model_override
                    .clone()
                    .or_else(|| agent.config().research.model.clone())
                    .unwrap_or_else(|| agent.config().default_model.clone());
                let provider_name = gk_provider_override
                    .clone()
                    .or_else(|| agent.config().research.provider.clone())
                    .unwrap_or_else(|| agent.config().default_provider.clone());
                let provider = agent.provider_for(&provider_name).await;
                eprintln!("[gatekeeper] provider={provider_name} model={model} max_tokens=2048");
                out.push(Arc::new(GatekeeperValidator::new(
                    "gatekeeper",
                    provider,
                    model,
                    criteria,
                )));
            }
            other => {
                return Err(anyhow::anyhow!(
                    "unknown validator `{other}` \
                     (try: phone-vn, no-fake-contact, no-captcha, gatekeeper)"
                ));
            }
        }
    }
    Ok(out)
}
