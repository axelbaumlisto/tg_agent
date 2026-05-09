//! CLI research subcommands.

use std::sync::Arc;

use anyhow::Result;
use naked_core::AgentCore;
use naked_core::config::Config;

use crate::commands::arg_values;

pub(crate) async fn research_cmd(args: &[String]) -> Result<()> {
    let sub = args.first().map(|s| s.as_str()).unwrap_or("help");
    let config = Config::load()?;
    if !config.research.enabled {
        eprintln!("Research subsystem is disabled in config.");
        std::process::exit(2);
    }
    let provider = naked_core::build_provider_from_config(&config)?;
    let agent = Arc::new(AgentCore::new(config.clone(), provider));
    agent.init_self_ref();

    match sub {
        "ls" => {
            let list = agent.list_research().await?;
            if list.is_empty() {
                eprintln!("No research defined.");
                return Ok(());
            }
            for spec in &list {
                println!(
                    "{}\t{}\tpaused={}\tcreated={}",
                    spec.id,
                    spec.topic,
                    spec.paused,
                    spec.created_at.format("%Y-%m-%d %H:%M UTC")
                );
            }
        }
        "show" | "delta" => {
            let fresh_only =
                sub == "delta" || args.iter().any(|a| a == "--fresh" || a == "--delta");
            let id = args
                .iter()
                .skip(1)
                .find(|a| !a.starts_with("--"))
                .ok_or_else(|| {
                    anyhow::anyhow!("usage: naked research show <id> [--fresh|--delta]")
                })?;
            let spec = agent.load_research(id).await?;
            println!("id:      {}", spec.id);
            println!("topic:   {}", spec.topic);
            println!("sources: {}", spec.sources.join(", "));
            println!("paused:  {}", spec.paused);
            println!(
                "provider/model: {}/{}",
                spec.provider.as_deref().unwrap_or(&config.default_provider),
                spec.model.as_deref().unwrap_or(&config.default_model)
            );
            let store = agent.research_store();
            let total = store.count_findings(id).await.unwrap_or(0);
            let runs = store.list_runs(id, Some(3)).await.unwrap_or_default();

            let mut findings = if fresh_only {
                store.list_findings(id, None).await.unwrap_or_default()
            } else {
                store.list_findings(id, Some(10)).await.unwrap_or_default()
            };
            if fresh_only && let Some(last_run) = runs.last() {
                let run_id = &last_run.run_id;
                findings.retain(|f| f.run_id == *run_id);
            }

            if fresh_only {
                println!("findings (total {total}, fresh {}):", findings.len());
            } else {
                println!("findings (total {total}):");
            }
            for f in findings.iter().rev() {
                let title = f.title.as_deref().unwrap_or("(untitled)");
                let date = f.listing_date.as_deref().unwrap_or("");
                let price = f.price.as_deref().unwrap_or("");
                if date.is_empty() && price.is_empty() {
                    println!("  - {title} {}", f.url);
                } else {
                    println!("  - {title} [{date}] {price} {}", f.url);
                }
            }
            println!("recent runs:");
            for r in runs.iter().rev() {
                println!(
                    "  - {} +{} (total {}) {}/{} — {}",
                    r.started_at.format("%Y-%m-%d %H:%M UTC"),
                    r.new_findings,
                    r.total_findings_after,
                    r.provider,
                    r.model,
                    r.stop_reason,
                );
            }
        }
        "new" => {
            let topic = args
                .iter()
                .skip(1)
                .find(|a| !a.starts_with("--"))
                .ok_or_else(|| {
                    anyhow::anyhow!("usage: naked research new <topic> [--source URL]...")
                })?
                .clone();
            let sources: Vec<String> = arg_values(args, "--source")
                .into_iter()
                .map(|s| s.to_string())
                .collect();
            let spec = agent
                .create_research(&topic, sources, None, None, None)
                .await?;
            println!("{}", spec.id);
        }
        "run" => {
            let id = args
                .get(1)
                .ok_or_else(|| anyhow::anyhow!("usage: naked research run <id> [--verify [N]]"))?;
            let verify = args.iter().any(|a| a == "--verify");
            let verify_rounds: u32 = args
                .iter()
                .position(|a| a == "--verify")
                .and_then(|i| args.get(i + 1))
                .and_then(|v| v.parse().ok())
                .unwrap_or(2);
            agent.init_mcp().await;

            if verify {
                let vr = agent
                    .clone()
                    .run_research_verified(id, verify_rounds)
                    .await?;
                eprintln!(
                    "{} gatekeeper: {} rounds, removed={}, replaced={}, final={}",
                    vr.last_run.spec_id,
                    vr.verification_rounds,
                    vr.dead_removed,
                    vr.replacements_found,
                    vr.final_findings,
                );
                if !vr.remaining_issues.is_empty() {
                    eprintln!("  remaining issues: {}", vr.remaining_issues.len());
                }
            } else {
                let report = agent.clone().run_research(id).await?;
                eprintln!(
                    "{} +{} findings (total {}), stop={} in {:.1?} via {}/{}",
                    report.spec_id,
                    report.new_findings,
                    report.total_findings_after,
                    report.stop_reason.as_str(),
                    report.elapsed,
                    report.provider,
                    report.model,
                );
                if matches!(
                    report.stop_reason,
                    naked_core::research::StopReason::Error
                        | naked_core::research::StopReason::StreamClosed,
                ) {
                    std::process::exit(1);
                }
            }
        }
        "run-many" => {
            // naked research run-many <id1> <id2> ... [-j N] [--verify [N]]
            //
            // Run several specs concurrently using the existing
            // `agent.run_research(id)` per spec. Concurrency is bounded
            // BOTH by the local `-j` cap (default = number of ids) AND
            // by the process-wide `research_run_permits` semaphore in
            // `AgentCore` — whichever is smaller wins. So calling
            // `run-many` over 100 specs with `-j 100` is still safe:
            // the global permits will throttle to whatever the
            // operator set in `naked.json`.
            //
            // Exit code: 1 if ANY spec finishes with a hard stop
            // (`Error`/`StreamClosed`), 0 otherwise.
            let value_flags: &[&str] = &["-j", "--jobs", "--verify"];
            let bool_flags: &[&str] = &["--verify"];
            let mut ids: Vec<String> = Vec::new();
            let mut i: usize = 1;
            while i < args.len() {
                let a = &args[i];
                if value_flags.contains(&a.as_str()) {
                    // `--verify` is dual-shape (bool OR `--verify N`);
                    // peek at the next arg to decide whether to consume.
                    if a == "--verify" {
                        if let Some(next) = args.get(i + 1)
                            && next.parse::<u32>().is_ok()
                        {
                            i += 2;
                            continue;
                        }
                        i += 1;
                        continue;
                    }
                    i += 2;
                    continue;
                }
                if bool_flags.contains(&a.as_str()) || a.starts_with("--") || a.starts_with('-') {
                    i += 1;
                    continue;
                }
                ids.push(a.clone());
                i += 1;
            }
            if ids.is_empty() {
                return Err(anyhow::anyhow!(
                    "usage: naked research run-many <id1> <id2> ... [-j N] [--verify [N]]"
                ));
            }
            let jobs_short = arg_values(args, "-j");
            let jobs_long = arg_values(args, "--jobs");
            let jobs: usize = jobs_short
                .first()
                .or_else(|| jobs_long.first())
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(ids.len())
                .max(1);
            let verify = args.iter().any(|a| a == "--verify");
            let verify_rounds: u32 = args
                .iter()
                .position(|a| a == "--verify")
                .and_then(|i| args.get(i + 1))
                .and_then(|v| v.parse().ok())
                .unwrap_or(2);

            agent.init_mcp().await;

            eprintln!(
                "[run-many] {} spec(s), local jobs={jobs} (global cap from naked.json applies), \
                 verify={verify}",
                ids.len(),
            );
            let started = std::time::Instant::now();

            // Local concurrency gate. The global research_run_permits
            // (held inside AgentCore::run_research) is the second
            // gate — both must clear for a spec to actually run.
            let local_gate = std::sync::Arc::new(tokio::sync::Semaphore::new(jobs));
            let mut join_set = tokio::task::JoinSet::new();

            for id in &ids {
                let id = id.clone();
                let agent_c = agent.clone();
                let gate = local_gate.clone();
                join_set.spawn(async move {
                    let _permit = gate.acquire().await.expect("local gate alive");
                    let started_one = std::time::Instant::now();
                    let outcome: Result<(String, String), (String, String)> = if verify {
                        match agent_c.run_research_verified(&id, verify_rounds).await {
                            Ok(vr) => Ok((
                                id.clone(),
                                format!(
                                    "verify ok in {:.1?}: {} rounds, removed={}, replaced={}, final={}, remaining={}",
                                    started_one.elapsed(),
                                    vr.verification_rounds,
                                    vr.dead_removed,
                                    vr.replacements_found,
                                    vr.final_findings,
                                    vr.remaining_issues.len(),
                                ),
                            )),
                            Err(e) => Err((id.clone(), e.to_string())),
                        }
                    } else {
                        match agent_c.run_research(&id).await {
                            Ok(report) => {
                                let stop = report.stop_reason.as_str().to_string();
                                let line = format!(
                                    "+{} findings (total {}), stop={} in {:.1?} via {}/{}",
                                    report.new_findings,
                                    report.total_findings_after,
                                    stop,
                                    report.elapsed,
                                    report.provider,
                                    report.model,
                                );
                                let hard_stop = matches!(
                                    report.stop_reason,
                                    naked_core::research::StopReason::Error
                                        | naked_core::research::StopReason::StreamClosed,
                                );
                                if hard_stop {
                                    Err((id.clone(), format!("HARD-STOP: {line}")))
                                } else {
                                    Ok((id.clone(), line))
                                }
                            }
                            Err(e) => Err((id.clone(), e.to_string())),
                        }
                    };
                    outcome
                });
            }

            let mut ok_count = 0u32;
            let mut err_count = 0u32;
            while let Some(joined) = join_set.join_next().await {
                match joined {
                    Ok(Ok((id, line))) => {
                        ok_count += 1;
                        println!("[run-many] OK   {id}: {line}");
                    }
                    Ok(Err((id, msg))) => {
                        err_count += 1;
                        println!("[run-many] FAIL {id}: {msg}");
                    }
                    Err(join_err) => {
                        err_count += 1;
                        println!("[run-many] PANIC: {join_err}");
                    }
                }
            }
            let elapsed = started.elapsed();
            println!();
            println!(
                "[run-many] done in {elapsed:.1?}: ok={ok_count} fail={err_count} \
                 of {} (avg {:.1?}/spec)",
                ids.len(),
                elapsed / (ids.len().max(1) as u32),
            );
            if err_count > 0 {
                std::process::exit(1);
            }
        }
        "probe" => {
            // Fast iteration harness: pin a tiny ephemeral spec to one or more
            // URLs, run a single pass with a tight wall-clock cap, and print a
            // structured summary of tool usage / captcha hits / extracted
            // contacts. Used to iterate the research prompt without burning a
            // full 10-minute verify cycle each time.
            //
            // Usage:
            //   naked research probe <url> [<url2> ...] [--topic STR]
            //                              [--max-wall N] [--keep] [--verbose]
            //
            // Cleans up the spec on exit unless `--keep` is passed. Exit code
            // mirrors `run`: 1 on `Error`/`StreamClosed`, 0 otherwise — so a
            // shell loop can iterate until the agent stops failing.
            // Walk args linearly so that values consumed by `--flag VALUE`
            // pairs are *not* picked up as positional URLs. Single-dash flags
            // (`-v`) are also skipped. Without this guard `--max-wall 120 -v`
            // would inject `120` and `-v` into the URL list and the agent
            // would treat them as seed URLs.
            let value_flags: &[&str] = &["--topic", "--max-wall", "--url"];
            let bool_flags: &[&str] = &["--keep", "--verbose", "-v"];
            let mut positional: Vec<String> = Vec::new();
            let mut i: usize = 1; // skip subcommand name
            while i < args.len() {
                let a = &args[i];
                if value_flags.contains(&a.as_str()) {
                    i += 2;
                    continue;
                }
                if bool_flags.contains(&a.as_str()) || a.starts_with("--") || a.starts_with("-") {
                    i += 1;
                    continue;
                }
                positional.push(a.clone());
                i += 1;
            }
            let urls: Vec<String> = if positional.is_empty() {
                arg_values(args, "--url")
                    .into_iter()
                    .map(|s| s.to_string())
                    .collect()
            } else {
                positional
            };
            if urls.is_empty() {
                return Err(anyhow::anyhow!(
                    "usage: naked research probe <url> [<url2> ...] [--topic STR] \
                     [--max-wall N] [--keep] [--verbose]"
                ));
            }
            let topic = arg_values(args, "--topic")
                .first()
                .map(|s| s.to_string())
                .unwrap_or_else(|| {
                    "Probe: open the URL(s) below, extract listing details \
                     (title, price, area, location), and the seller's real \
                     contact phone number. Save exactly one finding per URL."
                        .to_string()
                });
            // 90s is enough for ~10-12 tool calls on a normal page; if the
            // agent doesn't reach `Idle` by then the URL is almost certainly
            // hard-gated (real captcha, login wall) and longer waits do not
            // produce a finding. Bump explicitly with `--max-wall N` only when
            // exercising slow networks or paginated sites.
            let max_wall: u64 = arg_values(args, "--max-wall")
                .first()
                .and_then(|s| s.parse().ok())
                .unwrap_or(90);
            let keep = args.iter().any(|a| a == "--keep");
            let verbose = args.iter().any(|a| a == "--verbose" || a == "-v");

            agent.init_mcp().await;

            let spec = agent
                .create_research(&topic, urls.clone(), None, None, None)
                .await?;
            // Tighten the wall-clock cap so iteration loops are fast. Iteration
            // budget (`max_iterations`) is left at config default — the model
            // typically stops itself well below the ceiling for one URL.
            let _ = agent
                .update_research(
                    &spec.id,
                    naked_core::ResearchPatch {
                        max_wall_seconds: naked_core::PatchField::Set(max_wall),
                        ..Default::default()
                    },
                )
                .await;

            eprintln!("probe spec: {} (max_wall={max_wall}s)", spec.id);
            eprintln!("urls:");
            for u in &urls {
                eprintln!("  - {u}");
            }

            let started = std::time::Instant::now();
            let report = agent.clone().run_research(&spec.id).await?;
            let elapsed = started.elapsed();

            // Inspect the saved findings for the single signal we care about
            // most: did the agent extract a real digit-string phone number?
            let store = agent.research_store();
            let findings = store
                .list_findings(&spec.id, None)
                .await
                .unwrap_or_default();

            // Smell-test phone counter: scan for a leading `0` followed by
            // 8-10 more digits, allowing one space/dot/dash between groups.
            // VN mobile numbers are all 10-11 digits starting with 0; landlines
            // similar. False positives (year ranges, prices) are tolerable —
            // this metric only needs to be directional across iterations.
            fn count_vn_phones(text: &str) -> u32 {
                let bytes = text.as_bytes();
                let is_sep = |b: u8| matches!(b, b' ' | b'.' | b'-');
                let is_word_boundary_left = |i: usize| {
                    i == 0
                        || !bytes[i - 1].is_ascii_digit()
                            && !matches!(bytes[i - 1], b'+' | b'.' | b'-')
                };
                let mut hits = 0u32;
                let mut i = 0;
                while i < bytes.len() {
                    if bytes[i] == b'0' && is_word_boundary_left(i) {
                        let mut j = i;
                        let mut digits = 0u32;
                        while j < bytes.len() {
                            if bytes[j].is_ascii_digit() {
                                digits += 1;
                                j += 1;
                            } else if is_sep(bytes[j])
                                && j + 1 < bytes.len()
                                && bytes[j + 1].is_ascii_digit()
                            {
                                j += 1;
                            } else {
                                break;
                            }
                        }
                        if (9..=11).contains(&digits) {
                            hits += 1;
                            i = j;
                            continue;
                        }
                    }
                    i += 1;
                }
                hits
            }

            let mut total_phones = 0u32;
            let mut findings_with_phone = 0u32;
            let mut captcha_excerpts = 0u32;
            let mut fake_lien_he_qua = 0u32;
            for f in &findings {
                let blob = format!(
                    "{}\n{}",
                    f.excerpt.as_deref().unwrap_or(""),
                    f.title.as_deref().unwrap_or("")
                );
                let n = count_vn_phones(&blob);
                if n > 0 {
                    findings_with_phone += 1;
                    total_phones += n;
                }
                if blob.contains("Contacts hidden behind site captcha") {
                    captcha_excerpts += 1;
                }
                if blob.to_lowercase().contains("liên hệ qua") {
                    fake_lien_he_qua += 1;
                }
            }

            eprintln!();
            eprintln!("=== probe summary ===");
            eprintln!("spec_id:           {}", spec.id);
            eprintln!("elapsed:           {:.1?}", elapsed);
            eprintln!("stop_reason:       {}", report.stop_reason.as_str());
            eprintln!("findings:          {}", findings.len());
            eprintln!("  with real phone: {findings_with_phone} ({total_phones} phone matches)");
            eprintln!("  captcha-prefix:  {captcha_excerpts}  (good fallback)");
            eprintln!("  fake 'Liên hệ qua': {fake_lien_he_qua}  (must be 0)");
            if verbose {
                for f in &findings {
                    let title = f.title.as_deref().unwrap_or("(no title)");
                    let ex: String = f
                        .excerpt
                        .as_deref()
                        .unwrap_or("")
                        .chars()
                        .take(280)
                        .collect();
                    eprintln!("---");
                    eprintln!("  url:     {}", f.url);
                    eprintln!("  title:   {title}");
                    eprintln!("  excerpt: {ex}");
                }
            }

            if !keep {
                let _ = agent.delete_research(&spec.id).await;
                eprintln!("(spec deleted; pass --keep to retain)");
            } else {
                eprintln!("(spec kept: naked research show {})", spec.id);
            }

            if matches!(
                report.stop_reason,
                naked_core::research::StopReason::Error
                    | naked_core::research::StopReason::StreamClosed,
            ) {
                std::process::exit(1);
            }
        }
        "rm" => {
            let id = args
                .get(1)
                .ok_or_else(|| anyhow::anyhow!("usage: naked research rm <id>"))?;
            agent.delete_research(id).await?;
            eprintln!("deleted {id}");
        }
        "pause" => {
            let id = args
                .get(1)
                .ok_or_else(|| anyhow::anyhow!("usage: naked research pause <id>"))?;
            agent.set_research_paused(id, true).await?;
            eprintln!("paused {id}");
        }
        "resume" => {
            let id = args
                .get(1)
                .ok_or_else(|| anyhow::anyhow!("usage: naked research resume <id>"))?;
            agent.set_research_paused(id, false).await?;
            eprintln!("resumed {id}");
        }
        "export" => {
            let id = args
                .get(1)
                .ok_or_else(|| anyhow::anyhow!("usage: naked research export <id> <path>"))?;
            let dest = args
                .get(2)
                .ok_or_else(|| anyhow::anyhow!("usage: naked research export <id> <path>"))?;
            let store = agent.research_store();
            match store.read_report(id).await? {
                Some(md) => {
                    std::fs::write(dest, md.as_bytes())?;
                    eprintln!("wrote report to {dest}");
                }
                None => {
                    eprintln!("no report for {id} yet — run `naked research run {id}` first");
                    std::process::exit(3);
                }
            }
        }
        "schedule" => {
            let id = args.get(1).ok_or_else(|| {
                anyhow::anyhow!("usage: naked research schedule <id> [on|off|status]")
            })?;
            let action = args.get(2).map(|s| s.as_str()).unwrap_or("on");
            let timer = format!("naked-research@{id}.timer");
            match action {
                "off" | "disable" => {
                    let out = std::process::Command::new("systemctl")
                        .args(["--user", "disable", "--now", &timer])
                        .output()?;
                    if out.status.success() {
                        eprintln!("timer disabled for {id}");
                    } else {
                        eprintln!("{}", String::from_utf8_lossy(&out.stderr));
                        std::process::exit(1);
                    }
                }
                "on" | "enable" => {
                    let out = std::process::Command::new("systemctl")
                        .args(["--user", "enable", "--now", &timer])
                        .output()?;
                    if out.status.success() {
                        eprintln!("timer enabled for {id}");
                        let _ = std::process::Command::new("systemctl")
                            .args(["--user", "list-timers", &timer, "--no-pager"])
                            .status();
                    } else {
                        eprintln!("{}", String::from_utf8_lossy(&out.stderr));
                        std::process::exit(1);
                    }
                }
                "status" => {
                    let out = std::process::Command::new("systemctl")
                        .args(["--user", "is-enabled", &timer])
                        .output()?;
                    let state = String::from_utf8_lossy(&out.stdout).trim().to_string();
                    println!("timer {id}: {state}");
                    let _ = std::process::Command::new("systemctl")
                        .args(["--user", "list-timers", &timer, "--no-pager"])
                        .status();
                }
                other => {
                    eprintln!("unknown schedule action: {other}");
                    eprintln!("usage: naked research schedule <id> [on|off|status]");
                    std::process::exit(2);
                }
            }
        }
        "help" | "-h" | "--help" => {
            println!("naked research <subcommand>");
            println!("  new <topic> [--source URL]...   create a research");
            println!("  ls                              list all research");
            println!("  show <id> [--fresh|--delta]     show spec + findings");
            println!("  run <id>                        run one pass (used by systemd)");
            println!("  run-many <id1> <id2> ...        run several specs concurrently");
            println!(
                "           [-j N] [--verify [N]]    local job cap; respects naked.json permits"
            );
            println!(
                "  probe <url> [...]               fast iteration: ephemeral spec, single pass,"
            );
            println!(
                "                                  structured stats; --topic STR --max-wall N --keep -v"
            );
            println!("  schedule <id> [on|off|status]   manage systemd timer");
            println!("  pause <id> | resume <id>        toggle scheduled runs");
            println!("  export <id> <path>              write report.md to a file");
            println!("  rm <id>                         delete all data for id");
        }
        other => {
            eprintln!("unknown subcommand: {other}");
            eprintln!("try: naked research help");
            std::process::exit(2);
        }
    }
    Ok(())
}
